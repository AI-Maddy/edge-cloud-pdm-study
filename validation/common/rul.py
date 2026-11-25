"""Shared C-MAPSS RUL preprocessing and models (used by C2 and later concepts).

Standard protocol (Zheng et al. 2017; Li et al. 2018): 14 informative sensors, piecewise-linear RUL capped
at 125 cycles, sliding windows of 30 cycles, test score on the last window of each test unit.
Context-aware normalisation: z-score per operating-condition cell (cells from C1's K-Means on settings).
"""
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "models"))
from common import cmapss  # noqa: E402

SENS = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]   # 0-based sensor idx: s2 s3 s4 s7 s8 s9 s11 s12 s13 s14 s15 s17 s20 s21
CAP, WIN = 125, 30
PUBLISHED = {  # test RMSE
    "LSTM (Zheng 2017)": {"FD001": 16.14, "FD002": 24.49, "FD003": 16.18, "FD004": 28.17},
    "DCNN (Li 2018)": {"FD001": 12.61, "FD002": 22.36, "FD003": 12.64, "FD004": 23.31},
}


class Normaliser:
    def __init__(self, context=True, k=None):
        self.context, self.k = context, k

    def fit(self, settings, sensors):
        k = self.k or (6 if np.ptp(settings[:, 0]) > 1 else 1)
        self.km = KMeans(k, n_init=10, random_state=0).fit(settings / [42, 0.84, 100]) if (self.context and k > 1) else None
        lab = self.cells(settings)
        self.mu = np.stack([sensors[lab == c].mean(0) for c in range(lab.max() + 1)])
        self.sd = np.stack([sensors[lab == c].std(0) + 1e-6 for c in range(lab.max() + 1)])
        return self

    def cells(self, settings):
        return self.km.predict(settings / [42, 0.84, 100]) if self.km is not None else np.zeros(len(settings), int)

    def transform(self, settings, sensors):
        lab = self.cells(settings)
        return (sensors - self.mu[lab]) / self.sd[lab], lab


def windows(arr, z, rul=None, last_only=False):
    """arr: raw rows (unit, cycle, ...); z: normalised sensors. Returns X (N,WIN,F), y, unit ids."""
    X, y, u = [], [], []
    for unit in np.unique(arr[:, 0]):
        m = arr[:, 0] == unit
        zu = z[m]
        if len(zu) < WIN:
            zu = np.vstack([np.repeat(zu[:1], WIN - len(zu), 0), zu])
        r = None if rul is None else np.minimum(rul[m], CAP)
        if r is not None and len(r) < WIN:
            r = np.r_[np.repeat(r[:1], WIN - len(r)), r]
        starts = [len(zu) - WIN] if last_only else range(0, len(zu) - WIN + 1)
        for s in starts:
            X.append(zu[s:s + WIN]); u.append(unit)
            if r is not None:
                y.append(r[s + WIN - 1])
    return np.array(X, np.float32), (np.array(y, np.float32) if y else None), np.array(u)


def load_fd(fd, context=True):
    tr, te = cmapss.load(fd, "train"), cmapss.load(fd, "test")
    norm = Normaliser(context).fit(tr["arr"][:, 2:5], tr["arr"][:, 5:][:, SENS])
    ztr, _ = norm.transform(tr["arr"][:, 2:5], tr["arr"][:, 5:][:, SENS])
    zte, _ = norm.transform(te["arr"][:, 2:5], te["arr"][:, 5:][:, SENS])
    Xtr, ytr, utr = windows(tr["arr"], ztr, tr["rul"])
    Xte, _, ute = windows(te["arr"], zte, last_only=True)
    yte = np.minimum(te["rul_end"], CAP).astype(np.float32)
    return Xtr, ytr, utr, Xte, yte, ute


# ---------------------------------------------------------------- models
class CNN1D(nn.Module):
    """1-D temporal CNN in the style of Li et al. 2018 (smaller)."""
    def __init__(self, f, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(f, h, 9, padding=4), nn.ReLU(), nn.Conv1d(h, h, 9, padding=4), nn.ReLU(),
                                 nn.Conv1d(h, h, 9, padding=4), nn.ReLU(), nn.Flatten(), nn.Dropout(0.3),
                                 nn.Linear(h * WIN, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class RepoTransformerRUL(nn.Module):
    """The repository's TransformerEncoder (models/hfcf_models/neural.py) with a regression head."""
    def __init__(self, f, h=32):
        super().__init__()
        from hfcf_models.neural import TransformerEncoder
        self.enc = TransformerEncoder(input_dim=f, hidden_dim=h, embedding_dim=h, heads=4, layers=2)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(-1)


def make(name, f):
    return {"cnn1d": CNN1D, "transformer": RepoTransformerRUL}[name](f)


def train(model, X, y, units, seed, epochs=40, lr=1e-3, bs=256, val_frac=0.1, verbose=False):
    """Adam + MSE; early stopping on a held-out 10% of training units (never the test set)."""
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    uu = np.unique(units)
    val_u = g.choice(uu, max(1, int(len(uu) * val_frac)), replace=False)
    vm = np.isin(units, val_u)
    Xt, yt = torch.tensor(X[~vm]), torch.tensor(y[~vm])
    Xv, yv = torch.tensor(X[vm]), torch.tensor(y[vm])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = ((model(Xt[b]) - yt[b]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Xv) - yv) ** 2).mean().sqrt())
        if v < best - 1e-3:
            best, bad = v, 0
            state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(state)
    return model, best


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X)).numpy().clip(0, CAP)


def nasa_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))
