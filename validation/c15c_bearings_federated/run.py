"""C15c (review follow-up E6): federation, cold start and rule transfer on real measurements (XJTU-SY bearings).

The review noted that every federation / knowledge / cold-start result rests on C-MAPSS, a simulation. This
repeats the M3 and M5 questions on real run-to-failure vibration data, in the post-onset formulation of C15b
(same features, onset rule and target: remaining fraction of the post-onset period, 0..1).
Each bearing with a detected onset (10 of the C15 bearings) is one edge site. Leave-one-bearing-out: the held-out bearing is
a cold-start site with no failure history of its own; the other 9 are the federation.
  mean      predict the mean label of the other sites (what a site knows without a model)
  fedavg    a small MLP (features -> 32 -> 32 -> 1) trained by FedAvg over the 9 sites, 100 rounds x 1 epoch
  pooled    the same MLP trained on the 9 sites' pooled data (needs raw features in one place; upper bound)
Rules (M5): every site mines rules "context-normalised feature f > theta -> in the last 20% of post-onset life"
(theta at the site's 50/70/90th percentiles of f); confidence = precision at the source, support = firings,
quality = precision on the second half of the source's own post-onset data with theta set on the first half.
Transfer value = mean precision of the rule at the other bearings. Trust = 0.4 C + 0.2 min(1, S/200) + 0.2 Q + 0.2 R
with R = 0.5 (no reliability history), as in the design under test.
3 seeds.

Tests (fixed before running):
  T1 fedavg within 10% of pooled (mean leave-one-bearing-out RMSE)
  T2 fedavg <= 0.9 x mean predictor (shared knowledge helps a cold-start site on real data)
  T3 rule confidence predicts precision at other bearings: Spearman >= 0.5
  T4 the weighted trust ranks rules no better than confidence alone (replicates the C-MAPSS finding)
"""
import json, os, sys
import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "c15_bearings"))
import run as c15  # noqa: E402
import run_fpt as fpt_mod  # noqa: E402
torch.set_num_threads(1)
ROUNDS = 100


def mlp(f):
    return nn.Sequential(nn.Linear(f, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))


def epoch(m, X, y, g, lr=1e-3, bs=64):
    opt = torch.optim.Adam(m.parameters(), lr=lr); Xt, yt = torch.tensor(X), torch.tensor(y)
    p = torch.tensor(g.permutation(len(X)))
    for i in range(0, len(X), bs):
        b = p[i:i + bs]; opt.zero_grad(); ((m(Xt[b]).squeeze(-1) - yt[b]) ** 2).mean().backward(); opt.step()


def pred(m, X):
    with torch.no_grad():
        return m(torch.tensor(X)).squeeze(-1).numpy().clip(0, 1)


def data():
    d, feats, Xg, Xc = c15.prepare()
    bear, minute, life = d["bearing"].to_numpy(), d["minute"].to_numpy(), d["n_total"].to_numpy()
    sites = {}
    for b in dict.fromkeys(bear):
        m = bear == b
        rms = d.loc[m, "h_rms"].rolling(c15.SMOOTH, min_periods=1).mean().to_numpy()
        i = fpt_mod.onset(rms)
        if i is None:
            continue
        f0 = minute[m][i]; post = m & (minute >= f0)
        y = (life[post] - minute[post]) / max(1, life[post][0] - f0)
        sites[b] = (Xc[post].astype(np.float32), y.astype(np.float32))
    return sites, feats


def rules(sites, feats):
    rows = []
    for b, (X, y) in sites.items():
        pos = y <= 0.2; h = len(y) // 2
        for j, f in enumerate(feats):
            for q in (50, 70, 90):
                th = np.percentile(X[:, j], q); fire = X[:, j] > th
                if fire.sum() < 5:
                    continue
                C = float(pos[fire].mean()); S = int(fire.sum())
                th1 = np.percentile(X[:h, j], q); f2 = X[h:, j] > th1
                Q = float(pos[h:][f2].mean()) if f2.sum() else 0.0
                prec = []
                for b2, (X2, y2) in sites.items():
                    if b2 == b:
                        continue
                    f_ = X2[:, j] > th
                    if f_.sum():
                        prec.append(float((y2[f_] <= 0.2).mean()))
                if prec:
                    rows.append({"site": b, "feature": f, "q": q, "C": C, "S": S, "Q": Q, "transfer": float(np.mean(prec)),
                                 "trust": 0.4 * C + 0.2 * min(1, S / 200) + 0.2 * Q + 0.2 * 0.5})
    return rows


def main():
    sites, feats = data()
    B = list(sites); F = len(feats)
    res = {"sites": {b: int(len(v[1])) for b, v in sites.items()}, "loo": []}
    for seed in (0, 1, 2):
        for t in B:
            src = [b for b in B if b != t]; Xt, yt = sites[t]
            g = np.random.default_rng(seed); torch.manual_seed(seed)
            ytr = np.concatenate([sites[b][1] for b in src])
            r = {"seed": seed, "target": t, "mean": c15.rmse(np.full(len(yt), ytr.mean()), yt)}
            w0 = mlp(F).state_dict(); glob = mlp(F); glob.load_state_dict(w0)
            ns = np.array([len(sites[b][1]) for b in src], float)
            for _ in range(ROUNDS):
                ws = []
                for b in src:
                    m = mlp(F); m.load_state_dict(glob.state_dict()); epoch(m, *sites[b], g); ws.append(m.state_dict())
                glob.load_state_dict({k: sum(w[k] * (n / ns.sum()) for w, n in zip(ws, ns)) for k in w0})
            r["fedavg"] = c15.rmse(pred(glob, Xt), yt)
            Xp = np.concatenate([sites[b][0] for b in src])
            pm = mlp(F); pm.load_state_dict(w0)
            for _ in range(ROUNDS):
                epoch(pm, Xp, ytr, g)
            r["pooled"] = c15.rmse(pred(pm, Xt), yt)
            res["loo"].append(r); print(r, flush=True)
    M = {k: float(np.mean([r[k] for r in res["loo"]])) for k in ("mean", "fedavg", "pooled")}
    R = rules(sites, feats)
    tr = np.array([r["transfer"] for r in R])
    sc = float(spearmanr([r["C"] for r in R], tr).correlation); st = float(spearmanr([r["trust"] for r in R], tr).correlation)
    top = sorted(R, key=lambda r: -r["C"])[:max(1, len(R) // 10)]
    res.update({"loo_mean": M, "rules": {"n": len(R), "spearman_confidence": sc, "spearman_trust": st,
                "top_decile_claimed": float(np.mean([r["C"] for r in top])), "top_decile_transfer": float(np.mean([r["transfer"] for r in top]))}})
    res["verdict"] = {"T1": M["fedavg"] <= 1.10 * M["pooled"], "T2": M["fedavg"] <= 0.9 * M["mean"],
                      "T3": sc >= 0.5, "T4": st <= sc}
    json.dump(res, open(os.path.join(HERE, "results", "c15c_results.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("loo_mean", "rules", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
