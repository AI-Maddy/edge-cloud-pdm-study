"""C2 validation: local edge model (12 sec. 3 and 8) as an RUL predictor on all four C-MAPSS subsets.
Models: repository TransformerEncoder + head, 1-D CNN, ridge on the flattened window (linear baseline).
T1 best model beats the published LSTM test RMSE on every subset (Zheng et al. 2017).
T2 context-aware (per-operating-cell) normalisation beats global normalisation on FD002 and FD004.
Seeds 0-2; mean and std reported. Test sets are used only for the final score."""
import json, os, sys, time
import numpy as np
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import torch
torch.set_num_threads(4)
from common import rul

SEEDS = [0, 1, 2]
out = {"published_rmse": rul.PUBLISHED, "runs": []}
path = os.path.join(HERE, "results", "c2_runs.json")
done = {(r["fd"], r["model"], r["context_norm"], r["seed"]) for r in json.load(open(path))["runs"]} if os.path.exists(path) else set()
if done:
    out = json.load(open(path))
jobs = [(fd, m, True) for fd in ["FD001", "FD002", "FD003", "FD004"] for m in ["ridge", "cnn1d", "transformer"]]
jobs += [(fd, m, False) for fd in ["FD002", "FD004"] for m in ["ridge", "cnn1d", "transformer"]]
for fd, m, ctx in jobs:
    Xtr, ytr, utr, Xte, yte, _ = rul.load_fd(fd, context=ctx)
    for s in (SEEDS if m != "ridge" else [0]):
        if (fd, m, ctx, s) in done:
            continue
        t = time.time()
        if m == "ridge":
            p = Ridge(1.0).fit(Xtr.reshape(len(Xtr), -1), ytr).predict(Xte.reshape(len(Xte), -1)).clip(0, rul.CAP)
            params = Xtr.shape[1] * Xtr.shape[2] + 1
        else:
            model = rul.make(m, Xtr.shape[2])
            model, _ = rul.train(model, Xtr, ytr, utr, seed=s)
            p = rul.predict(model, Xte)
            params = sum(q.numel() for q in model.parameters())
        r = {"fd": fd, "model": m, "context_norm": ctx, "seed": s, "rmse": rul.rmse(p, yte),
             "nasa_score": rul.nasa_score(p, yte), "params": int(params), "seconds": round(time.time() - t, 1)}
        out["runs"].append(r)
        print(r, flush=True)
        json.dump(out, open(path, "w"), indent=1)
