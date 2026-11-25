"""C15 validation: the C1/C2/C3/C10 concepts on a second industrial asset, XJTU-SY run-to-failure bearings
(10 bearings, operating conditions 35 Hz/12 kN and 37.5 Hz/11 kN; features from common/xjtu.py).

Target: remaining-life fraction (1 - minute / life) of every one-minute snapshot; metric RMSE over all
snapshots of a held-out bearing (leave-one-bearing-out, so the test bearing is never seen in training).
Model: gradient-boosted trees on causal 5-minute rolling means of the 32 vibration features.

Context normalisation here = each bearing's own first 10 minutes (its healthy baseline, available on any
newly installed bearing without labels), the bearing analogue of C2's per-operating-condition normaliser.

Tests (fixed before running):
  T1 (C1) operating condition recovered from vibration alone: K-Means k=2 on each bearing's healthy baseline
     features groups bearings by condition, ARI >= 0.9
  T2 (C2) baseline (context) normalisation lowers leave-one-out RMSE by >= 10% vs global z-scoring
  T3 (C3) context cell: training only on bearings of the same condition is at least as good as training on
     all bearings (mean RMSE over the 10 held-out bearings)
  T4 (C10) cross-condition: a model trained on condition 1 only, applied to condition 2 (and vice versa),
     is within 20% of the in-condition leave-one-out RMSE
"""
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import xjtu  # noqa: E402

BASE, SMOOTH = 10, 5


def prepare():
    d = xjtu.load().sort_values(["bearing", "minute"]).reset_index(drop=True)
    feats = [c for c in d.columns if c[:2] in ("h_", "v_")]
    for c in feats:                                           # log-scale the magnitude features
        if c.endswith(("_rms", "_peak")):
            d[c] = np.log10(d[c])
    gsd = d[feats].std()
    raw, ctx = [], []
    for _, g in d.groupby("bearing", sort=False):
        sm = g[feats].rolling(SMOOTH, min_periods=1).mean()
        b = g[feats].iloc[:BASE]
        raw.append(sm)
        ctx.append((sm - b.mean()) / (b.std() + 0.1 * gsd))
    d_raw = np.vstack([r.to_numpy() for r in raw]); d_ctx = np.vstack([c.to_numpy() for c in ctx])
    mu, sd = d_raw.mean(0), d_raw.std(0) + 1e-9
    return d, feats, (d_raw - mu) / sd, d_ctx


def fit_predict(Xtr, ytr, Xte, seed):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    return np.clip(m.fit(Xtr, ytr).predict(Xte), 0, 1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    d, feats, Xg, Xc = prepare()
    y = (1 - d["minute"] / d["n_total"]).to_numpy()
    bear, cond = d["bearing"].to_numpy(), d["condition"].to_numpy()
    B = list(dict.fromkeys(bear))
    res = {"n_snapshots": int(len(d)), "bearings": {b: int((bear == b).sum()) for b in B}}

    # T1 condition from healthy baseline vibration
    base = np.stack([d.loc[bear == b, feats].iloc[:BASE].mean().to_numpy() for b in B])
    base = (base - base.mean(0)) / (base.std(0) + 1e-9)
    truth = [cond[bear == b][0] for b in B]
    lab = KMeans(2, n_init=20, random_state=0).fit_predict(base)
    res["T1"] = {"ari": float(adjusted_rand_score(truth, lab)), "clusters": {b: int(l) for b, l in zip(B, lab)}}

    # T2/T3 leave-one-bearing-out, 3 seeds
    rows = []
    for seed in (0, 1, 2):
        for b in B:
            te = bear == b
            same = (cond == cond[te][0]) & ~te
            r = {"seed": seed, "bearing": b, "condition": cond[te][0], "life_min": int(te.sum())}
            r["global_all"] = rmse(fit_predict(Xg[~te], y[~te], Xg[te], seed), y[te])
            r["context_all"] = rmse(fit_predict(Xc[~te], y[~te], Xc[te], seed), y[te])
            r["context_cell"] = rmse(fit_predict(Xc[same], y[same], Xc[te], seed), y[te])
            r["predict_mean"] = rmse(np.full(te.sum(), y[~te].mean()), y[te])
            rows.append(r)
    res["loo"] = rows
    m = lambda k, c=None: float(np.mean([r[k] for r in rows if c is None or r["condition"] == c]))
    res["loo_mean"] = {k: m(k) for k in ("global_all", "context_all", "context_cell", "predict_mean")}

    # T4 cross-condition (context normalisation)
    conds = list(dict.fromkeys(cond))
    cross = {}
    for src in conds:
        tgt = [c for c in conds if c != src][0]
        sm, tm = cond == src, cond == tgt
        errs = [rmse(fit_predict(Xc[sm], y[sm], Xc[(bear == b)], s), y[bear == b])
                for s in (0, 1, 2) for b in B if cond[bear == b][0] == tgt]
        cross[f"{src}->{tgt}"] = {"cross": float(np.mean(errs)), "in_condition_cell": m("context_cell", tgt)}
    res["T4"] = cross

    res["verdict"] = {
        "T1": res["T1"]["ari"] >= 0.9,
        "T2": res["loo_mean"]["context_all"] <= 0.9 * res["loo_mean"]["global_all"],
        "T3": res["loo_mean"]["context_cell"] <= res["loo_mean"]["context_all"],
        "T4": all(v["cross"] <= 1.2 * v["in_condition_cell"] for v in cross.values()),
    }
    json.dump(res, open(os.path.join(HERE, "results", "c15_results.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("T1", "loo_mean", "T4", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
