"""C15b: the C15 bearing tests re-run in the standard bearing formulation, prediction after a detected
degradation onset (first predicting time, FPT), as proposed in C15's README before this was run.

FPT = first minute at which the 5-minute mean of horizontal RMS exceeds the bearing's own first-10-minute
mean + 3 sd for 3 consecutive minutes (3-sigma rule, label-free). After FPT the target is the remaining
fraction of the post-onset period, (life - minute) / (life - FPT). Everything else as in C15 (features,
gradient-boosted trees, leave-one-bearing-out, 3 seeds); only post-FPT snapshots are trained and scored.

Tests (fixed before running):
  T1 onset detected before 90% of life on >= 8 of 10 bearings
  T2 post-onset model (baseline normalisation, all other bearings) RMSE <= 0.7 x predicting the mean
  T3 (C3) same-condition cell at least as good as all bearings
  T4 (C2) baseline normalisation >= 10% lower RMSE than global z-scoring
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as c15  # noqa: E402  (same features and model as C15)


def onset(rms, base=10, k=3.0, hold=3):
    mu, sd = rms[:base].mean(), rms[:base].std() + 1e-9
    above = rms > mu + k * sd
    for i in range(base, len(rms) - hold + 1):
        if above[i:i + hold].all():
            return i
    return None


def main():
    d, feats, Xg, Xc = c15.prepare()
    bear, cond = d["bearing"].to_numpy(), d["condition"].to_numpy()
    minute, life = d["minute"].to_numpy(), d["n_total"].to_numpy()
    B = list(dict.fromkeys(bear))
    fpt, keep, y = {}, np.zeros(len(d), bool), np.full(len(d), np.nan)
    for b in B:
        m = bear == b
        rms = d.loc[m, "h_rms"].rolling(c15.SMOOTH, min_periods=1).mean().to_numpy()   # already log10
        i = onset(rms)
        fpt[b] = None if i is None else int(minute[m][i])
        if i is not None:
            post = m & (minute >= fpt[b])
            keep |= post
            y[post] = (life[post] - minute[post]) / max(1, life[post][0] - fpt[b])
    res = {"fpt_minute": fpt, "life": {b: int(life[bear == b][0]) for b in B},
           "fpt_frac": {b: (None if fpt[b] is None else fpt[b] / life[bear == b][0]) for b in B}}
    usable = [b for b in B if fpt[b] is not None]
    rows = []
    for seed in (0, 1, 2):
        for b in usable:
            te = keep & (bear == b); tr = keep & ~(bear == b)
            same = tr & (cond == cond[te][0])
            r = {"seed": seed, "bearing": b, "n_post": int(te.sum())}
            r["global_all"] = c15.rmse(c15.fit_predict(Xg[tr], y[tr], Xg[te], seed), y[te])
            r["context_all"] = c15.rmse(c15.fit_predict(Xc[tr], y[tr], Xc[te], seed), y[te])
            r["context_cell"] = c15.rmse(c15.fit_predict(Xc[same], y[same], Xc[te], seed), y[te]) if same.sum() > 20 else None
            r["predict_mean"] = c15.rmse(np.full(te.sum(), y[tr].mean()), y[te])
            rows.append(r)
    res["loo"] = rows
    mean = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None]))
    paired = [r for r in rows if r["context_cell"] is not None]
    res["loo_mean"] = {k: mean(k) for k in ("global_all", "context_all", "context_cell", "predict_mean")}
    res["cell_vs_all_paired"] = {"cell": float(np.mean([r["context_cell"] for r in paired])),
                                 "all": float(np.mean([r["context_all"] for r in paired])), "n": len(paired)}
    early = sum(1 for b in B if fpt[b] is not None and res["fpt_frac"][b] < 0.9)
    res["verdict"] = {"T1": early >= 8,
                      "T2": res["loo_mean"]["context_all"] <= 0.7 * res["loo_mean"]["predict_mean"],
                      "T3": res["cell_vs_all_paired"]["cell"] <= res["cell_vs_all_paired"]["all"],
                      "T4": res["loo_mean"]["context_all"] <= 0.9 * res["loo_mean"]["global_all"]}
    json.dump(res, open(os.path.join(HERE, "results", "c15b_fpt_results.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("fpt_frac", "loo_mean", "cell_vs_all_paired", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
