"""C3b (review follow-up E3 + E4): federation at convergence, a centralised upper bound, and a region tier.

Same 35 C-MAPSS sites, CNN1D and local optimiser as C3 (imported from c3_federation_cells/run.py), but
100 rounds x 1 local epoch instead of 25. Methods:
  local       25 local epochs (as C3; local training does not improve with more epochs on ~20 engines)
  central     one CNN1D trained on the pooled training windows of all 35 sites with early stopping
              (rul.train, 10% of engines held out) - upper bound that needs all raw data in one place
  fedavg      flat FedAvg, every site uploads to the cloud every round
  cells       FedAvg inside context cells (k = 2 by silhouette: single- vs six-condition sites)
  oracle      FedAvg inside the true fleet
  hier_geo    region tier, HierFAVG-style (Liu et al. 2020): 5 regions of 7 sites, assigned at random
              (geography is not related to fleet); sites upload to their region every round, regions
              average and upload to the cloud every KAPPA = 5 rounds, the cloud averages regions
  hier_ctx    the same with regions = context cells (k = 2)
Backhaul bytes counted per round: flat = 35 site uploads + 35 downloads over the WAN; hierarchical = region
uploads/downloads over the WAN every KAPPA rounds (site<->region traffic stays in the region).
Statistics: per-site RMSE averaged over seeds, then paired Wilcoxon over the 35 sites, bootstrap 95% CI.

Tests (fixed before running):
  T1 (E4) at 100 rounds, context cells beat flat FedAvg by >= 0.5 RMSE (paired, p < 0.05)
  T2 (E4) at 100 rounds, flat FedAvg is within 15% of the centralised upper bound (mean site RMSE)
  T3 (E4) at 100 rounds, FedAvg beats local-only training (paired, p < 0.05)
  T4 (E3) the region tier (hier_geo) is within 0.5 RMSE of flat FedAvg while using <= 10% of its WAN bytes
  T5 (E3) context-defined regions (hier_ctx) beat random regions (hier_geo) by >= 0.5 RMSE
"""
import json, os, sys, time
import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "c3_federation_cells"))
from common import rul  # noqa: E402
import run as c3  # noqa: E402
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
ROUNDS, KAPPA, NREG = 100, 5, 5
OUT = os.path.join(HERE, "results", os.environ.get("C3B_OUT", "c3b_runs.json"))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",") if x]
# optional explicit job list "seed:method,seed:method" so runs can be spread over processes
JOBS = [(int(j.split(":")[0]), j.split(":")[1]) for j in os.environ.get("JOBS", "").split(",") if j]
METHODS = ["fedavg", "hier_geo", "cells", "hier_ctx", "oracle", "central", "local"]


def save(res, path):
    tmp = path + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, path)


def fed(sites, seed, groups, hier=None):
    """groups: FedAvg group per site (None = flat). hier: region per site for HierFAVG."""
    g = np.random.default_rng(seed); torch.manual_seed(seed)
    F = sites[0]["Xtr"].shape[2]
    w0 = c3.get(rul.CNN1D(F)); work = rul.CNN1D(F)
    nparam = sum(p.numel() for p in work.parameters())
    ns = np.array([len(s["ytr"]) for s in sites])
    wan_models = 0
    if hier is None:
        groups = np.zeros(len(sites), int) if groups is None else groups
        cw = [w0 for _ in range(groups.max() + 1)]
        curve = []
        for r in range(ROUNDS):
            new = []
            for c in range(len(cw)):
                idx = np.where(groups == c)[0]; ws = []
                for i in idx:
                    c3.put(work, cw[c]); c3.local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); ws.append(c3.get(work))
                new.append(c3.avg(ws, ns[idx]))
            cw = new; wan_models += 2 * len(sites)
            if (r + 1) % 25 == 0:
                curve.append(float(np.mean(eval_w([cw[groups[i]] for i in range(len(sites))], sites, F))))
        site_w = [cw[groups[i]] for i in range(len(sites))]
    else:
        R = hier.max() + 1
        cloud = w0; reg = [w0 for _ in range(R)]; curve = []
        for r in range(ROUNDS):
            for q in range(R):
                idx = np.where(hier == q)[0]; ws = []
                for i in idx:
                    c3.put(work, reg[q]); c3.local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); ws.append(c3.get(work))
                reg[q] = c3.avg(ws, ns[idx])
            if (r + 1) % KAPPA == 0:
                rn = [ns[hier == q].sum() for q in range(R)]
                cloud = c3.avg(reg, rn); reg = [cloud for _ in range(R)]; wan_models += 2 * R
            if (r + 1) % 25 == 0:
                curve.append(float(np.mean(eval_w([cloud] * len(sites), sites, F))))
        site_w = [cloud] * len(sites)
    return eval_w(site_w, sites, F), {"wan_bytes_total": int(wan_models) * 4 * int(nparam), "curve_every_25": curve}


def eval_w(ws, sites, F):
    m = rul.CNN1D(F); out = []
    for w, s in zip(ws, sites):
        c3.put(m, w); out.append(rul.rmse(rul.predict(m, s["Xte"]), s["yte"]))
    return out


def central(sites, seed):
    X = np.concatenate([s["Xtr"] for s in sites]); y = np.concatenate([s["ytr"] for s in sites])
    u = np.concatenate([np.full(len(s["ytr"]), i * 1000) for i, s in enumerate(sites)])
    # unit ids: windows do not carry engine ids here, so hold out whole sites' windows in blocks by engine order
    u = u + np.concatenate([np.cumsum(np.r_[0, np.diff(s["ytr"]) > 0]) for s in sites])
    m, _ = rul.train(rul.CNN1D(X.shape[2]), X, y, u, seed=seed)
    return [rul.rmse(rul.predict(m, s["Xte"]), s["yte"]) for s in sites], {}


def stats(a, b):
    a, b = np.asarray(a), np.asarray(b); d = a - b
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "p": float(wilcoxon(a, b).pvalue)}


def main():
    sites = c3.build_sites()
    C = np.array([s["ctx"] for s in sites])
    cells = KMeans(2, n_init=10, random_state=0).fit_predict(C)
    fds = np.array([s["fd"] for s in sites]); oracle = np.unique(fds, return_inverse=True)[1]
    geo = np.random.default_rng(123).permutation(np.arange(len(sites)) % NREG)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"meta": {"rounds": ROUNDS, "kappa": KAPPA,
          "regions_geo": geo.tolist(), "cells": cells.tolist(), "site_fd": fds.tolist()}, "runs": []}
    done = {(r["method"], r["seed"]) for r in res["runs"]}
    jobs = JOBS or [(sd, m) for sd in SEEDS for m in METHODS]
    for seed, method in jobs:
        if True:
            if (method, seed) in done:
                continue
            t = time.time()
            if method == "fedavg": sr, cost = fed(sites, seed, None)
            elif method == "cells": sr, cost = fed(sites, seed, cells)
            elif method == "oracle": sr, cost = fed(sites, seed, oracle)
            elif method == "hier_geo": sr, cost = fed(sites, seed, None, hier=geo)
            elif method == "hier_ctx": sr, cost = fed(sites, seed, None, hier=cells)
            elif method == "central": sr, cost = central(sites, seed)
            else: sr, cost = c3.run("local", sites, seed)
            r = {"method": method, "seed": seed, "site_rmse": sr, "mean_site_rmse": float(np.mean(sr)),
                 **{k: v for k, v in cost.items() if k in ("wan_bytes_total", "curve_every_25")},
                 "seconds": round(time.time() - t, 1)}
            res["runs"].append(r); save(res, OUT)
            print({k: v for k, v in r.items() if k != "site_rmse"}, flush=True)
    import glob
    allruns = {(r["method"], r["seed"]): r for f in sorted(glob.glob(os.path.join(HERE, "results", "c3b_runs*.json")))
               for r in json.load(open(f))["runs"]}
    res["runs"] = list(allruns.values())
    if len({k[1] for k in allruns}) < 3 or len(allruns) < 21:
        print("incomplete:", sorted(allruns)); return
    OUTF = os.path.join(HERE, "results", "c3b_runs.json")
    M = {}
    for r in res["runs"]: M.setdefault(r["method"], []).append(r["site_rmse"])
    A = {m: np.mean(v, 0) for m, v in M.items()}
    W = {r["method"]: r.get("wan_bytes_total") for r in res["runs"]}
    s = {"mean": {m: float(v.mean()) for m, v in A.items()},
         "cells_vs_fedavg": stats(A["cells"], A["fedavg"]), "fedavg_vs_local": stats(A["fedavg"], A["local"]),
         "hier_geo_vs_fedavg": stats(A["hier_geo"], A["fedavg"]), "hier_ctx_vs_hier_geo": stats(A["hier_ctx"], A["hier_geo"]),
         "wan_bytes": W}
    s["verdict"] = {
        "T1": s["cells_vs_fedavg"]["mean_diff"] <= -0.5 and s["cells_vs_fedavg"]["p"] < 0.05,
        "T2": s["mean"]["fedavg"] <= 1.15 * s["mean"]["central"],
        "T3": s["fedavg_vs_local"]["mean_diff"] < 0 and s["fedavg_vs_local"]["p"] < 0.05,
        "T4": abs(s["hier_geo_vs_fedavg"]["mean_diff"]) <= 0.5 and W["hier_geo"] <= 0.10 * W["fedavg"],
        "T5": s["hier_ctx_vs_hier_geo"]["mean_diff"] <= -0.5}
    res["summary"] = s; save(res, OUTF)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
