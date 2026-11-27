"""Sensitivity of the cells-vs-FedAvg comparison (review 2, point 3). Not a new test: the pre-specified decision rule
(mean difference <= -0.5 RMSE and paired Wilcoxon p < 0.05) is unchanged. Primary estimand: mean over the 35 sites
of the per-site RMSE difference (seed-averaged). Reported here: (a) site bootstrap CI (as in the paper), (b) fleet-
weighted mean (each of the 4 fleets counts equally), (c) cluster bootstrap that resamples fleets, then sites
within fleets, (d) per-fleet differences, (e) per-seed differences."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results", "c3b_runs.json")))
fd = np.array(r["meta"]["site_fd"]); M = {}
for x in r["runs"]: M.setdefault(x["method"], {})[x["seed"]] = np.array(x["site_rmse"])
rng = np.random.default_rng(0); F = sorted(set(fd)); out = {}
for a, b in [("cells", "fedavg"), ("hier_geo", "fedavg"), ("hier_ctx", "hier_geo"), ("fedavg", "central"), ("oracle", "fedavg")]:
    d = np.mean([M[a][s] for s in (0, 1, 2)], 0) - np.mean([M[b][s] for s in (0, 1, 2)], 0)
    site_bt = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    fw = float(np.mean([d[fd == f].mean() for f in F]))
    cl = []
    for _ in range(5000):
        fs = rng.choice(F, len(F))
        cl.append(np.mean([rng.choice(d[fd == f], (fd == f).sum()).mean() for f in fs]))
    out[f"{a}_vs_{b}"] = {"mean": float(d.mean()), "site_ci95": np.percentile(site_bt, [2.5, 97.5]).tolist(),
                          "fleet_weighted_mean": fw, "fleet_cluster_ci95": np.percentile(cl, [2.5, 97.5]).tolist(),
                          "per_fleet": {f: float(d[fd == f].mean()) for f in F},
                          "per_seed": [float((M[a][s] - M[b][s]).mean()) for s in (0, 1, 2)],
                          "share_sites_better": float((d < 0).mean())}
json.dump(out, open(os.path.join(HERE, "results", "sensitivity.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
