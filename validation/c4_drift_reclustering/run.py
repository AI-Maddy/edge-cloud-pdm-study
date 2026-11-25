"""C4 validation: context drift monitor and dynamic re-clustering (notes 10 'Dynamic Re-Clustering', 13 sec. 9;
repository scripts/hfcf/federation.py drift() and recluster()).

Real drift: Beijing 12-station network, 2013-03..2017-02. Year 1 = warm-up, years 2-4 = test (1,096 days).
Each day every station has a context vector (daily mean meteorology, C1). Federation cells = K-Means (k=6) on
contexts in a trailing 365-day window, refit according to a policy. Each cell owns a ridge model, trained on
all past station-days in that cell (federated sufficient statistics), predicting next-day station PM2.5 from
today's PM2.5 and context. The cell model of today's cell makes tomorrow's prediction.
Policies:
  static       cells fitted once on year 1
  repo_rule    repository drift(): recluster when any node's context moved > threshold (1.0) since its last
               fit-time context (the rule as implemented)
  periodic_N   refit every N days (N = 30, 90)
  dist_trigger refit when the trailing-30-day mean distance of contexts to their nearest centroid exceeds the
               fit-time mean by more than 15% (distribution-level monitor, proposed here)
  daily        refit every day (upper reference for adaptivity)
Tests (fixed before running):
  T1 repo_rule fires on more than half of the days (i.e., per-node distance is not a usable drift signal)
  T2 dist_trigger RMSE within 1% of daily, using at most 10% as many re-clusterings
  T3 dist_trigger RMSE better than static
"""
import json, os, sys
import numpy as np
from sklearn.cluster import KMeans
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import beijing

K, WARM = 6, 365
names, C, P = beijing.daily()
S, D, F = C.shape
for s in range(S):                       # carry context / PM forward over missing days
    for d in range(1, D):
        if not np.isfinite(C[s, d]).all(): C[s, d] = C[s, d - 1]
        if not np.isfinite(P[s, d]): P[s, d] = P[s, d - 1]
mu = C[:, :WARM].reshape(-1, F).mean(0); sd = C[:, :WARM].reshape(-1, F).std(0)
Z = (C - mu) / sd


def fit(day):
    lo = max(0, day - 365)
    return KMeans(K, n_init=4, random_state=0).fit(Z[:, lo:day].reshape(-1, F))


def run(policy):
    km = fit(WARM); fit_ctx = Z[:, WARM - 1].copy(); fit_mean = km.transform(Z[:, WARM - 365:WARM].reshape(-1, F)).min(1).mean()
    refits, churn, prev, errs = 0, 0, None, []
    # cell sufficient statistics for ridge: features [1, pm_today, ctx(7)]
    nf = 2 + F
    A = np.zeros((K, nf, nf)) + np.eye(nf) * 1.0; b = np.zeros((K, nf))
    hist_x, hist_y = [], []
    def feats(s, d): return np.r_[1.0, P[s, d], Z[s, d]]
    def rebuild():
        nonlocal A, b
        A = np.zeros((K, nf, nf)) + np.eye(nf); b = np.zeros((K, nf))
        if hist_x:
            X = np.array(hist_x); y = np.array(hist_y); lab = km.predict(X[:, 2:])
            for c in range(K):
                m = lab == c; A[c] += X[m].T @ X[m]; b[c] += X[m].T @ y[m]
    for d in range(1, WARM):
        for s in range(S): hist_x.append(feats(s, d - 1)); hist_y.append(P[s, d])
    rebuild()
    for d in range(WARM, D - 1):
        do = False
        if policy == "daily": do = True
        elif policy.startswith("periodic"): do = (d - WARM) % int(policy.split("_")[1]) == 0 and d > WARM
        elif policy == "repo_rule": do = bool((np.linalg.norm(Z[:, d] - fit_ctx, axis=1) > 1.0).any())
        elif policy == "dist_trigger":
            recent = km.transform(Z[:, d - 29:d + 1].reshape(-1, F)).min(1).mean()
            do = recent > 1.15 * fit_mean
        if do:
            km = fit(d + 1); refits += 1; fit_ctx = Z[:, d].copy()
            fit_mean = km.transform(Z[:, max(0, d - 364):d + 1].reshape(-1, F)).min(1).mean()
            rebuild()
        lab = km.predict(Z[:, d])
        if prev is not None: churn += int((lab != prev).sum())
        prev = lab
        for s in range(S):
            x = feats(s, d); c = lab[s]
            pred = x @ np.linalg.solve(A[c], b[c])
            errs.append(pred - P[s, d + 1])
        # tomorrow's labels arrive: add today's samples to statistics
        for s in range(S):
            x = feats(s, d); c = lab[s]
            A[c] += np.outer(x, x); b[c] += x * P[s, d + 1]; hist_x.append(x); hist_y.append(P[s, d + 1])
    e = np.array(errs)
    return {"rmse_log_pm25": float(np.sqrt(np.mean(e ** 2))), "refits": refits, "test_days": D - 1 - WARM,
            "refit_fraction": refits / (D - 1 - WARM), "cell_changes_per_station_day": churn / (S * (D - 1 - WARM))}


# global (single-cell) reference
res = {}
for pol in ["static", "repo_rule", "periodic_30", "periodic_90", "dist_trigger", "daily"]:
    res[pol] = run(pol); print(pol, res[pol], flush=True)
K_saved = K; K = 1; res["single_global_model"] = run("static"); K = K_saved
print("single_global_model", res["single_global_model"])
d, dt, st = res["daily"], res["dist_trigger"], res["static"]
res["verdict"] = {"T1": res["repo_rule"]["refit_fraction"] > 0.5,
                  "T2": dt["rmse_log_pm25"] <= 1.01 * d["rmse_log_pm25"] and dt["refits"] <= 0.10 * d["refits"],
                  "T3": dt["rmse_log_pm25"] < st["rmse_log_pm25"]}
json.dump(res, open(os.path.join(HERE, "results", "c4_results.json"), "w"), indent=2)
print(res["verdict"])
