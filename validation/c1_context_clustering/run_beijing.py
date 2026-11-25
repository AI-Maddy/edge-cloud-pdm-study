"""C1 part B: context clustering on a real, continuous sensing context (Beijing 12-station network).
No reference labels exist, so validation is by usefulness and stability:
  T6  silhouette-selected k; share of next-day station PM2.5 variance explained by today's context cell,
      versus random cells of equal size and versus calendar month (a strong naive context)
  T7  year-to-year stability: cells fit on year 1 vs refit on year k, compared on year-k days (ARI)
"""
import json, os, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import beijing

CRITERIA = {"T6_gain_over_random": 0.10, "T7_min_year_ari": 0.70}
names, C, P = beijing.daily()
S, D, F = C.shape
X = C.reshape(-1, F); y_next = np.concatenate([np.r_[p[1:], np.nan] for p in P])
ok = np.isfinite(X).all(1) & np.isfinite(y_next)
year = np.tile(np.arange(D) // 365, S).clip(max=3)
month = np.tile((np.arange(D) // 30.44).astype(int) % 12, S)
mu, sd = X[ok & (year == 0)].mean(0), X[ok & (year == 0)].std(0)
Z = (X - mu) / sd
rng = np.random.default_rng(0)

def eta2(lbl, y):
    m = y.mean(); return sum(((y[lbl == c].mean() - m) ** 2) * (lbl == c).sum() for c in np.unique(lbl)) / ((y - m) ** 2).sum()

tr = ok & (year == 0); te = ok & (year > 0)
sil = {}
for k in range(2, 11):
    km = KMeans(k, n_init=10, random_state=0).fit(Z[tr])
    sil[k] = float(silhouette_score(Z[tr], km.labels_, sample_size=3000, random_state=0))
k = max(sil, key=sil.get)
res = {"criteria": CRITERIA, "silhouette": sil, "selected_k": k}
for kk in sorted({k, 4, 6, 8}):
    km = KMeans(kk, n_init=10, random_state=0).fit(Z[tr])
    lab = km.predict(Z[te])
    res[f"k{kk}"] = {"eta2_next_day_pm25_test": float(eta2(lab, y_next[te])),
                     "eta2_random_cells": float(eta2(rng.permutation(lab), y_next[te]))}
res["eta2_calendar_month"] = float(eta2(month[te], y_next[te]))
kk = 6
km0 = KMeans(kk, n_init=10, random_state=0).fit(Z[tr])
res["T7_year_ari_k6"] = {}
for yr in (1, 2, 3):
    m = ok & (year == yr)
    kmy = KMeans(kk, n_init=10, random_state=0).fit(Z[m])
    res["T7_year_ari_k6"][f"year{yr+1}"] = float(ari(km0.predict(Z[m]), kmy.labels_))
best = max(v["eta2_next_day_pm25_test"] - v["eta2_random_cells"] for kx, v in res.items() if kx.startswith("k") and isinstance(v, dict) and "eta2_random_cells" in v)
res["verdict"] = {"T6": best >= CRITERIA["T6_gain_over_random"],
                  "T7": min(res["T7_year_ari_k6"].values()) >= CRITERIA["T7_min_year_ari"]}
json.dump(res, open(os.path.join(HERE, "results", "c1_beijing_results.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
