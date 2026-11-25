"""C1 validation: context vector + unsupervised context clustering (concept files 07, 08, 12 sec. 4 and 7).

Uses the repository's own pipeline (scripts/hfcf/features.py: encode -> clustering / scene_clusters) on
real NASA C-MAPSS data, where the six flight conditions of FD002/FD004 are published and give a reference
labelling.

Tests
  T1  Context clustering of operating settings recovers the 6 flight conditions (ARI, NMI) for each method.
  T2  Model selection: silhouette over k = 2..10 picks the true number of contexts.
  T3  Stability: ARI between runs with different seeds and between FD002 and FD004 (fit on one, apply to other).
  T4  Context discovery from sensors only (no settings): does clustering 21 raw sensor readings recover the
      flight condition?  (07 'environment clustering' from observations)
  T5  Usefulness: share of raw sensor variance explained by the context cell (what makes cells good
      federation groups), versus a random partition with the same cell sizes.
Pass criteria are fixed in CRITERIA before running.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from common import cmapss  # noqa: E402
from hfcf import features  # noqa: E402  (repository implementation under test)

CRITERIA = {"T1_best_ari": 0.95, "T2_selects_true_k": True, "T3_min_stability_ari": 0.99,
            "T4_sensor_only_ari": 0.90, "T5_min_variance_explained": 0.50}
rng = np.random.default_rng(0)


def repo_cluster(ctx, method, **cfg):
    """Run the repository's encode() and clustering() on context dicts."""
    rows = [{"id": str(i), "context": {f"c{j}": float(v) for j, v in enumerate(r)}} for i, r in enumerate(ctx)]
    enc = features.encode({"records": rows}, {})
    t = time.time()
    if method == "louvain":
        emb = np.array([r["context_embedding"] for r in enc["records"]])
        for r, e in zip(enc["records"], emb):
            r["embedding"] = e.tolist()
        out = features.scene_clusters({"records": enc["records"]}, {"method": "louvain", "field": "embedding",
                                                                      "threshold": cfg.get("threshold", 0.95)})
    else:
        out = features.clustering(enc, {**cfg, "field": "context_embedding"}, method)
    labels = np.array([r["cluster"] for r in out["records"]])
    return labels, time.time() - t


def main():
    res = {"criteria": CRITERIA}
    d2, d4 = cmapss.load("FD002"), cmapss.load("FD004")
    A = np.vstack([d2["arr"], d4["arr"]])
    truth, _ = cmapss.true_regime(A[:, 2:5])
    settings = A[:, 2:5]
    sensors = A[:, 5:]
    n = len(A)
    sub = rng.choice(n, 3000, replace=False)       # O(n^2) methods run on a fixed random subsample
    res["n_rows"] = int(n)

    # T1 every method on the settings context
    t1 = {}
    for name, idx, cfg in [("kmeans", None, {"clusters": 6}), ("hdbscan", sub, {"min_cluster_size": 50}),
                           ("dbscan", sub, {"eps": 0.3, "min_samples": 10}),
                           ("spectral", sub, {"clusters": 6}), ("louvain", sub, {"threshold": 0.95})]:
        ix = np.arange(n) if idx is None else idx
        lab, sec = repo_cluster(settings[ix], name, **cfg)
        t1[name] = {"n": int(len(ix)), "ari": ari(truth[ix], lab), "nmi": nmi(truth[ix], lab),
                    "clusters_found": int(len(set(lab) - {-1})), "noise_frac": float(np.mean(lab == -1)),
                    "seconds": round(sec, 2)}
    res["T1"] = t1

    # T2 choose k by silhouette (k-means through the repo pipeline)
    sil = {}
    for k in range(2, 11):
        lab, _ = repo_cluster(settings[sub], "kmeans", clusters=k)
        sil[k] = float(silhouette_score(StandardScaler().fit_transform(settings[sub]), lab))
    res["T2"] = {"silhouette": sil, "selected_k": int(max(sil, key=sil.get)), "true_k": 6}

    # T3 stability over seeds, and transfer FD002 -> FD004
    labs = [repo_cluster(settings[sub], "kmeans", clusters=6, seed=s)[0] for s in range(5)]
    seed_ari = [ari(labs[0], l) for l in labs[1:]]
    from sklearn.cluster import KMeans
    sc = StandardScaler().fit(d2["arr"][:, 2:5])
    km = KMeans(6, n_init=10, random_state=0).fit(sc.transform(d2["arr"][:, 2:5]))
    t4truth, _ = cmapss.true_regime(d4["arr"][:, 2:5])
    transfer = ari(t4truth, km.predict(sc.transform(d4["arr"][:, 2:5])))
    res["T3"] = {"seed_ari_min": float(min(seed_ari)), "fd002_to_fd004_ari": float(transfer)}

    # T4 sensors only: drop near-constant sensors, standardise, k-means k=6 via the repo pipeline
    keep = sensors.std(0) > 1e-3 * np.abs(sensors.mean(0)) + 1e-9
    lab, _ = repo_cluster(sensors[:, keep], "kmeans", clusters=6)
    res["T4"] = {"sensors_used": int(keep.sum()), "ari": float(ari(truth, lab)), "nmi": float(nmi(truth, lab))}

    # T5 variance explained by the context cells (k-means cells from T1) vs random equal-size partition
    lab_k, _ = repo_cluster(settings, "kmeans", clusters=6)
    Z = StandardScaler().fit_transform(sensors[:, keep])

    def r2(lbl):
        within = sum(((Z[lbl == c] - Z[lbl == c].mean(0)) ** 2).sum() for c in np.unique(lbl))
        return 1 - within / ((Z - Z.mean(0)) ** 2).sum()
    res["T5"] = {"variance_explained_context_cells": float(r2(lab_k)),
                 "variance_explained_random_cells": float(r2(rng.permutation(lab_k)))}

    best = max(v["ari"] for v in t1.values())
    res["verdict"] = {
        "T1": best >= CRITERIA["T1_best_ari"],
        "T2": res["T2"]["selected_k"] == 6,
        "T3": min(res["T3"]["seed_ari_min"], res["T3"]["fd002_to_fd004_ari"]) >= CRITERIA["T3_min_stability_ari"],
        "T4": res["T4"]["ari"] >= CRITERIA["T4_sensor_only_ari"],
        "T5": res["T5"]["variance_explained_context_cells"] >= CRITERIA["T5_min_variance_explained"],
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "c1_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
