# C1: Context vector and context clustering


Implementation under test: the repository's own `scripts/hfcf/features.py` (`encode`, `clustering`, `scene_clusters`), unchanged.

Run: `python c1_context_clustering/run.py` (about 1 min) and `python c1_context_clustering/run_beijing.py`.
Pass criteria were fixed in the scripts before running.

## A. NASA C-MAPSS FD002 + FD004 (115,008 engine cycles, 6 published flight conditions)

| Test | Result | Criterion | Pass |
| --- | --- | --- | --- |
| T1 recover flight conditions from operating settings | K-Means ARI 1.000; spectral 1.000; DBSCAN (eps 0.3) 1.000; HDBSCAN (min size 50) 0.511 with 18% noise; Louvain (cosine > 0.95) 0.666, 4 clusters | best ARI >= 0.95 | yes |
| T2 choose number of contexts by silhouette | k = 6 selected (silhouette 0.997) | selects 6 | yes |
| T3 stability | ARI 1.000 across 5 seeds; cells fit on FD002 applied to FD004: ARI 1.000 | >= 0.99 | yes |
| T4 discover context from 21 raw sensors only (no settings) | ARI 0.992 | >= 0.90 | yes |
| T5 sensor variance explained by context cell | 99.2% (random equal-size cells: 0.0%) | >= 50% | yes |

## B. Beijing 12-station air-quality network (daily meteorological context, no reference labels)

| Test | Result | Criterion | Pass |
| --- | --- | --- | --- |
| T6 next-day PM2.5 variance explained by today's context cell (fit year 1, test years 2-4) | k = 4 (silhouette pick): 19.8%; k = 6: 25.6%; random cells 0.0%; calendar month 5.1% | gain over random >= 10 pts | yes |
| T7 year-to-year stability of cells (year-1 cells vs refit, k = 6) | ARI 0.72, 0.61, 0.59 for years 2, 3, 4 | >= 0.70 | **no** |

## What this establishes

- The context vector + clustering stage works on real data when contexts are discrete (C-MAPSS), and even
  sensor-only context discovery works (T4). Context cells there explain almost all sensor variance, so
  they are the right federation grouping for C3.
- On a continuous real context (Beijing weather) cells are useful (5x more explanatory than calendar month)
  but **drift**: cells fitted once lose agreement over the years. This is the motivating evidence for
  C4 (drift monitor and dynamic re-clustering).
- Repository defaults need care: HDBSCAN (min_cluster_size 50) over-splits and Louvain on a cosine threshold
  graph under-splits. K-Means with silhouette-selected k is the recommended default for the cell builder.
