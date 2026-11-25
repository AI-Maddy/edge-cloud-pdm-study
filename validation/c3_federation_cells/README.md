# C3: Context-aware federation cells (CAFIG)


Clients: 35 sites of ~20 turbofan engines built from all four C-MAPSS subsets (FD001 5 sites, FD002 13, FD003 5,
FD004 12); official test engines split the same way. Site context = histogram of operating-condition cells (C1);
cells = K-Means on site contexts, k by silhouette, which picked k = 2 (single-condition vs six-condition sites,
silhouette 0.99). Same CNN1D (84k params), 25 rounds x 1 local epoch, sensors normalised per operating condition
(C1/C2). 3 seeds; per-site test RMSE (105 site-seed pairs). Run: `python c3_federation_cells/run.py` (about 1.5 h).

| Method | Mean site RMSE | FD001 | FD002 | FD003 | FD004 | Models downloaded per round |
| --- | --- | --- | --- | --- | --- | --- |
| Local only (25 epochs) | 21.21 ± 0.28 | 18.6 | 20.9 | 19.6 | 23.3 | 0 |
| FedAvg | 19.64 ± 0.18 | 18.7 | 19.1 | 19.6 | 20.6 | 1 |
| FedAvg + local fine-tune | 20.74 ± 0.10 | 19.2 | 20.3 | 21.3 | 21.6 | 1 |
| IFCA (k = 4) | 19.57 ± 0.20 | 18.9 | 19.2 | 19.4 | 20.4 | 4 |
| **CAFIG (context cells)** | 19.69 ± 0.16 | 18.9 | 19.2 | 20.1 | 20.4 | 1 |
| CAFIG + local fine-tune | 20.98 ± 0.27 | 19.7 | 20.5 | 21.9 | 21.6 | 1 |
| Oracle cells (true subset) | 19.62 ± 0.09 | 17.9 | 19.1 | 20.7 | 20.4 | 1 |

Paired over sites and seeds (Wilcoxon): FedAvg vs local -1.57 RMSE (p = 2e-6); CAFIG vs FedAvg +0.06 (p = 0.92);
CAFIG vs IFCA +0.13 (p = 0.48); CAFIG vs oracle +0.08 (p = 0.78).

| Test | Result | Pass |
| --- | --- | --- |
| T1 CAFIG beats FedAvg | tie (+0.06, p = 0.92) | **no** |
| T2 CAFIG within 1 RMSE of IFCA with 1 instead of k downloads | +0.13 at 1/4 of the downlink | yes |
| T3 CAFIG within 1 RMSE of oracle grouping | +0.08 | yes |

## What this establishes

- Federation clearly helps over isolated sites (-1.6 RMSE, about 7%), mostly on the six-condition subsets.
- Once sensors are normalised per operating condition (the C1 context used as a normaliser), sites become
  compatible and partitioning them into context cells adds nothing; even oracle cells do not beat one global
  model. The context earns its keep as an input normaliser, not as a partition, on this data.
- Context cells are still as good as loss-based clustering (IFCA) at a quarter of the download traffic, so cells
  are a cheap safe default when heterogeneity is unknown.
- Local fine-tuning after federation hurts (+1.1 to +1.3), because sites have few failures to learn from.
- Limitation: 25 rounds leave all federated models under-trained relative to centralised training (C2 reaches
  13.7-17 on the same subsets); comparisons are between methods at equal budget.
