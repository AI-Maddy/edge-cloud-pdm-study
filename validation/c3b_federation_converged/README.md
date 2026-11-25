# C3b Federation at convergence, centralised bound and a region tier (review follow-ups E3 + E4)

Why: the review said 25 rounds left the federated models under-trained, that there was no centralised baseline,
and that the region tier of the architecture was never tested on the maintenance task. Same 35 C-MAPSS sites,
CNN1D and local optimiser as C3, but 100 rounds; regions follow HierFAVG (sites -> region every round,
regions -> cloud every 5 rounds). Per-site RMSE averaged over 3 seeds; paired Wilcoxon over 35 sites with
bootstrap 95% CI. Criteria fixed in `run.py` before running.

Run: `python c3b_federation_converged/run.py` (one process, about 6 h on 4 cores), or spread the 21 runs over
processes with `JOBS="0:fedavg,0:cells" C3B_OUT=c3b_runs_a.json THREADS=1 python ...`; the summary is computed
from all `results/c3b_runs*.json` once every run is present. (A first launch lost the regional runs to a JSON
save bug, now fixed; they were rerun with identical settings and seeds.)

**Result: 3 of 5 tests pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 context cells beat FedAvg at 100 rounds | <= -0.5 RMSE, p < 0.05 | -0.44, CI [-0.86, -0.06], p = 0.14 | no |
| T2 FedAvg near centralised training | within 15% | 16.35 vs 14.74 (+10.9%) | yes |
| T3 FedAvg beats local only | p < 0.05 | -4.84 (23%), CI [-5.91, -3.73], p = 2e-8 | yes |
| T4 region tier ties FedAvg with <= 10% of WAN bytes | both | -0.08 (p = 0.25) at 2.9% of WAN bytes | yes |
| T5 context regions beat random regions | <= -0.5 | -0.05 (p = 0.52) | no |

| Method | Mean | FD001 | FD002 | FD003 | FD004 | WAN bytes (100 rounds) |
| --- | --- | --- | --- | --- | --- | --- |
| Local only | 21.19 | 18.6 | 20.9 | 19.5 | 23.3 | 0 |
| FedAvg | 16.35 | 15.5 | 15.8 | 17.0 | 17.0 | 2.36 GB |
| Context cells (k = 2) | 15.91 | 14.6 | 15.6 | 15.2 | 17.1 | 2.36 GB |
| Oracle groups (fleet) | 16.78 | 15.0 | 16.9 | 16.0 | 17.8 | 2.36 GB |
| Regions, random (5) | 16.27 | 15.5 | 15.7 | 16.8 | 17.0 | 67 MB |
| Regions = context cells (2) | 16.22 | 15.2 | 15.7 | 16.9 | 16.9 | 27 MB |
| Centralised (all raw data) | 14.74 | 14.3 | 14.3 | 14.8 | 15.4 | - |

**What this says.**
- 25 rounds were far from converged: FedAvg goes from 19.6 to 16.3 RMSE, and federation's gain over local training
  grows from 7% to 23%. It ends within 11% of centralised training, without moving raw data.
- With more rounds, context cells pull slightly ahead of one global model (-0.44, mostly FD001 and FD003), but
  not by the pre-set margin or significance; oracle fleet groups are worse, because the groups are smaller. The
  "cells tie FedAvg" conclusion of C3 holds at the pre-set threshold, with a small trend in favour of cells.
- A region tier costs nothing in accuracy and cuts WAN backhaul 35x (random regions) to 88x (two context
  regions); the site-to-region traffic stays local. Which sites share a region does not matter for accuracy.
- Learning curves (every 25 rounds) are still falling at round 100, so these are not final values either.

Results: `results/c3b_runs*.json` (summary in `c3b_runs.json`), logs `results/log*.txt`.
