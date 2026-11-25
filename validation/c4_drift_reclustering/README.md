# C4: Context drift monitor and dynamic re-clustering


Data: Beijing 12-station network. Year 1 warm-up, years 2-4 test (1,095 days x 12 stations). Each day a station's
context (daily mean meteorology) picks a federation cell (K-Means, k = 6, fitted on a trailing 365-day window);
the cell's ridge model (federated sufficient statistics over all past station-days in that cell) predicts the
station's next-day PM2.5. Run: `python c4_drift_reclustering/run.py` (about 40 min, mostly the daily-refit reference).

| Policy | Next-day RMSE (log PM2.5) | Re-clusterings | Cell changes per station-day |
| --- | --- | --- | --- |
| Single global model (no cells) | 0.681 | 0 | 0 |
| Static cells (year 1) | 0.665 | 0 | 0.40 |
| Repository rule (any node moved > 1.0) | 0.665 | 1,009 (92% of days) | 0.75 |
| Periodic, every 30 days | 0.665 | 36 | 0.38 |
| Periodic, every 90 days | 0.664 | 12 | 0.37 |
| Distribution trigger (+15% mean distance) | **0.663** | 130 (12%) | 0.42 |
| Daily refit | 0.665 | 1,095 | 0.81 |

| Test | Result | Pass |
| --- | --- | --- |
| T1 repository per-node rule fires on > 50% of days | fires on 92% | yes (the rule is not a usable drift signal) |
| T2 distribution trigger within 1% of daily refit using <= 10% of its re-clusterings | RMSE 0.3% better than daily, but 11.9% of re-clusterings | **no** (just over the budget) |
| T3 distribution trigger better than static | 0.663 vs 0.665 | yes (small) |

## What this establishes

- Context cells pay off: every cell policy beats the single global model by about 2.5%.
- Re-clustering policy hardly matters for accuracy here (all within 0.3%), even though the cells drift (C1 T7).
  The cell models are re-estimated from all past data in each cell, which absorbs most of the drift.
- The repository's per-node drift rule is unusable as written: daily weather always moves a node by more than the
  threshold, so it re-clusters almost every day and doubles cell churn (and hence control traffic) for no gain.
  A distribution-level monitor or a slow periodic refit (90 days) gives the same accuracy at 1-12% of the cost.
- Re-clustering should be justified by cost (control traffic, model churn), not by accuracy, on this kind of data.
