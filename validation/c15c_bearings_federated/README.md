# C15c Federation, cold start and rule transfer on real bearings (review follow-up E6)

Why: the internal review said every federation, knowledge-sharing and cold-start result rests on C-MAPSS, which
is a simulation. This repeats those questions on real run-to-failure vibration data (XJTU-SY, conditions 1-2,
same features and onset rule as C15b). Each bearing with a detected onset (10) is one edge site. Leave one bearing
out: it is the cold-start site, the other 9 are the federation. Target: remaining fraction of post-onset life (0-1).
Run: `python c15c_bearings_federated/run.py` (about 2 minutes, CPU). Criteria fixed in `run.py` before running.

**Result: 2 of 4 tests pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 FedAvg MLP vs pooled MLP | within 10% | 0.354 vs 0.276 RMSE (+28%) | no |
| T2 FedAvg vs mean predictor (cold start) | <= 0.9x | 0.354 vs 0.294 (worse) | no |
| T3 rule confidence predicts precision at other bearings | Spearman >= 0.5 | 0.63 (829 rules) | yes |
| T4 weighted trust no better than confidence | trust <= confidence | 0.61 vs 0.63 | yes |

**What this says.**
- With 10 real bearings, learning across sites hardly helps a new bearing: even the pooled model is only 6% better
  than predicting the average, and FedAvg over 9 very unequal sites (23 to 406 snapshots each) is worse than the
  average. Real bearing degradation after onset differs too much from bearing to bearing for this small fleet.
  This is a negative result for M3/M5-style sharing of models on this data, and the paper says so.
- Rules do carry over: a rule's confidence at its source ranks how well it works at other bearings (Spearman 0.63),
  but it is overconfident (top 10% of rules claim precision 1.00 and deliver 0.67 elsewhere).
- The weighted trust score adds nothing over confidence alone, the same finding as C5 on C-MAPSS.

Per-bearing and per-seed numbers: `results/c15c_results.json`; log: `results/log.txt`.
