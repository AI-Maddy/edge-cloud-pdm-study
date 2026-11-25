# C15 Concepts C1/C2/C3/C10 on a second asset: XJTU-SY bearings

**Data.** XJTU-SY run-to-failure bearings (Wang et al., IEEE Trans. Reliability 2020), conditions 1
(35 Hz, 12 kN) and 2 (37.5 Hz, 11 kN): 10 bearings, 2,182 one-minute vibration snapshots (2 x 32,768
samples at 25.6 kHz). Streamed once from the GitHub mirror `rajeevfee/XJTU-SY-Bearing-Dataset`
(condition 3 is not in that mirror); only per-snapshot features are kept, in `../data/xjtu_sy/features.csv.gz`,
with the SHA-256 of every source file in `sources.csv.gz`. Rebuild: `python common/xjtu.py`.

**Setup.** Target = remaining-life fraction of every snapshot; leave-one-bearing-out (the test bearing is
never in training); gradient-boosted trees on causal 5-minute means of 32 vibration features; 3 seeds.
"Context normalisation" = the bearing's own first 10 minutes (healthy baseline, label-free).

**Result: 1 of 4 tests pass.** Run: `OMP_NUM_THREADS=1 python c15_bearings/run.py` (about 3 minutes).

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 condition recovered from healthy vibration (C1) | ARI >= 0.9 | ARI 0.10 (8 of 10 bearings in one cluster) | no |
| T2 baseline normalisation vs global z-score (C2) | >= 10% lower RMSE | 0.236 vs 0.240 (-1.6%) | no |
| T3 same-condition cell vs all bearings (C3) | cell <= all | 0.253 vs 0.236 (cell 7% worse) | no |
| T4 cross-condition within 20% of in-condition (C10) | <= 1.2x | 0.201 / 0.229 vs 0.251 / 0.256 | yes |

Reference: always predicting the training mean gives 0.289.

**What this says.**
- The C-MAPSS findings do not carry over as they stand. On C-MAPSS the operating condition is an
  explicit input and splits the data cleanly (C1 ARI 1.0); here speed/load changes by only 7%/8% and is
  not separable from bearing-to-bearing variation in healthy vibration.
- With 4-5 run-to-failure bearings per condition, pooling everything (more data) beats a same-condition
  cell; T4 passes for the same reason (training on 5 other-condition bearings beats training on 4
  same-condition ones), so it is not evidence that context handling works.
- All models are only 15-20% better than predicting the mean. Bearings stay flat for most of their life
  and fail quickly; life fraction from minute one is largely unpredictable. The standard practice in the
  bearing literature is to predict only after a detected degradation onset (first predicting time).
  That is the obvious next test, with its criteria to be fixed before running it.

Per-bearing leave-one-out RMSE (mean of 3 seeds) is in `results/c15_results.json` (`loo`).

## C15b: after a detected degradation onset (first predicting time)

`run_fpt.py` (criteria fixed in the script before running). Onset = 3 consecutive minutes with 5-minute mean
horizontal RMS above the bearing's own first-10-minute mean + 3 sd; target = remaining fraction of the
post-onset period; only post-onset snapshots are used. **2 of 4 pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 onset found before 90% of life | >= 8 of 10 bearings | 9 of 10 (median at 30% of life; Bearing2_1 at 93%) | yes |
| T2 model clearly beats the mean | RMSE <= 0.7x mean predictor | 0.257 vs 0.294 (0.88x); best (cell) 0.229 (0.78x) | no |
| T3 same-condition cell vs all bearings (C3) | cell <= all | 0.229 vs 0.257 (cell 11% better) | yes |
| T4 baseline normalisation vs global (C2) | >= 10% better | 0.257 vs 0.254 (no gain) | no |

With the standard formulation the context-cell idea does hold on bearings (same-condition training is 11%
better), reversing C15's T3. Per-bearing baseline normalisation still adds nothing, and 4-5 bearings per
condition are too few for a strong remaining-life model.
