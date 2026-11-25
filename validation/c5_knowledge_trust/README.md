# C5: Knowledge objects, trust and confidence


Knowledge objects follow the original design's `thermal_runaway` example: a site mines condition -> outcome rules
("normalised sensor s above theta -> failure within 30 cycles") with confidence (source precision), support
(source firings), quality (precision on the source's own held-out engines) and source reliability. 24 C-MAPSS
sites (from C3) are sources; 11 never-seen sites are targets. 1,344 rules per seed, 5 seeds.
Run: `python c5_knowledge_trust/run.py` (about 2 min).

| How rules are ranked | Spearman with precision at unseen sites (mean of 5 seeds) |
| --- | --- |
| Repository trust, support scale 200 | 0.06 |
| Repository trust, default support scale 1000 | 0.46 |
| Confidence alone | **0.67** |
| Quality (source hold-out precision) alone | 0.56 |
| Learned non-negative weights (fitted on source sites only) | 0.67 (weights: confidence 0.6-0.7, quality 0.06, support 0, reliability 0) |

| Test | Result | Pass |
| --- | --- | --- |
| T1 repository trust Spearman >= 0.5 | 0.06 (scale 200) / 0.46 (default) | **no** |
| T2 repository trust ranks at least as well as confidence alone | 0.46 vs 0.67 | **no** |
| T3 top-decile confidence calibrated within 0.10 | claimed 0.968, achieved 0.909 at unseen sites | yes |
| T4 (added after T1/T2 failed) learned weights beat confidence alone | 0.671 vs 0.675 | **no** (tie) |

## What this establishes

- Knowledge objects mined at one site transfer: a rule's source confidence predicts its precision at sites it has
  never seen (rank correlation 0.67), and the most trusted decile still hits 91% precision.
- The trust formula in the notes is the weak part. Support (how often a rule fired) and source reliability carry no
  information about transfer here; adding them dilutes confidence, and the result depends heavily on the
  arbitrary support scale. Fitting the weights on source data drives both to zero.
- Recommendation for the design: trust = calibrated confidence (optionally shrunk toward hold-out quality), with
  support used only as a minimum-evidence filter; and report ~6 points of over-confidence correction.
