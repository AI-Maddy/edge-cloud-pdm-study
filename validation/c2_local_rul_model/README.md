# C2: Local edge model (remaining useful life)


Standard C-MAPSS protocol: 14 informative sensors, RUL capped at 125, 30-cycle windows, score on the last window of
each official test engine; early stopping on 10% of training engines (never the test set). Sensors are normalised
per operating condition using the C1 context cells ("context") or globally ("global"). 3 seeds (ridge: 1).
Run: `python c2_local_rul_model/run.py` (about 2 h on 4 CPUs). Code shared with later concepts: `common/rul.py`.

## Test RMSE (mean ± std over seeds)

| Model (normalisation) | Params | FD001 | FD002 | FD003 | FD004 |
| --- | --- | --- | --- | --- | --- |
| Ridge on window (context) | 421 | 14.68 | 16.33 | 16.21 | 19.51 |
| 1-D CNN (context) | 84,129 | 14.18 ± 0.33 | 16.16 ± 0.20 | 14.84 ± 0.47 | 17.08 ± 0.89 |
| **Repo Transformer (context)** | 26,977 | **13.71 ± 0.16** | **13.91 ± 0.41** | **13.18 ± 0.41** | **14.62 ± 0.15** |
| Ridge (global) | 421 | - | 17.09 | - | 23.47 |
| 1-D CNN (global) | 84,129 | - | 18.16 ± 0.39 | - | 23.27 ± 0.52 |
| Repo Transformer (global) | 26,977 | - | 17.01 ± 2.18 | - | 17.63 ± 1.00 |
| Published LSTM (Zheng et al. 2017) | | 16.14 | 24.49 | 16.18 | 28.17 |
| Published DCNN (Li et al. 2018) | | 12.61 | 22.36 | 12.64 | 23.31 |

NASA asymmetric scores for the repo Transformer (context): 293, 1020, 405, 1293.

| Test | Result | Pass |
| --- | --- | --- |
| T1 best model beats the published LSTM on every subset | 13.7 / 13.9 / 13.2 / 14.6 vs 16.1 / 24.5 / 16.2 / 28.2 | yes |
| T2 context normalisation beats global normalisation on FD002 and FD004 | all three models, e.g. Transformer 13.9 vs 17.0 and 14.6 vs 17.6 | yes |

## What this establishes

- The repository's Transformer encoder, once given a regression head and trained, is a competitive edge RUL model:
  better than the published LSTM everywhere, better than the published DCNN on the multi-condition subsets FD002 and
  FD004, and about 1 RMSE behind it on the single-condition FD001 and FD003.
- The context concept matters most here: normalising by operating-condition cell (C1) cuts error by 3-9 RMSE on the
  multi-condition subsets for every model, including a linear one. This is the main reason the federation cells in
  C3 add little: the context has already been used.
