# C7b Leave one fleet out: feature uploads and cold start (review follow-ups E1 + E5)

Why: the review noted two circular setups. C7's feature-upload test used an encoder trained on the same fleet's
raw data, and C6's "unseen" cold-start sites were random parts of fleets the cloud had already trained on. Here
each C-MAPSS fleet is held out in turn; the cloud's models are trained only on the other three fleets of the
35-site build (shared per-condition normalisation). Run: `THREADS=2 python c7b_lofo/run.py` (about 40 minutes,
CPU). Criteria fixed in `run.py` before running.

**Result: 4 of 4 tests pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 features from an other-fleet encoder vs raw upload | within 10% on >= 3 of 4 fleets | -7.4%, -0.9%, +1.8%, +0.3% (all 4) | yes |
| T2 feature objects smaller than raw at equal precision | >= 10x (float32) | 23x (float32), 43x (float16) | yes |
| T3 fine-tuning the other-fleet cloud model vs local only | >= 20% better at N = 1 and 3 | 56% (N=1), 45% (N=3), 37% (N=5) | yes |
| T4 other-fleet cloud model as received vs local at N = 1 | better on >= 3 of 4 fleets | 4 of 4 (14.4-18.8 vs 42-51) | yes |

Feature uploads (test RMSE, Transformer; the cloud never sees raw data from the held-out fleet in `feat_lofo`):

| Held-out fleet | raw upload | feat_same (C7, circular) | feat_lofo | zero-shot |
| --- | --- | --- | --- | --- |
| FD001 | 13.98 | 14.08 | 12.95 | 13.63 |
| FD002 | 13.86 | 13.85 | 13.73 | 13.88 |
| FD003 | 12.70 | 12.67 | 12.92 | 16.06 |
| FD004 | 14.79 | 14.68 | 14.84 | 15.52 |

Cold start (CNN1D, mean RMSE over the held-out fleet's sites):

| Fleet | N | local | cloud as received | fine-tune | output average |
| --- | --- | --- | --- | --- | --- |
| FD001 | 1 / 3 / 5 | 47.2 / 39.4 / 23.3 | 14.4 | 18.2 / 21.6 / 16.0 | 25.9 / 23.2 / 16.5 |
| FD002 | 1 / 3 / 5 | 49.0 / 37.3 / 32.1 | 15.5 | 24.2 / 22.6 / 21.7 | 27.9 / 22.5 / 20.5 |
| FD003 | 1 / 3 / 5 | 42.1 / 38.8 / 33.9 | 16.2 | 18.8 / 19.7 / 18.7 | 22.4 / 22.3 / 21.5 |
| FD004 | 1 / 3 / 5 | 50.6 / 40.5 / 36.5 | 18.8 | 22.0 / 22.0 / 21.7 | 30.3 / 26.2 / 24.3 |

**What this says.**
- The feature-upload result survives without circularity: an encoder that never saw the fleet gives embeddings
  from which the cloud predicts as well as from raw windows (within 2% on every fleet). The honest size ratio is
  23x at float32 (43x at float16, as in C7).
- A new fleet's sites gain most from the cloud model as received: it beats local training by 2.5-3.5x at N = 1.
  Fine-tuning it on 1-5 own failures always makes it worse (by 2-9 RMSE), and averaging outputs with a local model
  is worse still. The design should ship the cloud model and not adapt it until a site has many more failures.
  This replaces C6's "fine-tune the received model" rule.

Results: `results/c7b_results.json`; log: `results/log.txt`.
