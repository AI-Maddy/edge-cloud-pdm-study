# C6: Knowledge dissemination and consumption


Cold start: 8 target sites (2 per C-MAPSS subset) that the cloud never saw receive a cloud model trained on the
other 27 sites. Each target owns only N = 1, 3 or 5 run-to-failure engines. 3 seeds x 8 sites x 3 values of N = 72
cases; test RMSE on each target's share of the official test engines. Run: `python c6_knowledge_consumption/run.py`
(about 25 min).

| Consumption policy | N = 1 | N = 3 | N = 5 |
| --- | --- | --- | --- |
| Local only (ignore knowledge) | 46.2 | 40.3 | 32.3 |
| Cloud model as received | **16.0** | **16.0** | **16.0** |
| Original-design merge M + 0.5 (K - M) in weight space | 56.5 | 51.0 | 50.1 |
| Average of local and cloud predictions | 26.5 | 24.3 | 21.3 |
| Adapt: fine-tune the cloud model on local engines | 19.1 | 21.1 | 18.9 |
| Validity check (pick by local hold-out) | 17.5 | 20.1 | 17.7 |

| Test | Result | Pass |
| --- | --- | --- |
| T1 original-design weight merge beats local in >= 75% of cases | 11% of cases | **no** |
| T2 validity check never > 5% worse than local (harm <= 10%) | harm 0% | yes |
| T3 fine-tuning beats local for N <= 3 | 19-21 vs 40-46 | yes |

Validity-check choices: cloud 43 times, fine-tune 17, output blend 7, local 5. It was more than 5% worse than simply
using the cloud model in 25 of 72 cases.

## What this establishes

- Downstream knowledge is worth a lot at a cold-start site: the received model cuts error by 50-65% versus what a
  site can learn from its own 1-5 failures.
- The merge equation in the original design (M + alpha K on parameters) is wrong as stated for independently trained models:
  averaging unrelated weights gives a model worse than either parent in 89% of cases. Parameter merging needs a
  shared lineage (receiver starts from, or fine-tunes, the received model); otherwise merge in output space.
- With so few local failures, any local adaptation or local validation is noisy. A hold-out of one engine picks the
  wrong option a third of the time. The safe consumption rule at cold start is: adopt the received model, and
  switch to adaptation only once the site has enough failures to validate on.
