# C16 Lifelong learning at one node (note 10 continuous loop, note 14 sec. 10)

**Question.** A node meets four operating regimes one after another (C-MAPSS FD001 -> FD002 -> FD003 ->
FD004; each switch changes the environment or the fault mode). Does it forget earlier regimes, and does
stored knowledge keep the note's constraint ForgetRate < gamma?

**Setup.** Repo Transformer (C2), per-fleet context normaliser, 25 epochs max per stage with early stopping,
3 seeds. Methods: naive sequential training; replay (2,000 stored windows per earlier fleet, ~0.5 MB);
a knowledge library (one model per fleet) with context routing by Gaussian likelihood under each stored
normaliser; the same library with the true fleet id (oracle); joint training on all four (reference).
Run: `python c16_lifelong/run.py` (about 2.5 hours on 4 CPU cores; resumes finished seeds).

**Result: 3 of 4 tests pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 replay keeps forgetting small | mean forgetting <= 1.0 RMSE | 0.54 (naive: 3.67) | yes |
| T2 context routing picks the right stored model | >= 95% engines, library within 5% of oracle | 51%; 15.41 vs 14.08 (+9%) | no |
| T3 best continual method near joint training | within 10% | replay 14.26 vs joint 13.71 (+4.0%) | yes |
| T4 forward transfer | previous model >= scratch on the new fleet | 0.42 RMSE better than scratch | yes |

Mean test RMSE over the four fleets after the last stage: naive 16.40, routed library 15.41, replay 14.26,
oracle library 14.08, joint 13.71.

| Fleet | naive just learned -> final | replay final | joint | routing accuracy |
| --- | --- | --- | --- | --- |
| FD001 | 13.46 -> 19.74 | 14.05 | 12.88 | 5% |
| FD002 | 13.88 -> 17.30 | 14.41 | 13.81 | 86% |
| FD003 | 12.47 -> 13.80 | 13.79 | 13.14 | 57% |
| FD004 | 14.76 (last) | 14.78 | 15.02 | 54% |

**What this says.**
- Forgetting is real and large for the simplest regime: after learning the 6-condition fleets, FD001 error
  rises from 13.5 to 19.7. A small replay buffer removes almost all of it (0.5 RMSE) and ends within 4% of
  training on everything at once, so note 14's ForgetRate < gamma is achievable with a cheap mechanism.
- Starting each stage from the previous model helps the new regime slightly (knowledge does accumulate).
- A model zoo routed by context recognition does not work here: fleets that share operating conditions
  (FD001/FD003, FD002/FD004) differ only in fault mode, which unlabeled sensor statistics do not reveal
  (routing 51%; the simpler squared-z rule, also recorded, is no better). With the true id the library
  would be competitive (14.08), so the weak link is recognising the context, not storing knowledge.
