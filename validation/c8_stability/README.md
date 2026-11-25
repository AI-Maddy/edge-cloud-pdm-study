# C8: Convergence, stability and multi-objective claims


## A. Measured on real federated training (FedAvg, 35 C-MAPSS sites, 40 rounds, 2 seeds)

Each site holds out 10% of its training engines for validation. Run: `python c8_stability/run.py` (about 30 min).

| Round | 1 | 5 | 10 | 20 | 30 | 40 |
| --- | --- | --- | --- | --- | --- | --- |
| Validation RMSE, seed 0 | 51.3 | 29.7 | 24.1 | 21.5 | 20.3 | 19.6 |
| Validation RMSE, seed 1 | 51.8 | 30.1 | 24.3 | 21.3 | 20.0 | 19.5 |

| Claim in note 13 | Measurement | Pass |
| --- | --- | --- |
| T1 monotonic improvement (no round worsens validation RMSE by > 1%) | 0 such rounds in 78 transitions | yes |
| T2 Lyapunov-type energy non-increasing over the last 20 rounds (repo `stability()`, tolerance 5% of initial energy) | passes with that tolerance; but the raw update energy is not monotone (seed 1 rises from 0.015 to 0.042) | yes, only under the tolerance |
| T3 federation stability: client drift halves from first to last 10 rounds | 0.14 -> 0.10 and 0.16 -> 0.15 | **no** |

## B. Clustering convergence under real drift (from C1 and C4)

Note 13 §2 assumes cell assignments converge. On the Beijing network, cells refitted on a trailing year change for
0.37-0.81 of station-days depending on the policy (C4), and year-1 cells agree with later refits at ARI 0.59-0.72
(C1). With a drifting context, cell membership does not converge; it tracks the environment.

## C. Multi-objective view (note 14): what each design choice buys, measured

| Choice | Accuracy effect | Communication effect | Source |
| --- | --- | --- | --- |
| Federate vs isolated sites | -1.6 RMSE (-7%) | + one model up and down per round | C3 |
| Context cells vs IFCA | +0.13 RMSE (n.s.) | 1/4 of IFCA's downlink | C3 |
| Context cells vs global model (drifting data) | -2.5% error | cell id + occasional re-clustering | C4 |
| Distribution-triggered vs per-node re-clustering | equal or better | 12% vs 92% of days re-clustered; half the churn | C4 |
| Feature objects vs raw upload | +0.8% error | 43x fewer bytes | C7 |
| Adopt received model at cold start | -50 to -65% error | one model download | C6 |

## What this establishes

- On real federated training, quality improved round by round without regressions, so the monotonic-improvement
  observation holds here (an observation, not the theorem in note 13).
- Client drift does not vanish: sites stay heterogeneous, so "federation stability" in the sense of consensus is not
  reached in 40 rounds. The energy criterion passes only with a tolerance; it should be reported as a measurement.
- Cell membership does not converge when the context drifts; convergence claims should be stated per fixed context.
- The multi-objective trade-offs are now measured, not assumed; the largest wins are feature-level upload (bytes) and
  downstream knowledge at cold start (accuracy).
