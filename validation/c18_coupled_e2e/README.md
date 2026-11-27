# C18 Flat vs regional federation coupled to uplink delivery with deadlines (review-2 follow-up E7)

Why: the second review's main point was that learning (C3b) and delivery (C7c) had been measured separately.
Here each federated round is one 60 s reporting period, and a site's model update (CNN1D, 336,516 B) counts only
if the C7c packet-level simulation delivers it over the site's cell uplink before the deadline. 35 C-MAPSS sites,
5 cells of 7 sites (the C3b random regions), one recorded trace per cell; 100 periods, 3 seeds, same trace offsets
and start times for every method within a seed. Methods: flat FedAvg, regional (HierFAVG-style: region every
period, cloud every 5 periods over a wired WAN), flat/5 (flat FedAvg communicating every 5 periods, same local
work). Upload starts: sync or jitter (first 30 s). Stress: 10 s deadline, jitter in the first 5 s (`results/d10/`).
Criteria fixed in `run.py` before running (T5/T6 added before the stress runs, while the 60 s runs were going).

Run: `python c18_coupled_e2e/precompute.py <seed>` for seeds 0-2 (delivery tables, ~5 min each), then
`JOBS=0:flat_jit,... OUT=runs_a.json python c18_coupled_e2e/run.py` (each run ~13 min on one core; for the stress
setting add `DEADLINE=10 JIT=5 TAG=d10`), then `python c18_coupled_e2e/summarize.py`.

**Result: 4 of 6 tests pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 regional vs flat, 60 s deadline, jitter | within 0.5 RMSE, <= 10% WAN bytes | -0.05 (CI [-0.19, 0.10], p = 0.45), 2.8% of WAN bytes | yes |
| T2 jitter halves flat's deadline misses | <= 0.5x | 2.8% -> 2.0% | no |
| T3 flat with deadlines vs deadline-free FedAvg (C3b) | within 0.5 | 16.34 vs 16.35 | yes |
| T4 regional beats flat/5 (same cloud period) | <= -0.5 | -0.67 (CI [-0.97, -0.37], p = 0.0002) | yes |
| T5 10 s deadline: regional vs flat | within 0.5 | -0.05 | yes |
| T6 10 s deadline costs flat >= 0.5 RMSE | >= 0.5 | +0.36 with 54% of updates missed | no |

| Method | RMSE @25 | @50 | @75 | @100 periods | missed | access up | access down | WAN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat sync | 19.64 | 17.66 | 16.69 | 16.37 | 2.8% | 1.18 GB | 1.18 GB | 2.36 GB |
| flat jitter | 19.64 | 17.65 | 16.68 | 16.34 | 2.0% | 1.18 GB | 1.18 GB | 2.36 GB |
| regional sync | 19.62 | 17.62 | 16.79 | 16.28 | 2.8% | 1.18 GB | 1.18 GB | 67 MB |
| regional jitter | 19.58 | 17.59 | 16.71 | 16.29 | 2.0% | 1.18 GB | 1.18 GB | 67 MB |
| flat/5 sync | 19.95 | 18.09 | 17.19 | 17.09 | 2.1% | 0.24 GB | 0.24 GB | 471 MB |
| flat/5 jitter | 19.96 | 18.02 | 17.17 | 16.96 | 1.6% | 0.24 GB | 0.24 GB | 471 MB |
| flat, 10 s deadline | 19.63 | 17.90 | 17.02 | 16.71 | 53.8% | 1.18 GB | 1.18 GB | 2.36 GB |
| regional, 10 s deadline | 20.23 | 18.10 | 16.98 | 16.66 | 53.8% | 1.18 GB | 1.18 GB | 67 MB |

Missed updates per cell (every-period uploads): 60 s deadline 0/0/3.0/4.1/6.8% (sync) for T-Mobile LTE / Verizon LTE /
T-Mobile UMTS / Verizon EVDO / AT&T; 10 s deadline 2/19/100/100/100% (sync), 1/17/82/85/84% (jitter).

**What this says.**
- Coupled to delivery, the regional tier keeps flat FedAvg's accuracy along the whole learning curve with 2.8% of the
  WAN bytes. The access link carries the same traffic either way: regions save backhaul, not radio capacity.
- The benefit is not just "communicate less": flat FedAvg that talks to the cloud every 5 periods is 0.67 worse.
- At 7 sites per cell, a 60 s deadline is rarely missed (2-3%), so jitter has little to fix (T2 fails).
- With a 10 s deadline the three slow cells contribute almost nothing, yet accuracy drops only 0.36: every cell here
  holds sites of all four fleets, so the fast cells still see every fleet. A slow cell that holds a whole fleet
  would lose it; this placement does not test that.

**Limits.** Simulated transport (C7c), downloads not simulated, wired WAN assumed on time, sites randomly placed.
Results: `results/summary.json`, `results/runs_s*.json`, `results/d10/`.
