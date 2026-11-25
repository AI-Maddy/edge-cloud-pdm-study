# C7c Many sites sharing a cellular uplink: packet-level emulation (review follow-up E2)

Why: C7 counted delivery opportunities in each recorded trace for one upload at a time. The review asked for
transport behaviour and contention between sites. Mahimahi cannot be installed on the evaluation machine, so
`run.py` is a discrete-event simulation that keeps Mahimahi's trace semantics (one 1500-byte packet may leave at
each trace millisecond) and adds a shared drop-tail queue (300 packets), one Reno-like flow per site (slow start
from 10 packets, AIMD, fast retransmit, RTO 1 s doubling to 60 s) and a 40 ms base RTT. N = 1, 5 or 35 sites each
send one object per 60 s reporting period, all at once ("sync") or at random offsets ("jitter").
Objects: feature 1,440 B, raw window 62,400 B, model update 107,972 B. 40 random trace offsets per cell.
Run: `python c7c_uplink_sim/run.py` (about 15 minutes on 4 CPU cores). Criteria fixed in `run.py` before running.

**Result: 1 of 4 tests pass.**

| Test | Criterion | Result | Pass |
| --- | --- | --- | --- |
| T1 35 sites in sync, feature uploads | p95 <= 5 s on the 4 traces without long outages | 0.14 / 5.13 / 0.75 / 0.98 s | no (Verizon LTE just over) |
| T2 raw vs feature, 35 sites in sync, 2 LTE traces | raw p95 >= 5x feature | T-Mobile 17x; Verizon 4.6x | no (Verizon just under) |
| T3 random start offsets, model updates, 35 sites | p95 cut >= 50% on >= 3 of 4 traces | -96%, -83%, -64%, -35% | yes (3 of 4) |
| T4 AT&T (outages), one site, feature upload | p95 >= 10 s | 0.83 s | no (sampling, see below) |

p95 delivery time (s), 35 sites:

| Trace | feature sync | feature jitter | raw sync | raw jitter | model sync | model jitter |
| --- | --- | --- | --- | --- | --- | --- |
| T-Mobile LTE | 0.14 | 0.05 | 2.5 | 0.34 | 9.1 | 0.40 |
| Verizon LTE | 5.1 | 0.22 | 23.8 | 14.1 | 42.0 | 6.9 |
| T-Mobile UMTS | 0.75 | 0.42 | 51.8 | 27.6 | 89.4 | 32.6 |
| Verizon EVDO | 0.98 | 0.34 | 62.9 | 6.9 | 71.3 | 46.4 |
| AT&T LTE (outages) | 25.5 | 19.9 | 50.0 | 50.1 | 77.1 | 81.9 |

**What this says.**
- Contention changes the picture C7 gave for single uploads. With 35 sites reporting at the same moment, raw
  windows and model updates on 3G links take 50-90 s at p95, longer than the 60 s reporting period, so the
  queue never drains. Feature uploads stay under about 1 s on three of the four clean traces.
- Spreading start times over the period is as important as shrinking the payload: it cuts the model-update tail
  by 35-96% and the feature tail on Verizon LTE from 5.1 s to 0.2 s. HFIP should schedule uploads, not only
  shrink them.
- T4 fails as written, but the diagnostic `diag_att_single.py` (2,000 offsets instead of 40, not a test) gives a
  single-site feature p95 of 30 s on AT&T and 7% of uploads over 10 s, matching C7 (35 s). Forty offsets were too
  few to hit the outages often enough; the pre-specified verdict stays "no". With 35 sites the AT&T outages show
  up in every cell (20-82 s).

**Limits.** A simulation with trace-driven capacity, not a real TCP stack or a Mahimahi run; ACKs never lost;
the same trace serves all sites (one shared cell). Results: `results/c7c_results.json`,
`results/diag_single_site_2000.json`.
