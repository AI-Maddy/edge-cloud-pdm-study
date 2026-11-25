# C7: HFIP message accounting over recorded cellular uplinks


One edge site = 20 turbofan engines; one reporting period = 30 operating cycles. Objects are built and sized with
the repository serializer (JSON as implemented) and with a compact binary encoding. Delivery time is replayed over
5 recorded driving uplink traces (Mahimahi), 200 random start offsets each.
Run: `python c7_hfip_traffic/run.py` (about 2 min).

## Bytes per site per period

| Upstream strategy | JSON (repo serializer) | Binary | Cloud RUL error (FD001 test RMSE) |
| --- | --- | --- | --- |
| Raw observations (24 values per engine-cycle) | 171,930 | 62,400 | 13.72 (full edge model) |
| Feature objects (32-d embedding per engine) | 7,401 | 1,440 | **13.83** (ridge head on embeddings) |
| Inference objects (RUL estimate per engine) | 3,260 | 280 | 13.72 (same as edge) |
| Model update (edge Transformer, 26,977 params) | - | 107,972 | - |
| Knowledge objects (20 rules) | 3,520 | 480 | - |

## 95th-percentile delivery time (seconds)

| Trace (mean uplink) | Raw (binary) | Features (binary) | Model update |
| --- | --- | --- | --- |
| T-Mobile LTE (19.3 Mb/s) | 0.2 | 0.01 | 0.4 |
| Verizon LTE (6.2 Mb/s) | 0.9 | 0.07 | 3.6 |
| T-Mobile UMTS (0.9 Mb/s) | 2.4 | 1.6 | 5.4 |
| Verizon EVDO (0.8 Mb/s) | 3.5 | 0.3 | 5.4 |
| AT&T LTE (0.8 Mb/s, long outages) | 35.1 | 50.2 | 45.4 |

| Test | Result | Pass |
| --- | --- | --- |
| T1 feature objects cut upstream bytes >= 10x vs raw | 43x (binary), 23x (JSON) | yes |
| T2 cloud error from features within 10% of raw | 13.83 vs 13.72 (+0.8%) | yes |
| T3 model update within 10 s at p95 on every trace | 4 of 5 traces; AT&T 45 s | **no** |

## What this establishes

- Feature-level federation (note 01) holds on real data: 43x fewer bytes than raw observations with <1% loss in
  cloud-side RUL accuracy. Inference objects are another 5x smaller.
- The repository's JSON serialisation costs 3-12x over a binary encoding; HFIP needs a compact wire format.
- On a trace with connectivity gaps (AT&T), delivery time is set by outages, not payload size: even a 1.4 KB
  feature upload has a 50 s tail. The protocol needs store-and-forward and deadline-aware retry, not only smaller
  payloads.
