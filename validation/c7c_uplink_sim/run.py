"""C7c (review follow-up E2): trace-driven packet-level simulation of HFIP uploads with many sites sharing an uplink.

C7 counted 1500-byte delivery opportunities in each recorded trace for a single upload. The review asked for
transport behaviour and contention. Mahimahi itself cannot be installed on the evaluation machine, so this is a
discrete-event simulation that uses Mahimahi's trace semantics for the bottleneck (each trace line = one
1500-byte packet may leave at that millisecond) and adds:
  - a shared drop-tail bottleneck queue (QCAP packets) in front of the trace,
  - one TCP-Reno-like flow per site: slow start from 10 packets, AIMD, fast retransmit one RTT after a drop,
    retransmission timeout (1 s initial, doubling to 60 s) with go-back to the oldest unacknowledged packet,
  - a fixed 40 ms base RTT on top of queueing; ACKs are never lost,
  - N sites (1, 5, 35) uploading one object each in the same reporting period, either all at the start of the
    period ("sync") or at independent uniform offsets within a 60 s period ("jitter").
Objects (binary sizes from C7, per site per period): feature 1,440 B, raw 62,400 B, model update 107,972 B.
Metric: time from a site's own start to its last byte acknowledged; p95 over sites x 40 random trace offsets.

Tests (fixed before running):
  T1 with 35 sites in sync, feature uploads finish within 5 s at p95 on the four traces without long outages
  T2 with 35 sites in sync on the two LTE traces without long outages, raw uploads take >= 5x longer than
     feature uploads at p95 (contention grows with payload size)
  T3 jittering starts over the period cuts the p95 of model-update uploads with 35 sites by >= 50% on at least
     3 of the 4 traces without long outages
  T4 on the AT&T trace (long outages) a single site's feature upload still has p95 >= 10 s (outages, not size,
     set the tail), confirming C7
"""
import heapq, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "..", "data", "traces")
TRACES = ["TMobile-LTE-driving", "Verizon-LTE-driving", "TMobile-UMTS-driving", "Verizon-EVDO-driving", "ATT-LTE-driving"]
CLEAN = TRACES[:4]
OBJ = {"feature": 1440, "raw": 62400, "model": 107972}
PKT, RTT, QCAP, PERIOD, CAP_S = 1500, 0.040, 300, 60.0, 900.0


def load(name):
    import gzip
    with gzip.open(os.path.join(TR, name + ".up.gz"), "rt") as f:
        return np.array([int(l) for l in f if l.strip()], float) / 1000.0


def simulate(trace, t0, starts, nbytes):
    """One period: flows i start at starts[i] (seconds after t0). Returns completion time per flow."""
    T = trace[-1]
    n = len(starts); npk = int(np.ceil(nbytes / PKT))
    cwnd = np.full(n, 10.0); ssth = np.full(n, 1e9)
    nxt = np.zeros(n, int); acked = [np.zeros(npk, bool) for _ in range(n)]; una = np.zeros(n, int)
    inflight = np.zeros(n, int); rto = np.ones(n); last = np.zeros(n); done = np.full(n, np.nan)
    retx = [[] for _ in range(n)]; started = np.zeros(n, bool)
    q = []  # FIFO of (flow, seq)
    qh = 0
    ev = [(s, 0, i, -1) for i, s in enumerate(starts)]  # (time, kind, flow, seq): 0 start, 1 ack, 2 loss, 3 rto
    heapq.heapify(ev)
    k = np.searchsorted(trace, t0 % T); loops = 0
    def opp(k, loops):
        return trace[k] + loops * T - t0
    now = 0.0

    def pump(i, t):
        nonlocal q
        while inflight[i] < int(cwnd[i]) and (retx[i] or nxt[i] < npk):
            if retx[i]:
                s = retx[i].pop(0)
                if acked[i][s]:
                    continue
            else:
                s = nxt[i]; nxt[i] += 1
            inflight[i] += 1
            if len(q) - qh >= QCAP:
                heapq.heappush(ev, (t + RTT, 2, i, s))      # drop; sender notices one RTT later
            else:
                q.append((i, s))
        if inflight[i] > 0:
            heapq.heappush(ev, (t + rto[i], 3, i, int(last[i] * 1e6)))

    while np.isnan(done).any():
        t_opp = opp(k, loops)
        if ev and ev[0][0] <= t_opp:
            t, kind, i, s = heapq.heappop(ev); now = t
            if not np.isnan(done[i]):
                continue
            if kind == 0:
                started[i] = True; last[i] = t; pump(i, t)
            elif kind == 1:
                if not acked[i][s]:
                    acked[i][s] = True
                    cwnd[i] += 1.0 if cwnd[i] < ssth[i] else 1.0 / cwnd[i]
                    while una[i] < npk and acked[i][una[i]]:
                        una[i] += 1
                inflight[i] = max(0, inflight[i] - 1); last[i] = t; rto[i] = 1.0
                if una[i] >= npk:
                    done[i] = t - starts[i]
                else:
                    pump(i, t)
            elif kind == 2:
                inflight[i] = max(0, inflight[i] - 1)
                ssth[i] = max(cwnd[i] / 2, 2); cwnd[i] = ssth[i]; retx[i].append(s); pump(i, t)
            elif kind == 3:
                if int(last[i] * 1e6) == s and una[i] < npk:   # no progress since this timer was armed
                    ssth[i] = max(cwnd[i] / 2, 2); cwnd[i] = 1.0; rto[i] = min(rto[i] * 2, 60.0)
                    inflight[i] = 0; retx[i] = [x for x in range(una[i], nxt[i]) if not acked[i][x]]
                    last[i] = t; pump(i, t)
        else:
            now = t_opp
            if qh < len(q):
                i, s = q[qh]; qh += 1
                heapq.heappush(ev, (now + RTT, 1, i, s))
            k += 1
            if k >= len(trace):
                k = 0; loops += 1
            if now > CAP_S:
                done[np.isnan(done)] = CAP_S; break
    return done


def main():
    g = np.random.default_rng(0)
    res = {"params": {"pkt": PKT, "rtt_s": RTT, "qcap_pkts": QCAP, "period_s": PERIOD, "objects": OBJ}, "cells": []}
    for tn in TRACES:
        tr = load(tn)
        for N in (1, 5, 35):
            for mode in (("sync",) if N == 1 else ("sync", "jitter")):
                for obj, nb in OBJ.items():
                    times = []
                    for rep in range(40):
                        t0 = g.uniform(0, tr[-1])
                        st = np.zeros(N) if mode == "sync" else g.uniform(0, PERIOD, N)
                        times += list(simulate(tr, t0, st, nb))
                    c = {"trace": tn, "N": N, "mode": mode, "object": obj, "median_s": float(np.median(times)),
                         "p95_s": float(np.percentile(times, 95))}
                    res["cells"].append(c); print(c, flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "c7c_results.json"), "w"), indent=1)
    P = {(c["trace"], c["N"], c["mode"], c["object"]): c["p95_s"] for c in res["cells"]}
    res["verdict"] = {
        "T1": all(P[(t, 35, "sync", "feature")] <= 5 for t in CLEAN),
        "T2": all(P[(t, 35, "sync", "raw")] >= 5 * P[(t, 35, "sync", "feature")] for t in CLEAN[:2]),
        "T3": sum(P[(t, 35, "jitter", "model")] <= 0.5 * P[(t, 35, "sync", "model")] for t in CLEAN) >= 3,
        "T4": P[("ATT-LTE-driving", 1, "sync", "feature")] >= 10}
    json.dump(res, open(os.path.join(HERE, "results", "c7c_results.json"), "w"), indent=1)
    print(res["verdict"])


if __name__ == "__main__":
    main()
