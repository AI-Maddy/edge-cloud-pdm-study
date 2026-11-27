"""C7c diagnostics for review 2, point 2 (not new tests; C7c's pre-specified verdicts are unchanged).

(a) Multi-period backlog. 35 sites send one object in each of 10 consecutive 60 s periods on one trace (all 350
    flows in one simulation, so late uploads keep competing with the next period's). Reported per trace/object/
    mode: share of uploads that take longer than one period from their own start (deadline miss; for sync this is
    finishing after the next reporting boundary), share unfinished when
    the simulation is cut at 900 s (censored; counted as 900 s in percentiles, as in C7c), flows still active at
    each period boundary (backlog), and the median delivery time in periods 1-2 vs 9-10 (does the queue drain?).
    10 random trace offsets each.
(b) Offset-sampling protocol. Single-site feature upload, 2,000 uniformly random offsets per trace, p95 with a
    bootstrap 95% CI; also the p95 from 40 offsets (C7c's protocol) for 200 independent draws, to show its spread.
"""
import importlib.util, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c7c", os.path.join(HERE, "run.py")); c7c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c7c)
K, N = 10, 35


def multiperiod(tr, g, nbytes, mode):
    st = np.concatenate([k * c7c.PERIOD + (np.zeros(N) if mode == "sync" else g.uniform(0, c7c.PERIOD, N)) for k in range(K)])
    per = np.repeat(np.arange(K), N)
    done = c7c.simulate(tr, g.uniform(0, tr[-1]), st, nbytes)
    fin = st + done
    miss = done > c7c.PERIOD   # more than one period after its own start (= after the next boundary for sync)
    cens = done >= c7c.CAP_S - 1e-9
    backlog = [int(((st < b) & (fin > b)).sum()) for b in (np.arange(1, K + 1) * c7c.PERIOD)]
    return done, per, miss, cens, backlog


def main():
    out = {"multiperiod": [], "single_site": {}}
    g = np.random.default_rng(7)
    for tn in c7c.TRACES:
        tr = c7c.load(tn)
        for obj in ("feature", "raw", "model"):
            for mode in ("sync", "jitter"):
                D, Mi, Ce, B, early, late = [], [], [], [], [], []
                for rep in range(10):
                    done, per, miss, cens, backlog = multiperiod(tr, g, c7c.OBJ[obj], mode)
                    D += list(done); Mi += list(miss); Ce += list(cens); B.append(backlog)
                    early += list(done[per < 2]); late += list(done[per >= K - 2])
                c = {"trace": tn, "object": obj, "mode": mode, "p95_s": float(np.percentile(D, 95)),
                     "miss_rate": float(np.mean(Mi)), "censored_rate": float(np.mean(Ce)),
                     "backlog_at_boundaries_mean": np.mean(B, 0).round(1).tolist(),
                     "median_first2_s": float(np.median(early)), "median_last2_s": float(np.median(late))}
                out["multiperiod"].append(c); print(c, flush=True)
        t = np.array([c7c.simulate(tr, g.uniform(0, tr[-1]), np.zeros(1), c7c.OBJ["feature"])[0] for _ in range(2000)])
        bt = [np.percentile(g.choice(t, len(t)), 95) for _ in range(2000)]
        p40 = [np.percentile(g.choice(t, 40, replace=False), 95) for _ in range(200)]
        out["single_site"][tn] = {"p95_s": float(np.percentile(t, 95)), "p95_ci95": np.percentile(bt, [2.5, 97.5]).tolist(),
                                  "p95_from_40_offsets_range_5_95": np.percentile(p40, [5, 95]).tolist(),
                                  "share_over_10s": float((t > 10).mean())}
        print(tn, out["single_site"][tn], flush=True)
        json.dump(out, open(os.path.join(HERE, "results", "diag_multiperiod.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
