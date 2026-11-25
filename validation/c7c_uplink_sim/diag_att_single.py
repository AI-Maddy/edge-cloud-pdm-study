"""Diagnostic (not a test): single-site feature upload on every trace with 2,000 start offsets instead of 40,
to check whether the T4 miss comes from sampling too few offsets to hit the AT&T outages."""
import json, os
import numpy as np
import run
g = np.random.default_rng(1); out = {}
for tn in run.TRACES:
    tr = run.load(tn)
    t = [run.simulate(tr, g.uniform(0, tr[-1]), np.zeros(1), run.OBJ["feature"])[0] for _ in range(2000)]
    out[tn] = {"median_s": float(np.median(t)), "p95_s": float(np.percentile(t, 95)), "p99_s": float(np.percentile(t, 99)),
               "share_over_10s": float(np.mean(np.array(t) > 10))}
    print(tn, out[tn], flush=True)
json.dump(out, open(os.path.join(run.HERE, "results", "diag_single_site_2000.json"), "w"), indent=1)
