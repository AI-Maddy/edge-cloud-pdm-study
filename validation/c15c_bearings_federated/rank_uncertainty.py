"""C15c diagnostic for review 2, point 5 (not a new test): bearing-level uncertainty of the rule-ranking result.
Rules share data (each source bearing contributes many rules; each target bearing scores all of them), so a
rule-level Spearman overstates precision. Two-way cluster bootstrap: resample source bearings and target bearings
with replacement, recompute every rule's mean precision over the resampled targets (excluding its own source), and
the Spearman of confidence vs that precision; also the top-decile claimed vs achieved precision."""
import json, os
import numpy as np
from scipy.stats import spearmanr
import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c15c", os.path.join(HERE, "run.py")); c15c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c15c)
sites, feats = c15c.data(); B = list(sites)
rules = []
for b, (X, y) in sites.items():
    pos = y <= 0.2
    for j, f in enumerate(feats):
        for q in (50, 70, 90):
            th = np.percentile(X[:, j], q); fire = X[:, j] > th
            if fire.sum() < 5: continue
            prec = {}
            for b2, (X2, y2) in sites.items():
                if b2 == b: continue
                f_ = X2[:, j] > th
                if f_.sum(): prec[b2] = float((y2[f_] <= 0.2).mean())
            if prec: rules.append((b, float(pos[fire].mean()), prec))
rng = np.random.default_rng(0); rho, top_c, top_a = [], [], []
def stat(src, tgt):
    C, T = [], []
    for b, conf, prec in rules:
        k = src.count(b)
        if not k: continue
        p = [prec[t] for t in tgt if t in prec]
        if not p: continue
        C += [conf] * k; T += [float(np.mean(p))] * k
    C, T = np.array(C), np.array(T); o = np.argsort(-C)[:max(1, len(C) // 10)]
    return spearmanr(C, T).correlation, C[o].mean(), T[o].mean()
full = stat(B, B)
for _ in range(1000):
    r, c, a = stat(list(rng.choice(B, len(B))), list(rng.choice(B, len(B))))
    if np.isfinite(r): rho.append(r); top_c.append(c); top_a.append(a)
out = {"n_rules": len(rules), "spearman": float(full[0]), "spearman_ci95_bearing_bootstrap": np.percentile(rho, [2.5, 97.5]).tolist(),
       "top_decile_claimed": float(full[1]), "top_decile_achieved": float(full[2]),
       "top_decile_achieved_ci95": np.percentile(top_a, [2.5, 97.5]).tolist()}
json.dump(out, open(os.path.join(HERE, "results", "rank_uncertainty.json"), "w"), indent=1); print(out)
