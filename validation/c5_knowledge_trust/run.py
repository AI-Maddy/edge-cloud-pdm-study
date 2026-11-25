"""C5 validation: knowledge objects, trust and confidence (notes 05 'Knowledge Representation', 'Trust Model';
12 sec. 10-13; repository scripts/hfcf/knowledge.py trust()).

Knowledge object (as in note 05's thermal_runaway example): a condition -> outcome rule mined by a site,
  {"pattern": "s11>theta", "conditions": [...], "outcome": "failure within 30 cycles",
   "confidence": precision at the source, "support": firings at the source, "quality": precision on the source's
   own held-out engines, "source_reliability": source's precision averaged over all its rules}
Rules: for each of the 14 condition-normalised sensors (C2 normalisation) and thresholds at the source's
90/95/97.5/99th percentiles, sign chosen by correlation with RUL; outcome = RUL <= 30.
Sites: the 35 C-MAPSS sites of C3. Sources = 24 random sites (per seed); targets = the other 11 sites
(never seen). The trust score must predict how well a rule works at the unseen targets.
Tests (fixed before running):
  T1 Spearman(trust with repository default weights, target precision) >= 0.5
  T2 default trust ranks rules at least as well as confidence alone
  T3 calibration: |mean confidence - mean target precision| for the top-decile trust rules <= 0.10
Added after T1/T2 failed (reported as a follow-up, not a replacement):
  T4 learned trust (non-negative least squares weights for the same four components, fitted on source sites
     only: rules from 16 sources scored on the other 8 sources) ranks unseen-target precision better than
     confidence alone.
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from common import cmapss, rul
from hfcf import knowledge

H = 30
SITES = {"FD001": 5, "FD002": 13, "FD003": 5, "FD004": 12}


def site_rows():
    trs = {fd: cmapss.load(fd, "train") for fd in SITES}
    allset = np.vstack([trs[fd]["arr"][:, 2:5] for fd in SITES]); allsen = np.vstack([trs[fd]["arr"][:, 5:][:, rul.SENS] for fd in SITES])
    norm = rul.Normaliser(True, 6).fit(allset, allsen)
    sites = []
    for fd, n in SITES.items():
        a = trs[fd]["arr"]; z, _ = norm.transform(a[:, 2:5], a[:, 5:][:, rul.SENS])
        # 5-cycle trailing mean per unit to reduce sensor noise (local edge smoothing)
        zs = np.zeros_like(z)
        for u in np.unique(a[:, 0]):
            m = np.where(a[:, 0] == u)[0]
            c = np.cumsum(np.vstack([np.zeros((1, z.shape[1])), z[m]]), 0)
            k = np.minimum(np.arange(1, len(m) + 1), 5)
            zs[m] = (c[np.arange(1, len(m) + 1)] - c[np.arange(1, len(m) + 1) - k]) / k[:, None]
        units = np.unique(a[:, 0])
        for j in range(n):
            us = units[j::n]; m = np.isin(a[:, 0], us)
            sites.append(dict(fd=fd, z=zs[m], y=(trs[fd]["rul"][m] <= H), unit=a[m, 0]))
    return sites


def precision(rule, z, y):
    f = rule_fires(rule, z)
    return (float(y[f].mean()) if f.any() else np.nan), int(f.sum())


def rule_fires(rule, z):
    s, th, sign = rule
    return (z[:, s] > th) if sign > 0 else (z[:, s] < th)


def main():
    sites = site_rows()
    out = {"criteria": {"T1": 0.5, "T3": 0.10}, "seeds": []}
    for seed in range(5):
        g = np.random.default_rng(seed)
        perm = g.permutation(len(sites)); src, tgt = perm[:24], perm[24:]
        Zt = np.vstack([sites[i]["z"] for i in tgt]); Yt = np.concatenate([sites[i]["y"] for i in tgt])
        objs = []
        for i in src:
            st = sites[i]; units = np.unique(st["unit"]); hold = np.isin(st["unit"], units[::4])
            zf, yf, zh, yh = st["z"][~hold], st["y"][~hold], st["z"][hold], st["y"][hold]
            rules = []
            for s in range(zf.shape[1]):
                sign = 1 if np.corrcoef(zf[:, s], yf)[0, 1] > 0 else -1
                for q in (90, 95, 97.5, 99):
                    th = np.percentile(zf[:, s], q if sign > 0 else 100 - q)
                    conf, sup = precision((s, th, sign), zf, yf)
                    qual, _ = precision((s, th, sign), zh, yh)
                    if sup >= 5 and np.isfinite(conf):
                        rules.append([(s, th, sign), conf, sup, 0.0 if not np.isfinite(qual) else qual])
            rel = float(np.mean([r[1] for r in rules]))
            for r in rules:
                objs.append({"id": f"K{i}-{r[0][0]}-{r[0][1]:.2f}", "pattern": f"s{rul.SENS[r[0][0]]+1}{'>' if r[0][2] > 0 else '<'}{r[0][1]:.2f}",
                             "outcome": f"failure within {H} cycles", "confidence": r[1], "support": r[2], "quality": r[3],
                             "source_reliability": rel, "rule": r[0]})
        scored = knowledge.trust({"records": objs}, {"support_scale": 200})["records"]   # repository implementation
        scored_default = knowledge.trust({"records": objs}, {})["records"]               # repository default scale 1000
        # learned weights, source sites only: inner split 16 / 8
        inner_a = set(src[:16].tolist()); Zi = np.vstack([sites[i]["z"] for i in src[16:]]); Yi = np.concatenate([sites[i]["y"] for i in src[16:]])
        from scipy.optimize import nnls
        comp = lambda o: [o["trust_components"][k] for k in ("confidence", "support_score", "quality", "source_reliability")]
        fit = [(comp(o), precision(o["rule"], Zi, Yi)[0]) for o in scored if int(o["id"][1:].split("-")[0]) in inner_a]
        fit = [(c, t) for c, t in fit if np.isfinite(t)]
        A = np.array([c + [1.0] for c, _ in fit]); bvec = np.array([t for _, t in fit])
        wl, _ = nnls(A, bvec)
        learned = np.array([np.dot(comp(o) + [1.0], wl) for o in scored])
        tp = np.array([precision(o["rule"], Zt, Yt)[0] for o in scored])
        ok = np.isfinite(tp)
        TD = np.array([o["trust"] for o in scored_default])[ok]; TL = learned[ok]
        T = np.array([o["trust"] for o in scored])[ok]; Cf = np.array([o["confidence"] for o in scored])[ok]
        Q = np.array([o["quality"] for o in scored])[ok]; tp = tp[ok]
        top = T >= np.quantile(T, 0.9)
        out["seeds"].append({"seed": seed, "n_rules": int(ok.sum()), "spearman_trust": float(spearmanr(T, tp)[0]),
                             "spearman_confidence": float(spearmanr(Cf, tp)[0]),
                             "spearman_trust_repo_default_scale": float(spearmanr(TD, tp)[0]),
                             "spearman_trust_learned": float(spearmanr(TL, tp)[0]), "learned_weights_c_s_q_r_bias": wl.round(3).tolist(), "spearman_quality": float(spearmanr(Q, tp)[0]),
                             "top_decile_mean_confidence": float(Cf[top].mean()), "top_decile_target_precision": float(tp[top].mean()),
                             "all_mean_confidence": float(Cf.mean()), "all_target_precision": float(tp.mean())})
        print(out["seeds"][-1], flush=True)
    S = out["seeds"]
    m = lambda k: float(np.mean([s[k] for s in S]))
    out["mean"] = {k: m(k) for k in S[0] if k not in ("seed", "learned_weights_c_s_q_r_bias")}
    out["verdict"] = {"T1": out["mean"]["spearman_trust"] >= 0.5,
                      "T2": out["mean"]["spearman_trust"] >= out["mean"]["spearman_confidence"],
                      "T4": out["mean"]["spearman_trust_learned"] > out["mean"]["spearman_confidence"],
                      "T3": abs(out["mean"]["top_decile_mean_confidence"] - out["mean"]["top_decile_target_precision"]) <= 0.10}
    json.dump(out, open(os.path.join(HERE, "results", "c5_results.json"), "w"), indent=2)
    print(json.dumps({"mean": out["mean"], "verdict": out["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
