"""Summary of C18 (60 s runs in results/runs*.json, 10 s stress runs in results/d10/runs*.json) -> results/summary.json."""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
NREG = 5; TR = ["T-Mobile LTE", "Verizon LTE", "T-Mobile UMTS", "Verizon EVDO", "AT&T LTE"]
cell = np.random.default_rng(123).permutation(np.arange(35) % NREG)


def load(pat, tag):
    M = {}
    for f in glob.glob(pat):
        for r in json.load(open(f))["runs"]: M.setdefault(r["method"] + tag, {})[r["seed"]] = r
    return M


M = {**load(os.path.join(RES, "runs*.json"), ""), **load(os.path.join(RES, "d10", "runs*.json"), "_d10")}
A = {m: np.mean([r["site_rmse"] for r in v.values()], 0) for m, v in M.items()}
S = {}
for m, v in M.items():
    g = lambda k: float(np.mean([r[k] for r in v.values()]))
    S[m] = {"rmse": float(A[m].mean()), "curve_25_50_75_100": np.mean([r["curve_every_25"] for r in v.values()], 0).round(3).tolist(),
            "miss_rate": g("miss_rate"), "access_up_GB": g("access_up_bytes") / 1e9, "access_down_GB": g("access_down_bytes") / 1e9,
            "wan_GB": g("wan_bytes") / 1e9, "seeds": sorted(v), "per_cell_rmse": dict(zip(TR, [round(float(A[m][cell == q].mean()), 2) for q in range(NREG)]))}


def cmp(a, b):
    d = A[a] - A[b]; rng = np.random.default_rng(0); bt = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return {"mean_diff": float(d.mean()), "ci95": np.percentile(bt, [2.5, 97.5]).tolist(), "p": float(wilcoxon(A[a], A[b]).pvalue)}


C = {k: cmp(*k.split("__vs__")) for k in ["hier_jit__vs__flat_jit", "hier_sync__vs__flat_sync", "hier_jit__vs__flat_k5_jit",
                                         "hier_jit_d10__vs__flat_jit_d10", "flat_jit_d10__vs__flat_jit"] if all(x in A for x in k.split("__vs__"))}
V = {"T1": abs(C["hier_jit__vs__flat_jit"]["mean_diff"]) <= 0.5 and S["hier_jit"]["wan_GB"] <= 0.1 * S["flat_jit"]["wan_GB"],
     "T2": S["flat_jit"]["miss_rate"] <= 0.5 * S["flat_sync"]["miss_rate"],
     "T3": abs(S["flat_jit"]["rmse"] - 16.35) <= 0.5,
     "T4": C["hier_jit__vs__flat_k5_jit"]["mean_diff"] <= -0.5,
     "T5": abs(C["hier_jit_d10__vs__flat_jit_d10"]["mean_diff"]) <= 0.5,
     "T6": C["flat_jit_d10__vs__flat_jit"]["mean_diff"] >= 0.5} if len(C) == 5 else {}
miss_cell = {}
for tag, d in (("60s", RES), ("10s", os.path.join(RES, "d10"))):
    for mode in ("sync", "jit"):
        ok = [np.array(json.load(open(os.path.join(d, f"delivery_seed{s}.json")))["ok"][mode]) for s in (0, 1, 2)]
        miss_cell[f"{tag}_{mode}"] = dict(zip(TR, [round(float(1 - np.mean([o[:, cell == q].mean() for o in ok])), 3) for q in range(NREG)]))
print(json.dumps(miss_cell, indent=0))
json.dump({"methods": S, "comparisons": C, "verdict": V, "per_cell_miss_rate_every_period_upload": miss_cell}, open(os.path.join(RES, "summary.json"), "w"), indent=1)
print(json.dumps({"comparisons": C, "verdict": V}, indent=1))
for m in sorted(S): print(m.ljust(16), round(S[m]["rmse"], 2), S[m]["curve_25_50_75_100"], round(S[m]["miss_rate"], 3),
                          round(S[m]["access_up_GB"], 2), round(S[m]["access_down_GB"], 2), round(S[m]["wan_GB"], 3), S[m]["per_cell_rmse"])
