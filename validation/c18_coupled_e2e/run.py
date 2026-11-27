"""C18 (review-2 follow-up E7): flat vs regional federation coupled to trace-driven uplink delivery with deadlines.

The second review asked for one experiment in which delivery behaviour feeds back into model quality: compare flat
FedAvg with a HierFAVG-style region tier under the same recorded uplinks, deadlines and participation rules, with
the model, local work and client population fixed, and include a flat baseline that communicates less often.

Setup. The 35 C-MAPSS sites and CNN1D of C3/C3b (84,129 parameters; an update is 336,516 bytes in float32). Each
federated round is one 60 s reporting period. Sites are assigned to five cells, 7 per cell, one recorded uplink
trace per cell (the C3b random "geographic" regions; cell q uses trace q of C7c: T-Mobile LTE, Verizon LTE,
T-Mobile UMTS, Verizon EVDO, AT&T LTE with outages). In every period each uploading site trains one local epoch and
sends its update over its cell's uplink, simulated packet by packet with the C7c simulator (shared drop-tail
queue, Reno-like flows, 40 ms RTT) from a random trace offset. An update counts only if its last byte is
acknowledged within the period (deadline 60 s after the period starts); a late upload is abandoned at the deadline
and its work is lost. Downloads are not simulated (downlinks are faster); they are counted in bytes.
  flat      every period, all sites upload to the cloud, the cloud averages the updates that arrived
  hier      every period, sites upload to their cell's regional server (MEC), the region averages what arrived;
            every 5 periods the regions average at the cloud over the wired WAN (assumed on time)
  flat_k5   flat FedAvg that communicates every 5 periods: sites train one epoch per period on their own copy,
            upload every 5th period (same local work, same cloud period as hier)
Upload timing: "sync" = every site starts at the period start; "jit" = each site starts at a random offset in the
first 30 s of the period. Same trace offsets and jitter draws for every method within a seed (common random
numbers). 100 periods, 3 seeds. RMSE every 25 periods (elapsed time = periods x 60 s).
Bytes: access uplink (attempted uploads), access downlink (model downloads), WAN (site<->cloud for flat and
flat_k5, region<->cloud for hier).

Tests (fixed before running):
  T1 hier_jit within 0.5 RMSE of flat_jit at 100 periods while using <= 10% of its WAN bytes
  T2 jittered starts cut flat's deadline misses by >= 50% relative to sync
  T3 flat_jit within 0.5 RMSE of deadline-free FedAvg at 100 rounds (C3b: 16.35)
  T4 hier_jit beats flat_k5_jit (same cloud period, same local work) by >= 0.5 RMSE
Stress setting (added before running it, while the 60 s runs were in progress, because their miss rates were only
2-3%): updates must arrive within 10 s of the period start, jittered starts in the first 5 s (DEADLINE=10 JIT=5,
results in results/d10/; flat_jit and hier_jit only).
  T5 under the 10 s deadline, hier_jit is within 0.5 RMSE of flat_jit
  T6 under the 10 s deadline, flat_jit is >= 0.5 RMSE worse than flat_jit with the 60 s deadline (delivery limits
     accuracy, i.e. the coupling matters)
"""
import importlib.util, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "c3_federation_cells"))
from common import rul  # noqa: E402
import run as c3  # noqa: E402
spec = importlib.util.spec_from_file_location("c7c", os.path.join(HERE, "..", "c7c_uplink_sim", "run.py"))
c7c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c7c)
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
P, KAPPA, NREG = 100, 5, 5
PERIOD = float(os.environ.get("DEADLINE", "60")); JIT = float(os.environ.get("JIT", "30"))
RES = os.path.join(HERE, "results", os.environ.get("TAG", ""))
os.makedirs(RES, exist_ok=True)
JOBS = [(int(j.split(":")[0]), j.split(":")[1]) for j in os.environ.get("JOBS", "").split(",") if j]
METHODS = ["flat_sync", "flat_jit", "hier_sync", "hier_jit", "flat_k5_sync", "flat_k5_jit"]


def save(obj, path):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, path)


def delivery(cell, seed, nbytes, traces):
    """delivered[mode][period, site] for every period (uploads every period); cached per seed."""
    path = os.path.join(RES, f"delivery_seed{seed}.json")
    if os.path.exists(path):
        d = json.load(open(path)); return {m: np.array(v, bool) for m, v in d["ok"].items()}, d
    g = np.random.default_rng(1000 + seed); n = len(cell)
    t0 = g.uniform(0, 1, (P, NREG)); jit = g.uniform(0, JIT, (P, n))
    ok = {"sync": np.zeros((P, n), bool), "jit": np.zeros((P, n), bool)}; times = {"sync": [], "jit": []}
    for r in range(P):
        for q in range(NREG):
            idx = np.where(cell == q)[0]; tr = traces[q]
            for mode in ("sync", "jit"):
                st = np.zeros(len(idx)) if mode == "sync" else jit[r, idx]
                done = c7c.simulate(tr, t0[r, q] * tr[-1], st, nbytes)
                ok[mode][r, idx] = (st + done) <= PERIOD
                times[mode] += [float(x) for x in st + done]
    d = {"ok": {m: v.tolist() for m, v in ok.items()}, "finish_after_period_start": times}
    save(d, path); return ok, d


def run(sites, seed, method, cell, ok):
    g = np.random.default_rng(seed); torch.manual_seed(seed)
    F = sites[0]["Xtr"].shape[2]; w0 = c3.get(rul.CNN1D(F)); work = rul.CNN1D(F)
    ns = np.array([len(s["ytr"]) for s in sites]); n = len(sites)
    kind, mode = method.rsplit("_", 1); OK = ok[mode]
    curve, misses, attempts = [], 0, 0
    if kind == "flat":
        w = w0
        for r in range(P):
            ws, wn = [], []
            for i in range(n):
                c3.put(work, w); c3.local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); attempts += 1
                if OK[r, i]: ws.append(c3.get(work)); wn.append(ns[i])
                else: misses += 1
            if ws: w = c3.avg(ws, wn)
            if (r + 1) % 25 == 0: curve.append(float(np.mean(ev([w] * n, sites, F))))
        final = [w] * n
    elif kind == "hier":
        cloud = w0; reg = [w0] * NREG
        for r in range(P):
            for q in range(NREG):
                idx = np.where(cell == q)[0]; ws, wn = [], []
                for i in idx:
                    c3.put(work, reg[q]); c3.local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); attempts += 1
                    if OK[r, i]: ws.append(c3.get(work)); wn.append(ns[i])
                    else: misses += 1
                if ws: reg[q] = c3.avg(ws, wn)
            if (r + 1) % KAPPA == 0:
                cloud = c3.avg(reg, [ns[cell == q].sum() for q in range(NREG)]); reg = [cloud] * NREG
            if (r + 1) % 25 == 0: curve.append(float(np.mean(ev([cloud] * n, sites, F))))
        final = [cloud] * n
    else:  # flat_k5
        w = w0; loc = [w0] * n
        for r in range(P):
            ws, wn = [], []
            for i in range(n):
                c3.put(work, loc[i]); c3.local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); loc[i] = c3.get(work)
                if (r + 1) % KAPPA == 0:
                    attempts += 1
                    if OK[r, i]: ws.append(loc[i]); wn.append(ns[i])
                    else: misses += 1
            if (r + 1) % KAPPA == 0:
                if ws: w = c3.avg(ws, wn)
                loc = [w] * n
            if (r + 1) % 25 == 0: curve.append(float(np.mean(ev([w] * n, sites, F))))
        final = [w] * n
    nb = 4 * sum(p.numel() for p in work.parameters())
    up = attempts * nb
    if kind == "flat": down, wan = P * n * nb, 2 * P * n * nb
    elif kind == "hier": down, wan = P * n * nb, 2 * (P // KAPPA) * NREG * nb
    else: down, wan = (P // KAPPA) * n * nb, 2 * (P // KAPPA) * n * nb
    return ev(final, sites, F), {"curve_every_25": curve, "miss_rate": misses / attempts, "attempts": attempts,
                                 "access_up_bytes": int(up), "access_down_bytes": int(down), "wan_bytes": int(wan)}


def ev(ws, sites, F):
    m = rul.CNN1D(F); out = []
    for w, s in zip(ws, sites):
        c3.put(m, w); out.append(float(rul.rmse(rul.predict(m, s["Xte"]), s["yte"])))
    return out


def main():
    sites = c3.build_sites(); n = len(sites)
    cell = np.random.default_rng(123).permutation(np.arange(n) % NREG)   # = C3b hier_geo regions
    traces = [c7c.load(t) for t in c7c.TRACES]
    nb = 4 * sum(p.numel() for p in rul.CNN1D(sites[0]["Xtr"].shape[2]).parameters())
    out = os.path.join(RES, os.environ.get("OUT", "runs.json"))
    res = json.load(open(out)) if os.path.exists(out) else {"runs": []}
    done = {(r["method"], r["seed"]) for r in res["runs"]}
    jobs = JOBS or [(s, m) for s in (0, 1, 2) for m in METHODS]
    for seed, method in jobs:
        if (method, seed) in done: continue
        ok, _ = delivery(cell, seed, nb, traces); t = time.time()
        sr, info = run(sites, seed, method, cell, ok)
        r = {"method": method, "seed": seed, "site_rmse": sr, "mean_site_rmse": float(np.mean(sr)), **info,
             "seconds": round(time.time() - t, 1)}
        res["runs"].append(r); save(res, out); print({k: v for k, v in r.items() if k != "site_rmse"}, flush=True)


def summary():
    import glob
    from scipy.stats import wilcoxon
    runs = {(r["method"], r["seed"]): r for f in glob.glob(os.path.join(RES, "runs*.json")) for r in json.load(open(f))["runs"]}
    M = {}
    for (m, s), r in runs.items(): M.setdefault(m, []).append(r)
    A = {m: np.mean([r["site_rmse"] for r in v], 0) for m, v in M.items()}
    mean = lambda m, k: float(np.mean([r[k] for r in M[m]]))
    S = {m: {"rmse": float(A[m].mean()), "curve": np.mean([r["curve_every_25"] for r in M[m]], 0).tolist(),
             "miss_rate": mean(m, "miss_rate"), "access_up_MB": mean(m, "access_up_bytes") / 1e6,
             "access_down_MB": mean(m, "access_down_bytes") / 1e6, "wan_MB": mean(m, "wan_bytes") / 1e6, "seeds": len(M[m])}
         for m in M}
    def cmp(a, b):
        d = A[a] - A[b]; rng = np.random.default_rng(0)
        bt = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                "p": float(wilcoxon(A[a], A[b]).pvalue)}
    cell = np.random.default_rng(123).permutation(np.arange(35) % NREG)
    per_cell = {m: [float(A[m][cell == q].mean()) for q in range(NREG)] for m in A}
    C = {"hier_jit_vs_flat_jit": cmp("hier_jit", "flat_jit"), "hier_jit_vs_flat_k5_jit": cmp("hier_jit", "flat_k5_jit"),
         "hier_sync_vs_flat_sync": cmp("hier_sync", "flat_sync")}
    V = {"T1": abs(C["hier_jit_vs_flat_jit"]["mean_diff"]) <= 0.5 and S["hier_jit"]["wan_MB"] <= 0.1 * S["flat_jit"]["wan_MB"],
         "T2": S["flat_jit"]["miss_rate"] <= 0.5 * S["flat_sync"]["miss_rate"],
         "T3": abs(S["flat_jit"]["rmse"] - 16.35) <= 0.5,
         "T4": C["hier_jit_vs_flat_k5_jit"]["mean_diff"] <= -0.5}
    out = {"methods": S, "per_cell_rmse": per_cell, "cell_traces": c7c.TRACES, "comparisons": C, "verdict": V}
    save(out, os.path.join(RES, "summary.json")); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    summary() if os.environ.get("SUMMARY") else main()
