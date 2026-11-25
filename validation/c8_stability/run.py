"""C8 validation: convergence and stability claims (note 13) measured on real federated training, using the
repository's scripts/hfcf/evaluation.py stability().

FedAvg over the 35 C-MAPSS sites of C3 for 40 rounds (seeds 0, 1). Each site holds out 10% of its training engines
for validation (test engines untouched). Per round we log: validation RMSE (quality), global update energy
||w_t - w_{t-1}||^2 (model energy), client drift (mean ||w_i - w_t||^2 after local training). Cell energy is the
cell-membership churn (0 here: cells fixed; for drifting cells see C4).
Claims from note 13 tested (fixed before running):
  T1 'monotonic improvement': validation RMSE never increases by more than 1% from one round to the next
  T2 'Lyapunov stability': the repository's energy is non-increasing over the last 20 rounds (tolerance 5% of its
     initial value)
  T3 'federation stability': client drift in the last 10 rounds is below half of the first 10 rounds
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "c3_federation_cells")); sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from common import rul
import run as c3
from hfcf import evaluation
torch.set_num_threads(4)
R = 40


def flat(ws): return torch.cat([w.flatten() for w in ws])


def main():
    sites = c3.build_sites()
    for s in sites:   # hold out ~10% of engines (contiguous blocks of windows split at RUL resets)
        y = s["ytr"]; starts = np.r_[0, np.where(np.diff(y) > 0)[0] + 1]; ends = np.r_[starts[1:], len(y)]
        nv = max(1, len(starts) // 10); vi = np.concatenate([np.arange(a, b) for a, b in list(zip(starts, ends))[-nv:]])
        m = np.zeros(len(y), bool); m[vi] = True
        s["Xv"], s["yv"], s["Xtr"], s["ytr"] = s["Xtr"][m], s["ytr"][m], s["Xtr"][~m], s["ytr"][~m]
    out = {}
    for seed in (0, 1):
        g = np.random.default_rng(seed); torch.manual_seed(seed)
        work = rul.CNN1D(sites[0]["Xtr"].shape[2]); w = c3.get(work); hist = []
        for r in range(R):
            ws, ns = [], []
            for s in sites:
                c3.put(work, w); c3.local_epoch(work, s["Xtr"], s["ytr"], g); ws.append(c3.get(work)); ns.append(len(s["ytr"]))
            new = c3.avg(ws, ns)
            drift = float(np.mean([float(((flat(x) - flat(new)) ** 2).sum()) for x in ws]))
            energy = float(((flat(new) - flat(w)) ** 2).sum()); w = new
            c3.put(work, w)
            v = float(np.sqrt(np.mean(np.concatenate([(rul.predict(work, s["Xv"]) - s["yv"]) ** 2 for s in sites]))))
            hist.append({"round": r + 1, "val_rmse": v, "model_energy": energy, "client_drift": drift,
                         "cluster_energy": 0.0, "knowledge_energy": 0.0, "quality": -v})
            print(seed, hist[-1], flush=True)
        st = evaluation.stability({"history": hist[-20:]}, {"tolerance": 0.05 * hist[0]["model_energy"]})
        vr = np.array([h["val_rmse"] for h in hist]); dr = np.array([h["client_drift"] for h in hist])
        out[seed] = {"history": hist, "repo_stability_last20": st,
                     "T1": bool(np.all(vr[1:] <= 1.01 * vr[:-1])), "rounds_with_rmse_increase_gt1pct": int(np.sum(vr[1:] > 1.01 * vr[:-1])),
                     "T2": bool(st["nonincreasing_energy_observed"]),
                     "T3": bool(dr[-10:].mean() < 0.5 * dr[:10].mean()), "drift_first10": float(dr[:10].mean()), "drift_last10": float(dr[-10:].mean())}
        json.dump(out, open(os.path.join(HERE, "results", "c8_results.json"), "w"), indent=1)
    print({s: {k: v for k, v in o.items() if k not in ("history", "repo_stability_last20")} for s, o in out.items()})


if __name__ == "__main__":
    main()
