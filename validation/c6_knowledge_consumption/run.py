"""C6 validation: downstream knowledge dissemination and consumption (note 05 'Knowledge Consumption Engine':
Receive -> Validity Check -> Trust -> Merge -> Adaptation, M_new = M + alpha*K; note 12 sec. 14-15).

Cold start on C-MAPSS: 8 target sites (2 per subset, never used by the cloud) receive the cloud model, which was
trained on the other 27 sites (stands in for a converged federation, C3). Each target has only N = 1, 3 or 5
run-to-failure engines of its own; its test engines are its share of the official test set.
Consumption policies (same CNN1D everywhere):
  local        train from scratch on local data only
  cloud        use the received model as is
  weight_blend note-05 formula M + 0.5 (K - M) with M the independently trained local model
  output_blend average of local and cloud predictions
  finetune     adapt: continue training the received model on local data (5 epochs, lr 3e-4)
  validated    validity check: pick among local / cloud / output_blend / finetune by error on a local hold-out
               (last engine when N >= 3; else the last 30% of cycles of the single engine)
Tests (fixed before running):
  T1 the note's weight-space blend beats local-only in >= 75% of (site, N, seed) cases
  T2 validated consumption is never more than 5% worse than local-only (harm rate <= 10%)
  T3 finetune beats local-only on mean RMSE for N <= 3
"""
import json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "c3_federation_cells"))
from common import rul
import run as c3
torch.set_num_threads(4)


def train_fresh(X, y, seed, epochs=30):
    torch.manual_seed(seed); m = rul.CNN1D(X.shape[2]); g = np.random.default_rng(seed)
    c3.local_epoch(m, X, y, g, epochs=epochs); return m


def finetune(base, X, y, seed, epochs=5):
    m = rul.CNN1D(X.shape[2]); m.load_state_dict(base.state_dict()); g = np.random.default_rng(seed)
    opt_lr = c3.LR; c3.LR = 3e-4; c3.local_epoch(m, X, y, g, epochs=epochs); c3.LR = opt_lr; return m


def main():
    sites = c3.build_sites()
    fds = [s["fd"] for s in sites]
    out = {"runs": []}
    for seed in (0, 1, 2):
        g = np.random.default_rng(seed)
        targets = []
        for fd in c3.SITES:
            idx = [i for i, f in enumerate(fds) if f == fd]; targets += list(g.choice(idx, 2, replace=False))
        src = [i for i in range(len(sites)) if i not in targets]
        X = np.concatenate([sites[i]["Xtr"] for i in src]); y = np.concatenate([sites[i]["ytr"] for i in src])
        u = np.concatenate([np.full(len(sites[i]["ytr"]), i * 1000) + np.arange(len(sites[i]["ytr"])) // 200 for i in src])
        cloud, _ = rul.train(rul.CNN1D(X.shape[2]), X, y, u, seed=seed, epochs=25)
        for t in targets:
            s = sites[t]
            # recover per-engine windows: windows are contiguous per unit; split by RUL resets
            yy = s["ytr"]; starts = np.r_[0, np.where(np.diff(yy) > 0)[0] + 1]; ends = np.r_[starts[1:], len(yy)]
            for N in (1, 3, 5):
                eng = list(zip(starts, ends))[:N]
                if N >= 3:
                    fit_ix = np.concatenate([np.arange(a, b) for a, b in eng[:-1]]); hold_ix = np.arange(*eng[-1])
                else:
                    a, b = eng[0]; cut = a + int(0.7 * (b - a)); fit_ix = np.arange(a, cut); hold_ix = np.arange(cut, b)
                all_ix = np.concatenate([np.arange(a, b) for a, b in eng])
                Xl, yl = s["Xtr"][all_ix], s["ytr"][all_ix]
                local = train_fresh(Xl, yl, seed)
                ft = finetune(cloud, Xl, yl, seed)
                blend = rul.CNN1D(Xl.shape[2]); blend.load_state_dict({k: 0.5 * (local.state_dict()[k] + cloud.state_dict()[k]) for k in local.state_dict()})
                P = {"local": rul.predict(local, s["Xte"]), "cloud": rul.predict(cloud, s["Xte"]),
                     "weight_blend": rul.predict(blend, s["Xte"]), "finetune": rul.predict(ft, s["Xte"])}
                P["output_blend"] = 0.5 * (P["local"] + P["cloud"])
                # validity check on local hold-out, with candidates refit on fit_ix only (no peeking)
                l2 = train_fresh(s["Xtr"][fit_ix], s["ytr"][fit_ix], seed); f2 = finetune(cloud, s["Xtr"][fit_ix], s["ytr"][fit_ix], seed)
                Xh, yh = s["Xtr"][hold_ix], s["ytr"][hold_ix]
                ph = {"local": rul.predict(l2, Xh), "cloud": rul.predict(cloud, Xh), "finetune": rul.predict(f2, Xh)}
                ph["output_blend"] = 0.5 * (ph["local"] + ph["cloud"])
                choice = min(ph, key=lambda k: rul.rmse(ph[k], yh))
                P["validated"] = P[choice]
                r = {"seed": seed, "site": int(t), "fd": s["fd"], "N": N, "validated_choice": choice,
                     **{k: rul.rmse(v, s["yte"]) for k, v in P.items()}}
                out["runs"].append(r); print(r, flush=True)
        json.dump(out, open(os.path.join(HERE, "results", "c6_runs.json"), "w"), indent=1)
    R = out["runs"]
    wb = np.mean([r["weight_blend"] < r["local"] for r in R])
    harm = np.mean([r["validated"] > 1.05 * r["local"] for r in R])
    ft3 = np.mean([r["finetune"] for r in R if r["N"] <= 3]) < np.mean([r["local"] for r in R if r["N"] <= 3])
    out["summary"] = {N: {k: float(np.mean([r[k] for r in R if r["N"] == N])) for k in ["local", "cloud", "weight_blend", "output_blend", "finetune", "validated"]} for N in (1, 3, 5)}
    out["verdict"] = {"T1": bool(wb >= 0.75), "T1_rate": float(wb), "T2": bool(harm <= 0.10), "T2_harm_rate": float(harm), "T3": bool(ft3)}
    json.dump(out, open(os.path.join(HERE, "results", "c6_runs.json"), "w"), indent=1)
    print(json.dumps({"summary": out["summary"], "verdict": out["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
