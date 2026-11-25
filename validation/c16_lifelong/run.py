"""C16 validation: lifelong learning at one node (note 10 'Continuous Cognitive Loop'; note 14 sec. 10
'Lifelong Learning Objective': knowledge accumulates, I(t+1) = I(t) + dK(t), subject to ForgetRate < gamma).

A node meets four operating regimes in sequence, the four C-MAPSS fleets FD001 -> FD002 -> FD003 -> FD004
(each switch changes the environment or the fault mode; see C10). Each fleet keeps its own context
normaliser (C2). After every stage the node is scored on the test engines of every fleet seen so far.

Methods (repo Transformer from C2, same training routine, 25 epochs max with early stopping):
  naive     keep training the same model on each new fleet
  replay    same, plus a buffer of 2,000 stored windows from each earlier fleet (~0.5 MB per fleet)
  library   knowledge library: keep one model per fleet (trained from scratch on that fleet) and route each test
            engine to a stored model by context recognition (stored normaliser that fits its cycles best)
  oracle    library with the true fleet id (upper bound for routing)
  joint     one model trained on all four fleets at once (non-sequential reference)
Forgetting of fleet k = RMSE on k after the last stage minus RMSE on k right after it was learned.
Tests (fixed before running):
  T1 replay keeps mean forgetting <= 1.0 RMSE (the note's ForgetRate < gamma, with gamma = 1 RMSE)
  T2 context routing picks the right stored model for >= 95% of test engines, and the routed library is
     within 5% of the oracle library in final mean RMSE
  T3 the best continual method ends within 10% of joint training (mean RMSE over the four fleets)
  T4 forward transfer: starting a stage from the previous model (naive) is at least as good on the new fleet
     as training from scratch, on average over stages 2-4
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import cmapss, rul  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", 3)))
FLEETS = ["FD001", "FD002", "FD003", "FD004"]
EPOCHS, BUF = 25, 2000


def prep(fd):
    tr, te = cmapss.load(fd, "train"), cmapss.load(fd, "test")
    n = rul.Normaliser(True, 6).fit(tr["arr"][:, 2:5], tr["arr"][:, 5:][:, rul.SENS])
    ztr, _ = n.transform(tr["arr"][:, 2:5], tr["arr"][:, 5:][:, rul.SENS])
    zte, _ = n.transform(te["arr"][:, 2:5], te["arr"][:, 5:][:, rul.SENS])
    Xtr, ytr, utr = rul.windows(tr["arr"], ztr, tr["rul"])
    Xte, _, ute = rul.windows(te["arr"], zte, last_only=True)
    return {"norm": n, "Xtr": Xtr, "ytr": ytr, "utr": utr, "Xte": Xte, "ute": ute, "te": te,
            "yte": np.minimum(te["rul_end"], rul.CAP).astype(np.float32)}


def nll(norm, s, x):
    """Mean Gaussian negative log-likelihood of rows under a stored normaliser's per-cell mean/sd."""
    z, lab = norm.transform(s, x)
    return float(np.mean(z ** 2 + 2 * np.log(norm.sd[lab])))


def route(D, fleet, rule="nll"):
    """Context recognition: for each test engine of `fleet`, the stored fleet whose normaliser explains its
    cycles best. Uses unlabeled sensor data only. rule 'nll' = Gaussian likelihood (used for the library);
    rule 'z2' = mean squared z-score only (reported for comparison; it ignores the variance term and favours
    the broadest normaliser)."""
    te = D[fleet]["te"]["arr"]
    picks = []
    for u in np.unique(te[:, 0]):
        rows = te[te[:, 0] == u]
        s, x = rows[:, 2:5], rows[:, 5:][:, rul.SENS]
        cost = {k: (nll(D[k]["norm"], s, x) if rule == "nll" else float(np.mean(D[k]["norm"].transform(s, x)[0] ** 2)))
                for k in FLEETS}
        picks.append(min(cost, key=cost.get))
    return np.array(picks)


def score(m, d):
    return rul.rmse(rul.predict(m, d["Xte"]), d["yte"])


def main():
    D = {f: prep(f) for f in FLEETS}
    f_in = D["FD001"]["Xtr"].shape[2]
    out = {"fleets": FLEETS, "epochs": EPOCHS, "buffer_per_fleet": BUF, "runs": []}
    prev = os.path.join(HERE, "results", "c16_runs.json")
    if os.path.exists(prev):                              # resume: seeds finished before a restart are kept
        out["runs"] = json.load(open(prev))["runs"]
    for seed in (0, 1, 2):
        if any(x["seed"] == seed for x in out["runs"]):
            continue
        g = np.random.default_rng(seed)
        r = {"seed": seed, "naive": [], "replay": [], "scratch": {}}
        # from-scratch model per fleet (= library entries and the T4 reference)
        lib = {}
        for f in FLEETS:
            d = D[f]
            lib[f], _ = rul.train(rul.make("transformer", f_in), d["Xtr"], d["ytr"], d["utr"], seed=seed, epochs=EPOCHS)
            r["scratch"][f] = score(lib[f], d)
        # sequential: naive and replay
        for name in ("naive", "replay"):
            m = rul.make("transformer", f_in)
            buf = []
            for t, f in enumerate(FLEETS):
                d = D[f]
                X, y, u = d["Xtr"], d["ytr"], d["utr"] + 1000 * t
                if name == "replay" and buf:
                    X = np.concatenate([X] + [b[0] for b in buf]); y = np.concatenate([y] + [b[1] for b in buf])
                    u = np.concatenate([u] + [b[2] for b in buf])
                m, _ = rul.train(m, X, y, u, seed=seed, epochs=EPOCHS)
                r[name].append({k: score(m, D[k]) for k in FLEETS[:t + 1]})
                if name == "replay":
                    idx = g.choice(len(d["Xtr"]), BUF, replace=False)
                    buf.append((d["Xtr"][idx], d["ytr"][idx], d["utr"][idx] + 1000 * t))
            print(seed, name, r[name][-1], flush=True)
        # library with context routing vs oracle routing
        r["routing"], r["library"], r["oracle"] = {}, {}, {}
        for f in FLEETS:
            picks = route(D, f)
            r["routing"][f] = {"accuracy": float(np.mean(picks == f)), "accuracy_z2_rule": float(np.mean(route(D, f, "z2") == f)),
                               "picked": {k: int(np.sum(picks == k)) for k in FLEETS}}
            pred = np.array([rul.predict(lib[k], D[f]["Xte"][i:i + 1])[0] for i, k in enumerate(picks)])
            r["library"][f] = rul.rmse(pred, D[f]["yte"])
            r["oracle"][f] = r["scratch"][f]
        # joint reference
        X = np.concatenate([D[f]["Xtr"] for f in FLEETS]); y = np.concatenate([D[f]["ytr"] for f in FLEETS])
        u = np.concatenate([D[f]["utr"] + 1000 * t for t, f in enumerate(FLEETS)])
        mj, _ = rul.train(rul.make("transformer", f_in), X, y, u, seed=seed, epochs=EPOCHS)
        r["joint"] = {f: score(mj, D[f]) for f in FLEETS}
        print(seed, "library", r["library"], "joint", r["joint"], flush=True)
        out["runs"].append(r)
        json.dump(out, open(os.path.join(HERE, "results", "c16_runs.json"), "w"), indent=1)

    R = out["runs"]
    mean_final = lambda key: float(np.mean([np.mean(list((x[key][-1] if isinstance(x[key], list) else x[key]).values())) for x in R]))
    forget = lambda key: float(np.mean([np.mean([x[key][-1][f] - x[key][t][f] for t, f in enumerate(FLEETS[:-1])]) for x in R]))
    fwd = float(np.mean([np.mean([x["naive"][t][f] - x["scratch"][f] for t, f in enumerate(FLEETS) if t > 0]) for x in R]))
    acc = float(np.mean([x["routing"][f]["accuracy"] for x in R for f in FLEETS]))
    s = {m: mean_final(m) for m in ("naive", "replay", "library", "oracle", "joint")}
    s.update({"forgetting_naive": forget("naive"), "forgetting_replay": forget("replay"),
              "routing_accuracy": acc, "forward_transfer_naive_minus_scratch": fwd})
    best = min(s["replay"], s["library"], s["naive"])
    out["summary"] = s
    out["verdict"] = {"T1": s["forgetting_replay"] <= 1.0,
                      "T2": acc >= 0.95 and s["library"] <= 1.05 * s["oracle"],
                      "T3": best <= 1.10 * s["joint"],
                      "T4": fwd <= 0.0}
    json.dump(out, open(os.path.join(HERE, "results", "c16_results.json"), "w"), indent=1)
    print(json.dumps({"summary": s, "verdict": out["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
