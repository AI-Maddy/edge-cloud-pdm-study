"""C7b (review follow-ups E1 + E5): leave-one-fleet-out, so the cloud's models never saw the fleet they serve.

The review noted that C7's feature-upload test used an encoder trained on the same fleet's raw data (circular),
and that C6's "unseen" cold-start sites were random partitions of fleets the cloud had trained on. Here each
C-MAPSS fleet F in turn is held out; the cloud's models are trained only on the 35-site build's other three fleets.
Normalisation uses the shared per-condition table (C3 build).

E1 feature uploads, per held-out fleet F (Transformer, as C2/C7):
  raw        the cloud receives F's raw training windows and trains its own Transformer on them
  feat_same  encoder trained on F's raw data, ridge head on its embeddings (C7's original, circular setup)
  feat_lofo  encoder trained on the other three fleets only; F's sites upload embeddings + labels, the cloud
             fits a ridge head on them (no raw data from F ever reaches the cloud)
  zero_shot  the other-fleet Transformer applied to F as is
  bytes per site-period at equal precision (float32 raw vs float32 embeddings) and with float16 embeddings
E5 cold start, per held-out fleet F (CNN1D, as C6): cloud model trained on the other three fleets; every site of
F is a target with N = 1, 3, 5 own run-to-failure engines: local / cloud as received / fine-tune (5 epochs,
lr 3e-4) / output average.

Tests (fixed before running):
  T1 (E1) feat_lofo RMSE within 10% of raw on at least 3 of the 4 held-out fleets
  T2 (E1) feature objects >= 10x smaller than raw at equal (float32) precision
  T3 (E5) fine-tuning the other-fleet cloud model beats local-only by >= 20% (mean over fleets) for N = 1 and N = 3
  T4 (E5) the other-fleet cloud model as received beats local-only at N = 1 on at least 3 of 4 fleets
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "c3_federation_cells"))
from common import rul  # noqa: E402
import run as c3  # noqa: E402
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
OUT = os.path.join(HERE, "results", "c7b_results.json")
FLEETS = ["FD001", "FD002", "FD003", "FD004"]
ENG, PER = 20, 30


def units_of(sites):
    return np.concatenate([np.full(len(s["ytr"]), i * 1000) + np.cumsum(np.r_[0, np.diff(s["ytr"]) > 0])
                           for i, s in enumerate(sites)])


def pool(sites, key):
    return np.concatenate([s[key] for s in sites])


def emb(model, X):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model.enc(torch.tensor(X[i:i + 4096])).numpy() for i in range(0, len(X), 4096)])


def engines(yy):
    st = np.r_[0, np.where(np.diff(yy) > 0)[0] + 1]; return list(zip(st, np.r_[st[1:], len(yy)]))


def main(seed=0):
    sites = c3.build_sites()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"E1": {}, "E5": {}}
    for F in FLEETS:
        tgt = [s for s in sites if s["fd"] == F]; src = [s for s in sites if s["fd"] != F]
        Xs, ys, us = pool(src, "Xtr"), pool(src, "ytr"), units_of(src)
        Xf, yf, uf = pool(tgt, "Xtr"), pool(tgt, "ytr"), units_of(tgt)
        Xft, yft = pool(tgt, "Xte"), pool(tgt, "yte")
        if F not in res["E1"]:
            t = time.time()
            other, _ = rul.train(rul.make("transformer", Xs.shape[2]), Xs, ys, us, seed=seed, epochs=20)
            own, _ = rul.train(rul.make("transformer", Xf.shape[2]), Xf, yf, uf, seed=seed, epochs=20)
            raw = rul.rmse(rul.predict(own, Xft), yft)
            same = rul.rmse(Ridge(1.0).fit(emb(own, Xf), yf).predict(emb(own, Xft)).clip(0, rul.CAP), yft)
            lofo = rul.rmse(Ridge(1.0).fit(emb(other, Xf), yf).predict(emb(other, Xft)).clip(0, rul.CAP), yft)
            zs = rul.rmse(rul.predict(other, Xft), yft)
            d = emb(other, Xft[:1]).shape[1]
            res["E1"][F] = {"raw": raw, "feat_same": same, "feat_lofo": lofo, "zero_shot": zs,
                            "lofo_vs_raw": lofo / raw - 1, "embedding_dim": int(d), "seconds": round(time.time() - t)}
            json.dump(res, open(OUT, "w"), indent=1); print(F, res["E1"][F], flush=True)
        if F not in res["E5"]:
            t = time.time()
            cloud, _ = rul.train(rul.CNN1D(Xs.shape[2]), Xs, ys, us, seed=seed, epochs=25)
            rows = []
            for k, s in enumerate(tgt):
                eng = engines(s["ytr"])
                for N in (1, 3, 5):
                    ix = np.concatenate([np.arange(a, b) for a, b in eng[:N]])
                    X, y = s["Xtr"][ix], s["ytr"][ix]
                    m_loc = rul.CNN1D(X.shape[2]); torch.manual_seed(seed); c3.local_epoch(m_loc, X, y, np.random.default_rng(seed), epochs=30)
                    m_ft = rul.CNN1D(X.shape[2]); m_ft.load_state_dict(cloud.state_dict())
                    lr = c3.LR; c3.LR = 3e-4; c3.local_epoch(m_ft, X, y, np.random.default_rng(seed), epochs=5); c3.LR = lr
                    pl, pc, pf = (rul.predict(m, s["Xte"]) for m in (m_loc, cloud, m_ft))
                    rows.append({"site": k, "N": N, "local": rul.rmse(pl, s["yte"]), "cloud": rul.rmse(pc, s["yte"]),
                                 "finetune": rul.rmse(pf, s["yte"]), "output_avg": rul.rmse((pl + pc) / 2, s["yte"])})
            agg = {N: {m: float(np.mean([r[m] for r in rows if r["N"] == N])) for m in ("local", "cloud", "finetune", "output_avg")}
                   for N in (1, 3, 5)}
            res["E5"][F] = {"by_N": agg, "rows": rows, "seconds": round(time.time() - t)}
            json.dump(res, open(OUT, "w"), indent=1); print(F, agg, flush=True)
    e1, e5 = res["E1"], res["E5"]
    raw_b = ENG * PER * (24 * 4 + 8); d = e1[FLEETS[0]]["embedding_dim"]
    res["bytes"] = {"raw_float32": raw_b, "feature_float32": ENG * (d * 4 + 8), "feature_float16": ENG * (d * 2 + 8)}
    red = lambda N: float(np.mean([1 - e5[F]["by_N"][str(N) if str(N) in e5[F]["by_N"] else N]["finetune"] /
                                   e5[F]["by_N"][str(N) if str(N) in e5[F]["by_N"] else N]["local"] for F in FLEETS]))
    g = lambda F, N, m: e5[F]["by_N"][str(N)][m] if str(N) in e5[F]["by_N"] else e5[F]["by_N"][N][m]
    res["verdict"] = {
        "T1": sum(e1[F]["lofo_vs_raw"] <= 0.10 for F in FLEETS) >= 3,
        "T2": res["bytes"]["raw_float32"] / res["bytes"]["feature_float32"] >= 10,
        "T3": red(1) >= 0.20 and red(3) >= 0.20,
        "T4": sum(g(F, 1, "cloud") < g(F, 1, "local") for F in FLEETS) >= 3}
    res["finetune_reduction_vs_local"] = {N: red(N) for N in (1, 3, 5)}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("bytes", "verdict", "finetune_reduction_vs_local")}, indent=1))


if __name__ == "__main__":
    main()
