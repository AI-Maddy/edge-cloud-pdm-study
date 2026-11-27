"""C7b-b (review-2 point 4 and 5, diagnostics added after C7b; not new pass/fail tests).

Per held-out fleet, with the same seed and the same other-fleet Transformer encoder as C7b:
  feat_lofo_f32   C7b's feature result (ridge head on float32 embeddings), recomputed
  feat_lofo_f16   the same embeddings rounded to float16 before upload (the 1,440-byte object of Table VII)
  pca_lofo        a simple compressor instead of the learned encoder: PCA-32 of the flattened 30x14 window,
                  fitted on the other three fleets, ridge head fitted on the held-out fleet (same labels/split)
  ridge_raw       ridge on the held-out fleet's flattened raw windows (a linear model with raw access)
Every method uses the held-out fleet's training-engine labels and is scored on the same test engines (last window
of each test engine), exactly as raw/feat_lofo in C7b.
Cold start (C7b E5): two pooled simple predictors that use the same other-fleet failure histories as the cloud
model but no learned network, to separate "received model" from "access to more failure histories":
  pooled_mean     predict the mean RUL label of the other fleets' training windows
  pooled_ridge    ridge on flattened windows of the other fleets
"""
import importlib.util, json, os, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c7b", os.path.join(HERE, "run.py")); c7b = importlib.util.module_from_spec(spec); spec.loader.exec_module(c7b)
rul, c3 = c7b.rul, c7b.c3
OUT = os.path.join(HERE, "results", "c7b_b_results.json")


def main(seed=0):
    sites = c3.build_sites(); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for F in c7b.FLEETS:
        if F in res: continue
        tgt = [s for s in sites if s["fd"] == F]; src = [s for s in sites if s["fd"] != F]
        Xs, ys, us = c7b.pool(src, "Xtr"), c7b.pool(src, "ytr"), c7b.units_of(src)
        Xf, yf = c7b.pool(tgt, "Xtr"), c7b.pool(tgt, "ytr"); Xft, yft = c7b.pool(tgt, "Xte"), c7b.pool(tgt, "yte")
        flat = lambda X: X.reshape(len(X), -1)
        r = {}
        # cold-start pooled baselines (per target site, then mean over sites, as C7b E5)
        pr = Ridge(1.0).fit(flat(Xs), ys)
        r["cold_pooled_mean"] = float(np.mean([rul.rmse(np.full(len(s["yte"]), ys.mean()), s["yte"]) for s in tgt]))
        r["cold_pooled_ridge"] = float(np.mean([rul.rmse(pr.predict(flat(s["Xte"])).clip(0, rul.CAP), s["yte"]) for s in tgt]))
        pca = PCA(32, random_state=seed).fit(flat(Xs))
        r["pca_lofo"] = rul.rmse(Ridge(1.0).fit(pca.transform(flat(Xf)), yf).predict(pca.transform(flat(Xft))).clip(0, rul.CAP), yft)
        r["ridge_raw"] = rul.rmse(Ridge(1.0).fit(flat(Xf), yf).predict(flat(Xft)).clip(0, rul.CAP), yft)
        other, _ = rul.train(rul.make("transformer", Xs.shape[2]), Xs, ys, us, seed=seed, epochs=20)
        E, Et = c7b.emb(other, Xf), c7b.emb(other, Xft)
        r["feat_lofo_f32"] = rul.rmse(Ridge(1.0).fit(E, yf).predict(Et).clip(0, rul.CAP), yft)
        E16, Et16 = E.astype(np.float16).astype(np.float32), Et.astype(np.float16).astype(np.float32)
        r["feat_lofo_f16"] = rul.rmse(Ridge(1.0).fit(E16, yf).predict(Et16).clip(0, rul.CAP), yft)
        r["c7b_raw"] = json.load(open(c7b.OUT))["E1"][F]["raw"]
        res[F] = r; json.dump(res, open(OUT, "w"), indent=1); print(F, r, flush=True)


if __name__ == "__main__":
    main()
