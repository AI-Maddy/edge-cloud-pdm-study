"""Train and save the models produced by the concept validations, so they can be reused without retraining.

Writes validation/trained_models/:
  c2_transformer_<FD>.pt       repo TransformerEncoder + head, context-normalised (seed 0), per subset
  c2_normaliser_<FD>.json      per-operating-condition centroids, means and stds for that subset
  c3_fedavg_cnn1d.pt           FedAvg CNN1D over the 35 sites (25 rounds, seed 0)
  c3_cafig_cell<k>_cnn1d.pt    CAFIG cell models (seed 0)
  c3_normaliser.json           federation-wide normaliser
  c6_cloud_cnn1d.pt            cloud model trained on 27 source sites (seed 0 split)
  c9_<simclr|byol|ae>.pt       self-supervised encoders (seed 0)
  c4_cells.json                Beijing context cells (K-Means k=6 on year 1) and per-cell ridge coefficients
  c5_knowledge_objects.json    mined knowledge objects with repository trust scores (seed 0 sources)
  MANIFEST.json                file list, sizes, SHA-256, provenance and test metrics
Load example: see trained_models/README.md.
"""
import hashlib, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "trained_models")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "c3_federation_cells")); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from common import rul, cmapss
import run as c3
torch.set_num_threads(4)
manifest = {}


def norm_json(n):
    return {"centroids_scaled_settings": None if n.km is None else n.km.cluster_centers_.tolist(), "settings_scale": [42, 0.84, 100],
            "mean": n.mu.tolist(), "std": n.sd.tolist(), "sensor_columns": [f"s{i+1}" for i in rul.SENS], "window": rul.WIN, "rul_cap": rul.CAP}


def save_pt(model, name, meta):
    p = os.path.join(OUT, name); torch.save({"state_dict": model.state_dict(), "meta": meta}, p); manifest[name] = meta


def save_json(obj, name, meta):
    p = os.path.join(OUT, name); json.dump(obj, open(p, "w"), indent=1); manifest[name] = meta


def c2():
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        tr = cmapss.load(fd, "train"); n = rul.Normaliser(True).fit(tr["arr"][:, 2:5], tr["arr"][:, 5:][:, rul.SENS])
        Xtr, ytr, utr, Xte, yte, _ = rul.load_fd(fd)
        m, _ = rul.train(rul.make("transformer", Xtr.shape[2]), Xtr, ytr, utr, seed=0)
        save_pt(m, f"c2_transformer_{fd}.pt", {"class": "common.rul.RepoTransformerRUL", "args": {"f": 14}, "test_rmse": rul.rmse(rul.predict(m, Xte), yte), "seed": 0})
        save_json(norm_json(n), f"c2_normaliser_{fd}.json", {"for": f"c2_transformer_{fd}.pt"})
        print(fd, manifest[f"c2_transformer_{fd}.pt"], flush=True)


def c3_models():
    sites = c3.build_sites()
    allset = np.vstack([cmapss.load(fd, "train")["arr"][:, 2:5] for fd in c3.SITES]); allsen = np.vstack([cmapss.load(fd, "train")["arr"][:, 5:][:, rul.SENS] for fd in c3.SITES])
    save_json(norm_json(rul.Normaliser(True, 6).fit(allset, allsen)), "c3_normaliser.json", {"for": "c3_*, c6_*"})
    from sklearn.cluster import KMeans
    C = np.array([s["ctx"] for s in sites]); cells = KMeans(2, n_init=10, random_state=0).fit_predict(C)
    for name, groups in [("fedavg", np.zeros(len(sites), int)), ("cafig", cells)]:
        g = np.random.default_rng(0); torch.manual_seed(0)
        F = sites[0]["Xtr"].shape[2]; w0 = c3.get(rul.CNN1D(F)); cw = [w0 for _ in range(groups.max() + 1)]; work = rul.CNN1D(F)
        for r in range(c3.ROUNDS):
            new = []
            for c in range(len(cw)):
                ws, ns = [], []
                for i in np.where(groups == c)[0]:
                    c3.put(work, cw[c]); c3.local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); ws.append(c3.get(work)); ns.append(len(sites[i]["ytr"]))
                new.append(c3.avg(ws, ns))
            cw = new
        for c, w in enumerate(cw):
            m = rul.CNN1D(F); c3.put(m, w)
            idx = np.where(groups == c)[0]
            rm = float(np.mean([rul.rmse(rul.predict(m, sites[i]["Xte"]), sites[i]["yte"]) for i in idx]))
            nm = "c3_fedavg_cnn1d.pt" if name == "fedavg" else f"c3_cafig_cell{c}_cnn1d.pt"
            save_pt(m, nm, {"class": "common.rul.CNN1D", "args": {"f": 14}, "rounds": c3.ROUNDS, "sites": [sites[i]["fd"] for i in idx], "mean_site_test_rmse": rm, "seed": 0})
            print(nm, rm, flush=True)
    # C6 cloud model (seed 0 target split)
    fds = [s["fd"] for s in sites]; g = np.random.default_rng(0); targets = []
    for fd in c3.SITES:
        idx = [i for i, f in enumerate(fds) if f == fd]; targets += list(g.choice(idx, 2, replace=False))
    src = [i for i in range(len(sites)) if i not in targets]
    X = np.concatenate([sites[i]["Xtr"] for i in src]); y = np.concatenate([sites[i]["ytr"] for i in src])
    u = np.concatenate([np.full(len(sites[i]["ytr"]), i * 1000) + np.arange(len(sites[i]["ytr"])) // 200 for i in src])
    cloud, _ = rul.train(rul.CNN1D(X.shape[2]), X, y, u, seed=0, epochs=25)
    rm = float(np.mean([rul.rmse(rul.predict(cloud, sites[i]["Xte"]), sites[i]["yte"]) for i in targets]))
    save_pt(cloud, "c6_cloud_cnn1d.pt", {"class": "common.rul.CNN1D", "args": {"f": 14}, "source_sites": src, "target_sites": [int(t) for t in targets], "target_mean_test_rmse": rm, "seed": 0})
    print("c6", rm, flush=True)
    return sites


def c9(sites):
    sys.path.insert(0, os.path.join(HERE, "c9_self_supervised"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("c9run", os.path.join(HERE, "c9_self_supervised", "run.py")); c9m = importlib.util.module_from_spec(spec); spec.loader.exec_module(c9m)
    Xall = np.concatenate([s["Xtr"] for s in sites])
    for kind in ("simclr", "byol", "ae"):
        torch.manual_seed(0); g = np.random.default_rng(0)
        m = {"simclr": lambda: c9m.SimCLR(input_dim=c9m.D, hidden_dim=c9m.H, embedding_dim=c9m.E, projection_dim=c9m.E),
             "byol": lambda: c9m.BYOL(input_dim=c9m.D, hidden_dim=c9m.H, embedding_dim=c9m.E, projection_dim=c9m.E),
             "ae": lambda: c9m.Autoencoder(input_dim=c9m.D, hidden_dim=c9m.H, embedding_dim=c9m.E)}[kind]()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3); Xt = torch.tensor(Xall)
        for ep in range(15):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt) - 511, 512):
                xb = Xt[perm[i:i + 512]]
                loss = ((m(c9m.augment(xb, g)) - xb.reshape(len(xb), -1)) ** 2).mean() if kind == "ae" else m.loss(c9m.augment(xb, g), c9m.augment(xb, g))
                opt.zero_grad(); loss.backward(); opt.step()
                if kind == "byol": m.update_target()
        save_pt(m, f"c9_{kind}.pt", {"class": f"hfcf_models.neural.{ {'simclr':'SimCLR','byol':'BYOL','ae':'Autoencoder'}[kind] }", "args": {"input_dim": c9m.D, "hidden_dim": c9m.H, "embedding_dim": c9m.E}, "epochs": 15, "seed": 0})
        print("c9", kind, flush=True)


def c4_and_c5():
    from sklearn.cluster import KMeans
    from common import beijing
    names, C, P = beijing.daily()
    S, D, F = C.shape
    Z = C[:, :365].reshape(-1, F); ok = np.isfinite(Z).all(1); mu, sd = Z[ok].mean(0), Z[ok].std(0)
    km = KMeans(6, n_init=10, random_state=0).fit(((Z[ok] - mu) / sd))
    X, Y = [], []
    for s in range(S):
        for d in range(364):
            x = np.r_[1.0, P[s, d], (C[s, d] - mu) / sd]
            if np.isfinite(x).all() and np.isfinite(P[s, d + 1]): X.append(x); Y.append(P[s, d + 1])
    X, Y = np.array(X), np.array(Y); lab = km.predict(X[:, 2:])
    coef = [np.linalg.solve(X[lab == c].T @ X[lab == c] + np.eye(X.shape[1]), X[lab == c].T @ Y[lab == c]).tolist() for c in range(6)]
    save_json({"context_fields": beijing.CTX, "context_mean": mu.tolist(), "context_std": sd.tolist(), "centroids": km.cluster_centers_.tolist(),
               "ridge_features": ["1", "log_pm25_today", *beijing.CTX], "ridge_coef_per_cell": coef, "stations": names, "fit_period": "2013-03-01..2014-02-28"},
              "c4_cells.json", {"what": "Beijing context cells (k=6) and per-cell next-day log-PM2.5 ridge models, year 1"})
    import importlib.util
    spec = importlib.util.spec_from_file_location("c5run", os.path.join(HERE, "c5_knowledge_trust", "run.py")); c5m = importlib.util.module_from_spec(spec); spec.loader.exec_module(c5m)
    from hfcf import knowledge
    sites = c5m.site_rows(); g = np.random.default_rng(0); src = g.permutation(len(sites))[:24]
    objs = []
    for i in src:
        st = sites[i]
        for s in range(st["z"].shape[1]):
            sign = 1 if np.corrcoef(st["z"][:, s], st["y"])[0, 1] > 0 else -1
            th = np.percentile(st["z"][:, s], 97.5 if sign > 0 else 2.5); conf, sup = c5m.precision((s, th, sign), st["z"], st["y"])
            if sup >= 5 and np.isfinite(conf):
                objs.append({"id": f"K{i}-s{rul.SENS[s]+1}", "source_site": int(i), "pattern": f"s{rul.SENS[s]+1}{'>' if sign > 0 else '<'}{th:.3f}",
                             "conditions": [f"condition-normalised s{rul.SENS[s]+1} {'>' if sign > 0 else '<'} {th:.3f} (5-cycle mean)"],
                             "outcome": f"failure within {c5m.H} cycles", "confidence": conf, "support": sup, "quality": conf, "source_reliability": conf})
    scored = knowledge.trust({"records": objs}, {})["records"]
    save_json(scored, "c5_knowledge_objects.json", {"what": f"{len(scored)} knowledge objects (97.5th-percentile rules) from 24 source sites, repository trust scores"})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    c4_and_c5(); c2(); sites = c3_models(); c9(sites)
    for f in sorted(os.listdir(OUT)):
        if f in manifest:
            b = open(os.path.join(OUT, f), "rb").read(); manifest[f] = {**manifest[f], "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    print("saved", len(manifest), "files")
