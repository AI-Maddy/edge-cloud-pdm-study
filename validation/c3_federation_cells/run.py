"""C3 validation: context-aware federation cells (CAFIG; notes 07, 08, 10 Level-2, 12 sec. 7 and 9).

Clients are sites (fleets) of turbofan engines built from all four C-MAPSS subsets:
  FD001 sea level / HPC fault (5 sites), FD002 six conditions / HPC (13), FD003 sea level / HPC+fan (5),
  FD004 six conditions / HPC+fan (12)  ->  35 sites, ~20 engines each, official test engines split likewise.
Site context vector = histogram of the site's operating-condition cells (from C1), as the notes propose
(environment context, no labels). Cells: K-Means on site contexts, k by silhouette.
Methods (same CNN1D, same local optimiser, 25 rounds x 1 local epoch; local-only gets 25 epochs):
  local, fedavg, fedavg_ft (1 local fine-tune epoch), ifca (k=4, loss-based cluster choice; Ghosh 2020),
  cafig (FedAvg inside context cells), cafig_ft, oracle (FedAvg inside the true subset; reference only).
Tests (fixed before running):
  T1 cafig beats fedavg on mean site test RMSE (paired over seeds and sites)
  T2 cafig is within 1 RMSE of ifca while downloading 1 model per round instead of k
  T3 cafig is within 1 RMSE of the oracle grouping
"""
import json, os, sys, time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import cmapss, rul
torch.set_num_threads(4)

ROUNDS, BS, LR = 25, 256, 1e-3
SITES = {"FD001": 5, "FD002": 13, "FD003": 5, "FD004": 12}


def build_sites():
    trs = {fd: cmapss.load(fd, "train") for fd in SITES}
    tes = {fd: cmapss.load(fd, "test") for fd in SITES}
    allset = np.vstack([trs[fd]["arr"][:, 2:5] for fd in SITES])
    allsen = np.vstack([trs[fd]["arr"][:, 5:][:, rul.SENS] for fd in SITES])
    norm = rul.Normaliser(context=True, k=6).fit(allset, allsen)   # = federated per-cell mean/var
    sites = []
    for fd, n in SITES.items():
        tr, te = trs[fd], tes[fd]
        ztr, ctr = norm.transform(tr["arr"][:, 2:5], tr["arr"][:, 5:][:, rul.SENS])
        zte, _ = norm.transform(te["arr"][:, 2:5], te["arr"][:, 5:][:, rul.SENS])
        Xtr, ytr, utr = rul.windows(tr["arr"], ztr, tr["rul"])
        Xte, _, ute = rul.windows(te["arr"], zte, last_only=True)
        yte = np.minimum(te["rul_end"], rul.CAP).astype(np.float32)
        tr_units, te_units = np.unique(tr["arr"][:, 0]), np.unique(te["arr"][:, 0])
        for j in range(n):
            mu = np.isin(utr, tr_units[j::n]); mt = np.isin(ute, te_units[j::n])
            rows = np.isin(tr["arr"][:, 0], tr_units[j::n])
            ctx = np.bincount(ctr[rows], minlength=6) / rows.sum()
            sites.append(dict(fd=fd, Xtr=Xtr[mu], ytr=ytr[mu], Xte=Xte[mt], yte=yte[mt], ctx=ctx))
    return sites


def get(m): return [p.detach().clone() for p in m.parameters()]
def put(m, w):
    with torch.no_grad():
        for p, q in zip(m.parameters(), w): p.copy_(q)
def avg(ws, ns):
    t = float(sum(ns)); return [sum(w[i] * (n / t) for w, n in zip(ws, ns)) for i in range(len(ws[0]))]


def local_epoch(model, X, y, g, epochs=1):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    model.train()
    for _ in range(epochs):
        perm = torch.tensor(g.permutation(len(Xt)))
        for i in range(0, len(Xt), BS):
            b = perm[i:i + BS]; opt.zero_grad()
            ((model(Xt[b]) - yt[b]) ** 2).mean().backward(); opt.step()


def loss_on(model, X, y):
    model.eval()
    with torch.no_grad():
        return float(((model(torch.tensor(X)) - torch.tensor(y)) ** 2).mean())


def evaluate(models_for_site, sites):
    out = []
    for m, s in zip(models_for_site, sites):
        out.append(rul.rmse(rul.predict(m, s["Xte"]), s["yte"]))
    return out


def run(method, sites, seed, groups=None, k_ifca=4):
    g = np.random.default_rng(seed); torch.manual_seed(seed)
    F = sites[0]["Xtr"].shape[2]
    proto = rul.CNN1D(F)
    w0 = get(proto)
    down = 0
    nparam = sum(p.numel() for p in proto.parameters())
    if method == "local":
        models = []
        for s in sites:
            m = rul.CNN1D(F); put(m, w0); local_epoch(m, s["Xtr"], s["ytr"], g, epochs=ROUNDS); models.append(m)
        return evaluate(models, sites), {"down_models_per_round": 0, "up_models_per_round": 0}
    if method in ("fedavg", "fedavg_ft"):
        groups = np.zeros(len(sites), int)
    if method == "ifca":
        # k models, different random inits (Ghosh et al. 2020)
        cw = []
        for c in range(k_ifca):
            torch.manual_seed(seed * 100 + c); cw.append(get(rul.CNN1D(F)))
    else:
        cw = [w0 for _ in range(groups.max() + 1)]
    work = rul.CNN1D(F)
    for r in range(ROUNDS):
        if method == "ifca":
            assign = []
            for s in sites:
                ls = []
                for w in cw:
                    put(work, w); ls.append(loss_on(work, s["Xtr"][:512], s["ytr"][:512]))
                assign.append(int(np.argmin(ls)))
            assign = np.array(assign); down += k_ifca
        else:
            assign = groups; down += 1
        new = []
        for c in range(len(cw)):
            idx = np.where(assign == c)[0]
            if len(idx) == 0:
                new.append(cw[c]); continue
            ws, ns = [], []
            for i in idx:
                put(work, cw[c]); local_epoch(work, sites[i]["Xtr"], sites[i]["ytr"], g); ws.append(get(work)); ns.append(len(sites[i]["ytr"]))
            new.append(avg(ws, ns))
        cw = new
    models = []
    for i, s in enumerate(sites):
        m = rul.CNN1D(F)
        if method == "ifca":
            ls = []
            for w in cw:
                put(m, w); ls.append(loss_on(m, s["Xtr"], s["ytr"]))
            put(m, cw[int(np.argmin(ls))])
        else:
            put(m, cw[groups[i]])
        if method.endswith("_ft"):
            local_epoch(m, s["Xtr"], s["ytr"], g)
        models.append(m)
    return evaluate(models, sites), {"down_models_per_round": down / ROUNDS, "up_models_per_round": 1,
                                     "model_bytes": 4 * nparam}


def main():
    sites = build_sites()
    C = np.array([s["ctx"] for s in sites])
    sil = {k: float(silhouette_score(C, KMeans(k, n_init=10, random_state=0).fit_predict(C))) for k in range(2, 7)}
    k = max(sil, key=sil.get)
    cafig = KMeans(k, n_init=10, random_state=0).fit_predict(C)
    fds = np.array([s["fd"] for s in sites])
    oracle = np.unique(fds, return_inverse=True)[1]
    meta = {"n_sites": len(sites), "site_fd": fds.tolist(), "silhouette": sil, "cafig_k": k,
            "cafig_cells": cafig.tolist(), "train_windows": int(sum(len(s["ytr"]) for s in sites)),
            "test_engines": int(sum(len(s["yte"]) for s in sites))}
    print(meta, flush=True)
    path = os.path.join(HERE, "results", "c3_runs.json")
    res = json.load(open(path)) if os.path.exists(path) else {"meta": meta, "runs": []}
    done = {(r["method"], r["seed"]) for r in res["runs"]}
    for seed in (0, 1, 2):
        for method, grp in [("local", None), ("fedavg", None), ("fedavg_ft", None), ("ifca", None),
                            ("cafig", cafig), ("cafig_ft", cafig), ("oracle", oracle)]:
            if (method, seed) in done:
                continue
            t = time.time()
            site_rmse, cost = run(method, sites, seed, groups=None if grp is None else np.asarray(grp))
            r = {"method": method, "seed": seed, "site_rmse": site_rmse, "mean_site_rmse": float(np.mean(site_rmse)),
                 "by_fd": {fd: float(np.mean([x for x, f in zip(site_rmse, fds) if f == fd])) for fd in SITES},
                 **cost, "seconds": round(time.time() - t, 1)}
            res["runs"].append(r)
            print({k_: v for k_, v in r.items() if k_ != "site_rmse"}, flush=True)
            json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
