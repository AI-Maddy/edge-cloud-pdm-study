"""C7 validation: HFIP upstream message accounting over recorded cellular uplinks (note 05 protocol objects and
flows; note 01 'Feature-Level Federation'; repository scripts/hfcf/federation.py serialize()/publish()).

One edge site = 20 engines (C-MAPSS FD001 units). One reporting period = 30 operating cycles.
Upstream strategies (objects built and sized with the repository serializer, JSON as implemented, and a compact
binary encoding for comparison):
  raw          OBSERVATION per engine-cycle (21 sensors + 3 settings)
  feature      FEATURE per engine-window: 32-d embedding from the edge Transformer (C2)
  inference    INFERENCE per engine-window: RUL estimate + confidence
  model        MODEL_UPDATE per period: edge Transformer weights (C2 size) - federated learning upload
  knowledge    KNOWLEDGE objects: top-20 rules (C5)
Delivery time over 5 recorded uplinks (Mahimahi: Verizon/T-Mobile/AT&T LTE, T-Mobile UMTS, Verizon EVDO, all
driving), 200 random start offsets each: median and 95th percentile seconds to deliver one period's upload.
Accuracy side: the cloud predicts RUL from what it receives (raw -> full model; feature -> ridge head on
embeddings; inference -> as is).
Tests (fixed before running):
  T1 feature objects cut upstream bytes >= 10x vs raw observations (binary encoding), as note 01 claims
  T2 cloud RUL error from features within 10% of the error from raw data (FD001 test, RMSE)
  T3 a model-update upload completes within 10 s at the 95th percentile on every trace
"""
import gzip, json, os, sys
import numpy as np
import torch
from sklearn.linear_model import Ridge
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from common import rul
from hfcf import federation
torch.set_num_threads(2)
TR = os.path.join(HERE, "..", "data", "traces")
TRACES = ["Verizon-LTE-driving", "TMobile-LTE-driving", "ATT-LTE-driving", "TMobile-UMTS-driving", "Verizon-EVDO-driving"]
ENG, PER = 20, 30


def jbytes(objs): return sum(len(json.dumps(o, separators=(",", ":")).encode()) for o in objs)


def delivery_seconds(trace_ms, nbytes, g, reps=200):
    """Mahimahi semantics: each line is a ms timestamp at which one 1500-byte packet may leave."""
    n = int(np.ceil(nbytes / 1500)); T = trace_ms[-1]; out = []
    for _ in range(reps):
        t0 = g.uniform(0, T); k = np.searchsorted(trace_ms, t0); loops = 0
        idx = k + n - 1
        while idx >= len(trace_ms):
            idx -= len(trace_ms); loops += 1
        out.append((trace_ms[idx] + loops * T - t0) / 1000.0)
    return float(np.median(out)), float(np.percentile(out, 95))


def main():
    g = np.random.default_rng(0)
    Xtr, ytr, utr, Xte, yte, _ = rul.load_fd("FD001")
    model, _ = rul.train(rul.make("transformer", Xtr.shape[2]), Xtr, ytr, utr, seed=0)
    nparam = sum(p.numel() for p in model.parameters())
    p_raw = rul.predict(model, Xte)
    with torch.no_grad():
        emb = lambda X: model.enc(torch.tensor(X)).numpy()
        Etr, Ete = emb(Xtr), emb(Xte)
    head = Ridge(1.0).fit(Etr, ytr)
    p_feat = head.predict(Ete).clip(0, rul.CAP)
    acc = {"rmse_cloud_from_raw_model": rul.rmse(p_raw, yte), "rmse_cloud_from_features": rul.rmse(p_feat, yte),
           "embedding_dim": int(Etr.shape[1]), "edge_model_params": int(nparam)}

    # one period of objects for one site, via the repository serializer
    raw = [{"node_id": f"E{e}", "timestamp": f"c{c}", "values": [round(float(v), 4) for v in g.normal(size=24)]} for e in range(ENG) for c in range(PER)]
    feat = [{"node_id": f"E{e}", "parent_obs": f"E{e}-w", "embedding": [round(float(v), 4) for v in Ete[e % len(Ete)]], "confidence": 0.9} for e in range(ENG)]
    inf = [{"node_id": f"E{e}", "event": "rul_estimate", "value": round(float(p_raw[e % len(p_raw)]), 1), "confidence": 0.9, "support_count": PER, "evidence": []} for e in range(ENG)]
    know = [{"pattern": f"s{11+i%5}>1.{i}", "conditions": [f"s{11+i%5}>1.{i}"], "outcome": "failure within 30 cycles", "confidence": 0.95, "support": 120} for i in range(20)]
    sizes = {
        "raw": {"json": jbytes(federation.serialize({"records": raw}, {"object_type": "observation"})["records"]), "binary": ENG * PER * (24 * 4 + 8)},
        "feature": {"json": jbytes(federation.serialize({"records": feat}, {"object_type": "feature"})["records"]), "binary": ENG * (Etr.shape[1] * 2 + 8)},
        "inference": {"json": jbytes(federation.serialize({"records": inf}, {"object_type": "inference"})["records"]), "binary": ENG * (4 + 2 + 8)},
        "model": {"json": None, "binary": nparam * 4 + 64},
        "knowledge": {"json": jbytes(federation.serialize({"records": know}, {"object_type": "knowledge"})["records"]), "binary": 20 * 24},
    }
    traces = {}
    for t in TRACES:
        with gzip.open(os.path.join(TR, t + ".up.gz"), "rt") as f:
            ms = np.array([int(x) for x in f.read().split()], np.int64)
        traces[t] = {"mean_mbps": len(ms) * 1500 * 8 / (ms[-1] / 1000) / 1e6}
        for k, v in sizes.items():
            for enc in ("json", "binary"):
                if v[enc] is not None:
                    traces[t][f"{k}_{enc}_s_median_p95"] = delivery_seconds(ms, v[enc], g)
    res = {"accuracy": acc, "bytes_per_site_period": sizes, "delivery": traces,
           "verdict": {"T1": sizes["raw"]["binary"] / sizes["feature"]["binary"] >= 10,
                       "T2": acc["rmse_cloud_from_features"] <= 1.10 * acc["rmse_cloud_from_raw_model"],
                       "T3": all(traces[t]["model_binary_s_median_p95"][1] <= 10 for t in TRACES)}}
    json.dump(res, open(os.path.join(HERE, "results", "c7_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
