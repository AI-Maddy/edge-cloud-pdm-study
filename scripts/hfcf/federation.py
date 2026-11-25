"""Offline federation, protocol simulation, fusion and numeric model learning."""
import hashlib
import json
import numpy as np
from .common import array, cosine, group, identifier, matrix, packet, records, weighted_mean


def assign(data, cfg):
    rows, out = records(data), []
    centroids = cfg.get("centroids")
    if centroids:
        for row in rows:
            scores = [(cosine(row[cfg.get("field", "embedding")], v), str(k)) for k, v in centroids.items()]
            score, label = max(scores)
            out.append({**row, "cell_id": label if score > cfg.get("threshold", .8) else None, "assignment_similarity": score})
    else:
        for row in rows:
            label = row.get("cluster")
            if label is None:
                raise ValueError("Supply cluster labels or config.centroids")
            # Never federate all DBSCAN/HDBSCAN outliers as one artificial cell.
            out.append({**row, "cell_id": None if int(label) == -1 else f"cell-{label}"})
    return packet(out, unassigned=sum(x["cell_id"] is None for x in out))


def register(data, cfg):
    registry = {}
    for i, row in enumerate(records(data)):
        node = str(row.get("node_id", identifier(row, i)))
        entry = {"node_id": node, "capabilities": row.get("capabilities", []), "status": "REGISTERED", "last_seen": row.get("timestamp")}
        if node in registry and registry[node]["capabilities"] != entry["capabilities"]:
            raise ValueError(f"Conflicting capabilities for node {node}")
        registry[node] = entry
    return {"records": list(registry.values()), "mode": "offline-registry", "network_discovery_performed": False}


def serialize(data, cfg):
    typ = cfg.get("object_type", "observation")
    schemas = {
        "observation": ["node_id", "timestamp"],
        "feature": ["parent_obs", "embedding", "confidence"],
        "inference": ["event", "confidence", "support_count", "evidence"],
        "knowledge": ["pattern", "confidence", "support"],
    }
    if typ not in schemas:
        raise ValueError("object_type must be observation, feature, inference or knowledge")
    out = []
    for i, row in enumerate(records(data)):
        for key in schemas[typ]:
            if key not in row:
                raise ValueError(f"{typ} object missing {key}")
        if "confidence" in row and not 0 <= float(row["confidence"]) <= 1:
            raise ValueError("confidence must lie in [0, 1]")
        digest = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
        out.append({**row, "id": row.get("id", f"{typ}-{digest}"), "object_type": typ})
    return packet(out, schema_version="hfcf-reference/1")


def publish(data, cfg):
    messages = []
    for i, row in enumerate(records(data)):
        if "node_id" not in row:
            raise ValueError("Publishing requires node_id")
        kind = "FEATURE_PUBLISH" if any(k in row for k in ("features", "embedding")) else "OBSERVATION_PUBLISH"
        messages.append({"message_id": f"message-{i}", "type": kind, "sender": row["node_id"], "payload": row})
    return {"messages": messages, "mode": "offline-outbox", "network_messages_sent": 0}


def spatial(data, cfg):
    # Correspondences must be supplied; proximity alone is not object identity.
    rows = records(data)
    key = cfg.get("group_by", "entity_id")
    field = cfg.get("field", "features")
    output = []
    for name, peers in group(rows, key).items():
        for peer in peers:
            if peer.get("coordinate_frame") != peers[0].get("coordinate_frame"):
                raise ValueError("Spatial fusion requires a shared coordinate_frame")
        weights = [float(x.get("confidence", 1)) for x in peers]
        result = {key: name, "features": weighted_mean([x[field] for x in peers], weights).tolist(), "support": len(peers), "sources": [x.get("node_id") for x in peers]}
        if all("position" in p for p in peers):
            if any("coordinate_frame" not in p for p in peers):
                raise ValueError("Positions require an explicit common coordinate_frame")
            result.update(position=weighted_mean([p["position"] for p in peers], weights).tolist(), coordinate_frame=peers[0]["coordinate_frame"])
        output.append(result)
    return packet(output, method="supplied-correspondence-confidence-weighted-fusion")


def temporal(data, cfg):
    output = []
    for name, peers in group(records(data), cfg.get("group_by", "entity_id")).items():
        peers = sorted(peers, key=lambda x: float(x["timestamp"]))
        times = array([p["timestamp"] for p in peers], 1)
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("Entity timestamps must be unique; spatially fuse simultaneous views first")
        frames = {p.get("coordinate_frame") for p in peers}
        if len(frames) != 1 or None in frames:
            raise ValueError("Tracking positions requires a shared coordinate_frame")
        positions = array([p["position"] for p in peers], 2)
        velocity = np.diff(positions, axis=0) / np.diff(times)[:, None] if len(times) > 1 else np.empty((0, positions.shape[1]))
        output.append({"entity_id": name, "times": times.tolist(), "trajectory": positions.tolist(), "velocities": velocity.tolist(), "coordinate_frame": peers[0]["coordinate_frame"]})
    return packet(output, method="supplied-identity-temporal-correlation")


def multimodal(data, cfg):
    out = []
    for key, peers in group(records(data), cfg.get("group_by", "observation_group")).items():
        modalities = sorted({p["modality"] for p in peers})
        parts = []
        for modality in modalities:
            same = [p for p in peers if p["modality"] == modality]
            parts.extend(weighted_mean([p[cfg.get("field", "features")] for p in same], [float(p.get("confidence", 1)) for p in same]).tolist())
        out.append({"id": key, "features": parts, "modalities": modalities})
    # Consumers must preserve the modality layout; not every sensor combination is compatible.
    if len({tuple(p["modalities"]) for p in out}) > 1:
        raise ValueError("All fused groups must have the same modalities; align/mask missing modalities upstream")
    return packet(out, method="per-modality-weighted-mean-then-concatenation")


def train(data, cfg):
    out = []
    for node, samples in group(records(data), "node_id").items():
        if len({s.get("cell_id") for s in samples}) > 1:
            raise ValueError("Train one node/cell combination per batch")
        x = matrix(samples, cfg.get("field", "features"))
        y = array([s["target"] for s in samples], 1)
        x = np.c_[x, np.ones(len(x))]
        weights = array(cfg.get("initial_weights", [0.] * x.shape[1]), 1)
        if weights.shape != (x.shape[1],):
            raise ValueError("initial_weights must include one coefficient per feature plus intercept")
        rate, steps = float(cfg.get("learning_rate", .01)), int(cfg.get("steps", 100))
        if rate <= 0 or steps < 1:
            raise ValueError("learning_rate and steps must be positive")
        for _ in range(steps):
            weights -= rate * 2 * x.T @ (x @ weights - y) / len(x)
            if not np.all(np.isfinite(weights)):
                raise ValueError("Training diverged; scale inputs or reduce learning_rate")
        out.append({"node_id": node, "cell_id": samples[0].get("cell_id"), "weights": weights.tolist(), "sample_count": len(samples), "model_type": "linear-regression", "training_mse": float(np.mean((x @ weights - y)**2))})
    return packet(out)


def aggregate(data, cfg):
    rows = records(data)
    if any(r.get("cell_id") is None for r in rows):
        raise ValueError("Aggregation requires assigned cell_id for every model")
    out = []
    for cell, peers in group(rows, "cell_id").items():
        types = {p.get("model_type") for p in peers}
        if len(types) != 1 or None in types:
            raise ValueError("FedAvg requires a matching explicit model_type in each cell")
        weights = [float(p["sample_count"]) for p in peers]
        out.append({"cell_id": cell, "weights": weighted_mean([p["weights"] for p in peers], weights).tolist(), "sample_count": sum(weights), "model_type": peers[0]["model_type"], "support_nodes": len(peers)})
    return packet(out, method="sample-weighted-fedavg")


def federate(data, cfg):
    rows = records(data)
    out = []
    assigned = [r for r in rows if r.get("cell_id") is not None]
    for cell, peers in group(assigned, "cell_id").items():
        out.append({"cell_id": cell, "embedding": weighted_mean([p[cfg.get("field", "embedding")] for p in peers], [float(p.get("confidence", 1)) for p in peers]).tolist(), "support": len(peers), "source_count": len({p.get("node_id", identifier(p, i)) for i, p in enumerate(peers)})})
    return packet(out, unassigned=len(rows)-len(assigned), method="cell-embedding-average")


def node_health(data, cfg):
    now = float(cfg.get("now", 0))
    if "now" not in cfg:
        raise ValueError("Set config.now to an explicit numeric observation time")
    timeout = float(cfg.get("timeout", 60))
    out = []
    for r in records(data):
        age = now - float(r["last_seen"])
        if age < 0:
            raise ValueError("last_seen cannot be later than config.now")
        out.append({**r, "age": age, "healthy": age <= timeout and r.get("status") != "ERROR"})
    return packet(out)


def drift(data, cfg):
    out = []
    for r in records(data):
        current = array(r["context_embedding"], 1)
        previous = array(r["previous_context_embedding"], 1)
        if current.shape != previous.shape:
            raise ValueError("Context drift requires matching vector dimensions and encoding schema")
        d = float(np.linalg.norm(current-previous))
        out.append({**r, "context_drift": d, "recluster": d > float(cfg.get("threshold", 1.))})
    return packet(out)


def recluster(data, cfg):
    from .features import clustering
    result = clustering(data, {**cfg, "field": cfg.get("field", "context_embedding")}, cfg.get("method", "hdbscan"))
    assigned = assign(result, {})
    previous = {identifier(r, i): r.get("cell_id") for i, r in enumerate(records(data))}
    return {**assigned, "changed": sum(r["cell_id"] != previous[identifier(r, i)] for i, r in enumerate(assigned["records"])), "method": result["method"]}
