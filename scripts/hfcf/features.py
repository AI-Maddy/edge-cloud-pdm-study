"""Context, representation, clustering and perception stages."""
from pathlib import Path
import numpy as np
from .common import array, cosine, group, identifier, matrix, number, optional, packet, records, weighted_mean

DATASETS = ["DAIR-V2X", "OPV2V", "V2X-Sim", "AI City Challenge", "CityFlow", "DukeMTMC", "FLIR Thermal", "FLIR ADAS", "MVTec AD", "XJTU-SY", "FEMTO/PRONOSTIA", "NASA C-MAPSS"]
CONTEXT_FIELDS = {
    "automotive": ["gps_lat", "gps_lon", "speed", "heading", "weather", "time_of_day", "road_type", "traffic_density", "camera_type", "sensor_suite"],
    "city": ["district", "zone", "camera_angle", "camera_height", "location_type", "crowd_density", "event_type", "day_type"],
    "asset": ["factory", "machine_type", "vendor", "power_rating", "environment", "production_line"],
}


def prepare(data, cfg):
    rows = records(data)
    dataset = cfg.get("dataset")
    if dataset not in DATASETS:
        raise ValueError("Set config.dataset to one of: " + ", ".join(DATASETS))
    # A user-supplied field mapping avoids guessing incompatible native dataset layouts.
    mapping = cfg.get("field_map", {})
    required = cfg.get("required_fields", ["id", "node_id", "timestamp"])
    output = []
    for i, row in enumerate(rows):
        r = {mapping.get(k, k): v for k, v in row.items()}
        missing = [k for k in required if k not in r]
        if missing:
            raise ValueError(f"Record {i} missing normalized fields: {missing}")
        for key in cfg.get("numeric_fields", []):
            r[key] = float(r[key])
        r["dataset"] = dataset
        output.append(r)
    return packet(output, dataset=dataset, adapter="normalized-records", downloads_performed=False)


def acquire(data, cfg):
    out = []
    for i, row in enumerate(records(data)):
        if "timestamp" not in row or "node_id" not in row:
            raise ValueError("Observations require timestamp and node_id")
        if not any(k in row for k in ("values", "image_path", "features", "detections", "raw_reference")):
            raise ValueError("Observation requires values, image_path, features, detections or raw_reference")
        out.append({**row, "id": identifier(row, i), "object_type": "observation"})
    return packet(out, mode="offline-replay")


def enrich(data, cfg, domain):
    output, missing = [], {}
    defaults = cfg.get("defaults", {})
    for i, row in enumerate(records(data)):
        context = {**defaults, **row.get("context", {})}
        for field in CONTEXT_FIELDS[domain]:
            if field in row:
                context[field] = row[field]
        rid = identifier(row, i)
        missing[rid] = [k for k in CONTEXT_FIELDS[domain] if k not in context]
        output.append({**row, "context": context, "domain": domain})
    return packet(output, missing_context_fields=missing)


def encode(data, cfg):
    rows = records(data)
    contexts = [r.get("context", {}) for r in rows]
    if not rows or any(not isinstance(c, dict) or not c for c in contexts):
        raise ValueError("Each record needs a nonempty context object")
    schema = cfg.get("schema")
    if schema is None:
        fields = sorted(set().union(*(c.keys() for c in contexts)))
        numeric, categories = {}, {}
        for key in fields:
            vals = [c[key] for c in contexts if key in c and c[key] is not None]
            if vals and all(number(x) is not None for x in vals):
                a = np.asarray([float(x) for x in vals])
                numeric[key] = {"mean": float(a.mean()), "scale": float(a.std()) or 1.0}
            else:
                categories[key] = sorted({str(v) for v in vals})
        schema = {"numeric": numeric, "categorical": categories}
    output = []
    for row, context in zip(rows, contexts):
        v = []
        for key, spec in schema["numeric"].items():
            val = number(context.get(key))
            # Missingness is explicit; normalized numeric value is mean-imputed.
            v.extend([0.0 if val is None else (val - spec["mean"]) / spec["scale"], float(val is None)])
        for key, levels in schema["categorical"].items():
            val = str(context[key]) if key in context and context[key] is not None else None
            v.extend(float(val == level) for level in levels)
            v.append(float(val not in levels))  # unknown/missing bucket
        output.append({**row, "context_embedding": v})
    return packet(output, schema=schema)


def categorical_clusters(data, cfg, stage):
    defaults = {
        "locations": ["road_type", "location_type"],
        "environment": ["weather", "time_of_day"],
        "traffic_and_crowds": ["traffic_density", "crowd_density"],
        "assets": ["machine_type"],
        "thermal_and_health_states": ["thermal_state", "health_state"],
    }
    fields = cfg.get("fields", defaults[stage])
    output, labels = [], {}
    for row in records(data):
        context = {**row, **row.get("context", {})}
        vals = {k: context[k] for k in fields if k in context}
        if not vals:
            raise ValueError(f"Context grouping requires at least one of {fields}; provide explicit labels or use numeric clustering")
        signature = tuple((k, str(v)) for k, v in sorted(vals.items()))
        label = labels.setdefault(signature, len(labels))
        output.append({**row, "cluster": label, stage + "_cluster": label})
    return packet(output, method="exact-context-label-grouping", fields=fields)


def local_features(data, cfg):
    if cfg.get("model"):
        result = model_bridge().record_embeddings(data, cfg, "local")
        return {**result, "records": [{**r, "features": r["embedding"], "feature_method": "local-neural-model"} for r in result["records"]]}
    output = []
    for row in records(data):
        if "features" in row:
            f = array(row["features"], 1).tolist()
            method = "supplied-features"
        elif "values" in row:
            x = array(row["values"])
            if x.size == 0:
                raise ValueError("values cannot be empty")
            f = [float(x.mean()), float(x.std()), float(x.min()), float(x.max()), float(np.sqrt(np.mean(x*x)))]
            method = "numeric-moments"
        elif "image_path" in row:
            Image = optional("PIL.Image", "Image histogram features")
            with Image.open(row["image_path"]) as image:
                x = np.asarray(image.convert("RGB").resize((64, 64)), dtype=float) / 255
            f = np.r_[x.mean(axis=(0, 1)), x.std(axis=(0, 1))].tolist()
            method = "rgb-channel-moments"
        else:
            raise ValueError("Feature extraction requires features, numeric values or image_path")
        output.append({**row, "features": f, "feature_method": method})
    return packet(output)


def train_representations(data, cfg):
    model = cfg.get("model", "linear-autoencoder")
    if model != "linear-autoencoder":
        if model.lower() not in ("autoencoder", "autoencoders", "simclr", "byol"):
            raise ValueError("Training supports linear-autoencoder, Autoencoder, SimCLR or BYOL; other backbones require external task-specific training")
        if not records(data):
            raise ValueError("Training requires nonempty records")
        return model_bridge().train_record_representations(data, cfg)
    rows = records(data)
    x = matrix(rows, cfg.get("field", "features"))
    saved = cfg.get("representation")
    if saved:
        mean, components = array(saved["mean"], 1), array(saved["components"], 2)
    else:
        mean = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        d = int(cfg.get("dimensions", min(8, *x.shape)))
        if not 1 <= d <= min(x.shape):
            raise ValueError("dimensions must be between 1 and min(samples, features)")
        components = vt[:d]
    z = (x - mean) @ components.T
    reconstructed = z @ components + mean
    return packet([{**r, "embedding": z[i].tolist()} for i, r in enumerate(rows)],
                  representation={"model": "linear-autoencoder", "mean": mean.tolist(), "components": components.tolist()},
                  reconstruction_mse=float(np.mean((x - reconstructed)**2)))


def embeddings(data, cfg, kind):
    field = cfg.get("field", kind + "_embedding")
    if cfg.get("model") and any(field not in r for r in records(data)):
        if not cfg.get("checkpoint") and not cfg.get("allow_untrained") and not cfg.get("model_config", {}).get("local_pretrained"):
            raise ValueError("Inference requires trained local weights or explicit allow_untrained=true")
        return model_bridge().record_embeddings(data, cfg, kind)
    output = []
    for row in records(data):
        if field not in row:
            raise ValueError(f"Supply precomputed '{field}' and optional embedding_model. No named backbone is silently replaced by a synthetic encoder.")
        v = array(row[field], 1).tolist()
        output.append({**row, "embedding": v, "embedding_model": row.get("embedding_model", cfg.get("model", "external-unspecified"))})
    return packet(output, mode="precomputed-embedding-import", representation=kind)


def behavior(data, cfg):
    if cfg.get("model"):
        return embeddings(data, cfg, "behavior")
    out = []
    for row in records(data):
        x = array(row["trajectory"], 2)
        if len(x) < 2:
            raise ValueError("Trajectory needs at least two positions")
        times = array(row.get("times", list(range(len(x)))), 1)
        dt = np.diff(times)
        if len(times) != len(x) or np.any(dt <= 0):
            raise ValueError("times must be strictly increasing and match trajectory length")
        velocity = np.diff(x, axis=0) / dt[:, None]
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.diff(speed) / dt[1:]
        vec = [float(speed.mean()), float(speed.std()), float(accel.mean()) if len(accel) else 0., float(np.linalg.norm(x[-1]-x[0]))]
        out.append({**row, "behavior_embedding": vec, "embedding": vec, "embedding_model": "trajectory-kinematics"})
    return packet(out)


def health(data, cfg):
    if cfg.get("model"):
        return embeddings(data, cfg, "health")
    r = local_features(data, cfg)
    return packet([{**x, "health_embedding": x["features"], "embedding": x["features"], "embedding_model": x["feature_method"]} for x in r["records"]])


def detect(data, cfg):
    rows = records(data)
    if all("detections" in r for r in rows):
        for row in rows:
            if not isinstance(row["detections"], list):
                raise ValueError("detections must be a list")
        return packet(rows, mode="precomputed-detections")
    if cfg.get("checkpoint") or cfg.get("model_config") or cfg.get("allow_untrained"):
        return model_bridge().record_predictions(data, cfg)
    name, weights = cfg.get("model"), cfg.get("weights")
    if name not in ("YOLO", "YOLOv11", "RT-DETR") or not weights or not Path(weights).is_file():
        raise ValueError("Supply precomputed detections, or config.model YOLO/YOLOv11/RT-DETR with an existing local weights file. BEVFusion/PointPillars/CenterPoint require external inference.")
    package = optional("ultralytics", "YOLO/RT-DETR inference")
    model = (package.RTDETR if name == "RT-DETR" else package.YOLO)(weights)
    out = []
    for row in rows:
        result = model.predict(source=row["image_path"], verbose=False)[0]
        detections = [{"bbox": box.xyxy[0].tolist(), "confidence": float(box.conf[0]), "class_id": int(box.cls[0])} for box in result.boxes]
        out.append({**row, "detections": detections, "detection_model": name})
    return packet(out, mode="local-weights-inference")


def similarity_graph(data, cfg):
    rows = records(data)
    x = matrix(rows, cfg.get("field", "embedding"))
    threshold = float(cfg.get("threshold", .85))
    edges = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = cosine(x[i], x[j])
            if score > threshold:
                edges.append({"source": identifier(rows[i], i), "target": identifier(rows[j], j), "weight": score})
    return {"records": rows, "nodes": [{"id": identifier(r, i)} for i, r in enumerate(rows)], "edges": edges, "threshold": threshold}


def clustering(data, cfg, kind):
    if kind not in ("hdbscan", "dbscan", "kmeans", "spectral"):
        raise ValueError("method must be hdbscan, dbscan, kmeans or spectral")
    rows = records(data)
    x = matrix(rows, cfg.get("field", "context_embedding" if kind == "hdbscan" else "embedding"))
    sk = optional("sklearn.cluster", "Clustering")
    if kind == "hdbscan":
        if not hasattr(sk, "HDBSCAN"):
            raise ValueError("HDBSCAN requires scikit-learn >= 1.3")
        model = sk.HDBSCAN(min_cluster_size=int(cfg.get("min_cluster_size", 5)), min_samples=cfg.get("min_samples"))
    elif kind == "dbscan":
        model = sk.DBSCAN(eps=float(cfg.get("eps", .5)), min_samples=int(cfg.get("min_samples", 5)))
    elif kind == "kmeans":
        model = sk.KMeans(n_clusters=int(cfg.get("clusters", 2)), random_state=int(cfg.get("seed", 0)), n_init=10)
    else:
        k = int(cfg.get("clusters", 2))
        if not 1 <= k <= len(rows):
            raise ValueError("clusters must be between 1 and number of records")
        model = sk.SpectralClustering(n_clusters=k, affinity="rbf", random_state=int(cfg.get("seed", 0)), assign_labels="kmeans")
    labels = model.fit_predict(x)
    return packet([{**r, "cluster": int(labels[i])} for i, r in enumerate(rows)], method=kind)


def scene_clusters(data, cfg):
    graph = data if "edges" in data else similarity_graph(data, cfg)
    ids = [identifier(r, i) for i, r in enumerate(records(graph))]
    if len(ids) != len(set(ids)):
        raise ValueError("Graph node ids must be unique")
    method = cfg.get("method", "louvain")
    if method == "leiden":
        ig, la = optional("igraph", "Leiden"), optional("leidenalg", "Leiden")
        indexes = {node: i for i, node in enumerate(ids)}
        edges = [(indexes[e["source"]], indexes[e["target"]]) for e in graph["edges"]]
        g = ig.Graph(n=len(ids), edges=edges, directed=False)
        part = la.find_partition(g, la.ModularityVertexPartition, weights=[e["weight"] for e in graph["edges"]], seed=int(cfg.get("seed", 0)))
        labels = {ids[i]: k for k, cell in enumerate(part) for i in cell}
    elif method == "louvain":
        nx = optional("networkx", "Louvain")
        g = nx.Graph()
        g.add_nodes_from(ids)
        g.add_weighted_edges_from((e["source"], e["target"], e["weight"]) for e in graph["edges"])
        groups = nx.community.louvain_communities(g, seed=int(cfg.get("seed", 0))) if g.number_of_edges() else [{i} for i in ids]
        labels = {node: k for k, cell in enumerate(groups) for node in cell}
    else:
        raise ValueError("method must be leiden or louvain")
    return {**graph, "records": [{**r, "cluster": labels[identifier(r, i)]} for i, r in enumerate(records(graph))], "method": method}


def combine(data, cfg):
    # Concatenation supports different feature dimensions without invalid elementwise sums.
    weights = cfg.get("weights", {"features": .5, "context_embedding": .5})
    if not weights or any(float(v) < 0 for v in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Embedding weights must be nonnegative and sum to a positive value")
    out = []
    for row in records(data):
        parts = []
        for field, weight in weights.items():
            v = array(row[field], 1)
            norm = np.linalg.norm(v)
            parts.append(float(weight) * (v / norm if norm else v))
        out.append({**row, "embedding": np.concatenate(parts).tolist()})
    return packet(out, method="weighted-normalized-concatenation", weights=weights)


def model_bridge():
    """Load the sibling model package only when neural inference/training is requested."""
    import sys
    root = str(Path(__file__).resolve().parents[2] / "models")
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from hfcf_models import bridge
    except ImportError as exc:
        raise ValueError("Neural model integration requires models/requirements.txt") from exc
    return bridge
