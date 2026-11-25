"""JSON contracts and deterministic numerical helpers; no network or implicit downloads."""
import csv
import json
import math
from pathlib import Path
import numpy as np


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        if Path(path).suffix == ".jsonl":
            return {"records": [json.loads(line) for line in f if line.strip()]}
        if Path(path).suffix == ".csv":
            return {"records": list(csv.DictReader(f))}
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def records(data):
    value = data if isinstance(data, list) else data.get("records", [])
    if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
        raise ValueError("records must be a list of JSON objects")
    return value


def array(value, ndim=None):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError("Numeric data must be finite")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional numeric array")
    return a


def matrix(rows, field):
    if not rows or any(field not in x for x in rows):
        raise ValueError(f"Nonempty records with '{field}' are required")
    a = array([x[field] for x in rows], 2)
    if not a.shape[1]:
        raise ValueError("Empty vectors are not supported")
    return a


def identifier(row, i=0):
    return str(row.get("id", row.get("node_id", f"record-{i}")))


def cosine(a, b):
    a, b = array(a, 1), array(b, 1)
    if a.shape != b.shape:
        raise ValueError("Cosine similarity requires equal vector dimensions")
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else 0.0


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def packet(rows, **extra):
    return {"records": rows, **extra}


def group(rows, key):
    result = {}
    for row in rows:
        if key not in row:
            raise ValueError(f"Missing grouping field: {key}")
        result.setdefault(str(row[key]), []).append(row)
    return result


def weighted_mean(vectors, weights):
    x, w = array(vectors, 2), array(weights, 1)
    if len(x) != len(w) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be nonnegative, have matching length, and sum to a positive value")
    return np.average(x, weights=w, axis=0)


def optional(module, purpose):
    from importlib import import_module
    try:
        return import_module(module)
    except ImportError as e:
        raise ValueError(f"{purpose} needs optional dependency '{module}'. See requirements-optional.txt.") from e
