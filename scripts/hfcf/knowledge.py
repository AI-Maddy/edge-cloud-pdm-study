"""Knowledge objects, graph operations, explicit trust/relevance and adaptation gates."""
import hashlib
import json
import numpy as np
from .common import array, cosine, group, identifier, packet, records


def generate(data, cfg):
    out = []
    for i, row in enumerate(records(data)):
        if "embedding" not in row or "support" not in row:
            raise ValueError("Knowledge generation needs a cell embedding and observed support")
        # A statistical prototype is not an automatically discovered semantic/causal rule.
        out.append({**row, "id": row.get("id", f"knowledge-{i}"), "pattern": row.get("pattern", "cell-prototype"), "kind": "prototype", "confidence": row.get("confidence"), "embedding": array(row["embedding"], 1).tolist()})
    return packet(out, method="prototype-packaging", semantic_rule_mining_performed=False)


def regional(data, cfg):
    rows = records(data)
    nodes = [{"id": identifier(r, i), "kind": r.get("kind", "knowledge"), "payload": r} for i, r in enumerate(rows)]
    if len({x["id"] for x in nodes}) != len(nodes):
        raise ValueError("Knowledge ids must be unique")
    edges = []
    for i, a in enumerate(rows):
        for j in range(i+1, len(rows)):
            b = rows[j]
            score = cosine(a["embedding"], b["embedding"])
            if score > float(cfg.get("threshold", .8)):
                edges.append({"source": nodes[i]["id"], "target": nodes[j]["id"], "relation": "similar-prototype", "weight": score})
    return {"nodes": nodes, "edges": edges, "records": rows, "causal_relations_inferred": False}


def promote(data, cfg):
    selected, rejected = [], []
    for r in records(data):
        if r.get("confidence") is None:
            rejected.append({**r, "reason": "missing-confidence"})
        elif float(r["support"]) > float(cfg.get("min_support", 1000)) and float(r["confidence"]) > float(cfg.get("min_confidence", .9)):
            selected.append({**r, "promoted": True})
        else:
            rejected.append({**r, "reason": "promotion-threshold"})
    return packet(selected, rejected=rejected)


def fuse(data, cfg):
    nodes, edges = {}, {}
    for graph in data.get("graphs", []):
        for node in graph["nodes"]:
            key = str(node["id"])
            if key in nodes and nodes[key] != node:
                raise ValueError(f"Conflicting knowledge node {key}; resolve/version it explicitly")
            nodes[key] = node
        for edge in graph["edges"]:
            key = json.dumps(edge, sort_keys=True)
            edges[key] = edge
    if not data.get("graphs"):
        raise ValueError("Supply graphs: [{nodes: [...], edges: [...]}]")
    if any(e["source"] not in nodes or e["target"] not in nodes for e in edges.values()):
        raise ValueError("Graph edges must reference existing node ids")
    return {"nodes": list(nodes.values()), "edges": list(edges.values()), "records": [n.get("payload", n) for n in nodes.values()]}


def mine(data, cfg):
    counts = {}
    rows = records(data)
    for r in rows:
        key = str(r[cfg.get("field", "pattern")])
        counts[key] = counts.get(key, 0) + 1
    return {"patterns": [{"pattern": k, "count": v, "frequency": v/len(rows)} for k,v in sorted(counts.items())], "rare_patterns": [k for k,v in counts.items() if v < int(cfg.get("min_support", 2))], "method": "observed-pattern-frequency", "causal_discovery_performed": False}


def trust(data, cfg):
    weights = cfg.get("weights", {"confidence": .4, "support_score": .2, "quality": .2, "source_reliability": .2})
    if any(float(w) < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Trust weights must be nonnegative with positive sum")
    if float(cfg.get("support_scale", 1000)) <= 0:
        raise ValueError("support_scale must be positive")
    out = []
    for r in records(data):
        scores = {}
        for name in weights:
            if name == "support_score":
                scores[name] = min(1., float(r["support"])/float(cfg.get("support_scale", 1000)))
            else:
                if r.get(name) is None:
                    raise ValueError(f"Trust scoring requires measured '{name}'")
                scores[name] = float(r[name])
            if not 0 <= scores[name] <= 1:
                raise ValueError(f"Trust component {name} must lie in [0,1]")
        score = sum(float(weights[k])*v for k,v in scores.items())/sum(weights.values())
        out.append({**r, "trust": score, "trust_components": scores})
    return packet(out)


def relevance(data, cfg):
    nodes, knowledge = data.get("nodes", []), data.get("knowledge", [])
    out = []
    weights = cfg.get("weights", {"embedding": 1.})
    if not weights or any(float(w) < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("Relevance weights must be nonnegative with positive sum")
    for i, node in enumerate(nodes):
        for j, k in enumerate(knowledge):
            score = sum(float(w)*cosine(node[field], k[field]) for field,w in weights.items())/sum(weights.values())
            out.append({"node_id": identifier(node,i), "knowledge_id": identifier(k,j), "relevance": score, "knowledge": k})
    return packet(out)


def disseminate(data, cfg):
    messages, rejected = [], []
    for row in records(data):
        k = row["knowledge"]
        if row["relevance"] > float(cfg.get("min_relevance", .8)) and float(k.get("trust", 0)) > float(cfg.get("min_trust", .8)):
            messages.append({"type": "ADAPTED_KP", "parent_type": "KNOWLEDGE_BROADCAST", "recipient": row["node_id"], "relevance": row["relevance"], "payload": k})
        else:
            rejected.append(row)
    return {"messages": messages, "rejected": rejected, "mode": "offline-outbox", "network_messages_sent": 0}


def adapt(data, cfg):
    nodes = {str(n["node_id"]): dict(n) for n in data.get("nodes", [])}
    updates = []
    alpha = float(cfg.get("alpha", .1))
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0,1]")
    for m in data.get("messages", []):
        node = nodes[str(m["recipient"])]
        k = m["payload"]
        trust_score = float(k.get("trust", 0))
        rel = float(m.get("relevance", 0))
        outcome = {"node_id": node["node_id"], "knowledge_id": k.get("id")}
        if not trust_score > cfg.get("min_trust", .8) or not rel > cfg.get("min_relevance", .8):
            updates.append({**outcome, "accepted": False, "reason": "trust-or-relevance"})
            continue
        # Knowledge must carry a compatible model target: never add a knowledge object to weights.
        if "target_weights" not in k or k.get("model_type") != node.get("model_type"):
            updates.append({**outcome, "accepted": False, "reason": "missing-or-incompatible-model-target"})
            continue
        old, target = array(node["weights"],1), array(k["target_weights"],1)
        if old.shape != target.shape:
            raise ValueError("Knowledge and local model weight dimensions differ")
        validation = node.get("validation")
        if not validation:
            updates.append({**outcome, "accepted": False, "reason": "missing-held-out-validation"})
            continue
        if node.get("model_type") != "linear-regression":
            raise ValueError("Built-in adaptation validation supports linear-regression only")
        x, y = array(validation["features"],2), array(validation["targets"],1)
        if not len(x) or len(x) != len(y):
            raise ValueError("Validation features and targets must be nonempty and aligned")
        x = np.c_[x, np.ones(len(x))]
        candidate = old + alpha*rel*trust_score*(target-old)
        before, after = float(np.mean((x@old-y)**2)), float(np.mean((x@candidate-y)**2))
        accepted = after < before
        if accepted:
            node["weights"] = candidate.tolist()
        updates.append({**outcome, "accepted": accepted, "validation_mse_before": before, "validation_mse_after": after})
    return {"nodes": list(nodes.values()), "updates": updates, "method": "validation-gated-compatible-weight-interpolation"}


def cross_domain(data, cfg):
    projections = cfg.get("projections")
    target_domain = cfg.get("target_domain")
    if not projections or not target_domain:
        raise ValueError("Cross-domain transfer requires explicit per-domain projection matrices and target_domain; alignment is not assumed")
    out = []
    for r in records(data):
        p = array(projections[r["domain"]],2)
        embedding = array(r["embedding"],1)@p
        out.append({**r,"embedding": embedding.tolist(), "source_domain": r["domain"], "domain": target_domain})
    if len({len(r["embedding"]) for r in out}) > 1:
        raise ValueError("Projection matrices must map into the same target dimension")
    return packet(out, method="supplied-linear-domain-alignment", alignment_learned=False)
