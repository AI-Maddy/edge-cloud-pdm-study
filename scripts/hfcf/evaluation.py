"""Measured metrics and constrained selection; no claimed convergence proof."""
import json
import numpy as np
from .common import array, group, packet, records


def convergence(data, cfg):
    previous, current = data["previous"], data["current"]
    def cells(state):
        out = {}
        for r in state.get("records", []):
            if r.get("cell_id") is not None:
                out.setdefault(str(r["cell_id"]), set()).add(str(r["node_id"]))
        return out
    a, b = cells(previous), cells(current)
    drift = {}
    for name in a.keys() | b.keys():
        x,y = a.get(name,set()), b.get(name,set())
        drift[name] = len(x^y)/len(x|y) if x|y else 0.
    model_delta = {}
    old = {str(r["node_id"]): r for r in previous.get("records", [])}
    for r in current.get("records", []):
        p = old.get(str(r["node_id"]))
        if p and "weights" in p and "weights" in r:
            wa, wb = array(r["weights"],1), array(p["weights"],1)
            if wa.shape != wb.shape:
                raise ValueError("Comparable models must have the same dimensions")
            model_delta[str(r["node_id"])] = float(np.linalg.norm(wa-wb))
    variances = {}
    for cell, peers in group([r for r in current.get("records",[]) if r.get("cell_id") is not None], "cell_id").items():
        if all("weights" in r for r in peers):
            x = array([r["weights"] for r in peers],2)
            variances[cell] = float(np.mean(np.sum((x-x.mean(axis=0))**2,axis=1)))
    old_k = {json.dumps(x,sort_keys=True) for x in previous.get("knowledge",[])}
    new_k = {json.dumps(x,sort_keys=True) for x in current.get("knowledge",[])}
    stable = max(drift.values(), default=0) < float(cfg.get("membership_tolerance",.01))
    return {"membership_drift": drift, "model_change_l2": model_delta, "cell_model_variance": variances, "knowledge_symmetric_difference": len(old_k^new_k), "membership_stable_this_step": stable, "convergence_proven": False}


def stability(data, cfg):
    history = data["history"]
    if len(history)<2:
        raise ValueError("Stability evaluation needs at least two measured history states")
    fields = ["cluster_energy", "model_energy", "knowledge_energy"]
    weights = array(cfg.get("weights",[1,1,1]),1)
    if weights.shape != (3,) or np.any(weights<0):
        raise ValueError("Energy weights must be three nonnegative numbers")
    energy = array([[h[k] for k in fields] for h in history],2)@weights
    quality = array([h["quality"] for h in history],1)
    return {"energy": energy.tolist(), "energy_delta": np.diff(energy).tolist(), "quality_delta": np.diff(quality).tolist(), "nonincreasing_energy_observed": bool(np.all(np.diff(energy)<=float(cfg.get("tolerance",0)))), "quality_floor_preserved": bool(np.all(np.diff(quality)>=-float(cfg.get("max_quality_drop",0)))), "stability_proven": False}


def optimize(data, cfg):
    weights = cfg.get("weights")
    directions = cfg.get("directions")
    if not weights or not directions or set(weights)!=set(directions):
        raise ValueError("Provide matching weights and directions (+1 maximize, -1 minimize) for normalized candidate metrics")
    if any(float(w)<0 for w in weights.values()) or sum(weights.values())<=0 or any(d not in (-1,1) for d in directions.values()):
        raise ValueError("weights must be nonnegative with positive sum; directions must be +1 or -1")
    feasible, rejected = [], []
    for candidate in data["candidates"]:
        metrics = candidate["metrics"]
        failed = []
        for key,bounds in cfg.get("constraints",{}).items():
            v = float(metrics[key])
            if v < bounds.get("min",-float("inf")) or v > bounds.get("max",float("inf")):
                failed.append(key)
        score = sum(float(w)*directions[k]*float(metrics[k]) for k,w in weights.items())
        item = {**candidate,"objective_score":score,"violated_constraints":failed}
        (rejected if failed else feasible).append(item)
    feasible.sort(key=lambda r:r["objective_score"], reverse=True)
    return {"best": feasible[0] if feasible else None,"ranked_feasible":feasible,"rejected":rejected,"method":"finite-candidate-weighted-selection", "continuous_global_optimum_proven":False}


def perception(data, cfg):
    # Classification requires aligned samples/identities, not unassigned detection boxes.
    truth, predicted = data.get("y_true"), data.get("y_pred")
    result = {}
    if truth is not None or predicted is not None:
        if not truth or not isinstance(predicted,list) or len(truth)!=len(predicted):
            raise ValueError("Nonempty, aligned y_true/y_pred arrays are required")
        labels = sorted(set(truth)|set(predicted),key=str)
        per_class=[]
        for label in labels:
            tp=sum(a==label and b==label for a,b in zip(truth,predicted))
            fp=sum(a!=label and b==label for a,b in zip(truth,predicted))
            fn=sum(a==label and b!=label for a,b in zip(truth,predicted))
            precision=tp/(tp+fp) if tp+fp else 0.
            recall=tp/(tp+fn) if tp+fn else 0.
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.
            per_class.append({"label":label,"precision":precision,"recall":recall,"f1":f1,"support":sum(a==label for a in truth)})
        result.update(accuracy=sum(a==b for a,b in zip(truth,predicted))/len(truth),per_class=per_class,macro_f1=sum(r["f1"] for r in per_class)/len(labels))
    if "regression_targets" in data:
        a,b=array(data["regression_targets"],1),array(data["regression_predictions"],1)
        if a.shape!=b.shape or not a.size:
            raise ValueError("Regression arrays must be aligned and nonempty")
        result.update(rmse=float(np.sqrt(np.mean((a-b)**2))),mae=float(np.mean(np.abs(a-b))))
    if "latencies_ms" in data:
        x=array(data["latencies_ms"],1)
        if not len(x) or np.any(x<0):
            raise ValueError("latencies_ms must be nonempty and nonnegative")
        result["latency_ms"]={"mean":float(x.mean()),"p95":float(np.percentile(x,95))}
    if "raw_bytes" in data:
        raw, transmitted = float(data["raw_bytes"]),float(data["transmitted_bytes"])
        if raw<=0 or transmitted<0:
            raise ValueError("raw_bytes must be positive; transmitted_bytes nonnegative")
        result["bandwidth_reduction_fraction"]=1-transmitted/raw
    if not result:
        raise ValueError("Supply classification, regression, latency or measured bandwidth inputs")
    result["detection_map_and_tracking_metrics_computed"]=False
    return result


def utility(data, cfg):
    before, after = float(data["quality_before"]),float(data["quality_after"])
    higher = bool(cfg.get("higher_is_better",True))
    result={"quality_before":before,"quality_after":after,"utility_gain":(after-before) if higher else (before-after)}
    graph=data.get("graph")
    if graph:
        nodes=graph["nodes"];edges=graph["edges"]
        unique_nodes=len({str(n["id"]) for n in nodes})
        unique_edges=len({json.dumps(e,sort_keys=True) for e in edges})
        result["graph"]={"nodes":len(nodes),"edges":len(edges),"duplicate_nodes":len(nodes)-unique_nodes,"duplicate_edges":len(edges)-unique_edges,"edges_per_node":len(edges)/len(nodes) if nodes else 0}
    if "previous_task_quality" in data:
        result["forgetting"]=float(data["previous_task_quality"])-float(data["current_previous_task_quality"])
    return result
