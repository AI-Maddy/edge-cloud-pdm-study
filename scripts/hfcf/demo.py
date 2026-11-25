"""Explicitly synthetic or user-supplied numeric regression demonstrations."""
import numpy as np
from .common import array, group, packet, records
from . import features as fe, federation as fed, knowledge as kn, evaluation as ev

SCENARIOS = {"traffic":"automotive", "automotive":"automotive", "city":"city", "battlefield":"automotive", "factory":"industrial", "thermal":"industrial", "healthcare":"industrial", "agriculture":"industrial", "disaster":"city", "data-center":"industrial"}


def synthetic(domain, seed=0):
    rng=np.random.default_rng(seed)
    out=[]
    for node in range(6):
        context="wet" if node<3 else "dry"
        for sample in range(16):
            x=rng.normal(size=3)
            y=float(x@np.array([.5,-.2,.8])+(1. if node<3 else -1.)+rng.normal(scale=.1))
            out.append({"id":f"{domain}-{node}-{sample}","node_id":f"{domain}-{node}","timestamp":sample,"features":x.tolist(),"target":y,"context":{"operating_condition":context},"split":"validation" if sample>=12 else "train"})
    return packet(out,synthetic=True)


def run(data,cfg,domain):
    if cfg.get("synthetic"):
        data=synthetic(domain,int(cfg.get("seed",0)))
    rows=records(data)
    if not rows:
        raise ValueError("Supply normalized records, or use --synthetic for a labeled synthetic demonstration")
    encoded=fe.encode(packet(rows),{})
    # Exact context groups keep identical semantic labels together; no HDBSCAN claim.
    grouped=fe.categorical_clusters(encoded,{"fields":sorted(rows[0]["context"])},"environment")
    assigned=fed.assign(grouped,{})["records"]
    nodes={};history=[];cycles=int(cfg.get("cycles",1))
    if not 1<=cycles<=100:
        raise ValueError("cycles must be between 1 and 100")
    for cycle in range(cycles):
        models=[];validation={}
        for node,samples in group(assigned,"node_id").items():
            train=[r for r in samples if r.get("split")=="train"]
            val=[r for r in samples if r.get("split")=="validation"]
            if not train or not val:
                raise ValueError("Every demo node needs explicit train and validation splits")
            if len({r["cell_id"] for r in samples})!=1:
                raise ValueError("This demo expects one context/cell per node per input batch")
            tc={"steps":int(cfg.get("steps",100)),"learning_rate":float(cfg.get("learning_rate",.02))}
            if node in nodes:
                tc["initial_weights"]=nodes[node]["weights"]
            model=fed.train(packet(train),tc)["records"][0]
            validation[node]={"features":[r["features"] for r in val],"targets":[r["target"] for r in val]}
            models.append(model)
        aggregated=fed.aggregate(packet(models),{})["records"]
        knowledge=[]
        for model in aggregated:
            knowledge.append({"id":f"{model['cell_id']}-cycle-{cycle}","cell_id":model["cell_id"],"model_type":"linear-regression","target_weights":model["weights"],"support":model["sample_count"],"trust":1.,"trust_basis":"controlled-synthetic-or-local-demo-peers"})
        node_rows=[{**m,"validation":validation[m["node_id"]]} for m in models]
        messages=[{"recipient":m["node_id"],"relevance":1.,"payload":next(k for k in knowledge if k["cell_id"]==m["cell_id"])} for m in models]
        adapted=kn.adapt({"nodes":node_rows,"messages":messages},{"alpha":float(cfg.get("alpha",.5))})
        nodes={r["node_id"]:r for r in adapted["nodes"]}
        before=[];after=[];targets=[]
        for m in models:
            val=validation[m["node_id"]]
            x=np.c_[array(val["features"],2),np.ones(len(val["features"]))]
            before.extend((x@array(m["weights"],1)).tolist())
            after.extend((x@array(nodes[m["node_id"]]["weights"],1)).tolist())
            targets.extend(val["targets"])
        history.append({"cycle":cycle,"local_metrics":ev.perception({"regression_targets":targets,"regression_predictions":before},{}),"adapted_metrics":ev.perception({"regression_targets":targets,"regression_predictions":after},{}),"accepted_updates":sum(r["accepted"] for r in adapted["updates"])})
    return {"domain":domain,"synthetic":bool(data.get("synthetic",False)) if isinstance(data,dict) else False,"model":"linear-regression","grouping":"exact-context-labels","cycles":history,"nodes":list(nodes.values()),"cell_models":aggregated,"knowledge":knowledge,"limitations":["Numeric regression reference; no domain-specific perception or RUL model is run.","The same held-out validation set gates adaptation and reports validation metrics; these are not independent test results.","Trust/relevance are fixed to 1 for controlled same-cell demo peers, not empirically calibrated trust.","No network transport, causal discovery or convergence proof."]}


def scenarios(data,cfg):
    names=cfg.get("scenarios",["automotive","city","factory"])
    if any(n not in SCENARIOS for n in names):
        raise ValueError("Unknown scenario. Choose: "+", ".join(SCENARIOS))
    return {"scenarios":{name:run(data,cfg,SCENARIOS[name]) for name in names},"mode":"numeric-reference-scenarios"}
