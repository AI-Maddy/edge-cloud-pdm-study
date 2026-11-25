"""Uniform command-line interface for every inventory entry point."""
import argparse
import json
import sys
import numpy as np
from .common import read_json, write_json
from . import features as f, federation as r, knowledge as k, evaluation as e, demo as d


def context(domain):
    return lambda data,cfg:f.enrich(data,cfg,domain)


def grouping(stage):
    return lambda data,cfg:(f.clustering(data,cfg,"kmeans") if cfg.get("method")=="kmeans" else f.categorical_clusters(data,cfg,stage))


def embedding(kind):
    return lambda data,cfg:f.embeddings(data,cfg,kind)


COMMANDS = {
    "prepare_datasets":f.prepare,
    "acquire_observations":f.acquire,
    "enrich_automotive_context":context("automotive"),
    "enrich_city_context":context("city"),
    "enrich_asset_context":context("asset"),
    "cluster_locations":grouping("locations"),
    "cluster_environment":grouping("environment"),
    "cluster_traffic_and_crowds":grouping("traffic_and_crowds"),
    "cluster_assets":grouping("assets"),
    "cluster_thermal_and_health_states":grouping("thermal_and_health_states"),
    "encode_context":f.encode,
    "extract_local_features":f.local_features,
    "train_sensor_representations":f.train_representations,
    "detect_objects":f.detect,
    "extract_scene_embeddings":embedding("scene"),
    "extract_object_embeddings":embedding("object"),
    "extract_behavior_embeddings":f.behavior,
    "extract_health_embeddings":f.health,
    "build_similarity_graph":f.similarity_graph,
    "cluster_context_hdbscan":lambda data,cfg:f.clustering(data,cfg,"hdbscan"),
    "cluster_scenes":f.scene_clusters,
    "cluster_objects_dbscan":lambda data,cfg:f.clustering(data,cfg,"dbscan"),
    "cluster_behaviors_spectral":lambda data,cfg:f.clustering(data,cfg,"spectral"),
    "combine_cognitive_embeddings":f.combine,
    "assign_federation_cells":r.assign,
    "discover_and_register_nodes":r.register,
    "serialize_protocol_objects":r.serialize,
    "publish_observations_and_features":r.publish,
    "fuse_spatial_observations":r.spatial,
    "correlate_temporal_observations":r.temporal,
    "fuse_multimodal_features":r.multimodal,
    "train_local_models":r.train,
    "aggregate_cell_models":r.aggregate,
    "federate_features":r.federate,
    "generate_cell_knowledge":k.generate,
    "build_regional_graphs":k.regional,
    "promote_knowledge":k.promote,
    "fuse_global_knowledge":k.fuse,
    "mine_global_patterns":k.mine,
    "score_knowledge_trust":k.trust,
    "score_knowledge_relevance":k.relevance,
    "disseminate_knowledge":k.disseminate,
    "consume_and_adapt_knowledge":k.adapt,
    "transfer_cross_domain_knowledge":k.cross_domain,
    "monitor_node_health":r.node_health,
    "monitor_context_drift":r.drift,
    "recluster_federation_cells":r.recluster,
    "measure_convergence":e.convergence,
    "measure_cognitive_stability":e.stability,
    "optimize_hierarchy":e.optimize,
    "evaluate_perception_and_systems":e.perception,
    "evaluate_knowledge_utility":e.utility,
    "run_automotive_demo":lambda data,cfg:d.run(data,cfg,"automotive"),
    "run_city_demo":lambda data,cfg:d.run(data,cfg,"city"),
    "run_industrial_demo":lambda data,cfg:d.run(data,cfg,"industrial"),
    "run_application_scenarios":d.scenarios,
    "run_cognitive_feedback_loop":lambda data,cfg:d.run(data,cfg,cfg.get("domain","automotive")),
}


def main(command):
    parser=argparse.ArgumentParser(description=f"HFCF reference stage: {command}. See scripts/README.md for its input/config contract.")
    parser.add_argument("--input",help="JSON, JSONL or normalized CSV input; required except explicit synthetic demos")
    parser.add_argument("--output",required=True,help="New result JSON file; use --overwrite to replace an existing result")
    parser.add_argument("--config",help="JSON configuration object")
    parser.add_argument("--model",help="Set config.model; named models need explicit adapters/weights or precomputed outputs")
    parser.add_argument("--seed",type=int,default=None,help="Set config.seed")
    parser.add_argument("--synthetic",action="store_true",help="Generate labeled synthetic data (demo entry points only)")
    parser.add_argument("--overwrite",action="store_true",help="Allow replacing the output file")
    args=parser.parse_args()
    try:
        from pathlib import Path
        if Path(args.output).exists() and not args.overwrite:
            raise ValueError("Output already exists; choose another path or pass --overwrite")
        if args.input and Path(args.input).resolve()==Path(args.output).resolve():
            raise ValueError("Input and output must be different files")
        cfg=read_json(args.config) if args.config else {}
        if not isinstance(cfg,dict):
            raise ValueError("Config must be a JSON object")
        if args.seed is not None:
            cfg["seed"]=args.seed
        if args.model:
            cfg["model"]=args.model
        if args.synthetic:
            cfg["synthetic"]=True
        if cfg.get("synthetic") and not command.startswith("run_"):
            raise ValueError("--synthetic is available only for run_* demo commands")
        if cfg.get("synthetic") and args.input:
            raise ValueError("Choose input records or --synthetic, not both")
        if not args.input and not cfg.get("synthetic"):
            raise ValueError("Supply --input, or use --synthetic for a demo")
        data=read_json(args.input) if args.input else {}
        if isinstance(data,dict) and data.get("implementation", "").startswith("hfcf-reference/"):
            data=data["result"]
        result=COMMANDS[command](data,cfg)
        write_json(args.output,{"stage":command,"implementation":"hfcf-reference/0.1.0","result":result})
        print(json.dumps({"stage":command,"output":args.output,"status":"ok"}))
    except (ValueError,KeyError,TypeError,OSError,ImportError,RuntimeError,np.linalg.LinAlgError) as exc:
        parser.exit(2,f"{command}: {exc}\n")

