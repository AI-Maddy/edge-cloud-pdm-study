# Edge-cloud predictive maintenance: pre-registered real-data study

Code, derived data, results and trained models for the paper
"Context First, Knowledge Second: A Pre-Registered Real-Data Study of Edge-Cloud Predictive Maintenance".

Each experiment is a script whose header states its tests and numeric pass criteria. The criteria were fixed
before the first run; results (including failures) are in each folder's `README.md` and `results/`.
Folder READMEs sometimes refer to "notes"; these are the system's design notes, summarised in the paper's
System Design section.

| Folder | Mechanism (paper section) | Data |
| --- | --- | --- |
| `validation/c1_context_clustering` | M1 context vector and cells | C-MAPSS FD002/FD004; Beijing air quality |
| `validation/c2_local_rul_model` | M2 edge RUL model, context normalisation | C-MAPSS FD001-FD004 |
| `validation/c3_federation_cells` | M3 federation cells vs FedAvg / IFCA / oracle, 35 sites | C-MAPSS |
| `validation/c4_drift_reclustering` | drift monitor and re-clustering | Beijing |
| `validation/c5_knowledge_trust` | M5 knowledge objects and trust | C-MAPSS, unseen sites |
| `validation/c6_knowledge_consumption` | M5 cold-start consumption | C-MAPSS cold-start sites |
| `validation/c7_hfip_traffic` | M4 upstream objects over recorded uplinks | C-MAPSS + cellular traces |
| `validation/c8_stability` | convergence and stability | C-MAPSS FL; Beijing |
| `validation/c15_bearings` | generality on a second asset | XJTU-SY bearings (features) |
| `validation/c16_lifelong` | lifelong learning with replay | C-MAPSS FD001->FD004 |
| `validation/c3b_federation_converged` | follow-up: 100 rounds, centralised bound, regional tier | C-MAPSS, 35 sites |
| `validation/c7b_lofo` | follow-up: feature uploads and cold start with each fleet held out | C-MAPSS |
| `validation/c7c_uplink_sim` | follow-up: many sites sharing one uplink, packet-level simulation | cellular traces |
| `validation/c15c_bearings_federated` | follow-up: federation and rule transfer across real bearings | XJTU-SY bearings (features) |

`scripts/hfcf` and `models/hfcf_models` are the system implementation under test (context features, federation,
knowledge objects, Transformer encoder). `validation/common` holds shared loaders and the RUL training code.
`validation/trained_models` holds the trained models from these runs (`save_models.py` regenerates them).

## Data
All data are public; compact copies are in `validation/data` with SHA-256 checksums in `validation/data/SHA256SUMS`.
- NASA C-MAPSS turbofan degradation (Saxena et al., PHM 2008), NASA PCoE public data
- Beijing multi-site air quality (Zhang et al., Proc. R. Soc. A 2017), UCI id 501, CC BY 4.0
- Cellular uplink traces from Mahimahi (Netravali et al., USENIX ATC 2015; Winstein et al., NSDI 2013)
- XJTU-SY bearings, conditions 1-2, extracted features only (Wang et al., IEEE Trans. Reliability 2020)

## Running
Python 3.10+, `pip install -r requirements.txt` (CPU PyTorch is enough). From `validation/`:
`python c1_context_clustering/run.py` and so on; run times are listed in each README (from about 1 minute to
about 2 hours on 4 CPU cores). Set `OMP_NUM_THREADS=1` for the scikit-learn experiments.
