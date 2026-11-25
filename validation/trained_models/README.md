# Trained models and artefacts from the concept validations

Produced by [`../save_models.py`](../save_models.py) (seed 0). [`MANIFEST.json`](MANIFEST.json) lists every file with
its SHA-256, size, provenance and test metric. Total size is about 3 MB.

| File | What it is | Test metric |
| --- | --- | --- |
| `c2_transformer_FD00x.pt` + `c2_normaliser_FD00x.json` | Edge RUL model per C-MAPSS subset: repo TransformerEncoder + head, inputs normalised per operating condition | RMSE 12.2 / 14.1 / 13.0 / 14.9 |
| `c3_fedavg_cnn1d.pt` | FedAvg CNN1D over 35 sites, 25 rounds | mean site RMSE 19.4 |
| `c3_cafig_cell0/1_cnn1d.pt` | Context-cell models (single-condition / six-condition sites) | 19.5 / 19.5 |
| `c3_normaliser.json` | Federation-wide per-condition normaliser (for c3_* and c6_*) | - |
| `c6_cloud_cnn1d.pt` | Cloud model sent to cold-start sites (trained on 27 sites) | RMSE 15.4 on the 8 unseen sites |
| `c9_simclr.pt`, `c9_byol.pt`, `c9_ae.pt` | Self-supervised window encoders (repo classes), 32-d embeddings | see C9 |
| `c4_cells.json` | Beijing context cells (k = 6, year 1) and per-cell next-day PM2.5 ridge models | see C4 |
| `c5_knowledge_objects.json` | 300+ mined knowledge objects with repository trust scores | see C5 |

## Loading

```python
import sys, torch
sys.path += ["validation", "models"]          # run from hierarchical-ai-execution/
from common import rul
ck = torch.load("validation/trained_models/c2_transformer_FD001.pt", weights_only=False)
model = rul.RepoTransformerRUL(14); model.load_state_dict(ck["state_dict"])
Xtr, ytr, utr, Xte, yte, _ = rul.load_fd("FD001")   # applies the same normalisation
print(rul.rmse(rul.predict(model, Xte), yte))        # 12.22, as recorded in the manifest
```

CNN1D files load the same way with `rul.CNN1D(14)`. The C9 encoders load with the repository classes in
`models/hfcf_models/neural.py` using the `args` stored in each file. Retraining with `python save_models.py` reproduces
the files up to CPU thread non-determinism (for example, the FD001 Transformer scored 13.9 in the C2 run and 12.2 here
with the same seed).
