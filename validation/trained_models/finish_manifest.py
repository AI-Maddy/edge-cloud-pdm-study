"""Rebuild MANIFEST.json from the saved files (each .pt stores its own metadata)."""
import hashlib, json, os, torch
D = os.path.dirname(os.path.abspath(__file__)); man = {}
for f in sorted(os.listdir(D)):
    p = os.path.join(D, f)
    if f.endswith(".pt"):
        meta = torch.load(p, weights_only=False)["meta"]
    elif f.endswith(".json") and f != "MANIFEST.json":
        meta = {"kind": "json"}
    else:
        continue
    b = open(p, "rb").read(); man[f] = {**meta, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
json.dump(man, open(os.path.join(D, "MANIFEST.json"), "w"), indent=1, default=str); print(len(man), "files")
