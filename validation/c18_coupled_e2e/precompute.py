"""Precompute the per-seed delivery tables (results/delivery_seed{s}.json) before the federated runs start."""
import importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e7", os.path.join(HERE, "run.py")); e7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e7)
seed = int(sys.argv[1]); n = 35
cell = np.random.default_rng(123).permutation(np.arange(n) % e7.NREG)
nb = 4 * 84129
ok, _ = e7.delivery(cell, seed, nb, [e7.c7c.load(t) for t in e7.c7c.TRACES])
print(seed, {m: float(1 - v.mean()) for m, v in ok.items()})
