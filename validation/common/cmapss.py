"""NASA C-MAPSS turbofan loader (Saxena et al., PHM 2008). Files: validation/data/cmapss/*.txt.gz."""
import gzip
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cmapss")
COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
SENSORS = COLS[5:]
# the 6 flight conditions of FD002/FD004: (altitude kft, Mach, throttle resolver angle)
REGIMES = np.array([[0, 0.0, 100], [10, 0.25, 100], [20, 0.70, 100],
                    [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], float)


def _read(name):
    with gzip.open(os.path.join(DATA, name + ".txt.gz"), "rt") as f:
        return np.loadtxt(f)


def load(fd, split="train"):
    """Returns dict(arr, cols, rul) where rul is per-row remaining useful life for train, and the
    official end-of-sequence RUL for test (one value per unit)."""
    a = _read(f"{split}_{fd}")
    out = {"arr": a, "cols": COLS}
    if split == "train":
        last = {u: a[a[:, 0] == u, 1].max() for u in np.unique(a[:, 0])}
        out["rul"] = np.array([last[u] - c for u, c in a[:, :2]])
    else:
        out["rul_end"] = _read(f"RUL_{fd}")
    return out


def true_regime(settings):
    """Reference flight-condition label: nearest of the 6 published operating points (altitude
    reported in kft, i.e. setting 1)."""
    s = settings.copy()
    scale = np.array([42.0, 0.84, 100.0])
    d = (((s[:, None, :] - REGIMES[None]) / scale) ** 2).sum(-1)
    return d.argmin(1), np.sqrt(d.min(1))
