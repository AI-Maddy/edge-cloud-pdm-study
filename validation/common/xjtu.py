"""XJTU-SY rolling-element bearing run-to-failure data (Wang et al., IEEE Trans. Reliability 2020).

Each bearing is run to failure under one operating condition; every minute a 1.28 s snapshot of horizontal
and vertical vibration is recorded at 25.6 kHz (32,768 samples). Conditions 1 (35 Hz, 12 kN) and 2
(37.5 Hz, 11 kN), 10 bearings, 2,182 snapshots, are reachable from the cloud workspace (GitHub mirror
rajeevfee/XJTU-SY-Bearing-Dataset); condition 3 is not in that mirror.

`build()` streams every snapshot once, keeps only per-snapshot features (raw = 2.8 GB) and the SHA-256 of
each source file, and writes data/xjtu_sy/features.csv.gz + sources.csv.gz. `load()` reads the features.
"""
import concurrent.futures as cf
import hashlib
import io
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "xjtu_sy")
URL = "https://raw.githubusercontent.com/rajeevfee/XJTU-SY-Bearing-Dataset/main"
BEARINGS = {"35Hz12kN": {"Bearing1_1": 123, "Bearing1_2": 161, "Bearing1_3": 158, "Bearing1_4": 122, "Bearing1_5": 52},
            "37.5Hz11kN": {"Bearing2_1": 491, "Bearing2_2": 161, "Bearing2_3": 533, "Bearing2_4": 42, "Bearing2_5": 339}}
FS = 25600
BANDS = np.array([0, 500, 1000, 2000, 3000, 4000, 5000, 6500, 8000, 10000, 12800])
CH = ["h", "v"]


def snapshot_features(x):
    """x: (32768, 2). Time-domain statistics + log band energies + spectral centroid per channel."""
    out = {}
    f = np.fft.rfftfreq(len(x), 1 / FS)
    for i, c in enumerate(CH):
        s = x[:, i] - x[:, i].mean()
        rms = np.sqrt(np.mean(s ** 2))
        p = np.abs(np.fft.rfft(s)) ** 2
        out.update({f"{c}_rms": rms, f"{c}_peak": np.abs(s).max(), f"{c}_kurt": stats.kurtosis(s),
                    f"{c}_skew": stats.skew(s), f"{c}_crest": np.abs(s).max() / rms,
                    f"{c}_centroid": float((f * p).sum() / p.sum())})
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            out[f"{c}_band{lo}_{hi}"] = np.log10(p[(f >= lo) & (f < hi)].sum() + 1e-12)
    return out


def _one(args):
    cond, b, k = args
    for attempt in range(8):                          # raw.githubusercontent rate-limits bursts (HTTP 429)
        try:
            raw = urllib.request.urlopen(f"{URL}/{cond}/{b}/{k}.csv", timeout=60).read()
            break
        except Exception:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)
    x = pd.read_csv(io.BytesIO(raw)).to_numpy(np.float64)
    assert x.shape == (32768, 2), (b, k, x.shape)
    return {"condition": cond, "bearing": b, "minute": k, "n_total": BEARINGS[cond][b], **snapshot_features(x)}, \
           {"file": f"{cond}/{b}/{k}.csv", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build(workers=4):
    jobs = [(c, b, k) for c, bs in BEARINGS.items() for b, n in bs.items() for k in range(1, n + 1)]
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(_one, jobs))
    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame([r[0] for r in res]).to_csv(os.path.join(DATA, "features.csv.gz"), index=False, float_format="%.6g")
    pd.DataFrame([r[1] for r in res]).to_csv(os.path.join(DATA, "sources.csv.gz"), index=False)
    return len(res)


def load():
    d = pd.read_csv(os.path.join(DATA, "features.csv.gz"))
    d["rul"] = d["n_total"] - d["minute"]                 # minutes to failure
    d["life_frac"] = d["minute"] / d["n_total"]
    return d


if __name__ == "__main__":
    t = time.time(); n = build(); print(n, "snapshots in", round(time.time() - t), "s")
