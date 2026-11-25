"""Beijing multi-site air-quality network (Zhang et al., Proc. R. Soc. A 2017; UCI id 501).
12 monitoring stations, hourly, 2013-03-01 .. 2017-02-28 (35,064 h each). Missing values and compass wind
directions are handled here (short gaps <= 3 h interpolated; longer gaps stay missing)."""
import glob
import gzip
import io
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "beijing")
POLL = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
MET = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
COMPASS = {d: i * 22.5 for i, d in enumerate("N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}
CTX = ["TEMP", "DEWP", "PRES", "WSPM", "wu", "wv", "lrain"]


def load_hourly():
    out = {}
    for p in sorted(glob.glob(os.path.join(DATA, "PRSA_Data_*.csv.gz"))):
        with gzip.open(p, "rt") as f:
            df = pd.read_csv(io.StringIO(f.read()))
        assert len(df) == 35064
        for c in POLL + MET:
            df[c] = df[c].interpolate(limit=3, limit_area="inside")
        ang = np.deg2rad(df["wd"].map(COMPASS).astype(float).ffill(limit=3))
        df["wu"], df["wv"] = df["WSPM"] * np.sin(ang), df["WSPM"] * np.cos(ang)
        df["lrain"] = np.log1p(df["RAIN"].clip(lower=0))
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        out[df["station"].iloc[0]] = df
    assert len(out) == 12
    return out


def daily(hourly=None):
    """Per station: daily mean context (7 meteorological fields) and daily mean log PM2.5."""
    hourly = hourly or load_hourly()
    names = sorted(hourly)
    ctx, pm = [], []
    for s in names:
        g = hourly[s].groupby("date")
        ctx.append(g[CTX].mean().to_numpy())
        pm.append(np.log1p(g["PM2.5"].mean().to_numpy()))
    return names, np.stack(ctx), np.stack(pm)     # S x D x 7, S x D
