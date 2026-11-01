"""CAMELS-US loaders shared by every script: basin attributes, Daymet forcing and USGS flow for one gauge."""
import glob

import numpy as np
import pandas as pd

from common import DATA

CAM = DATA / "camels_us"
V12 = CAM / "basin_dataset_public_v1p2"
FORCING_COLS = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]


def attributes():
    """One row per gauge: name, location, climate and flow signatures (the camels_*.txt tables joined)."""
    r = lambda n: pd.read_csv(CAM / f"camels_{n}.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    a = r("name")
    for n in ("topo", "clim", "hydro", "soil", "vege", "geol"):
        a = a.join(r(n), rsuffix=f"_{n}")
    return a


def load_basin(gauge_id, source="daymet"):
    """Daily table 1980-2014: forcing (Daymet by default; "nldas" or "maurer" also ship with CAMELS) + observed flow in mm/day (NaN where the gauge has no value)."""
    f = glob.glob(str(V12 / "basin_mean_forcing" / source / "*" / f"{gauge_id}_lump_{'cida' if source == 'daymet' else source}_forcing_leap.txt"))[0]
    with open(f) as fh:
        fh.readline(); fh.readline()
        area_m2 = float(fh.readline())
    met = pd.read_csv(f, sep=r"\s+", skiprows=4, header=None, names=["y", "m", "d", "h"] + FORCING_COLS)
    met.index = pd.to_datetime(dict(year=met.y, month=met.m, day=met.d))
    q = glob.glob(str(V12 / "usgs_streamflow" / "*" / f"{gauge_id}_streamflow_qc.txt"))[0]
    flow = pd.read_csv(q, sep=r"\s+", header=None, names=["g", "y", "m", "d", "cfs", "flag"])
    flow.index = pd.to_datetime(dict(year=flow.y, month=flow.m, day=flow.d))
    cfs = flow.cfs.where(flow.cfs >= 0)
    df = met[FORCING_COLS].copy()
    df["q_mm"] = cfs * 0.0283168 * 86400 / area_m2 * 1000          # cubic feet per second -> mm per day over the basin
    df["q_flag"] = flow.flag
    df.attrs["area_km2"] = area_m2 / 1e6
    return df
