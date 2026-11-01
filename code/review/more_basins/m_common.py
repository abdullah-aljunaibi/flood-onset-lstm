"""Shared helpers of the independent review of E3 (code/17_more_basins.py): paths, the reviewer's own readers and scorers.

Nothing here trains, and nothing is written into the project: every check writes its output to $REVIEW_OUT (default: a folder in the temp directory).
The engine (code/flood.py) is imported where the work order allows it (load_preds, thresholds, evaluate, load_data, basins, LSTMNet, Run.predict); the
reviewer's own code is used for the raw CAMELS files, the scores, the pooled onset counts, the scalers, the distances and the rank statistics.
"""
import glob
import hashlib
import json
import os
import pathlib
import sys
import tempfile

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"      # no __pycache__ in the project
os.environ["FLOOD_THREADS"] = "1"                # one thread (flood.py reads this at import)
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

PROJ = pathlib.Path(os.environ.get("E3_PROJECT", "~/AI_Class/Project-Flood")).expanduser()
OUT = pathlib.Path(os.environ.get("REVIEW_OUT", pathlib.Path(tempfile.gettempdir()) / "more_basins_review"))
OUT.mkdir(parents=True, exist_ok=True)
CAM = PROJ / "data" / "camels_us"
V12 = CAM / "basin_dataset_public_v1p2"
PERIOD = ("1980-10-01", "2014-09-30")
SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
FORCING = ["prcp", "srad", "tmax", "tmin", "vp", "dayl"]
STATIC = ["log_area", "elev_mean", "slope_mean", "p_mean", "aridity", "p_seasonality", "frac_forest", "soil_depth_pelletier", "sand_frac", "geol_permeability"]
LEADS = (1, 2, 3)


def engine():
    """Import the project's engine with one torch thread (the project's code folder is put on sys.path as the numbered scripts do)."""
    code = str(PROJ / "code")
    if code not in sys.path:
        sys.path.insert(0, code)
    import torch
    import flood as F
    torch.set_num_threads(1)
    return F


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(name, obj):
    """Write one check's numbers to $REVIEW_OUT/<name>.json (strict JSON, NaN as null)."""
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    (OUT / f"{name}.json").write_text(json.dumps(clean(obj), indent=1))


# ---- prediction sets, read with numpy alone -------------------------------------------------------------------------------------------------
def load_npz(name, quick=False):
    """A prediction set as the reviewer reads it: basins, dates (datetime64[D]), obs [basin, day], pred [basin, day, lead], meta (dict)."""
    z = np.load(PROJ / "results" / "preds" / f"{name}.npz", allow_pickle=False)
    return dict(basins=[str(b) for b in z["basins"]], dates=pd.DatetimeIndex(z["dates"]), obs=z["obs"], pred=z["pred"], meta=json.loads(str(z["meta"])))


def own_nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return float(1.0 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def window(dates, part):
    """Boolean mask of the target days inside a split part (own date arithmetic: string bounds on a DatetimeIndex)."""
    a, b = SPLIT[part]
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def own_thresholds(ids, flow_by_id=None):
    """95th percentile (linear interpolation, numpy default) of the daily flow of the training years of each basin."""
    out = {}
    for g in ids:
        s = flow_by_id[g]
        m = (s.index >= pd.Timestamp(SPLIT["train"][0])) & (s.index <= pd.Timestamp(SPLIT["train"][1]))
        out[g] = float(np.quantile(s.values[m], 0.95))
    return out


def own_pooled(npz, thr, lead, part="test", trigger=1.0):
    """The reviewer's own scores of one prediction set at one lead: per-basin NSE (test target days with a finite forecast) and pooled warning counts.

    all   : high day = obs > thr; warning = forecast > trigger * thr
    onset : the same, kept only where the observed flow on the ISSUE day (target day - lead) was at or below thr
    Counting is done on arrays: the issue-day flow is obs[t - lead] (positions shifted by `lead`), never a pandas shift."""
    dates, obs, pred = npz["dates"], npz["obs"], npz["pred"][:, :, lead - 1]
    sel = window(dates, part)
    nse, tot = {}, dict(all=dict(h=0, m=0, f=0, c=0), onset=dict(h=0, m=0, f=0, c=0))
    for i, g in enumerate(npz["basins"]):
        o, s = obs[i], pred[i]
        ok = sel & np.isfinite(o) & np.isfinite(s)
        nse[g] = own_nse(o[ok], s[ok])
        issue = np.full(len(o), np.nan)
        issue[lead:] = o[:-lead]                           # observed flow on the issue day of each target day
        low = issue <= thr[g]                              # NaN compares False, as in the engine
        hi, warn = o > thr[g], s > trigger * thr[g]
        for kind, m in (("all", ok), ("onset", ok & low)):
            tot[kind]["h"] += int((hi & warn & m).sum())
            tot[kind]["m"] += int((hi & ~warn & m).sum())
            tot[kind]["f"] += int((~hi & warn & m).sum())
            tot[kind]["c"] += int((~hi & ~warn & m).sum())
    out = dict(nse=nse, nse_median=float(np.median(list(nse.values()))))
    for kind, t in tot.items():
        h, m, f = t["h"], t["m"], t["f"]
        out[kind] = dict(hits=h, misses=m, false_alarms=f, correct_neg=t["c"], pod=h / (h + m) if h + m else float("nan"),
                         far=f / (h + f) if h + f else float("nan"), csi=h / (h + m + f) if h + m + f else float("nan"))
    return out


# ---- raw CAMELS files, read by the reviewer ------------------------------------------------------------------------------------------------------
def raw_paths(gid):
    f = glob.glob(str(V12 / "basin_mean_forcing" / "daymet" / "*" / f"{gid}_lump_cida_forcing_leap.txt"))
    q = glob.glob(str(V12 / "usgs_streamflow" / "*" / f"{gid}_streamflow_qc.txt"))
    assert len(f) == 1 and len(q) == 1, f"{gid}: forcing {f} flow {q}"
    return f[0], q[0]


def raw_basin(gid):
    """The daily table of one basin from the raw CAMELS text files, parsed with the reviewer's own code (np.loadtxt for the forcing, a plain split of every line
    of the flow file), over the whole record 1980-01-01 .. 2014-12-31: columns dayl prcp srad swe tmax tmin vp (as CAMELS names them), q_cfs, q_flag, area_m2.
    Dates come from the year, month and day columns of each file (not from the row position), so a missing or repeated row is visible."""
    f, q = raw_paths(gid)
    with open(f) as fh:
        lat, elev, area_m2 = float(fh.readline()), float(fh.readline()), float(fh.readline())
    a = np.loadtxt(f, skiprows=4)
    assert a.shape[1] == 11
    fd = pd.to_datetime(pd.DataFrame(dict(year=a[:, 0].astype(int), month=a[:, 1].astype(int), day=a[:, 2].astype(int))))
    forc = pd.DataFrame(a[:, 4:], index=pd.DatetimeIndex(fd), columns=["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"])
    rows = [ln.split() for ln in open(q) if ln.strip()]
    assert all(len(r) == 6 and r[0] == gid for r in rows), f"{gid}: unexpected flow-file layout"
    qd = pd.to_datetime(pd.DataFrame(dict(year=[int(r[1]) for r in rows], month=[int(r[2]) for r in rows], day=[int(r[3]) for r in rows])))
    flow = pd.DataFrame(dict(cfs=[float(r[4]) for r in rows], flag=[r[5] for r in rows]), index=pd.DatetimeIndex(qd))
    return forc, flow, area_m2, (lat, elev)


def raw_table(gid):
    """The table the engine is meant to hold for this basin, built from raw: FORCING columns + q_mm over PERIOD (mm/day; a negative CAMELS flow = missing)."""
    forc, flow, area_m2, _ = raw_basin(gid)
    assert forc.index.is_unique and flow.index.is_unique
    full = pd.date_range(*PERIOD)
    t = forc.reindex(full)[FORCING].astype("float64")
    cfs = flow.cfs.reindex(full)
    cfs = cfs.where(cfs >= 0)
    t["q_mm"] = cfs * 0.0283168 * 86400 / area_m2 * 1000
    return t


def attribute_tables():
    """The seven CAMELS attribute tables joined on gauge_id, read by the reviewer."""
    r = lambda n: pd.read_csv(CAM / f"camels_{n}.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    a = r("name")
    for n in ("topo", "clim", "hydro", "soil", "vege", "geol"):
        a = a.join(r(n), rsuffix=f"_{n}")
    return a


def static_table(ids, A=None):
    """The ten basin properties of the engine (flood.STATIC) for the given gauges, from the attribute tables; log_area = log10(area_gages2)."""
    A = attribute_tables() if A is None else A
    t = A.loc[list(ids)].copy()
    t["log_area"] = np.log10(t["area_gages2"].astype(float))
    return t[STATIC].astype(float)


def haversine_km(la1, lo1, la2, lo2, R=6371.0088):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def handin_dir(arg=None):
    """The folder that holds Project-Flood/ as unzipped from the hand-in archive (the engine files, features.py, run_all.sh, make_handin_zip.sh): the argument, else $REVIEW_OUT/handin; made from
    ~/AI_Class/Project-Flood_handin.zip when a file is missing."""
    import zipfile
    h = pathlib.Path(arg) if arg else OUT / "handin"
    need = ["code/flood.py", "code/test_flood.py", "code/camels.py", "code/common.py", "code/features.py", "run_all.sh", "make_handin_zip.sh"]
    if not all((h / "Project-Flood" / n).exists() for n in need):
        with zipfile.ZipFile(pathlib.Path("~/AI_Class/Project-Flood_handin.zip").expanduser()) as zf:
            for n in need:
                zf.extract(f"Project-Flood/{n}", h)
    return h
