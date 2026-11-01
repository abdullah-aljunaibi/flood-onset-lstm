"""Shared setup of the independent reviewer's scripts for experiment E7 (code/28_multi_product.py: the NLDAS rain, the ERA5 rain to midnight and the AORC rain
to midnight side by side as inputs of the all-river LSTM; results/multi_product.json).

These scripts are an AI review (see README.md): they recompute the results of E7 with the reviewer's own code and write nothing into the project except this
folder and the review record results/review_multi_product.json (m6_assemble.py --write-into-project). Every other output goes to the folder OUT (environment
variable REVIEW_OUT; default: <system temp folder>/mp_review). Run from the project root with the project's environment and one torch thread:
    PYTHONUTF8=1 <python> code/review/multi_product/<script>.py
Order: m1_integrity, m3_columns (writes my_columns.npz), m2_headline, m4_networks, m5_outcome, m6_assemble.
The conventions under review are restated here from ROADMAP.md (section E7), the work order and code/flood.py's documented behaviour; nothing is imported from
code/28_multi_product.py (it runs on import).
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("FLOOD_THREADS", "1")                     # must be set before flood.py is imported
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                                  # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                           # <project>/code/review/multi_product
PROJ = HERE.parents[2]                                           # <project>
sys.path.insert(0, str(PROJ / "code"))
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "mp_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

# ---- the conventions under review, restated (ROADMAP.md E7; flood.py SPLIT / PERIOD as documented) -------------------------------------------------------------
SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
PERIOD = ("1980-10-01", "2014-09-30")
DAYS = pd.date_range(*PERIOD)
LEADS = (1, 2, 3)
FORCING = ["prcp", "srad", "tmax", "tmin", "vp", "dayl"]
STATIC = ["log_area", "elev_mean", "slope_mean", "p_mean", "aridity", "p_seasonality", "frac_forest", "soil_depth_pelletier", "sand_frac", "geol_permeability"]
NEW = ["wqME", "wqME_shuf", "wqME_older"]                         # trained by script 28, five seeds each
EXTRA = {"wqME": ("prcp_nldas", "rain_to_midnight", "aorc_to_midnight"),
         "wqME_shuf": ("prcp_nldas_shuf", "rain_to_midnight_shuf", "aorc_to_midnight"),
         "wqME_older": ("prcp_nldas_older", "rain_to_midnight_older", "aorc_to_midnight")}
STORED = {"wqa": [0, 1, 2, 3, 4], "wqnh": [0, 1, 2, 3, 4], "wqh": [0, 1, 2, 3, 4], "wq": [0, 1, 2], "wqn": [0, 1, 2]}     # what results/preds holds (checked in m2)
SEEDS = {**{n: [0, 1, 2, 3, 4] for n in NEW}, **STORED}
LSTM = NEW + list(STORED)
SIMPLE = ["base_linreg_wqME", "base_rf_wqME", "base_linreg_wqa", "base_rf_wqa"]
SHOWN = LSTM + SIMPLE
GRID = [round(0.50 + 0.05 * i, 2) for i in range(31)]          # 0.50 .. 2.00 step 0.05 (code/warning_policy.GRID, as documented there)
LEVELS = [round(0.30 + 0.05 * i, 2) for i in range(25)]         # 0.30 .. 1.50 step 0.05 (the level grid of the verdict readings)
FAR_CAP = 0.30
SHUF_SEED, SEED_RS, B_RS = 20261030, 20261004, 4000
RESULTS, SCRIPT, FIGURE, LOG = "results/multi_product.json", "code/28_multi_product.py", "figures/multi_product.png", "results/logs/28_multi_product_2026-10-30.log"
ZONE_OF_OFFSET = {-5: "America/New_York", -6: "America/Chicago"}
lk = lambda k: f"l{k}"


def jload(rel):
    with open(PROJ / rel, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def jsave(name, obj):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)


def jread(name):
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def sha256(rel):
    f = PROJ / rel
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def load_npz(name):
    """(dates, basins, obs[basin, day], pred[basin, day, lead], meta) of results/preds/<name>.npz, read with numpy only."""
    z = np.load(PROJ / "results" / "preds" / f"{name}.npz", allow_pickle=False)
    return pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]], z["obs"].astype(float), z["pred"].astype(float), json.loads(str(z["meta"]))


def within(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def zones():
    """Time zone of every basin from the winter UTC offset measured in results/day_windows.json (the rule the scripts state), restated."""
    dw = jload("results/day_windows.json")["day_ends"]["daymet"]["per_basin"]
    return {g: ZONE_OF_OFFSET[int(v["utc_offset_h"])] for g, v in dw.items()}


def close(a, b, tol=1e-9):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return False
    return abs(float(a) - float(b)) <= tol


class Compare:
    """Collects (key, file value, my value) triples and says whether they match."""

    def __init__(self, tol=1e-9):
        self.rows, self.tol = {}, tol

    def add(self, key, file_value, my_value, tol=None):
        t = self.tol if tol is None else tol
        if isinstance(file_value, (bool, np.bool_)) or isinstance(my_value, (bool, np.bool_)):
            ok = bool(file_value) == bool(my_value)
        elif isinstance(file_value, (str, list, dict, type(None))) and not isinstance(my_value, (float, np.floating)):
            ok = file_value == my_value
        else:
            ok = close(file_value, my_value, t)
        assert key not in self.rows, f"duplicate key {key}"
        self.rows[key] = dict(file=file_value, mine=my_value, ok=bool(ok))
        return ok

    def summary(self):
        bad = [k for k, v in self.rows.items() if not v["ok"]]
        return dict(entries=len(self.rows), matching=len(self.rows) - len(bad), not_matching=bad)
