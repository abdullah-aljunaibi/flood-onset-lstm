"""Shared setup of the independent reviewer's scripts for experiment E4b (code/21_within_day_inputs.py: within-day AORC rain and Q95-relative flow as
inputs of the all-river LSTM; results/within_day_inputs.json).

These scripts are an AI review (see README.md): they recompute the results of E4b with the reviewer's own code and write nothing into the project except this
folder and the review record results/review_within_day_inputs.json (r6_assemble.py --write-into-project). Every output goes to the folder OUT (environment
variable REVIEW_OUT; default: <system temp folder>/wdi_review). Run from the project root with the project's environment and one torch thread:
    PYTHONUTF8=1 <python> code/review/within_day_inputs/<script>.py
Order: r1_integrity, r3_columns (writes my_columns.npz), r2_headline, r4_networks, r5_outcome, r6_assemble.
"""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("FLOOD_THREADS", "1")                     # must be set before flood.py is imported
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                                  # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                           # <project>/code/review/within_day_inputs
PROJ = HERE.parents[2]                                           # <project>
sys.path.insert(0, str(PROJ / "code"))                           # flood.py, subdaily_inputs.py ... as the numbered scripts see them
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "wdi_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

PERIOD = ("1980-10-01", "2014-09-30")
SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
LEADS = (1, 2, 3)
NEW = ["wqaQ", "wqaW", "wqaW_older", "wqaWF_sub"]                  # the sets trained by script 21 (five seeds each)
STORED = ["wqa", "wqaF_sub", "wqaP"]                               # the stored comparison sets (three seeds each)
SHOWN = NEW + STORED
SEEDS = {**{n: [0, 1, 2, 3, 4] for n in NEW}, **{n: [0, 1, 2] for n in STORED}}
RESULTS = "results/within_day_inputs.json"
SCRIPT = "code/21_within_day_inputs.py"
WD_COLS = ("aorc_last3h", "aorc_last6h", "aorc_last12h", "aorc_max1h", "aorc_maxcell1h", "aorc_wet_hours")
FORCING = ["prcp", "srad", "tmax", "tmin", "vp", "dayl"]
STATIC = ["log_area", "elev_mean", "slope_mean", "p_mean", "aridity", "p_seasonality", "frac_forest", "soil_depth_pelletier", "sand_frac", "geol_permeability"]
BASE_DYN = tuple(FORCING) + ("aorc_to_midnight",)


def jload(rel):
    """A JSON file of the project, by its path from the project root."""
    with open(PROJ / rel, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def jsave(name, obj):
    """The output of a check: OUT/<name> (never inside the project)."""
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


def close(a, b, tol=1e-9):
    """Equal within tol; None/NaN on both sides count as equal."""
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
        ok = close(file_value, my_value, t) if not isinstance(file_value, (bool, str, list, dict, type(None))) or isinstance(my_value, float) else file_value == my_value
        if isinstance(file_value, bool) or isinstance(my_value, bool):
            ok = bool(file_value) == bool(my_value)
        self.rows[key] = dict(file=file_value, mine=my_value, ok=bool(ok))
        return ok

    def summary(self):
        bad = [k for k, v in self.rows.items() if not v["ok"]]
        return dict(entries=len(self.rows), matching=len(self.rows) - len(bad), not_matching=bad)
