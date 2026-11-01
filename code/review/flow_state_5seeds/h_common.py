"""Shared setup of the independent reviewer's scripts for experiment E4e (code/22_flow_state.py: the hourly USGS gauge flow at the issue time as inputs of the
all-river LSTM on the satellite-era split; results/flow_state.json).

These scripts are an AI review (see README.md): they recompute the results of E4e with the reviewer's own code and write nothing into the project except this
folder and the review record results/review_flow_state.json (h6_assemble.py --write-into-project). Every other output goes to the folder OUT (environment
variable REVIEW_OUT; default: <system temp folder>/fs_review). Run from the project root with the project's environment and one torch thread:
    PYTHONUTF8=1 <python> code/review/flow_state/<script>.py
Order: h1_integrity, h3_columns (writes my_columns.npz), h2_headline, h4_networks, h5_outcome, h6_assemble.
"""
import ast
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
HERE = Path(__file__).resolve().parent                           # <project>/code/review/flow_state
PROJ = HERE.parents[2]                                           # <project>
sys.path.insert(0, str(PROJ / "code"))                           # flood.py, subdaily_inputs.py ... as the numbered scripts see them
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "fs_5seeds_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

# ---- the conventions under review, restated from the brief and the work order (not imported from the code under review) --------------------------------------
SAT_SPLIT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}
T_ISS = pd.date_range("2000-01-01", "2014-09-30")              # the issue days of the satellite-era tables (script 18 / 22)
LEADS = (1, 2, 3)
NEW = ["wqaH_s", "wqaH22_s", "wqaWH_s", "wqaWH22_s", "wqaWHF_s", "wqaWHF22_s"]          # the sets trained by script 22 (five seeds each)
STORED = ["wqa_s", "wqaF_s", "wqaP_s"]                                                 # the stored comparison sets of script 18 (five seeds each since 30 October 2026)
SHOWN = NEW + STORED
SEEDS = {**{n: [0, 1, 2, 3, 4] for n in NEW}, **{n: [0, 1, 2, 3, 4] for n in STORED}}
BASE_OF = {"wqaH_s": "wqa_s", "wqaH22_s": "wqa_s", "wqaWH_s": "wqa_s", "wqaWH22_s": "wqa_s", "wqaWHF_s": "wqaF_s", "wqaWHF22_s": "wqaF_s"}
LAST_HOUR = {"wqaH_s": 24, "wqaH22_s": 22, "wqaWH_s": 24, "wqaWH22_s": 22, "wqaWHF_s": 24, "wqaWHF22_s": 22}
RESULTS = "results/flow_state.json"
SCRIPT = "code/22_flow_state.py"
FIGURE = "figures/flow_state.png"
H_COLS = ("q_end", "dq6", "dq12", "q_dmax", "hours_above")
H22_COLS = tuple(c + "_22" for c in H_COLS)
W_COLS = ("aorc_last3h", "aorc_last6h", "aorc_last12h", "aorc_max1h", "aorc_maxcell1h", "aorc_wet_hours")
FC_COLS = ("fc1", "fc2", "fc3")
FORCING = ["prcp", "srad", "tmax", "tmin", "vp", "dayl"]
STATIC = ["log_area", "elev_mean", "slope_mean", "p_mean", "aridity", "p_seasonality", "frac_forest", "soil_depth_pelletier", "sand_frac", "geol_permeability"]
BASE_DYN = tuple(FORCING) + ("aorc_to_midnight",)
CARRY_H, MISMATCH_REL, Q_MIN_FOR_MISMATCH = 6, 0.5, 0.5
GRID = [round(0.50 + 0.05 * i, 2) for i in range(31)]          # 0.50 .. 2.00 step 0.05, multiples of Q95 (code/warning_policy.GRID)
FAR_CAP = 0.30
POD_GO, FAR_GO, CSI_CAUTION = 0.70, 0.30, 0.40                 # results/flood_analysis.json verdict.criteria (checked in h2)
ZONE_OF_OFFSET = {-5: "America/New_York", -6: "America/Chicago"}
lk = lambda k: f"l{k}"


def jload(rel):
    """A JSON file of the project, by its path from the project root."""
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


def zones():
    """The time zone of every basin from the winter UTC offset measured in results/day_windows.json (the rule of scripts 16 and 22), restated here."""
    dw = jload("results/day_windows.json")["day_ends"]["daymet"]["per_basin"]
    return {g: ZONE_OF_OFFSET[int(v["utc_offset_h"])] for g, v in dw.items()}


def script_function(name, namespace):
    """The source of one top-level function of the script under review, executed into `namespace` (the script itself cannot be imported: it runs on import).
    Used only where the brief asks to compare the reviewer's columns with the SCRIPT's columns; nothing of the project is changed."""
    src = (PROJ / SCRIPT).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            code = ast.get_source_segment(src, node)
            exec(compile(code, f"<{SCRIPT}:{name}>", "exec"), namespace)
            return namespace[name], code
    raise KeyError(name)


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
        if isinstance(file_value, bool) or isinstance(my_value, bool):
            ok = bool(file_value) == bool(my_value)
        elif isinstance(file_value, (str, list, dict, type(None))) and not isinstance(my_value, float):
            ok = file_value == my_value
        else:
            ok = close(file_value, my_value, t)
        self.rows[key] = dict(file=file_value, mine=my_value, ok=bool(ok))
        return ok

    def summary(self):
        bad = [k for k, v in self.rows.items() if not v["ok"]]
        return dict(entries=len(self.rows), matching=len(self.rows) - len(bad), not_matching=bad)
