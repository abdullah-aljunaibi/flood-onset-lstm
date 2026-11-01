"""Shared setup of the independent reviewer's scripts for the five-seed numbers of code/15_rain_to_midnight.py (E1, ERA5 rain to midnight;
results/rain_to_midnight.json; seeds 3-4 trained on 30 October 2026 on the Windows PC).

An AI review: the numbers are recomputed from results/preds with the reviewer's own numpy code (no flood.evaluate, no script of the project imported).
Nothing is written into the project except this folder and the block appended to results/review_rain_to_midnight.json (r5_assemble.py --write-into-project).
Outputs go to OUT (environment variable REVIEW_OUT; default <system temp>/e1_5seeds_review). Run from the project root, one thread:
    PYTHONUTF8=1 <python> code/review/rain_to_midnight_5seeds/<script>.py
Order: r1_scores, r2_compare, r3_backup_diff, r4_texts, r5_assemble (run_checks.sh)."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
BACKUP = PROJ.parent / "Project-Flood_backup_2026-10-30"
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "e1_5seeds_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
LEADS = (1, 2, 3)
OWN = ["wqh", "wqh_older", "wh", "wh_older", "wqh_to09", "wqh_utc", "wqnh"]          # the sets of script 15 (five seeds each since 30 October)
LADDER = ["wq", "wqn", "wqn_older", "wqf", "w", "wn", "wn_older", "wf"]               # stored sets of 04_lstm_all.py and 10_rain_timing.py (three seeds)
SETS = OWN + LADDER
SEEDS = {**{n: [0, 1, 2, 3, 4] for n in OWN}, **{n: [0, 1, 2] for n in LADDER}}
SIMPLE = ["base_linreg_wqh", "base_rf_wqh", "base_linreg_wq", "base_rf_wq", "base_persistence"]
RESULTS = "results/rain_to_midnight.json"
SCRIPT = "code/15_rain_to_midnight.py"
RECORD = "results/review_rain_to_midnight.json"
P_CUT = 0.05


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def jsave(name, obj):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)


def jread(name):
    return jload(OUT / name)


def sha256_of(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def load_npz(name, root=PROJ):
    """(dates, basins, obs[basin, day], pred[basin, day, lead]) of results/preds/<name>.npz, numpy only."""
    with np.load(root / "results" / "preds" / f"{name}.npz", allow_pickle=False) as z:
        return pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]], z["obs"].astype(float), z["pred"].astype(float)


def within(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def isnum(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def same(a, b, tol=1e-9):
    """Recursive equality: numbers within tol (NaN/None equal NaN/None), everything else exactly."""
    nanlike = lambda x: x is None or (isnum(x) and np.isnan(float(x)))
    if nanlike(a) or nanlike(b):
        return nanlike(a) and nanlike(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isnum(a) and isnum(b):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    return a == b


class Compare:
    def __init__(self, tol=1e-9):
        self.rows, self.tol = {}, tol

    def add(self, key, file_value, my_value, tol=None):
        ok = same(file_value, my_value, self.tol if tol is None else tol)
        self.rows[key] = dict(file=file_value, mine=my_value, ok=bool(ok))
        return ok

    def summary(self):
        bad = [k for k, v in self.rows.items() if not v["ok"]]
        return dict(entries=len(self.rows), matching=len(self.rows) - len(bad), not_matching=bad)
