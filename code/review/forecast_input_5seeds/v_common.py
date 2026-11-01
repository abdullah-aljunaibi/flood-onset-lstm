"""Shared setup of the independent reviewer's round-2 scripts for experiment E2 (code/18_forecast_input.py, results/forecast_input.json) after seeds 3 and 4 were added
on 30 October 2026 (Windows PC). They recompute the five-seed numbers from results/preds with the reviewer's own numpy scoring (no flood.evaluate) and write nothing into the
project except this folder and the block round_2 of results/review_forecast_input.json (v5_assemble.py --write-into-project). Outputs go to REVIEW_OUT
(default <system temp>/e2_5seeds_review). Run from the project root with the project environment, PYTHONUTF8=1, one torch thread. Nothing is trained.
Order: v1_score, v2_compare, v3_networks, v4_readings_text, v5_assemble."""
import hashlib, json, os, sys, tempfile, time
from pathlib import Path
os.environ.setdefault("FLOOD_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
sys.path.insert(0, str(PROJ / "code"))
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "e2_5seeds_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

RESULTS = "results/forecast_input.json"
BACKUP = PROJ.parent / "Project-Flood_backup_2026-10-30" / "results" / "forecast_input.json"
SCRIPT = "code/18_forecast_input.py"
REVIEW = "results/review_forecast_input.json"
LEADS = (1, 2, 3)
SPLITS = {"main": {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")},
          "sat": {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}}
E2 = ["wqaF_sub", "wqaF_clim", "wqaP", "wqF_sub", "wqaF_s", "wqa_s", "wqaF_clim_s", "wq_s", "wqaP_s", "wqaF_sub_raw"]
DERIVED = ["wqaF_sub_rawpred", "wqaF_sub_perfectpred", "wqaP_forecastpred"]
CONTROLS = ["wq", "wqa", "wqf"]
SHOWN = E2 + CONTROLS + DERIVED
SEEDS = {**{n: [0, 1, 2, 3, 4] for n in E2 + DERIVED}, "wqa": [0, 1, 2, 3, 4], "wq": [0, 1, 2], "wqf": [0, 1, 2]}
ARM = {n: ("sat" if n in E2 and n.endswith("_s") else "main") for n in SHOWN}
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
POD_GO, FAR_GO, CSI_CAUTION, P_CUT = 0.70, 0.30, 0.40, 0.05


def jload(rel):
    with open(PROJ / rel, encoding="utf-8") as f:
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
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256(rel):
    return sha256_file(PROJ / rel)


def load_npz(name):
    z = np.load(PROJ / "results" / "preds" / f"{name}.npz", allow_pickle=False)
    return pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]], z["obs"].astype(float), z["pred"].astype(float), json.loads(str(z["meta"]))


def mask(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def rates(h, m, f):
    d = lambda a, b: a / b if b else float("nan")
    return dict(hits=int(h), misses=int(m), false_alarms=int(f), pod=d(h, h + m), far=d(f, h + f), csi=d(h, h + m + f))


def verdict(t):
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= POD_GO and t["far"] <= FAR_GO:
        return "GO"
    return "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"


def choose(tabs):
    """best pooled CSI over LEVELS; a tie goes to the level closest to 1.0 (the first such level)."""
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in tabs])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])


def own_ap(y, s):
    """average precision (step-wise, distinct score thresholds), own numpy."""
    o = np.argsort(-s, kind="mergesort"); y, s = y[o].astype(float), s[o]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[last]; fp = (last + 1) - tp
    prec, rec = tp / (tp + fp), tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def own_auc(y, s):
    """ROC area by the rank-sum (ties count one half), own numpy."""
    from scipy.stats import rankdata
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def close(a, b, tol=1e-9):
    na = a is None or (isinstance(a, float) and a != a)
    nb = b is None or (isinstance(b, float) and b != b)
    if na or nb:
        return na and nb
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= tol


class Compare:
    def __init__(self, tol=1e-9):
        self.rows, self.tol = {}, tol

    def add(self, key, file_value, my_value, tol=None):
        ok = close(file_value, my_value, self.tol if tol is None else tol)
        self.rows[key] = dict(file=file_value, mine=my_value, ok=bool(ok))
        return ok

    def summary(self):
        bad = [k for k, v in self.rows.items() if not v["ok"]]
        return dict(entries=len(self.rows), matching=len(self.rows) - len(bad), not_matching=bad)
