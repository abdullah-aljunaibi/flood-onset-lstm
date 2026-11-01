"""Shared setup of the independent reviewer's scripts for experiment E4a (the warning operating point on STORED forecasts:
code/20_warning_policy.py, code/warning_policy.py, results/warning_policy.json, figures/warning_policy.png).

These scripts are an AI review (see README.md): they recompute E4a's numbers with the reviewer's OWN code, straight from the arrays of
results/preds/lstm_*.npz (numpy and pandas only; the project's flood.py is imported only where a check compares against the project's
reference: flood.thresholds, flood.evaluate), and they write nothing into the project except this folder and the review record
(w9_assemble_review.py --write-into-project). Every other output goes to OUT (environment variable REVIEW_OUT; default
<system temp folder>/e4a_review). Nothing is trained; the project's numbered scripts are not run in the project (w8 runs a copy in a sandbox).
Run from the project root:   PYTHONUTF8=1 <python of .venv-flood> code/review/warning_policy/<script>.py
"""
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("FLOOD_THREADS", "1")                    # must be set before flood.py is imported (where a script imports it)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                                 # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                          # <project>/code/review/warning_policy
PROJ = HERE.parents[2]                                          # <project>
sys.path.insert(0, str(PROJ / "code"))                          # flood.py, warning_policy.py ... as the numbered scripts see them
sys.path.insert(0, str(HERE))                                   # the helpers of this folder
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "e4a_review"))
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

# ---- the conventions under review, restated here from the brief (not imported from the code under review) ---------------------------
SPLIT_MAIN = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
SPLIT_SAT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}
SPLITS = {"main": SPLIT_MAIN, "sat": SPLIT_SAT}
HALVES = {"main": (("2000-10-01", "2003-03-31"), ("2003-04-01", "2005-09-30")),      # the two halves of the validation years (stacker: fit / cut)
          "sat": (("2008-10-01", "2009-09-30"), ("2009-10-01", "2010-09-30"))}
GRID = [round(0.50 + 0.05 * i, 2) for i in range(31)]          # 0.50 .. 2.00 step 0.05, multiples of Q95
FAR_CAP = 0.30
POD_GO, FAR_GO, CSI_CAUTION = 0.70, 0.30, 0.40                 # results/flood_analysis.json verdict.criteria (checked in w0)
LEADS = (1, 2, 3)
RULES = ("csi", "far_cap")
BLOCK_DAYS = 30
SETS = {"lstm_wq": "main", "lstm_wqn": "main", "lstm_wqa": "main", "lstm_wqaF_sub": "main", "lstm_wqaP": "main", "lstm_wqf": "main",
        "lstm_wqa_s": "sat", "lstm_wqaF_s": "sat", "lstm_wqaP_s": "sat"}
BRIEF_SETS = ["lstm_wq", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqaP", "lstm_wqf", "lstm_wqa_s", "lstm_wqaF_s", "lstm_wqaP_s"]   # the sets the brief names
PRED = PROJ / "results" / "preds"
RECORD = PROJ / "results" / "warning_policy.json"
lk = lambda k: f"l{k}"


def record():
    """The results file under review."""
    with open(RECORD, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def jsave(name, obj):
    """The output of a check: OUT/<name> (never inside the project)."""
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)
    print(f"wrote {OUT / name}")


def jread(name):
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pc(x):
    """A share as the record prints it in the outcome: whole percent; 'no value' for None / NaN."""
    return "no value" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.0f}%"


# ---- the stored forecasts, read with numpy only -------------------------------------------------------------------------------------
class Preds:
    """One stored prediction set: dates (TARGET days), basins, obs [basin, day], pred [basin, day, lead], meta."""

    def __init__(self, name):
        z = np.load(PRED / f"{name}.npz", allow_pickle=False)
        self.name = name
        self.dates = pd.DatetimeIndex(z["dates"])
        self.basins = [str(b) for b in z["basins"]]
        self.obs = np.asarray(z["obs"], float)
        self.pred = np.asarray(z["pred"], float)
        self.meta = json.loads(str(z["meta"]))
        assert self.obs.shape == self.pred.shape[:2] and self.pred.shape[2] == len(LEADS) and self.pred.shape[0] == len(self.basins)

    def perturbed(self, period, rng, obs_too=True):
        """A copy whose forecasts (and, if obs_too, flows) on the TARGET days of `period` are replaced by random values: the made-up-values test."""
        c = Preds.__new__(Preds)
        c.name, c.dates, c.basins, c.meta = self.name + "_perturbed", self.dates, self.basins, self.meta
        m = mask(self.dates, *period)
        c.pred = self.pred.copy()
        c.pred[:, m, :] = rng.uniform(0.0, 50.0, size=c.pred[:, m, :].shape)
        c.obs = self.obs.copy()
        if obs_too:
            c.obs[:, m] = rng.uniform(0.0, 50.0, size=c.obs[:, m].shape)
        return c


def have(name):
    return (PRED / f"{name}.npz").exists()


def seeds_of(name, n=5):
    return [f"{name}_s{i}" for i in range(n) if have(f"{name}_s{i}")]


def mask(dates, a, b):
    """True on the TARGET days a..b (inclusive): a sample belongs to the split part of its target day."""
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


def own_thresholds(S, split):
    """Q95 per basin from the npz's own obs over the TRAINING years of the split (np.quantile, linear interpolation, pandas' default too)."""
    m = mask(S.dates, *split["train"])
    return {g: float(np.quantile(S.obs[i][m], 0.95)) for i, g in enumerate(S.basins)}


def rows(S, thr, lead, period, require_before_finite=False):
    """The pooled rows of one set and lead on the TARGET days of period = (a, b), with the conventions of flood.flood_scores: a row is kept when
    obs and the forecast of this lead are finite; hi = obs > Q95; low = the flow on the issue day (target day - lead) at or below Q95 (a NaN flow
    is not low, as pandas' shift(lead) <= thr gives False); f = forecast; t = the basin's Q95; q_rel = issue-day flow / Q95; day = days since a;
    basin = index of the basin. require_before_finite also drops rows whose issue-day flow is NaN (the extra condition of warning_policy.arrays)."""
    a, b = period
    m = mask(S.dates, a, b)
    out = {k: [] for k in ("hi", "low", "f", "t", "q_rel", "day", "basin")}
    for i, g in enumerate(S.basins):
        o, f = S.obs[i], S.pred[i, :, lead - 1]
        before = np.concatenate([np.full(lead, np.nan), o[:-lead]])          # the flow on the issue day, aligned to the target day
        ok = m & np.isfinite(o) & np.isfinite(f)
        if require_before_finite:
            ok &= np.isfinite(before)
        with np.errstate(invalid="ignore"):
            low = before <= thr[g]
        out["hi"].append(o[ok] > thr[g])
        out["low"].append(low[ok])
        out["f"].append(f[ok])
        out["t"].append(np.full(int(ok.sum()), thr[g]))
        out["q_rel"].append(before[ok] / thr[g])
        out["day"].append(np.asarray((S.dates[ok] - pd.Timestamp(a)).days, int))
        out["basin"].append(np.full(int(ok.sum()), i))
    return {k: np.concatenate(v) for k, v in out.items()}


def warn(R, trigger):
    """A warning is given when the forecast exceeds trigger x Q95 (the form of flood.flood_scores: f > trigger * thr)."""
    return R["f"] > trigger * R["t"]


def div(a, b):
    return a / b if b else float("nan")


def table(hi, w):
    """The warning table of two boolean arrays with the project's rates: pod = hits / (hits + misses), far = false alarms / (hits + false alarms),
    csi = hits / (hits + misses + false alarms), bias = (hits + false alarms) / (hits + misses)."""
    hi, w = np.asarray(hi, bool), np.asarray(w, bool)
    h, m, fa, c = int((hi & w).sum()), int((hi & ~w).sum()), int((~hi & w).sum()), int((~hi & ~w).sum())
    return dict(hits=h, misses=m, false_alarms=fa, correct_neg=c, pod=div(h, h + m), far=div(fa, h + fa), csi=div(h, h + m + fa), bias=div(h + fa, h + m))


def tables(R, trigger):
    """all = every kept row; onset = the rows whose issue-day flow was at or below Q95."""
    w = warn(R, trigger)
    return dict(all=table(R["hi"], w), onset=table(R["hi"][R["low"]], w[R["low"]]))


def choose(R_val, rule, grid=GRID, far_cap=FAR_CAP):
    """My own trigger choice on the VALIDATION rows.
    csi:      the trigger with the best pooled onset CSI; a tie goes to the trigger nearest 1.0 (ties are recorded, so the reader sees whether the tie rule mattered).
    far_cap:  the most onsets warned (POD) among the triggers whose pooled onset FAR is at or below the cap (a NaN FAR = no warning at all does not qualify);
              a tie goes to the better CSI; when no trigger qualifies: reached=False and the trigger with the lowest FAR (NaN last; the larger trigger on a tie)."""
    T = {t: tables(R_val, t)["onset"] for t in grid}
    nz = lambda x, bad: bad if x is None or (isinstance(x, float) and math.isnan(x)) else x
    if rule == "csi":
        best = max(nz(v["csi"], -1.0) for v in T.values())
        cands = [t for t in grid if nz(T[t]["csi"], -1.0) == best]
        pick = min(cands, key=lambda t: abs(t - 1.0))
        return dict(trigger=pick, reached=True, candidates_tied=cands if len(cands) > 1 else [], validation=T[pick])
    if rule == "far_cap":
        ok = [t for t in grid if nz(T[t]["far"], 9.0) <= far_cap]
        if ok:
            best_pod = max(nz(T[t]["pod"], -1.0) for t in ok)
            cands = [t for t in ok if nz(T[t]["pod"], -1.0) == best_pod]
            pick = max(cands, key=lambda t: nz(T[t]["csi"], -1.0))
            return dict(trigger=pick, reached=True, candidates_tied=cands if len(cands) > 1 else [], validation=T[pick])
        low = min(nz(T[t]["far"], 9.0) for t in grid)
        cands = [t for t in grid if nz(T[t]["far"], 9.0) == low]
        pick = max(cands)
        return dict(trigger=pick, reached=False, candidates_tied=cands if len(cands) > 1 else [], validation=T[pick])
    raise ValueError(rule)


def verdict(t, pod_go=POD_GO, far_go=FAR_GO, csi_caution=CSI_CAUTION):
    """The GO / CAUTION / NOT YET reading of one warning table (results/flood_analysis.json verdict.criteria); NO EVENTS without any high day."""
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    far = t["far"]
    if t["pod"] >= pod_go and far is not None and not (isinstance(far, float) and math.isnan(far)) and far <= far_go:
        return "GO"
    return "CAUTION" if t["csi"] >= csi_caution else "NOT YET"


def average_precision(y, s):
    """Average precision of the scores s for the labels y, my own implementation: precision at every distinct score threshold weighted by the recall step
    (the definition sklearn.metrics.average_precision_score uses); checked against sklearn in w1."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    last = np.r_[s[1:] != s[:-1], True]           # the last row of every run of equal scores
    tp, fp = tp[last], fp[last]
    rec, prec = tp / y.sum(), tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def close(a, b, tol=1e-9):
    """Equal numbers, with None / NaN equal to each other."""
    na = a is None or (isinstance(a, float) and math.isnan(a))
    nb = b is None or (isinstance(b, float) and math.isnan(b))
    if na or nb:
        return na and nb
    return abs(float(a) - float(b)) <= tol


def diff(a, b):
    na = a is None or (isinstance(a, float) and math.isnan(a))
    nb = b is None or (isinstance(b, float) and math.isnan(b))
    if na and nb:
        return 0.0
    if na or nb:
        return None
    return float(a) - float(b)
