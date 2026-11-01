"""Shared setup of the independent reviewer's round-2 scripts for experiment E1b (code/16_aorc_rain_to_midnight.py, AORC rain to midnight) after seeds 3 and 4 were
added on 30 October 2026, and for the rerun of code/check_stage4_realtime.py on the five wqa networks.

An AI review: own numpy scoring from results/preds (no flood.evaluate), nothing trained, one torch thread, nothing written into the project except this folder and the
block round_2 of results/review_aorc_rain_to_midnight.json (r6_assemble.py --write-into-project). Outputs go to OUT (environment variable REVIEW_OUT; default
<system temp>/aorc5_review). Run from the project root:   PYTHONUTF8=1 <python> code/review/aorc_rain_to_midnight_5seeds/<script>.py
"""
import hashlib, json, os, sys, tempfile
from pathlib import Path

os.environ.setdefault("FLOOD_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
sys.path.insert(0, str(PROJ / "code"))
sys.path.insert(0, str(HERE))
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "aorc5_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
LEADS = (1, 2, 3)
RESULTS = "results/aorc_rain_to_midnight.json"
STAGE4 = "results/stage4_realtime.json"
SCRIPT = "code/16_aorc_rain_to_midnight.py"
STAGE4_SCRIPT = "code/check_stage4_realtime.py"
REVIEW = "results/review_aorc_rain_to_midnight.json"
BACKUP = next((b for b in (PROJ.parent / "_archive" / "backups" / "Project-Flood_backup_2026-10-30",   # moved here on 30 Oct 2026 (folder tidy-up)
                           PROJ.parent / "Project-Flood_backup_2026-10-30") if b.exists()), PROJ.parent / "Project-Flood_backup_2026-10-30")
READ = {}                                                         # every project file my numbers read -> sha256 (filled by sha256())


def jload(rel, root=PROJ):
    with open(Path(root) / rel, encoding="utf-8") as f:
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


def sha256(rel, root=PROJ):
    f = Path(root) / rel
    if not f.exists():
        return None
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if Path(root) == PROJ:
        READ[str(rel).replace("\\", "/")] = h
    return h


def load_npz(name):
    """(dates, basins, obs[basin, day], pred[basin, day, lead], meta) of results/preds/<name>.npz, numpy only."""
    rel = f"results/preds/{name}.npz"
    sha256(rel)
    z = np.load(PROJ / rel, allow_pickle=False)
    out = pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]], z["obs"].astype(float), z["pred"].astype(float), json.loads(str(z["meta"]))
    z.close()
    return out


def within(dates, a, b):
    return np.asarray((dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b)))


_THR = {}


def thresholds(dates, basins, obs):
    """Q95 of the training years from the stored observed flow (pandas quantile = linear, NaN skipped)."""
    key = tuple(basins)
    if key not in _THR:
        tr = within(dates, *SPLIT["train"])
        _THR[key] = np.array([np.nanquantile(obs[i][tr], 0.95) for i in range(len(basins))])
    return _THR[key]


def rates(h, m, f):
    return dict(hits=int(h), misses=int(m), false_alarms=int(f), pod=h / (h + m) if h + m else float("nan"), far=f / (h + f) if h + f else float("nan"),
                csi=h / (h + m + f) if h + m + f else float("nan"))


def score_arrays(dates, basins, obs, pred, trigger=1.0, part="test"):
    """Own scoring of one prediction array: per lead the per-basin NSE, pooled onset and all-day tables, and the onset hit flags in a fixed order (basin, day)."""
    thr = thresholds(dates, basins, obs)
    m_part = within(dates, *SPLIT[part])
    res = {}
    for k in LEADS:
        nse, oh, om, of, ah, am, af = [], 0, 0, 0, 0, 0, 0
        onset_keys, onset_hit = [], []
        for i, g in enumerate(basins):
            o, s = obs[i], pred[i, :, k - 1]
            ok = m_part & np.isfinite(o) & np.isfinite(s)
            oo, ss = o[ok], s[ok]
            nse.append(1.0 - np.sum((ss - oo) ** 2) / np.sum((oo - oo.mean()) ** 2))
            issue = np.concatenate([np.full(k, np.nan), o[:-k]])
            low = issue <= thr[i]                                     # NaN -> False, as pandas' shift(lead) <= thr
            hi, w = o > thr[i], s > trigger * thr[i]
            ah += int((ok & hi & w).sum()); am += int((ok & hi & ~w).sum()); af += int((ok & ~hi & w).sum())
            on = ok & low
            oh += int((on & hi & w).sum()); om += int((on & hi & ~w).sum()); of += int((on & ~hi & w).sum())
            idx = np.flatnonzero(on & hi)
            onset_keys += [(g, int(j)) for j in idx]; onset_hit += list(w[idx])
        nse = np.array(nse)
        res[k] = dict(nse=nse, nse_median=float(np.median(nse)), nse_mean=float(nse.mean()), onset=rates(oh, om, of), all=rates(ah, am, af),
                      onset_keys=onset_keys, onset_hit=np.array(onset_hit, bool))
    return res


_CACHE = {}


def score(name, trigger=1.0):
    """Scores of results/preds/<name>.npz (cached; the array itself is dropped at once: one prediction set in memory at a time)."""
    key = (name, trigger)
    if key not in _CACHE:
        d, b, o, p, meta = load_npz(name)
        r = score_arrays(d, b, o, p, trigger)
        r["meta_seeds"] = meta.get("seeds")
        r["basins"] = b
        _CACHE[key] = r
        del p
    return _CACHE[key]


def seeds_of(name):
    """The single-seed prediction sets that exist for lstm_<name>."""
    return [s for s in range(10) if (PROJ / "results" / "preds" / f"lstm_{name}_s{s}.npz").exists()]


def close(a, b, tol=1e-9):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return False
    return abs(float(a) - float(b)) <= tol


class Compare:
    """(key, file value, my value) triples."""

    def __init__(self, tol=1e-9):
        self.rows, self.tol = {}, tol

    def add(self, key, file_value, my_value, tol=None):
        t = self.tol if tol is None else tol
        if isinstance(file_value, bool) or isinstance(my_value, bool):
            ok = file_value is not None and bool(file_value) == bool(my_value)
        elif isinstance(file_value, (str, list, dict)):
            ok = file_value == my_value
        else:
            ok = close(file_value, my_value, t)
        self.rows[key] = dict(file=file_value, mine=my_value, ok=bool(ok))
        return ok

    def summary(self):
        bad = [k for k, v in self.rows.items() if not v["ok"]]
        return dict(entries=len(self.rows), matching=len(self.rows) - len(bad), not_matching=bad)
