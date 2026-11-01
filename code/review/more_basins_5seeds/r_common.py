"""Shared helpers of round 2 of the independent review of E3 (code/17_more_basins.py), five seeds (30 October 2026).

The reviewer's own numpy readers and scorers. Nothing here imports the project's engine (code/flood.py); only r6_networks.py does, to reload networks.
Nothing is trained and nothing is written into the project: outputs go to $REVIEW_OUT (default <temp>/e3_5seeds_review); only
r7_assemble.py --write-into-project appends the block round_2 to results/review_more_basins.json.
"""
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np

PROJ = pathlib.Path(os.environ.get("E3_PROJECT", pathlib.Path(__file__).resolve().parents[3]))
OUT = pathlib.Path(os.environ.get("REVIEW_OUT", pathlib.Path(tempfile.gettempdir()) / "e3_5seeds_review"))
OUT.mkdir(parents=True, exist_ok=True)
PRED = PROJ / "results" / "preds"
MODELS = PROJ / "results" / "models"
RESULTS = PROJ / "results" / "more_basins.json"
BACKUP_DIR = PROJ.parent / "Project-Flood_backup_2026-10-30"
BACKUP = BACKUP_DIR / "results" / "more_basins.json"
SPLIT = {"train": ("1980-10-01", "2000-09-30"), "val": ("2000-10-01", "2005-09-30"), "test": ("2005-10-01", "2014-09-30")}
LEADS = (1, 2, 3)
NEW = ["lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic"]
SEEDS = [0, 1, 2, 3, 4]
CTRL = ["lstm_wq", "lstm_w", "lstm_wqf"]
CTRL_SEEDS = [0, 1, 2]
LOBO = ["lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w"]
ALL_SETS = NEW + CTRL + LOBO
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.05)]
P_CUT = 0.05


def seeds_of(n):
    return SEEDS if n in NEW else CTRL_SEEDS if n in CTRL else []


def files_of(n):
    """The prediction files of a set: the seed average (or the single training of 05) and its seed files."""
    return [n] + [f"{n}_s{s}" for s in seeds_of(n)]


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(clean(obj), indent=1), encoding="utf-8")


def load(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def results(path=RESULTS):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


# ---- a prediction set, read with numpy alone ------------------------------------------------------------------------------------------------------
def load_npz(name):
    """basins (list of str), dates (datetime64[D]), obs [basin, day], pred [basin, day, lead], meta (dict). Arrays copied out, file closed."""
    with np.load(PRED / f"{name}.npz", allow_pickle=False) as z:
        return dict(basins=[str(b) for b in z["basins"]], dates=z["dates"].astype("datetime64[D]"), obs=np.array(z["obs"], float),
                    pred=np.array(z["pred"], float), meta=json.loads(str(z["meta"])))


def mask(dates, part):
    a, b = (np.datetime64(x) for x in SPLIT[part])
    return (dates >= a) & (dates <= b)


def q95(x):
    """95th percentile, linear interpolation, NaN left out (numpy's default method)."""
    x = np.asarray(x, float)
    return float(np.quantile(x[np.isfinite(x)], 0.95))


def thresholds(z):
    """Observed high-flow level of each basin: the 95th percentile of the stored observed flow over the training years."""
    tr = mask(z["dates"], "train")
    return {g: q95(z["obs"][i][tr]) for i, g in enumerate(z["basins"])}


def own_nse(o, s):
    return float(1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def table(h, m, f, c):
    d = lambda a, b: a / b if b else float("nan")
    h, m, f, c = int(h), int(m), int(f), int(c)
    return dict(hits=h, misses=m, false_alarms=f, correct_neg=c, pod=d(h, h + m), far=d(f, h + f), csi=d(h, h + m + f))


def verdict(t, pod_go=0.70, far_go=0.30, csi_caution=0.40):
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= pod_go and t["far"] <= far_go:
        return "GO"
    return "CAUTION" if t["csi"] >= csi_caution else "NOT YET"


class Scored:
    """One prediction set, scored with the reviewer's own code.

    Per lead k and split part: per-basin NSE on the days of the part with an observation and a forecast; warning cells with high = obs > thr,
    warning = forecast > trigger * thr, onset = the observed flow k days earlier (by array position; the dates are checked contiguous) at or below thr."""

    def __init__(self, name, thr=None):
        z = load_npz(name)
        self.name, self.basins, self.dates, self.meta = name, z["basins"], z["dates"], z["meta"]
        assert np.all(np.diff(self.dates).astype(int) == 1), f"{name}: dates are not contiguous"
        self.obs, self.pred = z["obs"], z["pred"]
        self.thr = thr if thr is not None else thresholds(z)
        self.nb = len(self.basins)

    def _issue_low(self, i, k):
        o = self.obs[i]
        iss = np.full(len(o), np.nan)
        iss[k:] = o[:-k]
        return iss <= self.thr[self.basins[i]]

    def cells(self, k, part, trigger=1.0, level=None):
        """Pooled warning tables (all, onset) and per-basin tables at lead k. level: {basin: {lead: mm}} gives each basin its own trigger level / thr."""
        sel = mask(self.dates, part)
        per, tot = {}, {"all": np.zeros(4, int), "onset": np.zeros(4, int)}
        for i, g in enumerate(self.basins):
            o, s, t = self.obs[i], self.pred[i, :, k - 1], self.thr[g]
            ok = sel & np.isfinite(o) & np.isfinite(s)
            low, hi = self._issue_low(i, k), o > t
            trig = level[g][k] / t if level is not None else trigger
            w = s > trig * t
            per[g] = {}
            for kind, m in (("all", ok), ("onset", ok & low)):
                c = np.array([(hi & w & m).sum(), (hi & ~w & m).sum(), (~hi & w & m).sum(), (~hi & ~w & m).sum()])
                per[g][kind] = table(*c)
                tot[kind] += c
        return {kind: table(*tot[kind]) for kind in tot}, per

    def onset_vector(self, k, part="test"):
        """The onsets of the part pooled over the basins in basin order: (warned at the observed level, keys (basin, day index))."""
        sel = mask(self.dates, part)
        out, keys = [], []
        for i, g in enumerate(self.basins):
            o, s, t = self.obs[i], self.pred[i, :, k - 1], self.thr[g]
            ok = sel & np.isfinite(o) & np.isfinite(s)
            ev = ok & self._issue_low(i, k) & (o > t)
            out.append((s > t)[ev])
            keys += [(g, int(j)) for j in np.flatnonzero(ev)]
        return np.concatenate(out), keys

    def nse(self, k, part="test"):
        sel = mask(self.dates, part)
        r = {}
        for i, g in enumerate(self.basins):
            o, s = self.obs[i], self.pred[i, :, k - 1]
            ok = sel & np.isfinite(o) & np.isfinite(s)
            r[g] = own_nse(o[ok], s[ok])
        return r

    def coverage_min(self, k, part):
        sel = mask(self.dates, part)
        return min(float((sel & np.isfinite(self.obs[i]) & np.isfinite(self.pred[i, :, k - 1])).sum() / sel.sum()) for i in range(self.nb))

    def own_levels(self):
        """The model's own warning level (the rule of 05_lobo.py): per basin and lead, the 95th percentile of its training-years forecasts (by target day)."""
        tr = mask(self.dates, "train")
        return {g: {k: q95(self.pred[i, :, k - 1][tr]) for k in LEADS} for i, g in enumerate(self.basins)}

    def summary(self, k, part="test"):
        v = self.nse(k, part)
        pooled, per = self.cells(k, part)
        arr = np.array([v[g] for g in self.basins])
        return dict(nse_median=float(np.median(arr)), nse_mean=float(arr.mean()), onset=pooled["onset"], all=pooled["all"], nse=v, per=per)

    def choose_level(self, k, kind):
        """The warning level (multiple of the high-flow level) with the best pooled validation-years CSI; a tie goes to the level closest to 1.0 (the first such)."""
        csi = []
        for m in LEVELS:
            t, _ = self.cells(k, "val", trigger=m)
            c = t[kind]["csi"]
            csi.append(c if c == c else -1.0)
        csi = np.array(csi)
        best = np.flatnonzero(csi >= csi.max() - 1e-12)
        return LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]


# ---- tests (own code) -------------------------------------------------------------------------------------------------------------------------------
def wilcoxon_exact(a, b):
    """Two-sided exact Wilcoxon signed-rank p by the full distribution of the rank sum (2^n sign patterns); None when a difference is zero or tied."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.any(d == 0) or len(np.unique(np.abs(d))) < len(d):
        return None
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w_plus = int(ranks[d > 0].sum())
    total = n * (n + 1) // 2
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for r in range(1, n + 1):
        counts[r:] = counts[r:] + counts[:-r].copy()
    w = min(w_plus, total - w_plus)
    return float(min(1.0, 2 * counts[: w + 1].sum() / 2 ** n))


def mcnemar_exact(only_a, only_b):
    n = only_a + only_b
    if n == 0:
        return float("nan")
    k = min(only_a, only_b)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def spearman(x, y):
    """Spearman rho (own: Pearson correlation of average ranks) and the p of scipy.stats.spearmanr (t approximation, as the script)."""
    from scipy.stats import rankdata, spearmanr
    rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return dict(rho=rho, p=float(spearmanr(x, y).pvalue))


def close(a, b, tol=1e-9):
    nan = lambda x: x is None or (isinstance(x, float) and x != x)
    if nan(a) or nan(b):
        return nan(a) and nan(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def compare(mine, theirs, path="", out=None, tol=1e-9):
    """Walk my numbers; every leaf is compared with the same path in the results file. Returns [(path, mine, theirs, equal)]."""
    out = [] if out is None else out
    if isinstance(mine, dict):
        for k, v in mine.items():
            if not isinstance(theirs, dict) or k not in theirs:
                out.append((f"{path}/{k}", "<value>", "<missing>", False))
            else:
                compare(v, theirs[k], f"{path}/{k}", out, tol)
    elif isinstance(mine, (list, tuple)):
        if not isinstance(theirs, list) or len(theirs) != len(mine):
            out.append((path, str(mine)[:120], str(theirs)[:120], False))
        else:
            for i, (a, b) in enumerate(zip(mine, theirs)):
                compare(a, b, f"{path}[{i}]", out, tol)
    else:
        out.append((path, mine, theirs, close(mine, theirs, tol)))
    return out
