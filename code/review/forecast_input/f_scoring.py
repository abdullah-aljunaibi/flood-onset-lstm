"""The reviewer's own scoring helpers (no flood.evaluate): per-basin NSE and onset contingency counts straight from the npz arrays, for the two splits of E2 (thresholds from the training years OF THE SET'S OWN SPLIT).
Adapted from the E1 / E1b reviews (code/review/rain_to_midnight/r2_scoring.py, code/review/aorc_rain_to_midnight/a_scoring.py)."""
from f_common import *
import flood as F

_D = None
def data():
    global _D
    if _D is None:
        _D = F.load_data()
    return _D

def arm_of(name):
    """The arm ('main' or 'sat') of a prediction file name: lstm_<set>, lstm_<set>_s<seed>, base_<model>. A set ending in _s is of the satellite-era split (lstm_wqa_s: the set wqa_s; lstm_wqa_s0: the set wqa, seed 0)."""
    import re
    if not name.startswith("lstm_"):
        return "main"
    rest = name[5:]
    m = re.match(r"^(.*)_s([0-9])$", rest)
    if m and m.group(1) in ARM:
        return ARM[m.group(1)]
    return ARM[rest]

def split_of(name):
    return SPLITS[arm_of(name)]

def load_npz(name):
    z = np.load(f"{PROJ}/results/preds/{name}.npz", allow_pickle=False)
    return pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]], z["obs"], z["pred"]

def thresholds(basins, dates, split):
    """95th percentile of the daily flow over the training years of the split (own computation from the npz's own obs)."""
    D = data()
    tr = np.asarray((dates >= split["train"][0]) & (dates <= split["train"][1]))
    return {g: float(np.quantile(D[g].q_mm.values[tr], 0.95)) for g in basins}

def part(dates, a, b):
    return np.asarray((dates >= a) & (dates <= b))

def window(name, which="test"):
    s = split_of(name)
    return s[which]

def nse_per_basin(name, lead, which="test", a=None, b=None):
    dates, basins, obs, pred = load_npz(name)
    a, b = (a, b) if a else window(name, which)
    test = part(dates, a, b)
    out = {}
    for i, g in enumerate(basins):
        o, s = obs[i], pred[i, :, lead - 1]
        ok = test & np.isfinite(o) & np.isfinite(s)
        o, s = o[ok], s[ok]
        out[g] = 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    return out

def onset_table(name, lead, trigger=1.0, which="test", a=None, b=None):
    """{basin: (target dates, high, warn)} on the onset subset of the target days a..b of the set's own split: onset = the flow on the issue day (target day - lead) at or below the threshold."""
    dates, basins, obs, pred = load_npz(name)
    thr = thresholds(basins, dates, split_of(name))
    a, b = (a, b) if a else window(name, which)
    test = part(dates, a, b)
    res = {}
    for i, g in enumerate(basins):
        o = obs[i]; f = pred[i, :, lead - 1]
        issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
        ok = test & np.isfinite(o) & np.isfinite(f) & (issue <= thr[g])
        res[g] = (dates[ok], (o > thr[g])[ok], (f > trigger * thr[g])[ok])
    return res

def counts(name, lead, trigger=1.0, which="test", a=None, b=None):
    t = onset_table(name, lead, trigger, which, a, b)
    H = sum(int((h & w).sum()) for _, h, w in t.values())
    M = sum(int((h & ~w).sum()) for _, h, w in t.values())
    Fa = sum(int((~h & w).sum()) for _, h, w in t.values())
    return dict(hits=H, misses=M, false_alarms=Fa, n_onsets=H + M, pod=H / (H + M), far=Fa / (H + Fa) if H + Fa else float("nan"), csi=H / (H + M + Fa))
