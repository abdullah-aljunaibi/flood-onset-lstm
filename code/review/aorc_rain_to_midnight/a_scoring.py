"""The reviewer's own scoring helpers (no flood.evaluate): per-basin NSE and onset contingency counts straight from the npz arrays (copied from the E1 review, code/review/rain_to_midnight/r2_scoring.py)."""
from a_common import *
import flood as F

TEST = ("2005-10-01", "2014-09-30")
TRAIN = ("1980-10-01", "2000-09-30")
_D = None
def data():
    global _D
    if _D is None:
        _D = F.load_data()
    return _D

def load_npz(name):
    z = np.load(f"{PROJ}/results/preds/{name}.npz", allow_pickle=False)
    return pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]], z["obs"], z["pred"]

def thresholds(basins, dates):
    D = data()
    tr = np.asarray((dates >= TRAIN[0]) & (dates <= TRAIN[1]))
    return {g: float(np.quantile(D[g].q_mm.values[tr], 0.95)) for g in basins}

def part(dates, a, b):
    return np.asarray((dates >= a) & (dates <= b))

def nse_per_basin(name, lead, a=TEST[0], b=TEST[1]):
    dates, basins, obs, pred = load_npz(name)
    test = part(dates, a, b)
    out = {}
    for i, g in enumerate(basins):
        o, s = obs[i], pred[i, :, lead - 1]
        ok = test & np.isfinite(o) & np.isfinite(s)
        o, s = o[ok], s[ok]
        out[g] = 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)
    return out

def onset_table(name, lead, trigger=1.0, a=TEST[0], b=TEST[1]):
    """{basin: (target dates, high, warn)} on the onset subset of the target days a..b, own definitions: onset = the flow on the issue day (target day - lead) at or below the threshold."""
    dates, basins, obs, pred = load_npz(name)
    thr = thresholds(basins, dates)
    test = part(dates, a, b)
    res = {}
    for i, g in enumerate(basins):
        o = obs[i]; f = pred[i, :, lead - 1]
        issue = np.concatenate([np.full(lead, np.nan), o[:-lead]])
        ok = test & np.isfinite(o) & np.isfinite(f) & (issue <= thr[g])
        res[g] = (dates[ok], (o > thr[g])[ok], (f > trigger * thr[g])[ok])
    return res

def counts(name, lead, trigger=1.0, a=TEST[0], b=TEST[1]):
    t = onset_table(name, lead, trigger, a, b)
    H = sum(int((h & w).sum()) for _, h, w in t.values())
    M = sum(int((h & ~w).sum()) for _, h, w in t.values())
    Fa = sum(int((~h & w).sum()) for _, h, w in t.values())
    return dict(hits=H, misses=M, false_alarms=Fa, n_onsets=H + M, pod=H / (H + M), far=Fa / (H + Fa) if H + Fa else float("nan"), csi=H / (H + M + Fa))
