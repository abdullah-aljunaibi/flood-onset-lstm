"""Helper for the Stage IV analyses: the hourly Stage IV basin means of the builder (data/aorc/stage_iv_hindsight.csv, 50 days) and of the reviewer (CACHE/stage4_means.csv, fetched by a6_stage4.py), and the
AORC series with the same hour-ending labels; windows of a local day t by the wall clock of each gauge."""
import sys; sys.dont_write_bytecode = True
from a_common import *
import flood as F
import a_clock as C

ids = list(F.load_data())
DW = jload("results/day_windows.json")
UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}
ZONE = {g: "America/Chicago" if UTC_OFF[g] == -6 else "America/New_York" for g in ids}


def stage4(which="both"):
    """DataFrame of hourly Stage IV basin means (mm), index = hour-ending UTC label; which: 'builder', 'mine' or 'both'."""
    b = pd.read_csv(f"{PROJ}/data/aorc/stage_iv_hindsight.csv", index_col=0, parse_dates=True)
    m = pd.read_csv(f"{CACHE}/stage4_means.csv", index_col=0, parse_dates=True) if os.path.exists(f"{CACHE}/stage4_means.csv") else pd.DataFrame(columns=ids)
    if which == "builder": return b[ids]
    if which == "mine": return m[ids]
    both = pd.concat([b[ids], m[ids][~m.index.isin(b.index)]]).sort_index()
    return both


_A = {}
def aorc(g):
    """AORC hourly basin mean as a Series indexed by the hour-ending UTC label (datetime64[ns])."""
    if g not in _A:
        t, x, fl, nv, nc = C.read_aorc(g)
        _A[g] = pd.Series(x, index=pd.DatetimeIndex(t.astype("datetime64[ns]")))
    return _A[g]


def inst(day, hour, g):
    """UTC instant (pandas Timestamp) of HOUR o'clock wall-clock time of the calendar day `day` for gauge g (hour may exceed 24: day + hour // 24)."""
    return pd.Timestamp(C.local_instant_utc(np.datetime64(pd.Timestamp(day), "D"), hour, ZONE[g], "zoneinfo"))


def hours_between(a, b):
    """hour-ending labels in (a, b] (a and b whole UTC hours)."""
    return pd.date_range(a + pd.Timedelta(hours=1), b, freq="h")
