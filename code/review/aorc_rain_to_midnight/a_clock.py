"""The reviewer's own rebuild of the daily AORC series (no code of 16_aorc_rain_to_midnight.py or fetch_aorc.py is used), for ANY window.

The cached AORC file data/aorc/aorc_<gauge>.npz holds the UTC time stamp (seconds since 1970) of each hour and the basin-mean rain in mm; the stamp is taken, as the stores' metadata and the AWS registry
say ('accumulation ending at the top of each hour'), as the END of the hour. A window of day t ends at END_H o'clock of day t, on the wall clock of the gauge ('wall') or in UTC ('utc'), and starts 24 hours
earlier on that clock; an hour belongs to the window in which it ENDS (an hour ending exactly at the window end is its last hour). Summer time: two independent ways: a hand-written US rule (route R) and
python's zoneinfo (route Z). (Adapted from code/review/rain_to_midnight/r2_clock.py of the E1 review.)
"""
import json, datetime as dt
import numpy as np
from zoneinfo import ZoneInfo

PROJ = str(__import__("pathlib").Path(__file__).resolve().parents[3])   # the project folder (was the Mac path until 2026-10-28)
STD = {"America/New_York": -5, "America/Chicago": -6}
FIRST_DAY = np.datetime64("1980-10-01", "D")
LAST_DAY = np.datetime64("2014-09-30", "D")
N_DAYS = int((LAST_DAY - FIRST_DAY).astype(int)) + 1                    # 12418
D0 = int(FIRST_DAY.astype("int64"))
N_HOURS = 298296
FIRST_STAMP = np.datetime64("1980-09-25T00:00", "m")
LAST_STAMP = np.datetime64("2014-10-05T23:00", "m")


def read_aorc(g):
    """(hour-ending UTC stamps as datetime64[m], basin-mean rain mm, filled flags, n_valid cells, n_cells) of one basin from the cache file."""
    z = np.load(f"{PROJ}/data/aorc/aorc_{g}.npz", allow_pickle=False)
    t = (z["stamp_utc_s"].astype("int64") // 60).astype("datetime64[m]")
    x = z["rain_mm"].astype("float64")
    assert len(t) == len(x) == N_HOURS and t[0] == FIRST_STAMP and t[-1] == LAST_STAMP, "the cache does not hold the span 1980-09-25 .. 2014-10-05"
    assert np.all(np.diff(t) == np.timedelta64(60, "m")) and np.isfinite(x).all() and (x >= 0).all()
    return t, x, z["filled"].astype(bool), z["n_valid"].astype(int), int(z["n_cells"])


def _sunday(year, month, which):
    if which == -1:
        d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
        while d.weekday() != 6:
            d -= dt.timedelta(days=1)
        return d
    d = dt.date(year, month, 1)
    while d.weekday() != 6:
        d += dt.timedelta(days=1)
    return d + dt.timedelta(days=7 * (which - 1))


def dst_bounds_utc(year, std_off):
    if year <= 1986:
        s, e = _sunday(year, 4, -1), _sunday(year, 10, -1)
    elif year <= 2006:
        s, e = _sunday(year, 4, 1), _sunday(year, 10, -1)
    else:
        s, e = _sunday(year, 3, 2), _sunday(year, 11, 1)
    su = dt.datetime(s.year, s.month, s.day, 2) - dt.timedelta(hours=std_off)          # 02:00 standard time
    eu = dt.datetime(e.year, e.month, e.day, 1) - dt.timedelta(hours=std_off)          # 02:00 summer time = 01:00 standard time
    return np.datetime64(su, "m"), np.datetime64(eu, "m")


def offset_rules(t_utc, zone):
    std = STD[zone]
    years = t_utc.astype("datetime64[Y]").astype(int) + 1970
    out = np.full(len(t_utc), std, dtype=int)
    for y in np.unique(years):
        s, e = dst_bounds_utc(int(y), std)
        out[(years == y) & (t_utc >= s) & (t_utc < e)] = std + 1
    return out


def offset_zoneinfo(t_utc, zone):
    z = ZoneInfo(zone)
    secs = t_utc.astype("datetime64[s]").astype("int64")
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return np.array([int(((epoch + dt.timedelta(seconds=int(s))).astimezone(z).utcoffset()).total_seconds() // 3600) for s in secs], dtype=int)


def window_day(t_utc, off, kind, end_h):
    """Day index (days since 1970-01-01) of the window that holds the hour ending at each UTC stamp."""
    end = t_utc if kind == "utc" else t_utc + off.astype("timedelta64[h]").astype("timedelta64[m]")      # the end of each hour on the clock of the window
    return ((end - np.timedelta64(end_h * 60 + 1, "m")).astype("datetime64[D]")).astype("int64") + 1


def window_sum(t_utc, x, off, kind, end_h):
    d = window_day(t_utc, off, kind, end_h) - D0
    keep = (d >= 0) & (d < N_DAYS)
    return np.bincount(d[keep], weights=x[keep], minlength=N_DAYS), np.bincount(d[keep], minlength=N_DAYS)


def local_instant_utc(day64, hour, zone, how="zoneinfo"):
    """UTC instant (datetime64[m]) of HOUR o'clock wall-clock time on the calendar day day64 (hour 0..24 allowed: 24 = 00:00 of the next day); no ambiguous hours are used (02:00 is never asked for)."""
    d = (day64.astype("datetime64[D]") + (hour // 24)).astype(dt.date)
    h = hour % 24
    if how == "zoneinfo":
        loc = dt.datetime(d.year, d.month, d.day, h, tzinfo=ZoneInfo(zone))
        return np.datetime64(loc.astimezone(dt.timezone.utc).replace(tzinfo=None), "m")
    std = STD[zone]
    s, e = dst_bounds_utc(d.year, std)
    naive_std = np.datetime64(dt.datetime(d.year, d.month, d.day, h), "m") - np.timedelta64(std * 60, "m")    # the instant if standard time were in force
    summer = (naive_std - np.timedelta64(60, "m") >= s) and (naive_std - np.timedelta64(60, "m") < e)       # in summer the wall clock is one hour ahead: the same wall time is one hour earlier in UTC
    return naive_std - np.timedelta64(60 if summer else 0, "m")
