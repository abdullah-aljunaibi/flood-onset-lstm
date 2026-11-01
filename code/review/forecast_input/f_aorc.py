"""The reviewer's own AORC local-day series (no code of 16_aorc_rain_to_midnight.py, fetch_aorc.py or gefs_to_basins.py is used; the cache data/aorc/aorc_<gauge>.npz is read directly) and the zone of each gauge.
An AORC value stamped T is the rain of the hour ENDING at T (E1b review, c2); a local wall-clock day D (summer time included) holds the hours whose stamp lies in (D 00:00, D+1 00:00] on the gauge's wall clock;
the sum of the hours of the day is an exactly rounded sum (math.fsum) of the hourly values between the two UTC instants (a route different from the builder's, which assigns every hour to a day and sums per day with pandas). The zone of each gauge is read from the
'tz_cd' column of data/hourly/usgs_sites.rdb (EST -> America/New_York, CST -> America/Chicago)."""
import datetime as dt, math
from zoneinfo import ZoneInfo
from f_common import *

N_HOURS = 298296
FIRST_STAMP = np.datetime64("1980-09-25T00:00", "m")
LAST_STAMP = np.datetime64("2014-10-05T23:00", "m")


def gauge_zones(ids):
    rows = [l.rstrip("\n").split("\t") for l in open(f"{PROJ}/data/hourly/usgs_sites.rdb") if not l.startswith("#")]
    head, body = rows[0], rows[2:]
    i_site, i_tz = head.index("site_no"), head.index("tz_cd")
    tz = {r[i_site]: r[i_tz] for r in body}
    zones = {g: {"EST": "America/New_York", "CST": "America/Chicago"}[tz[g]] for g in ids}
    return zones, tz


def read_aorc(g):
    """(hour-ending UTC stamps as datetime64[m], basin-mean rain mm) of one basin from the cache file."""
    z = np.load(f"{PROJ}/data/aorc/aorc_{g}.npz", allow_pickle=False)
    t = (z["stamp_utc_s"].astype("int64") // 60).astype("datetime64[m]")
    x = z["rain_mm"].astype("float64")
    assert len(t) == len(x) == N_HOURS and t[0] == FIRST_STAMP and t[-1] == LAST_STAMP and np.all(np.diff(t) == np.timedelta64(60, "m")) and np.isfinite(x).all() and (x >= 0).all()
    return t, x


def local_midnight_utc(day64, zone):
    """UTC instant (datetime64[m]) of 00:00 on the wall clock of `zone` on the calendar day day64 (zoneinfo; no US local midnight is ambiguous or missing)."""
    d = day64.astype("datetime64[D]").astype(dt.date)
    loc = dt.datetime(d.year, d.month, d.day, 0, tzinfo=ZoneInfo(zone))
    return np.datetime64(loc.astimezone(dt.timezone.utc).replace(tzinfo=None), "m")


def aorc_local_days(g, zone, days):
    """AORC rain (mm) of each local wall-clock day in `days` (datetime64[D] array) for gauge g: sum of the hours whose stamp lies in (start, end]; also the number of hours of each day."""
    t, x = read_aorc(g)
    a = np.array([local_midnight_utc(d, zone) for d in days])
    b = np.array([local_midnight_utc(d + 1, zone) for d in days])
    ia = ((a - t[0]) / np.timedelta64(60, "m")).astype(int)         # index of the stamp = start of the day: the hours with stamp > start begin at ia + 1
    ib = ((b - t[0]) / np.timedelta64(60, "m")).astype(int)
    assert ia.min() >= 0 and ib.max() < len(t)
    # the sum of the hours with a stamp in (start, end]: an exactly rounded sum of the hourly values (math.fsum), so that no summation order or cumulative-sum round-off enters the float64 value of a day
    return np.array([math.fsum(x[i + 1:j + 1]) for i, j in zip(ia, ib)]), (ib - ia)
