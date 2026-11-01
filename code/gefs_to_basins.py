"""E2, step 2 (data): the GEFSv12 reforecast rain of the control member (data/gefs/c00, code/fetch_gefs.py) as basin means for the local days t+1, t+2 and t+3 of every issue day t.

For each of the 12 study basins (results/data_audit.json, selected) and each issue day t (a local calendar day, 2000-01-01 .. 2014-09-30):
 1. area weights of the 0.25 degree grid boxes of the forecast model over the basin outline: the exact intersection areas of each box (centre +- 0.125 degree) with the outline (CAMELS
    HCDN_nhru_final_671.shp, read as code/fetch_aorc.py reads it) in an equal-area projection (EPSG:5070, NAD83 / Conus Albers, through pyproj), checked against a fine raster of the outline;
 2. the run used is the 00 UTC run of the UTC date t+1 (the latest run that exists at the issue time, local midnight at the end of day t: 04, 05 or 06 UTC of that date; results/data_access.json,
    gefs_reforecast_record.task5_issue_time). Local day t+k (k = 1, 2, 3) is the wall-clock day of the gauge (America/New_York, America/Chicago for Leaf River, summer time included: the same day
    convention as the AORC input of 16_aorc_rain_to_midnight.py): the forecast rain of that day is the sum of the 3-hourly amounts whose windows lie inside the day's UTC start and end, a window that
    straddles a boundary being split in proportion to the time inside (1/3 or 2/3 of it); the amounts are the basin mean (the weights above) of the windows' rain;
 3. the AORC rain of the same local days t+k, from the hourly basin means of data/aorc summed over the wall-clock day by the code of 16_aorc_rain_to_midnight.py (copied here): the observed quantity
    the forecast is for, stored beside it;
 4. the forecast's own skill against it (correlation, mean ratio, by season and over the validation years, the Brier skill against climatology of days with at least 5 and 10 mm, and the lag check
    that the forecast for day t+k matches the AORC rain of day t+k and not of t+k-1 or t+k+1), and a bias factor per basin and lead = mean AORC / mean forecast over the VALIDATION years
    (target days 2000-10-01 .. 2005-09-30) only, applied to every day.
The clock is asserted for every basin and every day (the run's init is at or before the issue time; no fresher run exists; every forecast hour used lies after the issue time and inside the day's UTC span;
no hour beyond 78), by a second route from absolute times, and by replacing the later runs with made-up values (no forecast issued on or before day c changes).
Writes data/gefs/basins/gefs_c00_<gauge_id>.csv (one row per issue day: the run's date, the raw and the bias-corrected forecast rain of the days t+1..t+3 and the AORC rain of the same days, mm) and
results/gefs_basins.json (the weights, every check, the skill numbers, the bias factors, the caveats). No training. Reads only numpy-readable files (fetch_gefs.load_gefs: no GRIB decoder).
Run: python code/gefs_to_basins.py   (about a minute); 18_forecast_input.py calls ensure() when the files are missing.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

import fetch_aorc as AO
import flood as F
from common import DATA, RES
from fetch_gefs import FULL, LAT_BOX, LON_BOX, NSTEP, STEPS, load_gefs

SCRIPT = "gefs_to_basins.py"
OUT_DIR = DATA / "gefs" / "basins"
RECORD = RES / "gefs_basins.json"
CSV = lambda g: OUT_DIR / f"gefs_c00_{g}.csv"
FIRST_ISSUE, LAST_ISSUE = pd.Timestamp("2000-01-01"), pd.Timestamp("2014-09-30")        # the issue days: local calendar days; the project's record ends 2014-09-30
KS = (1, 2, 3)
HALF = 0.125                                                                            # a 0.25 degree grid box is its centre +- 0.125 degree
VAL = ("2000-10-01", "2005-09-30")                                                      # the validation years of the main split (flood.SPLIT["val"]): the bias factor is fitted on target days here
TEST = F.SPLIT["test"]
assert VAL == F.SPLIT["val"], "the validation years of the engine are not those this script fits the bias factor on"
HOUR_S = 3600
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
THRESHOLDS_MM = (5.0, 10.0)                                                              # days with at least this much rain (Guan et al. 2022 use 5 mm; 10 mm is the heavier class)
N_CUTS = 3                                                                               # random cut days per basin for the made-up-values test (the first days of 23 and of 25 hours are added)
LATENCY_FILE = RES / "data_access.json"
ZONE = {-5: "America/New_York", -6: "America/Chicago"}
HOUR = pd.Timedelta(hours=1)


def to_s(idx):
    """A naive DatetimeIndex (read as UTC) -> epoch seconds, int64."""
    return pd.DatetimeIndex(idx).as_unit("s").asi8.astype(np.int64)


def local_midnights_utc_s(days, zone):
    """00:00 on the wall clock of the zone at the start of each calendar day (naive dates), as a UTC time in epoch seconds. A local midnight that does not exist or is ambiguous would raise (none in the US)."""
    return to_s(pd.DatetimeIndex(days).tz_localize(ZoneInfo(zone)).tz_convert("UTC").tz_localize(None))


# ---- 1. the weights of the grid boxes over the outline -----------------------------------------------------------------------------------------
def cell_weights(poly, to_equal_area):
    """The 0.25 degree boxes (centre +- 0.125 degree, the model's grid boxes; no interpolation) that overlap the outline, with the area of each overlap in km2 in an equal-area projection and its share of the outline. The
    edges of a box are densified (0.01 degree) before they are projected, so that the projected box follows its parallels."""
    import shapely
    from shapely.geometry import box
    from shapely.ops import transform
    P = transform(to_equal_area, poly)
    x0, y0, x1, y1 = poly.bounds
    ii = np.flatnonzero((LAT_BOX + HALF > y0) & (LAT_BOX - HALF < y1))
    jj = np.flatnonzero((LON_BOX + HALF > x0) & (LON_BOX - HALF < x1))
    assert len(ii) and len(jj) and ii.min() > 0 and jj.min() > 0 and ii.max() < len(LAT_BOX) - 1 and jj.max() < len(LON_BOX) - 1, "an outline reaches the edge of the stored box"
    cells = []
    for i in ii:
        for j in jj:
            b = shapely.segmentize(box(LON_BOX[j] - HALF, LAT_BOX[i] - HALF, LON_BOX[j] + HALF, LAT_BOX[i] + HALF), 0.01)
            a = transform(to_equal_area, b)
            inter = P.intersection(a).area
            if inter > 0:
                cells.append(dict(ilat=int(i), ilon=int(j), lat=float(LAT_BOX[i]), lon=float(LON_BOX[j]), overlap_km2=float(inter / 1e6), cell_km2=float(a.area / 1e6)))
    tot = sum(c["overlap_km2"] for c in cells)
    for c in cells:
        c["weight"] = c["overlap_km2"] / tot
        c["share_of_the_box_inside_the_basin"] = c["overlap_km2"] / c["cell_km2"]
    return cells, float(P.area / 1e6)


def raster_weights(poly, step=0.002):
    """The same weights from a fine raster: points every `step` degree inside the outline (shapely.contains_xy), each assigned to the grid box that holds it. Equal degree steps, not equal areas, so the two differ by
    the (tiny) change of the cell size with latitude over a basin."""
    import shapely
    x0, y0, x1, y1 = poly.bounds
    X, Y = np.meshgrid(np.arange(x0, x1, step) + step / 2, np.arange(y0, y1, step) + step / 2)
    m = shapely.contains_xy(poly, X, Y)
    ilon, ilat = np.rint((X[m] - LON_BOX[0]) / 0.25).astype(int), np.rint((Y[m] - LAT_BOX[0]) / 0.25).astype(int)
    key, n = np.unique(ilat * 1000 + ilon, return_counts=True)
    return {(int(k // 1000), int(k % 1000)): float(c / n.sum()) for k, c in zip(key, n)}, int(m.sum())


# ---- the stored forecasts ------------------------------------------------------------------------------------------------------------------------
def read_cells(cells_all):
    """The 3-hourly rain (mm per window) of the needed grid boxes for every stored run: (init dates as a DatetimeIndex, array [run, 26 windows, box])."""
    il = np.array([c[0] for c in cells_all])
    jl = np.array([c[1] for c in cells_all])
    inits, rain = [], []
    for y in range(FULL[0].year, FULL[1].year + 1):
        for m in range(1, 13):
            first = f"{y}-{m:02d}-01"
            if pd.Timestamp(first) > pd.Timestamp(FULL[1]) or pd.Timestamp(first) + pd.offsets.MonthEnd(0) < pd.Timestamp(FULL[0]):
                continue
            init, flag, apcp, lat, lon, steps = load_gefs(first, f"{y}-{m:02d}-31")
            assert (flag == 0).all(), f"{first[:7]}: a stored run has flag {sorted(set(flag.tolist()))}: fetch_gefs.py --check"
            assert np.array_equal(steps, STEPS) and np.array_equal(lat, LAT_BOX) and np.array_equal(lon, LON_BOX), "the grid or the windows of the store are not those assumed"
            v = apcp[:, :, il, jl]
            assert (v != 65535).all(), f"{first[:7]}: no-data values in the boxes of the basins"
            inits.append(pd.to_datetime(init.astype(str), format="%Y%m%d")), rain.append(v.astype(np.float64) / 10.0)
    inits = pd.DatetimeIndex(np.concatenate([i.values for i in inits]))
    assert inits.is_monotonic_increasing and inits.is_unique and inits[0] == pd.Timestamp(FULL[0]) and inits[-1] == pd.Timestamp(FULL[1]) and len(inits) == (pd.Timestamp(FULL[1]) - pd.Timestamp(FULL[0])).days + 1, "the stored runs are not one run per day"
    return inits, np.concatenate(rain)


def day_weights(issue_days, zone):
    """For every issue day t and k = 1, 2, 3: the weight of each 3-hourly window of the 00 UTC run of the UTC date t+1 in the local day t+k (the share of the window inside the day: 0, 1/3, 2/3 or 1), the first and last
    forecast hour of the day, and the age of the run at the issue time. Hours are counted from the run's start (00 UTC of the date t+1), from the time-zone rules (zoneinfo).

    The window j (0 to 25) covers the forecast hours (3j, 3j+3]; local day t+k covers the hours (lo, hi] with lo = local midnight that starts the day, hi = the one that ends it. Returns dict(R, lead_h, lo, hi, w) with
    lo, hi, w indexed by k."""
    R = to_s(issue_days + pd.Timedelta(days=1))                                    # 00 UTC of the UTC date t+1: the run
    issue = local_midnights_utc_s(issue_days + pd.Timedelta(days=1), zone)           # local midnight at the end of day t, as a UTC time: the issue time
    lead = (issue - R) / HOUR_S
    out = dict(R=R, issue=issue, lead_h=lead, lo={}, hi={}, w={}, w_absolute_route={})
    j = np.arange(NSTEP)
    for k in KS:
        a = local_midnights_utc_s(issue_days + pd.Timedelta(days=k), zone)           # start of local day t+k (UTC)
        b = local_midnights_utc_s(issue_days + pd.Timedelta(days=k + 1), zone)       # end of local day t+k (UTC)
        lo, hi = (a - R) / HOUR_S, (b - R) / HOUR_S
        out["lo"][k], out["hi"][k] = lo, hi
        out["w"][k] = np.clip(np.minimum(3 * j[None, :] + 3, hi[:, None]) - np.maximum(3 * j[None, :], lo[:, None]), 0, 3) / 3
        ws, we = R[:, None] + HOUR_S * 3 * j[None, :], R[:, None] + HOUR_S * (3 * j[None, :] + 3)               # the second route: absolute times (epoch seconds), no hours after the run
        out["w_absolute_route"][k] = np.clip(np.minimum(we, b[:, None]) - np.maximum(ws, a[:, None]), 0, None) / (3 * HOUR_S)
    return out


def prove_clock(dw, zone, name):
    """The clock of the forecast columns, for every issue day of one time zone: asserts and the numbers that go into the record."""
    lead = dw["lead_h"]
    assert set(np.unique(lead)) <= {4.0, 5.0, 6.0}, f"{name}: the run is not 4, 5 or 6 hours old at the issue time"
    assert (dw["R"] <= dw["issue"]).all(), f"{name}: the run starts after the issue time"
    assert ((dw["R"] + 24 * HOUR_S) > dw["issue"]).all(), f"{name}: a fresher 00 UTC run exists at the issue time (the one of the date t+2 would have to start after it)"
    rec = dict(age_of_the_run_at_the_issue_time_h={str(int(v)): int((lead == v).sum()) for v in np.unique(lead)}, per_lead={})
    for k in KS:
        lo, hi, w, w2 = dw["lo"][k], dw["hi"][k], dw["w"][k], dw["w_absolute_route"][k]
        assert (lo >= lead).all() and (lo > 0).all(), f"{name}, k={k}: a forecast hour of day t+{k} lies before the issue time"
        if k == 1:
            assert (lo == lead).all(), f"{name}: day t+1 does not start at the issue time"
        else:
            assert (lo == dw["hi"][k - 1]).all(), f"{name}: day t+{k} does not follow day t+{k - 1} hour by hour"
        assert (hi <= 78).all() and (hi > lo).all(), f"{name}, k={k}: day t+{k} reaches beyond forecast hour 78 or is empty"
        n_h = hi - lo
        assert set(np.unique(n_h)) <= {23.0, 24.0, 25.0}, f"{name}, k={k}: a local day has other than 23, 24 or 25 hours"
        assert np.allclose(w.sum(1), n_h / 3, atol=1e-12), f"{name}, k={k}: the weights of a day do not add up to its length"
        used = w > 0
        j = np.arange(NSTEP)[None, :]
        assert ((3 * j + 3 > lo[:, None]) | ~used).all() and ((3 * j < hi[:, None]) | ~used).all(), f"{name}, k={k}: a window with weight lies outside the day"
        first_hour_used = np.where(used, np.maximum(3 * j, lo[:, None]), np.inf).min(1)
        last_hour_used = np.where(used, np.minimum(3 * j + 3, hi[:, None]), -np.inf).max(1)
        assert (first_hour_used >= lead).all() and (last_hour_used <= 78).all(), f"{name}, k={k}: the part of a window that is used lies before the issue time or after hour 78"
        assert np.array_equal(w, w2) or np.abs(w - w2).max() < 1e-12, f"{name}, k={k}: the two routes (forecast hours / absolute times) give different weights (largest {np.abs(w - w2).max():.2e})"
        frac = used & (w < 1 - 1e-12)
        rec["per_lead"][f"k{k}"] = dict(first_forecast_hour_used_min=float(first_hour_used.min()), first_forecast_hour_used_max=float(first_hour_used.max()), last_forecast_hour_used_min=float(last_hour_used.min()),
                                       last_forecast_hour_used_max=float(last_hour_used.max()), hours_per_day={str(int(v)): int((n_h == v).sum()) for v in np.unique(n_h)}, largest_difference_of_the_two_routes=float(np.abs(w - w2).max()),
                                       windows_split_per_day=sorted({int(v) for v in frac.sum(1)}), partial_weights_seen=sorted({round(float(x), 6) for x in np.unique(w[frac])}))
    rec["asserted"] = ["run init (00 UTC of the UTC date t+1) <= issue time (local midnight at the end of day t)", "run init + 24 h > issue time (no fresher 00 UTC run exists)",
                       "every forecast hour of days t+1..t+3 is after the issue time (hour > 0 of the run) and inside the day's UTC span", "day t+1 starts exactly at the issue time, day t+k+1 starts where day t+k ends",
                       "no forecast hour beyond 78 (the last stored window)", "days have 23, 24 or 25 hours, and the window weights of a day add up to hours / 3",
                       "the weights from forecast hours equal the weights from absolute (epoch-second) times"]
    return rec


def window_means(rain, weights):
    """The basin-mean rain of every window of every run: the boxes' rain times their weights, summed over the boxes (rain [run, window, box]; elementwise product and a sum per row, so that a row's value does not depend on the other rows)."""
    return (rain * weights).sum(axis=-1)


def day_totals(win_mean, pos, dw):
    """Forecast rain of the local days t+1..t+3 (mm): the weights of each window times the basin-mean rain of the window of the run `pos` (a row index per issue day)."""
    return np.stack([(dw["w"][k] * win_mean[pos]).sum(1) for k in KS], 1)


# ---- the AORC rain of the same local days (the code of 16_aorc_rain_to_midnight.py, copied: the series is built there for the days of the engine's record, here for a few days more) -----------------------
def wall_clock(t_utc, zone):
    """UTC times (no time zone attached) -> the time on the wall clock of the zone, summer time included (no time zone attached)."""
    return t_utc.tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)


def rain_by_wall_clock_day(stamp, x, zone):
    """Every hourly value goes to the local calendar day, by the wall clock, in which its hour ENDS (the stamp is the end of the hour). Returns the rain of every day and its number of hours."""
    by_day = pd.Series(x, index=(wall_clock(stamp, zone) - pd.Timedelta(minutes=1)).normalize()).groupby(level=0)
    return by_day.sum(), by_day.size()


def day_positions(start, days, zone):
    """The second route (as in script 16): the UTC start and end of every wall-clock day from the time-zone rules, the positions of the hours between them, and the proofs that every summed hour lies inside its day."""
    z = ZoneInfo(zone)
    a = days.tz_localize(z).tz_convert("UTC").tz_localize(None)
    b = (days + pd.Timedelta(days=1)).tz_localize(z).tz_convert("UTC").tz_localize(None)
    assert ((a - start[0]) % HOUR == pd.Timedelta(0)).all() and ((b - start[0]) % HOUR == pd.Timedelta(0)).all(), "a day does not start on a whole hour"
    i0, i1 = np.asarray((a - start[0]) // HOUR), np.asarray((b - start[0]) // HOUR)
    assert i0.min() >= 0 and i1.max() < len(start), "the hourly series does not cover every day of the period"
    assert (i1[:-1] == i0[1:]).all() and set(np.unique(i1 - i0)) <= {23, 24, 25}, "the days do not follow each other hour by hour, or a day has other than 23, 24 or 25 hours"
    assert (start[i0] == a).all() and (start[i1 - 1] + HOUR == b).all(), "the first hour of a day does not start at its 00:00, or the last does not end at its 24:00"
    day = np.repeat(np.arange(len(days)), i1 - i0)
    hr = start[i0[0]:i1[-1]]
    sw, ew = wall_clock(hr, zone), wall_clock(hr + HOUR, zone)
    assert (sw >= days[day]).all() and (ew <= days[day] + pd.Timedelta(days=1)).all(), "an hour summed into a day starts before 00:00 or ends after 24:00 of that day by the wall clock"
    assert (hr >= a[day]).all() and (hr + HOUR <= b[day]).all(), "an hour summed into a day starts before its start or ends after its end in absolute time"
    assert ((ew - pd.Timedelta(minutes=1)).normalize() == days[day]).all(), "the two routes put an hour in different days"
    return i0, i1


def aorc_local_days(g, zone, days):
    """The AORC basin-mean rain of every wall-clock day of `days` (mm/day), by the main route, proved against the second route (hours between the day's UTC start and end summed)."""
    stamp, x = AO.load_series(g)
    assert len(stamp) == AO.N_HOURS and stamp[0] == AO.FIRST_STAMP and stamp[-1] == AO.LAST_STAMP and np.isfinite(x).all() and (x >= 0).all(), f"aorc_{g}.npz does not hold the span or has missing or negative rain"
    assert ((stamp[1:] - stamp[:-1]) == HOUR).all(), f"aorc_{g}.npz: the hourly time axis has gaps or repeats"
    i0, i1 = day_positions(stamp - HOUR, days, zone)
    by_day, size = rain_by_wall_clock_day(stamp, x, zone)
    s = by_day.reindex(days)
    assert s.notna().all() and (size.reindex(days).values == (i1 - i0)).all(), "the two routes count different numbers of hours in a day, or a day has no hours"
    assert np.allclose(np.add.reduceat(x, np.append(i0, i1[-1]))[:-1], s.values, rtol=0, atol=1e-6), "the two ways of summing the hours give different daily rain"
    return s


def subdaily_alignment(g, win_g, run_start_s, shifts=range(-6, 7)):
    """The hours of the two products against each other, which the day-level lag check cannot see: the basin-mean rain of every 3-hourly window of every run (window j of the run starting at R covers (R+3j h, R+3j+3 h]) against the AORC
    rain of the window (R+3j h+s, R+3j+3 h+s] for s = -6 .. +6 hours (the AORC value stamped T is the rain of the hour ending at T, so the window holds the stamps R+3j+s+1 h .. R+3j+s+3 h), the correlation pooled over all runs and
    windows 0 to 25. If the stamps or the run times were off by an hour or more, the best s would not be 0. (The model's own timing error widens the peak; it does not move it.)"""
    stamp, x = AO.load_series(g)
    cs = np.concatenate([[0.0], np.cumsum(x)])                                              # cs[i] = sum of x[0..i-1]
    t0 = to_s(stamp[:1])[0]
    start = run_start_s[:, None] + 3 * HOUR_S * np.arange(NSTEP)[None, :]
    out = {}
    for s in shifts:
        i0 = (start + s * HOUR_S - t0) // HOUR_S                                            # the index of the stamp that ends the hour before the window starts
        ok = (i0 >= 0) & (i0 + 4 <= len(x))
        obs = cs[np.where(ok, i0 + 4, 0)] - cs[np.where(ok, i0 + 1, 0)]
        out[f"{s:+d}"] = corr(win_g[ok], obs[ok])
    return out


# ---- skill ------------------------------------------------------------------------------------------------------------------------------------------
def corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else float("nan")


def ratio(f, o):
    return float(np.mean(f) / np.mean(o)) if len(o) and np.mean(o) > 0 else float("nan")


def brier(p, o):
    return float(np.mean((np.asarray(p, float) - np.asarray(o, float)) ** 2))


def bss(bs, obar):
    """Brier skill score against the climatological frequency of the scored sample: 1 - BS / (obar (1 - obar)); 1 = perfect, 0 = no better than always giving the frequency, negative = worse."""
    return 1 - bs / (obar * (1 - obar)) if 0 < obar < 1 else float("nan")


def build(verbose=True):
    t_start = time.time()
    say = (lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)) if verbose else (lambda *a: None)
    ids = AO.ids_of_the_study()
    DW = json.loads((RES / "day_windows.json").read_text())
    UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}
    assert all(DW["day_ends"][p]["per_basin"][g]["utc_offset_h"] == UTC_OFF[g] for p in ("nldas", "maurer", "imerg") for g in ids) \
        and all(DW["river_flow_day"]["per_gauge"][g]["utc_offset_standard_time_h"] == UTC_OFF[g] for g in ids), "the time zones in day_windows.json disagree"
    assert set(UTC_OFF.values()) <= set(ZONE), "a basin has a standard-time offset without a time zone here"
    try:
        assert all(pd.Timestamp("2000-01-15 12:00", tz=ZoneInfo(z)).utcoffset() == o * HOUR for o, z in ZONE.items()), "a time zone is not at its standard time in winter"
    except ZoneInfoNotFoundError as e:
        raise SystemExit(f"Python's zoneinfo cannot find the time-zone database ({e}); it is needed for the wall-clock day (pip install tzdata)")
    zone_of = {g: ZONE[UTC_OFF[g]] for g in ids}
    T = pd.date_range(FIRST_ISSUE, LAST_ISSUE)                                              # the issue days
    DAYS_EXT = pd.date_range(F.PERIOD[0], str((LAST_ISSUE + pd.Timedelta(days=3)).date()))   # the AORC local days: the engine's record and the three days after it (the target days of the last issue days)
    TG = {k: T + pd.Timedelta(days=k) for k in KS}                                           # the target day of lead k, per issue day

    # -- 1. weights
    import shapely
    from pyproj import Transformer
    to_ea = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True).transform         # the shapefile is NAD83 geographic (HCDN_nhru_final_671.prj); the forecast grid is read as longitude / latitude
    polys = AO.outlines(ids)
    W, W_REC = {}, {}
    for g in ids:
        cells, area = cell_weights(polys[g], to_ea)
        rw, n_pts = raster_weights(polys[g])
        w_sum = sum(c["weight"] for c in cells)
        assert abs(w_sum - 1) < 1e-12 and all(c["weight"] > 0 for c in cells), f"{g}: the weights do not add up to 1"
        assert set(rw) <= {(c["ilat"], c["ilon"]) for c in cells}, f"{g}: the fine raster has a point in a box the intersection route does not know"
        diff = max(abs(rw.get((c["ilat"], c["ilon"]), 0.0) - c["weight"]) for c in cells)
        assert diff < 0.01, f"{g}: weights from the exact intersection and from the fine raster differ by {diff:.4f}"
        centroid_cell = (int(np.argmin(np.abs(LAT_BOX - polys[g].centroid.y))), int(np.argmin(np.abs(LON_BOX - polys[g].centroid.x))))
        W[g] = cells
        W_REC[g] = dict(short=F.SHORT[g], time_zone=zone_of[g], utc_offset_h=UTC_OFF[g], outline_area_km2_equal_area=area, n_boxes=len(cells), weights=cells,
                        sum_of_overlap_areas_over_outline_area=sum(c["overlap_km2"] for c in cells) / area, largest_weight=max(c["weight"] for c in cells),
                        raster_check=dict(points_inside=n_pts, step_deg=0.002, largest_difference_of_a_weight=float(diff), weights_raster={f"{k[0]},{k[1]}": v for k, v in sorted(rw.items())}),
                        centroid=[float(polys[g].centroid.y), float(polys[g].centroid.x)], box_of_the_centroid=dict(ilat=centroid_cell[0], ilon=centroid_cell[1], lat=float(LAT_BOX[centroid_cell[0]]), lon=float(LON_BOX[centroid_cell[1]]),
                                                                                                                       weight=float(next((c["weight"] for c in cells if (c["ilat"], c["ilon"]) == centroid_cell), 0.0))))
    cells_all = sorted({(c["ilat"], c["ilon"]) for g in ids for c in W[g]})
    col = {c: n for n, c in enumerate(cells_all)}
    say(f"weights: {len(cells_all)} grid boxes for {len(ids)} basins ({min(len(W[g]) for g in ids)} to {max(len(W[g]) for g in ids)} boxes per basin)")

    # -- the stored runs
    inits, cells = read_cells(cells_all)
    say(f"read {len(inits)} runs ({inits[0].date()} .. {inits[-1].date()}), {cells.shape[2]} boxes, {cells.shape[1]} windows")
    R_dates = T + pd.Timedelta(days=1)
    assert R_dates.isin(inits).all(), "a run needed by an issue day is not stored"
    pos = inits.get_indexer(R_dates)
    assert (inits[pos] == R_dates).all()

    # -- 2. the day totals of every basin, the clock proof per time zone
    DWt = {z: day_weights(T, z) for z in sorted(set(zone_of.values()))}
    CLOCK = {z: prove_clock(DWt[z], z, z) for z in DWt}
    box_rain = {g: np.ascontiguousarray(cells[:, :, [col[(c["ilat"], c["ilon"])] for c in W[g]]]) for g in ids}              # [run, window, the basin's boxes], one C-contiguous array per basin (a sum's rounding depends on the memory layout)
    win = {g: window_means(box_rain[g], np.array([c["weight"] for c in W[g]])) for g in ids}                                  # basin-mean rain of each window of each run, mm
    RAW = {g: day_totals(win[g], pos, DWt[zone_of[g]]) for g in ids}                           # [issue day, k] mm
    # the second route for the totals: weights from absolute times
    for g in ids:
        d2 = DWt[zone_of[g]]
        alt = np.stack([(d2["w_absolute_route"][k] * win[g][pos]).sum(1) for k in KS], 1)
        assert np.abs(alt - RAW[g]).max() < 1e-9, f"{g}: the day totals from the two routes differ"
    say("clock proofs passed for", ", ".join(f"{z} ({sum(CLOCK[z]['age_of_the_run_at_the_issue_time_h'].values())} days)" for z in CLOCK))

    # -- the made-up-values test: the runs after the cut are replaced, no forecast issued on or before the cut day may change
    rng = np.random.default_rng(0)
    cut_rec = {}
    for g in ids:
        z = zone_of[g]
        h = DWt[z]["hi"][1] - DWt[z]["lo"][1]
        first_changes = [int(np.flatnonzero(h == v)[0]) for v in (23, 25) if (h == v).any()]
        cuts = sorted({*map(int, rng.integers(10, len(T) - 10, N_CUTS)), *[c for c in first_changes if 10 < c < len(T) - 10]})
        for c in cuts:
            alt = box_rain[g].copy()
            alt[pos[c] + 1:] = rng.uniform(0.0, 200.0, alt[pos[c] + 1:].shape)               # every run after the one issued on day c (the run of the date c+1) is made up
            win_alt = window_means(alt, np.array([q["weight"] for q in W[g]]))
            r2 = day_totals(win_alt, pos, DWt[z])
            assert np.array_equal(r2[:c + 1], RAW[g][:c + 1]), f"{g}: runs after the one of day {T[c].date()} changed a forecast issued on or before that day"
            assert not np.allclose(r2[c + 1], RAW[g][c + 1]), f"{g}: the test could not see a change of the next day"
        cut_rec[g] = dict(cut_days=[str(T[c].date()) for c in cuts], first_days_of_23_and_25_hours_included=[str(T[c].date()) for c in first_changes])
    say("made-up-values test passed")

    # -- 3. the AORC rain of the same local days
    A = {g: aorc_local_days(g, zone_of[g], DAYS_EXT) for g in ids}
    AOR = {g: np.stack([A[g].reindex(TG[k]).values for k in KS], 1) for g in ids}                 # [issue day, k] mm; the local day t+k
    assert all(np.isfinite(AOR[g]).all() for g in ids), "AORC rain is missing for a target day"
    say("AORC local-day series built and proved")
    SUB = {g: subdaily_alignment(g, win[g][pos], to_s(T + pd.Timedelta(days=1))) for g in ids}                     # the hours of the products against each other (the clock below the day)
    SUB_BEST = {g: max(SUB[g], key=lambda s: SUB[g][s]) for g in ids}
    SUB_MED = {s: float(np.median([SUB[g][s] for g in ids])) for s in SUB[ids[0]]}
    assert max(SUB_MED, key=SUB_MED.get) == "+0" and all(b in ("+0", "-1") for b in SUB_BEST.values()), f"the 3-hourly rain of the forecast and the AORC rain are not aligned to within one hour: {SUB_BEST}, median over basins {SUB_MED}"
    say(f"sub-daily alignment: best shift 0 in {sum(b == '+0' for b in SUB_BEST.values())} basins and -1 h in {sum(b == '-1' for b in SUB_BEST.values())} (median over basins best at 0)")

    # -- 4. the bias factor, the skill, the lag check
    in_val = {k: np.asarray((TG[k] >= VAL[0]) & (TG[k] <= VAL[1])) for k in KS}
    in_test = {k: np.asarray((TG[k] >= TEST[0]) & (TG[k] <= TEST[1])) for k in KS}
    in_all = {k: np.asarray((TG[k] >= FIRST_ISSUE) & (TG[k] <= LAST_ISSUE)) for k in KS}            # the target day inside 2000-01-01 .. 2014-09-30
    BIAS = {g: {k: float(AOR[g][in_val[k], k - 1].sum() / RAW[g][in_val[k], k - 1].sum()) for k in KS} for g in ids}
    BC = {g: np.stack([RAW[g][:, k - 1] * BIAS[g][k] for k in KS], 1) for g in ids}
    season_of = {k: np.asarray(TG[k].month) for k in KS}
    isotonic = {}
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        IsotonicRegression = None
    for thr_mm in THRESHOLDS_MM:                                                                         # the calibrated-probability map: forecast amount -> P(AORC >= threshold), fitted on the validation years, all basins pooled, per lead
        for k in KS:
            x = np.concatenate([RAW[g][in_val[k], k - 1] for g in ids])
            y = np.concatenate([(AOR[g][in_val[k], k - 1] >= thr_mm).astype(float) for g in ids])
            isotonic[thr_mm, k] = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(x, y) if IsotonicRegression else None
    SK, LAGS = {}, {}
    for g in ids:
        SK[g], LAGS[g] = {}, {}
        for k in KS:
            f, fb, o = RAW[g][:, k - 1], BC[g][:, k - 1], AOR[g][:, k - 1]
            rec = dict(n_days_2000_2014=int(in_all[k].sum()), correlation_all=corr(f[in_all[k]], o[in_all[k]]), mean_ratio_all=ratio(f[in_all[k]], o[in_all[k]]), mean_forecast_mm_all=float(f[in_all[k]].mean()), mean_aorc_mm_all=float(o[in_all[k]].mean()),
                       validation_years=dict(n_days=int(in_val[k].sum()), correlation=corr(f[in_val[k]], o[in_val[k]]), mean_ratio=ratio(f[in_val[k]], o[in_val[k]]), mean_ratio_bias_corrected=ratio(fb[in_val[k]], o[in_val[k]])),
                       test_years=dict(n_days=int(in_test[k].sum()), correlation=corr(f[in_test[k]], o[in_test[k]]), mean_ratio=ratio(f[in_test[k]], o[in_test[k]]), mean_ratio_bias_corrected=ratio(fb[in_test[k]], o[in_test[k]])),
                       by_season={s: dict(n_days=int((in_all[k] & np.isin(season_of[k], m)).sum()), correlation=corr(f[in_all[k] & np.isin(season_of[k], m)], o[in_all[k] & np.isin(season_of[k], m)]),
                                          mean_ratio=ratio(f[in_all[k] & np.isin(season_of[k], m)], o[in_all[k] & np.isin(season_of[k], m)]),
                                          mean_ratio_validation_years=ratio(f[in_val[k] & np.isin(season_of[k], m)], o[in_val[k] & np.isin(season_of[k], m)]),
                                          mean_ratio_bias_corrected_test_years=ratio(fb[in_test[k] & np.isin(season_of[k], m)], o[in_test[k] & np.isin(season_of[k], m)])) for s, m in SEASONS.items()})
            rec["brier_test_years"] = {}
            tt = in_test[k]
            for thr_mm in THRESHOLDS_MM:
                ob = (o[tt] >= thr_mm).astype(float)
                obar = float(ob.mean())
                row = dict(n_days=int(tt.sum()), event_frequency=obar, n_events=int(ob.sum()), brier_climatology=obar * (1 - obar))
                for name, fx in (("raw", f), ("bias_corrected", fb)):
                    yes_no = (fx[tt] >= thr_mm).astype(float)
                    row[f"yes_no_{name}"] = dict(brier=brier(yes_no, ob), brier_skill=bss(brier(yes_no, ob), obar), hits=int(((yes_no == 1) & (ob == 1)).sum()), false_alarms=int(((yes_no == 1) & (ob == 0)).sum()),
                                                 misses=int(((yes_no == 0) & (ob == 1)).sum()))
                if isotonic[thr_mm, k] is not None:
                    pcal = isotonic[thr_mm, k].predict(f[tt])
                    row["calibrated_probability"] = dict(brier=brier(pcal, ob), brier_skill=bss(brier(pcal, ob), obar))
                rec["brier_test_years"][f"{thr_mm:g}mm"] = row
            SK[g][f"l{k}"] = rec
            LAGS[g][f"l{k}"] = {f"{j:+d}": corr(f[in_all[k]], np.array([A[g].get(d, np.nan) for d in (TG[k][in_all[k]] + pd.Timedelta(days=j))])) for j in (-2, -1, 0, 1, 2)}
    say("skill computed")

    # -- the real-time margin of the run (results/data_access.json: the posting times of the operational control member, measured on 4 days)
    lat = json.loads(LATENCY_FILE.read_text())["gefs_reforecast_record"]["task5_issue_time"]["operational_gefs_posting_latency_gec00_0p25"]
    hours = sorted({int(h[1:]) for d in lat.values() for h in d})
    late_h = {h: max(d[f"f{h:03d}"] for d in lat.values()) for h in hours}                     # the slowest of the 4 sampled days, hours after the run's start
    margin = dict(source="results/data_access.json, gefs_reforecast_record.task5_issue_time.operational_gefs_posting_latency_gec00_0p25: the OPERATIONAL control member on AWS (noaa-gefs-pds), posting time of each forecast-hour file after the run's start, "
                         "4 days (2026-10-03 .. 06), the slowest of the four used; the reforecast archive has no posting times and a real-time service would read the operational files", latency_f078_h=late_h[78], by_zone_and_clock={}, per_basin={})
    for z, d in DWt.items():
        for lead_h in sorted(np.unique(d["lead_h"])):
            n = int((d["lead_h"] == lead_h).sum())
            need_hi = max(float(d["hi"][3][d["lead_h"] == lead_h].max()), 0)
            last_file = int(math.ceil(need_hi / 3) * 3)
            late_files = [h for h in hours if h <= last_file and late_h[h] > lead_h]
            margin["by_zone_and_clock"][f"{z}, run {int(lead_h)} h old at the issue time"] = dict(n_issue_days=n, last_file_needed=f"f{last_file:03d}", latency_h_of_that_file=late_h[last_file], margin_minutes_of_the_last_file=round(60 * (lead_h - late_h[last_file]), 1),
                                                                                                    files_needed_that_are_posted_after_the_issue_time=len(late_files), first_such_file=f"f{min(late_files):03d}" if late_files else None,
                                                                                                    margin_minutes_of_the_first_such_file=round(60 * (lead_h - late_h[min(late_files)]), 1) if late_files else None,
                                                                                                    worst_lateness_minutes=round(60 * max([late_h[h] - lead_h for h in late_files] + [0]), 1))
    for g in ids:
        d = DWt[zone_of[g]]
        late_share = float(np.mean([(late_h[int(math.ceil(d['hi'][3][i] / 3) * 3)] > d["lead_h"][i]) for i in range(len(T))]))
        margin["per_basin"][g] = dict(short=F.SHORT[g], share_of_issue_days_on_which_the_last_needed_file_is_posted_after_the_issue_time=late_share)

    # -- write the files
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for g in ids:
        df = pd.DataFrame({"init_utc_date": inits[pos].strftime("%Y%m%d").astype(int)}, index=pd.Index(T, name="issue_day"))
        for k in KS:
            df[f"fc_raw_{k}"] = RAW[g][:, k - 1]
        for k in KS:
            df[f"fc_bc_{k}"] = BC[g][:, k - 1]
        for k in KS:
            df[f"aorc_{k}"] = AOR[g][:, k - 1]
        df.to_csv(CSV(g), float_format="%.6f")
        files[str(CSV(g).relative_to(DATA.parent))] = hashlib.sha256(CSV(g).read_bytes()).hexdigest()
    med = lambda fn: {f"l{k}": float(np.nanmedian([fn(SK[g][f"l{k}"]) for g in ids])) for k in KS}
    best_lag = {g: {l: max(LAGS[g][l], key=lambda s: LAGS[g][l][s] if LAGS[g][l][s] == LAGS[g][l][s] else -9) for l in LAGS[g]} for g in ids}
    lag_ok = all(best_lag[g][l] == "+0" for g in ids for l in best_lag[g])
    pooled = {}
    for l, k in (("l1", 1), ("l2", 2), ("l3", 3)):
        pooled[l] = {}
        for thr_mm in THRESHOLDS_MM:
            key = f"{thr_mm:g}mm"
            n = sum(SK[g][l]["brier_test_years"][key]["n_days"] for g in ids)
            clim = sum(SK[g][l]["brier_test_years"][key]["brier_climatology"] * SK[g][l]["brier_test_years"][key]["n_days"] for g in ids) / n
            row = dict(n_basin_days=n, event_frequency=sum(SK[g][l]["brier_test_years"][key]["event_frequency"] * SK[g][l]["brier_test_years"][key]["n_days"] for g in ids) / n, mean_of_the_basins_brier_skill={})
            for name in ("yes_no_raw", "yes_no_bias_corrected", "calibrated_probability"):
                if name in SK[ids[0]][l]["brier_test_years"][key]:
                    bs = sum(SK[g][l]["brier_test_years"][key][name]["brier"] * SK[g][l]["brier_test_years"][key]["n_days"] for g in ids) / n
                    row[name] = dict(brier=bs, brier_skill_pooled=1 - bs / clim, brier_skill_median_over_basins=float(np.nanmedian([SK[g][l]["brier_test_years"][key][name]["brier_skill"] for g in ids])))
            pooled[l][key] = row
    summary = dict(
        correlation_all_median_over_basins=med(lambda r: r["correlation_all"]), mean_ratio_all_median_over_basins=med(lambda r: r["mean_ratio_all"]),
        correlation_validation_years_median=med(lambda r: r["validation_years"]["correlation"]), mean_ratio_validation_years_median=med(lambda r: r["validation_years"]["mean_ratio"]),
        correlation_test_years_median=med(lambda r: r["test_years"]["correlation"]), mean_ratio_test_years_median=med(lambda r: r["test_years"]["mean_ratio"]),
        mean_ratio_bias_corrected_test_years_median=med(lambda r: r["test_years"]["mean_ratio_bias_corrected"]),
        correlation_by_season_median={l: {s: float(np.nanmedian([SK[g][l]["by_season"][s]["correlation"] for g in ids])) for s in SEASONS} for l in ("l1", "l2", "l3")},
        mean_ratio_by_season_median={l: {s: float(np.nanmedian([SK[g][l]["by_season"][s]["mean_ratio"] for g in ids])) for s in SEASONS} for l in ("l1", "l2", "l3")},
        mean_ratio_by_season_validation_years_median={l: {s: float(np.nanmedian([SK[g][l]["by_season"][s]["mean_ratio_validation_years"] for g in ids])) for s in SEASONS} for l in ("l1", "l2", "l3")},
        brier_test_years_pooled=pooled, bias_factor_median_over_basins={f"l{k}": float(np.median([BIAS[g][k] for g in ids])) for k in KS},
        bias_factor_range_over_basins={f"l{k}": [float(min(BIAS[g][k] for g in ids)), float(max(BIAS[g][k] for g in ids))] for k in KS},
        lag_check=dict(best_shift_by_basin=best_lag, every_basin_and_lead_best_at_zero=bool(lag_ok),
                       median_correlation_by_shift={l: {s: float(np.nanmedian([LAGS[g][l][s] for g in ids])) for s in ("-2", "-1", "+0", "+1", "+2")} for l in ("l1", "l2", "l3")}))
    code_hash = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    R = dict(
        what="GEFSv12 reforecast, control member, 3-hourly rain on the 0.25 degree grid (data/gefs/c00) as basin means for the local days t+1, t+2, t+3 of every issue day t, with the AORC rain of the same days beside it (E2, step 2 of ROADMAP.md)",
        written=time.strftime("%Y-%m-%d %H:%M"), code=dict(file=f"code/{SCRIPT}", sha256=code_hash), runtime_s=round(time.time() - t_start, 1),
        issue_days=dict(first=str(T[0].date()), last=str(T[-1].date()), n=int(len(T)), what="local calendar days; day t is issued at its local midnight (24:00 by the wall clock of the gauge, summer time included)"),
        rule=dict(run="the 00 UTC run of the UTC date t+1 (init_date = t+1): 4 h old at the Eastern issue time in summer, 5 h in winter, 5 or 6 h at the Central one (the latest 00 UTC run that exists at the issue time; the run of the date t+2 starts after it)",
                  local_day="local day t+k = the wall-clock day of the gauge (zoneinfo America/New_York, America/Chicago for Leaf River, summer time included), forecast hours (lo, hi] of the run with lo = local midnight that starts the day, hi = the one that ends it, "
                            "counted from the run's start; the amounts of the 3-hourly windows inside the day are summed, a window that straddles a boundary is split in proportion to the time inside (1/3 or 2/3, uniform rain inside a window assumed)",
                  basin_mean="sum over the grid boxes of weight x rain, the weights being the shares of the outline's area in each 0.25 degree box (exact intersections in EPSG:5070)",
                  tables="the first row is 2000-01-01; k = 1, 2, 3 in the column names; mm per local day; fc_raw = the forecast as the model gives it, fc_bc = fc_raw x the bias factor of the basin and lead, aorc = the AORC rain of the same local day t+k"),
        grid=dict(lat=[float(LAT_BOX[0]), float(LAT_BOX[-1])], lon=[float(LON_BOX[0]), float(LON_BOX[-1])], box_size_deg=0.25, window_end_hours=[int(STEPS[0]), int(STEPS[-1])], n_windows=NSTEP, boxes_used=len(cells_all),
                  units="the store holds tenths of a mm per 3 h window (data/gefs/README.md); here mm"),
        basins=W_REC, clock=dict(per_time_zone=CLOCK, made_up_values_test=dict(what="every run after the run issued on day c (the runs of the dates c+1 and later) was replaced by random rain (0 to 200 mm per window): the forecasts issued on or before day c did not change (array_equal), "
                                                                                    "and the one issued on c+1 did", cut_days_per_basin=cut_rec),
                                 subdaily_alignment=dict(what="the basin-mean 3-hourly rain of every window of every run against the AORC rain of the same absolute window moved by s hours (s = -6 .. +6; AORC value stamped T = the hour ending at T), correlation pooled over "
                                                              "all runs and the 26 windows, per basin. Asserted: the best shift of the median over the basins is 0 and that of every basin 0 or -1 hour (the curve is flat between -1 and 0: the forecast's rain comes about "
                                                              "half an hour earlier than the AORC rain, well inside the forecast model's own timing error): the UTC stamps of the two products and the run times agree to the hour",
                                                         per_basin=SUB, best_shift_by_basin=SUB_BEST, median_over_basins=SUB_MED),
                                 independent_route="the weights of the windows were computed twice, from forecast hours counted from the run's start and from absolute epoch-second times of the windows and of the day's start and end: identical (largest difference "
                                                   f"{max(c['largest_difference_of_the_two_routes'] for z in CLOCK for c in CLOCK[z]['per_lead'].values()):.1e}); the day totals of every basin from both: largest difference < 1e-9 mm"),
        realtime_margin=margin,
        aorc=dict(source="data/aorc (code/fetch_aorc.py), hourly basin means, UTC, the stamp = the end of the hour; summed over the wall-clock day by the code of 16_aorc_rain_to_midnight.py (copied here; its proofs of the day boundaries are run here for "
                         f"{DAYS_EXT[0].date()} .. {DAYS_EXT[-1].date()}); the AORC rain of local day t+k is read from the day series by date", days=[str(DAYS_EXT[0].date()), str(DAYS_EXT[-1].date())]),
        bias=dict(rule="bias factor per basin and lead = (sum of the AORC rain of the local days t+k) / (sum of the forecast rain of the same days) over the issue days whose TARGET day t+k lies in the validation years 2000-10-01 .. 2005-09-30 (flood.SPLIT['val']); "
                       "fc_bc = fc_raw x factor, for every day of 2000-2014 (a factor fitted on the validation years only, applied unchanged to the test years); a single factor per basin and lead: no seasonal correction",
                  fitted_on_target_days=list(VAL), factors={g: {f"l{k}": BIAS[g][k] for k in KS} for g in ids}),
        skill=dict(how_to_read="the raw forecast against the AORC rain of the same local day; per basin and lead (l1 = day t+1 ...). correlation = Pearson over daily values; mean_ratio = mean forecast / mean AORC; windows: all = target days 2000-01-01 .. 2014-09-30, validation_years = 2000-10-01 .. "
                                  "2005-09-30 (the years the bias factor is fitted on), test_years = 2005-10-01 .. 2014-09-30; by_season by the month of the target day. Brier: days with at least 5 mm (or 10 mm) of AORC rain, test years, "
                                  "climatology = the sample's own event frequency (BS_clim = f(1-f)); yes_no_* = the control member as a deterministic yes/no forecast (forecast >= threshold), calibrated_probability = a monotone (isotonic) map from the raw forecast amount "
                                  "to the probability of the event, fitted on the validation years with the 12 basins pooled (per lead and threshold) and scored on the test years. A single member is not an ensemble: "
                                  "Guan et al. (2022) quote Brier skill 0.42 (12-36 h) and 0.33 (60-84 h) for the 5 mm days for the probabilities of the whole ensemble; the numbers here are of another kind (one member, a map fitted on the "
                                  "validation years, the sample's own frequency as climatology, these basins and years) and are not a check of that paper",
                   per_basin=SK, summary=summary, lag_check_correlation_by_shift=dict(how_to_read="correlation of the raw forecast of local day t+k (issue day t) with the AORC rain of local day t+k+shift, shift -2 .. +2, target days 2000-2014",
                                                                                      per_basin=LAGS)),
        files=files, caveats=[
            "Control member only (c00, 'low-res ctl'): no ensemble mean, no spread. A single deterministic member is noisier than the 5-member mean and its day totals place the rain to within the forecast's own timing error.",
            "The grid boxes are 0.25 degree (about 25 km, 630 km2) and the basins 113 to 1,928 km2: 1 to 9 boxes overlap a basin (weights recorded in basins.<id>.weights). The forecast is the model's box-mean rain, smoother and drier in its peaks than a basin mean of 1 km rain.",
            "The issue-time margin (realtime_margin): a real-time service would have the run 40 to 100 minutes before the issue time in winter and about 15 minutes too late for the last files (hours after about 39) in Eastern summer; this archive does not check "
            "whether the operational control member had the same posting times in 2000-2014, and the 00 UTC run was the only one of the reforecast. The rule 'the run of the UTC date t+1' is kept for every day and basin, as the work order says.",
            "The bias factor is one number per basin and lead fitted on the validation years (mean AORC / mean forecast). The forecast's bias has a season (by_season in the skill record: a wetter control member in spring and summer), which one factor does not remove; "
            "a seasonal factor was not used.",
            "AORC is the reference, not the truth: an analysis anchored on gauges and radar (hours before 2002 disaggregated daily totals). The forecast's skill 'against AORC' is skill against that analysis.",
            "Rain amounts of a local day are the sum of 3-hourly windows with straddling windows split in proportion (uniform rain inside a window assumed); for a=4, 5 or 6 the first window of day t+1 is split 1/3-2/3 or 2/3-1/3 or not at all, and the same at the end of each day.",
            "The Brier scores belong to a deterministic forecast (yes/no) or to a calibration of its amount; neither is the probability forecast of an ensemble."])
    nulls = [0]

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            if math.isfinite(float(o)):
                return float(o)
            nulls[0] += 1
            return None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    R = clean(R)
    R["n_values_written_as_null"] = nulls[0]
    RECORD.write_text(json.dumps(R, indent=1, allow_nan=False))
    return R


def ensure(verbose=True):
    """The record, read; made first when the tables or the record are missing."""
    ids = AO.ids_of_the_study()
    if not RECORD.exists() or not all(CSV(g).exists() for g in ids):
        return build(verbose)
    return json.loads(RECORD.read_text())


def load_columns(g):
    """The table of one basin: DataFrame indexed by issue day (2000-01-01 .. 2014-09-30) with init_utc_date, fc_raw_k, fc_bc_k, aorc_k (k = 1, 2, 3), mm."""
    return pd.read_csv(CSV(g), index_col="issue_day", parse_dates=True)


if __name__ == "__main__":
    R = build()
    s = R["skill"]["summary"]
    print("\nweights: boxes per basin " + ", ".join(f"{g}:{R['basins'][g]['n_boxes']}" for g in R["basins"]))
    print("clock: " + "; ".join(f"{z}: run age at the issue time {c['age_of_the_run_at_the_issue_time_h']} h (days)" for z, c in R["clock"]["per_time_zone"].items()))
    print("bias factors (median over basins) " + str({k: round(v, 3) for k, v in s["bias_factor_median_over_basins"].items()}))
    for l in ("l1", "l2", "l3"):
        print(f"{l}: corr all {s['correlation_all_median_over_basins'][l]:.3f} ratio {s['mean_ratio_all_median_over_basins'][l]:.3f} | val corr {s['correlation_validation_years_median'][l]:.3f} ratio {s['mean_ratio_validation_years_median'][l]:.3f} "
              f"| test corr {s['correlation_test_years_median'][l]:.3f} ratio {s['mean_ratio_test_years_median'][l]:.3f} (bias corrected {s['mean_ratio_bias_corrected_test_years_median'][l]:.3f})")
        for key, v in s["brier_test_years_pooled"][l].items():
            print(f"      {key}: event frequency {v['event_frequency']:.3f}; Brier skill " + ", ".join(f"{n} {v[n]['brier_skill_pooled']:.3f}" for n in ("yes_no_raw", "yes_no_bias_corrected", "calibrated_probability") if n in v))
    print("lag check: every basin and lead best at shift 0:", s["lag_check"]["every_basin_and_lead_best_at_zero"], s["lag_check"]["median_correlation_by_shift"])
    print("margin:", json.dumps(R["realtime_margin"]["by_zone_and_clock"], indent=1))
    print(f"written {RECORD} in {R['runtime_s']} s")
