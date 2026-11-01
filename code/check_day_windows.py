"""Which 24 hours does a daily value cover? Every day stamp of the project, measured against hourly data.

A forecast is issued at the end of day t, the day of the daily river flow. The daily rain tables do not all use that day: code/10_rain_timing.py
inferred from correlations between DAILY products that the rain table every model reads (Daymet) ends its day in the morning. This script measures
it against HOURLY rain: for each daily rain product (Daymet, NLDAS and Maurer from the CAMELS files, IMERG from data/satellite) and each basin it
sums the hourly rain of a reference over 24-hour windows shifted hour by hour and looks for the window that matches the daily values best. The
answer is given in local standard time as the hour at which the product's day ENDS. Two hourly references are used, both at the basin centroid:
ERA5 (Open-Meteo archive, 1980-2014) and MERRA-2 (NASA POWER, 2001-2014). The script also
 - calibrates the method on products whose day is known (the daily MERRA-2 and IMERG values, both served by UTC day, and a made-up value that ends at 12:00 UTC);
 - measures how much of the rain that falls AFTER local midnight is inside the NLDAS value of the issue day (it must be none);
 - checks the day of the daily river flow against 15-minute flow of the gauges (USGS, January-June 2012).
Hourly and 15-minute files are cached in data/hourly and fetched only if missing, so the script reruns offline. No model is trained.
Writes results/day_windows.json and figures/day_windows.png.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.patches import Rectangle

import flood as F
from camels import load_basin
from common import *

use_style()
t_start = time.time()
HD = DATA / "hourly"
HD.mkdir(exist_ok=True)
(HD / ".metadata_never_index").touch()                      # keeps the 100 MB cache out of the Spotlight index
D = F.load_data()
ALL = list(D)
ids = ["03439000", "02298608"] if F.QUICK else ALL          # smoke test: French Broad (the flow check) and Myakka (the basin that does not fit)
STEP = 3 if F.QUICK else 1
H = np.arange(-30, 31, STEP)                                # shift of the 24-hour window, hours after 00:00 UTC of day d
DROPS = (0.03, 0.06, 0.10)                                  # the centre of a curve is read where it has fallen this far below its top
MAIN_DROP = 0.06
MIN_DAYS = 1000                                             # fewest days for a correlation curve
MISFIT_H = 4.0                                              # a basin "does not fit" when its day ends this many hours from the median basin
REF = {"era5": dict(label="ERA5", period=("1980-10-01", "2014-09-30")), "merra2": dict(label="MERRA-2", period=("2001-01-05", "2014-09-30"))}
ERAS = {"all_years": None, "before_1996": ("1980-10-01", "1995-12-31"), "2001_2014": ("2001-01-01", "2014-09-30"), "test_years": F.SPLIT["test"]}
PRODUCTS = {"daymet": "rain table (Daymet)", "nldas": "NLDAS rain", "maurer": "Maurer rain", "imerg": "satellite rain (IMERG)", "merra2_daily": "MERRA-2 daily rain"}
KNOWN_UTC = ("imerg", "merra2_daily")                       # served by UTC day (NASA POWER, results/satellite_data.json): their window is known
FLOW_DAYS = ("2012-01-05", "2012-06-25")
notes = []
r3 = lambda x: round(float(x), 3)


# ---- files: fetch only what is missing ---------------------------------------------------------------------------------
def fetch(f, url, min_bytes=10000):
    """Download url to f unless f is already there; False if it could not be had (the caller decides what that means)."""
    if f.exists():
        return True
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200 and len(r.content) > min_bytes:
                f.write_bytes(r.content)
                print(f"fetched {f.name} ({len(r.content)} bytes)", flush=True)
                time.sleep(3)                               # be polite to the service
                return True
            print(f"{f.name}: HTTP {r.status_code}, {len(r.content)} bytes", flush=True)
            if r.status_code in (400, 404):
                return False
        except Exception as e:
            print(f"{f.name}: {repr(e)[:160]}", flush=True)
        time.sleep(15 * (attempt + 1))
    return False


SAT = json.loads((F.RES / "satellite_data.json").read_text())
CENT = {g: SAT["cells_per_basin"][g]["centroid_lon_lat"] for g in ALL}                              # basin centroids (code/fetch_satellite.py)
URL = {"era5": lambda lon, lat: ("https://archive-api.open-meteo.com/v1/archive?latitude=%.4f&longitude=%.4f&start_date=1980-09-25&end_date=2014-10-05"
                                 "&hourly=precipitation&timezone=UTC&models=era5" % (lat, lon)),
       "merra2": lambda lon, lat: ("https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=PRECTOTCORR&community=AG&longitude=%.4f&latitude=%.4f"
                                   "&start=20010101&end=20141005&format=JSON&time-standard=UTC" % (lon, lat))}
FILE = {"era5": lambda g: HD / f"era5_{g}.json", "merra2": lambda g: HD / f"merra_{g}.json"}
for src in REF:
    for g in ids:
        assert fetch(FILE[src](g), URL[src](*CENT[g])), f"{FILE[src](g).name} is missing and could not be fetched"
SITES = HD / "usgs_sites.rdb"                               # the gauges' time zones, from the USGS site file
assert fetch(SITES, "https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + ",".join(ALL) + "&siteOutput=expanded", 2000), "USGS site file missing"
st = pd.read_csv(SITES, sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
TZ_CODE = {"EST": 5, "EDT": 4, "CST": 6, "CDT": 5}          # hours behind UTC
TZ = {g: TZ_CODE[st.tz_cd[g]] for g in ALL}                 # local STANDARD time of each gauge (the day of the daily flow, see the flow check)
iv_file = lambda g: HD / f"usgs_iv_{g}.rdb"
HAVE_IV = [g for g in ids if fetch(iv_file(g), f"https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={g}&parameterCd=00060&startDT=2012-01-01&endDT=2012-06-30", 100000)]
retrieved = lambda f: time.strftime("%Y-%m-%d", time.localtime(f.stat().st_mtime))                  # the date on the cached file


# ---- data ----------------------------------------------------------------------------------------------------------------
def hourly(g, src):
    """(first hour, rain per hour in mm): a complete hourly UTC series; element i is the rain of the hour that STARTS at first hour + i."""
    j = json.loads(FILE[src](g).read_text())
    if src == "era5":                                       # Open-Meteo: the value at time T is the rain of the hour before T
        t, x = j["hourly"]["time"], np.array(j["hourly"]["precipitation"], float)
        t0, t1 = pd.Timestamp(t[0]) - pd.Timedelta(hours=1), pd.Timestamp(t[-1]) - pd.Timedelta(hours=1)
        meta = dict(grid_point_lat_lon=[j["latitude"], j["longitude"]])
    else:                                                   # NASA POWER hourly, UTC: key YYYYMMDDHH is the hour that starts at HH
        p = j["properties"]["parameter"]["PRECTOTCORR"]
        k, x = list(p), np.array(list(p.values()), float)
        x[x < -900] = np.nan
        t0, t1 = pd.to_datetime(k[0], format="%Y%m%d%H"), pd.to_datetime(k[-1], format="%Y%m%d%H")
        meta = dict(point_lon_lat=j["geometry"]["coordinates"][:2], api_version=j["header"]["api"]["version"], title=j["header"]["title"],
                    sources=j["header"]["sources"], time_standard=j["header"]["time_standard"], units=j["parameters"]["PRECTOTCORR"]["units"])
    assert (t1 - t0) / pd.Timedelta(hours=1) == len(x) - 1, f"{src} {g}: the hourly series has gaps in its time axis"
    return t0, x, dict(meta, first_hour=str(t0), last_hour=str(t1), n_hours=len(x), n_missing=int(np.isnan(x).sum()))


def daily_products(g):
    d = D[g]
    sat = pd.read_csv(DATA / "satellite" / f"{g}.csv", parse_dates=["date"]).set_index("date").reindex(d.index)
    camels = lambda s: load_basin(g, s).loc[F.PERIOD[0]:F.PERIOD[1], "prcp"].astype(float)
    return {"daymet": d.prcp, "nldas": camels("nldas"), "maurer": camels("maurer"), "imerg": sat["imerg"], "merra2_daily": sat["merra_prcp"]}


def day_pos(idx, t0):
    """Position of 00:00 UTC of each day in an hourly series that starts at t0."""
    return np.asarray((idx - t0) // pd.Timedelta(hours=1), dtype=np.int64)


def curve(daily, t0, R, a, b):
    """Correlation of the daily values with the reference rain of the 24 hours that start h hours after 00:00 UTC of the day, for every h."""
    s = daily.loc[a:b].dropna()
    if len(s) < MIN_DAYS:
        return None, len(s)
    pos, y, out = day_pos(s.index, t0), s.values, []
    for h in H:
        i = pos + h
        inside = (i >= 0) & (i < len(R))
        x = np.full(len(y), np.nan)
        x[inside] = R[i[inside]]
        ok = np.isfinite(x)
        out.append(np.corrcoef(y[ok], x[ok])[0, 1])
    return np.array(out), len(s)


span = lambda daily, a, b: [str(t.date()) for t in daily.loc[a:b].dropna().index[[0, -1]]]         # first and last day that has a value


def centre(c, drop):
    """Middle between the two shifts at which the curve has fallen `drop` below its top (steadier than the flat top itself)."""
    i = int(np.argmax(c))
    lvl, l, r = c[i] - drop, i, i
    while l > 0 and c[l] > lvl:
        l -= 1
    while r < len(c) - 1 and c[r] > lvl:
        r += 1
    if c[l] > lvl or c[r] > lvl:
        return float("nan")                                 # the curve does not fall that far inside the shifts tried
    xl, xr = H[l] + STEP * (lvl - c[l]) / (c[l + 1] - c[l]), H[r - 1] + STEP * (c[r - 1] - lvl) / (c[r - 1] - c[r])
    return float((xl + xr) / 2)


def clock(h):
    """Hours after the local midnight that starts day d -> 'HH:MM' (with the day when it is not day d)."""
    if not np.isfinite(h):
        return None
    day, m = divmod(int(round(h * 60)), 1440)
    if (day, m) == (1, 0):
        return "24:00"
    return f"{m // 60:02d}:{m % 60:02d}" + ("" if day == 0 else f" on day d{day:+d}")


def half_hour(h):
    """Same, rounded to the half hour: the method is not finer than that."""
    return clock(round(h * 2) / 2)


ends_local = lambda g, shift: 24 - TZ[g] + shift            # window [00 UTC + shift, +24 h) ends at this local standard hour of day d
spread = lambda v: dict(median=r3(np.nanmedian(v)), min=r3(np.nanmin(v)), max=r3(np.nanmax(v)))

# ---- 1. the window of every daily product, against each hourly reference ---------------------------------------------------
HR, PR, src_meta = {}, {g: daily_products(g) for g in ids}, {s: {} for s in REF}
for src in REF:
    for g in ids:
        t0, x, src_meta[src][g] = hourly(g, src)
        HR[src, g] = (t0, x, pd.Series(x).rolling(24).sum().shift(-23).values)                     # R[i] = rain of the 24 hours that start at hour i
windows, shift_of = {}, {}
for src, ref in REF.items():
    windows[src] = {}
    for p in PRODUCTS:
        windows[src][p] = {}
        for era, per in ERAS.items():
            a, b = per or ref["period"]
            a, b = max(a, ref["period"][0]), min(b, ref["period"][1])
            if a > b:
                continue
            rows, curves = {}, {}
            for g in ids:
                t0, _, R = HR[src, g]
                c, n = curve(PR[g][p], t0, R, a, b)
                if c is None:
                    continue
                i, top = int(np.argmax(c)), np.flatnonzero(c >= c.max() - 0.01)
                cen = {d: centre(c, d) for d in DROPS}
                curves[g] = c
                shift_of[src, p, era, g] = cen[MAIN_DROP]
                rows[g] = dict(short=F.SHORT[g], n_days=n, best_shift_h=int(H[i]), corr_best=r3(c[i]), corr_utc_day=r3(c[list(H).index(0)]),
                               flat_top_h=[int(H[top[0]]), int(H[top[-1]])], centre_shift_h=r3(cen[MAIN_DROP]),
                               centre_shift_h_by_drop={str(d): r3(v) for d, v in cen.items()},
                               day_ends_local_hour=r3(ends_local(g, cen[MAIN_DROP])), day_ends_local=clock(ends_local(g, cen[MAIN_DROP])),
                               day_ends_local_best_shift=clock(ends_local(g, H[i])))
            if not rows:
                continue
            e = np.array([rows[g]["day_ends_local_hour"] for g in rows])
            med = float(np.nanmedian(e))
            had = [span(PR[g][p], a, b) for g in rows]      # the days the product has: IMERG starts in 1998, Maurer ends in 2008
            windows[src][p][era] = dict(
                period=[min(d[0] for d in had), max(d[1] for d in had)], n_basins=len(rows), centre_shift_h=spread([rows[g]["centre_shift_h"] for g in rows]),
                best_shift_h=spread([rows[g]["best_shift_h"] for g in rows]), corr_best_median=r3(np.median([rows[g]["corr_best"] for g in rows])),
                day_ends_local_hour=spread(e), day_ends_local=dict(median=clock(med), earliest=clock(np.nanmin(e)), latest=clock(np.nanmax(e))),
                day_ends_local_best_shift_median=clock(float(np.median([ends_local(g, rows[g]["best_shift_h"]) for g in rows]))),
                basins_that_do_not_fit=[rows[g]["short"] for g in rows if not abs(rows[g]["day_ends_local_hour"] - med) <= MISFIT_H],
                per_basin=rows, median_curve=[r3(v) for v in np.median([curves[g] for g in rows], axis=0)],
                **({"curves": {g: [r3(v) for v in curves[g]] for g in rows}} if era == "all_years" else {}))
    print(f"day windows against {ref['label']} done ({time.time() - t_start:.0f} s)", flush=True)

# ---- 2. calibration: products whose day is known must come out at their known window ------------------------------------
cal = {}
for src in REF:
    for p in KNOWN_UTC:
        w = windows[src][p]["all_years"]
        cal[f"{p}_against_{src}_hourly"] = dict(
            known_window="the UTC day: shift 0", period=w["period"], best_shift_h=w["best_shift_h"], centre_shift_h=w["centre_shift_h"],
            corr_best_median=w["corr_best_median"], per_basin={g: dict(short=v["short"], best_shift_h=v["best_shift_h"], centre_shift_h=v["centre_shift_h"],
                                                                       corr_best=v["corr_best"]) for g, v in w["per_basin"].items()})
self_test = cal["merra2_daily_against_merra2_hourly"]
# direction of the shift: a daily value made here with a known window - the 24 hours that END at 12:00 UTC of day d - must be found 12 hours before the UTC day
for src, ref in REF.items():
    rows = {}
    for g in ids:
        t0, x, R = HR[src, g]
        made = pd.Series(x, index=pd.date_range(t0, periods=len(x), freq="h")).shift(12, freq="h").resample("D").sum(min_count=24)     # summed by time stamp, not by the positions curve() uses
        c, _ = curve(made, t0, R, *ref["period"])
        i = int(np.argmax(c))
        rows[g] = dict(short=F.SHORT[g], best_shift_h=int(H[i]), centre_shift_h=r3(centre(c, MAIN_DROP)), corr_best=r3(c[i]), day_ends_local_best_shift=clock(ends_local(g, H[i])))
    assert all(v["best_shift_h"] == -12 for v in rows.values()), f"direction test against {src}: a window that ends at 12:00 UTC of day d was not found at shift -12"
    cal[f"made_up_value_against_{src}_hourly"] = dict(
        known_window="made here from the hourly rain of the 24 hours that END at 12:00 UTC of day d: shift -12 (07:00 Eastern, 06:00 Central standard time of day d)",
        period=list(ref["period"]), best_shift_h=spread([v["best_shift_h"] for v in rows.values()]), centre_shift_h=spread([v["centre_shift_h"] for v in rows.values()]),
        corr_best_median=r3(np.median([v["corr_best"] for v in rows.values()])), per_basin=rows)

# ---- the headline: when does each product's day end? four readings per basin ----------------------------------------------
yrs = lambda per: f"{per[0][:4]}-{per[1][:4]}"
READINGS = {"era5": f"against ERA5 hourly rain {yrs(REF['era5']['period'])}, as measured", "merra2": f"against MERRA-2 hourly rain {yrs(REF['merra2']['period'])}, as measured",
            "era5_set_to_imerg": "against ERA5, shifted so that IMERG falls on its documented UTC day",
            "merra2_set_to_imerg": "against MERRA-2, shifted so that IMERG falls on its documented UTC day"}
day_ends = {}
for p in ("daymet", "nldas", "maurer"):
    rows = {}
    for g in ids:
        raw = {src: shift_of.get((src, p, "all_years", g), np.nan) for src in REF}
        v = {src: ends_local(g, raw[src]) for src in REF}
        v.update({f"{src}_set_to_imerg": ends_local(g, raw[src] - shift_of[src, "imerg", "all_years", g]) for src in REF})
        rows[g] = dict(short=F.SHORT[g], utc_offset_h=-TZ[g], **{k: r3(x) for k, x in v.items()}, middle=r3(np.nanmean(list(v.values()))))
    mid = np.array([rows[g]["middle"] for g in ids])
    med = float(np.nanmedian(mid))
    by_reading = {k: float(np.nanmedian([rows[g][k] for g in ids])) for k in READINGS}
    day_ends[p] = dict(label=PRODUCTS[p], day_ends_local_hour=r3(med), day_ends_local=clock(med), day_ends_local_rounded=half_hour(med),
                       basin_range_local=[clock(np.nanmin(mid)), clock(np.nanmax(mid))], basin_range_local_hour=[r3(np.nanmin(mid)), r3(np.nanmax(mid))],
                       median_by_reading={k: dict(local_hour=r3(x), local=clock(x)) for k, x in by_reading.items()},
                       range_over_readings_local=[clock(min(by_reading.values())), clock(max(by_reading.values()))],
                       basins_that_do_not_fit=[F.SHORT[g] for g in ids if not abs(rows[g]["middle"] - med) <= MISFIT_H], per_basin=rows)
doc_end = np.array([24 - TZ[g] for g in ids], float)        # IMERG: the UTC day ends at 19:00 Eastern, 18:00 Central standard time
found = {g: {src: ends_local(g, shift_of[src, "imerg", "all_years", g]) for src in REF} for g in ids}
found_med = {src: float(np.median([found[g][src] for g in ids])) for src in REF}
day_ends["imerg"] = dict(label=PRODUCTS["imerg"], documented="the UTC day (NASA POWER; results/satellite_data.json)",
                         documented_day_ends_local_hour=r3(np.median(doc_end)), documented_day_ends_local=clock(np.median(doc_end)),
                         found_by_this_check_local={src: clock(found_med[src]) for src in REF},
                         hours_late_in_this_check={src: r3(np.median([shift_of[src, "imerg", "all_years", g] for g in ids])) for src in REF},
                         per_basin={g: dict(short=F.SHORT[g], utc_offset_h=-TZ[g], documented=24 - TZ[g], **{src: r3(found[g][src]) for src in REF}) for g in ids})
nldas_latest = max(day_ends["nldas"]["per_basin"][g][k] for g in ids for k in READINGS)             # the latest single reading of any basin

# ---- 3. is rain from after local midnight inside the value of the issue day? ----------------------------------------------
BINS = ["12h_before_the_utc_day", "utc_day_first_half", "utc_day_second_half", "evening_until_local_midnight", "after_local_midnight_until_12_utc", "the_12h_after_that"]
FC_RUNS = [("merra2", "test_years", F.SPLIT["test"]), ("era5", "test_years", F.SPLIT["test"]), ("era5", "before_1996", ERAS["before_1996"]),
           ("era5", "1996_2000", ("1996-01-01", "2000-09-30"))]
FC_PROD = ("nldas", "imerg", "merra2_daily", "daymet", "maurer")


def rain_weights(daily, t0, x, a, b, tz):
    """Daily value regressed on the reference rain of six stretches around the day: weight 1 = that rain is fully inside the daily value."""
    s = daily.loc[a:b].dropna()
    if len(s) < MIN_DAYS:
        return None
    edges = [-12, 0, 12, 24, 24 + tz, 36, 48]               # hours after 00:00 UTC of day d; 24 + tz is local midnight, when the forecast is issued
    c, pos = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))]), day_pos(s.index, t0)             # c[i] = rain before hour i
    ok = (pos + edges[0] >= 0) & (pos + edges[-1] < len(c))
    X = np.column_stack([c[pos[ok] + e1] - c[pos[ok] + e0] for e0, e1 in zip(edges[:-1], edges[1:])])
    return np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), s.values[ok], rcond=None)[0][:-1]


future = {}
for src, era, (a, b) in FC_RUNS:
    out = {}
    for p in FC_PROD:
        W = {g: rain_weights(PR[g][p], HR[src, g][0], HR[src, g][1], a, b, TZ[g]) for g in ids}
        W = {g: w for g, w in W.items() if w is not None}
        if W:
            had = [span(PR[g][p], a, b) for g in W]
            out[p] = dict(n_basins=len(W), period=[min(d[0] for d in had), max(d[1] for d in had)], median=dict(zip(BINS, (r3(v) for v in np.median(list(W.values()), axis=0)))),
                          after_local_midnight=dict(spread([w[4] for w in W.values()]), n_basins_above_0p1=int(sum(w[4] > 0.1 for w in W.values()))),
                          per_basin={g: dict(short=F.SHORT[g], **dict(zip(BINS, (r3(v) for v in w)))) for g, w in W.items()})
    future[f"{era}_against_{src}_hourly"] = dict(reference=REF[src]["label"], period=[a, b], products=out)
fut = lambda run, p: round(future[run]["products"][p]["after_local_midnight"]["median"], 2) + 0 if p in future[run]["products"] else float("nan")     # + 0: no '-0.00'

# ---- 4. the day of the daily river flow ------------------------------------------------------------------------------------
FLOW_H = list(range(-12, 13))
iv_raw = {g: pd.read_csv(iv_file(g), sep="\t", comment="#", dtype=str).iloc[1:] for g in HAVE_IV}
change = pd.Series([pd.Timestamp(t.datetime[t.tz_cd.isin(["EDT", "CDT"])].iloc[0]).normalize() for t in iv_raw.values()]).mode()[0]     # first day on summer time
day = pd.Timedelta(days=1)
PARTS = {"whole_period": FLOW_DAYS, "standard_time_in_force": (FLOW_DAYS[0], str((change - 2 * day).date())),
         "daylight_saving_time_in_force": (str((change + 2 * day).date()), FLOW_DAYS[1])}


def flow_day(g):
    """Error of 15-minute flow averaged over 24 hours that start h hours after 00:00 UTC, against the daily flow in the CAMELS file."""
    t = iv_raw[g]
    col = [c for c in t.columns if c.endswith("_00060")][0]
    utc = pd.to_datetime(t.datetime) + pd.to_timedelta(t.tz_cd.map(TZ_CODE), unit="h")
    q = pd.Series(pd.to_numeric(t[col], errors="coerce").values, index=utc).dropna()
    q = q.resample("15min").mean().interpolate(limit=8)     # gaps up to two hours are bridged
    m24 = q.rolling(96, min_periods=90).mean().shift(-95)   # mean of the 24 hours that start at each quarter hour
    d = load_basin(g)
    cfs_all = d.q_mm / (0.0283168 * 86400 / (d.attrs["area_km2"] * 1e6) * 1000)
    out = dict(short=F.SHORT[g], time_zones_in_file=t.tz_cd.value_counts().to_dict(), utc_offset_standard_time_h=-TZ[g], retrieved=retrieved(iv_file(g)))
    for part, (a, b) in PARTS.items():
        cfs = cfs_all.loc[a:b].dropna()
        M = np.column_stack([m24.reindex(cfs.index + pd.Timedelta(hours=h)).values for h in FLOW_H])
        ok = np.isfinite(M).all(axis=1)                     # every window is scored on the same days: those on which all of them have a value
        if ok.sum() < 30:
            continue
        o = cfs.values[ok]
        rows = [dict(shift_h=h, window_ends_local_standard_time=clock(ends_local(g, h)), rmse_cfs=r3(np.sqrt(np.mean((M[ok, j] - o) ** 2))),
                     mean_relative_difference_percent=r3(100 * np.mean(np.abs(M[ok, j] - o) / o)), n_days=int(ok.sum())) for j, h in enumerate(FLOW_H)]
        at = lambda h: next(r for r in rows if r["shift_h"] == h)
        best, best_rmse = min(rows, key=lambda r: r["mean_relative_difference_percent"]), min(rows, key=lambda r: r["rmse_cfs"])
        out[part] = dict(days=[a, b], n_days=rows[0]["n_days"], mean_flow_cfs=r3(cfs.mean()), best_shift_h=best["shift_h"], best_shift_h_by_rmse=best_rmse["shift_h"],
                         best_window_ends_local_standard_time=best["window_ends_local_standard_time"], best_window_ends_local_hour=ends_local(g, best["shift_h"]),
                         midnight_to_midnight_standard_time=at(TZ[g]), midnight_to_midnight_summer_time=at(TZ[g] - 1), utc_day=at(0),
                         **({"by_shift": rows} if part == "whole_period" else {}))
    return out


flow = {g: flow_day(g) for g in HAVE_IV}
flow = {g: v for g, v in flow.items() if all(p in v for p in PARTS)}
no_flow = [F.SHORT[g] for g in ids if g not in flow]
WIN = ("midnight_to_midnight_standard_time", "midnight_to_midnight_summer_time", "utc_day")
flow_summary = dict(n_gauges=len(flow), first_day_on_summer_time=str(change.date()), gauges_without_15_minute_flow=no_flow, parts={})
for part in PARTS:
    best = {g: v[part]["best_shift_h"] for g, v in flow.items()}
    flow_summary["parts"][part] = dict(
        days=list(PARTS[part]), n_gauges_best_midnight_standard_time=int(sum(best[g] == TZ[g] for g in flow)),
        n_gauges_best_midnight_summer_time=int(sum(best[g] == TZ[g] - 1 for g in flow)),
        gauges_best_midnight_standard_time=[F.SHORT[g] for g in flow if best[g] == TZ[g]],
        n_gauges_best_shift_same_by_rmse=int(sum(v[part]["best_shift_h"] == v[part]["best_shift_h_by_rmse"] for v in flow.values())),
        median_relative_difference_percent={w: r3(np.median([v[part][w]["mean_relative_difference_percent"] for v in flow.values()])) for w in WIN})
print(f"all checks done ({time.time() - t_start:.0f} s)", flush=True)

# ---- notes: what was found, and what the method cannot say ----------------------------------------------------------------
W = lambda src, p, era="all_years": windows[src][p][era]
move = lambda p: np.array([W("era5", p, "2001_2014")["per_basin"][g]["centre_shift_h"] - W("era5", p, "before_1996")["per_basin"][g]["centre_shift_h"] for g in ids])
era_change = dict(
    how_to_read="Change of the day end from 1980-1995 to 2001-2014 against ERA5, in hours, basin by basin (negative: the day ends earlier in the later years). The window of the "
                "daily MERRA-2 value is the UTC day in every year by construction: its change is a drift between the clocks of the two reanalyses, not a move of a window.",
    products={p: dict(spread(move(p)), n_basins_earlier=int((move(p) < 0).sum()), per_basin={g: r3(v) for g, v in zip(ids, move(p))}) for p in ("daymet", "nldas", "merra2_daily")})
ec, made_up = era_change["products"], {src: cal[f"made_up_value_against_{src}_hourly"] for src in REF}
gp = np.array([src_meta["era5"][g]["grid_point_lat_lon"] for g in ids])
off = np.abs(gp - np.array([CENT[g][::-1] for g in ids])).max()
era5_grid = (f"the grid points returned lie on a 0.25 degree grid, up to {off:.2f} degrees from the centroid" if np.allclose(gp * 4, np.round(gp * 4))
             else f"the grid points returned lie up to {off:.2f} degrees from the centroid")
de, fb = day_ends, flow.get("03439000")
rd = lambda p: ", ".join(f"{v['local']} ({READINGS[k].split(',')[0].replace('against ', '')}{', set to IMERG' if 'imerg' in k else ''})" for k, v in de[p]["median_by_reading"].items())
fit = lambda p: f"Basins more than {MISFIT_H:.0f} hours from the median: {', '.join(de[p]['basins_that_do_not_fit']) or 'none'}."
notes += [
    f"Method: a daily value of day d is correlated with the hourly reference rain summed over the 24 hours that start h hours after 00:00 UTC of day d, for h from "
    f"{H[0]} to +{H[-1]} in steps of {STEP} h. The curve has a broad top, so the window is placed at its centre: the middle between the two shifts at which the "
    f"correlation has fallen {MAIN_DROP} below its best value (centre_shift_h; the best single shift and the readings for drops {DROPS[0]} and {DROPS[2]} are "
    f"listed too). The day then ends at 24:00 UTC + h, given in local standard time: UTC-5 for the Eastern basins and UTC-6 for Leaf River, MS "
    f"(time zones from the USGS site file). 'Day d' is the calendar date written in the product's table.",
    f"Self-test of the code (calibration 1): the daily MERRA-2 value served by UTC day against the hourly MERRA-2 rain it is made of. Best shift "
    f"{self_test['best_shift_h']['median']:+.0f} h (basins {self_test['best_shift_h']['min']:+.0f} to {self_test['best_shift_h']['max']:+.0f}), correlation "
    f"{self_test['corr_best_median']:.3f}: the hour stamps and the zero of the shift are right. The direction of the shift is tested with a made-up daily value, "
    f"the hourly rain of the 24 hours that END at 12:00 UTC of day d (07:00 Eastern standard time, the morning of day d): it is found at shift "
    f"{made_up['era5']['best_shift_h']['median']:+.0f} h against ERA5 and {made_up['merra2']['best_shift_h']['median']:+.0f} h against MERRA-2 in every basin "
    f"(calibration.made_up_value_against_<reference>_hourly; the script stops if it is not), so a day that ends before the UTC day comes out as a negative shift "
    f"and an earlier local hour. The centre-of-curve reading of the self-test curves is "
    f"{self_test['centre_shift_h']['median']:+.1f} h (basins {self_test['centre_shift_h']['min']:+.1f} to {self_test['centre_shift_h']['max']:+.1f}) where 0 is true, "
    f"and of the made-up morning value {made_up['merra2']['centre_shift_h']['median']:+.1f} h (basins {made_up['merra2']['centre_shift_h']['min']:+.1f} to "
    f"{made_up['merra2']['centre_shift_h']['max']:+.1f}) where -12 is true: where rain falls mostly at one time of day (afternoon storms in Florida) the curve is "
    f"lopsided and its centre is pushed by up to that much. That is an error of the method itself, and it need not be the same for a product with another day window.",
    f"Calibration 2: IMERG is served by UTC day, so its day ends at {de['imerg']['documented_day_ends_local']} local standard time (18:00 in Leaf River, MS). "
    f"This check puts the end at {de['imerg']['found_by_this_check_local']['era5']} against ERA5 and {de['imerg']['found_by_this_check_local']['merra2']} against "
    f"MERRA-2: {de['imerg']['hours_late_in_this_check']['era5']:+.1f} h and {de['imerg']['hours_late_in_this_check']['merra2']:+.1f} h off. The daily MERRA-2 "
    f"against ERA5 comes out {cal['merra2_daily_against_era5_hourly']['centre_shift_h']['median']:+.1f} h off. So the hourly references carry timing errors of "
    f"their own of one to two hours (ERA5 the larger), and no day end below is known better than that.",
    f"Four readings are therefore given for every product (day_ends.<product>.median_by_reading): against each hourly reference as measured, and against each "
    f"reference after shifting it, basin by basin, so that IMERG falls on its documented UTC day. Which of the three clocks (ERA5, MERRA-2, IMERG) is nearest "
    f"the truth is not known. The headline value of a basin is the mean of its four readings, and the headline of a product is the median basin. The readings do "
    f"not cover the same years: ERA5 hourly rain is used for {yrs(REF['era5']['period'])}, MERRA-2 hourly rain for {yrs(REF['merra2']['period'])} (NASA POWER serves "
    f"it by the hour from 2001 only), and the IMERG shift is measured from {W('era5', 'imerg')['period'][0][:4]} against ERA5 and from "
    f"{W('merra2', 'imerg')['period'][0][:4]} against MERRA-2. Over 2001-2014 alone the ERA5 reading is {W('era5', 'daymet', '2001_2014')['day_ends_local']['median']} "
    f"for Daymet and {W('era5', 'nldas', '2001_2014')['day_ends_local']['median']} for NLDAS (windows.era5.<product>.2001_2014), to be set against the MERRA-2 readings below.",
    f"Rain table (Daymet): the value of day d ends at about {de['daymet']['day_ends_local_rounded']} local standard time ON day d (median basin; basins "
    f"{de['daymet']['basin_range_local'][0]} to {de['daymet']['basin_range_local'][1]}). The four readings: {rd('daymet')}. {fit('daymet')} "
    f"Against ERA5 the Daymet day ends at {W('era5', 'daymet', 'before_1996')['day_ends_local']['median']} in 1980-1995, at "
    f"{W('era5', 'daymet', '2001_2014')['day_ends_local']['median']} in 2001-2014 and at {W('era5', 'daymet', 'test_years')['day_ends_local']['median']} in the "
    f"test years. That does not show that the Daymet window moved: in the same check the daily MERRA-2 value, whose window is the UTC day in every year by "
    f"construction, ends at {W('era5', 'merra2_daily', 'before_1996')['day_ends_local']['median']} in 1980-1995 and at "
    f"{W('era5', 'merra2_daily', '2001_2014')['day_ends_local']['median']} in 2001-2014. Basin by basin the day end changes by {ec['daymet']['median']:+.1f} h for Daymet "
    f"(median; earlier in {ec['daymet']['n_basins_earlier']} of {len(ids)} basins), by {ec['merra2_daily']['median']:+.1f} h for the daily MERRA-2 "
    f"({ec['merra2_daily']['n_basins_earlier']} of {len(ids)}) and by {ec['nldas']['median']:+.1f} h for NLDAS ({ec['nldas']['n_basins_earlier']} of {len(ids)}; era_change). "
    f"A product whose window cannot have moved changes too, so this check cannot tell a move of the Daymet window from a drift between the clocks of the references. "
    f"Single basins change by more (Daymet: {ec['daymet']['min']:+.1f} h to {ec['daymet']['max']:+.1f} h), as a table built from rain gauges whose reading hours "
    f"change could; the reading hours of the gauges were not checked here. In the test years the basins that do not fit are {', '.join(W('era5', 'daymet', 'test_years')['basins_that_do_not_fit']) or 'none'}"
    + (f"; in Myakka, FL the Daymet day then ends at {W('era5', 'daymet', 'test_years')['per_basin']['02298608']['day_ends_local']} against ERA5 and "
       f"{W('merra2', 'daymet', 'test_years')['per_basin']['02298608']['day_ends_local']} against MERRA-2 - close to the local day, so that basin has little "
       f"time-stamp gap." if "02298608" in ids else "."),
    f"NLDAS rain: the value of day d ends at about {de['nldas']['day_ends_local_rounded']} local standard time on day d (basins "
    f"{de['nldas']['basin_range_local'][0]} to {de['nldas']['basin_range_local'][1]}). The four readings: {rd('nldas')}. {fit('nldas')} That is the UTC day "
    f"(19:00) or a little earlier, within the accuracy of the check. The latest single reading of any basin is {clock(nldas_latest)}: the NLDAS value of the "
    f"issue day is complete hours before the forecast is issued.",
    f"Maurer rain (1980-2008): median basin {de['maurer']['day_ends_local']}, but the basins range from {de['maurer']['basin_range_local'][0]} to "
    f"{de['maurer']['basin_range_local'][1]}. {fit('maurer')} Maurer has no single day window in these basins, and in the median basin its value of day d reaches "
    f"past local midnight: the regression below gives the rain of the hours after local midnight a weight of {fut('before_1996_against_era5_hourly', 'maurer'):.2f} "
    f"in the Maurer value (before 1996, against ERA5; {fut('test_years_against_merra2_hourly', 'maurer'):.2f} for October 2005 to 2008 against MERRA-2). "
    f"No model of the project reads Maurer rain.",
    f"After-midnight rain in the NLDAS value of the issue day (future_rain): the daily value is regressed on the reference rain of six stretches around the day; a "
    f"weight of 1 means that stretch is fully inside the daily value. Weight of the rain between local midnight and 12:00 UTC of the next day (the first hours "
    f"after the forecast is issued), median basin: test years {fut('test_years_against_merra2_hourly', 'nldas'):.2f} against MERRA-2 and "
    f"{fut('test_years_against_era5_hourly', 'nldas'):.2f} against ERA5 - none. Years before 1996, against ERA5 (NASA POWER serves MERRA-2 by the hour from 2001 only): "
    f"{fut('before_1996_against_era5_hourly', 'nldas'):.2f}. That number is inconclusive, not a leak: in the same regression the daily MERRA-2, which ends at "
    f"24:00 UTC by construction, gets {fut('before_1996_against_era5_hourly', 'merra2_daily'):.2f} "
    f"(test years: {fut('test_years_against_era5_hourly', 'merra2_daily'):.2f}, IMERG {fut('test_years_against_era5_hourly', 'imerg'):.2f}). ERA5 places rain "
    f"late enough to produce a weight of that size for a product known to hold no such rain. The test years, on which every score of the project rests, are clean; "
    f"for the training years before 1996 this check can neither show nor exclude a blurred NLDAS day boundary.",
    f"The same regression for Daymet, test years against MERRA-2: weight {future['test_years_against_merra2_hourly']['products']['daymet']['median']['12h_before_the_utc_day']:.2f} "
    f"for the 12 hours before the UTC day (07:00 to 19:00 Eastern time of day d-1), "
    f"{future['test_years_against_merra2_hourly']['products']['daymet']['median']['utc_day_first_half']:.2f} for 19:00 of day d-1 to 07:00 of day d, and "
    f"{future['test_years_against_merra2_hourly']['products']['daymet']['median']['utc_day_second_half']:.2f} for 07:00 to 19:00 of day d: most of the daytime "
    f"rain of day d is not in the Daymet value of day d.",
]
if flow:
    fs, fw, fd = flow_summary["parts"], flow_summary["parts"]["standard_time_in_force"], flow_summary["parts"]["daylight_saving_time_in_force"]
    pc = lambda part, w: fs[part]["median_relative_difference_percent"][w]
    notes.append(
        f"River flow: 15-minute flow of {len(flow)} of the {len(ids)} gauges (USGS, {FLOW_DAYS[0]} to {FLOW_DAYS[1]}) averaged over 24-hour windows shifted hour by "
        f"hour and compared with the daily flow in the CAMELS files, every window on the same days; the best window is the one with the smallest mean relative difference. While standard time "
        f"is in force ({fw['days'][0]} to {fw['days'][1]}) the window from local midnight to midnight is the best one in {fw['n_gauges_best_midnight_standard_time']} "
        f"of {len(flow)} gauges (median difference {pc('standard_time_in_force', 'midnight_to_midnight_standard_time'):.2f}% of the daily flow, against "
        f"{pc('standard_time_in_force', 'utc_day'):.2f}% for the UTC day). While daylight-saving time is in force ({fd['days'][0]} to {fd['days'][1]}) the best "
        f"window is midnight to midnight by the summer clock - one hour earlier in standard time - in {fd['n_gauges_best_midnight_summer_time']} gauges, and "
        f"standard-time midnight in {fd['n_gauges_best_midnight_standard_time']} ({', '.join(fd['gauges_best_midnight_standard_time']) or 'none'}); median "
        f"difference {pc('daylight_saving_time_in_force', 'midnight_to_midnight_summer_time'):.2f}% by the summer clock, "
        f"{pc('daylight_saving_time_in_force', 'midnight_to_midnight_standard_time'):.2f}% by standard time, {pc('daylight_saving_time_in_force', 'utc_day'):.2f}% "
        f"for the UTC day. So a daily flow value is the mean from midnight to midnight by the local clock, not the UTC day: 'the end of day t', when a forecast is "
        f"issued, is local midnight - in summer, at most gauges, 23:00 on the standard-time axis used in this file and in the figure."
        + (f" French Broad, NC, over the whole period, as the reviewer of 10_rain_timing.py measured it: RMSE {fb['whole_period']['midnight_to_midnight_standard_time']['rmse_cfs']:.2f} cfs "
           f"for standard-time midnight, {fb['whole_period']['midnight_to_midnight_summer_time']['rmse_cfs']:.2f} cfs for summer-time midnight and "
           f"{fb['whole_period']['utc_day']['rmse_cfs']:.2f} cfs for the UTC day (mean flow {fb['whole_period']['mean_flow_cfs']:.0f} cfs)." if fb else "")
        + " Six months of one year were checked, not the whole record.")
    notes.append(
        f"Summer time and the rain checks: all rain windows are given in standard time, and the regression counts rain as 'after the forecast' from standard-time "
        f"midnight on. In summer the flow day of most gauges ends one hour earlier, so the last hour before standard-time midnight is then also after the forecast. "
        f"That hour lies in the stretch 'evening until local midnight', whose weight in the NLDAS value of the test years is "
        f"{future['test_years_against_merra2_hourly']['products']['nldas']['median']['evening_until_local_midnight']:.2f} against MERRA-2 and "
        f"{future['test_years_against_era5_hourly']['products']['nldas']['median']['evening_until_local_midnight']:.2f} against ERA5 (the daily MERRA-2, which holds "
        f"none of that rain by construction, gets {future['test_years_against_era5_hourly']['products']['merra2_daily']['median']['evening_until_local_midnight']:.2f} "
        f"against ERA5).")
if no_flow:
    notes.append(f"No 15-minute flow could be had for: {', '.join(no_flow)}; the flow check does not cover them.")
notes += [
    "Limits. (1) Both references are reanalyses: their rain is computed by a weather model that is corrected with observations, and its hour-by-hour timing is "
    "less sure than its daily total; calibration 2 puts a number on that. (2) The references are values of one grid cell at the basin centroid (ERA5: " + era5_grid + "; MERRA-2: "
    "0.5 x 0.625 degrees), the daily products are basin averages (IMERG: its 0.1 degree cells weighted by area); a storm can reach the cell and "
    "the basin at different hours. (3) The method finds the ONE 24-hour window that fits best. A table built from stations that are read at different hours is a "
    "mix of windows: its curve is broad, and the centre is an average, not a sharp boundary - so 'ends at about " + de["daymet"]["day_ends_local_rounded"] + "' does not mean that no later rain is inside. "
    "(4) A shifted window cannot be told from a product whose rain is smeared in time, nor from a timing error of the reference. (5) Hours are standard time; "
    "the shifts are whole hours. (6) Correlations are dominated by the large storms.",
    f"Sources. ERA5 hourly precipitation: Open-Meteo historical weather API (archive-api.open-meteo.com, models=era5, timezone UTC), 1980-09-25 to 2014-10-05, one "
    f"request per basin centroid; the service returns the grid point it used (sources.era5.per_basin). Its value at time T is taken as the rain of the hour "
    f"before T, as the service documents it; if that were the hour after, every ERA5 reading would move by one hour. MERRA-2 hourly: NASA POWER hourly point API, "
    f"parameter PRECTOTCORR ('Precipitation Corrected', mm/hour), time standard UTC, 2001-01-01 to 2014-10-05. USGS: instantaneous-values service "
    f"(waterservices.usgs.gov/nwis/iv, parameter 00060, discharge) and site file. Retrieval dates are the dates of the cached files in data/hourly.",
    "Fair test: nothing is trained, chosen or tuned here, and no forecast is scored. The check describes the input data over the whole record; the eras "
    "(before 1996, 2001-2014, the test years) are reported side by side.",
]
if F.QUICK:
    notes.append(f"Smoke test: {len(ids)} basins and shifts every {STEP} hours - not a result.")

# ---- JSON ----------------------------------------------------------------------------------------------------------------
sources = dict(
    era5=dict(service="Open-Meteo historical weather API (archive-api.open-meteo.com/v1/archive)", product="ERA5 hourly precipitation (models=era5), mm per hour",
              period=["1980-09-25", "2014-10-05"], time_standard="UTC", grid="nearest grid point of the service to the basin centroid (per_basin)",
              retrieved=sorted({retrieved(FILE["era5"](g)) for g in ids}), files="data/hourly/era5_<gauge_id>.json", per_basin=src_meta["era5"]),
    merra2=dict(service="NASA POWER hourly point API (power.larc.nasa.gov/api/temporal/hourly/point)", product="MERRA-2 PRECTOTCORR (Precipitation Corrected), mm per hour",
                period=["2001-01-01", "2014-10-05"], time_standard="UTC", grid="MERRA-2 cell that holds the basin centroid, 0.5 x 0.625 degrees (results/satellite_data.json)",
                retrieved=sorted({retrieved(FILE["merra2"](g)) for g in ids}), files="data/hourly/merra_<gauge_id>.json", per_basin=src_meta["merra2"]),
    usgs_flow=dict(service="USGS instantaneous-values service (waterservices.usgs.gov/nwis/iv)", product="discharge (parameter 00060), 15-minute values, cubic feet per second",
                   period=["2012-01-01", "2012-06-30"], files="data/hourly/usgs_iv_<gauge_id>.rdb", retrieved=sorted({v["retrieved"] for v in flow.values()})),
    usgs_sites=dict(service="USGS site service (waterservices.usgs.gov/nwis/site)", product="time zone of each gauge", files="data/hourly/usgs_sites.rdb",
                    retrieved=retrieved(SITES), time_zone={g: st.tz_cd[g] for g in ids}),
    daily_products=dict(daymet="CAMELS-US basin_mean_forcing/daymet (basin average), 1980-2014", nldas="CAMELS-US basin_mean_forcing/nldas (basin average), 1980-2014",
                        maurer="CAMELS-US basin_mean_forcing/maurer (basin average), 1980-2008", imerg="data/satellite/<gauge_id>.csv: IMERG by UTC day, NASA POWER daily (code/fetch_satellite.py)",
                        merra2_daily="data/satellite/<gauge_id>.csv: MERRA-2 rain by UTC day, NASA POWER daily, basin centroid", river_flow="CAMELS-US usgs_streamflow (daily mean flow)"),
    basin_centroids_lon_lat={g: CENT[g] for g in ids})
R = dict(
    question="Which 24 hours does each daily value of 'day d' cover, measured against hourly rain (and, for the river flow, 15-minute flow)?",
    quick_mode=F.QUICK, basins={g: F.SHORT[g] for g in ids}, shifts_h=[int(H[0]), int(H[-1]), STEP],
    how_to_read="shift h: the best-matching 24-hour window starts h hours after 00:00 UTC of day d. day_ends_local: the hour of day d, in local standard time, at "
                "which that window ends (24:00 UTC + h). The forecast of the project is issued at the end of the river-flow day t: local midnight (river_flow_day).",
    day_ends=day_ends, readings=READINGS, river_flow_day=dict(summary=flow_summary, days=list(FLOW_DAYS), per_gauge=flow),
    calibration=cal, future_rain=dict(bins_hours_after_00_utc="[-12, 0, 12, 24, local midnight (29 Eastern, 30 Central), 36, 48]", bins=BINS, runs=future),
    windows=windows, era_change=era_change, sources=sources, notes=notes, runtime_s=round(time.time() - t_start, 1))
(F.RES_OUT / "day_windows.json").write_text(json.dumps(R, indent=1))

# ---- figure: the 24 hours behind 'day t', on one clock --------------------------------------------------------------------
FS = 8.2
X0, X1 = -42, 12                                            # hours from the issue time (local midnight at the end of day t)
fig, ax = plt.subplots(figsize=(6.7, 4.1))
ax.axvspan(-24, 0, color=GRID, alpha=.45, lw=0, zorder=0)
rel = lambda h: h - 24                                      # local hour of day t -> hours from the issue time
flow_end = float(np.median([v["standard_time_in_force"]["best_window_ends_local_hour"] for v in flow.values()])) if flow else 24.0
ROWS = [("river flow (gauge): midnight to midnight", INK2, flow_end, []),
        (f"rain table (Daymet): ends about {de['daymet']['day_ends_local_rounded']} of day t", ORANGE, de["daymet"]["day_ends_local_hour"],
         [de["daymet"]["per_basin"][g]["middle"] for g in ids]),
        (f"NLDAS rain: ends about {de['nldas']['day_ends_local_rounded']}", BLUE, de["nldas"]["day_ends_local_hour"], [de["nldas"]["per_basin"][g]["middle"] for g in ids]),
        (f"satellite rain (IMERG): ends {de['imerg']['documented_day_ends_local']} (documented)", AQUA, de["imerg"]["documented_day_ends_local_hour"],
         [np.mean([de["imerg"]["per_basin"][g][s] for s in REF]) for g in ids])]
BAR = 0.36
for i, (lab, col, end, per) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    ax.add_patch(Rectangle((rel(end) - 24, y - BAR / 2), 24, BAR, color=col, lw=0, zorder=3))
    ax.text(max(rel(end) - 24, X0 + .4), y + BAR / 2 + .07, lab, fontsize=FS, color=INK, ha="left", va="bottom", zorder=5)
    ax.vlines([rel(e) for e in per if np.isfinite(e)], y - BAR / 2 - .17, y - BAR / 2 - .05, color=col, lw=1.0, zorder=4)
    if i == 1:                                              # what the forecast misses: fallen, but booked under the next day
        ax.add_patch(Rectangle((rel(end), y - BAR / 2), -rel(end), BAR, facecolor=col, alpha=.14, lw=0, zorder=2))
        ax.add_patch(Rectangle((rel(end), y - BAR / 2), -rel(end), BAR, fill=False, edgecolor=col, lw=1, ls=(0, (3, 2)), zorder=3))
        ax.text(rel(end) / 2, y, "fallen, but stamped day t+1", fontsize=FS - .4, color=INK, ha="center", va="center", zorder=5)
        far = [(g, de["daymet"]["per_basin"][g]["middle"]) for g in ids if F.SHORT[g] in de["daymet"]["basins_that_do_not_fit"]]
        for g, e in far:
            ax.text(rel(e), y - BAR / 2 - .2, F.SHORT[g], fontsize=FS - .4, color=INK2, ha="center", va="top", zorder=5)
ax.axvline(0, color=INK, lw=1.4, zorder=6)
ax.text(.5, len(ROWS) - 1, "forecast is issued\nat the end of day t", fontsize=FS, color=INK, ha="left", va="center", zorder=6)
for x, lab in ((-33, "day t-1"), (-12, "day t"), (6, "day t+1")):
    ax.text(x, len(ROWS) - .18, lab, fontsize=FS + .6, color=INK2, ha="center", va="bottom", fontweight="bold")
ticks = np.arange(X0, X1 + 1, 6)
ax.set_xticks(ticks, [f"{int(t) % 24:02d}:00" for t in ticks], fontsize=FS)
ax.set_xlim(X0, X1); ax.set_ylim(-.62, len(ROWS) + .12); ax.set_yticks([]); ax.grid(axis="y", visible=False)
for x in (-24, 0):
    ax.axvline(x, color=AXIS, lw=1, zorder=1)
ax.spines["left"].set_visible(False)
ax.set_xlabel("local standard time", fontsize=FS)
ax.set_title("Which 24 hours does the value of 'day t' cover?", fontsize=FS + 2.3)
late = np.mean(list(de["imerg"]["hours_late_in_this_check"].values()))
n_summer = flow_summary["parts"]["daylight_saving_time_in_force"]["n_gauges_best_midnight_summer_time"]
fig.text(.012, .012, f"Bars: median of the {len(ids)} basins; small ticks: where the day ends in each basin. Rain windows are measured against hourly rain of two\n"
                     f"weather reanalyses (ERA5, MERRA-2). The same check puts the IMERG day about {late:.1f} h after its documented end (its ticks): the rain\n"
                     f"windows are good to about two hours. River flow: 15-minute flow of {len(flow)} gauges, January-June 2012; its day follows the local\n"
                     f"clock: in summer (daylight-saving time) it ends one hour earlier on this axis at {n_summer} of the {len(flow)} gauges.",
         fontsize=FS - .5, color=INK2, ha="left", va="bottom", linespacing=1.35)
fig.tight_layout(rect=(0, .17, 1, 1))
fig.savefig(F.FIG_OUT / "day_windows.png", dpi=220); plt.close(fig)

# ---- printout ------------------------------------------------------------------------------------------------------------
for p in ("daymet", "nldas", "maurer"):
    print(f"{PRODUCTS[p]:24s} day ends {de[p]['day_ends_local']} (basins {de[p]['basin_range_local'][0]} to {de[p]['basin_range_local'][1]}) | "
          + " | ".join(f"{k}: {v['local']}" for k, v in de[p]["median_by_reading"].items()) + f" | not fitting: {de[p]['basins_that_do_not_fit']}")
print(f"{PRODUCTS['imerg']:24s} documented end {de['imerg']['documented_day_ends_local']}; this check: {de['imerg']['found_by_this_check_local']}")
for k, v in cal.items():
    print(f"calibration {k}: best shift {v['best_shift_h']}, centre {v['centre_shift_h']}, corr {v['corr_best_median']}")
for k, v in future.items():
    print(f"after-midnight rain, {k}: " + ", ".join(f"{p} {x['after_local_midnight']['median']:.2f}" for p, x in v["products"].items()))
print("river flow:", json.dumps(flow_summary))
print(f"wrote {F.RES_OUT / 'day_windows.json'} and {F.FIG_OUT / 'day_windows.png'} in {time.time() - t_start:.0f} s")
