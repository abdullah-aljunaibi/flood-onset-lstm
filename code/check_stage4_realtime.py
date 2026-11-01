"""The real-time test of E1b: NCEP Stage IV rain in place of AORC on the issue day of every test-year onset, with the saved networks (nothing is trained).

Why. AORC is a retrospective analysis (daily totals on a gauge day that ends at 12 UTC, spread over the hours with radar; hours before 2002 are disaggregated daily totals; about 10 days behind real
time), so the 51% of onsets that the LSTM with the AORC rain to midnight warns of 1 day ahead (code/16_aorc_rain_to_midnight.py) is what well-measured rain up to midnight is worth, not what a service
at midnight would have. NCEP Stage IV (radar plus gauge analysis, hourly, about one hour behind the hour) is the product such a service would read. The independent review of E1b
(results/review_aorc_rain_to_midnight.json) tested this on 171 of the 611 onsets (56 of 368 issue days); this script does it for every onset.

What. For each of the 611 onsets at lead 1 of the test years (results/flood_events.csv: the issue day t is onset_date - 1; the onset is the first day above the high-flow level after a day at or below it)
the value of aorc_to_midnight of day t (the rain of the wall-clock day t, 00:00 to 24:00 at the gauge, summer time included) is replaced by the sum of the Stage IV hours of the same wall-clock day t
(hourly basin means of the 4.76 km cells whose centre lies inside the CAMELS outline; the file labelled T holds the hour that ENDS at T, UTC), and nothing else changes: every other day keeps AORC, so the
network state differs from the AORC run only through those days (variant 'own days'). The saved lstm_wqa networks (results/models/lstm_wqa_s<seed>.pt, every seed of results/aorc_rain_to_midnight.json, trained on AORC)
forecast again, the seeds are averaged as in the results file, and the onsets warned at trigger 1.0 are counted, against the same count with AORC and with the rain table alone (lstm_wq), both read from that
results file. Share of the AORC effect that
survives = (Stage IV - rain table) / (AORC - rain table), overall and per basin, with 95% intervals from drawing the onsets again and from drawing the issue days again (the onsets of one storm are not independent).
Variants, all from the same Stage IV hours: raw; times k = (AORC over Stage IV on those days), because the networks were trained on a product that holds about 10% more rain than Stage IV; with the last
hour of the issue day taken out (a service at midnight would have the hours to about 23:00: Stage IV is about one hour behind), and the last two hours; Stage IV on the issue day AND the day before
(needs the Stage IV hours of day t-1: used only if they were fetched); Stage IV on all 368 issue days of all 12 basins (the population on which the false warnings are counted: the warnings issued on those
days when the river was low, hits and false warnings, with AORC and with Stage IV); and three controls on the rain of the issue day (the Stage IV rain of another onset of the same basin, 10 random permutations;
the AORC rain of a random ordinary day of the basin, 10 draws; no rain), which show what the count can and cannot tell: it is very sensitive to heavy rain on the issue day and little to WHICH heavy-rain day, so
the fit of the forecasts to the observed flow on the onset days (mean absolute error, Spearman correlation) is reported with every variant as the measure that does depend on the exact rain. The same for the saved
lstm_wqa_rep networks (the Daymet rain column replaced by AORC; the same seeds) as a second model.
Honesty points: the networks were trained on AORC and Stage IV is substituted at forecast time only; Stage IV covers 2002 onward (all test years, none of the training years: nothing is said about
training on it); the one-hour delay of Stage IV is not simulated except by the 'last hour out' variants; Stage IV hours that the archive lacks (or in which no cell of the basin has radar coverage) are
replaced by the AORC hour and counted; the Iowa Mesonet archive holds an unknown mix of first postings and later versions of each hour (the review could not establish which); the issue days are those of
the onsets, so the false warnings on days without any onset are NOT tested except on the 368 issue days of all basins.

Data. data/aorc/stage4_issue_days.csv (hour-ending UTC label x 12 basin means in mm, and the source of each row: fetched here, the builder's data/aorc/stage_iv_hindsight.csv, or the review's cache)
holds every Stage IV hour of the wall-clock days of the issue days (26 labels a day cover both time zones). The hours missing from it are fetched from the Iowa Environmental Mesonet archive
(https://mesonet.agron.iastate.edu/archive/data/YYYY/MM/DD/stage4/ST4.YYYYMMDDHH.01h.grib; a volunteer-run service: at most 6 connections, retries with back-off, stop when it throttles), each file is read
(rasterio), reduced to the 12 basin means at once and dropped: about 1.2 MB a file: 7,525 hours (8.9 GB, about 48 minutes with 6 connections) for the issue days that the builder and the review had not fetched, and about 6,200 more (7.4 GB) for the days before (--day-before). Resume-safe: the CSV is
written after every batch and a second run asks for nothing. THIS NEEDS THE NETWORK ON A FRESH CLONE (not in run_all.sh for that reason); with the CSV in place it makes no request.
Run:  python code/check_stage4_realtime.py                 (fetches what is missing, then everything below)
      python code/check_stage4_realtime.py --fetch-only    (the data only; add --day-before to fetch day t-1 as well)
      FLOOD_QUICK=1 python code/check_stage4_realtime.py   (smoke test: 4 issue days that are cached, results/quick and figures/quick)
Environment: FLOOD_THREADS (torch threads, default 2: forward passes only), STAGE4_EXTRA_CACHES (more CSVs of hourly Stage IV basin means, separated by ':', read before anything is fetched).
Reads results/flood_events.csv, results/models/lstm_wqa*_s*.pt, results/preds/lstm_wq.npz and lstm_wqa*.npz, data/aorc (code/fetch_aorc.py), results/day_windows.json and
results/aorc_rain_to_midnight.json (the warning level chosen on the validation years). Writes results/stage4_realtime.json and figures/stage4_realtime.png.
"""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("FLOOD_THREADS", "2")                  # before flood.py is imported: forward passes only, and another job may hold the CPU
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import fetch_aorc as AO
import flood as F
from common import *

SCRIPT = "check_stage4_realtime.py"
OUT = "stage4_realtime"
QUICK = F.QUICK
FETCH_ONLY = "--fetch-only" in sys.argv
DAY_BEFORE = "--day-before" in sys.argv or os.environ.get("STAGE4_DAY_BEFORE") == "1"
MAX_CONN, BATCH, STOP_AFTER_BAD, STOP_AFTER_THROTTLED = 6, 120, 40, 30    # the archive is run by volunteers: 6 connections at most; a CSV write after every batch; stop after this many failures in a row, or this many 429/503 answers
HOUR = pd.Timedelta(hours=1)
N_QUICK_DAYS = 4
N_BOOT, N_PLACEBO = (200, 3) if QUICK else (2000, 10)
t_start = time.time()
if not QUICK and hasattr(os, "nice"):
    os.nice(10)                                              # forward passes and downloads yield to a training that holds the CPU
STAGE4_CSV = DATA / "aorc" / "stage4_issue_days.csv"
UNUSABLE_JSON = DATA / "aorc" / "stage4_issue_days_unusable.json"                       # the hours whose file was not the one-hour rain of its label, and why
FETCH_RUNS_JSON = DATA / "aorc" / "stage4_issue_days_fetch_runs.json"                   # one record per fetching run: files, bytes, seconds, what the archive did
BUILDER_CSV = DATA / "aorc" / "stage_iv_hindsight.csv"
EXTRA = [DATA / "aorc" / "stage_iv_hindsight.csv"] + [p for p in os.environ.get("STAGE4_EXTRA_CACHES", "").split(":") if p]
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)
say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)

D, B = F.load_data(), F.basins()
ids = list(D)
thr = F.thresholds(D)
DAYS = pd.date_range(F.PERIOD[0], F.PERIOD[1])
TEST = F.SPLIT["test"]
DW = json.loads((F.RES / "day_windows.json").read_text())
UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}
ZONE = {-5: "America/New_York", -6: "America/Chicago"}
ZONE_OF = {g: ZONE[UTC_OFF[g]] for g in ids}
assert ids == AO.ids_of_the_study()


# ---- the onsets (flood_events.csv) and the issue days ---------------------------------------------------------------------------------------------------
def onsets_table():
    """One row per onset at lead 1 in the test years: basin, target day (the onset date) and issue day t = onset date - 1; checked against the onsets that flood.flood_scores counts."""
    ev = pd.read_csv(F.RES / "flood_events.csv", dtype={"gauge_id": str})
    assert ev.river_low_at_issue_l1.all() and not ev.started_before_test.any()
    ev["target"] = pd.to_datetime(ev.onset_date)
    ev["issue"] = ev.target - pd.Timedelta(days=1)
    mine = set()
    for g in ids:
        q = D[g].q_mm
        on = (q > thr[g]) & (q.shift(1) <= thr[g]) & (q.index >= pd.Timestamp(TEST[0])) & (q.index <= pd.Timestamp(TEST[1]))
        mine |= {(g, d) for d in q.index[on.values]}
    assert mine == set(zip(ev.gauge_id, ev.target)), "the onsets of flood_events.csv are not the onsets that the flow table and the high-flow levels give"
    return ev[["gauge_id", "basin", "target", "issue"]].sort_values(["gauge_id", "target"]).reset_index(drop=True)


ONSETS = onsets_table()
assert len(ONSETS) == 611, len(ONSETS)


# ---- the hours of a wall-clock day, as labels (hour-ending UTC) -------------------------------------------------------------------------------------
def day_labels(day, zone):
    """Hour-ending UTC labels of the hours inside the wall-clock day `day` of a zone (24, or 23 or 25 where the clocks change)."""
    z = ZoneInfo(zone)
    a = pd.Timestamp(day).tz_localize(z).tz_convert("UTC").tz_localize(None)
    b = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(z).tz_convert("UTC").tz_localize(None)
    return pd.date_range(a + HOUR, b, freq="h")


def labels_of_days(days):
    """The labels for both time zones of the study (26 a day): the union over the zones."""
    return sorted({t for d in days for z in set(ZONE_OF.values()) for t in day_labels(d, z)})


# ---- Stage IV: the cache and the fetch ---------------------------------------------------------------------------------------------------------------
def read_cache(path):
    """A CSV of hourly Stage IV basin means (index: hour-ending UTC label); an empty table with a time index when the file is not there."""
    if not os.path.exists(path):
        return pd.DataFrame({**{g: pd.Series(dtype=float) for g in ids}, "source": pd.Series(dtype=object)}, index=pd.DatetimeIndex([]))
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    if "source" not in df.columns:
        df["source"] = "unlabelled"
    return df


def stage4_masks_builder():
    """A closure that builds the 12 basin masks on the grid of the first file read, and reduces a GRIB file to 12 basin means (mean of the cells whose centre lies inside the outline; the cell that holds
    the centroid for a basin without a cell centre; 9999, no radar coverage, left out). Same method as code/fetch_aorc.py (which made data/aorc/stage_iv_hindsight.csv); the review's own reducer agrees to 4e-16."""
    import pyproj
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.ops import transform as shp_transform
    polys = AO.outlines(ids)
    crs = pyproj.CRS.from_proj4(AO.STAGE_IV_CRS)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    state, lock = {}, threading.Lock()

    def build(tf, h, w):
        ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(*(tf * (0.5, h - 0.5)))
        assert -121.0 < ll[0] < -118.0 and 22.0 < ll[1] < 24.0, f"the Stage IV grid is not where it is assumed to be: lower-left cell centre {ll}"
        masks = {}
        for g in ids:
            pg = shp_transform(fwd, polys[g])
            m = geometry_mask([pg], out_shape=(h, w), transform=tf, invert=True)
            if m.sum() == 0:
                col, row = ~tf * (pg.centroid.x, pg.centroid.y)
                m = np.zeros((h, w), bool)
                m[int(row), int(col)] = True
            masks[g] = m
        return masks

    def reduce(raw, t):
        with tempfile.NamedTemporaryFile(suffix=".grib") as tmp:
            tmp.write(raw)
            tmp.flush()
            with rasterio.open(tmp.name) as ds:
                a = ds.read(1).astype("float64")
                tags, tf, h, w = ds.tags(1), ds.transform, ds.height, ds.width
        if not (tags.get("GRIB_ELEMENT") == "APCP" and int(tags.get("GRIB_FORECAST_SECONDS", -1)) == 3600 and int(tags.get("GRIB_VALID_TIME", -1)) == int((t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1))):
            # not the one-hour rain that ends at the label (for example 2008-03-16 00:00: a field with the parameter 'var237 of table 128 of center 7', unit '-'): not used, the AORC hour stands in for it and the hour is counted
            return "unusable", f"element {tags.get('GRIB_ELEMENT')!r}, unit {tags.get('GRIB_UNIT')!r}, forecast seconds {tags.get('GRIB_FORECAST_SECONDS')}, valid time {tags.get('GRIB_VALID_TIME')}"
        with lock:
            if "masks" not in state:
                state["masks"] = build(tf, h, w)
        a[a >= 9998] = np.nan
        return {g: float(np.nanmean(a[m])) if np.isfinite(a[m]).any() else np.nan for g, m in state["masks"].items()}
    return reduce


def fetch_labels(todo, cache_df):
    """Fetch the hour-ending labels in `todo` (at most MAX_CONN connections, batches, back-off, stop on throttling), reduce each file at once, write STAGE4_CSV after every batch.
    Returns (cache_df with the new rows, stats). A file the archive does not have (404 twice) is stored as NaN with source 'absent'; a file that could not be fetched is not stored (a rerun asks again)."""
    import requests
    reduce = stage4_masks_builder()
    loc, lock, stop = threading.local(), threading.Lock(), threading.Event()
    st = dict(files=0, bytes=0, absent=0, unusable=0, failed=0, bad_in_a_row=0, throttled=0)
    unusable = json.loads(UNUSABLE_JSON.read_text()) if UNUSABLE_JSON.exists() else {}            # label -> why the file could not be used

    def get(url):
        last = None
        for k in range(6):
            if stop.is_set():
                return None, "stopped"
            try:
                if getattr(loc, "s", None) is None:
                    loc.s = requests.Session()                                         # takes the proxy from the environment
                r = loc.s.get(url, timeout=120)
                if r.status_code == 200:
                    return r.content, 200
                if r.status_code in (403, 404):
                    if k >= 1:
                        return None, 404
                else:
                    with lock:
                        st["throttled"] += int(r.status_code in (429, 503))
                        if st["throttled"] >= STOP_AFTER_THROTTLED:                      # the archive asks us to slow down again and again: stop here and report what we have
                            stop.set()
                    if r.status_code in (429, 503):
                        time.sleep(30)
                last = f"HTTP {r.status_code}"
            except (requests.RequestException, OSError) as e:
                last = type(e).__name__
            loc.s = None
            time.sleep(min(60, 2 ** (k + 1)))
        return None, last

    def one(t):
        raw, status = get(AO.STAGE_IV.format(y=f"{t:%Y}", m=f"{t:%m}", d=f"{t:%d}", h=f"{t:%H}"))
        with lock:
            if raw is None and status != 404:
                st["failed"] += 1
                st["bad_in_a_row"] += 1
                if st["bad_in_a_row"] >= STOP_AFTER_BAD:
                    stop.set()
                return t, None, 0, status
            st["bad_in_a_row"] = 0
        if raw is None:
            return t, {g: np.nan for g in ids}, 0, "absent"
        try:
            r = reduce(raw, t)
            if isinstance(r, tuple) and r[0] == "unusable":
                with lock:
                    unusable[str(t)] = r[1]
                return t, {g: np.nan for g in ids}, len(raw), "unusable"
            return t, r, len(raw), "fetched"
        except Exception as e:                                                          # a broken download: not stored, a rerun asks again
            with lock:
                st["failed"] += 1
                st["bad_in_a_row"] += 1
            return t, None, 0, f"unreadable: {type(e).__name__}"

    rows, t0, done = {}, time.time(), 0
    with ThreadPoolExecutor(MAX_CONN) as ex:
        for i in range(0, len(todo), BATCH):
            if stop.is_set():
                break
            for t, r, nbytes, status in ex.map(one, todo[i:i + BATCH]):
                if r is None:
                    continue
                rows[t] = dict(r, source=status if status in ("absent", "unusable") else "fetched")
                st["files"] += int(status == "fetched")
                st["absent"] += int(status == "absent")
                st["unusable"] += int(status == "unusable")
                st["bytes"] += nbytes
            done = min(i + BATCH, len(todo))
            new = pd.DataFrame(rows).T[ids + ["source"]]
            new.index = pd.DatetimeIndex(new.index)
            for g in ids:
                new[g] = new[g].astype(float)
            out = pd.concat([cache_df, new]).sort_index()
            out.index.name = "hour_ending_utc"
            tmp = STAGE4_CSV.with_suffix(".tmp")
            out.to_csv(tmp)
            tmp.replace(STAGE4_CSV)
            UNUSABLE_JSON.write_text(json.dumps(unusable, indent=1))
            el = time.time() - t0
            say(f"  {done}/{len(todo)} hours, {st['bytes'] / 1e9:.2f} GB, {el:.0f} s, {done / el:.2f} files/s, eta {(len(todo) - done) / max(done / el, 1e-9) / 60:.0f} min, "
                f"absent {st['absent']}, unusable {st['unusable']}, failed {st['failed']}, throttled {st['throttled']}")
    st["seconds"] = round(time.time() - t0, 1)
    st["stopped_early"] = bool(stop.is_set())
    if unusable:
        UNUSABLE_JSON.write_text(json.dumps(unusable, indent=1))
    new = pd.DataFrame(rows).T[ids + ["source"]] if rows else pd.DataFrame(columns=ids + ["source"])
    if len(new):
        new.index = pd.DatetimeIndex(new.index)
        for g in ids:
            new[g] = new[g].astype(float)
    out = pd.concat([cache_df, new]).sort_index()
    out.index.name = "hour_ending_utc"
    return out, st


def ensure_stage4(labels):
    """The cache holding every label of `labels` that can be had: own CSV, then the other caches (copied in, tagged), then the archive. Returns (DataFrame, record of what was done)."""
    own = read_cache(STAGE4_CSV)
    want = pd.DatetimeIndex(sorted(set(labels)))
    rec = dict(labels_asked=int(len(want)), already_in_own_csv=int(want.isin(own.index).sum()))
    missing = want[~want.isin(own.index)]
    copied = {}
    for p in EXTRA:
        if len(missing) == 0 or not os.path.exists(p):
            continue
        other = read_cache(p)
        common = missing.intersection(other.index)
        if len(common):
            # the other caches were made by other code: where two of them hold the same hour they must agree
            both = common.intersection(own.index)
            if len(both):
                assert np.nanmax(np.abs(own.loc[both, ids].values - other.loc[both, ids].values.astype(float))) < 1e-9
            add = other.loc[common, ids].copy()
            add["source"] = "builder_csv" if os.path.abspath(p) == str(BUILDER_CSV) else "review_cache"
            own = pd.concat([own, add]).sort_index()
            copied[p] = int(len(common))
            missing = want[~want.isin(own.index)]
    rec["copied_from_other_caches"] = {rel(Path(p)): n for p, n in copied.items()}
    if copied:
        own.index.name = "hour_ending_utc"
        tmp = STAGE4_CSV.with_suffix(".tmp")
        own.to_csv(tmp)
        tmp.replace(STAGE4_CSV)
    rec["to_fetch"] = int(len(missing))
    if len(missing) and not os.environ.get("STAGE4_NO_FETCH"):
        say(f"{len(missing)} Stage IV hours to fetch from the Iowa Mesonet archive with {MAX_CONN} connections (about {len(missing) * 1.2 / 1000:.1f} GB)")
        own, st = fetch_labels(list(missing), own)
        rec["fetch"] = st
        runs = json.loads(FETCH_RUNS_JSON.read_text()) if FETCH_RUNS_JSON.exists() else []
        runs.append(dict(started=time.strftime("%Y-%m-%d %H:%M", time.localtime(t_start)), connections=MAX_CONN, hours_asked=int(len(missing)), day_before=bool(DAY_BEFORE),
                         **{k: st[k] for k in ("files", "bytes", "seconds", "absent", "unusable", "failed", "throttled", "stopped_early")}))
        FETCH_RUNS_JSON.write_text(json.dumps(runs, indent=1))
        say(f"fetched {st['files']} files, {st['bytes'] / 1e9:.2f} GB in {st['seconds']:.0f} s; absent {st['absent']}, unusable {st['unusable']}, failed {st['failed']}, stopped early: {st['stopped_early']}")
    return own, rec


# ---- the days to cover -----------------------------------------------------------------------------------------------------------------------------------
issue_days = [pd.Timestamp(d) for d in sorted(ONSETS.issue.unique())]
before_days = sorted(set(pd.Timestamp(d) - pd.Timedelta(days=1) for d in issue_days) - set(issue_days))
if QUICK:                                                    # a handful of issue days that are cached already (no network), otherwise the first ones
    cached = read_cache(STAGE4_CSV).index.union(read_cache(BUILDER_CSV).index)
    for p in EXTRA:
        if os.path.exists(p):
            cached = cached.union(read_cache(p).index)
    ok = [d for d in issue_days if all(t in cached for t in labels_of_days([d]))]
    issue_days = (ok or issue_days)[:N_QUICK_DAYS]
    before_days = sorted(set(pd.Timestamp(d) - pd.Timedelta(days=1) for d in issue_days) - set(issue_days))
want_days = list(issue_days) + (list(before_days) if DAY_BEFORE else [])
say(f"{len(ONSETS)} onsets on {ONSETS.issue.nunique()} issue days; {'quick: ' if QUICK else ''}{len(issue_days)} issue days used" + (f", {len(before_days)} days before" if want_days != list(issue_days) else ""))
S4, DATA_REC = ensure_stage4(labels_of_days(want_days))
_needed = S4.loc[S4.index.isin(labels_of_days(want_days))]
_unusable = json.loads(UNUSABLE_JSON.read_text()) if UNUSABLE_JSON.exists() else {}
DATA_REC.update(csv=rel(STAGE4_CSV), rows_in_csv=int(len(S4)), hours_needed=int(len(_needed)), sources_of_the_hours_needed={k: int(v) for k, v in _needed.source.value_counts().items()},
                hours_without_a_usable_file=dict(absent_from_the_archive=int((_needed.source == "absent").sum()), file_is_not_the_one_hour_rain_of_its_label=int((_needed.source == "unusable").sum()),
                                                 why_the_unusable_ones=_unusable, sidecar=rel(UNUSABLE_JSON) if _unusable else None, treatment="the AORC hour stands in for the hour, and the hours are counted (fallback)"),
                issue_days=int(len(issue_days)), days_before=int(len(before_days)) if want_days != list(issue_days) else 0)
if FETCH_RUNS_JSON.exists():
    _runs = json.loads(FETCH_RUNS_JSON.read_text())
    DATA_REC["fetch_runs"] = dict(runs=_runs, files=int(sum(r["files"] for r in _runs)), gb=round(sum(r["bytes"] for r in _runs) / 1e9, 2), seconds=round(sum(r["seconds"] for r in _runs), 1),
                                  hours=round(sum(r["seconds"] for r in _runs) / 3600, 2), connections=MAX_CONN, source="Iowa Environmental Mesonet archive, https://mesonet.agron.iastate.edu/archive/data/YYYY/MM/DD/stage4/ST4.YYYYMMDDHH.01h.grib",
                                  note="every file was read, reduced to the 12 basin means and dropped; the other hours of the CSV come from the builder's data/aorc/stage_iv_hindsight.csv and the review's cache (see sources_of_the_hours_needed)")
if FETCH_ONLY:
    say("fetch only: done", json.dumps(DATA_REC)[:400])
    sys.exit(0)

# ---- the AORC series, the daily table and the saved networks (as 16_aorc_rain_to_midnight.py builds them) -----------------------------------------
def wall_clock(t_utc, zone):
    return t_utc.tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)


def rain_by_wall_clock_day(stamp, x, zone):
    """The route of 16_aorc_rain_to_midnight.py: every hourly value goes to the local calendar day in which its hour ENDS (an hour that ends at exactly 24:00 belongs to the day that ends then)."""
    return pd.Series(x, index=(wall_clock(stamp, zone) - pd.Timedelta(minutes=1)).normalize()).groupby(level=0).sum()


A, RAIN = {}, {}
for g in ids:
    stamp, x = AO.load_series(g)
    A[g] = pd.Series(x, index=stamp)
    RAIN[g] = rain_by_wall_clock_day(stamp, x, ZONE_OF[g]).reindex(DAYS)
    assert RAIN[g].notna().all() and RAIN[g].index.equals(D[g].index)
D2 = {g: D[g].assign(aorc_to_midnight=RAIN[g].values) for g in ids}
COL = D2[ids[0]].columns.get_loc("aorc_to_midnight")


def load_run(name, seed):
    z = torch.load(F.RES / "models" / f"lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])


JSON16 = json.loads((F.RES / "aorc_rain_to_midnight.json").read_text(encoding="utf-8"))     # script 16's record: its seeds and the onsets warned it reports (moved up from the line after 'reproduced:')
SEEDS16 = [int(s) for s in JSON16["seeds"]]
MODELS = {n: int(JSON16["summary_test"]["sets"][n]["l1"]["onset"]["hits"]) for n in ("wqa", "wqa_rep")}   # the saved sets and the onsets warned the results file reports for them (checked below)
RUNS = {n: [load_run(n, s) for s in SEEDS16] for n in MODELS}
N_NETS = len(SEEDS16)
NETS_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(N_NETS, str(N_NETS))        # the network count in words, as script 16 writes it


def stored_preds(name):
    """A prediction set of the REAL results folder (flood.load_preds would read the smoke test's folder in quick mode)."""
    z = np.load(F.RES / "preds" / f"{name}.npz", allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"])
    return {str(b): pd.DataFrame({"obs": z["obs"][i], **{f"l{k}": z["pred"][i, :, j] for j, k in enumerate(F.LEADS)}}, index=idx) for i, b in enumerate(z["basins"])}


def forecasts(name, tables):
    """(seed-averaged prediction frames, one frame set per seed): the saved networks run on the given daily tables."""
    ps = [r.predict(tables, ids, B) for r in RUNS[name]]
    return F.ensemble(ps), ps


tz_test = (pd.Timestamp(TEST[0]), pd.Timestamp(TEST[1]))


def hits_by_basin(P, trig=1.0):
    """Onsets warned 1 day ahead per basin, own counts: target day in the test years, the river at or below the level on the issue day, above it on the target day, forecast above trig x level."""
    out = {}
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(1) <= thr[g]).values
        m = (p.index >= tz_test[0]) & (p.index <= tz_test[1]) & p.obs.notna().values & p.l1.notna().values
        out[g] = int((m & was_low & (p.obs > thr[g]).values & (p.l1 > trig * thr[g]).values).sum())
    return out


def warned_flags(P, trig=1.0):
    """Array over the ONSETS rows: was the onset warned (forecast for the target day above trig x level)?"""
    return np.array([bool(P[g].l1.loc[t] > trig * thr[g]) for g, t in zip(ONSETS.gauge_id, ONSETS.target)])


# ---- the stored forecasts and what this script must reproduce -------------------------------------------------------------------------------------
STORED = {n: stored_preds(f"lstm_{n}") for n in MODELS}
STORED_WQ = stored_preds("lstm_wq")
REPRO = {}
BASE = {}                                                    # the AORC run, reproduced from the saved networks and the daily table built here
for n in MODELS:
    BASE[n], seeds_ = forecasts(n, D2)
    mx = 0.0
    for s, p in enumerate(seeds_):
        saved = stored_preds(f"lstm_{n}_s{s}")
        for g in ids:
            a, b = saved[g][["l1", "l2", "l3"]].values, p[g][["l1", "l2", "l3"]].values
            assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True), f"{n} seed {s} {g}: the saved network does not reproduce its stored forecasts"
            mx = max(mx, float(np.nanmax(np.abs(a - b))))
    h_new, h_old = sum(hits_by_basin(BASE[n]).values()), sum(hits_by_basin(STORED[n]).values())
    assert h_new == h_old == MODELS[n], f"{n}: {h_new} onsets warned from the rerun, {h_old} from the stored forecasts, {MODELS[n]} expected"
    REPRO[n] = dict(onsets_warned_rerun=h_new, onsets_warned_stored=h_old, max_abs_difference_to_the_stored_single_seed_forecasts=mx)
HITS_WQ = hits_by_basin(STORED_WQ)
assert sum(HITS_WQ.values()) == int(JSON16["summary_test"]["sets"]["wq"]["l1"]["onset"]["hits"]), sum(HITS_WQ.values())
say("reproduced:", json.dumps(REPRO))
TRIGGER_VAL = float(JSON16["verdicts"]["sets"]["wqa"]["l1"]["chosen_on_validation_onsets"]["trigger"])        # the warning level chosen on the validation years for the onsets (for AORC wqa)


# ---- the Stage IV rain of each onset's issue day ---------------------------------------------------------------------------------------------------
def day_record(g, day, k=1.0):
    """The wall-clock day `day` of basin g, hour by hour: AORC, Stage IV, and Stage IV with the hours the archive lacks replaced by AORC. None if the hours are not in the cache at all."""
    lab = day_labels(day, ZONE_OF[g])
    if not all(t in S4.index for t in lab):
        return None
    a = A[g].reindex(lab).values
    s = S4.loc[lab, g].values.astype(float)
    fb = np.isnan(s)
    comb = np.where(fb, a, s)
    assert abs(a.sum() - RAIN[g].loc[day]) < 1e-9, f"{g} {day.date()}: the hours of the window do not sum to the daily AORC value of the table"
    return dict(aorc=float(a.sum()), s4=float(comb.sum()), n_hours=len(lab), n_fallback=int(fb.sum()), last_s4=float(comb[-1]), last2_s4=float(comb[-2:].sum()),
                last_aorc=float(a[-1]), last2_aorc=float(a[-2:].sum()), s4_hours=comb)


REC = {}                                                     # (gauge, issue day) -> day_record, for every onset whose hours are cached
for g, d in zip(ONSETS.gauge_id, ONSETS.issue):
    r = day_record(g, d)
    if r is not None:
        REC[(g, d)] = r
ONSETS["covered"] = [(g, d) in REC for g, d in zip(ONSETS.gauge_id, ONSETS.issue)]
if QUICK:
    ONSETS["covered"] &= ONSETS.issue.isin(issue_days).values
COV = ONSETS.covered.values
COV_KEYS = [(g, d) for g, d in zip(ONSETS.gauge_id[COV], ONSETS.issue[COV])]            # the covered onsets, in the order of the ONSETS rows
say(f"{int(COV.sum())} of {len(ONSETS)} onsets have their Stage IV hours ({ONSETS[COV].issue.nunique()} issue days)")
assert COV.sum() > 0, "no onset has Stage IV hours: nothing to test"
K = sum(REC[k]["aorc"] for k in COV_KEYS) / sum(REC[k]["s4"] for k in COV_KEYS)
FALLBACK = dict(basin_days=len(COV_KEYS), hours_total=int(sum(REC[k]["n_hours"] for k in COV_KEYS)), hours_replaced_by_aorc=int(sum(REC[k]["n_fallback"] for k in COV_KEYS)),
                basin_days_with_a_replaced_hour=int(sum(REC[k]["n_fallback"] > 0 for k in COV_KEYS)))


def tables_with(values_by_basin_day):
    """Daily tables with aorc_to_midnight replaced on the given (gauge, day) -> value; everything else as D2."""
    tabs = {g: D2[g].copy() for g in ids}
    for (g, d), v in values_by_basin_day.items():
        tabs[g].iloc[DAYS.get_loc(d), COL] = v
    return tabs


def variant_values(kind):
    """(gauge, issue day) -> the replacement of aorc_to_midnight for the covered onsets, by the kind of variant."""
    vals = {}
    for k in COV_KEYS:
        r = REC[k]
        vals[k] = dict(s4=r["s4"], s4_k=K * r["s4"], s4_no_last_hour=r["s4"] - r["last_s4"], s4_no_last_two_hours=r["s4"] - r["last2_s4"],
                       aorc_no_last_hour=r["aorc"] - r["last_aorc"], aorc_no_last_two_hours=r["aorc"] - r["last2_aorc"])[kind]
    return vals


VARIANTS = {"stage4": "Stage IV on the issue day (raw)", "stage4_k": f"Stage IV times {K:.2f} (the level of AORC on those days)",
            "stage4_no_last_hour": "Stage IV, last hour of the issue day (23:00-24:00) taken out", "stage4_no_last_two_hours": "Stage IV, last two hours (22:00-24:00) taken out",
            "aorc_no_last_hour": "AORC, last hour taken out (the delay alone, AORC's own rain)", "aorc_no_last_two_hours": "AORC, last two hours taken out"}
VALUE_KEY = {"stage4": "s4", "stage4_k": "s4_k", "stage4_no_last_hour": "s4_no_last_hour", "stage4_no_last_two_hours": "s4_no_last_two_hours", "aorc_no_last_hour": "aorc_no_last_hour",
             "aorc_no_last_two_hours": "aorc_no_last_two_hours"}
RES_BY = {n: {} for n in MODELS}                             # model -> variant -> dict(P, seeds)
for n in MODELS:
    for kind in VARIANTS:
        P, seeds_ = forecasts(n, tables_with(variant_values(VALUE_KEY[kind])))
        RES_BY[n][kind] = dict(P=P, seeds=seeds_)
        say(f"{n}: {kind}: {int(warned_flags(P)[COV].sum())} of {int(COV.sum())} covered onsets warned")

# Stage IV on the day before as well (when the hours of day t-1 are in the cache for the covered onsets)
RECB = {}
for g, d in COV_KEYS:
    r = day_record(g, d - pd.Timedelta(days=1))
    if r is not None:
        RECB[(g, d)] = r
DAYBEFORE_OK = len(RECB) == int(COV.sum())
if DAYBEFORE_OK:
    for n in MODELS:
        vals = {}
        for (g, d) in COV_KEYS:
            vals[(g, d)] = REC[(g, d)]["s4"]
            vals[(g, d - pd.Timedelta(days=1))] = RECB[(g, d)]["s4"]
        P, seeds_ = forecasts(n, tables_with(vals))
        RES_BY[n]["stage4_and_day_before"] = dict(P=P, seeds=seeds_)
        say(f"{n}: stage4_and_day_before: {int(warned_flags(P)[COV].sum())} of {int(COV.sum())} covered onsets warned")
VARIANTS["stage4_and_day_before"] = "Stage IV on the issue day and on the day before"

# ---- how much does the count depend on the rain of the issue day? Controls with the same networks and the same tables, only the rain of the issue day changed ---------------------------------
from scipy.stats import spearmanr
OBS_ON = np.array([float(D[g].q_mm.loc[t]) for g, t in zip(ONSETS.gauge_id, ONSETS.target)])         # the observed flow on the target day (the onset day) of every onset


def fit_on(P):
    """How the lead-1 forecasts fit the observed flow on the onset days of the covered onsets: mean absolute error (mm/day), Spearman correlation over the onsets, forecast total over observed total."""
    f = np.array([float(P[g].l1.loc[t]) for g, t in zip(ONSETS.gauge_id, ONSETS.target)])[COV]
    o = OBS_ON[COV]
    return dict(mae_mm_day=float(np.mean(np.abs(f - o))), spearman=float(spearmanr(f, o)[0]), forecast_over_observed=float(f.sum() / o.sum()))


rng = np.random.default_rng(0)
by_basin = {g: [(d, REC[(g, d)]["s4"]) for gg, d in COV_KEYS if gg == g] for g in ids}
test_days = pd.date_range(TEST[0], TEST[1])
all_issue = {g: sorted(set(ONSETS[ONSETS.gauge_id == g].issue)) for g in ids}
ordinary = {g: [d for d in test_days if all(abs((d - i_).days) > 2 for i_ in all_issue[g])] for g in ids}      # days of the test years more than 2 days from the issue day of any onset of the basin
CONTROLS = {"another_onset_stage4": "the Stage IV rain of the issue day of another onset of the same basin (a random permutation among the onsets of the basin)",
            "ordinary_day_aorc": "the AORC rain of a random ordinary day of the same basin (a day of the test years more than 2 days from the issue day of any onset of the basin)",
            "no_rain": "no rain on the issue day (0 mm)"}
CTRL = {n: {k: dict(counts=[], fits=[]) for k in CONTROLS} for n in MODELS}
for draw in range(N_PLACEBO):
    vals = {"another_onset_stage4": {}, "ordinary_day_aorc": {}, "no_rain": {}}
    for g in ids:
        items = by_basin[g]
        for (d, _v) in items:
            vals["ordinary_day_aorc"][(g, d)] = float(RAIN[g].loc[ordinary[g][int(rng.integers(0, len(ordinary[g])))]])
            vals["no_rain"][(g, d)] = 0.0
        if len(items) < 2:
            continue
        perm = rng.permutation(len(items))
        while (perm == np.arange(len(items))).any():
            perm = rng.permutation(len(items))
        for (d, _v), j_ in zip(items, perm):
            vals["another_onset_stage4"][(g, d)] = items[j_][1]
    for k, v_ in vals.items():
        if k == "no_rain" and draw > 0:
            continue                                                                       # nothing is random in the no-rain control: once
        for n in MODELS:
            P, _ = forecasts(n, tables_with(v_))
            CTRL[n][k]["counts"].append(int(warned_flags(P)[COV].sum()))
            CTRL[n][k]["fits"].append(fit_on(P))
say("controls:", json.dumps({n: {k: v_["counts"] for k, v_ in CTRL[n].items()} for n in MODELS}))
PLACEBO = {n: CTRL[n]["another_onset_stage4"]["counts"] for n in MODELS}

# Stage IV on all issue days of all basins: the population for the false warnings
ALLDAYS = {}
for d in issue_days:
    for g in ids:
        r = day_record(g, d)
        if r is not None:
            ALLDAYS[(g, d)] = r["s4"]
pop_cover = {g: [d for (gg, d) in ALLDAYS if gg == g] for g in ids}
ALL, ALL_K = {}, {}
for n in MODELS:
    P, _ = forecasts(n, tables_with(ALLDAYS))
    ALL[n] = P
    P, _ = forecasts(n, tables_with({k: K * v for k, v in ALLDAYS.items()}))
    ALL_K[n] = P


def day_population(P, days_by_basin, trig=1.0):
    """Warnings 1 day ahead issued on the given issue days of each basin when the river was at or below the level: hits (onsets warned), false warnings, misses, correct negatives."""
    c = dict(hits=0, misses=0, false_warnings=0, correct_negatives=0)
    for g in ids:
        p = P[g]
        for d in days_by_basin[g]:
            t = d + pd.Timedelta(days=1)
            if not (tz_test[0] <= t <= tz_test[1]) or not np.isfinite(p.l1.loc[t]) or not (p.obs.loc[d] <= thr[g]):
                continue
            w, h = p.l1.loc[t] > trig * thr[g], p.obs.loc[t] > thr[g]
            c["hits" if (w and h) else "false_warnings" if w else "misses" if h else "correct_negatives"] += 1
    c["far"] = c["false_warnings"] / (c["hits"] + c["false_warnings"]) if (c["hits"] + c["false_warnings"]) else None
    c["pod"] = c["hits"] / (c["hits"] + c["misses"]) if (c["hits"] + c["misses"]) else None
    return c


POP = {n: dict(aorc=day_population(BASE[n], pop_cover), stage4=day_population(ALL[n], pop_cover), stage4_times_the_level_of_aorc=day_population(ALL_K[n], pop_cover)) for n in MODELS}

# ---- the numbers -------------------------------------------------------------------------------------------------------------------------------------------
cov_idx = np.flatnonzero(COV)
hitA = {n: warned_flags(BASE[n]) for n in MODELS}
hitR = warned_flags(STORED_WQ)
issue_codes = pd.factorize(ONSETS.issue)[0]
boot_rng = np.random.default_rng(1)


def survive(a, r, s):
    return (s - r) / (a - r) if a > r else float("nan")


def pct(v):
    """2.5 and 97.5 percentiles of the draws that are defined (None, None when none is)."""
    v = v[np.isfinite(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [None, None]


def intervals(a, r, s, codes):
    """95% intervals (percentiles of 2000 draws) of the share surviving and of the difference S - A (as a share of the onsets), drawing onsets and drawing issue days; a, r, s are 0/1 arrays over the covered onsets."""
    n = len(a)
    out = {}
    idx = boot_rng.integers(0, n, (N_BOOT, n))
    A_, R_, S_ = a[idx].sum(1), r[idx].sum(1), s[idx].sum(1)
    sv = np.where(A_ > R_, (S_ - R_) / np.maximum(A_ - R_, 1), np.nan)
    out["over_onsets"] = dict(share_surviving=pct(sv), stage4_minus_aorc_share_of_onsets=[float(x) for x in np.percentile((S_ - A_) / n, [2.5, 97.5])])
    days_ = np.unique(codes)
    per = {c: np.flatnonzero(codes == c) for c in days_}
    sa, sr, ss, cnt = (np.array([x[per[c]].sum() for c in days_]) for x in (a, r, s, np.ones(n)))
    di = boot_rng.integers(0, len(days_), (N_BOOT, len(days_)))
    A_, R_, S_, N_ = sa[di].sum(1), sr[di].sum(1), ss[di].sum(1), cnt[di].sum(1)
    sv = np.where(A_ > R_, (S_ - R_) / np.maximum(A_ - R_, 1), np.nan)
    out["over_issue_days"] = dict(share_surviving=pct(sv), stage4_minus_aorc_share_of_onsets=[float(x) for x in np.percentile((S_ - A_) / N_, [2.5, 97.5])])
    return out


codes_cov = issue_codes[COV]
RESULT = {}
for n in MODELS:
    a, r = hitA[n][COV].astype(int), hitR[COV].astype(int)
    block = dict(onsets_tested=int(COV.sum()), aorc=int(a.sum()), rain_table_alone=int(r.sum()), variants={})
    for kind, v in RES_BY[n].items():
        P = v["P"]
        s = warned_flags(P)[COV].astype(int)
        seed_hits = [int(warned_flags(p_)[COV].sum()) for p_ in v["seeds"]]
        by_g = {g: dict(aorc=int(a[ONSETS.gauge_id[COV].values == g].sum()), rain_table_alone=int(r[ONSETS.gauge_id[COV].values == g].sum()), stage4=int(s[ONSETS.gauge_id[COV].values == g].sum())) for g in ids}
        for g in ids:
            by_g[g]["share_surviving"] = survive(by_g[g]["aorc"], by_g[g]["rain_table_alone"], by_g[g]["stage4"]) if by_g[g]["aorc"] > by_g[g]["rain_table_alone"] else None
            by_g[g]["onsets"] = int((ONSETS.gauge_id[COV].values == g).sum())
        trig = TRIGGER_VAL
        s_val = warned_flags(P, trig)[COV].astype(int)
        a_val, r_val = warned_flags(BASE[n], trig)[COV].astype(int), warned_flags(STORED_WQ, trig)[COV].astype(int)
        block["variants"][kind] = dict(description=VARIANTS[kind], onsets_warned=int(s.sum()), share_of_the_onsets=float(s.mean()), single_seed_onsets_warned=seed_hits,
                                       share_of_the_aorc_effect_surviving=survive(a.sum(), r.sum(), s.sum()), differs_from_aorc_by_onsets=int(s.sum() - a.sum()),
                                       intervals=intervals(a, r, s, codes_cov), per_basin=by_g,
                                       onsets_warned_at_the_level_chosen_on_validation=dict(trigger=trig, stage4=int(s_val.sum()), aorc=int(a_val.sum()), rain_table_alone=int(r_val.sum()),
                                                                                          share_surviving=survive(a_val.sum(), r_val.sum(), s_val.sum())),
                                       onsets_lost_and_gained=dict(warned_with_aorc_not_with_stage4=int(((a == 1) & (s == 0)).sum()), warned_with_stage4_not_with_aorc=int(((a == 0) & (s == 1)).sum()),
                                                                   warned_by_both=int(((a == 1) & (s == 1)).sum()), warned_by_neither=int(((a == 0) & (s == 0)).sum())))
    block["reference_fits_on_the_onset_days"] = dict(rain_table_alone=fit_on(STORED_WQ), aorc=fit_on(BASE[n]))
    for kind_, v_ in RES_BY[n].items():
        block["variants"][kind_]["fit_on_the_onset_days"] = fit_on(v_["P"])
    nofb = np.array([REC[k]["n_fallback"] == 0 for k in COV_KEYS])
    s0 = warned_flags(RES_BY[n]["stage4"]["P"])[COV].astype(int)
    block["onsets_without_any_replaced_hour"] = dict(onsets=int(nofb.sum()), aorc=int(a[nofb].sum()), rain_table_alone=int(r[nofb].sum()), stage4=int(s0[nofb].sum()),
                                                      share_surviving=survive(a[nofb].sum(), r[nofb].sum(), s0[nofb].sum()))
    block["controls_on_the_rain_of_the_issue_day"] = {k: dict(what=CONTROLS[k], draws=len(CTRL[n][k]["counts"]), onsets_warned=CTRL[n][k]["counts"], mean=float(np.mean(CTRL[n][k]["counts"])),
                                                              lowest=int(min(CTRL[n][k]["counts"])), highest=int(max(CTRL[n][k]["counts"])), share_of_the_aorc_effect_mean=survive(a.sum(), r.sum(), np.mean(CTRL[n][k]["counts"])),
                                                              fit_on_the_onset_days={m_: float(np.mean([f_[m_] for f_ in CTRL[n][k]["fits"]])) for m_ in ("mae_mm_day", "spearman", "forecast_over_observed")}) for k in CONTROLS}
    block["issue_days_of_all_basins"] = dict(what="the warnings issued on the issue days of the covered onsets, in all 12 basins, when the river was at or below the level (Stage IV substituted in every basin on those days)",
                                             aorc=POP[n]["aorc"], stage4=POP[n]["stage4"], stage4_times_the_level_of_aorc=POP[n]["stage4_times_the_level_of_aorc"])
    RESULT[n] = block

# the agreement of the two rain products on the issue days (before midnight)
pairs = [(REC[(g, d)]["aorc"], REC[(g, d)]["s4"], REC[(g, d)]["last_aorc"], REC[(g, d)]["last_s4"], g) for g, d in zip(ONSETS.gauge_id[COV], ONSETS.issue[COV])]
pa, ps_, la, ls, pg = (np.array([p[i] for p in pairs]) for i in range(5))
AGREE = dict(basin_days=int(len(pa)), correlation=float(np.corrcoef(pa, ps_)[0, 1]), mean_ratio_stage4_over_aorc=float(ps_.sum() / pa.sum()), median_ratio=float(np.median(ps_[pa > 1] / pa[pa > 1])),
             mean_mm_aorc=float(pa.mean()), mean_mm_stage4=float(ps_.mean()), share_of_the_day_in_the_last_hour_aorc=float(la.sum() / pa.sum()), share_of_the_day_in_the_last_hour_stage4=float(ls.sum() / ps_.sum()),
             per_basin={g: dict(correlation=float(np.corrcoef(pa[pg == g], ps_[pg == g])[0, 1]) if (pg == g).sum() > 2 else None, ratio=float(ps_[pg == g].sum() / pa[pg == g].sum())) for g in ids if (pg == g).any()})

for n in MODELS:                                                                                  # what distinguishes the onsets that are lost (warned with AORC, not with Stage IV) from those that are kept
    a_, s_ = hitA[n][COV], warned_flags(RES_BY[n]["stage4"]["P"])[COV]
    ratio_ = np.where(pa > 0, ps_ / np.maximum(pa, 1e-9), np.nan)
    ex_ = RESULT[n]["variants"]["stage4"]["onsets_lost_and_gained"]
    for key_, m_ in (("warned_with_aorc_not_with_stage4", a_ & ~s_), ("warned_by_both", a_ & s_), ("warned_with_stage4_not_with_aorc", ~a_ & s_)):
        ex_[f"mean_stage4_over_aorc_rain_of_the_onsets_{key_}"] = float(np.nanmean(ratio_[m_])) if m_.any() and np.isfinite(ratio_[m_]).any() else None

# ---- the sentence -------------------------------------------------------------------------------------------------------------------------------------------
v1 = RESULT["wqa"]["variants"]["stage4"]
n_t, a_t, r_t = RESULT["wqa"]["onsets_tested"], RESULT["wqa"]["aorc"], RESULT["wqa"]["rain_table_alone"]
pc = lambda x: f"{100 * x:.0f}%" if x is not None and x == x else "no value"
iv_o, iv_d = v1["intervals"]["over_onsets"]["share_surviving"], v1["intervals"]["over_issue_days"]["share_surviving"]
vk, vl, vl2 = (RESULT["wqa"]["variants"][k] for k in ("stage4_k", "stage4_no_last_hour", "stage4_no_last_two_hours"))
outcome = (f"With Stage IV rain (radar and gauge, the product a service at midnight would read) in place of AORC on the issue day of {'each of the' if n_t == len(ONSETS) else 'each of the tested'} {n_t} onsets"
           f"{'' if n_t == len(ONSETS) else ' (a partial sample: the other onsets were not tested)'}, the {NETS_WORD} saved lstm_wqa networks warn of {v1['onsets_warned']} ({pc(v1['share_of_the_onsets'])}) against {a_t} with AORC "
           f"({pc(a_t / n_t)}) and {r_t} with the rain table alone ({pc(r_t / n_t)}): {pc(v1['share_of_the_aorc_effect_surviving'])} of the AORC effect survives (95% interval {pc(iv_o[0])} to {pc(iv_o[1])} drawing "
           f"onsets, {pc(iv_d[0])} to {pc(iv_d[1])} drawing issue days). With the level of AORC restored (Stage IV times {K:.2f}) {vk['onsets_warned']} are warned; with the last hour of the issue day taken out "
           f"{vl['onsets_warned']}, with the last two hours {vl2['onsets_warned']}.")
ctl_ = RESULT["wqa"]["controls_on_the_rain_of_the_issue_day"]
ref_ = RESULT["wqa"]["reference_fits_on_the_onset_days"]
pop_ = RESULT["wqa"]["issue_days_of_all_basins"]
outcome += (f" On the issue days of all 12 basins (the warnings issued when the river was low; Stage IV in every basin on those days) Stage IV gives {pop_['stage4']['hits']} hits and {pop_['stage4']['false_warnings']} false warnings "
            f"({pc(pop_['stage4']['far'])} of the warnings) against {pop_['aorc']['hits']} hits and {pop_['aorc']['false_warnings']} false warnings ({pc(pop_['aorc']['far'])}) with AORC, and with the level of AORC restored "
            f"{pop_['stage4_times_the_level_of_aorc']['hits']} and {pop_['stage4_times_the_level_of_aorc']['false_warnings']} ({pc(pop_['stage4_times_the_level_of_aorc']['far'])}).")
outcome += (f" The count is more sensitive to whether the issue day was a heavy-rain day than to its exact rain: with the Stage IV rain of another onset of the same basin in its place the networks still warn of "
            f"{ctl_['another_onset_stage4']['mean']:.0f} ({ctl_['another_onset_stage4']['lowest']} to {ctl_['another_onset_stage4']['highest']} in {ctl_['another_onset_stage4']['draws']} draws), with the AORC rain of an ordinary day "
            f"{ctl_['ordinary_day_aorc']['mean']:.0f}, with no rain {ctl_['no_rain']['mean']:.0f}. The mean absolute error of the forecast on the onset days, which does depend on the exact rain, is {ref_['aorc']['mae_mm_day']:.2f} mm/day with AORC, "
            f"{v1['fit_on_the_onset_days']['mae_mm_day']:.2f} with Stage IV, {ctl_['another_onset_stage4']['fit_on_the_onset_days']['mae_mm_day']:.2f} with another onset's Stage IV rain, {ctl_['ordinary_day_aorc']['fit_on_the_onset_days']['mae_mm_day']:.2f} "
            f"with an ordinary day's rain and {ref_['rain_table_alone']['mae_mm_day']:.2f} with the rain table alone.")
RESULT_ALL = dict(
    question="If Stage IV, a real-time rain product, replaces AORC on the issue day of each test-year onset, how many onsets do the saved wqa networks still warn of 1 day ahead?",
    quick_mode=QUICK, outcome=outcome, honesty_points=[
        "the networks were trained on AORC (hours before 2002 disaggregated daily totals); Stage IV is substituted at forecast time only, and nothing is said about training on Stage IV",
        "Stage IV covers 2002 onward, all test years; the issue days are the days before the onsets, so the false warnings are tested only on the issue days of all basins (issue_days_of_all_basins), not on days without any onset",
        "the one-hour delay of Stage IV is not simulated, except by the variants that take the last hour or the last two hours of the issue day out",
        "the count of onsets warned is a coarse test of whether Stage IV carries the same information as AORC: the controls show that it falls little when the rain of another onset day of the same basin is put in its place (onset days are all heavy-rain days); "
        "the fit of the forecasts to the observed flow on the onset days (fit_on_the_onset_days) and the controls are what separate 'heavy rain' from 'the rain of that day'",
        "the Iowa Mesonet archive holds first postings or later versions of each hour (not established); hours the archive lacks, or in which no cell of the basin has radar coverage, are replaced by the AORC hour (fallback)"],
    code=dict(file=f"code/{SCRIPT}", sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest()),
    data=DATA_REC, fallback=FALLBACK, k_level_aorc_over_stage4=K, trigger_chosen_on_validation_for_aorc_wqa=TRIGGER_VAL,
    reproduction=REPRO, onsets=dict(total=int(len(ONSETS)), tested=int(COV.sum()), issue_days_tested=int(ONSETS[COV].issue.nunique()), issue_days_total=int(ONSETS.issue.nunique())),
    agreement_of_the_two_products_on_the_issue_days=AGREE, models=RESULT,
    day_before_variant=("run" if DAYBEFORE_OK else "not run: the Stage IV hours of the day before the issue day were not all fetched (use --day-before)"),
    runtime_s=round(time.time() - t_start, 1))
PER_ONSET = []
for (g, basin, t_, d_), c_ in zip(ONSETS[["gauge_id", "basin", "target", "issue"]].itertuples(index=False, name=None), COV):
    row_ = dict(gauge=g, basin=basin, issue_day=str(d_.date()), target_day=str(t_.date()), tested=bool(c_), high_flow_level_mm_day=round(thr[g], 4),
                forecast_mm_rain_table=round(float(STORED_WQ[g].l1.loc[t_]), 4), forecast_mm_aorc=round(float(BASE["wqa"][g].l1.loc[t_]), 4))
    if c_:
        r_ = REC[(g, d_)]
        row_.update(aorc_mm=round(r_["aorc"], 3), stage4_mm=round(r_["s4"], 3), stage4_hours_replaced_by_aorc=r_["n_fallback"], forecast_mm_stage4=round(float(RES_BY["wqa"]["stage4"]["P"][g].l1.loc[t_]), 4))
        row_.update(warned_rain_table=bool(row_["forecast_mm_rain_table"] > thr[g]), warned_aorc=bool(row_["forecast_mm_aorc"] > thr[g]), warned_stage4=bool(row_["forecast_mm_stage4"] > thr[g]))
    PER_ONSET.append(row_)
RESULT_ALL["per_onset"] = PER_ONSET


def clean(o):
    """Strict JSON: NaN and infinity become null (a share whose denominator is zero)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(F.RES_OUT / f"{OUT}.json").write_text(json.dumps(clean(RESULT_ALL), indent=1, default=str))
say(outcome)

# ---- the figure ----------------------------------------------------------------------------------------------------------------------------------------------
use_style()
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(16.5, 6.4))
gs = GridSpec(1, 3, width_ratios=[1.45, 1.0, 0.85], wspace=0.42, left=0.17, right=0.985, top=0.83, bottom=0.19)
ax, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
VW = RESULT["wqa"]["variants"]
share_of = lambda v: v["share_of_the_aorc_effect_surviving"]
FIT = lambda v: v["fit_on_the_onset_days"]["mae_mm_day"]
bars = [("rain table alone", r_t, None, MUTED, 1.0, None, ref_["rain_table_alone"]["mae_mm_day"]), ("AORC (as saved)", a_t, None, BLUE, 1.0, None, ref_["aorc"]["mae_mm_day"]),
        ("Stage IV on the issue day", v1["onsets_warned"], v1, ORANGE, 1.0, share_of(v1), FIT(v1)), (f"Stage IV x {K:.2f} (the level of AORC)", vk["onsets_warned"], vk, ORANGE, 1.0, share_of(vk), FIT(vk)),
        ("Stage IV, last hour out", vl["onsets_warned"], vl, ORANGE, 1.0, share_of(vl), FIT(vl)), ("Stage IV, last 2 hours out", vl2["onsets_warned"], vl2, ORANGE, 1.0, share_of(vl2), FIT(vl2))]
if "stage4_and_day_before" in VW:
    bars.append(("Stage IV on day t and t-1", VW["stage4_and_day_before"]["onsets_warned"], VW["stage4_and_day_before"], ORANGE, 1.0, share_of(VW["stage4_and_day_before"]), FIT(VW["stage4_and_day_before"])))
bars += [("AORC, last hour out", VW["aorc_no_last_hour"]["onsets_warned"], None, BLUE, 0.5, None, FIT(VW["aorc_no_last_hour"])), ("AORC, last 2 hours out", VW["aorc_no_last_two_hours"]["onsets_warned"], None, BLUE, 0.5, None, FIT(VW["aorc_no_last_two_hours"]))]
for k_, lab_ in (("another_onset_stage4", "control: Stage IV rain of another onset day"), ("ordinary_day_aorc", "control: AORC rain of an ordinary day"), ("no_rain", "control: no rain on the issue day")):
    bars.append((lab_, ctl_[k_]["mean"], (ctl_[k_]["lowest"], ctl_[k_]["highest"]), "#b9b7ae", 1.0, None, ctl_[k_]["fit_on_the_onset_days"]["mae_mm_day"]))
for i_, (lab, val, v, col, alpha, sh, mae) in enumerate(bars):
    ax.barh(i_, val, color=col, alpha=alpha, height=0.5)
    txt = f"{val:.0f}  ({100 * val / n_t:.0f}%)" + (f"   keeps {pc(sh)}" if sh is not None and sh == sh else "") + f"   MAE {mae:.2f}"
    right = val
    if isinstance(v, dict):
        lo_ = v["intervals"]["over_issue_days"]["stage4_minus_aorc_share_of_onsets"]
        ax.plot([a_t + lo_[0] * n_t, a_t + lo_[1] * n_t], [i_, i_], color=INK, lw=1.2, solid_capstyle="butt")
        right = max(val, a_t + lo_[1] * n_t)
    elif isinstance(v, tuple):
        ax.plot([v[0], v[1]], [i_, i_], color=INK, lw=1.2, solid_capstyle="butt")
        right = max(val, v[1])
    ax.text(right + n_t * 0.01, i_, txt, va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, n_t * 0.85)
ax.set_xlabel(f"onsets warned 1 day ahead, of {n_t} (the {NETS_WORD} saved lstm_wqa networks, forecasts averaged)")
ax.set_title("Onsets warned with each rain input", loc="left")
sa, ss_ = hitA["wqa"][COV], warned_flags(RES_BY["wqa"]["stage4"]["P"])[COV]
kept, lost, gain, none_ = (sa & ss_), (sa & ~ss_), (~sa & ss_), (~sa & ~ss_)
mx = float(np.percentile(np.concatenate([pa, ps_]), 99)) * 1.05
for m, col, lab in ((none_, "#b9b7ae", "warned by neither"), (kept, BLUE, "warned with both"), (lost, YELLOW, "warned with AORC only"), (gain, AQUA, "warned with Stage IV only")):
    ax2.scatter(pa[m], ps_[m], s=15, color=col, alpha=0.8, linewidths=0, label=f"{lab} ({int(m.sum())})")
ax2.plot([0, mx], [0, mx], color=MUTED, lw=1)
ax2.set_xlim(0, mx)
ax2.set_ylim(0, mx)
ax2.set_xlabel("AORC rain of the issue day, 00:00-24:00 (mm)")
ax2.set_ylabel("Stage IV rain of the same hours (mm)")
ax2.set_title(f"The two products (r = {AGREE['correlation']:.2f}; Stage IV {AGREE['mean_ratio_stage4_over_aorc']:.2f} x AORC)", loc="left")
ax2.legend(frameon=False, fontsize=8, loc="upper left", markerscale=1.6)
rows_b = [(g, d_) for g, d_ in v1["per_basin"].items() if d_["share_surviving"] is not None]
rows_b.sort(key=lambda t: t[1]["share_surviving"])
for i_, (g, d_) in enumerate(rows_b):
    ax3.plot([0, d_["share_surviving"]], [i_, i_], color=GRID, lw=2, solid_capstyle="butt", zorder=1)
    ax3.scatter(d_["share_surviving"], i_, s=42, color=ORANGE, zorder=3, linewidths=0)
ax3.axvline(1.0, color=MUTED, lw=1)
ax3.set_yticks(range(len(rows_b)), [f"{F.SHORT[g]} ({d_['aorc'] - d_['rain_table_alone']:+d})" for g, d_ in rows_b], fontsize=8)
ax3.set_xlabel("share of the AORC effect kept with Stage IV")
ax3.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
ax3.set_title("By basin", loc="left")
head = f"Stage IV in place of AORC keeps {pc(share_of(v1))} of the AORC effect on the {n_t} onsets (95% interval {pc(iv_d[0])} to {pc(iv_d[1])}), {pc(share_of(vl))} with the last hour of the day taken out" if n_t == len(ONSETS) else \
    f"Stage IV in place of AORC on the issue day of {n_t} of the {len(ONSETS)} onsets (a partial sample)"
fig.suptitle(head, x=0.01, y=0.985, ha="left", fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.01, 0.905, "E1b real-time test: the networks were trained on AORC; Stage IV replaces AORC on the issue day of each onset at forecast time only (test years 2005-2014, 12 basins, lead 1, warning level 1.0 x the high-flow level).",
         fontsize=9, color=INK2)
fig.text(0.01, 0.025, "'keeps': the share of the AORC effect that survives, (Stage IV - rain table) / (AORC - rain table). Black line: 95% interval of the count, drawing issue days (the onsets of one storm are not independent), laid around the AORC count.\n"
         "'Last hour out': the hour 23:00-24:00 of the issue day is taken out (Stage IV is about one hour behind); the AORC rows with hours out show what the delay alone does to the AORC forecasts.\n"
         "By basin: in brackets, the onsets that AORC adds to the rain table in that basin (a basin without a gain is left out). MAE: mean absolute error (mm/day) of the forecast on the onset days.\n"
         "Controls: the rain of the issue day replaced by other rain (mean of the draws, line: lowest to highest): the count falls little for another onset day's rain, the MAE shows the difference.",
         fontsize=8, color=MUTED, va="bottom")
fig.savefig(F.FIG_OUT / f"{OUT}.png")
say("figure:", rel(F.FIG_OUT / f"{OUT}.png"))
