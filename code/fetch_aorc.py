"""NOAA AORC v1.1 hourly rain on the 1 km grid, reduced to one hourly basin-mean series per study basin (experiment E1b, code/16_aorc_rain_to_midnight.py).

AORC (Analysis Of Record for Calibration, NOAA/NWS) is served as one open Zarr store per year on AWS Open Data (no login):
https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/YYYY.zarr, variable APCP_surface (int16, scale 0.1 kg m-2 = mm per hour, fill -32767, chunks of 144 hours x 128 rows x 256
columns, zstd). This script reads the stores' own metadata first (.zarray, .zattrs, the time, latitude and longitude arrays of each year) and uses what they say, then
 1. rasterises each study basin's outline (CAMELS HCDN_nhru_final_671.shp, as fetch_satellite.py reads it) onto the AORC grid: the cells whose centre lies inside the outline
    (rasterio.features.rasterize); the cells of this grid have equal area to within 1% over the largest basin (1/120 degree cells; the cell width varies with cos(latitude): 0.7% over Leaf River), so the basin mean is the plain
    mean of the cells; MEASURES how many spatial chunks the 12 outlines touch, how many chunk fetches and how many bytes 1980-09-25 .. 2014-10-05 needs, and the speed of the link, and
    falls back to the mean of the cells within 5 km of the outline's centroid only if that would exceed 12 GB or 6 hours (the rule is in the code; the choice is recorded);
 2. fetches every needed chunk (16 threads by default, AORC_THREADS, through the environment's proxy, retries, resume-safe), decodes it with the zstd command-line tool - NEEDED, and not a pip
    package: brew install zstd, or apt install zstd - reduces it AT ONCE to the sum and the
    count of the valid cells of each basin per hour (the raw chunk is never written to disk), and stores those sums per (year, spatial chunk) in data/aorc/parts/ - a part
    is written only when all its time chunks have arrived, so an interrupted run loses at most the units in flight and a restart skips every part that exists;
 3. assembles data/aorc/aorc_<gauge_id>.npz, one file per basin: the UTC time stamps and the basin-mean rain of each hour in mm, over the same span as the cached ERA5
    files of check_day_windows.py (1980-09-25 00:00 to 2014-10-05 23:00 UTC stamps, 298,296 hours). The source has holes (every cell of a tile holds the fill value for a few days here
    and there): an hour in which no cell of a basin has a value is filled with the cached ERA5 hourly rain of the basin and flagged ('filled'); an hour in which only some cells are missing
    keeps the mean of the cells that have a value. It validates the series (no gap on the time axis, no negative value, no unflagged missing hour, the fill value excluded cell by cell) and
    compares them with the other rain products (daily totals by UTC day against IMERG, Daymet, NLDAS and ERA5 with the lag correlations of 10_rain_timing.py; the hourly series against the
    cached hourly ERA5 at lags of -3 to +3 hours, before and after 2002; and against the hourly Stage IV files of four storm days of the test years, to see whether a stamp T is the hour that
    ENDS at T);
 4. writes results/aorc_data.json: what was chosen and why, the store facts, what was downloaded (chunks, bytes, seconds), what was filled, and every check.
Run it as python code/fetch_aorc.py (resume-safe; nothing happens when the cache is complete); code/16_aorc_rain_to_midnight.py calls ensure() and so downloads what is missing.
The download measured 4.5 GB (28,712 chunk fetches, 66 minutes with 16 connections on 2026-10-07; the estimate before it was 4.2 GB and 2.5 hours). Requirements besides the Python packages: the zstd command-line tool, and
a network that reaches noaa-nws-aorc-v1-1-1km.s3.amazonaws.com (and, for the check against the hourly Stage IV files, mesonet.agron.iastate.edu).
data/aorc is the reference version of the data: AWS announced on 2026-04-02 that about 0.04% of the AORC rows were masked by error and that the Zarr files are being regenerated, so a download made after the
regeneration may differ from data/aorc in the hours that were filled from ERA5 and, in principle, elsewhere (results/aorc_data.json, amendments, holds the sha256 of the 12 basin files).
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, RES

BASE = "https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com"
VAR = "APCP_surface"
AORC = Path(os.environ["FLOOD_AORC_DIR"]) if os.environ.get("FLOOD_AORC_DIR") else DATA / "aorc"      # FLOOD_AORC_DIR: a test hook for smoke tests with made-up series (16_aorc_rain_to_midnight.py allows it only with FLOOD_QUICK=1)
PARTS = AORC / "parts"
SHAPES = DATA / "camels_us" / "basin_set_full_res" / "HCDN_nhru_final_671.shp"
RECORD = RES / "aorc_data.json"
ZSTD = shutil.which("zstd") or "/usr/local/bin/zstd"                                                 # the zstd command-line tool decodes the chunks (brew install zstd, or apt install zstd)
YEARS = tuple(range(1980, 2015))
FIRST_STAMP, LAST_STAMP = pd.Timestamp("1980-09-25 00:00"), pd.Timestamp("2014-10-05 23:00")        # hour-ending UTC stamps: the span of the cached ERA5 files (data/hourly)
N_HOURS = int((LAST_STAMP - FIRST_STAMP) / pd.Timedelta(hours=1)) + 1                                # 298,296
LAT0, LON0, STEP, NLAT, NLON = 20.0, -130.0, 1 / 120, 4201, 8401                                     # the nominal grid (checked against the stores' coordinate arrays below)
CH_T, CH_Y, CH_X = 144, 128, 256                                                                     # chunk shape (checked against each year's .zarray)
FILL, SCALE = -32767, 0.1                                                                            # the stores' fill value; scale 0.1 mm per unit (their scale_factor is the float32 0.10000000149)
LIMIT_GB, LIMIT_HOURS = 12.0, 6.0                                                                    # the rule: more than this and the 5 km mean around the centroid is used instead
THREADS = int(os.environ.get("AORC_THREADS", 16))                                                    # the link gives about 0.1 to 0.4 chunks/s per connection (measured 2026-10-07: 3.2 chunks/s with 8 threads at 18:15, 0.67 with 8 at 18:30), so more connections, not faster ones, shorten the download
SPAN_FOR_THE_RECORD = f"{FIRST_STAMP} to {LAST_STAMP} UTC (hour-ending stamps)"


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ids_of_the_study():
    return [r["gauge_id"] for r in json.loads((RES / "data_audit.json").read_text())["selected"]]


# ---- the grid and the basin masks ------------------------------------------------------------------------------------
def outlines(ids):
    """{gauge_id: outline polygon}, read as fetch_satellite.basin_cells() reads them (hru_id = gauge id, an invalid ring repaired with buffer(0))."""
    import shapefile
    from shapely.geometry import shape
    out = {}
    for sr in shapefile.Reader(str(SHAPES)).iterShapeRecords():
        g = f"{int(sr.record['hru_id']):08d}"
        if g in ids:
            poly = shape(sr.shape.__geo_interface__)
            out[g] = poly if poly.is_valid else poly.buffer(0)
    assert set(out) == set(ids), "a basin outline is missing from the shapefile"
    return out


def cell_index(poly, mode):
    """The AORC cells of one basin as index arrays (ilat, ilon): ilat = round((lat - 20) * 120), ilon = round((lon + 130) * 120), cell centres on the nominal 1/120 degree grid.

    mode 'outline': the cells whose centre lies inside the outline (rasterio.features.rasterize, all_touched=False, as geometry_mask(invert=True) in 07_basin_maps.py).
    mode 'disc5km': the cells whose centre lies within 5 km of the outline's centroid (the fallback)."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    x0, y0, x1, y1 = poly.bounds
    pad = 0.06 if mode == "disc5km" else 0.0
    cx, cy = poly.centroid.x, poly.centroid.y
    if mode == "disc5km":
        x0, x1, y0, y1 = cx - pad, cx + pad, cy - pad, cy + pad
    ilat0, ilat1 = int(math.floor((y0 - LAT0) / STEP - 0.5)), int(math.ceil((y1 - LAT0) / STEP + 0.5))
    ilon0, ilon1 = int(math.floor((x0 - LON0) / STEP - 0.5)), int(math.ceil((x1 - LON0) / STEP + 0.5))
    ny, nx = ilat1 - ilat0 + 1, ilon1 - ilon0 + 1
    if mode == "outline":
        tf = Affine(STEP, 0, LON0 + (ilon0 - 0.5) * STEP, 0, -STEP, LAT0 + (ilat1 + 0.5) * STEP)          # north-up raster of the window; row r = ilat1 - r
        m = rasterize([(poly, 1)], out_shape=(ny, nx), transform=tf, fill=0, all_touched=False, dtype="uint8")[::-1].astype(bool)
    else:
        lat, lon = LAT0 + (ilat0 + np.arange(ny)) * STEP, LON0 + (ilon0 + np.arange(nx)) * STEP
        dy, dx = (lat[:, None] - cy) * 110.57, (lon[None, :] - cx) * 111.32 * math.cos(math.radians(cy))
        m = np.hypot(dy, dx) <= 5.0
    ii, jj = np.nonzero(m)
    return ii + ilat0, jj + ilon0


def plan_cells(ids, mode):
    """Per basin the cells and the spatial chunks they lie in, and the checks of the geometry (cell area against the outline's own area, the mask's centroid against the outline's centroid)."""
    from pyproj import Geod
    geod = Geod(ellps="GRS80")
    polys = outlines(ids)
    out = {}
    for g in ids:
        la, lo = cell_index(polys[g], mode)
        assert len(la) > 20, f"{g}: the mask holds only {len(la)} cells"
        lat_c, lon_c = LAT0 + la * STEP, LON0 + lo * STEP
        cell_km2 = float(np.mean(STEP * 110.57 * STEP * 111.32 * np.cos(np.radians(lat_c))))
        area = abs(geod.geometry_area_perimeter(polys[g])[0]) / 1e6
        chunks = sorted({(int(a // CH_Y), int(b // CH_X)) for a, b in zip(la, lo)})
        out[g] = dict(ilat=la, ilon=lo, n_cells=int(len(la)), chunks=chunks, mask_area_km2=float(len(la) * cell_km2), outline_area_km2=float(area),
                      mask_over_outline_area=float(len(la) * cell_km2 / area), centroid_offset_deg=[float(lat_c.mean() - polys[g].centroid.y), float(lon_c.mean() - polys[g].centroid.x)],
                      cell_area_variation_over_the_basin=float(np.ptp(np.cos(np.radians(lat_c))) / np.mean(np.cos(np.radians(lat_c)))),
                      lat_index_range=[int(la.min()), int(la.max())], lon_index_range=[int(lo.min()), int(lo.max())])
    return out


# ---- time axis: which chunks of which year ---------------------------------------------------------------------------
def year_hours(year):
    """The hour indices of a year store that the span needs: (first, last) inclusive, assuming the time axis starts at 00:00 on 1 January (checked against the store's time array)."""
    start = pd.Timestamp(f"{year}-01-01 00:00")
    n = 8784 if year % 4 == 0 else 8760
    a, b = max(FIRST_STAMP, start), min(LAST_STAMP, start + pd.Timedelta(hours=n - 1))
    return int((a - start) / pd.Timedelta(hours=1)), int((b - start) / pd.Timedelta(hours=1)), n


def time_chunks(year):
    a, b, _ = year_hours(year)
    return list(range(a // CH_T, b // CH_T + 1))


# ---- HTTP ------------------------------------------------------------------------------------------------------------
_loc = threading.local()


def session():
    import requests
    if getattr(_loc, "s", None) is None:
        _loc.s = requests.Session()                                                                  # takes the proxy from the environment (HTTPS_PROXY); one connection per thread, kept alive
    return _loc.s


def get(url, tries=8, allow_absent=False):
    """GET with retries and a new connection after a failure. Returns (bytes, status). A 403 or 404 on a chunk means an absent (all-fill) chunk in a Zarr store only if allow_absent, after one more try."""
    import requests
    last = None
    for k in range(tries):
        try:
            r = session().get(url, timeout=120)
            if r.status_code == 200:
                return r.content, 200
            if allow_absent and r.status_code in (403, 404) and k >= 1:
                return b"", r.status_code
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, OSError) as e:
            last = type(e).__name__
        _loc.s = None
        time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"{url}: {last} after {tries} tries")


def zstd_decode(raw):
    try:
        p = subprocess.run([ZSTD, "-d", "-c", "-q"], input=raw, capture_output=True)
    except FileNotFoundError:
        raise SystemExit("the zstd command-line tool is needed to decode the AORC chunks and was not found (brew install zstd, or apt install zstd)")
    if p.returncode:
        raise RuntimeError("zstd could not decode a chunk: " + p.stderr.decode()[:200])
    return p.stdout


# ---- the stores' own metadata ----------------------------------------------------------------------------------------
def store_meta(year):
    """Read a year store's .zarray and .zattrs of the variable and its whole time axis, check them against what this script assumes; cached in data/aorc/meta_<year>.json."""
    f = AORC / f"meta_{year}.json"
    if f.exists():
        return json.loads(f.read_text())
    root = f"{BASE}/{year}.zarr"
    za = json.loads(get(f"{root}/{VAR}/.zarray")[0])
    at = json.loads(get(f"{root}/{VAR}/.zattrs")[0])
    tz, ta = json.loads(get(f"{root}/time/.zarray")[0]), json.loads(get(f"{root}/time/.zattrs")[0])
    # the stores of 1979-2009 and of 2010-2025 are written differently (fill value in .zarray or as _FillValue in .zattrs, compression level, chunk length of the time axis): read what each says
    fill_declared = za["fill_value"] if za["fill_value"] is not None else at.get("_FillValue")
    assert za["dtype"] == "<i2" and za["chunks"] == [CH_T, CH_Y, CH_X] and za["shape"][1:] == [NLAT, NLON] and fill_declared == FILL and za["compressor"]["id"] == "zstd" and za["filters"] is None \
        and za["order"] == "C" and za.get("dimension_separator", ".") == ".", f"{year}: the store is not laid out as assumed: {za} {at}"
    assert at["_ARRAY_DIMENSIONS"] == ["time", "latitude", "longitude"] and at["units"] == "kg/m^2" and abs(at["scale_factor"] - SCALE) < 1e-8 and at["missing_value"] == FILL, f"{year}: attributes: {at}"
    assert tz["dtype"] == "<f8" and ta["units"] == "seconds since 1970-01-01" and tz["shape"] == [za["shape"][0]] and ta["calendar"] == "proleptic_gregorian", f"{year}: time axis: {tz} {ta}"
    n, ch = tz["shape"][0], tz["chunks"][0]
    parts = [zstd_decode(get(f"{root}/time/{c}")[0]) for c in range(math.ceil(n / ch))]
    t = np.frombuffer(b"".join(parts), "<f8")[:n]
    stamp = pd.to_datetime(t, unit="s")
    m = dict(year=year, zarray=za, zattrs_of_the_variable=at, time_zarray=tz, time_zattrs=ta, n_hours=int(n), first_stamp=str(stamp[0]), last_stamp=str(stamp[-1]),
             steps_s=sorted({float(x) for x in np.unique(np.diff(t))}), first_expected=f"{year}-01-01 00:00:00")
    assert m["steps_s"] == [3600.0] and m["first_stamp"] == m["first_expected"] and n == (8784 if year % 4 == 0 else 8760), f"{year}: the time axis is not hourly from 00:00 on 1 January: {m}"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1))
    tmp.replace(f)
    return m


def layout_variants(metas):
    """The distinct ways in which the yearly stores of 1980-2014 are written (the stores of the years before 2010 and from 2010 differ in where the fill value is declared, in the compression level and in the chunk length of the time axis): {description: [first year, last year, number of years]}."""
    out = {}
    for m in metas:
        key = (f"fill_value in .zarray: {m['zarray']['fill_value']}; _FillValue in the attributes: {m['zattrs_of_the_variable'].get('_FillValue')}; missing_value: {m['zattrs_of_the_variable'].get('missing_value')}; "
               f"zstd level {m['zarray']['compressor'].get('level')}; time chunk of {m['time_zarray']['chunks'][0]} hours; dimension_separator: {m['zarray'].get('dimension_separator', 'not stated (the default is a dot)')}")
        out.setdefault(key, []).append(m["year"])
    return {k: [min(v), max(v), len(v)] for k, v in out.items()}


def grid_meta():
    """The latitude and longitude arrays of one store (1990): ascending, step 0.008333 as stored (rounded), against the nominal 1/120 degree grid."""
    f = AORC / "grid_meta.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for name, n, nominal in (("latitude", NLAT, LAT0), ("longitude", NLON, LON0)):
        a = np.frombuffer(zstd_decode(get(f"{BASE}/1990.zarr/{name}/0")[0]), "<f8")
        assert len(a) == n and (np.diff(a) > 0).all(), f"{name} is not ascending or has the wrong length"
        nom = nominal + np.arange(n) * STEP
        out[name] = dict(first=float(a[0]), last=float(a[-1]), step_stored=float(np.median(np.diff(a))), step_nominal=STEP, ascending=True, n=int(n),
                         stored_minus_nominal_at_ends=[float(a[0] - nom[0]), float(a[-1] - nom[-1])], max_abs_stored_minus_nominal_deg=float(np.abs(a - nom).max()),
                         cell_shift_at_the_study_basins=None)
    out["note"] = ("the coordinate arrays are written with a rounded step (0.008333 instead of 1/120 = 0.0083333...), so they drift from the nominal grid by up to the amount above; the grid is "
                   "taken as the nominal 1/120 degree grid (ilat = round((lat - 20) * 120), ilon = round((lon + 130) * 120)); the stored arrays would shift the cells of a basin by a fraction of one cell at most")
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(f)
    return out


# ---- fetch, decode, reduce ---------------------------------------------------------------------------------------------
def fetch_reduce(year, c, sl, so, cells):
    """One chunk: fetch, decode, and reduce it to per-basin sums for the hours of the time chunk c that lie inside the year. cells: {gauge: (ii, jj)} indices inside the chunk.
    Returns (hours stored, {gauge: (sum of raw values of the valid cells, number of valid cells, minimum raw value, maximum raw value)}, bytes downloaded, absent)."""
    n_year = year_hours(year)[2]
    nh = min(CH_T, n_year - c * CH_T)
    raw, status = get(f"{BASE}/{year}.zarr/{VAR}/{c}.{sl}.{so}", allow_absent=True)
    if status != 200:
        arr = np.full((nh, CH_Y, CH_X), FILL, "<i2")
    else:
        buf = zstd_decode(raw)
        assert len(buf) == CH_T * CH_Y * CH_X * 2, f"{year} {c}.{sl}.{so}: a chunk decodes to {len(buf)} bytes"
        arr = np.frombuffer(buf, "<i2").reshape(CH_T, CH_Y, CH_X)[:nh]
    out = {}
    for g, (ii, jj) in cells.items():
        blk = arr[:, ii, jj].astype(np.int32)
        ok = blk != FILL
        out[g] = (np.where(ok, blk, 0).sum(1, dtype=np.int64), ok.sum(1).astype(np.int32), np.where(ok, blk, 32767).min(1).astype(np.int16), np.where(ok, blk, -32768).max(1).astype(np.int16))
    return nh, out, len(raw), status != 200


def part_file(year, sl, so):
    return PARTS / f"{year}_{sl}.{so}.npz"


def fetch(plan, years=YEARS, threads=THREADS, log_every=200):
    """Fetch every missing (year, spatial chunk) unit and write its part. Units run in order, two at a time, so the threads never wait for the end of a unit; resume-safe. Returns the run's record."""
    PARTS.mkdir(parents=True, exist_ok=True)
    spatial = sorted({s for p in plan.values() for s in p["chunks"]})
    todo = [(y, s) for y in years for s in spatial if not part_file(y, *s).exists()]
    n_tasks = sum(len(time_chunks(y)) for y, _ in todo)
    say(f"fetch: {len(todo)} units (year x spatial chunk) to do of {len(years) * len(spatial)}, {n_tasks} chunk fetches, {threads} threads")
    if not todo:
        return dict(units=0, chunks=0, bytes=0, seconds=0.0, absent_chunks=0, failures=[])
    local = {}                                                                                       # per spatial chunk: the cell indices of each basin that uses it, local to the chunk
    for s in spatial:
        local[s] = {g: (p["ilat"][(p["ilat"] // CH_Y == s[0]) & (p["ilon"] // CH_X == s[1])] - s[0] * CH_Y, p["ilon"][(p["ilat"] // CH_Y == s[0]) & (p["ilon"] // CH_X == s[1])] - s[1] * CH_X)
                    for g, p in plan.items() if s in p["chunks"]}
    t0, done_chunks, nbytes, absent, failures = time.time(), 0, 0, 0, []
    state, remaining = {}, {}                                                                        # unit -> {time chunk: result} (None once a chunk of it has failed); unit -> chunks still out
    pending, units, next_log = {}, iter(todo), log_every
    pool = ThreadPoolExecutor(threads)

    def submit(unit):
        y, s = unit
        cs = time_chunks(y)
        state[unit], remaining[unit] = {}, len(cs)
        for c in cs:
            pending[pool.submit(fetch_reduce, y, c, s[0], s[1], local[s])] = (unit, c)

    for _ in range(2):
        u = next(units, None)
        if u:
            submit(u)
    try:
        while pending:
            fin, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fu in fin:
                unit, c = pending.pop(fu)
                remaining[unit] -= 1
                try:
                    nh, red, nb, ab = fu.result()
                except Exception as e:                                                               # a chunk that could not be had after all tries: the unit stays undone
                    failures.append(f"{unit} chunk {c}: {e}")
                    say("FAILED", failures[-1])
                    state[unit] = None
                else:
                    done_chunks, nbytes, absent = done_chunks + 1, nbytes + nb, absent + int(ab)
                    if state[unit] is not None:
                        state[unit][c] = (nh, red, nb, ab)
                if remaining[unit] == 0:
                    if state[unit] is not None:
                        write_part(unit[0], unit[1], state[unit], local[unit[1]])
                    del state[unit], remaining[unit]
                    u = next(units, None) if len(failures) < 20 else None                           # twenty failures: the link is down, stop starting units
                    if u:
                        submit(u)
            if done_chunks >= next_log:
                next_log = done_chunks - done_chunks % log_every + log_every
                el = time.time() - t0
                say(f"  {done_chunks}/{n_tasks} chunks, {nbytes / 1e6:.0f} MB, {done_chunks / el:.2f} chunks/s, {nbytes / el / 1e6:.2f} MB/s, about {(n_tasks - done_chunks) / (done_chunks / el) / 3600:.2f} h to go")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return dict(units=len(todo), chunks=done_chunks, bytes=int(nbytes), seconds=round(time.time() - t0, 1), absent_chunks=int(absent), failures=failures, threads=threads,
                started=time.strftime("%Y-%m-%d %H:%M", time.localtime(t0)))


def write_part(year, s, results, cells):
    """The part of one (year, spatial chunk): per basin the sums over its cells in this chunk, for every hour from the first stored hour of the year to the last (a whole number of time chunks)."""
    cs = sorted(results)
    nh = sum(results[c][0] for c in cs)
    d = dict(hour0=np.int64(cs[0] * CH_T), n_hours=np.int64(nh), bytes=np.int64(sum(results[c][2] for c in cs)), absent=np.int64(sum(results[c][3] for c in cs)), n_chunks=np.int64(len(cs)))
    for g in cells:
        for k, name in enumerate(("sum", "count", "min", "max")):
            d[f"{name}_{g}"] = np.concatenate([results[c][1][g][k] for c in cs])
    f = part_file(year, *s)
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")                                              # numpy adds .npz to a name that does not end in it
    np.savez_compressed(tmp, **d)
    tmp.replace(f)


# ---- assemble and validate ---------------------------------------------------------------------------------------------
def basin_file(g):
    return AORC / f"aorc_{g}.npz"


def assemble(plan, years=YEARS):
    """Per basin: the hour-ending stamps and the basin-mean rain of every hour of the span, from the parts. A basin-year needs the parts of all spatial chunks of the basin."""
    out = {}
    for g, p in plan.items():
        stamps, mm, nvalid, mx, mn = [], [], [], [], []
        for y in years:
            a, b, n_year = year_hours(y)
            h = np.arange(a, b + 1)
            tot, cnt, hi, lo = np.zeros(len(h)), np.zeros(len(h), np.int64), np.full(len(h), -32768, np.int64), np.full(len(h), 32767, np.int64)
            for s in p["chunks"]:
                z = np.load(part_file(y, *s))
                h0, nh = int(z["hour0"]), int(z["n_hours"])
                assert h0 <= a and h0 + nh > b, f"{y} {s}: the part covers hours {h0}..{h0 + nh - 1}, not {a}..{b}"
                sl = slice(a - h0, b - h0 + 1)
                tot += z[f"sum_{g}"][sl]
                cnt += z[f"count_{g}"][sl]
                hi, lo = np.maximum(hi, z[f"max_{g}"][sl]), np.minimum(lo, z[f"min_{g}"][sl])
            stamps.append(pd.Timestamp(f"{y}-01-01 00:00") + pd.to_timedelta(h, unit="h"))
            mm.append(np.where(cnt > 0, tot * SCALE / np.maximum(cnt, 1), np.nan))
            nvalid.append(cnt)
            mx.append(hi * SCALE)
            mn.append(lo * SCALE)
        out[g] = dict(stamp=pd.DatetimeIndex(np.concatenate([s.values for s in stamps])), rain_mm=np.concatenate(mm), n_valid=np.concatenate(nvalid), max_cell_mm=np.concatenate(mx), min_cell_mm=np.concatenate(mn))
    return out


def era5_hourly(g):
    """The cached ERA5 hourly rain of a basin (check_day_windows.py; Open-Meteo, UTC, the value at time T is the rain of the hour ending at T) as a Series: used only to fill hours that AORC does not hold."""
    j = json.loads((DATA / "hourly" / f"era5_{g}.json").read_text())
    return pd.Series(np.array(j["hourly"]["precipitation"], float), index=pd.DatetimeIndex(pd.to_datetime(j["hourly"]["time"])))


def fill_gaps(series):
    """AORC holds the fill value in every cell of a tile for a few days here and there (found by this download: for example the tile of the French Broad and its neighbours from 1990-07-19 00:00 to 23:00 on 1990-07-23, 120 hours). An hour in
    which NO cell of a basin has a value is filled with the cached ERA5 hourly rain of the basin (the same hour-ending stamps), and flagged in 'filled'; an hour in which only some cells of a basin are missing keeps the mean of the cells that
    have a value (counted in n_valid). Returns what was filled, per basin."""
    log = {}
    for g, s_ in series.items():
        bad = ~np.isfinite(s_["rain_mm"])
        s_["filled"] = bad.copy()
        if not bad.any():
            continue
        e = era5_hourly(g).reindex(s_["stamp"])
        assert e.notna().all(), f"{g}: the cached ERA5 series does not cover the hours to fill"
        s_["rain_mm"] = np.where(bad, e.values, s_["rain_mm"])
        s_["max_cell_mm"] = np.where(bad, np.nan, s_["max_cell_mm"])
        s_["min_cell_mm"] = np.where(bad, np.nan, s_["min_cell_mm"])
        idx = np.flatnonzero(bad)
        runs, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        log[g] = dict(hours_filled=int(bad.sum()), share_of_the_hours=float(bad.mean()), runs=[dict(first_stamp=str(s_["stamp"][a]), last_stamp=str(s_["stamp"][b]), hours=int(b - a + 1),
                                                                                                     era5_mm_in_the_run=float(e.values[a:b + 1].sum())) for a, b in runs])
    return log


def save_basins(series, plan, mode, how):
    for g, s in series.items():
        t = (s["stamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
        meta = dict(gauge_id=g, what="AORC v1.1 APCP_surface, basin-mean rain of each hour in mm; time stamps are the END of the hour, UTC", mode=mode, n_cells=plan[g]["n_cells"], source=BASE,
                    span=SPAN_FOR_THE_RECORD, scale=SCALE, fill_value=FILL, how=how, written=time.strftime("%Y-%m-%d %H:%M"))
        f = basin_file(g)
        tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, stamp_utc_s=np.asarray(t, np.int64), rain_mm=s["rain_mm"].astype("float64"), n_valid=s["n_valid"].astype("uint16"), max_cell_mm=s["max_cell_mm"].astype("float32"),
                            min_cell_mm=s["min_cell_mm"].astype("float32"), filled=s["filled"].astype("uint8"), n_cells=np.int64(plan[g]["n_cells"]), meta=json.dumps(meta))
        tmp.replace(f)


def load_series(g):
    """(hour-ending UTC stamps as a naive DatetimeIndex, basin-mean rain of each hour in mm as float64) of one basin, read from data/aorc/aorc_<gauge_id>.npz."""
    z = np.load(basin_file(g), allow_pickle=False)
    return pd.to_datetime(z["stamp_utc_s"], unit="s").as_unit("us"), z["rain_mm"].astype("float64")


def cache_complete(ids):
    """True when every basin file exists and holds the whole span."""
    try:
        for g in ids:
            stamp, x = load_series(g)
            if len(stamp) != N_HOURS or stamp[0] != FIRST_STAMP or stamp[-1] != LAST_STAMP:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


# ---- validation against the other rain products ------------------------------------------------------------------------
corr = lambda a, b, k: float(pd.concat([a, b.shift(k)], axis=1).dropna().corr().iloc[0, 1])          # corr(a on day d, b on day d-k), as in 10_rain_timing.py
three = lambda a, b: dict(day_before=corr(a, b, 1), same_day=corr(a, b, 0), day_after=corr(a, b, -1))


def utc_day_totals(stamp, x):
    """Rain of the UTC day of each date: the hours that END from 01:00 to 24:00 (= 00:00 of the next date) of that date (an hour belongs to the day in which it lies)."""
    return pd.Series(x, index=(stamp - pd.Timedelta(minutes=1)).normalize()).groupby(level=0).sum()


def validate(ids, plan):
    """Integrity of the assembled series and their comparison with IMERG, Daymet, NLDAS (daily, by UTC day with the lag correlations of 10_rain_timing.py) and with the cached hourly ERA5."""
    from camels import load_basin
    PER = ("1980-10-01", "2014-09-30")
    days = pd.date_range(*PER)
    per_basin, expected = {}, pd.date_range(FIRST_STAMP, LAST_STAMP, freq="h")
    for g in ids:
        stamp, x = load_series(g)
        z = np.load(basin_file(g), allow_pickle=False)
        nv, nc, fl = z["n_valid"].astype(int), int(z["n_cells"]), z["filled"].astype(bool)
        assert stamp.equals(expected), f"{g}: the time axis has a gap, a repeat or the wrong span"
        assert np.isfinite(x).all(), f"{g}: missing hours"
        assert (x >= 0).all(), f"{g}: negative rain"
        assert ((nv > 0) | fl).all() and (nv[fl] == 0).all(), f"{g}: an hour without a valid cell that is not flagged as filled"
        assert (np.nan_to_num(z["min_cell_mm"], nan=0.0) >= 0).all() and (np.nan_to_num(z["max_cell_mm"], nan=0.0) >= 0).all() and np.nanmax(z["max_cell_mm"]) < 400, f"{g}: a negative cell value, or an implausible one"
        jd = load_basin(g, "daymet").loc[PER[0]:PER[1], "prcp"].astype(float)
        nl = load_basin(g, "nldas").loc[PER[0]:PER[1], "prcp"].astype(float)
        day = utc_day_totals(stamp, x).reindex(days)
        assert day.notna().all(), f"{g}: the UTC-day totals do not cover the period"
        f = DATA / "satellite" / f"{g}.csv"
        im = pd.read_csv(f, parse_dates=["date"]).set_index("date")["imerg"].reindex(days) if f.exists() else None
        e = json.loads((DATA / "hourly" / f"era5_{g}.json").read_text())
        es, ex = pd.DatetimeIndex(pd.to_datetime(e["hourly"]["time"])), np.array(e["hourly"]["precipitation"], float)
        assert es.equals(stamp), f"{g}: the cached ERA5 file has another time axis"
        eday = utc_day_totals(es, ex).reindex(days)
        both = lambda a, b: a.notna() & b.notna()
        rec = dict(n_cells=nc, n_hours=int(len(x)), first_stamp=str(stamp[0]), last_stamp=str(stamp[-1]), hours_with_some_cells_missing_not_filled=int(((nv < nc) & ~fl).sum()),
                   fill_cells_met=int((nc - nv)[~fl].sum()), hours_with_no_valid_cell=int(fl.sum()), hours_filled_from_era5=int(fl.sum()), hours_with_no_valid_cell_and_not_filled=int(((nv == 0) & ~fl).sum()), negative_hours=int((x < 0).sum()),
                   min_hourly_mm=float(x.min()), max_hourly_mm=float(x.max()), max_single_cell_mm_per_hour=float(np.nanmax(z["max_cell_mm"])), mean_mm_per_day_utc_days=float(day.mean()), wet_hours_share=float((x > 0.0).mean()),
                   mean_mm_day=dict(aorc=float(day.mean()), daymet=float(jd.mean()), nldas=float(nl.mean()), era5=float(eday.mean()), imerg=float(im.mean()) if im is not None else None),
                   ratio_aorc_to_daymet_mean=float(day.mean() / jd.mean()), ratio_aorc_to_nldas_mean=float(day.mean() / nl.mean()),
                   daily_by_utc_day=dict(daymet_vs_aorc=three(jd, day), nldas_vs_aorc=three(nl, day), era5_vs_aorc=three(eday, day)))
        if im is not None:
            rec["daily_by_utc_day"]["imerg_vs_aorc"] = three(im, day)
            rec["daily_by_utc_day"]["imerg_vs_aorc_days"] = int(both(im, day).sum())
            rec["ratio_aorc_to_imerg_mean_on_common_days"] = float(day[both(im, day)].mean() / im[both(im, day)].mean())
        lags = {}
        for label, (a, b) in {"before_2002": ("1980-09-25", "2001-12-31"), "from_2002": ("2002-01-01", "2014-10-05")}.items():
            m = (stamp >= pd.Timestamp(a)) & (stamp <= pd.Timestamp(b) + pd.Timedelta(hours=23)) & ~fl                     # the hours filled from ERA5 are left out of this comparison with ERA5
            A, E = pd.Series(x[m]), pd.Series(ex[m])
            lags[label] = {str(k): float(A.corr(E.shift(k))) for k in range(-3, 4)}                  # corr(AORC at hour h, ERA5 at hour h-k): k=0 means the same stamp
        rec["hourly_against_era5_by_lag"] = dict(how_to_read="correlation of the AORC basin-mean hourly rain at stamp T with the ERA5 grid-point hourly rain at stamp T-k hours (ERA5 is the rain of the hour ending at its stamp; AORC's convention is what is being tested; "
                                                             "the best lag is the offset between the two series, which holds the difference of the two stamp conventions and of the timing of the two products)", **lags,
                                                 best_lag_before_2002=int(max(lags["before_2002"], key=lags["before_2002"].get)), best_lag_from_2002=int(max(lags["from_2002"], key=lags["from_2002"].get)))
        per_basin[g] = rec
    med = lambda f: float(np.median([f(r) for r in per_basin.values()]))
    summary = dict(
        median_daily_correlation_same_day=dict(daymet=med(lambda r: r["daily_by_utc_day"]["daymet_vs_aorc"]["same_day"]), nldas=med(lambda r: r["daily_by_utc_day"]["nldas_vs_aorc"]["same_day"]),
                                               era5=med(lambda r: r["daily_by_utc_day"]["era5_vs_aorc"]["same_day"]), imerg=med(lambda r: r["daily_by_utc_day"]["imerg_vs_aorc"]["same_day"])),
        median_daily_correlation_day_before=dict(daymet=med(lambda r: r["daily_by_utc_day"]["daymet_vs_aorc"]["day_before"]), nldas=med(lambda r: r["daily_by_utc_day"]["nldas_vs_aorc"]["day_before"]),
                                                 imerg=med(lambda r: r["daily_by_utc_day"]["imerg_vs_aorc"]["day_before"])),
        median_daily_correlation_day_after=dict(daymet=med(lambda r: r["daily_by_utc_day"]["daymet_vs_aorc"]["day_after"]), nldas=med(lambda r: r["daily_by_utc_day"]["nldas_vs_aorc"]["day_after"]),
                                                imerg=med(lambda r: r["daily_by_utc_day"]["imerg_vs_aorc"]["day_after"])),
        median_ratio_aorc_to_daymet_mean_rain=med(lambda r: r["ratio_aorc_to_daymet_mean"]),
        median_hourly_correlation_with_era5_at_lag_0=dict(before_2002=med(lambda r: r["hourly_against_era5_by_lag"]["before_2002"]["0"]), from_2002=med(lambda r: r["hourly_against_era5_by_lag"]["from_2002"]["0"])),
        best_hourly_lag_against_era5_by_basin=dict(before_2002={g: r["hourly_against_era5_by_lag"]["best_lag_before_2002"] for g, r in per_basin.items()},
                                                   from_2002={g: r["hourly_against_era5_by_lag"]["best_lag_from_2002"] for g, r in per_basin.items()}),
        all_series_complete=all(r["hours_with_no_valid_cell_and_not_filled"] == 0 and r["negative_hours"] == 0 for r in per_basin.values()),
        hours_filled_from_era5_by_basin={g: r["hours_filled_from_era5"] for g, r in per_basin.items() if r["hours_filled_from_era5"]},
        hours_with_some_cells_missing_by_basin={g: r["hours_with_some_cells_missing_not_filled"] for g, r in per_basin.items() if r["hours_with_some_cells_missing_not_filled"]})
    return per_basin, summary


# ---- does the rounding of the stored coordinate arrays matter? --------------------------------------------------------------------------------------
def grid_drift_check(ids, year=2012):
    """The coordinate arrays of the stores are written with a rounded step (0.008333 instead of 1/120 degree), so their cell centres drift from the nominal grid used for the masks by up to 0.0014 degree north and 0.0028 degree
    east in the study area. Re-do the basin means of one year with the cells chosen on the STORED coordinates (cell centres inside the outline, shapely contains) and compare them with the means of the nominal grid: the cells that
    differ, and how much the hourly and daily basin means change. The year is fetched again (about 850 chunks) into a scratch folder of its own."""
    global PARTS
    import shapely
    polys = outlines(ids)
    lat_s = np.frombuffer(zstd_decode(get(f"{BASE}/1990.zarr/latitude/0")[0]), "<f8")
    lon_s = np.frombuffer(zstd_decode(get(f"{BASE}/1990.zarr/longitude/0")[0]), "<f8")
    nominal = plan_cells(ids, "outline")
    plan_s = {}
    for g in ids:
        p = polys[g]
        x0, y0, x1, y1 = p.bounds
        il, jl = np.flatnonzero((lat_s >= y0 - 0.02) & (lat_s <= y1 + 0.02)), np.flatnonzero((lon_s >= x0 - 0.02) & (lon_s <= x1 + 0.02))
        LON, LAT = np.meshgrid(lon_s[jl], lat_s[il])
        ii, jj = np.nonzero(shapely.contains_xy(p, LON, LAT))
        la, lo = il[ii], jl[jj]
        plan_s[g] = dict(ilat=la, ilon=lo, n_cells=len(la), chunks=sorted({(int(a // CH_Y), int(b // CH_X)) for a, b in zip(la, lo)}))
    keep, PARTS = PARTS, AORC / "drift_parts"
    try:
        fetch(plan_s, years=[year])
        ser = assemble(plan_s, years=[year])
    finally:
        PARTS = keep
    out = {}
    for g in ids:
        stamp, x = load_series(g)
        nom = pd.Series(x, index=stamp).loc[f"{year}-01-01 00:00":f"{year}-12-31 23:00"]
        sto = pd.Series(ser[g]["rain_mm"], index=ser[g]["stamp"])
        ok = np.isfinite(sto.values) & ~np.asarray(np.load(basin_file(g))["filled"].astype(bool))[(stamp >= nom.index[0]) & (stamp <= nom.index[-1])]
        d_n, d_s = (v.groupby((v.index - pd.Timedelta(minutes=1)).normalize()).sum() for v in (nom, sto))
        a_, b_ = set(zip(nominal[g]["ilat"].tolist(), nominal[g]["ilon"].tolist())), set(zip(plan_s[g]["ilat"].tolist(), plan_s[g]["ilon"].tolist()))
        out[g] = dict(cells_nominal=len(a_), cells_stored_coordinates=len(b_), cells_only_on_the_stored_grid=len(b_ - a_), cells_only_on_the_nominal_grid=len(a_ - b_),
                      hourly_correlation=float(np.corrcoef(nom.values[ok], sto.values[ok])[0, 1]), daily_correlation=float(np.corrcoef(d_n.values, d_s.values)[0, 1]),
                      ratio_of_the_means=float(sto.values[ok].sum() / nom.values[ok].sum()), max_abs_hourly_difference_mm=float(np.abs(nom.values[ok] - sto.values[ok]).max()))
    return dict(what=grid_drift_check.__doc__.split("\n")[0] + " " + " ".join(grid_drift_check.__doc__.split("\n")[1:]).strip(), year=year, per_basin=out,
                lowest_hourly_correlation=min(v["hourly_correlation"] for v in out.values()), largest_difference_of_the_ratio_from_1=max(abs(v["ratio_of_the_means"] - 1) for v in out.values()),
                largest_abs_hourly_difference_mm=max(v["max_abs_hourly_difference_mm"] for v in out.values()))


# ---- the stamp convention of the radar period against Stage IV (hour-ending files of the NCEP analysis) ---------------------------------------------
STAGE_IV = "https://mesonet.agron.iastate.edu/archive/data/{y}/{m}/{d}/stage4/ST4.{y}{m}{d}{h}.01h.grib"                  # Iowa Environmental Mesonet mirror; the file labelled T holds the hour that ends at T (GRIB_VALID_TIME = T)
STAGE_IV_CRS = "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +x_0=0 +y_0=0 +R=6371200 +units=m +no_defs"                   # the polar stereographic grid of the GRIB files (lower-left cell centre 23.117 N, 119.023 W, as NCEP documents)


def stage_iv_basin_hourly(ids, labels):
    """Hourly Stage IV rain of the basins, from the hourly files labelled with the END of the hour (labels: hour-ending UTC times): the mean of the 4.76 km cells whose centre lies inside the outline (the cell holding the
    centroid for a basin that holds none), 9999 (no radar coverage) left out. The files are fetched, read and dropped one by one (nothing is kept on disk). Returns a DataFrame indexed by the label."""
    import tempfile
    import pyproj
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.ops import transform as shp_transform
    polys = outlines(ids)
    crs = pyproj.CRS.from_proj4(STAGE_IV_CRS)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    lock, state = threading.Lock(), {}

    def build_masks(tf, h, w):
        ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(*(tf * (0.5, h - 0.5)))
        assert -121.0 < ll[0] < -118.0 and 22.0 < ll[1] < 24.0, f"the Stage IV grid is not where it is assumed to be: lower-left cell centre {ll}"          # 23.117 N, 119.02 W in 2002-2014; the files of 2000-2001 come on a larger grid with another corner
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

    def one(t):
        """The hourly Stage IV basin means of the file labelled t (the file is dropped at once)."""
        raw, _ = get(STAGE_IV.format(y=f"{t:%Y}", m=f"{t:%m}", d=f"{t:%d}", h=f"{t:%H}"))
        with tempfile.NamedTemporaryFile(suffix=".grib") as tmp:
            tmp.write(raw)
            tmp.flush()
            with rasterio.open(tmp.name) as ds:
                a = ds.read(1).astype("float64")
                tags, tf, h, w = ds.tags(1), ds.transform, ds.height, ds.width
        assert tags.get("GRIB_ELEMENT") == "APCP" and int(tags["GRIB_FORECAST_SECONDS"]) == 3600 and int(tags["GRIB_VALID_TIME"]) == int((t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)), \
            f"{t}: the file does not hold the hour ending at its label: {tags}"
        with lock:
            if "masks" not in state:
                state["masks"] = build_masks(tf, h, w)
        a[a >= 9998] = np.nan
        return {g: float(np.nanmean(a[m])) if np.isfinite(a[m]).any() else np.nan for g, m in state["masks"].items()}

    def safe_one(t):
        try:
            return one(t)
        except RuntimeError:                                                                         # a file that the mirror does not have: that hour stays empty
            return None
    with ThreadPoolExecutor(min(THREADS, 6)) as ex:                                                  # the Iowa Mesonet archive is run by volunteers: 6 connections at most (8 were used for the 1,800 hourly files of 2026-10-07)
        got = list(ex.map(safe_one, labels))
    assert sum(x is not None for x in got) >= 0.9 * len(labels), f"only {sum(x is not None for x in got)} of {len(labels)} Stage IV files could be fetched"
    rows = {t: (x if x is not None else {g: np.nan for g in ids}) for t, x in zip(labels, got)}
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "hour_ending_utc"
    return out


def stage_iv_check(ids, period=("2005-10-01", "2014-09-30"), n_days=4):
    """Does an AORC stamp T hold the rain of the hour that ENDS at T, as the Stage IV files labelled T do? The hourly AORC basin means are correlated with the hourly Stage IV basin means of a few storm days of the period
    (the days with the most ERA5 rain over the 12 basins, at least 30 days apart; Stage IV is the radar analysis that AORC follows from 2002 on, according to the summary of its methods) at lags of -3 to +3 hours,
    pooled over the basins and days. A stamp that is the START of the hour would peak at lag -1."""
    tot = None
    for g in ids:
        e = json.loads((DATA / "hourly" / f"era5_{g}.json").read_text())
        sr = pd.Series(np.array(e["hourly"]["precipitation"], float), index=pd.DatetimeIndex(pd.to_datetime(e["hourly"]["time"])))
        d = sr.groupby((sr.index - pd.Timedelta(minutes=1)).normalize()).sum()
        tot = d if tot is None else tot + d
    tot = tot.loc[period[0]:period[1]] / len(ids)
    days = []
    for d_, v_ in tot.sort_values(ascending=False).items():
        if all(abs((d_ - p).days) > 30 for p in days):
            days.append(d_)
        if len(days) == n_days:
            break
    labels = sorted({t for d in days for t in pd.date_range(d - pd.Timedelta(hours=3), d + pd.Timedelta(hours=27), freq="h")})
    st4 = stage_iv_basin_hourly(ids, labels)
    cube_a = np.array([pd.Series(load_series(g)[1], index=load_series(g)[0]).reindex(st4.index).values for g in ids])      # [basin, hour]: AORC at the stamps of the Stage IV labels

    def corr_at(k, i=None):
        """Correlation of AORC at stamp T with Stage IV at label T-k, over the hours both have: all basins pooled, or basin i."""
        sh = st4[ids].reindex(st4.index - pd.Timedelta(hours=k)).values.T                                    # row i: Stage IV at label T_i - k (missing where that hour was not fetched)
        a, b = (cube_a, sh) if i is None else (cube_a[i:i + 1], sh[i:i + 1])
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    lags = {str(k): corr_at(k) for k in range(-3, 4)}
    per_basin_best = {g: int(max(range(-3, 4), key=lambda k: corr_at(k, i))) for i, g in enumerate(ids)}
    return dict(what="correlation of the hourly AORC basin means (stamp T) with the hourly Stage IV basin means (the file labelled T: the hour ending at T) at lags k of -3 to +3 hours (AORC at T against Stage IV at T-k), pooled over the 12 basins "
                     "and the storm days; k = 0 is the stamp as the END of the hour, k = -1 the stamp as the START of the hour",
                days=[str(d.date()) for d in days], n_hourly_files=len(labels), source=STAGE_IV.replace("{y}", "YYYY").replace("{m}", "MM").replace("{d}", "DD").replace("{h}", "HH"),
                pooled_correlation_by_lag=lags, best_lag=int(max(lags, key=lags.get)), best_lag_by_basin=per_basin_best,
                stage_iv_hours_with_no_value=int(np.isnan(st4.values).sum()), mean_stage_iv_mm_per_hour=float(np.nanmean(st4.values)), mean_aorc_mm_per_hour=float(np.nanmean(cube_a)))


# ---- the measurement that decides between the outline mean and the 5 km mean ---------------------------------------------------
def measure(plan, n_head=120, n_get=96, seed=1):
    """Chunk sizes (HEAD requests on a random sample over years, seasons and spatial chunks) and the speed of the link (GET + decode with THREADS threads on another sample)."""
    import random
    rng = random.Random(seed)
    spatial = sorted({s for p in plan.values() for s in p["chunks"]})
    n_fetch = sum(len(time_chunks(y)) * len(spatial) for y in YEARS)
    sample = [(y, rng.choice(time_chunks(y)), rng.choice(spatial)) for y in [rng.choice(YEARS) for _ in range(n_head)]]
    import requests

    def head(t):
        y, c, s = t
        for k in range(6):
            try:
                r = session().head(f"{BASE}/{y}.zarr/{VAR}/{c}.{s[0]}.{s[1]}", timeout=60)
                return r.status_code, int(r.headers.get("Content-Length", -1))
            except (requests.RequestException, OSError):
                _loc.s = None
                time.sleep(1 + k)
        return -1, -1
    with ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(head, sample))
    sizes = [n for c, n in res if c == 200]
    assert len(sizes) > 0.9 * len(res), "most of the sampled chunks could not be found"
    mean_kb = float(np.mean(sizes)) / 1024
    sample2 = [(y, rng.choice(time_chunks(y)), rng.choice(spatial)) for y in [rng.choice(YEARS) for _ in range(n_get)]]

    def one(t):
        y, c, s = t
        raw, st = get(f"{BASE}/{y}.zarr/{VAR}/{c}.{s[0]}.{s[1]}", allow_absent=True)
        zstd_decode(raw) if raw else None
        return len(raw)
    t0 = time.time()
    with ThreadPoolExecutor(THREADS) as ex:
        got = list(ex.map(one, sample2))
    wall = time.time() - t0
    rate = n_get / wall
    return dict(n_basins=len(plan), spatial_chunks_touched=len(spatial), spatial_chunks=[list(s) for s in spatial], time_chunks_per_year_median=int(np.median([len(time_chunks(y)) for y in YEARS])),
                chunk_fetches=int(n_fetch), sample_head=dict(n=len(res), n_found=len(sizes), mean_kb=mean_kb, median_kb=float(np.median(sizes)) / 1024, min_kb=min(sizes) / 1024, max_kb=max(sizes) / 1024),
                sample_get=dict(n=n_get, threads=THREADS, seconds=round(wall, 1), chunks_per_s=round(rate, 2), mb_per_s=round(sum(got) / wall / 1e6, 2), mean_kb=float(np.mean(got)) / 1024),
                estimated_gb=float(n_fetch * np.mean(sizes + got) / 1e9), estimated_hours=float(n_fetch / rate / 3600))


def decide(meas):
    over = meas["estimated_gb"] > LIMIT_GB or meas["estimated_hours"] > LIMIT_HOURS
    return dict(rule=f"the mean over the cells inside each outline, unless the download would exceed about {LIMIT_GB:g} GB or about {LIMIT_HOURS:g} hours at the measured speed, in which case the mean of the cells within 5 km of the centroid",
                estimated_gb=meas["estimated_gb"], estimated_hours=meas["estimated_hours"], limit_gb=LIMIT_GB, limit_hours=LIMIT_HOURS, over_the_limit=bool(over),
                mode="disc5km" if over else "outline", chosen="mean of the cells within 5 km of the centroid" if over else "mean of the cells inside the basin outline")


# ---- main -----------------------------------------------------------------------------------------------------------------
def ensure(verbose=True):
    """Make sure data/aorc/aorc_<gauge_id>.npz exist for the 12 basins (the whole span), downloading what is missing; returns the record of results/aorc_data.json."""
    ids = ids_of_the_study()
    if cache_complete(ids) and RECORD.exists():
        return json.loads(RECORD.read_text())
    return run(ids)


def run(ids, years=YEARS):
    AORC.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "nice"):
        os.nice(10)                                                                                  # the downloads must not slow down a training that holds the CPU
    t_start = time.time()
    old = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    say("store metadata of the", len(YEARS), "yearly stores")
    with ThreadPoolExecutor(THREADS) as ex:
        metas = list(ex.map(store_meta, YEARS))
    gm = grid_meta()
    mfile = AORC / "choice.json"
    if mfile.exists():                                                                               # the choice made before the first fetch stands for every resumed run
        ch = json.loads(mfile.read_text())
        say("choice of an earlier run:", ch["decision"]["chosen"])
    else:
        say("planning: rasterising the outlines on the AORC grid")
        plan0 = plan_cells(ids, "outline")
        say("measuring chunk sizes and speed")
        meas = measure(plan0)
        dec = decide(meas)
        ch = dict(measured=meas, decision=dec, decided=time.strftime("%Y-%m-%d %H:%M"))
        tmp = mfile.with_suffix(".tmp")
        tmp.write_text(json.dumps(ch, indent=1))
        tmp.replace(mfile)
        say(f"estimate: {meas['chunk_fetches']} chunk fetches, {meas['estimated_gb']:.1f} GB, {meas['estimated_hours']:.1f} h at {meas['sample_get']['chunks_per_s']} chunks/s -> {dec['chosen']}")
    mode = ch["decision"]["mode"]
    plan = plan_cells(ids, mode)
    for g, p in plan.items():
        say(f"  {g}: {p['n_cells']} cells, chunks {p['chunks']}, mask area {p['mask_area_km2']:.0f} km2 ({p['mask_over_outline_area']:.3f} of the outline's {p['outline_area_km2']:.0f})")
    runs = []
    rj = AORC / "download_runs.json"
    if rj.exists():
        runs = json.loads(rj.read_text())
    r = fetch(plan, years)
    if r["chunks"] or r["failures"]:
        runs.append(r)
        rj.write_text(json.dumps(runs, indent=1))
    if r["failures"]:
        raise SystemExit(f"{len(r['failures'])} chunks could not be fetched; run again to resume (parts that are written are kept):\n" + "\n".join(r["failures"][:10]))
    say("assembling the basin series")
    series = assemble(plan, years)
    if tuple(years) != YEARS:
        return series
    parts_total = dict(chunks=0, bytes=0, absent=0)
    for y in years:
        for sp in sorted({c for p in plan.values() for c in p["chunks"]}):
            z = np.load(part_file(y, *sp))
            parts_total["chunks"] += int(z["n_chunks"])
            parts_total["bytes"] += int(z["bytes"])
            parts_total["absent"] += int(z["absent"])
    gap_log = fill_gaps(series)
    say("hours with no value in any cell of a basin, filled from the cached ERA5 hourly rain:", {g: v["hours_filled"] for g, v in gap_log.items()} or "none")
    save_basins(series, plan, mode, "sum and count of the valid cells per hour, from the parts of data/aorc/parts/ (code/fetch_aorc.py); hours without any valid cell filled from the cached ERA5 hourly rain (flag 'filled')")
    say("validating")
    per_basin, summary = validate(ids, plan)
    dfile = AORC / "drift_check.json"
    if dfile.exists():
        drift = json.loads(dfile.read_text())
    else:
        say("the rounding of the stored coordinates: the basin means of one year again, on the stored cell centres")
        try:
            drift = grid_drift_check(ids)
            dfile.write_text(json.dumps(drift, indent=1))
            say(f"  lowest hourly correlation {drift['lowest_hourly_correlation']:.5f}, largest ratio of means - 1: {drift['largest_difference_of_the_ratio_from_1']:.4f}")
        except Exception as e:
            drift = dict(failed=f"{type(e).__name__}: {e}")
            say("the drift check failed:", drift["failed"])
    sfile = AORC / "stage_iv_check.json"
    st4c = json.loads(sfile.read_text()) if sfile.exists() else {}                                   # periods that were checked on an earlier run are kept; failed or missing ones are tried again
    for key, (period, n) in dict(test_years=(("2005-10-01", "2014-09-30"), 4), years_2003_2004=(("2002-10-01", "2004-12-31"), 3), years_2000_2001=(("2000-06-01", "2001-12-31"), 3)).items():
        if "pooled_correlation_by_lag" in st4c.get(key, {}):
            continue
        say(f"the stamp convention against Stage IV, {key}: fetching the hourly files of storm days")
        try:
            st4c[key] = stage_iv_check(ids, period, n)
            say(f"  Stage IV, {key}: pooled correlation by lag {st4c[key]['pooled_correlation_by_lag']}")
        except Exception as e:                                                                       # a failed side check must not lose the download: it is recorded as failed
            st4c[key] = dict(failed=f"{type(e).__name__}: {e}")
            say(f"the Stage IV check {key} failed:", st4c[key]["failed"])
    if any("pooled_correlation_by_lag" in v for v in st4c.values()):
        sfile.write_text(json.dumps(st4c, indent=1))
    rec = dict(
        what="NOAA AORC v1.1 hourly rain (APCP_surface) reduced to one hourly basin-mean series per study basin, for experiment E1b (code/16_aorc_rain_to_midnight.py)",
        written=time.strftime("%Y-%m-%d %H:%M"), code=dict(file="code/fetch_aorc.py", sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(), note="the code that assembled and checked this record; the chunks were fetched and reduced by an earlier version of the same file that differs only in the checks and records that follow the fetch"), source=BASE + "/YYYY.zarr", span=SPAN_FOR_THE_RECORD, n_hours=N_HOURS, files="data/aorc/aorc_<gauge_id>.npz (stamp_utc_s, rain_mm, n_valid, max_cell_mm, min_cell_mm, filled, n_cells, meta); data/aorc/parts/: the sums per (year, spatial chunk) that these are made from",
        choice=ch, mask_mode=mode,
        mask_how=("the cells whose centre lies inside the basin outline (CAMELS HCDN_nhru_final_671.shp), cells of the nominal 1/120 degree grid, rasterio.features.rasterize with all_touched=False; the basin mean is the plain "
                  "mean of those cells: the cell area varies by less than 1% over the largest basin (cell_area_variation_over_the_basin), so area weights would change nothing worth keeping"),
        grid=dict(nominal=dict(lat0=LAT0, lon0=LON0, step_deg=STEP, n_lat=NLAT, n_lon=NLON), stored_coordinates=gm, index_formula="ilat = round((lat - 20) * 120), ilon = round((lon + 130) * 120); the latitude axis ascends (index 0 = 20.0 N)"),
        stores_checked=[dict(year=m["year"], n_hours=m["n_hours"], first_stamp=m["first_stamp"], last_stamp=m["last_stamp"], steps_s=m["steps_s"]) for m in metas],
        store_layout=dict(zarray=metas[0]["zarray"], zattrs_of_the_variable=metas[0]["zattrs_of_the_variable"], time_zarray=metas[0]["time_zarray"], time_zattrs=metas[0]["time_zattrs"],
                          chunks_and_dtype_same_in_every_year=all(m["zarray"]["chunks"] == metas[0]["zarray"]["chunks"] and m["zarray"]["dtype"] == metas[0]["zarray"]["dtype"] for m in metas),
                          variants_found=layout_variants(metas)),
        time_stamps=dict(convention="UTC; every yearly store starts at 00:00 on 1 January and the stamps of consecutive stores follow each other hour by hour (asserted for every year and over the whole span); the metadata call the stamp "
                                    "'verification time generated by wgrib2 function verftime()' and do not say whether it is the start or the end of the hour. Evidence on that, in order of sharpness: (1) stamp_check_against_stage_iv (this download): "
                                    "the stamp against the hourly Stage IV files (a file labelled T holds the hour that ends at T) of storm days of the test years, of 2003-2004 and of 2000-2001; (2) results/data_access.json: r = 0.991 against Stage IV at lag 0 and 0.53 at lag +-1 on 2014-01-02 "
                                    "(a scouting measurement of one day); (3) summary.best_hourly_lag_against_era5_by_basin and per_basin.hourly_against_era5_by_lag: the lag profile against the cached ERA5 hourly rain (the hour ending at the stamp), "
                                    "before 2002 and from 2002",
                         stamp_check_against_stage_iv=st4c),
        plan={g: {k: v for k, v in p.items() if k not in ("ilat", "ilon")} for g, p in plan.items()},
        download=dict(runs=runs, chunks_fetched=int(sum(x["chunks"] for x in runs)), bytes=int(sum(x["bytes"] for x in runs)), gb=round(sum(x["bytes"] for x in runs) / 1e9, 3),
                      seconds=round(sum(x["seconds"] for x in runs), 1), hours=round(sum(x["seconds"] for x in runs) / 3600, 2), absent_chunks=int(sum(x["absent_chunks"] for x in runs)),
                      chunks_in_all_parts=parts_total["chunks"], bytes_in_all_parts=parts_total["bytes"], gb_in_all_parts=round(parts_total["bytes"] / 1e9, 3), absent_chunks_in_all_parts=parts_total["absent"],
                      note="chunks, bytes and seconds are those of the fetching runs listed (bytes = compressed chunk bodies; the small metadata files and the sample fetches that measured size and speed are not counted); seconds = wall-clock time "
                           "of the fetching, summed over the runs; chunks_in_all_parts and bytes_in_all_parts count every chunk behind the parts on disk, including the chunks of an earlier test run of one year (2012) that is not in the runs"),
        grid_drift_check=drift,
        gap_fill=dict(policy="an hour in which no cell of a basin has a value (the source holds the fill value for a whole tile for a few days) is filled with the cached ERA5 hourly rain of the basin (Open-Meteo, the same hour-ending stamps) and flagged "
                             "in the 'filled' array of the basin file; an hour with only some cells missing keeps the mean of the cells that have a value (n_valid); the basin mean is never filled from another AORC basin",
                      per_basin=gap_log, total_hours_filled=int(sum(v["hours_filled"] for v in gap_log.values())), basin_hours_in_total=int(N_HOURS * len(ids))),
        per_basin=per_basin, summary=summary, runtime_s=round(time.time() - t_start, 1))
    if old.get("download") and not runs:
        rec["download"] = old["download"]
    tmp = RECORD.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)))
    tmp.replace(RECORD)
    say("done:", RECORD)
    return rec


if __name__ == "__main__":
    run(ids_of_the_study())
