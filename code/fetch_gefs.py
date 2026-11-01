"""E2 data, step 2: the NOAA GEFSv12 reforecast rain (control member c00), 2000-2014, fetched day by day and reduced to the south-eastern box.

Source: the open AWS bucket https://noaa-gefs-retrospective.s3.amazonaws.com/ (no login), one file per forecast run:
    GEFSv12/reforecast/YYYY/YYYYMMDD00/c00/Days:1-10/apcp_sfc_YYYYMMDD00_c00.grib2     (+ the same name with .idx beside it)
The run is issued at 00 UTC; the file holds 80 messages of total precipitation (APCP, kg m-2 = mm) on a global 0.25 degree grid: the windows
0-3, 0-6, 6-9, 6-12, 12-15, 12-18, ... hours (GFS style: a 3 h and a 6 h accumulation of the same 6 h block; the 6 h one contains the 3 h one).
The .idx lists the byte offset of each message, so ONE Range request per day fetches messages 1..26 (all windows that end by forecast hour 78,
about 9 MB of the 25 MB file): local days t+1..t+3 need forecast hours +5 to +77 (Eastern) or +6 to +78 (Central) of the 00 UTC run of UTC date t+1
(results/data_access.json, gefs_reforecast_record.task5_issue_time).

Each day is reduced at once and the GRIB bytes are deleted:
 - decoded with cfgrib (cfgrib.messages on eccodes) through a temporary file; every message's startStep, endStep, valid time, grid keys and byte length are checked
   against the .idx and the expected values; a day that fails any check is stored as a decode error (it is fetched again twice first);
 - the box longitude -91 to -75, latitude 25 to 38 is cut (the file's longitudes are 0..360: 269..285; its rows run north to south);
 - the overlapping accumulations are differenced to NON-overlapping 3-hourly amounts: 0-3, (0-6)-(0-3) = 3-6, 6-9, (6-12)-(6-9) = 9-12, ...; the windows are
   asserted to tile 0..78 h with no gap and no overlap; tiny negatives left by the packing are clamped to 0 (the most negative value of each day is kept);
 - stored per month in data/gefs/c00/apcp_YYYYMM.npz (+ apcp_YYYYMM.json, the provenance): init_date (YYYYMMDD int32, 00 UTC), step_end_hours (3, 6, ..., 78), lat (25..38,
   south to north), lon (-91..-75, west to east), apcp (uint16, tenths of a mm per 3 h window, [day, step, lat, lon], 65535 = no data), flag (per day: 0 ok, 1 missing on
   the server, 2 decode error, 255 not fetched yet). A month file is rewritten atomically (temp name, then replace) after every stored day, so a stop loses only the
   days in flight; data/gefs/c00/done_apcp.txt gets one line per stored day ("YYYYMMDD flag"). It is rebuilt from the month files at every start.
Run it as   ~/AI_Class/.venv-grib/bin/python code/fetch_gefs.py     (resume-safe: it fetches the days not yet stored)
 --days N            fetch only the first N days not yet stored (tests)       --start / --end YYYY-MM-DD   restrict the days to fetch (the files always cover 2000-01-01 .. 2014-10-05)
 --check             verify the stored months (shapes, flags, counts) without downloading; exit code 0 = consistent, 1 = problems
 --retry-errors / --retry-missing   fetch the days flagged 2 / 1 again           --verbose   print the decoded windows of every day     --out DIR   another data directory (tests)
Needs the environment ~/AI_Class/.venv-grib (eccodes, cfgrib, xarray, numpy, requests; the project's .venv has no GRIB decoder) and the proxy of the shell (HTTPS_PROXY): direct
reads of the bucket give 20 KB/s, through the proxy 0.5-4 MB/s with several streams. Exit codes: 0 done, 2 some days could not be fetched (rerun), 3 the link looks dead,
4 an unrecoverable error (see the last lines of the log), 130 stopped by a signal. `from fetch_gefs import load_gefs` (numpy only) reads the stored days.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                          # the decoder runs beside the fetch threads on a two-core laptop that is also training

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import importlib.metadata as imd
import json
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                  # as code/common.py (not imported: it loads matplotlib, which the GRIB environment does not have)
BUCKET = "https://noaa-gefs-retrospective.s3.amazonaws.com"
MEMBER = "c00"
KEY = "GEFSv12/reforecast/{year}/{ymd}00/{member}/Days:1-10/apcp_sfc_{ymd}00_{member}.grib2"
FULL = (dt.date(2000, 1, 1), dt.date(2014, 10, 5))          # the days the files cover (the project's record ends 2014-09-30; a few days of margin)
MAX_HOUR, STEP_H = 78, 3
STEPS = np.arange(STEP_H, MAX_HOUR + 1, STEP_H)             # window end hours 3, 6, ..., 78
NSTEP = len(STEPS)                                          # 26
BOX = dict(lon=(-91.0, -75.0), lat=(25.0, 38.0))
GRID = dict(Ni=1440, Nj=721, iScansNegatively=0, jScansPositively=0, latitudeOfFirstGridPointInDegrees=90.0, longitudeOfFirstGridPointInDegrees=0.0,
            latitudeOfLastGridPointInDegrees=-90.0, longitudeOfLastGridPointInDegrees=359.75, iDirectionIncrementInDegrees=0.25, jDirectionIncrementInDegrees=0.25)
LAT_ALL = 90.0 - 0.25 * np.arange(GRID["Nj"])               # the rows of the file, north to south
LON_ALL = 0.25 * np.arange(GRID["Ni"])                      # the columns of the file, degrees east 0 .. 359.75
_J = np.where((LAT_ALL >= BOX["lat"][0]) & (LAT_ALL <= BOX["lat"][1]))[0]
_I = np.where((LON_ALL >= BOX["lon"][0] % 360) & (LON_ALL <= BOX["lon"][1] % 360))[0]
J0, J1, I0, I1 = int(_J[0]), int(_J[-1]) + 1, int(_I[0]), int(_I[-1]) + 1
NLAT, NLON = J1 - J0, I1 - I0
assert (NLAT, NLON) == (53, 65) and len(_J) == NLAT and len(_I) == NLON
LAT_BOX = LAT_ALL[J0:J1][::-1].copy()                       # stored south to north: 25.0 .. 38.0
LON_BOX = LON_ALL[I0:I1] - 360.0                            # stored west to east, degrees east in -180..180: -91.0 .. -75.0
FLAG_OK, FLAG_MISSING, FLAG_ERROR, FLAG_PENDING = 0, 1, 2, 255
FLAG_LEGEND = "0 = ok; 1 = missing on the server (HTTP 404); 2 = decode or integrity error; 255 = not fetched yet (partial month only)"
UNITS = "uint16, tenths of a millimetre of rain per 3-hour window (mm = value / 10); 65535 = no data (flag is not 0)"
NODATA = 65535
NEG_LIMIT_MM = 1.0                                          # a difference below this is not packing noise: the day is a decode error
WAITS = (1, 3, 10, 30, 60, 120, 300, 300, 300, 300)         # seconds before retry 1, 2, ... of one request (about 23 minutes in all)
DEADLINE = 900                                              # seconds for one request
DECODE_TRIES = 3                                            # a day that fails a check is fetched again twice before it is stored as an error
EARLY = (1, 10, 50)                                         # progress lines after this many days, then every 100
LINK_DEAD_AFTER = 24                                        # days in a row that failed on the network
MISSING_IN_A_ROW, ERRORS_IN_A_ROW = 60, 8                   # more than this many in a row means the key pattern or the decoder is wrong: stop
MSG_REQUIRED = dict(edition=2, shortName="tp", typeOfLevel="surface", level=0, stepType="accum", stepUnits=1, units="kg m**-2", dataDate=None, dataTime=0, bitmapPresent=0, **GRID)
MSG_OPTIONAL = ("bitsPerValue", "perturbationNumber", "packingType")
STOP = threading.Event()
OUT = ROOT / "data" / "gefs" / MEMBER
TMP = DONE = None


def ymd_of(d):
    return d.strftime("%Y%m%d")


def days_between(a, b):
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


ALL_DAYS = days_between(*FULL)
MONTHS = collections.OrderedDict()                          # "YYYYMM" -> the days of the dataset in that month
for _d in ALL_DAYS:
    MONTHS.setdefault(ymd_of(_d)[:6], []).append(_d)


def say(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


class Missing(Exception):
    """HTTP 404: the file is not on the server."""


class Bad(Exception):
    """The bytes are there but fail a check (unreadable .idx, wrong windows, wrong grid, ...)."""


class NetFail(Exception):
    """A request failed on every retry."""


class Stopped(Exception):
    pass


class Fatal(Exception):
    """Unrecoverable: the run stops with this message."""

    def __init__(self, msg, code=4):
        super().__init__(msg)
        self.code = code


# ---- the .idx and the windows ------------------------------------------------------------------------------------------------
Row = collections.namedtuple("Row", "no off init start end tail")
IDX_LINE = re.compile(r"^(\d+):(\d+):d=(\d{10}):APCP:surface:(\d+)-(\d+) hour acc fcst:(.*)$")


def parse_idx(text, ymd):
    """(all rows, the rows of the windows that end by hour 78, last byte of the last of them) from the text of a .idx; Bad if it is not what it was on 2010-01-01."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            m = IDX_LINE.match(line)
            if m is None:
                raise Bad(f"unreadable .idx line {line!r}")
            no, off, init, s, e, tail = m.groups()
            rows.append(Row(int(no), int(off), init, int(s), int(e), tail))
    if not rows or [r.no for r in rows] != list(range(1, len(rows) + 1)):
        raise Bad("the .idx message numbers are not 1..N")
    if rows[0].off != 0 or any(b.off <= a.off for a, b in zip(rows, rows[1:])):
        raise Bad("the .idx byte offsets do not start at 0 and increase")
    if any(r.init != ymd + "00" for r in rows):
        raise Bad("the .idx is for another run than the file name says")
    if any("ctl" not in r.tail for r in rows):
        raise Bad(f"the .idx is not the control member: {rows[0].tail!r}")
    need = [r for r in rows if r.end <= MAX_HOUR]
    if len(rows) <= len(need):
        raise Bad("no message after hour 78: the end of the byte range is unknown")
    size = {r.no: rows[i + 1].off - r.off for i, r in enumerate(rows[:-1])}
    uniq = []
    for r in need:
        if uniq and (uniq[-1].start, uniq[-1].end) == (r.start, r.end):      # 5 of the 5392 files hold some messages twice (2003-04-11 and 2013-11-09 inside the first 78 h)
            if size[r.no] != size[uniq[-1].no]:
                raise Bad(f"window {r.start}-{r.end} h is listed twice with different byte lengths")
            continue                                                         # an exact twin of the message before it: fetched, checked equal to it in decode_day, then ignored
        uniq.append(r)
    if [r.end for r in uniq] != STEPS.tolist():
        raise Bad(f"the windows ending by hour {MAX_HOUR} end at {[r.end for r in uniq]}, expected {STEPS.tolist()}")
    return rows, need, rows[len(need)].off - 1


def tile_windows(raw):
    """[(start h, end h, accumulation)] in file order -> [(start, end, amount)] that tile 0..end without gap or overlap.
    A window that starts before the covered end (6-12 after 6-9, 0-6 after 0-3) is differenced against the message before it."""
    out, covered, prev = [], 0, None
    for s, e, a in raw:
        if s == covered:
            piece = a
        elif s < covered:
            if prev is None or prev[0] != s or prev[1] != covered or e <= covered:
                raise Bad(f"window {s}-{e} h overlaps the covered 0-{covered} h in a way that cannot be differenced")
            piece = a - prev[2]
        else:
            raise Bad(f"gap: window {s}-{e} h starts after the covered 0-{covered} h")
        out.append((covered, e, piece))
        covered, prev = e, (s, e, a)
    return out


def to_units(pieces):
    """3-hourly amounts in mm, rows north to south -> (uint16 tenths of a mm, rows south to north; facts about the day)."""
    mm = np.stack(pieces)
    neg = mm < 0
    info = dict(min_mm_before_clamp=round(float(mm.min()), 5), n_negative_clamped=int(neg.sum()), sum_negative_mm=round(float(mm[neg].sum()), 4))
    if info["min_mm_before_clamp"] < -NEG_LIMIT_MM:
        raise Bad(f"a difference of {info['min_mm_before_clamp']} mm is not packing noise")
    mm = np.where(neg, 0.0, mm)
    top = float(mm.max())
    if not np.isfinite(top) or top * 10 > NODATA - 1:
        raise Bad(f"the largest 3-hourly amount, {top} mm, does not fit uint16 tenths of a mm")
    # The file's values are exact multiples of 0.01 mm. They are rounded to tenths of a mm, exact halves up, on the integer hundredths, so that the result does not depend on float
    # noise (np.rint(mm * 10) sent nearly all of the 8% of values that sit on a half (x.x5 mm) down: the day total of the box came out 0.28% low; half to even gives 0.38% low; half up +0.26%).
    u = ((np.rint(mm * 100.0).astype(np.int64) + 5) // 10).astype(np.uint16)
    assert int(u.max()) < NODATA
    info.update(max_mm_3h=round(top, 3), sum_mm_float=round(float(mm.sum()), 2), sum_mm_stored=round(float(u.sum(dtype=np.int64)) / 10, 1))
    return np.ascontiguousarray(u[:, ::-1, :]), info


# ---- http -------------------------------------------------------------------------------------------------------------------
_tls = threading.local()


def session():
    s = getattr(_tls, "s", None)
    if s is None:
        import requests
        s = _tls.s = requests.Session()
    return s


def nap(seconds):
    end = time.time() + seconds
    while time.time() < end:
        if STOP.is_set():
            raise Stopped()
        time.sleep(max(0.0, min(1.0, end - time.time())))


def http_get(url, byte_range=None, expect=None):
    """The bytes of url (of one byte range, expecting HTTP 206). 404 -> Missing. Anything else that goes wrong is retried with back-off, then NetFail."""
    import requests
    headers = {"Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    last = "?"
    for attempt in range(len(WAITS) + 1):
        if STOP.is_set():
            raise Stopped()
        t0 = time.time()
        try:
            with session().get(url, headers=headers, stream=True, timeout=(20, 60)) as r:
                if r.status_code == 404:
                    raise Missing(f"HTTP 404 for {url.rsplit('/', 1)[-1]}")
                if r.status_code not in (200, 206):
                    last = f"HTTP {r.status_code}"
                elif byte_range and r.status_code != 206:
                    last = "the server ignored the Range header (HTTP 200)"
                elif byte_range and not re.match(r"bytes %d-%d/\d+$" % tuple(byte_range), r.headers.get("Content-Range", "")):
                    last = f"unexpected Content-Range {r.headers.get('Content-Range')!r}"
                else:
                    buf = bytearray()
                    for chunk in r.iter_content(1 << 18):
                        buf += chunk
                        if STOP.is_set():
                            raise Stopped()
                        if time.time() - t0 > DEADLINE:
                            raise TimeoutError(f"more than {DEADLINE} s")
                    if expect is not None and len(buf) != expect:
                        last = f"short read: {len(buf)} of {expect} bytes"
                    else:
                        return bytes(buf)
        except (Missing, Stopped):
            raise
        except (requests.RequestException, OSError, TimeoutError) as e:
            last = repr(e)[:200]
        if attempt < len(WAITS):
            wait = WAITS[attempt] * (0.8 + 0.4 * random.random())
            say(f"  retry {attempt + 1}/{len(WAITS)} in {wait:.0f} s: {url.rsplit('/', 1)[-1]}: {last}")
            nap(wait)
    raise NetFail(f"{url}: {last}")


def fetch_day(day):
    """In a fetch thread: the .idx, then one Range request for messages 1..26. Never decodes."""
    ymd = ymd_of(day)
    url = f"{BUCKET}/" + KEY.format(year=day.year, ymd=ymd, member=MEMBER)
    g = SimpleNamespace(day=day, status="data", reason="", data=None, rows=None, need=None, end_byte=None, n_idx=0, seconds=0.0, when=None)
    t0 = time.time()
    try:
        idx = http_get(url + ".idx")
        g.n_idx = len(idx)
        g.rows, g.need, g.end_byte = parse_idx(idx.decode("ascii", "replace"), ymd)
        g.data = http_get(url, (0, g.end_byte), expect=g.end_byte + 1)
    except Missing as e:
        g.status, g.reason = "missing", str(e)
    except Bad as e:
        g.status, g.reason = "bad", str(e)
    g.seconds = time.time() - t0
    g.when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return g


# ---- decoding (main thread only: eccodes and cfgrib keep global state) ---------------------------------------------------------
def decode_day(g, crosscheck=False, verbose=False):
    """(uint16 [26, 53, 65] tenths of a mm, rows south to north; facts about the day) from the fetched bytes. Raises Bad when any check fails."""
    from cfgrib import messages as cfm
    ymd, need, data = ymd_of(g.day), g.need, g.data
    offs = [r.off for r in need] + [g.end_byte + 1]
    if len(data) != g.end_byte + 1:
        raise Bad(f"{len(data)} bytes fetched, {g.end_byte + 1} expected")
    for k in range(len(need)):
        if data[offs[k]:offs[k] + 4] != b"GRIB" or data[offs[k + 1] - 4:offs[k + 1]] != b"7777":
            raise Bad(f"message {k + 1} is not a whole GRIB message at the .idx offsets")
    t0 = time.time()
    got = []
    with tempfile.NamedTemporaryFile(dir=TMP, prefix="gefs_", suffix=".grib2") as tf:
        tf.write(data)
        tf.flush()
        for _, m in cfm.FileStream(tf.name, errors="raise").items():      # read every message to the end; judge them afterwards
            keys = {}
            for k in list(MSG_REQUIRED) + ["startStep", "endStep", "validityDate", "validityTime", "totalLength"]:
                try:
                    keys[k] = m[k]
                except KeyError:
                    keys[k] = "<missing key>"
            for k in MSG_OPTIONAL:
                try:
                    keys[k] = m[k]
                except KeyError:
                    keys[k] = None
            v = m["values"]
            got.append((keys, v.reshape(GRID["Nj"], GRID["Ni"])[J0:J1, I0:I1].copy() if v.size == GRID["Nj"] * GRID["Ni"] else None))
        if len(got) != len(need):
            raise Bad(f"{len(got)} messages decoded, {len(need)} expected")
        raw, n_twins = [], 0
        for k, ((keys, box), r) in enumerate(zip(got, need)):
            for name, want in MSG_REQUIRED.items():
                want = int(ymd) if name == "dataDate" else want
                if keys[name] != want:
                    raise Bad(f"message {k + 1}: {name} = {keys[name]!r}, expected {want!r}")
            if keys["perturbationNumber"] not in (None, 0):
                raise Bad(f"message {k + 1} is not the control member (perturbationNumber {keys['perturbationNumber']})")
            if (keys["startStep"], keys["endStep"]) != (r.start, r.end):
                raise Bad(f"message {k + 1}: the file says window {keys['startStep']}-{keys['endStep']} h, the .idx says {r.start}-{r.end} h")
            valid = dt.datetime.strptime(ymd, "%Y%m%d") + dt.timedelta(hours=keys["endStep"])
            if (keys["validityDate"], keys["validityTime"]) != (int(valid.strftime("%Y%m%d")), valid.hour * 100):
                raise Bad(f"message {k + 1}: valid time {keys['validityDate']} {keys['validityTime']} is not the run plus {keys['endStep']} h")
            if keys["totalLength"] != offs[k + 1] - offs[k]:
                raise Bad(f"message {k + 1}: {keys['totalLength']} bytes in the file, {offs[k + 1] - offs[k]} in the .idx")
            if box is None or box.shape != (NLAT, NLON):
                raise Bad(f"message {k + 1}: the field does not have the {GRID['Nj']} x {GRID['Ni']} grid")
            if not np.isfinite(box).all() or float(box.max()) > 5000.0 or float(box.min()) < -1.0:
                raise Bad(f"message {k + 1}: values outside -1 .. 5000 mm in the box")
            if raw and (raw[-1][0], raw[-1][1]) == (keys["startStep"], keys["endStep"]):
                if not np.array_equal(box, raw[-1][2]):
                    raise Bad(f"message {k + 1} repeats window {r.start}-{r.end} h with different values")
                n_twins += 1
                continue
            raw.append((keys["startStep"], keys["endStep"], box.astype(np.float64)))
        if crosscheck and not n_twins:                      # xarray cannot build its cube from a file that holds a window twice
            crosscheck_xarray(tf.name, raw)
    tiles = tile_windows(raw)
    if [(s, e) for s, e, _ in tiles] != [(STEP_H * k, STEP_H * (k + 1)) for k in range(NSTEP)]:
        raise Bad("the differenced windows do not tile 0..78 h in 3-hour steps")
    u, info = to_units([p for _, _, p in tiles])
    bits = [k["bitsPerValue"] for k, _ in got if k["bitsPerValue"] is not None]
    info.update(messages=len(got), twin_messages_ignored=n_twins, ens=need[0].tail, packing=got[0][0]["packingType"], bits_per_value=[min(bits), max(bits)] if bits else None,
                seconds_decode=round(time.time() - t0, 2), xarray_crosscheck=bool(crosscheck and not n_twins))
    if verbose:
        say(f"  {ymd}  messages  {' '.join(f'{s}-{e}' for s, e, _ in raw)}")
        say(f"  {ymd}  3-hourly  {' '.join(f'{s}-{e}' for s, e, _ in tiles)}")
        say(f"  {ymd}  stored {u.shape} = (step, lat, lon); lat {LAT_BOX[0]}..{LAT_BOX[-1]} south to north, lon {LON_BOX[0]}..{LON_BOX[-1]}; valid times "
            f"{got[0][0]['validityDate']} {got[0][0]['validityTime']:04d} .. {got[-1][0]['validityDate']} {got[-1][0]['validityTime']:04d} UTC; "
            f"min before clamp {info['min_mm_before_clamp']} mm ({info['n_negative_clamped']} values clamped), largest 3-h amount {info['max_mm_3h']} mm; {info['seconds_decode']} s")
    return u, info


def crosscheck_xarray(path, raw):
    """The same file through xarray's cfgrib engine (indexpath='' writes no index): the same windows, coordinates and fields in the box."""
    import xarray as xr
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        tp = ds["tp"]
        ends = (tp.step.values / np.timedelta64(1, "h")).astype(int).tolist()
        if ends != [e for _, e, _ in raw]:
            raise Fatal(f"xarray cross-check: window ends {ends} differ from the message-level decode")
        if not (np.array_equal(ds.latitude.values[J0:J1], LAT_ALL[J0:J1]) and np.array_equal(ds.longitude.values[I0:I1], LON_ALL[I0:I1])):
            raise Fatal("xarray cross-check: its latitude / longitude differ from the grid assumed here")
        box = tp.isel(latitude=slice(J0, J1), longitude=slice(I0, I1)).values
        worst = max(float(np.abs(box[k] - a).max()) for k, (_, _, a) in enumerate(raw))
        if worst > 1e-3:
            raise Fatal(f"xarray cross-check: the fields differ by up to {worst} mm from the message-level decode")


# ---- the monthly files ------------------------------------------------------------------------------------------------------
def month_of(day):
    return ymd_of(day)[:6]


def atomic_write(path, writer):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Month:
    """One month file in memory: the arrays, the flags and the per-day facts. save() rewrites the .npz and the .json atomically."""

    def __init__(self, ym):
        self.ym, self.days = ym, MONTHS[ym]
        n = len(self.days)
        self.init_date = np.array([int(ymd_of(d)) for d in self.days], np.int32)
        self.flag = np.full(n, FLAG_PENDING, np.uint8)
        self.apcp = np.full((n, NSTEP, NLAT, NLON), NODATA, np.uint16)
        self.meta = {}
        self.npz, self.json = OUT / f"apcp_{ym}.npz", OUT / f"apcp_{ym}.json"
        if self.npz.exists():
            with np.load(self.npz) as z:
                problems = check_arrays(z, self.init_date)
                if problems:
                    raise Fatal(f"{self.npz.name} is inconsistent ({problems[0]}); run --check, move the file away to fetch the month again")
                self.flag, self.apcp = z["flag"].copy(), z["apcp"].copy()
            if self.json.exists():
                self.meta = json.loads(self.json.read_text())["days"]

    @property
    def complete(self):
        return bool((self.flag != FLAG_PENDING).all())

    def set_day(self, day, flag, arr, meta):
        i = self.days.index(day)
        self.flag[i] = flag
        self.apcp[i] = NODATA if arr is None else arr
        self.meta[ymd_of(day)] = dict(meta, flag=int(flag))

    def save(self):
        arrays = dict(init_date=self.init_date, step_end_hours=STEPS.astype(np.int16), lat=LAT_BOX, lon=LON_BOX, apcp=self.apcp, flag=self.flag,
                      flag_legend=np.array(FLAG_LEGEND), units=np.array(UNITS))
        save = np.savez_compressed if self.complete else np.savez          # a month in progress is rewritten after every day: no compression until it is complete
        atomic_write(self.npz, lambda f: save(f, **arrays))
        doc = dict(what=f"GEFSv12 reforecast total precipitation, control member {MEMBER}, runs of 00 UTC on the days of {self.ym}, reduced to the south-eastern box (code/fetch_gefs.py)",
                   month=self.ym, complete=self.complete, member=MEMBER, variable="apcp_sfc (total precipitation, kg m-2 = mm)", units=UNITS, flag_legend=FLAG_LEGEND,
                   source=dict(bucket=BUCKET, key_pattern=KEY, idx="the key + '.idx' (byte offset of each of the 80 messages)",
                               fetched="one HTTP Range request per day from byte 0 to the end of the message whose window ends at hour 78 (messages 1..26), after the .idx"),
                   grid=dict(lat=list(BOX["lat"]), lon=list(BOX["lon"]), step_deg=0.25, shape_lat_lon=[NLAT, NLON], lat_order="south to north",
                             lon_order="west to east", lon_convention="degrees east -180..180 (the source has 0..360: 269.0 .. 285.0)", source_grid="1440 x 721, 0.25 degree, rows north to south"),
                   windows="step k = 0..25 holds the rain of the 3-hour window (3k, 3k+3] hours after the 00 UTC run; step_end_hours = 3, 6, ..., 78; the file's 6 h accumulations "
                           "(0-6, 6-12, ...) were differenced against the 3 h accumulation before them",
                   clamp=f"negative 3-hourly differences (packing noise) set to 0; a day with a difference below -{NEG_LIMIT_MM} mm is stored as a decode error",
                   software=SOFTWARE, script=dict(file="code/fetch_gefs.py", sha256_16=SCRIPT_HASH), written_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   counts={str(k): int((self.flag == k).sum()) for k in (FLAG_OK, FLAG_MISSING, FLAG_ERROR, FLAG_PENDING)},
                   bytes_fetched=sum(d.get("bytes_idx", 0) + d.get("bytes_range", 0) for d in self.meta.values()), days=dict(sorted(self.meta.items())))
        atomic_write(self.json, lambda f: f.write(json.dumps(doc, indent=1).encode()))


def check_arrays(z, init_expected):
    """Problems (a list of texts) of an opened month file against the layout of this script; [] when it is as expected."""
    p, n = [], len(init_expected)
    need = ("init_date", "step_end_hours", "lat", "lon", "apcp", "flag", "flag_legend", "units")
    missing = [k for k in need if k not in z.files]
    if missing:
        return [f"arrays missing: {missing}"]
    init, steps, lat, lon, flag = z["init_date"], z["step_end_hours"], z["lat"], z["lon"], z["flag"]
    if not np.array_equal(init, init_expected):
        p.append(f"init_date is not the {n} days of the month")
    if not np.array_equal(steps, STEPS):
        p.append("step_end_hours is not 3, 6, ..., 78")
    if not (np.array_equal(lat, LAT_BOX) and np.array_equal(lon, LON_BOX)):
        p.append("lat / lon are not the box (25..38 south to north, -91..-75)")
    if flag.dtype != np.uint8 or flag.shape != (n,) or not np.isin(flag, (FLAG_OK, FLAG_MISSING, FLAG_ERROR, FLAG_PENDING)).all():
        p.append("flag is not uint8 [day] with values 0, 1, 2, 255")
    shp = z["apcp"].shape if "apcp" in z.files else None
    if shp != (n, NSTEP, NLAT, NLON):
        p.append(f"apcp has shape {shp}, expected {(n, NSTEP, NLAT, NLON)}")
    return p


def load_gefs(first="2000-01-01", last="2014-10-05", out=None):
    """The stored days between two dates as (init_date YYYYMMDD int32 [n], flag uint8 [n], apcp uint16 [n, 26, 53, 65] tenths of a mm per 3 h window (65535 = no data), lat, lon,
    step_end_hours). Days not fetched yet have flag 255. Numpy only. Convert with  mm = apcp / 10.0  where flag == 0."""
    out = Path(out) if out else OUT
    a, b = int(first.replace("-", "")), int(last.replace("-", ""))
    inits, flags, arrs = [], [], []
    for ym in MONTHS:
        if not (int(ym + "01") <= b and int(ym + "31") >= a):
            continue
        f = out / f"apcp_{ym}.npz"
        if f.exists():
            with np.load(f) as z:
                inits.append(z["init_date"]), flags.append(z["flag"]), arrs.append(z["apcp"])
        else:
            inits.append(np.array([int(ymd_of(d)) for d in MONTHS[ym]], np.int32))
            flags.append(np.full(len(MONTHS[ym]), FLAG_PENDING, np.uint8))
            arrs.append(np.full((len(MONTHS[ym]), NSTEP, NLAT, NLON), NODATA, np.uint16))
    init, flag, apcp = np.concatenate(inits), np.concatenate(flags), np.concatenate(arrs)
    keep = (init >= a) & (init <= b)
    return init[keep], flag[keep], apcp[keep], LAT_BOX.copy(), LON_BOX.copy(), STEPS.copy()


def stored_days():
    """{YYYYMMDD: flag} of every day that a month file holds (flag 255 = not fetched yet is left out). Reads only the two small arrays of each file."""
    out = {}
    for ym in MONTHS:
        f = OUT / f"apcp_{ym}.npz"
        if f.exists():
            with np.load(f) as z:
                for d, fl in zip(z["init_date"].tolist(), z["flag"].tolist()):
                    if fl != FLAG_PENDING:
                        out[str(d)] = int(fl)
    return out


def write_done(stored):
    atomic_write(DONE, lambda f: f.write("".join(f"{d} {fl}\n" for d, fl in sorted(stored.items())).encode()))


# ---- --check ------------------------------------------------------------------------------------------------------------------
def check():
    problems, warns, tot = [], [], collections.Counter()
    stored, firstday_mm, max3h, zero_days = {}, [], 0.0, []
    n_present = n_complete = 0
    for ym, days in MONTHS.items():
        npz, js = OUT / f"apcp_{ym}.npz", OUT / f"apcp_{ym}.json"
        init_expected = np.array([int(ymd_of(d)) for d in days], np.int32)
        if not npz.exists():
            tot[FLAG_PENDING] += len(days)
            if js.exists():
                problems.append(f"{js.name} exists without {npz.name}")
            continue
        try:
            with np.load(npz) as z:
                bad = check_arrays(z, init_expected)
                if bad:
                    problems += [f"{npz.name}: {b}" for b in bad]
                    continue
                flag, apcp = z["flag"], z["apcp"]
                n_present += 1
                n_complete += bool((flag != FLAG_PENDING).all())
                for i, d in enumerate(init_expected.tolist()):
                    fl = int(flag[i])
                    tot[fl] += 1
                    if fl != FLAG_PENDING:
                        stored[str(d)] = fl
                    day = apcp[i]
                    if fl == FLAG_OK:
                        if int(day.max()) >= NODATA:
                            problems.append(f"{npz.name}: day {d} is flagged ok but holds the no-data value")
                            continue
                        if int(day.max()) == 0:
                            zero_days.append(d)
                        firstday_mm.append(float(day[:8].sum(dtype=np.int64)) / 10 / (NLAT * NLON))      # box-mean rain of the first 24 h (steps 3..24), mm
                        max3h = max(max3h, int(day.max()) / 10)
                    elif not (day == NODATA).all():
                        problems.append(f"{npz.name}: day {d} has flag {fl} but holds data")
        except Exception as e:
            problems.append(f"{npz.name}: unreadable ({e!r})")
            continue
        try:
            doc = json.loads(js.read_text())
        except Exception as e:
            problems.append(f"{js.name}: missing or unreadable ({e!r})")
            continue
        meta = doc.get("days", {})
        if doc.get("month") != ym or doc.get("member") != MEMBER:
            problems.append(f"{js.name}: month / member fields wrong")
        for d, fl in stored.items():
            if d[:6] == ym and (d not in meta or meta[d].get("flag") != fl):
                problems.append(f"{js.name}: day {d} (flag {fl}) is not described correctly in the sidecar")
        if bool(doc.get("complete")) != bool((flag != FLAG_PENDING).all()):
            problems.append(f"{js.name}: its 'complete' field disagrees with the flags")
    listed = {}
    if DONE.exists():
        for line in DONE.read_text().splitlines():
            if line.strip():
                d, fl = line.split()
                listed[d] = int(fl)
    if listed != stored:
        problems.append(f"{DONE.name} disagrees with the month files: {len(set(listed) - set(stored))} listed days not stored, {len(set(stored) - set(listed))} stored days not listed "
                        f"(a start of the fetcher rebuilds it)")
    n_days = len(ALL_DAYS)
    print(f"check of {OUT}: {len(MONTHS)} months expected ({n_days} days, {FULL[0]} .. {FULL[1]})")
    print(f"  month files: {n_present} present ({n_complete} complete, {n_present - n_complete} partial), {len(MONTHS) - n_present} absent")
    print(f"  days: ok {tot[FLAG_OK]}, missing on the server {tot[FLAG_MISSING]}, decode error {tot[FLAG_ERROR]}, not fetched yet {tot[FLAG_PENDING]} (total {sum(tot.values())} of {n_days})")
    if firstday_mm:
        print(f"  ok days: box-mean rain of the first 24 h averages {np.mean(firstday_mm):.2f} mm/day (range {min(firstday_mm):.2f} .. {max(firstday_mm):.2f}); largest 3-hourly amount {max3h:.1f} mm; "
              f"{len(zero_days)} days with no rain at all in the box")
    if zero_days:
        warns.append(f"days with no rain at all in the box: {zero_days[:10]}{' ...' if len(zero_days) > 10 else ''}")
    for w in warns:
        print("  note:", w)
    if tot[FLAG_PENDING]:
        print(f"  INCOMPLETE: {tot[FLAG_PENDING]} days are not fetched yet (run the fetcher again to continue)")
    for p in problems:
        print("  PROBLEM:", p)
    print("  RESULT:", "consistent" if not problems else f"{len(problems)} problem(s)", "-", "complete" if not tot[FLAG_PENDING] and not problems else "not complete")
    return 1 if problems else 0


# ---- the run ------------------------------------------------------------------------------------------------------------------
SOFTWARE, SCRIPT_HASH = {}, ""


def collect_software():
    global SOFTWARE, SCRIPT_HASH
    for name in ("numpy", "eccodes", "eccodeslib", "cfgrib", "xarray", "requests", "urllib3"):
        try:
            SOFTWARE[name] = imd.version(name)
        except imd.PackageNotFoundError:
            SOFTWARE[name] = None
    try:
        import eccodes
        SOFTWARE["eccodes_library"] = eccodes.codes_get_api_version()
    except Exception:
        SOFTWARE["eccodes_library"] = None
    SOFTWARE["python"] = sys.version.split()[0]
    SOFTWARE["platform"] = platform.platform()
    SCRIPT_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def ensure_proxy():
    if any(os.environ.get(k) for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")):
        return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    try:
        socket.create_connection(("127.0.0.1", 7890), timeout=1).close()
        os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
        return "http://127.0.0.1:7890 (found listening; no proxy variable was set)"
    except OSError:
        say("WARNING: no proxy variable is set and nothing listens on 127.0.0.1:7890; direct reads of the bucket gave 20 KB/s on 2026-10-07")
        return None


def take_lock():
    import fcntl                                            # here, not at the top: Windows has no fcntl, and only the downloader needs the lock
    f = open(OUT / ".fetch.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise Fatal("another fetch_gefs.py is already running on this directory (its lock is held)")
    f.write(f"{os.getpid()}\n")
    f.flush()
    return f


class Run:
    """The state of one invocation: counters, the open months, the progress lines."""

    def __init__(self, todo, verbose):
        self.todo_total, self.verbose = len(todo), verbose
        self.t0 = time.monotonic()
        self.n = self.ok = self.missing = self.error = self.netfail = 0
        self.bytes = 0
        self.errors_in_a_row = self.missing_in_a_row = self.net_in_a_row = 0
        self.open = {}
        self.stored = stored_days()
        self.marks = collections.deque([(self.t0, 0, 0)], maxlen=4)
        self.crosschecked = False
        self.failed = []

    def month(self, day):
        ym = month_of(day)
        if ym not in self.open:
            self.open[ym] = Month(ym)
        return self.open[ym]

    def store(self, g, flag, arr, meta):
        day = g.day
        mo = self.month(day)
        mo.set_day(day, flag, arr, dict(meta, fetched_utc=g.when, bytes_idx=g.n_idx, bytes_range=len(g.data) if g.data else 0, seconds_fetch=round(g.seconds, 2)))
        mo.save()
        with open(DONE, "a") as f:
            f.write(f"{ymd_of(day)} {flag}\n")
        self.stored[ymd_of(day)] = int(flag)
        self.n += 1
        if mo.complete:
            c = collections.Counter(mo.flag.tolist())
            say(f"month {mo.ym} complete: {len(mo.days)} days (ok {c[FLAG_OK]}, missing {c[FLAG_MISSING]}, error {c[FLAG_ERROR]}); {mo.npz.stat().st_size / 1e6:.1f} MB on disk; "
                f"stored {len(self.stored)}/{len(ALL_DAYS)} days")
            del self.open[mo.ym]
        if self.n in EARLY or self.n % 100 == 0:
            self.progress()

    def progress(self):
        now = time.monotonic()
        el = now - self.t0
        t_old, n_old, b_old = self.marks[0]
        recent_dps = (self.n - n_old) / max(now - t_old, 1e-9) if self.n > n_old else self.n / max(el, 1e-9)
        recent_mbs = (self.bytes - b_old) / 1e6 / max(now - t_old, 1e-9) if self.n > n_old else self.bytes / 1e6 / max(el, 1e-9)
        left = self.todo_total - self.n
        eta = left / recent_dps if recent_dps > 0 else float("nan")
        done_at = time.strftime("%a %H:%M", time.localtime(time.time() + eta)) if eta == eta else "?"
        say(f"progress: {self.n} days this run (ok {self.ok}, missing {self.missing}, decode errors {self.error}) | stored {len(self.stored)}/{len(ALL_DAYS)} | "
            f"{self.bytes / 1e6:.0f} MB, {self.bytes / 1e6 / max(el, 1e-9):.2f} MB/s overall, {recent_mbs:.2f} recent | {1 / recent_dps if recent_dps else float('nan'):.1f} s/day | "
            f"to go {left} days, about {eta / 3600:.1f} h (finish about {done_at}) | network failures {self.netfail} | elapsed {el / 3600:.2f} h")
        self.marks.append((now, self.n, self.bytes))

    def handle(self, g, attempt, resubmit):
        """Decode and store the result of one fetched day (or fetch it again)."""
        self.bytes += g.n_idx + (len(g.data) if g.data else 0)
        day = ymd_of(g.day)
        if g.status == "missing":
            self.missing += 1
            self.missing_in_a_row += 1
            self.errors_in_a_row = 0
            say(f"{day}: missing on the server ({g.reason}); stored with flag {FLAG_MISSING}")
            self.store(g, FLAG_MISSING, None, dict(reason=g.reason))
            if self.missing_in_a_row >= MISSING_IN_A_ROW:
                raise Fatal(f"{MISSING_IN_A_ROW} days in a row are missing on the server: the key pattern or the proxy is probably wrong")
            return
        self.missing_in_a_row = 0
        reason, info, arr = g.reason, None, None
        if g.status == "data":
            try:
                arr, info = decode_day(g, crosscheck=not self.crosschecked or self.verbose, verbose=self.verbose)
                self.crosschecked = True
            except Fatal:
                raise
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                if not isinstance(e, Bad):
                    say(f"{day}: unexpected exception in the decoder:\n" + traceback.format_exc())
        if arr is not None:
            self.ok += 1
            self.errors_in_a_row = 0
            digest = hashlib.sha256(g.data).hexdigest()
            self.store(g, FLAG_OK, arr, dict(info, grib_sha256=digest, range=f"0-{g.end_byte}", messages_hours="0-78"))
            return
        if attempt + 1 < DECODE_TRIES:
            say(f"{day}: check failed ({reason}); fetching it again ({attempt + 2}/{DECODE_TRIES})")
            resubmit(g.day, attempt + 1)
            return
        self.error += 1
        self.errors_in_a_row += 1
        say(f"{day}: decode error after {DECODE_TRIES} fetches ({reason}); stored with flag {FLAG_ERROR}")
        self.store(g, FLAG_ERROR, None, dict(reason=reason))
        if self.errors_in_a_row >= ERRORS_IN_A_ROW:
            raise Fatal(f"{ERRORS_IN_A_ROW} days in a row failed the decoder checks: stopping; the cause must be fixed ({reason})")

    def run(self, days, threads):
        """Fetch and store `days` (in order) with `threads` fetch threads; returns the days that failed on the network (left not stored)."""
        failed, it, inflight = [], iter(days), {}
        window = threads * 3                                # days fetched or waiting to be decoded at any time: bounds the memory (9 MB each)
        ex = cf.ThreadPoolExecutor(threads, thread_name_prefix="fetch")

        def submit(day, attempt=0):
            inflight[ex.submit(fetch_day, day)] = (day, attempt)

        def refill():
            while len(inflight) < window and not STOP.is_set():
                day = next(it, None)
                if day is None:
                    return
                submit(day)

        try:
            refill()
            while inflight:
                done, _ = cf.wait(list(inflight), timeout=2, return_when=cf.FIRST_COMPLETED)
                if STOP.is_set():
                    for f in inflight:
                        f.cancel()
                for fut in done:
                    day, attempt = inflight.pop(fut)
                    try:
                        g = fut.result()
                    except (Stopped, cf.CancelledError):
                        continue
                    except NetFail as e:
                        self.netfail += 1
                        self.net_in_a_row += 1
                        failed.append(day)
                        say(f"{ymd_of(day)}: network failure, left for a later pass: {e}")
                        if self.net_in_a_row >= LINK_DEAD_AFTER:
                            raise Fatal(f"{LINK_DEAD_AFTER} days in a row failed on the network: the link or the proxy looks dead; run the fetcher again later", code=3)
                        continue
                    self.net_in_a_row = 0
                    self.handle(g, attempt, submit)
                refill()
        except BaseException:
            STOP.set()                                      # an error ends the run: the fetch threads give up their retries at once instead of working through the queue
            raise
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        return failed


def main():
    global OUT, TMP, DONE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, help="fetch only the first N days not yet stored")
    ap.add_argument("--start", help="first day to fetch, YYYY-MM-DD (default 2000-01-01)")
    ap.add_argument("--end", help="last day to fetch, YYYY-MM-DD (default 2014-10-05)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", help="data directory (default data/gefs/c00)")
    ap.add_argument("--check", action="store_true", help="verify the stored months; no download")
    ap.add_argument("--retry-errors", action="store_true", help="fetch the days stored with flag 2 again")
    ap.add_argument("--retry-missing", action="store_true", help="fetch the days stored with flag 1 again")
    ap.add_argument("--verbose", action="store_true", help="print the decoded windows of every day")
    ap.add_argument("--no-caffeinate", action="store_true")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in {OUT, OUT.parent}:
        (d / ".metadata_never_index").touch()               # keeps the monthly files out of the Spotlight index
    TMP, DONE = OUT / "tmp", OUT / "done_apcp.txt"
    collect_software()
    if args.check:
        return check()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: STOP.set())
    lock = take_lock()
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir()
    for f in OUT.glob("*.tmp"):
        f.unlink()                                          # a write that was cut off by a crash; the file it was to replace is intact
    try:
        os.nice(10)
    except OSError:
        pass
    if not args.no_caffeinate and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proxy = ensure_proxy()
    stored = stored_days()
    write_done(stored)
    first = dt.date.fromisoformat(args.start) if args.start else FULL[0]
    last = dt.date.fromisoformat(args.end) if args.end else FULL[1]
    if not (FULL[0] <= first <= last <= FULL[1]):
        raise Fatal(f"--start / --end must lie within {FULL[0]} .. {FULL[1]}")
    redo = ({FLAG_ERROR} if args.retry_errors else set()) | ({FLAG_MISSING} if args.retry_missing else set())
    todo = [d for d in days_between(first, last) if ymd_of(d) not in stored or stored[ymd_of(d)] in redo]
    if args.days is not None:
        todo = todo[:args.days]
    say(f"=== fetch_gefs.py start | pid {os.getpid()} | python {SOFTWARE['python']} | eccodes {SOFTWARE['eccodes']} cfgrib {SOFTWARE['cfgrib']} numpy {SOFTWARE['numpy']} | "
        f"script {SCRIPT_HASH}")
    say(f"    out {OUT} | stored {len(stored)}/{len(ALL_DAYS)} days | to fetch now {len(todo)} days"
        + (f" ({ymd_of(todo[0])} .. {ymd_of(todo[-1])})" if todo else "") + f" | {args.threads} threads | proxy {proxy} | nice {os.nice(0)}")
    if not todo:
        say("nothing to fetch: every requested day is stored")
        return 0
    run = Run(todo, args.verbose)
    code = 0
    try:
        failed = run.run(todo, args.threads)
        if failed and not STOP.is_set():
            say(f"second pass for {len(failed)} days that failed on the network")
            run.net_in_a_row = 0
            failed = run.run(failed, args.threads)
        if STOP.is_set():
            code = 130
        elif failed:
            code = 2
    except Fatal as e:
        say(f"FATAL: {e}")
        code = e.code
    el = time.monotonic() - run.t0
    left = len(ALL_DAYS) - len(stored_days())
    say(f"SUMMARY: exit {code} | {run.n} days stored this run in {el / 3600:.2f} h (ok {run.ok}, missing {run.missing}, decode errors {run.error}) | {run.bytes / 1e6:.0f} MB, "
        f"{run.bytes / 1e6 / max(el, 1e-9):.2f} MB/s | network failures {run.netfail} (left unstored: {len(failed) if code in (0, 2) else 'see above'}) | "
        f"dataset: {len(ALL_DAYS) - left}/{len(ALL_DAYS)} days stored, {left} to go")
    del lock
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as e:
        say(f"FATAL: {e}")
        sys.exit(e.code)
    except SystemExit:
        raise
    except BaseException:
        say("UNRECOVERABLE ERROR:\n" + traceback.format_exc())
        sys.exit(1)
