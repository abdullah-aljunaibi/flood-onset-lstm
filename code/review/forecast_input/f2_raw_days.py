"""E2 review, check (a): the stored GEFS box data against a decode of RAW archive days made here with the reviewer's own code (no code of fetch_gefs.py is used).
For each chosen day the script fetches the .idx of the control-member file in the open bucket https://noaa-gefs-retrospective.s3.amazonaws.com/ and ONE Range request for the messages up to forecast hour 78
(about 9.5 MB), decodes them with the eccodes Python module, rebuilds the non-overlapping 3-hourly amounts (the file holds 0-3, 0-6, 6-9, 6-12, ... accumulations: 3-6 = (0-6) - (0-3), ...), cuts the box
(lon -91..-75, lat 25..38; the file runs north to south with longitudes 0..360), turns it into tenths of a millimetre (exact halves rounded up) and compares it, value by value, with data/gefs/c00/apcp_YYYYMM.npz:
the units, the windows, the box orientation, the rounding and the clamp of the packing noise; for days whose .idx holds a message twice (2003-04-11, 2013-11-09) it also checks that the twins are identical.
Needs the GRIB environment (eccodes, numpy, requests) and the proxy of the shell:  ~/AI_Class/.venv-grib/bin/python code/review/forecast_input/f2_raw_days.py [YYYY-MM-DD ...]   (default: 2010-05-01 2013-11-09 2003-04-11).
Standalone on purpose (no pandas or torch needed). Writes f2_raw_days.json into $REVIEW_OUT. About 10 seconds per day; nothing is written into the project."""
import json, os, re, sys, tempfile, time
from pathlib import Path
import numpy as np
import requests
import eccodes

PROJ = str(Path(__file__).resolve().parents[3])
OUT = os.environ.get("REVIEW_OUT") or str(Path(tempfile.gettempdir()) / "e2_review")
os.makedirs(OUT, exist_ok=True)
BUCKET = "https://noaa-gefs-retrospective.s3.amazonaws.com"
KEY = "GEFSv12/reforecast/{y}/{ymd}00/c00/Days:1-10/apcp_sfc_{ymd}00_c00.grib2"
DAYS = sys.argv[1:] or ["2010-05-01", "2013-11-09", "2003-04-11"]
sess = requests.Session()


def get(url, headers=None):
    for k in range(6):
        try:
            r = sess.get(url, headers=headers, timeout=300)
            if r.status_code in (200, 206):
                return r.content
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(2 ** k)
    raise RuntimeError(f"{url}: {last}")


def idx_of(url):
    rows = []
    for line in get(url + ".idx").decode().strip().split("\n"):
        n, off, d, var, lev, win, ens = line.split(":")[:7]
        m = re.match(r"(\d+)-(\d+) hour acc fcst", win)
        rows.append(dict(n=int(n), off=int(off), a=int(m.group(1)), b=int(m.group(2)), var=var, ens=ens))
    return rows


def decode(raw):
    """[(startStep, endStep, validity as YYYYMMDDHHMM int, values [721, 1440] float64, Ni, Nj, jScansPositively, iScansNegatively, first lat, first lon, shortName)] for every message of the bytes."""
    path = Path(tempfile.gettempdir()) / f"e2_review_{os.getpid()}.grib"
    path.write_bytes(raw)
    out = []
    try:
        with open(path, "rb") as f:
            while True:
                h = eccodes.codes_grib_new_from_file(f)
                if h is None:
                    break
                g = lambda k: eccodes.codes_get(h, k)
                vals = eccodes.codes_get_values(h).astype(np.float64)
                out.append(dict(start=int(g("startStep")), end=int(g("endStep")), valid=int(g("validityDate")) * 10000 + int(g("validityTime")), date=int(g("dataDate")), time=int(g("dataTime")),
                                Ni=int(g("Ni")), Nj=int(g("Nj")), jpos=int(g("jScansPositively")), ineg=int(g("iScansNegatively")), lat1=float(g("latitudeOfFirstGridPointInDegrees")),
                                lon1=float(g("longitudeOfFirstGridPointInDegrees")), short=str(g("shortName")), unit=str(g("stepUnits")), values=vals.reshape(int(g("Nj")), int(g("Ni")))))
                eccodes.codes_release(h)
    finally:
        path.unlink(missing_ok=True)
    return out


def stored_day(ymd):
    z = np.load(f"{PROJ}/data/gefs/c00/apcp_{ymd[:6]}.npz")
    i = int(np.flatnonzero(z["init_date"] == int(ymd))[0])
    assert int(z["flag"][i]) == 0
    return z["apcp"][i], z["lat"], z["lon"], z["step_end_hours"]


results = {}
for day in DAYS:
    ymd = day.replace("-", "")
    url = f"{BUCKET}/{KEY.format(y=ymd[:4], ymd=ymd)}"
    t0 = time.time()
    rows = idx_of(url)
    keep = [r for r in rows if r["b"] <= 78]
    last_off = min(r["off"] for r in rows if r["b"] > 78)                         # the first byte of the first message after hour 78
    # twins: the same window twice in the .idx
    from collections import Counter
    cnt = Counter((r["a"], r["b"]) for r in keep)
    twins = sorted(w for w, c in cnt.items() if c > 1)
    raw = get(url, headers={"Range": f"bytes=0-{last_off - 1}"})
    assert len(raw) == last_off, (len(raw), last_off)
    msgs = decode(raw)
    assert len(msgs) == len(keep), (len(msgs), len(keep))
    # every message: window, grid, valid time
    run_time = int(ymd) * 10000
    bad = []
    by_win = {}
    for m, r in zip(msgs, keep):
        ok = (m["start"], m["end"]) == (r["a"], r["b"]) and m["short"] == "tp" and (m["Ni"], m["Nj"], m["jpos"], m["ineg"]) == (1440, 721, 0, 0) and (m["lat1"], m["lon1"]) == (90.0, 0.0) and m["date"] == int(ymd) and m["time"] == 0
        if not ok:
            bad.append((r["n"], r["a"], r["b"]))
        by_win.setdefault((m["start"], m["end"]), []).append(m["values"])
    twins_identical = {f"{a}-{b}": bool(all(np.array_equal(v[0], x) for x in v[1:])) for (a, b), v in by_win.items() if len(v) > 1}
    # the box (rows run north to south: lat = 90 - 0.25 j; columns lon = 0.25 i)
    j = np.arange(721); i = np.arange(1440)
    lat_all, lon_all = 90.0 - 0.25 * j, 0.25 * i
    J = np.flatnonzero((lat_all >= 25.0) & (lat_all <= 38.0)); I = np.flatnonzero((lon_all >= 269.0) & (lon_all <= 285.0))
    box = lambda v: v[J[0]:J[-1] + 1, I[0]:I[-1] + 1][::-1]                      # south to north, west to east
    assert box(np.zeros((721, 1440))).shape == (53, 65)
    acc = {w: box(v[0]) for w, v in by_win.items()}
    # the non-overlapping 3-hourly windows
    win3 = []
    for k in range(26):
        a, b = 3 * k, 3 * k + 3
        if (a, b) in acc and (a % 6 == 0):
            w = acc[(a, b)]                                                       # the 3 h message of the block (0-3, 6-9, ...)
        else:
            w = acc[(a - 3, b)] - acc[(a - 3, a)]                                 # the 6 h message minus the 3 h message before it (3-6 = (0-6) - (0-3))
        win3.append(w)
    win3 = np.stack(win3)                                                         # [26, 53, 65] mm, float64
    neg = win3 < 0
    clamped = np.where(neg, 0.0, win3)
    hund = np.rint(clamped * 100.0)                                               # the file's values are exact hundredths of a mm; float noise (0.04999999999999982) is removed first
    off_grid = float(np.abs(clamped * 100.0 - hund).max())                        # how far the differences are from whole hundredths (float round-off only)
    mine = ((hund + 5) // 10).astype(np.int64)                                    # tenths of a mm, exact halves (x.x5) rounded up
    naive = np.floor(clamped * 10.0 + 0.5).astype(np.int64)                       # the same without removing the noise: x.x5 that comes out as x.x4999999 is rounded down
    st, lat, lon, ends = stored_day(ymd)
    st = st.astype(np.int64)
    diff = mine - st
    # tiling and sums
    results[day] = dict(
        seconds=round(time.time() - t0, 1), bytes=int(len(raw)), n_messages=len(msgs), windows_up_to_78=len(set(by_win)), twin_windows_in_the_idx=[f"{a}-{b}" for a, b in twins], twins_identical=twins_identical,
        message_checks_failed=bad, lat_range_stored=[float(lat[0]), float(lat[-1])], lon_range_stored=[float(lon[0]), float(lon[-1])], step_end_hours_stored=[int(ends[0]), int(ends[-1])],
        n_values=int(diff.size), n_values_equal=int((diff == 0).sum()), n_values_different=int((diff != 0).sum()), max_abs_difference_tenths=int(np.abs(diff).max()), shares_of_different=float((diff != 0).mean()),
        largest_distance_of_the_differences_from_whole_hundredths=off_grid, naive_rounding_without_noise_removal=dict(n_values_different=int((naive != st).sum()), all_of_them_exact_halves=bool(((naive != st) & (np.abs(clamped * 100.0 - hund) < 1e-9) & (hund % 10 == 5)).sum() == (naive != st).sum())),
        n_negative_differences_clamped=int(neg.sum()), most_negative_mm=float(win3.min()), box_total_mm_unrounded=float(clamped.sum()), box_total_mm_stored=float(st.sum() / 10.0), box_total_mm_mine_rounded=float(mine.sum() / 10.0),
        max_3h_mm=float(clamped.max()), wettest_window_step=int(np.unravel_index(clamped.argmax(), clamped.shape)[0]), boxes_wet_share=float((st > 0).mean()),
        orientation_check=dict(note="the wettest cell of the whole 78 h total in my south-to-north box and in the stored array", mine=[int(x) for x in np.unravel_index(clamped.sum(0).argmax(), (53, 65))], stored=[int(x) for x in np.unravel_index(st.sum(0).argmax(), (53, 65))],
                               flipped_would_be=[int(x) for x in np.unravel_index(clamped[:, ::-1].sum(0).argmax(), (53, 65))]),
        flipped_box_would_differ=int(((np.floor(clamped[:, ::-1] * 10.0 + 0.5).astype(np.int64) - st) != 0).sum()),
        different_values_sample=[dict(step=int(k), lat_i=int(a), lon_i=int(b), mine=int(mine[k, a, b]), stored=int(st[k, a, b]), raw_mm=float(win3[k, a, b])) for k, a, b in list(zip(*np.nonzero(diff)))[:6]],
        windows_tile_0_to_78=bool(sorted((a, b) for a, b in by_win if b % 6 == 0 and a % 6 == 0) == [(6 * i, 6 * i + 6) for i in range(13)] and sorted((a, b) for a, b in by_win if b - a == 3 and a % 6 == 0) == [(6 * i, 6 * i + 3) for i in range(13)]))
    print(f"{day}: {len(msgs)} messages ({len(raw) / 1e6:.1f} MB, {results[day]['seconds']} s); twins in the .idx: {results[day]['twin_windows_in_the_idx']} identical: {twins_identical}; stored values equal to my decode: "
          f"{results[day]['n_values_equal']} of {results[day]['n_values']} (different: {results[day]['n_values_different']}, largest difference {results[day]['max_abs_difference_tenths']} tenth of a mm; the plain float rounding would differ in {results[day]['naive_rounding_without_noise_removal']['n_values_different']} values, all of them exact halves: {results[day]['naive_rounding_without_noise_removal']['all_of_them_exact_halves']}); box total stored {results[day]['box_total_mm_stored']:.1f} mm, mine {results[day]['box_total_mm_mine_rounded']:.1f} mm; "
          f"clamped negatives {results[day]['n_negative_differences_clamped']} (most negative {results[day]['most_negative_mm']:.3f} mm); wettest cell mine {results[day]['orientation_check']['mine']} stored {results[day]['orientation_check']['stored']}; "
          f"a flipped box would differ in {results[day]['flipped_box_would_differ']} values", flush=True)
with open(f"{OUT}/f2_raw_days.json", "w") as f:
    json.dump(results, f, indent=1)
