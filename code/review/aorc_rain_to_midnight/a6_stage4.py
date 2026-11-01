"""AORC review, check (d) data: hourly NCEP Stage IV rain (the radar + gauge multi-sensor analysis that is available about an hour after the hour: what a real-time service would read) reduced to hourly
basin means for the 12 study basins, with the reviewer's own code (the builder's reducer in code/fetch_aorc.py is not used). Source: the Iowa Environmental Mesonet archive,
https://mesonet.agron.iastate.edu/archive/data/YYYY/MM/DD/stage4/ST4.YYYYMMDDHH.01h.grib (the file labelled T holds the hour that ends at T; checked here from the file's own tags GRIB_VALID_TIME and
GRIB_FORECAST_SECONDS). Each file (about 1.1 MB) is fetched with at most 6 parallel connections, read with rasterio, reduced to 12 basin means AT ONCE and dropped; only the means are kept (CACHE/stage4_means.csv, resume-safe).
Basin mask: the cells whose centre lies inside the CAMELS outline (data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp, read with pyshp), centres computed from the file's own affine transform and
coordinate reference system; for a basin that holds no cell centre, the cell holding the outline's centroid. 9999 (no radar coverage) is left out of the mean.
Usage:  a6_stage4.py days DAY1 DAY2 ...   fetch the hours of local day t (00:00 of day t) to 07:00 of day t+1 (labels 05:00 UTC of t to 13:00 UTC of t+1) for each day
        a6_stage4.py labels 'YYYY-MM-DD HH:00' ...   fetch these hour-ending labels
        a6_stage4.py after DAY1 DAY2 ...    fetch the five hours 13:00-17:00 UTC of day t+1 (after the gauge day ends at 12 UTC, up to about 12:00 local: the placebo regressor and the 00:00-12:00 window of a7_hindsight.py)
        a6_stage4.py check   compare six labels with the builder's data/aorc/stage_iv_hindsight.csv
"""
import sys; sys.dont_write_bytecode = True
import tempfile, time, threading, math
from concurrent.futures import ThreadPoolExecutor
from a_common import *
import requests, shapefile, pyproj, rasterio, shapely
from shapely.geometry import shape
import flood as F

URL = "https://mesonet.agron.iastate.edu/archive/data/{y}/{m}/{d}/stage4/ST4.{y}{m}{d}{h}.01h.grib"
SHP = f"{PROJ}/data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp"
CSV = f"{CACHE}/stage4_means.csv"
ids = list(F.load_data())
_state, _lock, _loc = {}, threading.Lock(), threading.local()


def outlines():
    out = {}
    for sr in shapefile.Reader(SHP).iterShapeRecords():
        g = f"{int(sr.record['hru_id']):08d}"
        if g in ids:
            p = shape(sr.shape.__geo_interface__)
            out[g] = p if p.is_valid else p.buffer(0)
    assert set(out) == set(ids)
    return out


def masks_from(ds):
    """cell-centre masks of the 12 basins on the grid of an open Stage IV file; also the lon/lat of the lower-left cell centre (checked against NCEP's 23.117 N, 119.023 W)."""
    h, w = ds.height, ds.width
    tf = ds.transform
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    X, Y = tf * (cols, rows)
    lon, lat = pyproj.Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True).transform(np.asarray(X), np.asarray(Y))
    ll = (float(lon[h - 1, 0]), float(lat[h - 1, 0]))
    P = outlines(); masks, counts = {}, {}
    for g in ids:
        x0, y0, x1, y1 = P[g].bounds
        win = (lon >= x0 - 0.1) & (lon <= x1 + 0.1) & (lat >= y0 - 0.1) & (lat <= y1 + 0.1)
        inside = np.zeros((h, w), bool)
        rr, cc = np.nonzero(win)
        inside[rr, cc] = shapely.contains_xy(P[g], lon[rr, cc], lat[rr, cc])
        if not inside.any():                                                                  # a basin smaller than a cell: the cell nearest to the centroid
            d2 = (lon - P[g].centroid.x) ** 2 + ((lat - P[g].centroid.y)) ** 2
            inside[np.unravel_index(np.argmin(d2), d2.shape)] = True
        masks[g], counts[g] = inside, int(inside.sum())
    return masks, counts, ll, (h, w)


def get(url, tries=6):
    last = None
    for k in range(tries):
        try:
            if getattr(_loc, "s", None) is None:
                _loc.s = requests.Session()
            r = _loc.s.get(url, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code in (403, 404) and k >= 1:
                return None
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, OSError) as e:
            last = type(e).__name__
        _loc.s = None
        time.sleep(min(30, 2 ** k))
    raise RuntimeError(f"{url}: {last}")


def one(t):
    """basin means of the file labelled t (a pandas Timestamp, UTC, hour ending); None if the archive does not have it."""
    raw = get(URL.format(y=f"{t:%Y}", m=f"{t:%m}", d=f"{t:%d}", h=f"{t:%H}"))
    if raw is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib") as tmp:
        tmp.write(raw); tmp.flush()
        with rasterio.open(tmp.name) as ds:
            a = ds.read(1).astype("float64"); tags = ds.tags(1)
            with _lock:
                if "masks" not in _state:
                    _state["masks"], _state["counts"], _state["ll"], _state["shape"] = masks_from(ds)
                    assert -121 < _state["ll"][0] < -118 and 22 < _state["ll"][1] < 24, f"the Stage IV grid is not where NCEP puts it: lower-left centre {_state['ll']}"
                    json.dump(dict(cells=_state["counts"], lower_left_centre=list(_state["ll"])), open(f"{CACHE}/stage4_cells.json", "w"))          # kept for check mode, which may find every hour in the cache and fetch nothing
            assert a.shape == _state["shape"], "another grid size"
    assert tags.get("GRIB_ELEMENT") == "APCP" and int(tags["GRIB_FORECAST_SECONDS"]) == 3600 and int(tags["GRIB_VALID_TIME"]) == int((t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)), f"{t}: tags {tags}"
    a[a >= 9998] = np.nan
    return {g: (float(np.nanmean(a[m])) if np.isfinite(a[m]).any() else np.nan) for g, m in _state["masks"].items()}


def fetch_labels(labels, threads=6, verbose=True):
    """add the hour-ending labels not yet in the cache; returns the DataFrame of all cached means (index: hour-ending UTC label)."""
    have = pd.read_csv(CSV, index_col=0, parse_dates=True) if os.path.exists(CSV) else pd.DataFrame(columns=ids)
    todo = [t for t in sorted(set(pd.DatetimeIndex(labels))) if t not in have.index]
    if verbose:
        print(f"{len(labels)} labels asked, {len(todo)} to fetch ({threads} connections)", flush=True)
    rows, t0, n_none = {}, time.time(), 0
    def safe(t):
        try:
            return one(t)
        except Exception as e:
            print("  FAILED", t, e, flush=True)
            return None
    with ThreadPoolExecutor(threads) as ex:
        for k, (t, r) in enumerate(zip(todo, ex.map(safe, todo)), 1):
            rows[t] = r if r is not None else {g: np.nan for g in ids}
            n_none += r is None
            if verbose and k % 60 == 0:
                print(f"  {k}/{len(todo)} files, {time.time() - t0:.0f} s", flush=True)
                pd.concat([have, pd.DataFrame(rows).T[ids]]).sort_index().to_csv(CSV)
    new = pd.DataFrame(rows).T[ids] if rows else pd.DataFrame(columns=ids)
    out = pd.concat([have, new]).sort_index()
    out.index.name = "hour_ending_utc"
    out.to_csv(CSV)
    if verbose:
        print(f"done: {len(todo)} files in {time.time() - t0:.0f} s, {n_none} missing in the archive; cache has {len(out)} hours", flush=True)
    return out


def labels_of_day(d):
    """hour-ending UTC labels 05:00 UTC of day d .. 13:00 UTC of day d+1 (the local day t from 00:00, summer time included, to 07:00 local of day t+1 in the Central zone: 33 files)."""
    d = pd.Timestamp(d)
    return list(pd.date_range(d + pd.Timedelta(hours=5), d + pd.Timedelta(hours=37), freq="h"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "days":
        fetch_labels([t for d in sys.argv[2:] for t in labels_of_day(d)])
    elif cmd == "after":
        fetch_labels([t for d in sys.argv[2:] for t in pd.date_range(pd.Timestamp(d) + pd.Timedelta(hours=38), pd.Timestamp(d) + pd.Timedelta(hours=42), freq="h")])
    elif cmd == "labels":
        fetch_labels([pd.Timestamp(s) for s in sys.argv[2:]])
    elif cmd == "check":
        b = pd.read_csv(f"{PROJ}/data/aorc/stage_iv_hindsight.csv", index_col=0, parse_dates=True)
        pick = [b.index[i] for i in (10, 20, 40, 60, 90, 120)]
        mine = fetch_labels(pick, verbose=False).loc[pick]
        sc = json.load(open(f"{CACHE}/stage4_cells.json")) if os.path.exists(f"{CACHE}/stage4_cells.json") else {}
        cells = _state.get("counts") or sc.get("cells"); ll = _state.get("ll") or sc.get("lower_left_centre")
        # every hour that my cache and the builder's csv have in common (my days overlap the builder's windows by a few hours)
        m_all = pd.read_csv(CSV, index_col=0, parse_dates=True); common = b.index.intersection(m_all.index)
        dcom = (m_all.loc[common, ids] - b.loc[common, ids]).abs()
        print(f"  all {len(common)} hours that my cache and the builder's csv have in common ({len(common) * len(ids)} values): max abs difference {float(np.nanmax(dcom.values)):.2e}, values differing by more than 1e-12: {int((dcom.values > 1e-12).sum())}")
        dev = (mine[ids] - b.loc[pick, ids]).abs()
        print("my Stage IV basin means against the builder's csv for", [str(p) for p in pick])
        print("  cells per basin (mine):", cells)
        print("  max abs difference (mm):", float(np.nanmax(dev.values)), "| mean abs:", float(np.nanmean(dev.values)), "| mean value (mm):", float(np.nanmean(b.loc[pick, ids].values)), "| NaN in mine/builder:", int(mine[ids].isna().sum().sum()), int(b.loc[pick, ids].isna().sum().sum()))
        jsave("a6_check.json", dict(labels=[str(p) for p in pick], max_abs_diff=float(np.nanmax(dev.values)), mean_abs_diff=float(np.nanmean(dev.values)), cells=cells, lower_left_centre=ll,
                                    common_hours=dict(n_hours=int(len(common)), n_values=int(len(common) * len(ids)), max_abs_diff=float(np.nanmax(dcom.values)), n_values_above_1e_12=int((dcom.values > 1e-12).sum()))))
