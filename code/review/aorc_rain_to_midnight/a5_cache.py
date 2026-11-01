"""AORC review, check (c): the fetcher and the cache, with the reviewer's own code and a few small requests to the AORC bucket
(https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com): (1) the stores' own metadata (units, scale, fill value, chunk shape, compressor, time axis units) read for three years and compared with what the fetcher
assumes; (2) the cache: cells per basin, hours filled from ERA5 (408), hours with only some cells missing (264); the filled hours are ERA5's; (3) ONE BASIN-MONTH recomputed from raw chunks fetched here (Hyco Creek,
2010-03: 6 time chunks of one spatial chunk, about 1 MB): the basin mask built here (cell centres inside the CAMELS outline, on the nominal 1/120 degree grid and on the stored coordinate arrays), the mean of the valid
cells of every hour, against the cache; (4) one raw chunk of a filled run (1990-07-19..23, three basins) to see that the source really holds the fill value in every cell of the basin. About one minute."""
import sys; sys.dont_write_bytecode = True
import subprocess, math, time
from a_common import *
import requests, shapefile, shapely
from shapely.geometry import shape
import a_clock as C

BASE = "https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com"
SHP = f"{PROJ}/data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp"
R = jload("results/aorc_rain_to_midnight.json"); A = jload("results/aorc_data.json")
out = {}
sess = requests.Session()


def get(url, allow_absent=False):
    for k in range(5):
        r = sess.get(url, timeout=120)
        if r.status_code == 200:
            return r.content
        if allow_absent and r.status_code in (403, 404):
            return None
        time.sleep(2 ** k)
    raise RuntimeError(f"{url}: HTTP {r.status_code}")


def zstd(raw):
    p = subprocess.run(["zstd", "-d", "-c", "-q"], input=raw, capture_output=True)
    assert p.returncode == 0, p.stderr[:200]
    return p.stdout


# ---- 1. the stores' metadata, read here ----------------------------------------------------------------------------------------------------------
meta = {}
for y in (1990, 2005, 2012):
    m = json.loads(get(f"{BASE}/{y}.zarr/.zmetadata"))["metadata"]
    za, at, tz, ta = m["APCP_surface/.zarray"], m["APCP_surface/.zattrs"], m["time/.zarray"], m["time/.zattrs"]
    meta[y] = dict(shape=za["shape"], chunks=za["chunks"], dtype=za["dtype"], compressor=za["compressor"], fill_value_zarray=za["fill_value"], fill_attr=at.get("_FillValue"), missing_value=at.get("missing_value"), scale_factor=at["scale_factor"], units=at["units"],
                   long_name=at["long_name"], dims=at["_ARRAY_DIMENSIONS"], time_units=ta["units"], time_long_name=ta["long_name"], time_calendar=ta.get("calendar"), reference_date=ta.get("reference_date"), time_dtype=tz["dtype"],
                   lat_chunks=m["latitude/.zarray"]["chunks"], lon_chunks=m["longitude/.zarray"]["chunks"])
    print(f"{y}: APCP_surface {meta[y]['dtype']} shape {meta[y]['shape']} chunks {meta[y]['chunks']} {meta[y]['compressor']['id']} level {meta[y]['compressor'].get('level')}; scale {meta[y]['scale_factor']:.10f}, units {meta[y]['units']}, fill {meta[y]['fill_value_zarray']}/{meta[y]['fill_attr']}; time: {meta[y]['time_units']}, '{meta[y]['time_long_name']}'")
assumed = dict(dtype="<i2", chunks=[144, 128, 256], units="kg/m^2", fill=-32767, scale=0.1, time_units="seconds since 1970-01-01", compressor="zstd")
ok1 = all(v["dtype"] == assumed["dtype"] and v["chunks"] == assumed["chunks"] and v["units"] == assumed["units"] and (v["fill_value_zarray"] == assumed["fill"] or v["fill_attr"] == assumed["fill"]) and abs(v["scale_factor"] - assumed["scale"]) < 1e-8
          and v["time_units"] == assumed["time_units"] and v["compressor"]["id"] == "zstd" and v["dims"] == ["time", "latitude", "longitude"] for v in meta.values())
print("the three stores are laid out as the fetcher assumes (int16, scale 0.1 mm per unit, fill -32767, chunks 144 x 128 x 256, zstd, seconds since 1970, kg/m^2 = mm):", ok1)
out["metadata"] = dict(years=meta, as_assumed=ok1)

# ---- the time axis of one store: the first and last stamps, the step ------------------------------------------------------------------------------------
tz12 = json.loads(get(f"{BASE}/2012.zarr/time/.zarray")) if False else None
t_chunks = [np.frombuffer(zstd(get(f"{BASE}/2012.zarr/time/{c}")), "<f8") for c in range(math.ceil(8784 / 144))]
t12 = np.concatenate(t_chunks)[:8784]
stamp12 = pd.to_datetime(t12, unit="s")
print(f"2012 time axis: {len(t12)} values from {stamp12[0]} to {stamp12[-1]}, step(s) {sorted(set(np.diff(t12)))}")
out["time_axis_2012"] = dict(n=len(t12), first=str(stamp12[0]), last=str(stamp12[-1]), steps=sorted(set(float(x) for x in np.diff(t12))))

# ---- 2. the cache ---------------------------------------------------------------------------------------------------------------------------------------------
ids = list(F_ids := [g for g in A["per_basin"]])
n_cells, n_filled, n_part, era_equal = {}, 0, 0, True
era = {}
for g in ids:
    t, x, fl, nv, nc = C.read_aorc(g)
    n_cells[g] = nc
    n_filled += int(fl.sum()); n_part += int(((nv < nc) & ~fl).sum())
    if fl.any():
        j = json.load(open(f"{PROJ}/data/hourly/era5_{g}.json"))
        e = np.array(j["hourly"]["precipitation"], float)
        te = np.array(j["hourly"]["time"], dtype="datetime64[m]")
        assert np.array_equal(te, t)
        era_equal &= bool(np.array_equal(x[fl], e[fl]))
        era[g] = dict(hours_filled=int(fl.sum()), first=str(t[fl][0]), last=str(t[fl][-1]), equal_to_era5=bool(np.array_equal(x[fl], e[fl])), n_valid_in_filled_hours=int(nv[fl].max()))
print(f"cache: cells per basin {min(n_cells.values())} to {max(n_cells.values())} ({n_cells}); hours filled from ERA5: {n_filled} (JSON {A['gap_fill']['total_hours_filled']}); hours with only some cells missing: {n_part}; the filled hours equal the cached ERA5 hours: {era_equal}")
out["cache"] = dict(n_cells=n_cells, hours_filled=n_filled, partly_missing_hours=n_part, filled_runs=era, filled_equal_era5=era_equal, cells_equal_json=bool(n_cells == R["daily_rain"]["mask"]["n_cells_by_basin"]))

# ---- 2b. the masks of all 12 basins recounted with my own method (cell centres inside the outline on the nominal 1/120 degree grid, shapely.contains_xy) ----------------------------------
POLY0 = {}
for sr in shapefile.Reader(SHP).iterShapeRecords():
    g_ = f"{int(sr.record['hru_id']):08d}"
    if g_ in ids:
        p_ = shape(sr.shape.__geo_interface__); POLY0[g_] = p_ if p_.is_valid else p_.buffer(0)
mine_cells, plan_ranges = {}, A["plan"]["per_basin"] if "per_basin" in A["plan"] else A["plan"]
ok_ranges = True
for g_ in ids:
    bx0, by0, bx1, by1 = POLY0[g_].bounds
    il_ = np.arange(int(math.floor((by0 - 20.0) * 120)) - 3, int(math.ceil((by1 - 20.0) * 120)) + 4); jl_ = np.arange(int(math.floor((bx0 + 130.0) * 120)) - 3, int(math.ceil((bx1 + 130.0) * 120)) + 4)
    LON_, LAT_ = np.meshgrid(-130.0 + jl_ / 120, 20.0 + il_ / 120)
    ii_, jj_ = np.nonzero(shapely.contains_xy(POLY0[g_], LON_, LAT_))
    mine_cells[g_] = len(ii_)
    ok_ranges &= (int(il_[ii_].min()), int(il_[ii_].max())) == tuple(plan_ranges[g_]["lat_index_range"]) and (int(jl_[jj_].min()), int(jl_[jj_].max())) == tuple(plan_ranges[g_]["lon_index_range"])
print(f"\nmasks of all 12 basins recounted with my own method: cells {min(mine_cells.values())} to {max(mine_cells.values())}; equal to the cache's n_cells in every basin: {all(mine_cells[g_] == n_cells[g_] for g_ in ids)}; "
      f"first and last cell row and column equal the builder's record (results/aorc_data.json, plan): {ok_ranges}")
out["masks_recounted"] = dict(cells_mine=mine_cells, equal_cache=bool(all(mine_cells[g_] == n_cells[g_] for g_ in ids)), index_ranges_equal_builder_plan=bool(ok_ranges))

# ---- 3. one basin-month from raw chunks -----------------------------------------------------------------------------------------------------------------------
POLY = {}
for sr in shapefile.Reader(SHP).iterShapeRecords():
    g = f"{int(sr.record['hru_id']):08d}"
    if g in ids:
        p = shape(sr.shape.__geo_interface__); POLY[g] = p if p.is_valid else p.buffer(0)
G, YEAR, MONTH = "02077200", 2010, "2010-03"
x0, y0, x1, y1 = POLY[G].bounds
LAT0, LON0, STEP = 20.0, -130.0, 1 / 120
def mask_on(lat_of, lon_of):
    il = np.arange(int(math.floor((y0 - LAT0) / STEP)) - 3, int(math.ceil((y1 - LAT0) / STEP)) + 4); jl = np.arange(int(math.floor((x0 - LON0) / STEP)) - 3, int(math.ceil((x1 - LON0) / STEP)) + 4)
    LON, LAT = np.meshgrid(lon_of(jl), lat_of(il))
    ii, jj = np.nonzero(shapely.contains_xy(POLY[G], LON, LAT))
    return il[ii], jl[jj]
la_n, lo_n = mask_on(lambda i: LAT0 + i * STEP, lambda j: LON0 + j * STEP)                  # nominal grid
# stored coordinate arrays of the 2010 store, the chunks that cover the basin
mz = json.loads(get(f"{BASE}/{YEAR}.zarr/.zmetadata"))["metadata"]
chl, cho = mz["latitude/.zarray"]["chunks"][0], mz["longitude/.zarray"]["chunks"][0]
lat_s = {}; lon_s = {}
for c in range(int(la_n.min() - 3) // chl, int(la_n.max() + 3) // chl + 1):
    for k, v in enumerate(np.frombuffer(zstd(get(f"{BASE}/{YEAR}.zarr/latitude/{c}")), "<f8")): lat_s[c * chl + k] = v
for c in range(int(lo_n.min() - 3) // cho, int(lo_n.max() + 3) // cho + 1):
    for k, v in enumerate(np.frombuffer(zstd(get(f"{BASE}/{YEAR}.zarr/longitude/{c}")), "<f8")): lon_s[c * cho + k] = v
la_s, lo_s = mask_on(lambda i: np.array([lat_s[int(k)] for k in i]), lambda j: np.array([lon_s[int(k)] for k in j]))        # stored coordinates
chunks_needed = sorted({(int(a // 128), int(b // 256)) for a, b in zip(la_n, lo_n)} | {(int(a // 128), int(b // 256)) for a, b in zip(la_s, lo_s)})
print(f"\n{G} (Hyco Creek), {MONTH}: my mask holds {len(la_n)} cells on the nominal grid (cache: {n_cells[G]}), {len(la_s)} on the stored coordinates; spatial chunks {chunks_needed}")
h_first = int(((np.datetime64(f"{MONTH}-01T00:00") - np.datetime64(f"{YEAR}-01-01T00:00")) / np.timedelta64(1, "h")))
nh = 31 * 24
t_chunks = list(range(h_first // 144, (h_first + nh - 1) // 144 + 1))
raw_bytes = 0
blocks = {}
for sl, so in chunks_needed:
    arr = np.full((len(t_chunks) * 144, 128, 256), -32767, "<i2")
    for k, c in enumerate(t_chunks):
        raw = get(f"{BASE}/{YEAR}.zarr/APCP_surface/{c}.{sl}.{so}", allow_absent=True)
        if raw is not None:
            raw_bytes += len(raw); b = zstd(raw); assert len(b) == 144 * 128 * 256 * 2
            arr[k * 144:(k + 1) * 144] = np.frombuffer(b, "<i2").reshape(144, 128, 256)
    blocks[(sl, so)] = arr
print(f"fetched {len(t_chunks) * len(chunks_needed)} chunk files, {raw_bytes / 1e6:.2f} MB compressed")
def month_series(la, lo):
    tot = np.zeros(len(t_chunks) * 144); cnt = np.zeros(len(t_chunks) * 144, int)
    for (sl, so), arr in blocks.items():
        sel = (la // 128 == sl) & (lo // 256 == so)
        v = arr[:, la[sel] - sl * 128, lo[sel] - so * 256].astype(np.int64)
        ok = v != -32767
        tot += np.where(ok, v, 0).sum(1); cnt += ok.sum(1)
    s = np.where(cnt > 0, tot * 0.1 / np.maximum(cnt, 1), np.nan)
    o = h_first - t_chunks[0] * 144
    return s[o:o + nh], cnt[o:o + nh]
mine_n, cnt_n = month_series(la_n, lo_n); mine_s, cnt_s = month_series(la_s, lo_s)
t, x, fl, nv, nc = C.read_aorc(G)
sel = (t >= np.datetime64(f"{MONTH}-01T00:00")) & (t <= np.datetime64(f"{MONTH}-31T23:00"))
ref = x[sel]
d_n = float(np.nanmax(np.abs(mine_n - ref))); d_s = float(np.nanmax(np.abs(mine_s - ref)))
print(f"  my basin mean (nominal-grid mask) against the cache over the {nh} hours of {MONTH}: max |difference| {d_n:.2e} mm (month total {np.nansum(mine_n):.2f} mm, cache {ref.sum():.2f}); valid-cell counts equal the cache's n_valid: {bool(np.array_equal(cnt_n, nv[sel]))}")
print(f"  with the stored coordinate arrays ({len(la_s)} cells, {len(set(zip(la_s.tolist(), lo_s.tolist())) ^ set(zip(la_n.tolist(), lo_n.tolist())))} cells differ): max |difference| to the cache {d_s:.2e} mm, month total {np.nansum(mine_s):.2f} mm, correlation of the hours {np.corrcoef(mine_s, ref)[0, 1]:.5f}")
out["basin_month"] = dict(gauge=G, month=MONTH, cells_nominal=int(len(la_n)), cells_cache=n_cells[G], cells_stored_grid=int(len(la_s)), cells_differing_between_grids=int(len(set(zip(la_n.tolist(), lo_n.tolist())) ^ set(zip(la_s.tolist(), lo_s.tolist())))),
                          chunk_files=int(len(t_chunks) * len(chunks_needed)), mb_compressed=raw_bytes / 1e6, max_abs_diff_nominal_mm=d_n, max_abs_diff_stored_grid_mm=d_s, month_total_mine_mm=float(np.nansum(mine_n)), month_total_cache_mm=float(ref.sum()),
                          valid_counts_equal=bool(np.array_equal(cnt_n, nv[sel])), cell_index_sets_equal_to_a_recount_of_the_cache=None)

# ---- 4. a raw chunk of a filled run ----------------------------------------------------------------------------------------------------------------------------
H0 = int(((np.datetime64("1990-07-19T00:00") - np.datetime64("1990-01-01T00:00")) / np.timedelta64(1, "h")))
plan = A["plan"]["per_basin"] if "per_basin" in A["plan"] else A["plan"]
fills = {}
for g in ("02137727", "03439000", "03463300"):
    chs = plan[g]["chunks"]
    la_g, lo_g = [], []
    x0g, y0g, x1g, y1g = POLY[g].bounds
    il = np.arange(int(math.floor((y0g - LAT0) / STEP)) - 2, int(math.ceil((y1g - LAT0) / STEP)) + 3); jl = np.arange(int(math.floor((x0g - LON0) / STEP)) - 2, int(math.ceil((x1g - LON0) / STEP)) + 3)
    LON, LAT = np.meshgrid(LON0 + jl * STEP, LAT0 + il * STEP); ii, jj = np.nonzero(shapely.contains_xy(POLY[g], LON, LAT)); la_g, lo_g = il[ii], jl[jj]
    status = []
    for sl, so in sorted({(int(a // 128), int(b // 256)) for a, b in zip(la_g, lo_g)}):
        raw = get(f"{BASE}/1990.zarr/APCP_surface/{H0 // 144}.{sl}.{so}", allow_absent=True)
        if raw is None: status.append(dict(chunk=[H0 // 144, sl, so], present=False)); continue
        arr = np.frombuffer(zstd(raw), "<i2").reshape(144, 128, 256)
        sel = (la_g // 128 == sl) & (lo_g // 256 == so)
        v = arr[H0 % 144:H0 % 144 + 120][:, la_g[sel] - sl * 128, lo_g[sel] - so * 256]
        status.append(dict(chunk=[H0 // 144, sl, so], present=True, cells_in_this_chunk=int(sel.sum()), share_of_values_equal_to_fill_in_the_120_hours=float((v == -32767).mean()), max_raw_value=int(v.max())))
    fills[g] = dict(my_cells=int(len(la_g)), cache_cells=n_cells[g], chunks=status)
    print(f"  {g}: 1990-07-19 00:00 .. 07-23 23:00 (120 hours): my mask {len(la_g)} cells (cache {n_cells[g]}); raw chunks: {status}")
out["filled_run_raw_chunks"] = fills

# ---- 5. the other runs of missing data in the cache, from raw chunks: the 48 filled hours of 2006-08 in the TEST years (02221525) and the three runs in which only some cells are missing (1986 and 1989, training years) -------------------------
def cells_of(g):
    bx0, by0, bx1, by1 = POLY[g].bounds
    il_ = np.arange(int(math.floor((by0 - LAT0) / STEP)) - 3, int(math.ceil((by1 - LAT0) / STEP)) + 4); jl_ = np.arange(int(math.floor((bx0 - LON0) / STEP)) - 3, int(math.ceil((bx1 - LON0) / STEP)) + 4)
    LON_, LAT_ = np.meshgrid(LON0 + jl_ * STEP, LAT0 + il_ * STEP); ii_, jj_ = np.nonzero(shapely.contains_xy(POLY[g], LON_, LAT_))
    return il_[ii_], jl_[jj_]
runs5 = {}
for g, start, nh_, what in (("02221525", "2006-08-11T00:00", 48, "filled from ERA5 in the cache (test years)"), ("02472000", "1986-03-08T00:00", 120, "some cells missing (training years)"),
                            ("02296500", "1989-10-16T00:00", 72, "some cells missing (training years)"), ("02298608", "1989-10-16T00:00", 72, "some cells missing (training years)")):
    la_g, lo_g = cells_of(g); y_ = int(start[:4]); h0_ = int(((np.datetime64(start) - np.datetime64(f"{y_}-01-01T00:00")) / np.timedelta64(1, "h")))
    tch = list(range(h0_ // 144, (h0_ + nh_ - 1) // 144 + 1)); nbytes = 0
    tot_ = np.zeros(nh_); cnt_ = np.zeros(nh_, int)
    for sl, so in sorted({(int(a // 128), int(b // 256)) for a, b in zip(la_g, lo_g)}):
        arr = np.full((len(tch) * 144, 128, 256), -32767, "<i2")
        for k, c in enumerate(tch):
            raw = get(f"{BASE}/{y_}.zarr/APCP_surface/{c}.{sl}.{so}", allow_absent=True)
            if raw is not None:
                nbytes += len(raw); arr[k * 144:(k + 1) * 144] = np.frombuffer(zstd(raw), "<i2").reshape(144, 128, 256)
        sel = (la_g // 128 == sl) & (lo_g // 256 == so)
        v = arr[h0_ - tch[0] * 144:h0_ - tch[0] * 144 + nh_][:, la_g[sel] - sl * 128, lo_g[sel] - so * 256].astype(np.int64)
        ok_ = v != -32767; tot_ += np.where(ok_, v, 0).sum(1); cnt_ += ok_.sum(1)
    t_, x_, fl_, nv_, nc_ = C.read_aorc(g)
    sel_t = (t_ >= np.datetime64(start)) & (t_ < np.datetime64(start) + np.timedelta64(nh_, "h")); assert sel_t.sum() == nh_
    mean_ = np.where(cnt_ > 0, tot_ * 0.1 / np.maximum(cnt_, 1), np.nan)
    equal_counts = bool(np.array_equal(cnt_, nv_[sel_t])); filled_ = bool(fl_[sel_t].all()); mx = float(np.nanmax(np.abs(mean_ - x_[sel_t]))) if (cnt_ > 0).any() else None
    runs5[g + " " + start[:10]] = dict(what=what, my_cells=int(len(la_g)), cache_cells=n_cells[g], hours=nh_, mb=nbytes / 1e6, valid_cells_min=int(cnt_.min()), valid_cells_max=int(cnt_.max()), valid_counts_equal_cache=equal_counts,
                                       all_hours_filled_in_cache=filled_, max_abs_diff_of_the_hourly_mean_mm=mx)
    print(f"  {g} {start[:10]} ({nh_} hours, {what}): my mask {len(la_g)} cells (cache {n_cells[g]}); valid cells per hour {int(cnt_.min())} to {int(cnt_.max())}; counts equal the cache's n_valid: {equal_counts}; "
          + ("every hour filled in the cache: %s (the raw chunk holds no value at all: %s)" % (filled_, bool((cnt_ == 0).all())) if (cnt_ == 0).all() else f"hourly mean against the cache: max |difference| {mx:.2e} mm") + f"; {nbytes / 1e6:.2f} MB")
out["other_missing_runs_raw_chunks"] = runs5
jsave("a5_cache.json", out)
