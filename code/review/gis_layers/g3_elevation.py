"""Check 3: the 12 elevation GeoTIFFs. For each: CRS, dtype, nodata, resolution (about 1 arc-second), alignment, bounds against the outline, compression/tiling; then the mean
elevation of the cells whose centre lies inside the outline, recomputed here with shapely.contains_xy (a different algorithm from the rasteriser the export used), with and
without the cos(latitude) weight, compared with the outline attribute elev_mean_m, with results/basin_gis.json and with CAMELS elev_mean; the cell counts, voids, min/max; the
buffer ring; the cell values against the source SRTM-format tiles in data/dem; the elevation at the gauge for the exported point and for the 5-decimal CAMELS point."""
import time
from g_common import *
import geopandas as gpd
import pyproj
import rasterio
import shapely
from shapely.geometry import Point

ids, audit = basins()
t, v = camels()
bg = jload("results/basin_gis.json")["basins"]
rec = jload(RECORD)
gp = scratch_copy(GIS_DIR / "basins_12.gpkg", "basins_12.gpkg")
out = gpd.read_file(gp, layer="outlines").set_index("gauge_id", drop=False)
gau = gpd.read_file(gp, layer="gauges").set_index("gauge_id", drop=False)
geod = pyproj.Geod(ellps="GRS80")
res = {"per_basin": {}}


def window_for(bounds, tf, width, height, pad=0):
    minx, miny, maxx, maxy = bounds
    c0 = max(0, int(math.floor((minx - tf.c) / tf.a)) - pad)
    c1 = min(width, int(math.ceil((maxx - tf.c) / tf.a)) + pad)
    r0 = max(0, int(math.floor((maxy - tf.f) / tf.e)) - pad)          # tf.e < 0: north-up
    r1 = min(height, int(math.ceil((miny - tf.f) / tf.e)) + pad)
    return r0, r1, c0, c1


def centres(tf, r0, r1, c0, c1):
    xs = tf.c + (np.arange(c0, c1) + 0.5) * tf.a
    ys = tf.f + (np.arange(r0, r1) + 0.5) * tf.e
    return np.meshgrid(xs, ys)


t0 = time.time()
for i in ids:
    path = GIS_DIR / f"elevation_{i}.tif"
    geom = out.loc[i, "geometry"]
    with rasterio.open(path) as src:
        arr = src.read(1)
        tf = src.transform
        prof = {
            "crs_epsg": src.crs.to_epsg() if src.crs else None, "crs_wkt_name": src.crs.to_string() if src.crs else None,
            "dtype": src.dtypes[0], "nodata": src.nodata, "count": src.count, "width": src.width, "height": src.height,
            "res": [src.res[0], src.res[1]], "bounds": list(src.bounds), "compression": str(src.compression.value) if src.compression else None,
            "tiled": bool(src.profile.get("tiled")), "block": [src.profile.get("blockxsize"), src.profile.get("blockysize")],
            "image_structure": src.tags(ns="IMAGE_STRUCTURE"), "tags_default": src.tags(), "descriptions": list(src.descriptions), "units": list(src.units),
            "scales": list(src.scales), "offsets": list(src.offsets), "overviews": src.overviews(1), "colormap_present": bool(src.colorinterp[0].name == "palette"),
            "transform": [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f],
        }
        tags_all = {ns: src.tags(ns=ns) for ns in src.tag_namespaces()}
    H, W = arr.shape
    left, bottom, right, top = prof["bounds"]
    minx, miny, maxx, maxy = geom.bounds
    # --- grid facts
    dx, dy = tf.a, -tf.e
    grid = {
        "dx_minus_1arcsec": dx - ARCSEC, "dy_minus_1arcsec": dy - ARCSEC,
        "left_offset_in_cells": (left / ARCSEC) % 1.0,           # 0.5 = cell centres on whole arc-seconds
        "top_offset_in_cells": (top / ARCSEC) % 1.0,
        "margin_arcsec": {"west": (minx - left) / ARCSEC, "east": (right - maxx) / ARCSEC, "south": (miny - bottom) / ARCSEC, "north": (top - maxy) / ARCSEC},
        "bounds_contain_outline": bool(left <= minx and right >= maxx and bottom <= miny and top >= maxy),
    }
    latc = (top + bottom) / 2.0
    _, _, ew = geod.inv(0.0, latc, ARCSEC, latc)
    _, _, ns = geod.inv(0.0, latc - ARCSEC / 2, 0.0, latc + ARCSEC / 2)
    grid["cell_m_ew_at_raster_centre"], grid["cell_m_ns_at_raster_centre"] = ew, ns
    # --- cells whose centre lies inside the outline (shapely.contains_xy, chunked by rows)
    r0, r1, c0, c1 = window_for(geom.bounds, tf, W, H, pad=1)
    inside = np.zeros((H, W), dtype=bool)
    step = 300
    for a in range(r0, r1, step):
        b = min(r1, a + step)
        X, Y = centres(tf, a, b, c0, c1)
        inside[a:b, c0:c1] = shapely.contains_xy(geom, X, Y)
    valid = arr != DEM_NODATA
    n_in = int(inside.sum())
    z = arr[inside].astype(np.float64)
    n_void = int((~valid[inside]).sum())
    zv = z[z != DEM_NODATA]
    rows_in = np.where(inside)[0]
    lat_in = top + (rows_in + 0.5) * tf.e
    w = np.cos(np.radians(lat_in))
    zz = arr[inside].astype(np.float64)
    ok = zz != DEM_NODATA
    mean_w = float(np.sum(zz[ok] * w[ok]) / np.sum(w[ok]))
    mean_u = float(zv.mean())
    # --- the ring: valid cells vs my own 15-arc-second buffer of the outline
    buf = geom.buffer(BUFFER_ARCSEC * ARCSEC, quad_segs=64)
    in_buf = np.zeros((H, W), dtype=bool)
    rb0, rb1, cb0, cb1 = window_for(buf.bounds, tf, W, H, pad=1)
    for a in range(rb0, rb1, step):
        b = min(rb1, a + step)
        X, Y = centres(tf, a, b, cb0, cb1)
        in_buf[a:b, cb0:cb1] = shapely.contains_xy(buf, X, Y)
    ring = {
        "n_valid_cells": int(valid.sum()), "n_cells_in_my_buffer": int(in_buf.sum()),
        "valid_outside_my_buffer": int((valid & ~in_buf).sum()), "nodata_inside_my_buffer": int((~valid & in_buf).sum()),
        "nodata_inside_outline": n_void, "all_inside_valid": bool(valid[inside].all()),
        "valid_outside_outline_cells": int((valid & ~inside).sum()),
    }
    # --- cell values against the source tiles (every valid cell must equal the source sample)
    rr, cc = np.where(valid)
    lon_v = tf.c + (cc + 0.5) * tf.a
    lat_v = tf.f + (rr + 0.5) * tf.e
    vals = arr[rr, cc].astype(np.int32)
    compared = np.zeros(len(vals), dtype=bool)
    n_bad = 0
    tiles_used = []
    for lat0 in range(int(math.floor(lat_v.min())), int(math.floor(lat_v.max())) + 1):
        for lon0 in range(int(math.floor(lon_v.min())), int(math.floor(lon_v.max())) + 1):
            sel = (lat_v >= lat0 - 1e-9) & (lat_v <= lat0 + 1 + 1e-9) & (lon_v >= lon0 - 1e-9) & (lon_v <= lon0 + 1 + 1e-9)
            if not sel.any():
                continue
            tile = srtm_tile(lat0, lon0)
            if tile is None:
                continue
            ii_f = (lat0 + 1 - lat_v[sel]) * 3600.0
            jj_f = (lon_v[sel] - lon0) * 3600.0
            ii, jj = np.rint(ii_f).astype(int), np.rint(jj_f).astype(int)
            assert np.abs(ii_f - ii).max() < 1e-4 and np.abs(jj_f - jj).max() < 1e-4, "cell centres are not on whole arc-seconds"
            src_vals = tile[ii, jj]
            n_bad += int((src_vals != vals[sel]).sum())
            compared[np.where(sel)[0]] = True
            tiles_used.append(tile_name(lat0, lon0))
    src_check = {"tiles_compared": tiles_used, "tiles_equal_record": sorted(set(tiles_used)) == sorted(rec["basins"][i]["elevation_tiles"]), "n_valid_cells": int(len(vals)), "n_compared": int(compared.sum()), "n_not_found_in_any_cached_tile": int((~compared).sum()),
                 "n_value_mismatches": n_bad}
    # --- gauge cell elevation: exported point vs CAMELS 5-decimal point
    def cell_at(lon, lat):
        c = int(math.floor((lon - tf.c) / tf.a)); r = int(math.floor((lat - tf.f) / tf.e))
        return int(arr[r, c]) if (0 <= r < H and 0 <= c < W) else None
    gfile = gau.loc[i, "geometry"]
    gau_elev = {
        "elev_at_exported_point": cell_at(gfile.x, gfile.y), "attr": int(gau.loc[i, "elev_at_gauge_m"]),
        "elev_at_camels_5dec_point": cell_at(float(t.loc[i, "gauge_lon"]), float(t.loc[i, "gauge_lat"])), "basin_gis": bg[i]["gauge"]["elev_m"],
    }
    # --- compare with the attributes and sources
    b = bg[i]
    pb = {
        "short": out.loc[i, "short"], "raster": prof, "grid": grid, "ring": ring, "source_tiles": src_check, "gauge_cell": gau_elev, "tags_all": tags_all,
        "n_cells_inside_mine": n_in, "n_cells_inside_attr": int(out.loc[i, "n_elevation_cells"]), "n_cells_inside_basin_gis": int(b["n_elevation_cells"]),
        "n_voids_inside_mine": n_void, "n_voids_basin_gis": int(b["n_elevation_voids"]),
        "elev_mean_weighted_mine": mean_w, "elev_mean_unweighted_mine": mean_u,
        "elev_mean_attr": float(out.loc[i, "elev_mean_m"]), "elev_mean_basin_gis": float(b["elev_mean_m"]),
        "elev_min_mine": float(zv.min()), "elev_max_mine": float(zv.max()), "elev_min_attr": float(out.loc[i, "elev_min_m"]), "elev_max_attr": float(out.loc[i, "elev_max_m"]),
        "elev_min_basin_gis": float(b["elev_min_m"]), "elev_max_basin_gis": float(b["elev_max_m"]),
        "elev_mean_camels": float(t.loc[i, "elev_mean"]), "elev_mean_camels_attr": float(out.loc[i, "camels_elev_mean_m"]),
        "elev_mean_outline_file": float(out.loc[i, "outline_file_elev_mean_m"]),
        "elev_diff_camels_mine_weighted": mean_w - float(t.loc[i, "elev_mean"]), "elev_diff_camels_attr": float(out.loc[i, "elev_diff_camels_m"]),
        "elev_diff_camels_unweighted_mine": mean_u - float(t.loc[i, "elev_mean"]),
        "n_written_valid_cells_attr_json": int(rec["basins"][i]["n_elevation_cells_written"]),
        "elevation_file_attr": str(out.loc[i, "elevation_file"]),
        "README_cell_m_ew": float(re.search(r"\(([\d.]+) m east-west", [f for f in rec["files"] if f["file"].endswith(f"elevation_{i}.tif")][0]["resolution"]).group(1)),
        "README_cell_m_ns": float(re.search(r"x ([\d.]+) m north-south", [f for f in rec["files"] if f["file"].endswith(f"elevation_{i}.tif")][0]["resolution"]).group(1)),
    }
    res["per_basin"][i] = pb
    print(f"{i} {pb['short']:18s} {time.time() - t0:6.1f}s  cells {n_in:8d}  mean(w) {mean_w:9.4f}  mean(u) {mean_u:9.4f}  attr {pb['elev_mean_attr']:9.4f}  bgis {pb['elev_mean_basin_gis']:9.4f}  "
          f"CAMELS {pb['elev_mean_camels']:8.2f}  diff {pb['elev_diff_camels_mine_weighted']:+8.2f}  src mismatches {n_bad}/{len(vals)}", flush=True)

res["totals"] = {
    "cells_inside_mine": sum(p["n_cells_inside_mine"] for p in res["per_basin"].values()),
    "cells_inside_basin_gis": sum(p["n_cells_inside_basin_gis"] for p in res["per_basin"].values()),
    "cells_written_valid": sum(p["ring"]["n_valid_cells"] for p in res["per_basin"].values()),
    "voids_inside": sum(p["n_voids_inside_mine"] for p in res["per_basin"].values()),
}
jsave("g3_elevation.json", res)

print()
print("totals", res["totals"])
print(f"{'id':9s} {'crs':>5s} {'dtype':>6s} {'nodata':>7s} {'res(\")':>9s} {'ew m':>6s} {'ns m':>6s} {'README ew/ns':>13s} {'ctr off':>8s} {'margins W/E/S/N (arcsec)':>30s} {'tiled':>6s} {'comp':>5s} {'pred':>5s}")
for i in ids:
    p = res["per_basin"][i]; r = p["raster"]; g = p["grid"]
    m = g["margin_arcsec"]
    print(f"{i:9s} {r['crs_epsg']!s:>5s} {r['dtype']:>6s} {r['nodata']!s:>7s} {r['res'][0] * 3600:9.6f} {g['cell_m_ew_at_raster_centre']:6.2f} {g['cell_m_ns_at_raster_centre']:6.2f} "
          f"{p['README_cell_m_ew']:6.2f}/{p['README_cell_m_ns']:5.2f} {g['left_offset_in_cells']:8.4f} {m['west']:7.2f}/{m['east']:6.2f}/{m['south']:6.2f}/{m['north']:6.2f}   "
          f"{str(r['tiled']):>6s} {str(r['compression']):>5s} {str(r['image_structure'].get('PREDICTOR')):>5s}  contains={g['bounds_contain_outline']}")
print()
print(f"{'id':9s} {'mean w':>10s} {'mean u':>10s} {'attr':>10s} {'basin_gis':>10s} {'CAMELS':>8s} {'diff':>8s} {'outl.file':>9s}  cells(mine/attr/bgis)           voids  min/max(mine) attr gauge(file/attr, CAMELS/bgis)")
for i in ids:
    p = res["per_basin"][i]; gc = p["gauge_cell"]
    print(f"{i:9s} {p['elev_mean_weighted_mine']:10.4f} {p['elev_mean_unweighted_mine']:10.4f} {p['elev_mean_attr']:10.4f} {p['elev_mean_basin_gis']:10.4f} {p['elev_mean_camels']:8.2f} "
          f"{p['elev_diff_camels_mine_weighted']:+8.2f} {p['elev_mean_outline_file']:9.2f}  {p['n_cells_inside_mine']}/{p['n_cells_inside_attr']}/{p['n_cells_inside_basin_gis']}  "
          f"{p['n_voids_inside_mine']:3d}  {p['elev_min_mine']:.0f}/{p['elev_max_mine']:.0f} vs {p['elev_min_attr']:.0f}/{p['elev_max_attr']:.0f}  "
          f"{gc['elev_at_exported_point']}/{gc['attr']}, {gc['elev_at_camels_5dec_point']}/{gc['basin_gis']}")
print()
print("ring:", {i: {k: res['per_basin'][i]['ring'][k] for k in ('n_valid_cells', 'valid_outside_my_buffer', 'nodata_inside_my_buffer', 'all_inside_valid')} for i in ids})
print("source tiles:", {i: (res['per_basin'][i]['source_tiles']['n_value_mismatches'], res['per_basin'][i]['source_tiles']['n_not_found_in_any_cached_tile']) for i in ids})
