"""Check 4: the 12 land-cover GeoTIFFs. For each: CRS EPSG:5070, 30 m, nodata 250, dtype, colour table against landcover_classes.csv, bounds against the outline (projected
here with pyproj); then the cells whose centre lies inside the outline (shapely.contains_xy on the projected outline), the NLCD class shares and the forest share
(classes 41, 42, 43) recomputed and compared with the outline attribute forest_share, with results/basin_gis.json and with CAMELS frac_forest; the dominant class and group;
cells outside the outline are nodata and none inside is; the cell values against the source cache data/landcover/annual_nlcd_2011_<gauge>.tif; the legend CSV."""
import time
from g_common import *
import geopandas as gpd
import pyproj
import rasterio
import shapely
from shapely.ops import transform

ids, audit = basins()
t, v = camels()
bgfull = jload("results/basin_gis.json")
bg = bgfull["basins"]
classes_ref = {int(k): n for k, n in bgfull["sources"]["land_cover"]["classes"].items()}
groups_ref = {g: list(c) for g, c in bgfull["sources"]["land_cover"]["groups"].items()}
rec = jload(RECORD)
gp = scratch_copy(GIS_DIR / "basins_12.gpkg", "basins_12.gpkg")
out = gpd.read_file(gp, layer="outlines").set_index("gauge_id", drop=False)
tr = pyproj.Transformer.from_crs(4269, 5070, always_xy=True)
res = {"per_basin": {}}

# ---- the legend CSV ---------------------------------------------------------------------------------------------------------------------------------------------------
leg = pd.read_csv(GIS_DIR / "landcover_classes.csv")
leg_codes = [int(c) for c in leg["code"]]
legend = {
    "n_rows": len(leg), "codes": leg_codes,
    "classes_equal_basin_gis": {int(r.code): (r["class"] == classes_ref[int(r.code)]) for _, r in leg.iterrows() if int(r.code) in classes_ref},
    "groups_equal_basin_gis": {int(r.code): (r.group == next((g for g, cs in groups_ref.items() if int(r.code) in cs), None)) for _, r in leg.iterrows() if int(r.code) in classes_ref},
    "rgb": {int(r.code): [int(r.red), int(r.green), int(r.blue)] for _, r in leg.iterrows()},
}
res["legend"] = legend

t0 = time.time()
present_by_class = {c: 0 for c in classes_ref}
tot_in = 0
for i in ids:
    path = GIS_DIR / f"landcover_{i}.tif"
    geom = out.loc[i, "geometry"]
    g5070 = transform(tr.transform, geom)
    with rasterio.open(path) as s:
        arr = s.read(1)
        tf = s.transform
        cm = s.colormap(1)
        prof = {"crs_epsg": s.crs.to_epsg() if s.crs else None, "dtype": s.dtypes[0], "nodata": s.nodata, "width": s.width, "height": s.height, "res": [s.res[0], s.res[1]],
                "bounds": list(s.bounds), "compression": str(s.compression.value) if s.compression else None, "tiled": bool(s.profile.get("tiled")),
                "block": [s.profile.get("blockxsize"), s.profile.get("blockysize")], "colorinterp": s.colorinterp[0].name, "tags": s.tags(), "image_structure": s.tags(ns="IMAGE_STRUCTURE"),
                "descriptions": list(s.descriptions), "units": list(s.units), "transform": [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]}
        rgb_ok = {c: [cm[c][0], cm[c][1], cm[c][2]] == legend["rgb"][c] for c in legend["rgb"] if c in cm}
    H, W = arr.shape
    left, bottom, right, top = prof["bounds"]
    minx, miny, maxx, maxy = g5070.bounds
    bounds = {"contain_outline": bool(left <= minx and right >= maxx and bottom <= miny and top >= maxy),
              "margin_m": {"west": minx - left, "east": right - maxx, "south": miny - bottom, "north": top - maxy}}
    # cells whose centre lies inside the outline (projected to EPSG:5070)
    c0 = max(0, int(math.floor((minx - tf.c) / tf.a)) - 1); c1 = min(W, int(math.ceil((maxx - tf.c) / tf.a)) + 1)
    r0 = max(0, int(math.floor((maxy - tf.f) / tf.e)) - 1); r1 = min(H, int(math.ceil((miny - tf.f) / tf.e)) + 1)
    inside = np.zeros((H, W), dtype=bool)
    for a in range(r0, r1, 300):
        b = min(r1, a + 300)
        xs = tf.c + (np.arange(c0, c1) + 0.5) * tf.a
        ys = tf.f + (np.arange(a, b) + 0.5) * tf.e
        X, Y = np.meshgrid(xs, ys)
        inside[a:b, c0:c1] = shapely.contains_xy(g5070, X, Y)
    nod = arr == NLCD_NODATA
    n_in = int(inside.sum())
    cells_inside_nodata = int((inside & nod).sum())
    cells_outside_valid = int((~inside & ~nod).sum())
    cnt = np.bincount(arr[inside].astype(np.int64), minlength=256)
    unknown = {int(c): int(cnt[c]) for c in range(256) if cnt[c] and c not in classes_ref and c != NLCD_NODATA}
    n_valid_in = n_in - cells_inside_nodata
    share = {c: cnt[c] / n_in for c in classes_ref}
    forest = sum(cnt[c] for c in FOREST) / n_in
    forest_valid_denominator = sum(cnt[c] for c in FOREST) / max(1, n_valid_in)
    group_share = {g: sum(cnt[c] for c in cs) / n_in for g, cs in groups_ref.items()}
    dom_group = max(group_share, key=group_share.get)
    dom_class = max(classes_ref, key=lambda c: cnt[c])
    for c in classes_ref:
        if cnt[c]:
            present_by_class[c] += 1
    tot_in += n_in
    # the same count read straight off the exported raster (valid cells)
    n_valid_raster = int((~nod).sum())
    forest_raster_only = sum(int((arr == c).sum()) for c in FOREST) / max(1, n_valid_raster)
    # the source cache: every valid exported cell must equal the cached cell
    with rasterio.open(PROJ / "data" / "landcover" / f"annual_nlcd_2011_{i}.tif") as s2:
        src = s2.read(1); tf2 = s2.transform
        cache_nodata = s2.nodata
    dc = (tf.c - tf2.c) / tf2.a; dr = (tf.f - tf2.f) / tf2.e
    aligned = abs(dc - round(dc)) < 1e-4 and abs(dr - round(dr)) < 1e-4
    cache = {"aligned_to_cache_grid": bool(aligned), "col_offset": float(dc), "row_offset": float(dr), "cache_shape": list(src.shape)}
    if aligned:
        dc, dr = int(round(dc)), int(round(dr))
        sub = src[dr:dr + H, dc:dc + W]
        cache["window_inside_cache"] = bool(sub.shape == arr.shape)
        if sub.shape == arr.shape:
            vm = ~nod
            cache["n_valid_compared"] = int(vm.sum())
            cache["n_value_mismatches"] = int((sub[vm] != arr[vm]).sum())
            cache["cache_nodata_inside_outline"] = int((sub[inside] == cache_nodata).sum())
    # the README's file table says the outline vertices follow 30 m raster-cell edges: share of vertices (projected to EPSG:5070) that sit on the corners of this raster's cells
    xy = shapely.get_coordinates(g5070)
    fc = (xy[:, 0] - tf.c) / tf.a
    fr = (xy[:, 1] - tf.f) / tf.e
    on_lattice = (np.abs(fc - np.rint(fc)) < 0.01) & (np.abs(fr - np.rint(fr)) < 0.01)            # 0.01 cell = 0.3 m
    lattice = {"n_vertices": int(len(xy)), "share_on_cell_corners_0.3m": float(on_lattice.mean()),
               "share_on_cell_edge_lines_0.3m": float((((np.abs(fc - np.rint(fc)) < 0.01) | (np.abs(fr - np.rint(fr)) < 0.01))).mean())}
    b = bg[i]
    camels_ff = float(v.loc[i, "frac_forest"])
    pb = {
        "short": out.loc[i, "short"], "raster": prof, "bounds": bounds, "cache": cache, "outline_vertices_vs_cell_lattice": lattice,
        "colour_table_equals_csv_all_present_codes": bool(all(rgb_ok.values())), "colour_table_checked_codes": len(rgb_ok),
        "n_cells_inside_mine": n_in, "n_cells_inside_attr": int(out.loc[i, "n_landcover_cells"]), "n_cells_inside_basin_gis": int(b["n_landcover_cells"]),
        "n_valid_cells_in_raster": n_valid_raster, "cells_inside_but_nodata": cells_inside_nodata, "cells_outside_but_valid": cells_outside_valid,
        "unknown_class_codes_inside": unknown,
        "forest_share_mine": float(forest), "forest_share_mine_valid_denominator": float(forest_valid_denominator), "forest_share_from_raster_valid_cells": float(forest_raster_only),
        "forest_share_attr": float(out.loc[i, "forest_share"]), "forest_share_basin_gis": float(b["land_cover_share"]["forest"]),
        "frac_forest_camels": camels_ff, "frac_forest_camels_attr": float(out.loc[i, "camels_frac_forest"]),
        "forest_diff_points_mine": (float(forest) - camels_ff) * 100.0, "forest_diff_points_attr": float(out.loc[i, "forest_diff_camels_points"]),
        "dominant_group_mine": dom_group, "dominant_group_attr": str(out.loc[i, "dominant_land_cover"]), "dominant_group_basin_gis": b["dominant_land_cover"],
        "dominant_group_share_mine": float(group_share[dom_group]), "dominant_group_share_attr": float(out.loc[i, "dominant_group_share"]),
        "dominant_class_mine": f"{dom_class} {classes_ref[dom_class]}", "dominant_class_attr": str(out.loc[i, "dominant_nlcd_class"]),
        "dominant_class_share_mine": float(share[dom_class]), "dominant_class_share_attr": float(out.loc[i, "dominant_nlcd_share"]),
        "group_shares_mine": {g: float(x) for g, x in group_share.items()},
        "group_shares_basin_gis": {g: float(x) for g, x in b["land_cover_share"].items()},
        "class_shares_mine": {f"{c} {classes_ref[c]}": float(share[c]) for c in classes_ref},
        "max_abs_class_share_diff_vs_basin_gis": max(abs(share[c] - b["nlcd_class_share"].get(f"{c} {classes_ref[c]}", 0.0)) for c in classes_ref),
        "camels_dom_land_cover": str(v.loc[i, "dom_land_cover"]),
        "area_from_cells_km2": n_in * 900.0 / 1e6,
        "area_planar_km2_attr": float(out.loc[i, "area_km2"]),
    }
    res["per_basin"][i] = pb
    print(f"{i} {pb['short']:18s} {time.time() - t0:5.1f}s cells {n_in:8d} (attr {pb['n_cells_inside_attr']}, bgis {pb['n_cells_inside_basin_gis']}) forest {forest:.6f} attr {pb['forest_share_attr']:.6f} "
          f"bgis {pb['forest_share_basin_gis']:.6f} CAMELS {camels_ff:.4f} diff {pb['forest_diff_points_mine']:+6.2f} pts  dom {pb['dominant_class_mine']} / {dom_group}  "
          f"cache mism {cache.get('n_value_mismatches')}", flush=True)

res["classes_present_in_n_basins_mine"] = present_by_class
res["legend_basins_present_column"] = {int(r.code): (None if pd.isna(r.basins_present_of_12) else int(r.basins_present_of_12)) for _, r in leg.iterrows()}
res["total_cells_inside_mine"] = tot_in
res["total_cells_inside_basin_gis"] = sum(int(bg[i]["n_landcover_cells"]) for i in ids)
jsave("g4_landcover.json", res)

print()
print("total cells inside:", tot_in, "| basin_gis:", res["total_cells_inside_basin_gis"])
print("classes present (mine):", present_by_class)
print("legend column basins_present_of_12:", res["legend_basins_present_column"])
print("outline vertices on the cell lattice (share on corners / on edge lines):", {i: (round(res["per_basin"][i]["outline_vertices_vs_cell_lattice"]["share_on_cell_corners_0.3m"], 4),
                                                                                    round(res["per_basin"][i]["outline_vertices_vs_cell_lattice"]["share_on_cell_edge_lines_0.3m"], 4)) for i in ids})
print("legend: classes equal basin_gis:", all(legend["classes_equal_basin_gis"].values()), "| groups equal:", all(legend["groups_equal_basin_gis"].values()))
print(f"{'id':9s} {'crs':>5s} {'dtype':>6s} {'nodata':>6s} {'res':>20s} {'comp':>5s} {'tiled':>6s} {'bounds ok':>9s} {'min margin m':>12s} {'palette':>8s} {'csv rgb':>8s} {'in&nodata':>9s} {'out&valid':>9s} {'unknown':>8s}")
for i in ids:
    p = res["per_basin"][i]; r = p["raster"]
    mm = min(p["bounds"]["margin_m"].values())
    print(f"{i:9s} {r['crs_epsg']!s:>5s} {r['dtype']:>6s} {r['nodata']!s:>6s} {r['res'][0]:10.8f},{r['res'][1]:10.8f} {str(r['compression']):>5s} {str(r['tiled']):>6s} {str(p['bounds']['contain_outline']):>9s} {mm:12.2f} "
          f"{r['colorinterp']:>8s} {str(p['colour_table_equals_csv_all_present_codes']):>8s} {p['cells_inside_but_nodata']:9d} {p['cells_outside_but_valid']:9d} {len(p['unknown_class_codes_inside']):8d}")
