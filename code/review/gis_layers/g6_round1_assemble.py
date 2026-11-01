"""FROZEN - ROUND 1 ONLY (kept so that the round-1 record can be traced to its code; do not use for the current export).
This is g6_assemble.py as it was when it wrote round 1 of results/review_gis_layers.json (2026-10-30, export of 2026-10-29 18:25; sha256 of that file e770290b2815980481332b2a6d640e0a2556f5bdfa5f9341b11d2d284fedc5ca), with this note and the two guard lines after the imports added. Its checks 2e, 6b and 6c are written
as ok=False and its findings are the round-1 texts, which were true of the old export only; run against the regenerated export (2026-10-30 11:30) it would repeat them wrongly.
Round 2 is assembled by g6_assemble.py, which computes those flags. It refuses to run unless REVIEW_ALLOW_ROUND1=1 is set.

Assembles the review record from the outputs of g1 .. g5 (and g7 when it ran): the per-basin table (file against the reviewer's recomputation), the seven checks, the
findings with their fixes, the verdict. Also re-reads results/gis_layers/ and proves that nothing in it changed since g1 (names, sizes, mtimes, sha256) and that the review
created no other file in results/ or beside the export. Writes OUT/review_gis_layers.json; with --write-into-project also results/review_gis_layers.json."""
import statistics
import time
from g_common import *

if os.environ.get("REVIEW_ALLOW_ROUND1") != "1":
    raise SystemExit("g6_round1_assemble.py is frozen (round 1 only). Round 2 is assembled by g6_assemble.py.")
if "--write-into-project" in sys.argv:
    raise SystemExit("g6_round1_assemble.py must never overwrite results/review_gis_layers.json: round 1 is kept as it is.")
import g_common
import geopandas as gpd
import pyproj
import rasterio
import shapely

ids, audit = basins()
t, v = camels()
rec = jload(RECORD)
bg = jload("results/basin_gis.json")["basins"]
g1, g2, g3, g4, g5 = (jout(f"g{k}_{n}.json") for k, n in ((1, "files"), (2, "vectors"), (3, "elevation"), (4, "landcover"), (5, "readme")))
try:
    g7 = jout("g7_rebuild.json")
except Exception:
    g7 = {"skipped": True}
write_in = "--write-into-project" in sys.argv

# ---- comparison engine: every value compared is counted, every difference beyond its tolerance is listed --------------------------------------------------------------
N = {"values_compared": 0}
MISMATCH = []


def cmp(basin, label, a, b, tol=0.0):
    N["values_compared"] += 1
    if isinstance(a, str) or isinstance(b, str) or isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        ok = a == b
        d = None
    else:
        d = float(a) - float(b)
        ok = abs(d) <= tol
    if not ok:
        MISMATCH.append({"basin": basin, "what": label, "file": a if not isinstance(a, float) else float(a), "reviewer": b if not isinstance(b, float) else float(b), "diff": d, "tol": tol})
    return d


per = {}
dev = {"area_km2": 0.0, "area_geodesic": 0.0, "elev_mean": 0.0, "forest_share": 0.0, "area_vs_basin_gis": 0.0, "elev_vs_basin_gis": 0.0, "forest_vs_basin_gis": 0.0}
for i in ids:
    a, b, c = g2["per_basin"][i], g3["per_basin"][i], g4["per_basin"][i]
    bgi = bg[i]
    d_area = cmp(i, "area_km2 (planar EPSG:5070): file attribute vs own pyproj+shapely", a["area_km2_attr"], a["area_planar_5070_km2_mine"], 1e-6)
    d_geod = cmp(i, "area_geodesic_km2: file vs own GRS80 ring sum", a["area_geodesic_km2_attr"], a["area_geodesic_grs80_km2_mine"], 1e-6)
    d_abg = cmp(i, "area_km2 vs results/basin_gis.json area_albers_km2", a["area_km2_attr"], a["area_albers_km2_basin_gis"], TOL["area_km2"])
    cmp(i, "outline_file_area_km2 vs the shapefile AREA field / 1e6", a["outline_file_area_km2_attr"], a["outline_file_area_km2_dbf"], 1e-6)
    cmp(i, "camels_area_gages2_km2 vs camels_topo.txt", a["camels_area_gages2_attr"], a["camels_area_gages2"], 1e-9)
    cmp(i, "camels_area_geospa_fabric_km2 vs camels_topo.txt", a["camels_area_gf_attr"], a["camels_area_gf"], 1e-9)
    cmp(i, "area_diff_gages2_pct vs own", a["area_vs_gages2_pct_attr"], a["area_vs_gages2_pct_mine"], 1e-6)
    cmp(i, "n_parts vs own", a["n_parts_attr"], a["n_parts"])
    cmp(i, "outline valid", a["is_valid"], True)
    cmp(i, "camels_elev_mean_m vs camels_topo.txt", a["elev_mean_camels_attr"], a["elev_mean_camels"], 1e-9)
    cmp(i, "outline_file_elev_mean_m vs the shapefile elev_mean field", a["outline_file_elev_mean_attr"], a["outline_file_elev_mean_dbf"], 1e-6)
    cmp(i, "camels_frac_forest vs camels_vege.txt", a["frac_forest_camels_attr"], a["frac_forest_camels"], 1e-9)
    cmp(i, "camels_dom_land_cover vs camels_vege.txt", a["dom_land_cover_camels_attr"], a["dom_land_cover_camels"])
    cmp(i, "camels_dom_land_cover_frac vs camels_vege.txt", a["dom_land_cover_frac_camels_attr"], a["dom_land_cover_frac_camels"], 1e-9)
    # elevation
    d_el = cmp(i, "elev_mean_m: file vs own cos(lat)-weighted mean of the exported raster, cells with centre inside (shapely.contains_xy)", b["elev_mean_attr"], b["elev_mean_weighted_mine"], 1e-6)
    d_ebg = cmp(i, "elev_mean_m vs results/basin_gis.json", b["elev_mean_attr"], b["elev_mean_basin_gis"], TOL["elev_m"])
    cmp(i, "elev_min_m", b["elev_min_attr"], b["elev_min_mine"]); cmp(i, "elev_max_m", b["elev_max_attr"], b["elev_max_mine"])
    cmp(i, "elev_min_m vs basin_gis", b["elev_min_attr"], b["elev_min_basin_gis"]); cmp(i, "elev_max_m vs basin_gis", b["elev_max_attr"], b["elev_max_basin_gis"])
    cmp(i, "n_elevation_cells vs own", b["n_cells_inside_attr"], b["n_cells_inside_mine"]); cmp(i, "n_elevation_cells vs basin_gis", b["n_cells_inside_attr"], b["n_cells_inside_basin_gis"])
    cmp(i, "elevation voids inside = 0", b["n_voids_inside_mine"], 0); cmp(i, "n_elevation_voids vs basin_gis", b["n_voids_inside_mine"], b["n_voids_basin_gis"])
    cmp(i, "valid cells written vs record n_elevation_cells_written", b["ring"]["n_valid_cells"], b["n_written_valid_cells_attr_json"])
    cmp(i, "elev_diff_camels_m vs own", b["elev_diff_camels_attr"], b["elev_diff_camels_mine_weighted"], 1e-6)
    cmp(i, "elevation source tiles: values equal (mismatches)", b["source_tiles"]["n_value_mismatches"], 0)
    # land cover
    d_fo = cmp(i, "forest_share (NLCD 41+42+43): file vs own count inside the projected outline", c["forest_share_attr"], c["forest_share_mine"], 1e-9)
    d_fbg = cmp(i, "forest_share vs results/basin_gis.json", c["forest_share_attr"], c["forest_share_basin_gis"], TOL["share"])
    cmp(i, "forest_diff_camels_points vs own", c["forest_diff_points_attr"], c["forest_diff_points_mine"], 1e-6)
    cmp(i, "n_landcover_cells vs own", c["n_cells_inside_attr"], c["n_cells_inside_mine"]); cmp(i, "n_landcover_cells vs basin_gis", c["n_cells_inside_attr"], c["n_cells_inside_basin_gis"])
    cmp(i, "land-cover valid cells = cells inside", c["n_valid_cells_in_raster"], c["n_cells_inside_mine"])
    cmp(i, "dominant NLCD class", c["dominant_class_attr"], c["dominant_class_mine"]); cmp(i, "dominant NLCD share", c["dominant_class_share_attr"], c["dominant_class_share_mine"], 1e-9)
    cmp(i, "dominant group", c["dominant_group_attr"], c["dominant_group_mine"]); cmp(i, "dominant group vs basin_gis", c["dominant_group_attr"], c["dominant_group_basin_gis"])
    cmp(i, "dominant group share", c["dominant_group_share_attr"], c["dominant_group_share_mine"], 1e-9)
    for gname, gv in c["group_shares_mine"].items():
        cmp(i, f"group share {gname} vs basin_gis", gv, c["group_shares_basin_gis"][gname], TOL["share"])
    for cname, cv in c["class_shares_mine"].items():
        cmp(i, f"class share {cname} vs basin_gis", cv, bg[i]["nlcd_class_share"].get(cname, 0.0), TOL["share"])
    cmp(i, "land-cover cells source cache mismatches", c["cache"]["n_value_mismatches"], 0)
    cmp(i, "land-cover cells inside but nodata", c["cells_inside_but_nodata"], 0); cmp(i, "land-cover cells outside but valid", c["cells_outside_but_valid"], 0)
    # gauges (against the CAMELS table positions, which the README and the source attribute name, and against results/basin_gis.json)
    cmp(i, "gauge lat (file) vs camels_topo.txt gauge_lat", a["gauge_lat_file"], a["gauge_lat_camels_topo"], 1e-9)
    cmp(i, "gauge lon (file) vs camels_topo.txt gauge_lon", a["gauge_lon_file"], a["gauge_lon_camels_topo"], 1e-9)
    cmp(i, "elev_at_gauge_m (file) vs results/basin_gis.json gauge.elev_m", b["gauge_cell"]["attr"], b["gauge_cell"]["basin_gis"], 0.0)
    cmp(i, "distance_to_outline_km (file) vs results/basin_gis.json gauge.distance_to_outline_km", a["gauge_dist_edge_km_attr"], a["gauge_dist_edge_km_basin_gis"], 1e-6)
    cmp(i, "gauge inside its outline (file point)", a["gauge_inside_file_point"], True)
    cmp(i, "elev_at_gauge_m (file) vs raster cell at the file point", b["gauge_cell"]["attr"], b["gauge_cell"]["elev_at_exported_point"])
    dev["area_km2"] = max(dev["area_km2"], abs(d_area)); dev["area_geodesic"] = max(dev["area_geodesic"], abs(d_geod)); dev["elev_mean"] = max(dev["elev_mean"], abs(d_el))
    dev["forest_share"] = max(dev["forest_share"], abs(d_fo)); dev["area_vs_basin_gis"] = max(dev["area_vs_basin_gis"], abs(d_abg))
    dev["elev_vs_basin_gis"] = max(dev["elev_vs_basin_gis"], abs(d_ebg)); dev["forest_vs_basin_gis"] = max(dev["forest_vs_basin_gis"], abs(d_fbg))
    per[i] = {
        "short": a["short"], "landscape": audit[i]["landscape"],
        "area_km2": {"file": a["area_km2_attr"], "reviewer_epsg5070": a["area_planar_5070_km2_mine"], "basin_gis_albers": a["area_albers_km2_basin_gis"], "file_minus_reviewer": d_area,
                     "geodesic_file": a["area_geodesic_km2_attr"], "geodesic_reviewer_grs80": a["area_geodesic_grs80_km2_mine"], "shapefile_AREA_km2": a["outline_file_area_km2_dbf"],
                     "camels_area_gages2": a["camels_area_gages2"], "pct_vs_camels_file": a["area_vs_gages2_pct_attr"], "pct_vs_camels_reviewer": a["area_vs_gages2_pct_mine"]},
        "elev_mean_m": {"file": b["elev_mean_attr"], "reviewer_cos_weighted": b["elev_mean_weighted_mine"], "reviewer_unweighted": b["elev_mean_unweighted_mine"], "basin_gis": b["elev_mean_basin_gis"],
                        "file_minus_reviewer": d_el, "camels": b["elev_mean_camels"], "diff_vs_camels_file": b["elev_diff_camels_attr"], "diff_vs_camels_reviewer": b["elev_diff_camels_mine_weighted"],
                        "shapefile_elev_mean_field": b["elev_mean_outline_file"], "more_than_25_m_from_camels": bool(abs(b["elev_diff_camels_mine_weighted"]) > MOUNTAIN_THRESHOLD_M),
                        "min_max_file": [b["elev_min_attr"], b["elev_max_attr"]], "min_max_reviewer": [b["elev_min_mine"], b["elev_max_mine"]]},
        "elevation_cells": {"inside_file": b["n_cells_inside_attr"], "inside_reviewer": b["n_cells_inside_mine"], "inside_basin_gis": b["n_cells_inside_basin_gis"], "voids_inside": b["n_voids_inside_mine"],
                            "valid_cells_written": b["ring"]["n_valid_cells"], "cells_compared_with_source_tiles": b["source_tiles"]["n_compared"], "source_tile_value_mismatches": b["source_tiles"]["n_value_mismatches"],
                            "source_tiles": b["source_tiles"]["tiles_compared"]},
        "forest_share": {"file": c["forest_share_attr"], "reviewer": c["forest_share_mine"], "basin_gis": c["forest_share_basin_gis"], "file_minus_reviewer": d_fo, "camels_frac_forest": c["frac_forest_camels"],
                         "points_vs_camels_file": c["forest_diff_points_attr"], "points_vs_camels_reviewer": c["forest_diff_points_mine"]},
        "land_cover_cells": {"inside_file": c["n_cells_inside_attr"], "inside_reviewer": c["n_cells_inside_mine"], "inside_basin_gis": c["n_cells_inside_basin_gis"], "inside_but_nodata": c["cells_inside_but_nodata"],
                             "outside_but_valid": c["cells_outside_but_valid"], "cells_compared_with_cached_nlcd": c["cache"]["n_valid_compared"], "cache_value_mismatches": c["cache"]["n_value_mismatches"]},
        "dominant": {"nlcd_class_file": c["dominant_class_attr"], "nlcd_class_reviewer": c["dominant_class_mine"], "group_file": c["dominant_group_attr"], "group_reviewer": c["dominant_group_mine"],
                     "group_basin_gis": c["dominant_group_basin_gis"], "group_share_file": c["dominant_group_share_attr"], "group_share_reviewer": c["dominant_group_share_mine"]},
        "elevation_raster": {"crs_epsg": b["raster"]["crs_epsg"], "dtype": b["raster"]["dtype"], "nodata": b["raster"]["nodata"], "size_wh": [b["raster"]["width"], b["raster"]["height"]],
                             "res_arcsec": [b["raster"]["res"][0] * 3600, b["raster"]["res"][1] * 3600], "cell_m_ew_ns": [b["grid"]["cell_m_ew_at_raster_centre"], b["grid"]["cell_m_ns_at_raster_centre"]],
                             "cell_centres_on_whole_arcseconds": bool(abs(b["grid"]["left_offset_in_cells"] - 0.5) < 1e-6 and abs(b["grid"]["top_offset_in_cells"] - 0.5) < 1e-6),
                             "bounds_contain_outline": b["grid"]["bounds_contain_outline"], "margin_arcsec_min": min(b["grid"]["margin_arcsec"].values()), "margin_arcsec": b["grid"]["margin_arcsec"],
                             "compression_predictor_tiled": [b["raster"]["compression"], b["raster"]["image_structure"].get("PREDICTOR"), b["raster"]["tiled"], b["raster"]["block"]],
                             "valid_cells_outside_reviewer_15arcsec_buffer": b["ring"]["valid_outside_my_buffer"], "nodata_cells_inside_reviewer_15arcsec_buffer": b["ring"]["nodata_inside_my_buffer"]},
        "landcover_raster": {"crs_epsg": c["raster"]["crs_epsg"], "dtype": c["raster"]["dtype"], "nodata": c["raster"]["nodata"], "size_wh": [c["raster"]["width"], c["raster"]["height"]], "res_m": c["raster"]["res"],
                             "bounds_contain_outline": c["bounds"]["contain_outline"], "margin_m_min": min(c["bounds"]["margin_m"].values()), "colour_table_equals_csv": c["colour_table_equals_csv_all_present_codes"],
                             "compression_tiled": [c["raster"]["compression"], c["raster"]["tiled"], c["raster"]["block"]], "outline_vertices_on_cell_edge_lines": c["outline_vertices_vs_cell_lattice"]["share_on_cell_edge_lines_0.3m"]},
        "gauge": {"file_lat_lon": [a["gauge_lat_file"], a["gauge_lon_file"]], "camels_topo_lat_lon": [a["gauge_lat_camels_topo"], a["gauge_lon_camels_topo"]],
                  "data_audit_lat_lon_3_decimals": [a["gauge_lat_data_audit"], a["gauge_lon_data_audit"]], "file_equals_data_audit_rounding": a["gauge_file_equals_data_audit_rounded"],
                  "displacement_file_vs_camels_m": a["gauge_displacement_file_vs_camels_m"], "inside_outline_file_point": a["gauge_inside_file_point"], "inside_outline_camels_point": a["gauge_inside_camels_point"],
                  "distance_to_edge_km_file_attr": a["gauge_dist_edge_km_attr"], "distance_to_edge_km_reviewer_file_point": a["gauge_dist_edge_km_file_point_mine"],
                  "distance_to_edge_km_reviewer_camels_point": a["gauge_dist_edge_km_camels_point_mine"], "distance_basin_gis": a["gauge_dist_edge_km_basin_gis"],
                  "elev_at_gauge_m_file": b["gauge_cell"]["attr"], "elev_at_gauge_m_reviewer_file_point": b["gauge_cell"]["elev_at_exported_point"],
                  "elev_at_gauge_m_reviewer_camels_point": b["gauge_cell"]["elev_at_camels_5dec_point"], "elev_at_gauge_m_basin_gis": b["gauge_cell"]["basin_gis"],
                  "lowest_cell_of_basin_m": b["elev_min_mine"]},
    }

# ---- the gauge figures used in the findings -----------------------------------------------------------------------------------------------------------------------------------
disp = [per[i]["gauge"]["displacement_file_vs_camels_m"] for i in ids]
elev_diff = {i: (per[i]["gauge"]["elev_at_gauge_m_file"], per[i]["gauge"]["elev_at_gauge_m_basin_gis"]) for i in ids if per[i]["gauge"]["elev_at_gauge_m_file"] != per[i]["gauge"]["elev_at_gauge_m_basin_gis"]}
dist_diff = {i: abs(per[i]["gauge"]["distance_to_edge_km_file_attr"] - per[i]["gauge"]["distance_basin_gis"]) for i in ids}
worst_dist_i = max(dist_diff, key=dist_diff.get)
above12 = {i: per[i]["gauge"]["elev_at_gauge_m_file"] - per[i]["gauge"]["lowest_cell_of_basin_m"] for i in ids if per[i]["gauge"]["elev_at_gauge_m_file"] - per[i]["gauge"]["lowest_cell_of_basin_m"] > 12}
above12_5 = {i: per[i]["gauge"]["elev_at_gauge_m_basin_gis"] - per[i]["gauge"]["lowest_cell_of_basin_m"] for i in ids}
sh = {i: per[i]["short"] for i in ids}
max_edge_file = max(per[i]["gauge"]["distance_to_edge_km_reviewer_file_point"] for i in ids)
max_edge_camels = max(per[i]["gauge"]["distance_to_edge_km_reviewer_camels_point"] for i in ids)

def agree(v, unit=""):
    """A deviation for the prose: 'to the last digit' when it is 0, else 'to <value> <unit>'."""
    return "to the last digit" if v == 0 else f"to {v:.0e}" + (f" {unit}" if unit else "")


# ---- the seven checks ---------------------------------------------------------------------------------------------------------------------------------------------------------
claims = {c["claim"]: c for c in g5["claims"]}


def claim_ok(prefix):
    hits = [c for c in g5["claims"] if c["claim"].startswith(prefix)]
    assert hits, prefix
    return all(c["ok"] for c in hits)


vec = g2["checks"]
checks = [
    {"id": "1", "what": "every file of results/gis_layers.json 'files' exists with the recorded size and sha256; no other file in the folder; total bytes; script hashes; inputs read", "ok": bool(g1["all_exist_with_recorded_bytes_and_sha256"] and not g1["extra_files_in_folder"] and not g1["missing_files"] and g1["total_bytes_equal"]
              and all(x["equal"] for x in g1["code_hashes"].values()) and g1["inputs_read_equal"]["n"] == g1["inputs_read_equal"]["n_equal"] and g1["record_is_strict_json"] is True),
     "evidence": {"files": g1["n_files_recorded"], "on_disk": g1["n_files_on_disk"], "sum_bytes": g1["sum_of_file_bytes"], "scripts_equal": {k: x["equal"] for k, x in g1["code_hashes"].items()}, "inputs_read_equal": g1["inputs_read_equal"],
                  "record_is_strict_json": g1["record_is_strict_json"]}},
    {"id": "2a", "what": "GeoPackage: layers outlines + gauges, 12 + 12 features, EPSG:4269, ids/names/landscapes in the data_audit order, outlines valid, sqlite integrity ok", "ok": bool(vec["layer_names_are_outlines_gauges"] and vec["n_outlines"] == 12 and vec["n_gauges"] == 12
              and vec["crs_outlines"] == "EPSG:4269" and vec["crs_gauges"] == "EPSG:4269" and vec["outline_ids_in_audit_order"] and vec["gauge_ids_in_audit_order"] and vec["outline_names_equal_audit"]
              and vec["landscapes_equal_audit"] and all(vec["outline_validity"].values()) and g2["sqlite"]["integrity_check"] == "ok"),
     "evidence": {"layers": g2["layers"], "application_id": g2["sqlite"]["application_id"], "user_version": g2["sqlite"]["user_version"], "rtree": g2["sqlite"]["has_rtree_for"]}},
    {"id": "2b", "what": "planar area in EPSG:5070 recomputed (pyproj + shapely) equals the attribute area_km2, results/basin_gis.json and the shapefile AREA field; geodesic area equal; outlines exactly the source geometry (10) or its buffer(0) repair (2)", "ok": bool(dev["area_km2"] <= 1e-6 and dev["area_vs_basin_gis"] <= TOL["area_km2"]),
     "evidence": {"max_abs_file_minus_reviewer_km2": dev["area_km2"], "max_abs_file_minus_basin_gis_km2": dev["area_vs_basin_gis"], "max_abs_geodesic_diff_km2": dev["area_geodesic"],
                  "max_abs_reviewer_minus_shapefile_AREA_km2": max(abs(per[i]["area_km2"]["reviewer_epsg5070"] - per[i]["area_km2"]["shapefile_AREA_km2"]) for i in ids)}},
    {"id": "2c", "what": "area against CAMELS area_gages2: attribute area_diff_gages2_pct equals the reviewer's; range of the difference", "ok": True,
     "evidence": {"min_pct": min(per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids), "max_pct": max(per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids),
                  "median_pct": statistics.median(per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids)}},
    {"id": "2d", "what": "every gauge point lies inside its outline (exported points and the 5-decimal CAMELS positions)", "ok": bool(all(per[i]["gauge"]["inside_outline_file_point"] and per[i]["gauge"]["inside_outline_camels_point"] for i in ids)),
     "evidence": {"max_distance_to_edge_km_file_points": max_edge_file, "max_distance_to_edge_km_camels_points": max_edge_camels}},
    {"id": "2e", "what": "gauge points are at the CAMELS gauge_lat/gauge_lon (the README and the layer's source attribute say so); elev_at_gauge_m and distance equal results/basin_gis.json", "ok": False,
     "evidence": {"displacement_m": {"min": min(disp), "median": statistics.median(disp), "max": max(disp)}, "elev_differs_in": len(elev_diff), "elev_pairs_file_vs_basin_gis": {sh[i]: list(p) for i, p in elev_diff.items()},
                  "largest_distance_difference_km": dist_diff[worst_dist_i], "largest_distance_difference_basin": sh[worst_dist_i]}},
    {"id": "3a", "what": "12 elevation GeoTIFFs: EPSG:4326, int16, nodata -32768, 1.000000 arc-second, cell centres on whole arc-seconds, bounds contain the outline (>= 15.5 arc-seconds all round), LZW predictor 2, tiled 256", "ok": bool(all(
        per[i]["elevation_raster"]["crs_epsg"] == 4326 and per[i]["elevation_raster"]["dtype"] == "int16" and per[i]["elevation_raster"]["nodata"] == DEM_NODATA
        and abs(per[i]["elevation_raster"]["res_arcsec"][0] - 1) < 1e-9 and abs(per[i]["elevation_raster"]["res_arcsec"][1] - 1) < 1e-9 and per[i]["elevation_raster"]["cell_centres_on_whole_arcseconds"]
        and per[i]["elevation_raster"]["bounds_contain_outline"] and per[i]["elevation_raster"]["margin_arcsec_min"] > 15.0
        and per[i]["elevation_raster"]["compression_predictor_tiled"] == ["LZW", "2", True, [256, 256]] for i in ids)),
     "evidence": {"min_margin_arcsec": min(per[i]["elevation_raster"]["margin_arcsec_min"] for i in ids)}},
    {"id": "3b", "what": "mean elevation of the cells whose centre lies inside the outline (cos-latitude weighted) recomputed with shapely.contains_xy equals the attribute elev_mean_m and results/basin_gis.json; cell counts, min, max, voids equal; every exported sample equals the source tile sample", "ok": bool(dev["elev_mean"] <= 1e-6 and dev["elev_vs_basin_gis"] <= TOL["elev_m"]
              and all(per[i]["elevation_cells"]["source_tile_value_mismatches"] == 0 for i in ids) and g3["totals"]["voids_inside"] == 0
              and all(g3["per_basin"][i]["source_tiles"]["tiles_equal_record"] for i in ids)),
     "evidence": {"max_abs_file_minus_reviewer_m": dev["elev_mean"], "max_abs_file_minus_basin_gis_m": dev["elev_vs_basin_gis"], "cells_inside_total": g3["totals"]["cells_inside_mine"], "valid_cells_written_total": g3["totals"]["cells_written_valid"],
                  "cells_compared_with_source_tiles": sum(per[i]["elevation_cells"]["cells_compared_with_source_tiles"] for i in ids), "unweighted_mean_differs_from_weighted_by_up_to_m": max(abs(per[i]["elev_mean_m"]["reviewer_unweighted"] - per[i]["elev_mean_m"]["reviewer_cos_weighted"]) for i in ids)}},
    {"id": "3c", "what": "elevation difference against CAMELS elev_mean: attribute equals the reviewer's; exactly three basins beyond 25 m", "ok": bool(sorted(i for i in ids if per[i]["elev_mean_m"]["more_than_25_m_from_camels"]) == ["02137727", "03439000", "03463300"]),
     "evidence": {sh[i]: {"file": per[i]["elev_mean_m"]["file"], "camels": per[i]["elev_mean_m"]["camels"], "diff_m": per[i]["elev_mean_m"]["diff_vs_camels_reviewer"], "shapefile_elev_mean_field": per[i]["elev_mean_m"]["shapefile_elev_mean_field"]} for i in ("02137727", "03439000", "03463300")}},
    {"id": "4a", "what": "12 land-cover GeoTIFFs: EPSG:5070, uint8, nodata 250, 30 m, colour table equals the CSV, LZW tiled, bounds contain the outline (projected), cells outside the outline nodata and none inside", "ok": bool(all(
        per[i]["landcover_raster"]["crs_epsg"] == 5070 and per[i]["landcover_raster"]["dtype"] == "uint8" and per[i]["landcover_raster"]["nodata"] == NLCD_NODATA and all(abs(r - 30) < 1e-6 for r in per[i]["landcover_raster"]["res_m"])
        and per[i]["landcover_raster"]["bounds_contain_outline"] and per[i]["landcover_raster"]["colour_table_equals_csv"] and per[i]["land_cover_cells"]["inside_but_nodata"] == 0 and per[i]["land_cover_cells"]["outside_but_valid"] == 0 for i in ids)),
     "evidence": {"min_margin_m": min(per[i]["landcover_raster"]["margin_m_min"] for i in ids), "pixel_size_m_note": "30.00000001 to 30.00000012 m in the file (the WCS grid), within 1e-6 of 30"}},
    {"id": "4b", "what": "forest share (NLCD 41+42+43) recomputed inside the projected outline equals the attribute forest_share and results/basin_gis.json; cell counts and all 15 class shares and 7 group shares equal; every cell equals the cached NLCD cell", "ok": bool(dev["forest_share"] <= 1e-9 and dev["forest_vs_basin_gis"] <= TOL["share"]
              and all(per[i]["land_cover_cells"]["cache_value_mismatches"] == 0 for i in ids) and g4["total_cells_inside_mine"] == 7912882),
     "evidence": {"max_abs_file_minus_reviewer": dev["forest_share"], "max_abs_file_minus_basin_gis": dev["forest_vs_basin_gis"], "cells_inside_total": g4["total_cells_inside_mine"],
                  "cells_compared_with_cache": sum(per[i]["land_cover_cells"]["cells_compared_with_cached_nlcd"] for i in ids)}},
    {"id": "5a", "what": "README check tables (area, mean elevation, dominant land cover + forest) equal results/gis_layers.json check_table and the reviewer's recomputation, cell by cell within the printed rounding", "ok": claim_ok("README check tables equal"),
     "evidence": {"numeric_cells_compared_twice": g5["tables"]["numeric_cells_compared"], "text_cells_compared_twice": g5["tables"]["text_cells_compared"], "mismatches": 0}},
    {"id": "5b", "what": "README summary sentence, '12 of 12 agree', the known-difference paragraph (quotes the results/basin_gis.json note word for word; +47/+100/+163 m) and the note's own figures", "ok": bool(claim_ok("README summary sentence") and claim_ok("README says '12 of 12") and claim_ok("12 of 12 basins agree")
              and claim_ok("README quotes the 'Elevation:' note") and claim_ok("results/gis_layers.json known_note") and claim_ok("exactly three basins") and claim_ok("note:")), "evidence": {}},
    {"id": "6a", "what": "licences: every licence in the README source text, in results/gis_layers.json licences and in every files[].licence is UNVERIFIED (no licence recorded in results/data_flow.json or results/basin_gis.json - confirmed: neither holds a licence field); no licence text in the GeoPackage, GeoJSON, CSV or raster tags; the README asserts none of its own",
     "ok": bool(claim_ok("README source text") and claim_ok("results/data_flow.json and results/basin_gis.json hold no licence") and claim_ok("README asserts no licence") and claim_ok("no raster tag") and claim_ok("results/gis_layers.json: every files")
                and claim_ok("README: the 'login needed' text")), "evidence": {"inherit_sentence": g5["inherit_sentence"]}},
    {"id": "6b", "what": "licences as a reader sees them: the sources table of the README rendered (python-markdown tables, same rule as GitHub) shows UNVERIFIED in the licence column of all three rows; no angle-bracket text outside backticks", "ok": False,
     "evidence": {"rows_with_wrong_cell_count": claims["every README table row has the header's number of cells (a literal '|' inside a cell splits it)"]["detail"],
                  "rendered": claims["README rendered (python-markdown tables): the licence column of each of the 3 rows of the sources table shows UNVERIFIED"]["detail"]}},
    {"id": "6c", "what": "the citation sentence 'data/camels_us/readme.txt asks that publications cite ... Newman et al. 2015 (time series and outlines)' is what readme.txt says", "ok": False,
     "evidence": claims["README: 'data/camels_us/readme.txt asks that publications cite Addor et al. 2017 (attributes) and Newman et al. 2015 (time series and outlines)' - is the request for the outlines in that file?"]["detail"]},
    {"id": "7", "what": "the disclosure line 'Claude Code wrote and ran this script under the team lead's direction; the team's own check of these layers is not done' is in the README (line 5, second half in bold) and, with the date inserted, in results/gis_layers.json disclosure",
     "ok": bool(claim_ok("README carries the disclosure") and claim_ok("results/gis_layers.json carries the disclosure")),
     "evidence": {"readme_line": claims["README carries the disclosure line (markdown emphasis removed), within its first lines"]["detail"]["line_in_readme"], "json": rec["disclosure"],
                  "readme_mentions_an_AI_review": g5["readme_mentions_ai_review"]}},
    {"id": "x1", "what": "extra: a clean rebuild of code/27_gis_layers.py in a scratch project root (copies of the inputs) reproduces the sha256 of the 27 files the README calls deterministic (24 GeoTIFFs, 2 GeoJSON, the CSV); only the GeoPackage (write time) and the README (time stamp, gpkg hash) differ", "ok": bool(g7.get("returncode") == 0 and g7.get("n_equal_to_recorded") == 27 and sorted(g7.get("not_equal_to_recorded", [])) == ["README.md", "basins_12.gpkg"]) if not g7.get("skipped") else None,
     "evidence": {k: g7.get(k) for k in ("returncode", "seconds", "n_equal_to_recorded", "not_equal_to_recorded", "rebuilt_record_check_table_equal")} if not g7.get("skipped") else "skipped"},
]

# ---- the folder is unchanged ----------------------------------------------------------------------------------------------------------------------------------------------------
before = jout("manifest_before.json")
now = {n.name: {"bytes": n.stat().st_size, "mtime_ns": n.stat().st_mtime_ns, "sha256": sha256_file(n)} for n in sorted(GIS_DIR.iterdir())}
changed = sorted(n for n in set(before["gis_layers"]) | set(now) if before["gis_layers"].get(n) != now.get(n))
res_names_now = sorted(p.name for p in (PROJ / "results").iterdir())
new_in_results = sorted(set(res_names_now) - set(before["results_names"]))
integrity = {"export_folder_files_changed_since_g1": changed, "export_folder_file_count": len(now), "new_names_in_results_since_g1": new_in_results,
             "pycache_in_review_folder": (HERE / "__pycache__").exists(), "gis_layers_json_sha256": sha256_file(PROJ / RECORD), "readme_sha256": sha256_file(GIS_DIR / "README.md")}
integrity["side_files_in_export_folder"] = [n for n in now if re.search(r"(-wal|-shm|-journal|\.aux\.xml)$", n)]
checks.append({"id": "x2", "what": "extra: the review changed nothing in results/gis_layers/ (names, sizes, mtimes, sha256 equal to the manifest taken before the other checks) and left no side file there", "ok": bool(not changed and not integrity["side_files_in_export_folder"]), "evidence": integrity})

# ---- findings -------------------------------------------------------------------------------------------------------------------------------------------------------------------
n_failed = sum(1 for c in checks if c["ok"] is False)
fx = []
fx.append({"severity": "must-fix",
           "where": "code/27_gis_layers.py lines 95-100 (B is built from results/data_audit.json 'selected'; the np.allclose(..., atol=1e-3) loop meant to prove data_audit.json and camels_topo.txt agree cannot see a 0.001-degree rounding), lines 185, 232, 245 and 275-278 (gauge cell, inside test, distance, gauges layer); basins_12.gpkg layer 'gauges' (lat, lon, elev_at_gauge_m, distance_to_outline_km, source), gauges_12.geojson, results/gis_layers.json basins.<id>.gauge; README 'What the layers are', Gauges bullet",
           "text": (f"The 12 gauge points are not at the CAMELS positions that the layer's own 'source' attribute ('CAMELS camels_topo.txt gauge_lat, gauge_lon') and the README ('the 12 gauge points at the CAMELS gauge_lat/gauge_lon') name. lat/lon are the 3-decimal copies of results/data_audit.json "
                    f"(Hyco Creek 36.398, -79.197; camels_topo.txt has 36.39778, -79.19667). The exported points lie {min(disp):.1f} to {max(disp):.1f} m (median {statistics.median(disp):.0f} m) from the camels_topo.txt positions (largest: {sh[max(ids, key=lambda k: per[k]['gauge']['displacement_file_vs_camels_m'])]} {max(disp):.1f} m; smallest: {sh[min(ids, key=lambda k: per[k]['gauge']['displacement_file_vs_camels_m'])]} {min(disp):.1f} m). "
                    f"All 12 are inside their outlines at either position and 'at most 0.63 km' holds at either ({max_edge_file:.3f} / {max_edge_camels:.3f} km), but elev_at_gauge_m differs from results/basin_gis.json gauge.elev_m in {len(elev_diff)} of 12 basins "
                    f"({'; '.join(f'{sh[i]} {a} vs {int(b_)}' for i, (a, b_) in elev_diff.items())}) and distance_to_outline_km by up to {dist_diff[worst_dist_i]:.3f} km ({sh[worst_dist_i]} {per[worst_dist_i]['gauge']['distance_to_edge_km_file_attr']:.3f} vs {per[worst_dist_i]['gauge']['distance_basin_gis']:.3f}). "
                    f"The basin_gis.json note that the gauge cell is at most 12 m above the lowest cell of the basin holds at the CAMELS positions (largest {max(above12_5.values()):.0f} m) but not at the exported points ({'; '.join(f'{sh[i]} {d:.0f} m' for i, d in above12.items())}). "
                    f"The check table and the '12 of 12 basins agree with basin_gis.json' line cover area, mean elevation, forest share and cell counts only, so nothing flagged it. "
                    "Fix: in code/27_gis_layers.py after the allclose loop (lines 99-100) add B[\"gauge_lat\"], B[\"gauge_lon\"] = topo.gauge_lat.astype(float), topo.gauge_lon.astype(float) (keep the loop with atol=5.1e-4 as the proof that data_audit.json is the same table rounded to 3 decimals), "
                    "add gauge.elev_m and gauge.distance_to_outline_km against results/basin_gis.json to the check table and to agrees_with_basin_gis, rerun (10 s; the 24 GeoTIFFs, basins_12.geojson and the CSV keep their hashes - the clean rebuild of this review shows it - while basins_12.gpkg, gauges_12.geojson, README.md and gis_layers.json change). "
                    f"Expected afterwards: elev_at_gauge_m = {', '.join(str(int(per[i]['gauge']['elev_at_gauge_m_basin_gis'])) for i in ids)} (in the order of the table) and a largest distance of {max_edge_camels:.3f} km. "
                    "If the layer is to stay as it is, change the README sentence and the 'source' attribute to 'gauge_lat/gauge_lon of results/data_audit.json, rounded to 0.001 degree (up to about 60 m from camels_topo.txt)' and say that elev_at_gauge_m and distance_to_outline_km therefore differ from results/basin_gis.json.")})
fx.append({"severity": "must-fix",
           "where": "results/gis_layers/README.md line 54-55 (section 'Sources and licences', rows of the elevation tiles and the land cover; built in code/27_gis_layers.py where the README rows are written, from results/data_flow.json obtained.url_pattern)",
           "text": ("The licence marking of the elevation product does not survive rendering. The URL pattern 'skadi/<N|S><lat>/<N|S><lat><E|W><lon>.hgt.gz' contains three literal '|' that split the Markdown cell: the row has 7 cells under a 4-column header, and a renderer (python-markdown with tables, same rule as GitHub-flavoured Markdown) drops the excess, so the licence column of the elevation row shows 'S><lat>/<N' and its UNVERIFIED text is not displayed; "
                    "the source text of the README is correct. Also '<N|S>', '<lat>', '<lon>' and the file names elevation_<gauge_id>.tif / landcover_<gauge_id>.tif in the 'used for' cells (lines 54-55) sit outside backticks and are read as HTML tags (shown as 'elevation_.tif'). "
                    "Fix: where the README rows are written, replace '|' by '\\|' in every cell text (or write 'skadi/<N or S>...' inside backticks), put the URL pattern and the two file-name patterns in backticks as the Files table does, and add a check that every table row has the header's number of cells (g5_readme.py of this review does it in 10 lines).")})
fx.append({"severity": "should-fix",
           "where": "code/27_gis_layers.py line 362 (CITE) -> README 'Sources and licences' row 1, results/gis_layers.json licences.outlines_and_attributes and files[].licence of basins_12.gpkg, basins_12.geojson, gauges_12.geojson",
           "text": ("The sentence \"data/camels_us/readme.txt asks that publications cite Addor et al. 2017 (attributes) and Newman et al. 2015 (time series and outlines)\" puts 'and outlines' into a file that does not say it: readme.txt asks for Addor et al. 2017 when the CAMELS attributes are used and for Newman et al. 2015 when the CAMELS time series are used, and mentions neither outlines nor shapefiles. "
                    "That the outlines belong to Newman et al. 2015 is in results/data_flow.json (producer line) and in the header of readme_shapefiles.txt, which asks nothing. "
                    "Fix: 'data/camels_us/readme.txt asks that publications using the CAMELS attributes cite Addor et al. 2017 and publications using the CAMELS time series cite Newman et al. 2015; results/data_flow.json names Newman et al. 2015 as the source of the outlines (no request to cite is recorded for them)'.")})
fx.append({"severity": "note",
           "where": "results/review_gis_layers.json (counts block); code/final_report/numbers.py lines 1470-1478, code/build_slides_short.py lines 154-157, code/build_slides.py _counts_blocks (line 620)",
           "text": (f"Integration: the builders glob results/review_*.json and treat every file but review_documents_* as an experiment review record. Once results/review_gis_layers.json exists, the next rebuild of the documents counts it in review_records, review_rounds, review_checks_listed and review_none_failed (this record's counts block lists {n_failed} failed checks and {len(MISMATCH)} compared values that differ, all of them gauge values, so review_none_failed would read 'a check failed (see the records)' until the gauge points are fixed and re-reviewed), and names 'review_gis_layers' among 'the experiment review records'. "
                    "Fix (either): exclude it where exp is built (numbers.py line 1471: add 'and k != \"review_gis_layers\"', the same filter in build_slides_short.py line 154) and give it its own keys, or accept the count after the gauge fix and a second round (add a 'round_2' block to this record). The documents also quote gis_layers_exported ('outlines, gauges, clipped elevation and land cover; 29 files in results/gis_layers/, checked against the CAMELS tables', sections 3-1, 7-4, 9-1 and the deck); that sentence stays true after the fix (still 29 files), but until then the gauge points are not checked against the CAMELS positions.")})
fx.append({"severity": "note", "where": "README 'Verification' section and the JSON disclosure; HANDOFF.md section 7 ('Claude Code wrote and ran the code under the team lead's direction, and AI agents reviewed it')",
           "text": ("The disclosure line is present word for word in the README (line 5; only the second half is bold) and in results/gis_layers.json (there the date 'on 2026-10-29' is inserted, so the literal line is not a substring of the JSON field). Neither says that an AI agent reviewed the layers, as the HANDOFF rule describes. "
                    "Fix after this record is accepted: add one sentence to the README template and the JSON disclosure, for example 'An independent AI reviewer agent (Claude) recomputed these layers on 2026-10-30 (results/review_gis_layers.json); the team's own check of these layers is not done.', keeping the second clause.")})
fx.append({"severity": "note", "where": "basins_12.gpkg layer 'gauges' column distance_to_outline_km; code/27_gis_layers.py line 233",
           "text": (f"distance_to_outline_km is the geodesic distance to the boundary point that is nearest in DEGREE space (nearest_points on unprojected coordinates), not in metres; at 35 N a degree of longitude is 0.8 of a degree of latitude, so the value can be longer than the true minimum: by up to {max(per[i]['gauge']['distance_to_edge_km_file_attr'] - per[i]['gauge']['distance_to_edge_km_reviewer_file_point'] for i in ids) * 1000:.0f} m here (Catawba 0.200 vs 0.192 km). "
                    "The README's 'at most 0.63 km' is unaffected. Fix (optional): find the nearest point in EPSG:5070 or a local azimuthal projection, or say 'about'.")})
fx.append({"severity": "note", "where": "basins_12.gpkg columns elevation_file, landcover_file (and results/gis_layers.json basins.<id>.elevation_file); README 'Elevation' bullet ('about 462 m')",
           "text": ("(a) The two file columns hold project-relative paths such as 'results/gis_layers/elevation_02077200.tif', which do not resolve when the folder is copied to a teammate or opened in QGIS from elsewhere; bare file names would. "
                    f"(b) 'buffered by 15 arc-seconds (about 462 m)' is true north-south only; east-west the same 15 arc-seconds are {15 * min(per[i]['elevation_raster']['cell_m_ew_ns'][0] for i in ids):.0f} to {15 * max(per[i]['elevation_raster']['cell_m_ew_ns'][0] for i in ids):.0f} m in these basins ('about 460 m north-south, 370 to 410 m east-west').")})
fx.append({"severity": "note", "where": "README 'Sources and licences' intro sentence; results/gis_layers.json licences.note",
           "text": ("'the exported files inherit whatever terms their sources carry' is a legal assumption rather than a recorded fact. It asserts no licence, which is what the rule asks, but 'are derived from sources whose terms are not recorded here' says only what is known.")})
fx.append({"severity": "note", "where": "basins_12.geojson, gauges_12.geojson (GDAL GeoJSON writer defaults)",
           "text": (f"Both files carry the deprecated 'crs' member (urn:ogc:def:crs:OGC:1.3:CRS84) and every exterior ring is clockwise ({g2['geojson']['basins_12.geojson']['exterior_rings_ccw_share']:.0%} counter-clockwise), while RFC 7946 asks for counter-clockwise exterior rings and dropped the 'crs' member; Leaflet, geojson.io and QGIS read the files as they are. Coordinates are multiples of 1e-7 degree and equal the GeoPackage within 5e-8 degree (rounding), attributes identical. "
                    "Optional fix: write with RFC7946=YES.")})
fx.append({"severity": "note", "where": "README 'Check' tables, columns 'diff'",
           "text": "The area and elevation 'diff' columns print signed zeros ('-0.000000', '+0.0000', 12 rows each, the true differences are about 1e-11 km2 and 1e-13 m). Cosmetic; printing '0' or '< 1e-6' reads better."})
fx.append({"severity": "note", "where": "README 'Known difference' paragraph (the three mountain basins), results/basin_gis.json notes",
           "text": ("The sentence is stated as in results/basin_gis.json and its figures reproduce (median +1.6 m; DEM mean 8.4 m from the outline file's own elev_mean field on average, largest -24.7 m South Toe; 920 m averaging moves the mean by at most 2.5 m; rank correlation 0.99; the four mountain basins stay above 600 m). "
                    f"One more fact the layers already hold supports it: the shapefile's own elev_mean field for the three basins is {per['02137727']['elev_mean_m']['shapefile_elev_mean_field']:.1f}, {per['03439000']['elev_mean_m']['shapefile_elev_mean_field']:.1f} and {per['03463300']['elev_mean_m']['shapefile_elev_mean_field']:.1f} m against {per['02137727']['elev_mean_m']['file']:.1f}, {per['03439000']['elev_mean_m']['file']:.1f}, {per['03463300']['elev_mean_m']['file']:.1f} m from the DEM and {per['02137727']['elev_mean_m']['camels']:.1f}, {per['03439000']['elev_mean_m']['camels']:.1f}, {per['03463300']['elev_mean_m']['camels']:.1f} m in camels_topo.txt, "
                    "so the camels_topo.txt column is the odd one out in these three basins, not the DEM. No change needed; the README may add this line (column outline_file_elev_mean_m).")})
fx.append({"severity": "note", "where": "elevation_<gauge_id>.tif, the 15-arc-second ring",
           "text": (f"Against the reviewer's own 15-arc-second buffer (shapely buffer, 64 segments per quarter circle) {sum(per[i]['elevation_raster']['nodata_cells_inside_reviewer_15arcsec_buffer'] for i in ids)} cells in all ({min(per[i]['elevation_raster']['nodata_cells_inside_reviewer_15arcsec_buffer'] for i in ids)} to {max(per[i]['elevation_raster']['nodata_cells_inside_reviewer_15arcsec_buffer'] for i in ids)} per basin, {100 * min(per[i]['elevation_raster']['nodata_cells_inside_reviewer_15arcsec_buffer'] / per[i]['elevation_cells']['valid_cells_written'] for i in ids):.3f} to {100 * max(per[i]['elevation_raster']['nodata_cells_inside_reviewer_15arcsec_buffer'] / per[i]['elevation_cells']['valid_cells_written'] for i in ids):.3f}% of the valid cells) are nodata in the export and {sum(per[i]['elevation_raster']['valid_cells_outside_reviewer_15arcsec_buffer'] for i in ids)} valid cells lie outside it; "
                    "that is the round-join discretisation of the buffer at the ring edge (all cells inside the outline are valid and equal the source tiles). No action.")})

# the 'numbers compared'
n_checks = len(checks)
n_failed = sum(1 for c in checks if c["ok"] is False)
n_mismatch_values = len(MISMATCH)
sev = {k: sum(1 for f in fx if f["severity"] == k) for k in ("must-fix", "should-fix", "note")}
counts = {"checks_listed": n_checks, "failed": n_failed,
          "values_compared": N["values_compared"] + (g5["tables"]["numeric_cells_compared"] + g5["tables"]["text_cells_compared"]) * 2,
          "values_compared_scalars_file_vs_recomputation": N["values_compared"], "readme_table_cells_compared_against_check_table_and_recomputation": (g5["tables"]["numeric_cells_compared"] + g5["tables"]["text_cells_compared"]) * 2,
          "raster_cells_compared_with_their_sources_not_in_values_compared": sum(per[i]["elevation_cells"]["cells_compared_with_source_tiles"] for i in ids) + sum(per[i]["land_cover_cells"]["cells_compared_with_cached_nlcd"] for i in ids),
          "numerical_mismatches": n_mismatch_values, "numerical_mismatches_outside_the_gauges_layer": sum(1 for m in MISMATCH if not str(m["what"]).startswith(("gauge", "elev_at_gauge", "distance_to_outline"))),
          "must_fix": sev["must-fix"], "should_fix": sev["should-fix"], "notes": sev["note"], "basins_checked": 12, "files_hashed": g1["n_files_recorded"]}

verdict = "discrepancies"
verdict_text = (
    "Everything the export computes from the data reproduces: the 12 outlines are the source geometry (10 exactly, 2 after the buffer(0) repair, with the shapefile's own AREA unchanged); "
    f"planar areas in EPSG:5070 equal the attribute {agree(dev['area_km2'], 'km2')} and results/basin_gis.json {agree(dev['area_vs_basin_gis'], 'km2')}; the mean elevations of the cells whose centre lies inside the outline (cos-latitude weighted, found with an algorithm different from the export's rasteriser) equal the attribute {agree(dev['elev_mean'], 'm')} "
    f"and results/basin_gis.json {agree(dev['elev_vs_basin_gis'], 'm')}, with identical cell counts (8,930,167) and no voids; the forest shares equal the attribute {agree(dev['forest_share'])} and results/basin_gis.json {agree(dev['forest_vs_basin_gis'])}, with identical cell counts (7,912,882); "
    "every exported elevation sample equals the source tile sample and every land-cover cell equals the cached NLCD cell; the README's three check tables equal results/gis_layers.json check_table and the recomputation cell by cell; "
    "the three mountain basins (+47, +100, +163 m) are stated as in results/basin_gis.json; a clean rebuild reproduces the sha256 of the 27 deterministic files; the disclosure line is present; the export folder was not touched by the review. "
    "Two things are not right. (1) The gauge points are the 3-decimal rounding of the CAMELS positions (8.8 to 60.4 m off), not the camels_topo.txt positions that the README and the layer's source attribute name, and their elev_at_gauge_m differs from results/basin_gis.json in 9 of 12 basins; the check table does not cover the gauges. "
    "(2) In the rendered README the licence column of the elevation row is lost (literal pipes inside the URL pattern split the table row), so for one of the three products the UNVERIFIED mark is invisible; the source text is right. A third item is wording (the citation sentence). "
    "The gauge fix needs a rerun of the script (the 24 rasters, basins_12.geojson and the CSV keep their hashes); the other two are text fixes in the script's README template. "
    "With the three items fixed (or the gauge rounding stated in the README and the source attribute instead of fixed) nothing else needs to change and the verdict would read 'layers hold with wording fixes' at most; run_checks.sh repeats every check in about 40 s, so a second round is cheap.")

not_checked = [
    "the vertical reference (EGM96) and the SRTM/3DEP make-up of the elevation tiles: taken from the project's records (results/basin_gis.json, data_flow.json), not testable from the files",
    "the NLCD 2011 product itself at MRLC/USGS: only the cached copy in data/landcover (no download, no network use); likewise the CAMELS tables were read from data/camels_us as they are, not against the original download",
    "opening the layers in QGIS or ArcGIS (not run); readers used: geopandas/pyogrio, rasterio with GDAL 3.12.4, sqlite3 (GeoPackage integrity_check and the standard tables), python-markdown (rendering); no GeoPackage validator",
    "whether the CAMELS elev_mean column of the three mountain basins is right or wrong: only that the layers state the difference as results/basin_gis.json does and that the outline file's own field agrees with the DEM",
    "the team's own check of the layers (not done)",
]
env = {"python": sys.version.split()[0], "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "geopandas": gpd.__version__, "shapely": shapely.__version__, "pyproj": pyproj.__version__,
       "numpy": np.__version__, "pandas": pd.__version__, "platform": sys.platform}
env_same = {k: (rec["environment"].get(k) == env.get(k)) for k in env if k in rec["environment"]}

scripts = [f"code/review/gis_layers/{n}" for n in ("g_common.py", "g1_files.py", "g2_vectors.py", "g3_elevation.py", "g4_landcover.py", "g5_readme.py", "g7_rebuild.py", "g6_assemble.py", "run_checks.sh")]
record = {
    "who": "independent AI reviewer agent (Claude), 2026-10-30",
    "human_review": "not done: the team's own check of these layers is still open (team.json untouched by this review)",
    "when": "2026-10-30, after the run of code/27_gis_layers.py (2026-10-29 18:25, results/gis_layers.json 'written'); scripts in code/review/gis_layers/",
    "what": ("Independent review of the GIS-layer export (deliverable D3): code/27_gis_layers.py -> results/gis_layers/ (basins_12.gpkg with layers outlines and gauges, basins_12.geojson, gauges_12.geojson, 12 elevation and 12 land-cover GeoTIFFs, landcover_classes.csv, README.md; 29 files, 9.45 MB) and results/gis_layers.json. "
             "Seven checks with the reviewer's own code (shapely / pyproj / rasterio / numpy calls written for this review, nothing imported from code/27_gis_layers.py or 07_basin_maps.py): (1) all 29 files with the recorded sha256 and size; (2) the GeoPackage - 12 outlines, 12 gauges, EPSG:4269, gauges inside, planar area in EPSG:5070 against the attribute, basin_gis.json and CAMELS area_gages2; "
             "(3) the 12 elevation GeoTIFFs - CRS, nodata, resolution, bounds, mean elevation of the cells with centre inside the outline against the attribute, basin_gis.json and CAMELS; (4) the 12 land-cover GeoTIFFs - CRS, 30 m, nodata 250, forest share (41, 42, 43) against the attribute, basin_gis.json and CAMELS frac_forest; (5) the README check tables against check_table and the recomputation, and the three mountain basins; "
             "(6) licences; (7) the disclosure line. Extras: every exported elevation sample against the source SRTM-format tile and every land-cover cell against the cached NLCD file; the gauge positions against camels_topo.txt; the outline vertices against the shapefile; the README as rendered; a clean rebuild of the script in a scratch tree."),
    "files_reviewed": {"code/27_gis_layers.py": rec["code"]["sha256"], "code/07_basin_maps.py": rec["code"]["related_sha256"], "results/gis_layers.json": integrity["gis_layers_json_sha256"], "results/gis_layers/README.md": integrity["readme_sha256"],
                       "results/gis_layers/*": "28 further files, each sha256 equal to results/gis_layers.json files[] (check 1)"},
    "method": {"area": "outline of the GeoPackage transformed EPSG:4269 -> 5070 with pyproj, shapely area; geodesic area by ring-wise GRS80 sums (pyproj.Geod)",
               "cells_inside": "cell centres of the exported raster tested with shapely.contains_xy against the outline (elevation: in EPSG:4269/4326 degrees; land cover: outline projected to EPSG:5070) - not the rasteriser the export used; the outline vertices lie on the 30 m cell-edge lines, so no centre is near an edge in the land-cover grid",
               "elevation_mean": "sum(z * cos(lat of the cell row)) / sum(cos) over the cells inside (the README's definition), also unweighted",
               "source_copy": "own reader for the SRTM .hgt.gz tiles (big-endian int16, 3601 x 3601, row 0 = north); every valid exported cell mapped to its tile sample by latitude/longitude; the land-cover export against data/landcover/annual_nlcd_2011_<gauge>.tif by the integer cell offset",
               "readme": "the three check tables and the file table parsed from README.md and compared cell by cell with check_table and with the reviewer's numbers within the printed rounding; sentences tested by pattern; rendered with python-markdown (tables)",
               "no_write": "the GeoPackage was read from a scratch copy whose hash was proved equal; GDAL_PAM_ENABLED=NO; before/after manifest of results/gis_layers/ (check x2)"},
    "environment": env, "environment_equals_record_environment": env_same,
    "reproduced": {
        "per_basin": per,
        "maximum_deviation_file_vs_reviewer": dev,
        "totals": {"elevation_cells_inside": g3["totals"]["cells_inside_mine"], "elevation_cells_written_with_buffer": g3["totals"]["cells_written_valid"], "elevation_voids_inside": g3["totals"]["voids_inside"],
                   "land_cover_cells_inside": g4["total_cells_inside_mine"], "land_cover_classes_present": sum(1 for n in g4["classes_present_in_n_basins_mine"].values() if n > 0),
                   "files": g1["n_files_recorded"], "bytes": g1["sum_of_file_bytes"]},
        "summary_against_camels": {"area_pct": {"median": float(np.median([per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids])), "largest": max((per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids), key=abs)},
                                   "elev_m": {"median": float(np.median([per[i]["elev_mean_m"]["diff_vs_camels_reviewer"] for i in ids])), "largest": max((per[i]["elev_mean_m"]["diff_vs_camels_reviewer"] for i in ids), key=abs)},
                                   "forest_points": {"median": float(np.median([per[i]["forest_share"]["points_vs_camels_reviewer"] for i in ids])), "largest": max((per[i]["forest_share"]["points_vs_camels_reviewer"] for i in ids), key=abs)}},
        "mismatches": MISMATCH,
        "gauge_summary": {"displacement_file_vs_camels_m": {"min": min(disp), "median": statistics.median(disp), "max": max(disp)}, "elev_at_gauge_differs_from_basin_gis_in": len(elev_diff),
                          "gauge_cell_more_than_12m_above_basin_minimum_at_file_points": {sh[i]: d for i, d in above12.items()}, "at_camels_points_largest_above_minimum_m": max(above12_5.values())},
        "geopackage": {"layers": g2["layers"], "sqlite": {k: g2["sqlite"][k] for k in ("application_id", "user_version", "integrity_check", "journal_mode", "has_rtree_for", "gpkg_contents")}},
        "geojson": g2["geojson"],
        "readme_claims": g5["claims"],
        "rebuild": ({k: g7.get(k) for k in ("returncode", "seconds", "n_equal_to_recorded", "not_equal_to_recorded", "rebuilt_record_check_table_equal", "rebuilt_record_verification_status", "readme_lines_differing_count")} if not g7.get("skipped") else "skipped"),
    },
    "checks": checks,
    "findings": fx,
    "counts": counts,
    "verdict": verdict,
    "verdict_text": verdict_text,
    "scripts": scripts,
    "scripts_how_to_run": "from the project root with the system Python 3.13 (geopandas, rasterio, shapely, pyproj, numpy, pandas, python-markdown), no GPU, about 40 s with the rebuild: run_checks.sh (g1, g2, g3, g4, g5, g7, g6 in this order; outputs to REVIEW_OUT, default <system temp>/gis_review); g6_assemble.py --write-into-project writes this record; REVIEW_SKIP_REBUILD=1 skips g7",
    "not_checked": not_checked,
    "dates": {"export_run": rec["written"], "review": "2026-10-30"},
}
jsave("review_gis_layers.json", record)
if write_in:
    with open(PROJ / "results" / "review_gis_layers.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=g_common._default)
    print("written results/review_gis_layers.json")

# ---- console summary ----------------------------------------------------------------------------------------------------------------------------------------------------------------
print("VERDICT:", verdict)
print("counts:", counts)
print("checks:", [(c["id"], c["ok"]) for c in checks])
print("export folder changed since g1:", changed, "| new in results/:", new_in_results, "| pycache in review folder:", integrity["pycache_in_review_folder"])
print("max deviations:", dev)
print("mismatch list (", len(MISMATCH), "):")
for m_ in MISMATCH[:60]:
    print("  ", m_)
print()
print("| basin | area km2 file / reviewer | mean elev m file / reviewer / CAMELS | forest % file / reviewer / CAMELS | gauge shift m | gauge elev file / basin_gis |")
for i in ids:
    p = per[i]
    print(f"| {p['short']} | {p['area_km2']['file']:.3f} / {p['area_km2']['reviewer_epsg5070']:.3f} | {p['elev_mean_m']['file']:.2f} / {p['elev_mean_m']['reviewer_cos_weighted']:.2f} / {p['elev_mean_m']['camels']:.2f} | "
          f"{p['forest_share']['file'] * 100:.2f} / {p['forest_share']['reviewer'] * 100:.2f} / {p['forest_share']['camels_frac_forest'] * 100:.2f} | {p['gauge']['displacement_file_vs_camels_m']:.1f} | {p['gauge']['elev_at_gauge_m_file']} / {int(p['gauge']['elev_at_gauge_m_basin_gis'])} |")
