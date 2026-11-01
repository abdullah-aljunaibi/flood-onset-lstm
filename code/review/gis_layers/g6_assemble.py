"""ROUND 2 assembler. Assembles the round-2 block of the review record from the outputs of g1 .. g5 and g7 (the checks run again on the export regenerated on 2026-10-30
11:30): the per-basin comparison, the checks with ok flags that are COMPUTED (no literal False), the status of every round-1 item, what is still open, the verdict. It also
re-reads results/gis_layers/ and proves that nothing in it changed since g1 (names, sizes, mtimes, sha256).

Round 1 (g6_round1_assemble.py, frozen) is never rewritten. With --write-into-project this script APPENDS the key "round_2" to results/review_gis_layers.json (replacing an
earlier round_2 of its own, nothing else): the file is read, round 1 is re-serialised with json.dumps(indent=1, ensure_ascii=False) and the file's own CRLF, which must give
back the original bytes (checked), and round_2 is added at the end. Afterwards the file without round_2 is rebuilt and its sha256 compared with the one recorded before.
Without the flag the block is only written to OUT/review_gis_layers_round2.json."""
import hashlib
import statistics
import time
from g_common import *
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

# ---- round 1, as it is in the project --------------------------------------------------------------------------------------------------------------------------------------
r1_bytes, r1_info = review_record_round1()
r1 = json.loads(r1_bytes.decode("utf-8"))
project_text = REVIEW_RECORD.read_bytes().decode("utf-8")
project_obj = json.loads(project_text)
assert "verdict" in r1 and r1["who"].startswith("independent AI reviewer agent"), "results/review_gis_layers.json does not look like round 1"
r1_per = r1["reproduced"]["per_basin"]
manifest_before = jout("manifest_before.json")

# ---- comparison engine: every value compared is counted, every difference beyond its tolerance is listed -------------------------------------------------------------------------
N = {"values_compared": 0}
MISMATCH = []


def cmp(basin, label, a, b, tol=0.0):
    N["values_compared"] += 1
    if isinstance(a, str) or isinstance(b, str) or isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        ok, d = (a == b), None
    else:
        d = float(a) - float(b)
        ok = abs(d) <= tol
    if not ok:
        MISMATCH.append({"basin": basin, "what": label, "file": a if not isinstance(a, float) else float(a), "reviewer": b if not isinstance(b, float) else float(b), "diff": d, "tol": tol})
    return d


per = {}
dev = {"area_km2": 0.0, "area_geodesic": 0.0, "elev_mean": 0.0, "forest_share": 0.0, "area_vs_basin_gis": 0.0, "elev_vs_basin_gis": 0.0, "forest_vs_basin_gis": 0.0,
       "gauge_distance_vs_basin_gis_km": 0.0}
CT = {r["gauge_id"]: r for r in rec["check_table"]}
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
    d_el = cmp(i, "elev_mean_m: file vs own cos(lat)-weighted mean of the exported raster, cells with centre inside (shapely.contains_xy)", b["elev_mean_attr"], b["elev_mean_weighted_mine"], 1e-6)
    d_ebg = cmp(i, "elev_mean_m vs results/basin_gis.json", b["elev_mean_attr"], b["elev_mean_basin_gis"], TOL["elev_m"])
    cmp(i, "elev_min_m", b["elev_min_attr"], b["elev_min_mine"]); cmp(i, "elev_max_m", b["elev_max_attr"], b["elev_max_mine"])
    cmp(i, "elev_min_m vs basin_gis", b["elev_min_attr"], b["elev_min_basin_gis"]); cmp(i, "elev_max_m vs basin_gis", b["elev_max_attr"], b["elev_max_basin_gis"])
    cmp(i, "n_elevation_cells vs own", b["n_cells_inside_attr"], b["n_cells_inside_mine"]); cmp(i, "n_elevation_cells vs basin_gis", b["n_cells_inside_attr"], b["n_cells_inside_basin_gis"])
    cmp(i, "elevation voids inside = 0", b["n_voids_inside_mine"], 0); cmp(i, "n_elevation_voids vs basin_gis", b["n_voids_inside_mine"], b["n_voids_basin_gis"])
    cmp(i, "valid cells written vs record n_elevation_cells_written", b["ring"]["n_valid_cells"], b["n_written_valid_cells_attr_json"])
    cmp(i, "elev_diff_camels_m vs own", b["elev_diff_camels_attr"], b["elev_diff_camels_mine_weighted"], 1e-6)
    cmp(i, "elevation source tiles: values equal (mismatches)", b["source_tiles"]["n_value_mismatches"], 0)
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
    # gauges: against the CAMELS table positions (which the README and the source attribute name) and against results/basin_gis.json
    cmp(i, "gauge lat (file) vs camels_topo.txt gauge_lat", a["gauge_lat_file"], a["gauge_lat_camels_topo"], 1e-9)
    cmp(i, "gauge lon (file) vs camels_topo.txt gauge_lon", a["gauge_lon_file"], a["gauge_lon_camels_topo"], 1e-9)
    cmp(i, "gauge lat/lon attributes vs the point geometry", abs(a["gauge_lat_attr"] - a["gauge_lat_file"]) + abs(a["gauge_lon_attr"] - a["gauge_lon_file"]), 0.0, 1e-9)
    cmp(i, "elev_at_gauge_m (file) vs results/basin_gis.json gauge.elev_m", b["gauge_cell"]["attr"], b["gauge_cell"]["basin_gis"], 0.0)
    d_gd = cmp(i, "distance_to_outline_km (file) vs results/basin_gis.json gauge.distance_to_outline_km", a["gauge_dist_edge_km_attr"], a["gauge_dist_edge_km_basin_gis"], 1e-6)
    cmp(i, "gauge inside its outline (file point)", a["gauge_inside_file_point"], True)
    cmp(i, "gauge inside its outline (attribute)", a["gauge_inside_attr"], True)
    cmp(i, "elev_at_gauge_m (file) vs raster cell at the file point", b["gauge_cell"]["attr"], b["gauge_cell"]["elev_at_exported_point"])
    # the check table of the record (now with the gauge columns) against the reviewer's own numbers
    cmp(i, "check_table gauge_elev_m vs reviewer's raster lookup", CT[i]["gauge_elev_m"], b["gauge_cell"]["elev_at_exported_point"])
    cmp(i, "check_table gauge_distance_to_outline_km vs file attribute", CT[i]["gauge_distance_to_outline_km"], a["gauge_dist_edge_km_attr"], 1e-9)
    cmp(i, "check_table agrees_with_basin_gis", CT[i]["agrees_with_basin_gis"], True)
    gj_ = rec["basins"][i]["gauge"]
    cmp(i, "results/gis_layers.json basins.gauge.lat vs the file's point", gj_["lat"], a["gauge_lat_file"], 1e-9)
    cmp(i, "results/gis_layers.json basins.gauge.lon vs the file's point", gj_["lon"], a["gauge_lon_file"], 1e-9)
    cmp(i, "results/gis_layers.json basins.gauge.elev_m vs the file attribute", gj_["elev_m"], b["gauge_cell"]["attr"])
    cmp(i, "results/gis_layers.json basins.gauge.distance_to_outline_km vs the file attribute", gj_["distance_to_outline_km"], a["gauge_dist_edge_km_attr"], 1e-9)
    cmp(i, "results/gis_layers.json basins.gauge.inside_outline", gj_["inside_outline"], True)
    dev["area_km2"] = max(dev["area_km2"], abs(d_area)); dev["area_geodesic"] = max(dev["area_geodesic"], abs(d_geod)); dev["elev_mean"] = max(dev["elev_mean"], abs(d_el))
    dev["forest_share"] = max(dev["forest_share"], abs(d_fo)); dev["area_vs_basin_gis"] = max(dev["area_vs_basin_gis"], abs(d_abg))
    dev["elev_vs_basin_gis"] = max(dev["elev_vs_basin_gis"], abs(d_ebg)); dev["forest_vs_basin_gis"] = max(dev["forest_vs_basin_gis"], abs(d_fbg))
    dev["gauge_distance_vs_basin_gis_km"] = max(dev["gauge_distance_vs_basin_gis_km"], abs(d_gd))
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
                  "file_equals_camels_topo": a["gauge_file_equals_camels_topo"],
                  "displacement_file_vs_camels_m": a["gauge_displacement_file_vs_camels_m"], "inside_outline_file_point": a["gauge_inside_file_point"], "inside_outline_camels_point": a["gauge_inside_camels_point"],
                  "distance_to_edge_km_file_attr": a["gauge_dist_edge_km_attr"], "distance_to_edge_km_reviewer_file_point": a["gauge_dist_edge_km_file_point_mine"],
                  "distance_to_edge_km_reviewer_camels_point": a["gauge_dist_edge_km_camels_point_mine"], "distance_basin_gis": a["gauge_dist_edge_km_basin_gis"],
                  "elev_at_gauge_m_file": b["gauge_cell"]["attr"], "elev_at_gauge_m_reviewer_file_point": b["gauge_cell"]["elev_at_exported_point"],
                  "elev_at_gauge_m_reviewer_camels_point": b["gauge_cell"]["elev_at_camels_5dec_point"], "elev_at_gauge_m_basin_gis": b["gauge_cell"]["basin_gis"],
                  "lowest_cell_of_basin_m": b["elev_min_mine"], "source_attribute": a["gauge_source_attr"]},
    }

# ---- the summary block of the record against the reviewer's numbers ---------------------------------------------------------------------------------------------------------
for key, arr in (("area_vs_camels_gages2_pct", [per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids]), ("elev_vs_camels_m", [per[i]["elev_mean_m"]["diff_vs_camels_reviewer"] for i in ids]),
                 ("forest_vs_camels_points", [per[i]["forest_share"]["points_vs_camels_reviewer"] for i in ids])):
    arr = np.array(arr)
    cmp("all", f"summary.{key}.median", rec["summary"][key]["median"], float(np.median(arr)), 1e-9)
    cmp("all", f"summary.{key}.mean_absolute", rec["summary"][key]["mean_absolute"], float(np.mean(np.abs(arr))), 1e-9)
    cmp("all", f"summary.{key}.largest", rec["summary"][key]["largest"], float(arr[np.argmax(np.abs(arr))]), 1e-9)

# ---- what changed against round 1 -------------------------------------------------------------------------------------------------------------------------------------------


def leaves(o, path=()):
    if isinstance(o, dict):
        for k_, v_ in o.items():
            yield from leaves(v_, path + (k_,))
    elif isinstance(o, list):
        for k_, v_ in enumerate(o):
            yield from leaves(v_, path + (k_,))
    else:
        yield path, o


reg_diffs, n_reg = [], 0
for i in ids:
    la = dict(leaves({k_: v_ for k_, v_ in r1_per[i].items() if k_ != "gauge"}))
    lb = dict(leaves({k_: v_ for k_, v_ in per[i].items() if k_ != "gauge"}))
    for p_ in sorted(set(la) | set(lb), key=str):
        n_reg += 1
        va, vb = la.get(p_, "<missing>"), lb.get(p_, "<missing>")
        same = va == vb or (isinstance(va, float) and isinstance(vb, float) and abs(va - vb) <= 1e-12 * max(1.0, abs(va)))
        if not same:
            reg_diffs.append({"basin": i, "field": ".".join(map(str, p_)), "round_1": va, "round_2": vb})
since = g1["since_round_1"]
EXPECTED_CHANGED = ["README.md", "basins_12.gpkg", "gauges_12.geojson"]
gauge_fix = {}
for i in ids:
    o, n_ = r1_per[i]["gauge"], per[i]["gauge"]
    gauge_fix[i] = {"short": per[i]["short"], "lat_lon_round_1": o["file_lat_lon"], "lat_lon_now": n_["file_lat_lon"], "lat_lon_camels_topo": n_["camels_topo_lat_lon"],
                    "displacement_vs_camels_m": {"round_1": o["displacement_file_vs_camels_m"], "now": n_["displacement_file_vs_camels_m"]},
                    "elev_at_gauge_m": {"round_1": o["elev_at_gauge_m_file"], "now": n_["elev_at_gauge_m_file"], "basin_gis": n_["elev_at_gauge_m_basin_gis"], "reviewer_lookup": n_["elev_at_gauge_m_reviewer_file_point"]},
                    "distance_to_outline_km": {"round_1": o["distance_to_edge_km_file_attr"], "now": n_["distance_to_edge_km_file_attr"], "basin_gis": n_["distance_basin_gis"]},
                    "inside_outline": n_["inside_outline_file_point"]}
disp_now = [per[i]["gauge"]["displacement_file_vs_camels_m"] for i in ids]
disp_r1 = [r1_per[i]["gauge"]["displacement_file_vs_camels_m"] for i in ids]
above12_now = {per[i]["short"]: per[i]["gauge"]["elev_at_gauge_m_file"] - per[i]["gauge"]["lowest_cell_of_basin_m"] for i in ids}
max_edge_gap_m = max((per[i]["gauge"]["distance_to_edge_km_file_attr"] - per[i]["gauge"]["distance_to_edge_km_reviewer_file_point"]) * 1000 for i in ids)
sh = {i: per[i]["short"] for i in ids}


# ---- helpers for the checks -----------------------------------------------------------------------------------------------------------------------------------------------------
def claim(prefix):
    hits = [c_ for c_ in g5["claims"] if c_["claim"].startswith(prefix)]
    assert hits, prefix
    return hits[0]


def claim_ok(*prefixes):
    return all(c_["ok"] for p_ in prefixes for c_ in g5["claims"] if c_["claim"].startswith(p_)) and all(any(c_["claim"].startswith(p_) for c_ in g5["claims"]) for p_ in prefixes)


vec = g2["checks"]
gauge_mismatch = [m for m in MISMATCH if str(m["what"]).startswith(("gauge", "elev_at_gauge", "distance_to_outline", "check_table gauge"))]
checks = [
    {"id": "1", "what": "every file of results/gis_layers.json 'files' exists with the recorded size and sha256; no other file in the folder; total bytes; script hashes (code/27 now b205...); the 35 inputs read; the record is strict JSON",
     "ok": bool(g1["all_exist_with_recorded_bytes_and_sha256"] and not g1["extra_files_in_folder"] and not g1["missing_files"] and g1["total_bytes_equal"] and all(x["equal"] for x in g1["code_hashes"].values())
                and g1["inputs_read_equal"]["n"] == g1["inputs_read_equal"]["n_equal"] and g1["record_is_strict_json"] is True),
     "evidence": {"files": g1["n_files_recorded"], "on_disk": g1["n_files_on_disk"], "sum_bytes": g1["sum_of_file_bytes"], "scripts_equal": {k_: x["equal"] for k_, x in g1["code_hashes"].items()},
                  "inputs_read_equal": g1["inputs_read_equal"], "record_is_strict_json": g1["record_is_strict_json"]}},
    {"id": "2a", "what": "GeoPackage: layers outlines + gauges, 12 + 12 features, EPSG:4269, ids/names/landscapes in the data_audit order, outlines valid, sqlite integrity ok",
     "ok": bool(vec["layer_names_are_outlines_gauges"] and vec["n_outlines"] == 12 and vec["n_gauges"] == 12 and vec["crs_outlines"] == "EPSG:4269" and vec["crs_gauges"] == "EPSG:4269" and vec["outline_ids_in_audit_order"]
                and vec["gauge_ids_in_audit_order"] and vec["outline_names_equal_audit"] and vec["landscapes_equal_audit"] and all(vec["outline_validity"].values()) and g2["sqlite"]["integrity_check"] == "ok"),
     "evidence": {"layers": g2["layers"], "application_id": g2["sqlite"]["application_id"], "user_version": g2["sqlite"]["user_version"], "rtree": g2["sqlite"]["has_rtree_for"]}},
    {"id": "2b", "what": "planar area in EPSG:5070 recomputed (pyproj + shapely) equals the attribute area_km2, results/basin_gis.json and the shapefile AREA field; geodesic area equal; outlines exactly the source geometry (10) or its buffer(0) repair (2)",
     "ok": bool(dev["area_km2"] <= 1e-6 and dev["area_vs_basin_gis"] <= TOL["area_km2"] and dev["area_geodesic"] <= 1e-6),
     "evidence": {"max_abs_file_minus_reviewer_km2": dev["area_km2"], "max_abs_file_minus_basin_gis_km2": dev["area_vs_basin_gis"], "max_abs_geodesic_diff_km2": dev["area_geodesic"],
                  "max_abs_reviewer_minus_shapefile_AREA_km2": max(abs(per[i]["area_km2"]["reviewer_epsg5070"] - per[i]["area_km2"]["shapefile_AREA_km2"]) for i in ids)}},
    {"id": "2c", "what": "area against CAMELS area_gages2: attribute area_diff_gages2_pct equals the reviewer's; range of the difference", "ok": bool(all(abs(g2["per_basin"][i]["area_vs_gages2_pct_attr"] - g2["per_basin"][i]["area_vs_gages2_pct_mine"]) <= 1e-6 for i in ids)),
     "evidence": {"min_pct": min(per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids), "max_pct": max(per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids),
                  "median_pct": statistics.median(per[i]["area_km2"]["pct_vs_camels_reviewer"] for i in ids)}},
    {"id": "2d", "what": "every gauge point lies inside its outline (exported points and the 5-decimal CAMELS positions)", "ok": bool(all(per[i]["gauge"]["inside_outline_file_point"] and per[i]["gauge"]["inside_outline_camels_point"] for i in ids)),
     "evidence": {"max_distance_to_edge_km_file_points": max(per[i]["gauge"]["distance_to_edge_km_reviewer_file_point"] for i in ids), "max_distance_to_edge_km_camels_points": max(per[i]["gauge"]["distance_to_edge_km_reviewer_camels_point"] for i in ids)}},
    {"id": "2e", "what": "gauge points are at the CAMELS gauge_lat/gauge_lon of camels_topo.txt (the README and the layer's source attribute say so, not rounded); elev_at_gauge_m and distance_to_outline_km equal results/basin_gis.json; the check table's gauge columns equal the reviewer's values",
     "ok": bool(not gauge_mismatch and claim_ok("README Gauges bullet", "README: 'The elevation cell and the distance equal", "results/gis_layers.json check_table gauge columns")),
     "evidence": {"gauge_values_compared_that_differ": len(gauge_mismatch), "max_displacement_from_camels_topo_m": max(disp_now), "round_1_displacement_m": {"min": min(disp_r1), "median": statistics.median(disp_r1), "max": max(disp_r1)},
                  "elev_at_gauge_equal_basin_gis_in": sum(1 for i in ids if per[i]["gauge"]["elev_at_gauge_m_file"] == per[i]["gauge"]["elev_at_gauge_m_basin_gis"]), "max_abs_distance_difference_to_basin_gis_km": dev["gauge_distance_vs_basin_gis_km"],
                  "gauge_cell_above_lowest_cell_of_basin_m_max": max(above12_now.values()), "source_attribute": sorted({per[i]["gauge"]["source_attribute"] for i in ids})}},
    {"id": "3a", "what": "12 elevation GeoTIFFs: EPSG:4326, int16, nodata -32768, 1.000000 arc-second, cell centres on whole arc-seconds, bounds contain the outline (>= 15.5 arc-seconds all round), LZW predictor 2, tiled 256",
     "ok": bool(all(per[i]["elevation_raster"]["crs_epsg"] == 4326 and per[i]["elevation_raster"]["dtype"] == "int16" and per[i]["elevation_raster"]["nodata"] == DEM_NODATA and abs(per[i]["elevation_raster"]["res_arcsec"][0] - 1) < 1e-9
                    and abs(per[i]["elevation_raster"]["res_arcsec"][1] - 1) < 1e-9 and per[i]["elevation_raster"]["cell_centres_on_whole_arcseconds"] and per[i]["elevation_raster"]["bounds_contain_outline"]
                    and per[i]["elevation_raster"]["margin_arcsec_min"] > 15.0 and per[i]["elevation_raster"]["compression_predictor_tiled"] == ["LZW", "2", True, [256, 256]] for i in ids)),
     "evidence": {"min_margin_arcsec": min(per[i]["elevation_raster"]["margin_arcsec_min"] for i in ids)}},
    {"id": "3b", "what": "mean elevation of the cells whose centre lies inside the outline (cos-latitude weighted) recomputed with shapely.contains_xy equals the attribute elev_mean_m and results/basin_gis.json; cell counts, min, max, voids equal; every exported sample equals the source tile sample; tile lists equal the record's",
     "ok": bool(dev["elev_mean"] <= 1e-6 and dev["elev_vs_basin_gis"] <= TOL["elev_m"] and all(per[i]["elevation_cells"]["source_tile_value_mismatches"] == 0 for i in ids) and g3["totals"]["voids_inside"] == 0
                and all(g3["per_basin"][i]["source_tiles"]["tiles_equal_record"] for i in ids)),
     "evidence": {"max_abs_file_minus_reviewer_m": dev["elev_mean"], "max_abs_file_minus_basin_gis_m": dev["elev_vs_basin_gis"], "cells_inside_total": g3["totals"]["cells_inside_mine"], "valid_cells_written_total": g3["totals"]["cells_written_valid"],
                  "cells_compared_with_source_tiles": sum(per[i]["elevation_cells"]["cells_compared_with_source_tiles"] for i in ids),
                  "unweighted_mean_differs_from_weighted_by_up_to_m": max(abs(per[i]["elev_mean_m"]["reviewer_unweighted"] - per[i]["elev_mean_m"]["reviewer_cos_weighted"]) for i in ids)}},
    {"id": "3c", "what": "elevation difference against CAMELS elev_mean: attribute equals the reviewer's; exactly three basins beyond 25 m", "ok": bool(sorted(i for i in ids if per[i]["elev_mean_m"]["more_than_25_m_from_camels"]) == ["02137727", "03439000", "03463300"]),
     "evidence": {sh[i]: {"file": per[i]["elev_mean_m"]["file"], "camels": per[i]["elev_mean_m"]["camels"], "diff_m": per[i]["elev_mean_m"]["diff_vs_camels_reviewer"], "shapefile_elev_mean_field": per[i]["elev_mean_m"]["shapefile_elev_mean_field"]}
                  for i in ("02137727", "03439000", "03463300")}},
    {"id": "4a", "what": "12 land-cover GeoTIFFs: EPSG:5070, uint8, nodata 250, 30 m, colour table equals the CSV, LZW tiled, bounds contain the outline (projected), cells outside the outline nodata and none inside",
     "ok": bool(all(per[i]["landcover_raster"]["crs_epsg"] == 5070 and per[i]["landcover_raster"]["dtype"] == "uint8" and per[i]["landcover_raster"]["nodata"] == NLCD_NODATA and all(abs(r_ - 30) < 1e-6 for r_ in per[i]["landcover_raster"]["res_m"])
                    and per[i]["landcover_raster"]["bounds_contain_outline"] and per[i]["landcover_raster"]["colour_table_equals_csv"] and per[i]["land_cover_cells"]["inside_but_nodata"] == 0
                    and per[i]["land_cover_cells"]["outside_but_valid"] == 0 for i in ids)),
     "evidence": {"min_margin_m": min(per[i]["landcover_raster"]["margin_m_min"] for i in ids), "pixel_size_m_note": "30.00000001 to 30.00000012 m in the file (the WCS grid), within 1e-6 of 30"}},
    {"id": "4b", "what": "forest share (NLCD 41+42+43) recomputed inside the projected outline equals the attribute forest_share and results/basin_gis.json; cell counts and all 15 class shares and 7 group shares equal; every cell equals the cached NLCD cell",
     "ok": bool(dev["forest_share"] <= 1e-9 and dev["forest_vs_basin_gis"] <= TOL["share"] and all(per[i]["land_cover_cells"]["cache_value_mismatches"] == 0 for i in ids) and g4["total_cells_inside_mine"] == 7912882),
     "evidence": {"max_abs_file_minus_reviewer": dev["forest_share"], "max_abs_file_minus_basin_gis": dev["forest_vs_basin_gis"], "cells_inside_total": g4["total_cells_inside_mine"],
                  "cells_compared_with_cache": sum(per[i]["land_cover_cells"]["cells_compared_with_cached_nlcd"] for i in ids)}},
    {"id": "5a", "what": "README check tables (area, mean elevation, dominant land cover + forest) equal results/gis_layers.json check_table and the reviewer's recomputation, cell by cell within the printed rounding", "ok": claim_ok("README check tables equal"),
     "evidence": {"numeric_cells_compared_twice": g5["tables"]["numeric_cells_compared"], "text_cells_compared_twice": g5["tables"]["text_cells_compared"], "mismatches": len(g5["tables"]["mismatches"]["vs_check_table"]) + len(g5["tables"]["mismatches"]["vs_my_recomputation"])}},
    {"id": "5b", "what": "README summary sentence, '12 of 12 agree', the known-difference paragraph (the results/basin_gis.json note word for word; +47/+100/+163 m) and the note's own figures",
     "ok": bool(claim_ok("README summary sentence", "README says '12 of 12", "12 of 12 basins agree", "README quotes the 'Elevation:' note", "results/gis_layers.json known_note", "exactly three basins", "note:")), "evidence": {}},
    {"id": "6a", "what": "licences: every licence in the README source text, in results/gis_layers.json licences and in every files[].licence is UNVERIFIED; neither data_flow.json nor basin_gis.json holds a licence field; no licence text in the GPKG, GeoJSON, CSV or raster tags; the README asserts none of its own; 'derived from sources whose terms are not recorded here' replaces 'inherit'",
     "ok": bool(claim_ok("README source text", "results/data_flow.json and results/basin_gis.json hold no licence", "README asserts no licence", "no raster tag", "results/gis_layers.json: every files", "README: the 'login needed' text",
                         "README and results/gis_layers.json licences.note")), "evidence": {"licence_sentence": g5["licence_wording_sentence"], "inherit_sentences_left": g5["inherit_sentence"]}},
    {"id": "6b", "what": "licences as a reader sees them: every README table row has the header's number of cells (5 tables, 67 rows, equal to the script's own count); the sources table rendered (python-markdown tables, same rule as GitHub) shows UNVERIFIED in the licence column of all three rows; no angle-bracket text outside backticks",
     "ok": bool(claim_ok("every README table row has", "README verification line 'readme_tables_cells_per_row", "README rendered", "no <angle-bracket>")),
     "evidence": {"rendered": claim("README rendered")["detail"], "tables_and_rows": claim("README verification line 'readme_tables_cells_per_row")["detail"]}},
    {"id": "6c", "what": "the citation sentence says only what data/camels_us/readme.txt (attributes -> Addor 2017, time series -> Newman 2015) and results/data_flow.json (Newman 2015 named for the outlines) say, and that no request to cite is recorded for the outlines; it is in the README and in results/gis_layers.json (licences and files[])",
     "ok": claim_ok("README/JSON citation sentence"), "evidence": claim("README/JSON citation sentence")["detail"]},
    {"id": "7", "what": "the disclosure line 'Claude Code wrote and ran this script under the team lead's direction; the team's own check of these layers is not done' is in the README (line 5, second half in bold) and, with a date inserted, in results/gis_layers.json disclosure",
     "ok": bool(claim_ok("README carries the disclosure", "results/gis_layers.json carries the disclosure")),
     "evidence": {"readme_line": claim("README carries the disclosure line")["detail"]["line_in_readme"], "json": rec["disclosure"]}},
    {"id": "8", "what": "the independent-review line (README line 7, results/gis_layers.json disclosure_independent_review) and the block independent_review quote the review record faithfully as round 1 stood when the script read it (sha256 of the record without later rounds, who, date, verdict, export run reviewed, must/should-fix counts, human_review)",
     "ok": bool(claim_ok("README carries the independent-review line", "results/gis_layers.json independent_review quotes") and g1["review_record_block"]["equal"]),
     "evidence": {"block": rec["independent_review"], "line": rec["disclosure_independent_review"], "sha256_equals_record_without_later_rounds": g1["review_record_block"]["equal"]}},
    {"id": "x1", "what": "extra: a clean rebuild of code/27_gis_layers.py in a scratch project root (copies of the inputs and of round 1 of this record) reproduces the sha256 of the 27 files the README calls deterministic (24 GeoTIFFs, 2 GeoJSON incl. the new gauges_12.geojson, the CSV), the check_table, the licences and the independent_review block; only the GeoPackage (write time) and the README (time stamp, gpkg hash) differ",
     "ok": (bool(g7.get("returncode") == 0 and g7.get("n_equal_to_recorded") == 27 and sorted(g7.get("not_equal_to_recorded", [])) == ["README.md", "basins_12.gpkg"] and g7.get("rebuilt_record_check_table_equal")
                 and g7.get("rebuilt_independent_review_equal") and g7.get("rebuilt_licences_equal") and g7.get("readme_lines_differing_count") == 2) if not g7.get("skipped") else None),
     "evidence": ({k_: g7.get(k_) for k_ in ("returncode", "seconds", "n_equal_to_recorded", "not_equal_to_recorded", "rebuilt_record_check_table_equal", "rebuilt_independent_review_equal", "rebuilt_licences_equal", "readme_lines_differing_count")}
                  if not g7.get("skipped") else "skipped")},
    {"id": "x3", "what": "extra: against round 1 - 26 of the 29 files are byte-identical to the ones round 1 reviewed (24 GeoTIFFs, basins_12.geojson, the CSV); the three that changed are README.md, basins_12.gpkg and gauges_12.geojson; every non-gauge value of round 1's per-basin table (area, elevation, forest, cells, dominant class, raster properties) is unchanged",
     "ok": bool(len(since["same_as_round_1"]) == 26 and since["changed_since_round_1"] == EXPECTED_CHANGED and not reg_diffs),
     "evidence": {"same_as_round_1": len(since["same_as_round_1"]), "changed_since_round_1": since["changed_since_round_1"], "round_1_values_compared": n_reg, "round_1_values_that_differ": len(reg_diffs)}},
]

# ---- the folder is unchanged ------------------------------------------------------------------------------------------------------------------------------------------------------
now = {n_.name: {"bytes": n_.stat().st_size, "mtime_ns": n_.stat().st_mtime_ns, "sha256": sha256_file(n_)} for n_ in sorted(GIS_DIR.iterdir())}
changed = sorted(n_ for n_ in set(manifest_before["gis_layers"]) | set(now) if manifest_before["gis_layers"].get(n_) != now.get(n_))
res_names_now = sorted(p_.name for p_ in (PROJ / "results").iterdir())
new_in_results = sorted(set(res_names_now) - set(manifest_before["results_names"]))
integrity = {"export_folder_files_changed_since_g1": changed, "export_folder_file_count": len(now), "new_names_in_results_since_g1": new_in_results, "pycache_in_review_folder": (HERE / "__pycache__").exists(),
             "gis_layers_json_sha256": sha256_file(PROJ / RECORD), "readme_sha256": sha256_file(GIS_DIR / "README.md"), "code_27_sha256": sha256_file(PROJ / SCRIPT)}
integrity["side_files_in_export_folder"] = [n_ for n_ in now if re.search(r"(-wal|-shm|-journal|\.aux\.xml)$", n_)]
checks.append({"id": "x2", "what": "extra: round 2 changed nothing in results/gis_layers/ (names, sizes, mtimes, sha256 equal to the manifest taken before the other checks) and left no side file there",
               "ok": bool(not changed and not integrity["side_files_in_export_folder"]), "evidence": integrity})

# ---- round 1 findings: status now ----------------------------------------------------------------------------------------------------------------------------------------------
read = lambda rel: (PROJ / rel).read_text(encoding="utf-8", errors="replace")
numbers_py, short_slides_py, short_report_py, run_all_sh = read("code/final_report/numbers.py"), read("code/build_slides_short.py"), read("code/build_report_short.py"), read("run_all.sh")
numbers_excludes = 'k != "review_gis_layers"' in numbers_py
numbers_reads_top_verdict = 'RVG.get("verdict")' in numbers_py
short_slides_sums_all = 'k.startswith("review_")' in short_slides_py and "review_gis_layers" not in short_slides_py
short_report_sums_all = 'k.startswith("review_")' in short_report_py and "review_gis_layers" not in short_report_py
short_in_run_all = "build_slides_short.py" in run_all_sh
paths_relative = all(g3["per_basin"][i]["elevation_file_attr"].startswith("results/gis_layers/") for i in ids)
readme_text = (GIS_DIR / "README.md").read_text(encoding="utf-8")
signed_zero_cells = len(re.findall(r"\| [-+]0\.0{4,6} \|", readme_text))
dbf_line_in_readme = "661.7" in readme_text
integ_open = [txt_ for txt_, flag_ in (("numbers.py reads the top-level verdict of the record", numbers_reads_top_verdict), ("build_slides_short.py (in run_all.sh) sums every review record", short_slides_sums_all),
                                       ("build_report_short.py sums every review record", short_report_sums_all)) if flag_]
r1_f = r1["findings"]
assert [f_["severity"] for f_ in r1_f] == ["must-fix", "must-fix", "should-fix"] + ["note"] * 9, "round 1 has other findings than expected"
items = [
    {"item": "round 1 must-fix 1: gauge points rounded to 0.001 degree (not the camels_topo.txt positions), elev_at_gauge_m differing from basin_gis.json in 9 of 12, check table not covering gauges",
     "status": "closed" if next(c_ for c_ in checks if c_["id"] == "2e")["ok"] else "open",
     "evidence": f"check 2e: points equal camels_topo.txt (largest displacement {max(disp_now):.1f} m; round 1: {min(disp_r1):.1f}-{max(disp_r1):.1f} m), elev_at_gauge_m equal basin_gis.json and the reviewer's raster lookup in 12 of 12, distance equal to {dev['gauge_distance_vs_basin_gis_km']:.0e} km, the gauge cell at most {max(above12_now.values()):.0f} m above the lowest cell of its basin (SF Holston {above12_now['SF Holston, VA']:.0f} m), check_table now carries the gauge columns and agrees_with_basin_gis includes them"},
    {"item": "round 1 must-fix 2: licence mark of the elevation row lost in the rendered README (literal pipes in the URL pattern), angle-bracket words read as HTML tags",
     "status": "closed" if next(c_ for c_ in checks if c_["id"] == "6b")["ok"] else "open",
     "evidence": "check 6b: every row has its header's cell count (5 tables, 67 rows, the script's own check says the same), the rendered licence column shows UNVERIFIED in all three rows, no angle-bracket text outside backticks"},
    {"item": "round 1 should-fix: citation sentence put 'and outlines' into data/camels_us/readme.txt",
     "status": "closed" if next(c_ for c_ in checks if c_["id"] == "6c")["ok"] else "open", "evidence": "check 6c: the new sentence is my replacement text; readme.txt asks Addor 2017 for the attributes and Newman 2015 for the time series, data_flow.json names Newman 2015 for the outlines, and no request to cite is in readme.txt or readme_shapefiles.txt"},
    {"item": "round 1 note: integration with the document builders (results/review_*.json is globbed)",
     "status": ("closed" if numbers_excludes and not integ_open else "partly closed; still open: " + "; ".join(integ_open) if numbers_excludes else "open: numbers.py does not leave the record out of the experiment counts"),
     "evidence": {"numbers_py_excludes_review_gis_layers_from_the_experiment_counts": numbers_excludes, "numbers_py_reads_the_top_level_verdict_of_the_record": numbers_reads_top_verdict,
                  "build_slides_short_py_sums_every_review_record": short_slides_sums_all, "build_report_short_py_sums_every_review_record": short_report_sums_all, "build_slides_short_py_is_in_run_all_sh": short_in_run_all}},
    {"item": "round 1 note: disclosure should say that an AI agent reviewed the layers (HANDOFF section 7)", "status": "done, with two small caveats (see round 2 should-fix 1 and 2)",
     "evidence": {"readme_line": rec["disclosure_independent_review"], "json_blocks": ["disclosure_independent_review", "independent_review"], "team_check_clause_kept": "the team's own check of these layers is not done" in rec["disclosure_independent_review"]}},
    {"item": "round 1 note: distance_to_outline_km is taken at the boundary point nearest in degree space", "status": "unchanged (optional)", "evidence": f"the attribute equals the nearest-in-degrees distance exactly (difference 0); it is up to {max_edge_gap_m:.1f} m longer than the true minimum at the new points (round 1: 12.7 m); the README's 'at most 0.63 km' holds (largest {max(per[i]['gauge']['distance_to_edge_km_file_attr'] for i in ids):.3f} km)"},
    {"item": "round 1 note: elevation_file / landcover_file attributes hold project-relative paths; '15 arc-seconds (about 462 m)' is north-south only",
     "status": "half done: the buffer sentence is fixed ('about 460 m north-south, 370-410 m east-west', check g5), the path attributes are unchanged (optional)", "evidence": {"elevation_file_attribute_still_starts_with_results/gis_layers/": paths_relative}},
    {"item": "round 1 note: 'the exported files inherit whatever terms their sources carry'", "status": "closed", "evidence": g5["licence_wording_sentence"]},
    {"item": "round 1 note: GeoJSON 'crs' member and clockwise exterior rings", "status": "unchanged (optional)", "evidence": {"crs_member": g2["geojson"]["basins_12.geojson"]["crs_member"], "exterior_rings_ccw_share": g2["geojson"]["basins_12.geojson"]["exterior_rings_ccw_share"]}},
    {"item": "round 1 note: signed zeros in the README 'diff' columns", "status": "unchanged (cosmetic)", "evidence": {"cells_printing_a_signed_zero": signed_zero_cells}},
    {"item": "round 1 note: the shapefile's own elev_mean field supports the statement on the three mountain basins (the README may add the line)", "status": "not added (optional)", "evidence": {"README_contains_661.7": dbf_line_in_readme}},
    {"item": "round 1 note: 15-arc-second ring discretisation", "status": "no action needed", "evidence": {"nodata_cells_inside_reviewer_buffer": sum(per[i]["elevation_raster"]["nodata_cells_inside_reviewer_15arcsec_buffer"] for i in ids)}},
]
assert len(items) == len(r1_f)
n_closed = sum(1 for x in items if x["status"].startswith(("closed", "done")))                        # closed, or done as asked

# ---- round 2 findings ----------------------------------------------------------------------------------------------------------------------------------------------------------
disc_claim = claim("results/gis_layers.json disclosure names the day")
disc_date = disc_claim["detail"]["disclosure_date"]
fx = []
if not disc_claim["ok"]:
    fx.append({"severity": "should-fix", "title": "the JSON disclosure dates the run 2026-10-29, the files are from the run of 2026-10-30", "where": "results/gis_layers.json disclosure (code/27_gis_layers.py line 574, a literal)",
               "text": (f"The disclosure still reads \"Claude Code wrote and ran this script under the team lead's direction on {disc_date}; the team's own check of these layers is not done.\" but results/gis_layers.json 'written' is {rec['written']}: the files were produced by a revised script run on 2026-10-30 "
                        "(the docstring says 'Written 2026-10-29 ... Revised 2026-10-30 after the independent review'). The README line (no date) is right; the JSON field now reads as if the files came from the run of 2026-10-29. "
                        "Fix: make the date a computed text, for example \"Claude Code wrote and ran this script under the team lead's direction (written 2026-10-29, revised and rerun 2026-10-30, this run \" + NOW + \"); the team's own check of these layers is not done.\" "
                        "and keep the second clause word for word.")})
fx.append({"severity": "should-fix", "title": "independent_review (gis_layers.json) and numbers.py read round 1's top-level fields, which round 2 supersedes",
           "where": "results/gis_layers.json independent_review and disclosure_independent_review (code/27_gis_layers.py lines 143-148); code/final_report/numbers.py lines 466-471 (gis_layers_review_verdict, gis_layers_review_date)",
           "text": ("Both readers take their fields from the TOP LEVEL of results/review_gis_layers.json, which is round 1 and stays so: verdict 'discrepancies', dates.export_run 2026-10-29 18:25, counts.must_fix 2 and should_fix 1. Once this round 2 is appended, "
                    "results/gis_layers.json (written 2026-10-30 11:30, after the fixes) carries a block that rates the old export, and the documents' gis_layers_review_verdict reads 'discrepancies' for the corrected one; its sha256 is that of round 1 and will not equal the file's. "
                    "The README line itself ('recomputed these layers on 2026-10-30 (results/review_gis_layers.json)') is true only now that round 2 exists. "
                    "Fix: read the LAST round block, for example last = project_record[sorted(k for k in rv if re.fullmatch(r'round_\\d+', k))[-1]] if any round_N exists, else the top level, and take verdict, dates.review, dates.export_run, counts.must_fix, counts.should_fix, human_review and who from it - round_2 has all of them under the same names - "
                    "and store the hash of round 1 under another name ('sha256_of_round_1_when_read') or drop it, since a record that holds its reviewer's hash and a reviewer that holds the file's hash can never both be final. "
                    "After that change one rerun is enough: it rewrites only README.md, basins_12.gpkg and results/gis_layers.json (time stamps); the 27 deterministic files keep their hashes, and run_checks.sh of this review (about a minute) confirms it - the verdict of this round does not depend on the three timestamped files.")})
if short_slides_sums_all or short_report_sums_all:
    fx.append({"severity": "note", "title": "two older builders (build_slides_short.py in run_all.sh, build_report_short.py) still sum the record's round-1 counts (failed 3)",
               "where": "code/build_slides_short.py lines 154-160 (in run_all.sh), code/build_report_short.py lines 511-522 (reviews()); round 1's counts block",
               "text": (f"These two older builders still sum every results/review_*.json: records, rounds, checks_listed, failed, numerical_discrepancies. With this record they count two more rounds and round 2's {len(checks)} checks, and round 1's counts.failed = {r1['counts']['failed']} "
                        "(kept as it is) turns the slide phrase ', none failed' off and adds that number to the 'failed' of the short report, although nothing fails any more. "
                        f"numbers.py {'already leaves the record out of the experiment counts (verified in the file)' if numbers_excludes else 'does not leave the record out of the experiment counts'}. "
                        "Fix (the coordinator's choice): skip 'review_gis_layers' in both loops as numbers.py does, or leave it and let the slide say what round 1 found. Nothing to change in the record.")})
fx.append({"severity": "note", "title": "the optional round-1 notes left as they are (distance method, path attributes, GeoJSON crs/orientation, signed zeros, DBF line)", "where": "the optional items of round 1 left as they are",
           "text": (f"Unchanged and harmless: distance_to_outline_km is the distance to the point nearest in degree space (up to {max_edge_gap_m:.0f} m longer than the true minimum at the new points); the elevation_file / landcover_file attributes hold project-relative paths; "
                    f"the GeoJSON files carry the deprecated 'crs' member and clockwise exterior rings; the README 'diff' columns print {signed_zero_cells} signed zeros; the README does not add the shapefile's own elev_mean values (661.7, 976.2, 1213.3 m) to the mountain-basin paragraph. No action needed.")})

# ---- counts, verdict ---------------------------------------------------------------------------------------------------------------------------------------------------------------
sev = {k_: sum(1 for f_ in fx if f_["severity"] == k_) for k_ in ("must-fix", "should-fix", "note")}
n_checks = len(checks)
n_failed = sum(1 for c_ in checks if c_["ok"] is False)
n_pass = sum(1 for c_ in checks if c_["ok"] is True)
counts = {"checks_listed": n_checks, "checks_passed": n_pass, "findings_not_pass_fail": len(fx), "failed": n_failed, "must_fix": sev["must-fix"], "should_fix": sev["should-fix"], "notes": sev["note"],
          "numerical_discrepancies": 0, "numerical_mismatches": len(MISMATCH),
          "values_compared": N["values_compared"] + (g5["tables"]["numeric_cells_compared"] + g5["tables"]["text_cells_compared"]) * 2 + n_reg,
          "values_compared_scalars_file_vs_recomputation": N["values_compared"], "readme_table_cells_compared_against_check_table_and_recomputation": (g5["tables"]["numeric_cells_compared"] + g5["tables"]["text_cells_compared"]) * 2,
          "round_1_values_compared_again": n_reg, "round_1_values_that_differ": len(reg_diffs),
          "raster_cells_compared_with_their_sources_not_in_values_compared": sum(per[i]["elevation_cells"]["cells_compared_with_source_tiles"] for i in ids) + sum(per[i]["land_cover_cells"]["cells_compared_with_cached_nlcd"] for i in ids),
          "round_1_items": len(items), "round_1_items_closed": n_closed, "basins_checked": 12, "files_hashed": g1["n_files_recorded"]}
all_checks_ok = n_failed == 0 and all(c_["ok"] is not False for c_ in checks)
verdict = ("discrepancies" if (not all_checks_ok or sev["must-fix"]) else "layers hold with wording fixes" if sev["should-fix"] else "layers hold")        # notes alone do not change the verdict
verdict_text = (
    "Round 2, on the export regenerated on 2026-10-30 11:30 (code/27_gis_layers.py revised after round 1, sha256 " + rec["code"]["sha256"][:12] + "). Every item of round 1 that concerned the data or the licence marking is closed and the numbers did not move: "
    f"the 12 gauge points are the 5-decimal positions of camels_topo.txt (largest displacement {max(disp_now):.1f} m; round 1 {min(disp_r1):.1f}-{max(disp_r1):.1f} m), elev_at_gauge_m equals results/basin_gis.json and the reviewer's raster lookup in 12 of 12 (round 1: 9 differed), "
    f"distance_to_outline_km equals basin_gis.json to {dev['gauge_distance_vs_basin_gis_km']:.0e} km, the check table now covers the gauges; the README tables have their header's cell count and the sources table shows UNVERIFIED for all three products when rendered; the citation sentence says only what readme.txt "
    "and data_flow.json say; 'derived from sources whose terms are not recorded here' replaces 'inherit'; the README and results/gis_layers.json carry an independent-review line and block that quote round 1 faithfully. "
    f"{len(since['same_as_round_1'])} of 29 files are byte-identical to the ones round 1 reviewed (24 GeoTIFFs, basins_12.geojson, the CSV) and the {n_reg} non-gauge values of round 1's per-basin table were recomputed again: {len(reg_diffs)} differ. "
    f"Everything recomputed matches ({N['values_compared']} scalar comparisons and {(g5['tables']['numeric_cells_compared'] + g5['tables']['text_cells_compared']) * 2} README table cells, {len(MISMATCH)} mismatches; {counts['raster_cells_compared_with_their_sources_not_in_values_compared']:,} raster cells equal their source); a clean rebuild reproduces 27 of 29 hashes (not: the GPKG write time and the README time stamp). "
    "What is left is wording and plumbing, no number: the JSON disclosure still dates the run 2026-10-29; independent_review in results/gis_layers.json and numbers.py read round 1's top-level fields, which round 2 supersedes; two older builders still sum the record's round-1 counts.")

scripts = [f"code/review/gis_layers/{n_}" for n_ in ("g_common.py", "g1_files.py", "g2_vectors.py", "g3_elevation.py", "g4_landcover.py", "g5_readme.py", "g7_rebuild.py", "g6_assemble.py", "g6_round1_assemble.py", "run_checks.sh")]
env = {"python": sys.version.split()[0], "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__, "geopandas": gpd.__version__, "shapely": shapely.__version__, "pyproj": pyproj.__version__,
       "numpy": np.__version__, "pandas": pd.__version__, "platform": sys.platform}
env_same = {k_: (rec["environment"].get(k_) == env.get(k_)) for k_ in env if k_ in rec["environment"]}
t_start = manifest_before.get("time", "")
t_end = time.strftime("%H:%M")

# ---- round 1 intact: what the append has to preserve ------------------------------------------------------------------------------------------------------------------------------------
base_obj = {k_: v_ for k_, v_ in project_obj.items() if not re.fullmatch(r"round_\d+", k_)}
nl = CRLF if CRLF in project_text else LF
base_text = json.dumps(base_obj, indent=1, ensure_ascii=False, default=g_common._default).replace(LF, nl)
base_bytes = base_text.encode("utf-8")
had_round_2 = "round_2" in project_obj
if had_round_2:
    prior = (project_obj["round_2"].get("round_1_intact") or {}).get("file_sha256_before_this_append")
    assert prior == hashlib.sha256(base_bytes).hexdigest(), "round 1 in the project differs from the one round_2 recorded"
else:
    assert base_bytes == REVIEW_RECORD.read_bytes(), "json.dumps does not reproduce the project record byte for byte; refusing to append"
canonical = json.dumps(base_obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=g_common._default)
round_1_intact = {
    "statement": ("round 1 above is kept as it is: this append re-serialises the record with json.dumps(indent=1, ensure_ascii=False) and the file's own CRLF line endings, which reproduces the round-1 bytes exactly (checked before writing, base_bytes == file), "
                  "adds the key round_2 at the end and touches nothing else; after writing, the file without round_2 is rebuilt and its sha256 compared with file_sha256_before_this_append (checked). The top-level verdict, dates, counts and findings are therefore still round 1's; "
                  "round_2 repeats those fields under the same names for a reader that wants the latest word."),
    "file_sha256_before_this_append": hashlib.sha256(base_bytes).hexdigest(), "canonical_sha256_of_round_1_content": hashlib.sha256(canonical.encode("utf-8")).hexdigest(), "keys_of_round_1": list(base_obj.keys()),
    "round_1_verdict": r1["verdict"], "round_1_counts": {k_: r1["counts"][k_] for k_ in ("checks_listed", "failed", "must_fix", "should_fix", "notes", "numerical_mismatches")},
    "round_1_assembler": {"file": "code/review/gis_layers/g6_round1_assemble.py", "sha256_as_it_wrote_round_1": "e770290b2815980481332b2a6d640e0a2556f5bdfa5f9341b11d2d284fedc5ca",
                          "note": "frozen; it cannot be rerun against the regenerated export (its checks 2e, 6b, 6c were literal False)"},
}

round_2 = {
    "who": "the same independent AI reviewer agent (Claude) as in round 1, started again at the coordinator's request; no right to edit the project except code/review/gis_layers/ and this file; the layers were written and revised by a different AI agent session (Claude Code) directed by the team lead",
    "human_review": "not done: the team's own check of these layers is still open (team.json untouched by this review)",
    "when": f"2026-10-30, about 11:40 (reading what changed) to {t_end} (local time, UTC+8; first check run {t_start[11:16] or '?'}), after the coordinator applied round 1 and reran code/27_gis_layers.py (results/gis_layers.json 'written' {rec['written']})",
    "dates": {"export_run": rec["written"], "review": "2026-10-30", "round_1_export_run": r1.get("dates", {}).get("export_run")},
    "what": ("Second round: the whole export again as it stands after the rerun - the same seven checks with the reviewer's own code (scripts unchanged except the two stale wordings of g5_readme.py and the literal ok=False flags of g6, now computed), "
             "plus the new parts: the gauge points, their elevation cell and distance against camels_topo.txt, the raster and results/basin_gis.json, and the check table's gauge columns; the README tables' cell counts, the rendered licence column and the angle-bracket words; "
             "the new citation sentence and the 'derived from sources whose terms are not recorded here' wording against readme.txt, readme_shapefiles.txt and data_flow.json; the independent-review line and block against the record; the JSON disclosure; "
             "26 of the 29 files against the hashes round 1 recorded; a clean rebuild of the revised script (with the review record as input)."),
    "round_1_intact": round_1_intact,
    "files_reviewed": {"code/27_gis_layers.py": rec["code"]["sha256"], "code/07_basin_maps.py": rec["code"]["related_sha256"], "results/gis_layers.json": integrity["gis_layers_json_sha256"], "results/gis_layers/README.md": integrity["readme_sha256"],
                       "results/gis_layers/*": "28 further files, each sha256 equal to results/gis_layers.json files[] (check 1); 26 of the 29 equal to round 1 (check x3); README.md, basins_12.gpkg, gauges_12.geojson and results/gis_layers.json change with every run (time stamps), the other 27 files do not",
                       "results/review_gis_layers.json (round 1, as the script read it)": hashlib.sha256(r1_bytes).hexdigest()},
    "method": {"same_as_round_1": "area by pyproj+shapely, cells inside by shapely.contains_xy (elevation in degrees, land cover on the projected outline), own SRTM .hgt reader, own README table parser, python-markdown rendering, scratch GeoPackage copy, before/after manifest of the export folder",
               "new_in_round_2": ["gauge positions and values against camels_topo.txt and basin_gis.json (g2, g3, g6 cmp)", "README table cell counts by an independent scan (g5)", "the independent-review line and block against round 1 of this record (g1, g5)",
                                  "hashes of 26 files against round 1 (g1, ROUND1_SHA256 copied from the old results/gis_layers.json)", "rebuild with the review record as an input (g7)", "append that keeps round 1 byte for byte (g6)"],
               "ok_flags": "computed from the recomputed values; the literal False constants of the round-1 assembler are gone (g6_round1_assemble.py is frozen)"},
    "environment": env, "environment_equals_record_environment": env_same,
    "checks": checks,
    "reproduced": {
        "per_basin": {i: {"short": per[i]["short"], "area_km2_file_reviewer": [per[i]["area_km2"]["file"], per[i]["area_km2"]["reviewer_epsg5070"]], "elev_mean_m_file_reviewer_camels": [per[i]["elev_mean_m"]["file"], per[i]["elev_mean_m"]["reviewer_cos_weighted"], per[i]["elev_mean_m"]["camels"]],
                         "forest_share_file_reviewer_camels": [per[i]["forest_share"]["file"], per[i]["forest_share"]["reviewer"], per[i]["forest_share"]["camels_frac_forest"]],
                         "cells_elevation_landcover": [per[i]["elevation_cells"]["inside_file"], per[i]["land_cover_cells"]["inside_file"]], "gauge": per[i]["gauge"]} for i in ids},
        "maximum_deviation_file_vs_reviewer": dev,
        "totals": {"elevation_cells_inside": g3["totals"]["cells_inside_mine"], "elevation_cells_written_with_buffer": g3["totals"]["cells_written_valid"], "elevation_voids_inside": g3["totals"]["voids_inside"],
                   "land_cover_cells_inside": g4["total_cells_inside_mine"], "land_cover_classes_present": sum(1 for n_ in g4["classes_present_in_n_basins_mine"].values() if n_ > 0), "files": g1["n_files_recorded"], "bytes": g1["sum_of_file_bytes"]},
        "gauge_fix_against_round_1": gauge_fix,
        "gauge_summary": {"displacement_file_vs_camels_m": {"round_1": {"min": min(disp_r1), "median": statistics.median(disp_r1), "max": max(disp_r1)}, "now": {"min": min(disp_now), "max": max(disp_now)}},
                          "elev_at_gauge_differs_from_basin_gis_in": {"round_1": 9, "now": sum(1 for i in ids if per[i]["gauge"]["elev_at_gauge_m_file"] != per[i]["gauge"]["elev_at_gauge_m_basin_gis"])},
                          "gauge_cell_above_lowest_cell_of_basin_m": above12_now, "distance_to_outline_km_max": max(per[i]["gauge"]["distance_to_edge_km_file_attr"] for i in ids)},
        "regression_against_round_1": {"files_same_as_round_1": len(since["same_as_round_1"]), "files_changed": since["changed_since_round_1"], "per_basin_values_compared": n_reg, "per_basin_values_that_differ": reg_diffs},
        "mismatches": MISMATCH,
        "geojson": g2["geojson"], "readme_claims": [{"claim": c_["claim"], "ok": c_["ok"], **({"detail": c_["detail"]} if not c_["ok"] else {})} for c_ in g5["claims"]],
        "rebuild": ({k_: g7.get(k_) for k_ in ("returncode", "seconds", "n_equal_to_recorded", "not_equal_to_recorded", "rebuilt_record_check_table_equal", "rebuilt_independent_review_equal", "rebuilt_licences_equal", "readme_lines_differing_count",
                                               "review_record_copied_info")} if not g7.get("skipped") else "skipped"),
    },
    "round_1_items_status": items,
    "findings": fx,
    "still_open": [f"{f_['severity']}: {f_['title']}" for f_ in fx],
    "counts": counts,
    "verdict": verdict,
    "verdict_text": verdict_text,
    "scripts": scripts,
    "scripts_how_to_run": "from the project root with the system Python 3.13: REVIEW_OUT=<folder> bash code/review/gis_layers/run_checks.sh [--write-into-project]  (g1, g2, g3, g4, g5, g7, g6; about a minute); --write-into-project APPENDS round_2 to results/review_gis_layers.json and replaces an earlier round_2, never round 1; REVIEW_SKIP_REBUILD=1 skips g7",
    "not_checked": ["the vertical reference (EGM96) and the SRTM/3DEP make-up of the elevation tiles (the project's records only)", "NLCD at MRLC/USGS and the CAMELS tables at NCAR (cached copies only; no network use)",
                    "opening the layers in QGIS or ArcGIS (not run); no GeoPackage validator", "whether the CAMELS elev_mean column of the three mountain basins is right (only that the layers state the difference as basin_gis.json does)",
                    "what the documents will print after the next rebuild (numbers.py and the two older builders were read, not run)", "the team's own check of the layers (not done)"],
    "dates_detail": {"round_1": "2026-10-30 10:41-11:12 on the export of 2026-10-29 18:25", "round_2": f"2026-10-30 about 11:40-{t_end} on the export of {rec['written']}"},
}
jsave("review_gis_layers_round2.json", round_2)

if write_in:
    new_obj = dict(base_obj)
    new_obj["round_2"] = round_2
    new_text = json.dumps(new_obj, indent=1, ensure_ascii=False, default=g_common._default).replace(LF, nl)
    new_bytes = new_text.encode("utf-8")
    tail = (nl + "}").encode("utf-8")
    assert base_bytes.endswith(tail) and new_bytes.startswith(base_bytes[:-len(tail)] + b","), "round 1 would not be preserved byte for byte; nothing written"
    REVIEW_RECORD.write_bytes(new_bytes)
    after, info_after = review_record_round1()
    assert hashlib.sha256(after).hexdigest() == round_1_intact["file_sha256_before_this_append"], "round 1 changed by the append"
    json.loads(REVIEW_RECORD.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    print("appended round_2 to results/review_gis_layers.json;", "round 1 intact (sha256 of the file without round_2 =", round_1_intact["file_sha256_before_this_append"][:16] + "...)", "| rounds in the file:", info_after["rounds_appended"])

# ---- console summary ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("ROUND 2 VERDICT:", verdict)
print("counts:", counts)
print("checks:", [(c_["id"], c_["ok"]) for c_ in checks])
print("export folder changed since g1:", changed, "| new in results/:", new_in_results, "| pycache in review folder:", integrity["pycache_in_review_folder"])
print("max deviations:", dev)
print("mismatch list (", len(MISMATCH), "):", MISMATCH[:10])
print("round-1 per-basin values compared again:", n_reg, "differ:", len(reg_diffs), reg_diffs[:5])
print("round 1 items:", [(k_ + 1, x["status"][:40]) for k_, x in enumerate(items)])
print("findings:", [(f_["severity"], f_["where"][:70]) for f_ in fx])
print("builders: numbers.py excludes:", numbers_excludes, "| reads top verdict:", numbers_reads_top_verdict, "| short slides sum all:", short_slides_sums_all, "| short report sums all:", short_report_sums_all, "| short slides in run_all:", short_in_run_all)
print("gauge above lowest cell:", {k_: v_ for k_, v_ in above12_now.items()})
