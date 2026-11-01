"""Checks 5, 6 and 7: the README's three check tables against results/gis_layers.json check_table and against the reviewer's own recomputation (g2, g3, g4), the README's
file table against the record and the reviewer's measurements, the summary sentences, the known-difference sentence against results/basin_gis.json notes, the licence
wording against results/data_flow.json and data/camels_us/readme.txt, the disclosure line, the README's other factual statements, and a scan of the raster tags and of the
text files for a licence assertion. Needs g1 .. g4 to have run (their outputs are read from OUT)."""
from g_common import *
import g_common
import hashlib
import pyproj
import rasterio

ids, audit = basins()
t, v = camels()
rec = jload(RECORD)
bgfull = jload("results/basin_gis.json")
bg = bgfull["basins"]
flow = jload("results/data_flow.json")
g1, g2, g3, g4 = jout("g1_files.json"), jout("g2_vectors.json"), jout("g3_elevation.json"), jout("g4_landcover.json")
readme = (GIS_DIR / "README.md").read_text(encoding="utf-8")
CT = {r["gauge_id"]: r for r in rec["check_table"]}
res = {"claims": [], "tables": {}}


def md_table(text, heading):
    """The first markdown table after a heading line: (header cells, body rows as lists of cells)."""
    pos = text.index(heading)
    rows, started = [], False
    for ln in text[pos:].splitlines()[1:]:
        if ln.startswith("|"):
            started = True
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif started:
            break
    return rows[0], rows[2:]


def num(s):
    s = s.replace("−", "-").replace("`", "").replace("%", "").strip()
    return float(s)


def decimals(s):
    s = s.replace("`", "").strip()
    return len(s.split(".")[1]) if "." in s else 0


def close_to_rounded(cell, value):
    """True if the README cell (a rounded number) is the rounding of `value` to the cell's own number of decimals (half a unit in the last place of slack)."""
    d = decimals(cell)
    return abs(num(cell) - value) <= 0.5 * 10 ** (-d) + 1e-9


def add(name, ok, detail=None, where=None):
    res["claims"].append({"claim": name, "ok": bool(ok), "detail": detail, "where": where})
    return ok


# ======================= the three README check tables ==========================================================================================================================
h, body = md_table(readme, "### Area (km2)")
rows_area = {r[0]: r for r in body}
h2_, body2 = md_table(readme, "### Mean elevation (m)")
rows_elev = {r[0]: r for r in body2}
h3_, body3 = md_table(readme, "### Dominant land cover and forest share")
rows_forest = {r[0]: r for r in body3}
res["tables"]["headers"] = {"area": h, "elev": h2_, "forest": h3_}
add("README area table has 12 rows in audit order", list(rows_area) == ids and len(body) == 12)
add("README elevation table has 12 rows in audit order", list(rows_elev) == ids and len(body2) == 12)
add("README forest table has 12 rows in audit order", list(rows_forest) == ids and len(body3) == 12)

mism = {"vs_check_table": [], "vs_my_recomputation": []}
n_num_cells = n_txt_cells = 0
for i in ids:
    ra, re_, rf = rows_area[i], rows_elev[i], rows_forest[i]
    ct = CT[i]
    # my numbers
    a_mine = g2["per_basin"][i]["area_planar_5070_km2_mine"]
    a_bg = g2["per_basin"][i]["area_albers_km2_basin_gis"]
    a_c = g2["per_basin"][i]["camels_area_gages2"]
    z_mine = g3["per_basin"][i]["elev_mean_weighted_mine"]
    z_bg = g3["per_basin"][i]["elev_mean_basin_gis"]
    z_c = g3["per_basin"][i]["elev_mean_camels"]
    f_mine = g4["per_basin"][i]["forest_share_mine"]
    f_bg = g4["per_basin"][i]["forest_share_basin_gis"]
    f_c = g4["per_basin"][i]["frac_forest_camels"]
    # area table: [id, basin, exported, basin_gis, diff, camels, diff%]
    pairs = [
        ("area.exported", ra[2], ct["area_km2"], a_mine), ("area.basin_gis", ra[3], ct["basin_gis_area_km2"], a_bg),
        ("area.diff", ra[4], ct["area_vs_basin_gis_km2"], a_mine - a_bg), ("area.camels", ra[5], ct["camels_area_gages2_km2"], a_c),
        ("area.diff_pct", ra[6], ct["area_vs_camels_pct"], (a_mine - a_c) / a_c * 100.0),
        # elevation table: [id, basin, exported, basin_gis, diff, camels, diff, note]
        ("elev.exported", re_[2], ct["elev_mean_m"], z_mine), ("elev.basin_gis", re_[3], ct["basin_gis_elev_mean_m"], z_bg),
        ("elev.diff", re_[4], ct["elev_vs_basin_gis_m"], z_mine - z_bg), ("elev.camels", re_[5], ct["camels_elev_mean_m"], z_c),
        ("elev.diff_camels", re_[6], ct["elev_vs_camels_m"], z_mine - z_c),
        # forest table: [id, basin, dom group, basin_gis, same, dom class, forest %, forest % bgis, camels %, diff pts, camels dom]
        ("forest.exported_pct", rf[6], ct["forest_share"] * 100, f_mine * 100), ("forest.basin_gis_pct", rf[7], ct["basin_gis_forest_share"] * 100, f_bg * 100),
        ("forest.camels_pct", rf[8], ct["camels_frac_forest"] * 100, f_c * 100), ("forest.diff_points", rf[9], ct["forest_vs_camels_points"], (f_mine - f_c) * 100),
    ]
    for name, cell, ctv, mine in pairs:
        n_num_cells += 1
        if not close_to_rounded(cell, ctv):
            mism["vs_check_table"].append((i, name, cell, ctv))
        if not close_to_rounded(cell, mine):
            mism["vs_my_recomputation"].append((i, name, cell, mine))
    # text cells
    txt_pairs = [
        ("forest.dominant_group", rf[2], ct["dominant_land_cover"], g4["per_basin"][i]["dominant_group_mine"]),
        ("forest.dominant_group_basin_gis", rf[3], ct["basin_gis_dominant_land_cover"], g4["per_basin"][i]["dominant_group_basin_gis"]),
        ("forest.same", rf[4], "yes" if ct["dominant_same"] else "no", "yes" if g4["per_basin"][i]["dominant_group_mine"] == g4["per_basin"][i]["dominant_group_basin_gis"] else "no"),
        ("forest.dominant_class", rf[5], ct["dominant_nlcd_class"], g4["per_basin"][i]["dominant_class_mine"]),
        ("forest.camels_dom", rf[10], ct["camels_dom_land_cover"], g4["per_basin"][i]["camels_dom_land_cover"]),
        ("basin_name", ra[1], ct["short"], g2["per_basin"][i]["short"]),
    ]
    for name, cell, ctv, mine in txt_pairs:
        n_txt_cells += 1
        if cell != ctv:
            mism["vs_check_table"].append((i, name, cell, ctv))
        if cell != mine:
            mism["vs_my_recomputation"].append((i, name, cell, mine))
    # the note column of the elevation table: filled exactly for the basins more than 25 m from CAMELS
    flagged = abs(z_mine - z_c) > MOUNTAIN_THRESHOLD_M
    if bool(re_[7]) != flagged:
        mism["vs_my_recomputation"].append((i, "elev.note", re_[7], flagged))
    if bool(re_[7]) != bool(ct["elev_differs_from_camels_by_more_than_25_m"]):
        mism["vs_check_table"].append((i, "elev.note", re_[7], ct["elev_differs_from_camels_by_more_than_25_m"]))
res["tables"]["mismatches"] = mism
res["tables"]["numeric_cells_compared"] = n_num_cells          # each against the check_table AND against the recomputation
res["tables"]["text_cells_compared"] = n_txt_cells
add("README check tables equal results/gis_layers.json check_table (every cell, within the printed rounding)", not mism["vs_check_table"], mism["vs_check_table"][:10])
add("README check tables equal the reviewer's recomputation (every cell, within the printed rounding)", not mism["vs_my_recomputation"], mism["vs_my_recomputation"][:10])

# ---- the sentences around the tables ---------------------------------------------------------------------------------------------------------------------------------
da = np.array([(g2["per_basin"][i]["area_planar_5070_km2_mine"] - g2["per_basin"][i]["camels_area_gages2"]) / g2["per_basin"][i]["camels_area_gages2"] * 100 for i in ids])
dz = np.array([g3["per_basin"][i]["elev_mean_weighted_mine"] - g3["per_basin"][i]["elev_mean_camels"] for i in ids])
df = np.array([(g4["per_basin"][i]["forest_share_mine"] - g4["per_basin"][i]["frac_forest_camels"]) * 100 for i in ids])
sh = {i: g2["per_basin"][i]["short"] for i in ids}
m = re.search(r"area median ([+-][\d.]+)% \(largest ([+-][\d.]+)%, ([^)]+)\); mean elevation median ([+-][\d.]+) m \(largest ([+-][\d.]+) m, ([^)]+)\); forest share median ([+-][\d.]+) points \(largest ([+-][\d.]+), ([^)]+)\)", readme)
if m:
    got = m.groups()
    mine = (f"{np.median(da):+.2f}", f"{da[np.argmax(np.abs(da))]:+.1f}", sh[ids[int(np.argmax(np.abs(da)))]],
            f"{np.median(dz):+.1f}", f"{dz[np.argmax(np.abs(dz))]:+.0f}", sh[ids[int(np.argmax(np.abs(dz)))]],
            f"{np.median(df):+.1f}", f"{df[np.argmax(np.abs(df))]:+.1f}", sh[ids[int(np.argmax(np.abs(df)))]])
    add("README summary sentence (medians and largest differences against CAMELS) equals the recomputation", got == mine, {"readme": got, "mine": mine})
else:
    add("README summary sentence found", False, "pattern not found")
add("README says '12 of 12 basins agree'", "**12 of 12 basins agree.**" in readme)
# 12 of 12 against basin_gis.json: my own tolerances check
tol_ok = {i: (abs(g2["per_basin"][i]["area_attr_minus_basin_gis_km2"]) <= TOL["area_km2"]
              and abs(g3["per_basin"][i]["elev_mean_weighted_mine"] - g3["per_basin"][i]["elev_mean_basin_gis"]) <= TOL["elev_m"]
              and abs(g4["per_basin"][i]["forest_share_mine"] - g4["per_basin"][i]["forest_share_basin_gis"]) <= TOL["share"]
              and g3["per_basin"][i]["n_cells_inside_mine"] == g3["per_basin"][i]["n_cells_inside_basin_gis"]
              and g4["per_basin"][i]["n_cells_inside_mine"] == g4["per_basin"][i]["n_cells_inside_basin_gis"]
              and g4["per_basin"][i]["dominant_group_mine"] == g4["per_basin"][i]["dominant_group_basin_gis"]) for i in ids}
add("12 of 12 basins agree with results/basin_gis.json within the README's tolerances (own recomputation)", all(tol_ok.values()), tol_ok)

# ---- the three mountain basins and the known-difference sentence ----------------------------------------------------------------------------------------------------
notes = bgfull["notes"]
elev_note = next(n for n in notes if n.startswith("Elevation: our mean differs"))
add("README quotes the 'Elevation:' note of results/basin_gis.json word for word", f'The note of script 07 reads: "{elev_note}"' in readme)
add("results/gis_layers.json known_note_from_basin_gis equals the same note", rec["known_note_from_basin_gis"] == elev_note)
off = [(i, dz[k]) for k, i in enumerate(ids) if abs(dz[k]) > MOUNTAIN_THRESHOLD_M]
off_txt = ", ".join(f"{sh[i]} {d:+.0f} m" for i, d in off)
add("exactly three basins differ from CAMELS elev_mean by more than 25 m and the README names them with the same rounded differences (Catawba +47, French Broad +100, South Toe +163)",
    off_txt == "Catawba, NC +47 m, French Broad, NC +100 m, South Toe, NC +163 m" and off_txt in readme, {"mine": off_txt})
other_max = max(abs(d) for k, d in enumerate(dz) if abs(d) <= MOUNTAIN_THRESHOLD_M)
add("the largest difference among the other nine basins is below 25 m (SF Holston -16.5)", other_max < 25, {"largest_other_abs": other_max})
# the numeric statements inside the note, tested with the reviewer's numbers
bymean = {i: g3["per_basin"][i]["raster"] for i in ids}
dbf = np.array([g3["per_basin"][i]["elev_mean_outline_file"] for i in ids])
ours = np.array([g3["per_basin"][i]["elev_mean_weighted_mine"] for i in ids])
camels_z = np.array([g3["per_basin"][i]["elev_mean_camels"] for i in ids])
d_dbf = ours - dbf
from math import isclose
add("note: 'median +1.6 m'", isclose(np.median(dz), 1.6, abs_tol=0.05), {"median": float(np.median(dz))})
add("note: 'elev_mean field of the outline file is close to ours (mean absolute difference 8.4 m, largest -24.7 m, South Toe)'",
    isclose(np.mean(np.abs(d_dbf)), 8.4, abs_tol=0.05) and isclose(d_dbf[np.argmax(np.abs(d_dbf))], -24.7, abs_tol=0.05) and sh[ids[int(np.argmax(np.abs(d_dbf)))]] == "South Toe, NC",
    {"mean_abs": float(np.mean(np.abs(d_dbf))), "largest": float(d_dbf[np.argmax(np.abs(d_dbf))]), "dbf_values_of_the_three": {sh[i]: g3["per_basin"][i]["elev_mean_outline_file"] for i, _ in off}})
by_grid = [abs(bg[i]["elev_mean_by_grid_m"]["920 m"] - bg[i]["elev_mean_by_grid_m"]["30 m"]) for i in ids]
add("note: 'averaging our grid to coarser cells (up to 920 m) moves our mean by at most 2.5 m' (from basin_gis.json elev_mean_by_grid_m)", max(by_grid) <= 2.5 + 0.05, {"max": max(by_grid)})
rho = float(np.corrcoef(pd.Series(ours).rank().to_numpy(), pd.Series(camels_z).rank().to_numpy())[0, 1])      # Spearman = Pearson on the ranks
add("note: 'rank correlation 0.99' between our means and the CAMELS table", isclose(rho, 0.99, abs_tol=0.005), {"spearman": rho})
mount_ours = [sh[i] for k, i in enumerate(ids) if ours[k] > 600]
mount_camels = [sh[i] for k, i in enumerate(ids) if camels_z[k] > 600]
add("note: 'landscape classes are unchanged with our elevations (mountain = mean elevation above 600 m)'", mount_ours == mount_camels and len(mount_ours) == 4,
    {"mountain_by_ours": mount_ours, "mountain_by_camels": mount_camels, "smallest_mountain_ours": float(min(ours[ours > 600])), "smallest_mountain_camels": float(min(camels_z[camels_z > 600]))})

# ======================= README file table vs the record vs the reviewer's measurements ===========================================================================================
hf, bodyf = md_table(readme, "## Files")
fmap = {r[0].strip("`"): r for r in bodyf}
recf = {Path(f["file"]).name: f for f in rec["files"] if not f["file"].endswith("README.md")}
add("README file table lists the 28 files other than the README, in the record's order", list(fmap) == list(recf) and len(fmap) == 28)
bad_rows = []
g1rows = {Path(r["file"]).name: r for r in g1["rows"]}
for name, r in fmap.items():
    f = recf[name]
    mine = g1rows[name]
    issues = []
    if r[8].strip("`") != mine["sha256"]:
        issues.append("sha256 differs from the file")
    if r[7] != f"{mine['bytes'] / 1e6:.2f} MB":
        issues.append(f"size cell {r[7]} vs {mine['bytes'] / 1e6:.2f} MB")
    for k, col in (("kind", 1), ("layer", 2), ("crs", 3), ("resolution", 4), ("units", 5)):
        if r[col] != f[k]:
            issues.append(f"{k} differs from the record")
    if r[6] != str(f["nodata"]):
        issues.append("nodata differs from the record")
    if name.startswith("elevation_") and name.endswith(".tif"):
        i = name[len("elevation_"):-4]
        p = g3["per_basin"][i]
        want = f"1 arc-second ({round(p['grid']['cell_m_ew_at_raster_centre'], 2)} m east-west x {round(p['grid']['cell_m_ns_at_raster_centre'], 2)} m north-south here); {p['raster']['width']} x {p['raster']['height']} cells"
        if r[4] != want:
            issues.append(f"resolution cell {r[4]!r} vs measured {want!r}")
        if r[6] != "-32768":
            issues.append("nodata cell not -32768")
    if name.startswith("landcover_") and name.endswith(".tif"):
        i = name[len("landcover_"):-4]
        p = g4["per_basin"][i]
        want = f"30 m; {p['raster']['width']} x {p['raster']['height']} cells"
        if r[4] != want:
            issues.append(f"resolution cell {r[4]!r} vs measured {want!r}")
        if r[6] != "250":
            issues.append("nodata cell not 250")
    if issues:
        bad_rows.append((name, issues))
add("README file table: sha256, size, kind, CRS, resolution (cell size and shape measured by the reviewer), units and nodata equal the record and the files", not bad_rows, bad_rows[:6])

# ======================= other statements of the README ===========================================================================================================================
add("README: '28 files (9.4 MB) plus this README'", "28 files (9.4 MB) plus this README" in readme and g1["kinds"] == {"gpkg": 1, "geojson": 2, "csv": 1, "elevation_tif": 12, "landcover_tif": 12, "readme": 1}
    and round(g1["total_bytes_without_readme"] / 1e6, 1) == 9.4, {"without_readme_bytes": g1["total_bytes_without_readme"], "with_readme_bytes": g1["sum_of_file_bytes"]})
rep = [i for i in ids if not g2["per_basin"][i]["source_valid"]]
add("README: '2 invalid geometries repaired by buffer(0) (Hyco Creek, NC, Charlie Creek, FL; areas unchanged)'",
    rep == ["02077200", "02296500"] and all(abs(g2["per_basin"][i]["area_mine_minus_dbf_km2"]) < 1e-6 for i in ids) and all(g2["checks"]["outline_validity"].values()),
    {"invalid_in_source": rep, "max_abs_area_minus_shapefile_AREA_km2": max(abs(g2["per_basin"][i]["area_mine_minus_dbf_km2"]) for i in ids)})
add("README: 'taken unchanged' for the other ten outlines (exact vertex equality with the shapefile)", all(g2["per_basin"][i]["geometry_exact_equal_to_source"] for i in ids if i not in rep))
dist_max = max(g2["per_basin"][i]["gauge_dist_edge_km_attr"] for i in ids)
add("README: gauges 'inside its outline (12 of 12 do)' and 'at most 0.63 km'", all(g2["per_basin"][i]["gauge_inside_file_point"] for i in ids) and round(dist_max, 2) == 0.63 and "(12 of 12 do)" in readme
    and "(at most 0.63 km)" in readme, {"max_distance_attr_km": dist_max,
                                         "max_distance_mine_edge_file_point_km": max(g2["per_basin"][i]["gauge_dist_edge_km_file_point_mine"] for i in ids),
                                         "max_distance_mine_edge_camels_point_km": max(g2["per_basin"][i]["gauge_dist_edge_km_camels_point_mine"] for i in ids)})
camels_pos = all(g2["per_basin"][i]["gauge_file_equals_camels_topo"] for i in ids)
geodg = pyproj.Geod(ellps="GRS80")
audit_shift_m = {i: geodg.inv(float(audit[i]["gauge_lon"]), float(audit[i]["gauge_lat"]), float(t.loc[i, "gauge_lon"]), float(t.loc[i, "gauge_lat"]))[2] for i in ids}
audit_is_rounding = all(abs(float(audit[i]["gauge_lat"]) - round(float(t.loc[i, "gauge_lat"]), 3)) < 1e-9 and abs(float(audit[i]["gauge_lon"]) - round(float(t.loc[i, "gauge_lon"]), 3)) < 1e-9 for i in ids)
sources_attr = sorted({g2["per_basin"][i]["gauge_source_attr"] for i in ids})
add("README Gauges bullet: 'the 12 gauge points at the CAMELS gauge_lat/gauge_lon of data/camels_us/camels_topo.txt, as the table gives them (not rounded ...)' - the points ARE the table's 5-decimal positions, and the layer's source attribute says so",
    camels_pos and "of `data/camels_us/camels_topo.txt`, as the table gives them (not rounded" in readme and all("camels_topo.txt gauge_lat, gauge_lon" in s_ and "not rounded" in s_ for s_ in sources_attr),
    {"points_equal_camels_topo_5_decimals": camels_pos, "max_displacement_vs_camels_topo_m": max(g2["per_basin"][i]["gauge_displacement_file_vs_camels_m"] for i in ids), "source_attribute_values": sources_attr},
    "README 'What the layers are' > Gauges; basins_12.gpkg layer gauges columns lat, lon, source; gauges_12.geojson")
add("README Gauges bullet: 'results/data_audit.json holds the same values rounded to 0.001 degree, which would put the points 9 to 60 m away'",
    audit_is_rounding and f"{round(min(audit_shift_m.values())):.0f} to {round(max(audit_shift_m.values())):.0f} m away" in readme,
    {"data_audit_is_the_table_rounded_to_3_decimals": audit_is_rounding, "shift_m_min": min(audit_shift_m.values()), "shift_m_max": max(audit_shift_m.values())})
gauge_elev_eq = all(g3["per_basin"][i]["gauge_cell"]["attr"] == g3["per_basin"][i]["gauge_cell"]["basin_gis"] == g3["per_basin"][i]["gauge_cell"]["elev_at_exported_point"] for i in ids)
gauge_dist_max_diff = max(abs(g2["per_basin"][i]["gauge_dist_edge_km_attr"] - g2["per_basin"][i]["gauge_dist_edge_km_basin_gis"]) for i in ids)
add("README: 'The elevation cell and the distance equal results/basin_gis.json (part of the check below)' and '(tolerance 1e-06 km)': the gauge-cell elevation equals basin_gis.json and the reviewer's raster lookup in 12 of 12, the distance within 1e-6 km",
    gauge_elev_eq and gauge_dist_max_diff <= 1e-6 and "The elevation cell and the distance equal results/basin_gis.json (part of the check below)" in readme and "(tolerance 1e-06 km)" in readme,
    {"elev_equal_in_12": gauge_elev_eq, "max_abs_distance_difference_km": gauge_dist_max_diff})
ct_gauge_bad = [(i, k) for i in ids for k, want in (("gauge_elev_m", g3["per_basin"][i]["gauge_cell"]["elev_at_exported_point"]), ("basin_gis_gauge_elev_m", g3["per_basin"][i]["gauge_cell"]["basin_gis"]),
                                                      ("gauge_distance_to_outline_km", g2["per_basin"][i]["gauge_dist_edge_km_attr"]), ("basin_gis_gauge_distance_km", g2["per_basin"][i]["gauge_dist_edge_km_basin_gis"]))
                 if abs(float(CT[i][k]) - float(want)) > 1e-6]
ct_gauge_bad += [(i, "agrees_with_basin_gis") for i in ids if not CT[i]["agrees_with_basin_gis"]]
add("results/gis_layers.json check_table gauge columns (gauge_elev_m, gauge_distance_to_outline_km and their basin_gis.json twins) equal the reviewer's values; agrees_with_basin_gis is true in 12 of 12", not ct_gauge_bad and len(CT) == 12, ct_gauge_bad[:6])
tot_cells = g3["totals"]
add("README: '8,930,167 cells inside the 12 outlines, 9,780,302 written including the buffer. Land cover: 7,912,882 cells inside the outlines, 0 without a valid class'",
    "8,930,167 cells inside the 12 outlines, 9,780,302 written including the buffer" in readme and "7,912,882 cells inside the outlines, 0 without a valid class" in readme
    and tot_cells["cells_inside_mine"] == 8930167 and tot_cells["cells_written_valid"] == 9780302 and g4["total_cells_inside_mine"] == 7912882
    and all(g4["per_basin"][i]["cells_inside_but_nodata"] == 0 for i in ids), {"elev_inside": tot_cells["cells_inside_mine"], "elev_written": tot_cells["cells_written_valid"], "lc_inside": g4["total_cells_inside_mine"]})
add("README: 'Voids inside the outlines: 0 cells'", "Voids inside the outlines: 0 cells" in readme and tot_cells["voids_inside"] == 0)
add("README: 'Legend: landcover_classes.csv (15 classes, 15 of them present in the 12 basins)'",
    "(15 classes, 15 of them present in the 12 basins)" in readme and sum(1 for c, n in g4["classes_present_in_n_basins_mine"].items() if n > 0) == 15
    and all(g4["legend_basins_present_column"][str(c)] == n for c, n in g4["classes_present_in_n_basins_mine"].items()), g4["classes_present_in_n_basins_mine"])
ls = {}
for i in ids:
    ls[audit[i]["landscape"]] = ls.get(audit[i]["landscape"], 0) + 1
add("README: landscapes 'Northern lowlands and Piedmont (4), Mountains (above 600 m) (4), Southern lowlands (south of 32.5N) (4)'",
    all(f"{k} ({n})" in readme for k, n in ls.items()) and sorted(ls.values()) == [4, 4, 4], ls)
ns_m = [15 * g3["per_basin"][i]["grid"]["cell_m_ns_at_raster_centre"] for i in ids]
ew_m = [15 * g3["per_basin"][i]["grid"]["cell_m_ew_at_raster_centre"] for i in ids]
ring_words = f"(about {round(float(np.mean(ns_m)), -1):.0f} m north-south, {round(min(ew_m), -1):.0f}-{round(max(ew_m), -1):.0f} m east-west)"
add(f"README: the elevation layer is cut to the outline buffered by 15 arc-seconds, '{ring_words}'; the valid cells are within the reviewer's own 15-arc-second buffer (at most 1 cell outside in any basin)",
    all(g3["per_basin"][i]["ring"]["valid_outside_my_buffer"] <= 1 for i in ids) and f"15 arc-seconds {ring_words}" in readme,
    {"valid_outside_my_buffer": {i: g3["per_basin"][i]["ring"]["valid_outside_my_buffer"] for i in ids},
     "nodata_inside_my_buffer": {i: g3["per_basin"][i]["ring"]["nodata_inside_my_buffer"] for i in ids}, "15_arcsec_N_S_m": [min(ns_m), max(ns_m)], "15_arcsec_E_W_m": [min(ew_m), max(ew_m)]})
add("README: elevation 'cell centres on whole arc-seconds, origin at half a cell', 'int16', 'LZW', 'tiled' ; land cover 'uint8', 'LZW', colour table kept",
    all(abs(g3["per_basin"][i]["grid"]["left_offset_in_cells"] - 0.5) < 1e-6 and abs(g3["per_basin"][i]["grid"]["top_offset_in_cells"] - 0.5) < 1e-6 and g3["per_basin"][i]["raster"]["dtype"] == "int16"
        and g3["per_basin"][i]["raster"]["compression"] == "LZW" and g3["per_basin"][i]["raster"]["tiled"] and g3["per_basin"][i]["raster"]["image_structure"].get("PREDICTOR") == "2"
        and g4["per_basin"][i]["raster"]["dtype"] == "uint8" and g4["per_basin"][i]["raster"]["compression"] == "LZW" and g4["per_basin"][i]["raster"]["colorinterp"] == "palette"
        and g4["per_basin"][i]["colour_table_equals_csv_all_present_codes"] for i in ids))
add("every exported elevation sample equals the sample of the source tile in data/dem (reviewer's own SRTM reader); every exported land-cover cell equals the cached NLCD cell",
    all(g3["per_basin"][i]["source_tiles"]["n_value_mismatches"] == 0 and g3["per_basin"][i]["source_tiles"]["n_not_found_in_any_cached_tile"] == 0 for i in ids)
    and all(g4["per_basin"][i]["cache"].get("n_value_mismatches") == 0 and g4["per_basin"][i]["cache"].get("aligned_to_cache_grid") for i in ids),
    {"elevation_cells_compared": sum(g3["per_basin"][i]["source_tiles"]["n_compared"] for i in ids), "land_cover_cells_compared": sum(g4["per_basin"][i]["cache"]["n_valid_compared"] for i in ids)})

# the datum sentence for the GeoJSON twins (environment-dependent; recomputed here)
tg = pyproj.transformer.TransformerGroup(4269, 4326, always_xy=True)
desc0 = tg.transformers[0].description
add("README: GeoJSON twins 'NAD83 to WGS 84 (1) ... a null transformation, coordinates unchanged; 49 grid-shift operations are not installed'",
    desc0 in readme and "a null transformation, coordinates unchanged" in readme and g2["geojson"]["basins_12.geojson"]["max_vertex_diff_vs_gpkg_deg"] < 5.1e-8,
    {"pyproj_first_transformer": desc0, "n_unavailable_operations_here": len(tg.unavailable_operations), "max_vertex_diff_deg": g2["geojson"]["basins_12.geojson"]["max_vertex_diff_vs_gpkg_deg"]})

# ======================= licences (6) ==============================================================================================================================================
hs, bodys = md_table(readme, "## Sources and licences")
lic_cells = [r[-1] for r in bodys]                                   # the licence is the last cell of a row of the source text (see the column-count check below)
add("README source text: the last cell of each of the 3 rows of the sources table starts with UNVERIFIED", len(bodys) == 3 and all(c.startswith("UNVERIFIED") for c in lic_cells),
    [len(r) for r in bodys])


def cells(ln):
    return re.split(r"(?<!\\)\|", ln.strip().strip("|"))


tbl_issues, cur, start = [], [], 0
n_tbl = n_rows = 0
for n, ln in enumerate(readme.splitlines() + [""], 1):
    if ln.startswith("|"):
        if not cur:
            start = n
        cur.append((n, ln))
    elif cur:
        n_tbl += 1
        n_rows += len(cur) - 2                                         # header and the line of dashes are not data rows
        hdr = len(cells(cur[0][1]))
        tbl_issues += [{"table_starts_at_line": start, "line": n2, "cells": len(cells(l2)), "header_cells": hdr, "text_start": l2[:90]} for n2, l2 in cur if len(cells(l2)) != hdr]
        cur = []
add("every README table row has the header's number of cells (a literal '|' inside a cell splits it)", not tbl_issues, tbl_issues, "README 'Sources and licences' table, elevation-tiles row")
add("README verification line 'readme_tables_cells_per_row: 5 tables, 67 rows, each with its header's number of cells' equals the reviewer's own count", f"readme_tables_cells_per_row: {n_tbl} tables, {n_rows} rows, each with its header's number of cells" in readme
    and rec["verification"]["checks"]["readme_tables_cells_per_row"]["ok"] is True, {"tables": n_tbl, "data_rows": n_rows})
import markdown  # python-markdown, only to see what a reader's renderer shows
html = markdown.markdown(readme, extensions=["tables"])
t_i = html.index("<th>source</th>")
t_html = html[html.rindex("<table>", 0, t_i):html.index("</table>", t_i)]
rendered_rows = [re.findall(r"<t[dh]>(.*?)</t[dh]>", r, re.S) for r in re.findall(r"<tr>(.*?)</tr>", t_html, re.S)][1:]
rendered_lic = [r[-1] for r in rendered_rows]
add("README rendered (python-markdown tables): the licence column of each of the 3 rows of the sources table shows UNVERIFIED", all(c.startswith("UNVERIFIED") for c in rendered_lic) and len(rendered_lic) == 3,
    {"rendered_licence_cell_starts": [c[:60] for c in rendered_lic], "cells_per_rendered_row": [len(r) for r in rendered_rows]}, "README 'Sources and licences' table")
outside = []
for n, ln in enumerate(readme.splitlines(), 1):
    for m in re.finditer(r"<[A-Za-z_/][^>]*>", re.sub(r"`[^`]*`", "", ln)):
        outside.append({"line": n, "text": m.group(0)})
add("no <angle-bracket> text outside backticks (a renderer reads it as an HTML tag and shows nothing)", not outside, outside[:12], "README 'Sources and licences' table, 'used for' and 'what the project files record' cells")
fl = {d["key"]: d for d in flow["datasets"]}
login = {"camels_us": fl["camels_us"]["obtained"]["login_needed"], "elevation_tiles": fl["elevation_tiles"]["obtained"]["login_needed"], "nlcd": fl["nlcd"]["obtained"]["login_needed"]}
add("README: the 'login needed' text quoted in each licence cell equals results/data_flow.json", all(f"login needed = {login[k]}" in c for k, c in zip(("camels_us", "elevation_tiles", "nlcd"), lic_cells)), login)
def has_licence_field(o):
    if isinstance(o, dict):
        return any(("licen" in k.lower()) or has_licence_field(x) for k, x in o.items())
    if isinstance(o, list):
        return any(has_licence_field(x) for x in o)
    return False
add("results/data_flow.json and results/basin_gis.json hold no licence field (the basis of 'UNVERIFIED')", not has_licence_field(flow) and not has_licence_field(bgfull))
camels_readme = (PROJ / "data/camels_us/readme.txt").read_text(encoding="utf-8", errors="replace")
cite_attr = bool(re.search(r"CAMELS attributes in your publication\(s\), you agree to cite:\s*\nAddor", camels_readme)) or "agree to cite:\n\nAddor" in camels_readme or "Addor, N." in camels_readme
cite_ts = "By using the CAMELS time series in your publication(s), you agree to cite:" in camels_readme
mentions_outlines = bool(re.search(r"outline|shapefile|basin_set", camels_readme, re.I))
shp_readme = (PROJ / "data/camels_us/basin_dataset_public_v1p2/readme_shapefiles.txt").read_text(encoding="utf-8", errors="replace")
shp_asks = bool(re.search(r"agree to cite|please cite|should cite", shp_readme, re.I))
CITE = ("data/camels_us/readme.txt asks that publications using the CAMELS attributes cite Addor et al. 2017 and publications using the CAMELS time series cite Newman et al. 2015; "
        "results/data_flow.json names Newman et al. 2015 as the source of the outlines (no request to cite is recorded for them)")
cite_in_files = {Path(f["file"]).name: CITE in f["licence"] for f in rec["files"] if f["file"].endswith(("basins_12.gpkg", "basins_12.geojson", "gauges_12.geojson"))}
cite_facts = {"readme_txt_asks_Addor_for_attributes": cite_attr, "readme_txt_asks_Newman_for_time_series": cite_ts,
              "data_flow_producer_line_names_Newman_2015_for_time_series_and_outlines": "Newman et al. (2015) for the time series and outlines" in fl["camels_us"]["producer"],
              "readme_txt_mentions_outlines_or_shapefiles": mentions_outlines, "readme_shapefiles_txt_asks_to_cite": shp_asks}
add("README/JSON citation sentence (README row 1, licences.outlines_and_attributes, files[].licence of gpkg + 2 geojson) says only what data/camels_us/readme.txt and results/data_flow.json say: attributes -> Addor 2017, time series -> Newman 2015, outlines -> Newman 2015 named by data_flow.json, no request to cite recorded for them",
    CITE in readme and CITE in rec["licences"]["outlines_and_attributes"] and all(cite_in_files.values()) and len(cite_in_files) == 3
    and cite_attr and cite_ts and cite_facts["data_flow_producer_line_names_Newman_2015_for_time_series_and_outlines"] and not mentions_outlines and not shp_asks,
    {"sentence_in_readme": CITE in readme, "sentence_in_licences": CITE in rec["licences"]["outlines_and_attributes"], "sentence_in_files": cite_in_files, **cite_facts},
    "README 'Sources and licences' row 1; results/gis_layers.json licences.outlines_and_attributes and files[].licence")
# no other licence assertion in the README
lic_words = [m.group(0) for m in re.finditer(r"[^.|]*\b(public domain|CC0|CC-BY|CC BY|creative commons|open licen[cs]e|free to use|may be (re)?used|MIT licen[cs]e|ODbL)\b[^.|]*", readme, re.I)]
add("README asserts no licence of its own (no public-domain / Creative-Commons / open-licence wording)", not lic_words, lic_words)
WORDING = "the exported files are derived from sources whose terms are not recorded here"
inh = [m.group(0) for m in re.finditer(r"[^.]*inherit[^.]*\.", readme)]
licence_sentence = [m.group(0) for m in re.finditer(r"[^.]*" + re.escape(WORDING) + r"[^.]*\.", readme)]
add("README and results/gis_layers.json licences.note say 'the exported files are derived from sources whose terms are not recorded here' (no 'inherit' any more) and still assert no licence of their own",
    WORDING in readme and WORDING in rec["licences"]["note"] and not inh and "inherit" not in rec["licences"]["note"] and "assert no licence of their own" in rec["licences"]["note"] and "no licence of their own is asserted here" in readme,
    {"readme_sentence": licence_sentence, "json_note": rec["licences"]["note"], "inherit_sentences_left": inh})
res["licence_wording_sentence"] = licence_sentence
res["inherit_sentence"] = inh
res["licence_cells"] = lic_cells
# tags of every raster, text of the other files
tag_hits = {}
for p in sorted(GIS_DIR.glob("*.tif")):
    with rasterio.open(p) as s:
        allt = {ns or "default": s.tags(ns=ns) if ns else s.tags() for ns in [None] + list(s.tag_namespaces())}
    txt = json.dumps(allt)
    h = re.findall(r"licen[cs]e|copyright|public domain|creative commons|CC0|CC-BY|CC BY|terms of use|attribution", txt, re.I)
    if h:
        tag_hits[p.name] = h
add("no raster tag, GeoJSON body, CSV or GeoPackage table of the export carries a licence/copyright text", not tag_hits and not g2["sqlite"]["licence_scan_hits"] and all(not x for x in g2["licence_scan_text_files"].values()),
    {"raster_tag_hits": tag_hits, "gpkg_hits": g2["sqlite"]["licence_scan_hits"], "text_file_hits": g2["licence_scan_text_files"]})
rec_lic_ok = all(f["licence"].startswith("UNVERIFIED") or f["licence"] == "-" for f in rec["files"]) and all(v_.startswith("UNVERIFIED") for k_, v_ in rec["licences"].items() if k_ != "note")
add("results/gis_layers.json: every files[].licence and licences.* starts with UNVERIFIED (README row '-')", rec_lic_ok)

# ======================= disclosure (7) ============================================================================================================================================
LINE = "Claude Code wrote and ran this script under the team lead's direction; the team's own check of these layers is not done"
plain = readme.replace("**", "")
first_lines = "\n".join(readme.splitlines()[:8])
add("README carries the disclosure line (markdown emphasis removed), within its first lines", LINE in plain and LINE in first_lines.replace("**", ""), {"literal_in_raw_text": LINE in readme, "line_in_readme": next((l for l in readme.splitlines() if "Claude Code wrote and ran" in l), None)})
m_disc = re.fullmatch(r"Claude Code wrote and ran this script under the team lead's direction on (\d{4}-\d{2}-\d{2}); the team's own check of these layers is not done\.", rec["disclosure"])
add("results/gis_layers.json carries the disclosure (with a date inserted after 'direction')", bool(m_disc), {"disclosure": rec["disclosure"], "literal_line_in_it": LINE in rec["disclosure"]})
run_day = rec["written"][:10]
add("results/gis_layers.json disclosure names the day of the run that wrote these files (written " + rec["written"] + ")", bool(m_disc) and m_disc.group(1) == run_day,
    {"disclosure_date": m_disc.group(1) if m_disc else None, "written": rec["written"], "script_header_says": "Written 2026-10-29 ... Revised 2026-10-30 after the independent review"},
    "results/gis_layers.json disclosure (code/27_gis_layers.py, the J dict)")
res["readme_mentions_ai_review"] = bool(re.search(r"AI (agent|reviewer)|independent review|review_gis_layers", readme, re.I))
res["readme_mentions_team_check_not_done"] = "the team's own check of these layers is not done" in plain

# ---- the independent-review line and block the script writes from the review record (round 1 as it was when the script ran) ----------------------------------------------------
rv_bytes, rv_info = review_record_round1()
rv = json.loads(rv_bytes.decode("utf-8"))
ir, dl = rec.get("independent_review") or {}, rec.get("disclosure_independent_review")
want_line = f"An independent AI reviewer agent (Claude) recomputed these layers on {ir.get('date')} (results/review_gis_layers.json); the team's own check of these layers is not done."
add("README carries the independent-review line, word for word the one in results/gis_layers.json disclosure_independent_review, still ending with the team-check disclaimer",
    bool(ir) and dl == want_line and want_line in readme and readme.index(want_line) < readme.index("## What the layers are"), {"line": dl})
block_want = {"file": "results/review_gis_layers.json", "sha256": hashlib.sha256(rv_bytes).hexdigest(), "who": rv.get("who"), "date": (re.search(r"\d{4}-\d{2}-\d{2}", str((rv.get("dates") or {}).get("review") or rv.get("when"))) or [None])[0],
              "verdict": rv.get("verdict"), "export_run_reviewed": (rv.get("dates") or {}).get("export_run"), "must_fix": (rv.get("counts") or {}).get("must_fix"),
              "should_fix": (rv.get("counts") or {}).get("should_fix"), "human_review": rv.get("human_review")}
add("results/gis_layers.json independent_review quotes the review record faithfully (file, sha256 of round 1, who, date, verdict, export run reviewed, must/should-fix counts, human_review)", ir == block_want,
    {"block": ir, "from_the_record": block_want, "record_info": rv_info})
res["independent_review_block_is_round_1"] = {"block_verdict": ir.get("verdict"), "block_export_run_reviewed": ir.get("export_run_reviewed"), "export_written_now": rec["written"]}

# ======================= console summary ===========================================================================================================================================
jsave("g5_readme.json", res)
w = max(len(c["claim"]) for c in res["claims"])
for c in res["claims"]:
    print(("PASS " if c["ok"] else "FAIL ") + c["claim"][:230])
    if not c["ok"]:
        print("     ->", json.dumps(c["detail"], default=g_common._default)[:900])
print()
print("mismatches vs check_table:", len(mism["vs_check_table"]), "| vs my recomputation:", len(mism["vs_my_recomputation"]))
print("licence wording sentence:", licence_sentence, "| 'inherit' sentences left:", inh)
print("README mentions AI review:", res["readme_mentions_ai_review"])
