"""Checks 2 and part of 6: the GeoPackage (12 outlines, 12 gauges, EPSG:4269, every gauge inside its outline), the planar area of each outline recomputed in EPSG:5070 and
compared with the attribute area_km2, with results/basin_gis.json and with CAMELS area_gages2; the outlines against the source shapefile; the gauge positions against the
CAMELS table; the two GeoJSON twins against the GeoPackage; the GeoPackage's own tables; a scan of every vector/text file's metadata for a licence assertion."""
import json as _json
import sqlite3
from g_common import *
import geopandas as gpd
import pyogrio
import pyproj
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform

ids, audit = basins()
t, v = camels()
bg = jload("results/basin_gis.json")["basins"]
rec = jload(RECORD)
gp = scratch_copy(GIS_DIR / "basins_12.gpkg", "basins_12.gpkg")
res = {"layers": [], "per_basin": {}, "checks": {}}
LIC_RE = r"licen[cs]e|copyright|public domain|creative commons|CC0|CC-BY|CC BY|terms of use|attribution"


def as_multi(g):
    return g if g.geom_type == "MultiPolygon" else MultiPolygon([g])


# ---- the GeoPackage -------------------------------------------------------------------------------------------------------------------------------------------------
layers = [(str(a), str(b)) for a, b in pyogrio.list_layers(gp)]
res["layers"] = layers
out = gpd.read_file(gp, layer="outlines")
gau = gpd.read_file(gp, layer="gauges")
res["checks"]["layer_names_are_outlines_gauges"] = sorted(n for n, _ in layers) == ["gauges", "outlines"]
res["checks"]["n_outlines"] = len(out)
res["checks"]["n_gauges"] = len(gau)
res["checks"]["crs_outlines"] = out.crs.to_string() if out.crs else None
res["checks"]["crs_gauges"] = gau.crs.to_string() if gau.crs else None
res["checks"]["outline_ids_in_audit_order"] = list(out["gauge_id"]) == ids
res["checks"]["gauge_ids_in_audit_order"] = list(gau["gauge_id"]) == ids
res["checks"]["outline_names_equal_audit"] = [str(n) for n in out["name"]] == [audit[i]["gauge_name"] for i in ids]
res["checks"]["landscapes_equal_audit"] = [str(n) for n in out["landscape"]] == [audit[i]["landscape"] for i in ids]
res["checks"]["outline_validity"] = {i: bool(g.is_valid) for i, g in zip(out["gauge_id"], out.geometry)}
res["checks"]["outline_geom_types"] = sorted(set(out.geom_type))
res["checks"]["gauge_geom_types"] = sorted(set(gau.geom_type))
res["checks"]["outline_columns"] = [c for c in out.columns]
res["checks"]["gauge_columns"] = [c for c in gau.columns]

# ---- gpkg internals via sqlite (read-only, the scratch copy) ----------------------------------------------------------------------------------------------------------
con = sqlite3.connect(f"file:{gp.as_posix()}?mode=ro", uri=True)
cur = con.cursor()
sq = {}
sq["application_id"] = cur.execute("PRAGMA application_id").fetchone()[0]
sq["user_version"] = cur.execute("PRAGMA user_version").fetchone()[0]
sq["integrity_check"] = cur.execute("PRAGMA integrity_check").fetchone()[0]
sq["journal_mode"] = cur.execute("PRAGMA journal_mode").fetchone()[0]
tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")]
sq["tables"] = tabs
sq["gpkg_contents"] = [dict(zip(["table_name", "data_type", "identifier", "description", "last_change", "srs_id"], r))
                       for r in cur.execute("SELECT table_name, data_type, identifier, description, last_change, srs_id FROM gpkg_contents")]
sq["gpkg_geometry_columns"] = [dict(zip(["table_name", "column_name", "geometry_type_name", "srs_id"], r))
                               for r in cur.execute("SELECT table_name, column_name, geometry_type_name, srs_id FROM gpkg_geometry_columns")]
sq["gpkg_spatial_ref_sys"] = [dict(zip(["name", "srs_id", "organization", "organization_coordsys_id"], r))
                              for r in cur.execute("SELECT srs_name, srs_id, organization, organization_coordsys_id FROM gpkg_spatial_ref_sys")]
sq["has_rtree_for"] = [n for n in tabs if n.startswith("rtree_") and n.endswith("_geom")]
sq["has_gpkg_metadata_tables"] = [n for n in tabs if "metadata" in n]
lic_hits = []
for tn in tabs:
    if tn.startswith("rtree_") or tn in ("sqlite_sequence",):
        continue
    cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{tn}")')]
    for c in cols:
        if c in ("geom", "geometry"):
            continue
        try:
            for (val,) in cur.execute(f'SELECT "{c}" FROM "{tn}"'):
                if isinstance(val, str) and re.search(LIC_RE, val, re.I):
                    lic_hits.append({"table": tn, "column": c, "value": val[:200]})
        except Exception as e:  # pragma: no cover
            lic_hits.append({"table": tn, "column": c, "error": str(e)})
sq["licence_scan_hits"] = lic_hits
con.close()
res["sqlite"] = sq

# ---- planar areas in EPSG:5070 (own transformer), geodesic area on GRS80, attributes vs sources -------------------------------------------------------------------
tr = pyproj.Transformer.from_crs(4269, 5070, always_xy=True)
geod = pyproj.Geod(ellps="GRS80")


def geodesic_area_km2(geom):
    """GRS80 ellipsoidal area of a (multi)polygon: |exterior| minus |holes| per polygon, ring by ring (orientation-proof)."""
    tot = 0.0
    for p in as_multi(geom).geoms:
        xs, ys = p.exterior.xy
        a = abs(geod.polygon_area_perimeter(list(xs), list(ys))[0])
        for r in p.interiors:
            xs, ys = r.xy
            a -= abs(geod.polygon_area_perimeter(list(xs), list(ys))[0])
        tot += a
    return tot / 1e6


src = gpd.read_file(PROJ / "data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp", where="hru_id IN (" + ",".join(str(int(i)) for i in ids) + ")")
src["gid"] = src["hru_id"].astype(int).astype(str).str.zfill(8)
src = src.set_index("gid")
res["checks"]["source_shapefile_crs"] = src.crs.to_string()
res["checks"]["source_rows_found"] = len(src)

for i, row in out.set_index("gauge_id").loc[ids].iterrows():
    geom = row.geometry
    g5070 = transform(tr.transform, geom)
    a_planar = g5070.area / 1e6
    a_geod = geodesic_area_km2(geom)
    sg = src.loc[i, "geometry"]
    s_valid = bool(sg.is_valid)
    sg_fix = sg if s_valid else sg.buffer(0)
    sym = shapely.symmetric_difference(geom, sg_fix).area
    b = bg[i]
    c_area = float(t.loc[i, "area_gages2"])
    c_gf = float(t.loc[i, "area_geospa_fabric"])
    c_elev = float(t.loc[i, "elev_mean"])
    c_frac = float(v.loc[i, "frac_forest"])
    pb = {
        "short": row["short"],
        "n_parts": int(len(geom.geoms)),
        "n_parts_attr": int(row["n_parts"]),
        "n_parts_source": int(len(as_multi(sg).geoms)),
        "is_valid": bool(geom.is_valid),
        "source_valid": s_valid,
        "geometry_vs_source_symdiff_deg2": float(sym),
        "geometry_vs_source_rel": float(sym / sg_fix.area),
        "geometry_exact_equal_to_source": bool(shapely.equals_exact(as_multi(geom), as_multi(sg), 0)),
        "n_vertices_gpkg": int(shapely.get_num_coordinates(geom)),
        "n_vertices_source": int(shapely.get_num_coordinates(sg)),
        "area_planar_5070_km2_mine": a_planar,
        "area_geodesic_grs80_km2_mine": a_geod,
        "area_km2_attr": float(row["area_km2"]),
        "area_geodesic_km2_attr": float(row["area_geodesic_km2"]),
        "area_albers_km2_basin_gis": float(b["area_albers_km2"]),
        "area_geodesic_km2_basin_gis": float(b["area_km2"]),
        "outline_file_area_km2_dbf": float(src.loc[i, "AREA"]) / 1e6,          # the DBF AREA field is in m2
        "outline_file_area_km2_attr": float(row["outline_file_area_km2"]),
        "area_mine_minus_dbf_km2": a_planar - float(src.loc[i, "AREA"]) / 1e6,
        "camels_area_gages2": c_area,
        "camels_area_gages2_attr": float(row["camels_area_gages2_km2"]),
        "camels_area_gf": c_gf,
        "camels_area_gf_attr": float(row["camels_area_geospa_fabric_km2"]),
        "area_vs_gages2_pct_mine": (a_planar - c_area) / c_area * 100.0,
        "area_vs_gages2_pct_attr": float(row["area_diff_gages2_pct"]),
        "area_vs_gf_pct_mine": (a_planar - c_gf) / c_gf * 100.0,
        "area_attr_minus_mine_km2": float(row["area_km2"]) - a_planar,
        "area_attr_minus_basin_gis_km2": float(row["area_km2"]) - float(b["area_albers_km2"]),
        "area_geod_attr_minus_mine_km2": float(row["area_geodesic_km2"]) - a_geod,
        "elev_mean_camels": c_elev,
        "elev_mean_camels_attr": float(row["camels_elev_mean_m"]),
        "outline_file_elev_mean_dbf": float(src.loc[i, "elev_mean"]),
        "outline_file_elev_mean_attr": float(row["outline_file_elev_mean_m"]),
        "frac_forest_camels": c_frac,
        "frac_forest_camels_attr": float(row["camels_frac_forest"]),
        "dom_land_cover_camels": str(v.loc[i, "dom_land_cover"]),
        "dom_land_cover_camels_attr": str(row["camels_dom_land_cover"]).strip(),
        "dom_land_cover_frac_camels": float(v.loc[i, "dom_land_cover_frac"]),
        "dom_land_cover_frac_camels_attr": float(row["camels_dom_land_cover_frac"]),
        "bounds_deg": [float(x) for x in geom.bounds],
    }
    res["per_basin"][i] = pb


# ---- the gauges: positions vs the CAMELS table (5 decimals), inside the outline, distance to the outline edge ---------------------------------------------------------------
def dist_to_vertex_km(poly, lon, lat):
    """Distance (km) to the nearest outline VERTEX (same local projection), to test what the attribute distance_to_outline_km measures."""
    crs = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")
    tf = pyproj.Transformer.from_crs(4269, crs, always_xy=True)
    xy = shapely.get_coordinates(poly)
    X, Y = tf.transform(xy[:, 0], xy[:, 1])
    return float(np.sqrt(np.asarray(X) ** 2 + np.asarray(Y) ** 2).min()) / 1000.0


def dist_nearest_in_degrees_km(poly, lon, lat):
    """What the attribute turns out to be: the nearest boundary point chosen in degree space (shapely.ops.nearest_points), then the GRS80 geodesic distance to it.
    Degrees are not isotropic (1 deg of longitude is 0.8 of 1 deg of latitude at 35 N), so this can be slightly longer than the true minimum."""
    from shapely.ops import nearest_points
    q = Point(lon, lat)
    near = nearest_points(poly.boundary, q)[0]
    return geod.inv(near.x, near.y, lon, lat)[2] / 1000.0


def dist_to_edge_km(poly, lon, lat):
    """Distance (km) from (lon, lat) to the boundary of the polygon, in a local azimuthal equidistant projection on GRS80 centred on the point."""
    crs = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")
    tf = pyproj.Transformer.from_crs(4269, crs, always_xy=True)
    pl = transform(tf.transform, poly)
    return Point(0, 0).distance(pl.boundary) / 1000.0


outline_by_id = out.set_index("gauge_id").geometry
for i, row in gau.set_index("gauge_id").loc[ids].iterrows():
    poly = outline_by_id[i]
    lon_f, lat_f = row.geometry.x, row.geometry.y
    lat_c, lon_c = float(t.loc[i, "gauge_lat"]), float(t.loc[i, "gauge_lon"])
    pt_c = Point(lon_c, lat_c)
    _, _, dist_m = geod.inv(lon_f, lat_f, lon_c, lat_c)
    pb = res["per_basin"][i]
    pb.update({
        "gauge_lat_file": float(lat_f), "gauge_lon_file": float(lon_f),
        "gauge_lat_attr": float(row["lat"]), "gauge_lon_attr": float(row["lon"]),
        "gauge_lat_camels_topo": lat_c, "gauge_lon_camels_topo": lon_c,
        "gauge_lat_data_audit": float(audit[i]["gauge_lat"]), "gauge_lon_data_audit": float(audit[i]["gauge_lon"]),
        "gauge_file_equals_data_audit_rounded": bool(abs(lat_f - audit[i]["gauge_lat"]) < 1e-9 and abs(lon_f - audit[i]["gauge_lon"]) < 1e-9),
        "gauge_file_equals_camels_topo": bool(abs(lat_f - lat_c) < 1e-9 and abs(lon_f - lon_c) < 1e-9),
        "gauge_displacement_file_vs_camels_m": float(dist_m),
        "gauge_inside_file_point": bool(poly.contains(row.geometry)),
        "gauge_inside_attr": bool(row["inside_outline"]),
        "gauge_inside_camels_point": bool(poly.contains(pt_c)),
        "gauge_dist_edge_km_file_point_mine": dist_to_edge_km(poly, lon_f, lat_f),
        "gauge_dist_edge_km_camels_point_mine": dist_to_edge_km(poly, lon_c, lat_c),
        "gauge_dist_vertex_km_file_point_mine": dist_to_vertex_km(poly, lon_f, lat_f),
        "gauge_dist_vertex_km_camels_point_mine": dist_to_vertex_km(poly, lon_c, lat_c),
        "gauge_dist_nearest_in_degrees_km_file_point": dist_nearest_in_degrees_km(poly, lon_f, lat_f),
        "gauge_dist_edge_km_attr": float(row["distance_to_outline_km"]),
        "gauge_dist_edge_km_basin_gis": float(bg[i]["gauge"]["distance_to_outline_km"]),
        "gauge_elev_attr": float(row["elev_at_gauge_m"]),
        "gauge_elev_basin_gis": float(bg[i]["gauge"]["elev_m"]),
        "gauge_source_attr": str(row["source"]),
        "gauge_lat_basin_gis": float(bg[i]["gauge"]["lat"]), "gauge_lon_basin_gis": float(bg[i]["gauge"]["lon"]),
        "gauge_inside_basin_gis": bool(bg[i]["gauge"]["inside_outline"]),
    })


# ---- the GeoJSON twins vs the GeoPackage -------------------------------------------------------------------------------------------------------------------------------
def strict_load(p):
    def bad(x):
        raise ValueError("non-finite number token " + x)
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    return _json.loads(txt, parse_constant=bad), txt


def props_mismatch(feat_props, row):
    bad = []
    for k, vv in feat_props.items():
        ov = row[k]
        if isinstance(vv, (int, float)) and not isinstance(vv, bool):
            if abs(float(vv) - float(ov)) > 1e-9 * max(1.0, abs(float(ov))):
                bad.append((k, vv, float(ov)))
        elif str(vv) != str(ov):
            bad.append((k, str(vv)[:40], str(ov)[:40]))
    return bad


gj, feats_of = {}, {}
for name in ("basins_12.geojson", "gauges_12.geojson"):
    try:
        d, txt = strict_load(GIS_DIR / name)
    except Exception as e:
        gj[name] = {"strict_json": False, "error": str(e)}
        continue
    feats = d["features"]
    feats_of[name] = feats
    allxy = np.concatenate([shapely.get_coordinates(shape(f["geometry"])) for f in feats])
    gj[name] = {"strict_json": True, "top_keys": list(d.keys()), "type": d.get("type"), "crs_member": d.get("crs"), "n_features": len(feats),
                "coordinates_are_multiples_of_1e-7": bool(np.all(np.abs(allxy * 1e7 - np.round(allxy * 1e7)) < 1e-4)),
                "n_coordinates": int(len(allxy)), "geometry_types": sorted({f["geometry"]["type"] for f in feats})}

feats = feats_of["basins_12.geojson"]
fmap = {f["properties"]["gauge_id"]: f for f in feats}
orow_all = out.set_index("gauge_id", drop=False)
max_cd, attr_mis = 0.0, []
for i in ids:
    g = shape(fmap[i]["geometry"])
    og = outline_by_id[i]
    ca, cb = shapely.get_coordinates(g), shapely.get_coordinates(og)
    same_shape = ca.shape == cb.shape
    cd = float(np.abs(ca - cb).max()) if same_shape else float("nan")
    max_cd = max(max_cd, cd) if same_shape else float("nan")
    res["per_basin"][i]["geojson_vertex_count_equal"] = bool(same_shape)
    res["per_basin"][i]["geojson_max_coord_diff_deg"] = cd
    res["per_basin"][i]["geojson_symdiff_rel"] = float(shapely.symmetric_difference(g, og).area / og.area)
    mis = props_mismatch(fmap[i]["properties"], orow_all.loc[i])
    attr_mis += [(i, *m) for m in mis]
gj["basins_12.geojson"]["max_vertex_diff_vs_gpkg_deg"] = max_cd
gj["basins_12.geojson"]["attribute_mismatches_vs_gpkg"] = attr_mis
gj["basins_12.geojson"]["ids_in_audit_order"] = [f["properties"]["gauge_id"] for f in feats] == ids
gj["basins_12.geojson"]["property_keys_equal_gpkg_columns"] = sorted(feats[0]["properties"].keys()) == sorted(c for c in out.columns if c != "geometry")


def ring_ccw(ring):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        s += x1 * y2 - x2 * y1
    return s > 0


orient = []
for f in feats:
    polys = f["geometry"]["coordinates"] if f["geometry"]["type"] == "MultiPolygon" else [f["geometry"]["coordinates"]]
    orient += [ring_ccw(p[0]) for p in polys]
gj["basins_12.geojson"]["exterior_rings_ccw_share"] = float(np.mean(orient))

gf = feats_of["gauges_12.geojson"]
gmap = {f["properties"]["gauge_id"]: f for f in gf}
grow_all = gau.set_index("gauge_id", drop=False)
gd, gattr = 0.0, []
for i in ids:
    x, y = gmap[i]["geometry"]["coordinates"][:2]
    og = grow_all.loc[i]
    gd = max(gd, abs(x - og.geometry.x), abs(y - og.geometry.y))
    gattr += [(i, *m) for m in props_mismatch(gmap[i]["properties"], og)]
gj["gauges_12.geojson"]["max_coord_diff_vs_gpkg_deg"] = gd
gj["gauges_12.geojson"]["attribute_mismatches_vs_gpkg"] = gattr
gj["gauges_12.geojson"]["ids_in_audit_order"] = [f["properties"]["gauge_id"] for f in gf] == ids
res["geojson"] = gj

# ---- licence-assertion scan of the other text files (CSV legend, the GeoJSON bodies) ----------------------------------------------------------------------------------
scan = {}
for name in ("landcover_classes.csv", "basins_12.geojson", "gauges_12.geojson"):
    txt = (GIS_DIR / name).read_text(encoding="utf-8", errors="replace")
    scan[name] = [m.group(0) for m in re.finditer(LIC_RE, txt, re.I)]
res["licence_scan_text_files"] = scan

jsave("g2_vectors.json", res)

# ---- console summary ---------------------------------------------------------------------------------------------------------------------------------------------------
c = res["checks"]
print("layers", layers)
print({k: c[k] for k in ("layer_names_are_outlines_gauges", "n_outlines", "n_gauges", "crs_outlines", "crs_gauges", "outline_ids_in_audit_order", "gauge_ids_in_audit_order",
                           "outline_names_equal_audit", "landscapes_equal_audit", "outline_geom_types", "gauge_geom_types", "source_shapefile_crs", "source_rows_found")})
print("valid outlines:", all(c["outline_validity"].values()))
print("sqlite:", {k: sq[k] for k in ("application_id", "user_version", "integrity_check", "journal_mode", "has_rtree_for", "has_gpkg_metadata_tables")})
print("gpkg_contents:", sq["gpkg_contents"])
print("licence hits in gpkg tables:", sq["licence_scan_hits"])
print("licence hits in text files:", scan)
print()
print(f"{'id':9s} {'planar(mine)':>13s} {'attr':>13s} {'basin_gis':>13s} {'geod(mine)':>13s} {'geod(attr)':>13s} {'gages2':>9s} {'diff%':>7s} {'dbf AREA':>11s} {'symdiff':>9s}")
for i in ids:
    p = res["per_basin"][i]
    print(f"{i:9s} {p['area_planar_5070_km2_mine']:13.6f} {p['area_km2_attr']:13.6f} {p['area_albers_km2_basin_gis']:13.6f} {p['area_geodesic_grs80_km2_mine']:13.6f} "
          f"{p['area_geodesic_km2_attr']:13.6f} {p['camels_area_gages2']:9.2f} {p['area_vs_gages2_pct_mine']:7.2f} {p['outline_file_area_km2_dbf']:11.6f} {p['geometry_vs_source_rel']:9.1e}")
print("max |attr - mine| planar:", max(abs(p["area_attr_minus_mine_km2"]) for p in res["per_basin"].values()))
print("max |attr - basin_gis| planar:", max(abs(p["area_attr_minus_basin_gis_km2"]) for p in res["per_basin"].values()))
print("max |mine - DBF AREA| km2:", max(abs(p["area_mine_minus_dbf_km2"]) for p in res["per_basin"].values()))
print("max |attr distance - nearest-in-degrees distance| km:", max(abs(p["gauge_dist_edge_km_attr"] - p["gauge_dist_nearest_in_degrees_km_file_point"]) for p in res["per_basin"].values()),
      "| max (attr - true edge distance) km:", max(p["gauge_dist_edge_km_attr"] - p["gauge_dist_edge_km_file_point_mine"] for p in res["per_basin"].values()))
print("n_parts attr == mine:", all(p["n_parts"] == p["n_parts_attr"] for p in res["per_basin"].values()), "| n_parts vs source:", {i: (p["n_parts"], p["n_parts_source"]) for i, p in res["per_basin"].items() if p["n_parts"] != p["n_parts_source"]})
print("max |geod attr - mine|:", max(abs(p["area_geod_attr_minus_mine_km2"]) for p in res["per_basin"].values()))
print("source invalid:", [i for i in ids if not res["per_basin"][i]["source_valid"]], "| gpkg exactly equal to source geometry:", [i for i in ids if res["per_basin"][i]["geometry_exact_equal_to_source"]])
print()
print(f"{'id':9s} {'lat file':>9s} {'lon file':>10s} {'lat CAMELS':>11s} {'lon CAMELS':>11s} {'shift m':>8s} {'in(file)':>8s} {'in(CAMELS)':>10s} {'d edge file':>11s} {'d edge CAMELS':>13s} {'d attr':>7s} {'d bgis':>7s}")
for i in ids:
    p = res["per_basin"][i]
    print(f"{i:9s} {p['gauge_lat_file']:9.5f} {p['gauge_lon_file']:10.5f} {p['gauge_lat_camels_topo']:11.5f} {p['gauge_lon_camels_topo']:11.5f} {p['gauge_displacement_file_vs_camels_m']:8.1f} "
          f"{str(p['gauge_inside_file_point']):>8s} {str(p['gauge_inside_camels_point']):>10s} {p['gauge_dist_edge_km_file_point_mine']:11.3f} {p['gauge_dist_edge_km_camels_point_mine']:13.3f} "
          f"{p['gauge_dist_edge_km_attr']:7.3f} {p['gauge_dist_edge_km_basin_gis']:7.3f}  vertex(file) {p['gauge_dist_vertex_km_file_point_mine']:.3f} vertex(CAMELS) {p['gauge_dist_vertex_km_camels_point_mine']:.3f}")
print()
print("geojson:", {n: {k: vv for k, vv in d.items() if k not in ("attribute_mismatches_vs_gpkg",)} for n, d in gj.items()})
print("geojson attribute mismatches:", {n: d.get("attribute_mismatches_vs_gpkg") for n, d in gj.items()})
