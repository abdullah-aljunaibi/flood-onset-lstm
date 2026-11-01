"""D3: the 12 study basins as proper GIS layers (outlines, elevation, land cover), exported and checked against the CAMELS tables.

The mid-term promised "12 of 12 basins as GIS layers: outlines, elevation and land cover exported and checked against the CAMELS tables".
Until now the project held only maps and statistics (code/07_basin_maps.py -> results/basin_gis.json, figures/basin_maps.png). This script
exports the layers themselves from the same sources, with the same clipping rule as script 07 (a raster cell belongs to a basin when its
centre lies inside the outline), so that the statistics read back from the exported files are the ones of results/basin_gis.json.
Nothing here is fitted or tuned; the train/validation/test split is not involved. It writes, under results/gis_layers/:
  basins_12.gpkg           GeoPackage, EPSG:4269 (geographic NAD83, the CRS of the source shapefile). Layer "outlines": the 12 basin polygons with
                           gauge_id, name, landscape, area_km2 (planar area in EPSG:5070), the CAMELS area_gages2, the mean elevation of the exported
                           elevation layer, the dominant land-cover class and group, the forest share and the CAMELS values beside them.
                           Layer "gauges": the 12 gauge points (gauge_lat/gauge_lon of the CAMELS attribute table).
  basins_12.geojson        the outlines layer again in EPSG:4326 (WGS 84) for web tools;  gauges_12.geojson: the gauges likewise.
  elevation_<gauge_id>.tif  1-arc-second elevation (whole metres, int16) cut from the data/dem tiles, on the tiles' own grid and CRS (WGS 84, heights
                           above the EGM96 geoid), clipped to the outline buffered by BUFFER_ARCSEC arc-seconds; nodata -32768; LZW.
  landcover_<gauge_id>.tif  the Annual NLCD 2011 class codes (uint8) of data/landcover, clipped to the outline on the native 30 m grid in EPSG:5070
                           (NAD83 / Conus Albers); nodata 250 (as the source); LZW; the NLCD colour table kept.
  landcover_classes.csv    the class legend: code, name, group (as in results/basin_gis.json), colour.
  README.md                every file with its CRS, resolution, units, source and licence (UNVERIFIED where the project files record none), and the
                           check table: exported area, mean elevation and dominant land cover against results/basin_gis.json and the CAMELS attributes.
and results/gis_layers.json: the same tables, the verification, and the sha256 of every file written (the GeoPackage stores its write time, so its hash
changes with every run). Reads results/data_audit.json (the 12 basins, in the project's order), data/camels_us/camels_topo.txt and camels_vege.txt (the
attribute table), results/basin_gis.json (the statistics of script 07 and its notes), results/data_flow.json (the source records for the README),
data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp, data/dem/*.hgt.gz and data/landcover/annual_nlcd_2011_<gauge_id>.tif; and, if it exists,
results/review_gis_layers.json (the independent review of this export: only its date and verdict are read, for the disclosure line). Nothing is downloaded:
the script stops if a tile or land-cover file is missing (script 07 fetches them). Rerunnable: it overwrites only the files it writes itself.
Run from the project root with a Python that has rasterio, shapely, pyproj and geopandas (on the Windows PC the system Python 3.13, NOT the project venv:
    python -X utf8 code/27_gis_layers.py
flood.py is not imported, so torch is not needed). FLOOD_QUICK=1 writes to results/quick/gis_layers and results/quick/gis_layers.json instead.
Written 2026-10-29 by Claude Code under the team lead's direction (HANDOFF.md section 4 step 3); the team's own check of these layers is NOT done.
Revised 2026-10-30 after the independent review (results/review_gis_layers.json): the gauge points are the unrounded positions of camels_topo.txt (data_audit.json
holds them rounded to 0.001 degree), the README tables are checked for their cell counts, the citation sentence follows readme.txt, and the review is disclosed.
"""
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point
from shapely.ops import nearest_points

T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]                                     # the folder above code/, as in code/common.py
DATA, RES = ROOT / "data", ROOT / "results"
QUICK = os.environ.get("FLOOD_QUICK", "").lower() in ("1", "true", "yes")
RES_OUT = RES / "quick" if QUICK else RES
OUT = RES_OUT / "gis_layers"
JSON_OUT = RES_OUT / "gis_layers.json"
SHP = DATA / "camels_us" / "basin_set_full_res" / "HCDN_nhru_final_671.shp"
CAM = DATA / "camels_us"
DEM_DIR, LC_DIR = DATA / "dem", DATA / "landcover"
LC_YEAR = 2011
CRS_OUTLINES, CRS_DEM, CRS_LC, CRS_WEB = "EPSG:4269", "EPSG:4326", "EPSG:5070", "EPSG:4326"
BUFFER_ARCSEC = 15                                                             # the elevation layer reaches this far beyond the outline (about 460 m north-south, 370-410 m east-west)
DEM_NODATA, LC_NODATA = -32768, 250                                            # the .hgt void code; the nodata of the cached NLCD files
GEOD = pyproj.Geod(ellps="GRS80")                                              # the ellipsoid of NAD83 (geodesic areas and distances)
NLCD = {11: "open water", 21: "developed, open space", 22: "developed, low intensity", 23: "developed, medium intensity",
        24: "developed, high intensity", 31: "barren land", 41: "deciduous forest", 42: "evergreen forest", 43: "mixed forest",
        52: "shrub/scrub", 71: "grassland/herbaceous", 81: "pasture/hay", 82: "cultivated crops", 90: "woody wetlands",
        95: "emergent herbaceous wetlands"}                                     # as in code/07_basin_maps.py and results/basin_gis.json
GROUPS = {"forest": (41, 42, 43), "shrub and grass": (52, 71), "cropland and pasture": (81, 82), "wetland": (90, 95),
          "developed": (21, 22, 23, 24), "water": (11,), "bare ground": (31,)}
GROUP_OF = {c: k for k, codes in GROUPS.items() for c in codes}
TOL = dict(area_km2=0.001, elev_m=0.01, share=1e-6, distance_km=1e-6)                           # exported layer vs results/basin_gis.json: the same cells, so near-exact
OWN_FILES = ["basins_12.gpkg", "basins_12.geojson", "gauges_12.geojson", "landcover_classes.csv", "README.md"]
say = lambda *a: print(*a, flush=True)
rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()


def sha256(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_row(*cells):
    """One row of a Markdown table. A word that holds a <placeholder> would be read as an HTML tag, so it goes into backticks (its <A|B> choices written <A or B>);
    any other pipe would end the cell and is written with a backslash in front of it; a line break would end the row."""
    def tidy(text):
        text = re.sub(r"([^\s`]*<[^<>\s`]+>[^\s`]*?)([.,;:)]*)(?=\s|$)", lambda m: "`" + m.group(1).replace("|", " or ") + "`" + m.group(2), str(text).replace("\n", " "))
        return text.replace("|", "\\|")
    return "| " + " | ".join(tidy(c) for c in cells) + " |"


def table_problems(text):
    """The Markdown tables of `text`: (number of tables, number of data rows, problems). A row whose number of cells differs from its header's is a problem;
    a cell ends at every pipe that has no backslash in front of it."""
    tables = rows = 0
    problems, width, start = [], None, 0
    for n, line in enumerate(text.split("\n"), 1):
        if not line.startswith("|"):
            width = None
            continue
        cells = len(re.findall(r"(?<!\\)\|", line)) - 1
        if width is None:
            width, start, tables = cells, n, tables + 1
            continue
        rows += 0 if set(line) <= set("|-: ") else 1                           # the line of dashes under the header is not a data row
        if cells != width:
            problems.append(f"line {n}: {cells} cells, the header (line {start}) has {width}")
    return tables, rows, problems


# ---- inputs ----------------------------------------------------------------------------------------------------------
for f in (RES / "data_audit.json", RES / "basin_gis.json", RES / "data_flow.json", SHP, CAM / "camels_topo.txt", CAM / "camels_vege.txt"):
    assert f.exists(), f"missing input {rel(f)}"
AUDIT = json.loads((RES / "data_audit.json").read_text(encoding="utf-8"))
GIS = json.loads((RES / "basin_gis.json").read_text(encoding="utf-8"))
FLOW = json.loads((RES / "data_flow.json").read_text(encoding="utf-8"))
B = pd.DataFrame(AUDIT["selected"]).set_index("gauge_id")                      # the 12 basins in the project's order (01_data_audit.py)
assert len(B) == 12, f"{len(B)} basins selected, expected 12"
topo = pd.read_csv(CAM / "camels_topo.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id").loc[B.index]
vege = pd.read_csv(CAM / "camels_vege.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id").loc[B.index]
for col in ("gauge_lat", "gauge_lon", "elev_mean", "area_gages2"):           # data_audit.json carries the attribute table's values, the gauge positions rounded to 0.001 degree: prove it
    assert np.allclose(B[col].astype(float), topo[col].astype(float), rtol=0, atol=5.1e-4), f"data_audit.json is not camels_topo.txt rounded to 3 decimals ({col})"
GAUGE_SHIFT_M = [GEOD.inv(float(B.gauge_lon.loc[g]), float(B.gauge_lat.loc[g]), float(topo.gauge_lon.loc[g]), float(topo.gauge_lat.loc[g]))[2] for g in B.index]   # rounded copy vs table (m)
B["gauge_lat"], B["gauge_lon"] = topo.gauge_lat.astype(float), topo.gauge_lon.astype(float)      # the gauge points use the table's own positions, not the 3-decimal copy of data_audit.json
B["area_geospa_fabric"], B["slope_mean"] = topo.area_geospa_fabric.astype(float), topo.slope_mean.astype(float)
B["frac_forest"], B["dom_land_cover"], B["dom_land_cover_frac"] = vege.frac_forest.astype(float), vege.dom_land_cover.astype(str).str.strip(), vege.dom_land_cover_frac.astype(float)
B["short"] = [GIS["basins"][g]["short"] for g in B.index]
missing = [f for g in B.index for f in (LC_DIR / f"annual_nlcd_{LC_YEAR}_{g}.tif",) if not f.exists()]
assert not missing, "missing land-cover files (run code/07_basin_maps.py to fetch them): " + ", ".join(rel(f) for f in missing)

GAUGE_SRC = f"{rel(CAM / 'camels_topo.txt')} gauge_lat, gauge_lon, the table's own values (not rounded)"
REVIEW = None                                                                  # the independent review of this export (another agent's record): read for the disclosure line, never written
if (RES / "review_gis_layers.json").exists():
    _all = json.loads((RES / "review_gis_layers.json").read_text(encoding="utf-8"))
    _rounds = sorted((k for k in _all if re.fullmatch(r"round_\d+", k)), key=lambda k: int(k.split("_")[1]))
    _rv = _all[_rounds[-1]] if _rounds else _all                                # the LATEST round of the review (round 1 is the top level of the record)
    _day = re.search(r"\d{4}-\d{2}-\d{2}", str((_rv.get("dates") or {}).get("review") or _rv.get("when") or ""))
    assert _day and _rv.get("verdict"), "results/review_gis_layers.json holds no review date or verdict"
    REVIEW = dict(file=rel(RES / "review_gis_layers.json"), round=_rounds[-1] if _rounds else "round_1", rounds=1 + len(_rounds),
                  record_sha256_when_this_run_read_it=sha256(RES / "review_gis_layers.json"), who=_rv.get("who") or _all.get("who"), date=_day.group(0), verdict=_rv["verdict"],
                  export_run_reviewed=(_rv.get("dates") or {}).get("export_run"), must_fix=(_rv.get("counts") or {}).get("must_fix"),
                  should_fix=(_rv.get("counts") or {}).get("should_fix"), human_review=_rv.get("human_review") or _all.get("human_review"))
REVIEW_LINE = (f"An independent AI reviewer agent (Claude) recomputed these layers on {REVIEW['date']} ({REVIEW['file']}); the team's own check of these layers is not done." if REVIEW else "")

# ---- outlines --------------------------------------------------------------------------------------------------------
shp = gpd.read_file(SHP, where="hru_id IN (%s)" % ",".join(str(int(g)) for g in B.index))
assert shp.crs is not None and shp.crs.to_epsg() == 4269, f"the outlines are expected in EPSG:4269 (geographic NAD83), found {shp.crs}"
shp["gauge_id"] = shp.hru_id.astype(int).astype(str).str.zfill(8)
shp = shp.set_index("gauge_id").loc[B.index]
assert len(shp) == 12 and shp.index.is_unique, "every study basin must have exactly one outline"
repaired = [g for g in B.index if not shp.geometry[g].is_valid]
shp["geometry"] = [geom if geom.is_valid else geom.buffer(0) for geom in shp.geometry]      # the repair of 07_basin_maps.py
assert shp.is_valid.all(), "an outline is still invalid after buffer(0)"
OUTLINE_FILE = dict(area_km2=shp.AREA.astype(float) / 1e6, elev_mean_m=shp.elev_mean.astype(float))
GEOM = dict(shp.geometry)                                                      # gauge_id -> polygon in EPSG:4269
GEOM_LC = dict(shp.to_crs(CRS_LC).geometry)                                   # the same in EPSG:5070 (equal-area, metres)
_tg = pyproj.transformer.TransformerGroup(CRS_OUTLINES, CRS_WEB, always_xy=True)       # the NAD83 -> WGS 84 step geopandas.to_crs applies to the GeoJSON twins
DATUM_STEP = _tg.transformers[0].description + (" = a null transformation, coordinates unchanged" if _tg.transformers[0].to_proj4().strip() == "+proj=noop" else "") + \
    (f"; {len(_tg.unavailable_operations)} grid-shift operations are not installed" if _tg.unavailable_operations else "")


# ---- elevation tiles -------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def dem_tile(lat, lon):
    """One 1 x 1 degree tile named by its south-west corner: 3601 x 3601 big-endian int16, north-up, cell centres on whole arc-seconds."""
    name = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    f = DEM_DIR / f"{name}.hgt.gz"
    assert f.exists(), f"missing elevation tile {rel(f)} (run code/07_basin_maps.py to fetch it)"
    z = np.frombuffer(gzip.decompress(f.read_bytes()), ">i2")
    assert z.size == 3601 * 3601, f"{f.name} is not a 1-arc-second tile"
    return name, z.reshape(3601, 3601)


def dem_window(bounds):
    """The source grid cells whose centres lie within lon/lat bounds (int16, voids = DEM_NODATA), rows north to south; the affine transform; the tiles."""
    c0, c1 = int(np.floor(bounds[0] * 3600)), int(np.ceil(bounds[2] * 3600))   # arc-seconds east (cell centres)
    a0, a1 = int(np.floor(bounds[1] * 3600)), int(np.ceil(bounds[3] * 3600))   # arc-seconds north
    Z, names = np.full((a1 - a0 + 1, c1 - c0 + 1), DEM_NODATA, "int16"), []
    for la in range(a0 // 3600, (a1 - 1) // 3600 + 1):
        for lo in range(c0 // 3600, (c1 - 1) // 3600 + 1):
            name, t = dem_tile(la, lo)
            names.append(name)
            lo_a, hi_a, lo_c, hi_c = max(a0, la * 3600), min(a1, la * 3600 + 3600), max(c0, lo * 3600), min(c1, lo * 3600 + 3600)
            Z[a1 - hi_a:a1 - lo_a + 1, lo_c - c0:hi_c - c0 + 1] = t[la * 3600 + 3600 - hi_a:la * 3600 + 3600 - lo_a + 1, lo_c - lo * 3600:hi_c - lo * 3600 + 1]
    return Z, Affine(1 / 3600, 0, (c0 - 0.5) / 3600, 0, -1 / 3600, (a1 + 0.5) / 3600), sorted(set(names))


def cell_size_m(lat, step=1):
    """East-west and north-south length (m) of `step` arc-seconds at latitude lat (degrees) on the GRS80 ellipsoid."""
    p, d = np.radians(lat), np.radians(step / 3600)
    w = np.sqrt(1 - GEOD.es * np.sin(p) ** 2)
    return float(GEOD.a * np.cos(p) / w * d), float(GEOD.a * (1 - GEOD.es) / w ** 3 * d)


# ---- output folder (only this script's files are removed) ---------------------------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
own = OWN_FILES + [f"{kind}_{g}.tif" for kind in ("elevation", "landcover") for g in B.index]
for f in own:
    (OUT / f).unlink(missing_ok=True)
strays = sorted(p.name for p in OUT.iterdir() if p.name not in own and not p.name.startswith("."))
if strays:
    say(f"note: {len(strays)} file(s) in {rel(OUT)} were not written by this script and are left alone: {', '.join(strays)}")

COMMON_TAGS = dict(PROJECT="Project-Flood (AI course, 2026): 12 CAMELS-US basins", CREATED_BY="code/27_gis_layers.py",
                   CLIPPING="a cell belongs to the basin when its centre lies inside the outline (rasterio geometry_mask, all_touched=False), as in code/07_basin_maps.py")

# ---- per basin: elevation and land-cover layers -------------------------------------------------------------------------
R = {}
for g in B.index:
    b, geom, geom_lc, gis = B.loc[g], GEOM[g], GEOM_LC[g], GIS["basins"][g]
    # elevation: the tiles' own grid, clipped to the buffered outline
    buffered = geom.buffer(BUFFER_ARCSEC / 3600)
    Z, tf, tiles = dem_window(buffered.bounds)
    keep = geometry_mask([buffered], Z.shape, tf, invert=True)
    inside = geometry_mask([geom], Z.shape, tf, invert=True)
    Z = np.where(keep, Z, DEM_NODATA).astype("int16")
    valid = Z != DEM_NODATA
    lats = tf.f + tf.e * (np.arange(Z.shape[0]) + 0.5)                         # latitude of each row's cell centres
    w = np.repeat(np.cos(np.radians(lats))[:, None], Z.shape[1], 1)
    ok = inside & valid
    zin = Z[ok].astype("float64")
    elev_mean = float(np.average(zin, weights=w[ok]))
    row, col = rasterio.transform.rowcol(tf, b.gauge_lon, b.gauge_lat)
    gauge_elev = int(Z[row, col]) if 0 <= row < Z.shape[0] and 0 <= col < Z.shape[1] and Z[row, col] != DEM_NODATA else None
    f_dem = OUT / f"elevation_{g}.tif"
    ew, ns = cell_size_m((buffered.bounds[1] + buffered.bounds[3]) / 2)
    with rasterio.open(f_dem, "w", driver="GTiff", dtype="int16", count=1, width=Z.shape[1], height=Z.shape[0], crs=CRS_DEM, transform=tf, nodata=DEM_NODATA,
                       compress="lzw", predictor=2, tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(Z, 1)
        dst.set_band_description(1, "elevation above the EGM96 geoid, whole metres")
        dst.set_band_unit(1, "m")
        dst.update_tags(**COMMON_TAGS, GAUGE_ID=g, BASIN=b.gauge_name, SHORT_NAME=b.short, UNITS="metres (whole metres, int16)", VERTICAL_REFERENCE="EGM96 geoid (per the tile documentation)",
                        HORIZONTAL_CRS="WGS 84 (EPSG:4326), the datum of the SRTM-format tiles; the NAD83 outline was applied without a datum shift (about 1 m, far below the 30 m cell)",
                        RESOLUTION=f"1 arc-second: about {ew:.1f} m east-west x {ns:.1f} m north-south at this latitude", SOURCE=GIS["sources"]["elevation"]["product"],
                        SOURCE_TILES=",".join(tiles), SOURCE_URL=GIS["sources"]["elevation"]["url"], SOURCE_YEAR=GIS["sources"]["elevation"]["year"],
                        CLIP=f"outline buffered by {BUFFER_ARCSEC} arc-seconds; cells outside it are nodata ({DEM_NODATA}, also the tiles' void code)")
    # land cover: the cached NLCD file, clipped to the outline on its own grid
    f_src = LC_DIR / f"annual_nlcd_{LC_YEAR}_{g}.tif"
    with rasterio.open(f_src) as src:
        assert src.crs.to_epsg() == 5070 and abs(src.transform.a - 30) < 0.01 and abs(src.transform.e + 30) < 0.01, f"{f_src.name} is not on the 30 m Albers grid"
        assert src.nodata == LC_NODATA, f"{f_src.name}: nodata {src.nodata}, expected {LC_NODATA}"
        wb = from_bounds(*geom_lc.bounds, transform=src.transform)
        c0, r0 = int(np.floor(wb.col_off + 1e-6)), int(np.floor(wb.row_off + 1e-6))
        c1, r1 = int(np.ceil(wb.col_off + wb.width - 1e-6)), int(np.ceil(wb.row_off + wb.height - 1e-6))
        assert c0 >= 0 and r0 >= 0 and c1 <= src.width and r1 <= src.height, f"{f_src.name} does not cover the outline"
        win = Window(c0, r0, c1 - c0, r1 - r0)
        L, ltf, cmap = src.read(1, window=win), src.window_transform(win), src.colormap(1)
        lc_crs = src.crs
    lmask = geometry_mask([geom_lc], L.shape, ltf, invert=True)
    L = np.where(lmask, L, LC_NODATA).astype("uint8")
    cnt = np.bincount(L[lmask], minlength=256)
    known = int(sum(cnt[c] for c in NLCD))
    class_share = {c: float(cnt[c] / known) for c in NLCD if cnt[c]}
    group_share = {k: float(sum(cnt[c] for c in codes) / known) for k, codes in GROUPS.items()}
    dom_class = max(class_share, key=class_share.get)
    dom_group = max(group_share, key=group_share.get)
    f_lc = OUT / f"landcover_{g}.tif"
    with rasterio.open(f_lc, "w", driver="GTiff", dtype="uint8", count=1, width=L.shape[1], height=L.shape[0], crs=lc_crs, transform=ltf, nodata=LC_NODATA,
                       compress="lzw", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(L, 1)
        dst.write_colormap(1, cmap)
        dst.set_band_description(1, f"Annual NLCD land-cover class code, {LC_YEAR} layer (legend: landcover_classes.csv)")
        dst.update_tags(**COMMON_TAGS, GAUGE_ID=g, BASIN=b.gauge_name, SHORT_NAME=b.short, UNITS="class code (see landcover_classes.csv)", MAP_YEAR=str(LC_YEAR),
                        HORIZONTAL_CRS="NAD83 / Conus Albers (EPSG:5070), the native grid", RESOLUTION="30 m", SOURCE=GIS["sources"]["land_cover"]["product"],
                        SOURCE_URL=GIS["sources"]["land_cover"]["url"], SOURCE_LAYER=GIS["sources"]["land_cover"]["layer"], SOURCE_FILE=rel(f_src),
                        CLIP=f"the outline itself (no buffer); cells outside it are nodata ({LC_NODATA}, as in the source)")
    # areas, gauge
    area_lc = geom_lc.area / 1e6
    area_geod = abs(GEOD.geometry_area_perimeter(geom)[0]) / 1e6
    gauge = Point(b.gauge_lon, b.gauge_lat)
    near = nearest_points(geom.boundary, gauge)[0]
    R[g] = dict(
        gauge_id=g, name=b.gauge_name, short=b.short, landscape=b.landscape, n_parts=len(getattr(geom, "geoms", [geom])),
        area_km2=float(area_lc), area_geodesic_km2=float(area_geod), outline_file_area_km2=float(OUTLINE_FILE["area_km2"][g]),
        camels_area_gages2_km2=float(b.area_gages2), camels_area_geospa_fabric_km2=float(b.area_geospa_fabric),
        elev_mean_m=elev_mean, elev_min_m=float(zin.min()), elev_max_m=float(zin.max()), camels_elev_mean_m=float(b.elev_mean), outline_file_elev_mean_m=float(OUTLINE_FILE["elev_mean_m"][g]),
        n_elevation_cells=int(ok.sum()), n_elevation_voids_inside=int((inside & ~valid).sum()), n_elevation_cells_written=int(valid.sum()), elevation_tiles=tiles,
        elevation_shape=[int(v) for v in Z.shape], elevation_cell_m_ew=round(ew, 2), elevation_cell_m_ns=round(ns, 2),
        dominant_nlcd_class=f"{dom_class} {NLCD[dom_class]}", dominant_nlcd_share=class_share[dom_class], dominant_land_cover=dom_group, dominant_group_share=group_share[dom_group],
        forest_share=group_share["forest"], land_cover_share=group_share, nlcd_class_share={f"{c} {NLCD[c]}": v for c, v in class_share.items()},
        n_landcover_cells=int(lmask.sum()), n_landcover_unclassified=int(lmask.sum() - known), landcover_shape=[int(v) for v in L.shape],
        camels_frac_forest=float(b.frac_forest), camels_dom_land_cover=b.dom_land_cover, camels_dom_land_cover_frac=float(b.dom_land_cover_frac),
        gauge=dict(lat=float(b.gauge_lat), lon=float(b.gauge_lon), elev_m=gauge_elev, inside_outline=bool(geom.contains(gauge)),
                   distance_to_outline_km=float(GEOD.inv(near.x, near.y, gauge.x, gauge.y)[2] / 1000)),
        elevation_file=rel(f_dem), landcover_file=rel(f_lc))
    r = R[g]
    r["check"] = dict(                                                         # exported layer vs results/basin_gis.json (same cells) and vs the CAMELS table
        area_vs_basin_gis_km2=r["area_km2"] - gis["area_albers_km2"], area_vs_basin_gis_geodesic_km2=r["area_geodesic_km2"] - gis["area_km2"],
        area_vs_camels_gages2_pct=100 * (r["area_km2"] / b.area_gages2 - 1),
        elev_vs_basin_gis_m=elev_mean - gis["elev_mean_m"], elev_vs_camels_m=elev_mean - float(b.elev_mean),
        dominant_same_as_basin_gis=dom_group == gis["dominant_land_cover"], basin_gis_dominant_land_cover=gis["dominant_land_cover"],
        forest_vs_basin_gis_share=group_share["forest"] - gis["land_cover_share"]["forest"], forest_vs_camels_points=100 * (group_share["forest"] - float(b.frac_forest)),
        n_landcover_cells_same=int(lmask.sum()) == gis["n_landcover_cells"], n_elevation_cells_same=int(ok.sum()) == gis["n_elevation_cells"],
        gauge_elev_vs_basin_gis_m=None if gauge_elev is None else gauge_elev - gis["gauge"]["elev_m"],
        gauge_distance_vs_basin_gis_km=r["gauge"]["distance_to_outline_km"] - gis["gauge"]["distance_to_outline_km"],
        basin_gis=dict(area_km2=gis["area_km2"], area_albers_km2=gis["area_albers_km2"], elev_mean_m=gis["elev_mean_m"], forest_share=gis["land_cover_share"]["forest"], gauge_elev_m=gis["gauge"]["elev_m"], gauge_distance_to_outline_km=gis["gauge"]["distance_to_outline_km"]))
    c = r["check"]
    c["agrees_with_basin_gis"] = bool(abs(c["area_vs_basin_gis_km2"]) < TOL["area_km2"] and abs(c["elev_vs_basin_gis_m"]) < TOL["elev_m"] and c["dominant_same_as_basin_gis"]
                                      and abs(c["forest_vs_basin_gis_share"]) < TOL["share"] and c["n_landcover_cells_same"] and c["n_elevation_cells_same"]
                                      and c["gauge_elev_vs_basin_gis_m"] == 0 and abs(c["gauge_distance_vs_basin_gis_km"]) < TOL["distance_km"])
    c["elev_differs_from_camels_by_more_than_25_m"] = bool(abs(c["elev_vs_camels_m"]) > 25)
    say(f"{b.short:18s} area {r['area_km2']:7.1f} km2 ({c['area_vs_camels_gages2_pct']:+5.1f}% vs CAMELS)  elevation mean {elev_mean:7.1f} m ({c['elev_vs_camels_m']:+6.1f} m vs CAMELS, "
        f"{c['elev_vs_basin_gis_m']:+.4f} m vs basin_gis)  dominant: {dom_group} {100 * group_share[dom_group]:.0f}%  forest {100 * group_share['forest']:4.1f}% ({c['forest_vs_camels_points']:+5.1f} pts)  "
        f"{'agrees' if c['agrees_with_basin_gis'] else 'DIFFERS'} with basin_gis.json")
dem_tile.cache_clear()

# ---- vector layers ---------------------------------------------------------------------------------------------------
ATTR = ["gauge_id", "name", "short", "landscape", "n_parts", "area_km2", "area_geodesic_km2", "camels_area_gages2_km2", "camels_area_geospa_fabric_km2", "outline_file_area_km2",
        "elev_mean_m", "elev_min_m", "elev_max_m", "camels_elev_mean_m", "outline_file_elev_mean_m", "dominant_nlcd_class", "dominant_nlcd_share", "dominant_land_cover",
        "dominant_group_share", "forest_share", "camels_frac_forest", "camels_dom_land_cover", "camels_dom_land_cover_frac", "n_elevation_cells", "n_landcover_cells",
        "elevation_file", "landcover_file"]
rows = [{k: R[g][k] for k in ATTR} for g in B.index]
for row, g in zip(rows, B.index):
    row["area_diff_gages2_pct"], row["elev_diff_camels_m"], row["forest_diff_camels_points"] = (R[g]["check"][k] for k in ("area_vs_camels_gages2_pct", "elev_vs_camels_m", "forest_vs_camels_points"))
outlines = gpd.GeoDataFrame(rows, geometry=[GEOM[g] for g in B.index], crs=CRS_OUTLINES)
gauges = gpd.GeoDataFrame([dict(gauge_id=g, name=R[g]["name"], short=R[g]["short"], landscape=R[g]["landscape"], lat=R[g]["gauge"]["lat"], lon=R[g]["gauge"]["lon"],
                                elev_at_gauge_m=R[g]["gauge"]["elev_m"], inside_outline=R[g]["gauge"]["inside_outline"], distance_to_outline_km=R[g]["gauge"]["distance_to_outline_km"],
                                source=GAUGE_SRC) for g in B.index],
                          geometry=[Point(R[g]["gauge"]["lon"], R[g]["gauge"]["lat"]) for g in B.index], crs=CRS_OUTLINES)
f_gpkg, f_geojson, f_gauges = OUT / "basins_12.gpkg", OUT / "basins_12.geojson", OUT / "gauges_12.geojson"
outlines.to_file(f_gpkg, layer="outlines", driver="GPKG")
gauges.to_file(f_gpkg, layer="gauges", driver="GPKG")
outlines.to_crs(CRS_WEB).to_file(f_geojson, driver="GeoJSON", COORDINATE_PRECISION=7)
gauges.to_crs(CRS_WEB).to_file(f_gauges, driver="GeoJSON", COORDINATE_PRECISION=7)

# ---- legend ----------------------------------------------------------------------------------------------------------
present = {c: sum(1 for g in B.index if f"{c} {NLCD[c]}" in R[g]["nlcd_class_share"]) for c in NLCD}
with open(OUT / "landcover_classes.csv", "w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh)
    wr.writerow(["code", "class", "group", "red", "green", "blue", "basins_present_of_12", "note"])
    for c in NLCD:
        wr.writerow([c, NLCD[c], GROUP_OF[c], *cmap[c][:3], present[c], f"Annual NLCD class code, {LC_YEAR} layer; group as in results/basin_gis.json"])
    wr.writerow([LC_NODATA, "no data", "", *cmap[LC_NODATA][:3], "", "nodata value of the source and of the clipped files: outside the outline"])

# ---- verification: read everything back -------------------------------------------------------------------------------
V, fails = {}, []
check = lambda name, ok, detail="": (V.__setitem__(name, dict(ok=bool(ok), detail=detail)), None if ok else fails.append(f"{name}: {detail}"))
rd = gpd.read_file(f_gpkg, layer="outlines")
check("gpkg_outlines_count_12", len(rd) == 12, f"{len(rd)} features")
check("gpkg_outlines_crs_4269", rd.crs is not None and rd.crs.to_epsg() == 4269, str(rd.crs))
check("gpkg_outlines_ids", sorted(rd.gauge_id) == sorted(B.index), "gauge ids")
check("gpkg_outlines_geometry_unchanged", all(abs(rd.set_index("gauge_id").geometry[g].area - GEOM[g].area) < 1e-12 for g in B.index), "planar area in degrees identical")
rg = gpd.read_file(f_gpkg, layer="gauges")
check("gpkg_gauges_count_12", len(rg) == 12 and rg.crs.to_epsg() == 4269, f"{len(rg)} features, {rg.crs}")
check("gpkg_gauges_inside_outlines", all(GEOM[r.gauge_id].contains(r.geometry) for r in rg.itertuples()), "every gauge point lies inside its outline")
check("gpkg_layers", sorted(l for l, _ in gpd.list_layers(f_gpkg).itertuples(index=False)) == ["gauges", "outlines"], str(gpd.list_layers(f_gpkg).name.tolist()))
rj = gpd.read_file(f_geojson)
check("geojson_count_12_crs_4326", len(rj) == 12 and rj.crs is not None and rj.crs.equals(CRS_WEB, ignore_axis_order=True), f"{len(rj)} features, {rj.crs}")
check("geojson_matches_gpkg", all(abs(a - b) < 1e-9 for a, b in zip(rj.set_index("gauge_id").area_km2[B.index], rd.set_index("gauge_id").area_km2[B.index])), "area_km2 attribute identical")
rgj = gpd.read_file(f_gauges)
check("gauges_geojson_count_12_crs_4326", len(rgj) == 12 and rgj.crs.equals(CRS_WEB, ignore_axis_order=True), f"{len(rgj)} features, {rgj.crs}")
n_dem = n_lc = 0
for g in B.index:
    r, geom, geom_lc = R[g], GEOM[g], GEOM_LC[g]
    with rasterio.open(OUT / f"elevation_{g}.tif") as ds:
        n_dem += 1
        bb, Z = ds.bounds, ds.read(1)
        inside = geometry_mask([geom], Z.shape, ds.transform, invert=True)
        lats = ds.transform.f + ds.transform.e * (np.arange(Z.shape[0]) + 0.5)
        ok = inside & (Z != ds.nodata)
        m = float(np.average(Z[ok].astype("float64"), weights=np.repeat(np.cos(np.radians(lats))[:, None], Z.shape[1], 1)[ok]))
        check(f"elevation_{g}", ds.crs.to_epsg() == 4326 and ds.nodata == DEM_NODATA and ds.compression is not None and ds.compression.name.lower() == "lzw" and ds.dtypes[0] == "int16"
              and abs(ds.transform.a - 1 / 3600) < 1e-12 and abs(ds.transform.e + 1 / 3600) < 1e-12
              and bb.left <= geom.bounds[0] and bb.bottom <= geom.bounds[1] and bb.right >= geom.bounds[2] and bb.top >= geom.bounds[3]
              and abs(m - GIS["basins"][g]["elev_mean_m"]) < TOL["elev_m"] and int(ok.sum()) == GIS["basins"][g]["n_elevation_cells"],
              f"crs {ds.crs}, nodata {ds.nodata}, {ds.compression}, {ds.dtypes[0]}, bounds contain outline: {bb.left <= geom.bounds[0] and bb.bottom <= geom.bounds[1] and bb.right >= geom.bounds[2] and bb.top >= geom.bounds[3]}, "
              f"mean read back {m:.4f} vs basin_gis {GIS['basins'][g]['elev_mean_m']:.4f}, cells {int(ok.sum())} vs {GIS['basins'][g]['n_elevation_cells']}")
    with rasterio.open(OUT / f"landcover_{g}.tif") as ds:
        n_lc += 1
        bb, L = ds.bounds, ds.read(1)
        lmask = geometry_mask([geom_lc], L.shape, ds.transform, invert=True)
        cnt = np.bincount(L[lmask], minlength=256)
        known = sum(cnt[c] for c in NLCD)
        shares_ok = all(abs(cnt[c] / known - GIS["basins"][g]["nlcd_class_share"].get(f"{c} {NLCD[c]}", 0.0)) < TOL["share"] for c in NLCD)
        check(f"landcover_{g}", ds.crs.to_epsg() == 5070 and ds.nodata == LC_NODATA and ds.compression is not None and ds.compression.name.lower() == "lzw" and ds.dtypes[0] == "uint8"
              and abs(ds.transform.a - 30) < 0.01 and bb.left <= geom_lc.bounds[0] and bb.bottom <= geom_lc.bounds[1] and bb.right >= geom_lc.bounds[2] and bb.top >= geom_lc.bounds[3]
              and int(lmask.sum()) == GIS["basins"][g]["n_landcover_cells"] and shares_ok and (L[~lmask] == LC_NODATA).all() and len(ds.colormap(1)) > 0,
              f"crs {ds.crs}, nodata {ds.nodata}, {ds.compression}, {ds.dtypes[0]}, bounds contain outline: {bb.left <= geom_lc.bounds[0] and bb.bottom <= geom_lc.bounds[1] and bb.right >= geom_lc.bounds[2] and bb.top >= geom_lc.bounds[3]}, "
              f"cells {int(lmask.sum())} vs {GIS['basins'][g]['n_landcover_cells']}, class shares equal: {shares_ok}")
check("twelve_rasters_each", n_dem == 12 and n_lc == 12, f"{n_dem} elevation, {n_lc} land cover")
check("all_basins_agree_with_basin_gis", all(R[g]["check"]["agrees_with_basin_gis"] for g in B.index), ", ".join(g for g in B.index if not R[g]["check"]["agrees_with_basin_gis"]) or "all 12")

# ---- hashes, sizes --------------------------------------------------------------------------------------------------------
def file_record(f, kind, layer, crs, resolution, units, nodata, source, licence):
    return dict(file=f"{rel(OUT)}/{f}", kind=kind, layer=layer, crs=crs, resolution=resolution, units=units, nodata=nodata, source=source, licence=licence,
                bytes=(OUT / f).stat().st_size, sha256=sha256(OUT / f))


def licence_of(record):
    """The licence text a results/data_flow.json dataset record holds, or UNVERIFIED with what the record does say."""
    for k in ("licence", "license", "terms"):
        if record.get(k):
            return str(record[k])
    got = record.get("obtained", {})
    return f"UNVERIFIED (no licence recorded in results/data_flow.json or results/basin_gis.json; data_flow.json records: login needed = {got.get('login_needed', 'not stated')})"


DS = {d["key"]: d for d in FLOW["datasets"]}
DS.update({p["key"]: p for d in FLOW["datasets"] for p in d.get("parts", [])})
LIC = dict(outlines=licence_of(DS["camels_us"]), attributes=licence_of(DS["camels_us"]),                  # the CAMELS parts say "see CAMELS-US": use the parent record
           elevation=licence_of(DS["elevation_tiles"]), landcover=licence_of(DS["nlcd"]))
camels_readme = (CAM / "readme.txt").read_text(encoding="utf-8", errors="replace") if (CAM / "readme.txt").exists() else ""
CITE = ("data/camels_us/readme.txt asks that publications using the CAMELS attributes cite Addor et al. 2017 and publications using the CAMELS time series cite Newman et al. 2015; "
        "results/data_flow.json names Newman et al. 2015 as the source of the outlines (no request to cite is recorded for them)") if "agree to cite" in camels_readme else ""
SRC = dict(outlines=f"{rel(SHP)} (CAMELS-US basin_set_full_res; hru_id = gauge id); attributes from {rel(CAM / 'camels_topo.txt')} and camels_vege.txt (area_gages2, elev_mean, frac_forest, gauge position)",
           elevation=GIS["sources"]["elevation"]["product"] + "; tiles " + ", ".join(GIS["sources"]["elevation"]["tiles"]) + f" cached in {rel(DEM_DIR)}/",
           landcover=f"{GIS['sources']['land_cover']['product']}; layer {GIS['sources']['land_cover']['layer']}, year {LC_YEAR}; cached as {rel(LC_DIR)}/annual_nlcd_{LC_YEAR}_<gauge_id>.tif")
FILES = [file_record("basins_12.gpkg", "GeoPackage (vector)", "outlines (12 polygons), gauges (12 points)", "EPSG:4269 geographic NAD83 (the source shapefile's CRS)",
                     "vector; the outline vertices follow 30 m raster-cell edges (results/basin_gis.json notes)", "degrees; area_km2 in km2 (planar, EPSG:5070); elevations in m; shares 0-1", "none",
                     SRC["outlines"], LIC["outlines"] + (f"; {CITE}" if CITE else "")),
         file_record("basins_12.geojson", "GeoJSON (vector)", "outlines (12 polygons)", f"EPSG:4326 WGS 84 (from EPSG:4269 by pyproj: {DATUM_STEP})", "vector, coordinates to 7 decimals (about 1 cm)",
                     "as basins_12.gpkg", "none", SRC["outlines"], LIC["outlines"] + (f"; {CITE}" if CITE else "")),
         file_record("gauges_12.geojson", "GeoJSON (vector)", "gauges (12 points)", f"EPSG:4326 WGS 84 (from EPSG:4269 by pyproj: {DATUM_STEP})", "points, coordinates to 7 decimals",
                     "degrees; elev_at_gauge_m in m", "none", GAUGE_SRC, LIC["attributes"] + (f"; {CITE}" if CITE else "")),
         file_record("landcover_classes.csv", "CSV legend", "-", "-", "-", "class code, name, group, RGB colour (the colour table of the source files)", "-",
                     f"class names as in results/basin_gis.json sources.land_cover.classes; groups as in code/07_basin_maps.py", LIC["landcover"])]
for g in B.index:
    r = R[g]
    FILES.append(file_record(f"elevation_{g}.tif", "GeoTIFF int16, LZW, tiled", f"elevation, {r['short']}", "EPSG:4326 WGS 84 (the tiles' datum); heights above the EGM96 geoid",
                             f"1 arc-second ({r['elevation_cell_m_ew']} m east-west x {r['elevation_cell_m_ns']} m north-south here); {r['elevation_shape'][1]} x {r['elevation_shape'][0]} cells",
                             "metres (whole metres)", DEM_NODATA, SRC["elevation"].replace(", ".join(GIS["sources"]["elevation"]["tiles"]), ", ".join(r["elevation_tiles"])), LIC["elevation"]))
for g in B.index:
    r = R[g]
    FILES.append(file_record(f"landcover_{g}.tif", "GeoTIFF uint8 with colour table, LZW, tiled", f"land cover {LC_YEAR}, {r['short']}", "EPSG:5070 NAD83 / Conus Albers (the native grid)",
                             f"30 m; {r['landcover_shape'][1]} x {r['landcover_shape'][0]} cells", "NLCD class code (landcover_classes.csv)", LC_NODATA, SRC["landcover"], LIC["landcover"]))
total_bytes = sum(f["bytes"] for f in FILES)
check("total_size_under_150_mb", total_bytes < 150e6, f"{total_bytes / 1e6:.1f} MB")

# ---- summary tables ---------------------------------------------------------------------------------------------------
TABLE = [dict(gauge_id=g, short=R[g]["short"], landscape=R[g]["landscape"],
              area_km2=R[g]["area_km2"], basin_gis_area_km2=GIS["basins"][g]["area_albers_km2"], area_vs_basin_gis_km2=R[g]["check"]["area_vs_basin_gis_km2"],
              camels_area_gages2_km2=R[g]["camels_area_gages2_km2"], area_vs_camels_pct=R[g]["check"]["area_vs_camels_gages2_pct"],
              elev_mean_m=R[g]["elev_mean_m"], basin_gis_elev_mean_m=GIS["basins"][g]["elev_mean_m"], elev_vs_basin_gis_m=R[g]["check"]["elev_vs_basin_gis_m"],
              camels_elev_mean_m=R[g]["camels_elev_mean_m"], elev_vs_camels_m=R[g]["check"]["elev_vs_camels_m"], elev_differs_from_camels_by_more_than_25_m=R[g]["check"]["elev_differs_from_camels_by_more_than_25_m"],
              dominant_land_cover=R[g]["dominant_land_cover"], basin_gis_dominant_land_cover=GIS["basins"][g]["dominant_land_cover"], dominant_same=R[g]["check"]["dominant_same_as_basin_gis"],
              dominant_nlcd_class=R[g]["dominant_nlcd_class"], forest_share=R[g]["forest_share"], basin_gis_forest_share=GIS["basins"][g]["land_cover_share"]["forest"],
              camels_frac_forest=R[g]["camels_frac_forest"], forest_vs_camels_points=R[g]["check"]["forest_vs_camels_points"], camels_dom_land_cover=R[g]["camels_dom_land_cover"],
              gauge_elev_m=R[g]["gauge"]["elev_m"], basin_gis_gauge_elev_m=GIS["basins"][g]["gauge"]["elev_m"], gauge_elev_vs_basin_gis_m=R[g]["check"]["gauge_elev_vs_basin_gis_m"],
              gauge_distance_to_outline_km=R[g]["gauge"]["distance_to_outline_km"], basin_gis_gauge_distance_km=GIS["basins"][g]["gauge"]["distance_to_outline_km"],
              gauge_distance_vs_basin_gis_km=R[g]["check"]["gauge_distance_vs_basin_gis_km"], agrees_with_basin_gis=R[g]["check"]["agrees_with_basin_gis"]) for g in B.index]
elev_off = [f"{t['short']} {t['elev_vs_camels_m']:+.0f} m" for t in TABLE if t["elev_differs_from_camels_by_more_than_25_m"]]
elev_note = next((n for n in GIS["notes"] if n.startswith("Elevation:")), "")
agg = lambda vals: dict(median=float(np.median(vals)), mean_absolute=float(np.mean(np.abs(vals))), largest=float(vals[int(np.argmax(np.abs(vals)))]), largest_basin=TABLE[int(np.argmax(np.abs(vals)))]["short"])
SUMMARY = dict(area_vs_camels_gages2_pct=agg(np.array([t["area_vs_camels_pct"] for t in TABLE])), elev_vs_camels_m=agg(np.array([t["elev_vs_camels_m"] for t in TABLE])),
               forest_vs_camels_points=agg(np.array([t["forest_vs_camels_points"] for t in TABLE])), basins_agreeing_with_basin_gis=sum(t["agrees_with_basin_gis"] for t in TABLE),
               basins_with_elev_more_than_25_m_from_camels=elev_off)

# ---- README -----------------------------------------------------------------------------------------------------------
NOW = datetime.now().strftime("%Y-%m-%d %H:%M")
md = []
P = md.append
P(f"# GIS layers of the 12 study basins (deliverable D3)\n")
P(f"Written by `code/27_gis_layers.py` on {NOW}. {len(FILES)} files ({total_bytes / 1e6:.1f} MB) plus this README. Rebuild, never hand-edit: `python -X utf8 code/27_gis_layers.py` from the project root "
  f"(a Python with rasterio, shapely, pyproj, geopandas). The numbers below are the script's own; the full record with sha256 hashes is `{rel(JSON_OUT)}`.\n")
P("Claude Code wrote and ran this script under the team lead's direction; **the team's own check of these layers is not done.**\n")
if REVIEW_LINE:
    P(REVIEW_LINE + "\n")
P("## What the layers are\n")
P(f"- **Outlines** (`basins_12.gpkg` layer `outlines`, `basins_12.geojson`): the 12 CAMELS-US basin polygons taken unchanged from `{rel(SHP)}` (hru_id = gauge id), "
  f"{'with ' + str(len(repaired)) + ' invalid geometries repaired by buffer(0) as in script 07 (' + ', '.join(R[g]['short'] for g in repaired) + '; areas unchanged)' if repaired else 'all valid as shipped'}. "
  "Attributes: `gauge_id`, `name` (CAMELS gauge name), `short` (the project's short name), `landscape` (the class of 01_data_audit.py), `area_km2` (planar area in EPSG:5070, equal-area), "
  "`area_geodesic_km2` (GRS80 ellipsoid), `camels_area_gages2_km2` and `camels_area_geospa_fabric_km2` (the CAMELS table), `outline_file_area_km2` (the AREA field of the shapefile), "
  "`elev_mean_m` / `elev_min_m` / `elev_max_m` (from the exported elevation layer, cells with centre inside the outline, mean weighted by cos(latitude)), `camels_elev_mean_m`, "
  "`outline_file_elev_mean_m`, `dominant_nlcd_class` and `dominant_nlcd_share` (the single NLCD class with the largest share), `dominant_land_cover` and `dominant_group_share` "
  "(the seven groups of results/basin_gis.json), `forest_share` (NLCD 41+42+43), `camels_frac_forest`, `camels_dom_land_cover` (MODIS IGBP, another map), the differences to CAMELS "
  "(`area_diff_gages2_pct`, `elev_diff_camels_m`, `forest_diff_camels_points`), cell counts and the file names of the two rasters.")
P(f"- **Gauges** (`basins_12.gpkg` layer `gauges`, `gauges_12.geojson`): the 12 gauge points at the CAMELS `gauge_lat`/`gauge_lon` of `{rel(CAM / 'camels_topo.txt')}`, as the table gives them "
  f"(not rounded: results/data_audit.json holds the same values rounded to 0.001 degree, which would put the points {min(GAUGE_SHIFT_M):.0f} to {max(GAUGE_SHIFT_M):.0f} m away), "
  f"with the elevation cell at the gauge, whether the point lies "
  f"inside its outline ({sum(R[g]['gauge']['inside_outline'] for g in B.index)} of 12 do) and its geodesic distance to the outline (at most {max(R[g]['gauge']['distance_to_outline_km'] for g in B.index):.2f} km). The elevation cell and the distance equal results/basin_gis.json (part of the check below).")
BUF_NS = round(float(np.mean([R[g]["elevation_cell_m_ns"] for g in B.index])) * BUFFER_ARCSEC, -1)                  # the buffer in metres, to the nearest 10: north-south ...
BUF_EW = [round(float(fn(R[g]["elevation_cell_m_ew"] for g in B.index)) * BUFFER_ARCSEC, -1) for fn in (min, max)]   # ... and east-west (a degree of longitude is shorter further north)
P(f"- **Elevation** (`elevation_<gauge_id>.tif`, 12 files): the 1-arc-second cells of the SRTM-format tiles, on the tiles' own grid (cell centres on whole arc-seconds, north up), "
  f"cut to the outline buffered by {BUFFER_ARCSEC} arc-seconds (about {BUF_NS:.0f} m north-south, {BUF_EW[0]:.0f}-{BUF_EW[1]:.0f} m east-west) so that the ridge lines around the basin stay visible; "
  f"everything outside the buffer is nodata ({DEM_NODATA}, also the tiles' void code). Whole metres above the EGM96 geoid, int16, LZW. The tiles are WGS 84 and the outlines NAD83; "
  f"as in script 07 no datum shift was applied (about 1 m, far below the 30 m cell). Voids inside the outlines: {sum(R[g]['n_elevation_voids_inside'] for g in B.index)} cells.")
P(f"- **Land cover** (`landcover_<gauge_id>.tif`, 12 files): the Annual NLCD {LC_YEAR} class codes of `{rel(LC_DIR)}/annual_nlcd_{LC_YEAR}_<gauge_id>.tif`, on the native 30 m grid "
  f"in EPSG:5070, cut to the outline itself (cells whose centre lies inside; no buffer); outside = nodata {LC_NODATA}. The NLCD colour table is kept, so QGIS shows the usual colours. "
  f"Legend: `landcover_classes.csv` ({len(NLCD)} classes, {sum(1 for c in NLCD if present[c])} of them present in the 12 basins).")
P("")
P("## Files\n")
P("| file | kind | layer / content | CRS | resolution | units | nodata | size | sha256 |")
P("|---|---|---|---|---|---|---|---|---|")
for f in FILES:
    P(md_row(f"`{Path(f['file']).name}`", f["kind"], f["layer"], f["crs"], f["resolution"], f["units"], f["nodata"], f"{f['bytes'] / 1e6:.2f} MB", f"`{f['sha256']}`"))
P("")
P("## Sources and licences\n")
P("The licence column repeats what the project files record. **UNVERIFIED** means that neither `results/data_flow.json` nor `results/basin_gis.json` holds a licence for the "
  "product; the exported files are derived from sources whose terms are not recorded here, and no licence of their own is asserted here. The access facts beside it are from `results/data_flow.json`.\n")
P("| source | used for | what the project files record | licence |")
P("|---|---|---|---|")
for key, used, lic in (("camels_outlines", "outlines, gauges, attributes", LIC["outlines"]), ("elevation_tiles", "elevation_<gauge_id>.tif", LIC["elevation"]), ("nlcd", "landcover_<gauge_id>.tif", LIC["landcover"])):
    d = DS.get(key, DS["camels_us"])
    got = d.get("obtained", {})
    P(md_row(f"{d['name']} (producer: {d.get('producer', 'not stated')})", used,
             f"version: {d.get('version', 'not stated')}; obtained: {got.get('url_pattern', 'not stated')}; downloaded {got.get('date_downloaded', 'date not recorded')}; "
             f"login: {got.get('login_needed', 'not stated')}", lic + ('; ' + CITE if key == 'camels_outlines' and CITE else '')))
P("")
P("## Check: the exported layers against results/basin_gis.json and the CAMELS attribute table\n")
P(f"The clipping rule is the one of script 07 (a cell counts when its centre lies inside the outline), so the statistics read back from the exported rasters should equal "
  f"results/basin_gis.json to numerical precision: tolerance {TOL['area_km2']} km2 on area, {TOL['elev_m']} m on mean elevation, {TOL['share']} on the forest share, the same cell counts "
  f"and the same dominant group, and for the gauge point the same elevation cell and the same distance to the outline (tolerance {TOL['distance_km']} km). **{SUMMARY['basins_agreeing_with_basin_gis']} of 12 basins agree.** Against the CAMELS table (another source, other outlines and grids): area median "
  f"{SUMMARY['area_vs_camels_gages2_pct']['median']:+.2f}% (largest {SUMMARY['area_vs_camels_gages2_pct']['largest']:+.1f}%, {SUMMARY['area_vs_camels_gages2_pct']['largest_basin']}); "
  f"mean elevation median {SUMMARY['elev_vs_camels_m']['median']:+.1f} m (largest {SUMMARY['elev_vs_camels_m']['largest']:+.0f} m, {SUMMARY['elev_vs_camels_m']['largest_basin']}); "
  f"forest share median {SUMMARY['forest_vs_camels_points']['median']:+.1f} points (largest {SUMMARY['forest_vs_camels_points']['largest']:+.1f}, {SUMMARY['forest_vs_camels_points']['largest_basin']}). "
  "Differences are exported minus reference.\n")
P("### Area (km2)\n")
P("| gauge_id | basin | exported (EPSG:5070) | basin_gis.json | diff | CAMELS area_gages2 | diff % |")
P("|---|---|---|---|---|---|---|")
for t in TABLE:
    P(f"| {t['gauge_id']} | {t['short']} | {t['area_km2']:.3f} | {t['basin_gis_area_km2']:.3f} | {t['area_vs_basin_gis_km2']:+.6f} | {t['camels_area_gages2_km2']:.2f} | {t['area_vs_camels_pct']:+.2f} |")
P("")
P("### Mean elevation (m)\n")
P("| gauge_id | basin | exported layer | basin_gis.json | diff | CAMELS elev_mean | diff | note |")
P("|---|---|---|---|---|---|---|---|")
for t in TABLE:
    P(f"| {t['gauge_id']} | {t['short']} | {t['elev_mean_m']:.2f} | {t['basin_gis_elev_mean_m']:.2f} | {t['elev_vs_basin_gis_m']:+.4f} | {t['camels_elev_mean_m']:.2f} | {t['elev_vs_camels_m']:+.1f} | "
      f"{'more than 25 m from CAMELS (see below)' if t['elev_differs_from_camels_by_more_than_25_m'] else ''} |")
P("")
P(f"**Known difference, carried over from results/basin_gis.json:** {len(elev_off)} mountain basins differ from the CAMELS elev_mean by more than 25 m ({', '.join(elev_off)}). "
  f"The note of script 07 reads: \"{elev_note}\"\n")
P("### Dominant land cover and forest share\n")
P("| gauge_id | basin | dominant group (exported) | basin_gis.json | same | dominant NLCD class (exported) | forest % exported | forest % basin_gis | CAMELS frac_forest % | diff points | CAMELS dom_land_cover (MODIS) |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
for t in TABLE:
    P(f"| {t['gauge_id']} | {t['short']} | {t['dominant_land_cover']} | {t['basin_gis_dominant_land_cover']} | {'yes' if t['dominant_same'] else 'NO'} | {t['dominant_nlcd_class']} | "
      f"{100 * t['forest_share']:.1f} | {100 * t['basin_gis_forest_share']:.1f} | {100 * t['camels_frac_forest']:.1f} | {t['forest_vs_camels_points']:+.1f} | {t['camels_dom_land_cover']} |")
P("")
P("The forest share at 30 m is below the CAMELS frac_forest in most basins; results/basin_gis.json (note 'Forest:') tests two reasons (woody wetlands, grid size) and finds neither a full explanation. "
  "CAMELS dom_land_cover comes from a MODIS map with other classes (IGBP), so it is shown for orientation only.\n")
n_tables, n_rows, bad_rows = table_problems("\n".join(md))
check("readme_tables_cells_per_row", not bad_rows, f"{n_tables} tables, {n_rows} rows, each with its header's number of cells" if not bad_rows else "; ".join(bad_rows))
P("## Verification (read back after writing)\n")
for k, v in V.items():
    if not k.startswith(("elevation_", "landcover_")):
        P(f"- {'PASS' if v['ok'] else 'FAIL'} {k}: {v['detail']}")
P(f"- {'PASS' if all(V[k]['ok'] for k in V if k.startswith('elevation_')) else 'FAIL'} the 12 elevation GeoTIFFs: EPSG:4326, int16, nodata {DEM_NODATA}, LZW, 1-arc-second grid, bounds contain the outline, "
  f"mean elevation and cell count read back equal results/basin_gis.json")
P(f"- {'PASS' if all(V[k]['ok'] for k in V if k.startswith('landcover_')) else 'FAIL'} the 12 land-cover GeoTIFFs: EPSG:5070, uint8, nodata {LC_NODATA}, LZW, 30 m, colour table, bounds contain the outline, "
  f"cells outside the outline are nodata, class shares and cell counts read back equal results/basin_gis.json")
P("")
P("## How to open\n")
P("QGIS: drag `basins_12.gpkg` in (both layers), then the GeoTIFFs; the land-cover files carry their colour table. The three CRS differ (EPSG:4269, 4326, 5070): let QGIS reproject on the fly. "
  "Python: `geopandas.read_file('basins_12.gpkg', layer='outlines')`, `rasterio.open('elevation_02077200.tif')`. The GeoJSON twins are for web maps (Leaflet, geojson.io).\n")
P("## Notes\n")
P(f"- Order of the basins: results/data_audit.json `selected` (the order of every other results file). Landscapes: {', '.join(f'{k} ({v})' for k, v in B.landscape.value_counts().items())}.")
P(f"- The elevation grid is the tiles' own: cell centres on whole arc-seconds, origin at half a cell; {sum(R[g]['n_elevation_cells'] for g in B.index):,} cells inside the 12 outlines, "
  f"{sum(R[g]['n_elevation_cells_written'] for g in B.index):,} written including the buffer. Land cover: {sum(R[g]['n_landcover_cells'] for g in B.index):,} cells inside the outlines, "
  f"{sum(R[g]['n_landcover_unclassified'] for g in B.index)} without a valid class.")
P("- The GeoPackage records its write time, so its sha256 changes with every run even when nothing else does; the GeoTIFFs and GeoJSON files are deterministic.")
P(f"- The GeoJSON twins were reprojected with pyproj's choice for EPSG:4269 to EPSG:4326: \"{DATUM_STEP}\".")
P("- Nothing here is fitted, tuned or selected; the train/validation/test split plays no role. No model reads these layers (the models read the CAMELS attribute values).")
if strays:
    P(f"- Files in this folder not written by the script (left alone): {', '.join(strays)}.")
(OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
FILES.append(file_record("README.md", "Markdown", "this file list, sources, licences, check table", "-", "-", "-", "-", "code/27_gis_layers.py", "-"))
total_bytes = sum(f["bytes"] for f in FILES)

# ---- results/gis_layers.json ----------------------------------------------------------------------------------------
INPUTS = {rel(f): dict(bytes=f.stat().st_size, sha256=sha256(f)) for f in [RES / "data_audit.json", RES / "basin_gis.json", RES / "data_flow.json", CAM / "camels_topo.txt", CAM / "camels_vege.txt"]
          + [SHP.with_suffix(s) for s in (".shp", ".shx", ".dbf", ".prj")] + sorted(DEM_DIR.glob("*.hgt.gz")) + [LC_DIR / f"annual_nlcd_{LC_YEAR}_{g}.tif" for g in B.index]}
status = "PASS" if not fails else "FAIL"
J = dict(
    what="The 12 study basins as GIS layers (outlines, gauges, elevation, land cover) in results/gis_layers/, exported from the sources of results/basin_gis.json with the clipping rule of "
         "code/07_basin_maps.py, and checked against results/basin_gis.json and the CAMELS attribute table. Deliverable D3 of the mid-term plan.",
    written=NOW, quick_mode=QUICK, code=dict(file="code/27_gis_layers.py", sha256=sha256(Path(__file__)), related="code/07_basin_maps.py", related_sha256=sha256(ROOT / "code" / "07_basin_maps.py")),
    disclosure=f"Claude Code wrote and ran this script under the team lead's direction (first run 2026-10-29; revised after the independent review; this run {NOW}); the team's own check of these layers is not done.",
    disclosure_independent_review=REVIEW_LINE or None, independent_review=REVIEW,
    environment=dict(python=sys.version.split()[0], rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__, geopandas=gpd.__version__, shapely=shapely.__version__, pyproj=pyproj.__version__,
                     numpy=np.__version__, pandas=pd.__version__, platform=sys.platform),
    parameters=dict(buffer_arcsec_elevation=BUFFER_ARCSEC, elevation_nodata=DEM_NODATA, landcover_nodata=LC_NODATA, compression="LZW (predictor 2 for the int16 elevation), tiled 256 x 256",
                    clipping="a cell belongs to the basin when its centre lies inside the outline (rasterio geometry_mask, all_touched=False), as in code/07_basin_maps.py",
                    crs=dict(outlines=CRS_OUTLINES, geojson=CRS_WEB, elevation=CRS_DEM + " (heights above the EGM96 geoid)", landcover=CRS_LC), datum_step_geojson=DATUM_STEP,
                    outlines_repaired_with_buffer0=repaired, land_cover_year=LC_YEAR, nlcd_classes=NLCD, groups={k: list(v) for k, v in GROUPS.items()}, tolerances_vs_basin_gis=TOL),
    sources=dict(outlines=SRC["outlines"], elevation=SRC["elevation"], land_cover=SRC["landcover"], camels_attributes="data/camels_us/camels_topo.txt (gauge_lat, gauge_lon, elev_mean, area_gages2, "
                 "area_geospa_fabric, slope_mean) and camels_vege.txt (frac_forest, dom_land_cover, dom_land_cover_frac); the 12 basins and their order from results/data_audit.json"),
    licences=dict(outlines_and_attributes=LIC["outlines"] + (f"; {CITE}" if CITE else ""), elevation=LIC["elevation"], land_cover=LIC["landcover"],
                  note="UNVERIFIED = no licence is recorded in results/data_flow.json or results/basin_gis.json; the exported files are derived from sources whose terms are not recorded here and assert no licence of their own"),
    basins=R, check_table=TABLE, summary=SUMMARY, known_note_from_basin_gis=elev_note, files=FILES, total_bytes=total_bytes, total_mb=round(total_bytes / 1e6, 2),
    verification=dict(status=status, checks=V, failures=fails), inputs_read=INPUTS,
    notes=[f"{len(B)} basins, {len(FILES)} files, {total_bytes / 1e6:.1f} MB; every raster was reopened after writing and its bounds checked against the outline.",
           "The GeoPackage stores its write time, so its sha256 changes on every run; the GeoTIFFs and GeoJSON files are deterministic.",
           "Nothing is downloaded by this script; a missing tile or land-cover file stops it (code/07_basin_maps.py fetches them).",
           "FLOOD_QUICK=1 writes to results/quick instead; the computation is the same." if QUICK else "Full run."],
    runtime_s=round(time.perf_counter() - T0, 1))
JSON_OUT.write_text(json.dumps(J, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

# ---- summary ----------------------------------------------------------------------------------------------------------
say("")
say(f"{'basin':18s} {'area km2':>9s} {'gages2':>8s} {'diff%':>6s} | {'elev m':>8s} {'CAMELS':>8s} {'diff m':>7s} | {'dominant (exported)':21s} {'basin_gis':9s} {'forest%':>7s} {'CAMELS%':>7s} {'pts':>6s}")
for t in TABLE:
    say(f"{t['short']:18s} {t['area_km2']:9.2f} {t['camels_area_gages2_km2']:8.2f} {t['area_vs_camels_pct']:+6.2f} | {t['elev_mean_m']:8.1f} {t['camels_elev_mean_m']:8.1f} {t['elev_vs_camels_m']:+7.1f} | "
        f"{t['dominant_land_cover']:21s} {'same' if t['dominant_same'] else 'DIFFERS':9s} {100 * t['forest_share']:7.1f} {100 * t['camels_frac_forest']:7.1f} {t['forest_vs_camels_points']:+6.1f}")
say(f"\n{SUMMARY['basins_agreeing_with_basin_gis']} of 12 basins agree with results/basin_gis.json; elevation more than 25 m from CAMELS: {', '.join(elev_off) or 'none'}")
say(f"files: {len(FILES)} in {rel(OUT)} ({total_bytes / 1e6:.1f} MB): {sum(1 for f in FILES if Path(f['file']).name.startswith('elevation_'))} elevation + "
    f"{sum(1 for f in FILES if Path(f['file']).name.startswith('landcover_') and f['file'].endswith('.tif'))} land-cover GeoTIFFs, "
    f"basins_12.gpkg (outlines, gauges), basins_12.geojson, gauges_12.geojson, landcover_classes.csv, README.md; record {rel(JSON_OUT)}")
for f in FILES:
    say(f"  {Path(f['file']).name:28s} {f['bytes'] / 1e6:7.2f} MB  {f['sha256'][:16]}")
if REVIEW:
    say(f"independent review on file: {REVIEW['file']} ({REVIEW['date']}, verdict '{REVIEW['verdict']}'; it read the export written {REVIEW['export_run_reviewed']})")
say(f"GIS LAYERS CHECK: {status}" + (" - " + "; ".join(fails) if fails else f" ({len(V)} checks)") + f"  ({J['runtime_s']} s)")
if fails:
    sys.exit(1)
