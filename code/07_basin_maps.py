"""Week 5 (remote sensing / GIS) - what do the 12 basins look like, and can the basin properties given to the model be trusted?

For every study basin we cut a 1-arc-second (about 30 m) elevation grid and a 30 m land-cover grid (Annual NLCD, year 2011)
to the basin outline and compute our own GIS statistics: area, elevation, slope, shape and the share of each land-cover group.
These are then checked against the CAMELS attribute table, which was derived independently by its authors and is what the
all-basin LSTM reads as basin properties. Nothing here is fitted or tuned, so the train/validation/test split is not involved.
Writes results/basin_gis.json, figures/basin_maps.png (elevation), figures/basin_landcover.png (land cover) and
figures/basin_gis_check.png (our numbers against CAMELS). Downloads are cached under data/dem and data/landcover.
"""
import gzip
import json
import time
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pyproj
import rasterio
import requests
import shapefile
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from shapely.geometry import Point, shape
from shapely.ops import nearest_points, transform as shp_transform

import flood
from common import *

use_style()
T0, C0 = time.perf_counter(), time.process_time()
B = flood.basins()
SHP = DATA / "camels_us" / "basin_set_full_res" / "HCDN_nhru_final_671.shp"
DEM_DIR, LC_DIR = DATA / "dem", DATA / "landcover"
DEM_DIR.mkdir(exist_ok=True); LC_DIR.mkdir(exist_ok=True)
DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"                  # 1 x 1 degree tiles in SRTM .hgt format
LC_URL = "https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs"    # USGS GeoServer with the MRLC layers
LC_LAYER, LC_YEAR, LC_ORIGIN = "Land-Cover-Native_conus_year_data", 2011, (-2415585.0, 3314805.0)   # 30 m grid corner, EPSG:5070
ALBERS = "EPSG:5070"                                                           # NAD83 / Conus Albers: equal-area, metres
GEOD = pyproj.Geod(ellps="GRS80")                                              # the ellipsoid of NAD83, the outlines' datum
COARSER = (1, 3, 10, 30)                                                       # slope is also computed on 3, 10, 30 times coarser grids
KM_BLOCK, SMOOTH_M = 33, 100.0                                                 # 33 land-cover cells = 990 m; tolerance (m) for the smoothed outline
NLCD = {11: "open water", 21: "developed, open space", 22: "developed, low intensity", 23: "developed, medium intensity",
        24: "developed, high intensity", 31: "barren land", 41: "deciduous forest", 42: "evergreen forest", 43: "mixed forest",
        52: "shrub/scrub", 71: "grassland/herbaceous", 81: "pasture/hay", 82: "cultivated crops", 90: "woody wetlands",
        95: "emergent herbaceous wetlands"}
GROUPS = {"forest": (41, 42, 43), "shrub and grass": (52, 71), "cropland and pasture": (81, 82), "wetland": (90, 95),
          "developed": (21, 22, 23, 24), "water": (11,), "bare ground": (31,)}
# land-cover map colours: the usual map conventions (green forest, blue water ...), stepped so that every pair stays apart for
# colour-blind readers (checked with the dataviz palette validator; plain red next to forest green fails, hence the crimson)
LC_FOREST, LC_GRASS, LC_CROP, LC_WETLAND, LC_DEVELOPED, LC_WATER, LC_BARE = "#1d6b3a", "#a3d15a", "#d97800", "#39a9c2", "#c2255c", "#1c4fa0", MUTED
LC_COL = dict(zip(GROUPS, (LC_FOREST, LC_GRASS, LC_CROP, LC_WETLAND, LC_DEVELOPED, LC_WATER, LC_BARE)))
say = lambda *a: print(*a, flush=True)


# ---- downloads (cached; a file that is already there is never fetched again) ------------------------------------------
def fetch(url, f, params=None, tries=8, resume=True):
    """Download url to file f unless it is cached. A broken transfer is resumed (the proxy drops long connections)."""
    if f.exists():
        return False
    part = f.with_name(f.name + ".part")
    for k in range(tries):
        try:
            have = part.stat().st_size if resume and part.exists() else 0
            with requests.get(url, params=params, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=(20, 60)) as r:
                r.raise_for_status()
                if "xml" in r.headers.get("content-type", ""):
                    raise requests.RequestException("the server answered with an error document: " + r.text[-200:])
                with open(part, "ab" if r.status_code == 206 else "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                if int(r.headers.get("content-length", -1)) not in (-1, part.stat().st_size - (have if r.status_code == 206 else 0)):
                    raise requests.RequestException("transfer cut short")
            part.replace(f)
            say(f"  downloaded {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
            return True
        except requests.RequestException as e:
            say(f"  retry {k + 1} for {f.name}: {str(e)[:120]}")
            time.sleep(2)
    raise RuntimeError(f"could not download {url}")


@lru_cache(maxsize=4)
def dem_tile(lat, lon):
    """One elevation tile named by its south-west corner: 3601 x 3601 cells, north-up, cell centres on whole arc-seconds."""
    name = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    f = DEM_DIR / f"{name}.hgt.gz"
    fetch(f"{DEM_URL}/{name[:3]}/{name}.hgt.gz", f)
    z = np.frombuffer(gzip.decompress(f.read_bytes()), ">i2")
    assert z.size == 3601 * 3601, f"{f.name} is not a 1-arc-second tile"
    return name, z.reshape(3601, 3601)


def dem_window(bounds, margin=36):
    """Elevation (m; NaN = no data) on the 1-arc-second grid around lon/lat bounds, rows from north to south.

    Returns the grid, the longitude of its columns, the latitude of its rows, its affine transform and the tiles used."""
    c0, c1 = int(np.floor(bounds[0] * 3600)) - margin, int(np.ceil(bounds[2] * 3600)) + margin      # arc-seconds east
    a0, a1 = int(np.floor(bounds[1] * 3600)) - margin, int(np.ceil(bounds[3] * 3600)) + margin      # arc-seconds north
    Z, names = np.full((a1 - a0 + 1, c1 - c0 + 1), np.nan, "float32"), []
    for la in range(a0 // 3600, (a1 - 1) // 3600 + 1):
        for lo in range(c0 // 3600, (c1 - 1) // 3600 + 1):
            name, t = dem_tile(la, lo)
            names.append(name)
            lo_a, hi_a, lo_c, hi_c = max(a0, la * 3600), min(a1, la * 3600 + 3600), max(c0, lo * 3600), min(c1, lo * 3600 + 3600)
            Z[a1 - hi_a:a1 - lo_a + 1, lo_c - c0:hi_c - c0 + 1] = t[la * 3600 + 3600 - hi_a:la * 3600 + 3600 - lo_a + 1, lo_c - lo * 3600:hi_c - lo * 3600 + 1]
    Z[Z == -32768] = np.nan                                                    # the .hgt code for a void
    return Z, np.arange(c0, c1 + 1) / 3600, np.arange(a1, a0 - 1, -1) / 3600, Affine(1 / 3600, 0, (c0 - 0.5) / 3600, 0, -1 / 3600, (a1 + 0.5) / 3600), names


def landcover(g, geom_albers):
    """Land-cover class codes on the native 30 m Albers grid around one basin (WCS request, cached as a GeoTIFF)."""
    f = LC_DIR / f"annual_nlcd_{LC_YEAR}_{g}.tif"
    x0, y0, x1, y1 = geom_albers.bounds
    x0, x1 = (LC_ORIGIN[0] + 30 * (fn((v - LC_ORIGIN[0]) / 30) + d) for v, fn, d in ((x0, np.floor, -3), (x1, np.ceil, 3)))     # snap the box
    y0, y1 = (LC_ORIGIN[1] + 30 * (fn((v - LC_ORIGIN[1]) / 30) + d) for v, fn, d in ((y0, np.floor, -3), (y1, np.ceil, 3)))     # to the 30 m grid
    fetch(LC_URL, f, resume=False, params=dict(service="WCS", version="1.0.0", request="GetCoverage", coverage=LC_LAYER, crs=ALBERS, format="GeoTIFF",
                                               bbox=f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}", width=int(round((x1 - x0) / 30)), height=int(round((y1 - y0) / 30)),
                                               time=f"{LC_YEAR}-01-01T00:00:00.000Z"))
    with rasterio.open(f) as ds:
        assert ds.crs.to_epsg() == 5070 and abs(ds.transform.a - 30) < 0.01 and abs(ds.transform.e + 30) < 0.01, "land cover is not on the 30 m Albers grid"
        return ds.read(1), ds.transform


# ---- GIS helpers -----------------------------------------------------------------------------------------------------
def cell_size(lat, step=1):
    """East-west and north-south length (m) of `step` arc-seconds at latitude lat (degrees) on the GRS80 ellipsoid."""
    p, d = np.radians(lat), np.radians(step / 3600)
    w = np.sqrt(1 - GEOD.es * np.sin(p) ** 2)
    return GEOD.a * np.cos(p) / w * d, GEOD.a * (1 - GEOD.es) / w ** 3 * d


def slope_grid(Z, lats, step=1):
    """Slope in m per km at every cell: central differences in metres, with the east-west spacing shrinking with latitude."""
    dx, dy = cell_size(lats, step)
    gy, gx = np.gradient(Z)
    return 1000 * np.hypot(gx / dx[:, None], gy / dy[:, None])


def area_mean(v, inside, lats):
    """Mean over the basin cells, each weighted by its true area (a cell covers less ground further north)."""
    ok = inside & np.isfinite(v)
    return float(np.average(v[ok], weights=np.repeat(np.cos(np.radians(lats))[:, None], v.shape[1], 1)[ok]))


def coarsen(Z, inside, lats, f):
    """Average the grid over blocks of f x f cells; a block belongs to the basin when at least half of it does."""
    n0, n1 = Z.shape[0] // f * f, Z.shape[1] // f * f
    blk = lambda a: a[:n0, :n1].reshape(n0 // f, f, n1 // f, f).mean((1, 3))
    return blk(Z), blk(inside.astype("float32")) >= 0.5, lats[:n0].reshape(-1, f).mean(1)


def hillshade(Z, lats, exag, smooth, az=315.0, alt=45.0):
    """Relief shading (light from the north-west): 1 on flat ground, above 1 on lit slopes, below 1 in shadow. Display only."""
    dx, dy = cell_size(lats)
    gy, gx = np.gradient(gaussian_filter(np.where(np.isfinite(Z), Z, np.nanmean(Z)), smooth))
    sx, sy = exag * gx / dx[:, None], -exag * gy / dy[:, None]                 # rise per metre towards east and towards north
    a, h = np.radians(az), np.radians(alt)
    return (-sx * np.sin(a) * np.cos(h) - sy * np.cos(a) * np.cos(h) + np.sin(h)) / np.sqrt(1 + sx ** 2 + sy ** 2) / np.sin(h)


def majority_share(G, inside, f, k=0):
    """Share of the basin lying in f x f blocks whose most common land-cover group is group k: what a coarse map would report."""
    n0, n1 = G.shape[0] // f * f, G.shape[1] // f * f
    cnt = np.stack([((G == j) & inside)[:n0, :n1].reshape(n0 // f, f, n1 // f, f).sum((1, 3)) for j in range(len(GROUPS))])
    return float(cnt.sum(0)[cnt.argmax(0) == k].sum() / cnt.sum())


def parts_of(geom):
    return list(getattr(geom, "geoms", [geom]))


GRID_NAME = {f: f"{round(f * cell_size(33.0)[1] / 10) * 10:d} m" for f in COARSER}        # 1 arc-second is 30.8 m from north to south


# ---- outlines --------------------------------------------------------------------------------------------------------
prj = SHP.with_suffix(".prj").read_text()
CRS_OUT = pyproj.CRS.from_wkt(prj)
assert CRS_OUT.is_geographic and "North_American_1983" in prj, "the outlines are expected in geographic NAD83 coordinates"
to_albers = pyproj.Transformer.from_crs(CRS_OUT, ALBERS, always_xy=True).transform
sf = shapefile.Reader(str(SHP))
want = {int(g): g for g in B.index}
OUT, FABRIC_ELEV, FABRIC_AREA = {}, {}, {}
for i, rec in enumerate(sf.iterRecords()):
    if int(rec["hru_id"]) in want:
        geom = shape(sf.shape(i).__geo_interface__)
        OUT[want[int(rec["hru_id"])]] = geom if geom.is_valid else geom.buffer(0)
        FABRIC_ELEV[want[int(rec["hru_id"])]], FABRIC_AREA[want[int(rec["hru_id"])]] = float(rec["elev_mean"]), float(rec["AREA"]) / 1e6
assert len(OUT) == len(B), "every study basin must have an outline"

# ---- statistics per basin --------------------------------------------------------------------------------------------
R, MAPS, tiles_all = {}, {}, set()
for g in B.index:
    geom, b = OUT[g], B.loc[g]
    area_m2, perim_m = (abs(v) for v in GEOD.geometry_area_perimeter(geom))
    hull = np.array(geom.convex_hull.exterior.coords)
    length_km = float(GEOD.inv(*np.repeat(hull, len(hull), 0).T, *np.tile(hull, (len(hull), 1)).T)[2].max()) / 1000     # longest straight line in the basin
    Z, lons, lats, tf, tiles = dem_window(geom.bounds)
    inside = geometry_mask([geom], Z.shape, tf, invert=True)                   # cells whose centre lies inside the outline
    tiles_all |= set(tiles)
    zin = Z[inside]
    S = slope_grid(Z, lats)
    slope_res, elev_res = {}, {}
    for f in COARSER:
        Zc, ic, lc_ = coarsen(Z, inside, lats, f) if f > 1 else (Z, inside, lats)
        slope_res[GRID_NAME[f]], elev_res[GRID_NAME[f]] = area_mean(slope_grid(Zc, lc_, f), ic, lc_), area_mean(Zc, ic, lc_)
    gauge = Point(b.gauge_lon, b.gauge_lat)
    near = nearest_points(geom.boundary, gauge)[0]
    gi, gj = int(round((lats[0] - b.gauge_lat) * 3600)), int(round((b.gauge_lon - lons[0]) * 3600))

    ga = shp_transform(to_albers, geom)
    L, ltf = landcover(g, ga)
    lmask = geometry_mask([ga], L.shape, ltf, invert=True)
    lin = L[lmask]
    cnt = np.bincount(lin, minlength=256)
    known = sum(cnt[c] for c in NLCD)
    share = {name: float(sum(cnt[c] for c in codes) / known) for name, codes in GROUPS.items()}
    lut = np.full(256, 255, "uint8")
    for k, codes in enumerate(GROUPS.values()):
        lut[list(codes)] = k
    smooth_perim = ga.simplify(SMOOTH_M).length                                # perimeter without the 30 m staircase of the outline
    Lgeo = np.full(Z.shape, 255, "uint8")                                      # the land-cover groups on the elevation grid, for the map only
    reproject(lut[L], Lgeo, src_transform=ltf, src_crs=ALBERS, dst_transform=tf, dst_crs=CRS_OUT.to_wkt(), resampling=Resampling.nearest, dst_nodata=255)

    R[g] = dict(
        short=flood.SHORT[g], landscape=b.landscape, n_outline_points=int(sum(len(p.exterior.coords) for p in parts_of(geom))),
        n_outline_parts=len(parts_of(geom)), largest_part_share=float(max(p.area for p in parts_of(geom)) / geom.area),
        area_km2=area_m2 / 1e6, area_albers_km2=ga.area / 1e6, area_landcover_cells_km2=float(len(lin) * 900 / 1e6), perimeter_km=perim_m / 1000,
        perimeter_smoothed_km=smooth_perim / 1000,
        longest_axis_km=length_km, compactness_gravelius=float(smooth_perim / (2 * np.sqrt(np.pi * area_m2))), compactness_gravelius_raw_outline=float(perim_m / (2 * np.sqrt(np.pi * area_m2))),
        elongation_ratio=float(2 * np.sqrt(area_m2 / np.pi) / 1000 / length_km),
        elev_mean_m=area_mean(Z, inside, lats), elev_min_m=float(np.nanmin(zin)), elev_max_m=float(np.nanmax(zin)), relief_m=float(np.nanmax(zin) - np.nanmin(zin)),
        elev_p02_m=float(np.nanpercentile(zin, 2)), elev_p98_m=float(np.nanpercentile(zin, 98)),
        slope_mean_m_per_km=slope_res[next(iter(slope_res))], slope_median_m_per_km=float(np.nanmedian(S[inside])), slope_mean_by_grid_m_per_km=slope_res, elev_mean_by_grid_m=elev_res,
        n_elevation_cells=int(inside.sum()), n_elevation_voids=int(np.isnan(zin).sum()), elevation_tiles=sorted(tiles),
        gauge=dict(lat=float(b.gauge_lat), lon=float(b.gauge_lon), elev_m=float(Z[gi, gj]), distance_to_outline_km=float(GEOD.inv(near.x, near.y, gauge.x, gauge.y)[2] / 1000),
                   inside_outline=bool(geom.contains(gauge))),
        land_cover_share=share, dominant_land_cover=max(share, key=share.get), forest_plus_woody_wetland_share=float(share["forest"] + cnt[90] / known),
        forest_share_1km_majority=majority_share(lut[L], lmask, KM_BLOCK),
        nlcd_class_share={f"{c} {NLCD[c]}": float(cnt[c] / known) for c in NLCD if cnt[c]}, n_landcover_cells=int(len(lin)),
        landcover_unclassified_share=float(1 - known / len(lin)), landcover_file=f"data/landcover/annual_nlcd_{LC_YEAR}_{g}.tif",
        camels=dict(area_gages2_km2=float(b.area_gages2), area_geospatial_fabric_km2=float(b.area_geospa_fabric), area_outline_file_km2=FABRIC_AREA[g], elev_mean_m=float(b.elev_mean),
                    elev_mean_outline_file_m=FABRIC_ELEV[g], slope_mean_m_per_km=float(b.slope_mean), frac_forest=float(b.frac_forest),
                    dom_land_cover=str(b.dom_land_cover).strip(), dom_land_cover_frac=float(b.dom_land_cover_frac)))
    r = R[g]
    r["difference"] = dict(area_vs_gages2_pct=100 * (r["area_km2"] / b.area_gages2 - 1), area_vs_geospatial_fabric_pct=100 * (r["area_km2"] / b.area_geospa_fabric - 1),
                           elev_mean_m=r["elev_mean_m"] - b.elev_mean, elev_mean_vs_outline_file_m=r["elev_mean_m"] - FABRIC_ELEV[g],
                           slope_mean_pct=100 * (r["slope_mean_m_per_km"] / b.slope_mean - 1), forest_share_points=100 * (share["forest"] - b.frac_forest),
                           forest_plus_woody_wetland_points=100 * (r["forest_plus_woody_wetland_share"] - b.frac_forest))
    MAPS[g] = dict(Z=Z, inside=inside, lons=lons, lats=lats, L=Lgeo)
    say(f"{flood.SHORT[g]:18s} area {r['area_km2']:7.1f} km2 ({r['difference']['area_vs_gages2_pct']:+5.1f}% vs CAMELS)  elevation {r['elev_min_m']:5.0f}-{r['elev_max_m']:5.0f} m, mean "
        f"{r['elev_mean_m']:6.1f} ({r['difference']['elev_mean_m']:+6.1f} m)  slope {r['slope_mean_m_per_km']:6.1f} m/km ({r['difference']['slope_mean_pct']:+5.0f}%)  "
        f"forest {100 * share['forest']:4.1f}% ({r['difference']['forest_share_points']:+5.1f} points)")
dem_tile.cache_clear()


# ---- our numbers against CAMELS --------------------------------------------------------------------------------------
def compare(ours, ref, unit, relative):
    """Agreement of one statistic over the 12 basins: differences (ours minus CAMELS; in % of CAMELS when relative) and rank agreement."""
    o, c = np.array([ours(R[g]) for g in R]), np.array([ref(R[g]) for g in R])
    d = 100 * (o / c - 1) if relative else o - c
    worst = int(np.argmax(np.abs(d)))
    return dict(unit_of_difference="% of the CAMELS value" if relative else unit, mean_difference=float(d.mean()), median_difference=float(np.median(d)),
                mean_absolute_difference=float(np.abs(d).mean()), largest_difference=float(d[worst]), largest_difference_basin=flood.SHORT[list(R)[worst]],
                rank_correlation=float(spearmanr(o, c)[0]), n_basins=len(d))


grid_names = list(next(iter(R.values()))["slope_mean_by_grid_m_per_km"])
CMP = dict(
    area_vs_gages2=compare(lambda r: r["area_km2"], lambda r: r["camels"]["area_gages2_km2"], "km2", True),
    area_vs_geospatial_fabric=compare(lambda r: r["area_km2"], lambda r: r["camels"]["area_geospatial_fabric_km2"], "km2", True),
    area_vs_outline_file=compare(lambda r: r["area_km2"], lambda r: r["camels"]["area_outline_file_km2"], "km2", True),
    elev_mean=compare(lambda r: r["elev_mean_m"], lambda r: r["camels"]["elev_mean_m"], "m", False),
    elev_mean_vs_outline_file=compare(lambda r: r["elev_mean_m"], lambda r: r["camels"]["elev_mean_outline_file_m"], "m", False),
    slope_mean_by_grid={n: compare(lambda r, n=n: r["slope_mean_by_grid_m_per_km"][n], lambda r: r["camels"]["slope_mean_m_per_km"], "m/km", True) for n in grid_names},
    forest_share=compare(lambda r: 100 * r["land_cover_share"]["forest"], lambda r: 100 * r["camels"]["frac_forest"], "percentage points", False),
    forest_share_1km_majority=compare(lambda r: 100 * r["forest_share_1km_majority"], lambda r: 100 * r["camels"]["frac_forest"], "percentage points", False),
    forest_plus_woody_wetland_share=compare(lambda r: 100 * r["forest_plus_woody_wetland_share"], lambda r: 100 * r["camels"]["frac_forest"], "percentage points", False))
CMP["slope_mean"] = CMP["slope_mean_by_grid"][grid_names[0]]
best_grid = min(grid_names, key=lambda n: CMP["slope_mean_by_grid"][n]["mean_absolute_difference"])
same_class = all((r["elev_mean_m"] > 600) == (r["camels"]["elev_mean_m"] > 600) for r in R.values())          # the landscape rule of 01_data_audit.py

# ---- figures ---------------------------------------------------------------------------------------------------------
W_IN, PANEL, HEAD, NAME, TOP, ROWGAP = 6.7, 1.5, 0.27, 0.40, 0.45, 0.14         # inches; 6.7 in = 170 mm, so font sizes are true when printed
GAP = (W_IN - 4 * PANEL) / 5
GAUGE = dict(marker="o", ms=4.8, mfc=SURFACE, mec=INK, mew=1.1, ls="none")


def panel_grid(title, bottom):
    """Twelve square map panels, one row per landscape (basins ordered by size); returns the figure, its height and {gauge id: axes}."""
    row = HEAD + NAME + PANEL + ROWGAP
    H = TOP + 3 * row - ROWGAP + bottom
    fig, axes = plt.figure(figsize=(W_IN, H)), {}
    fig.text(GAP / W_IN, 1 - 0.14 / H, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    fig.text(1 - (GAP + 0.16) / W_IN, 1 - 0.215 / H, "north is up in every map", fontsize=7.5, color=INK2, ha="right", va="center")
    fig.add_artist(Line2D([1 - (GAP + 0.05) / W_IN] * 2, [1 - 0.31 / H, 1 - 0.14 / H], color=INK2, lw=1.2, marker="^", markevery=[1], ms=4.5))      # north arrow
    for r, (land, short) in enumerate(flood.LAND_SHORT.items()):
        y = H - TOP - r * row
        fig.add_artist(Line2D([GAP / W_IN, 1 - GAP / W_IN], [(y + 0.03) / H] * 2, color=flood.LAND_COL[land], lw=1.6))
        fig.text(GAP / W_IN, (y - 0.03) / H, short, fontsize=9, fontweight="bold", color=INK, va="top")
        for c, g in enumerate(B[B.landscape == land].sort_values("area_gages2").index):
            axes[g] = fig.add_axes([(GAP + c * (PANEL + GAP)) / W_IN, (y - HEAD - NAME - PANEL) / H, PANEL / W_IN, PANEL / H])
    return fig, H, axes


def draw_basin(ax, g, img, line2, **kw):
    """One basin map: the image inside the outline, the gauge, a km scale bar and two lines of text above the panel."""
    m, geom = MAPS[g], OUT[g]
    x0, y0, x1, y1 = geom.bounds
    kx, ky = (v * 3.6 for v in cell_size((y0 + y1) / 2))                        # km per degree of longitude and latitude here
    side = 1.2 * max((x1 - x0) * kx, (y1 - y0) * ky)                           # the panel shows a square of this many km
    ax.imshow(img, extent=[m["lons"][0] - 0.5 / 3600, m["lons"][-1] + 0.5 / 3600, m["lats"][-1] - 0.5 / 3600, m["lats"][0] + 0.5 / 3600], aspect="auto", **kw)
    for p in parts_of(geom):
        ax.plot(*p.exterior.xy, color=INK2, lw=0.6)
    ax.plot(B.gauge_lon[g], B.gauge_lat[g], zorder=5, **GAUGE)
    xl, yb = (x0 + x1) / 2 - side / 2 / kx, (y0 + y1) / 2 - 0.565 * side / ky   # the basin sits a little high: room for the scale bar
    ax.set_xlim(xl, xl + side / kx); ax.set_ylim(yb, yb + side / ky); ax.axis("off")
    bar = max(v for v in (1, 2, 5, 10, 20, 50) if v <= 0.36 * side)
    ax.plot([xl + 0.02 * side / kx, xl + (0.02 * side + bar) / kx], [yb + 0.04 * side / ky] * 2, color=INK, lw=1.6, solid_capstyle="butt")
    ax.text(xl + (0.05 * side + bar) / kx, yb + 0.04 * side / ky, f"{bar} km", fontsize=7.5, color=INK2, va="center")
    ax.text(0, 1.155, flood.SHORT[g], transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=INK)
    ax.text(0, 1.035, line2, transform=ax.transAxes, fontsize=7.5, color=INK2)


# elevation: every panel is stretched from the basin's own lowest to its own highest point (the range is printed above it)
fig, H, axes = panel_grid("The 12 study basins: elevation inside each basin outline", bottom=0.86)
for g, ax in axes.items():
    m, r = MAPS[g], R[g]
    exag = float(np.clip(400 / (r["elev_p98_m"] - r["elev_p02_m"]), 1, 12))    # flat basins get a stronger shading so that valleys show
    rgba = SEQ_CMAP(np.nan_to_num((m["Z"] - r["elev_min_m"]) / r["relief_m"]).clip(0, 1))
    rgba[..., :3] = np.clip(rgba[..., :3] * np.clip(1 + 0.75 * (hillshade(m["Z"], m["lats"], exag, smooth=1.0 if exag == 1 else 3.0) - 1), 0.5, 1.08)[..., None], 0, 1)
    rgba[..., 3] = m["inside"]
    r["map_shading_exaggeration"] = exag
    draw_basin(ax, g, rgba, f"{r['area_km2']:,.0f} km²,  {r['elev_min_m']:,.0f} to {r['elev_max_m']:,.0f} m", interpolation="bilinear")
cax = fig.add_axes([GAP / W_IN, 0.38 / H, 0.42, 0.12 / H])
cb = fig.colorbar(ScalarMappable(cmap=SEQ_CMAP), cax=cax, orientation="horizontal")
cb.set_ticks([0, 1]); cb.set_ticklabels(["lowest point of the basin", "highest point"]); cb.ax.tick_params(labelsize=7.5, length=0); cb.outline.set_visible(False)
for t, ha in zip(cb.ax.get_xticklabels(), ("left", "right")):
    t.set_ha(ha)
fig.text(GAP / W_IN, 0.60 / H, "Elevation: each basin is coloured from its own lowest to its own highest point (the range in metres is above each map)", fontsize=7.5, color=INK2)
fig.legend([Line2D([], [], **GAUGE)], ["river gauge at the basin outlet"], loc="lower right", bbox_to_anchor=(1 - GAP / W_IN, 0.30 / H), frameon=False, fontsize=7.5, handletextpad=0.3)
fig.savefig(flood.FIG_OUT / "basin_maps.png", dpi=220); plt.close(fig)

# land cover: the same panels, one shared legend
fig, H, axes = panel_grid(f"The 12 study basins: land cover in {LC_YEAR}", bottom=0.72)
shown = [k for k in GROUPS if max(r["land_cover_share"][k] for r in R.values()) >= 0.005]      # a group nobody could see is left out of the legend
LC_WORD = {"forest": "forest", "shrub and grass": "shrub/grass", "cropland and pasture": "farmland", "wetland": "wetland", "developed": "built-up", "water": "water", "bare ground": "bare"}
LC_LEGEND = {"forest": "forest", "shrub and grass": "shrub and grass", "cropland and pasture": "farmland (crops, pasture)", "wetland": "wetland", "developed": "built-up (incl. roads)",
             "water": "water", "bare ground": "bare ground"}
for g, ax in axes.items():
    top2 = sorted(R[g]["land_cover_share"].items(), key=lambda kv: -kv[1])[:2]
    draw_basin(ax, g, np.ma.masked_where(~MAPS[g]["inside"] | (MAPS[g]["L"] == 255), MAPS[g]["L"]), ",  ".join(f"{LC_WORD[k]} {100 * v:.0f}%" for k, v in top2),
               cmap=ListedColormap(list(LC_COL.values())), vmin=-0.5, vmax=len(GROUPS) - 0.5, interpolation="nearest")
fig.legend([Patch(color=LC_COL[k]) for k in shown] + [Line2D([], [], **GAUGE)], [LC_LEGEND[k] for k in shown] + ["river gauge at the basin outlet"], loc="lower left",
           bbox_to_anchor=(GAP / W_IN / 2, 0.08 / H), ncol=4, frameon=False, fontsize=8, handlelength=1.2, handletextpad=0.5, columnspacing=1.6)
fig.savefig(flood.FIG_OUT / "basin_landcover.png", dpi=220); plt.close(fig)

# check: our numbers against the CAMELS table
coarse = grid_names[-1]
fig, axs = plt.subplots(2, 2, figsize=(W_IN, 6.5))
PANELS = [("Basin area (km²)", lambda r: r["area_km2"], lambda r: r["camels"]["area_gages2_km2"], True, CMP["area_vs_gages2"], "{:+.1f}%", (0.04, 0.96, "left", "top")),
          ("Mean elevation (m above sea level)", lambda r: r["elev_mean_m"], lambda r: r["camels"]["elev_mean_m"], False, CMP["elev_mean"], "{:+.0f} m", (0.04, 0.96, "left", "top")),
          ("Mean slope (m per km)", lambda r: r["slope_mean_m_per_km"], lambda r: r["camels"]["slope_mean_m_per_km"], True, CMP["slope_mean"], "{:+.0f}%", (0.04, 0.96, "left", "top")),
          ("Forest share of the basin (%)", lambda r: 100 * r["land_cover_share"]["forest"], lambda r: 100 * r["camels"]["frac_forest"], False, CMP["forest_share"], "{:+.0f} points", (0.04, 0.96, "left", "top"))]
for ax, (title, ours, ref, logax, c, fmt, (tx, ty, ha, va)) in zip(axs.flat, PANELS):
    o, k = np.array([ours(R[g]) for g in R]), np.array([ref(R[g]) for g in R])
    lo, hi = (min(o.min(), k.min()) / 1.6, max(o.max(), k.max()) * (4.5 if "slope" in title else 1.6)) if logax else (0, max(o.max(), k.max()) * 1.1)     # slope: room above the dots for the text
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, zorder=1)
    note = f"typical difference {fmt.format(c['median_difference'])}\nlargest {fmt.format(c['largest_difference'])} ({c['largest_difference_basin']})"
    for land, col in flood.LAND_COL.items():
        s = np.array([R[g]["landscape"] == land for g in R])
        ax.scatter(k[s], o[s], s=42, color=col, edgecolors=SURFACE, linewidths=1.0, zorder=3)
        if "slope" in title:                                                   # the same statistic on the coarsest grid, as rings
            ax.scatter(k[s], np.array([R[g]["slope_mean_by_grid_m_per_km"][coarse] for g in R])[s], s=34, facecolors=SURFACE, edgecolors=col, linewidths=1.3, zorder=2)
    if "slope" in title:
        note = f"dots: {grid_names[0]} grid,\ntypically {fmt.format(c['median_difference'])}\nrings: {coarse} grid,\ntypically {fmt.format(CMP['slope_mean_by_grid'][coarse]['median_difference'])}"
    if logax:
        ax.set_xscale("log"); ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")); axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.tick_params(labelsize=7.5)
    ax.set_title(title, fontsize=9.5); ax.set_xlabel("CAMELS attribute table", fontsize=8); ax.set_ylabel("our GIS statistic", fontsize=8)
    ax.text(tx, ty, note, transform=ax.transAxes, fontsize=7.5, color=INK2, ha=ha, va=va, linespacing=1.45)
handles = [Line2D([], [], marker="o", ls="none", ms=6.5, mfc=col, mec=SURFACE) for col in flood.LAND_COL.values()] + [Line2D([], [], color=MUTED, lw=1.0)]
fig.legend(handles, list(flood.LAND_SHORT.values()) + ["both agree"], loc="lower center", ncol=4, frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.8)
fig.suptitle("Our GIS statistics against the CAMELS attribute table, one dot per basin", fontsize=10.5, fontweight="bold", color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.98)); fig.savefig(flood.FIG_OUT / "basin_gis_check.png", dpi=220); plt.close(fig)

# ---- write -----------------------------------------------------------------------------------------------------------
C = CMP
V = list(R.values())
multi = [f"{r['short']} ({r['n_outline_parts']} parts, largest {100 * r['largest_part_share']:.2f}% of the area)" for r in V if r["n_outline_parts"] > 1]
elev_off = [f"{r['short']} {r['difference']['elev_mean_m']:+.0f} m" for r in V if abs(r["difference"]["elev_mean_m"]) > 25]
elev_grid_shift = max(abs(v - r["elev_mean_m"]) for r in V for v in r["elev_mean_by_grid_m"].values())
area_spread = max(100 * (max(r["area_km2"], r["area_albers_km2"], r["area_landcover_cells_km2"]) / min(r["area_km2"], r["area_albers_km2"], r["area_landcover_cells_km2"]) - 1) for r in V)
flat = [r["short"] for r in V if r["slope_median_m_per_km"] == 0]
gain = lambda a, b: C[b]["mean_absolute_difference"] - C[a]["mean_absolute_difference"]              # points by which a is nearer to CAMELS than b
closer = lambda a, b: "clearly closer to CAMELS than" if gain(a, b) > 2 else "further from CAMELS than" if gain(a, b) < -2 else "about as far from CAMELS as"
n_closer = sum(abs(r["forest_share_1km_majority"] - r["camels"]["frac_forest"]) < abs(r["land_cover_share"]["forest"] - r["camels"]["frac_forest"]) for r in V)    # basins where the coarse grid helps
bare = max(V, key=lambda r: r["land_cover_share"]["bare ground"])
J = dict(
    question="What do the 12 basins look like (elevation, land cover), and do GIS statistics computed by us agree with the CAMELS attribute table?",
    sources=dict(
        outlines=dict(file="data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp", coordinate_system="geographic NAD83 (read from the .prj file)", note="shipped with CAMELS-US; hru_id = gauge id"),
        elevation=dict(product="1-arc-second elevation tiles in SRTM .hgt format (Mapzen/Tilezen terrain tiles, AWS open data)", url=DEM_URL + "/N35/N35W083.hgt.gz (example)",
                       resolution="1 arc-second, about 30 m; heights in whole metres", year="SRTM was flown in February 2000; the tile set was built in 2016-2017", vertical_reference="EGM96 geoid (per the tile documentation)",
                       tiles=sorted(tiles_all), cache="data/dem/"),
        land_cover=dict(product="Annual NLCD land cover (USGS/MRLC), read from the USGS GeoServer that hosts the MRLC layers by WCS 1.0.0 GetCoverage", url=LC_URL, layer=LC_LAYER, year=LC_YEAR,
                        resolution="30 m, native grid in NAD83 / Conus Albers (EPSG:5070)", classes=NLCD, groups={k: list(v) for k, v in GROUPS.items()}, cache="data/landcover/"),
        camels=dict(file="data/camels_us/camels_topo.txt and camels_vege.txt", note="area_gages2, elev_mean, slope_mean and frac_forest are listed in camels_attributes_v2.0.xlsx as 'N15 - USGS data' "
                    "(Newman et al. 2015); the table gives neither the grid resolution nor the land-cover year behind them. dom_land_cover is from MODIS (IGBP classes).")),
    methods=dict(
        area="geodesic area of the outline on the GRS80 ellipsoid (pyproj.Geod); area_albers_km2 (planar area after projecting to an equal-area grid) and area_landcover_cells_km2 (30 m cells inside x 900 m2) are two cross-checks",
        elevation="cells of the 1-arc-second grid whose centre lies inside the outline; the mean weights each cell by cos(latitude), i.e. by its true area",
        slope="central differences on the 1-arc-second grid, with the cell size in metres computed per row from the latitude on the GRS80 ellipsoid; slope = 1000 x sqrt((dz/dx)^2 + (dz/dy)^2) in m per km, "
              "area-weighted mean over the basin. slope_mean_by_grid repeats this after averaging the grid over blocks of 3, 10 and 30 cells (elev_mean_by_grid likewise)",
        shape=f"compactness_gravelius = perimeter / perimeter of a circle of equal area (1 = circle), with the perimeter of the outline smoothed to {SMOOTH_M:.0f} m tolerance; elongation_ratio = diameter of a circle of equal area / "
              "longest straight line between two points of the outline (1 = circle, smaller = longer)",
        land_cover="30 m cells of the native Albers grid whose centre lies inside the outline (projected to the same grid); shares are cell counts, valid because the grid is equal-area. forest_share_1km_majority: the grid is cut "
                   f"into blocks of {KM_BLOCK} x {KM_BLOCK} cells ({KM_BLOCK * 30} m), each block takes its most common group, and the share of the basin in 'forest' blocks is reported",
        gauge="gauge position from CAMELS (gauge_lat, gauge_lon); distance_to_outline_km is the geodesic distance to the nearest point of the outline; elev_m is the elevation cell at the gauge",
        comparison="difference = ours minus CAMELS (in % of the CAMELS value for area and slope); rank_correlation is Spearman's, over the 12 basins"),
    basins=R, comparison_with_camels=CMP, landscape_classes_unchanged_with_our_elevation=bool(same_class),
    figures=dict(
        basin_maps=dict(file="figures/basin_maps.png", caption="Elevation inside each basin outline, one row per landscape, basins ordered by size. Colours are stretched PER BASIN: light = the lowest point of that basin, "
                        "dark = its highest point, and the range in metres is printed above each map. This shows the valleys of the flat Florida basins (a few tens of metres of relief) as clearly as those of the "
                        "Appalachian basins (more than 1,000 m), but a colour does NOT mean the same height in two panels. Relief shading (light from the north-west) is added for texture; in low-relief basins it is "
                        "exaggerated (factor 400 m / the basin's elevation range between its 2nd and 98th percentile, at most 12, on a smoothed grid), so shading strength must not be compared between panels either. "
                        "North is up in every map; each panel has its own km scale bar; the white dot is the river gauge."),
        basin_landcover=dict(file="figures/basin_landcover.png", caption=f"Land-cover groups in {LC_YEAR} (Annual NLCD, 30 m) inside each outline; the line above each map names the two largest groups and their share. "
                             "'Farmland' is NLCD pasture/hay plus cultivated crops; 'built-up' is the four NLCD developed classes, which include roads and open developed land. "
                             "North is up in every map; each panel has its own km scale bar; the white dot is the river gauge."),
        basin_gis_check=dict(file="figures/basin_gis_check.png", caption="Our statistics (vertical) against the CAMELS table (horizontal); a dot on the line means both agree. Area and slope use logarithmic axes. "
                             f"'Typical' is the median over the 12 basins of ours minus CAMELS. In the slope panel the dots are the 30 m grid and the rings the same basins on the {coarse} grid.")),
    notes=[
        f"Area: three ways of measuring the outline (geodesic, planar on the equal-area grid, counting 30 m cells) agree within {area_spread:.3f}%. Our area differs from the AREA field stored in the outline file by "
        f"{C['area_vs_outline_file']['mean_absolute_difference']:.3f}% on average, from CAMELS area_geospa_fabric by {C['area_vs_geospatial_fabric']['mean_absolute_difference']:.2f}% (largest "
        f"{C['area_vs_geospatial_fabric']['largest_difference']:+.1f}%, {C['area_vs_geospatial_fabric']['largest_difference_basin']}) and from CAMELS area_gages2 by {C['area_vs_gages2']['mean_absolute_difference']:.1f}% (largest "
        f"{C['area_vs_gages2']['largest_difference']:+.1f}%, {C['area_vs_gages2']['largest_difference_basin']}). area_gages2 comes from another data set (GAGES-II, with its own outline of the same basin), so a few percent there is a difference "
        "between sources, not an error of calculation.",
        f"Elevation: our mean differs from CAMELS elev_mean by {C['elev_mean']['median_difference']:+.1f} m in the median, but by more than 25 m in: {', '.join(elev_off) or 'no basin'}. Averaging our grid to coarser cells "
        f"(up to {grid_names[-1]}) moves our mean by at most {elev_grid_shift:.1f} m, so grid size does not explain this. The elev_mean field stored in the outline file is close to ours (mean absolute difference "
        f"{C['elev_mean_vs_outline_file']['mean_absolute_difference']:.1f} m, largest {C['elev_mean_vs_outline_file']['largest_difference']:+.1f} m, {C['elev_mean_vs_outline_file']['largest_difference_basin']}). We cannot tell from the "
        f"CAMELS documentation why its table is lower in these basins; the ranking of the basins is {'the same' if C['elev_mean']['rank_correlation'] > 0.9999 else 'nearly the same'} "
        f"(rank correlation {C['elev_mean']['rank_correlation']:.2f}) and the landscape classes of 01_data_audit.py (mountain = mean elevation above 600 m) are {'unchanged' if same_class else 'NOT the same'} with our elevations.",
        f"Slope: on the 30 m grid our mean slope is {C['slope_mean']['median_difference']:+.0f}% (median) against CAMELS slope_mean - several times steeper - with rank correlation {C['slope_mean']['rank_correlation']:.2f}. "
        "Slope depends strongly on the grid: " + ", ".join(f"{n}: {C['slope_mean_by_grid'][n]['median_difference']:+.0f}%" for n in grid_names) + f" (median difference to CAMELS). The grid closest to CAMELS is {best_grid}, "
        "which suggests - but does not prove - that CAMELS slope_mean comes from a grid of about 1 km or coarser; its documentation does not say. The absolute value depends on the grid it is measured on. The model reads "
        "slope_mean after standardising it over the basins, so what matters to it is how the basins compare with one another, and their order is nearly the same.",
        (f"In {', '.join(flat)} more than half of the 30 m cells have zero slope (heights are whole metres and the land is flat), so the mean slope there mostly reflects the 1 m steps of the data." if flat else
         "No basin has a median 30 m slope of zero."),
        f"Forest: our forest share (NLCD deciduous + evergreen + mixed forest, {LC_YEAR}) is {C['forest_share']['mean_difference']:+.1f} percentage points against CAMELS frac_forest on average (mean absolute "
        f"{C['forest_share']['mean_absolute_difference']:.1f}, largest {C['forest_share']['largest_difference']:+.1f}, {C['forest_share']['largest_difference_basin']}), rank correlation {C['forest_share']['rank_correlation']:.2f}. "
        f"Two possible reasons were tested. Definition: counting woody wetlands (forested swamps) as forest gives a mean absolute difference of {C['forest_plus_woody_wetland_share']['mean_absolute_difference']:.1f} points - "
        f"{closer('forest_plus_woody_wetland_share', 'forest_share')} the plain forest share. Resolution: on a {KM_BLOCK * 30} m grid where every block takes its most common group the forest share is "
        f"{C['forest_share_1km_majority']['mean_difference']:+.1f} points against CAMELS (mean absolute {C['forest_share_1km_majority']['mean_absolute_difference']:.1f}) - {closer('forest_share_1km_majority', 'forest_share')} "
        f"the 30 m share. It is not a full explanation: the coarse grid is closer to CAMELS in {n_closer} of {len(V)} basins and further away in the other {len(V) - n_closer}, and its largest difference is "
        f"{C['forest_share_1km_majority']['largest_difference']:+.1f} points ({C['forest_share_1km_majority']['largest_difference_basin']}). CAMELS does not document the land-cover map or year behind frac_forest; at 30 m, roads, clearings and pasture inside a forested basin are counted separately, which a coarser map cannot do.",
        "Land cover is the 2011 layer of the yearly USGS land-cover series (Annual NLCD, same class codes as NLCD), not the older single-year 'NLCD 2011' release: www.mrlc.gov could not be reached through the proxy; "
        "the MRLC layers were read from the USGS GeoServer at dmsdata.cr.usgs.gov (workspaces named mrlc_...), where WCS 2.0.1 failed with a server error and WCS 1.0.0 worked. 2011 lies inside the study period (1980-2014); land cover is treated as fixed although it changed over 34 years.",
        "Elevation tiles: the provider documents the tile set as a composite - SRTM (30 m) worldwide, with USGS 3DEP elevation used in the United States at finer zoom levels - and the tiles do not say which source a cell "
        "came from. We therefore call it '1-arc-second elevation in SRTM format', not pure SRTM.",
        "Datum: outlines are NAD83, elevation tiles WGS84; the two differ by about a metre horizontally, far below the 30 m cell, so no datum shift was applied.",
        f"Outline detail: the outline vertices are {1000 * np.mean([r['perimeter_km'] / r['n_outline_points'] for r in V]):.0f} m apart on average, i.e. the outline follows the edges of 30 m raster cells. This staircase makes "
        f"the raw perimeter {100 * (np.mean([r['perimeter_km'] / r['perimeter_smoothed_km'] for r in V]) - 1):.0f}% longer on average than the outline smoothed to {SMOOTH_M:.0f} m, which is why the compactness uses the smoothed perimeter.",
        ("Outlines with more than one part: " + "; ".join(multi) + ". All parts are used.") if multi else "Every outline is a single polygon.",
        f"Gauges: {sum(r['gauge']['inside_outline'] for r in V)} of {len(V)} gauges lie inside their outline, at most {max(r['gauge']['distance_to_outline_km'] for r in V):.2f} km from its edge, and the elevation at the gauge "
        f"is at most {max(r['gauge']['elev_m'] - r['elev_min_m'] for r in V):.0f} m above the lowest cell of the basin: the outlets are where they should be.",
        f"Elevation voids inside the basins: {sum(r['n_elevation_voids'] for r in V)} cells. Land-cover cells without a valid class: {max(r['landcover_unclassified_share'] for r in V):.4%} at most.",
        f"Land-cover groups: besides forest, shrub and grass, cropland and pasture, wetland, developed and water a seventh group, bare ground (NLCD 31, barren land), is kept so that the shares sum to 1; it is at most {bare['land_cover_share']['bare ground']:.1%} of a basin ({bare['short']}). "
        "In the figure 'cropland and pasture' is labelled farmland and 'developed' is labelled built-up; the keys in this file keep the longer names.",
        "The land-cover maps are resampled (nearest neighbour) to the elevation grid for drawing only; all shares come from the native 30 m Albers grid.",
        "figures/basin_gis_check.png is an addition to the two figures named in the plan.",
        "Nothing in this script is fitted, tuned or selected, so the train/validation/test split plays no role here.",
        "Smoke-test mode (FLOOD_QUICK=1) runs the same calculation - it is light - and only writes to the quick folders." if flood.QUICK else "Full run."],
    cpu_s=round(time.process_time() - C0, 1), runtime_s=round(time.perf_counter() - T0, 1))
(flood.RES_OUT / "basin_gis.json").write_text(json.dumps(J, indent=1))
say(json.dumps({k: v for k, v in CMP.items() if k != "slope_mean_by_grid"}, indent=1))
say("\n".join("- " + n for n in J["notes"]))
say(f"wrote {flood.RES_OUT / 'basin_gis.json'} and three figures in {flood.FIG_OUT}  ({J['runtime_s']} s, {J['cpu_s']} s CPU)")
