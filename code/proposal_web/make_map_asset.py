"""Phase C of code/proposal_web/ARTISTIC_PLAN.md: the custom map asset of screen 3 of the web presentation.

Writes, under code/proposal_web/assets/map/:
  basin_map.json  the land of the south-eastern USA, the 12 basin outlines with their gauges, rings and centroids, the scale bar, a north
                  arrow, and the crop boxes of the 12 elevation panels of figures/basin_maps.png (data only; no code)
  preview.html    basin_map.json drawn as an inline SVG by a few lines of JavaScript (the data is embedded, so it opens from disk)
  preview.png     a Playwright screenshot of preview.html (skipped with --no-shot)
Run from the project root:  ~/AI_Class/.venv/bin/python code/proposal_web/make_map_asset.py [--no-shot]
The output is deterministic. The script does not touch deck5.html or code/build_proposal_web.py.

It reads: the 12 CAMELS-US outlines (fetch_aorc.outlines(): HCDN_nhru_final_671.shp, geographic NAD83 = EPSG:4269), results/data_audit.json
(landscape, gauge position, CAMELS mean elevation, CAMELS area_gages2), results/flood_analysis.json (the short names), results/basin_gis.json
(outline area and elevation range from our own GIS statistics: the numbers printed above the panels of the figure), the Natural Earth 1:50 m land and
lake polygons that cartopy keeps under ~/.local/share/cartopy (geographic WGS 84, about a metre from NAD83), and the PNG figures/basin_maps.png
(its size and pixels, to check the crop boxes against the layout code of 07_basin_maps.py).

How the map is made. Every geometry is projected to EPSG:5070 (NAD83 / Conus Albers, equal-area; central meridian 96 W, so north is up only along
that meridian: the meridians lean about 7 degrees at the middle of this map, see "grid_north"). The frame is an axis-aligned rectangle of that
projection, FRAME_W x FRAME_H px at MPP metres per px, holding all 12 basins (80 km or more to spare on every side), Cape Hatteras and the Florida mainland
(the Keys are cut by the bottom edge). The land is the Natural Earth
land of the longitude -92..-74, latitude 24..38.5 box (its edges densified so the cut follows the meridians and parallels) united with the frame
and a margin, so the land reaches past the viewBox on every side and no cut edge can show. Natural Earth keeps barrier islands as slivers 1 to 4 km wide
beside the mainland; simplified as they are they turn into broken double lines, so the land is first CLOSED (buffer out and in by CLOSE_M, which joins
islands and lagoons narrower than twice that) and islets thinner than 2 x MIN_ISLET_RADIUS_M are dropped; then it is simplified (TOL_LAND_M, in projected
metres). Each basin outline is simplified with TOL_BASIN_M. Coordinates are written in tenths of a px as relative path commands.
"""
import argparse
import json
import math
import re
import sys
import time
from functools import reduce
from pathlib import Path

import numpy as np
import shapefile
import shapely
from PIL import Image
from pyproj import Geod, Proj, Transformer
from shapely import affinity
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import transform as shp_transform, unary_union

HERE = Path(__file__).resolve().parent                     # code/proposal_web
ROOT = HERE.parents[1]                                     # the project root
sys.path.insert(0, str(ROOT / "code"))
import fetch_aorc as AO                                    # ids_of_the_study(), outlines(ids): the 12 CAMELS outlines, as the project reads them

OUT = HERE / "assets" / "map"
NE_DIR = Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical"
NE_LAND, NE_LAKES = NE_DIR / "ne_50m_land.shp", NE_DIR / "ne_50m_lakes.shp"
FIGURE = ROOT / "figures" / "basin_maps.png"

# ---- the choices ----------------------------------------------------------------------------------------------------------------------
GEO_BOX = (-92.0, 24.0, -74.0, 38.5)       # lon_min, lat_min, lon_max, lat_max: the land covers this box (as asked in the plan)
FRAME_W, FRAME_H = 1000, 940               # the viewBox, px (SVG user units); north is up
MPP = 1500.0                               # metres of EPSG:5070 per px: 1000 px = 1,500 km
FRAME_X0, FRAME_Y0 = 413e3, 300e3          # lower-left corner of the frame, EPSG:5070 metres: the 12 basins (80 km to spare at the top, 180 or more elsewhere), Cape Hatteras, the Florida mainland
CLIP_MARGIN_M = 15e3                       # the land covers GEO_BOX united with the frame plus this margin
TOL_LAND_M, TOL_BASIN_M = 2000.0, 300.0    # simplification tolerances, projected metres
CLOSE_M = 3000.0                           # closing of the land before it is simplified: water gaps narrower than 2 x this are filled
MIN_ISLET_RADIUS_M = 1400.0                # islets whose widest inscribed circle is smaller than this (a strip under 2.8 km wide, under 2 px) are dropped
LAKE_MIN_RADIUS_M = 10e3                   # lakes are cut out of the land only when a circle this big fits in them (Lake Okeechobee, the one lake of that size in the box)
MIN_PART_PX2 = 0.2                         # basin parts and holes smaller than this (px^2; 1 px^2 = 2.25 km^2) are dropped as dust
MIN_ISLET_PX2, MIN_HOLE_PX2 = 1.0, 8.0     # land islets and lagoon holes smaller than this are dropped
Q = 10                                     # path coordinates are written in tenths of a px
RING_FACTOR, RING_MIN_PX = 1.5, 9.0        # the locator ring around a basin: 1.5 x the radius of its smallest enclosing circle, at least 9 px
SCALE_KM = 200
STROKE_BASIN = 2.0                         # px, as the plan asks (the preview draws the outlines with it)
LABEL_PX = 12.5                            # font size the label positions are computed for

LANDSCAPES = [                             # the order of the rows of figures/basin_maps.png (flood.LAND_SHORT); key = the deck's --land-m / -n / -s
    dict(key="m", name="Mountains (above 600 m)", short="Mountains", color="#13315c"),
    dict(key="n", name="Northern lowlands and Piedmont", short="Northern lowlands", color="#1b8a7a"),
    dict(key="s", name="Southern lowlands (south of 32.5N)", short="Southern lowlands", color="#c9a45c"),
]
CRS_NOTE = "drawn in EPSG:5070 (NAD83 / Conus Albers); source outlines geographic (NAD83)"
LAND_COL, COAST_COL = "#eef3f9", "#9fb3c8"

# the layout of figures/basin_maps.png, as code/07_basin_maps.py draws it (inches; dpi 220); checked against the pixels below
FIG = dict(W_IN=6.7, PANEL=1.5, HEAD=0.27, NAME=0.40, TOP=0.45, ROWGAP=0.14, BOTTOM=0.86, DPI=220)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- projection and frame ---------------------------------------------------------------------------------------------------------------
TR_GEO = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True)       # NAD83 geographic -> NAD83 / Conus Albers (the outlines and the gauges)
TR_NE = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)        # WGS 84 geographic -> the same (Natural Earth)
FRAME_Y1 = FRAME_Y0 + FRAME_H * MPP
FRAME_X1 = FRAME_X0 + FRAME_W * MPP


def to_px(geom):
    """EPSG:5070 metres -> viewBox px: x = (X - FRAME_X0) / MPP, y = (FRAME_Y1 - Y) / MPP (north up)."""
    return affinity.affine_transform(geom, [1 / MPP, 0, 0, -1 / MPP, -FRAME_X0 / MPP, FRAME_Y1 / MPP])


def px_xy(lon, lat):
    X, Y = TR_GEO.transform(lon, lat)
    return (X - FRAME_X0) / MPP, (FRAME_Y1 - Y) / MPP


def lonlat_box_ring(w, s, e, n, step=0.02):
    """The ring of a lon/lat box with its edges densified (a straight edge in lon/lat is an arc on the map)."""
    return [(lo, s) for lo in np.arange(w, e, step)] + [(e, la) for la in np.arange(s, n, step)] + \
           [(lo, n) for lo in np.arange(e, w, -step)] + [(w, la) for la in np.arange(n, s, -step)]


# ---- SVG path text ------------------------------------------------------------------------------------------------------------------------
def num(v):
    """A number of tenths as short text: 15 -> '1.5', 5 -> '.5', -20 -> '-2', 0 -> '0'."""
    i, f = divmod(abs(int(v)), Q)
    s = (str(i) if i or not f else "") + (f".{f}" if f else "")
    return ("-" if v < 0 else "") + (s or "0")


def ring_d(coords, min_pts=3):
    """One ring as 'Mx y' + relative 'l' steps + 'z', from absolute px coordinates quantised to tenths (no drift: the steps are differences of the
    rounded absolute positions); consecutive duplicates are removed; returns '' when fewer than min_pts distinct points remain."""
    pts = []
    for x, y in coords:
        p = (round(x * Q), round(y * Q))
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < min_pts:
        return ""
    s, prev = "", pts[0]
    for p in pts[1:]:
        for t in (num(p[0] - prev[0]), num(p[1] - prev[1])):
            s += t if (not s or t.startswith("-")) else " " + t
        prev = p
    return f"M{num(pts[0][0])}{'' if pts[0][1] < 0 else ' '}{num(pts[0][1])}l{s}z"


def polygons_of(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for part in geom.geoms for g in polygons_of(part)]


def geom_d(geom_px, min_area=MIN_PART_PX2, min_hole=MIN_PART_PX2):
    """A (multi)polygon in px as one path string (every ring a subpath; draw with fill-rule evenodd). Parts smaller than min_area px^2 and holes smaller
    than min_hole px^2 are dropped as dust. Returns (d, n_parts, n_points, dropped_parts)."""
    d, n_parts, n_pts, dropped = [], 0, 0, 0
    for poly in polygons_of(geom_px):
        if poly.area < min_area:
            dropped += 1
            continue
        for r in [poly.exterior.coords] + [h.coords for h in poly.interiors if Polygon(h).area >= min_hole]:
            s = ring_d(list(r))
            if s:
                d.append(s)
                n_pts += len(r) - 1
        n_parts += 1
    return "".join(d), n_parts, n_pts, dropped


def parse_d(d):
    """Read back what ring_d wrote (M absolute, l relative, z) into rings of px coordinates: the round-trip check of the writer."""
    rings, cur, pos = [], None, None
    toks = re.findall(r"[Mlz]|-?(?:\d+\.?\d*|\.\d+)", d)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "M":
            pos = (float(toks[i + 1]), float(toks[i + 2])); cur = [pos]; i += 3
        elif t == "l":
            i += 1
            while i < len(toks) and toks[i] not in "Mlz":
                pos = (pos[0] + float(toks[i]), pos[1] + float(toks[i + 1])); cur.append(pos); i += 2
        else:                                                   # z
            rings.append(cur); i += 1
    return rings


def evenodd_area(d):
    polys = [Polygon(r) for r in parse_d(d) if len(r) >= 3]
    return reduce(lambda a, b: a.symmetric_difference(b), polys).area if polys else 0.0


# ---- the land ----------------------------------------------------------------------------------------------------------------------------
def read_ne(path, pre):
    out = []
    for sh in shapefile.Reader(str(path)).shapes():
        g = shape(sh.__geo_interface__)
        if g.intersects(pre):
            out.append(g.intersection(pre))
    return out


def widest_circle_m(poly):
    """Radius (m) of the biggest circle that fits inside a polygon (half its greatest width)."""
    return shapely.maximum_inscribed_circle(poly, 250).length


def make_land():
    """Natural Earth 1:50 m land, closed, cut by the lakes, clipped, simplified, as one path. Returns (path, stats, land polygons in px)."""
    pre = box(GEO_BOX[0] - 3, GEO_BOX[1] - 3, GEO_BOX[2] + 3, GEO_BOX[3] + 3)         # generous: the clip proper is done on the map (densified box)
    land_m = shp_transform(TR_NE.transform, unary_union(read_ne(NE_LAND, pre)))
    n_raw = int(shapely.get_num_coordinates(land_m))
    land_m = land_m.buffer(CLOSE_M, join_style="round").buffer(-CLOSE_M, join_style="round")
    lakes = [p for g in read_ne(NE_LAKES, pre) for p in polygons_of(shp_transform(TR_NE.transform, g)) if widest_circle_m(p) >= LAKE_MIN_RADIUS_M]
    if lakes:
        land_m = land_m.difference(unary_union(lakes))
    box_m = Polygon([TR_NE.transform(lo, la) for lo, la in lonlat_box_ring(*GEO_BOX)])
    frame = box(FRAME_X0, FRAME_Y0, FRAME_X1, FRAME_Y1)
    clip = unary_union([box_m, frame.buffer(CLIP_MARGIN_M)])
    assert clip.contains(frame.buffer(CLIP_MARGIN_M / 2)), "the clip does not cover the frame"
    land_m = land_m.intersection(clip)
    assert not land_m.is_empty and frame.intersects(land_m)
    parts = polygons_of(land_m)
    thin = [p for p in parts if widest_circle_m(p) < MIN_ISLET_RADIUS_M]
    keep = unary_union([p for p in parts if widest_circle_m(p) >= MIN_ISLET_RADIUS_M])
    simp = keep.simplify(TOL_LAND_M, preserve_topology=True)
    land_px = to_px(simp)
    d, n_parts, n_pts, dropped = geom_d(land_px, min_area=MIN_ISLET_PX2, min_hole=MIN_HOLE_PX2)
    stats = dict(vertices_in_the_source=n_raw, polygons_after_closing_in_box=len(parts), thin_islets_dropped=len(thin), lakes_cut_out=len(lakes),
                 polygons_drawn=n_parts, dust_dropped=dropped, points_drawn=n_pts, valid=bool(simp.is_valid))
    assert abs(evenodd_area(d) - sum(p.area for p in polygons_of(land_px) if p.area >= MIN_ISLET_PX2)) / land_px.area < 0.01, "the land path does not reproduce the polygons"
    return d, stats, land_px


# ---- the basins --------------------------------------------------------------------------------------------------------------------------
def landscape_key(name):
    for ls in LANDSCAPES:
        if ls["name"] == name:
            return ls["key"]
    raise ValueError(f"unknown landscape {name!r}")


def make_basins():
    ids = AO.ids_of_the_study()
    polys = AO.outlines(ids)                                                    # NAD83 geographic polygons keyed by gauge id
    DA = json.loads((AO.RES / "data_audit.json").read_text())
    info = {s["gauge_id"]: s for s in DA["selected"]}
    FA = json.loads((AO.RES / "flood_analysis.json").read_text())["basins"]
    GIS = json.loads((AO.RES / "basin_gis.json").read_text())["basins"]
    geod = Geod(ellps="GRS80")
    out, checks = [], []
    for g in ids:
        s = info[g]
        full_m = shp_transform(TR_GEO.transform, polys[g])
        area_geod = abs(geod.geometry_area_perimeter(polys[g])[0]) / 1e6
        assert abs(area_geod - GIS[g]["area_km2"]) < 0.01, f"{g}: the outline area differs from results/basin_gis.json"
        simp_px = to_px(full_m.simplify(TOL_BASIN_M, preserve_topology=True))
        d, n_parts, n_pts, dropped = geom_d(simp_px)
        # what the writer wrote must be the polygon it was given (quantisation to 0.1 px and the dust apart)
        clean = unary_union([p for p in polygons_of(simp_px) if p.area >= MIN_PART_PX2])
        a_path, a_poly = evenodd_area(d), clean.area
        assert abs(a_path - a_poly) / a_poly < 0.01, f"{g}: the path does not reproduce the polygon ({a_path:.2f} vs {a_poly:.2f} px2)"
        full_px = to_px(full_m)
        cen = full_px.centroid
        c = shapely.centroid(shapely.minimum_bounding_circle(clean))
        r_mbc = float(shapely.minimum_bounding_radius(clean))
        gx, gy = px_xy(s["gauge_lon"], s["gauge_lat"])
        checks.append((g, FA[g]["short"], full_px.boundary.distance(Point(gx, gy)) * MPP / 1e3))
        out.append(dict(
            id=g, name=FA[g]["short"], landscape=s["landscape"], landscape_key=landscape_key(s["landscape"]),
            area_km2=round(float(GIS[g]["area_km2"]), 1), area_gages2=round(float(s["area_gages2"]), 1),
            elev_mean=float(s["elev_mean"]), elev_min=float(GIS[g]["elev_min_m"]), elev_max=float(GIS[g]["elev_max_m"]),
            elev_mean_gis=round(float(GIS[g]["elev_mean_m"]), 1),
            path=d, gauge=[round(gx, 1), round(gy, 1)], gauge_lonlat=[float(s["gauge_lon"]), float(s["gauge_lat"])],
            centroid=[round(cen.x, 1), round(cen.y, 1)],
            ring=[round(c.x, 1), round(c.y, 1), round(max(RING_FACTOR * r_mbc, RING_MIN_PX), 1)],
            extent_px=[round(v, 1) for v in simp_px.bounds],
            _stats=dict(points=n_pts, parts=n_parts, dropped_parts=dropped, enclosing_radius_px=round(r_mbc, 1)),
        ))
    return out, checks


# ---- the panels of figures/basin_maps.png ---------------------------------------------------------------------------------------------
def panel_grid(basins):
    """Crop boxes [left, top, right, bottom] (px of the PNG) of the 12 elevation panels, from the layout code of 07_basin_maps.py (twelve square axes,
    one row per landscape in the order of LANDSCAPES, four per row ordered by CAMELS area_gages2), then CHECKED on the pixels: the panels hold map
    pixels, and a 3 px frame just outside each box is background."""
    f = FIG
    gap = (f["W_IN"] - 4 * f["PANEL"]) / 5
    row = f["HEAD"] + f["NAME"] + f["PANEL"] + f["ROWGAP"]
    h_in = f["TOP"] + 3 * row - f["ROWGAP"] + f["BOTTOM"]
    dpi = f["DPI"]
    im = Image.open(FIGURE).convert("RGB")
    W_PX, H_PX = im.size
    assert (W_PX, H_PX) == (round(f["W_IN"] * dpi), round(h_in * dpi)), f"{FIGURE} is {im.size}, the layout of 07_basin_maps.py gives {round(f['W_IN'] * dpi)} x {round(h_in * dpi)}"
    a = np.asarray(im).astype(int)
    bg = tuple(int(v) for v in a[2, 2])
    ink = np.abs(a - np.array(bg)).sum(axis=2) > 24                                  # True where the pixel is not the figure's surface colour
    side = round(f["PANEL"] * dpi)
    boxes, titled, rows = {}, {}, []
    for r, ls in enumerate(LANDSCAPES):
        members = sorted((b for b in basins if b["landscape_key"] == ls["key"]), key=lambda b: b["area_gages2"])
        assert len(members) == 4, ls["name"]
        rows.append(dict(landscape=ls["name"], key=ls["key"], ids=[b["id"] for b in members]))
        for c, b in enumerate(members):
            l = round((gap + c * (f["PANEL"] + gap)) * dpi)
            t = round((f["TOP"] + r * row + f["HEAD"] + f["NAME"]) * dpi)
            bx = [l, t, l + side, t + side]
            inside = ink[bx[1]:bx[3], bx[0]:bx[2]]
            assert inside.mean() > 0.10, f"{b['id']}: the panel box holds almost no map pixels ({inside.mean():.3f})"
            for strip in (ink[bx[1] - 3:bx[1], bx[0]:bx[2]], ink[bx[3]:bx[3] + 3, bx[0]:bx[2]], ink[bx[1]:bx[3], bx[0] - 3:bx[0]], ink[bx[1]:bx[3], bx[2]:bx[2] + 3]):
                assert not strip.any(), f"{b['id']}: map or text pixels touch the outside of the panel box {bx}"
            band = ink[bx[1] - round(f["NAME"] * dpi):bx[1] - 3, bx[0]:bx[2]]       # the two text lines above the panel (name; area and elevation range)
            top_txt = bx[1] - round(f["NAME"] * dpi) + int(np.argmax(band.any(axis=1)))
            boxes[b["id"]] = bx
            titled[b["id"]] = [bx[0], top_txt - 3, bx[2], bx[3]]
    return dict(
        image="figures/basin_maps.png", size=[W_PX, H_PX], dpi=dpi, background="#%02x%02x%02x" % bg, rows=rows,
        order="three rows (the landscapes, top to bottom) of four panels; within a row the basins run left to right by increasing CAMELS area_gages2",
        boxes=boxes,
        boxes_with_title=titled,
        footer_box=[0, round((h_in - f["BOTTOM"]) * dpi), W_PX, H_PX],
        note="boxes: [left, top, right, bottom] in px of the PNG, the square map panel only (330 x 330 px; the panel has its own km scale bar inside the square); "
             "boxes_with_title: the same with the two text lines above it (the name; the area and elevation range); footer_box: the colour bar and the key at the foot. "
             "Colours are stretched PER BASIN (light = its lowest point, dark = its highest), so a colour does not mean the same height in two panels.",
    )


# ---- scale bar, north arrow, labels --------------------------------------------------------------------------------------------------------
def scale_and_north():
    """The 200 km bar (px from the projection's metres per px), the north arrow, and what the projection does to scale and direction over the frame."""
    bar_px = SCALE_KM * 1e3 / MPP
    xs, ys = np.linspace(FRAME_X0, FRAME_X1, 21), np.linspace(FRAME_Y0, FRAME_Y1, 21)
    X, Y = np.meshgrid(xs, ys)
    lon, lat = TR_GEO.transform(X, Y, direction="INVERSE")
    fac = Proj("EPSG:5070").get_factors(lon, lat)
    kp, conv = np.asarray(fac.parallel_scale), np.asarray(fac.meridian_convergence)
    clon, clat = TR_GEO.transform((FRAME_X0 + FRAME_X1) / 2, (FRAME_Y0 + FRAME_Y1) / 2, direction="INVERSE")
    kc = Proj("EPSG:5070").get_factors(clon, clat)
    worst = max(abs(kp.max() - 1), abs(kp.min() - 1))
    return (dict(x=26, y=FRAME_H - 34, px=round(bar_px, 2), km=SCALE_KM, metres_per_px=MPP,
                 anchor="left end of the bar: the bar runs right from (x, y), with end ticks 6 px high; label it '%d km'" % SCALE_KM,
                 parallel_scale_over_the_frame=[round(float(kp.min()), 4), round(float(kp.max()), 4)],
                 note="px = %d km / metres_per_px, in the projected plane. EPSG:5070 is equal-area, not conformal: over this frame the length scale along a parallel runs from %.3f to %.3f "
                      "(along a meridian it is the reciprocal), so the bar is true to within %.1f%% anywhere on this map" % (SCALE_KM, kp.min(), kp.max(), 100 * worst)),
            dict(x=FRAME_W - 44, y=88, len=52, note="tail at (x, y), tip at (x, y - len): points to the top of the page (grid north)"),
            dict(convergence_deg=round(float(kc.meridian_convergence), 1), at_lonlat=[round(float(clon), 2), round(float(clat), 2)],
                 convergence_range_deg=[round(float(conv.min()), 1), round(float(conv.max()), 1)],
                 note="north is up on the page (grid north). True north leans this many degrees to the WEST of 'up' at the middle of the map "
                      "(the meridian convergence of EPSG:5070, central meridian 96 W); over the frame it runs from %.1f to %.1f degrees" % (conv.min(), conv.max())))


def place_labels(basins, land_px):
    """A greedy label placement beside each ring (optional use): the first of 16 directions whose text box meets no ring, no earlier label, no coast line and no
    frame edge (a second pass lets it cross the coast). Returns {id: {x, y, anchor}} for SVG text at LABEL_PX (x, y = the text anchor and baseline)."""
    rings = [tuple(b["ring"]) for b in basins]
    coast = unary_union([p.boundary for p in polygons_of(land_px)])
    order = sorted(range(len(basins)), key=lambda i: -sum(math.hypot(rings[i][0] - r[0], rings[i][1] - r[1]) < 90 for r in rings))   # the most crowded first
    placed, res = [], {}

    def hits_ring(rect, circ):
        x0, y0, x1, y1 = rect
        cx, cy, r = circ
        dx, dy = max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)
        return dx * dx + dy * dy < (r + 1.5) ** 2

    for i in order:
        b = basins[i]
        cx, cy, r = b["ring"]
        w = 0.58 * LABEL_PX * len(b["name"])
        for avoid_coast in (True, False):
            best = None
            for dist in (6, 12, 20, 30):
                for k in range(16):
                    ang = math.radians(k * 22.5)
                    tx, ty = cx + (r + dist) * math.cos(ang), cy - (r + dist) * math.sin(ang)
                    anchor = "start" if math.cos(ang) > 0.3 else ("end" if math.cos(ang) < -0.3 else "middle")
                    x0 = tx if anchor == "start" else (tx - w if anchor == "end" else tx - w / 2)
                    rect = (x0, ty - LABEL_PX * 0.8, x0 + w, ty + LABEL_PX * 0.3)
                    if rect[0] < 4 or rect[2] > FRAME_W - 4 or rect[1] < 4 or rect[3] > FRAME_H - 4:
                        continue
                    if any(hits_ring(rect, c) for c in rings) or any(not (rect[2] < p[0] or p[2] < rect[0] or rect[3] < p[1] or p[3] < rect[1]) for p in placed):
                        continue
                    if avoid_coast and box(*rect).intersects(coast):
                        continue
                    best = (round(tx, 1), round(ty, 1), anchor, rect)
                    break
                if best:
                    break
            if best:
                placed.append(best[3])
                res[b["id"]] = dict(x=best[0], y=best[1], anchor=best[2])
                break
    return res


# ---- the JSON -------------------------------------------------------------------------------------------------------------------------------
def dumps(obj):
    """json with indent, then every list of plain numbers on one line."""
    s = json.dumps(obj, indent=1, ensure_ascii=False)
    return re.sub(r"\[\s+(-?[\d.eE+-]+(?:,\s+-?[\d.eE+-]+)*)\s+\]", lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", s)


def build():
    t0 = time.time()
    land_d, land_stats, land_px = make_land()
    say(f"land: {land_stats}; path {len(land_d):,} characters")
    basins, checks = make_basins()
    for g, name, gap in checks:
        say(f"  gauge {g} {name}: {gap:.2f} km from its outline")
        assert gap < 3.0, f"{g}: the gauge is {gap:.1f} km from the outline (a wrong id or outline?)"
    labels = place_labels(basins, land_px)
    for b in basins:
        b["label"] = labels.get(b["id"])
    say(f"labels placed: {len(labels)} of {len(basins)}")
    bar, north, grid_north = scale_and_north()
    pg = panel_grid(basins)
    areas = [b["area_km2"] for b in basins]
    stats = {b["id"]: b.pop("_stats") for b in basins}
    data = dict(
        viewBox=[0, 0, FRAME_W, FRAME_H],
        crs=CRS_NOTE,
        crs_land="the Natural Earth land is geographic WGS 84 (EPSG:4326), projected to EPSG:5070 the same way; WGS 84 and NAD83 differ by about a metre here",
        scale_bar=bar,
        north_arrow=north,
        grid_north=grid_north,
        transform=dict(frame_epsg5070_m=[FRAME_X0, FRAME_Y0, FRAME_X1, FRAME_Y1], metres_per_px=MPP,
                       formula="x_px = (X - frame_x_min) / metres_per_px; y_px = (frame_y_max - Y) / metres_per_px, with X, Y in EPSG:5070 metres "
                               "(Transformer.from_crs('EPSG:4269', 'EPSG:5070', always_xy=True)); frame_epsg5070_m = [x_min, y_min, x_max, y_max]"),
        land=land_d,
        land_note=f"Natural Earth 1:50 m land of the lon {GEO_BOX[0]:g}..{GEO_BOX[2]:g}, lat {GEO_BOX[1]:g}..{GEO_BOX[3]:g} box united with the frame plus {CLIP_MARGIN_M / 1e3:g} km, "
                  f"so it reaches past the viewBox on every side (the deck can move or shrink the viewBox without a gap); closed by {CLOSE_M / 1e3:g} km (narrow lagoons "
                  f"and inlets filled), islets under 2.8 km wide dropped, simplified with a {TOL_LAND_M / 1e3:g} km tolerance; Lake Okeechobee is cut out; draw with fill-rule evenodd",
        simplification_m=dict(land=TOL_LAND_M, basins=TOL_BASIN_M, land_closing=CLOSE_M),
        colours=dict(land=LAND_COL, coast=COAST_COL, stroke_basin_px=STROKE_BASIN, coast_stroke_px=0.8),
        landscapes=LANDSCAPES,
        basins=basins,
        panel_grid=pg,
        caption=f"Map {CRS_NOTE}. Outlines: CAMELS-US (HCDN_nhru_final_671); coast: Natural Earth 1:50 m land. The basins cover {min(areas):,.0f} to {max(areas):,.0f} km², "
                f"so each is drawn with a heavier line and a ring to be seen.",
        sources=["CAMELS-US basin outlines (HCDN_nhru_final_671.shp, geographic NAD83)", "Natural Earth 1:50 m land and lakes (public domain)",
                 "gauge positions and landscape: results/data_audit.json (CAMELS gauge_lat, gauge_lon)", "area and elevation range: results/basin_gis.json (our GIS statistics, 1-arc-second elevation)"],
        notes=[f"area_km2 is the geodesic area of the outline (GRS80), the number printed above each panel and the one behind '{min(areas):,.0f} to {max(areas):,.0f} km²' in the text; "
               f"area_gages2 is the CAMELS table's value (what DATA.basins['area'] rounds): the two differ by up to {max(abs(b['area_km2'] - b['area_gages2']) for b in basins):.1f} km²",
               "elev_mean is the CAMELS table's value (results/data_audit.json); elev_min, elev_max and elev_mean_gis are our own GIS statistics (results/basin_gis.json), the range printed above each panel; "
               f"the two means differ by up to {max(abs(b['elev_mean'] - b['elev_mean_gis']) for b in basins):.0f} m ({max(basins, key=lambda b: abs(b['elev_mean'] - b['elev_mean_gis']))['name']})",
               "ring = [cx, cy, r]: a locator ring centred on the smallest circle that holds the outline, 1.5 x its radius (at least 9 px)",
               f"label = {{x, y, anchor}}: an optional placement of the short name as SVG text at {LABEL_PX:g} px (x, y = anchor point and baseline); null if none fitted",
               "extent_px = [x_min, y_min, x_max, y_max] of the simplified outline; gauge and centroid are in px; gauge_lonlat is [lon, lat] (NAD83)"],
        stats=dict(land=land_stats, basins=stats),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "basin_map.json").write_text(dumps(data) + "\n")
    say(f"wrote {OUT / 'basin_map.json'} ({(OUT / 'basin_map.json').stat().st_size / 1e3:.1f} kB) in {time.time() - t0:.1f} s")
    return data


# ---- the preview page -----------------------------------------------------------------------------------------------------------------------
PREVIEW = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Basin map preview</title>
<style>
@font-face{font-family:"Inter";src:url("../inter-latin-wght-normal.woff2") format("woff2");font-weight:100 900;font-display:block}
:root{--paper:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e}
*{box-sizing:border-box}
body{margin:0;padding:16px 20px 24px;background:var(--paper);color:var(--ink);font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{width:1000px;margin:0 auto}
.top{font-size:12px;color:var(--ink2);margin:0 0 8px;letter-spacing:.02em}
#stage svg{display:block;width:1000px;height:auto;background:#fff;border-radius:12px;box-shadow:0 1px 2px rgba(11,37,69,.10),0 8px 24px rgba(11,37,69,.10)}
.cap{font-size:13px;line-height:1.45;color:var(--ink2);margin:12px 4px 0}
.cap b{color:var(--ink)}
text{font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
</style></head><body>
<div class="wrap">
<p class="top">Preview of basin_map.json, drawn by make_map_asset.py (not part of the deck). Hover a basin for a tooltip.</p>
<div id="stage"></div>
<p class="cap" id="cap"></p>
</div>
<script id="map-data" type="application/json">__JSON__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('map-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const [, , W, H] = D.viewBox;
  const svg = el('svg', {viewBox: D.viewBox.join(' '), role: 'img', 'aria-label': 'Map of the twelve study basins in the south-eastern United States, in three landscape colours'});
  document.getElementById('stage').appendChild(svg);
  const col = {}; D.landscapes.forEach(l => col[l.key] = l.color);
  const INK = '#0b2545', C = D.colours, fmt = v => v.toLocaleString('en');
  el('path', {d: D.land, 'fill-rule': 'evenodd', fill: C.land, stroke: C.coast, 'stroke-width': C.coast_stroke_px, 'stroke-linejoin': 'round'}, svg);
  D.basins.forEach(b => {
    const c = col[b.landscape_key];
    const g = el('g', {'data-id': b.id, 'class': 'basin'}, svg);
    el('title', {}, g).textContent = b.name + ': ' + fmt(b.area_km2) + ' km2, ' + fmt(b.elev_min) + ' to ' + fmt(b.elev_max) + ' m (' + b.landscape + ')';
    el('circle', {cx: b.ring[0], cy: b.ring[1], r: b.ring[2], fill: 'none', stroke: c, 'stroke-width': 1.1, 'stroke-opacity': 0.55}, g);
    el('path', {d: b.path, 'fill-rule': 'evenodd', fill: c, 'fill-opacity': 0.9, stroke: c, 'stroke-width': C.stroke_basin_px, 'stroke-linejoin': 'round'}, g);
    el('circle', {cx: b.gauge[0], cy: b.gauge[1], r: 2.6, fill: '#fff', stroke: INK, 'stroke-width': 1}, g);
  });
  D.basins.forEach(b => {
    if (!b.label) return;
    el('text', {x: b.label.x, y: b.label.y, 'text-anchor': b.label.anchor, 'font-size': 12.5, 'font-weight': 600, fill: INK, stroke: '#fff', 'stroke-width': 3.2, 'paint-order': 'stroke', 'stroke-linejoin': 'round'}, svg).textContent = b.name;
  });
  const s = D.scale_bar, sb = el('g', {}, svg);                              // scale bar
  el('path', {d: `M${s.x} ${s.y - 3}v6M${s.x + s.px} ${s.y - 3}v6M${s.x} ${s.y}h${s.px}`, stroke: INK, 'stroke-width': 1.6, fill: 'none'}, sb);
  el('text', {x: s.x + s.px / 2, y: s.y - 9, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 600, fill: INK}, sb).textContent = s.km + ' km';
  const n = D.north_arrow, na = el('g', {}, svg);                           // north arrow
  el('path', {d: `M${n.x} ${n.y}V${n.y - n.len}`, stroke: INK, 'stroke-width': 1.6, fill: 'none'}, na);
  el('path', {d: `M${n.x} ${n.y - n.len - 1}l-6 14h12z`, fill: INK}, na);
  el('text', {x: n.x, y: n.y - n.len - 8, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: INK}, na).textContent = 'N';
  const lg = el('g', {transform: `translate(${s.x} ${s.y - 116})`}, svg);   // legend, above the scale bar
  D.landscapes.forEach((l, i) => {
    el('rect', {x: 0, y: i * 22 - 6, width: 22, height: 12, rx: 3, fill: l.color}, lg);
    el('text', {x: 32, y: i * 22 + 4, 'font-size': 13, fill: INK}, lg).textContent = l.name;
  });
  el('circle', {cx: 11, cy: 3 * 22, r: 2.6, fill: '#fff', stroke: INK, 'stroke-width': 1}, lg);
  el('text', {x: 32, y: 3 * 22 + 4, 'font-size': 13, fill: INK}, lg).textContent = 'river gauge at the basin outlet';
  document.getElementById('cap').innerHTML = D.caption.replace(/^Map ([^.]*)\./, '<b>Map $1.</b>');
})();
</script></body></html>
"""


def write_preview(data):
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (OUT / "preview.html").write_text(PREVIEW.replace("__JSON__", js))
    say(f"wrote {OUT / 'preview.html'}")


def shoot():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        pg = br.new_page(viewport={"width": 1060, "height": 1100}, device_scale_factor=1.5)
        errs = []
        pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
        pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
        pg.goto((OUT / "preview.html").resolve().as_uri())
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(400)
        pg.screenshot(path=str(OUT / "preview.png"), full_page=True)
        br.close()
    say(f"wrote {OUT / 'preview.png'}; page errors: {errs or 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shot", action="store_true", help="write the JSON and the preview page but no screenshot")
    args = ap.parse_args()
    data = build()
    write_preview(data)
    if not args.no_shot:
        shoot()
