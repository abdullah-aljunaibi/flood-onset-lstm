"""Week 5 - satellite data: daily satellite rain (GPM IMERG) and soil wetness / precipitable water (MERRA-2) for the 12 basins.

Source: the NASA POWER daily API, which needs no login. The script answers four questions before the data are used:
1. Which grid does POWER serve? Points a few hundredths of a degree apart are requested; where the series change is the cell edge.
2. Basin averages: every 0.1 degree IMERG cell that overlaps a basin outline is requested and averaged with the overlap area as
   weight. The MERRA-2 variables come on a coarse grid (0.5 x 0.625 degree): one request at the basin centroid.
3. Which day does a value belong to? IMERG is served by UTC day only; lag correlations with the CAMELS (Daymet) rain show the offset.
4. How good is the satellite rain? Correlation, bias and heavy-rain days against the gauge-based Daymet rain, basin by basin.
Every raw answer is cached under data/satellite/raw/ and never requested twice, so the script can be rerun offline.
Writes data/satellite/<gauge_id>.csv (one row per day), results/satellite_data.json and figures/satellite_rain_check.png.
Other scripts read the tables with: from fetch_satellite import satellite_table
"""
import html
import json
import re
import time

import numpy as np
import pandas as pd

from common import *

SAT, RAW = DATA / "satellite", DATA / "satellite" / "raw"
API = "https://power.larc.nasa.gov/api/temporal/daily/point"
CATALOGUE = "https://power.larc.nasa.gov/api/system/manager/parameters"
DOCS = "https://power.larc.nasa.gov/docs/methodology/data/sources/"
IMERG_SPAN = ("19980101", "20141231")       # asked from 1998 to learn where the series really starts (the plan expected 2000)
MERRA_SPAN = ("19810101", "20141231")       # POWER serves MERRA-2 from 1981-01-01
MERRA_PARS = {"PRECTOTCORR": "merra_prcp", "GWETTOP": "gwettop", "GWETROOT": "gwetroot", "GWETPROF": "gwetprof", "TQV": "pw", "PW": "pw_ceres"}
PAUSE, FILL = 1.5, -900.0                   # seconds between requests; POWER marks missing values with -999
SHAPES = DATA / "camels_us" / "basin_set_full_res" / "HCDN_nhru_final_671.shp"
PROBE_SPAN = ("20090101", "20101231")
PROBE_AT = (-82.85, 35.15)                  # centre of the 0.1 degree cell that holds the French Broad gauge
PROBE_LON = [-82.89, -82.85, -82.81, -82.79, -82.75, -82.71, -82.69, -82.65]      # along latitude 35.15
PROBE_LAT = [35.11, 35.15, 35.19, 35.21, 35.23, 35.27, 35.29, 35.31]              # along longitude -82.85
KIND = {"imerg": "satellite rain estimate (GPM IMERG; microwave and infrared sensors merged; mm/day)",
        "merra_prcp": "reanalysis rain (MERRA-2 bias-corrected total precipitation; mm/day) - a weather model, not a satellite retrieval",
        "gwettop": "reanalysis soil wetness of the top 5 cm (MERRA-2; 0 = dry, 1 = saturated) - a weather model's land surface, not a satellite retrieval",
        "gwetroot": "reanalysis soil wetness of the root zone, about the top 1 m (MERRA-2; 0-1) - not a satellite retrieval",
        "gwetprof": "reanalysis soil wetness of the whole profile down to bedrock (MERRA-2; 0-1) - not a satellite retrieval",
        "pw": "reanalysis precipitable water (MERRA-2 TQV; kg per m2 = mm of water in the air column) - a weather model that assimilates satellite observations",
        "pw_ceres": "precipitable water as served with the CERES SYN1deg product (cm; 1 degree grid; valid from 2001 only, so not usable for training 1981-2000)"}
STATS = dict(downloaded=0, from_cache=0, retries=0)


def satellite_table(g):
    """Daily satellite/reanalysis table of one basin (see KIND for the columns), indexed by UTC day."""
    return pd.read_csv(SAT / f"{g}.csv", parse_dates=["date"]).set_index("date")


def cached(name, url, params=None, check=None):
    """The text of one request; taken from data/satellite/raw/<name> when it is there, otherwise downloaded (paced, with back-off)."""
    import requests
    f = RAW / name
    if f.exists():
        STATS["from_cache"] += 1
        return f.read_text(encoding="utf-8")
    for attempt in range(7):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.encoding = "utf-8"
            if r.status_code == 200 and (check is None or check(r.text)):
                tmp = f.with_suffix(".tmp")
                tmp.write_text(r.text, encoding="utf-8")
                tmp.replace(f)
                STATS["downloaded"] += 1
                time.sleep(PAUSE)
                return r.text
            why = f"HTTP {r.status_code}: {r.text[:120]!r}"
        except requests.RequestException as e:
            why = repr(e)[:160]
        STATS["retries"] += 1
        wait = 5 * 2 ** attempt
        print(f"  request for {name} failed ({why}); waiting {wait} s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"could not download {name}")


def power(name, lon, lat, pars, span):
    """One POWER point request -> (table by UTC day with one column per parameter, NaN where missing; the JSON header)."""
    txt = cached(name, API, dict(parameters=",".join(pars), community="AG", longitude=lon, latitude=lat, start=span[0], end=span[1], format="JSON",
                                 **{"time-standard": "UTC"}), check=lambda t: '"properties"' in t)
    j = json.loads(txt)
    T = pd.DataFrame(j["properties"]["parameter"]).astype(float)
    T.index = pd.to_datetime(T.index, format="%Y%m%d")
    return T.where(T > FILL), j


def grid_probe():
    """Request points 0.02-0.04 degree apart along one west-east and one south-north line and see where each series changes."""
    pars = ["IMERG_PRECTOT", "PRECTOTCORR", "GWETTOP", "TQV", "PW"]
    edges = {"IMERG_PRECTOT": ("0.1 x 0.1 degree, cell edges on whole tenths", lambda lon, lat: (np.floor(lon * 10), np.floor(lat * 10))),
             "MERRA-2": ("0.5 (latitude) x 0.625 (longitude) degree, cell centres on multiples of 0.5 and 0.625", lambda lon, lat: (np.round(lon / 0.625), np.round(lat / 0.5))),
             "PW": ("1 x 1 degree, cell edges on whole degrees", lambda lon, lat: (np.floor(lon), np.floor(lat)))}
    out = dict(period=PROBE_SPAN, method="Neighbouring points are compared over two years of daily values: 'identical' means every day is the same, so both points "
                                         "lie in one grid cell; 'different' means a cell edge lies between them.", lines=[])
    agree, crossed = {p: True for p in pars}, {p: 0 for p in pars}      # crossed: edges of the assumed grid that lie between two neighbouring probe points
    for axis, pts in (("west-east at latitude %.2f" % PROBE_AT[1], [(x, PROBE_AT[1]) for x in PROBE_LON]), ("south-north at longitude %.2f" % PROBE_AT[0], [(PROBE_AT[0], y) for y in PROBE_LAT])):
        S = [power(f"probe_{lon:.2f}_{lat:.2f}.json", lon, lat, pars, PROBE_SPAN)[0] for lon, lat in pts]
        steps = []
        for (a, A), (b, B) in zip(zip(pts, S[:-1]), zip(pts[1:], S[1:])):
            row = {"from": list(a), "to": list(b)}
            for p in pars:
                same = bool(np.array_equal(A[p].values, B[p].values))
                rule = edges[p if p in edges else "MERRA-2"][1]
                expected_same = rule(*a) == rule(*b)
                agree[p] &= same == expected_same
                crossed[p] += not expected_same
                row[p] = "identical" if same else f"different (largest daily difference {float(np.nanmax(np.abs(A[p].values - B[p].values))):.2f})"
            steps.append(row)
        out["lines"].append(dict(line=axis, steps=steps))
    out["finding"] = {p: dict(grid_assumed=edges[p if p in edges else "MERRA-2"][0], every_step_agrees_with_that_grid=bool(agree[p]),
                              cell_edges_of_that_grid_crossed_by_the_probe=int(crossed[p])) for p in pars}      # 0 edges crossed = the probe could not test that grid
    assert agree["IMERG_PRECTOT"], "IMERG is not served on the native 0.1 degree grid: the basin averaging below would be wrong"
    return out


def basin_cells(ids):
    """{gauge_id: outline centroid, area and the 0.1 degree cells it overlaps with their overlap area in km2}."""
    import shapefile
    from pyproj import Geod
    from shapely.geometry import box, shape
    geod = Geod(ellps="GRS80")
    area = lambda geom: abs(geod.geometry_area_perimeter(geom)[0]) / 1e6
    out = {}
    for sr in shapefile.Reader(str(SHAPES)).iterShapeRecords():
        g = f"{int(sr.record['hru_id']):08d}"
        if g not in ids:
            continue
        poly = shape(sr.shape.__geo_interface__)
        poly = poly if poly.is_valid else poly.buffer(0)
        x0, y0, x1, y1 = poly.bounds
        cells = []
        for i in range(int(np.floor(x0 * 10)), int(np.floor(x1 * 10)) + 1):
            for j in range(int(np.floor(y0 * 10)), int(np.floor(y1 * 10)) + 1):
                part = poly.intersection(box(i / 10, j / 10, (i + 1) / 10, (j + 1) / 10))
                if part.area > 0:
                    cells.append(dict(lon=round(i / 10 + 0.05, 2), lat=round(j / 10 + 0.05, 2), overlap_km2=area(part)))
        out[g] = dict(centroid=(round(poly.centroid.x, 4), round(poly.centroid.y, 4)), outline_area_km2=area(poly), cells=cells)
    assert set(out) == set(ids), "a basin outline is missing from the shapefile"
    return out


def rain_check(day, sat, hi_q=0.95):
    """Satellite (or reanalysis) rain against Daymet rain on the days both have: correlation, bias, day offset, heavy-rain days."""
    ok = (day.notna() & sat.notna()).values
    d, s = day.values, sat.values
    lag = {}
    for k in (-1, 0, 1):                                    # Daymet day d against the other product's day d+k
        m = ok & np.roll(ok, -k)
        m[:1], m[-1:] = False, False
        lag[k] = float(np.corrcoef(d[m], np.roll(s, -k)[m])[0, 1])
    td, ts = np.quantile(d[ok], hi_q), np.quantile(s[ok], hi_q)
    heavy, s_hi = ok & (d > td), np.where(ok, s > ts, False)
    near = s_hi | np.roll(s_hi, 1) | np.roll(s_hi, -1)
    two = lambda x: (x + np.roll(x, 1))                     # rain of two consecutive days: insensitive to where the day boundary lies
    m2 = ok & np.roll(ok, 1)
    wet = ok & (d >= 1.0)
    return dict(n_days=int(ok.sum()), correlation=lag[0], correlation_satellite_one_day_earlier=lag[-1], correlation_satellite_one_day_later=lag[1],
                best_offset_days=int(max(lag, key=lag.get)), correlation_two_day_sums=float(np.corrcoef(two(d)[m2], two(s)[m2])[0, 1]),
                bias_percent=float(100 * (s[ok].sum() - d[ok].sum()) / d[ok].sum()), mean_mm_day_daymet=float(d[ok].mean()), mean_mm_day_other=float(s[ok].mean()),
                heavy_level_daymet_mm=float(td), heavy_level_other_mm=float(ts), n_heavy_days_daymet=int(heavy.sum()),
                heavy_days_seen_same_day=float(s_hi[heavy].mean()), heavy_days_seen_within_one_day=float(near[heavy].mean()),
                heavy_days_above_daymet_level_same_day=float((s[heavy] > td).mean()), wet_days_daymet=int(wet.sum()), wet_days_seen_as_wet=float((s[wet] >= 1.0).mean()))


def main():
    import flood as F
    t0 = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    B, D = F.basins(), F.load_data()
    ids = list(D)

    # ---- what the API offers, and what the documentation says about the IMERG run ------------------------------------------
    cat = json.loads(cached("parameters_daily_AG.json", CATALOGUE, dict(community="AG", temporal="daily")))
    wanted = ["IMERG_PRECTOT"] + list(MERRA_PARS)
    assert all(p in cat for p in wanted), "a parameter is not offered for daily data"
    doc = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", "", cached("docs_data_sources.html", DOCS), flags=re.S))))
    m_run, m_grid, m_utc = re.search(r"IMERG Final Run.*?14 hours", doc), re.search(r"The resolution of the IMERG precipitation data is.*?grid", doc), re.search(r"The IMERG data is only available[^.]*\.", doc)
    m_met = re.search(r"the meteorological data source is[^.]*\.", doc)
    m_gauge, m_ceres = re.search(r"GEOS\s*-IT has the same grid resolution as MERRA-2[^.]*\.", doc), re.search(r"primary solar resolution is[^.]*?CERES SYN1deg", doc)
    said = lambda m: m.group(0).strip() if m else "not stated by the source"

    # ---- 1. the grid -------------------------------------------------------------------------------------------------------
    print("grid probe ...", flush=True)
    probe = grid_probe()
    for p, f in probe["finding"].items():
        print(f"  {p}: {f}", flush=True)

    # ---- 2. downloads: IMERG per overlapping cell, MERRA-2 at the basin centroid ---------------------------------------------
    cells = basin_cells(ids)
    todo = sorted({(c["lon"], c["lat"]) for b in cells.values() for c in b["cells"]})
    print(f"{len(todo)} IMERG cells for {len(ids)} basins; {sum((RAW / f'imerg_{x:.2f}_{y:.2f}.json').exists() for x, y in todo)} already cached", flush=True)
    imerg, header = {}, None
    for n, (x, y) in enumerate(todo, 1):
        T, j = power(f"imerg_{x:.2f}_{y:.2f}.json", x, y, ["IMERG_PRECTOT"], IMERG_SPAN)
        imerg[(x, y)], header = T.IMERG_PRECTOT, j["header"]
        if n % 10 == 0 or n == len(todo):
            print(f"  IMERG cell {n} of {len(todo)}  ({STATS['downloaded']} downloaded, {STATS['from_cache']} from cache, {time.perf_counter() - t0:.0f} s)", flush=True)
    tables, merra_header, merra_units = {}, None, None
    for g in ids:
        lon, lat = cells[g]["centroid"]
        M, j = power(f"merra_{g}.json", lon, lat, list(MERRA_PARS), MERRA_SPAN)
        merra_header, merra_units = j["header"], j["parameters"]
        w = np.array([c["overlap_km2"] for c in cells[g]["cells"]])
        C = pd.concat([imerg[(c["lon"], c["lat"])] for c in cells[g]["cells"]], axis=1)
        assert not (C.isna().any(axis=1) & C.notna().any(axis=1)).any(), f"{g}: some cells are missing on days when others are present"
        T = M.rename(columns=MERRA_PARS)[list(MERRA_PARS.values())]
        T.insert(0, "imerg", pd.Series((C.values * w).sum(1) / w.sum(), index=C.index).reindex(T.index))      # empty before the series starts
        T.index.name = "date"
        T.round(4).to_csv(SAT / f"{g}.csv")
        tables[g] = T
        print(f"  {g} {F.SHORT[g]:18s} {len(w):2d} cells, table {T.index[0].date()} .. {T.index[-1].date()}", flush=True)

    # ---- 3. coverage -------------------------------------------------------------------------------------------------------
    cols = ["imerg"] + list(MERRA_PARS.values())
    coverage = {}
    for c in cols:
        first = [T[c].first_valid_index() for T in tables.values()]
        gaps = sum(int(T[c].loc[T[c].first_valid_index():].isna().sum()) for T in tables.values())
        coverage[c] = dict(what=KIND[c], first_valid_day=str(max(first).date()), same_first_day_in_all_basins=len(set(first)) == 1, last_day=str(tables[ids[0]].index[-1].date()),
                           missing_days_after_first_valid_day_all_basins=gaps, negative_values=int(sum((T[c] < 0).sum() for T in tables.values())),
                           smallest=float(min(T[c].min() for T in tables.values())), largest=float(max(T[c].max() for T in tables.values())))
    start = pd.Timestamp(coverage["imerg"]["first_valid_day"])

    # ---- 4. satellite rain against the gauge-based Daymet rain ---------------------------------------------------------------
    join = {g: pd.DataFrame({"daymet": D[g].prcp}).join(tables[g][["imerg", "merra_prcp"]]).loc[start:] for g in ids}
    check = {g: dict(short=F.SHORT[g], landscape=B.landscape[g], imerg=rain_check(join[g].daymet, join[g].imerg), merra2=rain_check(join[g].daymet, join[g].merra_prcp)) for g in ids}
    med = lambda prod, key: float(np.median([check[g][prod][key] for g in ids]))
    keys = ("correlation", "correlation_satellite_one_day_earlier", "correlation_satellite_one_day_later", "correlation_two_day_sums", "bias_percent",
            "heavy_days_seen_same_day", "heavy_days_seen_within_one_day", "wet_days_seen_as_wet")
    summary = {prod: dict({f"median_{k}": med(prod, k) for k in keys}, bias_percent_min=min(check[g][prod]["bias_percent"] for g in ids),
                          bias_percent_max=max(check[g][prod]["bias_percent"] for g in ids), correlation_min=min(check[g][prod]["correlation"] for g in ids),
                          correlation_max=max(check[g][prod]["correlation"] for g in ids),
                          basins_by_best_offset_days={str(k): int(sum(check[g][prod]["best_offset_days"] == k for g in ids)) for k in (-1, 0, 1)}) for prod in ("imerg", "merra2")}
    # the early record, year by year and - for the months around the expected start in 2000 - month by month
    def period_rows(freq, a, b):
        rows = []
        for per, _ in join[ids[0]].loc[a:b].groupby(pd.Grouper(freq=freq)):
            sl = {g: join[g].loc[per.to_period(freq[0]).start_time:per] for g in ids}
            r = [np.corrcoef(s.daymet, s.imerg)[0, 1] for s in sl.values() if s.daymet.std() > 0 and s.imerg.std() > 0]      # a month without rain has no correlation
            rows.append(dict(period=str(per.to_period(freq[0])), median_correlation_with_daymet=float(np.median(r)), median_daymet_total_mm=float(np.median([s.daymet.sum() for s in sl.values()])),
                             median_imerg_total_mm=float(np.median([s.imerg.sum() for s in sl.values()])),
                             median_bias_percent=float(np.nanmedian([100 * (s.imerg.sum() - s.daymet.sum()) / s.daymet.sum() if s.daymet.sum() > 0 else np.nan for s in sl.values()])),
                             largest_daily_value_mm=float(max(s.imerg.max() for s in sl.values())), days_all_basins_zero=int(np.all([s.imerg.values == 0 for s in sl.values()], axis=0).sum())))
        return rows
    by_year = period_rows("YE", start, "2014-09-30")
    by_month = period_rows("ME", start, "2001-06-30")
    later = [r["median_correlation_with_daymet"] for r in by_year if r["period"] >= "2001"]
    early_ok = [r for r in by_year if r["period"] < "2001"]
    h1 = [r for r in by_month if "2000-01" <= r["period"] <= "2000-06"]
    si, sm = summary["imerg"], summary["merra2"]
    # basins that do not follow the median day-offset pattern: the UTC day AFTER also matches the Daymet day
    ci, (BEFORE, AFTER) = {g: check[g]["imerg"] for g in ids}, ("correlation_satellite_one_day_earlier", "correlation_satellite_one_day_later")
    odd = [g for g in ids if ci[g][AFTER] >= 0.2]
    sym = [g for g in odd if ci[g][AFTER] >= ci[g][BEFORE] - 0.05]
    exception = "" if not odd else (
        f" The medians do not describe every basin: with the UTC day after, the correlation is below 0.10 in {sum(ci[g][AFTER] < 0.1 for g in ids)} of {len(ids)} basins, but 0.20 or more in "
        + " and ".join(f"{F.SHORT[g]} ({ci[g][BEFORE]:.2f} / {ci[g]['correlation']:.2f} / {ci[g][AFTER]:.2f} for the UTC day before / the same day / the day after)" for g in odd) + "."
        + (f" In {' and '.join(F.SHORT[g] for g in sym)} the day after matches as well as the day before: there the Daymet day is not earlier than the UTC day, and a UTC-day input "
           "holds no later rain than the Daymet value." if sym else ""))
    fg = probe["finding"]
    n_edge = lambda p: fg[p]["cell_edges_of_that_grid_crossed_by_the_probe"]
    merra_ok = all(fg[p]["every_step_agrees_with_that_grid"] and n_edge(p) > 0 for p in ("PRECTOTCORR", "GWETTOP", "TQV"))

    notes = [
        "Source: NASA POWER daily point API, time-standard UTC, no login. IMERG is served by UTC day only (with local solar time the API returns the fill value).",
        f"Grid (finding): {len(PROBE_LON) + len(PROBE_LAT) - 1} probe points 0.02 to 0.04 degree apart, along one west-east and one south-north line, return identical IMERG series except at the "
        f"{n_edge('IMERG_PRECTOT')} steps that cross a whole tenth of a degree (grid, lines): POWER serves IMERG on its native 0.1 degree grid and returns the value of the cell that holds the point, "
        "without interpolation. " + (f"The MERRA-2 series change only at the {n_edge('GWETTOP')} steps that cross an edge of its {fg['GWETTOP']['grid_assumed'].split(',')[0]} grid. " if merra_ok else
                                     "The MERRA-2 series do not change where the documented grid predicts (see grid, finding). ")
        + (f"For PW (CERES) the probe crosses no edge of a 1 degree grid, so it could not test that grid: PW is the same at all probe points, and the 1 degree is what the documentation states "
           f"('{said(m_ceres)}')." if n_edge("PW") == 0 else "PW changes where a 1 degree grid predicts." if fg["PW"]["every_step_agrees_with_that_grid"] else "PW does not follow a 1 degree grid (see grid, finding)."),
        f"IMERG run, quoted from {DOCS}: '{said(m_run)}'. So for 2001-2014 the series is the Final Run, which is adjusted with monthly rain-gauge analyses and arrives about 3.5 months late: "
        "it shows how good satellite rain can be after the fact, not what is at hand on the day. In real time only the Early/Late runs exist (no gauge adjustment); they were not tested here.",
        f"The documentation gives 2001-01-01 as the start of the Final Run, but the API returned valid IMERG values from {coverage['imerg']['first_valid_day']}. Which run the days before 2001 belong to "
        "is not stated by the source. Their agreement with Daymet is listed year by year (imerg_by_year) and month by month up to mid-2001 (imerg_by_month_early).",
        f"First months of 2000 (January-June): no missing or negative values; largest daily basin value {max(r['largest_daily_value_mm'] for r in h1):.0f} mm; monthly median correlation with Daymet "
        f"{min(r['median_correlation_with_daymet'] for r in h1):.2f} to {max(r['median_correlation_with_daymet'] for r in h1):.2f} and monthly median bias {min(r['median_bias_percent'] for r in h1):+.0f}% to "
        f"{max(r['median_bias_percent'] for r in h1):+.0f}%. Yearly median correlation: " + ", ".join(f"{r['period']} {r['median_correlation_with_daymet']:.2f}" for r in early_ok)
        + f"; 2001-2014 between {min(later):.2f} and {max(later):.2f}. Nothing implausible was found in the early record; single dry months show large relative differences on small totals "
        "(see median_daymet_total_mm next to median_bias_percent).",
        f"Day offset (finding): Daymet rain of day d correlates with IMERG of the same UTC day at {si['median_correlation']:.2f} and with IMERG of the UTC day before at "
        f"{si['median_correlation_satellite_one_day_earlier']:.2f}, but with the UTC day after at only {si['median_correlation_satellite_one_day_later']:.2f} (medians of 12 basins); two-day totals agree at "
        f"{si['median_correlation_two_day_sums']:.2f}. MERRA-2 rain shows the same pattern ({sm['median_correlation']:.2f} / {sm['median_correlation_satellite_one_day_earlier']:.2f} / "
        f"{sm['median_correlation_satellite_one_day_later']:.2f}), so it is a property of the Daymet day, not of IMERG: a Daymet day holds rain of two UTC days, d-1 and d. This is consistent with rain gauges "
        "that are read in the morning (the 24 hours up to the morning of day d), which was not checked against the Daymet documentation. Consequence: the low day-by-day correlation is partly a matter of "
        "day boundaries, and at the end of local day d a UTC-day value describes rain up to 19:00 of day d - later rain than the Daymet value of the same day." + exception,
        "MERRA-2 soil wetness, precipitable water (TQV) and rain are a reanalysis: a weather model that assimilates satellite and other observations. They are not direct satellite retrievals, and MERRA-2 "
        "is published weeks after the fact, so these inputs also describe the best case. PRECTOTCORR is the model's rain corrected with gauge-based analyses; the POWER documentation mentions this where it "
        f"describes the faster product that continues MERRA-2: '{said(m_gauge).replace('GEOS -IT', 'GEOS-IT').rstrip('.')}'. Soil wetness is a state of the reanalysis' land model, which that rain "
        "drives: a modelled quantity, and not independent of rain gauges.",
        "Precipitable water: the API offers two parameters. TQV (MERRA-2, kg/m2 = mm, from 1981) is the column 'pw' and is the one a model can be trained on; PW ('sourced from CERES' according to the "
        "API catalogue, cm, 1 degree grid according to the documentation) is valid from 2001-01-01 only and is kept as 'pw_ceres' for reference. How CERES obtains it is not stated by the source.",
        "Basin average of IMERG: every 0.1 degree cell that overlaps the basin outline (CAMELS HCDN_nhru_final_671.shp) is requested at its centre and weighted with the overlap area "
        "(geodesic area on the GRS80 ellipsoid). MERRA-2 and CERES variables: one request at the centroid of the outline, i.e. the value of the one coarse cell that holds the centroid; "
        "larger basins that straddle two coarse cells are therefore described by one of them.",
        "Day boundary: a UTC day d runs from 19:00 local standard time of day d-1 to 19:00 of day d in the eastern basins (18:00 for Leaf River, MS). A UTC-day value is therefore complete before the end of "
        "local day d, and using it as an input of issue day d uses nothing from after the issue time. The lag correlations show which Daymet day a UTC day matches best; "
        "correlation_two_day_sums removes the effect of the day boundary. This argument is about the day stamp. Both products were computed after the fact (the IMERG Final Run with monthly "
        "rain-gauge totals, MERRA-2 with gauge-corrected rain), so a value can still carry a correction that rests on later observations - one more reason why they are a best case.",
        "Rain check: all days from the first valid IMERG day to 2014-09-30. correlation = Pearson correlation of daily basin rain; bias_percent = 100 * (other total - Daymet total) / Daymet total; "
        "heavy day = Daymet rain above its own 95th percentile of all days; 'seen' = the other product is above ITS own 95th percentile on the same day (or on the day before, the same or the next day). "
        "Daymet is itself an interpolation of rain gauges, not the truth; in the mountains gauges are sparse too.",
        "The MERRA-2 rain comparison is given for context only: one coarse cell per basin, not an area average.",
    ]
    R = dict(source="NASA POWER daily point API (https://power.larc.nasa.gov)", api_version=header["api"]["version"], api_title=header["title"], header_sources_imerg_request=header["sources"],
             header_sources_merra_request=merra_header["sources"], time_standard=header["time_standard"], fill_value=header["fill_value"], retrieved=time.strftime("%Y-%m-%d"),
             parameters={col: dict(power_name=p, units=cat[p]["units"], name=cat[p]["name"], definition_in_catalogue=cat[p]["definition"], what_it_is=KIND[col])
                         for p, col in [("IMERG_PRECTOT", "imerg")] + list(MERRA_PARS.items())},
             csv_columns={c: KIND[c] for c in cols}, csv_files="data/satellite/<gauge_id>.csv, one row per UTC day, 1981-01-01 .. 2014-12-31; imerg is empty before its first valid day",
             documentation=dict(page=DOCS, imerg_run=said(m_run), imerg_grid=said(m_grid), imerg_time=said(m_utc), meteorology_grid=said(m_met), merra2_rain=said(m_gauge), ceres_grid=said(m_ceres)),
             imerg_run="Final Run from 2001-01-01 (gauge-adjusted, about 3.5 months late) according to the POWER documentation; for the days before 2001: not stated by the source",
             grid=probe, cells_per_basin={g: dict(short=F.SHORT[g], n_imerg_cells=len(cells[g]["cells"]), outline_area_km2=round(cells[g]["outline_area_km2"], 1),
                                                  camels_area_km2=float(B.area_gages2[g]), centroid_lon_lat=cells[g]["centroid"],
                                                  smallest_cell_weight=round(min(c["overlap_km2"] for c in cells[g]["cells"]) / cells[g]["outline_area_km2"], 4),
                                                  largest_cell_weight=round(max(c["overlap_km2"] for c in cells[g]["cells"]) / cells[g]["outline_area_km2"], 4),
                                                  cells=[[c["lon"], c["lat"], round(c["overlap_km2"], 2)] for c in cells[g]["cells"]]) for g in ids},
             n_imerg_cells_requested=len(todo), coverage=coverage, rain_check_period=(str(start.date()), "2014-09-30"), rain_check=check, rain_check_summary=summary,
             imerg_by_year=by_year, imerg_by_month_early=by_month,
             early_record=dict(years_before_2001=[r["period"] for r in early_ok], their_median_correlation=[round(r["median_correlation_with_daymet"], 3) for r in early_ok],
                               range_of_yearly_median_correlation_2001_2014=[round(min(later), 3), round(max(later), 3)],
                               lowest_monthly_median_correlation_before_2001=min((r for r in by_month if r["period"] < "2001-01"), key=lambda r: r["median_correlation_with_daymet"])),
             requests_in_this_run=dict(STATS, pause_s=PAUSE), n_raw_files_cached=len(list(RAW.glob("*.json"))) + len(list(RAW.glob("*.html"))), notes=notes, runtime_s=round(time.perf_counter() - t0, 1))
    (F.RES_OUT / "satellite_data.json").write_text(json.dumps(R, indent=1))
    return R


def figure(R):
    """Per basin: how closely the satellite rain follows the gauge-based rain (left) and how much too wet or dry it is (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import flood as F
    use_style()
    C = R["rain_check"]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 4.9), sharey=True, gridspec_kw=dict(width_ratios=[1.15, 1]))
    y, ticks = 0, {}
    for land, col in F.LAND_COL.items():
        grp = sorted([g for g in C if C[g]["landscape"] == land], key=lambda g: -C[g]["imerg"]["correlation"])
        for g in grp:
            a = C[g]["imerg"]
            axes[0].plot([a["correlation"], a["correlation_two_day_sums"]], [y, y], color=AXIS, lw=1.2, zorder=2)
            axes[0].plot(a["correlation_two_day_sums"], y, marker="o", ms=6.5, markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.6, zorder=3)
            axes[0].plot(a["correlation"], y, marker="o", ms=7.5, color=col, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
            axes[1].plot([0, a["bias_percent"]], [y, y], color=col, lw=5, alpha=.35, solid_capstyle="butt", zorder=2)
            axes[1].plot(a["bias_percent"], y, marker="o", ms=7.5, color=col, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
            axes[1].annotate(f"{a['bias_percent']:+.0f}%", (a["bias_percent"], y), xytext=(7 if a["bias_percent"] >= 0 else -7, 0), textcoords="offset points",
                             ha="left" if a["bias_percent"] >= 0 else "right", va="center", fontsize=7.8, color=INK2)
            ticks[y] = C[g]["short"]
            y += 1
        y += 0.6
    b = [C[g]["imerg"]["bias_percent"] for g in C]
    axes[0].set_ylim(y - 1.1, -0.6); axes[0].set_yticks(list(ticks), list(ticks.values()))
    axes[0].set_xlim(0.3, 1.0); axes[0].set_xlabel("correlation with gauge-based daily rain\n(1 = same ups and downs)", fontsize=8.5)
    axes[1].set_xlim(min(b + [0]) - 9, max(b + [0]) + 9); axes[1].axvline(0, color=MUTED, lw=1, zorder=1)
    axes[1].set_xlabel("satellite total minus gauge-based total\n(% of the gauge-based total)", fontsize=8.5)
    axes[0].set_title("Same rainy days?", fontsize=9.5); axes[1].set_title("Too wet or too dry overall?", fontsize=9.5)
    for ax in axes:
        ax.tick_params(labelsize=8.3, length=0); ax.grid(axis="y", visible=False)
    h = [Line2D([], [], ls="none", marker="o", ms=7.5, color=INK2, markeredgecolor=SURFACE, label="day by day"),
         Line2D([], [], ls="none", marker="o", ms=6.5, markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.6, label="two-day totals (day boundary does not matter)")]
    h += [Line2D([], [], ls="none", marker="s", ms=6.5, color=c, label=F.LAND_SHORT[l]) for l, c in F.LAND_COL.items()]
    fig.legend(handles=h[:2], loc="lower left", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.012, 0.19), columnspacing=1.6, handletextpad=.4, borderaxespad=0)
    fig.legend(handles=h[2:], loc="lower left", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.012, 0.142), columnspacing=1.6, handletextpad=.4, borderaxespad=0)
    s = R["rain_check_summary"]["imerg"]
    fig.text(0.012, 0.012, f"Satellite rain = GPM IMERG averaged over each basin; gauge-based rain = Daymet (CAMELS). Days {R['rain_check_period'][0]} to {R['rain_check_period'][1]}.\n"
                           f"Of the heaviest 5% of gauge-based rain days, the satellite ranks {100 * s['median_heavy_days_seen_same_day']:.0f}% among its own heaviest 5% on the same day\n"
                           f"and {100 * s['median_heavy_days_seen_within_one_day']:.0f}% within one day (median of the 12 basins).", ha="left", va="bottom", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.suptitle("Satellite rain against gauge-based rain, basin by basin", fontsize=10.2, fontweight="bold", color=INK, x=0.012, ha="left")
    fig.subplots_adjust(left=0.19, right=0.985, top=0.875, bottom=0.36, wspace=0.09)
    fig.savefig(F.FIG_OUT / "satellite_rain_check.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    R = json.loads(json.dumps(main()))
    figure(R)
    print(json.dumps({k: R[k] for k in ("api_version", "documentation", "coverage", "rain_check_summary", "early_record", "requests_in_this_run", "runtime_s")}, indent=1))
    print("notes:", *R["notes"], sep="\n  ")
