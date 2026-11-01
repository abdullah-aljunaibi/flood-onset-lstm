"""The data of the project: what it is, where it comes from, how it flows into a forecast, how it is computed. The single source of the data text of the
report and the slides.

The professor asks that the report and the slides explain the dataset: what it is, how it was obtained, the flow of the data, how the dataset is computed, and
all other information about it; and, where imagery was used, its capture date, sun angle and similar facts where they apply. This script writes that once, so
that every document reads the same words and numbers from one file:
  results/data_flow.json      summary (the datasets with one line each, the stages with counts), corrections_to_the_brief, imagery_note, datasets (one entry per
                              data product: what it is, producer, version, grid, time step, period used, time stamp, real-time latency, how it was obtained and when,
                              how it became a model input, where it is cached, its role, the imagery facts; CAMELS-US has parts), flow (the pipeline as ordered stages
                              with counts, scripts and outputs; the extra inputs of each experiment), how_computed (the formulas in words and symbols, with worked
                              numbers), all_information (counts, sizes, dates, scalers, thresholds, the 12 basins), unverified (what could not be sourced),
                              documentation_read (the documents read on the day of writing), inputs_read (every project file read), figure (the words and numbers the
                              drawings print, their sizes, and the number audit);
  figures/data_flow.png       the flow diagram at the width of the report's A4 text column (7.2 in), lettering 7 pt and up, with its own title; a page of its own;
  figures/data_flow_talk.png  the same flow for the slides: 12 in wide, lettering 10.8 pt and up, fewer words, no title (the slide has one).
Both drawings are made from the 'figure' key of the JSON file that was just written (one layout function for each width, one renderer), every text is checked to lie
inside its box, and every number they print is looked up again among the numbers of the rest of the file (figure.number_audit); the build stops if one is missing.

Where a fact is held by a file of the project it is read from that file (results/*.json, the CAMELS files, the caches, the saved networks) or recomputed from the data
with the engine's own code (code/flood.py, code/camels.py) and compared with the stored value (the flow conversion, the scalers against a saved network, the high-flow
levels, the headline score, the HURDAT2 file hash). Where a fact is documentation of a product (its sensor, producer, version, latency) the entry names the document it
was read from (documentation_read; read on 2026-10-08) or says 'unverified'; nothing is guessed. Facts the brief for this script had wrong are in
'corrections_to_the_brief' (the land cover is the Annual NLCD 2011 layer made from Landsat images, not ESA WorldCover).

No model is trained and nothing is chosen here. The only score read again is the headline of the LSTM with the river gauge, 1 day ahead, as the README reads it.
Run time: some seconds (it runs `du` on the cache folders). FLOOD_QUICK=1 writes results/quick/data_flow.json and figures/quick/*.png; it still reads the real
results, because there is nothing to shorten.
Log: results/logs/19_data_flow.log.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import flood as F
from camels import CAM, FORCING_COLS, V12, attributes, load_basin
from common import *

use_style()
SCRIPT = "19_data_flow.py"
T0 = time.time()
STAMP_FMT = "%Y-%m-%d %H:%M"
UTC_OFFSET = time.strftime("%z")                                           # of this laptop: the file times and the log times below are in it
DOC_DAY = "2026-10-08"                                                     # the day the documents named in DOCS were read
OUT_JSON, OUT_FIG, OUT_TALK = F.RES_OUT / "data_flow.json", F.FIG_OUT / "data_flow.png", F.FIG_OUT / "data_flow_talk.png"
READ = {}                                                                  # every project file this script read: path -> what for (written to the JSON)
say = lambda *a: print(*a, flush=True)


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if ROOT in p.parents or p == ROOT else str(p)


def note_read(p, why):
    READ[rel(p)] = why
    return rel(p)


def jread(name, why):
    f = RES / f"{name}.json"                                               # the real results, also in a smoke test
    note_read(f, why)
    return json.loads(f.read_text())


def stamp(t):
    return time.strftime(STAMP_FMT, time.localtime(t)) if t else None


def tree(path, skip=(".DS_Store", ".metadata_never_index")):
    """Files, bytes (sum of file sizes) and first/last modification time of everything under path."""
    n = b = 0
    lo = hi = None
    for root, _, files in os.walk(path):
        for f in files:
            if f in skip:
                continue
            st = os.stat(os.path.join(root, f))
            n += 1
            b += st.st_size
            lo = st.st_mtime if lo is None else min(lo, st.st_mtime)
            hi = st.st_mtime if hi is None else max(hi, st.st_mtime)
    return dict(path=rel(path), files=n, bytes=b, first_modified=stamp(lo), last_modified=stamp(hi))


def files_facts(paths):
    """The same for a list of single files."""
    st = [os.stat(p) for p in paths]
    return dict(files=len(st), bytes=int(sum(s.st_size for s in st)), first_modified=stamp(min(s.st_mtime for s in st)), last_modified=stamp(max(s.st_mtime for s in st)))


def du_bytes(path):
    """Disk use of a folder as `du -sk` reports it (blocks, not file sizes), in bytes."""
    out = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, check=True).stdout.split()[0]
    return int(out) * 1024


def gb(n, d=2):
    return round(n / 1e9, d)


def size_text(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            v = n / div
            return f"{v:.1f} {unit}" if v < 10 else f"{v:.0f} {unit}"
    return f"{n} B"


def cnt(n):
    return f"{int(round(n)):,}"


def span(a, b):
    return f"{a} to {b}"


def pct1(x):
    return f"{100 * x:.1f}%"


def flat_numbers(o, skip_keys=()):
    """Every number in a JSON structure: numeric leaves and the numbers written inside string leaves (for the number audit of the figure)."""
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k not in skip_keys:
                out += flat_numbers(v, skip_keys)
    elif isinstance(o, (list, tuple)):
        for v in o:
            out += flat_numbers(v, skip_keys)
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        out.append(float(o))
    elif isinstance(o, str):
        out += [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])\d[\d,]*\.?\d*", o) if x.replace(",", "").replace(".", "", 1).isdigit()]
    return out


# ---- documents read on DOC_DAY for the facts that no file of the project holds (a product's sensor, producer, version) ------------------------------------
DOCS = {
    "newman2015": dict(ref="Newman, A. J. et al. (2015), Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA, Hydrology and Earth System Sciences 19, 209-223 (Sect. 2.2: forcing and streamflow data)",
                       url="https://hess.copernicus.org/articles/19/209/2015/", read="full text (PDF)"),
    "addor2017": dict(ref="Addor, N., Newman, A. J., Mizukami, N., Clark, M. P. (2017), The CAMELS data set: catchment attributes and meteorology for large-sample studies, Hydrology and Earth System Sciences 21, 5293-5313",
                      url="https://hess.copernicus.org/articles/21/5293/2017/", read="full text (PDF); its reference list gives the Daymet citation 'Daymet: Daily surface weather on a 1 km grid for North America, 1980-2012'"),
    "lin2005": dict(ref="Lin, Y. and Mitchell, K. E. (2005), The NCEP Stage II/IV hourly precipitation analyses: development and applications, 19th Conference on Hydrology, American Meteorological Society, paper 1.2",
                    url="https://ams.confex.com/ams/pdfpapers/83847.pdf", read="full text (PDF)"),
    "camels_page": dict(ref="NCAR Research Applications Laboratory, CAMELS data set download page (coverage 1980-01-01 to 2014-12-31; DOIs 10.5065/D6MW2F4D for the time series and 10.5065/D6G73C3Q for the attributes)",
                        url="https://ral.ucar.edu/solutions/products/camels", read="web page"),
    "nldas2": dict(ref="NASA GES DISC / LDAS, NLDAS-2 forcing dataset information", url="https://ldas.gsfc.nasa.gov/nldas/v2/forcing", read="web page"),
    "imerg": dict(ref="NASA GPM mission, Integrated Multi-satellitE Retrievals for GPM (IMERG): product description and FAQ", url="https://gpm.nasa.gov/data/imerg", read="web page"),
    "merra2": dict(ref="NASA Global Modeling and Assimilation Office, MERRA-2", url="https://gmao.gsfc.nasa.gov/gmao-products/merra-2/", read="web page"),
    "era5_cds": dict(ref="Copernicus Climate Data Store, ERA5 hourly data on single levels from 1940 to present (ECMWF reanalysis; 0.25 degree; hourly; updated daily with a latency of about 5 days)",
                     url="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels", read="web page"),
    "power_sources": dict(ref="NASA POWER, data sources (IMERG Final Run, MERRA-2 grid, the daily time standard)", url="https://power.larc.nasa.gov/docs/methodology/data/sources/",
                          read="web page (the project quoted it on 2026-10-03: results/satellite_data.json)"),
    "openmeteo": dict(ref="Open-Meteo, Historical Weather API documentation", url="https://open-meteo.com/en/docs/historical-weather-api", read="web page"),
    "aws_aorc": dict(ref="Registry of Open Data on AWS, NOAA Analysis of Record for Calibration (AORC) Dataset", url="https://registry.opendata.aws/noaa-nws-aorc/", read="web page"),
    "aws_gefs": dict(ref="Registry of Open Data on AWS, NOAA Global Ensemble Forecast System (GEFS) Re-forecast", url="https://registry.opendata.aws/noaa-gefs-reforecast/", read="web page"),
    "joerd": dict(ref="Tilezen/Mapzen 'joerd' terrain tile builder: data sources, and its example configuration (SRTMGL1.003, folder 2000.02.11)",
                  url="https://github.com/tilezen/joerd/blob/master/docs/data-sources.md", url2="https://raw.githubusercontent.com/tilezen/joerd/master/config.example.yaml", read="documentation page and example configuration file"),
    "srtm_wiki": dict(ref="Shuttle Radar Topography Mission (flight of Space Shuttle Endeavour, February 2000; X-band and C-band radar; version 3 void filling)",
                      url="https://en.wikipedia.org/wiki/Shuttle_Radar_Topography_Mission", read="web page (a secondary source)"),
    "nlcd_meta": dict(ref="USGS, Annual National Land Cover Database (NLCD) Collection 1.0 product-suite metadata (the Land Cover Change Index product of the same suite), DOI 10.5066/P94UXNTS",
                      url="https://data.usgs.gov/datacatalog/metadata/USGS.66f5b27dd34e791ae5dfd2fa.xml", read="metadata file"),
    "nhc_hurdat": dict(ref="NOAA National Hurricane Center, Hurricane Research Division: HURDAT2 data page, and Landsea (April 2022), The revised Atlantic hurricane database (HURDAT2)",
                       url="https://www.nhc.noaa.gov/data/#hurdat", url2="https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf", read="web page and format document (PDF)"),
    "usgs_iv": dict(ref="USGS Water Services, Instantaneous Values Service", url="https://waterservices.usgs.gov/docs/instantaneous-values/", read="web page"),
}
for _k, _v in DOCS.items():
    _v["read_on"] = DOC_DAY
doc = lambda k: f"{DOCS[k]['url']} (read {DOC_DAY})"


# ---- 1. the records of the project --------------------------------------------------------------------------------------------------------------
say("reading the records ...")
A = jread("data_audit", "671 -> 78 -> 63 -> 12; estimated and zero-flow shares of the 12")
AUD = pd.read_csv(RES / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
note_read(RES / "basin_audit.csv", "the audit of the 78 candidates (completeness, estimated and zero-flow shares)")
SAT = jread("satellite_data", "NASA POWER: IMERG and MERRA-2 daily; grids, cells, coverage, day offsets")
GIS = jread("basin_gis", "SRTM-format elevation tiles, Annual NLCD land cover, outlines: sources, versions, dates")
DW = jread("day_windows", "which 24 hours each daily value covers; ERA5, MERRA-2 hourly and USGS 15-minute sources")
AO = jread("aorc_data", "AORC v1.1: grid, cells, download, filled hours, AWS notice")
GB = jread("gefs_basins", "GEFSv12 reforecast: weights, issue-time rule, bias factors, real-time margin")
S4 = jread("stage4_realtime", "NCEP Stage IV: files read, bytes, basin means for the issue days of the onsets")
ACC = jread("data_access", "URLs, sizes, dates and access facts of the products")
FA = jread("flood_analysis", "thresholds, events and onsets, HURDAT2 file, verdict criteria")
SC = jread("sources_checked", "what the project checked about the Daymet day")
RV = jread("review_aorc_rain_to_midnight", "the reviewer's reading of the AORC methods document and AWS registry page")
LA = jread("lstm_all", "the LSTM settings chosen on the validation years")
SI = jread("satellite_inputs", "the satellite-era split and the variants with IMERG and MERRA-2 inputs")
R15 = jread("rain_to_midnight", "the clocks of the inputs of experiment E1 (ERA5 rain to midnight)")
R16 = jread("aorc_rain_to_midnight", "the clocks of the inputs of experiment E1b (AORC rain to midnight)")
R18 = jread("forecast_input", "the clocks of the inputs of experiment E2 (GEFS forecast)")
RT = jread("rain_timing", "experiment of 10_rain_timing.py: NLDAS rain up to the evening")
MB = jread("more_basins", "experiment E3: 63 training rivers")

IDS = list(F.study_ids())
B = F.basins()
note_read(DATA / "derived" / "camels12_daymet.pkl", "the cached daily tables of the 12 basins (flood.load_data)")
D = F.load_data()
IDX = D[IDS[0]].index
N_DAYS = len(IDX)
PART_DAYS = {p: len(pd.date_range(*F.SPLIT[p])) for p in F.SPLIT}
assert sum(PART_DAYS.values()) == N_DAYS == len(pd.date_range(*F.PERIOD)), "the three parts must tile the period"
N_YEARS = N_DAYS / 365.25
SPLIT_SAT = SI["split_satellite_era"]
PART_DAYS_SAT = {p: len(pd.date_range(*SPLIT_SAT[p])) for p in SPLIT_SAT}
assert SI["split_main"] == {k: list(v) for k, v in F.SPLIT.items()}, "the split of the engine is not the one the satellite experiment recorded"

# the rule of 01_data_audit.py, read from its own source and checked against the audit table
_src = (ROOT / "code" / "01_data_audit.py").read_text()
note_read(ROOT / "code" / "01_data_audit.py", "the rule that picks the candidates and the 12 basins")
_m = re.search(r"A\.huc_02\.isin\(\[([\d, ]+)\]\) & \(A\.frac_snow < ([\d.]+)\) & \(A\.aridity < ([\d.]+)\) & A\.area_gages2\.between\((\d+), (\d+)\)", _src)
assert _m, "the candidate rule of 01_data_audit.py was not found"
RULE = dict(huc_regions=[int(x) for x in _m.group(1).split(",")], max_snow_share=float(_m.group(2)), max_aridity=float(_m.group(3)), area_km2=[int(_m.group(4)), int(_m.group(5))],
            per_landscape=int(re.search(r"N_PICK = (\d+)", _src).group(1)))
assert set(AUD.huc_02.unique()) == set(RULE["huc_regions"]) and AUD.frac_snow.max() < RULE["max_snow_share"] and AUD.aridity.max() < RULE["max_aridity"]
assert AUD.area_gages2.between(*RULE["area_km2"]).all() and len(AUD) == A["n_candidates"]
assert int(AUD.complete.sum()) == A["n_complete"] and int(AUD.selected.sum()) == len(A["selected"]) == len(IDS)

# CAMELS: the files as shipped
note_read(CAM / "basin_dataset_public_v1p2" / "readme_FIRST.txt", "CAMELS versions and periods of the three forcing products")
note_read(CAM / "basin_dataset_public_v1p2" / "readme_basin_mean_forcing.txt", "CAMELS description of the three forcing directories")
note_read(CAM / "basin_dataset_public_v1p2" / "readme_streamflow.txt", "CAMELS description of the flow files and flags")
note_read(CAM / "readme.txt", "CAMELS attribute versions and citations")
_first = (CAM / "basin_dataset_public_v1p2" / "readme_FIRST.txt").read_text()
CAMELS_V12_DATE = re.search(r"Updated:\s+(\d+ \w+ \d{4})\s+Version 1\.2", _first).group(1)                       # '11 Mar 2016'
ZIP_TS, ZIP_SHAPES = CAM / "basin_timeseries_v1p2_metForcing_obsFlow.zip", CAM / "basin_set_full_res.zip"
ATTR_TABLES = ("name", "topo", "clim", "hydro", "soil", "vege", "geol")
ATTR_FILES = [CAM / f"camels_{n}.txt" for n in ATTR_TABLES]
ATTR_DOCS = [CAM / "camels_attributes_v2.0.xlsx", CAM / "camels_attributes_v2.0.pdf", CAM / "readme.txt"]
N_CAMELS = len(attributes())
GLOBS = {s: sorted((V12 / "basin_mean_forcing" / s).glob("*/*_forcing_leap.txt")) for s in ("daymet", "nldas", "maurer")}
N_FORCING_FILES = {s: len(v) for s, v in GLOBS.items()}
N_FLOW_FILES = len(sorted((V12 / "usgs_streamflow").glob("*/*_streamflow_qc.txt")))
_have = {s: {f.name.split("_")[0] for f in v} for s, v in GLOBS.items()}
_have["flow"] = {f.name.split("_")[0] for f in (V12 / "usgs_streamflow").glob("*/*_streamflow_qc.txt")}
_ids671 = set(attributes().index)
assert N_CAMELS == A["n_all_camels"] and all(_ids671 <= h for h in _have.values()), "the CAMELS files do not hold the 671 basins the audit counted"
EXTRA_FILES = {s: len(h - _ids671) for s, h in _have.items()}              # the unzipped folders hold a few gauges beyond the 671 of the attribute tables: never used


def forcing_file(g, s="daymet"):
    return next(f for f in GLOBS[s] if f.name.startswith(g))


def forcing_header(g, s):
    """(latitude, elevation, area in m2, column names with units, first date, last date, rows) of a CAMELS forcing file."""
    f = forcing_file(g, s)
    with open(f) as fh:
        lat, elev, area = float(fh.readline()), float(fh.readline()), float(fh.readline())
        cols = fh.readline().split()
        rows = [ln.split()[:3] for ln in fh if ln.strip()]
    return dict(file=rel(f), area_m2=area, columns=cols, first=f"{rows[0][0]}-{rows[0][1]}-{rows[0][2]}", last=f"{rows[-1][0]}-{rows[-1][1]}-{rows[-1][2]}", rows=len(rows))


HDR = {s: forcing_header(IDS[0], s) for s in GLOBS}
note_read(forcing_file(IDS[0], "daymet"), "column names and units of the Daymet forcing files (header), rows and dates")
for _s in ("nldas", "maurer"):
    note_read(forcing_file(IDS[0], _s), f"column names and period of the {_s} forcing files")
UNITS = {c.split("(")[0].lower(): c.split("(")[1].rstrip(")") for c in HDR["daymet"]["columns"] if "(" in c}          # {'dayl': 's', 'prcp': 'mm/day', ...}
assert set(FORCING_COLS) <= set(UNITS), "a forcing column of camels.py is not in the file header"

# flow: read again from the raw files, converted with the formula, compared with the cache
def raw_flow(g):
    f = next((V12 / "usgs_streamflow").glob(f"*/{g}_streamflow_qc.txt"))
    fl = pd.read_csv(f, sep=r"\s+", header=None, names=["g", "y", "m", "d", "cfs", "flag"])
    fl.index = pd.to_datetime(dict(year=fl.y, month=fl.m, day=fl.d))
    return fl, f


CFS_TO_MM = dict(m3_per_ft3=0.0283168, s_per_day=86400, mm_per_m=1000)
FLOWFACTS, MAXDIFF_FLOW = {}, 0.0
for g in IDS:
    fl, ff = raw_flow(g)
    h = forcing_header(g, "daymet")
    cfs = fl.cfs.where(fl.cfs >= 0)
    q = cfs * CFS_TO_MM["m3_per_ft3"] * CFS_TO_MM["s_per_day"] / h["area_m2"] * CFS_TO_MM["mm_per_m"]
    qp = q.loc[F.PERIOD[0]:F.PERIOD[1]]
    MAXDIFF_FLOW = max(MAXDIFF_FLOW, float(np.nanmax(np.abs(qp.values - D[g].q_mm.values))))
    p = fl.loc[F.PERIOD[0]:F.PERIOD[1]]
    est = int(p.flag.astype(str).str.contains("e").sum())
    FLOWFACTS[g] = dict(file_rows=len(fl), file_first=str(fl.index[0].date()), file_last=str(fl.index[-1].date()), days_used=len(p), cfs_missing=int((p.cfs < 0).sum()), flag_missing=int((p.flag == "M").sum()),
                        estimated=est, estimated_share=est / len(p), zero_flow=int((qp == 0).sum()), zero_flow_share=float((qp == 0).mean()), area_km2_forcing_file=h["area_m2"] / 1e6,
                        area_km2_gages2=float(B.loc[g, "area_gages2"]), cfs_first_day=float(fl.cfs.loc[F.PERIOD[0]]))
    note_read(ff, "daily mean flow of a study basin (cfs and QC flag), converted and compared with the cache")
    assert FLOWFACTS[g]["days_used"] == N_DAYS and FLOWFACTS[g]["cfs_missing"] == 0 and FLOWFACTS[g]["flag_missing"] == 0, f"{g}: the study basins must have a complete flow record"
assert MAXDIFF_FLOW < 1e-9, "the flow in the cache is not cfs x 0.0283168 x 86400 / area x 1000"
_est_sel = {b["gauge_id"]: b["estimated_share"] for b in A["selected"]}
assert all(abs(FLOWFACTS[g]["estimated_share"] - _est_sel[g]) < 6e-4 for g in IDS), "estimated shares disagree with results/data_audit.json"
_zero_sel = {b["gauge_id"]: b["zero_flow_share"] for b in A["selected"]}
assert all(abs(FLOWFACTS[g]["zero_flow_share"] - _zero_sel[g]) < 6e-4 for g in IDS), "zero-flow shares disagree with results/data_audit.json"
N_BASIN_DAYS = N_DAYS * len(IDS)
N_MISSING = int(sum(int(D[g].isna().sum().sum()) for g in IDS))
assert N_MISSING == 0
N_VALUES = N_BASIN_DAYS * (len(F.FORCING) + 1)


# ---- 2. the ten basin properties: which CAMELS table, what the table says about them -----------------------------------------------------------------
def xlsx_rows(path):
    """The rows of the one sheet of an .xlsx file as lists of text (the standard library only: an .xlsx file is a zip of XML)."""
    z = zipfile.ZipFile(path)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ss = [("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"]))) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    out = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter("{%s}row" % ns["m"]):
        cells = []
        for c in row.findall("m:c", ns):
            v = c.find("m:v", ns)
            cells.append("" if v is None else (ss[int(v.text)] if c.get("t") == "s" else v.text))
        out.append([x.strip() for x in cells])
    return out


note_read(CAM / "camels_attributes_v2.0.xlsx", "the attribute names, descriptions, units and data sources of the CAMELS tables")
XL = xlsx_rows(CAM / "camels_attributes_v2.0.xlsx")
XL_ROW, XL_TABLE, _cur = {}, {}, None
for _r in XL:
    if _r and _r[0].startswith("camels_") and " - " in _r[0]:
        _cur = _r[0].split(" - ")[0]
        XL_TABLE[_cur] = _r[0]
    elif _r and _r[0] and _cur and _r[0] != "Attribute":
        XL_ROW[_r[0]] = dict(table=_cur, description=_r[1] if len(_r) > 1 else "", unit=_r[2] if len(_r) > 2 else "", data_source=_r[3] if len(_r) > 3 else "")
TABLE_COLS = {n: pd.read_csv(CAM / f"camels_{n}.txt", sep=";", nrows=0).columns.tolist() for n in ATTR_TABLES}
for _f in ATTR_FILES:
    note_read(_f, "the columns of this attribute table (which table holds each basin property)")


def xl_lookup(col):
    for key in (col, "_".join(col.split("_")[::-1])):                                  # the spreadsheet says forest_frac, the table frac_forest
        if key in XL_ROW:
            return XL_ROW[key]
    return dict(table="", description="", unit="", data_source="")


STATIC_INFO = []
for _p in F.STATIC:
    _col = "area_gages2" if _p == "log_area" else _p
    _tab = next(n for n, cols in TABLE_COLS.items() if _col in cols)
    _x = xl_lookup(_col)
    _v = B.loc[IDS, _p].astype(float)
    STATIC_INFO.append(dict(name=_p, table=f"camels_{_tab}.txt", column=_col, how=("log10 of the area_gages2 column" if _p == "log_area" else "as in the table"),
                            description=_x["description"], unit=_x["unit"], data_source=_x["data_source"], table_note=XL_TABLE.get(f"camels_{_tab}", ""),
                            min_over_the_12=float(_v.min()), max_over_the_12=float(_v.max())))
CLIM_AVG_YEARS = re.search(r"(\d{4}/\d\d/\d\d) to (\d{4}/\d\d/\d\d)", XL_TABLE["camels_clim"])
assert CLIM_AVG_YEARS, "the climate table's averaging period was not found in the spreadsheet"
CLIM_AVG = [CLIM_AVG_YEARS.group(1).replace("/", "-"), CLIM_AVG_YEARS.group(2).replace("/", "-")]
STATIC_CLIM = [s["name"] for s in STATIC_INFO if s["table"] == "camels_clim.txt"]

# ---- 3. scalers and the high-flow level, computed as flood.fit computes them, and checked against a saved network and the stored thresholds -------------
TR = np.asarray((IDX >= pd.Timestamp(F.SPLIT["train"][0])) & (IDX <= pd.Timestamp(F.SPLIT["train"][1])))
_dyn = np.concatenate([D[g].loc[TR, list(F.FORCING)].values for g in IDS])
SC_MU, SC_SD = np.nanmean(_dyn, 0), np.nanstd(_dyn, 0) + 1e-9
_q = np.concatenate([D[g].q_mm.values[TR] for g in IDS])
SC_Q = dict(q_mu=float(_q.mean()), q_sd=float(_q.std()), lq_mu=float(np.log1p(_q).mean()), lq_sd=float(np.log1p(_q).std()))
_S = B.loc[IDS, list(F.STATIC)].values.astype(float)
SC_ST_MU, SC_ST_SD = _S.mean(0), _S.std(0) + 1e-9
_ck_file = RES / "models" / "lstm_wh_s0.pt"
note_read(_ck_file, "the scalers stored in a saved network (12 basins, training years): the recomputed scalers must equal them")
_ck = torch.load(_ck_file, map_location="cpu", weights_only=False)
_rel = lambda a, b: float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)) / np.maximum(np.abs(np.asarray(b, float)), 1e-12)))
SC_CHECK = dict(file=rel(_ck_file), weather_columns_max_relative_difference=_rel(SC_MU, _ck["sc"]["dyn_mu"][:len(F.FORCING)]) + _rel(SC_SD, _ck["sc"]["dyn_sd"][:len(F.FORCING)]),
                flow_max_relative_difference=max(_rel(SC_Q[k], _ck["sc"][k]) for k in SC_Q), static_max_relative_difference=_rel(SC_ST_MU, _ck["sc"]["st_mu"]) + _rel(SC_ST_SD, _ck["sc"]["st_sd"]))
assert max(v for k, v in SC_CHECK.items() if k != "file") < 1e-9, "the recomputed scalers differ from those stored in the saved network"
THR = F.thresholds(D)
assert max(abs(THR[g] - FA["thresholds_mm_day"][g]) for g in IDS) < 1e-9, "the high-flow levels differ from results/flood_analysis.json"
FORECAST_SAMPLES = {p: {f"lead{k}": int(F.part_mask(IDX, k, p).sum()) for k in F.LEADS} for p in F.SPLIT}          # issue days whose target day lies in the part (per basin)

# ---- 4. the headline of the LSTM with the river gauge, as the README reads it ----------------------------------------------------------------------------
note_read(RES / "preds" / "lstm_wq.npz", "the forecasts of the LSTM with the river gauge (the headline read again)")
_quick, _pred_dir = F.QUICK, F.PRED
F.QUICK, F.PRED = False, RES / "preds"                         # the stored forecasts of the real runs, also in a smoke test (which would otherwise look in results/quick first)
try:
    HEAD = F.evaluate(F.load_preds("lstm_wq"), THR)["summary"]["l1"]
finally:
    F.QUICK, F.PRED = _quick, _pred_dir
_v = FA["verdict"]["services"]["lstm_wq"]["l1"]["onset"]
assert abs(_v["pod"] - HEAD["onset"]["pod"]) < 1e-9 and abs(_v["far"] - HEAD["onset"]["far"]) < 1e-9, "the headline differs from results/flood_analysis.json"
N_ONSETS = int(HEAD["onset"]["hits"] + HEAD["onset"]["misses"])
assert N_ONSETS == FA["events"]["n_onsets"], "the onsets counted from the forecasts differ from the event catalogue"
VERDICT = FA["verdict"]["services"]["lstm_wq"]["l1"]["onset"]["pooled"]

# ---- 5. what is stored: forecasts, networks, trainings ----------------------------------------------------------------------------------------------------
PRED_NAMES = sorted(p.stem for p in (RES / "preds").glob("*.npz"))
PRED_FAMILY = {}
for _n in PRED_NAMES:
    PRED_FAMILY[_n.split("_")[0]] = PRED_FAMILY.get(_n.split("_")[0], 0) + 1
N_PRED_SINGLE = sum(bool(re.search(r"_s\d+$", n)) for n in PRED_NAMES)
NET_FILES = sorted((RES / "models").glob("*.pt"))
NETS = []
for _f in NET_FILES:
    _c = torch.load(_f, map_location="cpu", weights_only=False)
    _dyn_cols = list(_c.get("dyn") or _c["cfg"]["dyn"])
    NETS.append(dict(file=_f.name, script=_c.get("script"), input_set=_c.get("input_set"), n_basins=len(_c["ids"]), dyn=_dyn_cols, use_flow=bool(_c["cfg"]["use_flow"]), seed=_c.get("seed"),
                     train=list(_c["split"]["train"]), test=list(_c["split"]["test"]), n_params=int(_c["n_params"])))
note_read(RES / "models", "the saved networks (the columns each one reads, its script, split and number of weights)")
N_TRAININGS_RECORDED = {}
for _f in sorted(RES.glob("*.json")):
    try:
        _j = json.loads(_f.read_text())
    except ValueError:
        continue
    if isinstance(_j, dict) and isinstance(_j.get("n_trainings"), int) and not _j.get("quick_mode"):
        N_TRAININGS_RECORDED[_f.name] = _j["n_trainings"]
        note_read(_f, "the number of trainings it records (n_trainings)")
CFG = LA["config"]
N_PARAMS = {n["input_set"]: n["n_params"] for n in NETS if n["script"] == "15_rain_to_midnight.py"}
N_PARAMS_WQ = next(r["n_params"] for r in LA["setting_check"]["table"] if r["chosen"])

# ---- 6. the caches: folders, sizes (du), file times -----------------------------------------------------------------------------------------------------
CACHE_DIRS = {"data/camels_us": "CAMELS-US as downloaded and unzipped", "data/derived": "daily tables of the basins, parsed once (flood.load_data)", "data/hourly": "ERA5 and MERRA-2 hourly rain, USGS 15-minute flow",
              "data/satellite": "IMERG and MERRA-2 daily series per basin (+ raw API answers)", "data/aorc": "AORC hourly basin means (+ parts, Stage IV hours)", "data/gefs": "GEFSv12 reforecast rain, control member (+ basin means)",
              "data/dem": "elevation tiles", "data/landcover": "Annual NLCD land-cover grids", "data/hurdat": "HURDAT2 storm tracks", "data/basemap": "US state outlines (drawing only)",
              "results/preds": "stored forecasts", "results/models": "saved networks"}
CACHE = {}
for _p, _w in CACHE_DIRS.items():
    _t = tree(ROOT / _p)
    CACHE[_p] = dict(what=_w, files=_t["files"], bytes_on_disk=du_bytes(ROOT / _p), bytes_of_files=_t["bytes"], first_modified=_t["first_modified"], last_modified=_t["last_modified"])
CACHE_DATA_BYTES = sum(v["bytes_on_disk"] for k, v in CACHE.items() if k.startswith("data/"))
CAMELS_DOWNLOAD = files_facts([ZIP_TS, ZIP_SHAPES] + ATTR_FILES + ATTR_DOCS)
CAMELS_ZIPS = files_facts([ZIP_TS, ZIP_SHAPES])


# ---- 7. facts of the individual products ------------------------------------------------------------------------------------------------------------------
HD = DATA / "hourly"
HOURLY = {"era5": files_facts(sorted(HD.glob("era5_*.json"))), "merra2": files_facts(sorted(HD.glob("merra_*.json"))), "usgs_iv": files_facts(sorted(HD.glob("usgs_iv_*.rdb"))),
          "usgs_sites": files_facts([HD / "usgs_sites.rdb"])}
POWER_RAW, POWER_TABLES = tree(DATA / "satellite" / "raw"), files_facts([DATA / "satellite" / f"{g}.csv" for g in IDS])
for _g in IDS:
    note_read(DATA / "satellite" / f"{_g}.csv", "daily IMERG and MERRA-2 series of a basin (present and complete)")
DEM, LANDCOVER, HURDAT_DIR = tree(DATA / "dem"), tree(DATA / "landcover"), tree(DATA / "hurdat")
_tz = DW["sources"]["usgs_sites"]["time_zone"]
GAUGE_ZONE = {g: ("America/Chicago" if _tz[g] == "CST" else "America/New_York") for g in IDS}                    # the wall clocks the local days follow


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    x = math.sin((lat2 - lat1) * p / 2) ** 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(x))


ERA5_OFFSET_KM = {g: haversine_km(SAT["cells_per_basin"][g]["centroid_lon_lat"][1], SAT["cells_per_basin"][g]["centroid_lon_lat"][0], *DW["sources"]["era5"]["per_basin"][g]["grid_point_lat_lon"]) for g in IDS}
AREA_KM2 = {g: GIS["basins"][g]["area_km2"] for g in IDS}                                                          # geodesic area of the outline (07_basin_maps.py)
N_ERA5_HOURS = {v["n_hours"] for v in DW["sources"]["era5"]["per_basin"].values()}
assert len(N_ERA5_HOURS) == 1 and N_ERA5_HOURS == {AO["n_hours"]}, "ERA5 and AORC must hold the same hours"
N_HOURS = AO["n_hours"]
AORC_NCELLS = {g: v["n_cells"] for g, v in AO["plan"].items()}
AORC_PARTIAL_HOURS = int(sum(AO["summary"]["hours_with_some_cells_missing_by_basin"].values()))
IMERG_NCELLS = {g: v["n_imerg_cells"] for g, v in SAT["cells_per_basin"].items()}
GEFS_NBOXES = {g: v["n_boxes"] for g, v in GB["basins"].items()}
_bf = [x for v in GB["bias"]["factors"].values() for x in v.values()]
GEFS_SIDECARS = sorted((DATA / "gefs" / "c00").glob("apcp_*.json"))
GEFS_RUN = dict(months=len(GEFS_SIDECARS), bytes_total=0, bytes_range=0, days=0, flags={}, first_utc=None, last_utc=None, idx_bytes=0, seconds=0.0)
for _f in GEFS_SIDECARS:
    _j = json.loads(_f.read_text())
    GEFS_RUN["bytes_total"] += _j["bytes_fetched"]
    for _d in _j["days"].values():
        GEFS_RUN["days"] += 1
        GEFS_RUN["flags"][str(_d["flag"])] = GEFS_RUN["flags"].get(str(_d["flag"]), 0) + 1
        GEFS_RUN["idx_bytes"] += _d.get("bytes_idx", 0)
        GEFS_RUN["bytes_range"] += _d.get("bytes_range", 0)
        GEFS_RUN["seconds"] += _d.get("seconds_fetch", 0)
        for k in ("first_utc", "last_utc"):
            t = _d["fetched_utc"]
            GEFS_RUN[k] = t if GEFS_RUN[k] is None or (t < GEFS_RUN[k] if k == "first_utc" else t > GEFS_RUN[k]) else GEFS_RUN[k]
note_read(DATA / "gefs" / "c00", "the provenance sidecars of the 178 monthly GEFS files: bytes read, fetch times, flags")
assert GEFS_RUN["days"] == 5392 and GEFS_RUN["flags"] == {"0": 5392}, "every GEFS day must be stored and flagged ok"
assert GEFS_RUN["bytes_total"] == GEFS_RUN["bytes_range"] + GEFS_RUN["idx_bytes"], "the month sidecars' bytes_fetched must be the range bytes plus the .idx bytes"
S4_RUNS = S4["data"]["fetch_runs"]
HURDAT_INFO = FA["events"]["hurdat"]
HURDAT_FILE = ROOT / HURDAT_INFO["file"]
assert hashlib.sha256(HURDAT_FILE.read_bytes()).hexdigest() == HURDAT_INFO["sha256"], "the HURDAT2 file is not the one the flood analysis used"
note_read(HURDAT_FILE, "the HURDAT2 file (hashed and compared with the one the flood analysis used)")
HURDAT_SUFFIX = re.search(r"hurdat2-1851-(\d{4})-(\d{2})(\d{2})(\d{2})\.txt", HURDAT_FILE.name).groups()           # last season, and the date written in the file name (MMDDYY)
DEM_CELLS = sum(v["n_elevation_cells"] for v in GIS["basins"].values())
DEM_VOIDS = sum(v["n_elevation_voids"] for v in GIS["basins"].values())
LC_CELLS = sum(v["n_landcover_cells"] for v in GIS["basins"].values())
LC_UNCLASSIFIED = max(v["landcover_unclassified_share"] for v in GIS["basins"].values())
NLCD_FILES = sorted((DATA / "landcover").glob("annual_nlcd_*.tif"))
assert len(NLCD_FILES) == len(IDS) and all(str(GIS["sources"]["land_cover"]["year"]) in f.name for f in NLCD_FILES), "the land-cover files are not the Annual NLCD layer the record names"
AORC_DL = AO["download"]["runs"][0]
DAYS_FILE = ROOT / "run_all.sh"
note_read(DAYS_FILE, "the order of the steps and the document builders")
RUN_ALL = DAYS_FILE.read_text()
BUILDERS = [(m.group(1), m.group(2).strip()) for m in re.finditer(r"\$PY code/(build_\w+\.py)\s*(?:#\s*)?(.*)", RUN_ALL)]
DOCUMENTS = sorted(p.name for p in ROOT.glob("*") if p.suffix in (".pdf", ".pptx") and p.is_file())


# ---- 8. the datasets ---------------------------------------------------------------------------------------------------------------------------------------
KEYS = ("key", "name", "short", "kind", "what_it_is", "producer", "version", "spatial_grid", "time_step", "period_used", "time_stamp", "latency_real_time", "obtained", "turned_into_input", "cached_in", "role", "role_code",
        "imagery", "one_line", "figure_line", "sources", "unverified")
OBT = ("url_pattern", "login_needed", "bytes_read", "bytes_read_note", "bytes_cached", "date_downloaded", "script")


def E(**kw):
    """One dataset entry; every entry has the same keys so that a document can loop over them."""
    kw.setdefault("sources", [])
    kw.setdefault("unverified", [])
    kw.setdefault("quality", [])
    kw.setdefault("imagery", "No image of this product was looked at, classified or interpreted in this project.")
    missing = [k for k in KEYS if k not in kw]
    assert not missing, f"{kw.get('key')}: missing {missing}"
    ob = kw["obtained"]
    assert all(k in ob for k in OBT), f"{kw['key']}: obtained lacks {[k for k in OBT if k not in ob]}"
    return kw


def dates_text(first, last):
    a, b = first[:10], last[:10]
    return f"{a} (file times, laptop clock UTC{UTC_OFFSET[:3]})" if a == b else f"{a} to {b} (file times, laptop clock UTC{UTC_OFFSET[:3]})"


def hm(day_end):
    return day_end["day_ends_local"]


DE = DW["day_ends"]
FLOWDAY = DW["river_flow_day"]["summary"]["parts"]
_att_ver = re.search(r"# VERSION (\d+\.\d+)", (CAM / "readme.txt").read_text()).group(1)
_area = [min(AREA_KM2.values()), max(AREA_KM2.values())]
_n_est = [FLOWFACTS[g]["estimated_share"] for g in IDS]
_n_zero = {g: FLOWFACTS[g]["zero_flow_share"] for g in IDS}
_zero_basins = sorted((g for g in IDS if _n_zero[g] > 0), key=lambda g: -_n_zero[g])
_area_diff = [100 * (FLOWFACTS[g]["area_km2_gages2"] - FLOWFACTS[g]["area_km2_forcing_file"]) / FLOWFACTS[g]["area_km2_forcing_file"] for g in IDS]
_STATE = {"NC": "North Carolina", "VA": "Virginia", "GA": "Georgia", "FL": "Florida", "MS": "Mississippi", "AL": "Alabama", "SC": "South Carolina", "TN": "Tennessee"}
_states = sorted({re.search(r"([A-Z]{2})\.?$", n.strip()).group(1) for n in B.loc[IDS, "gauge_name"]}, key=lambda c: -sum(n.strip().rstrip(".").endswith(c) for n in B.loc[IDS, "gauge_name"]))
STATE_TEXT = ", ".join(_STATE[c] for c in _states[:-1]) + " and " + _STATE[_states[-1]]
Q_EXAMPLE = dict(basin=F.SHORT["03439000"], gauge_id="03439000", date=F.PERIOD[0], cfs=FLOWFACTS["03439000"]["cfs_first_day"], area_m2=forcing_header("03439000", "daymet")["area_m2"])
Q_EXAMPLE["mm_per_day"] = Q_EXAMPLE["cfs"] * CFS_TO_MM["m3_per_ft3"] * CFS_TO_MM["s_per_day"] / Q_EXAMPLE["area_m2"] * CFS_TO_MM["mm_per_m"]
assert abs(Q_EXAMPLE["mm_per_day"] - float(D["03439000"].q_mm.loc[F.PERIOD[0]])) < 1e-9

CAMELS_PARTS = [
    E(key="camels_flow", name="USGS daily mean river flow (CAMELS usgs_streamflow)", short="USGS daily flow", kind="river flow measured at a gauge",
      what_it_is=("The daily mean discharge of the river at the USGS stream gauge of each basin, in cubic feet per second, as compiled from the USGS National Water Information System for CAMELS "
                  "(Newman et al. 2015, Sect. 2.2). Every day carries a quality flag: A (certified, measured), A:e (certified but estimated) or M (missing, value -999)."),
      producer="US Geological Survey (National Water Information System); compiled by NCAR for CAMELS",
      version=f"CAMELS time series v1.2 ({CAMELS_V12_DATE}); the files list no USGS data version",
      spatial_grid=f"one gauge per basin; the 12 study gauges lie inside their outlines, at most {max(GIS['basins'][g]['gauge']['distance_to_outline_km'] for g in IDS):.2f} km from the edge (results/basin_gis.json)",
      time_step="daily mean",
      period_used=f"{F.PERIOD[0]} to {F.PERIOD[1]}: {N_DAYS:,} days per basin (the files run {FLOWFACTS[IDS[0]]['file_first']} to {FLOWFACTS[IDS[0]]['file_last']})",
      time_stamp=(f"the local calendar day of the gauge, midnight to midnight by its wall clock (summer time included): measured against USGS 15-minute flow, January to June 2012: standard time in force, "
                  f"{FLOWDAY['standard_time_in_force']['n_gauges_best_midnight_standard_time']} of {DW['river_flow_day']['summary']['n_gauges']} gauges fit midnight to midnight best "
                  f"(median difference {FLOWDAY['standard_time_in_force']['median_relative_difference_percent']['midnight_to_midnight_standard_time']}% of the daily flow, against "
                  f"{FLOWDAY['standard_time_in_force']['median_relative_difference_percent']['utc_day']}% for the UTC day); summer time in force, {FLOWDAY['daylight_saving_time_in_force']['n_gauges_best_midnight_summer_time']} "
                  f"gauges fit the summer-time midnight best (results/day_windows.json, river_flow_day)"),
      latency_real_time=("none for this file (a finished research data set). The gauges themselves report every 15 minutes and transmit hourly (USGS Instantaneous Values Service); "
                         "how soon the daily mean of a finished day is published was not checked."),
      obtained=dict(url_pattern="as part of the CAMELS download (basin_timeseries_v1p2_metForcing_obsFlow.zip, folder usgs_streamflow/<region>/<gauge_id>_streamflow_qc.txt)", login_needed="see CAMELS-US",
                    bytes_read=None, bytes_read_note="read with the rest of the CAMELS time-series zip (see CAMELS-US)", bytes_cached=tree(V12 / "usgs_streamflow")["bytes"], date_downloaded="see CAMELS-US", script="none (unzipped by hand)"),
      turned_into_input=(f"q_mm = cfs x 0.0283168 x 86400 / area_m2 x 1000 (cubic feet per second to millimetres of water per day over the basin; area_m2 is line 3 of the basin's forcing file, "
                         f"e.g. {Q_EXAMPLE['basin']}, {Q_EXAMPLE['date']}: {Q_EXAMPLE['cfs']:.0f} cfs over {Q_EXAMPLE['area_m2'] / 1e6:.1f} km2 = {Q_EXAMPLE['mm_per_day']:.2f} mm/day; recomputed here for all {N_BASIN_DAYS:,} basin-days: "
                         f"largest difference from the cache {MAXDIFF_FLOW:g}). -999 and flag M would become missing; the 12 chosen basins have none in the period. It is the quantity every model forecasts "
                         f"(lead 1 to 3 days) and, in the sets with the river gauge, an input as log(1 + q_mm) scaled by the training-year mean and spread."),
      cached_in=dict(paths=["data/camels_us/basin_dataset_public_v1p2/usgs_streamflow", "data/derived/camels12_daymet.pkl (column q_mm)"], files=N_FLOW_FILES, bytes_on_disk=None),
      role="training input and target of every model (the river gauge)", role_code="every model",
      one_line=f"{N_DAYS:,} days per basin of daily mean flow, {len(IDS)} gauges, {pct1(min(_n_est))} to {pct1(max(_n_est))} of days estimated by the USGS, none missing",
      figure_line=["USGS daily flow", "gauge · daily · 1980–2014"],
      sources=["results/data_audit.json", "results/day_windows.json (river_flow_day)", "data/camels_us/basin_dataset_public_v1p2/readme_streamflow.txt", doc("newman2015"), doc("usgs_iv")]),
    E(key="camels_daymet", name="Daymet weather, basin means (CAMELS basin_mean_forcing/daymet)", short="Daymet weather", kind="weather interpolated from weather-station readings to a 1 km grid, averaged over the basin",
      what_it_is=("Daymet is a daily 1 km gridded weather data set for North America, derived solely from daily station observations of temperature and precipitation: stations are interpolated with an iterative "
                  "method and a truncated Gaussian filter, and the Mountain Climate Simulator (MT-CLIM) estimates shortwave radiation and humidity (Newman et al. 2015, Sect. 2.2). CAMELS gives the area-weighted mean "
                  "over each basin outline, computed by the USGS Geo Data Portal. Columns as shipped: "
                  + ", ".join(f"{c} ({UNITS[c]})" for c in FORCING_COLS) + "; the models read " + ", ".join(F.FORCING) + " (swe, snow-water equivalent, is not used)."),
      producer="Oak Ridge National Laboratory (Daymet; Thornton et al.); basin means by NCAR and the USGS Center for Integrated Data Analytics for CAMELS",
      version="unverified: not stated in the CAMELS readme files, nor in Newman et al. 2015 or Addor et al. 2017, which cite 'Daymet: Daily surface weather on a 1 km grid for North America, 1980-2012' (Thornton et al. 2012, ORNL DAAC); the project could only date it to before Daymet version 4 (2021)",
      spatial_grid=f"1 km grid (Newman et al. 2015), averaged to one value per basin with area weights; the 12 basins are {_area[0]:,.0f} to {_area[1]:,.0f} km2",
      time_step="daily",
      period_used=f"{F.PERIOD[0]} to {F.PERIOD[1]} (the files run {HDR['daymet']['first']} to {HDR['daymet']['last']})",
      time_stamp=(f"the rain of 'day d' ends at about {DE['daymet']['day_ends_local_rounded']} local standard time on day d (median basin {hm(DE['daymet'])}; basins {DE['daymet']['basin_range_local'][0]} to "
                  f"{DE['daymet']['basin_range_local'][1]}; four readings {DE['daymet']['range_over_readings_local'][0]} to {DE['daymet']['range_over_readings_local'][1]}; measured against hourly rain in results/day_windows.json). "
                  f"So the afternoon and evening rain of the issue day is filed under the next day. Consistent with the rain-gauge observers' day: Daymet version 4 (2021) documents the effect and corrects it, and CAMELS (2015, extended 2016) predates that version (published readings: published_reading_of_the_day); the other Daymet variables were not measured."),
      latency_real_time="none: a finished research data set (1980-2014)",
      published_reading_of_the_day=[dict(reference=x["reference"], url=x["url"], what_it_says=x["what_it_says"], not_said=x["what_it_does_not_say"]) for x in SC["sources"]],
      obtained=dict(url_pattern="as part of the CAMELS download (folder basin_mean_forcing/daymet/<region>/<gauge_id>_lump_cida_forcing_leap.txt)", login_needed="see CAMELS-US", bytes_read=None,
                    bytes_read_note="read with the rest of the CAMELS time-series zip (see CAMELS-US)", bytes_cached=tree(V12 / "basin_mean_forcing" / "daymet")["bytes"], date_downloaded="see CAMELS-US", script="none (unzipped by hand)"),
      turned_into_input=(f"read as shipped: the six columns {', '.join(F.FORCING)} of each day; no change of units or day. Each column is scaled with the mean and standard deviation of the training years "
                         f"({F.SPLIT['train'][0]} to {F.SPLIT['train'][1]}), pooled over the training basins: x_scaled = (x - mean_train) / sd_train. The rain column is the 'rain table' every model reads."),
      cached_in=dict(paths=["data/camels_us/basin_dataset_public_v1p2/basin_mean_forcing/daymet", "data/derived/camels12_daymet.pkl"], files=N_FORCING_FILES["daymet"], bytes_on_disk=None),
      role="training input of every model (the weather table)", role_code="every model",
      one_line=f"basin means of a 1 km grid, daily, {F.PERIOD[0][:4]}-{F.PERIOD[1][:4]}; the rain day ends at about {DE['daymet']['day_ends_local_rounded']} local time",
      figure_line=["Daymet weather", "1 km grid, basin mean · daily · 1980–2014"],
      sources=["data/camels_us/basin_dataset_public_v1p2/readme_basin_mean_forcing.txt", "results/day_windows.json (day_ends.daymet)", "results/sources_checked.json", doc("newman2015"), doc("addor2017")],
      unverified=["The Daymet version inside CAMELS: not stated in any CAMELS file or paper read; the project's own record (results/sources_checked.json) says the documentation of the Daymet version inside CAMELS was not checked."]),
    E(key="camels_nldas", name="NLDAS-2 rain, basin means (CAMELS basin_mean_forcing/nldas)", short="NLDAS-2 rain", kind="gauge-only daily rain analysis, split into hours with radar, averaged over the basin",
      what_it_is=("NLDAS-2, the second phase of the North American Land Data Assimilation System, is a forcing data set on a 1/8-degree grid (12 km in Newman et al. 2015), hourly; its non-rain fields come from the NCEP North American Regional Reanalysis. "
                  "Its rain is not a model's: it is the CPC gauge-only daily rain analysis with an orographic (PRISM) adjustment, spread over the hours of the day with weights from the NWS Stage II radar analysis and CMORPH (NLDAS-2 documentation). "
                  "CAMELS sums it to daily basin means."),
      producer="the NLDAS-2 project (Xia et al. 2012; rain: the NOAA CPC gauge analysis; hosted at NASA GES DISC); basin means by NCAR and the USGS for CAMELS",
      version="NLDAS-2 (Xia et al. 2012), as in CAMELS v1.2",
      spatial_grid="1/8 degree (about 12 km) grid, averaged to one value per basin",
      time_step="daily (sum of the hourly values)",
      period_used=f"{F.PERIOD[0]} to {F.PERIOD[1]} (the files run {HDR['nldas']['first']} to {HDR['nldas']['last']})",
      time_stamp=(f"the rain of 'day d' ends at about {DE['nldas']['day_ends_local_rounded']} local standard time on day d (median basin {hm(DE['nldas'])}; basins {DE['nldas']['basin_range_local'][0]} to {DE['nldas']['basin_range_local'][1]}; "
                  f"readings {DE['nldas']['range_over_readings_local'][0]} to {DE['nldas']['range_over_readings_local'][1]}): within the accuracy of the check this is the UTC day (19:00 local standard time). "
                  f"None of the rain after local midnight is in the value of the issue day in the test years (results/day_windows.json, future_rain)."),
      latency_real_time="a few days (GES DISC documentation, as noted in results/data_access.json; not verified there); not a real-time product in CAMELS",
      obtained=dict(url_pattern="as part of the CAMELS download (folder basin_mean_forcing/nldas/<region>/<gauge_id>_lump_nldas_forcing_leap.txt)", login_needed="see CAMELS-US", bytes_read=None,
                    bytes_read_note="read with the rest of the CAMELS time-series zip (see CAMELS-US)", bytes_cached=tree(V12 / "basin_mean_forcing" / "nldas")["bytes"], date_downloaded="see CAMELS-US", script="none (unzipped by hand)"),
      turned_into_input=("the 'prcp' column of the NLDAS table of the same day t, as one extra daily input beside the six (column prcp_nldas; experiment 10_rain_timing.py, sets wn and wqn); control: the same series one day older "
                         "(prcp_nldas_older). Also given together with ERA5 rain to midnight (set wqnh of 15_rain_to_midnight.py). Scaled like every input."),
      cached_in=dict(paths=["data/camels_us/basin_dataset_public_v1p2/basin_mean_forcing/nldas"], files=N_FORCING_FILES["nldas"], bytes_on_disk=None),
      role="extra input (rain up to the evening) and its control; also the check of which day the rain tables mean", role_code="extra input",
      one_line=f"daily basin-mean rain of a 1/8 degree analysis, {F.PERIOD[0][:4]}-{F.PERIOD[1][:4]}; the day ends at about {DE['nldas']['day_ends_local_rounded']} local time",
      figure_line=["NLDAS-2 rain", "1/8° (12 km) · daily · 1980–2014"],
      sources=["data/camels_us/basin_dataset_public_v1p2/readme_basin_mean_forcing.txt", "results/day_windows.json (day_ends.nldas, future_rain)", "results/rain_timing.json", doc("nldas2"), doc("newman2015")]),
    E(key="camels_maurer", name="Maurer rain, basin means (CAMELS basin_mean_forcing/maurer)", short="Maurer", kind="gridded daily weather (Maurer et al. 2002), averaged over the basin",
      what_it_is="The third gridded forcing product in CAMELS: the Maurer et al. (2002) daily data set on a 12 km grid, area-averaged to the basin like the others (Newman et al. 2015, Sect. 2.2).",
      producer="the Maurer et al. (2002) data set, as the CAMELS readme names it; basin means by NCAR and the USGS for CAMELS",
      version="Maurer et al. (2002), as in CAMELS v1.2", spatial_grid="12 km grid (Newman et al. 2015), averaged to one value per basin", time_step="daily",
      period_used=f"never read by a model; looked at only in the day-window check (the files run {HDR['maurer']['first']} to {HDR['maurer']['last']})",
      time_stamp=(f"no single day window: median basin {DE['maurer']['day_ends_local']}, basins {DE['maurer']['basin_range_local'][0]} to {DE['maurer']['basin_range_local'][1]} "
                  f"(results/day_windows.json); its value of day d reaches past local midnight"),
      latency_real_time="none: a finished research data set (1980-2008)",
      obtained=dict(url_pattern="as part of the CAMELS download (folder basin_mean_forcing/maurer/<region>/<gauge_id>_lump_maurer_forcing_leap.txt)", login_needed="see CAMELS-US", bytes_read=None,
                    bytes_read_note="read with the rest of the CAMELS time-series zip (see CAMELS-US)", bytes_cached=tree(V12 / "basin_mean_forcing" / "maurer")["bytes"], date_downloaded="see CAMELS-US", script="none (unzipped by hand)"),
      turned_into_input="not an input of any model of the project; only its rain column enters check_day_windows.py (which 24 hours does its day cover)",
      cached_in=dict(paths=["data/camels_us/basin_dataset_public_v1p2/basin_mean_forcing/maurer"], files=N_FORCING_FILES["maurer"], bytes_on_disk=None),
      role="check only (the clock of a rain table)", role_code="check or map",
      one_line=f"daily basin-mean weather, 12 km grid, 1980-2008; no model reads it",
      figure_line=["Maurer rain", "12 km · daily · 1980–2008"],
      sources=["results/day_windows.json (day_ends.maurer)", doc("newman2015")]),
    E(key="camels_attributes", name="CAMELS basin attributes (the seven camels_*.txt tables)", short="Basin properties", kind="a table of fixed properties per basin",
      what_it_is=(f"Seven tables with one row per basin ({', '.join('camels_' + n + '.txt' for n in ATTR_TABLES)}): name and location, topography, climate indices, hydrological signatures, soil, vegetation and geology, "
                  f"described in camels_attributes_v2.0.xlsx and Addor et al. 2017. The models read {len(F.STATIC)} of them: " + ", ".join(F.STATIC) + "."),
      producer="NCAR (Addor, Newman, Mizukami, Clark), from USGS, Daymet, STATSGO, Pelletier et al. 2016 and GLHYMPS, as the spreadsheet names them per attribute",
      version=f"attributes v{_att_ver} (readme.txt)", spatial_grid="one value per basin", time_step="fixed (not a time series)",
      period_used=(f"{', '.join(STATIC_CLIM)}: averages over {CLIM_AVG[0]} to {CLIM_AVG[1]} (starred in the spreadsheet), which overlaps the validation and test years; "
                   f"the other {len(F.STATIC) - len(STATIC_CLIM)} have no averaging period stated; their sources are " + "; ".join(f"{x['name']}: {x['data_source']}" for x in STATIC_INFO if x['name'] not in STATIC_CLIM)),
      time_stamp="not a daily value: the same number is repeated on every day of a basin", latency_real_time="none: a finished research data set",
      obtained=dict(url_pattern="CAMELS download: camels_<table>.txt, camels_attributes_v2.0.xlsx and .pdf, readme.txt", login_needed="see CAMELS-US", bytes_read=CAMELS_DOWNLOAD["bytes"] - CAMELS_ZIPS["bytes"],
                    bytes_read_note="the 7 tables plus the spreadsheet, the PDF and the readme", bytes_cached=CAMELS_DOWNLOAD["bytes"] - CAMELS_ZIPS["bytes"], date_downloaded=dates_text(CAMELS_DOWNLOAD["first_modified"], CAMELS_DOWNLOAD["last_modified"]),
                    script="none (downloaded by hand)"),
      turned_into_input=(f"flood.basins() joins the tables; log_area = log10(area_gages2); the {len(F.STATIC)} properties are repeated on every day of a basin and scaled over the training basins "
                         f"(mean and standard deviation of the {len(IDS)} study basins in the all-river models, of the 63 or 50 training rivers in experiment E3). None is computed from the flow record."),
      cached_in=dict(paths=["data/camels_us/camels_*.txt"], files=len(ATTR_FILES), bytes_on_disk=None),
      role="training input of every all-river model (basin properties)", role_code="every model",
      one_line=f"{len(F.STATIC)} fixed numbers per basin; {len(STATIC_CLIM)} of them are climate averages over {CLIM_AVG[0][:4]}-{CLIM_AVG[1][:4]}",
      figure_line=[f"{len(F.STATIC)} basin properties", "per basin · fixed"],
      sources=["data/camels_us/camels_attributes_v2.0.xlsx", "data/camels_us/readme.txt", doc("addor2017")]),
    E(key="camels_outlines", name="CAMELS basin outlines (basin_set_full_res, HCDN_nhru_final_671.shp)", short="Basin outlines", kind="polygons of the 671 basins",
      what_it_is="The polygon of each basin (USGS geospatial fabric, smoothed on a 100 m grid), with the gauge id in the field hru_id; geographic coordinates, NAD83 (from the .prj file).",
      producer="USGS (geospatial fabric), distributed with CAMELS", version=f"CAMELS v1.2 ({CAMELS_V12_DATE}); four basins were fixed by hand in v1.1 (readme_FIRST.txt)", spatial_grid="vector polygons (about 30 m between vertices)", time_step="fixed",
      period_used="fixed", time_stamp="not a time series", latency_real_time="none",
      obtained=dict(url_pattern="CAMELS download: basin_set_full_res.zip", login_needed="see CAMELS-US", bytes_read=ZIP_SHAPES.stat().st_size, bytes_read_note="the zip file", bytes_cached=tree(CAM / "basin_set_full_res")["bytes"],
                    date_downloaded=dates_text(stamp(ZIP_SHAPES.stat().st_mtime), stamp(ZIP_SHAPES.stat().st_mtime)), script="none (unzipped by hand)"),
      turned_into_input="the geometry behind every basin mean the project computes itself: IMERG (cells weighted by overlap area), GEFS (0.25 degree boxes weighted by overlap area), AORC and Stage IV (cells whose centre lies inside the outline), "
                        "and the elevation, slope and land-cover statistics of 07_basin_maps.py",
      cached_in=dict(paths=["data/camels_us/basin_set_full_res"], files=None, bytes_on_disk=None), role="geometry for every basin mean computed by the project", role_code="every model",
      one_line=f"671 basin polygons; the 12 study outlines are {_area[0]:,.0f} to {_area[1]:,.0f} km2", figure_line=["Basin outlines", "polygons · fixed"],
      sources=["results/basin_gis.json (sources.outlines)", "data/camels_us/basin_dataset_public_v1p2/readme_shapefiles.txt"]),
]
_camels_disk = CACHE["data/camels_us"]
CAMELS = E(key="camels_us", name="CAMELS-US", short="CAMELS-US", kind="a public data set of river basins: gauge flow, basin-mean weather, basin outlines and basin properties",
           what_it_is=(f"A public large-sample data set of {N_CAMELS} minimally disturbed river basins of the United States (from the USGS HCDN-2009 basin set): for each basin the daily weather averaged over the basin "
                       f"(three products: Daymet, NLDAS, Maurer), the daily mean river flow measured by the US Geological Survey, the basin outline and seven tables of basin properties. "
                       f"The project reads {len(IDS)} basins of it."),
           producer="NCAR with the USGS: Newman et al. (2015) for the time series and outlines, Addor et al. (2017) for the attributes",
           version=f"time series v1.2 ({CAMELS_V12_DATE}: Daymet and NLDAS forcing extended to {HDR['daymet']['last']}); attribute tables v{_att_ver}",
           spatial_grid=f"one value per basin (basin means); 671 outlines; the 12 study basins are {_area[0]:,.0f} to {_area[1]:,.0f} km2 and lie in {STATE_TEXT}",
           time_step="daily", period_used=f"{F.PERIOD[0]} to {F.PERIOD[1]}: {N_YEARS:.0f} water years, {N_DAYS:,} days per basin",
           time_stamp="differs by part: flow = the local wall-clock day; Daymet rain ends at about 09:00 and NLDAS rain at about 17:30 local standard time (see the parts)",
           latency_real_time="none: a research data set of 1980 to 2014",
           obtained=dict(url_pattern="the data set page is https://ral.ucar.edu/solutions/products/camels; the files used are basin_timeseries_v1p2_metForcing_obsFlow.zip, basin_set_full_res.zip, camels_attributes_v2.0.xlsx/.pdf and the seven camels_*.txt tables",
                         login_needed="no account recorded; the page asks the downloader to accept its terms of use", bytes_read=CAMELS_DOWNLOAD["bytes"],
                         bytes_read_note=f"the two zip files ({size_text(CAMELS_ZIPS['bytes'])}) and the attribute files, as sizes of the stored files", bytes_cached=CAMELS_DOWNLOAD["bytes"],
                         date_downloaded=dates_text(CAMELS_DOWNLOAD["first_modified"], CAMELS_DOWNLOAD["last_modified"]), script="none: downloaded by hand and unzipped in place (README: data/camels_us is the public download, untouched)"),
           turned_into_input="see the parts: the Daymet weather columns and the flow become the daily table of every basin; the ten properties become repeated columns; the outlines give the basin means of the other products",
           cached_in=dict(paths=["data/camels_us (as downloaded and unzipped)", "data/derived/camels12_daymet.pkl (the 12 study basins: 6 weather columns + flow, parsed once)"], files=_camels_disk["files"], bytes_on_disk=_camels_disk["bytes_on_disk"]),
           role="the data every model is trained and scored on", role_code="every model",
           one_line=f"{N_CAMELS} basins; the project uses {len(IDS)}: daily Daymet weather and USGS flow, {N_DAYS:,} days each, 1980-2014",
           figure_line=["CAMELS-US", "671 basins, 12 used · daily · 1980–2014"],
           imagery=("Not images. Daymet is interpolated from weather stations, NLDAS-2 rain is a gauge analysis, the flow is measured at gauges. No capture date or sun angle applies. "
                    "(Daymet's shortwave radiation 'srad' and humidity are estimated by the Mountain Climate Simulator, MT-CLIM, from daily station observations of temperature and precipitation: Daymet uses no satellite data.)"),
           sources=["data/camels_us/basin_dataset_public_v1p2/readme_FIRST.txt", "data/camels_us/readme.txt", "results/data_audit.json", doc("camels_page"), doc("newman2015"), doc("addor2017")],
           unverified=["The address the zip files were downloaded from was not recorded in the project files."])
CAMELS["parts"] = CAMELS_PARTS


# ---- 8b. the other products ------------------------------------------------------------------------------------------------------------------------------
_aq = next(c for c in RV["checks"] if c["id"] == "d1")
AORC_QUOTES = [dict(where=q["where"], quote=q["quote"]) for q in _aq["numbers"]["quotes"]]
AORC_READING = _aq["note"]
_lag = AO["time_stamps"]["stamp_check_against_stage_iv"]["test_years"]["pooled_correlation_by_lag"]
_gap = AO["gap_fill"]
AOS = AO["summary"]
IM, ME = SAT["rain_check_summary"]["imerg"], SAT["rain_check_summary"]["merra2"]
GSK = R18["forecast_data"]["skill_summary"]
S4A = S4["agreement_of_the_two_products_on_the_issue_days"]
GC = GIS["comparison_with_camels"]
GEFS_MARGIN = {k: v["margin_minutes_of_the_last_file"] for k, v in GB["realtime_margin"]["by_zone_and_clock"].items()}
_s4_start, _last = S4_RUNS["runs"][0]["started"], S4_RUNS["runs"][-1]
_s4_end = (pd.Timestamp(_last["started"]) + pd.Timedelta(seconds=_last["seconds"])).strftime(STAMP_FMT)
_aorc_end = (pd.Timestamp(AORC_DL["started"]) + pd.Timedelta(seconds=AORC_DL["seconds"])).strftime(STAMP_FMT)
local_from_utc = lambda t: pd.Timestamp(t.replace("Z", "+00:00")).to_pydatetime().astimezone().strftime(STAMP_FMT)               # an ISO time in UTC -> this laptop's clock
GEFS_FACTOR_RANGE = [min(_bf), max(_bf)]
GEFS_DATES = [str(GB["issue_days"]["first"]), str(GB["issue_days"]["last"])]
GEFS_NLAT, GEFS_NLON = int(round((GB["grid"]["lat"][1] - GB["grid"]["lat"][0]) / GB["grid"]["box_size_deg"])) + 1, int(round((GB["grid"]["lon"][1] - GB["grid"]["lon"][0]) / GB["grid"]["box_size_deg"])) + 1
AORC_FILES = tree(DATA / "aorc")
_gefs_c00 = CACHE["data/gefs"]
_raw = DATA / "satellite" / "raw"
POWER_BYTES = {"imerg": files_facts(sorted(_raw.glob("imerg_*.json"))), "merra2_daily": files_facts(sorted(_raw.glob("merra_*.json"))),
               "other": files_facts([f for f in sorted(_raw.iterdir()) if not f.name.startswith(("imerg_", "merra_"))])}
assert sum(v["files"] for v in POWER_BYTES.values()) == POWER_RAW["files"]
S4_FILES_FACTS = files_facts([DATA / "aorc" / n for n in ("stage4_issue_days.csv", "stage_iv_hindsight.csv", "stage4_issue_days_fetch_runs.json", "stage4_issue_days_unusable.json", "stage_iv_check.json")])

AORC = E(key="aorc", name="NOAA AORC v1.1 (Analysis of Record for Calibration)", short="AORC v1.1", kind="an hourly rain analysis anchored on rain gauges and radar, made after the fact",
         what_it_is=("The gridded record of near-surface weather of NOAA's National Weather Service Office of Water Prediction; the project reads its hourly total precipitation (APCP_surface). It is an analysis, "
                     "not an observation: a daily rain total is fixed first and then spread over the 24 hours. Daily total: NLDAS-2 gauge-based 24-hour totals ending 12 UTC for 1979-2001, NCEP Stage IV radar-and-gauge totals with manual quality control from 2002; "
                     "both bias-corrected with monthly Livneh totals (1979-2015 where available). Hours: the hourly shape of Stage IV where radar and satellite exist, GDAS where they do not, other sources before the radar era "
                     "(the reading of the NWS methods document in results/review_aorc_rain_to_midnight.json). So the hours of the training years 1980-2000 are daily totals spread by non-radar sources; the hours of the test years follow the radar."),
         producer="NOAA National Weather Service, Office of Water Prediction (OWP); copy on AWS Open Data",
         version=("AORC version 1.1 (version 1.0 was completed in November 2019; 1.1 addresses its issues: AWS registry). AWS announced on 2026-04-02 that about 0.04% of the rows were masked by error and that the Zarr files are being "
                  "regenerated: data/aorc is kept as the reference copy (sha256 of the 12 basin files in results/aorc_data.json, amendments)"),
         spatial_grid=(f"30 arc-second latitude-longitude grid (1/120 degree, about 0.8 km, called 1 km here; {AO['grid']['nominal']['n_lat']:,} x {AO['grid']['nominal']['n_lon']:,} cells); a basin = the cells whose centre lies inside its outline: "
                       f"{min(AORC_NCELLS.values()):,} to {max(AORC_NCELLS.values()):,} cells"),
         time_step="hourly", period_used=f"{AO['span']}: {N_HOURS:,} hours at each of the {len(IDS)} basins",
         time_stamp=(f"hour-ending, UTC: the value stamped T is the rain of the hour that ends at T (methods document p.10; checked against Stage IV hourly files on test-year storm days: correlation {_lag['0']:.2f} at lag 0, "
                     f"{_lag['1']:.2f} and {_lag['-1']:.2f} at lags +1 and -1); the project sums it over the wall-clock day of each gauge"),
         latency_real_time="about 10 days (AWS registry: 'The data creation runs with a 10-day lag to ensure the inclusion of any corrections to the input Stage IV and NLDAS data'): not a real-time product",
         obtained=dict(url_pattern="https://noaa-nws-aorc-v1-1-1km.s3.amazonaws.com/YYYY.zarr (variable APCP_surface, int16 in units of 0.1 mm, chunks of 144 hours x 128 rows x 256 columns, zstd; one HTTP request per chunk, "
                                   "decoded with the zstd command-line tool)", login_needed="no (AWS Open Data, anonymous)", bytes_read=AORC_DL["bytes"],
                       bytes_read_note=f"{AORC_DL['chunks']:,} chunk fetches with {AORC_DL['threads']} connections in {AORC_DL['seconds'] / 60:.0f} minutes; every chunk reduced at once to the basin sums, none kept",
                       bytes_cached=AORC_FILES["bytes"], date_downloaded=f"{AORC_DL['started']} to {_aorc_end} laptop time (UTC{UTC_OFFSET[:3]})", script="code/fetch_aorc.py (also called by 16_aorc_rain_to_midnight.py)"),
         turned_into_input=(f"basin mean = the plain mean of the cells whose centre lies inside the outline (their areas differ by under 1% within a basin), one value per hour; {_gap['total_hours_filled']} of {_gap['basin_hours_in_total']:,} basin-hours "
                            f"({len(_gap['per_basin'])} basins, runs of 48 to 120 hours) held no valid cell and were filled with the ERA5 hourly rain and flagged; {AORC_PARTIAL_HOURS} hours with some cells missing keep the mean of the others. "
                            f"Summed over the wall-clock day of the gauge (zoneinfo, summer time included): aorc_to_midnight (extra input of E1b, set wqa), aorc_to_09 (the window of the rain table, a control), aorc_utc_day (a control); "
                            f"in wqa_rep it replaces the Daymet rain column. The AORC rain of the local days t+1 to t+3 (p1..p3) is the perfect rain forecast, an upper bound, and the reference of the GEFS bias factors."),
         cached_in=dict(paths=["data/aorc/aorc_<gauge_id>.npz (12 basin series)", "data/aorc/parts (the sums per year and spatial chunk)", "data/aorc/stage4_issue_days.csv (Stage IV hours)"], files=CACHE["data/aorc"]["files"], bytes_on_disk=CACHE["data/aorc"]["bytes_on_disk"]),
         role="extra input (rain up to local midnight), the perfect-rain upper bound, and the reference rain of the forecast experiment", role_code="extra input",
         one_line=f"hourly rain on a 30 arc-second grid, {AO['span'][:4]} to {AO['span'].split(' to ')[1][:4]}; an analysis made after the fact, about 10 days behind real time",
         figure_line=["AORC v1.1", f"1 km · hourly · {AO['span'][:4]}–{AO['span'].split(' to ')[1][:4]}"],
         imagery=("Not an image product. Its hours in the radar era follow Stage IV radar rain and the earlier hours follow non-radar sources; no radar or satellite scene was looked at in this project, "
                  "so no capture date or sun angle applies."),
         quality=[f"daily totals (UTC day) correlate with NLDAS {AOS['median_daily_correlation_same_day']['nldas']:.2f}, with ERA5 {AOS['median_daily_correlation_same_day']['era5']:.2f}, with IMERG {AOS['median_daily_correlation_same_day']['imerg']:.2f} and with Daymet "
                  f"{AOS['median_daily_correlation_same_day']['daymet']:.2f} (median of the 12 basins; Daymet's day ends in the morning); its mean rain is {100 * AOS['median_ratio_aorc_to_daymet_mean_rain']:.0f}% of Daymet's"],
         documentation_quotes=AORC_QUOTES, reviewer_reading=AORC_READING,
         sources_by_period=[dict(period="1980-10-01 to 2001-12-31 (all training years and the first 15 months of the validation years)", daily_total="NLDAS-2 24-hour totals ending 12 UTC, bias-corrected with monthly Livneh totals",
                                 hours="spread over the hours by non-radar sources (MDR, NOWrad, Stage II, CMORPH, CFSR, GDAS: the reviewer's reading)"),
                            dict(period="2002-01-01 to 2014-09-30 (the rest of the validation years and all test years)", daily_total="NCEP Stage IV radar-and-gauge totals with manual quality control (bias-corrected with monthly Livneh totals to 2015)",
                                 hours="spread over the hours with the Stage IV hourly shape; GDAS where radar and satellite are missing")],
         sources=["results/aorc_data.json", "results/review_aorc_rain_to_midnight.json (checks, id d1: the methods document read in full)", "results/aorc_rain_to_midnight.json", "results/data_access.json", doc("aws_aorc"),
                  "https://www.weather.gov/media/owp/operations/aorc_v1_1_methods.pdf (the NWS methods document, read by the reviewer on 2026-10-07)"],
         unverified=["The methods document does not give the day boundary of the Stage IV daily product (12 UTC to 12 UTC to the reviewer's knowledge); the product documents of Stage IV and NLDAS-2 and Fall et al. 2023 were not read by the project."])

GEFS = E(key="gefs", name="NOAA GEFSv12 reforecast (control member)", short="GEFSv12 reforecast", kind="weather-model rain forecasts re-run for past dates, issued before the rain",
         what_it_is=("NOAA's Global Ensemble Forecast System version 12 was re-run for past dates (a 'reforecast') so that forecasts of the past can be studied: one forecast a day from the 00 UTC state, 2000 to 2019, "
                     "5 members (11 on Wednesdays). The project takes the control member (c00): total precipitation, 3-hourly, to forecast hour 78. It is a forecast, not an analysis: the rain it holds for the "
                     "next three days has not fallen when the forecast is issued."),
         producer="NOAA (NCEP), through the NOAA Open Data Dissemination programme on AWS Open Data",
         version="GEFS version 12 reforecast; member c00 (labelled 'ENS=low-res ctl' in the files)",
         spatial_grid=(f"0.25 degree global grid, cut to {GB['grid']['lon'][0]:g} to {GB['grid']['lon'][1]:g} degrees longitude and {GB['grid']['lat'][0]:g} to {GB['grid']['lat'][1]:g} degrees latitude "
                       f"({GEFS_NLAT} x {GEFS_NLON} boxes); a box is about 630 km2 at 35.7 N; a basin overlaps {min(GEFS_NBOXES.values())} to {max(GEFS_NBOXES.values())} boxes, weighted by overlap area"),
         time_step=f"3-hourly accumulations to forecast hour {GB['grid']['window_end_hours'][1]} ({GB['grid']['n_windows']} windows); one run a day at 00 UTC",
         period_used=f"runs of 2000-01-01 to 2014-10-05 ({GEFS_RUN['days']:,} days; the archive spans 2000-2019); issue days {GEFS_DATES[0]} to {GEFS_DATES[1]} ({GB['issue_days']['n']:,})",
         time_stamp=("UTC; window k holds the rain of forecast hours (3k, 3k+3] after 00 UTC of the run date. At the issue time (local midnight at the end of day t) the latest 00 UTC run is that of UTC date t+1, "
                     "4 to 6 hours old; local day t+k is cut from the windows by the gauge's wall clock"),
         latency_real_time=(f"an archive has none. In real time the operational control member's last needed file (hour 78) is posted about {GB['realtime_margin']['latency_f078_h']:.2f} hours after the run (measured on 4 days, 2026-10-03 to 06): "
                            f"margin at the issue time {min(GEFS_MARGIN.values()):.0f} to {max(GEFS_MARGIN.values()):.0f} minutes (negative: about 15 minutes too late in Eastern summer for the files after hour 39)"),
         obtained=dict(url_pattern="https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/YYYY/YYYYMMDD00/c00/Days:1-10/apcp_sfc_YYYYMMDD00_c00.grib2 (+ .idx)", login_needed="no (AWS Open Data; free to read)",
                       bytes_read=GEFS_RUN["bytes_total"], bytes_read_note=f"one HTTP Range request per day for the first 26 of 80 messages (about 9 MB of a 25 MB file) = {size_text(GEFS_RUN['bytes_range'])}, plus {size_text(GEFS_RUN['idx_bytes'])} of .idx files; "
                                                                                              f"{GEFS_RUN['days']:,} days; every GRIB file decoded and deleted at once", bytes_cached=_gefs_c00["bytes_on_disk"],
                       date_downloaded=f"{local_from_utc(GEFS_RUN['first_utc'])} to {local_from_utc(GEFS_RUN['last_utc'])[11:]} laptop time (UTC{UTC_OFFSET[:3]}); {GEFS_RUN['first_utc']} to {GEFS_RUN['last_utc']}",
                       script="code/fetch_gefs.py (environment ~/AI_Class/.venv-grib: eccodes, cfgrib)"),
         turned_into_input=(f"the file's overlapping 3 h and 6 h accumulations are differenced to non-overlapping 3-hourly rain (packing noise below 0 set to 0); basin mean = sum over boxes of weight x rain, the weight being the share of the "
                            f"basin's area in the box (exact intersection, equal-area projection EPSG:5070); local day t+k (k = 1, 2, 3) = the wall-clock day of the gauge, a 3-hour window cut at the day's edge is split in proportion; "
                            f"bias factor per basin and lead = sum of AORC rain / sum of forecast rain over the target days of the validation years ({F.SPLIT['val'][0]} to {F.SPLIT['val'][1]}), {GEFS_FACTOR_RANGE[0]:.2f} to {GEFS_FACTOR_RANGE[1]:.2f}, "
                            f"applied to every day. Three daily columns dated by the issue day (fs1..fs3 and variants) replace the AORC rain of the days t+1..t+3 from {F.SPLIT['val'][0]} on (the network is trained on the AORC rain and never sees a forecast in training)."),
         cached_in=dict(paths=["data/gefs/c00/apcp_YYYYMM.npz (+ .json provenance), 178 months", "data/gefs/basins/gefs_c00_<gauge_id>.csv (basin means)"], files=_gefs_c00["files"], bytes_on_disk=_gefs_c00["bytes_on_disk"]),
         role="extra input (the rain forecast for the next three days)", role_code="extra input",
         one_line=f"3-hourly rain forecasts of a weather model on a 0.25 degree grid, one run a day at 00 UTC, {GEFS_DATES[0][:4]} to {GEFS_DATES[1][:4]} (control member)",
         figure_line=["GEFSv12 reforecast", f"0.25° · 3-hourly · {GEFS_DATES[0][:4]}–{GEFS_DATES[1][:4]}"],
         imagery="Not images: the rain is a weather model's forecast. No capture date or sun angle applies.",
         quality=[f"correlation of the raw forecast with the AORC rain of the same local day, median of the 12 basins: {GSK['correlation_all_median_over_basins']['l1']:.2f} (day t+1), {GSK['correlation_all_median_over_basins']['l2']:.2f} (t+2), "
                  f"{GSK['correlation_all_median_over_basins']['l3']:.2f} (t+3); mean forecast / mean AORC {GSK['mean_ratio_all_median_over_basins']['l1']:.2f}, {GSK['mean_ratio_all_median_over_basins']['l2']:.2f}, {GSK['mean_ratio_all_median_over_basins']['l3']:.2f}; "
                  f"the bias has a season (a wetter control member in spring and summer: results/gefs_basins.json, caveats), which one factor per basin and lead does not remove"],
         sources=["data/gefs/README.md", "results/gefs_basins.json", "results/forecast_input.json (forecast_data)", "results/data_access.json (gefs_reforecast_record)", "data/gefs/c00/apcp_YYYYMM.json (provenance of every month)", doc("aws_gefs")],
         unverified=["Whether the operational control member had the same posting times in 2000-2014 (results/gefs_basins.json caveats): not checked.",
                     "The NOAA description document of the reforecast (Description_of_reforecast_data.pdf) was fetched but its text could not be extracted here; the facts above rest on the AWS registry page and the project's own checks."])

STAGE4 = E(key="stage4", name="NCEP Stage IV (Iowa Environmental Mesonet archive)", short="NCEP Stage IV", kind="hourly rain analysis from weather radar and rain gauges (the product a real-time service would read)",
           what_it_is=("The national hourly rain analysis of NOAA's NCEP: a mosaic of the regional multi-sensor analyses made by the 12 NWS River Forecast Centers, which combine rain estimated by the WSR-88D (NEXRAD) weather radars "
                       "with rain-gauge reports, with a manual quality-control step by a human analyst (Lin and Mitchell 2005). NCEP began to produce it routinely in late 2001 and it has been archived since January 2002."),
           producer="NOAA NCEP EMC with the NWS River Forecast Centers; copy used: Iowa Environmental Mesonet (Iowa State University), a volunteer-run archive",
           version="no version number; the Iowa copy may hold first postings or later versions of each hour (not established: results/stage4_realtime.json, honesty points)",
           spatial_grid="4.76 km polar-stereographic grid (1,121 x 881 cells of 4,763 m); a basin = the cells whose centre lies inside its outline",
           time_step="hourly (6-hourly and 24-hourly 12Z-12Z sums also exist)",
           period_used=(f"only the hours of the issue day (local midnight to midnight) of the {N_ONSETS} test-year onsets: {S4['data']['hours_needed']:,} hourly labels on {S4['data']['issue_days']} issue days "
                        f"(the archive holds 2002-2014 complete; 2000-2001 partial)"),
           time_stamp="end of hour, UTC: the file ST4.YYYYMMDDHH.01h.grib holds the hour that ends at that time (GRIB tags: reference time 23Z, forecast 3600 s, valid 00Z)",
           latency_real_time="a first analysis may appear an hour or two after the hour, with the contribution of one or two River Forecast Centers; hourly mosaics are finalized 12-18 hours later (Lin and Mitchell 2005); 'about 1 h behind' in results/data_access.json",
           obtained=dict(url_pattern="https://mesonet.agron.iastate.edu/archive/data/YYYY/MM/DD/stage4/ST4.YYYYMMDDHH.01h.grib", login_needed="no (the NCAR copy shows a sign-in link; not used)", bytes_read=int(sum(r["bytes"] for r in S4_RUNS["runs"])),
                         bytes_read_note=f"{S4_RUNS['files']:,} files of about 1.2 MB with at most {S4_RUNS['connections']} connections in {S4_RUNS['seconds'] / 3600:.2f} hours (the record gives {S4_RUNS['gb']} GB); every file reduced at once to the 12 basin means and dropped; "
                                         f"{S4['data']['sources_of_the_hours_needed']['builder_csv'] + S4['data']['sources_of_the_hours_needed']['review_cache']:,} more hours come from earlier fetches of the builder and the reviewer (their bytes are not in this figure)",
                         bytes_cached=S4_FILES_FACTS["bytes"], date_downloaded=f"{_s4_start} to {_s4_end} laptop time (UTC{UTC_OFFSET[:3]}), three runs", script="code/check_stage4_realtime.py (not part of run_all.sh)"),
           turned_into_input=(f"read with rasterio, reduced at once to the hourly basin mean (cells whose centre lies inside the outline); the sum over the wall-clock day of the issue day replaces the AORC rain of that day "
                              f"in the saved lstm_wqa networks (nothing is retrained): a real-time check. {S4['fallback']['hours_replaced_by_aorc']} hours (no usable file) were replaced by the AORC hour. On the issue days Stage IV agrees with AORC with "
                              f"correlation {S4A['correlation']:.2f} and a ratio of means {S4A['mean_ratio_stage4_over_aorc']:.2f}."),
           cached_in=dict(paths=["data/aorc/stage4_issue_days.csv (hourly basin means, 12 columns)", "data/aorc/stage_iv_hindsight.csv"], files=S4_FILES_FACTS["files"], bytes_on_disk=None),
           role="check (a real-time rain product in place of AORC on the issue day of every test-year onset)", role_code="check or map",
           one_line=f"hourly radar-and-gauge rain on a 4.76 km grid, 2002 on; the issue days of the {N_ONSETS} test-year onsets were read ({S4_RUNS['gb']} GB)",
           figure_line=["NCEP Stage IV", f"4.76 km · hourly · {F.SPLIT['test'][0][:4]}–{F.SPLIT['test'][1][:4]}, onset days"],
           imagery="Radar-derived rain estimates; no radar or satellite scene was looked at, so no capture date or sun angle applies.",
           sources=["results/stage4_realtime.json", "results/data_access.json", "data/aorc/stage4_issue_days_fetch_runs.json", doc("lin2005")],
           unverified=["Whether the Iowa archive holds first postings or the later, quality-controlled versions of each hour: not established by the project."])

ERA5 = E(key="era5", name="ERA5 hourly rain (Open-Meteo archive API)", short="ERA5 (Open-Meteo)", kind="a weather-model reanalysis: the model's rain, steered towards observations by data assimilation",
         what_it_is=("ERA5 is the fifth-generation ECMWF reanalysis: a weather model combined with observations from around the world (data assimilation), giving hourly estimates on a 0.25 degree grid from 1940. Its rain is the model's, "
                     "not measured. The project reads it through the Open-Meteo historical weather API (models=era5), one request per basin centroid."),
         producer="ECMWF (ERA5, distributed through the Copernicus Climate Data Store); served by Open-Meteo", version="ERA5 as served by Open-Meteo (models=era5); no release number is returned",
         spatial_grid=(f"0.25 degree grid; the service returns the grid point nearest the basin centroid, {min(ERA5_OFFSET_KM.values()):.0f} to {max(ERA5_OFFSET_KM.values()):.0f} km from it: a point value, not a basin mean "
                       f"(the basins are {_area[0]:,.0f} to {_area[1]:,.0f} km2)"),
         time_step="hourly", period_used=f"{DW['sources']['era5']['period'][0]} to {DW['sources']['era5']['period'][1]}: {N_HOURS:,} hours per basin, no hour missing",
         time_stamp="UTC; the value at time T is the rain of the preceding hour (Open-Meteo documentation: 'Preceding hour sum'), so the stamp is the end of the hour; if that were the hour after, every ERA5 reading of the day windows would move by one hour",
         latency_real_time="about 5 days (Copernicus: 'ERA5 is updated daily with a latency of about 5 days'): not a real-time product",
         obtained=dict(url_pattern="https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&start_date=1980-09-25&end_date=2014-10-05&hourly=precipitation&timezone=UTC&models=era5", login_needed="no key (free, non-commercial tier)",
                       bytes_read=HOURLY["era5"]["bytes"], bytes_read_note="size of the 12 cached answers (the transfer size was not recorded)", bytes_cached=HOURLY["era5"]["bytes"],
                       date_downloaded=dates_text(HOURLY["era5"]["first_modified"], HOURLY["era5"]["last_modified"]), script="code/check_day_windows.py"),
         turned_into_input=("each hour is assigned to the local calendar day in which it ends (zoneinfo America/New_York, America/Chicago for Leaf River, summer time included; a day has 24 hours, or 23 or 25 at the clock changes) and summed: "
                            "rain_to_midnight (extra input of E1, sets wh and wqh), rain_to_09 and rain_utc_day (matched-window controls), rain_to_midnight_older (control). It also fills the AORC hours that hold no value, and is the hourly reference of the day-window check."),
         cached_in=dict(paths=["data/hourly/era5_<gauge_id>.json"], files=HOURLY["era5"]["files"], bytes_on_disk=None),
         role="extra input of E1 (rain up to midnight), stand-in for AORC where it has holes, and an hourly reference for the clocks", role_code="extra input",
         one_line=f"hourly model rain at the 0.25 degree grid point nearest each basin centroid, {DW['sources']['era5']['period'][0][:4]} to {DW['sources']['era5']['period'][1][:4]}",
         figure_line=["ERA5 (Open-Meteo)", f"0.25° point · hourly · {DW['sources']['era5']['period'][0][:4]}–{DW['sources']['era5']['period'][1][:4]}"],
         imagery="Not images: a model's rain. ERA5 assimilates satellite radiances among its observations; none were looked at here. No capture date or sun angle applies.",
         quality=[f"daily totals (UTC day) correlate with AORC {AOS['median_daily_correlation_same_day']['era5']:.2f} (median of the 12 basins)"],
         sources=["results/day_windows.json (sources.era5)", "results/rain_to_midnight.json", "results/data_access.json", doc("openmeteo"), doc("era5_cds")])

MERRA2 = E(key="merra2", name="MERRA-2 via NASA POWER (rain, soil wetness, precipitable water)", short="MERRA-2 (POWER)", kind="a weather-model reanalysis (NASA): daily and hourly values at the basin centroid cell",
           what_it_is=("MERRA-2, NASA's Modern-Era Retrospective analysis for Research and Applications version 2: a weather model combined with satellite and other observations (data assimilation), from 1980. "
                       "The project reads, from the NASA POWER API at the basin centroid: PRECTOTCORR ('Precipitation Corrected': the average MERRA-2 bias-corrected total precipitation, in the POWER catalogue), GWETTOP, GWETROOT and GWETPROF (soil wetness of the top 5 cm, "
                       "the root zone and the whole profile, 0 = dry, 1 = saturated) and TQV (precipitable water, mm). The values are the model's, not satellite retrievals."),
           producer="NASA Global Modeling and Assimilation Office (GMAO); served by NASA POWER", version=f"MERRA-2 through the POWER API {SAT['api_version']} ({SAT['api_title']})",
           spatial_grid="0.5 (latitude) x 0.625 (longitude) degree cells: the cell that holds the basin centroid (about 55 x 57 km), one coarse cell for a basin (results/satellite_data.json: grid probe)",
           time_step="daily (UTC day) for the series in data/satellite; hourly rain for the day-window check",
           period_used=f"daily {SAT['coverage']['merra_prcp']['first_valid_day']} to {SAT['coverage']['merra_prcp']['last_day']}; hourly {DW['sources']['merra2']['period'][0]} to {DW['sources']['merra2']['period'][1]} (POWER serves hourly MERRA-2 from 2001 only)",
           time_stamp="UTC: the daily value is the UTC day (ends 19:00 local standard time; 18:00 in Leaf River, MS); the hourly key YYYYMMDDHH is the hour that starts at HH (confirmed: the daily value equals the sum of its hourly values, correlation 1.000)",
           latency_real_time="MERRA-2 is published weeks after the fact (results/satellite_data.json); the POWER hourly service appends the faster GEOS-IT product about 2 days behind real time (results/data_access.json)",
           obtained=dict(url_pattern="https://power.larc.nasa.gov/api/temporal/daily/point (and .../hourly/point?parameters=PRECTOTCORR), time-standard=UTC", login_needed="no",
                         bytes_read=POWER_BYTES["merra2_daily"]["bytes"] + HOURLY["merra2"]["bytes"], bytes_read_note="size of the cached answers: 12 daily files (5 parameters each) and 12 hourly files (the transfer size was not recorded)",
                         bytes_cached=POWER_BYTES["merra2_daily"]["bytes"] + HOURLY["merra2"]["bytes"], date_downloaded=f"daily {SAT['retrieved']}; hourly " + dates_text(HOURLY["merra2"]["first_modified"], HOURLY["merra2"]["last_modified"]),
                         script="code/fetch_satellite.py (daily), code/check_day_windows.py (hourly)"),
           turned_into_input=("daily columns merra_prcp, gwettop, gwetroot, gwetprof, pw (TQV) per basin in data/satellite/<gauge_id>.csv. Tested as extra inputs on the main split: soil wetness (top layer and root zone), precipitable water, both "
                              "(08_satellite_inputs.py, part b); the MERRA-2 rain and the hourly rain serve as checks of the clocks of the daily tables (10_rain_timing.py, check_day_windows.py)."),
           cached_in=dict(paths=["data/satellite/<gauge_id>.csv", "data/satellite/raw/merra_<gauge_id>.json", "data/hourly/merra_<gauge_id>.json"], files=POWER_TABLES["files"] + POWER_BYTES["merra2_daily"]["files"] + HOURLY["merra2"]["files"], bytes_on_disk=None),
           role="tested extra inputs (soil wetness, precipitable water) and a check of the day windows", role_code="extra input",
           one_line=f"model rain, soil wetness and precipitable water at the centroid cell (0.5 x 0.625 degree), daily from {SAT['coverage']['merra_prcp']['first_valid_day'][:4]}, hourly from {DW['sources']['merra2']['period'][0][:4]}",
           figure_line=["MERRA-2 (POWER)", f"0.5×0.625° · daily · {SAT['coverage']['merra_prcp']['first_valid_day'][:4]}–{SAT['coverage']['merra_prcp']['last_day'][:4]}"],
           imagery="A model reanalysis, not an image product (it assimilates satellite and other observations; none were looked at). No capture date or sun angle applies.",
           quality=[f"daily MERRA-2 rain against Daymet rain, median of the 12 basins: correlation {ME['median_correlation']:.2f}, mean bias {ME['median_bias_percent']:+.0f}% (one coarse cell per basin: for context only)"],
           sources=["results/satellite_data.json", "results/day_windows.json", "results/satellite_inputs.json", doc("merra2"), doc("power_sources")])

IMERG = E(key="imerg", name="GPM IMERG via NASA POWER (daily, Final Run)", short="IMERG (POWER)", kind="rain estimated from satellites: passive-microwave and infrared retrievals merged, adjusted to gauges",
          what_it_is=("NASA's Integrated Multi-satellitE Retrievals for GPM: rain estimated from the microwave sensors of the satellite constellation (the GPM Microwave Imager, the TRMM Microwave Imager and partner instruments), "
                      "filled in time with infrared data of geostationary satellites. The Final Run adjusts the satellite rain to monthly rain-gauge analyses (GPCC). The project reads the daily total, mm/day, from the NASA POWER API."),
          producer="NASA (GPM mission, Precipitation Processing System); served by NASA POWER", version=f"IMERG Final Run according to POWER (documented from 2001-01-01; the API returned values from {SAT['coverage']['imerg']['first_valid_day']}); the IMERG release number is not stated by POWER (release: unverified)",
          spatial_grid=(f"0.1 degree cells (about 10 km); a basin = the cells that overlap its outline, weighted by overlap area: {min(IMERG_NCELLS.values())} to {max(IMERG_NCELLS.values())} cells per basin, {SAT['n_imerg_cells_requested']} cells in all"),
          time_step="daily (UTC day only: POWER serves IMERG at no other time standard)",
          period_used=f"{SAT['coverage']['imerg']['first_valid_day']} to {SAT['coverage']['imerg']['last_day']}; the models use it from 2000 (satellite-era split: training {SPLIT_SAT['train'][0]} to {SPLIT_SAT['train'][1]})",
          time_stamp=(f"the UTC day: ends at 19:00 local standard time (18:00 in Leaf River, MS). Measured against hourly rain the day ends at {DE['imerg']['found_by_this_check_local']['era5']} (ERA5) or {DE['imerg']['found_by_this_check_local']['merra2']} (MERRA-2): "
                      f"+{DE['imerg']['hours_late_in_this_check']['era5']:.1f} and +{DE['imerg']['hours_late_in_this_check']['merra2']:.1f} hours off, which shows the accuracy of the check, not a different day"),
          latency_real_time="Final Run about 3.5 months behind real time; Early Run about 4 hours; Late Run about 14 hours (NASA IMERG page); the project tested only the Final Run",
          obtained=dict(url_pattern="https://power.larc.nasa.gov/api/temporal/daily/point?parameters=IMERG_PRECTOT&community=AG&longitude=..&latitude=..&time-standard=UTC", login_needed="no",
                        bytes_read=POWER_BYTES["imerg"]["bytes"], bytes_read_note=f"size of the {POWER_BYTES['imerg']['files']} cached answers, one per 0.1 degree cell (the transfer size was not recorded)",
                        bytes_cached=POWER_BYTES["imerg"]["bytes"], date_downloaded=dates_text(POWER_RAW["first_modified"], POWER_RAW["last_modified"]) + f"; results/satellite_data.json records 'retrieved' {SAT['retrieved']}", script="code/fetch_satellite.py"),
          turned_into_input=("one request per overlapping 0.1 degree cell at the cell centre (POWER returns the value of the cell that holds the point, no interpolation); basin value = sum(w_i x_i) / sum(w_i) with w_i = the geodesic "
                             "area of the overlap of cell i with the outline; column imerg of data/satellite/<gauge_id>.csv. Tested instead of, or beside, the Daymet rain on the satellite-era split (08_satellite_inputs.py, part a); "
                             "also compared with the rain tables to find their day offsets."),
          cached_in=dict(paths=["data/satellite/<gauge_id>.csv", "data/satellite/raw/imerg_<lon>_<lat>.json (153 cells)"], files=POWER_TABLES["files"] + POWER_BYTES["imerg"]["files"], bytes_on_disk=None),
          role="tested alternative or extra rain input (satellite era only) and a check of the day offsets", role_code="extra input",
          one_line=f"daily satellite rain on 0.1 degree cells (UTC day, Final Run), {SAT['coverage']['imerg']['first_valid_day'][:4]} to {SAT['coverage']['imerg']['last_day'][:4]}; microwave and infrared estimates merged, gauge-adjusted",
          figure_line=["IMERG (POWER)", f"0.1° · daily (UTC) · {SAT['coverage']['imerg']['first_valid_day'][:4]}–{SAT['coverage']['imerg']['last_day'][:4]}"],
          imagery=("Satellite-derived rain, used ready-made. The project did not process any satellite scene: no image, overpass time or sun angle was used. The sensors are passive microwave (GMI, TMI and partners) and "
                   "geostationary infrared; the value of a day is the UTC day's merged estimate, whatever the overpass times. No capture date or sun angle applies."),
          quality=[f"against Daymet rain, median of the 12 basins ({SAT['rain_check_period'][0]} to {SAT['rain_check_period'][1]}): daily correlation {IM['median_correlation']:.2f}, two-day sums {IM['median_correlation_two_day_sums']:.2f}, "
                   f"mean bias {IM['median_bias_percent']:+.1f}% (basins {IM['bias_percent_min']:+.0f}% to {IM['bias_percent_max']:+.0f}%); {100 * IM['median_heavy_days_seen_same_day']:.0f}% of the heaviest 5% of Daymet days are among IMERG's heaviest 5% the same day"],
          sources=["results/satellite_data.json", "results/day_windows.json (day_ends.imerg)", doc("imerg"), doc("power_sources")],
          unverified=["The IMERG release version (V06 or V07): POWER does not state it.", "Which IMERG run the days before 2001-01-01 belong to: not stated by the source (the documented Final Run starts 2001-01-01)."])

# ---- 8c. maps, tracks and the 15-minute flow ----------------------------------------------------------------------------------------------------------------
_el = GIS["sources"]["elevation"]
_lc = GIS["sources"]["land_cover"]
_elev_far = [GIS["basins"][g]["short"] for g in IDS if abs(GIS["basins"][g]["difference"]["elev_mean_m"]) > 25]
EVENTS = FA["events"]
_g12 = lambda f: f"{f:+.1f}"

TILES = E(key="elevation_tiles", name="1-arc-second elevation tiles in SRTM format (Mapzen/Tilezen terrain tiles)", short="Elevation tiles (SRTM format)", kind="a digital elevation model built from the Shuttle Radar Topography Mission and other elevation data",
          what_it_is=("A global elevation grid in the SRTM .hgt format, 1 arc-second (about 30 m), whole metres. The provider documents the tile set as a composite: NASA's Shuttle Radar Topography Mission (SRTM) is the land source at this "
                      "resolution, with USGS 3DEP (formerly NED) used in the United States at finer levels, and the tiles do not say which source a cell came from. The project therefore calls it 'elevation in SRTM format', not pure SRTM. "
                      "SRTM itself was flown on an 11-day mission of the Space Shuttle Endeavour in February 2000, with C-band and X-band radar antennas."),
          producer="Mapzen / Tilezen (Terrain Tiles on AWS Open Data), from NASA SRTM, USGS 3DEP and other sources",
          version=(f"{_el['year']} (results/basin_gis.json); the provider's version 1 was built in 2016Q2 and version 1.1 in 2017Q3. The SRTM release inside is not stated in the tiles: the tile builder's example configuration names "
                   "SRTMGL1.003 (NASA SRTM version 3, void-filled, folder 2000.02.11), the configuration of the production build was not read (SRTM release: unverified)"),
          spatial_grid=f"1 arc-second (about 30 m), 1 x 1 degree tiles of 3,601 x 3,601 cells, vertical reference {_el['vertical_reference'].split(' (')[0]}; {DEM['files']} tiles cover the {len(IDS)} basins; {DEM_CELLS:,} cells lie inside the outlines",
          time_step="not a time series (one acquisition)", period_used=f"SRTM data of February 2000; the tile set built in 2016-2017; {DEM_VOIDS} void cells inside the 12 basins",
          time_stamp="February 2000 (the shuttle mission); static", latency_real_time="none: a static data set",
          obtained=dict(url_pattern="https://s3.amazonaws.com/elevation-tiles-prod/skadi/N35/N35W083.hgt.gz (the pattern: skadi/<N|S><lat>/<N|S><lat><E|W><lon>.hgt.gz, named by the south-west corner)", login_needed="no (AWS Open Data)",
                        bytes_read=DEM["bytes"], bytes_read_note=f"size of the {DEM['files']} stored tiles (gzip files; the transfer size was not recorded)", bytes_cached=DEM["bytes"], date_downloaded=dates_text(DEM["first_modified"], DEM["last_modified"]),
                        script="code/07_basin_maps.py"),
          turned_into_input=(f"not an input of any model. 07_basin_maps.py cuts the tiles to each outline (cells whose centre lies inside it): mean elevation weighted by cos(latitude), slope by central differences in m per km, relief, and compares them with the "
                             f"CAMELS elev_mean and slope_mean (which the models read). Elevation: median difference {_g12(GC['elev_mean']['median_difference'])} m, more than 25 m in {', '.join(_elev_far)}, rank correlation {GC['elev_mean']['rank_correlation']:.2f}; "
                             f"slope on the 30 m grid is several times the CAMELS value (median {GC['slope_mean_by_grid']['30 m']['median_difference']:+.0f}%), and depends strongly on the grid."),
          cached_in=dict(paths=["data/dem/<tile>.hgt.gz"], files=DEM["files"], bytes_on_disk=CACHE["data/dem"]["bytes_on_disk"]),
          role="map and check of the CAMELS elevation and slope (no model reads it)", role_code="check or map",
          one_line=f"1 arc-second (30 m) elevation tiles in SRTM format, built 2016-2017 from SRTM (flown February 2000) and other sources; {DEM_VOIDS} voids in the basins",
          figure_line=["Elevation tiles (SRTM format)", "1 arc-second · static · composite"],
          imagery=("Elevation from radar, not from optical images. SRTM: C-band and X-band radar interferometry flown on the Space Shuttle Endeavour, February 2000 (an 11-day mission); the capture date is the mission, not a scene. "
                   "Radar is an active sensor, so a sun angle does not apply. The project used the finished elevation grid and processed no radar scene."),
          sources=["results/basin_gis.json (sources.elevation, notes)", doc("joerd"), doc("srtm_wiki")],
          unverified=["The SRTM release (version 2.1 non-void-filled or 3.0 void-filled) inside the tile set: not stated by the tiles; SRTMGL1.003 (void-filled) is what the builder's example configuration names, which is not proof for the 2016-2017 build."])

NLCD = E(key="nlcd", name="Annual NLCD land cover, 2011 layer (USGS / MRLC)", short="Annual NLCD (2011)", kind="a land-cover map classified from Landsat satellite images",
         what_it_is=(f"The yearly National Land Cover Database of the US Geological Survey: a 30 m map of the conterminous United States in land-cover classes (forest, pasture, cultivated crops, wetland, developed, water and others; "
                     f"{len(_lc['classes'])} classes occur in the 12 basins), made by USGS EROS from the Landsat satellite record with deep-learning models (Annual NLCD Collection 1.0 covers 1985 to 2023). The project reads the 2011 layer."),
         producer="US Geological Survey (EROS) with the Multi-Resolution Land Characteristics consortium",
         version=(f"the {_lc['year']} layer of the yearly series, read from the USGS GeoServer layer {_lc['layer']}; the collection number is not recorded in the project (the metadata read here is Collection 1.0, 1985-2023). "
                  f"It is not the older single-year 'NLCD 2011' release (collection: unverified)"),
         spatial_grid=f"30 m native grid, NAD83 / Conus Albers (EPSG:5070); {LC_CELLS:,} cells inside the 12 outlines, none without a class ({LC_UNCLASSIFIED:.4f} share)",
         time_step="one map per year; the 2011 map is used for the whole period", period_used=f"map year {_lc['year']} (inside the study period 1980-2014); treated as fixed although land cover changed over the 34 years",
         time_stamp=f"map year {_lc['year']}", latency_real_time="not a daily product; release lag not checked",
         obtained=dict(url_pattern="https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs (WCS 1.0.0 GetCoverage, one request per basin outline box, time=2011-01-01)", login_needed="no",
                       bytes_read=LANDCOVER["bytes"], bytes_read_note=f"size of the {LANDCOVER['files']} stored GeoTIFFs (the transfer size was not recorded); www.mrlc.gov could not be reached through the proxy and WCS 2.0.1 failed, WCS 1.0.0 worked",
                       bytes_cached=LANDCOVER["bytes"], date_downloaded=dates_text(LANDCOVER["first_modified"], LANDCOVER["last_modified"]), script="code/07_basin_maps.py"),
         turned_into_input=(f"not an input of any model. The shares of the {len(_lc['classes'])} classes, in {len(_lc['groups'])} groups, are counted in the 30 m cells whose centre lies inside each outline (an equal-area grid, so counts are shares of area); "
                            f"the forest share is compared with the CAMELS frac_forest: {GC['forest_share']['mean_difference']:+.1f} percentage points on average at 30 m, {GC['forest_share_1km_majority']['mean_difference']:+.1f} on a 990 m majority grid; and the maps are drawn."),
         cached_in=dict(paths=["data/landcover/annual_nlcd_2011_<gauge_id>.tif"], files=LANDCOVER["files"], bytes_on_disk=CACHE["data/landcover"]["bytes_on_disk"]),
         role="map and check of the CAMELS forest fraction (no model reads it)", role_code="check or map",
         one_line=f"30 m land-cover map of {_lc['year']} from the yearly USGS series (Landsat-based); {len(_lc['classes'])} classes in the 12 basins",
         figure_line=["Annual NLCD (2011)", "30 m · yearly · 2011 layer"],
         imagery=(f"Derived from Landsat satellite images (30 m) by USGS with deep-learning models; the map year is {_lc['year']}. The project used the finished map and classified no scene: the capture dates and sun angles of the Landsat "
                  "scenes behind a pixel are part of the producer's processing and were neither used nor recorded here, so they do not apply."),
         sources=["results/basin_gis.json (sources.land_cover, notes)", "code/07_basin_maps.py", doc("nlcd_meta")],
         unverified=["The Annual NLCD collection number (1.0, 1.1 or 1.2) of the 2011 layer read in October 2026: not recorded by the project."])

HURDAT = E(key="hurdat2", name="HURDAT2 Atlantic best-track storm records (NOAA National Hurricane Center)", short="HURDAT2", kind="storm tracks: the official best track of each Atlantic tropical cyclone, assembled after the season",
           what_it_is=("The National Hurricane Center's post-storm analysis of every Atlantic tropical and subtropical cyclone, made with all observations including those not available in real time: position, maximum wind and central pressure "
                       "every 6 hours (plus the times of landfall and intensity maxima), 1851 to 2025, revised after each season."),
           producer="NOAA National Hurricane Center",
           version=f"{HURDAT_FILE.name} (the NHC page read on {DOC_DAY} says it was updated on September 23rd, 2026 to include the revisions of the 1971 through 1975 seasons); sha256 {HURDAT_INFO['sha256'][:16]}... in results/flood_analysis.json",
           spatial_grid="points (latitude, longitude to 0.1 degree) along each track", time_step="6-hourly (00, 06, 12, 18 UTC) plus special times",
           period_used=f"{HURDAT_INFO['n_storms_in_file']:,} storms in the file (seasons to {HURDAT_SUFFIX[0]}); used for the {EVENTS['n_events']} test-year events ({F.SPLIT['test'][0]} to {F.SPLIT['test'][1]}) and for September 2004 at the French Broad gauge",
           time_stamp="UTC date and hour of each position", latency_real_time="post-season (best tracks are revised after the season): not a real-time product",
           obtained=dict(url_pattern=f"{HURDAT_INFO['url']} (the newest hurdat2-1851-YYYY-MMDDYY.txt found on https://www.nhc.noaa.gov/data/hurdat/)", login_needed="no",
                         bytes_read=HURDAT_DIR["bytes"], bytes_read_note="size of the stored file", bytes_cached=HURDAT_DIR["bytes"], date_downloaded=dates_text(HURDAT_DIR["first_modified"], HURDAT_DIR["last_modified"]), script="code/06_flood_analysis.py (also read by check_2004_floods.py)"),
           turned_into_input=(f"not an input of any model: the storm type of each of the {EVENTS['n_events']} events. 'Tropical' = a best-track position (every status, depressions and remnants included) lay within {EVENTS['hurdat']['radius_km']} km of the gauge on the onset day or the "
                              f"{EVENTS['hurdat']['days_before_onset']} days before: {EVENTS['by_storm_type']['tropical']} events ({EVENTS['tropical_within_300km']} within 300 km); otherwise warm-season (May-September: {EVENTS['by_storm_type']['warm-season']}) or "
                              f"cool-season (October-April: {EVENTS['by_storm_type']['cool-season']}). Nearness, not proof of cause."),
           cached_in=dict(paths=["data/hurdat/hurdat2-1851-2025-092326.txt"], files=HURDAT_DIR["files"], bytes_on_disk=CACHE["data/hurdat"]["bytes_on_disk"]),
           role="check (where and why the warnings fail: storm type)", role_code="check or map",
           one_line=f"6-hourly positions of {HURDAT_INFO['n_storms_in_file']:,} Atlantic storms, 1851-{HURDAT_SUFFIX[0]}; tags {EVENTS['by_storm_type']['tropical']} of the {EVENTS['n_events']} events as tropical",
           figure_line=["HURDAT2", f"6-hourly tracks · 1851–{HURDAT_SUFFIX[0]}"],
           imagery="Storm positions and intensities from the official analysis, not images. No capture date or sun angle applies.",
           sources=["results/flood_analysis.json (events.hurdat)", "results/floods_2004.json", doc("nhc_hurdat")])

USGS15 = E(key="usgs_iv", name="USGS 15-minute river flow (instantaneous values) and site file", short="USGS 15-min flow", kind="river flow measured at a gauge every 15 minutes",
           what_it_is=("The USGS Instantaneous Values Service: discharge readings (parameter 00060, cubic feet per second) 'usually made every 15 minutes and transmitted hourly'; and the USGS site file, which gives each gauge's time zone."),
           producer="US Geological Survey (Water Services)", version="values as served on 2026-10-03; the approval status of the 2012 values was not recorded",
           spatial_grid="the 12 gauges", time_step="15 minutes",
           period_used=f"{DW['sources']['usgs_flow']['period'][0]} to {DW['sources']['usgs_flow']['period'][1]} requested; compared {DW['river_flow_day']['days'][0]} to {DW['river_flow_day']['days'][1]} ({DW['river_flow_day']['per_gauge'][IDS[0]]['whole_period']['n_days']} days)",
           time_stamp="local clock time with a zone code (EST, EDT, CST, CDT); converted to UTC for the comparison", latency_real_time="readings are transmitted hourly (USGS): near real time",
           obtained=dict(url_pattern="https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites=<gauge_id>&parameterCd=00060&startDT=2012-01-01&endDT=2012-06-30 and https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=...&siteOutput=expanded",
                         login_needed="no", bytes_read=HOURLY["usgs_iv"]["bytes"] + HOURLY["usgs_sites"]["bytes"], bytes_read_note="size of the 12 flow files and the site file", bytes_cached=HOURLY["usgs_iv"]["bytes"] + HOURLY["usgs_sites"]["bytes"],
                         date_downloaded=dates_text(HOURLY["usgs_iv"]["first_modified"], HOURLY["usgs_iv"]["last_modified"]), script="code/check_day_windows.py"),
           turned_into_input=("not an input of any model: averaged over 24-hour windows shifted hour by hour and compared with the CAMELS daily flow of the same days; the best window is local midnight to midnight by the gauge's wall clock "
                              "(see the USGS daily flow). Six months of one year were checked, not the whole record."),
           cached_in=dict(paths=["data/hourly/usgs_iv_<gauge_id>.rdb", "data/hourly/usgs_sites.rdb"], files=HOURLY["usgs_iv"]["files"] + HOURLY["usgs_sites"]["files"], bytes_on_disk=None),
           role="check only (the clock of the daily flow)", role_code="check or map",
           one_line="15-minute flow of the 12 gauges, January to June 2012: shows that the daily flow is a local midnight-to-midnight mean",
           figure_line=["USGS 15-min flow", "gauge · 15 min · Jan–Jun 2012"],
           sources=["results/day_windows.json (river_flow_day, sources.usgs_flow)", doc("usgs_iv")])

DATASETS = [CAMELS, AORC, GEFS, STAGE4, ERA5, MERRA2, IMERG, TILES, NLCD, HURDAT, USGS15]
assert len({d["key"] for d in DATASETS}) == len(DATASETS) == 11
IMAGERY_NOTE = ("No optical satellite scene was classified or interpreted in this project, so capture date and sun angle do not apply. Satellite-derived products were used ready-made and are named for what they are: "
                f"GPM IMERG rain (passive-microwave and infrared estimates merged by NASA; the value of a day is the UTC day), the elevation tiles (a composite tile set in SRTM format: its main source, the SRTM radar mission of February 2000, is an active sensor, and the tiles do not say which source a cell came from), and the Annual NLCD land-cover map of {_lc['year']} "
                "(made by USGS from Landsat images; the map year is the date that applies). The weather tables (Daymet, NLDAS, MERRA-2, ERA5, AORC, Stage IV, GEFS) are gauge analyses, radar-and-gauge analyses, model reanalyses or forecasts: not images.")


# ---- 9. the extra inputs of the experiments (the side branch of the flow) ------------------------------------------------------------------------------------
COLUMN_CLOCK = {}
for _R in (R15, R16, R18):
    for _c in _R["inputs"]["clock"]:
        for _col in [t.strip() for t in re.sub(r"\([^)]*\)", "", _c["input"]).split(",") if t.strip()]:
            COLUMN_CLOCK.setdefault(_col, dict(covers=_c["covers"], source=_c.get("source", ""), result_file=None))
COLUMN_CLOCK["prcp_nldas"] = dict(covers=f"the NLDAS rain of the same day t: its day ends at about {DE['nldas']['day_ends_local_rounded']} local standard time, hours before the issue time", source="CAMELS basin_mean_forcing/nldas (10_rain_timing.py)", result_file="results/rain_timing.json")
COLUMN_CLOCK["prcp_nldas_older"] = dict(covers="the same NLDAS rain one day older (the control: it holds nothing the rain table does not already cover)", source="CAMELS basin_mean_forcing/nldas (10_rain_timing.py)", result_file="results/rain_timing.json")
_nets_by = lambda script: [n for n in NETS if n["script"] == script]
_ORDER = ["rain_to_midnight", "rain_to_09", "rain_utc_day", "rain_to_midnight_older", "aorc_to_midnight", "aorc_to_09", "aorc_utc_day", "p1", "p2", "p3", "fs1", "fs2", "fs3", "fk1", "fk2", "fk3", "fc1", "fc2", "fc3", "fx1", "fx2", "fx3", "fr1", "fr2", "fr3", "prcp_nldas"]
_extra = lambda script: sorted({c for n in _nets_by(script) for c in n["dyn"] if c not in F.FORCING}, key=lambda c: (_ORDER.index(c) if c in _ORDER else len(_ORDER), c))
_sets = lambda script: sorted({n["input_set"] for n in _nets_by(script) if n["input_set"]})
_col = lambda cols: [dict(column=c, **{k: v for k, v in COLUMN_CLOCK.get(c, dict(covers="", source="", result_file=None)).items() if k in ("covers", "source")}) for c in cols]
SAT_VARIANTS = {f"{e}.{v}": [c for c in x["daily_inputs"]] for e, ex in SI["experiments"].items() for v, x in ex["variants"].items()}
EXPERIMENTS = [
    dict(id="satellite", script="08_satellite_inputs.py", name="satellite rain, soil wetness and precipitable water", question=SI["experiments"]["a_rain_source"]["question"] + " " + SI["experiments"]["b_extra_inputs"]["question"],
         datasets=["imerg", "merra2"], sets=sorted(n for n in PRED_NAMES if n.startswith("sat_")), split="the satellite-era split for the rain source (training " + f"{SPLIT_SAT['train'][0]} to {SPLIT_SAT['train'][1]}), the main split for the extra inputs",
         extra_columns=[dict(column=c, covers=("the IMERG rain of the UTC day: replaces prcp or sits beside it" if c == "imerg" else "MERRA-2 daily value (UTC day) at the basin centroid cell"), source="data/satellite/<gauge_id>.csv")
                        for c in sorted({c for cols in SAT_VARIANTS.values() for c in cols if c not in F.FORCING})],
         control="the variants that leave the inputs as they are (gauge, base)", result_file="results/satellite_inputs.json"),
    dict(id="timing", script="10_rain_timing.py", name="NLDAS rain up to the evening of the issue day", question=RT["question"], datasets=["camels_nldas"], split="main split",
         sets=["wn", "wqn", "wn_older", "wqn_older"], extra_columns=_col(["prcp_nldas", "prcp_nldas_older"]), control="the same rain one day older (prcp_nldas_older)", result_file="results/rain_timing.json"),
    dict(id="E1", script="15_rain_to_midnight.py", name="ERA5 rain up to local midnight of the issue day", question=R15["question"], datasets=["era5", "camels_nldas"], split="main split", sets=_sets("15_rain_to_midnight.py"),
         extra_columns=_col(_extra("15_rain_to_midnight.py")), control="rain_to_midnight_older (the same series one day older); rain_to_09 and rain_utc_day (the same hours over the rain table's window and over the UTC day)", result_file="results/rain_to_midnight.json"),
    dict(id="E1b", script="16_aorc_rain_to_midnight.py", name="AORC rain up to local midnight of the issue day", question=R16["question"], datasets=["aorc"], split="main split", sets=_sets("16_aorc_rain_to_midnight.py"),
         extra_columns=_col(_extra("16_aorc_rain_to_midnight.py")), control="aorc_to_09 (the same AORC hours over the rain table's window) and aorc_utc_day; wqa_rep replaces the Daymet rain column by aorc_to_midnight", result_file="results/aorc_rain_to_midnight.json"),
    dict(id="E2", script="18_forecast_input.py", name="the GEFSv12 reforecast rain of the next three days", question=R18["question"], datasets=["gefs", "aorc"],
         split="main split (E2a: trained on the AORC rain of the target days, run on the forecast) and the satellite-era split (E2b: retrained with the forecast)", sets=_sets("18_forecast_input.py"),
         extra_columns=_col([c for c in _extra("18_forecast_input.py") if c != "aorc_to_midnight"]), control="the forecast shuffled within basin and calendar month across years (fk1..fk3, fx1..fx3): the right climate, no information about the day", result_file="results/forecast_input.json"),
    dict(id="E3", script="17_more_basins.py", name="63 training rivers instead of 12 (and 50 rivers for rivers never seen)", question=MB["question"], datasets=["camels_us"], split="main split", sets=_sets("17_more_basins.py") or ["lstm63_w", "lstm63_wq", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic"],
         extra_columns=[], control="the 12-river networks with the same settings and seeds", result_file="results/more_basins.json",
         note=f"no new column: the {len(candidate_ids_ := F.candidate_ids())} complete south-eastern candidates train one network ({sorted({n['n_basins'] for n in _nets_by('17_more_basins.py')})} basins in the saved networks)"),
]
# the sets of E3 carry no input_set in their files: name them by their file stems
EXPERIMENTS[-1]["sets"] = sorted({re.sub(r"_s\d+\.pt$", "", n["file"]) for n in _nets_by("17_more_basins.py")})
assert len(candidate_ids_) == A["n_complete"]
N_EXPERIMENT_COLUMNS = len({c["column"] for e in EXPERIMENTS for c in e["extra_columns"]})
CLOCK_DAYS = {}
for _zone in ("America/New_York", "America/Chicago"):
    _d = pd.date_range(*F.PERIOD)
    _a = _d.tz_localize(ZoneInfo(_zone)).tz_convert("UTC")
    _b = (_d + pd.Timedelta(days=1)).tz_localize(ZoneInfo(_zone)).tz_convert("UTC")
    _h = np.asarray((_b - _a) / pd.Timedelta(hours=1)).astype(int)
    CLOCK_DAYS[_zone] = {str(h): int((_h == h).sum()) for h in sorted(set(_h))}
N_BY_ZONE = {z: sum(v == z for v in GAUGE_ZONE.values()) for z in sorted(set(GAUGE_ZONE.values()))}


def local_day_example(zone, day):
    """The hour-ending UTC stamps that make up one local wall-clock day."""
    a = pd.Timestamp(day).tz_localize(ZoneInfo(zone)).tz_convert("UTC")
    b = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(ZoneInfo(zone)).tz_convert("UTC")
    return dict(zone=zone, local_day=day, first_stamp_utc=(a + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"), last_stamp_utc=b.strftime("%Y-%m-%d %H:%M"), hours=int((b - a) / pd.Timedelta(hours=1)))


LOCAL_DAY_EXAMPLES = [local_day_example("America/New_York", "2008-01-15"), local_day_example("America/New_York", "2008-07-15"), local_day_example("America/New_York", "2008-03-09"), local_day_example("America/New_York", "2008-11-02")]


# ---- 10. bytes, dates and the flow of the data -----------------------------------------------------------------------------------------------------------------
for _d in DATASETS:
    _d["obtained"]["bytes_read_kind"] = "transfer, recorded in a results file" if _d["key"] in ("aorc", "gefs", "stage4") else "size of the stored files (the transfer size was not recorded)"
BYTES = {d["key"]: int(d["obtained"]["bytes_read"]) for d in DATASETS}
BYTES["nasa_power_probes_and_documentation"] = int(POWER_BYTES["other"]["bytes"])
BYTES_TOTAL = sum(BYTES.values())
BYTES_TRANSFERS = sum(BYTES[k] for k in ("aorc", "gefs", "stage4"))
DOWNLOAD_DATES = {d["key"]: d["obtained"]["date_downloaded"] for d in DATASETS}
DL_WINDOWS = {"camels_us": (CAMELS_DOWNLOAD["first_modified"], CAMELS_DOWNLOAD["last_modified"]), "aorc": (AORC_DL["started"], _aorc_end), "gefs": (local_from_utc(GEFS_RUN["first_utc"]), local_from_utc(GEFS_RUN["last_utc"])),
              "stage4": (_s4_start, _s4_end), "era5": (HOURLY["era5"]["first_modified"], HOURLY["era5"]["last_modified"]), "merra2": (min(HOURLY["merra2"]["first_modified"], POWER_RAW["first_modified"]), max(HOURLY["merra2"]["last_modified"], POWER_RAW["last_modified"])),
              "imerg": (POWER_RAW["first_modified"], POWER_RAW["last_modified"]), "elevation_tiles": (DEM["first_modified"], DEM["last_modified"]), "nlcd": (LANDCOVER["first_modified"], LANDCOVER["last_modified"]),
              "hurdat2": (HURDAT_DIR["first_modified"], HURDAT_DIR["last_modified"]), "usgs_iv": (HOURLY["usgs_iv"]["first_modified"], HOURLY["usgs_iv"]["last_modified"])}
DL_DAYS = sorted({t[:10] for pair in DL_WINDOWS.values() for t in pair})
DL_SCRIPT_DAYS = sorted({t[:10] for k, pair in DL_WINDOWS.items() if k != "camels_us" for t in pair})
MB = lambda n: int(round(n / 1e6))
_cm = lambda p: CACHE[p]
SPLIT_YEARS = {p: round(PART_DAYS[p] / 365.25) for p in PART_DAYS}
_thr = np.array([THR[g] for g in IDS])
_share_above_train = [float((D[g].q_mm.values[TR] > THR[g]).mean()) for g in IDS]
N_UPDATE_SEQ, SEQ_DAYS = CFG["batch"], CFG["warmup"] + CFG["block"]
N_INPUTS_WQ = len(F.FORCING) + 1 + len(F.STATIC)
_assert_params = 4 * CFG["hidden"] * (N_INPUTS_WQ + CFG["hidden"] + 2) + (CFG["hidden"] * len(F.LEADS) + len(F.LEADS))
assert _assert_params == N_PARAMS_WQ, "the number of weights of the LSTM with the river gauge is not what 17 inputs, 64 cells and 3 outputs give"
BUILDER_NAMES = [b for b, _ in BUILDERS]
VERDICT_CRIT = FA["verdict"]["criteria"]
_chosen_by_land = {k: int(sum(b["landscape"] == k for b in A["selected"])) for k in A["complete_by_landscape"]}

STAGES = [
    dict(n=1, id="download", title="Raw downloads", counts=dict(products=len(DATASETS), gb_read=round(gb(BYTES_TOTAL), 1), gb_read_recorded_transfers=round(gb(BYTES_TRANSFERS), 1), gb_kept_under_data=round(gb(CACHE_DATA_BYTES), 1)),
         words=(f"{len(DATASETS)} data products were read from public services: CAMELS-US by hand on {DL_WINDOWS['camels_us'][0][:10]}, the rest by script on {', '.join(DL_SCRIPT_DAYS)}. About {gb(BYTES_TOTAL):.0f} GB were read "
                f"({gb(BYTES_TRANSFERS):.0f} GB of it as recorded transfers: AORC, GEFS, Stage IV); {gb(CACHE_DATA_BYTES):.1f} GB are kept under data/. No payment and no account is recorded for any of them."),
         scripts=["(by hand) CAMELS-US", "fetch_satellite.py", "07_basin_maps.py", "06_flood_analysis.py", "check_day_windows.py", "fetch_aorc.py", "fetch_gefs.py", "check_stage4_realtime.py"], outputs=["data/*"]),
    dict(n=2, id="audit", title="Audit of the basins", counts=dict(all_camels=A["n_all_camels"], candidates=A["n_candidates"], complete=A["n_complete"], chosen=len(A["selected"]), with_gaps=A["n_with_gaps"]),
         words=(f"{A['n_all_camels']} CAMELS basins -> {A['n_candidates']} candidates (HUC regions {' and '.join(f'{h:02d}' for h in RULE['huc_regions'])}, snow under {100 * RULE['max_snow_share']:.0f}% of precipitation, aridity under {RULE['max_aridity']:g}, "
                f"{RULE['area_km2'][0]}-{RULE['area_km2'][1]:,} km2) -> {A['n_complete']} complete (no missing flow day or weather value over {F.PERIOD[0]} to {F.PERIOD[1]}; {A['n_with_gaps']} rejected) -> {len(A['selected'])} chosen by rule "
                f"({RULE['per_landscape']} per landscape, evenly spaced in basin area; no flow statistic and no model score enters)."),
         scripts=["01_data_audit.py"], outputs=["results/data_audit.json", "results/basin_audit.csv", "figures/basin_selection.png"]),
    dict(n=3, id="caches", title="Caches on disk", counts=dict(derived_mb=MB(_cm("data/derived")["bytes_on_disk"]), hourly_mb=MB(_cm("data/hourly")["bytes_on_disk"]), satellite_mb=MB(_cm("data/satellite")["bytes_on_disk"]),
                                                                  aorc_mb=MB(_cm("data/aorc")["bytes_on_disk"]), gefs_mb=MB(_cm("data/gefs")["bytes_on_disk"]), camels_gb=round(gb(_cm("data/camels_us")["bytes_on_disk"]), 1)),
         words=("Every download is cached so that a script runs again offline: data/derived (the daily tables of the basins, parsed once), data/hourly (ERA5 and MERRA-2 hourly rain, USGS 15-minute flow), data/satellite (IMERG and MERRA-2 daily), "
                "data/aorc (hourly basin means), data/gefs (reforecast rain and basin means); sizes as `du` reports them."),
         scripts=["flood.load_data", "check_day_windows.py", "fetch_satellite.py", "fetch_aorc.py", "fetch_gefs.py", "gefs_to_basins.py"], outputs=["data/derived/*.pkl", "data/hourly", "data/satellite", "data/aorc", "data/gefs"]),
    dict(n=4, id="inputs", title="Daily inputs per basin", counts=dict(basins=len(IDS), days_per_basin=N_DAYS, basin_days=N_BASIN_DAYS, weather_columns=len(F.FORCING), flow_columns=1, properties=len(F.STATIC), missing_values=N_MISSING,
                                                                       extra_columns_of_the_experiments=N_EXPERIMENT_COLUMNS),
         words=(f"flood.load_data() reads the CAMELS files into one table per basin: {len(F.FORCING)} weather columns ({', '.join(F.FORCING)}) and the flow in mm/day, {N_DAYS:,} days, no value missing ({N_BASIN_DAYS:,} basin-days); "
                f"{len(F.STATIC)} basin properties are repeated on every day. Each experiment adds its own daily columns (rain to midnight, the forecast rain of days t+1..t+3 ...), built by its script."),
         scripts=["camels.py", "flood.py (load_data)"], outputs=["data/derived/camels12_daymet.pkl"]),
    dict(n=5, id="split", title="The split", counts=dict(train_days=PART_DAYS["train"], validation_days=PART_DAYS["val"], test_days=PART_DAYS["test"], total_days=N_DAYS, train_years=SPLIT_YEARS["train"], validation_years=SPLIT_YEARS["val"], test_years=SPLIT_YEARS["test"]),
         words=(f"By the day of the forecast target: training {F.SPLIT['train'][0]} to {F.SPLIT['train'][1]} ({PART_DAYS['train']:,} days), validation {F.SPLIT['val'][0]} to {F.SPLIT['val'][1]} ({PART_DAYS['val']:,}), test {F.SPLIT['test'][0]} to {F.SPLIT['test'][1]} "
                f"({PART_DAYS['test']:,}); together {N_DAYS:,}. Training fits the weights, the scalers and the high-flow levels; validation chooses settings and the stopping point; the test years are only scored."),
         scripts=["flood.py (SPLIT, part_mask)"], outputs=["no file: the dates are constants of code/flood.py (the satellite-era split is recorded in results/satellite_inputs.json)"]),
    dict(n=6, id="scaling", title="Scaling and the high-flow level", counts=dict(scaled_columns=len(F.FORCING) + 1 + len(F.STATIC), high_flow_percentile=int(round(100 * F.HIGH_Q)), level_min_mm_day=round(float(_thr.min()), 2), level_max_mm_day=round(float(_thr.max()), 2)),
         words=(f"Each input is put on one scale with the mean and standard deviation of the training years only: x_scaled = (x - mean_train) / sd_train (weather columns pooled over the training basins; the flow as log(1 + q); the properties over the training basins). "
                f"The high-flow level of a basin is the {100 * F.HIGH_Q:.0f}th percentile of its daily flow in the training years ({_thr.min():.2f} to {_thr.max():.2f} mm/day over the 12 basins): a day above it is a high-flow day."),
         scripts=["flood.py (fit, thresholds)"], outputs=["results/models/*.pt (key sc: the scalers of each saved network)", "results/flood_analysis.json and results/baselines.json (the high-flow levels)"]),
    dict(n=7, id="training", title="Training", counts=dict(weights=N_PARAMS_WQ, inputs=N_INPUTS_WQ, memory_cells=CFG["hidden"], sequences_per_update=N_UPDATE_SEQ, sequence_days=SEQ_DAYS, max_updates=CFG["updates"], check_every_updates=CFG["eval_every"], patience_checks=CFG["patience"], seeds="3 or 5"),
         words=(f"One sequence-to-sequence LSTM ({CFG['hidden']} memory cells, {N_PARAMS_WQ:,} weights with the river gauge, {N_INPUTS_WQ} inputs) reads a basin day by day and forecasts the flow of the next {len(F.LEADS)} days at every day. "
                f"Each update uses {N_UPDATE_SEQ} random sequences of {SEQ_DAYS} days ({CFG['warmup']} days read, {CFG['block']} scored) from the training years; every {CFG['eval_every']} updates the mean validation NSE is checked and training stops after {CFG['patience']} checks "
                f"without a gain (at most {CFG['updates']:,} updates); the best state is kept. Three random seeds per input set in 04_lstm_all.py and 10_rain_timing.py; five in the experiments of scripts 15 to 18, whose seeds 3 and 4 were added on 30 October 2026 "
                f"on a second machine after the three-seed scores were known (choices_made_after_looking_at_test_scores in their results files)."),
         scripts=["flood.py (fit)", "04_lstm_all.py and the experiment scripts"], outputs=["results/models/*.pt (the experiments of 15_ to 18_)"]),
    dict(n=8, id="forecasts", title="Forecasts stored", counts=dict(prediction_sets=len(PRED_NAMES), single_training_sets=N_PRED_SINGLE, saved_networks=len(NET_FILES), trainings_recorded=sum(N_TRAININGS_RECORDED.values())),
         words=(f"Every forecast is stored by its target day in results/preds/<model>_<inputs>.npz: {len(PRED_NAMES)} prediction sets ({N_PRED_SINGLE} of them single trainings), so any score can be recomputed without training; "
                f"{len(NET_FILES)} trained networks are saved (results/models) with their scalers, and a reloaded network reproduces its stored forecasts bit for bit. The results files record {sum(N_TRAININGS_RECORDED.values())} trainings."),
         scripts=["flood.py (save_preds, load_preds)"], outputs=["results/preds/*.npz", "results/models/*.pt"]),
    dict(n=9, id="scores", title="Scores", counts=dict(test_days=PART_DAYS["test"], onsets=N_ONSETS, basins=len(IDS), leads=len(F.LEADS)),
         words=(f"Test years only: median NSE over the {len(IDS)} basins, the onsets warned (an onset is a high-flow day whose forecast was issued while the river was still at or below the high-flow level: {N_ONSETS} in the test years), the false warnings, "
                f"and a verdict by criteria fixed in advance (GO: {VERDICT_CRIT['GO']}; CAUTION: {VERDICT_CRIT['CAUTION']}; otherwise NOT YET; {VERDICT_CRIT['definitions']}). Headline, LSTM with the river gauge, 1 day ahead: median NSE {HEAD['nse_median']:.2f}, "
                f"{100 * HEAD['onset']['pod']:.0f}% of onsets warned, {100 * HEAD['onset']['far']:.0f}% of warnings false: {VERDICT}."),
         scripts=["flood.py (evaluate)", "06_flood_analysis.py"], outputs=["results/flood_analysis.json", "results/flood_events.csv"]),
    dict(n=10, id="documents", title="Documents", counts=dict(builders=len(BUILDERS)),
         words=(f"{len(BUILDERS)} builders ({', '.join(BUILDER_NAMES)}) write the report, the slides, the question sheet, the guide, the overview, the plain-words page and the presentation plan from results/*.json and figures/: "
                f"no number is typed into a document. This file (results/data_flow.json) holds the data text they share."),
         scripts=BUILDER_NAMES, outputs=[f"{b}: {c}" for b, c in BUILDERS]),
]
STAGES[0]["note"] = "bytes of the products with no recorded transfer are the sizes of the stored files"
FLOW = dict(how_to_read="stages in the order the data pass through them; counts are read or computed from the files named in results/data_flow.json inputs_read", stages=STAGES,
            side_branch=dict(title="Extra inputs of the experiments: each adds daily columns (or training rivers) at stage 4, built by its own script", experiments=EXPERIMENTS),
            checks_and_maps=[dict(dataset="stage4", use="real-time check of the 611 onsets (Stage IV in place of AORC on the issue day)", script="check_stage4_realtime.py"),
                             dict(dataset="usgs_iv, era5, merra2", use="which 24 hours each daily value covers (the clocks of the tables)", script="check_day_windows.py"),
                             dict(dataset="hurdat2", use="storm type of each event (where and why the warnings fail)", script="06_flood_analysis.py"),
                             dict(dataset="elevation_tiles, nlcd", use="basin maps and a check of the CAMELS elevation, slope and forest fraction", script="07_basin_maps.py")],
            other_split=dict(name="satellite-era split (08_satellite_inputs.py, 18_forecast_input.py E2b)", parts={p: dict(first=SPLIT_SAT[p][0], last=SPLIT_SAT[p][1], days=PART_DAYS_SAT[p]) for p in SPLIT_SAT}),
            documents_present_when_written=DOCUMENTS)


# ---- 11. how the dataset is computed ---------------------------------------------------------------------------------------------------------------------------
_w_im = [c["overlap_km2"] for c in [dict(overlap_km2=x[2]) for x in SAT["cells_per_basin"][IDS[0]]["cells"]]]
_im_share = [w / sum(_w_im) for w in _w_im]
_gw = [x["weight"] for x in GB["basins"][IDS[0]]["weights"]]
_clock = lambda e: DE[e]["day_ends_local"]
HOW = dict(
    flow_mm_per_day=dict(
        title="River flow in mm per day",
        words=("The USGS gives flow in cubic feet per second. One cubic foot is 0.0283168 cubic metres and a day has 86,400 seconds, so cfs x 0.0283168 x 86,400 is the volume that passed the gauge in the day, in cubic metres; "
               "dividing by the basin area in square metres gives metres of water over the basin, and x 1000 gives millimetres. Rain and flow are then in the same unit."),
        formula="q_mm = cfs x 0.0283168 x 86400 / area_m2 x 1000", worked=Q_EXAMPLE,
        area=(f"area_m2 is line 3 of the basin's forcing file (the geospatial-fabric polygon), not the GAGES-II area of the attribute table; the two differ by {min(_area_diff):+.1f}% to {max(_area_diff):+.1f}% in the {len(IDS)} basins "
              f"(results/basin_gis.json: largest {GC['area_vs_gages2']['largest_difference']:.1f}%, {GC['area_vs_gages2']['largest_difference_basin']})"),
        check=f"recomputed from the raw flow files for all {N_BASIN_DAYS:,} basin-days: the largest difference from the cache is {MAXDIFF_FLOW:g}", where=["code/camels.py (load_basin)"]),
    basin_mean_with_overlap_weights=dict(
        title="A basin mean with overlap-area weights",
        words=("A grid cell lies wholly or partly inside a basin. IMERG (0.1 degree) and GEFS (0.25 degree) cells are large against the basins, so every cell that overlaps the outline counts in proportion to the area of the overlap. "
               "For AORC (1/120 degree) and Stage IV (4.76 km) the cells are small, so the basin mean is the plain mean of the cells whose centre lies inside the outline (their areas differ by under 1% within a basin)."),
        formula="x_basin = sum_i (w_i x_i) / sum_i w_i, with w_i = area of the overlap of cell i with the basin outline (geodesic area on the GRS80 ellipsoid for IMERG, equal-area projection EPSG:5070 for GEFS)",
        worked=dict(basin=F.SHORT[IDS[0]], imerg_cells=len(_w_im), imerg_weights_smallest=round(min(_im_share), 3), imerg_weights_largest=round(max(_im_share), 3),
                    gefs_boxes=len(_gw), gefs_weights=[round(w, 3) for w in _gw], gefs_weights_sum=round(sum(_gw), 3)),
        where=["code/fetch_satellite.py (basin_cells)", "code/gefs_to_basins.py (cell_weights)", "code/fetch_aorc.py (cell_index)", "code/check_stage4_realtime.py"]),
    value_at_the_centroid=dict(
        title="A value at the basin centroid",
        words=("ERA5 (the 0.25 degree grid point nearest the centroid, which Open-Meteo returns) and MERRA-2 (the 0.5 x 0.625 degree cell that holds the centroid) stand for the whole basin by one grid point or cell; "
               f"the ERA5 point lies {min(ERA5_OFFSET_KM.values()):.0f} to {max(ERA5_OFFSET_KM.values()):.0f} km from the centroid of basins of {_area[0]:,.0f} to {_area[1]:,.0f} km2."),
        formula="x_basin = x(grid point or cell nearest the centroid)", worked=dict(era5_point_km_from_centroid={F.SHORT[g]: round(ERA5_OFFSET_KM[g], 1) for g in IDS}), where=["code/check_day_windows.py", "code/fetch_satellite.py"]),
    local_day_sum=dict(
        title="Rain of a local wall-clock day from hourly rain",
        words=("A forecast is issued at local midnight at the end of day t, and hourly rain is stamped in UTC by the end of the hour. Every hour is given to the local calendar day in which it ends, by the gauge's wall clock "
               "(zoneinfo: America/New_York for 11 basins, America/Chicago for Leaf River; summer time included), and the hours of a day are summed. A day has 24 hours, or 23 or 25 where the clocks change "
               f"({CLOCK_DAYS['America/New_York']['23']} days of 23 hours and {CLOCK_DAYS['America/New_York']['25']} days of 25 hours in the {N_YEARS:.0f} years). No hour after the issue time is summed: proved for every hour by the wall clock, in UTC time, "
               "by a second route from the time-zone rules, and by replacing the later hours with made-up values (15_rain_to_midnight.py, 16_aorc_rain_to_midnight.py)."),
        formula="R_t = sum of the hourly rain r_h over the hours h whose end lies in (00:00, 24:00] of local day t", worked=LOCAL_DAY_EXAMPLES, days_by_length=CLOCK_DAYS, gauges_by_zone=N_BY_ZONE,
        where=["code/15_rain_to_midnight.py", "code/16_aorc_rain_to_midnight.py", "code/gefs_to_basins.py (aorc_local_days)"]),
    where_each_daily_value_ends=dict(
        title="Where each daily value ends (measured, local standard time)",
        words=(f"Each day's value was compared with hourly rain over 24-hour windows shifted hour by hour; the window that fits best gives the hour at which the product's 'day d' ends. Daymet rain ends at about {DE['daymet']['day_ends_local_rounded']} on day d "
               f"(median basin {_clock('daymet')}; basins {DE['daymet']['basin_range_local'][0]} to {DE['daymet']['basin_range_local'][1]}), NLDAS rain at about {DE['nldas']['day_ends_local_rounded']} ({_clock('nldas')}), IMERG and the daily MERRA-2 at the UTC day (19:00; 18:00 in Leaf River, MS), "
               f"the river flow at local midnight; the Maurer day has no single end. The forecast is issued at local midnight, so the afternoon and evening rain of the issue day is filed under the next day in the Daymet table. "
               f"The hourly references carry timing errors of their own of one to two hours (results/day_windows.json, calibration)."),
        values=dict(daymet=dict(median=_clock("daymet"), rounded=DE["daymet"]["day_ends_local_rounded"], basin_range=DE["daymet"]["basin_range_local"], readings_range=DE["daymet"]["range_over_readings_local"]),
                    nldas=dict(median=_clock("nldas"), rounded=DE["nldas"]["day_ends_local_rounded"], basin_range=DE["nldas"]["basin_range_local"], readings_range=DE["nldas"]["range_over_readings_local"]),
                    maurer=dict(median=_clock("maurer"), basin_range=DE["maurer"]["basin_range_local"]),
                    imerg=dict(documented="19:00 (the UTC day)", found=DE["imerg"]["found_by_this_check_local"]),
                    river_flow="local midnight (wall clock, summer time included)"),
        where=["code/check_day_windows.py", "results/day_windows.json"]),
    scaled_inputs=dict(
        title="Scaled inputs",
        words=(f"Each column is put on one scale with the mean and standard deviation of the training years ({F.SPLIT['train'][0]} to {F.SPLIT['train'][1]}), pooled over the training basins; the flow enters as log(1 + q_mm); the "
               f"{len(F.STATIC)} basin properties are scaled over the training basins. The values are fitted on the training years only, so nothing of the validation or test years leaks into the scale (recomputed here and equal to the scalers stored in a saved network)."),
        formula="x_scaled = (x - mean_train) / sd_train", worked=dict(column="prcp", unit=UNITS["prcp"], mean_train=round(float(SC_MU[0]), 2), sd_train=round(float(SC_SD[0]), 2), example_x=10.0,
                                                                  example_scaled=round((10.0 - float(SC_MU[0])) / float(SC_SD[0]), 2)),
        where=["code/flood.py (fit, _design)"]),
    nse=dict(
        title="Nash-Sutcliffe efficiency (NSE)",
        words="1 minus the squared error of the forecast divided by the squared error of always forecasting the mean of the scored days: 1 is perfect, 0 is no better than that mean, below 0 is worse. Scored per basin and lead on the test years; the median over the basins is reported.",
        formula="NSE = 1 - sum (sim - obs)^2 / sum (obs - mean(obs))^2", worked=dict(model="LSTM with the river gauge, 1 day ahead, test years", median_nse_over_the_12_basins=round(float(HEAD["nse_median"]), 3)), where=["code/flood.py (nse, evaluate)"]),
    high_flow_level_and_onset=dict(
        title="The high-flow level and the onset",
        words=(f"A basin's high-flow level is the {100 * F.HIGH_Q:.0f}th percentile of its daily flow in the training years: a day above it is a high-flow day. A warning is given when the forecast exceeds the level (times a warning level chosen on the validation years). "
               f"An onset is a high-flow day whose forecast was issued while the river was still at or below the level; 1 day ahead that is exactly the first day of each event (consecutive high-flow days). "
               f"The {len(IDS)} basins have {N_ONSETS} onsets in the test years. The share of onsets warned is the main score."),
        formula=f"level_b = quantile_{F.HIGH_Q:g}(q_mm of basin b over the training years); high day: q > level_b; onset at lead k: q(t+k) > level_b and q(t) <= level_b",
        worked=dict(levels_mm_per_day={F.SHORT[g]: round(THR[g], 2) for g in IDS}, mean_share_of_training_days_above=round(float(np.mean(_share_above_train)), 4), onsets=N_ONSETS), where=["code/flood.py (thresholds, flood_scores)", "code/06_flood_analysis.py"]),
    verdict=dict(
        title="The verdict: GO, CAUTION or NOT YET", words="Criteria fixed before the scores were looked at, applied to the test years at the warning level chosen on the validation years, pooled over the 12 basins and per basin, for all high-flow days and for onsets.",
        criteria=VERDICT_CRIT, worked=dict(model="LSTM with the river gauge, 1 day ahead", onsets_warned=round(float(HEAD["onset"]["pod"]), 3), warnings_false=round(float(HEAD["onset"]["far"]), 3), verdict=VERDICT), where=["code/06_flood_analysis.py", "results/flood_analysis.json"]),
    forecast_rule=dict(
        title="What a forecast may read",
        words=(f"A forecast is issued at the end of day t, knowing the weather (and, for the models with the river gauge, the flow) up to and including day t, and gives the flow of days t+1, t+2 and t+3. A sample belongs to the part of the split that contains its target day. "
               f"The network is causal (changing later days cannot change an earlier output: asserted before every training)."),
        formula="forecast(t, k) = f(inputs up to day t), k = 1, 2, 3", worked=dict(samples_per_basin={p: v for p, v in FORECAST_SAMPLES.items()}), where=["code/flood.py (module docstring, part_mask, _assert_causal)"]),
    gefs_bias_factor_and_issue_time=dict(
        title="The GEFS forecast: run, local days and bias factor",
        words=("The run used is " + GB["rule"]["run"] + ". The days: " + GB["rule"]["local_day"] + ". The bias factor: " + GB["bias"]["rule"] + "."),
        formula="factor_(b,k) = sum AORC(t+k) / sum forecast(t+k) over target days in the validation years; forecast_bc = forecast_raw x factor_(b,k)",
        worked=dict(factor_range=[round(GEFS_FACTOR_RANGE[0], 2), round(GEFS_FACTOR_RANGE[1], 2)], fitted_on=GB["bias"]["fitted_on_target_days"]), where=["code/gefs_to_basins.py"]),
    training_loss_and_stopping=dict(
        title="Loss and stopping point",
        words=(f"The loss is the squared error of the scaled flow, weighted so that wet and dry basins count nearly alike (the weight is about 1 over the basin's flow variance in the training years). "
               f"Every {CFG['eval_every']} updates the model is scored on the validation years of the same basins (mean NSE over basins and leads); the best state is kept and training stops after {CFG['patience']} checks without a gain or at {CFG['updates']:,} updates."),
        formula="loss = mean over scored days of w_b x (forecast_scaled - flow_scaled)^2, w_b = 1 / (sd_b / sd_all + 0.1)^2", worked=dict(settings=CFG), where=["code/flood.py (fit)"]),
)


# ---- 12. all the information: counts, sizes, dates ----------------------------------------------------------------------------------------------------------------
_sel = {b["gauge_id"]: b for b in A["selected"]}
_state = lambda g: re.search(r"([A-Z]{2})\.?$", B.loc[g, "gauge_name"].strip()).group(1)
BASIN_TABLE = []
for g in IDS:
    b = _sel[g]
    BASIN_TABLE.append(dict(gauge_id=g, name=F.SHORT[g], camels_name=b["gauge_name"], state=_state(g), landscape=b["landscape"], gauge_lat=b["gauge_lat"], gauge_lon=b["gauge_lon"], time_zone=GAUGE_ZONE[g],
                            area_km2_gages2=b["area_gages2"], area_km2_outline=round(AREA_KM2[g], 1), area_km2_forcing_file=round(FLOWFACTS[g]["area_km2_forcing_file"], 1), elev_mean_m=b["elev_mean"],
                            mean_rain_mm_day=b["p_mean_mm"], mean_flow_mm_day=b["q_mean_mm"], runoff_ratio=b["runoff_ratio"], baseflow_index=b["baseflow_index"], q_p95_mm_day=b["q_p95_mm"], q_max_mm_day=b["q_max_mm"],
                            high_flow_level_mm_day=round(THR[g], 3), days_estimated_by_usgs=FLOWFACTS[g]["estimated"], estimated_share=round(FLOWFACTS[g]["estimated_share"], 4), zero_flow_days=FLOWFACTS[g]["zero_flow"],
                            zero_flow_share=round(FLOWFACTS[g]["zero_flow_share"], 4), imerg_cells=IMERG_NCELLS[g], gefs_boxes=GEFS_NBOXES[g], aorc_cells=AORC_NCELLS[g], era5_point_km_from_centroid=round(ERA5_OFFSET_KM[g], 1)))
_est_days = sum(FLOWFACTS[g]["estimated"] for g in IDS)
_zero_days = sum(FLOWFACTS[g]["zero_flow"] for g in IDS)
_zip_ts = zipfile.ZipFile(ZIP_TS)
_n_zip_entries = len(_zip_ts.namelist())
_n_hours_merra = sorted({v["n_hours"] for v in DW["sources"]["merra2"]["per_basin"].values()})
_gefs_values = GEFS_RUN["days"] * GB["grid"]["n_windows"] * GEFS_NLAT * GEFS_NLON
_scalers = dict(fitted_on=f"the training years {F.SPLIT['train'][0]} to {F.SPLIT['train'][1]}, pooled over the {len(IDS)} basins ({int(TR.sum()) * len(IDS):,} basin-days)",
                weather=[dict(column=c, unit=UNITS[c], mean=float(SC_MU[i]), sd=float(SC_SD[i])) for i, c in enumerate(F.FORCING)], flow=dict(unit="mm/day", **SC_Q),
                properties=[dict(name=p, mean=float(SC_ST_MU[i]), sd=float(SC_ST_SD[i])) for i, p in enumerate(F.STATIC)], equal_to_the_scalers_stored_in_a_saved_network=SC_CHECK)
ALL = dict(
    basins=dict(all_camels=A["n_all_camels"], candidates=A["n_candidates"], with_gaps=A["n_with_gaps"], complete=A["n_complete"], chosen=len(IDS), rule=RULE, candidates_by_landscape=A["candidates_by_landscape"],
                complete_by_landscape=A["complete_by_landscape"], chosen_by_landscape=_chosen_by_land, worst_gaps=A["worst_missing"],
                rule_text=(f"candidates: HUC regions {', '.join(f'{h:02d}' for h in RULE['huc_regions'])}, snow under {100 * RULE['max_snow_share']:.0f}% of precipitation, aridity under {RULE['max_aridity']:g}, area {RULE['area_km2'][0]} to {RULE['area_km2'][1]:,} km2; "
                           f"complete = no flow day and no weather value missing over {F.PERIOD[0]} to {F.PERIOD[1]}; chosen = {RULE['per_landscape']} per landscape (mountains above 600 m, southern lowlands south of 32.5 N, the rest), evenly spaced in rank of area (01_data_audit.py)"),
                extra_files_in_the_unzipped_folders={s: n for s, n in EXTRA_FILES.items()}),
    period=dict(first=F.PERIOD[0], last=F.PERIOD[1], water_years=round(N_YEARS), days_per_basin=N_DAYS, files_cover=dict(first=FLOWFACTS[IDS[0]]["file_first"], last=FLOWFACTS[IDS[0]]["file_last"], days=FLOWFACTS[IDS[0]]["file_rows"])),
    records=dict(basin_days=N_BASIN_DAYS, values_weather_and_flow=N_VALUES, basin_property_values=len(IDS) * len(F.STATIC), missing_values=N_MISSING, missing_flow_days_in_the_12=int(sum(FLOWFACTS[g]["cfs_missing"] for g in IDS)),
                 estimated_flow_days=_est_days, estimated_share_of_all_basin_days=round(_est_days / N_BASIN_DAYS, 4), estimated_share_per_basin_min=round(min(_n_est), 4), estimated_share_per_basin_median=round(float(np.median(_n_est)), 4), estimated_share_per_basin_max=round(max(_n_est), 4),
                 estimated_share_all_78_candidates_median=round(A["median_estimated_share"], 4), estimated_share_all_78_candidates_max=round(A["max_estimated_share"], 4),
                 zero_flow_days=_zero_days, zero_flow_share_of_all_basin_days=round(_zero_days / N_BASIN_DAYS, 4), basins_with_zero_flow_days=[dict(basin=F.SHORT[g], days=FLOWFACTS[g]["zero_flow"], share=round(FLOWFACTS[g]["zero_flow_share"], 4)) for g in _zero_basins],
                 flow_conversion_largest_difference_from_cache=MAXDIFF_FLOW),
    split=dict(main=dict(first_last={p: list(F.SPLIT[p]) for p in F.SPLIT}, days=PART_DAYS, water_years=SPLIT_YEARS), satellite_era=dict(first_last={p: list(SPLIT_SAT[p]) for p in SPLIT_SAT}, days=PART_DAYS_SAT),
               forecast_samples_per_basin=FORECAST_SAMPLES),
    basin_table=BASIN_TABLE,
    static_properties=dict(items=STATIC_INFO, climate_averaging_period=CLIM_AVG, climate_properties=STATIC_CLIM,
                           note=f"{', '.join(STATIC_CLIM)} are averages over {CLIM_AVG[0]} to {CLIM_AVG[1]}, which overlaps the validation and test years; each is one number per basin and describes the climate, not any day"),
    scalers=_scalers,
    high_flow=dict(percentile=100 * F.HIGH_Q, fitted_on="training years", level_mm_per_day={g: round(THR[g], 4) for g in IDS}, equal_to_results_flood_analysis=True, onsets_in_the_test_years=N_ONSETS,
                   events_in_the_test_years=EVENTS["n_events"], events_lasting_one_day=EVENTS["one_day_events"], median_event_days=EVENTS["median_duration_days"], by_storm_type=EVENTS["by_storm_type"]),
    model=dict(settings=CFG, inputs_with_river_gauge=N_INPUTS_WQ, weights_with_river_gauge=N_PARAMS_WQ, weights_by_input_set_of_script_15=N_PARAMS, chosen_on="the validation years (04_lstm_all.py: three candidates, hidden 64 / lr 0.002 chosen)"),
    stored=dict(prediction_sets=len(PRED_NAMES), prediction_sets_by_family=PRED_FAMILY, single_training_sets=N_PRED_SINGLE, saved_networks=len(NET_FILES),
                saved_networks_by_script={s: sum(n["script"] == s for n in NETS) for s in sorted({n["script"] for n in NETS})}, trainings_recorded=sum(N_TRAININGS_RECORDED.values()), trainings_recorded_by_file=N_TRAININGS_RECORDED,
                folder_sizes={"results/preds": CACHE["results/preds"]["bytes_on_disk"], "results/models": CACHE["results/models"]["bytes_on_disk"]}),
    hourly_and_other_records=dict(hours_per_basin=N_HOURS, era5_hours_all_basins=N_HOURS * len(IDS), aorc_hours_all_basins=N_HOURS * len(IDS), aorc_filled_basin_hours=_gap["total_hours_filled"], aorc_partial_basin_hours=AORC_PARTIAL_HOURS,
                                  merra2_hourly_hours_per_basin=_n_hours_merra, gefs_runs=GEFS_RUN["days"], gefs_windows_per_run=GB["grid"]["n_windows"], gefs_boxes_in_the_cut=GEFS_NLAT * GEFS_NLON, gefs_values_stored=_gefs_values,
                                  gefs_issue_days=GB["issue_days"]["n"], stage4_hourly_labels=S4["data"]["hours_needed"], stage4_files_fetched=S4_RUNS["files"], stage4_issue_days=S4["data"]["issue_days"], imerg_cells_requested=SAT["n_imerg_cells_requested"],
                                  power_files_cached=POWER_RAW["files"], aorc_cells_per_basin=AORC_NCELLS, imerg_cells_per_basin=IMERG_NCELLS, gefs_boxes_per_basin=GEFS_NBOXES),
    camels_files=dict(zip_entries=_n_zip_entries, forcing_files=N_FORCING_FILES, flow_files=N_FLOW_FILES, columns_of_a_forcing_file=HDR["daymet"]["columns"], rows_of_a_forcing_file=HDR["daymet"]["rows"], first=HDR["daymet"]["first"], last=HDR["daymet"]["last"],
                      maurer_last=HDR["maurer"]["last"], attribute_tables=[f.name for f in ATTR_FILES], attribute_table_columns={n: len(c) for n, c in TABLE_COLS.items()}),
    bytes_read=dict(per_product=BYTES, total=BYTES_TOTAL, recorded_transfers=BYTES_TRANSFERS, kinds={d["key"]: d["obtained"]["bytes_read_kind"] for d in DATASETS}),
    caches=CACHE, data_folders_bytes_on_disk=CACHE_DATA_BYTES,
    download_dates=dict(by_product=DOWNLOAD_DATES, windows=DL_WINDOWS, days=DL_DAYS, clock=f"laptop clock, UTC{UTC_OFFSET[:3]}", how="file modification times of the downloads and the times written in the results records (results/aorc_data.json, stage4_realtime.json, data/gefs/c00/*.json, satellite_data.json, day_windows.json)"),
    clocks=dict(day_ends=dict(daymet=DE["daymet"]["day_ends_local"], nldas=DE["nldas"]["day_ends_local"], maurer=DE["maurer"]["day_ends_local"], imerg_documented="19:00", river_flow="local midnight"),
                river_flow_day={k: v for k, v in FLOWDAY.items()}, aorc_stamp_check_correlation_by_lag=_lag),
    other_files=dict(basemap="data/basemap/us-states.json: US state outlines, used only to draw the overview map; its source was not recorded in the project (unverified)", files_of_the_gefs_cache=CACHE["data/gefs"]["files"]),
)


# ---- 13. the words and numbers the drawings print (built from the numbers above; the drawings read them back from the JSON file) ----------------------------
S = {st["id"]: st for st in STAGES}
S["caches"]["counts"]["five_caches_mb"] = sum(S["caches"]["counts"][k] for k in ("derived_mb", "hourly_mb", "satellite_mb", "aorc_mb", "gefs_mb"))
_by_key = {d["key"]: d for d in DATASETS}
_part = {p["key"]: p for p in CAMELS_PARTS}
_fl = lambda key: (_by_key.get(key) or _part[key])["figure_line"]
yr = lambda part: f"{F.SPLIT[part][0][:4]}–{F.SPLIT[part][1][:4]}"
_n_ctrl = lambda e: len([c for c in e["extra_columns"] if c["column"].endswith("_older") or c["column"].endswith(("_to_09", "_to09", "_utc_day")) or c["column"].startswith(("fk", "fx"))])
EXP = {e["id"]: e for e in EXPERIMENTS}
_c = S["audit"]["counts"]
STAGE_FIG = {
    "download": dict(count=f"{S['download']['counts']['products']} products · {S['download']['counts']['gb_read']:.0f} GB read", detail=f"{S['download']['counts']['gb_kept_under_data']:.1f} GB kept under data/; no payment or account recorded", foot="fetch_*.py · check_*.py · 07_basin_maps.py",
                     talk=[f"{S['download']['counts']['products']} products", f"{S['download']['counts']['gb_read']:.0f} GB read"]),
    "audit": dict(count=f"{_c['all_camels']} → {_c['candidates']} → {_c['complete']} → {_c['chosen']}", detail="all CAMELS basins → candidates → complete record → chosen by rule (4 per landscape)", foot="01_data_audit.py", talk=[f"{_c['all_camels']} → {_c['candidates']} → {_c['complete']} → {_c['chosen']}", "all → candidates → complete → chosen"]),
    "caches": dict(count=f"{S['caches']['counts']['five_caches_mb']} MB cached", detail=f"derived {S['caches']['counts']['derived_mb']} MB · hourly {S['caches']['counts']['hourly_mb']} MB · satellite {S['caches']['counts']['satellite_mb']} MB · aorc {S['caches']['counts']['aorc_mb']} MB · gefs {S['caches']['counts']['gefs_mb']} MB",
                   foot="data/derived · hourly · satellite · aorc · gefs", talk=[f"{S['caches']['counts']['five_caches_mb']} MB cached"]),
    "inputs": dict(count=f"{S['inputs']['counts']['basins']} basins × {S['inputs']['counts']['days_per_basin']:,} days", detail=f"{S['inputs']['counts']['weather_columns']} weather columns + flow + {S['inputs']['counts']['properties']} basin properties; {S['inputs']['counts']['missing_values']} values missing", foot="camels.py · flood.load_data → data/derived/*.pkl",
                   talk=[f"{S['inputs']['counts']['basins']} basins × {S['inputs']['counts']['days_per_basin']:,} days", f"{S['training']['counts']['inputs']} inputs a day"]),
    "split": dict(count=f"{PART_DAYS['train']:,} | {PART_DAYS['val']:,} | {PART_DAYS['test']:,} days", detail=f"train {yr('train')} | validation {yr('val')} | test {yr('test')}, by the target day", foot="flood.SPLIT", talk=[f"train {PART_DAYS['train']:,} days", f"validation {PART_DAYS['val']:,}", f"test {PART_DAYS['test']:,}"]),
    "scaling": dict(count="training years only", detail=f"x → (x − mean) / sd; high-flow level = {S['scaling']['counts']['high_flow_percentile']}th percentile of training-year flow ({S['scaling']['counts']['level_min_mm_day']}–{S['scaling']['counts']['level_max_mm_day']} mm/day)", foot="flood.fit · flood.thresholds",
                    talk=["training years only", f"high-flow level = {S['scaling']['counts']['high_flow_percentile']}th percentile"]),
    "training": dict(count=f"LSTM · {S['training']['counts']['weights']:,} weights · {S['training']['counts']['seeds']} seeds", detail=f"{S['training']['counts']['sequences_per_update']} sequences per update; validation NSE checked every {S['training']['counts']['check_every_updates']} updates; stops after {S['training']['counts']['patience_checks']} checks without a gain; the best state is kept", foot="flood.fit",
                     talk=[f"LSTM · {S['training']['counts']['weights']:,} weights", "early stopping"]),
    "forecasts": dict(count=f"{S['forecasts']['counts']['prediction_sets']} sets · {S['forecasts']['counts']['saved_networks']} networks", detail="forecasts by target day in results/preds; networks with their scalers in results/models", foot="flood.save_preds", talk=[f"{S['forecasts']['counts']['prediction_sets']} sets", f"{S['forecasts']['counts']['saved_networks']} networks"]),
    "scores": dict(count=f"{S['scores']['counts']['onsets']} onsets · {S['scores']['counts']['test_days']:,} test days", detail="NSE · onsets warned · false warnings · GO / CAUTION / NOT\u00a0YET", foot="flood.evaluate · 06_flood_analysis.py", talk=[f"test years: {S['scores']['counts']['onsets']} onsets", "GO / CAUTION / NOT\u00a0YET"]),
    "documents": dict(count=f"{S['documents']['counts']['builders']} builders", detail="report, slides, Q&A, guide, overview: every number is read from results/*.json", foot="build_*.py", talk=["built from results/*.json"]),
}
SIDE_FIG = [dict(name="NLDAS rain, evening", line="prcp_nldas + control · 10_rain_timing.py", talk="NLDAS rain, evening"),
            dict(name="ERA5 rain to midnight (E1)", line=f"rain_to_midnight + {_n_ctrl(EXP['E1']) - 1} controls · 15_rain_to_midnight.py", talk="ERA5 rain to midnight"),
            dict(name="AORC rain to midnight (E1b)", line=f"aorc_to_midnight + {_n_ctrl(EXP['E1b'])} controls · 16_aorc_rain_to_midnight.py", talk="AORC rain to midnight"),
            dict(name=f"GEFS forecast, days t+1..t+{len(F.LEADS)} (E2)", line="fs1..fs3 + shuffled control · 18_forecast_input.py", talk=f"GEFS forecast, {len(F.LEADS)} days"),
            dict(name=f"{A['n_complete']} training rivers, not {len(IDS)} (E3)", line="more rivers, no new column · 17_more_basins.py", talk=f"{A['n_complete']} rivers, not {len(IDS)}"),
            dict(name="IMERG, soil wetness, vapour", line=f"tested on {SPLIT_SAT['train'][0][:4]}–{SPLIT_SAT['test'][1][:4]} · 08_satellite_inputs.py", talk="IMERG, soil wetness")]
CHECK_FIG = [dict(name="NCEP Stage IV", line=f"real-time check of the {N_ONSETS} onsets", talk="Stage IV"), dict(name="USGS 15-min · ERA5 · MERRA-2 hourly", line="which 24 hours a daily value covers", talk="clock checks"),
             dict(name="HURDAT2", line="storm type of each event", talk="HURDAT2"), dict(name="Elevation tiles · Annual NLCD", line="basin maps; check of CAMELS", talk="maps")]
SRC_GROUPS = [dict(title="Read by every model", role="every model", boxes=[_fl("camels_flow"), _fl("camels_daymet"), _fl("camels_attributes")]),
              dict(title="Extra inputs of the experiments", role="extra input", boxes=[_fl("camels_nldas"), _fl("era5"), _fl("aorc"), _fl("gefs"), _fl("imerg"), _fl("merra2")]),
              dict(title="Checks and maps, not model inputs", role="check or map", boxes=[_fl("stage4"), _fl("usgs_iv"), _fl("hurdat2"), _fl("elevation_tiles"), _fl("nlcd"), _fl("camels_maurer")])]
# the talk: CAMELS-US is one wide box; the other products one box each, in the order of the roles
TALK_TITLE = {"download": "Raw downloads", "audit": "Audit", "caches": "Caches", "inputs": "Daily inputs", "split": "Split", "scaling": "Scaling, level", "training": "Training", "forecasts": "Forecasts stored", "scores": "Scores", "documents": "Documents"}
_y = lambda k: (DW["sources"]["era5"]["period"][0][:4] + "–" + DW["sources"]["era5"]["period"][1][:4]) if k == "era5" else None
_ya = f"{AO['span'][:4]}–{AO['span'].split(' to ')[1][:4]}"
_yg = f"{GEFS_DATES[0][:4]}–{GEFS_DATES[1][:4]}"
_yi = f"{SAT['coverage']['imerg']['first_valid_day'][:4]}–{SAT['coverage']['imerg']['last_day'][:4]}"
_ym = f"{SAT['coverage']['merra_prcp']['first_valid_day'][:4]}–{SAT['coverage']['merra_prcp']['last_day'][:4]}"
TALK_SRC = [dict(name="CAMELS-US", lines=[f"Daymet weather, USGS flow, {len(F.STATIC)} properties", f"daily · {F.PERIOD[0][:4]}–{F.PERIOD[1][:4]}"], role="every model", wide=True),
            dict(name="ERA5", lines=["0.25° · hourly", _y("era5")], role="extra input"), dict(name="AORC", lines=["1 km · hourly", _ya], role="extra input"),
            dict(name="GEFSv12", lines=["0.25° · 3-hourly", _yg], role="extra input"), dict(name="IMERG", lines=["0.1° · daily", _yi], role="extra input"),
            dict(name="MERRA-2", lines=["0.5×0.625° · daily", _ym], role="extra input"), dict(name="Stage IV", lines=["4.76 km · hourly", yr("test")], role="check or map"),
            dict(name="USGS 15-min", lines=["15 min", "Jan–Jun 2012"], role="check or map"), dict(name="HURDAT2", lines=["6-hourly tracks", f"1851–{HURDAT_SUFFIX[0]}"], role="check or map"),
            dict(name="Elevation tiles", lines=["30 m · static", "SRTM format"], role="check or map"), dict(name="Land cover", lines=["30 m · yearly", f"{_lc['year']} map"], role="check or map")]
FIGURE = dict(
    caption_suggested=(f"Where the data come from and how they flow into a forecast. Top: the {len(DATASETS)} data products (the CAMELS-US parts shown separately), grouped by what they are used for. Below: the stages the data pass through, with the counts read "
                       f"from the files; at the right, the extra inputs of the experiments (they join at stage 4) and the products used only for checks and maps."),
    report=dict(width_in=7.2, title="Where the data come from and how they flow into a forecast", source_groups=SRC_GROUPS,
                stages=[dict(n=st["n"], title=st["title"], **{k: STAGE_FIG[st["id"]][k] for k in ("count", "detail", "foot")}) for st in STAGES],
                side=dict(title="Extra inputs, joined at stage 4", boxes=SIDE_FIG), checks=dict(title="Checks and maps (not inputs)", boxes=CHECK_FIG)),
    talk=dict(width_in=12.0, source_band_title="Data products", sources=TALK_SRC, stages=[dict(n=st["n"], title=TALK_TITLE[st["id"]], count=STAGE_FIG[st["id"]]["talk"]) for st in STAGES],
              side=dict(title="Extra inputs of the experiments join the daily inputs (stage 4)", boxes=[dict(name=b["talk"]) for b in SIDE_FIG]),
              legend=[dict(role="every model", text="read by every model"), dict(role="extra input", text="extra input of some experiments"), dict(role="check or map", text="check or map, not a model input")]),
)


# ---- 14. the drawings: one layout per width, one renderer; the words and numbers come from the 'figure' key of the JSON file ---------------------------------
ROLE_COL = {"every model": BLUE, "extra input": AQUA, "check or map": MUTED}
LEAD = 1.30                                                    # line height = LEAD x font size


def mix(c1, c2, t):
    a, b = np.array(matplotlib.colors.to_rgb(c1)), np.array(matplotlib.colors.to_rgb(c2))
    return tuple((1 - t) * a + t * b)


TINT = {k: mix(c, SURFACE, 0.90) for k, c in ROLE_COL.items()}


class Measure:
    """Text widths in inches at the drawing's dpi, and greedy wrapping at spaces."""

    def __init__(self, dpi=220):
        self.dpi = dpi
        self.fig = plt.figure(figsize=(4, 2), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.r = self.fig.canvas.get_renderer()
        self.cache = {}

    def w(self, s, fs, weight="normal"):
        k = (s, fs, weight)
        if k not in self.cache:
            t = self.ax.text(0, 0, s, fontsize=fs, fontweight=weight)
            self.cache[k] = t.get_window_extent(self.r).width / self.dpi
            t.remove()
        return self.cache[k]

    def wrap(self, s, fs, weight, max_w):
        lines, cur = [], ""
        for word in s.split(" "):
            trial = f"{cur} {word}".strip()
            if not cur or self.w(trial, fs, weight) <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        too = [ln for ln in lines if self.w(ln, fs, weight) > max_w + 1e-6]
        assert not too, f"a word does not fit a box of {max_w:.2f} in at {fs} pt: {too}"
        return lines

    def close(self):
        plt.close(self.fig)


lh = lambda fs: LEAD * fs / 72


class Cmds:
    """Drawing commands in inches (y downward), so that the layout can be computed before the figure's height is known."""

    def __init__(self):
        self.items = []

    def rect(self, x, y, w, h, fc, ec, lw=0.8, r=0.04, z=1):
        self.items.append(("rect", dict(x=x, y=y, w=w, h=h, fc=fc, ec=ec, lw=lw, r=r, z=z)))

    def stripe(self, x, y, w, h, fc, z=2):
        self.items.append(("stripe", dict(x=x, y=y, w=w, h=h, fc=fc, z=z)))

    def text(self, x, y, s, fs, color=INK2, weight="normal", ha="left", bound=None, z=3):
        self.items.append(("text", dict(x=x, y=y, s=s, fs=fs, color=color, weight=weight, ha=ha, bound=bound, z=z)))

    def arrow(self, x0, y0, x1, y1, color=INK2, lw=1.0, ms=9, z=4):
        self.items.append(("arrow", dict(x0=x0, y0=y0, x1=x1, y1=y1, color=color, lw=lw, ms=ms, z=z)))

    def line(self, x0, y0, x1, y1, color=AXIS, lw=0.8, z=2):
        self.items.append(("line", dict(x0=x0, y0=y0, x1=x1, y1=y1, color=color, lw=lw, z=z)))

    def circle(self, x, y, d, fc, z=3):
        self.items.append(("circle", dict(x=x, y=y, d=d, fc=fc, z=z)))

    def ymax(self):
        ys = [0.0]
        for k, d in self.items:
            if k in ("rect", "stripe"):
                ys.append(d["y"] + d["h"])
            elif k == "text":
                ys.append(d["y"] + lh(d["fs"]))
            elif k in ("arrow", "line"):
                ys.append(max(d["y0"], d["y1"]))
            elif k == "circle":
                ys.append(d["y"] + d["d"])
        return max(ys)


def block(c, m, x, y, w, texts, pad, role=None, fill=SURFACE, ec=AXIS, stripe=None, lw=0.8):
    """A box holding wrapped text blocks [(string, font size, weight, colour)]; returns its height. Text is placed top-down inside the padding."""
    inner = w - 2 * pad
    lines = [(ln, fs, wt, col) for (s, fs, wt, col) in texts for ln in m.wrap(s, fs, wt, inner)]
    h = 2 * pad + sum(lh(fs) for _, fs, _, _ in lines)
    return lines, h


def draw_block(c, x, y, w, h, lines, pad, fill, ec, stripe=None, lw=0.8):
    c.rect(x, y, w, h, fill, ec, lw=lw)
    if stripe is not None:
        c.stripe(x, y, w, 0.035, stripe)
    yy = y + pad + (0.02 if stripe is not None else 0)
    for ln, fs, wt, col in lines:
        c.text(x + pad, yy, ln, fs, color=col, weight=wt, bound=(x, y, x + w, y + h))
        yy += lh(fs)


def layout_report(spec, m):
    W, PAD, G_IN, G_GRP = spec["width_in"], 0.10, 0.06, 0.13
    FS = dict(title=9.2, group=7.9, name=7.6, line=7.1, st_title=8.2, st_count=8.2, st_detail=7.1, foot=7.0, side_name=7.4, side_line=7.0, num=7.2)
    c = Cmds()
    y = PAD
    c.text(PAD, y, spec["title"], FS["title"], color=INK, weight="bold")
    y += lh(FS["title"]) + 0.10
    # -- sources: three groups of boxes across the top (1 + 2 + 2 columns)
    ncols = [1, 2, 2]
    colw = (W - 2 * PAD - 2 * G_IN - 2 * G_GRP) / sum(ncols)
    bp = 0.05
    xs, x = [], PAD
    for gi, g in enumerate(spec["source_groups"]):
        xs.append(x)
        x += ncols[gi] * colw + (ncols[gi] - 1) * G_IN + G_GRP
    rows = max(int(math.ceil(len(g["boxes"]) / ncols[i])) for i, g in enumerate(spec["source_groups"]))
    prepared = {}
    for gi, g in enumerate(spec["source_groups"]):
        for bi, b in enumerate(g["boxes"]):
            prepared[(gi, bi)] = block(c, m, 0, 0, colw, [(b[0], FS["name"], "bold", INK), (b[1], FS["line"], "normal", INK2)], bp)
    row_h = [max(prepared[(gi, bi)][1] for gi, g in enumerate(spec["source_groups"]) for bi in range(len(g["boxes"])) if bi // ncols[gi] == r) for r in range(rows)]
    head_h = lh(FS["group"]) + 0.06
    gcx = []
    for gi, g in enumerate(spec["source_groups"]):
        col = ROLE_COL[g["role"]]
        gw = ncols[gi] * colw + (ncols[gi] - 1) * G_IN
        gcx.append(xs[gi] + gw / 2)
        c.text(xs[gi], y, g["title"], FS["group"], color=mix(col, INK, 0.35), weight="bold", bound=(xs[gi], y, xs[gi] + gw, y + head_h))
        c.line(xs[gi], y + head_h - 0.02, xs[gi] + gw, y + head_h - 0.02, color=col, lw=1.6)
        for bi, b in enumerate(g["boxes"]):
            r, cc = divmod(bi, ncols[gi])
            bx, by = xs[gi] + cc * (colw + G_IN), y + head_h + 0.04 + sum(row_h[:r]) + r * G_IN
            lines, _ = prepared[(gi, bi)]
            draw_block(c, bx, by, colw, row_h[r], lines, bp, TINT[g["role"]], mix(col, SURFACE, 0.45), stripe=col)
    y_src_end = y + head_h + 0.04 + sum(row_h) + (rows - 1) * G_IN
    # -- the chain (left) and the two side groups (right)
    x0, Wc = PAD, 4.62
    x1 = x0 + Wc + 0.40
    W1 = W - PAD - x1
    y = y_src_end + 0.30
    stage_boxes = []
    sp = 0.06
    for st in spec["stages"]:
        inner = Wc - 2 * sp - 0.24
        title = st["title"]
        side_by_side = m.w(title, FS["st_title"], "bold") + m.w(st["count"], FS["st_count"], "bold") + 0.12 <= inner
        parts = []
        if side_by_side:
            head_lines = 1
        else:
            head_lines = 2
        det = m.wrap(st["detail"], FS["st_detail"], "normal", inner)
        foot = m.wrap(st["foot"], FS["foot"], "normal", inner)
        h = 2 * sp + head_lines * lh(FS["st_title"]) + sum(lh(FS["st_detail"]) for _ in det) + sum(lh(FS["foot"]) for _ in foot)
        stage_boxes.append(dict(st=st, h=h, det=det, foot=foot, side_by_side=side_by_side, inner=inner))
    GAP = 0.15
    ytop = y
    for i, sb in enumerate(stage_boxes):
        st, h = sb["st"], sb["h"]
        sb["y"] = y
        c.rect(x0, y, Wc, h, SURFACE, AXIS, lw=0.9)
        c.circle(x0 + sp + 0.01, y + sp + 0.005, 0.165, INK2)
        c.text(x0 + sp + 0.0925, y + sp + 0.012, str(st["n"]), FS["num"], color=SURFACE, weight="bold", ha="center")
        tx = x0 + sp + 0.24
        c.text(tx, y + sp, st["title"], FS["st_title"], color=INK, weight="bold", bound=(x0, y, x0 + Wc, y + h))
        if sb["side_by_side"]:
            c.text(x0 + Wc - sp, y + sp, st["count"], FS["st_count"], color=INK, weight="bold", ha="right", bound=(x0, y, x0 + Wc, y + h))
            yy = y + sp + lh(FS["st_title"])
        else:
            c.text(tx, y + sp + lh(FS["st_title"]), st["count"], FS["st_count"], color=INK, weight="bold", bound=(x0, y, x0 + Wc, y + h))
            yy = y + sp + 2 * lh(FS["st_title"])
        for ln in sb["det"]:
            c.text(tx, yy, ln, FS["st_detail"], color=INK2, bound=(x0, y, x0 + Wc, y + h))
            yy += lh(FS["st_detail"])
        for ln in sb["foot"]:
            c.text(tx, yy, ln, FS["foot"], color=MUTED, bound=(x0, y, x0 + Wc, y + h))
            yy += lh(FS["foot"])
        y += h
        if i < len(stage_boxes) - 1:
            c.arrow(x0 + Wc / 2, y + 0.01, x0 + Wc / 2, y + GAP - 0.01, color=INK2, lw=1.0, ms=7)
            y += GAP
    y_chain_end = y
    # every group of products flows into stage 1: ticks under the groups, one line, one arrow
    y_line = y_src_end + 0.11
    for gi, g in enumerate(spec["source_groups"]):
        c.line(gcx[gi], y_src_end, gcx[gi], y_line, color=ROLE_COL[g["role"]], lw=1.4)
    c.line(gcx[0], y_line, gcx[-1], y_line, color=AXIS, lw=1.4)
    c.arrow(x0 + Wc / 2, y_line, x0 + Wc / 2, ytop - 0.02, color=INK2, lw=1.4, ms=9)

    def side_group(title, boxes, role, y_center, target):
        col = ROLE_COL[role]
        bw = W1 - 2 * 0.06
        pre = [block(c, m, 0, 0, bw, [(b["name"], FS["side_name"], "bold", INK), (b["line"], FS["side_line"], "normal", INK2)], 0.045) for b in boxes]
        head = m.wrap(title, FS["group"], "bold", bw)
        hh = sum(lh(FS["group"]) for _ in head) + 0.05
        gh = 0.06 + hh + sum(p[1] for p in pre) + 0.045 * (len(pre) - 1) + 0.06
        gy = y_center - gh / 2
        c.rect(x1, gy, W1, gh, TINT[role], mix(col, SURFACE, 0.35), lw=0.9, r=0.05, z=0)
        yy = gy + 0.06
        for ln in head:
            c.text(x1 + 0.06, yy, ln, FS["group"], color=mix(col, INK, 0.35), weight="bold", bound=(x1, gy, x1 + W1, gy + gh))
            yy += lh(FS["group"])
        yy += 0.05
        for (lines, h), b in zip(pre, boxes):
            draw_block(c, x1 + 0.06, yy, bw, h, lines, 0.045, SURFACE, mix(col, SURFACE, 0.45))
            yy += h + 0.045
        sb = stage_boxes[target]
        c.arrow(x1 - 0.01, y_center, x0 + Wc + 0.02, sb["y"] + sb["h"] / 2, color=col, lw=1.4, ms=9)
        return gy, gy + gh

    yc4 = stage_boxes[3]["y"] + stage_boxes[3]["h"] / 2
    g1 = side_group(spec["side"]["title"], spec["side"]["boxes"], "extra input", yc4, 3)
    yc9 = stage_boxes[8]["y"] + stage_boxes[8]["h"] / 2
    g2 = side_group(spec["checks"]["title"], spec["checks"]["boxes"], "check or map", yc9, 8)
    assert g1[1] < g2[0] - 0.05, "the two side groups overlap"
    return c, W, max(c.ymax(), g2[1], y_chain_end) + PAD


def layout_talk(spec, m):
    W, PAD = spec["width_in"], 0.15
    FS = dict(legend=11.0, src_name=11.8, src_line=10.8, st_title=13.0, st_count=12.0, side_title=11.8, side=11.0)
    c = Cmds()
    y = PAD
    # legend
    x = PAD
    c.text(x, y, spec["source_band_title"] + ":", FS["legend"], color=INK, weight="bold")
    x += m.w(spec["source_band_title"] + ":", FS["legend"], "bold") + 0.2
    for lg in spec["legend"]:
        col = ROLE_COL[lg["role"]]
        c.rect(x, y + 0.03, 0.14, 0.14, col, col, lw=0.5, r=0.02)
        c.text(x + 0.2, y, lg["text"], FS["legend"], color=INK2)
        x += 0.2 + m.w(lg["text"], FS["legend"]) + 0.35
    y += lh(FS["legend"]) + 0.10
    # source boxes: 12 slots, the wide box takes two
    n_slots = sum(2 if s.get("wide") else 1 for s in spec["sources"])
    gap = 0.07
    slot = (W - 2 * PAD - (len(spec["sources"]) - 1) * gap - (n_slots - len(spec["sources"])) * 0) / n_slots
    bp = 0.05
    pre, x, xs = [], PAD, []
    for s in spec["sources"]:
        w = slot * (2 if s.get("wide") else 1) + (gap * 0 if not s.get("wide") else 0)
        xs.append((x, w))
        pre.append(block(c, m, 0, 0, w, [(s["name"], FS["src_name"], "bold", INK)] + [(ln, FS["src_line"], "normal", INK2) for ln in s["lines"]], bp))
        x += w + gap
    bh = max(p[1] for p in pre) + 0.02
    for s, (bx, bw), (lines, _) in zip(spec["sources"], xs, pre):
        col = ROLE_COL[s["role"]]
        draw_block(c, bx, y, bw, bh, lines, bp, TINT[s["role"]], mix(col, SURFACE, 0.4), stripe=col)
    y_src_end = y + bh
    # chain: two rows of five, the second running right to left; the side band between them
    ncol, gx = 5, 0.46
    cw = (W - 2 * PAD - (ncol - 1) * gx) / ncol
    sp = 0.07
    boxes = []
    for st in spec["stages"]:
        t = m.wrap(f"{st['n']}  {st['title']}", FS["st_title"], "bold", cw - 2 * sp)
        k = [ln for part in st["count"] for ln in m.wrap(part, FS["st_count"], "normal", cw - 2 * sp)]
        boxes.append((t, k, 2 * sp + sum(lh(FS["st_title"]) for _ in t) + sum(lh(FS["st_count"]) for _ in k)))
    rhs = [max(b[2] for b in boxes[:ncol]), max(b[2] for b in boxes[ncol:])]            # the height of each row of boxes
    rh = rhs[0]
    y_r1 = y_src_end + 0.34
    # side band between the rows, under columns 1..4
    band_w = 4 * cw + 3 * gx
    chips = spec["side"]["boxes"]
    chip_w = (band_w - 2 * 0.08 - (len(chips) - 1) * 0.07) / len(chips)
    pre_c = [m.wrap(ch["name"], FS["side"], "normal", chip_w - 0.10) for ch in chips]
    chip_h = 2 * 0.045 + max(len(p) for p in pre_c) * lh(FS["side"])
    head = m.wrap(spec["side"]["title"], FS["side_title"], "bold", band_w - 0.16)
    band_h = 0.06 + len(head) * lh(FS["side_title"]) + 0.05 + chip_h + 0.07
    y_band = y_r1 + rhs[0] + 0.30
    y_r2 = y_band + band_h + 0.30
    pos = {}
    for i, st in enumerate(spec["stages"]):
        row, k = divmod(i, ncol)
        col = k if row == 0 else ncol - 1 - k
        pos[i] = (PAD + col * (cw + gx), y_r1 if row == 0 else y_r2)
    rhi = lambda i: rhs[0] if i < ncol else rhs[1]
    for i, (st, (t, k, h)) in enumerate(zip(spec["stages"], boxes)):
        bx, by = pos[i]
        c.rect(bx, by, cw, rhi(i), SURFACE, AXIS, lw=1.1)
        yy = by + sp
        for ln in t:
            c.text(bx + sp, yy, ln, FS["st_title"], color=INK, weight="bold", bound=(bx, by, bx + cw, by + rhi(i)))
            yy += lh(FS["st_title"])
        for ln in k:
            c.text(bx + sp, yy, ln, FS["st_count"], color=INK2, bound=(bx, by, bx + cw, by + rhi(i)))
            yy += lh(FS["st_count"])
    for i in range(len(spec["stages"]) - 1):
        (ax_, ay_), (bx_, by_) = pos[i], pos[i + 1]
        if abs(ay_ - by_) < 1e-9:
            if bx_ > ax_:
                c.arrow(ax_ + cw + 0.03, ay_ + rhi(i) / 2, bx_ - 0.03, by_ + rhi(i) / 2, lw=1.6, ms=12)
            else:
                c.arrow(ax_ - 0.03, ay_ + rhi(i) / 2, bx_ + cw + 0.03, by_ + rhi(i) / 2, lw=1.6, ms=12)
        else:
            c.arrow(ax_ + cw / 2, ay_ + rhi(i) + 0.03, bx_ + cw / 2, by_ - 0.03, lw=1.6, ms=12)
    y_line = y_src_end + 0.13
    for s_, (bx_, bw_) in zip(spec["sources"], xs):
        c.line(bx_ + bw_ / 2, y_src_end, bx_ + bw_ / 2, y_line, color=ROLE_COL[s_["role"]], lw=1.6)
    c.line(xs[0][0] + xs[0][1] / 2, y_line, xs[-1][0] + xs[-1][1] / 2, y_line, color=AXIS, lw=1.6)
    c.arrow(pos[0][0] + cw / 2, y_line, pos[0][0] + cw / 2, y_r1 - 0.03, lw=2.0, ms=13)
    # the band
    bx = PAD
    col = ROLE_COL["extra input"]
    c.rect(bx, y_band, band_w, band_h, TINT["extra input"], mix(col, SURFACE, 0.35), lw=1.0, r=0.05, z=0)
    yy = y_band + 0.06
    for ln in head:
        c.text(bx + 0.08, yy, ln, FS["side_title"], color=mix(col, INK, 0.35), weight="bold", bound=(bx, y_band, bx + band_w, y_band + band_h))
        yy += lh(FS["side_title"])
    yy += 0.05
    for j, (ch, lines) in enumerate(zip(chips, pre_c)):
        cx = bx + 0.08 + j * (chip_w + 0.07)
        c.rect(cx, yy, chip_w, chip_h, SURFACE, mix(col, SURFACE, 0.45), lw=0.8)
        ty = yy + 0.045
        for ln in lines:
            c.text(cx + 0.05, ty, ln, FS["side"], color=INK, bound=(cx, yy, cx + chip_w, yy + chip_h))
            ty += lh(FS["side"])
    s4x, s4y = pos[3]
    c.arrow(s4x + cw / 2, y_band - 0.03, s4x + cw / 2, s4y + rhs[0] + 0.03, color=col, lw=2.0, ms=13)
    return c, W, y_r2 + rhs[1] + PAD


def render(cmds, W, H, path, dpi=220):
    fig = plt.figure(figsize=(W, H), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    texts = []
    for kind, d in sorted(cmds.items, key=lambda it: it[1]["z"]):
        if kind == "rect":
            ax.add_patch(FancyBboxPatch((d["x"], d["y"]), d["w"], d["h"], boxstyle=f"round,pad=0,rounding_size={d['r']}", fc=d["fc"], ec=d["ec"], lw=d["lw"], zorder=d["z"]))
        elif kind == "stripe":
            ax.add_patch(Rectangle((d["x"] + 0.01, d["y"]), d["w"] - 0.02, d["h"], fc=d["fc"], ec="none", zorder=d["z"]))
        elif kind == "text":
            t = ax.text(d["x"], d["y"], d["s"], fontsize=d["fs"], color=d["color"], fontweight=d["weight"], ha=d["ha"], va="top", zorder=d["z"], clip_on=False)
            texts.append((t, d))
        elif kind == "arrow":
            ax.add_patch(FancyArrowPatch((d["x0"], d["y0"]), (d["x1"], d["y1"]), arrowstyle="-|>", mutation_scale=d["ms"], lw=d["lw"], color=d["color"], zorder=d["z"], shrinkA=0, shrinkB=0))
        elif kind == "line":
            ax.plot([d["x0"], d["x1"]], [d["y0"], d["y1"]], color=d["color"], lw=d["lw"], zorder=d["z"], solid_capstyle="butt")
        elif kind == "circle":
            ax.add_patch(matplotlib.patches.Ellipse((d["x"] + d["d"] / 2, d["y"] + d["d"] / 2), d["d"], d["d"], fc=d["fc"], ec="none", zorder=d["z"]))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for t, d in texts:
        if d["bound"] is None:
            continue
        bb = t.get_window_extent(r)
        x0, x1 = bb.x0 / dpi, bb.x1 / dpi
        y0, y1 = H - bb.y1 / dpi, H - bb.y0 / dpi
        bx0, by0, bx1, by1 = d["bound"]
        if x0 < bx0 - 0.004 or x1 > bx1 + 0.004 or y0 < by0 - 0.004 or y1 > by1 + 0.004:
            bad.append((d["s"], round(x0 - bx0, 3), round(bx1 - x1, 3), round(y0 - by0, 3), round(by1 - y1, 3)))
    assert not bad, f"text outside its box (text, left/right/top/bottom room in inches): {bad}"
    tmp = Path(str(path) + ".tmp.png")
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    tmp.replace(path)
    smallest, largest = min(d["fs"] for k, d in cmds.items if k == "text"), max(d["fs"] for k, d in cmds.items if k == "text")
    from PIL import Image
    with Image.open(path) as im:
        px = im.size
    return dict(file=rel(path), width_in=W, height_in=round(H, 2), dpi=dpi, pixels=list(px), smallest_lettering_pt=smallest, largest_lettering_pt=largest, n_texts=len(texts))


# ---- 15. the record, the number audit, the drawings --------------------------------------------------------------------------------------------------------
CORRECTIONS = [
    dict(brief="ESA WorldCover land cover, derived from Sentinel-1 and Sentinel-2 imagery",
         files="Annual NLCD land cover, the 2011 layer (USGS / MRLC), made by USGS from Landsat images. The word WorldCover occurs in no file, script, URL or document of the project.",
         evidence=["code/07_basin_maps.py (LC_URL, LC_LAYER, LC_YEAR)", "results/basin_gis.json (sources.land_cover, notes)", "data/landcover/annual_nlcd_2011_<gauge_id>.tif (12 files)"],
         done="datasets.nlcd and the imagery note say Annual NLCD, map year 2011, from Landsat imagery; the report and the question sheet already say 'Annual NLCD' (and not 'NLCD 2011')"),
    dict(brief="SRTM", files="a composite 1-arc-second elevation tile set in SRTM format (Mapzen/Tilezen terrain tiles) whose cells do not name their source; the project's own rule (Presentation_Plan.md, Professor_QA.md) is to call it a composite elevation tile set, not 'SRTM'",
         evidence=["results/basin_gis.json (sources.elevation, notes)"], done="datasets.elevation_tiles says so; the SRTM mission facts (February 2000, radar) are given as the facts of its main source"),
    dict(brief="Daymet v? basin means", files="the Daymet version is not stated in any CAMELS file or in the two CAMELS papers; recorded as unverified. The Daymet table as shipped also holds snow-water equivalent (swe), which no model reads.",
         evidence=["data/camels_us/basin_dataset_public_v1p2/readme_basin_mean_forcing.txt", "results/sources_checked.json"], done="datasets.camels_us.parts[camels_daymet].version and the unverified list"),
    dict(brief="GPM IMERG (POWER, Final run)", files="POWER documents the Final Run from 2001-01-01 but serves values from 1998-01-01 (which run the earlier days are is not stated); the IMERG release number is not stated",
         evidence=["results/satellite_data.json (documentation, imerg_run)"], done="datasets.imerg.version and the unverified list"),
    dict(brief="CAMELS-US, 671 basins", files=f"the unzipped time-series folders hold {N_FORCING_FILES['daymet']} Daymet, {N_FORCING_FILES['nldas']} NLDAS, {N_FORCING_FILES['maurer']} Maurer and {N_FLOW_FILES} flow files: a few gauges beyond the 671 of the attribute tables, never used",
         evidence=["data/camels_us/basin_dataset_public_v1p2 (counted by this script)"], done="all_information.basins.extra_files_in_the_unzipped_folders"),
]
UNVERIFIED = []
for _d in DATASETS:
    for _u in _d["unverified"]:
        UNVERIFIED.append(dict(dataset=_d["key"], status="unverified", fact=_u))
    for _p in _d.get("parts", []):
        for _u in _p["unverified"]:
            UNVERIFIED.append(dict(dataset=f"{_d['key']}.{_p['key']}", status="unverified", fact=_u))
UNVERIFIED.append(dict(dataset="basemap", status="unverified", fact=ALL["other_files"]["basemap"]))


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, pd.Timestamp)):
        return str(o)
    raise TypeError(f"cannot write {type(o)}")


def figure_numbers(o):
    """The numbers written in the strings of the figure spec: (token, value, decimals)."""
    out = []
    if isinstance(o, dict):
        for v in o.values():
            out += figure_numbers(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            out += figure_numbers(v)
    elif isinstance(o, str):
        for tok in re.findall(r"(?<![\w.])\d[\d,]*\.?\d*", o):
            txt = tok.replace(",", "")
            if txt.endswith("."):
                txt = txt[:-1]
            if txt.replace(".", "", 1).isdigit():
                out.append((tok, float(txt), len(txt.split(".")[1]) if "." in txt else 0))
    return out


def number_audit(figure, rest):
    pool = np.array(sorted(set(flat_numbers(rest, skip_keys=("figure",)))))
    toks = figure_numbers({k: v for k, v in figure.items() if k not in ("renderings", "number_audit")})
    missing = sorted({t for t, v, d in toks if not (np.abs(pool - v) <= 0.5 * 10 ** (-d) + 1e-9).any()})
    return dict(how="every number written in the strings of this figure spec is looked up among the numbers of the rest of the file (numeric values and numbers inside its sentences), to the precision it is printed with",
                numbers_checked=len(toks), distinct_checked=len({t for t, _, _ in toks}), not_found=missing)


RECORD = dict(
    what=("The data of the project and how they flow into a forecast: the single source of the data text of the report and the slides (what each product is, how it was obtained, the flow of the data, how the dataset is computed, "
          "all the counts and sizes, and the imagery facts). Written by code/19_data_flow.py."),
    written=time.strftime(STAMP_FMT), laptop_clock=f"UTC{UTC_OFFSET[:3]}", quick_mode=F.QUICK, code=dict(file="code/19_data_flow.py", sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    how_to_read=("datasets: one entry per data product (CAMELS-US has parts); each says what it is, who made it, the version, grid, time step, period used, time stamp, real-time latency, how it was obtained (address, login, bytes, date, script), "
                 "how it became a model input, where it is cached, its role, and the imagery facts. flow: the stages in order, with counts. how_computed: formulas in words and symbols. all_information: counts, sizes, dates, scalers, thresholds. "
                 "imagery_note: the plain statement about satellite images. unverified: what could not be sourced. figure: what the two drawings print. Numbers come from the project's files, named in inputs_read; "
                 "documents read on the day of writing are in documentation_read."),
    summary=dict(datasets=[dict(key=d["key"], name=d["name"], role=d["role_code"], one_line=d["one_line"]) for d in DATASETS],
                 flow=[dict(n=st["n"], title=st["title"], counts=st["counts"]) for st in STAGES], imagery_note=IMAGERY_NOTE),
    corrections_to_the_brief=CORRECTIONS, imagery_note=IMAGERY_NOTE, datasets=DATASETS, flow=FLOW, how_computed=HOW, all_information=ALL, unverified=UNVERIFIED,
    documentation_read=[dict(key=k, **v) for k, v in DOCS.items()], inputs_read=dict(sorted(READ.items())), figure=FIGURE, runtime_s=None)
say("writing the record ...")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
REC = json.loads(json.dumps(RECORD, default=jdefault, ensure_ascii=False))
AUDIT = number_audit(REC["figure"], REC)
say(f"number audit: {AUDIT['numbers_checked']} numbers in the figure text, {len(AUDIT['not_found'])} not found: {AUDIT['not_found']}")
assert not AUDIT["not_found"], "the figure prints numbers that are not in the record"
M = Measure()
c_rep, w_rep, h_rep = layout_report(REC["figure"]["report"], M)
c_talk, w_talk, h_talk = layout_talk(REC["figure"]["talk"], M)
M.close()
REND = {"report": render(c_rep, w_rep, h_rep, OUT_FIG), "talk": render(c_talk, w_talk, h_talk, OUT_TALK)}
REND["report"]["use"] = f"the report: the width of its A4 text column (7.2 in), lettering {REND['report']['smallest_lettering_pt']:g} to {REND['report']['largest_lettering_pt']:g} pt, with its own title; one page of its own"
REND["talk"]["use"] = f"the slides: {REND['talk']['width_in']:g} in wide, lettering {REND['talk']['smallest_lettering_pt']:g} to {REND['talk']['largest_lettering_pt']:g} pt, no title (the slide has one), the stages in two rows (the second runs right to left)"
REC["figure"]["renderings"] = REND
REC["figure"]["number_audit"] = AUDIT
REC["runtime_s"] = round(time.time() - T0, 1)
tmp = OUT_JSON.with_suffix(".json.tmp")
tmp.write_text(json.dumps(REC, indent=1, ensure_ascii=False))
tmp.replace(OUT_JSON)

# ---- 16. what to read in the log ---------------------------------------------------------------------------------------------------------------------------
say(f"wrote {rel(OUT_JSON)} ({OUT_JSON.stat().st_size / 1e3:.0f} KB), {rel(OUT_FIG)} and {rel(OUT_TALK)} in {REC['runtime_s']} s")
for k, r in REND.items():
    say(f"  {k}: {r['pixels'][0]} x {r['pixels'][1]} px, {r['width_in']} x {r['height_in']} in, smallest lettering {r['smallest_lettering_pt']} pt")
say("datasets:")
for d in DATASETS:
    say(f"  {d['key']:16s} {d['one_line']}")
say("flow:")
for st in STAGES:
    say(f"  {st['n']:2d} {st['title']:34s} {json.dumps(st['counts'])}")
say("imagery note:", IMAGERY_NOTE)
say(f"unverified: {len(UNVERIFIED)} facts")
for u in UNVERIFIED:
    say(f"  [{u['dataset']}] {u['fact']}")
