"""Shared setup of the independent reviewer's scripts for the GIS-layer export (code/27_gis_layers.py -> results/gis_layers/, results/gis_layers.json).

These scripts are an AI review: they recompute the export's checks with the reviewer's own code (shapely / pyproj / rasterio / numpy calls written here; nothing is imported
from code/27_gis_layers.py or code/07_basin_maps.py) and write nothing into the project except this folder and the review record results/review_gis_layers.json
(g6_assemble.py --write-into-project). Every other output goes to the folder OUT (environment variable REVIEW_OUT; default: <system temp folder>/gis_review).
Nothing in results/gis_layers/ is opened for writing; the GeoPackage is read from a scratch copy whose hash is first proved equal to the original, so SQLite cannot leave a
side file beside the export; GDAL_PAM_ENABLED=NO keeps rasterio from writing .aux.xml files. g1_files.py records a before/after manifest of the folder to prove it.
Run from the project root with the system Python 3.13 (geopandas, rasterio, shapely, pyproj, numpy, pandas):
    python -X utf8 -B code/review/gis_layers/<script>.py
Order: g1_files, g2_vectors, g3_elevation, g4_landcover, g5_readme, g6_assemble (run_checks.sh does it).
"""
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ["GDAL_PAM_ENABLED"] = "NO"                              # no .aux.xml beside an exported raster, whatever happens
sys.dont_write_bytecode = True                                     # no __pycache__ in the project
HERE = Path(__file__).resolve().parent                              # <project>/code/review/gis_layers
PROJ = HERE.parents[2]                                              # <project>
OUT = Path(os.environ.get("REVIEW_OUT") or (Path(tempfile.gettempdir()) / "gis_review"))
OUT.mkdir(parents=True, exist_ok=True)
import numpy as np
import pandas as pd

# ---- the conventions under review, restated from the brief (not imported from the code under review) ------------------------------------------------------------------
GIS_DIR = PROJ / "results" / "gis_layers"
RECORD = "results/gis_layers.json"
README = "results/gis_layers/README.md"
SCRIPT = "code/27_gis_layers.py"
FOREST = (41, 42, 43)                                               # NLCD deciduous, evergreen, mixed forest
NLCD_NODATA = 250
DEM_NODATA = -32768
ARCSEC = 1.0 / 3600.0
EPSG_OUT, EPSG_GEO, EPSG_WGS, EPSG_ALB = 4269, 4269, 4326, 5070
TOL = {"area_km2": 0.001, "elev_m": 0.01, "share": 1e-6}           # the tolerances the README claims against results/basin_gis.json
MOUNTAIN_THRESHOLD_M = 25.0                                         # the README's "more than 25 m from CAMELS"
BUFFER_ARCSEC = 15                                                  # the README's elevation buffer


def jload(rel):
    """A JSON file of the project, by its path from the project root."""
    with open(PROJ / rel, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def jsave(name, obj):
    """The output of a check: OUT/<name> (never inside the project)."""
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)
    return OUT / name


def jout(name):
    with open(OUT / name, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def basins():
    """The 12 gauge ids in the order of results/data_audit.json 'selected' (the order of every other results file), with name / landscape / the 5-figure position
    is NOT taken from there (it is rounded to 3 decimals) but from camels_topo.txt, see camels()."""
    sel = jload("results/data_audit.json")["selected"]
    return [s["gauge_id"] for s in sel], {s["gauge_id"]: s for s in sel}


def camels():
    """The CAMELS attribute rows of the 12 basins, parsed here (camels_topo.txt / camels_vege.txt, ';'-separated, gauge ids as text)."""
    ids, _ = basins()
    topo = pd.read_csv(PROJ / "data/camels_us/camels_topo.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    vege = pd.read_csv(PROJ / "data/camels_us/camels_vege.txt", sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    vege["dom_land_cover"] = vege["dom_land_cover"].astype(str).str.strip()
    t = topo.loc[ids]
    v = vege.loc[ids]
    return t, v


def scratch_copy(src, name=None):
    """A scratch copy of a file of the export, hash-checked equal to the original, so that opening it cannot create a side file in results/gis_layers/."""
    d = OUT / "scratch"
    d.mkdir(exist_ok=True)
    dst = d / (name or Path(src).name)
    shutil.copy2(src, dst)
    assert sha256_file(src) == sha256_file(dst), "scratch copy differs from the original"
    return dst


# ---- SRTM .hgt tiles (big-endian int16, 3601 x 3601, row 0 = north edge, column 0 = west edge; named by the south-west corner) --------------------------------------------
_TILES = {}


def srtm_tile(lat0, lon0):
    """The source tile whose south-west corner is (lat0, lon0) (integers; lon0 negative = west), read from data/dem/<name>.hgt.gz, or None if absent."""
    key = (lat0, lon0)
    if key in _TILES:
        return _TILES[key]
    name = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    p = PROJ / "data" / "dem" / f"{name}.hgt.gz"
    if not p.exists():
        _TILES[key] = None
        return None
    with gzip.open(p, "rb") as f:
        raw = f.read()
    a = np.frombuffer(raw, dtype=">i2")
    n = int(round(math.sqrt(a.size)))
    assert n * n == a.size and n == 3601, (name, a.size)
    _TILES[key] = a.reshape(n, n).astype(np.int32)
    return _TILES[key]


def tile_name(lat0, lon0):
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"



# ---- the review record itself (results/review_gis_layers.json): round 1 is kept byte for byte, later rounds are appended as round_2, round_3 ... -------------------------------------
REVIEW_RECORD = PROJ / "results" / "review_gis_layers.json"
CRLF, LF = "\r\n", "\n"


def review_record_round1():
    """(bytes, info) of the review record WITHOUT any round appended later. With no round appended it is the file as it is (info['reserialises_exactly'] says whether
    json.dumps(indent=1, ensure_ascii=False), with the file's own line ending, reproduces its bytes, which the append relies on). With round_2 appended it is the file
    re-serialised without the round keys; its sha256 must equal round_2.round_1_intact.file_sha256_before_this_append (checked by the caller)."""
    raw = REVIEW_RECORD.read_bytes()
    text = raw.decode("utf-8")
    nl = CRLF if CRLF in text else LF                                  # the record was written on Windows in text mode: CRLF
    obj = json.loads(text)
    rounds = [k for k in obj if re.fullmatch(r"round_\d+", k)]
    again = json.dumps({k: v for k, v in obj.items() if k not in rounds}, indent=1, ensure_ascii=False, default=_default).replace(LF, nl)
    info = {"rounds_appended": rounds, "line_ending": "CRLF" if nl == CRLF else "LF"}
    if not rounds:
        info["reserialises_exactly"] = again.encode("utf-8") == raw
        return raw, info
    info["reserialises_exactly"] = None
    return again.encode("utf-8"), info
