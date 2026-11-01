#!/usr/bin/env bash
# Runs the independent reviewer's checks of the GIS-layer export (code/27_gis_layers.py -> results/gis_layers/) in order. From the project root, with the system Python 3.13
# (geopandas, rasterio, shapely, pyproj, numpy, pandas, python-markdown). Nothing is written into the project unless g6_assemble.py is given --write-into-project (pass it as the
# first argument of this script): then it APPENDS the block "round_2" to results/review_gis_layers.json (an earlier round_2 is replaced; round 1 is kept byte for byte and never
# rewritten; g6_round1_assemble.py is the frozen assembler of round 1 and refuses to run). Every other output goes to REVIEW_OUT (default <system temp>/gis_review).
# g7_rebuild.py copies about 190 MB of inputs into REVIEW_OUT/rebuild and runs the script there; REVIEW_SKIP_REBUILD=1 skips it.
# Run it BEFORE appending a new round: g1 compares results/gis_layers.json 'independent_review.sha256' with the record without its later rounds, which it rebuilds either way.
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1        # no __pycache__ in the project (set before Python starts)
D=code/review/gis_layers
for s in g1_files g2_vectors g3_elevation g4_landcover g5_readme g7_rebuild; do
  echo "== $s"; "$PY" -X utf8 -B "$D/$s.py"
done
echo "== g6_assemble (round 2)"; "$PY" -X utf8 -B "$D/g6_assemble.py" "$@"
