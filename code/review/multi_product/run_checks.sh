#!/usr/bin/env bash
# Runs the independent reviewer's checks of experiment E7 in order (see README.md). From the project root. Nothing is trained; nothing is written into the project
# unless m6_assemble.py is given --write-into-project (pass it as the first argument of this script).
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS="${FLOOD_THREADS:-1}"      # no __pycache__ in the project (set before Python starts)
D=code/review/multi_product
for s in m1_integrity m3_columns m2_headline m4_networks m7_ridge m5_outcome; do
  echo "== $s"; "$PY" -X utf8 "$D/$s.py"
done
echo "== m6_assemble"; "$PY" -X utf8 "$D/m6_assemble.py" "$@"
