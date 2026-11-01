#!/usr/bin/env bash
# Runs the independent reviewer's checks of experiment E4b in order (see README.md). From the project root. Nothing is trained; nothing is written into the project
# unless r6_assemble.py is given --write-into-project (pass it as the first argument of this script).
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS="${FLOOD_THREADS:-1}"      # no __pycache__ in the project (set before Python starts: r_common.py's flag comes too late for itself)
D=code/review/within_day_inputs
for s in r1_integrity r3_columns r2_headline r4_networks r5_outcome; do
  echo "== $s"; "$PY" "$D/$s.py"
done
echo "== r6_assemble"; "$PY" "$D/r6_assemble.py" "$@"
