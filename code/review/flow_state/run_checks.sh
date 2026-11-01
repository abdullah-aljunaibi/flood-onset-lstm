#!/usr/bin/env bash
# Runs the independent reviewer's checks of experiment E4e in order (see README.md). From the project root. Nothing is trained; nothing is written into the project
# unless h6_assemble.py is given --write-into-project (pass it as the first argument of this script).
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS="${FLOOD_THREADS:-1}"      # no __pycache__ in the project (set before Python starts)
D=code/review/flow_state
for s in h1_integrity h3_columns h2_headline h4_networks h5_outcome; do
  echo "== $s"; "$PY" "$D/$s.py"
done
echo "== h6_assemble"; "$PY" "$D/h6_assemble.py" "$@"
