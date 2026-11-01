#!/usr/bin/env bash
# Round 2 of the independent review of E2 (five seeds). From the project root. Nothing is trained; nothing is written into the project
# unless v5_assemble.py is given --write-into-project (pass it as the first argument of this script).
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS="${FLOOD_THREADS:-1}"
D=code/review/forecast_input_5seeds
for s in v1_score v2_compare v3_networks v4_readings_text; do
  echo "== $s"; "$PY" "$D/$s.py"
done
echo "== v5_assemble"; "$PY" "$D/v5_assemble.py" "$@"
