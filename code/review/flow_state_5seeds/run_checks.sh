#!/usr/bin/env bash
# Round 2 of the independent review of E4e (five-seed stored sets, 30 Oct 2026). From the project root. Nothing is trained; nothing is written into the project
# unless h9_round2_assemble.py is given --write-into-project (first argument of this script; it refuses if round_2 exists).
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS="${FLOOD_THREADS:-1}"
D=code/review/flow_state_5seeds
for s in h2_headline h7_round2; do echo "== $s"; "$PY" "$D/$s.py"; done
echo "== h9_round2_assemble"; "$PY" "$D/h9_round2_assemble.py" "$@"
