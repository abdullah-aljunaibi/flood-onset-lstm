#!/usr/bin/env bash
# Round 2 of the independent review of E4a (five-seed stored sets, 30 Oct 2026). From the project root. Nothing is trained; nothing is written into the project
# unless w9_round2_assemble.py is given --write-into-project (pass it as the first argument of this script; it refuses if round_2 exists).
set -e
PY="${PY:-python}"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS="${FLOOD_THREADS:-1}"
D=code/review/warning_policy_5seeds
for s in w1_headline w2_seeds w3_any_lead w4_bootstrap w5_stacker w6_text_go w7_round2; do echo "== $s"; "$PY" "$D/$s.py"; done
echo "== w9_round2_assemble"; "$PY" "$D/w9_round2_assemble.py" "$@"
