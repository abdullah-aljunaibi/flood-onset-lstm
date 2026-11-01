#!/usr/bin/env bash
# Round 2 of the E1b review (five seeds). From the project root:  PY=<python> REVIEW_OUT=<scratch folder> AFTER_EXTEND=<folder with the 11:51 copy, optional> bash code/review/aorc_rain_to_midnight_5seeds/run_checks.sh
set -euo pipefail
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS=1
H=code/review/aorc_rain_to_midnight_5seeds
REVIEW_OUT="$REVIEW_OUT/round1_days" "$PY" code/review/aorc_rain_to_midnight/a7a_select_days.py > /dev/null   # round 1's 28 Stage IV days (needed by r6)
for s in r1_integrity r2_headline r4_stage4 r5_one_network r6_hindsight r7_outcome; do echo "== $s"; "$PY" $H/$s.py | tail -6; done
"$PY" $H/r8_assemble.py "$@"
