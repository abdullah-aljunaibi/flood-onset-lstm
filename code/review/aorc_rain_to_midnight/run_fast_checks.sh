#!/bin/bash
# The reviewer's fast checks of experiment E1b (AORC rain to midnight), one after the other: no network, nothing trained, one thread, nothing written into the project.
# Run from anywhere:   code/review/aorc_rain_to_midnight/run_fast_checks.sh        The output files go to $REVIEW_OUT (default: <temp folder>/aorc_review). About 3 minutes (see README.md for the times of each check).
# Optional: E1_ZIP_DIR = the code folder of the hand-in zip of 4 October (lets a1_integrity.py and a13_test_flood.py say how flood.py and test_flood.py differ from it).
cd "$(dirname "$0")/../../.." || exit 1                          # the project root
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/aorc_review}
D=code/review/aorc_rain_to_midnight
mkdir -p "$REVIEW_OUT"
run() {                                                          # run <script> [argument]: one check, its time, stop at the first failure
  local name=$1; shift; local t=$(date +%s)
  echo "################ $name $*"
  "$PY" "$D/$name.py" "$@" || { echo "FAILED: $name"; exit 1; }
  echo "---- $name $*: $(( $(date +%s) - t )) s"
}
T0=$(date +%s)
run a1_integrity; run a2_headline; run a3_clocks; run a4_networks; run a14_readings; run a9_byperiod; run a10_simple ridge; run a11_outcome; run a12_figure; run a13_test_flood
echo "all fast checks finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
