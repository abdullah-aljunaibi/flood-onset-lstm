#!/bin/bash
# The reviewer's fast checks of experiment E1, one after the other (about 3 to 4 minutes in all on the project laptop; one thread; nothing is trained; nothing is written into the project).
# Run from anywhere:   code/review/rain_to_midnight/run_fast_checks.sh        The output files go to $REVIEW_OUT (default: <temp folder>/rain_to_midnight_review).
cd "$(dirname "$0")/../../.." || exit 1                          # the project root
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/rain_to_midnight_review}
D=code/review/rain_to_midnight
mkdir -p "$REVIEW_OUT"
run() {                                                          # run <script> [argument]: one check, its time, stop at the first failure
  local name=$1; shift; local t=$(date +%s)
  echo "################ $name $*"
  "$PY" "$D/$name.py" "$@" || { echo "FAILED: $name"; exit 1; }
  echo "---- $name $*: $(( $(date +%s) - t )) s"
}
T0=$(date +%s)
run r2a_integrity; run r2b_headline; run r2c_clocks; run r2c2_networks; run r2d_simple ridge; run r2e_readings; run r2e2_yardstick; run r2f_byperiod; run r2g_figure; run r2h_records
run r2i_daymet_window; run r2j_late_window_basins; run run_test_flood_redirected; run r1_stamp_convention; run r1_hour_sensitivity; run r1_clock_midnight; run r1_gain_rule
run r3a_integrity; run r3c_clauses; run r3d_texts                                          # round 3: the 51 saved files and the passes, the new clauses / yardstick fields / standard errors / verdicts, the quoted statements
echo "all fast checks finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
