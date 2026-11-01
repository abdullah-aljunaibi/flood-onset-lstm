#!/bin/bash
# The reviewer's fast checks of experiment E2 (GEFSv12 reforecast rain as an LSTM input), one after the other: no network, nothing trained, one thread, nothing written into the project.
# Run from anywhere:   code/review/forecast_input/run_fast_checks.sh        The output files go to $REVIEW_OUT (default: <temp folder>/e2_review), the log to $REVIEW_OUT/fast_run.log.
# About 6 minutes (see README.md for the time of each check). Optional: E1_ZIP_DIR = the code folder of the hand-in zip of 4 October (lets f1_integrity.py and f14_test_flood.py say how flood.py and test_flood.py
# differ from it). The order matters: f4 writes f4_series.npz (read by f5, f6, f18), f5 writes the bias factors f_tables.py reads, f6 writes the table pickles (read by f7, f11, f12, f12c).
cd "$(dirname "$0")/../../.." || exit 1                          # the project root
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/e2_review}
D=code/review/forecast_input
mkdir -p "$REVIEW_OUT"
exec > >(tee "$REVIEW_OUT/fast_run.log") 2>&1
run() {                                                          # run <script> [argument]: one check, its time, stop at the first failure
  local name=$1; shift; local t=$(date +%s)
  echo "################ $name $*"
  "$PY" "$D/$name.py" "$@" || { echo "FAILED: $name"; exit 1; }
  echo "---- $name $*: $(( $(date +%s) - t )) s"
}
T0=$(date +%s)
run f1_integrity; run f3_store; run f4_basins; run f5_skill; run f5b_subdaily; run f6_tables; run f7_networks; run f8_headline; run f9_readings; run f10_figure
run f11_simple ridge; run f12_interpretation; run f12b_trigger; run f12c_drizzle; run f13_outcome; run f14_test_flood; run f17_robustness; run f18_roadmap; run f21_readme_snippets
echo "---- all fast checks: $(( $(date +%s) - T0 )) s"
echo "all fast checks finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
