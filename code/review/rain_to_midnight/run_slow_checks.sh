#!/bin/bash
# The reviewer's slower checks (about 15 minutes in all; one thread; nothing is trained; nothing is written into the project): the perturbation test of the simple-model fits, the verdict records, the forest refits
# (about 3 to 5 minutes each) and the sandbox runs of --rescore and --extend of 15_rain_to_midnight.py (they copy about 0.3 GB into the temp folder and remove it again).
# Needs r2c_clocks.py (it writes my_series_r2.npz into $REVIEW_OUT) to have been run once with the same REVIEW_OUT: run_fast_checks.sh does.
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/rain_to_midnight_review}
D=code/review/rain_to_midnight
mkdir -p "$REVIEW_OUT"
run() {
  local name=$1; shift; local t=$(date +%s)
  echo "################ $name $*"
  "$PY" "$D/$name.py" "$@" || { echo "FAILED: $name"; exit 1; }
  echo "---- $name $*: $(( $(date +%s) - t )) s"
}
T0=$(date +%s)
run r2d_simple perturb; run r2e3_verdicts; run r2d_simple forest; run r2d2_ridge_exact; run r2d3_forest_exact
t=$(date +%s); echo "################ r2k_sandbox_modes.sh"; bash "$D/r2k_sandbox_modes.sh" || { echo "FAILED: r2k_sandbox_modes.sh"; exit 1; }; echo "---- r2k_sandbox_modes.sh: $(( $(date +%s) - t )) s"
echo "all slow checks finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
