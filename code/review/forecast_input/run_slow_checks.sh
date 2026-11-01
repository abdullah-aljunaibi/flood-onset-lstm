#!/bin/bash
# The reviewer's slower checks of experiment E2: the three raw archive days (f2: NETWORK, 3 Range requests of 7.6-14.3 MB; needs ~/AI_Class/.venv-grib), the simple-model perturbation test and the forest refit (f11),
# the trigger-free comparison with its bootstrap (f12d), the sandbox --rescore of code/18_forecast_input.py (f15) and the downloader's own --check on a sandbox copy of the store (f16; needs ~/AI_Class/.venv-grib).
# One thread, nothing trained, nothing written into the project (the sandboxes live in $REVIEW_OUT and are removed). run_fast_checks.sh must have run once with the same REVIEW_OUT. About 14 minutes; log in $REVIEW_OUT/slow_run.log.
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
PYG=${PYG:-$HOME/AI_Class/.venv-grib/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/e2_review}
D=code/review/forecast_input
[ -f "$REVIEW_OUT/f6_tables_csv.pkl" ] || { echo "run run_fast_checks.sh first (it writes the table pickles and f4_series.npz)"; exit 1; }
exec > >(tee "$REVIEW_OUT/slow_run.log") 2>&1
run() { local py=$1; local name=$2; shift; shift; local t=$(date +%s); echo "################ $name $*"; "$py" "$D/$name.py" "$@" || { echo "FAILED: $name"; exit 1; }; echo "---- $name $*: $(( $(date +%s) - t )) s"; }
runsh() { local name=$1; shift; local t=$(date +%s); echo "################ $name $*"; PY="$PY" PYG="$PYG" bash "$D/$name.sh" "$@" || { echo "FAILED: $name"; exit 1; }; echo "---- $name $*: $(( $(date +%s) - t )) s"; }
T0=$(date +%s)
run "$PYG" f2_raw_days; run "$PY" f11_simple perturb; run "$PY" f11_simple forest; run "$PY" f12d_trigger_free; runsh f15_sandbox_rescore plain; runsh f16_fetch_check
echo "---- all slow checks: $(( $(date +%s) - T0 )) s"
echo "all slow checks finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
