#!/bin/bash
# The reviewer's slower checks of experiment E1b: a few small requests to the AORC bucket (about 4 MB), the Stage IV hindsight regressions and network tests (they read $REVIEW_OUT/cache/stage4_means.csv, made by fetch_stage4.sh:
# about 1.3 GB from the Iowa Mesonet archive, 8 to 9 minutes, run it once first), and the simple-model refits (ridge in the fast checks; here the perturbation test and the forest). One thread, nothing trained, nothing written into the project.
# run_fast_checks.sh must have run once with the same REVIEW_OUT (a3_clocks.py writes my_series_aorc.npz, which these checks read). About 12 minutes on the project laptop with other jobs running.
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/aorc_review}
D=code/review/aorc_rain_to_midnight
[ -f "$REVIEW_OUT/my_series_aorc.npz" ] || { echo "run run_fast_checks.sh first (it writes my_series_aorc.npz)"; exit 1; }
[ -f "${AORC_CACHE:-$REVIEW_OUT/cache}/stage4_means.csv" ] || { echo "no Stage IV cache: run fetch_stage4.sh first (about 1.3 GB, 8 to 9 minutes) or point AORC_CACHE at a folder that holds stage4_means.csv"; exit 1; }
run() { local name=$1; shift; local t=$(date +%s); echo "################ $name $*"; "$PY" "$D/$name.py" "$@" || { echo "FAILED: $name"; exit 1; }; echo "---- $name $*: $(( $(date +%s) - t )) s"; }
T0=$(date +%s)
[ -f "$REVIEW_OUT/a7a_days.json" ] || run a7a_select_days                  # the 28 days (deterministic; fetch_stage4.sh makes the same file)
run a5_cache; run a7_hindsight; run a5b_stamp; run a8_hindsight_networks; run a10_simple perturb; run a10_simple forest
echo "all slow checks finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
