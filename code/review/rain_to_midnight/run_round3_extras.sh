#!/bin/bash
# The round-3 checks that need copies of EARLIER states, which are not part of the project (give them in the variables; each script checks the copy against the sha256 recorded in the review file):
#   E1_PREVIOUS_RUN_OUT   the output folder of an EARLIER run of run_fast_checks.sh, for r3b_compare_outputs.py (which outputs differ, leaf by leaf); it compares with the outputs of run_fast_checks.sh in $REVIEW_OUT, so run that first with the same REVIEW_OUT
#   E1_FIRST_RUN_JSON     a copy of the first-run results/rain_to_midnight.json (sha256 317c1d28...), for r3f_regression.py (is every number of the first run unchanged?)
#   E1_R2_HASHES          optional, for r3a_integrity.py: the list 'sha256  results/...' of prediction and network files made in round 2
# and r3_sandbox_rescore.sh (the script's --rescore in a sandbox copy: same JSON, same figure bytes?). One thread; nothing is trained; nothing is written into the project. About 3 to 4 minutes.
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/rain_to_midnight_review}
D=code/review/rain_to_midnight
mkdir -p "$REVIEW_OUT"
T0=$(date +%s)
if [ -n "$E1_PREVIOUS_RUN_OUT" ]; then echo "################ r3b_compare_outputs"; "$PY" "$D/r3b_compare_outputs.py" "$E1_PREVIOUS_RUN_OUT" || exit 1; else echo "(E1_PREVIOUS_RUN_OUT not given: r3b_compare_outputs.py skipped)"; fi
if [ -n "$E1_FIRST_RUN_JSON" ]; then echo "################ r3f_regression"; "$PY" "$D/r3f_regression.py" || exit 1; else echo "(E1_FIRST_RUN_JSON not given: r3f_regression.py skipped)"; fi
echo "################ r3_sandbox_rescore.sh"; bash "$D/r3_sandbox_rescore.sh" || exit 1
echo "all round-3 extras finished in $(( $(date +%s) - T0 )) s; outputs in $REVIEW_OUT"
