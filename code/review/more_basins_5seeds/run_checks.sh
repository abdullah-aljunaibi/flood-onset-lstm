#!/usr/bin/env bash
# Round 2 of the independent review of E3 (script 17), five seeds. Run from the project root. Nothing is trained; one torch thread.
# bash code/review/more_basins_5seeds/run_checks.sh                        # checks only (outputs to $REVIEW_OUT)
# bash code/review/more_basins_5seeds/run_checks.sh --write-into-project   # also appends round_2 (refuses if it exists)
set -e
PY=${PY:-python}
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 FLOOD_THREADS=1
H=code/review/more_basins_5seeds
for s in r1_integrity r2_score r3_compare r4_three_vs_five r5_networks; do echo "== $s"; "$PY" $H/$s.py; done
echo "== r6_assemble"; "$PY" $H/r6_assemble.py "$@"
