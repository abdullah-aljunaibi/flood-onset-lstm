#!/usr/bin/env bash
# Run from the project root. One thread, no bytecode in the project; outputs in $REVIEW_OUT. Add --write-into-project to r5 to append the block to the record.
set -e
PY=${PY:-python}
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
for s in r1_scores r2_compare r3_backup_diff r4_texts; do "$PY" code/review/rain_to_midnight_5seeds/$s.py; done
"$PY" code/review/rain_to_midnight_5seeds/r5_assemble.py "$@"
