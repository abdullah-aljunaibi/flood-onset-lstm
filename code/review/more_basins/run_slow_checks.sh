#!/bin/bash
# The slower checks of the independent review of E3 (about 7 minutes with the laptop idle, one thread): test_flood.py and its eight broken variants of load_data (m5),
# the 15 saved networks rebuilt and scored (m7), and 17_more_basins.py --rescore in a sandbox copy (m14; needs about 0.2 GB in the temp folder, removed afterwards).
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/more_basins_review}
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
mkdir -p "$REVIEW_OUT"
cd "$HERE"
for s in m5_testflood_mutants m7_networks; do
  t0=$(date +%s)
  if nice -n 5 "$PY" -u "$s.py" > "$REVIEW_OUT/$s.log" 2>&1; then echo "ok      $s ($(( $(date +%s) - t0 )) s): $(tail -1 "$REVIEW_OUT/$s.log" | cut -c1-150)"; else echo "FAILED  $s"; tail -15 "$REVIEW_OUT/$s.log"; fi
done
t0=$(date +%s)
./m14_sandbox_rescore.sh > "$REVIEW_OUT/m14.log" 2>&1 && echo "ok      m14_sandbox_rescore ($(( $(date +%s) - t0 )) s): $(tail -1 "$REVIEW_OUT/m14.log" | cut -c1-200)" || { echo "FAILED  m14_sandbox_rescore"; tail -15 "$REVIEW_OUT/m14.log"; }
