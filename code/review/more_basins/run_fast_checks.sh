#!/bin/bash
# The fast checks of the independent review of E3, in order (about 3 minutes with the laptop idle, one thread). Nothing is trained; nothing is written into the project:
# every check writes its numbers to $REVIEW_OUT (default: <temp folder>/more_basins_review) and its printout to $REVIEW_OUT/<name>.log.
# Usage (from anywhere):  code/review/more_basins/run_fast_checks.sh        then   run_slow_checks.sh   then   m17_assemble_record.py (writes the record, see the README)
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/more_basins_review}
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
mkdir -p "$REVIEW_OUT"
cd "$HERE"
for s in m1_engine_diff m2_defaults m3_candidates_raw m3b_area_errors m6_headline m8_unseen m9_gain_rule m10_extra_rivers m11_figure m12_records_logs m13_handin m15_integrity m16_probe_properties; do
  t0=$(date +%s)
  if nice -n 5 "$PY" "$s.py" > "$REVIEW_OUT/$s.log" 2>&1; then echo "ok      $s ($(( $(date +%s) - t0 )) s): $(tail -1 "$REVIEW_OUT/$s.log" | cut -c1-150)"; else echo "FAILED  $s"; tail -15 "$REVIEW_OUT/$s.log"; fi
done
# the figure check must FAIL when every expected position is moved by 6 px (a check that cannot fail checks nothing)
if REVIEW_SELFTEST=6 nice -n 5 "$PY" m11_figure.py > "$REVIEW_OUT/m11_figure_selftest.log" 2>&1; then echo "FAILED  m11_figure self-test passed although the positions were moved by 6 px"; else echo "ok      m11_figure self-test fails when positions are moved by 6 px: $(tail -1 "$REVIEW_OUT/m11_figure_selftest.log" | cut -c1-120)"; fi
