#!/bin/bash
# Fetches the hourly Stage IV files the hindsight checks need, from the Iowa Environmental Mesonet archive (volunteer-run: at most 6 parallel connections), reduces each to the 12 basin means at once and drops the file.
# For the 28 days of a7a_days.json: 33 hours per day (924 files) and 5 hours after 12 UTC of the next day (140 files), plus six check hours: about 1,070 files of about 1.2 MB, that is ABOUT 1.3 GB in all, 8 to 9 minutes.
# Resume-safe: hours that are already in $REVIEW_OUT/cache/stage4_means.csv are not asked for again (a second run fetches nothing). Run a3_clocks.py and a7a_select_days.py first (run_fast_checks.sh does the first).
cd "$(dirname "$0")/../../.." || exit 1
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export REVIEW_OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/aorc_review}
D=code/review/aorc_rain_to_midnight
"$PY" "$D/a7a_select_days.py" || exit 1
DAYS=$("$PY" -c "import json,os; print(' '.join(json.load(open(os.environ['REVIEW_OUT'] + '/a7a_days.json'))['days']))")
"$PY" "$D/a6_stage4.py" days $DAYS || exit 1
"$PY" "$D/a6_stage4.py" after $DAYS || exit 1
"$PY" "$D/a6_stage4.py" check
