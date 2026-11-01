#!/bin/bash
# E2 review, check (a) continued: the project's own consistency check of the store, python code/fetch_gefs.py --check (no download), run on a COPY of data/gefs/c00 in a sandbox, never in the project (a start of the
# fetcher touches .metadata_never_index files and rewrites done_apcp.txt; this run is pointed at the copy with --out). Exit code 0 = consistent. Needs ~/AI_Class/.venv-grib. About 5 seconds.
set -e
PROJ="$(cd "$(dirname "$0")/../../.." && pwd)"
PYG=${PYG:-$HOME/AI_Class/.venv-grib/bin/python}
OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/e2_review}
mkdir -p "$OUT"; SB="$OUT/sandbox_fetch_check"
rm -rf "$SB"; mkdir -p "$SB/code" "$SB/data/gefs" "$SB/results/logs"
cp -p "$PROJ/code/fetch_gefs.py" "$SB/code/"
cp -Rp "$PROJ/data/gefs/c00" "$SB/data/gefs/c00"
MARK="$OUT/.mark_fetch"; touch "$MARK"
BEFORE=$(cd "$SB/data/gefs/c00" && shasum -a 256 done_apcp.txt apcp_200001.npz apcp_201410.npz | cut -c1-64 | tr '\n' ' ')
set +e
(cd "$SB" && PYTHONDONTWRITEBYTECODE=1 "$PYG" code/fetch_gefs.py --check --out "$SB/data/gefs/c00" --no-caffeinate) > "$OUT/f16_fetch_check.log" 2>&1
CODE=$?
set -e
cat "$OUT/f16_fetch_check.log" | cut -c1-260
echo "exit code of --check: $CODE (0 = consistent)"
AFTER=$(cd "$SB/data/gefs/c00" && shasum -a 256 done_apcp.txt apcp_200001.npz apcp_201410.npz | cut -c1-64 | tr '\n' ' ')
[ "$BEFORE" = "$AFTER" ] && echo "the copy's done_apcp.txt and two month files are unchanged by the check" || echo "NOTE: the check changed a file of the copy"
echo "files of the project newer than the start of this run (outside code/review/forecast_input):"; (cd "$PROJ" && find . -type f -newer "$MARK" -not -path "./code/review/forecast_input/*" -not -name ".DS_Store" | head -10); echo "(end of list)"
echo "{\"exit_code\": $CODE}" > "$OUT/f16_fetch_check.json"
rm -rf "$SB"
