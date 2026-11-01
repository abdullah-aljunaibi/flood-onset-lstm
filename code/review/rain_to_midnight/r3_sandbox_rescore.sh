#!/bin/bash
# ROUND 3, check (e) and (b): python code/15_rain_to_midnight.py --rescore run in a SANDBOX copy of the project (never in the project itself), and its JSON and figure compared with the project's:
# is everything in results/rain_to_midnight.json and figures/rain_to_midnight.png a function of the saved prediction sets and of the script on disk? Code copied, data and the saved forecasts/networks linked
# read-only, the results file and the review record copied. Nothing is trained. One thread, niced, about 2 to 4 minutes. The sandbox is removed at the end. Output: $REVIEW_OUT.
set -e
PROJ="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/rain_to_midnight_review}
mkdir -p "$OUT"; SB="$OUT/sandbox_rescore_r3"
rm -rf "$SB"; mkdir -p "$SB"
mkdir "$SB/code" "$SB/figures" "$SB/results" "$SB/results/logs"
cp "$PROJ"/code/*.py "$SB/code/"
ln -s "$PROJ/data" "$SB/data"
for f in data_audit.json day_windows.json flood_analysis.json satellite_data.json lstm_all.json rain_timing.json baselines.json basin_audit.csv; do [ -e "$PROJ/results/$f" ] && ln -s "$PROJ/results/$f" "$SB/results/$f"; done
cp "$PROJ/results/rain_to_midnight.json" "$PROJ/results/review_rain_to_midnight.json" "$SB/results/"
ln -s "$PROJ/results/preds" "$SB/results/preds"; ln -s "$PROJ/results/models" "$SB/results/models"
SHA_BEFORE=$(shasum -a 256 "$PROJ/code/15_rain_to_midnight.py" "$PROJ/results/rain_to_midnight.json" "$PROJ/figures/rain_to_midnight.png" | cut -c1-64 | tr '\n' ' ')
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "--rescore in the sandbox ..."; T0=$(date +%s); (cd "$SB" && nice -n 10 "$PY" code/15_rain_to_midnight.py --rescore > "$OUT/sandbox_rescore_r3.log" 2>&1) || { echo "the sandbox --rescore FAILED; see $OUT/sandbox_rescore_r3.log"; tail -5 "$OUT/sandbox_rescore_r3.log"; exit 1; }
echo "done in $(( $(date +%s) - T0 )) s"
same() { if cmp -s "$1" "$2"; then echo "$3: byte-identical"; else echo "$3: DIFFERS"; fi; }
same "$SB/figures/rain_to_midnight.png" "$PROJ/figures/rain_to_midnight.png" "the sandbox figure against figures/rain_to_midnight.png"
"$PY" - "$PROJ" "$SB" <<'PYEOF'
import json, sys
proj, sb = sys.argv[1:3]
A = json.load(open(f"{proj}/results/rain_to_midnight.json")); B = json.load(open(f"{sb}/results/rain_to_midnight.json"))
d = []
def walk(a, b, p=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: d.append((p + (k,), "missing in one of the two")); continue
            walk(a[k], b[k], p + (k,))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): d.append((p, f"list length {len(a)} vs {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, p + (i,))
    elif a != b: d.append((p, f"{str(a)[:60]} vs {str(b)[:60]}"))
walk(A, B)
print(f"the project's results/rain_to_midnight.json against the sandbox's: {len(d)} differing leaves")
for p, why in d: print("   ", ".".join(map(str, p)), "|", why)
PYEOF
echo "files of the project that the rescore must not have touched (sha256 before; compared after): $SHA_BEFORE"
SHA_AFTER=$(shasum -a 256 "$PROJ/code/15_rain_to_midnight.py" "$PROJ/results/rain_to_midnight.json" "$PROJ/figures/rain_to_midnight.png" | cut -c1-64 | tr '\n' ' ')
[ "$SHA_BEFORE" = "$SHA_AFTER" ] && echo "the project's script, results file and figure are unchanged by this run" || echo "WARNING: a project file changed while this ran (another agent?)"
find "$SB" -type l -delete; rm -rf "$SB"
