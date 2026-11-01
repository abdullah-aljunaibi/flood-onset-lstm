#!/bin/bash
# E2 review, check (h) and (f): python code/18_forecast_input.py --rescore run in a SANDBOX copy of the project (never in the project itself), and its JSON, figure and derived prediction sets compared with the project's.
# Is everything in results/forecast_input.json and figures/forecast_input.png a function of the saved prediction sets, the saved networks and the script on disk? Code, results files, prediction sets, networks, logs and data/derived
# are COPIED (times kept, so that "written at" notes agree), the large data folders are linked; the rescore writes the derived prediction sets, which land in the copy. Nothing is trained. One thread, niced, about 3 to 5 minutes.
# Usage: f15_sandbox_rescore.sh [with-review]   (with-review: the reviewer's own record results/review_forecast_input.json, if it exists, is copied in too, to see that the script reads its counts)
# The sandbox is removed at the end. Output: $REVIEW_OUT (log and a json of the comparison).
set -e
PROJ="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/e2_review}
MODE=${1:-plain}
mkdir -p "$OUT"; SB="$OUT/sandbox_rescore_$MODE"
rm -rf "$SB"; mkdir -p "$SB/code" "$SB/figures" "$SB/results" "$SB/data"
cp -p "$PROJ"/code/*.py "$SB/code/"
for d in "$PROJ"/data/*; do b=$(basename "$d"); if [ "$b" = derived ]; then cp -Rp "$d" "$SB/data/derived"; else ln -s "$d" "$SB/data/$b"; fi; done
for f in "$PROJ"/results/*; do b=$(basename "$f"); case "$b" in review_forecast_input.json) ;; quick) ;; *) cp -Rp "$f" "$SB/results/$b";; esac; done
if [ "$MODE" = with-review ] && [ -e "$PROJ/results/review_forecast_input.json" ]; then cp -p "$PROJ/results/review_forecast_input.json" "$SB/results/"; fi
mkdir -p "$SB/results/quick"
MARK="$OUT/.mark_$MODE"; touch "$MARK"
HASHES="code/18_forecast_input.py code/flood.py code/gefs_to_basins.py results/forecast_input.json results/gefs_basins.json figures/forecast_input.png"
SHA_BEFORE=$(cd "$PROJ" && shasum -a 256 $HASHES | cut -c1-64 | tr '\n' ' ')
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "--rescore in the sandbox ($MODE) ..."; T0=$(date +%s)
(cd "$SB" && nice -n 10 "$PY" code/18_forecast_input.py --rescore > "$OUT/sandbox_rescore_$MODE.log" 2>&1) || { echo "the sandbox --rescore FAILED; see $OUT/sandbox_rescore_$MODE.log"; tail -8 "$OUT/sandbox_rescore_$MODE.log"; exit 1; }
echo "done in $(( $(date +%s) - T0 )) s"
if cmp -s "$SB/figures/forecast_input.png" "$PROJ/figures/forecast_input.png"; then echo "the sandbox figure against figures/forecast_input.png: byte-identical"; else echo "the sandbox figure against figures/forecast_input.png: DIFFERS"; fi
"$PY" - "$PROJ" "$SB" "$OUT" "$MODE" <<'PYEOF'
import json, sys, glob, os
import numpy as np
proj, sb, out, mode = sys.argv[1:5]
A = json.load(open(f"{proj}/results/forecast_input.json")); B = json.load(open(f"{sb}/results/forecast_input.json"))
d = []
def walk(a, b, p=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: d.append((p + (k,), "missing in one of the two")); continue
            walk(a[k], b[k], p + (k,))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): d.append((p, f"list length {len(a)} vs {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, p + (i,))
    elif a != b: d.append((p, f"{str(a)[:70]} vs {str(b)[:70]}"))
walk(A, B)
print(f"the project's results/forecast_input.json against the sandbox's ({mode}): {len(d)} differing leaves")
for p, why in d[:12]: print("   ", ".".join(map(str, p)), "|", why)
# the derived prediction sets, by content (the zip headers carry the time of writing, so bytes differ)
names = sorted(os.path.basename(f) for f in glob.glob(f"{proj}/results/preds/*.npz"))
nd, ne = 0, 0
for n in names:
    za, zb = np.load(f"{proj}/results/preds/{n}", allow_pickle=True), np.load(f"{sb}/results/preds/{n}", allow_pickle=True)
    same = sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[k], zb[k], equal_nan=True) if za[k].dtype.kind in "fc" else np.array_equal(za[k], zb[k]) for k in za.files)
    ne += 1; nd += (not same)
    if not same and nd <= 5: print("   prediction set differs in content:", n)
print(f"prediction sets of the project compared with the sandbox's by content: {ne} files, {nd} differ")
rev = [k for k in ("review_item",)]
notes = B.get("what_was_not_done", [])
print("first item of what_was_not_done in the sandbox's file:", str(notes[0])[:230])
json.dump(dict(mode=mode, leaves_differing=[[".".join(map(str, p)), w] for p, w in d], prediction_files_compared=ne, prediction_files_differing=nd, first_not_done_item=str(notes[0])), open(f"{out}/f15_sandbox_{mode}.json", "w"), indent=1)
PYEOF
SHA_AFTER=$(cd "$PROJ" && shasum -a 256 $HASHES | cut -c1-64 | tr '\n' ' ')
[ "$SHA_BEFORE" = "$SHA_AFTER" ] && echo "the project's script, engine, builder, results files and figure are unchanged by this run" || echo "WARNING: a project file changed while this ran (another agent?)"
echo "files of the project newer than the start of this run (outside code/review/forecast_input):"; (cd "$PROJ" && find . -type f -newer "$MARK" -not -path "./code/review/forecast_input/*" -not -name ".DS_Store" | head -10); echo "(end of list)"
find "$SB" -type l -delete; rm -rf "$SB"
