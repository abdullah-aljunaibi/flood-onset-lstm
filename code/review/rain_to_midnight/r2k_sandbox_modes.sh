#!/bin/bash
# Round-2 check h2: python code/15_rain_to_midnight.py --rescore and --extend run in SANDBOX copies of the project (never in the project itself), and their outputs compared with the project's.
#  --rescore: code copied, data and the saved forecasts/networks linked read-only; the JSON and figure must equal the project's (but for the pass time).
#  --extend:  with every prediction set and network present nothing is trained and no file may be written; it gets COPIES of results/preds and results/models (about 0.3 GB), so that nothing could reach the project.
# One thread, about 4 minutes. The sandboxes are removed at the end.
set -e
PROJ="$(cd "$(dirname "$0")/../../.." && pwd)"
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
OUT=${REVIEW_OUT:-${TMPDIR:-/tmp}/rain_to_midnight_review}
mkdir -p "$OUT"; SB="$OUT/sandbox_rescore"; SE="$OUT/sandbox_extend"
rm -rf "$SB" "$SE"; mkdir -p "$SB" "$SE"
for D in "$SB" "$SE"; do
  mkdir "$D/code" "$D/figures" "$D/results" "$D/results/logs"
  cp "$PROJ"/code/*.py "$D/code/"
  ln -s "$PROJ/data" "$D/data"
  for f in data_audit.json day_windows.json flood_analysis.json satellite_data.json lstm_all.json rain_timing.json baselines.json; do ln -s "$PROJ/results/$f" "$D/results/$f"; done
  cp "$PROJ/results/rain_to_midnight.json" "$PROJ/results/review_rain_to_midnight.json" "$D/results/"
done
ln -s "$PROJ/results/preds" "$SB/results/preds"; ln -s "$PROJ/results/models" "$SB/results/models"
cp -R "$PROJ/results/preds" "$SE/results/preds"; cp -R "$PROJ/results/models" "$SE/results/models"
export FLOOD_THREADS=1 PYTHONDONTWRITEBYTECODE=1
echo "--rescore in the sandbox ..."; (cd "$SB" && nice -n 10 "$PY" code/15_rain_to_midnight.py --rescore > "$OUT/sandbox_rescore.log" 2>&1)
(cd "$SE" && find results/preds results/models -type f | sort | xargs shasum -a 256 > "$OUT/sandbox_extend_hashes_before.txt")
echo "--extend in the sandbox (every file present) ..."; (cd "$SE" && nice -n 10 "$PY" code/15_rain_to_midnight.py --extend > "$OUT/sandbox_extend.log" 2>&1)
(cd "$SE" && find results/preds results/models -type f | sort | xargs shasum -a 256 > "$OUT/sandbox_extend_hashes_after.txt")
same() { if cmp -s "$1" "$2"; then echo "$3: byte-identical"; else echo "$3: DIFFERS"; fi; }        # the sandbox results against the project's
same "$SB/figures/rain_to_midnight.png" "$PROJ/figures/rain_to_midnight.png" "--rescore: the figure against figures/rain_to_midnight.png"
same "$SE/figures/rain_to_midnight.png" "$PROJ/figures/rain_to_midnight.png" "--extend:  the figure against figures/rain_to_midnight.png"
same "$OUT/sandbox_extend_hashes_before.txt" "$OUT/sandbox_extend_hashes_after.txt" "--extend:  the sha256 list of all $(wc -l < "$OUT/sandbox_extend_hashes_after.txt" | tr -d ' ') prediction and network files before and after"
echo "--extend:  $(grep -c 'read, not trained again' "$OUT/sandbox_extend.log") lines 'read, not trained again' (21 expected: nothing trained)"
"$PY" - "$PROJ" "$SB" "$SE" <<'PYEOF'
import json, sys
proj, sb, se = sys.argv[1:4]
A = json.load(open(f"{proj}/results/rain_to_midnight.json"))
def diff(B):
    d = []
    def walk(a, b, p=()):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a or k not in b: d.append((p + (k,), "missing")); continue
                walk(a[k], b[k], p + (k,))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b): d.append((p, f"len {len(a)} vs {len(b)}")); return
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, p + (i,))
        elif a != b: d.append((p, f"{str(a)[:50]} vs {str(b)[:50]}"))
    walk(A, B); return d
for name, d in (("--rescore", sb), ("--extend", se)):
    dd = diff(json.load(open(f"{d}/results/rain_to_midnight.json")))
    print(f"{name}: {len(dd)} differing leaves between the project's results/rain_to_midnight.json and the sandbox's:", [x[0] for x in dd][:6])
PYEOF
find "$SB" "$SE" -type l -delete; rm -rf "$SB" "$SE"
