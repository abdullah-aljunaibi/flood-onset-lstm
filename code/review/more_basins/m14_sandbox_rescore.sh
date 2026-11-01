#!/bin/bash
# Check (a)/(g): `17_more_basins.py --rescore` run in a SANDBOX copy of the project (nothing is trained; nothing is written into the project).
# The sandbox holds a copy of code/, the small results files the script reads, clones (cp -c) of the prediction sets and networks it reads, clones of the two caches, and links to the
# read-only CAMELS inputs. After the run the results file and the figure of the sandbox are compared with the project's (m14_compare.py).
# Usage: m14_sandbox_rescore.sh [SANDBOX_DIR]     (needs about 0.2 GB in the temp folder; removed at the end unless KEEP=1)
set -e
PROJ=${E3_PROJECT:-$HOME/AI_Class/Project-Flood}
PY=${PY:-$HOME/AI_Class/.venv/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"
SB=${1:-${REVIEW_OUT:-${TMPDIR:-/tmp}/more_basins_review}/sandbox_rescore}
rm -rf "$SB"; mkdir -p "$SB/code" "$SB/results/preds" "$SB/results/models" "$SB/figures" "$SB/data/camels_us" "$SB/data/derived"
cp "$PROJ"/code/*.py "$SB/code/"
for f in basin_audit.csv data_audit.json lstm_all.json lstm_lobo.json flood_analysis.json more_basins.json; do cp "$PROJ/results/$f" "$SB/results/$f"; done
for n in lstm_wq lstm_w lstm_wqf; do for s in "" _s0 _s1 _s2; do cp -c "$PROJ/results/preds/$n$s.npz" "$SB/results/preds/"; done; done
for n in lobo_w_static lobo_w_nostatic lobo_ref_w_nostatic lobo_rf_w; do cp -c "$PROJ/results/preds/$n.npz" "$SB/results/preds/"; done
for n in lstm63_wq lstm63_w lstm63eq_wq unseen50_w_static unseen50_w_nostatic; do for s in "" _s0 _s1 _s2; do cp -c "$PROJ/results/preds/$n$s.npz" "$SB/results/preds/"; done; done
for n in lstm63_wq lstm63_w lstm63eq_wq unseen50_w_static unseen50_w_nostatic; do for s in 0 1 2; do cp -c "$PROJ/results/models/${n}_s$s.pt" "$SB/results/models/"; done; done
cp -c "$PROJ"/data/derived/camels12_daymet.pkl "$PROJ"/data/derived/camels63_daymet_*.pkl "$SB/data/derived/"
for f in "$PROJ"/data/camels_us/camels_*.txt; do ln -s "$f" "$SB/data/camels_us/"; done
ln -s "$PROJ/data/camels_us/basin_dataset_public_v1p2" "$SB/data/camels_us/basin_dataset_public_v1p2"
[ -n "$NO_OUTLINES" ] || ln -s "$PROJ/data/camels_us/basin_set_full_res" "$SB/data/camels_us/basin_set_full_res"
cp "$PROJ/results/more_basins.json" "$SB/results/more_basins_project_copy.json"
cp "$PROJ/figures/more_basins.png" "$SB/figures/more_basins_project_copy.png"
shasum -a 256 "$PROJ"/results/preds/*.npz "$PROJ"/results/models/*.pt "$PROJ"/data/derived/*.pkl "$PROJ"/results/more_basins.json "$PROJ"/figures/more_basins.png > "$SB/project_hashes_before.txt"
cd "$SB"
PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS=1 nice -n 8 "$PY" code/17_more_basins.py --rescore > "$SB/rescore_stdout.log" 2> "$SB/rescore_stderr.log" && echo "rescore finished" || { echo "rescore FAILED"; tail -20 "$SB/rescore_stderr.log"; }
shasum -a 256 "$PROJ"/results/preds/*.npz "$PROJ"/results/models/*.pt "$PROJ"/data/derived/*.pkl "$PROJ"/results/more_basins.json "$PROJ"/figures/more_basins.png > "$SB/project_hashes_after.txt"
cmp "$SB/project_hashes_before.txt" "$SB/project_hashes_after.txt" && echo "the project's prediction sets, networks, caches, results file and figure are byte-identical before and after (nothing was written into the project)"
"$PY" "$HERE/m14_compare.py" "$SB"
[ -n "$KEEP" ] || rm -rf "$SB"
