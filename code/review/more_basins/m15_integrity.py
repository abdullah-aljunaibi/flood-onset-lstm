"""Check (g), part 4: the files of this review (sha256, bytes, modification time), the hashes that results/more_basins.json recorded for its own prediction sets and networks against the files now,
and the quick-mode flags of every saved file of the script. Run last (or first): a changed hash between two runs means a file moved under the review.
Run: python m15_integrity.py"""
import time

from m_common import *

J = json.loads((PROJ / "results" / "more_basins.json").read_text())
fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
LIST = ["code/17_more_basins.py", "results/more_basins.json", "figures/more_basins.png", "results/logs/17_more_basins.log", "results/logs/17_more_basins_interim_noequal.log", "results/logs/17_more_basins_rescore.log",
        "code/flood.py", "code/test_flood.py", "code/camels.py", "code/common.py", "code/features.py", "code/01_data_audit.py", "code/04_lstm_all.py", "code/05_lobo.py", "run_all.sh", "make_handin_zip.sh", "ROADMAP.md", "README.md", "NEXT_STEPS_FOR_AGENTS.md",
        "results/basin_audit.csv", "results/data_audit.json", "results/lstm_all.json", "results/lstm_lobo.json", "results/flood_analysis.json", "results/engine_review.json", "results/review_rain_to_midnight.json",
        "data/derived/camels12_daymet.pkl", "data/derived/camels63_daymet_408f92a4b6.pkl"]
for n in ("lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic"):
    LIST += [f"results/preds/{n}.npz"] + [f"results/preds/{n}_s{s}.npz" for s in (0, 1, 2)] + [f"results/models/{n}_s{s}.pt" for s in (0, 1, 2)]
res = dict(files={})
for rel in LIST:
    f = PROJ / rel
    res["files"][rel] = dict(sha256=sha256(f), bytes=f.stat().st_size, modified=fmt(f.stat().st_mtime))
rec = J["integrity"]["sha256_of_saved_files"]
cmp = {k: (v == (sha256(PROJ / k) if (PROJ / k).exists() else None)) for k, v in rec.items()}
res["integrity_record_of_the_results_file"] = dict(n_files_recorded=len(rec), n_equal_now=sum(cmp.values()), all_equal=all(cmp.values()), not_equal=[k for k, v in cmp.items() if not v])
quick = {}
for n in ("lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic"):
    for s in ("", "_s0", "_s1", "_s2"):
        quick[f"{n}{s}"] = load_npz(f"{n}{s}")["meta"].get("quick_mode")
res["quick_mode_flags_of_the_20_prediction_files"] = dict(all_false=all(v is False for v in quick.values()), values=quick)
res["results_quick_folder_has_17_more_basins_output"] = [str(p.relative_to(PROJ)) for p in (PROJ / "results" / "quick").rglob("*") if p.is_file() and ("more_basins" in p.name or "lstm63" in p.name or "unseen50" in p.name)][:10]
print(f"{len(LIST)} files hashed; the results file's own record: {res['integrity_record_of_the_results_file']['n_equal_now']} of {res['integrity_record_of_the_results_file']['n_files_recorded']} equal now; "
      f"quick-mode flags of the 20 prediction files all false: {res['quick_mode_flags_of_the_20_prediction_files']['all_false']}; E3 files in results/quick: {res['results_quick_folder_has_17_more_basins_output']}")
for rel in ("code/17_more_basins.py", "code/flood.py", "code/test_flood.py", "results/more_basins.json", "figures/more_basins.png"):
    print(f"  {rel:32s} {res['files'][rel]['sha256'][:16]}  {res['files'][rel]['bytes']:>8d} bytes  {res['files'][rel]['modified']}")
save("m15_integrity", res)
