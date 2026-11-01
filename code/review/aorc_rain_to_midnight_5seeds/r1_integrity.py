"""Round 2, check i1: which files the five-seed numbers rest on. sha256 of the script, the Stage IV check, the results files and every stored set of script 16;
seeds 0-2 against the hashes of round 1 (they must be the files round 1 recomputed), the seed averages remade, the results file's own integrity record against
the files as they are now, the passes, and the version of the results file that check_stage4_realtime.py read (11:51) against the one written at 12:44."""
from r_common import *
import time

C = Compare()
R1 = jload(REVIEW)["files_reviewed"]
J = jload(RESULTS)
out = dict(script_sha256=sha256(SCRIPT), stage4_script_sha256=sha256(STAGE4_SCRIPT), results_sha256=sha256(RESULTS), stage4_sha256=sha256(STAGE4),
           stage4_csv_sha256=sha256("data/aorc/stage4_issue_days.csv"))
C.add("script sha256 = passes[-1].script_sha256", J["passes"][-1]["script_sha256"], out["script_sha256"])
C.add("script sha256 = integrity.script_sha256", J["integrity"]["script_sha256"], out["script_sha256"])
S4 = jload(STAGE4)
C.add("stage4 code sha256 = the file now", S4["code"]["sha256"], out["stage4_script_sha256"])
# seeds 0-2 and the simple models: the files round 1 hashed
same, changed = [], []
for rel, v in R1.items():
    if rel.startswith(("results/preds/", "results/models/")):
        now = sha256(rel)
        (same if now == v["sha256"] else changed).append(rel)
out["round1_files_unchanged"] = sorted(same)
out["round1_files_changed"] = sorted(changed)
avg = {f"results/preds/lstm_{n}.npz" for n in ("wqa", "wqa_to09", "wqa_utc", "wqa_rep")}
C.add("round-1 files changed = exactly the four seed averages", sorted(avg), sorted(changed))
# the results file's own record of the stored files against the files now
saved = J["integrity"]["sha256_of_saved_files"]
bad = [p for p, h in saved.items() if sha256(p) != h]
C.add("integrity.sha256_of_saved_files equal the files now", [], bad)
out["integrity_files"] = len(saved)
# every seed of the four sets exists, and seeds 3-4 are new files
for n in ("wqa", "wqa_to09", "wqa_utc", "wqa_rep"):
    C.add(f"{n} seeds on disk", [0, 1, 2, 3, 4], seeds_of(n))
    for s in (3, 4):
        C.add(f"{n} s{s} absent from the 30-Oct backup", False, (BACKUP / f"results/preds/lstm_{n}_s{s}.npz").exists())
out["file_times"] = {p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((PROJ / p).stat().st_mtime)) for p in (RESULTS, STAGE4, SCRIPT, STAGE4_SCRIPT, REVIEW, "results/aorc_hindsight.json")}
out["passes"] = [{k: p.get(k) for k in ("kind", "n_trainings", "seeds_trained", "finished_local_time", "script_sha256")} for p in J["passes"]]
C.add("seeds", [0, 1, 2, 3, 4], J["seeds"])
C.add("n_trainings", 20, J["n_trainings"])
# the version check_stage4_realtime.py read: the copy taken right after the extension run (11:51) by the builder session
ext = Path(os.environ.get("AFTER_EXTEND", "")) / "aorc_rain_to_midnight.json"
if ext.exists():
    E = jload(ext.name, ext.parent)
    diff = [n for n in J["summary_test"]["sets"] if json.dumps(J["summary_test"]["sets"][n], sort_keys=True) != json.dumps(E["summary_test"]["sets"][n], sort_keys=True)]
    C.add("summary_test of the 11:51 copy = 12:44 file", [], diff)
    diff2 = [n for n in J["seed_ranges"]["sets"] if json.dumps(J["seed_ranges"]["sets"][n], sort_keys=True) != json.dumps(E["seed_ranges"]["sets"][n], sort_keys=True)]
    C.add("seed_ranges of the 11:51 copy = 12:44 file", [], diff2)
    top = [k for k in J if k not in ("passes", "integrity", "outcome_caveat", "outcome_full", "stamp_convention_check", "rescore_runtime_s", "runtime_s") and json.dumps(J.get(k), sort_keys=True) != json.dumps(E.get(k), sort_keys=True)]
    out["keys_differing_between_11_51_copy_and_12_44_file"] = top
    out["after_extend_copy"] = str(ext)
out["files_read"] = READ
out["compare"] = C.rows
out["summary"] = C.summary()
jsave("r1_integrity.json", out)
print(json.dumps(out["summary"], indent=1)); print("changed:", out["round1_files_changed"]); print(out["file_times"]); print(out.get("keys_differing_between_11_51_copy_and_12_44_file"))
