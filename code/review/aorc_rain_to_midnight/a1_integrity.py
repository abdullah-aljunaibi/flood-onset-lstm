"""AORC review, check (h) part 1 and the inventory: the files under review (script, fetcher, hindsight check, results files, figure, logs, prediction sets, networks, cache), their sha256, size and modification time;
the results file's own integrity record against the disk (30 saved files); the script's hash against integrity.script_sha256 and the pass; modification times in order (script before the run, results after);
what flood.py and test_flood.py are now against the hand-in zip of 4 October (given by E1_ZIP_DIR when available); the line of run_all.sh. Writes a1_integrity.json (the hashes are the baseline for files_reviewed).
A few seconds."""
import sys; sys.dont_write_bytecode = True
import hashlib, time, os, re, glob
from a_common import *

sha = lambda rel: hashlib.sha256(open(f"{PROJ}/{rel}", "rb").read()).hexdigest()
mt = lambda rel: os.path.getmtime(f"{PROJ}/{rel}")
tf = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
fi = lambda rel: dict(sha256=sha(rel), bytes=os.path.getsize(f"{PROJ}/{rel}"), modified=tf(mt(rel)))
R = jload("results/aorc_rain_to_midnight.json")
out = {}
SETS = ("wqa", "wqa_to09", "wqa_utc", "wqa_rep")
files = ["code/16_aorc_rain_to_midnight.py", "code/fetch_aorc.py", "code/check_aorc_hindsight.py", "results/aorc_rain_to_midnight.json", "results/aorc_data.json", "results/aorc_hindsight.json", "figures/aorc_rain_to_midnight.png",
         "results/logs/16_aorc_rain_to_midnight.log", "results/logs/fetch_aorc.log", "data/aorc/stage_iv_hindsight.csv", "run_all.sh", "code/flood.py", "code/test_flood.py", "code/features.py", "code/camels.py", "code/02_baselines.py", "results/baselines.json"]
saved = [f"results/preds/lstm_{n}.npz" for n in SETS] + [p for n in SETS for s in (0, 1, 2) for p in (f"results/preds/lstm_{n}_s{s}.npz", f"results/models/lstm_{n}_s{s}.pt")] + ["results/preds/base_linreg_wqa.npz", "results/preds/base_rf_wqa.npz"]
cache = sorted(os.path.relpath(f, PROJ) for f in glob.glob(f"{PROJ}/data/aorc/aorc_*.npz"))
inv = {f: fi(f) for f in files + saved + cache}
out["inventory"] = inv
print(f"{len(inv)} files hashed: {len(files)} code/results/figure/log files, {len(saved)} saved prediction sets and networks, {len(cache)} cache files")
for f in files: print(f"  {f:46s} {inv[f]['modified']}  {inv[f]['bytes']:>9d} bytes  sha256 {inv[f]['sha256'][:12]}")

# the results file's own record
I = R["integrity"]; rec = I["sha256_of_saved_files"]
dis = sorted(p for p, h in rec.items() if not os.path.exists(f"{PROJ}/{p}") or sha(p) != h)
miss = sorted(set(saved) - set(rec))
print(f"\nintegrity record of the results file: {len(rec)} saved files; disagree with the disk: {dis}; saved files I list that it does not: {miss}; it lists files I do not: {sorted(set(rec) - set(saved))}")
out["json_record"] = dict(n=len(rec), disagree=dis, not_in_record=miss, extra_in_record=sorted(set(rec) - set(saved)), made_in_this_pass=len(I["made_in_this_pass"]), compared_with_the_earlier_results_file=I["compared_with_the_earlier_results_file"],
                          compared_with_the_files_the_review_hashed=I["compared_with_the_files_the_review_hashed"])

# the script hash and the pass
h = inv["code/16_aorc_rain_to_midnight.py"]["sha256"]
P = R["passes"]
print(f"script on disk {h[:12]}... modified {inv['code/16_aorc_rain_to_midnight.py']['modified']}; integrity.script_sha256 {I['script_sha256'][:12]}...: equal {I['script_sha256'] == h}; passes: " + "; ".join(f"{p.get('kind')} {p['n_trainings']} trainings, finished {p['finished_local_time']}, script {p['script_sha256'][:12]}" for p in P))
t_json, t_script = mt("results/aorc_rain_to_midnight.json"), mt("code/16_aorc_rain_to_midnight.py")
t_models = sorted(mt(f) for f in saved if f.endswith(".pt"))
started = t_json - R["runtime_s"] - R.get("waited_for_cpu_s", 0)
order = dict(script_modified=tf(t_script), first_network_written=tf(t_models[0]), last_network_written=tf(t_models[-1]), results_file_written=tf(t_json), pass_started_about=tf(started),
             script_older_than_the_first_network=t_script < t_models[0], results_file_newer_than_the_last_network=t_json >= t_models[-1])
print("times:", order)
out["script"] = dict(sha256=h, integrity_equal=(I["script_sha256"] == h), passes=[dict(kind=p.get("kind"), n_trainings=p["n_trainings"], finished=p["finished_local_time"], script_sha256=p["script_sha256"]) for p in P], order=order)

# other files that changed since the builder wrote the results file
later = {f: v["modified"] for f, v in inv.items() if mt(f) > t_json + 1}
print("files of the list modified after the results file was written:", later)
out["modified_after_the_results_file"] = later

# run_all.sh
ra = open(f"{PROJ}/run_all.sh").read().splitlines()
lines = {i + 1: l.strip()[:150] for i, l in enumerate(ra) if re.search(r"16_aorc|15_rain|fetch_aorc|check_day_windows|check_aorc|17_more", l)}
print("run_all.sh lines:", lines)
out["run_all"] = lines

# the engine files against the hand-in zip, when it is given
zd = os.environ.get("E1_ZIP_DIR", "")
if zd and os.path.exists(zd):
    out["engine_vs_zip"] = {f: (hashlib.sha256(open(f"{zd}/{f}", "rb").read()).hexdigest() == sha(f"code/{f}")) for f in ("flood.py", "test_flood.py", "features.py", "camels.py", "02_baselines.py", "common.py") if os.path.exists(f"{zd}/{f}")}
    print("code files equal to the hand-in zip of 4 October:", out["engine_vs_zip"])
jsave("a1_integrity.json", out)
