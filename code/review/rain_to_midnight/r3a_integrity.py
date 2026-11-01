"""ROUND 3, checks (a) and (d).
(a) Are the 51 saved prediction and network files (the script's own integrity list) still byte-identical to the hashes of round 2 (the 23 new files the round-2 record hashed, the 5 old ones round 1 hashed, and, when
    E1_R2_HASHES names it, the list of 119 prediction and network files the reviewer hashed in round 2: lines 'sha256  results/...')? Does the JSON's own record of the 51 agree with the disk?
(d) The `passes` entries (script_sha256 of the rescore, 'not recorded' for the earlier passes) and integrity.script_sha256 against the sha256 of the script on disk; the pass times against the logs.
Writes r3a_integrity.json with the 51 hashes as they are now (the baseline for a later round). A few seconds."""
import sys; sys.dont_write_bytecode = True
import hashlib, time, os, glob
from r2_common import *

sha = lambda rel: hashlib.sha256(open(f"{PROJ}/{rel}", "rb").read()).hexdigest()
mt = lambda rel: os.path.getmtime(f"{PROJ}/{rel}")
tf = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
R = jload("results/rain_to_midnight.json"); REV = jload("results/review_rain_to_midnight.json")
out = {}

# ---------------------------------------------------------------- (a)
saved = R["integrity"]["sha256_of_saved_files"]
now = {p: sha(p) for p in saved}
disagree = sorted(p for p in saved if now[p] != saved[p])
out["json_record_against_disk"] = dict(n_files=len(saved), disagree=disagree)
print(f"(a) the JSON's own record: {len(saved)} files, {len(disagree)} disagree with the disk now")
r2 = {p: v["sha256"] for p, v in REV["round_2"]["files_reviewed"].items() if p in saved}
r1 = {p: v["sha256"] for p, v in REV["files_reviewed"].items() if p in saved}
bad2 = sorted(p for p, h in r2.items() if now[p] != h); bad1 = sorted(p for p, h in r1.items() if now[p] != h)
out["against_the_review_record"] = dict(round_2_hashed=len(r2), round_2_changed=bad2, round_1_hashed=len(r1), round_1_changed=bad1, covered=len(set(r1) | set(r2)))
print(f"    round-2 record hashed {len(r2)} of the 51 (the new files), changed: {bad2}; round-1 record hashed {len(r1)} of the 51, changed: {bad1}; together {len(set(r1) | set(r2))} of 51")
lst = os.environ.get("E1_R2_HASHES", "")
if lst and os.path.exists(lst):
    mine = {l.split()[1]: l.split()[0] for l in open(lst).read().strip().splitlines()}
    ev = sorted(p for p in saved if p in mine)
    ev_bad = sorted(p for p in ev if now[p] != mine[p])
    rest = sorted(p for p in mine if p not in saved)
    rest_bad = sorted(p for p in rest if not os.path.exists(f"{PROJ}/{p}") or sha(p) != mine[p])
    now_files = sorted(os.path.relpath(f, PROJ) for pat in ("results/preds/*", "results/models/*") for f in glob.glob(f"{PROJ}/{pat}") if os.path.isfile(f))
    new_files = sorted(p for p in now_files if p not in mine)
    out["against_the_round_2_hash_list"] = dict(list_file=os.path.basename(lst), n_in_list=len(mine), of_the_51=len(ev), of_the_51_changed=ev_bad, other_files_in_list=len(rest), other_files_changed_or_gone=rest_bad, files_now_in_preds_and_models_not_in_the_list=new_files)
    print(f"    my round-2 list ({len(mine)} prediction and network files hashed at that time): {len(ev)} of the 51 in it, {len(ev_bad)} changed; the other {len(rest)} files of the list: {len(rest_bad)} changed or gone; new files in results/preds and results/models since: {len(new_files)} {new_files[:4]}")
else:
    print("    (E1_R2_HASHES not given: the list of 119 hashes made in round 2 was not used)")
si = R["integrity"]
print(f"    integrity.made_in_this_pass: {si['made_in_this_pass']} (a rescore writes no saved file); compared with the earlier results file: {si['compared_with_the_earlier_results_file']}; with the files the review hashed: {si['compared_with_the_files_the_review_hashed']['files_compared']} files, all equal {si['compared_with_the_files_the_review_hashed']['all_equal']}")
out["integrity_flags"] = dict(made_in_this_pass=si["made_in_this_pass"], earlier=si["compared_with_the_earlier_results_file"], review=dict(files_compared=si["compared_with_the_files_the_review_hashed"]["files_compared"], all_equal=si["compared_with_the_files_the_review_hashed"]["all_equal"]))
out["sha256_of_the_51_now"] = now

# ---------------------------------------------------------------- (d)
script = "code/15_rain_to_midnight.py"
h = sha(script)
P = R["passes"]
print(f"\n(d) script on disk: sha256 {h[:16]}..., modified {tf(mt(script))}; integrity.script_sha256 {si['script_sha256'][:16]}...: equal {si['script_sha256'] == h}; passes[-1].script_sha256 {P[-1]['script_sha256'][:16]}...: equal {P[-1]['script_sha256'] == h}")
for i, p in enumerate(P):
    print(f"    pass {i}: kind {p.get('kind', '(none)')!s:12s} n_trainings {p['n_trainings']:2d}  training_seconds {p['training_seconds']:7.1f}  runtime_s {p['runtime_s']:7.1f}  finished {p['finished_local_time']}  script_sha256 {p['script_sha256'][:16]}")
ok_shape = (len(P) == 3 and P[0]["script_sha256"] == "not recorded" and P[1]["script_sha256"] == "not recorded" and P[2]["script_sha256"] == h and P[2].get("kind") == "rescore")
sums = dict(n_trainings=sum(p["n_trainings"] for p in P), training_seconds=round(sum(p["training_seconds"] for p in P), 1), runtime_s_training_passes=round(sum(p["runtime_s"] for p in P if p.get("kind") != "rescore"), 1))
ok_sums = (sums["n_trainings"] == R["n_trainings"] and abs(sums["training_seconds"] - R["training_seconds"]) < 0.051 and abs(sums["runtime_s_training_passes"] - R["runtime_s"]) < 0.051 and abs(P[2]["runtime_s"] - R["rescore_runtime_s"]) < 1e-9)
print(f"    shape (3 passes: two 'not recorded', the rescore with the hash of the script on disk): {ok_shape}; totals {sums} against the JSON's n_trainings {R['n_trainings']}, training_seconds {R['training_seconds']}, runtime_s {R['runtime_s']} (the rescore's {R['rescore_runtime_s']} s kept apart): {ok_sums}")
logs = {"first run": ("results/logs/15_rain_to_midnight.log", P[0]), "extension": ("results/logs/15_rain_to_midnight_extend.log", P[1]), "rescore": ("results/logs/15_rain_to_midnight_rescore.log", P[2])}
tcheck = {}
for k, (lf, p) in logs.items():
    t = tf(mt(lf))[:16]; tcheck[k] = dict(log=lf, log_mtime=tf(mt(lf)), pass_says=p["finished_local_time"], same_minute=(t == p["finished_local_time"]))
    print(f"    {k:9s}: pass says finished {p['finished_local_time']}, log {lf} last written {tf(mt(lf))}: same minute {t == p['finished_local_time']}")
start_rescore = mt("results/rain_to_midnight.json") - P[2]["runtime_s"]
print(f"    the rescore started about {tf(start_rescore)} (results file written {tf(mt('results/rain_to_midnight.json'))} minus its {P[2]['runtime_s']} s); the script was last modified {tf(mt(script))}: the file that ran is the file on disk now: {mt(script) < start_rescore}")
note = [n for n in R["notes"] if "passes lists the training passes" in n]
print(f"    the note on passes (notes[{R['notes'].index(note[0]) if note else '-'}]) says the earlier hashes are 'not recorded' and not guessed: {bool(note) and 'not recorded' in note[0]}")
r1_script = REV["files_reviewed"][script]
print(f"    for the record: pass 0 (the first twelve trainings, finished 11:40) ran the version of the script that round 1 hashed (sha256 {r1_script['sha256'][:12]}..., modified {r1_script['modified']}, i.e. before the run began), so its hash is knowable from the review; the script does not record it")
out["passes"] = dict(script_sha256_on_disk=h, integrity_equal=(si["script_sha256"] == h), last_pass_equal=(P[-1]["script_sha256"] == h), shape_ok=ok_shape, sums=sums, sums_ok=ok_sums, log_times=tcheck,
                     script_modified=tf(mt(script)), rescore_started_about=tf(start_rescore), script_unchanged_since_the_rescore_started=(mt(script) < start_rescore), round_1_script_sha256=r1_script["sha256"])
jsave("r3a_integrity.json", out)
