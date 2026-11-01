"""ROUND 2, check (a): integrity. Which files changed since round 1, are the original 28 model/prediction files unchanged, does the script's integrity record agree with the disk,
are the engine files unchanged, and do the 12 original seed files still carry the numbers recorded in round 1 (regression)."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
import hashlib, time, os
from r2_common import *
import flood as F
from r2_scoring import *

sha = lambda rel: hashlib.sha256(open(f"{PROJ}/{rel}", "rb").read()).hexdigest()
mt = lambda rel: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f"{PROJ}/{rel}")))
R1 = jload("results/review_rain_to_midnight.json")
out = {}

# 1. the hashes recorded in round 1
rec = R1["files_reviewed"]
out["round1_recorded"] = {f: dict(recorded=v["sha256"][:16], now=sha(f)[:16], identical=(sha(f) == v["sha256"]), mtime_recorded=v["modified"], mtime_now=mt(f)) for f, v in rec.items()}
print("round-1 recorded hashes (12 files):")
for f, v in out["round1_recorded"].items():
    print(f"  {'IDENTICAL' if v['identical'] else 'CHANGED  '} {f}")

# 2. the 28 original files
orig = []
for n in ("wqh", "wqh_older", "wh", "wh_older"):
    orig.append(f"results/preds/lstm_{n}.npz")
    for s in (0, 1, 2):
        orig += [f"results/preds/lstm_{n}_s{s}.npz", f"results/models/lstm_{n}_s{s}.pt"]
assert len(orig) == 28
now = {f: sha(f) for f in orig}
in_r1 = [f for f in orig if f in rec]
out["original_28"] = dict(n=28, hashed_in_round_1=in_r1, n_hashed_in_round_1=len(in_r1), those_identical=all(now[f] == rec[f]["sha256"] for f in in_r1),
                          latest_mtime=max(mt(f) for f in orig), earliest_mtime=min(mt(f) for f in orig))
print(f"\nthe 28 original files: round 1 hashed {len(in_r1)} of them; those {len(in_r1)} are identical: {out['original_28']['those_identical']}; mtimes {out['original_28']['earliest_mtime']} to {out['original_28']['latest_mtime']}")
# the coordinator's record of 16:28 (before the extension) lists all 28
cf = os.environ.get("E1_MANIFEST_28", "")            # optional: a "sha256  ./path" list of the 28 original files made before the extension (the coordinator made one at 16:28); skipped when not given
if cf and os.path.exists(cf):
    lines = [l.split() for l in open(cf).read().strip().splitlines()]
    coord = {l[1][2:]: l[0] for l in lines}
    out["coordinator_record_16_28"] = dict(n=len(coord), all_28_present=set(coord) == set(orig), all_equal_to_now=all(coord[f] == now[f] for f in orig),
                                           the_5_i_hashed_agree_with_it=all(coord[f] == rec[f]["sha256"] for f in in_r1))
    print("the manifest given in E1_MANIFEST_28:", out["coordinator_record_16_28"])

# 3. the script's own integrity record
J = jload("results/rain_to_midnight.json")
H = J["integrity"]["sha256_of_saved_files"]
out["script_integrity_record"] = dict(n_files=len(H), disagree_with_disk=[f for f, h in H.items() if sha(f) != h], made_in_this_pass=J["integrity"]["made_in_this_pass"],
                                      compared_with_earlier=J["integrity"]["compared_with_the_earlier_results_file"], compared_with_review=J["integrity"]["compared_with_the_files_the_review_hashed"])
print("\nthe script's integrity record:", {k: (v if k != 'disagree_with_disk' else len(v)) for k, v in out["script_integrity_record"].items()})

# 4. files changed since round 1 (modified after 2026-10-07 12:30), and the engine files against the hand-in zip
import subprocess
res = subprocess.run(["find", PROJ, "-newermt", "2026-10-07 12:30:00", "-type", "f", "-not", "-path", "*/__pycache__/*", "-not", "-path", f"{PROJ}/data/*"], capture_output=True, text=True).stdout.split()
rel = sorted(os.path.relpath(p, PROJ) for p in res)
groups = dict(script=[f for f in rel if f.startswith("code/")], results_json_figures_logs=[f for f in rel if f in ("results/rain_to_midnight.json", "figures/rain_to_midnight.png") or f.startswith("results/logs/")],
              new_models=[f for f in rel if f.startswith("results/models/")], new_preds=[f for f in rel if f.startswith("results/preds/")], smoke_test=[f for f in rel if "/quick/" in f], other=[f for f in rel if not (f.startswith(("code/", "results/models/", "results/preds/", "results/logs/")) or "/quick/" in f or f in ("results/rain_to_midnight.json", "figures/rain_to_midnight.png"))])
out["changed_since_round_1"] = {k: v for k, v in groups.items()}
print("\nfiles modified after 12:30 today (outside data/):", {k: len(v) for k, v in groups.items()})
print("   code:", groups["script"], "| other:", groups["other"])
import zipfile
ZIP = Path(PROJ).parent / f"{Path(PROJ).name}_handin.zip"        # the hand-in archive of 2026-10-04 next to the project folder
def zip_sha(name):
    if not ZIP.exists(): return None
    with zipfile.ZipFile(ZIP) as z:
        try: return hashlib.sha256(z.read(f"{Path(PROJ).name}/code/{name}")).hexdigest()
        except KeyError: return None
eng = {}
for f in ("flood.py", "test_flood.py", "common.py", "camels.py", "features.py", "02_baselines.py", "10_rain_timing.py", "check_day_windows.py"):
    eng[f] = (zip_sha(f) == sha(f"code/{f}")) if zip_sha(f) else None
out["engine_and_recipe_files_equal_to_handin_zip_2026_10_04"] = eng
print("engine / recipe files identical to the hand-in zip of 2026-10-04:", eng)
# the new script file
out["script_now"] = dict(sha256=sha("code/15_rain_to_midnight.py"), mtime=mt("code/15_rain_to_midnight.py"), bytes=os.path.getsize(f"{PROJ}/code/15_rain_to_midnight.py"),
                         extend_log_mtime=mt("results/logs/15_rain_to_midnight_extend.log"), rescore_log_mtime=mt("results/logs/15_rain_to_midnight_rescore.log"), json_mtime=mt("results/rain_to_midnight.json"), figure_mtime=mt("figures/rain_to_midnight.png"))
print("\nscript now:", out["script_now"])

# 5. regression: the numbers of round 1 (stored in the review JSON) against the files now
byseed_pod, byseed_nse = R1["numbers"]["onsets_by_seed_lead1"], R1["numbers"]["median_nse_by_seed_lead1"]
dev = 0; n = 0
for s_, seeds in byseed_pod.items():
    for sd in (0, 1, 2):
        c = counts(f"lstm_{s_}_s{sd}", 1)["pod"]
        nse = float(np.median(list(nse_per_basin(f"lstm_{s_}_s{sd}", 1).values())))
        dev = max(dev, abs(c - seeds[sd]), abs(nse - byseed_nse[s_][sd])); n += 2
out["regression_vs_round1_numbers"] = dict(n_numbers=n, max_abs_deviation=dev, sets=sorted(byseed_pod))
print(f"\nregression: {n} per-seed onset shares and median NSEs of 12 sets (lead 1) recomputed now against the round-1 record: max deviation {dev}")
# the 4 averaged sets are exact means of the seed files
mean_ok = {}
for s_ in ("wqh", "wqh_older", "wh", "wh_older", "wqh_to09", "wqh_utc", "wqnh"):
    Z = [np.load(f"{PROJ}/results/preds/lstm_{s_}_s{sd}.npz") for sd in (0, 1, 2)]
    A = np.load(f"{PROJ}/results/preds/lstm_{s_}.npz")
    P = np.stack([z["pred"] for z in Z])
    mean_ok[s_] = dict(max_abs_diff=float(np.nanmax(np.abs(A["pred"] - (P[0] + P[1] + P[2]) / 3))), obs_equal_across_seeds=all(np.array_equal(Z[0]["obs"], z["obs"]) for z in Z), obs_equal_to_average=np.array_equal(Z[0]["obs"], A["obs"]),
                       nan_pattern_equal=all(np.array_equal(np.isnan(Z[0]["pred"]), np.isnan(z["pred"])) for z in Z) and np.array_equal(np.isnan(A["pred"]), np.isnan(Z[0]["pred"])))
out["seed_average_is_the_mean"] = mean_ok
print("seed averages equal the mean of their three seed files (7 sets):", {k: v["max_abs_diff"] for k, v in mean_ok.items()}, all(v["obs_equal_across_seeds"] and v["nan_pattern_equal"] for v in mean_ok.values()))
jsave("r2a_integrity.json", out)
