"""Check i1-i6: the files of seeds 3-4 and the bookkeeping of the five-seed pass of script 17.

i1  every prediction set and network of the five sets exists (5 sets x 5 seeds + 5 seed averages; 25 networks); sha256 of each against results/more_basins.json integrity
i2  seeds 0-2 files and the stored controls are the files of the three-seed state (sha256 recorded in the backup results file / round 1 of the review)
i3  meta of every seed file (script, config, training-river digest, quick mode) equal across the five seeds of a set; the seed-3/4 networks' own records
    (set, seed, script, training rivers, settings) against the seed-0 network of the same set; the training records of results 'training' and 'stopping' against the .pt files
i4  every seed average = the mean of its five (three) seed files, summed in seed order as flood.ensemble, bit for bit; obs identical in every file; 12 basins; dates
i5  passes: sums of n_trainings, training_seconds, runtime_s, waited; the trainings not covered by passes; files made by the killed run (12:45-12:47) against the rerun
i6  integrity.made_in_this_pass / earlier_pass_compared against what the two runs of 30 October made
Run from the project root: python code/review/more_basins_5seeds/r1_integrity.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import datetime as dt

from r_common import *

import torch

torch.set_num_threads(1)

R, B = results(), results(BACKUP)
out = dict(i1={}, i2={}, i3={}, i4={}, i5={}, i6={})

# ---- i1: files and their hashes -----------------------------------------------------------------------------------------------------------------------
SAVED = R["integrity"]["sha256_of_saved_files"]
want = [f"results/preds/{n}.npz" for n in NEW] + [f"results/preds/{n}_s{s}.npz" for n in NEW for s in SEEDS] + [f"results/models/{n}_s{s}.pt" for n in NEW for s in SEEDS]
now = {f: sha256(PROJ / f) if (PROJ / f).exists() else None for f in want}
mtime = {f: dt.datetime.fromtimestamp((PROJ / f).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if (PROJ / f).exists() else None for f in want}
out["i1"] = dict(n_expected=len(want), n_present=sum(v is not None for v in now.values()), missing=[f for f, v in now.items() if v is None],
                 recorded_in_results=sum(f in SAVED for f in want), extra_recorded=sorted(set(SAVED) - set(want)),
                 equal_to_recorded=sum(SAVED.get(f) == v for f, v in now.items()), differ=[f for f, v in now.items() if SAVED.get(f) != v], sha256_now=now, mtime=mtime)
print("i1 files:", out["i1"]["n_present"], "of", len(want), "present;", out["i1"]["equal_to_recorded"], "equal to the recorded sha256; differ:", out["i1"]["differ"], "extra:", out["i1"]["extra_recorded"])

# ---- i2: three-seed files unchanged --------------------------------------------------------------------------------------------------------------------
OLD = B["integrity"]["sha256_of_saved_files"]
old_seed = {f: h for f, h in OLD.items() if not any(f == f"results/preds/{n}.npz" for n in NEW)}
ctrl_files = [f"results/preds/{m}.npz" for n in CTRL + LOBO for m in files_of(n)]
rev = results(PROJ / "results" / "review_more_basins.json")
rev_hash = {k: v["sha256"] for k, v in rev["files_reviewed"].items() if isinstance(v, dict) and "sha256" in v}
out["i2"] = dict(seed_0_2_files_in_backup=len(old_seed), seed_0_2_files_equal_now=sum(sha256(PROJ / f) == h for f, h in old_seed.items()),
                 seed_0_2_differ=[f for f, h in old_seed.items() if sha256(PROJ / f) != h],
                 seed_averages_three_seed_sha_differs_now={f"results/preds/{n}.npz": OLD.get(f"results/preds/{n}.npz") != now[f"results/preds/{n}.npz"] for n in NEW},
                 seed_0_2_equal_round_1_review={f: rev_hash[f] == sha256(PROJ / f) for f in rev_hash if f.startswith("results/") and ("_s0." in f or "_s1." in f or "_s2." in f)},
                 controls={f: sha256(PROJ / f) for f in ctrl_files},
                 controls_unchanged_since_backup_mtime={f: (PROJ / f).stat().st_mtime < dt.datetime(2026, 10, 30).timestamp() for f in ctrl_files})
print("i2 seed 0-2 files equal to the three-seed state:", out["i2"]["seed_0_2_files_equal_now"], "of", len(old_seed), "| equal to round-1 review hashes:",
      sum(out["i2"]["seed_0_2_equal_round_1_review"].values()), "of", len(out["i2"]["seed_0_2_equal_round_1_review"]), "| controls written before 30 Oct:", all(out["i2"]["controls_unchanged_since_backup_mtime"].values()))

# ---- i3: meta and network records ------------------------------------------------------------------------------------------------------------------------
cfg04 = results(PROJ / "results" / "lstm_all.json")["config"]
meta_rows, pt_rows = {}, {}
for n in NEW:
    metas = {s: load_npz(f"{n}_s{s}")["meta"] for s in SEEDS}
    keys = sorted(set().union(*[m.keys() for m in metas.values()]))
    diff_keys = [k for k in keys if any(metas[s].get(k) != metas[0].get(k) for s in SEEDS)]
    am = load_npz(n)["meta"]
    exp_conf = dict(cfg04, updates=5 * cfg04["updates"], eval_every=5 * cfg04["eval_every"]) if n == "lstm63eq_wq" else cfg04
    meta_rows[n] = dict(keys=keys, keys_differing_between_seeds=diff_keys, values_differing={k: [metas[s].get(k) for s in SEEDS] for k in diff_keys},
                        script=[metas[s].get("script") for s in SEEDS], config_as_expected=[metas[s].get("config") == exp_conf for s in SEEDS],
                        quick=[metas[s].get("quick_mode") for s in SEEDS], average_meta=am)
    zs = {s: torch.load(MODELS / f"{n}_s{s}.pt", map_location="cpu", weights_only=True) for s in SEEDS}
    tr = {(t["set"], t["seed"]): t for t in R["training"]}
    st = R["stopping"]["sets"][n]
    rows = {}
    for s in SEEDS:
        z = zs[s]
        t = tr[(n, s)]
        rows[s] = dict(set_ok=z["set"] == n, seed_ok=z["seed"] == s, script=z["script"], ids_equal_seed0=list(z["ids"]) == list(zs[0]["ids"]), n_ids=len(z["ids"]),
                       cfg_equal_seed0_but_seed={k: v for k, v in z["cfg"].items() if k != "seed" and v != zs[0]["cfg"].get(k)}, cfg_seed=z["cfg"].get("seed"),
                       static_equal=z["static"] == zs[0]["static"], scored_ids_equal=z["scored_ids"] == zs[0]["scored_ids"], split_equal=z["split"] == zs[0]["split"],
                       sc_keys_equal=sorted(z["sc"]) == sorted(zs[0]["sc"]), torch_threads=z["torch_threads"], torch_version=z["torch_version"], numpy_version=z["numpy_version"],
                       updates_run=z["updates_run"], best_update=z["best_update"], seconds=z["seconds"], best_val_nse=z["best_val_nse"],
                       history_load_values=sorted({str(h.get("load")) for h in z["history"]})[:3],
                       training_record_equal=all(close(z[c], t[c], 1e-12) for c in ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")),
                       stopping_equal=st["updates_run"][s] == z["updates_run"] and st["best_update"][s] == z["best_update"] and close(st["seconds"][s], z["seconds"], 1e-12),
                       reached_limit=bool(z["updates_run"] >= z["cfg"]["updates"]), reached_limit_in_results=t["reached_the_update_limit"])
    pt_rows[n] = rows
    del zs
ok3 = all(all(r["set_ok"] and r["seed_ok"] and r["ids_equal_seed0"] and not r["cfg_equal_seed0_but_seed"] and r["static_equal"] and r["split_equal"] and r["training_record_equal"] and r["stopping_equal"]
              and r["reached_limit"] == r["reached_limit_in_results"] for r in rows.values()) for rows in pt_rows.values())
out["i3"] = dict(meta=meta_rows, networks=pt_rows, all_consistent=ok3)
print("i3 meta keys differing between seeds:", {n: v["keys_differing_between_seeds"] for n, v in meta_rows.items()}, "| config as expected:",
      all(all(v["config_as_expected"]) for v in meta_rows.values()), "| networks consistent:", ok3)
print("   torch threads / versions of the networks:", {n: sorted({(r["torch_threads"], r["torch_version"]) for r in rows.values()}) for n, rows in pt_rows.items()})
print("   best updates:", {n: [r["best_update"] for r in rows.values()] for n, rows in pt_rows.items()})

# ---- i4: seed averages ----------------------------------------------------------------------------------------------------------------------------------
ref = load_npz("lstm_wq")
for n in NEW + CTRL:
    acc, obs_eq, nan_pat = None, True, []
    for s in seeds_of(n):
        z = load_npz(f"{n}_s{s}")
        acc = z["pred"].copy() if acc is None else acc + z["pred"]
        obs_eq &= bool(np.array_equal(z["obs"], ref["obs"], equal_nan=True)) and z["basins"] == ref["basins"] and np.array_equal(z["dates"], ref["dates"])
        del z
    a = load_npz(n)
    mean = acc / len(seeds_of(n))
    out["i4"][n] = dict(n_seeds=len(seeds_of(n)), bit_for_bit=bool(np.array_equal(a["pred"], mean, equal_nan=True)), max_abs_diff=float(np.nanmax(np.abs(a["pred"] - mean))),
                        obs_basins_dates_equal_lstm_wq=obs_eq and bool(np.array_equal(a["obs"], ref["obs"], equal_nan=True)), n_basins=len(a["basins"]),
                        nan_values=int(np.isnan(a["pred"]).sum()), shape=list(a["pred"].shape))
    del a, acc, mean
print("i4 seed averages:", {n: (v["n_seeds"], v["bit_for_bit"], v["max_abs_diff"]) for n, v in out["i4"].items()}, "| obs/basins/dates equal:", all(v["obs_basins_dates_equal_lstm_wq"] for v in out["i4"].values()))

# ---- i5: passes --------------------------------------------------------------------------------------------------------------------------------------
P = R["passes"]
n_pass = sum(p["n_trainings"] for p in P)
secs_net = sum(t["seconds"] for t in R["training"])
out["i5"] = dict(n_passes=len(P), n_trainings_in_passes=n_pass, n_trainings_on_disk=len(R["training"]), n_trainings_top=R["n_trainings"],
                 training_seconds_top=R["training_seconds"], training_seconds_sum_of_networks=round(secs_net, 1),
                 training_seconds_sum_of_passes=round(sum(p["training_seconds"] for p in P), 1),
                 runtime_s_top=R["runtime_s"], runtime_s_sum_non_rescore=round(sum(p["runtime_s"] for p in P if p["what"] != "rescore"), 1),
                 waited_top=R["waited_for_cpu_s"], waited_sum=round(sum(p["waited_for_cpu_s"] for p in P), 1),
                 last_pass=P[-1], three_seed_passes_carried_over=P[:len(B["passes"])] == B["passes"],
                 trainings_of_the_last_pass_seconds=round(sum(t["seconds"] for t in R["training"] if t["seed"] in (3, 4) and not (t["set"] == "lstm63_wq" and t["seed"] == 3)), 1),
                 killed_run_training=dict(set="lstm63_wq", seed=3, seconds=[t["seconds"] for t in R["training"] if t["set"] == "lstm63_wq" and t["seed"] == 3][0],
                                          pred_mtime=mtime["results/preds/lstm63_wq_s3.npz"], model_mtime=mtime["results/models/lstm63_wq_s3.pt"]),
                 recorded_in_passes=False)
print("i5 passes:", len(P), "| trainings in passes", n_pass, "of", len(R["training"]), "| training_seconds top", R["training_seconds"], "= networks", round(secs_net, 1),
      "| passes sum", out["i5"]["training_seconds_sum_of_passes"], "| runtime", R["runtime_s"], "=", out["i5"]["runtime_s_sum_non_rescore"],
      "| last pass seconds", P[-1]["training_seconds"], "= 9 rerun trainings", out["i5"]["trainings_of_the_last_pass_seconds"], "| 3-seed passes carried:", out["i5"]["three_seed_passes_carried_over"])

# ---- i6: integrity bookkeeping -------------------------------------------------------------------------------------------------------------------------
I = R["integrity"]
rerun = [(n, s) for n in NEW for s in (3, 4) if not (n == "lstm63_wq" and s == 3)]
exp_made = sorted([f"results/preds/{n}_s{s}.npz" for n, s in rerun]) + sorted([f"results/models/{n}_s{s}.pt" for n, s in rerun])
made = I["made_in_this_pass"]
out["i6"] = dict(made_in_this_pass=made, made_contains_the_18_rerun_files=set(exp_made) <= set(made), made_contains_the_killed_run_files=[f for f in made if "lstm63_wq_s3" in f],
                 earlier_pass_compared=I["earlier_pass_compared"], files_compared_expected=len(SAVED) - len(set(made)),
                 lstm63_wq_s3_compared_against=("not in the three-seed file, so not compared" if "results/preds/lstm63_wq_s3.npz" not in OLD else "compared"))
print("i6 made_in_this_pass:", len(made), "files; holds the 18 rerun files:", out["i6"]["made_contains_the_18_rerun_files"], "| killed-run files listed:", out["i6"]["made_contains_the_killed_run_files"],
      "| compared", I["earlier_pass_compared"]["files_compared"], "expected", out["i6"]["files_compared_expected"], "all_equal", I["earlier_pass_compared"]["all_equal"])
save("r1_integrity", out)
