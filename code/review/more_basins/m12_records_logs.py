"""Check (g), part 2 and check (c), part 2: the training records, the logs, the timings, and what the builder saw before the equal-training arm was trained.

 1. results/logs/17_more_basins.log: every "training k of N" line (set, seed, rivers, best update of updates run, seconds, validation NSE of the training rivers and of the 12) against the `training`
    records of results/more_basins.json and against the history stored inside each network file; the learning curves of the json against the network files; the sum of the seconds;
 2. the timeline: the modification time of every network file against the start of the first run (18:50:56) and the cumulative training seconds of the log; the first run's wait for the CPU lock and what
    `waited_for_cpu_s` and `passes` of the json do and do not hold;
 3. the interim pass (it scored the test years of the four non-equal sets at 20:13, before the equal-training arm was trained): its json (kept in the builder's scratch folder, if it still exists) and its log
    (results/logs/17_more_basins_interim_noequal.log) against the final results for the same four sets and the stored controls: were they changed afterwards?; and the diff of the script version that was
    running (a copy the builder made at 19:35) against the script on disk: which lines changed after the interim scores were seen, and does any touch the equal-training arm's design (EQ, conf_eq, the pairs)?
 4. the numbers quoted in ROADMAP.md (section 4, "Step 1d result") against the json.
Run: python m12_records_logs.py [FOLDER_WITH_interim_more_basins.json_AND_17_more_basins.py.running_version]   (default: evidence_from_builder_scratch/ next to this script)
"""
import difflib
import re
import sys
import time

import torch

from m_common import *

J = json.loads((PROJ / "results" / "more_basins.json").read_text())
LOG = (PROJ / "results" / "logs" / "17_more_basins.log").read_text().splitlines()
HERE_EVIDENCE = pathlib.Path(__file__).resolve().parent / "evidence_from_builder_scratch"      # copies of two files the builder kept in its session scratch folder (the interim results of 20:13, the script version of 19:35)
SCRATCH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE_EVIDENCE if HERE_EVIDENCE.exists() else pathlib.Path(__file__).resolve().parent / "e3_scratch"
res = {}
SEEDS = (0, 1, 2)
NEW = ["lstm63_wq", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic", "lstm63eq_wq"]

# ---- 1. log lines against the records --------------------------------------------------------------------------------------------------------------
pat = re.compile(r"training (\d+) of (\d+) \| (\w+) seed (\d) \| (\d+) rivers \| best at update (\d+) of (\d+), (\d+) s \| validation NSE of the (\d+) training rivers ([0-9.]+), of the 12 ([0-9.]+)")
rows = [pat.search(ln) for ln in LOG]
rows = [m.groups() for m in rows if m]
log_rec = {(g[2], int(g[3])): dict(k=int(g[0]), n=int(g[1]), rivers=int(g[4]), best=int(g[5]), run=int(g[6]), sec=int(g[7]), val=float(g[9]), val12=float(g[10])) for g in rows}
train = {(t["set"], t["seed"]): t for t in J["training"]}
bad, devs = [], dict(seconds=0.0, val=0.0, val12=0.0)
hist_bad, curve_bad, n_ok = [], [], 0
for key, l in log_rec.items():
    t = train[key]
    z = torch.load(PROJ / "results" / "models" / f"{key[0]}_s{key[1]}.pt", map_location="cpu", weights_only=True)
    ok = (l["best"] == t["best_update"] == z["best_update"] and l["run"] == t["updates_run"] == z["updates_run"] and l["rivers"] == t["n_training_rivers"] == len(z["ids"]))
    devs["seconds"] = max(devs["seconds"], abs(l["sec"] - t["seconds"]), abs(t["seconds"] - z["seconds"]))
    devs["val"] = max(devs["val"], abs(l["val"] - round(t["best_val_nse"], 4)), abs(t["best_val_nse"] - z["best_val_nse"]))
    devs["val12"] = max(devs["val12"], abs(l["val12"] - round(t["validation_nse_of_the_12_at_that_update"], 4)))
    # learning curves: the json's curve and history against the network file
    cj = [[u, v] for u, v in t["curve"]]
    cz = [[h["update"], round(h["val_nse"], 4)] for h in z["history"]]
    if cj != cz:
        curve_bad.append(key)
    if abs(t["mean_load_average"] - float(np.mean([h.get("load", float("nan")) for h in z["history"]]))) > 1e-9:
        hist_bad.append(key)
    bad += [] if ok else [key]
    n_ok += ok
res["log_vs_records"] = dict(n_training_lines_in_log=len(log_rec), expected=15, all_sets_and_seeds_present=sorted(log_rec) == sorted((n, s) for n in NEW for s in SEEDS), n_consistent=n_ok, inconsistent=bad,
                             max_abs_diff=dict(seconds_log_vs_json_vs_file=devs["seconds"], val_nse_training_rivers=devs["val"], val_nse_of_the_12=devs["val12"]), curves_equal_network_files=not curve_bad,
                             mean_load_average_equals_network_files=not hist_bad, curve_mismatch=curve_bad, history_mismatch=hist_bad,
                             plan_in_first_line=LOG[0], k_of_n=sorted({(l["k"], l["n"]) for l in log_rec.values()}))
print("1. training lines in the log:", len(log_rec), "| consistent with the json and the network files:", n_ok, "| max abs diffs:", res["log_vs_records"]["max_abs_diff"],
      "| curves equal the network files:", not curve_bad, "| mean load averages equal:", not hist_bad)
print("   first log line:", LOG[0])
secs = sum(t["seconds"] for t in J["training"])
res["seconds"] = dict(sum_of_the_15_trainings=secs, json_training_seconds=J["training_seconds"], equal=abs(secs - J["training_seconds"]) < 0.06, minutes=secs / 60,
                      first_12_trainings_minutes=sum(t["seconds"] for t in J["training"] if t["set"] != "lstm63eq_wq") / 60, equal_arm_minutes=sum(t["seconds"] for t in J["training"] if t["set"] == "lstm63eq_wq") / 60)
print(f"   sum of the training seconds {secs:.1f} s = {secs / 60:.1f} min (json: {J['training_seconds']}); first twelve {res['seconds']['first_12_trainings_minutes']:.1f} min, equal arm {res['seconds']['equal_arm_minutes']:.1f} min")
six = sum(t["seconds"] for t in J["training"] if t["set"] in ("lstm63_wq", "lstm63_w")) / 60
res["scoping_estimate"] = dict(roadmap_says="six trainings take about 18 minutes against 16 for the 12 alone (step 1d, written before the run)", measured_six_63_river_trainings_minutes=six,
                               mean_load_average_at_the_checks=float(np.mean([t["mean_load_average"] for t in J["training"]])))
print(f"   the six 63-river trainings of lstm63_wq and lstm63_w took {six:.1f} min against the roadmap's scoping estimate of about 18 min (mean load average {res['scoping_estimate']['mean_load_average_at_the_checks']:.1f}, threads {sorted({t['torch_threads'] for t in J['training']})})")

# ---- 2. timeline -------------------------------------------------------------------------------------------------------------------------------------
t0 = time.mktime(time.strptime("2026-10-07 18:50:56", "%Y-%m-%d %H:%M:%S"))
order = sorted(log_rec, key=lambda k: log_rec[k]["k"] if k[0] != "lstm63eq_wq" else 100 + log_rec[k]["k"])
tl, cum = [], 0.0
for key in order[:12]:
    cum += log_rec[key]["sec"]
    mt = (PROJ / "results" / "models" / f"{key[0]}_s{key[1]}.pt").stat().st_mtime
    tl.append(dict(set=key[0], seed=key[1], saved_at=time.strftime("%H:%M:%S", time.localtime(mt)), minutes_since_start=(mt - t0) / 60, cumulative_training_minutes=cum / 60))
res["timeline_first_run"] = tl
print("2. first run: network file times against the start (18:50:56) and the cumulative training minutes:")
for r in tl:
    print(f"   {r['set']:20s} s{r['seed']} saved {r['saved_at']}  {r['minutes_since_start']:6.1f} min after the start; cumulative training {r['cumulative_training_minutes']:6.1f} min")
lock = [ln for ln in LOG if "waiting for the CPU" in ln]
res["cpu_lock"] = dict(log_line=lock[0] if lock else None, waited_s_by_the_log_annotation=(time.mktime(time.strptime("2026-10-07 20:44:29", "%Y-%m-%d %H:%M:%S")) - time.mktime(time.strptime("2026-10-07 20:10:44", "%Y-%m-%d %H:%M:%S"))),
                       json_waited_for_cpu_s=J["waited_for_cpu_s"], json_note_about_missing_passes=[n for n in J["notes"] if "passes accounts" in n][:1])
print(f"   the first run waited {res['cpu_lock']['waited_s_by_the_log_annotation'] / 60:.1f} min for the CPU lock (log annotation); json waited_for_cpu_s = {J['waited_for_cpu_s']}; its note says the first run's time is only in training_seconds")
res["passes"] = J["passes"]
print("   passes in the json:", [(p["what"][:12], p["n_trainings"], p["finished_local_time"]) for p in J["passes"]])

# ---- 3. the interim pass and the script versions --------------------------------------------------------------------------------------------------------
interim = SCRATCH / "interim_more_basins.json"
res["interim"] = dict(scratch_folder_exists=SCRATCH.exists(), interim_json_exists=interim.exists())
if interim.exists():
    I = json.loads(interim.read_text())
    cmp, dif = 0, []
    for n in ("lstm63_wq", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic", "lstm_wq", "lstm_w", "lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w", "lstm_wqf"):
        for k in ("l1", "l2", "l3"):
            a, b = I["summary_test"]["sets"][n][k], J["summary_test"]["sets"][n][k]
            cmp += 1
            if a != b:
                dif.append((n, k))
    res["interim"].update(summary_test_cells_compared=cmp, summary_test_cells_different=dif, interim_has_equal_arm="lstm63eq_wq" in I["summary_test"]["sets"], interim_written=time.strftime("%H:%M:%S", time.localtime(interim.stat().st_mtime)),
                          interim_gain_rule_keys=list(I["gain_rule"]), interim_outcome=I["outcome"], final_outcome=J["outcome"], outcome_same=I["outcome"] == J["outcome"])
    res["interim"]["seed_ranges_equal"] = all(I["seed_ranges"]["sets"][n] == J["seed_ranges"]["sets"][n] for n in ("lstm63_wq", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic"))
    res["interim"]["gain_rule_equal_for_the_main_pairs_that_exist_in_both"] = all(I["gain_rule"][k] == J["gain_rule"][k] for k in I["gain_rule"] if k in J["gain_rule"])
    res["interim"]["unseen_gain_rule_equal"] = I["unseen"]["against_the_leave_one_out_models"] == J["unseen"]["against_the_leave_one_out_models"]
    print(f"3. interim json of 20:13 (the builder's scratch): {cmp} summary cells of the four sets and seven controls compared with the final file, different: {dif}; seed ranges equal: "
          f"{res['interim']['seed_ranges_equal']}; gain rule of the shared pairs equal: {res['interim']['gain_rule_equal_for_the_main_pairs_that_exist_in_both']}; unseen gain rule equal: {res['interim']['unseen_gain_rule_equal']}; "
          f"equal arm in the interim file: {res['interim']['interim_has_equal_arm']}; outcome sentence equal: {res['interim']['outcome_same']}")
run_v = SCRATCH / "17_more_basins.py.running_version"
now_v = PROJ / "code" / "17_more_basins.py"
if run_v.exists():
    a, b = run_v.read_text().splitlines(), now_v.read_text().splitlines()
    d = [ln for ln in difflib.unified_diff(a, b, "running version (copy of 19:35)", "script now", lineterm="", n=0) if not ln.startswith(("---", "+++", "@@"))]
    touch = [ln for ln in d if re.search(r"\bEQ\b|conf_eq|lstm63eq|PAIRS_MAIN|SEEDS|CTRL_SEEDS|conf\b|P_CUT|rule\(|LEVELS", ln) and not ln.startswith("-")]
    sa, sb = run_v.read_text(), now_v.read_text()
    pick = lambda s: [m.group(0) for m in (re.search(r"^EQ = .*$", s, re.M), re.search(r"^conf_eq = .*$", s, re.M), re.search(r'"lstm63eq_wq": dict\(stage=3.*', s), re.search(r"^P_CUT = .*$", s, re.M), re.search(r"^SEEDS = .*$", s, re.M),
                                  re.search(r"^NESTED, NESTED_IN = .*$", s, re.M), re.search(r"^LEVELS = .*$", s, re.M)) if m]
    res["script_versions"] = dict(running_version_bytes=len(sa), now_bytes=len(sb), n_changed_lines=len([x for x in d if x.startswith(('+', '-'))]), design_lines_equal=pick(sa) == pick(sb), design_lines=pick(sb),
                                  rule_function_equal=re.search(r"def rule\(T, C, k\):.*?(?=\nPAIRS_MAIN)", sa, re.S).group(0) == re.search(r"def rule\(T, C, k\):.*?(?=\nPAIRS_MAIN)", sb, re.S).group(0),
                                  changed_lines_that_mention_the_equal_arm_or_settings=touch)
    print(f"   script on disk against the running version of 19:35: {res['script_versions']['n_changed_lines']} changed lines; the design lines (EQ, conf_eq, the stage-3 set, P_CUT, SEEDS, the nested gauge, LEVELS) equal: "
          f"{res['script_versions']['design_lines_equal']}; the rule function equal: {res['script_versions']['rule_function_equal']}")
    print("   changed lines that mention the equal arm or the settings (all lines about the --no-equal switch, EQUAL guards and wording):")
    for ln in touch[:12]:
        print("     ", ln[:170])
fb = LOG[[i for i, ln in enumerate(LOG) if ln.startswith("[2026-10-07 20:51]")][0]] if any(ln.startswith("[2026-10-07 20:51]") for ln in LOG) else None
res["log_annotations"] = [ln for ln in LOG if ln.startswith("[2026-")]
il = PROJ / "results" / "logs" / "17_more_basins_interim_noequal.log"
res["interim_log"] = dict(exists=il.exists(), modified=time.strftime("%H:%M:%S", time.localtime(il.stat().st_mtime)) if il.exists() else None, lines=len(il.read_text().splitlines()) if il.exists() else None)
print("   the interim log in the project:", res["interim_log"], "(copied there later: its file time is not the time of the pass)")

# ---- 4. ROADMAP.md numbers against the json ------------------------------------------------------------------------------------------------------------
R = (PROJ / "ROADMAP.md").read_text()
sec = R[R.index("### Step 1d result"):R.index("## 5. The satellite case")]
S, RG = J["summary_test"]["sets"], J["seed_ranges"]["sets"]
f2 = lambda x: f"{x:.2f}"
exp = {"lstm_wq": (f2(S["lstm_wq"]["l1"]["nse_median"]), f"{100 * S['lstm_wq']['l1']['onset']['pod']:.1f}%"), "lstm63_wq": (f2(S["lstm63_wq"]["l1"]["nse_median"]), f"{100 * S['lstm63_wq']['l1']['onset']['pod']:.1f}%"),
       "lstm63eq_wq": (f2(S["lstm63eq_wq"]["l1"]["nse_median"]), f"{100 * S['lstm63eq_wq']['l1']['onset']['pod']:.1f}%"), "lstm_w": (f2(S["lstm_w"]["l1"]["nse_median"]), f"{100 * S['lstm_w']['l1']['onset']['pod']:.1f}%"),
       "lstm63_w": (f2(S["lstm63_w"]["l1"]["nse_median"]), f"{100 * S['lstm63_w']['l1']['onset']['pod']:.1f}%"),
       "unseen50_w_static": f2(S["unseen50_w_static"]["l1"]["nse_median"]), "unseen50_w_nostatic": f2(S["unseen50_w_nostatic"]["l1"]["nse_median"]), "lobo_w_static": f2(S["lobo_w_static"]["l1"]["nse_median"]), "lobo_w_nostatic": f2(S["lobo_w_nostatic"]["l1"]["nse_median"])}
found = {}
for n, v in exp.items():
    vals = (v,) if isinstance(v, str) else v
    found[n] = {x: (x in sec) for x in vals}
res["roadmap_numbers_present"] = found
print("\n4. ROADMAP step 1d numbers present in its text (rounded as the json rounds):", {n: all(v.values()) for n, v in found.items()})
claims = {"loss 0.13 -> about 0.02": (round(J["unseen"]["loss_stored_by_05_lobo"]["with_properties"]["l1"], 2), round(J["unseen"]["loss_from_not_having_seen_the_river"]["50_rivers_with_properties__against_12_river_seed_average"]["l1"]["median_loss"], 3)),
          "0.43 against 0.44 (never seen, no properties, against the 12-river WITH-properties model lstm_w)": (round(S["unseen50_w_nostatic"]["l1"]["nse_median"], 3), round(S["lstm_w"]["l1"]["nse_median"], 3)),
          "like-for-like in-training partner without properties (lobo_ref_w_nostatic, one training)": round(S["lobo_ref_w_nostatic"]["l1"]["nse_median"], 3)}
res["roadmap_claims_checked"] = claims
print("   ", claims)
save("m12_records_logs", res)
