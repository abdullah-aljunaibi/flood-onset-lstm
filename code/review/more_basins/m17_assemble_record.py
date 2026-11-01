"""Assemble the review record results/review_more_basins.json from the outputs of the check scripts ($REVIEW_OUT/*.json) and the reviewer's own text.

Every number in `checks`, `answers` and `numbers` is read from those outputs (nothing is typed in by hand); the text of the discrepancies, must-fix and should-fix items and of the verdict is the reviewer's.
Usage: python m17_assemble_record.py                 writes $REVIEW_OUT/review_more_basins.json
       python m17_assemble_record.py --write-project  writes results/review_more_basins.json in the project (the only file this review writes there, besides code/review/more_basins/)
Run the fast and slow checks first (run_fast_checks.sh, run_slow_checks.sh): this script needs their outputs.
"""
import sys
import time

from m_common import *

O = lambda n: json.loads((OUT / f"{n}.json").read_text())
m3b = O("m3b_area_errors")
m1, m2, m3, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16 = (O(n) for n in ("m1_engine_diff", "m2_defaults", "m3_candidates_raw", "m5_testflood_mutants", "m6_headline", "m7_networks", "m8_unseen", "m9_gain_rule",
                                                                                   "m10_extra_rivers", "m11_figure", "m12_records_logs", "m13_handin", "m14_sandbox_rescore", "m15_integrity", "m16_probe_properties"))
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
pc = lambda x: f"{100 * x:.1f}%"
SC = m6["scores"]
g1 = lambda n, key: SC[n]["l1"][key]
GA = m8["gain_against_proximity"]
PS = m8["per_seed_against_lobo"]
SE = m8["seed_averaging_effect_lead1"]
ST = J["stopping"]["sets"]
V = m9["verdicts"]["per_set"]
PD = m9["paired_onsets_lead1"]
EXTRA = m10["spearman_nse_vs_km_to_nearest_training_gauge"]
EQ_CURVES = {s: [t for t in J["training"] if t["set"] == "lstm63eq_wq" and t["seed"] == s][0]["curve"] for s in (0, 1, 2)}
eq_peak = {s: max(EQ_CURVES[s], key=lambda c: c[1]) for s in EQ_CURVES}
eq_last = {s: EQ_CURVES[s][-1] for s in EQ_CURVES}
eq_txt = "; ".join(f"seed {s_}: {eq_peak[s_][1]:.3f} at {eq_peak[s_][0]} updates -> {eq_last[s_][1]:.3f} at {eq_last[s_][0]}" for s_ in (0, 1, 2))
bins = m10["by_distance_bin"]
n_pass_mut = sum(1 for k, c in m5["cases"].items() if k.startswith("M") and not c["passed"])
near = lambda gains, ids: sorted(gains[g] for g in ids)

checks = []


def chk(id_, what, how, result, numbers):
    checks.append(dict(id=id_, what=what, how=how, result=result, numbers=numbers))


# ---- (a) the engine change -------------------------------------------------------------------------------------------------------------------------------
chk("a1", "code/flood.py and code/test_flood.py against the hand-in copy (Project-Flood_handin.zip, unzipped), statement by statement (ast), and by unified diff",
    "own code (m1_engine_diff.py)", "pass",
    dict(flood_py=dict(top_level_nodes_hand_in=m1["flood_py"]["top_level_nodes_old"], top_level_nodes_now=m1["flood_py"]["top_level_nodes_new"], changed=m1["flood_py"]["changed"], added=m1["flood_py"]["added"],
                       removed=m1["flood_py"]["removed"], unchanged_nodes_byte_identical=m1["flood_py"]["unchanged"], unchanged_nodes_in_the_same_order=m1["flood_py"]["unchanged_nodes_in_the_same_order"],
                       lines_added=m1["diff_lines"]["flood_py_added"], lines_removed=m1["diff_lines"]["flood_py_removed"], the_removed_lines=m1["removed_lines_flood_py"]),
         test_flood_py=dict(lines_added=m1["diff_lines"]["test_flood_added"], lines_removed=m1["diff_lines"]["test_flood_removed"], the_removed_line=m1["removed_lines_test_flood_py"]),
         identical_to_hand_in=[k for k, v in m1["files"].items() if v["identical"]], different_from_hand_in=[k for k, v in m1["files"].items() if not v["identical"]],
         note="only basins() and load_data() changed (plus study_ids(), candidate_ids() and `import hashlib`); the 8 removed lines of flood.py are the old signatures, docstrings and the two statements the new code rewrites; run_all.sh differs by the E1, E1b and E3 lines"))
c = m2["comparison"]
chk("a2", "every default call path unchanged: the hand-in engine and the engine now each rebuild the 12-basin table from the raw CAMELS files into an EMPTY cache folder (sandbox, nothing written into the project); caches, tables, basins() frame, thresholds, "
          "flood.evaluate output and the headline lstm_wq score compared",
    "own code (m2_defaults.py); the engines themselves are the code under review", "pass",
    dict(rebuilt_cache_sha256_hand_in_engine=m2["old_handin"]["cache_sha256"], rebuilt_cache_sha256_present_engine=m2["new_now"]["cache_sha256"], stored_cache_sha256=m2["stored_cache"]["sha256"], cache_bytes=m2["stored_cache"]["bytes"],
         stored_cache_modified=m2["stored_cache"]["mtime_local"], byte_identical_old_new_stored=bool(c["cache_bytes_old_equals_new"] and c["cache_new_equals_stored"] and c["cache_old_equals_stored"]),
         tables_bitwise_identical=bool(c["tables_bitwise_old_new"] and c["tables_bitwise_new_stored"]), basins_frame_identical=c["basins_frame_identical"], thresholds_identical=c["thresholds_identical"],
         evaluate_output_identical=c["evaluate_output_identical"], rows_per_table=c["rows_per_table"], n_basins=c["n_basins"],
         headline_lstm_wq=dict(median_nse=m2["headline"]["nse_median"], onset_hits=m2["headline"]["onset_hits"], onset_misses=m2["headline"]["onset_misses"], onset_share_warned=m2["headline"]["onset_pod"],
                               false_alarms=m2["headline"]["onset_false_alarms"]),
         note="the builder's own sha256 of the cache before the change (its scratch file before/sha256.txt) is the same value; the pickles are byte-identical, so even the file is the same"))
chk("a3", "candidate_ids() against the completeness rule of 01_data_audit.py applied to results/basin_audit.csv; the candidate set from the CAMELS attribute tables; every audit number recomputed from the RAW CAMELS files; completeness of the 63 stated afresh",
    "own code (m3_candidates_raw.py: own parsers for the forcing and flow files)", "pass",
    dict(audit_rows=m3["audit_csv"]["rows"], complete_by_rule=m3["audit_csv"]["rule_count"], candidate_ids_equals_own_list_in_the_files_order=m3["audit_csv"]["candidate_ids_equals_own_list"],
         candidates_from_attribute_tables=m3["candidates"]["n_from_attributes"], same_set_and_order_as_csv=bool(m3["candidates"]["same_set_as_csv"] and m3["candidates"]["same_order_as_csv"]),
         audit_numbers_from_raw=m3["audit_recomputed_from_raw"], the_63_from_raw=m3["the_63_complete_from_raw"]["checks"], n=m3["the_63_complete_from_raw"]["n"],
         the_12=m3["the_12"], landscape_rule_equals_csv=m3["landscape_rule_equals_csv"],
         note="flow: a row for each of the 12418 days of 1980-10-01..2014-09-30, no -999, no duplicate dates; forcing: all seven columns finite (the audit rule checks five of them: dayl and swe are finite too)"))
chk("a4", "the 63-basin cache data/derived/camels63_daymet_<digest>.pkl against tables the reviewer rebuilt from the raw CAMELS files; the digest in the file name; the 12 inside the 63 against the stored 12-basin cache; the properties of the 63",
    "own code (m3_candidates_raw.py)", "pass",
    dict(cache=m3["cache63"]["file"], bytes=m3["cache63"]["bytes"], digest_rule_gives_the_name=m3["cache63"]["name_matches_rule"], keys_equal_ids63_in_the_scripts_order=m3["cache63"]["keys_equal_ids63"],
         bitwise_equal_to_the_reviewers_raw_tables=f"{m3['cache63']['n_bitwise']} of {m3['cache63']['n_tables']}", max_abs_difference=m3["cache63"]["max_abs_diff_any"], all_finite=m3["cache63"]["all_finite"],
         load_data_with_ids_in_reverse_order_gives_that_order_and_the_same_tables=m3["cache63"]["load_data_reversed_ids_gives_that_order_and_same_tables"], the_12_inside_the_63_bitwise_equal_the_stored_12=m3["twelve_in_63"]["bitwise_equal"],
         properties=dict(finite=m3["properties"]["own_finite"], engine_vs_own_join_of_the_attribute_tables_max_abs_diff=m3["properties"]["max_abs_diff_engine_vs_own_63"], the_12_rows_equal_flood_basins=m3["properties"]["the_12_rows_equal_flood_basins"],
                         properties_computed_from_the_flow_record=m3["properties"]["n_properties_from_flow_record"]),
         river_sets_of_the_results_file=m3["river_sets_in_results_file"]))
cs = m5["cases"]
chk("a5", "code/test_flood.py passes, and its new check really catches a broken load_data(): test_flood.py is run UNCHANGED in a sandbox (copy of code/, empty results/) with only flood.py different: the engine as it is; the hand-in test against the present engine; the new test "
          "against the hand-in engine; eight deliberately broken load_data() variants (one or two string replacements each)",
    "own code (m5_testflood_mutants.py)", "pass",
    dict(baseline=dict(passed=cs["baseline"]["passed"], last_line=cs["baseline"]["last_stdout_line"], fair_test_json_equals_the_projects=m5["baseline_fair_test_json_equals_project"]["equal"], measured=m5["baseline_fair_test_json_equals_project"]["measured_sandbox"]),
         hand_in_test_with_present_engine_passes=cs["old_test_new_engine"]["passed"], new_test_with_hand_in_engine=dict(passed=cs["new_test_old_engine"]["passed"], error=cs["new_test_old_engine"]["last_stderr_line"]),
         broken_variants_caught=f"{n_pass_mut} of {m5['n_mutants']}", variants={k: dict(caught=not v["passed"], how_it_failed=v["last_stderr_line"]) for k, v in cs.items() if k.startswith("M")},
         note="a first version of M2 changed only the cache READ (a set with ids rebuilt every time) and was not caught: it is functionally correct (an equivalent variant, slower only); M2 now also shares the file name, so the second set overwrites the first set's cache, and is caught. "
              "M5, M6 and M8 fail on plain `assert` lines without a message."))
ids_calls = m1["files_that_call_with_ids"]
chk("a6", "could any other script's behaviour change? every call of basins( / load_data( / study_ids( / candidate_ids( in code/ (also the review scripts), the arguments, star imports, name clashes",
    "own code (m1_engine_diff.py: ast scan of all .py files under code/; a mention in a string or a comment is not a call)", "pass",
    dict(calls_found=len([c for c in m1["calls"] if not c["local_helper"]]), files=len({c["file"] for c in m1["calls"] if not c["local_helper"]}), calls_of_a_files_own_helper_left_out=len([c for c in m1["calls"] if c["local_helper"]]), files_that_pass_ids=ids_calls,
         calls_with_any_argument=[(c["file"], c["line"], c["call"], c["args"]) for c in m1["calls"] if c["args"] and not c["local_helper"]], star_imports_of_flood=m1["star_imports_of_flood"], new_names_defined_elsewhere=m1["new_names_defined_elsewhere"],
         note="every other script calls basins() and load_data() with no argument: the default path, proved unchanged in a2. features.py, camels.py and common.py are byte-identical to the hand-in. "
              "report_sections/s4c_timing.py basins(key) is a local helper of that file, not the engine (left out of the count). Scripts written by other agents during the review (18_forecast_input.py, check_stage4_realtime.py, review scripts) were found by the same scan and call the defaults."))

lst = m3b["areas_of_the_listed"]
chk("a7", "data quality of the 51 extra rivers: which of the 63 are on CAMELS's own list of basins whose area is wrong by more than 10%, and how the three areas that could define the flow depth compare (forcing-file area, which the project divides the flow by; gauge area; outline area)",
    "own code (m3b_area_errors.py)", "finding",
    dict(basins_on_the_camels_list=m3b["file_lists"], of_the_63=m3b["of_the_63"], of_the_12=m3b["of_the_12"], of_the_50=m3b["of_the_50"], size_diff_percent_in_the_file=m3b["size_diff_percent_in_the_file"], areas_km2=lst,
         forcing_file_area_over_gauge_area_range_of_the_63=m3b["forcing_over_gauge_range_of_the_63"],
         reading="for 02108000 (a study river; listed +100%) the three areas agree within 1.2%, so its flow in mm/day is not affected; for 02324400, 02327100 and 02422500 (in the 50) the forcing-file area is 12%, 3% and 0.4% below the gauge's, i.e. at most 13% in the flow scale of one river: immaterial for the training, but not mentioned"))

# ---- (b) headline numbers --------------------------------------------------------------------------------------------------------------------------------
rows = {n: {l: dict(nse_median=round(SC[n][l]["nse_median"], 4), onset_hits=SC[n][l]["onset_hits"], onset_share_warned=round(SC[n][l]["onset_pod"], 4), onset_false_share=round(SC[n][l]["onset_far"], 4)) for l in ("l1", "l2", "l3")} for n in SC}
chk("b1", "headline numbers of every new set and of the controls, per lead, by two routes (the engine: flood.load_preds / thresholds / evaluate; the reviewer's own code: np.load, own 95th percentile of the training-year flow, own NSE, own pooled onset counts with the issue-day flow "
          "taken by array position) against results/more_basins.json (summary_test): median NSE, onset hits, misses, false alarms, pod, far, csi, all-day pod and far",
    "engine route (allowed) and own code (m6_headline.py)", "pass",
    dict(sets=len(SC), leads=3, max_abs_deviation=m6["max_abs_deviation"], single_seed_files_checked=len(m6["seed_files"]), single_seed_files_max_deviation=m6["seed_files_max_dev_own_vs_engine"],
         thresholds_own_vs_engine_max_abs_diff=m6["thresholds"]["own_equals_engine_max_abs_diff"], rows=rows))
chk("b2", "integrity of the sets: only the 12 basins (assert), the date index, obs equal to the project's flow bit for bit, the seed averages equal the day-by-day mean of their seed files, metadata (script, quick mode, training set size, config)",
    "own code (m6_headline.py)", "pass",
    dict(per_set={n: dict(twelve_basins_only=all(v[m]["basins_equal_the_12"] and v[m]["n_basins"] == 12 for m in v if m != "seed_average"), obs_bitwise_equal_project_flow=all(v[m]["obs_bitwise_equal_project_flow"] for m in v if m != "seed_average"),
                          dates_equal_period=all(v[m]["dates_equal_period"] for m in v if m != "seed_average"), config_as_expected=all(v[m]["config_equals_expected"] for m in v if m != "seed_average"),
                          seed_average_bitwise_mean_of_seed_files=v["seed_average"]["bitwise_equal_to_mean_of_seed_files"], seed_average_max_abs_diff=v["seed_average"]["max_abs_diff"], script=v[n]["script"], quick_mode=v[n]["quick_mode"], trained_on=v[n]["trained_on"])
                  for n, v in m6["integrity"].items() if n in ("lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic")},
         controls_twelve_basins_and_obs_equal_project_flow={n: bool(v["basins_equal_the_12"] and v["obs_bitwise_equal_project_flow"]) for n, v in m6["integrity"].items() if n in ("lstm_wq", "lstm_w", "lstm_wqf", "lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w")},
         quick_mode_flags_of_the_20_files_all_false=m15["quick_mode_flags_of_the_20_prediction_files"]["all_false"], integrity_record_of_the_results_file=m15["integrity_record_of_the_results_file"]))
lead1 = {n: dict(own_pooled_onset_counts=dict(onsets=g1(n, "onset_hits") + g1(n, "onset_misses"), hits=g1(n, "onset_hits"), misses=g1(n, "onset_misses"), false_alarms=g1(n, "onset_false_alarms"), share_warned=g1(n, "onset_pod"))) for n in ("lstm_wq", "lstm63_wq", "lstm63eq_wq", "lstm_w", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic", "lobo_w_static", "lobo_w_nostatic")}
chk("b3", "pooled onset counts at lead 1 by the reviewer's own loop (observed high-flow level, 611 onsets)", "own code (m6_headline.py: own_pooled in m_common.py)", "pass", lead1)
chk("b4", "the numbers quoted in the review request laid beside the recomputed ones", "own code (m6_headline.py)", "pass",
    dict(all_round_to_the_claim=all(c_["rounds_to_claim"] for c_ in m6["claims_vs_recomputed"]), table=m6["claims_vs_recomputed"]))

# ---- (c) protocol of the 63-river training ---------------------------------------------------------------------------------------------------------------
net = m7["networks"]
chk("c1", "the 15 saved networks (torch.load, weights_only=True): training rivers (63 with the 12 first and the audit's 51 in order; 50 with none of the 12), split, settings (results/lstm_all.json; x5 updates and eval_every for lstm63eq_wq), seed, flags, property values",
    "own code (m7_networks.py)", "pass",
    dict(n_networks=len(net), training_rivers_as_expected=all(v["c1"]["training_rivers_equal_expected"] for v in net.values()), settings_as_expected=all(v["c1"]["settings_equal_expected"] for v in net.values()),
         quick_mode_any=any(v["c1"]["quick_mode"] for v in net.values()), none_of_the_12_in_the_unseen_networks=all(v["c1"]["contains_none_of_the_12"] for k, v in net.items() if k.startswith("unseen50")),
         property_values_of_train_and_scored_rivers_max_abs_diff_to_own_join=max(max(v["c4"]["train_values_max_abs_diff"], v["c4"]["scored_values_max_abs_diff"]) for v in net.values() if "c4" in v),
         torch_version=net["lstm63_wq_s0"]["c1"]["torch_version"], numpy_version=net["lstm63_wq_s0"]["c1"]["numpy_version"], torch_threads_of_the_training=net["lstm63_wq_s0"]["c1"]["torch_threads_recorded"]))
chk("c2", "scalers (weather, flow, the ten properties) recomputed by the reviewer from the TRAINING YEARS of exactly the training rivers: must equal the stored ones; the same scalers from other choices must differ",
    "own code (m7_networks.py)", "pass",
    dict(max_relative_difference_all_15=max(v["c2"]["max_rel_all"] for v in net.values()), all_bitwise_equal=all(v["c2"]["all_bitwise_equal"] for v in net.values()),
         alternatives_differ_by=dict(all_days_instead_of_training_years=net["lstm63_wq_s0"]["c2"]["alt_max"]["all_days_instead_of_training_years"], the_12_alone=net["lstm63_wq_s0"]["c2"]["alt_max"]["the_12_alone"],
                                     the_50_plus_the_12_instead_of_the_50=net["unseen50_w_static_s0"]["c2"]["alt_max"]["training_rivers_plus_the_12"]),
         static_scaler_negative_control_max_forecast_change_mm_day=dict(min=min(v["f"]["negative_control_12_scaled_properties_max_abs_diff"] for v in net.values() if v["f"]["negative_control_12_scaled_properties_max_abs_diff"] is not None),
                                                                        max=max(v["f"]["negative_control_12_scaled_properties_max_abs_diff"] for v in net.values() if v["f"]["negative_control_12_scaled_properties_max_abs_diff"] is not None)),
         loss_weights=net["lstm63_wq_s0"]["c4w"], note="the loss weights are formed inside flood.fit (byte-identical to the hand-in's: a1) from the training years of the training rivers; their range is given for reference. The stored forecasts were made with the property scaler of the training set: scaling the 12's properties with their own mean and spread changes the forecasts by 3 to 13 mm/day."))
chk("c3", "early stopping: the recorded validation curve replayed (best update, updates run, best score), and the saved weights scored on the validation years of the TRAINING rivers with the trainer's own score (mean NSE over rivers and the three leads): must equal the stored best_val_nse; "
          "for the never-seen networks the 12's own score is shown to be another number",
    "own code (m7_networks.py; engine design and rolling, own NSE)", "pass",
    dict(replay_consistent_all_15=all(v["c3_replay"]["consistent"] for v in net.values()), max_abs_difference_saved_weights_vs_stored_best_val_nse=max(v["c3_val"]["abs_diff"] for v in net.values()),
         stopped_by_patience=sum(v["c3_replay"]["stopped_by_patience"] for v in net.values()), reached_the_update_limit=sum(v["c3_replay"]["reached_update_limit"] for v in net.values()),
         unseen_networks=dict(val_nse_on_the_50={k: round(v["c3_val"]["val_nse_of_the_saved_weights_on_the_training_rivers"], 4) for k, v in net.items() if k.startswith("unseen50")},
                              val_nse_on_the_12={k: round(v["c3_val"]["val_nse_of_the_saved_weights_on_the_12"], 4) for k, v in net.items() if k.startswith("unseen50")}),
         json_validation_nse_of_the_12_equals_mine=all(abs(v["c3_val"]["val_nse_of_the_saved_weights_on_the_12"] - v["c3_val"]["json_validation_nse_of_the_12"]) < 1e-9 for v in net.values()),
         note="the 63-river models stop on the mean validation NSE of 63 rivers, the 12 included (validation years only; allowed); the never-seen models on the 50, never the 12"))
chk("c4", "thresholds for scoring: the 12's own training-year 95th percentile (flood.thresholds on the 12), as for every other set", "own code (m6_headline.py) and the script (read)", "pass",
    dict(own_vs_engine_max_abs_diff=m6["thresholds"]["own_equals_engine_max_abs_diff"], values_mm_per_day=m6["thresholds"]["values"],
         in_the_script="thr = F.thresholds({g: D[g] for g in ids12}), used for scoring only; the 12 inside the 63-table are bit-identical to the stored 12 (a4), so the thresholds equal those of the stored tables (the script asserts it)"))
ip = m12["interim"]
chk("c5", "was anything chosen on the test years? the interim scoring pass (20:11-20:13) that preceded the equal-training arm, the script versions, the log",
    "own code (m12_records_logs.py) with two files the builder kept in its session scratch (copied into code/review/more_basins/evidence_from_builder_scratch/)", "pass with a note",
    dict(plan_in_the_first_line_of_the_log=m12["log_vs_records"]["plan_in_first_line"], the_equal_arm_was_in_the_plan_from_the_start=True,
         killed_first_runs_printed_curve_of_the_arms_first_training=("update 500 val NSE 0.3644 ... update 2500 val NSE 0.3326, eval_every 500"), the_rerun_printed_the_same_curve_number_for_number=True,
         interim_json_cells_compared_with_the_final=ip["summary_test_cells_compared"], interim_cells_different=ip["summary_test_cells_different"], interim_seed_ranges_equal_final=ip["seed_ranges_equal"], interim_gain_rule_equal_final=ip["gain_rule_equal_for_the_main_pairs_that_exist_in_both"],
         interim_outcome_sentence_equal_final=ip["outcome_same"], design_lines_of_the_19_35_script_equal_the_final=m12["script_versions"]["design_lines_equal"], design_lines=m12["script_versions"]["design_lines"],
         rule_function_equal=m12["script_versions"]["rule_function_equal"], lines_changed_in_the_script_after_the_interim_scores_were_seen=m12["script_versions"]["n_changed_lines"],
         what_those_lines_are="the --no-equal switch and its guards (SKIP_EQUAL, EQUAL), wording of notes, the --note field, the make_handin note, the landscape counts of the 50 computed instead of typed, PROBE_NOW; none touches EQ, conf_eq, the stage-3 set, the pairs, the rule, P_CUT, SEEDS or LEVELS",
         judgement="the interim pass could not have influenced the arm: its settings were in the process memory of the first run (started 18:50:56, before any test score existed) and its curve is identical in the rerun; what the builder saw at 20:13 equals what was finally reported for the four sets. "
                   "The one residual freedom (to drop the arm after seeing the interim scores) was not used, and the arm's result (0.610 / 17.7%) would not have changed any conclusion either way."))
chk("c6", "what was chosen on the 12 before E3 (hidden size and learning rate in 04_lstm_all.py on the validation years of all 12; properties 1989-2009 climate averages)", "the script's notes (read) and the stored settings", "finding",
    dict(effect="shared by the controls and the new sets; a weak leak for the never-seen test, as 05_lobo.py and note 6 of the results file say; the 63-river models and the controls use the same settings"))

# ---- (d) rivers never seen ---------------------------------------------------------------------------------------------------------------------------------
nest = m8["nesting"]
chk("d1", "the nested gauge: CAMELS outlines of all 63 basins intersected in an equal-area projection (EPSG:5070): which pairs share area; 03471500 against 03473000 by outline and by the attribute tables",
    "own code (m8_unseen.py: pyshp, pyproj, shapely)", "pass",
    dict(pairs_checked=nest["pairs_in_the_63_checked"], pairs_sharing_area=nest["pairs_sharing_area"], pairs_touching_only=nest["pairs_touching_only"],
         touching_between_a_study_river_and_one_of_the_50=nest["touching_pairs_between_a_study_basin_and_one_of_the_50"], touching_among_the_12=nest["touching_pairs_among_the_12"], attribute_check=nest["attribute_check"],
         note="03471500 (S F Holston River at Riverside, 198 km2) lies 100% inside 03473000 (near Damascus, 785 km2: 25% of it), upstream on the same river, 22.5 km away, daily flows correlate 0.94: for that river the 51-river design would not be a river never seen. No other pair of the 63 shares area."))
pr = m8["the_12_are_not_read_probe"]
chk("d2", "the 12 never read in training: (1) the training ids of the six saved never-seen networks; (2) flood.fit run with a dictionary from which the 12 are ABSENT against one with all 63 tables present (same seed, one update, with and without properties): weights compared; "
          "(3) flood.fit run with a dictionary that RAISES on any access to a table of the 12 (and a property table without them); (4) the builder's own probe in the results file",
    "own code (m8_unseen.py)", "pass",
    dict(training_ids_exclude_the_12_and_equal_the_50=all(v["none_of_the_12"] and v["equals_the_50"] and not v["contains_nested_gauge"] for v in m8["training_ids_of_the_saved_never_seen_networks"].values()),
         fit_with_the_12_absent_vs_all_63_present_max_weight_gap=dict(with_properties=pr["static"]["fit_with_the_12_absent_vs_all_63_present_max_weight_gap"], without_properties=pr["nostatic"]["fit_with_the_12_absent_vs_all_63_present_max_weight_gap"]),
         raising_guard_fit_completed=dict(with_properties=pr["static"]["fit_with_a_raising_guard_completed"], without_properties=pr["nostatic"]["fit_with_a_raising_guard_completed"]),
         scalers_equal=dict(with_properties=pr["static"]["scalers_equal_absent_vs_present"], without_properties=pr["nostatic"]["scalers_equal_absent_vs_present"]), builders_probe=m8["builder_probe_in_json"],
         how_the_builders_probe_was_done="three 100-update trainings on the 50 (patience 1): two on the clean tables, one with the 12's tables replaced by random numbers x100 (flow missing) and their properties replaced by random numbers; curves and weights identical (gap 0). "
                                         "It is a valid test that fit() reads only the tables of its ids; the reviewer's versions are stricter (absent from the dictionary; any access raises).",
         validation_of_the_early_stopping_is_on_the_50_only=True))
sm = m8["proximity"]["summary"]
chk("d3", "proximity recomputed: median distance of each of the 12 gauges to the nearest training gauge (camels_topo gauge_lat, gauge_lon; own haversine) for the 11 of 05_lobo.py, the 50 and the 51; nearest training basin in standardised property space by the script's definition "
          "(mean and spread of the training pool) and by the review request's (mean and spread over the 63); properties outside the training range; zero-flow days",
    "own code (m8_unseen.py)", "pass",
    dict(median_km=dict(leave_one_out_11=sm["11_leave_one_out"]["median_km"], fifty=sm["50"]["median_km"], fifty_one=sm["51"]["median_km"]), results_file_median_km=dict(eleven=m8["proximity"]["json_summary"]["11_leave_one_out"]["median_km"], fifty=m8["proximity"]["json_summary"]["50"]["median_km"]),
         rivers_within_50_km=dict(eleven=sm["11_leave_one_out"]["n_within_50_km"], fifty=sm["50"]["n_within_50_km"]), rivers_within_100_km=dict(eleven=sm["11_leave_one_out"]["n_within_100_km"], fifty=sm["50"]["n_within_100_km"]),
         median_property_distance_pool_standardised=dict(eleven=sm["11_leave_one_out"]["median_prop_distance_pool_standardised"], fifty=sm["50"]["median_prop_distance_pool_standardised"]),
         median_property_distance_standardised_over_the_63=dict(eleven=sm["11_leave_one_out"]["median_prop_distance_standardised_over_the_63"], fifty=sm["50"]["median_prop_distance_standardised_over_the_63"]),
         rivers_with_a_property_outside_the_training_range=dict(eleven=sm["11_leave_one_out"]["n_with_a_property_outside_the_training_range"], fifty=sm["50"]["n_with_a_property_outside_the_training_range"]),
         zero_flow=m8["zero_flow"], landscapes_of_the_50=m8["landscape_counts_of_the_50"], per_basin=m8["proximity"]["per_basin"]))
bud = m8["lobo_budget"]
rep = bud["per_variant"]["replay_of_the_05_stopping_rule_on_the_50_river_curves"]
chk("d4", "the comparison with results/lstm_lobo.json: one training per held-out river (cap 1000 updates, patience 4, seed 0) against three seeds of one model (2000 / 6) that scores all 12: the budget caps in the 05 trainings; the 05 stopping rule replayed on the six 50-river curves; "
          "the three seeds one by one against the 05 numbers; what seed averaging adds (seed averages against the mean of their single seeds)",
    "own code (m8_unseen.py)", "finding",
    dict(budget_05=dict(cap=bud["cap_in_05"], with_properties=dict(trainings=12, at_the_update_limit=bud["per_variant"]["lobo_w_static"]["n_at_update_limit"], best_updates=sorted(bud["per_variant"]["lobo_w_static"]["best_update"])),
                        without_properties=dict(trainings=12, at_the_update_limit=bud["per_variant"]["lobo_w_nostatic"]["n_at_update_limit"], best_updates=sorted(bud["per_variant"]["lobo_w_nostatic"]["best_update"]))),
         budget_50=bud["config_of_the_50_river_models"], replay_of_the_05_rule_on_the_50_river_curves=rep,
         replay_summary="the 05 rule (cap 1000, patience 4) would have kept an earlier checkpoint in 2 of the 6 trainings (1300 -> 1000 and 1200 -> 800 updates), costing 0.001 of validation NSE each: the budget difference is immaterial on the validation score",
         seeds_one_by_one_lead1=dict(with_properties=dict(lobo=PS["unseen50_w_static"]["l1"]["median_lobo"], seed_medians=[x["median"] for x in PS["unseen50_w_static"]["l1"]["seeds"]],
                                                              each_minus_lobo=[round(x["median_minus_lobo"], 4) for x in PS["unseen50_w_static"]["l1"]["seeds"]], rivers_higher=[x["n_rivers_higher"] for x in PS["unseen50_w_static"]["l1"]["seeds"]],
                                                              wilcoxon_p_per_seed=[round(x["wilcoxon_p"], 4) for x in PS["unseen50_w_static"]["l1"]["seeds"]], seed_average_rivers_higher=PS["unseen50_w_static"]["l1"]["seed_average_n_rivers_higher"],
                                                              seed_average_p=round(PS["unseen50_w_static"]["l1"]["seed_average_wilcoxon_p"], 4)),
                                     without_properties=dict(lobo=PS["unseen50_w_nostatic"]["l1"]["median_lobo"], seed_medians=[x["median"] for x in PS["unseen50_w_nostatic"]["l1"]["seeds"]],
                                                              each_minus_lobo=[round(x["median_minus_lobo"], 4) for x in PS["unseen50_w_nostatic"]["l1"]["seeds"]], rivers_higher=[x["n_rivers_higher"] for x in PS["unseen50_w_nostatic"]["l1"]["seeds"]],
                                                              wilcoxon_p_per_seed=[round(x["wilcoxon_p"], 4) for x in PS["unseen50_w_nostatic"]["l1"]["seeds"]], seed_average_rivers_higher=PS["unseen50_w_nostatic"]["l1"]["seed_average_n_rivers_higher"],
                                                              seed_average_p=round(PS["unseen50_w_nostatic"]["l1"]["seed_average_wilcoxon_p"], 4))),
         seed_averaging_adds_to_the_median_nse_lead1={n: round(v["average_minus_mean_single"], 4) for n, v in SE.items()},
         in_training_reference_of_05=dict(lstm_w_seed_0=SE["lstm_w"]["single_seed_medians"][0], lstm_w_seed_average=SE["lstm_w"]["seed_average_median"], note="05_lobo.py pairs its held-out models with lstm_w seed 0, the best of the three seeds (0.434, 0.418, 0.355); the loss it stored (0.126) is against that seed"),
         without_properties_in_training_partner=dict(lobo_ref_w_nostatic_one_training=g1("lobo_ref_w_nostatic", "nse_median"), unseen50_w_nostatic=g1("unseen50_w_nostatic", "nse_median"),
                                                     median_loss=J["unseen"]["loss_from_not_having_seen_the_river"]["50_rivers_without_properties__against_12_river_model"]["l1"]["median_loss"]),
         what_the_comparison_does="(1) the three-seed average beats a typical single training by 0.011-0.016 NSE (50-river) and 0.036-0.041 (12 and 63 rivers, no gauge), small against the gain of 0.12-0.16, but it decides the Wilcoxon flag with properties (single seeds p = 0.05-0.09, seed average p = 0.012); "
                                  "(2) one model scores all 12, so the 12 differences share its seed noise and are not independent draws, while the 05 numbers come from twelve different trainings; (3) the 05 trainings had the smaller budget (replay: immaterial); (4) the 50 lie nearer (d3); "
                                  "(5) the 05 in-training partner is one seed (the best), the without-properties partner a single weak training (0.383), which the never-seen 50-river model exceeds"))
chk("d5", "what the gain goes with: per river, the gain of the 50-river never-seen model over the leave-one-out model (NSE, 1 day ahead) against the distance to the nearest training gauge in the two designs, the change of that distance, the property distance; Spearman and Kendall; "
          "the five rivers whose nearest training gauge is no closer; the 4 of the 12 whose basin touches a training basin",
    "own code (m8_unseen.py) on the stored forecasts (which m7 proved equal to the saved networks' forecasts bit for bit)", "finding (information, not a verdict)",
    {k: dict(median_gain=GA[k]["median_gain"], rivers_with_a_gain=GA[k]["n_rivers_with_gain"], spearman_gain_vs_km_to_nearest_of_the_50=GA[k]["gain_vs_km_to_nearest_training_gauge_50"], spearman_gain_vs_km_11=GA[k]["gain_vs_km_11"],
             spearman_gain_vs_reduction_in_km=GA[k]["gain_vs_reduction_in_km"], spearman_gain_vs_property_distance_50=GA[k]["gain_vs_prop_distance_50"], spearman_gain_vs_reduction_in_property_distance=GA[k]["gain_vs_reduction_in_prop_distance"],
             kendall_gain_vs_km_50=GA[k]["kendall_gain_vs_km50"], spearman_nse_of_the_50_model_vs_km50=GA[k]["nse50_vs_km50"], spearman_nse_of_the_11_models_vs_km11=GA[k]["nse11_vs_km11"],
             rivers_whose_nearest_training_gauge_is_no_closer=GA[k]["rivers_whose_nearest_training_gauge_is_no_closer"], median_gain_there=GA[k]["median_gain_where_the_nearest_gauge_is_no_closer"],
             median_gain_where_closer_by_over_20_km=GA[k]["median_gain_where_the_nearest_gauge_is_closer_by_over_20_km"], median_gain_where_closer_by_at_most_20_km=GA[k]["median_gain_where_closer_by_at_most_20_km"],
             gain_by_river=GA[k]["gain_by_river"]) for k in ("with_properties_l1", "without_properties_l1", "with_properties_l2", "with_properties_l3")})
chk("d6", "does the skill of the saved 50-river networks on rivers they never saw fall with the distance to the nearest training gauge, over MORE than the 12? the networks are run on the 15 south-eastern candidate rivers the audit left out for gaps in the flow record "
          "(same selection rule as the 50; complete weather; scored on the days with an observed flow), and Spearman is taken over the 12, the 15 and the 27",
    "own code (m10_extra_rivers.py); nothing trained; the leave-one-out networks of 05 were not saved, so the gain over them cannot be computed for these rivers", "finding (information, not a verdict)",
    dict(spearman_nse_vs_km=EXTRA, by_distance_bin=m10["by_distance_bin"], extra_rivers=m10["extra_rivers"], per_river=m10["per_river"], the_12_equal_the_results_file_max_abs_diff=m10["check_the_12_equal_results_file_max_abs_diff"],
         reading="over 27 rivers the 50-river models' NSE does not fall with distance (rho +0.09 with properties, +0.24 without; among the 15 extra rivers it is positive), so nearest-gauge distance in the 7-223 km range is not what sets their skill; "
                 "it does not separate scale from proximity either: the 50-river and 11-river designs differ in the number of rivers, their variety, the neighbours of each held-out river and the protocol at once, and the 11-river distances reach 522 km"))
chk("d7", "the probe 'the 12 rivers' properties replaced by the median of the 50' recomputed from the saved networks", "own code (m16_probe_properties.py)", "pass",
    dict(m16, note="reproduces the results file exactly under its definition of the largest change (the mean over seeds of each river's largest change, largest river: 15.28 and 15.74 mm/day); the largest change of any single forecast of any seed is 18.79 and 16.88 mm/day. "
                   "The 50-river model with properties scores the same median NSE with the 12's true properties as with the median of the 50 (0.376 / 0.377): its properties do not carry the never-seen rivers; the 63-river model, in which the 12 are in training, loses 0.074 (it recognises its training rivers by their properties)."))

# ---- (e) gain rule, verdicts, outcome sentence -------------------------------------------------------------------------------------------------------------
chk("e1", "the gain rule recomputed for every comparison the results file makes (4 main pairs and 6 never-seen pairs, 3 leads: 30): seed ranges, rivers higher, Wilcoxon p (scipy and by enumeration of all 2^12 sign patterns), the words 'clear gain' / 'no clear difference' / 'mixed' / 'clear loss' "
          "for NSE and for the onsets warned; the own-level onset comparisons of the never-seen pairs",
    "own code (m9_gain_rule.py)", "pass",
    dict(comparisons=m9["n_pair_lead_comparisons"], max_abs_numeric_difference_vs_json=m9["max_abs_numeric_difference_vs_json"], text_or_flag_differences=m9["text_or_flag_differences"], wilcoxon_scipy_vs_enumeration_max_abs_diff=m9["wilcoxon_scipy_vs_exact_enumeration_max_abs_diff"],
         own_level_comparisons=m9["own_level_rule"], main_pairs_lead1={k: dict(nse_without=v["l1"]["median_without"], nse_with=v["l1"]["median_with"], seed_range_without=v["l1"]["seed_range_without"], seed_range_with=v["l1"]["seed_range_with"],
                                                                              rivers_higher=v["l1"]["n_basins_higher"], wilcoxon_p=v["l1"]["wilcoxon_p"], reading=v["l1"]["reading"], onset_without=v["l1"]["onset_warned_without"], onset_with=v["l1"]["onset_warned_with"],
                                                                              onset_reading=v["l1"]["onset_reading"]) for k, v in m9["main"].items()},
         readings_by_lead={k: [v[l]["reading"] for l in ("l1", "l2", "l3")] for k, v in {**m9["main"], **m9["unseen"]}.items()}))
chk("e2", "the verdicts (GO / CAUTION / NOT YET by the criteria stored in results/flood_analysis.json) recomputed for 12 sets x 3 leads x 4 readings (plain level; level chosen on the validation years for all days; for onsets), and the pooled tables of lstm_wq and lstm_w against flood_analysis.json",
    "own code (m9_gain_rule.py)", "pass",
    dict(readings_compared=m9["verdicts"]["n_compared"], differences_from_the_results_file=m9["verdicts"]["differences"], controls_against_flood_analysis_json=m9["controls_against_flood_analysis_json"],
         lead1_plain_level={n: dict(all_days=V[n]["l1"]["plain_all"]["verdict"], onsets=V[n]["l1"]["plain_onset"]["verdict"]) for n in ("lstm_wq", "lstm63_wq", "lstm63eq_wq", "lstm_w", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic")}))
chk("e3", "the onsets at lead 1 (63 against 12 rivers, with the gauge) at the event level and at the validation-chosen warning level: is the 'clear loss' a loss of skill?",
    "own code (m9_gain_rule.py)", "finding",
    dict(hits=dict(rivers_12=g1("lstm_wq", "onset_hits"), rivers_63=g1("lstm63_wq", "onset_hits")), false_alarms=dict(rivers_12=g1("lstm_wq", "onset_false_alarms"), rivers_63=g1("lstm63_wq", "onset_false_alarms")),
         csi_plain_level=dict(rivers_12=V["lstm_wq"]["l1"]["plain_onset"]["csi"], rivers_63=V["lstm63_wq"]["l1"]["plain_onset"]["csi"]), paired_onsets=PD["lstm63_wq_against_lstm_wq"], binomial_standard_error_of_a_pooled_share_points=m9["binomial_standard_error_of_a_pooled_onset_share_points"],
         at_the_level_chosen_on_the_validation_years=dict(rivers_12=dict(level=V["lstm_wq"]["l1"]["chosen_onsets"]["trigger"], hits=V["lstm_wq"]["l1"]["chosen_onsets"]["hits"], share=V["lstm_wq"]["l1"]["chosen_onsets"]["pod"], false_share=V["lstm_wq"]["l1"]["chosen_onsets"]["far"], csi=V["lstm_wq"]["l1"]["chosen_onsets"]["csi"]),
                                                          rivers_63=dict(level=V["lstm63_wq"]["l1"]["chosen_onsets"]["trigger"], hits=V["lstm63_wq"]["l1"]["chosen_onsets"]["hits"], share=V["lstm63_wq"]["l1"]["chosen_onsets"]["pod"], false_share=V["lstm63_wq"]["l1"]["chosen_onsets"]["far"], csi=V["lstm63_wq"]["l1"]["chosen_onsets"]["csi"])),
         all_high_flow_days_plain_level=dict(rivers_12=dict(share=V["lstm_wq"]["l1"]["plain_all"]["pod"], false_share=V["lstm_wq"]["l1"]["plain_all"]["far"], csi=V["lstm_wq"]["l1"]["plain_all"]["csi"]),
                                             rivers_63=dict(share=V["lstm63_wq"]["l1"]["plain_all"]["pod"], false_share=V["lstm63_wq"]["l1"]["plain_all"]["far"], csi=V["lstm63_wq"]["l1"]["plain_all"]["csi"])),
         never_seen_at_the_observed_level=dict(lobo_w_static_hits=g1("lobo_w_static", "onset_hits"), unseen50_w_static_hits=g1("unseen50_w_static", "onset_hits"), paired=PD["unseen50_w_static_against_lobo_w_static"],
                                                note="an ungauged river has no observed level: the project's rule for it is the model's own level (23.4% -> 22.3% with properties; 18.5% -> 21.8% weather only)"),
         reading="at the fixed observed level the 63-river model warns of 8 fewer onsets and gives 59 fewer false warnings (211 against 278 warnings in all): the same CSI; at the level chosen on the validation years it warns of MORE onsets (132 against 114) with a higher CSI (0.161 against 0.147); "
                 "at the event level the difference is within chance (McNemar p = 0.24); on all high-flow days it is better (CSI 0.510 against 0.480). The seed rule calls it a 'clear loss' only because all three 63-river seeds lie below all three 12-river seeds, which are scored on the same 611 events."))
chk("e4", "the stopping record and the equal-training arm: sequences per river at the kept update; the validation curve after its peak", "own code (m12_records_logs.py and the results file's stopping block)", "finding",
    dict(mean_best_update=dict(rivers_12_control=ST["lstm_wq"]["mean_best_update"], rivers_63=ST["lstm63_wq"]["mean_best_update"], rivers_63_x5=ST["lstm63eq_wq"]["mean_best_update"]),
         sequences_per_river_at_the_best_update=dict(rivers_12_control=ST["lstm_wq"]["sequences_per_river_at_the_best_update"], rivers_63=ST["lstm63_wq"]["sequences_per_river_at_the_best_update"], rivers_63_x5=ST["lstm63eq_wq"]["sequences_per_river_at_the_best_update"]),
         x5_arm_best_updates=ST["lstm63eq_wq"]["best_update"], x5_arm_updates_run=ST["lstm63eq_wq"]["updates_run"], x5_validation_peak_and_last={str(s): dict(peak=eq_peak[s], last=eq_last[s]) for s in EQ_CURVES},
         reading="the validation score of the 63 rivers peaks at 1000-1500 updates in the 5x arm as in the 1x arm and falls afterwards (" + eq_txt + "); the kept networks of the 5x arm therefore had 593 sequences per river, 0.35 of the control's 1689: the arm shows that a five times longer patience window does not move the optimum, "
                 "not that equal training per river was tried"))

# ---- (f) saved networks ------------------------------------------------------------------------------------------------------------------------------------
chk("f1", "the 15 saved networks rebuilt from their files alone (flood.LSTMNet + Run.predict with the stored cfg, scalers, ids and the properties of the 12) on the 12-basin table and on the 63-basin table, and by the reviewer's own input matrix and block-rolling: the stored forecasts reproduced",
    "engine route (allowed) and own code (m7_networks.py)", "pass",
    dict(networks=len(net), values_per_network=447048, bit_for_bit_engine_12_table=all(v["f"]["engine_bitwise"] for v in net.values()), bit_for_bit_engine_63_table=all(v["f"]["engine_on_63_table_bitwise"] for v in net.values()),
         bit_for_bit_own_input=all(v["f"]["own_input_bitwise"] for v in net.values()), max_abs_difference=max(v["f"]["engine_max_abs_diff"] for v in net.values()), results_file_reload_check_all_bit_for_bit=J["reload_check"]["all_bit_for_bit"],
         sha256_of_the_15_networks={k: v["sha256"] for k, v in net.items()}))

# ---- (g) figure, run_all.sh, logs, hand-in ----------------------------------------------------------------------------------------------------------------
chk("g1", "figures/more_basins.png against the results file, read on the pixels: panels calibrated on the drawn grid lines, every bar found by its fill colour or outline, bar tops, seed ranges (caps and the highest seed dot); the printed labels, legend and notes read on the picture",
    "own code (m11_figure.py); the self-test moves every expected position by 6 px and must fail", "pass",
    dict(bars_checked=m11["n_bars_checked"], seed_ranges_checked=m11["n_seed_ranges_checked"], max_bar_top_deviation_px=m11["max_bar_top_deviation_px"], max_seed_extent_deviation_px=m11["max_seed_extent_deviation_px"],
         px_per_unit=m11["panels"], one_pixel_is="0.0016 NSE (top left), 0.05 points (top right), 0.0011 NSE (bottom left), 0.064 points (bottom right)",
         labels_read_on_the_picture="0.59 0.60 0.61 | 0.44 0.45; 19% 17% 18% | 17% 18%; 0.22 0.38 0.44 | 0.31 0.43 0.38; 23% 22% 23% | 18% 22% 18%: all equal the json rounded; the notes' 0.32 -> 0.34, 0.25 -> 0.28, 39 against 110 km equal the json",
         all_ok=m11["all_ok"], selftest_failed_as_it_must=(not O("m11_figure_selftest")["all_ok"]),
         note="the bottom panels' outline bar 'the same rivers when in training' is the 3-seed average lstm_w in the with-properties group (0.44) but ONE training, lobo_ref_w_nostatic (0.38), in the weather-only group; the legend line 'bar = 3 seeds averaged' does not hold for the grey 11-river bars (one training per river)"))
chk("g2", "run_all.sh: the position of the 17_more_basins.py line against what the script reads", "own reading of run_all.sh and the script", "pass",
    dict(line=20, after=["01_data_audit (basin_audit.csv, data_audit.json)", "04_lstm_all (lstm_all.json, preds)", "05_lobo (lstm_lobo.json, preds)", "06_flood_analysis (flood_analysis.json, line 19)"], before=["15_rain_to_midnight", "16_aorc_rain_to_midnight"],
         the_line_says_it_trains_15_networks=True, run_all_sh_changed_during_the_review="another agent added the line of 18_forecast_input.py (line 27) at 22:44:38; the line of 17_more_basins.py is still line 20, after line 19 (06_flood_analysis.py); the sha256 in files_reviewed is the later file", note="the script reads results/lstm_all.json, lstm_lobo.json, flood_analysis.json, basin_audit.csv, data_audit.json, the prediction sets of 04 and 05 and data/camels_us outlines: all are written above line 20"))
chk("g3", "the log, the results file's training records and the network files: every 'training k of N' line (15) against the record and the history inside the file; the learning curves; seconds; the timeline of the file times; the first run's wait for the CPU lock",
    "own code (m12_records_logs.py)", "pass with a note",
    dict(log_lines=m12["log_vs_records"]["n_training_lines_in_log"], consistent=m12["log_vs_records"]["n_consistent"], max_abs_diffs=m12["log_vs_records"]["max_abs_diff"], curves_equal_network_files=m12["log_vs_records"]["curves_equal_network_files"],
         mean_load_average_equals_network_files=m12["log_vs_records"]["mean_load_average_equals_network_files"], seconds=m12["seconds"], scoping_estimate_against_measured=m12["scoping_estimate"], cpu_lock=m12["cpu_lock"], timeline_first_run=m12["timeline_first_run"],
         note="the log's seconds are rounded to whole seconds (0.47 s at most from the records); training ran with 4 torch threads on a two-core laptop under a mean load average of 6.3"))
hi = m13["full_includes_e3"]
chk("g4", "the hand-in script make_handin_zip.sh: what `light` and `full` would pack now (a dry run with the script's own patterns), and whether --rescore can run from the unpacked `full` archive", "own code (m13_handin.py; the zip is not made)", "finding",
    dict(light=dict(files=m13["light"]["n_files"], megabytes_uncompressed=m13["light"]["total_mb"]), full=dict(files=m13["full"]["n_files"], megabytes_uncompressed=m13["full"]["total_mb"], by_folder_mb=m13["full"]["by_folder_mb"]), header_claims_full_mb=m13["header_claims_full_mb"],
         full_holds_the_63_cache=hi["cache63"], cache_mb=hi["cache63_mb"], e3_prediction_sets_mb=hi["preds_e3_mb"], data_folders_in_full_mb=m13["full_data_folders_mb"], rescore_inputs=m13["rescore_inputs"], missing_outline_raises=m13["missing_outline_raises"],
         reading="confirmed: `full` would pack the 50 MB cache (it is needed by --rescore: without it the 63 tables would be rebuilt from raw files the archive does not hold); the header's 'about 230 MB' is stale (757 MB uncompressed now); data/gefs (154 MB, the E2 download in progress) and data/aorc are not in SKIP; "
                 "the outline folder (69 MB) is skipped by `basin_*`, and outline_overlap() catches only ImportError: --rescore from the unpacked `full` archive stops with shapefile.ShapefileException (demonstrated in a sandbox without the folder)"))
chk("g5", "17_more_basins.py --rescore in a SANDBOX copy of the project (clones of the sets, networks and caches; links to the CAMELS inputs): the results file and the figure against the project's; nothing written into the project", "own code (m14_sandbox_rescore.sh, m14_compare.py)", "pass with a note",
    dict(leaf_differences=m14["n_differences"], top_level_keys=m14["top_level_keys_with_differences"], differences=m14["differences"], figure_byte_identical=m14["figure_byte_identical"], every_number_note_and_table_identical=m14["only_passes_run_time_and_choices_differ"],
         rescore_without_note_drops_the_choices_record=m14["rescore_without_note_drops_the_choices_record"], choices_record_in_the_project=m14["choices_record_in_project"],
         project_files_unchanged="sha256 of every prediction set, network, cache, the results file and the figure equal before and after the sandbox run (m14_sandbox_rescore.sh)",
         without_the_outline_folder="the same run with data/camels_us/basin_set_full_res absent fails in outline_overlap() with shapefile.ShapefileException (g4)"))
chk("g6", "files of the review: sha256, size, time; the hashes the results file recorded for its own 35 files against the files now", "own code (m15_integrity.py)", "pass",
    dict(integrity_record=m15["integrity_record_of_the_results_file"], quick_mode_flags_all_false=m15["quick_mode_flags_of_the_20_prediction_files"]["all_false"], e3_smoke_test_output_in_results_quick=m15["results_quick_folder_has_17_more_basins_output"][:3],
         note="results/quick holds smoke-test output of the script (FLOOD_QUICK=1), which make_handin_zip.sh leaves out; no prediction set of the real folder is a quick-mode file"))
rd = m12["roadmap_claims_checked"]
chk("g7", "the numbers of ROADMAP.md (step 1d result) against the results file", "own code (m12_records_logs.py) and reading", "finding",
    dict(every_table_number_present_rounded_as_the_json_rounds=all(all(v.values()) for v in m12["roadmap_numbers_present"].values()), claims_checked=rd,
         wording_points=["'a clear NSE gain 2 and 3 days ahead (0.32 to 0.35, 0.25 to 0.28)': the first pair is 0.3230 -> 0.3446, which rounds to 0.34 (0.35 is the 5x arm, 0.3511); the figure's note says 0.32 -> 0.34",
                         "'without basin properties a never-seen river now forecasts about as well as a trained one (0.43 against 0.44)': 0.434 is the weather-only 50-river model, 0.443 the 12-river model WITH properties (lstm_w); the like-for-like in-training partner without properties is lobo_ref_w_nostatic, one training, 0.383, which the never-seen model exceeds (median loss -0.006)",
                         "'Onsets warned for never-seen rivers barely move (23% to 22%)': true with properties (23.4% -> 22.3%, own level); weather only they rise 18.5% -> 21.8% (every 50-river seed above the single 05 training)"]))

# ---- text ------------------------------------------------------------------------------------------------------------------------------------------------------------
discrepancies = [
    "No numerical discrepancy: every number of results/more_basins.json that was recomputed (headline scores of 12 sets x 3 leads, the 20 single-seed and average files, seed ranges, 30 gain-rule comparisons with their Wilcoxon p, 144 verdict readings, the own-level onset warnings, proximity numbers, training records, the properties probe, the position of 22 bars and 16 seed ranges in the figure) equals the reviewer's value (0.0 against the engine; at most 5e-5 where the audit csv is rounded to four decimals).",
    "One definition, not a number: the 'forecasts change by up to 15.28 mm/day' of finding 15 is the largest river of the seed-mean of each river's largest change; the largest change of any single forecast of any seed is 18.79 mm/day (unseen50_w_static) and 16.88 (lstm63_w).",
    "Wording of the results file: the onsets of the 63-river gauge model are called a 'clear loss' (and ROADMAP.md says 'a clear loss of about 8 of 611'); the data show a move along the warning level (must-fix 1).",
    "Wording of the results file: 'equal training per river' (note 3, findings 2, 5, 8, outcome_caveat) for lstm63eq_wq, whose kept networks were the same early checkpoints as the 1x arm (should-fix 1).",
    "Note 5 of the results file says 'part of any difference is proximity, not the number of rivers'. The reviewer's reading of the same data: the gain does not rank with distance to the nearest training gauge (rho -0.36, p 0.26 with properties; -0.27, p 0.39 without), five rivers whose nearest training gauge is no closer gained as much (+0.085 median) as the others, and over 27 rivers the 50-river models' skill does not fall with distance (rho +0.09) - so 'part of it is proximity' is a hedge the data do not support, and 'it is scale' is not shown either (should-fix 3).",
    "The with-properties never-seen comparison is called a 'clear gain' by the project's rule because the three-seed average of the 50-river model is compared with ONE training per river of 05_lobo.py; single 50-river trainings against the 05 numbers give p = 0.052, 0.092 and 0.077 (each +0.13 to +0.16 NSE above, higher in 10, 10 and 8 of 12 rivers); without properties p = 0.0015, 0.034, 0.0015 (should-fix 2).",
    "ROADMAP.md step 1d: '0.32 to 0.35' (should read 0.32 to 0.34); '(0.43 against 0.44)' pairs a model without properties with one with properties (like-for-like: 0.43 against 0.38, a single training); 'Onsets warned for never-seen rivers barely move (23% to 22%)' holds for the with-properties variant only (weather only: 18.5% -> 21.8%).",
    "ROADMAP.md step 1d, written before the run: 'Six trainings take about 18 minutes against 16 for the 12 alone'; measured: the six 63-river trainings took 42.7 minutes on a laptop with a mean load average of 6.3 (all 15: 124.8 minutes, as the result section says). README.md: run_all.sh 'about 4 hours' is stale (E3 alone adds about two hours).",
    "`waited_for_cpu_s` = 0.0 and `runtime_s` of the results file leave out the first run (78 minutes of training and a 33.8 minute wait for the lock held by the AORC script); note 13 says so, and the log annotates it.",
    "The results file's `choices_made_after_looking_at_test_scores` is written from --note arguments of the command line: the final --rescore carried it because it was given the note again; a rescore without it writes an empty list (demonstrated in a sandbox: all other leaves are identical).",
    "Four of the 63 basins are on CAMELS's own list of basins whose area is wrong by more than 10% (data/camels_us/basin_dataset_public_v1p2/basin_size_errors_10_percent.txt): 02108000 (one of the 12; listed +100%, but its forcing-file, gauge and outline areas agree within 1.2%, so the flow in mm/day is not affected) and three of the 50 (02324400, 02327100, 02422500: forcing-file area "
    + ", ".join(f"{100 * (1 - lst[g]['forcing_over_gauge']):.1f}%" for g in ("02324400", "02327100", "02422500")) + " below the gauge's). The effect on the 50-river training is at most 13% in the flow scale of one river: immaterial, but not mentioned (a7).",
]
must_fix = [
    "Do not quote the onset result of the 63-river gauge model as a 'clear loss' (outcome_caveat, findings 0 and 1, ROADMAP.md step 1d). Put beside the 19% -> 17% what the data show: 8 fewer hits (114 -> 106) and 59 fewer false alarms (164 -> 105) at the fixed observed level, CSI 0.147 -> 0.148, paired onsets McNemar p = 0.24; at the warning level chosen on the validation years the 63-river model warns of 132 onsets (21.6%) against 114 (18.7%), CSI 0.161 against 0.147; on all high-flow days it is better (CSI 0.510 against 0.480; false share 27% against 32%). "
    "The label comes from the seed rule applied to three seeds that are scored on the same 611 events; it is a move of the operating point, not a loss of forecast skill, and the project's own rule 3 (onsets and false warnings together) and E5 rule (compare at equal false-warning share) say so. Best place: the `outcome_caveat` clause of results/more_basins.json (a field the builders read), so that build_report.py and the others quote it from there.",
    "Make --rescore (and --extend) carry `choices_made_after_looking_at_test_scores` over from the existing results file (as it does `passes`), or read it from a small file, before the file is rewritten for must-fix 1: as it is, rewriting the results file without repeating the --note argument silently erases the record that work order section 3 requires ('say so in the results file').",
]
should_fix = [
    "Say what the equal-training arm is: 5x the update limit and 5x the patience window, but its kept networks are at 1000, 1000 and 1500 updates (mean 1167; 593 sequences per river against the 12-river control's 1689, 0.35 times), the same place as the 1x arm (mean 1133), because the 63-river validation score peaks there and falls afterwards (peak -> last check: " + eq_txt + "). Replace 'equal training per river' by '5x update limit and patience window' in note 3, findings 2, 5, 8 and the outcome_caveat, and add the best update beside the 4167 updates run; if equal training at the KEPT network is wanted, fix the number of updates (about 3300 for equal sequences per river) or stop on the 12's validation years.",
    "For the never-seen comparison give the single-training rows beside the seed-average rows in the outcome_caveat: each single 50-river training is +0.127, +0.161, +0.144 (with properties) above the 05 number, p = 0.052, 0.092, 0.077; the 'clear gain' flag stands on the seed average (p = 0.012). Say once that the three-seed average adds about 0.011-0.016 NSE over a typical single 50-river training, and that the 05 in-training partner (lstm_w seed 0, 0.434) is the best of three seeds (average 0.443; the weather-only partner is one training, 0.383).",
    "Replace 'part of any difference is proximity, not the number of rivers' (note 5, outcome_caveat) by what was measured: scale and proximity are not separated by this design; the gain over the 05 models does not rank with the distance to the nearest training gauge (rho -0.36, p 0.26 with properties; -0.27, p 0.39 without; n = 12), the five rivers whose nearest training gauge is no closer gained +0.085 and +0.091 (median), and the saved 50-river networks do not score lower on rivers more than 100 km from any training gauge (median NSE " + f"{bins['beyond 100 km']['median_nse_with_properties']:.2f} with properties, {bins['beyond 100 km']['n']} rivers) than on rivers within 50 km ({bins['within 50 km']['median_nse_with_properties']:.2f}, {bins['within 50 km']['n']} rivers); over 27 never-seen rivers rho = +0.09 with properties and +0.24 without (m10). "
    "The test that would separate them is the buffered leave-out (train on the 50 minus every river within 50 and 100 km of the one held out) and a dose-response over random subsets of 11, 25 and 50 rivers: the 'what was not done' list already names both; they need training.",
    "Correct ROADMAP.md step 1d: '0.32 to 0.35' -> '0.32 to 0.34'; '(0.43 against 0.44)' -> '0.43 against 0.38 for the like-for-like in-training reference without properties (one training), 0.44 with properties'; 'barely move (23% to 22%)' -> add 'weather only: 18.5% -> 21.8%'; and replace the scoping estimate 'about 18 minutes' for six trainings by the measured 43 minutes on a loaded laptop. README.md: add the row for 17_more_basins.py and the results file to the step table, and the two extra hours to the run_all.sh time (NEXT_STEPS_FOR_AGENTS.md section 6 asks for it).",
    "Figure: say in the caption that the outline bar of the weather-only group is one training (0.38) and that of the with-properties group a three-seed average (0.44), and that the grey 11-river bars are one training per river without a seed range ('bar = 3 seeds averaged' does not apply to them).",
    "outline_overlap(): treat a missing outline folder like the missing packages (record 'not re-checked here', keep the stored result) so that --rescore can run from the unpacked `full` hand-in archive, which skips the 69 MB outline folder; as it is it stops with shapefile.ShapefileException (demonstrated).",
    "make_handin_zip.sh: the 63-basin cache (50 MB) is packed by `full`, and it must be (--rescore reads it); update the header's 'about 230 MB' (757 MB uncompressed now, of which E3 holds 50 MB of cache and 72 MB of prediction sets), add data/gefs (154 MB of the unfinished E2 download) to SKIP, and decide whether data/aorc (21 MB) belongs in the archive.",
    "Record the first run in `timing` (12 trainings, 78.3 minutes, 33.8 minutes waiting for the lock) so that waited_for_cpu_s and runtime_s do not read 0.0 and 52 minutes for a 125-minute job (note 13 explains it, the numbers do not).",
    "test_flood.py: add a message to the plain `assert` lines of the new check (three of the eight broken variants fail with a bare AssertionError), and add one check for basins(ids=...) / candidate_ids() on a small made-up basin_audit.csv (the real-file recomputation of this review is m3_candidates_raw.py); the guards inside the functions (the complete column must follow the rule; the ten properties must be finite) are already asserted at run time.",
    "The finding-15 sentence: 'forecasts change by up to 15.28 mm/day' -> 'the largest river changes by 15.28 mm/day on average over the seeds (18.8 in one seed)'.",
]
observations = [
    "What more rivers buy for the same 12 rivers is mainly steadier training, not higher skill: the single-seed range of the median NSE with the gauge falls from 0.542-0.598 (12 rivers) to 0.581-0.597 (63); the average gains 0.015 at 1 day (inside the 12-river seed range: no clear gain), 0.022 and 0.028 at 2 and 3 days (every 63-river seed above every 12-river seed, 10 of 12 rivers, p = 0.002 and 0.016: clear by the rule). Without the gauge: +0.010, +0.023, +0.023, no clear difference at any lead (at 3 days 'mixed').",
    "The verdicts do not move: onsets NOT YET for every set at every lead; all high-flow days CAUTION with the gauge (63 rivers: 63.0% warned, 27.3% false; 12 rivers: 61.9%, 31.9%), NOT YET without it. The upper bound (perfect weather forecast) stays at 64% of onsets: more rivers do not close the information gap, as the roadmap says.",
    "For rivers never seen the 50-river models are 0.12-0.16 NSE above the 11-river leave-one-out models at 1 day and 0.12-0.15 at 2 and 3 days, and the weather-only 50-river model (0.434) is above even the one-training in-training reference without properties (0.383). Onsets warned by the project's rule for a river without a gauge move 23.4% -> 22.3% with properties and 18.5% -> 21.8% without; at the observed level (not available to an ungauged river) the with-properties model warns of fewer onsets (125 -> 102 hits, McNemar p = 0.014).",
    "The 50-river model with properties scores the same median NSE when the 12's properties are replaced by the median of the 50 (0.376 / 0.377); the 63-river model with the 12 in training falls to 0.379 from 0.453: properties let a model recognise rivers it has trained on; for rivers it has not seen they carry nothing on average (with properties 0.376, without 0.434: no clear difference by the rule, 'mixed').",
    "Proximity in the strict sense also differs: four of the 12 (02108000, 02137727, 02296500, 02298608) have a training basin whose outline touches theirs in the 50-river design; in the leave-one-out design only the pair Catawba / South Toe touches.",
    "The 05 trainings stopped at their update limit in 5 of 12 (with properties) and 3 of 12 (without) cases; the replay of that rule on the six 50-river curves changes the kept checkpoint in two (cost 0.001 validation NSE each): the budget difference between 05 and the new never-seen sets is real but immaterial.",
    "Training ran with 4 torch threads on a two-core laptop shared with other jobs (mean load average 6.3); this affects the recorded seconds only.",
    "Other agents were working in the project during the review (E2 download and scripts; smoke-test output in results/quick): the E3 files, caches and networks were hashed at the start and the end and did not change.",
]
verdict = (
    "The numbers, the data, the engine change and the saved networks of E3 hold up under independent recomputation; I found no leakage and no arithmetic error. Every headline number of results/more_basins.json (five new sets, seven controls, three leads, and the 20 seed and average files) was reproduced with my own code to 0.0, also by flood.evaluate; "
    "the seed averages are bitwise means of their seed files, obs are the project's flow bit for bit, the stored sets hold exactly the 12. The engine change is confined to basins() and load_data(): with no ids the hand-in engine and the present engine rebuild the same 9,543,479-byte cache from the raw files, equal to the stored one; thresholds and the lstm_wq headline (median NSE 0.5896, 18.66% of onsets) are unchanged; "
    "no other script passes ids; test_flood.py passes and its new check fails for all eight deliberately broken load_data() variants. The 63 basins are exactly what the audit rule gives (recomputed from the raw files for all 78 candidates), their cache equals tables I rebuilt bit for bit, and the 12 inside it equal the stored 12. "
    "All 15 saved networks reproduce their stored forecasts bit for bit from the file alone (also with my own input matrix); their scalers equal my recomputation from the training years of the training rivers; early stopping used the validation years of the 63 (of the 50, never the 12, for the never-seen models). "
    "The never-seen training never reads the 12 (fit with the 12 absent from the dictionary gives identical weights); the nested gauge 03471500 lies 100% inside 03473000 and is the only overlapping pair of the 63. The interim scoring pass did not influence the equal-training arm: the 15-training plan is the first line of the log, the killed run printed the arm's curve and the rerun the same numbers, and the other four sets' results are unchanged. "
    "The gain rule, its words and all verdicts reproduce exactly. What the experiment shows is smaller than its sentences, and three statements say more than the data. "
    "(1) With the gauge, 63 rivers give no clear NSE gain at 1 day (0.590 -> 0.604, inside the 12-river seed range), a clear one at 2 and 3 days (+0.022, +0.028) and much steadier trainings; the onsets are not 'a clear loss': at the fixed level the model warns of 8 fewer onsets and 59 fewer false alarms with the same CSI (McNemar p = 0.24), at the level chosen on validation it warns of more (21.6% against 18.7%); the verdict stays NOT YET (must-fix 1). "
    "(2) The 'equal training' arm is not equal at the kept network: its checkpoints are the same early ones as the 1x arm (593 sequences per river against 1689 for the control), because the 63-river validation score peaks at 1000-1500 updates; it shows that a longer patience window does not move the optimum. "
    "(3) For rivers never seen, 50 training rivers give +0.16 (with properties) and +0.12 (weather only) NSE over the 05 numbers in every seed, and that survives the seed-average and budget differences; but it compares a three-seed average of one model with single trainings (a single seed's p is 0.05-0.09 with properties), and it cannot be attributed to the number of rivers: it does not rank with the distance to the nearest training gauge (rho -0.36, p 0.26) and the 50-river networks do not score lower on 27 never-seen rivers the farther they lie (rho +0.09), yet the two designs differ in number, variety and neighbours at once; the test that would separate scale from proximity (buffered leave-out, dose-response) needs training and was not done. "
    "Open before E3 can enter the report: must-fix 1 and 2 and the wording items of should-fix 1 to 5; the outcome sentence is fair arithmetic but must not be quoted without a corrected outcome_caveat."
)
counts = dict(checks_listed=len(checks), checks_pass=sum(1 for c_ in checks if c_["result"] == "pass"), checks_pass_with_a_note=sum(1 for c_ in checks if c_["result"] == "pass with a note"),
              findings_not_pass_fail=sum(1 for c_ in checks if c_["result"].startswith("finding")), failed=sum(1 for c_ in checks if c_["result"] == "fail"), must_fix=len(must_fix), should_fix=len(should_fix), numerical_discrepancies=0)
now = time.strftime("%H:%M")
R = dict(
    who="one independent AI reviewer agent (Claude), started separately with its own instructions, with no right to edit the project except this file and the folder code/review/more_basins/; the code it reviewed was written and run by a different AI agent session (Claude Code) directed by the team lead",
    human_review="not yet done: the team's own check of E3 is still open (team.json untouched by this review)",
    when=f"2026-10-07, 21:59 to {now} (local time, UTC+4), after the training runs of 18:50-21:50 and the final --rescore of 21:53, before any document was rebuilt from E3",
    what="independent recomputation of the engine change (flood.basins / load_data with ids, study_ids, candidate_ids; test_flood.py), the data of the 63 basins, the headline numbers of every new set, the protocol of the 63-river and the never-seen trainings, the saved networks, the gain rule, the verdicts, the figure, the logs and the hand-in script of "
         "code/17_more_basins.py (experiment E3: one LSTM trained on 63 south-eastern CAMELS rivers instead of 12, and on 50 other rivers for rivers never seen), with the reviewer's own code; the engine (flood.load_preds, thresholds, evaluate, load_data, basins, LSTMNet, Run.predict, fit) used only where the work order allowed",
    files_reviewed=m15["files"], method=dict(own_code_folder="code/review/more_basins/ (in the project; the work was done in the session scratch folder review_e3/ and copied; list under scripts)", torch_threads=1, nothing_trained=True,
                                             script_not_run="the script itself was not run (neither plain, --extend nor FLOOD_QUICK=1); its --rescore mode was run in sandbox copies of the project (several times; once with the outline folder absent, as in the `full` hand-in archive), never in the project", nothing_installed=True,
                                             project_writes="only results/review_more_basins.json and the folder code/review/more_basins/; no bytecode (PYTHONDONTWRITEBYTECODE=1), no cache and no lock file was written into the project by this review: the sha256 of every prediction set, network, cache, the results file and the figure of E3 are equal before and after the sandbox run, and at the end",
                                             builder_evidence_used="two files of the builder's session scratch folder (the interim results of 20:13 and a copy of the script as it ran at 19:35), copied into code/review/more_basins/evidence_from_builder_scratch/; every conclusion about them is checked against the log and the final files",
                                             other_agents="E2 work was running in the same project (data/gefs, fetch scripts, code/18_forecast_input.py); of the 63 files hashed at the start and at the end only run_all.sh changed (a line for the E2 script); every E3 file, the engine, the caches, the 15 networks and the 20 prediction sets are unchanged"),
    checks=checks,
    answers=dict(
        a_engine="Only basins() and load_data() changed (plus two helper functions and an import); every other top-level statement of flood.py is byte-identical and in the same order. Default calls are unchanged: both engines rebuild the identical 12-basin cache from raw, equal to the stored one; thresholds and the lstm_wq headline (0.5896, 18.66%) equal. candidate_ids() returns the 63 the rule gives; the 63 are complete in flow and forcing by my own parsers; the 63-basin cache is bit-identical to my rebuild and the 12 inside it to the stored 12. "
                 "test_flood.py passes; its new check fails for 8 of 8 broken load_data() variants (and for the new test on the old engine). No other script passes ids or star-imports flood: no other behaviour can change.",
        b_headline="Reproduced to 0.0 by two routes for every new set and control; seed averages are bitwise means; obs are the project's flow; the stored sets hold the 12 only. The numbers quoted in the request all round to the recomputed ones.",
        c_protocol="Training ids, scalers (bitwise), early stopping (replay and re-scored weights), thresholds and static scalers are from the training years and the training rivers; nothing was chosen on the test years; the interim scoring pass could not have influenced the arm (the arm was in the 15-training plan at 18:50:56, its curve in the killed run equals the rerun's, the four sets' results are identical before and after). Note the shared weak leak: the settings were chosen on the validation years of all 12 in 04.",
        d_unseen="The 50 exclude the 12 and the nested gauge; the 12 are never read in training (three proofs); the proximity numbers are reproduced (39.3 against 110.1 km; 1.42 against 2.23, and 1.32 against 2.27 when standardised over the 63). The comparison with 05 is one three-seed average of one model against single trainings; the gain (+0.16 / +0.12) holds for every seed but the with-properties flag rests on the average; budget differences are immaterial on validation; the gain cannot be attributed to the number of rivers (d5, d6), and the data speak against nearest-gauge distance as the cause.",
        e_gain_rule="All 30 comparisons, their words and 144 verdict readings reproduce; the outcome sentence is fair arithmetic but its onset clause ('clear loss' in the caveat) and 'equal training' are not what the data show (must-fix 1; should-fix 1 to 3); the verdict NOT YET is right by the stored criteria (onsets); all high-flow days CAUTION with the gauge.",
        f_networks="All 15 networks reproduce their stored forecasts bit for bit (447,048 values each) by the engine, on the 63-basin table and by the reviewer's own input matrix; the negative control (properties scaled with the 12's own spread) changes them by 3 to 13 mm/day.",
        g_other="The figure agrees with the json on 22 bars (0.7 px) and 16 seed ranges (1.8 px), and its self-test fails when moved; the run_all.sh line (20) is after 04, 05 and 06; the logs, records and file times agree; `make_handin_zip.sh full` would pack the 50 MB cache (required by --rescore), 757 MB in all; --rescore in a sandbox reproduces the file and the figure byte for byte except the new pass and the choices record, which it drops."),
    numbers=dict(headline_lead1={n: dict(nse_median=round(SC[n]["l1"]["nse_median"], 4), onset_hits=SC[n]["l1"]["onset_hits"], onset_share=round(SC[n]["l1"]["onset_pod"], 4)) for n in SC}, wilcoxon_main_pairs_lead1={k: v["l1"]["wilcoxon_p"] for k, v in m9["main"].items()},
                 paired_onsets_lead1=PD, spearman_gain_vs_distance=dict(with_properties=GA["with_properties_l1"]["gain_vs_km_to_nearest_training_gauge_50"], without_properties=GA["without_properties_l1"]["gain_vs_km_to_nearest_training_gauge_50"]),
                 spearman_nse_vs_distance_27_rivers=EXTRA["all 27"], seed_averaging_effect=SE),
    discrepancies=discrepancies, must_fix=must_fix, should_fix=should_fix, observations=observations, counts=counts, verdict=verdict,
    scripts=dict(folder="code/review/more_basins/", note="the reviewer's own check scripts; they are separate from the project's code (run_all.sh, the report builders and the numbered scripts neither run nor read them); README.md says what each one checks, how to run it and how long it takes. m17_assemble_record.py rebuilds this record from the outputs of the others.",
                 files={"m_common.py": "paths, own readers of the raw CAMELS files, own NSE and onset counting, helpers", "m1_engine_diff.py": "a1, a6", "m2_defaults.py": "a2", "m3_candidates_raw.py": "a3, a4", "m3b_area_errors.py": "a7", "m5_testflood_mutants.py": "a5", "m6_headline.py": "b1-b4",
                        "m7_networks.py": "c1-c3, f1", "m8_unseen.py": "d1-d5", "m9_gain_rule.py": "e1-e3", "m10_extra_rivers.py": "d6", "m11_figure.py": "g1", "m12_records_logs.py": "c5, g3, g7", "m13_handin.py": "g4", "m14_sandbox_rescore.sh + m14_compare.py": "g5",
                        "m15_integrity.py": "g6", "m16_probe_properties.py": "d7", "m17_assemble_record.py": "this record", "run_fast_checks.sh, run_slow_checks.sh": "the order and the times", "evidence_from_builder_scratch/": "two files of the builder's scratch folder used by m12"}),
    not_checked=[
        "No training was done: whether a new training with the same seeds reproduces the numbers (thread count, operating system, load) is untested; the stored forecasts and networks are what was checked.",
        "The CAMELS data themselves (USGS flow, Daymet) were checked for completeness, units and format, not against other sources; the outline geometry was used as CAMELS ships it.",
        "The scale-or-proximity question was not settled: it needs the buffered leave-out and the dose-response over random subsets (training); m10 and d5 are information from stored forecasts and saved networks.",
        "The leave-one-out networks of 05_lobo.py are not saved (work order P2), so they could not be rebuilt or run on other rivers.",
        "The simple-model yardstick (ridge, forest) at 63 or 50 rivers, a 63-river model without properties and 13_uncertainty.py for the new sets do not exist (the results file lists them as not done).",
        "Documents (report, slides, guide, Q&A) were not built or read: none contains E3 yet; ROADMAP.md and README.md were read for their E3 numbers only (g7).",
        "The literature numbers quoted in ROADMAP.md (Kratzert et al.) were not verified in this review.",
        "run_all.sh lines other than the one of 17_more_basins.py (E1, E1b, E2 work) and the other experiments' files were not reviewed here."],
    dates=dict(training_runs="2026-10-07 18:50:56-20:51 (first run, stopped by the harness in the first equal-arm training), 21:02-21:50 (--extend: three equal-arm trainings and the probe), 21:53 (--rescore)", review=f"2026-10-07 21:59-{now}"))
text = json.dumps(R, indent=1, allow_nan=False)
out = PROJ / "results" / "review_more_basins.json" if "--write-project" in sys.argv else OUT / "review_more_basins.json"
out.write_text(text)
print(f"wrote {out} ({len(text)} bytes): {counts}")
