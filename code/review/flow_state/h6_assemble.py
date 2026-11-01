"""E4e review, step 6: the review record. Builds results/review_flow_state.json from the outputs of h1-h5 (OUT/h*.json) after checking that no reviewed file changed since h1 hashed it.
Without --write-into-project the record goes to OUT/review_flow_state.json only. Seconds."""
from h_common import *

WRITE = "--write-into-project" in sys.argv
R = jload(RESULTS)
h1, h2, h3, h4, h5 = (jread(f"h{i}_{n}.json") for i, n in ((1, "integrity"), (2, "headline"), (3, "columns"), (4, "networks"), (5, "outcome")))
S, M, ops = R["summary_test"]["sets"], h2["headline_mine"], h2["operating_points_mine"]
pc, p1 = (lambda x: f"{100 * x:.1f}%"), (lambda x: f"{100 * x:.1f}")
# nothing reviewed may have changed since h1 hashed it
changed = [f for f, h in h1["code_files_sha256"].items() if sha256(f) != h]
assert not changed, f"a reviewed file changed since h1 ran: {changed}"
assert h2["summary"]["not_matching"] == [], f"h2 found mismatches: {h2['summary']['not_matching']}"
W, fig, three, tk = h5["wordings"], h5["figure"], h2["success_with_three_seeds"], h2["turkey_creek"]
cov = R["data_coverage"]["per_basin"]
rt = lambda n, k="l1": dict(file=dict(nse_median=S[n][k]["nse_median"], onset_pod=S[n][k]["onset"]["pod"], onset_far=S[n][k]["onset"]["far"], onset_csi=S[n][k]["onset"]["csi"], onset_hits=S[n][k]["onset"]["hits"], all_csi=S[n][k]["all"]["csi"]),
                             mine=dict(nse_median=M[n][k]["nse_median"], onset_pod=M[n][k]["onset"]["pod"], onset_far=M[n][k]["onset"]["far"], onset_csi=M[n][k]["onset"]["csi"], onset_hits=M[n][k]["onset"]["hits"], all_csi=M[n][k]["all"]["csi"]),
                             seed_onset_pod_mine=M[n][k]["seed_onset_pod"], seed_onset_pod_file=R["seed_ranges"]["sets"][n][k]["by_seed"]["onset_pod"])
dl = R["delivery"]["families"]
op = lambda n, rule, k="l1": dict(file=dict(trigger=R["operating_points"]["sets"][n][k]["rules"][rule]["trigger"], **{c: R["operating_points"]["sets"][n][k]["rules"][rule]["test"]["onset"][c] for c in ("pod", "far", "csi", "hits")}, verdict=R["operating_points"]["sets"][n][k]["rules"][rule]["test_verdict"]),
                                  mine=dict(trigger=ops[f"{n} {k} {rule}"]["trigger"], **{c: ops[f"{n} {k} {rule}"]["test_onset"][c] for c in ("pod", "far", "csi", "hits")}, verdict=ops[f"{n} {k} {rule}"]["test_verdict"]))
findings = [
    dict(severity="must-fix", where="results/flow_state.json outcome_caveat, clause 'the hourly flow is the gauge's own record, an input a real-time service has' (code/22_flow_state.py, the outcome_caveat f-string); the same words in the figure footer (draw())",
         text=f"The clause overstates what was tested. Every hourly value used here is from the APPROVED record (approved share 1.0 in all 12 gauges, data_coverage), and the 24:00 variant reads the hour that ends at the issue time itself, i.e. it assumes "
              f"zero delivery delay; a real-time service gets PROVISIONAL values (revised later, most at high flow, when shifts and rating changes are applied) with the telemetry delay of the gauge (USGS transmits most stations hourly, some every four hours). "
              f"The 22:00 variant covers a delay of at most two hours and is the conservative one of the two only against such a delay. Since the documents print outcome_full word for word once this review exists (code/build_slides_final.py: e4e_reviewed -> "
              f"N('e4e_outcome_full')), the sentence must carry the two qualifications. Fix (replace the clause): 'the hourly flow is the gauge's own record, an input a real-time service has as provisional values and after a delivery delay (the 24:00 variant "
              f"assumes none and the 22:00 variant two hours; this study used the approved record), while the AORC rain to midnight is {R['outcome_caveat'].split('while the AORC rain to midnight is ')[1].split(')')[0]}) for the daily total; the within-day "
              f"columns of wqaWH_s and wqaWHF_s have no such measurement'. State the same in design.what or a note: 'the 24:00 variant assumes zero delivery delay and approved values'."),
    dict(severity="must-fix", where="figures/flow_state.png, top panel (code/22_flow_state.py draw(): the line at 70 with the text 'GO asks 70% warned (at most 30% false)' and the bar labels)",
         text=f"The wqaWHF22_s bar is labelled {fig['labels_read']['wqaWHF22_s']['pod']}% and sits on the line labelled 'GO asks 70% warned (at most 30% false)', and no false share appears anywhere on the figure; its false share is {pc(S['wqaWHF22_s']['l1']['onset']['far'])} "
              f"(verdict CAUTION, operating_points.sets.wqaWHF22_s.l1.rules.plain), and no set of this experiment meets the GO false share at trigger 1.0 (false shares {pc(min(S[n]['l1']['onset']['far'] for n in SHOWN))} to {pc(max(S[n]['l1']['onset']['far'] for n in SHOWN))}; "
              f"at the 30%-false operating point the best realistic set warns of {pc(max(ops[f'{n} l1 far_cap']['test_onset']['pod'] for n in NEW))}). A reader of the figure alone takes the bar as GO. Fix: print the false share with every bar label from summary_test "
              f"(e.g. '70%\\n42% false') and change the line's text to 'GO asks 70% warned AND at most 30% false: no bar here keeps the false share under 30% at trigger 1.0', or draw the 30%-false operating points as a second mark on each bar (operating_points far_cap test pod)."),
    dict(severity="should-fix", where="results/flow_state.json choices_made_after_looking_at_test_scores and e4e_success (code/22_flow_state.py CHOICES)",
         text=f"'Sets, inputs, seeds, stopping rule and the success rule ... were fixed in NEXT_EXPERIMENTS_2026-10-28.md and in this script before training': the work order's E4e section names no seed count (its table row says '{W['work_order_e4e_trainings_phrase'] or '10-20 trainings'}'; "
              f"E4b's section says 'three seeds, then five'), the script's default is three, and the run was launched with --seeds 5 in one pass (honest: before any test score was seen; h1: one training pass, seeds {R['seeds']}). The seed count decides the verdict here: with the first three "
              f"seeds the rule would be MET (wqaH_s seeds 0-2: {', '.join(p1(v) for v in three['wqaH_s_seeds_0_2'])}% against wqa_s {', '.join(p1(v) for v in three['wqa_s_seeds'])}%), and it fails only because seeds {three['seeds_of_wqaH_s_at_or_below_the_best_seed_of_wqa_s']} "
              f"({p1(M['wqaH_s']['l1']['seed_onset_pod'][3])}%, {p1(M['wqaH_s']['l1']['seed_onset_pod'][4])}%) fall below the best wqa_s seed ({p1(three['best_seed_of_wqa_s'])}%). Fix: 'Five seeds were chosen at launch (--seeds 5; the script's default is three; the work order "
              f"names no count), before any test score was seen; with the first three seeds the rule would be met' in choices, and e4e_success.met_with_first_three_seeds = true beside met = false. The spread of wqa_s's three seeds ({p1(three['wqa_s_seed_spread_points'] / 100)} points) "
              f"against {p1(three['wqaH_s_seed_spread_points'] / 100)} for wqaH_s is why an 'every seed above every seed' rule is fragile against a three-seed base; the seed-mean of wqaH_s is above every wqa_s seed."),
    dict(severity="should-fix", where="results/flow_state.json outcome_caveat, the clauses 'with the last available hour ending 22:00 instead of 24:00, 43%' and the 'With the six within-day AORC rain columns ... with the GEFS forecast as well' sentence; delivery",
         text=f"The 24:00-against-22:00 difference is inside the seed spread in every family at every lead (delivery: every_seed_24_above_22 false everywhere; H family lead 1 {dl['H']['l1']['difference_points_24_minus_22']:+.1f} points with the ranges touching at "
              f"{p1(dl['H']['l1']['seed_range_24'][0])}% = {p1(dl['H']['l1']['seed_range_22'][1])}% (an exact tie of 153 hits), WH {dl['WH']['l1']['difference_points_24_minus_22']:+.1f}, WHF {dl['WHF']['l1']['difference_points_24_minus_22']:+.1f}: the 22:00 forecast set is the higher one), "
              f"and the file says so only in tables. Fix: add to the sentence 'the two hours cost {dl['H']['l1']['difference_points_24_minus_22']:.1f} points in the river-only sets, {dl['WH']['l1']['difference_points_24_minus_22']:.1f} with the within-day rain and "
              f"{-dl['WHF']['l1']['difference_points_24_minus_22']:.1f} the other way with the forecast, all inside the seed spread (no family separates the two seed by seed)'. Likewise wqaWHF_s against wqaF_s is 'no clear difference' by the seed rule "
              f"(62% to 74% against 57% to 65%, readings.wqaWHF_s_vs_wqaF_s.l1) with 49 onsets warned only with the new columns against 25 only without (sign test p = {R['readings']['pairs']['wqaWHF_s_vs_wqaF_s']['paired_onsets']['l1']['sign_test_p']:.3f}): say which reading the +7 points rests on."),
    dict(severity="should-fix", where="results/flow_state.json outcome_caveat, clause 'the AORC rain to midnight is an analysis made after the fact (hindsight at most about 3 points)'",
         text="The 3-point bound was measured for the daily AORC total only (the E1b review: the saved wqa networks given Stage IV for the issue day); the wqaWH_s and wqaWHF_s numbers of the same sentence rest on the six within-day AORC columns of E4b, whose hindsight "
              "has no measurement (results/review_within_day_inputs.json, should-fix 2, still open). Fix: '(hindsight at most about 3 points for the daily total; unmeasured for the within-day columns of wqaWH_s and wqaWHF_s)' and add that measurement to what_was_not_done."),
    dict(severity="should-fix", where="results/flow_state.json what_was_not_done (code/22_flow_state.py what_was_not_done list)",
         text="The list holds 'five seeds (unless --seeds 5 was used)' although --seeds 5 was used (seeds [0, 1, 2, 3, 4], n_seeds_per_set 5 for every new set): evaluate the condition in the script and drop the entry (the E4b review raised the same point); add 'the hindsight "
              "of the within-day AORC columns (Stage IV test)' and 'a test with provisional (not approved) gauge values and the measured telemetry delay'."),
    dict(severity="should-fix", where="results/flow_state.json readings.pairs.<pair>.by_lead.l<k>.onset.difference_by_seed_points (code/22_flow_state.py rule(): zip over the two seed lists)",
         text=f"For the {h5['wordings']['difference_by_seed_points_truncated']} pair-leads that compare a five-seed set with a three-seed stored set (the six new sets against their bases and the two against wqaP_s) the list holds three entries (seeds 0-2) and seeds 3 and 4 of the new set are "
              "silently dropped (zip truncation; the E4b review raised the same point). Fix: name the list 'seeds_0_2_only' or give the full matrix of pairwise differences; nothing else reads it."),
    dict(severity="should-fix", where="figures/flow_state.png header (code/22_flow_state.py draw(): R['status'][:150])",
         text="The status line is cut mid-word ('... every number of this file is preliminary a'). Fix: cut at the first ';' or at a word boundary, or print only the first clause ('NOT YET REVIEWED: no independent review record exists for this experiment').",),
    dict(severity="should-fix", where="results/flow_state.json status; figures/flow_state.png header",
         text="Both say 'NOT YET REVIEWED', true when they were written; the script sets 'reviewed: results/review_flow_state.json exists' only on its next pass. After the wording fixes run `python code/22_flow_state.py --rescore --seeds 5` (nothing trained; it refuses if a saved "
              "file changed) so that the status, the figure and the fixed clauses are rebuilt together; then rerun this review's h5 and h6 on the rebuilt file (h1-h4 need no rerun: the prediction sets and networks are untouched by a rescore)."),
    dict(severity="note", where="results/flow_state.json e4e_success; summary_val; models.*.per_seed.best_val_nse",
         text=f"The pre-registered rule (wqaH_s above wqa_s in every seed at lead 1, pooled onsets at trigger 1.0) is NOT met with the five seeds, as the file says; the seed-mean gain of {R['readings']['pairs']['wqaH_s_vs_wqa_s']['by_lead']['l1']['onset']['gain_points']:+.1f} points "
              f"is 'no clear difference' by the onset seed rule and 'clear gain' in NSE (every seed above, 11 of 12 basins, p = 0.001), with 41 onsets warned only with the river's state against 19 only without (sign p = 0.006). Unlike E4b, the validation years foreshadow it: "
              f"onsets warned {pc(h2['validation_reading']['wqaH_s']['onset_pod_l1'])} against {pc(h2['validation_reading']['wqa_s']['onset_pod_l1'])} at trigger 1.0, every wqaH_s seed ({pc(min(h2['validation_reading']['wqaH_s']['single_seeds_onset_pod_l1']))} to "
              f"{pc(max(h2['validation_reading']['wqaH_s']['single_seeds_onset_pod_l1']))}) above every wqa_s seed ({pc(min(h2['validation_reading']['wqa_s']['single_seeds_onset_pod_l1']))} to {pc(max(h2['validation_reading']['wqa_s']['single_seeds_onset_pod_l1']))})."),
    dict(severity="note", where="results/flow_state.json data_coverage.per_basin.02349900; per_basin",
         text=f"Turkey Creek's {cov['02349900']['fallback_days_24_test_years']} fallback test days lie in two outages ({', '.join(f'{a} to {b} ({n} days)' for a, b, n in tk['fallback_runs_test_years'])}) that hold none of its {tk['per_set']['wqaH_s']['l1']['all']['onsets']} lead-1 test onsets, "
              f"so its onset table is the same with and without them ({tk['per_set']['wqaH_s']['l1']['all']['onset_hits']} hits, {tk['per_set']['wqaH_s']['l1']['all']['onset_false_alarms']} false alarms for wqaH_s against {tk['per_set']['wqa_s']['l1']['all']['onset_hits']} and "
              f"{tk['per_set']['wqa_s']['l1']['all']['onset_false_alarms']} for wqa_s); its lead-1 NSE is {tk['per_set']['wqaH_s']['l1']['all']['nse']:.3f} on all test days and {tk['per_set']['wqaH_s']['l1']['issue_day_known']['nse']:.3f} on the targets whose issue day is known "
              f"(wqa_s {tk['per_set']['wqa_s']['l1']['all']['nse']:.3f} and {tk['per_set']['wqa_s']['l1']['issue_day_known']['nse']:.3f}). Pooled, {tk['pooled_lead1_onsets_by_issue_day_status']['wqaH_s']['onsets_with_fallback_issue_day']} of the 335 lead-1 onsets have a fallback issue day: "
              f"wqaH_s warns {tk['pooled_lead1_onsets_by_issue_day_status']['wqaH_s']['hits_fallback']} of them against {tk['pooled_lead1_onsets_by_issue_day_status']['wqa_s']['hits_fallback']} for wqa_s, and on the other {tk['pooled_lead1_onsets_by_issue_day_status']['wqaH_s']['onsets_with_known_issue_day']} "
              f"{pc(tk['pooled_lead1_onsets_by_issue_day_status']['wqaH_s']['pod_known'])} against {pc(tk['pooled_lead1_onsets_by_issue_day_status']['wqa_s']['pod_known'])}: the fallback days cost the new columns about three hits, nothing more."),
    dict(severity="note", where="code/22_flow_state.py docstring and MISMATCH_REL comment ('South Toe on 17 Sep 2004, 53 against 175 mm/day'); data_coverage.mismatch_days_24",
         text=f"In the 24:00 variant none of the {h3['mismatch_rule_alone']['total_mismatch_days_24']} mismatch days is decided by the mismatch rule alone: every one already has hours missing after the carry (the gap rule sends it to the fallback first); in the 22:00 variant the rule alone decides "
              f"{h3['mismatch_rule_alone']['total_decided_alone_22']} days (a rise in the last two hours). The docstring's example is such a gap day: 17 Sep 2004 has one hour with values (the hour ending 24:00, {h3['south_toe_2004_09_17']['mean_of_the_hours_with_values_mm']:.1f} mm/day), "
              f"{h3['south_toe_2004_09_17']['hours_still_missing_after_the_carry']} hours still missing after the carry, a mean over the available hours of {h3['south_toe_2004_09_17']['hourly_mean_over_available_hours_mm']:.1f} against the daily mean {h3['south_toe_2004_09_17']['camels_daily_mean_mm']:.1f}; "
              f"'53' is the one known hour, not the quantity the rule compares. Harmless; the rule is a second guard."),
    dict(severity="note", where="results/flow_state.json data_coverage.per_basin.*.local_day_mean_vs_camels; code/22_flow_state.py flow_state_table (the wall clock with summer time)",
         text=f"The USGS daily mean's day follows the wall clock with summer time in 10 of 12 gauges (the script's convention: in June-August the local-day mean of the hourly flow reproduces the CAMELS daily mean better with it), but standard time all year fits better at Murder Creek (02221525) "
              f"and Turkey Creek (02349900): there the script's 24:00 in summer is 23:00 standard time, one hour before the end of the USGS day, so the columns read one hour LESS than the daily mean, never more (no leakage); the daily mean q_mm of day t, an input of every model of the project, "
              f"then covers one hour after the issue time at those two gauges in summer: a project-wide convention, not E4e's (h3 usgs_daily_value_clock)."),
    dict(severity="note", where="results/flow_state.json reload_check; causality; checks_against_stored_results",
         text=f"On this PC all 30 networks reproduce their stored forecasts bit for bit with one torch thread (the run used 4), on tables built as the script builds them and on tables whose hourly-flow columns are the reviewer's own (which equal the script's bit for bit on every day); the scalers "
              f"inside every file equal the training-row statistics; the reviewer's made-up-hours test ({h3['made_up_hours']['n_cases']} cases, two zones, clock-change days included, the script's builder and the reviewer's) and made-up-values test ({h4['made_up_values']['n_cases']} cases on "
              f"lstm_wqaH_s_s2 and lstm_wqaWHF_s_s1) pass; the three stored sets rescored here equal results/forecast_input.json to 1e-9."),
]
counts = dict(values_compared=h2["summary"]["entries"], numerical_mismatches=len(h2["summary"]["not_matching"]), networks_rebuilt=len(h4["reproduction"]), made_up_value_cases_on_networks=h4["made_up_values"]["n_cases"], made_up_hour_cases_on_data=h3["made_up_hours"]["n_cases"],
              outcome_clauses_rebuilt=len(h5["clauses_rebuilt_from_my_numbers"]), outcome_clauses_found=sum(v["found_in_outcome_full"] for v in h5["clauses_rebuilt_from_my_numbers"].values()), basins_columns_rebuilt=12, variants_rebuilt=2,
              must_fix=sum(f["severity"] == "must-fix" for f in findings), should_fix=sum(f["severity"] == "should-fix" for f in findings), notes=sum(f["severity"] == "note" for f in findings))
verdict = "discrepancies" if counts["numerical_mismatches"] or not h4["all_30_bit_for_bit_on_script_tables"] or not h3["all_basins_both_variants_equal_on_known_days"] else "numbers hold with wording fixes" if counts["must_fix"] + counts["should_fix"] else "numbers hold"
record = dict(
    who="independent AI reviewer agent (Claude), 2026-10-29", human_review="not yet done: the team's own check of E4e is still open (team.json untouched by this review)",
    when=f"2026-10-29, after the run of code/22_flow_state.py (results file written {R['written']}; training 12:11-12:37); scripts in code/review/flow_state/",
    what="Independent review of experiment E4e (code/22_flow_state.py: the hourly USGS gauge flow at the issue time - q_end, dq6, dq12, q_dmax, hours_above with the last available hour ending 24:00 or 22:00 local - as inputs of the all-river LSTM on the satellite-era split; "
         "results/flow_state.json, figures/flow_state.png, 30 networks, 36 prediction sets). Every headline number recomputed from results/preds with the reviewer's own numpy scoring (no flood.evaluate, no warning_policy); the hourly-flow columns rebuilt independently from "
         "data/hourly_flow for all 12 basins and both variants; all 30 networks rebuilt from their .pt files; own causality tests; leakage check; the sentence, the wordings, the disclosure and the figure.",
    files_reviewed=h1["code_files_sha256"], record_script_sha256=R["script_sha256"], record_written=R["written"],
    method=dict(scoring="own numpy code from the obs and pred arrays of results/preds/*.npz: onsets = target days above Q95 whose flow lead days earlier was at or below Q95; warning = forecast > trigger x Q95; pooled over the 12 basins; Q95 = 95th percentile of the daily flow of "
                        "2000-10-01..2008-09-30; triggers chosen on the validation years 2008-10..2010-09 by the CSI rule (best pooled onset CSI, ties to the trigger nearest 1.0) and the 30%-false rule (most onsets warned with pooled onset FAR <= 0.30) over 0.50..2.00 step 0.05",
                columns="own route (h3): zoneinfo instants of 00:00 and H:00 local on the UTC axis, reduceat per day, own carry (<= 6 h) and mismatch rule; compared with the script's flow_state_table executed from the script text",
                networks="flood.Cfg / LSTMNet / Run.predict as how_to_reload says, on tables built as the script builds them and with the reviewer's columns (h4)", scripts="h1_integrity, h3_columns, h2_headline, h4_networks, h5_outcome, h6_assemble (README.md)"),
    reproduced=dict(summary=h2["summary"], headline_lead1={n: rt(n) for n in SHOWN}, headline_lead2={n: rt(n, "l2") for n in SHOWN}, headline_lead3={n: rt(n, "l3") for n in SHOWN},
                    e4e_success=dict(file=dict(met=R["e4e_success"]["met"], seed_ranges=R["e4e_success"]["seed_ranges"]), mine=dict(h2["success_mine"], seed_ranges={n: [min(M[n]["l1"]["seed_onset_pod"]), max(M[n]["l1"]["seed_onset_pod"])] for n in SHOWN}), with_the_first_three_seeds=three),
                    paired_onsets_lead1_wqaH_s_vs_wqa_s=dict(file=R["readings"]["pairs"]["wqaH_s_vs_wqa_s"]["paired_onsets"]["l1"], mine=h2["paired_mine"]["wqaH_s_vs_wqa_s"]["l1"]),
                    paired_onsets_lead1_wqaWHF_s_vs_wqaF_s=dict(file=R["readings"]["pairs"]["wqaWHF_s_vs_wqaF_s"]["paired_onsets"]["l1"], mine=h2["paired_mine"]["wqaWHF_s_vs_wqaF_s"]["l1"]),
                    closure_wqaH_s_lead1=dict(file=R["closure"]["sets"]["wqaH_s"]["by_lead"][0], mine={k: h2["rows"][f"closure wqaH_s l1 {k}"]["mine"] for k in ("share_of_nse_gap_closed", "onset_share_of_gap_closed", "onset_gain_points", "basins_better_than_base")}),
                    operating_points_lead1={n: {rule: op(n, rule) for rule in ("plain", "csi", "far_cap")} for n in SHOWN}, operating_points_lead1_far_cap_22_variants={n: op(n, "far_cap") for n in ("wqaH22_s", "wqaWH22_s", "wqaWHF22_s")},
                    delivery_lead1={fam: dict(file={k: dl[fam]["l1"][k] for k in ("onset_warned_24", "onset_warned_22", "difference_points_24_minus_22", "every_seed_24_above_22", "every_seed_22_above_24", "onset_false_24", "onset_false_22")},
                                              mine=dict(onset_warned_24=M[f"wqa{fam}_s"]["l1"]["onset"]["pod"], onset_warned_22=M[f"wqa{fam}22_s"]["l1"]["onset"]["pod"], difference_points_24_minus_22=h2["rule_mine"][f"wqa{fam}_s_vs_wqa{fam}22_s l1"]["onset_gain_points"],
                                                        every_seed_24_above_22=h2["rule_mine"][f"wqa{fam}_s_vs_wqa{fam}22_s l1"]["every_seed_above_onsets"], every_seed_22_above_24=h2["rule_mine"][f"wqa{fam}_s_vs_wqa{fam}22_s l1"]["every_seed_below_onsets"],
                                                        onset_false_24=M[f"wqa{fam}_s"]["l1"]["onset"]["far"], onset_false_22=M[f"wqa{fam}22_s"]["l1"]["onset"]["far"])) for fam in ("H", "WH", "WHF")},
                    all_day_verdicts_lead1_plain={n: dict(file=R["operating_points"]["sets"][n]["l1"]["rules"]["plain"]["test_verdict"], mine=ops[f"{n} l1 plain"]["test_verdict"]) for n in SHOWN},
                    stored_sets_vs_forecast_input_json={n: dict(file_says_all_equal=R["checks_against_stored_results"][n]["all_equal"], my_rescoring_equals_forecast_input_json=all(h2["rows"][k]["ok"] for k in h2["rows"] if k.startswith(f"stored {n} forecast_input.json"))) for n in STORED},
                    go_scan_test_years=dict(onsets_go_anywhere=any(x["reading"].startswith("onsets") for x in h2["go_scan"]), all_days_go=[f"{x['set']} l{x['lead']} {x['rule']} (pod {x['pod']:.3f}, far {x['far']:.3f})" for x in h2["go_scan"] if x["reading"] == "all high-flow days"]),
                    thresholds_equal=all(h2["rows"][f"thresholds {g}"]["ok"] for g in R["basins"]), seed_means_equal_mean_of_seed_files=h2["seed_mean_check"], difference_by_seed_points_truncated_pair_leads=len(h2["difference_by_seed_points_truncated"]),
                    validation_years=h2["validation_reading"], turkey_creek=tk, outcome_clauses=h5["clauses_rebuilt_from_my_numbers"], outcome_numbers_not_covered=h5["numbers_not_covered_by_a_rebuilt_clause"]),
    columns_rebuilt=dict(what=h3["script_function"], all_basins_both_variants_equal_on_known_days=h3["all_basins_both_variants_equal_on_known_days"], all_basins_both_variants_equal_all_days=h3["all_basins_both_variants_equal_all_days"], counts_equal_file=h3["all_counts_equal_file"],
                         fallback_total=dict(mine=h3["fallback_total_mine"], file=h3["fallback_total_file"]), fallback_test_total_mine=h3["fallback_test_total_mine"], basin_days=h3["basin_days_total"], per_basin={g: {L: {k: v for k, v in h3["columns_vs_script"][g][L].items() if k not in ("mismatch_dates_mine",)} for L in ("24", "22")} for g in R["basins"]},
                         thresholds=h3["thresholds"], areas=h3["areas"], areas_all_from_forcing_header=h3["areas_all_from_forcing_header"], carry_rule=h3["carry_rule"], mismatch_rule_alone=h3["mismatch_rule_alone"], south_toe_2004_09_17=h3["south_toe_2004_09_17"],
                         usgs_daily_value_clock=dict(basins_fitting_standard_time_better=h3["usgs_daily_value_clock"]["basins_fitting_standard_time_better"], summer_median_relative_difference={g: {k: round(v[k]["median_relative_difference_summer"], 4) for k in ("wall_clock_with_summer_time", "standard_time_all_year")} for g, v in h3["usgs_daily_value_clock"]["per_basin"].items()}),
                         coverage={g: h3["columns_vs_script"][g]["coverage"] for g in R["basins"]}),
    networks_rebuilt=dict(torch=h4["torch"], threads=h4["threads"], n=len(h4["reproduction"]), all_30_bit_for_bit_on_script_tables=h4["all_30_bit_for_bit_on_script_tables"], all_30_bit_for_bit_on_my_tables=h4["all_30_bit_for_bit_on_my_tables"], worst_max_abs_diff=h4["worst_max_abs_diff"],
                          scalers_from_training_rows_in=h4["scalers_from_training_rows_in"], settings_true_in=h4["settings_true_in"], future_false_in_all=h4["future_false_in_all"], use_flow_true_in_all=h4["use_flow_true_in_all"], first_forecast_day=h4["first_forecast_day"],
                          stored_empty_before_2000_in_all=h4["stored_empty_before_2000_in_all"], pt_files=h1["networks"]["true_in"], prediction_meta=h1["prediction_meta"], integrity=dict(saved_files=h1["saved_files"], stored_set_files=h1["stored_set_files"], hourly_files=h1["hourly_files"], script=h1["script"], log=h1["log"])),
    causality=dict(made_up_hours_mine=h3["made_up_hours"], made_up_values_mine=h4["made_up_values"], scripts_own_tests_as_recorded=dict(made_up_hours_passed=R["causality"]["made_up_hours"]["passed"], made_up_values_sets=list(R["causality"]["made_up_values"]), data_side=R["causality"]["data_side"]["passed"]),
                   reading="no column of day t reads an hour after the issue instant (24:00 or 22:00 local; the 22:00 columns of day t never read hours 23 and 24 of day t, and the 24:00 columns of a cut day move when its last two hours change); no forecast issued on or before a cut day moves when every input after it is replaced"),
    leakage_check=dict(q95="the 95th percentile of the daily flow of the split's training years 2000-10-01..2008-09-30 only (recomputed from the stored obs: equal to the file in all 12 basins)",
                       scalers="dyn_mu/sd, q_mu/sd, lq_mu/sd inside every .pt equal the statistics of the training rows of the tables (30 of 30); st_mu/sd are basin properties",
                       stopping="best_update = argmax of the validation NSE of the history in every file; patience stop as configured (30 of 30); the validation years choose nothing else but the triggers of operating_points",
                       triggers="the CSI and 30%-false triggers replicated on the validation arrays: all 54 triggers and the test scores at them equal the file; the test years are scored once",
                       bias_factors="fc1..fc3 = fc_bc_1..3 of data/gefs/basins = fc_raw x the factor of results/gefs_basins.json (36 of 36 basin-leads, atol 2e-5), fitted on target days 2000-10-01..2005-09-30, inside this split's TRAINING years (nothing of its validation or test years)",
                       test_years_before_the_choice="the success rule and the sets were in the script hashed before the first network was written (script modified 12:10:56, first network 12:11:51, hash equal to the record); one training pass with --seeds 5; test scores printed per seed during the run did not change any choice recorded",
                       hourly_flow_known_at_the_issue_time="by construction and by the made-up-hours tests; the hourly values are the APPROVED record (approved share 1.0 in every gauge)",
                       delivery_assumptions="the 24:00 variant assumes ZERO delivery delay (the hour ending at the issue time is read) and approved values; the 22:00 variant assumes a two-hour delay and is the conservative one of the two against a delay of at most two hours (USGS telemetry is hourly for most stations, four-hourly for some); neither covers the provisional-to-approved revision",
                       verdict="no leakage found"),
    interpretation=dict(success_rule=dict(pre_registered=R["design"]["success_rule"], met=R["e4e_success"]["met"], mine=h2["success_mine"]["met"], with_first_three_seeds=three["met_with_three_seeds"], seeds_below_best_base_seed=three["seeds_of_wqaH_s_at_or_below_the_best_seed_of_wqa_s"]),
                        gauge_record_claim="fair only with two qualifications (approved values, delivery delay): see must-fix 1", delivery="the 24:00 variant assumes zero delivery delay: say so plainly; the 22:00 variant is the conservative one against a delay of at most two hours",
                        wqaWHF_s_vs_wqaF_s=dict(lead1=dict(warned_with=M["wqaWHF_s"]["l1"]["onset"]["pod"], warned_without=M["wqaF_s"]["l1"]["onset"]["pod"], seed_rule=h2["rule_mine"]["wqaWHF_s_vs_wqaF_s l1"]["onset_reading"], nse_rule=h2["rule_mine"]["wqaWHF_s_vs_wqaF_s l1"]["nse_reading"], paired=h2["paired_mine"]["wqaWHF_s_vs_wqaF_s"]["l1"]),
                                                leads_2_3_points=[h2["rule_mine"][f"wqaWHF_s_vs_wqaF_s l{k}"]["onset_gain_points"] for k in (2, 3)]),
                        wqaWHF22_s_70_percent=dict(onset_pod=M["wqaWHF22_s"]["l1"]["onset"]["pod"], onset_far=M["wqaWHF22_s"]["l1"]["onset"]["far"], verdict=ops["wqaWHF22_s l1 plain"]["test_verdict"], meets_go=False, in_outcome_full=W["wqaWHF22_mentioned_in_outcome_full"], on_the_figure_at_the_go_line=True),
                        go_rule="no set, lead, rule or single seed meets GO on the onsets in the test years; on all high-flow days every lead-1 set but wqa_s meets it at trigger 1.0 (as every realistic set of the earlier experiments did)",
                        turkey_creek="see note 2", validation_foreshadows_the_gain=True),
    wordings={k: W[k] for k in ("aorc_wording_in_outcome_full", "hindsight_wording_beside_the_aorc_clause", "hindsight_bound_measured_for", "upper_bound_wording_beside_wqaP_s", "split_caveat_comparable_only_inside_it", "base_weaker_than_main_split", "gauge_record_claim",
                                "gauge_record_claim_qualified_in_text", "provisional_mentioned_anywhere_in_file", "zero_delay_of_the_24_variant_stated_plainly", "status_says_not_yet_reviewed", "choices_claim_seeds_fixed_in_work_order", "work_order_e4e_names_a_seed_count",
                                "what_was_not_done_lists_five_seeds_although_five_were_used", "what_was_not_done_lists_the_review", "quoting_note")},
    figure=fig, findings=findings, counts=counts, verdict=verdict,
    verdict_text=f"Every number checked holds: {counts['values_compared']} values of results/flow_state.json recomputed from results/preds with the reviewer's own scoring match to 1e-9 (summary_test, summary_val, per_basin, seed_ranges, the 14 pair readings and paired onsets, "
                 f"the closure, 81 operating points with their single seeds and verdicts, the delivery table, e4e_success, the stored sets against results/forecast_input.json); the hourly-flow columns rebuilt by another route equal the script's bit for bit on every day of all 12 basins "
                 f"in both variants, with the fallback, carry and mismatch counts; all 30 networks reproduce their stored forecasts bit for bit; the reviewer's causality tests pass; no leakage found. The pre-registered rule is not met with the five seeds (and would be with the first three). "
                 f"The wording needs two fixes before the sentence is printed: the 'gauge's own record, an input a real-time service has' clause must say approved values and zero delivery delay (24:00) / two hours (22:00), and the figure must not show the 70% bar of wqaWHF22_s on the GO line "
                 f"without its 42% false share; plus the should-fix items on the seed-count disclosure, the delivery reading, the hindsight bound, what_was_not_done, the zip-truncated lists and the cut header line. {counts['must_fix']} must-fix, {counts['should_fix']} should-fix, {counts['notes']} notes.",
    scripts=[f"code/review/flow_state/{s}" for s in ("h_common.py", "h1_integrity.py", "h3_columns.py", "h2_headline.py", "h4_networks.py", "h5_outcome.py", "h6_assemble.py", "run_checks.sh", "README.md")],
    scripts_how_to_run="from the project root with the project environment, PYTHONUTF8=1, one torch thread: run_checks.sh (h1, h3, h2, h4, h5, h6 in this order; about a minute; outputs to REVIEW_OUT); h6_assemble.py --write-into-project writes this record",
    not_checked=["the three stored sets' Mac-trained networks (not reloaded, as the file says; their stored forecasts were rescored)", "the block bootstrap (not done by the experiment either)", "a run with provisional USGS values or with the real telemetry delay",
                 "the within-day AORC columns and the GEFS columns were taken from code/subdaily_inputs.py and data/gefs/basins (reviewed by the E4b and E2 reviews); only the hourly-flow columns were rebuilt here", "a sandbox rerun of 22_flow_state.py --rescore (the reload and scoring code was exercised by h2 and h4 instead)"],
    dates=dict(builder_training_pass="2026-10-29 12:11 to 12:38 (results file written 2026-10-29 12:38)", review="2026-10-29, afternoon"))
def clean(o):
    """Strict JSON: a number that is not finite becomes null (an undefined false share where a seed issued no warning)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


record = clean(record)
jsave("review_flow_state.json", record)
print(f"verdict: {verdict} | {counts}")
if WRITE:
    target = PROJ / "results" / "review_flow_state.json"
    target.write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
    print("written:", target)
else:
    print("dry run: record at", OUT / "review_flow_state.json", "(pass --write-into-project to write results/review_flow_state.json)")
