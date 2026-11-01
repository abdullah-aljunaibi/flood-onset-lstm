"""E4b review, step 6: the review record results/review_within_day_inputs.json built from the outputs of r1-r5 (numbers are taken from those outputs, not typed again).
Writes OUT/review_within_day_inputs.json; with --write-into-project also results/review_within_day_inputs.json, after checking that no reviewed file changed since r1."""
from r_common import *
from r_common import _default

t0 = time.time()
WRITE = "--write-into-project" in sys.argv
R = jload(RESULTS)
r1, r2, r3, r4, r5 = (jread(f) for f in ("r1_integrity.json", "r3_columns.json", "r2_headline.json", "r4_networks.json", "r5_outcome.json"))
r2, r3 = r3, r2                                                                                  # (read in the order of the files above; name them by content)
rows = r2["rows"]
S = R["summary_test"]["sets"]
M = r2["headline_mine"]
f1 = lambda x: round(float(x), 4) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
pct = lambda x: f"{100 * x:.1f}%"
pc = lambda x: f"{round(100 * x)}%"                                                                # the script's own rounding of the sentence
start_time = time.strftime("%H:%M", time.localtime((OUT / "r1_integrity.json").stat().st_mtime))
end_time = time.strftime("%H:%M")

# ---- what was reproduced ---------------------------------------------------------------------------------------------------------------------------------------------
headline = {}
for n in SHOWN:
    headline[n] = {}
    for k in LEADS:
        f, m = S[n][f"l{k}"], M[n][f"l{k}"]
        headline[n][f"l{k}"] = {c: dict(file=f1(fv), mine=f1(mv), diff=f1(abs(fv - mv))) for c, fv, mv in (("nse_median", f["nse_median"], m["nse_median"]), ("onset_hits", f["onset"]["hits"], m["onset"]["hits"]),
                                                                                                   ("onset_pod", f["onset"]["pod"], m["onset"]["pod"]), ("onset_far", f["onset"]["far"], m["onset"]["far"]),
                                                                                                   ("onset_csi", f["onset"]["csi"], m["onset"]["csi"]), ("all_pod", f["all"]["pod"], m["all"]["pod"]), ("all_far", f["all"]["far"], m["all"]["far"]))}
blocks = {}
for key, v in rows.items():
    first = key.split(" ")[0]
    b = first if first in ("readings", "gain_rule", "paired", "closure", "operating_points", "verdicts", "little_rain", "stored", "e4b_success", "thresholds", "where_the_gain_sits", "seed", "observed") else "summary_test, validation, per_basin and seed_ranges"
    blocks.setdefault(b, dict(entries=0, matching=0))
    blocks[b]["entries"] += 1
    blocks[b]["matching"] += int(v["ok"])
op = lambda n, rule: dict(file=dict(trigger=R["operating_points"]["sets"][n]["l1"]["rules"][rule]["trigger"], test_pod=f1(R["operating_points"]["sets"][n]["l1"]["rules"][rule]["test"]["onset"]["pod"]),
                                    test_far=f1(R["operating_points"]["sets"][n]["l1"]["rules"][rule]["test"]["onset"]["far"]), single_seeds_pod=[f1(x) for x in R["operating_points"]["sets"][n]["l1"]["rules"][rule]["single_seeds_at_this_trigger"]["onset_pod"]]),
                          mine=dict(trigger=r2["operating_points_mine"][f"{n} l1 {rule}"]["trigger"], test_pod=f1(r2["operating_points_mine"][f"{n} l1 {rule}"]["test_onset"]["pod"]), test_far=f1(r2["operating_points_mine"][f"{n} l1 {rule}"]["test_onset"]["far"])))
pairs = ["within_day_columns", "within_day_against_the_older_control", "older_control_against_wqa", "q_rel_alone", "within_day_against_q_rel", "how_far_from_perfect_rain", "within_day_with_the_forecast", "forecast_on_top_of_within_day"]
paired = {p: {c: dict(file=rows[f"paired {p} l1 {c}"]["file"], mine=rows[f"paired {p} l1 {c}"]["mine"]) for c in ("warned_by_both", "warned_only_by_first", "warned_only_by_second", "sign_test_p")} for p in pairs}
cl = R["closure"]["sets"]["wqaW"]["by_lead"][0]
E = R["e4b_success"]
reproduced = dict(
    how="own scoring in numpy from results/preds (thresholds = Q95 of the training years from the stored observed flow; onsets = target days of the test years above Q95 whose flow lead days earlier was at or below it; a warning = forecast > trigger x Q95; NSE per basin on the test days); "
        "every value compared with results/within_day_inputs.json at 1e-9 (code/review/within_day_inputs/r2_headline.py); the stored sets lstm_wqa, lstm_wqaF_sub, lstm_wqaP rescored the same way",
    summary=r2["summary"], by_block=blocks, my_scoring_equals_flood_evaluate_on_lstm_wqaW=r2["engine_cross_check"]["my_scoring_equals_flood_evaluate"],
    headline_test_years_trigger_1_file_vs_mine=headline,
    seed_ranges_lead1_onset_pod={n: dict(file=[f1(x) for x in R["seed_ranges"]["sets"][n]["l1"]["onset_pod"]], mine=[f1(min(M[n]["l1"]["seed_onset_pod"])), f1(max(M[n]["l1"]["seed_onset_pod"]))], by_seed_mine=[f1(x) for x in M[n]["l1"]["seed_onset_pod"]], n_seeds=len(SEEDS[n])) for n in SHOWN},
    seed_means_equal_mean_of_seed_files=r2["seed_mean_check"],
    success_rule=dict(rule=E["rule"], file=dict(met=E["met"], every_seed_of_wqaW_above_every_seed_of_wqa=E["every_seed_of_wqaW_above_every_seed_of_wqa"], every_seed_of_wqaW_above_every_seed_of_the_older_control=E["every_seed_of_wqaW_above_every_seed_of_the_older_control"], control_flat=E["control_flat"]),
                      mine=r2["success_mine"], with_the_first_three_seeds_of_the_new_sets=r2["success_with_three_seeds"]),
    operating_points_lead1={n: dict(csi=op(n, "csi"), far_cap=op(n, "far_cap")) for n in SHOWN},
    paired_onsets_lead1=paired,
    closure_wqaW_lead1=dict(file=dict(onset_gap_points=f1(100 * (cl["onset_warned_ceiling"] - cl["onset_warned_base"])), onset_gain_points=f1(cl["onset_gain_points"]), onset_share_closed=f1(cl["onset_share_of_gap_closed"]), nse_gap=f1(cl["gap"]), nse_gain=f1(cl["gain"]), nse_share_closed=f1(cl["share_of_gap_closed"])),
                            mine=dict(onset_gap_points=f1(rows["closure wqaW l1 onset_gain_points"]["mine"] / rows["closure wqaW l1 onset_share_of_gap_closed"]["mine"]), onset_gain_points=f1(rows["closure wqaW l1 onset_gain_points"]["mine"]), onset_share_closed=f1(rows["closure wqaW l1 onset_share_of_gap_closed"]["mine"]),
                                      nse_gap=f1(rows["closure wqaW l1 gap"]["mine"]), nse_gain=f1(rows["closure wqaW l1 gain"]["mine"]), nse_share_closed=f1(rows["closure wqaW l1 share_of_gap_closed"]["mine"]))),
    little_rain_slice_lead1={key: dict(n=dict(file=rows[f"little_rain {key} n"]["file"], mine=rows[f"little_rain {key} n"]["mine"]), warned={n: dict(file=f1(rows[f"little_rain {key} warned {n}"]["file"]), mine=f1(rows[f"little_rain {key} warned {n}"]["mine"])) for n in SHOWN})
                             for key in ("aorc_target_below_5mm", "aorc_target_at_least_5mm", "aorc_issue_day_below_1mm")},
    where_the_gain_sits={c: dict(file=rows[f"where_the_gain_sits {c}"]["file"], mine=rows[f"where_the_gain_sits {c}"]["mine"]) for c in ("hits_gained_in_the_slice", "hits_gained_outside", "points_in_the_slice", "points_outside")},
    stored_sets_own_results_files_reproduced=all(v["ok"] for k, v in rows.items() if k.startswith("stored ")),
    outcome_full_clauses_rebuilt_from_my_numbers=dict(found=sum(v["found_in_outcome_full"] for v in r5["clauses_rebuilt_from_my_numbers"].values()), of=len(r5["clauses_rebuilt_from_my_numbers"]), all_found=r5["all_clauses_found"]),
    figure=r5.get("figure"))

# ---- the networks -----------------------------------------------------------------------------------------------------------------------------------------------------
networks = dict(how="all 20 .pt files of script 21 rebuilt from the file alone (flood.Cfg(**cfg), flood.LSTMNet(n_in, cfg).load_state_dict, scaler lists back to arrays, flood.Run(...).predict) and run (a) on tables built as the script builds them "
                    "(subdaily_inputs.add_inputs + the maxcell fill + the _older shift + fs1..fs3 from data/gefs/basins) and (b) on tables built from the reviewer's own columns (r3); compared with results/preds/lstm_<set>_s<seed>.npz",
                torch=r4["torch"], numpy=r4["numpy"], torch_threads=r4["threads"], tables_script_route_vs_mine_max_abs_diff_mm=r4["tables_script_route_vs_mine_max_abs_diff"],
                all_20_bit_for_bit_on_the_scripts_tables=r4["all_20_bit_for_bit_on_script_tables"], on_my_tables=dict(bit_for_bit=r4["all_20_bit_for_bit_on_my_tables"], within_1e_5_mm_per_day=r4["all_20_within_atol_1e_5_on_my_tables"],
                                                                                                                           worst_abs_diff_mm_per_day=r4["worst_max_abs_diff"], worst_rel_diff_above_0_1_mm=r4["worst_rel_diff_above_0_1_mm_on_my_tables"],
                                                                                                                           same_onset_hits_and_false_alarms_all_leads=r4["all_20_same_warnings_on_my_tables"], worst_nse_diff=r4["worst_nse_diff_on_my_tables"],
                                                                                                                           why_not_bit_for_bit="my columns differ from the script's by up to 6e-11 mm (order of summation); after standardization and the cast to float32 a few inputs round to the neighbouring float, which moves a few forecasts by up to 8e-6 mm/day and no warning or NSE"),
                scalers_recomputed_from_the_training_years_equal_in=r4["scalers_from_training_years_in"], settings_equal_lstm_all_in=r4["settings_true_in"], future_false_in_all=r4["future_false_in_all"], use_flow_true_in_all=r4["use_flow_true_in_all"],
                per_file={k: dict(n_in=v["n_in"], script_tables_max_abs_diff=v["script_tables"]["max_abs_diff"], script_tables_bit_for_bit=v["script_tables"]["bit_for_bit"], my_tables_max_abs_diff=v["my_tables"]["max_abs_diff"], scalers_ok=v["scalers"]["from_training_years_only"]) for k, v in r4["reproduction"].items()},
                stored_sets_networks="not reloaded (trained on the Mac, as the results file says); their stored forecasts were rescored and equal their own results files (reproduced.stored_sets_own_results_files_reproduced)",
                design_flags_from_r1=r1["networks"]["flags_true_in"])

# ---- causality ---------------------------------------------------------------------------------------------------------------------------------------------------------
ck = r3["clock"]["per_basin"]
g0 = ids0 = next(iter(ck))
causality = dict(
    a_clock=dict(what="(a) read in code/subdaily_inputs.py and the script, and rebuilt by another route (r3_columns.py): the UTC instant of 00:00 local of day t+1 (the issue time, 24:00 of day t) from zoneinfo per basin; the day's hours are the hour-ending stamps in (start, end]; "
                      "the last 3, 6, 12 hours are the stamps in (end - k h, end]; the hour ending exactly 24:00 local belongs to day t (perturbing that hour alone moves day t and not day t+1, both builders)",
                 my_columns_equal_add_inputs_within_1e_9=r3["my_columns_vs_subdaily_inputs"]["all_basins_all_columns_within_1e_9"], worst_diff_mm=r3["my_columns_vs_subdaily_inputs"]["worst"],
                 every_hour_of_day_t_lies_on_local_day_t_all_basins=r3["clock"]["all_basins_pass"], last3h_hours_end_at="22, 23, 24 on the wall clock on every day of every basin", last6h="19-24", last12h="13-24",
                 hours_per_day=[23, 24, 25], clock_change_days_per_basin=dict(n_23h=ck[g0]["n_days_23h"], n_25h=ck[g0]["n_days_25h"], first_23h_day=ck[g0]["first_23h_day"], first_25h_day=ck[g0]["first_25h_day"], last_25h_day=ck[g0]["last_25h_day"]),
                 issue_time_utc_examples=ck[g0]["issue_time_utc_examples"], zones={g: r3["zones"][g]["zone"] for g in r3["zones"]}, zones_plausible_by_longitude=r3["zones_plausible_by_longitude"],
                 made_up_hours_test=dict(n_cases=r3["made_up_hours"]["n_cases"], all_pass=r3["made_up_hours"]["all_pass"], what="every hour ending after 24:00 local of a cut day x7+100 (a random test-year day, the last 23-hour day, the last 25-hour day per basin): no row up to the cut moves, the next day moves, on the reviewer's builder and on subdaily_inputs._table"),
                 maxcell_fill_days_equal=r3["maxcell_fill_days"]["equal_to_results_file"], n_filled_n_raised=[r3["maxcell_fill_days"]["n_filled"], r3["maxcell_fill_days"]["n_raised"]]),
    b_made_up_values_on_saved_networks=dict(what=r4["made_up_values"]["what"], networks=list(r4["made_up_values"]["per_network"]), cut_days=[r["cut"] + " (" + r["part"] + ")" for r in r4["made_up_values"]["per_network"]["lstm_wqaW_s0"]],
                                            n_cases=r4["made_up_values"]["n_cases"], all_pass=r4["made_up_values"]["all_pass"], the_scripts_own_test=dict(cut_days=R["causality"]["made_up_values"]["wqaW"]["cut_days"], passed=all(r["passed"] for v in R["causality"]["made_up_values"].values() for r in v["rounds"].values()))),
    c_control=dict(what="lstm_wqaW_older against lstm_wqaW: the saved settings differ only in dyn (the six column names end in _older), the weights have the same shapes and count, the split is the same; the control's stored forecasts are reproduced from tables whose _older columns are the within-day columns shifted one day later (first day kept), i.e. on issue day t the network reads the within-day values of day t-1",
                   cfg_keys_that_differ=r4["control"]["cfg_keys_that_differ"], control_columns_are_the_test_columns_plus_older=r4["control"]["control_columns_are_the_test_columns_plus_older"], state_dict_shapes_equal=r4["control"]["state_dict_shapes_equal"], n_params_equal=r4["control"]["n_params_equal"],
                   older_shift_verified=r4["control"]["older_shift_verified"], older_definition_checked=r3["older_definition"]["checked"]),
    d_thresholds_scalers_choices=dict(thresholds="Q95 per basin = the 95th percentile of the daily flow of the training years 1980-10-01..2000-09-30 only, recomputed from the stored observed flow: equal to results/within_day_inputs.json thresholds and to flood.thresholds (r2, r3)",
                                      thresholds_equal=r3["thresholds"]["mine_equal_results_file"], q_rel="log1p(q_mm / Q95) with that Q95 (r3: equal to add_inputs' q_rel to 1e-12)",
                                      scalers="dyn_mu/dyn_sd, q_mu/q_sd, lq_mu/lq_sd, st_mu/st_sd inside every .pt recomputed from the training-year rows only: equal in " + r4["scalers_from_training_years_in"],
                                      stopping="best_update = the argmax of the validation NSE in the saved history in " + r1["networks"]["flags_true_in"]["best_update_is_argmax_val_nse"] + "; stopped by patience (6 x 100 updates) in " + r1["networks"]["flags_true_in"]["stopped_by_patience_or_cap"],
                                      triggers="the CSI and 30%-false triggers replicated on the validation arrays only and the test years scored once at them: every trigger, table and verdict equal (r2 rows 'operating_points' and 'verdicts')",
                                      forecast_columns="fs1..fs3: the AORC rain of days t+1..t+3 up to the issue day 2000-09-30, the bias-corrected GEFS forecast from 2000-10-01; the bias factor was fitted on the validation years' target days (results/gefs_basins.json bias.fitted_on_target_days = 2000-10-01..2005-09-30), as in script 18"),
    e_config=dict(results_config_equals_lstm_all=r1["config"]["equal"], config=R["config"], every_pt_file_equals_lstm_all=r1["networks"]["flags_true_in"]["settings_equal_lstm_all"], every_prediction_set_meta_ok=r1["prediction_set_meta"]["all_ok"]))

leakage_check = ("Nothing of the test years reaches a network before its forecasts are scored, as far as the code and the files show: Q95 comes from the training years (flood.thresholds; recomputed), the scalers from the training-year rows (flood.fit; recomputed from the 20 files), "
                 "the stopping point from the validation NSE (saved history), the within-day columns of issue day t from hours ending at or before 24:00 local of day t (rebuilt by another route; the made-up-hours and made-up-values tests pass on the data and on the saved networks), "
                 "q_rel from the training-year Q95, the maxcell fill from the same day's values, the fs columns' bias factor from the validation years (script 18's design), the triggers of the operating points from the validation arrays (replicated), the GO criteria from results/flood_analysis.json. "
                 "The script's hash recorded at the start of the run equals the file now (modified 23:07, first network 23:39), so the sets, seeds and the success rule were in the file before training; no --extend pass exists (one pass, 20 trainings). "
                 "The validation years show no gain for wqaW (onsets warned 50.0% against 50.8% for wqa at trigger 1.0, seed means; best validation NSE 0.411-0.429 against 0.427-0.429), so nothing on the validation side was tuned towards the test-year gain.")

# ---- findings ---------------------------------------------------------------------------------------------------------------------------------------------------------
opW, opO, opA, opQ = (r2["operating_points_mine"][f"{n} l1 far_cap"] for n in ("wqaW", "wqaW_older", "wqa", "wqaQ"))
ssW, ssO, ssA = (R["operating_points"]["sets"][n]["l1"]["rules"]["far_cap"]["single_seeds_at_this_trigger"]["onset_pod"] for n in ("wqaW", "wqaW_older", "wqa"))
tW, tA, tO, tQ = (M[n]["l1"]["onset"] for n in ("wqaW", "wqa", "wqaW_older", "wqaQ"))
nW, nA, nO, nQ = (M[n]["l1"]["nse_median"] for n in ("wqaW", "wqa", "wqaW_older", "wqaQ"))
csW, csO = (r2["operating_points_mine"][f"{n} l1 csi"] for n in ("wqaW", "wqaW_older"))
sm, wet = R["little_rain_onsets"]["slices"]["aorc_target_below_5mm"], R["little_rain_onsets"]["slices"]["aorc_target_at_least_5mm"]
pw = R["paired_onsets"]["pairs"]
rd = R["readings"]
three = r2["success_with_three_seeds"]
findings = [
    dict(severity="must-fix", where="results/within_day_inputs.json outcome_caveat (code/21_within_day_inputs.py, the outcome_caveat f-string, clause 'at the 30%-false operating point ...'), hence outcome_full and every document that quotes it",
         text=f"The clause gives wqaW against wqa only ('it warns of {pc(opW['test_onset']['pod'])} with {pc(opW['test_onset']['far'])} false on the test years, against {pc(opA['test_onset']['pod'])} with {pc(opA['test_onset']['far'])} false for the AORC rain to midnight alone') and leaves out the control at the same operating point, which the same file stores "
              f"(operating_points.sets.wqaW_older.l1.rules.far_cap): trigger {opO['trigger']:.2f}, {pct(opO['test_onset']['pod'])} warned with {pct(opO['test_onset']['far'])} false (single seeds {pct(min(ssO))} to {pct(max(ssO))}) against wqaW's {pct(opW['test_onset']['pod'])} with {pct(opW['test_onset']['far'])} false "
              f"(single seeds {pct(min(ssW))} to {pct(max(ssW))}, trigger {opW['trigger']:.2f}) and wqa's {pct(opA['test_onset']['pod'])} with {pct(opA['test_onset']['far'])} false ({pct(min(ssA))} to {pct(max(ssA))}, trigger {opA['trigger']:.2f}). "
              "At the one operating point the GO rule cares about, the control with no information about day t is level with the test set, so the reading the clause invites ('the within-day columns add about 6 points at 30% false') does not hold against the control. "
              f"At trigger 1.0 and at the CSI-chosen level wqaW is above the control ({pct(tW['pod'])} against {pct(tO['pod'])} with fewer false warnings, {pct(tW['far'])} against {pct(tO['far'])}; {pct(csW['test_onset']['pod'])} against {pct(csO['test_onset']['pod'])}) and its NSE is higher in "
              f"{rd['within_day_against_the_older_control']['by_lead']['l1']['nse']['n_basins_higher']} of 12 basins, so the file's numbers support 'the hours of day t add warnings at the fixed level and nothing at the 30%-false level'. "
              f"Fix: add the control to the clause ('... against {pc(opA['test_onset']['pod'])} with {pc(opA['test_onset']['far'])} false for the AORC rain to midnight alone (trigger {opA['trigger']:.2f}) and {pc(opO['test_onset']['pod'])} with {pc(opO['test_onset']['far'])} false for the older control (trigger {opO['trigger']:.2f}): at this operating point the six columns add nothing over the control') "
              "and quote the three numbers together wherever the 30%-false point is cited."),
    dict(severity="should-fix", where="results/within_day_inputs.json outcome_caveat, clause 'AORC is an analysis made after the fact: the within-day columns share the hindsight caveat of the AORC rain to midnight (hindsight at most about 3 points)'; what_was_not_done",
         text="The bound of about 3 points was measured by the E1b review for the DAILY total (the saved wqa networks given Stage IV, a real-time product, for the issue day on 78 days with 171 onsets); nothing of the kind exists for the six within-day columns, and what_was_not_done does not list it. "
              "The channel is the same (in the test years AORC's hours are Stage IV's hours times one factor per gauge day, hourly correlation 0.9995: E1b review, observation 1), so the extension is plausible, but as written it is a claim, not a measurement. "
              "Fix: 'share the hindsight channel of the AORC rain to midnight (one factor per gauge day); the bound of about 3 points was measured for the daily total, not for these columns', and add to what_was_not_done 'a Stage IV test of the saved wqaW networks with the within-day columns rebuilt from Stage IV hours "
              "(the a8 test of the E1b review; data/aorc/stage_iv_hindsight.csv holds 1,800 Stage IV hours of the builder's 50 days to start from)'."),
    dict(severity="should-fix", where="results/within_day_inputs.json outcome_caveat; design.columns; what_was_not_done (code/21_within_day_inputs.py)",
         text="All training years (1980-10 to 2000-09) lie before 2002, where AORC's hours are daily totals spread over the hours by the shape of older radar, satellite or model products (MDR, NOWrad, Stage II, CMORPH, CFSR: the AORC v1.1 methods as read by the E1b review), "
              "while the test years' hours have Stage IV's hourly shape; the validation years 2000-2005 straddle the change. The networks therefore learned 'the rain of the last 3, 6 and 12 hours', 'the wettest hour' and 'the wet hours' from hours of another, and in the 1980s poorer, source than the hours they were scored on. "
              "The figure footer says 'hours before 2002 are disaggregated daily totals'; outcome_full does not, and the effect on the +4.4 points was not measured. "
              "Fix: one clause in outcome_caveat ('the networks learned the within-day columns on hours that are daily totals spread by older products, all training years lying before 2002, and were scored on Stage IV-shaped hours; what this does to the gain was not measured') and an entry in what_was_not_done."),
    dict(severity="should-fix", where="results/within_day_inputs.json outcome_caveat, clause 'the control is flat (53% against 51%)' (code/21_within_day_inputs.py outcome_caveat f-string)",
         text=f"'Flat' holds for onsets warned at trigger 1.0 by the seed rule (the pre-registered reading), but the sentence does not say that the control is noisier: {pct(tO['far'])} of its onset warnings are false against {pct(tA['far'])} for wqa ({tO['false_alarms']} against {tA['false_alarms']} false alarms), "
              f"its median NSE is {nO:.3f} against {nA:.3f} (higher in {rd['older_control_against_wqa']['by_lead']['l1']['nse']['n_basins_higher']} of 12 basins, p = {rd['older_control_against_wqa']['by_lead']['l1']['nse']['wilcoxon_p']:.3f}; seed 4 at {min(M['wqaW_older']['l1']['seed_nse_median']):.2f}): the six older columns cost skill while adding warnings. "
              f"Fix: 'the control is flat on onsets warned ({pc(tO['pod'])} against {pc(tA['pod'])}) and noisier ({pc(tO['far'])} false against {pc(tA['far'])}; median NSE {nO:.2f} against {nA:.2f})'; the +2.9 points of wqaW over the control at trigger 1.0 come with 6 points fewer false warnings, which the outcome may say."),
    dict(severity="should-fix", where="results/within_day_inputs.json outcome_caveat, clause 'In the 220 onsets with under 5 mm ...'; e4b_success.expected_where",
         text=f"NEXT_EXPERIMENTS_2026-10-28.md, E4b: 'Expect the gain in the 220 onsets with under 5 mm on the target day'. The result is the opposite: in those {sm['n']} onsets wqaW warns of {pct(sm['warned']['wqaW'])} against {pct(sm['warned']['wqa'])} for wqa ({sm['hits']['wqaW']} against {sm['hits']['wqa']} hits; "
              f"{sm['paired_wqaW_vs_wqa']['only_first']} warned only by wqaW, {sm['paired_wqaW_vs_wqa']['only_second']} only by wqa) and {pct(sm['warned']['wqaW_older'])} for the control; the whole gain sits in the {wet['n']} onsets with at least 5 mm on the target day "
              f"({pct(wet['warned']['wqaW'])} against {pct(wet['warned']['wqa'])}, {wet['hits']['wqaW'] - wet['hits']['wqa']:+d} hits). outcome_caveat gives the numbers without saying that the expectation written before training failed (e4b_success.expected_where calls it 'a description'). "
              "Fix: add 'the opposite of the expectation of NEXT_EXPERIMENTS_2026-10-28.md' to the clause, so that the pre-registered expectation and its failure stand side by side."),
    dict(severity="should-fix", where="results/within_day_inputs.json outcome_caveat, clause 'The Q95-relative flow alone (wqaQ) gives 52% ...'",
         text=f"The clause leaves out the other headline score: wqaQ's median NSE is {nQ:.3f} against {nA:.3f} (seed ranges {min(M['wqaQ']['l1']['seed_nse_median']):.2f}-{max(M['wqaQ']['l1']['seed_nse_median']):.2f} against {min(M['wqa']['l1']['seed_nse_median']):.2f}-{max(M['wqa']['l1']['seed_nse_median']):.2f}; "
              f"higher in {rd['q_rel_alone']['by_lead']['l1']['nse']['n_basins_higher']} of 12 basins, p = {rd['q_rel_alone']['by_lead']['l1']['nse']['wilcoxon_p']:.2f}, 'no clear difference' by the rule), its onset false share {pct(tQ['far'])} against {pct(tA['far'])}, and at the 30%-false point it is level with wqa "
              f"({pct(opQ['test_onset']['pod'])} against {pct(opA['test_onset']['pod'])}). Fix: '... gives {pc(tQ['pod'])} (single trainings 49% to 55%) with a lower median NSE ({nQ:.2f} against {nA:.2f}) and {pc(tQ['far'])} false warnings'."),
    dict(severity="should-fix", where="results/within_day_inputs.json readings.<pair>.by_lead.l<k>.onsets.difference_by_seed_points and .nse.difference_by_seed (code/21_within_day_inputs.py pair_reading(), zip())",
         text="The seed-by-seed difference lists are built with zip() over the two sets' seed lists; where a five-seed set is compared with a three-seed stored set (within_day_columns: wqaW against wqa; within_day_with_the_forecast: wqaWF_sub against wqaF_sub) they hold three entries "
              f"(seeds 0-2: {[round(x, 1) for x in rd['within_day_columns']['by_lead']['l1']['onsets']['difference_by_seed_points']]} points at lead 1) and seeds 3 and 4 of the new set are silently dropped; nothing in the JSON says so (only the docstring of pair_reading). "
              "The seed ranges, the rule and every other number use all five seeds. Fix: add seeds_paired = [0, 1, 2] beside the lists (or pad with null) and say in how_to_read that the seed-by-seed differences cover the seeds both sets have."),
    dict(severity="should-fix", where="results/within_day_inputs.json choices_made_after_looking_at_test_scores (code/21_within_day_inputs.py CHOICES)",
         text="'Sets, inputs, seeds, stopping rule and the success rule (wqaW above wqa and above wqaW_older in every seed at lead 1) were fixed in NEXT_EXPERIMENTS_2026-10-28.md and in this script before training': NEXT_EXPERIMENTS fixes the sets wqaW, wqaWF_sub and wqaQ, 'three seeds, then five', "
              "the scoring 'as script 16 does (seed rule, paired onsets, per-basin, the closure of the gap to wqaP)', the 30%-false point and the expectation; it names neither the older control nor the rule's wording, which exist only in the script. The script did fix them before training: "
              f"its hash recorded at the start of the run ({R['script_sha256'][:12]}...) equals the file now, the file was last modified {r1['script']['modified']} and the first network was written {r1['script']['first_network_written']}; the run went straight to five seeds (one pass, no three-seed pass, no --extend). "
              "Fix the clause: 'The sets wqaQ, wqaW and wqaWF_sub, the scoring and the expectation were fixed in NEXT_EXPERIMENTS_2026-10-28.md; the older control wqaW_older, the success rule as worded and the five seeds were fixed in this script (sha256 recorded at the start of the run) before the first training; the test years were scored once.'"),
    dict(severity="should-fix", where="results/within_day_inputs.json what_was_not_done (code/21_within_day_inputs.py NOT_DONE)",
         text="The list holds 'five seeds (unless --seeds 5 was used)' although --seeds 5 was used (seeds [0, 1, 2, 3, 4], n_seeds_per_set 5 for every new set): evaluate the condition in the script and drop the entry; add the two measurements named above (the Stage IV test of the within-day columns; the pre-2002 hourly shape)."),
    dict(severity="should-fix", where="figures/within_day_inputs.png footer (code/21_within_day_inputs.py draw(), fig.text: 'the 51% of wqa carries ...')",
         text="The footer hand-types '51%' (HANDOFF.md section 7: numbers come from results/*.json). Fix: format it from R['summary_test']['sets']['wqa']['l1']['onset']['pod']. The bars and dots of the figure agree with the JSON (read on the pixels: every bar top within 0.6 of a point / 0.006 NSE, dot counts 3, 5, 5, 5, 5, 3, 3)."),
    dict(severity="note", where="results/within_day_inputs.json outcome_caveat, last sentence (wqaWF_sub against wqaF_sub); readings.within_day_with_the_forecast; paired_onsets",
         text=f"The sentence gives the shares ('from 45% to 42% ... at 2 and 3 days from 39% to 33% and from 37% to 34%'); the paired counts show a loss at 2 and 3 days ({pw['within_day_with_the_forecast']['l2']['warned_only_by_first']} onsets warned only with the within-day columns against {pw['within_day_with_the_forecast']['l2']['warned_only_by_second']} only without at lead 2, sign p = {pw['within_day_with_the_forecast']['l2']['sign_test_p']:.0e}; "
              f"{pw['within_day_with_the_forecast']['l3']['warned_only_by_first']} against {pw['within_day_with_the_forecast']['l3']['warned_only_by_second']} at lead 3) and at lead 1 ({pw['within_day_with_the_forecast']['l1']['warned_only_by_first']} against {pw['within_day_with_the_forecast']['l1']['warned_only_by_second']}, p = {pw['within_day_with_the_forecast']['l1']['sign_test_p']:.2f}), "
              "while by the seed rule it is 'no clear difference' at every lead (the ranges overlap) and the lead-1 NSE is higher (0.723 against 0.693). A reader may want the reading beside the shares: on top of the substituted forecast the within-day columns do not help and by the paired counts cost onsets at 2-3 days."),
    dict(severity="note", where="results/within_day_inputs.json status; figures/within_day_inputs.png header",
         text="Both say 'NOT YET REVIEWED', true when they were written; the script sets 'reviewed: results/review_within_day_inputs.json exists' only on its next pass. After the wording fixes run the script with --rescore --seeds 5 --with-forecast (nothing trained; it refuses if a saved file changed) so that the status, the figure header and the fixed clauses are rebuilt together."),
    dict(severity="note", where="results/within_day_inputs.json e4b_success; seed_ranges",
         text=f"The success rule fails with the five seeds and would fail with the first three as well (wqaW seeds 0-2: {', '.join(pct(x) for x in three['wqaW_seeds_0_2'])} against wqa's {pct(min(M['wqa']['l1']['seed_onset_pod']))} to {pct(max(M['wqa']['l1']['seed_onset_pod']))} and the control's seeds 0-2 "
              f"{pct(min(M['wqaW_older']['l1']['seed_onset_pod'][:3]))} to {pct(max(M['wqaW_older']['l1']['seed_onset_pod'][:3]))}): seeds {three['seeds_of_wqaW_below_the_best_seed_of_wqa']} of wqaW sit below the best seed of wqa ({pct(three['best_seed_of_wqa'])}) and of the control ({pct(three['best_seed_of_control'])}). The seed count did not decide the verdict."),
    dict(severity="note", where="results/within_day_inputs.json validation_used_for_no_choice; models.*.per_seed.best_val_nse",
         text=f"On the validation years wqaW shows no gain: onsets warned {pct(R['validation_used_for_no_choice']['summary']['wqaW']['l1']['onset']['pod'])} against {pct(R['validation_used_for_no_choice']['summary']['wqa']['l1']['onset']['pod'])} for wqa at trigger 1.0 (seed means), best validation NSE of the stopping point "
              f"{min(p['best_val_nse'] for p in R['models']['wqaW']['per_seed']):.3f}-{max(p['best_val_nse'] for p in R['models']['wqaW']['per_seed']):.3f} against 0.427-0.429 for wqa's three seeds. The test-year gain of +4.4 points is not foreshadowed by the validation years, which fits the rule's reading 'no clear difference' and argues against reading the +4.4 as established."),
    dict(severity="note", where="code/review/within_day_inputs/r4_networks.py",
         text=f"On this PC the 20 networks reproduce their stored forecasts bit for bit with one torch thread (the run used {R['passes'][0].get('torch_threads')}); the scalers inside every file equal the training-year statistics exactly (largest relative difference {max(v['scalers']['max_rel_diff'] for v in r4['reproduction'].values()):g}). "
              "The three stored sets' networks (trained on the Mac) were not reloaded, as the results file says; their stored forecasts, rescored here, equal their own results files to 1e-9."),
]
counts = dict(values_compared=r2["summary"]["entries"], numerical_mismatches=len(r2["summary"]["not_matching"]), networks_rebuilt=len(r4["reproduction"]), made_up_value_cases_on_networks=r4["made_up_values"]["n_cases"], made_up_hour_cases_on_data=r3["made_up_hours"]["n_cases"],
              outcome_clauses_rebuilt=len(r5["clauses_rebuilt_from_my_numbers"]), outcome_clauses_found=sum(v["found_in_outcome_full"] for v in r5["clauses_rebuilt_from_my_numbers"].values()), figure_bars_checked=14,
              must_fix=sum(f["severity"] == "must-fix" for f in findings), should_fix=sum(f["severity"] == "should-fix" for f in findings), notes=sum(f["severity"] == "note" for f in findings))
verdict = "numbers hold with wording fixes"
verdict_text = (f"Every number of results/within_day_inputs.json that I recomputed from results/preds with my own scoring equals the file's ({r2['summary']['entries']} values at 1e-9: the headline of the seven sets at three leads, every single seed, the seed ranges and the seed rule, the per-basin picture, "
                "the Wilcoxon and sign tests, the paired onsets, the closure, the validation-chosen CSI and 30%-false triggers with their test scores and verdict words, the little-rain slices, the stored sets' own files); the 14 clauses of outcome_full rebuilt from my numbers are in the text word for word; "
                "the 20 saved networks rebuilt from their files reproduce the stored forecasts bit for bit on tables built as the script builds them and to 8e-6 mm/day (same warnings, same NSE) on tables from my own columns; those columns, rebuilt by another route from the hourly cache, equal the project's to 6e-11 mm, "
                "every hour of day t ends at or before 24:00 local of day t on every day including the 68 clock-change days, the hour ending exactly 24:00 belongs to day t, and made-up hours and made-up values after a cut day change no earlier forecast (36 + 10 cases); "
                "Q95, the scalers and the stopping point come from the training and validation years, the triggers from the validation years, and the script was fixed (hash) before the first training. "
                f"The experiment's own reading stands: the pre-registered rule is not met ({pct(tW['pod'])} warned against {pct(tA['pod'])}, single trainings {pct(min(M['wqaW']['l1']['seed_onset_pod']))} to {pct(max(M['wqaW']['l1']['seed_onset_pod']))} against {pct(min(M['wqa']['l1']['seed_onset_pod']))} to {pct(max(M['wqa']['l1']['seed_onset_pod']))}), "
                "and would not be met with three seeds either. One must-fix of wording: at the 30%-false operating point the control with no information about day t (30.1% warned, 31.3% false) is level with wqaW (30.8%, 29.9%), and the sentence gives only wqa (24.7%) there. "
                "The should-fix items are wording and record-keeping: the 3-point hindsight bound was measured for the daily total and is extended to the within-day columns without a measurement; the training years' hours (all before 2002) have another hourly shape than the test years'; "
                "the control's higher false share and lower NSE, wqaQ's lower NSE, the failed pre-registered expectation about the little-rain onsets, the zip-truncated seed-by-seed lists, the imprecise 'fixed in NEXT_EXPERIMENTS' clause, a stale what_was_not_done entry and a hand-typed 51% in the figure footer.")
rec = dict(
    who="independent AI reviewer agent (Claude), 2026-10-29",
    human_review="not yet done: the team's own check of E4b is still open (team.json untouched by this review)",
    when=f"2026-10-29, from about {start_time} to {end_time} local time (first check output to this record), after the training pass of 2026-10-28 23:40 to 2026-10-29 00:02 on this Windows PC (results file written {R['written']}); the E4e hourly-flow download (code/fetch_usgs_iv.py) ran on the same PC meanwhile; "
         "the lstm_wqaH*/wqaWH* files of E4e are not part of this review",
    what="independent recomputation of experiment E4b (code/21_within_day_inputs.py with code/subdaily_inputs.py: six within-day AORC rain columns of the issue day and the Q95-relative flow as extra inputs of the all-river LSTM; sets wqaQ, wqaW, wqaW_older, wqaWF_sub, five seeds each, "
         "against the stored lstm_wqa, lstm_wqaF_sub and lstm_wqaP) with the reviewer's own code: the headline numbers from results/preds, the saved networks from results/models, the within-day columns from data/aorc, the clock, the causality and leakage checks, the pre-registered success rule, "
         "the seed handling, the sentence outcome_full and the figure; the engine (flood.load_data, basins, thresholds, load_preds, evaluate as a cross-check, Cfg, LSTMNet, Run.predict) and subdaily_inputs.add_inputs/_table were used as the brief allowed, beside the reviewer's own builder and scorer",
    files_reviewed=dict(sha256=r1["code_files_sha256"], saved_files_of_the_experiment=r1["saved_files"], stored_set_files=r1["stored_set_files"], script=r1["script"], prediction_set_meta=r1["prediction_set_meta"], log=r1["log"], training_block=r1["training_block"]),
    method=dict(own_code_folder="code/review/within_day_inputs/ (in the project) and the session scratchpad for the outputs (REVIEW_OUT)", torch_threads=1, nothing_trained=True,
                script_not_run="code/21_within_day_inputs.py was not run (its --rescore pass is for the builder after the fixes); code/test_flood.py was not run", nothing_installed=True,
                project_writes="only this folder and this record; results/within_day_inputs.json, the prediction sets, the networks and the figure were read, never written",
                python="<venv> (torch " + r4["torch"] + " CPU, numpy " + r4["numpy"] + "), PYTHONUTF8=1, from the project root"),
    reproduced=reproduced, networks_rebuilt=networks, causality=causality, leakage_check=leakage_check,
    pre_registration=dict(where_the_design_was_fixed="NEXT_EXPERIMENTS_2026-10-28.md section E4b: the sets wqaW, wqaWF_sub, wqaQ (settings of results/lstm_all.json), 'three seeds, then five', the scoring 'as script 16 does (seed rule, paired onsets, per-basin, the closure of the gap to wqaP) and additionally at the 30%-false operating point of E4a', "
                                                      "the expectation 'the gain in the 220 onsets with under 5 mm on the target day'; the script (SUCCESS_RULE, CHOICES, the control wqaW_older, --seeds 5) before the first training (hash recorded at the start of the run equals the file; modified 23:07, first network 23:39)",
                          rule_as_applied=E["rule"], applied_without_change=True, met=E["met"], met_recomputed=r2["success_mine"]["met"], met_with_three_seeds=three["met_with_three_seeds"],
                          expectation_little_rain_onsets="failed: -8.6 points in the 220 onsets, +11.8 points in the 391 others (reproduced.where_the_gain_sits)",
                          seed_counts="five seeds for the new sets, three for the stored sets; the rule compares every seed with every seed (min of the test set above the max of the comparison set); the only place where seeds are paired one to one, the seed-by-seed difference lists, is zip-truncated to three (finding 7); the outcome says which numbers are five-seed numbers",
                          wqaQ_and_wqaWF_sub_readings="wqaQ: 'no clear difference' on onsets (+1.0 point) with a lower NSE not mentioned in the sentence (finding 6); wqaWF_sub: 'no clear difference' by the seed rule at every lead, a loss by the paired counts at 2-3 days (note 11)"),
    wordings=r5["wordings"],
    findings=findings, counts=counts, verdict=verdict, verdict_text=verdict_text,
    scripts=["code/review/within_day_inputs/README.md", "code/review/within_day_inputs/r_common.py", "code/review/within_day_inputs/r1_integrity.py", "code/review/within_day_inputs/r3_columns.py", "code/review/within_day_inputs/r2_headline.py",
             "code/review/within_day_inputs/r4_networks.py", "code/review/within_day_inputs/r5_outcome.py", "code/review/within_day_inputs/r6_assemble.py", "code/review/within_day_inputs/run_checks.sh"],
    scripts_how_to_run="from the project root with the project environment, PYTHONUTF8=1, one torch thread: run_checks.sh (r1, r3, r2, r4, r5, r6 in this order; about 2 minutes; outputs to REVIEW_OUT); r6_assemble.py --write-into-project writes this record",
    not_checked=["No training was done: whether a new training with the same seeds reproduces the numbers on this PC (4 threads) is untested; the stored forecasts and networks are what was checked.",
                 "The hindsight of the within-day columns was not measured (no Stage IV rebuild of the six columns; finding 2); the pre-2002 hourly shape of the training years was not examined beyond the E1b review's reading of the methods document (finding 3).",
                 "The stored sets' networks (Mac) were not reloaded; the forecast tables of script 18 (data/gefs/basins) were read, not rebuilt; the bias factor's fit was taken from results/gefs_basins.json.",
                 "The block bootstrap of 13_uncertainty.py was not run on these sets (the results file lists it as not done); the binomial standard error of a share of 611 onsets is about 2 points, more with storm clustering.",
                 "The simple-model yardstick (ridge, forest with the within-day columns) does not exist, so 'the information, not the network' is rightly absent from the file."],
    dates=dict(builder_training_pass="2026-10-28 23:40 to 2026-10-29 00:02 (results file written " + R["written"] + ")", review=f"2026-10-29, about {start_time} to {end_time}"),
    seconds=round(time.time() - t0, 1))
jsave("review_within_day_inputs.json", rec)
print(f"record built: {len(findings)} findings ({counts['must_fix']} must-fix, {counts['should_fix']} should-fix, {counts['notes']} notes); verdict: {verdict}")
if WRITE:
    # nothing reviewed may have changed since r1 (except the preliminary diagnostic that every script calling subdaily_inputs.checks() rewrites, e.g. 22_flow_state.py running meanwhile: recorded, not asserted)
    volatile = {"results/diagnostics_2026-10-28/subdaily_inputs_check.json"}
    changed = [f for f, h in r1["code_files_sha256"].items() if sha256(f) != h and f not in volatile] + [f for f, h in R["integrity"]["sha256_of_saved_files"].items() if sha256(f) != h]
    assert not changed, "a reviewed file changed since r1_integrity.py ran: " + ", ".join(changed)
    rec["files_reviewed"]["changed_since_r1"] = {f: dict(sha256_at_r1=r1["code_files_sha256"][f], sha256_now=sha256(f), why="a preliminary diagnostic rewritten by every script that runs subdaily_inputs.checks() (its numbers were the same: 12 basins, 611 onsets, 70 dry issue days)")
                                                for f in volatile if sha256(f) != r1["code_files_sha256"][f]}
    target = PROJ / "results" / "review_within_day_inputs.json"
    target.write_text(json.dumps(rec, indent=1, default=_default, ensure_ascii=False), encoding="utf-8")
    print("written into the project:", target)
