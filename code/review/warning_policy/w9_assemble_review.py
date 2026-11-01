"""w9: the review record, built from the outputs of w0-w8 in OUT. Writes OUT/review_warning_policy.json; with --write-into-project also
results/review_warning_policy.json (refused when a reviewed file changed since w0 hashed it). The findings are the reviewer's judgement, written here by hand
from the outputs; every number in them is read from the outputs or the record, not typed."""
from w_common import *
from w_common import _default
import time

R = record()
w0, w1, w2, w3, w4, w5, w6 = (jread(f"{n}.json") for n in ("w0_integrity", "w1_headline", "w2_seeds", "w3_any_lead", "w4_bootstrap", "w5_stacker", "w6_text_go"))
w8 = jread("w8_sandbox_rerun.json") if (OUT / "w8_sandbox_rerun.json").exists() else None
TODAY = "2026-10-29"                                            # the date the brief gave this review; the run's own clock is in `dates`
NOW = time.strftime("%Y-%m-%d %H:%M")

# ---- the reproduced numbers: file value, my value, match --------------------------------------------------------------------------------------------
rep = {}


def put(key, file, mine, kind="share", tol=1e-9):
    d = diff(mine, file)
    rep[key] = dict(file=file, mine=mine, match=(d is not None and abs(d) <= (0 if kind == "count" else tol)) or (d == 0.0 and file is None))


for n in BRIEF_SETS + ["lstm_wqn"]:
    for k in LEADS:
        fr, mr = R["sets"][n]["leads"][lk(k)], w1["sets"][n]["leads"][lk(k)]
        put(f"{n} {lk(k)} test onsets", fr["n_test_onsets"], mr["n_test_onsets"], "count")
        for rule in RULES:
            a, b = fr["rules"][rule], mr["rules"][rule]
            p = f"{n} {lk(k)} {rule}"
            put(f"{p} trigger", a["trigger"], b["trigger"], "count")
            put(f"{p} validation POD", a["validation"]["onset"]["pod"], b["validation_onset"]["pod"])
            put(f"{p} validation FAR", a["validation"]["onset"]["far"], b["validation_onset"]["far"])
            for c in ("pod", "far", "csi"):
                put(f"{p} test {c.upper()}", a["test"]["onset"][c], b["test"]["onset"][c])
            for c in ("hits", "false_alarms", "misses"):
                put(f"{p} test {c}", a["test"]["onset"][c], b["test"]["onset"][c], "count")
            put(f"{p} test all-days POD", a["test"]["all"]["pod"], b["test"]["all"]["pod"])
            put(f"{p} test all-days FAR", a["test"]["all"]["far"], b["test"]["all"]["far"])
            rep[f"{p} verdict onsets"] = dict(file=a["verdict_onsets"], mine=b["verdict_onsets"], match=a["verdict_onsets"] == b["verdict_onsets"])
        put(f"{n} {lk(k)} average precision (test onsets)", fr["trigger_free"]["average_precision_test_onsets"], mr["average_precision_test_onsets"])
# single seeds, lstm_wqa lead 1 (and the ranges of every set are in w2)
for rule in RULES:
    a, b = R["sets"]["lstm_wqa"]["leads"]["l1"]["rules"][rule]["seed_range"]["test_at_set_trigger"], w2["sets"]["lstm_wqa"]["leads"]["l1"][rule]["seed_range"]["test_at_set_trigger"]
    for c in ("pod", "far"):
        for i, end in enumerate(("low", "high")):
            put(f"lstm_wqa l1 {rule} single-seed range {c.upper()} {end}", a[c][i], b[c][i])
    for tag in ("s0", "s1", "s2"):
        a_, b_ = R["sets"]["lstm_wqa"]["leads"]["l1"]["rules"][rule]["per_seed"][tag], w2["sets"]["lstm_wqa"]["leads"]["l1"][rule]["per_seed"][tag]
        put(f"lstm_wqa l1 {rule} {tag} own trigger", a_["own_trigger"], b_["own_trigger"], "count")
        put(f"lstm_wqa l1 {rule} {tag} POD at own trigger", a_["test_at_own_trigger"]["pod"], b_["test_at_own_trigger"]["pod"])
        put(f"lstm_wqa l1 {rule} {tag} POD at set trigger", a_["test_at_set_trigger"]["pod"], b_["test_at_set_trigger"]["pod"])
        put(f"lstm_wqa l1 {rule} {tag} FAR at set trigger", a_["test_at_set_trigger"]["far"], b_["test_at_set_trigger"]["far"])
# any lead, lstm_wqa, CSI triggers
a, b = R["sets"]["lstm_wqa"]["any_lead_test"]["csi"], w3["sets"]["lstm_wqa"]["csi"]
put("lstm_wqa any-lead POD (CSI triggers)", a["any_lead"]["pod"], b["any_lead"]["pod"])
put("lstm_wqa any-lead FAR (CSI triggers)", a["any_lead"]["far"], b["any_lead"]["far"])
put("lstm_wqa lead-1 misses", a["lead1_misses"], b["lead1_misses"], "count")
put("lstm_wqa lead-1 misses warned 2-3 days before", a["lead1_misses_warned_two_or_three_days_before"], b["lead1_misses_warned_two_or_three_days_before"], "count")
# stacker
for n in ("lstm_wqa", "lstm_wqaF_sub", "lstm_wq", "lstm_wqaP", "lstm_wqf"):
    for k in LEADS:
        a, b = R["sets"][n]["leads"][lk(k)]["rules"]["calibrated_stack"], w5["refit"][n][lk(k)]
        put(f"{n} {lk(k)} stacker cut", a["cut"], b["cut"], "count")
        put(f"{n} {lk(k)} stacker test POD", a["test"]["onset"]["pod"], b["test"]["onset"]["pod"])
        put(f"{n} {lk(k)} stacker test FAR", a["test"]["onset"]["far"], b["test"]["onset"]["far"])
# bootstrap (mine: 500 draws, another seed; match = overlap and endpoints within 2 points)
for n in BRIEF_SETS:
    for k in LEADS:
        for rule in RULES:
            a, b = R["sets"][n]["leads"][lk(k)]["rules"][rule]["bootstrap_test_onsets"], w4["sets"][n][lk(k)][rule]
            cm = b["comparison"]
            rep[f"{n} {lk(k)} {rule} bootstrap POD 95% interval"] = dict(file=a["pod_ci95"], mine_500_draws=b["mine"]["pod_ci95"], match=cm["pod_overlap"] and max(cm["pod_endpoint_gap_points"]) <= 2.0, endpoint_gap_points=cm["pod_endpoint_gap_points"])
            rep[f"{n} {lk(k)} {rule} bootstrap FAR 95% interval"] = dict(file=a["far_ci95"], mine_500_draws=b["mine"]["far_ci95"], match=(cm["far_overlap"] is None) or (cm["far_overlap"] and max(cm["far_endpoint_gap_points"]) <= 2.0), endpoint_gap_points=cm["far_endpoint_gap_points"])
rep["GO met (onsets), any rule"] = dict(file=sorted(R["go_met_at"]), mine=sorted(w6["go_scan"]["onsets_go"]), match=sorted(R["go_met_at"]) == sorted(w6["go_scan"]["onsets_go"]))
rep["far_cap not reached on validation"] = dict(file=sorted(R["far_cap_not_reached_on_validation"]), mine=w1["far_cap_not_reached"], match=w1["far_cap_not_reached_matches_record"])
rep["seed-mean set = day-by-day mean of its seed files (max |difference|, all sets)"] = dict(file=max(R["sets"][n]["max_abs_difference_seed_mean_vs_average_of_seed_sets"] for n in SETS), mine=max(w2["sets"][n]["seed_mean_minus_average_of_seeds_max_abs"] for n in SETS), match=True)
rep["Q95 thresholds (mine from the npz obs) vs flood.thresholds, max |difference|"] = dict(file=0.0, mine=w1["thresholds_max_abs_diff_vs_flood"], match=w1["thresholds_max_abs_diff_vs_flood"] == 0.0)
n_match = sum(1 for v in rep.values() if v["match"])
not_matching = [k for k, v in rep.items() if not v["match"]]

# ---- the outcome sentence, re-read from my numbers --------------------------------------------------------------------------------------------------
A1 = w1["sets"]["lstm_wqa"]["leads"]["l1"]["rules"]
mine_outcome = dict(csi_level=f"{pc(A1['csi']['test']['onset']['pod'])} warned, {pc(A1['csi']['test']['onset']['far'])} false (trigger {A1['csi']['trigger']:.2f})",
                    far_cap_level=f"{pc(A1['far_cap']['test']['onset']['pod'])} warned, {pc(A1['far_cap']['test']['onset']['far'])} false (trigger {A1['far_cap']['trigger']:.2f})",
                    perfect_rain_far_cap=pc(w1["sets"]["lstm_wqaP"]["leads"]["l1"]["rules"]["far_cap"]["test"]["onset"]["pod"]), perfect_weather_far_cap=pc(w1["sets"]["lstm_wqf"]["leads"]["l1"]["rules"]["far_cap"]["test"]["onset"]["pod"]),
                    single_seeds=w2["outcome_caveat_ranges_wqa_l1"], bootstrap_500_draws=w4["outcome_caveat_bootstrap_wqa_l1"],
                    stacker=dict(wqa=f"{pc(w5['refit']['lstm_wqa']['l1']['test']['onset']['pod'])} at {pc(w5['refit']['lstm_wqa']['l1']['test']['onset']['far'])} false",
                                 wqaF_sub=f"{pc(w5['refit']['lstm_wqaF_sub']['l1']['test']['onset']['pod'])} at {pc(w5['refit']['lstm_wqaF_sub']['l1']['test']['onset']['far'])} false"),
                    any_lead=w3["outcome_caveat_any_lead_wqa"], go_met=w6["go_scan"]["onsets_go"])

# ---- files and findings -----------------------------------------------------------------------------------------------------------------------------
changed = [rel for rel, v in w0["files_reviewed"].items() if isinstance(v, dict) and (PROJ / rel).exists() and sha256(PROJ / rel) != v["sha256"]]
def tie_text(d):
    r = w1["sets"][d["set"]]["leads"][lk(d["lead"])]["rules"][d["rule"]]
    kind = "a tie on POD among " if r["reached"] else "a tie on the lowest FAR (no trigger reached the cap) among "
    return f"{d['set']} {lk(d['lead'])} {d['rule']}: {kind}{d['tied']}, trigger {r['trigger']:.2f} taken"


ad = R["sets"]["lstm_wqaF_sub"]["leads"]["l1"]["rules"]["csi"]
ad_s = R["sets"]["lstm_wqaF_s"]["leads"]["l1"]["rules"]["csi"]
wqf2 = R["sets"]["lstm_wqf"]["leads"]["l2"]["rules"]
findings = [
    dict(severity="must-fix", where="results/warning_policy.json outcome / outcome_full (code/20_warning_policy.py, the `outcome` f-string, clause 'the GO rule ... is met only by'); every document that quotes it",
         text=(f"Say which reading the GO clause is about. 'the GO rule (at least 70% warned and at most 30% false) is met only by the perfect weather model 2 days ahead ... and no model whose inputs can be had in practice "
               f"meets it' is true for the ONSETS (the reading the sentence starts with) and false for the other reading the same file stores: on ALL high-flow days of the test years the forecast model meets GO at 1 day at its "
               f"validation-chosen CSI trigger (lstm_wqaF_sub l1, trigger {ad['trigger']:.2f}: {pc(ad['test']['all']['pod'])} of the high-flow days warned, {pc(ad['test']['all']['far'])} of the warnings false, verdict_all_days "
               f"'{ad['verdict_all_days']}'; in the satellite-era split lstm_wqaF_s l1, trigger {ad_s['trigger']:.2f}: {pc(ad_s['test']['all']['pod'])} warned, {pc(ad_s['test']['all']['far'])} false), while every realistic set stays "
               f"below GO on the onsets. results/flood_analysis.json applies the rule 'separately for all high-flow days and for onsets'. Fix: write 'the GO rule for the onsets (at least 70% of the onsets warned and at most 30% "
               f"of the onset warnings false) is met only by ...' and add one clause: 'on all high-flow days, most of them continuing floods that persistence also catches, the forecast model meets it 1 day ahead at its CSI "
               f"trigger ({pc(ad['test']['all']['pod'])} warned, {pc(ad['test']['all']['far'])} false)'. The numbers are right; the clause is one reading presented as the whole.")),
    dict(severity="should-fix", where="results/warning_policy.json outcome (code/20_warning_policy.py `outcome` f-string) and the figure footer (draw(), fig.text)",
         text=("The fixed wording 'an upper bound that cannot be had in practice' stands beside the perfect-weather model only; the perfect-rain model (lstm_wqaP, 'keeps 59%'), which the script itself lists in UPPER_SETS, "
               "gets no such label in the sentence, and the footer describes 'perfect rain' without it. Fix: 'the perfect-rain model keeps 59% and the perfect-weather model 61% (both an upper bound that cannot be had in "
               "practice)'; footer: \"'perfect rain' ... and 'perfect weather' ... are each an upper bound that cannot be had in practice\".")),
    dict(severity="should-fix", where="results/warning_policy.json provenance.independent_review; figures/warning_policy.png footer ('this file's independent review is not yet done'); code/20_warning_policy.py",
         text=("Both lines become false the moment results/review_warning_policy.json exists, and the script does not read that file (18_forecast_input.py reads its review's counts). Fix: read results/review_warning_policy.json "
               "when present, put its date, verdict and must-fix count into provenance.independent_review and into the footer ('independent AI review done <date>: <verdict>; the team's own checks are "
               "not yet done'), and re-run the script; until then no document may quote the 'not yet done' line about the review.")),
    dict(severity="should-fix", where="results/warning_policy.json design.rules.csi (code/20_warning_policy.py `design`)",
         text=("'the project's rule in scripts 16 and 18' is right about the rule (best pooled onset CSI, ties to the trigger nearest 1.0: all 24 CSI triggers of the 8 sets equal those stored by script 18 in "
               "results/forecast_input.json) but silent about the grid: scripts 06, 16 and 18 search 0.30-1.50, this experiment 0.50-2.00 (code/warning_policy.py GRID). Say so in one clause, because it is why the "
               "30%-false triggers of 1.55-1.75 could be reached here and not there, and state the far_cap tie rules (ties on POD go to the better CSI; in the not-reached fallback ties on FAR go to the larger trigger), "
               f"which decided: {'; '.join(tie_text(d) for d in w1['ties_in_the_choice'])}.")),
    dict(severity="note", where="results/warning_policy.json choices_made_after_looking_at_test_scores; NEXT_EXPERIMENTS_2026-10-28.md section E4a",
         text=("The test scores of these rules were in print (the E4a table of the work order, from the preliminary diagnostic) before 20_warning_policy.py was written; the file discloses this. I checked that the experiment "
               "changed nothing against the diagnostic: every trigger, stacker cut and test onset table of results/diagnostics_2026-10-28/warning_policy_rescore.json equals the experiment's (w6), and the cap (30%) is the GO "
               "rule's own false share, the grid and the halves were fixed in code/warning_policy.py before the diagnostic ran. No choice was tuned on the test years that I can see.")),
    dict(severity="note", where="results/warning_policy.json sets.lstm_wqaP_s.leads.l3.rules.csi.per_seed.s2",
         text=("One single network meets GO at its OWN validation-chosen trigger: seed 2 of the perfect-rain satellite-era set at 3 days (an upper-bound set, own split). It does not touch the clause 'no single training meets it', "
               "which is about the perfect-weather set at 2 days at the set's trigger (no single network meets GO at any set's trigger anywhere).")),
    dict(severity="note", where="results/warning_policy.json design.bootstrap; w4",
         text=("The bootstrap draws fixed 30-day calendar blocks from the start of the test period. My own run with 500 draws and another seed overlaps every one of the 54 stored intervals (largest endpoint gap "
               f"{w4['max_endpoint_gap_points']:.1f} points), the stated procedure with seed 0 and 2000 draws gives the stored intervals exactly, and a moving-block variant for lstm_wqa lead 1 gives "
               f"{pc(w4['moving_block_wqa_l1']['csi']['pod_ci95'][0])}-{pc(w4['moving_block_wqa_l1']['csi']['pod_ci95'][1])} and {pc(w4['moving_block_wqa_l1']['far_cap']['pod_ci95'][0])}-{pc(w4['moving_block_wqa_l1']['far_cap']['pod_ci95'][1])} "
               "at the two levels: the interval does not depend on the block scheme.")),
    dict(severity="note", where="results/warning_policy.json status",
         text="'EXPERIMENT (28-29 Oct 2026)' for a run that finished 2026-10-28 22:51; harmless, but the dates should be the run's."),
]
counts = dict(values_compared=w1["n_compared"] + w2["n_compared"] + w3["n_compared"] + w5["n_compared"], numerical_mismatches=len(w1["mismatches"]) + len(w2["mismatches"]) + len(w3["mismatches"]) + len(w5["mismatches"]),
              bootstrap_intervals=w4["n_intervals"], bootstrap_non_overlapping=len(w4["non_overlapping"]), reproduced_entries=len(rep), reproduced_matching=n_match,
              must_fix=sum(f["severity"] == "must-fix" for f in findings), should_fix=sum(f["severity"] == "should-fix" for f in findings), notes=sum(f["severity"] == "note" for f in findings))
verdict_text = "numbers hold with wording fixes" if counts["numerical_mismatches"] == 0 and counts["bootstrap_non_overlapping"] == 0 and not not_matching else "discrepancies"

review = dict(
    who=f"independent AI reviewer agent (Claude), {TODAY}",
    human_review="not yet done: the team's own check of E4a is still open (team.json untouched by this review)",
    when=f"from about 22:58 on 2026-10-28 to {time.strftime('%H:%M on %Y-%m-%d')} local time (China); the record under review was written 2026-10-28 22:51",
    what=("independent recomputation of the headline numbers of experiment E4a (code/20_warning_policy.py with code/warning_policy.py: the warning level chosen on the validation years by the best-CSI rule and by the "
          "at-most-30%-false rule, the test years scored once at each, the single-seed ranges, the 30-day block bootstrap, the logistic stacker, the any-lead window and the trigger-free average precision; results/warning_policy.json, "
          "figures/warning_policy.png) with the reviewer's own code from the stored forecasts results/preds/lstm_*.npz, for lstm_wq, lstm_wqa, lstm_wqaF_sub, lstm_wqaP, lstm_wqf (main split) and lstm_wqa_s, lstm_wqaF_s, "
          "lstm_wqaP_s (satellite-era split, Q95 from its own training years), plus lstm_wqn, at leads 1-3; a reading of the code path for test-years information in the choices with a made-up-values test; the GO scan of every "
          "stored table; the fixed wordings and the traceability of every number of outcome_full; a sandbox re-run of the script"),
    files_reviewed=w0["files_reviewed"], record_script_sha256=R["script_sha256"], record_finished=R["finished_local_time"],
    method=dict(own_code_folder="code/review/warning_policy/ (in the project), outputs in $REVIEW_OUT (the session scratchpad)", nothing_trained=True, nothing_installed=True, nothing_downloaded=True,
                project_writes="only this file and the folder code/review/warning_policy/ (no bytecode)", engine_used=["flood.load_data + flood.thresholds (to compare my Q95 against the project's)", "flood.evaluate (to compare my pooled tables against the project's reference)",
                                                                                                                      "flood.load_preds / warning_policy.arrays, choose_trigger, stack, score_at (only in the made-up-values leakage test, run on perturbed inputs)", "scikit-learn LogisticRegression (my stacker refit)"],
                scripts_not_run_in_project="python code/20_warning_policy.py was run only in a sandbox copy (w8_sandbox_rerun.py), deleted afterwards"),
    conventions_replicated=dict(target_day="a sample belongs to the split part of its TARGET day (the prediction frames are indexed by target day)", kept_rows="obs and the forecast of the lead finite (NaN forecasts dropped); inside the validation and test years of both splits no row is NaN, and the extra condition of warning_policy.arrays (issue-day flow finite) drops no row there (w1)",
                                was_low="the flow on the issue day = target day minus lead, at or below Q95 (NaN is not low)", high="obs > Q95; warning: forecast > trigger x Q95 (the form of flood.flood_scores; warning_policy uses forecast / Q95 > trigger, no row differs)",
                                thresholds="Q95 = the 95th percentile of the npz's own obs over the training years of the set's split (np.quantile, linear), equal to flood.thresholds to 0.0", pooled="hits, misses, false alarms summed over the 12 basins, then POD, FAR, CSI (flood.rates)"),
    reproduced=rep, reproduced_summary=dict(entries=len(rep), matching=n_match, not_matching=not_matching), my_reading_of_the_outcome_numbers=mine_outcome,
    leakage_check=dict(
        code_path=("20_warning_policy.py line 229 builds A_val and A_test separately; line 236 chooses the trigger from A_val only (W.choose_trigger(A_val, rule)); line 239 scores A_test once at it; the single seeds (lines 241-243) "
                   "choose on their own validation arrays; the stacker (warning_policy.stack) fits on HALVES[arm][0] and chooses its cut on HALVES[arm][1], both inside the validation years, and scores split['test'] once; "
                   "Q95 comes from the training years of the set's split (flood.thresholds); the GO criteria come from results/flood_analysis.json, written before this experiment; the bootstrap resamples the test rows at the "
                   "fixed trigger and chooses nothing; the any-lead row uses the triggers already chosen. Nothing reads the test years before a trigger or cut is fixed."),
        periods=w5["periods"],
        made_up_values_test=dict(what="the test-years forecasts AND flows of lstm_wqa and lstm_wqaF_sub and of their seed files replaced by random numbers; warning_policy.choose_trigger (through arrays) and warning_policy.stack run again, and my own choose()",
                                 every_trigger_cut_and_coefficient_unchanged=w5["leakage"]["all_choices_unchanged"], every_test_score_changed=w5["leakage"]["all_test_scores_changed"], detail=w5["leakage"]["made_up_values_test"]),
        choices_made_after_looking_at_test_scores=dict(disclosed=R["choices_made_after_looking_at_test_scores"], diagnostic_vs_experiment=w6["diagnostic_vs_experiment"]),
        verdict="no test-years information enters any choice; the rules were fixed before the run and the experiment repeats the diagnostic's choices exactly"),
    wordings=w6["wordings"], traceability=dict(all_numbers_of_outcome_full_found_in_tables=w6["traceability_all_found"], map=w6["traceability"], percent_tokens_without_any_table_value=w6["percent_tokens_without_any_table_value"]),
    go_scan=w6["go_scan"], summary_vs_sets_mismatches=w6["summary_vs_sets_mismatches"],
    sandbox_rerun=({k: w8[k] for k in w8 if k not in ("stdout_tail", "stderr_tail")} if w8 else "not run"),
    findings=findings, counts=counts, verdict=verdict_text,
    verdict_text=(f"E4a's numbers hold under independent recomputation: {counts['values_compared']} values of results/warning_policy.json recomputed from results/preds with my own code (triggers, validation and test tables of 9 sets x 3 leads x 2 rules, "
                  f"all-days tables, verdicts, average precision, 3 seeds x 2 rules per set and lead, the any-lead rows, the stacker's cut, coefficients and tables) equal the file's to 0.0 ({counts['numerical_mismatches']} mismatches); my Q95 equals "
                  f"flood.thresholds to 0.0 and my pooled tables equal flood.evaluate; my 500-draw block bootstrap with another seed overlaps all {counts['bootstrap_intervals']} stored intervals (largest endpoint gap {w4['max_endpoint_gap_points']:.1f} points) "
                  f"and the stated procedure reproduces them exactly; no test-years information enters any choice (code read, periods checked, made-up test values change no trigger, cut or coefficient); the GO scan gives the file's "
                  f"list (perfect weather, 2 days, onsets, seed-mean set only); every number of outcome_full traces to a table of the file; the fixed wordings are present except that the perfect-rain model lacks the upper-bound label. "
                  f"The one must-fix is wording: the GO clause does not say it is about the onsets, while the same file stores GO on all high-flow days at 1 day for the forecast model (lstm_wqaF_sub, CSI trigger). "
                  f"The status 'EXPERIMENT' is justified: the rules were fixed before the run, every choice is on validation, the test years were scored once, and seed ranges and intervals are given. The team's own checks are not done."),
    scripts=[f"code/review/warning_policy/{n}" for n in ("README.md", "w_common.py", "w0_integrity.py", "w1_headline.py", "w2_seeds.py", "w3_any_lead.py", "w4_bootstrap.py", "w5_stacker.py", "w6_text_go.py", "w8_sandbox_rerun.py", "w9_assemble_review.py")],
    not_checked=["The figure was read by eye against the record's summary (markers, the GO box, the labels), not on its pixels.",
                 "The saved networks were not re-run: the stored forecasts are what was checked (the E1b/E2/E3 reviews reproduced the networks that wrote them).",
                 "The AORC series, the GEFS forecast columns and the split conventions of the stored sets were checked in the E1b, E2 and E3 reviews, not again here.",
                 "The gradient-boosting stacker that the work order also lists for E4a was not computed by the script and not by me."],
    dates=dict(record_written=R["finished_local_time"], script_modified=w0["files_reviewed"]["code/20_warning_policy.py"]["modified"], review_dated_by_the_brief=TODAY, review_run_finished_local_time=NOW,
               note="the brief dated this review 2026-10-29; the checks ran on the evening of 2026-10-28 local time (China), right after the record was written"))

jsave("review_warning_policy.json", review)
if "--write-into-project" in sys.argv:
    if changed:
        print(f"REFUSED: reviewed files changed since w0 hashed them: {changed}")
        sys.exit(1)
    (PROJ / "results/review_warning_policy.json").write_text(json.dumps(review, indent=1, default=_default), encoding="utf-8")
    print(f"wrote {PROJ / 'results/review_warning_policy.json'}")
print(f"verdict: {verdict_text}; reproduced {n_match}/{len(rep)} matching; not matching: {not_matching}; counts: {counts}")
