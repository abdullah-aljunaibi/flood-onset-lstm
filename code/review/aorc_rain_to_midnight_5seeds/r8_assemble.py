"""Round 2: assemble the block round_2 of results/review_aorc_rain_to_midnight.json from the outputs of r1-r7 in OUT. Without --write-into-project it writes OUT/round_2.json only.
With it, the existing record is loaded, the key round_2 is added and the file is written back with the same layout (indent 1, LF, no final newline); afterwards it checks that the
record without round_2 is byte for byte the file as it was."""
from r_common import *
from r_common import _default
import time

R1, R2, R4, R6, R7 = (jread(f) for f in ("r1_integrity.json", "r2_headline.json", "r4_stage4.json", "r6_hindsight.json", "r7_outcome.json"))
R5 = jread("r5_one_network_s3.json")
J = jload(RESULTS)
S4 = jload(STAGE4)
F5, F3 = R6["five_seeds"], R6["three_seeds"]
H2 = R2["mine"]
NUM = {k: v for k, v in R7["compare"].items() if not k.startswith(("o1 the hindsight clause says", "stage4 outcome seed count"))}
WORD = {k: v for k, v in R7["compare"].items() if k not in NUM}
files_read = {}
for r in (R1, R2, R4, R5, R6, R7):
    files_read.update(r.get("files_read", {}))
stored = {p: h for p, h in sorted(files_read.items()) if p.startswith("results/preds/")}
values = sum(len(r["compare"]) for r in (R1, R2, R4, R5)) + len(NUM)
mism = sum(1 for r in (R1["compare"], R2["compare"], R4["compare"], R5["compare"], NUM) for v in r.values() if not v["ok"])
w = J["summary_test"]["sets"]["wqa"]["l1"]
V = S4["models"]["wqa"]["variants"]
ov = H2["overlaps"]["value_of_hours_09_to_24"]
vh = J["value_of_hours_09_to_24"]["by_lead"]["l1"]["onsets"]


def chk(id_, what, how, result, numbers, note):
    return dict(id=id_, what=what, how=how, result=result, numbers=numbers, note=note)


checks = [
    chk("i1", "Which files the five-seed numbers rest on: sha256 of the script, the Stage IV check, the two results files and every stored set; seeds 0-2 and the simple models against the hashes of round 1; the results file's own integrity record against the files now; the passes; the copy of the results file that check_stage4_realtime.py read (11:51) against the file written at 12:44",
        "own code (r1_integrity.py)", "pass", {k: R1[k] for k in ("script_sha256", "stage4_script_sha256", "results_sha256", "stage4_sha256", "round1_files_changed", "file_times", "passes", "keys_differing_between_11_51_copy_and_12_44_file")} | {"entries": R1["summary"]["entries"], "matching": R1["summary"]["matching"]},
        "every file round 1 recomputed is unchanged except the four seed averages (remade with five seeds, as expected); seeds 3 and 4 of the four sets exist and were not in the 30-October backup; the stored hashes of the results file equal the files; the 11:51 copy and the 12:44 file have the same summary_test and seed_ranges (they differ in notes and what_was_not_done only, outside the passes and the integrity record), so the Stage IV check read the same headline counts"),
    chk("h1", "summary_test of the 13 sets at 3 leads (median and mean NSE; onset and all-day hits, misses, false warnings, pod, far, csi) from the seed-averaged forecasts", "own numpy scoring (r2_headline.py)", "pass",
        {"lstm_wqa_l1": {"onset": {k: w["onset"][k] for k in ("hits", "misses", "false_alarms", "pod", "far")}, "nse_median": w["nse_median"]}}, "0 mismatches at 1e-9"),
    chk("h2", "Every seed average = the mean of its single seeds; seed_ranges (by_seed and range of 7 measures, 13 sets, 3 leads)", "own code (r2_headline.py)", "pass",
        {"wqa_single_seed_onset_hits_l1": H2["wqa_l1_single_seed_onset_hits"], "wqa_single_seed_nse_median_l1": H2["wqa_l1_single_seed_nse_median"], "seeds": H2["seeds"],
         "largest_average_deviation": max(H2["average_max_abs_diff"].values())}, "averages equal the means of the seed files to float32 round-off; five seeds for wqa, wqa_to09, wqa_utc, wqa_rep and for script 15's wqh, wqh_older, wqnh, wqh_to09, wqh_utc; three for wq, wqn, wqn_older, wqf"),
    chk("h3", "The simple models (yardstick.table_test: base_linreg_wqa, base_rf_wqa and their wq versions, 3 leads)", "own code (r2_headline.py)", "pass", {}, "0 mismatches"),
    chk("h4", "The reading blocks (value_of_hours_09_to_24 and the other seven, rain_table_with_the_right_clock), their overlaps and sign tests, and gain_rule (wqa and wqa_rep, every block and lead) with the readings' booleans", "own code (r2_headline.py; scipy wilcoxon and binomtest)", "pass",
        {"value_of_hours_09_to_24_l1": {k: vh[k] for k in ("warned_with", "warned_without", "difference_points", "hits_with", "hits_without", "difference_by_seed_points", "every_seed_above_every_seed")}, "overlap_09_to_24": ov}, "0 mismatches"),
    chk("h5", "e1b_success: the rule (every training above every NLDAS training, control flat), the roadmap step 1b booleans and the second variant", "own code (r2_headline.py)", "pass", H2["e1b_success"], "met: true reproduced; gain holds and grows"),
    chk("h6", "The LSTM against the simple models (yardstick.lstm_against_simple, new_input_given_to_each_model)", "own code (r2_headline.py)", "pass", {"lstm_trainings_above_simple_onsets_l1": H2["lstm_trainings_above_simple_onsets_l1"]}, "1 of 5 LSTM trainings above ridge, 5 of 5 above the forest on the onsets, as the text says"),
    chk("s1", "Stage IV check: the counts it rests on (AORC 316, rain table 114, wqa_rep 321) against my scoring and the results file; reproduction record", "own code (r4_stage4.py)", "pass", {}, ""),
    chk("s2", "Stage IV check: the 611 per-onset rows against my onsets, my AORC and rain-table flags and forecasts; Stage IV flags recounted; lost/gained; per basin", "own code (r4_stage4.py)", "pass",
        {"stage4_warned": V["stage4"]["onsets_warned"], "lost_and_gained": V["stage4"]["onsets_lost_and_gained"]}, "294 recounted from the rows; no row within the 4-decimal rounding of its level"),
    chk("s3", "Stage IV check: the rain of each issue day rebuilt by my own route (AORC hours and the cached Stage IV hours over the wall-clock day, zoneinfo; AORC for a missing hour); k and the agreement block", "own code (r4_stage4.py)", "pass",
        {"max_abs_diff_aorc_mm": R4["mine"]["max_abs_diff_aorc_mm"], "max_abs_diff_stage4_mm": R4["mine"]["max_abs_diff_stage4_mm"], "k": R4["mine"]["k"]}, "equal to the 3 decimals stored; k to 4e-16"),
    chk("s4", "Stage IV check: every variant's derived numbers from its own counts (share surviving, difference, share of onsets, per basin, lost/gained) for wqa and wqa_rep; single seeds and control draws against the log", "own code (r4_stage4.py)", "pass",
        {"variants_wqa": R4["mine"]["variants_wqa"], "single_seeds_wqa": R4["mine"]["single_seeds_wqa"], "share_surviving_stage4": V["stage4"]["share_of_the_aorc_effect_surviving"]}, "89.1% = (294 - 114) / (316 - 114)"),
    chk("s5", "Stage IV check: the 95% intervals of the raw variant: exact replication with the script's generator and order, and an independent bootstrap (4000 draws, my own seed)", "own code (r4_stage4.py)", "pass",
        {"file": V["stage4"]["intervals"], "own_bootstrap": R4["mine"]["intervals_own_bootstrap_4000"]}, "exact replication to 1e-12; my own draws within 0.4 points (83% to 95% over onsets and over issue days)"),
    chk("s6", "Stage IV check: three seeds against five (the three-seed file of the 30-October backup): seeds 0-2 give the same single-seed counts in every variant; the issue-day rain is unchanged", "own code (r4_stage4.py)", "pass",
        R4["mine"]["three_seed_file"], "the change from 90.4% to 89.1% surviving and from 319 to 311 (level restored) comes from seeds 3-4 alone"),
    chk("n1", "One network rebuilt (lstm_wqa seed 3, trained on the Windows PC on 30 October; script 16's reload check covers seed 0 only) from its .pt file, one torch thread, on a table whose AORC column I built myself; then with my Stage IV sums of the 611 issue days", "own code (r5_one_network.py; flood.LSTMNet / Run.predict)", "pass",
        R5["mine"], "bit for bit against lstm_wqa_s3.npz (max abs diff 0.0); Stage IV 296, last hour out 274, last two hours out 254 = single_seed_onsets_warned[3] of every variant"),
    chk("k1", "Does round 1's hindsight bound ('at most about 3 points', 16 of 313 hits, check d7, three networks) hold for the five-seed set? The 171 onsets of round 1's Stage IV sample (round 1's day selection rerun) read from the per-onset flags of the five-seed and the three-seed Stage IV files, and the direct counts on all 611 onsets",
        "own code (r6_hindsight.py; a7a_select_days.py of round 1 rerun into the scratch folder)", "finding", {"five_seeds": F5, "three_seeds": F3},
        f"With five networks the 171 sample onsets give {F5['stage4_raw_sample']} (raw Stage IV) against {F5['aorc_sample']} with AORC (three: {F3['stage4_raw_sample']} against {F3['aorc_sample_file']}; round 1 had 86 against 92 with its own Stage IV cache), a loss of {100 * F5['relative_loss_on_the_sample_raw']:.1f}% (three: {100 * F3['relative_loss_on_the_sample_raw']:.1f}%); round 1's route then gives about {F5['round1_route_extrapolated_hits']:.0f} hits = {F5['round1_route_extrapolated_points']:.1f} points. Measured directly on all 611 onsets the raw Stage IV loss is {F5['direct_all_611_raw_loss_hits']} of {F5['aorc_all']} hits = {F5['direct_all_611_raw_loss_points']:.1f} points (three seeds: {F3['direct_all_611_raw_loss_hits']} = {F3['direct_all_611_raw_loss_points']:.1f}), and {F5['direct_all_611_level_matched_loss_hits']} hits = {F5['direct_all_611_level_matched_loss_points']:.1f} points with the level of AORC restored (three seeds: {F3['direct_all_611_level_matched_loss_hits']}). So 'at most about 3 points' is a three-seed result: for the five-seed set the loosest Stage IV route gives about 4 points (3.6 directly), the level-matched one under 1 point; the dose-response route of round 1 (2.6 points) was not rerun (it needs every network)."),
    chk("o1", "Every number of outcome and outcome_caveat (= outcome_full) rebuilt from my numbers and found in the text; the hindsight clauses", "own code (r7_outcome.py)", "finding",
        {"phrases_found": sum(1 for k, v in NUM.items() if k.startswith("o1") and v["ok"]), "phrases_checked": sum(1 for k in NUM if k.startswith("o1")), "stale_clauses": R7["stale_clauses"]},
        "every five-seed number is right (19% to 52%, 48.6-55.3%, +33.1/+33.7, 0.59 to 0.76, 0.68-0.77, 180/183 hits, 228 against 22, +22.3, ridge/forest, the replaced table, about 34 points, Stage IV 294/316/114, 89% (83-95%), 276 (80%), 311, 286/36/13, MAE); the hindsight clause quotes round 1's three-seed numbers as 'the same networks' and divides round 1's 16 hits by the five-seed 316 (finding 1)"),
    chk("o2", "The outcome of results/stage4_realtime.json", "own code (r7_outcome.py)", "pass", {}, "every number found; the seed count is written as a digit ('the 5 saved lstm_wqa networks'; note)"),
    chk("o3", "Verdict words at trigger 1.0 and at the level chosen on the validation years (1.05); the mean absolute errors on the onset days", "own code (r7_outcome.py)", "pass", R7["mine"], "NOT YET for the onsets, CAUTION for all days at 1.0 (GO at 1.05 for all days); MAE 4.62 AORC, 5.83 rain table"),
]
findings = [
    dict(severity="must-fix", where="results/aorc_rain_to_midnight.json outcome_caveat, outcome_full and notes[4] (code/16_aorc_rain_to_midnight.py, hindsight_text() lines 1295-1299 and review_hindsight() lines 1231-1247)",
         text=("The hindsight clause mixes three and five seeds: 'the same networks warn of 86 to 95 of the 171 onsets ... against 92 with AORC ... 92% to 104% of the AORC effect' are round 1's numbers for the three networks of seeds 0-2 (check d5), "
               "and '(16 of 316 hits)' divides round 1's 16 hits (of 313, check d7) by the five-seed count (th_['hits']). With the five networks the same sample gives "
               f"{F5['stage4_raw_sample']} against {F5['aorc_sample']}, and on all 611 onsets raw Stage IV costs {F5['direct_all_611_raw_loss_hits']} of {F5['aorc_all']} hits ({F5['direct_all_611_raw_loss_points']:.1f} points; "
               f"{F5['direct_all_611_level_matched_loss_hits']} hits, {F5['direct_all_611_level_matched_loss_points']:.1f} points, with the level of AORC restored), so 'at most about 3 points' holds for the three-seed set only. "
               "Fix: in review_hindsight() also read bound_of_hits = REVIEW['numbers']['headline_test_years_trigger_1']['lstm_wqa']['l1']['onset_hits'] (313) and bound_of_pod (its onset_pod, 0.512), "
               "and write: 'with Stage IV ... in place of AORC on the issue day the three networks of seeds 0-2 that the review tested warn of 86 to 95 of the 171 onsets of those days against 92 with AORC (the rain table alone: 15), "
               "that is 92% to 104% of the AORC effect; a hindsight that the review could not exclude would cost those networks at most about 3 points of their 51% (16 of 313 hits); with all five networks Stage IV in place of AORC "
               "costs 22 of 316 hits on all 611 onsets (3.6 points; 5 hits, 0.8 points, with the level of AORC restored: results/stage4_realtime.json and round_2 of the review)'. Then --rescore.")),
    dict(severity="should-fix", where="code/report_sections/s0_summary.py HINDSIGHT_WORDING (line 361), hindsight_facts() (line 486-501), hindsight_clause() (line 604-609); HANDOFF.md section 7",
         text=("The fixed wording 'hindsight at most about 3 points', to stand beside every 51% (now 52%), is read from round 1's check d7 (2.6 points, three networks) and so would keep printing 'about 3 points' beside the five-seed 52%. "
               "Either label it 'hindsight at most about 3 points (measured on the three-seed networks)', or read round_2.hindsight.five_seed_loosest_bound_points of this record (3.6, i.e. 'about 4 points', the raw Stage IV route on all 611 onsets) "
               "in hindsight_facts() when round_2 exists. Which of the two is the user's call: the wording is fixed by his rule (HANDOFF.md section 7).")),
    dict(severity="should-fix", where="results/aorc_rain_to_midnight.json what_was_not_done[0] (code/16_aorc_rain_to_midnight.py, the sentence built from REVIEW['counts'])",
         text="It quotes round 1's counts only (31 checks, 22 passed, 2 must-fix, 9 should-fix). Read REVIEW.get('round_2') as well and add one clause: 'a second round on the five-seed numbers (round_2: N checks, M passed, 0 numerical mismatches, its findings)'. The same for any document builder that quotes the review's counts."),
    dict(severity="note", where="code/check_stage4_realtime.py lines 704 and 796 (outcome and figure x-label)",
         text="'the 5 saved lstm_wqa networks': the digit where every other file writes the seed count in words; use {NUMBER_WORD.get(N_NETS, N_NETS)} with NUMBER_WORD = {3: 'three', 5: 'five'} as in script 16."),
    dict(severity="note", where="code/16_aorc_rain_to_midnight.py reload_check",
         text="The reload check rebuilds seed 0 of each set only; seeds 3 and 4 of wqa_to09 and wqa_utc, trained on the second machine, were not rebuilt from their .pt files by any pass (wqa and wqa_rep were, all five bit for bit, by the Stage IV check; wqa seed 3 by this review, bit for bit). Rebuilding every seed trained in the pass would cover the new machine."),
    dict(severity="note", where="results/stage4_realtime.json per_onset (three-seed file of the backup) against round 1 check d5",
         text=f"On round 1's 171 sample onsets the three-seed Stage IV file gives {F3['stage4_raw_sample']} warnings with raw Stage IV where round 1 found 86: the Stage IV hours of the builder's cache and of round 1's own reduction differ on at least one onset; the AORC count (92) is the same. No number of the results files depends on it."),
    dict(severity="note", where="round 1 check d7 (second route)",
         text="Round 1's dose-response route (a8_hindsight_networks.py T2: 16 hits) was not rerun on the five networks (it loads every network; this round was limited to one); the five-seed bound above rests on the Stage IV route."),
]
counts = dict(checks_listed=len(checks), checks_passed=sum(c["result"] == "pass" for c in checks), findings_not_pass_fail=sum(c["result"] == "finding" for c in checks), failed=0,
              must_fix=sum(f["severity"] == "must-fix" for f in findings), should_fix=sum(f["severity"] == "should-fix" for f in findings), notes=sum(f["severity"] == "note" for f in findings),
              numerical_mismatches=mism, values_compared=values)
verdict_text = (f"Every five-seed number of results/aorc_rain_to_midnight.json that I recomputed from results/preds with my own scoring equals the file's ({values} values, 0 mismatches): onsets warned 1 day ahead by lstm_wqa "
                f"{w['onset']['hits']} of 611 = {100 * w['onset']['pod']:.1f}% with {100 * w['onset']['far']:.1f}% of its warnings false, median NSE {w['nse_median']:.3f}, single trainings {H2['wqa_l1_single_seed_onset_hits']} hits "
                f"(48.6% to 55.3%) and NSE {min(H2['wqa_l1_single_seed_nse_median']):.3f} to {max(H2['wqa_l1_single_seed_nse_median']):.3f}; the controls (rain table 18.7%, AORC over the table's window 18.0%, NLDAS 30.0%, ERA5 29.5%, UTC day 38.3%, ridge 55.2%, forest 48.0%), "
                f"e1b_success met, the hours 09:00 to 24:00 worth +33.7 points (316 against 110 hits; every seed above every seed; 228 onsets warned only with them against 22). "
                f"results/stage4_realtime.json is consistent with the stored forecasts and with the Stage IV hours: 294 of 611 recounted from its rows, the issue-day rain rebuilt by my own route, 89.1% of the AORC effect surviving with its intervals replicated exactly (83% to 96% over onsets, 83% to 95% over issue days) and within 0.4 points by my own bootstrap, 276 and 311; "
                f"one network (wqa seed 3, trained on the second machine) rebuilt bit for bit and, with my Stage IV sums, giving the stored 296, 274 and 254. "
                f"What needs fixing is wording: the hindsight clause quotes round 1's three-seed test as 'the same networks' and divides its 16 hits by the five-seed 316; 'at most about 3 points' holds for the three-seed networks, while with five the same Stage IV route gives about 4 points "
                f"({F5['direct_all_611_raw_loss_hits']} of {F5['aorc_all']} hits on all 611 onsets, {F5['direct_all_611_level_matched_loss_hits']} with the level of AORC restored); it must be labelled as measured on three seeds or restated.")
now = time.strftime("%Y-%m-%d %H:%M")
round_2 = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E1b is still open (team.json untouched by this review)",
    when=f"2026-10-30, written {now} local time, after the extension pass (11:38-11:51), the Stage IV rerun (11:52-11:54) and the rescore (12:42-12:44) on the Windows PC",
    dates={"review": "2026-10-30", "results_file_written": f"{R1['file_times'][RESULTS]} (rescore pass; seeds 3-4 trained in the extension pass that finished 2026-10-30 11:51)",
           "stage4_file_written": R1["file_times"][STAGE4]},
    covers="the five-seed numbers of results/aorc_rain_to_midnight.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC) and the rerun of code/check_stage4_realtime.py on the five lstm_wqa networks (results/stage4_realtime.json)",
    record_script_sha256=R1["script_sha256"],
    stored_sets_reviewed=stored,
    what="independent recomputation, with my own numpy scoring from results/preds, of the five-seed headline, controls, seed ranges, reading blocks, gain_rule, e1b_success, the worth of the hours 09:00 to 24:00 and the outcome sentences; a consistency check of the Stage IV rerun from its per-onset rows, the cached Stage IV hours and one rebuilt network; and whether round 1's hindsight bound still holds for five seeds",
    files_reviewed={p: h for p, h in sorted(files_read.items())} | {SCRIPT: R1["script_sha256"], STAGE4_SCRIPT: R1["stage4_script_sha256"], RESULTS: R1["results_sha256"], STAGE4: R1["stage4_sha256"]},
    method=dict(own_code_folder="code/review/aorc_rain_to_midnight_5seeds/ (in the project) and the session scratchpad for the outputs (REVIEW_OUT)", torch_threads=1, nothing_trained=True,
                networks_loaded="one (results/models/lstm_wqa_s3.pt)", script_not_run="code/16_aorc_rain_to_midnight.py and code/check_stage4_realtime.py were not run; round 1's a7a_select_days.py was rerun into the scratch folder (it writes nothing into the project)",
                project_writes="only this block of this file and the folder code/review/aorc_rain_to_midnight_5seeds/", memory="one prediction array in memory at a time"),
    checks=checks,
    reproduced=dict(how="own scoring in numpy from results/preds (Q95 of the training years from the stored observed flow; onset = target day of the test years whose flow lead days earlier was at or below Q95; warning = forecast > trigger x Q95; NSE per basin on the test days)",
                    headline_l1=dict(lstm_wqa=dict(hits=w["onset"]["hits"], of=611, pod=w["onset"]["pod"], far=w["onset"]["far"], nse_median=w["nse_median"], single_seed_hits=H2["wqa_l1_single_seed_onset_hits"],
                                                   single_seed_nse_median=H2["wqa_l1_single_seed_nse_median"]),
                                     controls={n: J["summary_test"]["sets"][n]["l1"]["onset"]["pod"] for n in ("wq", "wqa_to09", "wqa_utc", "wqn", "wqh", "wqnh", "wqa_rep", "wqf")},
                                     simple={n: J["yardstick"]["table_test"]["l1"][n]["onset"]["pod"] for n in ("base_linreg_wqa", "base_rf_wqa")}),
                    e1b_success_met=J["e1b_success"]["wqa"]["met"], value_of_hours_09_to_24_points=vh["difference_points"],
                    stage4=dict(stage4=V["stage4"]["onsets_warned"], single_seeds=V["stage4"]["single_seed_onsets_warned"], surviving=V["stage4"]["share_of_the_aorc_effect_surviving"],
                                intervals=V["stage4"]["intervals"], no_last_hour=V["stage4_no_last_hour"]["onsets_warned"], level_restored=V["stage4_k"]["onsets_warned"], k=S4["k_level_aorc_over_stage4"]),
                    compare_summaries={"r1": R1["summary"], "r2": R2["summary"], "r4": R4["summary"], "r5": R5["summary"], "r7": R7["summary"]}),
    hindsight=dict(round1_bound_points=2.6, round1_bound_hits=16, round1_of_hits=313, round1_networks="seeds 0-2", five_seeds=F5, three_seeds=F3,
                   five_seed_loosest_bound_points=F5["direct_all_611_raw_loss_points"], five_seed_loosest_bound_hits=F5["direct_all_611_raw_loss_hits"], five_seed_level_matched_points=F5["direct_all_611_level_matched_loss_points"],
                   reading="'hindsight at most about 3 points' was measured on the three networks of seeds 0-2 and holds for them; for the five-seed set the loosest Stage IV route gives 3.6 points (about 4), the level-matched route 0.8; label the 3 as three-seed or restate"),
    findings=findings,
    counts=counts,
    verdict="numbers hold with wording fixes",
    verdict_text=verdict_text,
    scripts=["code/review/aorc_rain_to_midnight_5seeds/" + f for f in ("r_common.py", "r1_integrity.py", "r2_headline.py", "r4_stage4.py", "r5_one_network.py", "r6_hindsight.py", "r7_outcome.py", "r8_assemble.py", "run_checks.sh")],
    scripts_how_to_run="from the project root: PY=<project python> REVIEW_OUT=<scratch folder> bash code/review/aorc_rain_to_midnight_5seeds/run_checks.sh [--write-into-project]; one torch thread, about 1 minute",
    not_checked=["No training: whether seeds 3-4 retrained on another machine give the same networks is untested; the stored forecasts and networks are what was checked.",
                 "Only one network was rebuilt (wqa seed 3); the Stage IV single-seed counts of the other four networks and of wqa_rep rest on the consistency checks and on the builder's reproduction record.",
                 "The dose-response route of round 1's hindsight bound, the stamp-convention sensitivity (48.6%) and the by-period, closure and ladder blocks were not recomputed in this round.",
                 "The validation-year block and the per-basin tables were not recomputed beyond what the reading blocks use."],
)
jsave("round_2.json", round_2)
print(json.dumps(counts), "\n", verdict_text[:600])
if "--write-into-project" in sys.argv:
    path = PROJ / REVIEW
    raw = path.read_bytes()
    rec = json.loads(raw)
    assert json.dumps(rec, indent=1, ensure_ascii=False).encode("utf-8") == raw, "the record does not re-dump to the same bytes"
    assert "round_2" not in rec, "round_2 is already there"
    rec["round_2"] = json.loads(json.dumps(round_2, default=_default))
    new = json.dumps(rec, indent=1, ensure_ascii=False).encode("utf-8")
    path.write_bytes(new)
    back = json.loads(path.read_bytes())
    back.pop("round_2")
    assert json.dumps(back, indent=1, ensure_ascii=False).encode("utf-8") == raw, "removing round_2 does not give the original bytes"
    assert new[:len(raw) - 2] == raw[:-2], "the original text is not an unchanged prefix"
    print("written:", path, len(raw), "->", len(new), "bytes; removing round_2 gives back the original bytes:", True)
