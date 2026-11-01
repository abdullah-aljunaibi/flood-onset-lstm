"""Round 2, assembly: the block round_2 of results/review_forecast_input.json from OUT/v1..v4. Without --write-into-project it writes OUT/round_2.json only.
With it, it loads the existing record, adds the key round_2 (every existing key and value kept), writes it back with the same layout (indent 1, ASCII, LF, no final newline)
and verifies that removing round_2 gives back the original bytes."""
from v_common import *

R = jload(RESULTS)
v1, v2, v3, v4 = jread("v1_avg_check.json"), jread("v2_compare.json"), jread("v3_networks.json"), jread("v4_readings_text.json")
import pickle
SHA = pickle.load(open(OUT / "v1_scores.pkl", "rb"))["SHA"]
H = v2["headline"]
pc = lambda x: f"{100 * x:.1f}%"
lk = lambda k: f"l{k}"
cmp_ = v2["summary"]
clauses = v4["outcome_clauses"]
ps = v4["passes"]
sn = v4["seed_note"]

reproduced = {
    "how": "own numpy scoring of every prediction file (code/review/forecast_input_5seeds/v1_score.py, one file at a time): Q95 per basin = 95th percentile of the stored observed flow over the training years of the "
           "set's own split (main 1980-10-01..2000-09-30; satellite era 2000-10-01..2008-09-30; equal to flood.thresholds to 0.0); onset = a test-year target day above Q95 whose flow lead days earlier was at or below Q95; "
           "warning = forecast > trigger x Q95; NSE per basin on the test days with a forecast; pooled tables over the 12 basins; the validation-chosen level replicated (levels 0.30..1.50 step 0.05, best pooled CSI of "
           "the validation years, a tie to the level nearest 1.0); average precision and ROC area own numpy; every value compared with results/forecast_input.json at 1e-9 (1e-12 for AP/ROC) in v2_compare.py",
    "summary": cmp_, "by_block": v2["by_block"],
    "headline_test_years_onsets_warned_fixed_level": {n: {lk(k): dict(pod=round(H[n][lk(k)]["pod"], 6), hits=H[n][lk(k)]["onset_hits"], onsets=H[n][lk(k)]["onsets"], far=round(H[n][lk(k)]["far"], 6),
                                                                       nse_median=round(H[n][lk(k)]["nse_median"], 6), seed_range_pod=[round(x, 6) for x in H[n][lk(k)]["seed_range_pod"]],
                                                                       seed_range_nse=[round(x, 6) for x in H[n][lk(k)]["seed_range_nse"]], trigger_chosen_on_validation=H[n][lk(k)]["chosen_trigger"],
                                                                       pod_at_chosen=round(H[n][lk(k)]["chosen_pod"], 6), far_at_chosen=round(H[n][lk(k)]["chosen_far"], 6)) for k in LEADS} for n in SHOWN},
    "e2_success": dict(by_lead={k: dict(met=v["met"], every_seed_above_wqa=v["every_seed_above_wqa"], every_seed_above_the_control=v["every_seed_above_the_control"]) for k, v in v2["e2_success_mine"].items()},
                       met_at_2_and_3_days=v2["met_at_2_and_3_days"], file_says=R["e2_success"]["met_at_2_and_3_days"], met_only_at_2_and_3_days=(not v2["e2_success_mine"]["l1"]["met"]) and v2["met_at_2_and_3_days"],
                       same_as_three_seeds=v4["readings_changed"]["e2_success_three"] == v4["readings_changed"]["e2_success_five"]),
    "value_of_the_forecast_points_fixed_level": {lk(k): round(100 * (H["wqaF_sub"][lk(k)]["pod"] - H["wqa"][lk(k)]["pod"]), 4) for k in LEADS},
    "value_of_the_forecast_points_chosen_level": {lk(k): round(100 * (H["wqaF_sub"][lk(k)]["chosen_pod"] - H["wqa"][lk(k)]["chosen_pod"]), 4) for k in LEADS},
    "e2b_points_fixed_level": {lk(k): round(100 * (H["wqaF_s"][lk(k)]["pod"] - H["wqa_s"][lk(k)]["pod"]), 4) for k in LEADS},
    "seed_averages": dict(sets=len(v1["AVG_CHECK"]), all_bit_for_bit=all(v["bit_for_bit"] for v in v1["AVG_CHECK"].values()), all_meta_seed_lists_right=all(v["seeds_match"] for v in v1["AVG_CHECK"].values()),
                          seeds_per_set={n: v["expected_seeds"] for n, v in v1["AVG_CHECK"].items()},
                          how="each lstm_<set>.npz against (((s0 + s1) + s2) + s3) + s4) / 5 of its seed files (/ 3 for wq, wqf), the order of flood.ensemble: equal bit for bit, NaN pattern included"),
    "networks_reloaded": dict(how="rebuilt from the .pt file alone (flood.Cfg/LSTMNet/Run.predict, one torch thread), given the input tables of the first review's own builder (code/review/forecast_input/f_tables.py, variant csv)",
                              cases=v3["cases"], all_bit_for_bit=all(c["bit_for_bit"] for c in v3["cases"].values()), thresholds_engine_vs_mine_max_abs_diff=v3["thresholds_engine_vs_mine_max_abs_diff"]),
    "outcome_clauses": dict(rebuilt=clauses["rebuilt"], found_word_for_word=clauses["found"], missing=clauses["missing"]),
}
rc = v4["readings_changed"]
readings_changed = dict(
    how="every leaf named reading / met / clear_gain / clear_loss / verdict / trigger compared between the three-seed file (Project-Flood_backup_2026-10-30/results/forecast_input.json, written 2026-10-28) and the file now",
    reading_words=[dict(where=a, three_seeds=b, five_seeds=c) for a, b, c in rc["reading_words"] if a.split("/")[-1] == "reading"],
    verdict_words=[dict(where=a, three_seeds=b, five_seeds=c) for a, b, c in rc["verdicts"]],
    validation_chosen_onset_triggers_changed=len(rc["onset_triggers_chosen_on_validation"]),
    e2_success=dict(three_seeds=rc["e2_success_three"], five_seeds=rc["e2_success_five"]),
    numbers_that_moved_most=dict(main_1day_points_fixed_level=["-6.7", "-8.0"], main_points_chosen_level=["+2.6/+25.3/+29.7", "+5.4/+17.0/+28.6"], e2b_points_chosen_level=["+23.6/+19.8/+12.5", "+15.5/+9.1/+18.6"],
                                 e2b_nse_1day=["+0.051, every seed above, 11 of 12 basins, p = 0.002: clear gain", "+0.042, seeds overlap (0.717-0.753 against 0.656-0.733), 10 of 12, p = 0.002: no clear difference"]),
    text_says_which_readings_changed=False)

checks = [
    dict(id="c1", what="Prediction files: 92 files (16 sets: seed averages + 5 single seeds, 3 for wq and wqf); observed flow identical in every file, dates contiguous 1980-10-01..2014-09-30, meta seed lists", result="pass"),
    dict(id="c2", what="Every seed average equals the mean of its seed files (summed in seed order, as flood.ensemble), 16 sets", result="pass", numbers=dict(bit_for_bit=reproduced["seed_averages"]["all_bit_for_bit"])),
    dict(id="c3", what="Q95 of both splits from the stored observed flow against flood.thresholds(flood.load_data())", result="pass", numbers=v3["thresholds_engine_vs_mine_max_abs_diff"]),
    dict(id="c4", what="summary_test, validation_used_for_no_choice.summary and per_basin of 16 sets, leads 1-3, both splits, fixed level", result="pass", numbers={b: v2["by_block"][b] for b in ("summary_test", "validation", "per_basin")}),
    dict(id="c5", what="Operating points chosen on the validation years (onsets and all days, CSI rule replicated), their test scores and the verdict words incl. single seeds", result="pass", numbers=v2["by_block"]["verdicts"]),
    dict(id="c6", what="seed_ranges (lo, hi, by_seed, n_seeds), models.n_seeds_scored and models.<set>.per_seed", result="pass", numbers=dict(seed_ranges=v2["by_block"]["seed_ranges"], models=v2["by_block"]["models"])),
    dict(id="c7", what="gain_rule (12 pairs) and the 51 pair blocks with with_set/without_set (onsets, NSE rule with Wilcoxon, overlap of warned onsets with the sign test)", result="pass", numbers=dict(gain_rule=v2["by_block"]["gain_rule"], n_pair_blocks=v2["n_pair_blocks"])),
    dict(id="c8", what="e2_success: met at 2 and 3 days only (not at 1 day), same as with three seeds", result="pass", numbers=reproduced["e2_success"]),
    dict(id="c9", what="value_of_the_forecast by_lead and operating_points (fixed level, validation-chosen level, equal false-warning budgets, average precision, ROC area; bootstrap point estimates; not the intervals)", result="pass", numbers=v2["by_block"]["value_of_the_forecast"]),
    dict(id="c10", what="operating_points_by_set, 16 LSTM sets x 3 leads", result="pass", numbers=v2["by_block"]["operating_points_by_set"]),
    dict(id="c11", what="Network reload spot-check: lstm_wqaP_s3, lstm_wqaF_s_s4 and the derived lstm_wqaP_forecastpred_s3, lstm_wqaF_sub_rawpred_s4, lstm_wqaF_sub_perfectpred_s4 from their networks", result="pass", numbers={k: v["bit_for_bit"] for k, v in v3["cases"].items()}),
    dict(id="c12", what="Training records of the 20 seed 3-4 networks (training list) against their .pt files: seconds, best update, updates run, seed, set", result="pass", numbers=ps["pt_records"]),
    dict(id="c13", what="integrity: the 112 recorded sha256 equal the files now; script sha256 now = passes[-1] = integrity.script_sha256", result="pass", numbers=dict(hashes=ps["integrity_hashes_equal_files_now"], script=ps["last_pass"]["equal"])),
    dict(id="c14", what="26 number clauses of outcome / outcome_caveat rebuilt from my numbers and found word for word", result="pass", numbers=reproduced["outcome_clauses"]),
    dict(id="c15", what="Readings that changed from three to five seeds, and whether the text says so", result="finding", numbers=dict(reading_words=len(readings_changed["reading_words"]), verdict_words=len(readings_changed["verdict_words"]), triggers=readings_changed["validation_chosen_onset_triggers_changed"])),
    dict(id="c16", what="No three-seed sentence left for these sets", result="finding", numbers=dict(hits=len(v4["seed_wording_hits"]))),
    dict(id="c17", what="passes bookkeeping (n_trainings 30+20+0 = 50, training_seconds 4576.5+1411.0 = 5987.5, runtime 4827.4+1474.0+460.6 = 6762.0, the stopped pass, the second attempt)", result="finding",
         numbers={k: ps[k] for k in ("sum_n_trainings_ok", "sum_training_seconds_ok", "runtime_sum_ok", "stopped_pass", "second_attempt_recorded", "derived_set_files")}),
    dict(id="c18", what="SEED_NOTE disclosure in choices_made_after_looking_at_test_scores", result="finding", numbers=sn),
]

S18 = "code/18_forecast_input.py"
findings = [
    dict(severity="must-fix", where="results/forecast_input.json outcome_caveat, last clause (hence outcome_full, which every document must quote); source: code/18_forecast_input.py line 1623 (the f-string ending clause c7)",
         text="The clause now reads 'five trainings per set, so the single-training ranges are the only repetition (five would settle the 'mixed' readings of the NSE at 2 and 3 days).' The number was substituted, the sentence was not "
              "rewritten: five trainings were made and the NSE at 2 and 3 days is still 'mixed' (every single training of lstm_wqaF_sub above every one of lstm_wqa, higher in 8 and 9 of 12 basins, Wilcoxon p = 0.176 and 0.092: "
              "the basin test, not the seeds, leaves it open); and 'per set' is wrong for the stored controls lstm_wq and lstm_wqf (three). Fix: replace '(five would settle the 'mixed' readings of the NSE at 2 and 3 days).' "
              "by a computed clause, e.g. f\"{n_seeds_txt} trainings per set of this script ({NUMBER_WORD[3]} for the stored controls lstm_wq and lstm_wqf), so the single-training ranges are the only repetition; with "
              "{n_seeds_txt} the NSE at 2 and 3 days is still 'mixed' (every single training above, higher in {nb2} and {nb3} of 12 basins, Wilcoxon p = {p2:.3f} and {p3:.3f}: the basins, not the seeds, leave it open).\" "
              "with nb/p read from gain_rule['wqaF_sub_against_wqa'][l2/l3]['nse']; then --rescore."),
    dict(severity="should-fix", where="results/forecast_input.json choices_made_after_looking_at_test_scores[1] (SEED_NOTE, code/18_forecast_input.py lines 71-76) or a new notes entry",
         text="No text says which readings the two added seeds changed; outcome, outcome_caveat and notes were regenerated with the five-seed words, so a reader of this file cannot tell that, with seeds 0-2, "
              "(i) E2b's 1-day NSE (wqaF_s against wqa_s) read 'clear gain' (+0.051, every seed above, 11 of 12 basins, p = 0.002) and now reads 'no clear difference' (+0.042; seed ranges 0.717-0.753 against 0.656-0.733 overlap); "
              "(ii) E2a's 1-day NSE (wqaF_sub against wqa; also raw_vs_bias_corrected and retrained_vs_substituted.substituted) read 'mixed' and now 'no clear difference'; (iii) wqa_s against wq_s at 3 days, onsets, read 'clear gain' and now "
              "'no clear difference'; (iv) the validation-chosen comparisons moved a lot: main split +2.6/+25.3/+29.7 to +5.4/+17.0/+28.6 points, E2b +23.6/+19.8/+12.5 to +15.5/+9.1/+18.6 (20 chosen onset triggers changed); "
              "(v) 9 verdict words at the chosen levels (all high-flow days) and two yardstick readings ('lstm' to 'lstm_med' / 'lstm_avg'). e2_success is unchanged (met at 2 and 3 days, not at 1). Documents or the round-1 review "
              "quoting the three-seed readings (notes[15] of the old file: E2b 1-day NSE 'clear gain') are now out of date. Fix: append to SEED_NOTE a sentence listing these changes (they can be computed in the script by "
              "comparing the reading leaves with the previous results file before it is overwritten, as compare_results_json.py did)."),
    dict(severity="should-fix", where="SEED_NOTE (code/18_forecast_input.py line 76): 'the three-seed numbers remain in the independent review records (results/review_*.json) and in the earlier entries of passes'",
         text="passes holds timings and script hashes only, no scores, so 'and in the earlier entries of passes' is false; the round-1 review record holds only the headline numbers it recomputed (numbers.headline_test_years). "
              "The full three-seed file exists only in the backup outside the project (Project-Flood_backup_2026-10-30/results/forecast_input.json). Fix: replace that clause by 'the three-seed headline numbers remain in "
              "results/review_forecast_input.json (numbers.headline_test_years) and the complete three-seed file in the backup of 30 October 2026 (Project-Flood_backup_2026-10-30/results/forecast_input.json)', or copy that file into results/archive/ and name it."),
    dict(severity="should-fix", where="results/forecast_input.json passes[1].what, derived_set_files, notes (and scratchpad prior18.json, which supplied passes[1])",
         text="The second attempt (12:23:52-12:31:37, log results/logs/18_forecast_input_extend_2026-10-30_rerun.log, exit 1 at KeyError 'runtime_s', nothing trained) is recorded nowhere in the results file, yet it wrote "
              "files the file now reads: it remade the three derived seed averages (lstm_wqaF_sub_rawpred, lstm_wqaF_sub_perfectpred, lstm_wqaP_forecastpred) and wrote lstm_wqaF_sub_perfectpred_s3/_s4 and "
              "lstm_wqaP_forecastpred_s3/_s4; the first attempt also wrote lstm_wqaF_sub_rawpred_s3/_s4 (passes[1].what names only the networks and the seed averages). derived_set_files says 'left as it was' for all 18, "
              "true of the third pass only. The numbers are not affected (the averages equal the mean of their seed files bit for bit; the third pass compared every derived seed file with what the networks give; my reload of "
              "three of them is bit for bit). Fix: add to passes[1].what (the 'what' of prior18.json) 'and the derived files lstm_wqaF_sub_rawpred_s3/_s4'; add a note (script 18, EXTEND branch, when prior_passes is not empty) "
              "naming the second attempt, its log, that it trained nothing, the files it wrote, and its wall clock (465 s, not in runtime_s); then --rescore, or record it as a pass of kind 'stopped, nothing trained'."),
    dict(severity="should-fix", where="results/forecast_input.json passes[-1].what (code/18_forecast_input.py line 624)",
         text="The last pass is described as 'extension (--extend): the sets whose files were missing' with n_trainings 0: nothing was missing and nothing was trained; it recorded the stopped pass, checked the derived sets "
              "and wrote this file. Fix: when EXTEND and not trained_pairs, what = 'extension (--extend), nothing trained: the networks of the stopped pass (previous entry) read from their files, the derived sets "
              "checked, every score recomputed and this file written'."),
    dict(severity="should-fix", where="results/forecast_input.json what_was_not_done[0] (review_item, code/18_forecast_input.py lines 1846-1853)",
         text="It cites the round-1 counts (36 checks, 27 passed, 2 must-fix, 12 should-fix) without saying that the round-1 review covers seeds 0-2 only and none of the five-seed numbers; 'What this version ... changed after "
              "that review was not recomputed ... unless the file says so' is too vague when every number changed. Fix: if REVIEW has 'round_2', quote its counts and verdict and say it covers the five-seed numbers "
              "(compare REVIEW['round_2']['record_script_sha256'] with HASH_NOW); else say 'the review covers the three-seed version (seeds 0-2); the five-seed numbers of 30 October 2026 are not yet reviewed'."),
]
notes = [
    dict(severity="note", where="what_was_not_done[6] (code/18_forecast_input.py line 1865)", text="It still opens 'More seeds where the NSE decides:' under 'not done' although five seeds were done; its content (span 0.077 at lead 1 for lstm_wqa_s, recomputed) is right. Suggest 'Still more seeds ...'."),
    dict(severity="note", where="pair blocks with a three-seed control (forecast_without_rain_to_midnight.by_lead, closure against wqf, ...)", text="difference_by_seed_points / difference_by_seed hold three entries for a five-seed set against lstm_wq or lstm_wqf (zip truncates); the pairing of seed numbers across different networks is descriptive only. Say so in how_to_read or leave the lists out for those pairs."),
    dict(severity="note", where="outcome_caveat clause c7 (Stage IV)", text="The E1b review's numbers it quotes (86 to 95 of 171 onsets against 92 with AORC) are of the three wqa networks; results/stage4_realtime.json now holds the five-network numbers. The clause labels them as the review's, so it is not wrong; adding '(three networks; five-network numbers in results/stage4_realtime.json)' would avoid a mixed reading."),
    dict(severity="note", where="lstm_wqaF_sub_raw against lstm_wqaF_sub_rawpred", text="Seeds 1, 3 and 4 of the two are identical forecasts (same training trajectory, same stopping update); the averages differ only through seeds 0 and 2. Known from the design (shared trajectory), not an error."),
    dict(severity="note", where="readers of this record", text="round_2 is appended; the top-level keys (counts, verdict, must_fix ...) remain those of round 1, and code/18_forecast_input.py (review_item), code/report_sections/s0_summary.py, s4f_forecast.py (review_clause) and s8_limits.py read only those. They should read round_2 when present."),
    dict(severity="note", where="not recomputed", text="The block-bootstrap intervals (uncertainty, value_of_the_forecast.operating_points.*.block_bootstrap_fixed_level, average_precision_difference_ci95), closure, ladder, lead1_facts, forward_pass_tests, bias facts, yardstick refits and the figure were not recomputed in round 2 (their point estimates that are pair differences were); the round-1 methods apply to them unchanged."),
    dict(severity="note", where="documents outside the results file", text="Hand-written documents (README.md, ROADMAP.md, report texts) and earlier review records that quote E2 numbers were not searched; any that quote three-seed E2 numbers or readings need the builders' rerun or an edit."),
]
all_findings = findings + notes
counts = dict(checks_listed=len(checks), checks_passed=sum(c["result"] == "pass" for c in checks), checks_with_findings=sum(c["result"] == "finding" for c in checks), failed=0,
              must_fix=sum(f["severity"] == "must-fix" for f in all_findings), should_fix=sum(f["severity"] == "should-fix" for f in all_findings), notes=sum(f["severity"] == "note" for f in all_findings),
              numerical_mismatches=len(cmp_["not_matching"]), values_compared=cmp_["entries"] + clauses["rebuilt"], networks_reloaded=len(v3["cases"]), prediction_files_scored=len(SHA))
verdict = "numbers hold with wording fixes"
verdict_text = (f"Every five-seed number of results/forecast_input.json that I recomputed from results/preds with my own numpy scoring equals the file's ({cmp_['entries']} values at 1e-9, 0 mismatches): summary_test of the 16 sets at "
                "leads 1-3 on both splits (the satellite-era sets with Q95 of 2000-2008), the validation-chosen operating points and verdicts, the seed ranges and single seeds, gain_rule and the 51 pair blocks "
                "(NSE rule with Wilcoxon, onset seed rule, paired onsets), e2_success, value_of_the_forecast and operating_points_by_set (average precision and ROC area at 1e-12). Every seed average is the mean of its "
                "five (three) seed files bit for bit; five networks reloaded with one thread (two new seed-3/4 networks and three derived sets) reproduce their stored files bit for bit; the 112 recorded hashes equal the files. "
                "e2_success is met at 2 and 3 days only, as with three seeds: the forecast's gain at 2-3 days holds (+16.8/+23.9 points, every seed above), the 1-day loss at the fixed level is larger (-8.0 against -6.7). "
                "Two seeds changed readings that the text does not flag: E2b's 1-day NSE 'clear gain' is now 'no clear difference', E2a's 1-day NSE 'mixed' is now 'no clear difference', and the validation-chosen differences moved "
                "by up to 10 points. One must-fix: the last clause of outcome_caveat says five trainings 'would settle' the 2- and 3-day NSE readings, which with five are still 'mixed'. Bookkeeping: the passes sum correctly and the "
                "stopped pass is recorded with its trainings, but the second (KeyError) attempt, which wrote the derived five-seed files, is recorded nowhere, the last pass's 'what' is misleading, and the SEED_NOTE wrongly says "
                "the three-seed numbers remain in passes. The SEED_NOTE otherwise discloses the addition, its timing after the test scores and the machine, and its claims about versions and code/schedule.py check out.")

round_2 = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E2 is still open (team.json untouched by this review)",
    when="2026-10-30, about 13:25 to 14:00 local time, after the third --extend attempt of script 18 (12:32-12:40) on this Windows PC; script 17's extension was not running",
    dates={"review": "2026-10-30", "results_file_written": "2026-10-30 12:39:57 (file modification time; passes[-1].finished_local_time 2026-10-30 12:39)",
           "seeds_3_4_trained": "2026-10-30 11:55-12:19 (passes[1], the stopped first attempt)"},
    covers="the five-seed numbers of results/forecast_input.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC)",
    what="independent recomputation of the five-seed results of script 18 (E2): the ten sets of the script, the derived sets, the stored controls (lstm_wq, lstm_wqf three seeds; lstm_wqa five seeds from script 16); "
         "headline, validation-chosen operating points, seed ranges, readings, e2_success, value_of_the_forecast, operating points; the seed averages; a network reload spot-check; the readings that changed from three to five "
         "seeds against the backup; the passes bookkeeping and the SEED_NOTE disclosure",
    record_script_sha256=sha256(SCRIPT),
    files_reviewed={SCRIPT: sha256(SCRIPT), RESULTS: sha256(RESULTS), "backup: Project-Flood_backup_2026-10-30/results/forecast_input.json (three seeds)": sha256_file(BACKUP),
                    "results/logs/18_forecast_input_extend_2026-10-30.log": sha256("results/logs/18_forecast_input_extend_2026-10-30.log"),
                    "results/logs/18_forecast_input_extend_2026-10-30_rerun.log": sha256("results/logs/18_forecast_input_extend_2026-10-30_rerun.log"),
                    "results/logs/18_forecast_input_extend_2026-10-30_rerun2.log": sha256("results/logs/18_forecast_input_extend_2026-10-30_rerun2.log"), "code/schedule.py": sha256("code/schedule.py")},
    stored_sets_reviewed=dict(sorted(SHA.items())),
    networks_read={**{v["network"]: v["network_sha256"] for v in v3["cases"].values()}, "note": "plus the 20 seed 3-4 .pt files read for their training records (hashes in results/forecast_input.json integrity, equal now)"},
    method=dict(own_code_folder="code/review/forecast_input_5seeds/", torch_threads=1, nothing_trained=True, script_not_run="code/18_forecast_input.py was not run",
                project_writes="only that folder and the key round_2 of this record", python="<venv> (torch 2.2.2+cpu, numpy 1.26.4), PYTHONUTF8=1, from the project root",
                memory="prediction files scored one at a time; networks reloaded one at a time",
                reused="the input-table builder of the round-1 review (code/review/forecast_input/f_tables.py, f_aorc.py) for the network reload"),
    checks=checks, reproduced=reproduced, readings_changed=readings_changed, findings=all_findings, counts=counts, verdict=verdict, verdict_text=verdict_text,
    scripts=["code/review/forecast_input_5seeds/README.md", "code/review/forecast_input_5seeds/v_common.py", "code/review/forecast_input_5seeds/v1_score.py", "code/review/forecast_input_5seeds/v2_compare.py",
             "code/review/forecast_input_5seeds/v3_networks.py", "code/review/forecast_input_5seeds/v4_readings_text.py", "code/review/forecast_input_5seeds/v5_assemble.py", "code/review/forecast_input_5seeds/run_checks.sh"],
    not_checked=["No training; whether seeds 3-4 retrained on another machine give the same networks is untested (they are what was checked).",
                 "Bootstrap intervals, closure, ladder, lead1_facts, forward_pass_tests, the yardstick refits and the figure were not recomputed in this round.",
                 "The script version of the stopped pass (sha256 7c308c47...) was taken from PATCHES.md (reconstructed there two ways), not rebuilt here."],
)
jsave("round_2.json", round_2)
print(json.dumps(counts), verdict)
if "--write-into-project" in sys.argv:
    p = PROJ / REVIEW
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    assert json.dumps(d, indent=1, ensure_ascii=True).encode("utf-8") == raw, "layout of the record not reproduced"
    assert "round_2" not in d, "round_2 exists already"
    d["round_2"] = json.loads(json.dumps(round_2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    new = json.dumps(d, indent=1, ensure_ascii=True).encode("utf-8")
    back = json.loads(new.decode("utf-8")); back.pop("round_2")
    assert json.dumps(back, indent=1, ensure_ascii=True).encode("utf-8") == raw
    p.write_bytes(new)
    chk = json.loads(p.read_bytes().decode("utf-8")); chk.pop("round_2")
    ok = json.dumps(chk, indent=1, ensure_ascii=True).encode("utf-8") == raw
    print("written; removing round_2 gives the original bytes:", ok, "; original sha256", hashlib.sha256(raw).hexdigest())
