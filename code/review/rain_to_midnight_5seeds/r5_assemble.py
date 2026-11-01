"""Check 5: assemble the five-seed review block from OUT/*.json and, with --write-into-project, append it to results/review_rain_to_midnight.json under the key
ROUND (the record already holds round_2 and round_3, so this block is round_4). Everything already in the record stays byte-identical: the record is loaded,
the key added, and written back with the record's own layout (indent 1, ASCII, no newline at the end); afterwards the block is removed again in memory and the
bytes compared with the original. Without --write-into-project the block is written to OUT/round_block.json only."""
import time

from r_common import *

ROUND = "round_4"
R = jload(PROJ / RESULTS)
SCO, CP, BD, TX = jread("scores.json"), jread("compare.json"), jread("backup_diff.json"), jread("texts.json")
S = CP["summary"]
rel = lambda p: Path(p).resolve().relative_to(PROJ).as_posix()
script_sha = sha256_of(PROJ / SCRIPT)
results_sha = sha256_of(PROJ / RESULTS)
backup_sha = sha256_of(BACKUP / RESULTS)
assert R["integrity"]["script_sha256"] == script_sha, "the script changed after the results file was written"
written = R["passes"][-1]["finished_local_time"]
mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((PROJ / RESULTS).stat().st_mtime))
three_to_five = BD["changed"]
reading_changes = [c for c in three_to_five if c["where"].endswith(("reading", "clear_gain", "single_seeds_all_above"))]
claims = TX["seed_note_claims"]
false_claims = [k for k, v in claims.items() if not v]
clauses_ok = all(TX["clauses"].values())
bb = lambda name: f"{CP['by_block'][name]['matching']} of {CP['by_block'][name]['entries']}"

checks = [
    dict(id="5s-1", what="Integrity: sha256 of the 79 prediction sets read (7 own sets x 6 files, 8 stored sets x 4 files, 5 simple models) against integrity.sha256_of_saved_files of the results file (the 59 that it lists); the "
                         "45 seed files of seeds 0-2 (all 15 sets) against the backup of 30 October 10:41; the observed flow identical in all 79 files",
         result="pass", detail=f"{bb('seed averages and integrity hashes')} hash/average rows equal; seeds 0-2 identical to the backup: "
                               f"{sum(SCO['seeds_0_2_equal_to_backup'].values())} of {len(SCO['seeds_0_2_equal_to_backup'])}; observed flow identical: {all(SCO['obs_identical'].values())}"),
    dict(id="5s-2", what="Seed averages: every lstm_<set>.npz equals the day-by-day mean of its seed files (five for the seven sets of the script, three for the eight stored sets)",
         result="pass", detail="bit for bit in all 15 sets (max abs difference 0.0, same NaN pattern); the seven five-seed averages differ from the backup (remade), the eight stored averages are the backup's files"),
    dict(id="5s-3", what="summary_test of all 15 sets and yardstick.table_test of the 5 simple models recomputed with own numpy scoring (Q95 of 1980-10-01..2000-09-30 from the stored flow, test target days 2005-10-01..2014-09-30, "
                         "trigger 1.0, onsets = target day above Q95 whose flow lead days earlier was at or below it; median and mean NSE over the basins; pooled onset and all-day tables, leads 1-3)",
         result="pass", detail=bb("summary_test and the simple models' test table") + " values equal at 1e-9"),
    dict(id="5s-4", what="per_basin NSE and onset share of the 15 sets", result="pass", detail=f"{bb('per_basin')}"),
    dict(id="5s-5", what="seed_ranges (lowest, highest and by_seed of 7 scores at 3 leads, n_seeds 5 or 3), models.<set>.n_seeds_scored, models.<set>.per_seed, training rows (seeds 0-4 per own set)",
         result="pass", detail=f"{bb('seed_ranges, models.n_seeds_scored, per_seed, training rows')}"),
    dict(id="5s-6", what="Every seed rule of gain_rule (21 pairs x 3 leads): every single seed above / below every single seed (NSE and onsets), basins higher and lower, median basin difference, Wilcoxon p, p below 0.05, clear gain / loss, the reading words",
         result="pass", detail=f"{bb('gain_rule (every pair, every lead)')}"),
    dict(id="5s-7", what="The readings value_of_hours_09_to_24, value_of_evening_hours_beyond_utc_day, value_of_hours_09_to_utc_day_end, second_rain_product_alone, nldas_rain_against_era5_over_the_utc_day and both_products, "
                         "with the paired onset counts and sign tests (overlap_of_warned_onsets)",
         result="pass", detail=f"{bb('readings (value_of_*, second_rain_product_alone, nldas vs utc, both_products, paired onsets)')}"),
    dict(id="5s-8", what="e1_success (both families; the matched-window controls)", result="pass",
         detail=f"{bb('e1_success')}; met: wqh {CP['my_e1']['wqh']['met']}, wh {CP['my_e1']['wh']['met']} (as in the file and as with three seeds)"),
    dict(id="5s-9", what="closure (5 sets) and ladder (both families, matched controls)", result="pass",
         detail=f"closure {bb('closure')}; ladder {bb('ladder')}"),
    dict(id="5s-10", what="yardstick: lstm_against_simple (5 pairs x 3 leads, NSE reading of 11_key_figure.reading and the onset reading with its sign test) and new_input_given_to_each_model", result="pass",
         detail=f"{bb('yardstick (lstm_against_simple, new_input_given_to_each_model)')}"),
    dict(id="5s-11", what="outcome_full rebuilt from my numbers clause by clause (10 clauses) and found word for word; outcome_full = outcome + ' ' + outcome_caveat; the verdict word from my pooled table",
         result="pass" if clauses_ok and TX["outcome_parts"]["outcome_plus_caveat"] else "fail", detail=f"{sum(TX['clauses'].values())} of {len(TX['clauses'])} clauses found; composition {TX['outcome_parts']}; verdict {TX['verdict_mine']}"),
    dict(id="5s-12", what="What changed from three to five seeds: every boolean, count and reading word of gain_rule, the readings, both_products, e1_success, yardstick, ladder, models and verdicts compared between the backup (three seeds) and the file now; "
                         "the backup's by_seed values against seeds 0-2 now; the stored sets' summary_test",
         result="finding", detail=f"{len(three_to_five)} changed leaves, {len(reading_changes)} of them readings or the booleans behind them (list in findings); seeds 0-2 values unchanged in all 21 set-leads: {all(BD['seeds_0_2_values_unchanged'].values())}; "
                                  f"stored sets unchanged: {all(BD['ladder_summary_unchanged'].values())}; no verdict word changed; 23 validation-chosen triggers moved by 0.05-0.10 (the seed averages changed); e1_success unchanged (not met)"),
    dict(id="5s-13", what="Does the file say so? The SEED_NOTE disclosure (choices_made_after_looking_at_test_scores[-1]) and each of its factual claims checked against the files",
         result="fail" if false_claims else "pass", detail=f"{len(claims) - len(false_claims)} of {len(claims)} claims hold; false: {false_claims}; the file nowhere lists the readings that changed (findings 1 and 2)"),
    dict(id="5s-14", what="No text of the file still gives a seed count of three for the seven sets of the script (regex over every string of the file)", result="pass" if not TX["three_seed_wordings_about_the_seven_sets"] else "fail",
         detail=f"{len(TX['three_seed_wordings'])} matches, all about lstm_wq (3 trainings, correct) or the disclosure itself; 0 about the seven sets"),
    dict(id="5s-15", what="what_was_not_done[0] (the sentence on the independent review) against the record", result="fail",
         detail="it says the review exists 'in two rounds' (the record holds round_2 and round_3, now round_4) and does not say that rounds 1-3 scored seeds 0-2 only (finding 3)"),
]
n_pass = sum(c["result"] == "pass" for c in checks)
n_fail = sum(c["result"] == "fail" for c in checks)

findings = [
    dict(severity="must-fix", where="code/15_rain_to_midnight.py, SEED_NOTE (line 80-85), last sentence; results/rain_to_midnight.json choices_made_after_looking_at_test_scores[1]",
         text="The disclosure says 'the three-seed numbers remain in the independent review records (results/review_*.json) and in the earlier entries of passes'. passes holds no score at all (only trainings, seconds, "
              "time and script hash per pass), so the second half is false; the review records hold the three-seed headline (round 1: lstm_wqh 183 hits) but not every number of the file. "
              "FIX: replace 'and in the earlier entries of passes.' by 'and in a copy of the three-seed results file, results/archive/rain_to_midnight_3seeds_2026-10-30.json (copied unchanged from the backup of 30 October 10:41, "
              f"sha256 {backup_sha}); passes records only the trainings and times of each pass.' and copy that file there; then --rescore (nothing is trained). The same words are in the SEED_NOTE of scripts 16 and 18 (PATCHES.md B1): their reviews should check them."),
    dict(severity="should-fix", where="results/rain_to_midnight.json (no field); code/15_rain_to_midnight.py, SEED_NOTE or a new key five_seed_changes",
         text="The file does not say which readings changed when seeds 3-4 were added, and some changed: (a) outcome: onsets warned 1 day ahead 30% -> 29% (183 -> 180 hits of 611); NLDAS 'the same share (183 in both)' -> 'a higher share' than the rain to midnight; "
              "the gain from the hours after 09:00 9.5 -> 9.2 points (75/17 -> 76/20 onsets warned only by one side), the to09 control 20.5% -> 20.3%, the LSTM's gain from the new input 11.3 -> 10.8 points; "
              "(b) gain_rule.wh.against_control NSE at leads 1 and 2: 'clear gain' -> 'no clear difference' (lowest wh seed 0.436 now below the highest control seed 0.452 at lead 1; 0.275 below 0.292 at lead 2), although p fell (0.005 -> 0.0005; 0.016 -> 0.0015); "
              "(c) gain_rule.wh.against_rain_table_alone NSE at lead 2: 'clear gain' -> 'no clear difference'; (d) yardstick lstm_wqh against base_linreg_wqh and base_rf_wqh, onsets at lead 1: 'the LSTM warns of more onsets' -> 'no clear difference in the onsets' "
              "(sign test p 0.042 -> 0.089 and 0.041 -> 0.091); (e) 23 validation-chosen warning levels moved by one or two grid steps, no verdict word changed; e1_success unchanged (not met in both families). Every change weakens or leaves the E1 result, none favours it, "
              "which supports the note that the seeds were not added to rescue a result. FIX: append to SEED_NOTE: 'With five seeds instead of three the following readings changed (the three-seed values in brackets): onsets warned 1 day ahead with the rain to midnight 29% (30%; 180 against 183 hits), "
              "now below the NLDAS rain (183, the same share before); without the river gauge the NSE gain over the one-day-older control at leads 1 and 2 reads no clear difference (clear gain), and over the rain table alone at lead 2 no clear difference (clear gain); "
              "against ridge regression and the random forest on the same input the LSTM no longer warns of clearly more onsets 1 day ahead (sign test p 0.089 and 0.091; 0.042 and 0.041). No verdict and no success rule changed.' (or generate it from a stored three-seed copy, finding 1)."),
    dict(severity="should-fix", where="code/15_rain_to_midnight.py lines 1490-1503 (review_item); results/rain_to_midnight.json what_was_not_done[0]",
         text="The sentence on the independent review reads only REVIEW['round_2'] and says the review 'exists in two rounds' although the record holds round_3 (and now this round_4), and it does not say that rounds 1-3 scored the trainings of seeds 0-2 only, so a reader takes the "
              "five-seed numbers as covered by them. FIX: build the parts from every key round_N of REVIEW in order (rounds = sorted(k for k in REVIEW if k.startswith('round_'))), name each round's 'covers' field when present (this round has one), say "
              "'rounds 1-3 scored seeds 0-2; the five-seed numbers are covered by round_4' when round_4 exists, and else 'the five-seed numbers (seeds 3-4 added on 30 October 2026) have not been recomputed by a second agent'; then --rescore."),
    dict(severity="should-fix", where="results/rain_to_midnight.json readings second_rain_product_alone, nldas_rain_against_era5_over_the_utc_day, both_products.against_nldas_rain, both_products.against_rain_table_alone; code/15_rain_to_midnight.py pair_reading (line 833)",
         text="difference_by_seed_points and difference_by_seed pair seeds with zip, so for a five-seed set against a stored three-seed set they hold three entries (seeds 0-2) and nothing says that seeds 3-4 are left out. FIX: in pair_reading add "
              "seeds_paired=list(range(min(len(sT['by_seed']['onset_pod']), len(sC['by_seed']['onset_pod'])))) to both the onsets and nse dicts; then --rescore."),
    dict(severity="note", where="results/rain_to_midnight.json notes[2] (hour stamps)",
         text="The note sets the review's check 3g (-0.7 and -1.3 points of onsets when the series of the opposite hour convention is fed to the networks; measured in October on the networks of seeds 0-2) beside the five-seed gains 10.8 and 14.1 points. The comparison still holds in size, "
              "but the sentence should say '(measured by the review on the networks of seeds 0-2)'."),
    dict(severity="note", where="e1_success, gain_rule, readings",
         text="The seed rule now compares five seeds of the sets of this script with three seeds of the stored sets (wq, wqn, w, wn): more seeds on one side make 'every seed above' harder for that side and 'control flat' easier for a five-seed control. The rule is applied as written "
              "and the n_seeds fields show the counts; a reader should know the two sides are unequal."),
    dict(severity="note", where="whole project (grep)",
         text="No document outside the logs and the review files quotes the three-seed phrases of outcome_full ('the same share (30%', '11.3 to the LSTM', '9.5 points comes from'); documents built from final_report_numbers.json should still be rebuilt from the five-seed file before they are quoted."),
    dict(severity="note", where="results/review_rain_to_midnight.json",
         text=f"The brief asked for a block 'round_2', but the record already held round_2 (7 October, the extension) and round_3; to keep them byte-identical this block is '{ROUND}'."),
]
counts = dict(checks_listed=len(checks), checks_passed=n_pass, findings_not_pass_fail=sum(c["result"] == "finding" for c in checks), failed=n_fail,
              must_fix=sum(f["severity"] == "must-fix" for f in findings), should_fix=sum(f["severity"] == "should-fix" for f in findings), notes=sum(f["severity"] == "note" for f in findings),
              numerical_mismatches=len(S["not_matching"]), values_compared=S["entries"])
verdict = "numbers hold; disclosure needs one correction"
verdict_text = (f"Every number of results/rain_to_midnight.json that I recomputed from results/preds with my own numpy scoring equals the file's ({S['entries']} values at 1e-9, 0 mismatches): summary_test of all 15 sets and the "
                "simple models, per_basin, the five-seed ranges and by_seed lists, every seed rule of gain_rule with its basins and Wilcoxon p, the readings value_of_hours_09_to_24 / value_of_evening_hours_beyond_utc_day / "
                "09_to_utc_day_end / second_rain_product_alone / nldas vs utc / both_products with their paired onsets, e1_success, closure, ladder and the yardstick. The seven seed averages are bit for bit the mean of their five seed "
                "files, seeds 0-2 are byte-identical to the backup, the stored sets are unchanged, and the ten clauses of outcome_full come from these tables word for word (onsets 1 day ahead 19% -> 29%, control 21%, NLDAS 30% higher "
                "than 29%, five trainings 27.5-31.6%, success rule not met). No text gives a seed count of three for the seven sets. The SEED_NOTE disclosure is there and 12 of its 13 checkable claims hold, but it says the three-seed "
                "numbers remain 'in the earlier entries of passes', which hold no scores (must-fix), and the file does not list the readings that changed with seeds 3-4: headline 30% -> 29% and NLDAS now ahead, the NSE gain of wh over its "
                "control (leads 1-2) and over w (lead 2) from clear gain to no clear difference, and the LSTM's onset lead over ridge and forest from 'more onsets' to 'no clear difference'. All of these changes go against the E1 result, "
                "none for it. The sentence on the review in what_was_not_done still counts two rounds and does not say the earlier rounds covered seeds 0-2 only.")
stored = {rel(PROJ / p): h for p, h in SCO["sha256"].items()}
block = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not done: this is an AI review; the team's own check (team.json) is still open",
    when=f"2026-10-30, about 13:25 to 14:00 local time, after the rescore of 12:40-12:42 that wrote the results file ({written})",
    dates=dict(review="2026-10-30", results_file_written=f"{written} (file modified {mtime})"),
    covers="the five-seed numbers of results/rain_to_midnight.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC)",
    what="independent recomputation of the five-seed numbers of code/15_rain_to_midnight.py (E1) from results/preds with the reviewer's own numpy code, the seed rules and readings, e1_success, the seed averages, the outcome_full sentence, "
         "the changes from three to five seeds against the backup of 30 October 10:41, the SEED_NOTE disclosure and the seed-count wording",
    earlier_rounds_intact="round 1, round_2 and round_3 are byte-identical (the record without this key gives back the original bytes; checked after writing)",
    record_script_sha256=script_sha,
    files_reviewed={RESULTS: dict(sha256=results_sha, modified=mtime), SCRIPT: dict(sha256=script_sha),
                    "../Project-Flood_backup_2026-10-30/results/rain_to_midnight.json (three-seed state)": dict(sha256=backup_sha),
                    "scratchpad fiveseeds/PATCHES.md (the builder's patch and run log, outside the project)": dict(read=True)},
    stored_sets_reviewed=stored,
    method=dict(own_code_folder="code/review/rain_to_midnight_5seeds/ (outputs in the session scratchpad, REVIEW_OUT)", numpy_only_scoring=True, engine_not_used="flood.evaluate and the script were not imported or run",
                threads=1, nothing_trained=True, memory="one prediction set in memory at a time (scores kept, forecasts dropped)", project_writes="only this folder and this block",
                python="<venv> (Python 3.11.15, numpy 1.26.4, scipy for wilcoxon and binomtest), PYTHONUTF8=1, from the project root",
                definitions="Q95 per basin = numpy nanquantile 0.95 of the stored observed flow 1980-10-01..2000-09-30; test target days 2005-10-01..2014-09-30 with flow and forecast; high = flow > Q95; warning = forecast > 1.0 x Q95; "
                            "onset = high target day whose flow lead days earlier was <= Q95 (NaN earlier flow = no onset); tables pooled over the 12 basins; NSE per basin on the test days, median over basins"),
    checks=checks,
    reproduced=dict(summary=S, by_block=CP["by_block"], tolerance=1e-9,
                    headline_lead_1=dict((n, dict(nse_median=round(SCO["scores"][f"lstm_{n}"]["l1"]["nse_median"], 4), onset_hits=SCO["scores"][f"lstm_{n}"]["l1"]["onset"]["hits"], onset_pod=round(SCO["scores"][f"lstm_{n}"]["l1"]["onset"]["pod"], 4),
                                                         onset_far=round(SCO["scores"][f"lstm_{n}"]["l1"]["onset"]["far"], 4), onset_seed_range=[round(x, 4) for x in CP["ranges"][n]["l1"]["onset_pod"]], n_seeds=len(SEEDS[n])))
                                         for n in ("wqh", "wqh_older", "wh", "wh_older", "wqh_to09", "wqh_utc", "wqnh", "wq", "wqn", "wqf", "w", "wn")),
                    simple_lead_1={n: dict(onset_hits=SCO["scores"][n]["l1"]["onset"]["hits"], onset_pod=round(SCO["scores"][n]["l1"]["onset"]["pod"], 4), nse_median=round(SCO["scores"][n]["l1"]["nse_median"], 4)) for n in ("base_linreg_wqh", "base_rf_wqh")},
                    seed_averages=SCO["seed_average_check"], e1_success={k: dict(met=v["met"], control_flat=v["control_flat"], every_seed_above_every_seed_with_nldas_rain=v["every_seed_above_every_seed_with_nldas_rain"]) for k, v in CP["my_e1"].items()},
                    outcome_full_clauses=TX["clauses"]),
    three_to_five_seed_changes=dict(n_changed_leaves=len(three_to_five), changed=three_to_five, headline=BD["headline"], outcome_full_three_seeds=BD["texts"]["outcome_full"][0]),
    seed_note_claims=claims,
    findings=findings,
    counts=counts,
    verdict=verdict,
    verdict_text=verdict_text,
    scripts=["code/review/rain_to_midnight_5seeds/r_common.py", "code/review/rain_to_midnight_5seeds/r1_scores.py", "code/review/rain_to_midnight_5seeds/r2_compare.py", "code/review/rain_to_midnight_5seeds/r3_backup_diff.py",
             "code/review/rain_to_midnight_5seeds/r4_texts.py", "code/review/rain_to_midnight_5seeds/r5_assemble.py", "code/review/rain_to_midnight_5seeds/run_checks.sh"],
    not_checked=["The saved networks of seeds 3-4 were not rebuilt from their .pt files (the script's reload_check records them; it was not repeated).",
                 "The validation-year numbers (validation_used_for_no_choice) and the verdicts block at the validation-chosen levels were not recomputed; only that no verdict word changed from three to five seeds.",
                 "by_period, the figure figures/rain_to_midnight.png and the fixed text of the notes beyond the clauses listed were not checked.",
                 "Whether seeds 3-4 trained on Windows would equal Mac trainings of the same seeds (they would not be expected to bit for bit) was not tested; the training records only were read.",
                 "That the five-seed plan was promised at the proposal talk of 15 October could not be checked; schedule.py (and the archived proposal builders of 14 October) name 'five trainings per model where the NSE decides'."])
jsave("round_block.json", block)
print(json.dumps(counts), verdict)
if "--write-into-project" in sys.argv:
    f = PROJ / RECORD
    orig = f.read_bytes()
    rec = json.loads(orig.decode("utf-8"))
    assert json.dumps(rec, indent=1, ensure_ascii=True).encode("utf-8") == orig, "the record's layout is not indent 1 / ASCII / no final newline"
    assert ROUND not in rec, f"{ROUND} is already in the record"
    from r_common import _default

    def nan_to_null(o):
        if isinstance(o, float) and o != o:
            return None
        if isinstance(o, dict):
            return {k: nan_to_null(v) for k, v in o.items()}
        if isinstance(o, list):
            return [nan_to_null(v) for v in o]
        return o
    rec[ROUND] = nan_to_null(json.loads(json.dumps(block, default=_default)))
    new = json.dumps(rec, indent=1, ensure_ascii=True).encode("utf-8")
    f.write_bytes(new)
    back = json.loads(f.read_bytes().decode("utf-8"))
    back.pop(ROUND)
    same_bytes = json.dumps(back, indent=1, ensure_ascii=True).encode("utf-8") == orig
    prefix = new[:len(orig) - 2] == orig[:-2]
    print("written", len(orig), "->", len(new), "bytes; without the block the original bytes:", same_bytes, "; original bytes are the prefix:", prefix)
    assert same_bytes and prefix
