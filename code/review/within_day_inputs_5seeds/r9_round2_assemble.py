"""E4b review, round 2, assembly: the block round_2 of results/review_within_day_inputs.json from OUT/r3_columns.json, r2_headline.json and r7_round2.json. Without
--write-into-project it writes OUT/round_2.json only. With it, it adds the key round_2 to the existing record (every existing key and value kept), writes it back with the
record's own layout (indent 1, UTF-8 without ASCII escapes, CRLF line ends, no final newline) and verifies that removing round_2 gives back the original bytes."""
from r_common import *

R = jload(RESULTS)
H, C3, X = jread("r2_headline.json"), jread("r3_columns.json"), jread("r7_round2.json")
REVIEW = PROJ / "results" / "review_within_day_inputs.json"
SCRIPTS = ["README.md", "run_checks.sh", "r_common.py", "r3_columns.py", "r2_headline.py", "r7_round2.py", "r9_round2_assemble.py"]
M, ops = H["headline_mine"], H["operating_points_mine"]

checks = [
    dict(check="own scoring (no flood.evaluate) of the 7 sets and 35 single seeds against the file: summary_test and validation, per basin, seed ranges and by-seed lists, readings (seed rule, Wilcoxon, every pair), paired onsets and sign tests, closure, "
               "the CSI and 30%-false triggers chosen on the validation years and the test scores at them, verdicts, the success record, the little-rain slices, seed means = mean of the seed files, the stored sets' own results files (r2)",
         passed=not H["summary"]["not_matching"], values=H["summary"]["entries"], mismatches=len(H["summary"]["not_matching"])),
    dict(check="my scoring = flood.evaluate on lstm_wqaW (r2)", passed=H["engine_cross_check"]["my_scoring_equals_flood_evaluate"]),
    dict(check="the within-day columns rebuilt from data/aorc by another route = subdaily_inputs.add_inputs; the clock, the made-up-hours test (36 cases), fill days, Q95 (r3, rerun for the little-rain slices)",
         passed=bool(C3.get("my_columns_vs_subdaily_inputs", {}).get("all_basins_all_columns_within_1e_9"))),
] + [dict(check=k, passed=v["passed"], detail=v["detail"]) for k, v in X["checks"].items()]
values = H["summary"]["entries"] + X["new_set_leaves_compared"]
l1 = lambda n: M[n]["l1"]
reproduced = dict(
    lead1_trigger_1={n: dict(onset_pod_file=R["summary_test"]["sets"][n]["l1"]["onset"]["pod"], onset_pod_mine=l1(n)["onset"]["pod"], hits_file=R["summary_test"]["sets"][n]["l1"]["onset"]["hits"], hits_mine=l1(n)["onset"]["hits"],
                         onset_far_file=R["summary_test"]["sets"][n]["l1"]["onset"]["far"], onset_far_mine=l1(n)["onset"]["far"], nse_median_file=R["summary_test"]["sets"][n]["l1"]["nse_median"], nse_median_mine=l1(n)["nse_median"],
                         seed_range_file=R["seed_ranges"]["sets"][n]["l1"]["onset_pod"], seed_range_mine=[min(l1(n)["seed_onset_pod"]), max(l1(n)["seed_onset_pod"])]) for n in SHOWN},
    wqaW_minus_wqa_points=dict(file=R["readings"]["within_day_columns"]["by_lead"]["l1"]["onsets"]["difference_points"], mine=100 * (l1("wqaW")["onset"]["pod"] - l1("wqa")["onset"]["pod"]), three_seed_file=4.4),
    paired_onsets_wqaW_vs_wqa_l1=dict(file={k: R["paired_onsets"]["pairs"]["within_day_columns"]["l1"][k] for k in ("warned_only_by_first", "warned_only_by_second", "sign_test_p")},
                                      mine={k: H["rows"][f"paired within_day_columns l1 {k}"]["mine"] for k in ("warned_only_by_first", "warned_only_by_second", "sign_test_p")}),
    operating_points_l1={f"{n} {r}": dict(trigger=ops[f"{n} l1 {r}"]["trigger"], pod=ops[f"{n} l1 {r}"]["test_onset"]["pod"], far=ops[f"{n} l1 {r}"]["test_onset"]["far"],
                                          file=dict(trigger=R["operating_points"]["sets"][n]["l1"]["rules"][r]["trigger"], pod=R["operating_points"]["sets"][n]["l1"]["rules"][r]["test"]["onset"]["pod"], far=R["operating_points"]["sets"][n]["l1"]["rules"][r]["test"]["onset"]["far"]))
                         for n in ("wqaW", "wqa", "wqaW_older") for r in ("csi", "far_cap")},
    little_rain={k: dict(n=R["little_rain_onsets"]["slices"][k]["n"], warned=R["little_rain_onsets"]["slices"][k]["warned"]) for k in ("aorc_target_below_5mm", "aorc_target_at_least_5mm")},
    success_rule=dict(mine=H["success_mine"], with_three_seeds=H["success_with_three_seeds"], file_met=R["e4b_success"]["met"]),
    every_clause_of_outcome_full_found=all(v["found"] for v in X["clauses"].values()), clauses=X["clauses"],
    new_sets_unchanged=dict(leaves_compared=X["new_set_leaves_compared"], identical_to_the_29_october_file=X["checks"]["the new sets' own numbers are identical to the 29 October file (summary, validation, seed ranges, operating points, verdicts, per basin, models, pairs of new sets, little-rain shares)"]["passed"],
                            files_identical=X["checks"]["the new sets' files are byte-identical to those of the 29 October file (unchanged networks and forecasts)"]["passed"]),
    stored_sets_three_to_five_seeds_l1=X["stored_sets_three_to_five_seeds_l1"])

findings = [
    dict(severity="must-fix", where="results/within_day_inputs.json choices_made_after_looking_at_test_scores (code/21_within_day_inputs.py CHOICES); every document that quotes E4b",
         text="The file does not disclose that its three comparison sets (lstm_wqa, lstm_wqaF_sub, lstm_wqaP) gained seeds 3 and 4 on 30 October 2026, after this file's test scores were known (the +4.4 points of 29 October). "
              "Every comparison with them, and the success rule 'wqaW above wqa ... in every seed', now runs over five seeds of the base. Its second entry, 'One training pass ... went straight to five seeds per set; no --extend pass exists', is true only of the four new sets. "
              "The change moved the headline from +4.4 to +3.9 points (wqa 51.2% to 51.7%, 313 to 316 hits) and the closure of the gap to wqaP from 30% to 26%. It also moved wqa's validation-chosen CSI trigger from 1.00 (51% with 50% false) to 1.05 (48% with 48% false) and its 30%-false point from 1.60 (25%, 30% false) to 1.55 (26%, 32% false). "
              "Fix: add a CHOICES entry, for example: 'The stored comparison sets lstm_wqa, lstm_wqaF_sub and lstm_wqaP gained seeds 3 and 4 on 30 October 2026 on a second machine (scripts 16 and 18, see their SEED_NOTE), after this file's three-seed comparison was known (+4.4 points); "
              "they are five-seed means since; the success rule's verdict is the same with the three-seed base (the best wqa seed is seed 0, 55.3%).' Change the second entry to 'for the four sets of this script'. Then rescore."),
    dict(severity="should-fix", where="results/within_day_inputs.json design.success_rule and e4b_success.rule; readings.<pair>.how_to_read (code/21_within_day_inputs.py)",
         text="Two texts still describe three-seed stored sets. design.success_rule and e4b_success.rule say 'every single seed of lstm_wqa (stored, three seeds)', but the rule is now evaluated against five (e4b_success.n_seeds.wqa = 5). "
              "Every readings.<pair>.how_to_read says 'a five-seed set against a three-seed stored set gives three entries, seeds 3 and 4 of the new set are not in these lists', but seeds_paired is now [0, 1, 2, 3, 4] everywhere and the lists have five entries. "
              "Fix: build both texts from the seed lists (for example 'stored, {len(STORED[n][\"seeds\"])} seeds'; drop the three-entry sentence when every pair has equal seed lists)."),
    dict(severity="should-fix", where="code/21_within_day_inputs.py, the review guard (same_record, stored_changed, review_covers, review_line)",
         text="After round_2 is appended, the next --rescore will claim coverage on the wrong grounds. The record's sha256 changes, so same_record turns False and the earlier finding (stored_sets_changed_since_the_review) is no longer carried. stored_changed then compares the stored files only with the previous results file, whose stored_sets_read already holds the five-seed hashes, so it is empty. "
              "review_covers becomes True and review_line quotes the ROUND-1 verdict and counts (15 findings, 5364 values). The same happens after ANY edit of the record, including a round that finds the numbers wrong. "
              "Fix: compare STORED_NOW with the hashes the review itself recorded (round_2['stored_sets_reviewed'], or round 1's when there is no round_2), and take who, date, verdict and counts from the latest round."),
    dict(severity="note", where="results/within_day_inputs.json e4b_success",
         text="The success rule fails with the five-seed base as it failed with three. wqaW's seeds 0 and 3 (50.2%, 52.7%) sit below the best wqa seed, which is seed 0 (55.3%, an original seed), and below the control's best (53.8%). The added seeds 3-4 of wqa (52.5%, 49.3%) do not decide the verdict."),
    dict(severity="note", where="results/within_day_inputs.json little_rain_onsets; outcome_caveat",
         text="The slices are the same onsets (220 / 391), and the new sets' shares are unchanged. wqa's share is 80.5% in the little-rain slice (the same as with three seeds) and 35.5% outside it. The reading 'the gain sits in the wetter onsets, the opposite of the expectation' still holds by my numbers."),
    dict(severity="note", where="figures/within_day_inputs.png",
         text="Read by eye: header 'five seeds per new set, five per stored set', five dots on every bar, footer '52% of wqa' (taken from the JSON since the round-1 fix), status line NOT YET REVIEWED. Bar tops agree with summary_test by eye."),
]
n_must, n_should, n_note = (sum(f["severity"] == s for f in findings) for s in ("must-fix", "should-fix", "note"))
counts = dict(checks_listed=len(checks), checks_passed=sum(c["passed"] for c in checks), failed=sum(not c["passed"] for c in checks), must_fix=n_must, should_fix=n_should, notes=n_note,
              numerical_mismatches=len(H["summary"]["not_matching"]), values_compared=values, outcome_clauses_rebuilt=len(X["clauses"]), outcome_clauses_found=sum(v["found"] for v in X["clauses"].values()))
verdict_text = (f"Every number of results/within_day_inputs.json that depends on the stored sets reproduces exactly from results/preds with my own scoring ({H['summary']['entries']} values, 0 mismatches). "
                "wqaW warns of 340 of 611 onsets (55.6%) against 316 (51.7%) for the five-seed wqa, +3.9 points, with 64 onsets warned only with the within-day columns and 40 only without them. "
                "The control is flat (52.7%). At the 30%-false point wqaW gives 30.8% (trigger 1.50) against 25.9% for wqa (1.55) and 30.1% for the control (1.45). The success rule is not met. "
                "The little-rain slices are 220 / 391 onsets at 80% to 72% and 36% to 47%. The four new sets' files are byte-identical to 29 October and their own numbers are unchanged "
                f"({X['new_set_leaves_compared']} leaves). Every clause of outcome_full, rebuilt from my numbers, is in the text. The fixes are about the record: it does not disclose that the comparison sets gained seeds 3-4 after "
                "its test scores were known (must-fix), two texts still describe three-seed stored sets, and the script's review guard would treat any later edit of the record as covering the numbers.")
round_2 = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E4b is still open; this AI review does not replace it",
    when=f"2026-10-30, about 13:45 to {time.strftime('%H:%M')} local time (Windows PC), after the rescore of 12:55",
    dates=dict(review="2026-10-30", results_file_written=f"{R['written']} (written; passes[-1].finished_local_time {R['passes'][-1]['finished_local_time']})"),
    covers="the five-seed numbers of results/within_day_inputs.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC)",
    what=("Round 2 of the review of E4b after the stored comparison sets wqa, wqaF_sub and wqaP became five-seed means and code/21_within_day_inputs.py was rescored without training "
          "(--rescore --seeds 5 --with-forecast). Every number that depends on the stored sets was recomputed from results/preds with the reviewer's own code; the new sets' own numbers and files were checked against the 29 October file; the words and the disclosure were checked against the tables."),
    record_script_sha256=sha256(SCRIPT),
    files_reviewed={f: sha256(f) for f in (SCRIPT, "code/subdaily_inputs.py", "code/flood.py", RESULTS, "figures/within_day_inputs.png", "results/flood_analysis.json", "results/aorc_rain_to_midnight.json", "results/forecast_input.json")},
    stored_sets_reviewed=X["stored_sets_reviewed"],
    method=dict(own_code_folder="code/review/within_day_inputs_5seeds/ (copies of the round-1 checks r2 and r3 with the stored sets at five seeds, plus r7_round2.py and this assembler); outputs in $REVIEW_OUT (session scratchpad)",
                scoring="numpy from the obs and pred arrays of results/preds/*.npz; Q95 from 1980-10-01..2000-09-30; onsets = target days above Q95 whose flow lead days earlier was at or below Q95; warning = forecast > trigger x Q95; pooled over the 12 basins; flood.evaluate only as a cross-check",
                three_seed_reference="../Project-Flood_backup_2026-10-30/results/within_day_inputs.json",
                networks="not reloaded in round 2: their 44 files are byte-identical to those round 1 rebuilt bit for bit", nothing_trained=True, torch_threads=1,
                project_writes="only code/review/within_day_inputs_5seeds/ and the key round_2 of this record"),
    checks=checks, reproduced=reproduced, findings=findings, counts=counts, verdict="numbers hold with wording fixes", verdict_text=verdict_text,
    scripts=[f"code/review/within_day_inputs_5seeds/{s}" for s in SCRIPTS],
    not_checked=["the networks were not reloaded (their files are unchanged since round 1, which reproduced them bit for bit)", "the figure was read by eye, not on its pixels"])
jsave("round_2.json", round_2)
print(json.dumps(counts))
if "--write-into-project" in sys.argv:
    raw = REVIEW.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    dump = lambda x: json.dumps(x, indent=1, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")
    assert dump(d) == raw, "layout of the record not reproduced"
    assert "round_2" not in d, "round_2 exists already"
    d["round_2"] = json.loads(json.dumps(round_2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    new = dump(d)
    REVIEW.write_bytes(new)
    chk = json.loads(REVIEW.read_bytes().decode("utf-8")); chk.pop("round_2")
    print("written; removing round_2 gives the original bytes:", dump(chk) == raw, "| original sha256", hashlib.sha256(raw).hexdigest(), "| new sha256", hashlib.sha256(new).hexdigest())
