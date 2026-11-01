"""E4e review, round 2, assembly: the block round_2 of results/review_flow_state.json from OUT/h2_headline.json and h7_round2.json. Without --write-into-project it writes
OUT/round_2.json only. With it, it adds the key round_2 to the existing record (every existing key and value kept), writes it back with the record's own layout (indent 1,
UTF-8 without ASCII escapes, CRLF line ends, no final newline) and verifies that removing round_2 gives back the original bytes."""
from h_common import *

R = jload(RESULTS)
H, X = jread("h2_headline.json"), jread("h7_round2.json")
REVIEW = PROJ / "results" / "review_flow_state.json"
SCRIPTS = ["README.md", "run_checks.sh", "h_common.py", "h2_headline.py", "h7_round2.py", "h9_round2_assemble.py"]
M, ops = H["headline_mine"], H["operating_points_mine"]
l1 = lambda n: M[n]["l1"]

checks = [
    dict(check="own scoring (no flood.evaluate, no warning_policy) of the 9 sets and 45 single seeds against the file: summary_test and summary_val, per basin, seed ranges and by-seed lists, models.per_seed, readings of every pair (seed rule, Wilcoxon, onset rule), "
               "paired onsets and sign tests, closure, the plain / CSI / 30%-false triggers chosen on the validation years and the test scores and single seeds at them, verdicts, the delivery table, the success record, the stored sets against results/forecast_input.json (h2)",
         passed=not H["summary"]["not_matching"], values=H["summary"]["entries"], mismatches=len(H["summary"]["not_matching"])),
] + [dict(check=k, passed=v["passed"], detail=v["detail"]) for k, v in X["checks"].items()]
values = H["summary"]["entries"] + X["new_set_leaves_compared"]
reproduced = dict(
    lead1_trigger_1={n: dict(onset_pod_file=R["summary_test"]["sets"][n]["l1"]["onset"]["pod"], onset_pod_mine=l1(n)["onset"]["pod"], hits_file=R["summary_test"]["sets"][n]["l1"]["onset"]["hits"], hits_mine=l1(n)["onset"]["hits"],
                         onset_far_file=R["summary_test"]["sets"][n]["l1"]["onset"]["far"], onset_far_mine=l1(n)["onset"]["far"], nse_median_file=R["summary_test"]["sets"][n]["l1"]["nse_median"], nse_median_mine=l1(n)["nse_median"],
                         seed_range_file=R["seed_ranges"]["sets"][n]["l1"]["onset_pod"], seed_range_mine=[min(l1(n)["seed_onset_pod"]), max(l1(n)["seed_onset_pod"])]) for n in SHOWN},
    wqaH_s_minus_wqa_s_points=dict(file=R["readings"]["pairs"]["wqaH_s_vs_wqa_s"]["by_lead"]["l1"]["onset"]["gain_points"], mine=100 * (l1("wqaH_s")["onset"]["pod"] - l1("wqa_s")["onset"]["pod"]), three_seed_file=6.6),
    paired_onsets_l1={k: dict(file={f: R["readings"]["pairs"][k]["paired_onsets"]["l1"][f] for f in ("warned_only_by_first", "warned_only_by_second", "sign_test_p")},
                              mine={f: H["rows"][f"paired {k} l1 {f}"]["mine"] for f in ("warned_only_by_first", "warned_only_by_second", "sign_test_p")}) for k in ("wqaH_s_vs_wqa_s", "wqaWHF_s_vs_wqaF_s")},
    operating_points_l1={f"{n} {r}": dict(trigger=ops[f"{n} l1 {r}"]["trigger"], pod=ops[f"{n} l1 {r}"]["test_onset"]["pod"], far=ops[f"{n} l1 {r}"]["test_onset"]["far"],
                                          file=dict(trigger=R["operating_points"]["sets"][n]["l1"]["rules"][r]["trigger"], pod=R["operating_points"]["sets"][n]["l1"]["rules"][r]["test"]["onset"]["pod"], far=R["operating_points"]["sets"][n]["l1"]["rules"][r]["test"]["onset"]["far"]))
                         for n in ("wqaH_s", "wqa_s", "wqaWHF_s", "wqaF_s") for r in ("csi", "far_cap")},
    closure_wqaH_s_l1={k: dict(file=R["closure"]["sets"]["wqaH_s"]["by_lead"][0][k], mine=H["rows"][f"closure wqaH_s l1 {k}"]["mine"]) for k in ("onset_share_of_gap_closed", "share_of_nse_gap_closed")},
    success_rule=X["success_rule"],
    every_clause_of_outcome_full_found=all(v["found"] for v in X["clauses"].values()), clauses=X["clauses"],
    new_sets_unchanged=dict(leaves_compared=X["new_set_leaves_compared"], identical_to_the_29_october_file=True, files_identical=True),
    stored_sets_three_to_five_seeds_l1=X["stored_sets_three_to_five_seeds_l1"],
    go_scan=H["go_scan"])

findings = [
    dict(severity="must-fix", where="results/flow_state.json choices_made_after_looking_at_test_scores and outcome_caveat (code/22_flow_state.py CHOICES, the success clause); every document that quotes E4e",
         text="The file does not disclose that its three comparison sets (lstm_wqa_s, lstm_wqaF_s, lstm_wqaP_s) gained seeds 3 and 4 on 30 October 2026, after this file's test scores were known (+6.6 points on 29 October). "
              "Its only choices entry and the success clause ('five seeds were chosen at launch ... in one training pass before any test score was seen') are about the new sets. For E4e the change matters. "
              "The headline gain rose from +6.6 to +8.4 points only because the base fell: wqa_s 40.9% to 39.1% (137 to 131 hits), since its added seeds warn of 31.9% and 40.6%. wqaH_s did not change. "
              "The closure of the onset gap rose from 21% to 26% (NSE from 95% to 86%), and wqa_s's 30%-false point moved from 1.50 (14.9%, 24% false) to 1.45 (14.9%, 30% false). "
              "Fix: add a CHOICES entry, for example: 'The stored comparison sets lstm_wqa_s, lstm_wqaF_s and lstm_wqaP_s gained seeds 3 and 4 on 30 October 2026 on a second machine (script 18, see its SEED_NOTE), after this file's three-seed comparison was known (+6.6 points); "
              "the five-seed base is lower (39.1% against 40.9%), so the gain is now +8.4 points with wqaH_s unchanged; the success rule's verdict is the same (the best wqa_s seed is seed 2, 46.9%, an original seed).' Then rescore."),
    dict(severity="should-fix", where="results/flow_state.json readings.how_to_read and readings.pairs.<pair>.how_to_read (code/22_flow_state.py)",
         text="Every how_to_read says 'a five-seed set against a three-seed stored set gives three entries, seeds 3 and 4 of the new set are not in the list'. No such pair exists now: the stored sets have five seeds and every difference_by_seed_points list has five entries (h2: 0 of 42 lists shorter than the longer seed list). "
              "Fix: build the sentence from the seed lists, or drop it when every pair has equal seed lists."),
    dict(severity="should-fix", where="code/22_flow_state.py, the review guard (same_record, stored_changed, review_covers, review_line)",
         text="After round_2 is appended, the next --rescore will claim coverage on the wrong grounds. The record's sha256 changes, so same_record turns False and stored_sets_changed_since_the_review is no longer carried. stored_changed then compares only with the previous results file, which already holds the five-seed hashes, so it is empty. "
              "The script hash test passes (record_script_sha256 = the training pass), so review_covers becomes True and review_line quotes the ROUND-1 verdict and counts. Any later edit of the record would do the same, even one that finds the numbers wrong. "
              "Fix: compare STORED_NOW with the hashes the review recorded (round_2['stored_sets_reviewed'] when present) and take who, date, verdict and counts from the latest round."),
    dict(severity="note", where="results/flow_state.json e4e_success",
         text="The verdict is not decided by the added seeds. The rule fails because wqaH_s seeds 3 and 4 (46.6%, 45.7%) are not above wqa_s seed 2 (46.9%, an original seed). wqa_s seeds 3 and 4 (31.9%, 40.6%) are far below. "
              "met_with_first_three_seeds compares seeds 0-2 of wqaH_s (47.2-48.4%) with all five wqa_s seeds, as seed_count_note says. It would also be met three against three (wqa_s seeds 0-2: 33.1-46.9%). The sentence 'not met with the five seeds and would be with the first three' is right."),
    dict(severity="note", where="results/flow_state.json operating_points.sets.wqaP_s.l3 (single seeds)",
         text="Seed 3 of the stored perfect-rain set meets GO for onsets at lead 3 at trigger 1.0 (74.4% warned, 29.99% false; also at the CSI and 30%-false triggers, both 1.0). It is an upper-bound set and lead 3, and the file makes no claim about it. "
              "The figure's 'no bar here keeps the false share under 30% at trigger 1.0' concerns the seed means at lead 1 and holds (lowest 37.6%)."),
    dict(severity="note", where="figures/flow_state.png",
         text="Read by eye: header 'five seeds per new set, five per stored set', five dots per bar, every label with its false share, status line NOT YET REVIEWED, footer with the provisional-record clause. The hatched stored bars show the five-seed values (39%, 60%, 71%)."),
]
n_must, n_should, n_note = (sum(f["severity"] == s for f in findings) for s in ("must-fix", "should-fix", "note"))
counts = dict(checks_listed=len(checks), checks_passed=sum(c["passed"] for c in checks), failed=sum(not c["passed"] for c in checks), must_fix=n_must, should_fix=n_should, notes=n_note,
              numerical_mismatches=len(H["summary"]["not_matching"]), values_compared=values, outcome_clauses_rebuilt=len(X["clauses"]), outcome_clauses_found=sum(v["found"] for v in X["clauses"].values()))
verdict_text = (f"Every number of results/flow_state.json that depends on the stored sets reproduces exactly from results/preds with my own scoring ({H['summary']['entries']} values, 0 mismatches). "
                "wqaH_s warns of 159 of 335 onsets (47.5%) against 131 (39.1%) for the five-seed wqa_s, +8.4 points, with 45 onsets warned only with the river's state and 17 only without. wqaWHF_s gives 68.4% against 59.7% for wqaF_s. "
                "At the 30%-false point wqaH_s gives 37.6% (trigger 1.20) against 14.9% for wqa_s (1.45). The success rule is not met with five seeds and would be with the first three. "
                f"The six new sets' files are byte-identical to 29 October and their numbers, the delivery table and the data coverage are unchanged ({X['new_set_leaves_compared']} leaves). Every clause of outcome_full, rebuilt from my numbers, is in the text. "
                "The fixes are about the record: it does not disclose that the comparison sets gained seeds 3-4 after its test scores were known, which alone raised the headline gain from +6.6 to +8.4 points (must-fix); "
                "the how_to_read texts still describe three-seed stored sets; and the script's review guard would treat any later edit of the record as covering the numbers.")
round_2 = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E4e is still open; this AI review does not replace it",
    when=f"2026-10-30, about 13:45 to {time.strftime('%H:%M')} local time (Windows PC), after the rescore of 12:56",
    dates=dict(review="2026-10-30", results_file_written=f"{R['written']} (written; passes[-1].finished_local_time {R['passes'][-1]['finished_local_time']})"),
    covers="the five-seed numbers of results/flow_state.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC)",
    what=("Round 2 of the review of E4e after the stored comparison sets wqa_s, wqaF_s and wqaP_s became five-seed means and code/22_flow_state.py was rescored without training (--rescore --seeds 5). "
          "Every number that depends on the stored sets was recomputed from results/preds with the reviewer's own code; the new sets' own numbers and files were checked against the 29 October file; the words and the disclosure were checked against the tables."),
    record_script_sha256=sha256(SCRIPT),
    files_reviewed={f: sha256(f) for f in (SCRIPT, "code/flood.py", RESULTS, FIGURE, "results/flood_analysis.json", "results/forecast_input.json")},
    stored_sets_reviewed=X["stored_sets_reviewed"],
    method=dict(own_code_folder="code/review/flow_state_5seeds/ (a copy of the round-1 check h2 with the stored sets at five seeds and without the Turkey Creek reading, plus h7_round2.py and this assembler); outputs in $REVIEW_OUT (session scratchpad)",
                scoring="numpy from the obs and pred arrays of results/preds/*.npz; Q95 from 2000-10-01..2008-09-30; onsets = target days above Q95 whose flow lead days earlier was at or below Q95; warning = forecast > trigger x Q95; pooled over the 12 basins; triggers chosen on the validation years over 0.50-2.00",
                three_seed_reference="../Project-Flood_backup_2026-10-30/results/flow_state.json",
                networks="not reloaded in round 2: their files are byte-identical to those round 1 rebuilt bit for bit; the hourly-flow columns (data only) were not rebuilt", nothing_trained=True, torch_threads=1,
                project_writes="only code/review/flow_state_5seeds/ and the key round_2 of this record"),
    checks=checks, reproduced=reproduced, findings=findings, counts=counts, verdict="numbers hold with wording fixes", verdict_text=verdict_text,
    scripts=[f"code/review/flow_state_5seeds/{s}" for s in SCRIPTS],
    not_checked=["the networks were not reloaded and the hourly columns not rebuilt (files unchanged since round 1, which reproduced them bit for bit)",
                 "the Turkey Creek fallback-day reading of round 1 (it depends on the hourly columns)", "the figure was read by eye, not on its pixels"])
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
