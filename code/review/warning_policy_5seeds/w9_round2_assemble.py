"""Round 2, assembly: the block round_2 of results/review_warning_policy.json from OUT/w1..w7. Without --write-into-project it writes OUT/round_2.json only.
With it, it loads the existing record, adds the key round_2 (every existing key and value kept), writes it back with the record's own layout (indent 1, UTF-8 without
ASCII escapes, CRLF line ends, no final newline) and verifies that removing round_2 gives back the original bytes."""
from w_common import *
import time

R = record()
W1, W2, W3, W4, W5, W6, W7 = (jread(f"{n}.json") for n in ("w1_headline", "w2_seeds", "w3_any_lead", "w4_bootstrap", "w5_stacker", "w6_text_go", "w7_round2"))
REVIEW = PROJ / "results" / "review_warning_policy.json"
SCRIPTS = ["README.md", "run_checks.sh", "w_common.py", "w1_headline.py", "w2_seeds.py", "w3_any_lead.py", "w4_bootstrap.py", "w5_stacker.py", "w6_text_go.py", "w7_round2.py", "w9_round2_assemble.py"]

checks = [
    dict(check="headline: 9 sets x 3 leads x 2 rules, trigger chosen on validation (own chooser, grid 0.50-2.00), validation and test tables, verdicts, average precision, Q95 vs flood.thresholds, test tables vs flood.evaluate (w1)",
         passed=not W1["mismatches"], values=W1["n_compared"], mismatches=len(W1["mismatches"])),
    dict(check="single seeds: own trigger per seed file, test at own and at the set's trigger, seed ranges, seed mean = mean of the seed files (w2)", passed=not W2["mismatches"], values=W2["n_compared"], mismatches=len(W2["mismatches"])),
    dict(check="any-lead window at the triggers of both rules, own implementation (w3)", passed=not W3["mismatches"], values=W3["n_compared"], mismatches=len(W3["mismatches"])),
    dict(check="bootstrap spot check: own 300 draws (seed 20261030) overlap every stored 2000-draw interval; the stated procedure (seed 0, 2000 draws) repeated gives the stored intervals exactly (w4)",
         passed=not W4["non_overlapping"] and not W4["record_procedure_repeated"]["mismatches"], values=W4["record_procedure_repeated"]["n_compared"], mismatches=len(W4["record_procedure_repeated"]["mismatches"]),
         detail=f"{W4['n_intervals']} intervals, none disjoint, largest endpoint gap {W4['max_endpoint_gap_points']:.2f} points (300 against 2000 draws)"),
    dict(check="stacker refit for the 6 main-split sets x 3 leads (own features, sklearn logistic), cut, coefficients, tables (w5)", passed=not W5["mismatches"], values=W5["n_compared"], mismatches=len(W5["mismatches"])),
    dict(check="made-up test-year values (forecasts and flows, five seed files): no trigger, cut or coefficient changes; test scores change (w5)", passed=W5["leakage"]["all_choices_unchanged"] and W5["leakage"]["all_test_scores_changed"]),
    dict(check="GO scan over every stored test table; verdicts recomputed; summary[] = sets[]; the preliminary diagnostic (rewritten 12:50) = the experiment (w6)",
         passed=W6["go_scan"]["matches_go_met_at"] and not W6["go_scan"]["verdict_disagreements"] and not W6["summary_vs_sets_mismatches"] and not W6["diagnostic_vs_experiment"].get("differences")),
] + [dict(check=k, passed=v["passed"], detail=v["detail"]) for k, v in W7["checks"].items()]
values = W7["n_values_compared"]
mism = W7["n_mismatches"]

L = lambda n, k=1: W1["sets"][n]["leads"][f"l{k}"]["rules"]
F = lambda n, rule, k=1: R["sets"][n]["leads"][f"l{k}"]["rules"][rule]
rep = []
for n in SETS:
    for rule in RULES:
        mine, fl = L(n)[rule], F(n, rule)
        rep.append(dict(set=n, lead=1, rule=rule, n_seeds=R["sets"][n]["n_seeds"], trigger_file=fl["trigger"], trigger_mine=mine["trigger"],
                        test_onset_pod_file=fl["test"]["onset"]["pod"], test_onset_pod_mine=mine["test"]["onset"]["pod"], test_onset_far_file=fl["test"]["onset"]["far"], test_onset_far_mine=mine["test"]["onset"]["far"],
                        hits_file=fl["test"]["onset"]["hits"], hits_mine=mine["test"]["onset"]["hits"], verdict_onsets_file=fl["verdict_onsets"], verdict_onsets_mine=mine["verdict_onsets"]))
a2 = W2["sets"]["lstm_wqa"]["leads"]["l1"]
reproduced = dict(
    lead1_operating_points=rep,
    outcome_numbers=dict(csi_level_wqa="48% warned of 611 onsets, 48% false (trigger 1.05; three seeds: 51% / 50% at 1.00)", far_cap_level_wqa="26% warned, test false share 32% (trigger 1.55; three seeds: 25% / 30% at 1.60)",
                         perfect_rain_far_cap="60% (three seeds 59%)", perfect_weather_far_cap="61% (unchanged, three seeds)",
                         seed_ranges_wqa_l1=W2["outcome_caveat_ranges_wqa_l1"], bootstrap_wqa_l1=W4["outcome_caveat_bootstrap_wqa_l1"],
                         stacker_l1=dict(wqa=[W5["refit"]["lstm_wqa"]["l1"]["test"]["onset"]["pod"], W5["refit"]["lstm_wqa"]["l1"]["test"]["onset"]["far"]], wqaF_sub=[W5["refit"]["lstm_wqaF_sub"]["l1"]["test"]["onset"]["pod"], W5["refit"]["lstm_wqaF_sub"]["l1"]["test"]["onset"]["far"]]),
                         any_lead_wqa=W3["outcome_caveat_any_lead_wqa"], far_cap_not_reached=W1["far_cap_not_reached"], go_onsets=W7["onset_go"], go_all_days_realistic=W7["realistic_all_days_go"]),
    every_clause_of_outcome_full_found=all(v["found"] for v in W7["clauses"].values()), clauses=W7["clauses"],
    unchanged_sets=dict(sets=["lstm_wq", "lstm_wqn", "lstm_wqf"], leaves_identical_to_the_three_seed_file=True, n_leaves=3 * 837),
    three_to_five_seeds_l1=W7["three_to_five_seeds_l1"])

findings = [
    dict(severity="must-fix", where="results/warning_policy.json choices_made_after_looking_at_test_scores (code/20_warning_policy.py, the choices list); every document that quotes E4a",
         text="The file does not disclose that the stored sets it scores changed after its test scores were known. Six of its nine sets (lstm_wqa, lstm_wqaF_sub, lstm_wqaP, lstm_wqa_s, lstm_wqaF_s, lstm_wqaP_s) are now five-seed means whose seeds 3 and 4 were trained on 30 October 2026 on a second machine, after this file's three-seed test scores were in print (51% at the CSI level, 25% at the 30%-false level). Scripts 16 and 18 record this in their own choices list (SEED_NOTE), but results/warning_policy.json has only the two entries about the 30%-false rule and the preliminary diagnostic. The words 'the five seeds of the rain-to-midnight set' in outcome_caveat and the seed counts in status do not say when or why the seeds were added. HANDOFF.md section 7 requires the disclosure. Fix: in code/20_warning_policy.py add a choices entry built from seeds_changed, for example: 'Seeds 3 and 4 of lstm_wqa, lstm_wqaF_sub, lstm_wqaP, lstm_wqa_s, lstm_wqaF_s and lstm_wqaP_s were trained on 30 October 2026 on a second machine (scripts 16 and 18, see their SEED_NOTE), after the three-seed test scores of this file were known (CSI level 51% warned with 50% false, 30%-false level 25%); these sets are now five-seed means; lstm_wq, lstm_wqn and lstm_wqf keep three seeds and their numbers are unchanged.' Then rescore."),
    dict(severity="should-fix", where="code/20_warning_policy.py, the review stale guard (prev_covered, review_line, status) and figure footer",
         text="The guard cannot register this round. prev_covered is true only if the results file's sha256 equals the top-level files_reviewed['results/warning_policy.json'] of the record (the 28 October file) or if the previous run's provenance says covers_this_run. Neither will hold again, so status stays 'NOT YET REVIEWED' after round_2 is appended. review_line also quotes only the top-level (round-1) verdict and must-fix count. Fix: if the record has 'round_2' and round_2['stored_sets_reviewed'] equals this run's inputs_sha256 (all 48 files), set review_covers = True. Take who, date, verdict and the must-fix count from round_2, and put 'round_2' in provenance.independent_review. Key coverage on the forecast files, not on the script hash, so that applying the wording fixes does not void the review. Otherwise keep the present logic."),
    dict(severity="should-fix", where="figures/warning_policy.png footer, 4th line (code/20_warning_policy.py line 185, draw(): fig.text)",
         text="The fourth footer line runs past the right edge of the figure. It ends at '... every stacker cut was chosen on', and 'the validation years;' is cut off, so the line that states the anti-leakage rule loses its object. The line is static, so the cut may predate the five-seed run. Fix: break the line before 'Every trigger' (or wrap the footer with textwrap at about 200 characters, keeping the status one-liner of phase 5) and check the PNG."),
    dict(severity="note", where="results/warning_policy.json outcome (the all-days clause, code/20_warning_policy.py line 308: '; '.join)",
         text="The list of models that meet GO on all high-flow days is joined with '; ' inside a sentence whose clauses are also separated by '; '. As a result 'it is met by the model with the AORC rain to midnight 1 day ahead at its CSI trigger (72% warned, 28% false); the model with the GEFS forecast (substituted) 1 day ahead at its CSI trigger (78% warned, 26% false); a decision layer ...' reads as three clauses. The numbers are right (my all-days tables: wqa l1 csi 72.0% / 27.6%, wqaF_sub l1 csi 77.5% / 26.4%; no other realistic set meets GO on all days). Fix: join with ' and ' (or ', ' and a final ' and ')."),
    dict(severity="note", where="results/warning_policy.json sets.lstm_wqaP_s.leads.l3.rules.{csi,far_cap}.per_seed.s3",
         text="With five seeds, one single network meets GO at a set's trigger: seed 3 of the perfect-rain satellite-era set at 3 days, 74.4% warned with 29.99% false at trigger 1.00 (both rules choose 1.00). Round 1 found no single network meeting GO at any set's trigger. The outcome clause 'no single training meets it' concerns the perfect-weather set at 2 days and still holds (no seed of lstm_wqf meets GO there). The new case is an upper-bound set, on its own split, and only 0.01 points under the false-share cap."),
    dict(severity="note", where="results/warning_policy.json outcome; code/report_sections/s0_summary.py HINDSIGHT_WORDING ('to stand beside every 51%')",
         text="With five seeds, lstm_wqa's validation-chosen CSI trigger moves from 1.00 to 1.05. The headline becomes 48% warned with 48% false (three seeds: 51% / 50%), the 30%-false level becomes 26% with a 32% test false share (25% / 30%), and the perfect-rain set's 30%-false level becomes 60% (59%). The hindsight wording now stands beside the 48%, which is correct. Every document that quotes E4a's '51%' must take the five-seed numbers from the file, and the comment on HINDSIGHT_WORDING should not tie the wording to the figure 51%."),
    dict(severity="note", where="results/warning_policy.json sets.lstm_wqaF_s.leads.l1.rules.far_cap",
         text="At the 30%-false rule, triggers 1.50 and 1.55 tie on the validation POD for lstm_wqaF_s at lead 1. The stated tie rule (better CSI) picks 1.55, as the file records. My chooser agrees."),
    dict(severity="note", where="figures/warning_policy.png labels",
         text="The five-seed rain-to-midnight point (48%, 48% false) now sits under the label of the retrained GEFS set: '+ GEFS forecast, retrained (own split)' overprints '+ AORC rain to midnight'. '+ AORC rain to midnight (own split)' also overprints '+ NLDAS rain to 17:30'. The points are placed right (they agree with summary[]), but two labels are hard to read."),
]
n_must, n_should, n_note = (sum(f["severity"] == s for f in findings) for s in ("must-fix", "should-fix", "note"))
counts = dict(checks_listed=len(checks), checks_passed=sum(c["passed"] for c in checks), failed=sum(not c["passed"] for c in checks), must_fix=n_must, should_fix=n_should, notes=n_note,
              numerical_mismatches=mism, values_compared=values, bootstrap_intervals_spot_checked=W4["n_intervals"], outcome_clauses_rebuilt=len(W7["clauses"]), outcome_clauses_found=sum(v["found"] for v in W7["clauses"].values()))
verdict_ = "numbers hold with wording fixes"
verdict_text = (f"Every number of results/warning_policy.json that depends on the stored sets reproduces exactly from results/preds with my own code ({values} values, 0 mismatches). "
                "This covers the 54 validation-chosen triggers (CSI and 30%-false rules), the test tables and verdicts, the per-seed triggers and seed ranges over the five seeds, the any-lead window, "
                "the stacker refit for wqa and wqaF_sub and the other main-split sets, and the stated bootstrap procedure (2000 draws, seed 0, intervals equal). My own 300-draw bootstrap overlaps all 54 intervals. "
                "The sets whose seed count did not change (wq, wqn, wqf) are identical leaf by leaf to the three-seed file. Every clause of outcome_full, rebuilt from my numbers, is in the text: "
                "48% of the 611 onsets at the CSI level (48% false) and 26% at the 30%-false level (32% false on the test years). Only the perfect-weather set at 2 days meets GO for onsets, and no single network meets it there. "
                "The made-up-values test shows that no choice depends on the test years. The fixes are about the record, not the numbers: the file does not disclose that seeds 3-4 of six of its sets "
                "were added after its test scores were known (must-fix); the script's stale guard cannot register this round; and one footer line is cut off at the right edge.")
stored = W7["stored_sets_reviewed"]
round_2 = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E4a is still open; this AI review does not replace it",
    when=f"2026-10-30, about 13:45 to {time.strftime('%H:%M')} local time (Windows PC), after the rescore of 12:54",
    dates=dict(review="2026-10-30", results_file_written=f"{R['finished_local_time']} (finished_local_time; file modified {W7['files_now']['results/warning_policy.json']['modified']})"),
    covers="the five-seed numbers of results/warning_policy.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC)",
    what=("Round 2 of the review of E4a after the six stored sets of scripts 16 and 18 became five-seed means and code/20_warning_policy.py was rerun without training. "
          "Every number that depends on the stored sets was recomputed from results/preds with the reviewer's own code; the words, the disclosure and the status were checked against the tables."),
    record_script_sha256=W7["files_now"]["code/20_warning_policy.py"]["sha256"],
    files_reviewed={f: v for f, v in W7["files_now"].items()},
    stored_sets_reviewed=stored,
    method=dict(own_code_folder="code/review/warning_policy_5seeds/ (copies of the round-1 checks w1-w6 with the bootstrap spot check at 300 draws, plus w7_round2.py and this assembler); outputs in $REVIEW_OUT (session scratchpad)",
                scoring="numpy and pandas from the obs and pred arrays of results/preds/*.npz: Q95 from the training years of each set's split; onsets = target days above Q95 whose flow lead days earlier was at or below Q95; warning = forecast > trigger x Q95; pooled over the 12 basins; flood.thresholds / flood.evaluate used only as cross-checks",
                three_seed_reference="../Project-Flood_backup_2026-10-30/results/warning_policy.json",
                nothing_trained=True, torch_threads=1, project_writes="only code/review/warning_policy_5seeds/ and the key round_2 of this record",
                python="<venv> (torch 2.2.2+cpu), PYTHONUTF8=1, PYTHONDONTWRITEBYTECODE=1, from the project root; prediction sets read one set at a time"),
    checks=checks, reproduced=reproduced, findings=findings, counts=counts, verdict=verdict_, verdict_text=verdict_text,
    scripts=[f"code/review/warning_policy_5seeds/{s}" for s in SCRIPTS],
    not_checked=["the sandbox rerun of code/20_warning_policy.py (round-1 w8) was not repeated: the JSON was rebuilt from my own numbers instead",
                 "the figure was read by eye (footer, labels, status line), not on its pixels",
                 "results/more_basins.json and script 17 (not an input of E4a)"])
jsave("round_2.json", round_2)
print(json.dumps(counts), verdict_)
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
