"""Assemble the block round_2 of results/review_more_basins.json from the outputs of r1-r5 ($REVIEW_OUT).

Without arguments it writes $REVIEW_OUT/round_2.json only. With --write-into-project it appends the key round_2 to results/review_more_basins.json
(refuses if the key exists): the record is loaded, round_2 added as its last key, written with the record's own layout (indent 1, ASCII, LF, no final
newline), and then checked: the record without round_2, dumped the same way, must give back the original bytes.
Run: python code/review/more_basins_5seeds/r6_assemble.py [--write-into-project]
"""
import datetime as dt

from r_common import *

REC_FILE = PROJ / "results" / "review_more_basins.json"
R = results()
r1, r3, r4, r5 = load("r1_integrity"), load("r3_compare"), load("r4_three_vs_five"), load("r5_networks")
s9 = r3["s9"]
mt = lambda f: dt.datetime.fromtimestamp((PROJ / f).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
f3, pc1 = (lambda x: f"{x:.3f}"), (lambda x: f"{100 * x:.1f}%")

stored = {}
for n in ALL_SETS:
    for f in files_of(n):
        stored[f"results/preds/{f}.npz"] = sha256(PRED / f"{f}.npz")
networks = {f"results/models/{n}_s{s}.pt": r1["i1"]["sha256_now"][f"results/models/{n}_s{s}.pt"] for n in NEW for s in SEEDS}
logs = ["results/logs/17_more_basins_extend_2026-10-30.log", "results/logs/17_more_basins_extend_2026-10-30b.log"]
files_reviewed = {"code/17_more_basins.py": sha256(PROJ / "code/17_more_basins.py"), "results/more_basins.json": sha256(RESULTS),
                  "backup: Project-Flood_backup_2026-10-30/results/more_basins.json (three seeds)": sha256(BACKUP),
                  "results/lstm_lobo.json": sha256(PROJ / "results/lstm_lobo.json"), "results/lstm_all.json": sha256(PROJ / "results/lstm_all.json"),
                  **{f: sha256(PROJ / f) for f in logs}, "code/schedule.py": sha256(PROJ / "code/schedule.py"), "figures/more_basins.png (not checked, hash only)": sha256(PROJ / "figures/more_basins.png")}

changed = r4["changed"]
trig = [c for c in changed if c["where"].endswith("/trigger")]
words = [c for c in changed if c["where"].endswith("/reading")]
bb = r3["by_block"]
n_leaves = sum(v["entries"] for v in bb.values())
n_match = sum(v["matching"] for v in bb.values())
net_vals = sum(v["n_values"] for v in r5.values())

# ---- the reproduced numbers (the work order's list) ------------------------------------------------------------------------------------------------------
same12 = {f"l{k}": {n: dict(nse_median=s9["same_12"][f"l{k}"][n]["nse"], onsets_warned=s9["same_12"][f"l{k}"][n]["onsets"],
                            **({"onset_hits": s9["same_12"][f"l{k}"][n]["hits"]} if "hits" in s9["same_12"][f"l{k}"][n] else {}))
                    for n in ("lstm_wq", "lstm63_wq", "lstm63eq_wq", "lstm_w", "lstm63_w")} for k in LEADS}
never = {f"l{k}": s9["never_seen"][f"l{k}"] for k in LEADS}
rule = lambda T, C, k, blk="gain_rule": (R[blk] if blk == "gain_rule" else R["unseen"]["against_the_leave_one_out_models"])[f"{T}_against_{C}"][f"l{k}"]
readings = {f"{T}_against_{C}": {f"l{k}": dict(nse=rule(T, C, k, b)["nse"]["reading"], onset=rule(T, C, k, b)["onset"]["reading"], wilcoxon_p=rule(T, C, k, b)["nse"]["wilcoxon_p"],
                                              n_basins_higher=rule(T, C, k, b)["nse"]["n_basins_higher"]) for k in LEADS}
            for T, C, b in (("lstm63_wq", "lstm_wq", "gain_rule"), ("lstm63_w", "lstm_w", "gain_rule"), ("lstm63eq_wq", "lstm_wq", "gain_rule"), ("lstm63eq_wq", "lstm63_wq", "gain_rule"),
                            ("unseen50_w_static", "lobo_w_static", "u"), ("unseen50_w_nostatic", "lobo_w_nostatic", "u"), ("unseen50_w_static", "lstm63_w", "u"))}
B3 = results(BACKUP)
three = dict(outcome=B3["outcome"], same_12_l1=dict(lstm_wq=B3["summary_test"]["sets"]["lstm_wq"]["l1"]["nse_median"], lstm63_wq=B3["summary_test"]["sets"]["lstm63_wq"]["l1"]["nse_median"],
                                                    lstm63_wq_onsets=B3["summary_test"]["sets"]["lstm63_wq"]["l1"]["onset"]["pod"], lstm63_wq_hits=B3["summary_test"]["sets"]["lstm63_wq"]["l1"]["onset"]["hits"]),
             never_seen_l1={n: B3["summary_test"]["sets"][n]["l1"]["nse_median"] for n in ("lobo_w_static", "unseen50_w_static", "lobo_w_nostatic", "unseen50_w_nostatic")})
outcome_full_numbers = dict(
    nse_1_day_same_12=[same12["l1"]["lstm_wq"]["nse_median"], same12["l1"]["lstm63_wq"]["nse_median"]], onsets_1_day_same_12=[same12["l1"]["lstm_wq"]["onsets_warned"], same12["l1"]["lstm63_wq"]["onsets_warned"]],
    onset_hits_false_alarms_csi_1_day={k: R["gain_rule"]["lstm63_wq_against_lstm_wq"]["l1"]["onset"][k] for k in ("hits_without", "hits_with", "false_alarms_without", "false_alarms_with", "csi_without", "csi_with")},
    paired_onsets_1_day=R["gain_rule"]["lstm63_wq_against_lstm_wq"]["l1"]["onset"]["paired"],
    never_seen_1_day=[never["l1"]["lobo_w_static"], never["l1"]["unseen50_w_static"], never["l1"]["lobo_w_nostatic"], never["l1"]["unseen50_w_nostatic"]],
    never_seen_single_trainings_with_properties=[x["median_nse"] for x in R["unseen"]["against_the_leave_one_out_models"]["unseen50_w_static_against_lobo_w_static"]["l1"]["single_trainings"]],
    never_seen_single_trainings_without_properties=[x["median_nse"] for x in R["unseen"]["against_the_leave_one_out_models"]["unseen50_w_nostatic_against_lobo_w_nostatic"]["l1"]["single_trainings"]],
    verdict=s9["verdict_lstm63_wq_l1_onset_plain"], outcome_sentence_rebuilt_word_for_word=r3["s8"]["equal_to_file"], outcome_full_is_outcome_plus_caveat=r3["s8"]["outcome_full_equals_outcome_plus_caveat"])

L = lambda k: f"l{k}"
checks = [
    dict(id="i1", what="Files of the five sets: 25 seed files, 5 seed averages, 25 networks present; sha256 against results/more_basins.json integrity.sha256_of_saved_files", result="pass",
         numbers=dict(present=r1["i1"]["n_present"], expected=r1["i1"]["n_expected"], equal_to_recorded=r1["i1"]["equal_to_recorded"])),
    dict(id="i2", what="Seeds 0-2 (30 files) equal to the three-seed state (backup results file and round 1's hashes); the 16 control files (04, 05) written before 30 October", result="pass",
         numbers=dict(equal=r1["i2"]["seed_0_2_files_equal_now"], of=r1["i2"]["seed_0_2_files_in_backup"])),
    dict(id="i3", what="Meta of the seed files (script, config as results/lstm_all.json, x5 for lstm63eq_wq, training-river digest) and the seed-3/4 networks' own records (set, seed, training rivers, settings, split, scalers' keys) against seed 0; training and stopping records of the results file against the .pt files",
         result="pass", numbers=dict(best_updates={n: [r1["i3"]["networks"][n][str(s)]["best_update"] for s in SEEDS] for n in NEW})),
    dict(id="i4", what="Every seed average = mean of its seed files summed in seed order (flood.ensemble), bit for bit (5 sets x 5 seeds, 3 controls x 3); obs, basins and dates identical in every file", result="pass",
         numbers={n: v["max_abs_diff"] for n, v in r1["i4"].items()}),
    dict(id="i5", what="passes bookkeeping: sums of trainings, training seconds, run time, waiting; the run stopped by the low-memory guard (12:45-12:47, lstm63_wq seed 3)", result="pass with a finding",
         numbers={k: r1["i5"][k] for k in ("n_trainings_in_passes", "n_trainings_on_disk", "training_seconds_top", "training_seconds_sum_of_networks", "training_seconds_sum_of_passes", "runtime_s_top",
                                            "runtime_s_sum_non_rescore", "trainings_of_the_last_pass_seconds", "killed_run_training")}),
    dict(id="i6", what="integrity.made_in_this_pass and earlier_pass_compared against what the two runs of 30 October made", result="pass with a finding",
         numbers=dict(made_in_this_pass=len(r1["i6"]["made_in_this_pass"]), files_compared=r1["i6"]["earlier_pass_compared"]["files_compared"], files_neither_made_nor_compared=["results/preds/lstm63_wq_s3.npz", "results/models/lstm63_wq_s3.pt"])),
    *[dict(id=i, what=w, result="pass", numbers=bb[b]) for i, w, b in (
        ("s2", "summary_test (12 sets x 3 leads: median and mean NSE, onset and all-days tables)", "summary_test"),
        ("s2b", "validation_used_for_no_choice.summary", "validation_used_for_no_choice"),
        ("s3", "seed_ranges (lo, hi, by_seed of 8 scores) of 12 sets x 3 leads", "seed_ranges"),
        ("s3b", "per_river: nse, nse_by_seed (5 seeds), onset_warned, all_days_warned", "per_river"),
        ("s4", "gain_rule, 4 pairs x 3 leads: NSE rule (own exact Wilcoxon), onset seed rule, paired onsets (own exact McNemar)", "gain_rule"),
        ("s5", "gap_to_the_upper_bound", "gap_to_the_upper_bound"),
        ("s5b", "stopping: updates run, best update, means, sequences per river, ratios (from the network files)", "stopping"),
        ("s6", "verdicts of 12 sets x 3 leads: plain level, levels chosen on the validation years (own sweep 0.30-1.50), single seeds' verdicts", "verdicts"),
        ("s7", "unseen: sets and own-level warnings (05_lobo.py's rule, own code), 6 pairs against the 05 models with single trainings, losses, seed averaging, partners, by_river, by_distance, rank correlations", "unseen"))],
    dict(id="s8", what="The outcome sentence rebuilt word for word from own numbers; outcome_full = outcome + ' ' + outcome_caveat", result="pass", numbers=r3["s8"]),
    dict(id="n1", what="Three networks reloaded from their .pt files with ONE torch thread (trained with 4): lstm63_wq_s3 (the stopped run), lstm63eq_wq_s4, unseen50_w_static_s4; forecasts of the 12 against the stored seed files",
         result="pass", numbers={k: dict(bit_for_bit=v["bit_for_bit"], n_values=v["n_values"]) for k, v in r5.items()}),
    dict(id="t1", what="Readings, flags and validation-chosen levels that changed from three to five seeds (backup file against the file now)", result="pass with a finding",
         numbers=dict(n_changed=len(changed), reading_words_changed=words, levels_changed=len(trig), verdict_words_changed=0)),
    dict(id="t2", what="Rank correlations whose p crossed 0.05 or whose rho moved by more than 0.15", result="pass with a finding", numbers=r4["rank_correlations"]),
    dict(id="t3", what="Seed-count words in outcome_caveat, findings, notes, what_was_not_done, choices", result="pass with a note", numbers=dict(phrases=len(r4["seed_words"]))),
    dict(id="t4", what="SEED_NOTE (choices_made_after_looking_at_test_scores[2]): present, equal to the --note text (note17.txt), its machine and version claims against the seed-3/4 networks and the venv, code/schedule.py",
         result="pass with a finding", numbers=dict(present=r4["t4"]["seed_note_present"], equal_to_note17=r4["t4"]["note17_equal_to_choice"], versions_seeds_3_4=r4["t4"]["versions_seeds_3_4"],
                                                     python=r4["t4"]["python_of_the_venv"], earlier_choices_carried_over=r4["t4"]["three_seed_choices_carried_over"], passes_hold_scores=r4["t4"]["claims"]["passes_hold_scores"])),
    dict(id="t5", what="The note 'Independent review' and the block independent_review of the results file against what this file is (written by --extend with 9 trainings, five seeds)", result="finding"),
    dict(id="t6", what="Files downstream of results/more_basins.json (results/final_report_numbers.json, results/uncertainty.json)", result="finding"),
]

findings = [
    dict(severity="must-fix", where="results/more_basins.json notes[12] ('Independent review. ...') and the block independent_review; source code/17_more_basins.py lines 1091-1097 (the note) and 1163 (the block)",
         text="The note says 'This file was rewritten after the review by --rescore ... no training was made, and 30 saved prediction sets and networks have the sha256 recorded by the earlier pass', and the block "
              "independent_review gives round 1's counts, verdict and date (7 October, three seeds). This file was written by --extend on 30 October 2026 after 9 trainings (10 with the stopped run), and its five-seed numbers were not "
              "covered by round 1, so a reader is told the numbers are reviewed and nothing was trained. Fix: in 17 make the clause depend on the pass and on the record: rv2 = (REVIEW or {}).get('round_2'); after the round-1 sentence add "
              "f\"Seeds 3 and 4 were added on 30 October 2026 by --extend ({sum(p['n_trainings'] for p in passes if '2026-10-30' in p.get('finished_local_time', ''))} trainings in this file's passes, one more in a run stopped before it wrote a results file); \" + "
              "(f\"round 2 of the review ({rv2['dates']['review']}) recomputed the five-seed numbers: {rv2['counts']['values_compared']} values, {rv2['counts']['numerical_mismatches']} mismatches, {rv2['counts']['must_fix']} must-fix and {rv2['counts']['should_fix']} should-fix items.\" "
              "if rv2 else 'the review predates them and covers seeds 0-2 only.'); keep 'no training was made' only for the 7 October rescore it describes (\"the 7 October --rescore made no training\"); and at 1163 write "
              "independent_review=dict(file=..., round_1=dict(counts=REVIEW['counts'], verdict=REVIEW['verdict'], when=REVIEW['when'], covers='seeds 0-2'), round_2=dict(counts=rv2['counts'], verdict=rv2['verdict'], covers=rv2['covers']) if rv2 else None). "
              "Then --rescore (no training)."),
    dict(severity="should-fix", where="results/more_basins.json passes, integrity, notes[9] (Timing) and notes[13]; source code/17_more_basins.py lines 1024-1028 (--prior-pass) and 1129-1130 (note)",
         text="The run of 30 October 12:45-12:47 that the low-memory guard stopped (it saved lstm63_wq seed 3, 117 s; log results/logs/17_more_basins_extend_2026-10-30.log) is in no pass: passes account for 24 of 25 trainings, "
              "training_seconds (8894.2, the sum over the 25 network files) is right, but notes[13] sends the reader to results/logs/17_more_basins.log (the Mac log of 7 October, which does not hold it), notes[9] says passes lists each pass "
              "'the first run ... included', and integrity lists the two lstm63_wq_s3 files neither in made_in_this_pass nor in earlier_pass_compared (30 compared + 23 made = 53 of 55). --prior-pass cannot record it as it is: "
              "line 1026 counts every seed of each named set (5 trainings for sets=['lstm63_wq']) and line 1027 inserts the pass first. Fix: at 1026 use done_ = [tuple(x) for x in pp['pairs']] if 'pairs' in pp else [(n, s) for n in pp['sets'] for s in SEEDS]; "
              "at 1027 insert at pp.get('position', 0) and skip the keys 'pairs' and 'position' in the copied fields; at 1129-1130 name the log of the stopped run instead of the Mac log; add the stopped pass's files to INTEGRITY.made_in_this_pass "
              "(or a field made_by_a_stopped_pass). Then --rescore --prior-pass='{\"what\": \"extension (--extend) of 30 October 2026, stopped by the low-memory guard during its second training (lstm63_wq seed 4) before it wrote a results file\", "
              "\"sets\": [\"lstm63_wq\"], \"pairs\": [[\"lstm63_wq\", 3]], \"position\": 8, \"waited_for_cpu_s\": 0, \"runtime_s\": 126, \"finished_local_time\": \"2026-10-30 12:47\", \"source_of_the_times\": \"results/logs/17_more_basins_extend_2026-10-30.log and PATCHES.md\"}'."),
    dict(severity="should-fix", where="results/more_basins.json outcome_caveat (hence outcome_full) and notes[5], the clause 'the independent review also scored the saved 50-river networks ... (Spearman rho +0.09, p = 0.669 ...; +0.24, p = 0.238 ...)'; source code/17_more_basins.py lines 947-949 (review_clause)",
         text="The clause quotes round 1's 27-river check, which scored the three-seed averages of the 50-river networks (seeds 0-2, 7 October); read now beside five-seed numbers it says the saved networks (now five per variant) were scored. "
              "Not recomputed in this round (it needs the 15 gap rivers' tables; round 1's m10_extra_rivers.py). Fix: in review_clause write 'the independent review (round 1, 7 October) also scored the three-seed averages of the 50-river networks "
              "on the 12 and on ...', or rerun the 27-river check on the five-seed networks and quote that."),
    dict(severity="should-fix", where="results/more_basins.json choices_made_after_looking_at_test_scores[2] (SEED_NOTE) or outcome_caveat; source: the --note text (note17.txt) / code/17_more_basins.py section 6",
         text="Four reading words changed with the two seeds and no text says so: unseen50_w_static against lstm63_w at 1 day, NSE 'mixed' -> 'clear loss' (the never-seen model is now clearly below the same 50 rivers + the 12 in training: "
              "median loss +0.028, lower in 10 of 12 rivers, p = 0.027); lstm63_w against lstm_w at 3 days 'mixed' -> 'no clear difference'; unseen50_w_static against lobo_rf_w at 1 day 'mixed' -> 'no clear difference'; "
              "unseen50_w_nostatic against lobo_rf_w at 2 days, onsets 'clear gain' -> 'no clear difference'. The validation-chosen warning level moved in " + str(len(trig)) + " set/lead/kind cells (no verdict word changed). "
              "And the rank correlation of the never-seen loss (lstm63_w minus unseen50_w_static, 1 day) with the distance to the nearest of the 50 training gauges went from rho +0.57, p = 0.055 to +0.58, p = 0.048 "
              "(unseen.what_goes_with_the_loss; 12 rivers, uncorrected), while the caveat's proximity sentence speaks only of the gain over the 05 models (rho -0.18, p = 0.572 now, -0.36, p = 0.255 with three seeds). "
              "Fix: append to the SEED_NOTE (or compute in 17 from the backup / round_2.readings_changed) one sentence listing these four changed readings, and add the loss-distance correlation beside the gain-distance one in the "
              "proximity sentence of outcome_caveat (f\"the loss from not having seen the river ranks with that distance with properties (rho {w['with_properties_against_63_river_model']['distance_km']['rho']:+.2f}, {pv(...['p'])}; 12 rivers)\")."),
    dict(severity="should-fix", where="results/more_basins.json choices_made_after_looking_at_test_scores[2] (SEED_NOTE), last clause; source: the --note text",
         text="'the three-seed numbers remain in the independent review record (results/review_more_basins.json) and in the earlier entries of passes': passes hold no score (only trainings, seconds and times), so the second half is wrong; "
              "round 1 of the review holds only some three-seed numbers (numbers.headline_lead1 and the like). Fix: '... remain in round 1 of the independent review record (results/review_more_basins.json) and in the copy of the "
              "three-seed results file made before the extension (Project-Flood_backup_2026-10-30/results/more_basins.json; round_2.readings_changed of the review lists what changed)'; then --rescore with the corrected --note "
              "(the carried-over line must be replaced, not added: CHOICES keeps every old line, 17:1029-1030)."),
    dict(severity="should-fix", where="results/final_report_numbers.json (written 2026-10-30 11:59) and results/uncertainty.json (2026-10-28); the documents built from them",
         text="Both still hold the three-seed E3 numbers (final_report_numbers.json quotes 'from 0.22 to 0.38 with basin properties (0.31 to 0.43 without them'); now 0.22 -> 0.39 and 0.31 -> 0.44, 1-day NSE 0.59 -> 0.61, onsets 19% -> 16%. "
              "13_uncertainty.py asserts that results/more_basins.json holds the summary of each seed average (13:295-298), so it must be rerun before the builders. Fix: after the fixes above and the --rescore of 17, rerun "
              "code/13_uncertainty.py, then the report and slide builders; the documents must quote outcome_full and disclose the seed extension."),
    dict(severity="note", where="results/more_basins.json outcome_caveat; code/17_more_basins.py lines 971 and 975",
         text="'(5-seed average; ...' and 'Averaging 5 seeds adds' use digits where every other text uses SEED_WORD ('five seeds per variant', 'five-seed average'). Fix: {SEED_WORD}-seed average / Averaging {SEED_WORD} seeds."),
    dict(severity="note", where="results/preds/lstm63eq_wq_s0..s2.npz meta 'what' against _s3, _s4",
         text="Seeds 0-2 of lstm63eq_wq still carry the old description 'equal training per basin: ...' that round 1 asked to drop; seeds 3-4 and the seed average carry 'as lstm63_wq with 5x the update limit ...'. Information only (the meta is not read for any number)."),
    dict(severity="note", where="gain_rule.lstm63_wq_against_lstm_wq.l1 (the E3 comparison the extra seeds were meant to decide, HANDOFF.md section 4 step 1)",
         text=f"Five seeds did not decide it: NSE 0.590 (three seeds of the stored control, 0.542-0.598) against {f3(same12['l1']['lstm63_wq']['nse_median'])} (five seeds, 0.581-0.603), higher in 10 of 12 rivers, Wilcoxon p = 0.007, still "
              "'no clear difference' because the single seeds overlap; the control keeps three seeds (04 has no save code). The 1-day onset flag 'clear loss' stays; seed 4 of lstm63_wq warns of 73 onsets (11.9%) against 102-105 for seeds 0-3, "
              "so the five-seed average warns of 99 (16.2%) against 106 (17.3%) with three seeds; it also has the highest NSE of the five (0.603)."),
    dict(severity="note", where="results/models/*_s3.pt, *_s4.pt",
         text="The seed-3/4 networks were trained with 4 torch threads (torch 2.2.2+cpu, numpy 1.26.4, Windows; seeds 0-2: torch 2.2.2 on the Mac, also 4 threads); reloaded with one thread three of them reproduce their stored forecasts bit for bit "
              "(447,048 values each). Their training records hold no load average (NaN), as notes[9] says."),
]
counts = dict(checks_listed=len(checks), checks_passed=sum(c["result"] in ("pass", "pass with a note") for c in checks), checks_with_findings=sum(c["result"] in ("pass with a finding", "finding") for c in checks), failed=0,
              must_fix=sum(f["severity"] == "must-fix" for f in findings), should_fix=sum(f["severity"] == "should-fix" for f in findings), notes=sum(f["severity"] == "note" for f in findings),
              numerical_mismatches=n_leaves - n_match, values_compared=n_leaves, network_forecast_values_compared=net_vals, networks_reloaded=len(r5), prediction_files_scored=len(stored), file_hashes_compared=r1["i1"]["n_expected"] + r1["i2"]["seed_0_2_files_in_backup"])

verdict_text = (f"Every five-seed number of results/more_basins.json that I recomputed from results/preds with my own numpy scoring equals the file's ({n_match} of {n_leaves} values at 1e-9, 0 mismatches): summary_test and the validation summary of "
                "12 sets x 3 leads, seed ranges and per-river scores, the gain rule (own exact Wilcoxon and McNemar), the verdicts at the plain and the validation-chosen levels, the never-seen block (own-level warnings, the six pairs, "
                "single trainings, losses, rank correlations), stopping, and the outcome sentence word for word. All 55 files of the five sets are present with the recorded sha256; seeds 0-2 are the three-seed files; "
                "every seed average is the mean of its five seed files bit for bit; three seed-3/4 networks (one of them from the run the memory guard stopped) reproduce their forecasts bit for bit with one thread. "
                f"Same 12 rivers, 1 day: NSE 0.590 -> {f3(same12['l1']['lstm63_wq']['nse_median'])} (12 -> 63 training rivers; no clear difference), onsets warned 18.7% -> {pc1(same12['l1']['lstm63_wq']['onsets_warned'])} (seed-rule flag 'clear loss'); "
                f"2 and 3 days: clear NSE gains, onsets not different. Never-seen rivers, 1 day: 0.216 -> {f3(never['l1']['unseen50_w_static'])} with properties (three seeds: 0.376) and 0.313 -> {f3(never['l1']['unseen50_w_nostatic'])} without (three seeds: 0.434); "
                "the verdict stays NOT YET. Bookkeeping: the passes add up and every training is in training_seconds, but the stopped run of 12:45 is in no pass and the notes point to the wrong log. "
                "One must-fix: the 'Independent review' note and block present the file as reviewed and written without training, which is no longer true. Should-fix: the stopped pass, the round-1 27-river clause now beside five-seed numbers, "
                "four reading words that changed with the two seeds and are not disclosed (one new 'clear loss'), a wrong clause in the SEED_NOTE, and the stale downstream files.")
block = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E3 is still open (team.json untouched by this review)",
    when=f"2026-10-30, about 13:30 to 14:05 local time (UTC+8), after the rerun of 17_more_basins.py --extend (13:02-13:24) on this Windows PC",
    dates=dict(review="2026-10-30", results_file_written=f"{mt('results/more_basins.json')} (file modification time; passes[-1].finished_local_time {R['passes'][-1]['finished_local_time']})",
               seeds_3_4_trained="2026-10-30 12:45-12:47 (lstm63_wq seed 3, the run stopped by the low-memory guard) and 13:00-13:22 (the other nine, log ..._2026-10-30b.log)"),
    covers="the five-seed numbers of results/more_basins.json (seeds 0-4; seeds 3-4 trained on 30 October 2026 on the Windows PC)",
    what="independent recomputation of the five-seed results of script 17 (E3): the five sets lstm63_wq, lstm63_w, lstm63eq_wq, unseen50_w_static, unseen50_w_nostatic (seed averages and seeds 0-4), the controls lstm_wq, lstm_w, lstm_wqf "
         "(three seeds) and the four single trainings of 05_lobo.py; completeness and consistency of every file of seeds 3-4 after the stopped run and the rerun; the passes bookkeeping; what changed from three to five seeds; the texts and the --note disclosure",
    record_script_sha256=files_reviewed["code/17_more_basins.py"],
    files_reviewed=files_reviewed,
    stored_sets_reviewed=stored,
    networks_read=dict(**networks, note="all 25 read for their records (i3); lstm63_wq_s3, lstm63eq_wq_s4 and unseen50_w_static_s4 also reloaded and run (n1)"),
    method=dict(own_code_folder="code/review/more_basins_5seeds/", torch_threads=1, nothing_trained=True, script_not_run="code/17_more_basins.py was not run in any mode",
                project_writes="only that folder and the key round_2 of this record (no bytecode: PYTHONDONTWRITEBYTECODE=1)",
                python="<venv> (Python 3.11.15, torch 2.2.2+cpu, numpy 1.26.4), PYTHONUTF8=1, from the project root",
                memory="prediction files read one at a time (r2 holds only the scores); networks reloaded one at a time",
                engine_used="only in r5_networks.py (flood.LSTMNet, flood.Run.predict, flood.load_data on the stored 63-basin cache); every score, threshold, onset, test and level sweep is the reviewer's own numpy code; scipy only for the Spearman p"),
    checks=checks,
    reproduced=dict(how="own numpy scoring (r2_score.py): Q95 per basin = 95th percentile (linear) of the stored observed flow over 1980-10-01..2000-09-30; onset = a test-year day above Q95 whose flow lead days earlier (by array position) was at "
                        "or below Q95; warning = forecast > trigger x Q95; NSE per basin on the days with a forecast; tables pooled over the 12; own level = 95th percentile of the set's own training-years forecasts; validation-chosen level by own sweep "
                        "0.30..1.50 step 0.05, best pooled validation CSI, ties to the level nearest 1.0; exact Wilcoxon by enumeration, exact McNemar by the binomial sum; every leaf compared with results/more_basins.json at relative 1e-9 (r3_compare.py)",
                    summary=dict(entries=n_leaves, matching=n_match, not_matching=r3["not_matching"]), by_block=bb,
                    same_12_rivers=same12, never_seen_rivers=never, readings=readings, outcome_full_numbers=outcome_full_numbers, three_seed_numbers_for_comparison=three),
    readings_changed=dict(how="every leaf named reading / clear_gain / clear_loss / single_seeds_all_above / single_seeds_all_below / p_below_cut / higher_in_most_basins / verdict / trigger compared between the three-seed file "
                              "(Project-Flood_backup_2026-10-30/results/more_basins.json) and the file now (r4_three_vs_five.py); single seeds' verdicts left out (seeds 3-4 are new)",
                          changes=changed, rank_correlations=r4["rank_correlations"], outcome_three_seeds=r4["outcome_three"], outcome_five_seeds=r4["outcome_five"]),
    bookkeeping=dict(passes=r1["i5"], integrity=r1["i6"]),
    findings=findings,
    counts=counts,
    verdict="numbers hold with wording fixes",
    verdict_text=verdict_text,
    scripts=["code/review/more_basins_5seeds/README.md", "code/review/more_basins_5seeds/r_common.py", "code/review/more_basins_5seeds/r1_integrity.py", "code/review/more_basins_5seeds/r2_score.py",
             "code/review/more_basins_5seeds/r3_compare.py", "code/review/more_basins_5seeds/r4_three_vs_five.py", "code/review/more_basins_5seeds/r5_networks.py", "code/review/more_basins_5seeds/r6_assemble.py",
             "code/review/more_basins_5seeds/run_checks.sh"],
    not_checked=["No training; whether seeds 3-4 trained again (another machine, other thread count) give the same networks is untested; the files are what was checked.",
                 "The figure (figures/more_basins.png, rewritten 13:24), leak_probes, the properties probe, proximity and the outline check were not recomputed in this round (proximity distances were taken from unseen.by_river; round 1 reproduced them and they do not depend on seeds).",
                 "The 27-river check of round 1 (15 never-seen rivers with gaps) was not repeated on the five-seed networks.",
                 "Only 3 of the 25 networks were reloaded and run; the script's own reload_check reports all 25 bit for bit."],
)
block = clean(block)
save("round_2", block)
print(json.dumps(counts))
print("verdict:", block["verdict"])

if "--write-into-project" in sys.argv:
    orig = REC_FILE.read_bytes()
    rec = json.loads(orig.decode("utf-8"))
    assert "round_2" not in rec, "round_2 is already in the record"
    assert json.dumps(rec, indent=1, ensure_ascii=True).encode("utf-8") == orig, "the record's layout is not indent 1 / ASCII / no final newline: not written"
    rec["round_2"] = block
    new = json.dumps(rec, indent=1, ensure_ascii=True).encode("utf-8")
    back = json.loads(new.decode("utf-8"))
    del back["round_2"]
    assert json.dumps(back, indent=1, ensure_ascii=True).encode("utf-8") == orig
    tmp = REC_FILE.with_suffix(".json.tmp")
    tmp.write_bytes(new)
    tmp.replace(REC_FILE)
    chk = json.loads(REC_FILE.read_bytes().decode("utf-8"))
    del chk["round_2"]
    assert json.dumps(chk, indent=1, ensure_ascii=True).encode("utf-8") == orig
    print("round_2 appended; removing it gives back the original", len(orig), "bytes; sha256 before", hashlib.sha256(orig).hexdigest(), "after", sha256(REC_FILE))
