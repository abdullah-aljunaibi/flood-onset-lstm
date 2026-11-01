"""E7 review, check 6: builds the review record from the outputs of m1-m5 and m7 (OUT/*.json). Without arguments it writes OUT/review_multi_product.json only;
with --write-into-project it first checks that no reviewed file changed since m1 hashed it (the results file, the figure, the script, every saved set and network) and then
writes results/review_multi_product.json. The numbers in the findings and the verdict are formatted from the reviewer's outputs (m1-m5, m7), not typed."""
import sys

from m_common import *

WRITE = "--write-into-project" in sys.argv
R = jload(RESULTS)
m1, m2, m3, m4, m5, m7 = (jread(f) for f in ("m1_integrity.json", "m2_headline.json", "m3_columns.json", "m4_networks.json", "m5_outcome.json", "m7_ridge.json"))
M = m2["my"]
H, RG, X, OPS, RS, PAIRS = M["headline"], M["ranges"], M["extras"], M["ops"], M["resampling"], M["pairs"]
pc1 = lambda x: f"{100 * x:.1f}%"
pts = lambda x: f"{x:+.1f}"
on = lambda n, c="pod", k=1: H[n][lk(k)]["onset"][c]
med = lambda n, k=1: H[n][lk(k)]["nse_median"]
op = lambda n, r="far_cap": OPS[n]["l1"][r]
iv = lambda key, s=100, d=1: f"{s * RS[key]['both']['lo']:+.{d}f} to {s * RS[key]['both']['hi']:+.{d}f}"
LM, LEV, WX = X["level_matched_l1"], X["forecast_level_l1"], X["wilcoxon"]
pa = PAIRS["several_products_against_aorc_alone"]
b1 = M["rule"]["block_1"]
n_on = H["wqME"]["l1"]["onset"]["hits"] + H["wqME"]["l1"]["onset"]["misses"]
VAL = R["validation_used_for_no_choice"]["summary"]                      # equal to the reviewer's validation scores (m2, every value compared)
BV = [v["best_val_nse"] for k, v in m1["networks"]["per_file"].items() if k.startswith("lstm_wqME_s")]
HB = jload("results/review_aorc_rain_to_midnight.json")["round_2"]["hindsight"]["five_seed_loosest_bound_points"]
nd = m3["nldas_day"]
nl_hours = sorted({v["best_end_local_standard_hour"] for v in nd.values()})
sh = m3["shuffle"]

# ---- the headline table: file against mine ------------------------------------------------------------------------------------------------------------------
S = R["summary_test"]["sets"]
head = {}
for n in ("wqME", "wqME_shuf", "wqME_older", "wqa", "wqnh", "wq"):
    row = {}
    for k in LEADS:
        row[lk(k)] = {key: dict(file=f_, mine=m_, diff=abs(f_ - m_)) for key, f_, m_ in (("nse_median", S[n][lk(k)]["nse_median"], med(n, k)),
                                                                                        ("onset_hits", S[n][lk(k)]["onset"]["hits"], H[n][lk(k)]["onset"]["hits"]),
                                                                                        ("onset_pod", S[n][lk(k)]["onset"]["pod"], on(n, "pod", k)),
                                                                                        ("onset_far", S[n][lk(k)]["onset"]["far"], on(n, "far", k)))}
    fs = R["seed_ranges"]["sets"][n]["l1"]["by_seed"]
    row["single_seeds_l1"] = dict(seeds=fs["seeds"], onset_pod_file=fs["onset_pod"], onset_pod_mine=RG[n]["l1"]["by_seed"]["onset_pod"], onset_far_file=fs["onset_far"],
                                  onset_far_mine=RG[n]["l1"]["by_seed"]["onset_far"], nse_median_file=fs["nse_median"], nse_median_mine=RG[n]["l1"]["by_seed"]["nse_median"],
                                  all_equal=bool(np.allclose(fs["onset_pod"], RG[n]["l1"]["by_seed"]["onset_pod"], rtol=0, atol=1e-12) and np.allclose(fs["nse_median"], RG[n]["l1"]["by_seed"]["nse_median"], rtol=0, atol=1e-12)))
    head[n] = row
W = R["e7_rule"]["block_1_gain"]["wilcoxon_over_the_basins"]
rule_check = dict(
    block_1=dict(file=dict(holds=R["e7_rule"]["block_1_gain"]["holds"], every_seed_above=R["e7_rule"]["block_1_gain"]["every_seed_above"]["holds"],
                           false_share_not_higher=R["e7_rule"]["block_1_gain"]["false_share_not_higher"]["holds"], wilcoxon_p=W["p_value"], n_used=W["n_used"]),
                 mine=dict(holds=b1["holds"], every_seed_above=b1["every_seed_above"], false_share_not_higher=b1["false_wqME"] <= b1["false_wqa"], wilcoxon_p=b1["wilcoxon_p"],
                           wilcoxon_p_exact_by_enumeration=b1["wilcoxon_p_exact"], n_used=b1["n_used"], wqME_lowest_seed=min(RG["wqME"]["l1"]["by_seed"]["onset_pod"]),
                           wqa_highest_seed=max(RG["wqa"]["l1"]["by_seed"]["onset_pod"]), false_wqME=b1["false_wqME"], false_wqa=b1["false_wqa"])),
    block_2=dict(file=dict(holds=R["e7_rule"]["block_2_capacity_control_flat"]["holds"], median=R["e7_rule"]["block_2_capacity_control_flat"]["median_of_the_single_seeds"]),
                 mine=dict(holds=M["rule"]["blocks_2_3"]["block_2_capacity_control_flat"]["inside"], median=M["rule"]["blocks_2_3"]["block_2_capacity_control_flat"]["median"], wqa_range=X["wqa_range_l1"])),
    block_3=dict(file=dict(holds=R["e7_rule"]["block_3_clock_control_flat"]["holds"], median=R["e7_rule"]["block_3_clock_control_flat"]["median_of_the_single_seeds"]),
                 mine=dict(holds=M["rule"]["blocks_2_3"]["block_3_clock_control_flat"]["inside"], median=M["rule"]["blocks_2_3"]["block_3_clock_control_flat"]["median"])),
    block_4=dict(file=R["e7_rule"]["block_4_network_yardstick"]["gain_points_at_lead_1"], mine=M["rule"]["yardstick_gains"], reading_file=R["e7_rule"]["block_4_network_yardstick"]["reading"],
                 reading_mine=M["rule"]["yardstick_reading"]),
    overall=dict(file=dict(met=R["e7_rule"]["met"], reading=R["e7_rule"]["reading"]), mine=dict(met=M["rule"]["met"])),
    applied_without_change=True,
    interpretations_judged=[
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][0][:90] + " ...", judgement="fair: 'every single seed above every single seed' is min(test) > max(base), the 'above' of scripts 15-22; the strict reading is not decisive here (lowest wqME seed "
             + pc1(min(RG["wqME"]["l1"]["by_seed"]["onset_pod"])) + " is below the highest wqa seed " + pc1(max(RG["wqa"]["l1"]["by_seed"]["onset_pod"])) + " under any reading)"),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][1][:90] + " ...", judgement="fair (at most = not higher); not decisive here: " + pc1(b1["false_wqME"]) + " against " + pc1(b1["false_wqa"])),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][2][:90] + " ...", judgement=("fair. 'Per-basin onsets warned' read as each basin's share (POD) or as its count of hits gives the same signs: "
             f"{WX['n_used']} non-zero differences, all positive, {WX['n_zero']} zero; exact two-sided p = 2/2^{WX['n_used']} = {WX['p_exact_enumeration']:.5f} by my enumeration = scipy's default; the normal "
             f"approximation gives {WX['p_approx']:.4f}. Dropping the zeros (zero_method 'wilcox') is scipy's default and is recorded (n_used); with every non-zero difference positive the statistic is the most extreme one possible for that n.")),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][3][:90] + " ...", judgement="fair: inclusive 'within', numpy median of five values, range over all five stored wqa seeds (ROADMAP says five). See note on what blocks 2-3 can show."),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][4][:90] + " ...", judgement="fair as a reading of 'as much' at trigger 1.0 (not part of the verdict); see must-fix 2 for what the trigger-1.0 gains measure."),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][5][:90] + " ...", judgement="matches ROADMAP word for word (basin x calendar month x split part, one permutation shared by the two columns, fixed seed, same data for every seed); rebuilt here bit for bit (source-index sha256)."),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][6][:90] + " ...", judgement="matches ROADMAP ('one day older'); the first day keeping its own value touches only 1980-10-01, a warm-up day never scored."),
        dict(text=R["e7_rule"]["interpretations_of_the_rule"][7][:90] + " ...", judgement="matches 13_uncertainty.py's recipe (same seed, order of draws, designs, percentile interval); reproduced exactly here. The Monte-Carlo noise it leaves out is small: "
             f"{X['mc_noise_of_the_interval_ends_points'][0]:.2f} and {X['mc_noise_of_the_interval_ends_points'][1]:.2f} points for the two ends of the headline interval (16 parts, my computation).")])

# ---- findings ----------------------------------------------------------------------------------------------------------------------------------------------------
F_ = []
F_.append(dict(severity="must-fix", where="results/multi_product.json outcome (first clause 'Given the NLDAS rain of the issue day, the ERA5 rain to midnight and the AORC rain to midnight (an analysis made after the fact) ...') "
                                            "and outcome_caveat ('AORC is an analysis made after the fact: wqME carries the hindsight bound of wqa (hindsight at most about 4 points), measured for wqa and not measured again for wqME'); "
                                            "the figure footer; what_was_not_done (code/28_multi_product.py: X, outcome_caveat, draw(), NOT_DONE); ROADMAP.md E7 ('wqME carries the AORC hindsight bound of wqa ... unless measured again')",
               text=("Only AORC is labelled as made after the fact, so the sentence reads as if the NLDAS and ERA5 columns were inputs a forecaster has at midnight. They are not: the project's own records say 'NLDAS itself is "
                     "published a few days late, and it is made after the fact: its hourly rain is a daily rain-gauge analysis spread over the hours' (results/rain_timing.json notes[3]) and that the ERA5 rain to midnight "
                     "shows 'what rain up to midnight is worth if it were known as well as ERA5 knows it, not what a real-time service delivers' (results/rain_to_midnight.json notes[4]; ERA5 also assimilates US radar and "
                     "gauge rain from 17 July 2009, inside the test years). The whole wqME-minus-wqa difference (" + pts(100 * (on('wqME') - on('wqa'))) + " points) comes from adding exactly these two columns, while the "
                     f"'about {round(HB)} points' bound was measured for the AORC column of wqa only (Stage IV in place of AORC; results/review_aorc_rain_to_midnight.json round_2, five-seed loosest bound {HB:.1f} points). 'wqME carries the "
                     "hindsight bound of wqa' is therefore not a bound on wqME: it understates the hindsight of the set. results/multi_product.json contains none of 'real-time', 'reanalysis', 'published', 'days late', "
                     "'service' (searched). Fix: label the three products in the outcome ('the NLDAS rain of the issue day (published days late), the ERA5 rain to midnight (a reanalysis) and the AORC rain to midnight "
                     "(an analysis made after the fact): none of them is available at the issue time'); replace the hindsight clause by '(hindsight at most about 4 points for the AORC column of wqa; the hindsight of "
                     "wqME, whose NLDAS and ERA5 columns are made after the fact as well, was not measured)'; the same in the figure footer; add to what_was_not_done 'real-time stand-ins for the NLDAS and ERA5 "
                     "columns (Stage IV or IMERG-Early hours) in the saved wqME networks'; and correct ROADMAP.md's sentence the same way.")))
F_.append(dict(severity="must-fix", where="results/multi_product.json outcome ('onsets warned 1 day ahead go from 52% ... to 56% ... +4.6 points') and outcome_caveat (the per-basin clause, '36 onsets are warned only by wqME and 8 only by wqa', "
                                            "the resampling interval, the 30%-false clause, the network-yardstick reading) (code/28_multi_product.py: outcome, outcome_caveat)",
               text=(f"The trigger-1.0 difference is a higher warning level, not better discrimination, and the sentence does not say so. At trigger 1.0 wqME gives {LEV['wqME']['onset_warnings_at_1']} onset warnings "
                     f"against wqa's {LEV['wqa']['onset_warnings_at_1']} ({on('wqME', 'hits') - on('wqa', 'hits')} more hits, {on('wqME', 'false_alarms') - on('wqa', 'false_alarms')} more false); its forecasts sit higher on the onset days "
                     f"(median forecast / Q95 {LEV['wqME']['median_ratio_on_onset_days']:.2f} against {LEV['wqa']['median_ratio_on_onset_days']:.2f}; pooled test-year bias {LEV['wqME']['pbias_pooled_pct']:+.1f}% against "
                     f"{LEV['wqa']['pbias_pooled_pct']:+.1f}%). At the triggers chosen on the validation years the difference is gone: CSI rule {pc1(op('wqME', 'csi')['test_pod'])} with {pc1(op('wqME', 'csi')['test_far'])} false "
                     f"(trigger {op('wqME', 'csi')['trigger']:.2f}) against {pc1(op('wqa', 'csi')['test_pod'])} with {pc1(op('wqa', 'csi')['test_far'])} false ({op('wqa', 'csi')['trigger']:.2f}); 30%-false rule "
                     f"{pc1(op('wqME')['test_pod'])} against {pc1(op('wqa')['test_pod'])}, with the capacity control at {pc1(op('wqME_shuf')['test_pod'])} and the clock control at {pc1(op('wqME_older')['test_pod'])} "
                     f"(false shares {pc1(op('wqME')['test_far'])}, {pc1(op('wqa')['test_far'])}, {pc1(op('wqME_shuf')['test_far'])}, {pc1(op('wqME_older')['test_far'])}): at the operating point the GO rule cares about "
                     f"the treatment is the lowest of the four. Descriptively (a level picked on the test years, used for no choice): with as many onset warnings as wqa gives at 1.0 ({LM['wqa_onset_warnings_at_1']}, wqME "
                     f"trigger {LM['wqME_trigger_for_the_same_number']:.3f}) wqME warns {LM['wqME_hits']} onsets against wqa's {LM['wqa_hits']}. The per-basin Wilcoxon (p = {b1['wilcoxon_p']:.3f}), the paired onsets "
                     f"({pa['l1']['only_first']} against {pa['l1']['only_second']}) and the resampling interval ({iv('onsets_diff.wqME_minus_wqa.l1')} points) all measure this same shift at the fixed level, yet the sentence "
                     f"offers them as support for the {pts(100 * (on('wqME') - on('wqa')))} points, and the one matched comparison it gives (the 30%-false point) leaves out the controls and the CSI point. The rule's 'not a clear "
                     f"gain' is right (block 1 fails on the seed rule and on the false share). The same qualifier applies to block 4's reading ('the LSTM gains more from the wqME columns than ridge and forest'): it holds at "
                     f"trigger 1.0 only (at the 30%-false point the LSTM moves {pts(100 * (op('wqME')['test_pod'] - op('wqa')['test_pod']))} points, ridge {pts(100 * (op('base_linreg_wqME')['test_pod'] - op('base_linreg_wqa')['test_pod']))} "
                     f"(with {pc1(op('base_linreg_wqME')['test_far'])} and {pc1(op('base_linreg_wqa')['test_far'])} false on the test years), forest {pts(100 * (op('base_rf_wqME')['test_pod'] - op('base_rf_wqa')['test_pod']))}). Fix: after the false-share clause add 'the extra warnings come from a higher warning level ("
                     f"{LEV['wqME']['onset_warnings_at_1']} onset warnings against {LEV['wqa']['onset_warnings_at_1']}): at the levels chosen on the validation years wqME warns of {round(100 * op('wqME', 'csi')['test_pod'])}% "
                     f"(CSI rule) and {round(100 * op('wqME')['test_pod'])}% (30%-false rule) against {round(100 * op('wqa', 'csi')['test_pod'])}% and {round(100 * op('wqa')['test_pod'])}% for wqa, so the products side by side do not separate "
                     f"onsets better'; name the controls in the 30%-false clause; say that the per-basin, paired and resampled numbers are at trigger 1.0; and qualify the yardstick reading 'at trigger 1.0'.")))
F_.append(dict(severity="should-fix", where="results/multi_product.json outcome ('the median NSE goes from 0.76 to 0.73') and outcome_caveat (leads 2-3)",
               text=(f"The NSE clause gives the two medians only. wqME's NSE is lower than wqa's in {X['nse_l1_basins']['wqME_lower_than_wqa']} of 12 basins at 1 day (Wilcoxon p = {X['nse_l1_basins']['p']:.3f}; "
                     f"{PAIRS['several_products_against_aorc_alone']['l2']['nse_lower']} and {PAIRS['several_products_against_aorc_alone']['l3']['nse_lower']} of 12 at 2 and 3 days, p = "
                     f"{PAIRS['several_products_against_aorc_alone']['l2']['nse_p']:.3f} and {PAIRS['several_products_against_aorc_alone']['l3']['nse_p']:.3f}), lower than both controls ({PAIRS['several_products_against_the_capacity_control']['l1']['nse_lower']} "
                     f"and {PAIRS['several_products_against_the_clock_control']['l1']['nse_lower']} of 12 basins), and the basins-only resampling interval of the difference of medians lies below zero "
                     f"({RS['nse_diff_of_medians.wqME_minus_wqa.l1']['basins']['lo']:+.3f} to {RS['nse_diff_of_medians.wqME_minus_wqa.l1']['basins']['hi']:+.3f}). By the NSE rule of section 4.2 it is 'no clear "
                     f"difference' only because the single seeds overlap (wqME {RG['wqME']['l1']['nse_median'][0]:.3f}-{RG['wqME']['l1']['nse_median'][1]:.3f}, wqa {RG['wqa']['l1']['nse_median'][0]:.3f}-{RG['wqa']['l1']['nse_median'][1]:.3f}). "
                     f"Fix: '... the median NSE goes from {med('wqa'):.2f} to {med('wqME'):.2f}, lower in {X['nse_l1_basins']['wqME_lower_than_wqa']} of 12 basins (p = {X['nse_l1_basins']['p']:.3f}; not a clear loss by the seed rule)'.")))
F_.append(dict(severity="should-fix", where="results/multi_product.json outcome_caveat ('With the test water years and the basins drawn again ... lies between +1.3 to +7.6 points (above zero in 99% of the resamples)')",
               text=(f"The interval resamples the test water years and the basins of the seed-averaged forecasts; the training randomness is not in it, and the clause does not say so (the file's how_to_read does). "
                     f"The single trainings overlap (wqME {pc1(RG['wqME']['l1']['onset_pod'][0])}-{pc1(RG['wqME']['l1']['onset_pod'][1])}, wqa {pc1(RG['wqa']['l1']['onset_pod'][0])}-{pc1(RG['wqa']['l1']['onset_pod'][1])}); "
                     f"of the 25 pairs of single trainings {sum(x <= 0 for x in X['all_pairs_wqME_minus_wqa_points'])} go the other way (lowest {min(X['all_pairs_wqME_minus_wqa_points']):+.1f} points), and the seed-by-seed list runs "
                     f"from {min(X['seed_by_seed_wqME_minus_wqa_points']):+.1f} to {max(X['seed_by_seed_wqME_minus_wqa_points']):+.1f}. Fix: 'between {100 * RS['onsets_diff.wqME_minus_wqa.l1']['both']['lo']:+.1f} and "
                     f"{100 * RS['onsets_diff.wqME_minus_wqa.l1']['both']['hi']:+.1f} points when the test water years and the basins are drawn again (the training seeds are not: {sum(x <= 0 for x in X['all_pairs_wqME_minus_wqa_points'])} "
                     "of the 25 pairs of single trainings go the other way)'; 'between ... to' should read 'between ... and' (also in the NSE clause).")))
F_.append(dict(severity="should-fix", where="figures/multi_product.png (code/28_multi_product.py draw(): the header block and the subtitle)",
               text=("The status text runs into the top panel: its last line ('are not yet done.') is drawn over the panel title 'Onsets warned 1 day ahead (test years 2005-2014, trigger 1.0, 12 basins pooled)' (read on "
                     "the image). The seeds line names 'wqa 5, wq 3, wqn 3' and leaves out wqh and wqnh (five seeds each; their five dots are drawn). The footer carries the hindsight sentence of must-fix 1. "
                     "Fix: more top margin or a shorter status line; 'seeds of the stored sets: wqa, wqh, wqnh 5; wq, wqn 3'; the footer as in must-fix 1. Every bar label (onsets warned, false share, median NSE of the 12 bars) "
                     "equals the reviewer's numbers.")))
F_.append(dict(severity="should-fix", where="results/multi_product.json status; figures/multi_product.png header",
               text=("Both say 'NOT YET REVIEWED', true when they were written. After the wording fixes run `python code/28_multi_product.py --rescore` (nothing is trained; it refuses if a saved file changed) so that the status, "
                     "the figure and the fixed clauses are rebuilt together and the status picks up this record (its dates.results_file_written and stored_sets_reviewed are written for that check); then rerun this "
                     "review's m5 and m6 on the rebuilt file (m1-m4 and m7 need no rerun: a rescore touches no prediction set or network).")))
F_.append(dict(severity="note", where="results/multi_product.json e7_rule.block_2_capacity_control_flat, block_3_clock_control_flat; outcome_caveat (block text)",
               text=(f"Blocks 2 and 3 are applied as written and hold, but as written they cannot separate the treatment from the controls: wqME's own median single seed ({pc1(X['median_single_seed_l1']['wqME'])}) also lies "
                     f"inside wqa's range ({pc1(X['wqa_range_l1'][0])}-{pc1(X['wqa_range_l1'][1])}), as do the controls' ({pc1(X['median_single_seed_l1']['wqME_shuf'])}, {pc1(X['median_single_seed_l1']['wqME_older'])}). "
                     f"The direct comparisons the file holds say more: at trigger 1.0 wqME minus the capacity control {pts(PAIRS['several_products_against_the_capacity_control']['l1']['diff_points'])} points (interval "
                     f"{iv('onsets_diff.wqME_minus_wqME_shuf.l1')}; {PAIRS['several_products_against_the_capacity_control']['l1']['only_first']} against {PAIRS['several_products_against_the_capacity_control']['l1']['only_second']} paired onsets), "
                     f"minus the clock control {pts(PAIRS['several_products_against_the_clock_control']['l1']['diff_points'])} ({iv('onsets_diff.wqME_minus_wqME_older.l1')}); at the 30%-false point both controls are above wqME "
                     f"(must-fix 2). Suggested addition to the block text: 'wqME's own median single seed ({pc1(X['median_single_seed_l1']['wqME'])}) is inside that range too, so blocks 2 and 3 do not separate the treatment from the controls'. "
                     "Not a change of the rule, which was fixed before training.")))
F_.append(dict(severity="note", where="code/28_multi_product.py build_tables, shuffle_columns, the .pt files; results/multi_product.json inputs, causality",
               text=(f"Leak probes pass. (a) The seven input columns rebuilt by my own route (zoneinfo instants of local midnight, hour-ending stamps in (start, end], cumulative sums; pandas groupby on the same day assignment) "
                     f"have the sha256 the results file records, bit for bit (route A and B differ by at most {m3['route_a_vs_route_b_max_abs_diff_mm']:.1e} mm); every hour summed into day t ends after 00:00 and at or before "
                     f"24:00 local of day t (all 12 basins, both hourly products, 34 + 34 clock-change days); made-up hours after the issue time of a cut day move no earlier day ({m3['n_made_up_cases']} cases: 12 basins x 2 "
                     f"products x random test days and the days around the last clock changes). (b) The NLDAS day: the 24-hour window of AORC hours that best matches the CAMELS NLDAS daily total (2002-2014) ends at {' or '.join(f'{h}:00' for h in nl_hours)} local standard "
                     f"time (17:00 only in Leaf River, Central time) (r {min(v['r_best'] for v in nd.values()):.2f}-{max(v['r_best'] for v in nd.values()):.2f}, against {min(v['r_window_ending_at_local_midnight'] for v in nd.values()):.2f}-"
                     f"{max(v['r_window_ending_at_local_midnight'] for v in nd.values()):.2f} for the window ending at the issue time): before the issue time in every basin, as results/day_windows.json says (17:40). "
                     f"(c) ERA5's hourly rain matches AORC best one hour LATER (best lag +1 h in 3 of 4 basins checked): no early hours. (d) The _older columns are the columns moved one day (sha256 equal). (e) All 15 "
                     f"networks rebuilt from their .pt files reproduce their stored forecasts bit for bit on my tables; so do all ten stored networks of lstm_wqa and lstm_wqnh (the script checked seeds 0 and 3); the scalers "
                     f"in every file are the training-year statistics; my made-up-values test (3 networks of other seeds than the script's, 4 cut days among them the last validation day and a clock change, every input "
                     f"column and the flow) changes no forecast issued on or before the cut. (f) The ridge yardstick refitted from my columns equals base_linreg_wqME " + ("bit for bit." if m7['max_abs_diff'] == 0 else f"to {m7['max_abs_diff']:.1e} mm/day."))))
F_.append(dict(severity="note", where="code/28_multi_product.py shuffle_columns; results/multi_product.json inputs.shuffle",
               text=(f"The shuffle stays inside basin x calendar month x split part (rebuilt from default_rng(20261030): source-index sha256 and the counts equal the file's; {sh['counts']['from_another_split_part']} values "
                     f"cross a split part) and changes nothing else (the AORC column, the Daymet columns, the flow and the basin properties of wqME_shuf are those of wqME; the two shuffled columns move together). "
                     f"Inside a split part it moves values in time: {sh['test_days_reading_a_value_measured_1_to_31_days_later']:,} of {sh['test_days']:,} test basin-days read a value measured 1 to 31 days LATER in the same "
                     "month of the same year. That is future rain inside the control's own inputs; it can only help the control, so the control stays a conservative yardstick, and the script says the columns are no "
                     "forecast input. Fine as designed.")))
F_.append(dict(severity="note", where="results/multi_product.json fixed_before_training, passes; results/quick/multi_product.json",
               text=(f"Pre-registration holds: the script's sha256 now equals the hash recorded at the start of the one training pass; the script ({m1['script']['modified']}) and ROADMAP.md ({m1['script']['roadmap_modified']}) "
                     f"were last modified before the first network ({m1['script']['first_network_written']}); the ROADMAP E7 section's hash equals the recorded one; one pass, 15 trainings, seeds 0-4, no extension, "
                     f"choices_made_after_looking_at_test_scores empty. A smoke test of the same script (results/quick, {(m1.get('smoke_test') or {}).get('written')}, one seed, 40 updates) scored the test years before the "
                     "full run; the 'same split part' change came after an earlier smoke test that is not stored. Both are disclosed (ROADMAP, fixed_before_training) and neither is a choice made on the test scores "
                     f"of trained networks. The validation years do not foreshadow the test-year difference at trigger 1.0 (onsets warned {pc1(VAL['wqME']['l1']['onset']['pod'])} for wqME against "
                     f"{pc1(VAL['wqa']['l1']['onset']['pod'])} for wqa; the validation summary was recomputed in m2; best validation NSE of the wqME networks {min(BV):.3f}-{max(BV):.3f}).")))
F_.append(dict(severity="note", where="HANDOFF.md section 7 against results/multi_product.json and ROADMAP.md E7",
               text=(f"HANDOFF section 7 fixes 'hindsight at most about 3 points' word for word; the script prints 'about {round(HB)} points', read from results/review_aorc_rain_to_midnight.json round_2 (five-seed loosest bound {HB:.1f} points), "
                     "as ROADMAP's E7 section says. The 4 is the right number for the five-seed lstm_wqa; HANDOFF section 7 is out of date (the lead's file, not changed here).")))
F_.append(dict(severity="note", where="results/multi_product.json (all recomputed blocks)",
               text=(f"Every number checked holds: {m2['compare']['entries']:,} values of the results file recomputed from results/preds with my own numpy scoring match to 1e-9 (summary_test and the validation summary, per basin, "
                     "every single seed and the seed ranges, the rule blocks with the Wilcoxon, the seven pair readings and paired onsets, the operating points with single seeds and verdicts, the verdict readings at the "
                     "level grid, all 168 resampling statistics in the three designs, the stored sets against their own results files); flood.evaluate as a second route agrees; the seed-mean files are the means of "
                     "the seed files; the outcome's 22 numeric clauses rebuilt from my numbers are in the text word for word.")))

n_must = sum(f["severity"] == "must-fix" for f in F_)
n_should = sum(f["severity"] == "should-fix" for f in F_)
n_note = sum(f["severity"] == "note" for f in F_)
values_compared = sum(x["compare"]["entries"] for x in (m1, m2, m3, m4, m5, m7))
mismatches = sum(len(x["compare"]["not_matching"]) for x in (m1, m2, m3, m4, m5, m7))
counts = dict(values_compared=values_compared, numerical_mismatches=mismatches, networks_rebuilt=len(m4["per_file"]) + len(m4["identity"]),
              made_up_value_cases_on_networks=3 * 4, made_up_hour_cases_on_data=m3["n_made_up_cases"], outcome_clauses_rebuilt=m5["n_clauses"], outcome_clauses_found=m5["n_found"],
              figure_bars_checked=len(m5["figure"]["labels_read_on_the_image"]), must_fix=n_must, should_fix=n_should, notes=n_note)
verdict = "numbers hold with wording fixes"
verdict_text = (f"Every number I recomputed holds ({values_compared:,} values, {mismatches} mismatches): onsets warned 1 day ahead {pc1(on('wqME'))} for wqME against {pc1(on('wqa'))} for the stored wqa "
                f"({on('wqME', 'hits')} against {on('wqa', 'hits')} hits of {n_on}), false share {pc1(on('wqME', 'far'))} against {pc1(on('wqa', 'far'))}, median NSE {med('wqME'):.3f} against {med('wqa'):.3f}; the "
                f"capacity control {pc1(on('wqME_shuf'))}, the clock control {pc1(on('wqME_older'))}; every single seed, the per-basin Wilcoxon (p = {b1['wilcoxon_p']:.4f}, n = {b1['n_used']}), the paired onsets, the "
                "operating points and the 168 resampling statistics equal the file's. The ROADMAP rule applied block by block gives the file's answer: block 1 fails (seed rule and false share), blocks 2 and 3 hold, "
                "not a clear gain; the script's readings of the rule are fair. No leak found: the input columns are the script's bit for bit, every hour lies before the issue time, the NLDAS day ends about six hours "
                "before it, the shuffle stays inside basin x month x split part and changes nothing else, the older columns are one day older, all 25 networks (15 new, 10 stored) reproduce their forecasts bit for bit "
                "on my tables, and made-up hours and values after a cut change nothing earlier. Two must-fix items of wording: (1) NLDAS and ERA5 are not real-time products either, so the hindsight caveat that names "
                f"only AORC understates what wqME knows; (2) the {100 * (on('wqME') - on('wqa')):+.1f} points at trigger 1.0 is a higher warning level ({LEV['wqME']['onset_warnings_at_1']} onset warnings against "
                f"{LEV['wqa']['onset_warnings_at_1']}), not better discrimination: at the triggers chosen on the validation years wqME warns no more onsets than wqa ({pc1(op('wqME', 'csi')['test_pod'])} against "
                f"{pc1(op('wqa', 'csi')['test_pod'])} by the CSI rule; {pc1(op('wqME')['test_pod'])} against {pc1(op('wqa')['test_pod'])} at the 30%-false point, where both controls are above wqME), and the sentence's "
                f"per-basin, paired and resampled numbers all measure the trigger-1.0 shift. Should-fix: the NSE is lower in {X['nse_l1_basins']['wqME_lower_than_wqa']} of 12 basins, the interval leaves out the seeds, "
                "the figure's header overlaps, and a --rescore after the fixes. The numbers stay preliminary until the fixes are made and the file is rescored; the team's own checks are not done.")

saved = m1["files_reviewed_sha256"]
rec = dict(
    who="independent AI reviewer agent (Claude), 2026-10-30",
    human_review="not yet done: the team's own check of E7 is still open (team.json untouched by this review); this AI review does not replace it",
    when=f"2026-10-30, late evening, after the training pass of 22:31-22:50 on this Windows PC (results file written {R['written']}); scripts in code/review/multi_product/",
    what=("Independent review of experiment E7 (code/28_multi_product.py: the NLDAS rain, the ERA5 rain to midnight and the AORC rain to midnight side by side as inputs of the all-river LSTM; sets wqME, wqME_shuf, "
          "wqME_older, five seeds each, against the stored lstm_wqa; results/multi_product.json, figures/multi_product.png, 15 networks, 20 LSTM prediction sets and 2 simple-model sets). Every headline number "
          "recomputed from results/preds with the reviewer's own numpy scoring (flood.evaluate only as a second route); the seven input columns rebuilt by another route from data/aorc, data/hourly and the raw CAMELS "
          "NLDAS files; the ROADMAP E7 rule applied block by block; the 15 new and 10 stored networks rebuilt from their .pt files; own causality and leak probes (clock, NLDAS day, ERA5 lag, shuffle, older "
          "columns, made-up hours and values); the operating points and the resampling recomputed; the ridge yardstick refitted; the sentence, the wordings, the disclosure and the figure."),
    files_reviewed=dict(sha256=saved, saved_files_of_the_experiment=dict(n_files=m1["saved_files"]["n_files"], all_hashes_equal_to_the_record=not m1["saved_files"]["mismatches"], mismatches=m1["saved_files"]["mismatches"]),
                        stored_set_files=dict(n_files=m1["stored_set_files"]["n_files"], all_hashes_equal_to_the_record=not m1["stored_set_files"]["mismatches"]),
                        script=m1["script"], log=m1["log"], smoke_test=m1.get("smoke_test"), networks_checked_inside=len(m1["networks"]["per_file"]), network_problems=m1["networks"]["problems"]),
    stored_sets_reviewed=R["integrity"]["stored_sets_read"],
    record_script_sha256=m1["script"]["sha256_now"],
    method=dict(scoring="own numpy code from the obs and pred arrays of results/preds/*.npz: Q95 of the training years 1980-10-01..2000-09-30; onsets = target days of the test years above Q95 whose flow lead days "
                        "earlier was at or below Q95; warning = forecast > trigger x Q95; pooled over the 12 basins; NSE per basin over the test days; triggers chosen on the validation years by the documented rules of "
                        "code/warning_policy.choose_trigger (grid 0.50-2.00) and of the verdict readings (0.30-1.50); resampling by 13_uncertainty.py's documented recipe with per-basin water-year sums from pandas groupby",
                columns="own route (m3): zoneinfo instants of 00:00 local per day, hour-ending stamps in (start, end], cumulative sums and searchsorted; the same day assignment summed by pandas groupby for the bit-for-bit "
                        "comparison with the sha256 the results file records; NLDAS parsed from the raw CAMELS file; the shuffle rebuilt from its documented algorithm",
                networks="flood.Cfg / LSTMNet / Run.predict as how_to_reload says, on tables whose seven columns are the reviewer's, one torch thread",
                torch_threads=1, nothing_trained=True, script_not_run="code/28_multi_product.py was not run (its --rescore pass is for the lead after the fixes)", nothing_installed=True,
                project_writes="only code/review/multi_product/ and this record; the results file, the prediction sets, the networks and the figure were read, never written",
                python="<venv> (torch 2.2.2+cpu CPU, numpy 1.26.4), PYTHONUTF8=1, from the project root"),
    reproduced=dict(how="see method; every comparison is a (file, mine) pair at 1e-9 unless stated",
                    summary=dict(entries=values_compared, matching=values_compared - mismatches, not_matching=sum((x["compare"]["not_matching"] for x in (m1, m2, m3, m4, m5, m7)), [])),
                    by_check={k: x["compare"] for k, x in (("m1_integrity", m1), ("m2_headline", m2), ("m3_columns", m3), ("m4_networks", m4), ("m5_outcome", m5), ("m7_ridge", m7))},
                    headline_test_years_trigger_1_file_vs_mine=head,
                    seed_means_equal_mean_of_seed_files=M["seed_mean"],
                    per_basin_lead1_wqME_minus_wqa_points=dict(file=W["per_basin_difference_points"], wilcoxon_file=dict(p=W["p_value"], p_exact=W["p_value_exact"], p_approx=W["p_value_normal_approximation"], n_used=W["n_used"]),
                                                              wilcoxon_mine=WX),
                    paired_onsets_lead1={k: dict(file=R["readings"][k]["overlap_of_warned_onsets"]["l1"], mine=dict(warned_by_both=PAIRS[k]["l1"]["both"], warned_only_by_first=PAIRS[k]["l1"]["only_first"],
                                                                                                                 warned_only_by_second=PAIRS[k]["l1"]["only_second"], sign_test_p=PAIRS[k]["l1"]["sign_p"])) for k in PAIRS},
                    operating_points_lead1={n: dict(file={r: dict(trigger=R["operating_points"]["sets"][n]["l1"]["rules"][r]["trigger"], test_pod=R["operating_points"]["sets"][n]["l1"]["rules"][r]["test"]["onset"]["pod"],
                                                                  test_far=R["operating_points"]["sets"][n]["l1"]["rules"][r]["test"]["onset"]["far"]) for r in ("csi", "far_cap")},
                                                    mine={r: dict(trigger=OPS[n]["l1"][r]["trigger"], test_pod=OPS[n]["l1"][r]["test_pod"], test_far=OPS[n]["l1"][r]["test_far"]) for r in ("csi", "far_cap")})
                                            for n in ("wqME", "wqa", "wqME_shuf", "wqME_older", "wqnh", "base_linreg_wqME", "base_linreg_wqa", "base_rf_wqME", "base_rf_wqa")},
                    resampling_lead1={k: dict(file=R["resampling"]["statistics"][k]["both"], mine=RS[k]["both"]) for k in ("onsets_diff.wqME_minus_wqa.l1", "nse_diff_of_medians.wqME_minus_wqa.l1",
                                                                                                                         "onsets_diff.wqME_minus_wqME_shuf.l1", "onsets_diff.wqME_minus_wqME_older.l1")},
                    ridge_refit=dict(max_abs_diff=m7["max_abs_diff"], same_missing_days=m7["same_missing_days"], n_features=m7["n_features"])),
    columns_rebuilt=dict(sha256_equal_to_the_results_file={c: m3["rows"][f"sha256 of input column {c} (all basins stacked, float64) = inputs.sha256_of_input_columns"]["ok"] for c in R["inputs"]["sha256_of_input_columns"]},
                         route_a_vs_route_b_max_abs_diff_mm=m3["route_a_vs_route_b_max_abs_diff_mm"], shuffle=m3["shuffle"], nldas_day=nd, aorc_vs_era5_lag=m3["aorc_vs_era5_lag"],
                         clock_per_basin={g: {k: {kk: vv for kk, vv in v.items() if kk != "mean_mm_per_day"} for k, v in r.items()} for g, r in m3["clock"].items()}),
    networks_rebuilt=dict(new=m4["per_file"], stored_identity_all_five_seeds=m4["identity"], torch=m4["torch"], threads=m4["threads"]),
    causality=dict(made_up_hours=dict(cases=m3["n_made_up_cases"], all_pass=all(v["passed"] for v in m3["made_up_hours"].values())), made_up_values=m4["made_up_values"],
                   scalers_from_training_rows=all(v["scalers_from_training_rows"] for v in m4["per_file"].values())),
    leakage_check=("No leak found. The issue time is 24:00 local of day t: every AORC and ERA5 hour summed into day t ends at or before it (own route, bit for bit with the script's columns); the NLDAS day ends about "
                   "six hours before it (measured against the AORC hours); ERA5's hours are, if anything, an hour late against AORC; Q95 and the scalers come from the training years, the stopping point from the "
                   "validation NSE, the triggers of the operating points and verdict readings from the validation years; the shuffled columns never cross a split part (they do move values within the test years, "
                   "which can only help the control); the older columns read day t-1; the script was fixed (hash, modification times) before the first training; no extension pass. The two columns NLDAS and ERA5 "
                   "are, however, products made after the fact, like AORC (must-fix 1): that is hindsight in what the network is given, not a leak of the test years into training."),
    rule_check=rule_check,
    wordings=m5["wordings"],
    figure=m5["figure"],
    findings=F_,
    counts=counts,
    verdict=verdict,
    verdict_text=verdict_text,
    scripts=[f"code/review/multi_product/{x}" for x in ("README.md", "m_common.py", "m1_integrity.py", "m3_columns.py", "m2_headline.py", "m4_networks.py", "m7_ridge.py", "m5_outcome.py", "m6_assemble.py", "run_checks.sh")],
    scripts_how_to_run="from the project root with the project environment, PYTHONUTF8=1, one torch thread: run_checks.sh (m1, m3, m2, m4, m7, m5, m6 in this order; about 2 minutes; outputs to REVIEW_OUT); "
                       "m6_assemble.py --write-into-project writes this record",
    not_checked=["No training was done: whether a new training with the same seeds reproduces the networks on this PC (4 threads) is untested; the stored forecasts and networks are what was checked.",
                 "The forest yardstick (base_rf_wqME) was not refitted (about 80 s with 4 jobs); its forecasts were rescored, and the ridge, which shares the feature code, was refitted bit for bit.",
                 "The hindsight of the wqME networks was not measured (no Stage IV, IMERG or other real-time stand-in for the three columns); must-fix 1 is about the wording, not a measured number.",
                 "ERA5's and NLDAS's real-time latencies were not looked up again; the review relies on the project's own records (results/rain_timing.json, results/rain_to_midnight.json) for them.",
                 "code/28_multi_product.py --rescore and --figure were not run (the figure was read as an image)."],
    dates=dict(builder_training_pass="2026-10-30 22:31 to 22:50 (results file written 2026-10-30 22:50)", results_file_written=f"{R['written']} (written; passes[-1].finished_local_time {R['passes'][-1]['finished_local_time']})",
               review="2026-10-30"),
)
jsave("review_multi_product.json", rec)
print(json.dumps(counts), verdict)
if WRITE:
    now = {f: sha256(f) for f in saved}
    changed = [f for f in saved if now[f] != saved[f]]
    assert not changed, f"reviewed files changed since m1: {changed}"
    bad = [f for f, h in R["integrity"]["sha256_of_saved_files"].items() if sha256(f) != h]
    assert not bad, f"saved files changed: {bad}"
    (PROJ / "results" / "review_multi_product.json").write_text(json.dumps(rec, indent=1, default=lambda o: None if isinstance(o, float) and not math.isfinite(o) else str(o), allow_nan=False), encoding="utf-8")
    print("wrote results/review_multi_product.json")
