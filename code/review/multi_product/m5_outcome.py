"""E7 review, check 5: the words. Every clause of outcome and outcome_caveat rebuilt from the REVIEWER'S numbers (OUT/m2_headline.json) with the script's rounding
and looked up in the text; the fixed wordings (AORC, the hindsight bound and where it comes from), the status, the disclosure, what_was_not_done; phrases that would
overstate (observed rain, team checks done) or that are missing (NLDAS and ERA5 are not real-time products either); the figure's bar labels as read on the image
(typed by the reviewer from figures/multi_product.png) against the reviewer's numbers. About 1 s."""
import re

from m_common import *

R = jload(RESULTS)
M = jread("m2_headline.json")["my"]
H = M["headline"]
C = Compare()
txt, cav, full = R["outcome"], R["outcome_caveat"], R["outcome_full"]
pc = lambda x: f"{round(100 * x) + 0}%"
p1 = lambda x: f"{100 * x:.1f}%"
n2 = lambda x: f"{x:.2f}"
on = lambda n, c="pod", k=1: H[n][lk(k)]["onset"][c]
med = lambda n, k=1: H[n][lk(k)]["nse_median"]
rg = lambda n: M["ranges"][n]["l1"]["onset_pod"]
rv2 = jload("results/review_aorc_rain_to_midnight.json")["round_2"]["hindsight"]["five_seed_loosest_bound_points"]
HIND = f"hindsight at most about {round(rv2)} points"
b1 = M["rule"]["block_1"]
b23 = M["rule"]["blocks_2_3"]
pr = M["pairs"]
ops = M["ops"]
rs = M["resampling"]
gy = M["rule"]["yardstick_gains"]
ci = lambda key, s=100.0, dg=1: f"{s * rs[key]['both']['lo']:+.{dg}f} to {s * rs[key]['both']['hi']:+.{dg}f}"
opME, opA = ops["wqME"]["l1"]["far_cap"], ops["wqa"]["l1"]["far_cap"]
d_op = 100 * (opME["test_pod"] - opA["test_pod"])
sh = jread("m3_columns.json")["shuffle"]
# verdict at trigger 1.0 (criteria of results/flood_analysis.json)
vd = lambda T: "GO" if T["pod"] >= 0.70 and T["far"] <= 0.30 else "CAUTION" if T["csi"] >= 0.40 else "NOT YET"
clauses = {
    "o1 levels": f"onsets warned 1 day ahead go from {pc(on('wqa'))} with the AORC rain to midnight alone ({HIND}) to {pc(on('wqME'))} (single trainings {pc(rg('wqME')[0])} to {pc(rg('wqME')[1])};",
    "o2 points and hits": f"{100 * (on('wqME') - on('wqa')):+.1f} points ({H['wqME']['l1']['onset']['hits']} against {H['wqa']['l1']['onset']['hits']} hits of {H['wqME']['l1']['onset']['hits'] + H['wqME']['l1']['onset']['misses']})",
    "o3 controls": f"(the capacity control) {pc(on('wqME_shuf'))}, with them one day older (the clock control) {pc(on('wqME_older'))}",
    "o4 false share": f"the share of false warnings among the onset warnings is {pc(on('wqME', 'far'))} against {pc(on('wqa', 'far'))}",
    "o5 NSE": f"the median NSE goes from {n2(med('wqa'))} to {n2(med('wqME'))}",
    "o6 rule": f"this is {'a clear gain' if M['rule']['met'] else 'not a clear gain'}",
    "o7 verdict": f"the verdict of results/flood_analysis.json is {vd(H['wqME']['l1']['onset'])} for the onsets and {vd(H['wqME']['l1']['all'])} for all high-flow days",
    "c1 block 1": (f"block 1, the gain, {'holds' if b1['holds'] else 'fails'} (every seed of wqME above every seed of wqa: {'yes' if b1['every_seed_above'] else 'no'}; false share not higher: "
                   f"{'yes' if b1['false_wqME'] <= b1['false_wqa'] else 'no'}; Wilcoxon over the basins p = {b1['wilcoxon_p']:.3f}, n = {b1['n_used']})"),
    "c2 block 2": f"block 2, the capacity control flat, {'holds' if b23['block_2_capacity_control_flat']['inside'] else 'fails'} (median single seed {p1(b23['block_2_capacity_control_flat']['median'])}, within wqa's {p1(rg('wqa')[0])} to {p1(rg('wqa')[1])})",
    "c3 block 3": f"block 3, the clock control flat, {'holds' if b23['block_3_clock_control_flat']['inside'] else 'fails'} (median single seed {p1(b23['block_3_clock_control_flat']['median'])}, within that range)",
    "c4 per basin": f"wqME warns of more onsets than wqa in {b1['higher']} of 12 basins and of fewer in {b1['lower']} (median difference {b1['median_diff_points']:+.1f} points)",
    "c5 paired": f"{pr['several_products_against_aorc_alone']['l1']['only_first']} onsets are warned only by wqME and {pr['several_products_against_aorc_alone']['l1']['only_second']} only by wqa",
    "c6 interval onsets": f"lies between {ci('onsets_diff.wqME_minus_wqa.l1')} points (above zero in {pc(rs['onsets_diff.wqME_minus_wqa.l1']['both']['share_above_zero'])} of the resamples)",
    "c7 interval NSE": f"the difference of the median NSE between {ci('nse_diff_of_medians.wqME_minus_wqa.l1', 1.0, 3)}",
    "c8 wqnh": (f"(lstm_wqnh, {pc(on('wqnh'))}) wqME warns of {100 * (on('wqME') - on('wqnh')):+.1f} points more ({pr['several_products_against_nldas_and_era5']['l1']['only_first']} onsets only by wqME, "
                f"{pr['several_products_against_nldas_and_era5']['l1']['only_second']} only by wqnh)"),
    "c9 wq": f"the rain table alone (lstm_wq) warns of {pc(on('wq'))}",
    "c10 NSE leads 2-3": f"the median NSE of wqME is {n2(med('wqME', 2))} and {n2(med('wqME', 3))} against {n2(med('wqa', 2))} and {n2(med('wqa', 3))} for wqa",
    "c11 30%-false point": (f"{opME['trigger']:.2f}) wqME warns of {pc(opME['test_pod'])} with {pc(opME['test_far'])} false on the test years, against {pc(opA['test_pod'])} with {pc(opA['test_far'])} false for wqa "
                            f"(trigger {opA['trigger']:.2f}): {abs(d_op):.1f} points {'more' if d_op > 0 else 'fewer'}"),
    "c12 yardstick": (f"ridge regression moves by {gy['ridge regression']:+.1f} points and the random forest by {gy['random forest']:+.1f} against their wqa versions, the LSTM by {gy['lstm']:+.1f}: "
                      f"{M['rule']['yardstick_reading']}"),
    "c13 hindsight": f"wqME carries the hindsight bound of wqa ({HIND}), measured for wqa and not measured again for wqME",
    "c14 shuffle": f"({100 * sh['share_from_a_later_day']:.0f}% of them to a day before the day they were measured)",
    "c15 seeds": "Every number of the new sets is a five-seed number; lstm_wqa has five seeds, lstm_wq and lstm_wqn three.",
}
found = {k: v in full for k, v in clauses.items()}
for k, v in clauses.items():
    C.add(f"clause {k} found in outcome_full", True, found[k])
C.add("outcome_full = outcome + ' ' + outcome_caveat", full, txt + " " + cav)

# ---- wordings ----------------------------------------------------------------------------------------------------------------------------------------------
low = full.lower()
w = dict(
    aorc_wording_in_outcome="an analysis made after the fact" in txt,
    aorc_wording_count_in_outcome_full=full.count("an analysis made after the fact"),
    hindsight_wording_beside_the_52="(hindsight at most about 4 points)" in txt,
    hindsight_number_from=f"results/review_aorc_rain_to_midnight.json round_2 five_seed_loosest_bound_points = {rv2:.2f} -> 'about {round(rv2)}'",
    handoff_fixed_wording_says_3=("hindsight at most about 3 points" in (PROJ / "HANDOFF.md").read_text(encoding="utf-8")),
    roadmap_e7_says_about_4="at most about 4 points" in (PROJ / "ROADMAP.md").read_text(encoding="utf-8"),
    observed_rain_phrase=bool(re.search(r"observed rain", low)),
    nldas_or_era5_called_after_the_fact_or_not_real_time=bool(re.search(r"(nldas|era5)(?:(?!aorc)[^.;]){0,120}(after the fact|real[- ]time|reanalysis|published .{0,20}late|days late)", low)),   # no AORC in between (the AORC clause is not about them)
    hindsight_of_nldas_and_era5_mentioned=bool(re.search(r"hindsight[^.]{0,200}(nldas|era5)", low)),
    status_not_yet_reviewed=R["status"].startswith("NOT YET REVIEWED"),
    status_says_preliminary="preliminary" in R["status"],
    status_says_team_checks_not_done="The team's own checks are not yet done" in R["status"],
    provenance_team_check=R["provenance"]["team_check"],
    disclosure=R["provenance"]["written_and_run_by"],
    implies_team_checks_done=bool(re.search(r"team('s)? (own )?check(s)? (are|is|were|was) (done|complete)", json.dumps(R).lower())),
    outcome_alone_warning=any("never outcome alone" in x for x in R["notes"]),
    not_part_of_the_hand_in=any("not part of the 2 November hand-in" in x for x in R["what_was_not_done"]),
    what_was_not_done_stage4_only_for_aorc=any("Stage IV test of the saved wqME networks" in x for x in R["what_was_not_done"]) and not any(re.search(r"nldas|era5", x.lower()) and "real" in x.lower() for x in R["what_was_not_done"]),
    interval_says_seeds_not_resampled_in_the_sentence=("seed" in cav[cav.find("drawn again"):cav.find("drawn again") + 400].lower()),
    operating_point_controls_in_the_sentence=bool(re.search(r"30%-false operating point[^.]*(shuf|capacity|clock|older)", full)),
    csi_operating_point_in_the_sentence="CSI" in full,
    nse_basins_lower_in_the_sentence=bool(re.search(r"NSE[^.]{0,120}(of 12 basins|in \d+ of)", full)),
    split_part_in_the_capacity_control_clause_of_outcome="split part" in txt,
)
C.add("AORC wording in outcome", True, w["aorc_wording_in_outcome"])
C.add("hindsight wording beside the 52%", True, w["hindsight_wording_beside_the_52"])
C.add("no 'observed rain'", False, w["observed_rain_phrase"])
C.add("status: NOT YET REVIEWED, preliminary, team's checks not done", True, w["status_not_yet_reviewed"] and w["status_says_preliminary"] and w["status_says_team_checks_not_done"])
C.add("nothing implies the team's checks are done", False, w["implies_team_checks_done"])

# ---- the figure's labels as read on the image by the reviewer --------------------------------------------------------------------------------------------
read_on_image = {"wq": (19, 59, 0.59), "wqn": (30, 63, 0.67), "wqh": (29, 56, 0.66), "wqnh": (32, 59, 0.69), "wqa": (52, 51, 0.76), "wqME": (56, 54, 0.73), "wqME_shuf": (52, 55, 0.75),
                 "wqME_older": (51, 53, 0.74), "base_linreg_wqa": (55, 67, 0.64), "base_linreg_wqME": (56, 67, 0.63), "base_rf_wqa": (48, 53, 0.66), "base_rf_wqME": (49, 55, 0.66)}
fig_ok = {n: [round(100 * on(n)), round(100 * on(n, "far")), round(med(n), 2)] == [v[0], v[1], v[2]] for n, v in read_on_image.items()}
C.add("figure bar labels (12 sets: onsets warned, false share, median NSE) = reviewer's numbers", True, all(fig_ok.values()))
figure = dict(labels_read_on_the_image=read_on_image, labels_agree=fig_ok,
              status_line_overlaps_the_panel_title=True,
              note_on_overlap="the last line of the status text ('are not yet done.') is drawn over the top panel's title 'Onsets warned 1 day ahead ...' (read on the image)",
              footer_hindsight="'AORC rain is an analysis made after the fact: the 52% of wqa carries hindsight at most about 4 points; wqME carries the same bound, which was measured for wqa, not for wqME.' (no word on NLDAS or ERA5)",
              seeds_line="'seeds of the stored sets: wqa 5, wq 3, wqn 3' (wqh and wqnh, five seeds each, not named; their five dots are drawn)",
              go_line="'GO asks 70% warned (at most 30% false)'; the false share is printed beside every bar")
out = dict(compare=C.summary(), rows=C.rows, clauses=clauses, clauses_found=found, n_clauses=len(clauses), n_found=sum(found.values()), wordings=w, figure=figure)
jsave("m5_outcome.json", out)
print(json.dumps(out["compare"]))
print({k: v for k, v in found.items() if not v})
print(json.dumps(w, indent=0))
