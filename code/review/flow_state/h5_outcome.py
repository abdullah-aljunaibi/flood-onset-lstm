"""E4e review, check 5: the sentence, the wordings, the disclosure and the figure. Every clause of outcome_full rebuilt from the REVIEWER's recomputed numbers (h2_headline.json)
with the script's own rounding rules and looked for in the stored text (so every number of the sentence is traced to a table the review reproduced); the fixed wordings; the status;
the disclosure of the choices against the work order; what_was_not_done; the delivery assumptions and the 'gauge's own record' claim as the file words them; the figure's labels
(read by the reviewer on the image) against the JSON; the 70.1% of wqaWHF22_s against the GO rule. Seconds."""
from h_common import *
import re

t0 = time.time()
R = jload(RESULTS)
H = jread("h2_headline.json")
M, ops = H["headline_mine"], H["operating_points_mine"]
pc = lambda x: f"{round(100 * x)}%"
p1 = lambda x: f"{100 * x:.1f}%"
n2 = lambda x: f"{x:.2f}"
text = R["outcome_full"]
assert text == R["outcome"] + " " + R["outcome_caveat"], "outcome_full is not outcome + outcome_caveat"
l1 = lambda n: M[n]["l1"]
rngp = lambda n: (min(M[n]["l1"]["seed_onset_pod"]), max(M[n]["l1"]["seed_onset_pod"]))
rngn = lambda n: (min(M[n]["l1"]["seed_nse_median"]), max(M[n]["l1"]["seed_nse_median"]))
tH, tH22, tA, tWH, tF, tWHF, tP = (l1(x)["onset"] for x in ("wqaH_s", "wqaH22_s", "wqa_s", "wqaWH_s", "wqaF_s", "wqaWHF_s", "wqaP_s"))
n_on = tH["hits"] + tH["misses"]
ov = H["paired_mine"]["wqaH_s_vs_wqa_s"]["l1"]
opH, opA = ops["wqaH_s l1 far_cap"], ops["wqa_s l1 far_cap"]
vH = ops["wqaH_s l1 plain"]["test_verdict"]
gap_on, gain_on = tP["pod"] - tA["pod"], tH["pod"] - tA["pod"]
gap_nse, gain_nse = l1("wqaP_s")["nse_median"] - l1("wqa_s")["nse_median"], l1("wqaH_s")["nse_median"] - l1("wqa_s")["nse_median"]
cov = R["data_coverage"]["per_basin"]
N_FB, N_FB_TEST, N_DAYS = sum(v["fallback_days_24"] for v in cov.values()), sum(v["fallback_days_24_test_years"] for v in cov.values()), sum(v["days"] for v in cov.values())
h3 = jread("h3_columns.json")
clauses = {
    "onsets warned, base to test, seed range": f"onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone (lstm_wqa_s, stored) to {pc(tH['pod'])} (single trainings {pc(rngp('wqaH_s')[0])} to {pc(rngp('wqaH_s')[1])}; the onsets of one storm are not independent):",
    "points and hits": f"{100 * (tH['pod'] - tA['pod']):+.1f} points ({tH['hits']} against {tA['hits']} hits of {n_on});",
    "22:00 variant": f"with the last available hour ending 22:00 instead of 24:00, {pc(tH22['pod'])} (single trainings {pc(rngp('wqaH22_s')[0])} to {pc(rngp('wqaH22_s')[1])});",
    "false share": f"the share of false warnings among the onset warnings is {pc(tH['far'])} against {pc(tA['far'])};",
    "median NSE and seed range": f"the median NSE goes from {n2(l1('wqa_s')['nse_median'])} to {n2(l1('wqaH_s')['nse_median'])} (the seed-averaged forecast; single trainings {n2(rngn('wqaH_s')[0])} to {n2(rngn('wqaH_s')[1])});",
    "verdict words at trigger 1.0": f"the verdict of results/flood_analysis.json at trigger 1.0 is {vH['onset']} for the onsets and {vH['all']} for all high-flow days.",
    "success rule and the seed spans": f"is {'met' if H['success_mine']['met'] else 'not met'}: the five single trainings with the river's state span {p1(rngp('wqaH_s')[0])} to {p1(rngp('wqaH_s')[1])} against {p1(rngp('wqa_s')[0])} to {p1(rngp('wqa_s')[1])} without it;",
    "only-by counts": f"{ov['warned_only_by_first']} onsets are warned only with it against {ov['warned_only_by_second']} only without it.",
    "wqaWH_s": f"With the six within-day AORC rain columns beside it (wqaWH_s) {pc(tWH['pod'])} (single trainings {pc(rngp('wqaWH_s')[0])} to {pc(rngp('wqaWH_s')[1])});",
    "forecast sets": f"with the GEFS forecast as well (wqaWHF_s against the stored lstm_wqaF_s) 1-day onsets warned go from {pc(tF['pod'])} to {pc(tWHF['pod'])} (single trainings {pc(rngp('wqaWHF_s')[0])} to {pc(rngp('wqaWHF_s')[1])}; false {pc(tWHF['far'])} against {pc(tF['far'])}), "
                     f"at 2 and 3 days from {pc(M['wqaF_s']['l2']['onset']['pod'])} to {pc(M['wqaWHF_s']['l2']['onset']['pod'])} and from {pc(M['wqaF_s']['l3']['onset']['pod'])} to {pc(M['wqaWHF_s']['l3']['onset']['pod'])}.",
    "closure": f"The five columns close {pc(gain_on / gap_on)} of the gap in onsets warned from wqa_s ({pc(tA['pod'])}) to the perfect rain forecast in the slots (wqaP_s, {pc(tP['pod'])}, an upper bound that cannot be had in practice) and {pc(gain_nse / gap_nse)} of the gap in median NSE.",
    "30%-false operating point": f"At the 30%-false operating point (the trigger chosen on the validation years so that at most 30% of the onset warnings are false: {opH['trigger']:.2f}) wqaH_s warns of {pc(opH['test_onset']['pod'])} with {pc(opH['test_onset']['far'])} false on the test years, "
                                 f"against {pc(opA['test_onset']['pod'])} with {pc(opA['test_onset']['far'])} false for wqa_s (trigger {opA['trigger']:.2f}).",
    "fallback counts": f"The hourly record has gaps: on {h3['fallback_total_mine']['24']} of the {h3['basin_days_total']} basin-days ({h3['fallback_test_total_mine']} in the test years) the five columns fall back to the daily mean (hourly_known = 0);",
    "seed-count sentence": f"Every number of the new sets is a five-seed number; the stored sets have three seeds; the split has {n_on} test-year onsets and its base is weaker than the main-split wqa.",
}
found = {k: (v in text) for k, v in clauses.items()}
nums_in_text = re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?", text)
covered = set()
for v in clauses.values():
    covered |= set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?%?", v))
untraced = [x for x in nums_in_text if x not in covered and x not in ("1", "2", "3", "30%", "2000-10-01", "2008-09-30", "2008-10-01", "2010-09-30", "2010-10-01", "2014-09-30")]
untraced = [x for x in untraced if not re.fullmatch(r"(19|20)\d\d(-\d\d)*", x)]
out = dict(clauses_rebuilt_from_my_numbers={k: dict(text=v, found_in_outcome_full=found[k]) for k, v in clauses.items()}, all_clauses_found=all(found.values()), n_numbers_in_outcome_full=len(nums_in_text),
           numbers_not_covered_by_a_rebuilt_clause=untraced, split_text_present=R["split_text"] in text)
print("clauses of outcome_full rebuilt from my numbers and found in the text:", sum(found.values()), "of", len(found), [k for k, v in found.items() if not v], "| numbers not covered:", untraced)

# ---- the wordings, the status, the disclosure ------------------------------------------------------------------------------------------------------------------------------------
J = json.dumps(R)
work_order = (PROJ / "NEXT_EXPERIMENTS_2026-10-28.md").read_text(encoding="utf-8")
e4e_section = work_order.split("### E4e.")[1].split("### E4c.")[0]
W = dict(aorc_wording_in_outcome_full="an analysis made after the fact" in R["outcome_caveat"], aorc_wording_in_figure_footer="an analysis made after the fact" in (PROJ / SCRIPT).read_text(encoding="utf-8"),
         hindsight_wording_beside_the_aorc_clause="the AORC rain to midnight is an analysis made after the fact (hindsight at most about 3 points)" in text,
         hindsight_bound_measured_for="the daily total only (E1b review: the saved wqa networks given Stage IV for the issue day); nothing of the kind exists for the six within-day columns of wqaWH_s / wqaWHF_s (E4b review, should-fix 2), and the text attaches no caveat to the wqaWH_s numbers",
         upper_bound_wording_beside_wqaP_s=f"(wqaP_s, {pc(tP['pod'])}, an upper bound that cannot be had in practice)" in text and "an upper bound that cannot be had in practice" in R["design"]["stored_sets"]["wqaP_s"]["what"],
         split_caveat_comparable_only_inside_it="its numbers are comparable only inside it" in text and "comparable only inside it" in R["split_text"],
         base_weaker_than_main_split="its base is weaker than the main-split wqa" in text,
         gauge_record_claim="the hourly flow is the gauge's own record, an input a real-time service has" in text,
         gauge_record_claim_qualified_in_text=any(w in text for w in ("provisional", "delivery delay", "delay of", "latency")),
         provisional_mentioned_anywhere_in_file="provisional" in J.lower(), delay_mentioned_in_design="two hours before the issue time" in J,
         zero_delay_of_the_24_variant_stated_plainly=any(w in J for w in ("zero delivery delay", "no delivery delay", "assumes no delay", "delivered at the issue time")),
         seventy_percent_of_wqaWHF22_in_outcome_full=("70%" in text), wqaWHF22_mentioned_in_outcome_full="wqaWHF22_s" in text,
         wqaWHF22_plain_verdict_onsets=R["operating_points"]["sets"]["wqaWHF22_s"]["l1"]["rules"]["plain"]["test_verdict"]["onset"], wqaWHF22_plain_far=R["summary_test"]["sets"]["wqaWHF22_s"]["l1"]["onset"]["far"],
         go_met_by_any_set_on_onsets_test_years=any(x["reading"].startswith("onsets") for x in H["go_scan"]), go_met_on_all_days_by=[f"{x['set']} l{x['lead']} {x['rule']}" for x in H["go_scan"] if x["reading"] == "all high-flow days"],
         status_text=R["status"], status_says_not_yet_reviewed=R["status"].startswith("NOT YET REVIEWED"), status_flips_only_on_the_next_pass='"reviewed: ' in (PROJ / SCRIPT).read_text(encoding="utf-8") or "reviewed: {rel(REVIEW_FILE)} exists" in (PROJ / SCRIPT).read_text(encoding="utf-8"),
         choices=R["choices_made_after_looking_at_test_scores"], choices_claim_seeds_fixed_in_work_order="seeds" in R["choices_made_after_looking_at_test_scores"][0] and "NEXT_EXPERIMENTS_2026-10-28.md" in R["choices_made_after_looking_at_test_scores"][0],
         work_order_e4e_names_a_seed_count=bool(re.search(r"\b(three|five|3|5) seeds", e4e_section)),
         work_order_e4e_trainings_phrase=next((m.group(0) for m in [re.search(r"a night \+ \d+-\d+ trainings", line) for line in work_order.splitlines() if line.startswith("| E4e |")] if m), None),
         work_order_e4e_success_rule_stated="every seed" in e4e_section or "success rule" in e4e_section.lower(), script_default_seeds=3, run_seeds=R["seeds"], one_training_pass=len(R["passes"]) == 1 and R["passes"][0]["kind"] == "training run",
         success_rule_met_with_first_three_seeds=H["success_with_three_seeds"]["met_with_three_seeds"], seeds_of_wqaH_s_below_best_wqa_s_seed=H["success_with_three_seeds"]["seeds_of_wqaH_s_at_or_below_the_best_seed_of_wqa_s"],
         what_was_not_done=R["what_was_not_done"], what_was_not_done_lists_five_seeds_although_five_were_used=any("five seeds" in x for x in R["what_was_not_done"]) and len(R["seeds"]) == 5,
         what_was_not_done_lists_the_review=any("independent review" in x for x in R["what_was_not_done"]), quoting_note=[n for n in R["notes"] if n.startswith("Quoting")],
         mismatch_example_in_docstring="53 against 175 mm/day" in (PROJ / SCRIPT).read_text(encoding="utf-8"), mismatch_rule_decides_alone_24=h3["mismatch_rule_alone"]["total_decided_alone_24"], mismatch_rule_decides_alone_22=h3["mismatch_rule_alone"]["total_decided_alone_22"],
         delivery_24_vs_22_seed_separation={fam: {k: R["delivery"]["families"][fam][k]["every_seed_24_above_22"] or R["delivery"]["families"][fam][k]["every_seed_22_above_24"] for k in ("l1", "l2", "l3")} for fam in ("H", "WH", "WHF")},
         delivery_24_minus_22_points_l1={fam: R["delivery"]["families"][fam]["l1"]["difference_points_24_minus_22"] for fam in ("H", "WH", "WHF")},
         difference_by_seed_points_truncated=len(H["difference_by_seed_points_truncated"]), disclosure_sentence_present="Claude Code" in J or "AI" in R["status"])
out["wordings"] = W
# the figure: labels read by the reviewer on the image (figures/flow_state.png, 2300 x 1760) against the JSON
S = R["summary_test"]["sets"]
order = ["wqa_s", "wqaH_s", "wqaH22_s", "wqaWH_s", "wqaWH22_s", "wqaF_s", "wqaWHF_s", "wqaWHF22_s", "wqaP_s"]
read_pod = {"wqa_s": 41, "wqaH_s": 47, "wqaH22_s": 43, "wqaWH_s": 59, "wqaWH22_s": 57, "wqaF_s": 61, "wqaWHF_s": 68, "wqaWHF22_s": 70, "wqaP_s": 72}
read_nse = {"wqa_s": 0.71, "wqaH_s": 0.80, "wqaH22_s": 0.74, "wqaWH_s": 0.85, "wqaWH22_s": 0.81, "wqaF_s": 0.77, "wqaWHF_s": 0.83, "wqaWHF22_s": 0.80, "wqaP_s": 0.81}
read_dots = {"wqa_s": 3, "wqaH_s": 5, "wqaH22_s": 5, "wqaWH_s": 5, "wqaWH22_s": 5, "wqaF_s": 3, "wqaWHF_s": 5, "wqaWHF22_s": 5, "wqaP_s": 3}
fig = dict(how="the reviewer read the bar labels and counted the dots on the image; no pixel measurement", bars_in_order=order,
           pod_labels_equal_json=all(read_pod[n] == round(100 * S[n]["l1"]["onset"]["pod"]) for n in order), nse_labels_equal_json=all(abs(read_nse[n] - round(S[n]["l1"]["nse_median"], 2)) < 1e-9 for n in order),
           dot_counts_equal_n_seeds=all(read_dots[n] == R["n_seeds_per_set"][n] for n in order), labels_read={n: dict(pod=read_pod[n], nse=read_nse[n], dots=read_dots[n]) for n in order},
           go_line_label="GO asks 70% warned (at most 30% false)", false_share_shown_on_the_figure=False, wqaWHF22_bar_reaches_the_go_line=round(100 * S["wqaWHF22_s"]["l1"]["onset"]["pod"]) >= 70,
           header_status_line_cut_mid_word=("preliminary a" in "NOT YET REVIEWED: no independent review record (results/review_flow_state.json) exists for this experiment; every number of this file is preliminary a") and R["status"][:150].endswith(" a"),
           header_status_source="R['status'][:150] in draw()", footer_has_aorc_wording=True, footer_has_upper_bound_wording=True, footer_has_split_caveat=True, footer_gauge_record_claim="the hourly flow is the gauge's own record" )
out["figure"] = fig
print("figure labels equal the JSON:", fig["pod_labels_equal_json"], fig["nse_labels_equal_json"], "dots:", fig["dot_counts_equal_n_seeds"], "| header cut mid-word:", fig["header_status_line_cut_mid_word"])
print("wordings:", {k: v for k, v in W.items() if isinstance(v, bool)})
out["seconds"] = round(time.time() - t0, 1)
jsave("h5_outcome.json", out)
print("h5 done in", out["seconds"], "s")
