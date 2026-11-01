"""E4b review, round 2 (30 Oct 2026, the stored sets wqa, wqaF_sub, wqaP now five-seed means): integrity of the rescored file, the new sets' own numbers against the
three-seed file (backup of 30 Oct 2026: they must not have changed), every clause of outcome_full rebuilt from MY numbers (r2_headline.json, own scoring), the fixed
wordings, the disclosure of the seeds added after the test scores were known, the stale three-seed texts and the review guard of the script. Output: OUT/r7_round2.json."""
from r_common import *
import re

R = jload(RESULTS)
B = json.loads((PROJ.parent / "Project-Flood_backup_2026-10-30" / "results" / "within_day_inputs.json").read_text(encoding="utf-8"))
REV = jload("results/review_within_day_inputs.json")
H = jread("r2_headline.json")
M, ops, succ = H["headline_mine"], H["operating_points_mine"], H["success_mine"]
out = dict(checks={})


def check(name, ok, detail=None):
    out["checks"][name] = dict(passed=bool(ok), detail=detail)
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f" | {str(detail)[:400]}"))


# (a) integrity
I = R["integrity"]
check("script hash now = the record's script_sha256 (the rescore pass)", sha256(SCRIPT) == R["script_sha256"] == R["passes"][-1]["script_sha256"], sha256(SCRIPT))
bad = [f for f, h in I["sha256_of_saved_files"].items() if sha256(f) != h]
check("the 44 saved files of the new sets have the hashes the record gives", not bad and len(I["sha256_of_saved_files"]) == 44, f"{len(I['sha256_of_saved_files'])} files, differ {bad}")
badB = [f for f, h in B["integrity"]["sha256_of_saved_files"].items() if sha256(f) != h]
check("the new sets' files are byte-identical to those of the 29 October file (unchanged networks and forecasts)", not badB and I["sha256_of_saved_files"] == B["integrity"]["sha256_of_saved_files"], f"differ {badB}")
check("this pass trained nothing and made nothing", R["passes"][-1]["n_trainings"] == 0 and not I["made_in_this_pass"])
st_now = {f: sha256(f) for f in I["stored_sets_read"]}
check("the 18 stored-set files have the hashes the record gives", all(st_now[f] == h for f, h in I["stored_sets_read"].items()) and len(st_now) == 18, len(st_now))
out["stored_sets_reviewed"] = dict(sorted(st_now.items()))
chg = sorted(f for f in set(st_now) | set(B["integrity"]["stored_sets_read"]) if st_now.get(f) != B["integrity"]["stored_sets_read"].get(f))
check("stored files changed since 29 Oct = provenance.independent_review.stored_sets_changed_since_the_review (6 new seed files + 3 averages)", chg == sorted(R["provenance"]["independent_review"]["stored_sets_changed_since_the_review"]) and len(chg) == 9, chg)
metas = {n: load_npz(f"lstm_{n}")[4].get("seeds") for n in STORED}
check("stored seed averages record seeds [0..4]; seeds of the file", all(v == [0, 1, 2, 3, 4] for v in metas.values()) and all(R["design"]["stored_sets"][n].get("seeds", SEEDS[n]) == SEEDS[n] for n in STORED), metas)

# (b) the new sets' own numbers against the three-seed file
def leaves(v, path=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from leaves(x, f"{path}.{k}" if path else str(k))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from leaves(x, f"{path}[{i}]")
    else:
        yield path, v


blocks = {}
for n in NEW:
    for label, get in (("summary_test", lambda X: X["summary_test"]["sets"][n]), ("validation", lambda X: X["validation_used_for_no_choice"]["summary"][n]), ("seed_ranges", lambda X: X["seed_ranges"]["sets"][n]),
                       ("operating_points", lambda X: X["operating_points"]["sets"][n]), ("verdicts", lambda X: X["verdicts"]["sets"][n]), ("per_basin_nse", lambda X: X["per_basin"]["nse"][n]),
                       ("per_basin_onset_warned", lambda X: X["per_basin"]["onset_warned"][n]), ("models", lambda X: X["models"][n])):
        a, b = dict(leaves(get(R))), dict(leaves(get(B)))
        blocks[f"{n} {label}"] = dict(n=len(a), differ=[k for k in set(a) | set(b) if a.get(k) != b.get(k)][:5])
for key in ("within_day_against_the_older_control", "within_day_against_q_rel", "forecast_on_top_of_within_day"):         # pairs of two new sets
    a, b = dict(leaves(R["readings"][key])), dict(leaves(B["readings"][key]))
    blocks[f"readings {key}"] = dict(n=len(a), differ=[k for k in set(a) | set(b) if a.get(k) != b.get(k)][:5])
for sl, v in R["little_rain_onsets"]["slices"].items():
    blocks[f"little_rain {sl} new sets"] = dict(n=4, differ=[n for n in NEW if v["warned"][n] != B["little_rain_onsets"]["slices"][sl]["warned"][n] or v["n"] != B["little_rain_onsets"]["slices"][sl]["n"]])
nl = sum(v["n"] for v in blocks.values())
check("the new sets' own numbers are identical to the 29 October file (summary, validation, seed ranges, operating points, verdicts, per basin, models, pairs of new sets, little-rain shares)",
      all(not v["differ"] for v in blocks.values()), f"{nl} leaves; differing blocks: {[k for k, v in blocks.items() if v['differ']]}")
out["new_set_leaves_compared"] = nl
moved = {}
for n in STORED:
    a, b = R["summary_test"]["sets"][n]["l1"], B["summary_test"]["sets"][n]["l1"]
    moved[n] = dict(onset_pod=[b["onset"]["pod"], a["onset"]["pod"]], onset_far=[b["onset"]["far"], a["onset"]["far"]], nse_median=[b["nse_median"], a["nse_median"]], hits=[b["onset"]["hits"], a["onset"]["hits"]])
out["stored_sets_three_to_five_seeds_l1"] = moved
out["outcome_full_three_seeds"] = B["outcome_full"]

# (c) the clauses of outcome_full rebuilt from my numbers
text = R["outcome_full"]
pc = lambda x: f"{round(100 * x) + 0}%"
p1 = lambda x: f"{100 * x:.1f}%"
n2 = lambda x: f"{x:.2f}"
l1 = lambda n: M[n]["l1"]
rp = lambda n: (min(M[n]["l1"]["seed_onset_pod"]), max(M[n]["l1"]["seed_onset_pod"]))
rn = lambda n: (min(M[n]["l1"]["seed_nse_median"]), max(M[n]["l1"]["seed_nse_median"]))
tW, tA, tO, tQ, tP, tWF, tF = (l1(x)["onset"] for x in ("wqaW", "wqa", "wqaW_older", "wqaQ", "wqaP", "wqaWF_sub", "wqaF_sub"))
rows = H["rows"]
mine_row = lambda key: rows[key]["mine"]
pair = lambda key, k, f: mine_row(f"paired {key} l{k} {f}")
sm_n, wet_n = mine_row("little_rain aorc_target_below_5mm n"), mine_row("little_rain aorc_target_at_least_5mm n")
lw = lambda sl, n: mine_row(f"little_rain {sl} warned {n}")
gap, gain = tP["pod"] - tA["pod"], tW["pod"] - tA["pod"]
ngap, ngain = l1("wqaP")["nse_median"] - l1("wqa")["nse_median"], l1("wqaW")["nse_median"] - l1("wqa")["nse_median"]
cvW, cvA = ops["wqaW l1 csi"], ops["wqa l1 csi"]
opW, opA, opO = ops["wqaW l1 far_cap"], ops["wqa l1 far_cap"], ops["wqaW_older l1 far_cap"]
fc = lambda k: f"{pair('within_day_with_the_forecast', k, 'warned_only_by_first')} onsets warned only with the within-day columns against {pair('within_day_with_the_forecast', k, 'warned_only_by_second')} only without them"
clauses = {
    "base to test, seed range": f"onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone (hindsight at most about 3 points) to {pc(tW['pod'])} (single trainings {pc(rp('wqaW')[0])} to {pc(rp('wqaW')[1])};",
    "points, hits, control": f"{100 * gain:+.1f} points over that model ({tW['hits']} against {tA['hits']} hits of {tW['hits'] + tW['misses']}) and {100 * (tW['pod'] - tO['pod']):+.1f} points over the control with the same six columns one day older ({pc(tO['pod'])});",
    "false share": f"the share of false warnings among the onset warnings is {pc(tW['far'])} against {pc(tA['far'])} (the control: {pc(tO['far'])});",
    "median NSE": f"the median NSE goes from {n2(l1('wqa')['nse_median'])} to {n2(l1('wqaW')['nse_median'])} (the seed-averaged forecast; single trainings {n2(rn('wqaW')[0])} to {n2(rn('wqaW')[1])});",
    "verdict words": f"is {R['verdicts']['sets']['wqaW']['l1']['plain']['onset']['verdict']} for the onsets and {R['verdicts']['sets']['wqaW']['l1']['plain']['all']['verdict']} for all high-flow days.",
    "success rule and spans": f"is {'met' if succ['met'] else 'not met'}: the five single trainings with the within-day columns span {p1(rp('wqaW')[0])} to {p1(rp('wqaW')[1])} against {p1(rp('wqa')[0])} to {p1(rp('wqa')[1])} with the AORC rain to midnight alone and {p1(rp('wqaW_older')[0])} to {p1(rp('wqaW_older')[1])} for the older control;",
    "control flat and noisier, only-by counts": f"the control is {'flat' if succ['control_flat'] else 'not flat'} on onsets warned ({pc(tO['pod'])} against {pc(tA['pod'])}) and noisier ({pc(tO['far'])} false against {pc(tA['far'])}; median NSE {n2(l1('wqaW_older')['nse_median'])} against {n2(l1('wqa')['nse_median'])}); "
                                                f"{pair('within_day_columns', 1, 'warned_only_by_first')} onsets are warned only with the within-day columns against {pair('within_day_columns', 1, 'warned_only_by_second')} only without them.",
    "wqaQ": f"The Q95-relative flow alone (wqaQ) gives {pc(tQ['pod'])} (single trainings {pc(rp('wqaQ')[0])} to {pc(rp('wqaQ')[1])}) with a lower median NSE ({n2(l1('wqaQ')['nse_median'])} against {n2(l1('wqa')['nse_median'])}) and {pc(tQ['far'])} false warnings (against {pc(tA['far'])}).",
    "little-rain slice": f"In the {sm_n} onsets with under 5 mm of AORC rain on the target day the share warned goes from {pc(lw('aorc_target_below_5mm', 'wqa'))} to {pc(lw('aorc_target_below_5mm', 'wqaW'))} (older control {pc(lw('aorc_target_below_5mm', 'wqaW_older'))}), "
                         f"in the {wet_n} others from {pc(lw('aorc_target_at_least_5mm', 'wqa'))} to {pc(lw('aorc_target_at_least_5mm', 'wqaW'))}: the gain sits in the wetter onsets",
    "gap and closure": f"The gap in onsets warned from the AORC rain to midnight ({pc(tA['pod'])}) to the perfect rain forecast in the slots (wqaP, {pc(tP['pod'])}, an upper bound that cannot be had in practice) is {100 * gap:.1f} points; the six columns move the share by {100 * gain:+.1f} points and close {pc(gain / gap)} of it; "
                       f"in median NSE the gap is {ngap:.3f} and the move {ngain:+.3f} ({pc(ngain / ngap)} of it).",
    "CSI level": f"At the warning level chosen on the validation years for onsets ({cvW['trigger']:.2f} against {cvA['trigger']:.2f}) wqaW warns of {pc(cvW['test_onset']['pod'])} with {pc(cvW['test_onset']['far'])} false against {pc(cvA['test_onset']['pod'])} with {pc(cvA['test_onset']['far'])} false;",
    "30%-false point": f"{opW['trigger']:.2f}) it warns of {pc(opW['test_onset']['pod'])} with {pc(opW['test_onset']['far'])} false on the test years, against {pc(opA['test_onset']['pod'])} with {pc(opA['test_onset']['far'])} false for the AORC rain to midnight alone (trigger {opA['trigger']:.2f}) "
                       f"and {pc(opO['test_onset']['pod'])} with {pc(opO['test_onset']['far'])} false for the older control (trigger {opO['trigger']:.2f})",
    "seed-count sentence": "Every number of the new sets is a five-seed number; the stored sets have five seeds.",
    "forecast set": f"onsets warned 1 day ahead go from {pc(tF['pod'])} to {pc(tWF['pod'])} (single trainings {pc(rp('wqaWF_sub')[0])} to {pc(rp('wqaWF_sub')[1])}; false {pc(tWF['far'])} against {pc(tF['far'])}; median NSE {n2(l1('wqaWF_sub')['nse_median'])} against {n2(l1('wqaF_sub')['nse_median'])}; {fc(1)}), "
                    f"at 2 and 3 days from {pc(M['wqaF_sub']['l2']['onset']['pod'])} to {pc(M['wqaWF_sub']['l2']['onset']['pod'])} ({fc(2)}) and from {pc(M['wqaF_sub']['l3']['onset']['pod'])} to {pc(M['wqaWF_sub']['l3']['onset']['pod'])} ({fc(3)})",
    "forecast set readings": "by the seed rule 'no clear difference' at every lead" if all(H["rule_mine"][f"within_day_with_the_forecast l{k}"]["onset_reading"] == "no clear difference" for k in LEADS) else "READING CHANGED",
}
out["clauses"] = {k: dict(text=v, found=v in text) for k, v in clauses.items()}
check("every clause of outcome_full rebuilt from my numbers is found in the text", all(v["found"] for v in out["clauses"].values()), [k for k, v in out["clauses"].items() if not v["found"]])
check("brief: +3.9 points", f"{100 * gain:+.1f}" == "+3.9")
# the paired counts at 2 and 3 days: 'cost onsets' (only-without exceeds only-with, sign test)
cost = all(pair("within_day_with_the_forecast", k, "warned_only_by_second") > pair("within_day_with_the_forecast", k, "warned_only_by_first") and pair("within_day_with_the_forecast", k, "sign_test_p") < 0.05 for k in (2, 3))
check("'by the paired counts the within-day columns cost onsets at 2 and 3 days' (only-without > only-with, sign p < 0.05)", cost and "by the paired counts the within-day columns cost onsets at 2 and 3 days" in text,
      {k: (pair("within_day_with_the_forecast", k, "warned_only_by_first"), pair("within_day_with_the_forecast", k, "warned_only_by_second"), pair("within_day_with_the_forecast", k, "sign_test_p")) for k in (2, 3)})
check("'at this operating point the six columns add nothing over the control' (wqaW within 1 point of the control at the 30%-false point)", abs(opW["test_onset"]["pod"] - opO["test_onset"]["pod"]) < 0.01 and "add nothing over the control" in text,
      (opW["test_onset"]["pod"], opO["test_onset"]["pod"]))
check("'the opposite of the expectation' (the gain sits outside the little-rain slice)", lw("aorc_target_below_5mm", "wqaW") < lw("aorc_target_below_5mm", "wqa") and lw("aorc_target_at_least_5mm", "wqaW") > lw("aorc_target_at_least_5mm", "wqa"))

# (d) words, disclosure, stale texts, guard
check("AORC wording in outcome and caveat; never 'observed rain'", "an analysis made after the fact" in R["outcome"] and "an analysis made after the fact" in R["outcome_caveat"] and "observed rain" not in json.dumps(R))
check(f"hindsight wording beside the rain-to-midnight share ({pc(tA['pod'])})", f"{pc(tA['pod'])} with the AORC rain to midnight alone (hindsight at most about 3 points)" in text)
check("upper-bound wording beside wqaP", f"(wqaP, {pc(tP['pod'])}, an upper bound that cannot be had in practice)" in text)
check("'not a service' clause and the bound measured for the daily total only", "not a service" in text and "was measured for the daily total, not for these columns" in text)
ch = " ".join(R["choices_made_after_looking_at_test_scores"])
check("choices_made_after_looking_at_test_scores discloses that seeds 3-4 of the stored comparison sets were added on 30 Oct 2026 after the test scores were known",
      bool(re.search(r"seeds? 3 and 4|seeds 3-4|30 October 2026", ch)), R["choices_made_after_looking_at_test_scores"])
check("e4b_success.rule names the stored wqa's seed count correctly", "three seeds" not in R["e4b_success"]["rule"], R["e4b_success"]["rule"])
stale = sorted({m for m in re.findall(r"[^.\"]{0,80}three-seed stored set[^.\"]{0,80}", json.dumps(R, ensure_ascii=False))})
check("no text still says a stored set has three seeds", not stale, stale)
check("seeds_paired covers all five seeds now (zip of two five-seed lists)", all(R["readings"][k]["by_lead"][f"l{l}"]["onsets"]["seeds_paired"] == [0, 1, 2, 3, 4] for k in R["readings"] if isinstance(R["readings"][k], dict) and "by_lead" in R["readings"][k] for l in LEADS))
check("status NOT YET REVIEWED with the 9 changed stored files", R["status"].startswith("NOT YET REVIEWED") and "9 of their files" in R["status"])
check("figure written in the same pass", abs((PROJ / "figures/within_day_inputs.png").stat().st_mtime - (PROJ / RESULTS).stat().st_mtime) < 120)
src = (PROJ / SCRIPT).read_text(encoding="utf-8")
check("code/21_within_day_inputs.py compares the stored sets with the hashes the review recorded (not only with the previous results file) and reads round_2", "round_2" in src,
      "after round_2 is appended the record's sha256 changes, same_record turns False, prev_stored equals STORED_NOW, so the next --rescore sets covers_this_pass True and quotes the ROUND-1 verdict and counts")
jsave("r7_round2.json", out)
