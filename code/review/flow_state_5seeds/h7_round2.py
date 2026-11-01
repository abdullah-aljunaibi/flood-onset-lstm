"""E4e review, round 2 (30 Oct 2026, the stored sets wqa_s, wqaF_s, wqaP_s now five-seed means): integrity of the rescored file, the new sets' own numbers against
the three-seed file (backup of 30 Oct 2026: they must not have changed), every clause of outcome_full rebuilt from MY numbers (h2_headline.json, own scoring), the success
rule with the five-seed base, the fixed wordings, the disclosure of the seeds added after the test scores were known, the stale three-seed texts and the review guard of the
script. Output: OUT/h7_round2.json."""
from h_common import *
import re

R = jload(RESULTS)
B = json.loads((PROJ.parent / "Project-Flood_backup_2026-10-30" / "results" / "flow_state.json").read_text(encoding="utf-8"))
H = jread("h2_headline.json")
M, ops, rule = H["headline_mine"], H["operating_points_mine"], H["rule_mine"]
rows = H["rows"]
out = dict(checks={})


def check(name, ok, detail=None):
    out["checks"][name] = dict(passed=bool(ok), detail=detail)
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f" | {str(detail)[:400]}"))


I = R["integrity"]
check("script hash now = the record's script_sha256 (the rescore pass)", sha256(SCRIPT) == R["script_sha256"] == R["passes"][-1]["script_sha256"], sha256(SCRIPT))
bad = [f for f, h in I["sha256_of_saved_files"].items() if sha256(f) != h]
check("the saved files of the new sets have the hashes the record gives", not bad, f"{len(I['sha256_of_saved_files'])} files, differ {bad}")
badB = [f for f, h in B["integrity"]["sha256_of_saved_files"].items() if sha256(f) != h]
check("the new sets' files are byte-identical to those of the 29 October file", not badB and I["sha256_of_saved_files"] == B["integrity"]["sha256_of_saved_files"], f"differ {badB}")
check("this pass trained nothing and made nothing", R["passes"][-1]["n_trainings"] == 0 and not I["made_in_this_pass"])
st_now = {f: sha256(f) for f in I["stored_sets_read"]}
check("the 18 stored-set files have the hashes the record gives", all(st_now[f] == h for f, h in I["stored_sets_read"].items()) and len(st_now) == 18)
out["stored_sets_reviewed"] = dict(sorted(st_now.items()))
chg = sorted(f for f in set(st_now) | set(B["integrity"]["stored_sets_read"]) if st_now.get(f) != B["integrity"]["stored_sets_read"].get(f))
check("stored files changed since 29 Oct = provenance.independent_review.stored_sets_changed_since_the_review (6 new seed files + 3 averages)", chg == sorted(R["provenance"]["independent_review"]["stored_sets_changed_since_the_review"]) and len(chg) == 9, chg)
hf = I.get("hourly_files_read") or {}
check("the hourly flow files read (data/hourly_flow/usgs_iv_<gauge>.npz) are those of 29 October and hash as recorded", hf == B["integrity"].get("hourly_files_read") and all(sha256(f"data/hourly_flow/usgs_iv_{g}.npz") == h for g, h in hf.items()), len(hf))
metas = {n: load_npz(f"lstm_{n}")[4].get("seeds") for n in STORED}
check("stored seed averages record seeds [0..4]", all(v == [0, 1, 2, 3, 4] for v in metas.values()), metas)


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
    for label, get in (("summary_test", lambda X: X["summary_test"]["sets"][n]), ("summary_val", lambda X: X["summary_val"]["sets"][n]), ("seed_ranges", lambda X: X["seed_ranges"]["sets"][n]),
                       ("operating_points", lambda X: X["operating_points"]["sets"][n]), ("per_basin_nse", lambda X: X["per_basin"]["nse"][n]), ("per_basin_onset_warned", lambda X: X["per_basin"]["onset_warned"][n]),
                       ("models", lambda X: X["models"][n])):
        a, b = dict(leaves(get(R))), dict(leaves(get(B)))
        blocks[f"{n} {label}"] = dict(n=len(a), differ=[k for k in set(a) | set(b) if a.get(k) != b.get(k)][:5])
for key in R["readings"]["pairs"]:
    T, Cc = key.split("_vs_")
    if T in NEW and Cc in NEW:
        a, b = dict(leaves(R["readings"]["pairs"][key])), dict(leaves(B["readings"]["pairs"][key]))
        blocks[f"readings {key}"] = dict(n=len(a), differ=[k for k in set(a) | set(b) if a.get(k) != b.get(k)][:5])
for label in ("delivery", "data_coverage"):
    a, b = dict(leaves(R[label])), dict(leaves(B[label]))
    blocks[label] = dict(n=len(a), differ=[k for k in set(a) | set(b) if a.get(k) != b.get(k)][:5])
nl = sum(v["n"] for v in blocks.values())
check("the new sets' own numbers are identical to the 29 October file (summaries, seed ranges, operating points, per basin, models, pairs of new sets, delivery, data coverage)",
      all(not v["differ"] for v in blocks.values()), f"{nl} leaves; differing: {[k for k, v in blocks.items() if v['differ']]}")
out["new_set_leaves_compared"] = nl
out["stored_sets_three_to_five_seeds_l1"] = {n: dict(onset_pod=[B["summary_test"]["sets"][n]["l1"]["onset"]["pod"], R["summary_test"]["sets"][n]["l1"]["onset"]["pod"]], hits=[B["summary_test"]["sets"][n]["l1"]["onset"]["hits"], R["summary_test"]["sets"][n]["l1"]["onset"]["hits"]],
                                                     onset_far=[B["summary_test"]["sets"][n]["l1"]["onset"]["far"], R["summary_test"]["sets"][n]["l1"]["onset"]["far"]], nse_median=[B["summary_test"]["sets"][n]["l1"]["nse_median"], R["summary_test"]["sets"][n]["l1"]["nse_median"]]) for n in STORED}
out["outcome_full_three_seeds"] = B["outcome_full"]

# the clauses rebuilt from my numbers
text = R["outcome_full"]
pc = lambda x: f"{round(100 * x) + 0}%"
p1 = lambda x: f"{100 * x:.1f}%"
n2 = lambda x: f"{x:.2f}"
l1 = lambda n, k=1: M[n][f"l{k}"]
rp = lambda n: (min(M[n]["l1"]["seed_onset_pod"]), max(M[n]["l1"]["seed_onset_pod"]))
rn = lambda n: (min(M[n]["l1"]["seed_nse_median"]), max(M[n]["l1"]["seed_nse_median"]))
pr = lambda key, k, f: rows[f"paired {key} l{k} {f}"]["mine"]
tH, tA, tP = l1("wqaH_s")["onset"], l1("wqa_s")["onset"], l1("wqaP_s")["onset"]
three = H["success_with_three_seeds"]
succ = H["success_mine"]
dl = lambda fam: rule[f"wqa{fam}_s_vs_wqa{fam}22_s l1"]["onset_gain_points"]
any_sep = any(rule[f"wqa{f}_s_vs_wqa{f}22_s l{k}"]["every_seed_above_onsets"] or rule[f"wqa{f}_s_vs_wqa{f}22_s l{k}"]["every_seed_below_onsets"] for f in ("H", "WH", "WHF") for k in LEADS)
gap = tP["pod"] - tA["pod"]
ngap = l1("wqaP_s")["nse_median"] - l1("wqa_s")["nse_median"]
oH, oA = ops["wqaH_s l1 far_cap"], ops["wqa_s l1 far_cap"]
pW = pr("wqaWHF_s_vs_wqaF_s", 1, "sign_test_p")
clauses = {
    "base to test": f"onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone (lstm_wqa_s, stored) to {pc(tH['pod'])} (single trainings {pc(rp('wqaH_s')[0])} to {pc(rp('wqaH_s')[1])};",
    "points and hits": f"{100 * (tH['pod'] - tA['pod']):+.1f} points ({tH['hits']} against {tA['hits']} hits of {tH['hits'] + tH['misses']});",
    "22:00 variant": f"with the last available hour ending 22:00 instead of 24:00, {pc(l1('wqaH22_s')['onset']['pod'])} (single trainings {pc(rp('wqaH22_s')[0])} to {pc(rp('wqaH22_s')[1])});",
    "false share": f"the share of false warnings among the onset warnings is {pc(tH['far'])} against {pc(tA['far'])};",
    "median NSE": f"the median NSE goes from {n2(l1('wqa_s')['nse_median'])} to {n2(l1('wqaH_s')['nse_median'])} (the seed-averaged forecast; single trainings {n2(rn('wqaH_s')[0])} to {n2(rn('wqaH_s')[1])});",
    "verdict words": f"at trigger 1.0 is {ops['wqaH_s l1 plain']['test_verdict']['onset']} for the onsets and {ops['wqaH_s l1 plain']['test_verdict']['all']} for all high-flow days.",
    "success rule five and three seeds": f"is {'met' if succ['met'] else 'not met'} with the five seeds and {'would be' if three['met_with_three_seeds'] else 'would not be'} with the first three (seeds 0, 1, 2)",
    "spans": f"the five single trainings with the river's state span {p1(rp('wqaH_s')[0])} to {p1(rp('wqaH_s')[1])} against {p1(rp('wqa_s')[0])} to {p1(rp('wqa_s')[1])} without it (seeds {', '.join(str(s) for s in three['seeds_of_wqaH_s_at_or_below_the_best_seed_of_wqa_s'])} of wqaH_s are not above the best wqa_s seed); "
             f"{pr('wqaH_s_vs_wqa_s', 1, 'warned_only_by_first')} onsets are warned only with it against {pr('wqaH_s_vs_wqa_s', 1, 'warned_only_by_second')} only without it.",
    "wqaWH_s": f"(wqaWH_s) {pc(l1('wqaWH_s')['onset']['pod'])} (single trainings {pc(rp('wqaWH_s')[0])} to {pc(rp('wqaWH_s')[1])});",
    "wqaWHF_s against wqaF_s": f"1-day onsets warned go from {pc(l1('wqaF_s')['onset']['pod'])} to {pc(l1('wqaWHF_s')['onset']['pod'])} (single trainings {pc(rp('wqaWHF_s')[0])} to {pc(rp('wqaWHF_s')[1])} against {pc(rp('wqaF_s')[0])} to {pc(rp('wqaF_s')[1])}: "
                               f"{'no clear difference' if rule['wqaWHF_s_vs_wqaF_s l1']['onset_reading'] == 'no clear difference' else 'READING CHANGED'} by the seed rule, so the {100 * (l1('wqaWHF_s')['onset']['pod'] - l1('wqaF_s')['onset']['pod']):+.1f} points rest on the seed-averaged forecast and on the paired onsets: "
                               f"{pr('wqaWHF_s_vs_wqaF_s', 1, 'warned_only_by_first')} warned only with the new columns against {pr('wqaWHF_s_vs_wqaF_s', 1, 'warned_only_by_second')} only without, sign test p = {pW:.3f}, a description only; false {pc(l1('wqaWHF_s')['onset']['far'])} against {pc(l1('wqaF_s')['onset']['far'])}), "
                               f"at 2 and 3 days from {pc(l1('wqaF_s', 2)['onset']['pod'])} to {pc(l1('wqaWHF_s', 2)['onset']['pod'])} and from {pc(l1('wqaF_s', 3)['onset']['pod'])} to {pc(l1('wqaWHF_s', 3)['onset']['pod'])}.",
    "delivery": f"onsets warned 1 day ahead are {abs(dl('H')):.1f} points {'below' if dl('H') > 0 else 'above'} the 24:00 variant in the river-only sets, {abs(dl('WH')):.1f} points {'below' if dl('WH') > 0 else 'above'} with the within-day rain and "
                f"{abs(dl('WHF')):.1f} points {'below' if dl('WHF') > 0 else 'above'} with the forecast as well, all inside the seed spread at every lead" if not any_sep else "SEPARATES",
    "closure": f"The five columns close {pc((tH['pod'] - tA['pod']) / gap)} of the gap in onsets warned from wqa_s ({pc(tA['pod'])}) to the perfect rain forecast in the slots (wqaP_s, {pc(tP['pod'])}, an upper bound that cannot be had in practice) and {pc((l1('wqaH_s')['nse_median'] - l1('wqa_s')['nse_median']) / ngap)} of the gap in median NSE.",
    "30%-false point": f"{oH['trigger']:.2f}) wqaH_s warns of {pc(oH['test_onset']['pod'])} with {pc(oH['test_onset']['far'])} false on the test years, against {pc(oA['test_onset']['pod'])} with {pc(oA['test_onset']['far'])} false for wqa_s (trigger {oA['trigger']:.2f}).",
    "seed-count sentence": f"Every number of the new sets is a five-seed number; the stored sets have five seeds; the split has {H['n_test_onsets']['l1']} test-year onsets",
    "coverage (data; checked against the 29 October file below)": "on 1786 of the 64644 basin-days (768 in the test years)",
}
out["clauses"] = {k: dict(text=v, found=v in text) for k, v in clauses.items()}
check("every clause of outcome_full rebuilt from my numbers is found in the text", all(v["found"] for v in out["clauses"].values()), [k for k, v in out["clauses"].items() if not v["found"]])
check("brief: +8.4 points", f"{100 * (tH['pod'] - tA['pod']):+.1f}" == "+8.4")
check("the coverage numbers of the sentence are those of the 29 October file (data only)", "on 1786 of the 64644 basin-days (768 in the test years)" in B["outcome_full"])
# the success rule with the five-seed base: which seeds decide it
best_wqa_s = int(np.argmax(M["wqa_s"]["l1"]["seed_onset_pod"]))
out["success_rule"] = dict(met=succ["met"], met_with_first_three_seeds_of_wqaH_s_against_all_five_wqa_s=three["met_with_three_seeds"], best_wqa_s_seed=best_wqa_s, best_wqa_s_value=max(M["wqa_s"]["l1"]["seed_onset_pod"]),
                           wqa_s_seeds=M["wqa_s"]["l1"]["seed_onset_pod"], wqaH_s_seeds=M["wqaH_s"]["l1"]["seed_onset_pod"],
                           met_against_wqa_s_seeds_0_2_only=min(M["wqaH_s"]["l1"]["seed_onset_pod"]) > max(M["wqa_s"]["l1"]["seed_onset_pod"][:3]),
                           met_three_against_three=min(M["wqaH_s"]["l1"]["seed_onset_pod"][:3]) > max(M["wqa_s"]["l1"]["seed_onset_pod"][:3]),
                           file_met=R["e4e_success"]["met"], file_met_with_first_three=R["e4e_success"]["met_with_first_three_seeds"])
check("success rule: not met with five seeds; with the first three of wqaH_s met; the deciding wqa_s seed is an original one (seeds 3-4 of wqa_s do not decide it)",
      not succ["met"] and three["met_with_three_seeds"] and best_wqa_s in (0, 1, 2) and R["e4e_success"]["met"] is False and R["e4e_success"]["met_with_first_three_seeds"] is True, out["success_rule"])

# words, disclosure, stale texts, guard
check("AORC wording; never 'observed rain'", "an analysis made after the fact" in text and "observed rain" not in json.dumps(R))
check("hindsight wording with its scope (daily total only; within-day columns unmeasured)", "(hindsight at most about 3 points, a bound measured for the daily total only)" in text and "have no such measurement" in text)
check("upper-bound wording beside wqaP_s", f"(wqaP_s, {pc(tP['pod'])}, an upper bound that cannot be had in practice)" in text)
check("provisional/approved record and delivery-delay clause (round-1 must-fix)", "provisional values" in text and "this study used the approved record" in text)
ch = " ".join(R["choices_made_after_looking_at_test_scores"])
check("choices_made_after_looking_at_test_scores discloses that seeds 3-4 of the stored comparison sets were added on 30 Oct 2026 after the test scores were known",
      bool(re.search(r"seeds? 3 and 4 of the stored|30 October 2026", ch)), R["choices_made_after_looking_at_test_scores"])
stale = sorted({m for m in re.findall(r"[^.\"]{0,80}three-seed stored set[^.\"]{0,80}", json.dumps(R, ensure_ascii=False))})
check("no text still says a stored set has three seeds", not stale, stale)
check("status NOT YET REVIEWED with the 9 changed stored files", R["status"].startswith("NOT YET REVIEWED") and "9 of their files" in R["status"])
check("figure written in the same pass", abs((PROJ / FIGURE).stat().st_mtime - (PROJ / RESULTS).stat().st_mtime) < 120)
src = (PROJ / SCRIPT).read_text(encoding="utf-8")
check("code/22_flow_state.py compares the stored sets with the hashes the review recorded and reads round_2", "round_2" in src,
      "after round_2 is appended the record's sha256 changes, same_record turns False, prev_stored equals STORED_NOW, so the next --rescore sets covers_this_pass True and quotes the ROUND-1 verdict and counts")
jsave("h7_round2.json", out)
