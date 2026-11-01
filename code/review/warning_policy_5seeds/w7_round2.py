"""w7 (round 2, 30 Oct 2026): what changed with the five-seed stored sets, and the words.
(a) integrity: the script and module hashes now against the record; every prediction file the record read (inputs_sha256) hashed again; the seed files of each set
    on disk against the record's seed_sets; the meta of the seed averages (seeds [0..4]) and of the new seed files (seed 3/4); the files that differ from the
    three-seed state (backup of 30 Oct 2026, Project-Flood_backup_2026-10-30/results/warning_policy.json) against provenance.independent_review.inputs_changed_...
(b) the backup diff: the sets whose seed count did not change (wq, wqn, wqf) must be identical leaf by leaf; for the others the lead-1 numbers before and after.
(c) every number of outcome_full rebuilt from MY recomputed numbers (w1-w5 outputs), with the record's rounding, and looked for in the text.
(d) the fixed wordings, the disclosure of the seeds added after the test scores were known, the status and the stale guard of the script (what it will say
    once this round_2 is appended). Output: OUT/w7_round2.json."""
from w_common import *
import re

R = record()
B = json.loads((PROJ.parent / "Project-Flood_backup_2026-10-30" / "results" / "warning_policy.json").read_text(encoding="utf-8"))
REV = json.loads((PROJ / "results" / "review_warning_policy.json").read_text(encoding="utf-8"))
W1, W2, W3, W4, W5 = (jread(f"{n}.json") for n in ("w1_headline", "w2_seeds", "w3_any_lead", "w4_bootstrap", "w5_stacker"))
out = dict(checks={})


def check(name, ok, detail=None):
    out["checks"][name] = dict(passed=bool(ok), detail=detail)
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f" | {detail}"))


# (a) integrity -------------------------------------------------------------------------------------------------------------------------------------------
scr = {f: sha256(PROJ / f) for f in ("code/20_warning_policy.py", "code/warning_policy.py", "code/flood.py", "results/warning_policy.json", "figures/warning_policy.png",
                                      "results/diagnostics_2026-10-28/warning_policy_rescore.json")}
out["files_now"] = {f: dict(sha256=h, modified=pd.Timestamp((PROJ / f).stat().st_mtime, unit="s").tz_localize("UTC").tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S")) for f, h in scr.items()}
check("script hash now equals the record's script_sha256", scr["code/20_warning_policy.py"] == R["script_sha256"], scr["code/20_warning_policy.py"])
now_inputs = {p: sha256(PROJ / p) for p in R["inputs_sha256"]}
bad = [p for p in now_inputs if now_inputs[p] != R["inputs_sha256"][p]]
check("every prediction file the record read has the hash it recorded", not bad, f"{len(now_inputs)} files; differ: {bad}")
out["stored_sets_reviewed"] = dict(sorted(now_inputs.items()))
seeds_ok = {n: R["sets"][n]["seed_sets"] == seeds_of(n) for n in SETS}
check("seed files on disk = the record's seed_sets (5 for the six sets of scripts 16/18, 3 for wq, wqn, wqf)", all(seeds_ok.values()), {n: len(seeds_of(n)) for n in SETS})
metas = {}
for n in SETS:
    m = Preds(n).meta
    metas[n] = dict(seeds=m.get("seeds"), script=m.get("script"))
    for s in seeds_of(n)[3:]:
        ms = Preds(s).meta
        metas[s] = dict(seed=ms.get("seed"), script=ms.get("script"), inputs=ms.get("inputs"))
check("seed averages of the six extended sets record seeds [0, 1, 2, 3, 4]", all(metas[n]["seeds"] == [0, 1, 2, 3, 4] for n in SETS if len(seeds_of(n)) == 5), {n: metas[n]["seeds"] for n in SETS})
check("seed files _s3/_s4 record their own seed", all(metas[s]["seed"] == int(s[-1]) for s in metas if re.search(r"_s[34]$", s)))
w1max = max(v["max_abs_diff_vs_flood_thresholds"] for v in W1["thresholds"].values())
out["metas"] = metas
diff_inputs = sorted(p for p in set(R["inputs_sha256"]) | set(B["inputs_sha256"]) if R["inputs_sha256"].get(p) != B["inputs_sha256"].get(p))
ir = R["provenance"]["independent_review"]
check("the files that differ from the three-seed run = provenance.independent_review.inputs_changed_since_the_run_it_covered", diff_inputs == sorted(ir["inputs_changed_since_the_run_it_covered"]), f"{len(diff_inputs)} files")
check("the 12 new seed files are new and the 6 seed averages changed; no other forecast file changed", sorted(p for p in diff_inputs if p not in B["inputs_sha256"]) == sorted(f"results/preds/{s}.npz" for n in SETS for s in seeds_of(n)[3:])
      and sorted(p for p in diff_inputs if p in B["inputs_sha256"]) == sorted(f"results/preds/{n}.npz" for n in SETS if len(seeds_of(n)) == 5))
r1_inputs = (REV.get("files_reviewed") or {})
out["round1_files_reviewed_hashes_now"] = {f: dict(round1=(v.get("sha256") if isinstance(v, dict) else v), now=sha256(PROJ / f) if (PROJ / f).exists() else None) for f, v in r1_inputs.items()}

# (b) the backup diff -------------------------------------------------------------------------------------------------------------------------------------
def leaves(v, path=""):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from leaves(x, f"{path}.{k}" if path else str(k))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from leaves(x, f"{path}[{i}]")
    else:
        yield path, v


same = {}
for n in ("lstm_wq", "lstm_wqn", "lstm_wqf"):
    a, b = dict(leaves(R["sets"][n])), dict(leaves(B["sets"][n]))
    same[n] = dict(n_leaves=len(a), differing=[k for k in set(a) | set(b) if a.get(k) != b.get(k)][:10])
check("wq, wqn, wqf (seed counts unchanged): every leaf of sets[...] identical to the three-seed file", all(not v["differing"] for v in same.values()), {n: v["n_leaves"] for n, v in same.items()})
moved = {}
for n in SETS:
    if B["sets"][n]["n_seeds"] == R["sets"][n]["n_seeds"]:
        continue
    for rule in RULES:
        a, b = R["sets"][n]["leads"]["l1"]["rules"][rule], B["sets"][n]["leads"]["l1"]["rules"][rule]
        moved[f"{n} l1 {rule}"] = dict(trigger=[b["trigger"], a["trigger"]], pod=[round(b["test"]["onset"]["pod"], 4), round(a["test"]["onset"]["pod"], 4)], far=[round(b["test"]["onset"]["far"], 4), round(a["test"]["onset"]["far"], 4)])
out["three_to_five_seeds_l1"] = moved
out["outcome_three_seeds"], out["outcome_caveat_three_seeds"] = B["outcome"], B["outcome_caveat"]

# (c) outcome_full rebuilt from my numbers -----------------------------------------------------------------------------------------------------------------
of, o, cav = R["outcome_full"], R["outcome"], R["outcome_caveat"]
L = lambda n, k=1: W1["sets"][n]["leads"][f"l{k}"]
T = lambda n, rule, k=1, kind="onset": L(n, k)["rules"][rule]["test"][kind]
rg = lambda v: f"{pc(v[0])}-{pc(v[1])}"
s2 = W2["sets"]["lstm_wqa"]["leads"]["l1"]
bt = W4["sets"]["lstm_wqa"]["l1"]
st = lambda n: W5["refit"][n]["l1"]["test"]["onset"]
al = W3["sets"]["lstm_wqa"]["csi"]
go_onsets = sorted(f"{n} l{k} {rule}" for n in SETS for k in LEADS for rule in RULES if verdict(T(n, rule, k)) == "GO")
realistic = ["lstm_wq", "lstm_wqn", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqa_s", "lstm_wqaF_s"]
go_all_real = [(n, k, rule) for n in realistic for k in LEADS for rule in RULES if verdict(T(n, rule, k, "all")) == "GO"]
single_go_wqf = [t for t, v in W2["sets"]["lstm_wqf"]["leads"]["l2"]["csi"]["per_seed"].items() if verdict(dict(v["test_at_set_trigger"], correct_neg=0)) == "GO"] + \
                [t for t, v in W2["sets"]["lstm_wqf"]["leads"]["l2"]["far_cap"]["per_seed"].items() if verdict(dict(v["test_at_set_trigger"], correct_neg=0)) == "GO"]
label = {"lstm_wqa": "the model with the AORC rain to midnight", "lstm_wqaF_sub": "the model with the GEFS forecast (substituted)"}
clauses = {
    "CSI level, wqa": f"the LSTM with AORC rain to midnight (an analysis made after the fact) warns of {pc(T('lstm_wqa', 'csi')['pod'])} of the {L('lstm_wqa')['n_test_onsets']} onsets 1 day ahead (hindsight at most about 3 points) with {pc(T('lstm_wqa', 'csi')['far'])} of its warnings false",
    "30%-false level, wqa": f"it warns of {pc(T('lstm_wqa', 'far_cap')['pod'])} (test false share {pc(T('lstm_wqa', 'far_cap')['far'])})",
    "perfect rain / weather": f"the perfect-rain model keeps {pc(T('lstm_wqaP', 'far_cap')['pod'])} and the perfect-weather model {pc(T('lstm_wqf', 'far_cap')['pod'])} (both an upper bound that cannot be had in practice)",
    "GO onsets": "is met only by the perfect weather model 2 days ahead, and there only by the seed-mean set (no single training meets it)" if go_onsets == ["lstm_wqf l2 csi", "lstm_wqf l2 far_cap"] and not single_go_wqf else "GO LIST CHANGED",
    "GO all days, realistic": "it is met by " + "; ".join(f"{label.get(n, n)} {k} day{'s' if k > 1 else ''} ahead at its {'CSI' if rule == 'csi' else '30%-false'} trigger ({pc(T(n, rule, k, 'all')['pod'])} warned, {pc(T(n, rule, k, 'all')['far'])} false)" for n, k, rule in go_all_real),
    "seed ranges": f"Single networks (the five seeds of the rain-to-midnight set, each at the set's trigger): {rg(s2['csi']['seed_range']['test_at_set_trigger']['pod'])} warned with {rg(s2['csi']['seed_range']['test_at_set_trigger']['far'])} false at the CSI level, "
                   f"{rg(s2['far_cap']['seed_range']['test_at_set_trigger']['pod'])} with {rg(s2['far_cap']['seed_range']['test_at_set_trigger']['far'])} false at the 30%-false level",
    "bootstrap (the record's procedure repeated, 2000 draws, seed 0)": f"(30-day blocks, 2000 draws) puts the seed-mean set's share at {rg(bt['csi']['record_procedure_repeated']['pod_ci95'])} and {rg(bt['far_cap']['record_procedure_repeated']['pod_ci95'])} at the two levels",
    "bootstrap (my 300 draws, other seed), same rounding": f"share at {rg(bt['csi']['mine']['pod_ci95'])} and {rg(bt['far_cap']['mine']['pod_ci95'])} at the two levels",
    "stacker": f"warns of {pc(st('lstm_wqa')['pod'])} at {pc(st('lstm_wqa')['far'])} false for the rain-to-midnight set and {pc(st('lstm_wqaF_sub')['pod'])} at {pc(st('lstm_wqaF_sub')['far'])} false for the forecast set (against {pc(T('lstm_wqaF_sub', 'far_cap')['pod'])} at its 30%-false trigger)",
    "any lead": f"Warned by any lead (1, 2 or 3 days before) with the CSI triggers: {pc(al['any_lead']['pod'])} of the onsets at {pc(al['any_lead']['far'])} false; {al['lead1_misses_warned_two_or_three_days_before']} of the {al['lead1_misses']} lead-1 misses of the rain-to-midnight set were warned two or three days before",
    "not reached": "No trigger of the grid kept the validation false share at or below 30% for " + ", ".join(W1["far_cap_not_reached"]) + ": the lowest-false trigger is reported there",
    "GO rule": "the GO rule for the onsets (at least 70% of the onsets warned and at most 30% of the onset warnings false)",
}
out["clauses"] = {k: dict(text=v, found=v in of) for k, v in clauses.items()}
check("every clause of outcome_full rebuilt from my numbers is found in the text", all(v["found"] for v in out["clauses"].values()), [k for k, v in out["clauses"].items() if not v["found"]])
check("brief: 48% at the CSI level and 26% at the 30%-false level", pc(T("lstm_wqa", "csi")["pod"]) == "48%" and pc(T("lstm_wqa", "far_cap")["pod"]) == "26%")
nums = re.findall(r"\d+%", of)
check("outcome_full = outcome + ' ' + outcome_caveat", of == o + " " + cav)
out["realistic_all_days_go"] = [f"{n} l{k} {rule}" for n, k, rule in go_all_real]
out["onset_go"] = go_onsets
out["single_networks_go_at_set_trigger_anywhere"] = W2["single_networks_meeting_go_at_set_trigger"]

# (d) words, disclosure, status, guard ---------------------------------------------------------------------------------------------------------------------
wqa_pc = pc(T("lstm_wqa", "csi")["pod"])
check("AORC wording 'an analysis made after the fact' beside the AORC set; never 'observed rain'", "AORC rain to midnight (an analysis made after the fact)" in o and "observed rain" not in json.dumps(R))
check(f"hindsight wording beside the rain-to-midnight share ({wqa_pc})", all(of[m.start(): m.start() + 90].find("hindsight at most about 3 points") >= 0 for m in re.finditer(re.escape(wqa_pc) + " of the", of)))
check("upper-bound wording beside both perfect sets", "(both an upper bound that cannot be had in practice)" in o)
check("decision-layer wording", "a decision layer re-arranges the trade-off and does not create skill" in o)
check("satellite-era caveat", "comparable only inside that split" in cav)
check("seed count in outcome_caveat = the seed files of lstm_wqa (five)", "the five seeds of the rain-to-midnight set" in cav and R["sets"]["lstm_wqa"]["n_seeds"] == 5)
ch = " ".join(R["choices_made_after_looking_at_test_scores"])
check("choices_made_after_looking_at_test_scores discloses that seeds 3-4 of the stored sets were added on 30 Oct 2026 after the test scores were known",
      bool(re.search(r"seeds? 3 and 4|seeds 3-4|30 October 2026", ch)), R["choices_made_after_looking_at_test_scores"])
check("status says NOT YET REVIEWED and preliminary, with the 18 files and the seed counts", R["status"].startswith("NOT YET REVIEWED") and "preliminary" in R["status"] and "18 of the forecast files" in R["status"])
check("figure written after the record (same run)", out["files_now"]["figures/warning_policy.png"]["modified"] >= out["files_now"]["results/warning_policy.json"]["modified"][:16])
check("the preliminary diagnostic (rewritten 12:50 with five seeds) has the experiment's triggers, cuts and test tables", not jread("w6_text_go.json")["diagnostic_vs_experiment"].get("differences"))
# the guard of code/20_warning_policy.py after round_2 is appended: prev_covered needs the reviewed record's sha256 in the TOP-LEVEL files_reviewed or a previous run covered
fr_top = ((REV.get("files_reviewed") or {}).get("results/warning_policy.json") or {}).get("sha256")
prev_cov = scr["results/warning_policy.json"] == fr_top or bool(ir.get("covers_this_run", True))
src = (PROJ / "code/20_warning_policy.py").read_text(encoding="utf-8")
out["guard_after_round_2"] = dict(top_level_files_reviewed_sha=fr_top, current_file_sha=scr["results/warning_policy.json"], prev_covered_on_next_run=prev_cov,
                                  script_reads_round_2="round_2" in src, review_line_reads=("verdict", "findings (top level, round 1)"))
check("code/20_warning_policy.py can register a later review round (reads round_2)", "round_2" in src, out["guard_after_round_2"])
out["n_values_compared"] = W1["n_compared"] + W2["n_compared"] + W3["n_compared"] + W5["n_compared"] + W4["record_procedure_repeated"]["n_compared"]
out["n_mismatches"] = len(W1["mismatches"]) + len(W2["mismatches"]) + len(W3["mismatches"]) + len(W5["mismatches"]) + len(W4["record_procedure_repeated"]["mismatches"])
jsave("w7_round2.json", out)
print("values compared (w1, w2, w3, w5, w4 record procedure):", out["n_values_compared"], "mismatches:", out["n_mismatches"])
