"""w6: the words, the GO scan and the traceability of outcome_full.
(a) The fixed wordings of HANDOFF.md section 7 / the brief: AORC = 'an analysis made after the fact'; 'hindsight at most about 3 points' beside every
    '51% of the ... onsets'; 'an upper bound that cannot be had in practice' for the perfect-rain and perfect-weather sets; 'a decision layer re-arranges the
    trade-off and does not create skill'; the satellite-era caveat; the status not PRELIMINARY; outcome_full = outcome + ' ' + outcome_caveat.
(b) Every number of outcome_full traced to a table of the same file: an explicit map from each number to its table path, and the sentence fragment
    rebuilt from the table value must appear in the text.
(c) The GO scan: every stored test table (onsets, all days, any lead, per seed, stacker) read again with my verdict(); the GO list against go_met_at /
    go_met_by and the outcome's clause.
(d) summary[] against sets[] (the figure reads summary); (e) the preliminary diagnostic against the experiment (same triggers, cuts and test scores?);
(f) the static texts of the script: the figure footer and the provenance say the review is not done, and the script does not read the review record.
Output: OUT/w6_text_go.json."""
from w_common import *
import re

R = record()
o, cav, of = R["outcome"], R["outcome_caveat"], R["outcome_full"]
AORC, HIND, UPPER = "an analysis made after the fact", "hindsight at most about 3 points", "an upper bound that cannot be had in practice"
DECISION, SAT = "a decision layer re-arranges the trade-off and does not create skill", "comparable only inside that split"
out = {}


def get(path):
    v = R
    for p in path.split("."):
        v = v[int(p)] if isinstance(v, list) else v[p]
    return v


# (a) wordings
every_51 = [m.start() for m in re.finditer(r"51% of the", of)]
out["wordings"] = dict(
    aorc_in_outcome=AORC in o, aorc_beside_the_set=("AORC rain to midnight (" + AORC + ")") in o,
    hindsight_beside_every_51=all(of[i:i + 90].find(HIND) >= 0 for i in every_51) and len(every_51) >= 1, n_51_mentions=len(every_51),
    upper_in_outcome=UPPER in o,
    upper_beside_perfect_weather=bool(re.search(r"perfect-weather model \(" + re.escape(UPPER) + r"\)", o)),
    upper_beside_perfect_rain=bool(re.search(r"perfect-rain model[^;]*?\(" + re.escape(UPPER) + r"\)[^;]*?and the perfect-weather", o)) or bool(re.search(r"perfect-rain model keeps \d+% \(" + re.escape(UPPER), o)),
    upper_sets_in_script=["lstm_wqaP", "lstm_wqf", "lstm_wqaP_s"],
    decision_layer_in_outcome=DECISION in o, satellite_caveat_in_caveat=SAT in cav,
    outcome_full_is_outcome_plus_caveat=(of == o + " " + cav), status=R["status"], status_not_preliminary=("PRELIMINARY" not in R["status"].upper()) and ("PRELIMINARY" not in R["what"].upper()),
    not_reached_sentence_present=("No trigger of the grid kept the validation false share at or below 30% for " + ", ".join(R["far_cap_not_reached_on_validation"])) in cav,
    team_check_disclosed=("not yet done" in R["provenance"]["team_check"]), independent_review_line=R["provenance"]["independent_review"])

# (b) traceability: each number in outcome_full and the table it comes from; the fragment rebuilt from the table must occur in the text
A = "sets.lstm_wqa.leads.l1.rules"
P_, F_, SUB = "sets.lstm_wqaP.leads.l1.rules", "sets.lstm_wqf.leads.l1.rules", "sets.lstm_wqaF_sub.leads.l1.rules"
rng = lambda v: f"{pc(v[0])}-{pc(v[1])}"
go_pod, go_far = (float(x) for x in re.findall(r"([0-9.]+)", get("design.go_rule.GO")))
trace = [
    ("51% of the 611 onsets", f"warns of {pc(get(A + '.csi.test.onset.pod'))} of the {get('sets.lstm_wqa.leads.l1.n_test_onsets')} onsets 1 day ahead", [A + ".csi.test.onset.pod", "sets.lstm_wqa.leads.l1.n_test_onsets"]),
    ("50% false (CSI level)", f"with {pc(get(A + '.csi.test.onset.far'))} of its warnings false", [A + ".csi.test.onset.far"]),
    ("25% (30%-false level)", f"it warns of {pc(get(A + '.far_cap.test.onset.pod'))} (test false share {pc(get(A + '.far_cap.test.onset.far'))})", [A + ".far_cap.test.onset.pod", A + ".far_cap.test.onset.far"]),
    ("59% perfect rain", f"the perfect-rain model keeps {pc(get(P_ + '.far_cap.test.onset.pod'))}", [P_ + ".far_cap.test.onset.pod"]),
    ("61% perfect weather", f" {pc(get(F_ + '.far_cap.test.onset.pod'))}; the GO rule", [F_ + ".far_cap.test.onset.pod"]),
    ("GO rule 70% / 30%", f"at least {pc(go_pod)} warned and at most {pc(go_far)} false", ["design.go_rule.GO"]),
    ("GO met: wqf, 2 days", "is met only by the perfect weather model " + ", ".join(f"{k} day{'s' if k > 1 else ''}" for k in get("go_met_by.lstm_wqf.leads")) + " ahead", ["go_met_by.lstm_wqf.leads"]),
    ("no single training", "and there only by the seed-mean set (no single training meets it)" if not get("go_met_by.lstm_wqf.single_networks_meet_it") else "", ["go_met_by.lstm_wqf.single_networks_meet_it"]),
    ("seed ranges CSI level", f"{rng(get(A + '.csi.seed_range.test_at_set_trigger.pod'))} warned with {rng(get(A + '.csi.seed_range.test_at_set_trigger.far'))} false at the CSI level", [A + ".csi.seed_range.test_at_set_trigger.pod", A + ".csi.seed_range.test_at_set_trigger.far"]),
    ("seed ranges 30% level", f"{rng(get(A + '.far_cap.seed_range.test_at_set_trigger.pod'))} with {rng(get(A + '.far_cap.seed_range.test_at_set_trigger.far'))} false at the 30%-false level", [A + ".far_cap.seed_range.test_at_set_trigger.pod", A + ".far_cap.seed_range.test_at_set_trigger.far"]),
    ("bootstrap settings", f"({get('design.bootstrap.block_days')}-day blocks, {get(A + '.csi.bootstrap_test_onsets.draws')} draws)", ["design.bootstrap.block_days", A + ".csi.bootstrap_test_onsets.draws"]),
    ("bootstrap intervals", f"share at {rng(get(A + '.csi.bootstrap_test_onsets.pod_ci95'))} and {rng(get(A + '.far_cap.bootstrap_test_onsets.pod_ci95'))} at the two levels", [A + ".csi.bootstrap_test_onsets.pod_ci95", A + ".far_cap.bootstrap_test_onsets.pod_ci95"]),
    ("stacker wqa", f"warns of {pc(get(A + '.calibrated_stack.test.onset.pod'))} at {pc(get(A + '.calibrated_stack.test.onset.far'))} false for the rain-to-midnight set", [A + ".calibrated_stack.test.onset.pod", A + ".calibrated_stack.test.onset.far"]),
    ("stacker wqaF_sub", f"and {pc(get(SUB + '.calibrated_stack.test.onset.pod'))} at {pc(get(SUB + '.calibrated_stack.test.onset.far'))} false for the forecast set (against {pc(get(SUB + '.far_cap.test.onset.pod'))} at its 30%-false trigger)",
     [SUB + ".calibrated_stack.test.onset.pod", SUB + ".calibrated_stack.test.onset.far", SUB + ".far_cap.test.onset.pod"]),
    ("any lead", f"with the CSI triggers: {pc(get('sets.lstm_wqa.any_lead_test.csi.any_lead.pod'))} of the onsets at {pc(get('sets.lstm_wqa.any_lead_test.csi.any_lead.far'))} false", ["sets.lstm_wqa.any_lead_test.csi.any_lead.pod", "sets.lstm_wqa.any_lead_test.csi.any_lead.far"]),
    ("45 of 298", f"{get('sets.lstm_wqa.any_lead_test.csi.lead1_misses_warned_two_or_three_days_before')} of the {get('sets.lstm_wqa.any_lead_test.csi.lead1_misses')} lead-1 misses", ["sets.lstm_wqa.any_lead_test.csi.lead1_misses_warned_two_or_three_days_before", "sets.lstm_wqa.any_lead_test.csi.lead1_misses"]),
    ("not reached list", "at or below 30% for " + ", ".join(get("far_cap_not_reached_on_validation")), ["far_cap_not_reached_on_validation"]),
]
out["traceability"] = [dict(number=n, fragment_from_table=frag, in_text=(frag in of), tables=paths) for n, frag, paths in trace]
out["traceability_all_found"] = all(t["in_text"] for t in out["traceability"])
# every percentage token of the text has at least one table whose rounded value equals it (a coarse second net)
leaves = []
def walk(v, path):
    if isinstance(v, dict):
        for k, x in v.items():
            walk(x, f"{path}.{k}" if path else k)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            walk(x, f"{path}[{i}]")
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        leaves.append((path, v))
walk({"sets": R["sets"], "design": R["design"], "go_met_by": R["go_met_by"]}, "")
shares = {}
for p, v in leaves:
    if isinstance(v, float) and 0 <= v <= 1:
        shares.setdefault(pc(v), []).append(p)
tokens = sorted(set(re.findall(r"\d+%", of)))
out["percent_tokens_without_any_table_value"] = [t for t in tokens if t not in shares and t not in (pc(go_pod), pc(go_far), "30%")]

# (c) the GO scan over every stored test table
go_onsets, go_all, go_any, go_seed_set, go_seed_own, verdict_disagreements = [], [], [], [], [], []
for n, rec in R["sets"].items():
    for k in LEADS:
        for rule, r in rec["leads"][lk(k)]["rules"].items():
            if "test" not in r:
                continue
            v = verdict(r["test"]["onset"])
            if v != r["verdict_onsets"]:
                verdict_disagreements.append(dict(where=f"{n} {lk(k)} {rule} onsets", stored=r["verdict_onsets"], mine=v))
            if v == "GO":
                go_onsets.append(f"{n} {lk(k)} {rule}")
            if "verdict_all_days" in r:
                va = verdict(r["test"]["all"])
                if va != r["verdict_all_days"]:
                    verdict_disagreements.append(dict(where=f"{n} {lk(k)} {rule} all days", stored=r["verdict_all_days"], mine=va))
                if va == "GO":
                    go_all.append(f"{n} {lk(k)} {rule}")
            for tag, s in r.get("per_seed", {}).items():
                if verdict(dict(s["test_at_set_trigger"], correct_neg=0)) == "GO":
                    go_seed_set.append(f"{n} {lk(k)} {rule} {tag}")
                if verdict(dict(s["test_at_own_trigger"], correct_neg=0)) == "GO":
                    go_seed_own.append(f"{n} {lk(k)} {rule} {tag}")
    for rule, w in rec["any_lead_test"].items():
        va = verdict(w["any_lead"])
        if va != w["verdict_any_lead"]:
            verdict_disagreements.append(dict(where=f"{n} any_lead {rule}", stored=w["verdict_any_lead"], mine=va))
        if va == "GO":
            go_any.append(f"{n} any_lead {rule}")
realistic = {"lstm_wq", "lstm_wqn", "lstm_wqa", "lstm_wqaF_sub", "lstm_wqa_s", "lstm_wqaF_s"}
out["go_scan"] = dict(onsets_go=go_onsets, matches_go_met_at=(sorted(go_onsets) == sorted(R["go_met_at"])), go_met_by=R["go_met_by"],
                      all_days_go=go_all, all_days_go_realistic_sets=[g for g in go_all if g.split()[0] in realistic], any_lead_go=go_any,
                      single_networks_go_at_set_trigger=go_seed_set, single_networks_go_at_own_trigger=go_seed_own, verdict_disagreements=verdict_disagreements,
                      wqf_l2_tables={rule: R["sets"]["lstm_wqf"]["leads"]["l2"]["rules"][rule]["test"]["onset"] for rule in ("csi", "far_cap", "calibrated_stack")},
                      wqf_l2_per_seed_at_set_trigger={rule: {t: s["test_at_set_trigger"] for t, s in R["sets"]["lstm_wqf"]["leads"]["l2"]["rules"][rule]["per_seed"].items()} for rule in RULES},
                      wqf_l1_l3_onsets={lk(k): {rule: dict(pod=R["sets"]["lstm_wqf"]["leads"][lk(k)]["rules"][rule]["test"]["onset"]["pod"], far=R["sets"]["lstm_wqf"]["leads"][lk(k)]["rules"][rule]["test"]["onset"]["far"]) for rule in RULES} for k in (1, 3)})

# (d) summary against sets
bad = []
for k in LEADS:
    for row in R["summary"][lk(k)]:
        r = R["sets"][row["set"]]["leads"][lk(k)]["rules"]
        for rule in RULES:
            for a_, b_ in (("warned", "pod"), ("false", "far"), ("csi", "csi"), ("hits", "hits"), ("false_alarms", "false_alarms")):
                if not close(row[rule][a_], r[rule]["test"]["onset"][b_]):
                    bad.append(f"{row['set']} {lk(k)} {rule} {a_}")
            if not close(row[rule]["trigger"], r[rule]["trigger"]) or row[rule]["bootstrap"] != r[rule]["bootstrap_test_onsets"]:
                bad.append(f"{row['set']} {lk(k)} {rule} trigger/bootstrap")
        st = r.get("calibrated_stack", {})
        if "test" in st and not (close(row["stack"]["warned"], st["test"]["onset"]["pod"]) and close(row["stack"]["false"], st["test"]["onset"]["far"])):
            bad.append(f"{row['set']} {lk(k)} stack")
        if row["n_onsets"] != R["sets"][row["set"]]["leads"][lk(k)]["n_test_onsets"]:
            bad.append(f"{row['set']} {lk(k)} n_onsets")
out["summary_vs_sets_mismatches"] = bad

# (e) the preliminary diagnostic against the experiment
try:
    Dg = json.loads((PROJ / "results/diagnostics_2026-10-28/warning_policy_rescore.json").read_text(encoding="utf-8"))
    dd = []
    for n, rec in R["sets"].items():
        for k in LEADS:
            for rule in ("csi", "far_cap", "calibrated_stack"):
                a_, b_ = rec["leads"][lk(k)]["rules"].get(rule, {}), Dg["sets"].get(n, {}).get("leads", {}).get(str(k), {}).get("rules", {}).get(rule, {})
                if "test" not in a_ or "test" not in b_:
                    continue
                same = close(a_.get("trigger", a_.get("cut")), b_.get("trigger", b_.get("cut"))) and all(close(a_["test"]["onset"][c], b_["test"]["onset"][c]) for c in ("hits", "misses", "false_alarms", "pod", "far"))
                if not same:
                    dd.append(f"{n} {lk(k)} {rule}")
    out["diagnostic_vs_experiment"] = dict(differences=dd, note="the same triggers, cuts and test onset tables in results/diagnostics_2026-10-28/warning_policy_rescore.json and results/warning_policy.json" if not dd else "DIFFERENCES")
except Exception as e:
    out["diagnostic_vs_experiment"] = dict(error=repr(e))

# (f) the static texts of the script
src = (PROJ / "code/20_warning_policy.py").read_text(encoding="utf-8")
out["script_texts"] = dict(figure_footer_says_review_not_done=("this file's independent review is not yet done" in src), provenance_says_review_not_done=("not yet done" in R["provenance"]["independent_review"]),
                           script_reads_review_record=("review_warning_policy" in src and "read_text" in src.split("review_warning_policy")[0][-400:]) if "review_warning_policy" in src else False,
                           footer_labels_perfect_rain_as_upper_bound=bool(re.search(r"'perfect rain'[^\n]*\{UPPER\}", src.split("fig.text")[1][:600])) if "fig.text" in src else None,
                           footer_text_perfect_rain=re.findall(r"'perfect rain' gives[^;]*;", src)[:1],
                           grid_statement=R["design"]["rules"]["csi"], grid_in_scripts_06_16_18="0.30 .. 1.50 step 0.05 (code/06_flood_analysis.py GRID, 16/18 LEVELS)", grid_here=f"{R['design']['trigger_grid'][0]} .. {R['design']['trigger_grid'][-1]} step 0.05")
jsave("w6_text_go.json", out)
print(json.dumps(out["wordings"], indent=1))
print("traceability all found:", out["traceability_all_found"], "| not found:", [t["number"] for t in out["traceability"] if not t["in_text"]], "| % tokens without a table value:", out["percent_tokens_without_any_table_value"])
print("GO scan:", json.dumps({k: out["go_scan"][k] for k in ("onsets_go", "matches_go_met_at", "all_days_go", "any_lead_go", "single_networks_go_at_set_trigger", "single_networks_go_at_own_trigger", "verdict_disagreements")}, indent=1))
print("summary vs sets mismatches:", bad, "| diagnostic vs experiment:", out["diagnostic_vs_experiment"], "| script texts:", json.dumps(out["script_texts"], indent=1))
