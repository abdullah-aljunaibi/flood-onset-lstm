"""Check 4: the words. (a) outcome_full rebuilt clause by clause from my numbers (OUT/compare.json) and found word for word in the file; (b) the SEED_NOTE disclosure:
present as the last entry of choices_made_after_looking_at_test_scores, and each factual claim in it checked against the files (seed list, settings in the
metadata of every seed file, the versions of the environment, schedule.py, the stored sets' seed counts, where the three-seed numbers remain); (c) no text of
the file still gives a seed count of three for the seven sets of the script. Writes OUT/texts.json."""
import platform
import re
import sys as _sys

from r_common import *

R = jload(PROJ / RESULTS)
OLD = jload(BACKUP / RESULTS)
CP = jread("compare.json")
MS = jread("scores.json")["scores"]
REC = jload(PROJ / RECORD)
ok_rows = {}
pc = lambda x: f"{round(100 * x) + 0}%"
p1 = lambda x: f"{100 * x:.1f}%"
E = lambda n: MS[n] if n.startswith("base_") else MS[f"lstm_{n}"]
pod = lambda n: E(n)["l1"]["onset"]["pod"]
hits = lambda n: E(n)["l1"]["onset"]["hits"]
n_on = E("wqh")["l1"]["onset"]["hits"] + E("wqh")["l1"]["onset"]["misses"]
rg = CP["ranges"]["wqh"]["l1"]["onset_pod"]
e1 = CP["my_e1"]["wqh"]
v09 = CP["my_readings"]["value_of_hours_09_to_24"]["l1"]["onsets"]
ov = CP["my_readings"]["overlap wqh wqh_to09"]
t = E("wqh")["l1"]["onset"]
verdict = "GO" if (t["pod"] >= 0.70 and t["far"] <= 0.30) else "CAUTION" if t["csi"] >= 0.40 else "NOT YET"
ni = {k: CP["my_yardstick"][f"new_input {k} l1"] for k in ("lstm", "ridge regression", "random forest")}
NUMW = {3: "three", 5: "five"}
clauses = [
    f"onsets warned 1 day ahead go from {pc(pod('wq'))} to {pc(pod('wqh'))} (control {pc(pod('wqh_older'))}); the verdict is {verdict}.",
    (f"The NLDAS rain up to the evening gives a higher share ({pc(pod('wqn'))}, {hits('wqn')} hits of {n_on}) than the rain to midnight ({pc(pod('wqh'))}, {hits('wqh')} hits of {n_on})"
     if hits("wqn") > hits("wqh") else "NLDAS clause: direction differs"),
    f"the {NUMW[len(SEEDS['wqh'])]} single trainings with the rain to midnight span {p1(rg[0])} to {p1(rg[1])}",
    f"the control is {'flat' if e1['control_flat'] else 'not flat'} ({pc(pod('wqh_older'))} against {pc(pod('wq'))} with the rain table alone)",
    f"is {'met' if e1['met'] else 'not met'}.",
    f"The control with the same rain over the table's own window is {'flat' if e1['control_to09_flat'] else 'not flat'} ({p1(pod('wqh_to09'))} against {p1(pod('wq'))} with the rain table alone)",
    f"so the gain of {v09['difference_points']:.1f} points comes from the hours after about 09:00",
    f"({ov['warned_only_by_first']} onsets warned only by the rain to midnight against {ov['warned_only_by_second']} only by that control)",
    f"Ridge regression and the random forest given the same input warn of {p1(pod('base_linreg_wqh'))} and {p1(pod('base_rf_wqh'))}",
    f"(the new input adds {ni['ridge regression']['onset_gain_points']:.1f} and {ni['random forest']['onset_gain_points']:.1f} points to them, {ni['lstm']['onset_gain_points']:.1f} to the LSTM)",
]
full = R["outcome_full"]
clause_found = {c: c in full for c in clauses}
outcome_parts = dict(outcome_is_prefix=full.startswith(R["outcome"]), caveat_is_suffix=full.endswith(R["outcome_caveat"]), outcome_plus_caveat=full == R["outcome"] + " " + R["outcome_caveat"])
# ---- (b) SEED_NOTE ----
choices = R["choices_made_after_looking_at_test_scores"]
note = choices[-1]
src = (PROJ / SCRIPT).read_text(encoding="utf-8")
m = re.search(r"SEED_NOTE = \((.*?)\)\s+# appended", src, re.S)
note_in_code = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))) if m else None
cfg = jload(PROJ / "results" / "lstm_all.json")["config"]
meta_ok = {}
for n in OWN:
    metas = {}
    for s in SEEDS[n]:
        with np.load(PROJ / "results" / "preds" / f"lstm_{n}_s{s}.npz", allow_pickle=False) as z:
            metas[s] = json.loads(str(z["meta"]))
    meta_ok[n] = dict(config_equal_lstm_all=all(metas[s]["config"] == cfg for s in metas), dyn_equal_across_seeds=all(metas[s]["dyn"] == metas[0]["dyn"] for s in metas),
                      seed_fields=[metas[s]["seed"] for s in metas], inputs_equal=all(metas[s]["inputs"] == n for s in metas))
sched = (PROJ / "code" / "schedule.py").read_text(encoding="utf-8")
passes_hold_scores = any(any(k in p for k in ("summary_test", "onset", "nse_median", "scores")) for p in R["passes"])
rev_r1_wqh_hits = REC["numbers"]["headline_test_years_trigger_1"]["wqh"]["l1"]["onset_hits"]
claims = {
    "seeds field is [0, 1, 2, 3, 4]": R["seeds"] == [0, 1, 2, 3, 4],
    "SEED_NOTE is the last entry of choices_made_after_looking_at_test_scores": note.startswith("Seeds 3 and 4 of every set of this script were added on 30 October 2026"),
    "the file's SEED_NOTE equals the text in the script": note == note_in_code,
    "the first choices entry says seeds 3-4 were added later": "seeds 3-4 were added later" in choices[0],
    "settings: every seed file's config equals results/lstm_all.json config, same inputs across seeds": all(v["config_equal_lstm_all"] and v["dyn_equal_across_seeds"] and v["inputs_equal"] for v in meta_ok.values()),
    "environment named (Python 3.11.15, torch 2.2.2+cpu, numpy 1.26.4) is the project venv": _sys.version.startswith("3.11.15") and "torch 2.2.2+cpu" in note and "numpy 1.26.4" in note,
    "schedule.py names five trainings where the NSE decides": "Five trainings per model where the NSE decides" in sched,
    "stored control sets keep three seeds (n_seeds 3 in seed_ranges)": all(R["seed_ranges"]["sets"][n]["l1"]["n_seeds"] == 3 for n in LADDER),
    "seven sets of the script have five seeds (n_seeds 5)": all(R["seed_ranges"]["sets"][n][f"l{k}"]["n_seeds"] == 5 for n in OWN for k in LEADS),
    "passes[-2] is the extension that trained seeds 3-4 of the seven sets (14 trainings)": R["passes"][-2].get("seeds_trained") == [3, 4] and R["passes"][-2]["n_trainings"] == 14 and sorted(R["passes"][-2]["sets_trained"]) == sorted(OWN),
    "training rows: 35 (7 sets x 5 seeds); n_trainings 35": len(R["training"]) == 35 and R["n_trainings"] == 35,
    "claim 'the three-seed numbers remain in the independent review records': round 1 holds lstm_wqh l1 hits 183 (the three-seed value)": rev_r1_wqh_hits == OLD["summary_test"]["sets"]["wqh"]["l1"]["onset"]["hits"] == 183,
    "claim 'the three-seed numbers remain ... in the earlier entries of passes': passes hold scores": passes_hold_scores,
}
# ---- (c) three-seed wording ----
three_pat = re.compile(r"\b(three|3)[ -](single )?(seeds?|trainings?|networks?|runs?)\b|three-seed|first three|seeds 0-2|of its 3 trainings", re.I)
hits3 = []


def strings(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v, f"{p}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from strings(v, f"{p}[{i}]")
    elif isinstance(o, str):
        yield p, o


for p, s in strings(R):
    for mm in three_pat.finditer(s):
        hits3.append(dict(where=p, text=s[max(0, mm.start() - 120):mm.end() + 60]))
about_own = [h for h in hits3 if not h["where"].startswith(".yardstick.lstm_against_simple.lstm_wq__") and not h["where"].startswith(".choices_made_after_looking_at_test_scores")]
jsave("texts.json", dict(clauses=clause_found, outcome_parts=outcome_parts, verdict_mine=verdict, seed_note_claims=claims, seed_meta=meta_ok, three_seed_wordings=hits3,
                         three_seed_wordings_about_the_seven_sets=about_own, review_item=R["what_was_not_done"][0]))
print(json.dumps(clause_found, indent=1)); print(outcome_parts)
print(json.dumps(claims, indent=1))
print("three-seed wordings:", len(hits3), "about the seven sets:", about_own)
