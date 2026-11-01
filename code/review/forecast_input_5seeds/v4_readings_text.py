"""Round 2, check 4: (a) the readings that changed from three to five seeds (the three-seed results file in the backup of 30 October 2026 against the file now);
(b) the clauses of outcome / outcome_caveat rebuilt from the reviewer's own numbers (OUT/v2_compare.json) and searched in the text; (c) stale seed-count wording;
(d) the passes bookkeeping (sums, the stopped pass, the training records of seeds 3-4 against their .pt files, the logs); (e) the SEED_NOTE disclosure and its factual claims.
Output OUT/v4_readings_text.json."""
from v_common import *
import re, platform
import torch
torch.set_num_threads(1)

t0 = time.time()
R = jload(RESULTS)
with open(BACKUP, encoding="utf-8") as f:
    A = json.load(f)
H = jread("v2_compare.json")["headline"]
pc = lambda x: f"{100 * x:.0f}%"
lk = lambda k: f"l{k}"
out = {}

# (a) readings that changed
KEYS = {"reading", "every_seed_above_every_seed", "every_seed_below_every_seed", "single_seeds_all_above", "single_seeds_all_below", "clear_gain", "clear_loss", "met", "met_at_2_and_3_days",
        "verdict", "every_seed_above_wqa", "every_seed_above_the_control", "trigger", "trigger_with", "trigger_without"}
changed = []


def walk(a, b, p):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                walk(a[k], b[k], p + [k])
    elif p and p[-1] in KEYS and a != b and not isinstance(a, (dict, list)):
        changed.append(("/".join(p), a, b))


walk(A, R, [])
words = [c for c in changed if c[0].split("/")[-1] in ("reading", "met", "met_at_2_and_3_days", "clear_gain", "clear_loss") and "single_seeds_plain" not in c[0]]
verd = [c for c in changed if c[0].split("/")[-1] == "verdict" and "single_seeds_plain" not in c[0] and c[0].startswith("verdicts/")]
trig = [c for c in changed if c[0].split("/")[-1] == "trigger" and c[0].startswith("operating_points_by_set/")]
out["readings_changed"] = dict(n_leaves_changed=len(changed), reading_words=words, verdicts=verd, onset_triggers_chosen_on_validation=trig,
                               e2_success_three=dict(met_at_2_and_3_days=A["e2_success"]["met_at_2_and_3_days"], by_lead={k: v["met"] for k, v in A["e2_success"]["by_lead"].items()}),
                               e2_success_five=dict(met_at_2_and_3_days=R["e2_success"]["met_at_2_and_3_days"], by_lead={k: v["met"] for k, v in R["e2_success"]["by_lead"].items()}))
for c in words:
    print("reading changed:", c)
print(f"{len(verd)} verdict words, {len(trig)} validation-chosen onset triggers changed")
texts = {"outcome": R["outcome"], "outcome_caveat": R["outcome_caveat"], "notes": " ".join(R["notes"]), "choices": " ".join(R["choices_made_after_looking_at_test_scores"]), "what_was_not_done": " ".join(R["what_was_not_done"])}
alltext = " ".join(texts.values())
out["text_mentions_a_changed_reading"] = {pat: bool(re.search(pat, alltext, re.I)) for pat in (r"with three seeds", r"three-seed (reading|numbers)", r"(was|were) (a )?clear gain", r"changed (from|with) (three|the two added)", r"seeds 3 and 4 (changed|turned|moved)")}

# (b) the outcome clauses from my numbers
sg = lambda x, d=1: f"{x:+.{d}f}"
w = lambda f: "/".join(f(k) for k in LEADS)
a3 = lambda f, d=1: f"{sg(f(1), d)}, {sg(f(2), d)} and {sg(f(3), d)}"
h = lambda n, k: H[n][lk(k)]
P = lambda n, k: h(n, k)["pod"]
VOP = R["value_of_the_forecast"]["operating_points"]
OPS = R["operating_points_by_set"]["sets"]
clauses = {
    "outcome: onsets fixed level": f"go from {w(lambda k: pc(P('wqa', k)))} to {w(lambda k: pc(P('wqaF_sub', k)))}",
    "outcome: onsets chosen level": f"{w(lambda k: pc(h('wqa', k)['chosen_pod']))} to {w(lambda k: pc(h('wqaF_sub', k)['chosen_pod']))}",
    "outcome: E2a control": f"control, {w(lambda k: pc(P('wqaF_clim', k)))}",
    "outcome: median NSE": f"median NSE from {w(lambda k: format(h('wqa', k)['nse_median'], '.2f'))} to {w(lambda k: format(h('wqaF_sub', k)['nse_median'], '.2f'))} (control {w(lambda k: format(h('wqaF_clim', k)['nse_median'], '.2f'))})",
    "outcome: NSE readings": "by the rule of report section 4.2: " + ", ".join(R["gain_rule"]["wqaF_sub_against_wqa"][lk(k)]["nse"]["reading"] for k in (1, 2)) + " and " + R["gain_rule"]["wqaF_sub_against_wqa"]["l3"]["nse"]["reading"],
    "caveat: points fixed": "the forecast adds " + a3(lambda k: 100 * (P("wqaF_sub", k) - P("wqa", k))) + " points",
    "caveat: NSE differences": a3(lambda k: h("wqaF_sub", k)["nse_median"] - h("wqa", k)["nse_median"], 3) + " of median NSE",
    "caveat: chosen level points": "differences in onsets warned are " + a3(lambda k: 100 * (h("wqaF_sub", k)["chosen_pod"] - h("wqa", k)["chosen_pod"])) + " points",
    "caveat: equal budgets 40": a3(lambda k: VOP[lk(k)]["equal_false_warning_budgets"]["points_at_most_40"]) + " points and (at most 50% false) " + a3(lambda k: VOP[lk(k)]["equal_false_warning_budgets"]["points_at_most_50"]),
    "caveat: average precision": f"goes from {w(lambda k: format(OPS['wqa'][lk(k)]['average_precision'], '.3f'))} to {w(lambda k: format(OPS['wqaF_sub'][lk(k)]['average_precision'], '.3f'))}",
    "caveat: 1-day triggers": f"trigger {h('wqaF_sub', 1)['chosen_trigger']:.2f} against {h('wqa', 1)['chosen_trigger']:.2f}",
    "caveat: 1-day false": f"false warnings among the onset warnings {pc(h('wqaF_sub', 1)['far'])} against {pc(h('wqa', 1)['far'])}",
    "caveat: E2a control margin": "the forecast is " + a3(lambda k: 100 * (P("wqaF_sub", k) - P("wqaF_clim", k))) + " points above it",
    "caveat: E2b control vs base": f"({w(lambda k: pc(P('wqaF_clim_s', k)))} against {w(lambda k: pc(P('wqa_s', k)))})",
    "caveat: E2b over its control": "the forecast is " + a3(lambda k: 100 * (P("wqaF_s", k) - P("wqaF_clim_s", k))) + " points above it",
    "caveat: wqaP": f"lstm_wqaP warns of {w(lambda k: pc(P('wqaP', k)))}",
    "caveat: wqf": f"lstm_wqf: {w(lambda k: pc(P('wqf', k)))}",
    "caveat: E2b sets": f"go from {w(lambda k: pc(P('wqa_s', k)))} (no forecast) to {w(lambda k: pc(P('wqaF_s', k)))} at the fixed level (control {w(lambda k: pc(P('wqaF_clim_s', k)))}",
    "caveat: E2b perfect": f"perfect rain {w(lambda k: pc(P('wqaP_s', k)))}",
    "caveat: E2b points": a3(lambda k: 100 * (P("wqaF_s", k) - P("wqa_s", k))) + " points, against " + a3(lambda k: 100 * (P("wqaF_sub", k) - P("wqa", k))),
    "caveat: E2b chosen points": "the E2b differences are " + a3(lambda k: 100 * (h("wqaF_s", k)["chosen_pod"] - h("wqa_s", k)["chosen_pod"])) + " points",
    "caveat: wqa_s chosen triggers": f"is {w(lambda k: format(h('wqa_s', k)['chosen_trigger'], '.2f'))} at 1/2/3 days, at {w(lambda k: pc(h('wqa_s', k)['chosen_far']))} false warnings",
    "caveat: E2b AP gains": "average precision gains " + a3(lambda k: OPS["wqaF_s"][lk(k)]["average_precision"] - OPS["wqa_s"][lk(k)]["average_precision"], 3),
    "caveat: plain gauge model": f"from {w(lambda k: pc(P('wq', k)))} to {w(lambda k: pc(P('wqF_sub', k)))} (" + a3(lambda k: 100 * (P("wqF_sub", k) - P("wq", k))) + " points)",
    "caveat: raw vs corrected": "same onsets (" + a3(lambda k: 100 * (P("wqaF_sub", k) - P("wqaF_sub_rawpred", k))) + " points",
    "caveat: onsets E2b counts": f"{h('wqa_s', 1)['onsets']}/{h('wqa_s', 2)['onsets']}/{h('wqa_s', 3)['onsets']} onsets",
}
found = {k: (v in R["outcome"] or v in R["outcome_caveat"]) for k, v in clauses.items()}
out["outcome_clauses"] = dict(rebuilt=len(clauses), found=sum(found.values()), missing={k: clauses[k] for k, v in found.items() if not v}, clauses=clauses)
print(f"outcome clauses rebuilt {len(clauses)}, found {sum(found.values())}; missing: {out['outcome_clauses']['missing']}")
out["outcome_full_is_outcome_plus_caveat"] = R["outcome_full"].startswith(R["outcome"]) and R["outcome_caveat"] in R["outcome_full"]

# (c) seed-count wording
def strings(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from strings(v, p + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from strings(v, p + "/" + str(i))
    elif isinstance(x, str):
        yield p, x


pat = re.compile(r"(three|3)[ -](seeds?|trainings?|networks?|single)|seeds? 0-2|three-seed|would settle|only repetition \(|seeds averaged", re.I)
hits = []
for p, s in strings(R):
    for m in pat.finditer(s):
        hits.append(dict(where=p, text=s[max(0, m.start() - 140):m.end() + 140]))
out["seed_wording_hits"] = hits
span = {n: R["seed_ranges"]["sets"][n]["l1"]["nse_median"][1] - R["seed_ranges"]["sets"][n]["l1"]["nse_median"][0] for n in E2}
wide = max(span, key=span.get)
out["what_was_not_done_span_claim"] = dict(widest_set=wide, span=span[wide], text_has=f"{span[wide]:.3f} (lead 1, lstm_{wide})" in texts["what_was_not_done"])

# (d) passes bookkeeping
ps = R["passes"]
tr = R["training"]
new_tr = [t for t in tr if t["seed"] in (3, 4)]
pt_check = []
for t in new_tr:
    z = torch.load(PROJ / t["model_file"], map_location="cpu", weights_only=True)
    pt_check.append(dict(file=t["model_file"], seconds_file=float(z["seconds"]), seconds_record=t["seconds"], best_update_ok=int(z["best_update"]) == t["best_update"], updates_ok=int(z["updates_run"]) == t["updates_run"],
                         seed_ok=z["seed"] == t["seed"], set_ok=z["input_set"] == t["set"], torch=z.get("torch_version")))
    del z
log1 = (PROJ / "results/logs/18_forecast_input_extend_2026-10-30.log")
log2 = (PROJ / "results/logs/18_forecast_input_extend_2026-10-30_rerun.log")
log3 = (PROJ / "results/logs/18_forecast_input_extend_2026-10-30_rerun2.log")
tail = lambda f: f.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:] if f.exists() else None
out["passes"] = dict(
    n_passes=len(ps), kinds=[p["kind"] for p in ps], n_trainings=[p["n_trainings"] for p in ps],
    sum_n_trainings_ok=sum(p["n_trainings"] for p in ps) == R["n_trainings"] == len(tr) == 50,
    sum_training_seconds_ok=abs(sum(p["training_seconds"] for p in ps) - R["training_seconds"]) < 0.06,
    training_records_sum=round(sum(t["seconds"] for t in tr), 2), seeds34_training_seconds=round(sum(t["seconds"] for t in new_tr), 2), passes1_training_seconds=ps[1]["training_seconds"],
    runtime_sum_ok=abs(sum(p["runtime_s"] for p in ps if p["kind"] != "rescore") - R["runtime_s"]) < 0.06,
    stopped_pass=dict(seeds_trained=ps[1].get("seeds_trained"), seed_extension=ps[1].get("seed_extension"), results_file_written=ps[1].get("results_file_written"), runtime_s=ps[1].get("runtime_s"),
                      runtime_from_log_times=(pd.Timestamp("2026-10-30 12:19:47") - pd.Timestamp("2026-10-30 11:55:13")).total_seconds(), script_sha256=ps[1].get("script_sha256")),
    last_pass=dict(what=ps[-1]["what"], n_trainings=ps[-1]["n_trainings"], script_sha256=ps[-1]["script_sha256"], script_now=sha256(SCRIPT), equal=ps[-1]["script_sha256"] == sha256(SCRIPT),
                   integrity_script_sha256_equal=R["integrity"]["script_sha256"] == sha256(SCRIPT)),
    second_attempt_recorded=any("rerun.log" in json.dumps(p) or "KeyError" in json.dumps(p) for p in ps) or ("KeyError" in alltext) or ("second attempt" in alltext.lower()),
    derived_set_files=sorted(set(R["derived_set_files"]["files"].values())),
    integrity=dict(made_in_this_pass=len(R["integrity"]["made_in_this_pass"]), made_in_a_stopped_pass=len(R["integrity"]["made_in_a_stopped_pass"]), only_read=R["integrity"]["only_read_in_this_pass"],
                   hashed=len(R["integrity"]["sha256_of_saved_files"]), compare=R["integrity"]["compared_with_the_earlier_results_file"]),
    pt_records=dict(n=len(pt_check), seconds_match=all(abs(c["seconds_file"] - c["seconds_record"]) < 0.051 for c in pt_check), all_fields_match=all(c["best_update_ok"] and c["updates_ok"] and c["seed_ok"] and c["set_ok"] for c in pt_check),
                    torch_versions=sorted({c["torch"] for c in pt_check})),
    log_tails={"first": tail(log1), "second": tail(log2), "third": tail(log3)})
# hashes of the saved files recorded in integrity against the files now
hs = R["integrity"]["sha256_of_saved_files"]
mism = [f for f, v in hs.items() if sha256(f) != v]
out["passes"]["integrity_hashes_equal_files_now"] = dict(n=len(hs), mismatches=mism)
print("passes:", {k: v for k, v in out["passes"].items() if k not in ("log_tails",)})

# (e) the SEED_NOTE
ch = R["choices_made_after_looking_at_test_scores"]
note = ch[1] if len(ch) > 1 else ""
sched = (PROJ / "code" / "schedule.py").read_text(encoding="utf-8")
import numpy
out["seed_note"] = dict(present=note.startswith("Seeds 3 and 4"), position=1, first_entry_points_to_it="except that seeds 3 and 4 were added later (next entry)" in ch[0],
                        says_after_test_scores="after the test-year scores of seeds 0-2" in note, says_controls_keep_three="keep their three" in note,
                        schedule_py_has_five_trainings="Five trainings per model where the NSE decides" in sched,
                        claimed_versions_vs_env=dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__, claimed=re.findall(r"Python [0-9.]+, torch [^,]+, numpy [0-9.]+", note)),
                        claims_three_seed_numbers_in_passes="in the earlier entries of passes" in note,
                        passes_hold_scores=any(any(k in p for k in ("onset_warned", "summary", "nse_median", "scores")) for p in ps),
                        notes_35_mentions="Seeds 3 and 4 were added on 30 October 2026" in R["notes"][35])
out["seconds"] = round(time.time() - t0, 1)
jsave("v4_readings_text.json", out)
print("seed_note:", out["seed_note"])
print("wording hits:", len(hits))
for x in hits:
    print("  ", x["where"], "::", x["text"][:260].replace("\n", " "))
