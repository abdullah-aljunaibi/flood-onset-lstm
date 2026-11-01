"""ROUND 3, check (d)/(h): the remaining statements of the results file that a reader will quote, checked against the review record, the disk and the other documents:
 d1  strict JSON (no NaN token) and the count of null values given in the notes ('97 values ... are not numbers'): my own count of the null leaves, of which structural ones are told apart
 d2  the two onsets_in_words sentences are inside the yardstick note (notes[22]) exactly as the fields hold them
 d3  what_was_not_done[0]: the counts of round 1 and round 2 it quotes equal the review record's counts and hashes
 d4  what_was_not_done[10] says the update of PLAN.md and README.md 'is not made': what PLAN.md and README.md say about E1 now
A few seconds."""
import sys; sys.dont_write_bytecode = True
import re, collections
from r2_common import *

raw = open(f"{PROJ}/results/rain_to_midnight.json").read()
R = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError("non-finite constant " + c)))
REV = jload("results/review_rain_to_midnight.json")
out, bad = {}, []

# d1
def leaves(x, p=()):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, p + (str(i),))
    else: yield p, x
nulls = [".".join(p) for p, v in leaves(R) if v is None]
claim = int(re.search(r"(\d+) values in this file are not numbers", " ".join(R["notes"])).group(1))
by_top = collections.Counter(n.split(".")[0] for n in nulls)
structural = [n for n in nulls if n.split(".")[0] in ("integrity", "quick_mode", "figure_labels", "reload_check") or "compared_with" in n]
print(f"d1  strict JSON: no NaN/Infinity token in the file; null leaves: {len(nulls)} (the note says {claim} values are NaN written as null; the others are structural nulls: {len(nulls) - claim} expected)")
print(f"    nulls by top-level key: {dict(by_top)}; structural-looking: {structural[:8]}")
out["d1"] = dict(null_leaves=len(nulls), claimed_nan_nulls=claim, by_top_key=dict(by_top), structural=structural)

# d2
note22 = [n for n in R["notes"] if n.startswith("The yardstick (rule 2 of the work order; yardstick)")]
ys = R["yardstick"]["lstm_against_simple"]
s_lin, s_rf = ys["lstm_wqh__base_linreg_wqh"]["l1"]["onsets_in_words"], ys["lstm_wqh__base_rf_wqh"]["l1"]["onsets_in_words"]
ok2 = bool(note22) and s_lin in note22[0] and s_rf in note22[0]
print(f"\nd2  the yardstick note holds both onsets_in_words sentences verbatim: {ok2}")
out["d2"] = dict(both_sentences_in_note=ok2)
if not ok2: bad.append("d2")

# d3
t = R["what_was_not_done"][0]
c1, c2 = REV["counts"], REV["round_2"]["counts"]
want1 = (f"round 1 ({c1['checks_listed']} checks listed, {c1['checks_passed']} passed, {c1['findings_not_pass_fail']} findings that are neither, {c1['failed']} failed; {c1['must_fix']} must-fix and {c1['should_fix']} should-fix items, {c1['numerical_discrepancies']} numerical discrepancies)")
want2 = (f"round 2 (2026-10-07; {c2['checks_listed']} checks listed, {c2['checks_passed']} passed, {c2['findings_not_pass_fail']} findings that are neither, {c2['failed']} failed; {c2['must_fix']} must-fix and {c2['should_fix']} should-fix items, {c2['numerical_discrepancies']} numerical discrepancies)")
h1, h2 = REV["files_reviewed"]["code/15_rain_to_midnight.py"]["sha256"][:12], REV["round_2"]["files_reviewed"]["code/15_rain_to_midnight.py"]["sha256"][:12]
now = hashlib.sha256(open(f"{PROJ}/code/15_rain_to_midnight.py", "rb").read()).hexdigest() if (hashlib := __import__("hashlib")) else ""
ok3 = want1 in t and want2 in t and f"sha256 {h1}..." in t and f"sha256 {h2}..." in t and f"This version of the script has sha256 {now[:12]}... (not the file that was reviewed last)" in t
rounds_now = [k for k in REV if k.startswith("round_")]
print(f"\nd3  what_was_not_done[0] quotes round 1 ({c1['checks_listed']}/{c1['checks_passed']}/{c1['findings_not_pass_fail']}/{c1['failed']}; {c1['must_fix']} + {c1['should_fix']}) and round 2 ({c2['checks_listed']}/{c2['checks_passed']}/{c2['findings_not_pass_fail']}/{c2['failed']}; {c2['must_fix']} + {c2['should_fix']}) as the review record counts them, and the hashes {h1} and {h2}: {ok3}; the review record holds {rounds_now} (the sentence is built from rounds 1 and 2 only)")
out["d3"] = dict(equal=ok3, rounds_in_the_record=rounds_now)
if not ok3: bad.append("d3")

# d4
def lines(f, pat):
    return [(i + 1, l.strip()[:230]) for i, l in enumerate(open(f"{PROJ}/{f}", encoding="utf-8").read().splitlines()) if re.search(pat, l)]
print("\nd4  what_was_not_done[10]:", R["what_was_not_done"][10][:260])
for f in ("PLAN.md", "README.md"):
    ls = lines(f, r"rain_to_midnight|[Rr]ain to midnight|review_rain")
    print(f"    {f}: {len(ls)} lines mention E1 / the review:")
    for i, l in ls[:6]: print(f"       {i}: {l}")
out["d4"] = {f: lines(f, r"rain_to_midnight|[Rr]ain to midnight|review_rain") for f in ("PLAN.md", "README.md")}
jsave("r3d_texts.json", out)
