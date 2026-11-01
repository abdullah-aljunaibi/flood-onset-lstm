"""ROUND 3, check (f): the regression claim 'every number of the first-run results file is unchanged in the current one'. Own comparison, leaf by leaf, of the first-run results/rain_to_midnight.json (the copy named in
E1_FIRST_RUN_JSON, accepted only if its sha256 equals the one round 1 recorded in results/review_rain_to_midnight.json) with the current file; fields added later are not looked at (the loop runs over the leaves of
the first-run file). Lists whose elements carry a 'set' key (the ladder's rungs, the training records) are aligned by that key (a rung was inserted in the middle of the ladder), the lists of notes by content.
NaN in the first-run file (bare NaN tokens) counts as unchanged where the current file has null. Prints what is unchanged, what changed (by top-level key), what is missing. A few seconds."""
import sys; sys.dont_write_bytecode = True
import hashlib, math, collections
from r2_common import *

path = os.environ.get("E1_FIRST_RUN_JSON", "")
if not (path and os.path.exists(path)):
    print("E1_FIRST_RUN_JSON must name a copy of the first-run results/rain_to_midnight.json (round 1 recorded its sha256)"); sys.exit(1)
raw = open(path, "rb").read()
REV = jload("results/review_rain_to_midnight.json")
want = REV["files_reviewed"]["results/rain_to_midnight.json"]["sha256"]
got = hashlib.sha256(raw).hexdigest()
print(f"first-run copy: sha256 {got[:16]}...; the sha256 round 1 recorded for results/rain_to_midnight.json: {want[:16]}...: equal {got == want}")
assert got == want, "this is not the first-run file"
old = json.loads(raw.decode())                                  # Python reads the bare NaN tokens of the first-run file
new = jload("results/rain_to_midnight.json")

def flat(x, p=()):
    if isinstance(x, dict):
        for k, v in x.items(): yield from flat(v, p + (str(k),))
    elif isinstance(x, list):
        keyed = bool(x) and all(isinstance(e, dict) and "set" in e for e in x)
        for i, e in enumerate(x):
            tag = f"[{e['set']}|{e['seed']}]" if keyed and "seed" in e else f"[{e['set']}]" if keyed else f"[{i}]"
            yield from flat(e, p + (tag,))
    else:
        yield p, x
A, B = list(flat(old)), dict(flat(new))
isnum = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
kind = lambda v: "number" if isnum(v) else "bool" if isinstance(v, bool) else "string" if isinstance(v, str) else "null" if v is None else type(v).__name__
new_notes = set(new["notes"]) | set(new["what_was_not_done"])
tally = collections.defaultdict(lambda: collections.Counter())
changed = []
for p, v in A:
    top = p[0]; k = kind(v)
    if top in ("notes", "what_was_not_done"):                    # lists of sentences: found again verbatim, or rewritten
        key = "unchanged" if v in new_notes else "rewritten text"
        tally[top][(k, key)] += 1
        if key != "unchanged": changed.append((p, v, None))
        continue
    if p not in B: tally[top][(k, "missing")] += 1; changed.append((p, v, "MISSING")); continue
    w = B[p]
    if isnum(v) and v != v: key = "unchanged" if (w is None or (isnum(w) and w != w)) else "changed"        # NaN written as null
    elif isnum(v): key = "unchanged" if (isnum(w) and v == w) else "changed"
    else: key = "unchanged" if v == w and type(v) == type(w) else "changed"
    tally[top][(k, key)] += 1
    if key == "changed": changed.append((p, v, w))
n_all = len(A)
by_kind = collections.Counter(kind(v) for _, v in A)
print(f"leaves of the first-run file: {n_all} ({dict(by_kind)}; {sum(1 for _, v in A if isnum(v) and v != v)} of the numbers are NaN)")
tot = collections.Counter()
for top, c in tally.items():
    for (k, key), n in c.items(): tot[(k, key)] += n
print("all leaves by kind and fate:", {f"{k}/{key}": n for (k, key), n in sorted(tot.items())})
num_unch = tot[("number", "unchanged")]; num_all = by_kind["number"]
print(f"NUMBERS: {num_all} in the first-run file; {num_unch} unchanged (equal, or NaN now null), {tot[('number', 'changed')]} changed, {tot[('number', 'missing')]} missing")
print("\nleaves that are not unchanged, by top-level key and kind:")
other = collections.Counter()
for top, c in tally.items():
    for (k, key), n in c.items():
        if key != "unchanged": other[(top, k, key)] += n
for (top, k, key), n in sorted(other.items()): print(f"   {top:28s} {k:7s} {key:15s} {n}")
print("\nexamples:")
shown = collections.Counter()
for p, v, w in changed:
    if shown[p[0]] < 3:
        shown[p[0]] += 1; print(f"   {'.'.join(map(str, p))[:96]}: {str(v)[:70]!r} -> {str(w)[:70]!r}")
jsave("r3f_regression.json", dict(first_run_sha256=got, leaves=n_all, by_kind=dict(by_kind), numbers=num_all, numbers_unchanged=num_unch, numbers_changed=[(".".join(map(str, p)), v, w) for p, v, w in changed if isnum(v)],
                                  not_unchanged_by_top_key={f"{t}|{k}|{key}": n for (t, k, key), n in other.items()}, all_leaves_by_fate={f"{k}/{key}": n for (k, key), n in tot.items()}))
