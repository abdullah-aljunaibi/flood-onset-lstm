"""AORC review, last step: which outputs of two complete runs of the review scripts differ (the first run, made piece by piece while the scripts were written, against the final run of run_fast_checks.sh and
run_slow_checks.sh), leaf by leaf, so that each difference can be classified as a new field (by design) or a change of a number (float round-off below 1e-12 is not a change).
Usage: a16_compare_outputs.py OLD_DIR [NEW_DIR]   (NEW_DIR defaults to REVIEW_OUT). Only reads; prints; writes a16_compare_outputs.json into REVIEW_OUT. A few seconds."""
import sys; sys.dont_write_bytecode = True
import glob
from a_common import *

OLD = sys.argv[1]
NEW = sys.argv[2] if len(sys.argv) > 2 else OUT
TOL = 1e-12                                                   # float round-off between two runs is not a change

def leaves(x, p=()):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + (str(k),))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): yield from leaves(v, p + (str(i),))
    else:
        yield p, x

def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)

res = {}
names = sorted(os.path.basename(f) for f in glob.glob(f"{OLD}/*.json"))
for n in names:
    if not os.path.exists(f"{NEW}/{n}"):
        print(f"{n}: not in the new folder"); res[n] = "missing"; continue
    a = dict(("/".join(p), v) for p, v in leaves(json.load(open(f"{OLD}/{n}"))))
    b = dict(("/".join(p), v) for p, v in leaves(json.load(open(f"{NEW}/{n}"))))
    added = sorted(set(b) - set(a)); removed = sorted(set(a) - set(b))
    changed_num, changed_txt = [], []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if num(x) and num(y):
            if not (x == y or (x != x and y != y) or abs(x - y) <= TOL * max(1.0, abs(x))): changed_num.append((k, x, y))
        elif x != y: changed_txt.append((k, x, y))
    res[n] = dict(leaves_old=len(a), leaves_new=len(b), added=added, removed=removed, changed_numbers=changed_num, changed_text_or_flags=changed_txt)
    flag = "identical" if not (added or removed or changed_num or changed_txt) else f"added {len(added)}, removed {len(removed)}, numbers changed {len(changed_num)}, text/flags changed {len(changed_txt)}"
    print(f"{n:30s} {len(a):6d} leaves -> {len(b):6d}: {flag}")
    for k in added[:8]: print(f"     + {k[:150]}")
    if len(added) > 8: print(f"     + ... {len(added) - 8} more")
    for k in removed[:8]: print(f"     - {k[:150]}")
    for k, x, y in changed_num[:10]: print(f"     # {k[:110]}: {x!r} -> {y!r}")
    if len(changed_num) > 10: print(f"     # ... {len(changed_num) - 10} more numbers")
    for k, x, y in changed_txt[:8]: print(f"     ~ {k[:90]}: {str(x)[:70]!r} -> {str(y)[:70]!r}")
    if len(changed_txt) > 8: print(f"     ~ ... {len(changed_txt) - 8} more")
jsave("a16_compare_outputs.json", dict(old=OLD, new=NEW, files=res))
