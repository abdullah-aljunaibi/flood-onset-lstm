"""AORC review, check (h) part 3: runs the project's code/test_flood.py UNCHANGED with flood.RES_OUT redirected to the output folder (REVIEW_OUT), so that nothing is written into the project (the last step of
test_flood.py writes results/fair_test.json), one thread. Afterwards it compares the file it wrote with the project's results/fair_test.json and prints how flood.py and test_flood.py differ from the hand-in
zip of 4 October when E1_ZIP_DIR names its code folder (the line count of the difference and the names of the functions and options that the difference adds). About 15 seconds."""
import os, sys, time, runpy, filecmp, difflib, re
os.environ["FLOOD_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
CODE = str(PROJ / "code")
sys.path.insert(0, CODE)
import flood as F
import tempfile
OUT = Path(os.environ.get("REVIEW_OUT") or Path(tempfile.gettempdir()) / "aorc_review") / "test_flood_out"
OUT.mkdir(parents=True, exist_ok=True)
zd = os.environ.get("E1_ZIP_DIR", "")
facts = {}
if zd and os.path.exists(zd):
    for f in ("flood.py", "test_flood.py"):
        a, b = open(f"{zd}/{f}").read().splitlines(), open(f"{CODE}/{f}").read().splitlines()
        d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
        removed = [l for l in d if l.startswith("-")]; added = [l for l in d if l.startswith("+")]
        defs = [re.sub(r"\(.*", "", l[1:].strip()) for l in added if l[1:].lstrip().startswith("def ")]
        print(f"{f}: {len(added)} lines added, {len(removed)} removed against the zip; new functions {defs}; removed lines (changed in place): " + " | ".join(l[1:].strip()[:70] for l in removed))
        facts[f] = dict(lines_added=len(added), lines_removed=len(removed), added_or_changed_defs=defs, removed_lines=[l[1:].strip()[:120] for l in removed])
    print("the engine files are on disk as of", time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f"{CODE}/flood.py"))), "(flood.py) and", time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(f"{CODE}/test_flood.py"))), "(test_flood.py)")
facts["modified"] = {f: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f"{CODE}/{f}"))) for f in ("flood.py", "test_flood.py")}
F.RES_OUT = OUT                                                   # test_flood.py writes fair_test.json to F.RES_OUT
t0 = time.time(); runpy.run_path(CODE + "/test_flood.py", run_name="__main__"); print("test_flood.py finished in", round(time.time() - t0), "s")
same = filecmp.cmp(OUT / "fair_test.json", PROJ / "results" / "fair_test.json", shallow=False)
print("the fair_test.json written by this run is byte-identical to results/fair_test.json:", same)
import json
facts.update(test_flood_finished=True, seconds=round(time.time() - t0, 1), fair_test_json_byte_identical=bool(same))
(OUT.parent / "a13_test_flood.json").write_text(json.dumps(facts, indent=1))
