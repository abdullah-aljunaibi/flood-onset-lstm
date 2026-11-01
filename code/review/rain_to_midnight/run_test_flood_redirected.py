"""Runs the project's code/test_flood.py UNCHANGED with flood.RES_OUT redirected to the output folder (REVIEW_OUT), so that nothing is written into the project (the last step of test_flood.py writes
results/fair_test.json), one thread. Afterwards it compares the file it wrote with the project's results/fair_test.json. About 12 seconds."""
import os, sys, time, runpy, filecmp
os.environ["FLOOD_THREADS"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
CODE = str(PROJ / "code")
sys.path.insert(0, CODE)
import flood as F
import tempfile
OUT = Path(os.environ.get("REVIEW_OUT") or Path(tempfile.gettempdir()) / "rain_to_midnight_review") / "test_flood_out"
OUT.mkdir(parents=True, exist_ok=True)
F.RES_OUT = OUT                                                   # test_flood.py writes fair_test.json to F.RES_OUT
t0 = time.time(); runpy.run_path(CODE + "/test_flood.py", run_name="__main__"); print("test_flood.py finished in", round(time.time() - t0), "s")
same = filecmp.cmp(OUT / "fair_test.json", PROJ / "results" / "fair_test.json", shallow=False)
print("the fair_test.json written by this run is byte-identical to results/fair_test.json:", same)
