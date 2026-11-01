"""Check (a), part 5: code/test_flood.py passes, and its new check (a second basin set loaded with ids= has a cache of its own, leaves the first set's tables
alone, refuses a basin with a gap or a missing day) really catches a broken load_data().

For every case a sandbox is made (a copy of code/ and an empty results/ folder; test_flood.py reads no CAMELS file) and test_flood.py is run UNCHANGED in it with one
thread; only code/flood.py differs between the cases:
  baseline            the engine as it is                                   -> must print "all checks passed" (and write fair_test.json equal to the project's)
  old_test_new_engine the hand-in test_flood.py with the engine as it is    -> must pass (the old checks still hold)
  new_test_old_engine the new test against the hand-in flood.py             -> must fail (no ids argument there)
  mutants of load_data (each one string replacement, verified to apply once), every one must make the run FAIL:
  M1_shared_cache_read   the cache name ignores ids (one file for every set); a second set is read from the first set's file
  M2_shared_cache_write  two replacements: the cache name ignores ids AND a set with ids never reads the cache, so it overwrites the first set's file
  M3_no_day_guard        the new assertion that every table has a row for every day of PERIOD is removed (a basin with a missing day is accepted)
  M4_no_nan_guard        the old assertion that no table has a missing value is removed (a basin with a gap is accepted)
  M5_sorted_order        a set with ids comes back in sorted order instead of the order asked for (cache hit)
  M6_ignores_ids_on_hit  a cache hit returns the whole cached dict instead of the ids asked for
  M7_corrupts_default    after building a set with ids, the default cache camels12_daymet.pkl is overwritten with it
  M8_default_dtype       the default path returns float32 tables (the default call path changes)
Run: python m5_testflood_mutants.py [HANDIN_DIR]     (about 15-60 s per case)
"""
import shutil
import subprocess
import sys
import time

from m_common import *

HAND = handin_dir(sys.argv[1] if len(sys.argv) > 1 else None)
SB = OUT / "sandbox_testflood"
if SB.exists():
    shutil.rmtree(SB)
SB.mkdir(parents=True)
flood_now = (PROJ / "code" / "flood.py").read_text()

MUT = {
    "M1_shared_cache_read": ('f = DERIVED / (f"camels12_{source}.pkl" if ids is None else f"camels{len(ids)}_{source}_{hashlib.sha256(\',\'.join(sorted(ids)).encode()).hexdigest()[:10]}.pkl")',
                             'f = DERIVED / f"camels12_{source}.pkl"'),
    "M2_shared_cache_write": [('f = DERIVED / (f"camels12_{source}.pkl" if ids is None else f"camels{len(ids)}_{source}_{hashlib.sha256(\',\'.join(sorted(ids)).encode()).hexdigest()[:10]}.pkl")',
                               'f = DERIVED / f"camels12_{source}.pkl"'),
                              ('    if f.exists():\n        D = pd.read_pickle(f)\n        return D if ids is None else {g: D[g] for g in ids}',
                               '    if f.exists() and ids is None:\n        D = pd.read_pickle(f)\n        return D')],
    "M3_no_day_guard": ('    assert ids is None or (len(set(ids)) == len(ids) and all(d.index.equals(pd.date_range(*PERIOD)) for d in D.values())), "every basin needs a row for every day of PERIOD"\n', ''),
    "M4_no_nan_guard": ('    assert not any(d.isna().any().any() for d in D.values()), "the selected basins must have complete records"\n', ''),
    "M5_sorted_order": ('return D if ids is None else {g: D[g] for g in ids}', 'return D if ids is None else {g: D[g] for g in sorted(ids)}'),
    "M6_ignores_ids_on_hit": ('return D if ids is None else {g: D[g] for g in ids}', 'return D'),
    "M7_corrupts_default": ('    tmp.replace(f)\n    return D', '    tmp.replace(f)\n    if ids is not None:\n        (DERIVED / "camels12_daymet.pkl").write_bytes(f.read_bytes())\n    return D'),
    "M8_default_dtype": ('    tmp.replace(f)\n    return D', '    tmp.replace(f)\n    return D if ids is not None else {g: d.astype("float32") for g, d in D.items()}'),
}


def sandbox(label, flood_text=None, test_text=None):
    root = SB / label
    (root / "code").mkdir(parents=True)
    (root / "results").mkdir()
    for n in ("common.py", "camels.py", "features.py"):
        shutil.copy(PROJ / "code" / n, root / "code" / n)
    (root / "code" / "flood.py").write_text(flood_text if flood_text is not None else flood_now)
    (root / "code" / "test_flood.py").write_text(test_text if test_text is not None else (PROJ / "code" / "test_flood.py").read_text())
    return root


def run(root):
    t0 = time.time()
    r = subprocess.run([sys.executable, "test_flood.py"], cwd=root / "code", capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", FLOOD_THREADS="1", FLOOD_QUICK=""), timeout=1800)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    err = (r.stderr.strip().splitlines() or [""])[-1][:240]
    return dict(returncode=r.returncode, passed=r.returncode == 0 and tail.strip() == "all checks passed", last_stdout_line=tail, last_stderr_line=err, seconds=round(time.time() - t0, 1))


res = dict(cases={})
cases = [("baseline", None, None, True), ("old_test_new_engine", None, (HAND / "Project-Flood" / "code" / "test_flood.py").read_text(), True),
         ("new_test_old_engine", (HAND / "Project-Flood" / "code" / "flood.py").read_text(), None, False)]
for name, spec in MUT.items():
    text = flood_now
    for old, new in (spec if isinstance(spec, list) else [spec]):
        assert text.count(old) == 1, f"{name}: the text to replace occurs {text.count(old)} times"
        text = text.replace(old, new)
    assert text != flood_now
    cases.append((name, text, None, False))
for label, ftext, ttext, should_pass in cases:
    root = sandbox(label, ftext, ttext)
    r = run(root)
    r["expected"] = "pass" if should_pass else "FAIL"
    r["as_expected"] = (r["passed"] == should_pass)
    res["cases"][label] = r
    print(f"{label:24s} expected {r['expected']:5s} -> {'pass' if r['passed'] else 'FAIL'}  (rc {r['returncode']}, {r['seconds']} s)  {r['last_stderr_line'] if not r['passed'] else r['last_stdout_line']}")
    if label == "baseline":
        mine = json.loads((root / "results" / "fair_test.json").read_text())
        proj = json.loads((PROJ / "results" / "fair_test.json").read_text())
        res["baseline_fair_test_json_equals_project"] = dict(equal=mine == proj, measured_sandbox=mine["measured"], measured_project=proj["measured"], checks_listed_sandbox=len(mine["checks"]), checks_listed_project=len(proj["checks"]))
        print("   fair_test.json of the sandbox run equals the project's:", mine == proj, "| measured:", mine["measured"], "vs project", proj["measured"])
res["all_cases_as_expected"] = all(c["as_expected"] for c in res["cases"].values())
res["n_mutants_caught"] = sum(1 for k, c in res["cases"].items() if k.startswith("M") and not c["passed"])
res["n_mutants"] = len(MUT)
print(f"\nmutants caught: {res['n_mutants_caught']} of {res['n_mutants']}; all cases as expected: {res['all_cases_as_expected']}")
save("m5_testflood_mutants", res)
shutil.rmtree(SB, ignore_errors=True)
sys.exit(0 if res["all_cases_as_expected"] else 1)
