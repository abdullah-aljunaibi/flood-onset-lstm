"""Check (a), part 2: every DEFAULT call path of the engine is unchanged.

The hand-in engine (flood.py of Project-Flood_handin.zip) and the engine now in the project are each run in a sandbox (a copy of their code folder, the two small
results files they read, and a link to data/): with the cache folder pointed at an EMPTY temp folder, so both rebuild the 12-basin table from the raw CAMELS files.
Compared: the rebuilt cache files byte for byte (sha256), against each other and against data/derived/camels12_daymet.pkl; the tables with DataFrame.equals and bitwise
(column buffers); basins() frame; thresholds; and the headline score of lstm_wq (median NSE, onsets warned) through flood.evaluate: all must be identical.
Nothing is written into the project (the engines write their cache into the temp folder).

Run: python m2_defaults.py [HANDIN_DIR]
"""
import shutil
import subprocess
import sys
import textwrap

from m_common import *

HAND = handin_dir(sys.argv[1] if len(sys.argv) > 1 else None)
PY = sys.executable
SB = OUT / "sandbox_defaults"
if SB.exists():
    shutil.rmtree(SB)


def make(label, code_src):
    root = SB / label
    (root / "results").mkdir(parents=True)
    shutil.copytree(code_src, root / "code", ignore=shutil.ignore_patterns("__pycache__", "review", "report_sections", "guide_sections"))
    for n in ("data_audit.json", "basin_audit.csv"):
        shutil.copy(PROJ / "results" / n, root / "results" / n)
    (root / "data").symlink_to(PROJ / "data")                       # read-only use: the engine is pointed at an empty cache folder below
    (root / "results" / "preds").mkdir()
    return root


CHILD = textwrap.dedent('''
    import os, sys, json, hashlib, pickle
    os.environ["FLOOD_THREADS"] = "1"; sys.dont_write_bytecode = True
    sys.path.insert(0, sys.argv[1] + "/code")
    import pathlib, numpy as np, pandas as pd, torch
    torch.set_num_threads(1)
    import flood as F
    tmp = pathlib.Path(sys.argv[2]); tmp.mkdir(parents=True, exist_ok=True)
    F.DERIVED = tmp                                               # an empty cache folder: the table is rebuilt from the raw CAMELS files
    assert not list(tmp.glob("*.pkl"))
    D = F.load_data()
    B = F.basins()
    f = tmp / "camels12_daymet.pkl"
    out = dict(cache_written=f.exists(), cache_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), cache_bytes=f.stat().st_size, ids=list(D), n_rows={g: len(d) for g, d in D.items()},
               basins_sha256=hashlib.sha256(pickle.dumps(B)).hexdigest(), basins_index=list(B.index), basins_columns=len(B.columns),
               thresholds=F.thresholds(D))
    # the headline score through the engine: read the stored forecasts from the PROJECT (the sandbox has no preds folder of its own)
    F.PRED = pathlib.Path(sys.argv[3]) / "results" / "preds"
    E = F.evaluate(F.load_preds("lstm_wq"), F.thresholds(D))
    s = E["summary"]["l1"]
    out["headline_lstm_wq"] = dict(nse_median=s["nse_median"], onset_pod=s["onset"]["pod"], onset_hits=s["onset"]["hits"], onset_misses=s["onset"]["misses"], onset_false_alarms=s["onset"]["false_alarms"],
                                   all_pod=s["all"]["pod"], all_far=s["all"]["far"])
    out["evaluate_sha256"] = hashlib.sha256(json.dumps(E, sort_keys=True, default=float).encode()).hexdigest()
    json.dump(out, open(sys.argv[2] + "/child_out.json", "w"))
    pd.to_pickle(B, sys.argv[2] + "/basins.pkl")
''')

res = {}
for label, src in (("old_handin", HAND / "Project-Flood" / "code"), ("new_now", PROJ / "code")):
    root = make(label, src)
    tmp = SB / f"cache_{label}"
    r = subprocess.run([PY, "-c", CHILD, str(root), str(tmp), str(PROJ)], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", FLOOD_THREADS="1"))
    if r.returncode:
        print(label, "FAILED\n", r.stdout[-2000:], r.stderr[-3000:])
        sys.exit(1)
    res[label] = json.loads((tmp / "child_out.json").read_text())
    print(label, "rebuilt the 12-basin cache from raw:", res[label]["cache_bytes"], "bytes, sha256", res[label]["cache_sha256"][:16])

stored = PROJ / "data" / "derived" / "camels12_daymet.pkl"
res["stored_cache"] = dict(path=str(stored.relative_to(PROJ)), sha256=sha256(stored), bytes=stored.stat().st_size, mtime_local=__import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").localtime(stored.stat().st_mtime)))
o, n = res["old_handin"], res["new_now"]
cmp = dict(
    cache_bytes_old_equals_new=o["cache_sha256"] == n["cache_sha256"],
    cache_old_equals_stored=o["cache_sha256"] == res["stored_cache"]["sha256"],
    cache_new_equals_stored=n["cache_sha256"] == res["stored_cache"]["sha256"],
    basins_frame_identical=o["basins_sha256"] == n["basins_sha256"], basins_index_equal=o["basins_index"] == n["basins_index"],
    thresholds_identical=o["thresholds"] == n["thresholds"], headline_identical=o["headline_lstm_wq"] == n["headline_lstm_wq"], evaluate_output_identical=o["evaluate_sha256"] == n["evaluate_sha256"],
    ids_equal=o["ids"] == n["ids"])
# content comparison, independent of the pickle bytes: DataFrame.equals and the raw buffers, old rebuild / new rebuild / the stored file
D_old, D_new, D_st = (pd.read_pickle(SB / "cache_old_handin" / "camels12_daymet.pkl"), pd.read_pickle(SB / "cache_new_now" / "camels12_daymet.pkl"), pd.read_pickle(stored))
bit = lambda A, B_: all(list(A) == list(B_) and A[g].index.equals(B_[g].index) and list(A[g].columns) == list(B_[g].columns) and A[g].dtypes.equals(B_[g].dtypes)
                        and all(A[g][c].values.tobytes() == B_[g][c].values.tobytes() for c in A[g].columns) for g in A)
cmp.update(tables_equal_old_new=all(D_old[g].equals(D_new[g]) for g in D_old), tables_bitwise_old_new=bit(D_old, D_new),
           tables_equal_new_stored=all(D_new[g].equals(D_st[g]) for g in D_new), tables_bitwise_new_stored=bit(D_new, D_st), tables_bitwise_old_stored=bit(D_old, D_st),
           n_basins=len(D_st), rows_per_table=sorted({len(d) for d in D_st.values()}))
B_old, B_new = pd.read_pickle(SB / "cache_old_handin" / "basins.pkl"), pd.read_pickle(SB / "cache_new_now" / "basins.pkl")
cmp["basins_frame_equals"] = bool(B_old.equals(B_new))
res["comparison"] = cmp
res["headline"] = n["headline_lstm_wq"]
print()
for k, v in cmp.items():
    print(f"  {k:36s} {v}")
print("\nheadline lstm_wq (both engines):", n["headline_lstm_wq"])
ok = all(v is True for k, v in cmp.items() if isinstance(v, bool))
print("\nALL DEFAULT-PATH CHECKS EQUAL:", ok)
res["all_equal"] = ok
save("m2_defaults", res)
shutil.rmtree(SB, ignore_errors=True)
sys.exit(0 if ok else 1)
