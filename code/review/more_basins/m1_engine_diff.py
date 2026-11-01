"""Check (a), part 1: what changed in code/flood.py and code/test_flood.py against the hand-in copy (Project-Flood_handin.zip), statement by statement,
and which other script could be affected (every call of basins( and load_data( in code/, with its arguments).

Run: python m1_engine_diff.py [HANDIN_DIR]    (HANDIN_DIR: a folder holding Project-Flood/code/flood.py etc. unzipped from the hand-in archive;
default: unzip the four files into $REVIEW_OUT/handin)
"""
import ast
import difflib
import re
import subprocess
import sys

from m_common import *

HAND = handin_dir(sys.argv[1] if len(sys.argv) > 1 else None)
ZIP = pathlib.Path("~/AI_Class/Project-Flood_handin.zip").expanduser()
H = HAND / "Project-Flood"
res = {"hand_in_zip": str(ZIP), "hand_in_zip_sha256": sha256(ZIP)}

# 1. files: sha256 now against the hand-in copy
files = {}
for n in ("code/flood.py", "code/test_flood.py", "code/camels.py", "code/common.py", "code/features.py", "run_all.sh", "make_handin_zip.sh"):
    a, b = H / n, PROJ / n
    files[n] = dict(handin_sha256=sha256(a) if a.exists() else None, now_sha256=sha256(b), identical=bool(a.exists() and sha256(a) == sha256(b)))
res["files"] = files
for n, v in files.items():
    print(f"{n:24s} identical to hand-in: {v['identical']}")


# 2. flood.py statement by statement (top-level nodes by name)
def top_nodes(path):
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    out = {}
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            key = f"{type(node).__name__}:{node.name}"
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tg = node.targets[0] if isinstance(node, ast.Assign) else node.target
            key = "assign:" + ast.unparse(tg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            key = "import:" + ast.unparse(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            key = "docstring"
        else:
            key = "stmt:" + ast.unparse(node)[:60]
        out[key] = ast.get_source_segment(src, node)
    return out


old, new = top_nodes(H / "code" / "flood.py"), top_nodes(PROJ / "code" / "flood.py")
changed = sorted(k for k in old if k in new and old[k] != new[k])
added = sorted(k for k in new if k not in old)
removed = sorted(k for k in old if k not in new)
res["flood_py"] = dict(top_level_nodes_old=len(old), top_level_nodes_new=len(new), changed=changed, added=added, removed=removed,
                       unchanged=len([k for k in old if k in new and old[k] == new[k]]))
print("\nflood.py top-level nodes: old", len(old), "new", len(new))
print(" changed:", changed)
print(" added  :", added)
print(" removed:", removed)
# the order of the unchanged nodes is the same (nothing moved)
order_old = [k for k in old if k in new and old[k] == new[k]]
order_new = [k for k in new if k in old and old[k] == new[k]]
res["flood_py"]["unchanged_nodes_in_the_same_order"] = order_old == order_new
print(" unchanged nodes keep their order:", order_old == order_new)

# 3. what the changed functions do on the DEFAULT path: the old body of basins() and load_data() against the new one with ids=None
src_old, src_new = (H / "code" / "flood.py").read_text(), (PROJ / "code" / "flood.py").read_text()
diff = "".join(difflib.unified_diff(src_old.splitlines(True), src_new.splitlines(True), "flood.py (hand-in)", "flood.py (now)"))
(OUT / "flood_py.diff").write_text(diff)
tdiff = "".join(difflib.unified_diff((H / "code" / "test_flood.py").read_text().splitlines(True), (PROJ / "code" / "test_flood.py").read_text().splitlines(True), "test_flood.py (hand-in)", "test_flood.py (now)"))
(OUT / "test_flood_py.diff").write_text(tdiff)
res["diff_lines"] = dict(flood_py_added=sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")),
                         flood_py_removed=sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")),
                         test_flood_added=sum(1 for ln in tdiff.splitlines() if ln.startswith("+") and not ln.startswith("+++")),
                         test_flood_removed=sum(1 for ln in tdiff.splitlines() if ln.startswith("-") and not ln.startswith("---")))
print("\ndiff size:", res["diff_lines"])
removed_flood = [ln for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
print("lines REMOVED from flood.py (all of them):")
for ln in removed_flood:
    print("   ", ln[:200])
removed_test = [ln for ln in tdiff.splitlines() if ln.startswith("-") and not ln.startswith("---")]
print("lines removed from test_flood.py:")
for ln in removed_test:
    print("   ", ln[:200])
res["removed_lines_flood_py"] = removed_flood
res["removed_lines_test_flood_py"] = removed_test

# 4. every call of basins( / load_data( / study_ids( / candidate_ids( in the project's code (also the other review scripts), found with ast (a mention inside a string or a comment is not a call)
NAMES = ("basins", "load_data", "study_ids", "candidate_ids")
MINE = PROJ / "code" / "review" / "more_basins"                                           # this review's own scripts are not part of the code under review
calls, unparsable = [], []
for f in sorted((PROJ / "code").rglob("*.py")):
    if "__pycache__" in f.parts or MINE in f.parents:
        continue
    src = f.read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        unparsable.append(str(f.relative_to(PROJ)))
        continue
    local_defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in NAMES}      # a helper of the file with the same name is not the engine's
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
            if name in NAMES:
                args = ", ".join([ast.unparse(a) for a in node.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords])
                calls.append(dict(file=str(f.relative_to(PROJ)), line=node.lineno, call=name, args=args, local_helper=(name in local_defs and f.name != "flood.py"), text=lines[node.lineno - 1].strip()[:170]))
res["calls"] = calls
res["files_not_parsed"] = unparsable
real = [c for c in calls if not c["local_helper"]]
by_file = {}
for c in real:
    by_file.setdefault(c["file"], []).append(c)
print(f"\ncalls of basins( / load_data( / study_ids( / candidate_ids( : {len(real)} in {len(by_file)} files (ast; {len(calls) - len(real)} calls of a file's own helper of the same name left out; files not parsed: {unparsable})")
for fname, cs in by_file.items():
    kinds = sorted({(c["call"], c["args"]) for c in cs})
    print(f"  {fname}: {kinds}")
with_args = [c for c in real if c["args"] and c["call"] in ("basins",)]
print("\ncalls of basins( with an argument:", [(c["file"], c["line"], c["args"]) for c in with_args])
ld_args = [c for c in real if c["call"] == "load_data" and c["args"]]
print("calls of load_data( with an argument:", [(c["file"], c["line"], c["args"]) for c in ld_args])
res["files_that_call_with_ids"] = sorted({c["file"] for c in real if "ids" in c["args"] and c["call"] in ("basins", "load_data")})
res["calls_with_an_argument_other_than_a_source_name"] = [dict(file=c["file"], line=c["line"], call=c["call"], args=c["args"]) for c in real if c["args"] and not (c["call"] == "load_data" and c["args"] in ("source", "\"nldas\"", "'nldas'", "\"daymet\"", "'daymet'"))]
print("files that pass ids= :", res["files_that_call_with_ids"])
print("calls with any argument:", [(c["file"], c["line"], c["call"], c["args"]) for c in real if c["args"]])

# 5. does any script star-import flood or rebind the new names?
star = [str(f.relative_to(PROJ)) for f in (PROJ / "code").rglob("*.py") if "__pycache__" not in f.parts and MINE not in f.parents and re.search(r"from flood import \*", f.read_text(errors="replace"))]
names_clash = []
for f in (PROJ / "code").rglob("*.py"):
    if "__pycache__" in f.parts or MINE in f.parents:
        continue
    t = f.read_text(errors="replace")
    for nm in ("study_ids", "candidate_ids"):
        if re.search(rf"(def|^)\s*{nm}\b", t, re.M) and f.name != "flood.py":
            names_clash.append((f.name, nm))
res["star_imports_of_flood"] = star
res["new_names_defined_elsewhere"] = names_clash
print("star imports of flood:", star, "| new names defined in another file:", names_clash)

# 6. where else is the cache named
caches = [(str(f.relative_to(PROJ)), i + 1, ln.strip()[:150]) for f in (PROJ / "code").rglob("*.py") if "__pycache__" not in f.parts and MINE not in f.parents
          for i, ln in enumerate(f.read_text(errors="replace").splitlines()) if "camels12" in ln or "camels63" in ln or "derived" in ln and "pkl" in ln]
res["cache_name_mentions"] = caches
print("\nmentions of the cache files in code/:")
for c in caches:
    print("  ", c)
save("m1_engine_diff", res)
