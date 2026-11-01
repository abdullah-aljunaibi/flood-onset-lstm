"""Extra check (not one of the seven asked for): does a clean rebuild reproduce the export? The reviewer copies code/27_gis_layers.py, code/07_basin_maps.py (only hashed),
the three results/*.json it reads, the review record (round 1 as it was when the script read it) and the data it reads (CAMELS tables, the shapefile, the 14 DEM tiles, the
12 NLCD files) into a scratch project root under OUT/rebuild,
runs the script there with the system Python, and compares the sha256 of every file it writes with (a) results/gis_layers.json 'files' and (b) the exported files.
The README says: the GeoPackage records its write time (hash changes every run); the GeoTIFFs and GeoJSON files are deterministic. Nothing in the project is written.
Skipped with REVIEW_SKIP_REBUILD=1."""
import shutil
import subprocess
import time
from g_common import *

if os.environ.get("REVIEW_SKIP_REBUILD"):
    print("skipped")
    jsave("g7_rebuild.json", {"skipped": True})
    raise SystemExit(0)

rec = jload(RECORD)
root = OUT / "rebuild"
if root.exists():
    shutil.rmtree(root)
for d in ("code", "results", "data/camels_us/basin_set_full_res", "data/dem", "data/landcover"):
    (root / d).mkdir(parents=True, exist_ok=True)


def cp(rel):
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJ / rel, dst)


for rel in ["code/27_gis_layers.py", "code/07_basin_maps.py", "results/data_audit.json", "results/basin_gis.json", "results/data_flow.json",
            "data/camels_us/camels_topo.txt", "data/camels_us/camels_vege.txt", "data/camels_us/readme.txt"]:
    cp(rel)
# since the revision of 2026-10-30 the script also reads the review record (date and verdict, for its disclosure line): the scratch tree gets the record WITHOUT later rounds, i.e. the
# bytes the script read when it ran (round 1); a record with round_2 appended would make the script quote round 1's top-level fields all the same.
r1_bytes, r1_info = review_record_round1()
(root / "results" / "review_gis_layers.json").write_bytes(r1_bytes)
for p in sorted((PROJ / "data/camels_us/basin_set_full_res").glob("HCDN_nhru_final_671.*")):
    cp(p.relative_to(PROJ).as_posix())
for p in sorted((PROJ / "data/dem").glob("*.hgt.gz")):
    cp(p.relative_to(PROJ).as_posix())
for p in sorted((PROJ / "data/landcover").glob("annual_nlcd_2011_*.tif")):
    cp(p.relative_to(PROJ).as_posix())

env = dict(os.environ)
env.pop("FLOOD_QUICK", None)
env["PYTHONUTF8"] = "1"
env["PYTHONDONTWRITEBYTECODE"] = "1"
t0 = time.time()
cp_ = subprocess.run([sys.executable, "-X", "utf8", "-B", str(root / "code/27_gis_layers.py")], cwd=str(root), env=env, capture_output=True, text=True, encoding="utf-8")
dt = time.time() - t0
out = {"returncode": cp_.returncode, "seconds": round(dt, 1), "stdout_tail": cp_.stdout[-1500:], "stderr_tail": cp_.stderr[-1500:]}
if cp_.returncode == 0:
    new_dir = root / "results/gis_layers"
    cmp = []
    for f in rec["files"]:
        name = Path(f["file"]).name
        q = new_dir / name
        row = {"file": name, "recorded_sha256": f["sha256"]}
        if not q.exists():
            row.update(rebuilt=False)
        else:
            h = sha256_file(q)
            row.update(rebuilt=True, rebuilt_sha256=h, equal_to_recorded=(h == f["sha256"]), equal_to_exported=(h == sha256_file(PROJ / f["file"])), bytes_equal=(q.stat().st_size == f["bytes"]))
        cmp.append(row)
    out["files"] = cmp
    out["n_equal_to_recorded"] = sum(1 for r in cmp if r.get("equal_to_recorded"))
    out["not_equal_to_recorded"] = [r["file"] for r in cmp if not r.get("equal_to_recorded")]
    # the one expected difference is the GeoPackage (write time); the README and results/gis_layers.json carry a time stamp too
    j2 = json.loads((root / "results/gis_layers.json").read_text(encoding="utf-8"))
    out["rebuilt_record_check_table_equal"] = j2["check_table"] == rec["check_table"]
    out["rebuilt_independent_review_equal"] = j2.get("independent_review") == rec.get("independent_review") and j2.get("disclosure_independent_review") == rec.get("disclosure_independent_review")
    out["rebuilt_licences_equal"] = j2.get("licences") == rec.get("licences")
    out["review_record_copied_info"] = r1_info
    out["rebuilt_record_verification_status"] = j2["verification"]["status"]
    out["rebuilt_record_files_sha_equal_except"] = [Path(a["file"]).name for a, b in zip(j2["files"], rec["files"]) if a["sha256"] != b["sha256"]]
    # the README of the rebuild differs from the exported one only by its time stamp line?
    r_new = (new_dir / "README.md").read_text(encoding="utf-8").splitlines()
    r_old = (GIS_DIR / "README.md").read_text(encoding="utf-8").splitlines()
    diff_lines = [(n + 1, a[:100], b[:100]) for n, (a, b) in enumerate(zip(r_old, r_new)) if a != b]
    out["readme_lines_differing_count"] = len(diff_lines) + abs(len(r_old) - len(r_new))
    out["readme_lines_differing_first"] = diff_lines[:4]
jsave("g7_rebuild.json", out)
print("returncode", out["returncode"], "seconds", out["seconds"])
if out["returncode"] != 0:
    print(out["stderr_tail"])
else:
    print("rebuilt files equal to the recorded hash:", out["n_equal_to_recorded"], "of", len(out["files"]), "| not equal:", out["not_equal_to_recorded"])
    print("rebuilt check_table equal to the record's:", out["rebuilt_record_check_table_equal"], "| verification:", out["rebuilt_record_verification_status"])
    print("README lines differing:", out["readme_lines_differing_count"], out["readme_lines_differing_first"])
    print("rebuilt independent_review block + line equal to the export's:", out["rebuilt_independent_review_equal"], "| licences equal:", out["rebuilt_licences_equal"], "| review record copied:", out["review_record_copied_info"])
