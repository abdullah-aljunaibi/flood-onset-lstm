"""Check 1: every file listed in results/gis_layers.json 'files' exists with the recorded sha256 and size; no other file sits in the folder; the totals, the hashes of the
two scripts and of the inputs the record says it read. Also writes the before-manifest of results/gis_layers/ (g6_assemble.py compares it with the folder after all checks)."""
import time
import hashlib
from g_common import *

rec = jload(RECORD)
files = rec["files"]


def _bad_constant(x):
    raise ValueError("non-finite number token " + x)


try:                                                                  # strict JSON (no NaN / Infinity tokens): other parsers than Python's must be able to read the record
    json.loads((PROJ / RECORD).read_text(encoding="utf-8"), parse_constant=_bad_constant)
    strict_ok = True
except ValueError as e:
    strict_ok = str(e)
rows, bad = [], []
for f in files:
    p = PROJ / f["file"]
    r = {"file": f["file"], "recorded_bytes": f["bytes"], "recorded_sha256": f["sha256"]}
    if not p.exists():
        r.update(exists=False, ok=False)
        bad.append(f["file"])
    else:
        sz = p.stat().st_size
        h = sha256_file(p)
        r.update(exists=True, bytes=sz, sha256=h, bytes_equal=(sz == f["bytes"]), sha256_equal=(h == f["sha256"]))
        r["ok"] = bool(r["bytes_equal"] and r["sha256_equal"])
        if not r["ok"]:
            bad.append(f["file"])
    rows.append(r)

listed = {Path(f["file"]).name for f in files}
on_disk = sorted(p.name for p in GIS_DIR.iterdir())
extra = [n for n in on_disk if n not in listed]
missing = [n for n in listed if n not in on_disk]
sub = [p.name for p in GIS_DIR.rglob("*") if p.is_dir()]
side = [n for n in on_disk if re.search(r"(-wal|-shm|-journal|\.aux\.xml|\.ovr|\.msk|\.tmp)$", n)]

total = sum(r.get("bytes", 0) for r in rows)
kinds = {"gpkg": 0, "geojson": 0, "csv": 0, "elevation_tif": 0, "landcover_tif": 0, "readme": 0}
for f in files:
    n = f["file"]
    if n.endswith(".gpkg"): kinds["gpkg"] += 1
    elif n.endswith(".geojson"): kinds["geojson"] += 1
    elif n.endswith(".csv"): kinds["csv"] += 1
    elif "/elevation_" in n and n.endswith(".tif"): kinds["elevation_tif"] += 1
    elif "/landcover_" in n and n.endswith(".tif"): kinds["landcover_tif"] += 1
    elif n.endswith("README.md"): kinds["readme"] += 1

# round 2: which files are byte for byte what round 1 reviewed. ROUND1_SHA256 is copied from results/gis_layers.json 'files' as round 1 read it (the export of 2026-10-29 18:25;
# the export was regenerated on 2026-10-30 11:30, so the old record no longer exists in the project).
ROUND1_SHA256 = {
    "basins_12.gpkg": "614215550a8277968cc74269a1ec005c44fbc7cfbf8d576849e1120c10c38213",
    "basins_12.geojson": "7262b0bf31c1ad29659918d7f207461760e13748004fb6702d5db1a5591767bd",
    "gauges_12.geojson": "8b16f44cf6fb206979881cb6e0c837b0d318f4522bcff865944f3ff521696716",
    "landcover_classes.csv": "9e4685c8c110ebf5b60a2051ae89fb0383ae833b076990cd6f38c7f38ed604e9",
    "elevation_02077200.tif": "c3c571240ffd6b8b3f93281a0b733fc79e6de13fc4f4279a129341928cb2aa24",
    "elevation_02108000.tif": "bf4888e4eed3403fb1be6f1c25ecf1cbcbe5e3eb58f8a14151829e96cf687a08",
    "elevation_02128000.tif": "92d440aa107c32e9abb019d42f773cb9457eafb64ddd17578f3fe120f161ff21",
    "elevation_02137727.tif": "4a1b52ee0ef860c7ca0868df292b8a6db94ad4f6b630fdb31295e03718f8cca7",
    "elevation_02221525.tif": "a5dc4452cea2d9c3270c75277b88b5b0103067121935ce5d62f372de3abb7fc5",
    "elevation_02296500.tif": "a1b381220f94c194bf7d0c47b5c3ed68e5b04fde6fae87d36950a83bca456e11",
    "elevation_02298608.tif": "e65c338832811c3e76cda2c6a2b181b825ce76ee9e6106d4e314481f530ae9ea",
    "elevation_02349900.tif": "86457676abafdbc7650ca3f5ac3033ba08bcbe4e028186c66e7fbb527db11c01",
    "elevation_02472000.tif": "458e29f8a67fb18df91a2d0cc6c2c5b390b11827396fd905776cadb6f2ddd528",
    "elevation_03439000.tif": "d9cd942c8d2d4aeb632f330a0e38a2cad218f497e447b90593d4d7a8cbf6b928",
    "elevation_03463300.tif": "2f36ca0761aed419d423163b7933d921d2950ac0a7e3ab77fd03c616748e7ee3",
    "elevation_03473000.tif": "a6e2ad6b78f45b01cebe66fe02bad367c4648cecbba9be5551735d50b4f92ca1",
    "landcover_02077200.tif": "6d37e670850a845c4b2f06eadfa8c032a37e5e8712b68164fddbae10af3a3e62",
    "landcover_02108000.tif": "1308f1d88801b6021799ffe1006f7a97c6eadac2c89e6f8a3a466ea6d441e3f1",
    "landcover_02128000.tif": "ce1525cfa835358e45675fd069764c64082a357aef603585ba3001674cbfd36c",
    "landcover_02137727.tif": "f704f39d98203dbc74fdfdf67d78678a0b07c978fd6b05ff2c1554b79c6b3c7f",
    "landcover_02221525.tif": "264ee14b9ee192ed1be861c9ab782b31ab09aad5f3776053c2b9ec20d4726c2c",
    "landcover_02296500.tif": "27554f158c49217eab47ab04f5a2df795cad33aca1bd5146e9e0cc08ce5ffbe9",
    "landcover_02298608.tif": "c1240c0ccd41115eb08e99a18e63e301873d47cd73541e2ee1ba18023621ba32",
    "landcover_02349900.tif": "20c6d7159ed1484fab52601b7f4315cfb99a2d0746606074c8c2d4242ee10f00",
    "landcover_02472000.tif": "e03e0abc471430cc5f9ba68c490c4f6477ace0bd88cf959f41c4003572c2d177",
    "landcover_03439000.tif": "6d9819c7ce1f0b873cbfc5b985b9542c5321491a43f6ef711a0ad3fe95801ad2",
    "landcover_03463300.tif": "840dae8284c5c5daa1c2311e1811e0a23a8a261735516a0836f8a5f11ffb142f",
    "landcover_03473000.tif": "3a099e71374d48b096280c9763c80ce0a92a0d965e1d43bacc8ab3e038dbd2cf",
    "README.md": "4c3c22ac4bf67c244a721a76b682c3d3565dd9a3239e1acc306510f6cef588f6",
}
now_by_name = {Path(r["file"]).name: r.get("sha256") for r in rows}
since_round1 = {"n_round1_hashes": len(ROUND1_SHA256), "same_as_round_1": sorted(n for n, h in ROUND1_SHA256.items() if now_by_name.get(n) == h),
                "changed_since_round_1": sorted(n for n, h in ROUND1_SHA256.items() if now_by_name.get(n) != h)}

# the review record the script read for its independent_review block: its sha256 is that of the record WITHOUT later rounds (round 1 as it was)
r1_bytes, r1_info = review_record_round1()
ir = rec.get("independent_review") or {}
review_block = {"block_present": bool(ir), "recorded_sha256": ir.get("sha256"), "sha256_of_record_without_later_rounds": hashlib.sha256(r1_bytes).hexdigest(), "record_info": r1_info,
                "equal": ir.get("sha256") == hashlib.sha256(r1_bytes).hexdigest()}

# the scripts named by the record
code = {}
for key, shakey in (("file", "sha256"), ("related", "related_sha256")):
    p = PROJ / rec["code"][key]
    code[rec["code"][key]] = {"recorded": rec["code"][shakey], "now": sha256_file(p) if p.exists() else None}
    code[rec["code"][key]]["equal"] = code[rec["code"][key]]["recorded"] == code[rec["code"][key]]["now"]

# the inputs the record says it read (sizes + hashes now; the big shapefile takes a second)
inp = {}
for rel, d in rec["inputs_read"].items():
    p = PROJ / rel
    if not p.exists():
        inp[rel] = {"exists": False, "equal": False}
        continue
    sz, h = p.stat().st_size, sha256_file(p)
    inp[rel] = {"exists": True, "bytes_equal": sz == d["bytes"], "sha256_equal": h == d["sha256"], "equal": sz == d["bytes"] and h == d["sha256"]}

# the manifest of the folder before the other checks (names, sizes, mtimes, hashes) and of results/ and code/review (names only)
man = {n: {"bytes": (GIS_DIR / n).stat().st_size, "mtime_ns": (GIS_DIR / n).stat().st_mtime_ns, "sha256": sha256_file(GIS_DIR / n)} for n in on_disk}
res_names = sorted(p.name for p in (PROJ / "results").iterdir())
jsave("manifest_before.json", {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gis_layers": man, "results_names": res_names})

out = {
    "record_is_strict_json": strict_ok,
    "n_files_recorded": len(files),
    "n_files_on_disk": len(on_disk),
    "kinds": kinds,
    "all_exist_with_recorded_bytes_and_sha256": not bad and len(rows) == len(files),
    "bad": bad,
    "extra_files_in_folder": extra,
    "missing_files": missing,
    "subfolders": sub,
    "side_files": side,
    "sum_of_file_bytes": total,
    "recorded_total_bytes": rec["total_bytes"],
    "recorded_total_mb": rec["total_mb"],
    "total_bytes_equal": total == rec["total_bytes"],
    "total_bytes_without_readme": total - next(r["bytes"] for r in rows if r["file"].endswith("README.md")),
    "total_mb_recomputed": round(total / 1e6, 2),
    "total_mib_recomputed": round(total / 2**20, 2),
    "since_round_1": since_round1,
    "review_record_block": review_block,
    "code_hashes": code,
    "inputs_read_equal": {"n": len(inp), "n_equal": sum(1 for v in inp.values() if v["equal"]), "not_equal": [k for k, v in inp.items() if not v["equal"]]},
    "rows": rows,
}
jsave("g1_files.json", out)
print("files recorded", len(files), "| on disk", len(on_disk), "| all ok:", out["all_exist_with_recorded_bytes_and_sha256"], "| bad:", bad)
print("extra:", extra, "| missing:", missing, "| side files:", side, "| subfolders:", sub)
print("sum of bytes", total, "recorded", rec["total_bytes"], "equal", out["total_bytes_equal"], "| without README", out["total_bytes_without_readme"],
      "| MB", out["total_mb_recomputed"], "MiB", out["total_mib_recomputed"])
print("kinds", kinds)
print("code hashes equal:", {k: v["equal"] for k, v in code.items()})
print("inputs read equal:", out["inputs_read_equal"])
print("round 1 -> now: same", len(since_round1["same_as_round_1"]), "of", since_round1["n_round1_hashes"], "| changed:", since_round1["changed_since_round_1"])
print("independent_review.sha256 equals the review record without later rounds:", review_block["equal"], review_block["record_info"])
