"""Check (g), part 3: what make_handin_zip.sh would pack now (a dry run: the zip is NOT made, nothing is written).

The script's own patterns (ALWAYS and SKIP for `light` and for `full`) are read from make_handin_zip.sh and applied, with Info-ZIP's rule that `*` also matches `/`, to every file under the project folder;
the sizes (uncompressed) are totalled per top-level folder, and the files the E3 work adds are named. Also: whether `17_more_basins.py --rescore` could run from the unpacked `full` archive
(which files it reads that the archive would hold or skip), and what happens in outline_overlap() when the outline folder is absent.
Run: python m13_handin.py
"""
import re

from m_common import *

sh = (PROJ / "make_handin_zip.sh").read_text()
NAME = PROJ.name
get = lambda var: re.search(rf"^{var}=\((.*?)\)\s*$", sh, re.M).group(1)
split_pats = lambda txt: [p.replace("$NAME", NAME) for p in re.findall(r'"([^"]+)"', txt)]
ALWAYS = split_pats(get("ALWAYS"))
m_full = re.search(r'if \[ "\$KIND" = full \]; then\n\s+SKIP=\((.*?)\)\nelse\n\s+SKIP=\((.*?)\)\nfi', sh, re.S)
SKIP_FULL, SKIP_LIGHT = split_pats(m_full.group(1)), split_pats(m_full.group(2))
NOTES_SKIP = split_pats(re.search(r'SKIP\+=\((.*?)\)', sh).group(1))


def rx(p):
    return re.compile("^" + "".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in p) + "$")


def packed(kind, notes=False):
    skip = [rx(p) for p in ALWAYS + (SKIP_FULL if kind == "full" else SKIP_LIGHT) + ([] if notes else NOTES_SKIP)]
    files = []
    for f in PROJ.rglob("*"):
        if f.is_file() and not f.is_symlink():
            rel = f"{NAME}/{f.relative_to(PROJ)}"
            if not any(r.match(rel) for r in skip):
                files.append((rel, f.stat().st_size))
    return files


res = dict(patterns=dict(ALWAYS=ALWAYS, SKIP_FULL=SKIP_FULL, SKIP_LIGHT=SKIP_LIGHT, SKIP_without_notes=NOTES_SKIP))
for kind in ("light", "full"):
    files = packed(kind)
    tot = sum(s for _, s in files)
    by = {}
    for rel, s in files:
        top = "/".join(rel.split("/")[1:3]) if rel.split("/")[1] in ("data", "results") else rel.split("/")[1]
        by[top] = by.get(top, 0) + s
    res[kind] = dict(n_files=len(files), total_mb=tot / 1e6, by_folder_mb={k: round(v / 1e6, 1) for k, v in sorted(by.items(), key=lambda kv: -kv[1]) if v > 5e5})
    print(f"{kind}: {len(files)} files, {tot / 1e6:.0f} MB uncompressed; folders over 0.5 MB:", res[kind]["by_folder_mb"])
full = dict(packed("full"))
e3 = {k: v for k, v in full.items() if "camels63" in k or "more_basins" in k or "lstm63" in k or "unseen50" in k}
res["full_includes_e3"] = dict(cache63=[k for k in e3 if "camels63" in k], cache63_mb=sum(v for k, v in e3.items() if "camels63" in k) / 1e6, derived_folder_mb=sum(v for k, v in full.items() if "/data/derived/" in k) / 1e6,
                               models_mb=sum(v for k, v in full.items() if "/results/models/" in k) / 1e6, preds_e3_mb=sum(v for k, v in e3.items() if "/results/preds/" in k) / 1e6, review_scripts=[k for k in full if "code/review/more_basins" in k][:3])
print("full archive holds the 63-basin cache:", res["full_includes_e3"]["cache63"], f"{res['full_includes_e3']['cache63_mb']:.1f} MB; data/derived in total {res['full_includes_e3']['derived_folder_mb']:.1f} MB")
big = {}
for rel, s in full.items():
    p = rel.split("/")
    if p[1] == "data" and len(p) > 3:
        big[p[2]] = big.get(p[2], 0) + s
res["full_data_folders_mb"] = {k: round(v / 1e6, 1) for k, v in sorted(big.items(), key=lambda kv: -kv[1])}
print("what `full` would pack from data/ (MB):", res["full_data_folders_mb"])
header = re.search(r"full\s+adds the stored forecasts.*?\(about (\d+) MB\)", sh, re.S)
res["header_claims_full_mb"] = int(header.group(1)) if header else None
print(f"the script's header says full is about {res['header_claims_full_mb']} MB")
src = (PROJ / "code" / "17_more_basins.py").read_text()
res["rescore_inputs"] = dict(reads_outline_shapefile="HCDN_nhru_final_671.shp" in src, outline_folder_in_full_archive=any("basin_set_full_res" in k for k in full),
                             catches_only_importerror_in_outline_overlap=bool(re.search(r"try:\n        import shapefile.*?except ImportError", src, re.S)) and "ShapefileException" not in src,
                             cache63_needed_by_rescore="F.load_data(ids=ids63)" in src, cache63_in_full_archive=any("camels63" in k for k in full), raw_camels_forcing_in_full_archive=any("basin_dataset_public_v1p2" in k for k in full))
print("rescore from the unpacked full archive:", res["rescore_inputs"])
try:
    import shapefile
    shapefile.Reader("/nonexistent/folder/HCDN_nhru_final_671.shp")
    res["missing_outline_raises"] = "no error"
except Exception as e:
    res["missing_outline_raises"] = type(e).__name__
print("shapefile.Reader on a missing file raises:", res["missing_outline_raises"], "(outline_overlap() catches ImportError only)")
res["readme_claims"] = [ln.strip() for ln in (PROJ / "README.md").read_text().splitlines() if "4 hours" in ln or "make_handin_zip" in ln][:4]
save("m13_handin", res)
