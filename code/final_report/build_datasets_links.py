"""Hand-in item 5: "Datasets and Repository Links" (.docx + .md; the PDF twin is made by LibreOffice afterwards).

Every number and URL is read from the project files at run time: results/final_report_numbers.json (the registry),
results/data_flow.json, results/data_audit.json, data/hourly_flow/usgs_iv_meta.json, code/final_report/references.py
(DOIs), requirements.txt, code/fetch_usgs_iv.py and the file system. Nothing numeric is typed here.
Run from anywhere:  python -X utf8 build_item5_datasets_links.py
"""
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[2]   # the project folder (code/final_report/ is two levels below it)
OUT = ROOT.parent / "Handin_2026-11-02"   # the hand-in folder, next to the project folder
STEM = "5_Datasets_and_Repository_Links_Flood"
FONT = "Times New Roman"
BODY_PT = 11
TABLE_PT = 8

# ----------------------------------------------------------------------------------------------------- sources
config = json.loads((ROOT / "code/final_report/config.json").read_text(encoding="utf-8"))
registry = json.loads((ROOT / "results/final_report_numbers.json").read_text(encoding="utf-8"))
flow = json.loads((ROOT / "results/data_flow.json").read_text(encoding="utf-8"))
audit = json.loads((ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
meta = json.loads((ROOT / "data/hourly_flow/usgs_iv_meta.json").read_text(encoding="utf-8"))
refs_py = (ROOT / "code/final_report/references.py").read_text(encoding="utf-8")
fetch_py = (ROOT / "code/fetch_usgs_iv.py").read_text(encoding="utf-8")
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")


def num(key):
    """A registry value as text (the registry stores {'text': ..., 'source': ...} or a bare value)."""
    v = registry[key]
    return str(v["text"] if isinstance(v, dict) and "text" in v else v)


DS = {d["key"]: d for d in flow["datasets"]}
DOC = {d["key"]: d for d in flow["documentation_read"]}


def doi(key):
    m = re.search(r'_add\("%s".*?doi="([^"]+)"' % re.escape(key), refs_py, re.S)
    if not m:
        raise KeyError(key)
    return m.group(1)


def fmt_bytes(b, gb_decimals=2):
    if b >= 1e9:
        return f"{b / 1e9:.{gb_decimals}f} GB" if b / 1e9 < 10 else f"{b / 1e9:.1f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.1f} MB"
    return f"{b / 1e3:.0f} kB"


def folder(path):
    n, tot = 0, 0
    for r, _, fs in os.walk(ROOT / path):
        for f in fs:
            n += 1
            tot += os.path.getsize(os.path.join(r, f))
    return n, tot


def dates_in(s):
    return re.findall(r"\d{4}-\d{2}-\d{2}", s)


def accessed(key):
    ds_ = dates_in(num(f"ds_{key}_date_downloaded"))
    lo, hi = min(ds_), max(ds_)
    return lo if lo == hi else f"{lo} to {hi}"


def url_head(key):
    """obtained.url_pattern without the explanatory parenthesis."""
    return DS[key]["obtained"]["url_pattern"].split(" (")[0].strip()


def cut(s, sep):
    return s.split(sep)[0].strip()


def grab(pattern, text):
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"{pattern!r} not found in {text[:80]!r}")
    return m.group(1)


NOT_RECORDED = "licence not recorded in the project files"

# ---------------------------------------------------------------------------------------------- the twelfth product
raw_dir = ROOT / "data/hourly_flow/raw"
gauges = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
month_files = sorted(f for g in gauges for f in os.listdir(raw_dir / g) if f.endswith(".csv.gz"))
months = sorted({f[:-7] for f in month_files})
hourly_npz = sorted(p.name for p in (ROOT / "data/hourly_flow").glob("usgs_iv_*.npz"))
hf_n, hf_bytes = folder("data/hourly_flow")
mt = [os.path.getmtime(raw_dir / g / f) for g in gauges for f in os.listdir(raw_dir / g)]
hf_access_lo = dt.date.fromtimestamp(min(mt)).isoformat()
hf_access_hi = dt.date.fromtimestamp(max(mt)).isoformat()
signup = re.search(r"https://api\.waterdata\.usgs\.gov/signup/", fetch_py).group(0)
approved_all = all(g["hourly"]["share_approved"] == 1.0 for g in meta["gauges"].values())
assert len(gauges) == int(num("n_basins")) == len(meta["gauges"]) == len(hourly_npz)
assert all(g["hourly"]["months_on_disk"] == len(months) for g in meta["gauges"].values())

# ------------------------------------------------------------------------------------------------- measured sizes
preds_n, preds_b = folder("results/preds")
models_n, models_b = folder("results/models")
aorc_n, aorc_b = folder("data/aorc")
gefs_n, gefs_b = folder("data/gefs")
derived_n, derived_b = folder("data/derived")
camels_download_b = DS["camels_us"]["obtained"]["bytes_read"]
camels_disk_b = flow["all_information"]["caches"]["data/camels_us"]["bytes_of_files"]
preds_npz = len(list((ROOT / "results/preds").glob("*.npz")))
models_pt = len(list((ROOT / "results/models").glob("*.pt")))
assert str(preds_npz) == num("count_preds_npz") and str(models_pt) == num("count_models_pt"), "registry counts are stale: rerun numbers.py"
assert fmt_bytes(preds_b) == num("size_results_preds") and fmt_bytes(models_b) == num("size_results_models")
kept = {"aorc": aorc_b, "gefs": gefs_b}      # the folders of the bundle, measured now (data_flow.json recorded them on 8 Oct)


def read_kept(key):
    txt = num(f"ds_{key}_bytes_read") + " read"
    k = kept.get(key, DS[key]["obtained"]["bytes_cached"])
    if k != DS[key]["obtained"]["bytes_read"]:
        txt += f", {fmt_bytes(k)} kept"
    return txt


# ------------------------------------------------------------------------------------------------------- the rows
camels_ver = re.findall(r"v\d+\.\d+", num("ds_camels_us_version"))
n_all_camels = f"{audit['n_all_camels']:,}"
onsets_611 = grab(r"of the ([\d,]+) test-year onsets", num("ds_stage4_period_used"))
n_tiles = grab(r"(\d+) tiles cover", num("ds_elevation_tiles_spatial_grid"))
storms = grab(r"^([\d,]+ storms)", num("ds_hurdat2_period_used"))
rows = [
    ("1", f"{num('ds_camels_us_name')} (time series {camels_ver[0]}, attribute tables {camels_ver[1]}, basin outlines) [every model]",
     f"NCAR with the USGS; Newman et al. 2015 (time series, outlines) and Addor et al. 2017 (attributes). {DOC['camels_page']['url']}; "
     f"DOI {doi('camels_ts')} (time series), {doi('camels_attr')} (attributes)",
     f"{num('ds_camels_us_bytes_read')} download ({num('n_basins')} of {n_all_camels} basins used); {num('ds_camels_us_time_step')}, "
     f"{cut(num('ds_camels_us_period_used'), ':')}; accessed {accessed('camels_us')}",
     f"{num('ds_camels_us_login_needed')} (terms text not recorded); its readme asks to cite Newman et al. 2015 and Addor et al. 2017; not re-hosted"),
    ("2", f"{num('ds_aorc_name')} [extra input: E1b, E4b and the perfect-rain upper bound; an analysis made after the fact]",
     f"NOAA NWS Office of Water Prediction; AWS Open Data {DOC['aws_aorc']['url']}; files {url_head('aorc')}",
     f"{read_kept('aorc')}; {num('ds_aorc_time_step')}, {' to '.join(dates_in(num('ds_aorc_period_used'))[:2])}; accessed {accessed('aorc')}",
     f"no login (AWS Open Data, anonymous); {NOT_RECORDED}"),
    ("3", f"{num('ds_gefs_name')} [extra input: E2, E4e]",
     f"NOAA NCEP, NOAA Open Data Dissemination on AWS Open Data {DOC['aws_gefs']['url']}; Guan et al. 2022, DOI {doi('guan2022')}; files {url_head('gefs')}",
     f"{read_kept('gefs')}; {cut(num('ds_gefs_time_step'), ' (')}, one run a day at 00 UTC; runs {' to '.join(dates_in(num('ds_gefs_period_used'))[:2])}; accessed {accessed('gefs')}",
     f"no login (AWS Open Data, free to read); {NOT_RECORDED}"),
    ("4", f"{num('ds_stage4_name')} [check: a real-time rain product on the test years, not a service]",
     f"NOAA NCEP EMC with the NWS River Forecast Centers; Iowa Environmental Mesonet archive {url_head('stage4')}; Lin and Mitchell 2005",
     f"{read_kept('stage4')}; hourly; the issue-day hours of the {onsets_611} test-year onsets only; accessed {accessed('stage4')}",
     f"no login (the NCAR copy shows a sign-in link; not used); {NOT_RECORDED}"),
    ("5", f"{num('ds_era5_name')} [extra input: E1]",
     f"ECMWF ERA5 through the Copernicus Climate Data Store {DOC['era5_cds']['url']}; served by Open-Meteo {DOC['openmeteo']['url']}",
     f"{read_kept('era5')}; {num('ds_era5_time_step')}, {cut(num('ds_era5_period_used'), ':')}; accessed {accessed('era5')}",
     f"no key ({grab(r'\((.*?)\)', num('ds_era5_login_needed'))}); Copernicus/ECMWF {NOT_RECORDED}"),
    ("6", f"{num('ds_merra2_name')} [tested input; clock check]",
     f"NASA GMAO {DOC['merra2']['url']}; served by NASA POWER {DOC['power_sources']['url']}; API {url_head('merra2')}",
     f"{read_kept('merra2')}; {cut(num('ds_merra2_period_used'), ' (')}; accessed {accessed('merra2')}",
     f"no login; {NOT_RECORDED}"),
    ("7", f"{num('ds_imerg_name')} [tested input on the satellite-era split; day-offset check]",
     f"NASA GPM {DOC['imerg']['url']}; served by NASA POWER (parameter IMERG_PRECTOT at the same API)",
     f"{read_kept('imerg')}; {cut(num('ds_imerg_time_step'), ' (')}, {cut(num('ds_imerg_period_used'), ';')}; accessed {accessed('imerg')}",
     f"no login; {NOT_RECORDED}; IMERG release not stated by POWER"),
    ("8", f"{num('ds_elevation_tiles_name')} [map and check only]",
     f"Mapzen/Tilezen Terrain Tiles on AWS Open Data, from NASA SRTM, USGS 3DEP and others {DOC['joerd']['url']}; tiles {url_head('elevation_tiles')}",
     f"{read_kept('elevation_tiles')}, {n_tiles} tiles; static ({'; '.join(num('ds_elevation_tiles_period_used').split('; ')[:2])}); accessed {accessed('elevation_tiles')}",
     f"no login (AWS Open Data); {NOT_RECORDED} (a composite of several sources)"),
    ("9", f"{num('ds_nlcd_name')} [map and check only]",
     f"USGS EROS with the MRLC consortium; DOI {doi('nlcd_annual')}; WCS {url_head('nlcd')}",
     f"{read_kept('nlcd')}; {cut(num('ds_nlcd_time_step'), ';')}, {cut(num('ds_nlcd_period_used'), ' (')}; accessed {accessed('nlcd')}",
     f"no login; {NOT_RECORDED}; collection number not recorded"),
    ("10", f"{num('ds_hurdat2_name')} [check: storm type of the test-year events]",
     f"NOAA National Hurricane Center {DOC['nhc_hurdat']['url']}; file {url_head('hurdat2')}; Landsea and Franklin 2013",
     f"{read_kept('hurdat2')}; {cut(num('ds_hurdat2_time_step'), ' plus')}; {storms} (seasons to {grab(r'seasons to (\d{4})', num('ds_hurdat2_period_used'))}); accessed {accessed('hurdat2')}",
     f"no login; {NOT_RECORDED}"),
    ("11", f"{num('ds_usgs_iv_name')} [check: the clock of the daily flow]",
     f"USGS Water Services {DOC['usgs_iv']['url']} (nwis/iv and nwis/site requests, format=rdb, parameterCd=00060)",
     f"{read_kept('usgs_iv')}; {num('ds_usgs_iv_time_step')}, {num('ds_usgs_iv_spatial_grid')}, {cut(num('ds_usgs_iv_period_used'), ' requested')}; accessed {accessed('usgs_iv')}",
     f"no login; {NOT_RECORDED}"),
    ("12", "USGS 15-minute river flow, full record, reduced to hourly series (Water Data OGC API) [extra input: E4e, the river's state at the issue time]",
     f"US Geological Survey, Water Data OGC API {meta['api']} (parameter code 00060, one request per gauge-month; code/fetch_usgs_iv.py)",
     f"{fmt_bytes(hf_bytes)} on disk: {len(month_files):,} month files ({len(gauges)} gauges x {len(months)} months, {months[0]} to {months[-1]}) and {len(hourly_npz)} hourly files; "
     f"{'approved values' if approved_all else 'approved share per gauge in usgs_iv_meta.json'}; accessed {hf_access_lo} to {hf_access_hi}",
     f"no key needed at this volume (free key: {signup}); USGS data, a US government work in the public domain; licence text not recorded in the project files"),
]
assert len(rows) == int(num("n_products")) + 1

HEADERS = ("#", "Data product [role in the project]", "Origin, citation and URL (copied from results/data_flow.json)",
           "Size, period, accessed", "Licence / access, as recorded")

# -------------------------------------------------------------------------------------------------------- the text
members = "; ".join(f"{m['name']} ({m['role']})" for m in config["members"])
group = num("group_number") if config.get("group_number") is None else f"Group #{config['group_number']}"
today = dt.date.today().strftime("%#d %B %Y") if sys.platform == "win32" else dt.date.today().strftime("%-d %B %Y")
title_line = f"{group} \u2014 {config['course']} \u2014 Final Project Submission"
doc_title = "Item 5: Datasets and Repository Links"
head_lines = [
    f"Project: {config['title']}",
    f"Team: {members}. Instructor: {config['instructor']}.",
    f"Prepared {today} for the hand-in by e-mail {config['submission_deadline'].replace('(by e-mail)', '').strip()}; final presentation {config['final_presentation']}.",
]

py_minor = grab(r"Python (3\.\d+)", requirements)
pandas_pin = grab(r"\npandas==([\d.]+)", requirements)
torch_pin = grab(r"torch==([\d.]+)", requirements)
numpy_pin = grab(r"numpy==([\d.]+)", requirements)
run_lines = [
    ("pip install -r requirements.txt", f"once, Python {py_minor}; torch {torch_pin} and numpy {numpy_pin} are pinned as a pair; CPU only"),
    ("python code/test_flood.py", "the fairness checks on made-up data; needs no data"),
    ("python code/build_report_final.py", "the final report (.docx + PDF) from results/*.json and figures/; needs no data"),
]
fw_who = num("fw_who_did_what")
fw_team = num("fw_team_checks")
verdicts = {num("rev_warning_policy_verdict"), num("rev_within_day_inputs_verdict"), num("rev_flow_state_verdict")}
verdict_txt = ("verdict of each: \u201c" + verdicts.pop() + "\u201d") if len(verdicts) == 1 else "verdicts in results/review_*.json"

s1_intro = (
    f"Table 1 lists the {num('n_products')} data products recorded in results/data_flow.json (what each is, its version, grid, time stamp "
    f"and how it was obtained) and, as a twelfth, the hourly USGS gauge record fetched on {hf_access_lo} to {hf_access_hi} for experiment E4e "
    "(results/flow_state.json, reviewed in results/review_flow_state.json). Sizes are the bytes read from the service and, where different, "
    "the bytes kept under data/. \u201cLicence not recorded in the project files\u201d means that no licence text was captured when the product "
    "was read; each product remains under its provider\u2019s terms, to be confirmed at the source page before anything is published."
)
s1_after = (
    f"All {num('n_products')} recorded products together: {num('gb_read')} read from public services, {num('size_data')} kept under data/ "
    f"(the hourly gauge record included); no payment and no account is recorded for any of them. Versions, grids, clocks and the "
    f"{num('n_unverified_data_facts')} facts the data record could not verify are in Appendix B of the final report."
)

s2_items = [
    (
        "Code repository (GitHub, to be created by the group leader): [REPO URL]. ",
        f"It holds code/ ({num('count_code_py_files')} top-level scripts, the engine flood.py, the reviewers\u2019 check scripts, the document builders), "
        f"results/*.json and *.csv (every number quoted in the report; the registry results/final_report_numbers.json), results/logs, figures/ "
        f"({num('count_figures_png')} PNG files), README.md, requirements.txt, run_all.sh, make_handin_zip.sh and the hand-in documents. "
        f"results/gis_layers/ (the GIS layers of the {num('n_basins')} basins: outlines and gauges as GeoPackage and GeoJSON, clipped elevation and land cover as GeoTIFF; "
        f"{num('gis_layers_files')} files, reviewed in results/review_gis_layers.json: {num('gis_layers_review_verdict')}) is in the repository too. "
        "Not in the repository: data/, results/preds, results/models, results/quick, handoff/.",
    ),
    (
        "Data bundle (GitHub Release or Google Drive): [RELEASE OR DRIVE URL]. ",
        f"Sizes measured on {today}: the stored forecasts results/preds ({num('count_preds_npz')} .npz files, {num('size_results_preds')}); "
        f"the saved networks results/models ({num('count_models_pt')} .pt files, {num('size_results_models')}); the derived inputs data/aorc "
        f"({fmt_bytes(aorc_b)}, AORC hourly basin means, the reference copy), data/gefs ({fmt_bytes(gefs_b)}, GEFSv12 control-member basin means), "
        f"data/hourly_flow ({fmt_bytes(hf_bytes)}, product 12) and data/derived ({fmt_bytes(derived_b)}, the parsed CAMELS daily tables: "
        f"two pickles, pandas {pandas_pin}); and the hand-in zip made by make_handin_zip.sh. Unzip into the repository root; "
        "a SHA256SUMS.txt with the sha256 of every file is to accompany the bundle.",
    ),
    (
        "CAMELS-US is not re-hosted: ",
        f"the download ({fmt_bytes(camels_download_b)} as downloaded, {fmt_bytes(camels_disk_b, 1)} on disk with the unzipped copy) is referenced by its "
        f"official source, {DOC['camels_page']['url']} (DOI {doi('camels_ts')} and {doi('camels_attr')}); data/derived, a parsed copy of it, is shared "
        "for assessment and reproducibility only.",
    ),
]

s3_para1 = "Clone [REPO URL] and run the \u201cRun it\u201d lines of README.md from the repository root:"
s3_para2 = (
    "To check one headline number without training, unzip results/preds and data/derived from the bundle and run the one-line command under "
    "\u201cRun it\u201d in README.md: it prints the median NSE and the share of onsets warned 1 day ahead of the LSTM with the river gauge. "
    "Retraining: the training scripts run one at a time (they take the lock .cpu_lock; never start two); each reads its inputs from the bundle "
    f"(data/aorc, data/gefs, data/hourly_flow) or fetches them again with code/fetch_aorc.py, code/fetch_gefs.py and code/fetch_usgs_iv.py "
    f"({num('ds_aorc_bytes_read')} and {num('ds_gefs_bytes_read')} read for the first two). ./run_all.sh without FLOOD_QUICK=1 retrains everything "
    "from the raw data (about 8 hours on a two-core laptop) and overwrites results/; with FLOOD_QUICK=1 it is a smoke test whose output means nothing."
)
s4 = (
    f"{fw_who} Independent AI reviewer agents checked the shared engine and every experiment script (results/engine_review.json, "
    "results/review_*.json); the three experiments finished on 28 and 29 October 2026 (E4a, the warning operating point; E4b, the within-day "
    "AORC rain inputs; E4e, the river\u2019s state at the issue time from the hourly gauge record) each have their own review record "
    f"(results/review_warning_policy.json, results/review_within_day_inputs.json, results/review_flow_state.json; {verdict_txt}). "
    f"Seeds 3 and 4 were added on {num('seed_extension_date')} to every set of scripts {num('seed_extension_scripts')} ({num('seed_extension_trainings')} trainings, "
    "after the test scores of seeds 0 to 2 were known; disclosed in each results file), and every five-seed number was recomputed by independent reviewers "
    f"({num('review_five_seed_values_compared')} values, {num('review_five_seed_mismatches')} mismatches). "
    f"{fw_team} team.json records {num('team_checks_done')} checks done. A licence for the code and the results is still to be chosen, "
    "and the data licences are to be confirmed at the source pages, before the repository and the bundle are published."
)

# --------------------------------------------------------------------------------------------------------- the .md
md = [f"# {doc_title}", "", f"**{title_line}**", ""]
md += [ln + "  " for ln in head_lines]
md += ["", "## 1. Data products used", "", s1_intro, "",
       "Table 1. The data products, their origin, size and period, and the licence or access condition as recorded.", "",
       "| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
md += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in rows]
md += ["", s1_after, "", "## 2. Repository and data links", ""]
md += [f"- **{lead.strip()}** {rest}" for lead, rest in s2_items]
md += ["", "## 3. How to reproduce", "", s3_para1, "", "```bash"]
width = max(len(c) for c, _ in run_lines) + 3
md += [f"{c.ljust(width)}# {why}" for c, why in run_lines]
md += ["```", "", s3_para2, "", "## 4. Use of AI and status of checks", "", s4, ""]
(OUT / f"{STEM}.md").write_text("\n".join(md), encoding="utf-8")

# ------------------------------------------------------------------------------------------------------- the .docx
doc = Document()
sec = doc.sections[0]
sec.orientation = WD_ORIENT.PORTRAIT
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(1.5)
sec.top_margin = sec.bottom_margin = Cm(1.3)
TEXT_W = 21.0 - 3.0

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(BODY_PT)
rpr = normal.element.get_or_add_rPr()
rfonts = rpr.find(qn("w:rFonts"))
if rfonts is None:
    rfonts = OxmlElement("w:rFonts")
    rpr.append(rfonts)
for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
    rfonts.set(qn(attr), FONT)
normal.paragraph_format.space_after = Pt(2)
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.line_spacing = 1.0


def para(text="", *, bold=False, size=None, align=None, italic=False, space_after=None, space_before=None, keep_next=False):
    p = doc.add_paragraph()
    if text:
        run = p.add_run(text)
        run.bold, run.italic = bold, italic
        run.font.name = FONT
        if size:
            run.font.size = Pt(size)
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if space_before is not None:
        p.paragraph_format.space_before = Pt(space_before)
    if keep_next:
        p.paragraph_format.keep_with_next = True
    return p


para(title_line, bold=True, size=BODY_PT, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=1)
para(doc_title, bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)
for ln in head_lines:
    para(ln, size=10.5, space_after=0)

para("1. Data products used", bold=True, space_after=1, space_before=5, keep_next=True)
para(s1_intro, space_after=2)
para("Table 1. The data products, their origin, size and period, and the licence or access condition as recorded.",
     italic=True, size=10, space_after=1, keep_next=True)

widths_cm = (0.55, 3.15, 7.1, 3.8, 3.4)
assert abs(sum(widths_cm) - TEXT_W) < 1e-6, sum(widths_cm)
table = doc.add_table(rows=1, cols=len(HEADERS))
table.style = doc.styles["Table Grid"]
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False
tblPr = table._tbl.tblPr
layout = OxmlElement("w:tblLayout")
layout.set(qn("w:type"), "fixed")
tblPr.append(layout)
tblW = tblPr.find(qn("w:tblW"))
if tblW is None:
    tblW = OxmlElement("w:tblW")
    tblPr.append(tblW)
tblW.set(qn("w:type"), "dxa")
tblW.set(qn("w:w"), str(int(round(TEXT_W / 2.54 * 1440))))


def fill(cell, text, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.bold = bold
    r.font.name = FONT
    r.font.size = Pt(TABLE_PT)
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for side, w in (("top", 15), ("bottom", 15), ("start", 45), ("end", 45)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(w))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcPr.append(mar)


for i, h in enumerate(HEADERS):
    fill(table.rows[0].cells[i], h, bold=True)
for r in rows:
    cells = table.add_row().cells
    for i, c in enumerate(r):
        fill(cells[i], c)
for i, w in enumerate(widths_cm):              # grid + every cell: Word reads the cells, LibreOffice the grid
    table.columns[i].width = Cm(w)
    for row in table.rows:
        row.cells[i].width = Cm(w)
trPr = table.rows[0]._tr.get_or_add_trPr()     # repeat the header row after a page break
th = OxmlElement("w:tblHeader")
th.set(qn("w:val"), "true")
trPr.append(th)

para(s1_after, space_after=2, space_before=3)

para("2. Repository and data links", bold=True, space_after=1, space_before=5, keep_next=True)
for lead, rest in s2_items:
    p = para(space_after=1)
    p.paragraph_format.left_indent = Cm(0.3)
    r1 = p.add_run(lead)
    r1.bold = True
    r1.font.name = FONT
    r2 = p.add_run(rest)
    r2.font.name = FONT

para("3. How to reproduce", bold=True, space_after=1, space_before=5, keep_next=True)
para(s3_para1, space_after=1, keep_next=True)
for c, why in run_lines:
    p = para(space_after=0, keep_next=True)
    p.paragraph_format.left_indent = Cm(0.6)
    rc = p.add_run(c.ljust(width))
    rc.font.name = "Consolas"
    rc.font.size = Pt(9)
    rc._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
    rw = p.add_run("# " + why)
    rw.font.name = FONT
    rw.font.size = Pt(9)
    rw.italic = True
para(s3_para2, space_after=1, space_before=2)

para("4. Use of AI and status of checks", bold=True, space_after=1, space_before=5, keep_next=True)
para(s4, space_after=0)

docx_path = OUT / f"{STEM}.docx"
doc.save(docx_path)

# ---------------------------------------------------------------------------------------------------------- report
md_words = len((OUT / f"{STEM}.md").read_text(encoding="utf-8").split())
d2 = Document(docx_path)
docx_text = [p.text for p in d2.paragraphs] + [c.text for t in d2.tables for row in t.rows for c in row.cells]
docx_words = len(" ".join(docx_text).split())
print(f"wrote {docx_path} ({docx_words} words incl. table) and {OUT / (STEM + '.md')} ({md_words} words)")
print("twelfth product:", len(gauges), "gauges;", len(month_files), "month files;", months[0], "to", months[-1], ";", fmt_bytes(hf_bytes), "; accessed", hf_access_lo, "to", hf_access_hi)
print("bundle sizes: preds", preds_n, fmt_bytes(preds_b), "| models", models_n, fmt_bytes(models_b), "| aorc", aorc_n, fmt_bytes(aorc_b), "| gefs", gefs_n, fmt_bytes(gefs_b), "| hourly_flow", hf_n, fmt_bytes(hf_bytes), "| derived", derived_n, fmt_bytes(derived_b))
