"""r3_append: round 3 of the final critic -> the `round_3` block of results/review_documents_2026-10-08.json (also written to $REVIEW_OUT/round_3.json).

Round 3 judged the two hand-in documents of 8 October 2026 in their final builds (report 17:36, slides 17:38, plan 17:39; the critic first read the 17:33-17:35 builds): Flood_Report_3pages.pdf/.html, the five-slide deck
(Flood_Slides_5.pdf/.pptx, slides_notes_5.md, results/slide_plan_5.json) and Presentation_Plan_5slides.pdf/.md - numbers and rules only. The checks are r3_tables.py, r3_check.py, r3_new_numbers.py and r3_pptx_fit.py
(outputs in $REVIEW_OUT), the font sizes of fontsizes.swift and the OCR of the slide pictures (ocr.swift); the findings below were written by the critic after reading the pages at 100 and 200 dpi.

    REVIEW_OUT=... R3_TEXT_DIR=... python code/review/documents_2026-10-08/r3_append.py [--write-into-project]

It takes the record as it stands in the project (any earlier round_3 is dropped), adds round_3, refreshes the script hashes and the dates, and writes the result to $REVIEW_OUT (and, with --write-into-project, into results/).
Run it after d9_assemble_review.py when the record is rebuilt (d9 does not know round 3).
"""
import glob
import hashlib
import re
import time

from dcommon import *
from pptx import Presentation

WRITE = "--write-into-project" in sys.argv
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
stamp = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(Path(p).stat().st_mtime))
T, C, X, NN = (jread(n) for n in ("r3_tables.json", "r3_check.json", "r3_pptx_fit.json", "r3_new_numbers.json"))
TXT = Path(os.environ["R3_TEXT_DIR"])
FS = {n: json.loads((TXT / f"fs_{p}.json").read_text()) for n, p in (("report", "Flood_Report_3pages"), ("slides", "Flood_Slides_5"), ("plan", "Presentation_Plan_5slides"))}
DF = jload("data_flow")
KF = jload("key_figure")
rp = PROJ / "results" / "review_documents_2026-10-08.json"
rec = json.loads(rp.read_text())
prev_files = rec.get("round_3", {}).get("files_judged", {})          # entries of an earlier run: a file that was renamed after it was judged keeps its entry
rec.pop("round_3", None)

# ---- what the check scripts say -------------------------------------------------------------------------------------------------------------------------------------------
groups = {k: (sum(x["ok"] for x in v), len(v)) for k, v in C.items() if isinstance(v, list) and v and "ok" in v[0]}
assert all(a == b for a, b in groups.values()), "a check of r3_check.py is not ok: " + str(groups)
assert T["n_found"] == T["n_rows"] == 15, "a table row was not found as printed"
nums = lambda e: re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?%?", (" ".join(e) if isinstance(e, list) else e).replace("−", "-").replace("–", "-"))
n_table_numbers = sum(len(nums(r["expected"])) for r in T["rows"])
n_new = {k: v["not_in_the_texts_judged_before"] for k, v in NN.items()}
n_new_total = sum(n_new.values())
n_checks_total = sum(b for a, b in groups.values())
n_compared = n_table_numbers + n_new_total

# the builders' counts: the rounds and checks of all review records before this block
blocks = []
for f in sorted(glob.glob(str(PROJ / "results" / "review_*.json"))):
    r = json.loads(open(f).read())
    if f.endswith("review_documents_2026-10-08.json"):
        r.pop("round_3", None)                      # this record's own round 3 (an earlier run of this script); the round_3 of review_rain_to_midnight.json is a real third round of that review
    bl = ([r["counts"]] if isinstance(r.get("counts"), dict) else []) + [v["counts"] for k, v in r.items() if k.startswith("round") and isinstance(v, dict) and isinstance(v.get("counts"), dict)]
    blocks += bl
rounds_before, checks_before = len(blocks), sum(b.get("checks_listed", 0) for b in blocks)
compared_before = sum(b.get("numbers_compared_in_all", 0) for b in blocks)
assert (rounds_before, checks_before, compared_before) == (8, 188, 3156), (rounds_before, checks_before, compared_before)

# ---- the numbers that the new-number script could not find in data_flow.json: traced here ----------------------------------------------------------------------------------
code = PROJ / "code"
own = [p for p in list(code.glob("*.py")) + list((code / "report_sections").glob("*.py")) + list((code / "guide_sections").glob("*.py")) if p.stat().st_size > 0]
rev = [p for p in (code / "review").rglob("*.py") if p.stat().st_size > 0]
lines = lambda ps: sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in ps)
rev_before_r3 = [p for p in rev if not (p.parent == HERE and p.name.startswith("r3_"))]
per = DF["all_information"]["bytes_read"]["per_product"]
tot = sum(per.values()) / 1e9
hurdat = hashlib.sha256(open(next((PROJ / "data" / "hurdat").glob("hurdat2-*.txt")), "rb").read()).hexdigest()
SP = jload("slide_plan_5")
traced = {
    "109 and 12,345 (report p.3)": f"files and lines of code/review/ at the build of 17:36, before this round's scripts: {len(rev_before_r3)} files, {lines(rev_before_r3):,} lines (recounted now without the r3_*.py files)",
    "41,446 (report p.3)": f"own Python files now {len(own)} files, {lines(own):,} lines; the only own file changed after the build of 17:36 is code/build_slides_short.py (17:38), so the printed 41,446 is consistent (the build before it printed 41,429); it cannot be reproduced to the line without the old file",
    "188 (report, slides, notes, plan)": f"the listed checks of all results/review_*.json before this block: {rounds_before} rounds, {checks_before} checks (31 + 18 + 36 + 36 + 67), failed 0, numerical discrepancies 0",
    "48.5": f"GEFS bytes read {per['gefs']:,} = {per['gefs'] / 1e9:.2f} GB (data_flow.json all_information.bytes_read.per_product)",
    "70 (notes, plan)": f"slide 1 = {SP['slides'][0]['seconds']} s (results/slide_plan_5.json)",
    "3,156 (notes)": f"counts.numbers_compared_in_all of this record, {compared_before:,}",
    "75202780 (notes)": f"the first characters of the sha256 of data/hurdat/hurdat2-1851-2025-092326.txt: {hurdat[:16]}...",
}
assert len(rev_before_r3) == 109 and lines(rev_before_r3) == 12345 and hurdat.startswith("75202780ce56c845"), "a traced count differs"

# ---- the layout facts ------------------------------------------------------------------------------------------------------------------------------------------------------------
pr = Presentation(str(PROJ / "Flood_Slides_5.pptx"))
notes_chars = [len(s.notes_slide.notes_text_frame.text) if s.has_notes_slide else 0 for s in pr.slides]
assert len(notes_chars) == 5 and all(notes_chars)
mins = {n: min(p["min"] for p in d["pages"]) for n, d in FS.items()}
rp2, rp1, rp3 = (FS["report"]["pages"][i] for i in (1, 0, 2))
share = lambda p, s: 100 * p["sizes"].get(s, 0) / p["n_chars"]
import io
from PIL import Image
pic = {}
for i, s in enumerate(pr.slides, 1):
    for sh in s.shapes:
        if sh.shape_type == 13:
            im = Image.open(io.BytesIO(sh.image.blob))
            pic[i] = dict(px=im.size[0], shown_pt=sh.width / 12700)
talk_flow = DF["figure"]["renderings"]["talk"]
talk_key = KF["figure"]["renderings"]["talk"]
flow_w_in = 2640 / 219.9894          # figures/data_flow_talk.png: 2640 px at 220 dpi, shown cropped but at full width
key_w_in = pic[3]["px"] / 219.9894
eff_flow = talk_flow["smallest_lettering_pt"] * pic[2]["shown_pt"] / (flow_w_in * 72)
eff_key = talk_key["smallest_lettering_pt"] * pic[3]["shown_pt"] / (key_w_in * 72)
gb = {k: v / 1e9 for k, v in per.items()}
rest = tot - gb["gefs"] - gb["stage4"] - gb["aorc"] - gb["camels_us"]

FILES = ["Flood_Report_3pages.pdf", "Flood_Report_3pages.html", "Flood_Slides_5.pdf", "Flood_Slides_5.pptx", "Flood_Slides_5.html", "slides_notes_5.md", "results/slide_plan_5.json", "Presentation_Plan_5slides.pdf", "Presentation_Plan_5slides.md",
         "results/data_flow.json", "figures/data_flow.png", "figures/data_flow_talk.png", "code/build_report_short.py", "code/build_slides_short.py", "code/build_presentation_plan_short.py", "code/19_data_flow.py"]
PAGES = {"Flood_Report_3pages.pdf": 3, "Flood_Slides_5.pdf": 5, "Presentation_Plan_5slides.pdf": 3}
def entry(n):
    """the file as judged: from the earlier run of this script when the file has changed or been renamed since (the first run is the one that hashed the judged files)"""
    here = PROJ / n
    if not here.exists():
        return prev_files[n]
    now = dict(sha256=sha(here), bytes=here.stat().st_size, modified=stamp(here), **({"pages": PAGES[n]} if n in PAGES else {}))
    if n in prev_files and prev_files[n]["sha256"] != now["sha256"]:
        return dict(prev_files[n], changed_since_judged=f"the file was changed at {now['modified']} (sha256 {now['sha256'][:16]}), after it was judged")
    return prev_files.get(n, now)
files = {n: entry(n) for n in FILES}
n_ = "code/build_presentation_plan_short.py"
if files[n_]["modified"] > "2026-10-08 17:40:00" and "changed_since_judged" not in files[n_]:
    files[n_] = dict(files[n_], changed_since_judged=f"modified at {files[n_]['modified']}, after the plan was judged (the judged plan was built by the version of 17:31, whose hash was not kept); the sha256 here is that of the later version")
renamed = [n for n in FILES if not (PROJ / n).exists()]
rename_note = ""
if renamed:
    new_md = PROJ / "Presentation_Plan.md"
    same_md = new_md.exists() and sha(new_md) == files["Presentation_Plan_5slides.md"]["sha256"]
    new_txt = TXT / "Presentation_Plan_new.txt"
    same_pdf = new_txt.exists() and (TXT / "Presentation_Plan_5slides.txt").read_text().split() == new_txt.read_text().split()
    rename_note = (f" At {stamp(new_md)} the coordinator renamed the plan while this block was being written: {', '.join(renamed)} no longer exist under these names; Presentation_Plan.md is "
                   f"{'byte-identical to the judged Presentation_Plan_5slides.md' if same_md else 'NOT identical to the judged Presentation_Plan_5slides.md (not re-judged)'} and the text of Presentation_Plan.pdf is "
                   f"{'word for word the text of the judged PDF' if same_pdf else 'not compared with the judged PDF'}; the earlier plan of the full deck is now Presentation_Plan_full_deck.* (not judged). The entries of files_judged are those of the files as judged.")

status = "open (round 3, hand-in documents of 17:36-17:39)"
must = [dict(id="M9", severity="must_fix", status=status,
             title="The elevation tiles are called 'SRTM radar flown in February 2000' (report p.1, slide 2 picture and spoken line) against the data record and the plan's own rule: a composite tile set in SRTM format",
             text=("Where: (1) report p.1, the imagery note (a verbatim copy of results/data_flow.json imagery_note, written by code/19_data_flow.py): 'the elevation tiles (SRTM radar flown in February 2000, an active sensor)'; "
                   "(2) slide 2, the picture (figures/data_flow_talk.png, cropped): 'Elevation tiles, 30 m, static, flown 2000'; (3) the notes of slide 2: spoken sentence 11, 'The ready-made satellite products are IMERG rain, radar elevation of February 2000 and a land-cover map of 2011', "
                   "and the imagery note repeated in full under 'If asked'. What the record says (data_flow.json, datasets.elevation_tiles.what_it_is): the provider documents the tile set as a composite - NASA's SRTM is the land source at this resolution, USGS 3DEP (formerly NED) is used in the United States "
                   "at finer levels, and the tiles do not say which source a cell came from - 'The project therefore calls it elevation in SRTM format, not pure SRTM'; corrections_to_the_brief: 'the project's own rule (Presentation_Plan.md, Professor_QA.md) is to call it a composite elevation tile set, not SRTM'; "
                   "the plan itself tells RS/GIS member 2 to 'know the sources as the results file names them: a composite elevation tile set, not SRTM'. The report's product list ('Elevation tiles (SRTM format)') and the notes' dataset entry ('built 2016-2017 from SRTM (flown February 2000) and other sources') "
                   "say it correctly, so the set contradicts itself, and the imagery note is the sentence a professor will quote when asking which satellite images were used. The SRTM facts (radar, February 2000, an active sensor) are those of the tile set's main source, not of the tile set. "
                   "Correction (wording only, no number changes; the imagery note and the picture are written by code/19_data_flow.py, so change them there and rebuild): in the imagery note 'the elevation tiles (a composite tile set in SRTM format: its main source, the SRTM radar mission of February 2000, "
                   "is an active sensor, and the tiles do not say which source a cell came from)'; in the picture '30 m, static, SRTM format' in place of 'flown 2000'; in the spoken sentence 'an elevation tile set in SRTM format' in place of 'radar elevation of February 2000'."),
             where="Flood_Report_3pages.pdf/.html p.1 (Imagery); Flood_Slides_5 slide 2 (picture) and its notes (Say 11, If asked); results/data_flow.json imagery_note; figures/data_flow_talk.png")]
should = [
    dict(id="S18", severity="should_fix", status=status,
         title="The notes of slide 5 say that the final critic 'compared 3,156 numbers of the documents with the results files'",
         text=("slides_notes_5.md, slide 5, the paragraph 'Review rounds and checks listed': '... together 8 rounds and 188 checks: the four reviews of the experiments 6 rounds and 170 checks, the final critic's review of the documents 2 rounds and 18 checks; the final critic compared 3,156 numbers of the documents with the results files.' "
               f"3,156 is counts.numbers_compared_in_all of results/review_documents_2026-10-08.json (sum built in d9_assemble_review.py): 69 numbers of the documents compared with the results files (d1) and the rest are the numbers of the critic's recomputations from the stored forecasts, the Stage IV series, the ERA5 cache, the key figure's sources and the resampling (d2-d5). "
               f"The sentence says that all were numbers of the documents. The count also moves at the next rebuild, because this block adds round 3: the rounds line of report p.3, slide 5, the plan and the notes will read {rounds_before + 1} review rounds and {checks_before + 3} listed checks, and the comparisons {compared_before + n_compared:,}. "
               f"Correction: 'the final critic made {compared_before + n_compared:,} comparisons: {69 + n_compared:,} of numbers of the documents with the results files, the rest recomputations of the results from the stored forecasts and caches'."),
         where="slides_notes_5.md slide 5 (and PPTX notes of slide 5); code/build_slides_short.py (the sentence)"),
    dict(id="S19", severity="should_fix", status=status,
         title="Smallest print: 6 pt in the report's tables, 7 pt for most of page 1 of the report and for the plan; the slides' picture lettering is 9.3 pt",
         text=(f"Measured with PDFKit on the final PDFs (the printed sizes; the report and the plan are A4, the slides 960 x 540 pt). Report page 1: {rp1['sizes']['7.0']:,} of {rp1['n_chars']:,} characters ({share(rp1, '7.0'):.0f}%) at 7.0 pt - the title line, 'How it was done', the dataset table, the flow of the data, 'How computed' and 'Imagery' (the question and the short answer are 9 pt, the heading 10 pt). "
               f"Page 2: {rp2['sizes']['6.0']:,} characters at 6.0 pt ({share(rp2, '6.0'):.0f}%: the lowest-highest single training and the 95% range of every cell of Tables 1 and 2) and {rp2['sizes']['7.0']:,} at 7.0 pt ({share(rp2, '7.0'):.0f}%). Page 3: 9 pt text, the work table 8 pt, the list of sources 7.0 pt ({rp3['sizes']['7.0']:,} characters, {share(rp3, '7.0'):.0f}%). "
               f"The plan is 7 pt ({FS['plan']['pages'][0]['sizes']['7.0']:,} + {FS['plan']['pages'][1]['sizes']['7.0']:,} + {FS['plan']['pages'][2]['sizes']['7.0']:,} characters) and 8 pt for most of its text. The slides' own text is 10 pt or more (footers 10, captions 11, body 13-17). "
               f"The pictures on the slides are raster images: slide 2 (data flow) was drawn with 10.8 pt as its smallest lettering at 12 in and is shown at {pic[2]['shown_pt'] / (flow_w_in * 72):.2f} of that size = {eff_flow:.1f} pt; slide 3 (key figure) 10.4 pt at 8 in, shown at {pic[3]['shown_pt'] / (key_w_in * 72):.2f} = {eff_key:.1f} pt "
               "(sizes from results/data_flow.json and results/key_figure.json, scaled by the width of the picture on the slide; the map on slide 1 was not measured). Read at 200 dpi the tables are legible and the grey ranges are the smallest print of the set; 6 to 7 pt is below the 8 to 9 pt that a printed A4 page usually keeps. "
               "Correction (the page limit is the reason): print the grey ranges at 7 pt, or leave the single-training range out of Table 2 (the 95% range carries the uncertainty); give the dataset block of page 1 the 8 pt of the work table where words can be cut."),
         where="Flood_Report_3pages.pdf p.1 (dataset block) and p.2 (Tables 1 and 2); Presentation_Plan_5slides.pdf; Flood_Slides_5 slides 2 and 3 (pictures)"),
    dict(id="S20", severity="should_fix", status=status,
         title="The plan's honesty line, which every member is to say aloud, says that independent reviewers 'checked the code and the numbers' and leaves out the final critic",
         text=("Presentation_Plan_5slides.md section 8: 'independent AI reviewer agents checked the code and the numbers; the team members' own checks that the plan assigns are still to be done'. REVIEW_WORDING (report p.3, slide 5) says that they checked the shared engine and every experiment script (results/engine_review.json, results/review_*.json) and that "
               "the real-time check on all onsets, the rain-clock diagnostic, the resampling ranges and the key figure of 8 October 2026 were verified by the final critic agent only; slide 1 says 'checked the shared engine and the experiments; a critic agent made the final checks'. "
               "'The code' and 'the numbers' without the limits is the kind of sentence that M7 of round 1 closed in the report, the Q&A and the guide (the plan was not among them). Correction: say the slide-1 wording in section 8 and in the rehearsal notes."),
         where="Presentation_Plan_5slides.pdf p.3 / .md section 8"),
    dict(id="S21", severity="should_fix", status=status,
         title="Slides 4 and 5 and the plan's table print the 48% with a short form, 'a check, not a service', without 'on the test years'",
         text=("The rule of round 1 (M4, closed in round 2 for 'wherever the 48% is a headline') is 'a check on the test years with the real-time radar product in place of AORC, in networks trained on AORC, not a service'. Report pages 1-3, slide 3 (caption) and the notes carry it in full. "
               "Slide 4 ('Stage IV radar, a real-time product, in AORC's place: a check, not a service'), slide 5 ('48% with the real-time radar product in its place (a check, not a service)') and the plan (section 2, part 4 right half: '48% with Stage IV (a check, not a service)') carry a short form that leaves out 'on the test years' and 'in networks trained on AORC'. "
               "The 48% is never printed without 'not a service' (r3_check.py, b_wording), so this is a should-fix, not a must-fix: the short form keeps the sense but not the words that the coordinator's rule names. Correction: 'a check on the test years, not a service' in the three places (about 20 characters each)."),
         where="Flood_Slides_5 slides 4 and 5; Presentation_Plan_5slides.pdf/.md section 2 (part 4 right half)"),
    dict(id="S22", severity="should_fix", status=status,
         title="Counts at two precisions or in two senses: GEFS 48.5 GB and 48 GB; the parts of the 73 GB add up to 72.3; '193 networks' and '78 networks'",
         text=(f"The GEFS download is '48.5 GB' in the table of report p.1 and '48 GB' in the work table of p.3 and on slide 5 ('GEFS 48, Stage IV 16, AORC 4.5, CAMELS-US 3.5, the rest 0.3' = 72.3, printed beside '73 GB'; data_flow.json: {tot:.2f} GB = {gb['gefs']:.2f} + {gb['stage4']:.2f} + {gb['aorc']:.2f} + {gb['camels_us']:.2f} + {rest:.2f}), "
               "so a listener who adds the parts of slide 5 gets 72.3 against 73. Report p.1 says 'Forecasts stored: 189 sets · 78 networks' and p.3 'Trainings 193 networks': 193 trainings were run (167 recorded + 26 in the hyper-parameter log, data_flow.json flow stage 8) and 78 networks are saved; the word 'networks' is used for both. "
               f"Correction: one precision for the parts (GEFS 48.5, Stage IV 16.1, AORC 4.5, CAMELS-US 3.5, the rest 0.3: they add up to 72.9, the total stays '73 GB') and 'Trainings 193 (78 networks saved)'."),
         where="Flood_Report_3pages.pdf p.1 and p.3; Flood_Slides_5 slide 5"),
]
checks = [
    dict(id="r3a1", what="Tables 1 and 2 of the three-page report (page 2): the 15 rows, every printed number (onsets warned of the average of three trainings, the lowest-highest single training, the 95% range; the median NSE with the same three; the false-warning share; the two simple models; Stage IV; E2b)",
         how="r3_tables.py: the critic's own scoring of results/preds/<set>.npz and <set>_s0..2.npz (high-flow level = 95th percentile of the training years; onset = high-flow day forecast while the river was at or below the level lead days before), the 95% ranges from results/uncertainty.json (which d2 redrew with 4,000 draws), "
              "E2b from results/forecast_input.json; each row is built in the printed format and looked for in the text of page 2",
         result="pass", numbers=dict(rows=T["n_rows"], rows_found_as_printed=T["n_found"], printed_numbers_compared=n_table_numbers)),
    dict(id="r3a2", what="the dataset block and the flow of report p.1 and slides 2 and 5: 671 -> 78 -> 63 -> 12 basins, 12,418 days, 7,305 | 1,826 | 3,287, 73 GB read / 5.9 GB kept (69.1 GB recorded), 395 MB cached, 189 sets and 78 networks (counted in results/preds and results/models), 611 onsets, 21,443 weights, the high-flow levels 2.62-7.86 mm/day, "
              "149,016 basin-days, the fair test (0.797, -0.002, 0.000), the peaks (45%, 84%), bytes per product, grid, step and period of every product in the report's table, the slide-2 picture, the Daymet and NLDAS day ends (08:48, 07:26 to 18:16, 17:40)",
         how="r3_check.py group a_dataset_numbers: every string is built from results/data_flow.json (flow stages, all_information.bytes_read.per_product, the figure lines), results/fair_test.json, results/flood_analysis.json and results/day_windows.json and looked for in the report (PDF and HTML), the slides (PDF and PPTX) and the plan; OCR of the slide-2 picture (ocr.swift)",
         result="pass", numbers=dict(checks=groups["a_dataset_numbers"][1], ok=groups["a_dataset_numbers"][0])),
    dict(id="r3a3", what="all other numbers of the documents: the numbers that are new compared with the texts judged in rounds 1 and 2, the headline numbers of the five-slide set recomputed now, the counts printed by the builders",
         how="r3_new_numbers.py: the numeric tokens of each new document that occur in none of the texts judged before (the seven long PDFs of round 2 and the long deck's Markdown), looked for in the text of results/data_flow.json; the rest traced by hand (below); r3_check.py group a_headline: the number to beat (0.56/0.33/0.25 against 0.59/0.32/0.25, 5, 5 and 9 of 12 basins), "
              "the spread of the four kinds of model (0.0329 NSE, 8.0 points), 19% and 18%, Stage IV 294 (48%) and 275 (45%) from the critic's own recomputation (d3), 313 and 114 from results/stage4_realtime.json, 90% and its range 85% to 97% against the critic's draws (84.1% to 96.5%)",
         result="findings: S18, S22 (no number differs from the results)",
         numbers=dict(per_document={k: dict(distinct_numbers=v["distinct_numbers"], new=v["not_in_the_texts_judged_before"], found_in_data_flow_json=v["of_these_in_data_flow_json"], traced_by_hand=v["not_in_data_flow_json"]) for k, v in NN.items()},
                      traced_by_hand=traced, headline_checks=groups["a_headline"][1], headline_ok=groups["a_headline"][0], the_numbers_of_the_texts_judged_before="same values as in rounds 1 and 2 (d1: 69 claims; d2-d5 recomputations), not repeated here except the sample of the headline checks")),
    dict(id="r3b", what="the wording rules in both documents, the notes and the plan: FINDING and SUPPORT verbatim; AORC 'an analysis made after the fact' and never 'observed'; the hindsight bound beside every 51%; the Stage IV 48% with the check wording; E2b as another split; NSE_FORECAST_WORDING; MOVED_POINT scoped to the warnings; 'no clear difference'; "
              "the honesty line with REVIEW_WORDING; the team's checks open; 'an upper bound that cannot be had in practice' at the first mention of the perfect forecast",
         how="r3_check.py group b_wording: the constants of code/report_sections/s0_summary.py (read from the file) looked for in the report (PDF, HTML), the slides (PDF, PPTX), the notes and the plan; every '51%' and every '48%' of the report, the PPTX and the plan with the 260 characters around it; patterns for 'observed', 'in real time', 'a service could expect', 'was not tested'",
         result="findings: S20, S21 (all 40 rule checks pass)", numbers=dict(checks=groups["b_wording"][1], ok=groups["b_wording"][0], note="the fixed wordings are verbatim; S21 is the short form of the 48% wording on slides 4 and 5; S20 is the plan's spoken honesty line")),
    dict(id="r3c", what="the dataset facts: no WorldCover; the land cover is the Annual NLCD map of 2011 made by USGS from Landsat; the elevation tiles as a composite tile set in SRTM format; the imagery note present in both documents; nothing stated that data_flow.json marks as unverified",
         how="r3_check.py group c_dataset_facts and a search of the report, its HTML, the slides, the PPTX, the notes and the plan for versions, collections, releases and the 11 unverified facts of results/data_flow.json",
         result="finding: M9", numbers=dict(checks=groups["c_dataset_facts"][1], ok=groups["c_dataset_facts"][0], worldcover_hits=0, imagery_note="in full on report p.1 and in the notes of slide 2, as one line ('no optical scene was classified: capture date and sun angle do not apply') on slide 2",
                                          unverified_facts_in_data_flow_json=len(DF["unverified"]), unverified_stated_as_verified=0, tiles="'Elevation tiles (SRTM format)' in the product lists; 'SRTM radar flown in February 2000' in the imagery note, 'flown 2000' on slide 2: M9")),
    dict(id="r3d", what="layout: exactly 3 pages and exactly 5 slides, nothing cut off, the smallest print, the PPTX opens with python-pptx and has notes on all 5 slides",
         how="pdf2txt page counts; every page of the report (3), the plan (3) and every slide (5) read at 100 dpi, report page 2 at 200 dpi; PDFKit font sizes (fontsizes.swift); the pictures' lettering from the figure records; python-pptx: 5 slides of 13.33 x 7.5 in, notes of "
              f"{' / '.join(f'{n:,}' for n in notes_chars)} characters; r3_pptx_fit.py (Arial metrics, an estimate: PowerPoint was not available)",
         result="finding: S19 (pages, slides, notes and cut-offs pass)",
         numbers=dict(pages=3, slides=5, smallest_pt=mins, smallest_pt_note="the report's grey ranges (page 2) 6.0 pt; the report's body on pages 1 and 3 and the plan 7.0 pt; the slides' text 10 pt; the pictures' lettering 9.3 pt (slide 3) and 9.9 pt (slide 2)", notes_characters=notes_chars,
                      pptx_text_frames=X["frames"], pptx_frames_estimated_too_tall=[(r["slide"], r["frame"], r["ratio"]) for r in X["over"]], page_images_changed_by_the_rebuild_of_17_38="slide 2 only (one sentence under the figure)")),
    dict(id="r3e", what="the plan: the seconds sum to 600; the team lead 300 s (50%); a hand-over for every part; the shared slide 4 explained; the clock checkpoints; results/slide_plan_5.json = the slides = the plan's table; the sentences fit the seconds at 2.3 words a second",
         how="r3_check.py group e_plan: the six parts of the printed plan and of results/slide_plan_5.json (70, 120, 150, 90 + 90, 80 s), the running clocks, the titles of the PPTX, the words of the 'Say' blocks of slides_notes_5.md counted by whitespace",
         result="pass", numbers=dict(checks=groups["e_plan"][1], ok=groups["e_plan"][0], seconds=[70, 120, 150, 90, 90, 80], total=600, lead=300,
                                   words_per_second={k: round(v / s, 2) for (k, v), s in zip(C["e_plan_words"].items(), [70, 120, 150, 90, 90, 80])}, plan_says="at most 2.3 words a second (the busiest part is slide 3: 325 words in 150 s = 2.17)")),
]
n_pass = sum(c["result"] == "pass" for c in checks)
last = time.strftime("%H:%M")
round_3 = {
    "when": f"2026-10-08, from 17:36 to about {last} (local time, UTC+4), on the two hand-in documents in their final builds (report 17:36, slides 17:38, plan 17:39); the first reading was of the builds of 17:33-17:35",
    "what": "numbers and rules of the three-page report (Flood_Report_3pages.pdf/.html) and of the five-slide set (Flood_Slides_5.pdf/.pptx, slides_notes_5.md, results/slide_plan_5.json, Presentation_Plan_5slides.pdf/.md); their dataset text comes from results/data_flow.json and figures/data_flow*.png (code/19_data_flow.py)",
    "files_judged": files,
    "rebuilds_during_the_review": "the documents were rebuilt while this round ran: report 17:36:16 (first read: 17:33 build), slides 17:38:54, plan 17:39:00, slides_notes_5.md 17:38:55. Compared word by word with the builds first read: the report differs only in the order of the table text and in the line count (41,429 -> 41,446); the plan is identical; "
                                         "the slides differ in one sentence under the figure of slide 2 ('the whole flow is in the report (figures/data_flow.png)' -> 'the whole figure is figures/data_flow.png') and the notes in two sentences ('We store 189 sets of forecasts.' in slide 2's reference list; 'figures/data_flow.png, and the 3-page report lists all 10 stages' in the notes of slide 2). "
                                         "Every check of this block was run again on the final files; the page images of the report and the plan are identical to the ones read first, those of the slides differ in slide 2 only." + rename_note,
    "method": ["r3_tables.py: Tables 1 and 2 recomputed from the stored forecasts (own scoring) and results/uncertainty.json",
               "r3_check.py: dataset and flow numbers against results/data_flow.json and the files; headline numbers; wording rules; dataset facts; layout facts; the plan",
               "r3_new_numbers.py: the numeric tokens of each new document compared with those of the texts judged in rounds 1 and 2; the new ones traced to data_flow.json or by hand",
               "every page of the report (3), the plan (3) and every slide (5) read at 100 dpi (report page 2 also at 200 dpi); fontsizes.swift (PDFKit: the printed font sizes); ocr.swift on the pictures of the slides; r3_pptx_fit.py (Arial metrics) for the PPTX",
               "the code of the short builders read where they count (code/build_report_short.py scripts(), reviews(); code/build_slides_short.py counts)"],
    "checks": checks,
    "must_fix": must, "should_fix": should,
    "observations": [
        f"Tables 1 and 2: all {T['n_rows']} rows, {n_table_numbers} printed numbers, equal the critic's own recomputation to the printed digit, including the lowest-highest single training of every cell and the 95% ranges. The large number is the average of three trainings and can lie outside the range of the single trainings (Table 1: 'Perfect weather forecast' 64% against single trainings 62-63%; Table 2: 'perfect weather of the next three days', 2 days, NSE 0.82 against 0.77-0.81; E2b NSE 0.62 against 0.60); "
        "the table caption says 'average of three trainings'.",
        "The dataset block of page 1 and slide 2 agree with results/data_flow.json in every number; the unverified facts are not claimed as verified: the report names the Daymet version and the IMERG release as not verified and says that whether the Stage IV archive holds first-posted or later-corrected hours is not established; the zip address 'was not recorded'; no SRTM release, NLCD collection or IMERG release is stated anywhere (the notes quote the record's own 'unverified' entries as such).",
        "The five slides do not print the shuffled-forecast control (E2a: the onsets warned fall to 14% / 4% / 4%); the report prints it on page 2 and the plan keeps it as question H7.",
        "'Same inputs, four kinds of model: within 0.03 NSE and 8 points of onsets' (report p.1, slide 3) is the rounding of a spread of 0.0329 NSE (key_figure.json l1.spread.wq.nse) and 8.0 points; 'within about 0.03' would be exact.",
        f"The work-in-numbers lines of report p.3 count files and lines when the report is built ({len(own)} own files / {lines(own):,} lines now; 109 / 12,345 reviewer files at the 17:36 build): the reviewer count includes this critic's scripts, so it grows whenever one is added here (this round added r3_tables.py, r3_check.py, r3_new_numbers.py, r3_pptx_fit.py, r3_append.py).",
        f"The builders read every 'counts' block of every results/review_*.json: after this block the rebuild prints {rounds_before + 1} review rounds and {checks_before + n_pass} listed checks (now {rounds_before} and {checks_before}), 0 failed, 0 numbers differed (round_3.counts: failed 0, numerical_discrepancies 0), and the sentence about the final critic's comparisons reads {compared_before + n_compared:,} (S18). The documents were last built with 8 rounds and 188 checks: rebuild once after applying the fixes.",
        f"PPTX: the estimate of the text heights in its {X['frames']} text frames finds the five titles about 5% to 11% above their frames (one line more than the frame holds in a narrow title box) and the cells '41 / 61%' and '14 / 49%' of the last row of slide 4's table (71 pt wide columns), which may wrap into two lines in PowerPoint; the plan says to present from the PDF, in which nothing is cut off.",
    ],
    "counts": dict(checks_listed=n_pass, checks_passed=n_pass, findings_not_pass_fail=len(checks) - n_pass, failed=0, must_fix=len(must), should_fix=len(should), numerical_discrepancies=0,
                   numbers_compared_in_all=n_compared, numbers_compared_composition=dict(printed_numbers_of_tables_1_and_2=n_table_numbers, new_numbers_traced=n_new_total),
                   individual_checks_run=n_checks_total, individual_checks_ok=n_checks_total),
    "items": {f["id"]: dict(status="open", title=f["title"]) for f in must + should},
    "earlier_items_not_rejudged": {i: v["status"] for i, v in rec["round_2"]["items"].items() if v["status"] != "closed"},
    "verdict": ("Round 3, the two hand-in documents in their final builds (report 17:36, slides 17:38, plan 17:39). The numbers are right: Tables 1 and 2 of the three-page report (15 rows, "
                f"{n_table_numbers} printed numbers) equal my own recomputation from the stored forecasts to the printed digit; every number of the dataset block, the flow, the byte counts, the grids and periods, the file counts and the day-end clocks equals results/data_flow.json and the files; the {n_new_total} numbers that are new in the four documents are all traced; "
                "the headline numbers equal the results (the number to beat, Stage IV 294 / 275 of 611 against my own recomputation, 90%, the spread of the four kinds of model); the plan's seconds sum to 600 (the team lead 300 s, six hand-overs, the shared slide 4 explained, results/slide_plan_5.json equal to the slides and the plan). "
                "The wording rules hold: FINDING, SUPPORT, REVIEW_WORDING and NSE_FORECAST_WORDING verbatim; every 51% has the hindsight bound; every 48% has 'not a service' (in full on the report and slide 3, in a short form on slides 4 and 5 and in the plan: S21); E2b is marked as another split; MOVED_POINT is scoped to the warnings; "
                "no AORC is called observed; the team's own checks are still open in the report, on slides 1 and 5 and in the plan; there is no WorldCover anywhere and the land cover is the Annual NLCD of 2011 from Landsat; no unverified fact is stated as verified. The layout is as asked: exactly 3 pages and 5 slides, nothing cut off in any PDF, the PPTX opens with python-pptx and has notes on all five slides. "
                "The smallest print is 6.0 pt (the grey ranges of report page 2); the report's body is 7 pt on pages 1 and 3 and the plan 7 to 8 pt; the slides' text is 10 pt or more and their pictures' lettering 9.3 pt or more (S19). "
                "One must-fix item: the elevation tiles are called 'SRTM radar flown in February 2000' in the imagery note of the report and 'flown 2000' on slide 2, where the data record and the plan's own rule say that they are a composite tile set in SRTM format (M9: wording only, written by code/19_data_flow.py, a rebuild changes no number). "
                "Five should-fix items: the notes' sentence about the critic's 3,156 comparisons (S18), the print size (S19), the plan's honesty line (S20), the short form of the 48% wording on slides 4 and 5 (S21) and counts shown at two precisions (S22). "
                "With M9 corrected and the documents rebuilt once more (the record now counts 9 rounds and 191 checks), the two documents are ready to hand in as the project describes itself: an AI-built and AI-reviewed negative result, with the team members' own checks still to be done."),
    "not_checked": ["PowerPoint was not available: the PPTX was judged by python-pptx and an estimate of the text heights, the PDF by reading the pages; nothing was projected.",
                    "The printed colour, the paper and the exact print size were not checked; the pages were read at 100 dpi (report page 2 at 200 dpi) on screen.",
                    "The sentences of the notes (slides_notes_5.md, 85 KB) and of the plan that occur word for word in documents verified before were not read again; the new ones (the dataset entries, slide 5, the plan's rehearsal) were read against results/data_flow.json and the results.",
                    "results/data_flow.json itself (code/19_data_flow.py) is the source of the dataset numbers and was not recomputed from the raw archives; what can be redone cheaply was (file counts, sums, bytes, the day-end clocks, the 149,016 basin-days = 12 x 12,418, the 7,305 + 1,826 + 3,287 split, the lines of code, the hash of the HURDAT2 file), and the caveats about the sources are the record's own.",
                    "The 193 trainings and 8.2 hours of the work table are not new in these documents (they occur in the texts judged in rounds 1 and 2); the 193 = 167 recorded + 26 hyper-parameter trainings is the record's own count (data_flow.json flow stage 8), not recounted here.",
                    "The map of slide 1 (figures/overview_map.png) was looked at but its lettering was not measured."],
}
(OUT / "round_3.json").write_text(json.dumps(round_3, indent=1, ensure_ascii=False))

# ---- the record -------------------------------------------------------------------------------------------------------------------------------------------------------------------
rec["round_3"] = round_3
rec["when"] = rec["when"].split("; round 3")[0] + f"; round 3 on the two hand-in documents (final builds of 17:36-17:39) from 17:36 to {last}"
rec["dates"]["round_3"] = f"2026-10-08 17:36 to about {last} on the three-page report (17:36) and the five-slide set (17:38-17:39); first reading of the builds of 17:33-17:35"
rec["dates"]["this_file"] = time.strftime("%Y-%m-%d %H:%M:%S")
for i in [f["id"] for f in must + should]:
    rec["status_of_the_document_items"][i] = status
rec["scripts"]["script_hashes"] = {p.name: sha(p)[:16] for p in sorted(HERE.glob("*")) if p.suffix in (".py", ".swift", ".sh", ".json", ".md") and p.name not in ("r3_append.py",)} | {"r3_append.py": "(this script)"}
rec["scripts"]["round_3"] = "r3_tables.py, r3_check.py, r3_new_numbers.py, r3_pptx_fit.py (outputs in $REVIEW_OUT), fontsizes.swift, ocr.swift, then r3_append.py [--write-into-project] after d9_assemble_review.py; env R3_TEXT_DIR (the text of the three PDFs, fs_*.json, the OCR of the slide pictures) and R2_TEXT_DIR (the PDF texts of the long documents of round 2) as in the docstrings"
out = OUT / "review_documents_2026-10-08.json"
dump = lambda o: json.dumps(o, indent=1, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x))
out.write_text(dump(rec))
json.loads(out.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
print("written", out, out.stat().st_size, "bytes; round_3.counts:", round_3["counts"])
if WRITE:
    rp.write_text(out.read_text())
    print("written into the project:", rp)
