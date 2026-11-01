"""r3_check: round 3 of the final critic - the two hand-in documents built on 8 October 2026 (17:31-17:35): Flood_Report_3pages.pdf/.html and the five-slide deck (Flood_Slides_5.pdf/.pptx, slides_notes_5.md,
results/slide_plan_5.json, Presentation_Plan_5slides.pdf/.md). Numbers and rules only.

    R3_TEXT_DIR=<folder with Flood_Report_3pages.txt, Flood_Slides_5.txt, Presentation_Plan_5slides.txt (pdf2txt output), fs_*.json (fontsizes output, optional), slide<n>_pic.json (ocr output of the pictures, optional)>
    python code/review/documents_2026-10-08/r3_check.py

What it checks (every check prints its result and is stored in OUT/r3_check.json):
  a  the numbers of the dataset block and the flow (results/data_flow.json) against the text of the report (page 1), the slides (slide 2, slide 5) and the plan; the file counts (189 sets, 78 networks); the bytes per product;
     the grid, step and period of each product against the report's table and the OCR of the slide-2 picture; the sums (73 GB = the products; 7,305 + 1,826 + 3,287 = 12,418)
  b  the wording rules: FINDING and SUPPORT verbatim; AORC 'an analysis made after the fact' and never 'observed'; the hindsight bound beside every 51% that is a headline; the 48% with the check-on-the-test-years wording;
     E2b marked as another split; NSE_FORECAST_WORDING; MOVED_POINT scoped to the warnings; 'no clear difference'; REVIEW_WORDING; the team's checks open
  c  the dataset facts: no WorldCover; Annual NLCD 2011 from Landsat; the elevation tiles as a composite tile set in SRTM format; the imagery note; nothing that data_flow.json marks as unverified stated as verified
  d  layout: pages and slides, notes on every slide, the smallest print (PDFKit font sizes; OCR of the pictures)
  e  the plan: seconds sum to 600, the team lead 300 s, hand-overs, the shared slide 4
"""
import os
import re
from collections import Counter

from dcommon import *
from pptx import Presentation

TXT = Path(os.environ["R3_TEXT_DIR"])
norm = lambda s: re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("−", "-").replace("–", "-")).strip()
pages = lambda t: {int(m): norm(x) for m, x in zip(*[iter(re.split(r"\n=====PAGE (\d+)=====\n", "\n" + t)[1:])] * 2)}
REP = pages((TXT / "Flood_Report_3pages.txt").read_text())
SL = pages((TXT / "Flood_Slides_5.txt").read_text())
PL = pages((TXT / "Presentation_Plan_5slides.txt").read_text())
html = (PROJ / "Flood_Report_3pages.html").read_text(errors="ignore")
REP_HTML = norm(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)))
pr = Presentation(str(PROJ / "Flood_Slides_5.pptx"))
PPTX, NOTES = {}, {}
for i, s in enumerate(pr.slides, 1):
    ch = []
    for sh in s.shapes:
        if sh.has_text_frame:
            ch.append(sh.text_frame.text)
        if getattr(sh, "has_table", False) and sh.has_table:
            ch += [c.text for r in sh.table.rows for c in r.cells]
    PPTX[i], NOTES[i] = norm(" ".join(ch)), norm(s.notes_slide.notes_text_frame.text if s.has_notes_slide else "")
NOTES_MD = norm((PROJ / "slides_notes_5.md").read_text())
PLAN_MD = next(q for q in (PROJ / "Presentation_Plan_5slides.md", PROJ / "Presentation_Plan.md") if q.exists()).read_text()      # the coordinator renamed the plan to Presentation_Plan.md at 18:11 (byte-identical)
ALL_REP = " ".join(REP.values())
ALL_SL = " ".join(SL.values())
ALL_PPTX = " ".join(PPTX.values())
DF = jload("data_flow")
res = {}


def has(text, frag):
    return norm(frag).lower().replace("– ", "–") in text.lower().replace("– ", "–")


def record(group, name, ok, detail=""):
    res.setdefault(group, []).append(dict(check=name, ok=bool(ok), detail=detail))
    if not ok:
        print(f"  NOT OK [{group}] {name}: {detail}"[:400])


# ---- a. the dataset block and the flow ------------------------------------------------------------------------------------------------------------------------------------
flow = {s["n"]: s["counts"] for s in DF["flow"]["stages"]}
c2, c4, c5, c1, c3, c6, c7, c8, c9 = flow[2], flow[4], flow[5], flow[1], flow[3], flow[6], flow[7], flow[8], flow[9]
n = lambda x: f"{x:,}"
for frag, src in [(f"{n(c2['all_camels'])} river basins", "flow[2].all_camels"), (f"the project uses {c2['chosen']} of them, each with {n(c4['days_per_basin'])} days", "flow[2].chosen, flow[4].days_per_basin"),
                  (f"{c1['products']} products · {c1['gb_read']:.0f} GB read", "flow[1]"), (f"{n(c2['all_camels'])} {c2['candidates']} {c2['complete']} {c2['chosen']}", "flow[2]"),
                  (f"{c3['five_caches_mb']} MB cached", "flow[3].five_caches_mb"), (f"{c4['basins']} basins × {n(c4['days_per_basin'])} days", "flow[4]"),
                  (f"{n(c5['train_days'])} | {n(c5['validation_days'])} | {n(c5['test_days'])} days", "flow[5]"),
                  (f"{c6['level_min_mm_day']}–{c6['level_max_mm_day']} mm/day", "flow[6]"), (f"{n(c7['weights'])} weights", "flow[7].weights"), (f"{c9['onsets']} onsets · {n(c9['test_days'])} test days", "flow[9]"),
                  (f"recomputed for all {n(c4['basin_days'])} basin-days: largest difference 0", "flow[4].basin_days")]:
    record("a_dataset_numbers", f"report p.1: '{frag}'", has(REP[1], frag), src)
sets_n = len(list((RES / "preds").glob("*.npz")))
nets_n = len(list((RES / "models").glob("*.pt")))
record("a_dataset_numbers", f"189 sets / 78 networks equal the files (results/preds {sets_n}, results/models {nets_n}) and the report", sets_n == 189 and nets_n == 78 and has(REP[1], "189 sets · 78 networks"), f"{sets_n} npz, {nets_n} pt")
record("a_dataset_numbers", "7,305 + 1,826 + 3,287 = 12,418 days (flow[5].total_days = flow[4].days_per_basin)", c5["train_days"] + c5["validation_days"] + c5["test_days"] == c5["total_days"] == c4["days_per_basin"] == 12418, f"{c5['train_days'] + c5['validation_days'] + c5['test_days']}")
record("a_dataset_numbers", "189 sets, 78 networks, 193 = 167 recorded + 26 in the hyper-parameter log (data_flow.flow[8]); 611 onsets", c8["prediction_sets"] == 189 and c8["saved_networks"] == 78 and c8["trainings_recorded"] == 167 and c9["onsets"] == 611, str(c8))
# bytes per product: report table
per = DF["all_information"]["bytes_read"]["per_product"]
fmt = lambda b: (f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB")
prod_text = {"camels_us": "3.5 GB*", "aorc": "4.5 GB", "era5": "86 MB*", "gefs": "48.5 GB", "imerg": "15 MB*", "merra2": "40 MB*", "stage4": "16.1 GB"}
for k, frag in prod_text.items():
    b = per.get(k)
    ok = b is not None and fmt(b) == frag.rstrip("*") and has(REP[1], frag)
    record("a_dataset_numbers", f"bytes of {k}: {frag} (data_flow: {b})", ok, fmt(b) if b else "key missing: " + ", ".join(per))
record("a_dataset_numbers", "transfer recorded 69.1 GB; read 73 GB (sum of the products 72.8)", has(REP[3], "69.1") and abs(sum(per.values()) / 1e9 - c1["gb_read"]) < 0.06, f"sum of products {sum(per.values()) / 1e9:.2f} GB; gb_read {c1['gb_read']}; recorded {c1['gb_read_recorded_transfers']}")
record("a_dataset_numbers", "5.9 GB kept (slide 2) = flow[1].gb_kept_under_data", has(ALL_SL, "5.9 GB are kept") and c1["gb_kept_under_data"] == 5.9, str(c1["gb_kept_under_data"]))
record("a_dataset_numbers", "slide 5: GEFS 48, Stage IV 16, AORC 4.5, CAMELS-US 3.5, the rest 0.3 (sum 72.3 against 73 read: the parts are rounded)", has(ALL_PPTX, "GEFS 48, Stage IV 16, AORC 4.5, CAMELS-US 3.5, the rest 0.3"),
       f"parts {per['gefs'] / 1e9:.2f} + {per['stage4'] / 1e9:.2f} + {per['aorc'] / 1e9:.2f} + camels + rest = {sum(per.values()) / 1e9:.2f}")
# grid, step, period per product against the report's table
grid_frags = {"AORC": "1 km · hourly · 1980–2014", "ERA5": "0.25° point · hourly · 1980–2014", "GEFS": "0.25° · 3-hourly · 2000–2014", "IMERG": "0.1° · daily (UTC) · 1998–2014", "MERRA-2": "0.5×0.625° · daily · 1981–2014",
              "USGS flow": "gauge · daily · 1980–2014", "Daymet": "1 km grid, basin mean · daily · 1980–2014", "NLDAS-2": "1/8° (12 km) · daily · 1980–2014"}
fig_lines = {x["key"]: x["figure_line"][1] for x in DF["datasets"]}
fl = " ".join(DF["figure"]["report"]["source_groups"][i]["boxes"][j][1] for i in range(3) for j in range(len(DF["figure"]["report"]["source_groups"][i]["boxes"])))
for k, frag in grid_frags.items():
    record("a_dataset_numbers", f"grid · step · period of {k}: '{frag}' is in data_flow.json (figure lines) and in the report's table (checked on the HTML: the PDF wraps the cell)", frag in fl and has(REP_HTML, frag), "")
# the picture of slide 2 (OCR)
ocr2 = TXT / "slide2_pic.json"
if ocr2.exists():
    o = " | ".join(r["text"] for r in json.loads(ocr2.read_text()))
    for k in ("ERA5", "AORC", "GEFSv12", "IMERG", "MERRA-2", "Stage IV", "USGS", "HURDAT2"):
        record("a_dataset_numbers", f"slide-2 picture names {k}", k in o, "")
    for frag in ("1980-2014", "2000-2014", "1998-2014", "1981-2014", "2005-2014", "1851-2025", "Jan-Jun", "0.25", "0.1°", "4.76 km", "1 km", "10 properties"):
        record("a_dataset_numbers", f"slide-2 picture prints '{frag}'", frag.replace("-", "–") in o or frag in o, "")
for frag in ("eleven public products", "73 GB were read in all", "671 78 63 12", "12,418 days each, none missing", "7,305 training, 1,826 validation and 3,287 test days", "189 forecast sets stored", "0.0283168 × 86,400", "ends at about 09:00"):
    record("a_dataset_numbers", f"slide 2: '{frag}'", has(ALL_SL, frag) or has(PPTX[2], frag), "")
# the headline numbers of the report: recomputed in d1 (the same sources); here the ones only this document prints
Ch = jload("fair_test")["measured"]["past_weather_only"]
record("a_dataset_numbers", "fair test: 0.797, -0.002, 0.000", has(REP[1], f"scored {Ch[0]:.3f}, −{abs(Ch[1]):.3f}, {abs(Ch[2]):.3f}") or has(REP[1], "scored 0.797, -0.002, 0.000"), str(Ch))
FA = jload("flood_analysis")["failures"]["peak_size"]
record("a_dataset_numbers", "peaks: median event 45% (84% with the perfect weather forecast)", f"{100 * FA['lstm_wq']['all_events']['median']:.0f}" == "45" and f"{100 * FA['lstm_wqf']['all_events']['median']:.0f}" == "84" and has(REP[3], "45% of the observed peak"), "")
DW = jload("day_windows")
hm = lambda h: f"{round(h * 60) // 60:02d}:{round(h * 60) % 60:02d}"
dd, dn = DW["day_ends"]["daymet"], DW["day_ends"]["nldas"]
record("a_dataset_numbers", f"Daymet rain day ends about 09:00 (median basin {dd['day_ends_local']}; basins {dd['basin_range_local'][0]} to {dd['basin_range_local'][1]}) = day_windows.json", dd["day_ends_local"] == hm(dd["day_ends_local_hour"]) == "08:48" and has(REP[1], f"Daymet rain day ends about 09:00 (median basin {dd['day_ends_local']}; basins {dd['basin_range_local'][0]} to {dd['basin_range_local'][1]})"), str(dd["day_ends_local_hour"]))
record("a_dataset_numbers", f"NLDAS-2 day ends about 17:30 (median {dn['day_ends_local']}) = day_windows.json", dn["day_ends_local"] == hm(dn["day_ends_local_hour"]) == "17:40" and has(REP_HTML, f"NLDAS-2 about 17:30 (median {dn['day_ends_local']})"), str(dn["day_ends_local_hour"]))
# ---- a2. headline numbers the five-slide documents print, recomputed from the results now (the others are in Tables 1 and 2, checked by r3_tables.py, and in the texts judged in rounds 1 and 2)
LA = jload("lstm_all")["comparison"]["wq"]
bb = LA["best_baseline_by_validation"]
v = {k: LA["baselines"][bb[k]][k] for k in ("l1", "l2", "l3")}
b2, l2, hi = [f"{v[k]['baseline_nse_median']:.2f}" for k in v], [f"{v[k]['lstm_nse_median']:.2f}" for k in v], [v[k]["n_basins_lstm_higher"] for k in v]
record("a_headline", f"the number to beat: best simple model {b2[0]}, {b2[1]} and {b2[2]} against the LSTM {l2[0]}, {l2[1]} and {l2[2]} (median NSE, 1, 2, 3 days) = lstm_all.json; plan p.1", has(norm(PLAN_MD), f"best simple model {b2[0]}, {b2[1]} and {b2[2]} against the LSTM {l2[0]}, {l2[1]} and {l2[2]}"), str((b2, l2)))
record("a_headline", f"the LSTM is higher in {hi[0]}, {hi[1]} and {hi[2]} of 12 basins (slide 4)", has(PPTX[4], f"higher in {hi[0]}, {hi[1]} and {hi[2]} of 12 basins"), str(hi))
KS = jload("key_figure")["sources"]
four = ("lstm_wq", "base_linreg_wq", "base_rf_wq", "cmp_transformer")
sp_n = max(KS[f"l1.{m}.nse"]["value"] for m in four) - min(KS[f"l1.{m}.nse"]["value"] for m in four)
sp_o = max(KS[f"l1.{m}.onset_warned"]["value"] for m in four) - min(KS[f"l1.{m}.onset_warned"]["value"] for m in four)
record("a_headline", f"four kinds of model on the same inputs: NSE spread {sp_n:.4f} (printed 'within 0.03': the rounding of {sp_n:.3f}) and {100 * sp_o:.1f} points of onsets (printed 8)", f"{sp_n:.2f}" == "0.03" and round(100 * sp_o) == 8 and has(PPTX[3], "within 0.03 NSE and 8 points of onsets"), f"{sp_n:.4f}, {sp_o:.4f}")
record("a_headline", "onsets warned 1 day ahead with the rain table alone 19% (LSTM), at most 18% for the simple models (slide 4)", f"{100 * KS['l1.lstm_wq.onset_warned']['value']:.0f}" == "19" and max(KS["l1.base_linreg_wq.onset_warned"]["value"], KS["l1.base_rf_wq.onset_warned"]["value"]) < 0.185 and has(PPTX[4], "at most 18%"), "")
D3 = jread("d3_stage4.json")
vs = D3["variants"]
iv = D3["interval_of_the_share_surviving_95"]["drawing_onsets"]
S4 = jload("stage4_realtime")
record("a_headline", f"Stage IV: {vs['stage4']['onsets_warned']} ({100 * vs['stage4']['share']:.0f}%) and {vs['stage4_no_last_hour']['onsets_warned']} ({100 * vs['stage4_no_last_hour']['share']:.0f}%) of 611 onsets, my own recomputation (d3), equal the report; 313 with AORC and 114 with the rain table alone = stage4_realtime.json",
       (vs["stage4"]["onsets_warned"], vs["stage4_no_last_hour"]["onsets_warned"]) == (294, 275) and S4["models"]["wqa"]["aorc"] == 313 and S4["models"]["wqa"]["rain_table_alone"] == 114 and has(ALL_REP, "294 (48%) against 313 with AORC and 114 with the rain table alone") and has(ALL_REP, "275 (45%)"), "")
record("a_headline", f"90% of the AORC effect survives, printed range 85% to 97%: my own draws {100 * iv[0]:.1f}% to {100 * iv[1]:.1f}% (within a point: draws differ by chance); the project's own text gives 85% to 97%", round(100 * D3["share_of_the_aorc_effect_surviving"]["stage4"]) == 90 and abs(iv[0] - 0.85) < 0.011 and abs(iv[1] - 0.97) < 0.011 and "85% to 97%" in S4["outcome"] and has(ALL_REP, "90% of the AORC effect survives (85% to 97%"), str(iv))
# ---- b. the wording rules -------------------------------------------------------------------------------------------------------------------------------------------------
S0 = (PROJ / "code" / "report_sections" / "s0_summary.py").read_text()
const = lambda name: re.search(rf'^{name} = \(?\s*"(.*?)"', S0, re.M | re.S).group(1)
FINDING, SUPPORT, REALTIME, HIND, NSEF, MOVED = (const(k) for k in ("FINDING", "SUPPORT", "REALTIME_WORDING", "HINDSIGHT_WORDING", "NSE_FORECAST_WORDING", "MOVED_POINT"))
REVIEW = re.search(r'REVIEW_WORDING = \("(.*?)"\s*\n\s*"(.*?)"\)', S0, re.S)
REVIEW = REVIEW.group(1) + REVIEW.group(2)
docs = {"report": ALL_REP, "report_html": REP_HTML, "slides_pdf": ALL_SL, "slides_pptx": ALL_PPTX, "notes_md": NOTES_MD, "plan_pdf": " ".join(PL.values()), "plan_md": norm(PLAN_MD)}
record("b_wording", "FINDING verbatim in the report (p.1) and as the title of slide 3", has(REP[1], FINDING) and has(PPTX[3], FINDING), FINDING)
record("b_wording", "SUPPORT verbatim in the report (p.1)", has(REP[1], SUPPORT), SUPPORT)
record("b_wording", "SUPPORT verbatim in the slides or their notes", has(ALL_PPTX + " " + " ".join(NOTES.values()), SUPPORT), "the slides do not print it; see the notes")
record("b_wording", "REVIEW_WORDING verbatim: report p.3, slide 5", has(REP[3], REVIEW) and has(PPTX[5], REVIEW), REVIEW[:80] + "...")
record("b_wording", "the team's own checks are still to be done: report p.3, slides 1 and 5", has(REP[3], "team members' own checks are still to be done") and has(PPTX[1], "team's own checks are not yet done") and has(PPTX[5], "team's own checks are not yet done"), "")
record("b_wording", "NSE_FORECAST_WORDING verbatim in the report (p.2), on slide 4 (capitalised first letter)", has(REP[2], NSEF) and has(PPTX[4].lower(), NSEF.lower()), NSEF)
record("b_wording", "MOVED_POINT scoped to the warnings: report p.2 'said of the warnings only', slide 4 'for the warnings'", has(REP[2], MOVED + ", said of the warnings only") and has(PPTX[4], "for the warnings: " + MOVED), MOVED)
record("b_wording", "'no clear difference' (not 'tie') in the report and on slides 4", has(ALL_REP, "no clear difference") and has(PPTX[4], "no clear difference") and not re.search(r"\b(tie|tied)\b", ALL_REP + ALL_PPTX, re.I), "")
record("b_wording", "'the information, not the network' (report p.3, slide 4)", has(REP[3], "the information, not the network") and has(PPTX[4], "the information, not the network"), "")
bad = [(d, m.group(0)) for d, t in docs.items() for m in re.finditer(r"AORC[^.]{0,60}(observed|measured|real rain)|(observed|well-measured)[^.]{0,30}AORC|the observed AORC|rain that (really )?fell", t, re.I)]
record("b_wording", "never 'observed' / 'measured' rain for AORC (report, HTML, slides, PPTX, notes, plan)", not bad, str(bad[:3]))
bad = [(d, m.group(0)) for d, t in docs.items() for m in re.finditer(r"in real time(?! radar)(?! the operational)|service could expect|mixed at all three|(?<!What )was not tested", t, re.I)]
record("b_wording", "no 'in real time' alone, 'a service could expect', 'mixed at all three lead times', 'was not tested' (about a forecast)", not bad, str(bad[:4]))
# every 51% that is a headline carries the hindsight bound; every 48% carries the check wording
plan_md_pages = {1: norm(PLAN_MD.split("## 3.")[0]), 2: norm(PLAN_MD.split("## 3.")[1]) if "## 3." in PLAN_MD else ""}
for dname, text_by in (("report", REP), ("slides_pptx", PPTX), ("plan_md", plan_md_pages)):
    for pg, t in text_by.items():
        for m in re.finditer(r"(?<![\d.-])51%(?! (?:to|and))", t):
            w = t[max(0, m.start() - 260):m.end() + 260]
            is_question = bool(re.search(r"How sure are you of 19%|from AORC of 51%|from 51% to 45%|Quote the two", w))
            record("b_wording", f"{dname} {pg}: 51% has the hindsight bound within 260 characters" + (" (a question title of the Q&A sheet)" if is_question else ""), HIND in w or is_question or dname == "slides_pptx" and HIND in t, w[200:330] if HIND not in w else "")
        for m in re.finditer(r"(?<![\d.-])48%", t):
            w = t[max(0, m.start() - 160):m.end() + 160]
            if not re.search(r"stage iv|real-time|real time", w, re.I) or re.search(r"\d% / $", t[max(0, m.start() - 6):m.start()]):
                continue                                                  # a range or the random forest, not the Stage IV figure
            near = t[max(0, m.start() - 600):m.end() + 900] if dname != "slides_pptx" else t         # the footnote / caption of the same table or slide
            if "‡" in t[max(0, m.start() - 20):m.end() + 5]:                                       # a table cell with the footnote mark: the footnote is under the table
                near = t
            full = "check on the test years with the real-time radar product in place of AORC, in networks trained on AORC, not a service" in near
            short = bool(re.search(r"a check, not a service|a check on the test years, not a service", near))
            record("b_wording", f"{dname} {pg}: the Stage IV 48% with the 'check ... not a service' wording ({'full' if full else 'short' if short else 'MISSING'})", full or short, w)
paras = [norm(p) for p in re.split(r"\n\s*\n|\n(?=- )|\n(?=\d+\. )", (PROJ / "slides_notes_5.md").read_text()) if p.strip()]
n51 = [(p, m.start()) for p in paras for m in re.finditer(r"(?<![\d.-])51%(?! (?:to|and))", p)]
miss51 = [p[max(0, i - 60):i + 80] for p, i in n51 if HIND not in p and "analysis made after the fact" not in p]
record("b_wording", f"notes: {len(n51)} mentions of 51% in {len({p for p, _ in n51})} paragraphs; each paragraph carries the hindsight bound or the AORC wording", not miss51, str(miss51[:3]))
no_h = [p[max(0, i - 70):i + 90] for p, i in n51 if HIND not in p]
res.setdefault("b_notes_51_without_hindsight_in_the_same_paragraph", [dict(check="notes paragraph", ok=True, detail=x) for x in no_h])
n48 = [(p, m.start()) for p in paras for m in re.finditer(r"(?<![\d.-])48%", p) if re.search(r"stage iv|real-time", p, re.I) and not re.search(r"\d% / $", p[max(0, m.start() - 6):m.start()])]
miss48 = [p[max(0, i - 60):i + 100] for p, i in n48 if not re.search(r"not a service", p)]
record("b_wording", f"notes: {len(n48)} mentions of the Stage IV 48%; each in a paragraph that says 'not a service'", not miss48, str(miss48[:3]))
record("b_wording", "E2b marked as another split: report p.2 footnote '§ Another split', slide 4 '* another split ... comparable only inside it'", has(REP[2], "Another split: test years 2010–2014") and has(PPTX[4], "another split (train 2000–2008, test 2010–2014): comparable only inside it"), "")
record("b_wording", "perfect forecast = 'an upper bound that cannot be had in practice' at the first mention (report p.1, slide 3 caption)", has(REP[1], "a perfect weather forecast, an upper bound that cannot be had in practice") and has(PPTX[3], "an upper bound that cannot be had in practice"), "")
record("b_wording", "the shuffled control: E2a 'measures harm', E2b clean (report p.2); the slides print only 'control: the forecast shuffled'? (see detail)", has(REP[2], "measures harm in E2a"), "slides: " + str("harm" in ALL_PPTX))
record("b_wording", "AORC 'an analysis made after the fact' on every slide where AORC appears (3, 4, 5) and in the report (p.1, 2, 3)", all(has(PPTX[i], "analysis made after the fact") for i in (3, 4, 5)) and all(has(REP[i], "analysis made after the fact") for i in (1, 2, 3)), "")
# ---- c. the dataset facts --------------------------------------------------------------------------------------------------------------------------------------------------
allt = " ".join(docs.values()) + " " + " ".join(NOTES.values())
record("c_dataset_facts", "no 'WorldCover' in any of the documents, notes, plan or data_flow.json's own text except the correction note", not re.search(r"WorldCover", allt, re.I), "")
record("c_dataset_facts", "Annual NLCD (2011) from Landsat: report imagery note", has(REP[1], "Annual NLCD land-cover map of 2011 (made by USGS from Landsat images; the map year is the date that applies)"), "")
record("c_dataset_facts", "imagery note present in the report (p.1) and in the slides (slide 2) / notes", has(REP[1], DF["imagery_note"]) and has(PPTX[2], "no optical scene was classified") and ("no optical" in NOTES[2].lower() or "optical" in NOTES[2].lower()), "")
tiles = [m for m in re.finditer(r"elevation tiles[^.;]{0,200}", ALL_REP + " " + NOTES_MD, re.I)]
record("c_dataset_facts", "the elevation tiles are called a composite tile set / 'in SRTM format', not plain SRTM", any(re.search(r"composite|SRTM format", m.group(0), re.I) for m in tiles), "; ".join(m.group(0)[:120] for m in tiles[:4]))
record("c_dataset_facts", "report imagery note: the tiles described as 'SRTM radar flown in February 2000' (the producer's main source) - the composite character is not said there", has(REP[1], "the elevation tiles (SRTM radar flown in February 2000, an active sensor)"), "wording point")
unv = DF["unverified"]
record("c_dataset_facts", f"the report says that {len(unv)} facts are not verified and lists two examples", has(REP[1], f"{len(unv)} facts are not verified, all listed in results/data_flow.json") and has(REP[1], "the Daymet version inside CAMELS, the IMERG release"), "")
claims_unverified = []
for u in unv:
    if u["dataset"] == "stage4" and re.search(r"first[- ]posted|first postings", ALL_REP) and not re.search(r"not established|not verified", ALL_REP):
        claims_unverified.append(u["fact"][:80])
record("c_dataset_facts", "the first-posted-or-corrected question of Stage IV is stated as 'not established' (report p.3)", has(REP[3], "whether its archive holds first-posted or later-corrected hours is not established"), "")
record("c_dataset_facts", "the zip-file address 'was not recorded' (not given as verified)", has(REP[1], "the address of the zip files was not recorded"), "")
record("c_dataset_facts", "the SRTM release and the NLCD collection number are not claimed", not re.search(r"SRTMGL1|Collection 1\.[012]|version 3\.0|version 2\.1", ALL_REP), "")
# ---- d. layout -------------------------------------------------------------------------------------------------------------------------------------------------------------------
record("d_layout", "report: exactly 3 pages", len(REP) == 3, str(len(REP)))
record("d_layout", "slides: exactly 5 slides in the PDF and in the PPTX", len(SL) == 5 and len(pr.slides) == 5, f"{len(SL)} / {len(pr.slides)}")
record("d_layout", "notes on all 5 slides of the PPTX", all(len(v) > 500 for v in NOTES.values()), str({k: len(v) for k, v in NOTES.items()}))
record("d_layout", "slide size 13.33 x 7.5 in; the slide pictures inside the slide", abs(pr.slide_width / 914400 - 13.333) < 0.01 and all(sh.left >= 0 and sh.top >= 0 and sh.left + sh.width <= pr.slide_width + 10 and sh.top + sh.height <= pr.slide_height + 10 for s in pr.slides for sh in s.shapes), "")
for n_, p in (("report", "Flood_Report_3pages"), ("slides", "Flood_Slides_5"), ("plan", "Presentation_Plan_5slides")):
    f = TXT / f"fs_{p}.json"
    if f.exists():
        d = json.loads(f.read_text())["pages"]
        mn = min(x["min"] for x in d)
        res.setdefault("d_layout_fonts", []).append(dict(document=n_, smallest_pt=mn, per_page=[dict(page=x["page"], min=x["min"], sizes=x["sizes"]) for x in d]))
# ---- e. the plan -----------------------------------------------------------------------------------------------------------------------------------------------------------
sp = jload("slide_plan_5")
secs = [(s.get("key") or s.get("n") or i, s.get("seconds")) for i, s in enumerate(sp["slides"])]
pres = sp["presenters"]
record("e_plan", "seconds sum to 600 (slide_plan_5.json total_seconds and the sum of the parts)", sp["total_seconds"] == 600, f"slides {secs}")
rows = re.findall(r"\|\s*(\d(?: (?:left|right) half)?)\s*\|\s*\*\*(.*?)\*\*\s*\|\s*(\d+) s ([\d:–]+)\s*\|", PLAN_MD)
tot = sum(int(r[2]) for r in rows)
abd = sum(int(r[2]) for r in rows if r[1].startswith("Member 1"))
record("e_plan", f"the plan's six parts add up to 600 s (found {len(rows)} rows, {tot} s)", tot == 600 and len(rows) == 6, str([(r[0], r[1], r[2]) for r in rows]))
record("e_plan", f"The team lead speaks 300 s ({abd} s)", abd == 300 and "300 s of the 600 s, 50%" in PLAN_MD, "")
record("e_plan", "a hand-over sentence for every part", len(re.findall(r"\*\*→ ", PLAN_MD.split("## 3.")[0])) >= 6, str(len(re.findall(r"\*\*→ ", PLAN_MD.split('## 3.')[0]))))
record("e_plan", "slide 4 is shared and explained (left half / right half, hand-over inside the slide)", "Slide 4 is shared" in PLAN_MD and "inside the slide" in PLAN_MD, "")
# clock checkpoints
t = 0
cum = []
for r in rows:
    cum.append(t)
    t += int(r[2])
record("e_plan", "the printed clocks (0:00-1:10, 1:10-3:10, 3:10-5:40, 5:40-7:10, 7:10-8:40, 8:40-10:00) equal the running sums", [f"{c // 60}:{c % 60:02d}" for c in cum] == ["0:00", "1:10", "3:10", "5:40", "7:10", "8:40"], str(cum))
words = {}
nm = NOTES_MD
for h in re.split(r"(?m)^(?=#{2,3} )", (PROJ / "slides_notes_5.md").read_text()):
    m = re.match(r"#{2,3} (Slide \d+|Left half|Right half)", h)
    if not m:
        continue
    say = re.search(r"\*\*Say:\*\*\s*\n(.*?)(?=\n\*\*If asked|\n\*\*Do not|\Z)", h, re.S)
    if say:
        txt = re.sub(r"^\s*\d+\.\s*", "", re.sub(r"\*\*Hand-over[^*]*\*\*", "", say.group(1)), flags=re.M)
        words[m.group(1)] = len(txt.split())
res["e_plan_words"] = words
SP = jload("slide_plan_5")
titles = [norm(sh.text_frame.text) for sl in pr.slides for sh in sl.shapes if sh.has_text_frame and sh.name.startswith("Title")]
record("e_plan", "results/slide_plan_5.json: the five titles and seconds equal the five slides of the PPTX and the plan's table (70/120/150/180/80 s, slide 4 shared)", [norm(x["title"]) for x in SP["slides"]] == titles and [x["seconds"] for x in SP["slides"]] == [70, 120, 150, 180, 80] and [x["shared"] for x in SP["slides"]] == [False, False, False, True, False], str(titles)[:200])
jsave("r3_check.json", res)
tot_ok = sum(len(v) for k, v in res.items() if k.startswith(("a_", "b_", "c_", "d_layout", "e_plan")) and isinstance(v, list) and v and "ok" in v[0])
bad_n = sum(1 for k, v in res.items() if isinstance(v, list) and v and "ok" in v[0] for x in v if not x["ok"])
print(f"{tot_ok - bad_n} of {tot_ok} checks ok;", f"{bad_n} not ok")
