"""Six-slide PROPOSAL deck (10 minutes) -> proposal/Flood_Proposal_Slides.pptx / .pdf / _notes.md.

Run from the project root:  ~/AI_Class/.venv/bin/python code/build_proposal_slides.py
Text is copied from proposal/AI_and_ALLMs_Proposal_Flood.md and proposal/Meeting_Discussion_Points.md; no number is computed here.
Each slide is a list of elements in inches (text / image / table) drawn twice: as HTML (printed to PDF through report_common.page()/print_pdf(), 16:9)
and as native, editable python-pptx shapes with speaker notes. Nothing written by other builders is changed.
"""
import base64
import html
import io
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import team
import schedule
from common import FIG, ROOT
from report_common import page, print_pdf, INK, INK2, BLUE, MUTED, GRID, ORANGE, AQUA

OUT = ROOT / "proposal"
NAME = "Flood_Proposal_Slides"
W, H = 13.333, 7.5
DATA_CROP = (0, 0, 1, 0.528)              # same crop as code/build_slides_short.py
FONT_PPT = "Arial"
TOTAL = 600
SPEAKERS = {"A": "Member 1 (RS/GIS, team lead)", "R2": "RS/GIS member 2", "G1": "GNSS member 1", "G2": "GNSS member 2"}
WPS = 2.3

TITLE = "Predicting River Floods with AI: an LSTM against simple models"
SUB = "on twelve south-eastern US basins (CAMELS-US, 1980–2014)"
AIM = ("To determine, by a fair test on twelve south-eastern US river basins, what limits a one-to-three-day forecast of river flow and flood onsets, "
       "the kind of model or the information in its inputs, and whether the warnings can be trusted.")
GAP = ("No public forecast study on these small basins separates the clock of the rain record from the forecast: the Daymet day of CAMELS closes at about "
       "09:00 local time, not at the midnight when a forecast is issued.")
RQ = ["RQ1: Does the all-basin LSTM beat the best simple model on years never used for any choice? A gain is clear only if every training of the LSTM is "
      "above every training of the simple model and the LSTM is higher in most basins.",
      "RQ2: Which information closes the gap: rain up to the issue time, a weather forecast, or more rivers? Hypothesis: rain to midnight and a forecast raise "
      "the onsets warned clearly above their controls; more rivers raise the NSE on rivers never seen.",
      "RQ3: Can the warnings be trusted by criteria fixed in advance? GO needs at least 70% of onsets warned and at most 30% of warnings false."]
DISCLOSURE = ("Claude Code wrote and ran all code under the team lead's direction, independent AI reviewers checked it, and the team's own checks are still open.")


def P(text, size=14, bold=False, color=INK, bullet=False, after=4, runs=None):
    return dict(runs=runs or [(text, bold)], size=size, color=color, bullet=bullet, after=after)


def lead(head, rest, size=14, bullet=False, after=5):
    return P("", size=size, bullet=bullet, after=after, runs=[(head, True), (rest, False)])


def T(x, y, w, h, paras, fill=None, line=None):
    return dict(k="text", x=x, y=y, w=w, h=h, paras=paras, fill=fill, line=line)


def crop_png():
    im = Image.open(FIG / "data_flow_talk.png")
    w, h = im.size
    im = im.crop((int(DATA_CROP[0] * w), int(DATA_CROP[1] * h), int(DATA_CROP[2] * w), int(DATA_CROP[3] * h))).convert("RGB")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue(), im.size


def map_png():
    return (FIG / "basin_maps.png").read_bytes(), Image.open(FIG / "basin_maps.png").size


def title(text):
    return T(0.55, 0.3, 12.2, 0.7, [P(text, size=28, bold=True, after=0)])


def deck():
    S = []
    names = [team.label(r) for r in ("A", "R2", "G1", "G2")]
    # 1 ------------------------------------------------------------------------------------------------
    s1 = [T(0.6, 0.45, 7.6, 6.4, [
        P("MID-TERM EVALUATION – Artificial Intelligence and Advanced Large Language Models", size=16, bold=True, color=BLUE, after=14),
        P("PROJECT / RESEARCH TITLE", size=14, color=MUTED, after=2),
        P(TITLE, size=28, bold=True, after=4),
        P(SUB, size=18, color=INK2, after=16),
        P("GROUP PARTICIPANTS", size=14, color=MUTED, after=2),
        *[P(f"{i}. {n}", size=16, after=1) for i, n in enumerate(names, 1)],
        P("", size=8, after=6),
        P("Course Instructor: ______________________________", size=14, after=5),
        P("Submission Date: ________________________________", size=14, after=5),
        P("Academic Session: ______________________________", size=14, after=12),
        P("China", size=16, bold=True, after=0)]),
        T(8.6, 1.5, 4.2, 4.3, [P("OVERALL AIM", size=14, bold=True, color=BLUE, after=8), P(AIM, size=18, after=0)], fill="#f3f7fd", line=BLUE)]
    S.append(dict(els=s1, who="A", sec=40, cover=True))
    # 2 ------------------------------------------------------------------------------------------------
    s2 = [title("Background, gap and research questions"),
          T(0.55, 1.1, 12.2, 2.0, [
              P("Floods in small rain-fed basins build within a day or two of the rain, so a warning helps only if it comes a day or more ahead and sees the rain that has already fallen.", size=16, bullet=True, after=5),
              P("One LSTM trained on many basins is the published state of the art on the CAMELS data (Kratzert et al. 2019; Nearing et al. 2024).", size=16, bullet=True, after=5),
              P("Earth-observation inputs: gridded rain analyses (Daymet, AORC), satellite rain (IMERG), radar-and-gauge rain (Stage IV) and an archived weather forecast (GEFSv12).", size=16, bullet=True, after=0)]),
          T(0.55, 3.05, 12.2, 1.05, [lead("The gap. ", GAP, size=16, after=0)], fill="#fdf6f2", line=ORANGE),
          T(0.55, 4.3, 12.2, 2.6, [P("Research questions", size=16, bold=True, color=BLUE, after=4)] + [P(q, size=14, after=5) for q in RQ])]
    S.append(dict(els=s2, who="A", sec=60))
    # 3 ------------------------------------------------------------------------------------------------
    cap = ("Study area: the 12 study basins with elevation inside each outline, one row per landscape. Coordinate system: geographic latitude and longitude "
           "(outlines recorded as NAD83 in their .prj file, within 1–2 m of WGS84 here; elevation tiles on a 1-arc-second geographic grid); no graticule; each panel has its own scale bar. "
           "Sources: outlines and gauges, CAMELS-US [4]; elevation, 1-arc-second SRTM-format tiles (Mapzen/Tilezen, AWS Open Data; SRTM flown February 2000) [8].")
    rows = [["CAMELS-US", "Flow (target, gauge input); Daymet and NLDAS-2 weather; 10 basin properties; outlines", "671 basins, 12 used; daily; 1980–2014"],
            ["NOAA AORC v1.1", "Hourly rain, an analysis made after the fact: rain to local midnight", "1 km; hourly; 1980–2014"],
            ["GEFSv12 reforecast", "Archived weather forecast: rain of the next three days", "0.25°; 3-hourly to hour 78; 2000–2014"],
            ["Stage IV radar", "Real-time radar-and-gauge rain: check on the 611 test-year onsets", "4.76 km; hourly; 2005–2014"],
            ["ERA5 hourly rain", "Model rain: early test of rain to midnight; clock reference", "0.25°; hourly; 1980–2014"],
            ["IMERG, MERRA-2", "Satellite rain, soil wetness: tested extra inputs", "0.1° / 0.5°×0.625°; daily; 1998 / 1981–2014"],
            ["Elevation tiles (SRTM)", "Elevation, relief, slope; basin maps", "1 arc-second (~30 m); static"],
            ["Annual NLCD", "Land-cover shares per basin; basin maps", "30 m; 2011 layer"],
            ["HURDAT2 tracks", "Storm type of each onset (check)", "6-hourly; 2005–2014 onsets"]]
    s3 = [title("Study area and data"),
          dict(k="img", src=map_png, x=0.55, y=1.05, w=3.5, h=3.5 * 1782 / 1474),
          T(0.55, 5.35, 4.4, 1.6, [P(cap, size=11, color=INK2, after=0)]),
          dict(k="table", x=5.1, y=1.1, colw=[2.0, 3.5, 2.2], head=["Data product", "What it gives", "Resolution / period"], rows=rows, size=11),
          T(5.1, 6.3, 7.7, 0.6, [P("No optical or SAR scene is processed, so capture date and sun angle do not apply.", size=14, bold=True, color=BLUE, after=0)])]
    S.append(dict(els=s3, who="R2", sec=150))
    # 4 ------------------------------------------------------------------------------------------------
    iw = 11.6
    s4 = [title("Methodology and workflow"),
          dict(k="img", src=crop_png, x=(W - iw) / 2, y=1.0, w=iw, h=iw * 650 / 2640),
          T(0.55, 4.0, 12.2, 3.0, [
              lead("1 Acquire. ", "12 basins fixed by rule before any model runs; grids averaged to basin means; hourly rain summed over the local day; the clock of every rain product measured (Daymet day ends about 09:00, NLDAS-2 about 17:30).", after=5),
              lead("2 Features. ", "six Daymet weather columns, river gauge, ten basin properties and each experiment's extra rain; lagged features for the simple models, checked not to read past the issue day.", after=5),
              lead("3 Models. ", "one LSTM for all 12 basins against three simple baselines (tomorrow = today, ridge, random forest); three seeds each, early stopping on validation.", after=5),
              lead("4 Split. ", "temporal by target day: training 1980–2000, validation 2000–2005, test 2005–2014; leave-one-basin-out as the spatial check.", after=5),
              lead("5 Scoring. ", "NSE, KGE, onsets warned, false warnings; GO / CAUTION / NOT YET criteria fixed in advance; headline numbers recomputed independently.", after=0)])]
    S.append(dict(els=s4, who="G1", sec=150))
    # 5 ------------------------------------------------------------------------------------------------
    s5 = [title("Expected outputs and evaluation plan"),
          T(0.55, 1.1, 6.1, 3.8, [P("Deliverables", size=16, bold=True, color=BLUE, after=5),
              lead("D1 ", "Curated database: basin-by-day tables, 12 basins × 12,418 days, and a record of every product.", after=6),
              lead("D2 ", "The engine and scripts, with an independent review record per experiment script.", after=6),
              lead("D3 ", "Basin maps and GIS layers (outlines, elevation, land cover) and the key figure.", after=6),
              lead("D4 ", "The report, every number read from results/*.json.", after=6),
              lead("D5 ", "A 10-minute talk and the question sheet, rehearsed by the four members.", after=0)]),
          dict(k="table", x=6.9, y=1.15, colw=[2.6, 1.3, 2.0], head=["Indicator", "Baseline", "Target"], size=12, rows=[
              ["Onsets warned one day ahead", "19%", "above 45% (network trained on real-time radar, Stage IV)"],
              ["False onset warnings", "59% of warnings", "below 50% of warnings"],
              ["Trainings per model", "3", "5 where the NSE decides; each gain with its range"],
              ["Review records, GIS layers, own checks", "open", "100% of scripts; all 12 basins; 4 of 4 members"]]),
          T(0.55, 5.1, 12.2, 1.75, [lead("Preliminary (laptop runs, October 2026): ", "median NSE 0.59 for the LSTM against 0.56 for the best simple model (no clear difference); onsets warned a day ahead 19% with the river gauge, 51% with rain to midnight (AORC), 48% with real-time radar (Stage IV); verdict still NOT YET. Caveat: the real-time radar number comes from networks trained on AORC, the after-the-fact analysis, so it is a check, not a service.", size=14, after=0)], fill="#fdf6f2", line=ORANGE)]
    S.append(dict(els=s5, who="G2", sec=150))
    # 6 ------------------------------------------------------------------------------------------------
    tl = schedule.ROWS_SHORT
    s6 = [title("Timeline, roles, disclosure and questions"),
          T(0.55, 1.05, 5.6, 0.4, [P(schedule.HEADING + " (last project day " + schedule.d(schedule.DEADLINE) + ")", size=16, bold=True, color=BLUE, after=0)]),
          dict(k="table", x=0.55, y=1.55, colw=[1.6, 4.0], head=["Week", "Phase"], rows=tl, size=14),
          T(6.6, 1.05, 6.2, 5.9, [
              P("Roles", size=16, bold=True, color=BLUE, after=3),
              lead(team.label("A") + ": ", "data audit, report build, coordination", after=1),
              lead(team.label("R2") + ": ", "basin maps and the GIS section", after=1),
              lead(team.label("G1") + ": ", "scoring and the fair-test split", after=1),
              lead(team.label("G2") + ": ", "training follow-up and model comparison", after=9),
              P("AI-assistance disclosure", size=16, bold=True, color=BLUE, after=3),
              P(DISCLOSURE, size=16, after=9),
              P("Questions for the professor", size=16, bold=True, color=BLUE, after=3),
              P("Scope: is a US public dataset acceptable, given that no comparable open daily flow archive exists for Chinese rivers?", size=16, bullet=True, after=3),
              P("A negative result is a result: is it an acceptable outcome for the grade, if shown rigorously?", size=16, bullet=True, after=3),
              P("Use of AI tools: is it acceptable, and how should it be stated?", size=16, bullet=True, after=0)])]
    S.append(dict(els=s6, who="A", sec=50))
    return S


NOTES = [
    ("Good morning. We are four students of remote sensing, GIS and GNSS, in China. Our proposal asks what limits a one-to-three-day forecast of river floods "
     "on small basins: the kind of model, or the information we give it. We will test one LSTM against simple models on twelve south-eastern US basins, 1980 to 2014, "
     "and judge the warnings by criteria fixed in advance.",
     "I will now give the background and our three research questions."),
    ("Floods in small rain-fed basins build within a day or two of the rain, so a warning must come a day ahead and see rain that has already fallen. "
     "One LSTM trained on many basins is the published state of the art on CAMELS, from Kratzert in 2019 and Nearing in 2024. Our inputs are earth-observation products: "
     "gridded rain, satellite rain, radar rain and an archived forecast. The gap: no public study on these basins separates the clock of the rain record from the forecast. "
     "The three questions are on the slide: does the LSTM beat simple models, which information closes the gap, and can the warnings be trusted.",
     "My colleague now presents the study area and the data."),
    ("This is our study area: twelve basins chosen from the 671 in CAMELS-US, in North Carolina, Virginia, Georgia, Florida and Mississippi. A rule picked them before any model ran: "
     "78 candidates, 63 with a complete record, and four per landscape, evenly spaced in area. They cover 113 to 1,928 square kilometres in three landscapes: mountains, northern lowlands and "
     "Piedmont, and southern lowlands. The map shows elevation inside each outline, with the river gauge as a white dot; the coordinate system is geographic latitude and longitude. "
     "On the right are the data products, all public and read without payment. CAMELS-US gives flow, Daymet weather and basin properties. AORC gives hourly rain, an analysis made after the fact. "
     "The GEFSv12 reforecast is an archived weather forecast, so a forecast input can be tested on past years without hindsight. Stage IV is real-time radar rain, ERA5 gives hourly model rain, "
     "and IMERG and MERRA-2 give satellite rain and soil wetness. Elevation tiles and land cover serve the maps and checks. One point for a remote sensing course: "
     "no optical or SAR scene is processed, so capture date and sun angle do not apply.",
     "My colleague will now explain the methodology and workflow."),
    ("This figure shows how the data flow through the project: eleven public products at the top feed one pipeline. Step one, we acquire the products and average the grids into basin means. "
     "The twelve basins are fixed by rule before any model runs, and we measure the clock of every rain product, because the Daymet day ends at about 09:00 local time. "
     "Step two, features: six weather columns, the river gauge and ten basin properties, plus the extra input of each experiment. "
     "Step three, one LSTM trained on all twelve basins against three simple baselines: tomorrow equals today, ridge regression and a random forest. "
     "Each model is trained with three seeds, and training stops early on the validation score. "
     "Step four, the split is by target day: training 1980 to 2000, validation 2000 to 2005, test 2005 to 2014; leave-one-basin-out is the spatial check. "
     "Step five, scoring with NSE, KGE, onsets warned and false warnings, with the GO, CAUTION and NOT YET criteria fixed in advance, and every headline number recomputed independently.",
     "My colleague will now present the expected outputs and the evaluation plan."),
    ("We plan five deliverables. D1, a curated database of basin-by-day tables with a record of every product. D2, the engine and scripts, with an independent review record for each experiment. "
     "D3, basin maps and GIS layers. D4, the report. D5, this talk and a question sheet. On the right are the baseline and the targets. The baseline is the best simple model with the same inputs; "
     "in preliminary runs, 19% of flood onsets are warned a day ahead and 59% of the warnings are false. Our targets, to be met and not yet achieved: above 45% of onsets warned with a network "
     "trained on real-time radar, and below 50% of warnings false. We also plan five trainings per model instead of three where the NSE decides. "
     "One line on the preliminary state, from laptop runs in October 2026: the LSTM and the best simple model show no clear difference, median NSE 0.59 against 0.56, "
     "and rain known to midnight raises onsets warned to 51%. The caveat: the real-time radar number, 48%, comes from networks trained on AORC, the after-the-fact analysis, "
     "so it is a check, not a service. The verdict is still NOT YET.",
     "The team lead will now close with the timeline, roles and our questions."),
    ("The last project day is the fifth of November, three weeks from now: this week the proposal, your feedback and our own checks; next week the remaining trainings and the GIS layers; the last week the final report, slides and hand-in. "
     "Each member owns one part. AI assistance is disclosed: Claude Code writes and runs the code under my direction, independent AI reviewers check it, and our own checks are still open. "
     "We have three questions for you: is a US public dataset acceptable, is a rigorously shown negative result acceptable, and how should the use of AI tools be stated?",
     "Thank you; we welcome your questions and your advice on all three."),
]


# ---------------------------------------------------------------------------------------------- HTML
def esc(t):
    return html.escape(t)


def para_html(p):
    runs = "".join(f"<b>{esc(t)}</b>" if b else esc(t) for t, b in p["runs"])
    st = f"font-size:{p['size']}pt;color:{p['color']};margin:0 0 {p['after']}pt;"
    if p["bullet"]:
        st += "display:list-item;list-style:disc;margin-left:16pt;"
    return f'<div style="{st}">{runs or "&nbsp;"}</div>'


def el_html(e):
    pos = f"left:{e['x']}in;top:{e['y']}in;"
    if e["k"] == "text":
        box = f"{pos}width:{e['w']}in;height:{e['h']}in;"
        if e["fill"]:
            box += f"background:{e['fill']};border-left:4px solid {e['line']};padding:8pt 12pt;box-sizing:border-box;"
        return f'<div class="el" style="{box}">' + "".join(para_html(p) for p in e["paras"]) + "</div>"
    if e["k"] == "img":
        data, _ = e["src"]()
        return f'<img class="el" style="{pos}width:{e["w"]}in;height:{e["h"]}in" src="data:image/png;base64,{base64.b64encode(data).decode()}">'
    cols = "".join(f'<col style="width:{w}in">' for w in e["colw"])
    head = "".join(f"<th>{esc(h)}</th>" for h in e["head"])
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in e["rows"])
    return f'<table class="el" style="{pos}width:{sum(e["colw"])}in;font-size:{e["size"]}pt"><colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


CSS = f"""
@page{{size:{W}in {H}in;margin:0}}
html,body{{margin:0;padding:0}}
.slide{{position:relative;width:{W}in;height:{H}in;overflow:hidden;box-sizing:border-box;background:var(--surface);break-after:page;break-inside:avoid}}
.slide:last-child{{break-after:auto}}
.el{{position:absolute;line-height:1.28}}
.el table,table.el{{border-collapse:collapse;line-height:1.2}}
table.el th,table.el td{{padding:4pt 6pt;vertical-align:top;border-top:1px solid var(--grid)}}
table.el th{{color:var(--ink2);font-weight:700;text-align:left;border-top:0;border-bottom:2px solid var(--blue)}}
.foot{{position:absolute;left:0.55in;right:0.55in;bottom:0.12in;font-size:11pt;color:var(--muted);display:flex;justify-content:space-between}}
"""


def foot(i, s):
    return f'<div class="foot"><span>Flood proposal · AI and ALLMs mid-term</span><span>{esc(team.label(s["who"]))} · {s["sec"]} s · {i}/6</span></div>'


def deck_html(S):
    sl = "".join(f'<section class="slide">{"".join(el_html(e) for e in s["els"])}{foot(i, s)}</section>' for i, s in enumerate(S, 1))
    return page("Flood proposal - six slides", sl, CSS)


# ---------------------------------------------------------------------------------------------- PPTX
def rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def fill_tf(tf, paras):
    tf.word_wrap = True
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(p["after"])
        runs = p["runs"]
        if p["bullet"]:
            runs = [("• ", False)] + list(runs)
        for t, b in runs:
            r = para.add_run()
            r.text = t
            r.font.size = Pt(p["size"])
            r.font.bold = b
            r.font.name = FONT_PPT
            r.font.color.rgb = rgb(p["color"])


def deck_pptx(S, path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for i, s in enumerate(S, 1):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        for e in s["els"]:
            if e["k"] == "text":
                if e["fill"]:
                    bar = sl.shapes.add_shape(1, Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
                    bar.fill.solid()
                    bar.fill.fore_color.rgb = rgb(e["fill"])
                    bar.line.fill.background()
                    ln = sl.shapes.add_shape(1, Inches(e["x"]), Inches(e["y"]), Inches(0.05), Inches(e["h"]))
                    ln.fill.solid()
                    ln.fill.fore_color.rgb = rgb(e["line"])
                    ln.line.fill.background()
                pad = 0.17 if e["fill"] else 0.0
                tb = sl.shapes.add_textbox(Inches(e["x"] + pad), Inches(e["y"] + (0.08 if e["fill"] else 0)), Inches(e["w"] - 2 * pad), Inches(e["h"]))
                tb.text_frame.margin_left = tb.text_frame.margin_right = tb.text_frame.margin_top = tb.text_frame.margin_bottom = 0
                fill_tf(tb.text_frame, e["paras"])
            elif e["k"] == "img":
                data, _ = e["src"]()
                sl.shapes.add_picture(io.BytesIO(data), Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            else:
                n = len(e["rows"]) + 1
                rh = 0.42 if e["size"] <= 12 else 0.4
                gs = sl.shapes.add_table(n, len(e["colw"]), Inches(e["x"]), Inches(e["y"]), Inches(sum(e["colw"])), Inches(rh * n))
                tbl = gs.table
                tblPr = gs._element.graphic.graphicData.tbl.tblPr
                tblPr.set("firstRow", "0")
                tblPr.set("bandRow", "0")
                for j, w in enumerate(e["colw"]):
                    tbl.columns[j].width = Inches(w)
                for r in range(n):
                    tbl.rows[r].height = Inches(rh)
                    for c in range(len(e["colw"])):
                        cell = tbl.cell(r, c)
                        cell.margin_left = cell.margin_right = Inches(0.07)
                        cell.margin_top = cell.margin_bottom = Inches(0.04)
                        cell.vertical_anchor = MSO_ANCHOR.TOP
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = rgb("#e8eefb" if r == 0 else "#ffffff")
                        txt = e["head"][c] if r == 0 else e["rows"][r - 1][c]
                        fill_tf(cell.text_frame, [P(txt, size=e["size"], bold=(r == 0 or c == 0 and False), color=INK2 if r == 0 else INK, after=0)])
        ft = sl.shapes.add_textbox(Inches(0.55), Inches(7.1), Inches(12.2), Inches(0.3))
        ft.text_frame.margin_left = ft.text_frame.margin_top = 0
        fill_tf(ft.text_frame, [P(f"Flood proposal · AI and ALLMs mid-term  |  {team.label(s['who'])} · {s['sec']} s · {i}/6", size=11, color=MUTED, after=0)])
        say, hand = NOTES[i - 1]
        sl.notes_slide.notes_text_frame.text = f"[{team.label(s['who'])}, {s['sec']} s]\n\n{say}\n\nHand-over: {hand}"
    prs.save(path)


# ---------------------------------------------------------------------------------------------- notes
def words(t):
    return len(t.split())


def checks(S):
    tot = {k: 0 for k in SPEAKERS}
    for s, (say, hand) in zip(S, NOTES):
        tot[s["who"]] += s["sec"]
        n = words(say) + words(hand)
        assert n <= WPS * s["sec"], f"notes too long: {n} words for {s['sec']} s (max {WPS * s['sec']:.0f})"
    assert sum(tot.values()) == TOTAL and all(v == 150 for v in tot.values()), tot
    return tot


def notes_md(S, tot):
    out = ["# Proposal talk: speaker notes (10 minutes, six slides)", "",
           "| Part | Slide | Presenter | Seconds | Starts at |", "|---|---|---|---|---|"]
    t = 0
    for i, s in enumerate(S, 1):
        out.append(f"| {i} | {i} | {team.label(s['who'])} | {s['sec']} | {t // 60}:{t % 60:02d} |")
        t += s["sec"]
    out += ["", f"Total: {sum(s['sec'] for s in S)} s = 10 minutes. Per member: " + "; ".join(f"{team.label(k)} {v} s" for k, v in tot.items()) +
            ". The team lead speaks on slides 1, 2 and 6 (40 + 60 + 50 s). Words are at most 2.3 per second of each part, hand-over included.", ""]
    t = 0
    for i, (s, (say, hand)) in enumerate(zip(S, NOTES), 1):
        out += [f"## Slide {i} - {team.label(s['who'])} - {s['sec']} s (from {t // 60}:{t % 60:02d}; {words(say) + words(hand)} words)", "", say, "", f"**Hand-over:** {hand}", ""]
        t += s["sec"]
    return "\n".join(out)


def main():
    S = deck()
    tot = checks(S)
    OUT.mkdir(exist_ok=True)
    h = OUT / f"{NAME}.html"
    h.write_text(deck_html(S), encoding="utf-8")
    pdf = OUT / f"{NAME}.pdf"
    print_pdf(str(h.resolve()), str(pdf.resolve()))
    h.unlink()
    deck_pptx(S, OUT / f"{NAME}.pptx")
    (OUT / f"{NAME}_notes.md").write_text(notes_md(S, tot), encoding="utf-8")
    print("wrote", pdf, OUT / f"{NAME}.pptx", OUT / f"{NAME}_notes.md")


if __name__ == "__main__":
    main()
