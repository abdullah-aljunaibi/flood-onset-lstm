#!/usr/bin/env python
"""Fills the course's mid-term proposal template for the flood project.

    ~/AI_Class/.venv/bin/python code/build_proposal.py

Opens proposal/template.docx with python-docx and fills it IN PLACE, so that every style, heading, table and the cover layout of the
template are kept: each instruction paragraph or bracketed placeholder is replaced by the project's text in the same paragraph object
(a pure-instruction paragraph that sits before an "[Insert ...]" placeholder is deleted; the "Overall aim:", "Objective n:" and
"Research questions" paragraphs are kept and filled), the participants table, the data-source table (illustrative rows replaced by the
real sources) and the deliverables table are filled, and the study-area figure (figures/basin_maps.png) is inserted in section 5.
Course Instructor is the course instructor; Submission Date is 15 October 2026; Academic Session is Session 8 for the team to fill.

Writes proposal/AI_and_ALLMs_Proposal_Flood.docx and, from the finished .docx (so that the twins cannot differ), the Markdown twin
proposal/AI_and_ALLMs_Proposal_Flood.md and the PDF twin proposal/AI_and_ALLMs_Proposal_Flood.pdf (the .docx turned into HTML in the
template's look and printed through report_common.page() + print_pdf, headless Chrome; --no-md and --no-pdf skip the twins).

The voice is that of a PROPOSAL: the plan is written as the plan; the state of the work is told once, in section 8, as preliminary
(October 2026). Numbers: counts, ranges and extents are read from results/*.json when this script runs (and checked against the
text); the headline scores (NSE 0.59 / 0.56, 19%, 51%, 48%, 39% / 37%) are those of README.md, which is itself built from results/.
References are written by hand in IEEE style from code/report_common.REFERENCES (Mendeley was not used; nothing was looked up).
"""
import copy
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, RGBColor
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
PROP = ROOT / "proposal"
TEMPLATE = PROP / "template.docx"
OUT = PROP / "AI_and_ALLMs_Proposal_Flood.docx"
MD = PROP / "AI_and_ALLMs_Proposal_Flood.md"
FIGURE = ROOT / "figures" / "basin_maps.png"
RES = ROOT / "results"


def J(name):
    with open(RES / name, encoding="utf8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------------------------------------
# Facts read from results/*.json (and checked against what the text says)
# ---------------------------------------------------------------------------------------------------------------------------
FLOW, AUDIT, GIS, CLOCK = J("data_flow.json"), J("data_audit.json"), J("basin_gis.json"), J("onset_rain_clock.json")
STAGES = {s["id"]: s for s in FLOW["flow"]["stages"]}
N_ONSETS = CLOCK["n_onsets"]                                                    # onsets of the test years
mm = CLOCK["all"]["mean_mm"]
HIDDEN = round(100 * mm["hidden"] / sum(mm.values()))                          # share of the rain around onsets filed under the next day
COMING = round(100 * (mm["future1"] + mm["future2"]) / sum(mm.values()))      # share still to come after the issue time
N_UNVERIFIED = len(FLOW["unverified"])
N_PRODUCTS = STAGES["download"]["counts"]["products"]
GB_READ, GB_KEPT = round(STAGES["download"]["counts"]["gb_read"]), STAGES["download"]["counts"]["gb_kept_under_data"]
N_PREDS = int(re.search(r"(\d+) prediction sets", STAGES["forecasts"]["words"]).group(1))
N_NETS = int(re.search(r"(\d+) trained networks", STAGES["forecasts"]["words"]).group(1))
m_days = re.search(r"([\d,]+) days, no value missing \(([\d,]+) basin-days\)", STAGES["inputs"]["words"])
DAYS, BASIN_DAYS = m_days.group(1), m_days.group(2)
N_CAND, N_COMPLETE, N_GAPS, N_ALL = AUDIT["n_candidates"], AUDIT["n_complete"], AUDIT["n_with_gaps"], AUDIT["n_all_camels"]
SEL = AUDIT["selected"]
BAS = GIS["basins"]
lat = [s["gauge_lat"] for s in SEL]
lon = [s["gauge_lon"] for s in SEL]
LAT0, LAT1, LON0, LON1 = min(lat), max(lat), -max(lon), -min(lon)             # degrees N, degrees W
AREA0, AREA1 = round(min(b["area_km2"] for b in BAS.values())), round(max(b["area_km2"] for b in BAS.values()))
P0, P1 = min(s["p_mean_mm"] for s in SEL), max(s["p_mean_mm"] for s in SEL)
Q0, Q1 = min(s["q_mean_mm"] for s in SEL), max(s["q_mean_mm"] for s in SEL)
elev = {b["short"]: b["elev_mean_m"] for b in BAS.values()}
LOW, HIGH = min(elev, key=elev.get), max(elev, key=elev.get)


def share(landscape_start, group, pick=min):
    return pick(b["land_cover_share"][group] for b in BAS.values() if b["landscape"].startswith(landscape_start))


# the ranges quoted in section 5 (checked, so the text cannot drift from the file)
F_MTN = (round(100 * share("Mountains", "forest", min)), round(100 * share("Mountains", "forest", max)))
FLORIDA = [b for b in BAS.values() if b["short"] in ("Charlie Creek, FL", "Myakka, FL")]
FARM = (round(100 * min(b["land_cover_share"]["cropland and pasture"] for b in FLORIDA)),
        round(100 * max(b["land_cover_share"]["cropland and pasture"] for b in FLORIDA)))
WET = round(100 * min(b["land_cover_share"]["wetland"] for b in FLORIDA))
PIED = sorted(round(100 * b["land_cover_share"]["forest"]) for b in BAS.values()
              if b["landscape"].startswith("Northern") and b["short"] != "NE Cape Fear, NC")
assert (N_ONSETS, HIDDEN, COMING, N_UNVERIFIED, N_PRODUCTS, GB_READ, N_PREDS, N_NETS) == (611, 35, 46, 11, 11, 73, 189, 78), "facts moved"
assert (N_CAND, N_COMPLETE, N_GAPS, N_ALL, DAYS, BASIN_DAYS) == (78, 63, 15, 671, "12,418", "149,016")
assert (AREA0, AREA1, F_MTN, FARM, WET, PIED) == (113, 1928, (75, 94), (56, 61), 29, [63, 63, 65]), (AREA0, AREA1, F_MTN, FARM, WET, PIED)
assert (round(P0, 1), round(P1, 1), round(Q0, 1), round(Q1, 1)) == (3.2, 5.5, 0.7, 3.2)
assert (LOW, HIGH) == ("Myakka, FL", "South Toe, NC")
assert (round(LAT0, 1), round(LAT1, 1), round(LON0, 1), round(LON1, 1)) == (27.3, 36.7, 77.8, 89.4)
# the headline scores quoted in section 8 are those of README.md (written from results/ by build_report.py): check they are still there
README = (ROOT / "README.md").read_text(encoding="utf8")
for needle in ("0.59 vs 0.56", "19%", "51%", "48% with Stage IV", "39% / 37%", "611 onsets", "about 3 points"):
    assert needle in README, f"README.md no longer says {needle!r}"
EL_LOW, EL_HIGH = f"{round(elev[LOW]):,}", f"{round(elev[HIGH]):,}"

# ---------------------------------------------------------------------------------------------------------------------------
# The text of the proposal
# ---------------------------------------------------------------------------------------------------------------------------
TITLE_1 = "Predicting River Floods with AI: an LSTM against simple models"
TITLE_2 = "on twelve south-eastern US basins (CAMELS-US, 1980–2014)"
import team as _team
import schedule as _sched
INSTRUCTOR = "The course instructor"                                                 # (user, 2026-10-14)
PARTICIPANTS = [_team.label(k) for k in ("A", "R2", "G1", "G2")]       # names from team.json (the class list of 2026-10-14)
STUDENT_IDS = ["", "", "", ""]       # handwritten on the printed cover (user, 2026-10-15)
SUBMISSION_DATE = "15 October 2026"
ACADEMIC_SESSION = "Session 8"




AIM = ("Overall aim: To determine, by a fair test on twelve south-eastern US river basins, what limits a one-to-three-day forecast "
       "of river flow and flood onsets, the kind of model or the information in its inputs, and whether the warnings can be trusted.")
RQ_HEAD = "Research questions/hypotheses (RQ), linked to the objectives:"

FIG_LABEL = "Figure 1. "
FIG_ALT = ("Twelve small elevation maps, one per study basin, in three rows (mountains, northern lowlands and Piedmont, southern "
           "lowlands), each with its own scale bar, a colour scale from lowest to highest point and a white dot for the river gauge.")

DATA_HEAD = ("Data source", None, None)






GLOSSARY = [
    ("Term", "Meaning"),
    ("Issue time", "When the forecast is made: local midnight at the end of the issue day. Nothing after it may be read."),
    ("Flood level (Q95)", "For each river, the flow exceeded on only 5% of the training days."),
    ("Flood onset", "The first day the flow rises above the flood level after a day below it. A warning counts only if issued while "
                    "the river was still below. Onsets cluster in storms, so they are not independent."),
    ("Onsets warned / false warnings", "The share of onsets that received a warning in time / the share of onset warnings that were wrong."),
    ("NSE", "Nash–Sutcliffe efficiency of the flow forecast: 1 is perfect, 0 is no better than always forecasting the average flow."),
    ("Control", "The same added input with its useful part removed (for example rain from the wrong hours), to show that a gain "
                "comes from the information and not from an extra column."),
    ("GO / CAUTION / NOT YET", "Trust verdicts. GO: at least 70% of onsets warned and at most 30% of warnings false. CAUTION: not "
                               "GO, but hits are at least 40% of hits, misses and false warnings together. NOT YET: neither."),
]






# ---- section 8: state of the work, first results (table), targets (table), timeline (table)
FI = J("forecast_input.json")["ladder"]["main"]["rungs"]
RUNG = {r["set"]: r["by_lead"] for r in FI}
S4RT = J("stage4_realtime.json")


def pc(x):
    return f"{round(100 * x)}%"


def leads(s):
    return " / ".join(pc(RUNG[s][l]["onset_warned"]) for l in ("l1", "l2", "l3"))


m_s4 = re.search(r"warn of (\d+) \((\d+)%\) against (\d+) with AORC", S4RT["outcome"])
S4_PCT = int(m_s4.group(2))
LADDER = [
    ("Information given to the LSTM", "Onsets warned 1 / 2 / 3 days ahead", "False onset warnings (1 day)", "Median NSE (1 day)"),
    ("Past weather and river gauge (baseline)", leads("lstm_wq"), pc(RUNG["lstm_wq"]["l1"]["onset_false"]), f"{RUNG['lstm_wq']['l1']['nse_median']:.2f}"),
    ("+ rain up to midnight (AORC, after the fact)", leads("lstm_wqa"), pc(RUNG["lstm_wqa"]["l1"]["onset_false"]), f"{RUNG['lstm_wqa']['l1']['nse_median']:.2f}"),
    ("Same networks, real-time radar (Stage IV) on the issue day", f"{S4_PCT}% (1 day only)", "–", "–"),
    ("+ GEFS weather forecast of the next 3 days", leads("lstm_wqaF_sub"), pc(RUNG["lstm_wqaF_sub"]["l1"]["onset_false"]), f"{RUNG['lstm_wqaF_sub']['l1']['nse_median']:.2f}"),
    ("Perfect weather forecast (upper bound, not possible)", leads("lstm_wqf"), pc(RUNG["lstm_wqf"]["l1"]["onset_false"]), f"{RUNG['lstm_wqf']['l1']['nse_median']:.2f}"),
]
assert [r[1] for r in LADDER[1:]] == ["19% / 8% / 6%", "51% / 21% / 12%", "48% (1 day only)", "45% / 39% / 37%", "64% / 67% / 68%"], LADDER
assert [r[3] for r in LADDER[1:]] == ["0.59", "0.77", "–", "0.69", "0.86"], LADDER

S8_STATE = (
    "State of the work. Already built: the data pipeline and the database for 12 basins, the shared code with its fairness tests, "
    "one LSTM and three simple models, the first experiments with added information, and an independent AI review of each. "
    "Remaining weeks: the deciding runs repeated with five trainings, each member's own check, the GIS layers, the targets below, "
    "and the final report and talk."
)
S8_CAVEAT = (
    f"Onsets are counted per lead ({N_ONSETS} at one day ahead). Read with care: AORC is an analysis made after the fact, and its "
    "hours before 2002 (all training years) are daily totals spread over the day; the Stage IV row is a check on the test years in "
    "networks trained on AORC, not a service; and the warnings respond mainly to whether the issue day was a heavy-rain day (with "
    "the Stage IV rain of another onset of the same basin, the networks still warn of 284 onsets against 294). The verdict is NOT "
    "YET in every row."
)
TARGETS_HEAD = "Targets for the remaining weeks (the first two are the GO level of Section 3, not yet reached by any input):"
TIMELINE_HEAD = "Timeline (the last project day is " + _sched.d(_sched.DEADLINE, True) + "):"
TIMELINE = [("Week", "Dates", "Work", "Closes")] + [
    (str(w["n"]), f"{_sched.d(w['from_'])}–{_sched.d(w['to'])}", f"{w['title']}: {w['what']}", w["closes"]) for w in _sched.WEEKS]






S3 = [
    "A flood onset is the first day a river rises above its flood level (the flow exceeded on 5% of training days). In first runs "
    f"on the {N_ONSETS} onsets of the test years 2005–2014, an LSTM given past weather and the river gauge warned of only 19% one "
    "day ahead, and 59% of its warnings were false. A likely reason is the clock: rain falling at 3 pm on Monday is filed under "
    "Tuesday, so a forecast made at midnight on Monday cannot see it. Around the onsets, this hidden rain is on average "
    f"{HIDDEN}% of the rain of the 63 hours around the forecast. Assumed, not yet shown: that hidden rain, not the network, limits "
    "the warning; that radar is available at forecast time; that twelve humid basins stand for others. Gap: no study, to our "
    "knowledge, has measured this cost on small basins.",
]
OBJ = [
    "Objective 1: A basin-by-day table for 12 basins, with each product's clock measured.",
    "Objective 2: A fair test: time split, leak checks, one LSTM, three simple models.",
    "Objective 3: Scores 1–3 days ahead for each added input, against its control.",
    "Objective 4: Basin maps, GIS layers and what a warning service would need.",
]
RQ = [
    "RQ1 (O2, O3): Does the LSTM beat the best simple model? Hypothesis: information matters more than the network.",
    "RQ2 (O1, O3): Which added input helps most: rain to midnight, radar or a forecast? Hypothesis: each beats its control.",
    "RQ3 (O3, O4): Can the warnings be trusted (GO: ≥ 70% of onsets warned, ≤ 30% false)? Hypothesis: not yet.",
]
FIG_CAPTION = "The 12 basins; colour = elevation within each basin; white dot = river gauge; each panel has its own scale bar."
DATA = [
    ("CAMELS-US [4], [5]", "Flow (target), daily weather, basin properties, outlines", "Daily, 1980–2014"),
    ("NOAA AORC v1.1 [9]", "Hourly rain to midnight (analysis made after the fact)", "1 km, hourly"),
    ("NCEP Stage IV [11]", "Real-time radar-and-gauge rain (check)", "4 km, hourly, 2005–2014"),
    ("GEFSv12 reforecast [6]", "Archived weather forecast, next 3 days", "0.25°, 3-hourly, 2000–2014"),
    ("SRTM-format elevation; NLCD 2011", "Basin maps, relief, land cover", "30 m, static"),
]
DELIV = [
    ("D1. Curated database", f"12 basins × {DAYS} days; a record of every product"),
    ("D2. Code and review records", "Fairness tests pass; a review per experiment"),
    ("D3. Basin maps and GIS layers", "Scale bar, legend, coordinates"),
    ("D4. Final report", "Every number generated from the results files"),
    ("D5. Talk and question sheet", "10 minutes, rehearsed by all four members"),
]
S8_RESULTS = (
    f"First results (test years 2005–2014): LSTM NSE 0.59, best simple model 0.56 (no clear difference). Onsets warned one day ahead: 19% with the rain table, 51% with AORC rain to midnight, {S4_PCT}% with real-time "
    "Stage IV; with the GEFS forecast, 39% and 37% at two and three days. Verdict: NOT YET. Targets:"
)
TARGETS = [
    ("Indicator", "Baseline", "First result", "Target"),
    ("Onsets warned 1 day ahead, real-time inputs", "19%", f"{S4_PCT}%", "≥ 70% (GO)"),
    ("False onset warnings", "59%", f"{pc(RUNG['lstm_wqaF_sub']['l1']['onset_false'])} (AORC + GEFS)", "≤ 30% (GO)"),
    ("Trainings per model", "3", "3", "5"),
]
REFS = [  # short IEEE form; only the cited ones are printed, renumbered by first citation
    "F. Kratzert et al., \"Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets,\" HESS, vol. 23, pp. 5089–5110, 2019.",
    "G. Nearing et al., \"Global prediction of extreme floods in ungauged watersheds,\" Nature, vol. 627, pp. 559–563, 2024.",
    "F. Kratzert et al., \"HESS Opinions: Never train an LSTM network on a single basin,\" HESS, vol. 28, pp. 4187–4201, 2024.",
    "A. J. Newman et al., \"Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA,\" HESS, vol. 19, pp. 209–223, 2015.",
    "N. Addor et al., \"The CAMELS data set: catchment attributes and meteorology for large-sample studies,\" HESS, vol. 21, pp. 5293–5313, 2017.",
    "H. Guan et al., \"GEFSv12 reforecast dataset for supporting subseasonal and hydrometeorological applications,\" Mon. Weather Rev., vol. 150, pp. 647–665, 2022.",
    "C. Homer et al., \"Completion of the 2011 National Land Cover Database for the conterminous United States,\" PE&RS, vol. 81, pp. 345–354, 2015.",
    "T. G. Farr et al., \"The Shuttle Radar Topography Mission,\" Rev. Geophys., vol. 45, RG2004, 2007.",
    "NOAA, \"Analysis of Record for Calibration (AORC) v1.1,\" AWS Open Data, accessed Oct. 7, 2026.",
    "NOAA, \"GEFSv12 reforecast, 2000–2019,\" AWS Open Data, accessed Oct. 7, 2026.",
    "NCEP, \"Stage IV hourly precipitation analysis,\" Iowa Environmental Mesonet archive, accessed Oct. 7, 2026.",
    "NASA LaRC, \"POWER daily point data (IMERG, MERRA-2),\" accessed Oct. 3, 2026.",
    "R. Gelaro et al., \"MERRA-2,\" J. Clim., vol. 30, pp. 5419–5454, 2017.",
    "C. W. Landsea and J. L. Franklin, \"Atlantic hurricane database uncertainty,\" Mon. Weather Rev., vol. 141, pp. 3576–3592, 2013.",
    "S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" Neural Comput., vol. 9, pp. 1735–1780, 1997.",
    "J. E. Nash and J. V. Sutcliffe, \"River flow forecasting through conceptual models, part I,\" J. Hydrol., vol. 10, pp. 282–290, 1970.",
    "H. V. Gupta et al., \"Decomposition of the mean squared error and NSE performance criteria,\" J. Hydrol., vol. 377, pp. 80–91, 2009.",
]


EXEC = (
    "Small rivers in the rainy south-eastern United States can flood within a day or two of heavy rain, so a useful warning must "
    "come at least a day ahead. This project asks what limits such a warning: the model, or the information it is given. We test "
    f"one long short-term memory network (LSTM) against simple models on twelve CAMELS-US basins ({AREA0}–{AREA1:,} km², "
    "1980–2014), forecasting river flow one to three days ahead and scoring only on years never used for training. We then add "
    "information step by step: rain up to the moment the forecast is made, real-time radar rain and an archived weather forecast. "
    "The gap: the CAMELS rain table closes its day at about 09:00, so rain falling later in the day is invisible to a forecast made "
    "at midnight; to our knowledge, no study has measured what this costs flood warnings on small basins. All data are public and "
    "free. Outputs: a curated database, reviewed code, basin maps and GIS layers, a report and a talk. Warnings are judged by "
    "criteria fixed on 3 October 2026, before any added input was tried. First results already exist (Section 8). Duration: three "
    "weeks, to " + _sched.d(_sched.DEADLINE, True) + "."
)
S2 = [
    "One LSTM trained on many basins learns behaviour that carries over between rivers [1]; a global LSTM forecasts extreme floods "
    "days ahead [2]. The CAMELS data sets [4], [5] are the standard benchmark. "
    "Gap: their rain table closes its day at about 09:00, not at midnight when a forecast is issued.",
]
STUDY = (
    f"Twelve CAMELS-US basins [4] in North Carolina, Virginia, Georgia, Florida and Mississippi ({AREA0}–{AREA1:,} km²; mountains, "
    f"Piedmont, southern lowlands; humid, rain-fed), picked by a fixed rule from {N_CAND} candidates before any model ran."
)
DATA_NOTE = (f"Note: all public and free (ERA5, IMERG, MERRA-2 also tested); no optical or SAR scene is processed.")
S61 = (
    "A forecast is issued at local midnight ending day t, for days t+1 to t+3, and reads only what exists then: flow up to day t, "
    "the weather table (day ends about 09:00), hourly rain to midnight and the forecast run at 00 UTC; a test checks that nothing "
    "later is read. Grid cells are weighted by their area inside each basin."
)
S62 = ("LSTM: daily weather, gauge, ten basin properties and the added input; simple models get the same as rain sums. "
       "Elevation, slope and land cover are summarised per basin.")
S63 = (
    "One LSTM [15] for all basins [1], against 'tomorrow = today', ridge regression and a random forest. Training "
    "1 Oct 1980 – 30 Sep 2000, validation to 30 Sep 2005 (all choices), test to 30 Sep 2014 (scored once). Three trainings from "
    "different seeds (five later), averaged; their spread shows the role of chance."
)
S64 = (
    "NSE = 1 − Σ(obs − fc)² / Σ(obs − mean)² against gauge flow, median over basins (1 perfect, 0 no better than the mean). "
    "An onset warned in time is a hit, else a miss; a warning without onset is false. A gain counts only if every training with "
    "the input beats every training without it and its control."
)
S65 = ("No operational use while NOT YET. Disclosure: Claude Code wrote the code under the team lead's direction; AI reviewers and "
       "each member check it.")
S8_TAIL = [
    "Timeline: week 1 (15–21 Oct) feedback, own checks; week 2 (22–28 Oct) five-seed runs, GIS layers; week 3 (29 Oct – 5 Nov) "
    "report, hand-in. Leads: " + _team.label("A") + ", data; " + _team.label("R2") + ", maps; "
    + _team.label("G1") + ", scoring; " + _team.label("G2") + ", models.",
]
S9 = (f"Data ({GB_KEPT} GB) and results stay in the project folder; a data record lists each product's source, clock and licence.")


# ---------------------------------------------------------------------------------------------------------------------------
# python-docx helpers: everything works on the template's own paragraph and row objects
# ---------------------------------------------------------------------------------------------------------------------------
def strip_ids(el):
    """Drop w14:paraId / w14:textId from a copied element (they must be unique; Word makes new ones)."""
    for e in el.iter():
        for a in list(e.attrib):
            if a.endswith("}paraId") or a.endswith("}textId"):
                del e.attrib[a]


def find(doc, prefix):
    hits = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"template paragraph {prefix!r} found {len(hits)} times (the template changed?)")
    return hits[0]


def set_text(p, text, plain=True):
    """Put text in the FIRST run of p (keeping its formatting), drop the other runs. plain: remove the placeholder look (italic, grey)."""
    runs = p.runs
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    r0 = runs[0]
    if plain and r0._r.rPr is not None:
        for tag in ("w:i", "w:iCs", "w:color"):
            for el in r0._r.rPr.findall(qn(tag)):
                r0._r.rPr.remove(el)
    r0.text = text
    return p


def clone_after(ref, like=None):
    """A copy of `like` (default: ref) inserted right after ref, as a new Paragraph."""
    new = copy.deepcopy((like if like is not None else ref)._p)
    strip_ids(new)
    ref._p.addnext(new)
    return Paragraph(new, ref._parent)


def delete(p):
    p._p.getparent().remove(p._p)


def set_cell(cell, text):
    p = cell.paragraphs[0]
    if p.runs:
        set_text(p, text, plain=False)
    else:
        p.add_run(text)


def add_table(doc, after_el, rows, widths):
    """A new table in the style of the template's data table, inserted right after the element after_el; returns its element."""
    style = doc.tables[1].style
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = style
    grid = t._tbl.tblGrid.findall(qn("w:gridCol"))
    for g, w in zip(grid, widths):
        g.set(qn("w:w"), str(int(w * 1440)))
    for r, vals in zip(t.rows, rows):
        for c, (cell, v) in enumerate(zip(r.cells, vals)):
            cell.width = Inches(widths[c])
            set_cell(cell, v)
    after_el.addnext(t._tbl)
    return t._tbl


def par_after_el(el, like):
    """A copy of paragraph `like` inserted right after the element el (a table)."""
    new = copy.deepcopy(like._p)
    strip_ids(new)
    el.addnext(new)
    return Paragraph(new, like._parent)


def words(text):
    return len(re.findall(r"\S+", text))


# ---------------------------------------------------------------------------------------------------------------------------
def build():
    doc = Document(str(TEMPLATE))
    TBL_DATA, TBL_DELIV = doc.tables[1], doc.tables[2]                          # held before new tables shift the numbering

    # ---- cover: the two title lines and the participants; instructor, date and session stay as underscores
    lines = [p for p in doc.paragraphs if p.text.startswith("____")]
    assert len(lines) == 2, "the template's two title lines were not found"
    set_text(lines[0], TITLE_1, plain=False)
    set_text(lines[1], TITLE_2, plain=False)
    part = doc.tables[0]
    for row, name, sid in zip(part.rows[1:5], PARTICIPANTS, STUDENT_IDS):
        set_cell(row.cells[1], name)                                              # rows 5 and 6 stay blank
        set_cell(row.cells[2], sid)
    set_text(find(doc, "Course Instructor:"), "Course Instructor: " + INSTRUCTOR, plain=False)
    set_text(find(doc, "Submission Date:"), "Submission Date: " + SUBMISSION_DATE, plain=False)
    set_text(find(doc, "Academic Session:"), "Academic Session: " + ACADEMIC_SESSION, plain=False)

    # ---- 1 Executive summary: the "[Insert ...]" paragraph takes the text, the pure instruction before it goes
    set_text(find(doc, "[Insert one concise summary paragraph]"), EXEC)
    delete(find(doc, "Summarize the problem, geographic context"))

    # ---- 2 Background and rationale
    ph = find(doc, "[Insert background")
    set_text(ph, S2[0])
    p = ph
    for t in S2[1:]:
        p = clone_after(p, ph)
        set_text(p, t)
    delete(find(doc, "Describe the scientific or operational challenge"))

    # ---- 3 Problem statement and gap
    p3 = find(doc, "[Insert problem statement and gap")
    set_text(p3, S3[0])
    p = p3
    for t in S3[1:]:
        p = clone_after(p, p3)
        set_text(p, t)
    delete(find(doc, "State the exact problem that current data"))

    # ---- 4 Aim, objectives, research questions
    set_text(find(doc, "Overall aim:"), AIM)
    objs = [find(doc, f"Objective {i}:") for i in (1, 2, 3, 4)]
    for p, t in zip(objs, OBJ):
        set_text(p, t)
    rq = find(doc, "Research questions/hypotheses")
    set_text(rq, RQ_HEAD)
    p = rq
    for t in RQ:
        p = clone_after(p, objs[3])                                               # a bullet like the objectives
        set_text(p, t)

    # ---- 5 Study area, figure, data table, note
    study = find(doc, "Define administrative and geographic boundaries")
    set_text(study, STUDY)
    # (Figure 1 left out for the three-page limit; figures/basin_maps.png)
    data = TBL_DATA
    set_cell(data.rows[0].cells[0], DATA_HEAD[0])                                 # "Data source (illustrative)" -> "Data source"
    rows = list(data.rows)
    for k in range(len(DATA) - (len(rows) - 1)):                                  # add rows, keeping the banding of the style
        src = rows[2] if k % 2 == 0 else rows[1]
        new = copy.deepcopy(src._tr)
        strip_ids(new)
        data._tbl.append(new)
    for row, vals in zip(data.rows[1:], DATA):
        for cell, v in zip(row.cells, vals):
            set_cell(cell, v)
    set_text(find(doc, "Note: Include only datasets"), DATA_NOTE)

    # ---- 6 Methodology
    set_text(find(doc, "Specify data selection criteria, cloud screening"), S61)
    set_text(find(doc, "Describe relevant spectral indices"), S62)
    set_text(find(doc, "Specify algorithm(s) and justification"), S63)
    set_text(find(doc, "List evaluation indicators appropriate"), S64)
    set_text(find(doc, "Explain how results will be interpreted"), S65)

    # ---- 7 Deliverables table
    deliv = TBL_DELIV
    for row, (a, b) in zip(deliv.rows[1:], DELIV):
        set_cell(row.cells[0], a)
        set_cell(row.cells[1], b)

    # ---- 8 Monitoring, evaluation and impact
    p8 = find(doc, "Define baseline conditions")
    set_text(p8, S8_RESULTS)
    p8.paragraph_format.keep_with_next = True
    tb = add_table(doc, p8._p, TARGETS, [2.9, 0.9, 1.6, 1.0])
    p = par_after_el(tb, p8)
    set_text(p, S8_TAIL[0])

    # ---- 9 Sustainability and data management
    set_text(find(doc, "State data storage, backup"), S9)

    # ---- renumber the citations in order of first appearance (IEEE), then write the list in that order
    order = []
    texts = []
    for el in doc.element.body.iter(qn("w:t")):
        texts.append(el)
        for m in re.finditer(r"\[(\d+)\]", el.text or ""):
            n = int(m.group(1))
            if n not in order:
                order.append(n)
    assert set(order) <= set(range(1, len(REFS) + 1)), f"cited {sorted(order)} but {len(REFS)} references"      # uncited references are left out
    new_no = {old: i for i, old in enumerate(order, 1)}
    for el in texts:
        if el.text and "[" in el.text:
            el.text = re.sub(r"\[(\d+)\]", lambda m: f"[{new_no[int(m.group(1))]}]", el.text)
    refs_in_order = [REFS[old - 1] for old in order]

    # ---- 10 References (IEEE, by hand)
    ph = find(doc, "[Provide full, verifiable references")
    p = ph
    for i, ref in enumerate(refs_in_order, 1):
        if i > 1:
            p = clone_after(p, ph)
        set_text(p, f"[{i}] {ref}")
        p.paragraph_format.left_indent = Inches(0.45)
        p.paragraph_format.first_line_indent = Inches(-0.45)

    # empty paragraphs between the sections (after the cover) are dropped for the three-page limit
    seen_h1 = False
    for p in list(doc.paragraphs):
        if p.style.name == "Heading 1":
            seen_h1 = True
        elif seen_h1 and not p.text.strip() and not p._p.xpath(".//w:drawing") and p._p.getparent().tag.endswith("body"):
            nxt = p._p.getnext()
            if nxt is None or not nxt.tag.endswith("sectPr"):
                delete(p)
    doc.core_properties.title = f"{TITLE_1} {TITLE_2}"
    doc.save(str(OUT))
    return OUT


# ---------------------------------------------------------------------------------------------------------------------------
def to_markdown(path):
    """The Markdown twin, written from the finished .docx (same text, same order)."""
    d = Document(str(path))
    from docx.table import Table
    out, first_pic = [], True
    in_cover = True

    def cells(t):
        return [[c.text.replace("|", "\\|").replace("\n", " ") for c in r.cells] for r in t.rows]

    for ch in d.element.body.iterchildren():
        tag = ch.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(ch, d)
            sty = p.style.name
            if ch.xpath(".//pic:pic"):
                if first_pic:
                    first_pic = False                                              # the banner of the cover
                else:
                    out.append(f"![Figure 1: study area, the 12 study basins](../figures/{FIGURE.name})")
                continue
            t = p.text.strip()
            if not t:
                continue
            if sty == "Heading 1":
                in_cover = False
                out.append(f"## {t}")
            elif sty == "Heading 2":
                out.append(f"### {t}")
            elif sty == "List Bullet":
                out.append(f"- {t}")
            elif sty == "Caption":
                out.append(f"*{FIG_LABEL.strip()}* {t[len(FIG_LABEL):]}" if t.startswith(FIG_LABEL) else f"*{t}*")
            elif in_cover:
                out.append(f"**{t}**" if p.runs and p.runs[0].bold else t)
            else:
                out.append(t)
        elif tag == "tbl":
            rows = cells(Table(ch, d))
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * len(rows[0]))
            out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
        out.append("")
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # a list item must not be glued to the paragraph before or after it
    text = re.sub(r"(?m)^(?!- )(.+)\n(- )", r"\1\n\n\2", text)
    MD.write_text(text.strip() + "\n", encoding="utf8")
    return MD


# ---------------------------------------------------------------------------------------------------------------------------
# The PDF twin: the finished .docx is turned into HTML in the template's own look (Times New Roman, its colours, sizes, spacing and
# table styles) and printed through the project's route, report_common.page() + print_pdf (headless Chrome), as the other builders do.
# ---------------------------------------------------------------------------------------------------------------------------
PDF_NAME = "AI_and_ALLMs_Proposal_Flood.pdf"
PDF_CSS = """
@page{size:8.5in 11in;margin:19.8mm 22.9mm 18.3mm}
body{background:#fff;color:#000;font:11pt/1.15 'Times New Roman',Times,'Liberation Serif',serif}
.prop p{margin:0 0 5pt;orphans:2;widows:2}
.prop h1,.prop h2{font-family:'Times New Roman',Times,'Liberation Serif',serif;color:#1B3E59;font-size:12pt;font-weight:bold;line-height:1.15;margin:0;break-after:avoid}
.prop h1{margin-top:14pt} .prop h2{margin-top:10pt}
.prop ul{margin:0 0 5pt;padding-left:18pt;list-style:disc} .prop li{margin:0;padding:0}
.prop img{max-width:100%;height:auto}
.prop table{border-collapse:collapse;font-size:inherit;margin:0 auto;table-layout:fixed}
.prop table.grid td{border:0.5pt solid #000;padding:4.5pt;vertical-align:middle;line-height:1.15;text-align:left}
.prop table.ls{border-top:1pt solid #4F81BD;border-bottom:1pt solid #4F81BD;color:#365F91;margin:6pt auto}
.prop table.ls td{border:0.5pt solid #A9BFDC;padding:2.5pt 5.4pt;vertical-align:middle;text-align:left;line-height:1.15;overflow-wrap:anywhere}
.prop table.ls td.c1{font-weight:bold}
.prop table.ls tr.h td{font-weight:bold;border-bottom:1pt solid #4F81BD}
.prop table.ls tr:nth-child(even) td{background:#D3DFEE}
.prop tr{break-inside:avoid}
.prop .ref{overflow-wrap:anywhere}
"""


def to_html(path):
    """HTML of the finished .docx (same text, same order), in the look of the template; see PDF_CSS."""
    import base64
    import html as _html
    from docx.table import Table
    d = Document(str(path))
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    BASE = {"Normal": dict(size=11, bold=False, italic=False, color=None),
            "List Bullet": dict(size=11, bold=False, italic=False, color=None),
            "Heading 1": dict(size=12, bold=True, italic=False, color="1B3E59"),       # the template sets 12 pt on the runs of its headings
            "Heading 2": dict(size=12, bold=True, italic=False, color="1B3E59"),
            "Caption": dict(size=9, bold=True, italic=False, color="4F81BD")}
    SPACE = {"Heading 1": (14, 0), "Heading 2": (8, 0)}                                # (before, after) in pt where the style differs from Normal's (0, 5)

    def val(el, tag, name="val"):
        e = None if el is None else el.find(W + tag)
        return None if e is None else e.get(W + name)

    def style_of(rpr, base):
        s = dict(base)
        if rpr is not None:
            for key, tag in (("bold", "b"), ("italic", "i")):
                e = rpr.find(W + tag)
                if e is not None:
                    s[key] = e.get(W + "val") not in ("0", "false")
            c = val(rpr, "color")
            if c and c != "auto":
                s["color"] = c
            sz = val(rpr, "sz")
            if sz:
                s["size"] = int(sz) / 2
        return s

    def css(s):
        out = f"font-size:{s['size']:g}pt;"
        if s["bold"]:
            out += "font-weight:bold;"
        if s["italic"]:
            out += "font-style:italic;"
        if s["color"]:
            out += f"color:#{s['color']};"
        return out

    def inner(p_el, base):
        """The runs of a paragraph as spans; returns (html, size of the first run)."""
        parts, first = [], None
        for r in p_el.findall(W + "r"):
            text = "".join(t.text or "" for t in r.findall(W + "t"))
            if not text:
                continue
            s = style_of(r.find(W + "rPr"), base)
            first = first or s
            parts.append(f'<span style="{css(s)}">{_html.escape(text)}</span>')
        return "".join(parts), (first or base)

    def pstyle(p_el, sty, first):
        ppr = p_el.find(W + "pPr")
        sp = None if ppr is None else ppr.find(W + "spacing")
        before, after = SPACE.get(sty, (0, 5))
        line = 276 / 240
        if sp is not None:
            if sp.get(W + "before") is not None:
                before = int(sp.get(W + "before")) / 20
            if sp.get(W + "after") is not None:
                after = int(sp.get(W + "after")) / 20
            if sp.get(W + "line") is not None and sp.get(W + "lineRule", "auto") == "auto":
                line = int(sp.get(W + "line")) / 240
        jc = val(ppr, "jc")
        align = {"center": "center", "both": "justify", "right": "right"}.get(jc, "left")
        ind = None if ppr is None else ppr.find(W + "ind")
        extra = ""
        if ind is not None:
            if ind.get(W + "left"):
                extra += f"margin-left:{int(ind.get(W + 'left')) / 20:g}pt;"
            if ind.get(W + "hanging"):
                extra += f"text-indent:-{int(ind.get(W + 'hanging')) / 20:g}pt;"
            elif ind.get(W + "firstLine"):
                extra += f"text-indent:{int(ind.get(W + 'firstLine')) / 20:g}pt;"
        if sty == "Caption":
            extra += "break-inside:avoid;"
        if ppr is not None and ppr.find(W + "keepNext") is not None:
            extra += "break-after:avoid;"
        mark = val(None if ppr is None else ppr.find(W + "rPr"), "sz")                 # the paragraph mark's size sets the height of an empty line
        size = first["size"] if first is not None else (int(mark) / 2 if mark else 11)
        return f"text-align:{align};margin:{before:g}pt 0 {after:g}pt;line-height:{1.15 * line:.3f};font-size:{size:g}pt;{extra}"

    def table_html(t):
        tbl = t._tbl
        grid = [int(g.get(W + "w")) / 20 for g in tbl.tblGrid.findall(W + "gridCol")]
        cls = "grid" if t.style.name == "Table Grid" else "ls"
        out = [f'<table class="{cls}" style="width:{sum(grid):.1f}pt"><colgroup>' + "".join(f'<col style="width:{g:.1f}pt">' for g in grid) + "</colgroup>"]
        for ri, tr in enumerate(tbl.findall(W + "tr")):
            out.append('<tr class="h">' if (ri == 0 and cls == "ls") else "<tr>")
            for ci, tc in enumerate(tr.findall(W + "tc")):
                tcpr = tc.find(W + "tcPr")
                fill = val(tcpr, "shd", "fill")
                st = f"background:#{fill};" if fill and fill != "auto" else ""
                body = []
                for p_el in tc.findall(W + "p"):
                    h, s = inner(p_el, BASE["Normal"])
                    body.append(f'<div style="font-size:{s["size"]:g}pt">{h or "&nbsp;"}</div>')
                out.append(f'<td class="{"c1" if ci == 0 else ""}" style="{st}">{"".join(body)}</td>')
            out.append("</tr>")
        out.append("</table>")
        return "".join(out)

    out, in_list, first_h1, first_pic = ['<div class="prop">'], False, True, True
    for ch in d.element.body.iterchildren():
        tag = ch.tag.split("}")[1]
        sty = None
        if tag == "p":
            p = Paragraph(ch, d)
            sty = p.style.name
        if in_list and sty != "List Bullet":
            out.append("</ul>")
            in_list = False
        if tag == "tbl":
            out.append(table_html(Table(ch, d)))
            continue
        if tag != "p":
            continue
        base = BASE.get(sty, BASE["Normal"])
        blip = ch.xpath(".//a:blip")
        if blip:
            blob = d.part.related_parts[blip[0].get(R + "embed")].blob
            cx = int(ch.xpath(".//wp:extent")[0].get("cx")) / 12700
            alt = _html.escape(ch.xpath(".//wp:docPr")[0].get("descr") or "")
            out.append(f'<p style="{pstyle(ch, sty, None)}"><img alt="{alt}" style="width:{cx:.1f}pt" '
                       f'src="data:image/png;base64,{base64.b64encode(blob).decode()}"></p>')
            continue
        html_runs, first = inner(ch, base)
        if sty in ("Heading 1", "Heading 2"):
            n = sty[-1]
            cls = ' class="pb"' if (n == "1" and first_h1) else ""
            first_h1 = first_h1 and n != "1"
            out.append(f"<h{n}{cls}>{_html.escape(p.text)}</h{n}>")
        elif sty == "List Bullet":
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f'<li style="font-size:{first["size"]:g}pt;line-height:1.15">{html_runs}</li>')
        else:
            ref = ' class="ref"' if re.match(r"\[\d+\] ", p.text) else ""
            out.append(f'<p{ref} style="{pstyle(ch, sty, first if html_runs else None)}">{html_runs or "&nbsp;"}</p>')
    if in_list:
        out.append("</ul>")
    out.append("</div>")
    return "\n".join(out)


def write_pdf(path):
    """Print the PDF twin through the project's route (report_common.page() + print_pdf). Returns (pdf path or None, number of pages or None)."""
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_common import page, pdf_pages, print_pdf
    out = PROP / PDF_NAME
    html = page(f"{TITLE_1} {TITLE_2}", to_html(path), css_extra=PDF_CSS)
    with tempfile.TemporaryDirectory() as folder:
        src = Path(folder) / "AI_and_ALLMs_Proposal_Flood.html"
        src.write_text(html, encoding="utf-8")
        if not print_pdf(src, out):
            return None, None
    return out, pdf_pages(out)


if __name__ == "__main__":
    out = build()
    print("wrote", out)
    if "--no-md" not in sys.argv:
        print("wrote", to_markdown(out))
    if "--no-pdf" not in sys.argv:
        pdf, n = write_pdf(out)
        print("wrote", pdf, f"({n} pages)" if n else "")
