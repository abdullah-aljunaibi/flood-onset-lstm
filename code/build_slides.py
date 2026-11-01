"""Build the slides of the 10-minute presentation: Flood_Slides.html, Flood_Slides.pdf (16:9, one slide per page),
Flood_Slides.pptx (the same slides, editable) and slides_notes.md (speaker notes).

Nothing is computed here. Every number on a slide and in the notes is read from results/*.json and results/*.csv and
formatted inside an f-string; wording that depends on an outcome is chosen from the stored numbers, so a rerun cannot leave
a false sentence behind. ONE list of slides (title, bullets, table, figure file, notes) feeds all four outputs, so they
cannot drift apart. The figures are the PNGs of figures/ as the experiment scripts drew them, placed as large as the slide
allows (a figure gets up to about 60% of the slide's width, so that it can be read from the back of a room). Where the claim
of a slide rests on one panel of a figure with several panels, the slide shows that panel alone, enlarged: an exact crop made
in memory with PIL (_crop: the cut runs through the blank space between the panels, never through ink; the PNG file is only
read), with a label under it that says what the picture shows and where the full figure is in the report (no file name stands
on a slide; the notes name the file). If a figure is redrawn with another layout the crop is refused and the whole figure is
shown.
A slide = one message (its title, a full sentence), one figure or small table, a few short bullets. Its notes are the short
sentences to say ('notes', sized to the slide's time budget), the sentence that hands over to the next presenter ('handover')
and facts for questions ('asked').
The talk is 13 slides and 10 minutes; it opens with the answer and ends on the key figure of the main finding
(figures/key_figure_talk.png, or figures/key_figure.png when the talk-size drawing is not there). After the last talk slide
come backup slides (footer 'backup', no time budget), shown only if a question asks for them.
Since 8 October 2026 the talk has two slides on the experiments of 7 and 8 October (rain known up to the issue time; a real
weather forecast) between the rain-timing slide and the closing slides; to make room the slides on the Transformer and on satellite
rain are backup slides (as is the new one on more training rivers), and the Transformer is said in one sentence on the slide on the
LSTM. Those new slides read their numbers through the shared readers of report_sections/s0_summary.py (era5_facts, aorc_facts,
stage4_facts, hindsight_facts, forecast_facts, rivers_facts: the readers the report reads its sections 4.6 to 4.8 with), so that slides
and report cannot disagree, and the fixed wordings of that module (the finding and its supporting line, 'an analysis made after the
fact', 'a moved operating point, not lost skill', 'the information, not the network', 'about a quarter', 'mixed', 'no clear
difference') are imported from it, not typed again - and, since the final critic's report of 8 October 2026, its REALTIME_WORDING (the Stage IV figure is a check on the
test years, not a service: 'with the real-time radar product in its place'), HINDSIGHT_WORDING (beside every 51%), NSE_FORECAST_WORDING (through nse_forecast_words: the forecast
lowers the 1-day NSE, the gains at 2 and 3 days are 'mixed') and REVIEW_WORDING (who checked what: the independent reviews, and what only the final critic agent verified).
The base is named wherever a forecast gain is given (the rain table alone, or the AORC rain to midnight); a slide may ask for a wider text column ('tx_w', inches).
A result that is missing while its experiment runs (rain_timing.json, lstm_lobo.json, and the files of the experiments of 7 and 8
October: aorc_rain_to_midnight.json, forecast_input.json, more_basins.json) shows as 'result pending'; files that
came late (day_windows.json, sources_checked.json, key_figure.json, same_misses.json, uncertainty.json, training_curves.json,
published_scores.json) are read by key and only if they are there: without one of them its sentence or slide is left out
and one line on stderr says so.
The builder warns on stderr when a text is estimated to run past its slide, when the sentences to say on a slide are too many
for its time (PACE words a second, the hand-over counted) or one of them is longer than MAX_SENTENCE words, when a result is
pending, when a crop was refused, and when the PPTX it wrote does not pass its own check.
It also writes results/slide_plan.json: slide by slide who presents, from when to when, with which figure, the sentences to say, the hand-over and the agreed
cut for a late talk, and the share of the ten minutes of each presenter (plan_shares); the team's presentation plan (code/build_presentation_plan.py) reads
that file, so that the plan and the notes cannot disagree.
Rebuild after any rerun: python code/build_slides.py
"""
import base64
import datetime
import html
import io
import json
import re
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.util import Inches, Pt

import team
from common import AXIS, BLUE, FIG, GRID, INK, INK2, MUTED, RES, ROOT, SURFACE
from report_common import CITED, LEAD_WORD, REFERENCES, TAG_COL, cite, count, fig, load_results, model_name, n2, page, pct, print_pdf, signed, table, tag
from report_sections import s0_summary as S0                    # the report's own readers and fixed wordings: the slides on the new experiments use them as they are

DECK = "Predicting river floods with AI"
LEADS = sorted(LEAD_WORD)
TOTAL_SECONDS = 600                             # the talk is 10 minutes; the time budgets of the talk slides must add up to this
TALK_SLIDES = 13                                # the talk has 13 slides; backup slides come after them and have no time budget
PACE = 2.3                                      # words a second that a speaker who is not a native speaker can say calmly; the builder warns
                                                # when the sentences of a slide, with its hand-over sentence, need more
MAX_SENTENCE = 30                               # words: the builder warns when one sentence to say is longer
LATE = 20                                       # seconds behind the clock at which the notes' plan for a late talk applies
P_CUT = 0.05                                    # 'p is small', as in section 4.2 of the report
MEM_TOL = 0.02                                  # memory test: a change in NSE below this is called 'no gain' (as in section 3 of the report)
Q, QC = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
UPPER = "an upper bound that cannot be had in practice"
FINDING = S0.FINDING[0].lower() + S0.FINDING[1:]       # the finding: fixed wording, word for word (the capital of the sentence lowered, as it stands inside a sentence)
FINDING_SHORT = "among the models tried, the limit is the inputs, not the network"                   # its short form, for a title or a line on a slide
SUPPORT = S0.SUPPORT[0].lower() + S0.SUPPORT[1:]       # the supporting line: fixed wording; the qualifier at its start belongs to it
NO_CLEAR, MIXED = S0.reading_word("no clear difference"), S0.reading_word("mixed")                 # 'no clear difference'; 'a mixed result'
AORC_WORDING, MOVED_POINT, INFO_NOT_NET = S0.AORC_WORDING, S0.MOVED_POINT, S0.INFO_NOT_NET
REALTIME_WORDING, HINDSIGHT_WORDING = S0.REALTIME_WORDING, S0.HINDSIGHT_WORDING     # what the Stage IV figure is (not a service); the bound of the review, beside every 51%
NSE_FORECAST_WORDING, REVIEW_WORDING = S0.NSE_FORECAST_WORDING, S0.REVIEW_WORDING    # what the forecast does to the NSE (never 'mixed at all three leads'); who checked what     # 'an analysis made after the fact'; 'a moved operating point, not lost skill'; 'the information, not the network'
RAIN_REF, RIVERS_REF, FORECAST_REF = "section 4.6", "section 4.7", "section 4.8"   # where the report has the experiments of 7 and 8 October 2026
NUMBER = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
STATE = {"NC": "North Carolina", "VA": "Virginia", "GA": "Georgia", "FL": "Florida", "MS": "Mississippi", "SC": "South Carolina",
         "AL": "Alabama", "TN": "Tennessee"}    # the states of the south-eastern basins, for saying a basin's name in full

# who presents what: the roles of PLAN.md, printed through code/team.py (a member's name once team.json gives it, the role of the plan
# until then). A presenter speaks for a slide; it does not mean that this member wrote or checked it. Since 8 October 2026 the project lead's
# choice is that Member 1 (RS/GIS, team lead), who opens and closes the talk and leads the question time, speaks for about half of the 600 seconds: slides 1
# (the cover), 2 (the data), 5 (the LSTM against the simple models), 9 (the rain-timing test), 12 (what it took) and 13 (the key figure); the share
# is worked out from the slides' own seconds (plan_shares), printed in slides_notes.md and written to results/slide_plan.json. RS/GIS member 2
# presents slides 8 (where the onset warnings fail) and 10 (the rain known up to the issue time), GNSS member 1 slides 3, 4 and 7 (the fair test, the
# number to beat, the warnings: the scoring role of the plan) and GNSS member 2 slides 6 (the forecast shown day by day) and 11 (the real weather
# forecast: the member who follows the trainings). The plan names no owner for the rain-timing test, for where the onset warnings fail or for the
# experiments of 7 and 8 October, and gives the flood analysis to the whole team. The owner of a slide is the 'owner' key of its spec; the backup slides
# keep the owner they had, the member whose part of the plan they belong to (they have no presenter in the talk and are shown only if a question asks).
JOB = {"A": "project, data, report", "R2": "maps and satellite inputs", "G1": "scoring, fair test, warnings",
       "G2": "training, hyper-parameter log, model comparison"}
ROLE = {k: (html.escape(team.label(k), quote=False), what) for k, what in JOB.items()}

# ---- the slide, in inches (16:9) and points: one geometry for the HTML and for the PPTX ----------------------------------
W, H = 13.333, 7.5
MX, TOP, BOTTOM, FOOT = 0.55, 0.42, 6.78, 6.98          # side margin; top of the title; lowest edge of the content; the footer rule
FIG_TOP, FIG_H, FIG_MAX_W, FIG_RIGHT = 0.30, 6.52, 8.40, 0.42      # 'side' slides: the figure takes the full height at the right, up to 63% of the width
GAP, WIDE_TX, TITLE_GAP = 0.36, 3.85, 0.24               # text-to-figure gap; width of the text column of a 'wide' slide (the figure gets 60%); space under a title
CAP_PT, CAP_LH, CAP_GAP = 11, 1.25, 0.07                 # the label under a figure (which part of which figure; what its marks mean) and the space above it
SNAP = 24                                                # pixels an edge of a crop may move to reach the blank space between two panels
TITLE_PT, TITLE_LH, BODY_PT, BODY_LH, TABLE_PT, TABLE_LH, FOOT_PT = 26, 1.14, 17, 1.30, 16, 1.25, 10
COVER_PT, COVER_LH, SUB_PT, SUB_LH, SMALL_PT, SMALL_LH = 40, 1.08, 20, 1.3, 16, 1.34
NOTE_LH = 1.3                                            # line height of the footnote under a table
COVER_TOP, COVER_GAPS = 0.50, (0.12, 0.20, 0.24, 0.26, 0.12)   # the first slide: its top, and the spaces after the top line, the title, the blue
                                                         # rule, the question and each small paragraph
COVER_FIG_W = (4.60, 4.20, 3.80, 3.40)                   # the widths its map may have: the widest beside which the text fits is used
BULLET_IN, BULLET_AFTER, CELL_PAD = 20, 9, (8, 5)        # points: hanging indent of a bullet, space after it, cell padding (side, top/bottom)
PPT_FONT = "Arial"                               # the PPTX names the one face every PC and Mac has (widths within 1% of the report's
                                                 # Helvetica Neue), so the deck looks the same wherever the team opens and edits it
FONT_FILES = (("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
              ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"))        # (regular, bold) on the Mac and on the PC: for measuring text

CSS_SLIDES = f"""
@page{{size:{W}in {H}in;margin:0}}
html,body{{margin:0;padding:0}}
.slide{{position:relative;width:{W}in;height:{H}in;overflow:hidden;box-sizing:border-box;background:var(--surface);break-after:page;break-inside:avoid}}
.slide:last-child{{break-after:auto}}
.slide h1{{font-size:{TITLE_PT}pt;line-height:{TITLE_LH};font-weight:700;margin:0 0 {TITLE_GAP}in;color:var(--ink)}}
.slide ul.pts{{font-size:{BODY_PT}pt;line-height:{BODY_LH};margin:0;padding-left:{BULLET_IN}pt;color:var(--ink)}}
.slide ul.pts li{{margin:0 0 {BULLET_AFTER}pt}} .slide ul.pts li::marker{{color:var(--blue)}}
.slide .tx{{position:absolute;left:{MX}in;top:{TOP}in}}
.slide .fg{{position:absolute}} .slide .fg figure{{margin:0}} .slide figcaption{{display:none}}
.slide .fg img{{display:block;width:100%;height:100%;object-fit:contain;object-position:right top}}
.slide .cp{{font-size:{CAP_PT}pt;line-height:{CAP_LH};color:var(--ink2);margin:{CAP_GAP}in 0 0}}
.slide .body{{position:absolute;left:{MX}in;right:{MX}in;top:{TOP}in;bottom:{H - BOTTOM:.2f}in}}
.slide.wide .body{{display:grid;grid-template-columns:{WIDE_TX}in minmax(0,1fr);grid-template-rows:auto minmax(0,1fr);column-gap:{GAP}in}}
.slide.wide h1{{grid-column:1 / 3}} .slide.wide .tx,.slide.wide .fg{{position:static;min-height:0}} .slide.wide .fg{{margin-left:auto}}
.slide table{{font-size:{TABLE_PT}pt;line-height:{TABLE_LH};margin:0 0 0.2in;color:var(--ink)}}
.slide th,.slide td{{padding:{CELL_PAD[1]}pt {CELL_PAD[0]}pt}} .slide th{{color:var(--ink2);font-weight:700}}
.slide tr:last-child td{{border-bottom:1px solid var(--grid)}}
.slide .tag{{font-size:{TABLE_PT}pt;border-width:2.4px;padding:0 7pt;font-weight:700}} .slide .nw{{white-space:nowrap}}
.slide .star{{font-size:{SMALL_PT}pt;line-height:{NOTE_LH};color:var(--ink2);margin:-0.08in 0 0.18in}}
.slide .pend{{width:100%;height:100%;box-sizing:border-box;border:2px dashed var(--axis);border-radius:10px;display:flex;align-items:center;
  justify-content:center;font-size:{SUB_PT}pt;color:var(--ink2)}}
.slide .ft{{position:absolute;left:{MX}in;right:{MX}in;top:{FOOT}in;border-top:1px solid var(--grid);padding-top:4pt;display:flex;
  justify-content:space-between;font-size:{FOOT_PT}pt;color:var(--muted)}}
.slide.cover .tx{{top:{COVER_TOP}in}}
.slide.cover .kick{{font-size:{SMALL_PT}pt;line-height:{SMALL_LH};color:var(--ink2);margin:0 0 {COVER_GAPS[0]}in}}
.slide.cover h1{{font-size:{COVER_PT}pt;line-height:{COVER_LH};margin:0 0 {COVER_GAPS[1]}in}}
.slide.cover .rule{{width:1.1in;height:0.05in;background:var(--blue);margin:0 0 {COVER_GAPS[2]}in}}
.slide.cover .sub{{font-size:{SUB_PT}pt;line-height:{SUB_LH};color:var(--ink);margin:0 0 {COVER_GAPS[3]}in}}
.slide.cover .small{{font-size:{SMALL_PT}pt;line-height:{SMALL_LH};color:var(--ink2);margin:0 0 {COVER_GAPS[4]}in}}
@media screen{{body{{background:#e9e8e3}} .slide{{margin:0.2in auto;box-shadow:0 1px 8px rgba(0,0,0,.2)}}}}
"""

lk = lambda k: f"l{k}"


# ---- words -------------------------------------------------------------------------------------------------------------
def _word(n):
    return NUMBER.get(n, str(n))


def _cap(s):
    return s[0].upper() + s[1:]


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _leads(ks):
    """[1] -> '1 day ahead'; [2, 3] -> '2 and 3 days ahead'; [1, 2, 3] -> '1, 2 and 3 days ahead'."""
    ks = sorted(ks)
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def _date(iso):
    d = datetime.date.fromisoformat(iso[:10])
    return f"{d.day} {d:%B %Y}"


def _slash(vals):
    return " / ".join(vals)


def _land(name):
    """'Northern lowlands and Piedmont' -> 'northern lowlands'; 'Mountains (above 600 m)' -> 'mountains'."""
    return name.split(" (")[0].split(" and ")[0].lower()


def _share(v):
    """A share of a gap as part of a sentence: '58%'; outside 0..1 the words say so."""
    return "none" if v <= 0 else "all" if v >= 1 else pct(v)


def _shares(vals):
    """Several shares in one sentence: '8% and 3%'; 'none' when none of them is above zero."""
    return "none" if all(v <= 0 for v in vals) else _and(_share(v) for v in vals)


def _fraction(v):
    """A share as people say it: 0.25 -> 'about a quarter'; a share that is near no simple fraction stays a percentage. The words and their
    limits are those of share_word in report_sections/s0_summary.py (imported, not copied), which the report and the question sheet use, so that
    the three documents choose the same words for the same number."""
    word = S0.share_word(v)
    return _share(v) if word.endswith("%") else word


def _escape(text):
    """A sentence taken word for word from a results file, safe inside the HTML of a slide or of the notes."""
    return html.escape(str(text), quote=False)


# ---- facts that more than one slide needs --------------------------------------------------------------------------------
def _reading(c):
    """One LSTM-against-simple-model comparison -> lstm / lstm_avg / lstm_med / tie / simple (the rule of section 4.2 of the
    report: below every seed + higher in most basins + small p), or no_call where the two models do not read the same weather."""
    if not c.get("like_for_like", True):
        return "no_call"
    most, few = c["n_basins_lstm_higher"] > c["n_basins"] / 2, c["n_basins_lstm_higher"] < c["n_basins"] / 2
    small_p, below = c["wilcoxon_p"] < P_CUT, c["baseline_against_seed_spread"] == "below every seed"
    if most and small_p:
        return "lstm" if below else "lstm_avg"
    if few and small_p:
        return "simple"
    return "lstm_med" if most and below else "tie"


def _all_basins(R):
    """The all-basin LSTM against the best simple model given the same inputs, and the status of milestone 2 of the plan for the two
    forecasts that could be run in practice: with the river gauge (the stronger one) and with past weather only."""
    A = R["lstm_all"]
    C = A["comparison"]
    cmp_ = lambda inp, k: C[inp]["baselines"][C[inp]["best_baseline_by_validation"][lk(k)]][lk(k)]
    cls = {inp: {k: _reading(cmp_(inp, k)) for k in LEADS} for inp in C}
    got, got_w = set(cls["wq"].values()), set(cls["w"].values())
    n_seeds = _word(len(A["sets"]["w"]["seeds"]))
    most = all(cmp_("w", k)["n_basins_lstm_higher"] > cmp_("w", k)["n_basins"] / 2 for k in LEADS)
    gauge = ("not met with the river gauge (no clear difference)" if got <= {"tie", "lstm_med"} else
             "not met with the river gauge (the LSTM is not ahead)" if got <= {"tie", "simple", "lstm_med"} else
             "met with the river gauge" if got == {"lstm"} else "met in part with the river gauge")
    past = (f"with past weather only the average of {n_seeds} trainings is ahead{' in most basins' if most else ''}, a single training is not "
            f"clearly ahead" if got_w == {"lstm_avg"} else
            "with past weather only it is met" if got_w == {"lstm"} else
            "with past weather only there is no clear difference either" if got_w <= {"tie", "lstm_med"} else
            "with past weather only the result is not the same at every lead time")
    return A, cmp_, cls, f"{gauge}; {past}"


def _model_gap(R):
    """The largest gap in median NSE between the LSTM and the best simple model over the like-for-like comparisons (the 0.06 of section
    4.3 of the report)."""
    _, cmp_, cls, _ = _all_basins(R)
    return max(abs(cmp_(inp, k)["difference_of_medians"]) for inp in cls for k in LEADS if cmp_(inp, k).get("like_for_like", True))


def _days(R):
    """(day of the data audit and of the choice of the basins, day of the experiments) as results/engine_review.json records them; the
    first is None in a file that holds only the day of the review."""
    ER = R["engine_review"]
    d = ER.get("dates") or {}
    audit = d.get("data_audit_and_basin_selection")
    return (_date(audit) if audit else None), _date(d.get("everything_else", ER["when"]))


def _revised():
    """The day this builder was last changed (the date of the file), as words: the day the slides were last revised. Whoever changes
    the builder after Claude Code must also change the words 'also by Claude Code' where this date is printed."""
    return _date(datetime.date.fromtimestamp((ROOT / "code" / "build_slides.py").stat().st_mtime).isoformat())


def _criteria(R):
    """The verdict criteria fixed in advance (results/flood_analysis.json): (share warned for GO, false warnings for GO, index for CAUTION)."""
    crit = R["flood_analysis"]["verdict"]["criteria"]
    return (float(re.search(r"pod\s*>=\s*([\d.]+)", crit["GO"]).group(1)), float(re.search(r"far\s*<=\s*([\d.]+)", crit["GO"]).group(1)),
            float(re.search(r"csi\s*>=\s*([\d.]+)", crit["CAUTION"]).group(1)))


def _verdict_of(o, crit):
    """GO / CAUTION / NOT YET for one stored table of hits, misses and false warnings (keys pod, far, csi), by the criteria of _criteria."""
    go_pod, go_far, ca_csi = crit
    return "GO" if o["pod"] >= go_pod and o["far"] <= go_far else "CAUTION" if o["csi"] >= ca_csi else "NOT YET"


def _windows(R):
    """The day ends measured against hourly rain (check_day_windows.py -> results/day_windows.json), read by key - or None when the file
    is missing, is a smoke test or lacks a key (the check came late and may still change). Times are local standard time."""
    Wd = R.get("day_windows")
    try:
        if not Wd or Wd.get("quick_mode"):
            return None
        dm, nl, im = (Wd["day_ends"][k] for k in ("daymet", "nldas", "imerg"))
        late = [float(v) for v in im["hours_late_in_this_check"].values()]
        doc = {b["short"]: int(b["documented"]) for b in im["per_basin"].values()}
        usual = max(set(doc.values()), key=list(doc.values()).count)
        return dict(daymet=dm["day_ends_local_rounded"], daymet_readings=list(dm["range_over_readings_local"]), daymet_basins=list(dm["basin_range_local"]),
                    n_readings=len(dm["median_by_reading"]), n_references=len({k.split("_")[0] for k in dm["median_by_reading"]}),
                    misfits=list(dm["basins_that_do_not_fit"]), nldas=nl["day_ends_local_rounded"], nldas_readings=list(nl["range_over_readings_local"]),
                    utc_day=f"{usual:02d}:00", utc_day_other={name: f"{h:02d}:00" for name, h in doc.items() if h != usual}, off=(min(late), max(late)),
                    figure="day_windows.png" if (FIG / "day_windows.png").exists() else None)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _utc_day_end(R):
    """When a UTC day ends in the basins, as a phrase - '19:00 local standard time (18:00 at Leaf River, MS)' - from the day-window check
    or, without it, from the note of the satellite experiment; None when neither says it."""
    Wd = _windows(R)
    if Wd:
        other = _and(f"{h} at {name}" for name, h in Wd["utc_day_other"].items())
        return f"{Wd['utc_day']} local standard time" + (f" ({other})" if other else "")
    for note in R.get("satellite_inputs", {}).get("notes", []):
        m = re.search(r"ends at (\d\d:\d\d) local standard time(?: \((\d\d:\d\d) for ([^)]+)\))?", note)
        if m:
            return f"{m.group(1)} local standard time" + (f" ({m.group(2)} at {m.group(3)})" if m.group(2) else "")
    return None


def _one_river(R, T):
    """The LSTM trained on one river alone (03_lstm.py): the outcome as one sentence to say, and the facts for questions."""
    S, LOG = R["lstm_single"], T["hparam_log"]
    TU, C, basin = S["tuning"], S["comparison"], S["basin"]
    first = LEADS[0]
    V = lambda inp, k: C[inp]["verdict"][lk(k)]
    call = lambda inp, k: "tie" if V(inp, k)["nse_call"].startswith("too close") else "lstm" if V(inp, k)["nse_call"].startswith("LSTM ahead") else "simple"
    cells = [(inp, k) for inp in ("w", "wq") for k in LEADS]
    ahead = [c for c in cells if call(*c) == "lstm"]
    scores = [v for p in TU["per_knob"].values() for v in p["val_nse_mean"]]
    mean = lambda v: sum(v) / len(v)
    rows_ = S["diagnostic_fixed_length"]["wq"]
    a = next(r for r in rows_ if r["stopping_point"].startswith("chosen"))
    z = max((r for r in rows_ if not r["stopping_point"].startswith("chosen")), key=lambda r: r["updates"])
    memorises = mean(z["train_nse"]) > mean(a["train_nse"]) and mean(z["val_nse"]) < mean(a["val_nse"])
    g1 = V("wq", first)
    fam = {"base_rf": "the random forest", "base_linreg": "the ridge regression", "base_persistence": Q}
    rival = next(v for stem, v in fam.items() if g1["best_baseline"].startswith(stem))
    state1 = {"simple": "clearly behind", "lstm": "ahead", "tie": "not clearly different"}[call("wq", first)]
    n_ahead = "none" if not ahead else "all" if len(ahead) == len(cells) else _word(len(ahead))
    say = f"On one river alone, the tuned LSTM was ahead of the simple models in {n_ahead} of {_word(len(cells))} comparisons."
    learn = (f"Trained longer, it only memorises: its NSE rises from {n2(mean(a['train_nse']))} to {n2(mean(z['train_nse']))} in the training years "
             f"and falls from {n2(mean(a['val_nse']))} to {n2(mean(z['val_nse']))} in the validation years." if memorises else
             f"Trained longer, its NSE goes from {n2(mean(a['train_nse']))} to {n2(mean(z['train_nse']))} in the training years and from "
             f"{n2(mean(a['val_nse']))} to {n2(mean(z['val_nse']))} in the validation years.")
    asked = (f"One river alone ({basin['short']}, in the {_land(basin['landscape'])}): four settings were tuned one at a time on the validation "
             f"years, and the best run scores an NSE of {n2(max(scores))} there; all {len(LOG)} training runs are in the hyper-parameter log, "
             f"results/hparam_log.csv. In the test years, with the river gauge, {LEAD_WORD[first]}, the LSTM scores {n2(g1['lstm_nse'])} against "
             f"{n2(g1['best_baseline_nse'])} for {rival}: {state1}. {learn}")
    tol, check, chosen = TU["noise_seed_spread"], TU["combination_check"], TU["chosen"]
    n_train = int(S["split"]["train"][1][:4]) - int(S["split"]["train"][0][:4])
    tuning = (f"The settings tuned on that one river are model size, learning rate, dropout and the days of history. The chosen one "
              f"({chosen['hidden']} memory cells, learning rate {chosen['lr']:g}, {chosen['warmup']} days of history) scores "
              f"{n2(check['chosen_val_nse_seed_mean'])} against {n2(check['base_val_nse_seed_mean'])} for the base setting; the seed spread is "
              f"{n2(tol, 3)} on the validation years and {n2(g1['seed_spread_used'])} in the test comparison. The basin has "
              f"{count(basin['area_km2'])} km&sup2; and {n_train} training years; &lsquo;trained longer&rsquo; means from update {a['updates']}, "
              f"where the validation years stopped it, to update {z['updates']} (report section 3).")
    return dict(say=say, asked=asked, tuning=tuning)


def _side(a, b):
    """Single trainings a against single trainings b, each given as [lowest, highest]: 'above' (every one of a above every one of b),
    'below' or 'overlap'."""
    if not a or not b:
        return "overlap"
    return "above" if min(a) > max(b) else "below" if max(a) < min(b) else "overlap"


ONSET_KEYS = {"onset_share_of_gap_closed", "onset_seed_range_with_nldas", "onset_seed_range_control", "onset_seed_range_base", "onset_warned_base",
              "onset_warned_with_nldas", "onset_warned_control", "onset_warned_perfect_forecast"}


def _timing_case(r, pre=""):
    """What the extra rain did in one row of the rain-timing test, for the median NSE (pre = '') or for the share of onsets warned
    (pre = 'onset_'): the single trainings (seeds) of the three models are set against each other, the rule of section 4.5 of the report.
    The test adds rain stamped up to the evening of the issue day; the control adds the same rain made one day older; base is the model
    without either.
      stamp          the test is clear of the base and of the control           -> the gain is the rain of the issue day (the time stamp)
      stamp_likely   clear of the base; the control is not; the two overlap     -> probably the time stamp, the control does not settle it
      product        clear of the base, and so is the control; the two overlap  -> a second rain product, not the time stamp
      product_only   the control is clear of the base, the test is not          -> a second rain product helps; nothing clear from the time stamp
      none / worse   neither is clear of the base / the test lies below it      -> no time-stamp gap shown"""
    with_, control, base = r[f"{pre}seed_range_with_nldas"], r[f"{pre}seed_range_control"], r.get(f"{pre}seed_range_base")
    gap = r["onset_warned_perfect_forecast"] - r["onset_warned_base"] if pre else r["gap"]
    test, ctrl = _side(with_, base), _side(control, base)
    if test == "below":
        return "worse"
    if test == "above" and gap > 0:
        if _side(with_, control) == "above":
            return "stamp"
        return "product" if ctrl == "above" else "stamp_likely"
    return "product_only" if ctrl == "above" else "none"


def _timing(R):
    """The rain-timing test (10_rain_timing.py), or None while its result is pending. For 1 day ahead with the river gauge: the row of the
    results file (r1) and what the rain of the issue day did to the onsets warned (on_case, the result the documents lead with) and to
    the median NSE (case); whether the NSE result is mixed (the median of 12 basins rises although no more than half of the basins
    gain); the shares of the gap closed 2 and 3 days ahead (NSE: far; onsets: far_on); the 1-day-ahead row without the river gauge."""
    X = R.get("rain_timing")
    if not X or X.get("quick_mode") or "wqn" not in X.get("closure", {}):
        return None
    rows = {r["lead"]: r for r in X["closure"]["wqn"]["by_lead"]}
    r1 = rows[LEADS[0]]
    if not all(ONSET_KEYS <= set(r) for r in rows.values()):     # a file written before the onset results were stored
        return None
    no_gauge = {r["lead"]: r for r in X["closure"].get("wn", {}).get("by_lead", [])}.get(LEADS[0])
    if no_gauge is not None and not ONSET_KEYS <= set(no_gauge):
        no_gauge = None
    n = len(X["stamps"]["per_basin"])
    few = r1["basins_better"] <= n / 2                             # the median rises although no more than half of the basins gain
    try:                                                         # basin by basin: NSE with the later rain minus NSE without it
        nse = lambda name: {g: b[lk(LEADS[0])]["nse"] for g, b in X["models"][name]["test"]["basins"].items()}
        with_, base = nse("wqn"), nse("wq")
        change = [with_[g] - base[g] for g in with_]
    except (KeyError, TypeError):
        change = []
    case = _timing_case(r1)
    return dict(X=X, r1=r1, case=case, on_case=_timing_case(r1, "onset_"), mixed=few and case in ("stamp", "stamp_likely", "product"),
                far=[rows[k]["share_of_gap_closed"] for k in LEADS[1:]], far_on=[rows[k]["onset_share_of_gap_closed"] for k in LEADS[1:]],
                no_gauge=no_gauge, n=n, few=few, basins=f"higher in {'only ' if few else ''}{r1['basins_better']} of {n} basins", change=change)


def _lobo(R):
    """Leave one basin out (05_lobo.py), 1 day ahead, or None while its result is pending."""
    X = R.get("lstm_lobo")
    if not X or X.get("quick_mode"):
        return None
    s = X["summary"][lk(LEADS[0])]
    loss = s["loss_from_not_seeing_the_basin"]["with_properties"]
    ratios = [b["test_over_training"] for b in X.get("flow_per_rain", {}).get("basins", {}).values() if "test_over_training" in b]
    return dict(inn=s["nse_median"]["in_training"], out=s["nse_median"]["held_out_with_properties"], n=s["n_basins"],
                loss=loss["median_nse_loss"],                    # the paired loss: the median over the basins of (in training minus held out)
                n_lower=loss["n_basins_lower_when_held_out"], n_clear=loss["n_basins_loss_larger_than_seed_range"],
                label=X["settings"]["input_label"], higher=X.get("higher_when_held_out") or [], ratios=ratios)


def _typeset(text):
    """A sentence from team.json or a results file as it is printed on a slide: safe inside HTML, with a typographic apostrophe."""
    return _escape(text).replace("'", "&rsquo;")


def _team_check(R):
    """The team's own checks: (still open?, sentence). While no member has entered a finished check in team.json, the sentence is the
    one results/engine_review.json supports ('not yet done'); once a member has, it is the sentence of code/team.py, the same words as
    in the report and on the question sheet. No builder ever sets a check to done."""
    state = team.checks_state()
    if state["n_done"]:
        return state["n_done"] < state["n"], _typeset(team.checks_sentence(short=True)) + "."
    text = R["engine_review"]["human_review"]
    if text.lower().startswith("not yet"):
        return True, "The team&rsquo;s own checks are not yet done."
    return False, f"The team&rsquo;s own check: {text}."


def _check_say(R):
    """The state of the team's own checks as it is said aloud: today's sentence while no check is entered in team.json; after that the
    sentence of code/team.py without its bracket of names and without its second half ('3 of the team's 4 own checks are done.'), so
    that the sentences to say keep their length. The slides print the whole sentence (_team_check)."""
    if not team.checks_state()["n_done"]:
        return _team_check(R)[1]
    return _typeset(team.checks_sentence(short=True).split(" (")[0].split(";")[0]) + "."


LEFT_OUT = []                                   # what was left out because a figure, a results file or a key was missing: one line each on stderr


def _left_out(line):
    if line not in LEFT_OUT:
        LEFT_OUT.append(line)


def _late(R, stem, used_for):
    """A results file that came late (results/<stem>.json), or None - with one line for stderr - when it is not there or is a smoke test."""
    X = R.get(stem)
    if not isinstance(X, dict) or not X:
        _left_out(f"results/{stem}.json was not found: {used_for} left out")
        return None
    if X.get("quick_mode"):
        _left_out(f"results/{stem}.json is a smoke test (quick_mode): {used_for} left out")
        return None
    return X


def _said(short):
    """A basin's short name as it is said aloud, (river, state): 'NE Cape Fear, NC' -> ('Northeast Cape Fear', 'North Carolina'). A part
    that is not in the small lists stays as it is; a name without a state gives ('...', '')."""
    long = {"NE": "Northeast", "SF": "South Fork", "NF": "North Fork"}
    river, _, state = short.partition(", ")
    first, _, rest = river.partition(" ")
    return (f"{long.get(first, first)} {rest}" if rest else river), STATE.get(state.strip(), state.strip())


def _answer(R):
    """What the project found, as flags that the cover, the slides on the models and the last slides share, so that they cannot disagree:
      tie      with the river gauge the LSTM and the best simple model show no clear difference at any lead time (the rule of _reading)
      not_yet  for onsets the verdict of both forecasts that could be run in practice is NOT YET at every lead time
      limit    every gain from the weather of the days being forecast (the upper bound) exceeds the largest gap between the LSTM and
               the best simple model where the two read the same weather: 'the limit is the information in the inputs, not the network'
      little   changing the kind of model changed little: tie, and the Transformer is not clearly different from the LSTM trained beside it
      support  the supporting line may be said: little, limit, and every kind of model gains from the river gauge"""
    A, cmp_, cls, _ = _all_basins(R)
    first = LEADS[0]
    V = R["flood_analysis"]["verdict"]["services"]
    gains = [A["decomposition"]["steps"]["perfect_forecast_with_gauge"][lk(k)]["nse_median_gain"] for k in LEADS]
    tc = R["model_comparison"]["comparison"]["test"]["skill"]
    tf_clear = (not tc["seed_ranges_overlap"]) and abs(tc["transformer_minus_lstm"]) > tc["largest_seed_spread"]
    S = {(r["model"], r["lead"]): r["nse_median"] for r in R["baselines"]["summary_table"]}
    pairs = {name: (S.get((f"base_{m}_w", first)), S.get((f"base_{m}_wq", first))) for name, m in (("ridge regression", "linreg"), ("random forest", "rf"))}
    pairs["LSTM"] = (cmp_("w", first)["lstm_nse_median"], cmp_("wq", first)["lstm_nse_median"])
    pairs = {k: v for k, v in pairs.items() if None not in v}  # median NSE 1 day ahead without and with the river gauge, kind by kind
    tie = set(cls["wq"].values()) <= {"tie", "lstm_med"}
    limit = min(gains) > _model_gap(R)
    all_gain = all(b > a for a, b in pairs.values())
    return dict(tie=tie, not_yet={V[m][lk(k)]["onset"]["pooled"] for m in ("lstm_w", "lstm_wq") for k in LEADS} == {"NOT YET"},
                limit=limit, little=tie and not tf_clear, support=tie and not tf_clear and limit and all_gain, tf_clear=tf_clear,
                tf_better=tf_clear and tc["transformer_minus_lstm"] > 0, pairs=pairs, all_gain=all_gain, gains=gains)


def _memory(R):
    """The memory test of the one-river LSTM (03_lstm.py): test NSE 1 day ahead against the days of history the trained model may read."""
    S = R["lstm_single"]
    MEM = S["memory"]
    ctx, warm = MEM["contexts_days"], S["tuning"]["chosen"]["warmup"]
    at = lambda inp, c: MEM["test_nse_by_lead"][inp][str(c)][0]                      # test NSE 1 day ahead with c days of history
    c0, c1, c2 = min(ctx), min(ctx, key=lambda c: abs(c - warm)), max(c for c in ctx if c <= 365)
    return dict(at=at, c=(c0, c1, c2), warm=warm, gain={inp: at(inp, c2) - at(inp, c1) for inp in ("w", "wq")},
                river=S["basin"]["name"].split(" AT ")[0].title() if " AT " in S["basin"]["name"] else S["basin"]["short"],
                numbers=(f"NSE {n2(at('w', c0))} with {c0} days of history, {n2(at('w', c1))} with {c1} and {n2(at('w', c2))} with {c2}"))


def _key_image():
    """The drawing of the key figure for a slide: the talk-size rendering (larger lettering, no title, no footnote) if it is there,
    else the figure of the report, else None."""
    return next((name for name in ("key_figure_talk.png", "key_figure.png") if (FIG / name).exists()), None)


def _key(R):
    """The key figure of the main finding (11_key_figure.py) as the last slide needs it, every number read by key from
    results/key_figure.json - or None, with one line for stderr, when the drawing or the results file is missing, is a smoke test or
    lacks a key. on: the share of onsets warned by the LSTM with the river gauge, with rain up to the evening of the issue day, and
    with the perfect weather forecast; spread: how far the kinds of model given past weather + river gauge lie apart; ordered: every
    such spread is smaller than every step of the LSTM to the right of it (what the picture is drawn to show). Since 8 October 2026 the figure has
    two blocks: aorc and real (the Stage IV diamonds: whole issue day, last hour out) are the AORC rung of the upper block, far the rows of the lower block
    (2 and 3 days ahead: {lead: {row key: (share of onsets warned, median NSE)}}), headline the sentence the file states for the figure, talk what the
    talk-size drawing leaves out and words the names of its rows; a file of before that day lacks them and the slide then leaves those parts out."""
    image = _key_image()
    if not image:
        _left_out("figures/key_figure_talk.png and figures/key_figure.png were not found: the talk ends on the slide without the key figure")
        return None
    K = _late(R, "key_figure", "the slide with the key figure")
    if K is None:
        return None
    try:
        if K["lead_drawn"] != LEADS[0]:
            raise KeyError(f"lead_drawn is {K['lead_drawn']}")
        level = {l["key"]: l for l in K["levels"]}
        on = [float(level[k]["lstm"]["onset_warned"]) for k in ("wq", "wqn", "wqf")]
        nse = [float(level[k]["lstm"]["nse"]) for k in ("wq", "wqn", "wqf")]
        sp = K["steps"]["spread_between_model_kinds"]["wq"]
        spread = dict(nse=float(sp["nse"]["spread"]), onsets=float(sp["onset_warned"]["spread"]), kinds=len(sp["models"]),
                      nse_range=(float(sp["nse"]["lowest"]), float(sp["nse"]["highest"])),
                      onsets_range=(float(sp["onset_warned"]["lowest"]), float(sp["onset_warned"]["highest"])))
        ordered = (spread["onsets"] < min(on[1] - on[0], on[2] - on[1]) and spread["nse"] < min(nse[1] - nse[0], nse[2] - nse[1]))
        rain = K["steps"]["rain_up_to_evening"]                  # with rain up to the evening the NSE result is 'mixed' when only the median rises
        mixed = (int(rain["n_basins_better"]), int(rain["n_basins"])) if rain.get("nse_reading") == "mixed" else None
        aorc = level.get("wqa")                                  # the rung of 8 October 2026: AORC rain up to midnight, with the Stage IV check beside it
        rt = (aorc or {}).get("real_time") or {}
        variants = {v["key"]: v for v in rt.get("variants", [])}
        real = ((float(variants["stage4"]["onset_warned"]), float(variants["stage4_no_last_hour"]["onset_warned"])) if "stage4" in variants and "stage4_no_last_hour" in variants else None)
        far = {}
        for lead in K.get("leads_drawn_below") or []:
            blk = (K.get("far") or {}).get(f"l{lead}") or {}
            far[int(lead)] = {l["key"]: (float(l["lstm"]["onset_warned"]), float(l["lstm"]["nse"])) for l in blk.get("levels", []) if l.get("lstm")}
        fig_ = K.get("figure") or {}
        return dict(image=image, talk_size=image == "key_figure_talk.png", on=on, nse=nse, spread=spread, ordered=ordered, n=int(K["n_basins"]),
                    years=tuple(K["test_years"]), n_onsets=int(K["n_onsets"][lk(LEADS[0])]), control=float(K["control"]["onset_warned"]),
                    seeds=len(level["wq"]["lstm"]["seeds"]), mixed=mixed, asterisk=str(fig_.get("asterisk", "")), footnote=str(fig_.get("footnote", "")),
                    aorc=float(aorc["lstm"]["onset_warned"]) if aorc else None, aorc_nse=float(aorc["lstm"]["nse"]) if aorc else None, real=real, far=far,
                    n_far={int(k[1:]): int(v) for k, v in K["n_onsets"].items() if k != lk(LEADS[0])}, headline=str(K.get("headline", "")),
                    talk=((fig_.get("renderings") or {}).get("talk") or {}), words=fig_.get("drawn_words") or {})
    except (KeyError, TypeError, IndexError, ValueError) as e:
        _left_out(f"results/key_figure.json lacks a key ({type(e).__name__}: {e}): the talk ends on the slide without the key figure")
        return None


# ---- the experiments of 7 and 8 October 2026: facts the new slides and the closing slides share -------------------------------------
_NEW = {}


def _new(R):
    """The experiments of 7 and 8 October 2026 as the slides need them: the rain of the whole issue day up to midnight with ERA5 (E1) and with AORC (E1b),
    the hindsight question and the real-time check with Stage IV (report section 4.6), more training rivers (E3, 4.7) and a real weather forecast (E2,
    4.8). Every number comes through the shared readers of report_sections/s0_summary.py - the readers the report reads its own sections with - so
    that it is the report's number, and every reading ('clear gain', 'mixed', a verdict) is the stored one. A reader whose file is missing gives
    None (a reader whose file is not in the expected form says so on stderr itself). W: the day ends of the day-window check, or None."""
    if id(R) not in _NEW:
        _NEW[id(R)] = dict(E1=S0.era5_facts(R), AO=S0.aorc_facts(R), S4=S0.stage4_facts(R), HS=S0.hindsight_facts(R), FC=S0.forecast_facts(R),
                           RV=S0.rivers_facts(R), W=_windows(R))
    return _NEW[id(R)]


def _pcts(d):
    """The shares of 1, 2 and 3 days ahead as one short run: {1: 0.41, 2: 0.14, 3: 0.08} -> '41 / 14 / 8%'."""
    return " / ".join(f"{100 * d[k]:.0f}" for k in LEADS) + "%"


def _no_leak(rv):
    """Does a review record say that it found no leakage? The test of S0.review_clause, which the report uses."""
    return bool(re.search(r"no leakage|no way for information from after the issue time|no hour after a window's end enters any input", str(rv.get("verdict", ""))))


def _counts_blocks(rv):
    """The counts blocks of a review record: its own and those of its later rounds (round_2, round_3, ...), each as the record stores it."""
    return [rv.get("counts") or {}] + [v.get("counts") or {} for k, v in rv.items() if k.startswith("round") and isinstance(v, dict)]


def _reviews(R):
    """The independent AI reviews of the four experiments of 7 and 8 October 2026 (results/review_*.json): days (as words), n (records), checks (listed
    checks in all, summed over every round of every record: a record may have several rounds, each with its own counts block), rounds (the number of
    counts blocks), clean (no check failed and no number differed), leak (every record says it found no leakage: the test of S0.review_clause, which the
    report uses). None - with one line on stderr - while no record exists."""
    nw = S0.new_work(R)
    if not nw:
        _left_out("results/review_*.json were not found: the slides do not mention the AI reviews of the experiments of 7 and 8 October")
        return None
    recs = [R[n] for n in S0.NEW_REVIEWS if isinstance(R.get(n), dict) and not R[n].get("quick_mode")]
    cs = [c for r in recs for c in _counts_blocks(r)]
    clean = all(c.get("failed") == 0 and c.get("numerical_discrepancies") == 0 for c in cs)
    leak = all(_no_leak(r) for r in recs)
    return dict(days=nw["days"], n=nw["n"], checks=sum(c.get("checks_listed", 0) for c in cs), rounds=len(cs), clean=clean, leak=leak and clean,
                open=all(str(r.get("human_review", "")).startswith("not yet done") for r in recs))


def _review_sentence(R):
    """What was checked by whom, for the notes: first the fixed wording REVIEW_WORDING of s0_summary (the independent reviews cover the engine and every experiment script;
    the real-time check on all onsets, the rain-clock diagnostic, the resampling ranges and the key figure of 8 October were verified by the final critic agent
    only), then what the four reviews of the training experiments found: how many listed checks (summed over all their rounds), that none failed and no number
    differed, and that no leakage was found - each only as far as the records say it. The state of the team's own checks is said by the sentence of team.json
    (_team_check), beside it. '' while no record exists."""
    V = _reviews(R)
    if V is None:
        return ""
    outcome = (("none failed, no number differed" + (", no leakage found" if V["leak"] else "")) if V["clean"] else "not every check passed (see the records)")
    return (f"{_cap(REVIEW_WORDING)}. The {_word(V['n'])} reviews of the training experiments of {V['days']} (AI agents, not team members) listed {count(V['checks'])} checks "
            f"in {_word(V['rounds'])} rounds: {outcome}.")


def _transformer(R):
    """The Transformer experiment (09_model_comparison.py) as the slide on the LSTM says it in one sentence and the backup slide in full: acc ('no
    clear difference', 'the Transformer is more accurate' or 'the LSTM is more accurate': the rule of the seeds), times ('roughly 10 to 15 times') and
    dearer (it takes longer to train). None while the comparison is missing."""
    X = R.get("model_comparison")
    if not isinstance(X, dict) or X.get("quick_mode"):
        return None
    try:
        t, ratio = X["comparison"]["test"]["skill"], X["cost_of_skill"]["transformer_over_lstm"]
        clear = (not t["seed_ranges_overlap"]) and abs(t["transformer_minus_lstm"]) > t["largest_seed_spread"]
        lo, hi = sorted((ratio["seconds_per_model"], ratio["seconds_per_update"]))
    except (KeyError, TypeError, ValueError):
        return None
    return dict(acc=(NO_CLEAR if not clear else "the Transformer is more accurate" if t["transformer_minus_lstm"] > 0 else
                     "the LSTM is more accurate"),
                times=(f"roughly {lo:.0f} to {hi:.0f} times" if f"{lo:.0f}" != f"{hi:.0f}" else f"about {lo:.0f} times"), dearer=lo > 1)


def _pending(key, owner, seconds, title, what, image=None):
    """A slide for an experiment whose results file is missing: its title says 'result pending', as the rain-timing slide does."""
    return dict(key=key, layout="wide", owner=owner, seconds=seconds, image=image, pending=f"{what}: result pending", figure_words="",
                title=f"{title}: result pending", bullets=["Result pending."],
                notes=[f"Result pending: {what} was not there when these notes were built."])


# ---- the slides ------------------------------------------------------------------------------------------------------------
def s_cover(R, T):
    A, ER = R["data_audit"], R["engine_review"]
    n = len(A["selected"])
    audit_day, run_day = _days(R)
    two_days = bool(audit_day) and audit_day != run_day
    when = f"data audit and choice of the basins: {audit_day}; experiments: {run_day}" if two_days else f"experiments: {run_day}"
    reviewed = ("Independent AI reviewer agents checked the shared engine and the experiments; a critic agent made the final checks. "       # the short form of REVIEW_WORDING
                if "AI reviewer" in ER["who"] else "")
    NWK = S0.new_work(R)                                          # the experiments of 7 and 8 October 2026 and their reviews (results/review_*.json)
    if NWK:
        when += f"; {_word(NWK['n'])} more experiments: {NWK['days']}"
    d_ = ER.get("dates") or {}                                   # one check was reviewed and rerun a day later
    late = (f"the day-window check was reviewed and rerun on {_date(d_['day_window_check_rerun'])}; "
            if str(d_.get("day_window_check_rerun", ""))[:10] > str(d_.get("everything_else", ER["when"]))[:10] else "")
    _, check = _team_check(R)
    today = _date(datetime.date.today().isoformat())
    ask = (f"Can an LSTM forecast river flow {LEADS[0]} to {LEAD_WORD[LEADS[-1]]} better than simple models, and could its warnings be "
           f"trusted?")
    ans = _answer(R)
    answered = ans["tie"] and ans["not_yet"]                     # the short answer may be given: no clear difference; onsets NOT YET
    # ---- the opening: the event that slide 'event' shows day by day, if it is the case the rule looks for (missed with the river gauge,
    #      warned with the upper bound); the same test as in s_event
    e, E = R["flood_analysis"].get("event_shown"), T["flood_events"]
    hook, hook_fact = [], ""
    if e:
        row = E[(E.gauge_id == e["gauge_id"]) & (E.onset_date == e["onset_date"])]
        warned = lambda m: bool(row[f"{m}_warned_l{LEADS[0]}_chosen_level"].iloc[0])
        if len(row) == 1 and not warned("lstm_wq") and warned("lstm_wqf"):
            onset = datetime.date.fromisoformat(e["onset_date"])
            river = _said(e["basin"])[0]
            water = river if river.split()[-1] in ("River", "Creek") else f"{river} river"
            storm = (f", when {STORM.get(e.get('cyclone_status'), 'tropical cyclone')} {e['cyclone_name'].title()} passed," if e.get("cyclone_name")
                     else "")
            hook = [f"On {onset.day} {onset:%B %Y}{storm} the {water} rose above its high-flow level. The evening before, our LSTM gave no "
                    f"warning."]
            hook_fact = (f"The opening sentence is the event of [slide:event] ({e['basin']}). &lsquo;Our LSTM gave no warning&rsquo; is read from "
                         f"results/flood_events.csv: the LSTM with the river gauge, {LEAD_WORD[LEADS[0]]}, at the warning level chosen on the "
                         f"validation years; the same LSTM given the perfect weather forecast, {UPPER}, warned.")
    course = _typeset(team.course_line(sep=" | ")).replace(" | ", " &middot; ")
    if not ER.get("late_additions"):
        _left_out("results/engine_review.json has no key late_additions: the sentence on what was added from the stored results is left out")
    says = hook + [f"Our question: {ask[0].lower() + ask[1:]}"]
    if answered:
        says.append("The short answer: with a river gauge, the LSTM and the best simple model show no clear difference; for the warnings that "
                    "matter, NOT YET.")
        if ans["limit"]:
            says.append(f"We can say why: {FINDING}.")
    else:
        says.append(f"The test: {n} rivers in the south-eastern United States.")
    return dict(
        key="cover", layout="cover", owner="A", seconds=55, image="overview_map.png",
        kicker=f"{course or 'AI course group project'} &middot; 10-minute talk &middot; {today}", title=DECK,
        sub=ask + ("<br><b>Our answer:</b> with a river gauge, no clear difference; for onsets, the warnings that matter, NOT YET."
                   + (f" <b>{_cap(FINDING_SHORT)}.</b>" if ans["limit"] else "") if answered else
                   f" {n} river basins in the south-eastern United States, {_years(A['period'])}."),
        small=[f"<b>Team:</b> {_typeset(team.names_line(sep=' | ')).replace(' | ', ' &middot; ')}. <b>Coding assistant:</b> Claude Code.",
               f"<b>Who did what:</b> Claude Code wrote all code and ran all experiments under the team lead&rsquo;s direction ({when}). "
               f"{reviewed}{check}"],
        notes=says + [f"To be plain: Claude Code wrote the code and ran the experiments under the team lead&rsquo;s direction"
                      + ("; independent AI reviewers checked the engine and the experiments. " if reviewed else ". ") + _check_say(R)],
        asked=[((f"The data audit and the choice of the basins were made on {audit_day}, the experiments were run on {run_day}; " if two_days else
                 f"The experiments were run on {run_day}; ") + late + f"these slides were last revised on {_revised()}, also by Claude Code (the day "
                f"code/build_slides.py was last changed), and built on {today}. "
                + (f"{_review_sentence(R)} The new slides read their numbers from those experiments&rsquo; results files ([slide:rain], [slide:forecast], [slide:rivers])."
                   if NWK and _reviews(R) else reviewed)).strip(),
               _escape(ER.get("late_additions", "")),             # what was added from the stored results on the last day, in the words of the file
               hook_fact,
               f"The test: {n} river basins in the south-eastern United States. An LSTM is a neural network with a memory; the data are daily, "
               f"{_years(A['period'])}.",
               ("The answer on the slide in full: with the river gauge the LSTM and the best simple model show no clear difference, so "
                f"milestone 2 of our plan is not met there ([slide:all]); the onset warnings are NOT YET at {_leads(LEADS)} ([slide:verdict])."
                + (" The picture of the finding is the last slide ([slide:key])." if ans["limit"] and _key(R) else "") if answered else "")])


def s_memory(R, T):
    """The memory test of the one-river LSTM. A backup slide; a talk slide only when the key figure is missing (see slides)."""
    M = _memory(R)
    at, (c0, c1, c2), warm, gain, river = M["at"], M["c"], M["warm"], M["gain"], M["river"]
    three = lambda inp: _and(n2(at(inp, c)) for c in (c0, c1, c2))
    if gain["w"] > MEM_TOL and gain["wq"] <= MEM_TOL:
        title = f"A river remembers past rain: without a river gauge, weather older than {c1} days still improves the forecast"
        gauge = (f"With the river gauge the score stays near {n2(at('wq', c1))} beyond {c1} days. A likely reading, not a tested one: the gauge "
                 f"already tells the model how wet the basin is.")
    elif gain["w"] > MEM_TOL:
        title = f"A river remembers past rain: weather older than {c1} days still improves the forecast, with and without a river gauge"
        gauge = f"With the river gauge the score also rises beyond {c1} days, from {n2(at('wq', c1))} to {n2(at('wq', c2))}."
    else:
        title = f"In our memory test, weather older than {c1} days adds less than {n2(MEM_TOL)} NSE: the river&rsquo;s memory lies in the last weeks"
        gauge = f"With the river gauge the three scores are {three('wq')}."
    in_talk = _key(R) is None                                     # as a talk slide it also gives the score; as a backup slide the data slide has done so
    nse = "1 is a perfect forecast, 0 is no better than always forecasting the average flow"
    return dict(
        key="memory", layout="wide", owner="A", seconds=40, image="lstm_memory.png", title=title,
        bullets=[f"An LSTM reads the days in order and carries a memory from one day to the next {cite('hochreiter1997')}.",
                 (f"The score is the NSE: {nse}." if in_talk else
                  f"Test years, one river ({river}), {LEAD_WORD[LEADS[0]]}. The grey part: less history than the model had in training."),
                 f"One river, {LEAD_WORD[LEADS[0]]}, past weather only: NSE <b>{n2(at('w', c0))}</b> with {c0} days of history, "
                 f"<b>{n2(at('w', c1))}</b> with {c1}, <b>{n2(at('w', c2))}</b> with {c2}."],
        notes=["A river remembers: its flow today depends on the rain of the past days to months.",
               "An LSTM reads the days one after another and carries a memory from day to day."]
              + ([f"Our score is the NSE: {nse}."] if in_talk else [])
              + [f"On one river, {LEAD_WORD[LEADS[0]]}, the NSE is {n2(at('w', c0))} with {c0} days of past weather, {n2(at('w', c1))} with {c1} days "
                 f"and {n2(at('w', c2))} with {c2} days."],
        asked=[f"The river is the {river}; the scores are for the test years. {gauge}",
               f"The grey part is outside what the model saw in training: it was trained with at least {warm} days of history.",
               "The basin stores past rain as wet soil and groundwater; the LSTM can learn that store from the data instead of being handed it."],
        figure_words="&lsquo;history&rsquo; is the number of past days the trained model may read before each forecast.")


def s_data(R, T):
    A, C = R["data_audit"], T["basin_audit"]
    sel = A["selected"]
    n = len(sel)
    per = {}
    for b in sel:
        per[b["landscape"]] = per.get(b["landscape"], 0) + 1
    even = len(set(per.values())) == 1
    each = (f"{_word(next(iter(per.values())))} in each of {_word(len(per))} landscapes" if even else
            _and(f"{v} in the {_land(k)}" for k, v in per.items()))
    start, stop = (datetime.date.fromisoformat(d) for d in A["period"])
    n_years = round(((stop - start).days + 1) / 365.25)
    lands = _and(sorted((_land(k) for k in per), key=lambda x: (not x.startswith("mountain"), x)))      # the order of the rows of maps
    M = _memory(R)
    first = LEADS[0]
    in_talk = _key(R) is None                                     # without the key figure the memory slide is in the talk and says these two things
    nse = "1 is a perfect forecast, 0 is no better than always forecasting the average flow"
    return dict(
        key="data", layout="side", owner="A", seconds=31 if in_talk else 37, image="basin_maps.png",    # (without the key figure the memory slide, which has two of
        # this slide's sentences, is in the talk, and this slide has the less to say)
        title=f"The {n} basins were chosen by rule, not by hand: {each}, {n_years} years of daily data",
        bullets=[f"CAMELS-US {cite('newman2015')}: {count(A['n_all_camels'])} US basins with daily weather over the basin and river flow at a "
                 f"gauge, {_years(A['period'])}.",
                 f"The rule: <b>{count(A['n_candidates'])}</b> south-eastern candidates &rarr; <b>{count(A['n_complete'])}</b> without a missing day "
                 f"&rarr; <b>{n}</b>, evenly spaced by area in each landscape (the rows of maps: {lands}).",
                 (f"{n_years} years, {_years(A['period'])}: {count(int(C.n_days.max()))} days per basin. Flow is in mm/day." if in_talk else
                  f"The score is the NSE: {nse}.")],
        notes=[f"The data set is CAMELS-US: {count(A['n_all_camels'])} US basins with daily weather and river flow.",
               f"A rule chose our {n}: {each}, with no missing day in {n_years} years. No flow statistic and no model score enters the rule."]
              + ([] if in_talk else ["Why an LSTM? A river remembers past rain, and an LSTM is a network with a memory.",
                                     f"Our score is the NSE: {nse}."]),
        handover="First: can the test be trusted?",
        asked=[f"The rule in numbers: of {count(A['n_candidates'])} south-eastern candidates, {count(A['n_complete'])} have no missing day; from "
               f"these the script took {each}, evenly spaced by area. Each basin has {n_years} years of daily data, {_years(A['period'])}: "
               f"{count(int(C.n_days.max()))} days. Flow is in mm/day.",
               f"The landscapes are {lands}; the weather is averaged over the basin and the flow is measured at a gauge.",
               "In each map the colour runs from that basin&rsquo;s own lowest to its own highest point, so colours cannot be compared between maps.",
               f"An LSTM reads the days one after another and carries a memory from day to day {cite('hochreiter1997')}. How far back the memory "
               f"matters, on one river ({M['river']}), {LEAD_WORD[first]}, past weather only, test years: {M['numbers']} ([slide:memory])."],
        figure_words="the maps come from the remote-sensing and GIS part (basin outlines over a 30 m elevation grid).")


def s_fair(R, T):
    FT, ER, split = R["fair_test"], R["engine_review"], R["baselines"]["split"]
    n = len(R["data_audit"]["selected"])
    job = {"train": ("Training years", "fit the models and the high-flow levels"),
           "val": ("Validation years", "choose each model&rsquo;s settings and where training stops"),
           "test": ("Test years", "are only scored: nothing is chosen from them")}
    exp, mea = FT["expected"]["past_weather_only"], FT["measured"]["past_weather_only"]
    zero = lambda v: n2(0.0 if round(v, 3) == 0 else v, 3)
    ok = bool(FT["passed"])
    cnt = ER["counts"]
    open_, check = _team_check(R)
    majors = ER.get("major_findings", [])
    limits = [m for m in majors if "NOT re-timed" in m]          # a finding met by stating a limit, not by a fix: the file says so itself
    n_code = cnt["major"] - len(limits)
    timing = ""                                                  # what the limit of the timing is: said if asked, in the words the slide used before
    if not ER["all_fixed"]:
        fixed = fixed_say = ", not all fixed yet"
    elif limits and n_code >= 0:
        other, one = ("the other is", "one") if len(limits) == 1 else ("the others are", _word(len(limits)))
        fixed = f" &ndash; {_word(n_code)} fixed in the code, {one} a limit of the timing"
        fixed_say = f": {_word(n_code)} were fixed in the code, {other} a limit of the timing"
        timing = (f"The limit of the timing: the cost comparison was not timed again on an idle machine, so its ratios are rough "
                  f"([slide:cost]).")
    else:
        fixed = fixed_say = ", all fixed"
    in_words = (["The major findings, in the words of results/engine_review.json: " + " ".join(f"({i}) {_escape(m)}" for i, m in enumerate(majors, 1))
                 + (f" And on whether all were fixed: {_escape(ER['all_fixed_note'])}" if ER.get("all_fixed_note") else "")] if majors else [])
    ai = "AI reviewer" in ER["who"]
    reviewers = f"{_cap(_word(len(ER['reviews'])))} independent {'AI reviewer agents' if ai else 'reviewers'}"
    note = next((x for x in R.get("lstm_all", {}).get("notes", []) if "climate averages" in x), "")
    keys, words, span = (re.search(r"\(((?:\w+, )*\w+):", note), re.search(r"\((?:\w+, )*\w+: ([^)]+)\)", note),
                         re.search(r"(\d{4}) to (?:\w+ )?(\d{4})", note))
    climate = (f" And {_word(len(keys.group(1).split(', ')))} of the fixed basin properties the all-river LSTM reads ({words.group(1)}) are CAMELS "
               f"averages of {span.group(1)}&ndash;{span.group(2)}, years that reach into the validation and test years: one number per basin, not "
               f"daily data.") if keys and words and span else ""
    none_done = team.checks_state()["n_done"] == 0
    return dict(
        key="fair", layout="table", owner="G1", seconds=44,
        title=("The test is fair: the test years chose no setting, and a check on made-up data found no leak from later days" if ok else
               "The fair-test check on made-up data did not pass: the scores must be read with care"),
        table=dict(header=["Period", "Years", "Its one job"], widths=[0.24, 0.16, 0.60],
                   rows=[[f"<b>{job[p][0]}</b>", _years(split[p]), job[p][1]] for p in ("train", "val", "test")]),
        bullets=[f"Four sets of inputs: past weather; past weather + river gauge; each again with a perfect weather forecast, {UPPER}.",
                 f"Made-up rivers with a known answer: an honest model can reach NSE {_and(n2(v) for v in exp)} at most; measured "
                 f"<b>{_and(zero(v) for v in mea)}</b>. {'All' if ok else 'Not all of the'} {len(FT['checks'])} checks passed.",
                 f"{reviewers} tried to break the shared code: {count(cnt['checks_passed'])} checks passed, {count(cnt['major'])} major "
                 f"problems{fixed}.",
                 check],
        notes=["Each period has one job: the training years fit the models, the validation years choose the settings, and the test years are only "
               "scored.",
               "The models get four sets of inputs: past weather, with or without the river gauge, and each again with a perfect weather forecast.",
               f"The perfect weather forecast is {UPPER}.",
               ("A check on made-up rivers with a known answer found no leak." if ok else
                f"A check on made-up rivers with a known answer did not pass: not all of its {len(FT['checks'])} checks held."),
               f"{reviewers} tried to break the shared code. "
               + (f"The team&rsquo;s own check, which the plan gives to {ROLE['G1'][0]}, is not yet done." if open_ and none_done else
                  _check_say(R))],
        asked=[f"The periods: training years {_years(split['train'])}, validation years {_years(split['val'])}, test years {_years(split['test'])}; "
               f"a forecast belongs to the period of the day it forecasts.",
               f"The made-up rivers: an honest model can reach an NSE of at most {_and(n2(v) for v in exp)} for {_leads(LEADS)}; the trainer "
               f"measured {_and(zero(v) for v in mea)}; {len(FT['checks'])} checks in all.",
               f"The reviewers report {count(cnt['checks_passed'])} checks passed and {count(cnt['major'])} major problems{fixed_say}. {timing}".strip(),
               f"The catch: all {n} basins are in all three periods, so this tests later years of rivers the model has seen.{climate}"]
              + in_words)


def s_simple(R, T):
    B = R["baselines"]
    M = B["models"]
    S = {(r["model"], r["lead"]): r for r in B["summary_table"]}
    best = {k: B["best_by_validation"][lk(k)]["without_weather_forecast"] for k in LEADS}
    first, last = LEADS[0], LEADS[-1]
    shown = [n for n in ("base_persistence", "base_linreg_w", "base_rf_w", "base_linreg_wq", "base_rf_wq") if n in M]
    label = lambda n: f"{QC} (river gauge only)" if M[n]["model"] == "persistence" else model_name(n)
    mark = lambda n, k, v: f"<b>{v}</b>" if best[k]["model"] == n else v
    rows = [[label(n)] + [mark(n, k, n2(S[n, k]["nse_median"])) for k in LEADS] + [pct(S[n, first]["onset_pod"])] for n in shown]
    p1, p3 = S["base_persistence", first], S["base_persistence", last]
    with_gauge = all(M[best[k]["model"]]["uses_river_gauge"] for k in LEADS)
    top = max(S[n, k]["onset_pod"] for n in M if not M[n]["uses_weather_forecast"] for k in LEADS)
    silent = all(S["base_persistence", k]["onset_pod"] == 0 for k in LEADS)
    share = sum(v["share_train_days_above"] for v in B["high_flow"].values()) / len(B["high_flow"])
    trees = sorted({M[n]["settings"]["n_trees"] for n in M if M[n]["model"] == "rf"})
    beat = _and(f"{n2(best[k]['test']['nse_median'])} {LEAD_WORD[k]}" for k in LEADS)
    return dict(
        key="simple", layout="table", owner="G1", seconds=38,
        title=(f"The number to beat comes from simple models: median NSE {_and(n2(best[k]['test']['nse_median']) for k in LEADS)} for "
               f"{_leads(LEADS)}{' with a river gauge' if with_gauge else ''}"),
        table=dict(header=["Simple model (inputs)"] + [f"NSE, {LEAD_WORD[k]}" for k in LEADS] + [f"Onsets warned, {LEAD_WORD[first]}"],
                   widths=[0.44, 0.13, 0.13, 0.13, 0.17], rows=rows),
        bullets=[f"Median NSE of the {len(B['high_flow'])} basins, test years. Bold: the number to beat, the simple model with the best score in the "
                 f"validation years.",
                 f"Onset: a high-flow day forecast while the river was still below the high-flow level, the level exceeded on {pct(share)} of "
                 f"days in the training years.",
                 f"No simple model warns of more than <b>{pct(top)}</b> of the onsets" + (f"; {Q} warns of none." if silent else ".")],
        notes=[f"{QC} repeats today&rsquo;s flow; ridge regression and random forest are fitted on hand-made weather features.",
               f"{'With the river gauge they' if with_gauge else 'They'} set the number to beat: a median NSE of "
               f"{n2(best[first]['test']['nse_median'])} {LEAD_WORD[first]} and {n2(best[last]['test']['nse_median'])} {LEAD_WORD[last]}.",
               "For warnings the honest test is the onset: a high-flow day forecast while the river was still below the high-flow level.",
               f"No simple model warns of more than {pct(top)} of the onsets."],
        handover="That is the number to beat. Does the LSTM beat it?",
        asked=[f"The number to beat at every lead time: a median NSE of {beat}.",
               f"Ridge regression is a weighted sum of {len(B['features']['w'])} features, such as rain sums of the last days; the random forest "
               f"averages {_and(trees)} decision trees on the same features. The number to beat is the model with the best score in the validation years.",
               f"{QC} scores {n2(p1['nse_median'])} {LEAD_WORD[first]} with nothing fitted, then falls to {n2(p3['nse_median'])} {LEAD_WORD[last]}"
               + ("; it warns of no onset, because its forecast is high only when the river already is." if silent else ".")])


STORM = {"HU": "hurricane", "TS": "tropical storm", "TD": "tropical depression", "EX": "former tropical cyclone", "LO": "remnant low",
         "SS": "subtropical storm", "SD": "subtropical depression"}                 # HURDAT2 status codes, worded as in section 5 of the report


def s_event(R, T):
    """One forecast against the observed river: the example event that 06_flood_analysis.py picks by rule and draws day by day."""
    A, E = R["flood_analysis"], T["flood_events"]
    e = A.get("event_shown")
    first = LEADS[0]
    GAUGE, UP = "lstm_wq", "lstm_wqf"
    base = dict(key="event", layout="side", owner="G2", seconds=50, aspect=1.2, pending="figures/flood_event.png: no example event",
                image="flood_event.png" if e and (FIG / "flood_event.png").exists() else None)
    if not base["image"]:
        return dict(base, title="One forecast against the observed river: no example event is stored in the results",
                    bullets=["The flood analysis picks one onset by rule and draws the forecasts against the observed river, day by day.",
                             "No onset fits that rule in the stored results, so there is no picture to show."],
                    notes=["The flood analysis picks one onset by rule and draws the forecasts against the observed river.",
                           "No onset fits the rule in the stored results, so this slide has no picture."])
    row = E[(E.gauge_id == e["gauge_id"]) & (E.onset_date == e["onset_date"])]
    known = len(row) == 1
    warned = lambda m: bool(row[f"{m}_warned_l{first}_chosen_level"].iloc[0])
    miss_g, hit_u = known and not warned(GAUGE), known and warned(UP)
    pattern = miss_g and hit_u                                    # the case the rule looks for: missed with the river gauge, caught by the upper bound
    onset = datetime.date.fromisoformat(e["onset_date"])
    crosses = f"On {onset.day} {onset:%B} the river crosses the high-flow level: the onset."
    if pattern:                                                  # the pivot of the talk, from the model to the information
        title = "One onset, day by day: the same LSTM gives no warning with the river gauge and a warning with the upper bound"
        outcome = [f"{crosses} The LSTM with the river gauge gives no warning.",
                   f"The same LSTM, given a perfect weather forecast, does. That is {UPPER}."]
    elif known:
        title = (f"One onset, day by day: {'no warning' if miss_g else 'a warning'} with the river gauge, {'a warning' if hit_u else 'no warning'} "
                 f"with the upper bound")
        outcome = [f"{crosses} The LSTM with the river gauge {'gives no warning' if miss_g else 'warns'}.",
                   f"The same LSTM, given a perfect weather forecast, {'warns' if hit_u else 'gives no warning'}. That is {UPPER}."]
    else:
        title = "One high-flow event, day by day: the forecasts against the observed river"
        outcome = [crosses, f"The dashed line is the same LSTM given a perfect weather forecast. That is {UPPER}."]
    storm = f"{STORM.get(e.get('cyclone_status'), 'tropical cyclone')} {e['cyclone_name'].title()}" if e.get("cyclone_name") else ""
    share = sum(v["share_train_days_above"] for v in R["baselines"]["high_flow"].values()) / len(R["baselines"]["high_flow"])
    went = f"{count(e['onsets_with_this_pattern'])} of the {count(e['onsets'])} onsets went this way"
    river, state = _said(e["basin"])
    stamped = f"{count(e['rain_target_day_mm'])} mm of rain are stamped on the onset day, so they were not yet in the model&rsquo;s inputs."
    return dict(
        base, title=title, crop=(0.0, 0.048, 1.0, 0.876), panel="the part without the title and the footnote",
        caption="Top: rain as stamped in the rain table. Bottom rows: was a warning given, day by day. Full figure: report, section 5.",
        bullets=[f"{e['basin']}, {onset:%B %Y}{f' ({storm})' if storm else ''}. Black: the observed river flow. Blue: the forecasts made the "
                 f"evening before.",
                 f"<b>{count(e['rain_target_day_mm'])} mm</b> of rain are stamped on the onset day, so they were not yet in the model&rsquo;s inputs. "
                 f"That does not mean &lsquo;not yet fallen&rsquo; ([slide:timing]).",
                 (f"Picked by rule, not a typical score: <b>{count(e['onsets_with_this_pattern'])}</b> of the {count(e['onsets'])} onsets went this "
                  f"way." if pattern else "Picked by rule, not a typical score.")],
        notes=[(f"Here is that event: {river}{', ' + state if state else ''}, {onset:%B %Y}." if pattern else
                f"One forecast, day by day: {river}{', ' + state if state else ''}, {onset:%B %Y}."),
               "Black is the observed river flow; blue are the forecasts made the evening before."]
              + outcome
              + [(f"Same network, more information: {stamped}" if pattern else _cap(stamped)),
                 (f"The example was picked by rule; {went}." if pattern else "The example was picked by rule; it is not a typical score.")],
        handover="One event is not a score. How often does a warning come?",
        asked=[(f"{_cap(storm)} passed {count(e['cyclone_km'])} km from the gauge. Each forecast is drawn on the day it is for." if storm else
                "Each forecast is drawn on the day it is for."),
               f"The rule that picked it: {_escape(e['rule'])}",
               f"&lsquo;Not yet in the inputs&rsquo; does not mean &lsquo;not yet fallen&rsquo;: the rain table&rsquo;s day seems to end in the "
               f"morning, so part of the rain stamped on an onset day can have fallen before the forecast was made ([slide:timing]). How much of the "
               f"{count(e['rain_target_day_mm'])} mm of this event had fallen by then was not measured.",
               f"The high-flow level of this river is {n2(e['high_flow_level_mm_day'], 1)} mm/day, the level exceeded on {pct(share)} of days in the "
               f"training years; the event lasts {count(e['duration_days'])} days; {count(e['rain_issue_day_mm'])} mm of rain are stamped on the day "
               f"before the onset.",
               "A warning is a forecast above the warning level. In the bottom rows a circle is a high-flow day with a warning (a hit), a cross a "
               "high-flow day without one (a miss), a triangle a warning on a day that stayed below the level (a false warning)."],
        figure_words="the shaded days are the high-flow event; the whole figure, with its title and footnote, is in the report, section 5.")


def s_all(R, T):
    A, cmp_, cls, m2 = _all_basins(R)
    n = len(A["basins"])
    first = LEADS[0]
    g = {k: cmp_("wq", k) for k in LEADS}
    w = {k: cmp_("w", k) for k in LEADS}
    got = set(cls["wq"].values())
    if got <= {"tie", "lstm_med"}:                                # the finding first, the status of the milestone second
        title = "With a river gauge, the LSTM and the best simple model show no clear difference: milestone 2 is not met there"
        verdict, m2_say = "no clear difference", "With the river gauge, milestone 2 of our plan is not met."
    elif got <= {"tie", "lstm_med", "simple"}:
        title = "With a river gauge, the LSTM does not beat the best simple model: milestone 2 is not met there"
        verdict, m2_say = "the LSTM is not ahead", "With the river gauge, milestone 2 of our plan is not met."
    elif got == {"lstm"}:
        title = f"With a river gauge, the LSTM beats the best simple model {_leads(LEADS)}: milestone 2 is met"
        verdict, m2_say = "the LSTM is ahead", "With the river gauge, milestone 2 of our plan is met."
    else:
        won = [k for k in LEADS if cls["wq"][k] == "lstm"]
        title = (f"With a river gauge, the LSTM beats the best simple model only {_leads(won)}: milestone 2 is met in part" if won else
                 "With a river gauge, only the averaged LSTM forecast is ahead of the best simple model: milestone 2 is met in part")
        verdict, m2_say = "a mixed result", "With the river gauge, milestone 2 of our plan is met in part."
    wins_w = _and(w[k]["n_basins_lstm_higher"] for k in LEADS)
    got_w = set(cls["w"].values())
    n_seeds = _word(len(A["sets"]["w"]["seeds"]))
    if got_w == {"lstm_avg"}:                                     # past: on the slide; past_say: the same finding, short enough to say
        past = f"Past weather only: the average of {n_seeds} trainings is ahead (in {wins_w} of {n} basins); a single training is not clearly ahead."
        past_say = f"With past weather only, the average of {n_seeds} trainings is ahead; a single training is not clearly ahead."
    elif got_w == {"lstm"}:
        past = f"Past weather only: the LSTM is ahead {_leads(LEADS)} (in {wins_w} of {n} basins)."
        past_say = f"With past weather only, the LSTM is ahead {_leads(LEADS)}."
    elif got_w <= {"tie", "lstm_med"}:
        past = f"Past weather only: no clear difference either (LSTM higher in {wins_w} of {n} basins)."
        past_say = "With past weather only there is no clear difference either."
    else:
        past = f"Past weather only: the LSTM is higher in {wins_w} of {n} basins; the result is not the same at 1, 2 and 3 days ahead."
        past_say = "With past weather only the result is mixed."
    # with the perfect weather forecast the two models read the same days of weather only at some lead times (like_for_like in lstm_all.json)
    fair = [k for k in LEADS if cmp_("wqf", k).get("like_for_like", True)]
    unfair = [k for k in LEADS if k not in fair]
    say = {"lstm": "the LSTM is ahead", "lstm_avg": "only the averaged LSTM forecast is ahead", "lstm_med": "the LSTM scores higher, but not clearly",
           "tie": "there is no clear difference", "simple": "the simple model is ahead"}
    groups = {}
    for k in fair:
        groups.setdefault(cls["wqf"][k], []).append(k)
    fair_txt = "; ".join(f"{_leads(ks)}, {say[c]} (in {_and(cmp_('wqf', k)['n_basins_lstm_higher'] for k in ks)} of {n} basins, p = "
                         f"{_and(n2(cmp_('wqf', k)['wilcoxon_p']) for k in ks)})" for c, ks in groups.items())
    same_days = ("the only lead time at which both models read the same days of weather" if len(fair) == 1 else
                 "the lead times at which both models read the same days of weather")
    upper = (f"With the perfect weather forecast, {UPPER}, and the river gauge: {fair_txt} &ndash; {same_days}, though not exactly the same "
             f"variables: the LSTM reads every weather variable of those days, the simple model only rain and temperature. p is two-sided."
             + (f" {_cap(_leads(unfair))} the LSTM reads the weather of all {_word(len(LEADS))} coming days and the simple model only up to the "
                f"day being forecast, so no winner is called there." if unfair else "") if fair else
             f"With the perfect weather forecast, {UPPER}, the LSTM reads more days of weather than the simple model at every lead time, "
             f"so no winner is called.")
    fc = A["decomposition"]["steps"]["perfect_forecast_with_gauge"]
    gauge1 = A["decomposition"]["steps"]["river_gauge_without_forecast"][lk(first)]
    gains = [fc[lk(k)]["nse_median_gain"] for k in LEADS]
    stronger = all(g[k]["lstm_nse_median"] > w[k]["lstm_nse_median"] for k in LEADS)
    practice = "the stronger of the two forecasts that could be run in practice" if stronger else "one of the two forecasts that could be run in practice"
    one = _one_river(R, T)
    marks = (f"Circle: LSTM. Triangle, square: best simple model (ridge regression, random forest). Dots: single basins. Black bars: range of "
             f"the {n_seeds} single trainings. Dashed: LSTM with past weather only.")
    later = LEADS[1:]
    tf = _transformer(R)                                          # the Transformer slide is a backup slide: its result is said here in one sentence
    if tf is None:
        tr_say = "Would another network do better?"
    elif tf["acc"] == NO_CLEAR:
        tr_say = (f"A Transformer shows {NO_CLEAR}, and takes {tf['times']} as long. What does change the forecast?" if tf["dearer"] else
                  f"A Transformer shows {NO_CLEAR}, and costs no more to train. What does change the forecast?")
    else:
        tr_say = f"A Transformer: {tf['acc']}, at {tf['times']} the training time. What does change the forecast?"
    return dict(
        key="all", layout="side", owner="A", seconds=52, image="lstm_all.png", title=title,
        crop=(0.012, 0.417, 0.545, 0.815), panel="the panel &lsquo;+ river gauge&rsquo; (bottom left)",
        caption=f"The panel &lsquo;+ river gauge&rsquo;, enlarged. {marks} Full figure: report, section 4.2.",
        caption_whole=f"Look at the panel &lsquo;+ river gauge&rsquo; (bottom left). {marks}",
        bullets=[f"Median NSE, {_slash(str(k) for k in LEADS)} days ahead: LSTM <b>{_slash(n2(g[k]['lstm_nse_median']) for k in LEADS)}</b>, best "
                 f"simple model <b>{_slash(n2(g[k]['baseline_nse_median']) for k in LEADS)}</b>.",
                 f"At the foot of the panel: the LSTM is higher in {_and(g[k]['n_basins_lstm_higher'] for k in LEADS)} of {n} basins.",
                 past],
        notes=[f"One LSTM for all {n} rivers, trained {n_seeds} times from different random starts; the forecasts are averaged.",
               f"With past weather and the river gauge, {LEAD_WORD[first]}, its median NSE is {n2(g[first]['lstm_nse_median'])}; the best simple "
               f"model reaches {n2(g[first]['baseline_nse_median'])}. The numbers for {_leads(later)} are on the slide.",
               f"The LSTM is higher in {_and(g[k]['n_basins_lstm_higher'] for k in LEADS)} of {n} basins: {verdict}. {m2_say}",
               past_say,
               tr_say],
        asked=[f"{one['say']} That is why one LSTM is trained on all {n} rivers.",
               (f"A Transformer was trained beside the LSTM with the same data, inputs, trainer and test: {tf['acc']}, {tf['times']} the training time ([slide:cost])." if tf else ""),
               f"Past weather + river gauge is {practice}; {_leads(later)} the LSTM&rsquo;s median NSE is "
               f"{_and(n2(g[k]['lstm_nse_median']) for k in later)} and the best simple model reaches "
               f"{_and(n2(g[k]['baseline_nse_median']) for k in later)}.",
               upper,
               f"What moves the score is the input: the weather of the days being forecast, an upper bound, would add {signed(min(gains))} to "
               f"{signed(max(gains))} NSE; the river gauge adds {signed(gauge1['nse_median_gain'])} NSE {LEAD_WORD[first]}. The LSTM also reads "
               f"{_word(len(A['static_inputs']))} fixed basin properties.",
               f"{past} Milestone 2 is {m2}. A difference is called clear only if the simple model lies below every single training of the "
               f"LSTM, the LSTM is higher in most basins and p is below {P_CUT} (two-sided).",
               one["asked"], one["tuning"]],
        figure_words="the numbers at the foot of the panel count the basins where the LSTM is higher; in the full figure (report section 4.2) "
                     "an asterisk marks the comparisons in which the two models do not read the same days of weather.")


def s_verdict(R, T):
    A = R["flood_analysis"]
    V, others, S = A["verdict"]["services"], A["verdict"]["other_models_pooled"], A["skill"]
    go_pod, go_far, ca_csi = _criteria(R)
    first = LEADS[0]
    GAUGE, TODAY, SERV = "lstm_wq", ("lstm_w", "lstm_wq"), ("lstm_w", "lstm_wq", "lstm_wqf")
    name = {"lstm_w": "Past weather only", "lstm_wq": "Past weather + river gauge", "lstm_wqf": "+ perfect weather forecast (upper bound)"}
    kind_word = {"all": "all high-flow days", "onset": "onsets"}
    verd = lambda m, k, kind: V[m][lk(k)][kind]["pooled"]
    firm = lambda m, k, kind: all(v == verd(m, k, kind) for v in V[m][lk(k)][kind]["single_seeds"].values())
    cell = lambda m, kind: " &nbsp; ".join(f"[[{verd(m, k, kind)}]]" + ("" if firm(m, k, kind) else "*") for k in LEADS)
    shaky = [(m, k, kind) for m in SERV for kind in ("all", "onset") for k in LEADS if not firm(m, k, kind)]

    def star_part(m, k, kind):                                   # how many single trainings get the verdict of the averaged forecast
        seeds = list(V[m][lk(k)][kind]["single_seeds"].values())
        same_n = sum(v == verd(m, k, kind) for v in seeds)
        return (f"{'none' if same_n == 0 else _word(same_n)} of the {_word(len(seeds))} single trainings get{'s' if same_n <= 1 else ''} "
                f"{verd(m, k, kind)} for {kind_word[kind]} {LEAD_WORD[k]}")

    several = len({m for m, _, _ in shaky}) > 1
    stars = "; ".join((f"{name[m].lstrip('+ ')}: " if several else "") + star_part(m, k, kind) for m, k, kind in shaky)
    at_level = "; ".join(f"at the high-flow level itself the averaged forecast gets {V[m][lk(k)][kind]['at_plain_level']} for {kind_word[kind]} "
                         f"{LEAD_WORD[k]}" for m, k, kind in shaky if V[m][lk(k)][kind].get("at_plain_level") not in (None, verd(m, k, kind)))
    plain = S[GAUGE]["plain"]["test"]["summary"][lk(first)]
    n_days, n_on = plain["all"]["hits"] + plain["all"]["misses"], plain["onset"]["hits"] + plain["onset"]["misses"]
    later = n_days - n_on
    g_all, g_on = V[GAUGE][lk(first)]["all"], V[GAUGE][lk(first)]["onset"]
    on_today = {verd(m, k, "onset") for m in TODAY for k in LEADS}
    all_today = {(m, k): verd(m, k, "all") for m in TODAY for k in LEADS}
    best_cells = [mk for mk, v in all_today.items() if v != "NOT YET"]
    if on_today == {"NOT YET"} and best_cells == [(GAUGE, first)]:
        title = (f"Warnings: {verd(GAUGE, first, 'all')} for high-flow days {LEAD_WORD[first]} with a river gauge, NOT YET for onsets, the warnings "
                 f"that matter")
    elif on_today == {"NOT YET"} and not best_cells:
        title = "Warnings: NOT YET in every case that could run today, for all high-flow days and for onsets"
    else:
        title = (f"Warnings with a river gauge: {_slash(verd(GAUGE, k, 'all') for k in LEADS)} for all high-flow days, "
                 f"{_slash(verd(GAUGE, k, 'onset') for k in LEADS)} for onsets ({_slash(str(k) for k in LEADS)} days ahead)")
    # the same count can turn up twice in the report: the high-flow days that are not onsets 1 day ahead, and the onsets at a later lead time
    n_on_later = {k: S[GAUGE]["plain"]["test"]["summary"][lk(k)]["onset"]["hits"] + S[GAUGE]["plain"]["test"]["summary"][lk(k)]["onset"]["misses"]
                  for k in LEADS[1:]}
    twin = "".join(f" By chance {count(later)} is also the number of onsets {LEAD_WORD[k]} (a forecast made {k} days before, while the river was "
                   f"still below the high-flow level)." for k, v in n_on_later.items() if v == later)
    ref = others.get("base_persistence", {}).get(lk(first), {}).get("all")
    only_upper_go = (any(verd("lstm_wqf", k, kind) == "GO" for k in LEADS for kind in ("all", "onset"))
                     and not any(verd(m, k, kind) == "GO" for m in TODAY for k in LEADS for kind in ("all", "onset")))
    same = ref == verd(GAUGE, first, "all")
    pers_short = (f"{QC} gets {'the same ' + ref if same else ref}: on {count(later)} of the {count(n_days)} high-flow days the river is "
                  f"already high.") if ref else ""
    return dict(
        key="verdict", layout="table", owner="G1", seconds=46, title=title,
        table=dict(header=["The LSTM, given", f"All high-flow days<br>{_leads(LEADS)}", f"Onsets<br>{_leads(LEADS)}"], widths=[0.26, 0.37, 0.37],
                   rows=[[f"<b>{name[m]}</b>", cell(m, "all"), cell(m, "onset")] for m in SERV],
                   note=f"* the verdict of the forecast averaged over the trainings (seeds): {stars}." if shaky else ""),
        bullets=[f"Fixed in advance: GO = at least {pct(go_pod)} of high-flow days warned and at most {pct(go_far)} false warnings; "
                 f"CAUTION = hits are at least {pct(ca_csi)} of hits + misses + false warnings.",
                 f"With the river gauge, {LEAD_WORD[first]}: <b>{pct(g_all['pod'])}</b> warned, <b>{pct(g_all['far'])}</b> false warnings. But "
                 f"<b>{pct(1 - g_on['pod'])}</b> of the onsets get no warning, and <b>{pct(g_on['far'])}</b> of the onset warnings are false.",
                 pers_short],
        notes=[f"The criteria were fixed before the analysis: GO needs at least {pct(go_pod)} of high-flow days warned and at most {pct(go_far)} "
               f"false warnings.",
               f"With the river gauge, {LEAD_WORD[first]}, the LSTM warns of {pct(g_all['pod'])} of the high-flow days: {verd(GAUGE, first, 'all')}.",
               (f"But {Q} gets the same verdict: most high-flow days come when the river is already high." if same and later > n_days / 2 else
                f"But {Q} gets the same verdict." if same else f"{QC} gets {ref} there." if ref else ""),
               "For onsets, the warnings that matter, the verdict is "
               + (f"{verd(GAUGE, first, 'onset')} at {_leads(LEADS)}" if len({verd(GAUGE, k, 'onset') for k in LEADS}) == 1 else
                  f"{_slash(verd(GAUGE, k, 'onset') for k in LEADS)} for {_leads(LEADS)}")
               + f": {LEAD_WORD[first]}, {pct(1 - g_on['pod'])} of the onsets get no warning.",
               ("Only the upper bound, the same LSTM given a perfect weather forecast, reaches GO." if only_upper_go else
                "The last row is the upper bound: the same LSTM given a perfect weather forecast.")],
        handover="So why are the onsets missed?",
        asked=["A model warns when its forecast is above the warning level. Share warned is the share of high-flow days with a warning; false "
               f"warnings are warnings not followed by a high-flow day. CAUTION needs hits to be at least {pct(ca_csi)} of hits, misses and false "
               f"warnings together.",
               f"With the river gauge, {LEAD_WORD[first]}, {pct(g_all['far'])} of the LSTM&rsquo;s warnings for high-flow days are false.",
               f"{count(later)} of the {count(n_days)} high-flow days of the test years are the second or a later day of an event, when the gauge "
               f"already shows a high river; {LEAD_WORD[first]} there are {count(n_on)} onsets, and {pct(g_on['far'])} of the onset warnings with "
               f"the river gauge are false." + twin,
               (f"Marked *: the verdict is that of the forecast averaged over the trainings; {stars}{'; ' + at_level if at_level else ''}. Every "
                f"other verdict in the table is the same for every single training." if shaky else
                "Every single training gets the same verdict as the averaged forecast, in every cell of the table.")],
        figure_words="in the table the three labels of a cell are for 1, 2 and 3 days ahead.")


def s_fail(R, T):
    A = R["flood_analysis"]
    X, GAUGE = A["failures"], "lstm_wq"
    first = LEADS[0]
    on = lambda part, m, c: X[part][m]["tuned"][c]["onset"]
    bins = [c for c in X["by_rain_timing"][GAUGE]["tuned"] if c.endswith("%")]
    low, top = bins[0], bins[-1]
    rt = {c: on("by_rain_timing", GAUGE, c) for c in bins}
    res = X["perfect_forecast_rescue"]["tuned"]
    ra, rc = res["all_onsets"], res["by_rain_class"]
    seasons = list(X["by_season"][GAUGE]["tuned"])
    s_lo, s_hi = (f(on("by_season", GAUGE, c)["pod"] for c in seasons) for f in (min, max))
    trop = on("by_storm_type", GAUGE, "tropical")
    n_trop = trop["hits"] + trop["misses"]
    falls = rt[top]["pod"] < rt[low]["pod"]
    dash = lambda s: s.replace("-", "&ndash;")
    title = ("Onset warnings fail when the rain behind the rise is not yet in the model&rsquo;s inputs" if falls else
             "Rain timing does not explain where the onset warnings fail")
    caught = ra["of_those_caught_with_perfect_forecast"]
    in_groups = sum(rc[c]["onsets"] for c in bins)
    left_out = next((c for c in rc if c not in bins), "")
    marks = ("Bar height: share of onsets warned 1 day ahead, 0 to 100% (a grid line every 20%). Dark: LSTM with the river gauge. Light: upper "
             "bound. Red lines: share of that model&rsquo;s onset warnings that were false.")
    return dict(
        key="fail", layout="side", owner="R2", seconds=33, image="flood_failures.png", title=title,
        crop=(0.539, 0.395, 1.0, 0.783), panel="the panel &lsquo;By rain timing&rsquo; (bottom right)",
        caption=f"The panel &lsquo;By rain timing&rsquo;, enlarged. {marks} Full figure: report, section 5.",
        caption_whole=f"Look at the panel &lsquo;By rain timing&rsquo; (bottom right). {marks}",
        bullets=["The onsets in four groups: how much of the two days&rsquo; rain is stamped on the target day, and so was not yet in the "
                 "model&rsquo;s inputs.",
                 f"Right-hand group, {dash(top)} (<b>{count(rc[top]['onsets'])}</b> of {count(ra['onsets'])} onsets): the LSTM with the river gauge "
                 f"warns of <b>{pct(rt[top]['pod'])}</b>. Left-hand group ({count(rc[low]['onsets'])} onsets): <b>{pct(rt[low]['pod'])}</b>.",
                 "&lsquo;Not yet in the inputs&rsquo; does not mean &lsquo;not yet fallen&rsquo;: next slide."],
        notes=[f"Look at the right-hand group: {count(rc[top]['onsets'])} of the {count(ra['onsets'])} onsets.",
               f"There, {'most' if float(re.match(r'[0-9.]+', top).group()) >= 50 else dash(top)} of the two days&rsquo; rain is stamped on the "
               f"target day, so it was not yet in the model&rsquo;s inputs.",
               f"The LSTM warns of {pct(rt[top]['pod'])} of them. In the left-hand group, where most of the rain is already in its inputs, it "
               f"warns of {pct(rt[low]['pod'])}.",
               "Careful: not yet in the inputs does not mean not yet fallen.",
               "So when did that rain fall?"],
        asked=[f"Season by season, {pct(s_lo)} to {pct(s_hi)} of the onsets are warned: {'no season is easy' if s_hi < 0.5 else 'the seasons differ'}. "
               + ("The clear pattern is rain timing." if falls else "Rain timing shows no such pattern either."),
               f"Given the weather of the target day, the upper bound, the same LSTM catches {count(caught)} of the {count(ra['missed_with_gauge'])} "
               f"missed onsets ({pct(caught / ra['missed_with_gauge'])}).",
               f"With a tropical cyclone nearby {pct(trop['pod'])} of the onsets are warned, but of only {n_trop} onsets. The scores are for "
               f"{LEAD_WORD[first]}, the LSTM with the river gauge, at the warning level chosen on the validation years; "
               f"{count(rc[low]['onsets'])} onsets have most of their rain already in the inputs.",
               f"The four groups hold {count(in_groups)} of the {count(ra['onsets'])} onsets"
               + (f"; the {count(ra['onsets'] - in_groups)} with {left_out} are left out." if left_out and ra["onsets"] > in_groups else ".")
               + " The full figure in the report also shows the onsets by season, by storm type and by basin size."],
        if_late=((f"Onsets are missed where the rain behind the rise is not yet in the inputs: {pct(rt[top]['pod'])} are warned there, against {pct(rt[low]['pod'])} where most of the "
                  f"rain is in. Not yet in the inputs does not mean not yet fallen; the details are in the report, section 5.") if falls else
                 "Rain timing does not explain where the onset warnings fail; the details are in the report, section 5."),
        figure_words="light bars are the upper bound; red lines are that model&rsquo;s false warnings; the full figure, with the onsets by "
                     "season, by storm type and by basin size, is in the report, section 5.")


def s_timing(R, T):
    """When is rain in the model's inputs? The day stamp of the rain table (an inference from the data) and the rain-timing test. The
    onsets warned lead, as on page 1 of the report; the NSE result follows. The figure is the picture of the day windows if that check
    has been run (results/day_windows.json, read by key), otherwise the figure of the rain-timing test."""
    t = _timing(R)
    first = LEADS[0]
    Wd = _windows(R)
    SC = R.get("sources_checked") or {}
    windows_fig = Wd["figure"] if Wd else None
    stamp = [f"The data indicate that our rain table&rsquo;s day ends in the morning" + (f", at about {Wd['daymet']}." if Wd else ".")
             + " That is an inference from our own data.",
             "When a forecast is made in the evening, the rain since the morning has fallen but is not yet in the model&rsquo;s inputs."]
    if windows_fig:
        words = ("in the figure, &lsquo;rain table (Daymet)&rsquo; is the rain made from rain gauges that every model reads (it is not the river "
                 "gauge); the pale bar, &lsquo;fallen, but stamped day t+1&rsquo;, is rain that is not yet in the model&rsquo;s inputs; NLDAS is the "
                 "second rain product that the test adds; the small ticks show where the day ends in each basin; the issue day is the day the "
                 "forecast is made.")
    else:
        words = ("in the figure, &lsquo;rain table (Daymet)&rsquo; is the rain made from rain gauges (it is not the river gauge); NLDAS is a second "
                 "rain product whose day ends in the evening of the issue day, the day the forecast is made; left: the day stamp; right: the test "
                 "with the river gauge.")
    base = dict(key="timing", layout="wide", owner="A", seconds=54, figure_words=words, aspect=7.6 / 3.5,      # the figure size of 10_rain_timing.py
                image=windows_fig or ("rain_timing.png" if (FIG / "rain_timing.png").exists() else None),
                pending="figures/rain_timing.png: result pending")
    if t is None:
        return dict(base, image=None, title="When is rain &lsquo;known&rsquo;? The rain-timing test is still running: result pending",
                    bullets=["The data indicate that the rain table stamps rain by observer day: rain stamped &lsquo;day d&rsquo; fell mostly between "
                             "the morning of day d&minus;1 and the morning of day d.",
                             "The test: the LSTM also reads rain stamped up to the evening of the day the forecast is made, with a control.",
                             "Result pending."],
                    notes=stamp + ["The test gives the LSTM a second rain product that is stamped up to the evening of the day the forecast is made, "
                                   "and repeats it with that rain made one day older as a control.",
                                   "Result pending: the experiment was still running when these notes were built."])
    X, r1, case, on_case = t["X"], t["r1"], t["case"], t["on_case"]
    n, basins, change, far, far_on = t["n"], t["basins"], t["change"], t["far"], t["far_on"]
    dn = X["stamps"]["median"]["daymet_vs_nldas"]
    share1, on_share = r1["share_of_gap_closed"], r1["onset_share_of_gap_closed"]
    lead1, later, evening_rain = _cap(LEAD_WORD[first]), _leads(LEADS[1:]), "rain stamped up to the evening of the issue day"
    on_base, on_with, on_ctrl, on_top = (pct(r1[k]) for k in ("onset_warned_base", "onset_warned_with_nldas", "onset_warned_control",
                                                                "onset_warned_perfect_forecast"))
    rng = lambda r, key: f"{pct(r[key][0], 1)} to {pct(r[key][1], 1)}"

    # ---- the median NSE: clear, mixed (the median rises, most basins do not gain) or nothing
    if t["mixed"]:
        nse_title = "NSE: mixed"
        nse_slide = (f"NSE: mixed. Median <b>{n2(r1['nse_base'])} &rarr; {n2(r1['nse_with_nldas'])}</b>, but higher in only "
                     f"{r1['basins_better']} of {n} basins.")
        nse_say = f"In NSE the result is mixed: the median rises, but only {r1['basins_better']} of {n} basins gain."
    elif case in ("stamp", "stamp_likely"):
        nse_title = f"NSE: {_share(share1)} of the gap"
        nse_slide = (f"NSE: median <b>{n2(r1['nse_base'])} &rarr; {n2(r1['nse_with_nldas'])}</b>, {_share(share1)} of the gap; higher in "
                     f"{r1['basins_better']} of {n} basins.")
        nse_say = (f"The median NSE rises from {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}, in {r1['basins_better']} of {n} basins.")
    elif case == "worse":
        nse_title = "NSE: lower"
        nse_slide = f"NSE: the median falls <b>{n2(r1['nse_base'])} &rarr; {n2(r1['nse_with_nldas'])}</b>."
        nse_say = f"The median NSE falls from {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}."
    else:
        nse_title = "NSE: no clear gain"
        nse_slide = (f"NSE: no clear gain. Median <b>{n2(r1['nse_base'])} &rarr; {n2(r1['nse_with_nldas'])}</b>, control "
                     f"{n2(r1['nse_control_one_day_older'])}; {basins}.")
        nse_say = (f"In NSE there is no clear gain: the median goes from {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}, and the control "
                   f"reaches {n2(r1['nse_control_one_day_older'])}.")

    # ---- the onsets warned: the result the slide leads with
    quarter = _fraction(on_share)
    seeds_on = (f"Onsets warned by the single trainings: {rng(r1, 'onset_seed_range_with_nldas')} with the rain of the issue day, "
                f"{rng(r1, 'onset_seed_range_control')} with the control, {rng(r1, 'onset_seed_range_base')} without either")
    if on_case in ("stamp", "stamp_likely"):
        sure = on_case == "stamp"
        title = (f"{lead1}, {evening_rain} closes {quarter} of the gap in onsets warned" + ("" if sure else ", probably a time-stamp gap")
                 + f"; {nse_title}")
        on_end = f", {quarter} of the gap." if sure else f", {quarter} of the gap; the control does not settle that it is the time stamp."
        on_say = f"That is {quarter} of the gap." + ("" if sure else " The control does not settle whether it is the time stamp.")
        on_read = (f"{seeds_on}: every training with the rain of the issue day is above every training of the control and of the model without "
                   f"extra rain." if sure else
                   f"{seeds_on}: the trainings with the rain of the issue day are above those without extra rain but overlap the control, so the "
                   f"control does not settle it.")
    elif on_case == "product":
        title = f"{lead1}, {evening_rain} lifts the onsets warned, but so does the same rain one day older: no time-stamp gap is shown"
        on_end, on_say = "; the control gains as much.", "But the control gains as much: this is what a second rain product is worth, not the time stamp."
        on_read = f"{seeds_on}: the control is as clearly above the model without extra rain, and overlaps the trainings with the later rain."
    elif on_case == "worse":
        title = f"{_cap(evening_rain)} lowers the share of onsets warned {LEAD_WORD[first]}: the test shows no time-stamp gap"
        on_end, on_say = ", a loss.", "Every training with the extra rain warns of fewer onsets than every training without it."
        on_read = f"{seeds_on}: every training with the extra rain is below every training without it."
    else:
        title = f"{_cap(evening_rain)} brings no clear gain in onsets warned {LEAD_WORD[first]}: the test shows no time-stamp gap"
        on_end, on_say = ", no clear gain.", "The single trainings overlap: no clear difference."
        on_read = f"{seeds_on}: the trainings overlap, so there is no clear difference."
    shown = on_case in ("stamp", "stamp_likely") and 0 < on_share < 1
    NW = _new(R)                                                 # the experiments of 7 and 8 October 2026 gave the model the rain up to midnight (next slide)
    after = f"about {Wd['nldas']}" if Wd else "the evening"
    rest = (f"The hours after {after} are in neither model: next slide." if NW["AO"] else
            "The rest needs later rain, most of it after the forecast is issued." if shown else
            f"On this evidence the gap {LEAD_WORD[first]} is a forecast gap too.")
    almost = max(far + far_on) <= 0.10                           # as the report puts it: 'almost all' when 90% or more of the gap is left

    # ---- the three bullets: the day stamp (as the figure on the slide shows it), the onsets, the NSE and what is left
    if windows_fig:                                               # 'orange bar': the colour check_day_windows.py gives the rain table
        b_stamp = (f"The data indicate: the rain table&rsquo;s day ends at about <b>{Wd['daymet']}</b> (orange bar). Later rain is stamped on "
                   f"the next day.")
        where = ""
    else:
        b_stamp = "Left: the data indicate that the rain table&rsquo;s day ends early. Later rain is stamped on the next day."
        where = "Right, "
    b_on = (f"With the river gauge, plus rain stamped up to the evening: onsets warned <b>{on_base} &rarr; {on_with}</b> (control {on_ctrl}; "
            f"upper bound {on_top}){on_end}")
    b_nse = f"{where}{nse_slide} {rest}"                           # with the figure of the test: 'Right, NSE: ...' names its panel

    # ---- facts for questions
    how = (f"How the day stamp is known: it is an inference from our own data, not a documented property of the rain table. Its rain of day d "
           f"goes with the second rain product&rsquo;s rain of the day before (correlation {n2(dn['day_before'])}) and of the same day "
           f"({n2(dn['same_day'])}), hardly with the day after ({n2(dn['day_after'])}).")
    if Wd:
        misfit = f"; {_and(Wd['misfits'])} {'does' if len(Wd['misfits']) == 1 else 'do'} not fit" if Wd["misfits"] else ""
        how += (f" Measured against hourly rain from {_word(Wd['n_references'])} weather reanalyses, its day ends at about {Wd['daymet']} local "
                f"standard time ({_word(Wd['n_readings'])} readings: {Wd['daymet_readings'][0]} to {Wd['daymet_readings'][1]}{misfit}). The same "
                f"check puts the satellite day {n2(Wd['off'][0], 1)} to {n2(Wd['off'][1], 1)} hours after its documented end, so no day end is "
                f"known better than that.")
    if SC.get("how_to_word_it"):                                 # the sources as results/sources_checked.json says they may be worded, word for word
        said = re.split(r"(?<=[.!?])\s+", SC["how_to_word_it"].strip())
        start = next((i for i, x in enumerate(said) if x.startswith("It is consistent")), 0)
        refs = "; ".join(cite(x["key"])[1:-1] for x in SC.get("sources", []) if x.get("key") in REFERENCES)
        unread = (f" No team member has read the {_word(len(SC.get('sources', [])))} papers yet." if "no team member has read" in SC.get("what", "")
                  else "")
        how += f" {_escape(' '.join(said[start:]))}" + (f" (Sources: {refs}.)" if refs else "") + unread
    else:
        how += " It fits rain gauges that are read in the morning; the documentation of the rain table was not checked."
    try:                                                         # false onset warnings, and the verdict with the later rain by the fixed criteria
        o_with, o_base = (X["models"][m]["test"]["summary"][lk(first)]["onset"] for m in ("wqn", "wq"))
        level = X["models"]["wqn"]["test"].get("trigger")
        at = "at the high-flow level itself" if level == 1 else f"at {level:g} times the high-flow level" if level else "at its warning level"
        verdict = _verdict_of(o_with, _criteria(R))
        false_w = (f"With that rain {pct(r1['onset_false_with_nldas'])} of the onset warnings are false (without it "
                   f"{pct(r1['onset_false_base'])}): the false onset warnings go from {count(o_base['false_alarms'])} to "
                   f"{count(o_with['false_alarms'])}. By the criteria fixed in advance, the onset warnings with that rain are "
                   f"{'still ' if verdict == 'NOT YET' else ''}{verdict} ({at}).")
    except (KeyError, TypeError):
        false_w = ""
    up, down = [d for d in change if d > 0], [d for d in change if d <= 0]
    by_basin = (f" Basin by basin, {len(up)} gain, by {signed(min(up))} to {signed(max(up))} NSE; {len(down)} do not, losing up to "
                f"{n2(-min(down))}; the median basin changes by {signed(r1['median_basin_gain'])}." if up and down and "median_basin_gain" in r1 else "")
    try:
        m_with, m_base = (X["models"][m]["test"]["summary"][lk(first)]["nse_mean"] for m in ("wqn", "wq"))
        mean_nse = f" The mean over the basins goes from {n2(m_base)} to {n2(m_with)}."
    except (KeyError, TypeError):
        mean_nse = ""
    not_clear = (" So by the rule for a clear difference (every single training on one side, and higher in most basins) the NSE gain is not "
                 "clear." if t["mixed"] else "")
    nse_detail = (f"NSE with the river gauge, {LEAD_WORD[first]}: median {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}, control "
                  f"{n2(r1['nse_control_one_day_older'])}, upper bound {n2(r1['nse_perfect_forecast'])}. That is {_share(share1)} of the gap in the "
                  f"median (shares are taken before rounding){', but' if t['few'] else ', and'} the forecast is {basins}.{not_clear}{by_basin}"
                  f"{mean_nse}")
    ng = t["no_gauge"]
    if ng:
        ng_on = {"stamp": "also clear of every training of the control and of the model without extra rain",
                 "stamp_likely": "clear of the model without extra rain, but not of the control", "product": "but the control gains as much",
                 "product_only": "not a clear gain", "none": "not a clear gain", "worse": "a loss"}[_timing_case(ng, "onset_")]
        ng_nse = {"stamp": "a clear gain, above the control", "stamp_likely": "a clear gain; the control does not settle whether it is the time stamp",
                  "product": "a clear gain, but the control helps as much",
                  "product_only": "no clear gain from the later rain; there the control, a second rain product, "
                                  + ("explains most of the rise" if ng["gain"] > 0 and ng["gain_control"] > 0.5 * ng["gain"] else "helps by itself"),
                  "none": "no clear gain", "worse": "worse with the extra rain"}[_timing_case(ng)]
        no_gauge = (f"Without the river gauge the onsets warned go from {pct(ng['onset_warned_base'])} to {pct(ng['onset_warned_with_nldas'])} "
                    f"(control {pct(ng['onset_warned_control'])}; upper bound {pct(ng['onset_warned_perfect_forecast'])}): "
                    f"{_fraction(ng['onset_share_of_gap_closed'])} of that gap, {ng_on}. In NSE the median goes from {n2(ng['nse_base'])} to "
                    f"{n2(ng['nse_with_nldas'])}, control {n2(ng['nse_control_one_day_older'])}: {ng_nse}.")
    else:
        no_gauge = ""
    left = (f"{_cap(later)}, with the river gauge, that rain closes {_shares(far)} of the gap in NSE and {_shares(far_on)} of the gap in onsets "
            f"warned: {'almost all' if almost else 'most' if max(far + far_on) < 0.5 else 'part'} of the gap remains ({pct(1 - max(far))} or more of the "
            f"gap in NSE and {pct(1 - max(far_on))} or more of the gap in onsets warned). "
            + ("A real weather forecast is tested on [slide:forecast]." if NW["FC"] else "No real weather forecast was tested."))
    if Wd:
        limit = (f"What the test cannot show: the second rain product&rsquo;s day ends at about {Wd['nldas']} local standard time "
                 f"({_word(Wd['n_readings'])} readings: "
                 f"{Wd['nldas_readings'][0]} to {Wd['nldas_readings'][1]}), hours before the forecast is made at local midnight. Rain of the last "
                 f"hours of the issue day is in neither model, so what is left of the gap is not all rain still to come.")
    else:
        limit = ("What the test cannot show: the second rain product&rsquo;s day ends in the evening, some hours before the forecast is made. "
                 "Rain of the last hours of the issue day is in neither model, so what is left of the gap is not all rain still to come.")
    return dict(
        base, title=title, bullets=[b_stamp, b_on, b_nse],
        notes=stamp + [
            f"The test adds {evening_rain}. With the river gauge, {LEAD_WORD[first]}, onsets warned go from {on_base} to {on_with}; the "
            f"control, the same rain one day older, gives {on_ctrl}; the upper bound is {on_top}.",
            f"{on_say} {nse_say}",
            rest],
        asked=[how,
               f"The control is the same second rain product made one day older: a second rain product, but no later rain. {on_read}",
               false_w, nse_detail, no_gauge, left, limit])


def _drawn_again(R, name, k):
    """'41% to 61%': the range of the onsets warned of one stored prediction set when the test years and the basins are drawn again (13_uncertainty.py,
    95%), or '' where the file or the set is not there."""
    try:
        s = R["uncertainty"]["stats"][f"onsets_warned.{name}.{lk(k)}"]["both"]
        return f"{pct(s['lo'])} to {pct(s['hi'])}"
    except (KeyError, TypeError):
        return ""


def _review_one(R, name, what):
    """A sentence on the independent AI review of one experiment (results/review_<name>.json) for the notes, as the record states it: AI agent, listed checks,
    none failed, no number differed, no leakage, the team's own check still open; '' while the record is missing."""
    rv = R.get(f"review_{name}")
    if not isinstance(rv, dict) or rv.get("quick_mode"):
        return ""
    cs = _counts_blocks(rv)                                       # every round of the record
    clean = all(c.get("failed") == 0 and c.get("numerical_discrepancies") == 0 for c in cs)
    return (f"The independent AI review of {what} (results/review_{name}.json; an AI agent, not a team member) recomputed the headline numbers in "
            f"{count(sum(c.get('checks_listed', 0) for c in cs))} listed checks" + (f" over {_word(len(cs))} rounds" if len(cs) > 1 else "") + ": "
            + (("none failed, no number differed" + (", no leakage found" if _no_leak(rv) else "")) if clean else "not every check passed (see the record)")
            + ("; the team&rsquo;s own check of it is still open." if str(rv.get("human_review", "")).startswith("not yet done") else "."))


def s_rain(R, T):
    """Rain known up to the issue time: the rain of the whole issue day up to local midnight as one extra input - hourly ERA5 rain (E1, a stand-in) and AORC
    (E1b) - and the real-time check with Stage IV radar (report section 4.6). The onsets warned lead, as everywhere in the talk; the median NSE follows.
    AORC is an analysis made after the fact: the title, the first bullet and the notes say so."""
    N = _new(R)
    E1, AO, S4, HS, W = N["E1"], N["AO"], N["S4"], N["HS"], N["W"]
    FIGN = "aorc_rain_to_midnight.png"
    if AO is None or not (FIG / FIGN).exists():
        _left_out(f"results/aorc_rain_to_midnight.json or figures/{FIGN} was not found: the slide on the rain known up to the issue time is a placeholder")
        return _pending("rain", "R2", 59, "Rain known up to the issue time", "the AORC experiment (results/aorc_rain_to_midnight.json)")
    first = LEADS[0]
    one = LEAD_WORD[first]
    d9 = W["daymet"] if W else "09:00"                            # where the rain table's day ends (results/day_windows.json), as section 4.6 of the report reads it
    u19 = f"{W['utc_day']} local standard time" if W else "the evening"
    n1 = lambda v: n2(v, 1)
    b, w = AO["base"][first], AO["with_"][first]
    nb, nw = AO["nse_base"][first], AO["nse_with"][first]
    s4_on, s4_off = (S4["stage4"] / S4["n"], S4["no_last"] / S4["n"]) if S4 else (None, None)
    same = bool(E1) and E1["hits"] == E1["hits_nldas"]            # ERA5 and NLDAS warn of the same number of onsets
    g = AO["gain"]
    as_much = min(g["ridge"], g["forest"]) >= S0.AS_MUCH * g["lstm"]      # the rule of the report for 'the simple models gain as much'
    v_on, v_all = AO["verdict_onset"][first], AO["verdict_all"]
    hind = S0.hindsight_clause(HS)                                # beside every 51%: the bound of the independent review (HINDSIGHT_WORDING, word for word while the stored bound rounds to it)
    hc = f" ({hind})" if hind else ""
    ng = AO["nse_gain"]                                           # the NSE gain with this rain: the simple models gain as many onsets, not as much NSE (the rule of s0_summary)
    times = ng["lstm"] / max(ng["ridge"], ng["forest"]) if max(ng["ridge"], ng["forest"]) > 0 else None
    more = ("about twice as much" if times and 1.6 <= times < 2.6 else "more than twice as much" if times and times >= 2.6 else "more" if times and times > 1.1 else "")
    nse_more = f"in NSE the LSTM gains {more}" if more else ""
    n_seeds = _word(len(R["aorc_rain_to_midnight"]["seeds"]))
    marks = (f"Filled: the average of {n_seeds} trainings; bar: lowest to highest. Hollow: controls (diamond: the same AORC rain over the "
             f"table&rsquo;s own window). Shaded: the AORC rungs. AORC is {AORC_WORDING}{hc}; the perfect forecast is {UPPER}.")
    era = f" ERA5 model rain: {pct(E1['with_'])}" + (", as NLDAS." if same else f", against {pct(E1['nldas'])} with NLDAS.") if E1 else ""
    say = ([f"ERA5 model rain to midnight warns of {pct(E1['with_'])}, " + ("the same as" if same else f"against {pct(E1['nldas'])} with") + " NLDAS."] if E1 else [])
    say += [f"AORC, {AORC_WORDING}, takes the onsets from {pct(b)} to {pct(w)}" + (f", {hind}," if hind else "") + f" and the median NSE from {n2(nb)} to {n2(nw)}.",
            f"The control, the same rain over the table&rsquo;s own window, gives {pct(AO['to09'])}: the hours after {d9} are worth {n1(AO['hours'])} points."]
    say += [f"Stage IV radar, the real-time product, in AORC&rsquo;s place gives {pct(s4_on)}, {pct(s4_off)} without its last hour: a check on the test years, not a service."] if S4 else []
    say += [f"The simple models gain as many onsets: {INFO_NOT_NET}." + (f" In NSE the LSTM gains {more}." if more else "")] if as_much else []
    say += [f"Onsets {'stay' if v_on == 'NOT YET' else 'are'} {v_on}: {S0.share_word(AO['false'][first])} of the warnings are false."]
    # ---- facts for questions: numbers and sentences of the results files, read through the report's own readers
    again = _drawn_again(R, "lstm_wqa", first)
    yr = re.search(r"hours before (\d{4})", str(R["aorc_rain_to_midnight"].get("outcome_caveat", "")))      # as the results file words it
    qa = [f"{S0.AORC_LONG}." + (f" Its hours before {yr.group(1)}, all the training years, are disaggregated daily totals." if yr else "") + f" As the extra input it gets {count(AO['hits'])} of the {count(AO['n'])} onsets "
          f"warned (single trainings {pct(AO['seeds'][0], 1)} to {pct(AO['seeds'][1], 1)}; the range when the test years and the basins are drawn again, 95%: "
          f"{again or 'not stored'}), against {count(AO['hits_base'])} with the rain table alone."]
    if E1:
        qa.append(f"E1, model rain as a stand-in: hourly ERA5 rain to midnight warns of {pct(E1['with_'], 1)} (single trainings {pct(E1['seeds'][0], 1)} to {pct(E1['seeds'][1], 1)}), "
                  f"{'the same as' if same else 'against'} {pct(E1['nldas'], 1)} with the NLDAS rain up to the evening ({count(E1['hits'])} hits"
                  f"{' each' if same else ' against ' + count(E1['hits_nldas'])}); the control, the same rain one day older, {pct(E1['control'])}. The success rule fixed in advance, "
                  f"clearly above NLDAS with a flat control, {'is met' if E1['met'] else 'is not met'}: ERA5 is model output, so this first test was a stand-in. "
                  f"{_review_one(R, 'rain_to_midnight', 'the ERA5 test')}")
    qa.append(f"The hours: over the UTC day, which ends at {u19}, the same AORC rain warns of {pct(AO['utc'], 1)}, and over the table&rsquo;s own window, {d9} to {d9}, of "
              f"{pct(AO['to09'], 1)}; so the hours after {d9} are worth {n1(AO['hours'])} points, {n1(AO['evening'])} of them after the end of the UTC day. The rain table&rsquo;s day "
              f"ending at about {d9} is an inference from our own data.")
    if HS:
        best = "none" if HS["best"] == 0 else f"{n1(HS['best'])} points"
        qa.append(f"Because AORC is {AORC_WORDING}, its hours before midnight could be scaled by a daily total that includes later rain. The independent review found no positive "
                  f"dependence on the radar rain after midnight on {HS['days_review']} test-year days (coefficient {signed(HS['coef_review'])}, 95% interval "
                  f"{signed(HS['ci_review'][0])} to {signed(HS['ci_review'][1], 3)}) and put the most that hindsight could add at about {round(HS['bound'])} points of the "
                  f"{pct(w)} (best estimate: {best}). In the words beside the {pct(w)}: {hind}.")
    if S4:
        lo, hi = S4["interval"]
        qa.append(f"What the {pct(s4_on)} is: {REALTIME_WORDING}. Stage IV radar and gauge rain, the real-time product, stand in place of AORC on the issue day of all {count(S4['n'])} onsets. "
                  f"The same networks warn of {count(S4['stage4'])} ({pct(s4_on)}) against {count(S4['aorc'])} with AORC and {count(S4['table'])} with the rain table alone, so "
                  f"{pct(S4['share'])} of the AORC effect survives (95% interval {pct(lo)} to {pct(hi)}); {count(S4['no_last'])} ({pct(s4_off)}) with the last hour of the issue day taken out, "
                  f"{count(S4['no_last2'])} ({pct(S4['no_last2'] / S4['n'])}) with the last two hours out. The networks were trained on AORC, not on Stage IV, and false warnings on "
                  f"days without an onset were not tested. The count depends more on whether the issue day was a heavy-rain day than on its exact amount: the Stage IV rain of "
                  f"another onset gives {count(S4['another_onset'])}, an ordinary day&rsquo;s rain {count(S4['ordinary'])}, no rain {count(S4['no_rain'])}.")
    qa.append(f"Given the same AORC rain ridge regression warns of {pct(AO['warned']['ridge'])} and the random forest of {pct(AO['warned']['forest'])} of the onsets (the LSTM: "
              f"{pct(AO['warned']['lstm'])}): the input adds {n1(g['ridge'])} and {n1(g['forest'])} points to them against {n1(g['lstm'])} to the LSTM"
              + (f", so the gain in onsets is {INFO_NOT_NET}" if as_much else ", so they gain less than the LSTM")
              + (f"; in median NSE the LSTM gains {more} ({signed(ng['lstm'])} against {signed(ng['ridge'])} for ridge regression and {signed(ng['forest'])} for the random forest)" if more else "")
              + f"; {pct(AO['far']['ridge'])} of the warnings of ridge regression are false against {pct(AO['far']['lstm'])} for the LSTM, and in median NSE the LSTM "
                f"({n2(AO['nse_with'][first])}) is ahead of ridge regression ({n2(AO['nse_simple']['ridge'])}) and the random forest ({n2(AO['nse_simple']['forest'])}).")
    cr = _criteria(R)
    qa.append(f"By the criteria fixed in advance the onsets are {_and(sorted(set(AO['verdict_onset'].values())))} at every lead time ({pct(AO['false'][first])} of the warnings are "
              f"false {one}); all high-flow days are {v_all} ({pct(AO['all_pod'])} warned, {pct(AO['all_far'])} false, against the {pct(cr[1])} that GO allows).")
    qa.append(f"{_cap(_leads(LEADS[1:]))} the same rain lifts the onsets warned from {_and(pct(AO['base'][k]) for k in LEADS[1:])} to {_and(pct(AO['with_'][k]) for k in LEADS[1:])} and "
              f"the median NSE by {_and(signed(AO['nse_with'][k] - AO['nse_base'][k]) for k in LEADS[1:])}: that is forecast territory ([slide:forecast]).")
    qa.append(_review_one(R, "aorc_rain_to_midnight", "this experiment") or "")
    return dict(
        key="rain", layout="side", owner="R2", seconds=59, image=FIGN, max_w=6.3,
        title=f"Given the rain up to the issue time, onsets warned {one} go from {pct(b)} to {pct(w)}" + (f", or {pct(s4_on)} with the real-time radar product in its place" if S4 else ""),
        crop=(0.0, 0.045, 1.0, 0.855), panel="the two panels with the names of the rungs and the legend, without the title and the footnote",
        caption=f"Onsets warned (top) and median NSE (bottom), {one}, with the river gauge, test years. {marks}"
                + (f" The {pct(s4_on)} in the text, not in the drawing, is {REALTIME_WORDING}." if S4 else "") + f" Full figure: report, {RAIN_REF}.",
        caption_whole=f"Onsets warned (top) and median NSE (bottom), {one}, with the river gauge, test years. {marks}",
        bullets=[f"AORC, {AORC_WORDING}: rain to midnight gives <b>{pct(b)} &rarr; {pct(w)}</b> of onsets{hc}, median NSE <b>{n2(nb)} &rarr; {n2(nw)}</b>.",
                 f"Control, the same rain over the table&rsquo;s own window: <b>{pct(AO['to09'])}</b> (<b>{n1(AO['hours'])}</b> points for the hours after {d9})."
                 + (" The simple models gain as many onsets" + (f" ({nse_more})" if nse_more else "") + "." if as_much else ""),
                 (f"Stage IV radar, a real-time product, in place of AORC: <b>{pct(s4_on)}</b>, <b>{pct(s4_off)}</b> without its last hour; a check on the test years, not a service. " if S4 else "")
                 + f"Onsets [[{v_on}]], all high-flow days [[{v_all}]]."],
        notes=say, handover="That rain has fallen. What about the rain still to come?", asked=[x for x in qa if x],
        figure_words=(f"left to right: the rain table (its day ends at {d9}), the NLDAS rain (about {W['nldas'] if W else 'the evening'}), ERA5 rain to midnight, AORC rain to midnight, AORC "
                      f"in the place of the rain table, NLDAS and ERA5 together, and the perfect forecast ({UPPER}); the hour under each name is where that rain&rsquo;s day ends; the shaded "
                      f"rungs are the new ones; the hollow marks are controls (the same rain one day older; the same product over the table&rsquo;s own window; over the UTC day)."))


def s_forecast(R, T):
    """A real weather forecast as an input (E2, report section 4.8): the control member of the GEFSv12 reforecast, rain of the next three days, put into the
    network of the last slide (E2a, scored on the usual test years) and retrained on a shorter split (E2b, its own scores). The slide leads with the lead times where the room
    is largest, 2 and 3 days ahead, and says the 1-day result in one clause (a moved operating point); every reading is the stored one."""
    N = _new(R)
    FC, AO, W = N["FC"], N["AO"], N["W"]
    FIGN = "forecast_input.png"
    if FC is None or not (FIG / FIGN).exists():
        _left_out(f"results/forecast_input.json or figures/{FIGN} was not found: the slide on the real weather forecast is a placeholder")
        return _pending("forecast", "G2", 41, "A real weather forecast as an input", "the forecast experiment (results/forecast_input.json)")
    first, rest = LEADS[0], tuple(LEADS[1:])
    k2, k3 = rest
    one = LEAD_WORD[first]
    n1 = lambda v: n2(v, 1)
    vs = set(FC["verdict_onset"].values()) | set(FC["verdict_onset_chosen"].values())
    verd = next(iter(vs)) if len(vs) == 1 else _and(sorted(vs))
    still = "still " if vs == {"NOT YET"} else ""
    # ---- the median NSE: NSE_FORECAST_WORDING while the stored numbers say it (the 1-day median falls, the gains at 2 and 3 days are 'mixed' by the rule of 4.2) - never
    #      'mixed at all three lead times'; the numbers are the model with the forecast against the same model without it (the AORC rain to midnight)
    nse_w = S0.nse_forecast_words(FC)
    up1 = FC["nse_gain"][first] > 0
    n_1 = FC["nse_basins"][first] if up1 else FC["nse_n"] - FC["nse_basins"][first]
    nse_1 = f"{n2(FC['nse_rain'][first])} &rarr; {n2(FC['nse_fc'][first])}"
    nse_23 = f"{n2(FC['nse_rain'][k2])} / {n2(FC['nse_rain'][k3])} &rarr; {n2(FC['nse_fc'][k2])} / {n2(FC['nse_fc'][k3])}"
    nse_say = (f"{_cap(nse_w)}: the 1-day median goes from {n2(FC['nse_rain'][first])} to {n2(FC['nse_fc'][first])}." if "1-day score" in nse_w
               else f"{_cap(nse_w)} ({nse_1} {one}).")                # the fixed wording in full, then the number it is about
    base, fc = FC["base"], FC["fc"]
    # the warning scores only: the moved operating point is a statement about onsets warned and false warnings (the CSI is not lower), not about the NSE
    mid = (f"{_cap(one)}, for the warnings: fewer onsets ({pct(FC['pod'][1])} against {pct(FC['pod'][0])}), far fewer false warnings "
           f"({pct(FC['far'][1])} against {pct(FC['far'][0])}): {MOVED_POINT}." if FC["moved"] else
           f"{_cap(one)} it warns of {signed(FC['gain'][first], 0)} points of onsets at the fixed warning level ({S0.reading_word(FC['gain_reading'][first])}).")
    mid_say = (f"{_cap(one)}, for the warnings, it is {MOVED_POINT}: fewer onsets, far fewer false warnings." if FC["moved"] else
               f"{_cap(one)} it warns of {signed(FC['gain'][first], 0)} points of onsets ({S0.reading_word(FC['gain_reading'][first])}).")
    cr = _criteria(R)
    marks = (f"Blue: the usual test years; orange, shaded: the model retrained on its own, shorter split. Top: onsets warned (grey line: the {pct(cr[0])} that GO asks for); bottom: median NSE. "
             f"Dark bar: the same model at the warning level chosen on the validation years. Hollow circle: control, the forecast shuffled (blue: it measures harm to a network trained on exact slots, not information; orange: the clean control). Diamond: the forecast without the rain to midnight. The orange numbers are comparable only inside their own split. "
             f"Perfect rain: the AORC rain of the three days ({AORC_WORDING}), {UPPER}.")
    sp = lambda d: _and(f"{100 * d[k]:.0f}" for k in LEADS) + "%"       # '41, 14 and 8%': a share for 1, 2 and 3 days ahead, as it is said
    ci = lambda k: f"{signed(FC['ci'][k][0], 0)} to {signed(FC['ci'][k][1], 0)}"
    yrs = lambda k: f"{FC['years_above'][k][0]} of {FC['years_above'][k][1]}"
    say = [f"A real forecast, one GEFS member, lifts the onsets warned {_leads(rest)} from {pct(base[k2])} and {pct(base[k3])}, with the rain table alone, to {pct(fc[k2])} and {pct(fc[k3])}.",
           nse_say,
           mid_say,
           f"Onsets are {still}{verd}."]
    # ---- facts for questions
    qa = []
    try:                                                          # the run it uses, in the words of the results file (the clock of the input)
        rule = _escape(R["forecast_input"]["forecast_data"]["rule"]["run"])
    except (KeyError, TypeError):
        rule = "the latest run that exists at the issue time"
    late = (f"; in Eastern summer its last files appear about {abs(round(FC['late']))} minutes after the issue time (a margin read from {_word(FC['margin_days']) if FC['margin_days'] else 'a few'} "
            f"days of posting times of the operational member, not from the archive)" if FC["late"] < 0 else "")
    pub = (f", of the same order as the published {n2(FC['published'][0])} and {n2(FC['published'][1])} for the whole ensemble (computed differently)" if FC.get("published") else "")
    cells = f" ({FC['n_cells'][0]} to {FC['n_cells'][1]} grid cells per basin)" if FC["n_cells"] else ""
    qa.append(f"The forecast: the control member of the GEFSv12 reforecast ({'one of ' + _word(FC['members']) + ' members run every day' if FC['members'] else 'one member'}), "
              f"{FC['years'][0]} to {FC['years'][1]}, averaged over each basin{cells} and summed over the local days t+1 to t+3. The run it uses, as results/forecast_input.json words it: "
              f"{rule}{late}. Against AORC daily rain its correlation is {_and(n2(FC['corr'][k]) for k in LEADS)} at {_leads(LEADS)}; its calibrated Brier skill "
              f"for days with at least {FC['brier_event']} is {_and(n2(FC['brier'][k]) for k in LEADS)}{pub}. One bias factor per basin and lead, fitted on the validation years.")
    gr = {FC["gain_reading"][k] for k in rest}
    gain_words = (f"{S0.reading_word(next(iter(gr)))} at both lead times" if len(gr) == 1 else _and(f"{S0.reading_word(FC['gain_reading'][k])} {LEAD_WORD[k]}" for k in rest))
    qa.append(f"E2a, the forecast substituted: the network of the last slide is trained on the AORC rain of the days t+1 to t+3 (a perfect rain forecast), stopped on the validation years with "
              f"the forecast in those slots and scored on the usual test years. {_cap(_leads(rest))} the onsets warned go from {_and(pct(FC['rain'][k]) for k in rest)} with the AORC rain to midnight "
              f"({_and(pct(base[k]) for k in rest)} with the rain table alone) to {_and(pct(fc[k]) for k in rest)} with the forecast: the gain over the rain to midnight is "
              f"{_and(signed(FC['gain'][k], 0) for k in rest)} points ({gain_words}; bootstrap over blocks of days, 95% interval {_and(ci(k) for k in rest)}; positive in "
              f"{_and(yrs(k) for k in rest)} test years). {_cap(nse_w)}: {S0.nse_forecast_numbers(FC)}.")
    qa.append(f"{_cap(one)}, the warning scores: {pct(FC['pod'][1])} against {pct(FC['pod'][0])} of the onsets at the fixed level, {pct(FC['far'][1])} against {pct(FC['far'][0])} of the warnings false, CSI "
              f"{n2(FC['csi'][1], 3)} against {n2(FC['csi'][0], 3)}; at the warning level chosen on the validation years the forecast model warns of {signed(FC['chosen_gain'][first], 1)} points. "
              f"Why it warns of fewer at the fixed level is not isolated. The words &lsquo;{MOVED_POINT}&rsquo; are a statement about these warning scores only: the median NSE {one} is lower with the forecast "
              f"({nse_1.replace('&rarr;', 'to')}, {'higher' if up1 else 'lower'} in {n_1} of {FC['nse_n']} basins), and NSE has no warning level to move.")
    qa.append(f"The control of E2a, the forecast shuffled within the month, falls to {_pcts(FC['ctrl'])}: a network trained on exact slots and given noise, so it measures harm, not information. "
              f"The clean control is that of E2b: trained and run on shuffled slots it warns of {_pcts(FC['s_ctrl'])}, the same as the model without the forecast ({_pcts(FC['s_rain'])}), so the "
              f"gain of E2b is the forecast&rsquo;s information.")
    X_ = R["forecast_input"]
    sp_ = X_["split_satellite_era"]
    qa.append(f"E2b, retrained on the shorter split (train {_years(sp_['train'])}, test {_years(sp_['test'])}; {_and(count(FC['n_sat'][k]) for k in LEADS)} onsets): rain table + gauge {_pcts(FC['s_wq'])}, + rain to midnight "
              f"{_pcts(FC['s_rain'])}, + forecast {_pcts(FC['s_fc'])}, perfect rain {_pcts(FC['s_perfect'])}; median NSE with the forecast {_and(n2(FC['s_nse_fc'][k]) for k in LEADS)}. Its numbers "
              f"are comparable only inside that split, and its base is weaker.")
    qa.append(f"The perfect rain of the three days (the AORC rain of those days, {AORC_WORDING}, given as the forecast) warns of {_pcts(FC['perfect'])} on the main split; the perfect weather forecast, {UPPER}, of "
              f"{_pcts(FC['upper'])}. The forecast closes {_and(S0.share_word(FC['share_gap'][k]) for k in rest)} of the gap to perfect rain {_leads(rest)}.")
    g = FC["gain_simple"]
    qa.append(f"Given the same inputs, ridge regression gains {_and(signed(g['ridge regression'][k], 0) for k in rest)} points and the random forest {_and(signed(g['random forest'][k], 0) for k in rest)} "
              f"{_leads(rest)}, against {_and(signed(g['lstm'][k], 0) for k in rest)} for the LSTM: {INFO_NOT_NET}; their warnings are more often false ({pct(FC['far_simple']['ridge regression'][first])} and "
              f"{pct(FC['far_simple']['random forest'][first])} {one}, against {pct(FC['far_simple']['lstm'][first])}). The rain to midnight still adds {signed(FC['midnight_given_forecast'][first], 0)} points "
              f"{one} beside the forecast, and {_and(signed(FC['midnight_given_forecast'][k], 0) for k in rest)} {_leads(rest)}.")
    vall = lambda d: " / ".join(d[k] for k in LEADS)
    qa.append(f"Verdicts by the criteria fixed in advance: onsets {verd} at every lead time (at the fixed and at the chosen warning level; {pct(FC['false_fc'][k2])} and {pct(FC['false_fc'][k3])} of "
              f"the warnings false {_leads(rest)}); all high-flow days {vall(FC['verdict_all'])} at the fixed level and {vall(FC['verdict_all_chosen'])} at the level chosen on the validation years "
              f"(E2b on its split: {vall(FC['verdict_all_s_chosen'])}). The success rule fixed in advance, a clear gain in onsets {_leads(rest)}, is {'met' if FC['success'] else 'not met'}.")
    qa.append(f"Not done: the other {_word(FC['members'] - 1) if FC['members'] else 'other'} ensemble members, the forecast&rsquo;s temperature, the encoder-decoder form that reads a forecast day by day, "
              f"Stage IV in place of AORC with the forecast. The base rung is AORC, {AORC_WORDING}; three trainings per set. {_review_one(R, 'forecast_input', 'this experiment')}")
    return dict(
        key="forecast", layout="side", owner="G2", seconds=41, image=FIGN,
        title=f"A real weather forecast lifts the onsets warned {_leads(rest)} from {pct(base[k2])} and {pct(base[k3])} (rain table alone) to {pct(fc[k2])} and {pct(fc[k3])}: {still}{verd}",
        crop=(0.36, 0.105, 1.0, 0.885), panel=f"the columns for {_leads(rest)}, without the title, the legend and the footnote",
        caption=f"{marks} One ensemble member; AORC is {AORC_WORDING}. Full figure: report, {FORECAST_REF}.",
        caption_whole=f"Look at the two right-hand columns, {_leads(rest)}. {marks} One ensemble member; AORC is {AORC_WORDING}.",
        bullets=[f"Forecast rain (GEFSv12, one member) in place of the AORC rain the network was trained on ({AORC_WORDING}).",
                 f"{_cap(nse_w)} ({one}: <b>{nse_1}</b>, {'higher' if up1 else 'lower'} in {n_1} of {FC['nse_n']} basins; {_leads(rest)}: <b>{nse_23}</b>).",
                 mid],
        notes=say, handover="So what did it take, and what is missing?", asked=qa,
        figure_words=(f"in each block, left to right: the rain table, + rain to midnight (AORC, {AORC_WORDING}), + forecast (substituted into the network trained on the AORC rain, blue; "
                      f"retrained, orange), perfect rain of the three days, and, in blue, perfect weather ({UPPER}); this part of the figure shows {_leads(rest)}; "
                      f"the {one} columns are in the full figure (report, {FORECAST_REF}); the percentages beside the marks are the values."))


def s_space(R, T):
    S, X = R["satellite_data"], R["satellite_inputs"]
    A, B = X["experiments"]["a_rain_source"], X["experiments"]["b_extra_inputs"]
    ca, cb = A["against_base"], B["against_base"]
    n = len(X["basins"])
    first = LEADS[0]
    rs = S["rain_check_summary"]["imerg"]
    clear = lambda c: c["larger_than_seed_noise"] is True          # the rule of this experiment (section 6.2 of the report): the stored flag
    up = [k for k in LEADS if clear(ca["imerg_wq"][lk(k)]["nse_median"]) and ca["imerg_wq"][lk(k)]["nse_median"]["difference"] > 0]
    down = [k for k in LEADS if clear(ca["imerg_wq"][lk(k)]["nse_median"]) and ca["imerg_wq"][lk(k)]["nse_median"]["difference"] < 0]

    def arrow(k):                                               # on the slide: '0.54 -> 0.61 1 day ahead'
        c = ca["imerg_wq"][lk(k)]["nse_median"]
        return f"<b>{n2(c['base'])} &rarr; {n2(c['variant'])}</b> {LEAD_WORD[k]}"

    def spoken(k):                                              # the same, as it is said
        c = ca["imerg_wq"][lk(k)]["nse_median"]
        return f"from {n2(c['base'])} to {n2(c['variant'])} {LEAD_WORD[k]}"

    nd = {k: v["correlation_with_daymet_rain_of_the_next_day"] for k, v in X["what_inputs_know_about_next_daymet_day"].items()}
    sat_next = next(v for k, v in nd.items() if k.startswith("satellite"))
    own_next = next(v for k, v in nd.items() if k.startswith("gauge"))
    clock = _utc_day_end(R)                                      # '19:00 local standard time (18:00 at Leaf River, MS)', if a results file says it
    utc_day = f"the satellite day is the UTC day, which ends at {clock} on the issue day" if clock else "the satellite day ends in the evening of the issue day"
    t = _timing(R)
    later = up and sat_next > own_next                           # satellite rain of the issue day knows more of the rain the rain table stamps a day later
    # what the direct test of the time stamp (the slide before) lets us say here, read from its onset warnings: shown / likely / still open / not shown
    stamp = "open" if t is None else {"stamp": "shown", "stamp_likely": "likely"}.get(t["on_case"], "not shown")
    pw = [(key, k) for key in ("pw_w", "pw_wq") for k in LEADS]
    pw_nse = [c for c in pw if clear(cb[c[0]][lk(c[1])]["nse_median"])]
    pw_on = [c for c in pw if clear(cb[c[0]][lk(c[1])]["onset_pod"])]
    vapour = (f"no clear change in NSE in {_word(len(pw))} comparisons" if not pw_nse else
              f"a clear change in NSE in {_word(len(pw_nse))} of {_word(len(pw))} comparisons")
    vapour_say = "no clear change in NSE" if not pw_nse else vapour   # as it is said: the number of comparisons stays on the slide
    if later and stamp != "not shown":                           # 'probably': this experiment cannot tell better timing from better amounts
        title = f"Satellite rain lifts the forecast with a river gauge {_leads(up)}: probably a later time stamp, not better rain"
    elif up:
        title = f"Satellite rain lifts the forecast with a river gauge {_leads(up)}, although it matches the rain table only moderately"
    elif down:
        title = f"Satellite rain instead of the rain table lowers the forecast with a river gauge {_leads(down)}"
    else:
        title = "Satellite rain instead of the rain table makes no clear difference to the forecast"
    gain = (f"Right panel, with the river gauge: satellite rain (circle) raises the median NSE {'; '.join(arrow(k) for k in up)}." if up else
            f"Right panel, with the river gauge: satellite rain (circle) changes the median NSE by "
            f"{_and(signed(ca['imerg_wq'][lk(k)]['nse_median']['difference']) for k in LEADS)}: {'a clear loss ' + _leads(down) if down else 'no clear change'}.")
    agree = (f"It matches the rain table only moderately (correlation <b>{n2(rs['median_correlation'])}</b>)"
             + (", but its day ends in the evening, so it holds later rain." if later else "."))
    late = re.search(r"about ([\d.]+) months late", S["imerg_run"])
    w_clear = [k for k in LEADS if clear(ca["imerg_w"][lk(k)]["nse_median"])]                 # the same swap without the river gauge
    improved = [ca["imerg_wq"][lk(k)]["n_basins_improved"] for k in up]
    better = "" if not up else (f"{improved[0]} of {n} basins" + (" each time" if len(up) > 1 else "") if len(set(improved)) == 1 else
                                f"{_and(improved)} of {n} basins")
    why = (f"The likely reason is the time stamp: {utc_day}, so it holds rain that the rain table stamps on the next day (correlation "
           f"{n2(sat_next)} with that rain, against {n2(own_next)} for the rain table itself). This experiment cannot tell better timing from "
           f"better amounts. "
           + {"shown": "The rain-timing test ([slide:timing]) shows such an effect directly, in the onsets warned, with a second rain product and "
                       "a control.",
              "likely": "The rain-timing test ([slide:timing]) points the same way, but its control does not settle it.",
              "open": "The rain-timing test is the direct check; its result is pending.",
              "not shown": "But the rain-timing test ([slide:timing]) did not show a time-stamp gap, so this stays open."}[stamp]) if later else ""
    n_seeds = len(X["seeds"])
    fires = sorted(X.get("seed_noise_rule", {}).get("share_of_comparisons_flagged_when_there_is_no_real_difference", {}).values())
    rule = ("A change is called clear here when it is more than twice the seed spread and every training with the new input lies on the same side "
            "of every training without it: the flag stored with this experiment"
            + (f". Where nothing really differs, that rule still calls {pct(fires[0])} to {pct(fires[-1])} of comparisons clear." if fires else "."))
    g1 = cb["pw_wq"][lk(first)]
    pw_gauge = (f"With the river gauge, {LEAD_WORD[first]}, the water vapour changes the median NSE by {signed(g1['nse_median']['difference'])} and "
                f"the forecast is higher in {g1['n_basins_improved']} of {g1['n_basins']} basins, "
                + ("a clear change" if clear(g1["nse_median"]) else "but the change is inside the seed spread")
                + "; it is a UTC-day value too, so a gain there may again be the time stamp.")
    return dict(
        key="space", layout="wide", owner="R2", seconds=40, image="satellite_inputs.png", title=title,
        crop=(0.0, 0.045, 1.0, 0.4925), panel="the top row (rain source)",
        caption="The top row of the full figure, enlarged: the rain source. Filled marker: a clear change; hollow: it cannot be told from "
                "chance. Grey band: twice the seed spread. &lsquo;Gauge-based rain (Daymet)&rsquo; is the rain table. Full figure: report, "
                "section 6.2.",
        caption_whole="Look at the top right panel: the rain source, with the river gauge. Filled marker: a clear change; hollow: it cannot be "
                      "told from chance.",
        bullets=[gain, agree,
                 f"Water vapour from a reanalysis (model output): {vapour}. Real GNSS-derived vapour was not tested."],
        notes=[f"Satellite rain agrees only moderately with our rain table: the correlation is {n2(rs['median_correlation'])}.",
               (f"Yet with the river gauge it raises the median NSE {spoken(up[0])}." if up else
                f"With the river gauge it lowers the median NSE clearly {_leads(down)}." if down else "With the river gauge it brings no clear change."),
               ("The likely reason is the time stamp: its day ends in the evening of the issue day." if later and stamp != "not shown" else
                f"Without the river gauge the change is clear {_leads(w_clear)}." if w_clear else
                "Without the river gauge the change is not clear at 1, 2 or 3 days ahead."),
               f"Water vapour can be derived from the delay of GNSS signals. We tested water vapour from a reanalysis, a weather model: "
               f"{vapour_say}. Real GNSS-derived vapour was not tested."],
        handover="So what would better warnings take?",
        if_late=("Satellite rain points the same way, probably because of its later time stamp; the details are in the report, section 6.2."
                 if later and stamp != "not shown" else "The test with satellite rain as an input is in the report, section 6.2."),
        asked=[(f"Satellites see rain where there are no rain gauges. With the river gauge, satellite rain raises the median NSE "
                f"{' and '.join(spoken(k) for k in up)}; the correlation with the rain table is taken day by day." if up else
                "Satellites see rain where there are no rain gauges; the correlation with the rain table is taken day by day."),
               f"This experiment has its own shorter split, test years {_years(A['split']['test'])}, so only differences inside it count. Its "
               f"scores are the mean of {_word(n_seeds)} single trainings, not the score of an averaged forecast as on the slides before"
               + (f"; the gain shows in {better}, counted on the averaged forecast" if up else "") + f". The totals of satellite rain differ from "
               f"the rain table by {signed(rs['bias_percent_min'], 0)}% to {signed(rs['bias_percent_max'], 0)}%.",
               rule,
               why,
               f"A GNSS receiver measures signal delays; precipitable water, the water vapour above a place, is derived from the zenith wet delay "
               f"with surface pressure and temperature {cite('bevis1992')}. The water vapour tested here is a daily value from a reanalysis, not GNSS "
               f"data" + (f"; the share of onsets warned rose clearly in {_word(len(pw_on))} of the {_word(len(pw))} comparisons" if pw_on else "")
               + ". Real GNSS-derived vapour, sampled every few minutes, was not tested.",
               pw_gauge,
               "This is the best case for satellite rain, not a real-time result"
               + (f": the product is corrected with rain gauges and published about {late.group(1)} months late." if late else ".")],
        figure_words="the grey band is twice the seed spread; &lsquo;gauge-based rain (Daymet)&rsquo; is the rain table; the bottom row of "
                     "the full figure (soil wetness and precipitable water, which is the water vapour) is in the report, section 6.2; the two "
                     "rows use different years.")


def s_cost(R, T):
    X = R["model_comparison"]
    M, C, ratio = X["models"], X["comparison"], X["cost_of_skill"]["transformer_over_lstm"]
    t, v = C["test"]["skill"], C["val"]["skill"]
    n_seeds = len(X["seeds"])
    clear = lambda s: (not s["seed_ranges_overlap"]) and abs(s["transformer_minus_lstm"]) > s["largest_seed_spread"]
    lo, hi = sorted((ratio["seconds_per_model"], ratio["seconds_per_update"]))
    times = f"roughly {lo:.0f} to {hi:.0f} times" if f"{lo:.0f}" != f"{hi:.0f}" else f"about {lo:.0f} times"
    dearer = lo > 1
    secs = {m: M[m]["cost"]["seconds_per_model"] for m in M}
    if not clear(t):
        acc, title = "no clear difference", (f"A Transformer is not better here, and training it takes {times} as long" if dearer else
                                             "A Transformer is not better here, and costs no more to train")
    elif t["transformer_minus_lstm"] > 0:
        acc, title = "the Transformer is more accurate", f"A Transformer is more accurate here, at {times} the training time"
    else:
        acc, title = "the LSTM is more accurate", f"A Transformer is less accurate here, and training it takes {times} as long"
    nse = lambda m: _slash(n2(C["test"][lk(k)][m]["nse_median"], 3) for k in LEADS)
    v1 = C["val"][lk(LEADS[0])]
    one = v1.get("largest_basin_difference", {})
    one_basin = (f", but {LEAD_WORD[LEADS[0]]} most of that gap is one basin ({one['basin']})"
                 if one and abs(one["nse_median_transformer_minus_lstm_without_it"]) < 0.5 * abs(v1["nse_median_transformer_minus_lstm"]) else "")
    val_note = (f"On the validation years the LSTM&rsquo;s median is ahead by {n2(abs(v['transformer_minus_lstm']), 3)}{one_basin}; in the mean over "
                f"the basins the difference is {n2(abs(v['mean_nse_over_basins']['transformer_minus_lstm']), 3)}."
                if "mean_nse_over_basins" in v and v["transformer_minus_lstm"] < 0 else "")
    little = _answer(R)["little"]                                # with the river gauge neither the best simple model nor the Transformer differs clearly from the LSTM
    return dict(
        key="cost", layout="wide", owner="G2", seconds=35, image="model_comparison.png", title=title,
        bullets=[f"Same data, inputs, trainer and test; only the network differs {cite('vaswani2017')}. {_cap(_word(n_seeds))} trainings each.",
                 f"Left, test years, median NSE {_slash(str(k) for k in LEADS)} days ahead: LSTM <b>{nse('lstm')}</b>, Transformer "
                 f"<b>{nse('transformer')}</b>: {acc}.",
                 f"Right, one training: <b>{count(secs['lstm'])} s</b> against <b>{count(secs['transformer'])} s</b> on a busy two-core laptop; "
                 f"{n2(ratio['parameters'], 1)} times the weights."],
        notes=["A Transformer was trained with the same data, trainer and test.",
               f"In the test years: {acc}. The mean NSE is {n2(t['lstm'], 3)} for the LSTM and {n2(t['transformer'], 3)} for the Transformer.",
               f"Training it took {times} as long.",
               "This holds for this data size, not for Transformers in general.",
               (f"So {SUPPORT.split(';')[0]}. What does change the forecast?" if little else "What does change the forecast?")],
        asked=[f"The mean NSE is taken over {_leads(LEADS)}. One training took {count(secs['transformer'])} seconds for the Transformer against "
               f"{count(secs['lstm'])} for the LSTM.",
               f"Both networks read {X['settings']['sequence_days']} days and were trained {'twice' if n_seeds == 2 else _word(n_seeds) + ' times'}; "
               f"two trainings of one model differ "
               f"by up to {n2(t['largest_seed_spread'], 3)} NSE. {val_note}".strip(),
               f"The Transformer has {n2(ratio['parameters'], 1)} times the weights. The times were measured on a busy two-core laptop, which is why "
               f"the cost is given as a range: {n2(ratio['seconds_per_model'], 1)} times in total training time, {n2(ratio['seconds_per_update'], 1)} "
               f"times for a typical (median) update."
               + (" The comparison was not timed again on an idle machine."
                  if any("NOT re-timed" in m for m in R["engine_review"].get("major_findings", [])) else "")],
        figure_words="left: accuracy (short bars: single trainings); right: cost of training one model (in brackets: Transformer divided by LSTM).")


def _need(R):
    """What the closing slides say - what better warnings would take, a river never seen, the limits, the milestones of the plan and the
    evidence for the finding - worked out in one place, so that the slide on what it would take, the slide with the key figure and the
    single closing slide used when the key figure is missing cannot disagree."""
    A, cmp_, _, m2 = _all_basins(R)
    ans = _answer(R)
    n = len(A["basins"])
    first = LEADS[0]
    steps, lev = A["decomposition"]["steps"], A["decomposition"]["levels"]
    fc, gauge1 = steps["perfect_forecast_with_gauge"], steps["river_gauge_without_forecast"][lk(first)]
    gains, pairs = ans["gains"], ans["pairs"]
    gap = _model_gap(R)                                          # the largest like-for-like gap between the LSTM and the best simple model
    t, lobo = _timing(R), _lobo(R)
    check = _team_check(R)[1]
    n_test = int(A["split"]["test"][1][:4]) - int(A["split"]["test"][0][:4])
    gain_txt = _and(signed(v) for v in gains)
    on1 = fc[lk(first)]

    # ---- a weather forecast: needed at every lead time (1 day ahead too, once the rain known by the evening is counted)
    larger = t is not None and t["r1"]["onset_share_of_gap_closed"] < 0.5 and max(t["far"] + t["far_on"]) < 0.5
    f_for = f"{_leads(LEADS)}"
    f_cell = (("the larger part of the gap at every lead time. " if larger else "") + f"Upper bound: <b>{gain_txt}</b> NSE; onsets warned "
              f"{LEAD_WORD[first]} {pct(on1['onset_pod_without'])} &rarr; {pct(on1['onset_pod_with'])}")
    f_say = ("Better warnings need a weather forecast as an input, the larger part of the gap at every lead time; none was tested."
             if larger else f"Better warnings {_leads(LEADS)} need a weather forecast as an input; none was tested.")

    # ---- rain observations up to the hour: what the rain-timing test lets us say, read from its onset warnings
    hourly, h_for = "<b>Rain observations up to the hour</b> (they exist today)", LEAD_WORD[first]
    if t is None:
        h_for, t_cell = LEAD_WORD[first], "result pending"
        t_say = (f"Rain observations up to the hour, which exist today, are the candidate for {LEAD_WORD[first]}. The rain-timing test was still "
                 f"running when these notes were built (result pending).")
    elif t["on_case"] in ("stamp", "stamp_likely"):
        r1, sure = t["r1"], t["on_case"] == "stamp"
        quarter = _fraction(r1["onset_share_of_gap_closed"])
        try:                                                     # the verdict for onsets with that rain, by the criteria fixed in advance
            verdict = _verdict_of(t["X"]["models"]["wqn"]["test"]["summary"][lk(first)]["onset"], _criteria(R))
        except (KeyError, TypeError):
            verdict = None
        still = "" if verdict is None else f"; for onsets still {verdict}" if verdict == "NOT YET" else f"; for onsets {verdict}"
        t_cell = (f"with rain stamped up to the evening of the issue day, onsets warned <b>{pct(r1['onset_warned_base'])} &rarr; "
                  f"{pct(r1['onset_warned_with_nldas'])}</b>: {quarter} of the gap" + ("" if sure else " (the control does not settle it)") + still)
        t_say = (f"Rain observations up to the hour exist today and add a part {LEAD_WORD[first]}: {quarter} of the gap in onsets warned in our "
                 f"test." + ("" if sure else " The control does not settle whether that is the time stamp.")
                 + ("" if verdict is None else f" The onset warnings are {'still ' if verdict == 'NOT YET' else ''}{verdict}."))
    elif t["on_case"] == "product":
        hourly, h_for = "<b>A second rain product</b>; rain up to the hour is not shown to add more", LEAD_WORD[first]
        t_cell = "the same rain made one day older lifts the onsets warned as much: no time-stamp gap is shown"
        t_say = (f"For {LEAD_WORD[first]}, our test shows that a second rain product helps. It does not show that rain observations up to the "
                 f"hour add more, because the same rain made one day older helps as much.")
    else:
        hourly, h_for = "<b>Rain observations up to the hour</b> (not confirmed by our test)", LEAD_WORD[first]
        t_cell = "no clear gain in onsets warned in our rain-timing test"
        t_say = f"For {LEAD_WORD[first]}, rain stamped up to the evening of the issue day brought no clear gain in onsets warned in our test."

    # ---- a river never seen (one training held out against one training with the basin in it): the loss in the typical basin comes first,
    #      because the two medians come from different basins
    verb = "falls" if lobo and lobo["out"] < lobo["inn"] else "does not fall"
    avg_w = cmp_("w", first)["lstm_nse_median"]                  # the same model as the average of its trainings (slide 'all', report section 4.2)
    n_seeds = _word(len(A["sets"]["w"]["seeds"]))
    if lobo is None:
        l_cell = l_say = l_title = "result pending"
        river_say = "For a river the model has never seen, the result is pending."
        l_more = []
    else:
        loses = (f"the typical basin loses {n2(lobo['loss'])} NSE {LEAD_WORD[first]}" if lobo["loss"] > 0 else
                 f"the typical basin loses nothing {LEAD_WORD[first]} (its NSE changes by {signed(-lobo['loss'])})")
        medians = f"the median {verb} from {n2(lobo['inn'])} to {n2(lobo['out'])}"
        l_title = f"{loses}; {medians}"
        l_cell = (f"held out of training, the typical basin loses <b>{n2(lobo['loss'])}</b> NSE {LEAD_WORD[first]} (lower in {lobo['n_lower']} of "
                  f"{lobo['n']} basins); {medians} ({lobo['label']} + basin properties, one training each)" if lobo["loss"] > 0 else
                  f"held out of training, {loses}; {medians} ({lobo['label']} + basin properties, one training each)")
        river_say = f"A river never seen is harder: {loses}; {medians}." if lobo["loss"] > 0 else f"A river never seen: {loses}; {medians}."
        l_say = (f"held out of training, {loses} (lower in {lobo['n_lower']} of {lobo['n']} basins, in {lobo['n_clear']} by more than the seed "
                 f"spread); {medians}, with {lobo['label']} and the basin properties (no river gauge). The loss in the typical basin is the number "
                 f"to quote: the two medians come from different basins. &lsquo;In training&rsquo; is one training of the model, so that one "
                 f"training is set against one; the average of {n_seeds} trainings of that model scores {n2(avg_w)}")
        k_less = sum(v < 1 for v in lobo["ratios"])
        where = f"all {lobo['n']} basins" if k_less == lobo["n"] else f"{k_less} of the {lobo['n']} basins"
        shift = (f"In {where} a smaller share of the rain reached the river in the test years than in the training years "
                 f"({n2(min(lobo['ratios']))} to {n2(max(lobo['ratios']))} of the training share), so the loss from not seeing a basin is measured "
                 f"together with this change and is approximate." if lobo["ratios"] else "")
        odd = ""
        for h in lobo["higher"][:1]:                              # a basin that scores higher when held out, and what the results file found behind it
            try:
                nse_, rr, vol = h["nse_1_day_ahead"], h["share_of_rain_reaching_the_river"], h["forecast_volume_error_pct_1_day_ahead"]["test_years"]
                odd = (f" One basin, {h['basin']}, scores higher when held out ({n2(nse_['held_out_with_properties'])} against "
                       f"{n2(nse_['in_training'])}). That is not skill for unseen rivers: {pct(rr['training_years'])} of the rain reached this river "
                       f"in the training years and {pct(rr['test_years'])} in the test years, so the model that had learned the river forecasts "
                       f"{pct(vol['in_training'] / 100)} too much water in the test years (held out: {pct(vol['held_out_with_properties'] / 100)}).")
            except (KeyError, TypeError):
                odd = ""
        l_more = [f"{shift}{odd}".strip()] if shift or odd else []

    # ---- the milestones of PLAN.md, judged as on page 1 of the report: M3 by its named outputs, M4 by the files that exist
    m3 = ("met" if all((FIG / f).exists() for f in ("flood_skill.png", "basin_maps.png")) and "flood_analysis" in R and "basin_gis" in R else
          "not met yet: a figure or a results file is missing")
    m4 = "built" if (ROOT / "Flood_Report.pdf").exists() else "slides built, report not built yet"

    # ---- the evidence for 'the inputs, not the network': on the same inputs the kinds of model lie close together, and every kind gains
    #      from more information
    cm = R["model_comparison"]["comparison"]["test"][lk(first)]
    why = (f"Why the inputs: given the same inputs, the LSTM and the best simple model are at most <b>{n2(gap)}</b> NSE apart"
           + ("" if ans["tf_better"] else ", and a Transformer is not better") + f"; the river gauge adds <b>{signed(gauge1['nse_median_gain'])}</b> NSE "
           f"{LEAD_WORD[first]}.")
    evidence = (f"The evidence for &lsquo;the inputs, not the network&rsquo;. Same inputs, past weather and river gauge, {LEAD_WORD[first]}: median NSE "
                + _and(f"{n2(v[1])} for the {k}" for k, v in pairs.items())
                + f"; the Transformer scores {n2(cm['transformer']['nse_median'], 3)} against {n2(cm['lstm']['nse_median'], 3)} for the LSTM trained "
                f"beside it. The largest gap between the LSTM and the best simple model, where both read the same weather, is {n2(gap)}. More "
                f"information: the river gauge adds " + _and(f"{signed(b - a)} NSE to the {k}" for k, (a, b) in pairs.items())
                + f"; the weather of the days being forecast, an upper bound, would add {gain_txt} NSE to the LSTM. The networks tried are an LSTM "
                f"of {A['config']['hidden']} memory cells and one small Transformer; larger ones were not tried.")
    upper = (f"The upper bound adds {gain_txt} NSE {_leads(LEADS)} (taken before rounding: {LEAD_WORD[first]} the median goes from "
             f"{n2(lev['wq'][lk(first)]['nse_median'], 3)} to {n2(lev['wqf'][lk(first)]['nse_median'], 3)}); a real forecast, which has errors, "
             f"would give less.")
    limits = f"Limits: {n} basins, {n_test} test years, one laptop. {check}"
    limits_say = f"Limits: {n} basins, {n_test} test years, one laptop. {_check_say(R)}"
    milestones = f"Plan milestones: M1 met; M2 {m2}; M3 {m3}; M4 {m4}."
    miles_asked = (f"Milestones of the plan: M1, the number to beat, met; M2, the LSTM beats the simple models, {m2}; M3, the warning table, maps "
                   f"and figures, {m3}; M4, the report and the slides, {m4}.")

    # ---- since 8 October 2026 the three rows of the old table are answered: a weather forecast, rain observations up to the hour (with AORC) and more
    #      basins were tested. The slide then says what it took, what is missing and what more rivers did; every number comes through the shared
    #      readers of the report, every reading and verdict is the stored one. Without the results files the old table stands (as it was true then).
    NW = _new(R)
    AO, S4, FC, RV, HS = NW["AO"], NW["S4"], NW["FC"], NW["RV"], NW["HS"]
    rest = tuple(LEADS[1:])
    hind = S0.hindsight_clause(HS)                                # beside every 51%
    hc = f" ({hind})" if hind else ""
    tested = ((f"it was tested with {RV['n_train']['unseen50_w_static']} training rivers instead of {RV['n_train']['lobo_w_static']}: median NSE "
               f"{n2(RV['unseen']['lobo_w_static'][first])} to {n2(RV['unseen']['unseen50_w_static'][first])} ([slide:rivers])") if RV else "it was not tested here")
    title = "What better warnings would take, and the limits of this study"
    table = dict(header=["What better warnings would take, and for what", "What our results say"], widths=[0.36, 0.64],
                 rows=[[f"<b>A weather forecast as an input</b> (not tested): {f_for}", f_cell],
                       [f"{hourly}: {h_for}", t_cell],
                       ["<b>More basins in training</b> (not tested): a river never seen", l_cell]])
    say, bullets, asked, away = [f_say, t_say, river_say, limits_say], [limits, milestones], [upper], ""
    if AO and S4 and FC and RV:
        n12, n11, n50 = RV["n_train"]["lstm_w"], RV["n_train"]["lobo_w_static"], RV["n_train"]["unseen50_w_static"]
        vs = set(AO["verdict_onset"].values()) | set(FC["verdict_onset"].values()) | set(FC["verdict_onset_chosen"].values())
        verd = next(iter(vs)) if len(vs) == 1 else _and(sorted(vs))
        still = "still " if vs == {"NOT YET"} else ""
        both = bool(AO["met"] and FC["success"])          # the success rules fixed in advance for rain up to midnight and for the forecast
        s4_on, up = S4["stage4"] / S4["n"], RV["unseen"]
        title = ("The finding holds in both directions: taking information away lowered the skill, giving it raised it" if both else
                 "Taking information away and giving it: what the new inputs changed")
        took = (f"onsets warned {LEAD_WORD[first]} <b>{pct(AO['base'][first])} &rarr; {pct(AO['with_'][first])}</b>{hc} with the rain up to midnight from AORC, {AORC_WORDING}; "
                f"<b>{pct(s4_on)}</b> with the real-time radar product in its place; {_leads(rest)}, from the rain table alone to a weather forecast (one ensemble member): "
                f"<b>{pct(FC['base'][rest[0]])} / {pct(FC['base'][rest[1]])} &rarr; {pct(FC['fc'][rest[0]])} / {pct(FC['fc'][rest[1]])}</b>")
        n_tr = _word(len(R["aorc_rain_to_midnight"]["seeds"]))     # the trainings of each set in the new experiments (the 'five' is the number the reviews propose, not a result)
        missing = (f"{S0.share_word(AO['false'][first])} of the onset warnings are false ({pct(AO['false'][first])}): onsets {still}[[{verd}]]; "
                   f"the other ensemble members; a network trained on Stage IV, the real-time product; five trainings instead of {n_tr} where NSE decides")
        rivers = (f"the same {n12} rivers: {S0.reading_word(RV['nse_reading'][first])} {LEAD_WORD[first]} (median NSE {n2(RV['nse']['lstm_wq'][first])} &rarr; "
                  f"{n2(RV['nse']['lstm63_wq'][first])}); a river never seen: <b>{n2(up['lobo_w_static'][first])} &rarr; {n2(up['unseen50_w_static'][first])}</b> with {n50} training "
                  f"rivers instead of {n11}; scale and nearness not separated ([slide:rivers])")
        table = dict(header=["What it took, what is missing", "What our results say"], widths=[0.32, 0.68],
                     rows=[["<b>It took two inputs</b>: the rain up to midnight, a weather forecast", took], ["<b>Still missing</b>", missing],
                           ["<b>More rivers in training</b>", rivers]])
        away = (f"Taking information away lowers the skill: without the river gauge the LSTM loses <b>{n2(gauge1['nse_median_gain'])}</b> NSE {LEAD_WORD[first]}"
                + (f"; held out of training, the typical basin loses <b>{n2(lobo['loss'])}</b>" if lobo else "") + ".")
        bullets = [away, limits]                                  # the milestones of the plan stand in the notes (miles_asked) and on page 1 of the report
        say = [("Taking information away lowered the skill; giving it raised it." if both else
                "Taking information away and giving it: here is what the new inputs changed."),
               "It took two inputs: the rain up to midnight, and a weather forecast.",
               f"Still missing: {S0.share_word(AO['false'][first])} of the onset warnings are false, so onsets are {still}{verd}. Also missing: the other ensemble members, a network trained on "
               f"the real-time product, and five trainings instead of {n_tr} where NSE decides.",
               limits_say]
        asked = [upper,
                 f"What it took, in numbers: with AORC, {AORC_WORDING}, {count(AO['hits'])} of the {count(AO['n'])} onsets are warned {LEAD_WORD[first]} against {count(AO['hits_base'])} with the rain "
                 f"table alone ({hind}); with Stage IV in its place {count(S4['stage4'])} ({pct(s4_on)}), and {count(S4['no_last'])} with the last hour of the issue day taken out: "
                 f"{REALTIME_WORDING} ([slide:rain]). {_cap(_leads(rest))} a weather forecast (one ensemble member) takes the onsets warned from {_and(pct(AO['with_'][k]) for k in rest)} with the AORC rain to midnight "
                 f"({_and(pct(FC['base'][k]) for k in rest)} with the rain table alone) to {_and(pct(FC['fc'][k]) for k in rest)} ([slide:forecast]).",
                 f"What is missing: the other {_word(FC['members'] - 1) if FC['members'] else 'other'} ensemble members and the forecast&rsquo;s temperature; a network trained on Stage IV, "
                 f"and a test of its false warnings on days without an onset; the simple-model yardstick at {RV['n_train']['lstm63_w']} rivers; the team&rsquo;s own checks of the experiments "
                 f"of 7 and 8 October. {check}"]
    return dict(
        ans=ans, m2=m2, check=check, why=why, evidence=evidence, upper=upper, river_say=river_say, f_say=f_say, t_say=t_say, l_title=l_title,
        lobo=lobo, title=title, table=table, say=say, bullets=bullets, away=away,
        limits=limits, limits_say=limits_say, milestones=milestones,
        remedy=f"More basins in training is the usual remedy; {tested}.",
        river_asked=[f"For a river never seen: {l_say}. More basins in training is the usual remedy {cite('kratzert2019pub')}; {tested}."] + l_more,
        miles_asked=miles_asked,
        need_asked=asked + [x for x in [f"For a river never seen: {l_say}. More basins in training is the usual remedy {cite('kratzert2019pub')}; {tested}."] + l_more] + [miles_asked])


def s_need(R, T):
    """The conclusions of the talk, the slide before the key figure: what it took and what is missing (once the experiments of 7 and 8 October are
    in the results; before them: what better warnings would take), and the limits of the study."""
    N = _need(R)
    return dict(
        key="need", layout="table", owner="A", seconds=34, title=N["title"], table=N["table"], bullets=N["bullets"], notes=N["say"],
        asked=N["need_asked"])


def _same_misses(R):
    """Do the kinds of model miss the same onsets (check_same_misses.py)? One sentence for questions, or '' with one line for stderr."""
    SM = _late(R, "same_misses", "the sentence on whether the models miss the same onsets is")
    if SM is None:
        return ""
    try:
        L, kinds = SM[lk(LEADS[0])], list(SM["kinds"])
        all_, more = L["missed_by_all_kinds"], L["of_missed_by_all"]["warned_with_perfect_forecast"]
        chance = max(all_.get(k, 0.0) for k in ("share_if_independent", "share_if_independent_by_basin"))    # what unrelated misses would give
        yes = "Yes: " if all_["share"] > chance and all_["share"] > 0.5 else ""
        return (f"Do the models miss the same onsets? {yes}{LEAD_WORD[LEADS[0]]}, with the river gauge, {pct(all_['share'])} of the "
                f"{count(L['n_onsets'])} onsets are missed by all {_word(len(kinds))} kinds of model ({_and(kinds)}); the LSTM given the perfect "
                f"weather forecast, an upper bound, warns of {pct(more['share'])} of those. Warnings are counted at the high-flow level itself; "
                f"the Transformer is the model of the cost comparison ([slide:cost]).")
    except (KeyError, TypeError) as e:
        _left_out(f"results/same_misses.json lacks a key ({type(e).__name__}: {e}): the sentence on whether the models miss the same onsets is left out")
        return ""


def _published(R):
    """Published LSTM scores set beside ours (results/published_scores.json, worded as its 'how_to_word_it' asks): one answer for
    questions, or '' with one line for stderr. 'Of the same size', never more; and who read the sources."""
    P = _late(R, "published_scores", "the sentence on published LSTM scores is")
    if P is None:
        return ""
    try:
        A, S = R["lstm_all"], P["sources"]
        k19, feng = S["kratzert2019"], S["feng2020"]
        ours = lambda inp, k: A["sets"][inp]["test"]["summary"][lk(k)]["nse_median"]
        first, last = LEADS[0], LEADS[-1]
        day = _date(P["checked_on"])
        unread = " No team member has read the papers yet." if "no team member" in P.get("checked_by", "").lower() else ""
        return (f"How do our scores stand beside published LSTM scores? Published scores on CAMELS-US are for estimating the flow of a day "
                f"from the weather up to and including that day: in our terms, the perfect weather forecast. One LSTM for "
                f"{count(k19['n_basins'])} basins scores a median NSE of {n2(k19['median_nse']['lstm_with_basin_properties_nse_loss_single'])} "
                f"{cite('kratzert2019')}; with the flow observed the day before, "
                f"{n2(feng['median_nse']['with_flow_observed_1_day_before_671'])} on {count(feng['n_basins'])} basins {cite('feng2020')}. Given the "
                f"same kind of information our LSTM scores {n2(ours('wf', last))} without a river gauge ({LEAD_WORD[last]}) and "
                f"{n2(ours('wqf', first))} with it ({LEAD_WORD[first]}): of the same size. This is no like-for-like test: several hundred basins "
                f"across the United States against {len(A['basins'])} in the South-East, other years, another rain product, a larger network "
                f"({k19['n_cells']} memory cells against {A['config']['hidden']}), and {LEAD_WORD[first]} our model reads two more days of weather. "
                f"Without the weather of the day being forecast the same network scores {n2(ours('w', first))} and {n2(ours('wq', first))} "
                f"{LEAD_WORD[first]}. An AI agent read the published numbers at the source on {day} (results/published_scores.json).{unread}")
    except (KeyError, TypeError, ValueError) as e:
        _left_out(f"results/published_scores.json or results/lstm_all.json lacks a key ({type(e).__name__}: {e}): the sentence on published "
                  f"LSTM scores is left out")
        return ""


def _ranges(R):
    """The ranges of the three onset shares when the test years and the basins are drawn again (13_uncertainty.py): one sentence for
    questions, or '' with one line for stderr."""
    U = _late(R, "uncertainty", "the sentence with the ranges of the onset shares is")
    if U is None:
        return ""
    try:
        first = LEADS[0]
        one = lambda name: (lambda s: f"{pct(s['value'])} ({pct(s['both']['lo'])} to {pct(s['both']['hi'])})")(U["stats"][f"onsets_warned.{name}.{lk(first)}"])
        S, rest = U["stats"], LEADS[1:]
        new = ""
        if f"onsets_warned.lstm_wqa.{lk(first)}" in S and all(f"onsets_warned.lstm_wqaF_sub.{lk(k)}" in S for k in rest):     # the sets of 7 and 8 October (13_uncertainty.py)
            two = lambda k: (lambda s: f"{pct(s['value'])} ({pct(s['both']['lo'])} to {pct(s['both']['hi'])})")(S[f"onsets_warned.lstm_wqaF_sub.{lk(k)}"])
            hind = S0.hindsight_clause(S0.hindsight_facts(R))             # beside the 51%
            new = (f" With the AORC rain up to midnight ({AORC_WORDING}) {one('lstm_wqa')}" + (f", {hind}" if hind else "") + f"; with a weather forecast beside it, {_leads(rest)}, "
                   f"{_and(two(k) for k in rest)} (the onsets warned with the rain table alone: {_and(pct(S[f'onsets_warned.lstm_wq.{lk(k)}']['value']) for k in rest)}).")
        return (f"How sure are these shares? The range when the test years and the basins are drawn again (95%): onsets warned {LEAD_WORD[first]} "
                f"{one('lstm_wq')} with the river gauge, {one('lstm_wqn')} with rain up to the evening of the issue day, {one('lstm_wqf')} with the "
                f"upper bound.{new} This range is not the seed spread, and it says nothing about other regions.")
    except (KeyError, TypeError) as e:
        _left_out(f"results/uncertainty.json lacks a key ({type(e).__name__}: {e}): the sentence with the ranges of the onset shares is left out")
        return ""


def _key_words(K):
    """The words under the talk-size drawing of the key figure, which carries no title, subtitle or footnote of its own: the footnote of the full rendering, sentence by
    sentence and word for word from results/key_figure.json (so that a change to the figure reaches the slide) - without the sentences about marks that the talk
    rendering leaves out (renderings.talk.left_out: the control circle), and with the asterisk sentence only where the footnote does not already say it."""
    left = (K.get("talk") or {}).get("left_out") or {}
    sentences = re.split(r"(?<=[.!?])\s+", K["footnote"].strip()) if K["footnote"] else []
    kept = [x for x in sentences if not (left.get("control_circle") and x.startswith("Control:"))]
    for i, x in enumerate(kept):                                  # after the footnote's sentence on the diamonds: what the Stage IV figure is (REALTIME_WORDING)
        if "diamonds" in x:
            kept.insert(i + 1, f"The diamonds are {REALTIME_WORDING}.")
            break
    text = " ".join(kept)
    if K["asterisk"] and "own experiment" not in text:
        text += " " + K["asterisk"]
    return _escape(text)


def _key_left_out(K):
    """What the talk-size drawing leaves out, in words, from results/key_figure.json (figure.renderings.talk.left_out and the names of the rows in
    figure.drawn_words): [] where the file says nothing."""
    left, rows = (K.get("talk") or {}).get("left_out") or {}, (K.get("words") or {}).get("rows") or {}
    named = []
    for r in left.get("rows") or []:
        block, _, row = r.partition(".")
        named.append(f"&lsquo;{rows.get(block, {}).get(row, r)}&rsquo;")
    out = [f"the rows {_and(named)}"] if len(named) > 1 else ([f"the row {named[0]}"] if named else [])
    if left.get("small_marks_of_the_other_kinds"):
        out.append(f"the small marks of the other kinds of model ({left['small_marks_of_the_other_kinds']})")
    if left.get("control_circle"):
        out.append("the control circle")
    if left.get("single_seed_lines"):
        out.append("the lines of the single seeds")
    return out


def s_key(R, T):
    """The last slide of the talk: the key figure of the main finding, with the conclusion as its title. It stays on the screen during
    the questions. Every number on it is read from results/key_figure.json, the file the figure is drawn from. Built only when _key(R)
    finds the figure and its results file (see slides). Since 8 October 2026 the figure has two blocks, 1 day ahead above and 2 and 3 days ahead
    below, with the AORC rung (and the real-time diamonds) and the real weather forecast in them; the slide says so and states the findings of
    the experiments of that day in the words of the file."""
    N, K = _need(R), _key(R)
    ans, first = N["ans"], LEADS[0]
    on, sp = [pct(v) for v in K["on"]], K["spread"]
    t = _timing(R)
    if t is not None:                                            # the same three shares as the rain-timing slide shows, read from another file
        r1 = t["r1"]
        other = [r1["onset_warned_base"], r1["onset_warned_with_nldas"], r1["onset_warned_perfect_forecast"]]
        if max(abs(a - b) for a, b in zip(K["on"], other)) > 5e-4:
            _left_out("results/key_figure.json and results/rain_timing.json do not give the same shares of onsets warned: the last slide and "
                      "the rain-timing slide disagree - rerun code/11_key_figure.py")
    NW = _new(R)
    if NW["AO"] and K["aorc"] is not None and abs(K["aorc"] - NW["AO"]["with_"][first]) > 5e-4:
        _left_out("results/key_figure.json and results/aorc_rain_to_midnight.json do not give the same share of onsets warned with the AORC rain: the last slide and "
                  "the slide on the rain up to the issue time disagree - rerun code/11_key_figure.py")
    for lead, row in K["far"].items():
        if NW["FC"] and "wqaF_sub" in row and abs(row["wqaF_sub"][0] - NW["FC"]["fc"][lead]) > 5e-4:
            _left_out(f"results/key_figure.json and results/forecast_input.json do not give the same share of onsets warned with the forecast {LEAD_WORD[lead]}: the last slide and "
                      f"the forecast slide disagree - rerun code/11_key_figure.py")
    close = K["ordered"] and ans["little"]                       # the kinds of model lie closer together than any step of the LSTM to the right
    kinds = _word(sp["kinds"])
    b_kinds = (f"Same inputs, {kinds} kinds of model: within <b>{n2(sp['nse'])}</b> NSE and <b>{100 * sp['onsets']:.0f}</b> "
               f"points of onsets." if close else
               f"Past weather + river gauge, {kinds} kinds of model: median NSE {n2(sp['nse_range'][0])} to {n2(sp['nse_range'][1])}; onsets "
               f"warned {pct(sp['onsets_range'][0])} to {pct(sp['onsets_range'][1])}.")
    crit = _criteria(R)
    below = (f" Even the upper bound stays below the {pct(crit[0])} that GO asks for." if K["on"][2] < crit[0] else "")
    mixed = (f" With rain up to the evening the NSE result is mixed: higher in only {K['mixed'][0]} of {K['mixed'][1]} basins." if K["mixed"]
             else "")
    # under the talk-size drawing, which has no title and no footnote, the slide prints the words: the footnote of the report's figure, word
    # for word from the results file; without it, the same in our words
    words = _key_words(K) or (
        f"Test years {K['years'][0]}&ndash;{K['years'][1]}, {K['n']} basins, {LEAD_WORD[first]}. Large circles: the LSTM. Small marks: other "
        f"kinds of model given the same inputs. Hatched: the perfect weather forecast, {UPPER}." + (f" NSE with rain up to the evening: mixed (higher in only {K['mixed'][0]} of {K['mixed'][1]} basins)." if K["mixed"] else ""))
    # the two blocks of the figure: 1 day ahead above (with the AORC rung and the real-time diamonds), 2 and 3 days ahead below (with the real forecast)
    far = {lead: row for lead, row in K["far"].items() if "wq" in row and "wqaF_sub" in row}
    drawn = sorted(far)
    ladder = [on[0], on[1]] + ([pct(K["aorc"])] if K["aorc"] is not None else []) + [on[2]]
    hind = S0.hindsight_clause(NW["HS"])                          # beside the AORC rung (the 51%)
    hc = f" ({hind})" if hind and K["aorc"] is not None else ""
    steps = [f"<b>{x}</b>" for x in ladder]
    if hc:                                                        # the clause stands beside the AORC rung: the third of the four
        steps[2] += hc
    b_more = (f"More information, {LEAD_WORD[first]}: onsets warned {' &rarr; '.join(steps)}"
              + (f"; <b>{pct(K['real'][0])}</b> with the real-time radar product in its place" if K["real"] else "") + ".")
    vo = set(NW["AO"]["verdict_onset"].values()) | set(NW["FC"]["verdict_onset"].values()) if NW["AO"] and NW["FC"] else set()
    still = f" Onsets: [[{next(iter(vo))}]]." if len(vo) == 1 else ""
    b_far = (f"{_cap(_leads(drawn))}: <b>{' / '.join(pct(far[k]['wq'][0]) for k in drawn)}</b> with the rain table alone, <b>"
             f"{' / '.join(pct(far[k]['wqaF_sub'][0]) for k in drawn)}</b> with a real weather forecast.{still}" if drawn else "")
    reviewed = "AI reviewer" in R["engine_review"]["who"]
    b_who = ((f"Claude Code wrote and ran everything under the team lead&rsquo;s direction; independent AI reviewers checked the engine and the experiments; the final checks "
              f"by a critic agent. " if reviewed else f"Claude Code wrote all code and ran all experiments under the team lead&rsquo;s direction. ") + _check_say(R))
    figure_facts = (f"The figure in numbers ({LEAD_WORD[first]}, test years, {K['n']} basins). With past weather + river gauge the {kinds} kinds "
                    f"of model score a median NSE of {n2(sp['nse_range'][0])} to {n2(sp['nse_range'][1])} and warn of {pct(sp['onsets_range'][0])} "
                    f"to {pct(sp['onsets_range'][1])} of the {count(K['n_onsets'])} onsets. The LSTM warns of {on[0]}; with rain up to the evening "
                    f"of the issue day {on[1]} (the control, the same rain one day older: {pct(K['control'])}); "
                    + (f"with the rain up to midnight from AORC, {AORC_WORDING}, {pct(K['aorc'])}"
                       + (" (" + "; ".join(x for x in (hind, (f"{pct(K['real'][0])} with the real-time radar product in its place, {pct(K['real'][1])} without its last hour: "
                                                              f"{REALTIME_WORDING}") if K["real"] else "") if x) + ")" if hind or K["real"] else "") + "; "
                       if K["aorc"] is not None else "")
                    + f"with the perfect weather forecast {on[2]}. Its median NSE is {_and(n2(v) for v in K['nse'][:2] + ([K['aorc_nse']] if K['aorc_nse'] is not None else []) + K['nse'][2:])}."
                    f"{mixed}{below}"
                    + ("".join(f" {_cap(LEAD_WORD[k])} ({count(K['n_far'][k])} onsets): rain table + river gauge {pct(far[k]['wq'][0])}, + AORC rain {pct(far[k]['wqa'][0])}, + a real forecast "
                                f"{pct(far[k]['wqaF_sub'][0])}, perfect rain {pct(far[k]['wqaP'][0])}, perfect weather {pct(far[k]['wqf'][0])}; the median NSE with the forecast is "
                                f"{n2(far[k]['wqaF_sub'][1])}, {MIXED} by the rule of section 4.2." for k in drawn if all(x in far[k] for x in ("wqa", "wqaP", "wqf"))) if drawn else ""))
    left_out = _key_left_out(K)
    return dict(
        key="key", layout="wide", owner="A", seconds=57, image=K["image"], tx_w=4.7,
        title=f"{_cap(FINDING)}" if ans["limit"] else "What we learned: the kinds of model against the information they are given",
        caption=words if K["talk_size"] else "",
        bullets=[x for x in (b_kinds, b_more, b_far, b_who) if x],
        notes=[("Large marks: the LSTM; small marks: other models, close to it." if close else
                "The large marks are the LSTM, the small marks the other kinds of model."),
               f"{_cap(LEAD_WORD[first])}, with more information, the onsets warned rise from {ladder[0]} to "
               f"{_and([x + (f' ({hind})' if hc and i == 1 else '') for i, x in enumerate(ladder[1:])])}.",
               f"The last is {UPPER}." + (f" The diamonds, {pct(K['real'][0])} and {pct(K['real'][1])}, put real-time radar in AORC&rsquo;s place: a check, not a service." if K["real"] else "")]
              + ([f"{_cap(_leads(drawn))}, a real forecast lifts them from {_and(pct(far[k]['wq'][0]) for k in drawn)} (rain table alone) to {_and(pct(far[k]['wqaF_sub'][0]) for k in drawn)}."] if drawn else [])
              + ([f"So {SUPPORT}."] if ans["support"] and close else [])
              + ([f"{_cap(FINDING)}."] if ans["limit"] else [f"What we learned: milestone 2 of our plan is {N['m2']}."])
              + ["Thank you. We welcome your questions."],
        asked=[N["evidence"], figure_facts,
               (f"What the diamonds are: {REALTIME_WORDING}." if K["real"] else ""),
               (f"{_review_sentence(R)} {N['check']}".strip() if reviewed else ""),
               (f"The headline of the figure, as results/key_figure.json states it: {_escape(K['headline'])}" if K["headline"] else ""),
               _same_misses(R), _published(R), _ranges(R),
               "Every mark in the figure is read from results/key_figure.json, which code/11_key_figure.py writes from the other results files and "
               "checks against them. The Transformer mark comes from its own experiment ([slide:cost]). Only the LSTM is drawn at the other sets: the other kinds of model "
               "are drawn where they read the same inputs as the LSTM (the report&rsquo;s drawing, figures/key_figure.png, draws them also with the AORC rain and with the forecast, at "
               f"another operating point), and with the perfect weather forecast the comparison is not like-for-like {LEAD_WORD[first]}, so no winner is called there."],
        figure_words=("one row per set of information, from the least (top) to the most (bottom); the upper block is "
                      f"{(K['words'].get('groups') or {}).get('upper', LEAD_WORD[first])}, the lower block "
                      f"{(K['words'].get('groups') or {}).get('lower', 'the later lead times')}; left panels: the share of onsets warned, with the share that GO asks for as a green band; "
                      "right panels: the median NSE; bars start at the dashed line, the level of past weather + river gauge; hatched: an upper bound that cannot be had in practice; "
                      "the hollow diamonds: Stage IV radar in place of AORC (right: the whole issue day, left: without its last hour); "
                      f"&lsquo;{K['words'].get('mixed', 'mixed')}&rsquo; beside an NSE: the gain over past weather + river gauge is {MIXED} by the rule of section 4.2"
                      + (f". This drawing leaves out {_and(left_out)}; the report&rsquo;s drawing (figures/key_figure.png) has them." if left_out else ".")))


def s_end(R, T):
    """The closing slide used when the key figure is missing (see slides): the finding in the title, what better warnings would take as
    a table, the limits and the milestones."""
    N = _need(R)
    ans = N["ans"]
    drop = bool(N["away"]) and ans["limit"]          # this slide has the 'why' bullet, which says the same of the river gauge: the 'taking away' bullet goes to the notes
    return dict(
        key="end", layout="table", owner="A", seconds=57,
        title=(f"{_cap(FINDING_SHORT)}: what better warnings would take" if ans["limit"] else "What we learned, and what better warnings would take"),
        table=N["table"], bullets=([N["why"]] if ans["limit"] else []) + [b for b in N["bullets"] if not (drop and b == N["away"])],
        notes=[(f"{_cap(FINDING_SHORT)}." + (f" {_cap(SUPPORT)}." if ans["support"] else "") if ans["limit"] else
                f"What we learned: milestone 2 of our plan is {N['m2']}.")] + N["say"],
        asked=[N["evidence"]] + ([N["away"]] if drop else []) + N["need_asked"] + [_same_misses(R), _published(R), _ranges(R)])


# ---- backup slides: after the last talk slide, shown only if a question asks for them ---------------------------------------------
def b_curves(R, T):
    """Backup: the learning curves (12_training_curves.py), or None - with one line for stderr - without the figure or its results file."""
    if not (FIG / "training_curves.png").exists():
        _left_out("figures/training_curves.png was not found: the backup slide with the learning curves is left out")
        return None
    X = _late(R, "training_curves", "the backup slide with the learning curves is")
    if X is None:
        return None
    try:
        P, S = X["panels"], X["settings"]
        one, wq, wqf = P["one_river"], P["all_wq"], P["all_wqf"]
        o = one["per_training"][0]
        span = lambda p: (lambda b: f"{count(min(b))} to {count(max(b))}" if min(b) != max(b) else count(b[0]))([r["best_update"] for r in p["per_training"]])
        n = int(wq["n_rivers"])
        falls = o["val_nse_at_last"] < o["val_nse_at_best"] and o["train_loss_last"] < o["train_loss_at_best"]
        same = bool(X["same_settings_one_river_and_all"])
        title = (f"Learning curves: on one river the validation score peaks at update {count(o['best_update'])} and then falls; on {n} rivers it "
                 f"peaks at updates {span(wq)}" if falls else
                 f"Learning curves: the validation score peaks at update {count(o['best_update'])} on one river and at updates {span(wq)} on {n} rivers")
        left = (f"Left, one river, one training: the validation NSE peaks at update <b>{count(o['best_update'])}</b> ({n2(o['val_nse_at_best'])}) and "
                f"is {n2(o['val_nse_at_last'])} at update {count(o['last_update'])}"
                + (", while the training loss falls further: the network memorises its training years." if falls else "."))
        base_note = (f"The left column is the base setting of the tuning ({S['hidden']} memory cells, learning rate {S['lr']:g}) and one training, "
                     f"not the chosen one-river model" + (f" ({X['chosen_one_river']['settings']['hidden']} memory cells)" if "chosen_one_river" in X else "")
                     + ". Each column has its own scales.")
        return dict(
            key="curves", layout="wide", owner="G2", seconds=0, image="training_curves.png", title=title,
            caption=f"Grey: training loss. Blue: NSE in the validation years. Dot and band: the state that early stopping kept. Right column: "
                    f"with the perfect weather forecast, {UPPER}. Report: section 3.",
            bullets=[left,
                     f"Middle, all {n} rivers with the river gauge: the kept state lies at updates <b>{span(wq)}</b>; right, with the upper bound, "
                     f"at {span(wqf)}.",
                     ("The same network settings in all three columns. " if same else "") + "Early stopping keeps the state with the best "
                     "validation score."],
            notes=[f"These are the learning curves. Grey is the training loss, blue the NSE in the validation years.",
                   (f"On one river the validation score peaks at update {count(o['best_update'])} and then falls, while the training loss falls "
                    f"further: the network memorises its training years." if falls else
                    f"On one river the validation score peaks at update {count(o['best_update'])}."),
                   f"On {n} rivers the peak comes at updates {span(wq)}. Early stopping keeps the state at the peak."],
            asked=[base_note,
                   (f"Updates are not passes through the data: an update scores the same number of days in every column, so on one river it passes "
                    f"through the training years about {n} times as often. Counted in passes, the kept states come at about the same point: about "
                    f"{o['passes_at_best']:.0f} on one river, about {wq['passes_at_best_min']:.0f} to {wq['passes_at_best_max']:.0f} on {n} rivers."
                    if all(k in wq for k in ("passes_at_best_min", "passes_at_best_max")) and "passes_at_best" in o else
                    "Updates are not passes through the data: an update scores the same number of days in every column."),
                   f"The training loss on one river goes from {n2(o['train_loss_first'])} to {n2(o['train_loss_last'])}; the validation checks come "
                   f"every {X['check_every']['one_river']} updates on one river and every {X['check_every']['all']} updates on {n} rivers, so the "
                   f"curves are coarse. Training stops after {S['patience']} checks without a better score.",
                   "The validation score is the NSE of the one river on the left and the mean over the rivers in the other columns: it picks the "
                   "state to keep; it is not a number of the report&rsquo;s tables."],
            figure_words="top row: training loss; bottom row: NSE in the validation years, mean of 1 to 3 days ahead; the curves were stored "
                         "by the trainings themselves, nothing was trained again.")
    except (KeyError, TypeError, IndexError, ValueError) as e:
        _left_out(f"results/training_curves.json lacks a key ({type(e).__name__}: {e}): the backup slide with the learning curves is left out")
        return None


def b_lobo(R, T):
    """Backup: a river the model has never seen, basin by basin (05_lobo.py), or None without the figure or the result."""
    N = _need(R)
    lobo = N["lobo"]
    if lobo is None or not (FIG / "lobo.png").exists():
        _left_out("figures/lobo.png or results/lstm_lobo.json was not found: the backup slide on a river never seen is left out")
        return None
    first = LEADS[0]
    return dict(
        key="lobo", layout="side", owner="A", seconds=0, image="lobo.png",
        title=f"A river never seen in training: {N['l_title']}",
        caption="Filled dot: the basin was in training. Diamond and circle: held out, trained on the other basins (with and without the basin "
                "properties). Report: section 4.4.",
        bullets=[f"NSE {LEAD_WORD[first]} in the test years, basin by basin; no model reads a river gauge.",
                 f"Lower when held out in <b>{lobo['n_lower']}</b> of {lobo['n']} basins, in {lobo['n_clear']} by more than the seed spread. One "
                 f"training each.",
                 N["remedy"]],
        notes=[f"This is every basin {LEAD_WORD[first]}: the filled dot with the basin in training, the diamond with the basin held out.",
               N["river_say"],
               f"It is lower in {lobo['n_lower']} of {lobo['n']} basins."],
        asked=N["river_asked"],
        figure_words="one row per basin, grouped by landscape; the shaded bar is the range of the single trainings of the model that had the "
                     "basin in training; held-out scores below the axis are written beside their marker.")


def b_rivers(R, T):
    """Backup: more rivers in training (E3, report section 4.7): the same 12 rivers with 63 training rivers, and rivers never seen with 50 training rivers instead
    of 11. The readings are the stored ones; a lower onset share with fewer false warnings is worded as the report words it (a moved operating point)."""
    RV = _new(R)["RV"]
    FIGN = "more_basins.png"
    if RV is None or not (FIG / FIGN).exists():
        _left_out(f"results/more_basins.json or figures/{FIGN} was not found: the backup slide on more training rivers is left out")
        return None
    first = LEADS[0]
    rest = tuple(LEADS[1:])
    one = LEAD_WORD[first]
    n12, n63, n11, n50 = (RV["n_train"][s] for s in ("lstm_w", "lstm63_w", "lobo_w_static", "unseen50_w_static"))
    a, b = RV["nse"]["lstm_wq"], RV["nse"]["lstm63_wq"]
    p0, p1 = RV["onset"]["lstm_wq"][first], RV["onset"]["lstm63_wq"][first]
    (h0, h1), (fa0, fa1) = RV["hits"], RV["false_alarms"]
    moved = p1 < p0 and S0.moved_operating_point(p0, p1, fa0 / (h0 + fa0), fa1 / (h1 + fa1), RV["csi"][0], RV["csi"][1])
    on_say = (f"{pct(p0)} &rarr; {pct(p1)}: {MOVED_POINT}" if moved else f"{pct(p0)} &rarr; {pct(p1)} ({S0.reading_word(RV['onset_reading'])} by the seed rule)")
    u = RV["unseen"]
    read = S0.reading_word(RV["nse_reading"][first])
    ur = RV["unseen_reading"]["w_static"][first]
    gains = "gains" if ur == "clear gain" else "loses" if ur == "clear loss" else "shows " + S0.reading_word(ur)
    ps, pn = (u["lobo_w_static"][first], u["unseen50_w_static"][first]), (u["lobo_w_nostatic"][first], u["unseen50_w_nostatic"][first])
    vs = {v for v in (RV.get("verdict") or {}).values()} if isinstance(RV.get("verdict"), dict) else set()
    try:                                                          # the longer arm: its update limit over the plain arm's, from the stored configurations
        cfg = R["more_basins"]["design"]
        times = _word(round(cfg["config_equal_training"]["updates"] / cfg["config"]["updates"]))
        longer = f"{times} times the update limit and patience window"
    except (KeyError, TypeError, ZeroDivisionError):
        longer = "a larger update limit and patience window"
    marks = (f"Bars: the average of the three trainings, dots: single trainings (the grey {n11}-river bars are one training per river); outline bars: the same rivers when in training; "
             f"the number under a bar is the number of rivers in training.")
    rr = {RV["nse_reading"][k] for k in rest}
    rest_read = (f"{S0.reading_word(next(iter(rr)))} at both lead times" if len(rr) == 1 else _and(f"{S0.reading_word(RV['nse_reading'][k])} {LEAD_WORD[k]}" for k in rest))
    qa = [f"{_cap(_leads(rest))}, with the river gauge, the median NSE goes from {_and(n2(a[k]) for k in rest)} to {_and(n2(b[k]) for k in rest)} ({rest_read}) "
          f"with {n63} training rivers; the seed ranges of the two models overlap {one}, although the NSE is higher in {RV['nse_basins'][first]} of {RV['nse_n']} rivers.",
          f"The onsets {one}: {count(h0)} hits and {count(fa0)} false alarms with {n12} training rivers, {count(h1)} and {count(fa1)} with {n63}; CSI {n2(RV['csi'][0], 3)} against {n2(RV['csi'][1], 3)}; "
          f"at the warning level chosen on the validation years {pct(RV['chosen'][1], 1)} against {pct(RV['chosen'][0], 1)}. A second arm gave the {n63}-river model {longer}: it kept the same early networks (best update about {count(RV['best_update'][2])} against {count(RV['best_update'][1])}), so it shows that a longer patience window does "
          f"not move the optimum; it did not test equal training per river.",
          f"For rivers never seen the gain holds at all three lead times in the seed average; each single {n50}-river training is {signed(min(RV['single_gain']))} to {signed(max(RV['single_gain']))} "
          f"above the earlier test, which is one training per river (single-training p {n2(min(RV['single_p']))} to {n2(max(RV['single_p']))}). The loss from not having seen the river, "
          f"{n2(RV['loss'][0])} NSE in the typical river, shrinks to {n2(RV['loss'][1])}. Onsets warned move little for such rivers.",
          f"The number, the variety and the nearness of the training rivers changed together between the two designs (the nearest training gauge: a median of {count(RV['km'][0])} km with {n11} rivers, "
          f"{count(RV['km'][1])} km with {n50}) and were not separated; the gain does not rank with the distance to the nearest training gauge (rank correlation {signed(RV['rho'][0])}, "
          f"p = {n2(RV['rho'][1])}), so it can be credited neither to scale nor to proximity alone.",
          f"Published models train on hundreds of rivers" + (f", up to all {RV['published_basins']} of a benchmark set" if RV["published_basins"] else "")
          + f" {cite('kratzert2024')}; that was another task and network, so only the direction carries over. Not done: a buffered leave-out or a dose-response over subsets of the {n50}, "
            f"which would separate scale from nearness; the simple-model yardstick at {n63} rivers.",
          _review_one(R, "more_basins", "this experiment")]
    return dict(
        key="rivers", layout="side", owner="G2", seconds=0, image=FIGN,
        title=f"More training rivers: {read} for the {n12} rivers with a gauge {one}, but a river never seen {gains}: median NSE {n2(ps[0])} &rarr; {n2(ps[1])}",
        crop=(0.0, 0.04, 1.0, 0.855), panel="the four panels with the legend, without the title and the footnote",
        caption=f"Top: the same {n12} rivers; bottom: rivers never seen (weather only); left: median NSE, right: onsets warned, {one}. {marks} Full figure: report, {RIVERS_REF}.",
        caption_whole=f"Top: the same {n12} rivers; bottom: rivers never seen (weather only). {marks}",
        bullets=[f"The same {n12} rivers, with the river gauge, {one}: median NSE <b>{n2(a[first])} &rarr; {n2(b[first])}</b> with {n63} training rivers instead of {n12}: {read}. "
                 f"Onsets warned {on_say}.",
                 f"Rivers never seen, weather only: median NSE <b>{n2(ps[0])} &rarr; {n2(ps[1])}</b> with basin properties, <b>{n2(pn[0])} &rarr; {n2(pn[1])}</b> without, with {n50} "
                 f"training rivers instead of {n11}.",
                 f"Scale and nearness are not separated: the nearest training gauge is {count(RV['km'][1])} km away instead of {count(RV['km'][0])} km."],
        notes=[f"We trained on {n63} rivers instead of {n12}.",
               f"For the same {n12} rivers with a gauge, {one}, the median NSE goes from {n2(a[first])} to {n2(b[first])}: {read}.",
               f"A river never seen {gains}: from {n2(ps[0])} to {n2(ps[1])} with basin properties, and from {n2(pn[0])} to {n2(pn[1])} without them.",
               "But the number, the variety and the nearness of the training rivers changed together, so scale and nearness are not separated."],
        asked=[x for x in qa if x],
        figure_words=(f"top left and right: the same {n12} rivers, with the river gauge and without it; bottom: rivers never seen, with basin properties and without them; grey: {n12} training "
                      f"rivers (or {n11} in the leave-one-out test); blue: {n63}; dark blue: {n63} with {longer}; orange: {n50} other rivers, the {n12} never seen."))


def b_cost(R, T):
    """Backup: the Transformer against the LSTM. It was a slide of the talk until 8 October 2026 and is now one sentence on the slide on the LSTM (the same facts,
    from _transformer); here the sentence that led into the next slide of the talk is left out."""
    s = s_cost(R, T)
    s["notes"] = [x for x in (n.replace(" What does change the forecast?", "") for n in s["notes"]) if x != "What does change the forecast?"]
    return s


def b_space(R, T):
    """Backup: satellite rain as an input (08_satellite_inputs.py). It was a slide of the talk until 8 October 2026; the experiments of that day test the time stamp
    directly, on [slide:rain]."""
    return s_space(R, T)


def b_scores(R, T):
    """Backup: the warning scores at all three lead times (06_flood_analysis.py), or None without the figure."""
    if not (FIG / "flood_skill.png").exists():
        _left_out("figures/flood_skill.png was not found: the backup slide with the warning scores is left out")
        return None
    V = R["flood_analysis"]["verdict"]["services"]
    go_pod, go_far, ca_csi = _criteria(R)
    first, GAUGE = LEADS[0], "lstm_wq"
    g_all, g_on = V[GAUGE][lk(first)]["all"], V[GAUGE][lk(first)]["onset"]
    verd = lambda kind: _slash(V[GAUGE][lk(k)][kind]["pooled"] for k in LEADS)
    return dict(
        key="scores", layout="side", owner="G1", seconds=0, image="flood_skill.png",
        title=f"Warning scores {LEADS[0]} to {LEAD_WORD[LEADS[-1]]}: the share warned and the share of warnings that were false",
        caption=f"Top: all high-flow days. Bottom: onsets only. Left: share warned. Right: share of the warnings that were false. Green: what "
                f"GO asks for. Dashed: the LSTM given a perfect weather forecast, {UPPER}. Report: section 5.",
        bullets=[f"With the river gauge, {LEAD_WORD[first]}: <b>{pct(g_all['pod'])}</b> of the high-flow days warned, {pct(g_all['far'])} of the "
                 f"warnings false; of the onsets <b>{pct(g_on['pod'])}</b> warned, {pct(g_on['far'])} false.",
                 f"Verdicts with the river gauge, {_slash(str(k) for k in LEADS)} days ahead: {verd('all')} for all high-flow days, "
                 f"{verd('onset')} for onsets.",
                 f"GO = at least {pct(go_pod)} warned and at most {pct(go_far)} false warnings; CAUTION = hits are at least {pct(ca_csi)} of hits + "
                 f"misses + false warnings."],
        notes=["The top row is all high-flow days, the bottom row the onsets; left the share warned, right the share of warnings that were "
               "false.",
               f"The green bands are what GO asks for. With the river gauge, {LEAD_WORD[first]}, {pct(g_on['pod'])} of the onsets are warned.",
               f"The dashed line is the same LSTM given a perfect weather forecast. That is {UPPER}."],
        asked=["A model warns when its forecast is above the warning level. The verdicts are taken at the warning levels chosen on the "
               "validation years; the figure is drawn at the high-flow level itself."],
        figure_words="grey: &lsquo;tomorrow = today&rsquo;; green: the random forest; blue: the LSTM with the river gauge; black bars: the range "
                     "of the single trainings.")


SLIDE_REF = re.compile(r"\[slide:(\w+)\]")


def slides(R, T):
    """The deck: the talk slides in the order of the talk, then the backup slides.
    The talk: the question with its answer and the data (the team lead); the fair test and the number to beat (GNSS member 1); the model
    changes little - the LSTM against the simple models, with the Transformer in one sentence (the team lead) - and one event that turns from the
    model to the information (GNSS member 2); the verdict on the warnings (GNSS member 1); the information changes a lot - why onsets are missed
    (RS/GIS member 2), the rain-timing test (the team lead), the rain known up to the issue time (RS/GIS member 2); a real weather forecast
    (GNSS member 2); what it took and what is missing, and the key figure with the conclusion (the team lead). Hand-overs: a sentence that asks the
    next presenter's question ('handover'), or, where a slide has none, the last sentence of the slide itself.
    Without the key figure (its drawing or results/key_figure.json is missing) the talk ends on the single closing slide s_end and the
    memory slide is slide 2: the same number of slides and the same 10 minutes.
    Backup slides (key 'backup' = their number; no time budget) come after the last talk slide; one whose figure or results file is
    missing is left out. A text may point to another slide as '[slide:key]'; the number is written in here."""
    with_key = _key(R) is not None
    talk = ((s_cover, s_data, s_fair, s_simple, s_all, s_event, s_verdict, s_fail, s_timing, s_rain, s_forecast, s_need, s_key) if with_key else
            (s_cover, s_memory, s_data, s_fair, s_simple, s_all, s_event, s_verdict, s_fail, s_timing, s_rain, s_forecast, s_end))
    deck = [f(R, T) for f in talk]
    assert len(deck) == TALK_SLIDES, f"the talk has {TALK_SLIDES} slides"
    assert sum(s["seconds"] for s in deck) == TOTAL_SECONDS, "the time budgets of the talk slides must add up to 10 minutes"
    backup = [s for s in (f(R, T) for f in ((b_curves, s_memory, b_lobo, b_rivers, b_scores, b_cost, b_space) if with_key else
                                            (b_curves, b_lobo, b_rivers, b_scores, b_cost, b_space))) if s]
    for j, s in enumerate(backup, 1):
        s.update(backup=j, seconds=0)
        s.pop("handover", None)
    deck += backup
    at = {s["key"]: (f"backup slide B{s['backup']}" if s.get("backup") else f"slide {i}") for i, s in enumerate(deck, 1)}
    put = lambda x: SLIDE_REF.sub(lambda m: at.get(m.group(1), "another slide"), x)
    for s in deck:
        for k in ("title", "caption", "caption_whole", "figure_words", "handover"):
            if s.get(k):
                s[k] = put(s[k])
        for k in ("bullets", "notes", "asked"):
            if s.get(k):
                s[k] = [put(x) for x in s[k]]
        if s.get("table"):
            s["table"]["rows"] = [[put(c) for c in row] for row in s["table"]["rows"]]
    return deck


def talk_slides(deck):
    """The slides of the talk itself: the deck without its backup slides."""
    return [s for s in deck if not s.get("backup")]


# ---- text: one small markup (<b>, <br>, [[VERDICT]], HTML entities) read by all three writers ------------------------------
TOKEN = re.compile(r"(<b>|</b>|\[\[.+?\]\])")


def _runs(s):
    """'a <b>b</b> [[GO]]' -> [(text, bold, colour or None), ...] in plain Unicode, for the PPTX and for measuring."""
    out, bold = [], False
    for part in TOKEN.split(s):
        if part == "<b>":
            bold = True
        elif part == "</b>":
            bold = False
        elif part.startswith("[["):
            out += [("\u25a0 ", False, TAG_COL.get(part[2:-2], AXIS)), (part[2:-2], True, None)]
        elif part:
            out.append((html.unescape(re.sub(r"<[^>]+>", "", part)), bold, None))
    return out


def _plain(s):
    return "".join(t for t, _, _ in _runs(s))


def _md(s):
    """The same markup as Markdown."""
    s = re.sub(r"\[\[(.+?)\]\]", r"**\1**", s).replace("<b>", "**").replace("</b>", "**").replace("<br>", " ")
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def _html(s):
    """The same markup as HTML: a verdict becomes the coloured label of the report; a cell of verdicts (nothing but labels) stays on one line, while a
    sentence that holds a label wraps as any other (the label itself never breaks)."""
    out = re.sub(r"\[\[(.+?)\]\]", lambda m: tag(m.group(1)), s)
    only_labels = not re.sub(r"\[\[.+?\]\]|&nbsp;|\s|\*", "", s)
    return f'<span class="nw">{out}</span>' if "[[" in s and only_labels else out


@lru_cache(maxsize=None)
def _face(bold):
    """The deck's typeface for measuring text (None if its file is not on this machine: widths are then estimated)."""
    for files in FONT_FILES:
        try:
            return ImageFont.truetype(files[bool(bold)], 100)
        except OSError:
            continue
    return None


def _width(text, size, bold):
    """Width of a piece of text in points."""
    f = _face(bold)
    return (f.getlength(text) if f else len(text) * (54 if bold else 50)) * size / 100


def _lines(s, width_in, size, bold=False, indent=0):
    """Number of lines a paragraph takes in a box of the given width (inches); '<br>' starts a new line."""
    total = 0
    for piece in s.split("<br>"):
        limit, line, n = (width_in * 72 - indent) * 0.97, 0.0, 1
        for text, b, _ in _runs(piece):
            for word in re.findall(r"\S+\s*", text):
                w = _width(word.rstrip(), size, bold or b)
                if line and line + w > limit:
                    n, line = n + 1, 0.0
                line += _width(word, size, bold or b)
        total += n
    return total


@lru_cache(maxsize=None)
def _crop(name, box):
    """One part of figures/<name>, cut out in memory: (PNG bytes, (width, height) in pixels), or None when it cannot be cut cleanly.
    box = (left, top, right, bottom) as fractions of the figure. Each edge is moved to the nearest row or column of plain background
    within SNAP pixels - the blank space between two panels - so the cut never runs through a letter, a line or a marker; an edge on the
    border of the figure stays there. If no such blank line is found (the figure was redrawn with another layout) the answer is None and
    the slide shows the whole figure. The PNG file is only read; nothing is redrawn."""
    im = Image.open(FIG / name)
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    blank = np.abs(a - a[0, 0]).sum(axis=2) <= 6                  # the background colour of the figure, read from its corner
    hh, ww = blank.shape
    l, t, r, b = round(box[0] * ww), round(box[1] * hh), round(box[2] * ww), round(box[3] * hh)

    def snap(pos, limit, is_blank):
        for d in range(SNAP + 1):
            for p in (pos - d, pos + d):
                if p <= 0:
                    return 0
                if p >= limit:
                    return limit
                if is_blank(p):
                    return p
        return None

    for _ in range(2):                                           # twice: an edge that has moved changes the stretch the others are tested on
        l = snap(l, ww, lambda x: blank[t:b, x].all())
        r = None if l is None else snap(r, ww, lambda x: blank[t:b, x].all())
        t = None if r is None else snap(t, hh, lambda y: blank[y, l:r].all())
        b = None if t is None else snap(b, hh, lambda y: blank[y, l:r].all())
        if b is None or r - l < 50 or b - t < 50:
            return None
    if blank[t:b, l:r].all():                                    # nothing but background inside: not a panel
        return None
    out = io.BytesIO()
    im.crop((l, t, r, b)).save(out, format="PNG")
    return out.getvalue(), (r - l, b - t)


def _image(s):
    """What a slide shows of its figure: (PNG bytes of the crop, or None for the whole file; (width, height) in pixels; is it a crop?)."""
    if s.get("crop"):
        cut = _crop(s["image"], tuple(s["crop"]))
        if cut:
            return cut[0], cut[1], True
    return None, Image.open(FIG / s["image"]).size, False


def _caption(s):
    """The label under a slide's figure: the one written for the crop, or the one for the whole figure when the crop was refused."""
    if not s.get("image"):
        return ""
    return s.get("caption", "") if not s.get("crop") or _image(s)[2] else s.get("caption_whole", "")


def _aspect(s):
    """Width / height of what a slide shows of its figure (a pending figure keeps the shape its script draws)."""
    if s.get("image"):
        w, h = _image(s)[1]
        return w / h
    return s.get("aspect", 1.0)


def _fit(aspect, box_w, box_h):
    w = min(box_w, box_h * aspect)
    return w, w / aspect


def _place(s, box_w, box_h):
    """Size of a slide's figure in a box that must also hold the label under it: (width, height, height of the label with its gap)."""
    aspect, cap = _aspect(s), _caption(s)
    cap_h = lambda w: (_lines(cap, w, CAP_PT) * CAP_PT * CAP_LH / 72 + CAP_GAP) if cap else 0.0
    fw, fh = _fit(aspect, box_w, box_h)
    for _ in range(3):                                           # the label's height depends on the figure's width, and the other way round
        fw, fh = _fit(aspect, box_w, box_h - cap_h(fw))
    ch = cap_h(fw)
    if fh + ch > box_h:                                          # the last pass made the label one line longer: give it the room
        fw, fh = _fit(aspect, fw, box_h - ch)
    return fw, fh, ch


def _cover_geometry(s, fig_w):
    """The first slide with a map of the given width: top line, title, blue rule, the question and the answer, the small paragraphs. A
    title that fits on one line runs across the slide, with the top line; the map then starts level with the question."""
    g, full = {}, W - 2 * MX
    fw, fh, ch = _place(s, fig_w, FIG_H)
    g.update(fw=fw, fh=fh, cap_h=ch, fx=W - FIG_RIGHT - fw, fy=FIG_TOP + (FIG_H - fh - ch) / 2)
    g["tw"] = g["fx"] - GAP - MX
    g["title_w"] = full if _lines(s["title"], full, COVER_PT, bold=True) == 1 else g["tw"]
    g["title_h"] = _lines(s["title"], g["title_w"], COVER_PT, bold=True) * COVER_PT * COVER_LH / 72
    y = COVER_TOP
    g["kicker_w"] = g["title_w"]                                  # the top line is as wide as the title
    g["kicker_h"] = _lines(s["kicker"], g["kicker_w"], SMALL_PT) * SMALL_PT * SMALL_LH / 72
    g["kicker_y"], y = y, y + g["kicker_h"] + COVER_GAPS[0]
    g["title_y"], y = y, y + g["title_h"] + COVER_GAPS[1]
    g["rule_y"], y = y, y + 0.05 + COVER_GAPS[2]
    if g["title_w"] == full:
        fw, fh, ch = _place(s, fig_w, FIG_TOP + FIG_H - y)
        g.update(fw=fw, fh=fh, cap_h=ch, fx=W - FIG_RIGHT - fw, fy=y)
    g["sub_h"] = _lines(s["sub"], g["tw"], SUB_PT) * SUB_PT * SUB_LH / 72
    g["sub_y"], y = y, y + g["sub_h"] + COVER_GAPS[3]
    g["small"] = []
    for x in s["small"]:
        h = _lines(x, g["tw"], SMALL_PT) * SMALL_PT * SMALL_LH / 72
        g["small"].append((y, h))
        y += h + COVER_GAPS[4]
    g["bullets_y"], g["bullets_h"] = y - COVER_GAPS[4], 0.0         # where the text ends: overflow() checks it like any other slide
    return g


def _geometry(s):
    """Where things go on a slide, in inches: used as is by the PPTX; the HTML uses the same boxes and lets the text flow."""
    g, full = {}, W - 2 * MX
    if s["layout"] == "cover":                                    # the widest map beside which the text still fits (long names need a wider text column)
        for fig_w in COVER_FIG_W:
            g = _cover_geometry(s, fig_w)
            if g["bullets_y"] <= BOTTOM:
                break
        return g
    if s["layout"] == "side":
        fw, fh, ch = _place(s, s.get("max_w", FIG_MAX_W), FIG_H)
        g.update(fw=fw, fh=fh, cap_h=ch, fx=W - FIG_RIGHT - fw, fy=FIG_TOP + (FIG_H - fh - ch) / 2)
        g["tw"] = g["fx"] - GAP - MX
        g["title_w"] = g["tw"]
    else:
        g["title_w"] = full
    g["title_h"] = _lines(s["title"], g["title_w"], TITLE_PT, bold=True) * TITLE_PT * TITLE_LH / 72
    g["title_y"] = TOP
    g["body_y"] = TOP + g["title_h"] + TITLE_GAP
    if s["layout"] == "wide":
        tx = s.get("tx_w", WIDE_TX)                               # the width of the text column: a slide may ask for a wider one ('tx_w', inches)
        fw, fh, ch = _place(s, full - tx - GAP, BOTTOM - g["body_y"])
        g.update(fw=fw, fh=fh, cap_h=ch, fx=W - MX - fw, fy=g["body_y"], tw=tx)
    if s["layout"] == "table":
        g["tw"] = full
        tb = s["table"]
        widths = [full * f for f in tb["widths"]]
        pad = 2 * CELL_PAD[0] / 72
        g["col_w"] = widths
        g["row_h"] = [(max(_lines(c, w - pad, TABLE_PT, bold=(i == 0)) for c, w in zip(row, widths)) * TABLE_PT * TABLE_LH + 2 * CELL_PAD[1] + 2) / 72
                      for i, row in enumerate([tb["header"]] + tb["rows"])]
        g["note_y"] = g["body_y"] + sum(g["row_h"]) + 0.08
        g["note_h"] = _lines(tb["note"], full, SMALL_PT) * SMALL_PT * NOTE_LH / 72 if tb.get("note") else 0.0
        g["bullets_y"] = g["note_y"] + g["note_h"] + 0.12
    else:
        g["bullets_y"] = g["body_y"]
    g["bullets_h"] = sum(_lines(b, g["tw"], BODY_PT, indent=BULLET_IN) * BODY_PT * BODY_LH + BULLET_AFTER for b in s.get("bullets", [])) / 72
    return g


def overflow(deck):
    """Slides whose text is estimated to run past the lowest content edge: [(slide number, inches too long)]."""
    out = []
    for i, s in enumerate(deck, 1):
        g = _geometry(s)
        over = g["bullets_y"] + g["bullets_h"] - BOTTOM
        if over > 0:
            out.append((i, round(over, 2)))
    return out


# ---- HTML (and from it the PDF) ----------------------------------------------------------------------------------------
def _figure_html(s, g):
    """The figure of a slide at the size the geometry gives it: the PNG as its script drew it (report_common.fig) or the exact crop the
    slide asks for, and under it the label that says what is shown."""
    size = f'width:{g["fw"]:.3f}in;height:{g["fh"]:.3f}in'
    if not s.get("image"):
        return f'<div class="pend" style="{size}">{s["pending"]}</div>'
    data, _, cropped = _image(s)
    picture = (f'<figure><img src="data:image/png;base64,{base64.b64encode(data).decode()}" alt="{s["image"]}: {_plain(s.get("panel", "a part"))}">'
               f'</figure>' if cropped else fig(s["image"], ""))
    cap = _caption(s)
    return picture.replace("<figure>", f'<figure style="{size}">', 1) + (f'<p class="cp">{cap}</p>' if cap else "")


def _ul(items):
    return '<ul class="pts">' + "".join(f"<li>{_html(b)}</li>" for b in items if b) + "</ul>"


def _foot(s, i, deck):
    """The two ends of a slide's footer: who presents the slide and its number in the talk - or, on a backup slide, that it is one."""
    n_talk = len(talk_slides(deck))
    if s.get("backup"):
        return f"{DECK} &middot; backup slide, shown only if asked", f"backup {s['backup']} / {len(deck) - n_talk}"
    return f"{DECK} &middot; presenter: {ROLE[s['owner']][0]}", f"{i} / {n_talk}"


def slide_html(s, i, deck):
    g = _geometry(s)
    left, right = _foot(s, i, deck)
    foot = f'<div class="ft"><span>{left}</span><span>{right}</span></div>'
    box = f'left:{g["fx"]:.3f}in;top:{g["fy"]:.3f}in;width:{g["fw"]:.3f}in' if "fx" in g else ""
    if s["layout"] == "cover":
        small = "".join(f'<p class="small">{x}</p>' for x in s["small"])
        return (f'<section class="slide cover"><div class="tx" style="width:{g["tw"]:.3f}in">'
                f'<p class="kick" style="width:{g["kicker_w"]:.3f}in">{s["kicker"]}</p>'
                f'<h1 style="width:{g["title_w"]:.3f}in">{s["title"]}</h1>'
                f'<div class="rule"></div><p class="sub">{s["sub"]}</p>{small}</div><div class="fg" style="{box}">{_figure_html(s, g)}</div>{foot}</section>')
    if s["layout"] == "side":
        return (f'<section class="slide side"><div class="tx" style="width:{g["tw"]:.3f}in"><h1>{s["title"]}</h1>{_ul(s["bullets"])}</div>'
                f'<div class="fg" style="{box}">{_figure_html(s, g)}</div>{foot}</section>')
    if s["layout"] == "wide":                                    # the figure keeps the size of the geometry and sits at the right, under the title
        return (f'<section class="slide wide"><div class="body" style="grid-template-columns:{g["tw"]:.3f}in minmax(0,1fr)"><h1>{s["title"]}</h1><div class="tx">{_ul(s["bullets"])}</div>'
                f'<div class="fg" style="width:{g["fw"]:.3f}in">{_figure_html(s, g)}</div></div>{foot}</section>')
    tb = s["table"]
    note = f'<p class="star">{tb["note"]}</p>' if tb.get("note") else ""
    grid = table([_html(c) for c in tb["header"]], [[_html(c) for c in row] for row in tb["rows"]], widths=[f"{100 * f:.1f}%" for f in tb["widths"]])
    return f'<section class="slide tab"><div class="body"><h1>{s["title"]}</h1>{grid}{note}{_ul(s["bullets"])}</div>{foot}</section>'


def deck_html(deck):
    return page(f"{DECK} - slides", "\n".join(slide_html(s, i, deck) for i, s in enumerate(deck, 1)), CSS_SLIDES)


# ---- PPTX: the same slides as editable shapes ---------------------------------------------------------------------------
def _rgb(hex_):
    return RGBColor.from_string(hex_.lstrip("#").upper())


def _frame(tf):
    tf.word_wrap, tf.auto_size, tf.vertical_anchor = True, MSO_AUTO_SIZE.NONE, MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def _para(tf, text, size, first, color=INK, bold=False, lh=BODY_LH, after=0, bullet=False, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment, p.line_spacing, p.space_after = align, Pt(round(size * lh, 1)), Pt(after)
    for seg, b, col in _runs(text):
        r = p.add_run()
        r.text = seg
        r.font.size, r.font.bold, r.font.name = Pt(size), bool(bold or b), PPT_FONT
        r.font.color.rgb = _rgb(col or color)
    if bullet:                                                   # a real bullet with a hanging indent, in the blue of the report
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Pt(BULLET_IN))))
        pPr.set("indent", str(-int(Pt(BULLET_IN))))
        for xml in (f'<a:buClr {nsdecls("a")}><a:srgbClr val="{BLUE.lstrip("#").upper()}"/></a:buClr>',
                    f'<a:buFont {nsdecls("a")} typeface="{PPT_FONT}"/>', f'<a:buChar {nsdecls("a")} char="&#8226;"/>'):
            pPr.append(parse_xml(xml))
    return p


def _textbox(slide, x, y, w, h, paras, size, **kw):
    tf = _frame(slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame)
    for j, text in enumerate(paras):
        _para(tf, text, size, j == 0, **kw)
    return tf


def _picture(slide, s, g):
    """The figure of a slide (the whole PNG or the crop the slide asks for) with the label under it, or a dashed box for a pending result."""
    if s.get("image"):
        data = _image(s)[0]
        pic = slide.shapes.add_picture(io.BytesIO(data) if data else str(FIG / s["image"]), Inches(g["fx"]), Inches(g["fy"]), width=Inches(g["fw"]),
                                       height=Inches(g["fh"]))
        if _caption(s):
            _textbox(slide, g["fx"], g["fy"] + g["fh"] + CAP_GAP, g["fw"], max(g["cap_h"] - CAP_GAP, 0.2), [_caption(s)], CAP_PT, color=INK2, lh=CAP_LH)
        return pic
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(g["fx"]), Inches(g["fy"]), Inches(g["fw"]), Inches(g["fh"]))
    shape.fill.background()
    shape.shadow.inherit = False
    shape.line.color.rgb, shape.line.width, shape.line.dash_style = _rgb(AXIS), Pt(1.5), MSO_LINE.DASH
    tf = shape.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, s["pending"], SUB_PT, True, color=INK2, align=PP_ALIGN.CENTER)
    return shape


def _table(slide, tb, g):
    cells = [tb["header"]] + tb["rows"]
    frame = slide.shapes.add_table(len(cells), len(tb["header"]), Inches(MX), Inches(g["body_y"]), Inches(W - 2 * MX), Inches(sum(g["row_h"])))
    t = frame.table
    t._tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"       # 'No Style, No Grid': the rules are drawn below
    t.first_row = t.horz_banding = False
    for j, w in enumerate(g["col_w"]):
        t.columns[j].width = Inches(w)
    rule = lambda side: parse_xml(f'<a:{side} {nsdecls("a")} w="9525"><a:solidFill><a:srgbClr val="{GRID.lstrip("#").upper()}"/></a:solidFill></a:{side}>')
    for i, row in enumerate(cells):
        t.rows[i].height = Inches(g["row_h"][i])
        for j, text in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Pt(CELL_PAD[0])
            c.margin_top = c.margin_bottom = Pt(CELL_PAD[1])
            c.vertical_anchor = MSO_ANCHOR.TOP
            tf = c.text_frame
            tf.word_wrap = True
            for q, piece in enumerate(text.split("<br>")):
                _para(tf, piece, TABLE_PT, q == 0, color=INK2 if i == 0 else INK, bold=(i == 0), lh=TABLE_LH)
            tcPr = c._tc.get_or_add_tcPr()
            tcPr.insert(0, rule("lnT"))                               # a hairline above every row, as in the report's tables
            if i == len(cells) - 1:
                tcPr.insert(1, rule("lnB"))
            c.fill.background()
    return frame


def _footer(slide, s, i, deck):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(MX), Inches(FOOT), Inches(W - MX), Inches(FOOT))
    line.line.color.rgb, line.line.width = _rgb(GRID), Pt(0.75)
    left, right = _foot(s, i, deck)
    _textbox(slide, MX, FOOT + 0.06, 9.0, 0.25, [left], FOOT_PT, color=MUTED)
    tf = _frame(slide.shapes.add_textbox(Inches(W - MX - 2.0), Inches(FOOT + 0.06), Inches(2.0), Inches(0.25)).text_frame)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    if s.get("backup"):                                              # a backup slide has no number in the talk: plain text
        tail = right
    else:                                                            # the slide number is a field: it follows if slides are moved
        rpr = f'<a:rPr {nsdecls("a")} lang="en-GB" sz="{FOOT_PT * 100}"><a:solidFill><a:srgbClr val="{MUTED.lstrip("#").upper()}"/></a:solidFill><a:latin typeface="{PPT_FONT}"/></a:rPr>'
        p._p.append(parse_xml(f'<a:fld {nsdecls("a")} id="{{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}}" type="slidenum">{rpr}<a:t>{i}</a:t></a:fld>'))
        tail = right[len(str(i)):]                                   # ' / 13'
    r = p.add_run()
    r.text = tail
    r.font.size, r.font.name = Pt(FOOT_PT), PPT_FONT
    r.font.color.rgb = _rgb(MUTED)


def notes_text(s, i):
    """Speaker notes of one slide as plain text (the notes pane of the PPTX)."""
    who, what = ROLE[s["owner"]]
    if s.get("backup"):
        lines = [f"Backup slide B{s['backup']} (slide {i} of the file). Shown only if a question asks for it; it has no time in the talk.", "",
                 "If shown, say:"]
    else:
        lines = [f"Slide {i}. Presenter: {_plain(who)} ({what}). Time: {s['seconds']} s.", "", "Say:"]
    lines += [f"- {_plain(x)}" for x in s["notes"] if x]
    if s.get("handover"):
        lines += ["", "Hand-over: " + _plain(s["handover"])]
    if any(s.get("asked", [])):
        lines += ["", "If asked:"] + [f"- {_plain(x)}" for x in s["asked"] if x]
    if s.get("figure_words"):
        lines += ["", "Reading the slide: " + _plain(s["figure_words"])]
    return "\n".join(lines)


def _said_aloud(s):
    """Everything a presenter says on a slide: the sentences under 'Say' and the hand-over sentence."""
    return [x for x in list(s["notes"]) + [s.get("handover", "")] if x]


def spoken(s):
    """How long a slide is when spoken, in words: its 'say' sentences and its hand-over sentence. A number with a decimal point or a
    thousands comma counts as three words ('0.59' is said 'zero point five nine')."""
    text = " ".join(_plain(x) for x in _said_aloud(s))
    return len(text.split()) + 2 * len(re.findall(r"\d[.,]\d", text))


def too_fast(s):
    """True when the words to say on a slide need more than PACE words a second (compared in tenths of a word, so that rounding of the
    product cannot decide it)."""
    return bool(s["seconds"]) and 10 * spoken(s) > round(10 * PACE) * s["seconds"]


def long_sentences(s):
    """The sentences to say on a slide that are longer than MAX_SENTENCE words: [(words, sentence), ...]."""
    out = []
    for x in _said_aloud(s):
        for sentence in re.split(r"(?<=[.!?])\s+", _plain(x).strip()):
            if len(sentence.split()) > MAX_SENTENCE:
                out.append((len(sentence.split()), sentence))
    return out


def deck_pptx(deck, path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    cp = prs.core_properties
    cp.title, cp.subject = DECK, "AI course group project: slides of the 10-minute presentation"
    cp.author = "built by code/build_slides.py (written by Claude Code, the team's coding assistant, under the team lead's direction)"
    cp.last_modified_by, cp.created, cp.modified = "code/build_slides.py", datetime.datetime.now(), datetime.datetime.now()
    layout = prs.slide_layouts[5]                                     # 'Title Only': the title stays a real title for the outline view
    for i, s in enumerate(deck, 1):
        slide = prs.slides.add_slide(layout)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _rgb(SURFACE)
        g = _geometry(s)
        cover = s["layout"] == "cover"
        title = slide.shapes.title
        title.left, title.top, title.width, title.height = Inches(MX), Inches(g["title_y"]), Inches(g["title_w"]), Inches(g["title_h"])
        _para(_frame(title.text_frame), s["title"], COVER_PT if cover else TITLE_PT, True, bold=True, lh=COVER_LH if cover else TITLE_LH)
        if cover:
            _textbox(slide, MX, g["kicker_y"], g["kicker_w"], g["kicker_h"], [s["kicker"]], SMALL_PT, color=INK2, lh=SMALL_LH)
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MX), Inches(g["rule_y"]), Inches(1.1), Inches(0.05))
            bar.fill.solid()
            bar.fill.fore_color.rgb = _rgb(BLUE)
            bar.line.fill.background()
            bar.shadow.inherit = False
            _textbox(slide, MX, g["sub_y"], g["tw"], g["sub_h"], s["sub"].split("<br>"), SUB_PT, lh=SUB_LH)      # '<br>' starts a new paragraph
            for x, (y, h) in zip(s["small"], g["small"]):
                _textbox(slide, MX, y, g["tw"], h, [x], SMALL_PT, color=INK2, lh=SMALL_LH)
        if s["layout"] == "table":
            _table(slide, s["table"], g)
            if s["table"].get("note"):
                _textbox(slide, MX, g["note_y"], g["tw"], max(g["note_h"], 0.3), [s["table"]["note"]], SMALL_PT, color=INK2, lh=NOTE_LH)
        if "fx" in g:
            _picture(slide, s, g)
        bullets = [b for b in s.get("bullets", []) if b]
        if bullets:
            _textbox(slide, MX, g["bullets_y"], g["tw"], max(g["bullets_h"], 0.4), bullets, BODY_PT, bullet=True, after=BULLET_AFTER)
        _footer(slide, s, i, deck)
        slide.notes_slide.notes_text_frame.text = notes_text(s, i)
    prs.save(str(path))


def check_pptx(path, deck):
    """Open the PPTX again: the slide count, the title of every slide (the same as in the HTML and the PDF, which come from the same
    list), every shape inside the slide, no picture distorted, speaker notes on every slide. Returns a list of problems (empty = fine)."""
    prs = Presentation(str(path))
    bad = [] if len(prs.slides) == len(deck) else [f"{len(prs.slides)} slides instead of {len(deck)}"]
    for i, (slide, s) in enumerate(zip(prs.slides, deck), 1):
        if slide.shapes.title is None or slide.shapes.title.text_frame.text.strip() != _plain(s["title"]).strip():
            bad.append(f"slide {i}: its title is not the title of the slide list")
        for sh in slide.shapes:
            if sh.left < 0 or sh.top < 0 or sh.left + sh.width > prs.slide_width or sh.top + sh.height > prs.slide_height:
                bad.append(f"slide {i}: '{sh.name}' reaches outside the slide")
            if sh.shape_type is not None and "PICTURE" in str(sh.shape_type):
                w, h = sh.image.size
                if abs(sh.width / sh.height - w / h) > 0.01 * w / h:
                    bad.append(f"slide {i}: picture '{sh.name}' is distorted")
        if s.get("image") and not any(sh.shape_type is not None and "PICTURE" in str(sh.shape_type) for sh in slide.shapes):
            bad.append(f"slide {i}: its figure is missing")
        if not slide.has_notes_slide or not slide.notes_slide.notes_text_frame.text.strip():
            bad.append(f"slide {i}: no speaker notes")
    return bad


# ---- speaker notes as Markdown -------------------------------------------------------------------------------------------
def _full_reference(k):
    """A full reference with the letter of its short form in the year ('Kratzert et al., 2019b' -> '... (2019b). ...'), so that a citation
    in the text can be found in the list."""
    short, full = REFERENCES[k]
    m = re.search(r"(\d{4})([a-z])$", short)
    return full.replace(f"({m.group(1)})", f"({m.group(1)}{m.group(2)})", 1) if m else full


DIGIT = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", ".": "point"}


def _first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", _md(text).strip())[0]


def _shows(s):
    """What a slide shows, for the notes: the figure file (and the part of it), a table, or text."""
    if s.get("image") and s.get("crop") and _image(s)[2]:
        shows = (f"figure `figures/{s['image']}`: {_md(s.get('panel', 'a part of it'))}, cut out and enlarged by the builder (the file "
                 f"itself is not changed)")
    else:
        shows = (f"figure `figures/{s['image']}`" if s.get("image") else "a table" if s.get("table") else
                 f"the figure is not there yet ({_md(s['pending'])})" if s.get("pending") else "text only")
    return f"**On the slide:** {shows}" + (f"; {_md(s['figure_words'])}" if s.get("figure_words") else ".")


def _before_you_speak(talk, R):
    """The half page a presenter reads before the talk: how the numbers and names are said, three times to compare with the clock, the one
    agreed cut if the talk runs late, and every presenter's first words. All of it is worked out from the slides and the results."""
    clock = lambda sec: f"{sec // 60}:{sec % 60:02d}"
    name = lambda k: _md(ROLE[k][0])
    start, t = [], 0
    for s in talk:
        start.append(t)
        t += s["seconds"]
    A, cmp_, _, _ = _all_basins(R)
    score = _md(n2(cmp_("wq", LEADS[0])["lstm_nse_median"]))
    aloud = " ".join(DIGIT.get(c, c) for c in score)
    e = R["flood_analysis"].get("event_shown")
    full = ""
    if e and ", ".join(x for x in _said(e["basin"]) if x) != e["basin"]:
        full = f" {e['basin']} is said in full: {', '.join(x for x in _said(e['basin']) if x)}."
    Wd, hour = _windows(R), ""
    if Wd and re.fullmatch(r"\d\d:00", Wd["daymet"]) and int(Wd["daymet"][:2]) in NUMBER:
        hour = f" {Wd['daymet']} is said \u2018{NUMBER[int(Wd['daymet'][:2])]} o\u2019clock\u2019."
    out = ["## Before you speak", "",
           f"**Saying it aloud.** A score such as {score} is said digit by digit: \u2018{aloud}\u2019. NSE and LSTM are spelled out, letter by "
           f"letter, and so are AORC, GEFS and NLDAS; ERA5 is said \u2018E R A five\u2019 and Stage IV \u2018Stage four\u2019.{hour}{full}", ""]
    # ---- three times to compare with the clock: where the third, the fifth and the last presenter block start (or every block start, if
    #      there are fewer); a block is a run of slides with one presenter
    blocks = [i for i, s in enumerate(talk) if i and s["owner"] != talk[i - 1]["owner"]]
    marks = [blocks[1], blocks[3], blocks[-1]] if len(blocks) >= 5 else blocks
    if marks:
        out += ["**Clock checkpoints.** " + " ".join(f"When slide {i + 1} starts ({name(talk[i]['owner'])}), the clock should show {clock(start[i])}."
                                                    for i in sorted(set(marks))), ""]
    # ---- the one agreed cut: the slide that says what it would lose in one sentence ('if_late')
    cut = next((i for i, s in enumerate(talk) if s.get("if_late")), None)
    if cut is not None:
        s = talk[cut]
        words = len(_plain(s["if_late"]).split()) + len(_plain(s.get("handover", "")).split())
        saved = 5 * int((s["seconds"] - words / PACE) // 5)
        out += [f"**If the talk runs late.** If the clock shows more than {clock(start[cut] + LATE)} when slide {cut + 1} starts, {name(s['owner'])} "
                f"says only: \u201c{_md(s['if_late'])}\u201d" + (" - and then the hand-over sentence." if s.get("handover") else "")
                + f" That saves about {saved} s. No other sentence is skipped: the other sentences carry the caveats.", ""]
    out += ["**Your first words.** So that each presenter can start without reading:", ""]
    for k in dict.fromkeys(s["owner"] for s in talk):
        out.append(f"- {name(k)}")
        out += [f"  - slide {i}: \u201c{_first_sentence(s['notes'][0])}\u201d" for i, s in enumerate(talk, 1) if s["owner"] == k and s["notes"]]
    return out + [""]


def plan_shares(talk):
    """{presenter key: share of the talk's seconds}, from the slides' own time budgets; only the presenters who have a slide."""
    total = sum(s["seconds"] for s in talk)
    return {k: sum(s["seconds"] for s in talk if s["owner"] == k) / total for k in ROLE if any(s["owner"] == k for s in talk)}


def _presenters_sentence(talk, at):
    """Why the slides are shared as they are, said from the owners of the slides: the plan names no owner for the rain-timing test, for where the onset warnings fail
    or for the experiments of 7 and 8 October, and gives the flood analysis to the whole team, so the project lead shared those slides out; and how much of the talk the
    presenter who opens and closes it speaks for (S0.share_word: 'about half' where the share lies near one half)."""
    clock = lambda sec: f"{sec // 60}:{sec % 60:02d}"
    total = sum(s["seconds"] for s in talk)
    free = sorted(at[k] for k in ("fail", "timing", "rain", "forecast") if k in at)
    mine = sum(s["seconds"] for s in talk if s["owner"] == "A")
    opens = talk[0]["owner"] == "A" and talk[-1]["owner"] == "A"
    return ((f"The plan names no owner for the rain-timing test, for where the onset warnings fail or for the experiments of 7 and 8 October (slides {_and(free)}), and "
             f"it gives the flood analysis to the whole team; the project lead shared those slides out. " if free else "")
            + f"{ROLE['A'][0]} {'opens and closes the talk and ' if opens else ''}speaks for {clock(mine)} of the {clock(total)}, {S0.share_word(mine / total)}.")


def notes_md(deck, R):
    clock = lambda sec: f"{sec // 60}:{sec % 60:02d}"
    name = lambda k: _md(ROLE[k][0])
    talk, backup = talk_slides(deck), [s for s in deck if s.get("backup")]
    total = sum(s["seconds"] for s in talk)
    _, check = _team_check(R)
    audit_day, run_day = _days(R)
    when = (f"data audit and choice of the basins: {audit_day}; experiments: {run_day}" if audit_day and audit_day != run_day else
            f"experiments: {run_day}")
    NWK = S0.new_work(R)                                          # the experiments of 7 and 8 October 2026 and their independent AI reviews
    if NWK:
        when += f"; {_word(NWK['n'])} more experiments: {NWK['days']}"
    at = {s["key"]: i for i, s in enumerate(talk, 1)}
    n_b = len(backup)
    example = _md(n2(_all_basins(R)[1]("wq", LEADS[0])["lstm_nse_median"]))      # a score of the talk, as an example of how numbers are counted
    out = [f"# Speaker notes: {DECK}", "",
           f"Built on {_date(datetime.date.today().isoformat())} by `python code/build_slides.py` together with `Flood_Slides.html`, `Flood_Slides.pdf` and "
           f"`Flood_Slides.pptx`, from one list of slides. Every number below is read from `results/*.json` and `results/*.csv`; do not edit this "
           f"file by hand, rebuild it after a rerun.", "",
           f"**Present from `Flood_Slides.pdf` in full-screen mode.** It is the version that was looked at page by page when the slides were last "
           f"revised ({_revised()}, by Claude Code). `Flood_Slides.pptx` is the editable copy with the speaker notes: the same {len(deck)} slides, "
           f"positions, sizes and typeface (the builder checks this), but it was not opened in PowerPoint on the build machine, and its verdicts "
           f"are text marks instead of labels. If the PPTX is used, open every slide once before the talk.", "",
           f"**{len(talk)} slides, {clock(total)} minutes" + (f", then {_word(n_b)} backup slide{'s' if n_b != 1 else ''}" if n_b else "") + ".** "
           f"Under *Say* are the short sentences of each slide, with the numbers. They fit the time at a calm pace: at most {PACE:g} words a second "
           f"with the hand-over sentence counted (a number such as {example} counts as three words), and no sentence longer than {MAX_SENTENCE} words. "
           f"*Hand-over* is the one sentence that passes the talk to the next presenter. Under *If asked* are facts for questions; they are not "
           f"part of the 10 minutes." + (" The backup slides come after the last slide of the talk and are shown only if a question asks for "
                                         "them." if n_b else ""), "",
           _md(f"**Who did what - say it plainly if asked.** Claude Code, the team&rsquo;s coding assistant, wrote all code and ran all experiments "
               f"under the team lead&rsquo;s direction ({when}; these slides were last revised on {_revised()}, also by Claude Code). "
               + (f"{_review_sentence(R)} The review of the shared engine is `results/engine_review.json`. " if _review_sentence(R) else
                  "Independent AI reviewer agents checked the shared engine and every script (`results/engine_review.json`). ")
               + f"{check} The presenters below follow the roles of `PLAN.md` "
               f"where the plan names one, and the project lead’s choice where it does not: a presenter speaks for a slide and answers questions on it; it does "
               f"not mean that this member wrote or checked it. {_presenters_sentence(talk, at)}"), "",
           "| Slide | Presenter (role in the plan) | Time | Runs to | The slide&rsquo;s message |".replace("&rsquo;", "\u2019"), "|---|---|---|---|---|"]
    t = 0
    for i, s in enumerate(talk, 1):
        t += s["seconds"]
        out.append(f"| {i} | {name(s['owner'])} | {s['seconds']} s | {clock(t)} | {_md(s['title'])} |")
    share = plan_shares(talk)
    out += [f"| | | **{total} s** | **{clock(total)}** | |", "", "Time per presenter: " + "; ".join(
        f"{name(k)} {clock(sum(s['seconds'] for s in talk if s['owner'] == k))}, {pct(share[k])} (slides {_and(i for i, s in enumerate(talk, 1) if s['owner'] == k)})"
        for k in ROLE if any(s["owner"] == k for s in talk)) + ".", ""]
    out += _before_you_speak(talk, R)
    t = 0
    for i, s in enumerate(talk, 1):
        who, what = ROLE[s["owner"]]
        out += [f"## Slide {i}. {_md(s['title'])}", "",
                f"**Presenter:** {_md(who)} - {what}. **Time:** {s['seconds']} s ({clock(t)} to {clock(t + s['seconds'])}).", "", "**Say:**", ""]
        out += [f"{j}. {_md(x)}" for j, x in enumerate([x for x in s["notes"] if x], 1)]
        if s.get("handover"):
            nxt = talk[i]["owner"] if i < len(talk) else None
            out += ["", f"**Hand-over{' to ' + name(nxt) if nxt and nxt != s['owner'] else ''}:** {_md(s['handover'])}"]
        elif i < len(talk) and talk[i]["owner"] != s["owner"]:            # a slide with no hand-over sentence of its own: its last sentence hands over
            out += ["", f"**Hand-over to {name(talk[i]['owner'])}:** the last sentence above."]
        if any(s.get("asked", [])):
            out += ["", "**If asked:**", ""] + [f"- {_md(x)}" for x in s["asked"] if x]
        out += ["", _shows(s), ""]
        t += s["seconds"]
    if backup:
        out += ["## Backup slides (only if asked)", "",
                f"These {_word(n_b)} slide{'s' if n_b != 1 else ''} come after slide {len(talk)} in the HTML, the PDF and the PPTX. They have no time "
                f"in the talk: show one only if a question asks for it, and say then what stands under *If shown, say*.", ""]
        for i, s in enumerate(deck, 1):
            if not s.get("backup"):
                continue
            out += [f"### Backup slide B{s['backup']} (slide {i} of the file). {_md(s['title'])}", "", "**If shown, say:**", ""]
            out += [f"{j}. {_md(x)}" for j, x in enumerate([x for x in s["notes"] if x], 1)]
            if any(s.get("asked", [])):
                out += ["", "**If asked:**", ""] + [f"- {_md(x)}" for x in s["asked"] if x]
            out += ["", _shows(s), ""]
    if CITED:
        out += ["## Sources cited on the slides and in these notes", ""] + [f"- {_md(_full_reference(k))}" for k in sorted(CITED, key=lambda k: REFERENCES[k][1])] + [""]
    return "\n".join(out)


def slide_plan(deck, R):
    """The plan of the talk as data (results/slide_plan.json): for every slide of the file the presenter, the time (from when to when, for a talk slide), the figure, the
    sentences to say, the hand-over sentence and the cut for a late talk; the presenters with their slides, seconds and share of the talk; the clock checkpoints and the
    one agreed cut, worked out by the rules _before_you_speak writes into slides_notes.md (main() checks that the two say the same). Plain text, no markup."""
    clock = lambda sec: f"{sec // 60}:{sec % 60:02d}"
    plain = lambda x: _md(x).replace("**", "")
    talk = talk_slides(deck)
    total = sum(s["seconds"] for s in talk)
    start, t = [], 0
    for s in talk:
        start.append(t)
        t += s["seconds"]
    roles = {d["key"]: d["role"] for d in team.DEFAULT}
    rows = []
    for i, s in enumerate(deck, 1):
        row = dict(slide=i, key=s["key"], kind="backup" if s.get("backup") else "talk", backup=s.get("backup"), title=plain(s["title"]), presenter=s["owner"],
                   presenter_role=roles[s["owner"]], presenter_label=team.label(s["owner"]), job=JOB[s["owner"]], seconds=s["seconds"], figure=s.get("image"),
                   figure_part=plain(s["panel"]) if s.get("panel") and s.get("crop") else None, has_table=bool(s.get("table")),
                   say=[plain(x) for x in s["notes"] if x], handover=plain(s["handover"]) if s.get("handover") else "",
                   if_late=plain(s["if_late"]) if s.get("if_late") else "")
        if not s.get("backup"):
            row.update(start_s=start[i - 1], end_s=start[i - 1] + s["seconds"], start=clock(start[i - 1]), end=clock(start[i - 1] + s["seconds"]))
        rows.append(row)
    share = plan_shares(talk)
    presenters = [dict(key=k, role=roles[k], label=team.label(k), job=JOB[k], slides=[i for i, s in enumerate(talk, 1) if s["owner"] == k],
                       seconds=sum(s["seconds"] for s in talk if s["owner"] == k), share=share[k]) for k in ROLE if k in share]
    blocks = [i for i, s in enumerate(talk) if i and s["owner"] != talk[i - 1]["owner"]]              # the same rule as _before_you_speak
    marks = sorted(set([blocks[1], blocks[3], blocks[-1]] if len(blocks) >= 5 else blocks))
    checkpoints = [dict(slide=i + 1, presenter=talk[i]["owner"], presenter_label=team.label(talk[i]["owner"]), clock=clock(start[i])) for i in marks]
    cut = next((i for i, s in enumerate(talk) if s.get("if_late")), None)
    late_cut = None
    if cut is not None:
        s = talk[cut]
        words = len(_plain(s["if_late"]).split()) + len(_plain(s.get("handover", "")).split())
        late_cut = dict(slide=cut + 1, presenter=s["owner"], presenter_label=team.label(s["owner"]), clock_over=clock(start[cut] + LATE), seconds_over=start[cut] + LATE,
                        say=plain(s["if_late"]), saves_seconds=5 * int((s["seconds"] - words / PACE) // 5), then_handover=bool(s.get("handover")))
    return dict(written_by="code/build_slides.py", built=datetime.date.today().isoformat(), total_seconds=total, talk_slides=len(talk), backup_slides=len(deck) - len(talk),
                words_per_second=PACE, late_threshold_seconds=LATE, presenters=presenters, slides=rows, clock_checkpoints=checkpoints, late_cut=late_cut)


def main():
    R, T = load_results()
    deck = slides(R, T)
    talk = talk_slides(deck)
    out = ROOT / "Flood_Slides.html"
    out.write_text(deck_html(deck), encoding="utf-8")
    print("wrote", out, f"({len(talk)} talk slides, {sum(s['seconds'] for s in talk)} s; {len(deck) - len(talk)} backup slides)")
    if print_pdf(out, out.with_suffix(".pdf")):
        print("wrote", out.with_suffix(".pdf"))
    deck_pptx(deck, out.with_suffix(".pptx"))
    print("wrote", out.with_suffix(".pptx"))
    notes = notes_md(deck, R)
    (ROOT / "slides_notes.md").write_text(notes, encoding="utf-8")
    print("wrote", ROOT / "slides_notes.md")
    plan = slide_plan(deck, R)
    (RES / "slide_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", RES / "slide_plan.json", "(" + ", ".join(f"{p['label']} {p['seconds']} s, {pct(p['share'])}" for p in plan["presenters"]) + ")")
    said = lambda k: _md(ROLE[k][0])                                  # the notes and the plan file are made by two small pieces of code: say so if they ever differ
    for c in plan["clock_checkpoints"]:
        if f"When slide {c['slide']} starts ({said(c['presenter'])}), the clock should show {c['clock']}." not in notes:
            print(f"slide_plan.json and slides_notes.md disagree on the clock checkpoint of slide {c['slide']}", file=sys.stderr)
    if plan["late_cut"] and f"more than {plan['late_cut']['clock_over']} when slide {plan['late_cut']['slide']} starts" not in notes:
        print("slide_plan.json and slides_notes.md disagree on the cut for a late talk", file=sys.stderr)
    if plan["late_cut"] and f"That saves about {plan['late_cut']['saves_seconds']} s." not in notes:
        print("slide_plan.json and slides_notes.md disagree on the seconds the cut saves", file=sys.stderr)
    for problem in check_pptx(out.with_suffix(".pptx"), deck):
        print("PPTX check:", problem, file=sys.stderr)
    for i, over in overflow(deck):
        print(f"slide {i}: the text is estimated to run {over} in past the content area - shorten it", file=sys.stderr)
    for i, s in enumerate(deck, 1):
        if s.get("image") and s["layout"] != "cover":
            g = _geometry(s)
            cut = f", {_plain(s['panel'])}" if s.get("crop") and _image(s)[2] else ""
            print(f"slide {i}: figures/{s['image']}{cut} at {g['fw']:.1f} x {g['fh']:.1f} in ({100 * g['fw'] / W:.0f}% of the slide's width)")
        if s.get("image") and s.get("crop") and not _image(s)[2]:
            print(f"slide {i}: the crop of figures/{s['image']} was refused (no blank space where the cut should run) - the whole figure is "
                  f"shown; check the slide", file=sys.stderr)
        if too_fast(s):
            print(f"slide {i}: {spoken(s)} words to say in {s['seconds']} s - more than {PACE:g} a second, shorten the notes", file=sys.stderr)
        for words, sentence in long_sentences(s):
            print(f"slide {i}: a sentence to say has {words} words, more than {MAX_SENTENCE}: '{sentence}'", file=sys.stderr)
    key = next((s for s in talk if s["key"] == "key"), None)
    if key and key["image"] != "key_figure_talk.png":
        print("the last slide shows figures/key_figure.png, the figure of the report: the talk-size drawing figures/key_figure_talk.png is not there")
    print("words to say / words the time allows: " + ", ".join(f"slide {i} {spoken(s)}/{round(10 * PACE) * s['seconds'] // 10}" for i, s in enumerate(talk, 1)))
    for line in LEFT_OUT:
        print(line, file=sys.stderr)
    pending = [s["key"] for s in deck if "result pending" in " ".join([s["title"]] + s.get("bullets", []) + [str(s.get("table", ""))])]
    if pending:
        print("result pending on slides:", ", ".join(pending), file=sys.stderr)


if __name__ == "__main__":
    main()
