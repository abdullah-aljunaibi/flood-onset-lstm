"""Steps 14 and 15 of the step-by-step guide, and its last pages (build_guide.py -> Project_Guide.html / .pdf).

  14  How sure are these numbers?          two kinds of doubt: another training (the seeds), and other years and rivers (resampling);
                                           what no range covers
  15  What we found, and what it teaches   the finding word for word, the key figure read in four looks (the fourth is the rows added on 7 and
                                           8 October), why a milestone that was not met is still a result, the conclusions in one sentence
                                           each, the lessons about applying AI, and a pointer to what came next
  16  What we did next, and what it showed  written by g5_next.py and put in here, after step 15, so that the list of sources (which is
                                           made from everything cited before it) stays last
  17-23 The work after the mid-term talk    written by g6_final.py and put in here after step 16, for the same reason
  The concepts on one page                 the sentences of guide_common.concepts_list(), set in two columns so that the heading is true;
                                           under it 'The words your course uses': the plain words of the guide beside the terms of an AI
                                           course (TERMS)
  Run it yourself                          the commands of README.md and of the hand-in (the final report, the members' check), and what the folder
                                           cannot do (the networks of the first version are not stored)
  Who did what                             who did what in the final state, the AI-assistance declaration of the final report word for word (Appendix E,
                                           read from the .docx by g6_final.declaration(); without it, the statement of the earlier report, s0_summary.who),
                                           the state of the team's own checks, the sources
Every step ends with the pointer to the final report (guide_common.full_detail), by the section numbers that its .docx prints.

Written for readers who are new to machine learning: short sentences, plain words, a technical word only after its concept box.
Nothing is computed here beyond differences, counts and formatting: every number is read by key from results/*.json when the guide
is built (uncertainty.json, lstm_all.json, key_figure.json, flood_analysis.json, model_comparison.json, rain_timing.json,
lstm_lobo.json, baselines.json, fair_test.json and data_audit.json; several of them through the verified helpers of
report_sections/s0_summary.py).
A sentence that states an outcome ('no clear difference', 'NOT YET in every draw', 'clearly worse', ...) is printed only while the
stored numbers support it, by the same checks as section 8 of the report; otherwise that sentence is left out and one line on stderr
says why. The same holds for a results file or a key that is missing.
The source list is report_common.reference_list(): everything the parts before this one cited through cite(), so this part must be
built last (build_guide.SECTIONS has it last).

One thing here is not in the brief of the guide: the drawing of step 14 (the two main results with their ranges, drawn from
results/uncertainty.json). The brief lists no picture for that step; SHOW_RANGES = False takes the drawing out, and the text stands
without it.
"""
import html as _html
import re

import guide_common as G
import team
from common import AXIS, BLUE, FIG, GRID, INK, INK2, MUTED, ROOT, SEQ, SURFACE
from flood import LEADS
from report_common import CITED, LEAD_WORD, REFERENCES, TAG_TINT, cite, count, fig, n2, pct, reference_list, signed, table, tag
from report_sections.s0_summary import _leads
from guide_sections.g5_next import available as step16_available, html as step16_html
from guide_sections.g6_final import available as final_available, config, declaration, html as final_html, report_references
from report_sections.s0_summary import (AORC_WORDING, FINDING, REVIEW_WORDING, SUPPORT, _esc, criteria, facts, hindsight_clause, lobo_facts, new_work, note, onset_clear,
                                         review_sentence, timing_facts, who)
from report_sections.s1_data import _svg, _txt
from report_sections.s7_cost import _project_cost

SHOW_RANGES = True                               # the drawing of step 14 (not in the brief: see the note at the top of this file)
KEY_FIG = "key_figure.png"                       # the key figure with its own title, legend and footnote (11_key_figure.py)
# the names of the two pictures, with which their captions start (the text speaks of 'the picture' of step 14 and 'the next picture' of step 15)
RANGES_PIC, KEY_PIC = "The two main results of the project, with their ranges", "The main finding in one picture"
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
FIRST = LEADS[0]
lk = lambda k: f"l{k}"
# the three versions of the LSTM whose onset warnings step 14 shows with their ranges: (prediction set, the two lines of its label);
# the last one is the upper bound
SETS = (("lstm_wq", ("past weather + river gauge", "")),
        ("lstm_wqn", ("+ rain up to the evening", "of the issue day")),
        ("lstm_wqa", ("+ rain up to midnight", "(AORC, made after the fact)")),
        ("lstm_wqf", ("+ perfect weather forecast", "(upper bound)")))
TIE = "nse_median_of_diffs.lstm_wq_minus_best_simple"             # the LSTM's NSE minus that of the best simple model, in the typical basin
# layout only (the rule of the report, s0_summary.LAYOUT, as the other parts of the guide apply it): a picture is a full-width float, so
# that when it does not fit on the rest of a page it moves to the top of the next page and the text after it fills the page; every heading,
# box, table and block that must not be split clears the floats, so it starts below the picture before it, and the part itself starts below
# the pictures of the part before it. Then the lists of step 15, the two columns of the concept list, the table of commands, the sources.
# A list of step 15 is never cut in two. The conclusions are one block that does not clear the key figure (.together), so that they fill
# the page in front of a figure that has moved to the next page. The lessons are one block below the figure, together with the end of the
# step (guide_common.keep): 'What to remember' never stands alone on the page in front of the concept list, which always starts a new page.
# The concept list is set small enough to hold on one page with a few lines to spare; look at that page after adding a concept.
STYLE = ("<style>.g4{clear:both} .g4 .ff{float:left;width:100%;break-inside:avoid} "
         ".g4 h2,.g4 .concept,.g4 .remember,.g4 .try,.g4 .keep,.g4 table,.g4 .who{clear:both} .g4 .keep{break-inside:avoid} .g4 p.intro{break-after:avoid} "
         ".g4 .together{break-inside:avoid} "
         ".g4 ol.short,.g4 ul.short{margin:2pt 0 6pt;padding-left:17pt} .g4 .short li{margin-bottom:2pt} "
         ".g4 table.terms{margin:3pt 0 6pt} .g4 table.terms td,.g4 table.terms th{padding:2.2pt 6pt} "
         ".g4 .cols{columns:2;column-gap:18pt;column-rule:1px solid var(--grid);font-size:7.55pt;line-height:1.27} "
         ".g4 .cols p{margin:0 0 3pt;break-inside:avoid} "
         ".g4 table.cmds td:first-child{white-space:nowrap} .g4 table.cmds td{padding:3.5pt 6pt} "
         ".g4 .try code{overflow-wrap:anywhere} .g4 .refs{columns:2;column-gap:18pt} .g4 .refs li{break-inside:avoid} "
         ".g4 figure.draw{margin:8pt 0}</style>")


def _word(n):
    return NUMBER.get(n, str(n))


def _cap(s):
    return s[0].upper() + s[1:]


def _sg(v):
    """A signed difference with two decimals; a value that rounds to zero carries no sign (as section 8 of the report prints it)."""
    return "0.00" if round(v, 2) == 0 else signed(v, 2)


def _named(key):
    """A source named in a sentence, 'Nash and Sutcliffe (1970)'; remembered for the source list like cite()."""
    name, year = cite(key)[1:-1].rsplit(", ", 1)
    return f"{name} ({year})"


def _missing(what, e, consequence):
    note(f"g4_concepts: {what} is missing or not in the expected form ({type(e).__name__}: {e}): {consequence}")


# ---- step 14: how sure are these numbers? -----------------------------------------------------------------------------------
def _seed_doubt(R):
    """The first doubt, another training: the single trainings of the LSTM against the best simple model, with the river gauge, 1 day
    ahead (results/lstm_all.json: comparison.wq). '' with one line on stderr if the keys are not there."""
    try:
        A = R["lstm_all"]
        C = A["comparison"]["wq"]
        c = C["baselines"][C["best_baseline_by_validation"][lk(FIRST)]][lk(FIRST)]
        lo, hi, simple = c["lstm_seed_median_min"], c["lstm_seed_median_max"], c["baseline_nse_median"]
        n_seeds = len(A["sets"]["wq"]["seeds"])
        inside = c["baseline_against_seed_spread"] == "inside the seed spread"
    except (KeyError, TypeError) as e:
        _missing("results/lstm_all.json (comparison.wq)", e, "step 14 has no paragraph on the first doubt, another training")
        return ""
    verdict = (f"The best simple model reaches {n2(simple)}: it lies between them. So one training of the LSTM could land on either side of the "
               f"simple model. This is one reason why step 9 found no clear difference." if inside else
               f"The best simple model reaches {n2(simple)}.")
    return (f"<p><b>First doubt: another training.</b> Step 7 showed that two trainings with different seeds give slightly different models. "
            f"With the river gauge, the {_word(n_seeds)} single trainings of the LSTM reach an NSE between {n2(lo)} and {n2(hi)}, {LEAD_WORD[FIRST]}. "
            f"{verdict}</p>")


def _uncertainty(R):
    """results/uncertainty.json (13_uncertainty.py) when it can be quoted, else None with one line on stderr."""
    U = R.get("uncertainty")
    if isinstance(U, dict) and not U.get("quick_mode") and U.get("stats") and U.get("resampling"):
        return U
    note("g4_concepts: results/uncertainty.json is missing or a smoke test (code/13_uncertainty.py writes it): step 14 shows the concept box "
         "'Resampling' without the numbers of this project, and has no drawing and no ranges")
    return None


def _ranges(U):
    """What step 14 quotes from results/uncertainty.json, all for the first lead time and for the design 'both' (years and basins drawn
    again at once, the one the file says to quote): dict(B, n_years, n_basins, share, n_on, twice, rows, tie). rows: one (label lines,
    value, lo, hi) per set of SETS; tie: (value, lo, hi) of TIE; share: how many of 100 answers a range holds; twice: the units are
    drawn with replacement. None with one line on stderr if the file is not in the expected form."""
    try:
        RS, S = U["resampling"], U["stats"]
        m = re.search(r"([\d.]+)th to ([\d.]+)th percentile", RS["interval"])
        both = lambda name: (S[name]["value"], S[name]["both"]["lo"], S[name]["both"]["hi"])
        sets = [(s, label) for s, label in SETS if f"onsets_warned.{s}.{lk(FIRST)}" in S]            # a rung whose range the file does not hold is left out of the drawing
        forecast = {k: both(f"onsets_warned.lstm_wqaF_sub.{lk(k)}") for k in LEADS[1:] if f"onsets_warned.lstm_wqaF_sub.{lk(k)}" in S}     # the real forecast, 2 and 3 days ahead
        return dict(B=int(RS["B"]), n_years=int(RS["n_years"]), n_basins=int(RS["n_basins"]), share=float(m.group(2)) - float(m.group(1)),
                    n_on=int(U["n_onsets_test"][lk(FIRST)]), twice="with replacement" in RS["units"],
                    rows=[(label, *both(f"onsets_warned.{s}.{lk(FIRST)}")) for s, label in sets], tie=both(f"{TIE}.{lk(FIRST)}"), forecast=forecast)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        _missing("results/uncertainty.json", e, "step 14 has no drawing and no ranges")
        return None


def _stays(U, verdict="NOT YET"):
    """True when the onset warnings of the LSTM with the river gauge get `verdict` in every resample at every lead time
    (verdict_stability); False otherwise, and None with one line on stderr if the table is not there."""
    try:
        VS = U["verdict_stability"]
        return all(VS[f"lstm_wq.onset.{lk(k)}"]["verdict"] == verdict
                   and VS[f"lstm_wq.onset.{lk(k)}"]["share_of_resamples_with_each_verdict"].get(verdict, 0.0) == 1.0 for k in LEADS)
    except (KeyError, TypeError, AttributeError) as e:
        _missing("results/uncertainty.json (verdict_stability)", e, "step 14 does not say how often the verdict for onsets holds")
        return None


def _dot_note(R, v):
    """One or two sentences for the caption of the drawing: why the dot of its lower panel (v: the median of the river-by-river
    differences, results/uncertainty.json) is not the difference of the two medians that step 9 gives (results/lstm_all.json:
    comparison.wq). The two medians and the count of rivers are printed while the two have different signs, because then the dot looks
    like an error; '' with one line on stderr if a key is missing."""
    try:
        C = R["lstm_all"]["comparison"]["wq"]
        c = C["baselines"][C["best_baseline_by_validation"][lk(FIRST)]][lk(FIRST)]
        lstm, simple, higher, n = c["lstm_nse_median"], c["baseline_nse_median"], c["n_basins_lstm_higher"], c["n_basins"]
    except (KeyError, TypeError) as e:
        _missing("results/lstm_all.json (comparison.wq)", e, "the caption of the drawing of step 14 does not say why its dot is not the difference of step 9")
        return ""
    if round(v, 2) < 0 < round(lstm - simple, 2) and higher < n / 2:
        return (f" The dot is the middle of the {n} river-by-river differences, not the difference of the two medians of step 9 ({n2(lstm)} "
                f"and {n2(simple)}). The LSTM is ahead in only {higher} of the {n} rivers, so the middle lies "
                f"{'just ' if abs(v) < 0.02 else ''}below zero.")
    return " The dot is the difference in the typical river, not the difference of the two medians of step 9."


def _drawing(X, go):
    """The two main results with their ranges, as one drawing of two small panels, each with its own scale: the share of onsets
    warned for the three versions of the LSTM (0 to 100%, with what GO asks for), and the LSTM's NSE minus that of the best simple model
    (around zero). A dot is the value in the test as it was, a bar the range when years and basins are drawn again; every value is
    also printed beside its row. X: the dict of _ranges(); go: the share of onsets warned that GO asks for (flood_analysis.json).
    One hue for every row (the rows are one series); the upper bound is hatched and its dot hollow, as in the key figure."""
    W, x0, x1, xv = 700, 238, 548, 560
    bar, s = SEQ[2], []
    xa = lambda v: x0 + (x1 - x0) * min(max(v, 0.0), 1.0)
    mark = lambda x, y, lo, hi, tip, hollow=False: (
        f'<g><title>{tip}</title><rect x="{lo:.1f}" y="{y - 4}" width="{max(hi - lo, 1):.1f}" height="8" rx="4" fill="{bar}"/>'
        + (f'<circle cx="{x:.1f}" cy="{y}" r="5.2" fill="{SEQ[0]}" stroke="{BLUE}" stroke-width="2.4"/>' if hollow else
           f'<circle cx="{x:.1f}" cy="{y}" r="5.6" fill="{BLUE}" stroke="{SURFACE}" stroke-width="2"/>') + "</g>")
    # ---- top panel: the share of onsets warned
    top, ys = 24, [41 + 30 * i for i in range(len(X["rows"]))]
    bottom, strip = ys[-1] + 17, ys[-1] - 14
    s.append(_txt(0, 13, f"Share of the {count(X['n_on'])} onsets warned {LEAD_WORD[FIRST]}, by what the LSTM is given", anchor="start", size=12, weight=600, fill=INK))
    s.append(f'<rect x="{xa(go):.1f}" y="{top}" width="{x1 - xa(go):.1f}" height="{bottom - top}" fill="{TAG_TINT["GO"]}"/>')
    hatch = "".join(f'<line x1="{c}" y1="{strip + 28}" x2="{c + 28}" y2="{strip}" stroke="{AXIS}" stroke-width="1"/>' for c in range(x0 - 28, x1, 9))
    s.append(f'<clipPath id="g4strip"><rect x="{x0}" y="{strip}" width="{x1 - x0}" height="28"/></clipPath><g clip-path="url(#g4strip)">{hatch}</g>')
    for i in range(6):
        v = i / 5
        s.append(f'<line x1="{xa(v):.1f}" y1="{top}" x2="{xa(v):.1f}" y2="{bottom}" stroke="{AXIS if i == 0 else GRID}" stroke-width="1"/>')
        s.append(_txt(xa(v), bottom + 14, pct(v), size=10.5, fill=MUTED))
    mid = (xa(go) + x1) / 2
    s += [f'<rect x="{mid - 38:.1f}" y="{ys[0] - 7}" width="76" height="29" fill="{TAG_TINT["GO"]}"/>',                 # the label stands clear of the grid line
          _txt(mid, ys[0] + 4, "GO asks for", size=10.5), _txt(mid, ys[0] + 17, f"at least {pct(go)}", size=10.5)]
    for i, (y, (label, v, lo, hi)) in enumerate(zip(ys, X["rows"])):
        last = i == len(ys) - 1
        s.append(_txt(0, y - 2.5 if label[1] else y + 4, label[0], anchor="start", size=11.5, fill=INK))
        if label[1]:
            s.append(_txt(0, y + 10.5, label[1], anchor="start", size=11.5, fill=INK2 if last else INK))
        s.append(mark(xa(v), y, xa(lo), xa(hi), f"{pct(v)} ({pct(lo)} to {pct(hi)})", hollow=last))
        s.append(_txt(xv, y + 4, pct(v), anchor="start", size=11.5, weight=600, fill=INK))
        s.append(_txt(xv + 33, y + 4, f"({pct(lo)} to {pct(hi)})", anchor="start", size=11.5))
    # ---- bottom panel: the LSTM against the best simple model
    t = bottom + 44
    v, lo, hi = X["tie"]
    lim = max(0.10, round(max(abs(lo), abs(hi), abs(v)) * 20 + 0.5) / 20)                     # the scale: 0.10 each way, wider only if a value asks for it
    xc, half = (x0 + x1) / 2, (x1 - x0) / 2
    xb = lambda d: xc + half * d / lim
    y = t + 35
    s.append(_txt(0, t, f"NSE of the LSTM minus NSE of the best simple model, {LEAD_WORD[FIRST]}, both with the river gauge", anchor="start", size=12, weight=600, fill=INK))
    for d in (-lim, -lim / 2, lim / 2, lim):
        s.append(f'<line x1="{xb(d):.1f}" y1="{y - 13}" x2="{xb(d):.1f}" y2="{y + 13}" stroke="{GRID}" stroke-width="1"/>')
        s.append(_txt(xb(d), y + 27, signed(d, 2), size=10.5, fill=MUTED))
    s.append(f'<line x1="{xc:.1f}" y1="{y - 16}" x2="{xc:.1f}" y2="{y + 16}" stroke="{INK2}" stroke-width="1.3"/>')
    s += [_txt(xc, y + 27, "0", size=10.5, fill=INK2), _txt(xc - 9, y - 19, "&larr; simple model ahead", anchor="end", size=10.5, fill=MUTED),
          _txt(xc + 9, y - 19, "LSTM ahead &rarr;", anchor="start", size=10.5, fill=MUTED),
          _txt(0, y - 2.5, "the difference in the", anchor="start", size=11.5, fill=INK), _txt(0, y + 10.5, "typical basin", anchor="start", size=11.5, fill=INK),
          mark(xb(v), y, xb(lo), xb(hi), f"{_sg(v)} ({_sg(lo)} to {_sg(hi)})"),
          _txt(xv, y + 4, _sg(v), anchor="start", size=11.5, weight=600, fill=INK),
          _txt(xv + 39, y + 4, f"({_sg(lo)} to {_sg(hi)})", anchor="start", size=11.5)]
    return _svg(W, y + 33, "".join(s))


def _step14(R, f):
    U = _uncertainty(R)
    X = _ranges(U) if U else None
    stays = _stays(U) if X else None
    A = R["lstm_all"]
    n = len(A["basins"])
    try:
        go = criteria(R)["go_pod"]
    except (KeyError, TypeError, AttributeError) as e:
        _missing("results/flood_analysis.json (verdict.criteria)", e, "step 14 has no drawing")
        go = None
    n_years = X["n_years"] if X else None
    years = f"the {n_years} test years" if n_years else "the test years"

    # the second doubt, and the concept box with what was drawn again in this project (the word 'resampling' only after its box)
    second = (f"<p><b>Second doubt: other years, other rivers.</b> The {n} rivers and {years} are only a sample. Other years or other rivers "
              f"would give other numbers. There is a way to see by how much, without any new data.</p>")
    here = ""
    if X:
        here = (f"The {X['n_years']} test years and the {X['n_basins']} basins were drawn again {count(X['B'])} times. "
                + ("In a draw, a year or a basin can come up twice, or not at all. " if X["twice"] else "")
                + f"Nothing was trained again: every draw scores the stored forecasts anew, and each range holds {X['share']:.0f} of every 100 answers.")
    box = G.concept("resampling", here=here)

    # the picture of what the draws give, and the paragraph that names and reads it
    picture, reading = "", ""
    if X:
        _, v, lo, hi = X["rows"][0]
        tv, tlo, thi = X["tie"]
        tie_holds = bool(tlo < 0 < thi and max(abs(tlo), abs(thi)) < 0.05 and f["gauge_tie"])
        if SHOW_RANGES and go is not None:
            rest = X["rows"][:-1]                                                    # every version but the last, the upper bound
            below, well = max(r[3] for r in rest) < go, max(r[3] for r in rest) <= go - 0.15
            cap = (f"<b>{RANGES_PIC}.</b> <b>How to read it:</b> A dot is the result of the test as it was; "
                   f"its bar is the range of that result over the {count(X['B'])} draws."
                   + (f" Top: without the perfect weather forecast, {'both bars' if len(rest) == 2 else 'the bars'} stay {'well ' if well else ''}below "
                      f"the {pct(go)} that GO asks for." if below else " Top: the green band starts at the share of onsets warned that GO asks for.")
                   + (" Bottom: the bar lies on both sides of zero, so neither model is clearly ahead." if tlo < 0 < thi else
                      " Bottom: zero means that the two models score the same.")
                   + _dot_note(R, tv))
            picture = f'<div class="ff"><figure class="draw">{_drawing(X, go)}<figcaption>{cap}</figcaption></figure></div>'
        ranges = "The ranges in the picture" if picture else "The ranges"             # the picture: the drawing of this step, RANGES_PIC
        reading = (f"<p>{ranges} are wide, because {_word(X['n_basins'])} rivers and {_word(X['n_years'])} years are not many. With the river gauge "
                   f"the LSTM warned of {pct(v)} of the onsets; in the draws that share lies between {pct(lo)} and {pct(hi)}. "
                   + ("Still, the conclusions hold. " if stays and tie_holds else "")
                   + (f"The verdict for onsets is {tag('NOT YET')} in all {count(X['B'])} draws, at every lead time. "
                      if stays else "")
                   + (f"{'And between' if stays else 'Between'} the LSTM and the best simple model the difference stays within a few hundredths of NSE, in "
                      f"either direction: no clear difference." if tie_holds else "") + "</p>")
        fc = X.get("forecast") or {}                                                       # the real forecast of step 16, 2 and 3 days ahead
        aorc_row = next((r for r in X["rows"] if r[0][0].startswith("+ rain up to midnight")), None)
        if aorc_row and len(fc) == len(LEADS[1:]) and step16_available(R):
            below_go = go is not None and max(v[2] for v in fc.values()) < go and aorc_row[3] < go          # the three new ranges; the upper bound's own range, in the same picture, is not one of them
            reading += (f"<p>The third row of the top panel comes from a later experiment, which step 16 explains: the rain of the whole issue day, up to midnight, from an analysis "
                        f"made after the fact. Its range, {pct(aorc_row[2])} to {pct(aorc_row[3])}, is as wide as the others. So are the ranges of a real weather forecast "
                        + " and ".join(f"{LEAD_WORD[k]} ({pct(v[0])}, from {pct(v[1])} to {pct(v[2])})" for k, v in fc.items())
                        + (f". None of these {_word(len(fc) + 1)} ranges reaches the {pct(go)} that GO asks for.</p>" if below_go else ".</p>"))
        if stays is False:
            note("g4_concepts: the onset verdict NOT YET no longer holds in every resample (results/uncertainty.json: verdict_stability): the "
                 "sentence is left out of step 14")
        if not tie_holds:
            note("g4_concepts: the range of the LSTM's NSE minus that of the best simple model no longer lies within 0.05 on both sides of zero "
                 "(results/uncertainty.json), or section 4.2's reading is no longer a tie: the sentence is left out of step 14")

    # what no range covers: each limit one short sentence
    checks = team.checks_state()
    limits = [f"All {n} basins lie in one region, the south-eastern United States, so the results say nothing about rivers elsewhere.",
              (f"The test covers only {_word(n_years)} years." if n_years else "The test covers only a few years."),
              ("The perfect weather forecast is an upper bound that cannot be had in practice. The one real forecast that was tested is a single ensemble member, and the rain "
               f"up to midnight is {AORC_WORDING} (step 16)." if step16_available(R) else
               "The perfect weather forecast is an upper bound that cannot be had in practice, and no real forecast was tested.")]
    if checks["n_done"] == 0:
        limits.append("And the team members&rsquo; own checks of the code are still to be done.")
    elif checks["n_done"] < checks["n"]:
        limits.append(f"And {_esc(team.checks_sentence())}.")
    first = ("The ranges leave out the first doubt, because every draw uses the same trainings. " if X else "")
    limits_p = f"<p><b>What no range covers.</b> {first}{' '.join(limits)}</p>"

    lead = "A number from one experiment is never exact. This step asks how far the project&rsquo;s numbers could move, and which conclusions stay."
    keep = "One number from one test is not yet a result. Ask what another training would give, and what other years and rivers would give."
    if stays:
        keep += " Here the scores move, but the verdict NOT YET for onsets stays."
    opening = (f"<p>Every score in this guide comes from a few trainings, tested on {n} rivers in a few years. Two things could have "
               f"gone differently, and each can be checked.</p>")
    return f"""
{G.keep(G.step(14, lead), opening)}
{_seed_doubt(R)}
{second}
{box}
{picture}
{reading}
{limits_p}
{G.keep(G.remember(keep), G.full_detail("Sections 6.3 and 8.2"))}"""


# ---- step 15: what we found, and what it teaches -----------------------------------------------------------------------------
def _key_figure(R, f):
    """The key figure with a caption that reads it in three looks; the numbers of the caption are those printed inside the figure
    (results/key_figure.json: finding, levels, criteria). '' with one line on stderr while the picture or its results file is missing."""
    K = f["key"]
    if not K or not (FIG / KEY_FIG).exists():
        note(f"g4_concepts: figures/{KEY_FIG} or results/key_figure.json is missing (code/11_key_figure.py writes both): step 15 shows no key figure")
        return ""
    try:
        k = K["lead_drawn"]
        level = {lv["key"]: lv for lv in K["levels"]}
        place = str(level["wq"]["label"])
        kinds = 1 + sum(bool(o.get("drawn")) for o in level["wq"]["others"].values())
        apart = K["finding"]["spread_between_model_kinds"]
        up, go = K["levels"][-1]["lstm"]["onset_warned"], K["criteria"]["go_min_warned"]
        bound = "upper bound" in str(K["levels"][-1].get("label", ""))
        close = (f"First, the small marks at &lsquo;{place}&rsquo;: {_word(kinds)} kinds of model with the same inputs lie within "
                 f"{apart['onset_warned_points']:.0f} percentage points and {n2(apart['nse'])} NSE." if K["finding"]["holds"] else
                 f"First, the small marks at &lsquo;{place}&rsquo;: they are the other kinds of model, given the same inputs.")
        band = ("Third, the green band in the left panel: even the dot in the hatched area, an upper bound that cannot be had in practice, stays "
                "below what GO asks for." if bound and up < go else
                f"Third, the green band in the left panel: it starts at the {pct(go)} of onsets warned that GO asks for.")
        below = [int(x) for x in K.get("leads_drawn_below") or []]                      # the lower block of the picture: 2 and 3 days ahead
        where = f"{LEAD_WORD[k]} above, {_leads(below)} below" if below else LEAD_WORD[k]
        news = (f" Fourth, the rows added on {(new_work(R) or {}).get('days', 'the last days')} (step 16): the rain up to midnight, {AORC_WORDING}, with its real-time check, the hollow diamonds that stand for "
                f"Stage IV; and, in the lower block, a real weather forecast beside that rain." if "wqa" in level and below and step16_available(R) else "")
        cap = (f"<b>{KEY_PIC}: onsets warned (left) and the median NSE, the score of the middle basin (right), "
               f"{where}.</b> <b>How to read it:</b> {close} Second, the large blue dots, the LSTM: they climb to the "
               f"right, as the model is given more information. {band}{news}")
    except (KeyError, TypeError, IndexError, AttributeError) as e:
        _missing("results/key_figure.json", e, "step 15 shows no key figure")
        return ""
    return f'<div class="ff">{fig(KEY_FIG, cap)}</div>'


def _conclusions(R, T, f):
    """The conclusions of section 8 of the report in one sentence each (its headings, word for word), each with the steps of this guide
    that showed it. A conclusion is listed only while the stored numbers support it, by the checks of section 8."""
    A, B = R["lstm_all"], R["baselines"]
    ST = A["decomposition"]["steps"]
    cls, has_cmp = f["cls"], "model_comparison" in R and f["acc"] is not None
    no_clear = (f["gauge_tie"] and all(v in ("tie", "lstm_med", "lstm_avg") for v in cls["w"].values())
                and (not has_cmp or f["acc"] == "no clear difference in accuracy"))
    SB = {(r["model"], r["lead"]): r["nse_median"] for r in B["summary_table"]}
    fc_simple = [SB[f"base_{m}_wqf", FIRST] - SB[f"base_{m}_wq", FIRST] for m in ("linreg", "rf")
                 if (f"base_{m}_wqf", FIRST) in SB and (f"base_{m}_wq", FIRST) in SB]
    fc_gain = [ST["perfect_forecast_with_gauge"][lk(k)]["nse_median_gain"] for k in LEADS]
    lifted = bool(f["all_gain"] and min(fc_gain) > 0 and len(fc_simple) == 2 and min(fc_simple) > 0)
    tf, lf = timing_facts(R), lobo_facts(R)
    timing_clear = bool(tf and onset_clear(tf["gauge"][FIRST]))
    not_yet = f["warn"] == "Its warnings of an onset cannot be trusted yet"
    # the experiments of 7 and 8 October 2026 (step 16): they add to three conclusions and make a sixth, as section 8 of the report words them
    avail = step16_available(R)                                                              # step 16 exists: the other steps may point to it
    AO, FC, RV = (avail or {}).get("AO"), (avail or {}).get("FC"), (avail or {}).get("RV")
    more_rain = bool(AO and AO["met"])                                                       # the rain up to midnight lifts the onsets warned clearly, with a flat control
    more_rivers = bool(RV and all(v == "clear gain" for v in RV["unseen_reading"]["w_static"].values()))
    both_ways = bool(AO and FC and more_rain and FC["success"] and set(AO["verdict_onset"].values()) == {"NOT YET"})
    out = []
    for ok, name, sentence, steps in (
            (no_clear, "the kind of model", "With the inputs that can be had in practice, the kind of model made no clear difference", "steps 9 and 13"),
            (lifted, "more information", "More information lifted every kind of model", "steps 5 and 11"),
            (timing_clear, "rain that had already fallen", "Part of the missing information was rain that had already fallen", "steps 11 and 16" if more_rain else "step 11"),
            (not_yet, "the onset warnings", "The warnings of an onset cannot be trusted yet", "step 10"),
            (bool(lf and lf["clear"]), "a river never seen", "A river never seen in training is forecast clearly worse" + ("; more training rivers help" if more_rivers else ""),
             "steps 9 and 16" if more_rivers else "step 9"),
            (both_ways, "both directions", "The finding holds in both directions: taking information away lowered the skill, giving it raised it", "steps 8 to 11 and 16")):
        if ok:
            out.append(f"<li>{sentence} ({steps}).</li>")
        else:
            note(f"g4_concepts: the stored numbers no longer support the conclusion on {name} in the words of the guide (or its results file is "
                 f"missing): it is left out of step 15 - section 8 of the report words it from the outcome")
    return out


def _lessons(R, f):
    """The four lessons of section 8 about applying AI, in plain words; each example is printed only while the stored numbers support it."""
    A, F = R["lstm_all"], R["flood_analysis"]
    C = A["comparison"]["wq"]
    best = lambda k: C["baselines"][C["best_baseline_by_validation"][lk(k)]][lk(k)]
    n_seeds = len(A["sets"]["wq"]["seeds"])
    spread = lambda k: best(k)["lstm_seed_median_max"] - best(k)["lstm_seed_median_min"]
    wider = all(spread(k) > abs(best(k)["difference_of_medians"]) for k in LEADS)   # the trainings differ by more than LSTM and simple model do
    V = F["verdict"]["services"]["lstm_wq"][lk(FIRST)]
    on0 = A["decomposition"]["levels"]["wq"][lk(FIRST)]["onset_pod"]
    has_cmp = "model_comparison" in R and f["acc"] == "no clear difference in accuracy"
    tf, AO = timing_facts(R), (step16_available(R) or {}).get("AO")
    hind = hindsight_clause((step16_available(R) or {}).get("HS"))                           # 'hindsight at most about 3 points' beside the share that the AORC rain gives
    quarter = bool(tf and onset_clear(tf["gauge"][FIRST]) and 0.22 <= tf["gauge"][FIRST]["onset_share_of_gap_closed"] <= 0.28)
    gap = (f" That gap runs from the LSTM as it is ({pct(tf['gauge'][FIRST]['onset_warned_base'])}) to the upper bound "
           f"({pct(tf['gauge'][FIRST]['onset_warned_perfect_forecast'])})." if quarter else "")      # the two ends of the gap that the rain closes a part of
    return ["<b>Ask what the model is given before asking which model.</b> "
            + (f"The Transformer took {f['t_lo']:.0f} to {f['t_hi']:.0f} times as long to train, for no clear gain in accuracy." if has_cmp else
               "The same LSTM, given more and more information, showed where the room is."),
            "<b>Compare with a simple model that is given the same inputs, and train more than once.</b> "
            + f"{_cap(_word(n_seeds))} trainings of the same LSTM differ by {n2(spread(FIRST))} NSE {LEAD_WORD[FIRST]}"
            + (": more than the LSTM differs from the best simple model." if wider and f["gauge_tie"] else "."),
            f"<b>Score what the user needs.</b> {pct(V['all']['pod'])} of all high-flow days are warned {LEAD_WORD[FIRST]}, yet {pct(1 - on0)} of the "
            f"onsets get no warning.",
            "<b>Know the clock of every input.</b> Which hours a daily value covers decides what a model can know on the day the forecast is made, "
            "the issue day."
            + (f" Here, rain up to the evening of the issue day closes about a quarter of the gap in onsets warned {LEAD_WORD[FIRST]}.{gap}" if quarter else "")
            + (f" The whole day up to midnight, from {AORC_WORDING}, takes it to {pct(AO['with_'][FIRST])}{f' ({hind})' if hind else ''}, while the same rain over the table&rsquo;s "
               f"own window leaves it at {pct(AO['to09'])} (step 16)." if quarter and AO and AO["control_flat"] else "")]


def _next(R, f):
    """What came next. With the results of the experiments of 7 and 8 October 2026 (step 16) the paragraph points to that step, which ends with what is still open; without
    them it is the paragraph of the first version: the next steps of section 8 of the report, in their order, in a few short sentences."""
    if step16_available(R):
        later = (" Steps 17 to 22 follow the work after the mid-term talk, and step 23 gives the answer in its final form." if final_available(R) else "")
        return ("<p><b>What came next.</b> The first version of the project ended here, with the largest question still open: what would the rain up to midnight, and a "
                f"real weather forecast, add? Step 16 tells what the next experiments found.{later}</p>")
    checks = team.checks_state()
    tf = timing_facts(R)
    larger = bool(tf and onset_clear(tf["gauge"][FIRST]) and tf["gauge"][FIRST]["onset_share_of_gap_closed"] < 0.5)
    own = "First, the team members&rsquo; own checks of the code. Then a" if checks["n_done"] < checks["n"] else "First, a"
    return (f"<p><b>What would come next.</b> {own} real weather forecast as an input. "
            + ("The weather of the coming days is the larger part of the gap, and no real forecast was tested. " if larger else
               "No real forecast was tested. ")
            + f"Systems in daily use feed weather forecasts to an LSTM {cite('nearing2024')}. After that: rain measured up to the hour, hourly data, "
              f"more basins, and a training that aims at peaks and onsets.</p>")


def _not_a_failure(R, f):
    """Why a milestone that was not met is still a result: the same network was tested with several sets of inputs, and that design is what
    shows where the limit lies. Printed only while the stored numbers say what the paragraph says: the finding holds, there is no clear
    difference with the river gauge at any lead time, and the onset warnings cannot be trusted yet."""
    tie = all(v == "tie" for v in f["cls"]["wq"].values())
    if not (f["story"] and tie and f["warn"] == "Its warnings of an onset cannot be trusted yet"):
        note("g4_concepts: the stored numbers no longer say 'no clear difference with the river gauge' and 'the onset warnings cannot be trusted "
             "yet' together with the finding: the paragraph on why this is a result and not a failed project is left out of step 15")
        return ""
    n_sets = len(R["lstm_all"]["sets"])                                              # the sets of inputs that the same LSTM was given
    return (f"<p>With the river gauge the LSTM showed no clear difference from the best simple model, and its warnings of an onset cannot be "
            f"trusted yet. That is a result, not a failed project. The same network was tested with {_word(n_sets)} sets of inputs, so the "
            f"project can say where the limit lies among the models tried: in what a model is given. That tells the next project what to work "
            f"on first.</p>")


def _step15(R, T, f):
    figure = _key_figure(R, f)
    see = " The next picture shows it." if figure else ""                           # the key figure, whose caption starts with KEY_PIC
    if f["story"]:
        finding = f"<p>The finding, word for word: <b>{FINDING}.</b> {SUPPORT}.{see}</p>"
    else:                                                                            # the stored numbers no longer support the finding: the answer as page 1 of the report words it
        note("g4_concepts: the stored numbers no longer support the finding (s0_summary.facts()['story'] is False): step 15 prints the answer "
             "of page 1 of the report in its place")
        finding = f"<p>The answer of the project: {f['head']}{see}</p>"
    concl, lessons = _conclusions(R, T, f), _lessons(R, f)
    # one block each, so that the key figure never cuts a list in two: the conclusions fill the page in front of a figure that has moved
    # to the next page (.together does not clear it); the lessons stand below the figure, in one block with the end of the step (keep)
    concl_html = (f'<div class="together"><p class="intro"><b>{_cap(_word(len(concl)))} conclusions</b>, with the steps that showed them:</p>'
                  f'<ol class="short">{"".join(concl)}</ol></div>' if concl else "")
    lessons_html = (f'<p class="intro"><b>{_cap(_word(len(lessons)))} lessons for applying AI:</b></p><ul class="short">'
                    + "".join(f"<li>{x}</li>" for x in lessons) + "</ul>")
    lead = ("Step 1 asked whether a model can warn of a flood. Here is the answer: in one sentence, one picture, "
            f"{_word(len(concl))} conclusions and {_word(len(lessons))} lessons.")
    last = (f"{FINDING}. So ask first what a model is given, and only then which model it is." if f["story"] else
            "Ask first what a model is given, and only then which model it is.")
    return f"""
{G.keep(G.step(15, lead), finding)}
{figure}
{_not_a_failure(R, f)}
{concl_html}
{G.keep(lessons_html, _next(R, f), G.remember(last), G.full_detail("the Abstract, and Sections 8.1 and 9.1"))}"""


# ---- the last pages: the concepts on one page, run it yourself, who did what ------------------------------------------------
def _concepts():
    """Every concept of the guide with its one definition (guide_common.concepts_list(): those the guide showed first, in the order
    shown). The sentences are set as entries in two columns and kept together, so that the list holds on one page; while the table is
    not in the form this function expects, it is printed as guide_common made it."""
    listed = G.concepts_list()
    rows = re.findall(r"<tr><th>(.*?)</th><td>(.*?)</td></tr>", listed, re.S)
    if len(rows) == len(G.WORDS):
        body = '<div class="cols">' + "".join(f"<p><b>{term}.</b> {plain}</p>" for term, plain in rows) + "</div>"
    else:
        note("g4_concepts: guide_common.concepts_list() is not in the form this part expects: the concepts are printed as its table, which "
             "may not fit on one page")
        body = listed
    return f"""
<div class="keep"><h2 class="plain">The concepts on one page</h2>
<p>All {len(G.WORDS)} concepts of the guide in the order of the steps, each with the sentences of its blue box.</p>
{body}</div>
{_course_words()}"""


# the guide's plain words -> the words of an AI course. Only terms that are correct for what the project did: the comment of a row says where
# the project does it. No number stands here, so nothing is read from the results files.
_T = "<b>{}</b>".format                                                                 # a term of the course, set in bold
TERMS = (
    ("Learning from examples whose answer is known", _T("supervised learning")),                           # every model of the project
    ("Forecasting a number, such as the flow", f"{_T('regression')} (forecasting a class, such as yes or no, is {_T('classification')})"),
    ("Inputs; target", f"{_T('features')}; {_T('target')}, or {_T('label')}"),
    ("Summaries of the past weather made by hand (step 5)", _T("feature engineering")),                    # features.py
    ("Putting every input on the same scale", f"{_T('normalisation')}, or {_T('standardisation')}"),       # flood.fit: mean and spread of the training years
    ("Weights", _T("parameters")),
    ("Settings chosen before the training; trying values for them", f"{_T('hyper-parameters')}; hyper-parameter search, or {_T('tuning')}"),      # hparam_log.csv
    ("A network that carries a memory from day to day", f"{_T('recurrent neural network')} (RNN); the LSTM is one kind"),
    ("The bend in a unit of a network", _T("activation function")),
    ("The squared error that training makes small", f"{_T('mean squared error')} (MSE); here every river has its own weight"),      # flood.fit: the loss
    ("Working out the slopes of the loss for all weights", _T("backpropagation")),                         # loss.backward()
    ("Gradient descent on small batches drawn at random", f"{_T('stochastic gradient descent')} (SGD); the project uses the variant Adam"),
    ("One pass through the training data", _T("epoch")),
    ("Keeping a model from learning by heart", _T("regularisation")),                                      # dropout, early stopping, a small network
    ("Too simple against too flexible", f"underfitting against overfitting; {_T('bias')} against {_T('variance')}"),
    ("One change at a time", _T("ablation")),
    ("Leak", _T("data leakage")),
    ("Leaving each river out of the training in turn (step 9)", f"leave-one-out {_T('cross-validation')}"),      # 05_lobo.py
    ("NSE", f"{_T('R squared')}, the coefficient of determination: the same formula"),                     # flood.nse
    ("Hit; miss; false warning", f"{_T('true positive')}; {_T('false negative')}; {_T('false positive')}"),
    ("Share of high-flow days warned", f"{_T('recall')}, or probability of detection"),
    ("Share of warnings that were false", f"false-alarm ratio: 1 minus {_T('precision')}"),
    ("Share of all days called right", _T("accuracy")),
    ("The average of several trainings; a random forest; a set of forecast runs that differ a little", _T("ensemble")),    # three seeds; the weather forecast
    ("Resampling; the range it gives", f"the {_T('bootstrap')}; a {_T('confidence interval')}"),           # 13_uncertainty.py: percentile bootstrap
    ("The warning level; the operating point", f"the {_T('decision threshold')}; an {_T('operating point')}, a place on the trade-off between recall and precision"),
    ("A second run that differs in one thing (step 11)", _T("control experiment")),                        # the controls of steps 11 and 16
    ("Gradient boosting (step 17)", f"{_T('gradient-boosted decision trees')}, a kind of ensemble"),            # the Flood_Cloud_Model package
    ("Accuracy on a rare event (step 17)", f"{_T('class imbalance')}; the {_T('accuracy paradox')}"),
    ("The delivery delay of an input (step 20)", f"{_T('latency')}; what is {_T('available at prediction time')}"),
    ("A choice made after seeing the test scores (step 21)", f"{_T('test-set reuse')}; a plan fixed in advance is a {_T('pre-registration')}"),
)


def _course_words():
    """'The words your course uses': a small table under the concept list that maps the plain words of the guide to the terms of an AI
    course, so that a reader can answer a question that is asked in those terms."""
    return G.keep("<h3>The words your course uses</h3>",
                  "<p>The guide uses plain words. An AI course has its own words for the same ideas, and a question may be asked in them.</p>",
                  table(["In this guide", "In an AI course"], [list(row) for row in TERMS], cls="terms", widths=["47%", "53%"]))


def _one_liner():
    """The line of README.md that scores a stored forecast again ('cd code && python -c ...'), as it stands there; None with one line on
    stderr if README.md has no such line or the line no longer scores lstm_wq."""
    try:
        line = next(x.strip() for x in (ROOT / "README.md").read_text(encoding="utf-8").splitlines() if x.strip().startswith("cd code && python -c"))
    except (OSError, StopIteration) as e:
        _missing("the one-line check in README.md ('cd code && python -c ...')", e, "'Run it yourself' has no box 'Try it yourself'")
        return None
    if 'load_preds("lstm_wq")' not in line:
        note("g4_concepts: the one-line check in README.md no longer scores lstm_wq: 'Run it yourself' has no box 'Try it yourself'")
        return None
    return line


def _run_it(R, T):
    A = R["lstm_all"]
    C = A["comparison"]["wq"]
    c = C["baselines"][C["best_baseline_by_validation"][lk(FIRST)]][lk(FIRST)]
    pers = C["baselines"].get("base_persistence", {}).get(lk(FIRST))
    try:
        hours = sum(secs for _, _, secs, _ in _project_cost(R, T)) / 3600
    except (KeyError, TypeError, ValueError) as e:
        _missing("the computing time of the runs (results files, through s7_cost._project_cost)", e, "'Run it yourself' gives no time for the full run")
        hours = None
    FT = R.get("fair_test") or {}
    fair = (f"The fairness check of step 3: made-up rivers whose right answer is known, and {len(FT['checks'])} checks in all. It trains a few small "
            f"networks and ends with <code>all checks passed</code>." if FT.get("checks") and FT.get("passed") else
            "The fairness check of step 3: made-up rivers whose right answer is known.")
    both = "<code>results/</code>, <code>figures/</code>"
    rows = [["<code>python code/build_guide.py</code>", "Builds this guide again in seconds. Every number is read anew from the results files.", both],
            ["<code>python code/final_report/numbers.py</code>, then <code>python code/build_report_final.py</code>",
             "Rebuild the registry of every number of the final report, then the final report itself (the PDF needs LibreOffice).", both],
            ["<code>python team_checks/team_check.py --member &lt;name&gt;</code>",
             "A member&rsquo;s own check: it recomputes that member&rsquo;s headline numbers from the stored forecasts and prints AGREE or DIFFER beside each. "
             "It writes nothing; recording the check in <code>team.json</code> is the member&rsquo;s own act.", "<code>results/</code>; numpy and pandas"],
            ["<code>python code/test_flood.py</code>", fair, "nothing: it makes its own data"],
            ["<code>FLOOD_QUICK=1 ./run_all.sh</code>", "The quick mode: every script runs in a few minutes, with the trainings cut very short. Its output means "
             "nothing; it only shows that every script still runs.", "<code>data/</code>"],
            ["<code>./run_all.sh</code>", "Everything again from the raw data"
             + (f": about {hours:.0f} hours of computing on the two-core project laptop" if hours else "")
             + ". It trains every network anew and writes over <code>results/</code>, so run it only in a copy of the folder.", "<code>data/</code>"]]
    commands = table(["Command", "What it does", "What it needs"], rows, cls="cmds", widths=["29%", "50%", "21%"])
    line, box = _one_liner(), ""
    if line:
        swap = (f" Put <code>base_persistence</code> in place of <code>lstm_wq</code>, and the same line scores &lsquo;tomorrow = today&rsquo;: "
                f"{n2(pers['baseline_nse_median'])} and 0, because that model can never warn of an onset."
                if pers and pers.get("baseline_onset_pod") == 0 else "")
        # the command is printed as it must be typed: html.escape(quote=False) keeps its straight quotes (_esc would typeset the apostrophes,
        # and the copied line would then fail in a terminal)
        box = G.try_it(f"One line scores the stored forecasts of a model again, in a few seconds. It stands in <code>README.md</code>, ready to copy:<br>"
                       f"<code>{_html.escape(line, quote=False)}</code><br>It prints two numbers with many digits: {n2(c['lstm_nse_median'])} and {n2(c['lstm_onset_pod'])}. "
                       f"They are the NSE and the share of onsets warned, {LEAD_WORD[FIRST]}, of the LSTM with the river gauge.{swap} The line needs the "
                       f"folders <code>results/preds</code> and <code>data/derived</code>.")
    return f"""
<div class="keep"><h2 class="plain">Run it yourself</h2>
<p>Nothing in this guide has to be taken on trust: the project folder holds the code, the results and the pictures. Open a terminal in that
folder. On the Windows PC where the project ended, use the project&rsquo;s own Python, in the folder <code>.venv-flood</code> beside the project folder, and
write <code>python -X utf8</code> in place of <code>python</code>. On another computer, install the packages once: <code>pip install -r requirements.txt</code>.
A PDF is written only where Chrome or Edge is installed.</p>
{commands}</div>
<p>The folder <code>data/</code> holds the public CAMELS-US download and what the scripts fetched. It is on the project&rsquo;s machines and is not part of
the hand-in.</p>
{box}
{_cannot(R)}"""


def _cannot(R):
    """'What the folder cannot do': it keeps the forecasts of every model (results/preds), so every score can be worked out again in seconds. The networks of the experiments of
    7 and 8 October 2026 are kept too (results/models: one file per training, which gives back its stored forecasts); the networks of the first version are not. No script yet
    uses a kept network to forecast a day outside the record, so the last sentence holds while code/forecast.py does not exist."""
    kept = sorted((ROOT / "results" / "models").glob("*.pt"))
    runs = ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input") + (("within_day_inputs", "flow_state") if final_available(R) else ())
    trained = [R[k].get("n_trainings") for k in runs if isinstance(R.get(k), dict)]
    n_kept = f" ({count(len(kept))} files)" if kept and len(trained) == len(runs) and None not in trained and len(kept) == sum(trained) else ""
    has_script = (ROOT / "code" / "forecast.py").exists()
    which = (f"of {(new_work(R) or {}).get('days', 'the last days')} (step 16) and of steps 19 and 20, with the trainings that step 21 added," if len(runs) > 4 else
             f"of {(new_work(R) or {}).get('days', 'the last days')}")
    if kept:
        return (f"<p><b>What the folder can and cannot do.</b> It keeps the forecasts that every model made (<code>results/preds</code>), so every score can be worked out again in "
                f"seconds. For the experiments {which} it keeps the trained networks too (<code>results/models</code>{n_kept}): a network rebuilt from its file alone "
                f"gives back its stored forecasts. "
                + ("A script that loads such a network forecasts any day of the record." if has_script else
                   "No script yet uses one of them to forecast a day that is not in the record, so a new river, or tomorrow&rsquo;s flow from today&rsquo;s weather, cannot be "
                   "forecast from the folder. The networks of the first version are not kept.") + "</p>")
    return ("<p><b>What the folder cannot do.</b> It keeps the forecasts that every model made (<code>results/preds</code>), not the trained networks themselves. So every score "
            "can be worked out again in seconds. But a new river, or tomorrow&rsquo;s flow from today&rsquo;s weather, cannot be forecast without training again.</p>")


def _sources():
    """reference_list() of everything the guide cited, with the letter of a short form such as 'Kratzert et al., 2019a' carried into the
    full reference, so that two papers of one year can be told apart (formatting only, as section 9 of the report does it)."""
    out = reference_list()
    for key in CITED:
        short, full = REFERENCES[key]
        m = re.search(r"(\d{4})([a-z])$", short)
        if m:
            out = out.replace(full, full.replace(f"({m.group(1)})", f"({m.group(1)}{m.group(2)})", 1))
    return out


OLD_REVIEW = re.compile(r"Independent AI reviewer agents checked the shared code and every script(?:, the experiments of [^()]*? included)? \(results/review_\*\.json\)")


def _who_final(R):
    """'Who did what' in the final state (30 October 2026): who did what, the declaration of Appendix E of the final report word for word, the team's own checks,
    the sources. None while the final report or the files of steps 17 to 23 cannot be read (then the statement of the earlier report is printed, _who)."""
    paras = declaration() if final_available(R) else []
    if not paras:
        return None
    try:
        n_refs = report_references()
    except (ValueError, StopIteration) as e:
        _missing("the reference list of the final report", e, "'Who did what' does not count its sources")
        n_refs = None
    checks = team.checks_state()
    statement = "".join(f'<p class="who"><b>{_esc(label)}</b> {_esc(text)}</p>' for label, text in paras)
    data = f" The rain and forecast data of step 16: {REFERENCES['aorc'][0]}, {REFERENCES['stage4'][0]} and {REFERENCES['gefsv12'][0]}."
    return f"""
<h2 class="plain">Who did what</h2>
<p>Claude Code, an AI coding assistant, wrote all the code, ran all the experiments and drafted the documents, under the direction of the group leader,
{_esc(config()['group_leader'])}. Independent AI reviewer agents checked the shared engine and every experiment, and AI agents read the cited papers; no team member has read them yet.
The other members built the package of step 17. {_cap(_esc(team.checks_sentence()))}: {checks['n_done']} of {checks['n']} are recorded in
<code>team.json</code>, and <code>team_checks/TEAM_CHECK_SHEET.pdf</code> gives each member the check to run. This guide was written in the same way: by Claude
Code, from the results files and the final report.</p>
<p class="intro">The AI-assistance declaration of the final report (Appendix E), word for word:</p>
{statement}
<div class="keep"><p class="intro"><b>Sources.</b> Where to read more. The data set CAMELS-US: {_named('newman2015')} and {_named('addor2017')}. The NSE:
{_named('nash1970')}. The LSTM: {_named('hochreiter1997')}; its use for river flow: {_named('kratzert2018')}. The Transformer:
{_named('vaswani2017')}.{data} Below, every source that this guide cites, in full. Section 11 of the final report lists all
{f"{n_refs} " if n_refs else ""}sources of the project.</p>
{_sources()}</div>"""


def _who(R):
    final = _who_final(R)
    if final:
        return final
    step16_data = (f" The rain and forecast data of step 16: {REFERENCES['aorc'][0]}, {REFERENCES['stage4'][0]} and {REFERENCES['gefsv12'][0]}."
                   if step16_available(R) else "")
    try:
        statement = (who(R).replace("the text of this report", "the text of the report").replace("of section 4.5 was", "of section 4.5 of the report was")
                     .replace("(section 9)", "(section 9 of the report)"))
    except (KeyError, TypeError, ValueError) as e:
        _missing("results/engine_review.json", e, "'Who did what' prints no statement of the report")
        statement = ""
    # the sentence on the reviews is the wording of the project (REVIEW_WORDING): where the statement of the report still has the wider, older words, they are put in its place
    # here, and the lead-in then says so (it says 'word for word' only for a statement that is printed as the report words it)
    reworded = bool(statement and step16_available(R) and OLD_REVIEW.search(statement))
    if reworded:
        statement = OLD_REVIEW.sub(lambda m: review_sentence(), statement, count=1)
    lead_in = (f"The statement of {G.OLD_REPORT}, with the sentence on the reviews in the words that the project now uses for it:" if reworded else
               f"The statement of {G.OLD_REPORT}, word for word:")
    # who checked what: without step 16, the sentence of the first fifteen steps; with it, the statement below says it in the words of the project (REVIEW_WORDING); where that statement
    # is missing, the words are given here
    if not step16_available(R):
        checked = "Independent AI reviewer agents checked the code."
    elif statement and REVIEW_WORDING[1:] in statement:
        checked = "Independent AI reviewer agents checked the code and the experiments of step 16; the statement below says what they checked and what only the final critic agent verified."
    else:
        checked = f"{review_sentence()}."
    full = f'<p class="intro">{lead_in}</p><p class="who">{statement}</p>' if statement else ""
    return f"""
<h2 class="plain">Who did what</h2>
<p>Claude Code, an AI coding assistant, wrote all the code and ran all the experiments, under the team lead&rsquo;s direction.
{checked} {_cap(_esc(team.checks_sentence()))}. Section 9 of the report lists the check that the plan gives to each member, and
<code>team.json</code> is where a finished check is recorded. This guide was written in the same way: by Claude Code, from the results files and
the report.</p>
{full}
<div class="keep"><p class="intro"><b>Sources.</b> Where to read more. The data set CAMELS-US: {_named('newman2015')} and {_named('addor2017')}. The NSE:
{_named('nash1970')}. The LSTM: {_named('hochreiter1997')}; its use for river flow: {_named('kratzert2018')}. The Transformer:
{_named('vaswani2017')}.{step16_data} Below, every source that this guide cites, in full. Section 9 of the report lists all sources of the project.</p>
{_sources()}</div>"""


def html(R, T):
    f = facts(R, T)                                                                  # the verified sentences and readings of page 1 of the report
    return f"""{STYLE}<div class="g4">
{_step14(R, f)}
{_step15(R, T, f)}
{step16_html(R, T)}
{final_html(R, T)}
{_concepts()}
{_run_it(R, T)}
{_who(R)}
</div>"""
