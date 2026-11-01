"""Page 1 of the report - the finding first, then the question, three numbered findings with the numbers that decide them, the key figure,
the milestones of the plan and who did what.

Reads every results file by key; a step whose results file does not exist yet is shown as 'not run' (check: key in R), so the
page builds at any time and fills itself in after a rerun. Nothing is computed here beyond differences, counts and formatting;
every sentence whose wording depends on an outcome is chosen from the stored numbers. The finding-first page is built only while
the stored numbers support the finding (facts()['story']); otherwise the page is built as it was before 4 October 2026: the answer
box with its four paragraphs, the four big numbers and the table of steps.

Also here, because several sections quote them and must quote them alike:
  FINDING / SUPPORT                    - the finding and its supporting line, word for word (fixed by the project lead);
  timing_facts(R) / timing_sentence(R) - the numbers of the rain-timing test (results/rain_timing.json, section 4.5), onsets first;
  onset_clear(r) / nse_reading(r, n)   - how a gain of that test is judged (the rule of section 4.2 as far as the file stores it);
  timing_verdict(R)                    - the onset warnings with the extra rain, judged by the stored verdict criteria;
  criteria(R) / verdict_of(t, crit)    - the GO / CAUTION criteria of results/flood_analysis.json, parsed once;
  day_windows(R)                       - the day ends measured against hourly rain (results/day_windows.json), or None;
  stamp_source(R)                      - how the rain table's day stamp is known (results/sources_checked.json): the same words everywhere;
  dates(R) / who(R)                    - when the code was written and by whom (results/engine_review.json), the state of the team's own
                                         checks (team.json, through code/team.py) and what was added on the last day;
  lobo_facts(R) / unseen_sentence(R)   - the numbers of the unseen-river test (results/lstm_lobo.json, section 4.4): the paired loss first;
  key_figure(R) / key_caption(R, K)    - the key figure of the main finding (results/key_figure.json with figures/key_figure_plain.png for
                                         page 1, where the caption carries the words, and figures/key_figure.png elsewhere), or None;
  facts(R, T)                          - everything page 1 says, as a dict; html() and md() only arrange it;
  page1_plan(R, T)                     - which optional parts page 1 holds: found by test-printing the page until it is one page;
  LAYOUT / floated()                   - the one layout rule all sections share: large figures float to the top of the next page.
md(R, T) is the same content as plain Markdown for short_report.md, and readme_block(R, T) the block 'What we found' of README.md
(both written by build_report.py).
"""
import datetime
import html as _html
import re
import sys
import tempfile
from pathlib import Path

import team
from common import FIG, ROOT
from flood import HIGH_Q, LEADS, SPLIT
from report_common import LEAD_WORD, cite, count, fig, kpis, n2, page, pct, pdf_pages, print_pdf, signed, table, tag

LOBO_REF, TIMING_REF, SAT_REF = "section 4.4", "section 4.5", "section 6.2"      # the unseen-river test, the rain time stamps, the satellite inputs
QUOTED = "&lsquo;tomorrow = today&rsquo;"
# the finding and its supporting line: fixed sentences, used word for word wherever the documents state the main result (no full stop here)
FINDING = "Among the models tried, the limit is the information in the inputs, not the network"
SUPPORT = "With the inputs that can be had in practice, changing the kind of model changed little; giving a model more information changed a lot"
# the key figure of the main finding (11_key_figure.py, with results/key_figure.json): the rendering with its own title, subtitle and footnote
# (short_report.md, README.md), and the same drawing without them for page 1, where the tagline and the caption say the words
KEY_FIG, KEY_FIG_PLAIN = "key_figure.png", "key_figure_plain.png"
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
STYLE = ("<style>@page :first{margin-top:8mm;margin-bottom:10mm} .s0 .verdict{font-size:9.1pt;line-height:1.33;padding:4pt 10pt;margin:4pt 0} .s0 .verdict p{margin:0 0 2.5pt} .s0 .ask{margin:3pt 0;font-size:8.5pt;line-height:1.36}"
         ".s0 table.s0t th,.s0 table.s0t td{padding:1.2pt 4pt;line-height:1.25} .s0 table.s0t{font-size:8pt;margin-top:1pt} .s0 .kpi .v{font-size:13.5pt}"
         ".s0 table.s0t td:first-child{white-space:nowrap}"
         ".s0 .kpi{padding:2.5pt 8pt} .s0 .kpi .l{font-size:7.9pt;line-height:1.28} .s0 .kpis{margin:4pt 0} .s0 .small{font-size:8.5pt;margin:3pt 0 0} .s0 .sub{margin-bottom:3pt}"
         ".s0 .who{font-size:8.2pt;color:var(--ink2);margin:3pt 0 0}"
         ".s0 .tagline{font-size:10.8pt;line-height:1.28;font-weight:600;color:var(--blue);margin:1pt 0 2pt;text-wrap:balance}"
         ".s0.story .sub{font-size:8.6pt;margin-bottom:2pt} .s0.story .verdict p:last-child{margin-bottom:0}"
         ".s0 figure.key{margin:4pt 0 3pt} .s0 figure.key figcaption{font-size:8pt;line-height:1.32}"
         ".s0 .guide{font-size:8pt;line-height:1.32;color:var(--ink2);margin:3pt 0 0}</style>")       # layout only: page 1 must hold on one page
# What page 1 holds beside its fixed parts (title, finding, question, the three findings, the key figure with its caption, 'Who did what'),
# from the fullest page to the barest. page1_plan() takes the first of these that prints as ONE page. The order is the project lead's cut
# order - the list of step names, the reading guide (it then stands in section 8, in front of Table 8.1), the last sentence of finding 3
# (the unseen river) - and, only if the page still runs over, the paragraph 'What it would take' (section 8 has it in full as 'Next steps')
# and then the line of milestones (it then stands in section 8, and finding 1 says the second half of milestone 2 itself). A shorter part
# is tried again when a longer one has left. The figure is never made smaller and 'Who did what' never leaves.
PAGE1_PARTS = ("steps", "guide", "unseen", "take", "miles")
PAGE1_PLANS = tuple(dict(zip(PAGE1_PARTS, flags)) for flags in (
    (True, True, True, True, True), (False, True, True, True, True), (False, False, True, True, True), (False, False, False, True, True),
    (False, False, True, False, True), (False, False, False, False, True), (False, False, True, False, False), (False, False, False, False, False)))
PAGE1_HINT = 0                                  # the plan that fits with the key figure of 4 October 2026: where the test prints start, and the page without a browser
PRINT_CSS = ""                                  # the page rules of the report; build_report.py sets them, so that the test print uses the same page
lk = lambda k: f"l{k}"

# ---- layout shared by all sections (layout only, no colours) ----------------------------------------------------------
# A large figure is a full-width float: if it does not fit on the rest of a page it moves to the top of the next page and the text
# that follows it fills the page instead of leaving it empty. Every box, table and unbreakable block clears the floats (clear:both), so it
# always starts below the figure before it. This is needed, not cosmetic: without it the browser can place such a block beside a float that
# was moved to the next page, in a slot of zero width, and then shrinks the whole document to fit (seen on 3 October 2026 with the table
# block of section 6.1: the report came out at 60% scale). A new section (h2) also starts after every figure of the section before it.
# The text is set a little tighter than the shared default (9.4 pt on 1.34 instead of 9.6 pt on 1.42, 4 pt between paragraphs instead of 5)
# so that the figures can keep their size within the page limit of the report. (1.34 since 8 October 2026: 1.38 left gaps of 0.1 to 0.15 page on pages 5, 8 and 10 once the
# sentences of that day were added, and 1.355 to 1.365 on pages 5, 8 and 10 too; 1.33 to 1.345 packs the pages and keeps the report at 30.)
# .nb marks a paragraph that must not be split over two pages (used where a figure at the top of the next page would otherwise cut a
# sentence in two); .keep keeps a heading, its text and its table together and starts below any figure before it.
LAYOUT = ("<style>body{font-size:9.4pt;line-height:1.34} p{margin-bottom:4pt} ul,ol{margin-bottom:4pt} li{margin-bottom:1.5pt} "
          ".ff{float:left;width:100%;break-inside:avoid} .note,.verdict,.ask{display:flow-root} .note,.ask{margin:5pt 0} "
          ".note,.verdict,.ask,.two,.kpis,.blk,.keep,.refs,table,.clr,h2{clear:both} h2{margin-top:11pt} h3{margin-top:8pt} .refs{font-size:7.8pt} "
          ".nb{break-inside:avoid} .keep{break-inside:avoid} .keep h3{margin-top:6pt}</style>")


def floated(figure):
    """Wrap a figure (the HTML of report_common.fig) as a full-width float; see LAYOUT."""
    return f'<div class="ff">{figure}</div>'


def fig_cropped(name, caption, width=None, top=0.0, bottom=0.0):
    """A figure of figures/ whose picture is shown without its top `top` and bottom `bottom` fractions (the figure's own title and footnote, which the caption repeats in larger
    type than the footnote prints at). The picture is cut by the browser, the file is not changed. width: % of the text width, as report_common.fig."""
    import struct
    from report_common import img
    with open(FIG / name, "rb") as fh:
        w, h = struct.unpack(">II", fh.read(26)[16:24])                           # width and height from the IHDR chunk of the PNG
    keep = 1 - top - bottom
    style = f' style="width:{width}%;margin-left:auto;margin-right:auto"' if width else ""
    inner = img(name).replace("<img ", f'<img style="position:absolute;left:0;width:100%;top:-{100 * top / keep:.3f}%" ', 1)      # a percentage top is of the box height
    return (f'<figure{style}><div style="position:relative;overflow:hidden;width:100%;padding-bottom:{100 * keep * h / w:.3f}%">{inner}</div>'
            f"<figcaption>{caption}</figcaption></figure>")


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _leads(ks):
    """[1] -> '1 day ahead'; [2, 3] -> '2 and 3 days ahead'."""
    ks = sorted(ks)
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _word(n):
    return NUMBER.get(n, str(n))


def _years(part):
    return f"{SPLIT[part][0][:4]}&ndash;{SPLIT[part][1][:4]}"


def _plain(text):
    """HTML of this report -> plain text for Markdown: tags dropped, entities turned into characters."""
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", text.replace("<br>", " ")))).strip()


def _day(iso):
    """'2026-10-03' -> '3 October 2026'."""
    d = datetime.date.fromisoformat(str(iso)[:10])
    return f"{d.day} {d:%B %Y}"


_SAID = set()


def note(message):
    """One line on stderr, once per build however often the sentence that needs the missing file or key is asked for."""
    if message not in _SAID:
        _SAID.add(message)
        print(message, file=sys.stderr)


def _esc(text):
    """Plain text (a name or a sentence of team.py or of a results file) -> HTML of this report: escaped, apostrophes typeset as elsewhere."""
    return _html.escape(str(text), quote=False).replace("'", "&rsquo;")


def share_word(v):
    """A share in plain words where it lies near a simple fraction ('about a quarter'), otherwise as a percentage."""
    for lo, hi, word in ((0.22, 0.28, "about a quarter"), (0.30, 0.36, "about a third"), (0.45, 0.55, "about half"), (0.64, 0.69, "about two thirds"),
                         (0.72, 0.78, "about three quarters")):
        if lo <= v <= hi:
            return word
    return pct(v)


# ---- the verdict criteria, parsed once from the results file ----------------------------------------------------------
def criteria(R):
    """The GO / CAUTION criteria of results/flood_analysis.json (verdict.criteria) as numbers: dict(go_pod, go_far, ca_csi)."""
    crit = R["flood_analysis"]["verdict"]["criteria"]
    return dict(go_pod=float(re.search(r"pod\s*>=\s*([\d.]+)", crit["GO"]).group(1)), go_far=float(re.search(r"far\s*<=\s*([\d.]+)", crit["GO"]).group(1)),
                ca_csi=float(re.search(r"csi\s*>=\s*([\d.]+)", crit["CAUTION"]).group(1)))


def verdict_of(t, crit):
    """GO / CAUTION / NOT YET for one warning table (keys pod, far, csi) by the stored criteria."""
    if t["pod"] >= crit["go_pod"] and t["far"] <= crit["go_far"]:
        return "GO"
    return "CAUTION" if t["csi"] >= crit["ca_csi"] else "NOT YET"


def criteria_line(R, short=False):
    """The criteria as one short line (page 1 must stand alone). short: without 'by criteria fixed before the analysis', for a place where
    the text beside it has just said so (finding 3 above the key figure)."""
    c = criteria(R)
    return (f"<i>Verdicts</i>{'' if short else ', by criteria fixed before the analysis'}: GO, at least {pct(c['go_pod'])} warned and at most "
            f"{pct(c['go_far'])} false warnings; CAUTION, not GO, but hits are at least {pct(c['ca_csi'])} of hits, misses and false warnings together; "
            f"NOT YET, neither.")


# ---- facts that several sections quote --------------------------------------------------------------------------------
def timing_facts(R):
    """The rain-timing test (10_rain_timing.py) by lead time, or None while results/rain_timing.json does not exist.
    Returns {'gauge': {lead: row}, 'nogauge': {lead: row}, 'stamps': median correlations, 'n': basins}; a row is one entry of closure.<set>.by_lead."""
    X = R.get("rain_timing")
    if not X or X.get("quick_mode") or not X.get("closure"):
        return None
    out = {name: {r["lead"]: r for r in X["closure"][key]["by_lead"]} for name, key in (("gauge", "wqn"), ("nogauge", "wn")) if key in X["closure"]}
    out["stamps"] = X.get("stamps", {}).get("median", {})
    out["n"] = len(X["stamps"]["per_basin"])
    return out if "gauge" in out else None


def _above(a, b):
    """Every single seed of a above every single seed of b ([lowest, highest] each)."""
    return bool(a) and bool(b) and a[0] > b[1]


def onset_clear(r):
    """Onset warnings in one row of the rain-timing test. The warning scores count the basins together, so no basin count exists for them:
    the gain is clear when every single seed with the extra rain is above every seed without it and every seed of the control."""
    test = r.get("onset_seed_range_with_nldas")
    return _above(test, r.get("onset_seed_range_base")) and _above(test, r.get("onset_seed_range_control"))


def nse_reading(r, n):
    """The NSE in one row of the rain-timing test, by the rule of section 4.2 as far as the file stores what it needs (seed ranges and the
    basin count; no p-value): 'clear' = every seed with the extra rain above every seed without it AND higher in most basins;
    'mixed' = the seeds say so, the basins do not; 'none' = the single seeds overlap."""
    seeds = _above(r.get("seed_range_with_nldas"), r.get("seed_range_base"))
    return "clear" if seeds and r["basins_better"] > n / 2 else "mixed" if seeds else "none"


def timing_reading(r):
    """What the control says about the NSE gain in one row: 'stamp' (the seeds with the rain of the issue day are above those of the model
    without it and of the control), 'stamp_likely', 'product' (the control gains as much: a second rain product), 'none' or 'worse'."""
    test, base, ctrl = r["seed_range_with_nldas"], r.get("seed_range_base"), r.get("seed_range_control")
    if not _above(test, base):
        return "worse" if base and test[1] < base[0] else "none"
    if _above(test, ctrl):
        return "stamp"
    return "product" if _above(ctrl, base) else "stamp_likely"


def timing_nse_short(r, n):
    """The NSE result of the rain-timing test, 1 day ahead with the river gauge, as one sentence worded from the outcome."""
    move = f"the median {'rises' if r['gain'] > 0 else 'falls'} from {n2(r['nse_base'])} to {n2(r['nse_with_nldas'])}"
    what = nse_reading(r, n)
    if what == "clear":
        return f"In NSE the gain is clear as well: {move}, and the forecast is higher in {r['basins_better']} of {n} basins."
    if what == "mixed":
        return f"In NSE the result is mixed: {move}, but the forecast is higher in only {r['basins_better']} of {n} basins."
    return f"In NSE there is no clear difference: {move}, and the single seeds overlap."


def timing_sentence(R, short=False):
    """What rain stamped up to the evening of the issue day adds, with the river gauge: the onset warnings first (the firm result), then the
    NSE (empty while the test has not run). short: the 1-day-ahead result only."""
    tf = timing_facts(R)
    if not tf:
        return ""
    g, n = tf["gauge"], tf["n"]
    first, rest = LEADS[0], LEADS[1:]
    r = g[first]
    refs = f"control, the same rain one day older: {pct(r['onset_warned_control'])}; upper bound: {pct(r['onset_warned_perfect_forecast'])}"
    if onset_clear(r):
        one = (f"Rain stamped up to the evening of the issue day lifts the onsets warned {LEAD_WORD[first]} with the river gauge from "
               f"{pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} ({refs}): {share_word(r['onset_share_of_gap_closed'])} of that gap, "
               f"in every seed.")
    else:
        one = (f"Rain stamped up to the evening of the issue day takes the onsets warned {LEAD_WORD[first]} with the river gauge from "
               f"{pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} ({refs}); the single seeds overlap: no clear gain.")
    one += " " + timing_nse_short(r, n)
    if short:
        return one
    left = min(1.0, 1 - max(g[k]["share_of_gap_closed"] for k in rest))
    return one + f" {_leads(rest).capitalize()} almost all of the gap remains ({pct(left)} or more of the gap in NSE)." if left >= 0.9 else one


def timing_verdict(R):
    """The onset warnings 1 day ahead of the LSTM with the river gauge and the rain stamped up to the evening, judged by the stored criteria at
    the high-flow level itself: dict(verdict, pod, far, hits, false, false_base), or None when a file or key is missing."""
    try:
        M = R["rain_timing"]["models"]
        t, b = (M[s]["test"]["summary"][lk(LEADS[0])]["onset"] for s in ("wqn", "wq"))
        return dict(verdict=verdict_of(t, criteria(R)), pod=t["pod"], far=t["far"], hits=t["hits"], false=t["false_alarms"], false_base=b["false_alarms"],
                    n=t["hits"] + t["misses"])
    except (KeyError, TypeError, AttributeError):
        return None


def day_windows(R):
    """The day ends measured against hourly rain (check_day_windows.py), read by key, or None while results/day_windows.json does not exist
    or is not in the expected form. Clock times are local standard time, as stored."""
    W = R.get("day_windows")
    try:
        if not W or W.get("quick_mode"):
            return None
        dm, nl, im = W["day_ends"]["daymet"], W["day_ends"]["nldas"], W["day_ends"]["imerg"]
        off = sorted(im["hours_late_in_this_check"].values())                  # how far the check misplaces a day whose end is documented
        flow = W["river_flow_day"]["summary"]
        std = flow["parts"]["standard_time_in_force"]
        return dict(daymet=dm["day_ends_local_rounded"], daymet_hour=dm["day_ends_local_hour"], daymet_readings=dm["range_over_readings_local"],
                    daymet_basins=dm["basin_range_local"], misfits=list(dm["basins_that_do_not_fit"]),
                    nldas=nl["day_ends_local_rounded"], nldas_readings=nl["range_over_readings_local"], imerg=im["documented_day_ends_local"], off=off,
                    n_readings=len(dm["median_by_reading"]), n_refs=len(W["windows"]),
                    flow_gauges=flow["n_gauges"], flow_midnight=std["n_gauges_best_midnight_standard_time"], flow_days=std["days"])
    except (KeyError, TypeError, AttributeError, IndexError) as e:
        print(f"s0_summary: results/day_windows.json is not in the expected form ({type(e).__name__}: {e}); it is not quoted", file=sys.stderr)
        return None


def stamp_source(R):
    """How the rain table's day stamp is known - the wording of results/sources_checked.json (how_to_word_it), the same in every section:
    an inference from our own data, consistent with two published sources; the documentation of the Daymet version in CAMELS was not checked.
    Without that file the sentence says only what the results files say: an inference that was not checked against the documentation."""
    S = R.get("sources_checked")
    keys = {s["key"] for s in S.get("sources", [])} if isinstance(S, dict) else set()
    if {"thornton2021", "kratzert2021"} <= keys:
        unread = " No team member has read the two papers yet." if "no team member has read them yet" in str(S.get("what", "")) else ""
        return (f"This day stamp is an inference from our own data. It is consistent with a time-of-observation effect that the Daymet authors "
                f"describe and correct from version 4 on {cite('thornton2021')}, and with the one-day shifts between the rain products of CAMELS reported "
                f"elsewhere {cite('kratzert2021')}. The documentation of the Daymet version inside CAMELS was not checked.{unread}")
    return "This day stamp is an inference from our own data; it was not checked against the Daymet documentation."


def lobo_facts(R):
    """The unseen-river test (05_lobo.py), 1 day ahead, or None while results/lstm_lobo.json does not exist.
    loss: what the typical basin loses when it is held out (the median of the basin-by-basin losses: the number to quote, because the two
    medians 'seen' and 'unseen' come from different basins); clear: the condition of section 4.4 for 'clearly worse' (the loss is larger than
    the seed spread, the score is lower in most basins and p is small)."""
    X = R.get("lstm_lobo")
    if not X or X.get("quick_mode") or "summary" not in X:
        return None
    from report_sections.s4_all import P_CUT                                    # the line for 'p is small' that section 4.2 states
    s = X["summary"][lk(LEADS[0])]
    loss = s["loss_from_not_seeing_the_basin"]["with_properties"]
    p = loss.get("wilcoxon_p")
    clear = bool(loss["median_nse_loss"] > s["in_training_seed_range_median"] and loss["n_basins_lower_when_held_out"] > s["n_basins"] / 2
                 and p is not None and p < P_CUT)
    return dict(seen=s["nse_median"]["in_training"], unseen=s["nse_median"]["held_out_with_properties"], unseen_plain=s["nse_median"]["held_out_without_properties"],
                forest=s["nse_median"]["forest_held_out"], n_lower=loss["n_basins_lower_when_held_out"], n_clear=loss["n_basins_loss_larger_than_seed_range"],
                n=s["n_basins"], seed_range=s["in_training_seed_range_median"], properties=s["basin_properties"]["held_out"]["verdict"],
                loss=loss["median_nse_loss"], p=p, clear=clear)


def unseen_sentence(R, short=False):
    """The unseen-river result as every summary states it (no capital, no full stop): the paired loss first, then the two medians.
    short: without the basin count and the medians (inside a bracket). Empty while the test has not run."""
    f = lobo_facts(R)
    if not f:
        return ""
    lead = LEAD_WORD[LEADS[0]]
    first = (f"the typical basin loses {n2(f['loss'])} NSE {lead}" if f["loss"] >= 0 else f"the typical basin gains {n2(-f['loss'])} NSE {lead}")
    if short:
        return first
    side = "lower" if f["loss"] >= 0 else "higher"
    n_side = f["n_lower"] if f["loss"] >= 0 else f["n"] - f["n_lower"]
    move = "falls" if f["unseen"] < f["seen"] else "rises"
    return f"{first} ({side} in {n_side} of {f['n']} basins); the median {move} from {n2(f['seen'])} to {n2(f['unseen'])}"


def lobo_sentence(R):
    """Kept under its old name for any builder that still asks for it: the same sentence as unseen_sentence(R)."""
    s = unseen_sentence(R)
    return f"for a river never seen in training, {s}" if s else ""


# ---- the experiments of 7 and 8 October 2026 (sections 4.6 to 4.8): three fixed wordings and the facts several places quote ---------------
# The three wordings are used word for word wherever they apply (sections 4.6 to 4.8, page 1, section 8, short_report.md, README.md):
AORC_WORDING = "an analysis made after the fact"                       # what AORC is - never 'observed' rain
MOVED_POINT = "a moved operating point, not lost skill"                # a lower onset share with fewer false warnings
INFO_NOT_NET = "the information, not the network"                      # the gains of the simple models given the same inputs
REALTIME_WORDING = "a check on the test years with the real-time radar product in place of AORC, in networks trained on AORC, not a service"  # what the Stage IV figure is
HINDSIGHT_WORDING = "hindsight at most about 3 points"              # the bound of the independent review of E1b, to stand beside every 51%
NSE_FORECAST_WORDING = "in NSE the forecast lowers the 1-day score and the gains at 2 and 3 days are mixed"   # never "mixed at all three leads"
REVIEW_WORDING = ("independent AI reviewer agents checked the shared engine and every experiment script (results/engine_review.json, results/review_*.json); the real-time check on all onsets, the rain-clock diagnostic, "
                  "the resampling ranges and the key figure of 8 October 2026 were verified by the final critic agent only (results/review_documents_2026-10-08.json)")
AS_MUCH = 0.9                                                          # a simple model 'gains as much' when its gain is at least this share of the LSTM's
READING = {"clear gain": "a clear gain", "clear loss": "a clear loss", "no clear difference": "no clear difference", "mixed": "a mixed result"}   # stored readings as phrases
MONTH = ("", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")


def reading_word(reading):
    """A stored reading of a comparison ('clear gain', 'no clear difference', 'mixed', 'clear loss') as a phrase for a sentence."""
    return READING.get(reading, str(reading))

AORC_LONG = (f"AORC, a radar- and gauge-anchored hourly rain product of NOAA ({AORC_WORDING}: its daily totals come from rain gauges and are spread over the "
             f"hours with radar, with corrections about ten days behind real time)")


def by_lead_reading(readings):
    """{lead: stored reading} -> 'no clear difference 1 day ahead and a clear gain 2 and 3 days ahead' (leads with the same reading together, in the order of the leads)."""
    groups = {}
    for k in LEADS:
        groups.setdefault(readings[k], []).append(k)
    if len(groups) == 1:
        return f"{reading_word(next(iter(groups)))} at all {_word(len(LEADS))} lead times"
    return " and ".join(f"{reading_word(r)} {_leads(ks)}" for r, ks in groups.items())


def _dig(d, *path, default=None):
    """d[path[0]][path[1]]... or default where a key or index is missing."""
    for p in path:
        try:
            d = d[p]
        except (KeyError, TypeError, IndexError):
            return default
    return d


def _file(R, name):
    """results/<name>.json when it can be quoted (it exists and is no smoke test), else None."""
    X = R.get(name)
    return X if isinstance(X, dict) and not X.get("quick_mode") else None


def _check(R, review, cid):
    """One check (by id) of an independent review record results/review_<...>.json, or None."""
    rev = _file(R, review)
    return next((c for c in (rev or {}).get("checks", []) if c.get("id") == cid), None)


def _guard(name, fn):
    """Run a reader of one results file; a file that is missing or not in the expected form gives None and one line on stderr."""
    try:
        return fn()
    except (KeyError, TypeError, IndexError, ValueError, AttributeError, ZeroDivisionError) as e:
        note(f"s0_summary: results/{name}.json is not in the expected form ({type(e).__name__}: {e}); what it holds is left out of the text")
        return None


def era5_facts(R):
    """E1 (code/15_rain_to_midnight.py, results/rain_to_midnight.json): the hourly ERA5 rain of the whole issue day as one extra input, LSTM with the river
    gauge, 1 day ahead, test years. Shares of the onsets, seed ranges [lowest, highest], and the readings the file stores. None while the file is missing."""
    X = _file(R, "rain_to_midnight")
    if not X:
        return None

    def read():
        k, first = lk(LEADS[0]), LEADS[0]
        r = {x["lead"]: x for x in X["closure"]["wqh"]["by_lead"]}[first]
        S, ok = X["e1_success"]["wqh"], lambda s: X["verdicts"]["sets"][s][k]["plain"]["onset"]
        yard = X["yardstick"]["new_input_given_to_each_model"]
        pts = lambda blk: X[blk]["by_lead"][k]["onsets"]["difference_points"]
        return dict(n=ok("wqh")["hits"] + ok("wqh")["misses"], base=r["onset_warned_base"], with_=r["onset_warned_with_rain_to_midnight"], control=r["onset_warned_control"],
                    nldas=r["onset_warned_nldas"], upper=r["onset_warned_perfect_forecast"], to09=r["controls"]["to09"]["onset_warned"], utc=r["controls"]["utc"]["onset_warned"],
                    both=X["both_products"]["against_rain_to_midnight"][k]["onsets"]["warned_with"], seeds=r["onset_seed_range_with"], seeds_base=r["onset_seed_range_base"],
                    seeds_nldas=r["onset_seed_range_nldas"], gap=r["onset_share_of_gap_closed"], hits=ok("wqh")["hits"], hits_base=ok("wq")["hits"],
                    hits_nldas=ok("wqn")["hits"], false_with=r["onset_false_with_rain_to_midnight"], false_base=r["onset_false_base"],
                    nse_base=r["nse_base"], nse_with=r["nse_with_rain_to_midnight"], nse_reading=X["gain_rule"]["wqh"]["against_rain_table_alone"][k]["nse"]["reading"],
                    basins_better=r["basins_better"], n_basins=len(X["per_basin"]["basins"]), met=bool(S["met"]), above_nldas=bool(S["every_seed_above_every_seed_with_nldas_rain"]),
                    control_flat=bool(S["control_flat"]), hours=pts("value_of_hours_09_to_24"), evening=pts("value_of_evening_hours_beyond_utc_day"),
                    second_product=pts("second_rain_product_alone"), gain_ridge=yard["ridge regression"]["by_lead"][k]["onset_gain_points"],
                    gain_forest=yard["random forest"]["by_lead"][k]["onset_gain_points"], gain_lstm=yard["lstm"]["by_lead"][k]["onset_gain_points"],
                    verdict=ok("wqh")["verdict"], n_trainings=X["n_trainings"],
                    gain_before=_dig(X, "by_period", "differences", "wqh_minus_wq", "before_2009_07_17", "onset_points"),
                    gain_after=_dig(X, "by_period", "differences", "wqh_minus_wq", "from_2009_07_17", "onset_points"), assimilation=_dig(X, "by_period", "assimilation_of_radar_and_gauge_rain_from"))
    return _guard("rain_to_midnight", read)


def aorc_facts(R):
    """E1b (code/16_aorc_rain_to_midnight.py, results/aorc_rain_to_midnight.json): AORC rain of the whole issue day as one extra input, LSTM with the river gauge,
    test years, by lead: shares of the onsets, NSE, seed ranges, the controls, the yardstick, the verdicts. None while the file is missing."""
    X = _file(R, "aorc_rain_to_midnight")
    if not X:
        return None

    def read():
        first = LEADS[0]
        row = lambda key: {x["lead"]: x for x in X["closure"][key]["by_lead"]}
        c, c09, cutc, crep = (row(key) for key in ("wqa", "wqa_to09", "wqa_utc", "wqa_rep"))
        V = X["verdicts"]["sets"]["wqa"]
        yard = X["yardstick"]["new_input_given_to_each_model"]
        pts = lambda blk, k=first: X[blk]["by_lead"][lk(k)]["onsets"]["difference_points"]
        simple = {m: yard[m]["by_lead"][lk(first)] for m in yard}
        against = X["yardstick"]["lstm_against_simple"]
        return dict(
            n=V[lk(first)]["plain"]["onset"]["hits"] + V[lk(first)]["plain"]["onset"]["misses"],
            base={k: c[k]["onset_warned_base"] for k in LEADS}, with_={k: c[k]["onset_warned_with"] for k in LEADS}, upper={k: c[k]["onset_warned_perfect_forecast"] for k in LEADS},
            false={k: c[k]["onset_false_with"] for k in LEADS}, false_base={k: c[k]["onset_false_base"] for k in LEADS},
            seeds=c[first]["onset_seed_range_with"], nse_seeds=c[first]["seed_range_nse_with"], nse_base={k: c[k]["nse_base"] for k in LEADS}, nse_with={k: c[k]["nse_with"] for k in LEADS},
            gap={k: c[k]["onset_share_of_gap_closed"] for k in LEADS}, nse_gap=c[first]["share_of_gap_closed"], basins_better=c[first]["basins_better"], n_basins=len(X["per_basin"]["basins"]),
            to09=c09[first]["onset_warned_with"], utc=cutc[first]["onset_warned_with"], rep=crep[first]["onset_warned_with"], rep_seeds=crep[first]["onset_seed_range_with"],
            hits=V[lk(first)]["plain"]["onset"]["hits"], hits_base=X["verdicts"]["sets"]["wq"][lk(first)]["plain"]["onset"]["hits"],
            hours=pts("value_of_hours_09_to_24"), evening=pts("value_of_evening_hours_beyond_utc_day"), second_product=pts("second_rain_product_alone"),
            lag_days=(int(re.search(r"corrected about (\d+) days behind real time", X["outcome_caveat"]).group(1)) if re.search(r"corrected about (\d+) days behind real time", X["outcome_caveat"]) else None),
            era5=pts("aorc_against_era5"), era5_reading=X["aorc_against_era5"]["by_lead"][lk(first)]["onsets"]["reading"], hours_reading=X["value_of_hours_09_to_24"]["by_lead"][lk(first)]["onsets"]["reading"],
            control_flat=bool(X["e1b_success"]["wqa"]["control_flat"]), n_trainings=X["n_trainings"], met=bool(X["e1b_success"]["wqa"]["met"]), grows=bool(X["e1b_success"]["wqa"]["roadmap_step_1b"]["gain_grows"]),
            verdict_onset={k: V[lk(k)]["plain"]["onset"]["verdict"] for k in LEADS}, verdict_all=V[lk(first)]["plain"]["all"]["verdict"],
            all_pod=V[lk(first)]["plain"]["all"]["pod"], all_far=V[lk(first)]["plain"]["all"]["far"],
            gain=dict(lstm=simple["lstm"]["onset_gain_points"], ridge=simple["ridge regression"]["onset_gain_points"], forest=simple["random forest"]["onset_gain_points"]),
            warned=dict(lstm=simple["lstm"]["onset_warned_with"], ridge=simple["ridge regression"]["onset_warned_with"], forest=simple["random forest"]["onset_warned_with"]),
            far=dict(lstm=simple["lstm"]["onset_false_with"], ridge=simple["ridge regression"]["onset_false_with"], forest=simple["random forest"]["onset_false_with"]),
            nse_simple=dict(ridge=against["lstm_wqa__base_linreg_wqa"][lk(first)]["baseline_nse_median"], forest=against["lstm_wqa__base_rf_wqa"][lk(first)]["baseline_nse_median"]),
            nse_gain=dict(lstm=simple["lstm"]["nse_gain"], ridge=simple["ridge regression"]["nse_gain"], forest=simple["random forest"]["nse_gain"]),
            against={m: against[f"lstm_wqa__base_{m}_wqa"][lk(first)] for m in ("linreg", "rf")})
    return _guard("aorc_rain_to_midnight", read)


def hindsight_facts(R):
    """The hindsight question about AORC and its answers: the builder's check (results/aorc_hindsight.json), the independent review's regression, its Stage IV
    substitution on 171 of the 611 onsets and its bound (results/review_aorc_rain_to_midnight.json, checks d3, d5 and d7). None where a record is missing."""
    H, d3, d5, d7 = _file(R, "aorc_hindsight"), _check(R, "review_aorc_rain_to_midnight", "d3"), _check(R, "review_aorc_rain_to_midnight", "d5"), _check(R, "review_aorc_rain_to_midnight", "d7")
    if not (H and d3 and d5 and d7):
        return None

    def read():
        reg = H["numbers"]["regression"]
        both = d3["numbers"]["coefficient_on_stage_iv_after_midnight_mm_per_mm"]["whole day t, both"]
        w = d5["numbers"]["wqa_onsets_warned"]
        return dict(days_builder=H["numbers"]["n_days"], coef_builder=reg["b2"], ci_builder=reg["b2_95_interval_bootstrap_over_days"], days_review=d3["numbers"]["sample"]["days"],
                    coef_review=both[0], ci_review=both[1], covered=d5["numbers"]["onsets_in_the_sample"], aorc=w["aorc_as_saved"], raw=w["stage_iv_raw"], matched=w["stage_iv_level_matched"],
                    table=w["rain_table_alone_lstm_wq"], best=d7["numbers"]["best_estimate_points"], bound=d7["numbers"]["loosest_bound_points"], bound_hits=d7["numbers"]["loosest_bound_hits"])

    def read_latest():
        """Since 30 October 2026 lstm_wqa averages five networks: the bound beside every AORC share is the five-seed bound of the latest round of the
        review record that holds one (round_2: 3.6 points, 'about 4'); the bound of 7 October (seeds 0-2, 'about 3') stays as the fallback."""
        f = read()
        rev = _file(R, "review_aorc_rain_to_midnight") or {}
        rounds = sorted((k for k in rev if k.startswith("round_") and k[6:].isdigit()), key=lambda k: int(k[6:]))
        for k in reversed(rounds):
            h = (rev.get(k) or {}).get("hindsight") or {}
            if h.get("five_seed_loosest_bound_points") is not None:
                f.update(bound=h["five_seed_loosest_bound_points"], bound_hits=h.get("five_seed_loosest_bound_hits", f["bound_hits"]),
                         best=h.get("five_seed_level_matched_points", f["best"]), bound_round=k, bound_networks=5)
                break
        return f
    return _guard("aorc_hindsight", read_latest)


def stage4_facts(R):
    """The real-time test (code/check_stage4_realtime.py, results/stage4_realtime.json): Stage IV, a radar-and-gauge product about one hour behind real time, in
    place of AORC on the issue day of every onset, the saved networks of E1b. None while the file is missing."""
    X = _file(R, "stage4_realtime")
    if not X:
        return None

    def read():
        M, V = X["models"]["wqa"], X["models"]["wqa"]["variants"]
        n = M["onsets_tested"]
        iv = V["stage4"]["intervals"]["over_onsets"]["share_surviving"]
        return dict(n=n, aorc=M["aorc"], table=M["rain_table_alone"], stage4=V["stage4"]["onsets_warned"], no_last=V["stage4_no_last_hour"]["onsets_warned"], no_last2=V["stage4_no_last_two_hours"]["onsets_warned"],
                    level_matched=V["stage4_k"]["onsets_warned"], share=V["stage4"]["share_of_the_aorc_effect_surviving"], interval=iv, far_stage4=M["issue_days_of_all_basins"]["stage4"]["far"],
                    far_aorc=M["issue_days_of_all_basins"]["aorc"]["far"], another_onset=M["controls_on_the_rain_of_the_issue_day"]["another_onset_stage4"]["mean"],
                    ordinary=M["controls_on_the_rain_of_the_issue_day"]["ordinary_day_aorc"]["mean"], no_rain=M["controls_on_the_rain_of_the_issue_day"]["no_rain"]["mean"],
                    ratio=X["agreement_of_the_two_products_on_the_issue_days"]["mean_ratio_stage4_over_aorc"],
                    replaced=(X["fallback"]["hours_replaced_by_aorc"], X["fallback"]["hours_total"]),
                    archive_unestablished=any("first postings" in str(x) and "not established" in str(x) for x in X.get("honesty_points") or []),
                    no_far_on_other_days=any("false warnings are tested only on the issue days" in str(x) for x in X.get("honesty_points") or []))
    return _guard("stage4_realtime", read)


def clock_facts(R):
    """Where the rain of a flood onset falls against the issue time (code/check_onset_rain_clock.py, results/onset_rain_clock.json) and how the issue day is cut
    (results/day_windows.json, river_flow_day): the shares of the mean rain of the 48 hours around the onsets, the median share known to the model, and the number of gauges
    whose flow day follows the summer clock. None while a file is missing."""
    X = _file(R, "onset_rain_clock")
    if not X:
        return None

    def read():
        m = X["all"]["mean_mm"]
        total = m["table"] + m["hidden"] + m["future1"] + m["future2"]
        miss = X["missed_by_lstm_wq"]
        out = dict(n=X["n_onsets"], table=m["table"] / total, hidden=m["hidden"] / total, future=(m["future1"] + m["future2"]) / total, mm_hidden=m["hidden"],
                   n_missed=miss["n"], known_by_table=miss["median_share_known_by_table"], known_by_midnight=miss["median_share_known_by_midnight"])
        W = _dig(R, "day_windows", "river_flow_day", "summary")
        if W:
            part = W["parts"]["daylight_saving_time_in_force"]
            out.update(n_gauges=W["n_gauges"], summer_clock=part["n_gauges_best_midnight_summer_time"], standard_clock=part["n_gauges_best_midnight_standard_time"])
        return out
    return _guard("onset_rain_clock", read)


def rivers_facts(R):
    """E3 (code/17_more_basins.py, results/more_basins.json): training on 63 rivers instead of 12, for the same 12 rivers and for rivers never seen; test years, LSTM,
    by lead. None while the file is missing."""
    X = _file(R, "more_basins")
    if not X:
        return None

    def read():
        first, k = LEADS[0], lk(LEADS[0])
        T = X["summary_test"]["sets"]
        G = X["gain_rule"]["lstm63_wq_against_lstm_wq"]
        Gw = X["gain_rule"]["lstm63_w_against_lstm_w"]
        Geq = X["gain_rule"]["lstm63eq_wq_against_lstm_wq"]
        U = X["unseen"]
        loss = U["loss_from_not_having_seen_the_river"]
        dist = U["gain_over_the_05_models_against_distance"]["variants"]["with_properties"]["spearman_gain_against_km_to_the_nearest_of_the_50"]
        P, Vv = X["proximity"]["summary"], X["verdicts"]["sets"]
        st = X["stopping"]["sets"]
        n_seeds = len(X["design"]["seeds"])
        single = U["against_the_leave_one_out_models"]["unseen50_w_static_against_lobo_w_static"][k]["single_trainings"]
        pc = _dig(R, "published_context", "sources", "kratzert2024", "set_up", "data")
        return dict(
            n_onsets=Vv["lstm_wq"][k]["plain"]["onset"]["hits"] + Vv["lstm_wq"][k]["plain"]["onset"]["misses"],
            single_gain=[t["difference_of_medians"] for t in single], single_p=[t["wilcoxon_p"] for t in single],
            published_basins=int(re.search(r"(\d+) benchmark basins", pc).group(1)) if pc and re.search(r"(\d+) benchmark basins", pc) else None,
            nse={s: {j: T[s][lk(j)]["nse_median"] for j in LEADS} for s in ("lstm_wq", "lstm63_wq", "lstm63eq_wq", "lstm_w", "lstm63_w")},
            onset={s: {j: T[s][lk(j)]["onset"]["pod"] for j in LEADS} for s in ("lstm_wq", "lstm63_wq", "lstm63eq_wq", "lstm_w", "lstm63_w")},
            hits=(T["lstm_wq"][k]["onset"]["hits"], T["lstm63_wq"][k]["onset"]["hits"]), false_alarms=(T["lstm_wq"][k]["onset"]["false_alarms"], T["lstm63_wq"][k]["onset"]["false_alarms"]),
            csi=(T["lstm_wq"][k]["onset"]["csi"], T["lstm63_wq"][k]["onset"]["csi"]), csi_all=(T["lstm_wq"][k]["all"]["csi"], T["lstm63_wq"][k]["all"]["csi"]),
            chosen=(Vv["lstm_wq"][k]["chosen_on_validation_onsets"]["onset"]["pod"], Vv["lstm63_wq"][k]["chosen_on_validation_onsets"]["onset"]["pod"]),
            nse_range={s: X["seed_ranges"]["sets"][s][k]["nse_median"] for s in ("lstm_wq", "lstm63_wq")},
            nse_reading={j: G[lk(j)]["nse"]["reading"] for j in LEADS}, nse_basins={j: G[lk(j)]["nse"]["n_basins_higher"] for j in LEADS}, nse_n=G[k]["nse"]["n_basins"],
            nse_reading_w={j: Gw[lk(j)]["nse"]["reading"] for j in LEADS}, nse_reading_eq={j: Geq[lk(j)]["nse"]["reading"] for j in LEADS},
            onset_reading=G[k]["onset"]["reading"], mcnemar=G[k]["onset"]["paired"]["mcnemar_p"],
            best_update=(st["lstm_wq"]["mean_best_update"], st["lstm63_wq"]["mean_best_update"], st["lstm63eq_wq"]["mean_best_update"]),
            unseen={s: {j: U["sets"][s]["nse_median"][lk(j)] for j in LEADS} for s in ("lobo_w_static", "unseen50_w_static", "lobo_w_nostatic", "unseen50_w_nostatic", "lstm_w", "lstm63_w")},
            unseen_range={s: U["sets"][s]["nse_seed_range"][k] for s in ("unseen50_w_static", "unseen50_w_nostatic")},
            unseen_reading={w: {j: U["against_the_leave_one_out_models"][f"unseen50_{w}_against_lobo_{w}"][lk(j)]["nse"]["reading"] for j in LEADS} for w in ("w_static", "w_nostatic")},
            loss=(loss["leave_one_out_with_properties__05_recomputed"][k]["median_loss"], loss["50_rivers_with_properties__against_12_river_seed_average"][k]["median_loss"]),
            loss_n=(loss["leave_one_out_with_properties__05_recomputed"][k]["n_rivers_lower_when_never_seen"], loss["leave_one_out_with_properties__05_recomputed"][k]["n_rivers"]),
            km=(P["11_leave_one_out"]["median_km"], P["50"]["median_km"]), rho=(dist["rho"], dist["p"]), n_seeds=n_seeds,
            n_train={s: U["sets"][s]["n_training_rivers"] for s in ("lstm_w", "lstm63_w", "lobo_w_static", "unseen50_w_static")})
    return _guard("more_basins", read)


def moved_operating_point(pod0, pod1, far0, far1, csi0, csi1):
    """A lower share of onsets warned with a lower share of false warnings and no lower CSI (the hits, misses and false warnings together) is MOVED_POINT: the
    model warns less often and is right more often, which is a different operating point, not a loss of forecast skill. The three shares of the new model against the old."""
    return bool(pod1 < pod0 and far1 < far0 and csi1 >= csi0 - 0.005)


def review_sentence():
    """REVIEW_WORDING as a sentence (who checked what, and what only the final critic agent verified)."""
    return REVIEW_WORDING[0].upper() + REVIEW_WORDING[1:]


def hindsight_clause(H):
    """HINDSIGHT_WORDING ('hindsight at most about 3 points') to stand beside every 51%, word for word while the stored bound of the independent review (results/review_aorc_rain_to_midnight.json,
    check d7) rounds to the number in it; otherwise the same clause with the stored number. '' while the record is missing."""
    if not H:
        return ""
    return HINDSIGHT_WORDING if f"about {round(H['bound'])} points" in HINDSIGHT_WORDING else f"hindsight at most about {round(H['bound'])} points"


def nse_forecast_words(F):
    """What the forecast does to the median NSE, as a clause for every place that used to say 'mixed at all three lead times': NSE_FORECAST_WORDING word for word while the stored numbers say
    what it says (the 1-day median falls, the gains at 2 and 3 days are 'mixed' by the rule of section 4.2), otherwise a clause built from the stored gains and readings."""
    k1, rest = LEADS[0], LEADS[1:]
    if F["nse_gain"][k1] < 0 and all(F["nse_gain"][k] > 0 and F["nse_reading"][k] == "mixed" for k in rest):
        return NSE_FORECAST_WORDING
    return (f"in NSE the forecast changes the median by {_and([signed(F['nse_gain'][k]) for k in LEADS])} at {_and(LEADS)} days ahead "
            f"({_and([reading_word(F['nse_reading'][k]) for k in LEADS])})")


def nse_forecast_numbers(F, medians=True):
    """The numbers behind nse_forecast_words (results/forecast_input.json, gain_rule.wqaF_sub_against_wqa): the medians with and without the forecast (medians=False: where the text next to
    it has just given them), the basins that are higher or lower, whether every single training is on the same side, and the reading of section 4.2 at the leads with a positive gain."""
    k1, rest = LEADS[0], LEADS[1:]
    n = F["nse_n"]
    up = {k: F["nse_gain"][k] > 0 for k in LEADS}
    cnt = {k: (F["nse_basins"][k] if up[k] else n - F["nse_basins"][k]) for k in LEADS}
    every = {k: (F["nse_all_above"][k] if up[k] else F["nse_all_below"][k]) for k in LEADS}
    word = {True: "higher", False: "lower"}
    head = ((f"{LEAD_WORD[k1]} the median NSE goes from {n2(F['nse_rain'][k1])} to {n2(F['nse_fc'][k1])} (" if medians else f"{LEAD_WORD[k1]}: ")
            + f"{word[up[k1]]} in {cnt[k1]} of {n} basins" + (" and in every training" if every[k1] else "") + (")" if medians else ""))
    pairs = _and([f"{n2(F['nse_rain'][k])} to {n2(F['nse_fc'][k])}" for k in rest])
    where = (f"{word[up[rest[0]]]} in {_and([cnt[k] for k in rest])} of {n} basins" if len({up[k] for k in rest}) == 1 else
             _and([f"{word[up[k]]} in {cnt[k]} of {n}" for k in rest]) + " basins")
    rule = _and(sorted({F["nse_reading"][k] for k in rest}))
    return (f"{head}; {_leads(rest)} from {pairs} ({where}; &lsquo;{rule}&rsquo; by the rule of section 4.2)" if medians else
            f"{head}; {_leads(rest)}: {where}, &lsquo;{rule}&rsquo; by the rule of section 4.2")


def forecast_where(R, short=False):
    """The clause for the places that said a real weather forecast 'was not tested' or that 'the results do not say where' a real system would land (sections 2, 4.3, 4.5 and 5):
    where the one real forecast of section 4.8 lands at 2 and 3 days ahead, against the perfect rain, and its verdict for the onsets. None while results/forecast_input.json is missing -
    the caller then keeps its old sentence."""
    F = forecast_facts(R)
    if not F:
        return None
    k2, k3 = LEADS[1], LEADS[2]
    vo = set(F["verdict_onset"].values()) | set(F["verdict_onset_chosen"].values())
    still = f"still {next(iter(vo))}" if len(vo) == 1 else f"the verdict for the onsets is {_and(sorted(vo))}"
    return (f"section 4.8 tests one real forecast{'' if short else ', the control member of an archived reforecast'}: {_leads([k2, k3])} it warns of {pct(F['fc'][k2])} and {pct(F['fc'][k3])} of the "
            f"onsets (perfect rain: {pct(F['perfect'][k2])} and {pct(F['perfect'][k3])}), {still}")


def choices_after_test(R):
    """The number of decisions that each of the four results files of 7 and 8 October 2026 lists under choices_made_after_looking_at_test_scores (E1, E1b, E3, E2), or None."""
    out = []
    for name in ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input"):
        X = _file(R, name)
        c = X.get("choices_made_after_looking_at_test_scores") if X else None
        if not isinstance(c, list):
            return None
        out.append(len(c))
    return out


def four_experiments(R):
    """The four experiments of 7 and 8 October 2026, named once and counted one way (E1, E1b, E3, E2: the Stage IV check belongs to E1b)."""
    nw = new_work(R)
    when = f" on {nw['days']}" if nw else ""
    return (f"Four experiments were run{when}: E1 and E1b (rain to midnight from ERA5 and from AORC, with the Stage IV check; section 4.6), E3 (more rivers in training; section 4.7) and "
            f"E2 (a real weather forecast as an input; section 4.8)")



def forecast_facts(R):
    """E2 (code/gefs_to_basins.py and code/18_forecast_input.py, results/forecast_input.json): the GEFSv12 control member's rain forecast for the local days t+1 to t+3 as an
    input beside the AORC rain to midnight, LSTM with the river gauge. Main split (E2a: trained on the AORC rain of those days, run on the forecast, test years 2005-2014) and
    the satellite-era split (E2b: retrained with the forecast, test years 2010-2014); shares of the onsets by lead, at the fixed warning level unless a key says 'chosen' (the
    level chosen on the validation years). None while the file is missing."""
    X = _file(R, "forecast_input")
    if not X:
        return None

    def read():
        sets = {}
        for arm in ("main", "sat"):
            for r in X["ladder"][arm]["rungs"] + X["ladder"][arm].get("extras", []):
                sets[r["set"]] = r["by_lead"]
        by = lambda s, key: {k: sets[s][lk(k)][key] for k in LEADS}
        V, ret = X["value_of_the_forecast"], X["retrained_vs_substituted"]
        op = {k: V["operating_points"][lk(k)] for k in LEADS}
        rop = {k: ret["retrained_operating_points"][lk(k)] for k in LEADS}
        vs = {k: V["by_lead"][lk(k)] for k in LEADS}
        yard = X["yardstick"]["new_input_given_to_each_model"]
        sk = X["forecast_data"]["skill_summary"]
        margin = X["forecast_data"]["realtime_margin"]
        late = min(v["margin_minutes_of_the_last_file"] for v in margin.values())
        event = "5mm"                                                                      # the rain event of the stored Brier scores (the key of the results file)
        brier = {k: sk["brier_test_years_pooled"][lk(k)][event]["calibrated_probability"]["brier_skill_pooled"] for k in LEADS}
        v = X["verdicts"]["sets"]
        both = lambda s, k, lvl, kind: v[s][lk(k)][lvl][kind]["verdict"]
        boot = {k: op[k]["block_bootstrap_fixed_level"] for k in LEADS}
        ctrl_s = {k: ret["retrained_against_its_control"][lk(k)]["onsets"]["warned_without"] for k in LEADS}
        out = dict(
            n=by("lstm_wq", "n_onsets"), n_sat=by("lstm_wq_s", "n_onsets"),
            base=by("lstm_wq", "onset_warned"), rain=by("lstm_wqa", "onset_warned"), fc=by("lstm_wqaF_sub", "onset_warned"), ctrl={k: X["forecast_vs_climatology_control"]["by_lead"][lk(k)]["onsets"]["warned_without"] for k in LEADS},
            perfect=by("lstm_wqaP", "onset_warned"), upper=by("lstm_wqf", "onset_warned"), fc_alone=by("lstm_wqF_sub", "onset_warned"),
            nse_base=by("lstm_wq", "nse_median"), nse_rain=by("lstm_wqa", "nse_median"), nse_fc=by("lstm_wqaF_sub", "nse_median"), nse_perfect=by("lstm_wqaP", "nse_median"), nse_ctrl={k: X["forecast_vs_climatology_control"]["by_lead"][lk(k)]["nse"]["median_without"] for k in LEADS},
            fc_seeds=by("lstm_wqaF_sub", "onset_seed_range"), false_fc=by("lstm_wqaF_sub", "onset_false"), false_rain=by("lstm_wqa", "onset_false"),
            rain_chosen=by("lstm_wqa", "onset_warned_at_validation_chosen_level"), fc_chosen=by("lstm_wqaF_sub", "onset_warned_at_validation_chosen_level"),
            false_fc_chosen=by("lstm_wqaF_sub", "onset_false_at_validation_chosen_level"), false_rain_chosen=by("lstm_wqa", "onset_false_at_validation_chosen_level"),
            gain={k: vs[k]["onsets"]["difference_points"] for k in LEADS}, gain_reading={k: vs[k]["onsets"]["reading"] for k in LEADS},
            nse_gain={k: vs[k]["nse"]["gain"] for k in LEADS}, nse_reading={k: vs[k]["nse"]["reading"] for k in LEADS}, nse_basins={k: vs[k]["nse"]["n_basins_higher"] for k in LEADS},
            nse_all_below={k: bool(X["gain_rule"]["wqaF_sub_against_wqa"][lk(k)]["nse"]["single_seeds_all_below"]) for k in LEADS},
            nse_all_above={k: bool(X["gain_rule"]["wqaF_sub_against_wqa"][lk(k)]["nse"]["single_seeds_all_above"]) for k in LEADS},
            nse_gain_simple={m: {k: yard[m]["by_lead"][lk(k)]["nse_gain"] for k in LEADS} for m in yard},
            nse_n=vs[LEADS[0]]["nse"]["n_basins"], chosen_gain={k: op[k]["level_chosen_on_the_validation_years"]["points"] for k in LEADS},
            csi=(op[LEADS[0]]["fixed_level"]["csi_without"], op[LEADS[0]]["fixed_level"]["csi_with"]), far=(op[LEADS[0]]["fixed_level"]["false_without"], op[LEADS[0]]["fixed_level"]["false_with"]),
            pod=(op[LEADS[0]]["fixed_level"]["onsets_warned_without"], op[LEADS[0]]["fixed_level"]["onsets_warned_with"]),
            ap={k: (op[k]["trigger_free"]["average_precision_without"], op[k]["trigger_free"]["average_precision_with"]) for k in LEADS},
            ap_diff={k: op[k]["trigger_free"]["average_precision_difference"] for k in LEADS}, ap_ci={k: op[k]["trigger_free"]["average_precision_difference_ci95"] for k in LEADS},
            ci={k: boot[k]["pod_points_ci95"] for k in LEADS}, years_above={k: (boot[k]["water_years_above_zero"], boot[k]["n_water_years"]) for k in LEADS},
            midnight_given_forecast={k: X["forecast_without_rain_to_midnight"]["rain_to_midnight_given_the_forecast"][lk(k)]["onsets"]["difference_points"] for k in LEADS},
            share_gap={k: X["how_far_from_perfect"]["share_of_the_gap_closed"]["rain_only_gap"][k - 1]["onset_share_of_gap_closed"] for k in LEADS},
            s_wq=by("lstm_wq_s", "onset_warned"), s_rain=by("lstm_wqa_s", "onset_warned"), s_fc=by("lstm_wqaF_s", "onset_warned"), s_ctrl=ctrl_s, s_perfect=by("lstm_wqaP_s", "onset_warned"),
            s_nse_rain=by("lstm_wqa_s", "nse_median"), s_nse_fc=by("lstm_wqaF_s", "nse_median"), s_nse_wq=by("lstm_wq_s", "nse_median"),
            s_gain={k: ret["retrained"][lk(k)]["onsets"]["difference_points"] for k in LEADS}, s_gain_chosen={k: rop[k]["level_chosen_on_the_validation_years"]["points"] for k in LEADS},
            s_reading={k: ret["retrained"][lk(k)]["onsets"]["reading"] for k in LEADS}, s_ctrl_reading={k: ret["retrained_against_its_control"][lk(k)]["onsets"]["reading"] for k in LEADS},
            gain_simple={m: {k: yard[m]["by_lead"][lk(k)]["onset_gain_points"] for k in LEADS} for m in yard},
            warned_simple={m: {k: yard[m]["by_lead"][lk(k)]["onset_warned_with"] for k in LEADS} for m in yard},
            far_simple={m: {k: yard[m]["by_lead"][lk(k)]["onset_false_with"] for k in LEADS} for m in yard},
            corr={k: sk["correlation_all_median_over_basins"][lk(k)] for k in LEADS}, ratio={k: sk["mean_ratio_all_median_over_basins"][lk(k)] for k in LEADS},
            ratio_summer=sk["mean_ratio_by_season_median"][lk(LEADS[0])]["JJA"], ratio_winter=sk["mean_ratio_by_season_median"][lk(LEADS[0])]["DJF"], brier=brier, brier_event=event.replace("mm", " mm"),
            late=late, n_basins=len(X["per_basin"]["basins"]), n_cells=(min(len(b["weights"]) for b in R["gefs_basins"]["basins"].values()), max(len(b["weights"]) for b in R["gefs_basins"]["basins"].values())) if "gefs_basins" in R else None,
            verdict_onset={k: both("wqaF_sub", k, "plain", "onset") for k in LEADS},
            verdict_onset_chosen={k: both("wqaF_sub", k, "chosen_on_validation_all_days", "onset") for k in LEADS},
            verdict_all={k: both("wqaF_sub", k, "plain", "all") for k in LEADS}, verdict_all_chosen={k: both("wqaF_sub", k, "chosen_on_validation_all_days", "all") for k in LEADS},
            verdict_all_s={k: both("wqaF_s", k, "plain", "all") for k in LEADS}, verdict_all_s_chosen={k: both("wqaF_s", k, "chosen_on_validation_all_days", "all") for k in LEADS},
            success=bool(X["e2_success"]["met_at_2_and_3_days"]))
        out["moved"] = moved_operating_point(out["pod"][0], out["pod"][1], out["far"][0], out["far"][1], out["csi"][0], out["csi"][1])
        ages = [int(m.group(1)) for key in margin for m in [re.search(r"(\d+) h old", key)] if m]
        days = re.search(r"(\d+) days", str(_dig(R, "gefs_basins", "realtime_margin", "source", default="")))
        memb = re.search(r"c00 and p(\d+)-p(\d+)", str(_dig(R, "data_access", "summary", "archived_weather_forecasts", "gefsv12_reforecast_aws", "what", default="")))
        out.update(hours_old=(min(ages), max(ages)) if ages else None, margin_days=int(days.group(1)) if days else None,
                   members=1 + int(memb.group(2)) - int(memb.group(1)) + 1 if memb else None,
                   years=(_dig(R, "gefs_basins", "issue_days", "first", default="")[:4], _dig(R, "gefs_basins", "issue_days", "last", default="")[:4]))
        pub = _dig(R, "published_context", "sources", "guan2022", "numbers")
        if pub:                                                                  # the published Brier skill scores of the 11-member ensemble, 5 mm, 12-36 h and 60-84 h
            i = next(j for j, x in enumerate(pub) if ">= 5 mm" in x["quantity"])
            first = lambda x: float(re.search(r"[\d.]+", x["value"]).group())
            out["published"] = (first(pub[i]), first(pub[i + 1]))
        return out
    return _guard("forecast_input", read)


# shared table style of the ladders of sections 4.6 to 4.8 (a value with the lowest and highest single seed under it in small grey type, as Table 4.5)
LADDER_STYLE = ("<style>table.tl{font-size:8.3pt} table.tl th,table.tl td{padding:1.4pt 2.5pt;line-height:1.2;text-align:right} table.tl th{font-size:7.6pt;font-weight:400;white-space:nowrap}"
                "table.tl td:first-child,table.tl th:first-child{text-align:left;padding-left:4pt} table.tl th.grp{text-align:center;font-weight:600;font-size:7.9pt;"
                "border-bottom:1px solid var(--grid)} table.tl .g{border-left:1px solid var(--axis)} table.tl tr.sub th{border-top:none} table.tl .sr{font-size:7pt;color:var(--muted);"
                "white-space:nowrap} table.tl tr.ctl td{color:var(--ink2)} table.tl tr.ctl td:first-child{padding-left:11pt} table.tl tr.hd td{border-top:1px solid var(--axis);"
                "font-weight:600;font-size:7.8pt;color:var(--ink2);padding-top:3pt;text-align:left}"
                # the closing note of a section as a plain paragraph in the style of .note: it does not clear the floats, so it can fill the foot of the page before a figure that moved on
                ".nbx{border-left:3px solid var(--orange);padding:4pt 9pt;margin:5pt 0;font-size:8.9pt;background:#fdf6f2}</style>")


def ladder_cell(v, r=None, share=False):
    """A cell of a ladder: the value of the seed-averaged forecast and, under it in small grey type, the lowest and highest single training."""
    if v is None:
        return "&ndash;"
    if share:
        small = (pct(r[0]) if pct(r[0]) == pct(r[1]) else f"{100 * r[0]:.0f}&ndash;{100 * r[1]:.0f}%") if r else ""
    else:
        small = f"{n2(r[0])}&ndash;{n2(r[1])}" if r else ""
    big = pct(v) if share else n2(v)
    return f'{big}<br><span class="sr">{small}</span>' if r else big


def _days_words(isos):
    """['2026-10-07', '2026-10-08'] -> '7 and 8 October 2026'."""
    days = [datetime.date.fromisoformat(x) for x in sorted(set(isos))]
    if not days:
        return ""
    if len({(d.month, d.year) for d in days}) == 1:
        return _and([str(d.day) for d in days]) + f" {days[0]:%B %Y}"
    return _and([f"{d.day} {d:%B %Y}" for d in days])


NEW_REVIEWS = ("review_rain_to_midnight", "review_aorc_rain_to_midnight", "review_more_basins", "review_forecast_input")


def new_work(R):
    """What was done on the days after 4 October 2026, from the four independent review records (results/review_*.json): dict(days, n, files) or None.
    The days are the dates that the records give for the runs and the review ('when' and 'dates'); the experiments are those the records review."""
    revs = [(n, _file(R, n)) for n in NEW_REVIEWS]
    revs = [(n, r) for n, r in revs if r]
    if not revs:
        return None
    isos = {x for _, r in revs for v in [str(r.get("when", ""))] + [str(y) for y in (r.get("dates") or {}).values()] for x in re.findall(r"\d{4}-\d{2}-\d{2}", v)}
    return dict(days=_days_words(isos), n=len(revs), files=[f"results/{n}.json" for n, _ in revs])


def review_clause(R, *names):
    """One sentence on the independent AI reviews of the experiments (results/review_<name>.json for each name, read by key): that they recomputed the headline numbers, in how many
    listed checks, that none failed and no number differed (counts), that they found no leakage where a verdict says so, and that the items they asked to be worded differently
    (must_fix, should_fix) are followed. Ends with a full stop; empty while no record exists."""
    revs = [(n, _file(R, f"review_{n}")) for n in names]
    revs = [(n, r) for n, r in revs if r]
    if not revs:
        return ""
    def _rounds(r):                                                   # the counts of every round of one record (round 1 at the top, later rounds under round_<n>)
        out = [r.get("counts") or {}]
        out += [v["counts"] for k, v in r.items() if k.startswith("round_") and isinstance(v, dict) and isinstance(v.get("counts"), dict)]
        return out
    rounds = [(r, _rounds(r)) for _, r in revs]
    cs = [c for _, rc in rounds for c in rc]
    n_rounds = len(cs)
    clean = all(c.get("failed") == 0 and c.get("numerical_discrepancies") == 0 for c in cs)
    leak = all(re.search(r"no leakage|no way for information from after the issue time|no hour after a window's end enters any input", str(r.get("verdict", ""))) for _, r in revs)
    many = len(revs) > 1
    files = _and([f"<code>results/review_{n}.json</code>" for n, _ in revs])
    return (f"The independent AI review{'s' if many else ''} ({files}; {'AI agents, not team members' if many else 'an AI agent, not a team member'}) recomputed the headline numbers in "
            f"{sum(c.get('checks_listed', 0) for c in cs)} listed checks" + (f" over {_word(n_rounds)} rounds" if n_rounds > len(revs) else "") + ": "
            + ("none failed, no number differed" if clean else "not every check passed (see the record)") + (", no leakage" if leak and clean else "")
            + f"; the wording items {'they' if many else 'it'} asked for ({_word(sum(c.get('must_fix', 0) + c.get('should_fix', 0) for c in cs))}) are followed here.")


def ladder_table(rows, caption, far=True, label="Inputs"):
    """A ladder of input sets (sections 4.6 and 4.8): onsets warned and median NSE for 1, 2 and 3 days ahead, each value with the lowest and highest single training under it
    (small grey type), and, if far, the share of false warnings among the onset warnings 1 day ahead. rows: dicts - {'head': text} for a heading row, or {'label', 'cls'
    (optional: 'ctl' for a control), 'on' and 'nse' ({lead: value}), 'on_r' and 'nse_r' ({lead: [lowest, highest]}), 'far' (a share, optional)}."""
    cols = 7 + bool(far)
    head = (f'<tr><th rowspan="2">{label}</th><th class="g grp" colspan="3">Onsets warned</th><th class="g grp" colspan="3">Median NSE</th>'
            + ('<th class="g grp" rowspan="2">False<br>warnings</th>' if far else "") + "</tr><tr class=\"sub\">"
            + "".join(f'<th class="{"g" if n % 3 == 0 else ""}">{LEAD_WORD[k].replace(" ahead", "")}</th>' for n, k in enumerate(tuple(LEADS) * 2)) + "</tr>")
    body = ""
    for r in rows:
        if "head" in r:
            body += f'<tr class="hd"><td colspan="{cols}">{r["head"]}</td></tr>'
            continue
        cells = [ladder_cell(r["on"].get(k), (r.get("on_r") or {}).get(k), True) for k in LEADS] + [ladder_cell(r["nse"].get(k), (r.get("nse_r") or {}).get(k)) for k in LEADS]
        if far:
            cells.append(pct(r["far"]) if r.get("far") is not None else "&ndash;")
        body += (f'<tr class="{r.get("cls", "")}"><td>{r["label"]}</td>'
                 + "".join(f'<td class="{"g" if i in (0, 3) or (far and i == 6) else ""}">{c}</td>' for i, c in enumerate(cells)) + "</tr>")
    widths = [33] + [9.6] * 6 + ([9.4] if far else [])
    return ('<table class="num tl"><colgroup>' + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>" + head + body + "</table>"
            + f'<p class="cap">{caption}</p>')


def ladder_row(label, by_lead, cls="", prefix="", far_lead=None):
    """One row of a ladder from a dict {'l1': {...}, ...} of the results files: the keys 'onset_warned', 'onset_seed_range', 'nse_median', 'nse_seed_range' and 'onset_false', each
    with `prefix` in front (for a control stored inside its rung: 'control_'). far_lead: the lead of the false-warning share (the first lead by default)."""
    k0 = far_lead or LEADS[0]
    g = lambda key, k: by_lead[lk(k)].get(prefix + key)
    return dict(label=label, cls=cls, on={k: g("onset_warned", k) for k in LEADS}, on_r={k: g("onset_seed_range", k) for k in LEADS}, nse={k: g("nse_median", k) for k in LEADS},
                nse_r={k: g("nse_seed_range", k) for k in LEADS}, far=g("onset_false", k0))


def milestone2(cls, higher, n, n_seeds):
    """Milestone 2 of the plan for the two forecasts that could be run in practice, from the readings of section 4.2:
    (with the river gauge, with past weather only), each a short phrase. cls = {input set: {lead: reading}}; higher(inp) = the basin counts."""
    wq, w = cls["wq"], cls["w"]
    if all(v in ("tie", "simple", "lstm_med") for v in wq.values()):
        gauge = "not met with the river gauge" + (" (no clear difference)" if all(v == "tie" for v in wq.values()) else "")
    elif all(v == "lstm" for v in wq.values()):
        gauge = "met with the river gauge"
    else:
        gauge = "met in part with the river gauge"
    if all(v == "lstm_avg" for v in w.values()):
        most = all(c > n / 2 for c in higher("w", raw=True))
        weather = (f"with past weather only the {_word(n_seeds)}-seed average is ahead in {'most' if most else 'some'} basins "
                   f"({higher('w')} of {n}), a single training is not clearly ahead")
    elif all(v == "lstm" for v in w.values()):
        weather = "met with past weather only"
    elif all(v in ("tie", "simple", "lstm_med") for v in w.values()):
        weather = "not met with past weather only"
    else:
        weather = "met in part with past weather only"
    return gauge, weather


# ---- everything page 1 says -------------------------------------------------------------------------------------------
def _new_rungs(R, first, own):
    """The sentences, big numbers and table rows that the experiments of 7 and 8 October 2026 add to page 1, README.md, short_report.md and Table 8.1, every number read by key
    (aorc_facts, stage4_facts, forecast_facts, rivers_facts, era5_facts) and every outcome-dependent wording chosen from the stored numbers. Keys: yard (finding 1: the simple models
    given the same inputs), rungs (finding 2: the new rungs), verdict and rivers (finding 3), took (what it took and what is missing), b2 and c4 (short_report.md), big (numbers),
    rows (Table 8.1). A key is '' (or an empty list) while the results files it needs are missing."""
    from report_sections.s4_all import _reading
    out = dict(yard="", yard_p1="", rungs="", verdict="", verdict_p1="", rivers="", took="", b2="", c4="", big=[], rows=[])
    AO, S4, FC, RV, E1, HS = aorc_facts(R), stage4_facts(R), forecast_facts(R), rivers_facts(R), era5_facts(R), hindsight_facts(R)
    k2, k3 = LEADS[1], LEADS[2]
    if RV:
        n11, n50 = RV["n_train"]["lobo_w_static"], RV["n_train"]["unseen50_w_static"]
        out["rivers"] = (f"; with {n50} training rivers instead of {n11} the loss shrinks to {n2(RV['loss'][1])} and the median rises from {n2(RV['unseen']['lobo_w_static'][first])} to "
                         f"{n2(RV['unseen']['unseen50_w_static'][first])}, though the number, variety and nearness of the training rivers were not separated")
    if not (AO and S4 and FC):
        if RV:
            out["rows"] = [[f"<b>More rivers</b>", f"{RV['n_train']['lstm63_w']} training rivers instead of {RV['n_train']['lstm_w']}", _rivers_found(RV, first), "4.7"]]
        return out
    g, gs = AO["gain"], FC["gain_simple"]
    as_1 = min(g["ridge"], g["forest"]) >= AS_MUCH * g["lstm"]
    as_2 = all(min(gs["ridge regression"][k], gs["random forest"][k]) >= AS_MUCH * gs["lstm"][k] for k in LEADS[1:])
    ng = AO["nse_gain"]                                                                  # the NSE gain with the rain to midnight: the simple models do not gain as much as in onsets
    times = ng["lstm"] / max(ng["ridge"], ng["forest"]) if max(ng["ridge"], ng["forest"]) > 0 else None
    more = ("about twice as much" if times and 1.6 <= times < 2.6 else "more than twice as much" if times and times >= 2.6 else "more" if times and times > 1.1 else "")
    nse_end = (f"; in median NSE the LSTM gains {more} with rain up to midnight ({signed(ng['lstm'])} against {signed(ng['ridge'])} and {signed(ng['forest'])})" if more else "")
    info = (f": {INFO_NOT_NET}, for onsets" if as_1 and as_2 else "")
    out["yard"] = (f"Given the new inputs, ridge regression and the random forest gain {'as many onsets as' if as_1 and as_2 else 'not as many onsets as'} the LSTM does "
                   f"({signed(g['ridge'], 0)} and {signed(g['forest'], 0)} points against {signed(g['lstm'], 0)} with rain up to midnight; {signed(gs['ridge regression'][k2], 0)} and "
                   f"{signed(gs['random forest'][k2], 0)} against {signed(gs['lstm'][k2], 0)} with a forecast)" + info + nse_end + ".")
    out["yard_p1"] = (f"Given the new inputs, the simple models gain {'as many onsets as' if as_1 and as_2 else 'not as many onsets as'} the LSTM does" + info + nse_end + ".")   # page 1: without the points
    share48 = S4["stage4"] / S4["n"]
    hind = hindsight_clause(HS)
    out["rungs"] = (f"With the rain of the whole issue day up to midnight from AORC, {AORC_WORDING}, {pct(AO['with_'][first])} of the onsets are warned "
                    f"{LEAD_WORD[first]} (control {pct(AO['to09'])}; {hind + '; ' if hind else ''}{pct(share48)} with Stage IV: {REALTIME_WORDING}), and with a public weather forecast beside it "
                    f"(one ensemble member) {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])} {_leads([k2, k3])}, against {pct(FC['base'][k2])} and {pct(FC['base'][k3])} with the rain table alone.")
    vs = set(AO["verdict_onset"].values()) | set(FC["verdict_onset"].values()) | set(FC["verdict_onset_chosen"].values())
    same = f"stays {next(iter(vs))}" if len(vs) == 1 else f"is {_and(sorted(vs))}"
    out["verdict"] = (f"; the verdict for onsets {same} with every new input ({pct(AO['false'][first])} of the warnings false with rain up to midnight; "
                      f"{pct(FC['false_fc'][k2])} and {pct(FC['false_fc'][k3])} {_leads([k2, k3])} with the forecast)")
    out["verdict_p1"] = f"; the verdict for onsets {same} with every new input ({pct(AO['false'][first])} of the warnings false with rain up to midnight)"        # page 1: 'What it took' says the rest
    out["b2"] = (f"With the new inputs: {tag(next(iter(vs))) if len(vs) == 1 else _and(sorted(vs))} for onsets at every lead time ({LEAD_WORD[first]} {pct(AO['with_'][first])} warned with the rain up to midnight, "
                 f"{pct(AO['false'][first])} of the warnings false; {_leads([k2, k3])} {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])} warned with the forecast).")
    out["c4"] = out["rungs"] + " " + out["yard"]
    f1, f2, f3 = (FC["fc"][k] for k in LEADS)
    tail = own.replace("; and the team&rsquo;s own checks of the code.", ", the team&rsquo;s own checks.", 1)           # two lines on page 1: the shorter tail of the sentence
    out["took"] = (f"It took two new inputs, the rain of the whole issue day up to midnight and a public weather forecast. Missing: {share_word(AO['false'][first])} "
                   f"of the onset warnings are false (still {next(iter(vs)) if len(vs) == 1 else _and(sorted(vs))}), the ensemble, a network trained on Stage IV, five trainings in place of three "
                   f"where the NSE decides{tail}")
    still_v = f"still {next(iter(vs)) if len(vs) == 1 else _and(sorted(vs))}"
    out["big"] = [(pct(AO["with_"][first]), f"of onsets warned {LEAD_WORD[first]} with the rain up to midnight from AORC, {AORC_WORDING} (control {pct(AO['to09'])}; {hind + '; ' if hind else ''}"
                                           f"{pct(share48)} with Stage IV: {REALTIME_WORDING}); {still_v}"),
                  (f"{pct(f2)} / {pct(f3)}", f"of onsets warned {_leads([k2, k3]).replace(' ahead', '')} ahead with a public weather forecast beside that rain, against "
                                              f"{pct(FC['base'][k2])} / {pct(FC['base'][k3])} with the rain table alone; {still_v}")]
    if E1:
        era = (f"onsets warned {pct(E1['base'])} to {pct(E1['with_'])} with ERA5")
    else:
        era = ""
    out["rows"] = [[f"<b>Rain to midnight</b>", "the whole issue day, ERA5 and AORC rain", f"{era + '; ' if era else ''}{pct(AO['base'][first])} to {pct(AO['with_'][first])} with AORC ({AORC_WORDING}; "
                    f"{hind + '; ' if hind else ''}{pct(share48)} with Stage IV: {REALTIME_WORDING}); {still_v}", "4.6"]]
    if RV:
        out["rows"].append([f"<b>More rivers</b>", f"{RV['n_train']['lstm63_w']} training rivers instead of {RV['n_train']['lstm_w']}", f"{_rivers_found(RV, first)}; {still_v}", "4.7"])
    out["rows"].append([f"<b>Weather forecast</b>", "GEFSv12 control member, rain of the next three days",
                        f"onsets warned {_leads([k2, k3]).replace(' ahead', '')} ahead {pct(FC['rain'][k2])} / {pct(FC['rain'][k3])} to {pct(f2)} / {pct(f3)} beside the rain to midnight; {still_v}; "
                        f"{nse_forecast_words(FC)} ({LEAD_WORD[first]} {n2(FC['nse_rain'][first])} to {n2(FC['nse_fc'][first])})",
                        "4.8"])
    return out


def _rivers_found(RV, first):
    """The 'what was found' cell of Table 8.1 for E3."""
    return (f"the same {RV['n_train']['lstm_w']} rivers: NSE {n2(RV['nse']['lstm_wq'][first])} to {n2(RV['nse']['lstm63_wq'][first])} ({by_lead_reading(RV['nse_reading'])}); "
            f"rivers never seen: {n2(RV['unseen']['lobo_w_static'][first])} to {n2(RV['unseen']['unseen50_w_static'][first])}")


def facts(R, T):
    from report_sections.s4_all import PHRASE, _p, _reading, _say              # the rule of section 4.2 for 'better' / 'no clear difference'
    first, last = LEADS[0], LEADS[-1]
    A, B, F = R["lstm_all"], R["baselines"], R["flood_analysis"]
    C, LV, ST = A["comparison"], A["decomposition"]["levels"], A["decomposition"]["steps"]
    n = len(A["basins"])
    n_seeds = len(A["sets"]["wq"]["seeds"])
    best = lambda inp, k: C[inp]["best_baseline_by_validation"][lk(k)]
    cmp_ = lambda inp, k: C[inp]["baselines"][best(inp, k)][lk(k)]
    cls = {inp: {k: _reading(cmp_(inp, k)) for k in LEADS} for inp in C}
    three = lambda vals, fmt=n2: ", ".join(fmt(v) for v in vals[:-1]) + f" and {fmt(vals[-1])}"
    higher = lambda inp, ks=LEADS, raw=False: ([cmp_(inp, k)["n_basins_lstm_higher"] for k in ks] if raw else
                                               _and(cmp_(inp, k)["n_basins_lstm_higher"] for k in ks))

    # (a) how good are the forecasts, and against what
    wq, w = cls["wq"], cls["w"]
    lstm_nse, base_nse = [cmp_("wq", k)["lstm_nse_median"] for k in LEADS], [cmp_("wq", k)["baseline_nse_median"] for k in LEADS]
    m2_gauge, m2_weather = milestone2(cls, higher, n, n_seeds)
    stronger = all(cmp_("wq", k)["lstm_nse_median"] > cmp_("w", k)["lstm_nse_median"] for k in LEADS)
    practice = "the stronger of the two forecasts that could be run in practice" if stronger else "one of the two forecasts that could be run in practice"
    a1 = (f"With past weather + river gauge, {practice}, the LSTM&rsquo;s median NSE is {three(lstm_nse)} "
          f"({_leads(LEADS)}) against {three(base_nse)} for the best simple model: {_say(wq).replace('there is ', '')} (LSTM higher in {higher('wq')} of "
          f"{n} basins), so milestone 2 of the plan, &lsquo;the LSTM beats the baselines&rsquo;, is <b>{m2_gauge.split(' (')[0]}</b>.")
    if all(v == "lstm_avg" for v in w.values()):
        a2 = (f"With past weather only, the average of the LSTM&rsquo;s {_word(n_seeds)} seeds is ahead (in {higher('w')} of {n} basins), but by less than "
              f"the seed spread, the difference between trainings of one model: a single training is not clearly ahead.")
    else:
        a2 = f"With past weather only, {_say(w)}."
    fair = [k for k in LEADS if cmp_("wqf", k).get("like_for_like") is not False]
    soft = dict(PHRASE, lstm="the LSTM scores higher", simple="the simple model scores higher")       # upper bound: the two do not read exactly the same variables
    if fair and len(fair) < len(LEADS):
        c3_ = cmp_("wqf", fair[-1])
        nums = (f" ({n2(c3_['lstm_nse_median'])} against {n2(c3_['baseline_nse_median'])}; in {c3_['n_basins_lstm_higher']} of {n} basins, "
                f"p = {_p(c3_['wilcoxon_p'])})") if cls["wqf"][fair[-1]] in ("lstm", "simple", "lstm_avg") else ""
        a3 = (f"Given the gauge and a perfect weather forecast, an upper bound that cannot be had in practice, {soft[cls['wqf'][fair[-1]]]} "
              f"{_leads(fair)}{nums}, the only lead time at which both read the same days of weather.")
    else:
        a3 = (f"With the river gauge and a perfect weather forecast (an upper bound that cannot be had in practice), "
              f"{_say(cls['wqf']).replace(PHRASE['lstm'], soft['lstm']).replace(PHRASE['simple'], soft['simple'])}.")

    # (b) could the warnings be trusted
    V, S = F["verdict"]["services"], F["skill"]
    v_all, v_on = V["lstm_wq"][lk(first)]["all"], {k: V["lstm_wq"][lk(k)]["onset"] for k in LEADS}
    plain = lambda name, k, kind: S[name]["plain"]["test"]["summary"][lk(k)][kind]
    n_days = plain("lstm_wq", first, "all")["hits"] + plain("lstm_wq", first, "all")["misses"]
    n_on = plain("lstm_wq", first, "onset")["hits"] + plain("lstm_wq", first, "onset")["misses"]
    pers = F["verdict"]["other_models_pooled"].get("base_persistence", {}).get(lk(first), {}).get("all")
    same = (f", the same verdict as {QUOTED}: {pct((n_days - n_on) / n_days)} of high-flow days come when the river is already high"
            if pers == v_all["pooled"] else "")
    on_verdicts = {v_on[k]["pooled"] for k in LEADS}
    on_where = "at every lead time" if len(on_verdicts) == 1 else _and(f"{v_on[k]['pooled']} {LEAD_WORD[k]}" for k in LEADS)
    b1 = (f"The warnings: {tag(v_all['pooled'])} for all high-flow days {LEAD_WORD[first]} with the river gauge ({pct(v_all['pod'])} warned, "
          f"{pct(v_all['far'])} false warnings){same}. For onsets, the warnings that matter: "
          + (f"{tag(next(iter(on_verdicts)))} {on_where}" if len(on_verdicts) == 1 else on_where)
          + f" ({LEAD_WORD[first]} {pct(v_on[first]['pod'])} warned, {pct(v_on[first]['far'])} of those warnings false).")

    # (c) why: the same inputs give the same skill whatever the model; more information lifts every model
    gauge, fc = ST["river_gauge_without_forecast"], ST["perfect_forecast_with_gauge"]
    biggest = max(abs(cmp_(inp, k)["difference_of_medians"]) for inp in C for k in LEADS if cmp_(inp, k).get("like_for_like") is not False)
    limit = min(fc[lk(k)]["nse_median_gain"] for k in LEADS) > biggest
    up_on = LV["wqf"][lk(first)]["onset_pod"]
    SB = {(r["model"], r["lead"]): r["nse_median"] for r in B["summary_table"]}
    simple_gain = [SB[f"base_{m}_wq", first] - SB[f"base_{m}_w", first] for m in ("linreg", "rf") if (f"base_{m}_wq", first) in SB and (f"base_{m}_w", first) in SB]
    apart = (max(abs(R["model_comparison"]["comparison"]["test"][lk(k)]["nse_median_transformer_minus_lstm"]) for k in LEADS)
             if "model_comparison" in R and not R["model_comparison"].get("quick_mode") else None)
    tie = (f"With the inputs that can be had in practice, models of different kinds lie close together: the LSTM and the best simple model (ridge regression or random "
           f"forest) are never more than {n2(biggest)} NSE apart where both read the same days of weather"
           + (f", Transformer and LSTM never more than {n2(apart)}" if apart is not None else "") + ". ")
    g1 = gauge[lk(first)]["nse_median_gain"]
    all_gain = g1 > 0 and simple_gain and min(simple_gain) > 0
    gauge_adds = (f"the river gauge adds {signed(g1)} NSE {LEAD_WORD[first]} to the LSTM ({signed(min(simple_gain))} to "
                  f"{signed(max(simple_gain))} to the two simple models)" if all_gain else
                  f"the river gauge adds {signed(g1)} NSE {LEAD_WORD[first]} to the LSTM")
    more = ("More information lifts each: " + gauge_adds) if all_gain else gauge_adds[0].upper() + gauge_adds[1:]
    c1 = ((FINDING + ". " + tie if limit else "")
          + f"{more}; the weather of the days being forecast, an upper bound, would add {three([fc[lk(k)]['nse_median_gain'] for k in LEADS], signed)} "
            f"and lift the onsets warned {LEAD_WORD[first]} from {pct(LV['wq'][lk(first)]['onset_pod'])} to {pct(up_on)}.")
    tf = timing_facts(R)
    c2 = timing_sentence(R, short=True) if tf else ""
    c3, clear, s1, likely = "", [], None, False
    if "satellite_inputs" in R and "satellite_data" in R:
        sat = R["satellite_inputs"]["experiments"]["a_rain_source"]["against_base"]["imerg_wq"]
        clear = [k for k in LEADS if sat[lk(k)]["nse_median"]["larger_than_seed_noise"] is True and sat[lk(k)]["nse_median"]["difference"] > 0]
        r_sat = R["satellite_data"]["rain_check_summary"]["imerg"]["median_correlation"]
        know = R["satellite_inputs"].get("what_inputs_know_about_next_daymet_day", {})                # does satellite rain of the issue day hold later rain?
        nxt = lambda name: know.get(name, {}).get("correlation_with_daymet_rain_of_the_next_day")
        likely = bool(nxt("satellite rain (IMERG)") and nxt("gauge-based rain (Daymet) itself") is not None
                      and nxt("satellite rain (IMERG)") > nxt("gauge-based rain (Daymet) itself"))
        if clear:
            s1 = sat[lk(clear[0])]["nse_median"]
            c3 = (f"Satellite rain, whose day ends in the early evening of the issue day, improves the model with the river gauge {_leads(clear)} "
                  f"({n2(s1['base'])} to {n2(s1['variant'])} {LEAD_WORD[clear[0]]}, own split)"
                  + (f": probably the same time-stamp effect ({SAT_REF})." if likely else f" ({SAT_REF})."))
        else:
            c3 = f"Satellite rain instead of the rain table (correlation {n2(r_sat)}) brings no clear change with the river gauge."

    # (d) what would have to be added: a weather forecast at every lead time; rain observations up to the hour add a part 1 day ahead
    lf = lobo_facts(R)
    tv = timing_verdict(R)
    if tf and tv and onset_clear(tf["gauge"][first]):
        r = tf["gauge"][first]
        part = "the larger part" if r["onset_share_of_gap_closed"] < 0.5 else "part"
        obs = (f"a weather forecast as an input at every lead time (not tested): it is {part} of the gap even {LEAD_WORD[first]}; rain observations up to "
               f"the hour (they exist today) add a part {LEAD_WORD[first]}, but with them the onsets are still {tv['verdict']} "
               f"({pct(tv['pod'])} warned, {pct(tv['far'])} false)" if tv["verdict"] != "GO" else
               f"a weather forecast as an input at every lead time (not tested); rain observations up to the hour (they exist today), with which the "
               f"onsets {LEAD_WORD[first]} reach GO")
    else:
        obs = "a weather forecast as an input at every lead time (not tested); rain observations up to the hour (they exist today; their worth was not shown here)"
    checks = team.checks_state()                                                 # the members' own checks, as team.json has them
    own = ("; and the team&rsquo;s own checks of the code." if checks["n_done"] == 0 else
           f"; and the rest of the team&rsquo;s own checks of the code ({checks['n_done']} of {checks['n']} {'is' if checks['n_done'] == 1 else 'are'} done)."
           if checks["n_done"] < checks["n"] else ".")
    take = lambda bracket: (f"Before anyone relied on it: {obs}; {'' if own != '.' else 'and '}more basins for rivers never seen"
                            + (f" ({unseen_sentence(R, short=True)})" if lf and bracket else "") + own)
    d1, d1_bare = take(True), take(False)                                         # d1_bare: where the sentence before it has just given the unseen-river numbers
    new = _new_rungs(R, first, own)                                               # what the experiments of 7 and 8 October 2026 add to page 1 (empty strings while a results file is missing)
    d_head = "What it took, and what is missing." if new["took"] else "What it would take."
    if new["took"]:
        d1 = d1_bare = new["took"]
    # the answer in one line: with a river gauge, then without one
    flow = ("forecasts the flow no better than a simple model" if all(v in ("tie", "lstm_med") for v in wq.values()) else
            "forecasts the flow better than the simple models" if all(v == "lstm" for v in wq.values()) else
            "forecasts the flow worse than a simple model" if all(v == "simple" for v in wq.values()) else
            "forecasts the flow better than the simple models only at part of the lead times")
    without = (f"without one, only the average of its {_word(n_seeds)} seeds is ahead" if all(v == "lstm_avg" for v in w.values()) else
               "without one it is better" if all(v == "lstm" for v in w.values()) else
               "without one there is no clear difference either" if all(v in ("tie", "lstm_med") for v in w.values()) else
               "without one the result differs with the lead time")
    warn = ("Its warnings of an onset cannot be trusted yet" if on_verdicts == {"NOT YET"} else
            "Its warnings of an onset could be trusted" if on_verdicts == {"GO"} else "Its warnings of an onset could be trusted only with caution")
    head = f"With a river gauge, the LSTM {flow}; {without}. {warn}."

    # ---- big numbers
    c_first = cmp_("wq", first)
    big = [(f"{n2(c_first['lstm_nse_median'])} vs {n2(c_first['baseline_nse_median'])}",
            f"NSE {LEAD_WORD[first]} with the gauge, LSTM vs best simple model: {PHRASE[wq[first]].replace('there is ', '')}"),
           (pct(v_on[first]["pod"]), f"of the {count(n_on)} onsets warned {LEAD_WORD[first]} with the river gauge: {v_on[first]['pooled']}")]
    if tf:
        r = tf["gauge"][first]
        still = f"; still {tv['verdict']}" if tv and tv["verdict"] != "GO" else ""
        big.append((pct(r["onset_warned_with_nldas"]),
                    f"of onsets warned with rain up to the evening of the issue day (control {pct(r['onset_warned_control'])}): "
                    f"{pct(r['onset_share_of_gap_closed'])} of the gap{still}"))
    big += new["big"]
    big.append((pct(up_on), f"of onsets warned {LEAD_WORD[first]} given a perfect weather forecast, an upper bound"))

    # ---- what was done and found, one row per step of PLAN.md (each finding on one line, with its number) ---------------
    rows = []
    add = lambda step, done, found, where: rows.append([f"<b>{step}</b>", done, found, where])
    not_run = lambda step, where: rows.append([f"<b>{step}</b>", "&ndash;", "not run", where])

    if "data_audit" in R:
        D, HF = R["data_audit"], B["high_flow"]
        mean_test = sum(v["share_test_days_above"] for v in HF.values()) / len(HF)
        add("Data audit", f"{count(D['n_candidates'])} candidates, {len(D['selected'])} chosen",
            f"{count(D['n_complete'])} of {count(D['n_candidates'])} candidates complete; test years: {pct(mean_test, 1)} high-flow days, not {pct(1 - HIGH_Q)}", "1")
    else:
        not_run("Data audit", "1")

    nb = {k: B["best_by_validation"][lk(k)]["without_weather_forecast"]["test"]["nse_median"] for k in LEADS}
    add("Baselines", "four kinds of simple model", f"NSE to beat with the river gauge: {three([nb[k] for k in LEADS])}", "2")

    if "lstm_single" in R:
        S1 = R["lstm_single"]
        calls = [S1["comparison"][inp]["verdict"][lk(k)] for inp in ("w", "wq") for k in LEADS]
        n_ahead = sum(c["nse_call"].startswith("LSTM ahead") for c in calls)
        c1q = S1["comparison"]["wq"]["verdict"][lk(first)]
        behind = (f"; behind with the gauge ({n2(c1q['lstm_nse'])} against {n2(c1q['best_baseline_nse'])})"
                  if c1q["nse_call"].startswith("baseline ahead") else "")
        river = S1["basin"]["name"].split(" AT ")[0].title() if " AT " in S1["basin"]["name"] else S1["basin"]["short"]
        add("LSTM, one river", river, f"ahead in {n_ahead if n_ahead else 'none'} of {len(calls)} comparisons{behind}", "3")
    else:
        not_run("LSTM, one river", "3")

    add("LSTM, all twelve", f"one model, {_word(n_seeds)} seeds",
        f"with the river gauge no clear difference ({n2(lstm_nse[0])} against {n2(base_nse[0])} {LEAD_WORD[first]})"
        if all(v == "tie" for v in wq.values()) else f"with the river gauge: {_say(wq)}", "4.1&ndash;4.3")

    if lf:
        add("Unseen river", "each basin held out",
            f"{unseen_sentence(R, short=True)[4:]} (lower in {lf['n_lower']} of {lf['n']}); median {n2(lf['seen'])} to {n2(lf['unseen'])}"
            if lf["loss"] >= 0 else unseen_sentence(R), "4.4")                    # [4:]: without 'the ', as the other rows are worded
    else:
        not_run("Unseen river", "4.4")

    if tf:
        r = tf["gauge"][first]
        nse_word = {"clear": "NSE up too", "mixed": "NSE mixed", "none": "NSE: no clear difference"}[nse_reading(r, tf["n"])]
        on_word = (f"onsets warned {pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} (control {pct(r['onset_warned_control'])})"
                   + ("" if onset_clear(r) else ", not clear of the seeds"))
        add("Rain timing", "later rain as an input", f"{on_word}; {nse_word} (higher in {r['basins_better']} of {tf['n']} basins)", "4.5")
    else:
        not_run("Rain timing", "4.5")

    rows += new["rows"]

    add("Flood warnings", "fixed criteria",
        f"all high-flow days {LEAD_WORD[first]}: {v_all['pooled']}; "
        + (f"onsets: {next(iter(on_verdicts))} ({pct(v_on[first]['pod'])} warned)" if len(on_verdicts) == 1 else f"onsets: {on_where}"), "5")

    if "basin_gis" in R:
        G = R["basin_gis"]["comparison_with_camels"]
        ranks = [G[k]["rank_correlation"] for k in ("area_vs_gages2", "elev_mean", "slope_mean", "forest_share")]
        add("Maps and GIS", "elevation, land cover",
            (f"basin order nearly as in CAMELS (rank correlation {n2(min(ranks))} to {n2(max(ranks))})" if min(ranks) >= 0.9 else
             f"basin order differs from CAMELS (rank correlation {n2(min(ranks))} to {n2(max(ranks))})"), "6.1")
    else:
        not_run("Maps and GIS", "6.1")

    sat_step = "Satellite rain and<br>reanalysis inputs"
    if c3:
        cause = (": probably a time-stamp effect" if likely else "")
        add(sat_step, "satellite rain; soil wetness,<br>vapour (model output)",
            (f"satellite rain helps with the gauge ({n2(s1['base'])} to {n2(s1['variant'])}){cause}"
             if clear else "satellite rain: no clear change with the river gauge"), "6.2")
    else:
        not_run(sat_step, "6.2")

    acc = lo = hi = None                                                        # the Transformer clause; None while the comparison has not run
    if "model_comparison" in R:
        M = R["model_comparison"]
        t, ratio = M["comparison"]["test"]["skill"], M["cost_of_skill"]["transformer_over_lstm"]
        lo, hi = sorted((ratio["seconds_per_model"], ratio["seconds_per_update"]))
        acc = ("no clear difference in accuracy" if t["seed_ranges_overlap"] or abs(t["transformer_minus_lstm"]) <= t["largest_seed_spread"]
               else f"the {'Transformer' if t['transformer_minus_lstm'] > 0 else 'LSTM'} is more accurate")
        add("Transformer, cost", "same data and trainer", f"{acc}; {lo:.0f} to {hi:.0f} times the training time", "7")
    else:
        not_run("Transformer, cost", "7")

    # ---- the four milestones of PLAN.md
    figures_done = all((FIG / f).exists() for f in ("flood_skill.png", "basin_maps.png")) and "flood_analysis" in R and "basin_gis" in R
    slides = sorted(p.name for p in ROOT.glob("*.pptx"))
    miles = [("M1", "the number to beat is known", "met"),
             ("M2", "the LSTM beats the baselines, or we learn why not",
              f"{m2_gauge}; {re.sub(r' [(][^)]*[)]', '', m2_weather)}; why: sections 4.3 to 4.5"),      # the basin counts stand in the answer box
             ("M3", "flood-warning table, maps, figures", "met" if figures_done else "not met yet: a figure or a results file is missing"),
             ("M4", "report and slides drafted", "report built" + (", slides built" if slides else ", slides not built yet"))]

    # ---- the finding first: the lead line and three numbered findings. Used while the stored numbers support the finding (story);
    #      otherwise page 1 is built as before, from head, a, b, c, d, big and rows
    gauge_tie = all(v in ("tie", "lstm_med") for v in wq.values())
    story = bool(limit and gauge_tie and acc != "the Transformer is more accurate")
    K = key_figure(R)
    test_years = f"{A['split']['test'][0][:4]}&ndash;{A['split']['test'][1][:4]}"
    drawn_leads = _leads([first] + [int(x) for x in (K or {}).get("leads_drawn_below") or []])         # the key figure may draw 2 and 3 days below the 1 day
    lead_line = f"{SUPPORT} ({'key figure below; ' if K else ''}test years {test_years}, {n} basins, {drawn_leads if K else LEAD_WORD[first]})."
    f1 = (f"With past weather + river gauge, the LSTM&rsquo;s median NSE is {n2(lstm_nse[0])} against {n2(base_nse[0])} for the best simple model: "
          f"{_say(wq).replace('there is ', '')}, so milestone 2 of the plan, &lsquo;the LSTM beats the baselines&rsquo;, is "
          f"<b>{m2_gauge.split(' (')[0]}</b> (section 4.2)."
          + (f" A Transformer in place of the LSTM: {acc}, at {lo:.0f} to {hi:.0f} times the training time (section 7)." if acc else "")
          + (" " + new["yard"] if new["yard"] else ""))
    f2 = ((f"The river gauge adds {signed(g1)} NSE to the LSTM." if g1 > 0 else f"The river gauge does not lift the LSTM&rsquo;s NSE ({signed(g1)}).")
          + f" A perfect weather forecast, an upper bound that cannot be had in practice, would lift the onsets warned from "
            f"{pct(LV['wq'][lk(first)]['onset_pod'])} to {pct(up_on)}")
    if tf:
        r = tf["gauge"][first]
        nse_bit = {"clear": f"in NSE the gain is clear as well (higher in {r['basins_better']} of {tf['n']} basins)",
                   "mixed": f"in NSE that result is mixed (higher in only {r['basins_better']} of {tf['n']} basins)",
                   "none": "in NSE there is no clear difference"}[nse_reading(r, tf["n"])]
        if onset_clear(r):
            f2 += (f". Rain stamped up to the evening of the issue day closes {share_word(r['onset_share_of_gap_closed'])} of that gap "
                   f"({pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])}; control {pct(r['onset_warned_control'])}); {nse_bit}"
                   + ("" if new["rungs"] else ". The rest needs later rain, most of it after the forecast is issued"))
        else:
            f2 += (f". Rain stamped up to the evening of the issue day takes them from {pct(r['onset_warned_base'])} to "
                   f"{pct(r['onset_warned_with_nldas'])} (control {pct(r['onset_warned_control'])}), but the single seeds overlap: no clear gain; {nse_bit}")
        f2 += ((f"; " + new["rungs"][0].lower() + new["rungs"][1:-1] + f" (sections 4.3, {TIMING_REF.split()[-1]}, 4.6 and 4.8).") if new["rungs"] else f" (sections 4.3 and {TIMING_REF.split()[-1]}).")
    else:
        f2 += " (section 4.3)." + (" " + new["rungs"] + " (sections 4.6 and 4.8)." if new["rungs"] else "")
    head3 = warn.replace("Its warnings", "So the warnings" if on_verdicts == {"NOT YET"} else "The warnings")
    on3 = (f"{tag(next(iter(on_verdicts)))} for onsets, the warnings that matter, at every lead time" if len(on_verdicts) == 1 else
           f"for onsets, the warnings that matter, {on_where}")
    f3 = (f"By criteria fixed before the analysis: {tag(v_all['pooled'])} for all high-flow days {LEAD_WORD[first]} with the river gauge"
          + (f", the same verdict as {QUOTED}" if pers == v_all["pooled"] else "")
          + f"; {on3}: {LEAD_WORD[first]} {pct(v_on[first]['pod'])} of the {count(n_on)} onsets are warned" + new["verdict"])
    unseen = (f"A river never seen in training: {unseen_sentence(R)}" + new["rivers"]) if lf else ""     # the last sentence of finding 3: the first to go if page 1 runs over
    finds = [("The kind of model changed little.", f1), ("The information changed a lot.", f2), (head3 + ".", f3)]

    built = datetime.date.today()
    # c3 (satellite rain) is not repeated in the answer box: its row in the table of steps says it
    return dict(a=[a1, a2, a3], b=[x for x in (b1, new["b2"]) if x], c=[x for x in (c1, c2, new["c4"]) if x], d=[d1], d_head=d_head, big=big, rows=rows, miles=miles, m2=m2_gauge,
                built=f"{built.day} {built:%B %Y}", n=n, n_days=n_days, n_on=n_on, verdict=V, first=first, head=head, who=who(R), who_short=who(R, short=True),
                story=story, lead=lead_line, finds=finds, unseen=unseen, d_bare=[d1_bare], key=K, test_years=test_years,
                limit=limit, gauge_tie=gauge_tie, tie=tie.strip(), more=more, gauge_adds=gauge_adds, all_gain=bool(all_gain), warn=warn,
                acc=acc, t_lo=lo, t_hi=hi, cls=cls, n_seeds=n_seeds, m2_weather=re.sub(r" [(][^)]*[)]", "", m2_weather),
                sat_likely=bool(likely and clear), new=new)


def dates(R):
    """(day of the data audit and the choice of the basins, day of everything else), from results/engine_review.json: its key 'dates', or -
    in an older file without it - the day of the review for both."""
    ER = R["engine_review"]
    d = ER.get("dates") or {}
    rest = d.get("everything_else") or ER["when"][:10]
    return _day(d.get("data_audit_and_basin_selection") or rest), _day(rest)


def run_day(R):
    """The days on which the code was written and the experiments were run, as words: '2 and 3 October 2026' or '3 October 2026'."""
    audit, rest = dates(R)
    if audit == rest:
        return rest
    a, r = audit.split(" ", 1), rest.split(" ", 1)
    return f"{a[0]} and {rest}" if a[1] == r[1] else f"{audit} and {rest}"


def revised(R):
    """The day the builders of this report were last changed, as words, when that is later than the day the results file gives for 'everything
    else' (read from the file system, like 'what exists now' in section 9); otherwise an empty string."""
    files = list((ROOT / "code" / "report_sections").glob("*.py")) + [ROOT / "code" / "build_report.py", ROOT / "code" / "report_common.py"]
    last = datetime.date.fromtimestamp(max(p.stat().st_mtime for p in files if p.exists()))
    d = (R["engine_review"].get("dates") or {}).get("everything_else") or R["engine_review"]["when"][:10]
    return _day(last.isoformat()) if last > datetime.date.fromisoformat(d[:10]) else ""


def who(R, short=False):
    """Who wrote, ran and checked what - stated the same way on page 1, in section 9 and in short_report.md.
    short: the form for page 1, which leaves out the stored 'late additions' (the analyses of 4 October 2026 that needed no training) and the rerun of the
    day-window check (section 8 and short_report.md have both) - the dates, the reviews (results/review_*.json) and the team's open checks stay.
    Dates and reviewers: results/engine_review.json. The state of the team members' own checks: team.json, read through code/team.py
    (while none is done the sentence is the one the report has always printed). What was added on the last day from the stored results:
    the sentence engine_review.json keeps under 'late_additions' (left out, with one line on stderr, if the key is missing)."""
    ER = R["engine_review"]
    audit, rest = dates(R)
    nw = new_work(R)                                                                              # the experiments of sections 4.6 to 4.8 and their reviews
    if short:                                                                                     # page 1: the days only; section 9 has what was done on each
        when = f"on {run_day(R)}" + (f" and, for sections 4.6 to 4.8, on {nw['days']}" if nw else "")
    else:
        when = (f"the data audit and the choice of the basins on {audit}, everything else on {rest}" if audit != rest else f"on {rest}")
        if nw:
            when += f", the experiments of sections 4.6 to 4.8 on {nw['days']}"
    rev = revised(R)
    late = str((ER.get("dates") or {}).get("day_window_check_rerun") or "")                       # one check was reviewed and rerun a day later
    late = _day(late) if late[:10] > str((ER.get("dates") or {}).get("everything_else") or ER["when"])[:10] else ""
    if short:                                                                                     # page 1: the rerun of one check is told in section 8
        when += f" (this text last revised on {rev}, also by Claude Code)" if rev else ""
    elif rev and late == rev:
        when += f" (the day-window check of section 4.5 was reviewed and rerun, and the text of this report last revised, on {rev}, also by Claude Code)"
    else:
        bits = ([f"the day-window check of section 4.5 was reviewed and rerun on {late}"] if late else []) + ([f"the text of this report was last revised on {rev}, also by Claude Code"] if rev else [])
        when += f" ({'; '.join(bits)})" if bits else ""
    if team.checks_state()["n_done"] == 0:
        own = ("the team members&rsquo; own checks are still to be done (section 9). " if short else
               "the team members&rsquo; own checks that the plan assigns are still to be done (section 9). ")
    else:
        own = _esc(team.checks_sentence())
        own += ("" if own.endswith("(section 9)") else " (section 9)") + ". "
    added = str(ER.get("late_additions") or "").strip()
    if added:
        added = added.replace("Each new script was checked by an independent AI reviewer agent", "Each of these scripts was checked, as written on 4 October 2026, by an independent AI reviewer agent")
        added = _esc(added) + ("" if added.endswith(".") else ".") + " "
        if short:
            added = ""
    else:
        note("s0_summary: results/engine_review.json has no key 'late_additions': 'Who did what' does not mention the analyses added on the last day")
    return (f"All code was written and all experiments were run by Claude Code, the team&rsquo;s coding assistant, under the team lead&rsquo;s direction: "
            f"{when}. {review_sentence()}; {own}{added}"
            "&lsquo;We&rsquo; means the project, never a check made by a team member.")


QUESTION = ("Given the weather of the past year, can an AI model predict a river&rsquo;s flow one to three days ahead, and warn of high-flow days "
            "before they happen?")


def _reading_guide(n, R=None, brief=False):
    """The reading guide: the rivers and the three periods, then the terms. R: with the verdict criteria as its last sentence.
    brief: without NSE, high-flow day and onset - for page 1 under the key figure, whose caption has just explained them."""
    terms = ("" if brief else
             f" <i>NSE</i>: 1 is a perfect forecast, 0 is no better than always forecasting the average flow. <i>High-flow day</i>: flow above the level "
             f"exceeded on {pct(1 - HIGH_Q)} of days in the training years. <i>Onset</i>: a high-flow day forecast while the river was still below that level.")
    return (f"Tested on {n} south-eastern US rivers, each with its <i>basin</i> (the land whose rain drains to the river&rsquo;s gauge): training years "
            f"({_years('train')}), validation years ({_years('val')}), test years ({_years('test')}); all scores are for the test years.{terms} "
            f"<i>LSTM</i>: a neural network that reads the days in order and keeps a memory."
            + (f" {criteria_line(R)}" if R and "flood_analysis" in R else ""))


def reading_guide(R):
    """The reading guide without the criteria line, for section 8 when page 1 has no room for it (page1_plan)."""
    return _reading_guide(len(R["lstm_all"]["basins"]))


# ---- the key figure of the main finding -----------------------------------------------------------------------------------
def key_figure(R):
    """results/key_figure.json (written by 11_key_figure.py) when the figure it describes can be shown: one of its renderings exists in
    figures/ and the results file is not a smoke test. Otherwise None, with one line on stderr; page 1 then shows the four big numbers in
    its place."""
    K = R.get("key_figure")
    if ((FIG / KEY_FIG).exists() or (FIG / KEY_FIG_PLAIN).exists()) and isinstance(K, dict) and not K.get("quick_mode") and K.get("levels"):
        return K
    note(f"s0_summary: figures/{KEY_FIG} or results/key_figure.json is missing (code/11_key_figure.py writes both): page 1 shows the four big "
         f"numbers in place of the key figure, and README.md and short_report.md show no figure")
    return None


def key_caption(R, K, plain=True, short=True, terms=True):
    """The caption of the key figure. What the figure draws is read from results/key_figure.json: the lead time, the test years, the number
    of basins, whether every other kind of model is drawn only where it reads the same inputs as the LSTM, whether a Transformer and a
    control are drawn, and the footnote printed inside the full rendering.
    plain: the figure is the rendering without title, subtitle and footnote (page 1), so the caption carries every word the reader needs:
    what an onset and the NSE are, the marks, the control, the upper bound, the mixed NSE result of the rain-timing test (read from
    results/rain_timing.json) and the verdict criteria - page 1 stands alone. Not plain (the rendering with its own footnote): what that
    footnote already says is not said a second time.
    short: the criteria without 'by criteria fixed before the analysis' (finding 3, just above the figure on page 1, says so).
    terms: with what an onset, the high-flow level and the NSE are (False where the full reading guide follows the caption)."""
    k = K["lead_drawn"]
    note_in = "" if plain else str((K.get("figure") or {}).get("footnote") or "")   # the words printed under the panels, inside the figure
    below = [int(x) for x in K.get("leads_drawn_below") or []]                       # a lower block (2 and 3 days ahead): the figure's own panel titles say so
    blocks = [lv for blk in (K.get("far") or {}).values() if isinstance(blk, dict) for lv in blk.get("levels") or []]
    others = [o for lv in K["levels"] for o in (lv.get("others") or {}).values() if o.get("kind") != "persistence"]
    drawn = [o for o in others if o.get("drawn")]
    same = bool(others) and all(bool(o.get("drawn")) == bool(o.get("same_inputs")) for o in others)
    years = K.get("test_years") or []
    span = f" {years[0]}&ndash;{years[1]}" if len(years) == 2 else ""
    if below:
        where = "" if "est years" in note_in else f"Test years{span}, {K['n_basins']} basins. "
    else:
        where = "" if LEAD_WORD[k] in note_in and "basins" in note_in else f"{LEAD_WORD[k].capitalize()}, test years{span}, {K['n_basins']} basins. "
    level = f"the flow exceeded on {pct(1 - HIGH_Q)} of the days of the training years"
    onset = (f"The high-flow level is {level}. " if "Onset:" in note_in else
             f"An <i>onset</i> is a high-flow day forecast while the river was still below the high-flow level, {level}. ")
    star = (" (* the Transformer comes from its own experiment, section 7)"
            if any(o.get("kind") == "transformer" for o in drawn) and "own experiment" not in note_in else "")
    legend = ((K.get("figure") or {}).get("drawn_words") or {}).get("legend") or {}           # what the figure's own legend already says (an older file has none)
    marks = ("Large marks: the LSTM; small marks: the other kinds of model" + (", shown only where they read the same inputs" if same else "") + f"{star}. "
             if drawn else "")
    if legend.get("lstm") and legend.get("linreg") and drawn:                         # the legend names the LSTM and the others: the caption says only where they are drawn
        marks = ("Small marks: the other kinds of model" + (", only where they read the same inputs as the LSTM" if same else "") + f"{star}. ")
    C = K.get("control") or {}
    label = str(C.get("label") or "")
    control = (f"{label[0].upper() + label[1:]}. " if label and "ontrol" not in note_in else "")
    is_bound = "upper bound" in str(K["levels"][-1].get("label", ""))
    bound = ("Hatched: an upper bound that cannot be had in practice. "
             if is_bound and "upper bound that cannot be had in practice" not in note_in and not legend.get("bound") else "")
    mixed = ""
    tf = timing_facts(R)
    mixed_rows = ((K.get("figure") or {}).get("drawn_words") or {}).get("mixed_rows")        # the rows beside which the figure prints 'mixed'
    if below and mixed_rows and "mixed" not in note_in:
        mixed = "<i>mixed</i> beside an NSE: the median rises, but not clearly (section 4.2). "
    elif tf and any(lv.get("key") == "wqn" for lv in K["levels"]) and "mixed" not in note_in:
        r = tf["gauge"][k]
        mixed = {"mixed": f"With rain up to the evening the NSE result is mixed (higher in only {r['basins_better']} of {tf['n']} basins). ",
                 "none": "With rain up to the evening the NSE shows no clear difference. "}.get(nse_reading(r, tf["n"]), "")
    aorc = next((lv for lv in K["levels"] if lv.get("key") == "wqa"), None)
    real_time = (aorc or {}).get("real_time") or {}
    extra = ""
    if real_time and "diamond" not in note_in:                                       # the check of 8 October: a real-time product in place of AORC
        extra += (f"Hollow diamonds: {real_time.get('product', 'a real-time product')} radar, a real-time product, in place of AORC (right), "
                  f"without its last hour (left). ")
    if any(lv.get("key") == "wqaF_sub" for lv in blocks) and "GEFS" not in note_in:  # the forecast rungs of 8 October
        extra += ("GEFS: archived forecast rain, one ensemble member, put in place of the perfect rain that the model was trained on; below, the light part of a bar is the rain up to "
                  "midnight, the dark part what the forecast adds. ")
    explained = (f"{onset}<i>NSE</i>: 1 is a perfect forecast, 0 is no better than always forecasting the average flow. " if terms else "")
    return f"<b>Key figure. The main finding at a glance.</b> {where}{explained}{marks}{control}{bound}{extra}{mixed}{criteria_line(R, short=short)}"


def _key_figure_html(R, K):
    """The key figure on page 1 at full text width: the rendering without its own title, subtitle and footnote, with the caption that carries
    the words; while that file is missing, the full rendering with the shorter caption (one line on stderr)."""
    if (FIG / KEY_FIG_PLAIN).exists():
        name, plain = KEY_FIG_PLAIN, True
    else:
        name, plain = KEY_FIG, False
        note(f"s0_summary: figures/{KEY_FIG_PLAIN} is missing (code/11_key_figure.py writes it): page 1 shows figures/{KEY_FIG}, which carries its "
             f"own title and footnote and is taller")
    return fig(name, key_caption(R, K, plain=plain)).replace("<figure>", '<figure class="key">', 1)


def _step_names(f):
    """The names of the steps of the plan, as the table of steps lists them, in plain words."""
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r[0])).strip() for r in f["rows"]]


def _byline(f):
    """The line under the title: the names of the team and, when team.json gives them, the course and the university (code/team.py)."""
    course = "".join(f"{_esc(x)} &middot; " for x in team.course_line("\n").split("\n") if x)
    return (f"AI course group project &middot; {course}final report &middot; built on {f['built']} &middot; team: {_esc(team.names_line())} &middot; "
            f"coding assistant: Claude Code")


def _miles(f):
    return "; ".join(f"<b>{m}</b> ({what}): {status}" for m, what, status in f["miles"])


def _page1(f, R, plan):
    """Page 1 with the finding first. plan: which optional parts it holds (one of PAGE1_PLANS)."""
    K = f["key"]
    (h1, t1), (h2, t2), (h3, t3) = f["finds"]
    if f["new"]["yard"] and f["new"]["yard_p1"]:                                      # page 1 says it without the points of the simple models (4.6 and short_report.md have them)
        t1 = t1.replace(f["new"]["yard"], f["new"]["yard_p1"], 1)
    if f["new"]["verdict"] and f["new"]["verdict_p1"]:                                # ... and without the false warnings of the forecast (the paragraph 'What it took' has the share)
        t3 = t3.replace(f["new"]["verdict"], f["new"]["verdict_p1"], 1)
    if not plan["miles"]:                                                            # without the line of milestones, finding 1 states milestone 2 for both forecasts
        t1 = t1.replace("</b> (section 4.2).", f"</b>; {f['m2_weather']} (section 4.2).", 1)
    unseen = plan["unseen"] and f["unseen"]
    lobo = LOBO_REF.split()[-1]
    where = ["5"] + ([lobo] if unseen else []) + (["4.6 to 4.8"] if f["new"]["verdict"] else [])      # the sections the finding rests on
    if plan["take"]:
        refs = ("sections " + _and(where)) if len(where) > 1 else "section 5"
    else:                                                                            # without 'What it would take', the finding points to the next steps
        refs = ("sections " + _and(where + ["8"])) if len(where) > 1 else "section 5; next steps: section 8"
    t3 += (f". {f['unseen']}" if unseen else "") + f" ({refs})."
    take = f"<p><b>{f['d_head']}</b> {' '.join(f['d_bare'] if unseen else f['d'])}</p>" if plan["take"] else ""
    figure = _key_figure_html(R, K) if K else kpis(f["big"])
    guide = (f'<p class="guide"><b>How to read this page.</b> {_reading_guide(f["n"], None if K else R, brief=bool(K))}</p>' if plan["guide"] else "")
    steps = (f'<p class="guide"><b>What was done.</b> {"; ".join(_step_names(f))}. What each step found: section 8.</p>' if plan["steps"] else "")
    miles = f'<p class="small"><b>Milestones of the plan.</b> {_miles(f)}.</p>' if plan["miles"] else ""
    return f"""{LAYOUT}{STYLE}<div class="s0 story">
<h1>Predicting river floods with AI</h1>
<p class="tagline">What we found: {FINDING[0].lower() + FINDING[1:]}.</p>
<p class="sub">{_byline(f)}</p>
<div class="verdict"><p><b>The question.</b> {QUESTION}</p>
<p><b>What we found.</b> {f['lead']}</p>
<p><b>1. {h1}</b> {t1}</p>
<p><b>2. {h2}</b> {t2}</p>
<p><b>3. {h3}</b> {t3}</p>{take}</div>
{figure}
{guide}
{steps}
{miles}
<p class="who"><b>Who did what.</b> {f['who_short']}</p>
</div>"""


def _page1_before(f, R, big=True):
    """Page 1 as it was before 4 October 2026: the answer box with four paragraphs, the big numbers, the table of steps. Built when the
    stored numbers no longer support the finding (facts()['story'] is False). big: with the four big numbers (they leave if the page runs
    over: the text and the table hold the same numbers)."""
    answer = "".join(f"<p><b>{head}</b> {' '.join(f[key])}</p>" for head, key in
                     (("The forecasts.", "a"), ("The warnings.", "b"), ("Why.", "c"), (f["d_head"], "d")))
    answer = answer.replace("<b>The warnings.</b> The warnings: ", "<b>The warnings.</b> ")
    steps = table(["Step of the plan", "What was done", "What was found (every finding is read from the results files)", "Section"], f["rows"],
                  cls="num s0t", widths=["16%", "20%", "58%", "6%"])
    return f"""{LAYOUT}{STYLE}<div class="s0">
<h1>Predicting river floods with AI</h1>
<p class="sub">{_byline(f)}</p>
<div class="ask"><b>The question.</b> {QUESTION} {_reading_guide(f['n'], R)}</div>
<div class="verdict"><p><b>The answer.</b> {f['head']}</p>{answer}</div>
{kpis(f['big']) if big else ''}
{steps}
<p class="small"><b>Milestones of the plan.</b> {_miles(f)}.</p>
<p class="who"><b>Who did what.</b> {f['who']}</p>
</div>"""


_PLAN = {}


def page1_plan(R, T):
    """Which optional parts page 1 holds: one of PAGE1_PLANS, with 'story' (the finding-first page is built at all) and 'big' (the page as
    before, with its four big numbers) added.

    The page must be ONE page, and its length depends on the height of the key figure and on sentences that change with the results and
    with team.json. So it is not guessed: page 1 alone is printed to a temporary PDF with the same browser and the same page rules as the
    report, and the first plan of PAGE1_PLANS whose print has one page is taken (two or three test prints of a second or two each; the
    result is kept for the rest of the build). Section 8 asks this function what it has to carry in place of page 1. Without a browser no
    PDF is made at all; the plan is then the one of PAGE1_HINT."""
    if "plan" in _PLAN:
        return _PLAN["plan"]
    f = facts(R, T)
    tried = {}

    def fits(i, folder):
        if i not in tried:
            src = Path(folder) / f"page1_{i}.html"
            body = _page1(f, R, PAGE1_PLANS[i]) if f["story"] else _page1_before(f, R, big=not i)
            src.write_text(page("page 1, test print", body, css_extra=PRINT_CSS), encoding="utf-8")
            try:
                printed = print_pdf(src, src.with_suffix(".pdf"), quiet=True)
            except Exception as e:                                                   # a browser that fails here fails for the report too, and says so there
                note(f"s0_summary: the test print of page 1 failed ({type(e).__name__}); page 1 is built without the test")
                printed = False
            tried[i] = (pdf_pages(src.with_suffix(".pdf")) == 1) if printed else None
        return tried[i]

    if not f["story"]:                                       # the page as before: it holds the guide, the table of steps and the milestones itself
        with tempfile.TemporaryDirectory() as folder:
            big = fits(0, folder) is not False               # with the four big numbers, unless that print runs over one page
            if not big and fits(1, folder) is False:
                note("s0_summary: page 1 (built as before 4 October 2026) does not fit on one page even without the four big numbers: look at "
                     "pages 1 and 2 of the PDF")
        _PLAN["plan"] = dict(dict.fromkeys(PAGE1_PARTS, True), story=False, big=big)
        return _PLAN["plan"]
    last = len(PAGE1_PLANS) - 1
    i = min(max(PAGE1_HINT, 0), last)
    with tempfile.TemporaryDirectory() as folder:
        if fits(i, folder) is not None:                                              # None: no browser, nothing is printed anyway
            while fits(i, folder) and i > 0 and fits(i - 1, folder):                 # room for more: the next fuller page
                i -= 1
            while not fits(i, folder) and i < last:                                  # over one page: the next barer page
                i += 1
            if not fits(i, folder):
                note("s0_summary: page 1 does not fit on one page even in its barest form (no list of steps, no reading guide, no unseen-river "
                     "sentence, no 'What it would take'): look at pages 1 and 2 of the PDF - the key figure may have become taller")
    _PLAN["plan"] = dict(PAGE1_PLANS[i], story=True, big=False)
    return _PLAN["plan"]


def html(R, T):
    f, plan = facts(R, T), page1_plan(R, T)
    return _page1(f, R, plan) if f["story"] else _page1_before(f, R, big=plan["big"])


REPRODUCE = ("pip install -r requirements.txt   # in the project folder, Python {py}; torch and numpy are pinned",
             "./run_all.sh                      # every step in order; FLOOD_QUICK=1 ./run_all.sh is a fast smoke test",
             "python code/build_report.py       # Flood_Report.html, Flood_Report.pdf and this file, from results/ and figures/ only")


def md(R, T):
    """The readable one-page version (short_report.md): the finding, the three findings with their numbers, the key figure, the evidence in
    more detail, the key numbers, the verdict table, how to reproduce. It has room for everything, so no part of PAGE1_PARTS is left out."""
    f = facts(R, T)
    P = _plain
    byline = P(_byline(f)).replace(" · ", " - ")
    shown = bool(f["key"]) and (FIG / KEY_FIG).exists()                              # the rendering with its own title and footnote
    out = ["# Predicting river floods with AI - short report", ""]
    if f["story"]:
        (h1, t1), (h2, t2), (h3, t3) = f["finds"]
        where = ["5"] + ([LOBO_REF.split()[-1]] if f["unseen"] else []) + (["4.6 to 4.8"] if f["new"]["verdict"] else [])
        t3 += (f". {f['unseen']} (sections {_and(where)})." if f["unseen"] else f" (section{'s' if len(where) > 1 else ''} {_and(where)}).")
        lead = P(f["lead"]).replace("key figure below", "key figure below the three findings") if shown else P(f["lead"]).replace("key figure below; ", "")
        out += [f"**What we found: {P(FINDING[0].lower() + FINDING[1:])}.**", "", byline, "", f"**The question.** {P(QUESTION)}", "",
                "## What we found", "", lead, "",
                f"1. **{P(h1)}** {P(t1)}", f"2. **{P(h2)}** {P(t2)}", f"3. **{P(h3)}** {P(t3)}", "",
                f"**{P(f['d_head'])}** {P(' '.join(f['d_bare'] if f['unseen'] else f['d']))}", ""]
        if shown:
            out += [f"![Key figure: the main finding at a glance](figures/{KEY_FIG})", "",
                    P(key_caption(R, f["key"], plain=False, short=False, terms=False)), ""]
        out += [f"**How to read this page.** {P(_reading_guide(f['n'], None if shown else R))}", "", "## The evidence in more detail", ""]
        for head, key in (("The forecasts", "a"), ("The warnings", "b"), ("Why", "c")):
            out += [f"- **{head}.** " + P(" ".join(f[key])).replace("The warnings: ", "", 1)]
    else:
        out += [byline, "", f"**The question.** {P(QUESTION)}", "", P(_reading_guide(f['n'], R)), "", "## The answer", "", P(f["head"]), ""]
        for head, key in (("The forecasts", "a"), ("The warnings", "b"), ("Why", "c"), (f["d_head"].rstrip("."), "d")):
            out += [f"- **{head}.** " + P(" ".join(f[key])).replace("The warnings: ", "", 1)]
    out += ["", "## Key numbers", "", "| Number | What it is |", "|---|---|"] + [f"| {P(v)} | {P(l)} |" for v, l in f["big"]]
    out += ["", "## What we did and what we found", "", "| Step of the plan | What was done | What was found | Section |", "|---|---|---|---|"]
    out += ["| " + " | ".join(P(c) for c in r) + " |" for r in f["rows"]]
    V = f["verdict"]
    names = {"lstm_w": "Past weather only", "lstm_wq": "Past weather + river gauge", "lstm_wqf": "Past weather + river gauge + perfect weather forecast (upper bound)"}
    seeds_same = lambda c: sum(v == c["pooled"] for v in c.get("single_seeds", {}).values())
    star = lambda c: (f" [{seeds_same(c)} of {len(c['single_seeds'])} single seeds]" if c.get("single_seeds") and seeds_same(c) < len(c["single_seeds"]) else "")
    cell = lambda c: f"{c['pooled']}{star(c)} ({100 * c['pod']:.0f}% warned, {100 * c['far']:.0f}% false)"
    out += ["", "## The verdict (test years, 12 basins together)", "",
            "GO / CAUTION / NOT YET by criteria fixed in advance, for the forecast averaged over the seeds; in brackets: share warned, false warnings. "
            "Where not every single seed gets the same verdict, square brackets say how many do.", "",
            "| LSTM given | " + " | ".join(f"{kind}, {LEAD_WORD[k]}" for kind in ("All high-flow days", "Onsets") for k in LEADS) + " |",
            "|---|" + "---|" * (2 * len(LEADS))]
    out += [f"| {names[s]} | " + " | ".join(cell(V[s][lk(k)][kind]) for kind in ("all", "onset") for k in LEADS) + " |" for s in names if s in V]
    out += ["", "## Milestones of the plan", ""] + [f"- **{m}** ({P(what)}): {P(status)}" for m, what, status in f["miles"]]
    out += ["", "## How to reproduce", "", "```bash"] + [line.format(py=f"{sys.version_info.major}.{sys.version_info.minor}") for line in REPRODUCE]
    out += ["```", "", "## Who did what", "", P(f["who"]), ""]
    return "\n".join(out)


# ---- README.md: the block 'What we found' ---------------------------------------------------------------------------------
README_START = "<!-- FINDINGS: written by code/build_report.py from results/ - do not edit between these lines -->"
README_END = "<!-- END OF FINDINGS -->"


def readme_block(R, T):
    """The block '## What we found' of README.md as Markdown: the finding and its supporting line, the key figure (the rendering with its own
    title and footnote, if the file exists), the four big numbers of page 1, then - plainly, second - the status of milestone 2 and of the
    onset warnings. build_report.py writes it between README_START and README_END."""
    f = facts(R, T)
    P = _plain
    out = ["## What we found", ""]
    out += [f"**{FINDING}.** {SUPPORT}.", ""] if f["story"] else [f"**{P(f['head'])}**", ""]
    if f["key"] and (FIG / KEY_FIG).exists():
        out += [f"![Key figure: the main finding at a glance](figures/{KEY_FIG})", ""]
    out += ["| Number | What it is |", "|---|---|"] + [f"| {P(v)} | {P(l)} |" for v, l in f["big"]]
    m, what, status = f["miles"][1]
    out += ["", f"Milestone 2 of the plan ({P(what)}): {P(status).replace('why: sections', 'why: report sections')}. "
                f"{P(f['warn']).replace('Its warnings', 'The warnings')}. A negative result, reported as it is; what it shows is what limits the forecast. "
                f"The details are on page 1 of `Flood_Report.pdf`, the conclusions with their evidence in its section 8."]
    return "\n".join(out)
