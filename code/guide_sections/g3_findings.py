"""Guide, steps 10 to 13 - what the project found when it looked closely at its own forecasts.

  10  Could its warnings be trusted?                   (report section 5)
  11  Why not? What the model is given matters most    (report sections 4.3, 4.5 and 5)
  12  Looking from above: maps and satellite data      (report section 6)
  13  A newer network: the Transformer                 (report section 7)

Written for readers who are new to machine learning: short sentences, plain words, few numbers. The concept boxes of these steps
are 'control' (11), 'reanalysis' (12) and 'transformer' (13); every other technical word was explained in a box of an earlier step.

Where the numbers come from (each read by key, none typed in):
  flood_analysis.json, flood_events.csv   the warning scores, the verdicts, the example event, the onsets sorted by rain timing
  rain_timing.json, day_windows.json      the test of rain stamped up to the evening, with its control; where the rain table's day ends
  same_misses.json                        do the kinds of model miss the same onsets
  basin_gis.json                          the basin properties worked out again from the two rasters
  satellite_data.json, satellite_inputs.json   satellite rain, soil wetness and water vapour as inputs
  model_comparison.json                   Transformer against LSTM, and the cost of training
Nothing is computed beyond differences, counts, smallest and largest, the size of one grid cell (a unit conversion, as in section 6.2
of the report), the number of comparisons that would look clear by chance (their count times the share that the results file gives,
as in section 6.2) and formatting.

Every paragraph that states an outcome is built inside _try(). It first checks that the stored numbers still say what the sentence
says (_need). If a file or a key is missing, or a rerun has changed the outcome, the paragraph is left out and one line goes to
stderr: the guide never prints a sentence that the results files do not support.

Layout: one order, with no switch. Every picture is shown at the full text width, as a float (_fig), directly after the paragraph
that leads to it. A picture that does not fit on the rest of a page moves to the top of the next page and the paragraphs after it
fill the page, wherever on a page this part happens to start. So the paragraph that reads a picture can stand on the page before it:
that paragraph names the picture, in the words its caption starts with (PIC). Boxes, the grid of verdicts and the last lines of a step
clear the floats, so a step always ends below its pictures. The heading of a step, its lead and its first paragraph are one block
(guide_common.keep), so no heading is left alone at the foot of a page; 'What to remember' and the line 'Full detail' are another.
One picture needs nearly a whole page, the second one of step 12: the paragraph that leads to it is kept in one block with it
(_fig(before=...)), the box follows the picture, and its caption is kept to four lines. Together they fill one page; with the box in
front of the picture, or with a longer caption, the page before the picture stays more than half empty.
"""
import math
import re
import sys

import pandas as pd

from common import FIG
from flood import LAND_SHORT, LEADS, STATIC
from guide_common import concept, final_detail_for, keep, remember, step
from report_common import LEAD_WORD, count, fig, n2, pct, tag
from guide_sections.g5_next import available as step16_available
from report_sections.s0_summary import (_and, _leads, _word, criteria, day_windows, nse_reading, onset_clear, share_word, timing_facts,
                                         timing_verdict)
from report_sections.s5_flood import RAIN_BINS, STATUS
from report_sections.s6a_maps import CHECKED
from report_sections.s6b_satellite import KM_PER_DEGREE, _split

FIRST = LEADS[0]                                                # the lead time the text quotes: 1 day ahead
PERSIST, PAST, GAUGE, UPPER = "base_persistence", "lstm_w", "lstm_wq", "lstm_wqf"      # 'tomorrow = today'; the LSTM with its three sets of inputs
DRAWN = (PERSIST, "base_rf_wq", GAUGE, UPPER)                   # the four forecasts that figures/flood_skill.png draws (06_flood_analysis.py: SHOWN)
QUOTED, QUOTED_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
KINDS = ("all", "onset")                                        # all high-flow days; onsets only
KIND_WORD = {"all": "all high-flow days", "onset": "onsets"}
RANK = {"NOT YET": 0, "CAUTION": 1, "GO": 2}
# the names of the pictures: a caption starts with its name, and the paragraph that speaks about the picture quotes it (_named)
PIC = {"event": "One onset, day by day", "skill": "The warnings of four models", "fail": "Where the warnings of an onset fail",
       "day": "Which 24 hours a daily value covers", "cover": "Land cover inside the basin outlines",
       "inputs": "What the new inputs changed", "cost": "Transformer against LSTM"}
# layout only (see the docstring): the float for the pictures, what clears it, and the grid of verdicts (the block that keeps a heading
# with its first paragraph is guide_common.keep)
STYLE = ("<style>.g3 .ff{float:left;width:100%;break-inside:avoid} .g3 .ff figure{margin-top:2pt} "
         ".g3 .keep,.g3 .concept,.g3 .remember,.g3 .who,.g3 .vg{clear:both} .g3 .keep,.g3 .vg{break-inside:avoid} "
         ".g3 table.vg{margin:4pt 0 2pt} .g3 table.vg th,.g3 table.vg td{text-align:center;vertical-align:middle;padding:2.2pt 3pt} "
         ".g3 table.vg td:first-child,.g3 table.vg th:first-child{text-align:left} "
         ".g3 table.vg th.grp{color:var(--ink2);border-bottom:1px solid var(--grid)} "
         ".g3 table.vg tr.sub th{border-top:none;font-weight:400;font-size:8.6pt;white-space:nowrap}</style>")
lk = lambda k: f"l{k}"


# ---- helpers ------------------------------------------------------------------------------------------------------------------
def _try(what, make):
    """make() builds one paragraph (or box, or caption) from the results files. If a file or a key is missing, or the stored numbers no
    longer support the sentence (_need), the piece is left out and one line on stderr says so."""
    try:
        return make()
    except (KeyError, TypeError, IndexError, ValueError, AttributeError, ZeroDivisionError) as e:
        print(f"g3_findings: {what} left out ({type(e).__name__}: {e})", file=sys.stderr)
        return ""


def _need(holds, outcome):
    """Stop a sentence whose outcome the stored numbers no longer show (caught by _try)."""
    if not holds:
        raise ValueError(f"the results files no longer show that {outcome}")


def _fig(name, caption, before=""):
    """A picture of figures/ at the full text width, as a float (see STYLE). before: the paragraph that leads to the picture, kept in one
    block with it, above it; the two then never part (used where a picture needs nearly a whole page, see the docstring). Without its
    caption, or if the file is missing, the picture is left out with one line on stderr and the paragraph stands alone."""
    if not caption:
        return before
    if not (FIG / name).exists():
        print(f"g3_findings: figures/{name} is missing: the picture is left out", file=sys.stderr)
        return before
    return f'<div class="ff">{before}{fig(name, caption)}</div>'


def _cap(name, how):
    """A caption: the bold name of the picture, then how to read it."""
    return f"<b>{name}</b> <b>How to read it:</b> {how}"


def _named(key, first=False):
    """The name of a picture as a paragraph quotes it: 'the picture &lsquo;...&rsquo;' ('The picture ...' as the first words of a
    sentence). The caption of that picture starts with the same words (PIC), so a reader finds the picture when it stands on the next page."""
    return f"{'The' if first else 'the'} picture &lsquo;{PIC[key]}&rsquo;"


def _opening(n, lead, first):
    """The heading of step n, its lead and its first paragraph as one block that is not split over two pages."""
    return keep(step(n, lead), first)


def _closing(text, sections):
    """The end of a step as one block: what to remember, and where the report has the full detail."""
    return keep(remember(text), final_detail_for(sections))


# ---- step 10: could its warnings be trusted? ------------------------------------------------------------------------------------
def _step10(R, T):
    A = R.get("flood_analysis") or {}
    plain = lambda n, kind, k=FIRST: A["skill"][n]["plain"]["test"]["summary"][lk(k)][kind]      # warnings at the high-flow level itself
    level = lambda n, k: A["skill"][n]["tuned"][lk(k)]["trigger"]                                # the warning level the validation years chose, as a factor
    verdict = lambda n, k, kind: (A["verdict"]["services"][n][lk(k)][kind]["pooled"] if n in A["verdict"]["services"] else
                                  A["verdict"]["other_models_pooled"][n][lk(k)][kind])

    def what():
        a, o = plain(GAUGE, "all"), plain(GAUGE, "onset")
        n_days, n_on = a["hits"] + a["misses"], o["hits"] + o["misses"]
        _need(n_on < n_days, "only a part of the high-flow days are onsets")
        return (f"<p>A model warns for a day when its forecast for that day is above the river&rsquo;s high-flow level. A warning is then a hit "
                f"or a false warning, and a high-flow day without a warning is a miss. The test years hold {count(n_days)} high-flow days in the "
                f"{_word(len(A['basins']))} rivers. Only {count(n_on)} of them are onsets. On the other days the river was already high when the "
                f"forecast was made.</p>")

    def event():
        """The example event: (paragraph, caption). Its rule picks an onset that the LSTM with the river gauge missed and the upper bound caught."""
        e, E = A["event_shown"], T["flood_events"]
        row = E[(E.gauge_id == e["gauge_id"]) & (E.onset_date == e["onset_date"])].iloc[0]
        _need(not bool(row[f"{GAUGE}_warned_l{FIRST}_chosen_level"]) and bool(row[f"{UPPER}_warned_l{FIRST}_chosen_level"]),
              "the example onset is missed with the river gauge and warned with the perfect weather forecast")
        day = pd.Timestamp(e["onset_date"])
        storm = f", when {STATUS.get(e.get('cyclone_status'), 'tropical cyclone')} {e['cyclone_name'].title()} passed" if e.get("cyclone_name") else ""
        same, n_on = e.get("onsets_with_this_pattern"), e.get("onsets")              # how many onsets went the way of the example
        common = (f" Its pattern is common: {count(same)} of the {count(n_on)} onsets were missed with the river gauge and caught with a perfect "
                  f"weather forecast." if same and n_on and same >= n_on / 3 else "")
        counted = "; the next picture counts all of them" if (FIG / "flood_skill.png").exists() else ""
        text = (f"<p>{_named('event', first=True)} follows the river {e['basin']} in {day:%B %Y}{storm}. For the first day of the event the LSTM "
                f"with the river gauge gives no warning. The same LSTM given a perfect weather forecast, the dashed line, does warn. On the later "
                f"days a warning is easy, because the gauge itself shows that the river is high. The example was picked by a rule, not by eye."
                f"{common} But one event does not say how often a model is right{counted}.</p>")
        cap = _cap(f"{PIC['event']}: {e['basin']}, {day:%B %Y}.",
                   f"In the middle panel the black line is the measured flow. It crosses the red dashed high-flow level on {day.day} {day:%B}, "
                   f"the onset. The solid blue line, the forecast made the evening before, is still below the level on that day: the red cross "
                   f"in the bottom panel, a miss.")
        return text, cap

    def scores():
        g_all, g_on, p_all = plain(GAUGE, "all"), plain(GAUGE, "onset"), plain(PERSIST, "all")
        seeds = A["skill"][GAUGE]["seeds"][lk(FIRST)]["plain"]["all"]
        _need(g_on["pod"] < 0.5 < g_all["pod"], "the LSTM with the river gauge warns of most high-flow days and of few onsets")
        _need(seeds["pod_min"] <= p_all["pod"] <= seeds["pod_max"], "'tomorrow = today' warns of about as many high-flow days as the LSTM")
        return (f"<p>{_named('skill', first=True)} counts all events of the test years. {LEAD_WORD[FIRST].capitalize()}, the "
                f"LSTM with the river gauge warns of {pct(g_all['pod'])} of all high-flow days. That looks useful. But {QUOTED}, which only repeats the gauge reading, warns of "
                f"{pct(p_all['pod'])}. Onsets are another matter. The LSTM warns of {pct(g_on['pod'])} of them. And of the warnings that it gives "
                f"while a river is still low, {pct(g_on['far'])} are false: no high flow follows.</p>")

    def skill_caption():
        go = criteria(R)["go_pod"]
        solid = max(plain(n, "onset", k)["pod"] for n in DRAWN if n != UPPER for k in LEADS)      # the three models without a weather forecast
        dashed = [plain(UPPER, "onset", k)["pod"] for k in LEADS]
        _need(solid < 0.5 < min(dashed), "only the upper bound warns of most onsets")
        near = "comes near the green band that GO asks for" if max(dashed) < go else "reaches the green band that GO asks for"
        return _cap(f"{PIC['skill']}, 1 to 3 days ahead.",
                    f"Left: the share of days that were warned (high is good); right: the share of warnings that were false (low is good). Look "
                    f"at the bottom left panel, the onsets: none of the three solid lines rises above {pct(solid)}. Only the dashed line, the "
                    f"upper bound, {near}.")

    def reading():
        v_all = verdict(GAUGE, FIRST, "all")
        on = {verdict(n, k, "onset") for n in (PAST, GAUGE) for k in LEADS}
        _need(v_all == "CAUTION" and verdict(PERSIST, FIRST, "all") == v_all, "the LSTM with the river gauge and 'tomorrow = today' both get CAUTION")
        _need(on == {"NOT YET"}, "the verdict for onsets is NOT YET at every lead time")
        _need(all(RANK[verdict(UPPER, k, kind)] > RANK[verdict(GAUGE, k, kind)] for k in LEADS for kind in KINDS), "the upper bound gets a better verdict in every cell")
        missed = 1 - A["verdict"]["services"][GAUGE][lk(FIRST)]["onset"]["pod"]
        return (f"<p>The rule for a verdict was fixed before the analysis (step 4), and the grid of verdicts applies that rule. With the river "
                f"gauge, the LSTM gets {tag(v_all)} for all high-flow days {LEAD_WORD[FIRST]}. {QUOTED_CAP} gets the same verdict. So this "
                f"CAUTION says that a high river will stay high. It is not a flood warning. For onsets, the warnings that matter, the verdict is "
                f"{tag('NOT YET')} at every lead time: {LEAD_WORD[FIRST]}, {pct(missed)} of the onsets get no warning. Only the last row of the "
                f"grid does better, and that row is an upper bound that cannot be had in practice.</p>")

    def grid():
        """The verdicts as a small grid: one row per forecast, one column per kind of warning and lead time."""
        V, crit = A["verdict"]["services"], criteria(R)
        rows = ((PERSIST, f"{QUOTED_CAP} (nothing is learned)"), (PAST, "LSTM with past weather only"), (GAUGE, "LSTM with the river gauge"),
                (UPPER, "LSTM with a perfect weather forecast too (upper bound)"))
        starred, lone = False, []                                                  # lone: the starred verdicts that no single training gets

        def cell(n, k, kind):
            nonlocal starred
            seeds = V[n][lk(k)][kind]["single_seeds"] if n in V else {}
            agree = sum(v == verdict(n, k, kind) for v in seeds.values())
            shaky = agree < len(seeds)
            starred = starred or shaky
            if shaky and agree == 0:
                lone.append(f"the {verdict(n, k, kind)} of {KIND_WORD[kind]} {LEAD_WORD[k]}")
            return f"<td>{tag(verdict(n, k, kind))}{'&nbsp;*' if shaky else ''}</td>"

        body = "".join(f"<tr><td>{name}</td>" + "".join(cell(n, k, kind) for kind in KINDS for k in LEADS) + "</tr>" for n, name in rows)
        head = ('<tr><th rowspan="2">The forecast comes from</th><th class="grp" colspan="3">All high-flow days</th>'
                '<th class="grp" colspan="3">Onsets only: the warnings that matter</th></tr><tr class="sub">'
                + "".join(f"<th>{LEAD_WORD[k]}</th>" for _ in KINDS for k in LEADS) + "</tr>")
        cols = '<col style="width:34%">' + '<col style="width:11%">' * 6
        row_of = lambda n: [verdict(n, k, kind) for kind in KINDS for k in LEADS]
        same = (f"The rows of {QUOTED} and of the LSTM with the river gauge read the same: one CAUTION, then NOT YET. "
                if row_of(GAUGE) == row_of(PERSIST) == ["CAUTION"] + ["NOT YET"] * 5 else "")
        # the warning level that the validation years chose for each forecast, as a factor of the high-flow level: the grid applies it, the
        # picture above the grid does not (its lines are drawn at the high-flow level itself)
        levels = [level(n, k) for n, _ in rows for k in LEADS]
        drawn = [level(n, k) for n in DRAWN for k in LEADS]                         # the range that the note inside the picture gives
        chosen = ""
        if (min(levels) != 1 or max(levels) != 1) and not (FIG / "flood_skill.png").exists():      # no picture to compare with
            chosen = (f" For the verdicts each of these {_word(len(rows))} forecasts warns when it passes its own level: {n2(min(levels))} to "
                      f"{n2(max(levels))} times the high-flow level, chosen on the validation years.")
        elif min(levels) != 1 or max(levels) != 1:
            chosen = (f" For the verdicts each of these {_word(len(rows))} forecasts may warn a little earlier or later than in the picture above: "
                      f"when it passes {n2(min(levels))} to {n2(max(levels))} times the high-flow level. That factor was chosen on the validation "
                      f"years{' (step 16 says what moving this warning level does)' if step16_available(R) else ''}.")
            # a GO of the upper bound whose line in the picture lies outside the green band: the factor is the reason
            odd = [(k, kind) for kind in KINDS for k in LEADS if verdict(UPPER, k, kind) == "GO"
                   and (plain(UPPER, kind, k)["pod"] < crit["go_pod"] or plain(UPPER, kind, k)["far"] > crit["go_far"])]
            if len(odd) == 1 and plain(UPPER, odd[0][1], odd[0][0])["pod"] < crit["go_pod"]:
                k, kind = odd[0]
                just = "just " if crit["go_pod"] - plain(UPPER, kind, k)["pod"] < 0.05 else ""
                chosen += (f" This is why the last row can read GO for {KIND_WORD[kind]} {LEAD_WORD[k]}, although the dashed line of the picture "
                           f"stays {just}below the green band.")
            # the note inside the picture gives a narrower range, and the one row that the picture does not draw is the reason
            both = [level(n, k) for n, _ in rows if n in DRAWN for k in LEADS]
            wider = min(levels) <= min(drawn) and max(drawn) <= max(levels) and (min(levels), max(levels)) != (min(drawn), max(drawn))
            if wider and min(drawn) <= min(both) and max(both) <= max(drawn) and [n for n, _ in rows if n not in DRAWN] == [PAST]:
                chosen += (f" (The note in the picture gives {n2(min(drawn))} to {n2(max(drawn))}: the picture does not show the LSTM with past "
                           f"weather only.)")
        n_seeds = len(V[GAUGE][lk(FIRST)]["all"]["single_seeds"])
        star = ""
        if starred:
            star = f" * Not all {_word(n_seeds)} single trainings get this verdict" + (f"; for {_and(lone)}, none does." if lone else ".")
        cap = (f'<p class="cap"><b>The verdicts, test years, the {_word(len(A["basins"]))} rivers together.</b> <b>How to read it:</b> Each cell is '
               f"the verdict for one kind of warning and one lead time. {same}GO asks for at least {pct(crit['go_pod'])} warned and at most "
               f"{pct(crit['go_far'])} false warnings (step 4).{chosen}{star}</p>")
        return f'<div class="vg"><table class="num vg"><colgroup>{cols}</colgroup>{head}{body}</table>{cap}</div>'

    def not_yet():
        on = {verdict(n, k, "onset") for n in (PAST, GAUGE) for k in LEADS}
        return " Here the warnings of an onset cannot be trusted yet." if on == {"NOT YET"} else ""

    ev_text, ev_cap = _try("step 10, the example event,", event) or ("", "")
    return "\n".join([
        _opening(10, "A forecast of river flow is not yet a warning. This step turns the forecasts into warnings and checks them against what "
                     "the rivers did.", _try("step 10, what a warning is,", what)),
        _fig("flood_event.png", ev_cap), ev_text,
        _try("step 10, the scores,", scores), _fig("flood_skill.png", _try("step 10, the caption of flood_skill.png,", skill_caption)),
        _try("step 10, the reading of the verdicts,", reading), _try("step 10, the grid of verdicts,", grid),
        _closing("Score what the user needs. A model can look good on all high-flow days and still miss the first day, the only day on which a "
                 "warning comes before the water." + _try("step 10, the last sentence of what to remember,", not_yet), "section 5")])


# ---- step 11: why not? what the model is given matters most -----------------------------------------------------------------------
def _step11(R, T):
    A = R.get("flood_analysis") or {}
    onsets = lambda part, n, c: A["failures"][part][n]["tuned"][c]["onset"]          # the onsets of one class, 1 day ahead

    def timing():
        """(paragraph, caption): the onsets sorted by season and storm, and by where the rain of the two days is stamped."""
        low, top = RAIN_BINS[0], RAIN_BINS[-1]                                         # share of the two days' rain stamped on the day being forecast
        _need((low, top) == ("0-25%", "75-100%"), "the onsets are sorted into quarters of the rain stamped on the day being forecast")
        g_low, g_top, u_low, u_top = (onsets("by_rain_timing", n, c) for n in (GAUGE, UPPER) for c in (low, top))
        groups = [onsets(part, GAUGE, c)["pod"] for part in ("by_season", "by_storm_type") for c in A["failures"][part][GAUGE]["tuned"]]
        _need(max(groups) < 0.5, "the LSTM with the river gauge misses more than half of the onsets in every season and for every kind of storm")
        _need(g_top["pod"] < 0.2 < 0.5 < g_low["pod"] and u_top["pod"] > u_low["pod"], "the warnings fail where the rain is stamped on the day being forecast")
        text = (f"<p>Season and kind of storm do not explain the misses. The timing of the rain does. A forecast for tomorrow is made at the end "
                f"of today, the issue day. The model has the rain table, the daily rain of the data set, only up to that day. Now take the rain of "
                f"two days: the issue day and the day being forecast. Where three quarters or more of it is already in the table, the LSTM warns "
                f"of {pct(g_low['pod'])} of the onsets. Where three quarters or more carries tomorrow&rsquo;s date, the LSTM warns of "
                f"{pct(g_top['pod'])}. {_named('fail', first=True)} shows this in its bottom right panel.</p>")
        cap = _cap(f"{PIC['fail']}.",
                   f"Dark bars are the onsets warned by the LSTM with the river gauge, light bars by the same LSTM given a perfect weather "
                   f"forecast. The red marks are the share of each model&rsquo;s warnings that were false. In the two top panels the dark bars "
                   f"are low in every season and for every kind of storm: more than half of the onsets are missed everywhere. In the bottom "
                   f"right panel the dark bars fall from {pct(g_low['pod'])} to "
                   f"{pct(g_top['pod'])} as more of the rain carries the date of the day being forecast. The bottom left panel sorts the basins "
                   f"by size, with a pair of markers for each basin. With only {_word(len(A['basins']))} basins it shows a pattern, not a cause, and "
                   f"can be skipped.")
        return text, cap

    def same():
        M = R["same_misses"]
        L, kinds = M[lk(FIRST)], list(M["kinds"])
        missed, rescued = L["missed_by_all_kinds"], L["of_missed_by_all"]["warned_with_perfect_forecast"]["share"]
        _need(missed["share"] > 0.5 and rescued > 0.5, "most onsets are missed by every kind of model and the perfect weather forecast warns of most of those")
        name = {"LSTM": "the LSTM", "Transformer": "a Transformer, a newer kind of network (step 13)"}
        # the finding itself ('Among the models tried, ...') is stated once, word for word, in step 15; here the two numbers speak
        best = max(L["models"][M["kinds"][k]]["pod"] for k in kinds)                   # no kind warns of more onsets than the LSTM does
        answer = " Not among those tried." if "LSTM" in M["kinds"] and L["models"][M["kinds"]["LSTM"]]["pod"] >= best else ""
        return (f"<p>Would another kind of model see more?{answer} {_word(len(kinds)).capitalize()} kinds were given the same inputs, past "
                f"weather and river gauge: {_and(name.get(k, k) for k in kinds)}. All {_word(len(kinds))} miss the same {count(missed['count'])} "
                f"of the {count(L['n_onsets'])} onsets. Given a perfect weather forecast, the LSTM warns of {pct(rescued)} of these.</p>")

    def stamp():
        """(paragraph, caption): which 24 hours the rain table's day covers."""
        W = day_windows(R)
        _need(W is not None and W["daymet_hour"] < 12, "the rain table's day ends in the morning")
        text = (f"<p>&lsquo;Not yet in the table&rsquo; does not always mean &lsquo;not yet fallen&rsquo;. A daily value covers 24 hours, but "
                f"which ones? The rain table&rsquo;s day seems to end in the morning, at about {W['daymet']}. That is an inference from our own "
                f"data, not a documented fact. If it holds, the rain of the rest of the issue day has fallen when the forecast is made. But "
                f"the table gives that rain tomorrow&rsquo;s date, so the model does not see it. {_named('day', first=True)} shows it.</p>")
        cap = _cap(f"{PIC['day']}.",
                   f"Time runs from left to right, and the black vertical line is the moment the forecast is made. Each bar is the 24 hours that "
                   f"one source of data counts as the issue day, day t in the picture. The orange bar, the rain table, ends at about "
                   f"{W['daymet']}; the dashed box to its right is rain that has fallen but carries the next day&rsquo;s date. The blue bar "
                   f"(NLDAS in the picture) is the second rain data set that this step tests: its day ends at about {W['nldas']}. The green bar is "
                   f"the satellite rain of step 12 ({W['imerg']}), and the dark bar the river gauge.")
        return text, cap

    def nse_sentence(r, n):
        """The NSE result of the test in one sentence, worded from the outcome by the reading of the report (s0_summary.nse_reading). It
        says 'the NSE is higher in ... basins', so that nobody reads a higher flow into it."""
        move = f"the median {'rises' if r['gain'] > 0 else 'falls'} from {n2(r['nse_base'])} to {n2(r['nse_with_nldas'])}"
        what = nse_reading(r, n)
        if what == "clear":
            return f"In NSE the gain is clear as well: {move}, and the NSE is higher in {r['basins_better']} of the {n} basins."
        if what == "mixed":
            return f"In NSE the result is mixed: {move}, but the NSE is higher in only {r['basins_better']} of the {n} basins."
        return f"In NSE there is no clear difference: {move}, and the single trainings overlap."

    def test():
        tf, W = timing_facts(R), day_windows(R)
        r, tv = tf["gauge"][FIRST], timing_verdict(R)
        _need(onset_clear(r), "rain up to the evening lifts the onsets warned in every seed, above the control")
        _need(W is None or 16 <= int(W["nldas"][:2]) <= 19, "the day of the second rain data set ends in the early evening")      # where it was measured
        base, more, older, top = (pct(r[k]) for k in ("onset_warned_base", "onset_warned_with_nldas", "onset_warned_control", "onset_warned_perfect_forecast"))
        control = (f"The control stays at {older}. So the gain comes from the later hours." if older == base else
                   f"The control gives {older}. So the later hours make the difference between {older} and {more}.")
        still = f" And with that rain the verdict for onsets is still {tag(tv['verdict'])}." if tv and tv["verdict"] != "GO" else ""
        return (f"<p>This idea was tested. The LSTM with the river gauge was given one more input: rain from a second data set, whose day ends in "
                f"the early evening of the issue day. With it the LSTM warns of {more} of the onsets {LEAD_WORD[FIRST]}, not {base}. Does the gain "
                f"come from the later hours, or only from having a second rain value? A control answers that: a second run with the same extra "
                f"rain, made one day older. {control} A perfect weather forecast, an upper bound that cannot be had in practice, gives {top}. "
                f"From {base} to {top} is the gap; rain up to the evening closes {share_word(r['onset_share_of_gap_closed'])} of it. "
                f"{nse_sentence(r, tf['n'])}{still}</p>")

    def control_box():
        r = timing_facts(R)["gauge"][FIRST]
        _need(onset_clear(r), "the control does not show the gain of the test")
        base, more, older = (pct(r[k]) for k in ("onset_warned_base", "onset_warned_with_nldas", "onset_warned_control"))
        gone = (f"Then the gain is gone: {older} of the onsets are warned, as many as without any extra rain." if older == base else
                f"Then {older} of the onsets are warned, against {more} in the first run and {base} without any extra rain.")
        return concept("control", here=f"The first run gives the LSTM extra rain whose day ends in the early evening of the issue day. The control "
                                       f"gives the same rain made one day older. {gone}")

    t_text, t_cap = _try("step 11, the onsets sorted by rain timing,", timing) or ("", "")
    s_text, s_cap = _try("step 11, the day stamp of the rain table,", stamp) or ("", "")
    return "\n".join([
        _opening(11, "The misses follow one pattern. The rain that makes the river rise is not yet in what the model is given.", t_text),
        _fig("flood_failures.png", t_cap),
        _try("step 11, the same misses of all kinds of model,", same),
        s_text, _fig("day_windows.png", s_cap),
        _try("step 11, the test with a control,", test),
        _try("step 11, the box 'control',", control_box) or concept("control"),
        _closing("When a model fails, ask first what the model was given, and when. Here every kind of model misses the same onsets, and more "
                 "information lifts the warnings. A control showed that part of the missing rain had already fallen.", "sections 4.3, 4.5 and 5")])


# ---- step 12: maps and satellite data -----------------------------------------------------------------------------------------------
def _step12(R, T):
    G, S, X = R.get("basin_gis") or {}, R.get("satellite_data") or {}, R.get("satellite_inputs") or {}

    def maps(shown):
        """shown: the picture of the land cover follows this paragraph, so the paragraph names it."""
        SRC = G["sources"]
        cell = re.search(r"about (\d+) m", SRC["elevation"]["resolution"]).group(1)
        see = f" {_named('cover', first=True)} shows the land cover." if shown else ""
        return (f"<p>The remote-sensing and GIS part of the project treats the basins as maps. (A GIS, a geographic information system, is "
                f"software for maps as data.) Two open rasters were used: maps stored as a grid of cells. One holds the height of the ground in "
                f"cells of about {cell} metres, the other the land cover of {SRC['land_cover']['year']}. Each basin outline was laid over both."
                f"{see}</p>")

    def cover_caption():
        B = G["basins"]
        MTN, SOUTH = list(LAND_SHORT)[0], list(LAND_SHORT)[-1]
        forest = [r["land_cover_share"]["forest"] for r in B.values() if r["landscape"] == MTN]
        wet = [r["land_cover_share"]["wetland"] for r in B.values() if r["landscape"] == SOUTH]
        _need(min(forest) > 0.5 and min(wet) > max(r["land_cover_share"]["wetland"] for r in B.values() if r["landscape"] == MTN),
              "the mountain basins are mostly forest and the southern basins have more wetland")
        return _cap(f"{PIC['cover']}, {G['sources']['land_cover']['year']}.",
                    f"Each small map is one of the {_word(len(B))} basins and each row one landscape; the white dot is the river gauge. Dark green "
                    f"is forest, orange is farmland, light blue is wetland. The mountain basins in the top row are {pct(min(forest))} to "
                    f"{pct(max(forest))} forest; the southern basins in the bottom row are {pct(min(wet))} to {pct(max(wet))} wetland.")

    def compare():
        C = G["comparison_with_camels"]
        checked = [w for k, w in CHECKED.items() if k in STATIC]                       # the basin properties that can be worked out from the rasters
        ranks = [C[k]["rank_correlation"] for k in ("area_vs_gages2", "elev_mean", "slope_mean", "forest_share")]
        _need(len(checked) == len(ranks) and min(ranks) >= 0.9, "the basins come in nearly the same order")
        notes = " ".join(G.get("notes", []))                                           # what the script of the maps says about its own work
        _need("standardising it over the basins" in notes, "the network reads each property on one scale over the basins")
        again = (" The network was not trained again with the new values, so the results do not say whether they would change a forecast."
                 if "Nothing in this script is fitted" in notes else "")
        # the rank correlations themselves (0.98 to 0.99) stand in section 6.1 of the report: the guide says in words what they show,
        # and _need above stops the sentence if they fall below 0.9
        return (f"<p>From the cells inside each outline, {_word(len(checked))} basin properties were worked out again: {_and(checked)}. This "
                f"checks the table of the data set, which the LSTM reads. The values differ in places: slope, for one, depends on the size of "
                f"the cells. But the basins come in nearly the same order. That is what counts: each property is put on one scale, so the "
                f"network mainly sees how the basins compare.{again}</p>")

    def inputs():
        _need({"imerg_wq", "imerg_w"} <= set(X["experiments"]["a_rain_source"]["against_base"])
              and {"soil_w", "pw_w"} <= set(X["experiments"]["b_extra_inputs"]["against_base"]), "satellite rain, soil wetness and water vapour were tried")
        _need("gauge-based rain (Daymet) itself" in X["what_inputs_know_about_next_daymet_day"], "the rain table is worked out from rain gauges")
        return ("<p>The rain table of the data set is worked out from rain gauges: instruments on the ground that catch the rain at one spot. "
                "(They are not the river gauges, which measure the flow.) Satellites see rain where no rain gauge stands. So new inputs were "
                "tried in the same LSTM, one change at a time: satellite rain (a NASA product called IMERG), soil wetness, and the water vapour "
                "in the air.</p>")

    def box():
        ids = list(X["basins"])
        cells = S["cells_per_basin"]
        grid = re.search(r"([\d.]+) \(latitude\) x ([\d.]+) \(longitude\) degree", S["grid"]["finding"]["GWETTOP"]["grid_assumed"])
        lats = sorted(cells[g]["centroid_lon_lat"][1] for g in ids)
        mid = (lats[(len(ids) - 1) // 2] + lats[len(ids) // 2]) / 2                       # the latitude of the middle basin
        km2 = float(grid.group(1)) * KM_PER_DEGREE * float(grid.group(2)) * KM_PER_DEGREE * math.cos(math.radians(mid))      # one cell, as in section 6.2
        _need(all(cells[g]["camels_area_km2"] < km2 for g in ids), "one cell of the reanalysis is larger than every basin")
        return concept("reanalysis", here=f"Soil wetness and water vapour came from a NASA reanalysis. One cell of its grid covers about "
                                          f"{count(round(km2, -2))} square kilometres, more than the largest of the {_word(len(ids))} basins.")

    def by_chance():
        """The lesson of many comparisons: how many comparisons were made, and how many the rule of the experiment calls clear when
        nothing differs (the count times the share that the results file gives, as section 6.2 of the report works it out). '' with one
        line on stderr when a key is missing."""
        try:
            n_made = sum(v[1] for v in X["comparisons_larger_than_seed_noise_nse"].values())
            shares = list(X["seed_noise_rule"]["share_of_comparisons_flagged_when_there_is_no_real_difference"].values())
            expected = n_made * sum(shares) / len(shares)
        except (KeyError, TypeError, IndexError, ZeroDivisionError) as e:
            print(f"g3_findings: step 12, how many comparisons look clear by chance, left out ({type(e).__name__}: {e})", file=sys.stderr)
            return ""
        if not 1 <= round(expected) < n_made:
            return ""
        return (f" {count(n_made)} comparisons were made, and about {expected:.0f} would look clear by chance alone. So a pattern counts, and a "
                f"single filled marker does not.")

    def helped():
        """(paragraph, caption): what the new inputs changed, by the rule of the experiment's own results file (larger than the seed noise).
        The caption is kept to four lines: the picture and the paragraph in front of it (inputs) have to share one page."""
        ca, cb = X["experiments"]["a_rain_source"]["against_base"], X["experiments"]["b_extra_inputs"]["against_base"]
        up = _split(ca["imerg_wq"])[0]                                                     # lead times with a clear gain: satellite rain, with the gauge
        _need(up and up[0] == FIRST and not any(_split(ca["imerg_w"])[:2]), "satellite rain helps clearly with the river gauge and not without it")
        _need(_split(cb["soil_w"])[0] == [FIRST] and not _split(cb["soil_wq"])[0], "soil wetness helps clearly only without the river gauge, 1 day ahead")
        pw = [cb[key][lk(k)]["nse_median"]["larger_than_seed_noise"] is True for key in ("pw_w", "pw_wq") for k in LEADS]
        _need(not any(pw), "water vapour changes the NSE clearly in no comparison")
        m = ca["imerg_wq"][lk(FIRST)]["nse_median"]
        # the gain is given as a difference: this test has its own, shorter periods, so its starting score is not that of step 9 (the
        # note inside the picture says that its two rows use different years)
        text = (f"<p>What helped? {_named('inputs', first=True)} shows every comparison. With the river gauge, satellite rain in place of the "
                f"rain table raised the NSE clearly {_leads(up)}. {LEAD_WORD[FIRST].capitalize()} it rose by {n2(m['difference'])}. Without the "
                f"river gauge it made no clear difference. Soil wetness helped clearly only without the river gauge, {LEAD_WORD[FIRST]}. Water "
                f"vapour changed the NSE clearly in none of {_word(len(pw))} comparisons.{by_chance()}</p>")
        n_clear = sum(v[0] for v in X["comparisons_larger_than_seed_noise_nse"].values())
        top_right = sum(ca[key][lk(k)]["nse_median"]["larger_than_seed_noise"] is True for key in ("imerg_wq", "gauge_imerg_wq") for k in LEADS)
        _need(0 < top_right <= n_clear, "the clear changes of satellite rain with the river gauge are among the filled markers")
        cap = _cap(f"{PIC['inputs']}.",
                   f"Each marker is the change of the median NSE against the same LSTM without the new input; the zero line means no change. In "
                   f"the picture, precipitable water is the water vapour in the air, and skill is the score. The grey band is twice the usual "
                   f"difference between two trainings of one model, so only a filled marker outside the band is a clear change. "
                   f"{_word(top_right).capitalize()} of the {_word(n_clear)} filled markers are at the top right: satellite rain with the river "
                   f"gauge.")
        return text, cap

    def why():
        know = X["what_inputs_know_about_next_daymet_day"]
        nxt = lambda name: know[name]["correlation_with_daymet_rain_of_the_next_day"]
        _need(nxt("satellite rain (IMERG)") > nxt("gauge-based rain (Daymet) itself"), "satellite rain of the issue day holds rain that the rain table dates one day later")
        r = S["rain_check_summary"]["imerg"]["median_correlation"]
        _need(r < 0.8, "satellite rain and the rain table agree only in part")
        return ("<p>Why would satellite rain beat the rain table? Probably not because its rain is better. Day by day the two agree only in part. "
                "But the satellite&rsquo;s day ends in the early evening of the issue day. So its value holds rain that the rain table dates one "
                "day later (step 11). The gain is probably a time-stamp effect.</p>")

    # the box 'reanalysis' has said that the water vapour of this step is the output of a weather model, so the paragraph does not say it again
    gnss = ("<p>GNSS, satellite navigation such as GPS, could add a real measurement. A GNSS receiver measures how long the air delays the satellite "
            "signals. Part of that delay tells how much water vapour is above the receiver. Real GNSS-derived water vapour was not tested.</p>")
    c_cap = _try("step 12, the caption of basin_landcover.png,", cover_caption)
    h_text, h_cap = _try("step 12, what the new inputs changed,", helped) or ("", "")
    return "\n".join([
        _opening(12, "Maps and satellites show the basins and their rain from above. This step asks what they add to the forecast.",
                 _try("step 12, the two rasters,", lambda: maps(bool(c_cap) and (FIG / "basin_landcover.png").exists()))),
        _fig("basin_landcover.png", c_cap),
        _try("step 12, the properties worked out again,", compare),
        _fig("satellite_inputs.png", h_cap, before=_try("step 12, the new inputs,", inputs)),      # one block: together they fill a page
        _try("step 12, the box 'reanalysis',", box) or concept("reanalysis"),
        h_text,
        _try("step 12, why satellite rain helps,", why),
        gnss,
        _closing("A new input is tested like a new model: one change at a time, and against the difference between two trainings. Then ask what "
                 "the input really is, and which hours it covers. Here the gain from satellite rain is probably a time-stamp effect.", "section 6")])


# ---- step 13: the Transformer -----------------------------------------------------------------------------------------------------
def _step13(R, T):
    X = R.get("model_comparison") or {}

    def box():
        M, days = X["models"], X["settings"]["sequence_days"]
        _need(X["causality"]["transformer"]["largest_change_up_to_day_t"] < 1e-5, "the mask of the Transformer holds")
        scored = (X["settings"].get("shared") or {}).get("block")                    # the days of one stretch whose forecasts are scored
        mask = ("A stretch holds many forecast days, so for each forecast a mask hides the days after it from the Transformer: it cannot look "
                "ahead." if scored and scored > 1 else "For each forecast a mask hides the days after it from the Transformer: it cannot look ahead.")
        return concept("transformer", here=f"The Transformer of this project has {count(M['transformer']['n_params'])} weights, the LSTM trained "
                                           f"beside it {count(M['lstm']['n_params'])}. Both read the same stretches of {days} days. {mask}")

    def same_job():
        S = X["settings"]
        days, ref = S["sequence_days"], S.get("lstm_all_config_for_reference")
        shorter = (f" To keep the cost down, both networks read shorter stretches of days here: {days} days each. The LSTM of steps 8 to 11 was "
                   f"trained on stretches of {ref['warmup'] + ref['block']} days: {ref['warmup']} to fill its memory and {ref['block']} to forecast."
                   if ref and ref["warmup"] + ref["block"] > days else "")
        return (f"<p>Today&rsquo;s large language models are built from Transformers. Here a small Transformer got the same job as the LSTM, and "
                f"only the network was changed. The {_word(X['comparison']['test'][lk(FIRST)]['n_basins'])} basins, the inputs (past weather and "
                f"river gauge), the three periods, the loss, the scoring and the stopping rule stayed the same. Each network was trained from "
                f"{_word(len(X['seeds']))} seeds.{shorter}</p>")

    def result():
        """(paragraph, caption): the accuracy and the cost, worded as section 7 of the report words them."""
        C, ratio = X["comparison"]["test"], X["cost_of_skill"]["transformer_over_lstm"]
        t = C["skill"]
        _need(t["seed_ranges_overlap"] or abs(t["transformer_minus_lstm"]) <= t["largest_seed_spread"], "there is no clear difference in accuracy")
        lo, hi = sorted((ratio["seconds_per_model"], ratio["seconds_per_update"]))
        bars = [ratio[k] for k in ("parameters", "operations_per_update", "seconds_per_update", "seconds_per_model")]      # the four pairs of bars
        _need(lo > 1 and min(bars) > 1, "the Transformer costs more to train")
        apart = max(abs(C[lk(k)]["nse_median_transformer_minus_lstm"]) for k in LEADS)
        text = (f"<p>The result, in {_named('cost')}: no clear difference in accuracy. Averaged over the three lead times, the LSTM scores an NSE "
                f"of {n2(t['lstm'], 3)} and the Transformer {n2(t['transformer'], 3)}. Two trainings of the same network differ by up to "
                f"{n2(t['largest_seed_spread'], 3)}. The cost differs a lot: training the Transformer took {lo:.0f} to {hi:.0f} times as long.</p>")
        cap = _cap(f"{PIC['cost']}: accuracy and cost.",
                   f"Left: the median NSE 1, 2 and 3 days ahead; the blue line is the LSTM, the orange line the Transformer. The two lines are "
                   f"never more than {n2(apart)} NSE apart. Right: four ways of counting the cost of training; every orange bar is "
                   f"{min(bars):.0f} to {max(bars):.0f} times as long as the blue one.")
        return text, cap

    def lesson():
        years = int(X["split"]["train"][1][:4]) - int(X["split"]["train"][0][:4])
        return (f"<p>Why so much longer? Attention compares every day with every other day, so twice as many days mean four times the work. A "
                f"newer or larger network is not automatically better. This holds for this amount of data: "
                f"{X['comparison']['test'][lk(FIRST)]['n_basins']} basins, {years} training years and small networks on a laptop. It is not a "
                f"verdict on Transformers. Only the learning rate of each network was checked; a wider search of settings could change the picture.</p>")

    r_text, r_cap = _try("step 13, the result,", result) or ("", "")
    return "\n".join([
        _opening(13, "Newer kinds of network exist. Would one of them forecast these rivers better than the LSTM?",
                 _try("step 13, the box 'transformer',", box) or concept("transformer")),
        _try("step 13, what was kept the same,", same_job),
        _fig("model_comparison.png", r_cap), r_text,
        _try("step 13, the lesson,", lesson),
        _closing("A newer or larger network is not automatically better. Test it on the same data with the same test, and count the cost of "
                 "training as well as the score.", "section 7")])


def html(R, T):
    return STYLE + '<div class="g3">\n' + "\n".join(part(R, T) for part in (_step10, _step11, _step12, _step13)) + "\n</div>"
