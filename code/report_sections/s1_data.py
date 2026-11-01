"""Section 1 - the data, the 12 basins, the forecast set-up, the four sets of inputs and how the test is kept fair.

Reads results/data_audit.json, basin_audit.csv, baselines.json (thresholds, high_flow), fair_test.json and
engine_review.json. Nothing is computed here beyond counting, differences and formatting.

The page of this section opens with an introduction without a section number ('Why this question, and what is new here'),
two short paragraphs: why the question matters and what published work found; what this project adds. It holds no result
number and cites only through cite(). The forced page break stands in front of that block, not in front of the heading of section 1.
Who is in the team and whether a member's own check is done is read through code/team.py (team.json, filled in by the team).
"""
import re

import pandas as pd

import team
from common import AQUA, AXIS, BLUE, GRID, INK, INK2, ORANGE, SURFACE
from flood import FORCING, HIGH_Q, LAND_COL, LAND_SHORT, LEADS, PERIOD, SHORT, SPLIT, STATIC
from report_common import FONT, INPUT_NAME, LEAD_WORD, cite, count, fig, n2, pct, table
from report_sections.s0_summary import LAYOUT, _day, review_sentence

FORCING_WORD = {"prcp": "rain", "srad": "sunlight", "tmax": "highest temperature", "tmin": "lowest temperature", "vp": "humidity", "dayl": "day length"}
PART_WORD = {"train": "training years", "val": "validation years", "test": "test years"}
PART_USE = {"train": ["fit: model weights, high-flow levels and scalers", "(the mean and spread that put every input on one scale)"],
            "val": ["choose: each model's settings", "and where training stops"], "test": ["score only: nothing", "is chosen from them"]}
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _word(n):
    return NUMBER.get(n, str(n))


def _zero(v, d=3):
    """Like n2, but a value that rounds to zero is printed without a sign."""
    return n2(0.0 if round(v, d) == 0 else v, d)


def _years(part):
    a, b = SPLIT[part]
    return f"{a[:4]}&ndash;{b[:4]}"


def _txt(x, y, s, anchor="middle", size=9.6, weight=400, fill=INK2):
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>'


def _svg(w, h, body):
    return f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}" role="img">{body}</svg>'


def _timeline():
    """The forecast set-up: what is read up to the end of day t, and the three days that are forecast."""
    past, fut, x0 = 31, 68, 6                                   # width of a past-day cell and a forecast-day cell
    now = x0 + 4 * past                                         # the end of day t
    end = now + len(LEADS) * fut
    s = [_txt(now, 11, "the forecast is made at the end of day t", fill=INK, weight=600)]
    for i, lab in enumerate(["&hellip;", "t&minus;2", "t&minus;1", "t"]):
        s.append(_txt(x0 + (i + .5) * past, 30, lab))
    for i, k in enumerate(LEADS):
        s.append(_txt(now + (i + .5) * fut, 30, f"t+{k}"))
    bar = lambda x, y, w, col, dash="": f'<rect x="{x}" y="{y}" width="{w}" height="19" rx="3" fill="{col}" fill-opacity=".16" stroke="{col}" stroke-width="1.2"{dash}/>'
    s += [bar(x0, 37, now - x0 - 3, BLUE), _txt((x0 + now) / 2, 50.5, "past weather", fill=INK),
          bar(x0, 61, now - x0 - 3, AQUA), _txt((x0 + now) / 2, 74.5, "river gauge", fill=INK),
          bar(now + 3, 37, end - now - 3, ORANGE, ' stroke-dasharray="4 3"'), _txt((now + end) / 2 + 1.5, 50.5, "perfect weather forecast", fill=INK),
          _txt(now - 6, 104.5, "river flow to forecast &rarr;", anchor="end")]
    for i, k in enumerate(LEADS):
        s.append(f'<rect x="{now + 3 + i * fut}" y="91" width="{fut - 5}" height="20" rx="3" fill="{SURFACE}" stroke="{INK}" stroke-width="1.2"/>')
        s.append(_txt(now + 3 + i * fut + (fut - 5) / 2, 104.3, LEAD_WORD[k], size=8.8, fill=INK, weight=600))
    s.append(f'<line x1="{now}" y1="17" x2="{now}" y2="116" stroke="{INK}" stroke-width="1.4" stroke-dasharray="2 3"/>')
    return _svg(end + 6, 120, "".join(s))


def _split_bar():
    """The three periods to scale, with the one job of each."""
    x0, w = 4, 692
    days = {p: (pd.Timestamp(b) - pd.Timestamp(a)).days + 1 for p, (a, b) in SPLIT.items()}
    total = sum(days.values())
    fill = {"train": GRID, "val": AXIS, "test": INK2}
    s, x, edge = [], x0, {}
    for p in SPLIT:
        wp = w * days[p] / total
        s.append(f'<rect x="{x:.1f}" y="17" width="{wp:.1f}" height="17" fill="{fill[p]}" stroke="{SURFACE}" stroke-width="1.5"/>')
        edge[p] = x
        x += wp
    stamp = lambda d: f"{pd.Timestamp(d):%b %Y}"
    s += [_txt(x0, 11, stamp(SPLIT["train"][0]), anchor="start", size=9.6), _txt(edge["val"], 11, stamp(SPLIT["val"][0]), size=9.6),
          _txt(edge["test"], 11, stamp(SPLIT["test"][0]), size=9.6), _txt(x0 + w, 11, stamp(SPLIT["test"][1]), anchor="end", size=9.6)]
    for p in SPLIT:
        px, anchor = (x0 + w, "end") if p == "test" else (edge[p] + 1, "start")
        s.append(_txt(px, 49, f"{PART_WORD[p]} &middot; {round(days[p] / 365.25)} years", anchor=anchor, size=10.6, fill=INK, weight=600))
        for j, line in enumerate(PART_USE[p]):
            s.append(_txt(px, 62.5 + 12.5 * j, line, anchor=anchor, size=10.4))
    return _svg(x0 + w + 4, 80, "".join(s))


LOBO_REF, TIMING_REF = "section 4.4", "section 4.5"          # where the unseen-river test and the rain time stamps are reported
CLIMATE_WORD = {"p_mean": "mean rain", "aridity": "dryness", "p_seasonality": "rain season"}       # basin properties that CAMELS averages over 1989-2009
TIGHT = "<style>.s1t{font-size:8.4pt} .s1t th,.s1t td{padding:1.3pt 4pt;line-height:1.25}</style>"                     # layout only: closer rows in this section's tables
SHOW_TIMELINE = True                                          # the what-is-read-when diagram beside the inputs text (Figure 1.2)
SHOW_SPLIT_BAR = True                                         # the three periods drawn to scale, each with its job (Figure 1.3); False: one sentence instead
MAP_SHARE = 46                                                # % of the text width for the location map; below about 45 its legend prints under 6.5 pt


def _and(parts):
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def html(R, T):
    A, HF, THR, FT, ER = R["data_audit"], R["baselines"]["high_flow"], R["baselines"]["thresholds"], R["fair_test"], R["engine_review"]
    C = T["basin_audit"]
    sel = sorted(A["selected"], key=lambda b: (list(LAND_SHORT).index(b["landscape"]), b["area_gages2"]))
    n_sel = len(sel)
    top5 = pct(1 - HIGH_Q)                                      # "high flow" = the top 5% of training-year days
    name = lambda code: INPUT_NAME[code].replace("perfect forecast", "perfect weather forecast")

    # ---- the data and the selection --------------------------------------------------------------------------------
    start, stop = pd.Timestamp(PERIOD[0]), pd.Timestamp(PERIOD[1])
    n_years = round(((stop - start).days + 1) / 365.25)
    n_days = int(C.n_days.max())
    weather = ", ".join(FORCING_WORD[c] for c in FORCING)
    rej = C[~C.complete]
    gap_cause = "all because of gaps in the flow record" if int(C.forcing_missing.sum()) == 0 else "because of gaps in the flow or the weather record"
    per_land = {land: sum(b["landscape"] == land for b in sel) for land in LAND_SHORT}
    each = f"{_word(next(iter(per_land.values())))} in each" if len(set(per_land.values())) == 1 else ", ".join(f"{v} in the {LAND_SHORT[k].lower()}" for k, v in per_land.items())
    lands = "; ".join(f"{LAND_SHORT[k].lower()} {A['complete_by_landscape'][k]} of {A['candidates_by_landscape'][k]}" for k in LAND_SHORT)
    est = [b["estimated_share"] for b in sel]
    land_def = "; ".join(k[0].lower() + k[1:].replace("32.5N", "32.5&deg;N") for k in LAND_SHORT)        # the full names carry the definitions

    data_text = f"""
<p><b>The data.</b> CAMELS-US {cite('newman2015')} is a public data set of {count(A['n_all_camels'])} river basins in the United States; a <i>basin</i> is the
land whose rain drains to one river gauge. For every basin and day it gives the
weather averaged over the basin ({weather}; from Daymet, a grid interpolated from weather stations; its rain is the <i>rain table</i> of this report) and the river
flow measured at a gauge by the US Geological Survey. The project uses {n_years} years, {start.day} {start:%B %Y} to {stop.day} {stop:%B %Y}: {count(n_days)} days per basin.
Flow is in mm/day: the water passing the gauge in a day, spread over the basin area, so that rain and flow share one unit.</p>
<p><b>The {n_sel} basins were chosen by a rule, not by hand.</b> The candidates are the {count(A['n_candidates'])} CAMELS-US basins of two south-eastern river regions that
are humid and fed by rain (snow is at most {pct(C.frac_snow.max())} of precipitation), {count(C.area_gages2.min())} to {count(C.area_gages2.max())} km&sup2;.
<i>Step 1:</i> keep only basins without a missing day; this rejects {count(A['n_with_gaps'])} candidates, {gap_cause}. <i>Step 2:</i> sort the
{count(A['n_complete'])} complete basins into three landscapes &ndash; {land_def} &ndash; and take {each}, evenly spaced in rank from the smallest to the
largest area. No flow statistic and no model score enters the rule. {pct(min(est), 1)} to {pct(max(est), 1)} of the days in the {n_sel} basins are marked
as estimated, not measured; they were kept.</p>"""

    # ---- the 12 basins: map on the left, table on the right, the caveat below -----------------------------------------
    dot = lambda land: f'<span style="color:{LAND_COL[land]}">&#9679;</span>&nbsp;'                    # the landscape colours of the map
    rows = [[dot(b["landscape"]) + SHORT[b["gauge_id"]].replace(" ", "&nbsp;"), count(b["area_gages2"]), n2(b["p_mean_mm"]), n2(b["q_mean_mm"]), n2(THR[b["gauge_id"]]),
             pct(HF[b["gauge_id"]]["share_test_days_above"], 1)] for b in sel]
    order = _and(f"{_word(per_land[k])} {LAND_SHORT[k].lower().rstrip('s')}" for k in LAND_SHORT) + " basins"
    test_days = (pd.Timestamp(SPLIT["test"][1]) - pd.Timestamp(SPLIT["test"][0])).days + 1
    share_test = {b["gauge_id"]: HF[b["gauge_id"]]["share_test_days_above"] for b in sel}
    share_train = sum(HF[g]["share_train_days_above"] for g in share_test) / n_sel
    mean_test = sum(share_test.values()) / n_sel
    hi_days = {g: round(v * test_days) for g, v in share_test.items()}
    n_below = sum(v < share_train for v in share_test.values())
    where = f"all {n_sel}" if n_below == n_sel else f"{n_below} of {n_sel}"
    events = f"{count(sum(hi_days.values()))} high-flow days in all, {count(min(hi_days.values()))} to {count(max(hi_days.values()))} per basin"
    if mean_test < share_train:
        rare_head = "High flows were rarer in the test years."
        rare_tail = f"so the warning scores in later sections rest on fewer events than {top5} suggests: {events}"
    else:
        rare_head = "High flows were not rarer in the test years."
        rare_tail = f"and the warning scores in later sections rest on {events}"
    val_share = [HF[g]["share_val_days_above"] for g in share_test]
    uneven = max(val_share) - min(val_share) > 2 * (max(share_test.values()) - min(share_test.values()))
    val_tail = ": a far less even sample, so settings chosen on those years are fair but noisy" if uneven else ""
    basin_table = table(["Basin", "Area (km&sup2;)", "Rain", "Flow", "High&#8209;flow level", "Test&nbsp;days above"], rows, cls="num s1t",
                        caption=f"<b>Table 1.1. The {n_sel} basins.</b> From the top: {order} (dots as in the map), each group by area. Rain, flow and level in mm/day (rain and flow: means of the "
                                f"{n_years} years). Last column: share of test-year days above the level.",
                        widths=["33%", "13%", "10%", "10%", "15%", "19%"])
    rare_note = f"""<div class="note keep"><b>{rare_head}</b> A <i>high-flow day</i> is a day with flow above the level that was exceeded on {top5} of days in the training years
({_years('train')}); each basin has its own level. In the test years ({_years('test')}) the flow was above it on {pct(mean_test, 1)} of days on average, and on fewer than
{pct(share_train)} in {where} basins, {rare_tail}. In the validation years ({_years('val')}) the share ranges from {pct(min(val_share), 1)} to
{pct(max(val_share), 1)}{val_tail}.</div>"""
    basins = f"""{data_text}
<div class="two blk" style="grid-template-columns:{MAP_SHARE}% 1fr">{fig("overview_map.png",
        f"<b>Figure 1.1. Where the {n_sel} basins are.</b> Colour shows the landscape. Candidates with a complete record: {lands}.")}<div>{basin_table}</div></div>"""

    # ---- forecast set-up and the four sets of inputs ---------------------------------------------------------------
    inputs = f"""<i>Past weather</i> is the weather table up to day t; the <i>river gauge</i> is the observed flow up to day t. The <i>perfect weather forecast</i> is the
weather recorded for the days being forecast. Nobody can have it in practice: it is an upper bound that shows what the weather of those days is worth. {LEAD_WORD[LEADS[0]].capitalize()}, part of that worth is rain that had already
fallen but was not yet in the model&rsquo;s inputs ({TIMING_REF}). The later sections compare four sets of inputs: {name("w")}; {name("wq")}; and each of the two again with
the perfect weather forecast. Only the first two can be had in practice."""
    convention = f"""Every model in this report makes the same kind of forecast. At the end of day t it reads the data up to and including day t and gives the river flow of
each of the next {_word(len(LEADS))} days: {", ".join(LEAD_WORD[k] for k in LEADS)} (the {_word(len(LEADS))} <i>lead times</i>). The models differ in what they read."""
    if SHOW_TIMELINE:
        setup = f"""
<h3>The forecast set-up and the four sets of inputs</h3>
<p>{convention}</p>
<div class="two blk" style="grid-template-columns:47% 1fr">
<figure>{_timeline()}<figcaption><b>Figure 1.2. What a model may read (bars) and what it must forecast (boxes).</b> Dashed: the weather recorded for the days being
forecast, read only by the upper-bound models.</figcaption></figure>
<p>{inputs}</p>
</div>"""
    else:
        setup = f"""
<h3>The forecast set-up and the four sets of inputs</h3>
<p>{convention} {inputs}</p>"""

    # ---- the fair test ---------------------------------------------------------------------------------------------
    exp, mea = FT["expected"], FT["measured"]
    e_past, e_full = exp["past_weather_only"], exp["with_target_day_weather"]
    later = n2(e_past[1]) if len(set(e_past[1:])) == 1 else " and ".join(n2(v) for v in e_past[1:])
    full = n2(min(e_full))
    measured = lambda key: _and(_zero(v) for v in mea[key])
    passed = f"All {len(FT['checks'])} checks passed." if FT["passed"] else f"Not all of the {len(FT['checks'])} checks passed."
    climate = [CLIMATE_WORD[c] for c in STATIC if c in CLIMATE_WORD]
    strictly = (f" Strictly, {_word(len(climate))} of them ({_and(climate)}) are climate averages over years that overlap the validation and test years "
                f"(section 4.1)." if climate else "")
    if SHOW_SPLIT_BAR:
        split_text = (f""":</p>\n<figure>{_split_bar()}<figcaption><b>Figure 1.3. The fair test: three periods, one job each.</b> The {n_years} years, to scale: training years ({_years('train')}), validation years ({_years('val')}), """
                      f"""test years ({_years('test')}).</figcaption></figure>""")
    else:
        split_text = (f""". The <i>training years</i> ({_years('train')}) fit the model weights, the high-flow levels and the scalers (the mean and spread that put every """
                      f"""input on one scale). The <i>validation years</i> ({_years('val')}) choose each model&rsquo;s settings and where training stops. The <i>test """
                      f"""years</i> ({_years('test')}) are only scored: nothing is chosen from them.</p>""")
    cnt = ER["counts"]
    crit = "no critical problem" if cnt["critical"] == 0 else f"{cnt['critical']} critical problem{'s' if cnt['critical'] > 1 else ''}"
    distinct = cnt.get("minor_distinct", cnt["minor"])
    twice = f" ({count(distinct)} distinct)" if distinct != cnt["minor"] else ""
    sentence = lambda s: s[0].upper() + s[1:].replace(" - ", " &ndash; ") + ("" if s.endswith(".") else ".")
    if ER["all_fixed"]:
        fixed = sentence(ER["fixed_before"]) if ER.get("fixed_before") else f"All findings were fixed before the experiments they affect were run (the review was made on {ER['when']})."
        # what 'fixed' means for a finding that is a limit of the measurements, not a bug (key all_fixed_note, added by the project lead): the code
        # findings were fixed; the timing finding was dealt with by logging the load and by quoting rough ratios, NOT by a clean re-timing
        note = str(ER.get("all_fixed_note", ""))
        if "not a bug" in note and fixed.startswith("All findings were fixed"):
            rest = note.split(". ", 1)[1] if ". " in note else note                        # the sentence(s) after 'Every finding in the code was fixed.'
            fixed = (fixed.replace("All findings were fixed", "Every finding in the code was fixed", 1) + " "
                     + sentence(rest.replace("(report section", "(section")))
    else:
        fixed = f"Not all findings are fixed yet (the review was made on {ER['when']})."
    fixed = re.sub(r"\d{4}-\d{2}-\d{2}", lambda m: _day(m.group(0)), fixed).replace("leave-one-basin-out", "unseen-river")       # formatting only
    human = ER["human_review"].replace(" - ", " &ndash; ") + ("" if ER["human_review"].endswith(".") else ".")       # starts 'not yet done: ...'
    for m in team.load()["members"]:                               # a member's name in place of the role of the plan, once team.json gives one
        human = human.replace(m["role"], team.label(m["role"]))
    if team.checks_state()["n_done"] == 0:                         # no member has entered a finished check in team.json: today's sentence
        own_check = f"<b>The team&rsquo;s own check</b> is {human}"
    else:                                                          # the state of the members' checks, in the words every document uses
        done = team.checks_sentence()
        own_check = (f"<b>The team&rsquo;s own checks.</b> {done[0].upper() + done[1:]}{'' if 'section 9' in done else ' (section 9)'}. The check of "
                     f"the scoring and of the fair-test split, which the plan gives to {team.label('GNSS member 1')}: {team.check_status('GNSS member 1')}.")
    majors = "; ".join(f"({i}) {m.split(': ')[0][0].lower() + m.split(': ')[0][1:]}" for i, m in enumerate(ER["major_findings"], 1))       # the problem, not its fix
    effect = ER["effect_on_results"].split(";")[0].replace("by the author", "by the AI session that wrote the code")      # the first clause: what changed in the results
    ai_notes = " ".join(n for n in ER["notes"] if "Each experiment script was additionally checked" not in n) + f" {review_sentence()}."      # the stored claim about every script is replaced by REVIEW_WORDING
    fair = f"""
<h3>How the test is kept fair</h3>
<p>A test score can be believed only if nothing from the scored years reached the model. Two rules, written once in the shared code (<code>code/flood.py</code>), hold in
every experiment. A forecast belongs to the period of the day it <i>forecasts</i>, not of the day it was made. And each period has one job{split_text}
<p>The catch: all {n_sel} basins are in all three periods, so the test measures skill in <i>later years of rivers the model has seen</i>. A river it has never seen is a
different test ({LOBO_REF}). The {len(STATIC)} fixed basin properties (such as area, slope and forest cover) that a model trained on all basins also reads contain nothing
computed from the flow record.{strictly}</p>
<p><b>Check 1: made-up data with a known answer</b> (<code>code/test_flood.py</code>). Forecasts are scored with the Nash-Sutcliffe efficiency (NSE) {cite('nash1970')}: 1 is
a perfect forecast, 0 is no better than always forecasting the average flow, below 0 is worse. The script builds rivers that obey a known rule:
<code>{FT['rule']}</code>. An honest model that reads the weather up to day t can then reach an NSE of at most {n2(e_past[0])} {LEAD_WORD[LEADS[0]]} and {later}
{LEAD_WORD[LEADS[1]].replace(" ahead", "")} and {LEAD_WORD[LEADS[2]]}; with the perfect weather forecast, {full}. A higher score would mean the model reads
data it should not. Measured: {measured('past_weather_only')}; with the perfect weather forecast {measured('with_target_day_weather')}. The script also
corrupts all data after the training years and confirms that the high-flow levels and scalers do not change. {passed}</p>
<p><b>Check 2: independent review of the shared code.</b> The reviewers were {ER['who']}. They report {count(cnt['checks_passed'])} checks passed, {crit}, {count(cnt['major'])} major problems and {count(cnt['minor'])} minor
ones{twice}. The major ones: {majors}. {fixed} Effect on the results: {effect}. {ai_notes}
{own_check}</p>"""

    # ---- why this question, and what is new here: an introduction without a section number, at the top of the page of section 1 ----
    # No result number stands here. What the cited studies do is the reading recorded in results/published_scores.json (section 4.3 says who read
    # them); the sentence on the operational system is the one section 8 uses.
    sets = [s for s in ("w", "wq", "wf", "wqf") if s in INPUT_NAME]                 # the four sets of inputs of this report
    upper = [s for s in sets if s.endswith("f")]                                    # those that also hold the perfect weather forecast
    usual = f"({cite('kratzert2018')[1:-1]}; {cite('kratzert2019')[1:-1]})"
    intro = f"""<h3 class="pb" style="margin-top:0">Why this question, and what is new here</h3>
<p>A flood warning is worth most before the river rises: a day or more of notice is what lets people act. So the test of a forecast is not its
average error but whether it sees the rise coming. LSTMs, neural networks with a memory (section 3), are the usual data-driven model for daily
river flow: on hundreds of US basins they estimate the flow from the weather well {usual}, also for rivers without a gauge
{cite('kratzert2019pub')}. Most of these studies estimate the flow of a day from the weather up to and including that day; an operational system
feeds weather forecasts to an LSTM {cite('nearing2024')}.</p>
<p>This project asks the forecast question on a small scale: {n_sel} rivers, one laptop, forecasts that read nothing after the day they are
issued, simple models as the number to beat, and warnings judged by criteria fixed in advance. Its design separates two things that are easy to
mix up, what the kind of model adds and what the information adds: the same network is given {_word(len(sets))} sets of inputs,
{_word(len(upper))} of them upper bounds that cannot be had in practice, and the same inputs are given to several kinds of model.</p>"""

    # Layout: the note on the high-flow days stands after the set-up block (not directly under Table 1.1), so that the block with Figure 1.2
    # still fits on the page that now opens with the introduction; placed before it, the block moved to the next page and left a gap.
    return f"""{LAYOUT}{TIGHT}{intro}<h2>1 Data and the fair test</h2>
{basins}{setup}
{rare_note}{fair}"""
