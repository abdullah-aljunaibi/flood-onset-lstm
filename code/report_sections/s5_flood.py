"""Report section 5 - could the high-flow warnings be trusted, and where and why do they fail?

Reads results/flood_analysis.json and results/flood_events.csv (written by 06_flood_analysis.py) and includes its four figures.
Nothing is computed here beyond differences, counts, smallest/largest and formatting; every sentence whose wording depends on
an outcome is chosen from the stored numbers, and the verdict criteria are read from verdict.criteria of the results file.

Layout: the figures are drawn for a width of 170 mm, so each one is shown at about that width or left out, never shrunk.
flood_skill.png and flood_failures.png are always shown, and so is the example event (flood_event.png, SHOW_EVENT): the one
picture in this section of a forecast against the observed river. SHOW_TRADEOFF adds the warning-level curves,
SHOW_SCORES_TABLE the table of scores (its numbers at the high-flow level are those of flood_skill.png). Figure and table
numbers follow the switches. Each figure is a float placed right after the paragraph that first names it; the heading
'The verdict', its criteria and the verdict table are one unbreakable block (.keep) that starts below any figure; it stands first, before the scores.

Rule for the wording of rain: the models without a weather forecast miss onsets because of rain that was not yet in their
inputs - not because the rain 'had not fallen yet'. The note box ties this to what section 4.5 measured (the share of the gap
in onsets warned that rain stamped up to the evening closes), and the last bullet says what a warning system would need: a
weather forecast at every lead time; rain observations up to the hour add a part 1 day ahead, and with them the onsets are
still judged by the same criteria (s0_summary.timing_verdict). The verdict star is computed from verdict.services.*.single_seeds:
beside it stands how many single seeds get the verdict of the seed average.
"""
import math

import pandas as pd

from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed, table, tag
from report_sections.s0_summary import LAYOUT, _word, criteria, floated, forecast_where, onset_clear, share_word, timing_facts, timing_verdict

LEADS = sorted(LEAD_WORD)
KINDS = ("all", "onset")
KIND_WORD = {"all": "all high-flow days", "onset": "onsets"}
QUOTED = "&lsquo;tomorrow = today&rsquo;"
SCORED = ("base_persistence", "base_rf_wq", "lstm_wq", "lstm_wqf")                 # the rows of Table 5.1: the four models of Figures 5.1 and 5.2
NAME = {"base_persistence": "&lsquo;Tomorrow = today&rsquo;", "base_rf_wq": "Random forest with river gauge",
        "lstm_wq": "LSTM with river gauge", "lstm_wqf": "LSTM with river gauge and perfect weather forecast (upper bound)"}
TODAY, GAUGE, UPPER = ("lstm_w", "lstm_wq"), "lstm_wq", "lstm_wqf"                 # what could run today; the system with the gauge; the upper bound
SERVICES = ("lstm_w", "lstm_wq", "lstm_wqf")                                       # the rows of the verdict table
SYSTEM = {"lstm_w": "Past weather only", "lstm_wq": "Past weather + river gauge", "lstm_wqf": "Past weather + river gauge + perfect weather forecast (upper bound)"}
THE = {"lstm_w": "the LSTM with past weather only", "lstm_wq": "the LSTM with the river gauge",
       "lstm_wqf": "the LSTM with the river gauge and a perfect weather forecast"}
RANK = {"NO EVENTS": -1, "NOT YET": 0, "CAUTION": 1, "GO": 2}
STORMS = ("tropical", "warm-season", "cool-season")                                 # classes of 06_flood_analysis.py
STATUS = {"HU": "hurricane", "TS": "tropical storm", "TD": "tropical depression", "EX": "former tropical cyclone", "LO": "remnant low",
          "SS": "subtropical storm", "SD": "subtropical depression"}                # HURDAT2 status codes
RAIN_BINS = ("0-25%", "25-50%", "50-75%", "75-100%")                                # classes of 06_flood_analysis.py, in the order of Figure 5.3
SHOW_LEVEL = 0.70                                                                   # the lower warning level quoted as an example (a point of the stored curve)
FIG_SKILL, FIG_TRADE, FIG_FAIL, FIG_EVENT = 84, 84, 84, 84                          # % of the text width (184 mm); the figures are drawn for 170 mm (92%)
SHOW_SCORES_TABLE = False                       # True adds the table of scores at both warning levels (about a quarter of a page)
SHOW_TRADEOFF = False                           # True adds flood_tradeoff.png (about half a page); its key numbers are in the text either way
SHOW_EVENT = True                               # flood_event.png (about 0.6 page); False: the example is one sentence with its numbers
TIMING_REF = "section 4.5"                      # where the rain time stamps are tested
lk = lambda k: f"l{k}"
dash = lambda s: s.replace("-", "&ndash;")
num = lambda v: "&ndash;" if v != v else f"{100 * v:.0f}"                           # a share as a percentage without the sign (the unit is in the header)


def _and(items):
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _leads(ks):
    """[1] -> '1 day ahead'; [2, 3] -> '2 and 3 days ahead'."""
    ks = sorted(ks)
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _p(p):
    return "p &lt; 0.001" if p < 0.001 else f"p = {n2(p, 3 if p < 0.1 else 2)}"


def _date(s):
    t = pd.Timestamp(s)
    return f"{t.day} {t:%B %Y}"


def _years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def _where(avg, lo, hi, other):
    """Where another model's score lies against the LSTM: its averaged forecast (avg) and its single seeds (lo..hi)."""
    lo, hi = min(lo, avg), max(hi, avg)
    return "above" if other > hi else "below" if other < lo else "within"


def html(R, T):
    A, E = R["flood_analysis"], T["flood_events"]
    T_VERDICT, SCORES_REF = ("5.2", "Table 5.1") if SHOW_SCORES_TABLE else ("5.1", "Figure 5.1")      # numbers follow the switches
    shown = [name for name, on_ in (("skill", True), ("trade", SHOW_TRADEOFF), ("fail", True), ("event", SHOW_EVENT)) if on_]
    fno = {name: f"Figure 5.{i}" for i, name in enumerate(shown, 1)}
    S, X, G, V = A["skill"], A["failures"], A["warning_levels"], A["verdict"]["services"]
    split, n_basins, ev = A["split"], len(A["basins"]), A["events"]
    first = LEADS[0]
    plain = lambda n, k, kind: S[n]["plain"]["test"]["summary"][lk(k)][kind]        # warning level = the high-flow level itself
    tuned = lambda n, k, kind: S[n]["tuned"][lk(k)]["test"][kind]                   # warning level chosen on the validation years
    level = lambda n, k: S[n]["tuned"][lk(k)]["trigger"]
    n_days = plain(GAUGE, first, "all")["hits"] + plain(GAUGE, first, "all")["misses"]
    n_on = {k: plain(GAUGE, k, "onset")["hits"] + plain(GAUGE, k, "onset")["misses"] for k in LEADS}
    later = n_days - n_on[first]                                                     # high-flow days on which the river was already high the day before

    # ---- the criteria, exactly as stored ---------------------------------------------------------------------------------
    crit = criteria(R)                                                              # parsed once, in s0_summary (page 1 quotes them too)
    go_pod, go_far, ca_csi = crit["go_pod"], crit["go_far"], crit["ca_csi"]

    # ---- 5.1 what a warning is --------------------------------------------------------------------------------------------
    intro = f"""
<p>Had a warning been given each time a forecast was high, could people have relied on it &ndash; and where it failed, why? High-flow days,
share warned, false warnings and onsets are as defined in sections 1 and 2. A model warns for a day when its forecast for that day is above a
<i>warning level</i>: the high-flow level itself, or a multiple of it ({n2(G[0])} to {n2(G[-1])}) that the validation years choose for each model
and lead time by the highest <i>critical success index</i> (CSI in the figures): hits / (hits + misses + false warnings), 1 for a perfect record.
All scores are for the test years, the {n_basins} basins counted together.</p>"""
    two_ways = f"""
<p><b>Two ways of scoring.</b> <i>All high-flow days</i>: {count(n_days)} in the test years, but {count(later)} of them ({pct(later / n_days)}) are
the second or a later day of a high-flow event, when the gauge already shows that the river is high. <i>Onsets only</i>: the warning that matters.
{LEAD_WORD[first].capitalize()} the onsets are the first days of the {count(ev['n_events'])} high-flow events, {count(ev['one_day_events'])} of
which last a single day.</p>"""

    # ---- 5.2 the scores: Figures 5.1 and 5.2 with their text beside them, then Table 5.1 --------------------------------------
    n_seeds = len(A["models"][GAUGE]["seeds"])
    g_all, g_on, u_on = plain(GAUGE, first, "all"), plain(GAUGE, first, "onset"), plain(UPPER, first, "onset")
    sd = lambda kind: S[GAUGE]["seeds"][lk(first)]["plain"][kind]
    no_fc = [n for n in SCORED if n != UPPER]                                       # the three models without a weather forecast
    best_on = max(plain(n, k, "onset")["pod"] for n in no_fc for k in LEADS)
    p_all, r_all, r_on = plain("base_persistence", first, "all"), plain("base_rf_wq", first, "all"), plain("base_rf_wq", first, "onset")
    word = {"above": "more than any LSTM seed", "below": "less than any LSTM seed", "within": "within the LSTM&rsquo;s seed spread: no clear difference"}
    side = lambda kind, other: word[_where(plain(GAUGE, first, kind)["pod"], sd(kind)["pod_min"], sd(kind)["pod_max"], other)]
    scores = f"""
<p><b>What the scores say</b> ({fno['skill']}; at the high-flow level itself, while Table {T_VERDICT} uses the level chosen on the validation years). {LEAD_WORD[first].capitalize()} the LSTM with the river gauge warns of {pct(g_all['pod'])} of all
high-flow days ({pct(sd('all')['pod_min'])} to {pct(sd('all')['pod_max'])} for its single seeds), with {pct(g_all['far'])} false warnings. The random forest with the same inputs warns of {pct(r_all['pod'])} ({side('all', r_all['pod'])}), and
{QUOTED}, which only repeats the gauge reading, of {pct(p_all['pod'])} ({side('all', p_all['pod'])}). Onsets are another matter: the same LSTM
warns of {_and(pct(plain(GAUGE, k, 'onset')['pod']) + ' ' + LEAD_WORD[k] for k in LEADS)}, with {pct(g_on['far'])} false warnings
{LEAD_WORD[first]}; the random forest warns of {pct(r_on['pod'])} ({side('onset', r_on['pod'])}).
Only the upper bound, the same LSTM given the perfect weather forecast, warns of most onsets: {pct(u_on['pod'])}, with {pct(u_on['far'])} false
warnings.</p>"""
    in_green = lambda n: any(plain(n, k, kind)["pod"] >= go_pod or plain(n, k, kind)["far"] <= go_far for k in LEADS for kind in KINDS)
    reach = {(kind, c): any(plain(UPPER, k, kind)[c] >= go_pod if c == "pod" else plain(UPPER, k, kind)[c] <= go_far for k in LEADS)
             for kind in KINDS for c in ("pod", "far")}                                             # the four panels: does the upper bound reach the green?
    up_on = [plain(UPPER, k, "onset")["pod"] for k in LEADS]
    below = (f" &ndash; in the other three panels; bottom left it stops just below it ({pct(min(up_on))} to {pct(max(up_on))} of the onsets warned, "
             f"where GO asks for {pct(go_pod)})" if not reach["onset", "pod"] and sum(reach.values()) == 3 else "")
    green = ("only the dashed line, the upper bound, reaches the green" + below if in_green(UPPER) and not any(in_green(n) for n in no_fc) else
             "the green is not reached by the upper bound alone")
    caption1 = (f"<b>{fno['skill']}. Warnings at the high-flow level, test years.</b> Left: share warned; right: false warnings. Green: what GO asks for "
                f"(Table {T_VERDICT}). What to see: bottom left, none of the three models that read the rain "
                f"table only up to the day the forecast is made warns of more than {pct(best_on)} of the onsets; {green}.")

    others = A["verdict"]["other_models_pooled"]
    v1 = lambda n, kind: V[n][lk(first)][kind]["pooled"] if n in V else others[n][lk(first)][kind]
    best_csi = {n: max(c for c in S[n]["tradeoff_test"][lk(first)]["onset"]["csi"] if c == c) for n in no_fc}
    left = ("only the marker of the upper bound is in the green GO box; the three others are above the dotted CAUTION line"
            if v1(UPPER, "all") == "GO" and all(v1(n, "all") == "CAUTION" for n in no_fc) else
            "the verdicts are " + _and(f"{v1(n, 'all')} for {NAME[n]}" for n in SCORED))
    right = ("the three curves without a weather forecast never reach the dotted line" if max(best_csi.values()) < ca_csi else
             "not every curve without a weather forecast stays below the dotted line")
    caption2 = (f"<b>{fno.get('trade', '')}. The price of a lower warning level, {LEAD_WORD[first]}, test years.</b> Along a curve the warning level falls from "
                f"{n2(G[-1], 1)} times the high-flow level (lower left) to {n2(G[0], 1)} times (upper right). Large marker: the level "
                f"chosen on the validation years. What to see: for all high-flow days, {left}; for onsets, {right}.")
    both = lambda n, k, kind, c: plain(n, k, kind)[c] == plain(n, k, kind)[c] and tuned(n, k, kind)[c] == tuned(n, k, kind)[c]
    moved = max(abs(tuned(n, k, kind)[c] - plain(n, k, kind)[c]) for n in SCORED for k in LEADS for kind in KINDS for c in ("pod", "far") if both(n, k, kind, c))
    pk = X["peak_size"][GAUGE]["all_events"]
    i_low, i_one = min(range(len(G)), key=lambda i: abs(G[i] - SHOW_LEVEL)), min(range(len(G)), key=lambda i: abs(G[i] - 1.0))
    tr = S[GAUGE]["tradeoff_test"][lk(first)]["onset"]
    trip = lambda values: " / ".join(values)
    g_levels = [level(GAUGE, k) for k in LEADS]
    asked = "did not ask for one" if min(g_levels) >= 1.0 else "asked for a slightly lower one at some lead times"
    low_peak = (f"The LSTM with the river gauge forecasts peaks too low (in the median event, {pct(pk['median'])} of the observed peak the day before), "
                f"so a lower level might help." if pk["median"] < 1 else
                f"In the median event the LSTM with the river gauge forecasts {pct(pk['median'])} of the observed peak the day before.")
    tradeoff = f"""
<p><b>Would another warning level help?</b> {'No. ' if best_csi[GAUGE] < ca_csi else ''}{low_peak} But the validation years {asked} (they
chose {trip(n2(v) for v in g_levels)} times the high-flow level), and at the chosen levels no score in {SCORES_REF} moves by more than
{math.ceil(100 * moved - 1e-9)} percentage points. {f"{fno['trade']} shows why" if SHOW_TRADEOFF else "In the test years"}: at {n2(G[i_low])} times
the level this LSTM would warn of {pct(tr['pod'][i_low])} of the onsets instead of {pct(tr['pod'][i_one])}, but {pct(tr['far'][i_low])} of those
warnings would be false instead of {pct(tr['far'][i_one])}.</p>"""

    two = lambda a, b: f"{a}<br><span class='cap'>{b}</span>"                        # first line: the high-flow level; second line: the chosen level
    rows = [[f"<b>{NAME[n]}</b>", two(n2(1.0), trip(n2(level(n, k)) for k in LEADS))]
            + [two(trip(num(plain(n, k, kind)[c]) for k in LEADS), trip(num(tuned(n, k, kind)[c]) for k in LEADS)) for kind in KINDS for c in ("pod", "far")]
            for n in SCORED]
    table1 = table(["Model", "Warning level<br>(&times; high-flow level)", "All days:<br>warned (%)", "All days:<br>false (%)",
                    "Onsets:<br>warned (%)", "Onsets:<br>false (%)"], rows, widths=["31%", "17%", "13%", "13%", "13%", "13%"],
                   caption=f"<b>Table 5.1. Share warned and false warnings, test years, {n_basins} basins together.</b> Three numbers: {_leads(LEADS)}. First "
                           f"line of a cell: warning at the high-flow level; second line: at the level chosen on the validation years. All days: all "
                           f"high-flow days ({count(n_days)}); warned: share warned; false: false warnings. Onsets: {_and(count(n_on[k]) for k in LEADS)} "
                           f"({_leads(LEADS[1:])} the second and third day of an event also count when the river was still low on the day of the "
                           f"forecast). The random forest and the LSTMs also read past weather. A dash: no warning was given.")

    # ---- 5.3 where and why it fails ---------------------------------------------------------------------------------------
    on = lambda part, n, c: X[part][n]["tuned"][c]["onset"]                          # onset table of one class, validation-chosen level
    seasons = list(X["by_season"][GAUGE]["tuned"])
    s_lo, s_hi = min(seasons, key=lambda c: on("by_season", GAUGE, c)["pod"]), max(seasons, key=lambda c: on("by_season", GAUGE, c)["pod"])
    every = "No season is easy" if on("by_season", GAUGE, s_hi)["pod"] < 0.5 else "The seasons differ"
    season = (f"<p>The {count(n_on[first])} onsets of the test years, {LEAD_WORD[first]}, are sorted in four ways ({fno['fail']}). <b>Season.</b> {every}: "
              f"the LSTM with the river gauge warns of {pct(on('by_season', GAUGE, s_lo)['pod'])} ({dash(s_lo)}) to "
              f"{pct(on('by_season', GAUGE, s_hi)['pod'])} ({dash(s_hi)}) of the onsets. ")

    hur = ev["hurdat"]
    st = {c: on("by_storm_type", GAUGE, c) for c in STORMS}
    n_st = {c: st[c]["hits"] + st[c]["misses"] for c in st}
    trop_best = all(st["tropical"]["pod"] > st[c]["pod"] for c in st if c != "tropical")
    storm = (f"<b>Storm type.</b> Of the {n_st['tropical']} onsets with a tropical cyclone nearby (a position of the track record HURDAT2 "
             f"{cite('landsea2013')} within {hur['radius_km']} km of the gauge on the onset day or the {hur['days_before_onset']} days before), "
             f"{pct(st['tropical']['pod'])} are warned, against {pct(st['warm-season']['pod'])} of the {n_st['warm-season']} other onsets of "
             f"May&ndash;September and {pct(st['cool-season']['pod'])} of the {n_st['cool-season']} of October&ndash;April. {'But' if trop_best else 'And'} "
             f"{n_st['tropical']} onsets are a small group ({ev['tropical_within_300km']} remain at 300 km), and a cyclone nearby is not proof that it "
             f"brought the rain.</p>")

    bt = pd.DataFrame(X["by_basin"][GAUGE]["table"]).set_index("gauge_id")
    b_lo, b_hi = bt.tuned_onset_pod.idxmin(), bt.tuned_onset_pod.idxmax()
    land = X["by_basin"][GAUGE]["median_by_landscape"]["tuned"]
    by_land = _and(pct(v["onset_pod"]) + (" of the onsets in the " if i == 0 else " in the ") + name.lower()
                   for i, (name, v) in enumerate(sorted(land.items(), key=lambda kv: kv[1]["onset_pod"])))
    rho = lambda n, attr: X["by_basin"][n]["spearman"]["tuned"][attr]["onset_pod"]
    need = X["spearman_rho_needed_for_p05"]
    r_area, r_slope, r_up = rho(GAUGE, "area_km2"), rho(GAUGE, "slope_m_per_km"), rho(UPPER, "area_km2")
    link = lambda r, a, gloss="": (f"{'rises' if r['rho'] > 0 else 'falls'} with {a} ({gloss}{signed(r['rho'])}, {_p(r['p'])})" if abs(r["rho"]) >= need
                                   else f"shows no clear link with {a} ({gloss}{signed(r['rho'])}, {_p(r['p'])})")
    basin = (f"<p class='nb'><b>Basin size and landscape.</b> The median basin warns of {by_land}. The share warned "
             f"{link(r_area, 'basin area', 'rank correlation: ')} and {link(r_slope, 'slope')}; given the perfect weather forecast it "
             f"{link(r_up, 'area')}. With only {r_area['n']} basins, in which size, slope and landscape go together, this is a pattern worth a "
             f"hypothesis, not a cause.</p>")

    rt = {c: on("by_rain_timing", GAUGE, c) for c in RAIN_BINS}
    ru = {c: on("by_rain_timing", UPPER, c) for c in RAIN_BINS}
    res = X["perfect_forecast_rescue"]["tuned"]
    ra, rc = res["all_onsets"], res["by_rain_class"]
    top, low = RAIN_BINS[-1], RAIN_BINS[0]
    n_dry = sum(v["onsets"] for c, v in rc.items() if c not in RAIN_BINS)
    caption3 = (f"<b>{fno['fail']}. Where the warnings of an onset fail, {LEAD_WORD[first]}, test years, at the warning level chosen on the validation years.</b> "
                f"Dark: onsets warned by the LSTM with the river gauge; light: by the upper bound, the same LSTM given the perfect weather forecast; "
                f"red lines: that model&rsquo;s false warnings. Bottom right: onsets sorted by the share of the two days&rsquo; rain stamped on "
                f"the target day. What to see: the dark "
                f"bars are low in every season and storm type, but fall from {pct(rt[low]['pod'])} to {pct(rt[top]['pod'])} as the rain moves to the "
                f"target day, while the light bars rise from {pct(ru[low]['pod'])} to {pct(ru[top]['pod'])}.")

    share_caught = ra["of_those_caught_with_perfect_forecast"] / ra["missed_with_gauge"]
    surprise = (f" The upper bound is not better everywhere: in the {dash(low)} class it warns of {pct(ru[low]['pod'])}, less than the {pct(rt[low]['pod'])} "
                f"without the forecast." if ru[low]["pod"] < rt[low]["pod"] else "")
    timing = f"""
<p><b>Rain timing: the main explanation</b> ({fno['fail']}, bottom right). A model without a weather forecast reads the rain table up to the day the forecast is made. For each
onset {LEAD_WORD[first]}, take the rain stamped on that day and the rain stamped on the day being forecast (the target day): what share of the two
is stamped on the target day, and so was not yet in the model&rsquo;s inputs? In {count(rc[top]['onsets'])} of the {count(ra['onsets'])} onsets that share is
{dash(top)}, and the LSTM with the river gauge warns of {pct(rt[top]['pod'])} of them. Where most of the rain is already in its inputs (share
{dash(low)}, {count(rc[low]['onsets'])} onsets) it warns of {pct(rt[low]['pod'])}. Of the {count(ra['missed_with_gauge'])} onsets it misses, the model
that is also given the weather of the target day (the perfect weather forecast, an upper bound) catches
{count(ra['of_those_caught_with_perfect_forecast'])} ({pct(share_caught)}); {count(ra['missed_by_both'])} are missed by both.{surprise}</p>"""
    tf = timing_facts(R)                                                             # the rain-timing test of section 4.5, once it has run
    tv = timing_verdict(R)                                                           # the onsets with rain up to the evening, by the criteria of this section
    if tf and onset_clear(tf["gauge"][first]):
        r1 = tf["gauge"][first]
        timing_note = (f" That does not make the missed onsets mostly knowable: given rain stamped up to the evening of the day the forecast is made, "
                       f"and no weather forecast, the LSTM warns of {pct(r1['onset_warned_with_nldas'])} of the onsets instead of "
                       f"{pct(r1['onset_warned_base'])} (control {pct(r1['onset_warned_control'])}; upper bound "
                       f"{pct(r1['onset_warned_perfect_forecast'])}), {share_word(r1['onset_share_of_gap_closed'])} of the gap.")
        left = min(1 - max(tf["gauge"][k][key] for k in LEADS[1:]) for key in ("share_of_gap_closed", "onset_share_of_gap_closed"))
        later_note = (f" For {_leads(LEADS[1:])} {'almost all' if left >= 0.9 else 'most'} of the missing rain falls after the forecast is made.")
    elif tf:
        r1 = tf["gauge"][first]
        timing_note = (f" Given rain stamped up to the evening of the day the forecast is made, the LSTM warns of {pct(r1['onset_warned_with_nldas'])} "
                       f"of the onsets instead of {pct(r1['onset_warned_base'])}; the single seeds overlap, so no gain is shown.")
        later_note = f" For {_leads(LEADS[1:])} the rain stamped on the target day falls after the forecast is made."
    else:
        timing_note = f" {TIMING_REF.capitalize()} measures how much of it rain observations stamped up to the evening of that day supply."
        later_note = f" For {_leads(LEADS[1:])} the rain stamped on the target day falls after the forecast is made."
    stamp = f"""
<div class="note blk"><b>&lsquo;Not yet in the inputs&rsquo; does not mean &lsquo;not yet fallen&rsquo;.</b> The rain table seems to stamp rain by
a day that ends in the morning (an inference from our own data; {TIMING_REF}), so part of the rain stamped on the target day had already fallen
when the forecast for {LEAD_WORD[first]} was made.{timing_note}{later_note}</div>"""

    # the example event of Figure 5.4 (picked by rule in 06_flood_analysis.py; absent when no onset fits the rule)
    e = A.get("event_shown")
    has_event = bool(e)
    show_event = SHOW_EVENT and has_event                                              # the figure; without it the example is one sentence
    e = e or {"gauge_id": "", "onset_date": "1900-01-01", "cyclone_name": "", "basin": "", "rain_target_day_mm": float("nan"),
              "rain_issue_day_mm": float("nan"), "onsets_with_this_pattern": float("nan"), "onsets": float("nan")}
    row = E[(E.gauge_id == e["gauge_id"]) & (E.onset_date == e["onset_date"])]
    miss_g = has_event and not bool(row[f"{GAUGE}_warned_l{first}_chosen_level"].iloc[0])
    hit_u = has_event and bool(row[f"{UPPER}_warned_l{first}_chosen_level"].iloc[0])
    cyclone = (f"{STATUS.get(e['cyclone_status'], 'tropical cyclone')} {e['cyclone_name'].title()}, track {count(e['cyclone_km'])} km from the gauge"
               if e["cyclone_name"] else "no tropical cyclone nearby")
    outcome = ("the LSTM with the river gauge gives no warning for the onset day, and the model with the perfect weather forecast does" if miss_g and hit_u else
               f"the LSTM with the river gauge {'gives no warning' if miss_g else 'warns'} for the onset day, and the model with the perfect weather "
               f"forecast {'warns' if hit_u else 'gives no warning'}")
    onset_day = pd.Timestamp(e["onset_date"])
    caption4 = (f"<b>{fno.get('event', '')}. One onset, day by day: {e['basin']}, {_date(e['onset_date'])} ({cyclone}).</b> Top: rain as stamped in the "
                f"rain table. Middle: observed flow (black) and the two forecasts, each drawn on the day it is for and made the evening before. Bottom: "
                f"hit, miss or false warning for each day. What to see: on {onset_day.day} {onset_day:%B} the "
                f"river crosses the high-flow level; {outcome}. The {count(e['rain_target_day_mm'])} mm stamped "
                f"on the onset day are not all rain that fell after the forecast was made: by the rain table&rsquo;s day stamp, part of a day&rsquo;s "
                f"rain has fallen before the forecast for that day is issued ({TIMING_REF}).")
    event = (f"<p><b>One example</b> ({fno.get('event', '')}), picked by rule, not a typical score: {e['basin']}, onset on {_date(e['onset_date'])} "
             f"({cyclone}). {count(e['rain_target_day_mm'])} mm of rain are stamped on the onset day and {count(e['rain_issue_day_mm'])} mm on the day "
             f"before; {outcome}. {count(e['onsets_with_this_pattern'])} of the {count(e['onsets'])} onsets went this way.</p>")
    example = (f"<p><b>One example</b>, picked by rule, not a typical score: {e['basin']}, onset on {_date(e['onset_date'])} ({cyclone}). "
               f"{count(e['rain_target_day_mm'])} mm of rain are stamped on the onset day and {count(e['rain_issue_day_mm'])} mm on the day before; "
               f"{outcome}. {count(e['onsets_with_this_pattern'])} of the {count(e['onsets'])} onsets went this way.</p>") if e["gauge_id"] else ""

    # ---- 5.4 the verdict --------------------------------------------------------------------------------------------------
    verd = lambda n, k, kind: V[n][lk(k)][kind]["pooled"]
    seeds_of = lambda n, k, kind: V[n][lk(k)][kind]["single_seeds"]                   # the verdict of every single seed
    agree = lambda n, k, kind: sum(v == verd(n, k, kind) for v in seeds_of(n, k, kind).values())      # single seeds with the verdict of the seed average
    firm = lambda n, k, kind: agree(n, k, kind) == len(seeds_of(n, k, kind))          # every single seed gets the same verdict
    starred = lambda n, k, kind: "" if firm(n, k, kind) else f" <span class='cap' style='white-space:nowrap'>* {agree(n, k, kind)} of {len(seeds_of(n, k, kind))}</span>"
    cell = lambda n, k, kind: (f"{tag(verd(n, k, kind))}{starred(n, k, kind)}<br><span class='cap'>{num(V[n][lk(k)][kind]['pod'])} &middot; "
                               f"{num(V[n][lk(k)][kind]['far'])} &middot; {n2(V[n][lk(k)][kind]['csi'])}</span>")
    rows3 = [[f"<b>{SYSTEM[n]}</b>"] + [cell(n, k, kind) for kind in KINDS for k in LEADS] for n in SERVICES]
    all_cells = [(n, k, kind) for n in SERVICES for k in LEADS for kind in KINDS]
    same_plain = sum(V[n][lk(k)][kind]["at_plain_level"] == verd(n, k, kind) for n, k, kind in all_cells)
    same_onset = sum(V[n][lk(k)][kind]["at_level_chosen_for_onsets"] == verd(n, k, kind) for n, k, kind in all_cells)
    bc = {kind: V[GAUGE][lk(first)][kind]["basin_count"] for kind in KINDS}
    per_basin = lambda c: f"GO in {c['GO']}, CAUTION in {c['CAUTION']} and NOT YET in {c['NOT YET']}"
    n_seeds_v = len(seeds_of(*all_cells[0]))
    none_same = [c for c in all_cells if agree(*c) == 0]
    star = ((f"A verdict is that of the forecast averaged over the {_word(n_seeds_v)} seeds. * beside it: how many of the {_word(n_seeds_v)} single seeds "
             f"get the same verdict, where not all do"
             + (f" (in {_word(len(none_same))} cell{'s' if len(none_same) > 1 else ''} none does)" if none_same else "") + ". ")
            if any(not firm(*c) for c in all_cells) else f"Every single seed gets the same verdict in all {len(all_cells)} cells. ")
    table3 = table(["LSTM for all basins, given"] + [f"{'All days' if kind == 'all' else 'Onsets'}:<br>{LEAD_WORD[k]}" for kind in KINDS for k in LEADS],
                   rows3, widths=["25%"] + ["12.5%"] * 6,
                   caption=f"<b>Table {T_VERDICT}. The verdict, test years, {n_basins} basins together, at the warning level chosen on the validation years.</b> "
                           f"All days: all high-flow days. Under each verdict: share warned (%) &middot; false warnings (%) &middot; critical success index. "
                           f"{star}Same verdict in {same_plain} of the {len(all_cells)} cells at the high-flow level itself. Basin by basin, "
                           f"{LEAD_WORD[first]}, with the river gauge: {per_basin(bc['all'])} basins for all high-flow days; {per_basin(bc['onset'])} for "
                           f"onsets.")

    def where(systems, want):
        """'the LSTM with the river gauge for all high-flow days 1 day ahead, ...' for the cells with the wanted verdict."""
        return [f"{THE[n]} for {KIND_WORD[kind]} {_leads(ks)}" for n in systems for kind in KINDS
                for ks in [[k for k in LEADS if verd(n, k, kind) == want]] if ks]

    go_t, ca_t = where(TODAY, "GO"), where(TODAY, "CAUTION")
    if go_t:
        trusted = f"GO for {_and(go_t)}" + (f"; CAUTION for {_and(ca_t)}." if ca_t else ".")
    elif ca_t:
        trusted = (f"Nothing without reservation: no system that could run today reaches GO in any cell of Table {T_VERDICT}. The best is CAUTION, for "
                   f"{_and(ca_t)}.")
    else:
        trusted = f"Nothing: no system that could run today reaches GO or CAUTION in any cell of Table {T_VERDICT}."
    ref = others.get("base_persistence", {}).get(lk(first), {}).get("all")
    if verd(GAUGE, first, "all") == "CAUTION" and ref:
        p_t, g_t = tuned("base_persistence", first, "all"), tuned(GAUGE, first, "all")
        trusted += (f" But {QUOTED} gets {'the same verdict' if ref == 'CAUTION' else ref} there (index {n2(p_t['csi'])} against {n2(g_t['csi'])})."
                    + (" This CAUTION says that a high river will stay high. It is not a flood warning." if ref == "CAUTION" else ""))

    g1 = V[GAUGE][lk(first)]["onset"]
    if {verd(n, k, "onset") for n in TODAY for k in LEADS} == {"NOT YET"}:
        not_trusted = (f"The warning that matters: that a river which is still low will be high. Both systems that could run today are NOT YET for onsets "
                       f"{_leads(LEADS)}: {LEAD_WORD[first]}, with the river gauge, {pct(1 - g1['pod'])} of the onsets get no warning and "
                       f"{pct(g1['far'])} of the onset warnings are false.")
    else:
        not_trusted = f"NOT YET: {_and(where(TODAY, 'NOT YET'))}."
    if all(verd("lstm_w", k, kind) == "NOT YET" for k in LEADS for kind in KINDS):
        not_trusted += " Without a river gauge every cell is NOT YET."
    if all(verd(GAUGE, k, kind) == "NOT YET" for k in LEADS[1:] for kind in KINDS):
        not_trusted += f" With the gauge, {_leads(LEADS[1:])} are NOT YET even for all high-flow days."

    up = {kind: _and(f"{v} {_leads(ks)}" for v in ("GO", "CAUTION", "NOT YET") for ks in [[k for k in LEADS if verd(UPPER, k, kind) == v]] if ks) for kind in KINDS}
    n_cells = len(LEADS) * len(KINDS)
    better = sum(RANK[verd(UPPER, k, kind)] > RANK[verd(GAUGE, k, kind)] for k in LEADS for kind in KINDS)
    shaky = [(k, kind) for kind in KINDS for k in LEADS if not firm(UPPER, k, kind)]

    def weak(k, kind):
        """One verdict of the upper bound that not every single seed reaches, said exactly."""
        c, same = V[UPPER][lk(k)][kind], agree(UPPER, k, kind)
        others = sorted({v for v in c["single_seeds"].values() if v != c["pooled"]}, key=lambda v: RANK[v])
        text = f"{c['pooled']} for {KIND_WORD[kind]} {LEAD_WORD[k]} is reached by "
        text += (f"none of the {_word(len(c['single_seeds']))} single seeds (all {_and(others)})" if same == 0 else
                 f"{_word(same)} of the {_word(len(c['single_seeds']))} single seeds")
        return text + (f" and not at the high-flow level itself ({c['at_plain_level']} there)" if c["at_plain_level"] != c["pooled"] else "")

    shaky_txt = (" The upper bound&rsquo;s " + "; its ".join(weak(k, kind) for k, kind in sorted(shaky, key=lambda x: agree(UPPER, *x))) + ".") if shaky else ""
    # what a warning system would need: a weather forecast at every lead time; rain observations up to the hour add a part 1 day ahead
    tested = forecast_where(R)                                                        # section 4.8 has tested one real forecast: say where it lands (None: the old sentence)
    if tf and tv and onset_clear(tf["gauge"][first]):
        r1 = tf["gauge"][first]
        larger = "the larger part of the gap" if r1["onset_share_of_gap_closed"] < 0.5 else "part of the gap"
        still = (f"and the onsets are still {tv['verdict']}, with {pct(tv['far'])} false warnings" if tv["verdict"] != "GO" else "and the onsets reach GO")
        need = (f"A weather forecast as an input is needed at every lead time"
                + (f": it is {larger} even {LEAD_WORD[first]}, and {tested}. " if tested else f", and it was not tested: it is {larger} even {LEAD_WORD[first]}. ")
                + f"Rain "
                f"observations up to the hour exist today and add a part {LEAD_WORD[first]}: with rain stamped up to the evening the LSTM warns of "
                f"{pct(r1['onset_warned_with_nldas'])} of the onsets instead of {pct(r1['onset_warned_base'])}, "
                f"{share_word(r1['onset_share_of_gap_closed'])} of the gap, {still} ({TIMING_REF}).")
    else:
        need = (f"A weather forecast as an input is needed at every lead time" + (f": {tested}. " if tested else ", and it was not tested. ")
                + f"Whether rain observations up to the hour add a part {LEAD_WORD[first]} is the question of {TIMING_REF}.")
    add = (f"Rain that is not yet in the model&rsquo;s inputs. Given the weather of the days being forecast (the upper bound), the same LSTM is "
           f"{up['all']} for all high-flow days, and {up['onset']} for onsets"
           f"{'' if better == n_cells else f' (better than with the gauge alone in {better} of {n_cells} cells)'}.{shaky_txt} A real weather forecast "
           f"has errors, so a real system would land between the last two rows of Table {T_VERDICT}. {need}")

    verdict = f"""
<div class="keep"><h3>The verdict</h3>
<p>The criteria were fixed before the flood analysis was run. <b>GO</b>: at least {pct(go_pod)} warned and at most {pct(go_far)} false warnings.
<b>CAUTION</b>: not GO, but a critical success index of at least {n2(ca_csi)}, that is, hits make up at least {pct(ca_csi)} of all hits, misses and
false warnings. <b>NOT YET</b>: otherwise. The three systems are the same LSTM with different inputs.</p>
{table3}</div>"""
    readings = f"""
<ul>
<li class="nb"><b>What could be trusted today?</b> {trusted}</li>
<li class="nb"><b>What could not?</b> {not_trusted}</li>
</ul>"""
    needed = f"""
<p><b>What would have to be added?</b> {add}</p>"""

    # Order: the verdict first (heading, criteria and table as one block, right after the definitions) with its two readings; then the scores
    # behind it, with Figure 5.1 as a float, and whether another warning level would help; then where and why the warnings fail: the four
    # ways of sorting the onsets, with Figure 5.2 as a float after the paragraph that first names it, and the note box; the example with
    # its figure; what would have to be added closes the section.
    # Layout: every figure is followed by text that can fill the page behind it; the verdict block is not a float, so no figure can come
    # between it and its readings. The paragraphs on the rain timing and on what would have to be added may run over a page break: no
    # figure stands between their lines (Figure 5.2 lies before the first, Figure 5.3 before the second), so no sentence is cut by one.
    heading = "5 Could the warnings be trusted?" + (" Onsets: not yet" if all(verd(GAUGE, k, "onset") == "NOT YET" for k in LEADS) else "")
    return f"""{LAYOUT}<h2>{heading}</h2>{intro}
{verdict}
{readings}
<h3>The scores</h3>
{two_ways}
{scores}
{floated(fig("flood_skill.png", caption1, FIG_SKILL))}
{tradeoff}
{floated(fig("flood_tradeoff.png", caption2, FIG_TRADE)) if SHOW_TRADEOFF else ''}
{f'<div class="blk">{table1}</div>' if SHOW_SCORES_TABLE else ''}
<h3>Where and why the warnings fail</h3>
{season}{storm}
{floated(fig("flood_failures.png", caption3, FIG_FAIL))}
{basin}
{timing}
{stamp}
{event + floated(fig("flood_event.png", caption4, FIG_EVENT)) if show_event else example}{needed}"""
