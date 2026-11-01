"""Report section 4.8 - a real weather forecast as an input (E2: the GEFSv12 reforecast, control member, rain of the next three days).

Reads results/forecast_input.json (code/18_forecast_input.py) and results/gefs_basins.json (code/gefs_to_basins.py: the basin means, the issue-time margin), results/data_access.json
(the archive, its members) and results/published_context.json (the published Brier skill of the ensemble), through the shared reader forecast_facts of s0_summary.py; the ladder of
Table 4.8 comes straight from the ladders and the control blocks of results/forecast_input.json; the review record results/review_forecast_input.json is read by review_clause. Every
number is read by key and every sentence whose wording depends on an outcome is chosen from the stored numbers and readings. Nothing is computed beyond differences, shares of stored
counts and formatting.

What leads: the forecast's own skill (what the input is worth only makes sense beside what it is), the issue-time rule (a real service has the run about four hours old, the last files
about a quarter of an hour late in Eastern summer), then E2a (the forecast substituted into the network of section 4.6: the result on the usual test years), E2b (retrained with the
forecast on a shorter split, with the clean control), the simple models, and the verdicts with what was not done.

What the independent review forbids, and this section does not say: that the 1-day difference of E2a at the fixed warning level is a loss of skill, or that it is explained by over-trust
(it is a moved operating point - fewer onsets warned at the fixed level with far fewer false warnings, more at the level chosen on the validation years, a better ranking of the onset days;
its cause is not isolated); that the shuffled-forecast control of E2a shows that the forecast is informative (it measures the harm of misleading slots to a network that trusts them; the clean
control is E2b's); that the median-NSE gains are clear (they are 'mixed' by the rule of section 4.2 at all three lead times, and the onsets are the clear result); that the forecast's Brier
skill is 'the same as' the published one (it is of the same order, computed differently); that the gains of E2b hold at the plain trigger alone (the section gives the level chosen on the
validation years beside it); and that the E2a and E2b numbers can be set against each other (two splits, two bases).

Layout: Table 4.8 (the ladder with seed ranges in small grey type, in two blocks: the main split and the shorter split of E2b) and the figure as a float at the full text width, numbered
after the figure of section 4.7. The table and figure numbers are read by section 8 through numbers().
"""
import sys

from common import FIG
from flood import LEADS
from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed
from report_sections.s0_summary import (AORC_WORDING, AS_MUCH, INFO_NOT_NET, LADDER_STYLE, LAYOUT, MOVED_POINT, _and, _file, _leads, _word, by_lead_reading, clock_facts, criteria, fig_cropped, floated, forecast_facts,
                                         ladder_row, ladder_table, nse_forecast_numbers, nse_forecast_words, reading_word, review_clause)
from report_sections import s4d_midnight, s4e_rivers
from report_sections.s4c_timing import last_figure

T_NO = "Table 4.8"
SHOW_FIGURE = True                              # figures/forecast_input.png: drawn for 335 mm (13.2 in), shown at the full text width (about 0.45 page)
FIG_SHARE = 88                                 # % of the text width (184 mm)
FIG_FILE = "forecast_input.png"
lk = lambda k: f"l{k}"
n1 = lambda v: n2(v, 1)
s1 = lambda v: signed(v, 1)


def numbers(R):
    """(table number, figure number or None) of this section, for the pointers of section 8."""
    before = sum(int(bool(m.SHOW_FIGURE and (FIG / m.FIG_FILE).exists())) for m in (s4d_midnight, s4e_rivers))
    return T_NO, (f"Figure 4.{last_figure(R) + before + 1}" if SHOW_FIGURE and (FIG / FIG_FILE).exists() else None)


def _rows(X):
    """The rows of Table 4.8 from the ladders of results/forecast_input.json: the main split (E2a) and the shorter split (E2b); None where the file is not in the expected form."""
    try:
        by = lambda arm: {r["set"]: r["by_lead"] for r in X["ladder"][arm]["rungs"] + X["ladder"][arm].get("extras", [])}
        m, s = by("main"), by("sat")
        ctrl = lambda blk: {lk(k): dict(onset_warned=blk[lk(k)]["onsets"]["warned_without"], onset_seed_range=blk[lk(k)]["onsets"]["seed_range_without"], nse_median=blk[lk(k)]["nse"]["median_without"],
                                        nse_seed_range=blk[lk(k)]["nse"]["seed_range_without"], onset_false=blk[lk(k)]["onsets"]["false_warnings_without"]) for k in LEADS}
        n = {a: [X["ladder"][a]["rungs"][0]["by_lead"][lk(k)]["n_onsets"] for k in LEADS] for a in ("main", "sat")}
        yrs = lambda a: f"{X['split_main' if a == 'main' else 'split_satellite_era']['test'][0][:4]}&ndash;{X['split_main' if a == 'main' else 'split_satellite_era']['test'][1][:4]}"
        return [dict(head=f"Main split, test years {yrs('main')} ({_and([count(v) for v in n['main']])} onsets at 1, 2 and 3 days): trained on the AORC rain of the days being forecast, run on the forecast (E2a)"),
                ladder_row("+ AORC rain to midnight, no forecast", m["lstm_wqa"]),
                ladder_row("+ GEFS forecast of the next three days, substituted (E2a)", m["lstm_wqaF_sub"]),
                ladder_row("control: the forecast shuffled within the month (measures harm)", ctrl(X["forecast_vs_climatology_control"]["by_lead"]), "ctl"),
                ladder_row("perfect rain of the next three days (AORC)", m["lstm_wqaP"]),
                dict(head=f"Shorter split, test years {yrs('sat')} ({_and([count(v) for v in n['sat']])} onsets; its own scores, a weaker base): retrained with the forecast as an input (E2b)"),
                ladder_row("+ AORC rain to midnight, no forecast", s["lstm_wqa_s"]),
                ladder_row("+ GEFS forecast, retrained (E2b)", s["lstm_wqaF_s"]),
                ladder_row("control: shuffled forecast, trained and run on it (clean)", ctrl(X["retrained_vs_substituted"]["retrained_against_its_control"]), "ctl")]
    except (KeyError, TypeError, IndexError) as exc:
        print(f"s4f_forecast: results/forecast_input.json is not in the expected form for Table 4.8 ({type(exc).__name__}: {exc}); the table is left out", file=sys.stderr)
        return None


def html(R, T):
    F = forecast_facts(R)
    head = '<h3 class="clr">4.8 A real weather forecast as an input</h3>'
    if not F:
        print("s4f_forecast: results/forecast_input.json was not found or is not in the expected form; the section is a placeholder", file=sys.stderr)
        return f"{head}<p>The forecast-input experiment (code/gefs_to_basins.py and code/18_forecast_input.py) has not been run yet.</p>"
    X = _file(R, "forecast_input")
    first, rest = LEADS[0], tuple(LEADS[1:])
    one = LEAD_WORD[first]
    t_no, f_no = numbers(R)
    n_seeds = len(X["seeds"])
    C = clock_facts(R)
    slash = lambda d, f=pct: "/".join(f(d[k]) for k in LEADS)                      # 51%/21%/12%: the three lead times in a row
    three = lambda d, f=s1: _and([f(d[k]) for k in LEADS])

    # ---- the data, the issue-time rule, the forecast's own skill -----------------------------------------------------------------
    late = F["late"]
    days = _word(F["margin_days"]) if F["margin_days"] else "a few"
    margin = (f"its last files appear about {abs(round(late))} minutes after the Eastern summer issue time (before it elsewhere), a margin read from {days} days of posting times of the "
              f"operational member, not from the archive" if late < 0 else
              "its last files appear before the issue time, a margin read from the posting times of the operational member, not from the archive")
    age = f" ({F['hours_old'][0]} to {F['hours_old'][1]} hours old then)" if F["hours_old"] else ""
    cells = f" ({F['n_cells'][0]} to {F['n_cells'][1]} cells)" if F["n_cells"] else ""
    members = f"one of the {_word(F['members'])} members run every day" if F["members"] else "one member"
    pub = (f", of the same order as the published {n2(F['published'][0])} and {n2(F['published'][1])} for the whole ensemble ({cite('guan2022')[1:-1]}; computed differently, no check of that paper)"
           if F.get("published") else "")
    future = f", {pct(C['future'])} of the rain around the onsets," if C else ""
    opening = (f"<p>The rain that falls after the forecast is issued{future} can come only from a weather forecast. The GEFSv12 reforecast holds forecasts issued once a day in the past "
               f"{cite('gefsv12')}: its control member ({members}), {F['years'][0]} to {F['years'][1]}, was averaged over each basin by exact cell overlaps{cells} and summed over the local days "
               f"t+1 to t+3 from the 00 UTC run of the UTC date t+1, the latest that exists at the issue time{age}; {margin}. Against AORC daily rain its correlation is "
               f"{_and([n2(F['corr'][k]) for k in LEADS])} at {_leads(LEADS)} and its calibrated Brier skill for days with at least {F['brier_event']} {_and([n2(F['brier'][k]) for k in LEADS])}{pub}.</p>")

    # ---- E2a: the forecast substituted ----------------------------------------------------------------------------------------------
    years = _and([f"{F['years_above'][k][0]} of {F['years_above'][k][1]}" for k in rest])
    clear23 = all(F["gain_reading"][k] == "clear gain" for k in rest)
    ap = (f"; average precision {signed(F['ap_diff'][first], 3)}, 95% interval {signed(F['ap_ci'][first][0], 3)} to {signed(F['ap_ci'][first][1], 3)}" if F.get("ap_ci") else "")
    lead1 = (f"{one.capitalize()} the model warns of fewer onsets at the fixed warning level ({s1(F['gain'][first])} points) but its warnings are right more often "
             f"({pct(F['far'][1])} false against {pct(F['far'][0])}; CSI {n2(F['csi'][1], 3)} against {n2(F['csi'][0], 3)}{ap}), and at the level chosen on the validation years it warns of "
             f"{n1(abs(F['chosen_gain'][first]))} points {'more' if F['chosen_gain'][first] >= 0 else 'fewer'}: {MOVED_POINT}, said of the warning scores only (NSE has no warning level to move: "
             f"the {one} median NSE is lower); why it warns of fewer at the fixed level is not isolated."
             if F["moved"] else
             f"{one.capitalize()} the model warns of {s1(F['gain'][first])} points of onsets at the fixed warning level ({reading_word(F['gain_reading'][first])}).")
    e2a = (f"<p><b>E2a: the forecast substituted.</b> The network of section 4.6 gets three more inputs, the rain of the days t+1 to t+3; it is trained on the AORC rain of those days (a perfect rain forecast), stopped on "
           f"the validation years with the forecast in those slots and scored on the usual test years. Onsets warned {slash(F['rain'])} without the forecast go to {slash(F['fc'])} with it at "
           f"{_leads(LEADS)}; {nse_forecast_words(F)}: the median NSE goes from {slash(F['nse_rain'], n2)} to {slash(F['nse_fc'], n2)} ({nse_forecast_numbers(F, medians=False)}). "
           f"{_leads(rest).capitalize()} the gain in onsets is "
           f"{'clear in every training' if clear23 else 'not clear in every training'} ({_and([s1(F['gain'][k]) for k in rest])} points; positive in {years} test years). {lead1}</p>")

    # ---- the shuffled control, E2b ---------------------------------------------------------------------------------------------------
    s_same = all(abs(F["s_ctrl"][k] - F["s_rain"][k]) < 0.02 for k in LEADS)
    e2b = (f"<p>The control of E2a, the forecast shuffled within the month, collapses to {slash(F['ctrl'])}: a network trained on exact slots and given noise, so it measures harm, not information. "
           f"The clean control is E2b&rsquo;s. <b>E2b: retrained with the forecast.</b> On the shorter satellite-era split (its own scores; {_and([count(F['n_sat'][k]) for k in LEADS])} onsets; a weaker "
           f"base) onsets warned go from {slash(F['s_rain'])} to {slash(F['s_fc'])} ({three(F['s_gain'])} points at the fixed level, {three(F['s_gain_chosen'])} at the level chosen on the validation "
           f"years), the median NSE from {slash(F['s_nse_rain'], n2)} to {slash(F['s_nse_fc'], n2)}. Its control, trained and run on shuffled slots, warns of {slash(F['s_ctrl'])}: "
           + ("it equals the model without the forecast, so the gain is the forecast&rsquo;s information. " if s_same else "it differs from the model without the forecast. ")
           + f"A perfect rain forecast would give {slash(F['perfect'])} on the main split and {slash(F['s_perfect'])} there.</p>")

    # ---- the simple models and the rain to midnight --------------------------------------------------------------------------------
    g, w, fa = F["gain_simple"], F["warned_simple"], F["far_simple"]
    as_much = all(min(g["ridge regression"][k], g["random forest"][k]) >= AS_MUCH * g["lstm"][k] for k in rest)
    ng = F["nse_gain_simple"]
    sg = lambda m: "/".join(signed(ng[m][k]) for k in LEADS)
    nse_simple = (f" In median NSE the forecast adds {sg('ridge regression')} to ridge regression and {sg('random forest')} to the forest against {sg('lstm')} to the LSTM"
                  + (f": {one} the LSTM&rsquo;s NSE falls while theirs rises" if ng["lstm"][first] < 0 < min(ng["ridge regression"][first], ng["random forest"][first]) else "") + ".")
    simple = (f"<p><b>The simple models.</b> Given the same inputs ridge regression warns of {slash(w['ridge regression'])} of the onsets and the random forest of {slash(w['random forest'])} "
              f"(the LSTM: {slash(w['lstm'])}); the forecast adds {'/'.join(s1(g['ridge regression'][k]) for k in LEADS)} and {'/'.join(s1(g['random forest'][k]) for k in LEADS)} points to them "
              f"against {'/'.join(s1(g['lstm'][k]) for k in LEADS)} to the LSTM"
              + (f", so for onsets {_leads(rest)} the gain is {INFO_NOT_NET}" if as_much else "") + f", with more false warnings ({pct(fa['ridge regression'][first])} for ridge regression and {pct(fa['random forest'][first])} "
              f"for the forest against {pct(fa['lstm'][first])}: a different operating point). The rain to midnight still matters beside the forecast {one} ({s1(F['midnight_given_forecast'][first])} "
              f"points) and {'little' if max(abs(F['midnight_given_forecast'][k]) for k in rest) < 5 else 'also'} {_leads(rest)} ({_and([s1(F['midnight_given_forecast'][k]) for k in rest])}).{nse_simple}</p>")

    # ---- the table, the figure, verdicts and what was not done ---------------------------------------------------------------------
    rows = _rows(X)
    table = ""
    if rows:
        table = floated(ladder_table(rows, (
            f"<b>{t_no}. A real weather forecast as an input, with the river gauge.</b> Onsets warned and median NSE at the fixed warning level; large figures: the forecast averaged over the "
            f"{_word(n_seeds)} seeds; small grey figures: lowest and highest single training. False warnings: share of the onset warnings that were false, {one}. The two blocks have different "
            f"test years and are not comparable.")))
    figure = ""
    if SHOW_FIGURE and (FIG / FIG_FILE).exists():
        figure = floated(fig_cropped(FIG_FILE, (
            f"<b>{f_no}. A real weather forecast as an input: onsets warned (top) and median NSE (bottom), {_leads(LEADS)}.</b> Left of each panel: the main split "
            f"(blue); right, shaded: the shorter split of E2b (orange). Dots: the forecast averaged over the {_word(n_seeds)} trainings, whiskers: lowest to highest single training; the dark bar in the onset "
            f"panels: the same model at the warning level chosen on the validation years; hollow circle: the control, the forecast shuffled (left it measures harm, right it is clean); diamond: the forecast "
            f"given to the plain gauge model, without the rain to midnight. One ensemble member only; AORC rain to midnight is {AORC_WORDING}. The numbers are in {t_no}."), FIG_SHARE, top=0.112, bottom=0.085))
    vo = set(F["verdict_onset"].values()) | set(F["verdict_onset_chosen"].values())
    slashv = lambda d: " / ".join(d[k] for k in LEADS)
    note = (f'<p class="nbx"><b>What follows, and what was not done.</b> The verdict for the onsets is {_and(sorted(vo))} at every lead time (E2a, at both warning levels); for all high-flow days it is '
            f"{slashv(F['verdict_all_chosen'])} at the level chosen on the validation years ({slashv(F['verdict_all'])} at the fixed level; E2b: {slashv(F['verdict_all_s_chosen'])}). The success rule fixed in advance, a clear gain in "
            f"onsets 2 and 3 days ahead, is {'met' if F['success'] else 'not met'}. Not done: the other {_word(F['members'] - 1) if F['members'] else 'other'} ensemble members; temperature "
            f"forecasts; the encoder-decoder form that reads a forecast day by day {cite('nearing2024')}; Stage IV in place of AORC with the forecast. {review_clause(R, 'forecast_input')}</p>")
    return f"""{LAYOUT}{LADDER_STYLE}{head}
{opening}
{table}
{figure}
{e2a}
{e2b}
{simple}
{note}"""
