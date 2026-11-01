"""Report section 4.5 - when is rain 'known'? The forecast gap and the time-stamp gap.

Reads results/rain_timing.json (written by 10_rain_timing.py) and, each inside a guarded block so that the section stands without them:
results/day_windows.json + figures/day_windows.png (check_day_windows.py: the day ends measured against hourly rain), results/sources_checked.json
(what the two published sources say about the day stamp), results/flood_analysis.json (the verdict criteria), results/satellite_inputs.json and
results/lstm_all.json (one connecting sentence each). Nothing is computed here beyond differences, ratios of stored numbers, counts and formatting.
Every value is looked up by key: the median correlations in stamps.median.<pair>.<day>, the test in closure.<wn | wqn>.by_lead (the row with the
wanted "lead"), the seeds in "seeds".

What leads (decision of the project lead after the critics): the ONSET WARNINGS. They are the firm result - 1 day ahead every single seed with the
rain of the issue day is above every seed without it and every seed of the control, with and without the river gauge. The NSE follows, and it is
judged by the rule of section 4.2 as far as this file stores what the rule needs (s0_summary.nse_reading): every seed above AND higher in most
basins (no p-value is stored). With the river gauge the seeds agree but the basins do not: 'mixed', never 'clear'. '31% of the gap closed' in median
NSE is printed only with the basin count beside it.
What is left is NOT called 'rain still to come': the NLDAS day ends in the early evening, so the last hours of the issue day are in neither model.
The wording is 'the rest needs later rain, most of it after the forecast is issued'; 2 and 3 days ahead 'almost all' of the gap remains.
How the day stamp is known is worded by s0_summary.stamp_source (an inference from our own data, consistent with two published sources; the
documentation of the Daymet version in CAMELS was not checked) - the same words as in sections 5 and 6.2 and in the question sheet.

Every sentence whose wording depends on an outcome is chosen from the stored numbers (onset_clear, nse_reading, the control's seed ranges).
Where a gain is not general (NSE higher in half the basins or fewer) the text gives the largest gain and the largest loss of a single basin
(differences of models.<set>.test.basins.<gauge_id>.l1.nse, checked against the stored count basins_better).

Layout: Table 4.5 has one block for the onsets warned and one for the NSE (written out by hand: report_common.table() has no spanning header).
figures/day_windows.png (when it exists) follows the first paragraph, which names it, and figures/rain_timing.png follows the table, both as
floats (see LAYOUT in s0_summary.py), each at the width it is drawn for. The figure numbers follow section 4.4 (its figure is shown only when its switch is on).
"""
import sys

from common import FIG
from flood import PERIOD, SHORT
from report_common import LEAD_WORD, count, fig, n2, pct, signed
from report_sections.s0_summary import (LAYOUT, _above, _day, day_windows, floated, forecast_where, nse_reading, onset_clear, share_word, stamp_source,
                                         timing_verdict)
from report_sections.s4_all import P_CUT, _p
from report_sections.s4b_lobo import SHOW_FIGURE as LOBO_FIGURE

LEADS = sorted(LEAD_WORD)
T_NO = "Table 4.5"
SKILL_REF, CMP_REF, SAT_REF, VERDICT_REF = "section 4.3", "section 4.2", "section 6.2", "section 5"     # where the other results are reported
SKILL_TABLE = "Table 4.3"
FAMS = (("wn", "No river gauge", "without the river gauge"), ("wqn", "With river gauge", "with the river gauge"))     # key in closure, table label, in a sentence
MAIN = "wqn"                                    # the family the headline follows
SHOW_WINDOWS = True                             # figures/day_windows.png after the first paragraph (about 0.37 page), when the check has run
WIN_SHARE = 84                                  # % of the text width (184 mm) of that figure; it is drawn for 170 mm (92%)
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
STYLE = ("<style>table.t45{font-size:8.4pt} table.t45 th,table.t45 td{padding:1.6pt 2.5pt;line-height:1.22;text-align:right} table.t45 th{font-size:7.7pt;font-weight:400;white-space:nowrap}"
         "table.t45 td:first-child,table.t45 th:first-child{text-align:left;padding-left:4pt} table.t45 th.grp{text-align:center;font-weight:600;font-size:8pt;"
         "border-bottom:1px solid var(--grid)} table.t45 .g{border-left:1px solid var(--axis)} table.t45 tr.sub th{border-top:none}"
         "table.t45 .sr{font-size:7.1pt;color:var(--muted);white-space:nowrap} table.t45 .nw{white-space:nowrap}</style>")      # .sr: lowest-highest single seed


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _leads(ks):
    """[1] -> '1 day ahead', [2, 3] -> '2 and 3 days ahead'."""
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _cap(s):
    return s[0].upper() + s[1:]


def _pc(v):
    """A share in per cent with a real minus sign (and no 'minus zero')."""
    return pct(0.0 if abs(v) < 0.005 else v).replace("-", "&minus;")


def _prng(r):
    """[lowest, highest] single seed of a share -> '29% to 32%', or '18%' when both print alike."""
    return pct(r[0]) if pct(r[0]) == pct(r[1]) else f"{pct(r[0])} to {pct(r[1])}"


def _cell(v, r=None, share=False):
    """A table cell: the value of the seed-averaged forecast and, under it in small type, the lowest and highest single seed."""
    if share:
        small = (pct(r[0]) if pct(r[0]) == pct(r[1]) else f"{100 * r[0]:.0f}&ndash;{100 * r[1]:.0f}%") if r else ""
        return f'{pct(v)}<br><span class="sr">{small}</span>' if r else pct(v)
    return f'{n2(v)}<br><span class="sr">{n2(r[0])}&ndash;{n2(r[1])}</span>' if r else n2(v)


def last_figure(R):
    """The number of the last figure of section 4.5 (as an integer): the sections 4.6 to 4.8 number their figures from the next one, whatever
    the switches of sections 4.4 and 4.5 are (the same rule as in html(): the first figure is 4.2 or, with the figure of 4.4, 4.3)."""
    show_win = SHOW_WINDOWS and day_windows(R) is not None and (FIG / "day_windows.png").exists()
    return (3 if LOBO_FIGURE else 2) + int(bool(show_win))


def html(R, T):
    X = R.get("rain_timing")
    head = '<h3 class="clr">4.5 When is rain &lsquo;known&rsquo;? The forecast gap and the time-stamp gap</h3>'
    if not X or not X.get("closure") or MAIN not in X["closure"]:
        print("s4c_timing: results/rain_timing.json (with its closure table) was not found; the section is a placeholder", file=sys.stderr)
        return f"{head}<p>The rain-timing test (code/10_rain_timing.py) has not been run yet.</p>"
    M, notes = X["stamps"]["median"], " ".join(X.get("notes", []))
    n, n_seeds = len(X["stamps"]["per_basin"]), len(X["seeds"])
    first, rest = LEADS[0], LEADS[1:]
    fams = [f for f in FAMS if f[0] in X["closure"]]
    row = {key: {r["lead"]: r for r in X["closure"][key]["by_lead"]} for key, _, _ in fams}       # closure rows by family and lead
    seeds_word = NUMBER.get(n_seeds, n_seeds)
    W = day_windows(R)                                                                            # the day ends measured against hourly rain, or None
    show_win = SHOW_WINDOWS and W is not None and (FIG / "day_windows.png").exists()
    f0 = 3 if LOBO_FIGURE else 2                                                                   # section 4.4 shows a figure only when its switch is on
    F_WIN, F_NO = f"Figure 4.{f0}", f"Figure 4.{f0 + show_win}"

    # ---- the catch in the data: whose day is it? ------------------------------------------------------------------------
    dn, rise = M["daymet_vs_nldas"], M["river_rise_vs_daymet"]
    di, ni = M.get("daymet_vs_imerg"), M.get("nldas_vs_imerg")
    prods = [("NLDAS", dn)] + ([("satellite rain (IMERG)", di)] if di else [])
    shifted = all(p["day_before"] > 2 * p["day_after"] and p["day_before"] > 0.5 * p["same_day"] for _, p in prods)   # the day before counts, the day after does not
    vals = lambda d: _and(n2(p[d]) for _, p in prods)
    nldas = ("NLDAS (a second rain data set that comes with CAMELS"
             + (", made after the fact from rain gauges and radar)" if "rain-gauge analysis spread over the hours" in notes else ")"))
    names = (f"{nldas} and in satellite rain (IMERG, {SAT_REF}), two products stamped by UTC day" if di else f"{nldas}, a product stamped by UTC day")
    whose = (f" NLDAS and IMERG match each other on the same day ({n2(ni['same_day'])}; the day before: {n2(ni['day_before'])}), so the shifted one is "
             f"Daymet.") if ni and ni["same_day"] > 2 * ni["day_before"] else ""
    rivers = (f" The rivers rise on the day Daymet stamps the rain: the day&rsquo;s rise in flow correlates {n2(rise['same_day'])} with that rain and "
              f"{n2(rise['day_before'])} with the rain stamped a day earlier.") if rise["same_day"] > rise["day_before"] else ""
    opener = "<b>For part of the rain that is not true.</b>" if shifted else "<b>Is that true of the rain?</b>"
    if W:
        fit = (f"; {_and(W['misfits'])} {'does' if len(W['misfits']) == 1 else 'do'} not fit" if W["misfits"] else "")
        std0, std1 = (_day(d) for d in W["flow_days"])
        try:                                                                                       # where a UTC day ends at another hour (Central time)
            ends = {g["short"]: g["whole_period"]["utc_day"]["window_ends_local_standard_time"] for g in R["day_windows"]["river_flow_day"]["per_gauge"].values()}
            central = _and(f"{h} at {name}" for name, h in ends.items() if h != W["imerg"])
            central = f" ({central})" if central else ""
        except (KeyError, TypeError, AttributeError):
            central = ""
        hourly = (f"<b>The day ends, measured</b>{f' ({F_WIN})' if show_win else ''}. Against hourly rain of {NUMBER.get(W['n_refs'], W['n_refs'])} weather "
                  f"reanalyses, the rain table&rsquo;s day d ends at about {W['daymet']} local standard time on day d "
                  f"({NUMBER.get(W['n_readings'], W['n_readings'])} readings: {W['daymet_readings'][0]} to {W['daymet_readings'][1]}; single basins "
                  f"{W['daymet_basins'][0]} to {W['daymet_basins'][1]}{fit}) and the NLDAS day at about {W['nldas']} ({W['nldas_readings'][0]} to "
                  f"{W['nldas_readings'][1]}). The same check places the satellite day, which is documented to end at {W['imerg']}{central}, "
                  f"{W['off'][0]:.0f} to {W['off'][-1]:.0f} hours late, so no day end is known better than that. The daily river flow runs from local "
                  f"midnight to midnight (the best 24-hour window in {W['flow_midnight']} of {W['flow_gauges']} gauges, 15-minute flow of {std0} to "
                  f"{std1}): the end of day t, when a forecast is issued, is local midnight.")
    else:
        hourly = ""
    catch = f"""<p>{_cap(SKILL_REF)} treated the weather of the days being forecast as unknown when the forecast is made. {opener} The rain table
(Daymet) seems to stamp rain by a day that ends in the morning, as a table built from rain gauges that are read in the morning would: the rain it
stamps &lsquo;day d&rsquo; fell roughly between the morning of day d&minus;1 and the morning of day d{f" ({F_WIN})" if show_win else ""}.
{"The data show it" if shifted else "The data show it only in part"} ({F_NO}, left): Daymet&rsquo;s daily rain correlates
{vals('day_before')} with the rain of the UTC day before in {names}; {vals('same_day')} with the same day; {'only ' if shifted else ''}{vals('day_after')}
with the day after.{whose}{rivers}</p>"""
    measured = f"<p>{hourly} {stamp_source(R)}</p>"
    win_caption = "" if not show_win else (
        f"<b>{F_WIN}. Which 24 hours each daily value of &lsquo;day t&rsquo; covers,</b> in local standard time (bars: median of the {n} basins; ticks: "
        f"single basins). What to see: the forecast is issued at the end of day t (black line). The rain table&rsquo;s day t ended at about {W['daymet']} "
        f"that morning, so the rain of the roughly {24 - W['daymet_hour']:.0f} hours that follow has fallen but is stamped day t+1 (dashed), which only "
        f"the upper-bound models read. NLDAS and satellite rain reach to the early evening. The rain windows are good to about "
        f"{W['off'][-1]:.0f} hours.")
    win_figure = floated(fig("day_windows.png", win_caption, WIN_SHARE)) if show_win else ""

    # ---- what it means, the test, and how a gain is judged -----------------------------------------------------------------
    if W:
        nldas_day = (f"By the check above its day ends at about {W['nldas']} local standard time (the UTC day or a little earlier), hours before the "
                     f"forecast is issued at local midnight: no future information.")
    elif ni and ni["same_day"] > 2 * max(ni["day_before"], ni["day_after"]):
        nldas_day = ("Its day matches the UTC day of satellite rain, which ends in the early evening of the issue day, hours before the forecast is "
                     "made: no future information.")
    else:
        nldas_day = "Its day ends in the evening of the issue day, before the forecast is made."
    means = f"""<p><b>What it means.</b> A forecast is issued at the end of day t. The rain of the last half day and more has fallen by then, but the rain
table books it under day t+1, which only the models given the perfect weather forecast read. So {LEAD_WORD[first]}, the gap between rows 2
and 3 of {SKILL_TABLE} can hold rain that had already fallen (a time-stamp gap) as well as rain still to come (a forecast gap).
<b>The test</b> ({T_NO}, {F_NO}). The same all-basin LSTM, with the same settings and the same {seeds_word} seeds, gets one extra input: NLDAS
rain up to the issue day. {nldas_day} The control is the same rain made one day older: still a second rain product, but with no rain later than
the rain table&rsquo;s own. <b>How a gain is judged.</b> By the checks of {CMP_REF} as far as the results file stores what they need. NSE: every
single seed with the extra rain above every seed without it, and higher in most basins (no p is stored). Onsets warned: the {n} basins are
counted together, so only the seeds can be compared; a gain is clear when every seed with the extra rain is above every seed without it and
every seed of the control.</p>"""

    # ---- figure ----------------------------------------------------------------------------------------------------------
    m1 = row[MAIN][first]
    caption = (f"<b>{F_NO}. Whose day is it, and what rain up to the evening of the issue day adds to the NSE.</b> Left: how strongly the rain "
               f"table&rsquo;s daily rain correlates with the rain of the UTC-stamped products one day earlier, on the same day and one day later "
               f"({PERIOD[0][:4]}&ndash;{PERIOD[1][:4]}; IMERG from its first day). Middle and right: NSE in the test years (the onsets are in {T_NO}). "
               f"What to see: where the orange line leaves the grey line and the yellow control, and how far it stays below the dashed upper bound "
               f"({LEAD_WORD[first]} with the river gauge: grey {n2(m1['nse_base'])}, yellow {n2(m1['nse_control_one_day_older'])}, orange "
               f"{n2(m1['nse_with_nldas'])}, dashed {n2(m1['nse_perfect_forecast'])}).")
    figure = floated(fig("rain_timing.png", caption))                                               # drawn for the full text width: shown at it

    # ---- table: the onsets warned (the firm result) and the NSE, each without the extra rain / control / with it / upper bound -------
    sub = ["Without<br>extra<br>rain", "+ older<br>rain<br>(control)", "+ rain<br>up to the<br>evening", "Upper<br>bound&nbsp;&dagger;", "Gap<br>closed"]
    body = ""
    for key, label, _ in fams:
        for k in LEADS:
            r = row[key][k]
            cells = [_cell(r["onset_warned_base"], r.get("onset_seed_range_base"), True), _cell(r["onset_warned_control"], r.get("onset_seed_range_control"), True),
                     _cell(r["onset_warned_with_nldas"], r.get("onset_seed_range_with_nldas"), True), pct(r["onset_warned_perfect_forecast"]),
                     _pc(r["onset_share_of_gap_closed"]),
                     _cell(r["nse_base"], r["seed_range_base"]), _cell(r["nse_control_one_day_older"], r["seed_range_control"]),
                     _cell(r["nse_with_nldas"], r["seed_range_with_nldas"]), n2(r["nse_perfect_forecast"]), _pc(r["share_of_gap_closed"]), str(r["basins_better"]),
                     f'<span class="nw">{pct(r["onset_false_base"])} &rarr; {pct(r["onset_false_with_nldas"])}</span>']
            body += (f'<tr><td><span class="nw">{label},</span><br><span class="nw">{LEAD_WORD[k]}</span></td>'
                     + "".join(f'<td class="{"g" if i in (0, 5, 11) else ""}">{c}</td>' for i, c in enumerate(cells)) + "</tr>")
    widths = [14.3] + [6.8] * 5 + [7.2] * 4 + [6, 5.7] + [11.2]
    tab = ('<table class="num t45"><colgroup>' + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>"
           + '<tr><th rowspan="2">LSTM</th><th class="g grp" colspan="5">Onsets warned</th><th class="g grp" colspan="6">Median NSE</th>'
             '<th class="g grp" rowspan="2">False<br>warnings</th></tr><tr class="sub">'
           + "".join(f'<th class="{"g" if i == 0 else ""}">{h}</th>' for i, h in enumerate(sub))
           + "".join(f'<th class="{"g" if i == 0 else ""}">{h}</th>' for i, h in enumerate(sub + ["Basins<br>better"])) + "</tr>" + body + "</table>"
           + f'<p class="cap"><b>{T_NO}. Rain stamped up to the evening of the issue day as an extra input, test years.</b> Each block: the model without '
             f"the extra rain, the control (the same rain one day older), the model with the rain up to the evening, and the upper bound. Large figures: "
             f"the forecast averaged over the {seeds_word} seeds; small figures: lowest and highest single seed (the average can lie outside them). Onsets "
             f"warned: share of the onsets, the {n} basins together. NSE: median of the {n} basins. Gap closed: the gain from the rain up to the evening as "
             f"a share of the gap between the model without it and the upper bound; differences and shares are taken before rounding. Basins better: NSE "
             f"higher with that rain, of {n}. False warnings: share of the onset warnings that were false, without &rarr; with that rain. &dagger; the "
             f"perfect weather forecast, an upper bound that cannot be had in practice.</p>")

    # ---- the reading, strictly from the numbers: the onsets first, then the NSE, then what is left ---------------------------
    def basins(key):
        """How much is gained and lost basin by basin, for an NSE gain that is not general (higher in half the basins or fewer)."""
        r = row[key][first]
        try:
            new, old = X["models"][key]["test"]["basins"], X["models"][X["closure"][key]["base"].replace("lstm_", "", 1)]["test"]["basins"]
            d = [new[g][f"l{first}"]["nse"] - old[g][f"l{first}"]["nse"] for g in new]
            up, down = [v for v in d if v > 0], [v for v in d if v <= 0]
            if len(up) != r["basins_better"] or not up or not down:
                raise KeyError("per-basin scores do not match basins_better")
            return (f"the NSE is higher in only {len(up)} of {n} basins, by up to {signed(max(up))}, and lower in {len(down)}, by at most "
                    f"{n2(-min(down))}")
        except (KeyError, TypeError) as e:
            print(f"s4c_timing: per-basin gains not read ({type(e).__name__}: {e}); only the count is given", file=sys.stderr)
            return f"the NSE is higher in only {r['basins_better']} of {n} basins"

    def control_where(key):
        """Where the control's NSE gain sits, basin by basin (differences of stored scores): a clause when two basins carry nearly all of it."""
        try:
            c = X["closure"][key]
            ctrl, base = (X["models"][c[name].replace("lstm_", "", 1)]["test"]["basins"] for name in ("control", "base"))
            d = sorted(((ctrl[g][f"l{first}"]["nse"] - base[g][f"l{first}"]["nse"], g) for g in ctrl), reverse=True)
            (a, ga), (b, gb), others = d[0], d[1], [v for v, _ in d[2:]]
            if b > 3 * max(abs(v) for v in others):
                return (f" That gain of the control sits in two basins ({SHORT[ga]} {signed(a)}, {SHORT[gb]} {signed(b)}; the other {len(others)} change "
                        f"by {signed(min(others))} to {signed(max(others))}).")
        except (KeyError, TypeError, ValueError, IndexError) as e:
            print(f"s4c_timing: the control's gain by basin was not read ({type(e).__name__}: {e})", file=sys.stderr)
        return ""

    def follows():
        """Does the gain go with the day stamp, basin by basin? The stored rank correlations between the offset of the rain table's day and the
        NSE gain of the test over its control (closure.<set>.gain_over_control_against_stamp_offset), worded from their p-values."""
        try:
            sp = {key: X["closure"][key]["gain_over_control_against_stamp_offset"] for key, _, _ in fams}
            m = sp[MAIN]
            one = lambda v: f"{signed(v['spearman'])}, p {'&lt; 0.001' if v['p'] < 0.001 else '= ' + _p(v['p'])}"
            rest_ = "".join(f"; {where}: {one(sp[key])}" for key, _, where in fams if key != MAIN)
            verb = ("goes with how far the rain table&rsquo;s day is shifted against the UTC day" if m["p"] < P_CUT and m["spearman"] > 0 else
                    "shows no clear link with how far the rain table&rsquo;s day is shifted")
            return (f"<b>Basin by basin,</b> the NSE gain over the control {verb} (rank correlation with the river gauge: {one(m)}{rest_}). "
                    f"{m['n_basins']} basins that share storms: a description, not a proof.")
        except (KeyError, TypeError) as e:
            print(f"s4c_timing: the rank correlation of gain and day offset was not read ({type(e).__name__}: {e})", file=sys.stderr)
            return ""

    def onset_text(r, again=False):
        """The onset warnings 1 day ahead of one family, worded from the seed ranges."""
        move = (f"{pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} (control {pct(r['onset_warned_control'])}; upper bound "
                f"{pct(r['onset_warned_perfect_forecast'])})")
        if onset_clear(r):
            return (f"onsets warned rise from {move}, {_pc(r['onset_share_of_gap_closed'])} of the gap: {'again ' if again else ''}every single seed with "
                    f"the extra rain ({_prng(r['onset_seed_range_with_nldas'])}) is above every seed without it ({_prng(r['onset_seed_range_base'])}) and "
                    f"every seed of the control ({_prng(r['onset_seed_range_control'])})")
        if _above(r.get("onset_seed_range_with_nldas"), r.get("onset_seed_range_base")):
            return (f"onsets warned rise from {move}; every seed with the extra rain is above every seed without it, but not above every seed of the "
                    f"control: the control does not settle whether the gain is the rain of the issue day")
        return f"onsets warned go from {move}; the single seeds overlap: no clear gain"

    def nse_text(r, key):
        """The NSE 1 day ahead of one family, worded from its reading by the rule of section 4.2 and from the control."""
        what = nse_reading(r, n)
        nums = (f"the median {'rises' if r['gain'] > 0 else 'falls'} from {n2(r['nse_base'])} to {n2(r['nse_with_nldas'])} ({signed(r['gain'])}; "
                f"{_pc(r['share_of_gap_closed'])} of the gap to the upper bound, {n2(r['nse_perfect_forecast'])})")
        ctrl_above = _above(r.get("seed_range_control"), r.get("seed_range_base"))
        ctrl = f"The control gains {signed(r['gain_control'])}"
        if what == "mixed":
            return (f"{nums}, and every seed with the extra rain is above every seed without it. But {basins(key)}; the median basin changes by "
                    f"{signed(r['median_basin_gain'])}. By the basin count this is not a clear gain: a mixed result. {ctrl}.")
        if what == "clear":
            net = ("; all its seeds lie below those of the test: the gain is the rain of the issue day"
                   if _above(r["seed_range_with_nldas"], r.get("seed_range_control")) else "; its seeds overlap those of the test, so the control does not settle it")
            return f"{nums}, every seed with the extra rain is above every seed without it, and the NSE is higher in {r['basins_better']} of {n} basins: a clear gain. {ctrl}{net}."
        part = "most" if r["gain"] > 0 and r["gain_control"] / r["gain"] > 0.5 else "part"
        return (f"{nums}; the NSE is higher in {r['basins_better']} of {n} basins, but the single seeds overlap those of the model without the extra "
                f"rain: no clear difference. {ctrl}"
                + (f", with every seed above that model: in NSE a second rain product explains {part} of the gain.{control_where(key)}"
                   if ctrl_above and r["gain"] > 0 else "."))

    tv = timing_verdict(R)                                                                          # the onset warnings with that rain, as counts and as a verdict
    price = (f" The price is more false warnings: {pct(m1['onset_false_base'])} of the onset warnings without the extra rain and "
             f"{pct(m1['onset_false_with_nldas'])} with it; in number {count(tv['false_base'])} and {count(tv['false'])}." if tv and tv["false"] > tv["false_base"]
             else f" False warnings {'rise' if m1['onset_false_with_nldas'] > m1['onset_false_base'] else 'fall'} from {pct(m1['onset_false_base'])} to "
                  f"{pct(m1['onset_false_with_nldas'])} of the onset warnings.")
    items = [f"<li><b>Onsets, with the river gauge:</b> {onset_text(m1)}. So the gain is the rain of the issue day, not the second rain product.{price}</li>"
             if onset_clear(m1) else f"<li><b>Onsets, with the river gauge:</b> {onset_text(m1)}.{price}</li>",
             f"<li><b>NSE, with the river gauge:</b> {nse_text(m1, MAIN)}</li>"]
    for key, _, where in fams:
        if key != MAIN:
            r = row[key][first]
            items.append(f"<li><b>{_cap(where)}:</b> {onset_text(r, again=onset_clear(m1))}. NSE: {nse_text(r, key)}</li>")
    later = [(key, where, k) for key, _, where in fams for k in rest]
    on_gain = [100 * (row[key][k]["onset_warned_with_nldas"] - row[key][k]["onset_warned_base"]) for key, _, k in later]
    on_share = [row[key][k]["onset_share_of_gap_closed"] for key, _, k in later]
    on_all = all(onset_clear(row[key][k]) for key, _, k in later)
    gains, shares = [row[key][k]["gain"] for key, _, k in later], [row[key][k]["share_of_gap_closed"] for key, _, k in later]
    nse_clear = [(key, where, k) for key, where, k in later if _above(row[key][k]["seed_range_with_nldas"], row[key][k]["seed_range_base"])]
    further = (f"<b>{_cap(_leads(rest))}:</b> onsets warned {'rise' if min(on_gain) > 0 else 'change'} by {min(on_gain):.0f} to {max(on_gain):.0f} percentage "
               f"points, {_pc(min(on_share))} to {_pc(max(on_share))} of the gap"
               + (" (every seed with the extra rain is above every seed without it and of the control, but the step is small)" if on_all else "")
               + f". The median NSE changes by {signed(min(gains))} to {signed(max(gains))} ({_pc(min(shares))} to {_pc(max(shares))} of the gap)")
    if not nse_clear:
        further += ": the single seeds overlap."
    else:
        which = _and(f"{where} {LEAD_WORD[k]} ({signed(row[key][k]['gain'])}; the control {signed(row[key][k]['gain_control'])})" for key, where, k in nse_clear)
        further += f"; the single seeds overlap, except {which}."
    fw = [(row[MAIN][k]["onset_false_base"], row[MAIN][k]["onset_false_with_nldas"]) for k in rest]     # the catch further ahead: the false warnings
    if all(a == a and b == b and pct(a) != pct(b) for a, b in fw) and len({b > a for a, b in fw}) == 1:
        further += (f" With the river gauge the false warnings {'rise' if fw[0][1] > fw[0][0] else 'fall'} there, from {_and(pct(a) for a, _ in fw)} to "
                    f"{_and(pct(b) for _, b in fw)}.")
    if follows():
        items.append(f"<li>{follows()}</li>")
    items.append(f"<li>{further}</li>")

    share_on = m1["onset_share_of_gap_closed"]
    what = nse_reading(m1, n)
    nse_head = {"mixed": (f"In median NSE the result is mixed: up from {n2(m1['nse_base'])} to {n2(m1['nse_with_nldas'])}, but higher in only "
                          f"{m1['basins_better']} of {n} basins."),
                "clear": (f"The median NSE rises clearly too: from {n2(m1['nse_base'])} to {n2(m1['nse_with_nldas'])}, higher in {m1['basins_better']} of {n} "
                          f"basins."),
                "none": (f"In median NSE there is no clear difference ({n2(m1['nse_base'])} to {n2(m1['nse_with_nldas'])}; the single seeds overlap).")}[what]
    stamp_gap = onset_clear(m1) and 0 < share_on < 1
    if stamp_gap:
        frac = share_word(share_on)
        exact = "" if frac == pct(share_on) else f", {_pc(share_on)} of that gap"
        headline = (f"{LEAD_WORD[first].capitalize()}, {frac} of the gap in onsets warned is a time-stamp gap: with the river gauge, rain stamped up to "
                    f"the evening of the issue day lifts the onsets warned from {pct(m1['onset_warned_base'])} to {pct(m1['onset_warned_with_nldas'])} "
                    f"(control {pct(m1['onset_warned_control'])}; upper bound {pct(m1['onset_warned_perfect_forecast'])}){exact}. {nse_head}")
        left_nse = min(1.0, 1 - max(row[MAIN][k]["share_of_gap_closed"] for k in rest))
        left_on = min(1.0, 1 - max(row[MAIN][k]["onset_share_of_gap_closed"] for k in rest))
        left_word = "almost all" if min(left_nse, left_on) >= 0.9 else "most" if min(left_nse, left_on) > 0.5 else "part"
        hours = (f"the NLDAS day ends at about {W['nldas']}, so the last hours of the issue day are in neither model" if W else
                 "the NLDAS day ends in the early evening, so the last hours of the issue day are in neither model")
        left = (f"<b>What is left.</b> With the river gauge, {_pc(1 - share_on)} of the gap in onsets warned {LEAD_WORD[first]} is not closed by rain "
                f"stamped up to the evening: the rest needs later rain, most of it after the forecast is issued ({hours}). {_cap(_leads(rest))} {left_word} of the gap remains (with the river gauge {_pc(left_nse)} or more of the gap in NSE and "
                f"{_pc(left_on)} or more of the gap in onsets warned). The split is approximate, because NLDAS and Daymet are different products.")
        larger = "the larger part of the gap" if share_on < 0.5 else "part of the gap"
        still = (f", and with that rain the onsets are still {tv['verdict']} by the criteria of {VERDICT_REF} ({pct(tv['pod'])} warned, {pct(tv['far'])} "
                 f"false warnings, at the high-flow level)" if tv and tv["verdict"] != "GO" else "")
        tested = forecast_where(R)                                                    # section 4.8 has tested one real forecast: say where it lands (None: the old sentence)
        system = (f"A warning system needs a weather forecast as an input at every lead time: it is {larger} even {LEAD_WORD[first]}, "
                  + (f"and {tested}. " if tested else "and it was not tested. ") +
                  f"Rain observations up to the hour the forecast is made (radar and gauge analyses, satellite near-real-time runs) exist today; "
                  f"they add a part {LEAD_WORD[first]}, here {share_word(share_on)} of the gap in onsets warned{still}.")
    elif _above(m1.get("onset_seed_range_control"), m1.get("onset_seed_range_base")):
        headline = f"The test does not show a time-stamp gap: the control, the same rain one day older, gains as clearly as the rain up to the evening. {nse_head}"
        left = (f"The gain is what a second rain product is worth. The statement in {SKILL_REF} that part of the gap is rain that had already fallen is not "
                f"supported by this test.")
        system = ("For a warning system the test says: a second, independent rain product helps; whether rain observations up to the hour add more "
                  "is not shown, and the rain still to come needs a weather forecast.")
    else:
        headline = f"The test does not show a time-stamp gap of a measurable size in the onsets warned. {nse_head}"
        left = (f"On this evidence the gap to the perfect weather forecast is a forecast gap at all {NUMBER.get(len(LEADS), len(LEADS))} lead times; the "
                f"statement in {SKILL_REF} that part of it is rain that had already fallen is not supported by this test.")
        system = "For a warning system the test gives no evidence that rain observations up to the hour would help; the gap needs a weather forecast."
    items = [x.replace("<li>", '<li class="nb">', 1) for x in items]                # no bullet is split over two pages: a figure must not cut a sentence
    found = f'<p class="nb"><b>{headline}</b></p><ul>{"".join(items)}</ul><p class="nb">{left}</p>'

    # ---- the caveat of the notes, and two results elsewhere that agree (each read by key, each guarded) ----------------------
    late = " NLDAS itself is published a few days late: the point of the test is the time stamp, not the product." if "published a few days late" in notes else ""
    agree = []
    try:
        S = R["satellite_inputs"]["experiments"]["a_rain_source"]["against_base"]["gauge_imerg_wq"]
        up = [k for k in LEADS if S[f"l{k}"]["nse_median"]["larger_than_seed_noise"] and S[f"l{k}"]["nse_median"]["difference"] > 0]
        if up:
            agree.append(f"satellite rain, whose day also ends in the evening of the issue day, lifts the model with the river gauge {_leads(up)} "
                         f"when it is added to the rain table&rsquo;s rain ({_and(signed(S[f'l{k}']['nse_median']['difference']) for k in up)} NSE, clear "
                         f"by the seed rule of {SAT_REF}, on its own shorter split)")
    except (KeyError, TypeError) as e:
        print(f"s4c_timing: the sentence on satellite rain was left out ({type(e).__name__}: {e})", file=sys.stderr)
    try:
        A = R["lstm_all"]
        C = A["comparison"]["wf"]["baselines"]
        unfair = [k for k in LEADS if any(b[f"l{k}"].get("like_for_like") is False for b in C.values())]
        if unfair and any("observer day" in x for x in A["notes"]):
            agree.append(f"rain stamped one day after the day being forecast still tells about that day, which is why {CMP_REF} calls no winner "
                         f"{_leads(unfair)} where the LSTM reads the perfect weather forecast of all {NUMBER.get(len(LEADS), len(LEADS))} coming days")
    except (KeyError, TypeError) as e:
        print(f"s4c_timing: the sentence on the like-for-like comparison was left out ({type(e).__name__}: {e})", file=sys.stderr)
    same = f" Probably the same effect shows elsewhere: {agree[0]}." if agree and stamp_gap else ""      # the first that was found
    note = f'<div class="note blk"><b>What follows.</b> {system}{late}{same}</div>'

    # Layout: the figure of the day windows is a float between the first paragraph, which names it, and the paragraph that reads it ('The day
    # ends, measured'); that paragraph and 'What it means' can then fill the foot of the page, and the table and the figure of the test follow.
    return f"""{LAYOUT}{STYLE}{head}
{catch}
{win_figure}
{measured}
{means}
<div class="blk">{tab}</div>
{figure}
{found}
{note}"""
