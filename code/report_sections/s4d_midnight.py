"""Report section 4.6 - rain known up to the issue time (E1 with hourly ERA5 rain, E1b with AORC rain, the hindsight question about AORC, the real-time test with Stage IV).

Reads results/rain_to_midnight.json (15_rain_to_midnight.py), results/aorc_rain_to_midnight.json (16_aorc_rain_to_midnight.py), results/aorc_hindsight.json
(check_aorc_hindsight.py), results/stage4_realtime.json (check_stage4_realtime.py), results/onset_rain_clock.json (check_onset_rain_clock.py), results/day_windows.json
(the summer-clock count of the gauges) and the review records results/review_rain_to_midnight.json and results/review_aorc_rain_to_midnight.json. Every number is read
by key (the shared readers era5_facts, aorc_facts, hindsight_facts, stage4_facts and clock_facts of s0_summary.py; the ladder of Table 4.6 straight from the ladders of the
two results files) and every sentence whose wording depends on an outcome is chosen from the stored numbers and readings. Nothing is computed beyond differences, shares of
stored counts and formatting. Where a file is missing the paragraph that needs it is left out, with one line on stderr.

What leads, as in section 4.5: the ONSETS WARNED 1 day ahead with the river gauge, because that is the number the project is about; the NSE follows. The order of the text
is the order of the evidence: where the missing rain falls and the clock of the input (one sentence: the whole issue day by the wall clock of the gauge, summer time counted),
the ladder, E1 (the first test, with a stand-in: model rain), E1b (the same with AORC, which carries the gain), what the simple models do with the same input, then the two
questions that AORC raises because it is an analysis made after the fact (hindsight; what a real-time product gives), the verdicts and what was not done.

What the independent reviews forbid, and this section does not say: that the E1 outcome sentence is the result (it must be read with the facts that the NLDAS rain gives the
same share, that the control is not flat and that the success rule fixed in advance is not met); that 'the mis-stamped table itself costs skill' (the replaced table changes the product
and the clock together); that the LSTM is 'clearly ahead' of the simple models without saying that this is the NSE and that ridge regression warns of more onsets; that AORC
is observed rain (it is an analysis made after the fact); and that the hindsight of AORC is excluded for all 611 onsets by the review alone (it bounded it at about 3 points on
171 of them; the Stage IV test on all 611 follows).

Layout: Table 4.6 (the ladder with seed ranges in small grey type, as Table 4.5) and the figure of E1b (the figure of E1 is left out for space) as a float, numbered after the
last figure of section 4.5. The table and figure numbers are read by section 8 through numbers().
"""
import sys

from common import FIG
from flood import LEADS
from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed
from report_sections.s0_summary import (AORC_WORDING, AS_MUCH, INFO_NOT_NET, LADDER_STYLE, LAYOUT, MONTH, REALTIME_WORDING, _and, _file, _leads, _word, aorc_facts, choices_after_test,
                                         clock_facts, criteria, day_windows, era5_facts, fig_cropped, floated, four_experiments, hindsight_clause, hindsight_facts, ladder_row,
                                         ladder_table, reading_word, review_clause, share_word, stage4_facts)
from report_sections.s4_all import PHRASE, _reading
from report_sections.s4c_timing import _prng, last_figure

T_NO = "Table 4.6"
SHOW_FIGURE = True                              # figures/aorc_rain_to_midnight.png (about 0.4 page); the ERA5 figure of E1 is not shown
FIG_SHARE = 68                                  # % of the text width (184 mm); the figure is drawn for 259 mm (10.2 in), so its text is small at any share
FIG_FILE = "aorc_rain_to_midnight.png"
n1 = lambda v: n2(v, 1)                         # a number of percentage points with one decimal
s1 = lambda v: signed(v, 1)                     # ... with its sign


def numbers(R):
    """(table number, figure number or None) of this section, for the pointers of section 8."""
    return T_NO, (f"Figure 4.{last_figure(R) + 1}" if SHOW_FIGURE and (FIG / FIG_FILE).exists() else None)


def _ladder(R, nldas, utc):
    """The rows of Table 4.6, read from the ladders of the two results files (with the river gauge, test years); None while neither file exists."""
    E, A = _file(R, "rain_to_midnight"), _file(R, "aorc_rain_to_midnight")
    if not (E and A):
        return None
    try:
        by = lambda X: {r["set"]: r for r in X["ladder"]["wq"]["rungs"]}
        a, e = by(A), by(E)
        nine, utc_end = nldas[1], utc                                              # nldas: (the end of the NLDAS day, the end of the table's own window); utc: the end of the UTC day
        nldas = nldas[0]
        utc = {f"l{r['lead']}": dict(onset_warned=r["onset_warned_with"], onset_seed_range=r["onset_seed_range_with"], nse_median=r["nse_with"], nse_seed_range=r["seed_range_nse_with"],
                                     onset_false=r["onset_false_with"]) for r in A["closure"]["wqa_utc"]["by_lead"]}
        return [ladder_row("Rain table alone", a["lstm_wq"]["by_lead"]),
                ladder_row(f"+ NLDAS rain, day ends {nldas} (section 4.5)", e["lstm_wqn"]["by_lead"]),
                ladder_row("+ ERA5 rain to midnight (E1)", e["lstm_wqh"]["by_lead"]),
                ladder_row("+ AORC rain to midnight (E1b)", a["lstm_wqa"]["by_lead"]),
                ladder_row(f"control: AORC over the table&rsquo;s own window, {nine} to {nine}", a["lstm_wqa"]["by_lead"], "ctl", "control_"),
                ladder_row(f"control: AORC over the UTC day, ends {utc_end}", utc, "ctl"),
                ladder_row("Perfect weather forecast (an upper bound that cannot be had in practice)", a["lstm_wqf"]["by_lead"])]
    except (KeyError, TypeError, IndexError) as exc:
        print(f"s4d_midnight: the ladders of the two results files are not in the expected form ({type(exc).__name__}: {exc}); Table 4.6 is left out", file=sys.stderr)
        return None


def html(R, T):
    E, A, H, S4, C = era5_facts(R), aorc_facts(R), hindsight_facts(R), stage4_facts(R), clock_facts(R)
    head = '<h3 class="clr">4.6 Rain known up to the issue time</h3>'
    if not (E or A):
        print("s4d_midnight: neither results/rain_to_midnight.json nor results/aorc_rain_to_midnight.json was found; the section is a placeholder", file=sys.stderr)
        return f"{head}<p>The rain-to-midnight experiments (code/15_rain_to_midnight.py and code/16_aorc_rain_to_midnight.py) have not been run yet.</p>"
    first = LEADS[0]
    one = LEAD_WORD[first]
    W = day_windows(R)                                                           # the day ends measured against hourly rain (results/day_windows.json), or None
    D9 = W["daymet"] if W else "09:00"                                           # where the rain table's day ends
    U19 = W["imerg"] if W else "19:00"                                           # where the UTC day ends (local standard time)
    n_seeds = len((_file(R, "aorc_rain_to_midnight") or _file(R, "rain_to_midnight"))["seeds"])
    t_no, f_no = numbers(R)

    # ---- where the missing rain falls, the clock, the design --------------------------------------------------------------------
    where = ""
    if C:
        where = (f"Of the rain in the 63 hours around the {count(C['n'])} flood onsets (from 09:00 of the day before the issue day to midnight of the target day), {pct(C['hidden'])} fell between about {D9} and midnight of the issue day, filed under the next day and "
                 f"so hidden from the model; {pct(C['table'])} fell before {D9} and {pct(C['future'])} falls on the target day (<code>results/onset_rain_clock.json</code>). ")
    summer = (f" ({C['summer_clock']} of the {C['n_gauges']} gauges have a flow day that follows the summer clock; the other {C['standard_clock']} lose one hour of rain in summer)"
              if C and "summer_clock" in C else "")
    four = four_experiments(R) + ". "
    chosen = choices_after_test(R)
    after = (f" No setting was chosen on the test years; the designs of the later experiments were fixed after the earlier test scores were known, and each results file lists its decisions "
             f"(<code>choices_made_after_looking_at_test_scores</code>: {_and(chosen)} entries)." if chosen else "")
    design = (f"<p>{four}{where}E1 and E1b give the model the rain of the whole issue day, 00:00 to 24:00 by the wall clock of the gauge with summer time counted{summer}, as "
              f"one extra daily input, so that no hour after the issue time is read. Design as in section 4.5: {_word(n_seeds)} seeds, a control that differs only in the thing tested; the success "
              f"rules were written before the runs (NEXT_STEPS_FOR_AGENTS.md).{after}</p>")

    # ---- the ladder and the figure (both floats: they move to the top of the next page where they do not fit, and the text fills the page) -----
    rows = _ladder(R, (f"about {W['nldas']}" if W else "in the evening", D9), (f"about {W['imerg']}" if W else "in the evening"))
    table = ""
    if rows:
        table = floated(ladder_table(rows, (
            f"<b>{t_no}. What the rain of the whole issue day adds, with the river gauge, test years.</b> Onsets warned and median NSE; large figures: the forecast averaged over the "
            f"{_word(n_seeds)} seeds; small grey figures: lowest and highest single training. Onsets: share of the {count(A['n'] if A else E['n'])} onsets, the 12 basins together. False warnings: "
            f"share of the onset warnings that were false, {one}.")))
    figure = ""
    if SHOW_FIGURE and (FIG / FIG_FILE).exists() and A:
        figure = floated(fig_cropped(FIG_FILE, (
            f"<b>{f_no}. What the rain of the whole issue day adds, {one}, with the river gauge.</b> Onsets warned (top) and median NSE (bottom), test years; filled: forecast averaged over the "
            f"{_word(n_seeds)} trainings, bar: lowest to highest training; hollow: controls (circle: the same rain one day older; diamond: the same product over the table&rsquo;s own window; "
            f"triangle: over the UTC day); shaded: the AORC rungs; under each name, the hour at which that rain&rsquo;s day ends. AORC is {AORC_WORDING} (hours before "
            f"2002 are disaggregated daily totals); the perfect forecast is an upper bound that cannot be had in practice. The numbers are in {t_no}."), FIG_SHARE, top=0.045, bottom=0.105))

    # ---- E1: ERA5, a stand-in ---------------------------------------------------------------------------------------------------
    e1 = ""
    if E:
        same = (f"the same as the NLDAS rain up to the evening ({pct(E['nldas'])}; {count(E['hits'])} hits each)" if E["hits"] == E["hits_nldas"] else
                f"against {pct(E['nldas'])} with the NLDAS rain up to the evening ({count(E['hits_nldas'])} hits against {count(E['hits'])})")
        period = ""
        if E.get("gain_before") is not None and E.get("gain_after") is not None and E.get("assimilation"):
            y, m, d = (int(x) for x in E["assimilation"].split("-"))
            period = (f" ERA5 assimilates radar and gauge rain only from {d} {MONTH[m]} {y} {cite('beck2019')}; the gain is "
                      f"{'not smaller' if E['gain_before'] >= E['gain_after'] else 'smaller'} before ({s1(E['gain_before'])} points) than after ({s1(E['gain_after'])}).")
        e1 = (f"<p><b>E1: model rain as a stand-in.</b> Hourly ERA5 rain, model output and not a measurement, takes the onsets warned {one} from {pct(E['base'])} to {pct(E['with_'])} (single trainings "
              f"{_prng(E['seeds'])}; control, the same rain one day older: {pct(E['control'])}): {share_word(E['gap'])} of the gap to the upper bound and {same}, so the success rule fixed in advance, "
              f"clearly above NLDAS with a flat control, {'is met' if E['met'] else 'is <b>not met</b>'}. The hours decide: ERA5 over the rain table&rsquo;s own window "
              f"({D9} to {D9}) warns of {pct(E['to09'])}, over the UTC day of {pct(E['utc'])}; the hours after {D9} are worth {n1(E['hours'])} points, {n1(E['evening'])} of them after {U19}.{period}</p>")

    # ---- E1b: AORC and the simple models ----------------------------------------------------------------------------------------
    e1b = simple = ""
    if A:
        d2, d3 = (A["nse_with"][k] - A["nse_base"][k] for k in LEADS[1:])
        flat = " (flat)" if A["control_flat"] else ""
        clear_era5 = (f"AORC is clearly above ERA5 ({s1(A['era5'])} points)" if A["era5_reading"] == "clear gain" and A["grows"]
                      else f"Against ERA5, AORC gives {s1(A['era5'])} points of onsets ({reading_word(A['era5_reading'])})")
        e1b = (f"<p><b>E1b: AORC.</b> AORC is {AORC_WORDING} {cite('aorc')}: daily gauge totals spread over the hours with radar, with a production lag of about "
               f"{_word(A['lag_days']) if A['lag_days'] else 'several'} days; its hours before 2002, all the training years, are disaggregated daily totals. As the extra input it takes the onsets warned "
               f"{one} from {pct(A['base'][first])} to {pct(A['with_'][first])} (single trainings {_prng(A['seeds'])}{'; ' + hindsight_clause(H) if H else ''}) and the median NSE from {n2(A['nse_base'][first])} to "
               f"{n2(A['nse_with'][first])}. The same AORC rain over the table&rsquo;s own window warns of {pct(A['to09'])}{flat}, over the UTC day of {pct(A['utc'])}: the hours after {D9} are worth "
               f"{n1(A['hours'])} points, {n1(A['evening'])} of them after {U19}. {clear_era5} and the success rule fixed in advance {'is met' if A['met'] else 'is not met'}. "
               f"{_leads(LEADS[1:]).capitalize()} the median NSE moves little ({signed(d2)} and {signed(d3)}) while the onsets warned rise from {pct(A['base'][LEADS[1]])} and "
               f"{pct(A['base'][LEADS[2]])} to {pct(A['with_'][LEADS[1]])} and {pct(A['with_'][LEADS[2]])}.</p>")
        g = A["gain"]
        as_much = min(g["ridge"], g["forest"]) >= AS_MUCH * g["lstm"]
        ra = {m: _reading(A["against"][m]) for m in ("linreg", "rf")}
        nse_say = (f"in median NSE the LSTM ends ahead ({n2(A['nse_with'][first])} against {n2(A['nse_simple']['ridge'])} and {n2(A['nse_simple']['forest'])})" if all(v == "lstm" for v in ra.values()) else
                   f"in median NSE the LSTM scores {n2(A['nse_with'][first])} against {n2(A['nse_simple']['ridge'])} and {n2(A['nse_simple']['forest'])} (section 4.2 reads "
                   f"{' and '.join(PHRASE[ra[m]] for m in ('linreg', 'rf'))})")
        names = {"ridge": "ridge regression", "forest": "the random forest"}
        more = [names[m] for m in ("ridge", "forest") if A["warned"][m] > A["warned"]["lstm"]]
        worse = max(A["far"][m] for m in ("ridge", "forest") if names[m] in more) if more else None
        extra = (f" {_and(more)[0].upper() + _and(more)[1:]} warns of more onsets, with {pct(worse)} of its warnings false against {pct(A['far']['lstm'])}; " if more else " ")
        ng = A["nse_gain"]
        times = ng["lstm"] / max(ng["ridge"], ng["forest"]) if max(ng["ridge"], ng["forest"]) > 0 else None
        twice = ("about twice as much" if times and 1.6 <= times < 2.6 else "more than twice as much" if times and times >= 2.6 else "more" if times and times > 1.1 else "")
        nse_gain = (f" and gains {twice} ({signed(ng['lstm'])} against {signed(ng['ridge'])} and {signed(ng['forest'])})" if twice else "")
        simple = (f"<p><b>The simple models.</b> Given the same AORC input, ridge regression warns of {pct(A['warned']['ridge'])} and the random forest of {pct(A['warned']['forest'])} of the onsets (the "
                  f"LSTM: {pct(A['warned']['lstm'])}): the input adds {n1(g['ridge'])} and {n1(g['forest'])} points to them against {n1(g['lstm'])} to the LSTM"
                  + (f", so for onsets the gain is {INFO_NOT_NET}." if as_much else ", so they gain less than the LSTM.") + extra + nse_say + nse_gain + ".</p>")

    # ---- hindsight and a real-time product --------------------------------------------------------------------------------------
    hind = ""
    if A and (H or S4):
        part = real = ""
        if H:
            best = "none" if H["best"] == 0 else f"{n1(H['best'])} points"
            part = (f"The independent review found no positive dependence of AORC&rsquo;s rain before midnight on the radar rain after it on {H['days_review']} test-year days (coefficient "
                    f"{signed(H['coef_review'])}, 95% interval {signed(H['ci_review'][0])} to {signed(H['ci_review'][1], 3)}); its bound is {hindsight_clause(H)} of the {pct(A['with_'][first])} "
                    f"({count(H['bound_hits'])} of the {count(A['hits'])} hits; best estimate: {best}). ")
        if S4:
            lo, hi = S4["interval"]
            far = (f"false warnings were counted only on the issue days of all basins ({pct(S4['far_stage4'])} with Stage IV, {pct(S4['far_aorc'])} with AORC), not on days without an onset"
                   if S4["no_far_on_other_days"] else "")
            arch = (f"whether the Iowa Mesonet archive holds the first posting of each hour or a later, gauge-corrected version is not established ({S4['replaced'][0]} of "
                    f"{count(S4['replaced'][1])} hours came from AORC instead)" if S4["archive_unestablished"] else "")
            tail = "; ".join(x for x in (far, arch) if x)
            real = (f"The decisive test is {REALTIME_WORDING}: Stage IV {cite('stage4')} takes the place of AORC on the issue day of all {count(S4['n'])} onsets in the saved networks, which warn of "
                    f"{count(S4['stage4'])} ({pct(S4['stage4'] / S4['n'])}) against {count(S4['aorc'])} with AORC and {count(S4['table'])} with the rain table alone, so {pct(S4['share'])} of the AORC "
                    f"effect survives (95% interval {pct(lo)} to {pct(hi)}); {count(S4['no_last'])} ({pct(S4['no_last'] / S4['n'])}) with the last hour of the issue day taken out, as the delay of "
                    f"Stage IV would. The count depends more on whether the issue day was a heavy-rain day than on its exact amount, so a service with these inputs might reach about "
                    f"{pct(S4['no_last'] / S4['n'])} to {pct(S4['stage4'] / S4['n'])}. " + (tail[0].upper() + tail[1:] + "." if tail else ""))
        hind = (f"<p><b>Hindsight, and a real-time product.</b> Because AORC is {AORC_WORDING}, its {pct(A['with_'][first])} could hold hindsight: hours before midnight scaled by a daily total that "
                f"includes later rain. {part}{real}</p>")

    # ---- verdicts, what was not done, the reviews -------------------------------------------------------------------------------
    note = ""
    if A:
        cr = criteria(R)
        v = set(A["verdict_onset"].values())
        on = (f"The verdict for the onsets {'stays' if v == {'NOT YET'} else 'is'} {_and(sorted(v))} at every lead time: with the AORC rain to midnight {pct(A['with_'][first])} are warned and "
              f"{pct(A['false'][first])} of the warnings are false (GO asks for at least {pct(cr['go_pod'])} warned and at most {pct(cr['go_far'])} false"
              + f"); for all high-flow days it is {A['verdict_all']} ({pct(A['all_pod'])} warned, {pct(A['all_far'])} false). ")
        undone = "Not done: a network trained on Stage IV; AORC without the river gauge. "
        note = f'<p class="nbx"><b>What follows, and what was not done.</b> {on}{undone}{review_clause(R, "rain_to_midnight", "aorc_rain_to_midnight")}</p>'

    return f"""{LAYOUT}{LADDER_STYLE}{head}
{design}
{table}
{figure}
{e1}
{e1b}
{simple}
{hind}
{note}"""
