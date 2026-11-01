"""Report section 4.7 - more rivers in training (E3: 63 training rivers instead of 12, for the same 12 rivers and for rivers never seen).

Reads results/more_basins.json (17_more_basins.py) through the shared reader rivers_facts of s0_summary.py, the ladder of Table 4.7 straight from the same file
(summary_test, seed_ranges and unseen), and the review record results/review_more_basins.json. Every number is read by key; every sentence whose wording depends on an outcome
is chosen from the stored numbers and readings (the readings of the gain rule of the file: clear gain, no clear difference, mixed). Nothing is computed beyond differences, shares of
stored counts and formatting.

What leads: the 12 rivers the rest of the report scores, with the river gauge, 1 day ahead - the result a reader asks for first - and then, as the reason E3 was run, the rivers never
seen. The onsets are worded by the rule that the independent review set: a lower share of onsets warned with fewer false alarms and the same CSI is a moved operating point, not lost
skill (s0_summary.MOVED_POINT, chosen by moved_operating_point from the stored hits, false alarms and CSI); the review forbade calling it a clear loss. The arm with five times the
update limit is called what it is (five times the update limit and the patience window; its kept networks lie where those of the plain arm lie), never 'equal training per river'.
The number of rivers, their variety and their nearness to the 12 changed together between the two designs for rivers never seen; the text says so and that the gain does not rank with
the distance, and credits the gain neither to scale nor to proximity. The literature is quoted for its direction only (the numbers of Kratzert et al., 2024 were read by eye from a plot and
the record asks that they be confirmed before a report quotes them).

Layout: Table 4.7 (the ladder with seed ranges in small grey type; one training, no range, for the earlier leave-one-out models) and the figure as a float, numbered after the figure of
section 4.6. The table and figure numbers are read by section 8 through numbers().
"""
import sys

from common import FIG
from flood import LEADS
from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed
from report_sections.s0_summary import (LADDER_STYLE, LAYOUT, MOVED_POINT, _and, _file, _leads, _word, by_lead_reading, criteria, fig_cropped, floated, ladder_table, moved_operating_point,
                                         review_clause, reading_word, rivers_facts)
from report_sections import s4d_midnight
from report_sections.s4c_timing import last_figure

T_NO = "Table 4.7"
SHOW_FIGURE = True                              # figures/more_basins.png: drawn for 193 mm and 220 mm high, so it takes more than half a page even at 66%
FIG_SHARE = 56                                  # % of the text width (184 mm)
FIG_FILE = "more_basins.png"
lk = lambda k: f"l{k}"


def numbers(R):
    """(table number, figure number or None) of this section, for the pointers of section 8."""
    before = int(bool(s4d_midnight.SHOW_FIGURE and (FIG / s4d_midnight.FIG_FILE).exists()))
    return T_NO, (f"Figure 4.{last_figure(R) + before + 1}" if SHOW_FIGURE and (FIG / FIG_FILE).exists() else None)


def _rows(X):
    """The rows of Table 4.7 from the results file: the same 12 rivers with the river gauge (12 / 63 / 63 with five times the limit), then the rivers never seen with and without basin
    properties (11 / 50 training rivers); None where the file is not in the expected form."""
    try:
        T, SR, U = X["summary_test"]["sets"], X["seed_ranges"]["sets"], X["unseen"]["sets"]
        same = lambda label, s, cls="": dict(label=label, cls=cls, on={k: T[s][lk(k)]["onset"]["pod"] for k in LEADS}, nse={k: T[s][lk(k)]["nse_median"] for k in LEADS},
                                             on_r={k: SR[s][lk(k)]["onset_pod"] for k in LEADS}, nse_r={k: SR[s][lk(k)]["nse_median"] for k in LEADS})

        def never(label, s):
            many = U[s].get("trainings", 1) > 1                                  # one training per river (section 4.4): no range to show
            return dict(label=label, on={k: U[s]["own_level"][lk(k)]["onset"]["pod"] for k in LEADS}, nse={k: U[s]["nse_median"][lk(k)] for k in LEADS},
                        on_r={k: U[s]["own_level"][lk(k)]["seed_ranges"]["onset_pod"] for k in LEADS} if many else {}, nse_r={k: U[s]["nse_seed_range"][lk(k)] for k in LEADS} if many else {})
        n = {s: U[s]["n_training_rivers"] for s in ("lobo_w_static", "unseen50_w_static", "lobo_w_nostatic", "unseen50_w_nostatic", "lstm63_w")}
        return [dict(head="The same 12 rivers, with the river gauge (the 12-river row is the model of section 4.2)"),
                same("12 training rivers", "lstm_wq"), same("63 training rivers", "lstm63_wq"), same("63 training rivers, five times the update limit and patience window", "lstm63eq_wq"),
                dict(head="Rivers never seen, past weather only (onsets at the model&rsquo;s own warning level, as in section 4.4)"),
                never(f"with basin properties, {n['lobo_w_static']} training rivers (one training per river)", "lobo_w_static"),
                never(f"with basin properties, {n['unseen50_w_static']} training rivers", "unseen50_w_static"),
                never(f"without basin properties, {n['lobo_w_nostatic']} training rivers (one training per river)", "lobo_w_nostatic"),
                never(f"without basin properties, {n['unseen50_w_nostatic']} training rivers", "unseen50_w_nostatic")]
    except (KeyError, TypeError, IndexError) as exc:
        print(f"s4e_rivers: results/more_basins.json is not in the expected form for Table 4.7 ({type(exc).__name__}: {exc}); the table is left out", file=sys.stderr)
        return None


def html(R, T):
    F = rivers_facts(R)
    head = '<h3 class="clr">4.7 More rivers in training</h3>'
    if not F:
        print("s4e_rivers: results/more_basins.json was not found or is not in the expected form; the section is a placeholder", file=sys.stderr)
        return f"{head}<p>The experiment with more training rivers (code/17_more_basins.py) has not been run yet.</p>"
    X = _file(R, "more_basins")
    first = LEADS[0]
    one = LEAD_WORD[first]
    n63, n50, n11, n12 = (F["n_train"][s] for s in ("lstm63_w", "unseen50_w_static", "lobo_w_static", "lstm_w"))
    t_no, f_no = numbers(R)
    n_seeds = F["n_seeds"]

    design = (f"<p>Published models train on hundreds of rivers; in a same-day simulation task the median NSE of an LSTM rose with every increase in the number of training rivers up to "
              f"{('all ' + str(F['published_basins'])) if F['published_basins'] else 'the full benchmark set'} ({cite('kratzert2024')[1:-1]}; another task and network, so only the direction carries over). "
              f"This model saw {n12}. E3 trains it on the {n63} complete south-eastern CAMELS basins and scores the same {n12} rivers against the stored {n12}-river model ({_word(n_seeds)} seeds, with and without "
              f"the river gauge; a second arm gives the {n63}-river model five times the update limit and patience window). "
              f"For rivers never seen, a model trained on {n50} of the {n63} scores the {n12} against the leave-one-out test of section 4.4 ({n11} training rivers).</p>")

    # ---- the same 12 rivers --------------------------------------------------------------------------------------------------------
    h0, h1 = F["hits"]
    fa0, fa1 = F["false_alarms"]
    p0, p1 = F["onset"]["lstm_wq"][first], F["onset"]["lstm63_wq"][first]
    far0, far1 = fa0 / (h0 + fa0), fa1 / (h1 + fa1)
    moved = moved_operating_point(p0, p1, far0, far1, F["csi"][0], F["csi"][1]) if p1 < p0 else False
    n12_nse, n63_nse = F["nse"]["lstm_wq"], F["nse"]["lstm63_wq"]
    lower = f"fall from {pct(p0)} to {pct(p1)}" if p1 < p0 else f"go from {pct(p0)} to {pct(p1)}"
    onset = (f"the onsets warned {lower}"
             + (f": {MOVED_POINT} ({h0 - h1} fewer hits but {fa0 - fa1} fewer false alarms, CSI {n2(F['csi'][0], 3)} against {n2(F['csi'][1], 3)}; {pct(F['chosen'][1], 1)} against "
                f"{pct(F['chosen'][0], 1)} at the warning level chosen on the validation years)"
                if moved else f" ({reading_word(F['onset_reading'])} by the seed rule; hits {h0} to {h1}, false alarms {fa0} to {fa1})"))
    w12, w63 = (F["nse_range"][s][1] - F["nse_range"][s][0] for s in ("lstm_wq", "lstm63_wq"))
    steady = ", with steadier trainings" if w63 < w12 / 2 else ""
    eq_same = F["best_update"][2] <= 1.1 * F["best_update"][1]
    arm = (f"The arm with five times the update limit kept the same early networks (best update about {count(F['best_update'][2])} against {count(F['best_update'][1])}): a longer "
           f"patience window does not move the optimum, and it did not test equal training per river." if eq_same else
           f"The arm with five times the update limit kept later networks (best update about {count(F['best_update'][2])} against {count(F['best_update'][1])}).")
    same = (f"<p><b>The same {n12} rivers.</b> With the river gauge, {n63} training rivers give {by_lead_reading(F['nse_reading'])} in median NSE ({n2(n12_nse[first])} to {n2(n63_nse[first])}, "
            f"{n2(n12_nse[LEADS[1]])} to {n2(n63_nse[LEADS[1]])}, {n2(n12_nse[LEADS[2]])} to {n2(n63_nse[LEADS[2]])}){steady}. {onset[0].upper() + onset[1:]}. {arm}</p>")

    # ---- rivers never seen ---------------------------------------------------------------------------------------------------------
    un = F["unseen"]
    say = by_lead_reading(F["unseen_reading"]["w_static"])
    unseen = (f"<p><b>Rivers never seen.</b> With basin properties the median NSE {one} rises from {n2(un['lobo_w_static'][first])} with {n11} training rivers to "
              f"{n2(un['unseen50_w_static'][first])} with {n50} (without properties {n2(un['lobo_w_nostatic'][first])} to {n2(un['unseen50_w_nostatic'][first])}): {say} in the seed average; each "
              f"single {n50}-river training is {signed(min(F['single_gain']))} to {signed(max(F['single_gain']))} above the earlier test, which is one training per river (single-training p "
              f"{n2(min(F['single_p']))} to {n2(max(F['single_p']))}). The loss from not having seen the river, {n2(F['loss'][0])} NSE in the typical river, shrinks to {n2(F['loss'][1])}. "
              f"The number of training rivers, their variety and their nearness to the {n12} changed together between the two designs (the nearest training gauge: a median of "
              f"{count(F['km'][0])} km with {n11} rivers, {count(F['km'][1])} km with {n50}) and were not separated; the gain does not rank with the distance to the nearest training gauge "
              f"(rank correlation {signed(F['rho'][0])}, p = {n2(F['rho'][1])}), so it can be credited neither to scale nor to proximity alone.</p>")

    # ---- the table, the figure, what follows ----------------------------------------------------------------------------------------
    rows = _rows(X)
    table = ""
    if rows:
        table = floated(ladder_table(rows, (
            f"<b>{t_no}. Training on more rivers, test years.</b> Onsets warned and median NSE; large figures: the forecast averaged over the {_word(n_seeds)} seeds; small grey figures: lowest and "
            f"highest single training, none where a model is one training per river. Onsets: share of the {count(F['n_onsets'])} onsets, at the gauge&rsquo;s high-flow level (first block) "
            f"or the model&rsquo;s own level (second block)."), far=False, label="Training rivers"))
    figure = ""
    if SHOW_FIGURE and (FIG / FIG_FILE).exists():
        figure = floated(fig_cropped(FIG_FILE, (
            f"<b>{f_no}. Training on more rivers: the same {n12} rivers (top) and rivers never seen (bottom), {one}.</b> Bars: forecast averaged over the {_word(n_seeds)} trainings, dots: single "
            f"trainings (the grey {n11}-river bars are one training per river); outline bars: the same rivers when in training; the number under a bar is the number of rivers in "
            f"training. Top: a warning above the high-flow level of the river&rsquo;s flow record; bottom, for rivers without a gauge record to read: above the 95th percentile of the model&rsquo;s own forecasts. The numbers are in {t_no}."), FIG_SHARE, top=0.04, bottom=0.145))
    cr = criteria(R)
    v = {s: {X["verdicts"]["sets"][s][lk(k)]["plain"]["onset"]["verdict"] for k in LEADS} for s in ("lstm63_wq", "unseen50_w_static")} if X else {}
    allv = set().union(*v.values()) if v else set()
    note = (f'<p class="nbx"><b>What this shows, and what was not done.</b> More rivers are not where the gauged forecast&rsquo;s skill will come from: the information about tomorrow&rsquo;s '
            f"rain is missing; they matter for rivers without a gauge. The verdict for the onsets is {_and(sorted(allv)) if allv else 'not read'} at every lead time. Not done: the simple-model "
            f"yardstick at {n63} rivers; a buffered leave-out or a dose-response over subsets of the {n50}, which would separate scale from nearness. {review_clause(R, 'more_basins')}</p>")
    return f"""{LAYOUT}{LADDER_STYLE}{head}
{design}
{figure}
{table}
{same}
{unseen}
{note}"""
