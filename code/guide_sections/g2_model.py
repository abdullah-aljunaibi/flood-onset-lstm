"""Guide, part 2 - steps 6 to 9: the LSTM, how it is trained, one network for all twelve rivers, and whether it beat the simple models.

Written for readers who have never trained a model: short sentences, no formula, and a technical word only after its concept box
(guide_common.WORDS) or explained in the sentence that uses it. Reads results/lstm_single.json, hparam_log.csv, training_curves.json,
lstm_all.json, lstm_lobo.json and baselines.json by key. Nothing is computed here beyond differences, counts, smallest/largest and
formatting.

The project built the LSTM twice, and the text keeps one name for each from step 6 on: 'the one-river network' (ONE; the small network
of 03_lstm.py, with the settings its search chose) and 'the network for all twelve rivers' (ALL; 04_lstm_all.py). The left column of the
learning curves is a training on the same one river with the BASE setting of that search, so the text says 'on one river' there.

Every sentence that holds a number, or whose wording depends on an outcome, is built from the stored numbers inside a guard: when a
results file or a key is missing the sentence is left out and one line goes to stderr; when the stored numbers no longer support a
wording, a plainer sentence takes its place. A picture that does not exist (yet) is left out in the same way. The readings 'no clear
difference', 'only the averaged forecast is ahead' and the state of milestone 2 are not judged here: they come from the report
(s0_summary.facts, the rule of report section 4.2), and the unseen-river numbers from s0_summary.lobo_facts.

Layout: step 9 starts on a new page (NEW_PAGE), because its first picture fills most of a page; step 6 follows step 5 without a break
(a forced break there left most of a page empty whenever step 5 ended near the top of one). Every picture is shown at the full text
width, as a float (STYLE, the rule of the report: s0_summary.LAYOUT): a picture that does not fit on the rest of a page moves to the
top of the next page, and the paragraphs and boxes after it fill the page. The last lines of a step clear the floats, so a step always
ends below its pictures. The heading of a step, its lead and its first paragraph are one block (.keep), and so are 'What to remember'
and the line 'Full detail'; no heading is left alone at the foot of a page. The one exception to the full width is the drawing of
step 8: at the full width its lettering would be as large as a heading, so it is shown at the width at which its lettering matches the
pictures (guide_common.drawing_width). On a screen (Project_Guide.html) the floats are ordinary blocks in the order of the text.
"""
import datetime
import math
import sys

from common import FIG
from flood import FORCING, LEADS
from guide_common import concept, final_detail_for, drawing_width, keep, remember, small_figure, step
from report_common import cite, count, fig, n2
from guide_sections.g5_next import available as step16_available
from report_sections.s0_summary import _word, facts, lobo_facts
from report_sections.s1_data import _timeline
from report_sections.s3_single import MEM_TOL
from report_sections.s4_all import STATIC_WORD
from report_sections.s4_all import _say as reading_words

NEW_PAGE = True                                 # step 9 starts on a new page; see the note on layout above
# layout only (see the docstring): the float for the pictures, what clears it, the block that keeps a heading with its first paragraph,
# and the boxes as blocks of their own, so that none slides under a picture
STYLE = ("<style>.g2 .ff{float:left;width:100%;break-inside:avoid} .g2 .ff figure{margin-top:2pt} .g2 .keep,.g2 .remember,.g2 .who{clear:both} "
         ".g2 .keep{break-inside:avoid} .g2 .concept{display:flow-root}</style>")
FIRST = LEADS[0]                                # the lead time the text quotes: 1 day ahead
QUOTED_CAP = "&lsquo;Tomorrow = today&rsquo;"
ONE, ALL = "the one-river network", "the network for all twelve rivers"                  # the two fixed names of the two LSTMs (see the docstring)
N_RIVERS = 12                                   # the number of rivers in the name ALL; html() says so on stderr if the results hold another number
SHOWN_PROPERTIES = ("log_area", "slope_mean", "frac_forest", "soil_depth_pelletier")      # the basin properties named as examples in step 8
lk = lambda k: f"l{k}"


# ---- small helpers ---------------------------------------------------------------------------------------------------
def _note(message):
    print(f"g2_model: {message}", file=sys.stderr)


def _read(what, make):
    """make() -> numbers read from the results files; None, and one line on stderr, when a file or a key is missing."""
    try:
        return make()
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as e:
        _note(f"{what}: not read from the results files ({type(e).__name__}: {e}); the sentences that need it are left out")
        return None


def _p(*sentences):
    """One paragraph from the sentences that exist."""
    text = " ".join(s for s in sentences if s)
    return f"<p>{text}</p>" if text else ""


def _caption(title, how):
    """A caption: the bold name of the picture, then how to read it."""
    return f"<b>{title}</b> <b>How to read it:</b> {how}"


def _picture(name, title, how):
    """A picture of figures/ at the full text width, as a float (see STYLE). '' and one line on stderr while the file does not exist."""
    if not (FIG / name).exists():
        _note(f"figures/{name} does not exist; the picture is left out")
        return ""
    return f'<div class="ff">{fig(name, _caption(title, how))}</div>'


def _closing(text, sections):
    """The end of a step: 'What to remember' and the pointer to the report, kept together below the pictures of the step."""
    return keep(remember(text), final_detail_for(sections))


def _opening(n, lead, first, new_page=False):
    """The heading of step n, its lead and its first paragraph as one block that is not split over two pages; with new_page the
    step starts a page."""
    head = step(n, lead)
    if new_page and NEW_PAGE:
        paged = head.replace('<h2 class="step"', '<h2 class="step pb" style="margin-top:0"', 1)
        if paged == head:
            _note(f"the heading of step {n} is not in the expected form; the step does not start a new page")
        head = paged
    return f'<div class="keep">{head}\n{first}</div>'


def _span(values, fmt=count):
    """[500, 600, 800] -> '500 to 800'; one value when both ends print alike."""
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def _three(values, fmt=n2):
    """Three values (one, two and three days ahead) as 'a, b and c'."""
    v = [fmt(x) for x in values]
    return ", ".join(v[:-1]) + f" and {v[-1]}"


def _river(S):
    """'French Broad River' from the name of the gauge of the one-river experiment."""
    B = S["basin"]
    return B["name"].split(" AT ")[0].title() if " AT " in B["name"] else B["short"]


# ---- numbers that more than one step uses ------------------------------------------------------------------------------
def _sizes(R):
    """The two networks with past weather + river gauge: memory cells, inputs a day and weights. The stored weight count is checked
    against the cells and inputs read here (an LSTM layer has four blocks of cells x (inputs + cells + 2) weights; the output layer
    has one weight per cell and one more for each lead time)."""
    S, A = R["lstm_single"], R["lstm_all"]
    n_w, n_out = len(FORCING), len(LEADS)
    one, cost = S["finals"]["wq"]["seeds"], A["sets"]["wq"]["cost"]
    z = dict(cells1=S["tuning"]["chosen"]["hidden"], in1=n_w + 1, w1=one[min(one, key=int)]["training"]["n_params"],
             cells12=A["config"]["hidden"], n_props=len(A["static_inputs"]), w12=cost[min(cost, key=int)]["n_params"], n_w=n_w)
    z["in12"] = n_w + 1 + z["n_props"]
    for cells, n_in, stored in ((z["cells1"], z["in1"], z["w1"]), (z["cells12"], z["in12"], z["w12"])):
        if 4 * cells * (n_in + cells + 2) + n_out * cells + n_out != stored:
            raise ValueError(f"the stored weight count {stored} is not that of {cells} memory cells with {n_in} inputs a day")
    return z


# ---- step 6: the LSTM --------------------------------------------------------------------------------------------------
def _step6(R, T, Z):
    def memory():                                                                 # the memory test of 03_lstm.py, one day ahead
        S = R["lstm_single"]
        M, days = S["memory"]["test_nse_by_lead"], S["memory"]["contexts_days"]
        trained = S["tuning"]["chosen"]["warmup"]
        nse = lambda inp, c: M[inp][str(c)][LEADS.index(FIRST)]
        days = sorted(days)
        short, mid, year = min(days), min(days, key=lambda c: abs(c - trained)), max(c for c in days if c <= 365)
        level = lambda first: [nse("wq", c) for c in days if first <= c <= year]
        flat_from = next((c for c in days if c < year and max(level(c)) - min(level(c)) <= MEM_TOL), None)   # the upper line is level from here on
        return dict(river=_river(S), days=days, short=short, mid=mid, year=year, trained=trained, nse=nse, flat_from=flat_from,
                    old_helps=nse("w", year) - nse("w", mid) > MEM_TOL,                                    # weather older than `mid` days still helps
                    gauge_enough=nse("wq", year) - nse("wq", mid) <= MEM_TOL)                              # ... and not with the river gauge

    M = _read("the memory test (results/lstm_single.json)", memory)
    history = _read("the days of history (results/lstm_all.json)", lambda: R["lstm_all"]["config"]["warmup"])
    river = _read("the river of the one-river network (results/lstm_single.json)", lambda: _river(R["lstm_single"]))

    intro = _p(
        "A river remembers.",
        "Today&rsquo;s flow depends on the rain of the past days and months, because the basin stores water as wet soil and groundwater.",
        "The simple models of step 5 were handed this store ready-made, as sums of the rain over fixed numbers of past days.",
        "The model of this step gets only the daily values.",
        "It has to learn for itself what is worth remembering.",
        f"Networks of this kind are the usual choice for forecasting river flow {cite('kratzert2018')}.",
        "The project built such a network twice.",
        (f"First a small one for a single river, the {river}, to try out settings and to look inside it: {ONE}." if river else
         f"First a small one for a single river, to try out settings and to look inside it: {ONE}."),
        f"Then the main one: {ALL} (step 8).",
        "Four ideas explain how it works, each in a box: weights, a network of units, the scale of its inputs, and a memory.")

    here_weights = ("Ridge regression (step 5) has one weight for each of its summaries of the past weather. "
                    + (f"With the river gauge, {ALL} has {count(Z['w12'])} weights, and {ONE} {count(Z['w1'])}." if Z else ""))
    here_network = (f"With the river gauge, {ALL} takes in {Z['in12']} numbers a day: {_word(Z['n_w'])} weather values, the gauge "
                    f"and {_word(Z['n_props'])} fixed basin properties. It gives out {_word(len(LEADS))}: the flow one, two and "
                    f"three days ahead." if Z else "")
    here_scale = ("Rain is counted in millimetres, temperature in degrees and day length in seconds: numbers of very different size. Each input "
                  "gets its average and its spread from the training years; taking them from all years would be a small leak (step 3). The "
                  "fixed basin properties are put on one scale across the basins.")
    here_lstm = (f"LSTM is short for long short-term memory {cite('hochreiter1997')}."
                 + (f" Before any forecast counts, {ALL} has read at least {count(history)} days"
                    + (": a year to fill its memory." if history == 365 else ".") if history else ""))

    cells = (f"The memory is made of <i>memory cells</i>: {Z['cells12']} in {ALL}, each holding one number."
             if Z else "The memory is made of <i>memory cells</i>, each holding one number.")
    gates = _p(                                                                   # how the gates are set is said in the caption of the picture
        cells,
        "Think of a cell as a small store of water.",
        "Every day three <i>gates</i> act on it.",
        "The forget gate sets how much of yesterday&rsquo;s content is kept; the rest drains away.",
        "The input gate sets how much of today&rsquo;s news, such as today&rsquo;s rain, is let in.",
        "The output gate sets how much is passed on: to the forecast, and to the gates of the next day.",
        "Each gate is a number between 0 (shut) and 1 (open).",
        "Nobody tells the network that a cell should stand for wet soil: training (step 7) decides what each cell stores.")
    cell_picture = _picture(
        "guide_lstm_cell.png", "One memory cell, drawn as a store of water with three taps.",
        "Follow the wide blue line from left to right: yesterday&rsquo;s memory comes in, and today&rsquo;s memory goes on to tomorrow. "
        "Three taps act on the way: the first keeps part of the old memory, the second adds today&rsquo;s news from below, the third passes a "
        "copy to the forecast. The thin grey arrows show that the network sets all three taps afresh every day, from today&rsquo;s inputs and "
        "from what the cells passed on yesterday.")

    used = memory_picture = ""
    if M:
        nse, days = M["nse"], M["days"]
        steps = (f": first {days[0]} days, then {days[1]}, {days[2]} and so on" if len(days) > 3 else "")
        enough = "a few weeks" if 14 < M["mid"] <= 45 else f"{M['mid']} days"         # '30 days' is said as 'a few weeks'
        used = _p(
            "How much of the past does the network really use?",
            f"In a test, {ONE} was given only the last few days before each forecast{steps}.",
            "Each time it was scored again.",
            (f"Without the river gauge, its NSE one day ahead rose from {n2(nse('w', M['short']))} with {M['short']} days of history to "
             f"{n2(nse('w', M['year']))} with {M['year']} days."),
            (f"With the river gauge, {enough} were enough: {n2(nse('wq', M['mid']))} with {M['mid']} days, against "
             f"{n2(nse('wq', M['year']))} with {M['year']} days." if M["gauge_enough"] else
             f"With the river gauge the NSE was {n2(nse('wq', M['mid']))} with {M['mid']} days and {n2(nse('wq', M['year']))} with {M['year']} days."),
            ("A likely reading: the gauge already tells the network how wet the basin is."
             if M["old_helps"] and M["gauge_enough"] else ""))
        level = ("is already flat after one to two weeks" if M["flat_from"] and M["flat_from"] <= 14 else
                 f"is flat from {M['flat_from']} days on" if M["flat_from"] else "")
        lines = ("The light blue line (past weather only) keeps rising, so older weather still helps; the dark blue line (with the river gauge) "
                 f"{level}. " if M["old_helps"] and level else
                 "The light blue line is the network with past weather only, the dark blue line the network with the river gauge. ")
        memory_picture = _picture(
            "lstm_memory.png", f"How much history {ONE} uses (test years).",
            "From left to right the network may read more days before each forecast; higher is better. " + lines
            + f"This small network was trained with {M['trained']} days of history. In the grey area it reads fewer days than that, so read "
              f"that part with care.")

    return "\n".join(x for x in (
        _opening(6, "This step opens up the model that the project is about: what it is made of, how it remembers, and how much of the past "
                    "it uses.", intro),
        concept("weights", here=here_weights),
        concept("network", here=here_network),
        concept("normalising", here=here_scale),
        concept("lstm", here=here_lstm),
        cell_picture, gates, memory_picture, used,                # the two pictures follow each other, so that one page can hold both
        _closing("An LSTM reads the days one after another and keeps a memory: a set of numbers that three gates change every day. What the "
                 "memory holds is not programmed. It is learned. That is the difference from the regression of step 5: there the summaries "
                 "of the past were made by hand; here the network learns its own.", "section 3")) if x)


# ---- step 7: training --------------------------------------------------------------------------------------------------
def _step7(R, T, Z):
    def loop():                                                                   # the trainings of the all-river network with the river gauge
        A = R["lstm_all"]
        cost = [A["sets"]["wq"]["cost"][k] for k in sorted(A["sets"]["wq"]["cost"], key=int)]
        return dict(years=(A["split"]["train"][0][:4], A["split"]["train"][1][:4]), runs=[c["updates_run"] for c in cost],
                    minutes=math.ceil(max(c["seconds"] for c in cost) / 60), cfg=A["config"])

    def rates():                                                                  # the learning rates tried on one river, and the check on all rivers
        p = R["lstm_single"]["tuning"]["per_knob"]["lr"]
        rising = all(a < b for a, b in zip(p["val_nse_mean"], p["val_nse_mean"][1:])) and p["chosen"] == max(p["values"])
        A = R["lstm_all"]
        table = A["setting_check"]["table"]                                       # the settings compared on all rivers, on the validation years
        kept = [r for r in table if r["chosen"]]
        if len(kept) != 1 or kept[0]["lr"] != A["config"]["lr"]:
            raise ValueError("the setting check of results/lstm_all.json does not name the learning rate that the network uses")
        return dict(values=p["values"], scores=p["val_nse_mean"], rising=rising, all12=A["config"]["lr"],
                    checked=sorted({r["lr"] for r in table}), n_checked=len(table), minutes=min(r["seconds"] for r in table) >= 60)

    def curves():                                                                 # the stored learning curves (12_training_curves.py)
        TC = R["training_curves"]
        if TC.get("quick_mode"):
            raise ValueError("results/training_curves.json is the output of a smoke test")
        one, mid, right = TC["panels"]["one_river"], TC["panels"]["all_wq"], TC["panels"]["all_wqf"]
        t0, P = one["per_training"][0], TC["passes"]
        over = t0["last_update"] > t0["best_update"] and t0["val_nse_at_last"] < t0["val_nse_at_best"] and t0["train_loss_last"] < t0["train_loss_at_best"]
        on = {t["updates_after_best"] for t in mid["per_training"]}               # how long each all-river training ran on after its best check
        same = P["one_river_kept_update"] == t0["best_update"]                    # the passes stored are those of the training that is drawn
        return dict(t0=t0, over=over, kept=(mid["best_update_min"], mid["best_update_max"]), every=TC["check_every"],
                    patience=TC["settings"]["patience"], hundreds=min(mid["best_update_min"], right["best_update_min"]) >= 200,
                    ran_on=on.pop() if len(on) == 1 else None, mean_of_leads="mean of 1 to 3 days ahead" in one["validation_nse_is"],
                    years_per_update=round(P["scored_days_per_update"]["one_river"] / 365), years=round(one["training_days"] / 365.25),
                    passes=P["one_river_passes_at_kept_update"] if same else None,
                    as_many=P["all_rivers_updates_with_as_many_passes"] if same else None)

    def tuning():                                                                 # the settings tried on one river, and the seed spread
        S = R["lstm_single"]
        TU = S["tuning"]
        d = TU["per_knob"]["dropout"]
        h = TU["per_knob"]["hidden"]
        return dict(knobs=TU["per_knob"], n_runs=len(T["hparam_log"]), seeds=TU["base_val_nse_by_seed"], noise=TU["noise_seed_spread"],
                    minutes=S["runtime_s"] / 60, dropout_flat=max(d["val_nse_mean"]) - min(d["val_nse_mean"]) < TU["noise_seed_spread"],
                    small=h["chosen"] == min(h["values"]) and h["chosen"] in h["within_noise_of_best"] and h["best"] == max(h["values"]),
                    same_dropout=TU["chosen"]["dropout"] == R["lstm_all"]["config"]["dropout"], dropout=TU["chosen"]["dropout"])

    def too_simple():                                                             # the calendar-day average of step 5: a model that ignores the weather
        r = next((r for r in R["baselines"]["summary_table"] if r["model"] == "base_climatology" and r["lead"] == FIRST), None)
        if r is None or not abs(r["nse_median"]) < 0.1:
            raise ValueError("the calendar-day average is not in the table of the simple models, or does not score near 0")
        return r["nse_median"]

    L = _read("the trainings of the all-river network (results/lstm_all.json)", loop)
    LR = _read("the learning rates (results/lstm_single.json, results/lstm_all.json)", rates)
    C = _read("the learning curves (results/training_curves.json)", curves)
    TU = _read("the tuning (results/lstm_single.json, results/hparam_log.csv, results/lstm_all.json)", tuning)
    SIMPLE = _read("the calendar-day average (results/baselines.json)", too_simple)

    the_loop = _p(
        "Training is a loop of four moves.",
        "The network makes forecasts for days whose flow is already known.",
        "The error of these forecasts is measured.",
        "Every weight is changed a little, so that the error becomes smaller.",
        "Then the loop starts again with other days.")
    here_training = (f"The days with a known answer are the training years, {L['years'][0]} to {L['years'][1]}. One training of {ALL} went "
                     f"through the loop {_span(L['runs'])} times; each round is one update (see below). It took less than {L['minutes']} "
                     f"minutes on a laptop." if L else "")
    here_loss = ("The squared error of the forecasts: each error is multiplied by itself, so a large error counts far more than a small one. "
                 "Each river&rsquo;s error is set against how much that river&rsquo;s flow varies, so that wet and dry rivers count nearly "
                 "alike.")

    downhill = _p(
        "How does the computer know which way to change a weight?",
        "Picture the loss as a hilly landscape: every place is one choice of weights, and the height of the ground is the loss there.",
        "Training stands on a slope, works out which way is downhill and how steep the ground is, takes a step that way, and looks again.")
    here_gradient = ((f"The program library used here, PyTorch, works out the slopes of all {count(Z['w12'])} weights of {ALL} at once. " if Z else "")
                     + "The project uses a common variant of gradient descent called Adam, which also adjusts the step of each weight as it goes.")
    gradient_picture = _picture(
        "guide_gradient.png", "Gradient descent with three learning rates.",
        "Each curve is the loss for different values of one weight; the bottom of the valley is the best value. The dots are steps of "
        "training from the same start, and every step sets off downhill. In each panel the learning rate stays the same, yet the steps "
        "differ in size. A step is long where the slope is steep and short where the valley gets flat. With a small learning rate the steps "
        "creep. With the right one they reach the bottom. With one too large, each step overshoots the valley and lands higher on the other "
        "side.")
    here_rate = ""
    if LR:
        tried = f"On one river {_word(len(LR['values']))} learning rates were tried, from {min(LR['values']):g} to {max(LR['values']):g}."
        rose = (f" The larger the rate, the higher the NSE in the validation years (averaged over the {_word(len(LEADS))} lead times): from "
                f"{n2(LR['scores'][0])} to {n2(LR['scores'][-1])}. So the largest was kept for {ONE}. The best value lies at the edge of what "
                f"was tried, so a larger one might do better. A search should go on until the best value lies inside the range."
                if LR["rising"] else "")
        own = (f" {ALL.capitalize()} had a small check of its own, which compared "
               f"{' and '.join(f'{v:g}' for v in LR['checked'])} and kept {LR['all12']:g}." if len(LR["checked"]) == 2 else
               f" {ALL.capitalize()} uses {LR['all12']:g}.")
        here_rate = tried + rose + own
    here_update = ""
    if L and C:
        stretch = "a year of daily forecasts" if L["cfg"]["block"] == 365 else f"{L['cfg']['block']} days of forecasts"
        here_update = (f"A batch here is {L['cfg']['batch']} stretches of the training years, drawn at random; each stretch gives {stretch} for "
                       f"one river. {ALL.capitalize()} was at its best after {_span(C['kept'])} updates."
                       + (f" It then ran on for {count(C['ran_on'])} more, to make sure that no better score came (early stopping, below)."
                          if C["ran_on"] else ""))

    watch = here_over = here_stop = curves_picture = ""
    if C:
        t0 = C["t0"]
        passes = C["over"] and C["passes"] and L and L["cfg"]["batch"] == C["years_per_update"]       # an update of 32 stretches of a year each
        watch = _p(                                                               # two short paragraphs: the check, then what the picture shows
            "A falling loss proves little, because the loss is measured on the training years.",
            "The network could simply be learning those years by heart.",
            "So at regular checks the network is also scored on the validation years, which it never learns from.",
            (f"This validation score is the NSE in the validation years, averaged over the {_word(len(LEADS))} lead times."
             if C["mean_of_leads"] else "")) + "\n" + _p(
            "The picture below shows both curves in three cases: one river alone (left), all twelve rivers (middle), and all twelve with a "
            "perfect weather forecast (right; step 8 explains it).",
            (f"On one river (left), the validation score is best at update {count(t0['best_update'])} and then falls, while the training loss "
             f"keeps falling." if C["over"] else ""),
            (f"An update holds {C['years_per_update']} years of forecasts, so by update {count(t0['best_update'])} the network had gone through "
             f"its {C['years']} training years about {C['passes']:.0f} times." if passes else ""),
            "An AI course calls one such pass an epoch." if passes else "")
        if C["over"]:
            here_over = (f"On one river, the validation score peaks at update {count(t0['best_update'])} (NSE {n2(t0['val_nse_at_best'])}) and has "
                         f"fallen to {n2(t0['val_nse_at_last'])} by update {count(t0['last_update'])}. Over the same updates the training loss "
                         f"falls from {n2(t0['train_loss_at_best'])} to {n2(t0['train_loss_last'])}: the left column of the picture.")
        here_stop = (f"The validation score was checked every {C['every']['one_river']} updates on one river and every {C['every']['all']} on all "
                     f"twelve. Training ended after {C['patience']} checks without a better score, and the weights of the best check were kept: "
                     f"the dots in the picture.")
        many = ("With all twelve rivers (middle and right) the blue lines rise for hundreds of updates"
                + (f": {count(C['as_many'])} updates there are as many passes as {count(t0['best_update'])} on one river. "
                   if passes and C["as_many"] else ". ") if C["hundreds"] else "")
        curves_picture = _picture(
            "training_curves.png", "Learning curves in three cases.",
            "Grey, on top, is the training loss (lower is better). Blue, below, is the validation score (higher is better), and the dot is the "
            "state that was kept. "
            + (f"Look at the left column, one river: the blue line peaks at update {count(t0['best_update'])} and then falls while the grey "
               f"line keeps falling, which is overfitting. " if C["over"] else "")
            + many + "Each column has its own scales.")

    fifth = TU and TU["same_dropout"] and TU["dropout"] == 0.2                    # every network of the project sets one fifth to zero
    settings = _p(
        "Some numbers of a network are not learned. How many memory cells? How large a learning rate?",
        "Another is dropout, which the box on regularisation below explains.",
        ("In every update of the project&rsquo;s networks, one fifth of what the cells pass on is set to zero at random." if fifth else
         "In every update a random share of what the cells pass on is set to zero."),
        "When forecasts are scored, nothing is set to zero.",
        "The project tried a few values of each and compared them on the validation years, never on the test years.")
    here_hyper = ""
    if TU:
        K = TU["knobs"]
        flat = (f" Dropout of {', '.join(f'{v:g}' for v in K['dropout']['values'][:-1])} or {K['dropout']['values'][-1]:g} made no clear "
                f"difference." if TU["dropout_flat"] else "")
        cost = (f" On all twelve rivers one training takes minutes, so only {_word(LR['n_checked'])} settings were compared there."
                if LR and LR["minutes"] else "")
        here_hyper = (f"{_word(len(K)).capitalize()} settings were tried on one river, each changed alone: the number of memory cells "
                      f"({min(K['hidden']['values'])} to {max(K['hidden']['values'])}), the learning rate, the days of history, and dropout."
                      f"{flat} Every value tried costs one full training: the {TU['n_runs']} trainings on that river took about "
                      f"{n2(TU['minutes'], 0)} minutes, and the file <code>results/hparam_log.csv</code> lists them all.{cost}")

    here_under = ((f"The calendar-day average of step 5 is such a model: it ignores the weather, and its NSE is {n2(SIMPLE)}. " if SIMPLE is not None else "")
                  + ("The left column of the picture above shows the other end: a network that fits its training years ever better and new "
                     "years worse. " if C and C["over"] else "")
                  + "An AI course calls the error of a model that is too simple bias, and the error of one that is too flexible variance.")
    here_regular = ""
    if TU:
        K = TU["knobs"]
        small = (f" {ONE.capitalize()} is also small: the search kept the smallest size that scored about as well as the best, "
                 f"{K['hidden']['chosen']} memory cells, not {K['hidden']['best']}." if TU["small"] else "")
        both = "early stopping and dropout" if TU["same_dropout"] and TU["dropout"] > 0 else "early stopping"
        here_regular = (f"{ONE.capitalize()} and {ALL} both use {both}." + small
                        + " The word ridge in ridge regression (step 5) names one more guard: its weights are kept small. The strongest help "
                          "is not a guard at all: more data (step 8).")

    random_start = _p(
        "A training has random parts.",
        "So the same training, run twice, gives two slightly different networks.",
        "That is why each main network of the project was trained three times." if TU and len(TU["seeds"]) == 3 else "")
    here_seed = ""
    if TU:
        here_seed = (f"{_word(len(TU['seeds'])).capitalize()} seeds of one setting scored {n2(min(TU['seeds']), 3)} to {n2(max(TU['seeds']), 3)} on "
                     f"one river&rsquo;s validation years, so a gap below about {n2(TU['noise'], 3)} between two settings counted as noise. "
                     f"The forecast that the project reports is the day-by-day average of three trainings.")

    return "\n".join(x for x in (
        _opening(7, "A new network knows nothing: its weights are random numbers. This step shows how training turns them into useful ones, "
                    "and how to tell when to stop.", the_loop),
        concept("training", here=here_training),
        concept("loss", here=here_loss),
        downhill,
        concept("gradient", here=here_gradient),
        concept("learning_rate", here=here_rate),
        gradient_picture,
        concept("update", here=here_update),
        watch,
        concept("overfitting", here=here_over),
        concept("early_stopping", here=here_stop),
        curves_picture,
        settings,
        concept("hyperparameter", here=here_hyper),
        concept("underfitting", here=here_under),
        concept("regularisation", here=here_regular),
        random_start,
        concept("seed", here=here_seed),
        _closing("Training lowers the loss on the training years, one small step at a time. The validation years say when to stop and which "
                 "settings to keep. The test years are not touched until everything is chosen.", "section 3")) if x)


# ---- step 8: one network for all twelve rivers ------------------------------------------------------------------------------
def _step8(R, T, Z):
    def more_rivers():                                                            # the river of the one-river experiment, in both networks
        S, A = R["lstm_single"], R["lstm_all"]
        g, i = S["basin"]["gauge_id"], LEADS.index(FIRST)
        one_seeds = [float(v) for v in S["finals"]["wq"]["test_nse_by_seed"][lk(FIRST)]]
        all_seeds = [p["test_nse_by_basin"][g][i] for p in A["sets"]["wq"]["per_seed"].values()]
        return dict(river=_river(S), one=S["finals"]["wq"]["seed_average"]["test"][lk(FIRST)]["nse"],
                    all=A["sets"]["wq"]["test"]["basins"][g][lk(FIRST)]["nse"], helped=min(all_seeds) > max(one_seeds))

    def basics():
        A = R["lstm_all"]
        years = int(A["split"]["train"][1][:4]) - int(A["split"]["train"][0][:4])
        names = [STATIC_WORD[k] for k in SHOWN_PROPERTIES if k in A["static_inputs"] and k in STATIC_WORD]
        return dict(n=len(A["basins"]), years=years, n_props=len(A["static_inputs"]), names=names)

    B = _read("the all-river network (results/lstm_all.json)", basics)
    F = _read("the one-river and the all-river network on one river (results/lstm_single.json, results/lstm_all.json)", more_rivers)

    why = _p(
        "Step 7 showed the weak point of a network for one river.",
        (f"Its {B['years']} years of data are few, and going through them again and again teaches the network those years by heart." if B else
         "Its data are few, and going through them again and again teaches the network those years by heart."),
        "Twelve rivers give twelve times as many examples." if B and B["n"] == 12 else "")
    common = _p(                                                                  # a paragraph of its own, so that the heading block stays small
        "The rivers also have much in common, so what the network learns on one can help on the others.",
        f"Training one network on many rivers is the usual way today {cite('kratzert2019')}.",
        ("The one river that was trained both ways shows the gain." if F and F["helped"] else ""),
        (f"On the {F['river']}, {ALL} scored an NSE of {n2(F['all'])} one day ahead with the river gauge; {ONE} scored {n2(F['one'])}."
         if F and F["helped"] else ""))
    apart = _p(
        "But the twelve rivers are not alike.",
        (f"So the network is also given {_word(B['n_props'])} fixed properties of each basin, such as its "
         + ", ".join(B["names"][:-1]) + f" and {B['names'][-1]}." if B and len(B["names"]) > 1 else
         "So the network is also given fixed properties of each basin."),
        "They are the same on every day and tell the network what kind of river it is reading.",
        "None of them is worked out from the river&rsquo;s flow.")
    four = (keep(_p("Now comes the part of the design that matters most later.",                 # the list stays with the sentence that opens it
                    "The same network, with the same settings, was trained on four sets of inputs."),
                 '<ol style="margin-top:-2pt"><li>Past weather only.</li><li>Past weather and the river gauge.</li>'
                 "<li>Past weather and the weather of the days being forecast.</li><li>All three together.</li></ol>")
            + _p("Only the first two can be had in practice.",
                 "The other two cannot: nobody knows the weather of the coming days exactly.",
                 "They are there to measure what that knowledge would be worth."))
    here_change = ("The four versions of the network differ in their inputs and in nothing else. So the gap between two of their scores is "
                   "what one kind of information is worth: the river gauge, or the weather of the coming days.")
    here_bound = ("The perfect weather forecast is the weather that was really recorded on the days being forecast: an upper bound that cannot "
                  "be had in practice. A real forecast has errors, so a real system would score lower.")
    svg = _timeline()                                                             # the drawing of report section 1, at the width of its lettering
    drawing = '<div class="ff">' + small_figure(svg, _caption(
        "What a model may read, and what it must forecast.",
        "Time runs from left to right. Day t is the day on which the forecast is made; t+1 is the day after it, and t&minus;1 the day before. "
        "The dashed line marks the moment of the forecast: the end of day t. The two bars on the left are what a model may read: the past "
        "weather and, in some versions, the river gauge. The dashed bar on the right is the weather of the three days being forecast, which "
        "only the last two versions read; the boxes below are the flows to forecast."), drawing_width(svg)) + "</div>"
    purpose = _p(
        "Why go to this trouble?",
        "Suppose the forecasts could be better.",
        "Would the way be a cleverer network, or more information?",
        "Four versions of one network, different only in their inputs, can tell the two apart.",
        "Steps 9 to 11 use them for exactly that.")

    return "\n".join(x for x in (
        _opening(8, "One network was trained on all twelve rivers at once, and was then given four different sets of inputs: the design that "
                    "the later steps rest on.", why),
        common, apart, four, drawing,
        # the last blocks of the step are one block: step 9 starts on a new page, and a page that holds only the last lines of step 8
        # would be nearly empty. If the step runs over, the two boxes go to the next page together with its end.
        keep(concept("one_change", here=here_change),
             concept("upper_bound", here=here_bound),
             purpose,
             _closing("One network learned from all twelve rivers. Its four versions differ only in their inputs, so their scores show what "
                      "each kind of information is worth. The perfect weather forecast is an upper bound that cannot be had in practice.",
                      "section 4.1"))) if x)


# ---- step 9: did it beat the simple models? ------------------------------------------------------------------------------
def _step9(R, T, Z):
    def comparison():                                                             # the all-river LSTM against the best simple model (report section 4.2)
        A = R["lstm_all"]
        C, n = A["comparison"], len(A["basins"])
        f = facts(R, T)                                                           # the readings and milestone 2, as the report states them
        cmp_ = lambda inp, k: C[inp]["baselines"][C[inp]["best_baseline_by_validation"][lk(k)]][lk(k)]
        spread = lambda c: c["lstm_seed_median_max"] - c["lstm_seed_median_min"]
        return dict(n=n, cls=f["cls"], m2=f["m2"], n_seeds=f["n_seeds"],
                    lstm=[cmp_("wq", k)["lstm_nse_median"] for k in LEADS], simple=[cmp_("wq", k)["baseline_nse_median"] for k in LEADS],
                    higher={inp: [cmp_(inp, k)["n_basins_lstm_higher"] for k in LEADS] for inp in ("w", "wq")},
                    thin=all(cmp_("w", k)["difference_of_medians"] < spread(cmp_("w", k)) for k in LEADS))

    def one_river():                                                              # the one-river LSTM (report section 3)
        S = R["lstm_single"]
        inputs = ("w", "wq")                                                      # the two sets of inputs that can be had in practice
        calls = [S["comparison"][inp]["verdict"][lk(k)]["nse_call"] for inp in inputs for k in LEADS]
        on = S["comparison"]["wq"]["models"]["lstm1_wq"][lk(FIRST)]["onset"]
        peak = max(S["variance_concentration"]["test"]["days"], key=lambda d: d["flow_mm"])
        day = datetime.date.fromisoformat(peak["day"])
        return dict(river=_river(S), n=len(calls), ahead=sum(c.startswith("LSTM ahead") for c in calls), hits=on["hits"], onsets=on["hits"] + on["misses"],
                    peak=f"{day.day} {day:%B %Y}", n_inputs=len(inputs))

    def unseen():                                                                 # a river left out of training (report section 4.4)
        f = lobo_facts(R)
        if not f:
            raise KeyError("results/lstm_lobo.json")
        rows = [r for r in R["lstm_lobo"]["comparison"] if r["lead"] == FIRST]
        fails = [r["basin"].split(",")[0] for r in rows if r["held_out_with_properties"] <= 0]
        off = [r["basin"].split(",")[0] for r in rows if min(r["held_out_with_properties"], r["held_out_without_properties"]) < -0.5]
        plain = R["lstm_lobo"]["summary"][lk(FIRST)]["loss_from_not_seeing_the_basin"]["without_properties"]["median_nse_loss"]
        return dict(f, fails=fails, off=off, worse=f["loss"] > f["seed_range"], loss_plain=plain,
                    no_help=f["properties"] != "help" and 0 <= plain < f["loss"])      # without the basin properties the river loses less

    X = _read("the LSTM against the simple models (results/lstm_all.json)", comparison)
    O = _read("the one-river LSTM (results/lstm_single.json)", one_river)
    U = _read("the unseen-river test (results/lstm_lobo.json)", unseen)

    gauge = weather = all_picture = milestone = ""
    tie = bool(X) and all(v == "tie" for v in X["cls"]["wq"].values())
    if X:
        n, hi = X["n"], X["higher"]
        scores = (f"The LSTM reached a median NSE of {_three(X['lstm'])}, one, two and three days ahead. The best simple model reached "
                  f"{_three(X['simple'])}.")
        clear = ("A difference counts as clear only if the LSTM is ahead in all three trainings, in most rivers, and by more than chance "
                 "(a statistical test)." if X["n_seeds"] == 3 else "")
        if tie:
            gauge = _p("With the river gauge, the answer is: no clear difference.", scores, clear,
                       (f"At none of the {_word(len(LEADS))} lead times are all three conditions met: one day ahead, for example, the LSTM is "
                        f"ahead in only {hi['wq'][0]} of the {n} rivers." if clear and hi["wq"][0] <= n / 2 else ""))
        else:
            gauge = _p(f"With the river gauge, {reading_words(X['cls']['wq'])}.", scores, clear)
        if all(v == "lstm_avg" for v in X["cls"]["w"].values()):
            weather = _p("With past weather only, only the average of the LSTM&rsquo;s three trainings is ahead.",
                         f"It is ahead of the best simple model in {_three(hi['w'], str)} of the {n} rivers.",
                         "But the gap is smaller than the spread of its three trainings." if X["thin"] else "",
                         "So a single training is not clearly ahead.")
        else:
            weather = _p(f"With past weather only, {reading_words(X['cls']['w'])}.")
        all_picture = _picture(
            "lstm_all.png", "The LSTM against the best simple model, for each of the four sets of inputs.",
            "Each panel is one set of inputs, and higher is better. Blue is the LSTM; the grey line with the orange triangle (ridge regression) "
            "or the green square (random forest) is the best simple model. Small dots are single rivers, and the numbers at the foot of each "
            "panel say in how many rivers the LSTM scores higher. "
            + ("Look first at the bottom left panel, with the river gauge: the two lines lie on each other. " if tie else
               "Look first at the bottom left panel, with the river gauge, and compare the two lines. ")
            + "Then look right: with the perfect weather forecast both lines stand far above the dashed line, which repeats the LSTM with past "
              "weather only.")
        if X["m2"].startswith("not met with the river gauge"):
            milestone = _p("So milestone 2 of the plan is <b>not met with the river gauge</b>.",
                           "A negative result is a result: it says that a more complex model did not help here.",
                           "It also raises the question that steps 10 and 11 answer: why not?")
        else:
            milestone = _p(f"Milestone 2 of the plan is <b>{X['m2']}</b>.")

    alone = flood_picture = ""
    if O:
        silent = O["hits"] <= 0.1 * O["onsets"]
        which = (f" ({_word(O['n_inputs'])} sets of inputs, {_word(len(LEADS))} lead times)"
                 if O["n_inputs"] and O["n_inputs"] * len(LEADS) == O["n"] else "")
        alone = _p((f"{ONE.capitalize()} was ahead of the best simple model in none of the {_word(O['n'])} comparisons{which}."
                    if O["ahead"] == 0 else
                    f"{ONE.capitalize()} was ahead of the best simple model in {O['ahead']} of the {_word(O['n'])} comparisons{which}."),
                   (f"One day ahead with the river gauge, it warned of {O['hits']} of the {O['onsets']} onsets of its river." if silent else ""),
                   "The second picture shows the largest flood of that river in the test years: no forecast rises in time." if silent else "")
        flood_picture = _picture(
            "lstm_single_hydrograph.png", f"The largest flood of the test years on the {O['river']}, forecast one day ahead.",
            "Top: the rain of each day; bottom: the measured flow (black), the forecasts (other lines) and the high-flow level (red dashed "
            f"line). Look at {O['peak']}, the day of the peak"
            + (f": no forecast rises with the black line. {QUOTED_CAP} (grey) shows the peak one day late, and the two LSTM lines rise late and "
               f"far too little." if silent else f", and compare each forecast with the black line. {QUOTED_CAP} (grey) shows the peak one day late.")
            + " The rain bars carry the date that the rain table gives them. Step 11 looks at that date: part of the rain under it seems to "
              "have fallen before the forecast for that day was made.")

    never = here_general = unseen_picture = ""
    RV = (step16_available(R) or {}).get("RV")                     # the experiment of 8 October 2026 that repeats the test with 50 training rivers (step 16), or None
    if U:
        never = _p("One harder test remains.",
                   "Most rivers have no gauge.",
                   "Could the network forecast one?",
                   f"Each river in turn was left out of training, and a network trained on the other {U['n'] - 1} had to forecast it.",
                   "The network read the river&rsquo;s weather and basin properties, but no river gauge.",
                   (f"A river the network has never seen is forecast worse: the typical river loses {n2(U['loss'])} NSE one day ahead."
                    if U["worse"] else
                    f"For a river the network has never seen there is no clear loss: the typical river changes by {n2(abs(U['loss']))} NSE one day ahead."),
                   (f"Without the basin properties the loss is smaller, {n2(U['loss_plain'])} NSE." if U["worse"] and U["no_help"] else ""),
                   (f"With only {U['n'] - 1} rivers to learn from, the properties did not help on a river never seen."
                    if U["worse"] and U["no_help"] else ""),
                   f"Published work trains such networks on hundreds of rivers; with {U['n'] - 1}, this test says nothing about a large training set"
                   + (f" (step 16 repeats it with {RV['n_train']['unseen50_w_static']})." if RV else "."))
        here_general = ("The test years check the first kind, later years of rivers the network knows; leaving a river out of training checks "
                        "the second."
                        + (f" With only {U['n'] - 1} rivers to learn from, the score fell in {U['n_lower']} of the {U['n']} rivers"
                           + (f", and {_word(len(U['fails']))} river failed outright." if len(U["fails"]) == 1 else ".") if U["worse"] else ""))
        unseen_picture = _picture(
            "lobo.png", "A river the network has never seen, forecast one day ahead.",
            "Each row is one river; further right is better. The filled dot is the score with the river in the training, the diamond the score "
            "with the river left out. The open circle is the same test without the basin properties. "
            + (f"In {U['n_lower']} of the {U['n']} rivers the diamond lies left of the dot: the forecast got worse"
               + (f", and {U['off'][0]} falls off the scale." if len(U["off"]) == 1 else ".") if U["worse"] else ""))

    if tie and U and U["worse"]:
        last = ("With the river gauge, the LSTM and the best simple model show no clear difference, so milestone 2 is not met with the river "
                "gauge. A river the network has never seen is forecast worse. A negative result is a result, and it is reported as it is.")
    elif tie:
        last = ("With the river gauge, the LSTM and the best simple model show no clear difference, so milestone 2 is not met with the river "
                "gauge. A negative result is a result, and it is reported as it is.")
    else:
        last = "A model is judged against the simple models on the test years, and a difference counts only when it is clear."

    return "\n".join(x for x in (
        _opening(9, "The plan of the project (PLAN.md) set as its second milestone that the LSTM beats the simple models. This step gives the "
                    "answer of the test years, and one harder test.", gauge, new_page=True),
        all_picture, weather, alone, flood_picture, milestone, never,
        concept("generalisation", here=here_general),
        unseen_picture,
        _closing(last, "sections 4.2, 3 and 4.4")) if x)


def html(R, T):
    Z = _read("the size of the two networks (results/lstm_single.json, results/lstm_all.json)", lambda: _sizes(R))
    n = _read("the number of rivers (results/lstm_all.json)", lambda: len(R["lstm_all"]["basins"]))
    if n is not None and n != N_RIVERS:
        _note(f"the text calls the main network '{ALL}', but results/lstm_all.json holds {n} rivers")
    return STYLE + '<div class="g2">\n' + "\n".join(make(R, T, Z) for make in (_step6, _step7, _step8, _step9)) + "\n</div>"
