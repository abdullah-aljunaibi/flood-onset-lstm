"""Report section 3 - an LSTM trained on one river (French Broad River). Reads results/lstm_single.json and
results/hparam_log.csv only (both written by 03_lstm.py). Nothing is computed here beyond differences, means of the three
lead times, counts, smallest/largest and formatting; every sentence whose wording depends on an outcome is built from the
stored numbers, so a rerun of 03_lstm.py cannot leave a false sentence behind.

Figures (all drawn for a width of 170 mm, so each is shown at about that width or left out, never shrunk): the hydrograph of
the largest flood of the test years is shown - the one picture of a forecast against the observed river in this section; the
tuning figure is left out and its numbers stand in the compact tuning table (Table 3.1); lstm_memory.png is left out - its
numbers are in the text. SHOW_TUNING_FIG / SHOW_MEMORY_FIG / SHOW_HYDRO switch them; the figure and table numbers follow.

The network is specified here, once (decision of the project lead): its layers, its output, its inputs' scaling, the error it
reduces, the optimiser, the size of an update, gradient clipping and early stopping are read off code/flood.py (LSTMNet, Cfg,
fit, _design, _to_target_day) - fixed properties of the code, written as text and checked against it; every number among them
is read from the results file or from flood.Cfg. Section 4.1 adds how twelve rivers share one loss.
The September 2004 floods are named from results/floods_2004.json (check_2004_floods.py; storm tracks) when that file exists.

'Inside the LSTM' says what a memory cell does (gates, state, why the error signal survives many days). These are fixed properties of
torch.nn.LSTM as flood.LSTMNet uses it and are written as text; the start value of the forget-gate bias is read off a network built for
the purpose (forget_gate_start: nothing is trained), so the sentence cannot drift from the code.
The learning curves (Figure 3.2, figures/training_curves.png) and the paragraph 'What the training looks like' are read from
results/training_curves.json (12_training_curves.py) inside a guarded block: without the file or the figure they are left out and one
line goes to stderr. curves_figure(R) gives the figure's number to section 4.1, which points to it. The left column of that figure is the
BASE setting of the tuning (one training), not the chosen model of this section; the caption and the text say so.
"""
import datetime
import math
import sys

from common import FIG
from flood import FORCING, LAND_SHORT, Cfg, LSTMNet
from report_common import LEAD_WORD, cite, count, fig, n2, pct, table
from report_sections.s0_summary import LAYOUT, floated

LEADS = sorted(LEAD_WORD)
SHOW_TUNING_FIG, SHOW_MEMORY_FIG, SHOW_HYDRO = False, False, True    # which of the three figures of 03_lstm.py are shown (the page limit decides)
FIG_TUNING, FIG_MEMORY, FIG_HYDRO, FIG_CURVES = 84, 84, 84, 84      # % of the text width (184 mm); the figures are drawn for 170 mm (92%) with 7.5 to 9 pt lettering
MEM_TOL = 0.02                                  # memory test: a change in NSE below this is called 'no gain'
TIMING_REF, ALL_REF, BASE_REF = "section 4.5", "section 4.2", "section 2"
QUOTED, QUOTED_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
FAMILY = {"base_persistence": QUOTED, "base_linreg": "ridge regression", "base_rf": "random forest"}
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
KNOB = {"hidden": ("the model size", "the number of memory cells", "Model size, cells"),              # name in a sentence, what it is, table label
        "lr": ("the learning rate", "the size of each correction to the weights", "Learning rate"),
        "dropout": ("dropout", "the share of the cells&rsquo; outputs set to zero, a guard against memorising", "Dropout"),
        "warmup": ("the history", "the days read before the first scored forecast", "History, days")}
STATUS = {"HU": "hurricanes", "TS": "tropical storms", "TD": "tropical depressions", "EX": "former tropical cyclones", "LO": "remnant lows"}   # HURDAT2 status codes
TIGHT = ("<style>.s3t th,.s3t td{padding:1.6pt 4pt;line-height:1.28} .s3t td{white-space:nowrap} "
         ".s3l{margin-bottom:4pt} .s3l li{margin-bottom:1pt} sub,sup{line-height:0}</style>")        # layout only (sub, sup: a formula keeps the line spacing)


def _word(n):
    return NUMBER.get(n, str(n))


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _leads(ks):
    """[1] -> '1 day ahead', [2, 3] -> '2 and 3 days ahead'."""
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _cap(s):
    return s[0].upper() + s[1:]


def _mean(v):
    return sum(v) / len(v)


def _family(pred_set):
    return next(v for stem, v in FAMILY.items() if pred_set.startswith(stem))


def forget_gate_start(cells=4):
    """The value flood.LSTMNet gives the bias of the forget gates before training, read off a network built for the purpose (nothing is
    trained), or None when it cannot be read. torch.nn.LSTM keeps its four blocks in the order input gate, forget gate, new value, output
    gate, and LSTMNet fills the second block of bias_hh_l0; the other bias vector keeps its small random start, so a gate starts near this value."""
    try:
        b = LSTMNet(len(FORCING), Cfg(hidden=cells)).lstm.bias_hh_l0.detach()[cells:2 * cells]
        v = float(b[0])
        if not bool((b == b[0]).all()) or v <= 0:
            raise ValueError("the forget-gate biases do not start at one positive value")
        return v
    except Exception as e:                                                          # the network class changed: no sentence rather than a guessed one
        print(f"s3_single: the start value of the forget-gate bias was not read from flood.LSTMNet ({type(e).__name__}: {e}); its sentence is left out",
              file=sys.stderr)
        return None


def curves_figure(R):
    """'Figure 3.2' when the learning curves are shown at the end of this section (see learning_curves), otherwise None. Section 4.1 points
    to the figure through this function."""
    return learning_curves(R, quiet=True)[0]


def learning_curves(R, quiet=False):
    """(figure number, paragraph 'What the training looks like', figure with caption) from results/training_curves.json (12_training_curves.py)
    and figures/training_curves.png, every number by key; (None, '', '') and - unless quiet - one line on stderr when the file of a full run, the
    figure or a key is missing. The number follows the figures of this section that are switched on. Wording that depends on an outcome is
    chosen from the stored numbers. The left column of the figure is the BASE setting of the tuning (one training), not the chosen model."""
    say = (lambda msg: None) if quiet else (lambda msg: print(f"s3_single: {msg}; the learning curves are left out", file=sys.stderr))
    TC = R.get("training_curves")
    if not isinstance(TC, dict) or TC.get("quick_mode") or not (FIG / "training_curves.png").exists():
        say("results/training_curves.json (of a full run) or figures/training_curves.png was not found")
        return None, "", ""
    f_no = f"Figure 3.{sum((SHOW_TUNING_FIG, SHOW_MEMORY_FIG, SHOW_HYDRO)) + 1}"
    t_tune = "Figure 3.1" if SHOW_TUNING_FIG else "Table 3.1"                       # where the tuning is shown (as in html)
    try:
        base, chosen = R["lstm_single"]["tuning"]["base"], R["lstm_single"]["tuning"]["chosen"]
        one, mid, right = (TC["panels"][k] for k in ("one_river", "all_wq", "all_wqf"))
        t0, st = one["per_training"][0], TC["settings"]                            # st: the settings the three columns share when `same` is true
        kept = lambda p: (min(t["best_update"] for t in p["per_training"]), max(t["best_update"] for t in p["per_training"]))
        upd = lambda r: f"update {count(r[0])}" if r[0] == r[1] else f"updates {count(r[0])} to {count(r[1])}"
        n = mid.get("n_rivers") or len(R["lstm_all"]["basins"])
        same = TC.get("same_settings_one_river_and_all") is True
        over = t0["last_update"] > t0["best_update"] and t0["val_nse_at_last"] < t0["val_nse_at_best"] and t0["train_loss_last"] < t0["train_loss_at_best"]
        not_chosen = chosen != base                                                 # the left column is the base setting, not the model of this section
        left = (f"On one river (left: the base setting of {t_tune}, {base['hidden']} cells, one training"
                + (f"; not the chosen {chosen['hidden']}-cell model" if not_chosen and chosen["hidden"] != base["hidden"] else
                   "; not the chosen model" if not_chosen else "") + ") ")
        if over:
            left += (f"the validation score peaks at update {count(t0['best_update'])} (NSE {n2(t0['val_nse_at_best'])}) and has fallen to "
                     f"{n2(t0['val_nse_at_last'])} by update {count(t0['last_update'])}, while the training loss falls further ({n2(t0['train_loss_at_best'])} to "
                     f"{n2(t0['train_loss_last'])}): the network fits its training years better and the validation years worse (overfitting), and early "
                     f"stopping keeps the state at the peak.")
        else:
            left += f"the state was kept at update {count(t0['best_update'])} of {count(t0['last_update'])}."
        all_ = (f"{'With the same settings on' if same else 'On'} all {n} rivers (middle; section 4) the kept states are at {upd(kept(mid))}; given the "
                f"weather of the days being forecast, the upper bound, at {upd(kept(right))} (right).")
        # updates are not passes: an update scores the same number of days in every column, a small share of twelve rivers' training years
        passes = ""
        p1, pm = t0.get("passes_at_best"), (mid.get("passes_at_best_min"), mid.get("passes_at_best_max"))
        if same and p1 and all(pm) and t0["best_update"] > 0 and kept(mid)[0] > t0["best_update"]:
            ratio = (kept(mid)[0] / t0["best_update"], kept(mid)[1] / t0["best_update"])
            times = f"{ratio[0]:.0f}" if f"{ratio[0]:.0f}" == f"{ratio[1]:.0f}" else f"{ratio[0]:.0f} to {ratio[1]:.0f}"
            pm_txt = f"about {pm[0]:.0f}" if f"{pm[0]:.0f}" == f"{pm[1]:.0f}" else f"about {pm[0]:.0f} to {pm[1]:.0f}"
            late = "about as many" if pm[0] <= 1.25 * p1 and pm[1] >= 0.8 * p1 else "more" if pm[0] > p1 else "fewer"
            pr = (right.get("passes_at_best_min"), right.get("passes_at_best_max"))             # the upper-bound column was kept later, also in passes
            more_r = f"; with the upper bound more (about {pr[0]:.0f} to {pr[1]:.0f})" if all(pr) and pr[0] > 1.25 * p1 else ""
            passes = (f" With the river gauge (middle) those are more updates than on one river, but {late} passes through the training years "
                      f"({pm_txt}, against about {p1:.0f}){more_r}: an update scores the same number of days in every column.")
        text = f"<p><b>What the training looks like</b> ({f_no}). {left} {all_}{passes}</p>"
        hundreds = min(kept(mid)[0], kept(right)[0]) >= 200
        caption = (f"<b>{f_no}. Learning curves of three training set-ups.</b> Top: training loss; bottom: NSE in the validation years. Left column: the "
                   f"base setting of the tuning ({t_tune}) and one training"
                   + (f", not the chosen model of this section ({chosen['hidden']} cells)" if not_chosen else "") + ". What to see: "
                   + (f"on the left the blue line falls after update {count(t0['best_update'])} while the grey line falls further; " if over else "")
                   + ("in the middle and on the right the blue lines rise for hundreds of updates." if hundreds else
                      f"in the middle and on the right the states were kept at {upd(kept(mid))} and {upd(kept(right))}."))
        return f_no, text, fig("training_curves.png", caption, FIG_CURVES)
    except (KeyError, TypeError, IndexError, ValueError) as e:
        say(f"results/training_curves.json is not in the expected form ({type(e).__name__}: {e})")
        return None, "", ""


def html(R, T):
    S = R["lstm_single"]
    LOG = T["hparam_log"]
    B, TU, C, FIN, MEM, VC = S["basin"], S["tuning"], S["comparison"], S["finals"], S["memory"], S["variance_concentration"]
    split, base, chosen, tol = S["split"], TU["base"], TU["chosen"], TU["noise_seed_spread"]
    lk = lambda k: f"l{k}"
    first, last = LEADS[0], LEADS[-1]
    years = lambda part: f"{split[part][0][:4]}&ndash;{split[part][1][:4]}"
    n_train_years = int(split["train"][1][:4]) - int(split["train"][0][:4])
    named = " AT " in B["name"]
    river = B["name"].split(" AT ")[0].title() if named else B["short"]
    at = f" at {B['name'].split(' AT ')[1].split(',')[0].title()},{B['name'].split(',')[-1]}" if named and "," in B["name"] else ""
    land = LAND_SHORT.get(B["landscape"], B["landscape"]).lower()
    level = B["high_flow_level_mm_day"]

    # ---- the question, why an LSTM, how it is trained ----------------------------------------------------------------
    batch = _and(sorted({int(b) for b in LOG.batch}))
    budget, final_budget = TU["budget"], TU["final_budget"]
    fb = forget_gate_start()                                                        # read off flood.LSTMNet; None: the sentence is left out
    open_gates = (f" The forget gates start open (before training the code sets their bias to {fb:g}, which keeps about "
                  f"{pct(1 / (1 + math.exp(-fb)))} of a state from day to day), so an untrained network starts by remembering.") if fb else ""
    intro = f"""
<p>Can an LSTM trained on one river alone forecast that river better than the simple models of {BASE_REF}? The test river is the {river}{at}
({count(B['area_km2'])} km&sup2;, {land}); section 4 then trains one model on all twelve rivers.</p>
<p><b>Why an LSTM.</b> A river remembers. Today&rsquo;s flow depends on the rain of the past days to months, which the basin stores as wet soil and
groundwater. The simple models are handed that store ready-made, as rain sums over fixed numbers of past days. A long short-term memory network
(LSTM) {cite('hochreiter1997')} reads the days one after another and carries a set of numbers, its memory, from each day to the next; training
decides what it keeps and forgets. So it can learn the store from the data, which is how LSTMs are used for river flow {cite('kratzert2018')}.</p>
<p><b>Inside the LSTM.</b> Each memory cell holds one number, its state c. On every day three <i>gates</i>, each a number between 0 and 1 computed
from the day&rsquo;s inputs and the cells&rsquo; outputs of the day before, act on it: the forget gate f says how much of yesterday&rsquo;s state is
kept, the input gate i how much of today&rsquo;s new value g is written, the output gate o how much of the state is passed on as the cell&rsquo;s
output h. So c<sub>t</sub> = f<sub>t</sub>&thinsp;c<sub>t&minus;1</sub> + i<sub>t</sub>&thinsp;g<sub>t</sub> and h<sub>t</sub> =
o<sub>t</sub>&thinsp;tanh(c<sub>t</sub>). Read as hydrology, c is a store of water: f lets it drain, i lets today&rsquo;s rain in. Because the
state is carried from day to day by one multiplication and one addition, the error signal of training can travel back over hundreds of days; in a
plain recurrent network it fades much sooner {cite('hochreiter1997')}.{open_gates}</p>
<p><b>The network</b> (the same LSTM in every later section, apart from its number of memory cells and its inputs). One LSTM layer; after every
day one linear layer turns its memory cells into {_word(len(LEADS))} numbers, the flow {LEADS[0]}, {LEADS[1]} and {LEAD_WORD[LEADS[2]]} (a forecast below zero is set to zero). The
inputs are {_word(len(FORCING))} daily weather variables and, where stated, the river gauge. Each input is put on one scale with its mean and spread
in the training years; the river gauge enters as log(1 + flow).</p>
<p><b>How it is trained.</b> One training step (an <i>update</i>) takes {batch} random stretches of the training years. In each, the model reads some
days of history to warm up its memory, then forecasts each of the next {chosen['block']} days {LEADS[0]}, {LEADS[1]} and {LEAD_WORD[LEADS[2]]}; the
update changes the weights a little to reduce the squared error of those forecasts (the <i>loss</i>). The optimiser is Adam with a fixed learning
rate; the gradients are clipped at a norm of {Cfg().clip:g}; <i>dropout</i> sets a share of the memory cells&rsquo; outputs to zero at random during
training. Every
{budget['eval_every']} updates the model is scored on the validation years (mean NSE of the {_word(len(LEADS))} lead times); the best state is kept,
and training stops after {budget['patience']} checks without improvement (<i>early stopping</i>).</p>"""

    # ---- tuning -----------------------------------------------------------------------------------------------------
    PK = TU["per_knob"]
    fmt = lambda v: f"{v:g}"
    shown = [name for name, on_ in (("tuning", SHOW_TUNING_FIG), ("memory", SHOW_MEMORY_FIG), ("hydro", SHOW_HYDRO)) if on_]
    fno = {name: f"Figure 3.{i}" for i, name in enumerate(shown, 1)}              # figure numbers in the order shown
    T_CMP = "Table 3.1" if SHOW_TUNING_FIG else "Table 3.2"                       # with the tuning figure there is no tuning table

    def why(knob):
        """Why this value was chosen, from the stored reason and scores."""
        p = PK[knob]
        vals, sc, best, near = p["values"], p["val_nse_mean"], p["best"], p["within_noise_of_best"]
        up, down = all(a < b for a, b in zip(sc, sc[1:])), all(a > b for a, b in zip(sc, sc[1:]))
        if p["reason"].startswith("best"):
            return "the best score" + ("; it rises with every step" if up else "; it falls with every step" if down else "")
        if p["reason"].startswith("smallest"):
            return f"the smallest within the seed spread of the best, {fmt(best)}"
        return "the base value; all are within the seed spread" if len(near) == len(vals) else f"the base value; within the seed spread of the best, {fmt(best)}"

    def knob_row(knob):
        p = PK[knob]
        vals, sc, pick = p["values"], p["val_nse_mean"], p["chosen"]
        return [KNOB[knob][2], ", ".join(fmt(v) for v in vals), ", ".join(f"<b>{n2(s)}</b>" if v == pick else n2(s) for v, s in zip(vals, sc)),
                fmt(TU["base"][knob]), f"<b>{fmt(pick)}</b>", why(knob)]

    base_vals = TU["base_val_nse_by_seed"]
    all_scores = [v for p in PK.values() for v in p["val_nse_mean"]]
    check = TU["combination_check"]
    seeds_word = f"mean of {_word(len(base_vals))} seeds"
    if check["fell_back_to_base"]:
        combo = (f"Together the {_word(len(PK))} choices scored {n2(check['chosen_val_nse_seed_mean'])} on the validation years ({seeds_word}), below the "
                 f"base setting ({n2(check['base_val_nse_seed_mean'])}), so the base setting was used.")
    elif chosen == base:
        combo = "The chosen values are the base setting."
    else:
        gain = check["chosen_val_nse_seed_mean"] - check["base_val_nse_seed_mean"]
        combo = (f"Together the {_word(len(PK))} choices scored {n2(check['chosen_val_nse_seed_mean'])} on the validation years ({seeds_word}) against "
                 f"{n2(check['base_val_nse_seed_mean'])} for the base setting" + ("." if gain > tol else ": no clear difference."))
    n_runs = len(LOG)
    kinds = [(int(LOG.purpose.str.startswith(p).sum()), w) for p, w in (("tuning", "for tuning"), ("final", "final models"), ("diagnostic", "diagnostic runs"))]
    n_params = FIN["wq"]["seeds"][min(FIN["wq"]["seeds"], key=int)]["training"]["n_params"]
    stops = [s["training"]["best_update"] for inp in ("w", "wq") for s in FIN[inp]["seeds"].values()]
    edge_knobs = [KNOB[k][0] for k, p in PK.items() if chosen[k] in (min(p["values"]), max(p["values"]))]
    edge_cap = (f"{_cap(_word(len(edge_knobs)))} of the {_word(len(PK))} chosen values {'are' if len(edge_knobs) > 1 else 'is'} the smallest or largest "
                f"tried." if edge_knobs else "No chosen value is the smallest or largest tried.")
    tune_table = table(["Setting", "Values tried", "Validation NSE, same order", "Base", "Chosen", "Why this value"], [knob_row(k) for k in PK],
                       cls="num s3t", widths=["15%", "22%", "21%", "7%", "8%", "27%"],
                       caption=f"<b>Table 3.1. Tuning on the validation years.</b> Each row changes one setting and keeps the others at the base value. "
                               f"Bold: the chosen value and its score. What to see: no run scores above {n2(max(all_scores))}. {edge_cap}")
    fig_tuning = fig("lstm_single_tuning.png",
                     f"<b>{fno['tuning']}. Tuning on the validation years.</b> Each panel changes one setting and keeps the others at the base value. "
                     f"Grey band: the seed spread at the base setting; a step smaller than the band is noise. Circle: the chosen value. What to see: no "
                     f"run scores above {n2(max(all_scores))}.", FIG_TUNING) if SHOW_TUNING_FIG else ""
    unit = {"warmup": " days", "hidden": " cells"}
    brief = lambda k: ("the best score" if PK[k]["reason"].startswith("best") else "the smallest within the seed spread of the best"
                       if PK[k]["reason"].startswith("smallest") else "the base value kept")
    short_picks = "; ".join(f"{KNOB[k][0].replace('the ', '')} {fmt(PK[k]['chosen'])}{unit.get(k, '')} ({brief(k)})" for k in PK)
    tuned = (f"{floated(fig_tuning)}\n<p><b>Chosen:</b> {short_picks}. {edge_cap}" if SHOW_TUNING_FIG else f'<div class="blk">{tune_table}</div>\n<p>')
    tuning = f"""
<p><b>Tuning.</b> {_cap(_word(len(PK)))} settings were tuned: {_and(f'{KNOB[k][0]} ({KNOB[k][1]})' for k in PK)}. Each was changed alone around a
base setting, with past weather + river gauge as inputs, and judged on the validation years only ({fno["tuning"] if SHOW_TUNING_FIG else 'Table 3.1'}),
by the mean NSE of {LEADS[0]} to {LEAD_WORD[last]}. The yardstick is the <i>seed spread</i>: how much the score moves when the same model is trained
again from a different random start (a <i>seed</i>). {_cap(_word(len(base_vals)))} seeds of the base setting scored {n2(min(base_vals), 3)} to
{n2(max(base_vals), 3)}, so a difference below about {n2(tol, 3)} cannot be told from noise ({_word(len(base_vals))} runs give only a rough
yardstick).</p>
{tuned} {combo} The chosen LSTM has {count(n_params)} weights. The hyper-parameter log (<code>results/hparam_log.csv</code>) holds all {n_runs}
training runs ({_and(f'{n} {w}' for n, w in kinds if n)}; about {n2(S['runtime_s'] / 60, 0)} minutes on a laptop). The test years chose nothing.</p>"""

    # ---- memory test ------------------------------------------------------------------------------------------------
    ctx = MEM["contexts_days"]
    mem = lambda inp, c: MEM["test_nse_by_lead"][inp][str(c)][LEADS.index(first)]
    c0, c1, c2 = min(ctx), min(ctx, key=lambda c: abs(c - chosen["warmup"])), max(c for c in ctx if c <= 365)       # shortest; the training minimum; a year
    gain = {inp: mem(inp, c2) - mem(inp, c1) for inp in ("w", "wq")}
    three = lambda inp: _and(n2(mem(inp, c)) for c in (c0, c1, c2))
    if gain["w"] > MEM_TOL and gain["wq"] <= MEM_TOL:
        reading = (f"Weather older than {c1} days still helps the model without the gauge ({n2(gain['w'])} NSE) and not the one with it. A likely reading: "
                   f"the gauge already tells the model how wet the basin is.")
    elif gain["w"] > MEM_TOL:
        reading = f"Weather older than {c1} days still helps both models ({n2(gain['w'])} and {n2(gain['wq'])} NSE)."
    else:
        reading = f"Weather older than {c1} days adds less than {n2(MEM_TOL)} NSE to either model."
    memory = f"""
<p><b>How far back does it look?</b> The trained models were given only the last days before each forecast and scored on the test years,
{LEAD_WORD[first]}{f' ({fno["memory"]})' if SHOW_MEMORY_FIG else ''}. With {c0}, {c1} and {c2} days of history the NSE is {three('w')} with past weather
only, and {three('wq')} with the river gauge. {reading} The catch: these models were trained with at least {chosen['warmup']} days of history, so
the test shows how much of the skill needs old weather, not how a model trained on short histories would do.</p>"""
    fig_memory = fig("lstm_memory.png",
                     f"<b>{fno['memory']}. The memory test.</b> Test NSE {LEAD_WORD[first]} when the trained model may read only the given number of past "
                     f"days. Grey: histories shorter than the {chosen['warmup']} days the model had in training.", FIG_MEMORY) if SHOW_MEMORY_FIG else ""

    # ---- like-for-like comparison on the test years ------------------------------------------------------------------
    SETS = ("w", "wq")
    label = {"w": "Past weather", "wq": "Past weather + river gauge"}
    with_ = {"w": "with past weather only", "wq": "with the river gauge"}
    V = lambda inp, k: C[inp]["verdict"][lk(k)]
    score = lambda inp, name, k: C[inp]["models"][name][lk(k)]["nse"] if name in C[inp]["models"] else None
    simple = lambda inp: [n for n in (f"base_linreg_{inp}", f"base_rf_{inp}") if n in C[inp]["models"]]
    call = lambda inp, k: "tie" if V(inp, k)["nse_call"].startswith("too close") else "lstm" if V(inp, k)["nse_call"].startswith("LSTM ahead") else "simple"
    said = lambda inp, k: {"tie": "no clear difference", "lstm": "LSTM ahead", "simple": f"{_family(V(inp, k)['best_baseline'])} ahead"}[call(inp, k)]
    rows = [[label[inp] if k == first else "", LEAD_WORD[k], f"<b>{n2(V(inp, k)['lstm_nse'])}</b>", n2(V(inp, k)["seed_spread_used"]),
             n2(score(inp, f"base_linreg_{inp}", k)), n2(score(inp, f"base_rf_{inp}", k)),
             n2(score(inp, "base_persistence", k)), said(inp, k)] for inp in SETS for k in LEADS]
    n_seeds = V("wq", first)["n_seeds"]
    cmp_table = table(["LSTM inputs", "Forecast", "LSTM,<br>one river", "Seed<br>spread", "Ridge<br>regression", "Random<br>forest",
                       QUOTED_CAP.replace(" = ", "<br>= "), "Reading"], rows, cls="num s3t",
                      widths=["25%", "12.5%", "8.5%", "7.5%", "10%", "8.5%", "10.5%", "17.5%"],
                      caption=f"<b>{T_CMP}. The one-river LSTM against the simple models given the same inputs.</b> NSE in the test years ({years('test')}). "
                              f"LSTM: the day-by-day average forecast of {_word(n_seeds)} seeds. Seed spread: highest minus lowest NSE of the single "
                              f"seeds. {QUOTED_CAP} reads the gauge only. Reading: the LSTM against the best simple model of the row; a gap no larger "
                              f"than the seed spread is no clear difference (with one river the seed spread is the only yardstick; {ALL_REF} adds the "
                              f"count of basins and p). Differences quoted in the text are taken before rounding.")

    cells = [(inp, k) for inp in SETS for k in LEADS]
    ahead, behind, ties = ([c for c in cells if call(*c) == w] for w in ("lstm", "simple", "tie"))
    where = lambda cs: _and(f"{_leads([k for i, k in cs if i == inp])} {with_[inp]}" for inp in SETS if any(i == inp for i, k in cs))
    if not ahead:
        answer = f"<b>No.</b> In none of the {_word(len(cells))} rows of {T_CMP} is the one-river LSTM ahead of the best simple model."
    elif len(ahead) == len(cells):
        answer = f"<b>Yes.</b> In all {_word(len(cells))} rows of {T_CMP} the one-river LSTM is ahead of the best simple model by more than its seed spread."
    else:
        answer = f"<b>Only in part.</b> The one-river LSTM is ahead of the best simple model {where(ahead)}."
    detail = []
    if behind:
        lost = list(dict.fromkeys(_family(n) for inp, k in behind for n in simple(inp)
                                  if score(inp, n, k) - V(inp, k)["lstm_nse"] > V(inp, k)["seed_spread_used"]))
        gaps = [score(inp, n, k) - V(inp, k)["lstm_nse"] for inp, k in behind for n in simple(inp)
                if score(inp, n, k) - V(inp, k)["lstm_nse"] > V(inp, k)["seed_spread_used"]]
        spr = [V(inp, k)["seed_spread_used"] for inp, k in behind]
        detail.append(f"{_cap(where(behind))} it is clearly behind: {_and(lost)} score{'s' if len(lost) == 1 else ''} {n2(min(gaps))}"
                      + (f" to {n2(max(gaps))}" if n2(max(gaps)) != n2(min(gaps)) else "")
                      + f" NSE higher, more than the seed spread ({n2(min(spr))}" + (f" to {n2(max(spr))}" if n2(max(spr)) != n2(min(spr)) else "") + ").")
    if ties:
        gap = [abs(V(inp, k)["nse_margin"]) for inp, k in ties]
        lead_in = f"In each of the other {_word(len(ties))} rows" if behind or ahead else "In every row"
        detail.append(f"{lead_in} the gap to the best simple model ({n2(min(gap), 3)} to {n2(max(gap), 3)} NSE) is no larger than that row&rsquo;s seed "
                      f"spread: no clear difference.")
    pers = {k: score("wq", "base_persistence", k) for k in LEADS}
    if all(v is not None for v in pers.values()):
        p_state = {k: "lstm" if V("wq", k)["lstm_nse"] - pers[k] > V("wq", k)["seed_spread_used"] else
                   "pers" if pers[k] - V("wq", k)["lstm_nse"] > V("wq", k)["seed_spread_used"] else "tie" for k in LEADS}
        grp = {w: [k for k in LEADS if p_state[k] == w] for w in ("lstm", "tie", "pers")}
        bits = []
        if grp["lstm"]:
            bits.append(f"the LSTM with the river gauge does beat {QUOTED} {_leads(grp['lstm'])} ({_and(n2(V('wq', k)['lstm_nse']) for k in grp['lstm'])} "
                        f"against {_and(n2(pers[k]) for k in grp['lstm'])})")
        if grp["tie"]:
            bits.append(f"{_leads(grp['tie'])} the two are within the seed spread" if grp["lstm"] else
                        f"the LSTM with the river gauge and {QUOTED} are within the seed spread {_leads(grp['tie'])}")
        if grp["pers"]:
            bits.append(f"{_leads(grp['pers'])} {QUOTED} is ahead")
        detail.append(_cap("; ".join(bits)) + ".")

    # onsets: the warnings that matter
    on = lambda inp, name, k: C[inp]["models"][name][lk(k)]["onset"]
    n_on = on("wq", "lstm1_wq", first)["hits"] + on("wq", "lstm1_wq", first)["misses"]
    hits = {(inp, n): on(inp, n, first)["hits"] for inp in SETS for n in C[inp]["models"]}
    top_simple = max((key for key in hits if key[1].startswith("base_")), key=hits.get)
    lstm_hits = {inp: hits[inp, f"lstm1_{inp}"] for inp in SETS}
    most = max(hits.values())
    onsets = (f"{'No model gives the warnings that matter' if most <= 0.1 * n_on else 'The warnings that matter are rare'}: the onsets ({BASE_REF}), "
              f"high-flow days forecast while the river was still below the high-flow level, here {n2(level, 1)} mm/day. Of the {count(n_on)} onsets in the "
              f"test years {LEAD_WORD[first]}, the LSTM warned of {lstm_hits['wq']} with the river gauge and {lstm_hits['w']} without; the best simple "
              f"model ({_family(top_simple[1])}) of {hits[top_simple]}."
              + (f" {fno['hydro']} shows the largest flood of the test years." if SHOW_HYDRO else ""))

    # the upper bound: only the like-for-like leads
    fc = [(inp, k) for inp in ("wf", "wqf") for k in LEADS if inp in C and V(inp, k).get("like_for_like") and "best_baseline" in V(inp, k)]
    not_fair = sorted({k for inp in ("wf", "wqf") if inp in C for k in LEADS if not V(inp, k).get("like_for_like")})
    if fc:
        fc_call = {call(*c) for c in fc}
        fc_word = ("no clear difference" if fc_call == {"tie"} else "the LSTM ahead" if fc_call == {"lstm"} else "the simple model ahead" if fc_call == {"simple"}
                   else "a mixed result")
        fc_leads = sorted({k for _, k in fc})
        fc_fams = list(dict.fromkeys(_family(V(*c)["best_baseline"]) for c in fc))
        gauge_word = {"wf": "without", "wqf": "with"}
        fc_lstm = _and(n2(V(*c)["lstm_nse"]) + f" {gauge_word[c[0]]} the gauge" for c in fc)
        fc_base = _and(n2(V(*c)["best_baseline_nse"]) for c in fc)
        one_seed = all(V(*c)["n_seeds"] == 1 for c in fc)
        only = ", the only lead time at which both kinds of model read the same days of weather" if not_fair else ""
        plain_ = _and(n2(V({"wf": "w", "wqf": "wq"}[inp], k)["lstm_nse"]) for inp, k in fc)        # the same LSTM without the perfect forecast
        upper = (f"Given a perfect weather forecast (an upper bound that cannot be had in practice), the one-river LSTM reaches NSE {fc_lstm} "
                 f"{_leads(fc_leads)} (against {plain_} without it){only}; the {_and(fc_fams)} reaches {fc_base}: {fc_word}"
                 f"{' (one seed)' if one_seed else ''}.")
    else:
        upper = ""
    wins = f"""
<p><b>Does it beat the simple models?</b> {answer} {' '.join(detail)}</p>
<p>{onsets} {upper}</p>"""                                     # stands before the table it reads: text can fill a page foot, a table cannot

    # ---- why: what the diagnostic runs and the validation years show --------------------------------------------------
    def diag(inp):
        rows_ = S["diagnostic_fixed_length"][inp]
        a = next(r for r in rows_ if r["stopping_point"].startswith("chosen"))
        z = max((r for r in rows_ if not r["stopping_point"].startswith("chosen")), key=lambda r: r["updates"])
        m = lambda r, part: _mean(r[f"{part}_nse"])
        return dict(a=a, z=z, m=m, memorises=m(z, "train") > m(a, "train") and m(z, "val") < m(a, "val"))

    dq, dw = diag("wq"), diag("w")
    a, z, m = dq["a"], dq["z"], dq["m"]
    if dq["memorises"]:
        learn = (f"<li><b>It memorises its training years almost at once.</b> With the river gauge, early stopping kept the first seed as it was at "
                 f"update {a['updates']}: NSE {n2(m(a, 'train'))} in the training years, {n2(m(a, 'val'))} in the validation years and {n2(m(a, 'test'))} in the "
                 f"test years (mean of {LEADS[0]} to {LEAD_WORD[last]}). Trained on to update {z['updates']} as a check, it scores {n2(m(z, 'train'))}, "
                 f"{n2(m(z, 'val'))} and {n2(m(z, 'test'))}: better on the years it was fitted to, "
                 f"worse on {'the others' if m(z, 'test') < m(a, 'test') else 'the validation years'} (<i>overfitting</i>)"
                 + ("; the same happens without the gauge" if dw["memorises"] else "")
                 + f". The final models were at their best after {min(stops)} to {max(stops)} of the {count(final_budget['updates'])} updates allowed.</li>")
    else:
        learn = (f"<li><b>Longer training does not show memorising.</b> With the river gauge, stopped at update {a['updates']}: NSE {n2(m(a, 'train'))} "
                 f"(training years), {n2(m(a, 'val'))} (validation), {n2(m(a, 'test'))} (test); at update {z['updates']}: {n2(m(z, 'train'))}, "
                 f"{n2(m(z, 'val'))} and {n2(m(z, 'test'))}.</li>")

    top = max(VC["val"]["days"], key=lambda d: d["flow_mm"])
    same_month = sorted(datetime.date.fromisoformat(d["day"]) for d in VC["val"]["days"] if d["day"][:7] == top["day"][:7])
    floods = (f"{_word(len(same_month))} of them are the floods of {_and(d.day for d in same_month)} {same_month[0]:%B %Y}" if len(same_month) > 1
              else f"the largest is the flood of {same_month[0].day} {same_month[0]:%B %Y}")
    # which storms: results/floods_2004.json (check_2004_floods.py: storm tracks of HURDAT2 against the largest flow days of that month)
    F4 = R.get("floods_2004") or {}
    near = {c["date"]: c for c in F4.get("cyclones_within_500_km", [])} if F4.get("gauge_id") == B["gauge_id"] else {}
    storms = [near[d.isoformat()] for d in same_month if d.isoformat() in near]
    if storms and len(storms) == len(same_month):
        status = {c["status_then"] for c in storms}
        then = f" ({STATUS[next(iter(status))]} by then)" if len(status) == 1 and next(iter(status)) in STATUS and next(iter(status)) != "HU" else ""
        remnant = any("remnant of the hurricane" in x for x in F4.get("notes", [])) and status <= {"TD", "EX", "LO"}
        what = f"the remnants of hurricanes {_and(c['name'] for c in storms)}" if remnant else f"the tropical cyclones {_and(c['name'] for c in storms)}"
        floods += (f", when {what}{then} passed {_and(count(c['closest_km']) for c in storms)} km from the gauge (storm-track record HURDAT2, "
                   f"{cite('landsea2013')[1:-1]}; a storm nearby is not proof that it brought the rain)")
    seeds = FIN["wq"]["seeds"]
    va = {s: _mean([r["val"][lk(k)]["nse"] for k in LEADS]) for s, r in seeds.items()}
    te = {s: _mean([r["test"][lk(k)]["nse"] for k in LEADS]) for s, r in seeds.items()}
    sv = max(va, key=va.get)
    if len(seeds) > 1 and te[sv] == min(te.values()):
        rank = (f" It shows in the seeds: with the river gauge, the seed with the highest validation score has the lowest test score "
                f"({n2(te[sv])}; the other{'s' if len(seeds) > 2 else ''}: {_and(n2(v) for s, v in te.items() if s != sv)}).")
    elif len(seeds) > 1 and te[sv] < max(te.values()):
        rank = " It shows in the seeds: with the river gauge, the seed with the highest validation score is not the best in the test years."
    else:
        rank = ""
    n_days = len(VC["val"]["days"])
    more_rain = VC["val"]["max_rain_mm"] > VC["train"]["max_rain_mm"]
    judge = (f"<li><b>The validation years of one river are a poor judge.</b> In them, {_word(n_days)} days carry {pct(VC['val']['share_of_variance_in_top_days'])} "
             f"of the squared swings of the flow around its average, the quantity NSE is measured against (training years "
             f"{pct(VC['train']['share_of_variance_in_top_days'])}, test years {pct(VC['test']['share_of_variance_in_top_days'])}). {_cap(floods)}"
             + (f"; up to {n2(VC['val']['max_rain_mm'], 0)} mm of rain are stamped on one day, more than on any training day "
                f"({n2(VC['train']['max_rain_mm'], 0)} mm)" if more_rain else "")
             + f". So a handful of days chose the settings and the stopping points.{rank}</li>")
    untuned = any("Batch size" in x and "not tried" in x for x in S["notes"])
    if edge_knobs:
        small = (f"<li><b>The chosen setting sits at the edge of what was tried</b> ({_and(edge_knobs)})"
                 + (", and the number of stretches per update was not tuned" if untuned else "")
                 + ": a better one-river setting may exist; the result holds for the LSTM as tuned here.</li>")
    else:
        small = ""
    why = f"""
<p style="break-after:avoid"><b>{'Why not?' if not ahead else 'What limits it?'}</b> The results show {_word(2 + bool(small))} things.</p>
<ul class="s3l">{learn}{judge}{small}</ul>"""

    # ---- one flood --------------------------------------------------------------------------------------------------
    peak = max(VC["test"]["days"], key=lambda d: d["flow_mm"])
    peak_day = datetime.date.fromisoformat(peak["day"])
    silent = lstm_hits["w"] == 0 and lstm_hits["wq"] <= 0.1 * n_on                 # the LSTM forecasts stayed below the high-flow level at (nearly) every onset
    if silent:
        see = (f"no line rises on time. {QUOTED_CAP} repeats the observed peak one day late, as it must. The two LSTM lines also rise late, and far "
               f"less: at the onset they stay below the high-flow level, as the LSTM did at {count(n_on - lstm_hits['wq'])} of the {count(n_on)} onsets "
               f"of the test years even with the gauge.")
    else:
        see = f"{QUOTED} repeats the observed peak one day late, as it must; compare the two LSTM lines with the black one on the day of the rise."
    not_read = (f" How not to read it: the {n2(peak['rain_mm_same_day'], 0)} mm stamped on the peak day were not yet in the models&rsquo; inputs when that day "
                f"was forecast, but they are not all rain that fell after the forecast was made: by the rain table&rsquo;s day stamp, part of a day&rsquo;s "
                f"rain has fallen before the forecast for that day is issued ({TIMING_REF}).")
    fig_hydro = "" if not SHOW_HYDRO else fig("lstm_single_hydrograph.png",
                    f"<b>{fno['hydro']}. The largest flood of the test years on the {river}, forecasts made {LEAD_WORD[first]}.</b> Top: daily rain as "
                    f"stamped in the rain table. Bottom: observed flow (black; peak {n2(peak['flow_mm'], 1)} mm/day on {peak_day.day} {peak_day:%B %Y}), "
                    f"{QUOTED} (grey), the one-river LSTM without (light blue) and with (blue) the river gauge. What to see: {see}{not_read}", FIG_HYDRO)

    pointer = f"<p>{ALL_REF.capitalize()} sets the LSTM trained on all twelve rivers against this one on the {river}.</p>"

    # ---- what the training looks like: the stored learning curves (guarded: without the results file or the figure nothing is shown) ----
    _, curves_text, curves_fig = learning_curves(R)

    # the heading states the result when the one-river LSTM is ahead in no row of the comparison table (otherwise the plain heading)
    heading = "3 An LSTM for one river" + (": not ahead of the simple models" if not ahead else "")
    # Layout: both figures are floats placed right after the comparison table. Together they fill most of one page; the 'why' list and the
    # paragraph on the learning curves, which closes the section, follow them and fill the feet of the pages around them. (With the figure of
    # the learning curves as the last element, the foot of its page stayed empty: a new section starts below every figure of the section before.)
    return f"""{LAYOUT}{TIGHT}<h2>{heading}</h2>{intro}{tuning}
{memory}{fig_memory}{wins}
<div class="blk">{cmp_table}</div>
{floated(fig_hydro) if fig_hydro else ''}{floated(curves_fig) if curves_fig else ''}
{why}
{curves_text}
{pointer}"""
