"""Report section 4 (4.1 to 4.3) - one LSTM for all twelve rivers: against the simple models, and where its skill comes from.

Reads results/lstm_all.json and lstm_all_scores.csv (written by 04_lstm_all.py) and, for the one-river comparison only,
results/lstm_single.json (03_lstm.py). Nothing is computed here beyond differences, counts, smallest/largest and formatting;
every sentence whose wording depends on an outcome is chosen from the stored numbers.

Layout: the two figures of 04_lstm_all.py are drawn for a width of 170 mm (smallest lettering 7.5 pt), so a figure is shown at about
that width or left out, never shrunk. lstm_all.png, the central figure of the project, is shown (Figure 4.1); Table 4.1 adds what the
figure cannot show (seed spread as a number, p). lstm_all_basins.png is left out: the per-basin table (the 'per-basin score table' the
plan names, Table 4.2) carries its scores in a fifth of the space. SHOW_PANELS / SHOW_BASINS switch the two figures.

Perfect weather forecast: the comparison of the LSTM with the simple models is like-for-like only where the results file says so (key
like_for_like; 3 days ahead). At the other lead times no winner is called here and the cells of Table 4.1 are grey. Even 3 days ahead the two
read the same days but not exactly the same variables (note of 04_lstm_all.py), so there the text says 'scores higher', never 'is better'.

'Clear' means one thing in the report: the three checks stated in 4.2 (the other model outside the single seeds, higher in most basins, p below
0.05, two-sided). Section 4.3 applies them to the gains of Table 4.3 (the results file stores seeds, basin counts and p for them).
Milestone 2 is stated for both forecasts that could be run in practice (s0_summary.milestone2). Section 4.1 says how twelve rivers share one
loss; the network itself is specified in section 3. Table 4.1 gives every lead time its own column under a group header; Table 4.2 shows the
simple model that Table 4.1 uses for its case (the one with the highest validation score) beside the others.

Added on 4 October 2026, each inside a guarded block (a missing file or key leaves the piece out and prints one line to stderr):
  the table 'The three networks at a glance' in 4.1 (no table number): the settings of the one-river LSTM, the all-river LSTM and the Transformer,
    every cell from lstm_single.json, lstm_all.json and model_comparison.json; the LSTM weight count is written out and checked against the stored one;
  the loss as a formula, its constant read from the code of flood.fit, and its source from results/published_scores.json;
  'How large could the difference be?' in 4.2: the ranges of results/uncertainty.json (13_uncertainty.py), named as the lead fixed it
    ('the range when the test years and the basins are drawn again (95%)'); section 8 says what such a range covers;
  the note 'How these scores relate to published results' in 4.3: published numbers only from results/published_scores.json, set beside the
    upper-bound rows only, 'of the same size' and never more, with who read the sources and when;
  the pointers to the learning curves (Figure 3.2, when section 3 shows them) and to the key figure on page 1 (when figures/key_figure.png exists).
The headings of 4.2 and 4.3 state the result when the readings allow it, and fall back to the plain heading otherwise.
"""
import inspect
import re
import sys

import flood
from common import FIG
from flood import FORCING, LAND_SHORT, LSTMNet
from report_common import LEAD_WORD, MODEL_NAME, cite, count, fig, n2, pct, signed, table
from report_sections.s0_summary import LAYOUT, _day, floated, forecast_where, milestone2, nse_reading, onset_clear, share_word, timing_facts
from report_sections.s3_single import curves_figure

LEADS = sorted(LEAD_WORD)
SETS = ("w", "wq", "wf", "wqf")                 # the order of Table 4.1 and of the panels' reading order
UPPER = ("wf", "wqf")                           # given the weather of the days being forecast: an upper bound
SET_NAME = {"w": "past weather", "wq": "past weather + river gauge", "wf": "past weather + perfect weather forecast",
            "wqf": "past weather + river gauge + perfect weather forecast"}
STATIC_WORD = {"log_area": "area", "elev_mean": "elevation", "slope_mean": "slope", "p_mean": "mean rainfall", "aridity": "dryness of the climate",
               "p_seasonality": "season of the rain", "frac_forest": "forest share", "soil_depth_pelletier": "soil depth", "sand_frac": "sand in the soil",
               "geol_permeability": "permeability of the rock"}
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
QUOTED = "&lsquo;tomorrow = today&rsquo;"
P_CUT = 0.05                                    # the usual line for 'p is small'; the text of section 4.2 states it
SHOW_PANELS, SHOW_BASINS = True, False          # lstm_all.png (Figure 4.1) / lstm_all_basins.png (instead of the per-basin table)
FIG1_SHARE, FIG2_SHARE = 84, 84                 # % of the text width (184 mm) of a figure that is shown; the figures are drawn for 170 mm (92%)
TIMING_REF = "section 4.5"                      # where the rain time stamps are tested
RANGE_REF = "section 8"                         # where the report says what a resampling range covers and what it does not
SAME_SIZE = 0.10                                # published and own NSE 'of the same size': no pair further apart than this (used only without the ranges)
FEW_HUNDREDTHS = 0.05                           # 'a few hundredths at most': every end of every range closer to zero than this
TOP = 4                                         # 'the four easiest / largest / smallest' basins: a third of the twelve

# reading of one comparison (one input set, one lead), as part of a sentence; the rule is _reading() and is stated in section 4.2
PHRASE = {"lstm": "the LSTM is better", "lstm_avg": "only the averaged forecast of the LSTM is ahead",
          "lstm_med": f"the LSTM scores higher, but p is above {P_CUT:g}", "tie": "there is no clear difference",
          "simple": "the simple model is better", "nocall": "no winner is called"}
# with the perfect weather forecast the two kinds of model do not read exactly the same variables: 'scores higher', not 'is better'
SOFT = dict(PHRASE, lstm="the LSTM scores higher", simple="the simple model scores higher")


# the section's own table style: closer rows (three tables in under two pages); .basins is the table with one column per basin
TIGHT = ("<style>table.tight th,table.tight td{padding:1.8pt 4pt;line-height:1.3}"
         "table.basins{table-layout:fixed} table.basins th,table.basins td{padding:1.8pt 2pt;text-align:center} table.basins th{font-size:7.6pt}"
         "table.basins td:first-child{text-align:left;padding-left:4pt}"
         "table.basins .g{border-left:1px solid var(--axis)} .nl{color:var(--muted)}"
         "table.t41 th,table.t41 td{padding:1.9pt 3pt;text-align:right} table.t41 td:first-child,table.t41 th:first-child{text-align:left;padding-left:4pt}"
         "table.t41 th.grp{text-align:center;border-bottom:1px solid var(--grid)} table.t41 .g{border-left:1px solid var(--axis)}"
         "table.t41 tr.sub th{font-weight:400;font-size:7.9pt;border-top:none;padding-top:0} sub,sup{line-height:0}</style>")      # .nl: a cell that is not like-for-like


def _word(n):
    return NUMBER.get(n, str(n))


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _leads(ks):
    """[1] -> '1 day ahead', [2, 3] -> '2 and 3 days ahead'."""
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def _p(v):
    """A p-value: two decimals, three below 0.01."""
    if v is None or v != v:
        return "&ndash;"
    return "&lt;0.001" if v < 0.001 else f"{v:.3f}" if v < 0.0095 else f"{v:.2f}"


def _pts(v):
    """A difference of two shares in percentage points: 0.457 -> '+46'."""
    return signed(100 * v, 0)


def _three(vals, em, bold=False, grey=()):
    """Three formatted values (1, 2, 3 days ahead) side by side in one table cell, as in Table 2.1; positions in `grey` are printed grey."""
    cells = "".join(f'<span{" class=nl" if i in grey else ""} style="display:inline-block;width:{em}em;text-align:right">{v}</span>' for i, v in enumerate(vals))
    return f'<span style="white-space:nowrap">{"<b>" + cells + "</b>" if bold else cells}</span>'


def _cap(s):
    return s[0].upper() + s[1:]


def _named(key):
    """A source named in the sentence: 'Kratzert et al. (2019a)'. Remembered for the source list like cite()."""
    name, year = cite(key)[1:-1].rsplit(", ", 1)
    return f"{name} ({year})"


def _sg(v, d=2):
    """A signed difference; a value that rounds to zero is printed as a plain 0.00."""
    return n2(0.0, d) if round(v, d) == 0 else signed(v, d)


def _span(v, fmt=count):
    """[4.4, 41.6] -> '4 to 42'; one value when both ends print alike."""
    lo, hi = fmt(min(v)), fmt(max(v))
    return lo if lo == hi else f"{lo} to {hi}"


def _loss_constant():
    """The constant that flood.fit adds to a basin's flow spread in the loss weight, read from the code itself, or None."""
    try:
        m = re.search(r'q_sd"\]\s*\+\s*([0-9.]+)\)\s*\*\*\s*2', inspect.getsource(flood.fit))
        return float(m.group(1))
    except (AttributeError, OSError, TypeError, ValueError) as e:
        print(f"s4_all: the constant of the loss weight was not found in flood.fit ({type(e).__name__}: {e}); the loss formula is left out", file=sys.stderr)
        return None


def _lstm_weights(cells, n_in, stored, n_out=len(LEAD_WORD)):
    """The weight count of one LSTM layer and a linear output layer, written out. It is checked against the stored count: if the two differ a
    ValueError is raised, which networks_table turns into 'no table' and one line on stderr. An LSTM layer has four blocks (three gates and the
    new value), each with cells x (inputs + cells) weights and two bias vectors; the output layer has n_out x cells weights and n_out biases."""
    head = n_out * cells + n_out
    if 4 * cells * (n_in + cells + 2) + head != stored:
        raise ValueError(f"the stored weight count {stored} is not that of an LSTM layer with {cells} cells and {n_in} inputs")
    return f"{count(stored)}&nbsp;= 4&times;{cells}&times;({n_in}+{cells}+2)+{head}"


def networks_table(R, T, quiet=False):
    """'The three networks at a glance' (section 4.1; it has no table number): the settings of the final networks with past weather + river gauge,
    every cell read from lstm_single.json, lstm_all.json and model_comparison.json. Settings in which the three networks agree stand in the
    caption. Returns '' (and, unless quiet, prints one line) when a results file or a key is missing: section 4.1 then states the settings in a
    sentence, and section 7 (which asks with quiet=True whether the table exists) keeps its own list."""
    try:
        A, S1, M = R["lstm_all"], R["lstm_single"], R["model_comparison"]
        n_seeds_all = len(A["sets"]["wq"]["seeds"])
        if any(x.get("quick_mode") for x in (A, S1, M)):
            raise KeyError("a results file of a smoke test")
        n_w, clip = len(FORCING), flood.Cfg().clip
        days = lambda c: f"{count(c['warmup'])} of history + {count(c['block'])} scored"
        runs = lambda cost: [cost[k] for k in sorted(cost, key=int)]
        lstm = lambda cells: f"1 LSTM layer of {cells} memory cells, linear output layer"
        # one-river LSTM (section 3): the chosen setting, the final models with the river gauge
        TU, fin = S1["tuning"], S1["finals"]["wq"]["seeds"]
        ch, fb = TU["chosen"], TU["final_budget"]
        tr1 = [fin[k]["training"] for k in sorted(fin, key=int)]
        batch1 = sorted({int(b) for b in T["hparam_log"].batch})
        one = dict(net=lstm(ch["hidden"]), inputs=f"{n_w + 1}: weather ({n_w}), river gauge", weights=_lstm_weights(ch["hidden"], n_w + 1, tr1[0]["n_params"]),
                   days=days(ch), rate=f"{ch['lr']:g}; {ch['dropout']:g}", check=f"every {fb['eval_every']} updates; {count(fb['updates'])}",
                   kept=_span([t["best_update"] for t in tr1]), runs=f"{len(tr1)}; {_span([t['seconds'] for t in tr1])} s",
                   batch=batch1[0] if len(batch1) == 1 else None, patience=fb["patience"])
        # all-river LSTM (sections 4 to 6): the configuration of 04_lstm_all.py, the trainings with the river gauge
        cfg, tr2 = A["config"], runs(A["sets"]["wq"]["cost"])
        n_all = n_w + 1 + len(A["static_inputs"])
        all_ = dict(net=lstm(cfg["hidden"]), inputs=f"{n_all}: with {len(A['static_inputs'])} basin properties", weights=_lstm_weights(cfg["hidden"], n_all, tr2[0]["n_params"]),
                    days=days(cfg), rate=f"{cfg['lr']:g}; {cfg['dropout']:g}", check=f"every {cfg['eval_every']} updates; {count(cfg['updates'])}",
                    kept=_span([t["best_update"] for t in tr2]), runs=f"{len(tr2)}; {_span([t['seconds'] for t in tr2])} s",
                    batch=cfg["batch"], patience=cfg["patience"])
        # Transformer (section 7): fixed properties of TransformerNet in 09_model_comparison.py (position code, mask, output layer) and its stored settings
        TS, TM = M["settings"]["transformer"], M["models"]["transformer"]
        tr3 = [TM["per_seed"][k]["cost"] for k in sorted(TM["per_seed"], key=int)]
        tra = dict(net=f"{TS['layers']} attention layers ({TS['heads']} heads, width {TS['hidden']}, feed-forward {TS['feed_forward']}), sinusoidal position "
                       f"code, causal mask, linear output layer",
                   inputs=f"{TM['n_inputs']}" + (", as the all-river LSTM" if TM["n_inputs"] == n_all and M["static_inputs"] == A["static_inputs"] else ""),
                   weights=count(TM["n_params"]), days=days(TS), rate=f"{TS['lr']:g}; {TS['dropout']:g}",
                   check=f"every {TS['eval_every']} updates; {count(TS['updates'])}", kept=_span([t["best_update"] for t in tr3]),
                   runs=f"{len(tr3)}; {_span([t['seconds'] for t in tr3])} s", batch=TS["batch"], patience=TS["patience"])
        if not (len(tr2) == n_seeds_all and all(t["n_params"] == tr[0]["n_params"] for tr in (tr1, tr2, tr3) for t in tr)):
            raise ValueError("the trainings of one network do not share one weight count, or the seeds differ from those of the tables")
        cols = (one, all_, tra)
        rows = [("Network", "net"), ("Inputs per day", "inputs"), ("Weights to learn", "weights"), ("Days per stretch", "days"),
                ("Learning rate; dropout", "rate"), ("Check; update limit", "check"), ("Trainings; seconds", "runs")]       # where each state was kept: Figure 3.2, sections 3 and 7
        body = [[f"<b>{label}</b>"] + [str(c[key]) for c in cols] for label, key in rows]
        common = [f"Adam, gradients clipped at a norm of {clip:g}"]
        if len({c["batch"] for c in cols}) == 1 and one["batch"]:
            common.append(f"{one['batch']} stretches per update")
        if len({c["patience"] for c in cols}) == 1:
            common.append(f"the loss above, a stop after {one['patience']} checks without a better score")
        ref = M["settings"].get("lstm_all_config_for_reference")
        twin = (" Section 7 sets the Transformer against an LSTM trained like it, not against the middle column."
                if ref and (ref["warmup"], ref["block"]) != (TS["warmup"], TS["block"]) else "")
        caption = (f"<b>The three networks at a glance,</b> each with past weather + river gauge (how the settings were chosen: Table 3.1, the check "
                   f"below, section 7). LSTM "
                   f"weights: four blocks (three gates and the new value) of cells &times; (inputs + cells + 2), plus the output layer. All three: "
                   f"{', '.join(common)}. Seconds: busy laptop.{twin}")
        return table(["", "One-river LSTM (section 3)", "All-river LSTM (sections 4 to 6)", "Transformer (section 7)"], body, cls="num tight nets",
                     widths=["19%", "25%", "25%", "31%"], caption=caption)
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as e:
        if not quiet:
            print(f"s4_all: the table of the three networks was left out ({type(e).__name__}: {e}); section 4.1 states the settings in a sentence",
                  file=sys.stderr)
        return ""


def _how_large(R, best_wq, n):
    """'How large could the difference be?' (section 4.2), from results/uncertainty.json (13_uncertainty.py): the LSTM's NSE minus that of the best
    simple model with the river gauge, as the median of the basin differences with its range when the test years and the basins are drawn again,
    and on the same footing what the weather of the days being forecast adds. '' and one line on stderr when the file or a key is missing, or
    when the file's best simple model is not the one of Table 4.1. best_wq: {lead: the simple model of Table 4.1 with the river gauge}."""
    U = R.get("uncertainty")
    if not isinstance(U, dict) or U.get("quick_mode"):
        print("s4_all: results/uncertainty.json (of a full run) was not found; 'How large could the difference be?' is left out", file=sys.stderr)
        return ""
    try:
        theirs = U.get("best_simple_model", {}).get("wq")
        if theirs is not None and any(theirs[f"l{k}"] != best_wq[k] for k in LEADS):
            raise ValueError("its best simple model with the river gauge is not that of Table 4.1")
        st = lambda name, k: U["stats"][f"nse_median_of_diffs.{name}.l{k}"]
        num = lambda x: f"{_sg(x['value'])} ({_sg(x['both']['lo'])} to {_sg(x['both']['hi'])})"
        tie, gain = {k: st("lstm_wq_minus_best_simple", k) for k in LEADS}, {k: st("wqf_minus_wq", k) for k in LEADS}
        ends = [x["both"][e] for x in tie.values() for e in ("lo", "hi")]
        both_ways = all(x["both"]["lo"] < 0 < x["both"]["hi"] for x in tie.values())
        reading = ("a few hundredths at most, in either direction" if both_ways and max(abs(e) for e in ends) < FEW_HUNDREDTHS else
                   f"between {_sg(min(ends))} and {_sg(max(ends))}")
        return (f"<li><b>How large could the difference be?</b> With the river gauge, the LSTM&rsquo;s NSE minus that of the best simple model is "
                f"{_and(num(tie[k]) for k in LEADS)} in the typical basin ({_leads(list(LEADS))}; the median of the {n} basin differences, not the "
                f"difference of the medians in Table 4.1): {reading}. In brackets: the range when the test years and the basins are drawn again (95%; "
                f"{RANGE_REF} says what it covers). On the same footing the weather of the days being forecast adds {_and(num(gain[k]) for k in LEADS)}.</li>")
    except (KeyError, TypeError, ValueError) as e:
        print(f"s4_all: results/uncertainty.json is not in the expected form ({type(e).__name__}: {e}); 'How large could the difference be?' is left out",
              file=sys.stderr)
        return ""


def _published(R, A, n, table_skill, table_cmp, tie):
    """The note 'How these scores relate to published results' (section 4.3). Published numbers come only from results/published_scores.json and
    are set beside the upper-bound rows only, as that file says (how_to_word_it): 'of the same size', never more. With results/uncertainty.json
    the three own numbers carry their ranges. tie: with the river gauge every reading of section 4.2 is 'no clear difference'.
    '' and one line on stderr when the file or a key is missing."""
    P = R.get("published_scores")
    if not isinstance(P, dict):
        print("s4_all: results/published_scores.json was not found; the note on published results is left out", file=sys.stderr)
        return ""
    try:
        K, F_ = P["sources"]["kratzert2019"], P["sources"]["feng2020"]
        km, fm = K["median_nse"], F_["median_nse"]
        own = lambda inp, k: A["sets"][inp]["test"]["summary"][f"l{k}"]["nse_median"]
        first, last = LEADS[0], LEADS[-1]
        # (own value, the published values of the same kind of information)
        pairs = [(("wf", last), [km["lstm_with_basin_properties_nse_loss_single"], km["lstm_with_basin_properties_nse_loss_ensemble"], fm["projection_no_observed_flow_671"]]),
                 (("wqf", first), [fm["with_flow_observed_1_day_before_671"]]), (("wqf", last), [fm["with_flow_observed_3_days_before_671"]])]
        U = R.get("uncertainty") if isinstance(R.get("uncertainty"), dict) and not R["uncertainty"].get("quick_mode") else None
        rng = {}
        if U:
            try:
                rng = {key: U["stats"][f"nse_median.lstm_{key[0]}.l{key[1]}"]["both"] for key, _ in pairs}
                if any(abs(U["stats"][f"nse_median.lstm_{key[0]}.l{key[1]}"]["value"] - own(*key)) > 0.005 for key, _ in pairs):
                    raise ValueError("its values are not those of lstm_all.json")
            except (KeyError, TypeError, ValueError) as e:
                print(f"s4_all: the ranges of the upper-bound scores were not read from results/uncertainty.json ({type(e).__name__}: {e})", file=sys.stderr)
                rng = {}
        val = lambda key: n2(own(*key)) + (f" ({n2(rng[key]['lo'])} to {n2(rng[key]['hi'])})" if rng else "")
        gap = max(abs(own(*key) - v) for key, pub in pairs for v in pub)
        inside = bool(rng) and all(rng[key]["lo"] <= v <= rng[key]["hi"] for key, pub in pairs for v in pub)
        if inside:
            size = (f"But the scores are of the same size: every published value lies inside the range of ours when the test years and the basins are drawn "
                    f"again (in brackets; {RANGE_REF}).")
        elif gap <= SAME_SIZE:
            size = "But the scores are of the same size." + (f" In brackets: the range when the test years and the basins are drawn again ({RANGE_REF})." if rng else "")
        else:
            size = f"The scores differ by up to {n2(gap)} NSE." + (f" In brackets: the range when the test years and the basins are drawn again ({RANGE_REF})." if rng else "")
        cells = sorted({K["n_cells"], F_["n_cells"]})
        other_rain = all(str(x.get("forcing", "")).lower() != "daymet" for x in (K, F_))
        unlike = (f"This is no like-for-like test: {n} south-eastern basins against several hundred across the United States, other years, "
                  + ("another rain product, " if other_rain else "")
                  + f"{A['config']['hidden']} memory cells against {_and(cells)}, and {LEAD_WORD[first]} our model reads two more days of weather.")
        falls = own("w", first) < own("wf", last) and own("wq", first) < own("wqf", first)             # without that weather the same network scores lower
        withheld = (f" With that weather withheld (rows 1 and 2 of {table_skill}), the same network falls to {n2(own('w', first))} and "
                    f"{n2(own('wq', first))} {LEAD_WORD[first]}: one more sign that, among the models tried, the limit is the information, not the "
                    f"network.") if falls and (inside or gap <= SAME_SIZE) else ""
        unread = ", not yet by a team member" if "no team member has read" in str(P.get("checked_by", "")).lower() else ""
        who = f" (read at the source by an AI agent on {_day(P['checked_on'])}{unread}; <code>results/published_scores.json</code>)"
        return (f'<div class="note blk"><b>How these scores relate to published results.</b> Published LSTM scores on CAMELS{who} are for estimating the '
                f"flow of a day from the weather up to and including that day: in this report&rsquo;s terms, the perfect weather forecast. {_named(K['citation_key'])} "
                f"report a median NSE of {n2(km['lstm_with_basin_properties_nse_loss_single'])} for one LSTM and "
                f"{n2(km['lstm_with_basin_properties_nse_loss_ensemble'])} for the average of {K['n_seeds']} seeds, on {count(K['n_basins'])} basins; "
                f"{_named(F_['citation_key'])} report {n2(fm['projection_no_observed_flow_671'])} on {count(F_['n_basins'])} basins, "
                f"{n2(fm['with_flow_observed_1_day_before_671'])} with the flow observed the day before and "
                f"{n2(fm['with_flow_observed_3_days_before_671'])} with flow three days old. Given the same kind of information, our LSTM scores "
                f"{val(('wf', last))} without a gauge ({LEAD_WORD[last]}, where it reads the weather up to the day being forecast and no further), "
                f"{val(('wqf', first))} with the flow of the day before and {val(('wqf', last))} with flow three days old. {unlike} {size}{withheld}</div>")
    except (KeyError, TypeError, ValueError) as e:
        print(f"s4_all: results/published_scores.json is not in the expected form ({type(e).__name__}: {e}); the note on published results is left out",
              file=sys.stderr)
        return ""


def _reading(c):
    """One LSTM-against-simple-model comparison -> lstm / lstm_avg / lstm_med / tie / simple (the rule stated in section 4.2);
    nocall where the results file marks the comparison as not like-for-like."""
    if c.get("like_for_like") is False:
        return "nocall"
    most, few = c["n_basins_lstm_higher"] > c["n_basins"] / 2, c["n_basins_lstm_higher"] < c["n_basins"] / 2
    small_p, below = c["wilcoxon_p"] < P_CUT, c["baseline_against_seed_spread"] == "below every seed"
    if most and small_p:
        return "lstm" if below else "lstm_avg"
    if few and small_p:
        return "simple"
    return "lstm_med" if most and below else "tie"


def _say(classes):
    """{lead: reading} -> 'there is no clear difference at any of the three lead times' or '1 day ahead ...; 2 and 3 days ahead the LSTM is better'."""
    groups = {}
    for k in LEADS:
        groups.setdefault(classes[k], []).append(k)
    if len(groups) == 1:
        c = next(iter(groups))
        return PHRASE[c] + (" at any of the three lead times" if c == "tie" else " at all three lead times")
    return "; ".join(f"{_leads(ks)} {PHRASE[c]}" for c, ks in groups.items())


def html(R, T):
    A, S = R["lstm_all"], T["lstm_all_scores"]
    C, LV, ST = A["comparison"], A["decomposition"]["levels"], A["decomposition"]["steps"]
    ids, split = list(A["basins"]), A["split"]
    n = len(ids)
    lk = lambda k: f"l{k}"
    first, last = LEADS[0], LEADS[-1]
    best = lambda inp, k: C[inp]["best_baseline_by_validation"][lk(k)]                # the simple model chosen on the validation years
    cmp_ = lambda inp, k, name=None: C[inp]["baselines"][name or best(inp, k)][lk(k)]
    family = lambda name: MODEL_NAME["base_" + A["baselines"][name]["model"]].lower()
    seeds = A["sets"]["wq"]["seeds"]
    n_seeds = len(seeds)
    cls = {inp: {k: _reading(cmp_(inp, k)) for k in LEADS} for inp in SETS}
    wins = lambda inp, name=None: f"LSTM higher in {_and(cmp_(inp, k, name)['n_basins_lstm_higher'] for k in LEADS)} of {n} basins"
    higher = lambda inp, ks=LEADS, raw=False: ([cmp_(inp, k)["n_basins_lstm_higher"] for k in ks] if raw else
                                               _and(cmp_(inp, k)["n_basins_lstm_higher"] for k in ks))

    # ---- 4.1 what changes ---------------------------------------------------------------------------------------------
    check = A["setting_check"]
    cand = check["table"]
    chosen = next(c for c in cand if c["chosen"])
    gap = chosen["val_nse_mean"] - max(c["val_nse_mean"] for c in cand if not c["chosen"])
    spread = max(check["chosen_val_nse_mean_by_seed"]) - min(check["chosen_val_nse_mean_by_seed"])
    if gap <= spread:
        pick = (f"Its margin of {n2(gap, 3)} is smaller than the {n2(spread, 3)} by which {_word(n_seeds)} seeds of one setting differ, so the choice "
                f"is not a finding.")
    else:
        pick = f"Its margin of {n2(gap, 3)} is larger than the {n2(spread, 3)} by which {_word(n_seeds)} seeds of one setting differ."
    props = [STATIC_WORD.get(s, s) for s in A["static_inputs"]]
    # who trained what (results/published_scores.json, sources.kratzert2018 / kratzert2019: set_up): one network for all basins, with basin
    # properties as inputs, is the 2019 paper; the 2018 paper trained one network per basin and one per region
    approach = (f"the standard approach since {_named('kratzert2019')}, who added basin properties as inputs; {_named('kratzert2018')} still trained "
                f"one network per basin or per region")
    hidden, rates = sorted({c["hidden"] for c in cand}), sorted({c["lr"] for c in cand})
    scores = [c["val_nse_mean"] for c in cand]
    # fair-test caveat asked for by the reviewer of 04_lstm_all.py: some basin properties are climate averages of years that reach into the
    # validation and test years. The properties and the years are read from the note in lstm_all.json; without that note the sentence is left out.
    note = next((x for x in A.get("notes", []) if "climate averages" in x), "")
    keys, span = re.search(r"\(((?:\w+, )*\w+):", note), re.search(r"(\d{4}) to (?:\w+ )?(\d{4})", note)
    clim = [k for k in keys.group(1).split(", ") if k in A["static_inputs"]] if keys else []
    later = [w for w, part in (("validation", "val"), ("test", "test")) if span and int(span.group(1)) <= int(split[part][1][:4])
             and int(span.group(2)) >= int(split[part][0][:4])]
    if clim and later:
        climate = (f" {_cap(_word(len(clim)))} of them ({_and(STATIC_WORD.get(k, k) for k in clim)}) are CAMELS averages of "
                   f"{span.group(1)}&ndash;{span.group(2)}, years that reach into the {' and '.join(later)} years: one fixed number per basin, not daily data.")
    else:
        climate = ""
        print("s4_all: the note on the climate-average basin properties was not found; its sentence was left out", file=sys.stderr)
    cfg = A["config"]
    nets = networks_table(R, T)                                          # the settings of the three networks, or '' (then a sentence gives them)
    f_curves = curves_figure(R)                                                   # 'Figure 3.2' when section 3 shows the learning curves
    curves_ref = f" (learning curves: {f_curves}, middle)" if f_curves else ""
    settings = (f"those of section 3; the table below gives the settings{curves_ref}." if nets else
                f"those of section 3, here with {cfg['hidden']} memory cells, stretches of {cfg['warmup']} days of history and {cfg['block']} scored days, "
                f"{cfg['batch']} stretches per update, a validation check every {cfg['eval_every']} updates and at most {count(cfg['updates'])} "
                f"updates{curves_ref}.")
    # the loss as a formula: its constant is read from the code of flood.fit; its source from results/published_scores.json (guarded)
    eps = _loss_constant()
    if eps is not None:
        loss = (f"The flow of all basins is put on one common scale, and the loss is the mean over all scored forecasts of (forecast &minus; "
                f"observed)<sup>2</sup> / (s<sub>b</sub> + {eps:g})<sup>2</sup>, where s<sub>b</sub> is the flow spread of basin b on that scale: close to "
                f"dividing by that basin&rsquo;s flow variance, so that wet and dry rivers count nearly alike.")
        try:
            pub = R["published_scores"]["sources"]["kratzert2019"]
            if "NSE loss" not in pub["loss"]:
                raise KeyError("the record does not name the loss of that paper")
            loss += (f" It has the form of the basin-averaged NSE loss of {_named(pub['citation_key'])}"
                     + (f", where the constant is also {pub['loss_epsilon']:g}." if float(pub["loss_epsilon"]) == eps else "."))
        except (KeyError, TypeError, ValueError) as e:
            print(f"s4_all: the source of the loss was not read from results/published_scores.json ({type(e).__name__}: {e}); it is not named", file=sys.stderr)
    else:                                                                         # the constant was not found in the code: the description as it stood
        loss = ("The flow of all basins is put on one common scale, and in the loss each basin&rsquo;s squared error is divided by the square of (its own "
                "flow spread + 0.1 &times; the spread of all basins): close to dividing by that basin&rsquo;s flow variance, so that wet and dry rivers "
                "count nearly alike.")
    shared = (f"<p><b>How {_word(n)} rivers share one network.</b> The network and the training are {settings} Every stretch of an update comes from a "
              f"basin drawn at random. {loss}</p>")
    changes = f"""
<h3>4.1 What changes</h3>
<p>One LSTM is trained on all {n} basins together, {approach}. {_cap(_word(len(props)))} fixed basin
properties, repeated on every day as extra inputs, tell the rivers apart: {_and(props)}. None comes from the flow record.{climate} The river
gauge, where it is used, is one more daily input {cite('feng2020')}.</p>
{shared}
{floated(nets) if nets else ''}
<p>A check on the validation years compared {_word(len(cand))} settings with one seed each ({' or '.join(str(h) for h in hidden)} memory cells,
learning rate {' or '.join(f'{r:g}' for r in rates)}). They scored NSE {n2(min(scores), 3)} to {n2(max(scores), 3)}; the highest was kept
({chosen['hidden']} cells, {chosen['lr']:g}). {pick} Each of the {_word(len(SETS))} input sets was then trained from {_word(n_seeds)} seeds;
their day-by-day average forecast is what the tables call the LSTM.</p>"""

    # ---- 4.2 against the simple models: the method, then the table (with SHOW_PANELS the figure stands between them) ----
    no1, no2 = "4.1", "4.2" if SHOW_PANELS else "4.1"                             # numbers of the panel figure and of the basin figure
    T_CMP, T_BASIN, T_SKILL = "Table 4.1", "Table 4.2", "Table 4.3"               # the comparison, the per-basin scores, where the skill comes from
    unfair = {inp: [k for k in LEADS if cmp_(inp, k).get("like_for_like") is False] for inp in SETS}      # lead times that are not like-for-like
    any_unfair = any(unfair.values())
    c1 = cmp_("wq", first)
    caption1 = (f"<b>Figure {no1}. One LSTM against the best simple model given the same inputs, test years.</b> Right-hand panels: with a perfect "
                f"weather forecast, the upper bound. The blue marker is the averaged forecast of the {_word(n_seeds)} seeds, which usually scores a "
                f"little higher than each single seed (black bars)."
                + (" Grey counts with an asterisk: not like-for-like, no winner is called." if any_unfair else "")
                + f" What to see: bottom left the two lines lie on each other ({n2(c1['lstm_nse_median'])} against "
                  f"{n2(c1['baseline_nse_median'])} {LEAD_WORD[first]}); on the right both stand far above the dashed line.")
    figure1 = fig("lstm_all.png", caption1, FIG1_SHARE)
    with_persistence = any(A["baselines"][name]["model"] == "persistence" for name in C["wq"]["baselines"])
    two_sided = any("two-sided" in x for x in A.get("notes", []))                 # how 04_lstm_all.py calls the test (its note)
    method = f"""
<p><b>What is compared.</b> For each of the {_word(len(SETS))} input sets of section 1: the LSTM against the best simple model given the same
inputs, that is the ridge regression or random forest{f" (or {QUOTED}, where the gauge is read)" if with_persistence else ""} of section 2 with the
highest score in the validation years (Figure {no1}, {T_CMP}; test years, median NSE of the {n} basins).
<b>How a difference is judged.</b> A model is called better, and a difference <i>clear</i>, only when three checks agree, all in {T_CMP}: the
simple model is below every single seed of the LSTM (if it lies among them, one training of the LSTM could land on either side of it: a tie); the
LSTM is higher in most basins; and p is below {P_CUT:g}. p is how likely differences this lopsided are if the two models are equally good (Wilcoxon
signed-rank test over the {n} basins{", two-sided" if two_sided else ""}; the basins share storms, so p describes and does not prove). The later
sections use the same checks wherever their results files store what the checks need; where a file stores something else, the section says which
rule it uses.</p>"""

    W_NSE, W_N, W_P = 2.2, 2.0, 2.85
    sub = lambda em: _three(["1 d", "2 d", "3 d"], em)
    fams = {(inp, k): family(best(inp, k)) for inp in SETS for k in LEADS}
    major = max(sorted(set(fams.values())), key=list(fams.values()).count)
    odd = [f"the {f} for {_and(f'{SET_NAME[inp]} {_leads(ks)}' for inp in SETS for ks in [[k for k in LEADS if fams[inp, k] == f]] if ks)}"
           for f in sorted(set(fams.values()) - {major})]
    which = f"The simple model is the {major}" + (f", except {_and(odd)}." if odd else " in every cell.")
    grey = {inp: [LEADS.index(k) for k in unfair[inp]] for inp in SETS}
    grey_note = " Grey: the two do not read the same days of weather, no winner is called." if any_unfair else ""
    usable = cls["wq"]
    stronger = all(cmp_("wq", k)["lstm_nse_median"] > cmp_("w", k)["lstm_nse_median"] for k in LEADS)
    practice = "the stronger of the two forecasts that could be run in practice" if stronger else "one of the two forecasts that could be run in practice"
    groups = [("LSTM", lambda c: n2(c["lstm_nse_median"])), ("Simple model", lambda c: n2(c["baseline_nse_median"])),
              ("Seed spread", lambda c: n2(c["lstm_seed_median_max"] - c["lstm_seed_median_min"])),
              ("Basins", lambda c: str(c["n_basins_lstm_higher"])), ("p", lambda c: _p(c["wilcoxon_p"]))]
    td = lambda inp, i, v: f'<td class="{"g" if i == 0 else ""}{" nl" if i in grey[inp] else ""}">{v}</td>'
    body1 = "".join(f"<tr><td>{_cap(SET_NAME[inp])}{' &dagger;' if inp in UPPER else ''}</td>"
                    + "".join(td(inp, i, fmt(cmp_(inp, k))) for _, fmt in groups for i, k in enumerate(LEADS)) + "</tr>" for inp in SETS)
    w_first, w_cols = 33.5, {"LSTM": 4.3, "Simple model": 4.3, "Seed spread": 4.3, "Basins": 4.0, "p": 5.25}      # % of the table width
    table1 = (f'<table class="num t41"><colgroup><col style="width:{w_first}%">'
              + "".join(f'<col style="width:{w_cols[name]}%">' * len(LEADS) for name, _ in groups) + "</colgroup>"
              + '<tr><th rowspan="2">LSTM inputs</th>' + "".join(f'<th class="g grp" colspan="{len(LEADS)}">{name}</th>' for name, _ in groups) + "</tr>"
              + '<tr class="sub">' + "".join(f'<th class="{"g" if i == 0 else ""}">{k}&nbsp;d</th>' for _ in groups for i, k in enumerate(LEADS)) + "</tr>"
              + body1 + "</table>"
              + f'<p class="cap"><b>{T_CMP}. The LSTM against the best simple model given the same inputs.</b> Median NSE of the {n} basins, test years; '
                f"1 d, 2 d, 3 d: {_and(LEADS)} days ahead. Seed spread: highest minus lowest single seed of the LSTM. Basins: in how many of the {n} the "
                f"LSTM scores higher. &dagger; upper bound.{grey_note} {which} Rows 1 and 2 could be run in practice; look at row 2, "
                f"{practice.replace(' forecasts that could be run in practice', '')}.</p>")

    # the plain answer to milestone 2 of the plan, then one short paragraph per group of inputs
    pers = "base_persistence"
    m2_gauge, m2_weather = milestone2(cls, higher, n, n_seeds)                   # the same two phrases as on page 1
    why = (f"<li><b>With the river gauge</b>, a likely reason for the tie (not tested): {LEAD_WORD[first]} with a gauge, most of what these inputs hold "
           "is the river&rsquo;s own recent flow plus the recent rain, and a weighted sum already uses both.</li>") if usable[first] == "tie" else ""
    how_large = _how_large(R, {k: best("wq", k) for k in LEADS}, n) if usable[first] == "tie" else ""
    answer = (f'<div class="verdict blk"><b>Milestone 2 of the plan (&lsquo;the LSTM beats the baselines on unseen years in most basins, or we learn '
              f'why not&rsquo;): {m2_gauge}; {m2_weather}.</b> With {SET_NAME["wq"]}, {practice}, {_say(usable)} '
              f'(row 2 of {T_CMP}: {wins("wq")}).</div>')

    cw = {k: cmp_("w", k) for k in LEADS}
    by_seed = [v for k in LEADS for v in cw[k]["n_basins_lstm_higher_by_seed"]]
    thin = [k for k in LEADS if cw[k]["difference_of_medians"] < cw[k]["lstm_seed_median_max"] - cw[k]["lstm_seed_median_min"]]
    edge = (f" Its margin in median NSE ({_and(n2(cw[k]['difference_of_medians']) for k in LEADS)}) is "
            + ("smaller than" if len(thin) == len(LEADS) else "not always larger than")
            + f" its seed spread, and one seed alone is ahead in only {min(by_seed)} to {max(by_seed)} of {n} basins: for a single training, no "
              f"clear difference.") if any(v == "lstm_avg" for v in cls["w"].values()) else ""
    para_w = f"<li><b>Past weather only:</b> {_say(cls['w'])}.{edge}</li>"

    # the upper bound: a winner is called only at the lead times the results file marks as like-for-like
    fair = [k for k in LEADS if not any(k in unfair[inp] for inp in UPPER)]
    nse_of = lambda model, k: S[(S.model == model) & (S.lead == k)].set_index("gauge_id").nse
    diff = {k: nse_of("lstm_wf", k) - nse_of(best("wf", k), k) for k in LEADS}
    worst_k = min(fair or LEADS, key=lambda k: diff[k].min())
    behind = (f" (it falls far behind in a few basins, by up to {n2(-diff[worst_k].min())} NSE)"
              ) if all(cls["wf"][k] in ("tie", "lstm_med") for k in (fair or LEADS)) and diff[worst_k].min() < 0 else ""
    soften = lambda text: text.replace(PHRASE["lstm"], SOFT["lstm"]).replace(PHRASE["simple"], SOFT["simple"])
    said = lambda inp, ks: soften(_say({k: cls[inp][k] for k in ks})) if len(ks) > 1 else SOFT[cls[inp][ks[0]]]
    nums = lambda inp, k: (f"{n2(cmp_(inp, k)['lstm_nse_median'])} against {n2(cmp_(inp, k)['baseline_nse_median'])}, higher in "
                           f"{cmp_(inp, k)['n_basins_lstm_higher']} of {n} basins, p = {_p(cmp_(inp, k)['wilcoxon_p'])}"
                           + ("; the three checks agree" if cls[inp][k] in ("lstm", "simple") else ""))
    six = any(f"all {_word(len(FORCING))} weather variables" in x for x in A.get("notes", []))       # note of 04_lstm_all.py: not exactly the same variables
    not_same = (f" Even there the two do not read exactly the same: the LSTM reads all {_word(len(FORCING))} weather variables of those days, the simple "
                f"models rain and temperature only.") if six else ""
    worse = [g for g in ids if nse_of("lstm_wf", first)[g] < nse_of("lstm_w", first)[g]]                 # the weather of the coming days made the forecast worse
    lands_worse = {LAND_SHORT.get(A["basins"][g]["landscape"], A["basins"][g]["landscape"]) for g in worse}
    where_worse = (f"{_word(len(worse))} {next(iter(lands_worse)).lower().rstrip('s')} basin{'s' if len(worse) > 1 else ''}" if len(lands_worse) == 1
                   else f"{_word(len(worse))} basin{'s' if len(worse) > 1 else ''}")
    worse_txt = (f" In {where_worse} the perfect weather forecast even made the LSTM without a gauge worse than "
                 f"past weather alone {LEAD_WORD[first]} ({_and(A['basins'][g]['short'] + ': ' + n2(nse_of('lstm_wf', first)[g]) + ' against ' + n2(nse_of('lstm_w', first)[g]) for g in worse)}); "
                 f"the results do not show why.") if worse else ""
    if any_unfair and fair:
        para_f = (f"<li><b>With a perfect weather forecast</b> (upper bound) the two kinds of model read the same days of weather only "
                  f"{_leads(fair)}: the LSTM reads the weather stamped on all {_word(len(LEADS))} coming days at every lead time, the simple models only "
                  f"up to the day being forecast, and because the rain table seems to stamp rain by a day that ends in the morning ({TIMING_REF}), rain "
                  f"stamped one day later still tells about the day being forecast. So {_leads([k for k in LEADS if k not in fair])} no winner is called "
                  f"(grey in {T_CMP}). {_cap(_leads(fair))}: with the river gauge {said('wqf', fair)} ({'; '.join(nums('wqf', k) for k in fair)}); "
                  f"without the gauge {said('wf', fair)}.{not_same}{worse_txt}</li>")
    else:
        para_f = (f"<li class='blk'><b>With a perfect weather forecast</b> (upper bound): with the river gauge, {soften(_say(cls['wqf']))}. Without the "
                  f"gauge, {soften(_say(cls['wf']))}{behind}.{worse_txt}</li>")

    # ---- which rivers are easy: the per-basin table with text beside it (or, with SHOW_BASINS, the figure under the text) ----
    one = S[S.lead == first]
    nse1 = lambda m: one[one.model == m].set_index("gauge_id").nse
    L, P0, F0 = nse1("lstm_wq"), nse1(pers), nse1("base_rf_wq")
    short = lambda g: A["basins"][g]["short"]
    land = lambda g: LAND_SHORT.get(A["basins"][g]["landscape"], A["basins"][g]["landscape"])
    area = {g: A["basins"][g]["area_km2"] for g in ids}
    easy, hard = L.idxmax(), L.idxmin()
    ranges = {}
    for g in ids:
        lo, hi = ranges.get(land(g), (L[g], L[g]))
        ranges[land(g)] = (min(lo, L[g]), max(hi, L[g]))
    ranges = {name: ranges[name] for name in LAND_SHORT.values() if name in ranges}        # in the order of the report: mountains first
    names = list(ranges)
    overlap = any(ranges[a][0] <= ranges[b][1] and ranges[b][0] <= ranges[a][1] for i, a in enumerate(names) for b in names[i + 1:])
    top_l, top_p = set(L.nlargest(TOP).index), set(P0.nlargest(TOP).index)
    same = (f"The {_word(TOP)} easiest rivers are also the {_word(TOP)} where {QUOTED} scores highest" if top_l == top_p else
            f"Of the {_word(TOP)} easiest rivers, {_word(len(top_l & top_p))} are also among the {_word(TOP)} where {QUOTED} scores highest")
    big, small = sorted(ids, key=lambda g: -area[g])[:TOP], sorted(ids, key=lambda g: area[g])[:TOP]
    size = (f"Size sorts them better: each of the {_word(TOP)} largest basins scores higher than each of the {_word(TOP)} smallest "
            f"({count(min(area[g] for g in big))} km&sup2; and more against {count(max(area[g] for g in small))} km&sup2; and less)."
            if min(L[g] for g in big) > max(L[g] for g in small) else "Basin size does not sort them cleanly either.")
    cr = cmp_("wq", first, "base_rf_wq")
    cp = cmp_("wq", first, pers) if with_persistence else None
    chosen1 = best("wq", first)                                                    # the simple model of Table 4.1 for this case (chosen on the validation years)
    fitted = [name for name in C["wq"]["baselines"] if A["baselines"][name]["model"] in ("linreg", "rf")]     # the two fitted simple models, same inputs
    others1 = [name for name in fitted if name != chosen1]
    see = (f"the LSTM is above the {family(chosen1)}, the simple model of {T_CMP} for this case, in {cmp_('wq', first, chosen1)['n_basins_lstm_higher']} of "
           f"{n} basins"
           + "".join(f", above the {family(name)} in {cmp_('wq', first, name)['n_basins_lstm_higher']}" for name in others1)
           + (f" and above {QUOTED} in {cp['n_basins_lstm_higher']}" if cp else "")
           + f"; LSTM and random forest never differ by more than {n2((L - F0).abs().max())}")
    top_test = [name for name in others1 if nse1(name).median() > nse1(chosen1).median()]
    chosen_note = (f" The {family(chosen1)} is the simple model of {T_CMP} here: it had the highest score in the validation years"
                   + (f", though the {_and(family(x) for x in top_test)} scores higher in the test years." if top_test else "."))
    figure2 = fig("lstm_all_basins.png", f"<b>Figure {no2}. NSE {LEAD_WORD[first]} with the river gauge, basin by basin, test years.</b> Dot: LSTM "
                  f"(bar: its single seeds); square: random forest, same inputs; diamond: {QUOTED}. What to see: {see}.", FIG2_SHARE)
    # the same numbers as a table (the plan's 'per-basin score table'): a column per basin, by landscape and area as in the maps of section 6;
    # written out here because report_common.table() has no header cell that spans a group of columns
    lands = [name for name in LAND_SHORT.values() if name in ranges]
    order = [g for name in lands for g in sorted(ids, key=lambda g: area[g]) if land(g) == name]
    edge_ = {g: ' class="g"' if i == 0 or land(g) != land(order[i - 1]) else "" for i, g in enumerate(order)}        # a rule left of each landscape
    line = lambda label, cell: f"<tr><td>{label}</td>" + "".join(f"<td{edge_[g]}>{cell(g)}</td>" for g in order) + "</tr>"
    two_lines = lambda name: "&nbsp;".join(name.split(" ")[:-1]) + " " + name.split(" ")[-1]      # 'NE Cape Fear' breaks before its last word only
    basin_table = (
        f'<table class="num tight basins"><colgroup><col style="width:16%">{"<col>" * len(order)}</colgroup><tr><th></th>'
        + "".join(f'<th class="g" colspan="{sum(land(g) == name for g in order)}">{name}</th>' for name in lands)
        + "</tr><tr><th></th>" + "".join(f"<th{edge_[g]}>{two_lines(short(g).split(',')[0])}</th>" for g in order) + "</tr>"
        + line("Area, km&sup2;", lambda g: count(area[g]))
        + (line(QUOTED.replace("&lsquo;t", "&lsquo;T"), lambda g: n2(P0[g])) if cp else "")
        + "".join(line(_cap(family(name)), lambda g, v=nse1(name): n2(v[g])) for name in fitted)
        + line("<b>LSTM</b>", lambda g: f"<b>{n2(L[g])}</b>") + "</table>"
        + f'<p class="cap"><b>{T_BASIN}. NSE {LEAD_WORD[first]} with the river gauge, basin by basin, test years.</b> '
        + f"{_cap(_and(family(name) for name in fitted))} and LSTM read the same inputs"
        + (f"; {QUOTED} reads the gauge only" if cp else "") + f".{chosen_note} LSTM: average forecast of its {_word(n_seeds)} seeds. In each landscape "
        f"the basins are in order of area. What to see: {see}.</p>")
    by_land = _and(f"{name.lower()} {n2(lo)} to {n2(hi)}" for name, (lo, hi) in ranges.items())
    sorts = (f"Landscape alone does not sort them, the ranges overlap{'' if not SHOW_BASINS else f': {by_land}'}." if overlap else
             f"Landscape sorts them: {by_land}.")
    rivers = f"""
<p><b>Which rivers are easy, which are hard?</b> {LEAD_WORD[first].capitalize()} with the river gauge, the LSTM&rsquo;s NSE runs from {n2(L[hard])}
({short(hard)}) to {n2(L[easy])} ({short(easy)}; {T_BASIN}). {sorts} {size} {same}: a river that changes little from one day to the next is easy
for every model that reads the gauge.</p>"""

    # French Broad River: the all-river model against the one-river model of section 3 (lstm_single.json is rerun separately: guarded)
    try:
        One = R["lstm_single"]
        g = One["basin"]["gauge_id"]
        river = One["basin"]["name"].split(" AT ")[0].title() if " AT " in One["basin"]["name"] else short(g)
        f1 = One["finals"]["wq"]
        one_avg = {k: f1["seed_average"]["test"][lk(k)]["nse"] for k in LEADS}
        one_seed = {k: [float(v) for v in f1["test_nse_by_seed"][lk(k)]] for k in LEADS}
        all_avg = {k: A["sets"]["wq"]["test"]["basins"][g][lk(k)]["nse"] for k in LEADS}
        all_seed = {k: [p["test_nse_by_basin"][g][LEADS.index(k)] for p in A["sets"]["wq"]["per_seed"].values()] for k in LEADS}
        if min(len(one_seed[first]), len(all_seed[first])) < 2:
            raise ValueError("fewer than two seeds: no seed spread to judge the difference by")
        verdict = {k: "helped" if min(all_seed[k]) > max(one_seed[k]) else "hurt" if max(all_seed[k]) < min(one_seed[k]) else "tie" for k in LEADS}
        words = {"helped": "all its seeds are above those of the one-river model", "hurt": "all its seeds are below those of the one-river model",
                 "tie": "the seeds of the two models overlap"}
        groups = {}
        for k in LEADS:
            groups.setdefault(verdict[k], []).append(k)
        judged = "; ".join(f"{_leads(ks)} {words[v]}" for v, ks in groups.items()) if len(groups) > 1 else f"at all three lead times {words[next(iter(groups))]}"
        b = best("wq", first)
        b_nse = float(nse1(b)[g])
        even = "Even so, " if verdict[first] == "helped" else ""
        still = (f" {_cap(even + LEAD_WORD[first])} it stays below the {family(b)} fitted to this river alone ({n2(b_nse)})." if all_avg[first] < b_nse
                 else f" {LEAD_WORD[first].capitalize()} it also passes the {family(b)} fitted to this river alone ({n2(b_nse)}).")
        got = set(verdict.values())
        lead_in = ("Yes." if got == {"helped"} else "No, it did worse." if got == {"hurt"} else "Not clearly." if got == {"tie"} else
                   f"Yes, {_leads(groups['helped'])}." if "hurt" not in got else "The answer differs with the lead time.")
        french = (f"<p><b>Did more rivers help the river of section 3?</b> {lead_in} With the river gauge, the all-river model scores NSE "
                  f"{_and(n2(all_avg[k]) for k in LEADS)} on the {river}; the one-river model scored "
                  f"{_and(n2(one_avg[k]) for k in LEADS)}. {_cap(judged)}.{still}</p>")
    except (KeyError, TypeError, IndexError, ValueError) as e:
        print(f"s4_all: the one-river comparison was left out ({type(e).__name__}: {e})", file=sys.stderr)
        french = ""

    # ---- 4.3 where the skill comes from -------------------------------------------------------------------------------
    gauge, fc = ST["river_gauge_without_forecast"], ST["perfect_forecast_with_gauge"]
    n_on = {k: LV["wq"][lk(k)]["onset_hits"] + LV["wq"][lk(k)]["onset_misses"] for k in LEADS}
    W_NSE, W_PCT = 3.0, 2.6
    level = lambda inp, name: [name, _three([n2(LV[inp][lk(k)]["nse_median"]) for k in LEADS], W_NSE),
                               _three([pct(LV[inp][lk(k)]["onset_pod"]) for k in LEADS], W_PCT),
                               _three([pct(LV[inp][lk(k)]["onset_far"]) for k in LEADS], W_PCT)]
    step = lambda s, name: [f"<b>{name}</b>", _three([signed(s[lk(k)]["nse_median_gain"]) for k in LEADS], W_NSE, True),
                            _three([_pts(s[lk(k)]["onset_pod_gain"]) for k in LEADS], W_PCT, True),
                            _three([_pts(s[lk(k)]["onset_far_change"]) for k in LEADS], W_PCT, True)]
    all_clear = all(s[lk(k)]["clear_of_seed_noise"] for s in (gauge, fc) for k in LEADS)
    gains = [s[lk(k)] for s in (gauge, fc) for k in LEADS]                         # the NSE gains of the two bold rows
    has_p = all(isinstance(g_.get("wilcoxon_p"), float) for g_ in gains)
    by_rule = all_clear and all(g_["n_basins_better"] > n / 2 for g_ in gains) and (not has_p or all(g_["wilcoxon_p"] < P_CUT for g_ in gains))
    if by_rule:
        checks = (f" All {_word(len(gains))} NSE gains are clear by the checks of section 4.2: the single seeds of the two models do not overlap, the NSE "
                  f"is higher in {min(g_['n_basins_better'] for g_ in gains)} to {max(g_['n_basins_better'] for g_ in gains)} of {n} basins"
                  + (f" and p is at most {_p(max(g_['wilcoxon_p'] for g_ in gains))}." if has_p else " (no p is stored for them)."))
    elif all_clear:
        checks = f" In all {_word(len(gains))} NSE gains the single seeds of the two models do not overlap."
    else:
        checks = ""
    try:                                                                           # the same number for two different counts: say so
        t_all = A["sets"]["wq"]["test"]["summary"][lk(first)]["all"]
        not_onsets = t_all["hits"] + t_all["misses"] - n_on[first]
        twin = [k for k in LEADS[1:] if n_on[k] == not_onsets]
        chance = (f" ({count(n_on[twin[0]])}, the onsets {LEAD_WORD[twin[0]]}, is by chance also the number of high-flow days that are not onsets "
                  f"{LEAD_WORD[first]}.)") if twin else ""
    except (KeyError, TypeError):
        chance = ""
    table2 = table(["What the LSTM is given", f"Median NSE<br>{sub(W_NSE)}", f"Onsets warned<br>{sub(W_PCT)}", f"False warnings<br>{sub(W_PCT)}"],
                   [level("w", "1 &nbsp;Past weather"), level("wq", "2 &nbsp;Past weather + river gauge"),
                    level("wqf", "3 &nbsp;Past weather + river gauge + perfect weather forecast &dagger;"),
                    step(gauge, "The river gauge adds (row 2 minus row 1)"), step(fc, "A perfect weather forecast would add (row 3 minus row 2)")],
                   cls="num tight", widths=["51%", "17.5%", "15.75%", "15.75%"],
                   caption=f"<b>{T_SKILL}. Where the skill comes from.</b> Test years; cells as in {T_CMP}. Onsets warned: share of the "
                           f"{_and(count(n_on[k]) for k in LEADS)} onsets ({n} basins together).{chance} False warnings: share of the onset warnings "
                           f"not followed by a high-flow day. Bold rows: differences, in NSE and in percentage points, taken before rounding. "
                           f"&dagger; upper bound.{checks}")

    on_lo = min(gauge[lk(k)]["onset_pod_gain_seed_min"] for k in LEADS)
    on_hi = max(gauge[lk(k)]["onset_pod_gain_seed_max"] for k in LEADS)
    on_word = "no clear change" if on_lo < 0 < on_hi else "a gain" if on_lo >= 0 else "a loss"
    far_down = all(gauge[lk(k)]["onset_far_change"] < 0 for k in LEADS)
    g1, g3 = gauge[lk(first)], gauge[lk(last)]
    if min(gauge[lk(k)]["nse_median_gain"] for k in LEADS) <= 0:
        adds = "does not add accuracy at every lead time"
    elif g3["nse_median_gain"] < 0.5 * g1["nse_median_gain"]:
        adds = "adds accuracy for the next day and little after that"
    else:
        adds = "adds accuracy at every lead time"
    g_text = (f"<li class='blk'><b>The river gauge</b> {adds}: NSE {signed(g1['nse_median_gain'])} {LEAD_WORD[first]} (seed by seed "
              f"{signed(g1['nse_median_gain_seed_min'])} to {signed(g1['nse_median_gain_seed_max'])}; higher in {g1['n_basins_better']} of {n} "
              f"basins), {signed(g3['nse_median_gain'])} {LEAD_WORD[last]}. In onsets warned it brings {on_word} (seed by seed {_pts(on_lo)} to "
              f"{_pts(on_hi)} points)"
              + (f"; it does cut the false warnings, {LEAD_WORD[first]} from {pct(g1['onset_far_without'])} to {pct(g1['onset_far_with'])}.</li>"
                 if far_down else "; the share of false warnings does not fall at every lead time.</li>"))
    biggest = max(abs(cmp_(inp, k)["difference_of_medians"]) for inp in SETS for k in LEADS if k not in unfair[inp])       # LSTM against simple model, Table 4.1
    miss = {k: 1 - LV["wq"][lk(k)]["onset_pod"] for k in LEADS}
    larger = min(fc[lk(k)]["nse_median_gain"] for k in LEADS) > biggest
    ratio = min(fc[lk(k)]["nse_median_gain"] / gauge[lk(k)]["nse_median_gain"] if gauge[lk(k)]["nse_median_gain"] > 0 else float("inf") for k in LEADS)
    more = "adds far more" if ratio >= 1.5 else "adds more" if ratio > 1 else "does not add more at every lead time"
    f_text = (f"<li class='blk'><b>The weather of the days being forecast</b> {more} on top of the gauge (NSE higher in "
              f"{_and(fc[lk(k)]['n_basins_better'] for k in LEADS)} of {n} basins). Without it, the LSTM with the river gauge misses "
              f"{pct(miss[first])} of the onsets {LEAD_WORD[first]} and {pct(miss[last])} {LEAD_WORD[last]}.</li>")
    # the two halves of the evidence: (1) models given the same inputs lie close together; (2) every model that was given more information gained
    try:
        M = R["model_comparison"]
        apart = None if M.get("quick_mode") else max(abs(M["comparison"]["test"][lk(k)]["nse_median_transformer_minus_lstm"]) for k in LEADS)
    except (KeyError, TypeError):
        apart = None
    try:
        SB = {(r["model"], r["lead"]): r["nse_median"] for r in R["baselines"]["summary_table"]}
        simple_up = all(SB[f"base_{m}_{b}", k] > SB[f"base_{m}_{a}", k] for m in ("linreg", "rf") for a, b in (("w", "wq"), ("wq", "wqf")) for k in LEADS)
        rf_gap = max(abs(SB[f"base_linreg_{x}", k] - SB[f"base_rf_{x}", k]) for x in ("w", "wq") for k in LEADS)      # ridge against forest, same inputs
    except (KeyError, TypeError):
        simple_up, rf_gap = False, None
    lstm_up = all(s_[lk(k)]["nse_median_gain"] > 0 for s_ in (gauge, fc) for k in LEADS)
    practice_gap = max(abs(cmp_(inp, k)["difference_of_medians"]) for inp in ("w", "wq") for k in LEADS)             # LSTM against the best simple model
    near = max(v for v in (rf_gap, practice_gap, apart, biggest) if v is not None)
    larger = min(fc[lk(k)]["nse_median_gain"] for k in LEADS) > near
    n_kinds = 2 + (rf_gap is not None) + (apart is not None)
    key_fig = " The key figure on page 1 shows both at a glance." if (FIG / "key_figure.png").exists() else ""       # page 1 shows it under the same condition
    limit = (f"<li><b>Among the models tried, the limit is the information in the inputs, not the network.</b> With the inputs that could be had "
             f"in practice, {_word(n_kinds)} kinds of model lie close together: "
             + (f"ridge regression and random forest never differ by more than {n2(rf_gap)} NSE (section 2), " if rf_gap is not None else "")
             + f"the LSTM and the best simple model by at most {n2(practice_gap)} ({T_CMP})"
             + (f", the Transformer and its LSTM by at most {n2(apart)} (section 7)" if apart is not None else "")
             + f". Each NSE gain from the weather of the days being forecast is larger than that"
             + (", and the river gauge and that weather each lift the two simple models as well as the LSTM, at every lead time (Tables 2.1 and 2.2)."
                if simple_up and lstm_up else ".")
             + f" Not tried: an LSTM with more than {max(hidden)} memory cells, another loss"
             + (", a search over the Transformer&rsquo;s size" if apart is not None else "") + f".{key_fig}</li>") if larger else ""
    published = _published(R, A, n, T_SKILL, T_CMP, all(v == "tie" for v in usable.values()))
    top = LV["wqf"][lk(first)]
    tf = timing_facts(R)                                                          # the rain-timing test, once it has run
    timing_bit = ""
    if tf and onset_clear(tf["gauge"][first]):
        r_ = tf["gauge"][first]
        nse_bit = {"clear": "the median NSE rises clearly too", "mixed": f"in median NSE the result is mixed (higher in only {r_['basins_better']} of {tf['n']} basins)",
                   "none": "in median NSE there is no clear difference"}[nse_reading(r_, tf["n"])]
        timing_bit = (f": rain stamped up to the evening of the issue day lifts the onsets warned from {pct(r_['onset_warned_base'])} to "
                      f"{pct(r_['onset_warned_with_nldas'])}, {share_word(r_['onset_share_of_gap_closed'])} of the gap to row 3; {nse_bit}")
    tested = forecast_where(R, short=True)                                            # section 4.8 has tested one real forecast: say where it lands (None: the old sentence)
    where_lands = f"; {tested}" if tested else ", and the results do not say where"
    skill = f"""
<h3>4.3 Where the skill comes from{": the inputs, not the network" if limit else ""}</h3>
<p>The {_word(len(SETS))} LSTMs differ only in their inputs, so the difference between two of them is what one source of information
adds{f" (in Figure {no1}: the distance from the dashed line to the blue line)" if SHOW_PANELS else ""}. Warnings are scored on the onsets.</p>
<div class="blk">{table2}</div>
<ul>{g_text}{f_text}{limit}</ul>
<div class="note blk"><b>How to read row 3.</b> It is an upper bound: a real weather forecast has errors, so a real system would land between
rows 2 and 3{where_lands}. Nor is it a perfect warning: {LEAD_WORD[first]} it still misses {pct(1 - top['onset_pod'])} of
the onsets, and {pct(top['onset_far'])} of its onset warnings are false. And not all of the gap between rows 2 and 3 needs a weather forecast:
{LEAD_WORD[first]}, part of it is rain that had already fallen when the forecast was made but was not yet in the model&rsquo;s
inputs{timing_bit} ({TIMING_REF}).</div>
{published}"""

    block1 = method                                                              # then the figure (a float), the table it goes with and the milestone box
    block2 = f"{rivers}{french}{figure2}" if SHOW_BASINS else f'{rivers}<div class="blk">{basin_table}</div>{french}'
    return f"""{LAYOUT}{TIGHT}<h2>4 One LSTM for all twelve rivers</h2>{changes}
<h3>4.2 Does it beat the simple models?{" With the river gauge: no clear difference" if all(v == "tie" for v in usable.values()) else ""}</h3>
{block1}
{floated(figure1) if SHOW_PANELS else ''}
<div class="blk">{table1}</div>
{answer}
<ul>{why}{how_large}{para_w}{para_f}</ul>
{block2}{skill}"""
