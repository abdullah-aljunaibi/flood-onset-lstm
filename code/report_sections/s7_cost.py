"""Report section 7 - Transformer or LSTM, and what training costs.

Reads results/model_comparison.json (written by 09_model_comparison.py) and, for the cost of the whole project only, the
timing fields of the other results files. Nothing is computed here beyond differences, ratios, sums, smallest/largest and
formatting; every sentence whose wording depends on an outcome is chosen from the stored numbers.

The settings of both networks (layers, heads, width, weights, stretches, learning rate, updates) stand in the table 'The three networks at a
glance' of section 4.1 (s4_all.networks_table); this section points to it and lists the settings itself only when that table cannot be built.
The heading states the result when the Transformer is not clearly better and dearer to train, and is the plain heading otherwise.
"""
import sys

from report_common import LEAD_WORD, cite, count, fig, n2, pct, table
from report_sections.s0_summary import LAYOUT, floated
from report_sections.s4_all import networks_table

LEADS = sorted(LEAD_WORD)
MODELS = ("lstm", "transformer")
NAME = {"lstm": "LSTM", "transformer": "Transformer"}
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
SHOW_FIGURE = True                              # Figure 7.1 (model_comparison.png), drawn for a width of 170 mm: shown at about that width or left out
FIG_SHARE = 84                                  # % of the text width (184 mm) taken by Figure 7.1
SHOW_SCORE_TABLE = False                        # True adds the table of scores (NSE, seed spread, warnings); the figure and the text carry its numbers
SHOW_COST_TABLE = False                         # True adds the table of costs (weights, multiply-adds, seconds, energy); the figure shows the first four

# the runs whose computing time is added up for 'what the whole project cost': results file -> what it is called in the text.
# A file that does not exist yet (its experiment is still running) is left out, and the text lists only the runs it found.
RUNS = (("baselines", "the simple models of section 2"), ("lstm_single", "the one-river LSTM of section 3"),
        ("lstm_all", "the all-river LSTM of section 4"), ("lstm_lobo", "the unseen-river test of section 4.4"),
        ("rain_timing", "the rain-timing test of section 4.5"), ("rain_to_midnight", "the rain-to-midnight test with ERA5 of section 4.6"),
        ("aorc_rain_to_midnight", "the rain-to-midnight test with AORC of section 4.6"), ("more_basins", "the 63-river test of section 4.7"),
        ("forecast_input", "the weather-forecast test of section 4.8"), ("satellite_inputs", "the satellite inputs of section 6.2"),
        ("model_comparison", "this comparison"))
NOT_NETWORKS = ("baselines",)                   # runs that fit no neural network: they add time but no 'trainings'
SECONDS_KEYS = ("training_seconds", "training_seconds_total", "train_seconds", "fit_seconds", "runtime_s", "runtime_seconds", "seconds")
COUNT_KEYS = ("n_trainings", "n_training_runs", "n_runs")


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


def _three(vals, em):
    """Three formatted values (1, 2, 3 days ahead) side by side in one table cell, as in Tables 2.1 and 4.1."""
    cells = "".join(f'<span style="display:inline-block;width:{em}em;text-align:right">{v}</span>' for v in vals)
    return f'<span style="white-space:nowrap">{cells}</span>'


def _times(x):
    """A ratio Transformer / LSTM: 3.18 -> '3.2&times;'."""
    return f"{x:.1f}&times;"


def _formula(text):
    """The formula part of a 'formulas' entry of the results file (up to the colon), with real multiplication signs."""
    return text.split(":")[0].replace(" x ", " &times; ").replace("^2", "<sup>2</sup>")


def _first(d, keys):
    """The first of `keys` that the results file has as a number, or None."""
    return next((d[k] for k in keys if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)), None)


def _project_cost(R, T):
    """Computing time of the runs whose results files exist: [(file, label, seconds, trainings or None)]."""
    found = []
    for name, label in RUNS:
        if name not in R or not isinstance(R[name], dict) or R[name].get("quick_mode"):
            continue
        secs = _first(R[name], SECONDS_KEYS)
        if secs is None:
            print(f"s7_cost: {name}.json has no timing field among {SECONDS_KEYS}; left out of the project total", file=sys.stderr)
            continue
        n = _first(R[name], COUNT_KEYS)
        if n is None and name == "lstm_single" and "hparam_log" in T and R[name].get("hparam_log"):
            n = len(T["hparam_log"])                                             # 03_lstm.py logs one row per training
        found.append((name, label, float(secs), None if n is None else int(n)))
    return found


def html(R, T):
    X = R["model_comparison"]
    M, C, K, S = X["models"], X["comparison"], X["cost_of_skill"], X["settings"]
    ratio, shared, seeds = K["transformer_over_lstm"], S["shared"], [str(s) for s in X["seeds"]]
    lk = lambda k: f"l{k}"
    first = LEADS[0]
    n_seeds, days, watt = len(seeds), S["sequence_days"], S["assumed_power_watt"]
    n_basins = C["test"][lk(first)]["n_basins"]
    L, TF = M["lstm"], M["transformer"]

    # ---- the two models; what was kept identical ---------------------------------------------------------------------------
    checked = [len(X[f"{m}_check"]["candidates"]) for m in MODELS]
    n_checked = f"{_word(checked[0])} values each" if len(set(checked)) == 1 else f"{_and(checked)} values"
    if networks_table(R, T, quiet=True):                                         # section 4.1 shows the settings of both networks in its table
        identical = (f"the {n_basins} basins, the inputs ({X['input_label']} plus {_word(len(X['static_inputs']))} fixed basin properties), the three "
                     f"periods, the loss, the scoring and the stopping rule. The table in section 4.1 gives the settings of both networks.")
    else:
        identical = (f"the {n_basins} basins, the inputs ({X['input_label']} plus {_word(len(X['static_inputs']))} fixed basin properties, "
                     f"{L['n_inputs']} numbers per day), the three periods, the loss (the error that training reduces), the scoring, {shared['hidden']} "
                     f"units and the stopping rule (a check on the validation years every {shared['eval_every']} updates; stop after "
                     f"{shared['patience']} checks without improvement or at {count(shared['updates'])} updates).")
    method = f"""
<p>Would a Transformer forecast these rivers better than the LSTM, and what does each cost to train? The LSTM carries a memory from each day
to the next. The Transformer {cite('vaswani2017')} keeps no memory: for each forecast it looks back at all earlier days of the sequence at once
and weighs them, which is called attention; a mask hides the later days so that it cannot look ahead.</p>
<p><b>What was kept identical.</b> Everything except the network: {identical} Each model was trained from {_word(n_seeds)}
seeds and had only its learning rate checked ({n_checked}, on the validation years). With {_word(n_seeds)} seeds per model and no p-value stored, the
rule of section 4.2 cannot be applied in full: here one model is called ahead only if every seed of it is above every seed of the other and the gap
exceeds the seed spread; the count of basins is given beside it.</p>"""

    # ---- beside the figure: why the context is one year, and the proof that neither model looks ahead ----------------------
    ref = S.get("lstm_all_config_for_reference")
    ref_days = ref["warmup"] + ref["block"] if ref else None
    att_share = TF["operations"]["parts_per_sequence"]["attention_day_by_day"] / TF["operations"]["per_sequence_forward"]
    if ref_days and ref_days != days:
        longer = (f" The stretches of section 4 ({ref['warmup']} + {ref['block']} days) would need about "
                  f"{_word(round((ref_days / days) ** 2))} times that work, so both models read {days} days here ({shared['warmup']} + {shared['block']}): "
                  f"this LSTM is not the model of section 4.")
    else:
        longer = f" Both models read {days} days."
    cz = X["causality"]
    nets = [cz[m]["networks_tested"] for m in MODELS]
    per_model = f"{nets[0]} per model" if len(set(nets)) == 1 else f"{_and(nets)} per model"
    leak = max(cz[m]["largest_change_up_to_day_t"] for m in MODELS)
    unchanged = "did not change at all" if leak == 0 else f"changed by less than {leak:.0e}".replace("e-0", "e-").replace("-", "&minus;")
    control = cz["transformer"]["largest_change_up_to_day_t_without_mask"]
    beside = f"""
<p><b>Why one year of context.</b> Attention compares every day with every other day, so its work grows with the square of the sequence length;
at {days} days it is already {pct(att_share)} of the Transformer&rsquo;s operations.{longer} <b>Does the mask hold?</b> In all {sum(nets)} trained
networks ({per_model}) the inputs after a given day were replaced by random numbers: the forecasts issued up to that day {unchanged}. With the
mask taken off they moved (by {n2(control)}, in scaled flow), so the test can see a leak.</p>"""

    # ---- Figure 7.1 (its caption runs under the figure and the text beside it) --------------------------------------------
    shown = [ratio[k] for k in ("parameters", "operations_per_update", "seconds_per_update", "seconds_per_model")]
    apart = max(abs(C["test"][lk(k)]["nse_median_transformer_minus_lstm"]) for k in LEADS)
    bars = (f"every cost bar is {min(shown):.0f} to {max(shown):.0f} times as long for the Transformer" if min(shown) > 1 else
            f"the Transformer&rsquo;s cost bars are {min(shown):.1f} to {max(shown):.1f} times as long as the LSTM&rsquo;s")
    caption1 = (f"<b>Figure 7.1. Transformer against LSTM.</b> Left: median NSE of the {n_basins} basins, test years (dots: average forecast of "
                f"the {_word(n_seeds)} seeds; short bars: single seeds). Right: the cost of training one model (in brackets: Transformer "
                f"divided by LSTM). The two lines are never more than {n2(apart)} NSE apart, while {bars}.")
    figure = floated(fig("model_comparison.png", caption1, FIG_SHARE)) if SHOW_FIGURE else ""

    # ---- Table 7.1: the scores in the test years ---------------------------------------------------------------------------
    W_NSE, W_PCT = 2.9, 2.1
    sub = lambda em: _three(["1 d", "2 d", "3 d"], em)
    by_seed = lambda part, k, m: C[part][lk(k)][m]["nse_median_by_seed"]
    spread = lambda part, k, m: max(by_seed(part, k, m)) - min(by_seed(part, k, m))
    share = lambda m, key: _three([pct(C["test"][lk(k)][m][key]).replace("%", "") for k in LEADS], W_PCT)
    score_row = lambda m: [NAME[m], _three([n2(C["test"][lk(k)][m]["nse_median"], 3) for k in LEADS], W_NSE),
                           _three([n2(spread("test", k, m), 3) for k in LEADS], W_NSE),
                           share(m, "all_pod"), share(m, "all_far"), share(m, "onset_pod"), share(m, "onset_far")]
    score_table = table(["Test years", f"NSE<br>{sub(W_NSE)}", f"Seed spread<br>{sub(W_NSE)}", f"Share warned, %<br>{sub(W_PCT)}",
                         f"False warnings, %<br>{sub(W_PCT)}", f"Onsets warned, %<br>{sub(W_PCT)}", f"False onset warnings, %<br>{sub(W_PCT)}"],
                        [score_row(m) for m in MODELS], widths=["12%", "17%", "17%", "13.5%", "13.5%", "13.5%", "13.5%"],
                        caption=f"<b>Table 7.1. The scores, test years.</b> Three numbers in a cell: 1, 2 and 3 days ahead. NSE: median of the "
                                f"{n_basins} basins for the average forecast of the {_word(n_seeds)} seeds, with three decimals because the differences "
                                f"are about 0.01; seed spread: highest minus lowest single seed. Warning scores as in section 2, the {n_basins} basins "
                                f"together.")

    # ---- Table 7.2: what training costs -------------------------------------------------------------------------------------
    cost = lambda m, key: M[m]["cost"][key]
    cost_row = lambda m: [NAME[m], count(M[m]["n_params"]), count(M[m]["operations"]["per_update_forward"] / 1e6),
                          n2(cost(m, "s_per_update_median"), 3), count(cost(m, "best_update_mean")), count(cost(m, "seconds_per_model")),
                          n2(cost(m, "energy_wh_per_model_assumed"))]
    e_ratio = cost("transformer", "energy_wh_per_model_assumed") / cost("lstm", "energy_wh_per_model_assumed")
    ratios = ["<b>Ratio</b>"] + [f"<b>{_times(x)}</b>" for x in (
        ratio["parameters"], ratio["operations_per_update"], ratio["seconds_per_update"], ratio["updates_to_best"], ratio["seconds_per_model"], e_ratio)]
    F = X["formulas"]
    cost_table = table(["One model", "Weights to learn", "Multiply-adds per update, millions", "Seconds per update",
                        "Updates to the best validation score", "Training time, s", f"Energy, Wh (assumed&nbsp;{watt:g}&nbsp;W)"],
                       [cost_row("lstm"), cost_row("transformer"), ratios], widths=["12%", "13%", "19%", "13%", "19%", "10%", "14%"],
                       caption=f"<b>Table 7.2. The cost of training one model.</b> Mean of the {_word(n_seeds)} seeds; ratio: Transformer "
                               f"divided by LSTM; times are measured. Multiply-adds are counted, not measured, for the forward pass of one "
                               f"update: LSTM {_formula(F['lstm_forward'])}; Transformer "
                               f"{_formula(F['transformer_forward'])}, with B = {shared['batch']} sequences, T = {days} days, "
                               f"n = {L['n_inputs']} inputs, h = d = {shared['hidden']} units, L = {TF['settings']['layers']} layers, "
                               f"f = {TF['settings']['feed_forward']}. Energy: training seconds &times; an assumed {watt:g} W.")

    # ---- the plain answer -------------------------------------------------------------------------------------------------
    v, t = C["val"]["skill"], C["test"]["skill"]
    ahead = lambda s: "Transformer" if s["transformer_minus_lstm"] > 0 else "LSTM"
    clear = lambda s: (not s["seed_ranges_overlap"]) and abs(s["transformer_minus_lstm"]) > s["largest_seed_spread"]
    tf_better = clear(t) and t["transformer_minus_lstm"] > 0 and v["transformer_minus_lstm"] > 0
    costlier = ratio["seconds_per_model"] > 1
    r_low, r_high = sorted((ratio["seconds_per_model"], ratio["seconds_per_update"]))                  # per model and per update: the timings are noisy
    price = (f"roughly {r_low:.0f} to {r_high:.0f} times the training time" if costlier else f"{pct(ratio['seconds_per_model'])} of the training time")
    both = f"NSE {n2(t['lstm'], 3)} (LSTM) against {n2(t['transformer'], 3)} (Transformer), the mean of the three lead times"
    two_seeds = f"{_word(n_seeds)} seeds of one model differ by up to {n2(t['largest_seed_spread'], 3)}"
    if clear(t):
        test_says = f"In the test years the {ahead(t)} is more accurate: {both}; {two_seeds}."
    else:
        test_says = f"In the test years there is no clear difference in accuracy: {both}; {two_seeds}."
    # the validation years: the median-based lead can rest on one basin (keys added by the reviewer of 09_model_comparison.py)
    k_big = max(LEADS, key=lambda k: abs(C["val"][lk(k)]["nse_median_transformer_minus_lstm"]))
    one, v_mean = C["val"][lk(k_big)].get("largest_basin_difference"), v.get("mean_nse_over_basins")
    one_basin = bool(one) and abs(one["nse_median_transformer_minus_lstm_without_it"]) < 0.5 * abs(C["val"][lk(k_big)]["nse_median_transformer_minus_lstm"])
    val_says = (f"On the validation years, where the choice has to be made, the {ahead(v)}&rsquo;s median-based score is higher by "
                f"{n2(abs(v['transformer_minus_lstm']), 3)}, "
                + (f"more than the seed spread of {n2(v['largest_seed_spread'], 3)}" if clear(v) else
                   f"which is within the seed spread of {n2(v['largest_seed_spread'], 3)}" if abs(v["transformer_minus_lstm"]) <= v["largest_seed_spread"]
                   else f"but the single seeds of the two models overlap (seed spread {n2(v['largest_seed_spread'], 3)})")
                + (f"; but most of that is one basin {LEAD_WORD[k_big]} ({one['basin']}: {n2(one['lstm'])} against {n2(one['transformer'])}; without it the "
                   f"median difference there is {n2(abs(one['nse_median_transformer_minus_lstm_without_it']), 3)} instead of "
                   f"{n2(abs(C['val'][lk(k_big)]['nse_median_transformer_minus_lstm']), 3)})" if one_basin else "")
                + (f", and in the mean over the basins the two differ by {n2(abs(v_mean['transformer_minus_lstm']), 3)}." if v_mean else "."))
    if tf_better and costlier:
        head, cost_says = "It depends.", f"It needs {price}; whether that is worth it depends on how often the model must be trained again."
    elif tf_better:
        head, cost_says = "Yes.", f"It also needs only {price}."
    elif costlier:
        head, cost_says = "No.", f"For this set-up the Transformer is not better, and it needs {price}."
    else:
        head, cost_says = "Not for its accuracy.", f"The Transformer is not more accurate, but it needs only {price}."
    answer = (f'<div class="verdict blk"><b>Is the Transformer worth it for this data size? {head}</b> {test_says} {val_says} {cost_says}'
              + (" (Figure 7.1)" if SHOW_FIGURE else "") + "</div>")

    # accuracy lead by lead: a model is called ahead only if every seed of it is above every seed of the other AND the gap exceeds the seed spread
    lead = {}
    for k in LEADS:
        row = C["test"][lk(k)]
        d, sp = row["nse_median_transformer_minus_lstm"], max(spread("test", k, m) for m in MODELS)
        kind = "overlap" if row["seed_ranges_overlap"] else "win" if abs(d) > sp else "edge"
        lead[k] = dict(d=d, spread=sp, kind=kind, who="Transformer" if d > 0 else "LSTM", n_lstm=row["n_basins_lstm_higher"])
    of_kind = lambda kind: [k for k in LEADS if lead[k]["kind"] == kind]
    mean_d = lambda k: C["test"][lk(k)].get("nse_mean_transformer_minus_lstm")
    opposite = lambda k: mean_d(k) is not None and mean_d(k) * lead[k]["d"] < 0                       # the mean over the basins points the other way
    n_who = lambda k: lead[k]["n_lstm"] if lead[k]["who"] == "LSTM" else n_basins - lead[k]["n_lstm"]      # basins in which the model with the higher median scores higher
    says = [f"{LEAD_WORD[k].capitalize()} the {lead[k]['who']}&rsquo;s median is higher by {n2(abs(lead[k]['d']), 3)} NSE, more than the seed spread "
            f"there ({n2(lead[k]['spread'], 3)}), "
            + (f"and it scores higher in {n_who(k)} of {n_basins} basins" if n_who(k) > n_basins / 2 else f"but it scores higher in only {n_who(k)} of {n_basins} basins")
            + (f"; but the mean over the basins points the other way ({n2(abs(mean_d(k)), 3)})." if opposite(k) else ".") for k in of_kind("win")]
    says += [f"{LEAD_WORD[k].capitalize()} the {lead[k]['who']}&rsquo;s margin of {n2(abs(lead[k]['d']), 3)} is no larger than the seed spread "
             f"({n2(lead[k]['spread'], 3)})." for k in of_kind("edge")]
    if of_kind("overlap"):
        says.append(f"{_leads(of_kind('overlap')).capitalize()} the single seeds of the two models overlap.")
    if not of_kind("win"):
        says.insert(0, "At none of the three lead times is one model clearly ahead.")
    by_lead = " ".join(says)
    rough = f" {_word(n_seeds).capitalize()} seeds per model are a rough guide to the noise, not a confidence interval"
    if clear(v) and clear(t) and ahead(v) != ahead(t):
        carry = f"{rough}: the validation years and the test years point in opposite directions."
    else:
        carry = f"{rough}."
    accuracy = f"<li><b>Accuracy by lead time.</b> {by_lead}{carry}</li>"

    # warnings: onsets are the warnings that matter
    on = {m: C["test"][lk(first)][m] for m in MODELS}
    lo, hi = ({m: f(on[m]["onset_pod_by_seed"]) for m in MODELS} for f in (min, max))
    seed_rng = lambda m: f"{pct(lo[m])} to {pct(hi[m])}"
    if lo["lstm"] > hi["transformer"] or lo["transformer"] > hi["lstm"]:
        more = "lstm" if lo["lstm"] > hi["transformer"] else "transformer"
        less = "transformer" if more == "lstm" else "lstm"
        onset1 = (f"{LEAD_WORD[first].capitalize()} the {NAME[more]} warns of {pct(on[more]['onset_pod'])} of the onsets and the {NAME[less]} of "
                  f"{pct(on[less]['onset_pod'])} (single seeds {seed_rng(more)} against {seed_rng(less)}, no overlap);")
    else:
        onset1 = (f"{LEAD_WORD[first].capitalize()} the LSTM warns of {pct(on['lstm']['onset_pod'])} of the onsets and the Transformer of "
                  f"{pct(on['transformer']['onset_pod'])} (single seeds {seed_rng('lstm')} against {seed_rng('transformer')}, overlapping: no "
                  f"clear difference);")
    later = [C["test"][lk(k)][m]["onset_pod"] for k in LEADS[1:] for m in MODELS]
    far = [C["test"][lk(k)][m]["onset_far"] for k in LEADS for m in MODELS]
    warnings = (f"<li><b>Warnings.</b> {onset1} {_leads(LEADS[1:])} neither warns of more than {pct(max(later))}, and {pct(min(far))} to "
                f"{pct(max(far))} of all onset warnings are false.</li>")

    # cost: counted operations against measured time; the update limit
    runs = [M[m]["per_seed"][s]["cost"] for m in MODELS for s in seeds]
    at_limit = [r for r in runs if r["updates_run"] >= shared["updates"]]
    bests = sorted({r["best_update"] for r in runs})
    best_txt = f"update {bests[0]}" if len(bests) == 1 else f"updates {bests[0]} to {bests[-1]}"
    if len(at_limit) == len(runs):
        limit = (f"All {_word(len(runs))} trainings ran to the limit of {count(shared['updates'])} updates, set by the time available; the best "
                 f"validation score came at {best_txt}.")
    elif at_limit:
        limit = (f"{_word(len(at_limit)).capitalize()} of the {_word(len(runs))} trainings ran to the limit of {count(shared['updates'])} updates, "
                 f"set by the time available; the best validation scores came at {best_txt}.")
    else:
        limit = f"Every training stopped early, before the limit of {count(shared['updates'])} updates; the best validation scores came at {best_txt}."
    gap = (f"Per update the Transformer needs {_times(ratio['operations_per_update'])} the counted multiply-adds and "
           f"{_times(ratio['seconds_per_update'])} the measured time of a typical (median) update.")
    if ratio["seconds_per_update"] > ratio["operations_per_update"]:
        gap += (f" The count holds only multiplications by weights; it leaves out the rest of the attention step and moving its {days} &times; "
                f"{days} tables through memory (likely reasons for the larger factor; not measured).")
    with_checks = any("Training time includes the validation checks" in x for x in X.get("notes", []))
    total = (f", {_times(ratio['seconds_per_model'])} as long"
             + (" (the total also holds the validation checks and every slow-down by other jobs, so it need not match the factor per update)" if with_checks
                and abs(ratio["seconds_per_model"] - ratio["seconds_per_update"]) > 0.1 * ratio["seconds_per_update"] else ""))
    energy = (f" One model took {count(cost('lstm', 'seconds_per_model'))} s (LSTM) against {count(cost('transformer', 'seconds_per_model'))} s "
              f"(Transformer) to train{total}: {n2(cost('lstm', 'energy_wh_per_model_assumed'))} against "
              f"{n2(cost('transformer', 'energy_wh_per_model_assumed'))} Wh at an assumed {watt:g} W.") if not SHOW_COST_TABLE else ""
    costs = f"<li><b>Cost.</b> {gap}{energy} {limit}</li>"

    # ---- what the whole project cost to train ------------------------------------------------------------------------------
    found = _project_cost(R, T)
    total = sum(s for _, _, s, _ in found)
    n_all = sum(n for _, _, _, n in found if n)
    uncounted = [name for name, _, _, n in found if n is None and name not in NOT_NETWORKS]
    project = f"""
<p><b>What the whole project cost to train.</b> Added up from the results files: {n2(total / 3600, 1)} hours on a two-core laptop without a
graphics card, {'at least ' if uncounted else ''}{count(n_all)} network trainings and about {count(total * watt / 3600)} Wh at the same assumed
{watt:g} W (section 9 lists the runs with their times), not counting the waiting for the processor, test runs and repeated runs. No cloud or
graphics-card prices are quoted, because none were given.</p>"""

    # ---- the caveats the reader must not miss ------------------------------------------------------------------------------
    secs = {m: [M[m]["per_seed"][s]["cost"]["seconds"] for s in seeds] for m in MODELS}
    upd = {M["lstm"]["per_seed"][s]["cost"]["updates_run"] for s in seeds}
    same = f"the same {count(next(iter(upd)))} updates" if len(upd) == 1 else "their updates"
    r_lo, r_hi = min(secs["transformer"]) / max(secs["lstm"]), max(secs["transformer"]) / min(secs["lstm"])
    train_years = int(X["split"]["train"][1][:4]) - int(X["split"]["train"][0][:4])
    busy = any("load average" in x for x in X.get("notes", []))                    # the run's own record: other jobs ran alongside
    caveats = f"""
<div class="note blk"><b>How far this reaches.</b> <i>Timing:</i> taken on a {'busy' if busy else 'shared'} two-core laptop, with test runs and
other light jobs of the project alongside: the {_word(n_seeds)} LSTM seeds needed {count(min(secs['lstm']))} s and {count(max(secs['lstm']))} s for
{same}, so seed against seed the ratio of training times runs from {r_lo:.0f} to {r_hi:.0f}. <i>Energy:</i> {watt:g} W is the nominal power of the
processor, an assumption; nothing was measured. <i>Settings:</i> only the learning rate was checked; a search over size, depth and dropout could
change the picture, more likely for the Transformer. This holds for {n_basins} basins, {train_years} training years and small networks on a
laptop; it is not a verdict on Transformers.</div>"""

    # the heading states the result when the Transformer is not clearly better and dearer to train (the condition of 'No.' in the answer box)
    heading = (f"7 A Transformer: not better here, {r_low:.0f} to {r_high:.0f} times the training time" if not tf_better and costlier else
               "7 Transformer or LSTM, and what training costs")
    # Layout: the question and what was kept identical, then the answer box, then the two checks (one year of context; the mask). With both
    # paragraphs of checks before the box, the box did not fit on the page any more and left its foot empty.
    return f"""{LAYOUT}<h2>{heading}</h2>{method}
{answer}
{beside}
{figure}
{f'<div class="blk">{score_table}</div>' if SHOW_SCORE_TABLE else ''}
{f'<div class="blk">{cost_table}</div>' if SHOW_COST_TABLE else ''}
<ul>{accuracy}{warnings}{costs}</ul>{project}{caveats}"""
