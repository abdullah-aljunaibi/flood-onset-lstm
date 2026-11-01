"""Report section 4.4 - a river the model has never seen (leave one basin out).

Reads results/lstm_lobo.json (written by 05_lobo.py). Nothing is computed here beyond differences, counts and formatting; every
sentence whose wording depends on an outcome is chosen from the stored numbers, and every value is looked up by key: the medians in
summary.l<lead>, the per-basin scores in the comparison rows (by gauge_id and lead), the volume error and the warnings of one basin
in variants.<set>.test / .warnings_model_level (by gauge_id), what is unusual about a basin in basins.<gauge_id>.

Layout: figures/lobo.png is drawn for a width of 170 mm (92% of the 184 mm text width) and is then 147 mm high - more than half a page,
which the page limit of the report does not leave. It is left out and Table 4.4 carries the result, including what the figure cannot
show (the forest, 2 and 3 days ahead). SHOW_FIGURE = True puts the figure in at about its drawn size, never smaller; the figure of
section 4.5 then becomes Figure 4.3 (s4c_timing.py reads SHOW_FIGURE from here).
The terms NSE, seed spread, high-flow day and onset are defined in sections 1 to 3 and are not explained again.

How a difference is judged here: the held-out models have one seed each, so the yardstick is the typical seed spread of the in-training
model (the rule of 05_lobo.py); for the main claim - a never-seen river is forecast worse - the file also stores what the rule of section 4.2
needs (the basin count, a two-sided Wilcoxon p, and in how many basins the held-out score lies below every single in-training seed), and the
heading says 'clearly' only when those agree too.
A basin that scores HIGHER when held out is explained from the keys higher_when_held_out and flow_per_rain (added by the reviewer of
05_lobo.py): the share of the rain that reached the river changed between the training and the test years - in every basin - so the loss
from not seeing a basin is measured together with that change. 'In training' is one seed (lstm_w_s0); the caption of Table 4.4 says so.

Which number leads (decision of the project lead, 4 October 2026): the PAIRED loss - what the typical basin loses when it is held out - comes
first, with the count of basins; the two medians follow, because they are taken over different middle basins. The heading states the result
when the section's own rule says 'clearly worse'. The note at the end sets one published score beside the result (results/published_scores.json,
guarded): what hundreds of training basins made possible, for another task, so the level is not compared.
"""
import re
import sys

from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed, table
from report_sections.s0_summary import LAYOUT, _word, floated
from report_sections.s4_all import P_CUT, _p

LEADS = sorted(LEAD_WORD)
SHOW_FIGURE = False                             # lobo.png under the table (adds about 0.6 page)
T_NO, F_NO = "Table 4.4", "Figure 4.2"          # section 4 uses Tables 4.1-4.3 and Figure 4.1
FIG_SHARE = 84                                  # % of the text width (184 mm); the figure is drawn for 170 mm (92%)
HELD_SET = "lobo_w_static"                      # the prediction set of the never-seen LSTM with basin properties (05_lobo.py)
STATIC_WORD = {"log_area": "area", "elev_mean": "elevation", "slope_mean": "slope", "p_mean": "mean rainfall", "aridity": "dryness of the climate",
               "p_seasonality": "season of the rain", "frac_forest": "forest share", "soil_depth_pelletier": "soil depth", "sand_frac": "sand in the soil",
               "geol_permeability": "permeability of the rock"}
# table rows: key of the score in the results file -> (model and whether the basin was in training, inputs)
ROWS = (("in_training", "LSTM, basin in training", "past weather + basin properties"),
        ("in_training_no_properties", "LSTM, basin in training", "past weather only"),
        ("held_out_with_properties", "LSTM, basin never seen", "past weather + basin properties"),
        ("held_out_without_properties", "LSTM, basin never seen", "past weather only"),
        ("forest_held_out", "Random forest, basin never seen", "past weather + basin properties"))
HELD = ("held_out_with_properties", "held_out_without_properties", "forest_held_out")
TIGHT = "<style>table.tight th,table.tight td{padding:1.8pt 5pt;line-height:1.3}</style>"


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def html(R, T):
    L = R["lstm_lobo"]
    S, lk = L["summary"], (lambda k: f"l{k}")
    first = LEADS[0]
    s = S[lk(first)]
    n, yard = s["n_basins"], s["in_training_seed_range_median"]
    n_seeds, props, ref = len(L["settings"]["in_training_seeds_for_spread"]), L["settings"]["basin_properties"], L["settings"]["in_training_reference"]
    row = {r["gauge_id"]: r for r in L["comparison"] if r["lead"] == first}            # per-basin scores at the first lead, by gauge
    assert len(row) == n and L["variants"][HELD_SET]["uses_basin_properties"]
    inside = lambda d: abs(d) <= yard                                                  # a difference the seed spread could produce

    # ---- what we did --------------------------------------------------------------------------------------------------
    did = f"""<p>Most rivers have no gauge. Could the model forecast one? Each of the {n} basins was left out of training in turn: an LSTM
trained on the other {n - 1} then forecast the left-out river in the test years from its past weather and its {len(props)} basin properties
alone. It never saw that river&rsquo;s flow and reads no river gauge: that is what &lsquo;ungauged&rsquo; means. The same was done without the
basin properties, and with a random forest. Each score is compared with the same basin&rsquo;s score when it was in training; a difference counts
only if it exceeds the seed spread of {n2(yard)} NSE (how far {n_seeds} trainings of the in-training model lie apart in a typical basin).</p>"""

    # ---- the table ----------------------------------------------------------------------------------------------------
    body = [[model, inputs] + [n2(S[lk(k)]["nse_median"][key]) for k in LEADS] + [f"{s['n_basins_positive_nse'][key]} of {n}", n2(s["nse_min"][key])]
            for key, model, inputs in ROWS]
    one_seed = bool(re.search(r"_s\d+$", str(ref)))                                  # the in-training reference is a single seed (lstm_w_s0)
    try:                                                                               # the same model as the seed average of Tables 4.1 and 4.3
        avg = R["lstm_all"]["decomposition"]["levels"]["w"][lk(first)]["nse_median"]
        n_avg = len(R["lstm_all"]["sets"]["w"]["seeds"])
    except (KeyError, TypeError):
        avg = None
    seed_note = ""
    if one_seed:
        seed_note = " &lsquo;In training&rsquo; is one seed of the model of section 4.2 (each held-out model also has one seed)"
        seed_note += (f", so its first row reads {n2(s['nse_median']['in_training'])} where Tables 4.1 and 4.3 have {n2(avg)} for the {_word(n_avg)}-seed "
                      f"average." if avg is not None and n2(avg) != n2(s["nse_median"]["in_training"]) else ".")
    tab = table(["Model", "Inputs"] + [f"NSE {LEAD_WORD[k]}" for k in LEADS] + ["Basins with NSE above 0", "Lowest NSE"],
                body, cls="num tight", widths=["28%", "27%", "9%", "9%", "9%", "10%", "8%"],
                caption=f"<b>{T_NO}. Forecasting a basin that was left out of training, test years.</b> NSE columns: median of the {n} basins; last two "
                        f"columns: {LEAD_WORD[first]}. No model reads a river gauge. Compare each &lsquo;never seen&rsquo; row with the &lsquo;in training&rsquo; "
                        f"row that has the same inputs.{seed_note}")

    # ---- skill lost ---------------------------------------------------------------------------------------------------
    Lw, Lo = s["loss_from_not_seeing_the_basin"]["with_properties"], s["loss_from_not_seeing_the_basin"]["without_properties"]
    below_all = s.get("held_out_below_every_in_training_seed", {}).get("held_out_with_properties")       # basins below every single in-training seed
    p_loss = Lw.get("wilcoxon_p")
    by_rule = Lw["n_basins_lower_when_held_out"] > n / 2 and p_loss is not None and p_loss < P_CUT       # the checks of section 4.2 that the file stores
    lost = ("A never-seen river is forecast clearly worse" if Lw["median_nse_loss"] > yard and by_rule else
            "A never-seen river tends to be forecast worse" if Lw["median_nse_loss"] > yard else
            "A never-seen river is forecast better, which should not be trusted" if Lw["median_nse_loss"] < -yard else "No clear loss for a never-seen river")
    rule = (f" (p = {_p(p_loss)}, Wilcoxon, two-sided)" if p_loss is not None else "")
    below = (f" and in {below_all} below every single seed of the in-training model" if below_all is not None else "")
    better = [r for r in row.values() if r["loss_with_properties"] < 0]                # basins that score higher when held out
    H = {h["gauge_id"]: h for h in L.get("higher_when_held_out", [])}                  # what the reviewer of 05_lobo.py found for them
    odd = ""
    for r in better:
        h = H.get(r["gauge_id"])
        odd += (f" {r['basin']} is {'the' if len(better) == 1 else 'an'} exception ({n2(r['held_out_with_properties'])} held out against "
                f"{n2(r['in_training'])} in training")
        if not h:
            odd += "); the results file holds no explanation."
            continue
        nse_h, share, vol, val = (h["nse_1_day_ahead"], h["share_of_rain_reaching_the_river"], h["forecast_volume_error_pct_1_day_ahead"]["test_years"],
                                  h["nse_validation_years_1_day_ahead"])
        odd += f"; single in-training seeds {n2(nse_h['in_training_seed_min'])} to {n2(nse_h['in_training_seed_max'])})."
        drier = share["test_years"] < share["training_years"]
        too_much = vol["in_training"] > max(vol["held_out_with_properties"], 0)
        usual = val["in_training"] > val["held_out_with_properties"]
        if drier and too_much:
            odd += (f" That is not skill for unseen rivers: {pct(share['training_years'])} of the rain reached this river in the training years and "
                    f"{pct(share['test_years'])} in the test years, so the model that had learned the river forecasts {n2(vol['in_training'], 0)}% too much "
                    f"water in the test years, the held-out model {n2(abs(vol['held_out_with_properties']), 0)}% too "
                    f"{'much' if vol['held_out_with_properties'] > 0 else 'little'}"
                    + (f"; in the validation years the order is the usual one ({n2(val['in_training'])} in training, {n2(val['held_out_with_properties'])} "
                       f"held out)." if usual else "."))
        else:
            odd += " The results file does not explain it."
    FPR = [v["test_over_training"] for v in L.get("flow_per_rain", {}).get("basins", {}).values() if v.get("test_over_training") is not None]
    confound = ""
    if FPR and sum(v < 1 for v in FPR) > len(FPR) / 2:
        less = sum(v < 1 for v in FPR)
        confound = (f" {'In all ' + str(len(FPR)) if less == len(FPR) else f'In {less} of {len(FPR)}'} basins a smaller share of the rain reached the river "
                    f"in the test years than in the training years (test / training {n2(min(FPR))} to {n2(max(FPR))}), and a model trained on a river has "
                    f"learned the older relation. So every loss in {T_NO} is measured together with this change and is approximate.")
    m_in, m_out = s["nse_median"]["in_training"], s["nse_median"]["held_out_with_properties"]
    skill = f"""<p><b>{lost}.</b> With basin properties the typical basin {"loses" if Lw["median_nse_loss"] >= 0 else "gains"}
{n2(abs(Lw["median_nse_loss"]))} NSE {LEAD_WORD[first]} when it is held out: the score is lower in {Lw["n_basins_lower_when_held_out"]} of {n}
basins{rule}, in {Lw["n_basins_loss_larger_than_seed_range"]} by more than the seed spread{below}. This paired loss, not the difference of the two
medians, is the number to quote: the two medians come from different basins. The median {"falls" if m_out < m_in else "rises"} from {n2(m_in)} to
{n2(m_out)} ({T_NO}).{odd}{confound}</p>"""

    # ---- do the basin properties help? --------------------------------------------------------------------------------
    P = s["basin_properties"]["held_out"]
    head = {"help": "Basin properties help", "hurt": "Basin properties hurt"}.get(P["verdict"], "Basin properties do not help")
    lower = sum(r["held_out_with_properties"] < r["held_out_without_properties"] for r in row.values())
    same = all(S[lk(k)]["basin_properties"]["held_out"]["verdict"] == P["verdict"] for k in LEADS[1:])
    others = (f"the reading is the same {_and(LEADS[1:])} days ahead" if same else
              "further ahead the reading is " + _and(f"&lsquo;{S[lk(k)]['basin_properties']['held_out']['verdict']}&rsquo; {LEAD_WORD[k]}" for k in LEADS[1:]))
    less = "less" if Lo["median_nse_loss"] < Lw["median_nse_loss"] else "more"
    small = (f"{'smaller' if inside(Lo['median_nse_loss']) else 'larger'} than the seed spread, though the score is lower in "
             f"{Lo['n_basins_lower_when_held_out']} of {n} basins")
    properties = f"""<p><b>{head}.</b> The never-seen LSTM scores higher with them in {P["n_basins_higher"]} of {P["n_basins"]} basins and lower in {lower};
the median difference of {signed(P["median_nse_difference"])} NSE is {"smaller" if inside(P["median_nse_difference"]) else "larger"} than the seed spread, and
{others}. Without them the LSTM loses {less} when a basin is held out: {n2(Lo["median_nse_loss"])} NSE in the typical basin, {small}.</p>"""

    # ---- the simple forest under the same rule ------------------------------------------------------------------------
    Fw, Fo = s["lstm_against_forest_held_out"]["with_properties"], s["lstm_against_forest_held_out"]["without_properties"]
    d = [Fw["median_nse_difference"], Fo["median_nse_difference"]]
    forest_head = ("The LSTM beats the simple forest" if any(x > yard for x in d) else "The simple forest beats the LSTM" if all(x < -yard for x in d)
                   else "The LSTM does not beat the simple forest")
    size = ("both differences are smaller than the seed spread" if all(inside(x) for x in d) else "both differences are larger than the seed spread"
            if not any(inside(x) for x in d) else f"only the {'second' if inside(d[0]) else 'first'} difference is larger than the seed spread")
    forest = f"""<p><b>{forest_head}.</b> A random forest fitted on the other {n - 1} basins (settings not tuned) reaches a median NSE of
{n2(s["nse_median"]["forest_held_out"])} {LEAD_WORD[first]}. The LSTM with basin properties is ahead of it in {Fw["n_basins_higher"]} of {n} basins
(median difference {signed(d[0])} NSE), the weather-only LSTM in {Fo["n_basins_higher"]} of {n} ({signed(d[1])}); {size}.</p>"""

    # ---- which rivers fail, and the warnings ---------------------------------------------------------------------------
    fails = sorted((g for g, r in row.items() if min(r[c] for c in HELD[:2]) <= 0), key=lambda g: row[g]["held_out_with_properties"])
    V = L["variants"][HELD_SET]
    rank_free = ""                                                                     # one sentence for the warnings paragraph, set below
    if not fails:
        failing = f"<p><b>No river fails outright.</b> Every held-out LSTM keeps an NSE above 0 {LEAD_WORD[first]}.</p>"
    else:
        g = fails[0]
        r, B = row[g], L["basins"][g]
        title = f"One river fails outright: {r['basin']}" if len(fails) == 1 else f"{len(fails)} rivers fail outright ({_and(row[x]['basin'] for x in fails)}); the worst is {r['basin']}"
        vol = V["test"]["basins"][g][lk(first)]["pbias"]                             # forecast volume error of the held-out LSTM, %
        times = lambda v: n2(1 + v / 100, 1)
        failing = (f"<p><b>{title}.</b> Held out, its NSE {LEAD_WORD[first]} is {n2(r[HELD[0]])} with basin properties, {n2(r[HELD[1]])} with weather only "
                   f"and {n2(r[HELD[2]])} for the forest, against {n2(r['in_training'])} in training"
                   + (": every method does worse than always forecasting the average flow." if all(r[c] < 0 for c in HELD) else ".")
                   + f" The held-out LSTM forecasts {times(vol)} times the water that flowed.")
        shares = {x: b["flow_record"]["runoff_ratio"] for x, b in L["basins"].items()}      # the CAMELS value (its own years)
        audit = {b["gauge_id"]: b.get("runoff_ratio") for b in R.get("data_audit", {}).get("selected", [])}
        if all(audit.get(x) is not None for x in shares):                                    # the project's own years: what Table 1.1 shows
            shares, table11 = {x: audit[x] for x in shares}, " (mean flow over mean rain, Table 1.1)"
        else:
            table11 = ""
        share = shares[g]
        if share <= min(shares.values()) and vol > 0:
            failing += (f" What sets this river apart shows only in its flow record, which an ungauged forecast cannot use: {pct(share)} of its rain "
                        f"reaches the river{table11}, the lowest share of the {n} basins (a likely reason, not a tested one).")
        out = B["compared_with_other_11"]["outside_range_of_other_11"]
        also = sum(1 for x, b in L["basins"].items() if x not in fails and b["compared_with_other_11"]["outside_range_of_other_11"])
        if out:
            failing += (f" Its basin properties give little warning: {len(out)} of the {len(props)} {'lies' if len(out) == 1 else 'lie'} outside the range of "
                        f"the other {n - 1} basins" + (f", but so does a property of {also} basins that did not fail." if also else
                                                       ", and no basin that did not fail has a property outside that range."))
        failing += "</p>"
        o = V["warnings_model_level"]["basins"][g][lk(first)]["onset"]
        if vol > 0 and o["pod"] >= s["warnings_model_level"][HELD_SET]["onset"]["pod"]:
            rank_free = (f" The rule only ranks a model&rsquo;s forecasts against each other, so a forecast that is always too high still warns on its "
                         f"highest days: {r['basin'].split(',')[0]}, held out, had {o['hits']} of its {o['hits'] + o['misses']} onsets warned.")

    W = s["warnings_model_level"]
    ho, it = W[HELD_SET]["onset"], W[ref]["onset"]
    poor = " Either way, most onsets are missed and most warnings are false." if max(ho["pod"], it["pod"]) < 0.5 and min(ho["far"], it["far"]) > 0.5 else ""
    warn = f"""<p><b>Warnings without a flow record.</b> An ungauged river has no flow record to take the high-flow level from, so here a warning
is given when the forecast is above the level that the model&rsquo;s own forecasts for that river exceeded on 5% of training-year days (not
comparable with section 5, which takes the level from the gauge). The never-seen LSTM with basin properties warned of {pct(ho["pod"])} of the
onsets {LEAD_WORD[first]}, and {pct(ho["far"])} of its warnings were false; the in-training LSTM under the same rule: {pct(it["pod"])} and
{pct(it["far"])}.{poor}</p>"""

    # ---- what it means ------------------------------------------------------------------------------------------------
    worse = "is forecast worse than one the model trained on" if Lw["median_nse_loss"] > yard else "is not clearly forecast worse than one the model trained on"
    helped = {"help": "the basin properties help", "hurt": "the basin properties make it worse"}.get(P["verdict"], "the basin properties do not help")
    few = (f" {n - 1} rivers for {len(props)} properties are probably too few to learn how properties shape a river&rsquo;s response (a likely reason; "
           f"not tested)." if P["verdict"] != "help" else "")
    cfg = L["settings"]["config"]
    folds = [c for v in L["variants"].values() if v["model"] == "lstm" for c in v["cost_per_fold"].values()]
    capped = sum(c["updates_run"] == cfg["updates"] and c["updates_run"] - c["best_update"] < cfg["patience"] * cfg["eval_every"] for c in folds)
    budget = f" {capped} of the {len(folds)} held-out trainings ran into the limit of {count(cfg['updates'])} training steps;" if capped else ""
    published = ""
    try:                                                                               # results/published_scores.json: read at the source by an AI agent
        Pb = R["published_scores"]["sources"]["kratzert2019pub"]
        v_lstm, v_cal = Pb["median_nse"]["lstm_basin_never_seen"], Pb["median_nse"]["sac_sma_calibrated_on_each_basin"]
        if Pb["citation_key"] != "kratzert2019pub":
            raise KeyError("the record is not that of the study cited here")
        published = (f" In the first of these studies ({count(Pb['n_basins'])} basins) an LSTM that never saw a basin reached a median NSE of {n2(v_lstm)}, "
                     f"{'above' if v_lstm > v_cal else 'against'} {n2(v_cal)} for a conceptual hydrological model calibrated on that basin; that is for "
                     f"the flow of a day from the weather up to that day, so not comparable with {T_NO} (numbers read by an AI agent, see section 4.3).")
    except (KeyError, TypeError) as e:
        print(f"s4b_lobo: the published score for never-seen basins was not read from results/published_scores.json ({type(e).__name__}: {e}); "
              f"its sentence is left out", file=sys.stderr)
    note = f"""<div class="note blk"><b>What this shows, and what it does not.</b> Published work that forecasts ungauged rivers trains one model on
hundreds to thousands of basins ({cite("kratzert2019pub")[1:-1]}; {cite("nearing2024")[1:-1]}); here it is {n - 1}.{published} With {n - 1} training rivers a
never-seen river {worse} and {helped}; that says nothing about a large training set.{few} Limits: one seed per held-out model;{budget} model
settings chosen on the validation years of all {n} basins, the held-out one included (a weak leak).</div>"""

    figure = ""
    if SHOW_FIGURE:
        off = [r["basin"] for r in row.values() if min(r[c] for c in HELD[:2]) < -0.5]
        figure = floated(fig("lobo.png", f"<b>{F_NO}. NSE {LEAD_WORD[first]} in the test years, basin by basin.</b> Filled dot: the basin was in training; diamond: held "
                     f"out, with basin properties; open circle: held out, weather only; shaded bar: the range of the {n_seeds} seeds of the in-training model. "
                     f"The further a diamond or circle lies to the left of its dot, the more skill was lost"
                     + (f"; {_and(off)} {'is' if len(off) == 1 else 'are'} off the scale." if off else "."), FIG_SHARE))
    heading = "4.4 A river the model has never seen" + (" is forecast clearly worse" if lost == "A never-seen river is forecast clearly worse" else "")
    return f"""{LAYOUT}{TIGHT}<h3 class="clr">{heading}</h3>
{did}
<div class="blk">{tab}</div>
{figure}
{skill}
{properties}
{forest}
{failing}
{warn}
{note}"""
