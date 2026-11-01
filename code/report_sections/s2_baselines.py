"""Report section 2 - the simple models to beat. Reads results/baselines.json (written by 02_baselines.py); where results/forecast_input.json exists, one sentence says where the one real
weather forecast of section 4.8 lands (forecast_where)."""
from report_common import INPUT_NAME, LEAD_WORD, MODEL_NAME, count, fig, kpis, model_name, n2, pct, signed, table
from report_sections.s0_summary import forecast_where

LEADS = sorted(LEAD_WORD)
SHOW_FIGURE = False                            # True: Figure 2.1 (baselines.png) after the lessons, at the full text width - its lettering is drawn for
                                               # that width (about 4-5 pt at half width) and it then takes about 0.65 page; Tables 2.1 and 2.2 carry its numbers
SHOW_KPIS = False                              # False: the numbers to beat in one sentence instead of three boxes (saves about 20 mm)
TIMING_REF = "section 4.5"                     # where the rain time stamps are reported
TIGHT = "<style>.s2t th,.s2t td{padding:1.4pt 6pt;line-height:1.25}</style>"       # layout only: closer rows in this section's tables
QUOTED, QUOTED_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"


def _three(vals, em, bold=()):
    """Three formatted values (1, 2, 3 days ahead) side by side in one table cell; leads in `bold` are the number to beat."""
    return "".join(f'<span style="display:inline-block;width:{em}em;text-align:right">{f"<b>{v}</b>" if k in bold else v}</span>'
                   for k, v in zip(LEADS, vals))


def _years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def _range(vals, fmt):
    return f"{fmt(min(vals))} to {fmt(max(vals))}"


def _and(parts):
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def html(R, T):
    B = R["baselines"]
    M, F, split = B["models"], B["features"], B["split"]
    S = {(r["model"], r["lead"]): r for r in B["summary_table"]}                  # test-years scores by (model, lead)
    PERS, CAL = "base_persistence", "base_climatology"
    PAIR = {m: (f"base_{m}_w", f"base_{m}_wq") for m in ("linreg", "rf")}         # the same model without and with the river gauge
    plain = lambda m: MODEL_NAME[f"base_{m}"].lower()
    no_fc = [n for n in M if not M[n]["uses_weather_forecast"]]
    with_fc = [n for n in M if M[n]["uses_weather_forecast"]]
    fitted = [n for pair in PAIR.values() for n in pair]
    n_basins = len(M[PERS]["test"]["basins"])
    best = {k: B["best_by_validation"][f"l{k}"]["without_weather_forecast"] for k in LEADS}
    best_fc = {k: B["best_by_validation"][f"l{k}"]["with_perfect_forecast"] for k in LEADS}
    like = {k: B["best_by_validation_same_information"]["w"][f"l{k}"] for k in LEADS}                 # best baseline that sees no river gauge
    first, last = LEADS[0], LEADS[-1]
    nse = lambda n, k: S[n, k]["nse_median"]

    # ---- what we did ------------------------------------------------------------------------------------------------
    sums = sorted(int(c[len("rain_sum"):]) for c in F["w"] if c.startswith("rain_sum"))
    trees = sorted({M[n]["settings"]["n_trees"] for n in M if M[n]["model"] == "rf"})
    pooled = M[PERS]["test"]["summary"]                                           # the high-flow days are the same for every model
    n_high = pooled[f"l{first}"]["all"]["hits"] + pooled[f"l{first}"]["all"]["misses"]
    n_onset = {k: pooled[f"l{k}"]["onset"]["hits"] + pooled[f"l{k}"]["onset"]["misses"] for k in LEADS}
    did = f"""
<p>An LSTM is only worth its cost if it forecasts better than simple methods. The {len(M)} simple models of this section are each fitted for every basin and lead time
on the training years, with settings chosen on the validation years (about {n2(B['fit_seconds'] / 60, 0)} minutes of computing).</p>
<ul>
<li><b>&lsquo;Tomorrow = today&rsquo;</b>: the forecast is today&rsquo;s flow at the river gauge. Nothing is fitted.</li>
<li><b>Calendar-day average</b>: the average flow of that day of the year in the training years. It uses no live data.</li>
<li><b>Ridge regression</b>: a weighted sum of {len(F['w'])} hand-made features of the past weather, such as the rain sums of the last
{sums[0]} to {sums[-1]} days, the temperature and the season. &lsquo;Ridge&rsquo; means the weights are kept small so that they do not fit noise.</li>
<li><b>Random forest</b>: the average of {_and(str(t) for t in trees)} decision trees (chains of yes/no questions) on the same features. Unlike a
weighted sum, it can follow effects that are not straight lines.</li>
</ul>
<p>The last two were each run on the four sets of inputs of section 1: the river gauge adds {len(F['wq']) - len(F['w'])} features, the perfect weather forecast (an
upper bound, not a usable model) the rain and temperature recorded for the days up to the one being forecast.</p>
<p>Accuracy is scored with the NSE (1 is a perfect forecast). A model <i>warns</i> when its forecast is above the high-flow level. <i>Share warned</i> is the share of
high-flow days for which a warning was given; <i>false warnings</i> is the share of warnings not followed by a high-flow day. An <i>onset</i> is a high-flow day forecast
while the river was still below the high-flow level: the warning that matters, because the river is not yet high when it is issued. Of the {count(n_high)} high-flow days
in the test years, {count(n_onset[first])} are onsets for a forecast made {LEAD_WORD[first]} and {count(n_onset[last])} for one made {LEAD_WORD[last]}.</p>"""

    # ---- the number to beat -----------------------------------------------------------------------------------------
    top_test = {k: max(no_fc, key=lambda n: nse(n, k)) for k in LEADS}
    other = [k for k in LEADS if top_test[k] != best[k]["model"]]                 # leads where another baseline scored higher on the test years
    val = lambda n, k: M[n]["val"]["summary"][f"l{k}"]["nse_median"]
    margin = max(sorted((val(n, k) for n in no_fc), reverse=True)[0] - sorted((val(n, k) for n in no_fc), reverse=True)[1] for k in LEADS)
    close = f"where the first two models are never more than {n2(margin, 3)} NSE apart"
    if other:
        choice = (f"The choice is made on the validation years, {close}, so the number to beat is not always the highest test score of its column.")
    else:
        choice = (f"The choice is made on the validation years, {close}; the chosen model still has the highest test score of its column in Table 2.1 at every lead time.")
    gap = max(abs(nse(a, k) - nse(b, k)) for k in LEADS for a, b in zip(PAIR["linreg"], PAIR["rf"]))      # ridge against forest, same inputs
    like_text = _and(f"{n2(like[k]['test']['nse_median'])} {LEAD_WORD[k]}" for k in LEADS)
    rest = (f"{choice} On the same inputs without a weather forecast, ridge regression and random forest never differ by more than {n2(gap)} NSE, and the forest was fitted from one "
            f"random start only, so the two are not ranked. Without a river gauge, the number to beat is "
            f"{like_text}.")
    rule = "the test-years score of the simple model without a weather forecast with the highest median NSE in the validation years"
    if SHOW_KPIS:
        number = f"""
<p><b>The number to beat</b> at each lead time is {rule}:</p>
{kpis([(n2(best[k]["test"]["nse_median"]), f"median NSE to beat {LEAD_WORD[k]}:<br>{model_name(best[k]['model']).lower()}") for k in LEADS])}
<p>{rest}</p>"""
    else:
        groups = {}                                                               # model -> its leads, in order
        for k in LEADS:
            groups.setdefault(best[k]["model"], []).append(k)
        beat = "; ".join(_and(f"{n2(best[k]['test']['nse_median'])} {LEAD_WORD[k]}" for k in ks) + f" ({model_name(n).lower().replace(' (', ', ').rstrip(')')})"
                         for n, ks in groups.items())
        number = f"""
<p><b>The number to beat</b> at each lead time is {rule} (bold in Table 2.1): {beat}. {rest}</p>"""

    # ---- tables -----------------------------------------------------------------------------------------------------
    W_NSE, W_PCT = 3.0, 2.6
    sub = lambda em: _three(["1 d", "2 d", "3 d"], em)
    header = ["Model, inputs", f"Median NSE<br>{sub(W_NSE)}", f"Share warned<br>{sub(W_PCT)}", f"False warnings<br>{sub(W_PCT)}",
              f"Onsets warned<br>{sub(W_PCT)}"]
    widths = ["36.6%", "17.5%", "15.3%", "15.3%", "15.3%"]

    def row(n, chosen):
        m = M[n]
        inputs = INPUT_NAME[m["inputs"]] if m["model"] in ("persistence", "climatology") else INPUT_NAME["wq" if m["uses_river_gauge"] else "w"]
        bold = [k for k in LEADS if chosen[k]["model"] == n]
        return [f"{MODEL_NAME['base_' + m['model']].replace('Tomorrow = today', '&lsquo;Tomorrow = today&rsquo;')}, {inputs}",
                _three([n2(nse(n, k)) for k in LEADS], W_NSE, bold), _three([pct(S[n, k]["pod"]) for k in LEADS], W_PCT, bold),
                _three([pct(S[n, k]["far"]) for k in LEADS], W_PCT, bold), _three([pct(S[n, k]["onset_pod"]) for k in LEADS], W_PCT, bold)]

    def edge_note(names):
        """A sentence for the caption when a chosen setting is the smallest or largest value that was tried (the model may be under-tuned)."""
        hit = []
        for n in names:
            st = M[n]["settings"]
            for key, log in (("ridge_alpha", "validation_nse_by_alpha"), ("min_samples_leaf", "validation_nse_by_leaf")):
                tried = sorted(float(v) for v in st.get(log, {}))
                if len(tried) > 1 and st[key] in (tried[0], tried[-1]) and plain(M[n]["model"]) not in hit:
                    hit.append(plain(M[n]["model"]))
        return (f" For the {_and(hit)} the chosen setting was at the end of the list tried, so a better one may exist and "
                f"{'its' if len(hit) == 1 else 'their'} scores here are, if anything, a little low.") if hit else ""

    silent = any(S[n, k]["far"] != S[n, k]["far"] for n in no_fc for k in LEADS)                      # NaN: no warning was given
    top = max(((n, k) for n in no_fc for k in LEADS), key=lambda nk: S[nk]["onset_pod"])              # best onset score without a forecast
    tables = f"""
<div class="blk">
{table(header, [row(n, best) for n in no_fc], cls="num s2t", widths=widths, caption=f"<b>Table 2.1. Without a weather forecast: what could be run in practice.</b> "
    f"Test years. The three numbers in each cell are for 1, 2 and 3 days ahead (1 d, 2 d, 3 d). NSE is the median of the {n_basins} basins; the "
    f"warning scores count the {n_basins} basins together; &lsquo;onsets warned&rsquo; is the share warned among the onsets. Bold: the number to beat "
    f"at that lead time.{' A dash: the model gave no warning.' if silent else ''}{edge_note(no_fc)}")}
</div>
<div class="blk">
{table(header, [row(n, best_fc) for n in with_fc], cls="num s2t", widths=widths, caption=f"<b>Table 2.2. Upper bound: the same two models given a perfect weather "
    f"forecast as well</b> (the weather recorded up to the day being forecast). It cannot be had in practice. Bold: the best of the {len(with_fc)} in the validation years, the number to beat for a model that is "
    f"itself given a perfect weather forecast.{edge_note(with_fc)}")}
</div>"""

    # ---- figure -----------------------------------------------------------------------------------------------------
    p1, p3 = S[PERS, first], S[PERS, last]
    fc_nse = [nse(n, k) for n in with_fc for k in LEADS]
    figure = fig("baselines.png", f"<b>Figure 2.1. The simple models in the test years.</b> Left: accuracy; each small dot is one basin, each large "
                 f"marker the median of the {n_basins} (bottom: the upper bound). Right: warnings without a weather forecast. Filled markers use the "
                 f"river gauge, hollow ones do not. What to see: the grey {QUOTED} falls from "
                 f"{n2(p1['nse_median'])} to {n2(p3['nse_median'])}, and {LEAD_WORD[first]} its basins spread from "
                 f"{n2(pooled[f'l{first}']['nse_min'])} to {n2(pooled[f'l{first}']['nse_max'])}; with a perfect weather forecast the medians stay "
                 f"between {n2(min(fc_nse))} and {n2(max(fc_nse))} at every lead time.")

    # ---- what the simple models already teach -----------------------------------------------------------------------
    basin_nse = lambda n, k: {g: b[f"l{k}"]["nse"] for g, b in M[n]["test"]["basins"].items()}
    wins = lambda a, b, k: sum(basin_nse(a, k)[g] > basin_nse(b, k)[g] for g in basin_nse(a, k))      # basins where a scores higher than b
    wo1 = [nse(w, first) for w, g in PAIR.values()]
    beaten = sum(p1["nse_median"] > v for v in wo1)
    head_a = (f"{LEAD_WORD[first].capitalize()}, past weather alone does not beat {QUOTED}" if beaten == len(wo1)
              else f"{LEAD_WORD[first].capitalize()}, past weather alone beats {QUOTED}" if beaten == 0
              else f"{LEAD_WORD[first].capitalize()}, past weather alone beats {QUOTED} with one of the two models")
    useless = p3["nse_median"] < nse(CAL, last)
    head_b = f"{LEAD_WORD[last]} the rule is useless" if useless else f"{LEAD_WORD[last]} the rule still beats the calendar-day average"
    middle = [k for k in LEADS if k not in (first, last)]
    one = (f"<li><b>{head_a}; {head_b}.</b> With nothing fitted it reaches NSE {n2(p1['nse_median'])} {LEAD_WORD[first]} (the two models on past "
           f"weather: {_and(n2(v) for v in wo1)}; single basins: {n2(pooled[f'l{first}']['nse_min'])} to {n2(pooled[f'l{first}']['nse_max'])}) and warns of "
           f"{pct(p1['pod'])} of high-flow days. Then: "
           f"{_and(f'{n2(nse(PERS, k))} {LEAD_WORD[k]}' for k in middle + [last])}, {'below' if useless else 'above'} the calendar-day "
           f"average ({n2(nse(CAL, last))}), with {pct(p3['far'])} false warnings.</li>")

    gain = lambda k: [nse(g, k) - nse(w, k) for w, g in PAIR.values()]            # same model, river gauge added
    gauge_first = p1["nse_median"] > max(wo1)
    head_2 = (f"{LEAD_WORD[first].capitalize()}, the river gauge is worth more than the past weather" if gauge_first
              else f"{LEAD_WORD[first].capitalize()}, the river gauge alone is not worth more than the past weather")
    fades = "The gain fades with the lead time" if max(gain(last)) < min(gain(first)) else "The gain does not fade with the lead time"
    two = (f"<li><b>{head_2}.</b> The gauge alone ({QUOTED}, {n2(p1['nse_median'])}) scores {'higher' if gauge_first else 'no higher'} than "
           f"{len(F['w'])} weather features without it. Adding it takes "
           + _and(f"{plain(m)} from {n2(nse(w, first))} to {n2(nse(g, first))}" for m, (w, g) in PAIR.items())
           + ", and their false warnings "
           + _and(f"from {pct(S[w, first]['far'])} to {pct(S[g, first]['far'])}" for w, g in PAIR.values())
           + f". With it they beat {QUOTED} in {_and(str(wins(g, PERS, first)) for w, g in PAIR.values())} of {n_basins} basins. "
           f"{fades}: {LEAD_WORD[last]} it adds {_and(signed(v) for v in gain(last))}.</li>")

    on_far = [S[n, k]["onset_far"] for n in fitted for k in LEADS]
    on_fc = [S[n, k]["onset_pod"] for n in with_fc for k in LEADS]
    on_fc_far = [S[n, k]["onset_far"] for n in with_fc for k in LEADS]
    pers_silent = all(S[PERS, k]["onset_pod"] == 0 for k in LEADS)                # true by construction: it warns only when the river is already high
    all_higher = min(on_fc) > S[top]["onset_pod"]
    three = (f"<li><b>Without a weather forecast, the simple models warn of at most {pct(S[top]['onset_pod'])} of onsets.</b> That best case is the "
             f"{model_name(top[0]).lower()}, {LEAD_WORD[top[1]]}. Of the onset warnings of the four fitted models, {_range(on_far, pct)} are false. "
             + (f"{QUOTED_CAP} warns of none: its forecast is above the high-flow level only when the river already is. " if pers_silent else "")
             + f"Given a perfect weather forecast (the upper bound), the same two models warn of {_range(on_fc, pct)} of onsets ({_range(on_fc_far, pct)} false)"
             + (", more than any run without it" if all_higher else "")
             + f". That gap is what the weather of the days being forecast is worth; {LEAD_WORD[first]}, part of it is rain that had already fallen but was not yet "
               f"in the inputs ({TIMING_REF}).</li>")

    catch = (f"""<div class="note"><b>How not to read Table 2.1.</b> """
             + (f"A share warned of {pct(p1['pod'])} looks useful, but {QUOTED} warns only when the river is already high. "
                f"The overall share warned flatters any model that reads the gauge; the onset column is the honest test. " if pers_silent else
                "The overall share warned mixes warnings given when the river is already high with real onsets; the onset column is the honest test. ")
             + "Table 2.2 is no promise: a real weather forecast has errors, so a real system would land between the two tables"
             + (f"; {forecast_where(R, short=True)}." if forecast_where(R) else ", and the results do not say where.")
             + "</div>")

    lessons = f"<h3>What the simple models already teach</h3><ol>{one}{two}{three}</ol>"
    found = lessons + (figure if SHOW_FIGURE else "")
    return f"""{TIGHT}<h2>2 The simple models to beat</h2>{did}{number}{tables}
{found}
{catch}"""
