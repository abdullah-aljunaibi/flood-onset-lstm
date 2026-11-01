"""Section 6.2 - rain from space, wet soil and water vapour: the satellite and reanalysis inputs, how well satellite rain agrees
with gauge-based rain, and what the LSTM gains or loses with them.

Reads results/satellite_data.json (fetch_satellite.py: source, grid, coverage, rain check, day offset) and
results/satellite_inputs.json (08_satellite_inputs.py: experiment (a) rain source on the satellite-era split, experiment (b)
extra inputs on the main split). Nothing is computed here beyond counting, ranges, differences, unit conversion of the grid
size and formatting. Whether a change is "clear" is the rule of 08_satellite_inputs.py (key larger_than_seed_noise; how often
it fires by chance is read from seed_noise_rule), never a threshold of this file; every sentence that depends on an outcome is
built from those flags, so a rerun cannot leave a false sentence behind.

Two kinds of score stand side by side in the results file and the text says which is which: the median NSE of a variant is the MEAN of the
scores of its two seeds (key nse_median.base / .variant), the basin counts use the forecast AVERAGED over the seeds - unlike section 4,
whose scores are those of the averaged forecast. The water vapour tested is reanalysis output; a GNSS receiver measures signal delays, from
which precipitable water is derived (Bevis et al., 1992). MERRA-2 soil wetness is computed from gauge-corrected rain (notes of the results file).

Layout: both figures are drawn for a width of 170 mm, so a figure is shown at about that width or left out, never shrunk.
figures/satellite_inputs.png (the experiment) is always shown; figures/satellite_rain_check.png is off by default
(SHOW_RAIN_FIGURE): its numbers - ranges, medians and the two driest basins - are in the text. The figure numbers follow the
figures of section 6.1 (s6a_maps.py).
"""
import math
import re

from flood import LEADS, SHORT
from report_common import LEAD_WORD, cite, count, fig, n2, pct, signed
from report_sections.s0_summary import LAYOUT, floated, nse_reading, onset_clear, timing_facts
from report_sections.s6a_maps import SHOW_COVER

SHOW_RAIN_FIGURE = False            # True: also show figures/satellite_rain_check.png (about 0.45 page); its numbers are in the text either way
RAIN_SHARE = 84                     # % of the text width (184 mm) of figures/satellite_rain_check.png; drawn for 170 mm (92%)
SKILL_SHARE = 84                    # % of the text width of figures/satellite_inputs.png
_first = 3 if SHOW_COVER else 2                         # section 6.1 shows Figure 6.1 (and 6.2 with the land-cover map)
FIG_RAIN, FIG_SKILL = f"Figure 6.{_first}", f"Figure 6.{_first + SHOW_RAIN_FIGURE}"
TIMING_REF = "section 4.5"          # where the rain time stamps are tested (10_rain_timing.py), as in sections 1, 2, 4 and 5
LSTM_REF = "section 4"              # where the all-basin LSTM is introduced
KM_PER_DEGREE = 111.2               # length of one degree of latitude; unit conversion only
UTC_GAUGE = "02472000"              # Leaf River, MS: the one basin in the Central time zone, where the UTC day ends an hour earlier
MONTH = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
NUMBER = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
NEXT_DAY = {"gauge": "gauge-based rain (Daymet) itself", "satellite": "satellite rain (IMERG)",
            "soil": "day-to-day change of top-layer soil wetness (MERRA-2)", "vapour": "precipitable water (MERRA-2)"}     # keys of what_inputs_know_about_next_daymet_day


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _word(v):
    return NUMBER.get(v, str(v))


def _leads(ks):
    """[1] -> '1 day ahead'; [1, 2] -> '1 and 2 days ahead'."""
    ks = list(ks)
    return LEAD_WORD[ks[0]] if len(ks) == 1 else f"{_and(ks)} days ahead"


def _date(s):
    y, m, d = s.split("-")
    return f"{int(d)} {MONTH[int(m) - 1]} {y}"


def _years(span):
    return f"{span[0][:4]}&ndash;{span[1][:4]}"


def _n_years(span):
    return int(span[1][:4]) - int(span[0][:4])


def _chg(v, d=2):
    """A change of a score: signed, and a plain 0.00 instead of -0.00."""
    return n2(0.0, d) if round(v, d) == 0 else signed(v, d)


def _clear(c):
    return c["larger_than_seed_noise"] is True


def _split(cmp, metric="nse_median"):
    """Lead times at which a comparison of 08_satellite_inputs.py is clearly up, clearly down, or not clear (its own rule)."""
    up = [k for k in LEADS if _clear(cmp[f"l{k}"][metric]) and cmp[f"l{k}"][metric]["difference"] > 0]
    down = [k for k in LEADS if _clear(cmp[f"l{k}"][metric]) and cmp[f"l{k}"][metric]["difference"] < 0]
    return up, down, [k for k in LEADS if k not in up + down]


def _nse_story(cmp, short=None):
    """What a variant does to the median NSE, as the rest of a sentence that starts with the variant's name.
    short: '' or ' either' gives the brief form when nothing is clear (the changes only; the figure shows the rest)."""
    row = {k: cmp[f"l{k}"] for k in LEADS}
    up, down, rest = _split(cmp)
    arrow = lambda k: f"{n2(row[k]['nse_median']['base'])} &rarr; {n2(row[k]['nse_median']['variant'])} {LEAD_WORD[k]}"
    changes = lambda ks: _and(_chg(row[k]["nse_median"]["difference"]) for k in ks)

    def better(ks):
        nb = [row[k]["n_basins_improved"] for k in ks]
        same = len(set(nb)) == 1
        return f"better in {nb[0] if same else _and(nb)} of {row[ks[0]]['n_basins']} basins{' each time' if same and len(nb) > 1 else ''}"

    bits = []
    if up:
        bits.append(f"<b>raises the median NSE clearly {_leads(up)}</b> ({'; '.join(arrow(k) for k in up)}; {better(up)})")
    if down:
        bits.append(f"<b>lowers the median NSE clearly {_leads(down)}</b> ({'; '.join(arrow(k) for k in down)}; {better(down)})")
    if not bits and short is not None:
        return f"makes no clear difference{short} (change of the median NSE {changes(LEADS)})."
    if not bits:
        return (f"<b>makes no clear difference</b> at any lead time ({_leads(LEADS)}: change of the median NSE {changes(LEADS)}; seed spread "
                f"{_and(n2(row[k]['nse_median']['seed_noise']) for k in LEADS)}; {better(LEADS)}).")
    tail = f" {_leads(rest).capitalize()} the change ({changes(rest)}) is not clear." if rest else ""
    return " and ".join(bits) + "." + tail


def html(R, T):
    S, X = R["satellite_data"], R["satellite_inputs"]
    A, B = X["experiments"]["a_rain_source"], X["experiments"]["b_extra_inputs"]
    ca, cb = A["against_base"], B["against_base"]
    ids = list(X["basins"])
    n = len(ids)
    n_seeds = len(X["seeds"])
    quoted_final, quoted_early = "&lsquo;Final Run&rsquo;", "&lsquo;Early&rsquo; and &lsquo;Late&rsquo; runs"

    # ---- the data ---------------------------------------------------------------------------------------------------------
    cells = S["cells_per_basin"]
    n_cells = [cells[g]["n_imerg_cells"] for g in ids]
    area = [cells[g]["camels_area_km2"] for g in ids]
    fine = float(re.search(r"([\d.]+) x [\d.]+ degree", S["grid"]["finding"]["IMERG_PRECTOT"]["grid_assumed"]).group(1))
    coarse = re.search(r"([\d.]+) \(latitude\) x ([\d.]+) \(longitude\) degree", S["grid"]["finding"]["GWETTOP"]["grid_assumed"])
    c_lat, c_lon = float(coarse.group(1)), float(coarse.group(2))
    lats = sorted(cells[g]["centroid_lon_lat"][1] for g in ids)
    mid_lat = (lats[(n - 1) // 2] + lats[n // 2]) / 2
    coarse_km = (c_lat * KM_PER_DEGREE, c_lon * KM_PER_DEGREE * math.cos(math.radians(mid_lat)))        # one coarse cell at the latitude of the basins
    cell_km2 = coarse_km[0] * coarse_km[1]
    n_smaller = sum(a < cell_km2 for a in area)
    size_word = "more than the largest basin" if n_smaller == n else f"more than {n_smaller} of the {n} basins"
    cov = S["coverage"]
    gaps = sum(cov[c]["missing_days_after_first_valid_day_all_basins"] for c in ("imerg", "gwettop", "gwetroot", "pw"))
    run_from = re.search(r"Final Run from (\d{4})", S["imerg_run"])
    run_late = re.search(r"about ([\d.]+) months late", S["imerg_run"])
    if run_from and run_late:
        run = f"; from {run_from.group(1)} on it is IMERG&rsquo;s {quoted_final} (see the note below), before that POWER does not name the run"
        late = (f"The {quoted_final} is corrected with monthly rain-gauge totals and published about {run_late.group(1)} months late. On the day itself only the "
                f"uncorrected {quoted_early} exist, and they were not tested.")
    else:
        run = "; POWER does not say which IMERG run it serves"
        late = f"The satellite rain used here is not what is at hand on the day itself: the uncorrected real-time runs ({quoted_early}) were not tested."
    data = f"""<p><b>The data.</b> All of it comes from NASA&rsquo;s POWER service {cite("power")}, daily and without a login. <i>Satellite rain</i> is GPM
IMERG: rain estimated from microwave and infrared satellites on a {n2(fine, 1)}&deg; grid (about {n2(fine * KM_PER_DEGREE, 0)} km); the basin value is the
area-weighted average of the {min(n_cells)} to {max(n_cells)} cells a basin overlaps. The series starts on
{_date(cov["imerg"]["first_valid_day"])}{" and has no gaps" if gaps == 0 else ""}{run}. <i>Soil wetness</i> (top layer and root zone; 0 = dry, 1 = saturated)
and <i>precipitable water</i> (the water vapour in the whole air column) come from MERRA-2 {cite("gelaro2017")}, a reanalysis: output of a weather model
that assimilates satellite and other observations, not a direct satellite measurement. One cell of its {n2(c_lat, 1)}&deg; &times; {n2(c_lon, 3)}&deg;
grid covers about {count(round(cell_km2, -2))} km&sup2;, {size_word}.</p>"""

    # ---- satellite rain against gauge-based rain; whose day ----------------------------------------------------------------
    rs, rm, check = S["rain_check_summary"]["imerg"], S["rain_check_summary"]["merra2"], S["rain_check"]
    bias = {g: check[g]["imerg"]["bias_percent"] for g in ids}
    low = sorted(ids, key=bias.get)[:2]
    low_txt = _and(f"{check[g]['short'].split(',')[0]} ({signed(bias[g], 0)}%)" for g in low)
    low_where = "both mountain basins" if all(check[g]["landscape"].startswith("Mountains") for g in low) else "not both in the mountains"
    r_same, r_before, r_after = rs["median_correlation"], rs["median_correlation_satellite_one_day_earlier"], rs["median_correlation_satellite_one_day_later"]
    two_utc_days = abs(r_same - r_before) < 0.1 and r_after < 0.5 * min(r_same, r_before)             # the pattern the notes of fetch_satellite.py describe
    same_in_reanalysis = (abs(rm["median_correlation"] - rm["median_correlation_satellite_one_day_earlier"]) < 0.1
                          and rm["median_correlation_satellite_one_day_later"] < 0.5 * rm["median_correlation_satellite_one_day_earlier"])
    two_day = "" if SHOW_RAIN_FIGURE else (f" Two-day totals, for which the day boundary does not matter, correlate better (median "
                                           f"{n2(rs['median_correlation_two_day_sums'])}).")
    agreement = f"""<p><b>Does the satellite see the same rain?</b> Only in part{f" ({FIG_RAIN})" if SHOW_RAIN_FIGURE else ""}. Against the
gauge-based rain of the rain table (Daymet), {_years(S["rain_check_period"])}, the day-by-day
correlation is {n2(rs["correlation_min"])} to {n2(rs["correlation_max"])} (median {n2(r_same)}). The totals differ by
{signed(rs["bias_percent_min"], 0)}% to {signed(rs["bias_percent_max"], 0)}%; the two largest shortfalls, {low_txt}, are {low_where}. Of the heaviest 5% of
gauge-based rain days, the satellite puts {pct(rs["median_heavy_days_seen_same_day"])} among its own heaviest 5% on the same day and
{pct(rs["median_heavy_days_seen_within_one_day"])} within one day.{two_day}</p>"""
    if two_utc_days:
        offset = (f"its rain of day d matches the satellite&rsquo;s UTC day d ({n2(r_same)}) and UTC day d&minus;1 ({n2(r_before)}) about equally, and UTC day "
                  f"d+1 hardly at all ({n2(r_after)})" + ("; reanalysis rain shows the same pattern, so the offset comes from the rain table. Its " if same_in_reanalysis else ". Its ")
                  + f"day seems to end around the morning: an inference from these correlations ({TIMING_REF} gives the measurements and the sources).")
    else:
        offset = (f"its rain of day d correlates {n2(r_same)} with the satellite&rsquo;s UTC day d, {n2(r_before)} with UTC day d&minus;1 and {n2(r_after)} with UTC "
                  f"day d+1.")
    whose_day = f"""<p><b>Whose day is it?</b> POWER serves satellite rain by UTC day, which ends at 19:00 local standard time (18:00 at
{SHORT[UTC_GAUGE].split(",")[0]}): the value of day t is complete when the forecast is issued at the end of day t, and nothing from after the issue
time is used. The rain table counts a different day: {offset}</p>"""
    cap_rain = (f"<b>{FIG_RAIN}. Satellite rain against gauge-based rain, {_years(S['rain_check_period'])}.</b> Hollow: two-day totals, for which the day "
                f"boundary does not matter (median {n2(rs['median_correlation_two_day_sums'])}). Gauge-based rain is an interpolation too, not the truth.")
    rain_figure = floated(fig("satellite_rain_check.png", cap_rain, RAIN_SHARE)) if SHOW_RAIN_FIGURE else ""

    # ---- the two experiments ------------------------------------------------------------------------------------------------
    sa, sb = A["split"], B["split"]
    flags = X["comparisons_larger_than_seed_noise_nse"]
    n_flag, n_cmp = sum(v[0] for v in flags.values()), sum(v[1] for v in flags.values())
    by_chance = list(X["seed_noise_rule"]["share_of_comparisons_flagged_when_there_is_no_real_difference"].values())
    chance_txt = pct(min(by_chance)) if pct(min(by_chance)) == pct(max(by_chance)) else f"{pct(min(by_chance))} to {pct(max(by_chance))}"
    expected = n_cmp * sum(by_chance) / len(by_chance)
    same_days = all("same first forecast day" in e["equal_days_check"] for e in (A, B))
    main_updates = R.get("lstm_all", {}).get("config", {}).get("updates")                          # the training budget of section 4, if that result exists
    budget = (f" but a shorter training (at most {count(X['config']['updates'])} weight updates instead of {count(main_updates)})"
              if main_updates and X["config"]["updates"] < main_updates else "")
    method = f"""<p><b>Two experiments</b> ({FIG_SKILL}). Both use the all-basin LSTM of {LSTM_REF} with its settings{budget}. Every variant is trained from
{_word(n_seeds)} seeds, with past weather only and with past weather + river gauge: {X["n_trainings"]} trainings. <b>(a) Rain source:</b> gauge-based rain
(the base), satellite rain instead, or both. The satellite record is short, so this experiment has its own split (training {_years(sa["train"])}, validation
{_years(sa["val"])}, test {_years(sa["test"])}) and <b>its scores cannot be compared with <span style="white-space:nowrap">sections 2&ndash;5</span></b>;
only differences inside it count. <b>(b) Extra inputs</b> on the main split: the base, plus soil wetness, plus precipitable water, plus both.
{"Inside an experiment every variant, the base included, is trained and scored on exactly the same days; the base of (b) is therefore trained anew and is not the model of " + LSTM_REF + "." if same_days else ""}
Scores are the median NSE of the {n} basins in the test years, averaged over the {_word(n_seeds)} seeds (not the score of the averaged forecast, as
in {LSTM_REF}); the counts of basins use the averaged forecast. With {_word(n_seeds)} seeds per variant and no p-value stored, the rule of section 4.2
cannot be applied; the rule of this experiment&rsquo;s results file is used instead: a change is called <i>clear</i> only if it exceeds twice the seed
spread and all seeds of the variant lie on the same side of all seeds of the base. By chance alone this rule fires in {chance_txt} of comparisons
(about {n2(expected, 0)} of the {n_cmp} made here; {n_flag} are clear), so a pattern over lead times counts and a single clear change does not.</p>"""

    # ---- (a) rain source ------------------------------------------------------------------------------------------------------
    sun = X["daymet_sunlight_on_rain_days"]["ratio_wet_to_dry_median"]
    nosun_same = all(_clear(ca["imerg_nosun_w"][f"l{k}"]["nse_median"]) == _clear(ca["imerg_w"][f"l{k}"]["nse_median"])
                     and abs(ca["imerg_nosun_w"][f"l{k}"]["nse_median"]["difference"] - ca["imerg_w"][f"l{k}"]["nse_median"]["difference"])
                     <= ca["imerg_w"][f"l{k}"]["nse_median"]["seed_noise"] for k in LEADS)
    control = (f"A control without Daymet&rsquo;s sunlight, which is estimated partly from whether rain was reported (here {pct(1 - sun)} lower on rain days) "
               f"and could give the rain days away, gives " + ("the same result." if nosun_same else "a different result (yellow in the figure)."))
    either = "" if any(_split(ca["imerg_w"])[:2]) else " either"
    found_a = f"""<p><b>(a) Without the river gauge,</b> satellite rain instead of gauge-based rain {_nse_story(ca["imerg_w"])} Satellite rain added to gauge-based rain {_nse_story(ca["gauge_imerg_w"], short=either)} {control}</p>
<p><b>(a) With the river gauge,</b> satellite rain instead of gauge-based rain {_nse_story(ca["imerg_wq"])} Satellite rain added to gauge-based rain
{_nse_story(ca["gauge_imerg_wq"])}</p>"""
    cap_skill = (f"<b>{FIG_SKILL}. Change of the median NSE when an input is replaced or added.</b> Top: (a) rain source; bottom: (b) extra inputs; left "
                 f"without, right with the river gauge. A filled marker is a clear change: {flags['a_rain_source'][0]} of {flags['a_rain_source'][1]} in the top "
                 f"row, {flags['b_extra_inputs'][0]} of {flags['b_extra_inputs'][1]} in the bottom row. Basin counts in the text use the forecast averaged over "
                 f"the seeds.")
    skill_figure = floated(fig("satellite_inputs.png", cap_skill, SKILL_SHARE))

    # ---- why satellite rain can beat rain gauges: the time stamp ---------------------------------------------------------------
    nd = {k: X["what_inputs_know_about_next_daymet_day"][name]["correlation_with_daymet_rain_of_the_next_day"] for k, name in NEXT_DAY.items()}
    gain_any = any(_split(ca[key])[0] for key in ("imerg_w", "imerg_wq", "gauge_imerg_w", "gauge_imerg_wq"))
    knows_more = nd["satellite"] > nd["gauge"]
    stamp = (f"Satellite rain of the issue day t holds the rain up to the end of the UTC day, the early evening of that day, part of which the rain "
             f"table books on day t+1: it correlates "
             f"{n2(nd['satellite'])} with the rain table&rsquo;s rain of day t+1, the rain table&rsquo;s own rain of day t only {n2(nd['gauge'])}.")
    tf = timing_facts(R)                                                             # the direct test of section 4.5, once it has run
    if tf:
        r1 = tf["gauge"][LEADS[0]]
        on = (f"lifts the onsets warned {LEAD_WORD[LEADS[0]]} with the river gauge from {pct(r1['onset_warned_base'])} to {pct(r1['onset_warned_with_nldas'])} "
              f"(control {pct(r1['onset_warned_control'])})" if onset_clear(r1) else
              f"takes the onsets warned {LEAD_WORD[LEADS[0]]} with the river gauge from {pct(r1['onset_warned_base'])} to {pct(r1['onset_warned_with_nldas'])}, "
              f"not clear of the seeds")
        nse = {"clear": f"and the median NSE from {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}, in {r1['basins_better']} of {tf['n']} basins",
               "mixed": f"and the median NSE from {n2(r1['nse_base'])} to {n2(r1['nse_with_nldas'])}, though in only {r1['basins_better']} of {tf['n']} basins",
               "none": "with no clear difference in NSE"}[nse_reading(r1, tf["n"])]
        direct = f"{TIMING_REF} tests the timing directly, with rain of another product stamped by UTC day and a control: there such rain {on}, {nse}."
    else:
        direct = f"{TIMING_REF} tests the timing directly, with rain of another product stamped by UTC day and a control."
    if gain_any and knows_more:
        why = f"""<p><b>Why would a satellite beat rain gauges?</b> Probably not because its rain is better: the agreement above is moderate. The likely reason
is the time stamp. {stamp} This experiment cannot separate better timing from better amounts; {direct}</p>"""
    elif knows_more:
        why = f"""<p><b>A catch in the comparison: the time stamp.</b> {stamp} A satellite input can therefore gain from its later day boundary as well as
lose from its rain amounts, and this experiment cannot separate the two; {direct}</p>"""
    else:
        why = ""
    caveat = (f'<div class="note blk"><b>Not a real-time result.</b> {late} The experiment shows what satellite rain with an exact '
              f"time stamp is worth, not what a real-time warning system would get. Temperature and humidity also still came from weather stations.</div>")

    # ---- (b) soil wetness -------------------------------------------------------------------------------------------------------
    soil_w_up, soil_wq_up = _split(cb["soil_w"])[0], _split(cb["soil_wq"])[0]
    if soil_w_up and not soil_wq_up:
        soil_why = "A likely reason, which the experiment does not prove: the river itself already tells the model how wet the basin is."
    elif not soil_w_up and not soil_wq_up:
        soil_why = "So the expected pattern &ndash; more help where the river gauge is missing &ndash; is not clear in these runs."
    else:
        soil_why = "So soil wetness does not help only where the river gauge is missing, which was not expected."
    soil_catch = (f"The catch: reanalysis soil wetness is a UTC-day value too, and the day-to-day change of its top layer correlates {n2(nd['soil'])} with the "
                  f"rain table&rsquo;s rain of the next day. Part of the gain {LEAD_WORD[1]} may again be the time stamp, not water stored in the soil."
                  if nd["soil"] > nd["gauge"] and 1 in soil_w_up + soil_wq_up else "")
    all_notes = " ".join(X.get("notes", []))
    soil_source = ("The soil wetness is no independent measurement either: the reanalysis&rsquo; land model computes it from rain that is corrected with "
                   "rain gauges" + (", and it is published weeks after the fact." if "published weeks after the fact" in all_notes else ".")
                   if "not independent of rain gauges" in all_notes else "")
    soil = f"""<p><b>(b) Soil wetness.</b> Without the river gauge, soil wetness {_nse_story(cb["soil_w"])} With the river gauge it {_nse_story(cb["soil_wq"])}
{soil_why} {soil_catch} {soil_source}</p>"""

    # ---- (b) precipitable water and the link to GNSS ----------------------------------------------------------------------------
    pw_keys = {"pw_w": "without the river gauge", "pw_wq": "with the river gauge"}
    pw_nse = [(key, k) for key in pw_keys for k in LEADS if _clear(cb[key][f"l{k}"]["nse_median"])]
    pw_on = [(key, k) for key in pw_keys for k in LEADS if _clear(cb[key][f"l{k}"]["onset_pod"])]
    n_pw = len(pw_keys) * len(LEADS)
    on_txt = lambda key, k: (f"{pw_keys[key]}, {LEAD_WORD[k]}, from {pct(cb[key][f'l{k}']['onset_pod']['base'])} to {pct(cb[key][f'l{k}']['onset_pod']['variant'])} "
                             f"(false warnings {pct(cb[key][f'l{k}']['onset_far']['base'])} &rarr; {pct(cb[key][f'l{k}']['onset_far']['variant'])})")
    pw_changes = [_chg(cb[key][f"l{k}"]["nse_median"]["difference"]) + f" {pw_keys[key]}, {LEAD_WORD[k]}" for key, k in pw_nse]
    nse_txt = (f"precipitable water changed the median NSE clearly in {_word(len(pw_nse))} of the {_word(n_pw)} comparisons"
               + (f" ({_and(pw_changes)})" if pw_changes else ""))
    if not pw_on:
        onset_txt = "The share of onsets warned did not change clearly in any of them either."
        weigh = "So this test shows no effect of water vapour."
    else:
        moved = [cb[key][f"l{k}"]["onset_pod"]["difference"] for key, k in pw_on]
        verb = "rose" if all(m > 0 for m in moved) else "fell" if all(m < 0 for m in moved) else "changed"
        onset_txt = f"The share of onsets warned {verb} clearly in {_word(len(pw_on))}: {'; '.join(on_txt(key, k) for key, k in pw_on)}."
        weigh = ("One clear change among so many comparisons is weak evidence: a hint to follow up, not a result." if len(pw_on) == 1 else
                 "That is a pattern worth following up, at a low level of warnings.")
    k1 = LEADS[0]
    wide = [(key, cb[key][f"l{k1}"]) for key in pw_keys if not _clear(cb[key][f"l{k1}"]["nse_median"])
            and (cb[key][f"l{k1}"]["n_basins_improved"] >= cb[key][f"l{k1}"]["n_basins"] - 1) and cb[key][f"l{k1}"]["nse_median"]["difference"] > 0]
    wide_txt = ""                                                                    # a count near 12 of 12 carries weight even where the median is inside the seed noise
    if wide:
        key, c = wide[0]
        again = (" Precipitable water is a UTC-day value too, so this may again be the time stamp." if nd["vapour"] > nd["gauge"] else "")
        wide_txt = (f" One count stands out: {pw_keys[key]}, {LEAD_WORD[k1]}, the forecast averaged over the seeds is higher in {c['n_basins_improved']} "
                    f"of {c['n_basins']} basins, though the change of the median ({_chg(c['nse_median']['difference'])}) is inside the seed noise.{again}")
    vapour = f"""<p><b>(b) Water vapour, and the link to GNSS.</b> A GNSS receiver measures how long the atmosphere delays the satellite signals. The part of
that delay due to water vapour, counted straight upward (the zenith wet delay), gives the precipitable water above the station once the surface
pressure and temperature are known {cite("bevis1992")}. What was tested here is not such a measurement but reanalysis output: the daily mean of the water vapour of a weather model on a grid
of about {n2(5 * round(coarse_km[0] / 5), 0)} km. In this test {nse_txt}. {onset_txt} {weigh}{wide_txt} <b>Real GNSS-derived water vapour, at its
native sampling of 5 to 30 minutes, was not tested.</b></p>"""

    return f"""{LAYOUT}<h3 class="clr">6.2 Rain from space, wet soil and water vapour</h3>
<p>Most rivers of the world have few rain gauges, but satellites see rain everywhere. Can satellite rain replace rain gauges, or add to them? And do
soil wetness and the water vapour in the air make the forecast better?</p>
{data}
{agreement}
{rain_figure}{whose_day}
{method}
{skill_figure}{found_a}
{why}
{soil}
{vapour}
{caveat}"""
