"""Sections 8 and 9 - the conclusions with their evidence, what the project teaches about applying AI, how sure the numbers are, what
was done and found step by step (Table 8.1), the limits and the next steps; then who did what, what each member still has to check,
the four things every member must be able to explain, how to reproduce, and the sources.

Reads the results files by key (a conclusion or a limit whose results file does not exist is left out or says so), PLAN.md for the
roles, team.json (through code/team.py) for the names and the state of the members' own checks, and the file system for 'what exists
now'. Nothing is computed beyond differences, counts and formatting. The conclusions are built from the sentences of page 1 and of
sections 4 and 5 (s0_summary.facts and its helpers), so that no result is worded twice; every heading that states an outcome is chosen
from the stored numbers and has a neutral form for the other outcome.
Three results files of 4 October 2026 are read inside guards - uncertainty.json (ranges when the test years and the basins are drawn
again, the stability of the verdicts, the spread of the forecasts), same_misses.json (do the kinds of model miss the same onsets) and,
through s0_summary, key_figure.json: where a file or a key is missing the sentence is left out and one line goes to stderr.
What page 1 has no room for stands here: the reading guide and the line of milestones (s0_summary.page1_plan says which).
The source list is report_common.reference_list(): everything the earlier sections cited through cite(), so this module must
be built last (build_report.py does so).
"""
import re
import sys

import team
from common import ROOT
from flood import HIGH_Q, LEADS
from report_common import CITED, LEAD_WORD, REFERENCES, cite, count, n2, pct, reference_list, signed, table
from report_sections.s0_summary import (AORC_WORDING, AS_MUCH, FINDING, INFO_NOT_NET, LOBO_REF, MOVED_POINT, REALTIME_WORDING, SUPPORT, TIMING_REF, _dig, _esc, _file, aorc_facts,
                                         by_lead_reading, day_windows, era5_facts, facts, forecast_facts, four_experiments, hindsight_clause, hindsight_facts, lobo_facts,
                                         new_work, note, nse_forecast_words, onset_clear, page1_plan, reading_guide, rivers_facts, run_day, share_word, stage4_facts,
                                         timing_facts, timing_sentence, timing_verdict, unseen_sentence, who)
from report_sections import s4d_midnight, s4e_rivers, s4f_forecast
from report_sections.s7_cost import _project_cost

QUOTED, QUOTED_CAP = "&lsquo;tomorrow = today&rsquo;", "&lsquo;Tomorrow = today&rsquo;"
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
# What section 8 gives up when the whole report is over its page limit: 0 nothing; 1 Table 8.1 (one sentence that lists the steps
# stays); 2 also the reading guide moved here from page 1 - the project lead's order. 3: also, the three limits that sections 5, 6.2 and 7
# state where they arise are given as one line of pointers (not in the lead's order: used only if build_report.CUT_WORDS has a level 3).
# build_report.py raises CUTS step by step, and only as far as needed, when the printed report has more pages than its limit.
CUTS = 0
STYLE = ("<style>.s8 li{margin-bottom:2pt} .s8 table.s8t th,.s8 table.s8t td{padding:2pt 5pt;line-height:1.3} .s8 table.s8t{font-size:8.3pt}"
         ".s8 .refs{column-count:2;column-gap:16pt} .s8 .refs li{break-inside:avoid} .s8 code{font-size:7.9pt}"
         ".s8 .refs2{display:grid;grid-template-columns:1fr 1fr;column-gap:16pt;align-items:start;clear:both} .s8 .refs2 .refs{column-count:1;margin:0}"
         ".s8 ol.concl li{margin-bottom:3pt} .s8 .ev{color:var(--ink2)}"
         ".s8 table.t81 th,.s8 table.t81 td{padding:1.3pt 4pt;line-height:1.26} .s8 table.t81{font-size:8.2pt} .s8 table.t81 td:first-child{white-space:nowrap}"
         ".s8 .guide{font-size:8.6pt;color:var(--ink2)}</style>")       # layout only
lk = lambda k: f"l{k}"
dash = lambda c: str(c).replace("-", "&ndash;")

# what the plan gives each member (matched to the 'Member' column of the Roles table in PLAN.md): the files that should exist, and
# the member's own check before the hand-in. The checks are instructions, with their numbers read from the results files in html().
FILES = {
    "Member 1": ["code/01_data_audit.py", "results/data_audit.json", "code/build_report.py", "Flood_Report.pdf"],
    "RS/GIS member 2": ["code/07_basin_maps.py", "results/basin_gis.json", "code/08_satellite_inputs.py", "results/satellite_inputs.json"],
    "GNSS member 1": ["code/flood.py", "code/test_flood.py", "results/fair_test.json", "code/06_flood_analysis.py", "results/flood_analysis.json"],
    "GNSS member 2": ["code/03_lstm.py", "results/hparam_log.csv", "code/09_model_comparison.py", "results/model_comparison.json"],
}
# the scripts in the order they must run (the order of run_all.sh: each step after what it reads), with what each does; the time is
# read from the results file named (None: a light script, or one that only reads stored results)
STEPS = (("code/01_data_audit.py", "the 12 basins", None), ("code/test_flood.py", "the fairness checks", None),
         ("code/02_baselines.py", "the simple models", "baselines"), ("code/03_lstm.py", "one river", "lstm_single"),
         ("code/04_lstm_all.py", "all twelve rivers", "lstm_all"), ("code/05_lobo.py", "a river never seen", "lstm_lobo"),
         ("code/07_basin_maps.py", "maps and GIS", "basin_gis"), ("code/fetch_satellite.py", "satellite data, cached", None),
         ("code/10_rain_timing.py", "rain timing", "rain_timing"), ("code/06_flood_analysis.py", "the warnings", "flood_analysis"),
         ("code/17_more_basins.py", "63 training rivers instead of 12", "more_basins"),
         ("code/check_2004_floods.py", "the storms of September 2004", None),
         ("code/08_satellite_inputs.py", "satellite inputs", "satellite_inputs"),
         ("code/09_model_comparison.py", "Transformer against LSTM", "model_comparison"),
         ("code/check_day_windows.py", "the day windows against hourly rain, cached", None),
         ("code/check_onset_rain_clock.py", "when the rain of the onsets falls", "onset_rain_clock", "*"),
         ("code/15_rain_to_midnight.py", "the rain of the whole issue day, ERA5", "rain_to_midnight"),
         ("code/16_aorc_rain_to_midnight.py", "the same with AORC", "aorc_rain_to_midnight"),
         ("code/check_aorc_hindsight.py", "does AORC hold hindsight", "aorc_hindsight", "*"),
         ("code/check_stage4_realtime.py", "Stage IV in place of AORC", "stage4_realtime", "*"),
         ("code/gefs_to_basins.py", "the forecast rain as basin means", "gefs_basins", "*"),
         ("code/18_forecast_input.py", "a weather forecast as an input", "forecast_input"),
         ("code/12_training_curves.py", "the learning curves, from the stored trainings", None),
         ("code/13_uncertainty.py", "the ranges of the headline numbers", None),
         ("code/check_same_misses.py", "whether the kinds of model miss the same onsets", None),
         ("code/11_key_figure.py", "the key figure", None),
         ("code/build_report.py", "this report and short_report.md", None), ("code/build_slides.py", "the slides", None),
         ("code/build_qa.py", "the question-and-answer sheet", None), ("code/build_overview.py", "the one-page overview", None))
SECONDS = ("training_seconds", "fit_seconds", "runtime_s")       # the first of these that the results file has


def _word(n):
    return NUMBER.get(n, str(n))


def _and(parts):
    parts = [str(p) for p in parts]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _leads(ks):
    return LEAD_WORD[ks[0]] if len(ks) == 1 else _and(ks) + " days ahead"


def _cap(s):
    return s[0].upper() + s[1:]


def _plan():
    """(roles, decisions) from PLAN.md: the rows of its Roles table and the numbered list of decisions that were open when it was written."""
    text = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    part = lambda title: re.search(rf"## {title}\n(.*?)(?=\n## |\Z)", text, re.S).group(1)
    roles = [tuple(c.strip() for c in line.strip().strip("|").split("|")) for line in part("Roles").splitlines()
             if line.startswith("|") and not set(line) <= set("|- ") and "Member" not in line]
    decisions = [re.sub(r"^\d+\.\s*", "", line).rstrip(".") for line in part("Decisions still needed from the group").splitlines() if re.match(r"\d+\.", line)]
    return roles, decisions


def _references():
    """reference_list() of everything cited so far, with the letter of a short form such as 'Kratzert et al., 2019a' carried into the
    full reference ('(2019)' -> '(2019a)'), so that the reader can tell the two papers of one year apart; set in two columns of about
    equal length (split by the length of the text: the browser's own column balancing leaves one column clearly shorter when no
    reference may be split). Formatting only."""
    out = reference_list()
    for key in CITED:
        short, full = REFERENCES[key]
        m = re.search(r"(\d{4})([a-z])$", short)
        if m:
            out = out.replace(full, full.replace(f"({m.group(1)})", f"({m.group(1)}{m.group(2)})", 1))
    items = re.findall(r"<li>.*?</li>", out, re.S)
    if len(items) < 4 or "".join(items) not in out:                             # too few to split, or the list is not in the expected form
        return out
    size = [len(re.sub(r"<[^>]+>|&[a-z]+;", "x", x)) + 40 for x in items]         # the text, plus the gap between two references
    cut = min(range(1, len(items)), key=lambda i: abs(sum(size[:i]) - sum(size[i:])))
    column = lambda part: '<ul class="refs">' + "".join(part) + "</ul>"
    return f'<div class="refs2">{column(items[:cut])}{column(items[cut:])}</div>'


def _minutes(R, key):
    d = R.get(key)
    if not isinstance(d, dict) or d.get("quick_mode"):
        return None
    secs = next((d[k] for k in SECONDS if isinstance(d.get(k), (int, float))), None)
    return None if secs is None else secs / 60


# ---- the three guarded results files of 4 October 2026 ------------------------------------------------------------------------
def _uncertainty(R):
    """results/uncertainty.json (13_uncertainty.py) when it can be quoted, else None with one line on stderr."""
    U = R.get("uncertainty")
    if isinstance(U, dict) and not U.get("quick_mode") and U.get("stats"):
        return U
    note("s8_limits: results/uncertainty.json is missing or a smoke test (code/13_uncertainty.py writes it): section 8 has no paragraph "
         "'How sure are these numbers?' and gives no reason for the low peaks")
    return None


def _how_sure(U):
    """The paragraph 'How sure are these numbers?': the range of each headline number when the test years and the basins are drawn again
    (stats[name].value and .both.lo / .hi / .share_above_zero) and how often the verdicts hold (verdict_stability), all 1 day ahead unless
    said. '' with one line on stderr if the file is not in the expected form."""
    first = LEADS[0]
    try:
        S, VS, RS = U["stats"], U["verdict_stability"], U["resampling"]
        B = int(RS["B"])
        both = lambda name: S[name]["both"]
        rng = lambda name, fmt=pct: f"{fmt(both(name)['lo'])} to {fmt(both(name)['hi'])}"
        on = lambda s: f"{pct(S[f'onsets_warned.{s}.{lk(first)}']['value'])} ({rng(f'onsets_warned.{s}.{lk(first)}')})"
        how = (f"Each range below is the range when the test years and the basins are drawn again (95%): the {RS['n_years']} test years and the "
               f"{RS['n_basins']} basins were drawn again {count(B)} times from the stored forecasts (<code>results/uncertainty.json</code>).")
        onsets = (f"{_cap(LEAD_WORD[first])} with the river gauge the LSTM warns of {on('lstm_wq')} of the onsets; given rain stamped up to the evening "
                  f"of the issue day as well, of {on('lstm_wqn')}; given the perfect weather forecast, an upper bound that cannot be had in practice, "
                  f"of {on('lstm_wqf')}.")
        gap_name, gain_name = f"onsets_gap_share_closed.wq.{lk(first)}", f"onsets_gain.evening_rain_minus_control.wq.{lk(first)}"
        above = round(both(gain_name)["share_above_zero"] * B)
        gap = (f"The rain up to the evening closes {share_word(S[gap_name]['value'])} of that gap ({rng(gap_name)}), and its gain over the control "
               f"is above zero in {'every one' if above == B else count(above)} of the {count(B)} draws.")
        d = [f"nse_median_of_diffs.lstm_wq_minus_best_simple.{lk(k)}" for k in LEADS]
        sg = lambda v: "0.00" if round(v, 2) == 0 else signed(v, 2)             # two decimals, as section 4.2 prints the same numbers;
                                                                                # a value that rounds to zero carries no sign
        small = all(both(x)["lo"] < 0 < both(x)["hi"] and max(abs(both(x)["lo"]), abs(both(x)["hi"])) < 0.05 for x in d)
        tie = ("With the river gauge, the LSTM&rsquo;s NSE minus that of the best simple model is, in the typical basin, "
               + _and(f"{sg(S[x]['value'])} ({rng(x, sg)}) {LEAD_WORD[k]}" for x, k in zip(d, LEADS))
               + (": a few hundredths at most, in either direction." if small else "."))
        share = lambda cell, v: VS[cell]["share_of_resamples_with_each_verdict"].get(v, 0.0)
        on_cells, a1 = [f"lstm_wq.onset.{lk(k)}" for k in LEADS], f"lstm_wq.all.{lk(first)}"
        parts = []
        if len({VS[c]["verdict"] for c in on_cells}) == 1:
            v = VS[on_cells[0]]["verdict"]
            every = all(share(c, v) == 1.0 for c in on_cells)
            rain = [f"lstm_wqn.onset.{lk(k)}" for k in LEADS]
            also = every and all(c in VS and VS[c]["verdict"] == v and share(c, v) == 1.0 for c in rain)
            parts.append(f"for the model with the river gauge the verdict {v} for onsets holds in " + ("every draw" if every else f"{pct(min(share(c, v) for c in on_cells))} or more of the draws")
                         + " at every lead time" + (", also with rain up to the evening of the issue day" if also else ""))
            newer = [(f"lstm_wqa.onset.{lk(first)}", "rain up to midnight from AORC"), (f"lstm_wqaF_sub.onset.{lk(first)}", "the weather forecast beside it")]
            held = [(VS[c]["verdict"], share(c, VS[c]["verdict"]), name, share(c, "GO")) for c, name in newer if c in VS]
            if held:                                                    # the sets added on 7-8 October 2026: their verdict is less stable (results/uncertainty.json)
                parts.append("with " + _and(f"{name} the {vv} for onsets {LEAD_WORD[first]} holds in {pct(sh)} of the draws" for vv, sh, name, go in held)
                             + (" (no draw reaches GO)" if all(go == 0 for *_, go in held) else ""))
        va = VS[a1]["verdict"]
        parts.append(f"the {va} for all high-flow days {LEAD_WORD[first]} holds in {pct(share(a1, va))} of the draws ("
                     + ", ".join(f"{x} in {pct(share(a1, x))}" for x in ("GO", "CAUTION", "NOT YET") if x != va) + ")")
        chosen = lambda c: (VS[c].get("at_level_chosen_on_validation_years") or {}).get("share_of_resamples_with_each_verdict")
        level = ("Warnings are counted at the high-flow level itself (for the model with the river gauge, the warning level of Table 5.1 gives the same shares)."
                 if all(chosen(c) == VS[c]["share_of_resamples_with_each_verdict"] for c in on_cells + [a1]) else
                 "Warnings are counted at the high-flow level itself.")
        verdicts = _cap("; ".join(parts)) + ". " + level
        newsets = ""
        ka, kf2, kf3 = f"onsets_warned.lstm_wqa.{lk(first)}", f"onsets_warned.lstm_wqaF_sub.{lk(LEADS[1])}", f"onsets_warned.lstm_wqaF_sub.{lk(LEADS[2])}"
        if all(k in S for k in (ka, kf2, kf3)):                                    # the ranges of the sets of sections 4.6 and 4.8 (13_uncertainty.py), where they exist
            one = lambda k: f"{pct(S[k]['value'])} ({rng(k)})"
            newsets = (f" Given the AORC rain to midnight the LSTM warns of {one(ka)} of the onsets {LEAD_WORD[first]}; given a weather forecast beside it, of {one(kf2)} and {one(kf3)} "
                       f"{_leads(LEADS[1:])}.")
    except (KeyError, TypeError, ValueError) as e:
        note(f"s8_limits: results/uncertainty.json is not in the expected form ({type(e).__name__}: {e}): the paragraph 'How sure are these numbers?' is left out")
        return ""
    return (f"<p><b>How sure are these numbers?</b> {how} {onsets} {gap}{newsets} {tie} {verdicts} These ranges are not the seed spread, the difference "
            f"between two trainings of one network, and they say nothing about other regions.</p>")


def _why_low_peaks(U):
    """Why the peaks are forecast too low: the spread of the forecasts against their correlation with the observed flow
    (uncertainty.json: spread_and_skill.medians, the median of the basins). '' with one line on stderr if the table is not there."""
    first = LEADS[0]
    try:
        M = U["spread_and_skill"]["medians"]
        sp = lambda s: [M[f"spread_ratio_median.{s}.{lk(k)}"] for k in LEADS]
        co = lambda s: [M[f"correlation_median.{s}.{lk(k)}"] for k in LEADS]
        near = lambda s: all(a < 1 and abs(a - b) <= 0.1 for a, b in zip(sp(s), co(s)))        # a spread of about the correlation, at every lead time
        lstm_sp, lstm_co, up = sp("lstm_wq"), co("lstm_wq"), sp("lstm_wqf")[0]
        simple = [s for s in ("base_rf_wq", "base_linreg_wq") if f"spread_ratio_median.{s}.{lk(first)}" in M]
    except (KeyError, TypeError) as e:
        note(f"s8_limits: results/uncertainty.json has no table spread_and_skill.medians ({type(e).__name__}: {e}): the limit 'Peaks are forecast too low' gives no reason")
        return ""
    three = lambda v: ", ".join(n2(x) for x in v[:-1]) + f" and {n2(v[-1])}"
    same = (" (the random forest and the ridge regression show the same pattern)" if len(simple) == 2 and all(near(s) for s in simple) else "")
    out = (f" Why: it is what the squared error asks for. A forecast that minimises the squared error must hedge when it cannot know whether the rain "
           f"will come, and the best hedge has a spread of about its correlation with the observed flow times the observed spread. "
           f"{'The forecasts behave that way' if near('lstm_wq') else 'Here'}: with the river gauge the LSTM&rsquo;s forecasts have {three(lstm_sp)} of the "
           f"observed spread {_leads(LEADS)}, against correlations of {three(lstm_co)}{same}.")
    if up > lstm_sp[0] + 0.05:
        out += (f" Given the perfect weather forecast the spread is {n2(up)} of the observed one {LEAD_WORD[first]}. So, among the models tried, the low "
                f"peaks go with the missing information, and they are a reason to train a warning model on another target (the last of the next steps).")
    return out


def _same_misses(R):
    """Do the kinds of model miss the same onsets (results/same_misses.json, check_same_misses.py), 1 day ahead, as one short paragraph of
    sentences; '' with one line on stderr while the file is missing. The pairwise shares are not quoted (the file says why: most onsets
    are missed by every model, so part of any pairwise agreement is the base rate)."""
    M, first = R.get("same_misses"), LEADS[0]
    try:
        if not isinstance(M, dict) or M.get("quick_mode"):
            raise KeyError("no results file")
        L, kinds = M[lk(first)], list(M["kinds"])
        missed, any_, more = L["missed_by_all_kinds"], L["warned_by_any_kind"], L["of_missed_by_all"]
        up, rain = more["warned_with_perfect_forecast"]["share"], more["warned_with_rain_to_evening"]["share"]
        seeds = len(M["prediction_sets"][M["kinds"]["Transformer"]]["seeds"]) if "Transformer" in M["kinds"] else 0
        level = ("at the high-flow level itself" if float(M["warning_level"]) == 1.0 else f"at {n2(float(M['warning_level']))} times the high-flow level")
    except (KeyError, TypeError, ValueError) as e:
        note(f"s8_limits: results/same_misses.json is missing or not in the expected form ({type(e).__name__}: {e}; code/check_same_misses.py writes it): "
             f"conclusion 1 has no sentence on whether the models miss the same onsets")
        return ""
    return (f" {'The models also miss the same onsets. ' if missed['share'] > 0.5 else ''}{_cap(LEAD_WORD[first])} with the river gauge, "
            f"{pct(missed['share'])} of the {count(L['n_onsets'])} onsets are missed by all {_word(len(kinds))} kinds of model ({', '.join(kinds)}), and a "
            f"warning from any one of them would cover {pct(any_['share'])}, with {pct(any_['far'])} false warnings. Of the onsets that all of them miss, the "
            f"LSTM given the weather of the days being forecast, an upper bound that cannot be had in practice, warns of {pct(up)}; given rain stamped up to "
            f"the evening of the issue day, of {pct(rain)}. (Warnings {level}"
            + (f"; the Transformer is the {_word(seeds)}-seed model of section 7" if seeds else "") + ".)")


def html(R, T):
    first, last = LEADS[0], LEADS[-1]
    A, B, F, S1, ER, FT = R["lstm_all"], R["baselines"], R["flood_analysis"], R["lstm_single"], R["engine_review"], R["fair_test"]
    C, LV, ST = A["comparison"], A["decomposition"]["levels"], A["decomposition"]["steps"]
    n = len(A["basins"])
    best = lambda inp, k: C[inp]["best_baseline_by_validation"][lk(k)]
    cmp_ = lambda inp, k: C[inp]["baselines"][best(inp, k)][lk(k)]
    tf, lf = timing_facts(R), lobo_facts(R)
    f = facts(R, T)                                                                  # the sentences of page 1: the conclusions use them word for word
    plan = page1_plan(R, T)                                                          # which optional parts page 1 holds; the others stand here
    U = _uncertainty(R)
    three = lambda vals, fmt=n2: ", ".join(fmt(v) for v in vals[:-1]) + f" and {fmt(vals[-1])}"
    fc = ST["perfect_forecast_with_gauge"]
    cw = cmp_("wq", first)
    spread = cw["lstm_seed_median_max"] - cw["lstm_seed_median_min"]
    n_seeds = len(A["sets"]["wq"]["seeds"])
    on0, on_up = LV["wq"][lk(first)]["onset_pod"], LV["wqf"][lk(first)]["onset_pod"]
    V = F["verdict"]["services"]["lstm_wq"][lk(first)]
    on_verdicts = {F["verdict"]["services"]["lstm_wq"][lk(k)]["onset"]["pooled"] for k in LEADS}
    rt = F["failures"]["by_rain_timing"]["lstm_wq"]["tuned"]
    bins = list(rt)
    r1 = tf["gauge"][first] if tf else None
    timing_clear = bool(tf and onset_clear(r1))
    has_cmp = "model_comparison" in R and f["acc"] is not None

    # ---- 8 the conclusions, each with its evidence -------------------------------------------------------------------------
    concl, no = [], {}                                                               # no: the number of a conclusion, for the pointers in the limits
    # 1 the kind of model
    cls = f["cls"]
    no_clear = (f["gauge_tie"] and all(v in ("tie", "lstm_med", "lstm_avg") for v in cls["w"].values())
                and (not has_cmp or f["acc"] == "no clear difference in accuracy"))
    t1 = f"{f['a'][0]} {f['a'][1]} {f['tie']}"
    if has_cmp:
        t1 += f" A Transformer in place of the LSTM: {f['acc']}; training it takes {f['t_lo']:.0f} to {f['t_hi']:.0f} times as long."
    t1 += _same_misses(R)
    concl.append(("With the inputs that can be had in practice, the kind of model made no clear difference." if no_clear else
                  "With the inputs that can be had in practice: how the kinds of model compare.", t1,
                  "section 4.2 (Figure 4.1, Tables 4.1 and 4.2)" + (", section 7 (Figure 7.1)" if has_cmp else "")))
    # 2 the information
    SB = {(r["model"], r["lead"]): r["nse_median"] for r in B["summary_table"]}
    fc_simple = [SB[f"base_{m}_wqf", first] - SB[f"base_{m}_wq", first] for m in ("linreg", "rf")
                 if (f"base_{m}_wqf", first) in SB and (f"base_{m}_wq", first) in SB]
    fc_gain = [fc[lk(k)]["nse_median_gain"] for k in LEADS]
    lifted = f["all_gain"] and min(fc_gain) > 0 and len(fc_simple) == 2 and min(fc_simple) > 0
    t2 = (f"{_cap(f['gauge_adds'])}; the weather of the days being forecast, an upper bound, would add {three(fc_gain, signed)} NSE to the LSTM with the "
          f"river gauge ({_leads(LEADS)}"
          + (f"; {signed(min(fc_simple))} to {signed(max(fc_simple))} to the two simple models {LEAD_WORD[first]}" if len(fc_simple) == 2 else "")
          + f") and lift its onsets warned {LEAD_WORD[first]} from {pct(on0)} to {pct(on_up)}. {f['a'][2]}"
          + (" The Transformer was run with one set of inputs only, so it is not part of this comparison." if has_cmp else "")
          + f" Not tried: an LSTM with more than {A['config']['hidden']} memory cells, another loss, a search over the Transformer&rsquo;s size.")
    concl.append(("More information lifted every kind of model." if lifted else "What more information changed.", t2,
                  "section 4.3 (Table 4.3), section 2 (Tables 2.1 and 2.2)" + (", the key figure on page 1" if f["key"] else "")))
    # 3 rain that had already fallen
    if tf:
        no["timing"] = len(concl) + 1
        sat = ("; section 6.2 (satellite rain: probably the same time-stamp effect)"
               if any("probably a time-stamp effect" in str(r[2]) for r in f["rows"]) else "")
        concl.append(("Part of the missing information was rain that had already fallen." if timing_clear else
                      "Rain that had already fallen: no clear gain was shown.",
                      f"The rain table seems to stamp rain by a day that ends in the morning (an inference from our own data). {timing_sentence(R)}"
                      + (" The rest needs later rain, most of it after the forecast is issued." if timing_clear else ""),
                      f"{TIMING_REF} (Table 4.5, Figures 4.2 and 4.3){sat}"))
    AO, E1, S4, HS, FC, RV = aorc_facts(R), era5_facts(R), stage4_facts(R), hindsight_facts(R), forecast_facts(R), rivers_facts(R)
    k2, k3 = LEADS[1], LEADS[2]
    n1, s1 = (lambda v: n2(v, 1)), (lambda v: signed(v, 1))
    slash = lambda d: "/".join(pct(d[k]) for k in LEADS)
    # 4 the warnings
    head4 = f["warn"].replace("Its warnings", "The warnings") + "."
    t4 = f["b"][0].replace("The warnings: ", "By criteria fixed before the analysis: ", 1)
    top, low = rt[bins[3]]["onset"], rt[bins[0]]["onset"]
    if top["pod"] < low["pod"]:
        t4 += (f" Onsets are missed where the rain behind the rise is not yet in the model&rsquo;s inputs: of the {count(top['hits'] + top['misses'])} "
               f"onsets with {dash(bins[3])} of the two days&rsquo; rain stamped on the day being forecast, {pct(top['pod'])} are warned {LEAD_WORD[first]}; "
               f"of the {count(low['hits'] + low['misses'])} with {dash(bins[0])}, {pct(low['pod'])}.")
    concl.append((head4, t4, "section 5 (Table 5.1, Figures 5.1 to 5.3)"))
    # 5 a river never seen
    if lf:
        no["unseen"] = len(concl) + 1
        head5 = ("A river never seen in training is forecast clearly worse." if lf["clear"] else
                 "A river never seen in training tends to be forecast worse." if lf["loss"] > lf["seed_range"] else
                 "A river never seen in training: no clear loss.")
        prop = {"no clear difference": " Giving the model the basin properties makes no clear difference for such a river.",
                "help": " The basin properties help for such a river.", "hurt": " The basin properties hurt for such a river."}.get(lf["properties"], "")
        more = ""
        if RV:
            n11, n50 = RV["n_train"]["lobo_w_static"], RV["n_train"]["unseen50_w_static"]
            more = (f" More rivers help: with {n50} training rivers instead of {n11} the median NSE rises from {n2(RV['unseen']['lobo_w_static'][first])} to {n2(RV['unseen']['unseen50_w_static'][first])} "
                    f"(number, variety and nearness of the training rivers not separated).")
        t47, f47 = s4e_rivers.numbers(R)
        concl.append((head5, f"{_cap(unseen_sentence(R))}.{prop} Each held-out model of section 4.4 was trained once; {LOBO_REF} lists what else limits this test.{more}",
                      f"{LOBO_REF} (Table 4.4)" + (f", section 4.7 ({t47}" + (f", {f47}" if f47 else "") + ")" if RV else "")))
    # 6 the experiments of 7 and 8 October 2026: the finding holds in both directions (taking information away and giving it)
    if AO and FC and f["story"]:
        t46, f46 = s4d_midnight.numbers(R)
        t48, f48 = s4f_forecast.numbers(R)
        g_gauge = ST["river_gauge_without_forecast"][lk(first)]["nse_median_gain"]
        vall = set(AO["verdict_onset"].values()) | set(FC["verdict_onset"].values()) | set(FC["verdict_onset_chosen"].values())
        g = AO["gain"]
        as_much = min(g["ridge"], g["forest"]) >= AS_MUCH * g["lstm"]
        away = (f"Taking information away lowers the skill: without the river gauge the LSTM loses {n2(g_gauge)} NSE {LEAD_WORD[first]}"
                + (f", and a river never seen loses {n2(lf['loss'])} NSE with {RV['n_train']['lobo_w_static']} training rivers ({n2(RV['loss'][1])} with {RV['n_train']['unseen50_w_static']})"
                   if lf and RV else "") + ". ")
        era = (f"{pct(E1['with_'])} with ERA5 model rain (the same as the NLDAS rain; the success rule fixed in advance {'was met' if E1['met'] else 'was not met'}) and to " if E1 else "")
        hind = f"; {hindsight_clause(HS)}" if HS else ""
        real = (f"With Stage IV the share is {pct(S4['stage4'] / S4['n'])} ({pct(S4['no_last'] / S4['n'])} with the last hour of the issue day out): {REALTIME_WORDING}. " if S4 else "")
        ng = AO["nse_gain"]
        times = ng["lstm"] / max(ng["ridge"], ng["forest"]) if max(ng["ridge"], ng["forest"]) > 0 else None
        twice = ("about twice as much" if times and 1.6 <= times < 2.6 else "more than twice as much" if times and times >= 2.6 else "more" if times and times > 1.1 else "")
        in_nse = (f"; in median NSE the LSTM gains {twice} with the rain to midnight ({signed(ng['lstm'])} against {signed(ng['ridge'])} and {signed(ng['forest'])})" if twice else "")
        give = (f"Giving it raises the skill: the rain of the whole issue day up to midnight takes the onsets warned {LEAD_WORD[first]} from {pct(AO['base'][first])} to {era}{pct(AO['with_'][first])} "
                f"with AORC, {AORC_WORDING} (control {pct(AO['to09'])}{hind}). {real}A weather forecast beside the rain to midnight takes the onsets {_leads([k2, k3])} from "
                f"{pct(FC['rain'][k2])} and {pct(FC['rain'][k3])} without it to {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])} (one ensemble member; {nse_forecast_words(FC)}). "
                f"The simple models given the same inputs gain {'as many onsets' if as_much else 'fewer onsets'}" + (f": {INFO_NOT_NET}, for onsets" if as_much else "") + in_nse + ". ")
        kind = ("Changing the kind of model changes little. " if no_clear else "")
        concl.append(("The finding holds in both directions: taking information away lowered the skill, giving it raised it." if AO["met"] and FC["success"] else
                      "Taking information away and giving it: what the new inputs changed.",
                      away + give + kind + f"Yet with every input tried the verdict for the onsets is {_and(sorted(vall))} at every lead time ({pct(AO['false'][first])} of the warnings false "
                      f"{LEAD_WORD[first]} with the rain to midnight).",
                      f"section 4.6 ({t46}" + (f", {f46}" if f46 else "") + f"), section 4.8 ({t48}" + (f", {f48}" if f48 else "") + ")" + (", the key figure on page 1" if f["key"] else "")))
    concl_html = '<ol class="concl">' + "".join(f'<li><b>{h}</b> {t} <span class="ev">Evidence: {e}.</span></li>' for h, t, e in concl) + "</ol>"
    if f["story"]:
        opening = (f"<p><b>The conclusion.</b> {FINDING}: {SUPPORT[0].lower() + SUPPORT[1:]} ({'key figure, page 1' if f['key'] else 'section 4.3'}). "
                   f"{_cap(_word(len(concl)))} conclusions follow, each with its evidence; all numbers are for the test years ({f['test_years']}) and {n} basins.</p>")
    else:
        opening = (f"<p><b>The conclusions.</b> {f['head']} {_cap(_word(len(concl)))} conclusions follow, each with its evidence; all numbers are for the "
                   f"test years ({f['test_years']}) and {n} basins.</p>")

    # ---- what this teaches about applying AI: each lesson tied to a number of this project -----------------------------------
    pers = [C["wq"]["baselines"]["base_persistence"][lk(k)]["n_basins_lstm_higher"] for k in LEADS] if "base_persistence" in C["wq"]["baselines"] else None
    same_verdict = F["verdict"]["other_models_pooled"].get("base_persistence", {}).get(lk(first), {}).get("all") == V["all"]["pooled"]
    lessons = [
        "<b>Ask what the model is given before asking which model.</b> The same LSTM given more and more information, up to an upper bound that cannot "
        "be had in practice, showed where the room is"
        + (f"; a newer kind of network did not help here: the Transformer took {f['t_lo']:.0f} to {f['t_hi']:.0f} times as long to train for no clear gain "
           f"in accuracy." if has_cmp and f["acc"] == "no clear difference in accuracy" else "."),
        "<b>The yardstick is a simple model given the same inputs, and more than one training.</b> "
        + _cap((f"against {QUOTED} the LSTM with the river gauge is higher in {_and(pers)} of {n} basins; " if pers else "")
               + ("against the best simple model given the same inputs there is no clear difference, and " if all(v == "tie" for v in cls["wq"].values()) else "")
               + f"{_word(n_seeds)} trainings of the same LSTM differ by {n2(spread)} NSE {LEAD_WORD[first]}."),
        f"<b>Score what the user needs.</b> {pct(V['all']['pod'])} of all high-flow days are warned {LEAD_WORD[first]}, yet {pct(1 - on0)} of the onsets "
        f"get no warning" + (f", and a model that only repeats today&rsquo;s flow gets the same {V['all']['pooled']} for all high-flow days."
                             if same_verdict else "."),
        "<b>Know the clock of every input.</b> Which hours a daily value covers decides what a model can know when the forecast is issued"
        + (f"; here rain stamped up to the evening of the issue day closes {share_word(r1['onset_share_of_gap_closed'])} of the gap in onsets warned "
           f"{LEAD_WORD[first]}." if timing_clear else "."),
    ]
    lessons_html = "<p><b>What this teaches about applying AI.</b></p><ul>" + "".join(f"<li>{x}</li>" for x in lessons) + "</ul>"

    sure = _how_sure(U) if U else ""

    # ---- what page 1 has no room for, and the table of steps -----------------------------------------------------------------
    moved = ""
    if plan.get("story") and not plan.get("guide", True) and CUTS < 2:
        moved += f'<p class="guide"><b>Words used on page 1 and in this section.</b> {reading_guide(R)}</p>'
    if plan.get("story") and not plan.get("miles", True):
        moved += "<p><b>Milestones of the plan.</b> " + "; ".join(f"<b>{m}</b> ({what}): {status}" for m, what, status in f["miles"]) + ".</p>"
    if not plan.get("story"):
        steps_done = ""                                                              # the page 1 of before holds the table of steps itself
    elif CUTS < 1:
        steps_done = '<div class="keep">' + table(["Step of the plan", "What was done", "What was found", "Section"], f["rows"], cls="num t81",
                                                  widths=["14%", "19%", "61%", "6%"],
                                                  caption="<b>Table 8.1. What was done and what was found, step by step.</b> Every finding is read "
                                                          "from the results files; the last column gives the section.") + "</div>"
    else:
        name = lambda cell: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cell)).strip()
        listed = [f"{name(r[0])} ({'section ' if i == 0 else ''}{r[3]})" for i, r in enumerate(f["rows"])]
        steps_done = (f"<p><b>What was done, step by step.</b> {'; '.join(listed)}. What each step found is in its section; the conclusions above "
                      f"draw the findings together.</p>")

    # ---- limits: each one or two sentences, tied to a number or a fact of the results --------------------------------------
    limits = []
    seen = (f" A river never seen is a harder test (conclusion {no['unseen']})." if lf else f" A river never seen is a different test ({LOBO_REF}).")
    limits.append(f"<b>Twelve basins, the same rivers in all three periods, {_word(n_seeds)} seeds.</b> The test years measure later years of rivers the "
                  f"models had seen.{seen} Every median rests on n = {n} basins, so one basin can move it, and {_word(n_seeds)} trainings of the same LSTM "
                  f"differ by {n2(spread)} NSE {LEAD_WORD[first]} (the seed spread), as much as most differences between models. The LSTM is a "
                  f"{_word(n_seeds)}-seed average, the simple models are single models; the p-values are not corrected for the many comparisons.")

    ev = F["events"]
    tv = timing_verdict(R)
    stamp = (f"Conclusion {no['timing']} says what that costs" if tf else
             f"{TIMING_REF.capitalize()} measures how much of the gap to the perfect weather forecast is rain that had already fallen")
    limits.append(f"<b>Daily data, and the rain table&rsquo;s day stamp.</b> Rain stamped on a day seems to have fallen mostly between the morning before "
                  f"and that morning (an inference from our own data). {stamp}. Daily steps also hide what happens within a day; "
                  f"{count(ev['one_day_events'])} of the {count(ev['n_events'])} high-flow events last a single day.")

    fair = [k for k in LEADS if cmp_("wqf", k).get("like_for_like") is not False]
    like = (f" Against the simple models, both read the same days of weather only {_leads(fair)}." if 0 < len(fair) < len(LEADS) else "")
    FCt, AOt = forecast_facts(R), aorc_facts(R)
    if FCt:
        limits.append(f"<b>The perfect weather forecast is an upper bound.</b> A real forecast has errors that grow with every day ahead. One was tested (section 4.8: "
                      f"the control member of a reforecast, one of {_word(FCt['members']) if FCt['members'] else 'several'} members): "
                      f"{_leads(LEADS[1:])} it warns of {pct(FCt['fc'][LEADS[1]])} and {pct(FCt['fc'][LEADS[2]])} of the onsets where perfect rain would give {pct(FCt['perfect'][LEADS[1]])} and "
                      f"{pct(FCt['perfect'][LEADS[2]])}, so where a real system would land is known only for that forecast.{like}")
    else:
        limits.append(f"<b>The perfect weather forecast is an upper bound.</b> A real forecast has errors that grow with every day ahead, and none was "
                      f"tested: the results do not say where between {pct(on0)} and {pct(on_up)} of onsets warned a real system would land.{like}")
    if AOt and FCt:
        HSt, S4t = hindsight_facts(R), stage4_facts(R)
        limits.append(f"<b>The new inputs are idealised, and the experiments are small.</b> AORC is {AORC_WORDING}: its hours before 2002, all the training years, are disaggregated daily totals"
                      + (f", and the independent review bounded the hindsight in the {pct(AOt['with_'][first])}: {hindsight_clause(HSt)} ({count(HSt['bound_hits'])} of the {count(AOt['hits'])} hits)" if HSt else "")
                      + (f". Stage IV gives {pct(S4t['stage4'] / S4t['n'])} ({pct(S4t['no_last'] / S4t['n'])} with the last hour out): {REALTIME_WORDING}; whether the archive holds first-posted "
                         f"or later-corrected hours is not established" if S4t else "")
                      + f". Three trainings per set ({nse_forecast_words(FCt)}); the forecast is one ensemble member; the shorter split of E2b has "
                        f"{_and([count(FCt['n_sat'][k]) for k in LEADS])} onsets and a weaker base, so its scores are comparable only inside it (sections 4.6 and 4.8).")

    pk = F["failures"]["peak_size"]
    if "lstm_wq" in pk:
        g, u = pk["lstm_wq"]["all_events"], pk.get("lstm_wqf", {}).get("all_events")
        limits.append(f"<b>Peaks are forecast too low.</b> In the median event the LSTM with the river gauge forecasts {pct(g['median'])} of the observed "
                      f"peak the day before" + (f" ({pct(u['median'])} given the perfect weather forecast)." if u else ".") + (_why_low_peaks(U) if U else ""))

    HF = B["high_flow"]
    test_share = [v["share_test_days_above"] for v in HF.values()]
    val_share = [v["share_val_days_above"] for v in HF.values()]
    has_sat, has_cost = "satellite_data" in R and "satellite_inputs" in R, "model_comparison" in R
    if CUTS < 3:
        limits.append(f"<b>Fewer high-flow days in the test years than in the training years.</b> {pct(sum(test_share) / len(test_share), 1)} of test-year "
                      f"days are above the high-flow level, not {pct(1 - HIGH_Q)}; in the validation years the share runs from {pct(min(val_share), 1)} to "
                      f"{pct(max(val_share), 1)} between basins, so choices made there are fair but noisy. The level is a statistic of each river, not an "
                      f"official flood stage.")
        if has_sat:
            late = re.search(r"about ([\d.]+) months late", R["satellite_data"]["imerg_run"])
            limits.append("<b>The reanalysis inputs are not satellite measurements, and the best satellite rain arrives late.</b> Soil wetness and precipitable "
                          "water are output of a weather model"
                          + (f"; the satellite rain used is published about {late.group(1)} months late" if late else "")
                          + ". Real-time satellite runs and real GNSS water vapour were not tested (section 6.2).")
        if has_cost:
            M = R["model_comparison"]
            secs = [M["models"]["lstm"]["per_seed"][str(s)]["cost"]["seconds"] for s in M["seeds"]]
            limits.append(f"<b>Training times were measured on a busy two-core laptop.</b> Two trainings of the same LSTM took {count(min(secs))} s and "
                          f"{count(max(secs))} s for the same {count(M['settings']['shared']['updates'])} updates, so every cost ratio is rough (section 7).")
    else:                                                                            # the same three limits as pointers to where the report states them
        limits.append("<b>Stated where they arise:</b> " + "; ".join(
            [f"the test years have fewer high-flow days than the training years ({pct(sum(test_share) / len(test_share), 1)} of days, not {pct(1 - HIGH_Q)}), "
             f"and the high-flow level is a statistic of each river, not an official flood stage (sections 1 and 5)"]
            + (["the reanalysis inputs are output of a weather model, not satellite measurements, and the best satellite rain arrives months late "
                "(section 6.2)"] if has_sat else [])
            + (["training times were measured on a busy two-core laptop, so every cost ratio is rough (section 7)"] if has_cost else [])) + ".")

    checks = team.checks_state()                                                 # the members' own checks, as team.json has them
    if checks["n_done"] == 0:
        nwk = new_work(R)
        limits.append(f"<b>The team&rsquo;s own checks of the code are still to be done.</b> All experiments were run on {run_day(R)}"
                      + (f", and those of sections 4.6 to 4.8 on {nwk['days']}," if nwk else "") + f" by Claude Code; the reviews so far are by AI reviewer agents (sections 1 and 9).")
    elif checks["n_done"] < checks["n"]:
        nwk = new_work(R)
        limits.append(f"<b>{_esc(team.checks_sentence(short=True))}.</b> All experiments were run on {run_day(R)}" + (f", and those of sections 4.6 to 4.8 on {nwk['days']}," if nwk else "")
                      + " by Claude Code; the other reviews are by AI reviewer agents (sections 1 and 9).")

    # ---- next steps: what was done since the first version, then what is still open, in order of expected value ----------------------------
    nw = new_work(R)
    done = (f"<p><b>Done since the first version</b> ({nw['days']}): the four experiments, E1 and E1b (section 4.6), E3 (section 4.7) and E2 (section 4.8).</p>") if nw else ""
    Ut = U["stats"] if U else {}
    ranges_open = not all(k in Ut for k in (f"onsets_warned.lstm_wqa.{lk(first)}", f"onsets_warned.lstm_wqaF_sub.{lk(LEADS[1])}"))
    checks_done = checks["n_done"] >= checks["n"]
    steps = ([f"<b>{'The' if checks['n_done'] == 0 else 'The rest of the'} team&rsquo;s own checks</b> (section 9): a few hours that decide whether the "
              f"numbers can be defended."] if not checks_done else [])
    if FCt:
        steps.append(f"<b>Five trainings in place of three</b> where the NSE decides: {nse_forecast_words(FCt)} (the rule of section 4.2; section 4.8).")
    if ranges_open and (AOt or FCt):
        steps.append("<b>The resampling ranges of <code>13_uncertainty.py</code></b> for the sets of sections 4.6 to 4.8, which the draws of test years and basins have not reached yet.")
    steps += ([f"<b>The other ensemble members</b> in place of the control member alone, and the forecast&rsquo;s temperature (section 4.8)."] if FCt else [])
    steps += ([f"<b>A network trained on Stage IV</b>, the real-time product, and a test of its false warnings on days without an onset (section 4.6)."] if AOt else [])
    steps += ([f"<b>The encoder-decoder form</b> that reads a forecast day by day, the structure of an operational flood model {cite('nearing2024')}."] if FCt else
              [f"<b>A real weather forecast as an input</b>, at every lead time. Operational systems feed weather forecasts to an LSTM {cite('nearing2024')}."])
    steps += ["<b>Satellite soil moisture</b> as one more input: the reanalysis soil wetness of section 6.2 is model output, not a measurement.",
              "<b>Hourly rain and flow</b>, to take the day stamp out of the question, and <b>a training target aimed at peaks and onsets</b> instead of the squared error of every day."]

    limits_html = "<ul>" + "".join(f"<li>{x}</li>" for x in limits) + "</ul>"
    steps_html = done + "<p><b>Still open, in order of expected value:</b></p><ol>" + "".join(f"<li>{x}</li>" for x in steps) + "</ol>"

    # ---- 9 (a) who did what, and what each member still has to check ----------------------------------------------------
    roles, _ = _plan()
    exp, mea = FT["expected"]["past_weather_only"], FT["measured"]["past_weather_only"]
    PK = S1["tuning"]["chosen"]
    D = R["data_audit"]
    diag = S1["diagnostic_fixed_length"]["wq"]
    far = max(diag, key=lambda r: r["updates"])
    mean = lambda v: sum(v) / len(v)
    cz = R["model_comparison"]["causality"] if "model_comparison" in R else None
    G = R["basin_gis"]["comparison_with_camels"] if "basin_gis" in R else None
    fine = next(iter(G["slope_mean_by_grid"])) if G else ""
    SC = R.get("sources_checked") or {}                                          # the two published sources on the day stamp: looked up by the AI session, not yet read by the team
    read_src = (f" Read the two sources on the rain table&rsquo;s day stamp ({TIMING_REF}; <code>results/sources_checked.json</code>): no team member "
                f"has yet.") if "no team member has read them yet" in str(SC.get("what", "")) else ""
    todo = {
        "Member 1": (f"Rerun <code>code/01_data_audit.py</code> and compare with <code>results/data_audit.json</code> ({count(D['n_candidates'])} "
                     f"candidates, {count(D['n_complete'])} complete, {len(D['selected'])} chosen); rebuild the report. Question: why these "
                     f"{len(D['selected'])} basins, and no others?"),
        "RS/GIS member 2": ((f"Recompute area, mean elevation and forest share of one basin in a GIS and compare with <code>results/basin_gis.json</code>. "
                             f"Questions: why is the slope {n2(1 + G['slope_mean_by_grid'][fine]['median_difference'] / 100, 1)} times the CAMELS value on "
                             f"a {fine} grid, and why is the gain from satellite rain probably a time-stamp effect?{read_src}") if G else
                            "Check <code>code/07_basin_maps.py</code> once it has run."),
        "GNSS member 1": (f"Run <code>code/test_flood.py</code> ({len(FT['checks'])} checks; NSE {_and(n2(v) for v in exp)} expected on made-up data). "
                          f"Compute NSE, hits, misses and false warnings by hand for one basin and lead time and compare with "
                          f"<code>results/lstm_all_scores.csv</code>. Question: why is the {V['all']['pooled']} for all high-flow days not a flood warning?"),
        "GNSS member 2": (f"From <code>results/hparam_log.csv</code> ({len(T['hparam_log'])} runs) confirm the chosen setting ({PK['hidden']} cells, learning "
                          f"rate {PK['lr']:g}, dropout {PK['dropout']:g}, {PK['warmup']} days of history) by the validation scores alone. "
                          + (f"Confirm the mask test in <code>results/model_comparison.json</code> (largest change "
                             f"{max(cz[m]['largest_change_up_to_day_t'] for m in ('lstm', 'transformer')):g}). " if cz else "")
                          + f"Question: why does longer training hurt the one-river LSTM (validation NSE {n2(mean(far['val_nse']))} after {far['updates']} "
                            f"updates)?"),
    }
    rows = []
    for member, owns in roles:                                                   # member: the role as PLAN.md writes it; printed with the name team.json gives, if any
        key = next((k for k in FILES if member.startswith(k)), None)
        files = FILES.get(key, [])
        have = [x for x in files if (ROOT / x).exists()]
        exists = (", ".join(f"<code>{x.split('/')[-1]}</code>" for x in have) or "&ndash;") + (f"; missing: {', '.join(x for x in files if x not in have)}" if len(have) < len(files) else "")
        rows.append([f"<b>{_esc(team.label(member))}</b>", owns, exists,
                     f"{todo.get(key, '&ndash;')}<br><b>Status of this check:</b> {_esc(team.check_status(member))}."])
    roles_table = table(["Owner in the plan", "Part of the work", "What exists now (written and run by Claude Code)",
                         "What the owner still has to check before the hand-in, and the status of that check"],
                        rows, cls="s8t", widths=["10%", "17%", "25%", "48%"])

    # ---- 9 (b) four things every member must be able to explain ----------------------------------------------------------
    mem = S1["memory"]
    m_of = lambda c: mem["test_nse_by_lead"]["w"][str(c)][0]
    ctx = [c for c in (min(mem["contexts_days"]), 30, 365) if c in mem["contexts_days"]]
    nb = {k: B["best_by_validation"][lk(k)]["without_weather_forecast"]["test"]["nse_median"] for k in LEADS}
    pers_nse = {r["lead"]: r["nse_median"] for r in B["summary_table"] if r["model"] == "base_persistence"}
    miss = {k: 1 - LV["wq"][lk(k)]["onset_pod"] for k in LEADS}
    c_wq = [cmp_("wq", k) for k in LEADS]
    tie = all(c["baseline_against_seed_spread"] != "below every seed" for c in c_wq)
    four = [
        ("Why an LSTM?",
         f"A river remembers: today&rsquo;s flow depends on the rain of the past days to months, and an LSTM learns what to keep in its memory (section 3). "
         f"With {_and(ctx)} days of history the one-river LSTM without a gauge scores NSE {_and(n2(m_of(c)) for c in ctx)} {LEAD_WORD[first]}: it uses old "
         f"weather. " + ("The honest second half: with the river gauge it did not beat the simple models here." if tie else "")),
        ("What are the baselines?",
         f"{QUOTED_CAP} (NSE {n2(pers_nse[first])} {LEAD_WORD[first]}, {n2(pers_nse[last])} {LEAD_WORD[last]}), the calendar-day average, and a ridge regression "
         f"and a random forest on {len(B['features']['w'])} hand-made weather features ({len(B['features']['wq'])} with the river gauge). The number to "
         f"beat with the gauge is {_and(n2(nb[k]) for k in LEADS)} (section 2)."),
        ("How is the test kept fair?",
         f"A forecast belongs to the period of the day it forecasts; the training years fit, the validation years choose, the test years are only scored. "
         f"On made-up data the engine scores {_and(n2(0.0 if round(v, 3) == 0 else v, 3) for v in mea)} where {_and(n2(v) for v in exp)} is the most an "
         f"honest model can reach (section 1). The catch: the same {n} rivers are in all three periods."),
        ("Where does the model fail?",
         f"On onsets: with the river gauge it gives no warning for {pct(miss[first])} of them {LEAD_WORD[first]} and {pct(miss[last])} {LEAD_WORD[last]}. "
         f"It warns of {pct(rt[bins[0]]['onset']['pod'])} of the onsets whose rain is already in its inputs and of {pct(rt[bins[3]]['onset']['pod'])} of "
         f"those whose rain is stamped on the day being forecast (section 5); it forecasts peaks too low"
         + (f"; and for a river never seen {unseen_sentence(R)}." if lf else ".")),
    ]
    four_html = "".join(f"<li><b>{q}</b> {a}</li>" for q, a in four)

    # ---- 9 (c) how to reproduce ------------------------------------------------------------------------------------------
    def step(path, what, key, mark=""):
        mins = _minutes(R, key) if key else None
        timing = "" if mins is None else f", about {mins:.0f} min" if mins >= 1.5 else ", under 2 min"
        gone = "" if (ROOT / path).exists() else " (not written yet)"
        return f"<code>{path.split('/')[-1]}</code>{mark} ({what}{timing}){gone}"
    found = _project_cost(R, T)
    total = sum(s for _, _, s, _ in found) / 3600
    install = (f"In the project folder: <code>pip install -r requirements.txt</code> (Python {sys.version_info.major}.{sys.version_info.minor}; torch and "
               f"numpy are pinned)" if (ROOT / "requirements.txt").exists() else
               f"In the project folder, with Python {sys.version_info.major}.{sys.version_info.minor} and the packages the scripts import")
    assert all(st[0].startswith("code/") for st in STEPS)                        # the sentence below names the folder once
    new_files = ("rain_to_midnight", "aorc_rain_to_midnight", "more_basins", "forecast_input")                 # the four experiments of 7 and 8 October 2026
    added_h = sum(float(_minutes(R, k) or 0) for k in new_files) / 60
    aorc_dl, s4_dl = _dig(R, "aorc_data", "download"), _dig(R, "stage4_realtime", "data", "fetch_runs")
    gefs_dl = _dig(R, "data_access", "summary", "archived_weather_forecasts", "gefsv12_reforecast_aws", "download_done")
    gefs_min = None
    if gefs_dl:                                                                  # '2026-10-07 21:13-22:06 laptop time' -> 53 minutes
        mm = re.search(r"(\d+):(\d+)-(\d+):(\d+)", str(gefs_dl.get("when", "")))
        gefs_min = (int(mm.group(3)) * 60 + int(mm.group(4)) - int(mm.group(1)) * 60 - int(mm.group(2))) if mm else None
    downloads = ""
    if aorc_dl and s4_dl and gefs_dl:
        downloads = (f" The steps marked * are not run by <code>run_all.sh</code>; three need more than the Python packages and are resume-safe: <code>fetch_aorc.py</code> (AORC, "
                     f"{aorc_dl['gb']:.1f} GB in {aorc_dl['hours']:.1f} hours; needs the <code>zstd</code> command-line tool; <code>16_aorc_rain_to_midnight.py</code> calls it), "
                     f"<code>fetch_gefs.py</code> (the GEFSv12 reforecast rain, {gefs_dl['bytes_read_GB']:.0f} GB" + (f" in {gefs_min} minutes" if gefs_min else "")
                     + f"; it runs in the separate environment <code>~/AI_Class/.venv-grib</code>, and <code>gefs_to_basins.py</code> makes the basin means) and <code>check_stage4_realtime.py</code> "
                     f"(Stage IV files from the Iowa Mesonet archive, {s4_dl['gb']:.0f} GB in {s4_dl['hours']:.1f} hours). <code>run_all.sh</code> now takes about {n2(added_h, 1)} hours longer, for "
                     f"the training steps of the four experiments (E1, E1b, E3 and E2).")
    reproduce = (f"<p><b>How to reproduce.</b> {install}, then run the scripts of <code>code/</code> with <code>python</code>, in this order: "
                 + "; ".join(step(*s) for s in STEPS)
                 + f". <code>./run_all.sh</code> runs them in this order.{downloads} The times are those of the runs behind this "
                 f"report on a busy two-core laptop ({n2(total, 1)} hours of computing in all). <code>FLOOD_QUICK=1</code> in front of any script, or of "
                 f"<code>./run_all.sh</code>, gives a fast smoke test with tiny budgets; its outputs go to <code>results/quick</code> and "
                 f"<code>figures/quick</code> and mean nothing. The builders read only <code>results/</code> and <code>figures/</code>.</p>")

    # ---- the optional parts of the plan ------------------------------------------------------------------------------------
    opt = (("satellite_inputs", "the satellite-input experiment", "6.2"), ("model_comparison", "the Transformer experiment", "7"))
    ran_ = [(name, sec) for key, name, sec in opt if key in R]
    if len(ran_) == len(opt):
        optional = (f"<p><b>Optional parts of the plan.</b> The plan made {_word(len(opt))} experiments optional: {_and(name for name, _ in ran_)}. Both were run "
                    f"and are reported in sections {_and(sec for _, sec in ran_)}.</p>")
    else:
        optional = (f"<p><b>Optional parts of the plan.</b> The plan made {_word(len(opt))} experiments optional: {_and(name for _, name, _ in opt)}. "
                    + (f"Only {ran_[0][0]} was run; it is reported in section {ran_[0][1]}.</p>" if ran_ else "Neither was run.</p>"))

    revs = [(n, _file(R, n)) for n in ("review_rain_to_midnight", "review_aorc_rain_to_midnight", "review_more_basins", "review_forecast_input")]
    revs = [(n, r) for n, r in revs if r]
    review_note = ""
    if revs and all(str(r.get("human_review", "")).startswith("not yet done") for _, r in revs):      # each record says that the team's own check of that experiment is still open
        review_note = (f" The reviews of sections 4.6 to 4.8 (<code>results/review_*.json</code>) were written by AI reviewer agents (Claude), each started separately and allowed to edit only its "
                       f"own record; every record says that the team&rsquo;s own check is still open (<code>team.json</code> untouched).")
    return f"""{STYLE}<div class="s8">
<h2>8 Conclusions, limits and next steps</h2>
{opening}
{concl_html}
{lessons_html}
{sure}
{moved}
{steps_done}
<h3>Limits</h3>
{limits_html}
<h3>Next steps</h3>
{steps_html}
<h2>9 Who did what, how to reproduce, sources</h2>
<p><b>Who did what.</b> {who(R).replace(' (section 9)', '')}{review_note} The table follows the roles of the plan (PLAN.md); the status of each check is read from
<code>team.json</code>, which only the members fill in.</p>
{roles_table}
<p><b>Four things every member must be able to explain</b> (the plan asks for exactly these).</p>
<ol>{four_html}</ol>
{reproduce}
{optional}
<p><b>Sources.</b> The data (CAMELS-US with its weather tables, NASA POWER, the elevation tiles, the land-cover layer, the storm tracks, ERA5 and AORC hourly rain and the GEFSv12
reforecast rain) are cached under <code>data/</code>. Every source cited in this report:</p>
{_references()}
</div>"""
