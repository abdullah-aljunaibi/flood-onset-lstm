"""How sure are the headline numbers? The range in which each stays when the test years and the basins are drawn again.

The main numbers of the report are medians over 12 basins and counts of warnings over 9 test years: few of each. This script asks how
far those numbers move when other years and other basins of the same kind are in the test, and it adds the scores that say why the
peaks are forecast too low. Nothing is trained: only the forecasts stored by the other scripts are read. The script
 1. computes, for every basin and test water year, the sums the NSE needs and the counts of high-flow days, onsets, hits and false
    warnings at the high-flow level itself (an onset exactly as in flood.flood_scores), and asserts - before anything is resampled -
    that these sums give the scores of flood.evaluate and the numbers stored in the results files, and that the best simple model
    named there is the one with the highest median NSE in the validation years;
 2. draws the test water years and the 12 basins again, each with replacement, B = 4,000 times (a whole water year stays together, so
    the days of one storm - and the same storm in several basins - stay together), recomputes every statistic exactly on each resample
    and takes the 2.5th to 97.5th percentile as the interval; the years-only and the basins-only designs are stored next to it, and
    how far an end of an interval moves with the random numbers is judged from 16 equal parts of the resamples;
 3. counts, for every verdict cell (GO / CAUTION / NOT YET by the criteria stored in results/flood_analysis.json), the share of the
    resamples with each verdict - at the high-flow level itself and, for the sets that have one in that file, also at the warning
    level it chose on the validation years;
 4. gives, without resampling, per basin over the test days and then as the median of the basins: the spread of the forecasts against
    the spread of the observed flow, their correlation, their total, the skill against 'tomorrow = today' and the KGE (these medians
    are also among the statistics of 2, so they have an interval too).
Extended on 2026-10-08, so that a range stands beside every new headline number (SETS_NEW; everything of 4 October is recomputed unchanged: every old statistic,
verdict cell and spread-and-skill number equals the file of that day). Added are the prediction sets of the four experiments of 7-8 October, all on the same 12 basins
and the test years of the main split, with the high-flow levels as the engine gives them:
  script 15 (hourly ERA5 rain up to local midnight): lstm_wqh, lstm_wqh_older, lstm_wqh_to09, lstm_wqh_utc, lstm_wqnh, lstm_wh, lstm_wh_older, base_linreg_wqh, base_rf_wqh;
  script 16 (hourly AORC rain): lstm_wqa, lstm_wqa_to09, lstm_wqa_utc, lstm_wqa_rep, base_linreg_wqa, base_rf_wqa;
  script 17 (63 training rivers; 50 training rivers, the 12 never seen), scored on the 12: lstm63_wq, lstm63_w, lstm63eq_wq, unseen50_w_static, unseen50_w_nostatic,
  and, because its never-seen comparison needs them, lobo_w_nostatic and lobo_ref_w_nostatic of 05_lobo.py (the script held no no-properties reference before);
  script 18 (the GEFSv12 forecast rain, substituted into a network trained on the AORC rain of those days, an analysis made after the fact; main split only): lstm_wqaF_sub, lstm_wqaF_clim, lstm_wqF_sub, lstm_wqaP,
  base_linreg_wqaF, base_rf_wqaF.
For each set: the median NSE per lead and, for every set with warnings (WARN_NEW: the added sets and lobo_w_static, which the comparisons with the never-seen sets need), the six
warning rates per lead at the high-flow level itself (not at a level chosen on the validation years: flood_analysis.json holds none for these sets). For each difference of PAIRS_NEW (against the base, the control, the matched-window controls, the simple
models): the NSE as the difference of the two medians and as the median of the basin differences, and onsets_diff, the same difference in the share of onsets warned.
For each closure of CLOSE: the share of the gap in onsets warned, to an upper bound that cannot be had in practice, that a set closes. And a verdict cell (GO / CAUTION /
NOT YET) for every set with warnings, as for the older sets. Before anything is resampled the script asserts that its sums per basin and year give the scores of
flood.evaluate for the added sets and that these are the numbers stored by the four scripts (the whole evaluation for 15, 16 and 18, the pooled summary and the per-river
scores for 17), that the differences and closures on the test years as they are equal the comparison numbers those files store (gain_rule, closure, yardstick, unseen), and
that the verdict of every added set is the one its own script stored.
Left out on purpose: the sets of the retraining with the forecast (E2b; names ending in _s in forecast_input.json) lie on another split (test years 2010-2014, four water
years), so they share no test days with the sets above and cannot be drawn again with them (forecast_input.json holds their block-bootstrap intervals); and the single
trainings, the derived variants of script 18, spread, skill and KGE, and warning levels chosen on the validation years for the added sets.
Reads the prediction sets named in LABEL (results/preds/<set>.npz) and results/lstm_all.json, rain_timing.json, flood_analysis.json,
lstm_lobo.json and model_comparison.json (high-flow levels, best simple model, verdict criteria, and the stored numbers to check against) and, for the added sets,
rain_to_midnight.json, aorc_rain_to_midnight.json, more_basins.json and forecast_input.json.
Writes results/uncertainty.json.
FLOOD_QUICK=1: reads and writes results/quick only, with 200 resamples; if a smoke-test input is missing it says so and stops. In a smoke test the stored differences
and closures whose controls are real results of the older scripts (lstm_wq, lstm_wqn, lobo_*, ...) are not compared, only those between two added sets.
The checks are assert statements, so the script refuses to run under python -O (which would skip them).
"""
import json
import re
import sys
import time

import numpy as np
import pandas as pd

import flood as F
from common import ROOT

if not __debug__:                                            # python -O drops every assert below, and the results file says that they passed
    sys.exit("13_uncertainty.py: run it without -O (its checks are assert statements); nothing written.")
t_start = time.time()
B = 200 if F.QUICK else 4000                                 # resamples
SEED = 20261004                                              # the only source of random numbers, so a rerun gives the same file
N_DIRECT = 2                                                 # resamples that are also scored the slow way, to check the fast sums
MC_PARTS = 16                                                # equal parts of the resamples from which the Monte-Carlo noise of an interval end is judged
LABEL = {"lstm_w": "LSTM, past weather", "lstm_wq": "LSTM, past weather + river gauge",
         "lstm_wf": "LSTM, past weather + perfect weather forecast", "lstm_wqf": "LSTM, past weather + river gauge + perfect weather forecast",
         "lstm_wn": "LSTM, past weather + rain up to the evening of the issue day", "lstm_wn_older": "control: LSTM, past weather + the same rain one day older",
         "lstm_wqn": "LSTM, past weather + river gauge + rain up to the evening of the issue day",
         "lstm_wqn_older": "control: LSTM, past weather + river gauge + the same rain one day older",
         "base_persistence": "'tomorrow = today'", "base_linreg_w": "ridge regression, past weather", "base_rf_w": "random forest, past weather",
         "base_linreg_wq": "ridge regression, past weather + river gauge", "base_rf_wq": "random forest, past weather + river gauge",
         "lstm_w_s0": "LSTM, past weather, one training, the basin in training (reference of the unseen-river test)",
         "lobo_w_static": "LSTM, past weather, the basin held out of training (a river never seen)",
         "cmp_lstm": "LSTM of the model comparison (365-day sequences), past weather + river gauge", "cmp_transformer": "Transformer of the model comparison, same inputs"}
SETS_OLD = list(LABEL)                                       # the prediction sets of 4 October 2026: nothing computed for them changes
# the prediction sets added on 2026-10-08, by the script that wrote them (all on the 12 basins and the test years of the main split)
E1_SETS = {"lstm_wqh": "wqh", "lstm_wqh_older": "wqh_older", "lstm_wqh_to09": "wqh_to09", "lstm_wqh_utc": "wqh_utc", "lstm_wqnh": "wqnh",
           "lstm_wh": "wh", "lstm_wh_older": "wh_older"}      # 15_rain_to_midnight.py: set -> its key in rain_to_midnight.json (models, verdicts.sets)
E1_SIMPLE = ("base_linreg_wqh", "base_rf_wqh")               # ... and its simple models (yardstick.sets)
E1B_SETS = {"lstm_wqa": "wqa", "lstm_wqa_to09": "wqa_to09", "lstm_wqa_utc": "wqa_utc", "lstm_wqa_rep": "wqa_rep"}      # 16_aorc_rain_to_midnight.py
E1B_SIMPLE = ("base_linreg_wqa", "base_rf_wqa")
E2_SETS = {"lstm_wqaF_sub": "wqaF_sub", "lstm_wqaF_clim": "wqaF_clim", "lstm_wqF_sub": "wqF_sub", "lstm_wqaP": "wqaP"}   # 18_forecast_input.py, main split only
E2_SIMPLE = ("base_linreg_wqaF", "base_rf_wqaF")
E3_SETS = ("lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic")                     # 17_more_basins.py, scored on the 12 rivers only
LOBO_NEW = ("lobo_w_nostatic", "lobo_ref_w_nostatic")        # 05_lobo.py: the no-properties partners that the unseen-river comparison of script 17 needs
LABEL.update({
    "lstm_wqh": "LSTM, past weather + river gauge + ERA5 rain from 00:00 to 24:00 of the issue day (up to local midnight)",
    "lstm_wqh_older": "control: LSTM, past weather + river gauge + the same ERA5 rain one day older",
    "lstm_wqh_to09": "control: LSTM, past weather + river gauge + ERA5 rain over the rain table's own window (09:00 to 09:00), no new hours",
    "lstm_wqh_utc": "control: LSTM, past weather + river gauge + ERA5 rain over the UTC day (ends 19:00 local standard time)",
    "lstm_wqnh": "LSTM, past weather + river gauge + the rain up to the evening (NLDAS) + ERA5 rain up to local midnight",
    "lstm_wh": "LSTM, past weather + ERA5 rain up to local midnight of the issue day (no river gauge)",
    "lstm_wh_older": "control: LSTM, past weather + the same ERA5 rain one day older (no river gauge)",
    "base_linreg_wqh": "ridge regression, past weather + river gauge + ERA5 rain up to local midnight",
    "base_rf_wqh": "random forest, past weather + river gauge + ERA5 rain up to local midnight",
    "lstm_wqa": "LSTM, past weather + river gauge + AORC rain from 00:00 to 24:00 of the issue day (up to local midnight)",
    "lstm_wqa_to09": "control: LSTM, past weather + river gauge + AORC rain over the rain table's own window (09:00 to 09:00), no new hours",
    "lstm_wqa_utc": "control: LSTM, past weather + river gauge + AORC rain over the UTC day (ends 19:00 local standard time)",
    "lstm_wqa_rep": "LSTM, past weather with the rain column replaced by the AORC rain up to local midnight (no extra column) + river gauge",
    "base_linreg_wqa": "ridge regression, past weather + river gauge + AORC rain up to local midnight",
    "base_rf_wqa": "random forest, past weather + river gauge + AORC rain up to local midnight",
    "lstm63_wq": "LSTM trained on 63 rivers, past weather + river gauge + basin properties, scored on the 12",
    "lstm63_w": "LSTM trained on 63 rivers, past weather + basin properties (no river gauge), scored on the 12",
    "lstm63eq_wq": "LSTM trained on 63 rivers with five times the update limit and patience, past weather + river gauge + basin properties, scored on the 12",
    "unseen50_w_static": "LSTM trained on the 50 other rivers, past weather + basin properties: the 12 are rivers never seen (warnings at the observed high-flow level, which such a river does not have)",
    "unseen50_w_nostatic": "LSTM trained on the 50 other rivers, past weather only: the 12 are rivers never seen (warnings at the observed high-flow level, which such a river does not have)",
    "lobo_w_nostatic": "LSTM, past weather only, the basin held out of training (a river never seen, 11 training rivers, no basin properties)",
    "lobo_ref_w_nostatic": "LSTM, past weather only, the basin in training (12 rivers; the reference of the unseen-river test without basin properties)",
    "lstm_wqaF_sub": "LSTM, past weather + river gauge + AORC rain to midnight + the GEFS forecast of the rain of the next three days, substituted into a network trained on the AORC rain of those days (an analysis made after the fact) (main split)",
    "lstm_wqaF_clim": "control: the same with the forecast shuffled within basin and calendar month (no information about the day)",
    "lstm_wqF_sub": "LSTM, past weather + river gauge (the rain table, no rain to midnight) + the GEFS forecast of the rain of the next three days, substituted",
    "lstm_wqaP": "LSTM, past weather + river gauge + AORC rain to midnight + the observed AORC rain of the next three days: a perfect rain forecast, an upper bound that cannot be had in practice",
    "base_linreg_wqaF": "ridge regression, past weather + river gauge + AORC rain to midnight + the GEFS forecast (substituted)",
    "base_rf_wqaF": "random forest, past weather + river gauge + AORC rain to midnight + the GEFS forecast (substituted)"})
SETS = list(LABEL)
SETS_NEW = SETS[len(SETS_OLD):]
STORED = ("lstm_all.json", "rain_timing.json", "flood_analysis.json", "lstm_lobo.json", "model_comparison.json")
STORED_NEW = ("rain_to_midnight.json", "aorc_rain_to_midnight.json", "more_basins.json", "forecast_input.json")      # the results files of scripts 15, 16, 17 and 18
PAIRS = {"lstm_w_minus_best_simple": ("lstm_w", "best_simple_w"), "lstm_wq_minus_best_simple": ("lstm_wq", "best_simple_wq"),     # name: (model, minus model)
         "lstm_wq_minus_rf_wq": ("lstm_wq", "base_rf_wq"), "lstm_wq_minus_linreg_wq": ("lstm_wq", "base_linreg_wq"),
         "wq_minus_w": ("lstm_wq", "lstm_w"), "wf_minus_w": ("lstm_wf", "lstm_w"), "wqf_minus_wq": ("lstm_wqf", "lstm_wq"),
         "wqn_minus_wq": ("lstm_wqn", "lstm_wq"), "wqn_minus_wqn_older": ("lstm_wqn", "lstm_wqn_older"),
         "held_out_minus_in_training": ("lobo_w_static", "lstm_w_s0"), "transformer_minus_lstm": ("cmp_transformer", "cmp_lstm")}
PAIRS_OLD = dict(PAIRS)                                      # the differences of 4 October
PAIRS_NEW = {                                                # added on 2026-10-08; name: (model, minus model)
    # E1, script 15: rain up to local midnight (ERA5) against the rain table alone, its control (the same rain one day older), the matched-window controls and the NLDAS rain
    "wqh_minus_wq": ("lstm_wqh", "lstm_wq"), "wqh_minus_wqh_older": ("lstm_wqh", "lstm_wqh_older"), "wqh_older_minus_wq": ("lstm_wqh_older", "lstm_wq"),
    "wqh_minus_wqn": ("lstm_wqh", "lstm_wqn"), "wqh_minus_wqh_to09": ("lstm_wqh", "lstm_wqh_to09"), "wqh_minus_wqh_utc": ("lstm_wqh", "lstm_wqh_utc"),
    "wqh_to09_minus_wq": ("lstm_wqh_to09", "lstm_wq"), "wqh_utc_minus_wq": ("lstm_wqh_utc", "lstm_wq"), "wqh_utc_minus_wqh_to09": ("lstm_wqh_utc", "lstm_wqh_to09"),
    "wqnh_minus_wqh": ("lstm_wqnh", "lstm_wqh"), "wqnh_minus_wqn": ("lstm_wqnh", "lstm_wqn"), "wqnh_minus_wq": ("lstm_wqnh", "lstm_wq"),
    "wh_minus_w": ("lstm_wh", "lstm_w"), "wh_minus_wh_older": ("lstm_wh", "lstm_wh_older"), "wh_older_minus_w": ("lstm_wh_older", "lstm_w"), "wh_minus_wn": ("lstm_wh", "lstm_wn"),
    "lstm_wqh_minus_linreg_wqh": ("lstm_wqh", "base_linreg_wqh"), "lstm_wqh_minus_rf_wqh": ("lstm_wqh", "base_rf_wqh"),
    "linreg_wqh_minus_linreg_wq": ("base_linreg_wqh", "base_linreg_wq"), "rf_wqh_minus_rf_wq": ("base_rf_wqh", "base_rf_wq"),
    # E1b, script 16: AORC rain up to local midnight
    "wqa_minus_wq": ("lstm_wqa", "lstm_wq"), "wqa_minus_wqa_to09": ("lstm_wqa", "lstm_wqa_to09"), "wqa_minus_wqa_utc": ("lstm_wqa", "lstm_wqa_utc"),
    "wqa_to09_minus_wq": ("lstm_wqa_to09", "lstm_wq"), "wqa_utc_minus_wq": ("lstm_wqa_utc", "lstm_wq"), "wqa_utc_minus_wqa_to09": ("lstm_wqa_utc", "lstm_wqa_to09"),
    "wqa_minus_wqh": ("lstm_wqa", "lstm_wqh"), "wqa_minus_wqn": ("lstm_wqa", "lstm_wqn"), "wqa_minus_wqnh": ("lstm_wqa", "lstm_wqnh"),
    "wqa_rep_minus_wq": ("lstm_wqa_rep", "lstm_wq"), "wqa_rep_minus_wqa": ("lstm_wqa_rep", "lstm_wqa"), "wqa_rep_minus_wqn": ("lstm_wqa_rep", "lstm_wqn"),
    "wqa_rep_minus_wqh": ("lstm_wqa_rep", "lstm_wqh"),
    "lstm_wqa_minus_linreg_wqa": ("lstm_wqa", "base_linreg_wqa"), "lstm_wqa_minus_rf_wqa": ("lstm_wqa", "base_rf_wqa"),
    "linreg_wqa_minus_linreg_wq": ("base_linreg_wqa", "base_linreg_wq"), "rf_wqa_minus_rf_wq": ("base_rf_wqa", "base_rf_wq"),
    # E3, script 17: 63 training rivers instead of 12 (the same 12 scored), and rivers never seen
    "lstm63_wq_minus_wq": ("lstm63_wq", "lstm_wq"), "lstm63_w_minus_w": ("lstm63_w", "lstm_w"), "lstm63eq_wq_minus_wq": ("lstm63eq_wq", "lstm_wq"),
    "lstm63eq_wq_minus_lstm63_wq": ("lstm63eq_wq", "lstm63_wq"),
    "unseen50_w_static_minus_lobo_w_static": ("unseen50_w_static", "lobo_w_static"), "unseen50_w_nostatic_minus_lobo_w_nostatic": ("unseen50_w_nostatic", "lobo_w_nostatic"),
    "unseen50_w_static_minus_unseen50_w_nostatic": ("unseen50_w_static", "unseen50_w_nostatic"), "unseen50_w_static_minus_lstm63_w": ("unseen50_w_static", "lstm63_w"),
    "unseen50_w_static_minus_lstm_w": ("unseen50_w_static", "lstm_w"), "unseen50_w_nostatic_minus_lobo_ref_w_nostatic": ("unseen50_w_nostatic", "lobo_ref_w_nostatic"),
    "lobo_w_nostatic_minus_lobo_ref_w_nostatic": ("lobo_w_nostatic", "lobo_ref_w_nostatic"),
    # E2a, script 18: the GEFS rain forecast, main split
    "wqaF_sub_minus_wqa": ("lstm_wqaF_sub", "lstm_wqa"), "wqaF_sub_minus_wq": ("lstm_wqaF_sub", "lstm_wq"), "wqaF_sub_minus_wqaF_clim": ("lstm_wqaF_sub", "lstm_wqaF_clim"),
    "wqaF_sub_minus_wqaP": ("lstm_wqaF_sub", "lstm_wqaP"), "wqaF_sub_minus_wqF_sub": ("lstm_wqaF_sub", "lstm_wqF_sub"), "wqaF_clim_minus_wqa": ("lstm_wqaF_clim", "lstm_wqa"),
    "wqaP_minus_wqa": ("lstm_wqaP", "lstm_wqa"), "wqF_sub_minus_wq": ("lstm_wqF_sub", "lstm_wq"),
    "lstm_wqaF_sub_minus_linreg_wqaF": ("lstm_wqaF_sub", "base_linreg_wqaF"), "lstm_wqaF_sub_minus_rf_wqaF": ("lstm_wqaF_sub", "base_rf_wqaF"),
    "linreg_wqaF_minus_linreg_wqa": ("base_linreg_wqaF", "base_linreg_wqa"), "rf_wqaF_minus_rf_wqa": ("base_rf_wqaF", "base_rf_wqa")}
assert not set(PAIRS_NEW) & set(PAIRS_OLD) and all(a in SETS and b in SETS for a, b in PAIRS_NEW.values()), "a difference names a set that is not read here"
PAIRS.update(PAIRS_NEW)
SKILL = ("lstm_wq", "base_rf_wq", "base_linreg_wq", "lstm_w", "lstm_wqf")                       # skill against 'tomorrow = today', with an interval
WARN_OLD = ("lstm_w", "lstm_wq", "lstm_wqn", "lstm_wqn_older", "lstm_wqf", "lstm_wn", "lstm_wn_older", "lstm_wf",
            "base_rf_wq", "base_linreg_wq", "base_rf_w", "base_linreg_w", "base_persistence")       # sets whose warnings are counted (4 October)
WARN_NEW = tuple(SETS_NEW) + ("lobo_w_static",)              # every set added on 2026-10-08, and lobo_w_static, which the comparisons with the never-seen sets of script 17 need
WARN = WARN_OLD + WARN_NEW                                   # (for a river never seen the warnings are counted at the observed high-flow level, which such a river does not have)
FAMILY = {"wq": ("lstm_wq", "lstm_wqn", "lstm_wqn_older", "lstm_wqf"), "w": ("lstm_w", "lstm_wn", "lstm_wn_older", "lstm_wf")}      # base, + evening rain, its control, upper bound
CLOSE = {                                                    # added on 2026-10-08; name: (set, base, upper bound, control or None): the share of the gap in onsets warned that the set closes
    "wqh__base_wq__ceiling_wqf": ("lstm_wqh", "lstm_wq", "lstm_wqf", "lstm_wqh_older"), "wh__base_w__ceiling_wf": ("lstm_wh", "lstm_w", "lstm_wf", "lstm_wh_older"),
    "wqh_to09__base_wq__ceiling_wqf": ("lstm_wqh_to09", "lstm_wq", "lstm_wqf", None), "wqh_utc__base_wq__ceiling_wqf": ("lstm_wqh_utc", "lstm_wq", "lstm_wqf", None),
    "wqnh__base_wq__ceiling_wqf": ("lstm_wqnh", "lstm_wq", "lstm_wqf", None),
    "wqa__base_wq__ceiling_wqf": ("lstm_wqa", "lstm_wq", "lstm_wqf", "lstm_wqa_to09"), "wqa_to09__base_wq__ceiling_wqf": ("lstm_wqa_to09", "lstm_wq", "lstm_wqf", None),
    "wqa_utc__base_wq__ceiling_wqf": ("lstm_wqa_utc", "lstm_wq", "lstm_wqf", None), "wqa_rep__base_wq__ceiling_wqf": ("lstm_wqa_rep", "lstm_wq", "lstm_wqf", None),
    "lstm63_wq__base_wq__ceiling_wqf": ("lstm63_wq", "lstm_wq", "lstm_wqf", None), "lstm63eq_wq__base_wq__ceiling_wqf": ("lstm63eq_wq", "lstm_wq", "lstm_wqf", None),
    "wqaF_sub__base_wqa__ceiling_wqaP": ("lstm_wqaF_sub", "lstm_wqa", "lstm_wqaP", None), "wqaF_sub__base_wqa__ceiling_wqf": ("lstm_wqaF_sub", "lstm_wqa", "lstm_wqf", None),
    "wqaF_sub__base_wq__ceiling_wqf": ("lstm_wqaF_sub", "lstm_wq", "lstm_wqf", None), "wqaF_clim__base_wqa__ceiling_wqaP": ("lstm_wqaF_clim", "lstm_wqa", "lstm_wqaP", None),
    "wqF_sub__base_wq__ceiling_wqf": ("lstm_wqF_sub", "lstm_wq", "lstm_wqf", None), "wqF_sub__base_wq__ceiling_wqaP": ("lstm_wqF_sub", "lstm_wq", "lstm_wqaP", None),
    "wqa__base_wq__ceiling_wqaP": ("lstm_wqa", "lstm_wq", "lstm_wqaP", None), "wqaP__base_wqa__ceiling_wqf": ("lstm_wqaP", "lstm_wqa", "lstm_wqf", None)}
assert all(x in WARN for v in CLOSE.values() for x in v if x), "a closure names a set whose warnings are not counted"
DIAG = ("lstm_wq", "base_rf_wq", "base_linreg_wq", "base_persistence", "lstm_wqf", "lstm_w")    # the sets of part 4
PARTS = ("spread_ratio", "correlation", "total_ratio", "skill_vs_persistence", "kge")           # what part 4 gives for each of them
RATE = {"all": ("all_warned", "all_false_warnings", "all_csi"), "onset": ("onsets_warned", "onset_false_warnings", "onset_csi")}    # pod, far, csi of flood.rates
AT = "_at_chosen_level"                                      # suffix of the same six rates at the warning level chosen on the validation years
AT_KEY = "at_level_chosen_on_validation_years"               # where a verdict cell holds its verdict at that level
lk = lambda k: f"l{k}"

# ---- inputs --------------------------------------------------------------------------------------------------------------
need = [F.PRED / f"{n}.npz" for n in SETS] + [F.RES_OUT / f for f in STORED + STORED_NEW]
missing = [str(f.relative_to(ROOT)) for f in need if not f.exists()]
if missing:
    what = f"13_uncertainty.py: {len(missing)} input file(s) not found ({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''})"
    if F.QUICK:
        print(what + " - smoke test skipped, nothing written.")
        sys.exit(0)
    sys.exit(what + ". Run the scripts that write them first (see run_all.sh).")
A, RT, FA, LO, MC = (json.loads((F.RES_OUT / f).read_text()) for f in STORED)
RM, AO, MB, FI = (json.loads((F.RES_OUT / f).read_text()) for f in STORED_NEW)       # scripts 15, 16, 17 and 18
FILES = dict(RM=RM, AO=AO, MB=MB, FI=FI)
FILE_NAME = dict(zip(FILES, STORED_NEW))
assert all(x["quick_mode"] == F.QUICK for x in (A, RT, FA, LO, MC, RM, AO, MB, FI)), "a results file is a smoke-test output (or, in a smoke test, a real one)"
assert all(list(split) == list(F.SPLIT["test"]) for split in (RM["split"]["test"], AO["split"]["test"], MB["split"]["test"], FI["split_main"]["test"])), \
    "a results file of the added sets was scored on other test years than flood.SPLIT"


def load(name):
    """One stored prediction set as arrays: basins, days (the day that is forecast), observed flow [basin, day], forecasts [basin, day, lead]."""
    with np.load(F.PRED / f"{name}.npz", allow_pickle=False) as z:
        return [str(b) for b in z["basins"]], pd.DatetimeIndex(z["dates"]), z["obs"].astype(float), z["pred"].astype(float)


P = {n: load(n) for n in SETS}
ids, days, obs, _ = P[SETS[0]]
for n in SETS:
    assert P[n][0] == ids and P[n][1].equals(days) and np.array_equal(P[n][2], obs), f"{n}: not the same basins, days and observed flow as {SETS[0]}"
nb = len(ids)
thr_of = F.thresholds({g: pd.DataFrame({"q_mm": obs[i]}, index=days) for i, g in enumerate(ids)})      # the engine's rule on the stored observed flow
assert all(abs(thr_of[g] - x[g]) < 1e-9 for x in (A["thresholds"], FA["thresholds_mm_day"]) for g in ids), "the high-flow levels differ from the stored ones"
thr = np.array([thr_of[g] for g in ids])
BEST = {g: A["comparison"][g]["best_baseline_by_validation"] for g in FAMILY}                          # {"wq": {"l1": "base_linreg_wq", ...}, ...}
assert all(b in WARN for g in BEST for b in BEST[g].values()), "a best simple model is not among the sets read here"

# ---- the test days, cut into water years (from flood.SPLIT: no year is typed here) ------------------------------------------
t0, t1 = (pd.Timestamp(x) for x in F.SPLIT["test"])
T = np.flatnonzero((days >= t0) & (days <= t1))              # the test days: a forecast belongs to the day it forecasts
assert len(T) == len(pd.date_range(t0, t1)) and T[0] >= max(F.LEADS), "the stored forecasts do not cover every test day"
starts = pd.date_range(t0, t1, freq=pd.DateOffset(years=1))  # first day of each 12-month block of the test period
ends = starts + pd.DateOffset(years=1) - pd.Timedelta(days=1)
assert ends[-1] == t1, "the test period of flood.SPLIT is not a whole number of years"
YEARS = [int(e.year) for e in ends]                          # a water year is named after the calendar year in which it ends
ny = len(YEARS)
yi = np.searchsorted(starts.values, days[T].values, side="right") - 1                                  # the block of each test day
Y = (yi[:, None] == np.arange(ny)).astype(float)             # [test day, year]: 1 where the day lies in the year
per_year = lambda x: np.einsum("bd...,dy->by...", x.astype(float), Y)                                  # [basin, day, ...] -> sums [basin, year, ...]

# ---- per basin and water year: everything a resample needs -----------------------------------------------------------------
o = obs[:, T]                                                # observed flow on the test days [basin, day]
PT = {n: P[n][3][:, T] for n in SETS}                        # forecasts on the test days [basin, day, lead]
for n in SETS:                                               # every set has a forecast for every basin, test day and lead: all are scored on the same basin-days
    assert np.isfinite(PT[n]).all(), f"{n}: days without a forecast in the test years"
hi = o > thr[:, None]                                        # high-flow days
low = np.stack([obs[:, T - k] <= thr[:, None] for k in F.LEADS], 2)        # the river was still low when the forecast was issued, k days before [basin, day, lead]
n_y, so, soo = per_year(np.ones_like(o)), per_year(o), per_year(o ** 2)    # days, sum and sum of squares of the observed flow [basin, year]
n_hi, n_on = per_year(hi), per_year(hi[:, :, None] & low)    # high-flow days [basin, year] and onsets [basin, year, lead]
sse = {n: per_year((PT[n] - o[:, :, None]) ** 2) for n in SETS}            # squared error [basin, year, lead]
mom = {n: (per_year(PT[n]), per_year(PT[n] ** 2), per_year(PT[n] * o[:, :, None])) for n in DIAG}      # of the forecasts: sum, sum of squares, sum of products with the observed flow


def tables(name, level=1.0):
    """Hits and false warnings [basin, year, lead], over all high-flow days and over onsets, when a warning is given above level x the high-flow level."""
    w, h = PT[name] > np.asarray(level) * thr[:, None, None], hi[:, :, None]
    return dict(hit_all=per_year(h & w), fa_all=per_year(~h & w), hit_on=per_year(h & w & low), fa_on=per_year(~h & w & low))


W = {n: tables(n) for n in WARN}                             # at the high-flow level itself
CHOSEN = {n: [FA["skill"][n]["tuned"][lk(k)]["trigger"] for k in F.LEADS] for n in WARN if n in FA["skill"]}      # warning levels chosen on the validation years by 06_flood_analysis.py
WC = {n: tables(n, CHOSEN[n]) for n in CHOSEN}
TABLE = [n for n in FA["verdict"]["services"] if n in CHOSEN]                                           # the sets of the verdict table in flood_analysis.json

# ---- check 1: these sums against the engine, and the engine against the stored files ----------------------------------------
n_compared = 0


def close(a, b):
    """Two numbers agree (an empty rate is NaN in both - or null in a results file written without NaN; counts must be identical)."""
    global n_compared
    n_compared += not isinstance(a, str)
    a, b = (float("nan") if x is None else x for x in (a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (a != a and b != b) or abs(a - b) <= 1e-9
    return a == b


def same(a, b):
    """Two outputs of flood.evaluate hold the same numbers."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return close(a, b)


by_file = {}                                                 # numbers compared for the sets and comparisons added on 2026-10-08, by results file


def tally(where, c0, n=None):
    """Count the numbers compared since c0 for an added set (n in SETS_NEW) or, with n None, for an added comparison."""
    if n is None or n in SETS_NEW:
        by_file[where] = by_file.get(where, 0) + n_compared - c0


def covers(a, b):
    """Every number stored in b (nested dicts) is in a and the same (a may hold more)."""
    if isinstance(b, dict):
        return isinstance(a, dict) and all(k in a and covers(a[k], b[k]) for k in b)
    return close(a, b)


def check_summary(n, E):
    """results/more_basins.json stores the pooled summary and the per-river scores of a set (not the whole evaluation): they are those of flood.evaluate on its forecasts."""
    st = MB["summary_test"]["sets"][n]
    assert all(covers(E["summary"][lk(k)], st[lk(k)]) for k in F.LEADS), f"results/more_basins.json does not hold the summary of results/preds/{n}.npz: rerun 17_more_basins.py"
    for key, get in (("nse", lambda r: r["nse"]), ("onset_warned", lambda r: r["onset"]["pod"]), ("all_days_warned", lambda r: r["all"]["pod"])):
        pr = MB["per_river"][key][n]
        assert all(close(get(E["basins"][g][lk(k)]), pr[lk(k)][g]) for k in F.LEADS for g in ids), f"results/more_basins.json does not hold the per-river {key} of results/preds/{n}.npz"


stored = {**{f"lstm_{s}": [("lstm_all.json", A["sets"][s]["test"])] for s in ("w", "wq", "wf", "wqf")},
          **{n: [("lstm_all.json", A["baselines"][n]["test"])] for n in SETS_OLD if n.startswith("base_")},
          **{f"lstm_{s}": [("rain_timing.json", RT["models"][s]["test"])] for s in ("wn", "wn_older", "wqn", "wqn_older")},
          "lstm_w_s0": [("lstm_lobo.json", LO["references"]["lstm_w_s0"]["test"])], "lobo_w_static": [("lstm_lobo.json", LO["variants"]["lobo_w_static"]["test"])],
          "cmp_lstm": [("model_comparison.json", MC["models"]["lstm"]["seed_average"]["test"])],
          "cmp_transformer": [("model_comparison.json", MC["models"]["transformer"]["seed_average"]["test"])]}
HOME = {}                                                    # an added set -> (file, its key in it): where scripts 15, 16 and 18 store its scores and its verdicts
for fk, sets, simple in (("RM", E1_SETS, E1_SIMPLE), ("AO", E1B_SETS, E1B_SIMPLE), ("FI", E2_SETS, E2_SIMPLE)):
    for n, key in sets.items():
        m = FILES[fk]["models"][key]
        assert m["saved_as"] == n and m.get("arm", "main") == "main", f"results/{FILE_NAME[fk]}: models.{key} is not the main-split set {n}"
        stored[n], HOME[n] = [(FILE_NAME[fk], m["test"])], (fk, key)
    for n in simple:
        y = FILES[fk]["yardstick"]["sets"][n]
        assert y["saved_as"] == n, f"results/{FILE_NAME[fk]}: yardstick.sets.{n} is not the set {n}"
        stored[n], HOME[n] = [(FILE_NAME[fk], y["test"])], (fk, n)
for n, where, key in (("lobo_w_nostatic", "variants", "lobo_w_nostatic"), ("lobo_ref_w_nostatic", "references", "lobo_ref_w_nostatic")):
    assert LO[where][key]["saved_as"] == n
    stored[n] = [("lstm_lobo.json", LO[where][key]["test"])]
for n in E3_SETS:                                            # script 17 stores the pooled summary and the per-river scores of these sets, not the whole evaluation
    assert MB["design"]["sets"][n]["saved_as"] == n and n in MB["summary_test"]["sets"] and n not in stored
    stored[n] = []
SUMMARY_STORED = E3_SETS
SET_VERDICTS = {n: FILES[fk]["verdicts"]["sets"][key] for n, (fk, key) in HOME.items()}                 # the verdict tables that the scripts of the added sets stored
SET_VERDICTS.update({n: MB["verdicts"]["sets"][n] for n in E3_SETS + LOBO_NEW + ("lobo_w_static",)})
assert set(SET_VERDICTS) == set(WARN_NEW) and all(n in stored for n in SETS_NEW), "an added set has no stored scores or no stored verdicts"
for n in SETS:
    if n in FA["skill"]:
        stored[n].append(("flood_analysis.json", FA["skill"][n]["plain"]["test"]))
frame = lambda n: {g: pd.DataFrame({"obs": obs[i], **{lk(k): P[n][3][i, :, j] for j, k in enumerate(F.LEADS)}}, index=days) for i, g in enumerate(ids)}
ENG = {}
sst_all = soo.sum(1) - so.sum(1) ** 2 / n_y.sum(1)           # per basin: squared distance of the observed flow from its mean, all test days
for n in SETS:
    E = ENG[n] = F.evaluate(frame(n), thr_of)                # the engine's own scores of the stored forecasts: test years, high-flow level itself
    for where, S in stored[n]:
        c0 = n_compared
        assert same(E, S), f"results/{where} does not hold the scores of results/preds/{n}.npz: rerun the script that writes it"
        tally(where, c0, n)
    if n in SUMMARY_STORED:
        c0 = n_compared
        check_summary(n, E)
        tally("more_basins.json", c0, n)
    eng = lambda *path: np.array([[E["basins"][g][lk(k)][path[0]] if len(path) == 1 else E["basins"][g][lk(k)][path[0]][path[1]] for k in F.LEADS] for g in ids])
    assert (eng("coverage") == 1).all() and (eng("n") == n_y.sum(1)[:, None]).all(), f"{n}: not scored on every test day"
    assert np.abs(1 - sse[n].sum(1) / sst_all[:, None] - eng("nse")).max() < 1e-9, f"{n}: the yearly sums do not give the engine's NSE"
    n_compared += 3 * eng("nse").size
    if n in WARN:
        for kind, total in (("all", np.repeat(n_hi.sum(1)[:, None], len(F.LEADS), 1)), ("onset", n_on.sum(1))):
            hits, fa = W[n]["hit_all" if kind == "all" else "hit_on"].sum(1), W[n]["fa_all" if kind == "all" else "fa_on"].sum(1)
            assert (hits == eng(kind, "hits")).all() and (total - hits == eng(kind, "misses")).all() and (fa == eng(kind, "false_alarms")).all(), \
                f"{n}: the yearly counts are not those of flood.flood_scores ({kind})"
            n_compared += 3 * hits.size
for g in BEST:                                               # the best simple model: the highest median NSE in the validation years, recomputed here from the stored forecasts
    c = A["comparison"][g]["baseline_val_nse_median"]        # {simple model: [its median NSE in the validation years at each lead]}, as stored by 04_lstm_all.py
    assert all(n in SETS for n in c), f"a simple model of the comparison '{g}' in results/lstm_all.json is not among the sets read here"
    val = {n: F.evaluate(frame(n), thr_of, "val")["summary"] for n in c}
    for j, k in enumerate(F.LEADS):
        assert all(close(val[n][lk(k)]["nse_median"], c[n][j]) for n in c) and BEST[g][lk(k)] == max(c, key=lambda n: val[n][lk(k)]["nse_median"]), \
            f"the best simple model of results/lstm_all.json ({g}, {lk(k)}) is not the one with the highest median NSE in the validation years"
print(f"{len(SETS)} prediction sets, {nb} basins, {len(T)} test days in {ny} water years ({YEARS[0]}-{YEARS[-1]}): the yearly sums give the scores of flood.evaluate, "
      f"and those are the stored ones; the best simple model is the one of the validation years ({time.time() - t_start:.0f} s)", flush=True)


# ---- every statistic, for many resamples at once ------------------------------------------------------------------------------
def ratio(a, b):
    """a / b; no value (NaN) where b is 0, as in flood.rates."""
    return np.divide(a, b, out=np.full(np.shape(a), np.nan), where=b != 0)


def stats(WB, MY, BD):
    """{name: values [resample]} of every statistic. WB [resample, basin] and MY [resample, year] say how often each basin and each
    test water year is in the resample; BD [resample, draw] lists its basins. All ones and the basins in order: the test years as they are."""
    rows = np.arange(len(WB))[:, None]
    med = lambda v: (np.nanmedian if np.isnan(v).any() else np.median)(v[rows, BD], axis=1)      # v [resample, basin] -> median over the basins of the resample
    pool = lambda a: np.einsum("rb,by,ry->r", WB, a, MY)                   # a count [basin, year] summed over the basins and years of the resample
    n_days, s_o = MY @ n_y.T, MY @ so.T                                    # days and total observed flow of the resampled days [resample, basin]
    sst = MY @ soo.T - s_o ** 2 / n_days                                   # squared distance of the observed flow from its mean over those days
    err = {n: np.einsum("ry,byl->rbl", MY, sse[n]) for n in SETS}          # squared error [resample, basin, lead]
    s, new = {}, {}                                                        # s: the statistics of 4 October (unchanged); new: those added on 2026-10-08
    for j, k in enumerate(F.LEADS):
        nse = {n: 1 - err[n][:, :, j] / sst for n in SETS}                 # NSE of every basin on the resampled days [resample, basin]
        nse.update({f"best_simple_{g}": nse[BEST[g][lk(k)]] for g in BEST})
        for n in nse:
            (new if n in SETS_NEW else s)[f"nse_median.{n}.{lk(k)}"] = med(nse[n])
        for name, (a, b) in PAIRS.items():
            into = new if name in PAIRS_NEW else s
            into[f"nse_diff_of_medians.{name}.{lk(k)}"] = med(nse[a]) - med(nse[b])
            into[f"nse_median_of_diffs.{name}.{lk(k)}"] = med(nse[a] - nse[b])
        for n in SKILL:                                                    # 1 - squared error of the model / squared error of 'tomorrow = today', per basin
            s[f"skill_vs_persistence_median.{n}.{lk(k)}"] = med(1 - err[n][:, :, j] / err["base_persistence"][:, :, j])
        for n in DIAG:                                                     # spread, correlation and total of the forecasts against the observed flow, and the KGE they give
            s_p, s_pp, s_po = (MY @ x[:, :, j].T for x in mom[n])
            var_p, cov = s_pp - s_p ** 2 / n_days, s_po - s_p * s_o / n_days                     # both times the number of days, like sst
            a, r, b = np.sqrt(var_p / sst), cov / np.sqrt(var_p * sst), s_p / s_o
            for part, v in zip(("spread_ratio", "correlation", "total_ratio", "kge"), (a, r, b, 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))):
                s[f"{part}_median.{n}.{lk(k)}"] = med(v)
        days_hi, onsets = pool(n_hi), pool(n_on[:, :, j])
        plain = {**W, **{f"best_simple_{g}": W[BEST[g][lk(k)]] for g in BEST}}
        for tag, group in (("", plain), (AT, WC)):                         # warnings: the basins counted together
            for n, t in group.items():
                into = new if n in WARN_NEW else s
                h, f, h_on, f_on = (pool(t[c][:, :, j]) for c in ("hit_all", "fa_all", "hit_on", "fa_on"))
                into[f"onsets_warned{tag}.{n}.{lk(k)}"], into[f"onset_false_warnings{tag}.{n}.{lk(k)}"] = ratio(h_on, onsets), ratio(f_on, h_on + f_on)
                into[f"all_warned{tag}.{n}.{lk(k)}"], into[f"all_false_warnings{tag}.{n}.{lk(k)}"] = ratio(h, days_hi), ratio(f, h + f)
                into[f"all_csi{tag}.{n}.{lk(k)}"], into[f"onset_csi{tag}.{n}.{lk(k)}"] = ratio(h, days_hi + f), ratio(h_on, onsets + f_on)
        for g, (base, plus, ctrl, top) in FAMILY.items():                  # differences in the share of onsets warned
            a, b, c, d = (s[f"onsets_warned.{x}.{lk(k)}"] for x in (base, plus, ctrl, top))
            s[f"onsets_gain.evening_rain_minus_base.{g}.{lk(k)}"], s[f"onsets_gain.evening_rain_minus_control.{g}.{lk(k)}"] = b - a, b - c
            s[f"onsets_gain.upper_bound_minus_base.{g}.{lk(k)}"] = d - a
            s[f"onsets_gap_share_closed.{g}.{lk(k)}"], s[f"onsets_gap_share_closed_beyond_control.{g}.{lk(k)}"] = ratio(b - a, d - a), ratio(b - c, d - a)
            for name, other in (("best_simple", f"best_simple_{g}"), ("rf", f"base_rf_{g}"), ("linreg", f"base_linreg_{g}")):
                s[f"onsets_gain.lstm_minus_{name}.{g}.{lk(k)}"] = a - s[f"onsets_warned.{other}.{lk(k)}"]
        on = lambda n: (new if n in WARN_NEW else s)[f"onsets_warned.{n}.{lk(k)}"]                   # share of onsets warned (the sets of both dicts)
        for name in PAIRS_NEW:                                             # the same differences as above, in the share of onsets warned
            a, b = PAIRS[name]
            new[f"onsets_diff.{name}.{lk(k)}"] = on(a) - on(b)
        for name, (w, base, ceil, ctrl) in CLOSE.items():                  # the share of the gap to an upper bound, in onsets warned, that a set closes
            new[f"onsets_gap_share_closed.{name}.{lk(k)}"] = ratio(on(w) - on(base), on(ceil) - on(base))
            if ctrl is not None:
                new[f"onsets_gap_share_closed_beyond_control.{name}.{lk(k)}"] = ratio(on(w) - on(ctrl), on(ceil) - on(base))
    return s, new


def direct(bd, yd):
    """One resample scored the slow way: the days of the drawn years are laid end to end for the drawn basins and scored by flood.nse and flood.contingency."""
    d = np.concatenate([np.flatnonzero(yi == y) for y in yd])              # a year drawn twice appears twice
    sel, out = np.ix_(bd, d), {}
    for j, k in enumerate(F.LEADS):
        nse = {n: np.array([F.nse(o[i, d], PT[n][i, d, j]) for i in bd]) for n in SETS}
        nse.update({f"best_simple_{g}": nse[BEST[g][lk(k)]] for g in BEST})
        for n in SETS:
            out[f"nse_median.{n}.{lk(k)}"] = np.median(nse[n])
        for name, (a, b) in PAIRS.items():
            out[f"nse_median_of_diffs.{name}.{lk(k)}"] = np.median(nse[a] - nse[b])
        for n in SKILL:
            out[f"skill_vs_persistence_median.{n}.{lk(k)}"] = np.median([1 - np.sum((PT[n][i, d, j] - o[i, d]) ** 2) / np.sum((PT["base_persistence"][i, d, j] - o[i, d]) ** 2) for i in bd])
        for n in DIAG:
            parts = np.array([[PT[n][i, d, j].std() / o[i, d].std(), np.corrcoef(o[i, d], PT[n][i, d, j])[0, 1], PT[n][i, d, j].mean() / o[i, d].mean(), F.kge(o[i, d], PT[n][i, d, j])] for i in bd])
            for part, v in zip(("spread_ratio", "correlation", "total_ratio", "kge"), parts.T):
                out[f"{part}_median.{n}.{lk(k)}"] = np.nanmedian(v)
        for tag, n, level in [("", n, 1.0) for n in WARN] + [(AT, n, CHOSEN[n][j]) for n in CHOSEN]:
            oh, w, lw = hi[sel], (PT[n][:, :, j] > level * thr[:, None])[sel], low[:, :, j][sel]
            for kind, t in (("all", F.contingency(oh, w)), ("onset", F.contingency(oh[lw], w[lw]))):
                for name, rate in zip(RATE[kind], ("pod", "far", "csi")):
                    out[f"{name}{tag}.{n}.{lk(k)}"] = t[rate]
    return out


# ---- check 2: the statistics on the test years as they are, against the engine and the stored files ---------------------------
with np.errstate(invalid="ignore", divide="ignore"):         # a forecast that never changes has no correlation: NaN, as in flood.kge
    p_old, p_new = stats(np.ones((1, nb)), np.ones((1, ny)), np.arange(nb)[None])
point = {key: float(v[0]) for key, v in {**p_old, **p_new}.items()}
OLD_KEYS = set(p_old)                                        # the statistics of 4 October; the others were added on 2026-10-08
assert not OLD_KEYS & set(p_new), "a statistic of 4 October is also among the added ones"
for n in SETS:
    for k in F.LEADS:
        e = ENG[n]["summary"][lk(k)]
        assert close(point[f"nse_median.{n}.{lk(k)}"], e["nse_median"]), f"median NSE of {n}"
        for kind in RATE if n in WARN else ():
            assert all(close(point[f"{name}.{n}.{lk(k)}"], e[kind][rate]) for name, rate in zip(RATE[kind], ("pod", "far", "csi"))), f"warnings of {n} ({kind})"
for j, k in enumerate(F.LEADS):
    for g in BEST:                                           # LSTM against the best simple model, as stored by 04_lstm_all.py
        c = A["comparison"][g]["baselines"][BEST[g][lk(k)]][lk(k)]
        assert close(point[f"nse_diff_of_medians.lstm_{g}_minus_best_simple.{lk(k)}"], c["difference_of_medians"]), "LSTM minus best simple model"
        assert close(point[f"nse_median_of_diffs.lstm_{g}_minus_best_simple.{lk(k)}"], c["median_nse_difference"]), "LSTM minus best simple model, basin by basin"
    for g, name in (("wq", "wqn"), ("w", "wn")):             # rain up to the evening of the issue day, as stored by 10_rain_timing.py
        if name in RT["closure"]:
            c = RT["closure"][name]["by_lead"][j]
            assert c["lead"] == k and close(point[f"onsets_gap_share_closed.{g}.{lk(k)}"], c["onset_share_of_gap_closed"]) \
                and close(point[f"onsets_gap_share_closed_beyond_control.{g}.{lk(k)}"], c["onset_share_of_gap_closed_beyond_control"]), "share of the onset gap closed"
            if g == "wq":
                assert close(point[f"nse_diff_of_medians.wqn_minus_wq.{lk(k)}"], c["gain"]) and close(point[f"nse_median_of_diffs.wqn_minus_wq.{lk(k)}"], c["median_basin_gain"]) \
                    and close(point[f"nse_median_of_diffs.wqn_minus_wqn_older.{lk(k)}"], c["median_basin_gain_over_control"]), "NSE gain from the evening rain"
    c = LO["summary"][lk(k)]                                 # a river never seen, as stored by 05_lobo.py
    assert close(point[f"nse_median_of_diffs.held_out_minus_in_training.{lk(k)}"], -c["loss_from_not_seeing_the_basin"]["with_properties"]["median_nse_loss"]) \
        and close(point[f"nse_diff_of_medians.held_out_minus_in_training.{lk(k)}"], c["nse_median"]["held_out_with_properties"] - c["nse_median"]["in_training"]), "unseen rivers"
    assert close(point[f"nse_diff_of_medians.transformer_minus_lstm.{lk(k)}"], MC["comparison"]["test"][lk(k)]["nse_median_transformer_minus_lstm"]), "Transformer minus LSTM"
    for n in CHOSEN:                                         # the warning tables at the level chosen on the validation years, as stored by 06_flood_analysis.py
        c = FA["skill"][n]["tuned"][lk(k)]
        assert n not in TABLE or FA["verdict"]["services"][n][lk(k)]["trigger"] == CHOSEN[n][j], f"{n}: two different chosen warning levels in flood_analysis.json"
        for kind, total in (("all", n_hi.sum()), ("onset", n_on[:, :, j].sum())):
            hits, fa = WC[n]["hit_all" if kind == "all" else "hit_on"][:, :, j].sum(), WC[n]["fa_all" if kind == "all" else "fa_on"][:, :, j].sum()
            t = c["test"][kind]
            assert close(int(hits), t["hits"]) and close(int(total - hits), t["misses"]) and close(int(fa), t["false_alarms"]), f"{n}: counts at the chosen warning level ({kind})"
            assert all(close(point[f"{name}{AT}.{n}.{lk(k)}"], t[rate]) for name, rate in zip(RATE[kind], ("pod", "far", "csi"))), f"{n}: rates at the chosen warning level ({kind})"


# the differences and closures of 2026-10-08 on the test years as they are, against the comparison numbers that the four scripts stored
def dig(obj, *path):
    for p in path:
        obj = obj[p]
    return obj


def check_pair(fk, path, kind, name):
    """The statistics of one difference of PAIRS_NEW equal the numbers a results file stores for it (kinds: gain_rule, vs_simple, new_input, closure_gain, with_without, loss)."""
    a, b = PAIRS_NEW[name]
    entry = dig(FILES[fk], *path)
    where = f"results/{FILE_NAME[fk]} {'/'.join(path)}"
    for k in F.LEADS:
        if kind == "gain_rule":                              # gain_rule and unseen.against_the_leave_one_out_models: an nse and an onset block per lead
            c = entry[lk(k)]
            assert c["against"] == b and c["lead"] == k, f"{where}: not the comparison with {b}"
            got = [(c["nse"]["median_with"], f"nse_median.{a}"), (c["nse"]["median_without"], f"nse_median.{b}"), (c["nse"]["gain"], f"nse_diff_of_medians.{name}"),
                   (c["nse"]["median_basin_difference"], f"nse_median_of_diffs.{name}"), (c["onset"]["warned_with"], f"onsets_warned.{a}"),
                   (c["onset"]["warned_without"], f"onsets_warned.{b}"), (c["onset"]["gain"], f"onsets_diff.{name}"),
                   (c["onset"]["false_with"], f"onset_false_warnings.{a}"), (c["onset"]["false_without"], f"onset_false_warnings.{b}")]
        elif kind == "vs_simple":                            # yardstick.lstm_against_simple: the LSTM against a simple model on the same input
            c = entry[lk(k)]
            got = [(c["lstm_nse_median"], f"nse_median.{a}"), (c["baseline_nse_median"], f"nse_median.{b}"), (c["difference_of_medians"], f"nse_diff_of_medians.{name}"),
                   (c["median_nse_difference"], f"nse_median_of_diffs.{name}"), (c["lstm_onset_pod"], f"onsets_warned.{a}"), (c["baseline_onset_pod"], f"onsets_warned.{b}"),
                   (c["lstm_onset_pod"] - c["baseline_onset_pod"], f"onsets_diff.{name}"), (c["lstm_onset_far"], f"onset_false_warnings.{a}"),
                   (c["baseline_onset_far"], f"onset_false_warnings.{b}")]
        elif kind == "new_input":                            # yardstick.new_input_given_to_each_model: a simple model with the new input against the same model without it
            c = entry["by_lead"][lk(k)]
            got = [(c["nse_median_with"], f"nse_median.{a}"), (c["nse_median_without"], f"nse_median.{b}"), (c["nse_gain"], f"nse_diff_of_medians.{name}"),
                   (c["median_basin_difference"], f"nse_median_of_diffs.{name}"), (c["onset_warned_with"], f"onsets_warned.{a}"),
                   (c["onset_warned_without"], f"onsets_warned.{b}"), (c["onset_gain_points"] / 100, f"onsets_diff.{name}"),
                   (c["onset_false_with"], f"onset_false_warnings.{a}"), (c["onset_false_without"], f"onset_false_warnings.{b}")]
        elif kind == "closure_gain":                         # closure.<...>.by_lead: a set against the base of that closure
            c = entry["by_lead"][F.LEADS.index(k)]
            assert entry["set"] == a and entry["base"] == b and c["lead"] == k, f"{where}: not the comparison of {a} with {b}"
            got = [(c["nse_with"], f"nse_median.{a}"), (c["nse_base"], f"nse_median.{b}"), (c["gain"], f"nse_diff_of_medians.{name}"), (c["median_basin_gain"], f"nse_median_of_diffs.{name}"),
                   (c["onset_warned_with"], f"onsets_warned.{a}"), (c["onset_warned_base"], f"onsets_warned.{b}"), (c["onset_warned_with"] - c["onset_warned_base"], f"onsets_diff.{name}"),
                   (c["onset_false_with"], f"onset_false_warnings.{a}"), (c["onset_false_base"], f"onset_false_warnings.{b}")]
        elif kind == "with_without":                         # a with_set / without_set block per lead (aorc_against_nldas_and_era5_together, rain_to_midnight_given_the_forecast)
            c = entry[lk(k)]
            assert c["with_set"] == a and c["without_set"] == b, f"{where}: not the comparison of {a} with {b}"
            got = [(c["nse"]["median_with"], f"nse_median.{a}"), (c["nse"]["median_without"], f"nse_median.{b}"), (c["nse"]["gain"], f"nse_diff_of_medians.{name}"),
                   (c["nse"]["median_basin_difference"], f"nse_median_of_diffs.{name}"), (c["onsets"]["warned_with"], f"onsets_warned.{a}"),
                   (c["onsets"]["warned_without"], f"onsets_warned.{b}"), (c["onsets"]["difference_points"] / 100, f"onsets_diff.{name}"),
                   (c["onsets"]["false_warnings_with"], f"onset_false_warnings.{a}"), (c["onsets"]["false_warnings_without"], f"onset_false_warnings.{b}")]
        else:                                                # unseen.loss_from_not_having_seen_the_river: median over the rivers of (partner minus never seen)
            got = [(-entry[lk(k)]["median_loss"], f"nse_median_of_diffs.{name}")]
        for value, stem in got:
            assert close(point[f"{stem}.{lk(k)}"], value), f"{where}, {lk(k)}: {stem} is {point[f'{stem}.{lk(k)}']} here but {value} there"


def check_closure(fk, path, name, beyond):
    """The closure statistics equal the shares of the gap that a results file stores (closure.*.by_lead, gap_to_the_upper_bound)."""
    w, base, ceil, ctrl = CLOSE[name]
    entry = dig(FILES[fk], *path)
    where = f"results/{FILE_NAME[fk]} {'/'.join(path)}"
    if fk == "MB":
        rows = [entry[lk(k)] for k in F.LEADS]
    else:
        assert entry["base"] == base and entry["ceiling"] == ceil and entry.get("set", w) == w and [r["lead"] for r in entry["by_lead"]] == list(F.LEADS), \
            f"{where}: not the closure {name}"
        rows = entry["by_lead"]
    for k, c in zip(F.LEADS, rows):
        assert close(point[f"onsets_gap_share_closed.{name}.{lk(k)}"], c["onset_share_of_gap_closed"]), f"{where}, {lk(k)}: share of the onset gap closed"
        if beyond:
            assert ctrl is not None and close(point[f"onsets_gap_share_closed_beyond_control.{name}.{lk(k)}"], dig(c, *beyond)), f"{where}, {lk(k)}: share closed beyond the control"


STORED_PAIRS = []                                            # (file, path to the stored comparison, kind, name in PAIRS_NEW)
for fk, kind, head, rows in (
        ("RM", "gain_rule", ("gain_rule",),
         (("wqh", "against_rain_table_alone", "wqh_minus_wq"), ("wqh", "against_control", "wqh_minus_wqh_older"), ("wqh", "control_against_rain_table_alone", "wqh_older_minus_wq"),
          ("wqh", "against_nldas_rain", "wqh_minus_wqn"), ("wqh", "against_matched_window_to_09", "wqh_minus_wqh_to09"), ("wqh", "against_utc_day", "wqh_minus_wqh_utc"),
          ("wqh", "matched_window_to_09_against_rain_table_alone", "wqh_to09_minus_wq"), ("wqh", "utc_day_against_rain_table_alone", "wqh_utc_minus_wq"),
          ("wqh", "utc_day_against_matched_window_to_09", "wqh_utc_minus_wqh_to09"), ("wqnh", "against_rain_to_midnight", "wqnh_minus_wqh"),
          ("wqnh", "against_nldas_rain", "wqnh_minus_wqn"), ("wqnh", "against_rain_table_alone", "wqnh_minus_wq"), ("wh", "against_rain_table_alone", "wh_minus_w"),
          ("wh", "against_control", "wh_minus_wh_older"), ("wh", "control_against_rain_table_alone", "wh_older_minus_w"), ("wh", "against_nldas_rain", "wh_minus_wn"))),
        ("RM", "vs_simple", ("yardstick", "lstm_against_simple"),
         (("lstm_wqh__base_linreg_wqh", "lstm_wqh_minus_linreg_wqh"), ("lstm_wqh__base_rf_wqh", "lstm_wqh_minus_rf_wqh"))),
        ("RM", "new_input", ("yardstick", "new_input_given_to_each_model"),
         (("ridge regression", "linreg_wqh_minus_linreg_wq"), ("random forest", "rf_wqh_minus_rf_wq"))),
        ("AO", "gain_rule", ("gain_rule",),
         (("wqa", "against_rain_table_alone", "wqa_minus_wq"), ("wqa", "against_control", "wqa_minus_wqa_to09"), ("wqa", "against_utc_day", "wqa_minus_wqa_utc"),
          ("wqa", "control_against_rain_table_alone", "wqa_to09_minus_wq"), ("wqa", "utc_day_against_rain_table_alone", "wqa_utc_minus_wq"),
          ("wqa", "utc_day_against_matched_window_to_09", "wqa_utc_minus_wqa_to09"), ("wqa", "against_era5_rain_to_midnight", "wqa_minus_wqh"),
          ("wqa", "against_nldas_rain", "wqa_minus_wqn"), ("wqa_rep", "against_rain_table_alone", "wqa_rep_minus_wq"),
          ("wqa_rep", "against_aorc_as_an_extra_column", "wqa_rep_minus_wqa"), ("wqa_rep", "against_nldas_rain", "wqa_rep_minus_wqn"),
          ("wqa_rep", "against_era5_rain_to_midnight", "wqa_rep_minus_wqh"))),
        ("AO", "with_without", ("aorc_against_nldas_and_era5_together",), (("by_lead", "wqa_minus_wqnh"),)),
        ("AO", "vs_simple", ("yardstick", "lstm_against_simple"),
         (("lstm_wqa__base_linreg_wqa", "lstm_wqa_minus_linreg_wqa"), ("lstm_wqa__base_rf_wqa", "lstm_wqa_minus_rf_wqa"))),
        ("AO", "new_input", ("yardstick", "new_input_given_to_each_model"),
         (("ridge regression", "linreg_wqa_minus_linreg_wq"), ("random forest", "rf_wqa_minus_rf_wq"))),
        ("MB", "gain_rule", ("gain_rule",),
         (("lstm63_wq_against_lstm_wq", "lstm63_wq_minus_wq"), ("lstm63_w_against_lstm_w", "lstm63_w_minus_w"), ("lstm63eq_wq_against_lstm_wq", "lstm63eq_wq_minus_wq"),
          ("lstm63eq_wq_against_lstm63_wq", "lstm63eq_wq_minus_lstm63_wq"))),
        ("MB", "gain_rule", ("unseen", "against_the_leave_one_out_models"),
         (("unseen50_w_static_against_lobo_w_static", "unseen50_w_static_minus_lobo_w_static"),
          ("unseen50_w_nostatic_against_lobo_w_nostatic", "unseen50_w_nostatic_minus_lobo_w_nostatic"),
          ("unseen50_w_static_against_unseen50_w_nostatic", "unseen50_w_static_minus_unseen50_w_nostatic"),
          ("unseen50_w_static_against_lstm63_w", "unseen50_w_static_minus_lstm63_w"))),
        ("MB", "loss", ("unseen", "loss_from_not_having_seen_the_river"),
         (("leave_one_out_without_properties__05_recomputed", "lobo_w_nostatic_minus_lobo_ref_w_nostatic"),
          ("50_rivers_with_properties__against_12_river_seed_average", "unseen50_w_static_minus_lstm_w"),
          ("50_rivers_with_properties__against_63_river_seed_average", "unseen50_w_static_minus_lstm63_w"),
          ("50_rivers_without_properties__against_12_river_model", "unseen50_w_nostatic_minus_lobo_ref_w_nostatic"))),
        ("FI", "gain_rule", ("gain_rule",),
         (("wqaF_sub_against_wqa", "wqaF_sub_minus_wqa"), ("wqaF_sub_against_wqaF_clim", "wqaF_sub_minus_wqaF_clim"), ("wqaF_sub_against_wqaP", "wqaF_sub_minus_wqaP"),
          ("wqF_sub_against_wq", "wqF_sub_minus_wq"), ("wqaF_clim_against_wqa", "wqaF_clim_minus_wqa"), ("wqaP_against_wqa", "wqaP_minus_wqa"))),
        ("FI", "closure_gain", ("closure",), (("wqaF_sub__base_wq__ceiling_wqf", "wqaF_sub_minus_wq"),)),
        ("FI", "with_without", ("forecast_without_rain_to_midnight",), (("rain_to_midnight_given_the_forecast", "wqaF_sub_minus_wqF_sub"),)),
        ("FI", "vs_simple", ("yardstick", "lstm_against_simple"),
         (("lstm_wqaF_sub__base_linreg_wqaF", "lstm_wqaF_sub_minus_linreg_wqaF"), ("lstm_wqaF_sub__base_rf_wqaF", "lstm_wqaF_sub_minus_rf_wqaF"))),
        ("FI", "new_input", ("yardstick", "new_input_given_to_each_model"),
         (("ridge regression", "linreg_wqaF_minus_linreg_wqa"), ("random forest", "rf_wqaF_minus_rf_wqa")))):
    STORED_PAIRS += [(fk, head + r[:-1], kind, r[-1]) for r in rows]
CLOSE_REF = ("references", "control_window_09_to_09", "onset_share_of_gap_closed_beyond_it")      # where script 16 stores the share closed beyond its matched-window control
STORED_CLOSE = ([("RM", ("closure", s), name, ("onset_share_of_gap_closed_beyond_control",) if s in ("wqh", "wh") else None)
                 for s, name in (("wqh", "wqh__base_wq__ceiling_wqf"), ("wh", "wh__base_w__ceiling_wf"), ("wqh_to09", "wqh_to09__base_wq__ceiling_wqf"),
                                 ("wqh_utc", "wqh_utc__base_wq__ceiling_wqf"), ("wqnh", "wqnh__base_wq__ceiling_wqf"))]
                + [("AO", ("closure", s), name, CLOSE_REF if s == "wqa" else None)
                   for s, name in (("wqa", "wqa__base_wq__ceiling_wqf"), ("wqa_to09", "wqa_to09__base_wq__ceiling_wqf"), ("wqa_utc", "wqa_utc__base_wq__ceiling_wqf"),
                                   ("wqa_rep", "wqa_rep__base_wq__ceiling_wqf"))]
                + [("MB", ("gap_to_the_upper_bound", "sets", s), name, None)
                   for s, name in (("lstm63_wq_against_lstm_wq", "lstm63_wq__base_wq__ceiling_wqf"), ("lstm63eq_wq_against_lstm_wq", "lstm63eq_wq__base_wq__ceiling_wqf"))]
                + [("FI", ("closure", name), name, None) for name in ("wqaF_sub__base_wqa__ceiling_wqaP", "wqaF_sub__base_wqa__ceiling_wqf", "wqaF_sub__base_wq__ceiling_wqf",
                                                                      "wqaF_clim__base_wqa__ceiling_wqaP", "wqF_sub__base_wq__ceiling_wqf", "wqF_sub__base_wq__ceiling_wqaP",
                                                                      "wqa__base_wq__ceiling_wqaP", "wqaP__base_wqa__ceiling_wqf")])
assert len({(fk, path) for fk, path, _, _ in STORED_PAIRS}) == len(STORED_PAIRS) and all(name in PAIRS_NEW for _, _, _, name in STORED_PAIRS), "a stored comparison is listed twice or names no difference"
PAIRS_UNSTORED = sorted(set(PAIRS_NEW) - {name for _, _, _, name in STORED_PAIRS})           # differences with no stored counterpart (plain arithmetic of checked statistics)
assert not set(CLOSE) - {name for _, _, name, _ in STORED_CLOSE}, "a closure has no stored counterpart"
if F.QUICK:         # in a smoke test the stored comparisons set smoke-test sets beside the real controls of the older scripts: only the comparisons between two added sets are made
    CHECK_PAIRS = [row for row in STORED_PAIRS if all(x in SETS_NEW and x not in LOBO_NEW for x in PAIRS_NEW[row[3]])]
    CHECK_CLOSE = []
    print(f"smoke test: {len(STORED_PAIRS) - len(CHECK_PAIRS)} of the {len(STORED_PAIRS)} stored differences and all {len(STORED_CLOSE)} stored closures are not compared (their controls are real results)", flush=True)
else:
    CHECK_PAIRS, CHECK_CLOSE = STORED_PAIRS, STORED_CLOSE
n_before_extension = n_compared
for fk, path, kind, name in CHECK_PAIRS:
    c0 = n_compared
    check_pair(fk, path, kind, name)
    tally(FILE_NAME[fk], c0)
for fk, path, name, beyond in CHECK_CLOSE:
    c0 = n_compared
    check_closure(fk, path, name, beyond)
    tally(FILE_NAME[fk], c0)
print(f"the statistics on the test years as they are equal the stored numbers ({n_compared:,} numbers compared so far; {n_compared - n_before_extension:,} of them the differences and closures "
      f"of the added sets against {len(CHECK_PAIRS)} + {len(CHECK_CLOSE)} stored comparisons)", flush=True)
n_before = n_compared                                        # the numbers compared before anything is resampled

# ---- the verdict rule, read from the stored criteria ------------------------------------------------------------------------------
crit = FA["verdict"]["criteria"]
m_go, m_caution = re.fullmatch(r"pod >= ([\d.]+) and far <= ([\d.]+)", crit["GO"]), re.fullmatch(r"csi >= ([\d.]+) but not GO", crit["CAUTION"])
assert m_go and m_caution and crit["NOT YET"] == "otherwise", "the verdict criteria in results/flood_analysis.json are not written as this script reads them"
GO_POD, GO_FAR, CAUTION_CSI = float(m_go[1]), float(m_go[2]), float(m_caution[1])
VERDICTS = ("GO", "CAUTION", "NOT YET")


def verdict(pod, far, csi):
    """GO / CAUTION / NOT YET for numbers or arrays (no warning given: the false share has no value, so it is not GO - as in 06_flood_analysis.py)."""
    return np.where((pod >= GO_POD) & (far <= GO_FAR), "GO", np.where(csi >= CAUTION_CSI, "CAUTION", "NOT YET"))


# ---- part 2: the resamples -----------------------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
BD, YD = np.empty((B, nb), int), np.empty((B, ny), int)
for r in range(B):                                           # for each resample the basins, then the years, drawn with replacement
    BD[r], YD[r] = rng.integers(0, nb, nb), rng.integers(0, ny, ny)
WB = np.stack([np.bincount(x, minlength=nb) for x in BD]).astype(float)                                 # how often each basin was drawn [resample, basin]
MY = np.stack([np.bincount(x, minlength=ny) for x in YD]).astype(float)                                 # how often each year was drawn [resample, year]
in_order, once_b, once_y = np.tile(np.arange(nb), (B, 1)), np.ones((B, nb)), np.ones((B, ny))
designs = {}
with np.errstate(invalid="ignore", divide="ignore"):
    for d, args in dict(years=(once_b, MY, in_order), basins=(WB, once_y, BD), both=(WB, MY, BD)).items():
        t_d = time.time()
        s_old, s_new = stats(*args)
        designs[d] = {**s_old, **s_new}
        print(f"  {B:,} resamples of {d:6s} done ({time.time() - t_d:.1f} s)", flush=True)
    worst = 0.0
    for r in range(min(N_DIRECT, B)):                        # check 3: the fast sums against a direct scoring of the same resample
        for key, x in direct(BD[r], YD[r]).items():
            y = designs["both"][key][r]
            assert (x != x and y != y) or abs(x - y) < 1e-9, f"resample {r}: {key} is {y} from the yearly sums but {x} when scored directly"
            worst = max(worst, 0.0 if x != x else abs(x - y))
            n_compared += 1
    print(f"  the first {min(N_DIRECT, B)} resamples scored directly by flood.nse and flood.contingency give the same numbers (largest difference {worst:.1e})", flush=True)


def interval(v):
    """2.5th to 97.5th percentile of the resampled values, and the share of them above zero (None when no resample has a value)."""
    v = v[np.isfinite(v)]
    return dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), share_above_zero=float((v > 0).mean())) if v.size else dict(lo=None, hi=None, share_above_zero=None)


U = {key: dict(value=point[key], **{d: interval(designs[d][key]) for d in designs}) for key in point}
no_value = {key: {d: int((~np.isfinite(designs[d][key])).sum()) for d in designs} for key in point if any((~np.isfinite(designs[d][key])).any() for d in designs)}
with_range = [key for key in U if all(U[key][d]["lo"] is not None for d in designs)]
range_old = [key for key in with_range if key in OLD_KEYS]  # the statistics of 4 October: the aggregates below that were computed on 4 October stay over these
range_new = [key for key in with_range if key not in OLD_KEYS]
width = lambda key, d: U[key][d]["hi"] - U[key][d]["lo"]
widest = lambda keys: sum(width(key, "both") >= max(width(key, "years"), width(key, "basins")) for key in keys)      # statistics whose range is the widest when both are drawn again
n_widest, n_widest_new = widest(range_old), widest(range_new)


def mc_noise(v):
    """How far the two ends of interval(v) move with the random numbers (a standard error): the spread of the same two ends over MC_PARTS equal
    parts of the resamples, divided by the square root of MC_PARTS. NaN where fewer than two parts have a value."""
    ends = np.array([np.percentile(p[np.isfinite(p)], [2.5, 97.5]) for p in np.array_split(v, MC_PARTS) if np.isfinite(p).any()])
    return ends.std(0, ddof=1) / np.sqrt(len(ends)) if len(ends) > 1 else np.full(2, np.nan)


MC = {key: mc_noise(designs["both"][key]) for key in with_range}                                            # [noise of lo, noise of hi] of every interval of the design 'both'
mc_aggregates = lambda keys: (lambda a: (float(np.median(a)), float(a.max())) if a.size else (float("nan"), float("nan")))(np.array([x for key in keys for x in MC[key] if x == x]))
mc_typical, mc_largest = mc_aggregates(range_old)           # of the statistics of 4 October, as then
mc_largest_for = max(range_old, key=lambda key: np.nan_to_num(MC[key]).max()) if range_old else None
mc_typical_new, mc_largest_new = mc_aggregates(range_new)    # of the statistics added on 2026-10-08
mc_largest_for_new = max(range_new, key=lambda key: np.nan_to_num(MC[key]).max()) if range_new else None
NEAR = 4                                                     # an end closer to zero than NEAR times its noise (and not at zero itself): whether the range holds zero is not firm
near = lambda keys: [key for key in keys if any(U[key]["both"][e] != 0 and abs(U[key]["both"][e]) < NEAR * MC[key][j] for j, e in enumerate(("lo", "hi")))]
near_zero_old, near_zero_new = near(range_old), near(range_new)
near_zero = near_zero_old + near_zero_new                    # the statistics of 4 October first, then the added ones

# ---- part 3: how often each verdict comes out ------------------------------------------------------------------------------------
n_labels = 0                                                 # verdict labels compared with the ones that the scripts of the added sets stored
V = {}
with np.errstate(invalid="ignore"):
    for n in WARN:
        for kind, names in RATE.items():
            for k in F.LEADS:
                cell = lambda tag: dict(verdict=str(verdict(*(np.float64(point[f"{x}{tag}.{n}.{lk(k)}"]) for x in names))),
                                        share_of_resamples_with_each_verdict={x: float((verdict(*(designs["both"][f"{y}{tag}.{n}.{lk(k)}"] for y in names)) == x).mean()) for x in VERDICTS})
                V[f"{n}.{kind}.{lk(k)}"] = cell("")
                if n in SET_VERDICTS:                        # an added set: its verdict at the high-flow level itself is the one its own script stored
                    stored_label = SET_VERDICTS[n][lk(k)]["plain"][kind]["verdict"]
                    assert V[f"{n}.{kind}.{lk(k)}"]["verdict"] == stored_label, f"{n} {kind} {lk(k)}: the verdict is {V[f'{n}.{kind}.{lk(k)}']['verdict']} here but {stored_label} in the results file of its script"
                    n_labels += 1
                if n in CHOSEN:                              # the same cell at the warning level at which flood_analysis.json takes its verdicts
                    V[f"{n}.{kind}.{lk(k)}"][AT_KEY] = dict(warning_level=CHOSEN[n][F.LEADS.index(k)], **cell(AT))
                    c = FA["verdict"]["services"][n][lk(k)][kind] if n in TABLE else dict(pooled=FA["verdict"]["other_models_pooled"][n][lk(k)][kind])
                    assert V[f"{n}.{kind}.{lk(k)}"][AT_KEY]["verdict"] == c["pooled"] and V[f"{n}.{kind}.{lk(k)}"]["verdict"] == c.get("at_plain_level", V[f"{n}.{kind}.{lk(k)}"]["verdict"]), \
                        f"{n} {kind} {lk(k)}: not the verdict stored in results/flood_analysis.json"
same_verdict = [V[f"{n}.{kind}.{lk(k)}"]["verdict"] == V[f"{n}.{kind}.{lk(k)}"][AT_KEY]["verdict"] for n in TABLE for kind in RATE for k in F.LEADS]

# ---- part 4: spread, correlation, total, skill against 'tomorrow = today' and KGE (no resampling) -------------------------------
per_basin, medians = {}, {}
with np.errstate(invalid="ignore", divide="ignore"):
    for n in DIAG:
        per_basin[n] = {}
        for i, g in enumerate(ids):
            per_basin[n][g] = {}
            for j, k in enumerate(F.LEADS):
                x, y, ref = PT[n][i, :, j], o[i], PT["base_persistence"][i, :, j]
                c = dict(spread_ratio=float(x.std() / y.std()), correlation=float(np.corrcoef(y, x)[0, 1]) if np.ptp(x) else float("nan"), total_ratio=float(x.mean() / y.mean()),
                         skill_vs_persistence=float(1 - np.sum((x - y) ** 2) / np.sum((ref - y) ** 2)), kge=F.kge(y, x))
                assert close(c["kge"], ENG[n]["basins"][g][lk(k)]["kge"]), f"KGE of {n}"
                assert c["kge"] != c["kge"] or abs(c["kge"] - (1 - np.sqrt((c["correlation"] - 1) ** 2 + (c["spread_ratio"] - 1) ** 2 + (c["total_ratio"] - 1) ** 2))) < 1e-9, "the three parts do not give the KGE"
                per_basin[n][g][lk(k)] = c
        for k in F.LEADS:
            for part in PARTS:
                medians[f"{part}_median.{n}.{lk(k)}"] = float(np.nanmedian([per_basin[n][g][lk(k)][part] for g in ids]))
            assert close(medians[f"kge_median.{n}.{lk(k)}"], ENG[n]["summary"][lk(k)]["kge_median"]), f"median KGE of {n}"
            assert all(close(medians[f"{part}_median.{n}.{lk(k)}"], point[f"{part}_median.{n}.{lk(k)}"]) for part in PARTS if f"{part}_median.{n}.{lk(k)}" in point), \
                f"{n}: the medians of part 4 are not the statistics of the same name"
gap = lambda n, k: np.array([per_basin[n][g][lk(k)]["spread_ratio"] - per_basin[n][g][lk(k)]["correlation"] for g in ids])      # spread ratio minus correlation, basin by basin
spread_gap = {n: {lk(k): dict(min=float(np.nanmin(gap(n, k))), median=float(np.nanmedian(gap(n, k))), max=float(np.nanmax(gap(n, k)))) for k in F.LEADS} for n in DIAG}

# ---- notes: every number in them is read from the tables above -------------------------------------------------------------------
ok = lambda x: x is not None and x == x


def num(x, d=3, sign=False):
    """A number for a sentence; one that rounds to zero is written without a sign."""
    if not ok(x):
        return "no value"
    x = round(x, d) + 0
    return f"{x:+.{d}f}" if sign and x else f"{x:.{d}f}"


def share(x):
    """A share of the resamples in per cent; 0% and 100% only when it is none or all of them."""
    if not ok(x):
        return "no value"
    r = round(100 * x)
    return f"{r}%" if 0 < r < 100 or x in (0, 1) else f"{100 * x:.3f}".rstrip("0").rstrip(".") + "%"


sg, n3 = lambda x: num(x, sign=True), num                                                              # an NSE difference, an NSE
pc = lambda x: f"{round(100 * x) + 0}%" if ok(x) else "no value"                                        # a share in per cent
pt = lambda x: f"{round(100 * x) + 0}" if ok(x) else "no value"                                         # a difference of two shares, in percentage points
pts = lambda x: (f"{round(100 * x):+d}" if round(100 * x) else "0") if ok(x) else "no value"            # the same with its sign
count = lambda x: "no resample with a value" if not ok(x) else f"every one of the {B:,} resamples" if x == 1 else f"{round(x * B):,} of the {B:,} resamples"
rng_of = lambda key, f, flip=False: (f"{f(-U[key]['both']['hi'])} to {f(-U[key]['both']['lo'])}" if flip else f"{f(U[key]['both']['lo'])} to {f(U[key]['both']['hi'])}") if ok(U[key]["both"]["lo"]) else "no range"
vr = lambda key, f, flip=False: f"{f(-U[key]['value'] if flip else U[key]['value'])} ({rng_of(key, f, flip)})"                        # value (range)
by_lead = lambda stem, f, flip=False: ", ".join(vr(f"{stem}.{lk(k)}", f, flip) for k in F.LEADS)
holds_zero = lambda key: ok(U[key]["both"]["lo"]) and U[key]["both"]["lo"] <= 0 <= U[key]["both"]["hi"]
all_hold_zero = lambda *stems: all(holds_zero(f"{s}.{lk(k)}") for s in stems for k in F.LEADS)
ahead = lambda k: f"{k} day{'s' * (k > 1)} ahead"
leads_txt = " / ".join(str(k) for k in F.LEADS) + " days ahead"
simple = lambda n: LABEL[n].split(",")[0]
med_txt = lambda part, n: " / ".join(n3(medians[f"{part}_median.{n}.{lk(k)}"]) for k in F.LEADS)
vs = lambda key: V[key]["share_of_resamples_with_each_verdict"]


def best_txt(g):
    """Which simple model is the best one at which lead: 'ridge regression 1 day ahead, random forest 2 and 3 days ahead'."""
    who = {}
    for k in F.LEADS:
        who.setdefault(simple(BEST[g][lk(k)]), []).append(k)
    return ", ".join(f"{name} {' and '.join(str(k) for k in ks)} day{'s' * (ks[-1] > 1)} ahead" for name, ks in who.items())


def firm(cells, x="NOT YET"):
    """How often verdict x comes out in the cells of the three leads, in words."""
    v = [c["share_of_resamples_with_each_verdict"][x] for c in cells]
    return f"every one of the {B:,} resamples at each of {leads_txt}" if min(v) == 1 else " / ".join(share(s) for s in v) + f" of the resamples at {leads_txt}"


def reach(stem):
    """The largest distance from zero that the ranges of the three leads reach, rounded up to two decimals."""
    ends = [U[f"{stem}.{lk(k)}"]["both"][e] for k in F.LEADS for e in ("lo", "hi")]
    return f"{np.ceil(max(abs(x) for x in ends) * 100) / 100:.2f}" if all(ok(x) for x in ends) else "no value"


def one_inside_the_other(stem, sets):
    """Every set's value lies inside the range of every other set, at every lead."""
    e = lambda n, k: U[f"{stem}.{n}.{lk(k)}"]
    return all(ok(e(b, k)["both"]["lo"]) and e(b, k)["both"]["lo"] <= e(a, k)["value"] <= e(b, k)["both"]["hi"] for k in F.LEADS for a in sets for b in sets if a != b)


k1, d1 = lk(F.LEADS[0]), ahead(F.LEADS[0])
tie = "nse_median_of_diffs.lstm_wq_minus_best_simple"
rf_k, lin_k = f"onsets_gain.lstm_minus_rf.wq.{k1}", f"onsets_gain.lstm_minus_linreg.wq.{k1}"
wide = {d: U[f"nse_median.lstm_wq.{k1}"][d] for d in ("years", "basins")}
gain1 = np.array([ENG["lstm_wqf"]["basins"][g][k1]["nse"] - ENG["lstm_wq"]["basins"][g][k1]["nse"] for g in ids])             # what the perfect forecast adds, basin by basin
fitted = [n for n in DIAG if n not in ("base_persistence", "lstm_wqf", "lstm_w")]                                             # fitted models that read the river gauge and no forecast
kge_copy_higher = [k for k in F.LEADS if all(medians[f"kge_median.base_persistence.{lk(k)}"] > medians[f"kge_median.{n}.{lk(k)}"] for n in fitted)]
onset_cells = lambda n, at=False: [V[f"{n}.onset.{lk(k)}"][AT_KEY] if at else V[f"{n}.onset.{lk(k)}"] for k in F.LEADS]
plain_txt, chosen_txt = firm(onset_cells("lstm_wq")), firm(onset_cells("lstm_wq", True))
differ = [f"{LABEL[n]}, {'all high-flow days' if kind == 'all' else 'onsets'}, {ahead(k)}: {V[f'{n}.{kind}.{lk(k)}']['verdict']} at the high-flow level itself, "
          f"{V[f'{n}.{kind}.{lk(k)}'][AT_KEY]['verdict']} at {V[f'{n}.{kind}.{lk(k)}'][AT_KEY]['warning_level']:.2f} times that level"
          for n in TABLE for kind in RATE for k in F.LEADS if V[f"{n}.{kind}.{lk(k)}"]["verdict"] != V[f"{n}.{kind}.{lk(k)}"][AT_KEY]["verdict"]]
top = lambda kind: [V[f"lstm_wqf.{kind}.{lk(k)}"][AT_KEY] for k in F.LEADS]                                                   # the verdict table's cells of the upper bound
held = lambda cells: " / ".join(share(c["share_of_resamples_with_each_verdict"][c["verdict"]]) for c in cells) + " (" + " / ".join(c["verdict"] for c in cells) + ")"
caut, pers = V[f"lstm_wq.all.{k1}"], V[f"base_persistence.all.{k1}"]
fit_gap = [medians[f"spread_ratio_median.{n}.{lk(k)}"] - medians[f"correlation_median.{n}.{lk(k)}"] for n in fitted for k in F.LEADS]      # median spread ratio minus median correlation
basin_gap = max(abs(spread_gap[n][lk(k)][e]) for n in fitted for k in F.LEADS for e in ("min", "max"))                          # the largest such difference in a single basin
show = [f"nse_median.{n}.{k1}" for n in ("lstm_w", "lstm_wq", "lstm_wqn", "lstm_wqf", "best_simple_wq")] + [f"{tie}.{lk(k)}" for k in F.LEADS] \
    + [f"onsets_warned.{n}.{k1}" for n in ("lstm_w", "lstm_wq", "lstm_wqn", "lstm_wqn_older", "lstm_wqf", "best_simple_wq")] \
    + [f"onsets_gain.evening_rain_minus_control.wq.{k1}", f"onsets_gap_share_closed.wq.{k1}", f"onsets_gain.upper_bound_minus_base.wq.{k1}", f"nse_median_of_diffs.wqf_minus_wq.{k1}",
       f"nse_median_of_diffs.held_out_minus_in_training.{k1}", f"nse_diff_of_medians.held_out_minus_in_training.{k1}", f"nse_median_of_diffs.transformer_minus_lstm.{k1}"]      # the headline statistics: printed at the end
mc_headline = max([x for key in show if key in MC for x in MC[key] if x == x], default=float("nan"))                            # the largest noise of an end among them
# ---- the added sets: the headline statistics to print, helpers, and notes (every number in them is read from the tables above) ------------------------
show_new = [f"nse_median.{n}.{k1}" for n in ("lstm_wqa", "lstm63_wq", "unseen50_w_static")] + [f"onsets_warned.{n}.{k1}" for n in ("lstm_wqh", "lstm_wqa", "lstm63_wq")] \
    + [f"onsets_warned.lstm_wqaF_sub.{lk(k)}" for k in F.LEADS] \
    + [f"nse_diff_of_medians.unseen50_w_static_minus_lobo_w_static.{k1}", f"nse_median_of_diffs.unseen50_w_static_minus_lobo_w_static.{k1}"] \
    + [f"{stem}.wqa_minus_wq.{k1}" for stem in ("nse_diff_of_medians", "nse_median_of_diffs", "onsets_diff")] \
    + [f"{stem}.wqaF_sub_minus_wqa.{lk(k)}" for stem in ("nse_diff_of_medians", "nse_median_of_diffs", "onsets_diff") for k in F.LEADS]
pp = lambda x: (f"{100 * x:+.1f}" if round(100 * x, 1) else "0.0") if ok(x) else "no value"             # a difference of two shares in percentage points, one decimal, with its sign
ons = lambda n, k=1: vr(f"onsets_warned.{n}.{lk(k)}", pc)                    # "value (lo to hi)" of a share of onsets warned
odf = lambda name, k=1: vr(f"onsets_diff.{name}.{lk(k)}", pp)                # ... of a difference of two such shares, in percentage points
nsm = lambda n, k=1: vr(f"nse_median.{n}.{lk(k)}", n3)                       # ... of a median NSE
bl_on = lambda n: " / ".join(ons(n, k) for k in F.LEADS)                    # the same by lead
bl_odf = lambda name: " / ".join(odf(name, k) for k in F.LEADS)
bl_nsm = lambda n: " / ".join(nsm(n, k) for k in F.LEADS)
nmd = lambda name, k=1: vr(f"nse_median_of_diffs.{name}.{lk(k)}", sg)        # ... of the median over the basins of the NSE differences
ndm = lambda name, k=1: vr(f"nse_diff_of_medians.{name}.{lk(k)}", sg)        # ... of the difference of the two median NSE
clo = lambda name, k=1, beyond=False: vr(f"onsets_gap_share_closed{'_beyond_control' if beyond else ''}.{name}.{lk(k)}", pc)      # ... of the share of the onset gap closed
zero = lambda key: "the range holds zero: no clear difference" if holds_zero(key) else "the range does not hold zero"
kd = lambda name, k=1: f"onsets_diff.{name}.{lk(k)}"
kn = lambda name, k=1: f"nse_median_of_diffs.{name}.{lk(k)}"
split_sat = FI["split_satellite_era"]
sat_sets = [v["saved_as"] for v in FI["sets"].values() if v["arm"] != "main"]                              # the sets of E2b, on the satellite-era split
sat_years = round((pd.Timestamp(split_sat["test"][1]) - pd.Timestamp(split_sat["test"][0])).days / 365.25)
verdict_sets = ("lstm_wqh", "lstm_wqa", "lstm63_wq", "lstm_wqaF_sub")
cells_of = lambda n, kind: [V[f"{n}.{kind}.{lk(k)}"] for k in F.LEADS]
SEED_COUNT = {n: sum(F.have_preds(f"{n}_s{i}") for i in range(10)) for n in SETS}        # 2026-10-30: seeds 3 and 4 were added to some sets; the wording names the counts as they are
_by_count = {c: [n for n in SETS if SEED_COUNT[n] == c] for c in sorted({c for c in SEED_COUNT.values() if c})}
_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
SEED_TXT = ("the single trainings (" + "; ".join(f"_s0 to _s{c - 1}, {_word.get(c, c)} per set, for {', '.join(ns)}" for c, ns in _by_count.items()) + ")" if len(_by_count) > 1
            else "the single trainings (" + "".join(f"_s0 to _s{c - 1}, {_word.get(c, c)} per set" for c in _by_count) + ")")
unstored_txt = (f"The {len(PAIRS_UNSTORED)} differences without a stored counterpart ({', '.join(PAIRS_UNSTORED)}) are plain arithmetic of checked statistics." if PAIRS_UNSTORED
                else f"Every one of the {len(PAIRS_NEW)} differences has a stored counterpart to be compared with.")
new_notes = [
    f"Extension of 2026-10-08: {len(SETS_NEW)} prediction sets and {len(PAIRS_NEW)} differences were added to the {len(SETS_OLD)} sets and {len(PAIRS_OLD)} differences of 4 October, so that a range stands beside "
    f"the headline numbers of the four experiments of 7-8 October. Nothing of 4 October changed: its {len(OLD_KEYS)} statistics, its verdict cells and its spread-and-skill tables come from the same code, the "
    f"same forecasts and the same random numbers. Added, by script: 15 (the rain of the issue day up to local midnight, hourly ERA5) {', '.join(list(E1_SETS) + list(E1_SIMPLE))}; 16 (hourly AORC rain) "
    f"{', '.join(list(E1B_SETS) + list(E1B_SIMPLE))}; 17 (63 training rivers, and 50 for rivers never seen; scored on the 12) {', '.join(E3_SETS)}, together with {' and '.join(LOBO_NEW)} of 05_lobo.py, the "
    f"no-properties partners that its never-seen comparison needs and that this script did not hold; 18 (the GEFSv12 rain forecast, main split only) {', '.join(list(E2_SETS) + list(E2_SIMPLE))}. For every added set: "
    f"the median NSE per lead and the six warning rates per lead at the high-flow level itself, with a verdict cell (no warning level chosen on the validation years is computed for them: "
    f"flood_analysis.json holds none). For the differences (stat_names.pairs): the NSE as the difference of the two medians and as the median of the basin differences, and onsets_diff in the share of "
    f"onsets warned; for the closures (stat_names.closures) the share of the gap in onsets warned, to an upper bound that cannot be had in practice, that a set closes (a ratio, wide where the gap is small; a "
    f"resample in which the gap is exactly 0 has no value and is left out of the interval, as for the false-warning shares: resampling.statistics_without_a_value_in_some_resamples). "
    f"The NSE share of the gap is not resampled: its denominator can be near zero (lstm_wqaP minus lstm_wqa {d1}: {ndm('wqaP_minus_wqa')}). Every difference is taken on the same basin-days. "
    f"Checks before the resampling: the yearly sums give the scores of flood.evaluate for the added sets, which are the numbers stored in {', '.join(STORED_NEW)} (and lstm_lobo.json for the two lobo sets); "
    f"the statistics of {len(CHECK_PAIRS)} differences and {len(CHECK_CLOSE)} closures on the test years as they are equal the comparison numbers those files store (gain_rule, closure, yardstick, unseen, "
    f"gap_to_the_upper_bound); {n_labels} verdict labels equal the ones stored; {sum(by_file.values()):,} numbers compared for the added sets and comparisons ({', '.join(f'{v:,} for {k}' for k, v in by_file.items())}). "
    f"{unstored_txt}",
    f"Left out: the sets of the retraining with the forecast (E2b; {', '.join(sat_sets)}, whose names end in _s in results/forecast_input.json) lie on another split (test years {split_sat['test'][0]} to "
    f"{split_sat['test'][1]}: {sat_years} water years, not the {ny} drawn here), so they share no test days with the sets above and one set of resamples cannot draw them together. Their numbers are comparable "
    "only inside that split, and the intervals of results/forecast_input.json (uncertainty.block_bootstrap: blocks of consecutive test days drawn again) are the ones to quote for them. Also not added: " + SEED_TXT + ", "
    "the derived variants of script 18 (lstm_wqaF_sub_raw, _rawpred, _perfectpred, lstm_wqaP_forecastpred), spread, skill against 'tomorrow = today' and KGE of the added sets, and "
    "a best simple model for the new inputs (ridge regression and the random forest are given separately).",
    f"E1, the rain of the issue day up to local midnight (hourly ERA5, script 15), with the river gauge, {d1}: onsets warned {ons('lstm_wqh')} against {ons('lstm_wq')} with the rain table alone and "
    f"{ons('lstm_wqh_older')} with the control (the same rain one day older). The gain is {odf('wqh_minus_wq')} percentage points over the rain table alone and {odf('wqh_minus_wqh_older')} over the control. "
    f"The hours from 09:00 to 24:00 are worth {odf('wqh_minus_wqh_to09')} points (the control over the rain table's own window, no new hours, warns of {ons('lstm_wqh_to09')}) and the evening hours beyond the "
    f"UTC day {odf('wqh_minus_wqh_utc')} points (over the UTC day: {ons('lstm_wqh_utc')}). The NLDAS rain up to the evening warns of {ons('lstm_wqn')}; ERA5 minus NLDAS is {odf('wqh_minus_wqn')} points "
    f"({zero(kd('wqh_minus_wqn'))}) and the two together warn of {ons('lstm_wqnh')}. Of the gap to the perfect weather forecast (an upper bound that cannot be had in practice) the rain to midnight closes "
    f"{clo('wqh__base_wq__ceiling_wqf')}, {clo('wqh__base_wq__ceiling_wqf', beyond=True)} beyond the control. Median NSE {nsm('lstm_wq')} -> {nsm('lstm_wqh')} (control {nsm('lstm_wqh_older')}); in the typical "
    f"basin {nmd('wqh_minus_wq')} (against the control {nmd('wqh_minus_wqh_older')}). Two and three days ahead the gain in onsets over the rain table alone is {odf('wqh_minus_wq', 2)} and "
    f"{odf('wqh_minus_wq', 3)} points. Without the river gauge: {ons('lstm_wh')} against {ons('lstm_w')} (control {ons('lstm_wh_older')}; gain {odf('wh_minus_w')} points). Ridge regression and the random "
    f"forest given the same input warn of {ons('base_linreg_wqh')} and {ons('base_rf_wqh')}; the input adds {odf('linreg_wqh_minus_linreg_wq')} and {odf('rf_wqh_minus_rf_wq')} points to them and "
    f"{odf('wqh_minus_wq')} to the LSTM; the LSTM minus ridge regression is {odf('lstm_wqh_minus_linreg_wqh')} points in onsets and {nmd('lstm_wqh_minus_linreg_wqh')} NSE in the typical basin, minus the random "
    f"forest {odf('lstm_wqh_minus_rf_wqh')} points and {nmd('lstm_wqh_minus_rf_wqh')} NSE.",
    f"E1b, the same with AORC rain (1 km, gauge- and radar-anchored; script 16), with the river gauge, {d1}: onsets warned {ons('lstm_wqa')} against {ons('lstm_wq')} with the rain table alone. The gain is "
    f"{odf('wqa_minus_wq')} percentage points over the rain table alone and {odf('wqa_minus_wqa_to09')} over the control with the same AORC rain over the rain table's own window (which warns of "
    f"{ons('lstm_wqa_to09')}); over the UTC day the rain warns of {ons('lstm_wqa_utc')}, so the hours after it are worth {odf('wqa_minus_wqa_utc')} points. AORC minus ERA5 to midnight is "
    f"{odf('wqa_minus_wqh')} points, minus the NLDAS rain {odf('wqa_minus_wqn')}, minus the two older products together {odf('wqa_minus_wqnh')} ({zero(kd('wqa_minus_wqnh'))}). The AORC rain in the place of the "
    f"rain column warns of {ons('lstm_wqa_rep')}: {odf('wqa_rep_minus_wqa')} points against AORC as an extra column ({zero(kd('wqa_rep_minus_wqa'))}). It closes {clo('wqa__base_wq__ceiling_wqf')} of the gap to the "
    f"perfect weather forecast, an upper bound that cannot be had in practice ({clo('wqa__base_wq__ceiling_wqf', beyond=True)} beyond the control). Median NSE {nsm('lstm_wq')} -> {nsm('lstm_wqa')}: "
    f"{nmd('wqa_minus_wq')} in the typical basin, {ndm('wqa_minus_wq')} as a difference of the two medians. Two and three days ahead the gain in onsets is {odf('wqa_minus_wq', 2)} and "
    f"{odf('wqa_minus_wq', 3)} points, in NSE {nmd('wqa_minus_wq', 2)} and {nmd('wqa_minus_wq', 3)}. Ridge regression and the random forest given the same input warn of {ons('base_linreg_wqa')} and "
    f"{ons('base_rf_wqa')} (the input adds {odf('linreg_wqa_minus_linreg_wq')} and {odf('rf_wqa_minus_rf_wq')} points to them and {odf('wqa_minus_wq')} to the LSTM); the LSTM minus ridge regression is "
    f"{odf('lstm_wqa_minus_linreg_wqa')} points in onsets and {nmd('lstm_wqa_minus_linreg_wqa')} NSE, minus the random forest {odf('lstm_wqa_minus_rf_wqa')} points and {nmd('lstm_wqa_minus_rf_wqa')} NSE.",
    f"E3, 63 training rivers instead of 12 (script 17; the same 12 rivers scored), with the river gauge, {d1}: median NSE {nsm('lstm_wq')} -> {nsm('lstm63_wq')}; in the typical basin {nmd('lstm63_wq_minus_wq')} "
    f"({zero(kn('lstm63_wq_minus_wq'))}), as a difference of the two medians {ndm('lstm63_wq_minus_wq')}; onsets warned {ons('lstm_wq')} -> {ons('lstm63_wq')} ({odf('lstm63_wq_minus_wq')} points; "
    f"{zero(kd('lstm63_wq_minus_wq'))}). With five times the update limit and patience: NSE {nsm('lstm63eq_wq')} ({nmd('lstm63eq_wq_minus_wq')} against the 12 rivers), onsets {ons('lstm63eq_wq')}. Without the "
    f"river gauge: {nsm('lstm_w')} -> {nsm('lstm63_w')} ({nmd('lstm63_w_minus_w')} in the typical basin). At {leads_txt} the typical basin gains {by_lead('nse_median_of_diffs.lstm63_wq_minus_wq', sg)} with the "
    f"gauge. Rivers never seen (weather only), {d1}: median NSE with basin properties {nsm('lobo_w_static')} (11 training rivers) -> {nsm('unseen50_w_static')} (50); {ndm('unseen50_w_static_minus_lobo_w_static')} "
    f"as a difference of the two medians, {nmd('unseen50_w_static_minus_lobo_w_static')} in the typical basin. Without properties {nsm('lobo_w_nostatic')} -> {nsm('unseen50_w_nostatic')}: "
    f"{ndm('unseen50_w_nostatic_minus_lobo_w_nostatic')} and {nmd('unseen50_w_nostatic_minus_lobo_w_nostatic')}. The typical river loses {vr(kn('unseen50_w_static_minus_lstm_w'), n3, flip=True)} NSE from not "
    f"having been seen when 50 rivers are in training (against the three-seed 12-river model of the same inputs) and {vr(kn('held_out_minus_in_training'), n3, flip=True)} with 11 (one training, "
    f"held_out_minus_in_training): the two designs differ in the number, the variety and the nearness of the training rivers, which these ranges do not separate (results/more_basins.json). The warnings of the "
    f"never-seen sets are counted at the observed high-flow level, which a river without a gauge does not have.",
    f"E2a, the GEFS forecast of the rain of the next three days (script 18, main split), with the river gauge and the rain to midnight, onsets warned at {leads_txt}: {bl_on('lstm_wqaF_sub')} with the forecast "
    f"substituted into the network, against {bl_on('lstm_wqa')} without it, {bl_on('lstm_wqaF_clim')} with the forecast shuffled within month (not a neutral control: it measures the harm of misleading inputs to a "
    f"network trained on exact ones, not the information in the forecast) and {bl_on('lstm_wqaP')} given the perfect rain of the three days (an upper bound that cannot be had in practice). The forecast adds "
    f"{bl_odf('wqaF_sub_minus_wqa')} percentage points to lstm_wqa; median NSE {bl_nsm('lstm_wqaF_sub')} against {bl_nsm('lstm_wqa')}; in the typical basin the forecast changes the NSE by "
    f"{by_lead('nse_median_of_diffs.wqaF_sub_minus_wqa', sg)} (as a difference of the two medians {by_lead('nse_diff_of_medians.wqaF_sub_minus_wqa', sg)}). Against the rain table alone (lstm_wq) it adds "
    f"{bl_odf('wqaF_sub_minus_wq')} points. Of the gap in onsets to the perfect rain it closes {by_lead('onsets_gap_share_closed.wqaF_sub__base_wqa__ceiling_wqaP', pc)} (a ratio: the gap, lstm_wqaP minus lstm_wqa, is {bl_odf('wqaP_minus_wqa')} points, "
    f"small at 1 day ahead, so the range there says little). Without the rain to midnight (lstm_wqF_sub) the "
    f"forecast warns of {bl_on('lstm_wqF_sub')}, {bl_odf('wqF_sub_minus_wq')} points over the rain table alone; the rain to midnight is worth {bl_odf('wqaF_sub_minus_wqF_sub')} points beside the forecast. Ridge "
    f"regression and the random forest given the same inputs warn of {bl_on('base_linreg_wqaF')} and {bl_on('base_rf_wqaF')}; the forecast adds {bl_odf('linreg_wqaF_minus_linreg_wqa')} and "
    f"{bl_odf('rf_wqaF_minus_rf_wqa')} points to them; the LSTM minus ridge regression is {bl_odf('lstm_wqaF_sub_minus_linreg_wqaF')} points and minus the random forest {bl_odf('lstm_wqaF_sub_minus_rf_wqaF')}.",
    "Verdicts of the added sets when years and basins are drawn again (the criteria of results/flood_analysis.json at the high-flow level itself; verdict at the test years as they are, then the share of the "
    "resamples in which it comes out, at 1 / 2 / 3 days ahead): " + "; ".join(
        f"{n}: onsets {held(cells_of(n, 'onset'))}, all high-flow days {held(cells_of(n, 'all'))}" for n in verdict_sets) + f". Stored for every added set with warnings ({len(WARN_NEW)} sets) in verdict_stability.",
    f"Noise of the added intervals from the random numbers: an end moves by about {num(mc_typical_new)} for the typical added statistic and by up to {num(mc_largest_new)} (for {mc_largest_for_new}); for "
    f"{len(near_zero_new)} of the {len(range_new)} added statistics an end lies closer to zero than {NEAR} times its noise, so whether such a range holds zero can change with the random numbers "
    "(resampling.monte_carlo_noise.statistics_added_2026_10_08 and ends_near_zero).",
]
ADDED = dict(
    what="The prediction sets of the four experiments of 7-8 October 2026 (scripts 15 to 18), added on 2026-10-08 so that a range stands beside every new headline number. Everything of 4 October is unchanged: "
         "its statistics, verdict cells and spread-and-skill tables are computed by the same code from the same forecasts and the same random numbers.",
    sets_by_script={"15_rain_to_midnight.py": list(E1_SETS) + list(E1_SIMPLE), "16_aorc_rain_to_midnight.py": list(E1B_SETS) + list(E1B_SIMPLE),
                    "17_more_basins.py (scored on the 12 rivers)": list(E3_SETS), "05_lobo.py (the no-properties partners of the never-seen sets, not held before)": list(LOBO_NEW),
                    "18_forecast_input.py (main split only)": list(E2_SETS) + list(E2_SIMPLE)},
    n_sets_added=len(SETS_NEW), n_statistics_added=len(point) - len(OLD_KEYS), n_differences_added=len(PAIRS_NEW), n_closures_added=len(CLOSE),
    sets_with_warning_statistics_and_verdict_cells=list(WARN_NEW), of_these_not_new=[n for n in WARN_NEW if n not in SETS_NEW],
    of_these_not_new_why="lobo_w_static is a set of 4 October; its warnings (at the observed high-flow level, which a river without a gauge does not have) are counted from 2026-10-08 so that the never-seen sets of script 17 can be set beside it",
    closures={name: dict(set=w, base=base, upper_bound=ceil, control=ctrl) for name, (w, base, ceil, ctrl) in CLOSE.items()},
    stored_comparisons=dict(differences_checked=len(CHECK_PAIRS), closures_checked=len(CHECK_CLOSE), differences_without_a_stored_counterpart=PAIRS_UNSTORED,
                            verdict_labels_compared=n_labels, numbers_compared_by_file=by_file),
    left_out=dict(
        e2b=dict(sets=sat_sets, reason=f"retrained on the satellite-era split (test years {split_sat['test'][0]} to {split_sat['test'][1]}, {sat_years} water years): no test day in common with the sets "
                                       "resampled here; comparable only inside that split; its intervals are the block bootstrap of results/forecast_input.json (uncertainty.block_bootstrap)"),
        other=[SEED_TXT, "the derived variants of script 18 (lstm_wqaF_sub_raw, _rawpred, _perfectpred, lstm_wqaP_forecastpred)",
               "spread, skill against 'tomorrow = today' and KGE of the added sets", "warning levels chosen on the validation years for the added sets",
               "the NSE share of the gap to an upper bound (its denominator can be near zero)", "a best simple model for the new inputs (ridge regression and the random forest are given separately)"]))
notes = [
    f"No model was trained here. Every number is recomputed from the forecasts stored in results/preds, on the test years. A warning is counted at the high-flow level itself; only the "
    f"names ending in {AT}, and {AT_KEY} in a verdict cell, use the warning level chosen on the validation years. Before anything is resampled the script asserts that its sums per "
    f"basin and year give the scores of flood.evaluate for all {len(SETS)} prediction sets, that these are the numbers stored in {', '.join(STORED + STORED_NEW)}, and that the best simple model named "
    f"there is the one with the highest median NSE in the validation years ({n_before:,} numbers compared; {n_compared:,} with the checks made after the resampling). All sets have a "
    f"forecast for every basin, test day and lead, so every difference is taken on the same basin-days.",
    f"Resampling: the {ny} test water years ({YEARS[0]} to {YEARS[-1]}) and the {nb} basins are each drawn again with replacement, {B:,} times, from one fixed seed. A whole water year stays "
    f"together, so the days of one storm - and the same storm in several basins - stay together. Every statistic is recomputed exactly on each resample from the sums per basin and year; "
    f"the first {min(N_DIRECT, B)} resamples were also scored directly with flood.nse and flood.contingency and gave the same numbers. An interval is the 2.5th to 97.5th percentile of the "
    f"{B:,} results: the range in which the number stays when the test years and the basins are drawn again. It is not a proof.",
    f"Three designs are stored for every number: only the years drawn again, only the basins, and both at once. Both at once is the one to quote. The other two show where the range comes "
    f"from: for the median NSE of the LSTM with the river gauge {d1} it is {n3(wide['years']['lo'])} to {n3(wide['years']['hi'])} when only the years are drawn again and "
    f"{n3(wide['basins']['lo'])} to {n3(wide['basins']['hi'])} when only the basins are"
    + (f": which {nb} basins are in the set matters more than which {ny} years." if ok(wide["years"]["lo"]) and wide["basins"]["hi"] - wide["basins"]["lo"] > wide["years"]["hi"] - wide["years"]["lo"] else "."),
    f"What an interval does not cover: how the {nb} basins, the region and the test period were chosen (rivers of another kind, another region or years unlike these {ny} can give numbers "
    f"outside it), and the difference between two trainings of the same network (lstm_<set> is the average forecast of its seeds and is resampled as one forecast; the seed ranges are in "
    f"results/lstm_all.json and results/rain_timing.json). With {nb} basins that share storms and {ny} year-blocks the intervals are wide, and an interval from so few units is itself rough. An end of an interval "
    f"also moves a little with the random numbers, because {B:,} is a finite number of resamples: by about {num(mc_typical)} for the typical statistic and by up to {num(mc_headline)} "
    f"for the headline statistics (a standard error, judged from {MC_PARTS} parts of the resamples; resampling.monte_carlo_noise), so the last digit printed for an end is not firm. "
    f"For {len(near_zero_old)} of the {len(range_old)} statistics an end lies closer to zero than {NEAR} times its noise (resampling.monte_carlo_noise.ends_near_zero): whether such a range holds "
    "zero can change with the random numbers, so no sentence should rest on it.",
    f"LSTM against the best simple model with the river gauge ({best_txt('wq')}; chosen on the validation years and kept in every resample): the LSTM's NSE minus that of the simple model, "
    f"in the typical basin (median of the basin differences), is {by_lead(tie, sg)} at {leads_txt}. "
    + (f"Every range holds zero and none reaches further than {reach(tie)} NSE from it: with the river gauge the LSTM and the simple models show no clear difference, among the models tried."
       if all_hold_zero(tie) else "Not every range holds zero.")
    + f" As a difference of the two medians: {by_lead('nse_diff_of_medians.lstm_wq_minus_best_simple', sg)}.",
    f"The same comparison in onsets warned {d1}, LSTM with the river gauge minus simple model, in percentage points: random forest {vr(rf_k, pts)}, ridge regression {vr(lin_k, pts)}. "
    + ("Both ranges hold zero." if holds_zero(rf_k) and holds_zero(lin_k) else "Neither range holds zero." if not holds_zero(rf_k) and not holds_zero(lin_k) else
       f"The range against the {'random forest' if holds_zero(rf_k) else 'ridge regression'} holds zero; the one against the {'ridge regression' if holds_zero(rf_k) else 'random forest'} does not.")
    + f" (The best simple model, chosen by its NSE in the validation years, is the {simple(BEST['wq'][k1])} {d1}.)",
    f"LSTM against the best simple model with past weather only ({best_txt('w')}): {by_lead('nse_median_of_diffs.lstm_w_minus_best_simple', sg)} NSE in the typical basin at {leads_txt}"
    + ("; every range holds zero." if all_hold_zero("nse_median_of_diffs.lstm_w_minus_best_simple") else "; not every range holds zero."),
    f"Onsets warned {d1}, the {nb} basins counted together ({int(n_on[:, :, 0].sum()):,} onsets in the test years): {vr(f'onsets_warned.lstm_w.{k1}', pc)} with past weather, "
    f"{vr(f'onsets_warned.lstm_wq.{k1}', pc)} with the river gauge added, {vr(f'onsets_warned.lstm_wqn.{k1}', pc)} with rain up to the evening of the issue day added to that, "
    f"{vr(f'onsets_warned.lstm_wqn_older.{k1}', pc)} with its control (the same rain one day older), and {vr(f'onsets_warned.lstm_wqf.{k1}', pc)} with the river gauge and the perfect "
    f"weather forecast, an upper bound that cannot be had in practice. Of the simple models with the river gauge, the random forest warns of {vr(f'onsets_warned.base_rf_wq.{k1}', pc)} "
    f"and the ridge regression of {vr(f'onsets_warned.base_linreg_wq.{k1}', pc)}.",
    f"Rain up to the evening of the issue day, with the river gauge, {d1}: {pt(U[f'onsets_gain.evening_rain_minus_base.wq.{k1}']['value'])} percentage points more onsets warned than "
    f"without it ({rng_of(f'onsets_gain.evening_rain_minus_base.wq.{k1}', pt)}; above zero in {count(U[f'onsets_gain.evening_rain_minus_base.wq.{k1}']['both']['share_above_zero'])}) and "
    f"{pt(U[f'onsets_gain.evening_rain_minus_control.wq.{k1}']['value'])} points more than the control ({rng_of(f'onsets_gain.evening_rain_minus_control.wq.{k1}', pt)}; above zero in "
    f"{count(U[f'onsets_gain.evening_rain_minus_control.wq.{k1}']['both']['share_above_zero'])}). That is {vr(f'onsets_gap_share_closed.wq.{k1}', pc)} of the gap to the perfect weather forecast "
    f"(an upper bound that cannot be had in practice), {vr(f'onsets_gap_share_closed_beyond_control.wq.{k1}', pc)} beyond the control. "
    + " ".join(f"{ahead(k).capitalize()} the gain is {pt(U[f'onsets_gain.evening_rain_minus_base.wq.{lk(k)}']['value'])} points ({rng_of(f'onsets_gain.evening_rain_minus_base.wq.{lk(k)}', pt)})"
               + (", a range with one end at about zero." if f"onsets_gain.evening_rain_minus_base.wq.{lk(k)}" in near_zero else
                  ", a range that holds zero." if holds_zero(f"onsets_gain.evening_rain_minus_base.wq.{lk(k)}") else ".") for k in F.LEADS[1:])
    + f" In NSE the typical basin changes by {by_lead('nse_median_of_diffs.wqn_minus_wq', sg)} at {leads_txt} (against the control: {by_lead('nse_median_of_diffs.wqn_minus_wqn_older', sg)}).",
    f"The perfect weather forecast, an upper bound that cannot be had in practice, with the river gauge: {by_lead('onsets_gain.upper_bound_minus_base.wq', pt)} percentage points more onsets "
    f"warned at {leads_txt}, and {by_lead('nse_median_of_diffs.wqf_minus_wq', sg)} NSE in the typical basin ({by_lead('nse_diff_of_medians.wqf_minus_wq', sg)} as a difference of the two "
    f"medians). {d1.capitalize()} the gain in NSE runs from {sg(gain1.min())} to {sg(gain1.max())} over the {nb} basins and is below 0.100 in {int((gain1 < 0.1).sum())} of them: when many of "
    f"those are drawn the median is small, and the range reaches down to {sg(U[f'nse_median_of_diffs.wqf_minus_wq.{k1}']['both']['lo'])}. The size of what the weather of the coming days "
    "adds is therefore better read from the onsets and from the longer leads.",
    f"A river the model has never seen: the typical basin loses {by_lead('nse_median_of_diffs.held_out_minus_in_training', n3, flip=True)} NSE at {leads_txt} (median of the basin "
    f"differences, in training minus held out). The difference of the two medians {d1}, {n3(U[f'nse_median.lobo_w_static.{k1}']['value'])} held out against "
    f"{n3(U[f'nse_median.lstm_w_s0.{k1}']['value'])} in training, is {vr(f'nse_diff_of_medians.held_out_minus_in_training.{k1}', sg)}: the two medians come from different basins"
    + (", and this range holds zero while the range of the paired loss does not, so the paired loss is the number to quote first."
       if holds_zero(f"nse_diff_of_medians.held_out_minus_in_training.{k1}") and not holds_zero(f"nse_median_of_diffs.held_out_minus_in_training.{k1}") else "."),
    f"Transformer minus its LSTM twin (same inputs and training budget, results/model_comparison.json): {by_lead('nse_median_of_diffs.transformer_minus_lstm', sg)} NSE in the typical basin "
    f"at {leads_txt}; as a difference of the two medians {by_lead('nse_diff_of_medians.transformer_minus_lstm', sg)}"
    + ("; every range holds zero: no clear difference in accuracy, among the models tried." if all_hold_zero("nse_median_of_diffs.transformer_minus_lstm", "nse_diff_of_medians.transformer_minus_lstm")
       else "; not every range holds zero."),
    f"Verdicts when years and basins are drawn again (criteria of results/flood_analysis.json: GO when at least {pc(GO_POD)} are warned and at most {pc(GO_FAR)} of the warnings are false; "
    f"CAUTION when not GO and CSI is at least {CAUTION_CSI:.2f}; otherwise NOT YET). For the LSTM with the river gauge the onset verdict is NOT YET in "
    + (f"{plain_txt}, at the high-flow level itself and at the warning level chosen on the validation years alike" if plain_txt == chosen_txt else
       f"{plain_txt} at the high-flow level itself and in {chosen_txt} at the warning level chosen on the validation years")
    + f"; with rain up to the evening of the issue day in {firm(onset_cells('lstm_wqn'))}. For all high-flow days {d1} its verdict {caut['verdict']} holds in "
    f"{share(vs(f'lstm_wq.all.{k1}')[caut['verdict']])} of the resamples (" + ", ".join(f"{x} in {share(vs(f'lstm_wq.all.{k1}')[x])}" for x in VERDICTS if x != caut["verdict"])
    + f"); 'tomorrow = today' gets {pers['verdict']} there in {share(pers['share_of_resamples_with_each_verdict'][pers['verdict']])}"
    + (f", and at its own chosen warning level ({pers[AT_KEY]['warning_level']:.2f} times the high-flow level) {pers[AT_KEY]['verdict']} in "
       f"{share(pers[AT_KEY]['share_of_resamples_with_each_verdict'][pers[AT_KEY]['verdict']])}." if AT_KEY in pers else "."),
    f"Which warning level: verdict_stability counts a warning at the high-flow level itself. results/flood_analysis.json takes its verdicts at a warning level chosen on the validation years "
    f"for each model and lead. For the {len(CHOSEN)} sets that have such a level here ({', '.join(CHOSEN)}) the verdict and the shares at that level (kept fixed in every resample) are stored "
    f"in the same cell under {AT_KEY}, and the six rates under the stat names ending in {AT}. For the {len(TABLE)} sets of its verdict table ({', '.join(TABLE)}) the two levels give the same "
    f"verdict in {sum(same_verdict)} of the {len(same_verdict)} cells" + (f" (not for: {'; '.join(differ)})." if differ else ".")
    + f" With the perfect weather forecast (an upper bound that cannot be had in practice) the verdicts of that table hold, at {leads_txt}, in {held(top('all'))} of the resamples for all "
    f"high-flow days and in {held(top('onset'))} for onsets.",
    f"Why the peaks are too low (spread_and_skill; medians of the {nb} basins at {leads_txt}): the forecasts of the LSTM with the river gauge have {med_txt('spread_ratio', 'lstm_wq')} of the "
    f"spread of the observed flow, and their correlation with it is {med_txt('correlation', 'lstm_wq')}; the random forest {med_txt('spread_ratio', 'base_rf_wq')} against "
    f"{med_txt('correlation', 'base_rf_wq')}, the ridge regression {med_txt('spread_ratio', 'base_linreg_wq')} against {med_txt('correlation', 'base_linreg_wq')}. A forecast that minimises "
    f"the squared error has a spread of about its correlation times the observed spread: when it cannot know whether the rain will come, it hedges. "
    + (f"In the medians of these three models the spread ratio is {num(min(fit_gap))} to {num(max(fit_gap))} above the correlation: about the spread that the correlation predicts, not "
       f"exactly it." if min(fit_gap) >= 0 else
       f"In the medians of these three models the spread ratio minus the correlation runs from {sg(min(fit_gap))} to {sg(max(fit_gap))}.")
    + f" In a single basin the two differ by up to {num(basin_gap)}. Given the perfect weather forecast (an "
    f"upper bound that cannot be had in practice) the LSTM's spread ratio is {med_txt('spread_ratio', 'lstm_wqf')} and its correlation {med_txt('correlation', 'lstm_wqf')}. The mean "
    f"forecast of the LSTM with the river gauge is {med_txt('total_ratio', 'lstm_wq')} times the mean observed flow. Basin by basin, the spread ratio of the LSTM with the river gauge "
    f"minus its correlation lies between {sg(spread_gap['lstm_wq'][k1]['min'])} and {sg(spread_gap['lstm_wq'][k1]['max'])} {d1} (median {sg(spread_gap['lstm_wq'][k1]['median'])}). When "
    f"years and basins are drawn again, its median spread ratio {d1} stays in {rng_of(f'spread_ratio_median.lstm_wq.{k1}', n3)} and its median correlation in "
    f"{rng_of(f'correlation_median.lstm_wq.{k1}', n3)}.",
    f"Skill against 'tomorrow = today' (1 - squared error of the model / squared error of 'tomorrow = today', median of the {nb} basins, with the river gauge) at {leads_txt}: LSTM "
    f"{by_lead('skill_vs_persistence_median.lstm_wq', n3)}, random forest {by_lead('skill_vs_persistence_median.base_rf_wq', n3)}, ridge regression "
    f"{by_lead('skill_vs_persistence_median.base_linreg_wq', n3)}"
    + ("; each value lies inside the ranges of the other two: no clear difference." if one_inside_the_other("skill_vs_persistence_median", fitted) else "."),
    f"KGE (median of the {nb} basins, {leads_txt}): LSTM with the river gauge {med_txt('kge', 'lstm_wq')}, random forest {med_txt('kge', 'base_rf_wq')}, ridge regression "
    f"{med_txt('kge', 'base_linreg_wq')}, 'tomorrow = today' {med_txt('kge', 'base_persistence')}."
    + (f" At {' and '.join(str(k) for k in kge_copy_higher)} day{'s' * (kge_copy_higher[-1] > 1)} ahead 'tomorrow = today' has a higher KGE than each of the three fitted models: a copy of "
       "the river has the right spread and total by construction, so KGE is not read as forecast skill here." if kge_copy_higher else ""),
    "'Tomorrow = today' can never warn of an onset at the high-flow level itself (the river was still low when the forecast was issued), so the share of false warnings among its onset "
    "warnings has no value: NaN, with null for the interval. In a few resamples another set gives no onset warning at all; those resamples are left out of the interval of its false-warning "
    "share (resampling.statistics_without_a_value_in_some_resamples counts them); in a verdict cell such a resample counts as NOT YET. The warnings of the drawn basins are counted "
    "together, so a basin without a high-flow day or without an onset in the drawn years adds nothing to either count"
    + ("; every resample has high-flow days and onsets." if not any(key.startswith(("onsets_warned", "all_warned")) for key in no_value) else
       "; a resample without any onset has no share of onsets warned and is left out in the same way."),
    "runtime_s is the time of the whole script, including the checks.",
] + new_notes + ([f"QUICK smoke test: {B} resamples of the tiny smoke-test models - numbers are not results."] if F.QUICK else [])

R = dict(
    question="How sure are the headline numbers: in which range does each stay when the test years and the basins are drawn again?",
    what="Bootstrap intervals for the headline numbers, from the stored forecasts (nothing trained). Test years, warnings at the high-flow level itself "
         f"(names ending in {AT}, and {AT_KEY} in a verdict cell: at the warning level chosen on the validation years).",
    quick_mode=F.QUICK,
    how_to_read=f"An interval is the range in which the number stays when the test years and the basins are drawn again: {B:,} times, {ny} water years are drawn from the {ny} test "
                f"water years and {nb} basins from the {nb} basins, with replacement, and the interval runs from the 2.5th to the 97.5th percentile of the results (95 of 100 of them "
                f"lie inside it). With only {nb} basins and {ny} year-blocks the intervals are wide; they show how much a number depends on which years and which basins it was "
                f"measured on, and they are not a proof. They say nothing about rivers of another kind, about another region or about years unlike these {ny}, and they do not contain "
                "the difference between two trainings of the same network.",
    resampling=dict(B=B, units=f"{ny} test water years ({YEARS[0]}-{YEARS[-1]}) and {nb} basins, each drawn with replacement; a whole water year stays together",
                    designs=f"years: only the years are resampled; basins: only the basins; both: both at once (the widest for {n_widest} of the {len(range_old)} statistics that have "
                            "a range - use this one in the text)",
                    designs_of_the_statistics_added_2026_10_08=f"the widest for {n_widest_new} of the {len(range_new)} statistics added on 2026-10-08 that have a range (the sentence above is about the "
                                                               f"{len(range_old)} of 4 October)",
                    interval="2.5th to 97.5th percentile of the resampled statistic (percentile bootstrap)", seed=SEED, n_years=ny, n_basins=nb, test_water_years=YEARS,
                    water_year=f"the 12 months from {t0.day} {t0.month_name()}, named after the year in which they end; taken from the test period of flood.SPLIT ({F.SPLIT['test'][0]} to {F.SPLIT['test'][1]})",
                    statistics_without_a_value_in_some_resamples=no_value,
                    monte_carlo_noise=dict(
                        what=f"How far an end of an interval (design both) moves with the random numbers, because {B:,} is a finite number of resamples: the spread of the same end "
                             f"over {MC_PARTS} equal parts of the resamples, divided by the square root of {MC_PARTS} (a standard error, itself rough). With another seed an end "
                             "differs by about this much as a rule and seldom by more than 4 times this, so the last digit printed for an end is not firm. "
                             f"ends_near_zero: the statistics with an end closer to zero than {NEAR} times its noise (an end at zero itself is not counted); whether such a "
                             "range holds zero can change with the random numbers.",
                        ends_near_zero=near_zero, typical=mc_typical, largest_among_headline_statistics=mc_headline, largest=mc_largest, largest_for=mc_largest_for, parts=MC_PARTS,
                        headline_statistics={key: dict(lo=float(MC[key][0]), hi=float(MC[key][1])) for key in show if key in MC},
                        statistics_added_2026_10_08=dict(
                            what="the same figures for the statistics added on 2026-10-08 only (typical, largest, largest_for, n_ends_near_zero); the figures above are those of the statistics of 4 October, "
                                 "as then; ends_near_zero lists the statistics of both (those of 4 October first)",
                            n_statistics=len(range_new), typical=mc_typical_new, largest=mc_largest_new, largest_for=mc_largest_for_new, n_ends_near_zero=len(near_zero_new),
                            headline_statistics={key: dict(lo=float(MC[key][0]), hi=float(MC[key][1])) for key in show_new if key in MC}))),
    n_onsets_test={lk(k): int(n_on[:, :, j].sum()) for j, k in enumerate(F.LEADS)}, n_high_flow_days_test=int(n_hi.sum()), n_test_days_per_basin=int(len(T)),
    best_simple_model=BEST, prediction_sets=LABEL,
    stat_names={
        "<stat>.<set>.l<k>": "set = a prediction set (see prediction_sets), or best_simple_w / best_simple_wq = the simple model chosen on the validation years for that lead "
                             "(best_simple_model); k = days ahead. Each entry: value (the test years as they are) and, for the designs years / basins / both, lo, hi and share_above_zero "
                             "(the share of the resamples above 0).",
        "nse_median": "median over the basins of the NSE",
        "nse_diff_of_medians.<a>_minus_<b>": "median NSE of a minus median NSE of b (the two medians can come from different basins)",
        "nse_median_of_diffs.<a>_minus_<b>": "median over the basins of (NSE of a minus NSE of b in the same basin): the change in the typical basin",
        "pairs": {name: f"{a} minus {b}" for name, (a, b) in PAIRS.items()},
        "skill_vs_persistence_median": "median over the basins of 1 - squared error of the set / squared error of 'tomorrow = today'",
        "spread_ratio_median, correlation_median, total_ratio_median, kge_median": "medians over the basins of the parts explained in spread_and_skill, for its sets: the same values as "
                                                                                   "spread_and_skill.medians, here with intervals",
        "onsets_warned, onset_false_warnings, onset_csi": "an onset is a high-flow day whose forecast was issued while the river was still at or below the high-flow level "
                                                           "(flood.flood_scores); share of the onsets that were warned, share of the warnings issued in that situation that were false, "
                                                           "hits / (hits + misses + false warnings); the basins counted together",
        "all_warned, all_false_warnings, all_csi": "the same three rates for all high-flow days",
        f"<rate>{AT}": f"the same six rates at the warning level that 06_flood_analysis.py chose on the validation years ({', '.join(CHOSEN)}; the levels are in verdict_stability)",
        "onsets_gain.<what>.<w or wq>": "difference in the share of onsets warned, without (w) or with (wq) the river gauge: evening_rain_minus_base = with rain up to the evening of the issue "
                                        "day minus without; evening_rain_minus_control = minus the control (the same rain one day older); upper_bound_minus_base = perfect weather forecast "
                                        "minus without; lstm_minus_best_simple, lstm_minus_rf, lstm_minus_linreg = LSTM minus the best simple model, the random forest, the ridge regression",
        "onsets_gap_share_closed": "evening_rain_minus_base / upper_bound_minus_base: the share of the gap to the perfect weather forecast (an upper bound that cannot be had in practice) "
                                   "that the rain up to the evening of the issue day closes",
        "onsets_gap_share_closed_beyond_control": "evening_rain_minus_control / upper_bound_minus_base",
        "onsets_diff.<pair>": "added on 2026-10-08, for the pairs of PAIRS_NEW (see pairs): the share of onsets warned by a minus the share warned by b, the basins counted together (a fraction: 0.10 is 10 percentage "
                              "points); the same two sets as in nse_diff_of_medians.<pair>",
        "onsets_gap_share_closed.<closure>": "added on 2026-10-08, for the closures under closures: (share of onsets warned by the set - by the base) / (by the upper bound - by the base), the share of the gap "
                                             "in onsets warned that the set closes; the upper bound is a model that cannot be had in practice; <name>_beyond_control, where the closure has a control: "
                                             "(set - control) / (upper bound - base)",
        "closures": {name: f"({w} - {base}) / ({ceil} - {base})" + (f"; beyond the control: ({w} - {ctrl}) / ({ceil} - {base})" if ctrl else "") for name, (w, base, ceil, ctrl) in CLOSE.items()}},
    checks=dict(numbers_compared=n_compared, numbers_compared_before_resampling=n_before, largest_difference_in_a_directly_scored_resample=worst,
                what=["the high-flow levels are those of flood.thresholds on the stored observed flow, and equal the stored ones",
                      f"flood.evaluate on each of the {len(SETS_OLD)} stored prediction sets gives the evaluation stored in the results files (test years, every basin and lead)",
                      "the best simple model named in lstm_all.json is the one with the highest median NSE in the validation years, recomputed from the stored forecasts",
                      "the sums per basin and water year give that evaluation: NSE to 1e-9, hits, misses and false warnings exactly, for all high-flow days and for onsets",
                      "the statistics on the test years as they are equal the stored comparison numbers of lstm_all.json, rain_timing.json, lstm_lobo.json, model_comparison.json and "
                      "the warning tables and verdicts of flood_analysis.json",
                      f"the first {min(N_DIRECT, B)} resamples scored directly with flood.nse and flood.contingency give the numbers of the fast sums"]
                     + [f"added on 2026-10-08: flood.evaluate on each of the {len(SETS_NEW)} added prediction sets gives the evaluation stored by its script (the whole evaluation in "
                        f"{', '.join(STORED_NEW[i] for i in (0, 1, 3))} and lstm_lobo.json; for the {len(E3_SETS)} sets of {STORED_NEW[2]} the pooled summary and the per-river scores), and the sums per basin "
                        "and water year give that evaluation as for the older sets",
                        f"added on 2026-10-08: the statistics on the test years as they are of {len(CHECK_PAIRS)} differences and {len(CHECK_CLOSE)} closures equal the comparison numbers that the four scripts "
                        "store (gain_rule, closure, yardstick.lstm_against_simple, yardstick.new_input_given_to_each_model, unseen, gap_to_the_upper_bound): medians, onset shares, false shares, shares of the gap",
                        "added on 2026-10-08: the verdict of every added set with warnings, at the high-flow level itself on the test years as they are, is the verdict its own script stored",
                        "added on 2026-10-08: all four results files give the test years of flood.SPLIT, and the sets taken from forecast_input.json are those of its main-split arm"],
                numbers_compared_by_file_for_the_added_sets_and_comparisons=by_file, verdict_labels_compared=n_labels,
                differences_without_a_stored_counterpart=PAIRS_UNSTORED),
    stats=U,
    verdict_stability=V,
    verdict_criteria={**{x: crit[x] for x in VERDICTS}, "source": "results/flood_analysis.json verdict.criteria",
                      "how_to_read": "verdict = at the high-flow level itself, on the test years as they are; share_of_resamples_with_each_verdict = how often each verdict comes out "
                                     f"when years and basins are drawn again (design both). {AT_KEY} repeats both at the warning level that 06_flood_analysis.py chose on the "
                                     "validation years (warning_level, a multiple of the high-flow level, kept fixed in every resample)."},
    spread_and_skill=dict(
        what=f"Per basin over the {len(T)} test days, then the median of the {nb} basins; no resampling. Sets: " + "; ".join(f"{n} = {LABEL[n]}" for n in DIAG) + ".",
        how_to_read="spread_ratio = standard deviation of the forecasts / standard deviation of the observed flow (1 = as lively as the river; below 1 = the forecasts vary less than "
                    "the river: peaks too low, low flows too high); correlation = between forecast and observed flow; total_ratio = mean forecast / mean observed flow (1 = the right "
                    "amount of water); skill_vs_persistence = 1 - squared error of the set / squared error of 'tomorrow = today' (0 = no better than a copy of the last flow); "
                    "kge = 1 - distance of (correlation, spread_ratio, total_ratio) from (1, 1, 1). "
                    "medians['<part>_median.<set>.l<k>'] is the median of the basins; the medians of the parts need not come from the same basin. The same medians are in stats "
                    "under the same names, with intervals (only the skill of 'tomorrow = today' against itself, which is 0, is not). spread_ratio_minus_correlation gives, over the "
                    "basins, the smallest, the median and the largest difference of the two in the same basin.",
        medians=medians, spread_ratio_minus_correlation=spread_gap, per_basin=per_basin),
    added_2026_10_08=ADDED,
    notes=notes, runtime_s=round(time.time() - t_start, 1))
tmp = F.RES_OUT / "uncertainty.json.tmp"                     # written whole, then moved into place: a document builder that runs meanwhile never reads half a file
tmp.write_text(json.dumps(R, indent=1))
tmp.replace(F.RES_OUT / "uncertainty.json")

f3 = lambda x: f"{x:7.3f}" if ok(x) else "    nan"
print(f"{'statistic (value, then the range when years and basins are drawn again)':66s}   value  [    lo,     hi]   years only          basins only")
for key in show + show_new:
    e = U[key]
    print(f"{key:66s} {f3(e['value'])}  [{f3(e['both']['lo'])}, {f3(e['both']['hi'])}]  [{f3(e['years']['lo'])}, {f3(e['years']['hi'])}]  [{f3(e['basins']['lo'])}, {f3(e['basins']['hi'])}]")
print(f"wrote {(F.RES_OUT / 'uncertainty.json').relative_to(ROOT)}: {len(U)} statistics, {len(V)} verdict cells, {len(medians)} medians of spread and skill; {R['runtime_s']} s")
