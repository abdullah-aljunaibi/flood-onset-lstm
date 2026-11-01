"""E7: several rain products side by side as inputs of the all-river LSTM (ROADMAP.md, section 'E7 - several rain products side by side', rule fixed on 2026-10-30 before the first training;
design NEXT_EXPERIMENT_E7_multi_product.md, section 2 = E7a; the MRMS arm E7b was ruled out by the team lead on 2026-10-30 and is not here).

Every LSTM of this project with a second rain product kept the Daymet rain table and read ONE more product beside it: NLDAS (lstm_wqn, script 10), the ERA5 rain to midnight (lstm_wqh, script 15),
both (lstm_wqnh, script 15) or the AORC rain to midnight (lstm_wqa, script 16; AORC is an analysis made after the fact). This script gives one LSTM all of them side by side and asks whether it warns
of more onsets 1 day ahead than lstm_wqa, the best single product. The three daily columns are rebuilt from the data on disk by the routes of the scripts that trained on them:
   prcp_nldas        the NLDAS 'prcp' column shipped with CAMELS (camels.load_basin(g, 'nldas')), read as scripts 10 and 15 read it
   rain_to_midnight  the hourly ERA5 rain of data/hourly/era5_<gauge>.json (check_day_windows.py) summed over the local wall-clock day t (hour-ENDING stamps, summer time included), as script 15
   aorc_to_midnight  the hourly AORC basin means of data/aorc (made by code/fetch_aorc.py) summed the same way, as script 16 and code/check_reload.py
 Column-identity check, BEFORE any training: the stored networks lstm_wqa (script 16: AORC) and lstm_wqnh (script 15: NLDAS + ERA5), seeds 0 (trained on the Mac) and 3 (trained on this PC), are
 rebuilt from results/models and run on this script's table; their forecasts must equal the stored ones of results/preds (numpy.allclose rtol 1e-6, atol 1e-8, the tolerance of the scripts' reload
 asserts), else the script stops (SystemExit) and trains nothing.
 Sets (settings of results/lstm_all.json, unchanged; the river gauge and the 10 basin properties; seeds 0-4 as ROADMAP.md fixes; --seeds 3 only for a staged run, then recorded):
   wqME        FORCING + prcp_nldas + rain_to_midnight + aorc_to_midnight                                   the treatment
   wqME_shuf   as wqME, prcp_nldas and rain_to_midnight shuffled within basin x calendar month across the years of the same split part (one permutation per basin-part-month group, shared by
               the two columns, random seed SHUF_SEED, built once: the same data for every training seed)                  the capacity control: same column count and distributions, no timing information
   wqME_older  as wqME, prcp_nldas and rain_to_midnight one day older (day t reads day t-1; the first day keeps its own value, as the _older controls of scripts 10, 15, 16)   the clock control
   base_linreg_wqME, base_rf_wqME   ridge regression and random forest by the recipe of 02_baselines.py / features.py as script 16 does it for one extra column, here with the rain features of the
               three columns; the recipe is proved first, before any training: with aorc_to_midnight alone this code reproduces the stored base_linreg_wqa and, outside a smoke test, base_rf_wqa
               of script 16 (asserted, below 1e-9 mm/day)
 Compared with the STORED sets lstm_wqa (the base, five seeds), lstm_wq, lstm_wqn, lstm_wqh, lstm_wqnh and the stored simple models base_linreg_wqa, base_rf_wqa, always read from results/preds and
 rescored by this script's code, never copied (checks_against_stored_results proves the same numbers as their own results files).
 The rule of ROADMAP.md as four recorded blocks (lead 1, trigger 1.0, onsets pooled over the 12 basins, test years; e7_rule):
   1 gain: every single seed of wqME above every single seed of lstm_wqa on onsets warned; the onset false share of the seed-averaged wqME not higher than lstm_wqa's; two-sided Wilcoxon signed-rank
     over the 12 basins on the per-basin onsets warned of the seed-averaged forecasts, p < 0.05 (n, the zero differences and the null distribution recorded)
   2 capacity control flat: the median of the single-seed onsets-warned values of wqME_shuf lies within the lowest-highest single-seed range of lstm_wqa
   3 clock control flat: the same test for wqME_older
   4 network yardstick, reported and not part of the verdict: the gain of ridge and forest from the wqME columns against their wqa versions of script 16, beside the LSTM's gain
   a clear gain only if 1, 2 and 3 all hold. Also: median NSE at leads 1-3, paired onsets, the GO / CAUTION / NOT YET verdict of results/flood_analysis.json, the 30%-false operating point of
   code/warning_policy.py (chosen on the validation years), and resampling intervals by the recipe of 13_uncertainty.py (test water years and basins drawn again with replacement).
 Causality: the data side (two routes to every daily total of both hourly products, and a made-up-hours test: hours ending after the issue time of a cut day changed -> no day up to the cut moves);
 flood._assert_causal inside every flood.fit; the made-up-values test of scripts 18 and 21 on one saved network per new set; features.py's own check on the simple-model features with the three columns.
 The shuffled columns move values between the years of the same split part (about half of them to a day BEFORE the day they were measured; never across training, validation and test years): a capacity
 control whose columns carry no timing information by construction, never a forecast input (shuffle record in inputs.shuffle).
Reads data/hourly, data/aorc, the CAMELS NLDAS and Daymet forcing, results/lstm_all.json, results/flood_analysis.json, results/day_windows.json (subdaily_inputs.zones), results/baselines.json,
results/aorc_rain_to_midnight.json and results/rain_to_midnight.json (the check of the rescored numbers), results/review_aorc_rain_to_midnight.json (the hindsight wording, as script 21), ROADMAP.md
(its E7 section is hashed into every pass), the stored sets of results/preds and results/models, and results/review_multi_product.json when it exists (the independent review; read, never written).
Writes results/multi_product.json, figures/multi_product.png, results/models/lstm_<set>_s<seed>.pt, the prediction sets lstm_<set>_s<seed>, lstm_<set> (the seed mean), base_linreg_wqME, base_rf_wqME.
Downloads nothing: fetch_aorc is read through load_series / cache_complete only, and a missing input stops the script.
Modes: plain run = train everything; --rescore = recompute from the saved sets (nothing trained); --extend = train only the missing (set, seed) pairs and simple models; --figure = the figure again from
the results file. FLOOD_QUICK=1: one seed, 40 updates, tiny forests, 200 resamples, results/quick and figures/quick (the stored sets are read from results/preds and results/models).
The syntax of this file is checked with python -X utf8 -m py_compile code/28_multi_product.py.
Run (project root, PYTHONUTF8=1): python -X utf8 code/28_multi_product.py [--rescore | --extend] [--seeds 3|5] [--figure]
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_aorc as AO                                     # read only: load_series, basin_file, cache_complete (its download path is never called)
import flood as F                                           # the shared engine (it imports torch)
import subdaily_inputs as SI                                # zones() and wall_clock(): the time zone of every basin, the rule of scripts 16 and 21
import warning_policy as WP
from camels import load_basin
from common import AQUA, AXIS, BLUE, INK, INK2, MUTED, ORANGE, ROOT, YELLOW, use_style
from features import feature_table

use_style()
SCRIPT, OUT, t_start = "28_multi_product.py", "multi_product", time.time()        # results/<OUT>.json, figures/<OUT>.png
HASH_NOW = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()      # the code that runs, hashed once at the start: every pass records it

ap = argparse.ArgumentParser(description="E7: several rain products side by side as LSTM inputs (ROADMAP.md, section E7)")
ap.add_argument("--rescore", action="store_true", help="no training: scores, tables, outcome and figure again from the saved prediction sets")
ap.add_argument("--extend", action="store_true", help="as --rescore, but trains the (set, seed) pairs and simple models whose files are missing")
ap.add_argument("--figure", action="store_true", help="only the figure again, from the results file")
ap.add_argument("--seeds", type=int, default=5, choices=(3, 5), help="5 (seeds 0-4, the number ROADMAP.md fixes) or 3 (seeds 0-2, a staged run; --extend --seeds 5 adds 3 and 4 and records it)")
ARGS = ap.parse_args()
RESCORE, EXTEND = ARGS.rescore, ARGS.extend
assert not (RESCORE and EXTEND), "--rescore trains nothing and --extend trains what is missing: use one of the two"
LOAD = RESCORE or EXTEND                                    # the saved sets and the training records of the results file are read, not made again
SEEDS = [0] if F.QUICK else list(range(ARGS.seeds))
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")        # the training record of one model (Run.summary)
P_CUT = 0.05                                                # 'p is small', as in section 4.2 of the report and in rule block 1
MODELS, STORED_MODELS = F.RES_OUT / "models", F.RES / "models"                # this run's networks (results/quick/models in a smoke test); the stored networks of scripts 15 and 16 (always the real folder)
RESULTS_FILE, FIG_FILE = F.RES_OUT / f"{OUT}.json", F.FIG_OUT / f"{OUT}.png"
REVIEW_FILE, REAL_PRED = F.RES / f"review_{OUT}.json", F.RES / "preds"       # the independent review record, when it exists; the stored sets are ALWAYS read from results/preds
ROADMAP_FILE, E7_HEADING, FIXED_ON = ROOT / "docs" / "ROADMAP.md", "### E7 - several rain products side by side", "2026-10-30"
HOUR, DAYS = pd.Timedelta(hours=1), pd.date_range(*F.PERIOD)
NEW2 = ("prcp_nldas", "rain_to_midnight")                  # the two columns the controls change (NLDAS, ERA5); AORC stays as it is in every set
ME_COLS = NEW2 + ("aorc_to_midnight",)                     # the three rain products beside the rain table, in the order ROADMAP.md lists them
SHUF_SEED = 20261030                                        # the random seed of the shuffle (fixed in this script before the first training)
B_RS, SEED_RS, N_DIRECT = (200 if F.QUICK else 4000), 20261004, 2              # resamples and seed of 13_uncertainty.py; resamples also scored the slow way
IDENTITY_SEEDS = (0, 3)                                     # the stored networks of the column-identity check: seed 0 trained on the Mac, seed 3 on this PC (scripts 15 and 16, SEED_NOTE)
CONF_KEYS = {"hidden", "lr", "dropout", "warmup", "block", "batch", "updates", "eval_every", "patience"}       # the settings results/lstm_all.json holds
AORC_WORDING = "an analysis made after the fact"            # what AORC is - never 'observed' rain (code/report_sections/s0_summary.py)
UPPER_WORDING = "an upper bound that cannot be had in practice"
AFTER_THE_FACT_ALL = ("none of the three extra rain products is had at midnight of the issue day: NLDAS is published a few days late and is a daily gauge analysis spread over the hours "
                      "(results/rain_timing.json), ERA5 is a reanalysis (results/rain_to_midnight.json), and AORC is " + AORC_WORDING)     # review of 2026-10-30, must-fix 1


def _hindsight_wording():
    """'hindsight at most about N points' beside the AORC rain-to-midnight share: N = the loosest bound of the Stage IV route for the networks the stored
    lstm_wqa set averages (results/review_aorc_rain_to_midnight.json, the latest round with a five-seed bound; 30 October 2026: 3.6 points, five networks);
    the review of 7 October 2026 measured 'about 3 points' on the three networks of seeds 0-2, the fallback when no later round holds a bound. (The source and code of script 21.)"""
    import json as _json
    from pathlib import Path as _Path
    try:
        rec = _json.loads((_Path(__file__).resolve().parents[1] / "results" / "review_aorc_rain_to_midnight.json").read_text(encoding="utf-8"))
        rounds = sorted((k for k in rec if k.startswith("round_") and k[6:].isdigit()), key=lambda k: int(k[6:]))
        for k in reversed(rounds):
            pts = ((rec[k] or {}).get("hindsight") or {}).get("five_seed_loosest_bound_points")
            if pts is not None:
                return f"hindsight at most about {round(float(pts))} points"
    except (OSError, ValueError, TypeError):
        pass
    return "hindsight at most about 3 points"


HINDSIGHT_WORDING = _hindsight_wording()          # five-seed bound since 30 October 2026 (registry key fw_hindsight prints the same)
NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
lk = lambda k: f"l{k}"
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)                         # a path as the project shows it
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
pc = lambda x: f"{round(100 * x) + 0}%" if x == x and x is not None else "no value"
p1 = lambda x: f"{100 * x:.1f}%" if x == x and x is not None else "no value"
n2 = lambda x: f"{x:.2f}"
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
pv = lambda p: "p not defined" if p != p else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None       # lowest and highest single seed (a seed with no value is left out)
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                             # every single seed of a above every single seed of b
div = lambda a, b: a / b if b else float("nan")
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
close = lambda a, b: (a != a and b != b) or abs(a - b) < 1e-9
notes = []
real_have = lambda name: (REAL_PRED / f"{name}.npz").exists()
own_have = lambda name: (F.PRED / f"{name}.npz").exists()                                          # a file of THIS run's folder (a smoke test looks at results/quick only)


def real_preds(name):
    """A stored prediction set of the real run (results/preds), in the form of flood.load_preds (flood.load_preds would prefer a smoke-test copy of the same name in results/quick/preds)."""
    z = np.load(REAL_PRED / f"{name}.npz", allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"])
    return {str(b): pd.DataFrame({"obs": z["obs"][i], **{f"l{k}": z["pred"][i, :, j] for j, k in enumerate(F.LEADS)}}, index=idx) for i, b in enumerate(z["basins"])}


def real_meta(name):
    return json.loads(str(np.load(REAL_PRED / f"{name}.npz", allow_pickle=False)["meta"]))


def own_preds(name):
    """A prediction set of THIS run's folder (results/preds, or results/quick/preds in a smoke test), read from that folder only."""
    z = np.load(F.PRED / f"{name}.npz", allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"])
    return {str(b): pd.DataFrame({"obs": z["obs"][i], **{f"l{k}": z["pred"][i, :, j] for j, k in enumerate(F.LEADS)}}, index=idx) for i, b in enumerate(z["basins"])}


def roadmap_e7():
    """The E7 section of ROADMAP.md as it reads now (from its heading to the next level-2 heading) and its sha256: the rule this script codes, recorded with every pass."""
    if not ROADMAP_FILE.exists():
        return None
    text = ROADMAP_FILE.read_text(encoding="utf-8")
    i = text.find(E7_HEADING)
    if i < 0:
        return None
    j = text.find("\n## ", i)
    sec = text[i:j if j > 0 else len(text)].strip()
    return dict(file="docs/ROADMAP.md", heading=sec.splitlines()[0].lstrip("# ").strip(), section_sha256=hashlib.sha256(sec.encode("utf-8")).hexdigest(), n_characters=len(sec))


E7_SECTION = roadmap_e7()
assert E7_SECTION and FIXED_ON in E7_SECTION["heading"], f"ROADMAP.md holds no E7 section fixed on {FIXED_ON}: the rule this script codes must be fixed there before the first training"
RULE = dict(
    where=f"ROADMAP.md, section '{E7_SECTION['heading']}'",
    scope="lead 1, trigger 1.0, onsets pooled over the 12 basins, test years (flood.SPLIT); seed-averaged forecasts unless 'single seed'",
    block_1_gain="every single seed of wqME above every single seed of lstm_wqa on onsets warned; the onset false share of the seed-averaged wqME not higher than that of lstm_wqa; Wilcoxon signed-rank "
                 "(two-sided) over the 12 basins on the per-basin onsets warned of the seed-averaged forecasts, p < 0.05",
    block_2_capacity_control_flat="the median of the five single-seed onsets-warned values of wqME_shuf lies within the lowest-highest single-seed range of lstm_wqa (if wqME_shuf gains as much as wqME, "
                                  "the gain is capacity, not information)",
    block_3_clock_control_flat="the same test for wqME_older",
    block_4_network_yardstick="reported, not part of the verdict: if ridge and forest gain as much from the wqME columns (against their wqa versions of script 16) as the LSTM does, the reading is "
                              "'the information, not the network' again",
    overall="a clear gain only if blocks 1, 2 and 3 all hold",
    expected_before_running="0 to 4 points over wqa, probably failing the strict seed rule, as E4b did (ROADMAP.md, NEXT_EXPERIMENT_E7_multi_product.md section 6)")
INTERPRETATIONS = [
    "Block 1, seed rule: 'above' is strict: the lowest single seed of wqME must be greater than the highest single seed of lstm_wqa (the 'above' of scripts 15-22).",
    "Block 1, false share: 'not higher' means the pooled onset false share of the seed-averaged wqME is at most that of the seed-averaged stored lstm_wqa (equal counts as not higher).",
    "Block 1, Wilcoxon: the per-basin onsets warned are each basin's onset POD at trigger 1.0 from the seed-averaged forecasts; a basin without an onset in the test years would have no value and is "
    "left out (counted); zero differences are dropped (scipy's default zero_method='wilcox'), so n_used is the number of basins with a non-zero difference; the p is scipy.stats.wilcoxon with its "
    "default method='auto' (as paired() and the scripts before), and the normal-approximation and exact p are recorded beside it. p < 0.05 is required as written; the direction is carried by the seed "
    "rule of the same block (a significant loss cannot pass it).",
    "Blocks 2 and 3: 'within' is inclusive (lowest <= median <= highest); the median is numpy's median of the single-seed onsets-warned values of the control (five seeds: the third value); the range of "
    "lstm_wqa is over all its stored single seeds (five since 30 October 2026).",
    "Block 4: 'as much' is read on onsets warned at lead 1 (seed-averaged LSTM forecasts, the deterministic simple models): ridge and forest gain as much when each one's gain in points is at least the "
    "LSTM's gain; the reading is given in words and is not part of the verdict.",
    "The shuffle (capacity control): within each basin, split part (training, validation, test years) and calendar month, ONE permutation of all days of that month in that part is drawn with numpy's "
    "default_rng(SHUF_SEED), basin after basin in the order of the study basins, part after part and month after month (no value crosses a split part); day i of the group then reads the NLDAS and the ERA5 value "
    "measured on day perm(i) (one permutation shared by the two columns, so their joint distribution is kept). It is built once, before any training: every seed trains on the same shuffled table.",
    "The older control: day t reads the NLDAS and ERA5 values of day t-1; the first day of the record keeps its own value (the _older controls of scripts 10, 15 and 16).",
    "The resampling interval: 13_uncertainty.py's recipe (test water years and basins drawn again with replacement, B = 4,000, seed 20261004, 2.5th-97.5th percentile of the seed-averaged "
    "forecasts' statistic; the 'both' design is the interval quoted, 'years' and 'basins' are stored beside it); the Monte-Carlo noise of the interval ends (13's mc_noise) is not computed."]


COLUMN_TEXT = {
    "prcp_nldas": "mm: the NLDAS basin-mean rain shipped with CAMELS (the 'prcp' column of the nldas forcing files read by camels.load_basin), the column of scripts 10 and 15 (their sets wqn and wqnh); "
                  "its day ends at about 17:30 local standard time on day t as measured (results/day_windows.json, day_ends.nldas; the reading of 10_rain_timing.py, not proved again here)",
    "rain_to_midnight": "mm: the hourly ERA5 rain of data/hourly/era5_<gauge>.json (the value at time T is the rain of the hour ENDING at T, UTC) summed over the local wall-clock day t of the gauge "
                        "(00:00 to 24:00, summer time included: every hour goes to the day in which it ends; 23 or 25 hours where the clocks change): the column of script 15 (sets wqh, wqnh)",
    "aorc_to_midnight": "mm: the hourly AORC basin means of data/aorc (code/fetch_aorc.py; hour-ENDING UTC stamps; " + AORC_WORDING + ") summed over the same local wall-clock day: the column of "
                        "script 16 (set wqa) and of code/check_reload.py",
    "older": "prcp_nldas_older, rain_to_midnight_older: the same two columns one day older: on issue day t the network reads the values of day t-1 (the first day keeps its own value)",
    "shuf": "prcp_nldas_shuf, rain_to_midnight_shuf: the same two columns shuffled within basin x calendar month across the years of the same split part, one permutation per basin-part-month group shared "
            "by the two columns (inputs.shuffle): the same values and month-by-month distributions per basin and split part, no timing information about any day"}


# ---- figure -----------------------------------------------------------------------------------------------------------------------------------------
ORDER = ("wq", "wqn", "wqh", "wqnh", "wqa", "wqME", "wqME_shuf", "wqME_older", "base_linreg_wqa", "base_linreg_wqME", "base_rf_wqa", "base_rf_wqME")
LABEL = {"wq": "rain table\nalone\n(wq)", "wqn": "+ NLDAS\n(wqn)", "wqh": "+ ERA5 to\nmidnight\n(wqh)", "wqnh": "+ NLDAS\n+ ERA5\n(wqnh)", "wqa": "+ AORC to\nmidnight\n(wqa, base)",
         "wqME": "+ NLDAS\n+ ERA5\n+ AORC\n(wqME)", "wqME_shuf": "NLDAS, ERA5\nshuffled\n(capacity\ncontrol)", "wqME_older": "NLDAS, ERA5\none day\nolder (clock\ncontrol)",
         "base_linreg_wqa": "ridge,\nwqa\ncolumns", "base_linreg_wqME": "ridge,\nwqME\ncolumns", "base_rf_wqa": "forest,\nwqa\ncolumns", "base_rf_wqME": "forest,\nwqME\ncolumns"}
COLOUR = {"wq": AXIS, "wqn": AXIS, "wqh": AXIS, "wqnh": AXIS, "wqa": AQUA, "wqME": BLUE, "wqME_shuf": BLUE, "wqME_older": BLUE,
          "base_linreg_wqa": YELLOW, "base_linreg_wqME": YELLOW, "base_rf_wqa": ORANGE, "base_rf_wqME": ORANGE}
HOLLOW = {"wqME_shuf", "wqME_older"}


def draw(R):
    """Lead 1, test years: onsets warned (top; each label also gives the false share of the onset warnings) and median NSE (bottom) per set; bars = the seed-averaged forecast, dots = every single
    seed; the band = the lowest-highest single seed of lstm_wqa (the band of rule blocks 2 and 3); stored sets hatched, controls hollow. Every number on the figure comes from the results file."""
    import textwrap
    FS = 8.5
    S, RG, E = R["summary_test"]["sets"], R["seed_ranges"]["sets"], R["e7_rule"]
    stored = set(R["design"]["stored_sets"]) | set(R["design"]["stored_simple_models"])
    order = [n for n in ORDER if n in S]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.6, 10.4), sharex=True, gridspec_kw=dict(hspace=0.32, top=0.80, bottom=0.2, left=0.07, right=0.985))
    x = np.arange(len(order))
    band = RG["wqa"]["l1"]["onset_pod"]
    ax1.axhspan(100 * band[0], 100 * band[1], color=AQUA, alpha=0.2, linewidth=0, zorder=0)
    for ax, key in ((ax1, "onset_pod"), (ax2, "nse_median")):
        for i, n in enumerate(order):
            y = 100 * S[n]["l1"]["onset"]["pod"] if key == "onset_pod" else S[n]["l1"]["nse_median"]
            ax.bar(i, y, width=0.62, color="none" if n in HOLLOW else COLOUR[n], edgecolor=COLOUR[n], linewidth=1.5, alpha=0.95 if n in HOLLOW else 0.75,
                   hatch="///" if n in stored else None, zorder=2)
            top = y
            if n in RG:
                seeds = [100 * s if key == "onset_pod" else s for s in RG[n]["l1"]["by_seed"][key]]
                ax.scatter(np.full(len(seeds), i) + np.linspace(-0.14, 0.14, len(seeds)), seeds, s=16, color=INK, zorder=4, linewidths=0)
                top = max([y] + seeds)
            far = S[n]["l1"]["onset"]["far"]
            lab = (f"{y:.0f}%\n({100 * far:.0f}% false)" if far is not None else f"{y:.0f}%") if key == "onset_pod" else f"{y:.2f}"
            ax.text(i, top + (1.0 if key == "onset_pod" else 0.012), lab, ha="center", va="bottom", fontsize=FS - 1.2, color=INK2, zorder=5, linespacing=1.1)
        ax.set_xticks(x)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=FS)
        ax.axvline(len([n for n in order if not n.startswith("base_")]) - 0.5, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
    ax1.axhline(70, color=MUTED, linewidth=1.0, zorder=1)
    ax1.text(len(order) - 0.55, 71, "GO asks 70% warned (at most 30% false)", fontsize=FS - 1, color=MUTED, ha="right", va="bottom")
    ax1.text(-0.45, 100 * band[1] + 0.8, "band: lowest to highest single seed of lstm_wqa (rules 2 and 3)", fontsize=FS - 1.2, color=INK2, ha="left", va="bottom")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("onsets warned 1 day ahead (%)", fontsize=FS + 0.5)
    ax1.set_title("Onsets warned 1 day ahead (test years 2005-2014, trigger 1.0, 12 basins pooled)", fontsize=FS + 1.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("median NSE over the basins", fontsize=FS + 0.5)
    ax2.set_title("Median NSE at 1 day (test years)", fontsize=FS + 1.5)
    ax2.set_xticklabels([LABEL[n] for n in order], fontsize=FS - 0.5)
    wrap = lambda s, w: "\n".join(textwrap.fill(p, w) for p in s.split("\n"))
    n_new = len(R["seeds"])
    yes = lambda b: "holds" if b else "fails"
    rule_line = (f"Rule of ROADMAP.md (fixed {FIXED_ON} before the first training): 1 gain {yes(E['block_1_gain']['holds'])}; 2 capacity control flat {yes(E['block_2_capacity_control_flat']['holds'])}; "
                 f"3 clock control flat {yes(E['block_3_clock_control_flat']['holds'])}: {E['reading']}.")
    stored_seeds = ", ".join(f"{m} {RG[m]['l1']['n_seeds']}" for m in ("wqa", "wqh", "wqnh", "wq", "wqn") if m in RG)
    fig.text(0.07, 0.962, "E7: several rain products side by side as inputs of the LSTM with the river gauge", fontsize=12, weight="bold", color=INK, ha="left")
    fig.text(0.07, 0.935, wrap(f"Bars: the seed-averaged forecast; dots: every single training ({NUMBER_WORD.get(n_new, n_new)} seed{'s' if n_new != 1 else ''} per new set; seeds of the stored sets: {stored_seeds}). "
                               f"Hatched: stored sets, read from results/preds and rescored here. Hollow: the controls. Right of the dotted line: ridge regression and random forest (no seeds).\n{rule_line}\n{R['status']}", 175),
             fontsize=FS - 0.3, color=INK2, ha="left", va="top", linespacing=1.4)
    t_wqa = S["wqa"]["l1"]["onset"]
    review_line = (R.get("provenance") or {}).get("review_line", "this file's independent review is not yet done")
    fig.text(0.07, 0.018, wrap(f"After the fact: {AFTER_THE_FACT_ALL}. The {100 * t_wqa['pod']:.0f}% of wqa carries {HINDSIGHT_WORDING}, measured for AORC alone; it does not bound wqME, whose "
                               "NLDAS and ERA5 columns are after the fact too: wqME is a test of information, not a forecast that could be issued. "
                               f"The shuffled columns of the capacity control move values between the years of the same split part ({100 * R['inputs']['shuffle']['share_from_a_later_day']:.0f}% of them to a day "
                               "before they were measured); they carry no timing "
                               f"information by construction and are no forecast input. Onsets: target days above Q95 whose issue-day flow was at or below it ({t_wqa['hits'] + t_wqa['misses']} in the test years).\n"
                               f"The team's own checks are not yet done; {review_line}. Code written by Claude Code under the team lead's direction. Written by code/{SCRIPT} from results/{OUT}.json.", 190),
             fontsize=FS - 1.3, color=INK2, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


if ARGS.figure:                                             # only the figure again, from the results file (no tables, no networks)
    assert RESULTS_FILE.exists(), "--figure needs the results file of a run"
    draw(json.loads(RESULTS_FILE.read_text(encoding="utf-8")))
    print(f"figure written: {FIG_FILE}")
    sys.exit(0)

# ---- 0. data: the daily tables, the three rain products, the two controls ------------------------------------------------------------------------------
D, B = F.load_data(), F.basins()
ids, thr = list(D), F.thresholds(D)
n_b = len(ids)
for what, f in (("results/lstm_all.json", F.RES / "lstm_all.json"), ("results/flood_analysis.json", F.RES / "flood_analysis.json"), ("results/baselines.json", F.RES / "baselines.json"),
                ("results/day_windows.json", F.RES / "day_windows.json")):
    assert f.exists(), f"{what} is missing: this script reads it"
assert AO.cache_complete(ids), "the AORC series of the 12 basins are not (all) in data/aorc: this script downloads nothing (code/fetch_aorc.py makes them)"
ERA5_FILE = lambda g: F.DATA / "hourly" / f"era5_{g}.json"
assert all(ERA5_FILE(g).exists() for g in ids), "the cached ERA5 hourly rain (data/hourly/era5_<gauge>.json, check_day_windows.py) is missing for some basin: this script downloads nothing"
CONF_REAL = json.loads((F.RES / "lstm_all.json").read_text())["config"]
assert set(CONF_REAL) == CONF_KEYS, f"results/lstm_all.json config holds other keys than expected: {sorted(CONF_REAL)}"
FA = json.loads((F.RES / "flood_analysis.json").read_text())
ZONES = SI.zones()
assert set(ids) <= set(ZONES), "results/day_windows.json gives no time zone for some basin"


def era5_hours(g):
    """(hour-ending UTC stamps, rain of each hour in mm) of the cached Open-Meteo ERA5 file of one basin: the reading of script 15 (era5_hours) and check_day_windows.py."""
    j = json.loads(ERA5_FILE(g).read_text())
    assert j["timezone"] == "GMT" and j["utc_offset_seconds"] == 0 and j["hourly_units"]["precipitation"] == "mm", f"era5_{g}.json is not hourly rain in mm on a UTC axis"
    stamp, x = pd.DatetimeIndex(pd.to_datetime(j["hourly"]["time"])), np.array(j["hourly"]["precipitation"], float)
    assert len(stamp) == len(x) and ((stamp[1:] - stamp[:-1]) == HOUR).all(), f"era5_{g}.json: the hourly time axis has gaps or repeats"
    assert np.isfinite(x).all() and (x >= 0).all(), f"era5_{g}.json: missing or negative hourly rain"
    return stamp, x


def aorc_hours(g):
    """(hour-ending UTC stamps, basin-mean rain of each hour in mm) of data/aorc/aorc_<gauge>.npz: the reading of script 16 (aorc_hours) and code/check_reload.py."""
    stamp, x = AO.load_series(g)
    assert len(stamp) == len(x) == AO.N_HOURS and stamp[0] == AO.FIRST_STAMP and stamp[-1] == AO.LAST_STAMP, f"aorc_{g}.npz does not hold the span {AO.SPAN_FOR_THE_RECORD}"
    assert ((stamp[1:] - stamp[:-1]) == HOUR).all() and np.isfinite(x).all() and (x >= 0).all(), f"aorc_{g}.npz: gaps, repeats, missing or negative hourly rain"
    return stamp, x


def day_key(stamp, zone):
    """The local calendar day (wall clock of the gauge, summer time included) in which each hour ENDS; an hour that ends at exactly 24:00 belongs to the day that ends then (scripts 15 and 16)."""
    return (SI.wall_clock(stamp, zone) - pd.Timedelta(minutes=1)).normalize()


def rain_by_wall_clock_day(x, key):
    """rain_by_wall_clock_day of scripts 15 and 16 (the same grouping and sum), with the day of every hour computed once by day_key: the rain of every day and its number of hours."""
    by_day = pd.Series(x, index=key).groupby(level=0)
    return by_day.sum(), by_day.size()


def second_route(stamp, x, zone):
    """The daily totals by a second route (subdaily_inputs.second_route, for any hourly series): the UTC instants of 00:00 and 24:00 of every wall-clock day from the time-zone rules, the hours whose
    start lies between them summed by cumulative sums. Returns the totals and the number of hours of every day."""
    start = stamp - HOUR
    zi = ZoneInfo(zone)
    a = DAYS.tz_localize(zi).tz_convert("UTC").tz_localize(None)
    b = (DAYS + pd.Timedelta(days=1)).tz_localize(zi).tz_convert("UTC").tz_localize(None)
    i0, i1 = np.asarray((a - start[0]) // HOUR, int), np.asarray((b - start[0]) // HOUR, int)
    assert i0.min() >= 0 and i1.max() <= len(x), "the hourly series does not cover every day of the period"
    assert (i1[:-1] == i0[1:]).all() and set(np.unique(i1 - i0)) <= {23, 24, 25}, "the days do not follow each other hour by hour, or a day has other than 23, 24 or 25 hours"
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, float))])
    return c[i1] - c[i0], i1 - i0


def made_up_hours(stamp, x, key, zone, s, cuts):
    """Every hour that ENDS after the issue time of a cut day c (24:00 of day c on the wall clock, as a UTC time) is made up (x7 + 100): no daily value up to day c may change, the next day must."""
    for c in cuts:
        issue = (DAYS[c] + pd.Timedelta(days=1)).tz_localize(ZoneInfo(zone)).tz_convert("UTC").tz_localize(None)
        x2 = np.where(np.asarray(stamp > issue), x * 7 + 100.0, x)
        s2 = rain_by_wall_clock_day(x2, key)[0].reindex(DAYS).values
        assert np.array_equal(s2[:c + 1], s[:c + 1]), f"rain after the issue time of {DAYS[c].date()} changed a daily value up to that day"
        assert s2[c + 1] != s[c + 1], f"the made-up-hours test could not see a change of the day after {DAYS[c].date()}"


def build_tables():
    """D2: the daily table of every basin with the three rain products, their older and shuffled versions; and the record of the data-side checks and of the shuffle."""
    D2, rec, rng = {}, {}, np.random.default_rng(0)
    for g in ids:
        zone = ZONES[g]
        r = dict(time_zone=zone)
        cols = {}
        for name, (stamp, x) in (("aorc_to_midnight", aorc_hours(g)), ("rain_to_midnight", era5_hours(g))):
            key = day_key(stamp, zone)
            by_day, size = rain_by_wall_clock_day(x, key)
            s = by_day.reindex(DAYS)
            assert s.index.equals(D[g].index) and s.notna().all(), f"{g}, {name}: the daily series must cover all of flood.PERIOD without a gap"
            two, n_h = second_route(stamp, x, zone)
            assert np.array_equal(size.reindex(DAYS).values, n_h), f"{g}, {name}: the two routes count different numbers of hours in a day"
            assert np.allclose(two, s.values, rtol=0, atol=1e-6), f"{g}, {name}: the two routes to the daily total disagree"
            first_change = [int(np.flatnonzero(n_h == k)[0]) for k in (23, 25) if (n_h == k).any()]                    # the first days on which the clocks changed are tested too
            cuts = sorted({*map(int, rng.integers(1, len(DAYS) - 1, 2)), *[c for c in first_change if 0 < c < len(DAYS) - 1]})
            made_up_hours(stamp, x, key, zone, s.values, cuts)
            cols[name] = s.values
            r[name] = dict(n_hours=int(len(x)), first_stamp_utc=str(stamp[0]), last_stamp_utc=str(stamp[-1]), days_with_23_hours=int((n_h == 23).sum()), days_with_25_hours=int((n_h == 25).sum()),
                           mean_mm_per_day=float(s.mean()), max_abs_difference_of_the_two_routes=float(np.abs(two - s.values).max()), made_up_hours_cut_days=[str(DAYS[c].date()) for c in cuts])
        nl = load_basin(g, "nldas").loc[F.PERIOD[0]:F.PERIOD[1], "prcp"].astype(float)             # the NLDAS column of 10_rain_timing.py and 15_rain_to_midnight.py, read the same way
        assert nl.index.equals(D[g].index) and nl.notna().all(), f"{g}: the NLDAS rain does not cover the days of the Daymet table without gaps"
        cols["prcp_nldas"] = nl.values
        r["prcp_nldas"] = dict(mean_mm_per_day=float(nl.mean()))
        d = D[g].assign(**cols)
        for c in NEW2:
            d[c + "_older"] = d[c].shift(1).bfill()             # day t holds the value of day t-1; the first day keeps its own value (as the _older controls of scripts 10, 15, 16)
            assert np.array_equal(d[c + "_older"].values[1:], d[c].values[:-1]) and d[c + "_older"].values[0] == d[c].values[0], f"{g}: the older column of {c} is not the column shifted one day"
        D2[g], rec[g] = d, r
    return D2, rec


def shuffle_columns(D2):
    """The capacity control's columns: within each basin, split part and calendar month ONE permutation of all days of that month in that part, shared by prcp_nldas and rain_to_midnight
    (default_rng(SHUF_SEED), basins in the order of ids, months 1-12). Day i reads the two values measured on day src[i]. Returns the record (counts of values moved to earlier or later days and into
    other split parts; sha256 of the source index)."""
    rng = np.random.default_rng(SHUF_SEED)
    tr_end, val_end = pd.Timestamp(F.SPLIT["train"][1]), pd.Timestamp(F.SPLIT["val"][1])
    tot = dict(n=0, from_a_later_day=0, from_an_earlier_day=0, same_day=0, from_another_split_part=0, test_year_values_in_training_days=0, training_days=0)
    srcs = []
    for g in ids:
        d = D2[g]
        months, n = d.index.month.values, len(d)
        part = np.select([d.index <= tr_end, d.index <= val_end], [0, 1], 2)       # 0 training, 1 validation, 2 test years: the shuffle never crosses a split part
        src = np.empty(n, int)
        for p in (0, 1, 2):
            for m in range(1, 13):
                pos = np.flatnonzero((months == m) & (part == p))
                src[pos] = pos[rng.permutation(len(pos))]
        assert np.array_equal(np.sort(src), np.arange(n)) and (months[src] == months).all() and (part[src] == part).all(), \
            f"{g}: the shuffle is not a permutation within calendar months and split parts"
        for c in NEW2:
            d[c + "_shuf"] = d[c].values[src]
            for m in range(1, 13):
                pos = np.flatnonzero(months == m)
                assert np.array_equal(np.sort(d[c + "_shuf"].values[pos]), np.sort(d[c].values[pos])), f"{g}: the shuffled {c} does not hold the same values in month {m}"
        assert np.array_equal(d["prcp_nldas_shuf"].values, d["prcp_nldas"].values[src]) and np.array_equal(d["rain_to_midnight_shuf"].values, d["rain_to_midnight"].values[src]), \
            f"{g}: the two columns were not moved together"
        tot["n"] += n
        tot["from_a_later_day"] += int((src > np.arange(n)).sum())
        tot["from_an_earlier_day"] += int((src < np.arange(n)).sum())
        tot["same_day"] += int((src == np.arange(n)).sum())
        tot["from_another_split_part"] += int((part[src] != part).sum())
        tot["test_year_values_in_training_days"] += int(((part == 0) & (part[src] == 2)).sum())
        tot["training_days"] += int((part == 0).sum())
        srcs.append(src)
    assert tot["from_another_split_part"] == 0, "the shuffle moved values across split parts"
    return dict(
        what=COLUMN_TEXT["shuf"], random_seed=SHUF_SEED, unit="single days within basin x calendar month x split part (training, validation and test years each shuffled among themselves)",
        built="once, before any training; every training seed of wqME_shuf reads the same shuffled table",
        counts=tot, share_from_a_later_day=tot["from_a_later_day"] / tot["n"], share_from_another_split_part=tot["from_another_split_part"] / tot["n"],
        share_of_training_days_holding_a_test_year_value=tot["test_year_values_in_training_days"] / tot["training_days"],
        source_index_sha256=hashlib.sha256(np.ascontiguousarray(np.stack(srcs)).tobytes()).hexdigest(),
        statement=("The shuffle is a capacity control: its two columns keep each basin's month-by-month values and the pairing of the NLDAS and ERA5 value of a day, but carry no timing information "
                   "about the day they sit on by construction. Shuffling across years inherently places values measured after a day into that day's inputs (from_a_later_day), but only within the "
                   "same split part: no test-year value enters a training or validation day and no training-year value a test day (from_another_split_part = 0, asserted). Nothing else is moved: "
                   "every other input of every day is its own, and the NLDAS and ERA5 values are moved only within the same basin, calendar month and split part. The columns are therefore never a "
                   "forecast input, only a test of whether two more rain-like columns alone raise the score. Changed on 30 October 2026 before the full training (the smoke test had shuffled over "
                   "the whole record) so that the split parts stay clean."))


t_d = time.time()
D2, DATA_SIDE = build_tables()
SHUFFLE = shuffle_columns(D2)
INPUT_COLS = ME_COLS + tuple(c + "_older" for c in NEW2) + tuple(c + "_shuf" for c in NEW2)
for g in ids:
    assert np.isfinite(D2[g][list(tuple(F.FORCING) + INPUT_COLS)].values).all(), f"{g}: an input column has gaps"
INPUT_SHA = {c: hashlib.sha256(np.ascontiguousarray(np.stack([D2[g][c].values.astype("float64") for g in ids])).tobytes()).hexdigest() for c in INPUT_COLS}
print(f"tables: {n_b} basins, {len(DAYS)} days; NLDAS, ERA5 to midnight and AORC to midnight built (two routes agree, made-up hours pass for both hourly products); "
      f"shuffle: {100 * SHUFFLE['share_from_a_later_day']:.1f}% of the shuffled values come from a later day ({time.time() - t_d:.0f} s)", flush=True)


# ---- 1. the column-identity check: the stored networks of scripts 15 and 16 must forecast again from THIS script's columns ------------------------------------
def run_from_file(f):
    """A trained network rebuilt from its file alone: (flood.Run, the saved dict)."""
    import torch
    z = torch.load(f, map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    assert z["state_dict"]["lstm.weight_ih_l0"].shape[1] == n_in and list(cfg.dyn) == z["dyn"], f"{f.name}: the input columns do not fit the weights"
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z


def frames_agree(old_, new):
    """The stored and the new forecasts of one set: numpy.allclose rtol 1e-6, atol 1e-8 (the tolerance of the scripts' reload asserts), the same missing days."""
    cols = ["obs"] + [lk(k) for k in F.LEADS]
    assert set(old_) == set(new) == set(ids) and all(old_[g].index.equals(new[g].index) for g in ids), "the two prediction sets cover different basins or days"
    a, b = np.stack([old_[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    ok = bool(same_nan and np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
    both = ~np.isnan(a) & ~np.isnan(b)
    return dict(n_values_compared=int(a.size), n_missing_in_both=int((np.isnan(a) & np.isnan(b)).sum()), max_abs_difference=float(np.abs(a - b)[both].max()) if both.any() else None,
                bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), same_missing_days=same_nan, allclose=ok, passed=ok)


IDENTITY_SPEC = (("wqa", "16_aorc_rain_to_midnight.py", ("aorc_to_midnight",), "the AORC rain to midnight"),
                 ("wqnh", "15_rain_to_midnight.py", NEW2, "the NLDAS rain and the ERA5 rain to midnight"))


def identity_check():
    """Rebuild stored networks of scripts 16 (lstm_wqa: aorc_to_midnight) and 15 (lstm_wqnh: prcp_nldas + rain_to_midnight) from results/models, run them on THIS script's table and compare with
    their stored forecasts of results/preds. Equal forecasts prove that the rebuilt columns are the ones those networks were trained on. Stops the script (SystemExit) on any failure."""
    out = dict(what="before any training: stored networks of scripts 16 (lstm_wqa, AORC) and 15 (lstm_wqnh, NLDAS + ERA5) rebuilt from results/models, run on this script's input table and compared with their "
                    "stored forecasts in results/preds; equal forecasts prove that this script's prcp_nldas, rain_to_midnight and aorc_to_midnight (and the Daymet columns) are the columns those networks were trained on",
               tolerance="numpy.allclose rtol 1e-6, atol 1e-8, the same missing days (the tolerance of the reload asserts of scripts 15-22 and code/check_reload.py)", checks=[], missing=[])
    for name, script, cols, text in IDENTITY_SPEC:
        have = lambda s: (STORED_MODELS / f"lstm_{name}_s{s}.pt").exists() and real_have(f"lstm_{name}_s{s}")
        seeds = [s for s in IDENTITY_SEEDS if have(s)]
        for s in IDENTITY_SEEDS:
            if s not in seeds:
                out["missing"].append(f"results/models/lstm_{name}_s{s}.pt or results/preds/lstm_{name}_s{s}.npz is missing")
                notes.append(f"Column-identity check: the stored network lstm_{name}_s{s} (or its forecasts) is missing; another stored seed was used where none of {IDENTITY_SEEDS} exists.")
        if not seeds:
            seeds = [s for s in range(5) if have(s)][:1]
        if not seeds:
            raise SystemExit(f"COLUMN-IDENTITY CHECK FAILED: no stored network of lstm_{name} (script {script}) with its stored forecasts exists in results/models and results/preds; "
                             f"the columns of {text} cannot be proved to be the ones trained on. Nothing was trained.")
        for s in seeds:
            f = STORED_MODELS / f"lstm_{name}_s{s}.pt"
            run, z = run_from_file(f)
            assert z["input_set"] == name and z["script"] == script and z["seed"] == s and set(z["ids"]) == set(ids), f"{f.name} is not the network of {name}, seed {s}, of {script}"
            if tuple(z["dyn"]) != tuple(F.FORCING) + tuple(cols):
                raise SystemExit(f"COLUMN-IDENTITY CHECK FAILED: {f.name} reads {z['dyn']}, not {list(F.FORCING) + list(cols)}. Nothing was trained.")
            res = frames_agree(real_preds(f"lstm_{name}_s{s}"), run.predict(D2, z["ids"], B))
            out["checks"].append(dict(set=f"lstm_{name}", seed=s, script=script, model_file=rel(f), columns_proved=list(cols), dyn=list(z["dyn"]), **res))
            if not res["passed"]:
                raise SystemExit(f"COLUMN-IDENTITY CHECK FAILED: {f.name} run on this script's table does not reproduce results/preds/lstm_{name}_s{s}.npz (largest difference {res['max_abs_difference']}, "
                                 f"same missing days {res['same_missing_days']}): the rebuilt columns of {text} are not the ones that network was trained on. Nothing was trained.")
    out["passed"] = True
    out["columns_proved"] = sorted({c for c_ in out["checks"] for c in c_["columns_proved"]} | set(F.FORCING))
    return out


t_i = time.time()
IDENTITY = identity_check()
IDENTITY["seconds"] = round(time.time() - t_i, 1)
print("COLUMN-IDENTITY CHECK: PASS - " + "; ".join(f"{c['set']}_s{c['seed']} max difference {c['max_abs_difference']:.1e}{' (bit for bit)' if c['bit_for_bit'] else ''}" for c in IDENTITY["checks"])
      + f" ({IDENTITY['seconds']:.0f} s)", flush=True)

# ---- 2. the sets --------------------------------------------------------------------------------------------------------------------------------------
SETS = {"wqME": dict(cols=ME_COLS, control=False, role="treatment", what="the rain table + the NLDAS rain + the ERA5 rain to midnight + the AORC rain to midnight, side by side (the treatment)"),
        "wqME_shuf": dict(cols=("prcp_nldas_shuf", "rain_to_midnight_shuf", "aorc_to_midnight"), control=True, role="capacity control",
                          what="as wqME, the NLDAS and ERA5 columns shuffled within basin x calendar month across years (the same column count and distributions, no timing information)"),
        "wqME_older": dict(cols=("prcp_nldas_older", "rain_to_midnight_older", "aorc_to_midnight"), control=True, role="clock control",
                           what="as wqME, the NLDAS and ERA5 columns one day older (is any gain from the issue day's hours?)")}
for v in SETS.values():
    v["dyn"] = tuple(F.FORCING) + v["cols"]                                                       # the daily input columns of each set, in the order the network reads them
    v["n_in"] = len(v["dyn"]) + 1 + len(F.STATIC)                                                 # + the flow channel + the basin properties
STORED = {"wqa": dict(script="16_aorc_rain_to_midnight.py", dyn=tuple(F.FORCING) + ("aorc_to_midnight",), role="base", what="the rain table + the AORC rain to midnight (E1b): the number to beat"),
          "wq": dict(script="04_lstm_all.py", dyn=tuple(F.FORCING), role="shown", what="the rain table alone"),
          "wqn": dict(script="10_rain_timing.py", dyn=None, role="shown", what="the rain table + the NLDAS rain of the issue day"),
          "wqh": dict(script="15_rain_to_midnight.py", dyn=tuple(F.FORCING) + ("rain_to_midnight",), role="shown", what="the rain table + the ERA5 rain to midnight (E1)"),
          "wqnh": dict(script="15_rain_to_midnight.py", dyn=tuple(F.FORCING) + NEW2, role="shown", what="the rain table + the NLDAS rain + the ERA5 rain to midnight (two products, no AORC)")}
for n, v in STORED.items():
    assert real_have(f"lstm_{n}"), f"the stored set lstm_{n} ({v['script']}) was not found in results/preds: this script compares with it"
    meta = real_meta(f"lstm_{n}")
    assert meta.get("config") == CONF_REAL and meta.get("script") == v["script"], f"lstm_{n} is not the set of {v['script']} trained with the settings of results/lstm_all.json"
    assert v["dyn"] is None or "dyn" not in meta or tuple(meta["dyn"]) == v["dyn"], f"lstm_{n} reads {meta.get('dyn')}, not {v['dyn']}"
    v["dyn"] = tuple(meta["dyn"]) if "dyn" in meta else v["dyn"]
    v["seeds"] = [s for s in range(5) if real_have(f"lstm_{n}_s{s}")]
    assert len(v["seeds"]) >= 3, f"lstm_{n}: fewer than three single-seed sets were found"
assert STORED["wqa"]["seeds"] == [0, 1, 2, 3, 4], "ROADMAP.md compares with the five stored seeds of lstm_wqa (script 16): not all five are in results/preds"
SIMPLE_NEW, SIMPLE_STORED = ("base_linreg_wqME", "base_rf_wqME"), ("base_linreg_wqa", "base_rf_wqa")
PARTNER = dict(zip(SIMPLE_NEW, SIMPLE_STORED))
LSTM_SHOWN = list(SETS) + list(STORED)
SHOWN = LSTM_SHOWN + list(SIMPLE_NEW) + list(SIMPLE_STORED)

# ---- the simple models given the three columns: rule block 4, the recipe of 02_baselines.py as script 16 runs it for one extra column ---------------------------
BASE_J = json.loads((F.RES / "baselines.json").read_text())["models"]                    # the settings chosen there on the validation years for the input set wq (the real file, also in a smoke test)
RIDGE_ALPHA, RF_SET = BASE_J["base_linreg_wq"]["settings"]["ridge_alpha"], BASE_J["base_rf_wq"]["settings"]
assert BASE_J["base_linreg_wq"]["settings"]["standardised"] and RF_SET["max_features"] == "one third" and RF_SET["random_state"] == 0, "the recipe of 02_baselines.py has changed: read it again"
RF_TREES, RF_JOBS = (8, 1) if F.QUICK else (RF_SET["n_trees"], 4)                        # a smoke test fits tiny forests, as 02_baselines.py and script 16 do; 4 jobs as there
MAKE = {"linreg": lambda: make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
        "rf": lambda: RandomForestRegressor(RF_TREES, min_samples_leaf=RF_SET["min_samples_leaf"], max_features=1 / 3, n_jobs=RF_JOBS, random_state=RF_SET["random_state"])}
KIND = {"base_linreg_wqME": "linreg", "base_rf_wqME": "rf", "base_linreg_wqa": "linreg", "base_rf_wqa": "rf"}
SETTINGS = {"linreg": dict(ridge_alpha=RIDGE_ALPHA, standardised=True), "rf": dict(n_trees=RF_TREES, min_samples_leaf=RF_SET["min_samples_leaf"], max_features="one third", random_state=RF_SET["random_state"])}


def feature_table_with(d, cols=()):
    """The features of features.feature_table(d, flow=True) for the input set wq and, for every column of cols in turn, the same rain features (the rain of the last three days, the sums over 3 to 90
    days, the two antecedent-rain indices) computed on that series, prefixed '<column>__': script 16's feature_table_with for one column, applied to each column. A row without the full history is NaN in
    every column."""
    X = feature_table(d, True, 0)
    if not cols:
        return X
    parts = [X]
    for col in cols:
        Y = feature_table(d.assign(prcp=d[col]), False, 0)
        parts.append(Y[[c for c in Y.columns if c.startswith(("rain_", "wetness_"))]].add_prefix(f"{col}__"))
    X2 = pd.concat(parts, axis=1)
    return X2.where(X2.notna().all(axis=1))


def simple_design(cols):
    """What 02_baselines.design() builds for one input set, for wq + the series cols: per basin and lead the feature rows, the complete rows, the rows used for fitting (target day in the training
    years) and the flow to predict."""
    T = {}
    for g in ids:
        d = D2[g]
        X = feature_table_with(d, cols)
        ok = X.notna().all(axis=1).values
        for k in F.LEADS:
            T[g, k] = X.values, ok, ok & F.part_mask(d.index, k, "train"), d.q_mm.shift(-k).values
    return T


def forecast_simple(make, T):
    """One model per basin and lead, fitted on the training years; forecasts every issue day that has features (02_baselines.forecast() as script 16 copies it)."""
    P = {}
    for g in ids:
        d, out = D2[g], {}
        for k in F.LEADS:
            X, ok, fit, y = T[g, k]
            m = make().fit(X[fit], y[fit])
            out[k] = np.full(len(d), np.nan)
            out[k][ok] = np.clip(m.predict(X[ok]), 0, None)                                     # a river cannot have negative flow
        P[g] = F.frame_from_issue(d, out)
    return P


def features_are_causal(cols=ME_COLS, cut=5000):
    """features.py's own check on the real tables with the three columns: EVERY column of the daily table changed after a cut day; no feature row up to the cut may change, the next one must."""
    for g in ids:
        d = D2[g]
        bad = d.copy()
        bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
        A, A2 = feature_table_with(d, cols), feature_table_with(bad, cols)
        assert A.columns.equals(A2.columns) and A.iloc[:cut + 1].equals(A2.iloc[:cut + 1]), f"{g}: a feature reads data after the issue day"
        assert not np.allclose(A.iloc[cut + 1].values, A2.iloc[cut + 1].values), f"{g}: the check of the features has no power"
        assert (A.isna().all(axis=1) | A.notna().all(axis=1)).all(), f"{g}: a feature row is partly missing"
    return dict(what=f"features.py's check on the real tables: every column of the daily table (the rain table, the three rain products, the flow ...) was replaced after day number {cut} of the record; "
                     "the feature rows up to the cut did not change and the next row did, in every basin", passed=True, n_basins=n_b, n_features=int(feature_table_with(D2[ids[0]], cols).shape[1]))


def recipe_check(with_forest):
    """With aorc_to_midnight as the only extra column this code must be script 16's: it must reproduce the stored base_linreg_wqa and (with_forest: outside a smoke test, when the forest is fitted in
    this pass) base_rf_wqa of results/preds to below 1e-9 mm/day, the same missing days - else the script stops before any training. A forest with its random_state is repeatable on one machine and
    library version; base_rf_wqa was fitted on the Mac and reproduced on this PC on 2026-10-30 (largest difference 1.4e-14 mm/day, a scratch check while this script was written). The scores of the
    refit and of the stored set are recorded beside each other."""
    T0, out = simple_design(("aorc_to_midnight",)), dict(what="the recipe of script 16 (features.feature_table for wq + the rain features of aorc_to_midnight, one model per basin and lead fitted on the training "
                                                               "years, the settings of results/baselines.json for wq), rebuilt here and compared with its stored forecasts (asserted below 1e-9 mm/day)", checks={})
    assert list(feature_table_with(D2[ids[0]], ("aorc_to_midnight",)).columns) == real_meta("base_linreg_wqa")["features"], "the feature columns differ from those script 16 stored for base_linreg_wqa"
    for n in SIMPLE_STORED:
        if KIND[n] == "rf" and not with_forest:
            out["checks"][n] = "not run in this pass (" + ("tiny forests in a smoke test" if F.QUICK else "no forest is fitted in this pass; an earlier pass's record is carried below when there is one") + ")"
            continue
        t0 = time.perf_counter()
        P, S = forecast_simple(MAKE[KIND[n]], T0), real_preds(n)
        same_nan = all(np.array_equal(np.isnan(P[g][lk(k)].values), np.isnan(S[g][lk(k)].values)) for g in ids for k in F.LEADS)
        diff = float(max(np.nanmax(np.abs(P[g][lk(k)].values - S[g][lk(k)].values)) for g in ids for k in F.LEADS))
        e_new, e_old = F.evaluate(P, thr)["summary"]["l1"], F.evaluate(S, thr)["summary"]["l1"]
        rec = dict(same_missing_days=same_nan, max_abs_difference=diff, reproduced=bool(same_nan and diff < 1e-9), seconds=round(time.perf_counter() - t0, 1),
                   test_lead_1_refit_here=dict(onset_warned=e_new["onset"]["pod"], onset_false=e_new["onset"]["far"], nse_median=e_new["nse_median"]),
                   test_lead_1_stored=dict(onset_warned=e_old["onset"]["pod"], onset_false=e_old["onset"]["far"], nse_median=e_old["nse_median"]))
        out["checks"][n] = rec
        assert same_nan, f"{n}: this code forecasts other days than script 16 did: the recipe is not script 16's"
        assert rec["reproduced"], (f"{n} (script 16) is not reproduced by this code (largest difference {diff:.3g} mm/day; onsets warned 1 day ahead {p1(e_new['onset']['pod'])} here against "
                                   f"{p1(e_old['onset']['pod'])} stored): the simple-model recipe is not script 16's on this machine. Nothing was trained.")
    return out


# ---- 3. training (or reading) every set, SEEDS seeds each, and the two simple models -------------------------------------------------------------------
def save_model(run, name, seed, v):
    """results/models/lstm_<set>_s<seed>.pt: a plain dict (tensors, numbers, lists and text) that torch.load(weights_only=True) reads, enough to forecast again (the form of scripts 16 and 21)."""
    import torch
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save(dict(state_dict={k: t.detach().cpu().clone() for k, t in run.net.state_dict().items()}, cfg=asdict(run.cfg),
                    sc={k: (x.tolist() if isinstance(x, np.ndarray) else float(x)) for k, x in run.sc.items()}, ids=list(run.ids), dyn=list(run.cfg.dyn),
                    split=dict(run.split), best_val_nse=float(run.best_val_nse), n_params=int(run.n_params), input_set=name, script=SCRIPT, seed=seed,
                    history=run.history, seconds=float(run.seconds), cpu_seconds=float(run.cpu_seconds), updates_run=int(run.updates_run), best_update=int(run.best_update),
                    torch_version=str(torch.__version__), quick_mode=F.QUICK, extra_columns=list(v["cols"]), shuffle_seed=SHUF_SEED, input_sha256={c: INPUT_SHA[c] for c in v["cols"]},
                    how_to_reload="cfg = flood.Cfg(**cfg); net = flood.LSTMNet(n_in, cfg); net.load_state_dict(state_dict); sc lists back to numpy arrays; "
                                  "flood.Run(cfg, sc, net, ids, history, seconds, cpu_seconds, updates_run, best_update, best_val_nse, n_params, split).predict(D, ids, static_df); "
                                  f"D needs the columns in dyn: FORCING from flood.load_data(); prcp_nldas (camels.load_basin(g, 'nldas').prcp), rain_to_midnight (hourly ERA5) and aorc_to_midnight "
                                  f"(hourly AORC) summed over the local wall-clock day as {SCRIPT} builds them; the _older columns shifted one day later (first day kept); the _shuf columns by {SCRIPT}'s "
                                  f"shuffle_columns() with seed {SHUF_SEED} (input_sha256 holds the hash of every column this network read)"), tmp)
    tmp.replace(f)
    return f


def reload_and_forecast(name, seed):
    """Rebuild a network of this run from its file alone, forecast again on the SAME tables and compare with the stored forecasts of the set (same machine: rtol 1e-6)."""
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    run, z = run_from_file(f)
    assert z["input_set"] == name and z["script"] == SCRIPT and z["seed"] == seed, f"{f.name} is not the network of {name}, seed {seed}"
    res = frames_agree(own_preds(f"lstm_{name}_s{seed}"), run.predict(D2, ids, B))
    assert res["passed"], f"{f.name}: the reloaded network does not reproduce the stored forecasts"
    return dict(set=name, seed=seed, model_file=rel(f), size_kb=round(f.stat().st_size / 1024, 1), n_in=int(len(run.cfg.dyn) + 1 + len(run.cfg.static)), n_params=int(z["n_params"]), **res)


def record_of(name, seed):
    """The training record of a saved network: from the results file of the earlier run when it holds that set and seed, otherwise from the network's own file, which holds the same numbers."""
    for r in (old or {}).get("models", {}).get(name, {}).get("per_seed", []):
        if r["seed"] == seed:
            return r
    import torch
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    assert f.exists(), f"lstm_{name}_s{seed}: neither the results file nor a network file holds its training record"
    z = torch.load(f, map_location="cpu", weights_only=True)
    return dict(seed=seed, **{k: z[k] for k in COST}, model_file=rel(f), history=z["history"])


have_seed = lambda name, seed: own_have(f"lstm_{name}_s{seed}") and (MODELS / f"lstm_{name}_s{seed}.pt").exists()
if LOAD:
    assert RESULTS_FILE.exists(), "--rescore and --extend need the results file of a training run: run the script without them first"
    old = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    conf = old["config"]
    if not F.QUICK:
        assert conf == CONF_REAL, "the settings of the existing sets are not those of results/lstm_all.json: new sets would not be trained with the same settings"
    old_sha = (old.get("inputs") or {}).get("sha256_of_input_columns") or {}
    changed = [c for c in INPUT_SHA if c in old_sha and old_sha[c] != INPUT_SHA[c]]
    assert not changed, f"the input columns {changed} differ from those of the earlier pass (inputs.sha256_of_input_columns): its networks were trained on other data"
else:
    old = None
    conf = dict(CONF_REAL, updates=40, eval_every=20) if F.QUICK else dict(CONF_REAL)
missing_sets = [(n, s) for n in SETS for s in SEEDS if not have_seed(n, s)]
missing_simple = [n for n in SIMPLE_NEW if not (LOAD and own_have(n))]
if RESCORE:
    assert not missing_sets and not missing_simple, "--rescore trains nothing, and these are missing: " + ", ".join([f"lstm_{n}_s{s}" for n, s in missing_sets] + missing_simple) + ": run --extend first"
FEATURES = list(feature_table_with(D2[ids[0]], ME_COLS).columns)
t_r = time.time()
RECIPE = recipe_check(with_forest=("base_rf_wqME" in missing_simple and not F.QUICK))          # before any training: a recipe that is not script 16's stops the script before it costs anything
FEAT_CAUSAL = features_are_causal()
if LOAD and not isinstance(RECIPE["checks"]["base_rf_wqa"], dict):
    prev = (old.get("yardstick") or {}).get("recipe_check") or {}
    for key in ("base_rf_wqa", "base_rf_wqa_in_an_earlier_pass"):
        if isinstance((prev.get("checks") or {}).get(key), dict):
            RECIPE["checks"]["base_rf_wqa_in_an_earlier_pass"] = prev["checks"][key]
            break
recipe_text = lambda r: (("reproduced" if r["reproduced"] else "NOT bit for bit") + f", max difference {r['max_abs_difference']:.1e}") if isinstance(r, dict) else r
print("simple-model recipe: " + "; ".join(f"{n} {recipe_text(r)}" for n, r in RECIPE["checks"].items()) + f"; features causal ({time.time() - t_r:.0f} s)", flush=True)
need_cpu = bool(missing_sets or missing_simple)
t_ask = time.time()
runs, ENS, costs, files, HIST, trained_now = {}, {}, {}, {}, {}, {}
SIMPLE_P, SIMPLE_SEC, simple_now = {}, {}, []
with F.cpu_lock(SCRIPT) if need_cpu else contextlib.nullcontext():
    waited = time.time() - t_ask
    for name, v in SETS.items():
        runs[name], costs[name], files[name], HIST[name], trained_now[name] = [], {}, {}, {}, []
        for seed in SEEDS:
            if LOAD and have_seed(name, seed):                           # a saved set is read, never made again
                p, meta, rec = own_preds(f"lstm_{name}_s{seed}"), json.loads(str(np.load(F.PRED / f"lstm_{name}_s{seed}.npz", allow_pickle=False)["meta"])), record_of(name, seed)
                assert meta["config"] == conf and meta["seed"] == rec["seed"] == seed and meta["script"] == SCRIPT and meta["dyn"] == list(v["dyn"]), f"lstm_{name}_s{seed} is not from the training run described in {OUT}.json"
                cost, mf, hist = {k: rec[k] for k in COST}, rec.get("model_file"), rec.get("history")
            else:
                cfg = F.Cfg(dyn=v["dyn"], use_flow=True, static=tuple(F.STATIC), seed=seed, **conf)
                print(f"training lstm_{name} seed {seed}: {len(cfg.dyn)} daily inputs + flow + {len(cfg.static)} basin properties = {v['n_in']} inputs", flush=True)
                run = F.fit(cfg, D2, ids, static_df=B, log=lambda s, n_=name, s_=seed: print(f"  [lstm_{n_} s{s_}]{s}", flush=True))
                assert run.cfg.dyn == v["dyn"] and run.n_params == 4 * conf["hidden"] * (v["n_in"] + conf["hidden"] + 2) + (conf["hidden"] + 1) * len(F.LEADS), "the network has not the expected size"
                p, cost = run.predict(D2, static_df=B), run.summary()
                mf = rel(save_model(run, name, seed, v))
                F.save_preds(f"lstm_{name}_s{seed}", p, dict(script=SCRIPT, model="lstm", inputs=name, seed=seed, config=conf, dyn=list(cfg.dyn), extra_inputs=list(v["cols"]), control=v["control"], role=v["role"],
                                                             model_file=mf, shuffle_seed=SHUF_SEED, what="one LSTM for all 12 basins, a single seed (E7)"))
                trained_now[name].append(seed)
                hist = run.history
            e = F.evaluate(p, thr)
            runs[name].append(p)
            costs[name][seed], files[name][seed], HIST[name][seed] = cost, mf, hist
            print(f"lstm_{name} seed {seed} | best at update {cost['best_update']} of {cost['updates_run']}, {cost['seconds']:.0f} s | validation NSE {cost['best_val_nse']:.3f} | test NSE median "
                  + " ".join(f"{e['summary'][lk(k)]['nse_median']:.3f}" for k in F.LEADS) + f" | onsets warned 1 day {100 * e['summary']['l1']['onset']['pod']:.1f}%"
                  + ("" if seed in trained_now[name] else " | read, not trained again"), flush=True)
        ens = F.ensemble(runs[name])
        if trained_now[name]:                                            # a rescore reads the prediction sets and leaves them untouched
            F.save_preds(f"lstm_{name}", ens, dict(script=SCRIPT, model="lstm", inputs=name, seeds=SEEDS, config=conf, dyn=list(v["dyn"]), extra_inputs=list(v["cols"]), control=v["control"], role=v["role"],
                                                   shuffle_seed=SHUF_SEED, what="one LSTM for all 12 basins, day-by-day average of the seeds (E7)"))
        ENS[name] = ens
    T_ME = simple_design(ME_COLS) if missing_simple else None
    for n in SIMPLE_NEW:
        if n in missing_simple:
            t0 = time.perf_counter()
            SIMPLE_P[n] = forecast_simple(MAKE[KIND[n]], T_ME)
            SIMPLE_SEC[n] = round(time.perf_counter() - t0, 1)
            F.save_preds(n, SIMPLE_P[n], dict(script=SCRIPT, model=KIND[n], inputs="wqME", settings=SETTINGS[KIND[n]], features=FEATURES, extra_inputs=list(ME_COLS), fit_seconds=SIMPLE_SEC[n], quick_mode=F.QUICK,
                                              what="one model per basin and lead on the features of the input set wq plus the same rain features of each of the three rain products (the recipe of 02_baselines.py "
                                                   "as script 16 runs it for one column; settings as chosen there on the validation years for wq)"))
            simple_now.append(n)
            print(f"{n}: one model per basin and lead, fitted on the training years in {SIMPLE_SEC[n]:.0f} s", flush=True)
        else:
            SIMPLE_P[n] = own_preds(n)
            meta = json.loads(str(np.load(F.PRED / f"{n}.npz", allow_pickle=False)["meta"]))
            assert meta["script"] == SCRIPT and meta["features"] == FEATURES, f"{n} is not from the run described in {OUT}.json"
            SIMPLE_SEC[n] = meta.get("fit_seconds")
trained_pairs = [(n, s) for n in SETS for s in trained_now[n]]
# the proof that a saved network forecasts again: one seed per set, rebuilt from its file alone (networks trained on this machine)
reload_check = dict(what="one trained network per set (the first seed) rebuilt from its .pt file alone (flood.LSTMNet + the saved weights, settings and scalers), run on the same tables, and compared with the forecasts stored by save_preds",
                    tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal; same machine", per_set={n: reload_and_forecast(n, SEEDS[0]) for n in SETS})
print("reload check passed: " + ", ".join(f"{n} (max difference {r['max_abs_difference']:.1e}{', bit for bit' if r['bit_for_bit'] else ''})" for n, r in reload_check["per_set"].items()), flush=True)


# ---- 4. causality: the data side (section 0), the engine (flood._assert_causal), the saved networks end to end (made-up values), the simple-model features ---------
def made_up_values_test(name, seed, n_cuts=3):
    """The made-up-values test on a saved network (scripts 18 and 21): replace the set's three rain columns (and, in a second round, every input column and the flow) of all days AFTER a cut day c by
    random values, predict again and compare. A forecast issued on or before day c must not change (stored at the target day t+k: array_equal on every lead up to c+k), and the ones issued after c must
    change (or the test proves nothing)."""
    v = SETS[name]
    run_, _ = run_from_file(MODELS / f"lstm_{name}_s{seed}.pt")
    idx = D2[ids[0]].index
    a, b = F.SPLIT["test"]
    cand = np.flatnonzero((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))[200:-200]
    rng = np.random.default_rng(7)
    cuts = sorted(int(c) for c in rng.choice(cand, n_cuts, replace=False))
    base = run_.predict(D2, ids, B)
    out = dict(set=name, seed=seed, cut_days=[str(idx[c].date()) for c in cuts], rounds={})
    for what, cols in (("the three rain columns of the set only", list(v["cols"])), ("every input column and the observed flow", sorted(set(run_.cfg.dyn) | {"q_mm"}))):
        n_same, n_moved = 0, 0
        for c in cuts:
            alt = {g: t.copy() for g, t in D2.items()}
            for g in ids:
                for col in cols:
                    x = alt[g][col].values.copy()
                    x[c + 1:] = rng.uniform(0.0, 120.0, len(x) - c - 1)
                    alt[g][col] = x
            new = run_.predict(alt, ids, B)
            for g in ids:
                for k in F.LEADS:
                    assert np.array_equal(base[g][lk(k)].values[:c + k + 1], new[g][lk(k)].values[:c + k + 1], equal_nan=True), \
                        f"{name} {g}: changing {what} after {idx[c].date()} changed a forecast issued on or before that day (lead {k})"
                    n_same += 1
            moved = [not np.allclose(base[g][lk(1)].values[c + 2:c + 12], new[g][lk(1)].values[c + 2:c + 12], rtol=0, atol=1e-9) for g in ids]
            assert any(moved), f"{name}: the test could not see a change after the cut (it proves nothing)"
            n_moved += int(sum(moved))
        out["rounds"][what] = dict(forecasts_compared_equal=n_same, basins_whose_later_forecasts_moved_summed_over_the_cuts=n_moved, passed=True)
    return out


t_c = time.time()
causality = dict(
    what="(1) data side: both hourly products summed into the wall-clock day by two routes that agree, and the made-up-hours test (every hour ending after 24:00 of a cut day changed: no daily value up to the "
         "cut moved, the next day did), per basin, on random days and the first days on which the clocks changed (inputs.data_side); the NLDAS day ends before the issue time by the reading of 10_rain_timing.py "
         "(not proved again here); (2) engine: flood.fit runs flood._assert_causal on every network before training (a trained network exists only if it passed); (3) end to end: the made-up-values test on one "
         "SAVED network per set (first seed): the set's three rain columns (and in a second round every input column and the flow) of all days after a cut day were replaced by random values (0 to 120 per day); "
         "no forecast issued on or before the cut day changed (exactly equal, all three leads, all 12 basins) and the forecasts issued after it did; (4) the simple models: features.py's check with the three "
         "columns; (5) the shuffled columns of the capacity control move values between the years of the record (inputs.shuffle.statement): they are no forecast input",
    engine_check="flood._assert_causal at the start of every flood.fit (asserted: a trained network exists only if it passed)",
    made_up_values={n: made_up_values_test(n, SEEDS[0]) for n in SETS}, simple_model_features=FEAT_CAUSAL)
causality["seconds"] = round(time.time() - t_c, 1)
print("made-up-values test passed on " + ", ".join(causality["made_up_values"]) + f" ({causality['seconds']:.0f} s)", flush=True)

# ---- 5. every set scored the same way (the stored sets rescored by this code, never copied) ------------------------------------------------------------
PRED = {**ENS, **{n: real_preds(f"lstm_{n}") for n in STORED}, **SIMPLE_P, **{n: real_preds(n) for n in SIMPLE_STORED}}
SEEDP = {**runs, **{n: [real_preds(f"lstm_{n}_s{s}") for s in STORED[n]["seeds"]] for n in STORED}}
SEED_LABELS = {**{n: list(SEEDS) for n in SETS}, **{n: list(STORED[n]["seeds"]) for n in STORED}}
for n in SIMPLE_STORED:
    m_ = real_meta(n)
    assert m_.get("script") == "16_aorc_rain_to_midnight.py" and m_.get("settings") == SETTINGS[KIND[n]] | ({"n_trees": RF_SET["n_trees"]} if KIND[n] == "rf" else {}), f"{n} is not script 16's model with the settings of results/baselines.json"
EV = {n: dict(test=F.evaluate(PRED[n], thr), val=F.evaluate(PRED[n], thr, "val"), seeds=[F.evaluate(p, thr) for p in SEEDP.get(n, [])]) for n in SHOWN}
assert all(EV[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in SHOWN for part in ("test", "val") for k in F.LEADS), "days without a forecast: the models would not be compared on identical days"
assert all(np.allclose(PRED[n][g].obs.values, D[g].q_mm.values) and PRED[n][g].index.equals(D[g].index) for n in SHOWN for g in ids), "the observed flow or the days differ between the sets"

med = lambda E, k: E["summary"][lk(k)]["nse_median"]                                                 # median NSE over the basins of one evaluation, at lead k
ons = lambda E, k, c="pod": E["summary"][lk(k)]["onset"][c]                                          # onset warnings pooled over the basins
nse_of = lambda E, k: np.array([E["basins"][g][lk(k)]["nse"] for g in ids])
row = lambda E, k: dict(nse_median=med(E, k), nse_mean=E["summary"][lk(k)]["nse_mean"], onset=CUT(E["summary"][lk(k)]["onset"]), all=CUT(E["summary"][lk(k)]["all"]))


def paired(a, b):
    """Two per-basin score arrays (first minus second): in how many basins the first is higher, the median difference, Wilcoxon p (as in 04_lstm_all.py)."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:                                           # all differences zero
        p = float("nan")
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=int(len(d)), median_nse_difference=float(np.median(d)), wilcoxon_p=p)


def seed_range(n, k):
    """Lowest and highest single-seed value of a set at one lead, for the scores the rule and the figure use."""
    S = [e["summary"][lk(k)] for e in EV[n]["seeds"]]
    return dict(n_seeds=len(S), nse_median=lo_hi([s["nse_median"] for s in S]), **{f"{kind}_{c}": lo_hi([s[kind][c] for s in S]) for kind in ("onset", "all") for c in ("pod", "far", "csi")},
                by_seed=dict(seeds=SEED_LABELS[n], nse_median=[s["nse_median"] for s in S], **{f"{kind}_{c}": [s[kind][c] for s in S] for kind in ("onset", "all") for c in ("pod", "far", "csi")}))


summary_test = {n: {lk(k): row(EV[n]["test"], k) for k in F.LEADS} for n in SHOWN}
summary_val = {n: {lk(k): row(EV[n]["val"], k) for k in F.LEADS} for n in SHOWN}
ranges = {n: {lk(k): seed_range(n, k) for k in F.LEADS} for n in LSTM_SHOWN}
per_basin = {"how_to_read": "test years, one value per basin; the seed average of each LSTM set, the simple models as they are (flood.evaluate); gauge ids are text",
             "basins": {g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g])) for g in ids},
             "nse": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["nse"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "onset_warned": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "onset_false": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["far"] for g in ids} for k in F.LEADS} for n in SHOWN}}
models = {n: dict(saved_as=n if n.startswith("base_") else f"lstm_{n}", n_seeds_scored=len(EV[n]["seeds"]), seeds=SEED_LABELS.get(n, []), stored=n in STORED or n in SIMPLE_STORED, test=EV[n]["test"]) for n in SHOWN}
for n in SETS:
    models[n]["per_seed"] = [dict(seed=s, **{c: costs[n][s][c] for c in COST}, model_file=files[n][s], history=HIST[n][s], test_nse_median=[med(e, k) for k in F.LEADS],
                                  test_onset_warned=[ons(e, k) for k in F.LEADS], test_onset_false=[ons(e, k, "far") for k in F.LEADS], test_all_days_warned=[e["summary"][lk(k)]["all"]["pod"] for k in F.LEADS])
                             for s, e in zip(SEEDS, EV[n]["seeds"])]
training = [dict(set=n, seed=s, saved_forecasts=f"lstm_{n}_s{s}", model_file=files[n][s], **{c: costs[n][s][c] for c in COST}) for n in SETS for s in SEEDS]


def rule(T, C, k):
    """Set T against set C at lead k (LSTM sets). NSE: every single seed of T above every single seed of C, higher in most basins, Wilcoxon p below P_CUT - all three make a clear gain.
    Onsets warned: pooled over the basins, so only the seeds can be compared (the rule of scripts 10, 16, 18 and 21)."""
    ET, EC = EV[T]["test"], EV[C]["test"]
    pr, n_low = paired(nse_of(ET, k), nse_of(EC, k)), int((nse_of(ET, k) < nse_of(EC, k)).sum())
    sT, sC = ranges[T][lk(k)], ranges[C][lk(k)]
    up, down = above(sT["nse_median"], sC["nse_median"]), above(sC["nse_median"], sT["nse_median"])
    small_p = bool(pr["wilcoxon_p"] < P_CUT)
    gain, loss = up and pr["n_basins_higher"] > n_b / 2 and small_p, down and n_low > n_b / 2 and small_p
    o_up, o_down = above(sT["onset_pod"], sC["onset_pod"]), above(sC["onset_pod"], sT["onset_pod"])
    return dict(against=f"lstm_{C}", lead=k,
                nse=dict(median_with=med(ET, k), median_without=med(EC, k), gain=med(ET, k) - med(EC, k), seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                         single_seeds_all_above=up, single_seeds_all_below=down, n_basins_higher=pr["n_basins_higher"], n_basins_lower=n_low, n_basins=n_b,
                         higher_in_most_basins=bool(pr["n_basins_higher"] > n_b / 2), median_basin_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"],
                         p_below_cut=small_p, clear_gain=bool(gain), clear_loss=bool(loss), reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                onset=dict(warned_with=ons(ET, k), warned_without=ons(EC, k), gain=ons(ET, k) - ons(EC, k), seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"],
                           false_with=ons(ET, k, "far"), false_without=ons(EC, k, "far"), single_seeds_all_above=o_up, single_seeds_all_below=o_down,
                           clear_gain=bool(o_up), clear_loss=bool(o_down), reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"))


a_test, b_test = F.SPLIT["test"]


def onset_flags(P, k):
    """Which onsets were warned: one True / False per onset of the test years (a target day above the high-flow level while the river was at or below it when the forecast was issued, as flood.flood_scores counts them),
    basin after basin in the order of ids; the same days for every set (coverage 1, asserted above)."""
    out = []
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(k) <= thr[g]).loc[a_test:b_test]
        q = p.loc[a_test:b_test]
        ok = (q.obs.notna() & q[lk(k)].notna()).values
        o, s, w = (q.obs > thr[g]).values[ok], (q[lk(k)] > thr[g]).values[ok], was_low.values[ok]
        out.append(s[o & w])
    return np.concatenate(out)


def overlap(A, C, k=1):
    """Do two sets warn of the same onsets? Counts of the onsets warned by both, by only one, by neither; the exact sign-test p of 'only A' against 'only C' (a description: the onsets of one storm are not independent)."""
    fa, fc = onset_flags(PRED[A], k), onset_flags(PRED[C], k)
    ta, tc = EV[A]["test"]["summary"][lk(k)]["onset"], EV[C]["test"]["summary"][lk(k)]["onset"]
    assert len(fa) == len(fc) == ta["hits"] + ta["misses"] and int(fa.sum()) == ta["hits"] and int(fc.sum()) == tc["hits"], "the onsets counted here are not those of flood.evaluate"
    both, only_a, only_c = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    name = lambda x: x if x.startswith("base_") else f"lstm_{x}"
    return dict(first=name(A), second=name(C), lead=k, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=only_a, warned_only_by_second=only_c,
                warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=both + only_a + only_c, sign_test_p=float(binomtest(only_a, only_a + only_c, 0.5).pvalue) if only_a + only_c else float("nan"))


def paired_seeds(sT, sC):
    """The seeds the seed-by-seed difference lists cover: zip() pairs the two sets' seed lists in order and stops at the shorter one."""
    pairs = list(zip(sT["by_seed"]["seeds"], sC["by_seed"]["seeds"]))
    assert all(a == b for a, b in pairs), "the seed-by-seed differences would pair unequal seed numbers"
    return [a for a, _ in pairs]


def pair_reading(T, C):
    """LSTM set T minus LSTM set C at every lead: onsets warned (pooled; hits, seed ranges, seed by seed over the seeds both sets have - seeds_paired) and the median NSE with the checks of section 4.2."""
    out = {}
    for k in F.LEADS:
        r = rule(T, C, k)
        ET, EC, sT, sC = EV[T]["test"], EV[C]["test"], ranges[T][lk(k)], ranges[C][lk(k)]
        tT, tC = ET["summary"][lk(k)]["onset"], EC["summary"][lk(k)]["onset"]
        out[lk(k)] = dict(with_set=f"lstm_{T}", without_set=f"lstm_{C}",
                          onsets=dict(warned_with=tT["pod"], warned_without=tC["pod"], difference_points=100 * (tT["pod"] - tC["pod"]), hits_with=tT["hits"], hits_without=tC["hits"], onsets_in_the_test_years=tT["hits"] + tT["misses"],
                                      hits_difference=tT["hits"] - tC["hits"], seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"], every_seed_above_every_seed=r["onset"]["single_seeds_all_above"],
                                      every_seed_below_every_seed=r["onset"]["single_seeds_all_below"], seeds_paired=paired_seeds(sT, sC), difference_by_seed_points=[100 * (x - y) for x, y in zip(sT["by_seed"]["onset_pod"], sC["by_seed"]["onset_pod"])],
                                      false_warnings_with=tT["far"], false_warnings_without=tC["far"], reading=r["onset"]["reading"]),
                          nse=dict(median_with=r["nse"]["median_with"], median_without=r["nse"]["median_without"], gain=r["nse"]["gain"], seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                                   every_seed_above_every_seed=r["nse"]["single_seeds_all_above"], seeds_paired=paired_seeds(sT, sC), difference_by_seed=[x - y for x, y in zip(sT["by_seed"]["nse_median"], sC["by_seed"]["nse_median"])],
                                   n_basins_higher=r["nse"]["n_basins_higher"], n_basins=n_b, median_basin_difference=r["nse"]["median_basin_difference"], wilcoxon_p=r["nse"]["wilcoxon_p"], reading=r["nse"]["reading"]))
    return out


PAIRS = [("several_products_against_aorc_alone", "wqME", "wqa", "the rule's comparison: lstm_wqME minus the stored lstm_wqa (the AORC rain to midnight beside the rain table; script 16)"),
         ("several_products_against_the_capacity_control", "wqME", "wqME_shuf", "lstm_wqME minus lstm_wqME_shuf (the NLDAS and ERA5 columns shuffled within basin and calendar month): what their timing adds"),
         ("several_products_against_the_clock_control", "wqME", "wqME_older", "lstm_wqME minus lstm_wqME_older (the NLDAS and ERA5 columns one day older): what the issue day's NLDAS and ERA5 values add"),
         ("capacity_control_against_aorc_alone", "wqME_shuf", "wqa", "is the capacity control flat? lstm_wqME_shuf minus lstm_wqa: two more rain-like columns with no timing information"),
         ("clock_control_against_aorc_alone", "wqME_older", "wqa", "is the clock control flat? lstm_wqME_older minus lstm_wqa: the NLDAS and ERA5 columns of the day before"),
         ("several_products_against_nldas_and_era5", "wqME", "wqnh", "lstm_wqME minus the stored lstm_wqnh (NLDAS + ERA5 without AORC; script 15): what AORC adds beside the other two"),
         ("several_products_against_the_rain_table", "wqME", "wq", "lstm_wqME minus the stored lstm_wq (the rain table alone; 04_lstm_all.py)")]
READING_HOW = ("by_lead.l<k>: set T (with_set) minus set C (without_set) on the test years at trigger 1.0, seed-averaged forecasts; seed_range_* and every_seed_* use all seeds of each set; difference_by_seed_points and "
               "difference_by_seed pair the same seed number of both sets in order and cover only the seeds both sets have (seeds_paired); overlap_of_warned_onsets: the onsets warned by both sets, by only one, by neither")
readings = {key: dict(what=what, how_to_read=READING_HOW, by_lead=pair_reading(T, C), overlap_of_warned_onsets={lk(k): overlap(T, C, k) for k in F.LEADS}) for key, T, C, what in PAIRS}
paired_onsets = dict(how_to_read="per pair and lead: the onsets of the test years warned by both sets, by only one, by neither (flood.flood_scores' onsets at trigger 1.0; the sign test is a description, the onsets of one storm are not independent)",
                     pairs={key: readings[key]["overlap_of_warned_onsets"] for key in readings})
gain_rule = {T: {f"against_{C}": {lk(k): rule(T, C, k) for k in F.LEADS} for key, T2, C, what in PAIRS if T2 == T} for T in SETS}

# ---- the verdict of results/flood_analysis.json at three warning levels, and the two operating points of code/warning_policy.py --------------------------
CRIT = FA["verdict"]["criteria"]


def number(text, key, op):
    m = re.search(rf"{key} {op} ([0-9.]+)", text)
    assert m, f"cannot read '{key} {op}' in the stored criterion: {text}"
    return float(m.group(1))


POD_GO, FAR_GO, CSI_CAUTION = number(CRIT["GO"], "pod", ">="), number(CRIT["GO"], "far", "<="), number(CRIT["CAUTION"], "csi", ">=")
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.10 if F.QUICK else 0.05)]       # warning level as a multiple of the high-flow level (06_flood_analysis.py, scripts 15-22)
KINDS, COUNTS = ("all", "onset"), ("hits", "misses", "false_alarms", "correct_neg")
pool = lambda tabs: F.rates({c: int(sum(t[c] for t in tabs)) for c in COUNTS})


def verdict(t):
    """The project's rule (results/flood_analysis.json, verdict.criteria) applied to one warning table pooled over the basins."""
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= POD_GO and t["far"] <= FAR_GO:
        return "GO"
    return "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"


def score(P, part, trig):
    tabs = {g: {lk(k): F.flood_scores(P[g], thr[g], k, part, trig[k]) for k in F.LEADS} for g in P}
    return {lk(k): {kind: pool([tabs[g][lk(k)][kind] for g in P]) for kind in KINDS} for k in F.LEADS}


def sweep(P, part):
    a, b = F.SPLIT[part]
    out = {}
    for k in F.LEADS:
        cells = []
        for g, p in P.items():
            low, q = (p.obs.shift(k) <= thr[g]).loc[a:b].values, p.loc[a:b]
            cells.append(((q.obs > thr[g]).values, q[lk(k)].values, low, thr[g]))
        out[lk(k)] = dict(all=[pool([F.contingency(o, f > m * t) for o, f, w, t in cells]) for m in LEVELS], onset=[pool([F.contingency(o[w], (f > m * t)[w]) for o, f, w, t in cells]) for m in LEVELS])
    return out


def choose(tabs):
    """Index of the level with the best pooled CSI; a tie goes to the level closest to the high-flow level itself."""
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in tabs])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])


rec_of = lambda t: dict(verdict=verdict(t), **CUT(t))


def verdict_record(P, E_test, seed_labels, seed_evals):
    """The verdict readings of one prediction set, per lead: at the plain level (trigger 1.0) and at the two levels chosen on the validation years (best pooled CSI of all high-flow days / of onsets); and the verdict of every single seed at the plain level."""
    sv, plain = sweep(P, "val"), E_test["summary"]
    chosen = {}
    for label, crit in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
        trig = {k: LEVELS[choose(sv[lk(k)][crit])] for k in F.LEADS}
        chosen[label] = (trig, score(P, "test", trig))
    return {lk(k): dict(plain=dict(trigger=1.0, **{kind: rec_of(plain[lk(k)][kind]) for kind in KINDS}),
                        **{label: dict(trigger=trig[k], chosen_by=f"best pooled validation-years CSI of {'all high-flow days' if 'all_days' in label else 'onsets'} over the levels {LEVELS[0]:.2f} to {LEVELS[-1]:.2f}",
                                       **{kind: rec_of(sc_[lk(k)][kind]) for kind in KINDS}) for label, (trig, sc_) in chosen.items()},
                        single_seeds_plain={kind: {str(s): verdict(e["summary"][lk(k)][kind]) for s, e in zip(seed_labels, seed_evals)} for kind in KINDS}) for k in F.LEADS}


verdicts = {n: verdict_record(PRED[n], EV[n]["test"], SEED_LABELS.get(n, []), EV[n]["seeds"]) for n in SHOWN}
VERDICT_NOTE = {"plain": "trigger 1.0: a warning when the forecast is above the high-flow level itself (the level of every other score in this file)",
                "chosen_on_validation_all_days": "the warning level chosen on the validation years by the best pooled CSI of all high-flow days: the level of the verdicts stored in results/flood_analysis.json",
                "chosen_on_validation_onsets": "the warning level chosen on the validation years by the best pooled CSI of the onsets"}
WP_RULE_TEXT = {"csi": "best pooled onset CSI (the project's rule, as in scripts 16 and 18)", "far_cap": f"highest pooled onset POD whose pooled onset FAR is at most {WP.FAR_CAP:.2f} (the false share the GO rule allows)"}


def operating_points(n):
    """Two warning levels chosen on the VALIDATION years with code/warning_policy.py and scored once on the test years: the CSI rule and the far_cap rule (the highest pooled onset POD whose pooled onset FAR is at
    most 0.30 on validation: the false share the GO rule allows; E4a, as script 21). The test FAR is reported as it comes out: it is not 30% by construction. Every single seed is scored at the same trigger."""
    out = {}
    for k in F.LEADS:
        A_val, A_test = WP.arrays(PRED[n], thr, k, "val"), WP.arrays(PRED[n], thr, k, "test")
        A_seeds = [WP.arrays(p, thr, k, "test") for p in SEEDP.get(n, [])]
        rec = dict(n_test_onsets=int(A_test["obs_hi"][A_test["was_low"]].sum()), rules={})
        for rule_ in ("csi", "far_cap"):
            c = WP.choose_trigger(A_val, rule_)
            t = c["trigger"]
            st = WP.score_at(A_test, t)
            seeds = [WP.score_at(A, t)["onset"] for A in A_seeds]
            rec["rules"][rule_] = dict(trigger=float(t), chosen_on="validation years: " + WP_RULE_TEXT[rule_], reached=bool(c.get("reached", True)), **({"note": c["note"]} if "note" in c else {}),
                                       validation=dict(onset=CUT(c["validation"]["onset"]), all=CUT(c["validation"]["all"])), test=dict(onset=CUT(st["onset"]), all=CUT(st["all"])),
                                       test_verdict=dict(onset=verdict(st["onset"]), all=verdict(st["all"])),
                                       single_seeds_at_this_trigger=dict(n_seeds=len(seeds), onset_pod=lo_hi([s["pod"] for s in seeds]), onset_far=lo_hi([s["far"] for s in seeds]),
                                                                         by_seed=dict(seeds=SEED_LABELS.get(n, []), onset_pod=[s["pod"] for s in seeds], onset_far=[s["far"] for s in seeds])))
        out[lk(k)] = rec
    return out


OPS = {n: operating_points(n) for n in SHOWN}
OPS_NOTE = dict(how_to_read="per set and lead, two triggers chosen on the validation years by code/warning_policy.choose_trigger over its grid (0.50 to 2.00 step 0.05) and scored once on the test years: csi (best pooled onset CSI, the project's rule) and "
                            f"far_cap (highest pooled onset POD with pooled onset FAR at most {WP.FAR_CAP:.2f} on validation; reached=false means no trigger of the grid met the cap on validation and the lowest-FAR trigger is given). "
                            "The test FAR is the one that came out, not the cap. The csi trigger here may differ from verdicts.chosen_on_validation_onsets, which uses the project's level grid 0.30 to 1.50 (both are reported).",
                criteria=CRIT, sets=OPS)


# ---- 6. the rule of ROADMAP.md, four recorded blocks -----------------------------------------------------------------------------------------------------
def basin_wilcoxon(T, C, k=1):
    """Block 1, third part: the per-basin onsets warned (each basin's pooled onset POD at trigger 1.0, seed-averaged forecasts) of T and C; two-sided Wilcoxon signed-rank test over the basins.
    Basins without an onset in the test years have no value and are left out (listed); zero differences are dropped by scipy's default zero_method='wilcox' (counted); p is scipy.stats.wilcoxon's
    default (method='auto'); the p of the normal approximation and, when scipy can compute it, of the exact null distribution are recorded beside it."""
    a = np.array([EV[T]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids], float)
    b = np.array([EV[C]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids], float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    nz = d[d != 0]
    p = p_approx = p_exact = float("nan")
    if len(nz):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = float(wilcoxon(a[ok], b[ok]).pvalue)
            p_approx = float(wilcoxon(a[ok], b[ok], method="approx").pvalue)
            try:
                p_exact = float(wilcoxon(a[ok], b[ok], method="exact").pvalue)
            except (ValueError, NotImplementedError, TypeError):
                pass
    return dict(first=f"lstm_{T}", second=f"lstm_{C}", lead=k, n_basins=n_b, n_basins_with_onsets_in_both=int(ok.sum()), basins_left_out_without_onsets=[g for g, o in zip(ids, ok) if not o],
                n_zero_differences=int((d == 0).sum()), n_used=int(len(nz)), ties_among_the_absolute_differences=bool(len(np.unique(np.abs(nz))) < len(nz)),
                n_basins_higher=int((d > 0).sum()), n_basins_lower=int((d < 0).sum()), median_difference_points=float(100 * np.median(d)) if len(d) else float("nan"),
                per_basin_difference_points={g: float(100 * (x - y)) for g, x, y, o in zip(ids, a, b, ok) if o},
                zero_method="wilcox (scipy's default): zero differences are dropped before ranking; n_used counts the basins with a non-zero difference",
                method=f"scipy.stats.wilcoxon defaults (method='auto', scipy {scipy.__version__}: the exact, permutation or normal-approximation null distribution chosen by scipy from n and the zeros and ties)",
                alternative="two-sided", p_value=p, p_value_normal_approximation=p_approx, p_value_exact=p_exact, p_below_cut=bool(p < P_CUT))


def e7_rule():
    """The rule of ROADMAP.md (section E7, fixed 2026-10-30), four blocks as written, with the numbers each block reads."""
    r = {x: ranges[x]["l1"]["onset_pod"] for x in ("wqME", "wqa", "wqME_shuf", "wqME_older")}
    bs = {x: ranges[x]["l1"]["by_seed"] for x in r}
    fME, fA = ons(EV["wqME"]["test"], 1, "far"), ons(EV["wqa"]["test"], 1, "far")
    wx = basin_wilcoxon("wqME", "wqa")
    b1 = dict(rule=RULE["block_1_gain"],
              every_seed_above=dict(holds=above(r["wqME"], r["wqa"]), wqME_single_seeds=dict(seeds=bs["wqME"]["seeds"], onset_warned=bs["wqME"]["onset_pod"]), wqME_lowest=r["wqME"][0] if r["wqME"] else None,
                                    wqa_single_seeds=dict(seeds=bs["wqa"]["seeds"], onset_warned=bs["wqa"]["onset_pod"]), wqa_highest=r["wqa"][1] if r["wqa"] else None),
              false_share_not_higher=dict(holds=bool(fME == fME and fA == fA and fME <= fA), wqME_seed_averaged=fME, wqa_seed_averaged=fA),
              wilcoxon_over_the_basins=dict(holds=wx["p_below_cut"], **wx))
    b1["holds"] = bool(b1["every_seed_above"]["holds"] and b1["false_share_not_higher"]["holds"] and b1["wilcoxon_over_the_basins"]["holds"])

    def flat(n, rule_text):
        vals = [x for x in bs[n]["onset_pod"] if x == x]
        m = float(np.median(vals)) if vals else float("nan")
        lo, hi = r["wqa"]
        inside = bool(m == m and lo <= m <= hi)
        return dict(rule=rule_text, set=f"lstm_{n}", seeds=bs[n]["seeds"], single_seed_onsets_warned=bs[n]["onset_pod"], n_seeds=len(vals), median_of_the_single_seeds=m,
                    wqa_lowest_highest_single_seed=[lo, hi], wqa_n_seeds=ranges["wqa"]["l1"]["n_seeds"], holds=inside, position="within" if inside else "above" if m > hi else "below",
                    seed_averaged_onsets_warned=ons(EV[n]["test"], 1), seed_averaged_onset_false=ons(EV[n]["test"], 1, "far"))
    b2, b3 = flat("wqME_shuf", RULE["block_2_capacity_control_flat"]), flat("wqME_older", RULE["block_3_clock_control_flat"])
    rows = {}
    for kind, (T, C) in {"lstm": ("wqME", "wqa"), "ridge regression": ("base_linreg_wqME", "base_linreg_wqa"), "random forest": ("base_rf_wqME", "base_rf_wqa")}.items():
        ET, EC = EV[T]["test"], EV[C]["test"]
        rows[kind] = {lk(k): dict(with_set=T if T.startswith("base_") else f"lstm_{T}", without_set=C if C.startswith("base_") else f"lstm_{C}", onset_warned_with=ons(ET, k), onset_warned_without=ons(EC, k),
                                  onset_gain_points=100 * (ons(ET, k) - ons(EC, k)), hits_with=ons(ET, k, "hits"), hits_without=ons(EC, k, "hits"), onset_false_with=ons(ET, k, "far"), onset_false_without=ons(EC, k, "far"),
                                  nse_median_with=med(ET, k), nse_median_without=med(EC, k), nse_gain=med(ET, k) - med(EC, k), **{f"nse_{x}": y for x, y in paired(nse_of(ET, k), nse_of(EC, k)).items()})
                      for k in F.LEADS}
    g = {kind: rows[kind]["l1"]["onset_gain_points"] for kind in rows}
    if not g["lstm"] > 0:
        y_read = "the LSTM does not gain onsets warned from the wqME columns over lstm_wqa, so there is no LSTM gain for the simple models to match"
    elif g["ridge regression"] >= g["lstm"] and g["random forest"] >= g["lstm"]:
        y_read = "the information, not the network: ridge and forest each gain at least as much from the wqME columns as the LSTM"
    elif g["ridge regression"] < g["lstm"] and g["random forest"] < g["lstm"]:
        y_read = "the LSTM gains more from the wqME columns than ridge and forest"
    else:
        y_read = "mixed: one simple model gains at least as much as the LSTM, the other less"
    b4 = dict(rule=RULE["block_4_network_yardstick"], part_of_the_verdict=False, gain_points_at_lead_1=g, reading=y_read, by_model=rows,
              stored_partners="base_linreg_wqa and base_rf_wqa of script 16 (results/preds), rescored here; base_linreg_wqME and base_rf_wqME fitted by this script with the same recipe and settings")
    met = bool(b1["holds"] and b2["holds"] and b3["holds"])
    failed = [t for t, b in (("block 1 (gain)", b1), ("block 2 (capacity control flat)", b2), ("block 3 (clock control flat)", b3)) if not b["holds"]]
    return dict(where=RULE["where"], fixed_on=FIXED_ON, scope=RULE["scope"], overall_rule=RULE["overall"], met=met, reading="a clear gain" if met else "not a clear gain", blocks_failing=failed,
                block_1_gain=b1, block_2_capacity_control_flat=b2, block_3_clock_control_flat=b3, block_4_network_yardstick=b4, interpretations_of_the_rule=INTERPRETATIONS,
                expected_before_running=RULE["expected_before_running"])


E7 = e7_rule()


# ---- 7. resampling intervals: the recipe of 13_uncertainty.py (test water years and basins drawn again with replacement) ---------------------------------------
RS_SETS = ("wqME", "wqME_shuf", "wqME_older", "wqa", "wqnh", "wq", "base_linreg_wqME", "base_rf_wqME", "base_linreg_wqa", "base_rf_wqa")
RS_PAIRS = (("wqME", "wqa"), ("wqME", "wqME_shuf"), ("wqME", "wqME_older"), ("wqME_shuf", "wqa"), ("wqME_older", "wqa"), ("wqME", "wqnh"),
            ("base_linreg_wqME", "base_linreg_wqa"), ("base_rf_wqME", "base_rf_wqa"))


def resampling():
    """13_uncertainty.py's recipe on the seed-averaged forecasts of RS_SETS: per basin and test water year the sums the NSE needs and the onset counts at trigger 1.0; B resamples that draw the test water
    years and the 12 basins again with replacement (a water year stays together); every statistic recomputed exactly; 2.5th-97.5th percentile. The point values are asserted to equal flood.evaluate's, and the
    first N_DIRECT resamples are scored again the slow way (flood.nse, flood.contingency) and must agree."""
    t0_ = time.time()
    names = list(RS_SETS)
    idx = D[ids[0]].index
    a_, b_ = (pd.Timestamp(x) for x in F.SPLIT["test"])
    T = np.flatnonzero((idx >= a_) & (idx <= b_))
    assert len(T) == len(pd.date_range(a_, b_)) and T[0] >= max(F.LEADS), "the tables do not cover every test day"
    starts = pd.date_range(a_, b_, freq=pd.DateOffset(years=1))
    ends = starts + pd.DateOffset(years=1) - pd.Timedelta(days=1)
    assert ends[-1] == b_, "the test period of flood.SPLIT is not a whole number of years"
    years = [int(e.year) for e in ends]
    ny = len(years)
    yi = np.searchsorted(starts.values, idx[T].values, side="right") - 1
    Y = (yi[:, None] == np.arange(ny)).astype(float)
    per_year = lambda x_: np.einsum("bd...,dy->by...", x_.astype(float), Y)
    obs, th = np.stack([D[g].q_mm.values for g in ids]), np.array([thr[g] for g in ids])
    o = obs[:, T]
    PT = {n: np.stack([PRED[n][g][[lk(k) for k in F.LEADS]].values for g in ids])[:, T] for n in names}
    assert all(np.isfinite(PT[n]).all() for n in names), "days without a forecast in the test years"
    hi_ = o > th[:, None]
    low = np.stack([obs[:, T - k] <= th[:, None] for k in F.LEADS], 2)
    n_y, so, soo, n_on = per_year(np.ones_like(o)), per_year(o), per_year(o ** 2), per_year(hi_[:, :, None] & low)
    warn = {n: PT[n] > th[:, None, None] for n in names}
    sse = {n: per_year((PT[n] - o[:, :, None]) ** 2) for n in names}
    hit = {n: per_year(hi_[:, :, None] & warn[n] & low) for n in names}
    fal = {n: per_year(~hi_[:, :, None] & warn[n] & low) for n in names}
    ratio = lambda x_, y_: np.divide(x_, y_, out=np.full(np.shape(x_), np.nan), where=y_ != 0)

    def stats(WB, MY, BD):
        rows = np.arange(len(WB))[:, None]
        med_ = lambda v: (np.nanmedian if np.isnan(v).any() else np.median)(v[rows, BD], axis=1)
        pool_ = lambda x_: np.einsum("rb,by,ry->r", WB, x_, MY)
        n_days, s_o = MY @ n_y.T, MY @ so.T
        sst = MY @ soo.T - s_o ** 2 / n_days
        out = {}
        for j, k in enumerate(F.LEADS):
            nse = {n: 1 - np.einsum("ry,by->rb", MY, sse[n][:, :, j]) / sst for n in names}
            onsets = pool_(n_on[:, :, j])
            for n in names:
                h, f = pool_(hit[n][:, :, j]), pool_(fal[n][:, :, j])
                out[f"nse_median.{n}.{lk(k)}"] = med_(nse[n])
                out[f"onsets_warned.{n}.{lk(k)}"], out[f"onset_false.{n}.{lk(k)}"] = ratio(h, onsets), ratio(f, h + f)
            for A_, C_ in RS_PAIRS:
                out[f"onsets_diff.{A_}_minus_{C_}.{lk(k)}"] = out[f"onsets_warned.{A_}.{lk(k)}"] - out[f"onsets_warned.{C_}.{lk(k)}"]
                out[f"nse_diff_of_medians.{A_}_minus_{C_}.{lk(k)}"] = med_(nse[A_]) - med_(nse[C_])
                out[f"nse_median_of_diffs.{A_}_minus_{C_}.{lk(k)}"] = med_(nse[A_] - nse[C_])
            for m in ("base_linreg", "base_rf"):          # block 4: the LSTM's onset gain minus the simple model's onset gain
                out[f"onsets_gain_difference.lstm_minus_{m}.{lk(k)}"] = out[f"onsets_diff.wqME_minus_wqa.{lk(k)}"] - out[f"onsets_diff.{m}_wqME_minus_{m}_wqa.{lk(k)}"]
        return out

    def direct(bd, yd):
        dd = np.concatenate([np.flatnonzero(yi == y) for y in yd])          # a year drawn twice appears twice
        sel, out = np.ix_(bd, dd), {}
        for j, k in enumerate(F.LEADS):
            nse = {n: np.array([F.nse(o[i, dd], PT[n][i, dd, j]) for i in bd]) for n in names}
            for n in names:
                t = F.contingency(hi_[sel][low[:, :, j][sel]], warn[n][:, :, j][sel][low[:, :, j][sel]])
                out[f"nse_median.{n}.{lk(k)}"], out[f"onsets_warned.{n}.{lk(k)}"], out[f"onset_false.{n}.{lk(k)}"] = float(np.median(nse[n])), t["pod"], t["far"]
            for A_, C_ in RS_PAIRS:
                out[f"nse_median_of_diffs.{A_}_minus_{C_}.{lk(k)}"] = float(np.median(nse[A_] - nse[C_]))
        return out

    with np.errstate(invalid="ignore", divide="ignore"):
        point = {key: float(v[0]) for key, v in stats(np.ones((1, n_b)), np.ones((1, ny)), np.arange(n_b)[None]).items()}
    for n in names:
        for k in F.LEADS:
            e = EV[n]["test"]["summary"][lk(k)]
            assert close(point[f"nse_median.{n}.{lk(k)}"], e["nse_median"]) and close(point[f"onsets_warned.{n}.{lk(k)}"], e["onset"]["pod"]) \
                and close(point[f"onset_false.{n}.{lk(k)}"], e["onset"]["far"]), f"{n}, {lk(k)}: the yearly sums do not give the scores of flood.evaluate"
    rng = np.random.default_rng(SEED_RS)
    BD, YD = np.empty((B_RS, n_b), int), np.empty((B_RS, ny), int)
    for r_ in range(B_RS):                                    # for each resample the basins, then the years, drawn with replacement (13_uncertainty.py)
        BD[r_], YD[r_] = rng.integers(0, n_b, n_b), rng.integers(0, ny, ny)
    WB = np.stack([np.bincount(x_, minlength=n_b) for x_ in BD]).astype(float)
    MY = np.stack([np.bincount(x_, minlength=ny) for x_ in YD]).astype(float)
    designs = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for d_, args in dict(years=(np.ones((B_RS, n_b)), MY, np.tile(np.arange(n_b), (B_RS, 1))), basins=(WB, np.ones((B_RS, ny)), BD), both=(WB, MY, BD)).items():
            designs[d_] = stats(*args)
    worst, n_cmp = 0.0, 0
    for r_ in range(min(N_DIRECT, B_RS)):
        for key, x_ in direct(BD[r_], YD[r_]).items():
            y_ = float(designs["both"][key][r_])
            assert close(x_, y_) or abs(x_ - y_) < 1e-9, f"resample {r_}: {key} is {y_} from the yearly sums but {x_} when scored directly"
            worst, n_cmp = max(worst, 0.0 if x_ != x_ else abs(x_ - y_)), n_cmp + 1

    def interval(v):
        v = v[np.isfinite(v)]
        return dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), share_above_zero=float((v > 0).mean())) if v.size else dict(lo=None, hi=None, share_above_zero=None)

    U = {key: dict(value=point[key], **{d_: interval(designs[d_][key]) for d_ in designs}) for key in point}
    return dict(how_to_read="statistics.<name>.<set or pair>.l<k>: value = the test years as they are (equal to flood.evaluate, asserted); years / basins / both = the 2.5th-97.5th percentile over the "
                            "resamples that draw the test water years / the basins / both again with replacement, and share_above_zero; quote 'both' (13_uncertainty.py). Seed-averaged LSTM forecasts; the "
                            "single seeds are not resampled. onsets_* are shares (0-1), not points.",
                recipe="13_uncertainty.py: the per-basin, per-water-year sums, basins and water years drawn with replacement (a whole water year stays together, so the days of one storm stay together), every "
                       "statistic recomputed exactly on each resample, percentile interval", n_resamples=B_RS, seed=SEED_RS, test_water_years=years, n_basins=n_b,
                direct_check=dict(resamples_scored_directly=min(N_DIRECT, B_RS), values_compared=n_cmp, largest_difference=worst, passed=True),
                not_done="the Monte-Carlo noise of the interval ends (13_uncertainty.py mc_noise) and intervals at warning levels chosen on the validation years", seconds=round(time.time() - t0_, 1), statistics=U)


RS = resampling()
d1 = RS["statistics"]["onsets_diff.wqME_minus_wqa.l1"]
print(f"resampling: {B_RS} resamples ({RS['seconds']:.0f} s); wqME minus wqa, onsets warned 1 day: {100 * d1['value']:+.1f} points (interval {100 * d1['both']['lo']:+.1f} to {100 * d1['both']['hi']:+.1f})", flush=True)

# did the code of this script, run on the stored forecasts, reproduce what scripts 15 and 16 stored for them? (the stored sets are the real ones also in a smoke test)
stored_check = {}
AR_FILE, RM_FILE = F.RES / "aorc_rain_to_midnight.json", F.RES / "rain_to_midnight.json"
AR, RM = (json.loads(f.read_text(encoding="utf-8")) if f.exists() else None for f in (AR_FILE, RM_FILE))


def compare_summary(mine, there):
    diffs = []
    for k in F.LEADS:
        a, b = there[lk(k)], mine[lk(k)]
        for x, y in ((a["nse_median"], b["nse_median"]), (a["onset"]["pod"], b["onset"]["pod"]), (a["onset"]["far"], b["onset"]["far"]), (a["onset"]["hits"], b["onset"]["hits"]), (a["all"]["csi"], b["all"]["csi"])):
            if x is not None and y == y:
                diffs.append(abs(x - y))
    return diffs


for n in list(STORED) + list(SIMPLE_STORED):
    for f, J, there in ((AR_FILE, AR, lambda J_: J_["summary_test"]["sets"].get(n) if not n.startswith("base_") else (J_["yardstick"]["sets"].get(n) or {}).get("test", {}).get("summary")),
                        (RM_FILE, RM, lambda J_: J_["summary_test"]["sets"].get(n) if not n.startswith("base_") else None)):
        s = there(J) if J else None
        if s is None:
            if J is None:
                notes.append(f"{rel(f)} was not found: the numbers this script computes for {n} were not compared with the ones stored there.")
            continue
        diffs = compare_summary(summary_test[n], s)
        key = f"{n}_against_{f.stem}"
        stored_check[key] = dict(file=rel(f), set=n, values_compared=len(diffs), max_abs_difference=float(max(diffs)) if diffs else None, all_equal=bool(diffs and max(diffs) < 1e-9))
        assert stored_check[key]["all_equal"], f"the numbers stored for {n} in {f.name} are not reproduced by this script's code"
assert any(v["set"] == "wqa" for v in stored_check.values()), "lstm_wqa was not compared with its own results file"

# ---- 8. the sentence --------------------------------------------------------------------------------------------------------------------------------------
tME, tA, tS, tO, tNH, tQ = (EV[x]["test"]["summary"]["l1"]["onset"] for x in ("wqME", "wqa", "wqME_shuf", "wqME_older", "wqnh", "wq"))
rME, rA, rS, rO = (ranges[x]["l1"] for x in ("wqME", "wqa", "wqME_shuf", "wqME_older"))
n_on = tME["hits"] + tME["misses"]
vME = verdicts["wqME"]["l1"]
b1, b2, b3, b4 = (E7[k] for k in ("block_1_gain", "block_2_capacity_control_flat", "block_3_clock_control_flat", "block_4_network_yardstick"))
wx = b1["wilcoxon_over_the_basins"]
yes = lambda b: "holds" if b else "fails"
block_text = (f"block 1, the gain, {yes(b1['holds'])} (every seed of wqME above every seed of wqa: {'yes' if b1['every_seed_above']['holds'] else 'no'}; false share not higher: "
              f"{'yes' if b1['false_share_not_higher']['holds'] else 'no'}; Wilcoxon over the basins {pv(wx['p_value'])}, n = {wx['n_used']}); block 2, the capacity control flat, {yes(b2['holds'])} "
              f"(median single seed {p1(b2['median_of_the_single_seeds'])}, {b2['position']} wqa's {span(rA['onset_pod'], p1)}); block 3, the clock control flat, {yes(b3['holds'])} "
              f"(median single seed {p1(b3['median_of_the_single_seeds'])}, {b3['position']} that range)")
n_warn = lambda t: t["hits"] + t["false_alarms"]                                    # onset warnings issued (review of 2026-10-30, must-fix 2: the level, not the skill)
csME, csA = (OPS[x]["l1"]["rules"]["csi"] for x in ("wqME", "wqa"))
fcME, fcA, fcS, fcO = (OPS[x]["l1"]["rules"]["far_cap"] for x in ("wqME", "wqa", "wqME_shuf", "wqME_older"))
d_cs, d_fc = (100 * (a["test"]["onset"]["pod"] - b["test"]["onset"]["pod"]) for a, b in ((csME, csA), (fcME, fcA)))
disc_text = (f"{'not better discrimination' if d_cs < 1.0 and d_fc < 1.0 else 'and possibly some discrimination'}: at the triggers chosen on the validation years wqME warns of "
             f"{p1(csME['test']['onset']['pod'])} against {p1(csA['test']['onset']['pod'])} (best onset CSI) and of {p1(fcME['test']['onset']['pod'])} against {p1(fcA['test']['onset']['pod'])} "
             f"(at most {100 * WP.FAR_CAP:.0f}% false; capacity control {p1(fcS['test']['onset']['pod'])}, clock control {p1(fcO['test']['onset']['pod'])})")
nse_l1 = gain_rule["wqME"]["against_wqa"]["l1"]["nse"]
X = ("the NLDAS rain of the issue day, the ERA5 rain to midnight and the AORC rain to midnight (all three made after the fact: NLDAS a gauge analysis published days late, ERA5 a reanalysis, AORC "
     + AORC_WORDING + ") side by side, beside the rain table, as inputs of the LSTM with the river gauge")
outcome = (f"Given {X}, onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone ({HINDSIGHT_WORDING}) to {pc(tME['pod'])} (single trainings {span(rME['onset_pod'])}; the onsets "
           f"of one storm are not independent): {100 * (tME['pod'] - tA['pod']):+.1f} points ({tME['hits']} against {tA['hits']} hits of {n_on}); with the NLDAS and ERA5 columns shuffled within basin, calendar "
           f"month and split part (the capacity control) {pc(tS['pod'])}, with them one day older (the clock control) {pc(tO['pod'])}; the share of false warnings among the onset warnings is {pc(tME['far'])} against "
           f"{pc(tA['far'])} ({n_warn(tME)} onset warnings against {n_warn(tA)}: the trigger-1.0 gain is a higher warning level, {disc_text}); the median NSE goes from "
           f"{n2(med(EV['wqa']['test'], 1))} to {n2(med(EV['wqME']['test'], 1))}, lower in {nse_l1['n_basins_lower']} of {nse_l1['n_basins']} basins ({pv(nse_l1['wilcoxon_p'])}; "
           f"{'a clear loss' if nse_l1['clear_loss'] else 'not a clear loss'} by the seed rule); by the rule fixed in ROADMAP.md before the first training (section E7, {FIXED_ON}) this "
           f"is {E7['reading']}; the verdict of results/flood_analysis.json is {vME['plain']['onset']['verdict']} for the onsets and {vME['plain']['all']['verdict']} for all high-flow days.")


def ci(key, scale=100.0, digits=1):
    """The 'both' interval of one resampled statistic in words (points by default)."""
    lo_, hi_ = RS["statistics"][key]["both"]["lo"], RS["statistics"][key]["both"]["hi"]
    return "no range" if lo_ is None else f"{scale * lo_:+.{digits}f} and {scale * hi_:+.{digits}f}"


ci_onset, ci_nse = ci("onsets_diff.wqME_minus_wqa.l1"), ci("nse_diff_of_medians.wqME_minus_wqa.l1", 1.0, 3)
share_up = RS["statistics"]["onsets_diff.wqME_minus_wqa.l1"]["both"]["share_above_zero"]
ov1, ovNH = readings["several_products_against_aorc_alone"]["overlap_of_warned_onsets"]["l1"], readings["several_products_against_nldas_and_era5"]["overlap_of_warned_onsets"]["l1"]
opME, opA = (OPS[x]["l1"]["rules"]["far_cap"] for x in ("wqME", "wqa"))
cap_note = lambda op: "" if op["reached"] else ", cap not reached on validation"
d_op = 100 * (opME["test"]["onset"]["pod"] - opA["test"]["onset"]["pod"])
pct_cap = f"{100 * WP.FAR_CAP:.0f}%"
gy = b4["gain_points_at_lead_1"]
sME_, sA_ = rME["by_seed"]["onset_pod"], rA["by_seed"]["onset_pod"]
seed_pairs_n, seed_pairs_down = len(sME_) * len(sA_), sum(a < b for a in sME_ for b in sA_)     # review of 2026-10-30, should-fix 2
n_s, n_st = rME["n_seeds"], rA["n_seeds"]
outcome_caveat = (
    f"The rule, block by block: {block_text}. "
    f"At trigger 1.0 (and so measuring the higher warning level as well), per basin wqME warns of more onsets than wqa in {wx['n_basins_higher']} of {wx['n_basins_with_onsets_in_both']} basins and of fewer in "
    f"{wx['n_basins_lower']} (median difference {wx['median_difference_points']:+.1f} points); {ov1['warned_only_by_first']} onsets are warned only by wqME and {ov1['warned_only_by_second']} only by wqa. "
    f"The difference in onsets warned 1 day ahead lies between {ci_onset} points when the test water years and the basins are drawn again ({B_RS:,} resamples, the recipe of 13_uncertainty.py; above zero "
    f"in {pc(share_up) if share_up is not None else 'no'} of them); the training seeds are not drawn again: {seed_pairs_down} of the {seed_pairs_n} pairs of single trainings go the other way. "
    f"The difference of the median NSE lies between {ci_nse}. "
    f"Beside the two products without AORC (lstm_wqnh, {pc(tNH['pod'])}) wqME warns of {100 * (tME['pod'] - tNH['pod']):+.1f} points more ({ovNH['warned_only_by_first']} onsets only by wqME, "
    f"{ovNH['warned_only_by_second']} only by wqnh); the rain table alone (lstm_wq) warns of {pc(tQ['pod'])}. Two and three days ahead the median NSE of wqME is {n2(med(EV['wqME']['test'], 2))} and "
    f"{n2(med(EV['wqME']['test'], 3))} against {n2(med(EV['wqa']['test'], 2))} and {n2(med(EV['wqa']['test'], 3))} for wqa. "
    f"At the {pct_cap}-false operating point (the trigger chosen on the validation years so that at most {pct_cap} of the onset warnings are false: {opME['trigger']:.2f}{cap_note(opME)}) wqME warns of "
    f"{pc(opME['test']['onset']['pod'])} with {pc(opME['test']['onset']['far'])} false on the test years, against {pc(opA['test']['onset']['pod'])} with {pc(opA['test']['onset']['far'])} false for wqa "
    f"(trigger {opA['trigger']:.2f}{cap_note(opA)}): {abs(d_op):.1f} points {'more' if d_op > 0 else 'fewer' if d_op < 0 else 'apart'}. "
    f"The network yardstick (block 4, not part of the verdict): given the same columns, ridge regression moves by {gy['ridge regression']:+.1f} points and the random forest by {gy['random forest']:+.1f} "
    f"against their wqa versions, the LSTM by {gy['lstm']:+.1f}: {b4['reading']} (at trigger 1.0 only). "
    f"After the fact: {AFTER_THE_FACT_ALL}. The hindsight bound of wqa ({HINDSIGHT_WORDING}) was measured for AORC alone and does not bound wqME, whose NLDAS and ERA5 columns are after the fact "
    f"too: wqME is a test of information, not a forecast that could be issued. "
    f"The shuffled columns of the capacity control move values between the years of the same split part ({100 * SHUFFLE['share_from_a_later_day']:.0f}% of them to a day before the day they were measured): they carry "
    f"no timing information by construction and are a capacity control, not a forecast input. "
    f"Every number of the new sets is a {NUMBER_WORD.get(n_s, n_s)}-seed number; lstm_wqa has {NUMBER_WORD.get(n_st, n_st)} seeds, lstm_wq and lstm_wqn {NUMBER_WORD.get(ranges['wq']['l1']['n_seeds'], '?')}.")
outcome_full = outcome + " " + outcome_caveat

# ---- 9. status, integrity, timing ---------------------------------------------------------------------------------------------------------------------------
review = json.loads(REVIEW_FILE.read_text(encoding="utf-8")) if REVIEW_FILE.exists() and not F.QUICK else None       # the independent review record: read, never written here (as script 21)
rounds = sorted(int(k[6:]) for k in (review or {}) if re.fullmatch(r"round_\d+", k))      # the latest round of the record speaks (round_N with the highest N; the top level is round 1)
rv = review[f"round_{rounds[-1]}"] if rounds else review
review_covers = bool(review) and RESCORE                    # a review covers the saved sets it was made on; a pass that trains (plain or --extend) makes sets no review has seen
STORED_NOW = {rel(REAL_PRED / f"{x}.npz"): sha(REAL_PRED / f"{x}.npz") for x in [f"lstm_{n}{s}" for n in STORED for s in [""] + [f"_s{i}" for i in STORED[n]["seeds"]]] + list(SIMPLE_STORED)}
prev_stored = ((old or {}).get("integrity") or {}).get("stored_sets_read") or {}
stored_changed = sorted(p for p in set(STORED_NOW) | set(prev_stored) if STORED_NOW.get(p) != prev_stored.get(p)) if prev_stored else []
if rounds or review:        # the review covers this rescore while the stored sets are the files it recomputed its numbers with (its stored_sets_reviewed) and no training pass finished after the results file it reviewed
    ssr = (rv or {}).get("stored_sets_reviewed") or {}
    if ssr:
        stored_changed = sorted(p for p in set(STORED_NOW) | set(ssr) if STORED_NOW.get(p) != ssr.get(p))
    rev_written = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", str(((rv or {}).get("dates") or {}).get("results_file_written", "")))
    review_covers = review_covers and bool(rev_written) and all(str(p.get("finished_local_time", "9")) <= rev_written[0] for p in (old or {}).get("passes", []) if p.get("kind") in ("training run", "extension"))
review_covers = review_covers and not stored_changed
if review:
    sev = [str(f.get("severity", "")) for f in rv.get("findings", [])]
    n_must, n_should, n_note = sum(x.startswith("must") for x in sev), sum(x.startswith("should") for x in sev), sum(x.startswith("note") for x in sev)
    who = str(rv.get("who", "AI reviewer agent"))
    review_date = next(iter(re.findall(r"\d{4}-\d{2}-\d{2}", " ".join(str(x) for x in ((rv.get("dates") or {}).get("review", ""), rv.get("when", ""), who)))), "date not given")
    cnt = rv.get("counts") or {}
    mism = (f"{cnt['numerical_mismatches']} numerical mismatch{'es' if cnt['numerical_mismatches'] != 1 else ''} in {cnt['values_compared']} values recomputed from results/preds with its own code"
            if {"numerical_mismatches", "values_compared"} <= set(cnt) else "its numbers recomputed from results/preds with its own code")
    review_line = (f"independent review done ({who if review_date in who else who + ', ' + review_date}{'; round ' + str(rounds[-1]) if rounds else ''}): verdict '{rv.get('verdict', '')}', {n_must} must-fix, {n_should} should-fix and "
                   f"{n_note} note{'s' if n_note != 1 else ''}; {mism} ({rel(REVIEW_FILE)})")
    if not review_covers:
        review_line = (f"{rel(REVIEW_FILE)} ({review_date}) reviews an earlier training pass, not the networks trained in this pass: a new review is needed before any number is quoted" if not RESCORE else
                       f"{rel(REVIEW_FILE)} ({review_date}) recomputed this file's numbers with the stored comparison sets as they were then; {len(stored_changed)} of their files have changed or are new since, "
                       "or a training pass finished after the results file it reviewed: the review predates these numbers and a new review is needed before any number is quoted")
else:
    n_must = n_should = n_note = review_date = None
    cnt = {}
    review_line = "this file's independent review is not yet done"
if F.QUICK:
    status = "SMOKE TEST (FLOOD_QUICK=1): one seed, 40 updates, tiny forests, results/quick: nothing in this file is a result"
elif review_covers:
    status = f"REVIEWED{f' (round {rounds[-1]})' if rounds else ''}: {review_line}. The team's own checks are not yet done."
elif review:
    status = f"NOT YET REVIEWED for this pass: {review_line}; every number of this file is preliminary and must carry that label in any document (HANDOFF.md section 7)"
else:
    status = (f"NOT YET REVIEWED: no independent review record ({rel(REVIEW_FILE)}) exists for this experiment; every number of this file is preliminary and must carry that label in any document "
              "(HANDOFF.md section 7: each new script gets its own independent review that recomputes the headline numbers from results/preds). The team's own checks are not yet done.")
SAVED = {}
for n in SETS:
    for f in [F.PRED / f"lstm_{n}.npz"] + [p for s in SEEDS for p in (F.PRED / f"lstm_{n}_s{s}.npz", MODELS / f"lstm_{n}_s{s}.pt")]:
        SAVED[rel(f)] = sha(f)
for n in SIMPLE_NEW:
    SAVED[rel(F.PRED / f"{n}.npz")] = sha(F.PRED / f"{n}.npz")
made_now = ({rel(F.PRED / f"lstm_{n}_s{s}.npz") for n, s in trained_pairs} | {rel(MODELS / f"lstm_{n}_s{s}.pt") for n, s in trained_pairs} | {rel(F.PRED / f"lstm_{n}.npz") for n in SETS if trained_now[n]}
            | {rel(F.PRED / f"{n}.npz") for n in simple_now})
earlier_hashes = (old or {}).get("integrity", {}).get("sha256_of_saved_files", {}) if LOAD else {}
compared = [f for f in SAVED if f not in made_now and f in earlier_hashes]
assert all(earlier_hashes[f] == SAVED[f] for f in compared), "a saved file changed since the earlier results file recorded its hash: " + ", ".join(f for f in compared if earlier_hashes[f] != SAVED[f])
INTEGRITY = dict(how_to_read="sha256 of every prediction set and network file of this script, as they are when this file is written; made_in_this_pass lists the files this pass wrote; every other file was only read. A later pass compares "
                             "the hashes of the files it only reads with these and stops if one differs. stored_sets_read: the files of the stored comparison sets as read (never written by this script).",
                 script_sha256=HASH_NOW, sha256_of_saved_files=SAVED, made_in_this_pass=sorted(made_now), only_read_in_this_pass=len(SAVED) - len(made_now), stored_sets_read=STORED_NOW,
                 stored_networks_read_by_the_column_identity_check={c["model_file"]: sha(ROOT / c["model_file"]) for c in IDENTITY["checks"]},
                 compared_with_the_earlier_results_file=(dict(files_compared=len(compared), all_equal=True) if compared else dict(files_compared=0, all_equal=None, why="nothing was read from an earlier pass")) if LOAD else None)
import torch                                                # already imported by flood; only for the version record
MACHINE = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, pandas=pd.__version__)
this_pass = dict(kind="training run" if not LOAD else "extension" if EXTEND else "rescore",
                 what=("training run: every set and simple model from scratch" if not LOAD else "extension (--extend): the (set, seed) pairs and simple models whose files were missing" if EXTEND
                       else "rescore (--rescore): scores, tables, outcome and figure recomputed from the saved sets, nothing trained"),
                 n_trainings=len(trained_pairs), trained=[f"lstm_{n}_s{s}" for n, s in trained_pairs], training_seconds=round(sum(costs[n][s]["seconds"] for n, s in trained_pairs), 1),
                 simple_models_fitted=simple_now, simple_models_fit_seconds={n: SIMPLE_SEC[n] for n in simple_now}, waited_for_cpu_s=round(waited, 1), seeds=SEEDS, script_sha256=HASH_NOW,
                 roadmap_e7_section_sha256=E7_SECTION["section_sha256"], finished_local_time=time.strftime("%Y-%m-%d %H:%M"), torch_threads=int(os.environ.get("FLOOD_THREADS", 1 if F.QUICK else 4)), machine=MACHINE)
first = [dict(p) for p in old.get("passes", []) if p.get("kind") != "rescore"] if LOAD else []
passes = first + [this_pass]
n_trainings, training_seconds = sum(p["n_trainings"] for p in passes), round(sum(p["training_seconds"] for p in passes), 1)
train_passes = [p for p in passes if p.get("kind") == "training run"]
train_pass = train_passes[0] if train_passes else this_pass
FIXED = dict(
    where=f"ROADMAP.md, section '{E7_SECTION['heading']}' (fixed on {FIXED_ON}, before the first training); the design: NEXT_EXPERIMENT_E7_multi_product.md, section 2 (E7a; the MRMS arm E7b was ruled out on {FIXED_ON})",
    fixed_on=FIXED_ON, rule_as_coded=RULE, roadmap_e7_section_sha256_at_the_training_pass=train_pass.get("roadmap_e7_section_sha256"), roadmap_e7_section_sha256_now=E7_SECTION["section_sha256"],
    same_rule_text_as_at_the_training_pass=bool(train_pass.get("roadmap_e7_section_sha256") == E7_SECTION["section_sha256"]),
    training_pass_script_sha256=train_pass.get("script_sha256"), this_pass_script_sha256=HASH_NOW,
    fixed_in_this_script_before_the_first_training=[f"the shuffle's random seed ({SHUF_SEED}) and unit (single days within basin x calendar month x split part; 'split part' added on 30 October 2026 after the smoke test, before the full training)",
                                                    f"the stored networks of the column-identity check (seeds {', '.join(map(str, IDENTITY_SEEDS))} of lstm_wqa and lstm_wqnh)",
                                                    "the readings of the rule written in e7_rule.interpretations_of_the_rule",
                                                    f"the resampling design of 13_uncertainty.py (seed {SEED_RS}, {4000:,} resamples outside a smoke test)"],
    test_years="scored once per pass; no choice of this script was made on test-year scores")
choices = []
if EXTEND and old and trained_pairs:
    choices.append(f"{', '.join(f'lstm_{n}_s{s}' for n, s in trained_pairs)} were trained with --extend on {this_pass['finished_local_time'][:10]} after the test-year scores of the earlier pass "
                   f"({len(first)} pass(es)) had been seen.")
if EXTEND and old and simple_now:
    choices.append(f"{', '.join(simple_now)} were fitted with --extend on {this_pass['finished_local_time'][:10]} after the test-year scores of the earlier pass had been seen.")
for p in first[1:]:                      # carried over from earlier extension passes
    if p.get("kind") == "extension" and p.get("trained"):
        choices.append(f"{', '.join(p['trained'])} were trained with --extend (pass finished {p.get('finished_local_time')}) after the test-year scores of an earlier pass had been seen.")
choices = list(dict.fromkeys(choices))
if not FIXED["same_rule_text_as_at_the_training_pass"]:
    notes.append("ROADMAP.md's E7 section has changed since the training pass (fixed_before_training: the two sha256 differ): the rule coded here is the one fixed before training; read the change before quoting.")
if len(train_pass.get("seeds", SEEDS)) < 5 and not F.QUICK:
    notes.append(f"ROADMAP.md fixes seeds 0-4; the training pass trained {len(train_pass.get('seeds', SEEDS))}: run --extend --seeds 5 (recorded in choices_made_after_looking_at_test_scores).")
if F.QUICK:
    notes.append("Smoke test (FLOOD_QUICK=1): one seed, 40 updates, tiny forests, 200 resamples, outputs in results/quick and figures/quick; the stored sets are the real ones of results/preds, the new sets are not results.")
notes.append("The stored sets lstm_wqa, lstm_wq, lstm_wqn, lstm_wqh, lstm_wqnh and the stored simple models base_linreg_wqa, base_rf_wqa were rescored by this script's code from results/preds "
             "(checks_against_stored_results proves the same numbers as their own results files). The stored networks of lstm_wqa and lstm_wqnh named in column_identity_check were run on this script's "
             "table to prove the columns; no other stored network was reloaded.")
notes.append("lstm_wq and lstm_wqn are three-seed sets (04_lstm_all.py, 10_rain_timing.py); they are shown, not part of the rule.")
notes.append("Quoting. Any document must quote outcome_full (outcome plus outcome_caveat), never outcome alone; every number is a seed-averaged number unless it says 'single trainings'; the status field says whether "
             "the experiment has been reviewed. E7 is not part of the 2 November hand-in (ROADMAP.md).")
NOT_DONE = ([] if review_covers else [f"an independent review of this pass that recomputes the headline numbers from results/preds and the saved networks ({rel(REVIEW_FILE)}; NEXT_EXPERIMENT_E7_multi_product.md section 5)"]) \
    + ["the team's own checks", "a hindsight measurement for wqME (the bound quoted was measured for wqa's AORC column alone; wqME's NLDAS and ERA5 columns are after the fact too and have no real-time stand-in here)", "the MRMS arm E7b (ruled out by the team lead on 2026-10-30)",
       "the Monte-Carlo noise of the resampling interval ends (13_uncertainty.py mc_noise)", "any document: E7 is not part of the 2 November hand-in"] \
    + ([f"five seeds (--extend --seeds 5; this run has {NUMBER_WORD.get(len(SEEDS), len(SEEDS))})"] if len(SEEDS) < 5 and not F.QUICK else [])
PROVENANCE = dict(
    written_and_run_by="Claude Code wrote this script (30 October 2026) and ran it under the team lead's direction; AI agents review it",
    independent_review=(dict(record=rel(REVIEW_FILE), record_sha256=sha(REVIEW_FILE), who=rv.get("who"), date=review_date, verdict=rv.get("verdict"), findings=len(rv.get("findings", [])),
                             **({"round": f"round_{rounds[-1]}"} if rounds else {}), must_fix=n_must, should_fix=n_should, notes=n_note, numerical_mismatches=cnt.get("numerical_mismatches"),
                             values_compared=cnt.get("values_compared"), covers_this_pass=review_covers, stored_sets_changed_since_the_review=stored_changed)
                        if review else f"not yet done: no {rel(REVIEW_FILE)} exists; HANDOFF.md section 7 asks for one that recomputes the headline numbers from results/preds"),
    review_line=review_line, team_check="The team's own checks are not yet done (team.json)")

R = dict(
    question="E7: does an LSTM fed several rain products side by side (the rain table, the NLDAS rain of the issue day, the ERA5 rain to midnight and the AORC rain to midnight, " + AORC_WORDING + ") warn of "
             "more flood onsets 1 day ahead than the LSTM fed the best single product (lstm_wqa, the AORC rain to midnight beside the rain table)?",
    script=SCRIPT, written=time.strftime("%Y-%m-%d %H:%M"), status=status, provenance=PROVENANCE, quick_mode=F.QUICK, split=F.SPLIT,
    outcome=outcome, outcome_caveat=outcome_caveat, outcome_full=outcome_full,
    fixed_before_training=FIXED, choices_made_after_looking_at_test_scores=choices, e7_rule=E7,
    design=dict(what="one LSTM (flood.LSTMNet, hidden 64) for all 12 basins with the river gauge (log1p flow) and the basin properties (flood.STATIC), the settings of results/lstm_all.json chosen there on the validation years; "
                     "every set differs only in the daily input columns listed; the trainer stops on the mean validation NSE; the test years 2005-2014 are scored once",
                sets={n: dict(saved_as=f"lstm_{n}", role=v["role"], control=v["control"], what=v["what"], extra_columns=list(v["cols"]), dyn=list(v["dyn"]), n_in=v["n_in"], seeds=SEEDS, n_params=costs[n][SEEDS[0]]["n_params"]) for n, v in SETS.items()},
                stored_sets={n: dict(saved_as=f"lstm_{n}", script=v["script"], role=v["role"], what=v["what"], dyn=list(v["dyn"]) if v["dyn"] else None, seeds=v["seeds"], rescored_here=True) for n, v in STORED.items()},
                simple_models={n: dict(kind=KIND[n], settings=SETTINGS[KIND[n]], features=FEATURES, stored_partner=PARTNER[n]) for n in SIMPLE_NEW},
                stored_simple_models={n: dict(script="16_aorc_rain_to_midnight.py", kind=KIND[n], rescored_here=True) for n in SIMPLE_STORED},
                base="lstm_wqa", order_of_training=list(SETS), seeds=SEEDS, settings_source="results/lstm_all.json, config", columns=COLUMN_TEXT),
    inputs=dict(how_to_read="the three rain products as built here, per basin: the data-side checks of both hourly products (two routes, made-up hours), the shuffle record, and the sha256 of every input column "
                            "(all basins stacked, float64): a later pass stops if its columns differ",
                data_side=DATA_SIDE, time_zones=ZONES, shuffle=SHUFFLE, sha256_of_input_columns=INPUT_SHA),
    column_identity_check=IDENTITY,
    config=conf, config_source="results/lstm_all.json, config" + (" (updates and eval_every reduced for the smoke test)" if F.QUICK else ""), seeds=SEEDS, n_seeds_per_set={n: ranges[n]["l1"]["n_seeds"] for n in LSTM_SHOWN},
    thresholds=thr,
    summary_test=dict(how_to_read="test years, seed average of each LSTM set (the simple models as they are), trigger 1.0, per lead; onset and all = warning tables pooled over the basins; the stored sets rescored here", sets=summary_test),
    validation_used_for_no_choice=dict(label="USED FOR NO CHOICE. Validation years (2000-2005), shown for reference only: no set, setting or stopping point of this script was chosen on these scores (the settings came from 04_lstm_all.py; "
                                             "the trainer stopped each training on its own validation score, so these scores flatter the models). The validation years choose only the warning levels of verdicts and operating_points, which is labelled there.",
                                       summary=summary_val),
    per_basin=per_basin, seed_ranges=dict(how_to_read="lowest and highest single-seed value of the test-year scores, per LSTM set and lead (by_seed: the single values, in the order of the seeds)", sets=ranges),
    gain_rule=dict(how_to_read="the checks of report section 4.2 for every LSTM pair (set T against set C): NSE clear gain = every seed of T above every seed of C, higher in most basins, Wilcoxon p below 0.05; onsets clear gain = every seed above every seed (pooled POD)",
                   sets=gain_rule),
    readings=readings, paired_onsets=paired_onsets, resampling=RS,
    operating_points=OPS_NOTE, verdicts=dict(criteria=CRIT, criteria_source="results/flood_analysis.json, verdict.criteria", levels=VERDICT_NOTE, level_grid=LEVELS, sets=verdicts),
    yardstick=dict(what="ridge regression and the random forest given the three rain products (rule block 4), by the recipe of 02_baselines.py as script 16 runs it for one column; read in e7_rule.block_4_network_yardstick",
                   recipe_check=RECIPE, features=FEATURES, settings_source="results/baselines.json, models.base_linreg_wq.settings and models.base_rf_wq.settings",
                   fit_seconds={n: SIMPLE_SEC.get(n) for n in SIMPLE_NEW}),
    causality=causality, reload_check=reload_check, checks_against_stored_results=stored_check,
    models=models, training=training, integrity=INTEGRITY,
    n_trainings=n_trainings, training_seconds=training_seconds, waited_for_cpu_s=round(waited, 1), passes=passes, runtime_s=round(time.time() - t_start, 1), script_sha256=HASH_NOW,
    notes=notes, what_was_not_done=NOT_DONE)


def clean(o, count=None):
    """Strict JSON: numpy values become Python ones, and a number that is not finite (NaN: a share whose denominator is zero) becomes None (null). count[0] is raised by one for every value replaced."""
    if isinstance(o, dict):
        return {str(k): clean(v, count) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, count) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist(), count)
    if isinstance(o, (np.floating, float)):
        if math.isfinite(float(o)):
            return float(o)
        if count is not None:
            count[0] += 1
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def count_null(o):
    """Null leaves of the cleaned record: the NaN of this pass and the nulls read back from the earlier pass's results file."""
    if isinstance(o, dict):
        return sum(count_null(v) for v in o.values())
    if isinstance(o, list):
        return sum(count_null(v) for v in o)
    return int(o is None)


nulls = [0]
n_null = count_null(clean(R, nulls))
notes.append(f"{n_null} values in this file are null: a number that is not finite (NaN: a share whose denominator is zero, a p-value that is not defined, the load average, which Windows does not have; "
             f"{nulls[0]} of them NaN in this pass's own numbers, the others null already in what this pass read back from the earlier pass) or a value that does not exist. A reader must treat null as 'no value'.")
R["runtime_s"] = round(time.time() - t_start, 1)
JSON_TEXT = json.dumps(clean(R), indent=1, allow_nan=False)
RESULTS_FILE.write_text(JSON_TEXT, encoding="utf-8")
R = json.loads(JSON_TEXT)                                   # the figure reads exactly what the JSON holds
draw(R)
print(f"\n{outcome_full}\n")
print(f"status: {status}")
print(f"wrote {RESULTS_FILE} and {FIG_FILE} ({R['runtime_s']} s; {n_trainings} trainings in all, {this_pass['n_trainings']} in this pass)", flush=True)
