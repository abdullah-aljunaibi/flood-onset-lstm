"""E4b: within-day AORC rain and Q95-relative flow as inputs of the all-river LSTM (NEXT_EXPERIMENTS_2026-10-28.md, section E4b).

The LSTM of script 16 (set wqa) reads one daily total of the issue day's rain: aorc_to_midnight, the hourly AORC rain (an analysis made after the fact) summed over the local
wall-clock day t. Basins of 112 to 1,928 km2 respond within hours, so WHEN the rain fell inside day t, and how hard, is information the daily total hides. code/subdaily_inputs.py
builds from the same hours (data/aorc, made by code/fetch_aorc.py) six within-day columns of day t, each from hours that END at or before 24:00 of day t by the wall clock (the issue
time): aorc_last3h, aorc_last6h, aorc_last12h (the rain of the last 3, 6 and 12 hours of the day), aorc_max1h (the wettest hour, basin mean), aorc_maxcell1h (the wettest 1 km cell in
any hour), aorc_wet_hours (hours with at least 0.5 mm); and q_rel = log1p(q_mm / Q95): the flow of day t relative to the basin's own high-flow level (the engine scales the flow with one
global mean and spread, so 'near the flood level' is a different number in every basin).
 Sets (settings of results/lstm_all.json; the river gauge and the basin properties as in wqa; the seeds 0, 1, 2 by default, --seeds 5 adds 3 and 4):
   wqaQ        FORCING + aorc_to_midnight + q_rel                        isolates the scaling change
   wqaW        FORCING + aorc_to_midnight + the six within-day columns    the test
   wqaW_older  the same six columns one day older (on issue day t the network reads the within-day columns of day t-1): the control - the same parameters, no new information about day t
   wqaWF_sub   wqaW + fs1..fs3, the substituted forecast columns of 18_forecast_input.py (only with --with-forecast; compared with the stored lstm_wqaF_sub)
 Compared with the STORED sets lstm_wqa (the base; script 16), lstm_wqaF_sub and lstm_wqaP (the perfect rain forecast, the ceiling; script 18), rescored here by this script's own code,
 never copied: the fixed level (trigger 1.0: pooled onset POD / FAR, median NSE, the seed rule 'every seed of T above every seed of C', per-basin wins), paired onsets (warned by both /
 by only one), the warning level chosen on the validation years (best pooled onset CSI), the 30%-false operating point of code/warning_policy.py (far_cap rule, chosen on validation),
 the closure of the gap to wqaP, and the onsets with under 5 mm of rain on the target day (the slice of results/review_forecast_input.json, must_fix[0] (iv)).
 Success rule, fixed before training: wqaW above wqa AND above wqaW_older in every seed at lead 1 (onsets warned, pooled over the 12 basins, trigger 1.0).
 Causality: subdaily_inputs.checks() (routes, window order, clock-change days, made-up hours) runs first and its record is stored; flood._assert_causal runs inside every flood.fit; and the
 made-up-values test of script 18 runs on one saved network per set: every input after a cut day replaced by random values -> no forecast issued on or before the cut changes, later ones do.
 Hazard (critic C14): aorc_maxcell1h is NaN on days whose every hour was filled from ERA5 (max_cell_mm is NaN on filled hours; flood._first_valid would stop on the gap); it is filled with
 aorc_max1h on those days and the days are recorded (maxcell_fill_days).
Reads results/review_within_day_inputs.json when it exists (the independent review: its verdict, date and must-fix count go into status, provenance and the figure's header and footer; its
wording findings of 2026-10-29 are applied in this file). Writes results/within_day_inputs.json, figures/within_day_inputs.png, results/models/lstm_<set>_s<seed>.pt, the prediction sets lstm_<set>_s<seed> and lstm_<set> (the seed mean), and
rewrites results/diagnostics_2026-10-28/subdaily_inputs_check.json (the record of subdaily_inputs.checks(), a preliminary diagnostic that stays one).
Modes: plain run = train everything; --rescore = recompute from the saved sets (nothing trained); --extend = train only the missing (set, seed) pairs; --figure = the figure again from the results file.
FLOOD_QUICK=1: one seed, 40 updates, results/quick and figures/quick (the stored sets are read from results/preds). torch is imported inside the functions that use it directly (flood brings it in
anyway); the syntax of this file is checked with python -X utf8 -m py_compile code/21_within_day_inputs.py.
Run: python -X utf8 code/21_within_day_inputs.py [--rescore | --extend] [--seeds 3|5] [--with-forecast]
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_aorc as AO
import flood as F                                           # the shared engine (it imports torch)
import subdaily_inputs as SI
import warning_policy as WP
from common import AQUA, AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, ROOT, YELLOW, use_style

use_style()
SCRIPT = "21_within_day_inputs.py"
OUT = "within_day_inputs"                                   # results/<OUT>.json, figures/<OUT>.png
t_start = time.time()
HASH_NOW = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()      # the code that runs, hashed once at the start: every pass records it

ap = argparse.ArgumentParser(description="E4b: within-day AORC rain and Q95-relative flow as LSTM inputs")
ap.add_argument("--rescore", action="store_true", help="no training: scores, tables, outcome and figure again from the saved prediction sets")
ap.add_argument("--extend", action="store_true", help="as --rescore, but trains the (set, seed) pairs whose files are missing")
ap.add_argument("--figure", action="store_true", help="only the figure again, from the results file")
ap.add_argument("--seeds", type=int, default=3, choices=(3, 5), help="3 (seeds 0, 1, 2) or 5 (adds 3 and 4)")
ap.add_argument("--with-forecast", action="store_true", help="also train wqaWF_sub (wqaW + the substituted forecast columns fs1..fs3 of script 18)")
ARGS = ap.parse_args()
RESCORE, EXTEND, WITH_FORECAST = ARGS.rescore, ARGS.extend, ARGS.with_forecast
assert not (RESCORE and EXTEND), "--rescore trains nothing and --extend trains what is missing: use one of the two"
LOAD = RESCORE or EXTEND                                    # the saved sets and the training records of the results file are read, not made again
SEEDS = [0] if F.QUICK else list(range(ARGS.seeds))
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")        # the training record of one model (Run.summary)
P_CUT = 0.05                                                # 'p is small', as in section 4.2 of the report
MODELS = F.RES_OUT / "models"                               # one file per trained network (results/quick/models in a smoke test)
RESULTS_FILE, FIG_FILE = F.RES_OUT / f"{OUT}.json", F.FIG_OUT / f"{OUT}.png"
REVIEW_FILE = F.RES / f"review_{OUT}.json"                  # the independent review record, when it exists
WD_COLS = ("aorc_last3h", "aorc_last6h", "aorc_last12h", "aorc_max1h", "aorc_maxcell1h", "aorc_wet_hours")
OLDER_COLS = tuple(c + "_older" for c in WD_COLS)
FS_COLS = ("fs1", "fs2", "fs3")
BASE_DYN = tuple(F.FORCING) + ("aorc_to_midnight",)        # the daily inputs of wqa: the rain table (Daymet) + the AORC rain to midnight
SMALL_MM, DRY_MM = 5.0, 1.0                                 # 'little rain on the target day' (the review's slice); 'dry issue day' (the work order's 70 onsets)
CONF_KEYS = {"hidden", "lr", "dropout", "warmup", "block", "batch", "updates", "eval_every", "patience"}       # the settings results/lstm_all.json holds
AORC_WORDING = "an analysis made after the fact"            # what AORC is - never 'observed' rain (code/report_sections/s0_summary.py)
def _hindsight_wording():
    """'hindsight at most about N points' beside the AORC rain-to-midnight share: N = the loosest bound of the Stage IV route for the networks the stored
    lstm_wqa set averages (results/review_aorc_rain_to_midnight.json, the latest round with a five-seed bound; 30 October 2026: 3.6 points, five networks);
    the review of 7 October 2026 measured 'about 3 points' on the three networks of seeds 0-2, the fallback when no later round holds a bound."""
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
UPPER_WORDING = "an upper bound that cannot be had in practice"
SUCCESS_RULE = ("wqaW above wqa and above wqaW_older in every seed at lead 1: onsets warned 1 day ahead, pooled over the 12 basins at trigger 1.0, every single seed of wqaW above every single seed "
                "of lstm_wqa (stored, three seeds) and of lstm_wqaW_older")
TRAIN_A, TRAIN_B = F.SPLIT["train"]


def success_rule_now():
    """SUCCESS_RULE as fixed on 28 October 2026 (stored lstm_wqa: three seeds), with the seed count of lstm_wqa as it is read now (seeds 3-4 added on 30 October 2026)."""
    n_ = ranges["wqa"][lk(1)]["n_seeds"]
    return SUCCESS_RULE if n_ == 3 else SUCCESS_RULE.replace("(stored, three seeds)", f"(stored, {NUMBER_WORD.get(n_, n_)} seeds since 30 October 2026; three when the rule was fixed)")
PRE_2002 = pd.Timestamp(TRAIN_B).year < 2002                # AORC's hours before 2002 are daily totals spread by older products (AORC v1.1 methods, as read by the E1b review); from 2002 they have Stage IV's hourly shape


def fixed_before_training(n_seeds, train_hash):
    """What was fixed where before the first training (review of 2026-10-29, finding 8): the work order fixed the sets, the scoring and the expectation; the control, the rule's wording and the seed count exist only in this script."""
    return (f"The sets wqaQ, wqaW and wqaWF_sub, the scoring (as script 16: seed rule, paired onsets, per-basin, the closure of the gap to wqaP; and the {100 * WP.FAR_CAP:.0f}%-false operating point of E4a) and the expectation "
            f"(the gain in the onsets with little rain on the target day) were fixed in NEXT_EXPERIMENTS_2026-10-28.md; the older control wqaW_older, the success rule as worded ({SUCCESS_RULE.split(':')[0]}) and the "
            f"{NUMBER_WORD.get(n_seeds, n_seeds)} seeds were fixed in this script before the first training (its sha256 is recorded at the start of every pass: {train_hash} for the training pass, passes[0].script_sha256); "
            f"the test years were scored once.")


NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
lk = lambda k: f"l{k}"
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)                         # a path as the project shows it
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
notes = []
REAL_PRED = F.RES / "preds"                                 # the stored sets are ALWAYS read from here, also in a smoke test (flood.load_preds would prefer a smoke-test copy of the same name in results/quick/preds)
real_have = lambda name: (REAL_PRED / f"{name}.npz").exists()


def real_preds(name):
    """A stored prediction set of the real run (results/preds), in the form of flood.load_preds."""
    z = np.load(REAL_PRED / f"{name}.npz", allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"])
    return {str(b): pd.DataFrame({"obs": z["obs"][i], **{f"l{k}": z["pred"][i, :, j] for j, k in enumerate(F.LEADS)}}, index=idx) for i, b in enumerate(z["basins"])}


def real_meta(name):
    return json.loads(str(np.load(REAL_PRED / f"{name}.npz", allow_pickle=False)["meta"]))

COLUMN_TEXT = {
    "aorc_to_midnight": "mm: the AORC rain of the whole local wall-clock day t (00:00 to 24:00 at the gauge, summer time included; the column of script 16, built here by code/subdaily_inputs.py and proved against it by a second route)",
    "aorc_last3h": "mm: the AORC rain of the hours ending 22:00, 23:00 and 24:00 of day t (the last 3 hours before the issue time)",
    "aorc_last6h": "mm: the AORC rain of the hours ending 19:00 to 24:00 of day t (the last 6 hours)",
    "aorc_last12h": "mm: the AORC rain of the hours ending 13:00 to 24:00 of day t (the last 12 hours)",
    "aorc_max1h": "mm: the wettest hour of day t (basin mean of the 1 km cells)",
    "aorc_maxcell1h": "mm: the wettest 1 km cell in any hour of day t; NaN where every hour of the day was filled from ERA5 (max_cell_mm is NaN on filled hours) and then filled with aorc_max1h; raised to aorc_max1h on partly filled days where it sat below it (maxcell_fill_days)",
    "aorc_wet_hours": "count: hours of day t with at least 0.5 mm of basin-mean rain",
    "q_rel": "log1p(q_mm / Q95): the daily mean flow of day t relative to the basin's high-flow level (Q95 of its training years, flood.thresholds); the engine also reads log1p(q_mm) scaled by one global mean and spread",
    "older": "the same six columns one day older: on issue day t the network reads the within-day columns of day t-1 (the first day keeps its own value, as the _older controls of scripts 10, 15 and 16)",
    "fs": "fs1..fs3: the AORC rain of the local days t+1..t+3 for the issue days up to 2000-09-30, the bias-corrected GEFS control-member forecast of the same days from 2000-10-01 (the substituted table of script 18)",
    "hourly_shape": f"AORC's hours before 2002 are daily totals spread over the hours by the shape of older radar, satellite or model products (AORC v1.1 methods, as read by the E1b review); from 2002 they have Stage IV's hourly shape. "
                    f"The training years {F.SPLIT['train'][0][:7]} to {F.SPLIT['train'][1][:7]} {'all lie before 2002' if PRE_2002 else 'straddle 2002'}, the validation years {F.SPLIT['val'][0][:4]}-{F.SPLIT['val'][1][:4]} straddle the change, "
                    f"the test years {F.SPLIT['test'][0][:4]}-{F.SPLIT['test'][1][:4]} lie after it: the networks learned the six columns on hours of another source than the hours they were scored on; the effect on the gain was not measured (what_was_not_done)"}
WP_RULE_TEXT = {"csi": "best pooled onset CSI (the project's rule, as in scripts 16 and 18)", "far_cap": f"highest pooled onset POD whose pooled onset FAR is at most {WP.FAR_CAP:.2f} (the false share the GO rule allows)"}


# ---- figure -----------------------------------------------------------------------------------------------------------------------------------------
def draw(R):
    """Lead 1, test years: onsets warned (top) and median NSE (bottom) per set, the seed mean as a bar with every single seed as a dot; wqa and wqaP as reference lines; the control hollow."""
    FS = 8.5
    S, RG = R["summary_test"]["sets"], R["seed_ranges"]["sets"]
    order = [n for n in ("wqa", "wqaQ", "wqaW", "wqaW_older", "wqaWF_sub", "wqaF_sub", "wqaP") if n in S]
    LABEL = {"wqa": "AORC rain to\nmidnight alone\n(wqa, stored)", "wqaQ": "+ flow over Q95\n(wqaQ)", "wqaW": "+ six within-day\ncolumns (wqaW)", "wqaW_older": "the six columns\none day older\n(control)",
             "wqaWF_sub": "within-day columns\n+ GEFS forecast\n(wqaWF_sub)", "wqaF_sub": "GEFS forecast,\nsubstituted\n(wqaF_sub, stored)", "wqaP": "perfect rain in\nthe slots\n(wqaP, stored)"}
    COL = {"wqa": AQUA, "wqaQ": BLUE, "wqaW": BLUE, "wqaW_older": BLUE, "wqaWF_sub": ORANGE, "wqaF_sub": ORANGE, "wqaP": YELLOW}
    hollow = {"wqaW_older"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 9.0), sharex=True, gridspec_kw=dict(hspace=0.3, top=0.79, bottom=0.235, left=0.08, right=0.98))
    x = np.arange(len(order))
    for ax, key in ((ax1, "onset_pod"), (ax2, "nse_median")):
        for i, n in enumerate(order):
            y = 100 * S[n]["l1"]["onset"]["pod"] if key == "onset_pod" else S[n]["l1"]["nse_median"]
            seeds = RG[n]["l1"]["by_seed"][key]
            seeds = [100 * s for s in seeds] if key == "onset_pod" else seeds
            ax.bar(i, y, width=0.58, color="none" if n in hollow else COL[n], edgecolor=COL[n], linewidth=1.6, alpha=0.9 if n in hollow else 0.75, zorder=2)
            ax.scatter(np.full(len(seeds), i) + np.linspace(-0.12, 0.12, len(seeds)), seeds, s=22, color=INK, zorder=4, linewidths=0)
            ax.text(i, y, f"{y:.0f}%" if key == "onset_pod" else f"{y:.2f}", ha="center", va="bottom", fontsize=FS, color=INK2, zorder=5)
        for n, ls in (("wqa", "--"), ("wqaP", ":")):
            if n in S:
                v = 100 * S[n]["l1"]["onset"]["pod"] if key == "onset_pod" else S[n]["l1"]["nse_median"]
                ax.axhline(v, color=COL[n], linestyle=ls, linewidth=1.2, zorder=1)
        ax.set_xticks(x)
        ax.tick_params(axis="x", length=0)
    ax1.axhline(70, color=MUTED, linewidth=1.0, zorder=1)
    ax1.text(len(order) - 0.6, 71, "GO asks 70% warned (at most 30% false)", fontsize=FS - 1, color=MUTED, ha="right", va="bottom")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("onsets warned 1 day ahead (%)", fontsize=FS + 0.5)
    ax1.set_title("Onsets warned 1 day ahead (test years 2005-2014, trigger 1.0, 12 basins pooled)", fontsize=FS + 1.5)
    ax2.set_ylabel("median NSE over the basins", fontsize=FS + 0.5)
    ax2.set_ylim(0, 1)
    ax2.set_title("Median NSE at 1 day (test years)", fontsize=FS + 1.5)
    ax2.set_xticklabels([LABEL[n] for n in order], fontsize=FS)
    for ax in (ax1, ax2):
        ax.tick_params(axis="y", labelsize=FS)
    import textwrap
    n_new = R["seeds"]
    wrap = lambda s, w: "\n".join(textwrap.fill(p, w) for p in s.split("\n"))
    fig.text(0.08, 0.955, "E4b: within-day AORC rain and Q95-relative flow as inputs of the LSTM with the river gauge", fontsize=12, weight="bold", color=INK, ha="left")
    fig.text(0.08, 0.925, wrap(f"Bars: the seed-averaged forecast; dots: every single training ({NUMBER_WORD.get(len(n_new), len(n_new))} seed{'s' if len(n_new) != 1 else ''} per new set, {' or '.join(NUMBER_WORD.get(c, str(c)) for c in sorted({len(v['seeds']) for v in R['models'].values() if v.get('stored')}))} per stored set). "
                               f"Hollow: the control (no new information about day t). Dashed line: wqa; dotted line: wqaP.\n{R['status']}", 150), fontsize=FS, color=INK2, ha="left", va="top", linespacing=1.4)
    t_wqa = R["summary_test"]["sets"]["wqa"]["l1"]["onset"]                  # every number of the footer comes from the JSON (review 2026-10-29, finding 10)
    pre_2002 = int(R["split"]["train"][1][:4]) < 2002
    review_line = (R.get("provenance") or {}).get("review_line", "this file's independent review is not yet done")
    fig.text(0.08, 0.02, wrap(f"AORC rain is {AORC_WORDING} (hours before 2002 are disaggregated daily totals; {'all training years lie before 2002' if pre_2002 else 'the training years straddle 2002'}, the test years' hours are Stage IV-shaped): "
                              f"the {100 * t_wqa['pod']:.0f}% of wqa carries {HINDSIGHT_WORDING}; the within-day columns share that hindsight channel, and the bound was measured for the daily total, not for them. "
                              f"'Perfect rain in the slots' (wqaP): the AORC rain of the target days given as the forecast, {UPPER_WORDING}. "
                              "GEFS forecast, substituted: trained on the AORC rain of the target days, run on the bias-corrected control-member forecast (script 18). "
                              f"Onsets: target days above Q95 whose issue-day flow was at or below it ({t_wqa['hits'] + t_wqa['misses']} in the test years).\n"
                              f"The team's own checks are not yet done; {review_line}. Written by code/{SCRIPT} from results/{OUT}.json.", 165), fontsize=FS - 1.3, color=INK2, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


if ARGS.figure:                                             # only the figure again, from the results file (no tables, no networks)
    assert RESULTS_FILE.exists(), "--figure needs the results file of a run"
    draw(json.loads(RESULTS_FILE.read_text()))
    print(f"figure written: {FIG_FILE}")
    sys.exit(0)

# ---- 0. data: the daily tables, the within-day columns, the controls ------------------------------------------------------------------------------------
D, B = F.load_data(), F.basins()
ids, thr = list(D), F.thresholds(D)
n_b = len(ids)
assert AO.cache_complete(ids), "the AORC series of the 12 basins are not (all) in data/aorc: code/fetch_aorc.py makes them"
ZONES = SI.zones()
for what, f in (("results/lstm_all.json", F.RES / "lstm_all.json"), ("results/flood_analysis.json", F.RES / "flood_analysis.json")):
    assert f.exists(), f"{what} is missing: this script reads it"
CONF_REAL = json.loads((F.RES / "lstm_all.json").read_text())["config"]
assert set(CONF_REAL) == CONF_KEYS, f"results/lstm_all.json config holds other keys than expected: {sorted(CONF_REAL)}"
FA = json.loads((F.RES / "flood_analysis.json").read_text())


def build_tables():
    """D2: the daily table of every basin with the within-day columns and q_rel joined (subdaily_inputs.add_inputs), aorc_maxcell1h filled where NaN, the older control columns added; and the filled days."""
    D2 = SI.add_inputs(D, thr)
    fill = {}
    for g in ids:
        d = D2[g]
        assert d.index.equals(D[g].index) and set(WD_COLS + ("aorc_to_midnight", "aorc_filled_hours", "q_rel")) <= set(d.columns), f"{g}: subdaily_inputs.add_inputs did not add the expected columns"
        nan = d.aorc_maxcell1h.isna()
        assert (d.aorc_filled_hours[nan] >= 23).all(), f"{g}: aorc_maxcell1h is missing on a day that was not wholly filled from ERA5"
        assert not d.drop(columns=["aorc_maxcell1h"]).isna().any().any(), f"{g}: a column other than aorc_maxcell1h has gaps"
        d["aorc_maxcell1h"] = d.aorc_maxcell1h.where(~nan, d.aorc_max1h)
        low = (d.aorc_maxcell1h < d.aorc_max1h - 1e-6)                # partly filled days: the cell maximum covers the un-filled hours only, so it can sit below the basin-mean wettest hour
        assert (d.aorc_filled_hours[low] > 0).all(), f"{g}: the wettest cell is below the basin-mean wettest hour on a day without filled hours"
        d["aorc_maxcell1h"] = d.aorc_maxcell1h.where(~low, d.aorc_max1h)
        fill[g] = dict(filled_where_nan=[str(x.date()) for x in d.index[nan]], raised_to_max1h_on_partly_filled_days=[str(x.date()) for x in d.index[low]])
        assert (d.aorc_maxcell1h >= d.aorc_max1h - 1e-6).all(), f"{g}: the wettest cell is below the basin-mean wettest hour on some day"
        for c in WD_COLS:
            d[c + "_older"] = d[c].shift(1).bfill()             # day t holds the value of day t-1; the first day keeps its own value (as the _older controls of scripts 10, 15, 16)
        for c in WD_COLS:
            assert np.array_equal(d[c + "_older"].values[1:], d[c].values[:-1]) and d[c + "_older"].values[0] == d[c].values[0], f"{g}: the older column of {c} is not the column shifted one day"
        assert np.isfinite(d[list(BASE_DYN + WD_COLS + OLDER_COLS + ("q_rel",))].values).all(), f"{g}: an input column has gaps"
        assert np.allclose(d.q_rel.values, np.log1p(d.q_mm.values / thr[g])), f"{g}: q_rel is not log1p(q_mm / Q95)"
        assert ((d.aorc_last3h <= d.aorc_last6h + 1e-9) & (d.aorc_last6h <= d.aorc_last12h + 1e-9) & (d.aorc_last12h <= d.aorc_to_midnight + 1e-9)).all(), f"{g}: window sums out of order"
    return D2, fill


D2, FILL = build_tables()
N_FILL, N_RAISED = sum(len(v["filled_where_nan"]) for v in FILL.values()), sum(len(v["raised_to_max1h_on_partly_filled_days"]) for v in FILL.values())
FILL_SHORT = {g: v for g, v in FILL.items() if v["filled_where_nan"] or v["raised_to_max1h_on_partly_filled_days"]}
print(f"tables: {n_b} basins, {len(D2[ids[0]])} days; aorc_maxcell1h filled with aorc_max1h on {N_FILL} wholly filled basin-days and raised to it on {N_RAISED} partly filled ones: "
      + "; ".join(f"{g}: {', '.join(v['filled_where_nan'])} | raised {', '.join(v['raised_to_max1h_on_partly_filled_days']) or 'none'}" for g, v in FILL_SHORT.items()), flush=True)


def add_forecast_columns():
    """fs1..fs3 of script 18 on this script's tables: the AORC rain of the local day t+k (aorc_to_midnight shifted) up to the issue day 2000-09-30, the bias-corrected GEFS forecast from 2000-10-01
    (data/gefs/basins, made by code/gefs_to_basins.py for script 18; read, never rebuilt). The AORC values are checked against the ones script 18's tables hold."""
    import gefs_to_basins as GB
    assert GB.RECORD.exists() and all(GB.CSV(g).exists() for g in ids), "the GEFS basin tables of code/gefs_to_basins.py (data/gefs/basins/*.csv, results/gefs_basins.json) are missing: script 18's data step makes them"
    GBR = json.loads(GB.RECORD.read_text())
    sub_from = pd.Timestamp(F.SPLIT["val"][0])
    rec = dict(what=COLUMN_TEXT["fs"], substituted_from_issue_day=str(sub_from.date()), source="data/gefs/basins/gefs_c00_<gauge_id>.csv (code/gefs_to_basins.py), columns fc_bc_1..3; results/gefs_basins.json",
               bias_factor_fitted_on_target_days=GBR["bias"]["fitted_on_target_days"], checks={})
    assert GBR["bias"]["fitted_on_target_days"] == list(F.SPLIT["val"]), "the bias factor of the forecast was not fitted on the validation years"
    for g in ids:
        assert GBR["basins"][g]["time_zone"] == ZONES[g], f"{g}: the time zone of the forecast tables differs from subdaily_inputs.zones()"
        c, d = GB.load_columns(g), D2[g]
        assert c.index[0] <= sub_from and c.index[-1] == d.index[-1] and np.isfinite(c.drop(columns="init_utc_date").values).all(), f"{g}: the forecast table does not cover the validation and test years without gaps"
        n_cmp = 0
        for k in F.LEADS:
            p = d.aorc_to_midnight.shift(-k)                        # the AORC rain of the local day t+k, dated by the issue day t (NaN on the last k days, which lie after sub_from)
            pk = p.reindex(c.index)
            ok = pk.notna().values
            assert np.allclose(c[f"aorc_{k}"].values[ok], pk.values[ok], rtol=0, atol=2e-6), f"{g}: the AORC rain of day t+{k} in script 18's table is not this script's aorc_to_midnight shifted by {k}"
            n_cmp += int(ok.sum())
            fs = p.copy()
            fs.loc[sub_from:] = c[f"fc_bc_{k}"].reindex(d.index).loc[sub_from:].values
            assert fs.notna().all(), f"{g}: fs{k} has gaps"
            d[f"fs{k}"] = fs.values
        rec["checks"][g] = dict(aorc_values_compared_with_script_18_table=n_cmp, equal_within_2e_6_mm=True)
    return rec


# ---- 1. the sets ---------------------------------------------------------------------------------------------------------------------------------------
SETS = {"wqaQ": dict(cols=("q_rel",), control=False, role="test", what="wqa + q_rel = log1p(q_mm / Q95): the flow of day t relative to the basin's own high-flow level (isolates the scaling change; the flow itself is an input of every set)"),
        "wqaW": dict(cols=WD_COLS, control=False, role="test", what="wqa + the six within-day columns of the issue day (the test)"),
        "wqaW_older": dict(cols=OLDER_COLS, control=True, role="control", what="wqa + the same six columns one day older: on issue day t the network reads the within-day columns of day t-1 (the same parameters, no new information about day t)")}
have_seed = lambda name, seed: (F.PRED / f"lstm_{name}_s{seed}.npz").exists() and (MODELS / f"lstm_{name}_s{seed}.pt").exists()        # the files of THIS run's folder (a smoke test looks at results/quick only)
FORECAST_TABLE = None
if WITH_FORECAST or (LOAD and all(have_seed("wqaWF_sub", s) for s in SEEDS)):                   # the optional set: trained with --with-forecast, scored whenever its files exist
    FORECAST_TABLE = add_forecast_columns()
    SETS["wqaWF_sub"] = dict(cols=WD_COLS + FS_COLS, control=False, role="test", what="wqaW + the substituted forecast columns fs1..fs3 of script 18 (trained on the AORC rain of the target days, stopped and scored with the bias-corrected GEFS forecast); compared with the stored lstm_wqaF_sub")
for v in SETS.values():
    v["dyn"] = BASE_DYN + v["cols"]                                                              # the daily input columns of each set, in the order the network reads them
    v["n_in"] = len(v["dyn"]) + 1 + len(F.STATIC)                                                # + the flow channel + the basin properties
STORED = {"wqa": dict(script="16_aorc_rain_to_midnight.py", role="base", what="the LSTM with the river gauge and the AORC rain to midnight (E1b): the base of every comparison here"),
          "wqaF_sub": dict(script="18_forecast_input.py", role="forecast", what="wqa + the substituted GEFS forecast columns (E2a): the base of wqaWF_sub"),
          "wqaP": dict(script="18_forecast_input.py", role="ceiling", what="wqa + the AORC rain of the three target days in the slots: a perfect rain forecast, the ceiling (" + UPPER_WORDING + ")")}
for n, v in STORED.items():
    assert real_have(f"lstm_{n}"), f"the stored set lstm_{n} ({v['script']}) was not found in results/preds: this script compares with it"
    meta = real_meta(f"lstm_{n}")
    assert meta.get("config") == CONF_REAL and meta.get("script") == v["script"] and tuple(meta.get("dyn", ())[:len(BASE_DYN)]) == BASE_DYN, f"lstm_{n} is not the set of {v['script']} trained with the settings of results/lstm_all.json"
    v["dyn"], v["seeds"] = tuple(meta["dyn"]), [s for s in range(5) if real_have(f"lstm_{n}_s{s}")]
    assert len(v["seeds"]) >= 3, f"lstm_{n}: fewer than three single-seed sets were found"
SHOWN = list(SETS) + list(STORED)


def save_model(run, name, seed, v):
    """results/models/lstm_<set>_s<seed>.pt: a plain dict (tensors, numbers, lists and text) that torch.load(weights_only=True) reads, enough to forecast again (the form of script 16)."""
    import torch
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save(dict(state_dict={k: t.detach().cpu().clone() for k, t in run.net.state_dict().items()}, cfg=asdict(run.cfg),
                    sc={k: (x.tolist() if isinstance(x, np.ndarray) else float(x)) for k, x in run.sc.items()}, ids=list(run.ids), dyn=list(run.cfg.dyn),
                    split=dict(run.split), best_val_nse=float(run.best_val_nse), n_params=int(run.n_params), input_set=name, script=SCRIPT, seed=seed,
                    history=run.history, seconds=float(run.seconds), cpu_seconds=float(run.cpu_seconds), updates_run=int(run.updates_run), best_update=int(run.best_update),
                    torch_version=str(torch.__version__), quick_mode=F.QUICK, extra_columns=list(v["cols"]), maxcell_fill_days=FILL_SHORT,
                    how_to_reload="cfg = flood.Cfg(**cfg); net = flood.LSTMNet(n_in, cfg); net.load_state_dict(state_dict); sc lists back to numpy arrays; "
                                  "flood.Run(cfg, sc, net, ids, history, seconds, cpu_seconds, updates_run, best_update, best_val_nse, n_params, split).predict(D, ids, static_df); "
                                  f"D needs the columns in dyn: FORCING from flood.load_data(); aorc_to_midnight, the within-day columns and q_rel from code/subdaily_inputs.add_inputs (aorc_maxcell1h filled with aorc_max1h where NaN); "
                                  f"the _older columns are the within-day columns shifted one day later (first day kept); fs1..fs3 as {SCRIPT} builds them from data/gefs/basins"), tmp)
    tmp.replace(f)
    return f


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


def compare_frames(old_, new):
    cols = ["obs"] + [lk(k) for k in F.LEADS]
    a, b = np.stack([old_[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    close = bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
    assert same_nan and close, "the reloaded network does not reproduce the stored forecasts"
    return dict(n_values_compared=int(a.size), n_missing_in_both=int(np.isnan(a).sum()), max_abs_difference=float(np.nanmax(np.abs(a - b))), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)),
                same_missing_days=same_nan, allclose=close, passed=True)


def reload_and_forecast(name, seed):
    """Rebuild the trained network from its file alone, forecast again on the SAME tables and compare with the stored forecasts of the set (same machine: rtol 1e-6)."""
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    run, z = run_from_file(f)
    assert z["input_set"] == name and z["script"] == SCRIPT and z["seed"] == seed, f"{f.name} is not the network of {name}, seed {seed}"
    new, old_ = run.predict(D2, ids, B), F.load_preds(f"lstm_{name}_s{seed}")
    return dict(set=name, seed=seed, model_file=rel(f), size_kb=round(f.stat().st_size / 1024, 1), n_in=int(len(run.cfg.dyn) + 1 + len(run.cfg.static)), n_params=int(z["n_params"]), **compare_frames(old_, new))


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


# ---- 2. training (or reading) every set, SEEDS seeds each -----------------------------------------------------------------------------------------------
if LOAD:
    assert RESULTS_FILE.exists(), "--rescore and --extend need the results file of a training run: run the script without them first"
    old = json.loads(RESULTS_FILE.read_text())
    old_written = time.strftime("%Y-%m-%d %H:%M", time.localtime(RESULTS_FILE.stat().st_mtime))
    conf = old["config"]
    if not F.QUICK:
        assert conf == CONF_REAL, "the settings of the existing sets are not those of results/lstm_all.json: new sets would not be trained with the same settings"
else:
    old = None
    conf = dict(CONF_REAL, updates=40, eval_every=20) if F.QUICK else dict(CONF_REAL)
missing_sets = [(n, s) for n in SETS for s in SEEDS if not have_seed(n, s)]
if RESCORE:
    assert not missing_sets, "--rescore trains nothing, and these are missing: " + ", ".join(f"lstm_{n}_s{s}" for n, s in missing_sets) + ": run --extend first"
need_cpu = bool(missing_sets)
t_ask = time.time()
runs, ENS, costs, files, HIST, trained_now = {}, {}, {}, {}, {}, {}
with F.cpu_lock(SCRIPT) if need_cpu else contextlib.nullcontext():
    waited = time.time() - t_ask
    for name, v in SETS.items():
        runs[name], costs[name], files[name], HIST[name], trained_now[name] = [], {}, {}, {}, []
        for seed in SEEDS:
            if LOAD and have_seed(name, seed):                           # a saved set is read, never made again
                p, meta, rec = F.load_preds(f"lstm_{name}_s{seed}"), F.preds_meta(f"lstm_{name}_s{seed}"), record_of(name, seed)
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
                                                             model_file=mf, maxcell_fill_days=N_FILL, what="one LSTM for all 12 basins, a single seed (E4b)"))
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
                                                   maxcell_fill_days=N_FILL, what="one LSTM for all 12 basins, day-by-day average of the seeds (E4b)"))
        ENS[name] = ens
trained_pairs = [(n, s) for n in SETS for s in trained_now[n]]
# the proof that a saved network forecasts again: one seed per set, rebuilt from its file alone (networks trained on this machine)
reload_check = dict(what="one trained network per set (the first seed) rebuilt from its .pt file alone (flood.LSTMNet + the saved weights, settings and scalers), run on the same tables, and compared with the forecasts stored by save_preds",
                    tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal; same machine (the networks of the stored sets, trained on the Mac, are not reloaded here)",
                    per_set={n: reload_and_forecast(n, SEEDS[0]) for n in SETS})
print("reload check passed: " + ", ".join(f"{n} (max difference {r['max_abs_difference']:.1e}{', bit for bit' if r['bit_for_bit'] else ''})" for n, r in reload_check["per_set"].items()), flush=True)

# ---- 3. causality: the data side (subdaily_inputs.checks), the engine (flood._assert_causal), and the saved networks end to end (made-up values) -------
t_c = time.time()
SI_REC = SI.checks(log=lambda s: print("  [subdaily_inputs.checks] " + s, flush=True))
DATA_SIDE = dict(what=SI_REC["what"], how="code/subdaily_inputs.checks(): per basin the two routes to the daily total agree, the window sums are in order (3 h <= 6 h <= 12 h <= day), every day has 23, 24 or 25 hours, "
                                        "and the made-up-hours test (every hour after the issue time of a cut day x7+100: no row up to the cut day moves, the next day moves) passed; its record is rewritten to "
                                        + rel(SI.OUT / "subdaily_inputs_check.json"),
                 passed=True, basins=SI_REC["basins"], runtime_s=SI_REC["runtime_s"],
                 note="the rain-timing statistic of that diagnostic (issue_day_rain_timing_test_years) is not carried into this file: it is a preliminary diagnostic, not a result of this experiment")


def made_up_values_test(name, seed, n_cuts=3):
    """The made-up-values test on a saved network (the idea of script 18): replace the set's new columns (and, in a second round, every input column and the flow) of all days AFTER a cut day c by random
    values, predict again and compare. A forecast issued on or before day c must not change (stored at the target day t+k: array_equal on every lead up to c+k), and the ones issued after c must change (or the test proves nothing)."""
    v = SETS[name]
    run_, _ = run_from_file(MODELS / f"lstm_{name}_s{seed}.pt")
    idx = D2[ids[0]].index
    a, b = F.SPLIT["test"]
    cand = np.flatnonzero((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))[200:-200]
    rng = np.random.default_rng(7)
    cuts = sorted(int(c) for c in rng.choice(cand, n_cuts, replace=False))
    base = run_.predict(D2, ids, B)
    out = dict(set=name, seed=seed, cut_days=[str(idx[c].date()) for c in cuts], rounds={})
    for what, cols in (("the new columns of the set only", list(v["cols"])), ("every input column and the observed flow", sorted(set(run_.cfg.dyn) | {"q_mm"}))):
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


causality = dict(
    what="(1) data side: code/subdaily_inputs.checks() on the real series (data_side); (2) engine: flood.fit runs flood._assert_causal on every network before training (random inputs: a change after day T/2 must not change the output "
         "before it; a trained network exists only if it passed); (3) end to end: the made-up-values test on one SAVED network per set (first seed): the set's new columns (and in a second round every input column and the flow) "
         "of all days after a cut day were replaced by random values (0 to 120 per day); no forecast issued on or before the cut day changed (exactly equal, all three leads, all 12 basins) and the forecasts issued after it did; "
         "(4) the within-day columns are dated by the ISSUE day t and hold hours ending at or before 24:00 of day t, so a forecast issued on t reads only hours up to the issue time (data_side); the _older columns hold day t-1",
    data_side=DATA_SIDE, engine_check="flood._assert_causal at the start of every flood.fit (asserted: a trained network exists only if it passed)",
    made_up_values={n: made_up_values_test(n, SEEDS[0]) for n in SETS}, seconds=round(time.time() - t_c, 1))
print("made-up-values test passed on " + ", ".join(causality["made_up_values"]) + f" ({causality['seconds']:.0f} s)", flush=True)

# ---- 4. every set scored the same way (the stored sets rescored by this code, never copied) ------------------------------------------------------------
PRED = {**ENS, **{n: real_preds(f"lstm_{n}") for n in STORED}}
SEEDP = {**runs, **{n: [real_preds(f"lstm_{n}_s{s}") for s in STORED[n]["seeds"]] for n in STORED}}
SEED_LABELS = {**{n: list(SEEDS) for n in SETS}, **{n: list(STORED[n]["seeds"]) for n in STORED}}
EV = {n: dict(test=F.evaluate(PRED[n], thr), val=F.evaluate(PRED[n], thr, "val"), seeds=[F.evaluate(p, thr) for p in SEEDP[n]]) for n in SHOWN}
assert all(EV[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in SHOWN for part in ("test", "val") for k in F.LEADS), "days without a forecast: the models would not be compared on identical days"
assert all(np.allclose(PRED[n][g].obs.values, D[g].q_mm.values) for n in SHOWN for g in ids), "the observed flow differs between the sets"

med = lambda E, k: E["summary"][lk(k)]["nse_median"]                                                 # median NSE over the basins of one evaluation, at lead k
ons = lambda E, k, c="pod": E["summary"][lk(k)]["onset"][c]                                          # onset warnings pooled over the basins
nse_of = lambda E, k: np.array([E["basins"][g][lk(k)]["nse"] for g in ids])
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None       # lowest and highest single seed (a seed with no value is left out)
div = lambda a, b: a / b if b else float("nan")
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                             # every single seed of a above every single seed of b
pc = lambda x: f"{round(100 * x) + 0}%" if x == x else "no value"
p1 = lambda x: f"{100 * x:.1f}%" if x == x else "no value"
n2 = lambda x: f"{x:.2f}"
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
pv = lambda p: "p not defined" if p != p else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
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
ranges = {n: {lk(k): seed_range(n, k) for k in F.LEADS} for n in SHOWN}
per_basin = {"how_to_read": "test years, one value per basin; the seed average of each set (flood.evaluate); gauge ids are text",
             "basins": {g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g])) for g in ids},
             "nse": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["nse"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "onset_warned": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "onset_false": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["far"] for g in ids} for k in F.LEADS} for n in SHOWN}}
models = {n: dict(saved_as=f"lstm_{n}", n_seeds_scored=len(EV[n]["seeds"]), seeds=SEED_LABELS[n], stored=n in STORED, test=EV[n]["test"]) for n in SHOWN}
for n in SETS:
    models[n]["per_seed"] = [dict(seed=s, **{c: costs[n][s][c] for c in COST}, model_file=files[n][s], history=HIST[n][s], test_nse_median=[med(e, k) for k in F.LEADS],
                                  test_onset_warned=[ons(e, k) for k in F.LEADS], test_onset_false=[ons(e, k, "far") for k in F.LEADS], test_all_days_warned=[e["summary"][lk(k)]["all"]["pod"] for k in F.LEADS])
                             for s, e in zip(SEEDS, EV[n]["seeds"])]
training = [dict(set=n, seed=s, saved_forecasts=f"lstm_{n}_s{s}", model_file=files[n][s], **{c: costs[n][s][c] for c in COST}) for n in SETS for s in SEEDS]


# ---- the checks of report section 4.2 ---------------------------------------------------------------------------------------------------------------
def rule(T, C, k):
    """Set T against set C at lead k. NSE: every single seed of T above every single seed of C, higher in most basins, Wilcoxon p below P_CUT - all three make a clear gain.
    Onsets warned: pooled over the basins, so only the seeds can be compared (the rule of scripts 10, 16 and 18)."""
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
    return dict(first=f"lstm_{A}", second=f"lstm_{C}", lead=k, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=only_a, warned_only_by_second=only_c,
                warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=both + only_a + only_c, sign_test_p=float(binomtest(only_a, only_a + only_c, 0.5).pvalue) if only_a + only_c else float("nan"))


def paired_seeds(sT, sC):
    """The seeds the seed-by-seed difference lists cover: zip() pairs the two sets' seed lists in order and stops at the shorter one (a five-seed set against a three-seed stored set: seeds 0, 1, 2; seeds 3 and 4 are not in the lists)."""
    pairs = list(zip(sT["by_seed"]["seeds"], sC["by_seed"]["seeds"]))
    assert all(a == b for a, b in pairs), "the seed-by-seed differences would pair unequal seed numbers"
    return [a for a, _ in pairs]


READING_HOW = ("by_lead.l<k>: set T (with_set) minus set C (without_set) on the test years at trigger 1.0, seed-averaged forecasts; seed_range_* and every_seed_* use all seeds of each set; difference_by_seed_points and "
               "difference_by_seed pair the same seed number of both sets in order and cover only the seeds both sets have (seeds_paired; {cover}); overlap_of_warned_onsets: the onsets warned by both sets, by only one, by neither")


def reading_how(T, C):
    """READING_HOW with the seeds this pair's lists cover, from the two seed lists (review round 2 of 30 October 2026, should-fix 2)."""
    sT, sC = (ranges[x][lk(1)]["by_seed"]["seeds"] for x in (T, C))
    sp = paired_seeds(ranges[T][lk(1)], ranges[C][lk(1)])
    w = lambda n: NUMBER_WORD.get(n, n)
    return READING_HOW.format(cover=(f"here seeds {', '.join(map(str, sp))}, every seed of both sets" if len(sp) == len(sT) == len(sC) else
                                     f"here seeds {', '.join(map(str, sp))}: lstm_{T} has {w(len(sT))} seeds and lstm_{C} {w(len(sC))}, so the seeds of the larger set beyond these are not in the lists"))


def pair_reading(T, C):
    """Set T minus set C at every lead: onsets warned (pooled over the basins; hits, seed ranges, seed by seed: the same seed number of both sets, in order, over the seeds both sets have - seeds_paired) and the median NSE with the checks of section 4.2."""
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


PAIRS = [("within_day_columns", "wqaW", "wqa", "the worth of the six within-day columns: lstm_wqaW minus lstm_wqa (the same inputs without them; stored, script 16)"),
         ("within_day_against_the_older_control", "wqaW", "wqaW_older", "lstm_wqaW minus lstm_wqaW_older: the same six columns one day older (the same parameters, no new information about day t); the difference is what the hours of day t add"),
         ("older_control_against_wqa", "wqaW_older", "wqa", "is the control flat? lstm_wqaW_older minus lstm_wqa: six more columns that hold nothing about day t (if they are not above the seeds of wqa, the control is flat)"),
         ("q_rel_alone", "wqaQ", "wqa", "the worth of the scaling change alone: lstm_wqaQ (q_rel = log1p(q_mm / Q95) as one extra column) minus lstm_wqa"),
         ("within_day_against_q_rel", "wqaW", "wqaQ", "lstm_wqaW minus lstm_wqaQ: the within-day columns against the scaling change"),
         ("how_far_from_perfect_rain", "wqaW", "wqaP", "lstm_wqaW minus lstm_wqaP (the AORC rain of the target days in the slots: a perfect rain forecast, " + UPPER_WORDING + "): what the within-day rain of day t cannot reach")]
if "wqaWF_sub" in SETS:
    PAIRS += [("within_day_with_the_forecast", "wqaWF_sub", "wqaF_sub", "lstm_wqaWF_sub minus lstm_wqaF_sub (stored, script 18): the within-day columns on top of the substituted forecast"),
              ("forecast_on_top_of_within_day", "wqaWF_sub", "wqaW", "lstm_wqaWF_sub minus lstm_wqaW: the substituted forecast on top of the within-day columns")]
readings = {key: dict(what=what, how_to_read=reading_how(T, C), by_lead=pair_reading(T, C), overlap_of_warned_onsets={lk(k): overlap(T, C, k) for k in F.LEADS}) for key, T, C, what in PAIRS}
paired_onsets = dict(how_to_read="per pair and lead: the onsets of the test years warned by both sets, by only one, by neither (flood.flood_scores' onsets at trigger 1.0; the sign test is a description, the onsets of one storm are not independent)",
                     pairs={key: readings[key]["overlap_of_warned_onsets"] for key in readings})
gain_rule = {T: {f"against_{C}": {lk(k): rule(T, C, k) for k in F.LEADS} for key, T2, C, what in PAIRS if T2 == T} for T in SETS}


def closure_of(name, base, ceil_):
    """The share of the gap from the base set to the ceiling that the set closes, per lead, in median NSE and in onsets warned (the closure table of scripts 15, 16 and 18)."""
    rows = []
    for k in F.LEADS:
        Eb, Ec, En = EV[base]["test"], EV[ceil_]["test"], EV[name]["test"]
        b, c, n = med(Eb, k), med(Ec, k), med(En, k)
        ob, oc, on = (ons(E, k) for E in (Eb, Ec, En))
        rows.append(dict(lead=k, nse_base=b, nse_with=n, nse_ceiling=c, gap=c - b, gain=n - b, share_of_gap_closed=div(n - b, c - b), basins_better_than_base=int((nse_of(En, k) > nse_of(Eb, k)).sum()),
                         median_basin_gain=float(np.median(nse_of(En, k) - nse_of(Eb, k))), onset_warned_base=ob, onset_warned_with=on, onset_warned_ceiling=oc, onset_false_base=ons(Eb, k, "far"),
                         onset_false_with=ons(En, k, "far"), onset_false_ceiling=ons(Ec, k, "far"), onset_gain_points=100 * (on - ob), onset_share_of_gap_closed=div(on - ob, oc - ob),
                         seed_range_nse_with=ranges[name][lk(k)]["nse_median"], seed_range_nse_base=ranges[base][lk(k)]["nse_median"], onset_seed_range_with=ranges[name][lk(k)]["onset_pod"], onset_seed_range_base=ranges[base][lk(k)]["onset_pod"]))
    return dict(set=f"lstm_{name}", base=f"lstm_{base}", ceiling=f"lstm_{ceil_}", by_lead=rows)


closure = {n: closure_of(n, "wqa", "wqaP") for n in SETS}
if "wqaWF_sub" in SETS:
    closure["wqaWF_sub_from_the_forecast_model"] = closure_of("wqaWF_sub", "wqaF_sub", "wqaP")
closure["wqaF_sub"] = closure_of("wqaF_sub", "wqa", "wqaP")


# ---- the verdict of results/flood_analysis.json at three warning levels, and the two operating points of code/warning_policy.py --------------------------
CRIT = FA["verdict"]["criteria"]


def number(text, key, op):
    import re
    m = re.search(rf"{key} {op} ([0-9.]+)", text)
    assert m, f"cannot read '{key} {op}' in the stored criterion: {text}"
    return float(m.group(1))


POD_GO, FAR_GO, CSI_CAUTION = number(CRIT["GO"], "pod", ">="), number(CRIT["GO"], "far", "<="), number(CRIT["CAUTION"], "csi", ">=")
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.10 if F.QUICK else 0.05)]       # warning level as a multiple of the high-flow level (06_flood_analysis.py, scripts 15-18)
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


verdicts = {n: verdict_record(PRED[n], EV[n]["test"], SEED_LABELS[n], EV[n]["seeds"]) for n in SHOWN}
VERDICT_NOTE = {"plain": "trigger 1.0: a warning when the forecast is above the high-flow level itself (the level of every other score in this file)",
                "chosen_on_validation_all_days": "the warning level chosen on the validation years by the best pooled CSI of all high-flow days: the level of the verdicts stored in results/flood_analysis.json",
                "chosen_on_validation_onsets": "the warning level chosen on the validation years by the best pooled CSI of the onsets"}


def operating_points(n):
    """Two warning levels chosen on the VALIDATION years with code/warning_policy.py and scored once on the test years: the CSI rule (best pooled onset CSI; the project's rule) and the far_cap rule (the highest pooled onset
    POD whose pooled onset FAR is at most 0.30 on validation: the false share the GO rule allows; E4a). The test FAR is reported as it comes out: it is not 30% by construction. Every single seed is scored at the same trigger."""
    out = {}
    for k in F.LEADS:
        A_val, A_test = WP.arrays(PRED[n], thr, k, "val"), WP.arrays(PRED[n], thr, k, "test")
        A_seeds = [WP.arrays(p, thr, k, "test") for p in SEEDP[n]]
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
                                                                         by_seed=dict(seeds=SEED_LABELS[n], onset_pod=[s["pod"] for s in seeds], onset_far=[s["far"] for s in seeds])))
        out[lk(k)] = rec
    return out


OPS = {n: operating_points(n) for n in SHOWN}
OPS_NOTE = dict(how_to_read="per set and lead, two triggers chosen on the validation years by code/warning_policy.choose_trigger over its grid (0.50 to 2.00 step 0.05) and scored once on the test years: csi (best pooled onset CSI, the project's rule) and "
                            f"far_cap (highest pooled onset POD with pooled onset FAR at most {WP.FAR_CAP:.2f} on validation; reached=false means no trigger of the grid met the cap on validation and the lowest-FAR trigger is given). "
                            "The test FAR is the one that came out, not the cap. The csi trigger here may differ from verdicts.chosen_on_validation_onsets, which uses the project's level grid 0.30 to 1.50 (both are reported).",
                sets=OPS)


# ---- the onsets with little rain on the target day: the slice of the review of E2a (must_fix[0] (iv)), where the work order expects the gain -------------------------
def onset_frame(P, k):
    """The onsets of the test years as rows: basin, target day, whether the set warned at the fixed level (the days flood.flood_scores counts)."""
    rows = []
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(k) <= thr[g]).loc[a_test:b_test].values
        q = p.loc[a_test:b_test]
        ok = (q.obs.notna() & q[lk(k)].notna()).values
        m = (q.obs > thr[g]).values[ok] & was_low[ok]
        rows.append(pd.DataFrame(dict(gauge=g, target=q.index[ok][m], warned=(q[lk(k)] > thr[g]).values[ok][m])))
    return pd.concat(rows, ignore_index=True)


def slice_facts(k=1):
    OF = {n: onset_frame(PRED[n], k) for n in SHOWN}
    E = OF["wqa"][["gauge", "target"]].copy()
    for n, f in OF.items():
        assert (f.gauge.values == E.gauge.values).all() and (f.target.values == E.target.values).all(), "the onsets differ between the sets"
        E[n] = f.warned.values
    SF = {n: [onset_frame(p, k).warned.values for p in SEEDP[n]] for n in SHOWN}
    assert all(len(f) == len(E) for fs in SF.values() for f in fs), "a single-seed set counts other onsets"
    E["aorc_target"] = [float(D2[r.gauge].aorc_to_midnight.loc[r.target]) for r in E.itertuples()]
    E["daymet_target"] = [float(D2[r.gauge].prcp.loc[r.target]) for r in E.itertuples()]
    E["aorc_issue"] = [float(D2[r.gauge].aorc_to_midnight.loc[r.target - pd.Timedelta(days=k)]) for r in E.itertuples()]
    E["aorc_last6h_issue"] = [float(D2[r.gauge].aorc_last6h.loc[r.target - pd.Timedelta(days=k)]) for r in E.itertuples()]

    def block(mask, rule_):
        S = E[mask.values]
        rec = dict(rule=rule_, n=int(mask.sum()), share_of_onsets=float(mask.mean()), warned={n: float(S[n].mean()) if len(S) else float("nan") for n in SHOWN}, hits={n: int(S[n].sum()) for n in SHOWN},
                   single_seeds={n: lo_hi([float(f[mask.values].mean()) if len(S) else float("nan") for f in SF[n]]) for n in SHOWN})
        for T, C in (("wqaW", "wqa"), ("wqaW", "wqaW_older"), ("wqaQ", "wqa")) + ((("wqaWF_sub", "wqaF_sub"),) if "wqaWF_sub" in SETS else ()):
            rec[f"paired_{T}_vs_{C}"] = dict(both=int((S[T] & S[C]).sum()), only_first=int((S[T] & ~S[C]).sum()), only_second=int((~S[T] & S[C]).sum()), neither=int((~S[T] & ~S[C]).sum()),
                                             gain_points=float(100 * (S[T].mean() - S[C].mean())) if len(S) else float("nan"))
        return rec

    slices = dict(aorc_target_below_5mm=block(E.aorc_target < SMALL_MM, f"AORC rain of the target day below {SMALL_MM:g} mm (the rule of script 18's lead1_facts: results/forecast_input.json, lead1_facts.little_rain_on_the_target_day)"),
                  aorc_target_at_least_5mm=block(E.aorc_target >= SMALL_MM, f"AORC rain of the target day at least {SMALL_MM:g} mm"),
                  daymet_target_below_5mm=block(E.daymet_target < SMALL_MM, f"Daymet rain stamped on the target day below {SMALL_MM:g} mm (the rain table's day ends at about 09:00, so this is mostly the rain of the night before the target day)"),
                  daymet_target_at_least_5mm=block(E.daymet_target >= SMALL_MM, f"Daymet rain stamped on the target day at least {SMALL_MM:g} mm"),
                  aorc_issue_day_below_1mm=block(E.aorc_issue < DRY_MM, f"AORC rain of the issue day below {DRY_MM:g} mm (the onsets the work order says need the forecast or the river's state, not within-day rain)"),
                  aorc_issue_day_at_least_1mm=block(E.aorc_issue >= DRY_MM, f"AORC rain of the issue day at least {DRY_MM:g} mm"),
                  aorc_last6h_of_issue_day_at_least_5mm=block(E.aorc_last6h_issue >= SMALL_MM, f"at least {SMALL_MM:g} mm of AORC rain in the last 6 hours of the issue day (where the within-day columns hold the most)"),
                  aorc_last6h_of_issue_day_below_5mm=block(E.aorc_last6h_issue < SMALL_MM, f"under {SMALL_MM:g} mm of AORC rain in the last 6 hours of the issue day"))
    ref = None
    f18 = F.RES / "forecast_input.json"
    if f18.exists():
        r18 = json.loads(f18.read_text()).get("lead1_facts", {}).get("little_rain_on_the_target_day")
        if r18:
            ref = dict(file="results/forecast_input.json", key="lead1_facts.little_rain_on_the_target_day", n_there=r18["n"], n_here=slices["aorc_target_below_5mm"]["n"], wqa_warned_there=r18["no_slots"],
                       wqa_warned_here=slices["aorc_target_below_5mm"]["warned"]["wqa"], same_count=bool(r18["n"] == slices["aorc_target_below_5mm"]["n"]),
                       same_wqa_share=bool(abs(r18["no_slots"] - slices["aorc_target_below_5mm"]["warned"]["wqa"]) < 1e-9))
            assert ref["same_count"] and ref["same_wqa_share"], "the little-rain slice of script 18 is not reproduced"
    sm, wet = slices["aorc_target_below_5mm"], slices["aorc_target_at_least_5mm"]
    where = dict(what="where the hit gain of wqaW over wqa sits: inside the slice of onsets with under 5 mm of AORC rain on the target day and outside it (hits of the seed-averaged forecasts)",
                 hits_gained_in_the_slice=sm["hits"]["wqaW"] - sm["hits"]["wqa"], hits_gained_outside=wet["hits"]["wqaW"] - wet["hits"]["wqa"], points_in_the_slice=100 * (sm["warned"]["wqaW"] - sm["warned"]["wqa"]),
                 points_outside=100 * (wet["warned"]["wqaW"] - wet["warned"]["wqa"]))
    return dict(what=f"lead {k}, main split: the {len(E)} onsets of the test years sliced by the rain of the target day and of the issue day (AORC rain to midnight, Daymet rain), with the share each set warned of at the fixed level "
                     "(seed-averaged forecasts; single_seeds = lowest and highest single seed), the paired counts, and where the gain of wqaW sits. Read from the saved sets and the tables; nothing is trained",
                n_onsets=int(len(E)), slices=slices, check_against_script_18=ref, where_the_gain_sits=where)


SLICE = slice_facts(1)
sm_ = SLICE["slices"]["aorc_target_below_5mm"]
print(f"little-rain onsets ({sm_['n']} of {SLICE['n_onsets']} with under 5 mm of AORC rain on the target day): warned by wqa {p1(sm_['warned']['wqa'])}, wqaW {p1(sm_['warned']['wqaW'])}, wqaW_older {p1(sm_['warned']['wqaW_older'])}, "
      f"wqaQ {p1(sm_['warned']['wqaQ'])}, wqaP {p1(sm_['warned']['wqaP'])}", flush=True)


# ---- the success rule, fixed before training -----------------------------------------------------------------------------------------------------------
def where_text(w):
    """Where the hit gain of wqaW over wqa sits, set against the expectation of NEXT_EXPERIMENTS_2026-10-28.md (E4b: the gain in the onsets with under 5 mm on the target day), in words computed from the two slices (review 2026-10-29, finding 5)."""
    a, b = w["points_in_the_slice"], w["points_outside"]
    if a > 0 and a >= b:
        return "the gain sits in the little-rain onsets, as NEXT_EXPERIMENTS_2026-10-28.md expected"
    if b > 0 and a <= 0:
        return "the gain sits in the wetter onsets, the opposite of the expectation of NEXT_EXPERIMENTS_2026-10-28.md"
    if b > a > 0:
        return "the gain sits mostly in the wetter onsets, not where NEXT_EXPERIMENTS_2026-10-28.md expected it"
    return "there is no gain in either slice, against the expectation of NEXT_EXPERIMENTS_2026-10-28.md"


expectation_met = lambda w: bool(w["points_in_the_slice"] > 0 and w["points_in_the_slice"] >= w["points_outside"])


def success_record():
    r1 = {x: ranges[x][lk(1)]["onset_pod"] for x in ("wqaW", "wqa", "wqaW_older", "wqaQ", "wqaP")}
    parts = dict(every_seed_of_wqaW_above_every_seed_of_wqa=above(r1["wqaW"], r1["wqa"]), every_seed_of_wqaW_above_every_seed_of_the_older_control=above(r1["wqaW"], r1["wqaW_older"]))
    extra = dict(control_flat=not above(r1["wqaW_older"], r1["wqa"]), control_below_wqa_in_every_seed=above(r1["wqa"], r1["wqaW_older"]), every_seed_of_wqaQ_above_every_seed_of_wqa=above(r1["wqaQ"], r1["wqa"]),
                 every_seed_of_wqaW_above_every_seed_of_wqaQ=above(r1["wqaW"], r1["wqaQ"]))
    w = SLICE["where_the_gain_sits"]
    return dict(rule=success_rule_now(), lead=1, met=bool(all(parts.values())), **parts, **extra, n_seeds={x: ranges[x][lk(1)]["n_seeds"] for x in r1},
                onset_warned={x: ons(EV[x]["test"], 1) for x in r1}, onset_false={x: ons(EV[x]["test"], 1, "far") for x in r1}, seed_ranges=r1,
                expected_where=dict(what="NEXT_EXPERIMENTS_2026-10-28.md (E4b) and the work order expect the gain in the onsets with under 5 mm of rain on the target day (results/review_forecast_input.json, must_fix[0] (iv)); "
                                         "this is a description, not part of the rule; expectation_met and reading say whether it held",
                                    slice_n=SLICE["slices"]["aorc_target_below_5mm"]["n"], gain_points_in_the_slice=w["points_in_the_slice"], gain_points_outside=w["points_outside"],
                                    hits_gained_in_the_slice=w["hits_gained_in_the_slice"], hits_gained_outside=w["hits_gained_outside"], expectation_met=expectation_met(w), reading=where_text(w)))


E4B = success_record()

# did the code of this script, run on the stored forecasts, reproduce what scripts 16 and 18 stored for them? (the stored sets are the real ones also in a smoke test)
stored_check = {}
for n, f in (("wqa", F.RES / "aorc_rain_to_midnight.json"), ("wqaF_sub", F.RES / "forecast_input.json"), ("wqaP", F.RES / "forecast_input.json")):
    if not f.exists():
        notes.append(f"{rel(f)} was not found: the numbers this script computes for lstm_{n} were not compared with the ones stored there.")
        continue
    s = json.loads(f.read_text())["summary_test"]["sets"][n]
    diffs = []
    for k in F.LEADS:
        mine, there = summary_test[n][lk(k)], s[lk(k)]
        for a, b in ((there["nse_median"], mine["nse_median"]), (there["onset"]["pod"], mine["onset"]["pod"]), (there["onset"]["far"], mine["onset"]["far"]), (there["onset"]["hits"], mine["onset"]["hits"]), (there["all"]["csi"], mine["all"]["csi"])):
            if a is not None and b == b:
                diffs.append(abs(a - b))
    stored_check[n] = dict(file=rel(f), key=f"summary_test.sets.{n}", values_compared=len(diffs), max_abs_difference=float(max(diffs)) if diffs else None, all_equal=bool(diffs and max(diffs) < 1e-9))
    assert stored_check[n]["all_equal"], f"the numbers stored for lstm_{n} in {f.name} are not reproduced by this script's code"

# ---- 5. the sentence --------------------------------------------------------------------------------------------------------------------------------------
def closed_text(share, gain, gap):
    """How much of the gap from the base to the ceiling a set closes, in words that survive a negative gain or a gap of zero."""
    if not gap > 0:
        return "no gap to close"
    if gain <= 0:
        return "none of it"
    return f"{pc(share)} of it" if share <= 1 else f"all of it and more ({pc(share)})"


n_on =EV["wqaW"]["test"]["summary"]["l1"]["onset"]["hits"] + EV["wqaW"]["test"]["summary"]["l1"]["onset"]["misses"]
tW, tA, tO, tQ, tP = (EV[x]["test"]["summary"]["l1"]["onset"] for x in ("wqaW", "wqa", "wqaW_older", "wqaQ", "wqaP"))
rW, rA, rO = (ranges[x]["l1"] for x in ("wqaW", "wqa", "wqaW_older"))
vW = verdicts["wqaW"]["l1"]
X = ("the rain of the last 3, 6 and 12 hours of the issue day, its wettest hour (basin mean and wettest 1 km cell) and its number of wet hours (hourly AORC rain, " + AORC_WORDING + ") as six extra inputs "
     "beside the AORC rain to midnight to the LSTM with the river gauge")
outcome = (f"Given {X}, onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone ({HINDSIGHT_WORDING}) to {pc(tW['pod'])} (single trainings {span(rW['onset_pod'])}; the onsets of one storm are not independent): "
           f"{100 * (tW['pod'] - tA['pod']):+.1f} points over that model ({tW['hits']} against {tA['hits']} hits of {n_on}) and {100 * (tW['pod'] - tO['pod']):+.1f} points over the control with the same six columns one day older ({pc(tO['pod'])}); "
           f"the share of false warnings among the onset warnings is {pc(tW['far'])} against {pc(tA['far'])} (the control: {pc(tO['far'])}); the median NSE goes from {n2(med(EV['wqa']['test'], 1))} to {n2(med(EV['wqaW']['test'], 1))} "
           f"(the seed-averaged forecast; single trainings {span(rW['nse_median'], n2)}); the verdict of results/flood_analysis.json is {vW['plain']['onset']['verdict']} for the onsets and {vW['plain']['all']['verdict']} for all high-flow days.")
cl = closure["wqaW"]["by_lead"][0]
opW, opA, opO = (OPS[x]["l1"]["rules"]["far_cap"] for x in ("wqaW", "wqa", "wqaW_older"))
cvW, cvA = verdicts["wqaW"]["l1"]["chosen_on_validation_onsets"], verdicts["wqa"]["l1"]["chosen_on_validation_onsets"]
ov1 = readings["within_day_columns"]["overlap_of_warned_onsets"]["l1"]
wet_ = SLICE["slices"]["aorc_target_at_least_5mm"]
n_s, n_st = ranges["wqaW"]["l1"]["n_seeds"], ranges["wqa"]["l1"]["n_seeds"]
medA_, medO_, medQ_ = (med(EV[x]["test"], 1) for x in ("wqa", "wqaW_older", "wqaQ"))
cap_note = lambda op: "" if op["reached"] else ", cap not reached on validation"
cmp_word = lambda a, b: "lower" if a < b else "higher" if a > b else "the same"
d_op = 100 * (opW["test"]["onset"]["pod"] - opO["test"]["onset"]["pod"])                 # wqaW minus the control at the 30%-false point, in points (review 2026-10-29, must-fix): 'nothing' when within one point
op_text = ("at this operating point the six columns add nothing over the control" if abs(d_op) <= 1.0 + 1e-9
           else f"at this operating point the six columns {'add' if d_op > 0 else 'lose'} {abs(d_op):.1f} points against the control")
pct_cap = f"{100 * WP.FAR_CAP:.0f}%"
shape_years = f"all training years, {TRAIN_A[:4]} to {TRAIN_B[:4]}, lie before 2002" if PRE_2002 else f"the training years {TRAIN_A[:4]} to {TRAIN_B[:4]} straddle 2002"
outcome_caveat = (
    f"The success rule fixed before training ({SUCCESS_RULE.split(':')[0]}) is {'met' if E4B['met'] else 'not met'}: the {NUMBER_WORD.get(n_s, n_s)} single training{'s' if n_s != 1 else ''} with the within-day columns "
    f"span{'' if n_s != 1 else 's'} {span(rW['onset_pod'], p1)} against "
    f"{span(rA['onset_pod'], p1)} with the AORC rain to midnight alone and {span(rO['onset_pod'], p1)} for the older control; the control is {'flat' if E4B['control_flat'] else 'not flat'} on onsets warned "
    f"({pc(tO['pod'])} against {pc(tA['pod'])}) and {'noisier' if tO['far'] > tA['far'] else 'not noisier'} ({pc(tO['far'])} false against {pc(tA['far'])}; median NSE {n2(medO_)} against {n2(medA_)}); "
    f"{ov1['warned_only_by_first']} onsets are warned only with the within-day columns against {ov1['warned_only_by_second']} only without them. "
    f"The Q95-relative flow alone (wqaQ) gives {pc(tQ['pod'])} (single trainings {span(ranges['wqaQ']['l1']['onset_pod'])}) with a {cmp_word(medQ_, medA_)} median NSE ({n2(medQ_)} against {n2(medA_)}) "
    f"and {pc(tQ['far'])} false warnings (against {pc(tA['far'])}). "
    f"In the {sm_['n']} onsets with under 5 mm of AORC rain on the target day the share warned goes from {pc(sm_['warned']['wqa'])} to {pc(sm_['warned']['wqaW'])} (older control {pc(sm_['warned']['wqaW_older'])}), "
    f"in the {wet_['n']} others from {pc(wet_['warned']['wqa'])} to {pc(wet_['warned']['wqaW'])}: {where_text(SLICE['where_the_gain_sits'])}. "
    f"The gap in onsets warned from the AORC rain to midnight ({pc(tA['pod'])}) to the perfect rain forecast in the slots (wqaP, {pc(tP['pod'])}, {UPPER_WORDING}) is {100 * (cl['onset_warned_ceiling'] - cl['onset_warned_base']):.1f} points; "
    f"the six columns move the share by {cl['onset_gain_points']:+.1f} points and close {closed_text(cl['onset_share_of_gap_closed'], cl['onset_gain_points'], cl['onset_warned_ceiling'] - cl['onset_warned_base'])}; "
    f"in median NSE the gap is {cl['gap']:.3f} and the move {cl['gain']:+.3f} ({closed_text(cl['share_of_gap_closed'], cl['gain'], cl['gap'])}). "
    f"At the warning level chosen on the validation years for onsets ({cvW['trigger']:.2f} against {cvA['trigger']:.2f}) wqaW warns of {pc(cvW['onset']['pod'])} with {pc(cvW['onset']['far'])} false against {pc(cvA['onset']['pod'])} with {pc(cvA['onset']['far'])} false; "
    f"at the {pct_cap}-false operating point (the trigger chosen on the validation years so that at most {pct_cap} of the onset warnings are false: {opW['trigger']:.2f}{cap_note(opW)}) it warns of "
    f"{pc(opW['test']['onset']['pod'])} with {pc(opW['test']['onset']['far'])} false on the test years, against {pc(opA['test']['onset']['pod'])} with {pc(opA['test']['onset']['far'])} false for the AORC rain to midnight alone (trigger {opA['trigger']:.2f}{cap_note(opA)}) "
    f"and {pc(opO['test']['onset']['pod'])} with {pc(opO['test']['onset']['far'])} false for the older control (trigger {opO['trigger']:.2f}{cap_note(opO)}): {op_text}. "
    f"AORC is {AORC_WORDING}: the within-day columns share the hindsight channel of the AORC rain to midnight (one factor per gauge day); the bound of that caveat ({HINDSIGHT_WORDING}) was measured for the daily total, "
    f"not for these columns; they are a bound on what a real-time hourly rain product could add, not a service. "
    f"The networks learned the within-day columns on hours that are daily totals spread by older products ({shape_years}) and were scored on Stage IV-shaped hours; what this does to the gain was not measured. "
    f"Every number of the new sets is a {NUMBER_WORD.get(n_s, n_s)}-seed number; the stored sets have {NUMBER_WORD.get(n_st, n_st)} seeds.")
if "wqaWF_sub" in SETS:
    tWF, tF = EV["wqaWF_sub"]["test"]["summary"]["l1"]["onset"], EV["wqaF_sub"]["test"]["summary"]["l1"]["onset"]
    rdF = readings["within_day_with_the_forecast"]
    ovF = {k: rdF["overlap_of_warned_onsets"][lk(k)] for k in F.LEADS}
    rules_F = [rdF["by_lead"][lk(k)]["onsets"]["reading"] for k in F.LEADS]
    rule_text = f"'{rules_F[0]}' at every lead" if len(set(rules_F)) == 1 else "; ".join(f"'{r}' at {k} day{'s' if k != 1 else ''}" for k, r in zip(F.LEADS, rules_F))
    fewer = [ovF[k]["warned_only_by_first"] < ovF[k]["warned_only_by_second"] for k in (2, 3)]
    more = [ovF[k]["warned_only_by_first"] > ovF[k]["warned_only_by_second"] for k in (2, 3)]
    paired_text = ("by the paired counts the within-day columns cost onsets at 2 and 3 days" if all(fewer) else "by the paired counts the within-day columns add onsets at 2 and 3 days" if all(more)
                   else "the paired counts at 2 and 3 days point both ways")                       # the paired-onset reading beside the shares (review 2026-10-29, note 11)
    only_text = lambda k: f"{ovF[k]['warned_only_by_first']} onsets warned only with the within-day columns against {ovF[k]['warned_only_by_second']} only without them"
    outcome_caveat += (f" With the substituted GEFS forecast beside them (wqaWF_sub against the stored wqaF_sub) onsets warned 1 day ahead go from {pc(tF['pod'])} to {pc(tWF['pod'])} "
                       f"(single trainings {span(ranges['wqaWF_sub']['l1']['onset_pod'])}; false {pc(tWF['far'])} against {pc(tF['far'])}; median NSE {n2(med(EV['wqaWF_sub']['test'], 1))} against {n2(med(EV['wqaF_sub']['test'], 1))}; "
                       f"{only_text(1)}), at 2 and 3 days " + " and ".join(f"from {pc(ons(EV['wqaF_sub']['test'], k))} to {pc(ons(EV['wqaWF_sub']['test'], k))} ({only_text(k)})" for k in (2, 3))
                       + f": by the seed rule {rule_text}; {paired_text}.")
outcome_full = outcome + " " + outcome_caveat

# ---- status, integrity, timing ----------------------------------------------------------------------------------------------------------------------------
review = json.loads(REVIEW_FILE.read_text(encoding="utf-8")) if REVIEW_FILE.exists() and not F.QUICK else None       # the independent review record: read, never written here (as code/20_warning_policy.py does)
rounds = sorted(int(k[6:]) for k in (review or {}) if re.fullmatch(r"round_\d+", k))      # 2026-10-30: the latest round of the record speaks (round_N with the highest N; the top level is round 1)
rv = review[f"round_{rounds[-1]}"] if rounds else review
review_covers = bool(review) and RESCORE                    # a review covers the saved sets it was made on; a pass that trains (plain or --extend) makes sets no review has seen
# 2026-10-30: a rescore is covered only while the stored comparison sets are the files the review recomputed its numbers with. A stored-set file that changed since the earlier
# results file (seeds 3 and 4 of scripts 16 and 18 were added on 30 October 2026), or a change an earlier pass found under the same review record, means the review predates this pass's numbers.
STORED_NOW = {rel(F.RES / "preds" / f"lstm_{n}{s}.npz"): sha(F.RES / "preds" / f"lstm_{n}{s}.npz") for n in STORED for s in [""] + [f"_s{i}" for i in STORED[n]["seeds"]]}
prev_stored = ((old or {}).get("integrity") or {}).get("stored_sets_read") or {}
prev_ir = ((old or {}).get("provenance") or {}).get("independent_review")
same_record = bool(review) and isinstance(prev_ir, dict) and prev_ir.get("record_sha256") == sha(REVIEW_FILE)
n_stored_seeds = lambda files, n: sum(1 for p in files if re.fullmatch(rf"results/preds/lstm_{n}_s\d+\.npz", p))
stored_changed = sorted(set((prev_ir.get("stored_sets_changed_since_the_review") or []) if same_record else [])
                        | ({p for p in set(STORED_NOW) | set(prev_stored) if STORED_NOW.get(p) != prev_stored.get(p)} if prev_stored else set()))
stored_seed_change = {**((prev_ir.get("stored_seed_counts_changed_since_the_review") or {}) if same_record else {}),
                      **({n: [n_stored_seeds(prev_stored, n), n_stored_seeds(STORED_NOW, n)] for n in STORED if n_stored_seeds(prev_stored, n) != n_stored_seeds(STORED_NOW, n)} if prev_stored else {})}
if rounds:      # the latest round covers this rescore while the stored sets are the files it recomputed its numbers with (its stored_sets_reviewed) and no training pass finished after the results file it reviewed
    ssr = rv.get("stored_sets_reviewed") or {}
    stored_changed = sorted(p for p in set(STORED_NOW) | set(ssr) if STORED_NOW.get(p) != ssr.get(p))
    stored_seed_change = {n: [n_stored_seeds(ssr, n), n_stored_seeds(STORED_NOW, n)] for n in STORED if n_stored_seeds(ssr, n) != n_stored_seeds(STORED_NOW, n)}
    rev_written = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", str((rv.get("dates") or {}).get("results_file_written", "")))
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
    review_line = (f"independent review done ({who if review_date in who else who + ', ' + review_date}{'; round ' + str(rounds[-1]) if rounds else ''}): verdict '{rv.get('verdict', '')}', {n_must} must-fix, {n_should} should-fix and {n_note} note{'s' if n_note != 1 else ''}; "
                   f"{mism}; its {'must-fix wording is' if rounds else 'wording findings are'} applied in this pass ({rel(REVIEW_FILE)})")
    if not review_covers:
        review_line = (f"{rel(REVIEW_FILE)} ({review_date}) reviews an earlier training pass, not the networks trained in this pass: a new review is needed before any number is quoted" if not RESCORE else
                       f"{rel(REVIEW_FILE)} ({review_date}) recomputed this file's numbers with the stored comparison sets as they were then; {len(stored_changed)} of their files have changed or are new since"
                        + (" (seeds " + ", ".join(f"lstm_{n} {a} -> {b}" for n, (a, b) in stored_seed_change.items()) + ")" if stored_seed_change else "")
                        + ": the review predates these numbers and a new review is needed before any number is quoted")
else:
    n_must = n_should = n_note = review_date = None
    cnt = {}
    review_line = "this file's independent review is not yet done"
if F.QUICK:
    status = "SMOKE TEST (FLOOD_QUICK=1): one seed, 40 updates, results/quick: nothing in this file is a result"
elif review_covers:
    status = f"REVIEWED{f' (round {rounds[-1]})' if rounds else ''}: {review_line}. The team's own checks are not yet done."
elif review:
    status = f"NOT YET REVIEWED for this pass: {review_line}; every number of this file is preliminary and must carry that label in any document (HANDOFF.md section 7)"
else:
    status = ("NOT YET REVIEWED: no independent review record (results/review_within_day_inputs.json) exists for this experiment; every number of this file is preliminary and must carry that label in any document "
              "(HANDOFF.md section 7: each new script gets its own independent review that recomputes the headline numbers from results/preds)")
SAVED = {}
for n in SETS:
    for f in [F.PRED / f"lstm_{n}.npz"] + [p for s in SEEDS for p in (F.PRED / f"lstm_{n}_s{s}.npz", MODELS / f"lstm_{n}_s{s}.pt")]:
        SAVED[rel(f)] = sha(f)
made_now = {rel(F.PRED / f"lstm_{n}_s{s}.npz") for n, s in trained_pairs} | {rel(MODELS / f"lstm_{n}_s{s}.pt") for n, s in trained_pairs} | {rel(F.PRED / f"lstm_{n}.npz") for n in SETS if trained_now[n]}
earlier_hashes = (old or {}).get("integrity", {}).get("sha256_of_saved_files", {}) if LOAD else {}
compared = [f for f in SAVED if f not in made_now and f in earlier_hashes]
assert all(earlier_hashes[f] == SAVED[f] for f in compared), "a saved file changed since the earlier results file recorded its hash: " + ", ".join(f for f in compared if earlier_hashes[f] != SAVED[f])
STORED_FILES = {rel(F.RES / "preds" / f"lstm_{n}{s}.npz"): sha(F.RES / "preds" / f"lstm_{n}{s}.npz") for n in STORED for s in [""] + [f"_s{i}" for i in STORED[n]["seeds"]]}
INTEGRITY = dict(how_to_read="sha256 of every prediction set and network file of this script, as they are when this file is written; made_in_this_pass lists the files this pass wrote; every other file was only read. A later pass compares "
                             "the hashes of the files it only reads with these and stops if one differs. stored_sets_read: the files of the stored comparison sets as read (never written by this script).",
                 script_sha256=HASH_NOW, sha256_of_saved_files=SAVED, made_in_this_pass=sorted(made_now), only_read_in_this_pass=len(SAVED) - len(made_now), stored_sets_read=STORED_FILES,
                 compared_with_the_earlier_results_file=(dict(files_compared=len(compared), all_equal=True) if compared else dict(files_compared=0, all_equal=None, why="nothing was read from an earlier pass")) if LOAD else None)
this_pass = dict(kind="training run" if not LOAD else "extension" if EXTEND else "rescore",
                 what=("training run: every set from scratch" if not LOAD else "extension (--extend): the (set, seed) pairs whose files were missing" if EXTEND else "rescore (--rescore): scores, tables, outcome and figure recomputed from the saved sets, nothing trained"),
                 n_trainings=len(trained_pairs), trained=[f"lstm_{n}_s{s}" for n, s in trained_pairs], training_seconds=round(sum(costs[n][s]["seconds"] for n, s in trained_pairs), 1), waited_for_cpu_s=round(waited, 1),
                 seeds=SEEDS, with_forecast_set="wqaWF_sub" in SETS, script_sha256=HASH_NOW, finished_local_time=time.strftime("%Y-%m-%d %H:%M"), torch_threads=int(os.environ.get("FLOOD_THREADS", 1 if F.QUICK else 4)))
first = [dict(p) for p in old.get("passes", []) if p.get("kind") != "rescore"] if LOAD else []
passes = first + [this_pass]
n_trainings, training_seconds = sum(p["n_trainings"] for p in passes), round(sum(p["training_seconds"] for p in passes), 1)
train_passes = [p for p in passes if p.get("kind") == "training run"]
train_pass = train_passes[0] if train_passes else this_pass
n_tp = len(train_pass.get("seeds", SEEDS))
choices = [fixed_before_training(n_tp, train_pass.get("script_sha256", HASH_NOW))]
if len(train_passes) == 1 and not any(p.get("kind") == "extension" for p in passes):
    choices.append(f"One training pass (finished {train_pass['finished_local_time']}) went straight to {NUMBER_WORD.get(n_tp, n_tp)} seeds per set for the {NUMBER_WORD.get(len(SETS), len(SETS))} sets of this script; no --extend pass exists.")
if EXTEND and old and any(s >= 3 for n, s in trained_pairs):
    choices.append(f"Seeds {sorted({s for n, s in trained_pairs if s >= 3})} were added with --extend on {this_pass['finished_local_time'][:10]} after the test-year scores of the earlier pass ({len(first)} pass(es)) had been seen.")
added5 = [f"lstm_{n}" for n in STORED if len(STORED[n]["seeds"]) == 5]
if added5:      # review round 2 (2026-10-30), must-fix: the stored comparison sets gained seeds 3 and 4 after this file's test scores were known
    bs, bv = max(zip(ranges["wqa"][lk(1)]["by_seed"]["seeds"], ranges["wqa"][lk(1)]["by_seed"]["onset_pod"]), key=lambda t: t[1])
    choices.append(f"The stored comparison sets {', '.join(added5[:-1]) + ' and ' + added5[-1] if len(added5) > 1 else added5[0]} gained seeds 3 and 4 on 30 October 2026 on a second machine (scripts 16 and 18, see their SEED_NOTE), "
                   f"after this file's three-seed comparison was known (+4.4 points on 29 October); they are five-seed means since ({100 * (E4B['onset_warned']['wqaW'] - E4B['onset_warned']['wqa']):+.1f} points now); "
                   f"the best wqa seed, above which every wqaW seed must lie, is seed {bs} ({100 * bv:.1f}%), "
                   + ("an original seed, so the success rule's verdict is the same with the three-seed base." if bs < 3 else "an added seed: with the three-seed base the verdict could differ."))
if F.QUICK:
    notes.append("Smoke test (FLOOD_QUICK=1): one seed, 40 updates, outputs in results/quick and figures/quick; the stored sets are the real ones of results/preds, the new sets are not results.")
if FORECAST_TABLE is None:
    notes.append("The set wqaWF_sub (wqaW + the substituted forecast columns fs1..fs3 of script 18) was not trained: run with --with-forecast to add it; the comparison with lstm_wqaF_sub is then made.")
notes.append(f"aorc_maxcell1h was filled with aorc_max1h on {N_FILL} basin-days whose every hour was filled from ERA5 and raised to aorc_max1h on {N_RAISED} partly filled basin-days (maxcell_fill_days); "
             "aorc_filled_hours is the quality flag and is not an input.")
notes.append("The stored sets lstm_wqa, lstm_wqaF_sub and lstm_wqaP were rescored by this script's code from results/preds (checks_against_stored_results proves the same numbers as their own results files); "
             "their networks, trained on the Mac, were not reloaded here (a cross-machine reload is not expected to pass rtol 1e-6).")
notes.append("The simple-model yardstick (ridge regression and the random forest given the within-day columns, rule 2 of the work order) was not run: the engine audit of 2026-10-28 made it optional for the deadline; "
             "the hourly columns would need an extension of features.py.")
notes.append("Quoting. Any document must quote outcome_full (outcome plus outcome_caveat), never outcome alone; every number is a seed-averaged number unless it says 'single trainings'; the status field says whether the experiment has been reviewed.")
NOT_DONE = ([] if review_covers else ["an independent review of this pass that recomputes the headline numbers from results/preds" + (f" ({rel(REVIEW_FILE)} reviews an earlier pass)" if review else f" ({rel(REVIEW_FILE)})")]) \
    + ["the simple-model yardstick given the within-day columns", "block-bootstrap intervals over target-day blocks (13_uncertainty.py pattern)"] \
    + ([f"{NUMBER_WORD[5]} seeds (--seeds 5; this run has {NUMBER_WORD.get(len(SEEDS), len(SEEDS))})"] if len(SEEDS) < 5 else []) \
    + ["a Stage IV test of the saved wqaW networks with the six within-day columns rebuilt from Stage IV hours (the a8 test of the E1b review, which measured the hindsight bound for the daily total only; "
       "data/aorc/stage_iv_hindsight.csv, the Stage IV hours of that review, is the starting point): the hindsight of the within-day columns is not measured",
       f"the effect on the gain of the hourly shape of the training years ({TRAIN_A} to {TRAIN_B}, {'all before 2002' if PRE_2002 else 'straddling 2002'}: AORC's hours there are daily totals spread by older products, "
       "while the test years' hours are Stage IV-shaped): not measured",
       "E4e: the river at the issue time from hourly USGS flow"] \
    + (["wqaWF_sub (--with-forecast)"] if FORECAST_TABLE is None else [])
PROVENANCE = dict(
    written_and_run_by="Claude Code (Fable 5.1) wrote and ran this script under the team lead's direction; AI agents reviewed it",
    independent_review=(dict(record=rel(REVIEW_FILE), record_sha256=sha(REVIEW_FILE), who=rv.get("who"), date=review_date, verdict=rv.get("verdict"), findings=len(rv.get("findings", [])),
                             **({"round": f"round_{rounds[-1]}", "record_script_sha256": rv.get("record_script_sha256")} if rounds else {}),
                             must_fix=n_must, should_fix=n_should, notes=n_note, numerical_mismatches=cnt.get("numerical_mismatches"), values_compared=cnt.get("values_compared"), covers_this_pass=review_covers,
                             stored_sets_changed_since_the_review=stored_changed, stored_seed_counts_changed_since_the_review=stored_seed_change,
                             note=((f"round {rounds[-1]} of the review recomputed this file's numbers with the stored comparison sets as they are now (its stored_sets_reviewed equals integrity.stored_sets_read), with no training pass after the results file it reviewed; "
                                   f"this {this_pass['kind']} applies its must-fix wording (the choices entry on seeds 3 and 4 of the stored sets) and changes no number") if review_covers and rounds else
                                   (f"the review was made on the training pass finished {train_pass['finished_local_time']} (script sha256 {train_pass['script_sha256']}); this {this_pass['kind']} applies its wording findings "
                                    f"(the control at the {100 * WP.FAR_CAP:.0f}%-false point, the hindsight bound measured for the daily total only, the pre-2002 hourly shape of the training years, the control's and wqaQ's false share and NSE, "
                                    "the failed little-rain expectation, seeds_paired, the choices clause, what_was_not_done, the figure footer, this status) and changes no number")
                                   if review_covers else "the review record predates the networks trained in this pass and does not cover them" if not RESCORE else "the review recomputed this file's numbers with the stored comparison sets as they were when it was made; those sets have changed since (stored_sets_changed_since_the_review), so it does not cover this pass"))
                        if review else "not yet done: no results/review_within_day_inputs.json exists; HANDOFF.md section 7 asks for one that recomputes the headline numbers from results/preds"),
    review_line=review_line, team_check="The team's own checks are not yet done (team.json)")

R = dict(
    question="E4b: does the rain of the last hours of the issue day, its peak intensity and its number of wet hours (hourly AORC rain, " + AORC_WORDING + ") - information the daily total hides - and the flow relative to the basin's own Q95 "
             "make the all-river LSTM warn of more of the flood onsets 1 day ahead, and in particular of the onsets with little rain on the target day?",
    script=SCRIPT, written=time.strftime("%Y-%m-%d %H:%M"), status=status, provenance=PROVENANCE, quick_mode=F.QUICK, split=F.SPLIT,
    outcome=outcome, outcome_caveat=outcome_caveat, outcome_full=outcome_full,
    design=dict(what="one LSTM (flood.LSTMNet, hidden 64) for all 12 basins with the river gauge (log1p flow) and the basin properties (flood.STATIC), the settings of results/lstm_all.json chosen there on the validation years; "
                     "every set differs from wqa only in the daily input columns listed; the trainer stops on the mean validation NSE; the test years 2005-2014 are scored once",
                sets={n: dict(saved_as=f"lstm_{n}", role=v["role"], control=v["control"], what=v["what"], extra_columns=list(v["cols"]), dyn=list(v["dyn"]), n_in=v["n_in"], seeds=SEEDS, n_params=(costs[n][SEEDS[0]]["n_params"])) for n, v in SETS.items()},
                stored_sets={n: dict(saved_as=f"lstm_{n}", script=v["script"], role=v["role"], what=v["what"], dyn=list(v["dyn"]), seeds=v["seeds"], rescored_here=True) for n, v in STORED.items()},
                base="lstm_wqa", ceiling="lstm_wqaP", success_rule=success_rule_now(), columns=COLUMN_TEXT, forecast_columns=FORECAST_TABLE,
                order_of_training=list(SETS), seeds=SEEDS, settings_source="results/lstm_all.json, config"),
    config=conf, config_source="results/lstm_all.json, config" + (" (updates and eval_every reduced for the smoke test)" if F.QUICK else ""), seeds=SEEDS, n_seeds_per_set={n: ranges[n]["l1"]["n_seeds"] for n in SHOWN},
    maxcell_fill_days=dict(what="filled_where_nan: days on which aorc_maxcell1h was NaN (every hour of the day filled from ERA5; max_cell_mm is NaN on filled hours) and was set to aorc_max1h; "
                                "raised_to_max1h_on_partly_filled_days: days with some filled hours on which the cell maximum of the un-filled hours sat below the basin-mean wettest hour and was raised to it; per basin, as dates",
                           n_basin_days_filled=N_FILL, n_basin_days_raised=N_RAISED, per_basin=FILL),
    thresholds=thr,
    summary_test=dict(how_to_read="test years, seed average of each set, trigger 1.0, per lead; onset and all = warning tables pooled over the basins; the stored sets rescored here", sets=summary_test),
    validation_used_for_no_choice=dict(label="USED FOR NO CHOICE. Validation years (2000-2005), shown for reference only: no set, setting or stopping point of this script was chosen on these scores (the settings came from 04_lstm_all.py; "
                                             "the trainer stopped each training on its own validation score, so these scores flatter the models). The validation years choose only the warning levels of verdicts and operating_points, which is labelled there.",
                                       summary=summary_val),
    per_basin=per_basin, seed_ranges=dict(how_to_read="lowest and highest single-seed value of the test-year scores, per set and lead (by_seed: the single values, in the order of the seeds)", sets=ranges),
    gain_rule=dict(how_to_read="the checks of report section 4.2 for every pair (set T against set C): NSE clear gain = every seed of T above every seed of C, higher in most basins, Wilcoxon p below 0.05; onsets clear gain = every seed above every seed (pooled POD)",
                   sets=gain_rule),
    readings=readings, paired_onsets=paired_onsets, little_rain_onsets=SLICE, closure=dict(how_to_read="the share of the gap from the base to the ceiling (lstm_wqaP, the perfect rain forecast in the slots) that each set closes, per lead, in median NSE and in onsets warned", sets=closure),
    operating_points=OPS_NOTE, verdicts=dict(criteria=CRIT, criteria_source="results/flood_analysis.json, verdict.criteria", levels=VERDICT_NOTE, level_grid=LEVELS, sets=verdicts),
    e4b_success=E4B, causality=causality, reload_check=reload_check, checks_against_stored_results=stored_check,
    models=models, training=training, integrity=INTEGRITY, choices_made_after_looking_at_test_scores=choices,
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
    """Null leaves of the cleaned record: the NaN of this pass and the nulls read back from the earlier pass's results file (a rescore reads the training records there, where NaN is already null)."""
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
