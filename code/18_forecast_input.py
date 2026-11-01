"""E2: an archived real weather forecast (NOAA GEFSv12 reforecast, control member, rain) as an input of the flood-forecasting LSTM.

The forecast at the end of day t is only as good as what is known about the rain still to come (ROADMAP.md, section 2: 47% of the rain around a flood onset falls on the target day). E1 and E1b gave the model the rain that had
already fallen (the AORC rain of the whole local day t, up to local midnight: set wqa). This script gives it, beside that, the rain FORECAST for the local days t+1, t+2 and t+3 from the 00 UTC run of the UTC date t+1 (the latest run
that exists at the issue time; code/gefs_to_basins.py makes the basin means, their clock proofs, the forecast's own skill and the bias factors: data/gefs/basins, results/gefs_basins.json), as three more daily columns dated by the
ISSUE day t (the engine reads dyn columns up to and including the issue day, so they enter causally by construction; the made-up-values test below proves it on the trained networks).
 E2a, substitution, scored on the usual test years 2005-2014 (the main split). The base is wqa of E1b (FORCING + AORC rain to midnight + river gauge + basin properties, the settings of results/lstm_all.json). Sets, three seeds each:
   wqaP       wqa + p1..p3 (the AORC rain of the local days t+1..t+3: a perfect rain forecast, the rain-only upper bound with the right clock), in every year;
   wqaF_sub   wqa + the table fs1..fs3 that holds p in the training years (issue days up to 2000-09-30) and the bias-corrected GEFS forecast from 2000-10-01 on (validation and test years): the network is trained on perfect rain,
              its stopping point is chosen by the trainer on the validation years WITH the real forecast in the slots (the trainer scores validation from the same table), and it is scored on the test years with the
              real forecast: 'train on the AORC rain of the target days, run on the forecast'; it never sees a forecast in training;
   wqaF_clim  as wqaF_sub with the forecast shuffled within (basin, calendar month) across years from 2000-10-01 on (a forecast with the right climate and no information about the day: the control);
   wqF_sub    as wqaF_sub without the AORC rain-to-midnight column (the forecast on top of the plain gauge model: how the two kinds of information combine).
   The stored sets lstm_wqa, lstm_wq and lstm_wqf (the old perfect-weather upper bound) are scored beside them; the saved networks of lstm_wqa are rebuilt on THIS script's table and must reproduce their stored forecasts.
 E2b, retrained with the forecast as an input, on the satellite-era split of 08_satellite_inputs.py (train 2000-10-01..2008-09-30, validation 2008-10-01..2010-09-30, test 2010-10-01..2014-09-30; flood thresholds from ITS training
   years): wq_s, wqa_s (no forecast), wqaF_s (the bias-corrected forecast in every year), wqaF_clim_s (shuffled), wqaP_s (perfect: the ceiling of this split), three seeds each. Its numbers are comparable only inside it.
 Scoring: flood.evaluate per lead on the respective test years; seed ranges; the checks of report section 4.2; the closure of the gap to the upper bounds; the GO / CAUTION / NOT YET verdict of results/flood_analysis.json; the readings
   value_of_the_forecast (wqaF_sub minus wqa), forecast_vs_climatology_control (wqaF_sub minus wqaF_clim), retrained_vs_substituted (E2b wqaF_s minus wqa_s beside the E2a reading), forecast_without_rain_to_midnight (wqF_sub minus wq),
   how_far_from_perfect (wqaF_sub against wqaP), raw_vs_bias_corrected (the saved wqaF_sub networks given the raw forecast at prediction time; with --raw, three more trainings), substituted_network_given_perfect_rain (the same saved
   wqaF_sub networks given the perfect rain at prediction time: E2a's 'score twice'), stopping_point_chosen_with_the_forecast (wqaF_sub against the saved wqaP networks given the forecast at prediction time); the simple-model yardstick
   (ridge, random forest given wqa + the forecast).
Reads data/gefs/basins and results/gefs_basins.json (gefs_to_basins.py; made now when missing), data/aorc, the settings of results/lstm_all.json, the sets lstm_wq, lstm_wqa, lstm_wqf and their networks (script 16), results/baselines.json,
results/flood_analysis.json. Writes results/forecast_input.json, figures/forecast_input.png, results/models/lstm_<set>_s<seed>.pt, the prediction sets lstm_<set> (+ one per seed) and base_linreg_wqaF, base_rf_wqaF.
Without training: python code/18_forecast_input.py --rescore  recomputes every score, table, note, the outcome and the figure from the saved sets. python code/18_forecast_input.py --extend  trains only the sets whose files are missing (--raw
adds wqaF_sub_raw: the substitution model trained and stopped on the RAW forecast). FLOOD_QUICK=1: one seed, 40 updates, results/quick and figures/quick (the real forecast and AORC tables are read).
"""
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import binomtest, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fetch_aorc as AO
import flood as F
import gefs_to_basins as GB
from common import *
from features import feature_table

use_style()
SCRIPT = "18_forecast_input.py"
OUT = "forecast_input"                                     # results/<OUT>.json, figures/<OUT>.png
t_start = time.time()
HASH_NOW = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()      # the code that runs, hashed once at the start: every pass records it
RESCORE = "--rescore" in sys.argv                          # no training: scores, notes and figure again from the saved prediction sets
EXTEND = "--extend" in sys.argv                            # as --rescore, but trains the sets whose files are missing
RAW_TRAIN = "--raw" in sys.argv                            # also train wqaF_sub_raw: the substitution model trained and stopped on the RAW (not bias-corrected) forecast
assert not (RESCORE and EXTEND), "--rescore trains nothing and --extend trains what is missing: use one of the two"
LOAD = RESCORE or EXTEND                                   # the saved sets and the training records of the results file are read, not made again
D, B = F.load_data(), F.basins()
ids = list(D)
n_b = len(ids)
SEEDS = [0] if F.QUICK else [0, 1, 2, 3, 4]        # seeds 3-4 added 30 October 2026 on the Windows PC: see choices_made_after_looking_at_test_scores (SEED_NOTE)
SEED_NOTE = ("Seeds 3 and 4 of every set of this script were added on 30 October 2026 on a second machine (Windows 11 Home, Intel Core i7-10750H, CPU only, Python 3.11.15, torch 2.2.2+cpu, numpy 1.26.4); "
             "seeds 0-2 were trained on the first machine (macOS, Intel Core i5) and were kept, not retrained. They were added after the test-year scores of seeds 0-2 and the independent review of them were known. "
             "Nothing else changed: the same settings (results/lstm_all.json, key config), inputs, split, stopping rule, warning level and success rule. Every seed average (lstm_<set>.npz) and every seed range of this "
             "script now covers five trainings; the control sets of 04_lstm_all.py and 10_rain_timing.py keep their three. The two extra seeds were planned on 14 October 2026 (code/schedule.py, week 2: five trainings per "
             "model where the NSE decides; promised at the proposal talk of 15 October) for every set alike, whatever the direction of its result; the three-seed numbers remain in the independent review records "
             "(round 1 of results/review_forecast_input.json, numbers.headline_test_years) and in a copy of the three-seed results file, results/archive/forecast_input_3seeds_2026-10-30.json (copied unchanged "
             "from the backup of 30 October 2026, 10:41); passes records only the trainings and times of each pass. five_seed_changes lists what changed with five seeds.")        # appended to choices_made_after_looking_at_test_scores (agent8 audit, section 3)
KS = (1, 2, 3)
SAT_SPLIT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}      # the satellite-era split of 08_satellite_inputs.py
ARMS = {"main": dict(split=F.SPLIT, thr=F.thresholds(D), name="main split", test_years="2005-2014"),
        "sat": dict(split=SAT_SPLIT, thr=F.thresholds(D, split=SAT_SPLIT), name="satellite-era split", test_years="2010-2014")}
P_CUT = 0.05                                               # 'p is small', as in section 4.2 of the report
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")        # the training record of one model (Run.summary)
MODELS = F.RES_OUT / "models"                              # one file per trained network
REAL_MODELS = F.RES / "models"                             # the networks of script 16 (read for the check of the AORC column), also in a smoke test
lk = lambda k: f"l{k}"
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)
notes = []
for what, f in (("results/flood_analysis.json", F.RES / "flood_analysis.json"), ("results/baselines.json", F.RES / "baselines.json"), ("results/lstm_all.json", F.RES / "lstm_all.json"),
                ("results/aorc_rain_to_midnight.json", F.RES / "aorc_rain_to_midnight.json")):
    assert f.exists(), f"{what} is missing: this script reads it (06_flood_analysis.py, 02_baselines.py, 04_lstm_all.py and 16_aorc_rain_to_midnight.py write them; run_all.sh runs them before this script)"
FA = json.loads((F.RES / "flood_analysis.json").read_text())
E1B = json.loads((F.RES / "aorc_rain_to_midnight.json").read_text())
REVIEW_FILE = F.RES / "review_forecast_input.json"
REVIEW = json.loads(REVIEW_FILE.read_text()) if REVIEW_FILE.exists() else None                 # the independent recomputation by a second agent, when it is there

# ---- figure ------------------------------------------------------------------------------------------------------------------------------------------------------------
def draw(R):
    """The ladder of information at leads 1, 2 and 3 (columns), onsets warned above and median NSE below (rows). Filled: a model (the forecast averaged over the trainings, with the lowest to highest single training as a bar); hollow: the control
    (the same forecasts shuffled within basin and calendar month); the satellite-era split, whose scores are its own, is drawn apart on the right in a second colour."""
    FS = 8.0
    main, sat = R["ladder"]["main"]["rungs"], R["ladder"]["sat"]["rungs"]
    extra = R["ladder"]["main"]["extras"][0]
    X_MAIN = list(range(len(main)))
    X_SAT = [len(main) + 0.9 + i for i in range(len(sat))]
    TICK_MAIN = ["rain table", "+ rain to midnight", "+ forecast (substituted)", "perfect rain, 3 days", "perfect weather"]
    TICK_SAT = ["rain table", "+ rain to midnight", "+ forecast (retrained)", "perfect rain, 3 days"]
    assert len(TICK_MAIN) == len(main) and len(TICK_SAT) == len(sat)
    OFF = dict(model=-0.13, control=0.13, extra=0.36)                                                                 # the marks of one rung side by side
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.9), sharey="row", sharex=True)
    fig.subplots_adjust(left=0.065, right=0.992, top=0.85, bottom=0.225, wspace=0.07, hspace=0.12)
    for row, (kind, ylab) in enumerate((("onset", "share of the river rises (onsets) warned"), ("nse", "median NSE (1 = perfect)"))):
        for col, lead in enumerate(F.LEADS):
            ax, key = axes[row, col], lk(lead)
            val = lambda e, pre="": (e[f"{pre}onset_warned"], e[f"{pre}onset_seed_range"]) if kind == "onset" else (e[f"{pre}nse_median"], e[f"{pre}nse_seed_range"])
            lab = lambda y: f"{100 * y:.0f}%" if kind == "onset" else f"{y:.2f}"
            ax.axvspan(X_SAT[0] - .62, X_SAT[-1] + .62, color=SEQ[0], alpha=.28, lw=0, zorder=0)                       # the own-split group, apart
            for xs_, rungs, color in ((X_MAIN, main, BLUE), (X_SAT, sat, ORANGE)):
                xm, ym = [], []
                for x0, r in zip(xs_, rungs):
                    e = r["by_lead"][key]
                    y, rg = val(e)
                    x = x0 + (OFF["model"] if r["control_set"] else 0.0)
                    xm.append(x), ym.append(y)
                    if kind == "onset":                                                                               # the same model at the warning level chosen on the validation years: a short dark bar through the dot
                        yb = e["onset_warned_at_validation_chosen_level"]
                        ax.plot([x - 0.23, x + 0.23], [yb, yb], color=INK, lw=2.2, solid_capstyle="butt", alpha=.9, zorder=6)
                    if rg:                                                                                            # lowest to highest single training
                        ax.plot([x, x], rg, color=color, lw=1.6, solid_capstyle="butt", zorder=2)
                        ax.plot([x, x], [rg[0], rg[0]], color=color, marker="_", ms=5, mew=1.4, zorder=2)
                        ax.plot([x, x], [rg[1], rg[1]], color=color, marker="_", ms=5, mew=1.4, zorder=2)
                    if r["control_set"]:                                                                              # the control of this rung, hollow
                        yc, rc = val(e, "control_")
                        xc = x0 + OFF["control"]
                        if rc:
                            ax.plot([xc, xc], rc, color=color, lw=1.2, alpha=.55, solid_capstyle="butt", zorder=2)
                        ax.plot([xc], [yc], ls="none", marker="o", ms=8, markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.8, zorder=3)
                        if kind == "onset":
                            ybc = e["control_onset_warned_at_validation_chosen_level"]
                            ax.plot([xc - 0.2, xc + 0.2], [ybc, ybc], color=INK, lw=2.2, solid_capstyle="butt", alpha=.9, zorder=6)
                        ax.annotate(lab(yc), (xc, yc), xytext=(5, -3), textcoords="offset points", ha="left", va="top", fontsize=FS - 1, color=INK2)
                ax.plot(xm, ym, color=color, lw=1.3, alpha=.5, zorder=1)
                ax.plot(xm, ym, ls="none", marker="o", ms=8.5, color=color, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
                for x, y in zip(xm, ym):
                    ax.annotate(lab(y), (x, y), xytext=(-5, 5), textcoords="offset points", ha="right", va="bottom", fontsize=FS - 0.5, color=INK2)
            ye, re_ = val(extra["by_lead"][key])                                                                    # the forecast on the plain gauge model: a diamond beside the forecast rung
            xe = X_MAIN[2] + OFF["extra"]
            ax.plot([xe], [ye], ls="none", marker="D", ms=7.5, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
            if kind == "onset":
                ybe = extra["by_lead"][key]["onset_warned_at_validation_chosen_level"]
                ax.plot([xe - 0.2, xe + 0.2], [ybe, ybe], color=INK, lw=2.2, solid_capstyle="butt", alpha=.9, zorder=6)
            if re_:
                ax.plot([xe, xe], re_, color=BLUE, lw=1.2, alpha=.55, zorder=2)
            ax.annotate(lab(ye), (xe, ye), xytext=(5, 3), textcoords="offset points", ha="left", va="bottom", fontsize=FS - 1, color=INK2)
            if kind == "onset":
                ax.axhline(0.70, color=AXIS, lw=1.0, zorder=1)
                if col == 0:
                    ax.annotate("GO asks for 70% warned (and at most 30% false)", (-0.55, 0.70), xytext=(0, 3), textcoords="offset points", ha="left", va="bottom", fontsize=FS - 1.5, color=MUTED)
            ax.set_xlim(-0.6, X_SAT[-1] + 0.62)
            if kind == "onset":
                ax.set_ylim(0, 0.93)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
            else:
                ax.set_ylim(0.0, 1.0)
            ax.set_xticks(X_MAIN + X_SAT, TICK_MAIN + TICK_SAT, rotation=40, ha="right", rotation_mode="anchor", fontsize=FS - 0.5)
            ax.tick_params(labelsize=FS)
            ax.tick_params(axis="x", labelbottom=(row == 1))
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(f"{lead} day{'s' * (lead > 1)} ahead", fontsize=FS + 2)
                ax.annotate(f"main split, test years 2005-2014\n({main[0]['by_lead'][key]['n_onsets']} onsets)", ((X_MAIN[0] + X_MAIN[-1]) / 2, 0.92), ha="center", va="top", fontsize=FS - 1, color=INK2)
                ax.annotate(f"own split, test 2010-2014 ({sat[0]['by_lead'][key]['n_onsets']} onsets;\nnot comparable, weaker base)", ((X_SAT[0] + X_SAT[-1]) / 2, 0.92), ha="center", va="top", fontsize=FS - 1, color=INK2)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=FS)
    n_seed = max(r["by_lead"]["l1"]["n_seeds"] for r in main)
    lg = lambda marker, label, **kw: Line2D([], [], ls="none", marker=marker, ms=8, label=label, **kw)
    h = [lg("o", "model, main split: trained on the AORC rain of those days (an analysis made after the fact), run on the forecast", color=BLUE, markeredgecolor=SURFACE),
         lg("o", "model, satellite-era split: retrained (own scores)", color=ORANGE, markeredgecolor=SURFACE),
         lg("o", "control: same forecasts, shuffled (left: measures harm; right: clean)", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.8),
         lg("D", "forecast on the plain gauge model (no rain to midnight)", color=BLUE, markeredgecolor=SURFACE),
         Line2D([], [], color=INK2, lw=1.6, marker="_", ms=5, mew=1.4, label=f"lowest to highest of {n_seed} training{'s' * (n_seed > 1)}"),
         Line2D([], [], color=INK, lw=2.2, alpha=.9, label="same model at the validation-chosen level (bar, onset panels)")]
    fig.suptitle("A real weather forecast (GEFS control member, rain) as an input: onsets warned and accuracy, 1, 2 and 3 days ahead", fontsize=11, fontweight="bold", color=INK, x=0.012, y=0.985, ha="left")
    fig.legend(handles=h, loc="upper left", ncol=2, frameon=False, fontsize=FS, bbox_to_anchor=(0.012, 0.958), handlelength=1.4, columnspacing=1.8, labelspacing=0.5, handletextpad=0.5)
    fig.text(.012, .008, "Filled: forecast averaged over the trainings; the number beside it is that value (fixed warning level 1.0); the short dark bar through a dot in the onset panels is the same model at the warning level chosen on the validation years. Every model has the river gauge and the 12 basins' properties.\n"
                         "Hollow: the control (same forecasts, shuffled: the right climate, no information about the day). On the left it is NOT a neutral baseline: networks trained on exact slots collapse when given noise, so it measures harm; on the right, trained on shuffled slots, it equals the model without the forecast.\n"
                         "Left (blue): the usual test years 2005-2014. 'GEFS forecast (substituted)': the network is trained on the AORC rain of the days t+1..t+3 (an analysis made after the fact) and run on the bias-corrected forecast.\n"
                         "'perfect rain': the AORC rain of those days given as the forecast; 'perfect weather': an upper bound that cannot be had in practice.\n"
                         "Right (orange, shaded): the shorter satellite-era split (train 2000-2008, test 2010-2014), retrained with the forecast as an input: its scores are its own and are not comparable with the left. The line at 70% is what the GO verdict asks of onsets warned (with at most 30% false).\n"
                         "One member only (no ensemble); 0.25 degree cells over 113 to 1,928 km2 basins; in Eastern summer the last files of the run are up to 15 minutes late at the issue time. AORC rain to midnight is an analysis made after the fact (hours before 2002 are disaggregated daily totals).",
             fontsize=FS - 1.4, color=INK2, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(F.FIG_OUT / f"{OUT}.png", dpi=200)
    plt.close(fig)



if "--figure" in sys.argv:                                 # only the figure again, from the results file (no tables, no networks)
    assert (F.RES_OUT / f"{OUT}.json").exists(), "--figure needs the results file of a run"
    draw(json.loads((F.RES_OUT / f"{OUT}.json").read_text()))
    print(f"figure written: {F.FIG_OUT / (OUT + '.png')}")
    sys.exit(0)

# ---- 0. the forecast tables (code/gefs_to_basins.py) and the AORC rain of the local days -------------------------------------------------------------
GBR = GB.ensure(verbose=False)                              # results/gefs_basins.json and data/gefs/basins/*.csv, made now when missing
assert AO.cache_complete(ids), "the AORC series of the 12 basins are not (all) in data/aorc: code/fetch_aorc.py makes them"
ZONE_OF = {g: GBR["basins"][g]["time_zone"] for g in ids}
DAYS = pd.date_range(*F.PERIOD)                            # the engine's record, 1980-10-01 .. 2014-09-30
DAYS_EXT = pd.date_range(F.PERIOD[0], "2014-10-03")        # and the three days after it (the target days of the last issue days)
T_ISS = pd.date_range("2000-01-01", "2014-09-30")          # the issue days that have a forecast
SUB_FROM = F.SPLIT["val"][0]                               # E2a: from this ISSUE day on the forecast takes the place of the AORC rain in the slots (the validation years begin)
AORC = {g: GB.aorc_local_days(g, ZONE_OF[g], DAYS_EXT) for g in ids}          # the AORC rain of every wall-clock day (code of 16_aorc_rain_to_midnight.py, copied in gefs_to_basins.py)
FCT = {g: GB.load_columns(g) for g in ids}
for g in ids:
    c = FCT[g]
    assert c.index.equals(T_ISS) and np.isfinite(c.drop(columns="init_utc_date").values).all(), f"{g}: the forecast table does not cover 2000-01-01 .. 2014-09-30 or has gaps"
    assert (c.init_utc_date.values == (T_ISS + pd.Timedelta(days=1)).strftime("%Y%m%d").astype(int)).all(), f"{g}: a row does not hold the run of the UTC date t+1"
    for k in KS:
        assert np.allclose(c[f"aorc_{k}"].values, AORC[g].shift(-k).reindex(T_ISS).values, rtol=0, atol=2e-6), f"{g}: the AORC rain of the local day t+{k} in the forecast table is not this script's"
        assert np.allclose(c[f"fc_bc_{k}"].values, c[f"fc_raw_{k}"].values * GBR["bias"]["factors"][g][lk(k)], rtol=0, atol=2e-5), f"{g}: fc_bc is not fc_raw times the bias factor of the record"
assert GBR["bias"]["fitted_on_target_days"] == list(F.SPLIT["val"]), "the bias factor was not fitted on the validation years of the engine"


def shuffled_rows(df, key):
    """The rows of df (issue days x the three leads, one basin) permuted within each calendar month so that no row stays in its own calendar year: the same forecasts, the same climate, no information about the day.
    The three leads of one day move together. key seeds the generator (fixed). Returns the permuted frame (same index) and the permutation of positions per month."""
    out = df.copy()
    yrs, mon = df.index.year.values, df.index.month.values
    for m in range(1, 13):
        pos = np.flatnonzero(mon == m)
        rng = np.random.default_rng([*key, m])
        y = yrs[pos]
        perm = rng.permutation(len(pos))
        for _ in range(500):
            bad = np.flatnonzero(y[perm] == y)
            if not len(bad):
                break
            others = np.setdiff1d(np.arange(len(pos)), bad)
            pool = np.concatenate([bad, rng.choice(others, size=min(len(bad), len(others)), replace=False)])
            perm[pool] = perm[rng.permutation(pool)]
        assert np.array_equal(np.sort(perm), np.arange(len(pos))) and not (y[perm] == y).any(), "the shuffle is not a permutation across years"
        out.iloc[pos] = df.iloc[pos[perm]].values
    return out


SLOT = {"p": ("p1", "p2", "p3"), "fs": ("fs1", "fs2", "fs3"), "fk": ("fk1", "fk2", "fk3"), "fr": ("fr1", "fr2", "fr3"), "fc": ("fc1", "fc2", "fc3"), "fx": ("fx1", "fx2", "fx3")}
SLOT_TEXT = {"p": "p1..p3: the AORC rain of the local days t+1, t+2, t+3 (a perfect rain forecast), in every year",
             "fs": "fs1..fs3: the AORC rain of the local days t+1..t+3 (p) for the issue days up to 2000-09-30, the bias-corrected GEFS control-member forecast of the same days from 2000-10-01 (validation and test years)",
             "fk": "fk1..fk3: p for the issue days up to 2000-09-30; from 2000-10-01 the bias-corrected forecast shuffled within (basin, calendar month) across years, the three leads of a day together (no information about the day)",
             "fr": "fr1..fr3: p up to 2000-09-30; from 2000-10-01 the RAW (not bias-corrected) GEFS forecast",
             "fc": "fc1..fc3: the bias-corrected GEFS control-member forecast of the local days t+1..t+3, in every year of the table (2000-01-01 on)",
             "fx": "fx1..fx3: the bias-corrected forecast shuffled within (basin, calendar month) across years, the three leads together, in every year of the table (the control)"}
TABLE = {"main": {}, "sat": {}}
for bi, g in enumerate(ids):
    base = D[g].assign(aorc_to_midnight=AORC[g].reindex(DAYS).values)
    for k in KS:
        base[f"p{k}"] = AORC[g].shift(-k).reindex(DAYS).values                     # dated by the ISSUE day t: the AORC rain of local day t+k
    assert np.isfinite(base.values).all(), f"{g}: the daily table has gaps"
    fcb, fcr = FCT[g][[f"fc_bc_{k}" for k in KS]], FCT[g][[f"fc_raw_{k}" for k in KS]]
    fcb.columns = fcr.columns = list(KS)
    sh_main = shuffled_rows(fcb.loc[SUB_FROM:], (18, 1, bi))
    m = base.copy()
    for k in KS:
        for pre, src in (("fs", fcb.loc[SUB_FROM:]), ("fr", fcr.loc[SUB_FROM:]), ("fk", sh_main)):
            v = m[f"p{k}"].copy()
            v.loc[SUB_FROM:] = src[k].values
            m[f"{pre}{k}"] = v
    TABLE["main"][g] = m
    s = base.loc[str(T_ISS[0].date()):].copy()
    assert s.index.equals(T_ISS)
    sh_sat = shuffled_rows(fcb, (18, 2, bi))
    for k in KS:
        s[f"fc{k}"], s[f"fx{k}"] = fcb[k].values, sh_sat[k].values
    TABLE["sat"][g] = s
assert all(np.isfinite(TABLE[a][g][list(F.FORCING) + ["aorc_to_midnight"] + [c for t in SLOT.values() for c in t if c in TABLE[a][g].columns]].values).all() for a in TABLE for g in ids), "a table column has gaps"
corr_ = lambda a, b: float(np.corrcoef(a, b)[0, 1])
TABLE_CHECK = dict(
    what="the daily tables the networks read (columns dated by the ISSUE day t), checked: the substituted columns equal p before 2000-10-01 and the bias-corrected forecast from then on; the control columns hold the same forecasts in another order "
         "(the sorted values of each calendar month are the same, no day keeps its own year) and no information about the day (correlation with the AORC rain of that day)",
    main=dict(first_day=str(DAYS[0].date()), last_day=str(DAYS[-1].date()), n_days=len(DAYS), substituted_from_issue_day=SUB_FROM, n_issue_days_with_aorc_rain_in_the_slots=int((DAYS < pd.Timestamp(SUB_FROM)).sum()),
              n_issue_days_with_the_forecast_in_the_slots=int((DAYS >= pd.Timestamp(SUB_FROM)).sum())),
    sat=dict(first_day=str(T_ISS[0].date()), last_day=str(T_ISS[-1].date()), n_days=len(T_ISS)))
for g in ids:
    tm, ts = TABLE["main"][g], TABLE["sat"][g]
    for k in KS:
        assert np.array_equal(tm[f"fs{k}"].loc[:"2000-09-30"].values, tm[f"p{k}"].loc[:"2000-09-30"].values) and np.array_equal(tm[f"fs{k}"].loc[SUB_FROM:].values, FCT[g][f"fc_bc_{k}"].loc[SUB_FROM:].values), f"{g}: fs{k} is not p then the forecast"
        assert np.array_equal(tm[f"fr{k}"].loc[SUB_FROM:].values, FCT[g][f"fc_raw_{k}"].loc[SUB_FROM:].values) and np.array_equal(tm[f"fk{k}"].loc[:"2000-09-30"].values, tm[f"p{k}"].loc[:"2000-09-30"].values), f"{g}: fr{k} or fk{k} is wrong"
        assert np.array_equal(ts[f"fc{k}"].values, FCT[g][f"fc_bc_{k}"].values), f"{g}: fc{k} is not the bias-corrected forecast"
for a, (cols_a, first) in {"main": (("fs", "fk"), SUB_FROM), "sat": (("fc", "fx"), str(T_ISS[0].date()))}.items():
    r_true, r_ctrl, same = [], [], True
    for g in ids:
        t = TABLE[a][g].loc[first:]
        for k in KS:
            r_true.append(corr_(t[f"{cols_a[0]}{k}"].values, t[f"p{k}"].values)), r_ctrl.append(corr_(t[f"{cols_a[1]}{k}"].values, t[f"p{k}"].values))
        for m_ in range(1, 13):                                                              # the same forecasts: the sorted values of each calendar month agree
            sel = t.index.month == m_
            same &= all(np.array_equal(np.sort(t[f"{cols_a[0]}{k}"].values[sel]), np.sort(t[f"{cols_a[1]}{k}"].values[sel])) for k in KS)
    assert same, f"{a}: the control columns do not hold the same values as the forecast columns"
    TABLE_CHECK[a].update(forecast_columns=cols_a[0], control_columns=cols_a[1], control_has_the_same_values_per_calendar_month=bool(same), median_correlation_with_the_aorc_rain_of_the_day=dict(forecast=float(np.median(r_true)), control=float(np.median(r_ctrl))),
                          control_correlation_range=[float(min(r_ctrl)), float(max(r_ctrl))], no_day_keeps_its_own_calendar_year=True)
assert abs(TABLE_CHECK["main"]["median_correlation_with_the_aorc_rain_of_the_day"]["control"]) < 0.1 and TABLE_CHECK["main"]["median_correlation_with_the_aorc_rain_of_the_day"]["forecast"] > 0.5, "the control or the forecast columns do not behave as designed"

# ---- the input sets ------------------------------------------------------------------------------------------------------------------------------
AORC_COL = ("aorc_to_midnight",)
SETS = {  # name -> definition; all with the river gauge and the basin properties; the order is the order of training (the readings of E2a first)
    "wqaF_sub": dict(arm="main", aorc=True, slot="fs", role="test", what="wqa + the forecast, substituted: trained on the AORC rain of t+1..t+3 (an analysis made after the fact), stopped and scored with the bias-corrected GEFS forecast (E2a)"),
    "wqaF_clim": dict(arm="main", aorc=True, slot="fk", role="control", what="as wqaF_sub with the forecast shuffled within (basin, calendar month) across years (E2a control)"),
    "wqaP": dict(arm="main", aorc=True, slot="p", role="ceiling", what="wqa + the AORC rain of the three target days in every year: a perfect rain forecast (E2a, the rain-only upper bound with the right clock)"),
    "wqF_sub": dict(arm="main", aorc=False, slot="fs", role="test", what="the plain gauge model (rain table, no rain to midnight) + the forecast, substituted (E2a)"),
    "wqaF_s": dict(arm="sat", aorc=True, slot="fc", role="test", what="wqa + the bias-corrected forecast in every year, retrained on the satellite-era split (E2b)"),
    "wqa_s": dict(arm="sat", aorc=True, slot=None, role="base", what="wqa (rain to midnight, no forecast) on the satellite-era split (E2b)"),
    "wqaF_clim_s": dict(arm="sat", aorc=True, slot="fx", role="control", what="as wqaF_s with the forecast shuffled within (basin, calendar month) across years, in every year (E2b control)"),
    "wq_s": dict(arm="sat", aorc=False, slot=None, role="reference", what="the plain gauge model (rain table only) on the satellite-era split (E2b)"),
    "wqaP_s": dict(arm="sat", aorc=True, slot="p", role="ceiling", what="wqa + perfect rain of the three target days on the satellite-era split (E2b, the ceiling of this split)"),
}
if RAW_TRAIN:
    SETS["wqaF_sub_raw"] = dict(arm="main", aorc=True, slot="fr", role="variant", what="as wqaF_sub with the RAW (not bias-corrected) forecast in the validation and test years (trained and stopped on it)")
for v in SETS.values():
    v["dyn"] = tuple(F.FORCING) + (AORC_COL if v["aorc"] else ()) + (SLOT[v["slot"]] if v["slot"] else ())
STORED = ("wq", "wqa", "wqf")                                  # sets of earlier scripts, scored here with the same code (main split): the rain table alone, the rain to midnight (script 16), the perfect weather forecast
ARM_OF = {**{n: v["arm"] for n, v in SETS.items()}, **{n: "main" for n in STORED}}
arm_of = lambda n: ARMS[ARM_OF[n]]
RAW_VARIANT = "wqaF_sub_rawpred"                               # the saved wqaF_sub networks given the raw forecast at prediction time (a scored variant, not a training)
PERFECT_VARIANT = "wqaF_sub_perfectpred"                       # the saved wqaF_sub networks given the perfect rain (p) in the validation and test years at prediction time (ROADMAP.md, E2a: 'score twice: with the perfect rain and with the GEFS rain')
STOP_VARIANT = "wqaP_forecastpred"                             # the saved wqaP networks (trained and stopped on perfect rain) given the bias-corrected forecast in the validation and test years at prediction time
DERIVED_NAMES = (RAW_VARIANT, PERFECT_VARIANT, STOP_VARIANT)
for _n in DERIVED_NAMES:
    ARM_OF[_n] = "main"
SPLIT_NAME = {"main": "main split", "sat": "satellite-era split"}

# ---- helpers for scores (the words and numbers of scripts 15 and 16) ------------------------------------------------------------------------------
med = lambda E, k: E["summary"][lk(k)]["nse_median"]                                              # median NSE over the basins of one evaluation, at lead k
ons = lambda E, k, c="pod": E["summary"][lk(k)]["onset"][c]                                       # onset warnings pooled over the basins
nse_of = lambda E, k: np.array([E["basins"][g][lk(k)]["nse"] for g in ids])
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None      # lowest and highest single seed (a seed with no value is left out)
div = lambda a, b: a / b if b else float("nan")
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                          # every single seed of a above every single seed of b
pc = lambda x: f"{round(100 * x) + 0}%" if x == x else "no value"
p1 = lambda x: f"{100 * x:.1f}%" if x == x else "no value"
n3 = lambda x: f"{x:.3f}"
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
yes = lambda b: "yes" if b else "no"
pv = lambda p: "p not defined" if p != p else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
row = lambda E, k: dict(nse_median=med(E, k), nse_mean=E["summary"][lk(k)]["nse_mean"], onset=CUT(E["summary"][lk(k)]["onset"]), all=CUT(E["summary"][lk(k)]["all"]))
NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def evaluate_set(P, arm, part):
    """flood.evaluate of a prediction set on one split part of an arm, with that arm's thresholds (from ITS training years) and split."""
    a = ARMS[arm]
    return F.evaluate(P, a["thr"], part, split=a["split"])


# ---- the simple models given the new input: by the recipe of 02_baselines.py (code/features.py), as 16_aorc_rain_to_midnight.py does it ---------------------------------
BASE = json.loads((F.RES / "baselines.json").read_text())["models"]                        # the settings chosen there on the validation years for the input set wq
RIDGE_ALPHA, RF_SET = BASE["base_linreg_wq"]["settings"]["ridge_alpha"], BASE["base_rf_wq"]["settings"]
assert BASE["base_linreg_wq"]["settings"]["standardised"] and RF_SET["max_features"] == "one third" and RF_SET["random_state"] == 0, "the recipe of 02_baselines.py has changed: read it again"
RF_TREES, RF_JOBS = (8, 1) if F.QUICK else (RF_SET["n_trees"], 4)
MAKE = {"base_linreg_wqaF": lambda: make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
        "base_rf_wqaF": lambda: RandomForestRegressor(RF_TREES, min_samples_leaf=RF_SET["min_samples_leaf"], max_features=1 / 3, n_jobs=RF_JOBS, random_state=RF_SET["random_state"])}
YARD = tuple(MAKE)
SETTINGS = {"base_linreg_wqaF": dict(ridge_alpha=RIDGE_ALPHA, standardised=True), "base_rf_wqaF": dict(n_trees=RF_TREES, min_samples_leaf=RF_SET["min_samples_leaf"], max_features="one third", random_state=RF_SET["random_state"])}
KIND = {"base_linreg_wqaF": "linreg", "base_rf_wqaF": "rf"}
FC_FEATURES = ("fc_day1", "fc_day2", "fc_day3", "fc_3day_total")                           # the extra features: the forecast rain of the three target days and their sum (known at the issue time)


def feature_table_with(d, col=None, fc=None):
    """The features of features.feature_table(d, flow=True) for the input set wq, plus, when col is given, the same rain features (rain of the last three days, sums over 3 to 90 days, the two antecedent-rain indices) computed on the
    extra rain series d[col] (feature_table called a second time with the series in the place of prcp), plus, when fc is given (three column names), the forecast rain of the target days t+1..t+3 and their sum as four features.
    A row without the full history is NaN in every column (as there)."""
    X = feature_table(d, True, 0)
    if col is not None:
        Y = feature_table(d.assign(prcp=d[col]), False, 0)
        X = pd.concat([X, Y[[c for c in Y.columns if c.startswith(("rain_", "wetness_"))]].add_prefix(f"{col}__")], axis=1)
    if fc is not None:
        for name, c in zip(FC_FEATURES[:3], fc):
            X[name] = d[c].values
        X[FC_FEATURES[3]] = d[list(fc)].sum(axis=1).values
    return X.where(X.notna().all(axis=1))


def simple_design(col, fc):
    """Per basin and lead: the feature rows, the complete rows, the rows used for fitting (target day in the training years of the MAIN split) and the flow to predict, on the substituted table of the main split."""
    T = {}
    for g in ids:
        d = TABLE["main"][g]
        X = feature_table_with(d, col, fc)
        ok = X.notna().all(axis=1).values
        for k in F.LEADS:
            T[g, k] = X.values, ok, ok & F.part_mask(d.index, k, "train"), d.q_mm.shift(-k).values
    return T


def features_are_causal(col="aorc_to_midnight", fc=SLOT["fs"], cut=5000):
    """The self-check of features.py on the real tables with the new columns: change EVERYTHING after a cut day (every column, the forecast columns too): no feature row up to the cut may change, and the next one must."""
    for g in ids:
        d = TABLE["main"][g]
        bad = d.copy()
        bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
        A, A2 = feature_table_with(d, col, fc), feature_table_with(bad, col, fc)
        assert A.columns.equals(A2.columns) and A.iloc[:cut + 1].equals(A2.iloc[:cut + 1]), f"{g}: a feature reads data after the issue day"
        assert not np.allclose(A.iloc[cut + 1].values, A2.iloc[cut + 1].values), f"{g}: the check of the features has no power"
        assert (A.isna().all(axis=1) | A.notna().all(axis=1)).all(), f"{g}: a feature row is partly missing"
    return dict(what="features.py's own check on the real main-split tables with the forecast columns: every column of the daily table was replaced after a cut day (day number " + str(cut) + " of the record); the feature rows up to the cut "
                     "did not change and the next row did, in every basin", passed=True, n_basins=len(ids), n_features=int(feature_table_with(TABLE["main"][ids[0]], col, fc).shape[1]))


def forecast_simple(make, T):
    """One model per basin and lead, fitted on the training years; forecasts every issue day that has features (a copy of 02_baselines.forecast())."""
    P = {}
    for g in ids:
        d, out = TABLE["main"][g], {}
        for k in F.LEADS:
            X, ok, fit_rows, y = T[g, k]
            m = make().fit(X[fit_rows], y[fit_rows])
            out[k] = np.full(len(d), np.nan)
            out[k][ok] = np.clip(m.predict(X[ok]), 0, None)
        P[g] = F.frame_from_issue(d, out)
    return P


def recipe_check():
    """With the new columns left out this code must be the code of 02_baselines.py: on the input set wq it must give the forecasts that 02_baselines.py stored as base_linreg_wq (and, outside a smoke test, base_rf_wq)."""
    T0 = simple_design(None, None)
    out = dict(what="the recipe of 02_baselines.py rebuilt in this script (feature_table_with without the new columns, forecast_simple) run on the input set wq: its forecasts against the stored base_linreg_wq and base_rf_wq")
    for n, stored in (("base_linreg_wqaF", "base_linreg_wq"), ("base_rf_wqaF", "base_rf_wq")):
        if n == "base_rf_wqaF" and F.QUICK:
            out["forest_wq_max_abs_difference_from_stored_base_rf_wq"] = "not run in a smoke test (tiny forests)"
            continue
        P, S = forecast_simple(MAKE[n], T0), F.load_preds(stored)
        assert all(np.array_equal(np.isnan(P[g][lk(k)].values), np.isnan(S[g][lk(k)].values)) for g in ids for k in F.LEADS), f"{stored}: the missing days differ"
        diff = float(max(np.nanmax(np.abs(P[g][lk(k)].values - S[g][lk(k)].values)) for g in ids for k in F.LEADS))
        assert diff < 1e-9, f"{stored} is not reproduced by this code (largest difference {diff})"
        out[("ridge" if n == "base_linreg_wqaF" else "forest") + "_wq_max_abs_difference_from_stored_" + stored] = diff
    return out


def fit_simple(names):
    """Ridge regression and random forest for wqa + the forecast (substituted table), one model per basin and lead, fitted on the training years only (where the forecast slots hold the AORC rain of t+1..t+3), the settings of 02_baselines.py for wq."""
    T, out = simple_design("aorc_to_midnight", SLOT["fs"]), {}
    for n in names:
        t0 = time.perf_counter()
        out[n] = (forecast_simple(MAKE[n], T), round(time.perf_counter() - t0, 1))
        print(f"{n}: one model per basin and lead, fitted on the training years in {out[n][1]:.0f} s", flush=True)
    return out


# ---- the networks: saved, reloaded, predicted -----------------------------------------------------------------------------------------------------
def frames_of(run, tab, arm):
    """The forecasts of a trained network on a table, in the form that is stored: {gauge_id: frame by target day over the engine's whole record, with obs}; the satellite-era tables start in 2000, so their frames are put back on the whole record (as 08_satellite_inputs.py does)."""
    p = run.predict(tab, ids, B)
    if arm == "sat":
        p = {g: p[g].reindex(D[g].index).assign(obs=D[g].q_mm) for g in ids}
    return p


def run_from_file(f):
    """A trained network rebuilt from its file alone: (flood.Run, the saved dict)."""
    z = torch.load(f, map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    assert z["state_dict"]["lstm.weight_ih_l0"].shape[1] == n_in and list(cfg.dyn) == z["dyn"], f"{f.name}: the input columns do not fit the weights"
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z


def save_model(run, name, seed):
    """results/models/lstm_<set>_s<seed>.pt: a plain dict (tensors, numbers, lists and text) that torch.load(weights_only=True) reads, enough to forecast again."""
    v = SETS[name]
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save(dict(state_dict={k: x.detach().cpu().clone() for k, x in run.net.state_dict().items()}, cfg=asdict(run.cfg),
                    sc={k: (x.tolist() if isinstance(x, np.ndarray) else float(x)) for k, x in run.sc.items()}, ids=list(run.ids), dyn=list(run.cfg.dyn),
                    split=dict(run.split), arm=v["arm"], best_val_nse=float(run.best_val_nse), n_params=int(run.n_params), input_set=name, script=SCRIPT, seed=seed,
                    history=run.history, seconds=float(run.seconds), cpu_seconds=float(run.cpu_seconds), updates_run=int(run.updates_run), best_update=int(run.best_update),
                    torch_version=str(torch.__version__), quick_mode=F.QUICK,
                    how_to_reload="cfg = flood.Cfg(**cfg); net = flood.LSTMNet(n_in, cfg); net.load_state_dict(state_dict); sc lists back to numpy arrays; "
                                  "flood.Run(cfg, sc, net, ids, history, seconds, cpu_seconds, updates_run, best_update, best_val_nse, n_params, split).predict(D, ids, static_df); "
                                  f"D needs the columns in dyn: FORCING from flood.load_data(), aorc_to_midnight and p1..p3 from the AORC local-day series, the forecast columns ({SLOT_TEXT.get(v['slot'], 'none')}) "
                                  f"built by {SCRIPT} from data/gefs/basins (code/gefs_to_basins.py); the satellite-era sets read the table from 2000-01-01 on"), tmp)
    tmp.replace(f)
    return f


def reload_and_forecast(name, seed):
    """Rebuild the trained network from its file alone, forecast again on the SAME table and compare with the stored forecasts of the set."""
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    run, z = run_from_file(f)
    assert z["input_set"] == name and z["script"] == SCRIPT and z["seed"] == seed and z["arm"] == SETS[name]["arm"], f"{f.name} is not the network of {name}, seed {seed}"
    new, old_ = frames_of(run, TABLE[SETS[name]["arm"]], SETS[name]["arm"]), F.load_preds(f"lstm_{name}_s{seed}")
    return dict(set=name, seed=seed, model_file=rel(f), size_kb=round(f.stat().st_size / 1024, 1), n_in=int(len(run.cfg.dyn) + 1 + len(run.cfg.static)), n_params=int(z["n_params"]), **compare_frames(old_, new))


def compare_frames(old_, new):
    cols = ["obs"] + [lk(k) for k in F.LEADS]
    a, b = np.stack([old_[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    close = bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
    assert same_nan and close, "the reloaded network does not reproduce the stored forecasts"
    return dict(n_values_compared=int(a.size), n_missing_in_both=int(np.isnan(a).sum()), max_abs_difference=float(np.nanmax(np.abs(a - b))), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), same_missing_days=same_nan,
                allclose=close, passed=True)


def real_preds(name):
    """A stored prediction set of the real run (results/preds), also in a smoke test: the check of script 16's networks compares with the forecasts script 16 stored, not with a smoke-test file of the same name in results/quick."""
    z = np.load(F.RES / "preds" / f"{name}.npz", allow_pickle=False)
    idx = pd.DatetimeIndex(z["dates"])
    return {str(b): pd.DataFrame({"obs": z["obs"][i], **{f"l{k}": z["pred"][i, :, j] for j, k in enumerate(F.LEADS)}}, index=idx) for i, b in enumerate(z["basins"])}


def reload_external_wqa(seed):
    """The saved networks of 16_aorc_rain_to_midnight.py (set wqa) given THIS script's table: they must reproduce the forecasts that script stored, which proves that the column aorc_to_midnight built here (code copied from
    script 16, a longer calendar) is the column that trained them."""
    f = REAL_MODELS / f"lstm_wqa_s{seed}.pt"
    run, z = run_from_file(f)
    assert z["script"] == "16_aorc_rain_to_midnight.py" and z["input_set"] == "wqa" and z["seed"] == seed and list(z["dyn"]) == list(F.FORCING) + ["aorc_to_midnight"], f"{f.name} is not the network of script 16, set wqa, seed {seed}"
    return dict(set="wqa (script 16)", seed=seed, model_file=rel(f), **compare_frames(real_preds(f"lstm_wqa_s{seed}"), run.predict(TABLE["main"], z["ids"], B)))


FEATURES = list(feature_table_with(TABLE["main"][ids[0]], "aorc_to_midnight", SLOT["fs"]).columns)
have_seed = lambda name, seed: (F.PRED / f"lstm_{name}_s{seed}.npz").exists() and (MODELS / f"lstm_{name}_s{seed}.pt").exists()      # the files of THIS run's folder (a smoke test looks at results/quick only)
have_simple = lambda n: (F.PRED / f"{n}.npz").exists()
if not RAW_TRAIN and "wqaF_sub_raw" not in SETS and all(have_seed("wqaF_sub_raw", s) for s in SEEDS):                   # the optional variant is scored when its files exist
    SETS["wqaF_sub_raw"] = dict(arm="main", aorc=True, slot="fr", role="variant", what="as wqaF_sub with the RAW (not bias-corrected) forecast in the validation and test years (trained and stopped on it)")
    SETS["wqaF_sub_raw"]["dyn"] = tuple(F.FORCING) + AORC_COL + SLOT["fr"]
    ARM_OF["wqaF_sub_raw"] = "main"


def record_of(name, seed):
    """The training record of a saved network: from the results file of the earlier run when it holds that set, otherwise from the network's own file, which holds the same numbers."""
    for r in old["models"].get(name, {}).get("per_seed", []):
        if r["seed"] == seed:
            return r
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    assert f.exists(), f"lstm_{name}_s{seed}: neither the results file nor a network file holds its training record"
    z = torch.load(f, map_location="cpu", weights_only=True)
    return dict(seed=seed, **{k: z[k] for k in COST}, model_file=rel(f))


old, old_written = None, None
if LOAD:                                                                                   # the settings and training records of the training run
    assert (F.RES_OUT / f"{OUT}.json").exists(), "--rescore and --extend need the results file of a training run: run the script without them first"
    old = json.loads((F.RES_OUT / f"{OUT}.json").read_text())
    old_written = time.strftime("%Y-%m-%d %H:%M", time.localtime((F.RES_OUT / f"{OUT}.json").stat().st_mtime))
    conf = old["config"]
missing_sets = [(n, s) for n in SETS for s in SEEDS if not have_seed(n, s)]
missing_simple = [n for n in YARD if not have_simple(n)]
if RESCORE:
    assert not missing_sets and not missing_simple, ("--rescore trains nothing, and these are missing: " + ", ".join([f"lstm_{n}_s{s}" for n, s in missing_sets] + missing_simple) + ": run --extend first")
old_checks = old.get("yardstick", {}).get("checks") if LOAD else None
run_checks = (not LOAD) or bool(missing_simple) or (EXTEND and not old_checks)
need_cpu = not RESCORE and (not EXTEND or bool(missing_sets or missing_simple or run_checks))
t_ask = time.time()
runs, ENS, costs, files, trained_now = {}, {}, {}, {}, {}
n_total = len(SETS) * len(SEEDS) if not EXTEND else len(missing_sets)
done = [0]
with F.cpu_lock(SCRIPT) if need_cpu else contextlib.nullcontext():
    waited = time.time() - t_ask
    t_train0 = time.time()
    if not LOAD:
        conf = json.loads((F.RES / "lstm_all.json").read_text())["config"]          # read once the CPU is ours: 04_lstm_all.py writes it when it finishes
        if F.QUICK:
            conf = dict(conf, updates=40, eval_every=20)
    elif EXTEND and missing_sets and not F.QUICK:
        assert conf == json.loads((F.RES / "lstm_all.json").read_text())["config"], "the settings of the existing sets are not those of results/lstm_all.json: the new sets would not be trained with the same settings"
    print(f"settings (results/lstm_all.json): {conf}; seeds {SEEDS}; {n_total} trainings", flush=True)
    for name, v in SETS.items():
        runs[name], costs[name], files[name], trained_now[name] = [], {}, {}, []
        for seed in SEEDS:
            if LOAD and (RESCORE or have_seed(name, seed)):          # a saved set is read, never made again
                p, meta, rec = F.load_preds(f"lstm_{name}_s{seed}"), F.preds_meta(f"lstm_{name}_s{seed}"), record_of(name, seed)
                assert meta["config"] == conf and meta["seed"] == rec["seed"] == seed and meta["script"] == SCRIPT, f"lstm_{name}_s{seed} is not from the training run described in {OUT}.json"
                cost, mf = {k: rec[k] for k in COST}, rec.get("model_file")
            else:
                cfg = F.Cfg(dyn=v["dyn"], use_flow=True, static=tuple(F.STATIC), seed=seed, **conf)
                run = F.fit(cfg, TABLE[v["arm"]], ids, split=ARMS[v["arm"]]["split"], static_df=B)
                p, cost = frames_of(run, TABLE[v["arm"]], v["arm"]), run.summary()
                mf = rel(save_model(run, name, seed))
                F.save_preds(f"lstm_{name}_s{seed}", p, dict(script=SCRIPT, model="lstm", inputs=name, seed=seed, config=conf, dyn=list(cfg.dyn), arm=v["arm"], split=ARMS[v["arm"]]["split"], forecast_columns=SLOT_TEXT.get(v["slot"]),
                                                             aorc_rain_to_midnight=v["aorc"], role=v["role"], model_file=mf, what=v["what"]))
                trained_now[name].append(seed)
                done[0] += 1
            e = evaluate_set(p, v["arm"], "test")
            runs[name].append(p)
            costs[name][seed], files[name][seed] = cost, mf
            print(f"{'training ' + str(done[0]) + ' of ' + str(n_total) + ' | ' if seed in trained_now[name] else ''}lstm_{name} seed {seed} ({SPLIT_NAME[v['arm']]}) | best at update {cost['best_update']} of {cost['updates_run']}, {cost['seconds']:.0f} s | "
                  f"validation NSE {cost['best_val_nse']:.3f} | test NSE median " + " ".join(f"{med(e, k):.3f}" for k in F.LEADS) + " | onsets warned " + " ".join(f"{100 * ons(e, k):.1f}%" for k in F.LEADS)
                  + ("" if seed in trained_now[name] else " | read, not trained again") + f" | {time.time() - t_train0:.0f} s so far", flush=True)
        ENS[name] = F.ensemble(runs[name])
        if trained_now[name]:                              # a rescore, or an extension for a set that exists, reads the prediction sets and leaves them untouched
            F.save_preds(f"lstm_{name}", ENS[name], dict(script=SCRIPT, model="lstm", inputs=name, seeds=SEEDS, config=conf, dyn=list(v["dyn"]), arm=v["arm"], split=ARMS[v["arm"]]["split"], forecast_columns=SLOT_TEXT.get(v["slot"]),
                                                         aorc_rain_to_midnight=v["aorc"], role=v["role"], what=v["what"] + "; day-by-day average of the seeds"))
    # the simple models given the forecast (main split): checks first, then the fits
    SIMPLE = {}
    if run_checks:
        yard_check = dict(features=features_are_causal(), recipe=recipe_check())
    elif old_checks:
        yard_check = old_checks
    else:
        yard_check = None
        notes.append("The checks of the simple-model code (yardstick.checks) were not found in the results file of the earlier run, and --rescore does not make them: run --extend to make them.")
    if not LOAD or missing_simple:
        fitted = fit_simple([n for n in YARD if not LOAD or n in missing_simple])
        for n, (P, sec) in fitted.items():
            F.save_preds(n, P, dict(script=SCRIPT, model=KIND[n], inputs="wqaF", settings=SETTINGS[n], features=FEATURES, fit_seconds=sec, quick_mode=F.QUICK,
                                    what="one model per basin and lead on the features of the input set wq + the rain features of aorc_to_midnight + the forecast rain of the three target days (substituted table: the AORC rain in the training years, "
                                         "the bias-corrected forecast in the validation and test years); 02_baselines.py recipe, settings as chosen there on the validation years for wq"))
    for n in YARD:
        SIMPLE[n] = dict(P=F.load_preds(n), meta=F.preds_meta(n))
        assert SIMPLE[n]["meta"]["script"] == SCRIPT and SIMPLE[n]["meta"]["settings"] == SETTINGS[n] and SIMPLE[n]["meta"]["features"] == FEATURES, f"{n} is not from the fit described here"
trained_pairs = [(n, s) for n in SETS for s in trained_now[n]]
PRIOR_PASS_CLI = [json.loads(a[len("--prior-pass="):]) for a in sys.argv if a.startswith("--prior-pass=")]      # an --extend pass that trained and then stopped before it wrote this file (as --prior-pass in 17_more_basins.py; added 2026-10-30)
stopped_pairs = [(n, s) for n in SETS for s in SEEDS if EXTEND and s not in old["seeds"] and s not in trained_now[n]]       # seeds new to this file that this pass did not train: a pass that stopped before writing this file trained them
assert not stopped_pairs or PRIOR_PASS_CLI, ("the networks " + ", ".join(f"lstm_{n}_s{s}" for n, s in stopped_pairs) + " are on disk but in no pass of the results file: "
                                             "describe the pass that trained them with --prior-pass='{\"what\": ..., \"finished_local_time\": ..., \"script_sha256\": ...}'")
prior_passes = [dict(kind="extension", what=pp.get("what", "an extension (--extend) that trained these networks and stopped before it wrote this file"), n_trainings=len(stopped_pairs),
                     sets_trained=sorted({n for n, _ in stopped_pairs}), seeds_trained=sorted({s for _, s in stopped_pairs}), seed_extension=bool(min(s for _, s in stopped_pairs) > max(old["seeds"])),
                     training_seconds=round(sum(costs[n][s]["seconds"] for n, s in stopped_pairs), 1), waited_for_cpu_s=0.0, simple_models_fit_seconds=0.0, results_file_written=False,
                     **{k: v for k, v in pp.items() if k != "what"}) for pp in PRIOR_PASS_CLI[:1]] if stopped_pairs else []
this_pass = dict(kind="training run" if not LOAD else "extension" if EXTEND else "rescore",
                 what=("training run: every set from scratch" if not LOAD else "extension (--extend): the sets whose files were missing" if EXTEND else "rescore (--rescore): scores, tables, notes and figure recomputed from the saved sets, nothing trained"),
                 n_trainings=len(trained_pairs), sets_trained=sorted({n for n, _ in trained_pairs}), seeds_trained=sorted({s for _, s in trained_pairs}), seed_extension=bool(EXTEND and trained_pairs and min(s for _, s in trained_pairs) > max(old["seeds"])), training_seconds=round(sum(costs[n][s]["seconds"] for n, s in trained_pairs), 1), waited_for_cpu_s=round(waited, 1),
                 simple_models_fit_seconds=round(sum(SIMPLE[n]["meta"]["fit_seconds"] for n in YARD if n in missing_simple or not LOAD), 1), script_sha256=HASH_NOW)
if LOAD:                                                                                   # the passes of the earlier runs
    first = [dict(p) for p in old["passes"]]
    for p in first:
        p.setdefault("script_sha256", "not recorded")
    first = [p for p in first if p.get("kind") != "rescore"]                               # only the last rescore wrote the numbers of the file it leaves
if RESCORE:
    timing = dict({k: old[k] for k in ("n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s") if k in old}, passes=first + [this_pass])
elif EXTEND:
    first = first + prior_passes                                                           # a stopped extension pass, described by --prior-pass (its trainings are on disk)
    timing = dict(n_trainings=sum(p["n_trainings"] for p in first) + this_pass["n_trainings"], training_seconds=round(sum(p["training_seconds"] for p in first) + this_pass["training_seconds"], 1),
                  waited_for_cpu_s=round(sum(p["waited_for_cpu_s"] for p in first) + waited, 1), passes=first + [this_pass])
else:
    timing = dict(n_trainings=this_pass["n_trainings"], training_seconds=this_pass["training_seconds"], waited_for_cpu_s=this_pass["waited_for_cpu_s"], passes=[this_pass])

sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def saved_hashes():
    """sha256 of every prediction set and network file of this script's sets (and of the simple models)."""
    out = {}
    for n in SETS:
        for f in [F.PRED / f"lstm_{n}.npz"] + [p for s in SEEDS for p in (F.PRED / f"lstm_{n}_s{s}.npz", MODELS / f"lstm_{n}_s{s}.pt")]:
            out[rel(f)] = sha(f)
    for n in YARD:
        out[rel(F.PRED / f"{n}.npz")] = sha(F.PRED / f"{n}.npz")
    return out


if not LOAD:                                                # a plain run first writes a provisional record (the training records, the passes, the hashes), so that --rescore or --extend can continue if the scoring below stops
    prov_pass = dict(this_pass, runtime_s=round(time.time() - t_start - waited, 1), finished_local_time=time.strftime("%Y-%m-%d %H:%M"))
    (F.RES_OUT / f"{OUT}.json").write_text(json.dumps(dict(
        provisional=True, what="written by the training pass before its scoring: the training records, the passes and the hashes of the saved files; the complete record replaces it when the pass finishes; a file that still says provisional=True means the pass stopped before the scoring",
        config=conf, seeds=SEEDS, models={n: dict(per_seed=[dict(seed=s, **{c: costs[n][s][c] for c in COST}, model_file=files[n][s]) for s in SEEDS]) for n in SETS},
        n_trainings=timing["n_trainings"], training_seconds=timing["training_seconds"], waited_for_cpu_s=timing["waited_for_cpu_s"], runtime_s=prov_pass["runtime_s"], passes=[prov_pass],
        integrity=dict(sha256_of_saved_files=saved_hashes(), script_sha256=HASH_NOW)), indent=1, default=str))

# ---- the proofs on the saved networks: reload, the AORC column of script 16, the raw forecast at prediction time, the made-up-values test --------------------------------------
try:
    reload_check = dict(what="one trained network per input set (the first seed) rebuilt from its .pt file alone (flood.LSTMNet + the saved weights, settings and scalers), run on the same table, and compared with the forecasts stored by save_preds",
                        tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal", per_set={n: reload_and_forecast(n, SEEDS[0]) for n in SETS}, carried_over_from_training_run=False)
except FileNotFoundError:
    assert LOAD, "a trained network was not saved"
    reload_check = dict(old["reload_check"], carried_over_from_training_run=True)
    notes.append("The network files were not found, so the reload check of the training run was carried over, not repeated.")
aorc_column_check = dict(
    what="the saved networks of 16_aorc_rain_to_midnight.py (set wqa, all seeds) rebuilt from their files and given THIS script's main-split table (the column aorc_to_midnight built here by the code of script 16, copied, over a longer calendar): they must reproduce "
         "the forecasts script 16 stored; this proves that the base column of every set here is the column that trained lstm_wqa",
    tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal", per_seed={str(s): reload_external_wqa(s) for s in SEEDS})
print("reload check: " + "; ".join(f"{n}: max difference {x['max_abs_difference']:.1e}, bit for bit {x['bit_for_bit']}" for n, x in reload_check["per_set"].items()), flush=True)
print("script 16's wqa networks on this script's table: " + "; ".join(f"seed {s}: max difference {x['max_abs_difference']:.1e}, bit for bit {x['bit_for_bit']}" for s, x in aorc_column_check["per_seed"].items()), flush=True)

trajectory = dict(
    what="wqaF_sub, wqaF_clim and wqaP (E2a) have the SAME table in the training years (the AORC rain of the target days in the slots: the substitution starts at the first validation day) and differ only in the table of the validation and test years, so for the same "
         "seed the trainer takes the same random sequences and the same updates: the training loss at every common validation check is equal. They differ only in the update the trainer keeps (its validation score uses the forecast, the shuffled forecast or the "
         "perfect rain) and in the input they are given at run time; wqaF_clim and wqaP are therefore 'the same network with another input', up to the stopping update. (wqF_sub has other columns; the E2b sets have the forecast or the perfect rain "
         "in their training years and so are trained separately.)", per_seed={})
for seed in SEEDS:
    hs = {n: torch.load(MODELS / f"lstm_{n}_s{seed}.pt", map_location="cpu", weights_only=True) for n in ("wqaF_sub", "wqaF_clim", "wqaP")}
    m_ = min(len(z["history"]) for z in hs.values())
    same_ = all(abs(hs["wqaF_sub"]["history"][i]["train_loss"] - hs[n]["history"][i]["train_loss"]) < 1e-9 for n in ("wqaF_clim", "wqaP") for i in range(m_))
    trajectory["per_seed"][str(seed)] = dict(n_common_checks=int(m_), identical_training_loss=bool(same_), best_update={n: int(z["best_update"]) for n, z in hs.items()}, updates_run={n: int(z["updates_run"]) for n, z in hs.items()})
trajectory["all_identical"] = all(v["identical_training_loss"] for v in trajectory["per_seed"].values())
print("E2a sets share their training trajectory (same seed): " + str(trajectory["all_identical"]) + "; best updates " + "; ".join(f"seed {s}: " + "/".join(str(v["best_update"][n]) for n in ("wqaF_sub", "wqaF_clim", "wqaP")) for s, v in trajectory["per_seed"].items()), flush=True)

# the derived sets: saved networks given another table at PREDICTION time (scored variants by substitution, labelled so; nothing is trained or chosen on them)
TAB_RAW = {g: TABLE["main"][g].assign(**{f"fs{k}": TABLE["main"][g][f"fr{k}"].values for k in KS}) for g in ids}                  # the table of wqaF_sub with the RAW forecast in the slots (the AORC rain up to 2000-09-30, as there)
TAB_PERF = {g: TABLE["main"][g].assign(**{f"fs{k}": TABLE["main"][g][f"p{k}"].values for k in KS}) for g in ids}                  # ... with the perfect rain (the AORC rain of t+1..t+3) in the slots in every year
TAB_FC_FOR_P = {g: TABLE["main"][g].assign(**{f"p{k}": TABLE["main"][g][f"fs{k}"].values for k in KS}) for g in ids}              # the table of wqaP with the bias-corrected forecast in the p slots from 2000-10-01 on (the AORC rain before, as in wqaF_sub)
DERIVED = {RAW_VARIANT: ("wqaF_sub", TAB_RAW, "the saved networks of wqaF_sub (trained on the AORC rain of the target days, stopped on the bias-corrected forecast) given the RAW forecast in the slots at prediction time"),
           PERFECT_VARIANT: ("wqaF_sub", TAB_PERF, "the saved networks of wqaF_sub given the PERFECT rain of the three target days in the slots at prediction time (in the validation and test years too): the same networks as wqaF_sub, scored twice (ROADMAP.md, E2a)"),
           STOP_VARIANT: ("wqaP", TAB_FC_FOR_P, "the saved networks of wqaP (trained AND stopped on perfect rain) given the bias-corrected forecast in the slots from 2000-10-01 on at prediction time: the same input as wqaF_sub, but a stopping point chosen with perfect rain in the slots")}
def save_derived(name, P, meta):
    """Write a derived set - unless a --rescore finds the same forecasts already saved: the derived sets are functions of the saved networks and are computed again in every pass, and a file that holds what they give now is left byte for byte as it is."""
    f = F.PRED / f"{name}.npz"
    if LOAD and f.exists() and "seeds" in meta and list(F.preds_meta(name).get("seeds") or []) != list(meta["seeds"]):      # a seed average over another seed list (seeds 3-4 added by --extend, 30 October 2026): remade, not compared
        was = list(F.preds_meta(name).get("seeds") or [])
        F.save_preds(name, P, meta)
        return f"remade: the average of seeds {list(meta['seeds'])} replaces that of seeds {was}"
    if LOAD and f.exists():
        old_ = F.load_preds(name)
        assert all(old_[g].index.equals(P[g].index) and np.array_equal(old_[g].values, P[g].values, equal_nan=True) for g in ids), f"{name}: the derived set on disk is not what the saved networks give now"
        return "left as it was"
    F.save_preds(name, P, meta)
    return "written"


DRUNS, DERIVED_FILES = {}, {}
for dname, (src, tab, what) in DERIVED.items():                                                                  # a scored variant, not a training
    DRUNS[dname] = []
    for seed in SEEDS:
        run_, z_ = run_from_file(MODELS / f"lstm_{src}_s{seed}.pt")
        DRUNS[dname].append(frames_of(run_, tab, "main"))
        DERIVED_FILES[f"lstm_{dname}_s{seed}"] = save_derived(f"lstm_{dname}_s{seed}", DRUNS[dname][-1], dict(script=SCRIPT, model="lstm", inputs=src, seed=seed, config=conf, arm="main", role="variant", derived=True, what=what + " (labelled so: a scored variant, not a training)"))
    DERIVED_FILES[f"lstm_{dname}"] = save_derived(f"lstm_{dname}", F.ensemble(DRUNS[dname]), dict(script=SCRIPT, model="lstm", inputs=src, seeds=SEEDS, config=conf, arm="main", role="variant", derived=True, what="day-by-day average of the seeds of: " + what))
raw_runs = DRUNS[RAW_VARIANT]


def forward_tests():
    """Two cheap forward-pass tests of the over-trust reading on the SAVED wqaF_sub networks (nothing is trained, nothing is saved, nothing is chosen): (a) slots = (1 - w) x the perfect rain + w x the forecast for w = 0, 0.25, 0.5, 0.75, 1 (the dose-response
    of the networks to the exactness of their slots); (b) the forecast values below 0.5 mm and below 1 mm set to 0 (the forecast has fewer exact zeros than the AORC rain). The mixtures and the zeroing apply to every issue day from the first validation day
    (the training years hold the AORC rain in the slots, as always). Scored like every set: the seeds averaged, the test years, the fixed warning level, with the onsets warned, their false warnings and the median NSE."""
    tabs = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        tabs[f"w={w:g}"] = ({g: TABLE["main"][g].assign(**{f"fs{k}": ((1 - w) * TABLE["main"][g][f"p{k}"] + w * TABLE["main"][g][f"fs{k}"]).values for k in KS}) for g in ids}, "slots = (1 - w) x the perfect rain + w x the forecast, w = " + f"{w:g}")
    for c in (0.5, 1.0):
        tabs[f"below_{c:g}_mm_set_to_zero"] = ({g: TABLE["main"][g].assign(**{f"fs{k}": np.where((TABLE["main"][g].index >= pd.Timestamp(SUB_FROM)) & (TABLE["main"][g][f"fs{k}"].values < c), 0.0, TABLE["main"][g][f"fs{k}"].values) for k in KS}) for g in ids},
                                               f"the forecast values below {c:g} mm set to 0 (from the first validation day on)")
    nets = [run_from_file(MODELS / f"lstm_wqaF_sub_s{seed}.pt")[0] for seed in SEEDS]
    out = {}
    for name, (tab, what) in tabs.items():
        E = evaluate_set(F.ensemble([frames_of(r, tab, "main") for r in nets]), "main", "test")
        zeroed = None
        if name.startswith("below"):
            c = float(name.split("_")[1])
            sl = np.concatenate([TABLE["main"][g].loc[SUB_FROM:, [f"fs{k}" for k in KS]].values.ravel() for g in ids])
            zeroed = float(((sl < c) & (sl > 0)).sum() / (sl > 0).sum())
        out[name] = dict(what=what, share_of_the_nonzero_forecast_values_set_to_zero=zeroed, by_lead={lk(k): dict(onset_warned=ons(E, k), onset_false=ons(E, k, "far"), onset_csi=ons(E, k, "csi"), nse_median=med(E, k)) for k in F.LEADS})
    return out


FORWARD = forward_tests()
print("forward-pass tests on the saved wqaF_sub networks (onsets warned at 1/2/3 days): " + "; ".join(f"{n}: " + "/".join(f"{100 * v['by_lead'][lk(k)]['onset_warned']:.1f}" for k in F.LEADS) for n, v in FORWARD.items()), flush=True)


def made_up_values_test(name, seed, n_cuts=3):
    """The made-up-values test on a saved network: replace the forecast columns (and, in a second round, every input column and the flow) of all days AFTER a cut day c by random values, predict again and compare. A forecast issued on or before
    day c must not change (it is stored at the target day t+k; array_equal on every lead), and the ones issued after c must change (or the test proves nothing)."""
    v = SETS[name]
    run_, _ = run_from_file(MODELS / f"lstm_{name}_s{seed}.pt")
    tab = TABLE[v["arm"]]
    idx = tab[ids[0]].index
    a, b = ARMS[v["arm"]]["split"]["test"]
    cand = np.flatnonzero((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))[200:-200]
    rng = np.random.default_rng(7)
    cuts = sorted(int(c) for c in rng.choice(cand, n_cuts, replace=False))
    base = frames_of(run_, tab, v["arm"])
    out = dict(set=name, seed=seed, cut_days=[str(idx[c].date()) for c in cuts], rounds={})
    for what, cols in (("the forecast columns only", list(SLOT[v["slot"]])), ("every input column and the observed flow", sorted(set(run_.cfg.dyn) | {"q_mm"}))):
        n_same, n_moved = 0, 0
        for c in cuts:
            alt = {g: t.copy() for g, t in tab.items()}
            for g in ids:
                for col in cols:
                    x = alt[g][col].values.copy()
                    x[c + 1:] = rng.uniform(0.0, 120.0, len(x) - c - 1)
                    alt[g][col] = x
            new = frames_of(run_, alt, v["arm"])
            for g in ids:
                for k in F.LEADS:
                    assert np.array_equal(base[g][lk(k)].values[:c + k + 1], new[g][lk(k)].values[:c + k + 1], equal_nan=True), f"{name} {g}: changing {what} after {idx[c].date()} changed a forecast issued on or before that day (lead {k})"
                    n_same += 1
            moved = [not np.allclose(base[g][lk(1)].values[c + 2:c + 12], new[g][lk(1)].values[c + 2:c + 12], rtol=0, atol=1e-9) for g in ids]
            assert any(moved), f"{name}: the test could not see a change after the cut (it proves nothing)"
            n_moved += int(sum(moved))
        out["rounds"][what] = dict(forecasts_compared_equal=n_same, basins_whose_later_forecasts_moved_summed_over_the_cuts=n_moved, passed=True)
    return out


causality = dict(
    what="(1) flood.fit runs flood._assert_causal on every network before training (random inputs: a change of the inputs after day T/2 must not change the output before it); (2) the made-up-values test on the SAVED networks of every set that reads forecast "
         "columns (first seed): the forecast columns (and in a second round every input column and the flow) of all days after a cut day were replaced by random values (0 to 120 per day); no forecast issued on or before the cut day changed "
         "(exactly equal, all three leads, all 12 basins), and the forecasts issued after it did change; the columns are dated by the ISSUE day t, so a forecast issued on t reads only columns up to t; "
         "(3) code/gefs_to_basins.py proves the content of the columns: the run used for issue day t is the 00 UTC run of the UTC date t+1, issued before the issue time, and replacing every later run by random rain changes no forecast issued earlier (results/gefs_basins.json, clock)",
    engine_check="flood._assert_causal at the start of every flood.fit (asserted: a trained network exists only if it passed)",
    made_up_values={n: made_up_values_test(n, SEEDS[0]) for n in SETS if SETS[n]["slot"]},
    gefs_source_check=dict(file="results/gefs_basins.json", key="clock", made_up_values_test=GBR["clock"]["made_up_values_test"]["what"]))
print("made-up-values test passed on " + ", ".join(causality["made_up_values"]), flush=True)

# ---- integrity: what was made in this pass, and that what was only read is what it was ---------------------------------------------------------------
SAVED = saved_hashes()
made_now = {rel(F.PRED / f"lstm_{n}_s{s}.npz") for n, s in trained_pairs} | {rel(MODELS / f"lstm_{n}_s{s}.pt") for n, s in trained_pairs} | {rel(F.PRED / f"lstm_{n}.npz") for n in SETS if trained_now[n]} \
    | {rel(F.PRED / f"{n}.npz") for n in YARD if not LOAD or n in missing_simple}
earlier_hashes = old.get("integrity", {}).get("sha256_of_saved_files", {}) if LOAD else {}
made_stopped = {rel(F.PRED / f"lstm_{n}_s{s}.npz") for n, s in stopped_pairs} | {rel(MODELS / f"lstm_{n}_s{s}.pt") for n, s in stopped_pairs} | {rel(F.PRED / f"lstm_{n}.npz") for n in {n_ for n_, _ in stopped_pairs}}
for n in sorted({n_ for n_, _ in stopped_pairs}):                                          # the seed average a stopped pass wrote must be the average of the seeds on disk
    on_disk = F.load_preds(f"lstm_{n}")
    assert list(F.preds_meta(f"lstm_{n}")["seeds"]) == list(SEEDS) and all(on_disk[g].index.equals(ENS[n][g].index) and np.array_equal(on_disk[g].values, ENS[n][g].values, equal_nan=True) for g in ids), \
        f"lstm_{n}: the seed average on disk is not the average of the seeds on disk"
compared = [f for f in SAVED if f not in made_now and f not in made_stopped and f in earlier_hashes]
assert all(earlier_hashes[f] == SAVED[f] for f in compared), "a saved file changed since the earlier results file recorded its hash: " + ", ".join(f for f in compared if earlier_hashes[f] != SAVED[f])
INPUT_FILES = {rel(GB.CSV(g)): sha(GB.CSV(g)) for g in ids}
ARCHIVE_COPY = ROOT / "code" / "archive" / "18_forecast_input_training_version.py"
TRAIN_SCRIPT = dict(file=rel(ARCHIVE_COPY), exists=ARCHIVE_COPY.exists(), sha256=sha(ARCHIVE_COPY), first_pass_script_sha256=timing["passes"][0]["script_sha256"],
                    equal_to_the_script_of_the_first_pass=bool(ARCHIVE_COPY.exists() and sha(ARCHIVE_COPY) == timing["passes"][0]["script_sha256"]), note="code/archive/18_forecast_input_training_version.NOTE.txt: the byte-for-byte copy of the version that ran the 30 trainings")
INTEGRITY = dict(how_to_read="sha256 of every prediction set and network file of this script, as they are when this file is written; made_in_this_pass lists the files this pass wrote; every other file was only read. A later pass compares the hashes "
                             "of the files it only reads with these and stops if one differs. The derived sets lstm_wqaF_sub_rawpred*, lstm_wqaF_sub_perfectpred* and lstm_wqaP_forecastpred* are recomputed from the saved networks in every pass and are not hashed.",
                 script_sha256=HASH_NOW, sha256_of_saved_files=SAVED, made_in_this_pass=sorted(made_now), made_in_a_stopped_pass=sorted(made_stopped), only_read_in_this_pass=len(SAVED) - len(made_now) - len(made_stopped),
                 compared_with_the_earlier_results_file=(dict(files_compared=len(compared), all_equal=True) if compared else dict(files_compared=0, all_equal=None, why="nothing was read from an earlier pass")) if LOAD else None,
                 sha256_of_the_forecast_tables_read=INPUT_FILES, gefs_basins_json_code_sha256=GBR["code"]["sha256"], training_version_of_the_script=TRAIN_SCRIPT)

# ---- 3. every set scored the same way ----------------------------------------------------------------------------------------------------------------
missing_stored = [n for n in STORED if not F.have_preds(f"lstm_{n}")]
assert not missing_stored, "the stored sets " + ", ".join(f"lstm_{n}" for n in missing_stored) + " were not found (04_lstm_all.py and 16_aorc_rain_to_midnight.py write them)"
SHOWN = list(SETS) + list(STORED) + list(DERIVED)
PRED = {n: ENS[n] if n in SETS else F.ensemble(DRUNS[n]) if n in DERIVED else F.load_preds(f"lstm_{n}") for n in SHOWN}
SEEDP = {n: runs[n] if n in SETS else DRUNS[n] if n in DERIVED else [F.load_preds(f"lstm_{n}_s{s}") for s in range(5) if F.have_preds(f"lstm_{n}_s{s}")] for n in SHOWN}
EV = {n: dict(test=evaluate_set(PRED[n], ARM_OF[n], "test"), val=evaluate_set(PRED[n], ARM_OF[n], "val"), seeds=[evaluate_set(p, ARM_OF[n], "test") for p in SEEDP[n]]) for n in SHOWN}
assert all(EV[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in SHOWN for part in ("test", "val") for k in F.LEADS), "days without a forecast: the models would not be compared on identical days"
assert all(np.allclose(PRED[n][g].obs.loc[F.SPLIT["val"][0]:].values, D[g].q_mm.loc[F.SPLIT["val"][0]:].values) for n in SHOWN for g in ids), "observed flow differs"
print("scored: " + ", ".join(SHOWN), flush=True)
assert all(abs(FORWARD["w=1"]["by_lead"][lk(k)]["onset_warned"] - ons(EV["wqaF_sub"]["test"], k)) < 1e-12 and abs(FORWARD["w=0"]["by_lead"][lk(k)]["onset_warned"] - ons(EV[PERFECT_VARIANT]["test"], k)) < 1e-12 for k in F.LEADS), \
    "the forward-pass tests do not reproduce the scores of the sets they contain (w = 1: the forecast; w = 0: the perfect rain)"

# ---- summaries, seed ranges, per-basin scores --------------------------------------------------------------------------------------------------------
def seed_range(n, k):
    """Lowest and highest single-seed value of a set at one lead, for the scores the rule and the figure use."""
    S = [e["summary"][lk(k)] for e in EV[n]["seeds"]]
    return dict(n_seeds=len(S), nse_median=lo_hi([s["nse_median"] for s in S]), **{f"{kind}_{c}": lo_hi([s[kind][c] for s in S]) for kind in ("onset", "all") for c in ("pod", "far", "csi")},
                by_seed=dict(nse_median=[s["nse_median"] for s in S], **{f"{kind}_{c}": [s[kind][c] for s in S] for kind in ("onset", "all") for c in ("pod", "far", "csi")}))


summary_test = {n: {lk(k): row(EV[n]["test"], k) for k in F.LEADS} for n in SHOWN}
summary_val = {n: {lk(k): row(EV[n]["val"], k) for k in F.LEADS} for n in SHOWN}
ranges = {n: {lk(k): seed_range(n, k) for k in F.LEADS} for n in SHOWN}
per_basin = {"how_to_read": "test years of each set's own split (arm), one value per basin; the seed average of each set (flood.evaluate); gauge ids are text",
             "basins": {g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g])) for g in ids},
             "nse": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["nse"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "onset_warned": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "all_days_warned": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["all"]["pod"] for g in ids} for k in F.LEADS} for n in SHOWN}}
models = {n: dict(saved_as=f"lstm_{n}", arm=ARM_OF[n], n_seeds_scored=len(EV[n]["seeds"]), test=EV[n]["test"]) for n in SHOWN}
for n in SETS:
    models[n]["per_seed"] = [dict(seed=s, **{c: costs[n][s][c] for c in COST}, model_file=files[n][s], test_nse_median=[med(e, k) for k in F.LEADS], test_onset_warned=[ons(e, k) for k in F.LEADS],
                                  test_onset_false=[ons(e, k, "far") for k in F.LEADS], test_all_days_warned=[e["summary"][lk(k)]["all"]["pod"] for k in F.LEADS]) for s, e in zip(SEEDS, EV[n]["seeds"])]
training = [dict(set=n, seed=s, arm=SETS[n]["arm"], saved_forecasts=f"lstm_{n}_s{s}", model_file=files[n][s], **{c: costs[n][s][c] for c in COST}) for n in SETS for s in SEEDS]


# ---- the checks of report section 4.2 (copied from 15 / 16: the helpers) ------------------------------------------------------------------------------
def paired(a, b):
    """Two per-basin score arrays (first minus second): in how many basins the first is higher, the median difference, Wilcoxon p (as in 04_lstm_all.py)."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)                        # two-sided signed-rank test over the basins
    except ValueError:                                          # all differences zero
        p = float("nan")
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=int(len(d)), median_nse_difference=float(np.median(d)), wilcoxon_p=p)


def rule(T, C, k):
    """Set T against set C (same split) at lead k. NSE: every single seed of T above every single seed of C, higher in most basins, Wilcoxon p below P_CUT - all three make a clear gain. Onsets warned: pooled over the basins, so only
    the seeds can be compared."""
    assert ARM_OF[T] == ARM_OF[C], f"{T} and {C} are scored on different splits: their numbers are not comparable"
    ET, EC = EV[T]["test"], EV[C]["test"]
    pr, n_low = paired(nse_of(ET, k), nse_of(EC, k)), int((nse_of(ET, k) < nse_of(EC, k)).sum())
    sT, sC = ranges[T][lk(k)], ranges[C][lk(k)]
    up, down = above(sT["nse_median"], sC["nse_median"]), above(sC["nse_median"], sT["nse_median"])
    small_p = bool(pr["wilcoxon_p"] < P_CUT)                       # False when p is not defined
    gain, loss = up and pr["n_basins_higher"] > n_b / 2 and small_p, down and n_low > n_b / 2 and small_p
    o_up, o_down = above(sT["onset_pod"], sC["onset_pod"]), above(sC["onset_pod"], sT["onset_pod"])
    return dict(against=f"lstm_{C}", lead=k,
                nse=dict(median_with=med(ET, k), median_without=med(EC, k), gain=med(ET, k) - med(EC, k), seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                         single_seeds_all_above=up, single_seeds_all_below=down, n_basins_higher=pr["n_basins_higher"], n_basins_lower=n_low, n_basins=n_b,
                         higher_in_most_basins=bool(pr["n_basins_higher"] > n_b / 2), median_basin_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"],
                         p_below_cut=small_p, clear_gain=bool(gain), clear_loss=bool(loss),
                         reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                onset=dict(warned_with=ons(ET, k), warned_without=ons(EC, k), gain=ons(ET, k) - ons(EC, k), seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"],
                           false_with=ons(ET, k, "far"), false_without=ons(EC, k, "far"), single_seeds_all_above=o_up, single_seeds_all_below=o_down,
                           clear_gain=bool(o_up), clear_loss=bool(o_down), reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"))


def onset_flags(P, k, arm):
    """Which onsets were warned: one True / False per onset of the arm's test years (a target day above the high-flow level while the river was at or below it when the forecast was issued, as flood.flood_scores counts them),
    basin after basin in the order of ids; the same days for every set of the arm (coverage 1, asserted above)."""
    a, b = ARMS[arm]["split"]["test"]
    thr = ARMS[arm]["thr"]
    out = []
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(k) <= thr[g]).loc[a:b]
        q = p.loc[a:b]
        ok = (q.obs.notna() & q[lk(k)].notna()).values
        o, s, w = (q.obs > thr[g]).values[ok], (q[lk(k)] > thr[g]).values[ok], was_low.values[ok]
        out.append(s[o & w])
    return np.concatenate(out)


def overlap(A, C, k=1, frames=None):
    """Do two sets warn of the same onsets? Counts of the onsets warned by both, by only one, by neither; the exact sign-test p of 'only A' against 'only C' (a description: the onsets of one storm are not independent)."""
    frames = frames or (lambda n: PRED[n])
    arm = ARM_OF[A] if A in ARM_OF else "main"
    fa, fc = onset_flags(frames(A), k, arm), onset_flags(frames(C), k, arm)
    return dict(first=A, second=C, lead=k, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=int((fa & fc).sum()), warned_only_by_first=int((fa & ~fc).sum()), warned_only_by_second=int((~fa & fc).sum()),
                warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=int((fa | fc).sum()),
                sign_test_p=float(binomtest(int((fa & ~fc).sum()), int((fa ^ fc).sum()), 0.5).pvalue) if (fa ^ fc).sum() else float("nan"))


def pair_reading(T, C):
    """Set T minus set C at every lead: onsets warned (pooled over the basins; hits, seed ranges, seed by seed: the same seed number of both sets, in order), the median NSE with the checks of section 4.2, and the onsets warned by both / only one."""
    out = {}
    for k in F.LEADS:
        r = rule(T, C, k)
        ET, EC, sT, sC = EV[T]["test"], EV[C]["test"], ranges[T][lk(k)], ranges[C][lk(k)]
        tT, tC = ET["summary"][lk(k)]["onset"], EC["summary"][lk(k)]["onset"]
        out[lk(k)] = dict(with_set=f"lstm_{T}", without_set=f"lstm_{C}", arm=ARM_OF[T],
                          onsets=dict(warned_with=tT["pod"], warned_without=tC["pod"], difference_points=100 * (tT["pod"] - tC["pod"]), hits_with=tT["hits"], hits_without=tC["hits"], onsets_in_the_test_years=tT["hits"] + tT["misses"],
                                      hits_difference=tT["hits"] - tC["hits"], seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"], every_seed_above_every_seed=r["onset"]["single_seeds_all_above"],
                                      every_seed_below_every_seed=r["onset"]["single_seeds_all_below"], difference_by_seed_points=[100 * (x - y) for x, y in zip(sT["by_seed"]["onset_pod"], sC["by_seed"]["onset_pod"])],
                                      false_warnings_with=tT["far"], false_warnings_without=tC["far"], reading=r["onset"]["reading"]),
                          nse=dict(median_with=r["nse"]["median_with"], median_without=r["nse"]["median_without"], gain=r["nse"]["gain"], seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                                   every_seed_above_every_seed=r["nse"]["single_seeds_all_above"], difference_by_seed=[x - y for x, y in zip(sT["by_seed"]["nse_median"], sC["by_seed"]["nse_median"])],
                                   n_basins_higher=r["nse"]["n_basins_higher"], n_basins=n_b, median_basin_difference=r["nse"]["median_basin_difference"], wilcoxon_p=r["nse"]["wilcoxon_p"], reading=r["nse"]["reading"]),
                          overlap_of_warned_onsets=overlap(T, C, k))
    return out


# ---- the closure of the gap to an upper bound ----------------------------------------------------------------------------------------------------------
def closure(S, base, ceil_):
    """How much of the gap between a base set and an upper-bound set (the same split) the set S closes: per lead the scores of the three, the gain over the base, the share of the gap closed (NSE and onsets warned), the basins better than the base."""
    assert ARM_OF[S] == ARM_OF[base] == ARM_OF[ceil_]
    Eb, Ec, En = (EV[x]["test"] for x in (base, ceil_, S))
    rows = []
    for k in F.LEADS:
        b, c, n = med(Eb, k), med(Ec, k), med(En, k)
        ob, oc, on = (ons(E, k) for E in (Eb, Ec, En))
        rows.append(dict(lead=k, nse_base=b, nse_with=n, nse_ceiling=c, gap=c - b, gain=n - b, share_of_gap_closed=div(n - b, c - b), basins_better_than_base=int((nse_of(En, k) > nse_of(Eb, k)).sum()),
                         median_basin_gain=float(np.median(nse_of(En, k) - nse_of(Eb, k))), onset_warned_base=ob, onset_warned_with=on, onset_warned_ceiling=oc, onset_false_base=ons(Eb, k, "far"), onset_false_with=ons(En, k, "far"),
                         onset_false_ceiling=ons(Ec, k, "far"), onset_share_of_gap_closed=div(on - ob, oc - ob), seed_range_nse_with=ranges[S][lk(k)]["nse_median"], seed_range_nse_base=ranges[base][lk(k)]["nse_median"],
                         onset_seed_range_with=ranges[S][lk(k)]["onset_pod"], onset_seed_range_base=ranges[base][lk(k)]["onset_pod"]))
    return dict(set=f"lstm_{S}", base=f"lstm_{base}", ceiling=f"lstm_{ceil_}", arm=ARM_OF[S], by_lead=rows)


CLOSURES = [("wqaF_sub", "wqa", "wqaP"), ("wqaF_sub", "wqa", "wqf"), ("wqaF_sub", "wq", "wqf"), ("wqaF_clim", "wqa", "wqaP"), ("wqF_sub", "wq", "wqf"), ("wqF_sub", "wq", "wqaP"), ("wqa", "wq", "wqf"), ("wqa", "wq", "wqaP"), ("wqaP", "wqa", "wqf"),
            ("wqaF_s", "wqa_s", "wqaP_s"), ("wqaF_s", "wq_s", "wqaP_s"), ("wqaF_clim_s", "wqa_s", "wqaP_s"), ("wqa_s", "wq_s", "wqaP_s")]
closures = {f"{S}__base_{b}__ceiling_{c}": closure(S, b, c) for S, b, c in CLOSURES}
gain_rule = {f"{T}_against_{C}": {lk(k): rule(T, C, k) for k in F.LEADS} for T, C in (("wqaF_sub", "wqa"), ("wqaF_sub", "wqaF_clim"), ("wqaF_sub", "wqaP"), ("wqF_sub", "wq"), ("wqaF_clim", "wqa"), ("wqaP", "wqa"), ("wqaF_s", "wqa_s"), ("wqaF_s", "wqaF_clim_s"),
                                                                                         ("wqaF_s", "wqaP_s"), ("wqa_s", "wq_s"), ("wqaF_clim_s", "wqa_s"), ("wqaF_sub", RAW_VARIANT))}

# ---- the readings: pairs of sets that differ in one thing ----------------------------------------------------------------------------------------------
TOGETHER = "Quote it with its companions (forecast_vs_climatology_control and retrained_vs_substituted): the forecast's value is read against the control with the same climate and no information about the day."
readings = {}
readings["value_of_the_forecast"] = dict(
    what="THE POINT OF E2. lstm_wqaF_sub minus lstm_wqa: the worth of the bias-corrected GEFS control-member rain forecast of the local days t+1..t+3 as three extra daily inputs beside the AORC rain to midnight, with the river gauge, on the main split (test years "
         "2005-2014). wqaF_sub was trained on the AORC rain of those days (an analysis made after the fact; it never saw a forecast in training), stopped on the validation years with the real forecast in the slots and scored on the test years with it; wqa is the same network without the "
         "three columns (script 16). The two have different numbers of inputs, so the same seed number gives different starting weights. The numbers of by_lead are at the FIXED warning level (trigger 1.0) and the readings of the seed rule are about that level: at 1 day it moves the operating point "
         "(fewer, more often right warnings), so read operating_points beside it: the same comparison at the level chosen on the validation years, at equal false-warning budgets, without a trigger (average precision) and with block-bootstrap intervals. " + TOGETHER, by_lead=pair_reading("wqaF_sub", "wqa"))
readings["forecast_vs_climatology_control"] = dict(
    what="lstm_wqaF_sub minus lstm_wqaF_clim: the forecast against a forecast with the right climate and no information about the day (the same forecasts shuffled within basin and calendar month across years, the three leads of a day together). The control was "
         "trained, stopped and scored exactly like wqaF_sub. THIS CONTROL IS NOT A NEUTRAL BASELINE: the E2a networks were trained on exact slots, so given slots with no information they collapse far below lstm_wqa (see value_of_a_misleading_input): the difference measures the harm of misleading "
         "slots to a network that trusts them, not the information of the forecast, and it is large at every lead whatever the forecast is worth. The clean information test is E2b's control (retrained_vs_substituted.retrained_against_its_control), trained and run on shuffled slots, which equals the model without slots.", by_lead=pair_reading("wqaF_sub", "wqaF_clim"))
readings["value_of_a_misleading_input"] = dict(
    what="lstm_wqaF_clim minus lstm_wqa: what the shuffled forecast does to a network that was trained on perfect rain and trusts its slots: the harm of misleading slots (not a measure of the forecast's information).", by_lead=pair_reading("wqaF_clim", "wqa"))
readings["retrained_vs_substituted"] = dict(
    what="E2b against E2a. retrained = lstm_wqaF_s minus lstm_wqa_s, both retrained on the satellite-era split (train 2000-10-01..2008-09-30, validation 2008-10-01..2010-09-30, test 2010-10-01..2014-09-30), so the network learns the forecast's own errors; "
         "substituted = lstm_wqaF_sub minus lstm_wqa on the main split (test 2005-2014), set beside it. THE TWO ARE NOT COMPARABLE IN LEVEL: the E2b split trains on 8 years instead of 20, its test years are 2010-2014 (4 years, other onsets, other flood "
         "thresholds from its own training years; the E2b base lstm_wqa_s is weaker than lstm_wqa), so compare the gains, not the scores, and read a difference between the two gains as a hint, not as a measurement: it mixes two splits and two bases. "
         "The E2b control (retrained_against_its_control), trained and run on shuffled slots, equals the model without slots and is the clean information test; retrained_operating_points gives the E2b comparison at the level chosen on the validation years, at equal false-warning budgets and without a trigger.",
    split_caveat="E2b: train 2000-10-01..2008-09-30, validation 2008-10-01..2010-09-30, test 2010-10-01..2014-09-30, thresholds from its training years; E2a: train 1980-10-01..2000-09-30, validation 2000-10-01..2005-09-30, test 2005-10-01..2014-09-30",
    retrained=pair_reading("wqaF_s", "wqa_s"), substituted=pair_reading("wqaF_sub", "wqa"), retrained_against_its_control=pair_reading("wqaF_s", "wqaF_clim_s"), retrained_against_perfect_rain=pair_reading("wqaP_s", "wqaF_s"))
readings["forecast_without_rain_to_midnight"] = dict(
    what="lstm_wqF_sub minus lstm_wq: the forecast on top of the plain gauge model (rain table, no rain to midnight), substituted as in wqaF_sub. Beside it, rain_to_midnight_given_the_forecast (wqaF_sub minus wqF_sub) shows how the two kinds of "
         "information combine, and the forecast alone against the rain to midnight alone (wqF_sub minus wqa).", by_lead=pair_reading("wqF_sub", "wq"),
    rain_to_midnight_given_the_forecast=pair_reading("wqaF_sub", "wqF_sub"), forecast_alone_against_the_rain_to_midnight_alone=pair_reading("wqF_sub", "wqa"))
readings["how_far_from_perfect"] = dict(
    what="lstm_wqaP (the AORC rain of the three target days, a perfect rain forecast with the right clock) minus lstm_wqaF_sub: how far the real forecast stays from a perfect one for rain alone; the share of the gap between wqa and wqaP that wqaF_sub closes is in "
         "closure.wqaF_sub__base_wqa__ceiling_wqaP, and in closure.wqaF_sub__base_wqa__ceiling_wqf against the old perfect-weather upper bound (which also holds the other weather of the three days). The same for E2b (retrained_against_perfect_rain in "
         "retrained_vs_substituted).", perfect_rain_minus_substituted_forecast=pair_reading("wqaP", "wqaF_sub"), perfect_rain_minus_rain_to_midnight=pair_reading("wqaP", "wqa"),
    share_of_the_gap_closed=dict(rain_only_gap=closures["wqaF_sub__base_wqa__ceiling_wqaP"]["by_lead"], gap_to_the_perfect_weather_forecast=closures["wqaF_sub__base_wqa__ceiling_wqf"]["by_lead"]))
readings["raw_vs_bias_corrected"] = dict(
    what="the bias correction (one factor per basin and lead, fitted on the validation years): the saved wqaF_sub networks given the RAW forecast in the slots at prediction time (lstm_wqaF_sub_rawpred: a scored variant by substitution, NOT a retraining; the networks "
         "were trained on the AORC rain of the target days and stopped on the bias-corrected forecast) against the same networks given the bias-corrected one. The factors are within 0.85 to 1.4 (median about 1.0): see results/gefs_basins.json, bias.",
    at_prediction_time=pair_reading("wqaF_sub", RAW_VARIANT))
if "wqaF_sub_raw" in SETS:
    readings["raw_vs_bias_corrected"]["retrained_on_the_raw_forecast"] = dict(what="lstm_wqaF_sub minus lstm_wqaF_sub_raw: a network trained on the AORC rain of the target days and stopped on the RAW forecast, scored with it", by_lead=pair_reading("wqaF_sub", "wqaF_sub_raw"))
readings["substituted_network_given_perfect_rain"] = dict(
    what="the same saved wqaF_sub networks scored twice (ROADMAP.md, E2a: 'score 2005-2014 twice: with the perfect rain and with the GEFS rain'): lstm_wqaF_sub_perfectpred (given the AORC rain of the three target days in the slots, in the validation and test years too: "
         "a scored variant by substitution at prediction time, NOT a training) minus lstm_wqaF_sub (given the forecast). The difference is what the forecast's errors cost this network; the networks were stopped with the forecast in the slots, so they are not tuned to perfect rain.",
    perfect_rain_given_minus_forecast_given=pair_reading(PERFECT_VARIANT, "wqaF_sub"), against_wqaP=pair_reading("wqaP", PERFECT_VARIANT))
readings["stopping_point_chosen_with_the_forecast"] = dict(
    what="lstm_wqaF_sub minus lstm_wqaP_forecastpred: the same input (the AORC rain up to 2000-09-30, the bias-corrected forecast from 2000-10-01 on, in the slots), the same training years, but the stopping point of wqaF_sub was chosen on the validation years WITH the forecast in the slots and that of the "
         "wqaP networks (given the forecast only at prediction time: a scored variant by substitution, NOT a training) with perfect rain in them. It is the worth of choosing the stopping point with the real forecast.", by_lead=pair_reading("wqaF_sub", STOP_VARIANT))

# ---- the verdict of results/flood_analysis.json -------------------------------------------------------------------------------------------------------
CRIT = FA["verdict"]["criteria"]


def number(text, key, op):
    m = re.search(rf"{key} {op} ([0-9.]+)", text)
    assert m, f"cannot read '{key} {op}' in the stored criterion: {text}"
    return float(m.group(1))


POD_GO, FAR_GO, CSI_CAUTION = number(CRIT["GO"], "pod", ">="), number(CRIT["GO"], "far", "<="), number(CRIT["CAUTION"], "csi", ">=")        # the numbers come from the stored text
LEVELS = [round(float(m), 2) for m in np.arange(0.30, 1.501, 0.10 if F.QUICK else 0.05)]       # warning level as a multiple of the high-flow level (06_flood_analysis.py)
KINDS, COUNTS = ("all", "onset"), ("hits", "misses", "false_alarms", "correct_neg")
pool = lambda tabs: F.rates({c: int(sum(t[c] for t in tabs)) for c in COUNTS})


def verdict(t):
    """The project's rule (results/flood_analysis.json, verdict.criteria) applied to one warning table pooled over the basins."""
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= POD_GO and t["far"] <= FAR_GO:                # far is NaN when no warning was given: then pod is 0 and this is False
        return "GO"
    return "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"


def score(P, part, trig, arm):
    """Warning tables pooled over the basins at one warning level per lead {lead: multiple}, as flood.evaluate pools them."""
    a = ARMS[arm]
    tabs = {g: {lk(k): F.flood_scores(P[g], a["thr"][g], k, part, trig[k], a["split"]) for k in F.LEADS} for g in P}
    return {lk(k): {kind: pool([tabs[g][lk(k)][kind] for g in P]) for kind in KINDS} for k in F.LEADS}


def sweep(P, part, arm):
    """Pooled warning tables for every warning level of LEVELS: {lead: {kind: [table per level]}}, counted with flood.contingency on the days flood.flood_scores selects."""
    a = ARMS[arm]
    lo_, hi_ = a["split"][part]
    out = {}
    for k in F.LEADS:
        cells = []
        for g, p in P.items():
            low, q = (p.obs.shift(k) <= a["thr"][g]).loc[lo_:hi_].values, p.loc[lo_:hi_]
            cells.append(((q.obs > a["thr"][g]).values, q[lk(k)].values, low, a["thr"][g]))
        out[lk(k)] = dict(all=[pool([F.contingency(o, f > m * t) for o, f, w, t in cells]) for m in LEVELS], onset=[pool([F.contingency(o[w], (f > m * t)[w]) for o, f, w, t in cells]) for m in LEVELS])
    return out


def choose(tabs):
    """Index of the level with the best pooled CSI; a tie goes to the level closest to the high-flow level itself."""
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in tabs])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])


rec_of = lambda t: dict(verdict=verdict(t), **CUT(t))


def verdict_record(P, E_test, arm, seed_labels=None, seed_evals=None):
    """The three verdict readings of one prediction set, per lead: at the plain level (trigger 1.0) and at the two levels chosen on the validation years of its split; with seed_evals, also the verdict of every single seed."""
    sv, plain = sweep(P, "val", arm), E_test["summary"]
    chosen = {}
    for label, crit in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
        trig = {k: LEVELS[choose(sv[lk(k)][crit])] for k in F.LEADS}
        chosen[label] = (trig, score(P, "test", trig, arm))
    return {lk(k): dict(
        plain=dict(trigger=1.0, **{kind: rec_of(plain[lk(k)][kind]) for kind in KINDS}),
        **{label: dict(trigger=trig[k], chosen_by=f"best pooled validation-years CSI of {'all high-flow days' if 'all_days' in label else 'onsets'}", **{kind: rec_of(sc_[lk(k)][kind]) for kind in KINDS})
           for label, (trig, sc_) in chosen.items()},
        **({"single_seeds_plain": {kind: {str(s): verdict(e["summary"][lk(k)][kind]) for s, e in zip(seed_labels, seed_evals)} for kind in KINDS}} if seed_evals is not None else {})) for k in F.LEADS}


verdicts = {n: verdict_record(PRED[n], EV[n]["test"], ARM_OF[n], list(range(len(EV[n]["seeds"]))), EV[n]["seeds"]) for n in SHOWN}
VERDICT_NOTE = {"plain": "trigger 1.0: a warning when the forecast is above the high-flow level itself (the level of every other score in this file)",
                "chosen_on_validation_all_days": "the warning level chosen on the validation years of the set's own split by the best pooled CSI of all high-flow days: the level of the verdicts stored in results/flood_analysis.json",
                "chosen_on_validation_onsets": "the warning level chosen on the validation years of the set's own split by the best pooled CSI of the onsets"}
for n_ in SHOWN:                                                                              # the summary tables at BOTH warning levels: the fixed one (the keys onset and all) and the one chosen on the validation years of the set's own split
    for k_ in F.LEADS:
        for kind_, key_ in (("onset", "chosen_on_validation_onsets"), ("all", "chosen_on_validation_all_days")):
            ch_ = verdicts[n_][lk(k_)][key_]
            summary_test[n_][lk(k_)][f"{kind_}_at_validation_chosen_level"] = dict(trigger=ch_["trigger"], chosen_by=ch_["chosen_by"], **ch_[kind_])
print("verdicts, onsets, plain level, leads 1-3: " + "; ".join(f"{n}: " + "/".join(verdicts[n][lk(k)]["plain"]["onset"]["verdict"] for k in F.LEADS) for n in ("wq", "wqa", "wqaF_sub", "wqaF_clim", "wqaP", "wqf", "wqaF_s")), flush=True)

# ---- the yardstick (rule 2 of the work order): ridge regression and random forest given wqa + the forecast, beside the LSTM (main split) -------------------------------------
SIMPLE_OLD = tuple(n for n in ("base_persistence", "base_linreg_wq", "base_rf_wq", "base_linreg_wqa", "base_rf_wqa") if F.have_preds(n))      # the stored simple models: 02_baselines.py (wq) and script 16 (wq + the AORC rain to midnight)
SP = {n: F.load_preds(n) for n in SIMPLE_OLD}
SP.update({n: SIMPLE[n]["P"] for n in YARD})
ES = {n: dict(test=evaluate_set(P, "main", "test"), val=evaluate_set(P, "main", "val")) for n, P in SP.items()}
assert all(ES[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in ES for part in ("test", "val") for k in F.LEADS), "days without a forecast: the simple models would not be compared on identical days"
assert all(np.allclose(SP[n][g].obs.loc[F.SPLIT["val"][0]:].values, D[g].q_mm.loc[F.SPLIT["val"][0]:].values) for n in SP for g in ids), "observed flow differs"
for n in ES:
    verdicts[n] = verdict_record(SP[n], ES[n]["test"], "main")
READING = {"lstm": "the LSTM is ahead", "lstm_avg": "the LSTM is ahead only as the average of its seeds", "lstm_med": f"the LSTM scores higher, but p is above {P_CUT:g}", "tie": "no clear difference", "simple": "the simple model is ahead"}      # the words of 11_key_figure.py
frames_by_name = lambda n: PRED[n] if n in PRED else SP[n]
test_of = lambda n: EV[n]["test"] if n in EV else ES[n]["test"]


def reading_vs_simple(c):
    """The reading of 11_key_figure.reading() (report section 4.2): the LSTM is ahead only when the simple model is below every single seed, the LSTM is higher in most basins and p is small."""
    most, few = c["n_basins_lstm_higher"] > c["n_basins"] / 2, c["n_basins_lstm_higher"] < c["n_basins"] / 2
    small_p, below = c["wilcoxon_p"] < P_CUT, c["baseline_against_seed_spread"] == "below every seed"
    if most and small_p:
        return "lstm" if below else "lstm_avg"
    if few and small_p:
        return "simple"
    return "lstm_med" if most and below else "tie"


def lstm_against(T, n, k):
    """One LSTM set (seed average and single seeds) against one simple model on the same inputs at lead k, with the reading of section 4.2 (a rule about the NSE); beside it, as a second sentence, the onsets at the plain level (hits, how many
    single trainings are above, the paired counts over the onsets and the comparison basin by basin)."""
    a, b = nse_of(EV[T]["test"], k), nse_of(ES[n]["test"], k)
    sm = ranges[T][lk(k)]["by_seed"]["nse_median"]
    ls, bs = EV[T]["test"]["summary"][lk(k)], ES[n]["test"]["summary"][lk(k)]
    bm, pr = bs["nse_median"], paired(a, b)
    c = dict(lstm_nse_median=ls["nse_median"], baseline_nse_median=bm, difference_of_medians=ls["nse_median"] - bm, n_basins_lstm_higher=pr["n_basins_higher"], n_basins=pr["n_basins"],
             median_nse_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"], lstm_seed_median_min=min(sm), lstm_seed_median_max=max(sm),
             baseline_against_seed_spread="below every seed" if bm < min(sm) else "above every seed" if bm > max(sm) else "inside the seed spread",
             lstm_onset_pod=ls["onset"]["pod"], baseline_onset_pod=bs["onset"]["pod"], lstm_onset_far=ls["onset"]["far"], baseline_onset_far=bs["onset"]["far"], lstm_onset_pod_seed_range=ranges[T][lk(k)]["onset_pod"],
             lstm_csi=ls["all"]["csi"], baseline_csi=bs["all"]["csi"])
    r = reading_vs_simple(c)
    seeds_pod, ov = ranges[T][lk(k)]["by_seed"]["onset_pod"], overlap(T, n, k, frames_by_name)
    pl, pb_ = (np.array([E["basins"][g][lk(k)]["onset"]["pod"] for g in ids], float) for E in (EV[T]["test"], ES[n]["test"]))
    ok = np.isfinite(pl) & np.isfinite(pb_)
    pro = paired(pl[ok], pb_[ok])
    plain = dict(lstm_warned=ls["onset"]["pod"], simple_warned=bs["onset"]["pod"], lstm_hits=ls["onset"]["hits"], simple_hits=bs["onset"]["hits"], onsets=ls["onset"]["hits"] + ls["onset"]["misses"],
                 lstm_seed_range=ranges[T][lk(k)]["onset_pod"], n_lstm_trainings=len(seeds_pod), lstm_trainings_above_the_simple_model=int(sum(x > bs["onset"]["pod"] for x in seeds_pod)),
                 every_lstm_training_above=bool(min(seeds_pod) > bs["onset"]["pod"]), every_lstm_training_below=bool(max(seeds_pod) < bs["onset"]["pod"]),
                 paired_onsets=dict(warned_by_both=ov["warned_by_both"], only_the_lstm=ov["warned_only_by_first"], only_the_simple_model=ov["warned_only_by_second"], sign_test_p=ov["sign_test_p"]),
                 basins=dict(n_basins_lstm_higher=pro["n_basins_higher"], n_basins_compared=pro["n_basins"], median_difference_points=100 * pro["median_nse_difference"], wilcoxon_p=pro["wilcoxon_p"]))
    small = plain["paired_onsets"]["sign_test_p"] < P_CUT
    on_read = "lstm" if plain["every_lstm_training_above"] and small else "simple" if plain["every_lstm_training_below"] and small else "tie"
    pa_ = plain["paired_onsets"]
    sentence = (f"On the onsets at the plain level the LSTM warns of {p1(plain['lstm_warned'])} ({plain['lstm_hits']} of {plain['onsets']}) against {p1(plain['simple_warned'])} ({plain['simple_hits']}); "
                f"{plain['lstm_trainings_above_the_simple_model']} of its {plain['n_lstm_trainings']} trainings are above the simple model; of the onsets warned by only one of the two, {pa_['only_the_lstm']} are the LSTM's and "
                f"{pa_['only_the_simple_model']} the simple model's (sign test, {pv(pa_['sign_test_p'])}; the onsets of one storm are not independent); the LSTM warns of more of the onsets in {plain['basins']['n_basins_lstm_higher']} of "
                f"{plain['basins']['n_basins_compared']} basins ({pv(plain['basins']['wilcoxon_p'])}).")
    return dict(c, reading=r, reading_in_words=READING[r], onsets=dict(plain_level=plain, reading={"lstm": "the LSTM warns of more onsets", "simple": "the simple model warns of more onsets", "tie": "no clear difference in the onsets"}[on_read],
                                                                       reading_rule="every single LSTM training above (or below) the simple model and a paired sign test over the onsets with p below " + f"{P_CUT:g}"), onsets_in_words=sentence)


def rows_of(En_, Eo_):
    """One new-input comparison per lead: the median NSE, the basins won and p, and the onsets (hits, false warnings) of an evaluation with the input and one without."""
    rows = {}
    for k in F.LEADS:
        pr, tn, to = paired(nse_of(En_, k), nse_of(Eo_, k)), En_["summary"][lk(k)]["onset"], Eo_["summary"][lk(k)]["onset"]
        rows[lk(k)] = dict(nse_median_with=med(En_, k), nse_median_without=med(Eo_, k), nse_gain=med(En_, k) - med(Eo_, k), n_basins_higher=pr["n_basins_higher"], n_basins=n_b, median_basin_difference=pr["median_nse_difference"],
                           wilcoxon_p=pr["wilcoxon_p"], onset_warned_with=tn["pod"], onset_warned_without=to["pod"], onset_gain_points=100 * (tn["pod"] - to["pod"]), onset_hits_with=tn["hits"], onset_hits_without=to["hits"],
                           onsets_in_the_test_years=tn["hits"] + tn["misses"], onset_false_with=tn["far"], onset_false_without=to["far"])
    return rows


YS = [("wqaF_sub", "base_linreg_wqaF"), ("wqaF_sub", "base_rf_wqaF"), ("wqaF_sub", "base_persistence"), ("wqa", "base_linreg_wqa"), ("wqa", "base_rf_wqa")]          # (the LSTM, the simple model on the same inputs)
yardstick = dict(
    what="Rule 2 of NEXT_STEPS_FOR_AGENTS.md: the new input (wqa + the forecast of the three target days) is also given to ridge regression and the random forest, built and fitted by the recipe of 02_baselines.py (features.feature_table for wq, plus the "
         "same rain features of the AORC rain to midnight, plus the forecast rain of the three target days and their sum; one model per basin and lead, fitted on the training years of the main split only, where the slots hold the AORC rain of t+1..t+3 "
         "as in the substitution design of wqaF_sub, and run on the validation and test years with the bias-corrected forecast; the ridge strength and the forest settings are those 02_baselines.py chose on the validation years for the input set wq and are "
         "not chosen again; the test years are only scored). The scores of lstm_wqaF_sub are read beside them in the way 04_lstm_all.py reads the LSTM against the simple models: the single seeds, the basins won, the Wilcoxon p, and the reading of report "
         "section 4.2 (11_key_figure.reading); table_test also holds the simple models of 02_baselines.py and of script 16 (the same recipe without the forecast).",
    recipe=dict(code="code/features.py (feature_table) called twice, once on the daily table and once with the AORC rain to midnight in the place of prcp, plus four forecast features: the hook is feature_table_with in this script; the design and the fit of "
                     "code/02_baselines.py are copied here (simple_design, forecast_simple: a numbered script is not imported) and proved by checks.recipe to give the forecasts that 02_baselines.py stored for the input set wq",
                settings_source="results/baselines.json, models.base_linreg_wq.settings and models.base_rf_wq.settings", ridge=SETTINGS["base_linreg_wqaF"], forest=SETTINGS["base_rf_wqaF"], forest_trees_in_this_run=RF_TREES,
                features=FEATURES, n_features=len(FEATURES), n_features_of_wq=len(feature_table(TABLE["main"][ids[0]], True, 0).columns), forecast_features=list(FC_FEATURES)),
    checks=yard_check, fit_seconds={n: SIMPLE[n]["meta"]["fit_seconds"] for n in YARD}, fit_seconds_total=round(sum(SIMPLE[n]["meta"]["fit_seconds"] for n in YARD), 1),
    sets={n: dict(saved_as=n, test=ES[n]["test"], val=ES[n]["val"]) for n in YARD}, stored_simple_models_used=list(SIMPLE_OLD),
    table_test={lk(k): {n: dict(nse_median=ES[n]["test"]["summary"][lk(k)]["nse_median"], onset=CUT(ES[n]["test"]["summary"][lk(k)]["onset"]), all=CUT(ES[n]["test"]["summary"][lk(k)]["all"]),
                                onset_verdict_plain=verdict(ES[n]["test"]["summary"][lk(k)]["onset"]), val_nse_median=ES[n]["val"]["summary"][lk(k)]["nse_median"]) for n in ES} for k in F.LEADS},
    lstm_beside={lk(k): {T: dict(nse_median=EV[T]["test"]["summary"][lk(k)]["nse_median"], nse_seed_range=ranges[T][lk(k)]["nse_median"], onset=CUT(EV[T]["test"]["summary"][lk(k)]["onset"]),
                                 onset_seed_range=ranges[T][lk(k)]["onset_pod"], onset_verdict_plain=verdict(EV[T]["test"]["summary"][lk(k)]["onset"])) for T in ("wq", "wqa", "wqaF_sub", "wqaF_clim", "wqaP")} for k in F.LEADS},
    lstm_against_simple={f"lstm_{T}__{n}": {lk(k): lstm_against(T, n, k) for k in F.LEADS} for T, n in YS}, new_input_given_to_each_model={},
    reading_rule="11_key_figure.reading(): the LSTM is ahead only when the simple model is below every single LSTM seed, the LSTM is higher in most basins and p is below " + f"{P_CUT:g}" + "; the simple model is ahead in the mirror case; otherwise no clear difference (or: higher but p not small)")
for kind, (T, new, old_) in {"lstm": ("wqaF_sub", "wqaF_sub", "wqa"), "ridge regression": (None, "base_linreg_wqaF", "base_linreg_wqa"), "random forest": (None, "base_rf_wqaF", "base_rf_wqa")}.items():
    En, Eo = (EV[new]["test"], EV[old_]["test"]) if T else (ES[new]["test"], ES[old_]["test"])
    yardstick["new_input_given_to_each_model"][kind] = dict(what=f"{'lstm_wqaF_sub against lstm_wqa (seed averages)' if T else new + ' against ' + old_ + ' (one fit each)'}: what the forecast adds to this model, beside the rain to midnight", by_lead=rows_of(En, Eo))

# ---- the operating point, the trigger-free scores, the block bootstrap and the facts behind the 1-day reading (added after the independent review of E2: results/review_forecast_input.json) ------------
# Everything here is read from the saved prediction sets and the tables (no network is trained); it describes the results, it chooses nothing.
from sklearn.metrics import average_precision_score, roc_auc_score

B_AP, B_BLOCK = (40, 200) if F.QUICK else (1000, 4000)         # replicates of the bootstrap of the average precision and of the block bootstrap of the onset shares (the review's settings)
BLOCK_DAYS, BOOT_SEED = 30, 20261008                           # blocks of 30 consecutive target days (they keep the correlation of neighbouring days and of the basins of one storm that a test over onsets ignores)
ALL_SCORED = list(SHOWN) + list(ES)
ARM_ALL = {**ARM_OF, **{n: "main" for n in ES}}
frames_all = lambda n: PRED[n] if n in PRED else SP[n]
FAR_CAPS = (0.40, 0.50)                                        # equal false-warning budgets: the share of the onset warnings that may be false


def onset_rows(P, k, arm):
    """The onset subset of the arm's test years pooled over the basins in the order of ids (the issue-day flow at or below the high-flow level): the target day, whether the target day is high (an onset), and the forecast flow divided by the
    basin's high-flow level (so that the 12 basins can be pooled)."""
    a, b = ARMS[arm]["split"]["test"]
    thr = ARMS[arm]["thr"]
    dates, high, score = [], [], []
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(k) <= thr[g]).loc[a:b].values
        q = p.loc[a:b]
        ok = (q.obs.notna() & q[lk(k)].notna()).values & was_low
        dates.append(q.index[ok].values), high.append((q.obs > thr[g]).values[ok]), score.append((q[lk(k)] / thr[g]).values[ok])
    return pd.DatetimeIndex(np.concatenate(dates)), np.concatenate(high), np.concatenate(score)


OR = {n: {k: onset_rows(frames_all(n), k, ARM_ALL[n]) for k in F.LEADS} for n in ALL_SCORED}
for n in ALL_SCORED:                                           # every set of an arm is scored on the same rows
    ref = next(m for m in ALL_SCORED if ARM_ALL[m] == ARM_ALL[n])
    assert all((OR[n][k][0] == OR[ref][k][0]).all() and (OR[n][k][1] == OR[ref][k][1]).all() for k in F.LEADS), f"{n}: the onset rows differ from those of {ref}"
AP = {n: {k: dict(ap=float(average_precision_score(OR[n][k][1], OR[n][k][2])), roc_auc=float(roc_auc_score(OR[n][k][1], OR[n][k][2]))) for k in F.LEADS} for n in ALL_SCORED}
SWEEP_TEST = {n: sweep(frames_all(n), "test", ARM_ALL[n]) for n in ALL_SCORED}          # pooled warning tables at every warning level, TEST years (an optimistic bound that treats the two sets of a comparison alike)


def pod_at_far_cap(n, k, cap):
    """The highest share of onsets warned at any warning level whose share of false warnings among the onset warnings is at most cap (test years)."""
    v = [t["pod"] for t in SWEEP_TEST[n][lk(k)]["onset"] if t["far"] == t["far"] and t["far"] <= cap]
    return max(v) if v else float("nan")


# the bootstrap of the average precision, in the order of the review (main split: lead by lead, pair by pair; then the satellite-era split), then the pairs this script adds
AP_PAIRS = {"main": [("wqaF_sub", "wqa"), ("wqaF_sub", "wqaF_clim"), ("wqaP", "wqaF_sub"), ("base_rf_wqaF", "wqaF_sub"), ("base_linreg_wqaF", "wqaF_sub")],
            "sat": [("wqaF_s", "wqa_s"), ("wqaF_s", "wqaF_clim_s"), ("wqaP_s", "wqaF_s")]}
AP_EXTRA = [("main", "wqF_sub", "wq"), ("main", "base_linreg_wqaF", "base_linreg_wqa"), ("main", "base_rf_wqaF", "base_rf_wqa")]
rng_ap = np.random.default_rng(BOOT_SEED)
APD = {}


def ap_difference(T, C, k, arm):
    """Average precision of T minus that of C on the onset rows, with the 95% interval of a block bootstrap over target days (blocks of BLOCK_DAYS days, B_AP replicates)."""
    dates, high, sT = OR[T][k]
    sC = OR[C][k][2]
    blk = ((dates - pd.Timestamp(ARMS[arm]["split"]["test"][0])).days // BLOCK_DAYS).astype(int)
    nb = int(blk.max()) + 1
    idx_of = [np.flatnonzero(blk == j) for j in range(nb)]
    dd = np.empty(B_AP)
    for r in range(B_AP):
        ii = np.concatenate([idx_of[j] for j in rng_ap.integers(0, nb, nb)])
        h = high[ii]
        dd[r] = average_precision_score(h, sT[ii]) - average_precision_score(h, sC[ii])
    return dict(diff=float(AP[T][k]["ap"] - AP[C][k]["ap"]), ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))], share_of_replicates_above_zero=float((dd > 0).mean()), n_onsets=int(high.sum()))


for arm_ in ("main", "sat"):
    for k_ in F.LEADS:
        for T_, C_ in AP_PAIRS[arm_]:
            APD[T_, C_, k_] = ap_difference(T_, C_, k_, arm_)
for arm_, T_, C_ in AP_EXTRA:
    for k_ in F.LEADS:
        APD[T_, C_, k_] = ap_difference(T_, C_, k_, arm_)

# the block bootstrap of the onset shares at the fixed level (and of the false-warning share and of the median NSE), by water year, as the review did it
BB_PAIRS = [("main", "wqaF_sub", "wqa"), ("main", "wqaF_sub", "wqaF_clim"), ("main", "wqaP", "wqa"), ("main", "wqaP", "wqaF_sub"), ("main", "wqF_sub", "wq"), ("main", "base_linreg_wqaF", "base_linreg_wqa"),
            ("main", "base_rf_wqaF", "base_rf_wqa"), ("main", "wqaF_sub", "base_rf_wqaF"), ("main", "wqaF_sub", "base_linreg_wqaF"), ("sat", "wqaF_s", "wqa_s"), ("sat", "wqaF_s", "wqaF_clim_s"), ("sat", "wqaP_s", "wqaF_s")]
rng_bb = np.random.default_rng(BOOT_SEED)


def block_sums(T, C, k, arm):
    """Per block of target days: the onsets, the hits of T and C and their false warnings (onset subset, fixed level), per basin and block the sums for the NSE, and by water year the onsets and hits."""
    a, b = (pd.Timestamp(x) for x in ARMS[arm]["split"]["test"])
    thr = ARMS[arm]["thr"]
    nb = (b - a).days // BLOCK_DAYS + 1
    ev, nse, wy = np.zeros((nb, 5)), np.zeros((len(ids), nb, 5)), {}
    PT, PC = frames_all(T), frames_all(C)
    for j, g in enumerate(ids):
        qT, qC = PT[g].loc[a:b], PC[g].loc[a:b]
        ok = (qT.obs.notna() & qT[lk(k)].notna() & qC[lk(k)].notna()).values
        date, o, iss = qT.index[ok], qT.obs.values[ok], PT[g].obs.shift(k).loc[a:b].values[ok]
        fT, fC = qT[lk(k)].values[ok], qC[lk(k)].values[ok]
        blk = ((date - a).days // BLOCK_DAYS).astype(int)
        on, high, wT, wC = iss <= thr[g], o > thr[g], fT > thr[g], fC > thr[g]
        yr = np.where(date.month >= 10, date.year + 1, date.year)
        for bi in range(nb):
            m = blk == bi
            if not m.any():
                continue
            mo = m & on
            ev[bi] += [(mo & high).sum(), (mo & high & wT).sum(), (mo & high & wC).sum(), (mo & ~high & wT).sum(), (mo & ~high & wC).sum()]
            oo = o[m]
            nse[j, bi] = [m.sum(), oo.sum(), (oo ** 2).sum(), ((fT[m] - oo) ** 2).sum(), ((fC[m] - oo) ** 2).sum()]
        for y in np.unique(yr):
            m = (yr == y) & on & high
            h = wy.setdefault(int(y), [0, 0, 0])
            h[0] += int(m.sum())
            h[1] += int((m & wT).sum())
            h[2] += int((m & wC).sum())
    return ev, nse, wy


def boot_stat(ev, nse):
    """Difference of the onset shares, of the false-warning shares and of the median NSE (T minus C) from sums over blocks (ev [..., 5], nse [basins, ..., 5])."""
    n, hT, hC, fT, fC = (ev[..., i] for i in range(5))
    farT = np.where(hT + fT > 0, fT / np.maximum(hT + fT, 1), np.nan)
    farC = np.where(hC + fC > 0, fC / np.maximum(hC + fC, 1), np.nan)
    N, S, SS, ET, EC = (nse[..., i] for i in range(5))
    sst = SS - S ** 2 / N
    return hT / n - hC / n, farT - farC, np.median(1 - ET / sst, axis=0) - np.median(1 - EC / sst, axis=0)


BB = {}
for arm_, T_, C_ in BB_PAIRS:
    for k_ in F.LEADS:
        ev_, nse_, wy_ = block_sums(T_, C_, k_, arm_)
        nb_ = ev_.shape[0]
        d_pod, d_far, d_nse = boot_stat(ev_.sum(0), nse_.sum(1))
        cnt_ = np.stack([np.bincount(r_, minlength=nb_) for r_ in rng_bb.integers(0, nb_, size=(B_BLOCK, nb_))])
        bp, bf, bn = boot_stat(cnt_ @ ev_, np.einsum("rb,jbk->jrk", cnt_.astype(float), nse_))
        ci_ = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        yrs_ = {y: 100 * (v[1] - v[2]) / v[0] for y, v in sorted(wy_.items()) if v[0] > 0}
        BB[T_, C_, k_] = dict(arm=arm_, n_onsets=int(ev_[:, 0].sum()), pod_points=float(100 * d_pod), pod_points_ci95=[100 * x for x in ci_(bp)], share_of_replicates_pod_above_zero=float((bp > 0).mean()),
                              far_points=float(100 * d_far), far_points_ci95=[100 * x for x in ci_(bf)], nse=float(d_nse), nse_ci95=ci_(bn), share_of_replicates_nse_above_zero=float((bn > 0).mean()),
                              water_years_points=yrs_, water_years_above_zero=int(sum(v > 0 for v in yrs_.values())), n_water_years=len(yrs_))


def operating_view(T, C):
    """T against C at every lead, read at the fixed level, at the level chosen on the validation years, at equal false-warning budgets and without a trigger (average precision), with the block-bootstrap intervals."""
    out = {}
    for k in F.LEADS:
        pT, pC = test_of(T)["summary"][lk(k)]["onset"], test_of(C)["summary"][lk(k)]["onset"]
        vT, vC = verdicts[T][lk(k)]["chosen_on_validation_onsets"], verdicts[C][lk(k)]["chosen_on_validation_onsets"]
        ref_far = pC["far"]
        ap_ = APD.get((T, C, k))
        bb_ = BB.get((T, C, k))
        out[lk(k)] = dict(
            fixed_level=dict(trigger=1.0, points=100 * (pT["pod"] - pC["pod"]), onsets_warned_with=pT["pod"], onsets_warned_without=pC["pod"], false_with=pT["far"], false_without=pC["far"], csi_with=pT["csi"], csi_without=pC["csi"]),
            level_chosen_on_the_validation_years=dict(trigger_with=vT["trigger"], trigger_without=vC["trigger"], points=100 * (vT["onset"]["pod"] - vC["onset"]["pod"]), onsets_warned_with=vT["onset"]["pod"], onsets_warned_without=vC["onset"]["pod"],
                                                      false_with=vT["onset"]["far"], false_without=vC["onset"]["far"], csi_with=vT["onset"]["csi"], csi_without=vC["onset"]["csi"]),
            equal_false_warning_budgets=dict(how_to_read="the highest share of onsets warned at any warning level (test years) with at most this share of false warnings among the onset warnings, for each set; an optimistic bound that treats the two sets alike",
                                             points_at_most_40=100 * (pod_at_far_cap(T, k, 0.40) - pod_at_far_cap(C, k, 0.40)), points_at_most_50=100 * (pod_at_far_cap(T, k, 0.50) - pod_at_far_cap(C, k, 0.50)),
                                             reference_false_share=ref_far, points_at_the_reference_false_share=100 * (pod_at_far_cap(T, k, ref_far) - pod_at_far_cap(C, k, ref_far)),
                                             onsets_warned_with_at_most_40=pod_at_far_cap(T, k, 0.40), onsets_warned_without_at_most_40=pod_at_far_cap(C, k, 0.40), onsets_warned_with_at_most_50=pod_at_far_cap(T, k, 0.50), onsets_warned_without_at_most_50=pod_at_far_cap(C, k, 0.50)),
            trigger_free=dict(average_precision_with=AP[T][k]["ap"], average_precision_without=AP[C][k]["ap"], roc_area_with=AP[T][k]["roc_auc"], roc_area_without=AP[C][k]["roc_auc"], average_precision_difference=ap_["diff"] if ap_ else None,
                              average_precision_difference_ci95=ap_["ci95"] if ap_ else None, n_onsets=ap_["n_onsets"] if ap_ else None),
            block_bootstrap_fixed_level=bb_)
    return out


# the facts behind the 1-day reading of E2a (lstm_wqaF_sub against lstm_wqa), read from the saved sets, the tables and the AORC series
def onset_frame(P, k, arm):
    """The onsets of the arm's test years (an onset: the target day is high and the issue-day flow is at or below the level) as rows: basin, target day, whether the forecast set warned at the fixed level."""
    a, b = ARMS[arm]["split"]["test"]
    thr = ARMS[arm]["thr"]
    rows = []
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(k) <= thr[g]).loc[a:b].values
        q = p.loc[a:b]
        ok = (q.obs.notna() & q[lk(k)].notna()).values
        m = (q.obs > thr[g]).values[ok] & was_low[ok]
        rows.append(pd.DataFrame(dict(gauge=g, target=q.index[ok][m], warned=(q[lk(k)] > thr[g]).values[ok][m])))
    return pd.concat(rows, ignore_index=True)


def lead1_facts():
    OF = {n: onset_frame(PRED[n], 1, "main") for n in ("wqa", "wqaF_sub", PERFECT_VARIANT, "wqaF_clim", RAW_VARIANT)}
    E = OF["wqa"][["gauge", "target"]].copy()
    for n, f in OF.items():
        assert (f.gauge.values == E.gauge.values).all() and (f.target.values == E.target.values).all()
        E[n] = f.warned.values
    E["aorc"] = [float(AORC[r.gauge].loc[r.target]) for r in E.itertuples()]
    E["slot"] = [float(TABLE["main"][r.gauge]["fs1"].loc[r.target - pd.Timedelta(days=1)]) for r in E.itertuples()]
    lost = E[E.wqa & ~E.wqaF_sub]
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    mon = E.target.dt.month
    small, wet = E[E.aorc < 5.0], E[E.aorc >= 5.0]
    by_basin = {g: dict(onsets=int((E.gauge == g).sum()), wqa=float(E[E.gauge == g].wqa.mean()), wqaF_sub=float(E[E.gauge == g].wqaF_sub.mean())) for g in ids}
    zero_f = np.concatenate([TABLE["main"][g]["fs1"].loc[SUB_FROM:].values for g in ids])
    zero_a = np.concatenate([TABLE["main"][g]["p1"].loc[SUB_FROM:].values for g in ids])
    return dict(
        what="lead 1, main split: the 611 onsets of the test years, lstm_wqa against lstm_wqaF_sub at the fixed level, and what the same networks and the AORC rain say about where the difference sits (read from the saved sets; nothing is trained)",
        n_onsets=int(len(E)), warned=dict(no_slots=float(E.wqa.mean()), forecast=float(E.wqaF_sub.mean()), same_networks_given_the_perfect_rain=float(E[PERFECT_VARIANT].mean()), same_networks_given_a_shuffled_forecast=float(E.wqaF_clim.mean()),
                                         same_networks_given_the_raw_forecast=float(E[RAW_VARIANT].mean())),
        paired=dict(both=int((E.wqa & E.wqaF_sub).sum()), only_without_the_forecast=int((E.wqa & ~E.wqaF_sub).sum()), only_with_the_forecast=int((~E.wqa & E.wqaF_sub).sum()), neither=int((~E.wqa & ~E.wqaF_sub).sum())),
        lost_onsets=dict(n=int(len(lost)), warned_again_with_the_perfect_rain=float(lost[PERFECT_VARIANT].mean()), missed_even_with_the_perfect_rain=int((~lost[PERFECT_VARIANT]).sum()), share_missed_even_with_the_perfect_rain=float((~lost[PERFECT_VARIANT]).mean())),
        little_rain_on_the_target_day=dict(rule="AORC rain of the target day below 5 mm", n=int(len(small)), no_slots=float(small.wqa.mean()), forecast=float(small.wqaF_sub.mean()), perfect_rain=float(small[PERFECT_VARIANT].mean()),
                                           n_wet=int(len(wet)), wet_no_slots=float(wet.wqa.mean()), wet_forecast=float(wet.wqaF_sub.mean()), wet_perfect_rain=float(wet[PERFECT_VARIANT].mean())),
        by_season={s: dict(n=int(mon.isin(m).sum()), no_slots=float(E[mon.isin(m)].wqa.mean()), forecast=float(E[mon.isin(m)].wqaF_sub.mean()), points=float(100 * (E[mon.isin(m)].wqaF_sub.mean() - E[mon.isin(m)].wqa.mean())),
                       perfect_rain=float(E[mon.isin(m)][PERFECT_VARIANT].mean())) for s, m in seasons.items()},
        by_basin=dict(basins_where_the_forecast_model_warns_of_fewer=int(sum(v["wqaF_sub"] < v["wqa"] for v in by_basin.values())), basins_where_it_warns_of_more=int(sum(v["wqaF_sub"] > v["wqa"] for v in by_basin.values())), per_basin=by_basin),
        forecast_on_the_onset_days=dict(mean_ratio_to_the_aorc_rain=float(E.slot.sum() / E.aorc.sum()), n_days_with_aorc_rain_of_at_least_5_mm=int(len(wet)), median_ratio_on_those_days=float((wet.slot / wet.aorc).median())),
        slot_values_exactly_zero=dict(forecast=float((zero_f == 0).mean()), aorc=float((zero_a == 0).mean()), where="the slot of lead 1, issue days from 2000-10-01 on (validation and test years), 12 basins pooled"))


LEAD1 = lead1_facts()
print(f"lead 1 facts: {LEAD1['lost_onsets']['n']} onsets warned only without the forecast, {LEAD1['lost_onsets']['missed_even_with_the_perfect_rain']} of them missed even with the perfect rain; small-rain onsets {LEAD1['little_rain_on_the_target_day']['n']}", flush=True)


# the bias factor and the forecast's conditional bias: how noisy the factor is, what bias is left, how light the forecast is on wet days (read from the forecast tables of code/gefs_to_basins.py)
def bias_facts():
    halves = ((pd.Timestamp("2000-10-01"), pd.Timestamp("2003-03-31")), (pd.Timestamp("2003-04-01"), pd.Timestamp("2005-09-30")))
    out = dict(halves_of_the_validation_years=[[str(a.date()), str(b.date())] for a, b in halves], median_over_basins_of_the_absolute_difference_of_the_two_halves_factors={}, median_relative_difference={}, wet_days_test_years={})
    for k in KS:
        d_abs, d_rel, wet_n = [], [], 0
        for g in ids:
            c = FCT[g]
            tgt = c.index + pd.Timedelta(days=k)
            f = [c.loc[(tgt >= a) & (tgt <= b), f"aorc_{k}"].sum() / c.loc[(tgt >= a) & (tgt <= b), f"fc_raw_{k}"].sum() for a, b in halves]
            d_abs.append(abs(f[0] - f[1])), d_rel.append(abs(f[0] - f[1]) / GBR["bias"]["factors"][g][lk(k)])
        out["median_over_basins_of_the_absolute_difference_of_the_two_halves_factors"][lk(k)] = float(np.median(d_abs))
        out["median_relative_difference"][lk(k)] = float(np.median(d_rel))
        fc_sum = o_sum = 0.0
        for g in ids:
            c = FCT[g]
            tgt = c.index + pd.Timedelta(days=k)
            m = (tgt >= F.SPLIT["test"][0]) & (tgt <= F.SPLIT["test"][1]) & (c[f"aorc_{k}"].values >= 5.0)
            fc_sum += float(c.loc[m, f"fc_bc_{k}"].sum())
            o_sum += float(c.loc[m, f"aorc_{k}"].sum())
            wet_n += int(m.sum())
        out["wet_days_test_years"][lk(k)] = dict(n_basin_days=wet_n, bias_corrected_forecast_over_aorc_rain=float(fc_sum / o_sum))
    s = GBR["skill"]["summary"]
    out["seasonal_bias_left"] = dict(raw_mean_ratio_by_season_lead1_all_days=s["mean_ratio_by_season_median"]["l1"],
                                     bias_corrected_mean_ratio_by_season_lead1_test_years=({sn: float(np.nanmedian([GBR["skill"]["per_basin"][g]["l1"]["by_season"][sn]["mean_ratio_bias_corrected_test_years"] for g in ids])) for sn in SEASONS_ALL}))
    return out


SEASONS_ALL = ("DJF", "MAM", "JJA", "SON")
BIAS_FACTS = bias_facts()
for k_ in F.LEADS:                                                                            # the yardstick table at both warning levels and without a trigger
    for n_ in ES:
        yardstick["table_test"][lk(k_)][n_]["onset_at_validation_chosen_level"] = dict(trigger=verdicts[n_][lk(k_)]["chosen_on_validation_onsets"]["trigger"], **verdicts[n_][lk(k_)]["chosen_on_validation_onsets"]["onset"])
        yardstick["table_test"][lk(k_)][n_]["average_precision_onsets"], yardstick["table_test"][lk(k_)][n_]["roc_area_onsets"] = AP[n_][k_]["ap"], AP[n_][k_]["roc_auc"]
readings["value_of_the_forecast"]["operating_points"] = operating_view("wqaF_sub", "wqa")
readings["forecast_vs_climatology_control"]["operating_points"] = operating_view("wqaF_sub", "wqaF_clim")
readings["retrained_vs_substituted"]["retrained_operating_points"] = operating_view("wqaF_s", "wqa_s")
readings["retrained_vs_substituted"]["retrained_against_its_control_operating_points"] = operating_view("wqaF_s", "wqaF_clim_s")
readings["forecast_without_rain_to_midnight"]["operating_points"] = operating_view("wqF_sub", "wq")
readings["how_far_from_perfect"]["operating_points_perfect_rain_minus_forecast"] = operating_view("wqaP", "wqaF_sub")
yardstick["operating_points_simple_models_minus_lstm"] = dict(ridge_minus_lstm=operating_view("base_linreg_wqaF", "wqaF_sub"), forest_minus_lstm=operating_view("base_rf_wqaF", "wqaF_sub"))
yardstick["forecast_added_to_each_model_operating_points"] = dict(ridge=operating_view("base_linreg_wqaF", "base_linreg_wqa"), forest=operating_view("base_rf_wqaF", "base_rf_wqa"), lstm=readings["value_of_the_forecast"]["operating_points"])
OPS_BY_SET = dict(
    how_to_read="every set and lead on its own split's test years, the onsets only: (1) at the fixed warning level (trigger 1.0); (2) at the level chosen on the validation years of the set's own split (the best pooled CSI of the onsets; the second reading of the verdicts); (3) the highest share of onsets "
                "warned at any level with at most 40% (or 50%) of the onset warnings false (test years: an optimistic bound that treats the sets alike); (4) without a trigger: the average precision and the ROC area of the forecast flow divided by the basin's high-flow level, on the onset rows (issue-day flow at or below the level; label = the target day is above it)",
    sets={n_: {lk(k_): dict(arm=ARM_ALL[n_], fixed_level=dict(trigger=1.0, **CUT(test_of(n_)["summary"][lk(k_)]["onset"])),
                            level_chosen_on_the_validation_years=dict(trigger=verdicts[n_][lk(k_)]["chosen_on_validation_onsets"]["trigger"], **verdicts[n_][lk(k_)]["chosen_on_validation_onsets"]["onset"]),
                            onsets_warned_with_at_most_40_percent_false=pod_at_far_cap(n_, k_, 0.40), onsets_warned_with_at_most_50_percent_false=pod_at_far_cap(n_, k_, 0.50),
                            average_precision=AP[n_][k_]["ap"], roc_area=AP[n_][k_]["roc_auc"], n_onsets=int(OR[n_][k_][1].sum()), n_rows=int(len(OR[n_][k_][1]))) for k_ in F.LEADS} for n_ in ALL_SCORED})
UNCERT = dict(
    how_to_read="block bootstrap over TARGET DAYS (blocks of 30 consecutive days drawn with replacement; the blocks keep the correlation of neighbouring days and of the basins of one storm, which the paired sign test over onsets ignores) of the test years of each pair's own split. "
                "block_bootstrap: the difference (first minus second) in the share of onsets warned at the FIXED level (points), in the share of false warnings among the onset warnings (points) and in the median NSE over the basins, with 95% intervals (2.5th to 97.5th percentile) and the "
                "share of replicates above zero, and by water year how many test years the difference is above zero in; average_precision_bootstrap: the difference in average precision of the onset warnings (no trigger) with its interval.",
    settings=dict(block_days=BLOCK_DAYS, replicates_block_bootstrap=B_BLOCK, replicates_average_precision=B_AP, seed=BOOT_SEED, quick_mode=F.QUICK),
    block_bootstrap={f"{T_} minus {C_}": {lk(k_): BB[T_, C_, k_] for k_ in F.LEADS} for _, T_, C_ in BB_PAIRS},
    average_precision_bootstrap={f"{T_} minus {C_}": {lk(k_): APD[T_, C_, k_] for k_ in F.LEADS} for (T_, C_, _k) in sorted({(a_, b_, 1) for (a_, b_, _) in APD})})
FORWARD_JSON = dict(how_to_read="forward passes of the SAVED wqaF_sub networks (" + NUMBER_WORD.get(len(SEEDS), str(len(SEEDS))) + " seeds averaged) on other slot contents, scored like every set (test years, fixed level); INFORMATION ONLY: nothing was retrained, saved or chosen on it. w = 0 is the perfect rain and w = 1 the forecast "
                                "(asserted equal to the scores of lstm_wqaF_sub_perfectpred and lstm_wqaF_sub); below_c_mm_set_to_zero: the forecast values below c mm set to 0 from the first validation day on", tests=FORWARD)


# ---- the ladder of information: one entry per rung, with the controls and the seed ranges (the figure draws it) --------------------------------------------------------------
LADDER_DEF = {"main": dict(label="Main split (test years 2005-2014)", rungs=[
                  ("rain table + river gauge (Daymet)", "wq", None, "the rain table (a day that ends at about 09:00 local time)"),
                  ("+ rain to midnight (AORC)", "wqa", None, "the AORC rain of the whole local day t (E1b)"),
                  ("+ GEFS forecast, substituted", "wqaF_sub", "wqaF_clim", "wqa + the bias-corrected GEFS control-member rain forecast of the days t+1..t+3, trained on the AORC rain of those days (an analysis made after the fact)"),
                  ("perfect rain of the 3 days", "wqaP", None, "wqa + the AORC rain of the days t+1..t+3 (a perfect rain forecast)"),
                  ("perfect weather (upper bound)", "wqf", None, "the Daymet weather stamped t+1..t+3: an upper bound that cannot be had in practice")]),
              "sat": dict(label="Satellite-era split (test years 2010-2014): its own scores, not comparable with the main split", rungs=[
                  ("rain table + river gauge (Daymet)", "wq_s", None, "the rain table, retrained on the satellite-era split"),
                  ("+ rain to midnight (AORC)", "wqa_s", None, "wqa, retrained on the satellite-era split"),
                  ("+ GEFS forecast, retrained", "wqaF_s", "wqaF_clim_s", "wqa + the bias-corrected forecast in every year, trained on it"),
                  ("perfect rain of the 3 days", "wqaP_s", None, "wqa + the AORC rain of the days t+1..t+3 (a perfect rain forecast), satellite-era split")])}
EXTRAS = {"wqF_sub": "the forecast on the plain gauge model, without the rain to midnight (substituted; main split)"}
rung_entry = lambda n, k: {"nse_median": med(EV[n]["test"], k), "nse_seed_range": ranges[n][lk(k)]["nse_median"], "onset_warned": ons(EV[n]["test"], k), "onset_seed_range": ranges[n][lk(k)]["onset_pod"],
                           "onset_false": ons(EV[n]["test"], k, "far"), "n_seeds": ranges[n][lk(k)]["n_seeds"], "n_onsets": EV[n]["test"]["summary"][lk(k)]["onset"]["hits"] + EV[n]["test"]["summary"][lk(k)]["onset"]["misses"],
                           "onset_warned_at_validation_chosen_level": verdicts[n][lk(k)]["chosen_on_validation_onsets"]["onset"]["pod"], "onset_false_at_validation_chosen_level": verdicts[n][lk(k)]["chosen_on_validation_onsets"]["onset"]["far"],
                           "validation_chosen_trigger": verdicts[n][lk(k)]["chosen_on_validation_onsets"]["trigger"]}
ladder = {}
for arm, spec in LADDER_DEF.items():
    rungs = []
    for label, name, ctrl, reads in spec["rungs"]:
        rungs.append(dict(rung=label, set=f"lstm_{name}", control_set=f"lstm_{ctrl}" if ctrl else None, reads=reads, by_lead={lk(k): dict(rung_entry(name, k), **({f"control_{c_}": v_ for c_, v_ in rung_entry(ctrl, k).items()} if ctrl else {})) for k in F.LEADS}))
    ladder[arm] = dict(label=spec["label"], arm=arm, test_years=ARMS[arm]["test_years"], rungs=rungs,
                       extras=[dict(set=f"lstm_{n}", what=w, by_lead={lk(k): rung_entry(n, k) for k in F.LEADS}) for n, w in EXTRAS.items()] if arm == "main" else [])

# E2's success rule (NEXT_STEPS_FOR_AGENTS.md, E2): "a clear gain in onsets warned at 2 and 3 days ahead, where the room is largest"; read as every single training of wqaF_sub above every single training of wqa and of the control
e2_success = dict(
    rule="NEXT_STEPS_FOR_AGENTS.md, E2: a clear gain in onsets warned at 2 and 3 days ahead, where the room is largest. Read here as: at that lead every single training of lstm_wqaF_sub warns of more onsets than every single training of lstm_wqa (no forecast) "
         "AND than every single training of the control lstm_wqaF_clim (the criterion of report section 4.2 for onsets: the seeds are the only repetition the pooled onsets have)",
    by_lead={lk(k): dict(onset_warned=ons(EV["wqaF_sub"]["test"], k), onset_warned_wqa=ons(EV["wqa"]["test"], k), onset_warned_control=ons(EV["wqaF_clim"]["test"], k), seed_range=ranges["wqaF_sub"][lk(k)]["onset_pod"],
                         every_seed_above_wqa=gain_rule["wqaF_sub_against_wqa"][lk(k)]["onset"]["single_seeds_all_above"], every_seed_above_the_control=gain_rule["wqaF_sub_against_wqaF_clim"][lk(k)]["onset"]["single_seeds_all_above"],
                         met=bool(gain_rule["wqaF_sub_against_wqa"][lk(k)]["onset"]["single_seeds_all_above"] and gain_rule["wqaF_sub_against_wqaF_clim"][lk(k)]["onset"]["single_seeds_all_above"])) for k in F.LEADS})
e2_success["met_at_2_and_3_days"] = bool(e2_success["by_lead"]["l2"]["met"] and e2_success["by_lead"]["l3"]["met"])
e2_success["second_half_of_the_rule_says_little"] = ("the margin over the E2a control is " + "/".join(f"{100 * (e2_success['by_lead'][lk(k)]['onset_warned'] - e2_success['by_lead'][lk(k)]['onset_warned_control']):+.0f}" for k in F.LEADS)
                                                       + " points at 1/2/3 days: the control measures the harm of misleading slots to a network that trusts them, not the information of the forecast, so 'above the control' is met by a wide margin whatever the forecast is worth; the first half (above lstm_wqa) carries the rule")

# ---- the one sentence, its caveat (every clause chosen from the numbers), and the quoting rule ----------------------------------------------------------------------------------
def per_lead(f, spec=".2f", sep="/"):
    """f(k) for k = 1, 2, 3 formatted with spec and joined (the numbers of the three leads in one string)."""
    return sep.join(format(f(k), spec) for k in F.LEADS)


def pc_lead(f, sep="/"):
    return sep.join(pc(f(k)) for k in F.LEADS)


def words_lead(f):
    """'a, b and c' for the three leads."""
    v = [f(k) for k in F.LEADS]
    return f"{v[0]}, {v[1]} and {v[2]}"


on = lambda n: pc_lead(lambda k: ons(EV[n]["test"], k))                                               # onsets warned at leads 1/2/3, in per cent
ns = lambda n: per_lead(lambda k: med(EV[n]["test"], k))                                              # median NSE at leads 1/2/3
pts = lambda r, k: r[lk(k)]["onsets"]["difference_points"]
gns = lambda r, k: r[lk(k)]["nse"]["gain"]
X = "the GEFS control-member rain forecast for the next three days (bias-corrected) as an input beside the rain to midnight"
V = verdicts["wqaF_sub"]["l1"]["plain"]["onset"]["verdict"]
onc = lambda n: pc_lead(lambda k: verdicts[n][lk(k)]["chosen_on_validation_onsets"]["onset"]["pod"])        # onsets warned at the level chosen on the validation years, leads 1/2/3
vf, cc, ml, rt = readings["value_of_the_forecast"]["by_lead"], readings["forecast_vs_climatology_control"]["by_lead"], readings["value_of_a_misleading_input"]["by_lead"], readings["retrained_vs_substituted"]
fp, nm = readings["how_far_from_perfect"], readings["forecast_without_rain_to_midnight"]
cl_rain, cl_all = closures["wqaF_sub__base_wqa__ceiling_wqaP"]["by_lead"], closures["wqaF_sub__base_wqa__ceiling_wqf"]["by_lead"]
ng = yardstick["new_input_given_to_each_model"]
ov, ovb = readings["value_of_the_forecast"]["operating_points"], readings["retrained_vs_substituted"]["retrained_operating_points"]
ovc = readings["retrained_vs_substituted"]["retrained_against_its_control_operating_points"]
n_seeds_txt = NUMBER_WORD.get(len(SEEDS), len(SEEDS))
vf_pts = words_lead(lambda k: format(pts(vf, k), "+.1f"))
vf_above = words_lead(lambda k: vf[lk(k)]["onsets"]["reading"])
vf_nse = words_lead(lambda k: format(gns(vf, k), "+.3f"))
vf_read_list = [vf[lk(k)]["nse"]["reading"] for k in F.LEADS]
vf_read = words_lead(lambda k: vf[lk(k)]["nse"]["reading"])
nse_read_short = ("'mixed' by the rule of report section 4.2 at all three leads" if all(r_ == "mixed" for r_ in vf_read_list) else "by the rule of report section 4.2: " + vf_read)
outcome = (f"Given {X}, onsets warned 1/2/3 days ahead at the fixed warning level go from {on('wqa')} to {on('wqaF_sub')} (at the level chosen on the validation years: {onc('wqa')} to {onc('wqaF_sub')}; the shuffled-forecast control, {on('wqaF_clim')}, "
           f"is not a neutral baseline - it measures the harm of misleading slots to a network that trusts them - and the clean control, E2b's, equals the model without the forecast); median NSE from {ns('wqa')} to {ns('wqaF_sub')} (control {ns('wqaF_clim')}), {nse_read_short}; the verdict is {V}.")
ap_wo = per_lead(lambda k: ov[lk(k)]["trigger_free"]["average_precision_without"], ".3f")
ap_w = per_lead(lambda k: ov[lk(k)]["trigger_free"]["average_precision_with"], ".3f")
ap_ci_txt = "; ".join(f"{k} day{'s' * (k > 1)}: {ov[lk(k)]['trigger_free']['average_precision_difference']:+.3f} ({ov[lk(k)]['trigger_free']['average_precision_difference_ci95'][0]:+.3f} to {ov[lk(k)]['trigger_free']['average_precision_difference_ci95'][1]:+.3f})" for k in F.LEADS)
pts_chosen = words_lead(lambda k: format(ov[lk(k)]["level_chosen_on_the_validation_years"]["points"], "+.1f"))
bud40 = words_lead(lambda k: format(ov[lk(k)]["equal_false_warning_budgets"]["points_at_most_40"], "+.1f"))
bud50 = words_lead(lambda k: format(ov[lk(k)]["equal_false_warning_budgets"]["points_at_most_50"], "+.1f"))
c1 = (f"At the fixed warning level (trigger 1.0, the level of every score of the ladder) the forecast adds {vf_pts} points of onsets warned at 1, 2 and 3 days ahead against lstm_wqa (reading of the onsets by the seed rule, every single training of one above every single training of the other: {vf_above}) "
      f"and {vf_nse} of median NSE (reading of the NSE by the rule of report section 4.2: {vf_read}); at the level chosen on the validation years the differences in onsets warned are {pts_chosen} points; at equal false-warning budgets (test years, any level; at most 40% of the onset warnings false) "
      f"{bud40} points and (at most 50% false) {bud50}; and the trigger-free average precision of the onset warnings goes from {ap_wo} to {ap_w} (differences with 95% block-bootstrap intervals: {ap_ci_txt}).")
o1 = ov["l1"]
fx1, ch1, bb1 = o1["fixed_level"], o1["level_chosen_on_the_validation_years"], o1["block_bootstrap_fixed_level"]
L1 = LEAD1
season_txt = ", ".join(f"{s_} {L1['by_season'][s_]['points']:+.1f}" for s_ in SEASONS_ALL)
fw1 = {n_: v_["by_lead"]["l1"]["onset_warned"] for n_, v_ in FORWARD.items()}
fw_share = [(FORWARD["w=0"]["by_lead"][lk(k)]["onset_warned"] - FORWARD["w=0.25"]["by_lead"][lk(k)]["onset_warned"]) / (FORWARD["w=0"]["by_lead"][lk(k)]["onset_warned"] - FORWARD["w=1"]["by_lead"][lk(k)]["onset_warned"]) for k in F.LEADS]
fw_zero_max = 100 * max(abs(FORWARD[n_]["by_lead"][lk(k)]["onset_warned"] - FORWARD["w=1"]["by_lead"][lk(k)]["onset_warned"]) for n_ in ("below_0.5_mm_set_to_zero", "below_1_mm_set_to_zero") for k in F.LEADS)
fw_steep = any(x_ > 0.4 for x_ in fw_share)
fw_txt = ("slots = (1 - w) x the perfect rain + w x the forecast, onsets warned at 1 day for w = 0, 0.25, 0.5, 0.75, 1: " + ", ".join(p1(fw1[f"w={w_:g}"]) for w_ in (0.0, 0.25, 0.5, 0.75, 1.0))
          + f"; the first quarter of the way from the perfect rain to the forecast costs {'/'.join(pc(x_) for x_ in fw_share)} of the drop in onsets warned at 1/2/3 days (a linear response would cost 25%), so the response is "
          + ("steep at small departures from exact slots, as a reliance on exactness would predict" if fw_steep else "gradual, not steep at small departures from exact slots, which a reliance on exactness would predict")
          + f"; the forecast values below 0.5 mm set to zero: {p1(fw1['below_0.5_mm_set_to_zero'])}, below 1 mm: {p1(fw1['below_1_mm_set_to_zero'])} (at most {fw_zero_max:.1f} points of difference from the forecast at any lead)")
c1b = (f"The 1-day difference at the fixed level ({fx1['points']:+.1f} points) is a moved operating point and is not established as a loss of skill: the forecast model's warnings are fewer and more often right (false warnings among the onset warnings {pc(fx1['false_with'])} against {pc(fx1['false_without'])}; "
       f"CSI {fx1['csi_with']:.3f} against {fx1['csi_without']:.3f}); at the level chosen on the validation years (trigger {ch1['trigger_with']:.2f} against {ch1['trigger_without']:.2f}) it warns of {pc(ch1['onsets_warned_with'])} against {pc(ch1['onsets_warned_without'])} of the onsets "
       f"({ch1['points']:+.1f} points; false warnings {pc(ch1['false_with'])} against {pc(ch1['false_without'])}; CSI {ch1['csi_with']:.3f} against {ch1['csi_without']:.3f}); a block bootstrap over the test days (30-day blocks) puts the fixed-level difference at {bb1['pod_points']:+.1f} points "
       f"({bb1['pod_points_ci95'][0]:+.1f} to {bb1['pod_points_ci95'][1]:+.1f}), which is borderline, and the paired sign test over onsets counts onsets as independent, which they are not (onsets cluster in storms and across basins).")
lost_ = L1["lost_onsets"]
sm_ = L1["little_rain_on_the_target_day"]
c1c = (f"That the substitution networks over-trust slots that were exact in training is a hypothesis, not a finding. For it: the same networks warn of {pc(L1['warned']['same_networks_given_the_perfect_rain'])} of the 1-day onsets given the perfect rain, {pc(L1['warned']['forecast'])} given the forecast and "
       f"{pc(L1['warned']['same_networks_given_a_shuffled_forecast'])} given a shuffled forecast ({pc(L1['warned']['no_slots'])} with no slots), and the networks retrained with the forecast (E2b) gain {ovb['l1']['fixed_level']['points']:+.1f} points at 1 day with a harmless control. "
       f"Against it, or not separating it from other causes: of the {lost_['n']} onsets that lstm_wqa warns and the forecast model misses, {lost_['missed_even_with_the_perfect_rain']} ({pc(lost_['share_missed_even_with_the_perfect_rain'])}) are missed even when the slots hold the perfect rain; "
       f"the difference at the fixed level sits in onsets with little rain on the target day ({sm_['n']} of {L1['n_onsets']} with under 5 mm: {pc(sm_['no_slots'])} warned without the forecast, {pc(sm_['forecast'])} with it, {pc(sm_['perfect_rain'])} with the perfect rain in the slots), "
       f"follows the seasons (points of onsets warned, forecast model minus lstm_wqa: {season_txt}) and is found in {L1['by_basin']['basins_where_the_forecast_model_warns_of_fewer']} of 12 basins; and the forecast is light where the onsets are (it carries {L1['forecast_on_the_onset_days']['mean_ratio_to_the_aorc_rain']:.2f} of the AORC rain on the onset days "
       f"and {L1['forecast_on_the_onset_days']['median_ratio_on_those_days']:.2f} on those with at least 5 mm, a conditional bias that the single bias factor does not remove) and has {pc(L1['slot_values_exactly_zero']['forecast'])} exact zeros against {pc(L1['slot_values_exactly_zero']['aorc'])} in the AORC rain. "
       f"No experiment here isolates the cause: the alternatives are the warning level, small-rain onsets and a conditional bias of the forecast. Forward passes of the saved networks (forward_pass_tests; information only, nothing retrained): {fw_txt}.")
cc_pts = words_lead(lambda k: format(pts(cc, k), "+.1f"))
ml_pts = words_lead(lambda k: format(pts(ml, k), "+.1f"))
cc_s_pts = words_lead(lambda k: format(ovc[lk(k)]["fixed_level"]["points"], "+.1f"))
c2 = (f"The shuffled-forecast control of E2a is not a neutral baseline: the network was trained on exact slots and collapses when the slots hold noise ({on('wqaF_clim')} against {on('wqa')} with no slots; median NSE {ns('wqaF_clim')} against {ns('wqa')}), so the comparison with it measures the harm of misleading slots to a network that trusts them, "
      f"not the information of the forecast: the forecast is {cc_pts} points above it, which says little, and the second half of the success rule (above the control) is met by a wide margin at every lead. The information test is E2b's control, trained and run on shuffled slots: it equals the model without slots "
      f"({on('wqaF_clim_s')} against {on('wqa_s')}), and the forecast is {cc_s_pts} points above it.")
cl_on = words_lead(lambda k: pc(cl_rain[k - 1]["onset_share_of_gap_closed"]))
cl_ns = words_lead(lambda k: pc(cl_rain[k - 1]["share_of_gap_closed"]))
cl_all_on = words_lead(lambda k: pc(cl_all[k - 1]["onset_share_of_gap_closed"]))
c3 = (f"Of the gap in onsets warned between lstm_wqa and a perfect rain forecast of the three days (lstm_wqaP warns of {on('wqaP')}) the forecast closes {cl_on} at 1, 2 and 3 days ahead at the fixed level, and {cl_ns} of the NSE gap (a negative share: the forecast model is below lstm_wqa); "
      f"against the old perfect-weather upper bound (lstm_wqf: {on('wqf')}) it closes {cl_all_on} of the onset gap beyond lstm_wqa.")
rr = rt["retrained"]
rr_pts = words_lead(lambda k: format(pts(rr, k), "+.1f"))
n_on_s = "/".join(str(EV["wqa_s"]["test"]["summary"][lk(k)]["onset"]["hits"] + EV["wqa_s"]["test"]["summary"][lk(k)]["onset"]["misses"]) for k in F.LEADS)
rr_ch_pts = words_lead(lambda k: format(ovb[lk(k)]["level_chosen_on_the_validation_years"]["points"], "+.1f"))
rr_csi = words_lead(lambda k: format(ovb[lk(k)]["fixed_level"]["csi_with"] - ovb[lk(k)]["fixed_level"]["csi_without"], "+.3f"))
rr_csi_ch = words_lead(lambda k: format(ovb[lk(k)]["level_chosen_on_the_validation_years"]["csi_with"] - ovb[lk(k)]["level_chosen_on_the_validation_years"]["csi_without"], "+.3f"))
trig_s = "/".join(f"{ovb[lk(k)]['level_chosen_on_the_validation_years']['trigger_without']:.2f}" for k in F.LEADS)
far_s = "/".join(pc(ovb[lk(k)]["level_chosen_on_the_validation_years"]["false_without"]) for k in F.LEADS)
rr_ap = words_lead(lambda k: format(ovb[lk(k)]["trigger_free"]["average_precision_difference"], "+.3f"))
c4 = (f"Retrained with the forecast as an input on the shorter satellite-era split (train 2000-2008, 4 test years with {n_on_s} onsets at 1/2/3 days ahead; its own scores, NOT comparable with the main split; its base is weaker: lstm_wqa_s warns of {on('wqa_s')} against {on('wqa')} for lstm_wqa), "
      f"onsets warned go from {on('wqa_s')} (no forecast) to {on('wqaF_s')} at the fixed level (control {on('wqaF_clim_s')}, equal to the model without the forecast; perfect rain {on('wqaP_s')}), {rr_pts} points, against {vf_pts} for the substitution on the main split - the contrast mixes two splits and two bases and is not a measurement. "
      f"At the level chosen on the validation years the E2b differences are {rr_ch_pts} points (the validation-chosen trigger of lstm_wqa_s, the model without the forecast, is {trig_s} at 1/2/3 days, at {far_s} false warnings: where it is below 1 it buys onsets with false warnings, so the fixed level flatters the gain of the forecast there); "
      f"the CSI differences are {rr_csi} at the fixed level and {rr_csi_ch} at the chosen level; the trigger-free average precision gains {rr_ap}. "
      f"The bias factors were fitted on 2000-2005, inside its training years (scalars; nothing of its validation or test years).")
nm_pts = words_lead(lambda k: format(pts(nm["by_lead"], k), "+.1f"))
c5 = f"On the plain gauge model, without the rain to midnight, the same forecast takes onsets warned from {on('wq')} to {on('wqF_sub')} ({nm_pts} points)."
if all(k_ in ng for k_ in ("lstm", "ridge regression", "random forest")):                    # the yardstick: does the gain need the network?
    g_ridge = "/".join(f"{ng['ridge regression']['by_lead'][lk(k)]['onset_gain_points']:+.1f}" for k in F.LEADS)
    g_rf = "/".join(f"{ng['random forest']['by_lead'][lk(k)]['onset_gain_points']:+.1f}" for k in F.LEADS)
    simple_gain_everywhere = all(ng[m_]["by_lead"][lk(k)]["onset_gain_points"] > 0 and ng[m_]["by_lead"][lk(k)]["nse_gain"] > 0 for m_ in ("ridge regression", "random forest") for k in F.LEADS)
    g_lstm = "/".join(f"{ng['lstm']['by_lead'][lk(k)]['onset_gain_points']:+.1f}" for k in F.LEADS)
    s_ridge = pc_lead(lambda k: yardstick["table_test"][lk(k)]["base_linreg_wqaF"]["onset"]["pod"])
    s_rf = pc_lead(lambda k: yardstick["table_test"][lk(k)]["base_rf_wqaF"]["onset"]["pod"])
    r_ridge = yardstick["lstm_against_simple"]["lstm_wqaF_sub__base_linreg_wqaF"]["l1"]["onsets"]["reading"]
    r_rf = yardstick["lstm_against_simple"]["lstm_wqaF_sub__base_rf_wqaF"]["l1"]["onsets"]["reading"]
    n_ridge = yardstick["lstm_against_simple"]["lstm_wqaF_sub__base_linreg_wqaF"]["l1"]["reading_in_words"]
    n_rf = yardstick["lstm_against_simple"]["lstm_wqaF_sub__base_rf_wqaF"]["l1"]["reading_in_words"]
    nse_ridge = per_lead(lambda k: yardstick["table_test"][lk(k)]["base_linreg_wqaF"]["nse_median"])
    nse_rf = per_lead(lambda k: yardstick["table_test"][lk(k)]["base_rf_wqaF"]["nse_median"])
    far_ridge = pc_lead(lambda k: yardstick["table_test"][lk(k)]["base_linreg_wqaF"]["onset"]["far"])
    far_rf = pc_lead(lambda k: yardstick["table_test"][lk(k)]["base_rf_wqaF"]["onset"]["far"])
    far_lstm = pc_lead(lambda k: ons(EV["wqaF_sub"]["test"], k, "far"))
    ap_ridge, ap_rf, ap_lstm = (per_lead(lambda k, n_=n_: AP[n_][k]["ap"], ".3f") for n_ in ("base_linreg_wqaF", "base_rf_wqaF", "wqaF_sub"))
    d_rf = [APD["base_rf_wqaF", "wqaF_sub", k] for k in F.LEADS]
    d_ridge = [APD["base_linreg_wqaF", "wqaF_sub", k] for k in F.LEADS]
    apd_rf = ", ".join(f"{d_['diff']:+.3f}" for d_ in d_rf)
    apd_ridge = ", ".join(f"{d_['diff']:+.3f}" for d_ in d_ridge)
    rf_level = all(d_["ci95"][0] <= 0 <= d_["ci95"][1] for d_ in d_rf)
    ridge_below = all(d_["ci95"][1] < 0 for d_ in d_ridge)
    c6 = (f"Ridge regression and the random forest given the same inputs (the forecast substituted in the same way) warn of {s_ridge} and {s_rf} of the onsets at 1/2/3 days ahead at the fixed level, with a median NSE of {nse_ridge} and {nse_rf} "
          f"(the LSTM: {on('wqaF_sub')}, {ns('wqaF_sub')}; the forecast adds {g_ridge} and {g_rf} points of onsets to them, {g_lstm} to the LSTM; false warnings among the onset warnings: ridge {far_ridge}, forest {far_rf}, LSTM {far_lstm}: they warn of more onsets with more false warnings, a different operating point). "
          f"Without a trigger (average precision of the onset warnings at 1/2/3 days: ridge {ap_ridge}, forest {ap_rf}, LSTM {ap_lstm}) the forest minus the LSTM is {apd_rf} ({'intervals include 0: level with the LSTM' if rf_level else 'some intervals exclude 0'}) and ridge minus the LSTM {apd_ridge} "
          f"({'intervals exclude 0: below the LSTM' if ridge_below else 'some intervals include 0'}); at 1 day ahead, by the paired reading of the onsets at the fixed level: ridge, {r_ridge}; forest, {r_rf}; by the NSE (report section 4.2): against ridge, {n_ridge}; against the forest, {n_rf} - "
          + ("the gain of the forecast, in onsets and in NSE at every lead, does not need the network." if simple_gain_everywhere else "the simple models do not gain at every lead."))
else:
    c6 = ""
a_lo = min(GBR["basins"][g]["outline_area_km2_equal_area"] for g in ids)
a_hi = max(GBR["basins"][g]["outline_area_km2_equal_area"] for g in ids)
mg = GBR["realtime_margin"]["by_zone_and_clock"]
mg_ne4 = mg["America/New_York, run 4 h old at the issue time"]
mg_ne5 = mg["America/New_York, run 5 h old at the issue time"]
mg_ch5, mg_ch6 = mg["America/Chicago, run 5 h old at the issue time"], mg["America/Chicago, run 6 h old at the issue time"]
bias_med = GBR["skill"]["summary"]["bias_factor_median_over_basins"]["l1"]
bh = BIAS_FACTS["median_over_basins_of_the_absolute_difference_of_the_two_halves_factors"]
halves_txt = "/".join(f"{100 * bh[lk(k)]:.0f}%" for k in F.LEADS)
sb = BIAS_FACTS["seasonal_bias_left"]["raw_mean_ratio_by_season_lead1_all_days"]
wet_txt = "/".join(f"{BIAS_FACTS['wet_days_test_years'][lk(k)]['bias_corrected_forecast_over_aorc_rain']:.2f}" for k in F.LEADS)
raw_pts = words_lead(lambda k: format(readings["raw_vs_bias_corrected"]["at_prediction_time"][lk(k)]["onsets"]["difference_points"], "+.1f"))
gq_ = gain_rule["wqaF_sub_against_wqa"]                                                     # the comparison the extra seeds were planned for (E2: wqaF_sub against wqa on the NSE at 2 and 3 days)
few_ = {}
for n_ in sorted(n_ for n_ in ranges if ranges[n_]["l1"]["n_seeds"] != len(SEEDS)):             # the stored controls that keep fewer trainings (lstm_wq and lstm_wqf of 04_lstm_all.py: three)
    few_.setdefault(ranges[n_]["l1"]["n_seeds"], []).append(f"lstm_{n_}")
and_ = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else "".join(xs)
few_txt_ = "; ".join(f"{NUMBER_WORD.get(k_, k_)} for the stored control{'s' * (len(v_) > 1)} {and_(v_)}" for k_, v_ in sorted(few_.items()))
r2_, r3_ = gq_["l2"]["nse"], gq_["l3"]["nse"]
rd_ = (f"is still '{r2_['reading']}'" if r2_["reading"] == r3_["reading"] == "mixed" else f"reads '{r2_['reading']}'" if r2_["reading"] == r3_["reading"] else f"reads '{r2_['reading']}' and '{r3_['reading']}'")
seed_tail = (f"{n_seeds_txt} training{'s' * (len(SEEDS) > 1)} per set of this script{' (' + few_txt_ + ')' if few_txt_ else ''}, so the single-training ranges are the only repetition; "
             + (f"with {n_seeds_txt} the NSE of lstm_wqaF_sub against lstm_wqa at 2 and 3 days {rd_} (higher in {r2_['n_basins_higher']} and {r3_['n_basins_higher']} of {r2_['n_basins']} basins, "
                f"Wilcoxon p = {r2_['wilcoxon_p']:.3f} and {r3_['wilcoxon_p']:.3f})" if len(SEEDS) >= 5 else
                f"the NSE of lstm_wqaF_sub against lstm_wqa at 2 and 3 days reads '{r2_['reading']}' and '{r3_['reading']}' (more seeds may settle it)"))
c7 = (f"Caveats that are part of the result: control member only (no ensemble); 0.25 degree boxes over basins of {a_lo:.0f} to {a_hi:,.0f} km2; the bias factor (one per basin and lead, median {bias_med:.2f}) was fitted on the validation years 2000-2005, is noisy (fitted on the two halves of those years "
      f"the factors differ by a median {halves_txt} at 1/2/3 days) and leaves a seasonal bias (mean ratio forecast / AORC at 1 day: {sb['JJA']:.2f} in summer, {sb['DJF']:.2f} in winter) and a conditional one (the corrected forecast carries {wet_txt} of the AORC rain on days with at least 5 mm at 1/2/3 days); "
      f"raw and corrected forecasts give nearly the same onsets ({raw_pts} points, corrected minus raw), so the correction is not what carries the result; the substitution networks never saw a forecast in training; "
      "the rain to midnight of the base rung is the AORC analysis, made after the fact (corrections about 10 days after the hour; hours before 2002 are disaggregated daily totals), not rain that could be read at midnight - "
      "the independent review of E1b tested the hindsight risk on 78 test-year days (coefficient of the pre-midnight AORC rain on the later Stage IV rain about -0.03, 95% interval -0.06 to -0.001: nothing positive) and found that with Stage IV, a real-time product, in its place for the "
      "issue day the three networks of seeds 0-2 that it tested warned of 86 to 95 of the 171 onsets of those days against 92 with AORC; the other 440 of the 611 onsets were not tested (numbers as the review's verdict states them: results/review_aorc_rain_to_midnight.json); "
      f"in Eastern summer the last files of the run (from forecast hour {int(mg_ne4['first_such_file'][1:])} on) are posted up to {mg_ne4['worst_lateness_minutes']:.0f} minutes after the issue time (results/gefs_basins.json, realtime_margin), and that margin rests on the posting times of the OPERATIONAL "
      "control member on 4 days of October 2026 (results/data_access.json), not on a long record and not on the reforecast archive, which is not an operational archive (its skill is an estimate of the operational skill, to be checked on a few months of operational GEFS before any real-time claim); "
      f"{seed_tail}.")
outcome_caveat = " ".join(x for x in (c1, c1b, c1c, c2, c3, c4, c5, c6, c7) if x)
outcome_full = outcome + " " + outcome_caveat

# ---- the notes: every number in them is read from the tables above ----------------------------------------------------------------------------------------------------------
fcs = GBR["skill"]["summary"]
n_box = [GBR["basins"][g]["n_boxes"] for g in ids]
bias_rng = fcs["bias_factor_range_over_basins"]
sat_split_txt = f"train {SAT_SPLIT['train'][0]}..{SAT_SPLIT['train'][1]}, validation {SAT_SPLIT['val'][0]}..{SAT_SPLIT['val'][1]}, test {SAT_SPLIT['test'][0]}..{SAT_SPLIT['test'][1]}"


def reading_note(head, T, C, by_lead):
    """One note per reading: the three leads, with the seed ranges, the hits, the checks of section 4.2 and the overlap of the warned onsets."""
    parts = []
    for k in F.LEADS:
        r = by_lead[lk(k)]
        o, s_, ov = r["onsets"], r["nse"], r["overlap_of_warned_onsets"]
        by_seed = ", ".join(f"{x:+.1f}" for x in o["difference_by_seed_points"])
        parts.append(f"{k} day{'s' * (k > 1)} ahead: onsets warned {p1(o['warned_with'])} ({span(o['seed_range_with'], p1)}) against {p1(o['warned_without'])} ({span(o['seed_range_without'], p1)}), {o['difference_points']:+.1f} points "
                     f"({o['hits_with']} against {o['hits_without']} hits of {o['onsets_in_the_test_years']} onsets; seed by seed {by_seed} points), every seed above every seed: {yes(o['every_seed_above_every_seed'])}, reading of the onsets: {o['reading']}, "
                     f"false warnings among the onset warnings {pc(o['false_warnings_with'])} against {pc(o['false_warnings_without'])}; median NSE {n3(s_['median_with'])} ({span(s_['seed_range_with'], n3)}) against {n3(s_['median_without'])} "
                     f"({span(s_['seed_range_without'], n3)}), {s_['gain']:+.3f}, higher in {s_['n_basins_higher']} of {n_b} basins ({pv(s_['wilcoxon_p'])}), reading of the NSE: {s_['reading']}; onsets warned by both, only by the first, only by the second: "
                     f"{ov['warned_by_both']}, {ov['warned_only_by_first']}, {ov['warned_only_by_second']} (sign test, {pv(ov['sign_test_p'])}: a description, the onsets of one storm are not independent)")
    return f"{head} (lstm_{T} minus lstm_{C}; test years of the sets' split; forecasts averaged over the seeds, lowest to highest single seed in brackets). " + " | ".join(parts) + "."


first_sets = {n for p_ in timing["passes"][:1] for n in p_.get("sets_trained", [])}
later_sets = sorted({n for p_ in timing["passes"][1:] if p_.get("kind") != "rescore" and not p_.get("seed_extension") for n in p_.get("sets_trained", [])})      # a pass that only added seeds is disclosed in choices_made_after_looking_at_test_scores, not as 'sets trained later'
t_main, t_sat = TABLE_CHECK["main"], TABLE_CHECK["sat"]
ctrl_corr, fc_corr = t_main["median_correlation_with_the_aorc_rain_of_the_day"]["control"], t_main["median_correlation_with_the_aorc_rain_of_the_day"]["forecast"]
ctrl_rng = t_main["control_correlation_range"]
sk_corr, sk_ratio = per_lead(lambda k: fcs["correlation_all_median_over_basins"][lk(k)]), per_lead(lambda k: fcs["mean_ratio_all_median_over_basins"][lk(k)])
season_corr = ", ".join(f"{a} {b:.2f}" for a, b in fcs["correlation_by_season_median"]["l1"].items())
season_ratio = ", ".join(f"{a} {b:.2f}" for a, b in fcs["mean_ratio_by_season_median"]["l1"].items())
season_ratio_val = ", ".join(f"{a} {b:.2f}" for a, b in fcs["mean_ratio_by_season_validation_years_median"]["l1"].items())
bk = fcs["brier_test_years_pooled"]
b5_yn, b5_cal = per_lead(lambda k: bk[lk(k)]["5mm"]["yes_no_raw"]["brier_skill_pooled"]), per_lead(lambda k: bk[lk(k)]["5mm"]["calibrated_probability"]["brier_skill_pooled"])
b10_yn, b10_cal = per_lead(lambda k: bk[lk(k)]["10mm"]["yes_no_raw"]["brier_skill_pooled"]), per_lead(lambda k: bk[lk(k)]["10mm"]["calibrated_probability"]["brier_skill_pooled"])
lag = fcs["lag_check"]
bias_med_l, bias_rng_l = per_lead(lambda k: fcs["bias_factor_median_over_basins"][lk(k)]), "/".join(f"{bias_rng[lk(k)][0]:.2f} to {bias_rng[lk(k)][1]:.2f}" for k in F.LEADS)
raster_dev = max(GBR["basins"][g]["raster_check"]["largest_difference_of_a_weight"] for g in ids)
notes += [
    "Quoting. The sentence in `outcome` is accurate arithmetic but incomplete as a statement of the result: any document (report, README, slides, Q&A, guide) must quote `outcome_full` (the outcome plus `outcome_caveat`), never `outcome` alone. Both are computed from "
    "the tables of this file. The numbers of E2b (the satellite-era split) are comparable only with each other.",
    f"Clock of the forecast columns. Every forecast column is dated by the ISSUE day t: the value on row t is the forecast rain of the local wall-clock day t+k (k = 1, 2, 3; zoneinfo America/New_York, America/Chicago for Leaf River, summer time included, the same day "
    f"convention as the AORC input of script 16) from the 00 UTC run of the UTC date t+1, the latest run that exists at the issue time (local midnight at the end of day t: 04, 05 or 06 UTC of that date; the run is 4 to 6 hours old). code/gefs_to_basins.py asserts for all "
    f"{GBR['issue_days']['n']} issue days and both time zones that the run starts at or before the issue time, that no fresher 00 UTC run exists, that every forecast hour of the three days lies after the issue time and inside the day's UTC span, that the three days follow "
    f"each other hour by hour and that no hour beyond 78 is read; it computes the window weights by a second route (absolute times) and replaces all later runs by random rain (no forecast issued earlier changes); and it checks the hours of the two products "
    f"against each other (results/gefs_basins.json, clock). The engine reads the dyn columns of the days up to and including the issue day; the made-up-values test of this file (causality) replaces the forecast columns (and, in a second round, every input column and the flow) "
    f"after a cut day by random values on the saved networks: no forecast issued on or before the cut day changes, the later ones do. flood.fit also ran flood._assert_causal on every network.",
    f"The substitution design (E2a). The slot columns hold, for the issue days up to 2000-09-30 (the training years; {t_main['n_issue_days_with_aorc_rain_in_the_slots']} days), the AORC rain of the local days t+1..t+3 (an analysis made after the fact: a perfect forecast), "
    f"and from 2000-10-01 (validation and test years; {t_main['n_issue_days_with_the_forecast_in_the_slots']} days) the bias-corrected GEFS forecast of the same days. The networks are trained on the first part only (the trainer samples sequences that end "
    "inside the training years and scores the loss on training target days), so they never see a forecast in training; the trainer scores the validation years from the same table, so its stopping point is chosen with the real forecast in the slots; the test years are scored "
    "with it. The split is by the ISSUE day of the slots, so the forecasts for the first validation target days (2000-10-01 .. 2000-10-03: 1 forecast at lead 1, 2 at lead 2, 3 at lead 3 per basin, 6 of about 5,500) are issued on days whose slots still hold "
    "the AORC rain; the test years are not affected. The GEFS forecast of 2000-01-01 .. 2000-09-30 exists and is not used in E2a (those days are training years). The forecast columns of E2b hold the forecast in every year of its table (2000-01-01 on).",
    f"The control. fk (E2a) and fx (E2b) hold the same forecasts in another order: within each basin and calendar month the rows (the three leads of one day together) are permuted so that no day keeps a forecast of its own calendar year (fixed seeds; asserted: the sorted values of every calendar month are "
    f"the same as the forecast's, no day keeps its year). The median correlation of the control columns with the AORC rain of the day is {ctrl_corr:.3f} (E2a; range {ctrl_rng[0]:.3f} to {ctrl_rng[1]:.3f} over basins and leads), that of the forecast {fc_corr:.3f}: the control has the right "
    "climate of each calendar month (hence a small positive correlation through the seasonal cycle) and no information about the day. It is NOT a neutral baseline for E2a: the E2a networks were trained on exact slots, so given such a control they collapse "
    f"({on('wqaF_clim')} of the onsets against {on('wqa')} with no slots, median NSE {ns('wqaF_clim')} against {ns('wqa')}); the comparison with it measures the harm of misleading slots to a network that trusts them, not the information of the forecast. The information test is E2b's control, "
    f"trained and run on shuffled slots, which equals the model without slots ({on('wqaF_clim_s')} against {on('wqa_s')}).",
    f"The forecast's own skill (results/gefs_basins.json; raw control-member forecast against the AORC rain of the same local days, all days 2000-2014, median over the 12 basins), 1/2/3 days ahead: correlation {sk_corr}, mean ratio forecast / AORC {sk_ratio}; "
    f"by season (lead 1) the correlation is {season_corr} and the mean ratio {season_ratio}. Brier skill of days with at least 5 mm (test years; yes/no forecast of the control member / a calibrated probability fitted on the validation years): {b5_yn} / {b5_cal}; "
    f"at least 10 mm: {b10_yn} / {b10_cal}. The lag check holds (best shift 0 in every basin and lead: {lag['every_basin_and_lead_best_at_zero']}; median correlation at lead 1 {lag['median_correlation_by_shift']['l1']['+0']:.2f} at shift 0 against "
    f"{lag['median_correlation_by_shift']['l1']['-1']:.2f} and {lag['median_correlation_by_shift']['l1']['+1']:.2f} one day earlier and later).",
    f"Bias correction. One factor per basin and lead = mean AORC rain / mean forecast rain over the validation years (target days 2000-10-01 .. 2005-09-30), applied unchanged to every day of 2000-2014, test years included (a factor from earlier years): median over the basins "
    f"{bias_med_l}, range over the basins {bias_rng_l} (the two Florida basins need 1.2 to 1.4: the forecast is drier than AORC there; a cause was not looked for). It is a single factor, so the seasonal bias (mean ratio by season, lead 1, validation years: {season_ratio_val}) "
    "is not removed. In E2b the factor was fitted on 2000-2005, which lies inside that split's training years (a scalar fitted with the training-year rain; nothing of its validation or test years). The raw forecast is scored too: readings.raw_vs_bias_corrected.",
    f"0.25 degree boxes over small basins. The model's grid boxes are about 630 km2; the 12 outlines ({a_lo:.0f} to {a_hi:,.0f} km2) overlap {min(n_box)} to {max(n_box)} boxes each (the weights are the exact area shares in an equal-area projection, recorded in results/gefs_basins.json, basins, "
    f"and checked against a fine raster to within {raster_dev:.3f}); the forecast is the box-mean rain of a model with 0.25 degree physics, smoother than a basin mean of 1 km rain, and the correlation with AORC is lower in summer convection (JJA, lead 1: {fcs['correlation_by_season_median']['l1']['JJA']:.2f}).",
    "Control member only. Only the control member (c00) of the GEFSv12 reforecast was fetched; the archive holds more members (ROADMAP.md: 5 members; data/gefs/README.md: more on Wednesdays), their mean and their spread were not used. A single member is a sharper and noisier forecast than the ensemble mean; the ensemble would give a smoother "
    "amount and, with its spread, a probability, neither of which is tried here. The Brier skills above are those of one member (yes/no, or calibrated), not of an ensemble, and are not a check of the ensemble numbers of Guan et al. (2022).",
    f"The issue-time margin. The 00 UTC run is the latest run that exists at the issue time, but a real-time service would have its files only after the model has run: the operational control member posts its forecast-hour-78 file about 4.26 hours after the run's start (results/data_access.json; "
    f"4 days sampled in October 2026: a short record of the OPERATIONAL system, not the reforecast archive, which has no posting times and is not an operational archive; the skill of the reforecast is an estimate of the operational skill). Margin of the last needed file at the issue time: Eastern winter (run 5 h old) {mg_ne5['margin_minutes_of_the_last_file']:.0f} minutes, Eastern summer (run 4 h old) {mg_ne4['margin_minutes_of_the_last_file']:.0f} minutes "
    f"(the {mg_ne4['files_needed_that_are_posted_after_the_issue_time']} files from hour {int(mg_ne4['first_such_file'][1:])} on are posted up to {mg_ne4['worst_lateness_minutes']:.0f} minutes after it), Central {mg_ch5['margin_minutes_of_the_last_file']:.0f} and {mg_ch6['margin_minutes_of_the_last_file']:.0f} minutes. "
    "The rule 'the run of the UTC date t+1' is kept for every day and basin (the work order's rule), so the experiment is slightly optimistic in Eastern summer; issuing at 00:30 local would restore the margin.",
    f"E2b is a different experiment on a different split ({sat_split_txt}; flood thresholds from its own training years): 8 years of training instead of 20, 4 test years instead of 9, other onsets (see the number of onsets in the readings). Its scores are comparable only with each other (wq_s, wqa_s, wqaF_s, "
    "wqaF_clim_s, wqaP_s: one table starting 2000-01-01, the same days for every set, all with the river gauge and the basin properties) and set beside E2a only as gains (readings.retrained_vs_substituted). The settings are those of results/lstm_all.json, chosen on the validation years of the main split, "
    "not tuned again for the shorter split.",
    "The saved networks and the AORC column. results/models/lstm_<set>_s<seed>.pt holds each network; reload_check rebuilds one per set from its file alone and shows that it gives the stored forecasts again on the table built by this script, and aorc_column_check shows that the saved networks of script 16 (set wqa), given "
    "this script's table, reproduce the forecasts script 16 stored: the column aorc_to_midnight that every set reads here is the column that trained lstm_wqa. To forecast a new day, the columns in dyn must be built the same way (the forecast columns need the 00 UTC run of the next UTC date).",
]
ladder_rows = [("wq", "the rain table + the river gauge"), ("wqa", "+ the AORC rain to midnight"), ("wqaF_clim", "control: the shuffled forecast"), ("wqaF_sub", "+ the GEFS forecast, substituted"), ("wqF_sub", "the forecast without the rain to midnight"),
               ("wqaP", "perfect rain of the three days"), ("wqf", "perfect weather (upper bound)"), ("wq_s", "[own split] rain table + gauge"), ("wqa_s", "[own split] + rain to midnight"), ("wqaF_clim_s", "[own split] control"),
               ("wqaF_s", "[own split] + the forecast, retrained"), ("wqaP_s", "[own split] perfect rain")]
rung_txt = []
for n_, lab_ in ladder_rows:
    t1 = EV[n_]["test"]["summary"]["l1"]["onset"]
    rung_txt.append(f"{lab_}: {on(n_)} ({span(ranges[n_]['l1']['onset_pod'], p1)}; {t1['hits']} of {t1['hits'] + t1['misses']}); NSE {ns(n_)}")
notes.append("The rungs side by side (with the river gauge; test years of each set's own split; onsets warned at 1/2/3 days ahead, seed average, lowest to highest single seed at 1 day in brackets, hits of the onsets at 1 day; median NSE at 1/2/3 days): " + "; ".join(rung_txt) + ".")
for key, T, C, head in (("value_of_the_forecast", "wqaF_sub", "wqa", "Value of the forecast (value_of_the_forecast)"), ("forecast_vs_climatology_control", "wqaF_sub", "wqaF_clim", "Forecast against the shuffled-forecast control (forecast_vs_climatology_control)"),
                        ("value_of_a_misleading_input", "wqaF_clim", "wqa", "The control against no forecast (value_of_a_misleading_input)")):
    notes.append(reading_note(head, T, C, readings[key]["by_lead"]))
notes.append(reading_note("E2b, the forecast retrained (retrained_vs_substituted.retrained; satellite-era split, its own numbers)", "wqaF_s", "wqa_s", rt["retrained"]))
notes.append(reading_note("E2b against its control (retrained_vs_substituted.retrained_against_its_control; satellite-era split)", "wqaF_s", "wqaF_clim_s", rt["retrained_against_its_control"]))
notes.append(reading_note("The forecast without the rain to midnight (forecast_without_rain_to_midnight)", "wqF_sub", "wq", nm["by_lead"]))
notes.append(reading_note("The rain to midnight on top of the forecast (forecast_without_rain_to_midnight.rain_to_midnight_given_the_forecast)", "wqaF_sub", "wqF_sub", nm["rain_to_midnight_given_the_forecast"]))
notes.append(reading_note("How far from a perfect rain forecast (how_far_from_perfect; perfect rain minus the substituted forecast)", "wqaP", "wqaF_sub", fp["perfect_rain_minus_substituted_forecast"]))
notes.append(reading_note("The worth of a perfect rain forecast of the three days (how_far_from_perfect.perfect_rain_minus_rain_to_midnight)", "wqaP", "wqa", fp["perfect_rain_minus_rain_to_midnight"]))
notes.append(reading_note("The same wqaF_sub networks given the PERFECT rain at prediction time (substituted_network_given_perfect_rain: a scored variant, not a training; the first set minus the second is what the forecast's errors cost)", PERFECT_VARIANT, "wqaF_sub",
                          readings["substituted_network_given_perfect_rain"]["perfect_rain_given_minus_forecast_given"]))
notes.append(reading_note("Stopping point chosen with the forecast (stopping_point_chosen_with_the_forecast: wqaF_sub against the wqaP networks, stopped on perfect rain, given the same forecast at prediction time - a scored variant, not a training)", "wqaF_sub", STOP_VARIANT,
                          readings["stopping_point_chosen_with_the_forecast"]["by_lead"]))
notes.append(reading_note("Raw against bias-corrected forecast (raw_vs_bias_corrected.at_prediction_time: the saved wqaF_sub networks given the RAW forecast at prediction time - a scored variant by substitution, not a retraining; "
                          "the first set minus the second is the bias-corrected minus the raw)", "wqaF_sub", RAW_VARIANT, readings["raw_vs_bias_corrected"]["at_prediction_time"]))
if "retrained_on_the_raw_forecast" in readings["raw_vs_bias_corrected"]:
    notes.append(reading_note("Raw against bias-corrected forecast, retrained (raw_vs_bias_corrected.retrained_on_the_raw_forecast)", "wqaF_sub", "wqaF_sub_raw", readings["raw_vs_bias_corrected"]["retrained_on_the_raw_forecast"]["by_lead"]))
notes.append("The 1-day result of E2a, read with its evidence (a reading of the numbers in this file; the review of E2 asked for it). " + c1b + " " + c1c)
op_lines = []
for n_, lab_ in (("wq", "rain table + gauge"), ("wqa", "+ rain to midnight"), ("wqaF_sub", "+ forecast, substituted"), ("wqaF_clim", "shuffled-forecast control"), ("wqF_sub", "forecast without the rain to midnight"), ("wqaP", "perfect rain"), ("wqf", "perfect weather"),
                 ("wq_s", "[own split] rain table + gauge"), ("wqa_s", "[own split] + rain to midnight"), ("wqaF_s", "[own split] + forecast, retrained"), ("wqaF_clim_s", "[own split] control"), ("wqaP_s", "[own split] perfect rain"),
                 ("base_linreg_wqa", "ridge, no forecast"), ("base_linreg_wqaF", "ridge + forecast"), ("base_rf_wqa", "forest, no forecast"), ("base_rf_wqaF", "forest + forecast")):
    parts_ = []
    for k_ in F.LEADS:
        t_ = test_of(n_)["summary"][lk(k_)]["onset"]
        ch_ = verdicts[n_][lk(k_)]["chosen_on_validation_onsets"]
        parts_.append(f"{k_} d: fixed level {pc(t_['pod'])} (false {pc(t_['far'])}), level {ch_['trigger']:.2f} chosen on the validation years {pc(ch_['onset']['pod'])} (false {pc(ch_['onset']['far'])}), average precision {AP[n_][k_]['ap']:.3f}")
    op_lines.append(f"{lab_} ({n_}): " + "; ".join(parts_))
notes.append("Onsets warned at the fixed level, at the level chosen on the validation years of each set's own split and without a trigger (operating_points_by_set; the false warnings are the share of the onset warnings that were false; the average precision is that of the forecast flow on the onset rows of the "
             "test years). " + " | ".join(op_lines) + ". The fixed level hides the 1-day gain of E2a and flatters the 2- and 3-day gains of E2b (its model without the forecast lowers its trigger there); the average precision does not depend on the trigger.")
bb_lines = []
for (arm_, T_, C_), lab_ in zip(BB_PAIRS, ("wqaF_sub minus wqa", "wqaF_sub minus the shuffled control", "perfect rain minus wqa", "perfect rain minus wqaF_sub", "wqF_sub minus wq", "ridge with minus without the forecast", "forest with minus without the forecast",
                                          "wqaF_sub minus the forest + forecast", "wqaF_sub minus ridge + forecast", "[own split] wqaF_s minus wqa_s", "[own split] wqaF_s minus its control", "[own split] perfect rain minus wqaF_s")):
    bb_lines.append(f"{lab_}: " + "; ".join(f"{k_} d {BB[T_, C_, k_]['pod_points']:+.1f} points ({BB[T_, C_, k_]['pod_points_ci95'][0]:+.1f} to {BB[T_, C_, k_]['pod_points_ci95'][1]:+.1f}; NSE {BB[T_, C_, k_]['nse']:+.3f} ({BB[T_, C_, k_]['nse_ci95'][0]:+.3f} to {BB[T_, C_, k_]['nse_ci95'][1]:+.3f}); "
                                                          f"{BB[T_, C_, k_]['water_years_above_zero']} of {BB[T_, C_, k_]['n_water_years']} test years above zero)" for k_ in F.LEADS))
notes.append(f"Block-bootstrap intervals of the main differences (uncertainty.block_bootstrap; {B_BLOCK} replicates, blocks of {BLOCK_DAYS} consecutive target days drawn with replacement, seed {BOOT_SEED}; onsets warned at the fixed level in points, median NSE; 95% intervals; test years of each set's own split). "
             + " | ".join(bb_lines) + ". The paired sign test over onsets that the reading notes quote counts onsets as independent, which they are not (onsets cluster in storms and across basins): it is a description and not evidence on its own; "
             "the seed rule, the basin rule and these intervals are the measures.")
nse_up_leads = [k for k in F.LEADS if max(FORWARD["w=0.25"]["by_lead"][lk(k)]["nse_median"], FORWARD["w=0.5"]["by_lead"][lk(k)]["nse_median"]) > FORWARD["w=0"]["by_lead"][lk(k)]["nse_median"]]
fw_lines = []
for n_, v_ in FORWARD.items():
    fw_lines.append(f"{n_}: onsets warned " + "/".join(pc(v_["by_lead"][lk(k)]["onset_warned"]) for k in F.LEADS) + " (false " + "/".join(pc(v_["by_lead"][lk(k)]["onset_false"]) for k in F.LEADS) + "), median NSE " + "/".join(f"{v_['by_lead'][lk(k)]['nse_median']:.2f}" for k in F.LEADS))
notes.append("Two cheap forward-pass tests of the over-trust reading on the saved wqaF_sub networks (forward_pass_tests; INFORMATION ONLY: nothing was retrained, saved or chosen; " + NUMBER_WORD.get(len(SEEDS), str(len(SEEDS))) + " seeds averaged; test years; fixed level; leads 1/2/3). (a) Slots = (1 - w) x the perfect rain + w x the forecast "
             "(from the first validation day on): w = 0 is the perfect rain, w = 1 the forecast (both equal the scores of the sets above, asserted); (b) the forecast values below 0.5 mm and below 1 mm set to zero. " + " | ".join(fw_lines) + ". "
             "The dose-response shows how fast the networks lose onsets as the slots depart from exactness: " + f"the first quarter of the way from the perfect rain to the forecast costs {'/'.join(pc(x_) for x_ in fw_share)} of the drop in onsets warned at 1/2/3 days (a linear response would cost 25%): "
             + ("a steep response at small departures" if fw_steep else "a gradual response, not the steep drop at small departures that a reliance on exact slots would predict") + "; "
             + (f"the median NSE at w = 0.25 or 0.5 is higher than at w = 0 at the leads {', '.join(str(k) for k in nse_up_leads)}: the networks, stopped with the forecast in the slots, are not tuned to exact slots; " if nse_up_leads else "")
             + f"the zeroing shows whether the forecast's drizzle (it has fewer exact zeros than the AORC rain) matters: it changes the onsets by at most {fw_zero_max:.1f} points. Neither test separates a conditional bias from random error: synthetic slots with the measured bias but no "
             "noise (and the reverse), a retraining with a mixture or noise in the training slots, and a constant-slot control were not run.")
bf_ = BIAS_FACTS
notes.append("The bias factor and the forecast's conditional bias (bias_and_conditional_bias; from the forecast tables). One scalar per basin and lead is noisy: fitted on the first half of the validation years (2000-10 to 2003-03) and on the second (2003-04 to 2005-09) the factors differ by a median of "
             + "/".join(f"{bf_['median_over_basins_of_the_absolute_difference_of_the_two_halves_factors'][lk(k)]:.2f}" for k in F.LEADS) + " at 1/2/3 days. The seasonal bias left (mean ratio forecast / AORC, 1 day, all days, raw): "
             + ", ".join(f"{s_} {v_:.2f}" for s_, v_ in bf_["seasonal_bias_left"]["raw_mean_ratio_by_season_lead1_all_days"].items()) + "; after the correction, test years: " + ", ".join(f"{s_} {v_:.2f}" for s_, v_ in bf_["seasonal_bias_left"]["bias_corrected_mean_ratio_by_season_lead1_test_years"].items())
             + ". On days with at least 5 mm of AORC rain (test years, 12 basins pooled) the corrected forecast carries " + "/".join(f"{bf_['wet_days_test_years'][lk(k)]['bias_corrected_forecast_over_aorc_rain']:.2f}" for k in F.LEADS) + " of the AORC rain at 1/2/3 days; on the "
             f"{LEAD1['n_onsets']} 1-day onset days {LEAD1['forecast_on_the_onset_days']['mean_ratio_to_the_aorc_rain']:.2f} (median {LEAD1['forecast_on_the_onset_days']['median_ratio_on_those_days']:.2f} on those with at least 5 mm). The raw and the corrected forecast give nearly the same onsets "
             "(readings.raw_vs_bias_corrected), and the raw forecast even scores a higher median NSE at 1 day, so the correction is not what carries the result; a seasonal or quantile-mapped correction fitted on 2000-2005 is the cheap next step (not tried).")
closure_txt = []
for key_, c_ in closures.items():
    closure_txt.append(key_.replace("__", ", ").replace("_", " ") + ": " + ", ".join(f"{k} day{'s' * (k > 1)}: {pc(r_['onset_share_of_gap_closed'])} of the onset gap (NSE {pc(r_['share_of_gap_closed'])})" for k, r_ in zip(F.LEADS, c_["by_lead"])))
notes.append("Closure of the gaps (closure). " + "; ".join(closure_txt) + ". A negative share means that the set is below its base.")
ys_ = yardstick["lstm_against_simple"]
n_extra = len(FEATURES) - yardstick["recipe"]["n_features_of_wq"] - len(FC_FEATURES)
fit_txt = ", ".join(f"{n}: {yardstick['fit_seconds'][n]:g} s" for n in YARD)
if yard_check:
    rf_diff = yard_check["recipe"]["forest_wq_max_abs_difference_from_stored_base_rf_wq"]
    chk_txt = (f"Checks: this code reproduces the stored base_linreg_wq (largest difference {yard_check['recipe']['ridge_wq_max_abs_difference_from_stored_base_linreg_wq']:.1e})"
               + (f" and base_rf_wq ({rf_diff:.1e}); " if isinstance(rf_diff, float) else "; ") + "the features pass the self-check of features.py on the real tables. ")
else:
    chk_txt = "The checks of the simple-model code were not found. "
simple_txt = "; ".join(f"{n} {on_simple}, NSE {nse_simple}" for n, on_simple, nse_simple in
                       [(n, pc_lead(lambda k: yardstick["table_test"][lk(k)][n]["onset"]["pod"]), per_lead(lambda k: yardstick["table_test"][lk(k)][n]["nse_median"]))
                        for n in ("base_persistence", "base_linreg_wq", "base_rf_wq", "base_linreg_wqa", "base_rf_wqa") + YARD if n in ES])
vs_txt = " ".join(f"Against {n}, by the NSE: {ys_['lstm_wqaF_sub__' + n]['l1']['reading_in_words']} (higher in {ys_['lstm_wqaF_sub__' + n]['l1']['n_basins_lstm_higher']} of {n_b} basins, {pv(ys_['lstm_wqaF_sub__' + n]['l1']['wilcoxon_p'])}, the simple model is "
                  f"{ys_['lstm_wqaF_sub__' + n]['l1']['baseline_against_seed_spread']}). {ys_['lstm_wqaF_sub__' + n]['l1']['onsets_in_words']}" for n in YARD)
add_txt = "; ".join(f"{kind} {ng[kind]['by_lead']['l1']['nse_gain']:+.3f}, {ng[kind]['by_lead']['l1']['onset_gain_points']:+.1f} points (higher in {ng[kind]['by_lead']['l1']['n_basins_higher']} of {n_b} basins, {pv(ng[kind]['by_lead']['l1']['wilcoxon_p'])})"
                    for kind in ("lstm", "ridge regression", "random forest"))
notes.append(
    "The yardstick (rule 2 of the work order; yardstick). Ridge regression and the random forest were given the inputs of wqa plus the forecast by the recipe of 02_baselines.py: " + f"{len(FEATURES)} features, the {yardstick['recipe']['n_features_of_wq']} of the input set wq, "
    f"{n_extra} rain features computed on aorc_to_midnight as those of prcp are, and {len(FC_FEATURES)} forecast features (the forecast rain of the three target days and their sum); the ridge strength {RIDGE_ALPHA} and the forest settings "
    f"({RF_SET['n_trees']} trees, leaf size {RF_SET['min_samples_leaf']}, one third of the features per split, random_state {RF_SET['random_state']}) are those that 02_baselines.py chose on the validation years for wq and were not chosen again; one model per basin and lead on the training years only, where the "
    f"forecast slots hold the AORC rain of the target days (the same substitution as wqaF_sub). The fit took {fit_txt}. {chk_txt}Onsets warned and median NSE at 1/2/3 days ahead: {simple_txt}; lstm_wqa {on('wqa')}, NSE {ns('wqa')}; lstm_wqaF_sub {on('wqaF_sub')}, NSE {ns('wqaF_sub')}. "
    f"The LSTM against the simple models on the new input, 1 day ahead: {vs_txt} What the forecast adds to each model, 1 day ahead (median NSE, onsets warned): {add_txt}. The simple models are one fit each (the forest has one random seed), so no seed range exists for them: "
    "for them the numbers of basins and p are descriptions. The LSTM is judged by the onsets and by the NSE separately: a reading by the NSE alone does not carry over to the onsets.")
e2_ = e2_success["by_lead"]
e2_on = "; ".join(f"{k} day{'s' * (k > 1)} {pc(e2_[lk(k)]['onset_warned'])} / {pc(e2_[lk(k)]['onset_warned_wqa'])} / {pc(e2_[lk(k)]['onset_warned_control'])}" for k in F.LEADS)
e2_a = words_lead(lambda k: yes(e2_[lk(k)]["every_seed_above_wqa"]))
e2_c = words_lead(lambda k: yes(e2_[lk(k)]["every_seed_above_the_control"]))
notes.append(f"Success rule of NEXT_STEPS_FOR_AGENTS.md (E2) applied: {e2_success['rule']}. Onsets warned (wqaF_sub / wqa / control): {e2_on}. Every seed above wqa: {e2_a}; above the control: {e2_c}; rule met at 2 and 3 days ahead: {yes(e2_success['met_at_2_and_3_days'])}. The second half of the rule (above the control) is met by 30 points or more at every lead and says little, because the E2a control measures the harm of misleading slots (see the note on the control); the first half (above lstm_wqa) carries the rule.")
vq = verdicts["wqaF_sub"]
vq_txt = "; ".join(f"{lab}: " + "/".join(vq[lk(k)][key_]["onset"]["verdict"] for k in F.LEADS) for lab, key_ in (("plain level", "plain"), ("level chosen for all high-flow days", "chosen_on_validation_all_days"), ("level chosen for onsets", "chosen_on_validation_onsets")))
vq_all = "/".join(vq[lk(k)]["plain"]["all"]["verdict"] for k in F.LEADS)
vq_perf = "/".join(verdicts["wqaP"][lk(k)]["plain"]["onset"]["verdict"] for k in F.LEADS)
notes.append(f"Verdict (verdicts): the criteria stored in results/flood_analysis.json (GO: {CRIT['GO']}; CAUTION: {CRIT['CAUTION']}; NOT YET: {CRIT['NOT YET']}) are applied to the warning tables pooled over the {n_b} basins, for the onsets and for all high-flow days, at three warning levels: the plain one "
             f"(trigger 1.0; the one the sentence in `outcome` uses) and the two chosen on the validation years of the set's own split (as 06_flood_analysis.py does). lstm_wqaF_sub, onsets, 1/2/3 days ahead: {vq_txt}; all high-flow days at the plain level: {vq_all}; "
             f"lstm_wqaP (perfect rain), onsets, plain level: {vq_perf}.")
notes += [
    "Fair test: nothing is chosen on the test years. The settings are those chosen by 04_lstm_all.py on the validation years of the main split, the trainer picks each stopping point on the validation years (of the set's own split; the forecast is in the slots there), the warning levels of the second and third verdict were chosen "
    f"on the validation years, and the test years ({F.SPLIT['test'][0][:4]}-{F.SPLIT['test'][1][:4]}; {SAT_SPLIT['test'][0][:4]}-{SAT_SPLIT['test'][1][:4]} for E2b) are only scored. The bias factor was fitted on the main split's validation years. The validation scores (validation_used_for_no_choice) are shown apart and chose nothing in this script; "
    "they are not an independent measure, because the trainer used them to stop each training. The scores printed after each training in the log are test-year scores and chose nothing.",
    "The order of the choices (the rule of the work order: say so when test scores were seen before a change). The design of E2 follows from ROADMAP.md, step 2, and was fixed AFTER the test-year scores of E1 (script 15) and E1b (script 16) were known; the base set wqa and its settings are those of script 16. The sets, the substitution boundary "
    "(the first validation day, 2000-10-01), the control, the bias-correction rule (validation years only), the seeds, the stopping rule, the warning levels and the success rule were written down before any real E2 training, and were not changed after the first one. Before the real run the script was smoke-tested with FLOOD_QUICK=1 "
    "(one seed, 40 updates; results/quick): their printed test scores are those of barely trained networks and chose nothing. Every set trained is in this file. Seeds 3 and 4 were added on 30 October 2026; see choices_made_after_looking_at_test_scores."
    + (f" Sets trained in a later pass, that is AFTER the test scores of the first pass had been seen: {', '.join(later_sets)} (their design is the same as the first pass; they were added to answer a question the first pass left open, see passes)." if later_sets else "")
    + " The real run printed its test-year scores after every training and they were read while the later trainings ran; after that the script was changed only in what it computes about or says of the saved results, never in what it trains: first the wording of outcome_caveat and of the notes, the check that the three E2a sets share their training trajectory, the note on the seed average, the layout of the figure and the --figure option; then, after the independent review of E2 (results/review_forecast_input.json), the reading of the 1-day result (the plain and the validation-chosen warning level, equal false-warning budgets, the trigger-free average precision with block-bootstrap intervals, the facts behind the over-trust hypothesis, the two forward-pass tests of the saved networks, the facts about the bias factor, the summary tables at both warning levels, the wording about the controls). These descriptive analyses were added after the test scores were seen and chose nothing; no input set, setting, seed, substitution boundary, control, bias-correction rule, stopping rule, warning level or success rule was changed, and no training was repeated or dropped. Each version of the script has its sha256 in passes; the version that ran the trainings is kept in code/archive/18_forecast_input_training_version.py.",
    "Models saved (results/models/lstm_<set>_s<seed>.pt, one per training): the weights, the settings (cfg), the scalers, the basin ids, the input column names, the split, the arm and the training record. reload_check rebuilds one network per set from its file alone.",
    f"What the three E2a sets share (training_trajectories_of_e2a). wqaF_sub, wqaF_clim and wqaP read the same table in the training years, so for the same seed they follow the same training (training loss identical at every common check: {yes(trajectory['all_identical'])}); they differ "
    "only in the update the trainer keeps (chosen on the validation years with the forecast, the shuffled forecast or the perfect rain in the slots; best updates per seed in the record) and in the input they are given at run time. The control and the perfect-rain set are therefore "
    "'the same network with another input' up to the stopping update: the control measures what a network that trusts its slots does when they hold no information, and wqaP what it does when they are right.",
]
outside_ = []
for n_ in SETS:
    for k_ in F.LEADS:
        r_ = ranges[n_][lk(k_)]["onset_pod"]
        v_ = ons(EV[n_]["test"], k_)
        if r_ and (v_ > r_[1] + 1e-12 or v_ < r_[0] - 1e-12):
            outside_.append(f"{n_} at {k_} day{'s' * (k_ > 1)} ({p1(v_)} against {span(r_, p1)})")
notes.append("The seed average and the single trainings. Every score of a set is that of the day-by-day average of its trainings (flood.ensemble), and the ranges are those of the single trainings; the average of several forecasts has less noise than each of them and can score outside the range of the singles: "
             + ("this happens for the onsets warned of " + "; ".join(outside_) + "." if outside_ else "it did not happen for the onsets warned of any set.") + " The rules of section 4.2 compare single trainings with single trainings (every one above every one), not averages.")
if F.QUICK:
    notes.append("QUICK smoke test: one seed and 40 updates per training - numbers are not results. The stored sets of the ladder come from the quick prediction files where there are any and from the real ones otherwise; the check of script 16's networks reads the real files.")
this_pass["sets_trained"] = sorted({n for n, _ in trained_pairs})
if RESCORE:
    timing["rescore_runtime_s"] = round(time.time() - t_start, 1)
    this_pass["runtime_s"], this_pass["finished_local_time"] = timing["rescore_runtime_s"], time.strftime("%Y-%m-%d %H:%M")
    notes.append("This file was written by --rescore: every score, table and note and the figure were recomputed from the saved prediction sets lstm_<set>_s<seed> (and base_linreg_wqaF, base_rf_wqaF) without training again; the training records in models.<set>.per_seed, training, n_trainings, "
                 "training_seconds and runtime_s are those of the training runs; the reload check, the check of script 16's networks and the made-up-values test were run again on the saved networks. rescore_runtime_s is the time of this pass. passes lists the training passes and, last, this rescore.")
else:
    this_pass["runtime_s"] = round(time.time() - t_start - waited, 1)
    this_pass["finished_local_time"] = time.strftime("%Y-%m-%d %H:%M")
    timing["runtime_s"] = round(sum(p_["runtime_s"] for p_ in timing["passes"] if p_.get("kind") != "rescore"), 1)
    if EXTEND:
        notes.append(f"This file was written by --extend: {this_pass['n_trainings']} networks were trained now ({', '.join(sorted({n for n, _ in trained_pairs})) or 'none'}), every other set was read from its saved files, never trained again and not written again (integrity "
                     "holds the sha256 of every saved file; the files the earlier results file recorded are compared with their hashes and the script stops if one differs). Every score, table, note and the figure were recomputed from all saved sets.")
notes.append("runtime_s leaves out the time the script waited for the CPU (waited_for_cpu_s); training_seconds is the time of the trainings alone.")
# ---- the independent review in rounds, and what the two extra seeds changed (added 2026-10-30 after the five-seed reviews; texts only, no number of this file depends on it) ----
ARCHIVE_DIR = F.RES / "archive"                  # the three-seed results files, copied unchanged from the backup of 30 October 2026 (Project-Flood_backup_2026-10-30, 10:41) before seeds 3-4 were scored
CHANGE_KEYS = ("reading", "reading_in_words", "verdict", "met")              # the words of the seed rule, of the verdicts and of the success rules
CHANGE_SKIP = {"summary_test", "validation_used_for_no_choice", "per_basin", "per_river", "seed_ranges", "integrity", "passes", "notes", "what_was_not_done", "findings", "training", "models",
               "choices_made_after_looking_at_test_scores", "outcome", "outcome_caveat", "outcome_full", "independent_review", "five_seed_changes"}
COUNT_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def review_rounds(rv, script_rel, covers_default=None):
    """Every round of an independent review record, in order (round 1 = the top level of the record, round_N = a later round): when, the counts, what it covers and the script hash it read."""
    if not rv:
        return []
    later = sorted((k for k in rv if re.fullmatch(r"round_\d+", k) and isinstance(rv[k], dict)), key=lambda k: int(k.split("_")[1]))
    out = []
    for i, r in [(1, rv)] + [(int(k.split("_")[1]), rv[k]) for k in later]:
        c, fr = r.get("counts") or {}, (r.get("files_reviewed") or {}).get(script_rel)
        out.append(dict(round=i, when=str(r.get("when", "")).split(",")[0].strip(), when_full=r.get("when"), covers=r.get("covers") or (covers_default or {}).get(i) or "the three-seed numbers (seeds 0-2)",
                        verdict=r.get("verdict"), counts=c, script_sha256=r.get("record_script_sha256") or (fr.get("sha256") if isinstance(fr, dict) else None),
                        text=(f"{c.get('checks_listed')} checks listed, {c.get('checks_passed', c.get('checks_pass'))} passed, {c.get('failed')} failed; {c.get('must_fix')} must-fix and {c.get('should_fix')} should-fix items, "
                              f"{c.get('numerical_mismatches', c.get('numerical_discrepancies'))} numerical mismatches")))
    return out


def review_sentence(rows, record, n_seeds, hash_now):
    """One sentence on the rounds of the review: what each covers, which round covers the numbers of this file, and whether this version of the script is the one the last round read."""
    span_ = lambda xs: f"round {xs[0]}" if len(xs) == 1 else f"rounds {xs[0]}-{xs[-1]}" if xs == list(range(xs[0], xs[-1] + 1)) else "rounds " + ", ".join(map(str, xs))
    cap_ = lambda t: t[:1].upper() + t[1:]
    five_ = [r["round"] for r in rows if "five-seed" in r["covers"]]
    three_ = [r["round"] for r in rows if r["round"] not in five_]
    s_ = (f"The independent recomputation of the numbers by a second AI agent exists in {COUNT_WORD.get(len(rows), len(rows))} round{'s' * (len(rows) > 1)} ({record}): "
          + "; ".join(f"round {r['round']} ({r['when']}; {r['text']}) covers {r['covers']}" for r in rows) + ". ")
    if n_seeds > 3:
        if five_:
            s_ += (f"{cap_(span_(three_))} scored seeds 0-2 only; " if three_ else "") + f"the {COUNT_WORD.get(n_seeds, n_seeds)}-seed numbers of this file are covered by {span_(five_)}. "
        else:
            s_ += f"Every round scored seeds 0-2 only: the {COUNT_WORD.get(n_seeds, n_seeds)}-seed numbers of this file have not been recomputed by a second agent. "
    last_ = rows[-1]["script_sha256"]
    return s_ + (f"This version of the script has sha256 {hash_now[:12]}... (" + ("the version the last round read)." if hash_now == last_ else
                 "not the version the last round read: what it changed after that round was not recomputed by a second agent)."))


def five_seed_changes(new, name, label=lambda n: n):
    """The readings, verdict words and success rules, the lead-1 scores and the onset points of the value_of_* readings of this file that differ from the three-seed results file
    (results/archive/<name>_3seeds_2026-10-30.json), with one sentence that lists them; None in a smoke test or when that file is not there."""
    path = ARCHIVE_DIR / f"{name}_3seeds_2026-10-30.json"
    if F.QUICK or not path.exists():
        return None
    old = json.loads(path.read_text(encoding="utf-8"))
    seeds_of = lambda J: J.get("seeds") or (J.get("design") or {}).get("seeds") or []
    w3, w5 = (COUNT_WORD.get(len(seeds_of(J)), len(seeds_of(J))) for J in (old, new))
    words, rhos = [], []

    def walk(where, a, b):
        if "rho" in a and "p" in a and "rho" in b and "p" in b and all(isinstance(v, (int, float)) and v == v for v in (a["p"], b["p"], a["rho"], b["rho"])):
            if (a["p"] < 0.05) != (b["p"] < 0.05):
                rhos.append(dict(where=".".join(where), three_seeds=dict(rho=a["rho"], p=a["p"]), five_seeds=dict(rho=b["rho"], p=b["p"])))
        for k, x in a.items():
            if k not in b:
                continue
            y = b[k]
            if isinstance(x, dict) and isinstance(y, dict):
                walk(where + [k], x, y)
            elif k in CHANGE_KEYS and not (k == "reading" and "reading_in_words" in a) and x != y and all(isinstance(v, (str, bool)) and len(str(v)) <= 80 for v in (x, y)):
                words.append(dict(where=".".join(where + [k]), three_seeds=x, five_seeds=y))
    for k in new:
        if k in old and k not in CHANGE_SKIP and isinstance(new[k], dict) and isinstance(old[k], dict):
            walk([k], old[k], new[k])
    so, sn = old["summary_test"]["sets"], new["summary_test"]["sets"]
    scores = [dict(set=label(n), lead=1, nse_median=[so[n]["l1"]["nse_median"], sn[n]["l1"]["nse_median"]], onset_pod=[so[n]["l1"]["onset"]["pod"], sn[n]["l1"]["onset"]["pod"]],
                   onset_hits=[so[n]["l1"]["onset"]["hits"], sn[n]["l1"]["onset"]["hits"]]) for n in sn if n in so and so[n] != sn[n]]
    pts_ = lambda J, k, l_: J[k]["by_lead"][l_]["onsets"]["difference_points"]
    points = [dict(reading=k, onset_difference_points={l_: [pts_(old, k, l_), pts_(new, k, l_)] for l_ in ("l1", "l2", "l3")})
              for k in new if k.startswith("value_of_") and k in old and all(isinstance(J[k], dict) and "by_lead" in J[k] for J in (old, new))
              and any(abs(pts_(old, k, l_) - pts_(new, k, l_)) > 1e-9 for l_ in ("l1", "l2", "l3"))]
    drop = {"by_lead", "sets", "pairs", "lstm_against_simple", "against_the_leave_one_out_models"} | set(CHANGE_KEYS)
    place = lambda w: ", ".join({"l1": "1 day", "l2": "2 days", "l3": "3 days", "nse": "NSE", "onset": "onsets"}.get(p, p.replace("__", " against ").replace("_", " ")) for p in w.split(".") if p not in drop)
    val = lambda v, k: ("met" if v else "not met") if k == "met" else str(v)
    key_ = lambda x: x["where"].rsplit(".", 1)[1]
    reads = [x for x in words if key_(x) not in ("verdict", "met")]
    verds = [x for x in words if key_(x) in ("verdict", "met")]
    p1_ = lambda x: f"{100 * x:.1f}%"
    rel_ = path.relative_to(F.RES.parent).as_posix()
    sentence = (f"With {w5} seeds instead of {w3} (the {w3}-seed results file is kept unchanged as {rel_}) these numbers and words changed, the {w3}-seed values in brackets. "
                + ("Lead 1, test years: " + "; ".join(f"{s['set']} onsets warned {p1_(s['onset_pod'][1])} ({p1_(s['onset_pod'][0])}; {s['onset_hits'][1]} hits against {s['onset_hits'][0]}), median NSE "
                                                      f"{s['nse_median'][1]:.3f} ({s['nse_median'][0]:.3f})" for s in scores) + ". " if scores else "No lead-1 score changed. ")
                + ("Onsets, test minus control in points at 1/2/3 days: " + "; ".join(f"{place(p['reading'])} " + "/".join(f"{p['onset_difference_points'][l_][1]:+.1f}" for l_ in ("l1", "l2", "l3"))
                                                                                       + " (" + "/".join(f"{p['onset_difference_points'][l_][0]:+.1f}" for l_ in ("l1", "l2", "l3")) + ")" for p in points) + ". " if points else "")
                + ("Readings of the seed rule and of report section 4.2: " + "; ".join(f"{place(x['where'])}: {val(x['five_seeds'], '')} ({val(x['three_seeds'], '')})" for x in reads) + ". "
                   if reads else "No reading of the seed rule changed. ")
                + ("Rank correlations whose p crossed 0.05: " + "; ".join(f"{place(x['where'])}: rho {x['five_seeds']['rho']:+.2f}, p {x['five_seeds']['p']:.3f} (rho {x['three_seeds']['rho']:+.2f}, p {x['three_seeds']['p']:.3f})"
                                                                         for x in rhos) + ". " if rhos else "")
                + ("Verdict words and success rules: " + "; ".join(f"{place(x['where'])}: {val(x['five_seeds'], key_(x))} ({val(x['three_seeds'], key_(x))})" for x in verds) + "."
                   if verds else "No verdict word and no success rule changed."))
    return dict(compared_with=rel_, seeds=dict(three_seed_file=seeds_of(old), this_file=seeds_of(new)),
                how_to_read=f"what changed when seeds 3 and 4 were added: this file against the {w3}-seed results file named in compared_with, leaf by leaf over the reading words (reading, reading_in_words), the verdict words "
                            "and the success rules (verdict, met) of every block but summary_test (its verdicts repeat those of verdicts), the lead-1 test scores of every set whose scores changed, the onset points of the value_of_* "
                            "readings and the rank correlations whose p crossed 0.05; each pair is [three seeds, five seeds]. Computed when this file is written.",
                scores_changed_lead_1=scores, value_of_onset_points_changed=points, readings_changed=reads, rank_correlations_crossing_p_005=rhos, verdicts_and_success_rules_changed=verds, sentence=sentence)


if REVIEW:
    try:
        review_item = (review_sentence(review_rounds(REVIEW, "code/18_forecast_input.py"), "results/review_forecast_input.json", len(SEEDS), HASH_NOW)
                       + " The reviews were written by an AI agent, not by a team member: the team's own check (team.json) is still open.")
    except (KeyError, TypeError):
        review_item = "A file results/review_forecast_input.json exists but its counts could not be read here; read it directly. It was written by an AI agent, not by a team member: the team's own check (team.json) is still open."
else:
    review_item = ("No independent recomputation of the headline numbers by a second agent was found (results/review_forecast_input.json does not exist); the work order requires one (sections 3 and 6): it covers the clock of the forecast columns "
                   "(code/gefs_to_basins.py), the substitution tables, the saved networks and the numbers of this file. The team's own check (team.json) is open as well.")
span_of = {n_: ranges[n_]["l1"]["nse_median"][1] - ranges[n_]["l1"]["nse_median"][0] for n_ in SETS}
wide = max(span_of, key=span_of.get)
NOT_DONE = [
    review_item,
    "Ensemble members: only the control member (c00) of the GEFSv12 reforecast was fetched; the other members, their mean and their spread (a probability of rain) were not used. The ensemble mean is the natural next step if the control member's gain is clear.",
    "Forecast temperature (and any other forecast field): only rain enters; temperature, sunlight and humidity stay known up to the issue day, as in every other set.",
    "The encoder-decoder form that reads the forecast day by day (the structure of Google's 2024 model; 09_model_comparison.py is the pattern): the forecast enters here as three extra daily columns of the same LSTM.",
    "The resampling ranges of code/13_uncertainty.py (test years and basins drawn again) are not extended to the new prediction sets; the seed ranges and the checks of gain_rule are all this file holds.",
    "The documents: no report subsection, no row in the key figure (code/11_key_figure.py), no update of PLAN.md, README.md or ROADMAP.md, and the builders (build_report.py, build_slides.py, build_qa.py, build_guide.py) were not run (NEXT_STEPS_FOR_AGENTS.md, sections 3 and 6): documents must quote outcome_full, not outcome alone.",
    f"More seeds where the NSE decides: the median NSE of the single trainings of a set spans up to {span_of[wide]:.3f} (lead 1, lstm_{wide}); with {NUMBER_WORD.get(len(SEEDS), len(SEEDS))} seeds the NSE rule of report section 4.2 can hardly ever call a gain of a few hundredths clear. The onsets warned, pooled over the basins, are decided by the seeds more easily.",
    "The simple-model yardstick was run on the main split only (the substitution design); ridge regression and the random forest were not retrained on the satellite-era split with the forecast.",
    "A forecast issued later than the 00 UTC run (00:30 local, the 06 UTC run) was not tried; the Eastern-summer margin of the rule is slightly negative (see the notes).",
    "Training with a mixture of AORC and forecast rain in the slots (or noise on the AORC rain), which could narrow the gap between the substitution and the retrained model, was not tried; the settings of the LSTM (hidden size, learning rate, dropout, budget) were not tuned for the forecast sets.",
    "No seasonal bias correction (one factor per basin and lead), no spatial detail finer than the 0.25 degree boxes, no lead other than 1 to 3 days, no leave-one-basin-out test, no peaks-and-onsets loss (E5), no change of the network (E6).",
    "The real-time products that would replace AORC in the 'rain to midnight' column (Stage IV, MRMS) were not used here: the base rung inherits E1b's caveats about AORC.",
    "The tests of the over-trust reading that need more than forward passes: synthetic slots with the forecast's measured conditional bias but no random error (and the reverse), a retraining of wqaF_sub with a mixture or noise in the training slots, and a control with constant slots (the monthly climatology of the rain) were not run; "
    "until they are, the over-trust sentence stays a hypothesis.",
    "A seasonal or quantile-mapped bias correction fitted on 2000-2005 (the single factor per basin and lead is noisy and leaves a seasonal and a conditional bias) was not tried, and neither was a check of the reforecast's skill against a few months of operational GEFS before any real-time claim.",
] + (["QUICK smoke test: one seed, 40 updates per training: nothing here is a result."] if F.QUICK else [])
INPUTS = dict(
    how_to_read="the hours each input covers, as the model reads them on issue day t. The forecast is issued at the end of day t, local midnight of the river-flow day",
    clock=[dict(input="prcp (Daymet rain, the rain table), srad, tmax, tmin, vp, dayl", used_by="every set", covers="for rain, a day that ends at about 09:00 local standard time on day t (results/day_windows.json, day_ends.daymet); the window of the other Daymet variables was not measured", source="results/day_windows.json"),
           dict(input="q_mm (observed river flow)", used_by="every set (all with the river gauge)", covers="the daily mean flow of local day t, midnight to midnight by the local clock, so the day ends at the issue time", source="results/day_windows.json, river_flow_day"),
           dict(input="aorc_to_midnight", used_by="every set except wqF_sub and wq_s", covers="the AORC rain of the wall-clock day t, 00:00 to 24:00 (summer time included): the hours up to the issue time; AORC is an analysis made after the fact", source="data/aorc, summed by gefs_to_basins.aorc_local_days (the code of 16_aorc_rain_to_midnight.py)"),
           dict(input="p1, p2, p3", used_by="wqaP, wqaP_s", covers="the AORC rain of the local days t+1, t+2, t+3: rain that has not fallen at the issue time (a perfect forecast, an upper bound that cannot be had in practice)", source="data/aorc"),
           dict(input="fs1, fs2, fs3 (E2a)", used_by="wqaF_sub, wqF_sub", covers=SLOT_TEXT["fs"], source="data/gefs/basins (code/gefs_to_basins.py) and data/aorc"),
           dict(input="fk1, fk2, fk3 (E2a control)", used_by="wqaF_clim", covers=SLOT_TEXT["fk"], source="data/gefs/basins"),
           dict(input="fr1, fr2, fr3 (raw variant)", used_by="wqaF_sub_raw (with --raw); the raw forecast is also given to the saved wqaF_sub networks at prediction time", covers=SLOT_TEXT["fr"], source="data/gefs/basins"),
           dict(input="fc1, fc2, fc3 (E2b)", used_by="wqaF_s", covers=SLOT_TEXT["fc"], source="data/gefs/basins"),
           dict(input="fx1, fx2, fx3 (E2b control)", used_by="wqaF_clim_s", covers=SLOT_TEXT["fx"], source="data/gefs/basins"),
           dict(input="basin properties (flood.STATIC)", used_by="every set", covers="fixed numbers per basin, not daily data", source="CAMELS attribute tables")])
R = dict(question="Given the GEFSv12 control-member rain forecast for the local days t+1, t+2 and t+3 (basin means, bias-corrected on the validation years, from the 00 UTC run of the UTC date t+1, the latest run that exists at the issue time) as three extra daily inputs beside "
                  "the AORC rain to midnight, how many more onsets are warned 1, 2 and 3 days ahead - when the network was trained on the AORC rain of those days (an analysis made after the fact) and run on the forecast (E2a, main split), and when it is retrained with the forecast as an input "
                  "(E2b, satellite-era split) - against the same network without the forecast, against a shuffled forecast with the right climate and no information about the day, and against a perfect rain forecast?",
         quick_mode=F.QUICK, split_main=F.SPLIT, split_satellite_era=SAT_SPLIT, outcome=outcome, outcome_caveat=outcome_caveat, outcome_full=outcome_full, e2_success=e2_success, inputs=INPUTS,
         forecast_data=dict(record="results/gefs_basins.json (code/gefs_to_basins.py)", written=GBR["written"], code_sha256=GBR["code"]["sha256"], rule=GBR["rule"], skill_summary=GBR["skill"]["summary"], bias=GBR["bias"], boxes_per_basin={g: GBR["basins"][g]["n_boxes"] for g in ids},
                            realtime_margin=GBR["realtime_margin"]["by_zone_and_clock"], clock_checks=GBR["clock"]["per_time_zone"], subdaily_alignment=GBR["clock"]["subdaily_alignment"]["median_over_basins"], caveats=GBR["caveats"], tables_sha256=INPUT_FILES),
         tables=TABLE_CHECK, sets={n: dict(saved_as=f"lstm_{n}", arm=v["arm"], role=v["role"], dyn=list(v["dyn"]), forecast_columns=SLOT_TEXT.get(v["slot"]), rain_to_midnight=v["aorc"], what=v["what"]) for n, v in SETS.items()},
         stored_sets={n: dict(saved_as=f"lstm_{n}", arm="main", what={"wq": "the rain table + the river gauge (04_lstm_all.py)", "wqa": "wq + the AORC rain to midnight (16_aorc_rain_to_midnight.py)", "wqf": "wq + the perfect weather forecast of the three days (04_lstm_all.py): an upper bound that cannot be had in practice"}[n]) for n in STORED},
         derived_sets={n: dict(saved_as=f"lstm_{n}", source_networks=f"lstm_{v_[0]}_s<seed>.pt", what=v_[2] + " (a scored variant at prediction time, not a training)") for n, v_ in DERIVED.items()},
         config=conf, seeds=SEEDS, config_source="results/lstm_all.json, key config (chosen by 04_lstm_all.py on the validation years of the main split)",
         summary_test=dict(how_to_read="test years of each set's own split (arm), seed average of each set, per lead; onset and all = warning tables pooled over the basins at the FIXED level (trigger 1.0); onset_at_validation_chosen_level and all_at_validation_chosen_level = the same at the warning level chosen on the validation years of the set's own split "
                                  "(the best pooled CSI of the onsets, or of all high-flow days; the trigger is given); sets of the satellite-era split (names ending _s) are comparable only with each other", sets=summary_test),
         validation_used_for_no_choice=dict(label="USED FOR NO CHOICE. Validation years of each set's own split, shown for reference only: no input set, setting or stopping point of this script was chosen on these scores. The trainer stopped each training on its own validation score "
                                                  "(with the forecast in the slots), so these scores flatter the models.", summary=summary_val, models={n: EV[n]["val"] for n in SHOWN}),
         per_basin=per_basin, seed_ranges=dict(how_to_read="lowest and highest single-seed value of the test-year scores, per set and lead (by_seed: the single values, in the order of the seeds)", sets=ranges),
         gain_rule=gain_rule, closure=closures, ladder=ladder, verdicts=dict(criteria=CRIT, criteria_source="results/flood_analysis.json, verdict.criteria", levels=VERDICT_NOTE, sets=verdicts),
         **readings, yardstick=yardstick, causality=causality, reload_check=reload_check, aorc_column_check=aorc_column_check, training_trajectories_of_e2a=trajectory, operating_points_by_set=OPS_BY_SET, uncertainty=UNCERT, lead1_facts=LEAD1, forward_pass_tests=FORWARD_JSON, bias_and_conditional_bias=BIAS_FACTS, derived_set_files=dict(how_to_read="what this pass did with each derived set: 'written' (a plain run, or a single seed new to the file) or 'left as it was' (a --rescore or --extend found the same forecasts on disk and did not touch the file) or 'remade: ...' (a seed average whose seed list changed: an --extend that added seeds)", files=DERIVED_FILES), integrity=INTEGRITY,
         choices_made_after_looking_at_test_scores=["The design of E2 follows from ROADMAP.md, step 2, and was fixed after the test-year scores of E1 (script 15) and E1b (script 16) were known (the base set wqa is script 16's). No E2 score was seen before the sets, settings, seeds, stopping rule, warning level, substitution boundary and success rule "
                                                    "were fixed, and none of them was changed after the first real E2 training, except that seeds 3 and 4 were added later (next entry); smoke tests (40 updates) ran before and chose nothing. Every set trained is in this file.", SEED_NOTE]
                                                   + ([f"Trained in a later pass, after the test scores of the first pass were seen: {', '.join(later_sets)}."] if later_sets else [])
                                                   + ["The real run printed its test-year scores after every training and they were read while the later trainings ran; after that the script was changed only in what it computes about or says of the saved results, never in what it trains: first the wording of outcome_caveat and of the notes, the check that the three E2a sets share their training trajectory, the note on the seed average, the layout of the figure and the --figure option; then, after the independent review of E2 (results/review_forecast_input.json), the reading of the 1-day result (the plain and the validation-chosen warning level, equal false-warning budgets, the trigger-free average precision with block-bootstrap intervals, the facts behind the over-trust hypothesis, the two forward-pass tests of the saved networks, the facts about the bias factor, the summary tables at both warning levels, the wording about the controls). These descriptive analyses were added after the test scores were seen and chose nothing; no input set, setting, seed, substitution boundary, control, bias-correction rule, stopping rule, warning level or success rule was changed, and no training was repeated or dropped. Each version of the script has its sha256 in passes; the version that ran the trainings is kept in code/archive/18_forecast_input_training_version.py."],
         models=models, training=training, **timing, notes=notes, what_was_not_done=NOT_DONE)
_py = lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)


def clean(o, count=None):
    """Strict JSON: numpy values become Python ones, and a number that is not finite (NaN: a share whose denominator is zero, such as the share of false warnings of a model that gave no warning) becomes None, which json writes as null.
    count[0] is raised by one for every value replaced."""
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
    return o


FIVE = five_seed_changes(clean(R), "forecast_input", lambda n: n if n.startswith("base_") else f"lstm_{n}")        # what the two extra seeds changed, against results/archive (E2 review round 2, should-fix 2)
if FIVE:
    R["five_seed_changes"] = FIVE
    R["choices_made_after_looking_at_test_scores"] = [c + " " + FIVE["sentence"] if c == SEED_NOTE else c for c in R["choices_made_after_looking_at_test_scores"]]
nulls = [0]
clean(R, nulls)
R["notes"].append(f"{nulls[0]} values in this file are not numbers (NaN: shares whose denominator is zero, for example the share of false warnings of a model that gave no warning, or a p-value that is not defined); they are written as null (strict JSON). A reader must treat null as 'no value'.")
JSON_TEXT = json.dumps(clean(R), indent=1, allow_nan=False, default=_py)
(F.RES_OUT / f"{OUT}.json").write_text(JSON_TEXT)
R = json.loads(JSON_TEXT)                                  # the figure reads exactly what the JSON holds


draw(R)

print(f"\n{R['outcome_full']}\n")
for n in SHOWN:
    s = R["summary_test"]["sets"][n]
    print(f"  lstm_{n:16s} ({SPLIT_NAME[ARM_OF[n]]:19s}) test years: median NSE by lead " + " ".join(f"{s[lk(k)]['nse_median']:.3f}" for k in F.LEADS) + "   onsets warned " + " ".join(f"{s[lk(k)]['onset']['pod']:.3f}" for k in F.LEADS)
          + "   verdict (onsets, plain level) " + " ".join(R["verdicts"]["sets"][n][lk(k)]["plain"]["onset"]["verdict"] for k in F.LEADS))
print("\nreload check: " + "; ".join(f"{n}: max difference {x['max_abs_difference']:.2e}, bit for bit {x['bit_for_bit']}" for n, x in R["reload_check"]["per_set"].items()))
for key in ("value_of_the_forecast", "forecast_vs_climatology_control", "value_of_a_misleading_input"):
    print(f"{key}: " + "; ".join(f"{k} days: onsets {100 * R[key]['by_lead'][lk(k)]['onsets']['warned_with']:.1f}% against {100 * R[key]['by_lead'][lk(k)]['onsets']['warned_without']:.1f}% ({R[key]['by_lead'][lk(k)]['onsets']['difference_points']:+.1f} points, every seed above: "
                                         f"{R[key]['by_lead'][lk(k)]['onsets']['every_seed_above_every_seed']}), NSE {R[key]['by_lead'][lk(k)]['nse']['median_with']:.3f} against {R[key]['by_lead'][lk(k)]['nse']['median_without']:.3f} ({R[key]['by_lead'][lk(k)]['nse']['gain']:+.3f}, {R[key]['by_lead'][lk(k)]['nse']['reading']})" for k in F.LEADS))
print(f"{R['n_trainings']} trainings in {R['training_seconds']:.0f} s; runtime {R.get('runtime_s', R.get('rescore_runtime_s')):.0f} s")
