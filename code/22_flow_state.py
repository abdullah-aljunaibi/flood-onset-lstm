"""E4e: the river at the issue time - hourly USGS gauge flow as inputs of the all-river LSTM (NEXT_EXPERIMENTS_2026-10-28.md, section E4e; satellite-era split, variant b).

Every LSTM of this project reads the river as one daily mean of the issue day t; the state of the river at the issue time (local midnight) is the one input it has never seen.
From the hourly reduction of the 15-minute USGS discharge (code/fetch_usgs_iv.py: hour-ENDING UTC stamps, q_mean_cfs, q_last_cfs, approved) this script builds, per basin and issue
day t (wall clock of the gauge, summer time included; cfs to mm/day with the basin area code/camels.py uses; Q95 = flood.thresholds of the TRAINING years of the split):
   q_end = log1p(q_last of the hour ending 24:00 / Q95)   dq6 = log1p(q_end / Q95) - log1p(q at 18:00 / Q95), dq12 likewise from 12:00   q_dmax = log1p(highest hourly mean of day t / Q95)
   hours_above = hours of day t with the hourly mean flow above Q95 (0-24; 23 or 25 where the clocks change)
 Delivery: the last available hour ends at 24:00 (the issue time) or at 22:00 (columns *_22, the record stops two hours before the issue time); both variants are trained and reported.
 Missing hours (ice, equipment: 0.6-3.9% of the hours after 2000 in the first gauges checked, outages up to weeks) are never filled with zero: a missing hour carries the last known value
 for at most CARRY_H hours and is counted (hourly_missing_hours); a day whose hours up to the last available hour are not all known after that falls back to the daily mean (q_end =
 log1p(q_mm / Q95), dq6 = dq12 = 0, q_dmax = q_end, hours_above = 24 x (q_mm > Q95): the work order's variant (a); hourly_known = 0); so does a day whose hourly mean over the available hours
 differs from the CAMELS daily mean of the same day by more than 50% (q_mm > 0.5 mm/day; hourly_mismatch = 1: a corrupt or incomplete 15-minute record, e.g. South Toe on 17 Sep 2004:
 one known hour of 53 mm/day against a daily mean of 175 mm/day; that day falls back by the gap rule, and the mismatch rule is a second guard that reads nothing after the last available
 hour); all three are quality columns, not inputs; counts per basin. Delivery assumptions (review of 2026-10-29, must-fix 1): the 24:00 variant assumes zero delivery delay and approved
 values (it reads the hour that ends at the issue time, and every hourly value used is from the approved record), the 22:00 variant a delay of at most two hours; a real-time service gets
 provisional values after the gauge's telemetry delay, and neither variant covers the provisional-to-approved revision.
 Sets (settings of results/lstm_all.json; gauge and basin properties as in wqa_s; seeds 0, 1, 2, --seeds 5 adds 3 and 4) on the satellite-era split of script 18 (train 2000-10-01..
 2008-09-30, validation 2008-10-01..2010-09-30, test 2010-10-01..2014-09-30; tables from 2000-01-01): wqaH_s = FORCING + aorc_to_midnight + the five columns; wqaWH_s = + the six
 within-day AORC columns of code/subdaily_inputs.py (aorc_maxcell1h filled with aorc_max1h where NaN); wqaWHF_s = + fc1..fc3, the bias-corrected GEFS forecast of script 18's wqaF_s;
 wqaH22_s, wqaWH22_s, wqaWHF22_s = the same with the 22:00 columns. Controls: the STORED sets lstm_wqa_s (base), lstm_wqaF_s (base of the forecast sets) and lstm_wqaP_s (perfect
 rain in the slots, the ceiling of this split), rescored here by this script's own code, never copied. Scores (test years of this split, scored once): pooled onset POD/FAR/CSI and
 median NSE at trigger 1.0 per lead, the seed rule, per-basin wins, paired onsets, the CSI trigger and the 30%-false operating point of code/warning_policy.py (chosen on the validation
 years), the closure of the gap from wqa_s to wqaP_s, the 24:00 and 22:00 variants side by side. Success rule, fixed before training: wqaH_s above wqa_s in every seed at lead 1.
 Causality: a made-up-hours test on the hourly-flow columns (every hourly value after the issue instant replaced by random numbers: no column of day t moves, the next day moves; for
 22:00 the 24:00 columns of day t move), subdaily_inputs.checks() (record stored), flood._assert_causal inside every flood.fit, script 18's made-up-values test on one saved network per set.
Reads data/hourly_flow/usgs_iv_<gauge>.npz (fetch_usgs_iv.py --assemble; a missing file is assembled in memory from data/hourly_flow/raw/<gauge>/*.csv.gz with the same logic, nothing
written), data/aorc (via subdaily_inputs), data/gefs/basins, results/lstm_all.json, results/flood_analysis.json, results/forecast_input.json, results/preds/lstm_{wqa_s,wqaF_s,wqaP_s}*.npz,
and results/review_flow_state.json when it exists (the independent review: its verdict, date and must-fix count go into status, provenance and the figure's header and footer; its wording
findings of 2026-10-29 are applied in this file; it covers a pass only when nothing was trained after the pass it reviewed, else the status says a new review is needed).
Writes results/flow_state.json, figures/flow_state.png, results/models/lstm_<set>_s<seed>.pt, the prediction sets lstm_<set>_s<seed> and lstm_<set> (seed mean), and rewrites
results/diagnostics_2026-10-28/subdaily_inputs_check.json (the record of subdaily_inputs.checks(), a preliminary diagnostic that stays one).
Run (project root, PYTHONUTF8=1): python code/22_flow_state.py [--rescore | --extend] [--seeds 3|5] [--figure]; FLOOD_QUICK=1: one seed, 40 updates, results/quick and figures/quick;
--gauges-available (smoke test only): the gauges whose hourly files are on disk while the download runs. torch is imported only through flood (and inside the save/reload helpers).
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
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flood as F                                           # the shared engine (it imports torch)
import gefs_to_basins as GB
import subdaily_inputs as SI
import warning_policy as WP
from common import AQUA, BLUE, INK, INK2, MUTED, ORANGE, ROOT, YELLOW, use_style

use_style()
SCRIPT, OUT, t_start = "22_flow_state.py", "flow_state", time.time()
HASH_NOW = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
ap = argparse.ArgumentParser(description="E4e: the river at the issue time (hourly USGS flow) as LSTM inputs, satellite-era split")
ap.add_argument("--rescore", action="store_true", help="no training: scores, tables, outcome and figure again from the saved sets")
ap.add_argument("--extend", action="store_true", help="as --rescore, but trains the (set, seed) pairs whose files are missing")
ap.add_argument("--figure", action="store_true", help="only the figure again, from the results file")
ap.add_argument("--seeds", type=int, default=3, choices=(3, 5), help="3 (seeds 0, 1, 2) or 5 (adds 3 and 4)")
ap.add_argument("--gauges-available", action="store_true", help="smoke test only: use the gauges whose hourly files (npz or all 348 monthly files) are on disk")
ARGS = ap.parse_args()
RESCORE, EXTEND = ARGS.rescore, ARGS.extend
assert not (RESCORE and EXTEND), "--rescore trains nothing and --extend trains what is missing: use one of the two"
assert not ARGS.gauges_available or F.QUICK, "--gauges-available is a smoke-test option (FLOOD_QUICK=1): a real run needs all 12 gauges"
LOAD, SEEDS = RESCORE or EXTEND, [0] if F.QUICK else list(range(ARGS.seeds))
SAT_SPLIT = WP.SAT_SPLIT                                    # the satellite-era split of 18_forecast_input.py (checked below against the stored sets' meta)
T_ISS = pd.date_range("2000-01-01", "2014-09-30")          # the issue days of the satellite-era tables (script 18)
HOURLY, MONTHS = F.DATA / "hourly_flow", pd.period_range("1985-10", "2014-09", freq="M")
CARRY_H = 6                                                 # a missing hour carries the last known value for at most this many hours
MISMATCH_REL = 0.5      # a day whose hourly mean over the available hours differs from the CAMELS daily mean by more than this share (q_mm > 0.5 mm/day) uses the daily-mean fallback: a corrupt or incomplete 15-minute record (South Toe, 17 Sep 2004: one known hour of 53 mm/day against a daily mean of 175 mm/day; that day falls back by the gap rule, this rule is a second guard), seen 2026-10-29
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")
MODELS, RESULTS_FILE, FIG_FILE = F.RES_OUT / "models", F.RES_OUT / f"{OUT}.json", F.FIG_OUT / f"{OUT}.png"
REVIEW_FILE, REAL_PRED = F.RES / f"review_{OUT}.json", F.RES / "preds"
H_COLS = ("q_end", "dq6", "dq12", "q_dmax", "hours_above")
H22_COLS = tuple(c + "_22" for c in H_COLS)
W_COLS = ("aorc_last3h", "aorc_last6h", "aorc_last12h", "aorc_max1h", "aorc_maxcell1h", "aorc_wet_hours")
FC_COLS, BASE_DYN, P_CUT = ("fc1", "fc2", "fc3"), tuple(F.FORCING) + ("aorc_to_midnight",), 0.05
AORC_WORDING, UPPER_WORDING = "an analysis made after the fact", "an upper bound that cannot be had in practice"
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
SUCCESS_RULE = "wqaH_s above wqa_s in every seed at lead 1 on this split: onsets warned 1 day ahead, pooled over the basins at trigger 1.0, every single seed of wqaH_s above every single seed of the stored lstm_wqa_s"
SPLIT_TEXT = "the satellite-era split (train 2000-10-01..2008-09-30, validation 2008-10-01..2010-09-30, test 2010-10-01..2014-09-30; flood thresholds from its training years; its numbers are comparable only inside it)"
NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
lk, rel, sha = (lambda k: f"l{k}"), (lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)), (lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None)
pc, p1, n2 = (lambda x: f"{round(100 * x)}%" if x == x else "no value"), (lambda x: f"{100 * x:.1f}%" if x == x else "no value"), (lambda x: f"{x:.2f}")
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]     # every single seed of a above every single seed of b
div = lambda a, b: a / b if b else float("nan")
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
notes = []


# ---- figure ---------------------------------------------------------------------------------------------------------------------------------------------
def draw(R):
    """Lead 1, test years of the split: onsets warned (top; every bar label also gives the false share of its onset warnings at trigger 1.0, and a white diamond marks the onsets warned at the 30%-false
    operating point) and median NSE (bottom) per set; bars = seed mean, dots = single seeds; the stored sets hatched. Every number on the figure comes from the results file (review of 2026-10-29, must-fix 2)."""
    import textwrap
    FS, S, RG, OP = 8.5, R["summary_test"]["sets"], R["seed_ranges"]["sets"], R["operating_points"]["sets"]
    crit = R["operating_points"]["criteria"]["GO"]                                        # "pod >= 0.70 and far <= 0.30" of results/flood_analysis.json
    pod_go, far_go = (float(crit.split(f"{w} ")[1].split()[0]) for w in ("pod >=", "far <="))
    order = [n for n in ("wqa_s", "wqaH_s", "wqaH22_s", "wqaWH_s", "wqaWH22_s", "wqaF_s", "wqaWHF_s", "wqaWHF22_s", "wqaP_s") if n in S]
    LABEL = {"wqa_s": "rain to midnight\nalone (wqa_s,\nstored)", "wqaH_s": "+ river at\n24:00 (wqaH_s)", "wqaH22_s": "+ river at\n22:00 (wqaH22_s)", "wqaWH_s": "+ river 24:00\n+ within-day rain\n(wqaWH_s)",
             "wqaWH22_s": "+ river 22:00\n+ within-day rain\n(wqaWH22_s)", "wqaF_s": "GEFS forecast\n(wqaF_s, stored)", "wqaWHF_s": "forecast + river\n24:00 + within-day\n(wqaWHF_s)",
             "wqaWHF22_s": "forecast + river\n22:00 + within-day\n(wqaWHF22_s)", "wqaP_s": "perfect rain in\nthe slots (wqaP_s,\nstored)"}
    COL = {"wqa_s": AQUA, "wqaF_s": ORANGE, "wqaWHF_s": ORANGE, "wqaWHF22_s": ORANGE, "wqaP_s": YELLOW}
    wrap = lambda s, w: "\n".join(textwrap.fill(p, w) for p in s.split("\n"))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 9.5), sharex=True, gridspec_kw=dict(hspace=0.3, top=0.775, bottom=0.215, left=0.08, right=0.98))
    for ax, key in ((ax1, "onset_pod"), (ax2, "nse_median")):
        for i, n in enumerate(order):
            y = 100 * S[n]["l1"]["onset"]["pod"] if key == "onset_pod" else S[n]["l1"]["nse_median"]
            seeds, stored = [100 * s if key == "onset_pod" else s for s in RG[n]["l1"]["by_seed"][key]], n in R["design"]["stored_sets"]
            ax.bar(i, y, width=0.58, color=COL.get(n, BLUE), alpha=0.45 if stored else 0.8, hatch="//" if stored else None, edgecolor=COL.get(n, BLUE), linewidth=1.2, zorder=2)
            ax.scatter(np.full(len(seeds), i) + np.linspace(-0.12, 0.12, len(seeds)), seeds, s=22, color=INK, zorder=4, linewidths=0)
            label = f"{y:.0f}%\n{100 * S[n]['l1']['onset']['far']:.0f}% false" if key == "onset_pod" else f"{y:.2f}"      # the false share beside every onset share (must-fix 2)
            ax.text(i, max([y] + seeds) + (1.5 if key == "onset_pod" else 0.015), label, ha="center", va="bottom", fontsize=FS - 0.5 if key == "onset_pod" else FS, color=INK2, zorder=5, linespacing=1.1,
                    bbox=dict(facecolor=ax.get_facecolor(), edgecolor="none", pad=1.0) if key == "onset_pod" else None)       # the GO line passes behind a label, not through it
            if key == "onset_pod":                                                        # the 30%-false operating point of the set (trigger chosen on the validation years), test years
                ax.scatter([i], [100 * OP[n]["l1"]["rules"]["far_cap"]["test"]["onset"]["pod"]], s=34, marker="D", facecolor="white", edgecolor=INK, linewidths=1.0, zorder=4)
        ax.set_xticks(np.arange(len(order))), ax.tick_params(axis="x", length=0), ax.tick_params(axis="y", labelsize=FS)
    go_ok = [n for n in order if S[n]["l1"]["onset"]["far"] <= far_go]                    # the sets whose false share at trigger 1.0 meets the GO cap (none: the clause says so)
    go_text = (f"GO asks {100 * pod_go:.0f}% warned AND at most {100 * far_go:.0f}% false:\n"
               + (f"no bar here keeps the false share under {100 * far_go:.0f}% at trigger 1.0" if not go_ok else f"only {', '.join(go_ok)} keep{'s' if len(go_ok) == 1 else ''} the false share at or under {100 * far_go:.0f}% at trigger 1.0"))
    ax1.axhline(100 * pod_go, color=MUTED, linewidth=1.0, zorder=1)
    ax1.text(-0.4, 100 * pod_go + 1.5, go_text, fontsize=FS - 1, color=MUTED, ha="left", va="bottom", linespacing=1.2)
    ax1.set_ylim(0, 100), ax2.set_ylim(0, 1), ax1.set_ylabel("onsets warned 1 day ahead (%)", fontsize=FS + 0.5), ax2.set_ylabel("median NSE over the basins", fontsize=FS + 0.5)
    n_on, n_new = S[order[0]]["l1"]["onset"]["hits"] + S[order[0]]["l1"]["onset"]["misses"], len(R["seeds"])
    ax1.set_title(f"Onsets warned 1 day ahead (satellite-era split, test years 2010-2014, {n_on} onsets, trigger 1.0, {R['n_basins']} basins pooled)", fontsize=FS + 1.5)
    ax2.set_title("Median NSE at 1 day (test years 2010-2014)", fontsize=FS + 1.5), ax2.set_xticklabels([LABEL[n] for n in order], fontsize=FS - 0.5)
    appr_min = min(v["approved_share_after_2000"] for v in R["data_coverage"]["per_basin"].values())
    approved_text = "this study used the approved record" if appr_min >= 1 else f"this study used the record as downloaded, approved share {100 * appr_min:.1f}% or more in every gauge"
    review_line = (R.get("provenance") or {}).get("review_line", "this file's independent review is not yet done")
    fig.text(0.08, 0.955, "E4e: the river at the issue time (hourly USGS gauge flow) as an input of the LSTM with the river gauge, satellite-era split", fontsize=12, weight="bold", color=INK, ha="left")
    fig.text(0.08, 0.93, wrap(f"Bars: the seed-averaged forecast; dots: every single training ({NUMBER_WORD.get(n_new, n_new)} seed{'s' if n_new != 1 else ''} per new set, {' or '.join(NUMBER_WORD.get(c, str(c)) for c in sorted({len(v['seeds']) for v in R['models'].values() if v.get('stored')}))} per stored set); hatched: the stored sets of script 18, "
                              f"rescored here. Top panel: each label gives the onsets warned and, under it, the share of the onset warnings that were false (trigger 1.0); the white diamond is the onsets warned at the "
                              f"{100 * far_go:.0f}%-false operating point (trigger chosen on the validation years; its test false share is in the file). "
                              f"'River at 24:00 / 22:00': the last available hour of the gauge record ends at the issue time / two hours before it.\n{R['status']}", 165), fontsize=FS, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.08, 0.02, wrap(f"Every number is of the satellite-era split (train 2000-2008, test 2010-2014) and is comparable only inside it (its base, wqa_s, is weaker than the main-split wqa). "
                              f"AORC rain is {AORC_WORDING} ({HINDSIGHT_WORDING}, a bound measured for the daily total only; unmeasured for the within-day columns of wqaWH_s and wqaWHF_s); the hourly flow is the gauge's own "
                              f"record, an input a real-time service has as provisional values and after a delivery delay (the 24:00 variant assumes none and the 22:00 variant two hours; {approved_text}). "
                              f"'Perfect rain in the slots' (wqaP_s): the AORC rain of the target days given as the forecast, {UPPER_WORDING}. "
                              f"Onsets: target days above Q95 whose issue-day flow was at or below it ({n_on} in the test years).\n"
                              f"The team's own checks are not yet done; {review_line}. Written by code/{SCRIPT} from results/{OUT}.json.", 185), fontsize=FS - 1.3, color=INK2, ha="left", va="bottom", linespacing=1.5)
    fig.savefig(FIG_FILE, dpi=200)
    plt.close(fig)


if ARGS.figure:
    assert RESULTS_FILE.exists(), "--figure needs the results file of a run"
    draw(json.loads(RESULTS_FILE.read_text(encoding="utf-8")))
    sys.exit(print(f"figure written: {FIG_FILE}"))

# ---- 0. data: the hourly flow, the daily tables of the satellite-era split -------------------------------------------------------------------------------
D, B, ZONES = F.load_data(), F.basins(), SI.zones()
have_hourly = lambda g: (HOURLY / f"usgs_iv_{g}.npz").exists() or len(list((HOURLY / "raw" / g).glob("*.csv.gz"))) == len(MONTHS)
ids = [g for g in D if have_hourly(g)] if ARGS.gauges_available else list(D)
assert len(ids) > 1 and all(have_hourly(g) for g in ids), "hourly flow is missing for " + ", ".join(g for g in ids if not have_hourly(g)) + ": run code/fetch_usgs_iv.py (then --assemble)"
D, B, n_b = {g: D[g] for g in ids}, B.loc[ids], len(ids)
thr = F.thresholds(D, split=SAT_SPLIT)                       # Q95 of the TRAINING years of this split (mm/day), as script 18
CONF_REAL, FA = json.loads((F.RES / "lstm_all.json").read_text())["config"], json.loads((F.RES / "flood_analysis.json").read_text())
AREA = {g: float(D[g].attrs.get("area_km2") or np.nan) for g in ids}          # the area code/camels.py converted the daily flow with (the CAMELS forcing-file header)
assert all(np.isfinite(AREA[g]) for g in ids), "the daily tables of flood.load_data() carry no area_km2 (code/camels.py sets it): rebuild data/derived/camels12_daymet.pkl"
MM = lambda g: 0.0283168 * 86400 / (AREA[g] * 1e6) * 1000  # cfs -> mm/day over the basin (code/camels.py)


def hourly_series(g):
    """(hour-ending UTC stamps, q_mean_cfs, q_last_cfs, approved share, route): the npz of fetch_usgs_iv.py, else the same reduction done in memory from the monthly files (nothing written)."""
    f = HOURLY / f"usgs_iv_{g}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return pd.DatetimeIndex(z["stamp_utc_s"].astype("datetime64[s]")), z["q_mean_cfs"].astype(float), z["q_last_cfs"].astype(float), z["approved"].astype(float), rel(f)
    frames = []
    for m in sorted((HOURLY / "raw" / g).glob("*.csv.gz")):
        try:
            frames.append(pd.read_csv(m, usecols=["time", "value", "approval_status"], dtype={"value": str, "approval_status": str}))
        except pd.errors.EmptyDataError:                      # a month the API answered with an empty body (before the gauge's record, or no values)
            continue
    d = pd.concat(frames, ignore_index=True)
    t, v = pd.to_datetime(d["time"], utc=True).dt.tz_convert(None), pd.to_numeric(d["value"], errors="coerce")
    ok = v.notna().values
    s = pd.DataFrame(dict(v=v.values[ok], appr=(d["approval_status"].values[ok] == "Approved")), index=pd.DatetimeIndex(t.values[ok])).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    grp = s.groupby((s.index - pd.Timedelta(microseconds=1)).ceil("h"))      # the value at 00:15 belongs to the hour ending 01:00; the one at exactly 01:00 too
    H = pd.DataFrame(dict(q_mean=grp["v"].mean(), q_last=grp["v"].last(), approved=grp["appr"].mean()))
    H = H.reindex(pd.date_range(H.index.min(), H.index.max(), freq="h"))
    return H.index, H.q_mean.values, H.q_last.values, H.approved.values, f"assembled in memory from data/hourly_flow/raw/{g}/*.csv.gz (the logic of fetch_usgs_iv.assemble; the npz was not on disk)"


def flow_state_table(stamp, q_mean, q_last, g, q_day, last_hour):
    """The five columns of day t from the hourly flow with the last available hour ending `last_hour` (24: the issue time; 22: two hours before it), plus the quality columns
    hourly_known (1 = every hour of day t up to last_hour known after a carry of at most CARRY_H hours) and hourly_missing_hours (hours carried or still missing among them)."""
    end = SI.wall_clock(stamp, ZONES[g]) - pd.Timedelta(minutes=1)      # an hour that ends at exactly 24:00 belongs to the day that ends then
    day, h_end, known = end.normalize(), end.hour.values + 1, np.isfinite(q_mean)
    qm, ql = pd.Series(q_mean * MM(g)).ffill(limit=CARRY_H).values, pd.Series(q_last * MM(g)).ffill(limit=CARRY_H).values
    T = pd.DataFrame(dict(qm=qm, ql=ql, carried=~known & np.isfinite(qm), gap=~np.isfinite(qm), h=h_end), index=day)[h_end <= last_hour]
    g_, rx = T.groupby(level=0), lambda s: s.reindex(q_day.index)
    at = lambda h: rx(T.ql.where(T.h == h).groupby(level=0).max())    # the flow at the end of the hour ending h (h >= 4 occurs once a day, also when the clocks change)
    q_end, q6, q12, q_dmax, n_gap, carried = at(last_hour), at(last_hour - 6), at(last_hour - 12), rx(g_.qm.max()), rx(g_.gap.sum()), rx(g_.carried.sum())
    dmean = rx(g_.qm.mean())                                           # the hourly mean over the hours up to last_hour (mm/day), against the CAMELS daily mean of the same day
    mismatch = (q_day > 0.5) & ((dmean - q_day).abs() > MISMATCH_REL * q_day)
    ok, lg = (n_gap == 0) & q_end.notna() & q6.notna() & q12.notna() & ~mismatch, lambda q: np.log1p(q / thr[g])
    out = pd.DataFrame(dict(q_end=lg(q_end), dq6=lg(q_end) - lg(q6), dq12=lg(q_end) - lg(q12), q_dmax=lg(q_dmax), hours_above=rx((T.qm > thr[g]).groupby(level=0).sum())), index=q_day.index)
    fb = pd.DataFrame(dict(q_end=lg(q_day), dq6=0.0, dq12=0.0, q_dmax=lg(q_day), hours_above=float(last_hour) * (q_day > thr[g])), index=q_day.index)
    out.loc[~ok] = fb.loc[~ok].values
    out["hourly_known"], out["hourly_missing_hours"], out["hourly_mismatch"] = ok.astype(int), (carried.fillna(0) + n_gap.fillna(last_hour)).astype(int), mismatch.astype(int)
    return out


HOURLY_DATA, COVERAGE, TAB, FILL, t0 = {}, {}, {}, {}, time.time()
for g in ids:
    stamp, qm_cfs, ql_cfs, appr, route = HOURLY_DATA[g] = hourly_series(g)
    assert stamp.is_monotonic_increasing and ((stamp[1:] - stamp[:-1]) == pd.Timedelta(hours=1)).all() and stamp[0] <= pd.Timestamp("2000-01-01"), f"{g}: the hourly axis has gaps or starts after 2000"
    d = D[g].loc[T_ISS[0]:].copy()
    assert d.index.equals(T_ISS), f"{g}: the daily table does not cover 2000-01-01 .. 2014-09-30"
    W = SI.within_day_table(g, ZONES[g], T_ISS).drop(columns=["hours"])
    nan = W.aorc_maxcell1h.isna()
    assert (W.aorc_filled_hours[nan] >= 23).all() and not W.drop(columns=["aorc_maxcell1h"]).isna().any().any(), f"{g}: a within-day column has gaps on a day not wholly filled from ERA5"
    W["aorc_maxcell1h"], FILL[g] = W.aorc_maxcell1h.where(~nan, W.aorc_max1h), [str(x.date()) for x in W.index[nan]]
    c = GB.load_columns(g)
    assert c.index.equals(T_ISS) and np.isfinite(c.drop(columns="init_utc_date").values).all(), f"{g}: the forecast table of code/gefs_to_basins.py does not cover 2000-01-01 .. 2014-09-30 without gaps"
    p1_ = W.aorc_to_midnight.shift(-1)
    assert np.allclose(c.aorc_1.values[p1_.notna().values], p1_.values[p1_.notna().values], rtol=0, atol=2e-6), f"{g}: aorc_to_midnight of subdaily_inputs is not the AORC column of script 18's tables"
    H24, H22 = flow_state_table(stamp, qm_cfs, ql_cfs, g, d.q_mm, 24), flow_state_table(stamp, qm_cfs, ql_cfs, g, d.q_mm, 22).add_suffix("_22")
    TAB[g] = pd.concat([d, W, pd.DataFrame({f"fc{k}": c[f"fc_bc_{k}"].values for k in F.LEADS}, index=T_ISS), H24, H22], axis=1)
    assert np.isfinite(TAB[g][list(BASE_DYN + W_COLS + FC_COLS + H_COLS + H22_COLS)].values).all(), f"{g}: an input column has gaps"
    assert ((TAB[g].hours_above <= 25) & (TAB[g].hours_above_22 <= 23)).all() and ((TAB[g].hourly_known_22 >= TAB[g].hourly_known) | (TAB[g].hourly_mismatch_22 == 1)).all(), f"{g}: the hourly-flow columns are out of range"      # a day known at 24:00 may still be a mismatch day at 22:00 (a rise in the last two hours)
    after = stamp >= pd.Timestamp("2000-01-01")                                 # coverage: the local-day mean of the hourly flow against the CAMELS daily mean (same gauge, same area), the gaps, the approval
    dm = pd.Series(qm_cfs[after] * MM(g), index=(SI.wall_clock(stamp[after], ZONES[g]) - pd.Timedelta(minutes=1)).normalize()).groupby(level=0).mean().reindex(T_ISS)
    both = dm.notna() & (d.q_mm > 0)
    corr_all, ratio_all = float(np.corrcoef(dm[both], d.q_mm[both])[0, 1]), float(dm[both].sum() / d.q_mm[both].sum())
    used = both & (TAB[g].hourly_known == 1)                                   # the days whose hourly record the inputs use (mismatch and gap days fall back to the daily mean)
    corr_, ratio_ = float(np.corrcoef(dm[used], d.q_mm[used])[0, 1]), float(dm[used].sum() / d.q_mm[used].sum())
    assert corr_ > 0.98 and 0.9 < ratio_ < 1.1, f"{g}: the hourly flow does not reproduce the CAMELS daily mean on the days it is used (corr {corr_:.4f}, ratio {ratio_:.4f}): units, clock or gauge"
    gaps = pd.Series(~np.isfinite(qm_cfs[after]))
    runs_ = gaps.groupby((gaps != gaps.shift()).cumsum()).sum()[gaps.groupby((gaps != gaps.shift()).cumsum()).first()] if gaps.any() else pd.Series(dtype=int)
    te = TAB[g].loc[SAT_SPLIT["test"][0]:SAT_SPLIT["test"][1]]
    COVERAGE[g] = dict(source=route, zone=ZONES[g], area_km2_used=AREA[g], area_gages2_km2=float(B.area_gages2[g]), q95_mm_per_day=thr[g], q95_cfs=thr[g] / MM(g), first_hour_end_utc=str(stamp[0]), last_hour_end_utc=str(stamp[-1]),
                       hours_after_2000=int(after.sum()), missing_hours_after_2000=int(gaps.sum()), missing_share_after_2000=float(gaps.mean()), n_gaps_after_2000=int(len(runs_)), longest_gap_h=int(runs_.max()) if len(runs_) else 0,
                       gaps_longer_than_a_day=int((runs_ > 24).sum()), approved_share_after_2000=float(np.nanmean(appr[after])), local_day_mean_vs_camels=dict(corr_days_used=corr_, ratio_of_sums_days_used=ratio_, days_used=int(used.sum()), corr_all_days=corr_all, ratio_of_sums_all_days=ratio_all, days_compared=int(both.sum())),
                       mismatch_days_24=int(TAB[g].hourly_mismatch.sum()), mismatch_days_22=int(TAB[g].hourly_mismatch_22.sum()), mismatch_rule=f"hourly mean over the available hours differs from the CAMELS daily mean by more than {MISMATCH_REL:.0%} (q_mm > 0.5 mm/day): daily-mean fallback", mismatch_dates_24=[str(x.date()) for x in TAB[g].index[TAB[g].hourly_mismatch == 1]][:40],
                       hours_carried_24=int(TAB[g].hourly_missing_hours[TAB[g].hourly_known == 1].sum()), fallback_days_24=int((TAB[g].hourly_known == 0).sum()), fallback_days_24_test_years=int((te.hourly_known == 0).sum()),
                       fallback_days_22=int((TAB[g].hourly_known_22 == 0).sum()), fallback_days_22_test_years=int((te.hourly_known_22 == 0).sum()), days=int(len(TAB[g])), test_days=int(len(te)),
                       share_of_days_with_an_hour_above_q95=float((TAB[g].hours_above > 0).mean()), maxcell_filled_days=FILL[g])
    print(f"{g}: hourly {COVERAGE[g]['first_hour_end_utc'][:10]}.., {gaps.sum()} missing hours after 2000 ({100 * gaps.mean():.2f}%, longest {COVERAGE[g]['longest_gap_h']} h), {COVERAGE[g]['fallback_days_24']} fallback days ({COVERAGE[g]['mismatch_days_24']} of them daily-mean mismatches) "
          f"({COVERAGE[g]['fallback_days_24_test_years']} in the test years), corr with CAMELS daily {corr_:.4f}, ratio {ratio_:.4f}  [{time.time() - t0:.0f} s]", flush=True)
N_FB = {h: sum(v[f"fallback_days_{h}"] for v in COVERAGE.values()) for h in (24, 22)}
N_FILL = sum(len(v) for v in FILL.values())

# ---- 1. the sets ------------------------------------------------------------------------------------------------------------------------------------------
SETS = {}
for tag, hc, hour in (("", H_COLS, 24), ("22", H22_COLS, 22)):
    where = f"the last available hour ending {hour}:00 local"
    SETS[f"wqaH{tag}_s"] = dict(cols=hc, hour=hour, base="wqa_s", what=f"wqa_s + the five hourly-flow columns of the issue day ({where})")
    SETS[f"wqaWH{tag}_s"] = dict(cols=hc + W_COLS, hour=hour, base="wqa_s", what=f"wqaH{tag}_s + the six within-day AORC columns of code/subdaily_inputs.py ({where})")
    SETS[f"wqaWHF{tag}_s"] = dict(cols=hc + W_COLS + FC_COLS, hour=hour, base="wqaF_s", what=f"wqaWH{tag}_s + fc1..fc3, the bias-corrected GEFS control-member forecast of the days t+1..t+3 (script 18's columns; {where})")
SETS = {n: SETS[n] for n in ("wqaH_s", "wqaH22_s", "wqaWH_s", "wqaWH22_s", "wqaWHF_s", "wqaWHF22_s")}          # the order of training
for v in SETS.values():
    v["dyn"], v["role"] = BASE_DYN + v["cols"], "test"
    v["n_in"] = len(v["dyn"]) + 1 + len(F.STATIC)
STORED = {"wqa_s": dict(role="base", what="the LSTM with the river gauge and the AORC rain to midnight on the satellite-era split (script 18, E2b): the base"),
          "wqaF_s": dict(role="forecast base", what="wqa_s + the bias-corrected GEFS forecast fc1..fc3 (script 18, E2b): the base of the forecast sets"),
          "wqaP_s": dict(role="ceiling", what="wqa_s + the AORC rain of the three target days in the slots: a perfect rain forecast, the ceiling of this split (" + UPPER_WORDING + ")")}


def real_preds(name):
    """A stored prediction set of the real run (results/preds, also in a smoke test), restricted to the basins of this run."""
    z = np.load(REAL_PRED / f"{name}.npz", allow_pickle=False)
    idx, bas = pd.DatetimeIndex(z["dates"]), [str(b) for b in z["basins"]]
    return {g: pd.DataFrame({"obs": z["obs"][bas.index(g)], **{lk(k): z["pred"][bas.index(g), :, j] for j, k in enumerate(F.LEADS)}}, index=idx) for g in ids}


for n, v in STORED.items():
    assert (REAL_PRED / f"lstm_{n}.npz").exists(), f"the stored set lstm_{n} (18_forecast_input.py) was not found in results/preds"
    meta = json.loads(str(np.load(REAL_PRED / f"lstm_{n}.npz", allow_pickle=False)["meta"]))
    assert meta.get("script") == "18_forecast_input.py" and meta.get("arm") == "sat" and meta.get("config") == CONF_REAL and {k: tuple(x) for k, x in meta["split"].items()} == SAT_SPLIT \
        and tuple(meta["dyn"][:len(BASE_DYN)]) == BASE_DYN, f"lstm_{n} is not script 18's satellite-era set trained with the settings of results/lstm_all.json on this split"
    v["dyn"], v["seeds"] = tuple(meta["dyn"]), [s for s in range(5) if (REAL_PRED / f"lstm_{n}_s{s}.npz").exists()]
    assert len(v["seeds"]) >= 3, f"lstm_{n}: fewer than three single-seed sets were found"
SHOWN = list(SETS) + list(STORED)
have_seed = lambda name, seed: (F.PRED / f"lstm_{name}_s{seed}.npz").exists() and (MODELS / f"lstm_{name}_s{seed}.pt").exists()
frames_of = lambda run, tab: {g: p.reindex(D[g].index).assign(obs=D[g].q_mm) for g, p in run.predict(tab, ids, B).items()}      # back on the whole record, as script 18 stores its satellite-era sets


def save_model(run, name, seed):
    """results/models/lstm_<set>_s<seed>.pt: a plain dict (tensors, numbers, lists and text) that torch.load(weights_only=True) reads, enough to forecast again (the form of scripts 16 and 18)."""
    import torch
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save(dict(state_dict={k: t.detach().cpu().clone() for k, t in run.net.state_dict().items()}, cfg=asdict(run.cfg), sc={k: (x.tolist() if isinstance(x, np.ndarray) else float(x)) for k, x in run.sc.items()},
                    ids=list(run.ids), dyn=list(run.cfg.dyn), split=dict(run.split), arm="sat", best_val_nse=float(run.best_val_nse), n_params=int(run.n_params), input_set=name, script=SCRIPT, seed=seed,
                    history=run.history, seconds=float(run.seconds), cpu_seconds=float(run.cpu_seconds), updates_run=int(run.updates_run), best_update=int(run.best_update), torch_version=str(torch.__version__),
                    quick_mode=F.QUICK, extra_columns=list(SETS[name]["cols"]), last_available_hour=SETS[name]["hour"], carry_h=CARRY_H,
                    how_to_reload="cfg = flood.Cfg(**cfg); net = flood.LSTMNet(n_in, cfg); net.load_state_dict(state_dict); sc lists back to numpy arrays; flood.Run(cfg, sc, net, ids, history, seconds, cpu_seconds, "
                                  f"updates_run, best_update, best_val_nse, n_params, split).predict(D, ids, static_df); D = the satellite-era tables from 2000-01-01 that {SCRIPT} builds (see its docstring)"), tmp)
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


def reload_and_forecast(name, seed):
    """Rebuild the trained network from its file alone, forecast again on the SAME tables and compare with the stored forecasts (same machine: rtol 1e-6)."""
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    run, z = run_from_file(f)
    assert z["input_set"] == name and z["script"] == SCRIPT and z["seed"] == seed, f"{f.name} is not the network of {name}, seed {seed}"
    cols, new, old_ = ["obs"] + [lk(k) for k in F.LEADS], frames_of(run, TAB), F.load_preds(f"lstm_{name}_s{seed}")
    a, b = np.stack([old_[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    same_nan, close = bool(np.array_equal(np.isnan(a), np.isnan(b))), bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
    assert same_nan and close, f"{f.name}: the reloaded network does not reproduce the stored forecasts"
    return dict(set=name, seed=seed, model_file=rel(f), size_kb=round(f.stat().st_size / 1024, 1), n_in=int(len(run.cfg.dyn) + 1 + len(run.cfg.static)), n_params=int(z["n_params"]), n_values_compared=int(a.size),
                max_abs_difference=float(np.nanmax(np.abs(a - b))), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), same_missing_days=same_nan, allclose=close, passed=True)


def record_of(name, seed):
    """The training record of a saved network: from the earlier results file when it holds that set and seed, otherwise from the network's own file."""
    for r in (old or {}).get("models", {}).get(name, {}).get("per_seed", []):
        if r["seed"] == seed:
            return r
    import torch
    z = torch.load(MODELS / f"lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    return dict(seed=seed, **{k: z[k] for k in COST}, model_file=rel(MODELS / f"lstm_{name}_s{seed}.pt"), history=z["history"])


# ---- 2. training (or reading) every set, SEEDS seeds each -------------------------------------------------------------------------------------------------
old = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if LOAD and RESULTS_FILE.exists() else None
conf = old["config"] if old else dict(CONF_REAL, updates=40, eval_every=20) if F.QUICK else dict(CONF_REAL)
assert F.QUICK or conf == CONF_REAL, "the settings of the existing sets are not those of results/lstm_all.json: new sets would not be trained with the same settings"
missing = [(n, s) for n in SETS for s in SEEDS if not have_seed(n, s)]
assert not (RESCORE and missing), "--rescore trains nothing, and these are missing: " + ", ".join(f"lstm_{n}_s{s}" for n, s in missing) + ": run --extend first"
t_ask = time.time()
runs, ENS, costs, files, HIST, trained_now = {}, {}, {}, {}, {}, {}
with F.cpu_lock(SCRIPT) if missing else contextlib.nullcontext():
    waited = time.time() - t_ask
    for name, v in SETS.items():
        runs[name], costs[name], files[name], HIST[name], trained_now[name] = [], {}, {}, {}, []
        for seed in SEEDS:
            if LOAD and have_seed(name, seed):
                p, meta, rec = F.load_preds(f"lstm_{name}_s{seed}"), F.preds_meta(f"lstm_{name}_s{seed}"), record_of(name, seed)
                assert meta["config"] == conf and meta["seed"] == rec["seed"] == seed and meta["script"] == SCRIPT and meta["dyn"] == list(v["dyn"]), f"lstm_{name}_s{seed} is not from the training run of this script"
                cost, mf, hist = {k: rec[k] for k in COST}, rec.get("model_file"), rec.get("history")
            else:
                cfg = F.Cfg(dyn=v["dyn"], use_flow=True, static=tuple(F.STATIC), seed=seed, **conf)
                print(f"training lstm_{name} seed {seed}: {len(cfg.dyn)} daily inputs + flow + {len(cfg.static)} basin properties = {v['n_in']} inputs", flush=True)
                run = F.fit(cfg, TAB, ids, split=SAT_SPLIT, static_df=B, log=lambda s, n_=name, s_=seed: print(f"  [lstm_{n_} s{s_}]{s}", flush=True))
                assert run.cfg.dyn == v["dyn"] and run.n_params == 4 * conf["hidden"] * (v["n_in"] + conf["hidden"] + 2) + (conf["hidden"] + 1) * len(F.LEADS), "the network has not the expected size"
                p, cost, hist, mf = frames_of(run, TAB), run.summary(), run.history, rel(save_model(run, name, seed))
                F.save_preds(f"lstm_{name}_s{seed}", p, dict(script=SCRIPT, model="lstm", inputs=name, seed=seed, config=conf, dyn=list(cfg.dyn), extra_inputs=list(v["cols"]), arm="sat", split=SAT_SPLIT, role=v["role"],
                                                             last_available_hour=v["hour"], model_file=mf, basins=ids, what=f"one LSTM for all {n_b} basins on the satellite-era split, a single seed (E4e)"))
                trained_now[name].append(seed)
            e = F.evaluate(p, thr, "test", split=SAT_SPLIT)
            runs[name].append(p)
            costs[name][seed], files[name][seed], HIST[name][seed] = cost, mf, hist
            print(f"lstm_{name} seed {seed} | best at update {cost['best_update']} of {cost['updates_run']}, {cost['seconds']:.0f} s | validation NSE {cost['best_val_nse']:.3f} | test NSE median "
                  + " ".join(f"{e['summary'][lk(k)]['nse_median']:.3f}" for k in F.LEADS) + f" | onsets warned 1 day {100 * e['summary']['l1']['onset']['pod']:.1f}%" + ("" if seed in trained_now[name] else " | read"), flush=True)
        ENS[name] = F.ensemble(runs[name])
        if trained_now[name]:
            F.save_preds(f"lstm_{name}", ENS[name], dict(script=SCRIPT, model="lstm", inputs=name, seeds=SEEDS, config=conf, dyn=list(v["dyn"]), extra_inputs=list(v["cols"]), arm="sat", split=SAT_SPLIT, role=v["role"],
                                                      last_available_hour=v["hour"], basins=ids, what=f"one LSTM for all {n_b} basins on the satellite-era split, day-by-day average of the seeds (E4e)"))
trained_pairs = [(n, s) for n in SETS for s in trained_now[n]]
reload_check = dict(what="one trained network per set (the first seed) rebuilt from its .pt file alone, run on the same tables, compared with the forecasts stored by save_preds (same machine; the Mac-trained stored sets are not reloaded)",
                    tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal", per_set={n: reload_and_forecast(n, SEEDS[0]) for n in SETS})
print("reload check passed: " + ", ".join(f"{n} (max difference {r['max_abs_difference']:.1e}{', bit for bit' if r['bit_for_bit'] else ''})" for n, r in reload_check["per_set"].items()), flush=True)

# ---- 3. causality: made-up hours (data side), subdaily_inputs.checks, flood._assert_causal (engine), made-up values on the saved networks (end to end) --------
t_c = time.time()


def made_up_hours_test(g, last_hour):
    """Every hourly value after the issue instant of a cut day (the hour ending `last_hour` local) is replaced by random numbers: no column of any day up to the cut day may move, the next day must; for the 22:00 variant
    the 24:00 columns of the cut day itself must move (they read the hours ending 23:00 and 24:00). Cut days: three test-year days on which both variants are hourly_known on the day and the next."""
    stamp, qm, ql, _, _ = HOURLY_DATA[g]
    q_day, rng, rec = TAB[g].q_mm, np.random.default_rng(22), []
    base = {h: flow_state_table(stamp, qm, ql, g, q_day, h) for h in (24, 22)}
    ok = (TAB[g].hourly_known == 1) & (TAB[g].hourly_known_22 == 1)
    cand = TAB[g].index[(ok & ok.shift(-1, fill_value=False)).values & (TAB[g].index >= pd.Timestamp(SAT_SPLIT["test"][0]))]
    for cut in (cand[len(cand) // 4], cand[len(cand) // 2], cand[3 * len(cand) // 4]):
        instant = (cut + pd.Timedelta(hours=last_hour)).tz_localize(ZoneInfo(ZONES[g])).tz_convert("UTC").tz_localize(None)
        later = np.asarray(stamp > instant)
        qm2, ql2 = np.where(later, rng.uniform(0, 5000, len(qm)), qm), np.where(later, rng.uniform(0, 5000, len(ql)), ql)
        alt, nxt = {h: flow_state_table(stamp, qm2, ql2, g, q_day, h) for h in (24, 22)}, cut + pd.Timedelta(days=1)
        same = base[last_hour].loc[:cut].fillna(-1).equals(alt[last_hour].loc[:cut].fillna(-1))
        moved = not base[last_hour].loc[nxt].fillna(-1).equals(alt[last_hour].loc[nxt].fillna(-1))
        moved_24 = last_hour == 24 or not base[24].loc[cut].fillna(-1).equals(alt[24].loc[cut].fillna(-1))
        assert same and moved and moved_24, f"{g}, last hour {last_hour}: an hourly-flow column read an hour after the issue instant (same={same}, next day moved={moved}, 24:00 columns moved={moved_24})"
        rec.append(dict(cut_day=str(cut.date()), issue_instant_utc=str(instant), hours_replaced=int(later.sum()), rows_up_to_cut_unchanged=True, next_day_moved=True, **({"columns_24_of_cut_day_moved": True} if last_hour == 22 else {})))
    return rec


made_up_hours = dict(what="every hourly value (mean and last, missing hours too) after the issue instant of a cut day replaced by random numbers (0 to 5000 cfs) and the columns rebuilt: no row up to the cut day changed (exact equality "
                          "of all columns), the next day changed; for the 22:00 variant the issue instant is 22:00 of the cut day and the 24:00 columns of the cut day changed too", per_basin={g: {"24": made_up_hours_test(g, 24), "22": made_up_hours_test(g, 22)} for g in ids}, passed=True)
SI_REC = SI.checks(log=lambda s: print("  [subdaily_inputs.checks] " + s, flush=True))
DATA_SIDE = dict(what=SI_REC["what"], how="code/subdaily_inputs.checks(): routes agree, window order, 23/24/25-hour days, the made-up-hours test of the within-day rain columns; record rewritten to " + rel(SI.OUT / "subdaily_inputs_check.json"),
                 passed=True, basins=SI_REC["basins"], runtime_s=SI_REC["runtime_s"])


def made_up_values_test(name, seed, n_cuts=3):
    """Script 18's test on a saved network: the set's new columns (and, in a second round, every input column and the flow) of all days AFTER a cut day c are replaced by random values; a forecast issued on or before c
    must not change (stored at the target day t+k: array_equal on every lead up to c+k), and the forecasts issued after c must change (or the test proves nothing)."""
    run_, _ = run_from_file(MODELS / f"lstm_{name}_s{seed}.pt")
    idx, (a, b), rng = TAB[ids[0]].index, SAT_SPLIT["test"], np.random.default_rng(7)
    cuts = sorted(int(c) for c in rng.choice(np.flatnonzero((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))[200:-200], n_cuts, replace=False))
    base, off = frames_of(run_, TAB), len(D[ids[0]].index) - len(idx)        # the frames are on the whole record; the tables start in 2000
    out = dict(set=name, seed=seed, cut_days=[str(idx[c].date()) for c in cuts], rounds={})
    for what, cols in (("the new columns of the set only", list(SETS[name]["cols"])), ("every input column and the observed flow", sorted(set(run_.cfg.dyn) | {"q_mm"}))):
        n_same, n_moved = 0, 0
        for c in cuts:
            alt = {g: t.copy() for g, t in TAB.items()}
            for g in ids:
                for col in cols:
                    x = alt[g][col].values.copy()
                    x[c + 1:] = rng.uniform(0.0, 120.0, len(x) - c - 1)
                    alt[g][col] = x
            new = frames_of(run_, alt)
            for g in ids:
                for k in F.LEADS:
                    assert np.array_equal(base[g][lk(k)].values[:off + c + k + 1], new[g][lk(k)].values[:off + c + k + 1], equal_nan=True), f"{name} {g}: changing {what} after {idx[c].date()} changed a forecast issued on or before that day (lead {k})"
                    n_same += 1
            moved = [not np.allclose(base[g][lk(1)].values[off + c + 2:off + c + 12], new[g][lk(1)].values[off + c + 2:off + c + 12], rtol=0, atol=1e-9) for g in ids]
            assert any(moved), f"{name}: the test could not see a change after the cut (it proves nothing)"
            n_moved += int(sum(moved))
        out["rounds"][what] = dict(forecasts_compared_equal=n_same, basins_whose_later_forecasts_moved_summed_over_the_cuts=n_moved, passed=True)
    return out


causality = dict(
    what="(a) data side: the made-up-hours test on the hourly-flow columns (every basin, both variants) and code/subdaily_inputs.checks() on the within-day rain columns; (b) engine: flood.fit runs flood._assert_causal on every network "
         "before training (a trained network exists only if it passed); (c) end to end: the made-up-values test on one SAVED network per set (first seed; the new columns, then every input column and the flow, replaced by random values "
         "after a cut day: no forecast issued on or before the cut changed, later ones did); (d) every column is dated by the ISSUE day t and holds hours ending at or before the last available hour of day t",
    made_up_hours=made_up_hours, data_side=DATA_SIDE, engine_check="flood._assert_causal at the start of every flood.fit", made_up_values={n: made_up_values_test(n, SEEDS[0]) for n in SETS}, seconds=round(time.time() - t_c, 1))
print(f"causality: made-up hours passed on {n_b} basins x 2 variants; made-up values passed on " + ", ".join(causality["made_up_values"]) + f" ({causality['seconds']:.0f} s)", flush=True)

# ---- 4. every set scored the same way (the stored sets rescored by this code, never copied) ----------------------------------------------------------------
PRED = {**ENS, **{n: real_preds(f"lstm_{n}") for n in STORED}}
SEEDP = {**runs, **{n: [real_preds(f"lstm_{n}_s{s}") for s in STORED[n]["seeds"]] for n in STORED}}
SEED_LABELS = {**{n: list(SEEDS) for n in SETS}, **{n: list(STORED[n]["seeds"]) for n in STORED}}
ev = lambda P, part="test": F.evaluate(P, thr, part, split=SAT_SPLIT)
EV = {n: dict(test=ev(PRED[n]), val=ev(PRED[n], "val"), seeds=[ev(p) for p in SEEDP[n]]) for n in SHOWN}
assert all(EV[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in SHOWN for part in ("test", "val") for k in F.LEADS), "days without a forecast: the models would not be compared on identical days"
assert all(np.allclose(PRED[n][g].obs.values, D[g].q_mm.values) for n in SHOWN for g in ids), "the observed flow differs between the sets"
med = lambda E, k: E["summary"][lk(k)]["nse_median"]
ons = lambda E, k, c="pod": E["summary"][lk(k)]["onset"][c]
nse_of = lambda E, k: np.array([E["basins"][g][lk(k)]["nse"] for g in ids])
row = lambda E, k: dict(nse_median=med(E, k), nse_mean=E["summary"][lk(k)]["nse_mean"], onset=CUT(E["summary"][lk(k)]["onset"]), all=CUT(E["summary"][lk(k)]["all"]))


def seed_range(n, k):
    S = [e["summary"][lk(k)] for e in EV[n]["seeds"]]
    return dict(n_seeds=len(S), nse_median=lo_hi([s["nse_median"] for s in S]), **{f"{kind}_{c}": lo_hi([s[kind][c] for s in S]) for kind in ("onset", "all") for c in ("pod", "far", "csi")},
                by_seed=dict(seeds=SEED_LABELS[n], nse_median=[s["nse_median"] for s in S], **{f"{kind}_{c}": [s[kind][c] for s in S] for kind in ("onset", "all") for c in ("pod", "far", "csi")}))


summary_test, summary_val = ({n: {lk(k): row(EV[n][part], k) for k in F.LEADS} for n in SHOWN} for part in ("test", "val"))
ranges = {n: {lk(k): seed_range(n, k) for k in F.LEADS} for n in SHOWN}
per_basin = {"how_to_read": "test years of the split, one value per basin; the seed average of each set (flood.evaluate); gauge ids are text", "basins": {g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=AREA[g]) for g in ids},
             **{key: {n: {lk(k): {g: (EV[n]["test"]["basins"][g][lk(k)]["nse"] if c is None else EV[n]["test"]["basins"][g][lk(k)]["onset"][c]) for g in ids} for k in F.LEADS} for n in SHOWN}
                for key, c in (("nse", None), ("onset_warned", "pod"), ("onset_false", "far"))}}
models = {n: dict(saved_as=f"lstm_{n}", n_seeds_scored=len(EV[n]["seeds"]), seeds=SEED_LABELS[n], stored=n in STORED, test=EV[n]["test"]) for n in SHOWN}
for n in SETS:
    models[n]["per_seed"] = [dict(seed=s, **{c: costs[n][s][c] for c in COST}, model_file=files[n][s], history=HIST[n][s], test_nse_median=[med(e, k) for k in F.LEADS], test_onset_warned=[ons(e, k) for k in F.LEADS],
                                  test_onset_false=[ons(e, k, "far") for k in F.LEADS]) for s, e in zip(SEEDS, EV[n]["seeds"])]
training = [dict(set=n, seed=s, saved_forecasts=f"lstm_{n}_s{s}", model_file=files[n][s], **{c: costs[n][s][c] for c in COST}) for n in SETS for s in SEEDS]


def paired_seeds(sT, sC):
    """The seeds the seed-by-seed difference list covers: zip() pairs the two sets' seed lists in order and stops at the shorter one (a five-seed set against a three-seed stored set: seeds 0, 1, 2;
    seeds 3 and 4 of the new set are not in the list; review of 2026-10-29, should-fix 7)."""
    pairs = list(zip(sT["by_seed"]["seeds"], sC["by_seed"]["seeds"]))
    assert all(a == b for a, b in pairs), "the seed-by-seed differences would pair unequal seed numbers"
    return [a for a, _ in pairs]


READING_HOW = ("per pair (T minus C) and lead: the seed rule and the checks of report section 4.2 (by_lead) and the onsets warned by both, only one, neither (paired_onsets); seed_range_* and single_seeds_all_* use all seeds "
               "of each set; difference_by_seed_points pairs the same seed number of both sets in order and covers only the seeds both sets have (seeds_paired: {cover})")


def reading_how(T=None, C=None):
    """READING_HOW with the seeds the lists of pair T, C cover, from the two seed lists (review round 2 of 30 October 2026, should-fix 2); without a pair, a pointer to the pairs."""
    if T is None:
        return READING_HOW.format(cover="the how_to_read of each pair names the seeds its list covers")
    sT, sC = (ranges[x][lk(1)]["by_seed"]["seeds"] for x in (T, C))
    sp = paired_seeds(ranges[T][lk(1)], ranges[C][lk(1)])
    w = lambda n: NUMBER_WORD.get(n, n)
    return READING_HOW.format(cover=(f"here seeds {', '.join(map(str, sp))}, every seed of both sets" if len(sp) == len(sT) == len(sC) else
                                     f"here seeds {', '.join(map(str, sp))}: lstm_{T} has {w(len(sT))} seeds and lstm_{C} {w(len(sC))}, so the seeds of the larger set beyond these are not in the list"))


def rule(T, C, k):
    """Set T against set C at lead k (report section 4.2). NSE: every seed of T above every seed of C, higher in most basins, Wilcoxon p below P_CUT = clear gain. Onsets: pooled, so only the seeds compare."""
    ET, EC = EV[T]["test"], EV[C]["test"]
    d = nse_of(ET, k) - nse_of(EC, k)
    try:
        p = float(wilcoxon(nse_of(ET, k), nse_of(EC, k)).pvalue)
    except ValueError:
        p = float("nan")
    sT, sC = ranges[T][lk(k)], ranges[C][lk(k)]
    up, down, n_hi, n_lo = above(sT["nse_median"], sC["nse_median"]), above(sC["nse_median"], sT["nse_median"]), int((d > 0).sum()), int((d < 0).sum())
    gain, loss = up and n_hi > n_b / 2 and p < P_CUT, down and n_lo > n_b / 2 and p < P_CUT
    o_up, o_down = above(sT["onset_pod"], sC["onset_pod"]), above(sC["onset_pod"], sT["onset_pod"])
    return dict(against=f"lstm_{C}", lead=k,
                nse=dict(median_with=med(ET, k), median_without=med(EC, k), gain=med(ET, k) - med(EC, k), seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"], single_seeds_all_above=up, single_seeds_all_below=down,
                         n_basins_higher=n_hi, n_basins_lower=n_lo, n_basins=n_b, median_basin_difference=float(np.median(d)), wilcoxon_p=p, clear_gain=bool(gain), clear_loss=bool(loss),
                         reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                onset=dict(warned_with=ons(ET, k), warned_without=ons(EC, k), gain_points=100 * (ons(ET, k) - ons(EC, k)), hits_with=ET["summary"][lk(k)]["onset"]["hits"], hits_without=EC["summary"][lk(k)]["onset"]["hits"],
                           seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"], seeds_paired=paired_seeds(sT, sC), difference_by_seed_points=[100 * (x - y) for x, y in zip(sT["by_seed"]["onset_pod"], sC["by_seed"]["onset_pod"])],
                           false_with=ons(ET, k, "far"), false_without=ons(EC, k, "far"), single_seeds_all_above=o_up, single_seeds_all_below=o_down, reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference"))


def onset_flags(P, k):
    """One True / False per onset of the test years (warned or not), basin after basin, as flood.flood_scores counts them."""
    a, b, out = *SAT_SPLIT["test"], []
    for g in ids:
        p = P[g]
        was_low, q = (p.obs.shift(k) <= thr[g]).loc[a:b], p.loc[a:b]
        ok = (q.obs.notna() & q[lk(k)].notna()).values
        o, s, w = (q.obs > thr[g]).values[ok], (q[lk(k)] > thr[g]).values[ok], was_low.values[ok]
        out.append(s[o & w])
    return np.concatenate(out)


def overlap(A, C, k):
    """The onsets warned by both sets, by only one, by neither; the exact sign test is a description only (the onsets of one storm are not independent)."""
    fa, fc = onset_flags(PRED[A], k), onset_flags(PRED[C], k)
    ta, tc = EV[A]["test"]["summary"][lk(k)]["onset"], EV[C]["test"]["summary"][lk(k)]["onset"]
    assert len(fa) == len(fc) == ta["hits"] + ta["misses"] and int(fa.sum()) == ta["hits"] and int(fc.sum()) == tc["hits"], "the onsets counted here are not those of flood.evaluate"
    both, only_a, only_c = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    return dict(first=f"lstm_{A}", second=f"lstm_{C}", lead=k, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=only_a, warned_only_by_second=only_c,
                warned_by_neither=int((~fa & ~fc).sum()), sign_test_p=float(binomtest(only_a, only_a + only_c, 0.5).pvalue) if only_a + only_c else float("nan"))


PAIRS = [(n, v["base"], f"lstm_{n} minus lstm_{v['base']} (stored): the worth of the new columns") for n, v in SETS.items()]
PAIRS += [(f"wqa{f}_s", f"wqa{f}22_s", f"delivery: lstm_wqa{f}_s (last hour ending 24:00) minus lstm_wqa{f}22_s (ending 22:00)") for f in ("H", "WH", "WHF")]
PAIRS += [("wqaWH_s", "wqaH_s", "lstm_wqaWH_s minus lstm_wqaH_s: the within-day rain on top of the river's state"), ("wqaWHF_s", "wqaWH_s", "lstm_wqaWHF_s minus lstm_wqaWH_s: the forecast on top of both"),
          ("wqaH_s", "wqaP_s", "lstm_wqaH_s minus lstm_wqaP_s (perfect rain in the slots, " + UPPER_WORDING + ")"), ("wqaWHF_s", "wqaP_s", "lstm_wqaWHF_s minus lstm_wqaP_s"), ("wqaF_s", "wqa_s", "the stored forecast set minus the stored base")]
readings = {f"{T}_vs_{C}": dict(what=what, how_to_read=reading_how(T, C), by_lead={lk(k): rule(T, C, k) for k in F.LEADS}, paired_onsets={lk(k): overlap(T, C, k) for k in F.LEADS}) for T, C, what in PAIRS}


def closure_of(name, base, ceil_):
    rows = []
    for k in F.LEADS:
        Eb, Ec, En = (EV[x]["test"] for x in (base, ceil_, name))
        b, c, n, ob, oc, on = med(Eb, k), med(Ec, k), med(En, k), ons(Eb, k), ons(Ec, k), ons(En, k)
        rows.append(dict(lead=k, nse_base=b, nse_with=n, nse_ceiling=c, share_of_nse_gap_closed=div(n - b, c - b), basins_better_than_base=int((nse_of(En, k) > nse_of(Eb, k)).sum()), onset_warned_base=ob, onset_warned_with=on,
                         onset_warned_ceiling=oc, onset_gain_points=100 * (on - ob), onset_share_of_gap_closed=div(on - ob, oc - ob), onset_false_base=ons(Eb, k, "far"), onset_false_with=ons(En, k, "far"), onset_false_ceiling=ons(Ec, k, "far")))
    return dict(set=f"lstm_{name}", base=f"lstm_{base}", ceiling=f"lstm_{ceil_}", by_lead=rows)


closure = {**{n: closure_of(n, v["base"], "wqaP_s") for n, v in SETS.items()}, "wqaF_s": closure_of("wqaF_s", "wqa_s", "wqaP_s")}
CRIT = FA["verdict"]["criteria"]
POD_GO, FAR_GO, CSI_CAUTION = (float(CRIT[key].split(f"{w} ")[1].split()[0]) for key, w in (("GO", "pod >="), ("GO", "far <="), ("CAUTION", "csi >=")))
verdict = lambda t: "NO EVENTS" if t["hits"] + t["misses"] == 0 else "GO" if t["pod"] >= POD_GO and t["far"] <= FAR_GO else "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"


def operating_points(n):
    """Per lead: trigger 1.0, and two triggers chosen on the VALIDATION years with code/warning_policy.choose_trigger (csi: best pooled onset CSI, the project's rule; far_cap: most onsets warned with at most 30% of the onset
    warnings false), scored once on the test years; the test FAR is reported as it comes out. Every single seed is scored at the same trigger."""
    out = {}
    for k in F.LEADS:
        A_val, A_test = WP.arrays(PRED[n], thr, k, "val", split=SAT_SPLIT), WP.arrays(PRED[n], thr, k, "test", split=SAT_SPLIT)
        A_seeds, rec = [WP.arrays(p, thr, k, "test", split=SAT_SPLIT) for p in SEEDP[n]], dict(n_test_onsets=int(A_test["obs_hi"][A_test["was_low"]].sum()), rules={})
        for rule_ in ("plain", "csi", "far_cap"):
            c = dict(trigger=1.0, reached=True, validation=WP.score_at(A_val, 1.0)) if rule_ == "plain" else WP.choose_trigger(A_val, rule_)
            t, st = c["trigger"], WP.score_at(A_test, c["trigger"])
            seeds = [WP.score_at(A, t)["onset"] for A in A_seeds]
            rec["rules"][rule_] = dict(trigger=float(t), reached=bool(c["reached"]), **({"note": c["note"]} if "note" in c else {}), validation=dict(onset=CUT(c["validation"]["onset"]), all=CUT(c["validation"]["all"])),
                                       test=dict(onset=CUT(st["onset"]), all=CUT(st["all"])), test_verdict=dict(onset=verdict(st["onset"]), all=verdict(st["all"])),
                                       single_seeds_at_this_trigger=dict(n_seeds=len(seeds), onset_pod=lo_hi([s["pod"] for s in seeds]), onset_far=lo_hi([s["far"] for s in seeds]), by_seed=dict(seeds=SEED_LABELS[n], onset_pod=[s["pod"] for s in seeds], onset_far=[s["far"] for s in seeds])))
        out[lk(k)] = rec
    return out


OPS = {n: operating_points(n) for n in SHOWN}
OPS_NOTE = dict(how_to_read="per set and lead: plain = trigger 1.0; csi and far_cap = triggers chosen on the validation years of this split by code/warning_policy.choose_trigger (grid 0.50 to 2.00 step 0.05) and scored once on the test years "
                            f"(far_cap: most onsets warned with pooled onset FAR at most {WP.FAR_CAP:.2f} on validation; reached=false: no trigger met the cap, the lowest-FAR trigger is given); the test FAR is the one that came out; "
                            "test_verdict applies the GO / CAUTION / NOT YET criteria of results/flood_analysis.json", criteria=CRIT, sets=OPS)
fc_pt = lambda n, k: dict(trigger=OPS[n][lk(k)]["rules"]["far_cap"]["trigger"], **OPS[n][lk(k)]["rules"]["far_cap"]["test"]["onset"])
delivery = dict(how_to_read="the same inputs with the last available hour of the gauge record ending at the issue time (24:00) and two hours before it (22:00), side by side, per family and lead (test years; onsets pooled at trigger 1.0; "
                            "far_cap = the 30%-false operating point; the rule and the paired onsets of each pair are in readings)",
                families={fam: {lk(k): dict(last_hour_24=a, last_hour_22=b, onset_warned_24=ons(EV[a]["test"], k), onset_warned_22=ons(EV[b]["test"], k), difference_points_24_minus_22=r["onset"]["gain_points"],
                                            seed_range_24=r["onset"]["seed_range_with"], seed_range_22=r["onset"]["seed_range_without"], every_seed_24_above_22=r["onset"]["single_seeds_all_above"], every_seed_22_above_24=r["onset"]["single_seeds_all_below"],
                                            onset_false_24=r["onset"]["false_with"], onset_false_22=r["onset"]["false_without"], nse_median_24=r["nse"]["median_with"], nse_median_22=r["nse"]["median_without"], nse_reading=r["nse"]["reading"],
                                            far_cap_point_24=fc_pt(a, k), far_cap_point_22=fc_pt(b, k)) for k in F.LEADS for a, b, r in [(f"wqa{fam}_s", f"wqa{fam}22_s", readings[f"wqa{fam}_s_vs_wqa{fam}22_s"]["by_lead"][lk(k)])]}
                          for fam in ("H", "WH", "WHF")})
r1 = {x: ranges[x]["l1"]["onset_pod"] for x in SHOWN}
by1 = lambda n: list(zip(ranges[n]["l1"]["by_seed"]["seeds"], ranges[n]["l1"]["by_seed"]["onset_pod"]))      # (seed, onsets warned at lead 1) of a set's single trainings
FIRST3 = [0, 1, 2]                                                                                           # the script's default seed count (--seeds 3); the work order names none (review of 2026-10-29, should-fix 3)
r1_first3 = lo_hi([v for s, v in by1("wqaH_s") if s in FIRST3])
below_best = [s for s, v in by1("wqaH_s") if v <= r1["wqa_s"][1]]                                             # the wqaH_s seeds not above the best wqa_s seed: the ones that decide the verdict
E4E = dict(rule=SUCCESS_RULE, lead=1, met=above(r1["wqaH_s"], r1["wqa_s"]), met_with_first_three_seeds=above(r1_first3, r1["wqa_s"]), first_three_seeds=FIRST3, seed_range_of_wqaH_s_first_three_seeds=r1_first3,
           seeds_of_wqaH_s_not_above_the_best_wqa_s_seed=below_best,
           seed_count_note=f"the work order names no seed count and the script's default is {NUMBER_WORD[ap.get_default('seeds')]}; met = the rule over all {ranges['wqaH_s']['l1']['n_seeds']} seeds of wqaH_s against the "
                           f"{ranges['wqa_s']['l1']['n_seeds']} of the stored wqa_s; met_with_first_three_seeds = the same rule with seeds 0, 1, 2 of wqaH_s only (the verdict depends on the seed count when the two differ; "
                           "review of 2026-10-29, should-fix 3)",
           every_seed_of_wqaH22_s_above_every_seed_of_wqa_s=above(r1["wqaH22_s"], r1["wqa_s"]), every_seed_of_wqaWH_s_above_every_seed_of_wqa_s=above(r1["wqaWH_s"], r1["wqa_s"]),
           every_seed_of_wqaWHF_s_above_every_seed_of_wqaF_s=above(r1["wqaWHF_s"], r1["wqaF_s"]), n_seeds={x: ranges[x]["l1"]["n_seeds"] for x in SHOWN}, onset_warned={x: ons(EV[x]["test"], 1) for x in SHOWN},
           onset_false={x: ons(EV[x]["test"], 1, "far") for x in SHOWN}, seed_ranges=r1)
stored_check = {}
if n_b == 12 and (F.RES / "forecast_input.json").exists():                   # did this code, run on the stored forecasts, reproduce what script 18 stored for them?
    S18 = json.loads((F.RES / "forecast_input.json").read_text(encoding="utf-8"))["summary_test"]["sets"]
    for n in STORED:
        diffs = [abs(S18[n][lk(k)][a][b] - summary_test[n][lk(k)][a][b]) if b else abs(S18[n][lk(k)][a] - summary_test[n][lk(k)][a]) for k in F.LEADS for a, b in (("nse_median", None), ("onset", "pod"), ("onset", "far"), ("onset", "hits"), ("all", "csi"))]
        stored_check[n] = dict(file="results/forecast_input.json", key=f"summary_test.sets.{n}", values_compared=len(diffs), max_abs_difference=float(max(diffs)), all_equal=bool(max(diffs) < 1e-9))
        assert stored_check[n]["all_equal"], f"the numbers stored for lstm_{n} in results/forecast_input.json are not reproduced by this script's code"
else:
    notes.append("The rescored numbers of the stored sets were not compared with results/forecast_input.json (a subset of basins, or the file is missing).")

# ---- 5. the sentence --------------------------------------------------------------------------------------------------------------------------------------
n_on = EV["wqaH_s"]["test"]["summary"]["l1"]["onset"]["hits"] + EV["wqaH_s"]["test"]["summary"]["l1"]["onset"]["misses"]
tH, tH22, tA, tWH, tF, tWHF, tP = (EV[x]["test"]["summary"]["l1"]["onset"] for x in ("wqaH_s", "wqaH22_s", "wqa_s", "wqaWH_s", "wqaF_s", "wqaWHF_s", "wqaP_s"))
rH, rA, cl, n_s = ranges["wqaH_s"]["l1"], ranges["wqa_s"]["l1"], closure["wqaH_s"]["by_lead"][0], ranges["wqaH_s"]["l1"]["n_seeds"]
opH, opA, vH, ov = OPS["wqaH_s"]["l1"]["rules"]["far_cap"], OPS["wqa_s"]["l1"]["rules"]["far_cap"], OPS["wqaH_s"]["l1"]["rules"]["plain"]["test_verdict"], readings["wqaH_s_vs_wqa_s"]["paired_onsets"]["l1"]
seeds_word = f"{NUMBER_WORD.get(n_s, n_s)} single training{'s' if n_s != 1 else ''}"
X = ("the state of the river at the issue time from the gauge's own 15-minute record reduced to hours (the flow at local midnight relative to Q95, its change over the last 6 and 12 hours, the day's highest hourly flow "
     "and the hours above Q95) as five extra inputs beside the AORC rain to midnight to the LSTM with the river gauge")
outcome = (f"Given {X}, on {SPLIT_TEXT}, onsets warned 1 day ahead go from {pc(tA['pod'])} with the AORC rain to midnight alone (lstm_wqa_s, stored) to {pc(tH['pod'])} (single trainings {span(rH['onset_pod'])}; the onsets of one storm "
           f"are not independent): {100 * (tH['pod'] - tA['pod']):+.1f} points ({tH['hits']} against {tA['hits']} hits of {n_on}); with the last available hour ending 22:00 instead of 24:00, {pc(tH22['pod'])} (single trainings "
           f"{span(ranges['wqaH22_s']['l1']['onset_pod'])}); the share of false warnings among the onset warnings is {pc(tH['far'])} against {pc(tA['far'])}; the median NSE goes from {n2(med(EV['wqa_s']['test'], 1))} to "
           f"{n2(med(EV['wqaH_s']['test'], 1))} (the seed-averaged forecast; single trainings {span(rH['nse_median'], n2)}); the verdict of results/flood_analysis.json at trigger 1.0 is {vH['onset']} for the onsets and {vH['all']} for all high-flow days.")
# the passes of this file (the training pass first; a rescore pass replaces the earlier rescore pass): the seed count and the script hash of the training pass are quoted in the sentence and in choices
this_pass = dict(kind="training run" if not LOAD else "extension" if EXTEND else "rescore", n_trainings=len(trained_pairs), trained=[f"lstm_{n}_s{s}" for n, s in trained_pairs],
                 training_seconds=round(sum(costs[n][s]["seconds"] for n, s in trained_pairs), 1), waited_for_cpu_s=round(waited, 1), seeds=SEEDS, script_sha256=HASH_NOW, finished_local_time=time.strftime("%Y-%m-%d %H:%M"),
                 torch_threads=int(os.environ.get("FLOOD_THREADS", 1 if F.QUICK else 4)), machine=os.environ.get("FLOOD_MACHINE", platform.machine()))
passes = [p for p in (old or {}).get("passes", []) if p.get("kind") != "rescore"] + [this_pass]
train_passes = [p for p in passes if p.get("kind") == "training run"]
train_pass = train_passes[0] if train_passes else this_pass
n_tp, one_pass = len(train_pass.get("seeds", SEEDS)), len(train_passes) == 1 and not any(p.get("kind") == "extension" for p in passes)
# the delivery reading (review of 2026-10-29, should-fix 4): the 24:00 - 22:00 differences at lead 1 per family, and whether any family separates its two variants seed by seed at any lead
DL = delivery["families"]
rel_word = lambda d: f"{abs(d):.1f} points {'below' if d > 0 else 'above'}" if d else "level with"            # d = onsets warned with the record ending 24:00 minus 22:00, in points
separated = [f"{DL[f][lk(k)]['last_hour_24']} against {DL[f][lk(k)]['last_hour_22']} at {k} day{'s' if k != 1 else ''}" for f in DL for k in F.LEADS if DL[f][lk(k)]["every_seed_24_above_22"] or DL[f][lk(k)]["every_seed_22_above_24"]]
delivery_text = (f"Delivery: with the record ending 22:00 instead of 24:00, onsets warned 1 day ahead are {rel_word(DL['H']['l1']['difference_points_24_minus_22'])} the 24:00 variant in the river-only sets, "
                 f"{rel_word(DL['WH']['l1']['difference_points_24_minus_22'])} with the within-day rain and {rel_word(DL['WHF']['l1']['difference_points_24_minus_22'])} with the forecast as well, "
                 + ("all inside the seed spread at every lead (no family separates its 24:00 and 22:00 variants seed by seed at any lead)" if not separated else "and seed by seed the two variants separate in " + ", ".join(separated)) + ".")
# the seed-rule reading of wqaWHF_s against wqaF_s with the paired onsets (should-fix 4)
rWHF, oWHF = readings["wqaWHF_s_vs_wqaF_s"]["by_lead"]["l1"]["onset"], readings["wqaWHF_s_vs_wqaF_s"]["paired_onsets"]["l1"]
pv = lambda p: f"{p:.3f}" if p == p else "no value"
whf_reading = (f"a clear {'gain' if rWHF['single_seeds_all_above'] else 'loss'} by the seed rule (every single seed {'above' if rWHF['single_seeds_all_above'] else 'below'} every single seed of wqaF_s)" if rWHF["reading"] != "no clear difference"
               else f"{rWHF['reading']} by the seed rule, so the {rWHF['gain_points']:+.1f} points rest on the seed-averaged forecast and on the paired onsets: {oWHF['warned_only_by_first']} warned only with the new columns against "
                    f"{oWHF['warned_only_by_second']} only without, sign test p = {pv(oWHF['sign_test_p'])}, a description only")
# the success rule and the seed count (should-fix 3): the verdict with all seeds and with the first three, and which seeds decide it
rule_text = ("is met" if E4E["met"] else "is not met" if n_s <= 3 else
             f"is not met with the {NUMBER_WORD.get(n_s, n_s)} seeds and would be with the first three (seeds 0, 1, 2): the verdict depends on the seed count ({NUMBER_WORD.get(n_tp, n_tp)} seeds were chosen at launch with "
             f"--seeds {n_tp}, the script's default being {NUMBER_WORD[ap.get_default('seeds')]}, {'in one training pass ' if one_pass else ''}before any test score was seen)" if E4E["met_with_first_three_seeds"] else
             f"is not met (neither with the {NUMBER_WORD.get(n_s, n_s)} seeds nor with the first three)")
below_text = (f" (seed{'s' if len(below_best) != 1 else ''} {', '.join(str(s) for s in below_best)} of wqaH_s {'are' if len(below_best) != 1 else 'is'} not above the best wqa_s seed)" if below_best and not E4E["met"] else "")
appr_min = min(v["approved_share_after_2000"] for v in COVERAGE.values())
approved_text = "this study used the approved record" if appr_min >= 1 else f"this study used the record as downloaded, approved share {p1(appr_min)} or more in every gauge"
outcome_caveat = (
    f"The success rule fixed before training ({SUCCESS_RULE.split(':')[0]}) {rule_text}: the {seeds_word} with the river's state span {span(rH['onset_pod'], p1)} against "
    f"{span(rA['onset_pod'], p1)} without it{below_text}; {ov['warned_only_by_first']} onsets are warned only with it against {ov['warned_only_by_second']} only without it. With the six within-day AORC rain columns beside it (wqaWH_s) "
    f"{pc(tWH['pod'])} (single trainings {span(ranges['wqaWH_s']['l1']['onset_pod'])}); with the GEFS forecast as well (wqaWHF_s against the stored lstm_wqaF_s) 1-day onsets warned go from {pc(tF['pod'])} to {pc(tWHF['pod'])} "
    f"(single trainings {span(ranges['wqaWHF_s']['l1']['onset_pod'])} against {span(ranges['wqaF_s']['l1']['onset_pod'])}: {whf_reading}; false {pc(tWHF['far'])} against {pc(tF['far'])}), at 2 and 3 days "
    + " and ".join(f"from {pc(ons(EV['wqaF_s']['test'], k))} to {pc(ons(EV['wqaWHF_s']['test'], k))}" for k in (2, 3))
    + f". {delivery_text} The five columns close {pc(cl['onset_share_of_gap_closed'])} of the gap in onsets warned from wqa_s ({pc(tA['pod'])}) to the perfect rain forecast in the slots (wqaP_s, {pc(tP['pod'])}, {UPPER_WORDING}) and "
    f"{pc(cl['share_of_nse_gap_closed'])} of the gap in median NSE. At the 30%-false operating point (the trigger chosen on the validation years so that at most 30% of the onset warnings are false: {opH['trigger']:.2f}"
    f"{'' if opH['reached'] else ', cap not reached on validation'}) wqaH_s warns of {pc(opH['test']['onset']['pod'])} with {pc(opH['test']['onset']['far'])} false on the test years, against {pc(opA['test']['onset']['pod'])} with "
    f"{pc(opA['test']['onset']['far'])} false for wqa_s (trigger {opA['trigger']:.2f}). The hourly record has gaps: on {N_FB[24]} of the {sum(v['days'] for v in COVERAGE.values())} basin-days ({sum(v['fallback_days_24_test_years'] for v in COVERAGE.values())} "
    f"in the test years) the five columns fall back to the daily mean (hourly_known = 0); the hourly flow is the gauge's own record, an input a real-time service has as provisional values and after a delivery delay "
    f"(the 24:00 variant assumes none and the 22:00 variant two hours; {approved_text}), while the AORC rain to midnight is {AORC_WORDING} ({HINDSIGHT_WORDING}, a bound measured for the daily total only); "
    f"the within-day columns of wqaWH_s and wqaWHF_s have no such measurement. Every number of the new sets is a {NUMBER_WORD.get(n_s, n_s)}-seed number; the stored sets have {' or '.join(NUMBER_WORD.get(c, str(c)) for c in sorted({len(v['seeds']) for v in STORED.values()}))} seeds; the split has {n_on} test-year onsets "
    "and its base is weaker than the main-split wqa.")
outcome_full = outcome + " " + outcome_caveat

# ---- status, integrity, the file, the figure --------------------------------------------------------------------------------------------------------------
review = json.loads(REVIEW_FILE.read_text(encoding="utf-8")) if REVIEW_FILE.exists() and not F.QUICK else None       # the independent review record: read, never written here (as scripts 20 and 21 do)
rounds = sorted(int(k[6:]) for k in (review or {}) if re.fullmatch(r"round_\d+", k))      # 2026-10-30: the latest round of the record speaks (round_N with the highest N; the top level is round 1)
rv = review[f"round_{rounds[-1]}"] if rounds else review
trainings_so_far = [p for p in passes if p.get("kind") in ("training run", "extension")]
last_training = trainings_so_far[-1] if trainings_so_far else this_pass
rev_hash = (rv or {}).get("record_script_sha256") or ((rv or {}).get("files_reviewed") or {}).get(f"code/{SCRIPT}")      # the script whose networks the review checked
review_covers = bool(review) and RESCORE and rev_hash in (None, last_training.get("script_sha256"))      # a review covers the saved sets it was made on: a pass that trains makes sets no review has seen, and a record made on another script's networks is stale
# 2026-10-30: a rescore is covered only while the stored comparison sets are the files the review recomputed its numbers with. A stored-set file that changed since the earlier
# results file (seeds 3 and 4 of scripts 16 and 18 were added on 30 October 2026), or a change an earlier pass found under the same review record, means the review predates this pass's numbers.
STORED_NOW = {rel(REAL_PRED / f"lstm_{n}{s}.npz"): sha(REAL_PRED / f"lstm_{n}{s}.npz") for n in STORED for s in [""] + [f"_s{i}" for i in STORED[n]["seeds"]]}
prev_stored = ((old or {}).get("integrity") or {}).get("stored_sets_read") or {}
prev_ir = ((old or {}).get("provenance") or {}).get("independent_review")
same_record = bool(review) and isinstance(prev_ir, dict) and prev_ir.get("record_sha256") == sha(REVIEW_FILE)
n_stored_seeds = lambda files, n: sum(1 for p in files if re.fullmatch(rf"results/preds/lstm_{n}_s\d+\.npz", p))
stored_changed = sorted(set((prev_ir.get("stored_sets_changed_since_the_review") or []) if same_record else [])
                        | ({p for p in set(STORED_NOW) | set(prev_stored) if STORED_NOW.get(p) != prev_stored.get(p)} if prev_stored else set()))
stored_seed_change = {**((prev_ir.get("stored_seed_counts_changed_since_the_review") or {}) if same_record else {}),
                      **({n: [n_stored_seeds(prev_stored, n), n_stored_seeds(STORED_NOW, n)] for n in STORED if n_stored_seeds(prev_stored, n) != n_stored_seeds(STORED_NOW, n)} if prev_stored else {})}
if rounds:      # the latest round covers this rescore while the stored sets are the files it recomputed its numbers with (its stored_sets_reviewed) and no training happened since the script it reviewed
    ssr = rv.get("stored_sets_reviewed") or {}
    stored_changed = sorted(p for p in set(STORED_NOW) | set(ssr) if STORED_NOW.get(p) != ssr.get(p))
    stored_seed_change = {n: [n_stored_seeds(ssr, n), n_stored_seeds(STORED_NOW, n)] for n in STORED if n_stored_seeds(ssr, n) != n_stored_seeds(STORED_NOW, n)}
    rev_written = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", str((rv.get("dates") or {}).get("results_file_written", "")))
    review_covers = bool(review) and RESCORE and (rev_hash in (None, last_training.get("script_sha256"))       # the reviewed script is the last training's, or a later version with no training pass after the file it reviewed
                                                  or (bool(rev_written) and all(str(p.get("finished_local_time", "9")) <= rev_written[0] for p in trainings_so_far)))
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
                       (f"{rel(REVIEW_FILE)} ({review_date}) recomputed this file's numbers with the stored comparison sets as they were then; {len(stored_changed)} of their files have changed or are new since"
                        + (" (seeds " + ", ".join(f"lstm_{n} {a} -> {b}" for n, (a, b) in stored_seed_change.items()) + ")" if stored_seed_change else "")
                        + ": the review predates these numbers and a new review is needed before any number is quoted") if stored_changed else
                       f"{rel(REVIEW_FILE)} ({review_date}) was made on the networks of another version of this script (sha256 {rev_hash}), not on those of the last training pass ({last_training.get('script_sha256')}): "
                       "a new review is needed before any number is quoted")
else:
    n_must = n_should = n_note = review_date = None
    cnt = {}
    review_line = "this file's independent review is not yet done"
if F.QUICK:
    status = f"SMOKE TEST (FLOOD_QUICK=1{', ' + str(n_b) + ' basins' if n_b != 12 else ''}): one seed, 40 updates, results/quick: nothing in this file is a result"
elif review_covers:
    status = f"REVIEWED{f' (round {rounds[-1]})' if rounds else ''}: {review_line}. The team's own checks are not yet done."
elif review:
    status = f"NOT YET REVIEWED for this pass: {review_line}; every number of this file is preliminary and must carry that label in any document (HANDOFF.md section 7)"
else:
    status = ("NOT YET REVIEWED: no independent review record (results/review_flow_state.json) exists for this experiment; every number of this file is preliminary and must carry that label in any document "
              "(HANDOFF.md section 7: each new script gets its own independent review that recomputes the headline numbers from results/preds)")
SAVED = {rel(f): sha(f) for n in SETS for f in [F.PRED / f"lstm_{n}.npz"] + [p for s in SEEDS for p in (F.PRED / f"lstm_{n}_s{s}.npz", MODELS / f"lstm_{n}_s{s}.pt")]}
made_now = {rel(F.PRED / f"lstm_{n}_s{s}.npz") for n, s in trained_pairs} | {rel(MODELS / f"lstm_{n}_s{s}.pt") for n, s in trained_pairs} | {rel(F.PRED / f"lstm_{n}.npz") for n in SETS if trained_now[n]}
earlier = (old or {}).get("integrity", {}).get("sha256_of_saved_files", {})
compared = [f for f in SAVED if f not in made_now and f in earlier]
assert all(earlier[f] == SAVED[f] for f in compared), "a saved file changed since the earlier results file recorded its hash: " + ", ".join(f for f in compared if earlier[f] != SAVED[f])
INTEGRITY = dict(how_to_read="sha256 of every prediction set and network file of this script as they are when this file is written; made_in_this_pass lists the files this pass wrote, every other file was only read and its hash compared "
                             "with the earlier results file; stored_sets_read: the files of the stored comparison sets (never written by this script)", script_sha256=HASH_NOW, sha256_of_saved_files=SAVED, made_in_this_pass=sorted(made_now),
                 files_compared_with_the_earlier_results_file=len(compared), stored_sets_read={rel(REAL_PRED / f"lstm_{n}{s}.npz"): sha(REAL_PRED / f"lstm_{n}{s}.npz") for n in STORED for s in [""] + [f"_s{i}" for i in STORED[n]["seeds"]]},
                 hourly_files_read={g: (sha(HOURLY / f"usgs_iv_{g}.npz") or "assembled in memory from the monthly files") for g in ids})
choices = [(f"The sets, the inputs, the stopping rule and the success rule ({SUCCESS_RULE.split(':')[0]}) were fixed in NEXT_EXPERIMENTS_2026-10-28.md (section E4e, which names no seed count) and in this script before the first "
            f"training (its sha256 is recorded at the start of every pass: {train_pass.get('script_sha256', HASH_NOW)} for the training pass, passes[0].script_sha256); the {NUMBER_WORD.get(n_tp, n_tp)} seeds were chosen at launch "
            f"(--seeds {n_tp}; the script's default is {NUMBER_WORD[ap.get_default('seeds')]}), " + ("in one training pass before any test score was seen" if one_pass else "see the extension clause") + "; the test years were scored once"
            + (f"; with the first three seeds the success rule would be {'met' if E4E['met_with_first_three_seeds'] else 'not met'} (e4e_success.met_with_first_three_seeds"
               + ("; the verdict depends on the seed count" if E4E["met"] != E4E["met_with_first_three_seeds"] else "") + ")" if n_tp > 3 else "") + ".")]
if EXTEND and old and any(s >= 3 for n, s in trained_pairs):
    choices.append(f"Seeds {sorted({s for n, s in trained_pairs if s >= 3})} were added with --extend on {this_pass['finished_local_time'][:10]} after the test-year scores of the earlier pass had been seen.")
added5 = [f"lstm_{n}" for n in STORED if len(STORED[n]["seeds"]) == 5]
if added5:      # review round 2 (2026-10-30), must-fix: the stored comparison sets gained seeds 3 and 4 after this file's test scores were known
    bs, bv = max(by1("wqa_s"), key=lambda t: t[1])
    base5, gain5 = 100 * E4E["onset_warned"]["wqa_s"], 100 * (E4E["onset_warned"]["wqaH_s"] - E4E["onset_warned"]["wqa_s"])
    choices.append(f"The stored comparison sets {', '.join(added5[:-1]) + ' and ' + added5[-1] if len(added5) > 1 else added5[0]} gained seeds 3 and 4 on 30 October 2026 on a second machine (script 18, see its SEED_NOTE), "
                   f"after this file's three-seed comparison was known (+6.6 points on 29 October, wqa_s 40.9%); the five-seed base is {'lower' if base5 < 40.9 else 'higher'} ({base5:.1f}% against 40.9%), "
                   f"so the gain is now {gain5:+.1f} points with wqaH_s unchanged; the best wqa_s seed, above which every wqaH_s seed must lie, is seed {bs} ({100 * bv:.1f}%), "
                   + ("an original seed, so the success rule's verdict is the same with the three-seed base." if bs < 3 else "an added seed: with the three-seed base the verdict could differ."))
notes += (["Smoke test (FLOOD_QUICK=1): one seed, 40 updates, outputs in results/quick and figures/quick; the stored sets are the real ones of results/preds, the new sets are not results."] if F.QUICK else []) + [
    "Hourly flow: cfs -> mm/day with the area code/camels.py uses for the daily flow (the CAMELS forcing-file header, D[g].attrs['area_km2']), so the hourly flow sits on the scale of q_mm and Q95; area_gages2 differs from it by up to a few "
    "percent and is recorded beside it; the local-day means of the hourly flow reproduce the CAMELS daily means (data_coverage: corr and ratio per basin).",
    f"Missing hours are never filled with zero: a missing hour carries the last known value for at most {CARRY_H} hours (hourly_missing_hours); a day not wholly known after that falls back to the daily mean (hourly_known = 0): "
    f"{N_FB[24]} basin-days in the 24:00 variant, {N_FB[22]} in the 22:00 variant (data_coverage per basin). The quality columns are not inputs. aorc_maxcell1h was filled with aorc_max1h on {N_FILL} basin-days wholly filled from ERA5.",
    "The stored sets lstm_wqa_s, lstm_wqaF_s and lstm_wqaP_s were rescored by this script's code from results/preds (checks_against_stored_results); their Mac-trained networks were not reloaded here.",
    f"Delivery assumptions (review of 2026-10-29, must-fix 1): the 24:00 variant reads the hour that ends at the issue time, so it assumes zero delivery delay, and every hourly value used is from the approved record (approved share "
    f"{'1.0 in every gauge' if appr_min >= 1 else 'at least ' + p1(appr_min) + ' in every gauge'}, data_coverage): a real-time service gets provisional values, revised later, after the gauge's telemetry delay (the review: hourly for most USGS stations, four-hourly for some); "
    "the 22:00 variant assumes a delay of at most two hours and is the conservative one of the two only against such a delay; neither covers the provisional-to-approved revision.",
    "Quoting. Any document must quote outcome_full (outcome plus outcome_caveat), never outcome alone; every number is a seed-averaged number unless it says 'single trainings'; the status field says whether the experiment has been reviewed."]
NOT_DONE = ([] if review_covers else ["an independent review of this pass that recomputes the headline numbers from results/preds" + (f" ({rel(REVIEW_FILE)} does not cover it)" if review else f" ({rel(REVIEW_FILE)})")]) \
    + ["the main-split variant (a) of the work order (flag column, 1980-2000 kept)", "block-bootstrap intervals over target-day blocks (13_uncertainty.py pattern)"] \
    + ([f"five seeds (--seeds 5; this run has {NUMBER_WORD.get(len(SEEDS), len(SEEDS))})"] if len(SEEDS) < 5 else []) \
    + ["the simple-model yardstick given the hourly-flow columns",
       "a test with provisional (not approved) gauge values and the measured telemetry delay of each gauge: the 24:00 variant assumes zero delivery delay and approved values, the 22:00 variant a delay of at most two hours; "
       "neither covers the provisional-to-approved revision",
       f"a Stage IV test of the within-day AORC columns of wqaWH_s and wqaWHF_s: the hindsight bound ({HINDSIGHT_WORDING}) was measured for the daily total only (the E1b review); the hindsight of the within-day columns "
       "is not measured (should-fix 2 of results/review_within_day_inputs.json, still open)"]
PROVENANCE = dict(
    written_and_run_by="Claude Code (Fable 5.1) wrote and ran this script under the team lead's direction; AI agents reviewed it",
    independent_review=(dict(record=rel(REVIEW_FILE), record_sha256=sha(REVIEW_FILE), who=rv.get("who"), date=review_date, verdict=rv.get("verdict"), findings=len(rv.get("findings", [])),
                             **({"round": f"round_{rounds[-1]}", "record_script_sha256": rv.get("record_script_sha256")} if rounds else {}),
                             must_fix=n_must, should_fix=n_should, notes=n_note, numerical_mismatches=cnt.get("numerical_mismatches"), values_compared=cnt.get("values_compared"), covers_this_pass=review_covers,
                             reviewed_script_sha256=rev_hash, stored_sets_changed_since_the_review=stored_changed, stored_seed_counts_changed_since_the_review=stored_seed_change,
                             note=((f"round {rounds[-1]} of the review recomputed this file's numbers with the stored comparison sets as they are now (its stored_sets_reviewed equals integrity.stored_sets_read), with no training pass after the results file it reviewed; "
                                   f"this {this_pass['kind']} applies its must-fix wording (the choices entry on seeds 3 and 4 of the stored sets) and changes no number") if review_covers and rounds else
                                   (f"the review was made on the training pass finished {train_pass['finished_local_time']} (script sha256 {train_pass.get('script_sha256')}); this {this_pass['kind']} applies its wording findings "
                                    "(the gauge-record clause of the caveat and of the figure footer: provisional values, a delivery delay of zero (24:00) or two hours (22:00), the approved record used; the false share on every bar "
                                    "label and the GO line of the figure; the seed-count disclosure with met_with_first_three_seeds; the delivery reading inside the seed spread and the seed-rule reading of wqaWHF_s against wqaF_s "
                                    "with the paired onsets; the hindsight bound measured for the daily total only; what_was_not_done; seeds_paired and how_to_read in readings; the figure header; the docstring's mismatch "
                                    "example; this status) and changes no number")
                                   if review_covers else "the review recomputed this file's numbers with the stored comparison sets as they were when it was made; those sets have changed since (stored_sets_changed_since_the_review), so it does not cover this pass" if (RESCORE and stored_changed) else "the review record does not cover the networks of the last training pass (review_line says why)"))
                        if review else "not yet done: no results/review_flow_state.json exists; HANDOFF.md section 7 asks for one that recomputes the headline numbers from results/preds"),
    review_line=review_line, team_check="The team's own checks are not yet done (team.json)")
R = dict(
    question="E4e: does the state of the river at the issue time (hourly USGS gauge flow: the flow at local midnight relative to Q95, its 6- and 12-hour change, the day's peak hour and the hours above Q95) make the all-river LSTM "
             "warn of more flood onsets 1 day ahead on the satellite-era split, alone, with the within-day AORC rain, and with the GEFS forecast; and how much is lost when the record stops two hours before the issue time?",
    script=SCRIPT, written=time.strftime("%Y-%m-%d %H:%M"), status=status, provenance=PROVENANCE, quick_mode=F.QUICK, split=SAT_SPLIT, split_text=SPLIT_TEXT, n_basins=n_b, basins=ids, outcome=outcome, outcome_caveat=outcome_caveat,
    outcome_full=outcome_full,
    design=dict(what="one LSTM (flood.LSTMNet, hidden 64) for all basins with the river gauge (log1p daily flow) and the basin properties (flood.STATIC), the settings of results/lstm_all.json; every set differs from the stored wqa_s only in the "
                     "daily input columns listed; the trainer stops on the mean validation NSE of this split; tables from 2000-01-01 as script 18 builds them; the test years 2010-2014 are scored once",
                sets={n: dict(saved_as=f"lstm_{n}", role=v["role"], what=v["what"], extra_columns=list(v["cols"]), last_available_hour=v["hour"], base=f"lstm_{v['base']}", dyn=list(v["dyn"]), n_in=v["n_in"], seeds=SEEDS, n_params=costs[n][SEEDS[0]]["n_params"]) for n, v in SETS.items()},
                stored_sets={n: dict(saved_as=f"lstm_{n}", script="18_forecast_input.py", role=v["role"], what=v["what"], dyn=list(v["dyn"]), seeds=v["seeds"], rescored_here=True) for n, v in STORED.items()},
                columns={"q_end": "log1p(q_last of the hour ending 24:00 local / Q95)", "dq6": "log1p(q_end / Q95) - log1p(q at 18:00 / Q95)", "dq12": "likewise from 12:00", "q_dmax": "log1p(highest hourly mean flow of day t / Q95)",
                         "hours_above": "hours of day t with the hourly mean flow above Q95 (0-24; 23 or 25 where the clocks change)", "*_22": "the same five with the last available hour ending 22:00 local (dq6 from 16:00, dq12 from 10:00)",
                         "fallback": f"a missing hour carries the last known value for at most {CARRY_H} hours; a day not wholly known after that: q_end = log1p(q_mm / Q95), dq6 = dq12 = 0, q_dmax = q_end, hours_above = last hour x (q_mm > Q95)",
                         "within_day": "the six columns of code/subdaily_inputs.py (hours ending at or before 24:00 of day t; aorc_maxcell1h filled with aorc_max1h where NaN)", "fc": "fc1..fc3 = fc_bc_1..3 of data/gefs/basins (script 18's wqaF_s columns)",
                         "units": "mm/day over the basin, cfs x 0.0283168 x 86400 / (area_km2 x 1e6) x 1000 (code/camels.py); Q95 in mm/day; the engine standardizes every column by the training-years mean and spread"},
                thresholds_source="flood.thresholds(D, split=SAT_SPLIT): the 95th percentile of the daily flow of the training years 2000-10-01..2008-09-30 of this split, per basin",
                hourly_source="code/fetch_usgs_iv.py (USGS Water Data OGC API, parameter 00060, approved 15-minute values, hour-ending UTC stamps)",
                delivery_assumptions="the 24:00 variant assumes zero delivery delay (it reads the hour that ends at the issue time) and approved values (every hourly value used is from the approved record); the 22:00 variant assumes a "
                                     "delivery delay of at most two hours; a real-time service gets provisional values after the gauge's telemetry delay, and neither variant covers the provisional-to-approved revision (review of 2026-10-29, must-fix 1)",
                base="lstm_wqa_s", forecast_base="lstm_wqaF_s", ceiling="lstm_wqaP_s",
                success_rule=SUCCESS_RULE, order_of_training=list(SETS), settings_source="results/lstm_all.json, config"),
    thresholds=thr, data_coverage=dict(how_to_read="per basin: the hourly source, time zone, areas, Q95, first and last hour, the gaps after 2000-01-01 (hours without 15-minute values before any carry), the approved share, the local-day mean of the hourly flow "
                                                   "against the CAMELS daily mean, the hours carried and the days that fell back to the daily mean (both variants; also in the test years)", per_basin=COVERAGE, fallback_days_total=N_FB),
    config=conf, config_source="results/lstm_all.json, config" + (" (updates and eval_every reduced for the smoke test)" if F.QUICK else ""), seeds=SEEDS, n_seeds_per_set={n: ranges[n]["l1"]["n_seeds"] for n in SHOWN},
    summary_test=dict(how_to_read="test years 2010-2014 of the split, seed average of each set, trigger 1.0, per lead; onset and all = warning tables pooled over the basins; the stored sets rescored here", sets=summary_test),
    summary_val=dict(label="USED FOR NO CHOICE: validation years 2008-2010, for reference only; the trainer stopped each training on its own validation score, and the validation years choose only the triggers of operating_points", sets=summary_val),
    per_basin=per_basin, seed_ranges=dict(how_to_read="lowest and highest single-seed value of the test-year scores, per set and lead (by_seed: the single values, in the order of the seeds)", sets=ranges),
    readings=dict(how_to_read=reading_how(), pairs=readings),
    delivery=delivery, closure=dict(how_to_read="the share of the gap from the base to lstm_wqaP_s (the perfect rain forecast in the slots) that each set closes, per lead, in median NSE and in onsets warned", sets=closure),
    operating_points=OPS_NOTE, e4e_success=E4E, causality=causality, reload_check=reload_check, checks_against_stored_results=stored_check, models=models, training=training, integrity=INTEGRITY,
    choices_made_after_looking_at_test_scores=choices, n_trainings=sum(p["n_trainings"] for p in passes), training_seconds=round(sum(p["training_seconds"] for p in passes), 1), waited_for_cpu_s=round(waited, 1), passes=passes,
    runtime_s=round(time.time() - t_start, 1), script_sha256=HASH_NOW, notes=notes, what_was_not_done=NOT_DONE)


def clean(o, count):
    """Strict JSON: numpy values become Python ones; a number that is not finite becomes null (count[0] is raised for each)."""
    if isinstance(o, dict):
        return {str(k): clean(v, count) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v, count) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (np.floating, float)):
        count[0] += not math.isfinite(float(o))
        return float(o) if math.isfinite(float(o)) else None
    if o is None:                                           # a null read back from the earlier results file (a rescore keeps the training records as they were written): counted too
        count[0] += 1
        return None
    return bool(o) if isinstance(o, np.bool_) else int(o) if isinstance(o, np.integer) else str(o.date()) if isinstance(o, pd.Timestamp) else o


nulls = [0]
clean(R, nulls)
notes.append(f"{nulls[0]} values in this file are not numbers (NaN: shares whose denominator is zero, undefined p-values, the load average Windows does not have); they are written as null, which a reader must treat as 'no value'.")
R["runtime_s"] = round(time.time() - t_start, 1)
JSON_TEXT = json.dumps(clean(R, [0]), indent=1, allow_nan=False)
RESULTS_FILE.write_text(JSON_TEXT, encoding="utf-8")
draw(json.loads(JSON_TEXT))
print(f"\n{outcome_full}\n\nstatus: {status}\nwrote {RESULTS_FILE} and {FIG_FILE} ({R['runtime_s']} s; {R['n_trainings']} trainings in all, {this_pass['n_trainings']} in this pass)", flush=True)
