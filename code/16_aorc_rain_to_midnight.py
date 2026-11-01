"""E1b: the rain of the whole issue day up to local midnight, from hourly AORC rain (a gauge- and radar-anchored analysis on a 1 km grid) instead of ERA5.

Experiment E1 (15_rain_to_midnight.py) gave the all-river LSTM the rain of the whole local day t, midnight to midnight, summed from hourly ERA5 rain - model output, not observed rain - and found that the
share of onsets warned 1 day ahead goes from 19% to 30%; its matched-window controls showed that the gain comes from the hours after about 09:00, and NLDAS rain, a gauge-based analysis whose day ends in the
evening, reached the same 30%, so the quality of the rain product counted as well. This script repeats E1 with NOAA's AORC v1.1 (hourly rain on a 1 km grid, anchored on rain gauges and radar; code/fetch_aorc.py
reduces it to one hourly series per basin, the mean of the 1 km cells inside the basin outline, in data/aorc/) and asks (ROADMAP.md, step 1b) whether the E1 gain holds or grows with that rain, and whether a rain
table with the right clock beats the mis-stamped one (Daymet) on its own.
 1. builds from the AORC hours (the source has holes: hours in which no cell of a basin has a value are filled from the cached ERA5 hourly rain, flagged and counted by code/fetch_aorc.py) the daily input aorc_to_midnight (mm/day: the rain of the wall-clock day t at the gauge, summer time included: every hour goes to the local calendar day in which it ends, so a
    day has 24 hours, or 23 or 25 where the clocks change) and proves, given the stamp convention of the stores (note 'Hour stamps'), its clock exactly as 15_rain_to_midnight.py does: no hour after the
    issue time is summed (checked for every hour by the wall clock and in UTC time, by a second route from the time-zone rules, and by replacing the later hours with made-up values). The same code and the same
    proof build two more daily series from the same hours, for the controls: aorc_to_09 (the wall-clock window from 09:00 of day t-1 to 09:00 of day t: the window of the rain table itself, as measured, rounded
    to 09:00) and aorc_utc_day (the UTC day of day t, which ends at 00:00 UTC of day t+1: 19:00 local standard time in the Eastern basins, 18:00 in Leaf River);
 2. trains the all-basin LSTM with the river gauge, three seeds each, with the settings chosen by 04_lstm_all.py (results/lstm_all.json): wqa (the weather of the rain table + aorc_to_midnight as one extra input),
    wqa_to09 (+ aorc_to_09: the same AORC product over the window of the rain table, no new hour: the control), wqa_utc (+ aorc_utc_day: what a rain product served by UTC day, or IMERG-Early with its
    delay of about 4 hours, would hold at the issue time) and wqa_rep (= wqa_daymet_replaced: the Daymet rain column REPLACED by aorc_to_midnight, no extra column: does a rain table with the right clock beat
    the mis-stamped table on its own?);
 3. saves every trained network (results/models/lstm_<set>_s<seed>.pt), reloads one per input set from its file alone and checks that it gives the stored forecasts again, bit for bit;
 4. scores them on the test years (the validation years are shown apart and chose nothing) against the ladder of information 1 day ahead - rain table, NLDAS rain, ERA5 to midnight (script 15), AORC to midnight,
    AORC replacing the rain table, NLDAS + ERA5 together (script 15), perfect forecast - with the checks of report section 4.2, the share of the gap to the perfect forecast that is closed and the
    GO / CAUTION / NOT YET verdict of results/flood_analysis.json; reads the controls as the worth of the hours 09:00 to 24:00 of day t (value_of_hours_09_to_24) and of the evening hours beyond the UTC day
    (value_of_evening_hours_beyond_utc_day), and sets AORC against ERA5 (aorc_against_era5: both over the same hours, 00:00 to 24:00) and against NLDAS (aorc_against_nldas);
 5. gives ridge regression and the random forest the new input (wqa) by the recipe of 02_baselines.py (code/features.py; the ridge strength, forest settings and seed that results/baselines.json records for the
    wq input set), saves them as base_linreg_wqa and base_rf_wqa and reads them beside lstm_wqa as 04_lstm_all.py and 11_key_figure.py read the LSTM against the simple models (rule 2 of the work order).
Reads data/aorc (code/fetch_aorc.py makes it; this script calls it when the series are missing: 4.5 GB, about an hour with 16 connections; it needs the zstd command-line tool, which is not a pip package: brew install zstd), the settings and prediction sets of 04_lstm_all.py (lstm_wq, lstm_wqf), those of
10_rain_timing.py (lstm_wqn and its control), those of 15_rain_to_midnight.py (lstm_wqh and its three controls, lstm_wqnh, base_linreg_wqh, base_rf_wqh), the settings and forecasts of 02_baselines.py
(results/baselines.json, base_*_wq), results/day_windows.json, the ERA5 files of check_day_windows.py (data/hourly: for the checks against ERA5), the NLDAS rain shipped with CAMELS, the IMERG files of
fetch_satellite.py (data/satellite), results/flood_analysis.json, results/satellite_data.json, results/aorc_data.json and, when they exist, results/review_aorc_rain_to_midnight.json, results/aorc_hindsight.json (code/check_aorc_hindsight.py) and results/stage4_realtime.json (code/check_stage4_realtime.py): the numbers of these three files are quoted in the caveat of the outcome and in the notes.
Writes results/aorc_rain_to_midnight.json, figures/aorc_rain_to_midnight.png, results/models/*.pt and the prediction sets lstm_wqa, lstm_wqa_to09, lstm_wqa_utc, lstm_wqa_rep (+ one per seed each) and
base_linreg_wqa, base_rf_wqa.
Without training again: python code/16_aorc_rain_to_midnight.py --rescore  recomputes every score, table, note, the outcome sentence and the figure from the saved prediction sets (the training records are
carried over from the JSON; the reload check is run again on the saved networks).
Training only what is missing: python code/16_aorc_rain_to_midnight.py --extend  behaves like --rescore for every set whose seed files and network files exist and trains the others (and the two simple
models, when their forecasts are missing); the plain run trains everything from scratch.
FLOOD_QUICK=1 (smoke test: one seed, 40 updates, results/quick and figures/quick) uses the AORC series of data/aorc when they are there and otherwise stops with a message.
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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
from camels import load_basin
from common import *
from features import feature_table

use_style()
SCRIPT = "16_aorc_rain_to_midnight.py"
OUT = "aorc_rain_to_midnight"                              # results/<OUT>.json, figures/<OUT>.png
t_start = time.time()
HASH_NOW = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()      # the code that runs, hashed once at the start: every pass records it
RESCORE = "--rescore" in sys.argv                          # no training: scores, notes and figure again from the saved prediction sets
EXTEND = "--extend" in sys.argv                            # as --rescore, but trains the sets (and simple models) whose files are missing
assert not (RESCORE and EXTEND), "--rescore trains nothing and --extend trains what is missing: use one of the two"
LOAD = RESCORE or EXTEND                                   # the saved sets and the training records of the results file are read, not made again
D, B = F.load_data(), F.basins()
ids, thr = list(D), F.thresholds(D)
n_b = len(ids)
SEEDS = [0] if F.QUICK else [0, 1, 2, 3, 4]        # seeds 3-4 added 30 October 2026 on the Windows PC: see choices_made_after_looking_at_test_scores (SEED_NOTE)
SEED_NOTE = ("Seeds 3 and 4 of every set of this script were added on 30 October 2026 on a second machine (Windows 11 Home, Intel Core i7-10750H, CPU only, Python 3.11.15, torch 2.2.2+cpu, numpy 1.26.4); "
             "seeds 0-2 were trained on the first machine (macOS, Intel Core i5) and were kept, not retrained. They were added after the test-year scores of seeds 0-2 and the independent review of them were known. "
             "Nothing else changed: the same settings (results/lstm_all.json, key config), inputs, split, stopping rule, warning level and success rule. Every seed average (lstm_<set>.npz) and every seed range of this "
             "script now covers five trainings; the control sets of 04_lstm_all.py and 10_rain_timing.py keep their three. The two extra seeds were planned on 14 October 2026 (code/schedule.py, week 2: five trainings per "
             "model where the NSE decides; promised at the proposal talk of 15 October) for every set alike, whatever the direction of its result; the three-seed numbers remain in the independent review records "
             "(round 1 of results/review_aorc_rain_to_midnight.json) and in a copy of the three-seed results file, results/archive/aorc_rain_to_midnight_3seeds_2026-10-30.json (copied unchanged "
             "from the backup of 30 October 2026, 10:41); passes records only the trainings and times of each pass. five_seed_changes lists what changed with five seeds.")        # appended to choices_made_after_looking_at_test_scores (agent8 audit, section 3)
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")        # the training record of one model (Run.summary)
P_CUT = 0.05                                               # 'p is small', as in section 4.2 of the report
N_CUTS = 3                                                 # random days per basin on which the hours after midnight are replaced to test the daily value
HOUR = pd.Timedelta(hours=1)
HOURLY, MODELS = DATA / "hourly", F.RES_OUT / "models"     # the cached ERA5 files (read for the checks against ERA5); one file per trained network
lk = lambda k: f"l{k}"
rel = lambda f: f.relative_to(ROOT).as_posix() if ROOT in f.parents else str(f)                         # a path as the project shows it
notes = []
for what, f in (("results/day_windows.json", F.RES / "day_windows.json"), ("results/flood_analysis.json", F.RES / "flood_analysis.json"), ("results/satellite_data.json", F.RES / "satellite_data.json"),
                ("results/baselines.json", F.RES / "baselines.json"), ("results/rain_timing.json", F.RES / "rain_timing.json")):
    assert f.exists(), f"{what} is missing: this script reads it (check_day_windows.py, 06_flood_analysis.py, fetch_satellite.py, 02_baselines.py and 10_rain_timing.py write them; run_all.sh runs them before this script)"
DW = json.loads((F.RES / "day_windows.json").read_text())             # the real files also in a smoke test (the quick day_windows.json holds two basins)
FA = json.loads((F.RES / "flood_analysis.json").read_text())
SAT = json.loads((F.RES / "satellite_data.json").read_text())
RT = json.loads((F.RES / "rain_timing.json").read_text())             # 10_rain_timing.py: the facts about the NLDAS day that this script relies on (the real file also in a smoke test)
REVIEW_FILE = F.RES / "review_aorc_rain_to_midnight.json"
REVIEW = json.loads(REVIEW_FILE.read_text()) if REVIEW_FILE.exists() else None                 # the independent recomputation by a second agent, when it is there
HIND_FILE, S4RT_FILE = F.RES / "aorc_hindsight.json", F.RES / "stage4_realtime.json"            # the two checks of the hindsight question (check_aorc_hindsight.py, check_stage4_realtime.py), when they have been run
HIND = json.loads(HIND_FILE.read_text()) if HIND_FILE.exists() else None
S4RT = json.loads(S4RT_FILE.read_text()) if S4RT_FILE.exists() else None

# ---- 0. the AORC series (data/aorc): there, or made now by code/fetch_aorc.py ------------------------------------------------------------
if os.environ.get("FLOOD_AORC_DIR"):
    assert F.QUICK, "FLOOD_AORC_DIR (another folder for the AORC series) is a test hook for smoke tests and works only with FLOOD_QUICK=1"
if F.QUICK and not AO.cache_complete(ids):
    print("SKIPPED (FLOOD_QUICK=1): the AORC series of the 12 basins are not (all) in data/aorc. code/fetch_aorc.py downloads and reduces them (4.5 GB from NOAA's open AWS bucket, about an hour with 16 connections through the proxy, and the zstd command-line tool: brew install zstd; "
          "the plain run of this script calls it when the series are missing). Nothing was written.", flush=True)
    sys.exit(0)
AORC_REC = AO.ensure() if not F.QUICK else (json.loads(AO.RECORD.read_text()) if AO.RECORD.exists() else None)       # the record of the download and of the checks of the series (results/aorc_data.json)
if AORC_REC is None:
    notes.append("results/aorc_data.json was not found (smoke test): the record of the download and of the checks of the series is not quoted.")
# ---- 1. the rain of the whole local day (the wall-clock day of the gauge, summer time included), from hourly AORC --------------------
UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}       # local STANDARD time minus UTC in winter, in hours (day_windows.json): it picks the time zone
assert all(DW["day_ends"][p]["per_basin"][g]["utc_offset_h"] == UTC_OFF[g] for p in ("nldas", "maurer", "imerg") for g in ids) \
    and all(DW["river_flow_day"]["per_gauge"][g]["utc_offset_standard_time_h"] == UTC_OFF[g] for g in ids), "the time zones in day_windows.json disagree"
ZONE = {-5: "America/New_York", -6: "America/Chicago"}                   # the time zone of the basins with that winter offset; zoneinfo holds its summer-time rules since 1980
assert set(UTC_OFF.values()) <= set(ZONE), "a basin has a standard-time offset without a time zone here"
try:
    assert all(pd.Timestamp("2000-01-15 12:00", tz=ZoneInfo(z)).utcoffset() == o * HOUR for o, z in ZONE.items()), "a time zone is not at its standard time in winter"
except ZoneInfoNotFoundError as e:                                       # on Windows, pandas brings the database with it (the tzdata package)
    raise SystemExit(f"Python's zoneinfo cannot find the time-zone database ({e}); it is needed for the wall-clock day (pip install tzdata)")
DAYS = pd.date_range(F.PERIOD[0], F.PERIOD[1])
by_off = {o: [F.SHORT[g] for g in ids if UTC_OFF[g] == o] for o in sorted(set(UTC_OFF.values()), reverse=True)}
major = max(by_off, key=lambda o: len(by_off[o]))
zone_text = "; ".join(f"{ZONE[o]} for {', '.join(b) if o != major else 'all other basins'} (UTC{o:+d} in winter)" for o, b in sorted(by_off.items(), key=lambda t: t[0] == major))
clock_hm = lambda td: f"{int(td.total_seconds() // 3600):02d}:{int(td.total_seconds() % 3600 // 60):02d}"          # a time of day from a timedelta



def aorc_hours(g):
    """(time stamps, rain of the hour ending at each stamp in mm, number of 1 km cells in the basin mask, number of hours filled from ERA5): the cached AORC series of one basin (code/fetch_aorc.py), UTC.

    The stamp is taken as the END of the hour (note 'Hour stamps' gives the evidence and the sensitivity): the value at time T is the rain of the hour that ENDS at T, and an hour starts one hour before its stamp, as for the
    ERA5 files of script 15 (so every function below that takes hours takes them in this form). Hours in which AORC holds no value in any cell of the basin were filled by code/fetch_aorc.py (flagged); they are counted here."""
    stamp, x = AO.load_series(g)
    z = np.load(AO.basin_file(g), allow_pickle=False)
    assert len(stamp) == len(x) == AO.N_HOURS and stamp[0] == AO.FIRST_STAMP and stamp[-1] == AO.LAST_STAMP, f"aorc_{g}.npz does not hold the span {AO.SPAN_FOR_THE_RECORD}"
    assert ((stamp[1:] - stamp[:-1]) == HOUR).all(), f"aorc_{g}.npz: the hourly time axis has gaps or repeats"
    fl = z["filled"].astype(bool)
    assert np.isfinite(x).all() and (x >= 0).all() and (np.nan_to_num(z["min_cell_mm"], nan=0.0) >= 0).all(), f"aorc_{g}.npz: missing or negative hourly rain"        # min_cell_mm is NaN in the filled hours
    assert ((z["n_valid"].astype(int) > 0) | fl).all(), f"aorc_{g}.npz: an hour in which no cell of the basin had a value and that is not flagged as filled"
    return stamp, x, int(z["n_cells"]), int(fl.sum())


def era5_local_day(g):
    """The cached ERA5 hourly rain of one basin (check_day_windows.py; the value at time T is the rain of the hour ending at T, as in script 15) summed over the wall-clock days: only for the comparison with AORC."""
    j = json.loads((HOURLY / f"era5_{g}.json").read_text())
    assert j["timezone"] == "GMT" and j["utc_offset_seconds"] == 0 and j["hourly_units"]["precipitation"] == "mm", f"era5_{g}.json is not hourly rain in mm on a UTC axis"
    stamp, x = pd.DatetimeIndex(pd.to_datetime(j["hourly"]["time"])), np.array(j["hourly"]["precipitation"], float)
    return stamp, x


def wall_clock(t_utc, zone):
    """UTC times (no time zone attached) -> the time on the wall clock of the zone, summer time included (no time zone attached)."""
    return t_utc.tz_localize("UTC").tz_convert(ZoneInfo(zone)).tz_localize(None)


def rain_by_wall_clock_day(stamp, x, zone):
    """The main route: every hourly value goes to the local calendar day, by the wall clock, in which its hour ENDS (an hour that ends at exactly 24:00 belongs to the day that ends then).

    A day has 24 hours, or 23 or 25 on the days on which the clocks change. Returns the rain of every day and its number of hours."""
    by_day = pd.Series(x, index=(wall_clock(stamp, zone) - pd.Timedelta(minutes=1)).normalize()).groupby(level=0)
    return by_day.sum(), by_day.size()


def day_positions(start, days, zone):
    """The second route: the UTC start and end of every wall-clock day from the time-zone rules, and the positions in the hourly series of the hours between them.

    start is the UTC start of each hour. Day t runs from 00:00 on the wall clock of day t to 00:00 of day t+1, the issue time. Every hour that is summed into a day is PROVED to
    start at or after 00:00 and to end at or before 24:00 of that day, by the wall clock and in absolute (UTC) time, so no hour after the issue time is summed; the days follow each
    other hour by hour, and each hour lies in the same day as the main route puts it."""
    z = ZoneInfo(zone)
    a = days.tz_localize(z).tz_convert("UTC").tz_localize(None)                              # 00:00 on the wall clock at the start of each day, as a UTC time
    b = (days + pd.Timedelta(days=1)).tz_localize(z).tz_convert("UTC").tz_localize(None)       # 24:00 = 00:00 of the next day: the end of each day and the issue time
    assert ((a - start[0]) % HOUR == pd.Timedelta(0)).all() and ((b - start[0]) % HOUR == pd.Timedelta(0)).all(), "a day does not start on a whole hour"
    i0, i1 = np.asarray((a - start[0]) // HOUR), np.asarray((b - start[0]) // HOUR)
    assert i0.min() >= 0 and i1.max() < len(start), "the hourly series does not cover every day of the period"
    assert (i1[:-1] == i0[1:]).all() and set(np.unique(i1 - i0)) <= {23, 24, 25}, "the days do not follow each other hour by hour, or a day has other than 23, 24 or 25 hours"
    assert (start[i0] == a).all() and (start[i1 - 1] + HOUR == b).all(), "the first hour of a day does not start at its 00:00, or the last does not end at its 24:00"
    day = np.repeat(np.arange(len(days)), i1 - i0)                                          # the day each hour is summed into, from the first hour of the first day to the last of the last
    hr = start[i0[0]:i1[-1]]
    sw, ew = wall_clock(hr, zone), wall_clock(hr + HOUR, zone)                              # the wall-clock time at which each summed hour starts and ends
    assert (sw >= days[day]).all() and (ew <= days[day] + pd.Timedelta(days=1)).all(), "an hour summed into a day starts before 00:00 or ends after 24:00 of that day by the wall clock"
    assert (hr >= a[day]).all() and (hr + HOUR <= b[day]).all(), "an hour summed into a day starts before its start or ends after its end in absolute time"
    assert ((ew - pd.Timedelta(minutes=1)).normalize() == days[day]).all(), "the two routes put an hour in different days"
    return i0, i1, a, b, sw, ew




corr = lambda a, b, k: float(pd.concat([a, b.shift(k)], axis=1).dropna().corr().iloc[0, 1])        # corr(a on day d, b on day d-k)
three = lambda a, b: dict(day_before=corr(a, b, 1), same_day=corr(a, b, 0), day_after=corr(a, b, -1))
RAIN, CLOCK, DAYLEN, rng = {}, {}, {}, np.random.default_rng(0)
for g in ids:
    stamp, x, n_cells_g, n_filled_g = aorc_hours(g)
    start, off, zone = stamp - HOUR, UTC_OFF[g], ZONE[UTC_OFF[g]]
    i0, i1, a, b, sw, ew = day_positions(start, DAYS, zone)
    n_h = i1 - i0                                                                           # hours in each day: 24, or 23 and 25 where the clocks change
    by_day, size = rain_by_wall_clock_day(stamp, x, zone)
    s = by_day.reindex(DAYS)
    assert s.index.equals(D[g].index) and len(s) == len(D[g]) and s.notna().all(), "the daily series must cover all of flood.PERIOD without a gap"
    assert (size.reindex(DAYS).values == n_h).all(), "the two routes count different numbers of hours in a day"
    second = np.add.reduceat(x, np.append(i0, i1[-1]))[:-1]                                 # the sum of the hours between the start and the end of each day
    assert np.allclose(second, s.values, rtol=0, atol=1e-6), "the two ways of summing the hours give different daily rain"
    start_wall = wall_clock(start, zone)
    first_changes = [int(np.flatnonzero(n_h == k)[0]) for k in (23, 25) if (n_h == k).any()]                  # the first days on which the clocks changed are tested too
    cuts = sorted({*map(int, rng.integers(1, len(DAYS) - 1, N_CUTS)), *[c for c in first_changes if 0 < c < len(DAYS) - 1]})
    for c in cuts:                                                                          # the rain after 24:00 (wall clock) of day c is made up: no day up to c may change
        s2 = rain_by_wall_clock_day(stamp, np.where(np.asarray(start_wall >= DAYS[c] + pd.Timedelta(days=1)), x + 100.0, x), zone)[0].reindex(DAYS).values
        assert np.array_equal(s2[:c + 1], s.values[:c + 1]), f"{g}: rain after midnight of {DAYS[c].date()} changed a value up to that day"
        assert s2[c + 1] != s.values[c + 1], f"{g}: the test could not see a change of the next day"
    RAIN[g] = s
    at = lambda arr, pos: arr[pos - i0[0]]                                                  # the wall-clock start or end of the hour at a position of the hourly series
    first_start, last_start, last_end = at(sw, i0) - DAYS, at(sw, i1 - 1) - DAYS, at(ew, i1 - 1) - DAYS
    assert (first_start == pd.Timedelta(0)).all() and (last_start == pd.Timedelta(hours=23)).all() and (last_end == pd.Timedelta(days=1)).all(), "a day does not run from 00:00 to 24:00"
    summer = np.asarray((wall_clock(b, zone) - b) > off * HOUR)                             # the wall clock is ahead of standard time at the end of the day, the issue time
    CLOCK[g] = dict(short=F.SHORT[g], utc_offset_h=off, time_zone=zone, n_days=int(len(s)), first_day=str(DAYS[0].date()), last_day=str(DAYS[-1].date()),
                    first_hour_starts_local=clock_hm(first_start.min()), last_hour_starts_local=clock_hm(last_start.max()), last_hour_ends_local=clock_hm(last_end.max()),
                    last_hour_summed_ends_utc_on_last_day=str(start[i1[-1] - 1] + HOUR), aorc_first_stamp_utc=str(stamp[0]), aorc_last_stamp_utc=str(stamp[-1]), n_hours=int(len(x)),
                    made_up_hours_cut_days=[str(DAYS[c].date()) for c in cuts], n_cells_of_the_basin_mask=n_cells_g, n_hours_filled_from_era5=n_filled_g)
    DAYLEN[g] = dict(short=F.SHORT[g], time_zone=zone, days_with_23_hours=int((n_h == 23).sum()), days_with_24_hours=int((n_h == 24).sum()), days_with_25_hours=int((n_h == 25).sum()),
                     first_day_with_23_hours=str(DAYS[n_h == 23][0].date()) if (n_h == 23).any() else None, first_day_with_25_hours=str(DAYS[n_h == 25][0].date()) if (n_h == 25).any() else None,
                     n_days_on_summer_time_at_the_issue_time=int(summer.sum()), share_of_days_on_summer_time_at_the_issue_time=float(summer.mean()))
assert {c["last_hour_ends_local"] for c in CLOCK.values()} == {"24:00"} and {c["first_hour_starts_local"] for c in CLOCK.values()} == {"00:00"}
ERA5_DAY = {}                                                                               # ERA5 over the wall-clock day, for the comparison with AORC (the proofs of its clock are those of script 15)
for g in ids:
    e_stamp, e_x = era5_local_day(g)
    assert e_stamp.equals(aorc_hours(g)[0]), f"era5_{g}.json and the AORC series do not share a time axis"
    ERA5_DAY[g] = rain_by_wall_clock_day(e_stamp, e_x, ZONE[UTC_OFF[g]])[0].reindex(DAYS)
rain_check = {g: dict(short=F.SHORT[g], mean_mm_day=dict(aorc=float(RAIN[g].mean()), daymet=float(D[g].prcp.mean()), era5=float(ERA5_DAY[g].mean())), daymet_against_aorc_local_day=three(D[g].prcp, RAIN[g]),
                      era5_against_aorc_local_day=three(ERA5_DAY[g], RAIN[g])) for g in ids}
rc_med = {c: float(np.median([rain_check[g]["daymet_against_aorc_local_day"][c] for g in ids])) for c in ("day_before", "same_day", "day_after")}
rc_era5 = {c: float(np.median([rain_check[g]["era5_against_aorc_local_day"][c] for g in ids])) for c in ("day_before", "same_day", "day_after")}
rc_ratio = float(np.median([rain_check[g]["mean_mm_day"]["aorc"] / rain_check[g]["mean_mm_day"]["daymet"] for g in ids]))
rc_ratio_era5 = float(np.median([rain_check[g]["mean_mm_day"]["era5"] / rain_check[g]["mean_mm_day"]["daymet"] for g in ids]))
fits_alignment = bool(rc_med["day_after"] < 0.5 * min(rc_med["same_day"], rc_med["day_before"]))        # the two local days that share hours with a Daymet day correlate, the next one does not
n_cells_by_basin = {g: CLOCK[g]["n_cells_of_the_basin_mask"] for g in ids}

# ---- the same hours over two other windows (the controls), and the NLDAS column of 10_rain_timing.py (for the comparison of the daily series) ----------
SPEC = {"aorc_to_midnight": dict(clock="wall", end_h=24), "aorc_to_09": dict(clock="wall", end_h=9), "aorc_utc_day": dict(clock="utc", end_h=24)}      # the window of day t ENDS at end_h o'clock of day t, on the wall clock of the gauge or in UTC, and starts one day earlier
NEW_WINDOWS = ("aorc_to_09", "aorc_utc_day")
H_END = lambda spec: pd.Timedelta(hours=spec["end_h"])


def rain_by_window_day(stamp, x, zone, spec):
    """The main route for any window: every hourly value goes to the day whose window holds its hour. The window of day t ends at spec['end_h'] o'clock of day t, on the wall clock of the gauge
    (summer time included) or, when spec['clock'] is 'utc', in UTC, and starts one day earlier; an hour belongs to the window in which it ENDS (an hour that ends exactly where a window ends is
    its last hour). With end_h 24 on the wall clock this is rain_by_wall_clock_day() above (asserted below for every basin). Returns the rain of every day and its number of hours."""
    end = stamp if spec["clock"] == "utc" else wall_clock(stamp, zone)                       # the time at which each hour ends, on the clock the window is defined on
    by_day = pd.Series(x, index=(end - H_END(spec) - pd.Timedelta(minutes=1)).normalize() + pd.Timedelta(days=1)).groupby(level=0)
    return by_day.sum(), by_day.size()


def window_instants(days, zone, spec):
    """The second route: the UTC start and end of the window of every day, from the time-zone rules (the UTC day of a date for spec['clock'] == 'utc')."""
    if spec["clock"] == "utc":
        return days, days + pd.Timedelta(days=1)
    to_utc = lambda t: t.tz_localize(ZoneInfo(zone)).tz_convert("UTC").tz_localize(None)
    return to_utc(days - pd.Timedelta(days=1) + H_END(spec)), to_utc(days + H_END(spec))


def prove_window(start, zone, spec):
    """Locate the hours of every window in the hourly series (start is the UTC start of each hour) and PROVE for every hour that is summed into day t: it starts at or after the start of the window
    and ends at or before its end, in absolute (UTC) time and on the clock of the window; it ends at or before the issue time of day t (local midnight at the end of day t, as a UTC time, and 24:00 by
    the wall clock): so no hour after the issue time is summed. The windows follow each other hour by hour, have 24 hours (23 or 25 on the wall clock where the clocks change), and the two routes
    put every hour in the same day."""
    a, b = window_instants(DAYS, zone, spec)
    issue = window_instants(DAYS, zone, SPEC["aorc_to_midnight"])[1]                         # the end of day t, the issue time, as a UTC time
    assert ((a - start[0]) % HOUR == pd.Timedelta(0)).all() and ((b - start[0]) % HOUR == pd.Timedelta(0)).all(), "a window does not start or end on a whole hour"
    i0, i1 = np.asarray((a - start[0]) // HOUR), np.asarray((b - start[0]) // HOUR)
    assert i0.min() >= 0 and i1.max() < len(start), "the hourly series does not cover every window of the period"
    assert (i1[:-1] == i0[1:]).all() and set(np.unique(i1 - i0)) <= ({24} if spec["clock"] == "utc" else {23, 24, 25}), "the windows do not follow each other hour by hour, or a window has a wrong number of hours"
    assert (start[i0] == a).all() and (start[i1 - 1] + HOUR == b).all(), "the first hour of a window does not start at its start, or the last does not end at its end"
    day = np.repeat(np.arange(len(DAYS)), i1 - i0)
    hr = start[i0[0]:i1[-1]]
    sw, ew = wall_clock(hr, zone), wall_clock(hr + HOUR, zone)
    assert (hr >= a[day]).all() and (hr + HOUR <= b[day]).all(), "an hour summed into a window starts before its start or ends after its end in absolute time"
    assert (hr + HOUR <= issue[day]).all(), "an hour summed into day t ends after the issue time of day t in absolute time"
    assert (ew <= DAYS[day] + pd.Timedelta(days=1)).all(), "an hour summed into day t ends after 24:00 of day t by the wall clock"
    if spec["clock"] == "wall":
        assert (sw >= DAYS[day] - pd.Timedelta(days=1) + H_END(spec)).all() and (ew <= DAYS[day] + H_END(spec)).all(), "an hour summed into a window starts before its start or ends after its end by the wall clock"
        key = (ew - H_END(spec) - pd.Timedelta(minutes=1)).normalize() + pd.Timedelta(days=1)
    else:
        assert (hr >= DAYS[day]).all() and (hr + HOUR <= DAYS[day] + pd.Timedelta(days=1)).all(), "an hour summed into the UTC day of a date starts before 00:00 UTC of that date or ends after 00:00 UTC of the next date"
        key = (hr + HOUR - pd.Timedelta(minutes=1)).normalize()
    assert (key == DAYS[day]).all(), "the two routes put an hour in different days"
    return i0, i1, a, b, issue, sw, ew


NL = {g: load_basin(g, "nldas").loc[F.PERIOD[0]:F.PERIOD[1], "prcp"].astype(float) for g in ids}             # the NLDAS column of 10_rain_timing.py, read the same way
assert all(NL[g].index.equals(D[g].index) for g in ids), "the NLDAS and the Daymet table must cover the same days (as asserted in 10_rain_timing.py)"
assert not any(NL[g].isna().any() for g in ids), "NLDAS rain has gaps (as asserted in 10_rain_timing.py)"
RAIN2, CLOCK2, LEN2, rng2 = {w: {} for w in NEW_WINDOWS}, {w: {} for w in NEW_WINDOWS}, {w: {} for w in NEW_WINDOWS}, np.random.default_rng(1)
for g in ids:
    stamp, x, _, _ = aorc_hours(g)
    start, off, zone = stamp - HOUR, UTC_OFF[g], ZONE[UTC_OFF[g]]
    assert np.array_equal(rain_by_window_day(stamp, x, zone, SPEC["aorc_to_midnight"])[0].reindex(DAYS).values, RAIN[g].values), f"{g}: the code for any window does not reproduce aorc_to_midnight"
    for w in NEW_WINDOWS:
        spec = SPEC[w]
        i0, i1, a, b, issue, sw, ew = prove_window(start, zone, spec)
        n_h = i1 - i0
        by_day, size = rain_by_window_day(stamp, x, zone, spec)
        s = by_day.reindex(DAYS)
        assert s.index.equals(D[g].index) and len(s) == len(D[g]) and s.notna().all(), f"{w}: the daily series must cover all of flood.PERIOD without a gap"
        assert (size.reindex(DAYS).values == n_h).all(), f"{w}: the two routes count different numbers of hours in a window"
        second = np.add.reduceat(x, np.append(i0, i1[-1]))[:-1]                             # the sum of the hours between the start and the end of each window
        assert np.allclose(second, s.values, rtol=0, atol=1e-6), f"{w}: the two ways of summing the hours give different daily rain"
        first_changes = [int(np.flatnonzero(n_h == k)[0]) for k in (23, 25) if (n_h == k).any()]
        cuts = sorted({*map(int, rng2.integers(1, len(DAYS) - 1, N_CUTS)), *[c for c in first_changes if 0 < c < len(DAYS) - 1]})
        for c in cuts:                                                                      # the rain after the end of the window of day c is made up: no day up to c may change
            s2 = rain_by_window_day(stamp, np.where(np.asarray(start >= b[c]), x + 100.0, x), zone, spec)[0].reindex(DAYS).values
            assert np.array_equal(s2[:c + 1], s.values[:c + 1]), f"{g}, {w}: rain after the end of the window of {DAYS[c].date()} changed a value up to that day"
            assert s2[c + 1] != s.values[c + 1], f"{g}, {w}: the test could not see a change of the next day"
        RAIN2[w][g] = s
        at = lambda arr, pos: arr[pos - i0[0]]                                              # the wall-clock start or end of the hour at a position of the hourly series
        first_start, last_end, margin = at(sw, i0) - DAYS, at(ew, i1 - 1) - DAYS, (issue - b) / HOUR
        CLOCK2[w][g] = dict(short=F.SHORT[g], utc_offset_h=off, time_zone=zone, n_days=int(len(s)), first_day=str(DAYS[0].date()), last_day=str(DAYS[-1].date()),
                            hours_per_window=[int(k) for k in np.unique(n_h)], first_hour_starts_local_hours_from_00_of_day_t=[float(first_start.min() / HOUR), float(first_start.max() / HOUR)],
                            last_hour_ends_local_min=clock_hm(last_end.min()), last_hour_ends_local_max=clock_hm(last_end.max()),
                            last_hour_summed_ends_utc_on_last_day=str(start[i1[-1] - 1] + HOUR), last_hour_ends_before_the_issue_time_h=[float(margin.min()), float(margin.max())],
                            made_up_hours_cut_days=[str(DAYS[c].date()) for c in cuts], same_hours_as_aorc_to_midnight=True)
        LEN2[w][g] = dict(short=F.SHORT[g], time_zone=zone, days_with_23_hours=int((n_h == 23).sum()), days_with_24_hours=int((n_h == 24).sum()), days_with_25_hours=int((n_h == 25).sum()))
clock_lo = lambda w, key: min(c[key] for c in CLOCK2[w].values())
clock_hi = lambda w, key: max(c[key] for c in CLOCK2[w].values())
margin_lo, margin_hi = {w: min(c["last_hour_ends_before_the_issue_time_h"][0] for c in CLOCK2[w].values()) for w in NEW_WINDOWS}, {w: max(c["last_hour_ends_before_the_issue_time_h"][1] for c in CLOCK2[w].values()) for w in NEW_WINDOWS}
assert margin_lo["aorc_to_09"] > 0 and margin_lo["aorc_utc_day"] > 0, "a window ends at or after the issue time"
WCHECK = {w: {g: dict(daymet=three(D[g].prcp, S[g]), nldas=three(NL[g], S[g])) for g in ids} for w, S in (("aorc_to_midnight", RAIN), ("aorc_to_09", RAIN2["aorc_to_09"]), ("aorc_utc_day", RAIN2["aorc_utc_day"]))}
WCHECK_MED = {w: {k: {c: float(np.median([WCHECK[w][g][k][c] for g in ids])) for c in ("day_before", "same_day", "day_after")} for k in ("daymet", "nldas")} for w in WCHECK}      # correlation of each product with each AORC window, median of the basins
nldas_nearest = max(WCHECK_MED, key=lambda w: WCHECK_MED[w]["nldas"]["same_day"])                                   # the AORC window that the NLDAS day resembles most
DL = {k: float(np.median([v[k] for v in DAYLEN.values()])) for k in ("days_with_23_hours", "days_with_24_hours", "days_with_25_hours", "share_of_days_on_summer_time_at_the_issue_time")}
DL_same = len({tuple(v[k] for k in ("days_with_23_hours", "days_with_24_hours", "days_with_25_hours", "n_days_on_summer_time_at_the_issue_time")) for v in DAYLEN.values()}) == 1       # all basins alike
FLOW = DW["river_flow_day"]["summary"]                                                      # which clock the daily flow follows (15-minute flow, results/day_windows.json)
n_flow_gauges, whole_days, summer_days = FLOW["n_gauges"], FLOW["parts"]["whole_period"], FLOW["parts"]["daylight_saving_time_in_force"]
n_summer_clock, std_gauges = summer_days["n_gauges_best_midnight_summer_time"], summer_days["gauges_best_midnight_standard_time"]       # gauges whose flow day follows the summer clock, and those that follow standard time

med = lambda E, k: E["summary"][lk(k)]["nse_median"]                   # median NSE over the basins of one evaluation, at lead k

# ---- 2. the same LSTM with the AORC rain up to midnight, with its controls, and with the rain table replaced ------------------------------------------
nld, dym = DW["day_ends"]["nldas"], DW["day_ends"]["daymet"]
doc_end = sorted({24 + UTC_OFF[g] for g in ids}, reverse=True)                      # the hour (local standard time) at which a UTC day ends: 19 in the Eastern basins, 18 in Central time
D2 = {g: D[g].assign(aorc_to_midnight=RAIN[g].values, aorc_to_09=RAIN2["aorc_to_09"][g].values, aorc_utc_day=RAIN2["aorc_utc_day"][g].values) for g in ids}
HOURS = {"aorc_to_midnight": "AORC hourly rain (the mean of the 1 km cells inside the basin outline, data/aorc) of the local wall-clock day t, summer time included: every hour goes to the local calendar day in which it ends, "
                             "so the last hour of day t ends at 24:00 by the wall clock, the end of day t and the issue time; a day has 24 hours, or 23 or 25 where the clocks change; time zones: " + zone_text,
         "aorc_to_09": f"AORC hourly rain over the wall-clock window from 09:00 of day t-1 to 09:00 of day t, the window of the rain table itself (as measured in results/day_windows.json, day_ends.daymet: "
                       f"{dym['day_ends_local']} local standard time, rounded to {dym['day_ends_local_rounded']}), summer time included: every hour goes to the window in which it ends; a window has 24 hours, or 23 or 25 where "
                       f"the clocks change; the last hour ends at {clock_hi('aorc_to_09', 'last_hour_ends_local_max')} of day t, {margin_lo['aorc_to_09']:g} hours before the issue time (local midnight at the end of day t); "
                       "time zones: " + zone_text,
         "aorc_utc_day": f"AORC hourly rain of the UTC day of the date of day t: from 00:00 UTC of that date to 00:00 UTC of the next date, always 24 hours (the day that a rain product served by UTC day holds, as "
                         f"IMERG at NASA POWER); in local standard time it ends at {doc_end[0]}:00 in the Eastern basins and {doc_end[-1]}:00 in Central time, by the wall clock between {clock_lo('aorc_utc_day', 'last_hour_ends_local_min')} and "
                         f"{clock_hi('aorc_utc_day', 'last_hour_ends_local_max')} (summer time), {margin_lo['aorc_utc_day']:g} to {margin_hi['aorc_utc_day']:g} hours before the issue time (local midnight at the end of day t)",
         "aorc_replaces_prcp": "the input column that holds the Daymet rain (prcp) in every other set holds aorc_to_midnight here (same position, same number of columns as the set wq): the rain table with the right clock, "
                               "and nothing else changed; the Daymet rain is not an input of this set"}
REPLACED = {"prcp": "aorc_to_midnight"}                                                               # wqa_rep reads this column of FORCING from the AORC series (its position in the input stays)
VAR = {"wqa": dict(use_flow=True, base="wq", ceiling="wqf", nldas="wqn", era5="wqh", cols=("aorc_to_midnight",), replaces=None, control=False, what="rain to midnight (AORC, hourly, 1 km)"),
       "wqa_to09": dict(use_flow=True, base="wq", ceiling="wqf", nldas="wqn", era5="wqh", cols=("aorc_to_09",), replaces=None, control=True,
                        what="AORC rain over the rain table's own window, 09:00 of day t-1 to 09:00 of day t (control: a second rain product, no new hour)"),
       "wqa_utc": dict(use_flow=True, base="wq", ceiling="wqf", nldas="wqn", era5="wqh", cols=("aorc_utc_day",), replaces=None, control=True, what="AORC rain of the UTC day (control)"),
       "wqa_rep": dict(use_flow=True, base="wq", ceiling="wqf", nldas="wqn", era5="wqh", cols=(), replaces=REPLACED, control=False,
                       what="the Daymet rain column replaced by AORC rain to midnight (no extra column; wqa_daymet_replaced)")}
for v_ in VAR.values():                                                                              # the daily input columns of each set, in the order the network reads them
    v_["dyn"] = tuple(REPLACED.get(c, c) if v_["replaces"] else c for c in F.FORCING) + v_["cols"]
MAIN = {"wqa": "wqa_to09"}                                                                           # the test and its control: the same AORC rain over the table's own window, no new hour
UTC_CONTROL, REP, BOTH, ERA5, NLDAS, BASE_SET = "wqa_utc", "wqa_rep", "wqnh", "wqh", "wqn", "wq"       # the second control, the replaced table, NLDAS + ERA5 together (script 15), and the sets the AORC sets are compared with
CONTROLS = {"wqn": {"wqn_older": "older"}, "wqh": {"wqh_older": "older", "wqh_to09": "to09", "wqh_utc": "utc"}, "wqa": {"wqa_to09": "to09", "wqa_utc": "utc"}}      # controls drawn hollow on a rung: set -> {control: kind}
FAMILY = {"wq": dict(label="With river gauge", test="wqa",
                     rungs=[(0, "wq", None), (1, "wqn", "wqn_older"), (2, "wqh", "wqh_older"), (3, "wqa", "wqa_to09"), (4, REP, None), (5, BOTH, None), (6, "wqf", None)])}        # (position on the ladder, set, its control)
RUNG = ["rain table (Daymet)", "NLDAS rain", "rain to midnight (ERA5)", "rain to midnight (AORC)", "AORC rain replaces the rain table", "NLDAS rain + ERA5 to midnight", "perfect weather forecast"]
LADDER = [n for f in FAMILY.values() for r in f["rungs"] for n in r[1:] if n and n not in VAR] + [c for cs in CONTROLS.values() for c in cs if c not in VAR]
LADDER = list(dict.fromkeys(LADDER))                                                                 # the sets read from the files of the earlier scripts, once each
runs, ENS, costs, files, trained_now = {}, {}, {}, {}, {}


def save_model(run, name, seed):
    """results/models/lstm_<set>_s<seed>.pt: a plain dict (tensors, numbers, lists and text) that torch.load(weights_only=True) reads, enough to forecast again."""
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in run.net.state_dict().items()}, cfg=asdict(run.cfg),
                    sc={k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in run.sc.items()}, ids=list(run.ids), dyn=list(run.cfg.dyn),
                    split=dict(run.split), best_val_nse=float(run.best_val_nse), n_params=int(run.n_params), input_set=name, script=SCRIPT, seed=seed,
                    history=run.history, seconds=float(run.seconds), cpu_seconds=float(run.cpu_seconds), updates_run=int(run.updates_run), best_update=int(run.best_update),
                    torch_version=str(torch.__version__), quick_mode=F.QUICK,
                    how_to_reload="cfg = flood.Cfg(**cfg); net = flood.LSTMNet(n_in, cfg); net.load_state_dict(state_dict); sc lists back to numpy arrays; "
                                  "flood.Run(cfg, sc, net, ids, history, seconds, cpu_seconds, updates_run, best_update, best_val_nse, n_params, split).predict(D, ids, static_df); "
                                  f"D needs the columns in dyn: aorc_to_midnight, aorc_to_09 and aorc_utc_day are built by {SCRIPT} (the hourly AORC basin means of data/aorc, made by code/fetch_aorc.py, summed over the local wall-clock day, "
                                  "over the window of the rain table, over the UTC day; in the set wqa_rep aorc_to_midnight stands where the Daymet rain prcp stands in the other sets); the other columns are the Daymet table of flood.load_data()"), tmp)
    tmp.replace(f)
    return f


def reload_and_forecast(name, seed):
    """Rebuild the trained network from its file alone, forecast again on the SAME tables and compare with the stored forecasts of the set."""
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    z = torch.load(f, map_location="cpu", weights_only=True)
    assert z["input_set"] == name and z["script"] == SCRIPT and z["seed"] == seed, f"{f.name} is not the network of {name}, seed {seed}"
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    assert z["state_dict"]["lstm.weight_ih_l0"].shape[1] == n_in and list(cfg.dyn) == z["dyn"], f"{f.name}: the input columns do not fit the weights"
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    new, old = run.predict(D2, z["ids"], B), F.load_preds(f"lstm_{name}_s{seed}")
    cols = ["obs"] + [lk(k) for k in F.LEADS]
    a, b = np.stack([old[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    close = bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True))
    assert same_nan and close, f"{f.name}: the reloaded network does not reproduce the stored forecasts"
    return dict(set=name, seed=seed, model_file=rel(f), size_kb=round(f.stat().st_size / 1024, 1), n_in=int(n_in), n_params=int(z["n_params"]),
                n_values_compared=int(a.size), n_missing_in_both=int(np.isnan(a).sum()), max_abs_difference=float(np.nanmax(np.abs(a - b))),
                bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), same_missing_days=same_nan, allclose=close, passed=True)



# ---- the simple models given the new input: the yardstick of rule 2 of the work order, by the recipe of 02_baselines.py, copied here and checked against its stored forecasts ---------
BASE = json.loads((F.RES / "baselines.json").read_text())["models"]                        # the settings chosen there on the validation years for the input set wq (the real file, also in a smoke test)
RIDGE_ALPHA, RF_SET = BASE["base_linreg_wq"]["settings"]["ridge_alpha"], BASE["base_rf_wq"]["settings"]
assert BASE["base_linreg_wq"]["settings"]["standardised"] and RF_SET["max_features"] == "one third" and RF_SET["random_state"] == 0, "the recipe of 02_baselines.py has changed: read it again"
RF_TREES, RF_JOBS = (8, 1) if F.QUICK else (RF_SET["n_trees"], 4)                          # a smoke test fits tiny forests, as 02_baselines.py does; 4 jobs as there
MAKE = {"base_linreg_wqa": lambda: make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
        "base_rf_wqa": lambda: RandomForestRegressor(RF_TREES, min_samples_leaf=RF_SET["min_samples_leaf"], max_features=1 / 3, n_jobs=RF_JOBS, random_state=RF_SET["random_state"])}
YARD = tuple(MAKE)
SETTINGS = {"base_linreg_wqa": dict(ridge_alpha=RIDGE_ALPHA, standardised=True), "base_rf_wqa": dict(n_trees=RF_TREES, min_samples_leaf=RF_SET["min_samples_leaf"], max_features="one third", random_state=RF_SET["random_state"])}
KIND = {"base_linreg_wqa": "linreg", "base_rf_wqa": "rf"}


def feature_table_with(d, col=None, flow=True):
    """The features of features.feature_table(d, flow) for the input set wq (flow=True) and, when col is given, the same rain features (the rain of the last three days, the sums over 3 to 90 days and the two
    antecedent-rain indices) computed on the extra rain series d[col]: by the same code, which reads the rain from d.prcp, so feature_table is called a second time with the series in its place. This is the
    smallest hook, here and not in features.py. As there, a row without the full history is NaN in every column."""
    X = feature_table(d, flow, 0)
    if col is None:
        return X
    Y = feature_table(d.assign(prcp=d[col]), False, 0)
    X2 = pd.concat([X, Y[[c for c in Y.columns if c.startswith(("rain_", "wetness_"))]].add_prefix(f"{col}__")], axis=1)
    return X2.where(X2.notna().all(axis=1))


def simple_design(col):
    """What 02_baselines.design() builds for one input set, for wq + the series col (col=None: wq itself): per basin and lead the feature rows, the complete rows, the rows used for fitting (target day in the
    training years) and the flow to predict (the same arrays as 02_baselines.design('wq'), rebuilt here)."""
    T = {}
    for g in ids:
        d = D2[g]
        X = feature_table_with(d, col)
        ok = X.notna().all(axis=1).values
        for k in F.LEADS:
            T[g, k] = X.values, ok, ok & F.part_mask(d.index, k, "train"), d.q_mm.shift(-k).values
    return T


def features_are_causal(col="aorc_to_midnight", cut=5000):
    """The self-check of features.py, run on the real table of every basin with the new series: change EVERYTHING after a cut day (every column, the new series too): no feature row up to the cut may change,
    and the next one must (or the check proves nothing)."""
    for g in ids:
        d = D2[g]
        bad = d.copy()
        bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
        A, A2 = feature_table_with(d, col), feature_table_with(bad, col)
        assert A.columns.equals(A2.columns) and A.iloc[:cut + 1].equals(A2.iloc[:cut + 1]), f"{g}: a feature reads data after the issue day"
        assert not np.allclose(A.iloc[cut + 1].values, A2.iloc[cut + 1].values), f"{g}: the check of the features has no power"
        assert (A.isna().all(axis=1) | A.notna().all(axis=1)).all(), f"{g}: a feature row is partly missing"
    return dict(what="features.py's own check on the real tables: every column of the daily table (rain, the new series, the flow ...) was replaced after a cut day (day number " + str(cut) + " of the record); "
                     "the feature rows up to the cut did not change and the next row did, in every basin", passed=True, n_basins=len(ids), n_features=int(feature_table_with(D2[ids[0]], col).shape[1]))


def forecast_simple(make, T, Dtab):
    """One model per basin and lead, fitted on the training years; forecasts every issue day that has features. A copy of 02_baselines.forecast() (a numbered script is not imported); recipe_check() proves that
    this copy, run on the input set wq, gives the forecasts that 02_baselines.py stored."""
    P = {}
    for g, d in Dtab.items():
        out = {}
        for k in F.LEADS:
            X, ok, fit, y = T[g, k]
            m = make().fit(X[fit], y[fit])
            out[k] = np.full(len(d), np.nan)
            out[k][ok] = np.clip(m.predict(X[ok]), 0, None)                                 # a river cannot have negative flow
        P[g] = F.frame_from_issue(d, out)
    return P


def recipe_check():
    """With the new series left out this code must be the code of 02_baselines.py: it must give the forecasts that 02_baselines.py stored as base_linreg_wq and, outside a smoke test, base_rf_wq (a forest with
    its random_state is repeatable). 02_baselines.py is not imported (it is a numbered script): the feature rows (features.feature_table for wq) and the fit are rebuilt here, so equal forecasts are the proof."""
    T0 = simple_design(None)
    out = dict(what="the recipe of 02_baselines.py (feature rows of features.feature_table for the input set wq, one model per basin and lead fitted on the training years), rebuilt in this script and run on the input set wq "
                    "(without the new series): its forecasts against the stored base_linreg_wq and base_rf_wq")
    for n, stored in (("base_linreg_wqa", "base_linreg_wq"), ("base_rf_wqa", "base_rf_wq")):
        if n == "base_rf_wqa" and F.QUICK:
            out["forest_wq_max_abs_difference_from_stored_base_rf_wq"] = "not run in a smoke test (tiny forests)"
            continue
        P, S = forecast_simple(MAKE[n], T0, D2), F.load_preds(stored)
        assert all(np.array_equal(np.isnan(P[g][lk(k)].values), np.isnan(S[g][lk(k)].values)) for g in ids for k in F.LEADS), f"{stored}: the missing days differ"
        diff = float(max(np.nanmax(np.abs(P[g][lk(k)].values - S[g][lk(k)].values)) for g in ids for k in F.LEADS))
        assert diff < 1e-9, f"{stored} is not reproduced by this code (largest difference {diff})"
        out[("ridge" if n == "base_linreg_wqa" else "forest") + "_wq_max_abs_difference_from_stored_" + stored] = diff
    return out


def fit_simple(names):
    """Ridge regression and random forest for wq + aorc_to_midnight, one model per basin and lead, fitted on the training years only, with the settings of 02_baselines.py for wq; {name: (forecasts, seconds)}."""
    T, out = simple_design("aorc_to_midnight"), {}
    for n in names:
        t0 = time.perf_counter()
        out[n] = (forecast_simple(MAKE[n], T, D2), round(time.perf_counter() - t0, 1))
        print(f"{n}: one model per basin and lead, fitted on the training years in {out[n][1]:.0f} s", flush=True)
    return out


FEATURES = list(feature_table_with(D2[ids[0]], "aorc_to_midnight").columns)
have_seed = lambda name, seed: (F.PRED / f"lstm_{name}_s{seed}.npz").exists()               # the prediction files of THIS run's folder (a smoke test looks at results/quick only); never retrained when they exist
have_simple = lambda n: (F.PRED / f"{n}.npz").exists()


def record_of(name, seed):
    """The training record of a saved network: from the results file of the earlier run when it holds that set, otherwise from the network's own file, which holds the same numbers."""
    for r in old["models"].get(name, {}).get("per_seed", []):
        if r["seed"] == seed:
            return r
    f = MODELS / f"lstm_{name}_s{seed}.pt"
    assert f.exists(), f"lstm_{name}_s{seed}: neither the results file nor a network file holds its training record"
    z = torch.load(f, map_location="cpu", weights_only=True)
    return dict(seed=seed, **{k: z[k] for k in COST}, model_file=rel(f))


if LOAD:                                                                                   # the settings and training records of the training run
    assert (F.RES_OUT / f"{OUT}.json").exists(), "--rescore and --extend need the results file of a training run: run the script without them first"
    old = json.loads((F.RES_OUT / f"{OUT}.json").read_text())
    old_written = time.strftime("%Y-%m-%d %H:%M", time.localtime((F.RES_OUT / f"{OUT}.json").stat().st_mtime))
    conf = old["config"]
    notes += [x for x in old["notes"] if x.startswith("results/lstm_all.json was not found")]
missing_sets = [(n, s) for n in VAR for s in SEEDS if not have_seed(n, s)]
missing_simple = [n for n in YARD if not have_simple(n)]
if RESCORE:
    assert not missing_sets and not missing_simple, ("--rescore trains nothing, and these are missing: " + ", ".join([f"lstm_{n}_s{s}" for n, s in missing_sets] + missing_simple) + ": run --extend first")
old_checks = old.get("yardstick", {}).get("checks") if LOAD else None                       # the checks of the simple-model code, when an earlier pass made them
run_checks = (not LOAD) or bool(missing_simple) or (EXTEND and not old_checks)
need_cpu = not RESCORE and (not EXTEND or bool(missing_sets or missing_simple or run_checks))      # --extend asks for the CPU only when it has something to make
t_ask = time.time()

extra_of = lambda v: " + ".join(v["cols"]) or "none: the Daymet rain column prcp is replaced by " + ", ".join(v["replaces"].values())                 # the extra input(s) of a set, in words
hours_of = lambda v: " | ".join([HOURS[c] for c in v["cols"]] + ([HOURS["aorc_replaces_prcp"], HOURS["aorc_to_midnight"]] if v["replaces"] else []))       # the hours its rain input(s) cover
with F.cpu_lock(SCRIPT) if need_cpu else contextlib.nullcontext():
    waited = time.time() - t_ask
    if not LOAD:
        conf_file = F.RES / "lstm_all.json"                # read once the CPU is ours: 04_lstm_all.py writes it when it finishes
        conf = json.loads(conf_file.read_text())["config"] if conf_file.exists() else {}
        if not conf:
            notes.append("results/lstm_all.json was not found: the engine's default settings were used.")
        if F.QUICK:
            conf = dict(conf, updates=40, eval_every=20)
    elif EXTEND and missing_sets and not F.QUICK and (F.RES / "lstm_all.json").exists():
        assert conf == json.loads((F.RES / "lstm_all.json").read_text())["config"], "the settings of the existing sets are not those of results/lstm_all.json: the new sets would not be trained with the same settings"
    for name, v in VAR.items():
        runs[name], costs[name], files[name], trained_now[name] = [], {}, {}, []
        for i, seed in enumerate(SEEDS):
            if LOAD and (RESCORE or have_seed(name, seed)):          # a saved set is read, never made again
                p, meta, rec = F.load_preds(f"lstm_{name}_s{seed}"), F.preds_meta(f"lstm_{name}_s{seed}"), record_of(name, seed)
                assert meta["config"] == conf and meta["seed"] == rec["seed"] == seed and meta["script"] == SCRIPT, f"lstm_{name}_s{seed} is not from the training run described in {OUT}.json"
                cost, mf = {k: rec[k] for k in COST}, rec.get("model_file")
            else:
                cfg = F.Cfg(dyn=v["dyn"], use_flow=v["use_flow"], static=tuple(F.STATIC), seed=seed, **conf)
                run = F.fit(cfg, D2, ids, static_df=B)
                p, cost = run.predict(D2, static_df=B), run.summary()
                mf = rel(save_model(run, name, seed))
                F.save_preds(f"lstm_{name}_s{seed}", p, dict(script=SCRIPT, model="lstm", inputs=name, seed=seed, config=conf, dyn=list(cfg.dyn), extra_input=extra_of(v), extra_inputs=list(v["cols"]), replaces=v["replaces"],
                                                             control=v["control"], hours_covered=hours_of(v), model_file=mf, what="one LSTM for all 12 basins, a single seed"))
                trained_now[name].append(seed)
            e = F.evaluate(p, thr)
            runs[name].append(p)
            costs[name][seed], files[name][seed] = cost, mf
            seed_med = " ".join(f"{med(e, k):.3f}" for k in F.LEADS)
            print(f"lstm_{name} seed {seed} | best at update {cost['best_update']} of {cost['updates_run']}, {cost['seconds']:.0f} s | validation NSE {cost['best_val_nse']:.3f} | "
                  f"test NSE median {seed_med}" + ("" if seed in trained_now[name] else " | read, not trained again"), flush=True)
        ens = F.ensemble(runs[name])
        if trained_now[name]:                              # a rescore, or an extension for a set that exists, reads the prediction sets and leaves them untouched
            F.save_preds(f"lstm_{name}", ens, dict(script=SCRIPT, model="lstm", inputs=name, seeds=SEEDS, config=conf, dyn=list(v["dyn"]), extra_input=extra_of(v), extra_inputs=list(v["cols"]), replaces=v["replaces"],
                                                   control=v["control"], hours_covered=hours_of(v), what="one LSTM for all 12 basins, day-by-day average of the seeds"))
        ENS[name] = ens
    # the proof that a saved network forecasts again: one seed per input set, rebuilt from its file alone
    try:
        reload_check = dict(what="one trained network per input set (the first seed) rebuilt from its .pt file alone (flood.LSTMNet + the saved weights, settings and scalers), "
                                 "run on the same tables, and compared with the forecasts stored by save_preds", tolerance="numpy.allclose, rtol 1e-6, atol 1e-8, missing values equal",
                            per_set={n: reload_and_forecast(n, SEEDS[0]) for n in VAR}, carried_over_from_training_run=False)
    except FileNotFoundError:
        assert LOAD, "a trained network was not saved"
        reload_check = dict(old["reload_check"], carried_over_from_training_run=True)
        notes.append("The network files were not found, so the reload check of the training run was carried over, not repeated.")
    # the simple models given the new input: fitted when they are missing (or in the plain run), otherwise read
    SIMPLE = {}
    if run_checks:                                                                          # before the fit, so that a code that does not follow the recipe is stopped before it writes a prediction set
        yard_check = dict(features=features_are_causal(), recipe=recipe_check())
    elif old_checks:
        yard_check = old_checks
    else:
        yard_check = None
        notes.append("The checks of the simple-model code (yardstick.checks) were not found in the results file of the earlier run, and --rescore does not make them: run --extend to make them.")
    if not LOAD or missing_simple:
        fitted = fit_simple([n for n in YARD if not LOAD or n in missing_simple])
        for n, (P, sec) in fitted.items():
            meta = dict(script=SCRIPT, model=KIND[n], inputs="wqa", settings=SETTINGS[n], features=FEATURES, extra_input="aorc_to_midnight", hours_covered=HOURS["aorc_to_midnight"], fit_seconds=sec, quick_mode=F.QUICK,
                        what="one model per basin and lead on the features of the input set wq plus the same rain features of aorc_to_midnight (02_baselines.py recipe; settings as chosen there on the validation years for wq)")
            F.save_preds(n, P, meta)
    for n in YARD:
        SIMPLE[n] = dict(P=F.load_preds(n), meta=F.preds_meta(n))
        assert SIMPLE[n]["meta"]["script"] == SCRIPT and SIMPLE[n]["meta"]["settings"] == SETTINGS[n] and SIMPLE[n]["meta"]["features"] == FEATURES, f"{n} is not from the fit described here"
trained_pairs = [(n, s) for n in VAR for s in trained_now[n]]
this_pass = dict(kind="training run" if not LOAD else "extension" if EXTEND else "rescore",
                 what=("training run: every set from scratch" if not LOAD else "extension (--extend): the sets whose files were missing" if EXTEND else "rescore (--rescore): scores, tables, notes and figure recomputed from the saved sets, nothing trained"),
                 n_trainings=len(trained_pairs), sets_trained=sorted({n for n, _ in trained_pairs}), seeds_trained=sorted({s for _, s in trained_pairs}), training_seconds=round(sum(costs[n][s]["seconds"] for n, s in trained_pairs), 1), waited_for_cpu_s=round(waited, 1),
                 simple_models_fit_seconds=round(sum(SIMPLE[n]["meta"]["fit_seconds"] for n in YARD if n in missing_simple or not LOAD), 1), script_sha256=HASH_NOW)
if LOAD:                                                                                   # the passes of the earlier runs (a results file written before 'passes' existed holds one training run)
    first = [dict(p) for p in (old.get("passes") or [dict(what="training run: every set from scratch (a results file written without a list of passes)", n_trainings=old["n_trainings"], training_seconds=old["training_seconds"],
                                                          waited_for_cpu_s=old["waited_for_cpu_s"], runtime_s=old["runtime_s"], finished_local_time=old_written)])]
    for p in first:
        p.setdefault("script_sha256", "not recorded")                                      # made by a version of this script that no longer exists: no guess
    first = [p for p in first if p.get("kind") != "rescore"]                               # only the last rescore wrote the numbers of the file it leaves
if RESCORE:
    timing = dict({k: old[k] for k in ("n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s") if k in old}, passes=first + [this_pass])
elif EXTEND:
    timing = dict(n_trainings=sum(p["n_trainings"] for p in first) + this_pass["n_trainings"], training_seconds=round(sum(p["training_seconds"] for p in first) + this_pass["training_seconds"], 1),
                  waited_for_cpu_s=round(sum(p["waited_for_cpu_s"] for p in first) + waited, 1), passes=first + [this_pass])
else:
    timing = dict(n_trainings=this_pass["n_trainings"], training_seconds=this_pass["training_seconds"], waited_for_cpu_s=this_pass["waited_for_cpu_s"], passes=[this_pass])

# ---- integrity: what was made in this pass, and that what was only read is what it was ------------------------------------------------------------
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
SAVED = {}                                                                                   # sha256 of every file of this script's sets: prediction sets, single seeds, networks, simple models
for n in VAR:
    for f in [F.PRED / f"lstm_{n}.npz"] + [p for s in SEEDS for p in (F.PRED / f"lstm_{n}_s{s}.npz", MODELS / f"lstm_{n}_s{s}.pt")]:
        SAVED[rel(f)] = sha(f)
for n in YARD:
    SAVED[rel(F.PRED / f"{n}.npz")] = sha(F.PRED / f"{n}.npz")
made_now = {rel(F.PRED / f"lstm_{n}_s{s}.npz") for n, s in trained_pairs} | {rel(MODELS / f"lstm_{n}_s{s}.pt") for n, s in trained_pairs} | {rel(F.PRED / f"lstm_{n}.npz") for n in VAR if trained_now[n]} \
    | {rel(F.PRED / f"{n}.npz") for n in YARD if not LOAD or n in missing_simple}
earlier_hashes = old.get("integrity", {}).get("sha256_of_saved_files", {}) if LOAD else {}
compared = [f for f in SAVED if f not in made_now and f in earlier_hashes]
assert all(earlier_hashes[f] == SAVED[f] for f in compared), "a saved file changed since the earlier results file recorded its hash: " + ", ".join(f for f in compared if earlier_hashes[f] != SAVED[f])
review_hashes = {p: v["sha256"] for p, v in REVIEW["files_reviewed"].items() if p.startswith(("results/preds/", "results/models/")) and isinstance(v, dict) and v.get("sha256")} \
    if (REVIEW and not F.QUICK and isinstance(REVIEW.get("files_reviewed"), dict)) else {}
AVERAGES = {rel(F.PRED / f"lstm_{n}.npz") for n in VAR}          # the seed averages are remade whenever a seed is added (seeds 3-4, 30 October 2026): the review's hash of them is a record, not a check
review_cmp = {p: h for p, h in review_hashes.items() if p not in AVERAGES}
INTEGRITY = dict(how_to_read="sha256 of every prediction set and network file of this script, as they are when this file is written; made_in_this_pass lists the files this pass wrote; every other file was only read. "
                             "A later pass compares the hashes of the files it only reads with these and stops if one differs.",
                 script_sha256=HASH_NOW, sha256_of_saved_files=SAVED, made_in_this_pass=sorted(made_now), only_read_in_this_pass=len(SAVED) - len(made_now),
                 compared_with_the_earlier_results_file=(dict(files_compared=len(compared), all_equal=True) if compared else dict(files_compared=0, all_equal=None, why="the earlier results file holds no hashes (it was written before this record existed)")) if LOAD else None,
                 compared_with_the_files_the_review_hashed=dict(files_compared=len(review_cmp), all_equal=all(SAVED.get(p) == h for p, h in review_cmp.items()), files=sorted(review_cmp),
                                                               seed_averages_the_review_hashed={p: dict(review_sha256=h, now_sha256=SAVED.get(p), remade_since_the_review=SAVED.get(p) != h) for p, h in review_hashes.items() if p in AVERAGES}) if review_hashes else None)
assert INTEGRITY["compared_with_the_files_the_review_hashed"] is None or INTEGRITY["compared_with_the_files_the_review_hashed"]["all_equal"], "a file the review hashed has changed since the review"


# ---- 3. every set scored the same way ---------------------------------------------------------------------------------------------
missing = [n for n in LADDER if not F.have_preds(f"lstm_{n}")]
notes += [f"lstm_{n} was not found: it is left out of the ladder and of every comparison with it." for n in missing]
SHOWN = list(VAR) + [n for n in LADDER if n not in missing]
PRED = {n: ENS[n] if n in VAR else F.load_preds(f"lstm_{n}") for n in SHOWN}
SEEDP = {n: runs[n] if n in VAR else [F.load_preds(f"lstm_{n}_s{s}") for s in range(5) if F.have_preds(f"lstm_{n}_s{s}")] for n in SHOWN}
EV = {n: dict(test=F.evaluate(PRED[n], thr), val=F.evaluate(PRED[n], thr, "val"), seeds=[F.evaluate(p, thr) for p in SEEDP[n]]) for n in SHOWN}
assert all(EV[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in SHOWN for part in ("test", "val") for k in F.LEADS), "days without a forecast: the models would not be compared on identical days"

nse_of = lambda E, k: np.array([E["basins"][g][lk(k)]["nse"] for g in ids])
ons = lambda E, k, c="pod": E["summary"][lk(k)]["onset"][c]                                         # onset warnings pooled over the basins
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None      # lowest and highest single seed (a seed with no value is left out)
div = lambda a, b: a / b if b else float("nan")                                                      # a share of a gap that may be zero in a smoke test
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                            # every single seed of a above every single seed of b
pc = lambda x: f"{round(100 * x) + 0}%" if x == x else "no value"                                    # a share in per cent (never "-0%")
p1 = lambda x: f"{100 * x:.1f}%" if x == x else "no value"
n3 = lambda x: f"{x:.3f}"
n2 = lambda x: f"{x:.2f}"
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
yes = lambda b: "yes" if b else "no"
pv = lambda p: "p not defined" if p != p else "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
row = lambda E, k: dict(nse_median=med(E, k), nse_mean=E["summary"][lk(k)]["nse_mean"], onset=CUT(E["summary"][lk(k)]["onset"]), all=CUT(E["summary"][lk(k)]["all"]))


def paired(a, b):
    """Two per-basin score arrays (first minus second): in how many basins the first is higher, the median difference, Wilcoxon p (as in 04_lstm_all.py)."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)                        # two-sided signed-rank test over the basins
    except ValueError:                                          # all differences zero
        p = float("nan")
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=int(len(d)), median_nse_difference=float(np.median(d)), wilcoxon_p=p)


def seed_range(n, k):
    """Lowest and highest single-seed value of a set at one lead, for the scores the rule and the figure use."""
    S = [e["summary"][lk(k)] for e in EV[n]["seeds"]]
    return dict(n_seeds=len(S), nse_median=lo_hi([s["nse_median"] for s in S]), **{f"{kind}_{c}": lo_hi([s[kind][c] for s in S]) for kind in ("onset", "all") for c in ("pod", "far", "csi")},
                by_seed=dict(nse_median=[s["nse_median"] for s in S], **{f"{kind}_{c}": [s[kind][c] for s in S] for kind in ("onset", "all") for c in ("pod", "far", "csi")}))


summary_test = {n: {lk(k): row(EV[n]["test"], k) for k in F.LEADS} for n in SHOWN}
summary_val = {n: {lk(k): row(EV[n]["val"], k) for k in F.LEADS} for n in SHOWN}
ranges = {n: {lk(k): seed_range(n, k) for k in F.LEADS} for n in SHOWN}
per_basin = {"how_to_read": "test years, one value per basin; the seed average of each set (flood.evaluate); gauge ids are text",
             "basins": {g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g])) for g in ids},
             "nse": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["nse"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "onset_warned": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["onset"]["pod"] for g in ids} for k in F.LEADS} for n in SHOWN},
             "all_days_warned": {n: {lk(k): {g: EV[n]["test"]["basins"][g][lk(k)]["all"]["pod"] for g in ids} for k in F.LEADS} for n in SHOWN}}
models = {n: dict(saved_as=f"lstm_{n}", n_seeds_scored=len(EV[n]["seeds"]), test=EV[n]["test"]) for n in SHOWN}
for n in VAR:
    models[n]["per_seed"] = [dict(seed=s, **{c: costs[n][s][c] for c in COST}, model_file=files[n][s], test_nse_median=[med(e, k) for k in F.LEADS],
                                  test_onset_warned=[ons(e, k) for k in F.LEADS], test_onset_false=[ons(e, k, "far") for k in F.LEADS],
                                  test_all_days_warned=[e["summary"][lk(k)]["all"]["pod"] for k in F.LEADS]) for s, e in zip(SEEDS, EV[n]["seeds"])]
training = [dict(set=n, seed=s, saved_forecasts=f"lstm_{n}_s{s}", model_file=files[n][s], **{c: costs[n][s][c] for c in COST}) for n in VAR for s in SEEDS]


# ---- the checks of report section 4.2, and the share of the gap to the perfect forecast that is closed -------------------------------
def rule(T, C, k):
    """Set T against set C at lead k. NSE: every single seed of T above every single seed of C, higher in most basins, Wilcoxon p below P_CUT - all three
    make a clear gain (as 10_rain_timing.py reads its results; there the p-value was not stored, here it is added with the test of 04_lstm_all.py).
    Onsets warned: pooled over the basins, so only the seeds can be compared (10_rain_timing.py does the same)."""
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

# the checks of section 4.2 for the main test (the AORC rain to midnight) against its controls, the rain table alone, ERA5 and NLDAS; and for the replaced table
GR = lambda T, C: {lk(k): rule(T, C, k) for k in F.LEADS}
gain_rule, closure = {}, {}
need_main = [x for x in ("wqa", "wqa_to09", "wqa_utc", "wq", "wqf") if x not in EV]
if need_main:
    notes.append("These sets were not found, so the checks and the closure of the main test are left out: " + ", ".join(f"lstm_{x}" for x in need_main) + ".")
else:
    gain_rule["wqa"] = dict(against_control=GR("wqa", MAIN["wqa"]), against_utc_day=GR("wqa", "wqa_utc"), against_rain_table_alone=GR("wqa", "wq"),
                            **({"against_nldas_rain": GR("wqa", "wqn")} if "wqn" in EV else {}), **({"against_era5_rain_to_midnight": GR("wqa", "wqh")} if "wqh" in EV else {}),
                            control_against_rain_table_alone=GR("wqa_to09", "wq"), utc_day_against_rain_table_alone=GR("wqa_utc", "wq"), utc_day_against_matched_window_to_09=GR("wqa_utc", "wqa_to09"))
    gain_rule[REP] = dict(against_rain_table_alone=GR(REP, "wq"), against_aorc_as_an_extra_column=GR(REP, "wqa"),
                          **({"against_nldas_rain": GR(REP, "wqn")} if "wqn" in EV else {}), **({"against_era5_rain_to_midnight": GR(REP, "wqh")} if "wqh" in EV else {}))


def versus(T, C, base, ceil_, k):
    """Set T against the reference set C at lead k, in the units of the closure table: the reference's own scores, T minus C, and the share of the gap to the perfect forecast that T closes beyond C."""
    ET, EC, Eb, Ec = (EV[x]["test"] for x in (T, C, base, ceil_))
    return dict(set=f"lstm_{C}", nse_median=med(EC, k), nse_seed_range=ranges[C][lk(k)]["nse_median"], nse_gain_over_it=med(ET, k) - med(EC, k), basins_better_than_it=int((nse_of(ET, k) > nse_of(EC, k)).sum()),
                median_basin_gain_over_it=float(np.median(nse_of(ET, k) - nse_of(EC, k))), share_of_nse_gap_closed_beyond_it=div(med(ET, k) - med(EC, k), med(Ec, k) - med(Eb, k)),
                onset_warned=ons(EC, k), onset_seed_range=ranges[C][lk(k)]["onset_pod"], onset_false=ons(EC, k, "far"), onset_gain_over_it=ons(ET, k) - ons(EC, k),
                onset_share_of_gap_closed_beyond_it=div(ons(ET, k) - ons(EC, k), ons(Ec, k) - ons(Eb, k)),
                own_nse_gain_over_base=med(EC, k) - med(Eb, k), own_share_of_nse_gap_closed=div(med(EC, k) - med(Eb, k), med(Ec, k) - med(Eb, k)),
                own_onset_gain_over_base=ons(EC, k) - ons(Eb, k), own_onset_share_of_gap_closed=div(ons(EC, k) - ons(Eb, k), ons(Ec, k) - ons(Eb, k)))


def closure_rows(name, refs):
    """The closure table for a set without a one-day-older control: scores against the base set and the perfect forecast as above, and, for every reference set in refs ({label: set}), the same set against it."""
    v = VAR[name]
    base, ceil_ = v["base"], v["ceiling"]
    if not all(x in EV for x in (base, ceil_)):
        return None
    Eb, Ec, En = (EV[x]["test"] for x in (base, ceil_, name))
    rows = []
    for k in F.LEADS:
        b, c, n = med(Eb, k), med(Ec, k), med(En, k)
        ob, oc, on = (ons(E, k) for E in (Eb, Ec, En))
        rows.append(dict(lead=k, nse_base=b, nse_with=n, nse_perfect_forecast=c, gap=c - b, gain=n - b, share_of_gap_closed=div(n - b, c - b), basins_better=int((nse_of(En, k) > nse_of(Eb, k)).sum()),
                         median_basin_gain=float(np.median(nse_of(En, k) - nse_of(Eb, k))), onset_warned_base=ob, onset_warned_with=on, onset_warned_perfect_forecast=oc,
                         onset_false_base=ons(Eb, k, "far"), onset_false_with=ons(En, k, "far"), onset_false_perfect_forecast=ons(Ec, k, "far"), onset_share_of_gap_closed=div(on - ob, oc - ob),
                         seed_range_nse_with=ranges[name][lk(k)]["nse_median"], seed_range_nse_base=ranges[base][lk(k)]["nse_median"],
                         onset_seed_range_with=ranges[name][lk(k)]["onset_pod"], onset_seed_range_base=ranges[base][lk(k)]["onset_pod"],
                         references={label: versus(name, C, base, ceil_, k) for label, C in refs.items() if C in EV}))
    return dict(base=f"lstm_{base}", ceiling=f"lstm_{ceil_}", references={label: f"lstm_{C}" for label, C in refs.items() if C in EV}, by_lead=rows)


# the closure of the gap to the perfect forecast, and every control and neighbour of each set inside it (what each set closes beyond them)
if not need_main:
    closure["wqa"] = closure_rows("wqa", {"control_window_09_to_09": MAIN["wqa"], "utc_day": "wqa_utc", "era5_rain_to_midnight": "wqh", "nldas_rain": "wqn", "nldas_and_era5_together": BOTH, "aorc_replacing_the_rain_table": REP})
    for c_ in ("wqa_to09", "wqa_utc", REP):
        closure[c_] = closure_rows(c_, {"aorc_to_midnight": "wqa"})
    closure = {k_: v_ for k_, v_ in closure.items() if v_}



# ---- the readings: pairs of sets that differ in one thing ---------------------------------------------------------------------------------------------
a_test, b_test = F.SPLIT["test"]


def onset_flags(P, k):
    """Which onsets were warned: one True / False per onset of the test years (a target day above the high-flow level while the river was at or below it when the forecast was issued, as flood.flood_scores
    counts them), basin after basin in the order of ids; the same days for every set (coverage 1, asserted above)."""
    out = []
    for g in ids:
        p = P[g]
        was_low = (p.obs.shift(k) <= thr[g]).loc[a_test:b_test]
        q = p.loc[a_test:b_test]
        ok = (q.obs.notna() & q[lk(k)].notna()).values
        o, s, w = (q.obs > thr[g]).values[ok], (q[lk(k)] > thr[g]).values[ok], was_low.values[ok]
        out.append(s[o & w])
    return np.concatenate(out)


def frames_of(n):
    """The forecasts of a set by its short name (an LSTM set) or by its file name (a simple model)."""
    return PRED[n] if n in PRED else SP[n]


def test_of(n):
    return EV[n]["test"] if n in EV else ES[n]["test"]


label_of = lambda n: n if n.startswith("base_") else f"lstm_{n}"


def overlap(A, C, k=1):
    """Do two sets warn of the same onsets? Counts of the onsets warned by both, by only one, by neither; the exact sign-test p of 'only A' against 'only C' (a description: the onsets of one storm are not independent)."""
    fa, fc = onset_flags(frames_of(A), k), onset_flags(frames_of(C), k)
    ta, tc = test_of(A)["summary"][lk(k)]["onset"], test_of(C)["summary"][lk(k)]["onset"]
    assert len(fa) == len(fc) == ta["hits"] + ta["misses"] and int(fa.sum()) == ta["hits"] and int(fc.sum()) == tc["hits"], "the onsets counted here are not those of flood.evaluate"
    both, only_a, only_c = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
    return dict(first=label_of(A), second=label_of(C), lead=k, n_onsets=int(len(fa)), hits_first=int(fa.sum()), hits_second=int(fc.sum()), warned_by_both=both, warned_only_by_first=only_a, warned_only_by_second=only_c,
                warned_by_neither=int((~fa & ~fc).sum()), warned_by_at_least_one=both + only_a + only_c,
                sign_test_p=float(binomtest(only_a, only_a + only_c, 0.5).pvalue) if only_a + only_c else float("nan"))


def pair_reading(T, C):
    """Set T minus set C at every lead: onsets warned (pooled over the basins; hits, seed ranges, seed by seed: the same seed number of both sets, in order) and the median NSE with the checks of section 4.2."""
    out = {}
    for k in F.LEADS:
        r = rule(T, C, k)
        ET, EC, sT, sC = EV[T]["test"], EV[C]["test"], ranges[T][lk(k)], ranges[C][lk(k)]
        tT, tC = ET["summary"][lk(k)]["onset"], EC["summary"][lk(k)]["onset"]
        out[lk(k)] = dict(with_set=f"lstm_{T}", without_set=f"lstm_{C}",
                          onsets=dict(warned_with=tT["pod"], warned_without=tC["pod"], difference_points=100 * (tT["pod"] - tC["pod"]), hits_with=tT["hits"], hits_without=tC["hits"], onsets_in_the_test_years=tT["hits"] + tT["misses"],
                                      hits_difference=tT["hits"] - tC["hits"], seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"], every_seed_above_every_seed=r["onset"]["single_seeds_all_above"],
                                      every_seed_below_every_seed=r["onset"]["single_seeds_all_below"], difference_by_seed_points=[100 * (x - y) for x, y in zip(sT["by_seed"]["onset_pod"], sC["by_seed"]["onset_pod"])],
                                      false_warnings_with=tT["far"], false_warnings_without=tC["far"], reading=r["onset"]["reading"]),
                          nse=dict(median_with=r["nse"]["median_with"], median_without=r["nse"]["median_without"], gain=r["nse"]["gain"], seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                                   every_seed_above_every_seed=r["nse"]["single_seeds_all_above"], difference_by_seed=[x - y for x, y in zip(sT["by_seed"]["nse_median"], sC["by_seed"]["nse_median"])],
                                   n_basins_higher=r["nse"]["n_basins_higher"], n_basins=n_b, median_basin_difference=r["nse"]["median_basin_difference"], wilcoxon_p=r["nse"]["wilcoxon_p"], reading=r["nse"]["reading"]))
    return out



readings = {}
if all(x in EV for x in ("wq", "wqa", "wqa_to09", "wqa_utc")):
    readings["value_of_hours_09_to_24"] = dict(
        what="the worth of the hours 09:00 to 24:00 of day t, with the river gauge: lstm_wqa (AORC rain from 00:00 to 24:00 of day t) minus lstm_wqa_to09 (the same AORC hours over the window of the rain table itself, 09:00 of day t-1 to "
             "09:00 of day t: a second rain product that holds no hour the table does not cover, as far as the table's window is known). The two differ only in the extra column; the same seed number gives the same starting weights and the same sequence "
             "of training samples (difference_by_seed).", by_lead=pair_reading("wqa", "wqa_to09"), overlap_of_warned_onsets=overlap("wqa", "wqa_to09"))
    readings["value_of_evening_hours_beyond_utc_day"] = dict(
        what="the worth of the evening hours beyond the UTC day, with the river gauge: lstm_wqa (00:00 to 24:00 of day t) minus lstm_wqa_utc (the same AORC hours over the UTC day of day t, which ends at 00:00 UTC of day t+1, "
             f"{doc_end[0]}:00 local standard time in the Eastern basins and {doc_end[-1]}:00 in Leaf River): what a rain product served by UTC day, or IMERG-Early with its delay of about 4 hours, would hold at the issue time, "
             "so the difference is what the last hours before the issue time add to it (AORC summed over the UTC day stands in for such a product; it is not one).", by_lead=pair_reading("wqa", "wqa_utc"), overlap_of_warned_onsets=overlap("wqa", "wqa_utc"))
    readings["value_of_hours_09_to_utc_day_end"] = dict(
        what="the hours in between: lstm_wqa_utc minus lstm_wqa_to09, the worth of the hours from about 09:00 to the end of the UTC day (19:00 local standard time) of day t, "
             "so that the controls read as a ladder: rain table window -> UTC day -> whole day.", by_lead=pair_reading("wqa_utc", "wqa_to09"))
    readings["second_rain_product_alone"] = dict(
        what="what a second rain product with no new hours is worth: lstm_wqa_to09 minus lstm_wq (the rain table alone). If it is not above the seeds of lstm_wq the control is flat in the sense of the work order.",
        by_lead=pair_reading("wqa_to09", "wq"))
if all(x in EV for x in ("wqa", ERA5)):
    readings["aorc_against_era5"] = dict(
        what="AORC against ERA5 over the same hours, with the river gauge: lstm_wqa minus lstm_wqh (script 15). Both give the rain of the wall-clock day t from 00:00 to 24:00 as one extra daily input; they differ in the product only: ERA5 is model rain "
             "at the 0.25 degree grid point of the basin centroid, AORC an analysis anchored on gauges and radar, the mean of the 1 km cells inside the outline (and, in the training years before 2002, with hours that are disaggregated daily totals). "
             "So the difference is the worth of the product, not of the hours.", by_lead=pair_reading("wqa", ERA5), overlap_of_warned_onsets=overlap("wqa", ERA5))
if all(x in EV for x in ("wqa", NLDAS)):
    readings["aorc_against_nldas"] = dict(
        what="AORC to midnight against NLDAS rain, with the river gauge: lstm_wqa minus lstm_wqn (script 10). NLDAS is the gauge-based hourly analysis shipped with CAMELS; its day ends in the evening (about 17:30 local standard time as measured, "
             f"{doc_end[0]}:00 if it is stamped by the UTC day), so the two differ in the product and in the last hours of the day; for two products over about the same window see nldas_rain_against_aorc_over_the_utc_day.",
        by_lead=pair_reading("wqa", NLDAS), overlap_of_warned_onsets=overlap("wqa", NLDAS))
if all(x in EV for x in (NLDAS, "wqa_utc")):
    readings["nldas_rain_against_aorc_over_the_utc_day"] = dict(
        what="two rain products over about the same window: lstm_wqn (the NLDAS rain, whose day ends at about 17:30 local standard time as measured, at 19:00 if it is stamped by the UTC day) minus lstm_wqa_utc (the AORC rain over the UTC day, which ends at 19:00): "
             "the NLDAS day is known only to within the spread of the readings in results/day_windows.json, so the difference is mostly that of the product, a gauge-based daily analysis against an analysis made with hourly radar, and not that of the hours.",
        by_lead=pair_reading(NLDAS, "wqa_utc"), overlap_of_warned_onsets=overlap(NLDAS, "wqa_utc"))
if all(x in EV for x in ("wqa", BOTH)):
    readings["aorc_against_nldas_and_era5_together"] = dict(
        what="AORC to midnight against the two older rain products together: lstm_wqa minus lstm_wqnh (script 15: the NLDAS rain of the issue day and the ERA5 rain to midnight as two extra columns). Does one product with the right clock beat the two together?",
        by_lead=pair_reading("wqa", BOTH), overlap_of_warned_onsets=overlap("wqa", BOTH))
if all(x in EV for x in (REP, "wq", "wqa")):
    readings["rain_table_with_the_right_clock"] = dict(
        what="does a rain table with the right clock beat the mis-stamped table on its own? lstm_wqa_rep (= wqa_daymet_replaced: the Daymet rain column replaced by the AORC rain of the wall-clock day t, no extra column, the same number of inputs as lstm_wq) "
             "against lstm_wq (the Daymet rain table, whose day ends at about 09:00 local time), and against lstm_wqa (the AORC rain as an extra column beside the Daymet rain). The replaced table holds, day by day, all the hours that the table plus the extra column hold, "
             "except that the Daymet product itself is gone: against lstm_wq it is the worth of the right clock together with the change of product; against lstm_wqa it is the worth of keeping the Daymet column as well.",
        against_rain_table_alone=pair_reading(REP, "wq"), against_aorc_as_an_extra_column=pair_reading(REP, "wqa"),
        overlap_of_warned_onsets=dict(replaced_against_rain_table_alone=overlap(REP, "wq"), replaced_against_aorc_as_an_extra_column=overlap(REP, "wqa")))
if "value_of_evening_hours_beyond_utc_day" in readings and "nldas_rain_against_aorc_over_the_utc_day" in readings:       # a reading of AORC: quoted with the NLDAS result over the same window
    nu_, ev_ = readings["nldas_rain_against_aorc_over_the_utc_day"]["by_lead"]["l1"]["onsets"], readings["value_of_evening_hours_beyond_utc_day"]["by_lead"]["l1"]["onsets"]
    readings["value_of_evening_hours_beyond_utc_day"]["what"] += (
        f" THIS IS A READING OF AORC and is to be quoted together with nldas_rain_against_aorc_over_the_utc_day (and, for ERA5's own value of the same hours, with script 15's value_of_evening_hours_beyond_utc_day): the NLDAS rain, whose day ends at about the hour at which "
        f"the UTC day ends ({nld['day_ends_local_rounded']} as measured, {doc_end[0]}:00 if it is stamped by the UTC day), reaches {p1(nu_['warned_with'])} ({nu_['hits_with']} hits), the AORC rain over the UTC day {p1(nu_['warned_without'])} ({nu_['hits_without']} hits) "
        f"and the AORC rain to midnight {p1(ev_['warned_with'])} ({ev_['hits_with']} hits). So the {ev_['difference_points']:+.1f} points are what the evening hours beyond the UTC day add to AORC rain, "
        + ("and a product whose day ends about there is already at the level of the AORC rain to midnight" if nu_["hits_with"] >= ev_["hits_with"] else "and a product whose day ends about there stays below the AORC rain to midnight") + ".")
if "value_of_hours_09_to_24" in readings and "second_rain_product_alone" in readings:                                           # its companion: is the control flat?
    sp_ = readings["second_rain_product_alone"]["by_lead"]["l1"]["onsets"]
    flat_ = not sp_["every_seed_above_every_seed"]
    readings["value_of_hours_09_to_24"]["what"] += (
        f" ITS COMPANION is second_rain_product_alone (lstm_wqa_to09 minus lstm_wq: {p1(sp_['warned_with'])} against {p1(sp_['warned_without'])}, {sp_['difference_points']:+.1f} points, every seed above every seed: {yes(not flat_)}): "
        f"the control is {'flat' if flat_ else 'not flat'}, so this difference is {'the worth of the hours after about 09:00 and not that of having a second product' if flat_ else 'the worth of the hours plus part of the effect of having a second product'}. Quote the two together.")
if "aorc_against_era5" in readings and "aorc_against_nldas" in readings:                                                         # the two product comparisons belong together
    for key_, other_ in (("aorc_against_era5", "aorc_against_nldas"), ("aorc_against_nldas", "aorc_against_era5")):
        readings[key_]["what"] += f" Quote it together with {other_}: the two product comparisons are read in pairs, ERA5 over the same hours and NLDAS over the evening window."



# ---- ERA5 changes its rain on 17 July 2009: the test years before and after (a description made after the fact) ---------------------------------------
ASSIM = pd.Timestamp("2009-07-17")                 # from this day on the ERA5 analyses assimilate radar and gauge rain over the USA: Beck et al. 2019, HESS 23, 207, citing Lopez 2011 and 2013 (a literature fact read on 2026-10-07, not in ECMWF's own text)
PERIODS = {"before_2009_07_17": (F.SPLIT["test"][0], str((ASSIM - pd.Timedelta(days=1)).date())), "from_2009_07_17": (str(ASSIM.date()), F.SPLIT["test"][1])}
n_test_days = len(pd.date_range(*F.SPLIT["test"]))
by_period = dict(
    how_to_read="onsets warned 1 day ahead (seed averages, pooled over the basins) and the median NSE, scored on the target days of the test years before and from 17 July 2009 (flood.evaluate with the test part cut in two). "
                "A description made after the fact, not one of the readings and not used for any choice. It belongs to the comparison with ERA5 (aorc_against_era5, and the ERA5 sets of the ladder): ERA5's rain was made without rain observations before that day and with radar and gauge rain assimilated over the USA from it; "
                "AORC has no such change within the test years (all of them lie after 2002, the first year whose hours come from radar according to the summary of its methods in results/data_access.json), "
                "all training and validation years lie before it, and the two parts hold different numbers of onsets, so a difference between them is weak evidence unless it is large; onsets_warned_standard_error is sqrt(p(1-p)/n) of the share.",
    assimilation_of_radar_and_gauge_rain_from=str(ASSIM.date()), source="Beck et al. 2019, Hydrology and Earth System Sciences 23, 207 (citing Lopez 2011 and 2013); read on 2026-10-07; not found in ECMWF's own documentation page",
    periods={p: dict(first_target_day=a, last_target_day=b, n_days=len(pd.date_range(a, b)), share_of_the_test_days=len(pd.date_range(a, b)) / n_test_days) for p, (a, b) in PERIODS.items()}, sets={}, differences={})
for label, (a, b) in PERIODS.items():
    sp = dict(F.SPLIT, test=(a, b))
    for n in ("wq", "wqn", "wqh", "wqh_older", "wqh_to09", "wqh_utc", "wqnh", "wqa", "wqa_to09", "wqa_utc", "wqa_rep", "wqf"):
        if n in PRED:
            E = F.evaluate(PRED[n], thr, "test", 1.0, sp)
            t = E["summary"][lk(1)]["onset"]
            n_on_ = t["hits"] + t["misses"]
            by_period["sets"].setdefault(n, {})[label] = dict(onsets_warned=t["pod"], onsets_warned_standard_error=math.sqrt(t["pod"] * (1 - t["pod"]) / n_on_) if n_on_ else float("nan"), hits=t["hits"], onsets=n_on_,
                                                              false_warnings=t["far"], nse_median=med(E, 1))
for T, C in (("wqa", "wq"), ("wqa", "wqa_to09"), ("wqa", "wqa_utc"), ("wqa", "wqh"), ("wqa", "wqn"), ("wqa_rep", "wq"), ("wqh", "wq"), ("wqh", "wqh_to09")):
    if T in by_period["sets"] and C in by_period["sets"]:
        by_period["differences"][f"{T}_minus_{C}"] = {p: dict(onset_points=100 * (by_period["sets"][T][p]["onsets_warned"] - by_period["sets"][C][p]["onsets_warned"]),
                                                              hits=by_period["sets"][T][p]["hits"] - by_period["sets"][C][p]["hits"], nse=by_period["sets"][T][p]["nse_median"] - by_period["sets"][C][p]["nse_median"]) for p in PERIODS}

# the ladder of information, 1 day ahead and at every lead: one entry per rung and family

# the ladder of information, 1 day ahead and at every lead: one entry per rung; a rung's controls (hollow in the figure) are listed with it
RUNG_ENDS = [dym["day_ends_local_rounded"], nld["day_ends_local_rounded"], CLOCK[ids[0]]["last_hour_ends_local"], CLOCK[ids[0]]["last_hour_ends_local"],
             CLOCK[ids[0]]["last_hour_ends_local"], f"{nld['day_ends_local_rounded']} and {CLOCK[ids[0]]['last_hour_ends_local']}", None]
RUNG_READS = ["a day that ends in the morning (results/day_windows.json, day_ends.daymet)", "a day that ends in the evening (results/day_windows.json, day_ends.nldas)",
              "ERA5: the whole local day, midnight to midnight by the wall clock (script 15, set wqh)", "AORC: the whole local day, midnight to midnight by the wall clock, as one extra column (this script, set wqa)",
              "AORC: the whole local day in the place of the Daymet rain column, no extra column (this script, set wqa_rep = wqa_daymet_replaced)",
              "both of two columns: the NLDAS day and the ERA5 whole local day (script 15, set wqnh)", "the weather stamped t+1, t+2 and t+3 (an upper bound that cannot be had in practice)"]
KIND_TEXT = {"older": "the same rain one day older", "to09": "the same product over the window of the rain table (09:00 to 09:00): no new hour", "utc": "the same product over the UTC day (ends 19:00 standard time)"}
rung_entry = lambda n, k: {"nse_median": med(EV[n]["test"], k), "nse_seed_range": ranges[n][lk(k)]["nse_median"], "onset_warned": ons(EV[n]["test"], k),
                           "onset_seed_range": ranges[n][lk(k)]["onset_pod"], "onset_false": ons(EV[n]["test"], k, "far"), "n_seeds": ranges[n][lk(k)]["n_seeds"]}
ladder = {}
for fam, spec in FAMILY.items():
    rungs = []
    for pos, name, ctrl in spec["rungs"]:
        if name not in EV:
            continue
        rec = dict(rung=RUNG[pos], set=f"lstm_{name}", control_set=f"lstm_{ctrl}" if ctrl in EV else None, day_ends_local=RUNG_ENDS[pos], reads=RUNG_READS[pos], by_lead={}, controls={})
        cs = {c: kind for c, kind in CONTROLS.get(name, {}).items() if c in EV}
        for k in F.LEADS:
            e = rung_entry(name, k)
            if ctrl in EV:
                e.update(control_nse_median=med(EV[ctrl]["test"], k), control_nse_seed_range=ranges[ctrl][lk(k)]["nse_median"], control_onset_warned=ons(EV[ctrl]["test"], k),
                         control_onset_seed_range=ranges[ctrl][lk(k)]["onset_pod"], control_onset_false=ons(EV[ctrl]["test"], k, "far"))
            rec["by_lead"][lk(k)] = e
        for c, kind in cs.items():
            rec["controls"][f"lstm_{c}"] = dict(kind=kind, what=KIND_TEXT[kind], is_the_main_control=bool(c == ctrl), by_lead={lk(k): rung_entry(c, k) for k in F.LEADS})
        rungs.append(rec)
    ladder[fam] = dict(label=spec["label"], rungs=rungs)

# the E1 success rule of NEXT_STEPS_FOR_AGENTS.md applied to AORC (onsets warned 1 day ahead clearly above the evening-rain result, in every seed, with the control flat), and the success wording of ROADMAP.md, step 1b:
# "the E1 gain holds or grows" and "the second variant tells whether the mis-stamped table itself costs skill"
e1 = {}
if all(x in EV for x in ("wqa", "wq", "wqn", ERA5, MAIN["wqa"])):
    name, ctrl, base, nl = "wqa", MAIN["wqa"], BASE_SET, NLDAS
    r1 = {x: ranges[x][lk(1)]["onset_pod"] for x in (name, ctrl, base, nl, ERA5)}
    parts = dict(every_seed_above_every_seed_with_nldas_rain=above(r1[name], r1[nl]), every_seed_above_every_seed_of_the_control=above(r1[name], r1[ctrl]),
                 every_seed_above_every_seed_with_the_rain_table_alone=above(r1[name], r1[base]), control_flat=not above(r1[ctrl], r1[base]))
    holds = parts["every_seed_above_every_seed_of_the_control"] and parts["every_seed_above_every_seed_with_the_rain_table_alone"] and not above(r1[ERA5], r1[name])
    grows = holds and above(r1[name], r1[ERA5])
    e1[name] = dict(rule="onsets warned 1 day ahead clearly above the result with NLDAS rain (the evening rain), in every seed, with the control flat (the control is flat when its seeds are not all above "
                         "the seeds of the model with the rain table alone); the control here is the AORC rain over the table's own window (no new hour)", lead=1,
                    onset_warned=dict(with_aorc_rain_to_midnight=ons(EV[name]["test"], 1), nldas=ons(EV[nl]["test"], 1), era5_rain_to_midnight=ons(EV[ERA5]["test"], 1), control=ons(EV[ctrl]["test"], 1),
                                      rain_table_alone=ons(EV[base]["test"], 1)), seed_ranges=r1, **parts, met=bool(all(parts.values())))
    e1[name].update(every_seed_above_every_seed_with_era5_rain_to_midnight=above(r1[name], r1[ERA5]), every_seed_of_era5_rain_to_midnight_above_every_seed_with_aorc=above(r1[ERA5], r1[name]))
    if UTC_CONTROL in EV:
        rc = ranges[UTC_CONTROL][lk(1)]["onset_pod"]
        e1[name].update(control_utc_onset_warned=ons(EV[UTC_CONTROL]["test"], 1), control_utc_seed_range=rc, control_utc_flat=not above(rc, r1[base]), every_seed_above_every_seed_of_the_control_utc=above(r1[name], rc))
    e1[name]["roadmap_step_1b"] = dict(
        rule="success is that the E1 gain holds or grows. Read here as: the gain over the rain table alone and over the control with no new hours is clear (every single training above every single training of each), "
             "and the AORC rain is not clearly below the ERA5 rain to midnight of script 15 (no training with ERA5 above every training with AORC); it grows when, besides, every training with AORC is above every training with ERA5",
        gain_over_the_rain_table_alone_is_clear=parts["every_seed_above_every_seed_with_the_rain_table_alone"], gain_over_the_control_is_clear=parts["every_seed_above_every_seed_of_the_control"],
        not_clearly_below_era5=not above(r1[ERA5], r1[name]), gain_holds=bool(holds), gain_grows=bool(grows),
        onset_warned=dict(aorc=ons(EV[name]["test"], 1), era5=ons(EV[ERA5]["test"], 1), rain_table_alone=ons(EV[base]["test"], 1)), seed_ranges=dict(aorc=r1[name], era5=r1[ERA5], rain_table_alone=r1[base]))
    if REP in EV:
        rr = ranges[REP][lk(1)]["onset_pod"]
        rrule = rule(REP, base, 1)
        e1[name]["roadmap_step_1b"]["second_variant"] = dict(
            question="does a rain table with the right clock beat the mis-stamped Daymet table on its own? (lstm_wqa_rep, the Daymet rain column replaced by the AORC rain of the wall-clock day t, against lstm_wq; the clock and the product change together)",
            onset_warned=dict(replaced=ons(EV[REP]["test"], 1), rain_table_alone=ons(EV[base]["test"], 1), aorc_as_an_extra_column=ons(EV[name]["test"], 1)), seed_ranges=dict(replaced=rr, rain_table_alone=r1[base], aorc_as_an_extra_column=r1[name]),
            every_seed_above_every_seed_of_the_rain_table_alone=above(rr, r1[base]), every_seed_below_every_seed_of_the_rain_table_alone=above(r1[base], rr), nse_reading_against_the_rain_table_alone=rrule["nse"]["reading"],
            reading=(("the replaced table warns of more onsets than the rain table in every training" if above(rr, r1[base]) else
                      "the replaced table warns of fewer onsets than the rain table in every training" if above(r1[base], rr) else "no clear difference in the onsets between the replaced table and the rain table")
                     + "; the replaced table changes the product (Daymet to AORC) and the clock (a day that ends at about 09:00 to one that ends at midnight) together"
                     + (f", and the same AORC rain over the table's own window adds {readings['second_rain_product_alone']['by_lead']['l1']['onsets']['difference_points']:+.1f} points to the rain table"
                        + (", so the hours after about 09:00 and not the product carry the gain" if above(rr, r1[base]) and e1[name]["control_flat"] else "") if "second_rain_product_alone" in readings else "")
                     + "; whether the Daymet product itself costs skill apart from its clock was not isolated (AORC over 09:00 to 09:00 in the place of the Daymet column was not run)"))

# ---- the verdict of results/flood_analysis.json -----------------------------------------------------------------------------------------
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


def score(P, part, trig):
    """Warning tables pooled over the basins at one warning level per lead {lead: multiple}, as flood.evaluate pools them."""
    tabs = {g: {lk(k): F.flood_scores(P[g], thr[g], k, part, trig[k]) for k in F.LEADS} for g in P}
    return {lk(k): {kind: pool([tabs[g][lk(k)][kind] for g in P]) for kind in KINDS} for k in F.LEADS}


def sweep(P, part):
    """Pooled warning tables for every warning level of LEVELS: {lead: {kind: [table per level]}}, counted with flood.contingency on the days flood.flood_scores selects."""
    a, b = F.SPLIT[part]
    out = {}
    for k in F.LEADS:
        cells = []
        for g, p in P.items():
            low, q = (p.obs.shift(k) <= thr[g]).loc[a:b].values, p.loc[a:b]
            cells.append(((q.obs > thr[g]).values, q[lk(k)].values, low, thr[g]))
        out[lk(k)] = dict(all=[pool([F.contingency(o, f > m * t) for o, f, w, t in cells]) for m in LEVELS],
                          onset=[pool([F.contingency(o[w], (f > m * t)[w]) for o, f, w, t in cells]) for m in LEVELS])
    return out


def choose(tabs):
    """Index of the level with the best pooled CSI; a tie goes to the level closest to the high-flow level itself."""
    csi = np.array([t["csi"] if t["csi"] == t["csi"] else -1.0 for t in tabs])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])


rec_of = lambda t: dict(verdict=verdict(t), **CUT(t))


def verdict_record(P, E_test, seed_labels=None, seed_evals=None):
    """The three verdict readings of one prediction set, per lead: at the plain level (trigger 1.0) and at the two levels chosen on the validation years; with seed_evals, also the verdict of every single seed."""
    sv, plain = sweep(P, "val"), E_test["summary"]                                   # the validation years choose the level; the test years are only scored
    chosen = {}
    for label, crit in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
        trig = {k: LEVELS[choose(sv[lk(k)][crit])] for k in F.LEADS}
        chosen[label] = (trig, score(P, "test", trig))
    return {lk(k): dict(
        plain=dict(trigger=1.0, **{kind: rec_of(plain[lk(k)][kind]) for kind in KINDS}),
        **{label: dict(trigger=trig[k], chosen_by=f"best pooled validation-years CSI of {'all high-flow days' if 'all_days' in label else 'onsets'}", **{kind: rec_of(sc_[lk(k)][kind]) for kind in KINDS})
           for label, (trig, sc_) in chosen.items()},
        **({"single_seeds_plain": {kind: {str(s): verdict(e["summary"][lk(k)][kind]) for s, e in zip(seed_labels, seed_evals)} for kind in KINDS}} if seed_evals is not None else {})) for k in F.LEADS}


verdicts = {n: verdict_record(PRED[n], EV[n]["test"], list(SEEDS) if n in VAR else list(range(len(EV[n]["seeds"]))), EV[n]["seeds"]) for n in SHOWN}
VERDICT_NOTE = {"plain": "trigger 1.0: a warning when the forecast is above the high-flow level itself (the level of every other score in this file)",
                "chosen_on_validation_all_days": "the warning level chosen on the validation years by the best pooled CSI of all high-flow days: the level of the verdicts stored in results/flood_analysis.json",
                "chosen_on_validation_onsets": "the warning level chosen on the validation years by the best pooled CSI of the onsets"}

# ---- the yardstick (rule 2 of the work order): ridge regression and random forest given the AORC input, beside the LSTM ---------------------------------
SIMPLE_OLD = tuple(n for n in ("base_persistence", "base_linreg_wq", "base_rf_wq", "base_linreg_wqh", "base_rf_wqh") if F.have_preds(n))     # the stored simple models: those of 02_baselines.py (the information of lstm_wq) and those of script 15 (the same recipe with the ERA5 rain to midnight)
SP = {n: F.load_preds(n) for n in SIMPLE_OLD}
SP.update({n: SIMPLE[n]["P"] for n in YARD})
ES = {n: dict(test=F.evaluate(P, thr), val=F.evaluate(P, thr, "val")) for n, P in SP.items()}
assert all(ES[n][part]["summary"][lk(k)]["coverage_min"] == 1 for n in ES for part in ("test", "val") for k in F.LEADS), "days without a forecast: the simple models would not be compared on identical days"
assert all(np.allclose(SP[n][g].obs.loc[F.SPLIT["val"][0]:].values, D[g].q_mm.loc[F.SPLIT["val"][0]:].values) for n in SP for g in ids), "observed flow differs"
for n in ES:
    verdicts[n] = verdict_record(SP[n], ES[n]["test"])
READING = {"lstm": "the LSTM is ahead", "lstm_avg": "the LSTM is ahead only as the average of its seeds", "lstm_med": f"the LSTM scores higher, but p is above {P_CUT:g}",
           "tie": "no clear difference", "simple": "the simple model is ahead"}                         # the words of 11_key_figure.py


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
    """One LSTM set (seed average and single seeds) against one simple model on the same inputs at lead k, as 04_lstm_all.py's comparison table holds it, with the reading of section 4.2 (a rule about the NSE).
    Beside it, as a second sentence, the onsets at the plain level (hits, how many single trainings are above, the paired counts over the onsets and the comparison basin by basin) and the CSI at the warning level
    chosen on the validation years for onsets."""
    a, b = nse_of(EV[T]["test"], k), nse_of(ES[n]["test"], k)
    sm = [x for x in ranges[T][lk(k)]["by_seed"]["nse_median"]]
    ls, bs = EV[T]["test"]["summary"][lk(k)], ES[n]["test"]["summary"][lk(k)]
    bm, pr = bs["nse_median"], paired(a, b)
    c = dict(lstm_nse_median=ls["nse_median"], baseline_nse_median=bm, difference_of_medians=ls["nse_median"] - bm, n_basins_lstm_higher=pr["n_basins_higher"], n_basins=pr["n_basins"],
             median_nse_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"], lstm_seed_median_min=min(sm), lstm_seed_median_max=max(sm),
             baseline_against_seed_spread="below every seed" if bm < min(sm) else "above every seed" if bm > max(sm) else "inside the seed spread",
             n_basins_lstm_higher_by_seed=[int((nse_of(e, k) > b).sum()) for e in EV[T]["seeds"]],
             lstm_onset_pod=ls["onset"]["pod"], baseline_onset_pod=bs["onset"]["pod"], lstm_onset_far=ls["onset"]["far"], baseline_onset_far=bs["onset"]["far"], lstm_onset_pod_seed_range=ranges[T][lk(k)]["onset_pod"],
             lstm_csi=ls["all"]["csi"], baseline_csi=bs["all"]["csi"])
    r = reading_vs_simple(c)
    seeds_pod, ov = ranges[T][lk(k)]["by_seed"]["onset_pod"], overlap(T, n, k)
    pl, pb_ = (np.array([E["basins"][g][lk(k)]["onset"]["pod"] for g in ids], float) for E in (EV[T]["test"], ES[n]["test"]))
    ok = np.isfinite(pl) & np.isfinite(pb_)
    pro = paired(pl[ok], pb_[ok])
    plain = dict(lstm_warned=ls["onset"]["pod"], simple_warned=bs["onset"]["pod"], lstm_hits=ls["onset"]["hits"], simple_hits=bs["onset"]["hits"], onsets=ls["onset"]["hits"] + ls["onset"]["misses"],
                 lstm_seed_range=ranges[T][lk(k)]["onset_pod"], n_lstm_trainings=len(seeds_pod), lstm_trainings_above_the_simple_model=int(sum(x > bs["onset"]["pod"] for x in seeds_pod)),
                 every_lstm_training_above=bool(min(seeds_pod) > bs["onset"]["pod"]), every_lstm_training_below=bool(max(seeds_pod) < bs["onset"]["pod"]),
                 paired_onsets=dict(warned_by_both=ov["warned_by_both"], only_the_lstm=ov["warned_only_by_first"], only_the_simple_model=ov["warned_only_by_second"], sign_test_p=ov["sign_test_p"]),
                 basins=dict(n_basins_lstm_higher=pro["n_basins_higher"], n_basins_compared=pro["n_basins"], median_difference_points=100 * pro["median_nse_difference"], wilcoxon_p=pro["wilcoxon_p"]))
    chosen = None
    if T in verdicts and n in verdicts:
        vl, vb = verdicts[T][lk(k)]["chosen_on_validation_onsets"], verdicts[n][lk(k)]["chosen_on_validation_onsets"]
        chosen = dict(lstm_level=vl["trigger"], simple_level=vb["trigger"], lstm_warned=vl["onset"]["pod"], simple_warned=vb["onset"]["pod"], lstm_false=vl["onset"]["far"], simple_false=vb["onset"]["far"],
                      lstm_csi=vl["onset"]["csi"], simple_csi=vb["onset"]["csi"])
    small = plain["paired_onsets"]["sign_test_p"] < P_CUT
    on_read = "lstm" if plain["every_lstm_training_above"] and small else "simple" if plain["every_lstm_training_below"] and small else "tie"
    pa_ = plain["paired_onsets"]
    sentence = (f"On the onsets at the plain level the LSTM warns of {p1(plain['lstm_warned'])} ({plain['lstm_hits']} of {plain['onsets']}) against {p1(plain['simple_warned'])} ({plain['simple_hits']}); "
                f"{plain['lstm_trainings_above_the_simple_model']} of its {plain['n_lstm_trainings']} trainings are above the simple model; of the onsets warned by only one of the two, {pa_['only_the_lstm']} are the LSTM's and "
                f"{pa_['only_the_simple_model']} the simple model's (sign test, {pv(pa_['sign_test_p'])}; the onsets of one storm are not independent); the LSTM warns of more of the onsets in {plain['basins']['n_basins_lstm_higher']} of "
                f"{plain['basins']['n_basins_compared']} basins ({pv(plain['basins']['wilcoxon_p'])})"
                + (f". At the warning level chosen on the validation years for onsets the CSI is {chosen['lstm_csi']:.3f} against {chosen['simple_csi']:.3f} (levels {chosen['lstm_level']:.2f} and {chosen['simple_level']:.2f})." if chosen else "."))
    return dict(c, reading=r, reading_in_words=READING[r], onsets=dict(plain_level=plain, at_the_level_chosen_on_validation_for_onsets=chosen,
                                                                       reading={"lstm": "the LSTM warns of more onsets", "simple": "the simple model warns of more onsets", "tie": "no clear difference in the onsets"}[on_read],
                                                                       reading_rule="every single LSTM training above (or below) the simple model and a paired sign test over the onsets with p below " + f"{P_CUT:g}"), onsets_in_words=sentence)



YS = [("wqa", "base_linreg_wqa"), ("wqa", "base_rf_wqa"), ("wqa", "base_persistence"), ("wq", "base_linreg_wq"), ("wq", "base_rf_wq")]          # (the LSTM, the simple model on the same inputs)
best_simple = {lk(k): max(("base_persistence",) + YARD, key=lambda n: ES[n]["val"]["summary"][lk(k)]["nse_median"]) for k in F.LEADS}          # the best of the simple models on the new input by the validation years, as 04_lstm_all.py picks
yardstick = dict(
    what="Rule 2 of NEXT_STEPS_FOR_AGENTS.md: the new input (wqa: past weather + river gauge + the AORC rain to midnight) is also given to ridge regression and the random forest, built and fitted by the recipe of 02_baselines.py "
         "(features.feature_table for wq, plus the same rain features of the new series; one model per basin and lead, fitted on the training years only; the ridge strength and the forest settings are those "
         "02_baselines.py chose on the validation years for the input set wq and are not chosen again; the test years are only scored). The scores of lstm_wqa are read beside them in the way 04_lstm_all.py "
         "reads the LSTM against the simple models: the single seeds, the basins won, the Wilcoxon p, and the reading of report section 4.2 (11_key_figure.reading); the simple models of script 15 (the same recipe with the ERA5 rain to midnight) "
         "are in table_test, and aorc_against_era5_for_each_model sets each model's AORC result against its ERA5 result.",
    recipe=dict(code="code/features.py (feature_table) called twice, once on the daily table and once with the new rain series in the place of prcp: the hook is feature_table_with in this script; the design and the fit of code/02_baselines.py "
                     "(design, forecast) are copied here (simple_design, forecast_simple: a numbered script is not imported) and proved by checks.recipe to give the forecasts that 02_baselines.py stored for the input set wq",
                settings_source="results/baselines.json, models.base_linreg_wq.settings and models.base_rf_wq.settings", ridge=SETTINGS["base_linreg_wqa"], forest=SETTINGS["base_rf_wqa"],
                forest_trees_in_this_run=RF_TREES, features=FEATURES, n_features=len(FEATURES), n_features_of_wq=len(feature_table(D2[ids[0]], True, 0).columns),
                extra_features=[c for c in FEATURES if c.startswith("aorc_to_midnight__")],
                extra_features_how="the same rain features that every set has for prcp - rain of days t, t-1, t-2, sums over 3, 7, 14, 30 and 90 days, and the two antecedent-rain indices (decay 0.85 and 0.95) - computed on aorc_to_midnight, "
                                   "day t included (the rain of day t is complete at the issue time)"),
    checks=yard_check, fit_seconds={n: SIMPLE[n]["meta"]["fit_seconds"] for n in YARD}, fit_seconds_total=round(sum(SIMPLE[n]["meta"]["fit_seconds"] for n in YARD), 1),
    sets={n: dict(saved_as=n, test=ES[n]["test"], val=ES[n]["val"]) for n in YARD},
    stored_simple_models_used=list(SIMPLE_OLD),
    table_test={lk(k): {n: dict(nse_median=ES[n]["test"]["summary"][lk(k)]["nse_median"], onset=CUT(ES[n]["test"]["summary"][lk(k)]["onset"]), all=CUT(ES[n]["test"]["summary"][lk(k)]["all"]),
                                onset_verdict_plain=verdict(ES[n]["test"]["summary"][lk(k)]["onset"]), val_nse_median=ES[n]["val"]["summary"][lk(k)]["nse_median"]) for n in ES}
                for k in F.LEADS},
    lstm_beside={lk(k): {T: dict(nse_median=EV[T]["test"]["summary"][lk(k)]["nse_median"], nse_seed_range=ranges[T][lk(k)]["nse_median"], onset=CUT(EV[T]["test"]["summary"][lk(k)]["onset"]),
                                 onset_seed_range=ranges[T][lk(k)]["onset_pod"], onset_verdict_plain=verdict(EV[T]["test"]["summary"][lk(k)]["onset"])) for T in ("wq", ERA5, "wqa") if T in EV} for k in F.LEADS},
    best_simple_on_the_new_input_by_validation=best_simple,
    lstm_against_simple={f"lstm_{T}__{n}": {lk(k): lstm_against(T, n, k) for k in F.LEADS} for T, n in YS},
    new_input_given_to_each_model={}, aorc_against_era5_for_each_model={},
    reading_rule="11_key_figure.reading(): the LSTM is ahead only when the simple model is below every single LSTM seed, the LSTM is higher in most basins and p is below " + f"{P_CUT:g}" + "; the simple model is ahead in the mirror case; "
                 "otherwise no clear difference (or: higher but p not small)")


def rows_of(En_, Eo_):
    """One new-input comparison per lead: the median NSE, the basins won and p, and the onsets (hits, false warnings) of an evaluation with the input and one without."""
    rows = {}
    for k in F.LEADS:
        pr, tn, to = paired(nse_of(En_, k), nse_of(Eo_, k)), En_["summary"][lk(k)]["onset"], Eo_["summary"][lk(k)]["onset"]
        rows[lk(k)] = dict(nse_median_with=med(En_, k), nse_median_without=med(Eo_, k), nse_gain=med(En_, k) - med(Eo_, k), n_basins_higher=pr["n_basins_higher"], n_basins=n_b, median_basin_difference=pr["median_nse_difference"],
                           wilcoxon_p=pr["wilcoxon_p"], onset_warned_with=tn["pod"], onset_warned_without=to["pod"], onset_gain_points=100 * (tn["pod"] - to["pod"]), onset_hits_with=tn["hits"], onset_hits_without=to["hits"],
                           onsets_in_the_test_years=tn["hits"] + tn["misses"], onset_false_with=tn["far"], onset_false_without=to["far"])
    return rows


for kind, (T, new, old_) in {"lstm": ("wqa", "wqa", "wq"), "ridge regression": (None, "base_linreg_wqa", "base_linreg_wq"), "random forest": (None, "base_rf_wqa", "base_rf_wq")}.items():
    En, Eo = (EV[new]["test"], EV[old_]["test"]) if T else (ES[new]["test"], ES[old_]["test"])
    yardstick["new_input_given_to_each_model"][kind] = dict(what=f"{'lstm_wqa against lstm_wq (seed averages)' if T else new + ' against ' + old_ + ' (one fit)'}: what the new input adds to this model", by_lead=rows_of(En, Eo))
for kind, (T, new, old_) in {"lstm": ("wqa", "wqa", ERA5), "ridge regression": (None, "base_linreg_wqa", "base_linreg_wqh"), "random forest": (None, "base_rf_wqa", "base_rf_wqh")}.items():
    if (T and old_ in EV) or (not T and old_ in ES):
        En, Eo = (EV[new]["test"], EV[old_]["test"]) if T else (ES[new]["test"], ES[old_]["test"])
        yardstick["aorc_against_era5_for_each_model"][kind] = dict(what=f"{'lstm_wqa against lstm_wqh (seed averages)' if T else new + ' against ' + old_ + ' (one fit each)'}: the AORC rain to midnight against the ERA5 rain to midnight, given to the same kind of model",
                                                                    by_lead=rows_of(En, Eo))

# did the code of this script, run on the stored forecasts, reproduce what the earlier scripts stored? (not in a smoke test: its forecasts are not the stored ones)
stored = None
if not F.QUICK:
    rt_pairs = []
    for nm, c in RT["closure"].items():
        part = {x: c[x][5:] for x in ("base", "ceiling", "control")}                          # lstm_wq -> wq
        if nm in EV and all(x in EV for x in part.values()):
            for r in c["by_lead"]:
                k = r["lead"]
                rt_pairs += [(r["nse_base"], med(EV[part["base"]]["test"], k)), (r["nse_with_nldas"], med(EV[nm]["test"], k)), (r["nse_perfect_forecast"], med(EV[part["ceiling"]]["test"], k)),
                             (r["onset_warned_base"], ons(EV[part["base"]]["test"], k)), (r["onset_warned_with_nldas"], ons(EV[nm]["test"], k)),
                             (r["onset_warned_control"], ons(EV[part["control"]]["test"], k)), (r["onset_warned_perfect_forecast"], ons(EV[part["ceiling"]]["test"], k))]
    fa_pairs = []
    for n in ("wq", "w", "wqf"):
        st = FA["verdict"]["services"].get(f"lstm_{n}")
        for k in F.LEADS if st and n in verdicts else ():
            for kind in KINDS:
                s, mine = st[lk(k)][kind], verdicts[n][lk(k)]
                fa_pairs += [(s["at_plain_level"], mine["plain"][kind]["verdict"]), (s["pooled"], mine["chosen_on_validation_all_days"][kind]["verdict"]),
                             (st[lk(k)]["trigger"], mine["chosen_on_validation_all_days"]["trigger"]), (s["at_level_chosen_for_onsets"], mine["chosen_on_validation_onsets"][kind]["verdict"]),
                             (s["pod"], mine["chosen_on_validation_all_days"][kind]["pod"]), (s["far"], mine["chosen_on_validation_all_days"][kind]["far"])]
    same = lambda a, b: (a == b) or (isinstance(a, float) and isinstance(b, float) and (np.isclose(a, b, rtol=0, atol=1e-9) or (a != a and b != b)))
    stored = dict(what="the scores and verdicts of this script, computed from the stored forecasts of 04_lstm_all.py and 10_rain_timing.py, against the numbers those scripts stored",
                  rain_timing_json=dict(compared=len(rt_pairs), agree=int(sum(same(a, b) for a, b in rt_pairs))),
                  flood_analysis_json=dict(compared=len(fa_pairs), agree=int(sum(same(a, b) for a, b in fa_pairs))))
    stored["all_agree"] = all(v["compared"] == v["agree"] and v["compared"] > 0 for v in (stored["rain_timing_json"], stored["flood_analysis_json"]))
    if not stored["all_agree"]:
        notes.append("WARNING: the scores or verdicts of this script do not all equal those stored by 10_rain_timing.py / 06_flood_analysis.py for the same forecasts (checks_against_stored_results): "
                     "look into this before the numbers are used.")
        print("WARNING: checks_against_stored_results found a difference", flush=True)


# ---- what is known of the hindsight of AORC (its hours are daily totals on a gauge day that ends at 12 UTC): the builder's check, the independent review, the real-time test with Stage IV ------------
def review_hindsight():
    """The numbers of the independent review's hindsight tests, read from its results file at run time; None when the file or one of the numbers is not there."""
    if REVIEW is None:
        return None
    try:
        H_ = REVIEW["numbers"]["hindsight"]
        reg_ = H_["regressions"]["both (78 days)|A_pre~S_pre+S_post"]
        t1_ = H_["T1"]["wqa"]
        lvl_ = next(k_ for k_ in t1_ if k_.startswith("Stage IV x"))
        d3_ = next(c_ for c_ in REVIEW["checks"] if c_.get("id") == "d3")["numbers"]["sample"]
        d7_ = next(c_ for c_ in REVIEW["checks"] if c_.get("id") == "d7")["numbers"]
        return dict(n_days=len(H_["selected_days"]["days"]) + len(H_["selected_days"]["builder_days"]), n_added_days=len(H_["selected_days"]["days"]), n_basin_days=int(reg_["n"]), b2=float(reg_["b2"]),
                    b2_ci=[float(x_) for x_ in reg_["b2_ci"]], onsets_in_sample=int(t1_["AORC (as saved)"]["onsets_in_sample"]), aorc=int(t1_["AORC (as saved)"]["warned"]), stage4_raw=int(t1_["Stage IV raw"]["warned"]),
                    stage4_level=int(t1_[lvl_]["warned"]), rain_table=int(H_["T1_wq_in_sample"]), bound_points=float(d7_["loosest_bound_points"]), bound_hits=int(d7_["loosest_bound_hits"]),
                    onsets_total=int(H_["selected_days"]["onsets_total"]), night_rain_mm_builder=float(d3_["mean_stage_iv_mm_after_midnight_builder"]), night_rain_mm_added=float(d3_["mean_stage_iv_mm_after_midnight_mine"]),
                    hits_r1=int(REVIEW["numbers"]["headline_test_years_trigger_1"]["lstm_wqa"]["l1"]["onset_hits"]), pod_r1=float(REVIEW["numbers"]["headline_test_years_trigger_1"]["lstm_wqa"]["l1"]["onset_pod"]))
    except (KeyError, StopIteration, TypeError, ValueError, IndexError, AttributeError):
        return None


def builder_hindsight():
    """The numbers of the builder's own check (code/check_aorc_hindsight.py), when its results file exists."""
    if HIND is None:
        return None
    try:
        r_ = HIND["numbers"]["regression"]
        return dict(n_days=int(HIND["numbers"]["n_days"]), n_basin_days=int(HIND["numbers"]["n_basin_days"]), b2=float(r_["b2"]), b2_ci=[float(x_) for x_ in r_["b2_95_interval_bootstrap_over_days"]],
                    night_rain_mm=float(r_["mean_s_post_mm"]), day_rain_mm=float(r_["mean_s_pre_mm"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def realtime_result():
    """The result of code/check_stage4_realtime.py (Stage IV in place of AORC on the issue day of the onsets), when its results file exists."""
    if S4RT is None or S4RT.get("quick_mode"):
        return None
    try:
        m_ = S4RT["models"]["wqa"]
        v_ = m_["variants"]["stage4"]
        out_ = dict(tested=int(S4RT["onsets"]["tested"]), total=int(S4RT["onsets"]["total"]), stage4=int(v_["onsets_warned"]), aorc=int(m_["aorc"]), rain_table=int(m_["rain_table_alone"]),
                    survives=float(v_["share_of_the_aorc_effect_surviving"]), interval=[float(x_) for x_ in v_["intervals"]["over_issue_days"]["share_surviving"]])
        try:                                                                               # the delay of Stage IV (last hour out), the level of AORC restored, the controls and the fit on the onset days: quoted when they are there
            vl_, vk_, c_ = m_["variants"]["stage4_no_last_hour"], m_["variants"]["stage4_k"], m_["controls_on_the_rain_of_the_issue_day"]
            out_.update(last_hour_out=int(vl_["onsets_warned"]), last_hour_out_survives=float(vl_["share_of_the_aorc_effect_surviving"]), level_restored=int(vk_["onsets_warned"]), k=float(S4RT["k_level_aorc_over_stage4"]),
                        another_onset=float(c_["another_onset_stage4"]["mean"]), ordinary_day=float(c_["ordinary_day_aorc"]["mean"]), no_rain=float(c_["no_rain"]["mean"]),
                        mae_aorc=float(m_["reference_fits_on_the_onset_days"]["aorc"]["mae_mm_day"]), mae_stage4=float(v_["fit_on_the_onset_days"]["mae_mm_day"]),
                        mae_another_onset=float(c_["another_onset_stage4"]["fit_on_the_onset_days"]["mae_mm_day"]))
        except (KeyError, TypeError, ValueError):
            pass
        return out_
    except (KeyError, TypeError, ValueError, IndexError):
        return None


RH, BH, RT_ = review_hindsight(), builder_hindsight(), realtime_result()
RETRO = ("AORC is a retrospective analysis, not rain that could be read at midnight: its hours are daily totals on a gauge day that ends at 12 UTC, spread over the hours with radar and corrected about 10 days behind real time, "
         "and the hours before 2002, which are all the training years, are disaggregated daily totals")


def hindsight_text(th_, short=False):
    """The sentence(s) on what is known of the hindsight of AORC, for the caveat of the outcome (short) and for the note on the series, from the files that exist: the review, the builder's check, the real-time test."""
    pcx = lambda x_: f"{round(100 * x_) + 0}%" if x_ == x_ else "no value"
    out_ = ""
    if RH:
        surv_ = lambda s_: (s_ - RH["rain_table"]) / (RH["aorc"] - RH["rain_table"]) if RH["aorc"] != RH["rain_table"] else float("nan")
        more_ = len(SEEDS) > 3                                                       # round 1 of the review tested the three networks of seeds 0-2; seeds 3-4 were added on 30 October 2026
        nets_ = "the three networks of seeds 0-2 that the review tested" if more_ else "the same networks"
        lost_ = lambda n_: (RT_["aorc"] - n_, 100 * (RT_["aorc"] - n_) / RT_["total"])      # hits lost on all onsets against the AORC count of the Stage IV file, and the points of all onsets
        all_nets_ = (f"; with all {NUMBER_WORD.get(len(SEEDS), len(SEEDS))} networks, Stage IV in place of AORC on the issue day costs {lost_(RT_['stage4'])[0]} of {RT_['aorc']} hits on all {RT_['total']} onsets "
                     f"({lost_(RT_['stage4'])[1]:.1f} points; {lost_(RT_['level_restored'])[0]} hits, {lost_(RT_['level_restored'])[1]:.1f} points, with the level of AORC restored)"
                     if more_ and RT_ and RT_["tested"] == RT_["total"] and "level_restored" in RT_ else "")
        out_ = (f"An independent review (results/review_aorc_rain_to_midnight.json) tested what that could do to the test years: on {RH['n_days']} days ({RH['n_basin_days']} basin-days) the AORC rain before local midnight "
                f"does not depend on the Stage IV rain after it (coefficient {RH['b2']:+.3f} mm per mm, 95% interval {RH['b2_ci'][0]:+.3f} to {RH['b2_ci'][1]:+.3f}), and with Stage IV, a real-time product (about one hour behind), "
                f"in place of AORC on the issue day {nets_} warn of {RH['stage4_raw']} to {RH['stage4_level']} of the {RH['onsets_in_sample']} onsets of those days against {RH['aorc']} with AORC (the rain table alone: {RH['rain_table']}), "
                f"that is {pcx(surv_(RH['stage4_raw']))} to {pcx(surv_(RH['stage4_level']))} of the AORC effect; a hindsight that the review could not exclude would cost {'those networks ' if more_ else ''}at most about {RH['bound_points']:.0f} points of "
                f"{'their' if more_ else 'the'} {pcx(RH['pod_r1'] if more_ else th_['pod'])} ({RH['bound_hits']} of {RH['hits_r1'] if more_ else th_['hits']} hits); the other {RH['onsets_total'] - RH['onsets_in_sample']} onsets were not tested by the review"
                + all_nets_ + ".")
        if not short and BH:
            out_ += (f" The builder's own check (results/aorc_hindsight.json: {BH['n_days']} stormiest days, {BH['n_basin_days']} basin-days) found a coefficient of {BH['b2']:+.3f} ({BH['b2_ci'][0]:+.3f} to {BH['b2_ci'][1]:+.3f}), "
                     f"but its days hold only {BH['night_rain_mm']:.1f} mm of Stage IV rain in the 7 hours after midnight against {BH['day_rain_mm']:.1f} mm before, so its interval mostly reflects how little night rain it contains; "
                     f"the {RH['n_added_days']} days that the review added, chosen by onsets and by night storms, hold {RH['night_rain_mm_added']:.1f} mm.")
    elif BH:
        out_ = (f"Whether the hours before midnight carry a hindsight of the rain after midnight was measured only by the builder's check (results/aorc_hindsight.json: coefficient of the AORC rain before midnight on the Stage IV rain after it "
                f"{BH['b2']:+.3f}, 95% interval {BH['b2_ci'][0]:+.3f} to {BH['b2_ci'][1]:+.3f}, on {BH['n_basin_days']} basin-days of the {BH['n_days']} stormiest days, which hold only {BH['night_rain_mm']:.1f} mm of Stage IV rain in the 7 hours after midnight "
                f"against {BH['day_rain_mm']:.1f} mm before, so the interval mostly reflects how little night rain they contain); no independent review of it was found.")
    else:
        out_ = "Whether the hours before midnight carry a hindsight of the rain after midnight was not measured (neither results/aorc_hindsight.json nor results/review_aorc_rain_to_midnight.json was found)."
    if RT_:
        out_ += (f" Stage IV in place of AORC on the issue day of {'all' if RT_['tested'] == RT_['total'] else str(RT_['tested']) + ' of'} {RT_['total']} onsets (results/stage4_realtime.json): {RT_['stage4']} of {RT_['tested']} warned against "
                 f"{RT_['aorc']} with AORC and {RT_['rain_table']} with the rain table alone, {pcx(RT_['survives'])} of the AORC effect survives (95% interval {pcx(RT_['interval'][0])} to {pcx(RT_['interval'][1])}, drawing issue days)")
        if "last_hour_out" in RT_:
            out_ += (f"; with the last hour of the issue day taken out, as the delay of Stage IV would, {RT_['last_hour_out']} are warned ({pcx(RT_['last_hour_out_survives'])} of the effect) and with the level of AORC restored (Stage IV times {RT_['k']:.2f}) {RT_['level_restored']}. "
                     f"That count is driven by heavy rain on the issue day more than by its exact amount: with the Stage IV rain of another onset day of the same basin in its place {RT_['another_onset']:.0f} onsets are warned, with the AORC rain of an ordinary day "
                     f"{RT_['ordinary_day']:.0f}, with no rain {RT_['no_rain']:.0f}; the mean absolute error of the forecast on the onset days, which does depend on the exact rain, is {RT_['mae_stage4']:.2f} mm/day with Stage IV, {RT_['mae_aorc']:.2f} with AORC and "
                     f"{RT_['mae_another_onset']:.2f} with another onset's rain.")
        else:
            out_ += "."
    return out_


th_note = EV["wqa"]["test"]["summary"][lk(1)]["onset"]                                                  # the pooled onsets of the AORC set, for the note on the series
# ---- the one sentence ----------------------------------------------------------------------------------------------------------------
X = "the rain of the whole issue day up to local midnight (hourly AORC rain, 1 km, from 00:00 to 24:00 by the local wall clock) as one extra input to the LSTM with the river gauge"
b1, ctrl1 = VAR["wqa"]["base"], MAIN["wqa"]
if b1 in EV and ctrl1 in EV:
    gain_tab_ = 100 * (ons(EV["wqa"]["test"], 1) - ons(EV[b1]["test"], 1))                          # the gain in points over the rain table alone, and over the control with the same AORC rain over the table's own window
    gain_ctl_ = 100 * (ons(EV["wqa"]["test"], 1) - ons(EV[ctrl1]["test"], 1))
    outcome = (f"Given {X}, onsets warned 1 day ahead go from {pc(ons(EV[b1]['test'], 1))} with the rain table alone to {pc(ons(EV['wqa']['test'], 1))} (single trainings {span(ranges['wqa'][lk(1)]['onset_pod'])}; the onsets of one storm are not independent): "
               f"{gain_tab_:+.1f} points over the rain table alone and {gain_ctl_:+.1f} points over the control with the same AORC rain over the table's own window ({pc(ons(EV[ctrl1]['test'], 1))}); "
               f"the median NSE goes from {n2(med(EV[b1]['test'], 1))} to {n2(med(EV['wqa']['test'], 1))} (the seed-averaged forecast; single trainings {span(ranges['wqa'][lk(1)]['nse_median'], n2)}); "
               f"the verdict of results/flood_analysis.json is {verdicts['wqa']['l1']['plain']['onset']['verdict']} for the onsets and {verdicts['wqa']['l1']['plain']['all']['verdict']} for all high-flow days.")
else:
    outcome = f"No outcome: the model without the extra rain, lstm_{b1}, or its control, lstm_{ctrl1}, was not found."
NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
if b1 in EV and ctrl1 in EV and "wqa" in e1 and "aorc_against_era5" in readings and "value_of_hours_09_to_24" in readings:        # the caveat: every clause chosen from the numbers
    th = EV["wqa"]["test"]["summary"][lk(1)]["onset"]
    n_on, rg, n_s = th["hits"] + th["misses"], ranges["wqa"][lk(1)]["onset_pod"], ranges["wqa"][lk(1)]["n_seeds"]

    def share_clause(name_, label_):
        t_ = EV[name_]["test"]["summary"][lk(1)]["onset"]
        if (t_["hits"], t_["misses"]) == (th["hits"], th["misses"]):
            return f"{label_} gives the same share ({pc(th['pod'])}, {th['hits']} hits of {n_on} in both)"
        return f"{label_} gives {'a higher' if t_['pod'] > th['pod'] else 'a lower'} share ({pc(t_['pod'])}, {t_['hits']} hits of {n_on}) than the AORC rain to midnight ({pc(th['pod'])}, {th['hits']} hits of {n_on})"
    era5_clause = share_clause(ERA5, "the ERA5 rain to midnight of script 15") if ERA5 in EV else "the ERA5 rain to midnight was not found"
    nldas_clause = share_clause(NLDAS, "the NLDAS rain up to the evening") if NLDAS in EV else "the NLDAS rain was not found"
    seed_clause = (f"the {NUMBER_WORD.get(n_s, n_s)} single trainings with the AORC rain to midnight span {p1(rg[0])} to {p1(rg[1])}" if n_s > 1 and rg else
                   f"the single training with the AORC rain to midnight gives {p1(rg[0])}" if rg else "there are no single trainings to give a span")
    rule_clause = f"the success rule of the work order (clearly above the NLDAS result in every training, control flat) is {'met' if e1['wqa']['met'] else 'not met'}"
    outcome_caveat = f"{era5_clause[0].upper() + era5_clause[1:]}; {nldas_clause}; {seed_clause}; {rule_clause}."
    o1_, ov_ = readings["value_of_hours_09_to_24"]["by_lead"]["l1"]["onsets"], readings["value_of_hours_09_to_24"]["overlap_of_warned_onsets"]          # the control (the same AORC rain over the table's own window): what the gain is made of
    flat_, above_ = e1["wqa"]["control_flat"], o1_["every_seed_above_every_seed"]
    only_ = f"({ov_['warned_only_by_first']} onsets warned only by the AORC rain to midnight against {ov_['warned_only_by_second']} only by that control)"
    head_ = f"The control with the same AORC rain over the table's own window is {'flat' if flat_ else 'not flat'} ({p1(o1_['warned_without'])} against {p1(ons(EV[b1]['test'], 1))} with the rain table alone)"
    if flat_ and above_:
        tail_ = f", so the gain of {o1_['difference_points']:.1f} points over that control ({gain_tab_:.1f} points over the rain table alone) comes from the hours after about 09:00 {only_}."
    elif flat_:
        tail_ = f", so the gain of {o1_['difference_points']:.1f} points over that control ({gain_tab_:.1f} over the rain table alone) is the best estimate of what the hours after about 09:00 add, but the single trainings overlap {only_}."
    else:
        tail_ = f", so part of the gain may come from having a second rain product; the gain over that control is {o1_['difference_points']:.1f} points ({gain_tab_:.1f} over the rain table alone) {only_}."
    outcome_caveat += " " + head_ + tail_
    ae_ = readings["aorc_against_era5"]["by_lead"]["l1"]["onsets"]                                   # step 1b of the roadmap: does the E1 gain hold or grow with this rain?
    s1b_ = e1["wqa"]["roadmap_step_1b"]
    outcome_caveat += (f" Against the ERA5 rain to midnight over the same hours the AORC rain warns of {ae_['difference_points']:+.1f} points ({ae_['hits_with']} against {ae_['hits_without']} hits of {n_on}; every training with AORC above every training "
                       f"with ERA5: {yes(ae_['every_seed_above_every_seed'])}), so the E1 gain {'grows' if s1b_['gain_grows'] else 'holds' if s1b_['gain_holds'] else 'does not hold'} with it (step 1b of the roadmap).")
    ng_ = yardstick["new_input_given_to_each_model"]
    if all(k_ in ng_ for k_ in ("lstm", "ridge regression", "random forest")):                    # the yardstick: does the gain need the network?
        share_ = [yardstick["table_test"]["l1"][n_]["onset"]["pod"] for n_ in YARD]
        gain_ = [ng_[k_]["by_lead"]["l1"]["onset_gain_points"] for k_ in ("ridge regression", "random forest")]
        ya_ = [yardstick["lstm_against_simple"][f"lstm_wqa__{n_}"]["l1"] for n_ in YARD]
        far_ = [yardstick["table_test"]["l1"][n_]["onset"]["far"] for n_ in YARD]
        nse_s_ = [yardstick["table_test"]["l1"][n_]["nse_median"] for n_ in YARD]
        seed_pods_ = [ons(e_, 1) for e_ in EV["wqa"]["seeds"]]                                           # the onsets warned by each single LSTM training
        n_above_ = [sum(sp_ > share_[j_] for sp_ in seed_pods_) for j_ in range(len(YARD))]
        lstm_pod_, lstm_far_ = ya_[0]["lstm_onset_pod"], ya_[0]["lstm_onset_far"]
        names_ = ("ridge regression", "the random forest")
        ons_part_ = "; ".join(f"{names_[j_]} warns of {'more' if share_[j_] > lstm_pod_ else 'fewer' if share_[j_] < lstm_pod_ else 'the same number'} ({p1(share_[j_])} against {p1(lstm_pod_)}, {pc(far_[j_])} of its warnings false against "
                              f"{pc(lstm_far_)}; {n_above_[j_]} of the {len(seed_pods_)} LSTM trainings {'is' if n_above_[j_] == 1 else 'are'} above it)" for j_ in range(len(YARD)))
        tail2_ = "so the gain does not need the network" if min(gain_) > 0 else "so the simple models do not gain as the LSTM does"
        outcome_caveat += (f" Ridge regression and the random forest given the same input warn of {p1(share_[0])} and {p1(share_[1])} (the new input adds {gain_[0]:.1f} and {gain_[1]:.1f} points to them, "
                           f"{ng_['lstm']['by_lead']['l1']['onset_gain_points']:.1f} to the LSTM), {tail2_}. On the median NSE the LSTM ({n2(med(EV['wqa']['test'], 1))}) against ridge regression ({n2(nse_s_[0])}) and the random forest ({n2(nse_s_[1])}): "
                           f"report section 4.2 reads '{ya_[0]['reading_in_words']}' for ridge regression and '{ya_[1]['reading_in_words']}' for the forest; on the onsets, the headline, {ons_part_}.")
    if "second_variant" in s1b_:                                                                  # the second variant of step 1b: the rain table with the right clock on its own
        sv_ = s1b_["second_variant"]
        if "rain_table_with_the_right_clock" in readings and "second_rain_product_alone" in readings:
            rp_, cp_ = readings["rain_table_with_the_right_clock"], readings["second_rain_product_alone"]["by_lead"]["l1"]["onsets"]
            d_tab_, d_col_ = rp_["against_rain_table_alone"]["l1"]["onsets"], rp_["against_aorc_as_an_extra_column"]["l1"]["onsets"]
            col_reading_ = ("every training of one above every training of the other" if d_col_["every_seed_above_every_seed"] or d_col_["every_seed_below_every_seed"] else "no clear difference")
            outcome_caveat += (f" With the AORC rain in the place of the Daymet rain column (no extra column) the LSTM warns of {pc(sv_['onset_warned']['replaced'])} of the onsets against {pc(sv_['onset_warned']['rain_table_alone'])} with the Daymet "
                               f"rain table ({span(sv_['seed_ranges']['replaced'], p1)} against {span(sv_['seed_ranges']['rain_table_alone'], p1)} in the single trainings; {d_tab_['difference_points']:+.1f} points over the rain table alone"
                               f"{', more in every training' if sv_['every_seed_above_every_seed_of_the_rain_table_alone'] else ''}), but that comparison changes the product (Daymet to AORC) and the clock (a day that ends at about 09:00 to one that ends at midnight) together "
                               f"and is not a verdict on the Daymet product: the same AORC rain over the table's own window adds {cp_['difference_points']:+.1f} points to the rain table alone, over the whole day it adds {o1_['difference_points']:+.1f} points to that control, "
                               f"the replaced table adds {d_tab_['difference_points']:+.1f} points to the rain table alone, and against AORC as an extra column it adds {d_col_['difference_points']:+.1f} points ({col_reading_}); the data support that the hours after about 09:00 are worth "
                               f"about {round((o1_['difference_points'] + d_tab_['difference_points']) / 2)} points (the two gains above, averaged), and whether the Daymet product itself costs skill apart from its clock was not isolated (AORC over 09:00 to 09:00 in the place of the Daymet column was not run).")
    outcome_caveat += " " + RETRO + ". " + hindsight_text(th, short=True)
else:
    outcome_caveat = "No caveat could be computed: the sets it compares (lstm_wqa, lstm_wqa_to09, lstm_wqn, lstm_wqh, lstm_wq) were not all found."
outcome_full = outcome + " " + outcome_caveat

# ---- the one premise of the clock that rests on evidence and not on proof: that the stamps are the END of the hour ----------------------------------
def load_run(name, seed):
    """A trained network rebuilt from its file alone (as reload_and_forecast does): a flood.Run that can forecast any table with its input columns."""
    z = torch.load(MODELS / f"lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    net = F.LSTMNet(len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static), cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])


def stamp_convention_check():
    """Feed the saved networks that read aorc_to_midnight (wqa, wqa_rep) the series that results from the OPPOSITE stamp convention (the value at stamp T is the rain of the hour that STARTS at T, so the hour ends
    at T + 1 h and the wall-clock day t holds the stamps 00:00 to 23:00 of day t, one hour earlier than the main series), and score the forecasts again. A sensitivity, not a retraining: it says how much the
    result would move if the documented-by-evidence convention were wrong. Nothing is chosen from it."""
    alt = {g: rain_by_wall_clock_day(aorc_hours(g)[0] + HOUR, aorc_hours(g)[1], ZONE[UTC_OFF[g]])[0].reindex(DAYS) for g in ids}
    assert all(alt[g].notna().all() for g in ids)
    D3 = {g: D2[g].assign(aorc_to_midnight=alt[g].values) for g in ids}
    out = dict(what="the saved networks of wqa and wqa_rep given the daily series that results from the opposite stamp convention (each value would hold the first hour after midnight in place of the first hour of the day); "
                    "forecasts of the " + NUMBER_WORD.get(len(SEEDS), str(len(SEEDS))) + " seeds averaged as for the sets themselves; the test years, 1 day ahead", changed_hours_per_day="one hour in 24 (the hour 23:00 to 24:00 is replaced by the hour 00:00 to 01:00 of day t)",
               mean_abs_difference_of_the_daily_series_mm=float(np.mean([np.abs(alt[g].values - RAIN[g].values).mean() for g in ids])),
               correlation_of_the_two_daily_series=float(np.median([np.corrcoef(alt[g].values, RAIN[g].values)[0, 1] for g in ids])), sets={})
    for name in ("wqa", REP):
        ps = [load_run(name, s).predict(D3, ids, B) for s in SEEDS]
        E_alt, E_now = F.evaluate(F.ensemble(ps), thr), EV[name]["test"]
        per_seed = [F.evaluate(p, thr)["summary"][lk(1)] for p in ps]
        out["sets"][name] = dict(by_lead={lk(k): dict(onset_warned_documented_convention=ons(E_now, k), onset_warned_opposite_convention=ons(E_alt, k), difference_points=100 * (ons(E_alt, k) - ons(E_now, k)),
                                                       hits_documented=E_now["summary"][lk(k)]["onset"]["hits"], hits_opposite=E_alt["summary"][lk(k)]["onset"]["hits"],
                                                       nse_median_documented_convention=med(E_now, k), nse_median_opposite_convention=med(E_alt, k)) for k in F.LEADS},
                                 onset_warned_by_seed_opposite_convention=[s_["onset"]["pod"] for s_ in per_seed])
    return out


try:
    stamp_check = stamp_convention_check()
except FileNotFoundError:
    stamp_check = None
    notes.append("The network files were not found, so the sensitivity to the stamp convention (stamp_convention_check) was not computed.")

# ---- notes: every number in them is read from the tables above ------------------------------------------------------------------------
C0 = CLOCK[ids[0]]
hours_late = DW["day_ends"]["imerg"]["hours_late_in_this_check"]
NI = RT["stamps"]["median"]["nldas_vs_imerg"]
AD = AORC_REC or {}                                                                                  # results/aorc_data.json: the record of the download and of the checks of the series
ad_dl, ad_dec, ad_sum, ad_pl = AD.get("download", {}), AD.get("choice", {}).get("decision", {}), AD.get("summary", {}), AD.get("plan", {})
cells_lo, cells_hi = min(n_cells_by_basin.values()), max(n_cells_by_basin.values())
if ad_sum:
    from collections import Counter
    bl, hc = ad_sum["best_hourly_lag_against_era5_by_basin"], ad_sum["median_hourly_correlation_with_era5_at_lag_0"]
    lagtxt = lambda era: ", ".join(f"lag {k:+d} in {n} basins" for k, n in sorted(Counter(bl[era].values()).items()))
    prof = lambda era, k: float(np.median([AD["per_basin"][g]["hourly_against_era5_by_lag"][era][str(k)] for g in ids if g in AD.get("per_basin", {})]))
    stamp_evidence = (f"(c) the lag profile of the AORC basin-mean hourly rain against the cached ERA5 hourly rain (the hour ending at its stamp, see script 15), median of the basins: from 2002 on the correlation is {prof('from_2002', -1):.2f} at lag -1, {prof('from_2002', 0):.2f} at lag 0 "
                      f"and {prof('from_2002', 1):.2f} at lag +1 (best lag: {lagtxt('from_2002')}); before 2002 it is {prof('before_2002', -1):.2f}, {prof('before_2002', 0):.2f} and {prof('before_2002', 1):.2f} (best lag: {lagtxt('before_2002')}) "
                      "(results/aorc_data.json, per_basin.hourly_against_era5_by_lag; a lag of -1 would mean that the AORC value at stamp T matches the ERA5 value of the hour that ENDS one hour later, as a stamp at the START of the hour would, but it also holds the "
                      "difference in timing between the two products, which is large before 2002, where AORC's hours are disaggregated daily totals); ")
    s4 = AD.get("time_stamps", {}).get("stamp_check_against_stage_iv", {})

    def s4txt(key, label):
        c4 = s4[key]["pooled_correlation_by_lag"]
        return (f"{label}: {len(s4[key]['days'])} storm days ({', '.join(s4[key]['days'])}), pooled correlation {c4['-1']:.3f} at lag -1, {c4['0']:.3f} at lag 0 and {c4['1']:.3f} at lag +1, best lag {s4[key]['best_lag']:+d} "
                f"(best lag basin by basin: {', '.join(f'{k_:+d} in {n_}' for k_, n_ in sorted(Counter(s4[key]['best_lag_by_basin'].values()).items()))})")
    if s4.get("test_years", {}).get("pooled_correlation_by_lag"):
        stamp_evidence += ("(d) the stamp against the hourly Stage IV files (a file labelled T holds the hour ending at T; the radar analysis that AORC follows from 2002 on, according to the summary of its methods), the hourly basin means of both: "
                           + "; ".join(s4txt(k_, l_) for k_, l_ in (("test_years", "test years"), ("years_2003_2004", "2003-2004"), ("years_2000_2001", "2000-2001")) if s4.get(k_, {}).get("pooled_correlation_by_lag"))
                           + " (results/aorc_data.json, time_stamps.stamp_check_against_stage_iv). ")
    elif s4:
        failed_txt = "; ".join(f"{k_}: {v_.get('failed', 'no reason recorded')}" for k_, v_ in s4.items())
        stamp_evidence += f"(d) the check against hourly Stage IV files failed when it ran ({failed_txt}). "
    stamp_reading = ""
    if s4.get("test_years", {}).get("pooled_correlation_by_lag"):
        c4, b4 = s4["test_years"]["pooled_correlation_by_lag"], s4["test_years"]["best_lag"]
        stamp_reading = (f"Reading: in the test years the stamp is the END of the hour, to the resolution of the hourly Stage IV files (best lag {b4:+d}; the correlation at lag 0 is {c4['0'] - max(c4['-1'], c4['1']):.2f} higher than at the better of the two neighbouring lags). "
                         if b4 == 0 else
                         f"Reading: the check against Stage IV puts the stamp at lag {b4:+d} in the test years, NOT at the end of the hour: the proofs of the clock below hold only for a series shifted accordingly, and this run did not shift it. ")
        stamp_reading += ("Before 2002 the hours are disaggregated daily totals and no hourly radar reference exists for them (the Stage IV files of 2000-2001 do not give a sharp peak: "
                          + (f"pooled correlation {s4['years_2000_2001']['pooled_correlation_by_lag']['-1']:.2f} at lag -1, {s4['years_2000_2001']['pooled_correlation_by_lag']['0']:.2f} at lag 0, {s4['years_2000_2001']['pooled_correlation_by_lag']['1']:.2f} at lag +1"
                             if s4.get("years_2000_2001", {}).get("pooled_correlation_by_lag") else "no result") + "), so the convention of those years is inferred from the same lag profile against ERA5 before and after 2002, not shown. ")
else:
    stamp_evidence = "(c) the cross-correlation with the cached ERA5 hourly rain and the check against Stage IV were not available (results/aorc_data.json not found). "
    stamp_reading = ""
if stamp_check:
    sc_ = stamp_check["sets"]["wqa"]["by_lead"]["l1"]
    gain_w = 100 * (ons(EV["wqa"]["test"], 1) - ons(EV["wq"]["test"], 1)) if all(x in EV for x in ("wqa", "wq")) else float("nan")
    stamp_flip = (f"If the values were instead the rain of the hour AFTER their stamp, each daily value would hold the first hour after midnight in place of the first hour of the day (the daily series would change by "
                  f"{stamp_check['mean_abs_difference_of_the_daily_series_mm']:.2f} mm on average, correlation {stamp_check['correlation_of_the_two_daily_series']:.3f} with the main series): the saved networks of wqa, given the series that results "
                  f"from that convention (stamp_convention_check; a sensitivity, not a retraining), warn of {p1(sc_['onset_warned_opposite_convention'])} of the onsets 1 day ahead instead of {p1(sc_['onset_warned_documented_convention'])} "
                  f"({sc_['difference_points']:+.1f} points; median NSE {sc_['nse_median_opposite_convention']:.3f} instead of {sc_['nse_median_documented_convention']:.3f}), against a gain of {gain_w:.1f} points that the input brings over the rain table alone.")
else:
    stamp_flip = "How much the result would move if the stamps were the START of the hour was not computed (the network files were not found)."
notes += [
    f"Clock of aorc_to_midnight (the first new input; the windows of the controls follow in the note on the clock of the controls). The value of day t is the rain of the wall-clock day t at the gauge, summer time included (time zones: {zone_text}). Every hourly value goes to the local calendar day in which its hour ENDS, "
    f"and an hour that ends at exactly 24:00 belongs to the day that ends then. A day has 24 hours, or 23 or 25 where the clocks change: {DL['days_with_23_hours']:.0f} days of 23 hours, {DL['days_with_24_hours']:.0f} of 24 "
    f"and {DL['days_with_25_hours']:.0f} of 25 ({'the same in every basin' if DL_same else 'median basin'}; daily_rain.day_lengths), and summer time is in force at the end of {pc(DL['share_of_days_on_summer_time_at_the_issue_time'])} of the days. "
    f"The first hour of a day starts at {C0['first_hour_starts_local']} and the last ends at {C0['last_hour_ends_local']}, the end of day t, when the forecast is issued. The code checks it for all {n_b} basins and all "
    f"{C0['n_days']} days ({C0['first_day']} to {C0['last_day']}, no gap, asserted), exactly as 15_rain_to_midnight.py does for ERA5: (1) every hour summed into a day starts at or after 00:00 and ends at or before 24:00 of that day, by the wall clock and in absolute (UTC) time; "
    "(2) the same daily rain by a second route, in which the UTC start and end of every day come from the time-zone rules (zoneinfo) and the hours between them are summed, and every hour lies in the same day as in "
    f"the main route; (3) the rain of all hours after 24:00 of {N_CUTS} random days per basin and of the first day of 23 and of 25 hours was replaced by made-up values: no value up to that day changed (daily_rain.clock).",
    "Hour stamps. Each yearly AORC store gives every hour a time stamp, 'verification time generated by wgrib2 function verftime()', in seconds since 1970-01-01 (results/aorc_data.json: store_layout, time_stamps); this script and code/fetch_aorc.py read the time axis of "
    "every store from the store itself and assert that it is hourly, that each store starts at 00:00 on 1 January and that the stamps of one store are followed, one hour later, by those of the next, from 1980 to 2014. That the stamp is the END of the hour is not written in the "
    "metadata of the stores; it rests on (a) the wording 'verification time', which for an accumulated field is normally the end of its period (not checked here against the wgrib2 documentation), (b) the measurement of results/data_access.json: the hourly rain at the 12 centroids on 2014-01-02 correlates r = 0.991 with the Stage IV hour-ending files "
    "at lag 0 and 0.53 at lags of plus or minus one hour (a measurement of one day, made by the scouting agent, not repeated here), " + stamp_evidence + "(An oddity that fits the stamps, not an error: the first slot of each yearly store, 00:00 on 1 January, is the hour that ends at 00:00, "
    "that is the last hour of the year before.) " + stamp_reading + "The proofs of the clock in this file hold GIVEN that convention. " + stamp_flip,
    f"Summer time. One rule for all basins: the day is the wall-clock day of the gauge, summer time included, not the standard-time day. Why: results/day_windows.json (15-minute flow of the gauges, {summer_days['days'][0]} to "
    f"{summer_days['days'][1]}, while summer time was in force) found that the daily flow of {n_summer_clock} of {n_flow_gauges} gauges is a midnight-to-midnight mean by the summer clock (over the whole check, "
    f"{whole_days['days'][0]} to {whole_days['days'][1]}, standard-time weeks included, the tally is {whole_days['n_gauges_best_midnight_summer_time']} by the summer clock and "
    f"{whole_days['n_gauges_best_midnight_standard_time']} by standard time), so their river-flow day, and with it the issue time of the forecast, ends at 24:00 by the wall clock, which is 23:00 standard time. "
    f"For the other {len(std_gauges)} gauges ({'; '.join(std_gauges)}) the flow day follows standard time, so the wall-clock day ends one hour before their flow day: for them one hour of rain is left out in summer, which "
    "can only lose information and never reads anything after the issue time. In winter the wall clock is standard time. "
    f"Only {whole_days['days'][0]} to {whole_days['days'][1]} was checked, so how the flow days of the other years were cut is not known; the rule reads nothing after the issue time whether a gauge's flow day "
    "follows the wall clock or standard time. What the hour that is left out for some gauges in summer would be worth was not measured.",
    "What the series is. AORC v1.1 (NOAA, National Weather Service, Office of Water Prediction: 'Analysis Of Record for Calibration') is an hourly analysis on a 1/120 degree (about 1 km) grid, 1979 to 2025, one open Zarr store per year (AWS Open Data). "
    "As results/data_access.json records it (from a search summary of the NWS methods read on 2026-10-07; the methods document itself could not be rendered there, so this is a secondary source), the daily precipitation of NLDAS-2 / Stage IV is "
    "bias-corrected to gauge and PRISM climatology and disaggregated to hours with radar, satellite, reanalysis and GDAS data; so the hours before 2002 are disaggregated daily totals, not hourly observations, and from 2002 on they follow the radar analysis. "
    "All training years (1980-2000) lie in the first period and all test years (2005-2014) in the second, so the network learns the timing of rain within the day from a disaggregation and is scored on radar-based timing"
    + (f" (against the hourly Stage IV files the pooled hourly correlation at lag 0 is {s4['test_years']['pooled_correlation_by_lag']['0']:.2f} on storm days of the test years and {s4['years_2000_2001']['pooled_correlation_by_lag']['0']:.2f} on storm days of 2000-2001"
       f"; against the cached ERA5 hourly rain it is {hc['before_2002']:.2f} before 2002 and {hc['from_2002']:.2f} from 2002 on, median of the basins: ERA5 does not tell the two kinds of hours apart)"
       if ad_sum and s4.get("test_years", {}).get("pooled_correlation_by_lag") and s4.get("years_2000_2001", {}).get("pooled_correlation_by_lag") else "")
    + ". AORC is an analysis made after the fact (about 10 days after the hour in operation, from the same summary, unverified), not rain that anyone could have read at midnight: in real time the same hours would come from Stage IV or MRMS radar "
    "(about 1 hour behind) or from satellite rain such as IMERG-Early (about 4 hours behind). One more caveat: the daily totals that anchor the hours cover a gauge day that does not end at local midnight (12 UTC), so the value given to an hour before midnight could have been "
    "scaled with information from the rain after it; the clock proof shows which hours are summed, not that every hour's value was made without later information. That question was open when this script was first run; it has been measured since. "
    + hindsight_text(th_note) + " E1b therefore measures what gauge- and radar-anchored rain up to midnight is worth "
    "if it were known as well as AORC knows it, not what a real-time service delivers.",
    (f"The series (data/aorc, code/fetch_aorc.py). The mean of the AORC 1 km cells whose centre lies inside the basin outline (CAMELS HCDN_nhru_final_671.shp), {cells_lo} to {cells_hi} cells per basin; the area of the mask is "
     f"{min(ad_pl[g]['mask_over_outline_area'] for g in ids):.3f} to {max(ad_pl[g]['mask_over_outline_area'] for g in ids):.3f} of the outline's own, and the cells have equal area to within "
     f"{100 * max(ad_pl[g]['cell_area_variation_over_the_basin'] for g in ids):.1f}% over a basin, so the basin mean is the plain mean of the cells; ERA5 in script 15 was the value at one 0.25 degree grid point. "
     f"Download: {ad_dec.get('chosen', 'the mean of the cells inside the outline')} (estimated {ad_dec.get('estimated_gb', float('nan')):.1f} GB and {ad_dec.get('estimated_hours', float('nan')):.1f} hours at the measured speed against the limits of "
     f"{ad_dec.get('limit_gb', float('nan')):g} GB and {ad_dec.get('limit_hours', float('nan')):g} hours); {ad_dl.get('chunks_fetched', 'n/a')} chunk fetches, {ad_dl.get('gb', float('nan')):.2f} GB, {ad_dl.get('hours', float('nan')):.1f} hours of fetching "
     f"({AD.get('choice', {}).get('measured', {}).get('sample_get', {}).get('threads', 'n/a')} parallel connections; results/aorc_data.json, download). Checks of the series (results/aorc_data.json, per_basin and summary): the {AO.N_HOURS} hourly stamps of every basin are complete "
     f"({AO.SPAN_FOR_THE_RECORD}) and no value is negative. The source has holes: AORC holds the fill value in every cell of a tile for a few days here and there, and {AD['gap_fill']['total_hours_filled']} basin-hours "
     f"({100 * AD['gap_fill']['total_hours_filled'] / AD['gap_fill']['basin_hours_in_total']:.3f}% of all) in which NO cell of the basin has a value were filled with the cached ERA5 hourly rain of the basin and flagged ("
     + "; ".join(f"{F.SHORT[g]}: {v['hours_filled']} hours, " + " and ".join(f"{r_['first_stamp'][:13]} to {r_['last_stamp'][:13]}" for r_ in v["runs"]) for g, v in AD["gap_fill"]["per_basin"].items())
     + f"; every filled hour lies in the training years: {yes(all(r_['last_stamp'] < '2000-10-01' for v in AD['gap_fill']['per_basin'].values() for r_ in v['runs']))}); "
     f"{sum(r['hours_with_some_cells_missing_not_filled'] for r in AD.get('per_basin', {}).values())} further basin-hours have some cells missing and keep the mean of the cells that have a value "
     f"({sum(r['fill_cells_met'] for r in AD.get('per_basin', {}).values())} cell values with the fill value were met in them and left out). Against the other rain products, by UTC day "
     f"with the lag correlations of 10_rain_timing.py (median of the basins, same day): IMERG {ad_sum['median_daily_correlation_same_day']['imerg']:.2f}, Daymet {ad_sum['median_daily_correlation_same_day']['daymet']:.2f}, NLDAS {ad_sum['median_daily_correlation_same_day']['nldas']:.2f}, "
     f"ERA5 {ad_sum['median_daily_correlation_same_day']['era5']:.2f}; the mean rain is {ad_sum['median_ratio_aorc_to_daymet_mean_rain']:.2f} times the Daymet rain in the median basin."
     if ad_sum and ad_pl and "gap_fill" in AD else "The series (data/aorc, code/fetch_aorc.py): results/aorc_data.json was not found (or is of an older form without gap_fill), so the record of the download and of the checks is not quoted."),
    f"Day alignment. The Daymet rain stamped day d correlates {rc_med['same_day']:.2f} with the AORC rain of the local day d, {rc_med['day_before']:.2f} with that of day d-1 and "
    f"{rc_med['day_after']:.2f} with that of day d+1 (median of the {n_b} basins; daily_rain.check); the ERA5 rain of script 15 correlates {rc_era5['same_day']:.2f} with the AORC rain of the same local day ({rc_era5['day_before']:.2f} with d-1, {rc_era5['day_after']:.2f} with d+1). "
    f"A Daymet day that ends in the morning ({dym['day_ends_local_rounded']} local time, readings {dym['range_over_readings_local'][0]} to {dym['range_over_readings_local'][1]}) shares its hours with two local days, d-1 and d, and none with d+1: "
    + ("the correlations show that pattern, so the two series are aligned as described." if fits_alignment else "the correlations do NOT show that pattern: check the alignment of the series before the result is used.")
    + f" Over the whole record the AORC daily rain is {rc_ratio:.2f} times the Daymet rain in the median basin (ERA5: {rc_ratio_era5:.2f}).",
    f"Clock of the controls (aorc_to_09 and aorc_utc_day). The same hourly AORC values as aorc_to_midnight, summed over other windows by the same code, and proved for all {n_b} basins and all {C0['n_days']} days in the same way as in script 15: "
    "(1) every hour summed into day t starts at or after the start and ends at or before the end of its window, in absolute (UTC) time and on the clock of the window, and ends at or before the issue time of day t (local midnight at the end of day t, "
    "as a UTC time, and 24:00 by the wall clock); (2) the same daily rain by a second route, from the time-zone rules, and every hour lies in the same day in both routes; (3) the rain after the end of the window of "
    f"{N_CUTS} random days per basin, and of the first days of 23 and of 25 hours, was replaced by made-up values: no value up to that day changed (daily_rain.windows). The code for any window reproduces aorc_to_midnight exactly (asserted). "
    f"aorc_to_09 is the wall-clock window from 09:00 of day t-1 to 09:00 of day t, the window of the rain table itself: results/day_windows.json (day_ends.daymet) measured it to end at {dym['day_ends_local']} local standard time "
    f"(basins {dym['basin_range_local'][0]} to {dym['basin_range_local'][1]}, readings {dym['range_over_readings_local'][0]} to {dym['range_over_readings_local'][1]}), rounded to {dym['day_ends_local_rounded']}; the code uses the wall clock like the main series, "
    f"so in summer it ends one hour earlier in standard time, well inside the spread of the readings. Its last hour ends {margin_lo['aorc_to_09']:g} hours before the issue time. aorc_utc_day is the UTC day of the date of day t: from "
    f"00:00 UTC of that date to 00:00 UTC of the next date, always 24 hours, {doc_end[0]}:00 local standard time in the Eastern basins and {doc_end[-1]}:00 in Leaf River, by the wall clock between {clock_lo('aorc_utc_day', 'last_hour_ends_local_min')} and "
    f"{clock_hi('aorc_utc_day', 'last_hour_ends_local_max')}; its last hour ends {margin_lo['aorc_utc_day']:g} to {margin_hi['aorc_utc_day']:g} hours before the issue time, so the proof of nothing after the issue time holds with that margin. "
    "A rain product served by UTC day (IMERG at NASA POWER) or IMERG-Early with its delay of about 4 hours would have that UTC day at about the issue time; AORC summed over the UTC day stands in for it and is not it.",
    f"What the controls are for. Test minus control is the worth of what the test series holds and the control does not. aorc_to_09 is the same AORC product over the window of the table (as nearly as that window is known: it differs "
    f"from basin to basin and between the readings), so it holds no hour that the table does not already cover, give or take the uncertainty of that window, and lstm_wqa minus lstm_wqa_to09 is the worth of the hours 09:00 to 24:00 of day t "
    f"(value_of_hours_09_to_24), without the effect of having a second product; the Daymet rain stamped d correlates {WCHECK_MED['aorc_to_09']['daymet']['same_day']:.2f} with aorc_to_09 of the same day, against "
    f"{WCHECK_MED['aorc_to_midnight']['daymet']['same_day']:.2f} with aorc_to_midnight (median of the basins; daily_rain.check_windows). aorc_utc_day holds the hours up to the end of the UTC day, so lstm_wqa minus lstm_wqa_utc is the worth of the evening "
    f"hours beyond it (value_of_evening_hours_beyond_utc_day). The NLDAS day, as measured, is about {nld['day_ends_local_rounded']} local standard time and, if stamped by the UTC day, {doc_end[0]}:00; the NLDAS rain of the CAMELS files correlates "
    f"{WCHECK_MED['aorc_utc_day']['nldas']['same_day']:.2f} with aorc_utc_day of the same day, {WCHECK_MED['aorc_to_midnight']['nldas']['same_day']:.2f} with aorc_to_midnight and {WCHECK_MED['aorc_to_09']['nldas']['same_day']:.2f} with aorc_to_09 "
    f"(the AORC window that NLDAS resembles most: {nldas_nearest}).",
    "The replaced table (wqa_rep = wqa_daymet_replaced). The input column that holds the Daymet rain (prcp) in every other set holds aorc_to_midnight here, in the same position: the set has the same inputs as lstm_wq with one column changed, so the same seed gives "
    "the same starting weights and the same sequence of training samples, and the difference to lstm_wq is the content of the rain column (the AORC rain of the wall-clock day t, which ends at the issue time, against the Daymet day that ends at about 09:00 and "
    "belongs to another product). The daily sequence the network reads still holds every hour the table plus the extra column holds - the hours before day t sit in the earlier days - except that the Daymet product itself is gone; the comparison with lstm_wqa "
    "(AORC as an extra column beside Daymet) shows what keeping the Daymet column adds. It addresses the second question of step 1b of the roadmap (whether the mis-stamped table itself costs skill) only in part: it changes the clock and the product together, and the same AORC rain over the table's own window in the place of the Daymet column (aorc_to_09 as the only rain column) was not run.",
    f"Ladder (ladder, closure). The rungs are the rain table alone (Daymet; its day ends at about {dym['day_ends_local_rounded']}), NLDAS rain (its day ends at about {nld['day_ends_local_rounded']} on the readings "
    f"of day_windows.json, {nld['range_over_readings_local'][0]} to {nld['range_over_readings_local'][1]}; stamped by the UTC day it would end at {doc_end[0]}:00, {doc_end[-1]}:00 in the Central-time basin), "
    f"ERA5 rain to midnight (script 15), AORC rain to midnight (this script), AORC rain in the place of the rain table, the NLDAS rain and the ERA5 rain to midnight together (script 15, set wqnh), and the perfect weather forecast, an upper bound that cannot be had in practice. "
    "The sets of all rungs except the two AORC ones, and their controls, are read from the prediction files of 04_lstm_all.py, 10_rain_timing.py and 15_rain_to_midnight.py, not trained again. All sets are compared on identical days (coverage 1.0 for every model, lead and basin, "
    "asserted for the test and the validation years). The share closed is not the whole time-stamp part of the gap: the perfect-forecast sets read the Daymet weather stamped t+1 to t+3, which also holds "
    "rain still to come, and AORC, ERA5 and Daymet are different products, so the split is approximate. Rungs and controls exist for the river-gauge input only (no set without the gauge was trained in this script).",
    "Same settings and seeds as 04_lstm_all.py (results/lstm_all.json, key config): the trainings differ from its models only in the extra input column (wqa_rep: in the content of the rain column), so a difference to lstm_wq is the "
    "effect of that column (and of the training noise the seed range shows). For two sets with the same number of inputs the same seed gives the same starting weights and the same training samples, "
    "so the seed-by-seed differences in value_of_hours_09_to_24 and value_of_evening_hours_beyond_utc_day compare two trainings that start alike and differ in the extra column (the later steps differ, because the data do).",
    f"The checks (gain_rule). A gain in the median NSE is called clear when three things agree: every single seed with the new input is above every single seed of the model it is compared with (the "
    f"comparison is made with the control, with the rain table alone, with NLDAS and with ERA5); the NSE is higher in most of the {n_b} basins; and the Wilcoxon signed-rank p (two-sided, over the basins) is "
    f"below {P_CUT:g}. The onsets warned are pooled over the basins, so no basin count or p exists for them: there only the seeds are compared (as in 10_rain_timing.py). The basins share storms, so p "
    "describes and does not prove. With one seed (a smoke test) no seed range can say anything.",
    "The order of the choices (the rule of the work order: say so when test scores were seen before a change). The design of E1b follows from ROADMAP.md, step 1b, and from the controls that the review of E1 asked for; both were fixed AFTER the test-year scores of E1 "
    "(15_rain_to_midnight.py) were known, because the matched-window controls and the yardstick exist for what those scores showed. No AORC score of any kind (validation or test) was seen before the four input sets, the settings, the seeds, the stopping rule, "
    "the warning levels and the success rule were fixed, none of them was changed after the first AORC training, and every set trained is in this file. The choice of the basin mean of the cells inside the outline against the mean of the cells within 5 km of the centroid "
    "was made before the download from the measured size and speed (results/aorc_data.json, choice), without any score.",
]
ladder_of_hours = [("wq", "the rain table alone (Daymet)"), ("wqa_to09", "AORC over the window of the table (09:00 to 09:00; a second product, no new hour)"), ("wqa_utc", f"AORC over the UTC day (ends {doc_end[0]}:00 standard time)"),
                   ("wqa", "AORC to midnight"), (REP, "AORC in the place of the rain table"), (ERA5, "ERA5 to midnight (script 15)"), ("wqn", f"NLDAS rain (day ends about {nld['day_ends_local_rounded']})"),
                   (BOTH, "NLDAS rain and ERA5 to midnight together (script 15)"), ("wqf", "the perfect forecast (upper bound)")]
if all(n in EV for n, _ in ladder_of_hours):
    notes.append("The rungs side by side (with the river gauge, 1 day ahead, test years; onsets warned: seed average, lowest to highest single seed, hits of the onsets): " + "; ".join(
        f"{lab}: {p1(ons(EV[n]['test'], 1))} ({span(ranges[n][lk(1)]['onset_pod'], p1)}; {EV[n]['test']['summary'][lk(1)]['onset']['hits']} of {EV[n]['test']['summary'][lk(1)]['onset']['hits'] + EV[n]['test']['summary'][lk(1)]['onset']['misses']})"
        for n, lab in ladder_of_hours) + ". Median NSE in the same order: " + ", ".join(n3(med(EV[n]["test"], 1)) for n, _ in ladder_of_hours) + ".")
for key, T, C, head in (("value_of_hours_09_to_24", "wqa", "wqa_to09", "Value of the hours 09:00 to 24:00 of day t (value_of_hours_09_to_24)"),
                        ("value_of_evening_hours_beyond_utc_day", "wqa", "wqa_utc", "Value of the evening hours beyond the UTC day (value_of_evening_hours_beyond_utc_day)"),
                        ("aorc_against_era5", "wqa", ERA5, "AORC against ERA5 over the same hours (aorc_against_era5)"),
                        ("aorc_against_nldas", "wqa", NLDAS, "AORC to midnight against NLDAS rain (aorc_against_nldas)"),
                        ("nldas_rain_against_aorc_over_the_utc_day", NLDAS, "wqa_utc", "Two rain products over about the same window (nldas_rain_against_aorc_over_the_utc_day)"),
                        ("aorc_against_nldas_and_era5_together", "wqa", BOTH, "AORC to midnight against the NLDAS rain and the ERA5 rain to midnight together (aorc_against_nldas_and_era5_together)")):
    if key not in readings:
        continue
    r1_ = readings[key]["by_lead"]["l1"]
    o_, s_ = r1_["onsets"], r1_["nse"]
    later_on = " and ".join(f"{readings[key]['by_lead'][lk(k)]['onsets']['difference_points']:+.1f}" for k in (2, 3))
    later_nse = " and ".join(f"{readings[key]['by_lead'][lk(k)]['nse']['gain']:+.3f}" for k in (2, 3))
    ov_ = readings[key]["overlap_of_warned_onsets"]
    notes.append(
        f"{head}, with the river gauge, 1 day ahead (lstm_{T} minus lstm_{C}; test years; forecasts averaged over the seeds, lowest to highest single seed in brackets). Onsets warned: {p1(o_['warned_with'])} ({span(o_['seed_range_with'], p1)}) against "
        f"{p1(o_['warned_without'])} ({span(o_['seed_range_without'], p1)}), a difference of {o_['difference_points']:+.1f} points ({o_['hits_with']} against {o_['hits_without']} hits of {o_['onsets_in_the_test_years']} onsets; "
        f"seed by seed {', '.join(f'{x:+.1f}' for x in o_['difference_by_seed_points'])} points). Every single seed of lstm_{T} above every single seed of lstm_{C}: {yes(o_['every_seed_above_every_seed'])}; "
        f"reading of the onsets: {o_['reading']}. False warnings among the onset warnings: {pc(o_['false_warnings_with'])} against {pc(o_['false_warnings_without'])}. Median NSE: {n3(s_['median_with'])} ({span(s_['seed_range_with'], n3)}) against "
        f"{n3(s_['median_without'])} ({span(s_['seed_range_without'], n3)}), a difference of {s_['gain']:+.3f} (seed by seed {', '.join(f'{x:+.3f}' for x in s_['difference_by_seed'])}); higher in {s_['n_basins_higher']} of {n_b} basins "
        f"({pv(s_['wilcoxon_p'])}); every seed above every seed: {yes(s_['every_seed_above_every_seed'])}; reading of the NSE: {s_['reading']}. "
        f"Onsets warned by both, by only lstm_{T}, by only lstm_{C}: {ov_['warned_by_both']}, {ov_['warned_only_by_first']}, {ov_['warned_only_by_second']} "
        f"(sign test, {pv(ov_['sign_test_p'])}: a description, the onsets of one storm are not independent). 2 and 3 days ahead: {later_on} points of onsets and {later_nse} of NSE.")
if "second_rain_product_alone" in readings:
    sp_ = readings["second_rain_product_alone"]["by_lead"]["l1"]
    notes.append(f"A second rain product with no new hour (second_rain_product_alone; lstm_wqa_to09 minus lstm_wq), with the river gauge, 1 day ahead: onsets warned {p1(sp_['onsets']['warned_with'])} ({span(sp_['onsets']['seed_range_with'], p1)}) against "
                 f"{p1(sp_['onsets']['warned_without'])} ({span(sp_['onsets']['seed_range_without'], p1)}), {sp_['onsets']['difference_points']:+.1f} points; every seed above every seed of lstm_wq: {yes(sp_['onsets']['every_seed_above_every_seed'])}, so this control is "
                 f"{'not flat' if sp_['onsets']['every_seed_above_every_seed'] else 'flat'} in the sense of the work order. Median NSE {n3(sp_['nse']['median_with'])} against {n3(sp_['nse']['median_without'])} ({sp_['nse']['gain']:+.3f}; "
                 f"{sp_['nse']['n_basins_higher']} of {n_b} basins, {pv(sp_['nse']['wilcoxon_p'])}; reading {sp_['nse']['reading']}). For the hours in between, value_of_hours_09_to_utc_day_end (lstm_wqa_utc minus lstm_wqa_to09): "
                 f"{readings['value_of_hours_09_to_utc_day_end']['by_lead']['l1']['onsets']['difference_points']:+.1f} points of onsets, {readings['value_of_hours_09_to_utc_day_end']['by_lead']['l1']['nse']['gain']:+.3f} of NSE.")
if "rain_table_with_the_right_clock" in readings:
    rt_ = readings["rain_table_with_the_right_clock"]
    notes.append(
        "A rain table with the right clock on its own (rain_table_with_the_right_clock; lstm_wqa_rep = wqa_daymet_replaced), with the river gauge, 1 day ahead. " + "; ".join(
            f"against lstm_{C}: onsets warned {p1(rt_[key]['l1']['onsets']['warned_with'])} ({span(rt_[key]['l1']['onsets']['seed_range_with'], p1)}) against {p1(rt_[key]['l1']['onsets']['warned_without'])} "
            f"({span(rt_[key]['l1']['onsets']['seed_range_without'], p1)}), {rt_[key]['l1']['onsets']['difference_points']:+.1f} points ({rt_[key]['l1']['onsets']['hits_with']} against {rt_[key]['l1']['onsets']['hits_without']} hits of "
            f"{rt_[key]['l1']['onsets']['onsets_in_the_test_years']}), every seed above every seed: {yes(rt_[key]['l1']['onsets']['every_seed_above_every_seed'])}, every seed below every seed: {yes(rt_[key]['l1']['onsets']['every_seed_below_every_seed'])}; "
            f"median NSE {n3(rt_[key]['l1']['nse']['median_with'])} against {n3(rt_[key]['l1']['nse']['median_without'])} ({rt_[key]['l1']['nse']['gain']:+.3f}; {rt_[key]['l1']['nse']['n_basins_higher']} of {n_b} basins, "
            f"{pv(rt_[key]['l1']['nse']['wilcoxon_p'])}; reading {rt_[key]['l1']['nse']['reading']})" for key, C in (("against_rain_table_alone", "wq"), ("against_aorc_as_an_extra_column", "wqa"))) + ". "
        f"Onsets warned by both, only by the replaced table, only by lstm_wq: {rt_['overlap_of_warned_onsets']['replaced_against_rain_table_alone']['warned_by_both']}, {rt_['overlap_of_warned_onsets']['replaced_against_rain_table_alone']['warned_only_by_first']}, "
        f"{rt_['overlap_of_warned_onsets']['replaced_against_rain_table_alone']['warned_only_by_second']}.")
if all(n in by_period["sets"] for n in ("wq", "wqh", "wqa", "wqa_to09", "wqa_utc", "wqa_rep", "wqn", "wqnh")):
    pl = {p: by_period["periods"][p] for p in PERIODS}
    one = lambda p: "; ".join(f"{n} {p1(by_period['sets'][n][p]['onsets_warned'])} ({by_period['sets'][n][p]['hits']} of {by_period['sets'][n][p]['onsets']})" for n in ("wq", "wqn", "wqh", "wqa", "wqa_to09", "wqa_utc", "wqa_rep", "wqnh", "wqf") if n in by_period["sets"])
    dd = by_period["differences"]
    bk0, bk1 = "before_2009_07_17", "from_2009_07_17"
    notes.append(f"Before and after 17 July 2009 (by_period; a description made after the fact, with the river gauge, 1 day ahead, seed averages, onsets warned as the share of the onsets of that part of the test years and in brackets hits of onsets; "
                 f"not one of the readings above and not used for any choice; it concerns the comparison with ERA5, whose rain changes in kind on that day, see script 15). Before ({pl[bk0]['first_target_day']} to {pl[bk0]['last_target_day']}, {pl[bk0]['n_days']} days): {one(bk0)}. "
                 f"From {pl[bk1]['first_target_day']} ({pl[bk1]['n_days']} days): {one(bk1)}. Gain of the AORC rain to midnight over the rain table alone: {dd['wqa_minus_wq'][bk0]['onset_points']:+.1f} points before and {dd['wqa_minus_wq'][bk1]['onset_points']:+.1f} after; "
                 f"over the matched-window control {dd['wqa_minus_wqa_to09'][bk0]['onset_points']:+.1f} and {dd['wqa_minus_wqa_to09'][bk1]['onset_points']:+.1f}; over the UTC-day control {dd['wqa_minus_wqa_utc'][bk0]['onset_points']:+.1f} and {dd['wqa_minus_wqa_utc'][bk1]['onset_points']:+.1f}; "
                 f"over the ERA5 rain to midnight {dd['wqa_minus_wqh'][bk0]['onset_points']:+.1f} and {dd['wqa_minus_wqh'][bk1]['onset_points']:+.1f}; over NLDAS {dd['wqa_minus_wqn'][bk0]['onset_points']:+.1f} and {dd['wqa_minus_wqn'][bk1]['onset_points']:+.1f}; "
                 f"the replaced table over the rain table {dd['wqa_rep_minus_wq'][bk0]['onset_points']:+.1f} and {dd['wqa_rep_minus_wq'][bk1]['onset_points']:+.1f}. The standard error of a share is {100 * by_period['sets']['wqa'][bk0]['onsets_warned_standard_error']:.1f} points before and "
                 f"{100 * by_period['sets']['wqa'][bk1]['onsets_warned_standard_error']:.1f} after (sqrt(p(1-p)/n) for AORC to midnight). The two parts hold different numbers of onsets, so a difference between them is weak evidence unless it is large. "
                 "All test days lie in the radar period of AORC (from 2002); the training years do not.")
ys = yardstick["lstm_against_simple"]
notes.append(
    "The yardstick (rule 2 of the work order; yardstick). Ridge regression and the random forest were given the new input by the recipe of 02_baselines.py: " + f"{len(FEATURES)} features, the {yardstick['recipe']['n_features_of_wq']} of the input set wq plus "
    f"{len(yardstick['recipe']['extra_features'])} rain features computed on aorc_to_midnight in the same way as those of prcp (features.feature_table called a second time with the series in the place of prcp: the smallest hook, in this script, not in features.py); "
    f"the ridge strength {RIDGE_ALPHA} and the forest settings ({RF_SET['n_trees']} trees, leaf size {RF_SET['min_samples_leaf']}, one third of the features per split, random_state {RF_SET['random_state']}) are those that 02_baselines.py chose on the validation years for wq and were not chosen again; "
    "one model per basin and lead on the training years only. The fit took " + ", ".join(f"{n}: {yardstick['fit_seconds'][n]:g} s" for n in YARD) + ". "
    + ("Checks: " + f"this code (the design and the fit of 02_baselines.py, copied here) reproduces the stored base_linreg_wq (largest difference {yard_check['recipe']['ridge_wq_max_abs_difference_from_stored_base_linreg_wq']:.1e})"
       + (f" and base_rf_wq ({yard_check['recipe']['forest_wq_max_abs_difference_from_stored_base_rf_wq']:.1e}); " if isinstance(yard_check['recipe']['forest_wq_max_abs_difference_from_stored_base_rf_wq'], float) else "; ")
       + "the features pass the self-check of features.py on the real tables. " if yard_check else "The checks of the simple-model code were not found. ")
    + "1 day ahead, test years, median NSE and onsets warned: " + "; ".join(
        f"{n} {n3(yardstick['table_test']['l1'][n]['nse_median'])}, {p1(yardstick['table_test']['l1'][n]['onset']['pod'])}" for n in ("base_persistence", "base_linreg_wq", "base_rf_wq") + tuple(x for x in ("base_linreg_wqh", "base_rf_wqh") if x in ES) + YARD)
    + f"; lstm_wq {n3(yardstick['lstm_beside']['l1']['wq']['nse_median'])}, {p1(yardstick['lstm_beside']['l1']['wq']['onset']['pod'])}; "
    + (f"lstm_wqh {n3(yardstick['lstm_beside']['l1'][ERA5]['nse_median'])}, {p1(yardstick['lstm_beside']['l1'][ERA5]['onset']['pod'])}; " if ERA5 in yardstick['lstm_beside']['l1'] else "")
    + f"lstm_wqa {n3(yardstick['lstm_beside']['l1']['wqa']['nse_median'])} ({span(yardstick['lstm_beside']['l1']['wqa']['nse_seed_range'], n3)}), "
    f"{p1(yardstick['lstm_beside']['l1']['wqa']['onset']['pod'])} ({span(yardstick['lstm_beside']['l1']['wqa']['onset_seed_range'], p1)}). The LSTM against the simple models on the new input, by the reading of report section 4.2 (11_key_figure.reading), 1 day ahead: "
    + " ".join(f"Against {n}, by the NSE: {ys[f'lstm_wqa__{n}']['l1']['reading_in_words']} (higher in {ys[f'lstm_wqa__{n}']['l1']['n_basins_lstm_higher']} of {n_b} basins, {pv(ys[f'lstm_wqa__{n}']['l1']['wilcoxon_p'])}, the simple model is "
               f"{ys[f'lstm_wqa__{n}']['l1']['baseline_against_seed_spread']}). {ys[f'lstm_wqa__{n}']['l1']['onsets_in_words']}" for n in ("base_linreg_wqa", "base_rf_wqa")) + " What the new input adds to each model, 1 day ahead (median NSE, onsets warned): "
    + "; ".join(f"{kind} {yardstick['new_input_given_to_each_model'][kind]['by_lead']['l1']['nse_gain']:+.3f}, {yardstick['new_input_given_to_each_model'][kind]['by_lead']['l1']['onset_gain_points']:+.1f} points "
                f"(higher in {yardstick['new_input_given_to_each_model'][kind]['by_lead']['l1']['n_basins_higher']} of {n_b} basins, {pv(yardstick['new_input_given_to_each_model'][kind]['by_lead']['l1']['wilcoxon_p'])})" for kind in ("lstm", "ridge regression", "random forest"))
    + ". "
    + ("AORC against ERA5 given to each kind of model (aorc_against_era5_for_each_model; the ERA5 models are those of script 15), 1 day ahead, onsets warned and NSE: " + "; ".join(
        f"{kind} {yardstick['aorc_against_era5_for_each_model'][kind]['by_lead']['l1']['onset_gain_points']:+.1f} points ({yardstick['aorc_against_era5_for_each_model'][kind]['by_lead']['l1']['onset_hits_with']} against "
        f"{yardstick['aorc_against_era5_for_each_model'][kind]['by_lead']['l1']['onset_hits_without']} hits), {yardstick['aorc_against_era5_for_each_model'][kind]['by_lead']['l1']['nse_gain']:+.3f}" for kind in yardstick["aorc_against_era5_for_each_model"]) + ". "
       if yardstick["aorc_against_era5_for_each_model"] else "")
    + "The simple models are one fit each (the forest has one random seed), so no seed range exists for them: for them the numbers of basins and p are descriptions.")
if "wqa" in closure and "wqa" in gain_rule:
    r = closure["wqa"]["by_lead"][0]
    ref = r["references"]
    gc, gb = gain_rule["wqa"]["against_control"]["l1"], gain_rule["wqa"]["against_rain_table_alone"]["l1"]
    gn, ge = gain_rule["wqa"].get("against_nldas_rain", {}).get("l1"), gain_rule["wqa"].get("against_era5_rain_to_midnight", {}).get("l1")
    later = closure["wqa"]["by_lead"][1:]
    ORDER = [(lab, n) for lab, n in (("the rain table alone", BASE_SET), ("NLDAS rain up to the evening", NLDAS), ("the ERA5 rain to midnight", ERA5), ("the control (AORC rain over the table's own window)", MAIN["wqa"]),
                                     ("the AORC rain to midnight", "wqa"), ("the AORC rain in the place of the rain table", REP), ("the perfect forecast", "wqf")) if n in EV]
    cl_ = ref.get("control_window_09_to_09")                                                          # the control inside the closure table (the same AORC rain over the table's own window)
    parts_ = [f"{lab_}: {pc(ref[key_]['own_onset_share_of_gap_closed'])}" for key_, lab_ in (("nldas_rain", "NLDAS rain"), ("era5_rain_to_midnight", "ERA5 rain to midnight"), ("aorc_replacing_the_rain_table", "AORC in the place of the rain table")) if key_ in ref]
    refs_txt = f" ({'; '.join(parts_)})" if parts_ else ""
    notes.append(
        "Result 1 day ahead with the river gauge (test years; forecasts averaged over the seeds, lowest to highest single seed in brackets - the averaged forecast can score above every single seed). Onsets warned: "
        + ", ".join(f"{p1(ons(EV[n]['test'], 1))} ({span(ranges[n][lk(1)]['onset_pod'], p1)}) with {lab}" for lab, n in ORDER)
        + f": the AORC rain to midnight closes {pc(r['onset_share_of_gap_closed'])} of the gap between the rain table and the perfect forecast"
        + (f", {pc(cl_['onset_share_of_gap_closed_beyond_it'])} beyond the control" if cl_ else "")
        + refs_txt
        + f". Every single seed with it is above every seed of the control: {yes(gc['onset']['single_seeds_all_above'])}; above every seed with the rain table alone: {yes(gb['onset']['single_seeds_all_above'])}"
        + (f"; above every seed with NLDAS rain: {yes(gn['onset']['single_seeds_all_above'])}" if gn else "") + (f"; above every seed with the ERA5 rain to midnight: {yes(ge['onset']['single_seeds_all_above'])}" if ge else "")
        + ". False warnings among the onset warnings, in the same order: " + ", ".join(pc(ons(EV[n]["test"], 1, "far")) for lab, n in ORDER)
        + ". Median NSE, in the same order: " + ", ".join(f"{n3(med(EV[n]['test'], 1))} ({span(ranges[n][lk(1)]['nse_median'], n3)})" for lab, n in ORDER)
        + f": the AORC rain to midnight closes {pc(r['share_of_gap_closed'])} of the NSE gap" + (f", {pc(cl_['share_of_nse_gap_closed_beyond_it'])} beyond the control" if cl_ else "")
        + f". Basin by basin the NSE is higher than with the rain table alone in {r['basins_better']} of {n_b} basins ({pv(gb['nse']['wilcoxon_p'])}; reading: {gb['nse']['reading']}) and higher than with the control in "
        + (f"{cl_['basins_better_than_it']} ({pv(gc['nse']['wilcoxon_p'])}; reading: {gc['nse']['reading']})" if cl_ else "n/a") + f"; median basin difference {r['median_basin_gain']:+.3f}"
        + (f" and {cl_['median_basin_gain_over_it']:+.3f}" if cl_ else "") + ": the difference of two medians and the median of the basin differences need not agree. 2 and 3 days ahead: "
        f"{' and '.join(pc(x['share_of_gap_closed']) for x in later)} of the NSE gap and {' and '.join(pc(x['onset_share_of_gap_closed']) for x in later)} of the onset-warning gap.")
for name, x in e1.items():
    sv_ = x["roadmap_step_1b"].get("second_variant")
    notes.append(f"Success rule of NEXT_STEPS_FOR_AGENTS.md (E1) applied to AORC, with the river gauge: {x['rule']}. Onsets warned 1 day ahead: {p1(x['onset_warned']['with_aorc_rain_to_midnight'])} "
                 f"with the AORC rain up to midnight, {p1(x['onset_warned']['nldas'])} with NLDAS rain, {p1(x['onset_warned']['era5_rain_to_midnight'])} with the ERA5 rain to midnight, {p1(x['onset_warned']['control'])} with the control, "
                 f"{p1(x['onset_warned']['rain_table_alone'])} with the rain table alone. Every seed above the NLDAS result: {yes(x['every_seed_above_every_seed_with_nldas_rain'])}; above the control: {yes(x['every_seed_above_every_seed_of_the_control'])}; "
                 f"above the rain table alone: {yes(x['every_seed_above_every_seed_with_the_rain_table_alone'])}; above the ERA5 result: {yes(x['every_seed_above_every_seed_with_era5_rain_to_midnight'])}; control flat: {yes(x['control_flat'])}. "
                 f"Rule met: {yes(x['met'])}. Step 1b of ROADMAP.md: {x['roadmap_step_1b']['rule']}. Gain over the rain table alone clear: {yes(x['roadmap_step_1b']['gain_over_the_rain_table_alone_is_clear'])}; over the control clear: "
                 f"{yes(x['roadmap_step_1b']['gain_over_the_control_is_clear'])}; not clearly below ERA5: {yes(x['roadmap_step_1b']['not_clearly_below_era5'])}; holds: {yes(x['roadmap_step_1b']['gain_holds'])}; grows: {yes(x['roadmap_step_1b']['gain_grows'])}."
                 + (f" Second variant (a rain table with the right clock on its own; the clock and the product change together): {p1(sv_['onset_warned']['replaced'])} of the onsets with the AORC rain in the place of the Daymet rain against {p1(sv_['onset_warned']['rain_table_alone'])} with the Daymet rain table "
                    f"({span(sv_['seed_ranges']['replaced'], p1)} against {span(sv_['seed_ranges']['rain_table_alone'], p1)}), NSE reading {sv_['nse_reading_against_the_rain_table_alone']}: {sv_['reading']}." if sv_ else ""))
vq = verdicts["wqa"]["l1"] if "wqa" in verdicts else None
if vq:
    other = {k: vq[k]["onset"]["verdict"] for k in ("chosen_on_validation_all_days", "chosen_on_validation_onsets")}
    notes.append(f"Verdict (verdicts): the criteria stored in results/flood_analysis.json (GO: {CRIT['GO']}; CAUTION: {CRIT['CAUTION']}; NOT YET: {CRIT['NOT YET']}) are applied to the warning tables pooled over the {n_b} basins, "
                 f"for the onsets and for all high-flow days, at three warning levels: the plain one (trigger 1.0; the one the sentence in `outcome` uses, as the shares of onsets warned do), and the two levels "
                 f"chosen on the validation years (as 06_flood_analysis.py does; the first of them is the level of the stored verdicts). With the river gauge and the AORC rain up to midnight the onset verdict 1 day "
                 f"ahead is {vq['plain']['onset']['verdict']} at the plain level, {other['chosen_on_validation_all_days']} at the level chosen for all high-flow days (trigger "
                 f"{vq['chosen_on_validation_all_days']['trigger']:.2f}) and {other['chosen_on_validation_onsets']} at the level chosen for onsets (trigger {vq['chosen_on_validation_onsets']['trigger']:.2f}).")
notes += [
    f"Fair test: nothing is chosen on the test years. The settings were chosen by 04_lstm_all.py on the validation years, the trainer picks each stopping point on the validation years, the warning levels "
    f"of the second and third verdict were chosen on the validation years, and the test years ({F.SPLIT['test'][0][:4]}-{F.SPLIT['test'][1][:4]}) are only scored. The validation scores (validation_used_for_no_choice) "
    "are shown apart and chose nothing in this script; they are not an independent measure either, because the trainer used those years to stop each training. Nothing in this script reads a test "
    "score before a choice is made; the scores printed after each training in the log are test-year scores and chose nothing. If anyone looks at test scores and then changes the inputs, the settings or the code "
    "and runs again, the file must say so (see the note 'The order of the choices').",
    "Models saved (results/models/lstm_<set>_s<seed>.pt, one per training): the weights, the settings (cfg), the scalers, the basin ids, the input column names, the split and the training record. "
    "reload_check rebuilds one network per input set from its file alone and shows that it forecasts the stored forecasts again on the same tables. To forecast a new day or river, the daily columns in "
    "dyn must be built the same way as here (the aorc_to_midnight series needs the hourly AORC rain of the whole day) and the river flow and basin properties are needed; nothing has to be trained again.",
]
if stored:
    notes.append(f"Check of this script's own scoring code against the stored numbers of the earlier scripts (checks_against_stored_results): {stored['rain_timing_json']['agree']} of "
                 f"{stored['rain_timing_json']['compared']} numbers of 10_rain_timing.py (median NSE and onsets warned of the rungs it holds) and {stored['flood_analysis_json']['agree']} of "
                 f"{stored['flood_analysis_json']['compared']} verdict numbers of 06_flood_analysis.py agree.")
if F.QUICK:
    notes.append("QUICK smoke test: a few updates and one seed - numbers are not results. The sets of the ladder come from the quick prediction files where there are any and from the real ones otherwise.")
if RESCORE:
    timing["rescore_runtime_s"] = round(time.time() - t_start, 1)
    this_pass["runtime_s"], this_pass["finished_local_time"] = timing["rescore_runtime_s"], time.strftime("%Y-%m-%d %H:%M")
    notes.append("This file was written by --rescore: every score, table and note and the figure were recomputed from the saved prediction sets lstm_<set>_s<seed> (and base_linreg_wqa, base_rf_wqa) without training again; the training "
                 "records in models.<set>.per_seed, training, n_trainings, training_seconds and runtime_s are those of the training runs; the reload check was run again on the saved networks "
                 "unless reload_check says it was carried over. rescore_runtime_s is the time of this pass. passes lists the training passes and, last, this rescore, each with the sha256 of the script that ran it.")
else:
    this_pass["runtime_s"] = round(time.time() - t_start - waited, 1)
    this_pass["finished_local_time"] = time.strftime("%Y-%m-%d %H:%M")
    timing["runtime_s"] = round(sum(p["runtime_s"] for p in timing["passes"] if p.get("kind") != "rescore"), 1)
    if EXTEND:
        notes.append(f"This file was written by --extend: {this_pass['n_trainings']} networks were trained now (the sets whose seed files were missing: "
                     f"{', '.join(sorted({n for n, _ in trained_pairs})) or 'none'}), {'the simple models were fitted now' if missing_simple else 'the simple models were read from their saved forecasts'}, and every other set was read from its saved files, "
                     "never trained again and not written again (integrity holds the sha256 of every saved file and says which files this pass made; the files the earlier results file recorded are compared with "
                     "their hashes and the script stops if one differs). Every score, table, note and the figure were recomputed from all saved sets. The training records of the earlier sets are carried over from the results file of the earlier run; passes lists each pass (n_trainings, training_seconds, runtime_s: this pass and the earlier ones, summed in the "
                     "fields of the same name at the top of the file).")
notes.append("runtime_s leaves out the time the script waited for the CPU (waited_for_cpu_s); training_seconds is the time of the trainings alone." if "waited_for_cpu_s" in timing else
             "runtime_s is the time of the training run from start to end.")
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
        review_item = (review_sentence(review_rounds(REVIEW, "code/16_aorc_rain_to_midnight.py"), "results/review_aorc_rain_to_midnight.json", len(SEEDS), HASH_NOW)
                       + " The reviews were written by an AI agent, not by a team member: the team's own check (team.json) is still open.")
    except (KeyError, TypeError):
        review_item = "A file results/review_aorc_rain_to_midnight.json exists but its counts could not be read here; read it directly. It was written by an AI agent, not by a team member: the team's own check (team.json) is still open."
else:
    review_item = ("No independent recomputation of the headline numbers by a second agent was found (results/review_aorc_rain_to_midnight.json does not exist); the work order requires one (sections 3 and 6): it covers the clock proofs, the saved networks, "
                   "the download and reduction of the AORC series (code/fetch_aorc.py) and the numbers of this file. The team's own check (team.json) is open as well.")
span_of = {n_: ranges[n_][lk(1)]["nse_median"][1] - ranges[n_][lk(1)]["nse_median"][0] for n_ in VAR}
wide = max(span_of, key=span_of.get)
wide_val = [costs[wide][s_]["best_val_nse"] for s_ in SEEDS]
nse09 = readings["value_of_hours_09_to_24"]["by_lead"]["l1"]["nse"] if "value_of_hours_09_to_24" in readings else None
def filled_in(first, last):
    """Basin-hours of the AORC holes (filled from ERA5) that lie between two dates, the last day included (the stamps of the filled runs are in results/aorc_data.json, gap_fill)."""
    lo, hi = pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(days=1)
    n = 0
    for v_ in AD.get("gap_fill", {}).get("per_basin", {}).values():
        for r_ in v_["runs"]:
            a_, b_ = max(pd.Timestamp(r_["first_stamp"]), lo), min(pd.Timestamp(r_["last_stamp"]), hi)
            n += max(0, int((b_ - a_) / HOUR) + 1)
    return n


NOT_DONE = [
    review_item,
    "The ranges of code/13_uncertainty.py (test years and basins drawn again) are not extended to the new prediction sets (lstm_wqa, lstm_wqa_to09, lstm_wqa_utc, lstm_wqa_rep, base_linreg_wqa, base_rf_wqa); "
    "the seed ranges and the checks of gain_rule are all this file holds.",
    "Only the model WITH the river gauge was run: wa, wa_to09, wa_utc and wa_rep (without the gauge, where the rain should matter more) do not exist, so what the AORC hours and the table with the right clock are worth to the model without the river gauge is not known.",
    f"More seeds where the NSE decides: the median NSE of the single trainings of a set spans up to {span_of[wide]:.3f} (lead 1, lstm_{wide}: {ranges[wide][lk(1)]['nse_median'][0]:.3f} to {ranges[wide][lk(1)]['nse_median'][1]:.3f}), while the validation NSE "
    f"of the same trainings spans only {min(wide_val):.3f} to {max(wide_val):.3f}, so the validation years cannot tell a good training from a weak one. With {NUMBER_WORD.get(len(SEEDS), len(SEEDS))} seeds the NSE rule of report section 4.2 (every single seed above every single seed, higher "
    "in most basins, p below 0.05) can hardly ever call a gain of this size clear"
    + (f": value_of_hours_09_to_24 is {nse09['gain']:+.3f} in NSE, higher in {nse09['n_basins_higher']} of {n_b} basins ({pv(nse09['wilcoxon_p'])}), and its reading is '{nse09['reading']}'. " if nse09 else ". ")
    + "The onsets warned, pooled over the basins, are decided by the seeds more easily.",
    (f"AORC has holes: {AD['gap_fill']['total_hours_filled']} basin-hours in which no cell of the basin holds a value were filled with the cached ERA5 hourly rain (results/aorc_data.json, gap_fill; "
     f"{filled_in(*F.SPLIT['test'])} of them lie in the test years). What these hours do to the scores was not measured; they are a tiny share of the hours "
     f"({100 * AD['gap_fill']['total_hours_filled'] / AD['gap_fill']['basin_hours_in_total']:.3f}% of the basin-hours of the record)." if AD.get("gap_fill", {}).get("total_hours_filled") else
     "No hour had to be filled in the AORC series (or results/aorc_data.json was not found)."),
    "The AORC series are the mean of the cells inside the basin outline for every basin (the choice is recorded in results/aorc_data.json); the mean of the cells within 5 km of the centroid, a point value, or the rain over the basin's upstream cells only were not tried, and "
    "the 12 basins are not weighted by anything but the cells (equal area to within 1%). The grid of the stores is the nominal 1/120 degree grid; the stored coordinate arrays are rounded and drift from it by up to 0.003 degree (a fraction of one cell, results/aorc_data.json, grid); "
    + ((lambda dr: f"the basin means of {dr['year']} recomputed with the cells chosen on the stored coordinates instead (results/aorc_data.json, grid_drift_check) correlate at least {dr['lowest_hourly_correlation']:.4f} hourly with the means used, their sums differ by at most "
                   f"{100 * dr['largest_difference_of_the_ratio_from_1']:.2f}% and no hourly value by more than {dr['largest_abs_hourly_difference_mm']:.2f} mm; the other years were not recomputed.")(AD["grid_drift_check"])
       if AD.get("grid_drift_check", {}).get("per_basin") else "the effect of using them instead was not measured."),
    "No other windows than the three tried (the whole day, the window of the rain table, the UTC day) and no other hours (for example rain up to the early evening in steps, to see how the gain grows with the hours); no other rain product than ERA5, NLDAS and AORC "
    "(IMERG-Early, Stage IV and MRMS were not tried as the training input; Stage IV was substituted for AORC at forecast time by the review and by code/check_stage4_realtime.py); AORC summed over the UTC day is not IMERG.",
    "The hindsight of AORC's daily anchoring (hours before midnight scaled by a gauge day that includes later rain) was measured by results/aorc_hindsight.json (the builder's check, days with little night rain)"
    + (f" and by the independent review ({RH['n_days']} days; Stage IV in place of AORC on the issue day of {RH['onsets_in_sample']} of the {RH['onsets_total']} onsets)" if RH else " (no independent review was found)")
    + (", and by results/stage4_realtime.json for the issue day of " + ("every onset" if RT_["tested"] == RT_["total"] else f"{RT_['tested']} of the {RT_['total']} onsets") if RT_ else
       ("; not for the other " + str(RH["onsets_total"] - RH["onsets_in_sample"]) + " onsets (code/check_stage4_realtime.py does that when it has been run)" if RH else ""))
    + "; not measured: the false warnings on days without any onset (except those of the issue days of all basins in results/stage4_realtime.json), the training years' hours (disaggregated daily totals), and the monthly bias correction of AORC as a channel of its own.",
    "AORC's hours before 2002 are disaggregated daily totals and all training years lie before 2002 (all test years after): a network trained on radar-era hours (from 2002 on) would need a training period of its own, which this split does not have; "
    "this was not tried. What the hours of a daily-total disaggregation can teach the network about the timing of rain within the day was not measured separately.",
    "For the gauges whose flow day follows standard time in summer (results/day_windows.json), the wall-clock day ends one hour before their flow day, so one hour of rain is left out of the new input for them in "
    "summer; what that hour would add was not measured, and the flow-day check covers only part of one year.",
    "The settings of the LSTM (hidden size, learning rate, dropout, budget) were not tuned again for the new inputs: they are those chosen by 04_lstm_all.py for the other input sets. The simple models use the settings chosen by 02_baselines.py for wq; "
    "each was fitted once (the forest has one random seed), so no seed range exists for them.",
    f"No leave-one-basin-out test, no lead other than {F.LEADS[0]} to {F.LEADS[-1]} days, no peaks-and-onsets loss (E5), no change of the network (E6).",
    "The ladder sets of the earlier scripts (lstm_wq, lstm_wqf, lstm_wqn and its control, the four ERA5 sets, lstm_wqnh) were not trained again; they were scored here with the same code as the new sets. Their networks were "
    "never saved, except those of script 15, so only the sets of scripts 15 and 16 can forecast again without training.",
    "Problem P2 of NEXT_STEPS_FOR_AGENTS.md also asks for a script that loads the saved networks and forecasts a new day or river; none was written. reload_check shows that the files are enough to forecast again "
    "on the tables of this project.",
    "No report subsection, no row in the key figure, and no update of PLAN.md, README.md or ROADMAP.md (NEXT_STEPS_FOR_AGENTS.md, sections 3 and 6): documents must quote outcome_full, not outcome alone.",
] + (["QUICK smoke test: one seed, 40 updates per training: nothing here is a result."] if F.QUICK else [])
INPUTS = dict(
    how_to_read="the hours each input covers, as the model reads them on issue day t. The forecast is issued at the end of day t, local midnight of the river-flow day",
    clock=[dict(input="prcp (Daymet rain, the rain table), srad, tmax, tmin, vp, dayl", used_by="every set (in wqa_rep the rain column prcp is replaced by aorc_to_midnight)", covers=f"for rain, a day that ends at about {dym['day_ends_local_rounded']} local standard time on day t "
                f"(basins {dym['basin_range_local'][0]} to {dym['basin_range_local'][1]}; the four readings {dym['range_over_readings_local'][0]} to {dym['range_over_readings_local'][1]}); the window of the other "
                "Daymet variables was not measured", source="results/day_windows.json, day_ends.daymet"),
           dict(input="q_mm (observed river flow, with-gauge sets only)", used_by="every set of this script, wq, wqn, wqh, wqnh, wqf and their controls", covers=f"the daily mean flow of local day t, midnight to midnight by the local clock, so the day ends at the issue time (in the 2012 check, while summer time was in force, by the summer clock at "
                f"{n_summer_clock} of {n_flow_gauges} gauges and by standard time at {len(std_gauges)}; by standard time in winter)",
                source="results/day_windows.json, river_flow_day"),
           dict(input="aorc_to_midnight (new)", used_by="wqa (an extra column), wqa_rep (in the place of the Daymet rain)", covers=HOURS["aorc_to_midnight"], source="data/aorc/aorc_<gauge_id>.npz (NOAA AORC v1.1, code/fetch_aorc.py), UTC, summed by this script",
                day_ends_local=CLOCK[ids[0]]["last_hour_ends_local"]),
           dict(input="aorc_to_09 (new: the control with no new hour)", used_by="wqa_to09", covers=HOURS["aorc_to_09"], source="data/aorc/aorc_<gauge_id>.npz, summed by this script", day_ends_local=clock_hi("aorc_to_09", "last_hour_ends_local_max")),
           dict(input="aorc_utc_day (new: UTC-day control)", used_by="wqa_utc", covers=HOURS["aorc_utc_day"], source="data/aorc/aorc_<gauge_id>.npz, summed by this script",
                day_ends_local=f"{clock_lo('aorc_utc_day', 'last_hour_ends_local_min')} to {clock_hi('aorc_utc_day', 'last_hour_ends_local_max')}"),
           dict(input="rain_to_midnight and its controls (ERA5; read from script 15, not built here)", used_by="wqh, wqh_older, wqh_to09, wqh_utc, wqnh", covers="ERA5 hourly rain at the grid point of the centroid, summed over the wall-clock day (or the control windows) as in 15_rain_to_midnight.py, with the NLDAS column in wqnh",
                source="results/rain_to_midnight.json"),
           dict(input="prcp_nldas (NLDAS rain; the ladder, 10_rain_timing.py)", used_by="wqn, wqn_older (read from script 10), wqnh (script 15)", covers=f"a day that ends at about {nld['day_ends_local_rounded']} local standard time on day t (readings "
                f"{nld['range_over_readings_local'][0]} to {nld['range_over_readings_local'][1]}); if NLDAS is stamped by the UTC day, at {doc_end[0]}:00 ({doc_end[-1]}:00 in the Central-time basin)",
                source="results/day_windows.json, day_ends.nldas"),
           dict(input="the perfect forecast (the ladder; lstm_wqf)", used_by="wqf", covers="the Daymet weather stamped t+1, t+2 and t+3: an upper bound that cannot be had in practice", source="flood.Cfg(future=True)"),
           dict(input="basin properties (flood.STATIC)", used_by="every set", covers="fixed numbers per basin, not daily data", source="CAMELS attribute tables")])
R = dict(question="Given the rain of the whole local day t (00:00 to 24:00 by the local wall clock, from hourly AORC rain: 1 km, anchored on gauges and radar, the mean over the basin outline) as one extra daily input, how many more onsets are warned 1 day ahead, "
                  "and how much of the gap to the perfect weather forecast is closed - against the same AORC rain over the rain table's own window (no new hour), against the rain table alone, against the ERA5 rain to midnight of experiment E1 and against "
                  "NLDAS rain whose day ends in the evening? And does a rain table with the right clock (the Daymet rain column replaced by that AORC rain, no extra column) beat the mis-stamped Daymet table on its own?",
         quick_mode=F.QUICK, split=F.SPLIT, outcome=outcome, outcome_caveat=outcome_caveat, outcome_full=outcome_full, inputs=INPUTS,
         aorc_data=dict(record="results/aorc_data.json (written by code/fetch_aorc.py)", found=bool(AD),
                        **({k: AD[k] for k in ("what", "written", "source", "span", "n_hours", "mask_mode", "mask_how", "grid", "time_stamps", "store_layout", "choice", "download", "summary") if k in AD} if AD else {}),
                        per_basin={g: dict(n_cells=r_["n_cells"], mask_over_outline_area=ad_pl[g]["mask_over_outline_area"], spatial_chunks=ad_pl[g]["chunks"], mean_mm_day=r_["mean_mm_day"], daily_by_utc_day=r_["daily_by_utc_day"],
                                           ratio_aorc_to_daymet_mean=r_["ratio_aorc_to_daymet_mean"], hourly_against_era5_by_lag=r_["hourly_against_era5_by_lag"], max_hourly_mm=r_["max_hourly_mm"],
                                           fill_cells_met=r_["fill_cells_met"], hours_with_no_valid_cell=r_["hours_with_no_valid_cell"]) for g, r_ in AD.get("per_basin", {}).items() if g in ad_pl}),
         questions=["Does the E1 gain hold or grow when the rain to midnight comes from AORC (gauge- and radar-anchored, 1 km, basin mean) instead of ERA5? (wqa, against wqh and wqn)",
                    "What are the hours 09:00 to 24:00 of day t worth, apart from having a second rain product? (wqa_to09: the same AORC hours over the window of the rain table)",
                    "What are the evening hours beyond the UTC day worth? (wqa_utc: the same AORC hours over the UTC day, what a rain product served by UTC day or IMERG-Early would hold at the issue time)",
                    "Does a rain table with the right clock beat the mis-stamped table on its own? (wqa_rep = wqa_daymet_replaced: the Daymet rain column replaced by the AORC rain of the wall-clock day t)",
                    "Would ridge regression and a random forest gain from the new input as the LSTM does? (the yardstick, rule 2 of the work order)"],
         sets={n: dict(saved_as=f"lstm_{n}", inputs=f"{F.INPUT_LABEL[v['base']]} + {v['what']}", river_gauge=v["use_flow"], dyn=list(v["dyn"]), extra_input=extra_of(v), extra_inputs=list(v["cols"]), replaces=v["replaces"], control=v["control"],
                       hours_covered=hours_of(v)) for n, v in VAR.items()},
         control=dict(sets=MAIN, what="the same AORC series over the window of the rain table itself (09:00 of day t-1 to 09:00 of day t): a second rain product for the hours the table already covers and no hour after about 09:00 of day t. "
                                      "Test minus control is the worth of the hours 09:00 to 24:00 of day t. Everything else (settings, seeds, inputs, training years) is equal.",
                      second_control=dict(sets={"wqa": UTC_CONTROL}, what="the same AORC hours over the UTC day (what a rain product served by UTC day, or IMERG-Early, would hold at the issue time): test minus control is the worth of the evening hours beyond the UTC day")),
         figure_labels=dict(wqa_to09="control: AORC over the table's window (09:00 to 09:00)", wqa_utc=f"control: AORC over the UTC day (ends {doc_end[0]}:00 standard time)", wqh_to09="control: ERA5 over the table's window", wqh_utc="control: ERA5 over the UTC day",
                            older="control: the same rain, one day older (NLDAS, ERA5)", nldas_if_utc_stamped=f"{doc_end[0]}:00 if stamped by the UTC day"),
         config=conf, seeds=SEEDS, config_source="results/lstm_all.json, key config (chosen by 04_lstm_all.py on the validation years)",
         daily_rain=dict(source="data/aorc/aorc_<gauge_id>.npz (NOAA AORC v1.1 APCP_surface, mean of the 1 km cells inside the basin outline; code/fetch_aorc.py reads the open Zarr stores on AWS)", retrieved=AD.get("written"),
                         period_of_the_files=AO.SPAN_FOR_THE_RECORD, unit="mm/day", winter_utc_offset_h={g: UTC_OFF[g] for g in ids}, time_zone={g: ZONE[UTC_OFF[g]] for g in ids},
                         clock=CLOCK, check=dict(per_basin=rain_check, median_daymet_against_aorc_local_day=rc_med, median_era5_against_aorc_local_day=rc_era5, median_ratio_aorc_to_daymet_mean_rain=rc_ratio, median_ratio_era5_to_daymet_mean_rain=rc_ratio_era5,
                                                 how_to_read="correlation of the Daymet rain (or the ERA5 rain) stamped day d with the AORC local-day rain of day d-1 (day_before), d (same_day) and d+1 (day_after)"),
                         day_lengths=dict(per_basin=DAYLEN, median_basin=DL, all_basins_alike=DL_same,
                                          how_to_read="the number of days of 23, 24 and 25 hours (where the clocks change) and the share of days on which the wall clock is on summer time at the end of the day, the issue time"),
                         mask=dict(n_cells_by_basin=n_cells_by_basin, how="the cells of the 1/120 degree grid whose centre lies inside the outline; see aorc_data.mask_how and aorc_data.per_basin"),
                         windows=dict(how_to_read="the same hourly AORC values summed over other windows (aorc_to_09, aorc_utc_day), built and proved like aorc_to_midnight: per basin the clock record (the hours of the windows, the wall-clock time at "
                                                  "which the last hour ends, how many hours before the issue time it ends, the days on which the rain after the window was made up for the test)",
                                      aorc_to_09=dict(description=HOURS["aorc_to_09"], clock=CLOCK2["aorc_to_09"], day_lengths=LEN2["aorc_to_09"]),
                                      aorc_utc_day=dict(description=HOURS["aorc_utc_day"], clock=CLOCK2["aorc_utc_day"], day_lengths=LEN2["aorc_utc_day"]),
                                      last_hour_ends_before_the_issue_time_h={w: [margin_lo[w], margin_hi[w]] for w in NEW_WINDOWS}),
                         check_windows=dict(per_basin=WCHECK, median=WCHECK_MED, nldas_resembles_most=nldas_nearest,
                                            how_to_read="correlation of the Daymet rain (daymet) and of the NLDAS rain (nldas) stamped day d with the AORC series of the same name of day d-1 (day_before), d (same_day) and d+1 (day_after)")),
         stamp_convention_check=stamp_check if stamp_check else dict(what="not computed (the network files were not found)"),
         summary_test=dict(how_to_read="test years, seed average of each set, trigger 1.0, per lead; onset and all = warning tables pooled over the basins", sets=summary_test),
         validation_used_for_no_choice=dict(label="USED FOR NO CHOICE. Validation years (2000-2005), shown for reference only: no input set, setting or stopping point of this script was chosen on these scores (the settings came "
                                                  "from 04_lstm_all.py). The trainer stopped each training on its own validation score, so these scores flatter the models. The one thing this script chooses on "
                                                  "the validation years is the warning level of two of the three verdict readings; that is labelled in verdicts, and it does not use these tables.", summary=summary_val, models={n: EV[n]["val"] for n in SHOWN}),
         per_basin=per_basin, seed_ranges=dict(how_to_read="lowest and highest single-seed value of the test-year scores, per set and lead (by_seed: the single values, in the order of the seeds)", sets=ranges),
         gain_rule=gain_rule, closure=closure, ladder=ladder, e1b_success=e1,
         verdicts=dict(criteria=CRIT, criteria_source="results/flood_analysis.json, verdict.criteria", levels=VERDICT_NOTE, sets=verdicts),
         **readings, by_period=by_period, yardstick=yardstick, integrity=INTEGRITY,
         choices_made_after_looking_at_test_scores=["The design of E1b (its four input sets, the controls and the yardstick) follows from ROADMAP.md, step 1b, and from the controls that the review of E1 asked for, and was fixed after the test-year scores of E1 (script 15) were known. "
                                                    "No AORC score was seen before the sets, settings, seeds, stopping rule, warning level and success rule were fixed, and none of them was changed after the first AORC training, except that seeds 3 and 4 "
                                                    "were added later (next entry); every set trained is in this file.", SEED_NOTE],
         checks_against_stored_results=stored if stored else dict(what="not run in a smoke test: its forecasts are not the stored ones"),
         models=models, training=training, reload_check=reload_check, **timing, notes=notes, what_was_not_done=NOT_DONE)
_py = lambda o: o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)


def clean(o, count=None):
    """Strict JSON: numpy values become Python ones, and a number that is not finite (NaN: a share whose denominator is zero, such as the share of false warnings of a model that gave no warning) becomes None, which json
    writes as null. count[0] is raised by one for every value replaced."""
    if isinstance(o, dict):
        return {k: clean(v, count) for k, v in o.items()}
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


FIVE = five_seed_changes(clean(R), "aorc_rain_to_midnight", lambda n: n if n.startswith("base_") else f"lstm_{n}")        # what the two extra seeds changed, against results/archive (review round 4 of E1, should-fix 2, applied to E1b alike)
if FIVE:
    R["five_seed_changes"] = FIVE
    R["choices_made_after_looking_at_test_scores"] = [c + " " + FIVE["sentence"] if c == SEED_NOTE else c for c in R["choices_made_after_looking_at_test_scores"]]
notes.insert(0, "Quoting. The sentence in `outcome` is accurate arithmetic but incomplete as a statement of the result: any document (report, README, slides, Q&A, guide) must quote `outcome_full` (the outcome plus `outcome_caveat`), "
                "never `outcome` alone. Both are computed from the tables of this file. The readings of the controls are quoted in pairs (see the text of each reading).")
nulls = [0]
clean(R, nulls)
notes.append(f"{nulls[0]} values in this file are not numbers (NaN: shares whose denominator is zero, for example the share of false warnings of a model that gave no warning, or a p-value that is not defined); "
             "they are written as null (strict JSON; several other results files of this project still hold bare NaN tokens). A reader must treat null as 'no value'.")
JSON_TEXT = json.dumps(clean(R), indent=1, allow_nan=False, default=_py)
(F.RES_OUT / f"{OUT}.json").write_text(JSON_TEXT)
R = json.loads(JSON_TEXT)                                  # the figure reads exactly what the JSON holds


# ---- figure ------------------------------------------------------------------------------------------------------------------------------
def draw(R):
    """The ladder of information 1 day ahead, with the river gauge: onsets warned above and median NSE below. Filled: a model (the forecast averaged over the trainings, with the lowest to highest single training as a bar);
    hollow: a control, the same training with another rain series or window."""
    FS, H = 8.5, 9.0
    TICK = dict(zip(RUNG, ["rain table\n(Daymet)", "NLDAS\nrain", "rain to midnight\n(ERA5, hourly)", "rain to midnight\n(AORC, 1 km)", "AORC rain replaces\nthe rain table\n(no extra column)", "NLDAS rain +\nERA5 to midnight\n(both)", "perfect\nforecast"]))
    POS = dict(treat=-.30, older=-.18, to09=-.06, utc=.06)                                                           # the marks of a rung side by side; a kind of control keeps its place on every rung
    SHAPE = dict(older="o", to09="D", utc="^")                                                                         # the shape of each kind of control
    shaded = [RUNG.index(r) for r in ("rain to midnight (AORC)", "AORC rain replaces the rain table") if r in RUNG]
    rungs = R["ladder"]["wq"]["rungs"]
    ends = {r["rung"]: r["day_ends_local"] for r in rungs}
    fig = plt.figure(figsize=(10.2, H))
    axpos = lambda top, h: [0.075, 1 - (top + h) / H, 0.915, h / H]                                                  # inches from the top of the figure
    ax1 = fig.add_axes(axpos(0.62, 2.5))
    ax2 = fig.add_axes(axpos(3.6, 2.5), sharex=ax1)
    ax1.tick_params(labelbottom=False)
    panels = [(ax1, "onset", "Onsets warned, 1 day ahead", "share of the river rises (onsets) warned", FuncFormatter(lambda v, _: f"{100 * v:.0f}%")),
              (ax2, "nse", "Accuracy, 1 day ahead", "median NSE (1 = perfect)", None)]
    hollow = lambda ax, x, y, shape, ms=7: ax.plot(x, y, ls="none", marker=shape, ms=ms, markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.6, zorder=3)
    for ax, kind, title, ylab, fmt in panels:
        val = lambda e: (e["onset_warned"], e["onset_seed_range"]) if kind == "onset" else (e["nse_median"], e["nse_seed_range"])
        lab = lambda y: f"{100 * y:.0f}%" if kind == "onset" else f"{y:.2f}"
        for pos in shaded:
            ax.axvspan(pos - .5, pos + .5, color=SEQ[0], alpha=.3, lw=0, zorder=0)                                  # the rungs this script adds
        allv, xs, ys = [], [], []
        for r in rungs:
            e, rung = r["by_lead"]["l1"], RUNG.index(r["rung"])
            y, rg = val(e)
            x = rung + POS["treat"]
            xs.append(x); ys.append(y); allv.append(y)
            if rg:                                                                                                  # lowest to highest single training
                ax.plot([x, x], rg, color=BLUE, lw=1.6, solid_capstyle="butt", zorder=2)
                ax.plot([x, x], [rg[0], rg[0]], color=BLUE, marker="_", ms=5, mew=1.4, zorder=2)
                ax.plot([x, x], [rg[1], rg[1]], color=BLUE, marker="_", ms=5, mew=1.4, zorder=2)
                allv += rg
            for c in r["controls"].values():                                                                        # the controls of this rung, hollow
                yc, rc = val(c["by_lead"]["l1"])
                xc = rung + POS[c["kind"]]
                if rc:
                    ax.plot([xc, xc], rc, color=BLUE, lw=1.2, alpha=.5, solid_capstyle="butt", zorder=2)
                    allv += rc
                hollow(ax, xc, yc, SHAPE[c["kind"]])
                allv.append(yc)
                if c["kind"] != "older":                                                                            # one number under the window control, the other over the UTC-day control
                    below = c["kind"] == "to09"
                    ax.annotate(lab(yc), (xc, yc), xytext=(0, -7 if below else 7), textcoords="offset points", ha="center", va="top" if below else "bottom", fontsize=FS - 2, color=INK2)
        ax.plot(xs, ys, color=BLUE, lw=1.3, alpha=.5, zorder=1)
        ax.plot(xs, ys, ls="none", marker="o", ms=7.5, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
        for x, y in zip(xs, ys):
            ax.annotate(lab(y), (x, y), xytext=(-5, 5), textcoords="offset points", ha="right", va="bottom", fontsize=FS - 1, color=INK2)
        lo, hi = min(allv), max(allv)
        ax.set_ylim((0, hi * 1.1) if kind == "onset" else (np.floor((lo - .05) * 10) / 10, min(1.0, hi + .06)))
        ax.set_xlim(-.62, len(RUNG) - .38)
        ax.set_xticks(range(len(RUNG)), [TICK[n] + "\n" + (f"ends {ends[n]}" if ends.get(n) else "upper bound") + ("\n(" + R["figure_labels"]["nldas_if_utc_stamped"] + ")" if n == "NLDAS rain" else "") for n in RUNG], fontsize=FS - 1)
        ax.tick_params(labelsize=FS)
        ax.tick_params(axis="x", labelsize=FS - 1)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        ax.set_title(title, fontsize=FS + 1.5)
        ax.set_ylabel(ylab, fontsize=FS)
        ax.grid(axis="x", visible=False)
    n_seed = max(r["by_lead"]["l1"]["n_seeds"] for r in rungs)
    lg = lambda marker, label, **kw: Line2D([], [], ls="none", marker=marker, ms=7.5, label=label, **kw)
    kinds = {c["kind"] for r in rungs for c in r["controls"].values()}
    h = [lg("o", "model with the river gauge", color=BLUE, markeredgecolor=SURFACE), Line2D([], [], color=INK2, lw=1.6, marker="_", ms=5, mew=1.4, label=f"lowest to highest of {n_seed} trainings")]
    LEG = dict(older="control: the same rain, one day older (NLDAS and ERA5 rungs)", to09="control: the same product over the table's window, 09:00 to 09:00 (ERA5 and AORC rungs)",
               utc="control: the same product over the UTC day, ends 19:00 standard time (ERA5 and AORC rungs)")
    h += [lg(SHAPE[k], LEG[k], markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.6) for k in ("older", "to09", "utc") if k in kinds]
    fig.suptitle("Giving the model the rain of the whole day, up to midnight: AORC (gauge- and radar-anchored, 1 km) against ERA5", fontsize=10.5, fontweight="bold", color=INK, x=0.012, y=0.992, ha="left")
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=FS - .5, bbox_to_anchor=(.5, 1 - 7.0 / H), handlelength=1.6, columnspacing=1.2, labelspacing=.55, handletextpad=.5)
    a, b = R["split"]["test"]
    fig.text(.012, .008, f"Test years {a[:4]}-{b[:4]}, {len(R['per_basin']['basins'])} river basins, all models with the river gauge. Filled: forecast averaged over the trainings; numbers: the averaged forecast.\n"
                         "Hollow: a control, the same training with another rain series (the window controls: their value is printed beside the mark).\n"
                         "Under each name: the hour at which that rain's day ends (as measured, in standard time; ERA5 and AORC series: by the wall clock, NLDAS: " + R["figure_labels"]["nldas_if_utc_stamped"] + ").\n"
                         "The rungs 'ERA5' and 'NLDAS rain + ERA5 to midnight' and their controls come from scripts 10 and 15, not from this script. The shaded rungs are the new ones.\n"
                         "The perfect forecast is an upper bound that cannot be had in practice. AORC is an analysis made after the fact (hours before 2002 are disaggregated daily totals), not rain that could be read at midnight.",
             fontsize=FS - 1.3, color=INK2, ha="left", va="bottom", linespacing=1.45)
    fig.savefig(F.FIG_OUT / f"{OUT}.png", dpi=220)
    plt.close(fig)


draw(R)

print(f"\n{outcome_full}\n")
for n in SHOWN:
    s = R["summary_test"]["sets"][n]
    print(f"  lstm_{n:10s} test years: median NSE by lead " + " ".join(f"{s[lk(k)]['nse_median']:.3f}" for k in F.LEADS) + "   onsets warned " + " ".join(f"{s[lk(k)]['onset']['pod']:.3f}" for k in F.LEADS)
          + "   verdict (onsets, plain level) " + " ".join(R["verdicts"]["sets"][n][lk(k)]["plain"]["onset"]["verdict"] for k in F.LEADS))
print("\nreload check: " + "; ".join(f"{n}: max difference {x['max_abs_difference']:.2e}, bit for bit {x['bit_for_bit']}" for n, x in R["reload_check"]["per_set"].items()))
for key in ("value_of_hours_09_to_24", "value_of_evening_hours_beyond_utc_day", "value_of_hours_09_to_utc_day_end", "second_rain_product_alone", "aorc_against_era5", "aorc_against_nldas", "nldas_rain_against_aorc_over_the_utc_day",
            "aorc_against_nldas_and_era5_together"):
    if key in R:
        o_, s_ = R[key]["by_lead"]["l1"]["onsets"], R[key]["by_lead"]["l1"]["nse"]
        print(f"{key}: onsets warned {100 * o_['warned_with']:.1f}% against {100 * o_['warned_without']:.1f}% ({o_['difference_points']:+.1f} points, {o_['hits_with']} against {o_['hits_without']} hits, every seed above: {o_['every_seed_above_every_seed']}); "
              f"NSE {s_['median_with']:.3f} against {s_['median_without']:.3f} ({s_['gain']:+.3f}, {s_['reading']})")
if "rain_table_with_the_right_clock" in R:
    for key in ("against_rain_table_alone", "against_aorc_as_an_extra_column"):
        o_, s_ = R["rain_table_with_the_right_clock"][key]["l1"]["onsets"], R["rain_table_with_the_right_clock"][key]["l1"]["nse"]
        print(f"rain_table_with_the_right_clock {key}: onsets warned {100 * o_['warned_with']:.1f}% against {100 * o_['warned_without']:.1f}% ({o_['difference_points']:+.1f} points, {o_['hits_with']} against {o_['hits_without']} hits, every seed above: {o_['every_seed_above_every_seed']}); "
              f"NSE {s_['median_with']:.3f} against {s_['median_without']:.3f} ({s_['gain']:+.3f}, {s_['reading']})")
Y = R["yardstick"]
for n in ("base_persistence", "base_linreg_wq", "base_rf_wq") + tuple(x for x in ("base_linreg_wqh", "base_rf_wqh") if x in ES) + YARD:
    print(f"  {n:18s} test years: median NSE by lead " + " ".join(f"{Y['table_test'][lk(k)][n]['nse_median']:.3f}" for k in F.LEADS) + "   onsets warned " + " ".join(f"{Y['table_test'][lk(k)][n]['onset']['pod']:.3f}" for k in F.LEADS)
          + "   verdict (onsets, plain level) " + " ".join(Y['table_test'][lk(k)][n]['onset_verdict_plain'] for k in F.LEADS))
for key, c in Y["lstm_against_simple"].items():
    if key.startswith("lstm_wqa__"):
        print(f"  {key}: " + "; ".join(f"{k}: {c[k]['reading_in_words']}" for k in c))
if "stamp_convention_check" in R and "sets" in R["stamp_convention_check"]:
    sc_ = R["stamp_convention_check"]["sets"]["wqa"]["by_lead"]["l1"]
    print(f"stamp convention (opposite reading, saved networks of wqa): onsets warned {100 * sc_['onset_warned_opposite_convention']:.1f}% instead of {100 * sc_['onset_warned_documented_convention']:.1f}% ({sc_['difference_points']:+.1f} points)")
print(f"{R['n_trainings']} trainings in {R['training_seconds']:.0f} s; runtime {R.get('runtime_s', R.get('rescore_runtime_s')):.0f} s")
