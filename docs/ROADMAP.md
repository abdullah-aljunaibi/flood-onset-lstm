# Roadmap: from an experiment to a flood forecaster that could run

Written 2026-10-07 by the session lead (Claude Fable 5.1, directing six Sonnet scout and build agents) under the team lead's direction.
`NEXT_STEPS_FOR_AGENTS.md` stays the work order: its rules (section 3) are not changed here. This file adds what was *measured* today -
where the information is lost, which data can actually be had from this laptop, what each step costs - and puts the experiments in the
order that follows from it. Every number below is read from a results file named beside it. Records written today, all in `results/`:
`onset_rain_clock.json` (+ `.csv`), `data_access.json`, `published_context.json`, and `rain_to_midnight.json` (experiment E1, with
`figures/rain_to_midnight.png` and the saved networks in `results/models/`).

## 1. Where the model stands, and why

Test years 2005-2014, 12 basins, 1 day ahead, pooled over the basins (`results/lstm_all.json`, `results/rain_timing.json`,
`results/flood_analysis.json`; recomputed from `results/preds` with `flood.evaluate` on 2026-10-07):

| Inputs of the all-river LSTM | Median NSE | Onsets warned | Warnings false |
|---|---|---|---|
| Past weather + river gauge (what could run today) | 0.59 | 19% | 59% |
| + rain of the issue day up to about 17:30 (NLDAS; `10_rain_timing.py`) | 0.67 | 30% | 63% |
| + the weather of the three days being forecast (a perfect forecast: an upper bound that cannot be had in practice) | 0.86 | 64% | 26% |

Two and three days ahead the realistic model falls to NSE 0.32 and 0.25 while the upper bound stays at 0.82 and 0.78. The published
531-basin LSTMs reach 0.86 when given yesterday's flow and the weather of the day itself (Feng et al. 2020, `results/published_scores.json`) -
the same as our upper-bound row. So, among the models tried, the network is not what holds the forecast back; the information is.
Nothing in today's work changes that finding; everything below acts on it.

## 2. What the forecast is missing, measured (new today)

`code/check_onset_rain_clock.py` -> `results/onset_rain_clock.json`. For each of the 611 flood onsets of the test years, the rain of the
48 hours around the forecast (issue day t, target day t+1) was split by when it fell, from the cached hourly ERA5 rain at the basin
centroid:

| Rain of the 48 hours around an onset | Mean | Share | At the issue time (local midnight) |
|---|---|---|---|
| Fell before about 09:00 of the issue day | 6.5 mm | 18% | In the rain table: the model sees it |
| Fell between about 09:00 and 24:00 of the issue day | 12.6 mm | 35% | Already fallen, but filed under the next day: hidden from the model |
| Falls on the target day | 16.6 mm | 47% | Still to come: only a weather forecast can supply it |

Of the 497 onsets the current model misses, the median share of that rain known at the issue time is 4%; with the rain up to midnight it
would be 50%. The 335 onsets whose rain is filed almost entirely under the target day are warned 2% of the time today and 71% of the time
with the perfect forecast (`results/flood_analysis.json`, failures by rain timing).

What this says about the order of work: the single largest block of missing information (35%) has *already fallen* when the forecast is
issued - experiment E1 recovers it without any forecast. The rest (47%) needs an archived real weather forecast (E2). More basins (E3) and a
better network (E6) cannot supply either; they can only use the given information a little better.

## 3. Data that can be had, verified from this laptop today

Details, byte counts, timings and URLs: `results/data_access.json`. All checks ran through the Clash proxy; nothing was bought and no
account was created. "est." = extrapolated from the files sampled.

| Source | What it gives | Years | Access | Size for us | Time or money | Role in the plan |
|---|---|---|---|---|---|---|
| **GEFSv12 reforecast** (NOAA, AWS Open Data) | weather forecasts *issued* daily at 00 UTC, 0.25 degree, 3-hourly, 5 members | 2000-2019 | open; byte ranges via `.idx`; decodes with pip wheels (eccodes + cfgrib) on this Mac | 61 GB est. for the control member's rain to 96 h, 2000-2014 (stream, reduce, delete: the disk has 61 GB free) | 5-30 h on the laptop at the measured 0.5-4 MB/s, or about USD 2-3 on a small cloud VM next to the data | **E2**, the forecast input |
| GEFS reforecast v2 (NOAA PSL) | the older archive, 1 degree, 11 members | 1985-2020 | web page discontinued 2020; FTP only at 0.03 MB/s per connection | 29 GB est. for rain to 96 h, 1985-2014 | about 60 h at 8 connections | the only archive that reaches the training years; coarse (one cell holds several of our basins); fallback |
| **AORC v1.1** (NOAA, AWS) | hourly rain, 1 km, gauge- and radar-anchored | 1979-2025 | open Zarr, no login | 4.1 GB est. for the 12 basins, 1980-2014 | a few hours, free | **E1b**: rain to midnight from radar-like data over the whole period; also a rain table with the right clock |
| Stage IV (NCEP, via Iowa Mesonet archive) | the hourly radar+gauge product a real-time service reads (about 1 h behind) | 2002-2014 | open, no login | 128 GB est. | about 30 h, free | the true real-time product, for a check of a few flood seasons |
| **IMERG V07 Early / Late** (NASA, cloud copy by dynamical.org, AWS) | half-hourly satellite rain, 0.1 degree; Early is about 4 h behind real time | 1998-now | open, no login; needs Python 3.12+ and the icechunk library (a scratch environment worked) | 0.4-0.7 GB est. for 12 points | an hour, free | **the satellite real-time case** (section 5) |
| ERA5 hourly (Open-Meteo) | model rain at the centroid, 0.25 degree | 1940-now | no key; cached already | cached | done | E1 as run today (a stand-in) |
| GES DISC IMERG half-hourly, NLDAS-2 hourly (NASA) | the gauge-calibrated IMERG; hourly gauge analysis | 1998- / 1979- | needs a free NASA Earthdata login (anonymous requests are redirected) | point series through NASA's subsetters | small, free, after the login | optional cross-check |
| **ESA CCI soil moisture v09.2** (CEDA) | daily satellite soil moisture, 0.25 degree | 1978-2024 | open, no login | 10 GB est. for 2000-2014 (whole global files; about 23 GB for 1980-2014) | overnight, free | the satellite soil-moisture input (optional) |
| NWM retrospective (NOAA, AWS) | hourly simulated flow of a physics model for every reach | 1979-2023 | open Zarr; our 12 gauges mapped to reach ids | about 14 GB est. | hours, free | a physics baseline or extra input (optional) |
| NASA POWER, CMORPH, PERSIANN | daily IMERG by UTC day (cached); hourly satellite records | - | open | - | - | no hourly IMERG at POWER; CMORPH/PERSIANN not needed |
| CAMELS-US, the other 659 basins | flow and weather of more rivers | 1980-2014 | on this disk | 63 south-eastern basins complete (verified file by file); 516 of 671 complete | 0 | **E3** |

## 4. The plan, in order, with what decides each step

### Step 0 - done today: E1, rain observed up to the issue time (`code/15_rain_to_midnight.py`)

The rain of the whole issue day, 00:00 to 24:00 by the wall clock of the gauge (summer time included, so that no hour after the issue time
is read for the gauges whose flow day follows the summer clock), from the cached hourly ERA5 rain, as one extra daily input; the control is
the same series one day older; with and without the river gauge; three seeds each. Every trained network is saved to `results/models/`
and the script proves that a reloaded network reproduces its stored forecasts bit for bit - problem P2 of the work order is fixed for the
new sets (the earlier sets would need one retraining with the same seeds to be saved too). The clock of the input was proved in code by two
independent routes and a test that changing the rain after midnight changes no value.

**Result:** see section 8, written after the run finished.

### Step 1 - this week, free, on the laptop (about one evening of computing in total)

1a. **Independent check of E1** by a second agent, recomputing the headline numbers from `results/preds` with its own code (rule of the
    work order). Then one subsection after 4.5 of the report and one rung in the key figure, built by the builders (not yet done).
1b. **DONE 2026-10-07 evening (result below).** E1 again with AORC instead of ERA5 (4 GB download, no login): the same experiment with gauge- and radar-anchored hourly rain at
    1 km, averaged over the basin outline (`data/camels_us/basin_set_full_res`). Two variants: the rain to midnight as the extra input
    (as E1), and AORC's daily rain *replacing* the Daymet rain altogether (a rain table whose day ends at the issue time). Success: the
    E1 gain holds or grows; the second variant tells whether the mis-stamped table itself costs skill.
1c. **The satellite real-time case** with IMERG Early half-hourly (under 1 GB, no login): rain to about 19:00-20:00 local (what a service
    using satellite rain would have at midnight, given its delay of about 4 hours) on the satellite-era split of `08_satellite_inputs.py`, against the
    NLDAS rung and the AORC rung on the same split. Settle the stamp convention of the cloud copy first (one event showed it unresolved).
1d. **DONE 2026-10-07 night (result below).** E3 at 63 basins (`code/17_more_basins.py`; the smallest engine change is an optional `ids` argument to `flood.basins()` and
    `flood.load_data()` with a separate cache - about 30 lines, spec in the session scratchpad `scoping/E3_P2_SPEC.md`; the default calls
    stay bit-identical). Train on the 63 complete south-eastern basins (51 extra, verified), score the same 12 and the same test years, with
    and without the gauge, three seeds. Six trainings take about 18 minutes against 16 for the 12 alone (the cost per update does not grow
    with basins; only the validation pass does). Control: the stored `lstm_wq` / `lstm_w`. Then **rivers never seen, at scale**: train on
    the 50 extra basins (one is nested inside SF Holston and is left out), score the 12 as unseen, against `results/lstm_lobo.json`
    (held out with 11 training basins: median NSE 0.22 with properties, 0.31 without, 1 day ahead). Caveat: the 51 lie closer to the 12
    (median 39 km to the nearest training gauge against 110 km in the leave-one-out test), so part of any gain is proximity.
    Expected from the literature (`results/published_context.json`, Kratzert et al. 2024, same-day simulation, read by eye): about +0.02
    median NSE from 12 to 63 basins and about +0.05 at 531; unknown for a forecast with a gauge, where it is probably smaller.
1e. **P2 for the engine**: move save/load into `flood.py` (`save_run`, `load_run`), retrain `lstm_wq`, `lstm_w`, `lstm_wf`, `lstm_wqf`
    with their seeds to save them (about 40 minutes; check that the forecasts match the stored sets), and add `code/forecast.py`, which
    loads a saved network and forecasts any date of the record for any basin. `test_flood.py` must still pass.

### Step 2 - E2, the archived real weather forecast as an input (DONE 2026-10-08 00:29; result below the design)

Data: GEFSv12 reforecast, control member first, rain only, leads to 78 h (the local days t+1 to t+3 need forecast hours +5 to +77 Eastern,
+6 to +78 Central), 2000-2014: about 50-61 GB streamed and reduced to the south-eastern box (about 2 GB kept) and then to basin means.
The latest run usable at the issue time is the 00 UTC run of the UTC date t+1 (19:00 local of day t); its products appear about 4 h 21 min
after the run, 40 minutes before the Eastern issue time - a real-time service would have it, barely; say so.

Two experiments, in this order:
- **E2a, substitution, scored on the usual test years.** Train a model on 1980-2000 whose extra inputs are the *observed* rain of t+1, t+2,
  t+3 (a perfect rain forecast only; temperature and the rest stay known up to day t), choose its stopping point on 2000-2005 with the GEFS
  rain in those slots (the validation years are inside the archive), save it, and score 2005-2014 twice: with the perfect rain (the ceiling
  for rain alone) and with the GEFS rain. Control: the GEFS rain of a random other day of the same month (the same climate, no information
  about this day). This needs no change to the engine (the forecast rain enters as daily columns dated by the issue day) and gives numbers
  on the same test years as every row of the report.
- **E2b, retraining with the forecast as an input**, on the satellite-era split (train 2000-2008, validation 2008-2010, test 2010-2014,
  as in `08_satellite_inputs.py`), so the network learns the forecast's own errors; `wq` and the E1 rung retrained on the same split as
  controls. Numbers of this split are comparable only inside it.
- Then, if the gain is clear: the 5-member mean (five times the download, or a cloud VM), temperature from the forecast, and the
  encoder-decoder form that reads the forecast day by day (the structure of Google's 2024 model; `09_model_comparison.py` is the pattern).

Expected (my judgement, to be replaced by the run): 1 day ahead between the E1 result and the 64% upper bound; 2 and 3 days ahead the
largest gain of the whole project, from NSE 0.32 / 0.25, because today's model has nothing there but the river's own momentum. GEFSv12
rain is skilful but far from perfect (Brier skill 0.42 at 12-36 h and 0.33 at 60-84 h for days with at least 5 mm, Guan et al. 2022), so
the onset verdict will probably stay NOT YET (GO needs 70% warned with at most 30% false; even the upper bound is at 64%). That is still
the honest result the report is for: how much of the gap a real forecast closes.

### Step 2 result (2026-10-08, 00:29): the archived real weather forecast as an input (`code/gefs_to_basins.py`, `code/18_forecast_input.py`,
`results/gefs_basins.json`, `results/forecast_input.json`)

Data: the GEFSv12 control member's rain, basin-averaged with exact cell overlaps (1 to 9 cells per basin), summed over the local wall-clock days
t+1..t+3 from the 00 UTC run of the UTC date t+1 (4 to 6 hours old at the issue time; the margin in Eastern summer is about -15 minutes for the
last files, noted as a caveat). Its own skill against AORC daily rain (median of basins): correlation 0.72 / 0.66 / 0.59 at 1 / 2 / 3 days,
mean ratio about 1.0 (summer 1.12, winter 0.88; Florida basins 1.2-1.4), calibrated Brier skill for 5-mm days 0.44 / 0.39 / 0.33 (one member, an
isotonic map fitted on the validation years; of the same order as the published GEFSv12 numbers, 0.42 and 0.33, which were computed
differently and are not reproduced here). Bias factors per basin and lead fitted on the validation
years only. 30 trainings, 80 minutes.

E2a, main split, test years 2005-2014 (seed averages; single trainings in the results file):

| Set with the river gauge | Onsets warned 1 / 2 / 3 days | Median NSE 1 / 2 / 3 days |
|---|---|---|
| Rain table alone (`lstm_wq`) | 19% / 8% / 6% | 0.59 / 0.32 / 0.25 |
| + AORC rain to midnight (`lstm_wqa`, E1b) | 51% / 21% / 12% | 0.77 / 0.38 / 0.25 |
| + GEFS forecast of the next three days, substituted into a network trained on the AORC rain of those days, an analysis made after the fact (`lstm_wqaF_sub`) | 45% / 39% / 37% | 0.69 / 0.53 / 0.45 |
| control: the same with the forecast shuffled within month (`lstm_wqaF_clim`) | 14% / 4% / 4% | 0.60 / 0.19 / 0.03 |
| the forecast without the rain to midnight (`lstm_wqF_sub`) | 31% / 36% / 36% | 0.59 / 0.51 / 0.45 |
| perfect rain of the three days (`lstm_wqaP`, rain-only ceiling with the right clock) | 66% / 69% / 71% | 0.80 / 0.78 / 0.77 |
| perfect weather of the three days (`lstm_wqf`, the old upper bound) | 64% / 67% / 69% | 0.86 / 0.82 / 0.78 |
| ridge / random forest given the same inputs | 69 / 55% at 1 day, more false warnings | 0.71 / 0.72 at 1 day |

E2b, retrained with the forecast as an input, on its own split (train 2000-10..2008-09, test 2010-10..2014-09; numbers comparable only
inside this table): rain table + gauge 15% / 6% / 4% and NSE 0.53 / 0.33 / 0.26; + rain to midnight 41% / 14% / 8%, 0.71 / 0.39 / 0.28;
**+ forecast, retrained: 61% / 49% / 44%, 0.77 / 0.62 / 0.52**; shuffled control 41% / 15% / 8% (harmless); perfect rain 72% / 77% / 77%.

Readings (as corrected by the independent review, `results/review_forecast_input.json`): the forecast is worth +18 and +25 points of onsets
at 2 and 3 days ahead (clear in every training; 30-day block bootstrap +10 to +26 and +18 to +32; positive in 7 of 9 and 9 of 9 test years)
and +0.16 / +0.20 NSE ("mixed" by the seed rule: 8 and 9 of 12 basins). At 1 day the substituted model warns of 7 points fewer onsets at
the fixed warning level but with far fewer false warnings (35% against 50%); at the level the project chooses on the validation years it
is 2.6 points ahead, at equal false-warning budgets 9 to 11 points ahead, and its trigger-free average precision is higher (0.46 to 0.53):
a moved operating point, not lost skill. The drop at the fixed level sits in onsets with little rain on the target day; whether the
network's trust in slots that were exact in training causes it is plausible but not isolated (the same networks given perfect rain at run
time warn of 69%; a third of the lost onsets are lost even then). Retraining with the forecast (E2b) gives +20 / +35 / +36 points over
the same base on its split and closes 65 / 56 / 52% of the gap to perfect rain. The E2a shuffled control is not a neutral baseline: a
network trained on exact slots and given noise collapses, so it measures harm, not information; the E2b control (41% against 41% without
slots) is the clean one and shows the gain there is the forecast's information. The rain to midnight still matters with the forecast at
1 day (+13 points) and not at 2-3 days. The simple models given the same inputs gain as much or more (ridge +39 / +40 points at 2-3 days):
the information, not the network, once more.
Verdicts: onsets NOT YET at every lead (E2a 35-41% of onset warnings false); all high-flow days CAUTION / CAUTION / NOT YET in E2a and
GO / CAUTION / CAUTION in E2b on its split. The E2 success rule of the work order (a clear gain at 2 and 3 days) is met.

Caveats: control member only (an ensemble mean would be smoother and probably better); 0.25-degree cells over small basins; a single bias
factor per basin (the seasonal bias remains); in E2b the bias factor's fitting years lie inside the training years; three seeds (NSE readings
"mixed" at 1-2 days); the base rung is AORC (retrospective); the issue-time margin in Eastern summer. Independent review done
(`results/review_forecast_input.json`, scripts in `code/review/forecast_input/`): no numerical discrepancy, no leakage; its wording fixes
are being applied to the results file. Not done: ensemble members, temperature forecasts, the encoder-decoder form, Stage IV in place of
AORC with the forecast, forecasts issued later than 00 UTC, the `13_uncertainty.py` ranges, the documents.

### Step 3 - later, each optional

- **516 basins** (all CAMELS basins with complete flow): six trainings in about 45 minutes on the laptop at equal updates; "equal
  training per basin" would take 7-20 laptop hours or 1-3 hours on a rented GPU (AutoDL RTX 4090, about CNY 9-14; Kaggle's free quota).
  Needs the trainer to mask missing targets if the 155 basins with gaps are ever added.
- **Satellite soil moisture** (ESA CCI, open): one extra daily input, with and without the gauge. The literature has no real-data number;
  the earlier MERRA-2 soil-wetness input gave no clear gain. A natural piece for the RS/GIS member.
- **The hourly project** (Stage IV hourly rain + USGS 15-minute flow, 2002-2014): a project of its own; scope it only after E2.
- **A physics baseline** from the NWM retrospective for the 12 gauges (14 GB), to set beside the LSTM the way the published papers do.
- **E5** (a loss aimed at peaks) and **E6** (network changes) stay last: no published real-data study shows a peak loss helping, and the
  four kinds of model here miss the same onsets.

### Step 4 - the documents

After each finished experiment: its results JSON, one report subsection, a rung in the key figure (`11_key_figure.py`), `13_uncertainty.py`
extended to the new sets, then `build_report.py`, `build_slides.py`, `build_qa.py`, `build_guide.py`. The main finding of page 1 is
expected to hold; what changes is the ladder of information and the "next steps" of section 8. The team's own checks (`team.json`) stay
open until the members do them.

### Step 1b result (2026-10-07 evening): AORC rain to midnight (`code/16_aorc_rain_to_midnight.py`, `results/aorc_rain_to_midnight.json`)

Data: NOAA AORC v1.1 hourly rain, 1 km, averaged over each basin outline (159 to 2,649 cells), 1980-2014, 4.5 GB fetched in 1.1 h
(`code/fetch_aorc.py`, `data/aorc/`, `results/aorc_data.json`); daily totals correlate 0.90 with NLDAS, 0.84 with IMERG, 0.74 with ERA5 and
0.68 with the rain table; hour stamps checked against Stage IV radar (r = 0.98 at lag 0 on test-year storm days). Same design as E1, 12 trainings:

| Input set with the river gauge, 1 day ahead, test years | Onsets warned | Median NSE |
|---|---|---|
| Rain table alone (`lstm_wq`) | 18.7% | 0.59 |
| AORC over the table's own window, 09:00 to 09:00 (control, no new hours) | 17.7% (17.3-17.7) | 0.55 |
| AORC over the UTC day, ends 19:00 local (`lstm_wqa_utc`) | 37.5% (36.5-38.5) | 0.69 |
| **AORC to midnight (`lstm_wqa`)** | **51.2% (48.6-55.3), 313 hits** | **0.77 (0.70-0.77)** |
| AORC replacing the Daymet rain column, no extra column (`lstm_wqa_rep`) | 51.6% (44.8-54.8) | 0.73 |
| ERA5 to midnight (E1) / NLDAS (`10_rain_timing`) | 30.0% / 30.0% | 0.66 / 0.67 |
| Ridge / random forest given the AORC rain to midnight | 55.2% / 48.0% | 0.64 / 0.66 |
| Perfect weather forecast (upper bound) | 64.3% | 0.86 |

Readings: the hours 09:00 to 24:00 are worth +33.6 points (every training above every control training), of which +13.7 come after 19:00;
the same product over the table's window adds nothing (-1.0 point): it is the hours, not a second product. AORC to midnight closes 71% of
the onset gap and 65% of the NSE gap to the perfect forecast and is better in 10 of 12 basins. All high-flow days: CAUTION (74% warned,
31% false, 0.8 points of false warnings short of GO); onsets: NOT YET (51% warned, 50% false). Two and three days ahead the NSE barely
moves (+0.05, +0.00) while onsets warned rise from 8% to 21% and from 6% to 12%: still forecast territory, with a little more warning. The simple models given the same input gain about as much (ridge warns of more onsets with more
false warnings; the LSTM has the best NSE): the finding "information, not network" holds here too. The E1 success rule of the work
order is met.

Independent review (`results/review_aorc_rain_to_midnight.json`, scripts in `code/review/aorc_rain_to_midnight/`): every number
reproduced, the 12 networks rebuilt bit for bit from independently built series, no hour after a window's end in any input, the AORC
cache equal to the reviewer's own reduction of raw chunks. The hindsight channel exists by construction (AORC spreads a daily total that
runs to 12 UTC over the hours with radar) but does not show in the test years: on 78 storm days the pre-midnight AORC rain does not depend
on the post-midnight radar rain (coefficient about 0, interval -0.06 to 0.00 for the whole day), 35 basin-days dry before midnight and wet
after hold 0.08 mm of AORC rain before midnight, and when real-time Stage IV radar hours replace AORC on the issue day of 171 onsets,
92 to 104% of the gain survives. Bound: best estimate 0 points of hindsight, at most about 3. The reviewer's reading: trust the 51% as a
measurement of what rain known up to midnight as well as AORC knows it is worth (seed range 49 to 55%, standard error 2 to 3.5 points); a
real-time version should land at most about 3 points lower; it is not yet an operational warning (half the onset warnings are false).
Two wording fixes to the results file were required and are being applied (the "mis-stamped table" clause; a clause that AORC is a
retrospective analysis, with the hindsight bound); the decisive real-time test on the issue days of all 611 onsets needs only about
10 GB of Stage IV files and is being run (`code/check_stage4_realtime.py` -> `results/stage4_realtime.json`).

**The real-time test (2026-10-07 night, `code/check_stage4_realtime.py` -> `results/stage4_realtime.json`; 7.2 GB of Stage IV files for the
368 issue days of the 611 onsets, 39 minutes):** with Stage IV radar-and-gauge rain, the product a service at midnight would read, in place of
AORC on the issue day of every onset, the saved networks warn of 294 onsets (48%) against 313 with AORC (51%) and 114 with the rain table
alone (19%): 90% of the AORC effect survives (interval 85 to 97%). With the last hour of the issue day left out, as a service that has
radar to about 23:00 would see it, 275 (45%); with the last two hours out, 255 (42%). So the honest real-time figure is about 45 to 48% of
onsets warned a day ahead, against 19% today. The count depends on whether the issue day was a heavy-rain day more than on the exact
amounts (the Stage IV rain of another onset in the same basin still gives 284; an ordinary day's rain 34; no rain 11); the exact rain shows
in the forecast error instead (mean absolute error 4.64 mm/day with Stage IV and with AORC, 5.62 with another onset's rain, 5.83 with the
rain table alone). With Stage IV on the day before as well, 289 (88%). Stage IV holds 6% less rain than AORC on these days; scaled to
AORC's level it gives 319, so there is no sign of hindsight in AORC at all. False warnings on the issue days: 25% with Stage IV against 30%
with AORC. Not tested: a network trained on Stage IV itself, false warnings on days other than the issue days (97 GB of files).

What to trust and what not yet:
- AORC is an analysis made after the fact from daily gauge totals spread over the hours with radar and other sources; before 2002 its
  hours are disaggregated daily totals. So the result says what *well-measured rain up to midnight* is worth, not what a real-time service
  delivers. The real-time equivalents are Stage IV and MRMS radar (about one hour behind), which Stage IV data would test directly.
- **Hindsight risk:** if AORC's daily anchoring uses a gauge day that ends in the morning, the pre-midnight hours could be scaled by a total
  that includes rain after the issue time. The builder's check against hourly Stage IV on the 50 stormiest test-year days
  (`code/check_aorc_hindsight.py`, `results/aorc_hindsight.json`) found no dependence of the pre-midnight AORC rain on the post-midnight
  radar rain (coefficient +0.04, interval -0.08 to +0.14) and near-zero AORC rain before midnight on the four basin-days that were dry
  before and wet after; the reviewer's larger test (above) agrees. The decisive test on the issue days of all onsets is running.
- The NSE gain is not uniform over the test years (post hoc split at 2009-07-17: gain over ERA5 -0.09 before, +0.14 after); the onset
  gain is stable (+21 points in both halves). Three trainings only; the headline NSE is the seed average at the top of the single-seed range.
- The clause of the results file that reads "the mis-stamped table itself costs skill" is stronger than the test supports (the replaced
  table changes product and clock together); quote the flat same-window control instead.

### Step 1d result (2026-10-07 night): 63 training rivers instead of 12 (`code/17_more_basins.py`, `results/more_basins.json`)

Engine: `flood.basins(ids=None)` and `flood.load_data(source, ids=None)` with a separate cache per basin set, `study_ids()`, `candidate_ids()`
(the 63 complete south-eastern basins); the defaults are proven unchanged (the default table equals the stored pickle byte for byte, the
headline `lstm_wq` score is unchanged, `test_flood.py` has one more check and passes). 15 trainings, 125 minutes on a shared laptop.

| Model, test years, the same 12 rivers, 1 day ahead | Median NSE | Onsets warned |
|---|---|---|
| With gauge, 12 training rivers (stored `lstm_wq`) | 0.59 (0.54-0.60) | 18.7% |
| With gauge, 63 training rivers (`lstm63_wq`) | 0.60 (0.58-0.60) | 17.3% (16.7-17.0) |
| With gauge, 63 rivers, five times the training (`lstm63eq_wq`) | 0.61 (0.59-0.60) | 17.7% |
| No gauge, 12 rivers (stored `lstm_w`) | 0.44 | 17.2% |
| No gauge, 63 rivers (`lstm63_w`) | 0.45 (0.41-0.43) | 18.0% |

| Rivers never seen (weather only), median NSE 1 day ahead | With basin properties | Without |
|---|---|---|
| 11 training rivers (`05_lobo.py`, one training per river) | 0.22 | 0.31 |
| 50 training rivers (`unseen50_w_*`, three seeds) | 0.38 (0.34-0.38) | 0.43 (0.41-0.43) |
| For scale: the same rivers when they were in training | 0.44 (12 rivers) / 0.45 (63 rivers) | |

Readings (as corrected by the independent review, `results/review_more_basins.json`): for the 12 gauged rivers, 63 training rivers give no
clear difference 1 day ahead (NSE higher in 10 of 12 rivers but inside the seed range) and a clear NSE gain 2 and 3 days ahead (0.32 to
0.34, 0.25 to 0.28). The lower onset share at the fixed warning level (19% to 17%) is a moved operating point, not lost skill: 8 fewer hits
but 59 fewer false warnings, the same CSI (0.147 against 0.148), and at the level chosen on validation the 63-river model warns of more
onsets (21.6% against 18.7%) and is better on all high-flow days. The arm with five times the update limit and patience kept the same
early networks as the plain arm (best update about 1,170), so it shows that a longer patience window does not move the optimum; it did
not test equal training per river. For rivers never seen the gain is clear at all three leads in the seed average (+0.16 NSE with
properties, +0.12 without; single trainings p = 0.05-0.09), and the loss from not having seen the river shrinks from 0.13 to about 0.02
NSE. Onsets warned for never-seen rivers move little (23% to 22% with properties, 18% to 22% without). Caveats: the number of training
rivers, their variety and their nearness to the 12 change together between the two designs (median 39 km against 110 km to the nearest
training gauge) and were not separated; the gain does not rank with distance (rank correlation -0.36, p = 0.26), so it cannot be
attributed to scale nor dismissed as proximity; the earlier test is one training per river against a three-seed average here; two of
the 12 have zero-flow days that no training river has; six 63-river trainings took 43 minutes on the shared laptop, not the 18 estimated. The outcome sentence of the results file: *Given training on 63 rivers
instead of 12, median NSE 1 day ahead for the same 12 rivers goes from 0.59 to 0.60 (onsets warned 19% to 17%); for rivers never seen,
from 0.22 to 0.38 with basin properties (0.31 to 0.43 without them; 50 training rivers instead of 11); the verdict is NOT YET.*
Independent check: done (`results/review_more_basins.json`, scripts in `code/review/more_basins/`): no numerical discrepancy, no leakage,
the 15 networks rebuilt bit for bit, the engine defaults proven unchanged; its wording fixes are being applied to the results file. Not done:
the simple-model yardstick at 63 rivers, the `13_uncertainty.py` ranges, a buffered leave-out or dose-response design, the documents.

What it means for the plan: more rivers is not where the gauged forecast's skill will come from (as section 2 said: the information about
tomorrow's rain is missing, and no number of rivers supplies it); it is where the *ungauged* forecast comes from, which is half of the
satellite case (section 5, point 2). The 516-basin run (step 3) is therefore only worth doing for the never-seen-river question.

### E7 - several rain products side by side: the rule, fixed on 2026-10-30 before the first training

Design: `NEXT_EXPERIMENT_E7_multi_product.md` (named E5 there at first; renamed because E5 and E6 above are taken). The MRMS arm
(E7b) was ruled out by the team lead on 2026-10-30, so E7 is E7a only. Script `code/28_multi_product.py`. Not part of the 2 November hand-in.
Main split (`flood.SPLIT`), settings of `results/lstm_all.json` unchanged, seeds 0-4, test years scored once.

Sets (all with the river gauge and the 10 basin properties): `wqME` = Daymet + `prcp_nldas` + `rain_to_midnight` (ERA5) +
`aorc_to_midnight` (the treatment); `wqME_shuf` = as `wqME`, with the NLDAS and ERA5 columns shuffled within basin and calendar month
across the years of the same split part (one permutation per basin-part-month group, shared by the two columns, fixed random seed, the
same data for every seed; "the same split part" added on 2026-10-30 before the full training, so no test-year value enters a training day);
`wqME_older` = as `wqME`, with the NLDAS and ERA5 columns one day older; `base_linreg_wqME` and `base_rf_wqME` (the recipe of
`02_baselines.py` on the same columns). The base is the stored `lstm_wqa` (script 16, five seeds); also shown: `lstm_wq`, `lstm_wqn`,
`lstm_wqh`, `lstm_wqnh`.

All scores: lead 1, trigger 1.0, onsets pooled over the 12 basins, test years.
1. **Gain** (all three): every single seed of `wqME` above every single seed of `lstm_wqa` on onsets warned; the onset false share of
   the seed-averaged `wqME` not higher than that of `lstm_wqa`; Wilcoxon signed-rank (two-sided) over the 12 basins on the per-basin
   onsets warned of the seed-averaged forecasts, p < 0.05.
2. **Capacity control flat:** the median of the five single-seed onsets-warned values of `wqME_shuf` lies within the lowest-highest
   single-seed range of `lstm_wqa`. If `wqME_shuf` gains as much as `wqME`, the gain is capacity, not information.
3. **Clock control flat:** the same test for `wqME_older`.
4. **Network yardstick (reported, not part of the verdict):** if ridge and forest gain as much from the `wqME` columns (against their
   `wqa` versions of script 16) as the LSTM does, the reading is "the information, not the network" again.

E7 is a clear gain only if 1, 2 and 3 all hold. Also reported: median NSE at leads 1-3, the 30%-false operating point
(`warning_policy.py`, as E4a), the resampling interval (`13_uncertainty.py` recipe), the GO / CAUTION / NOT YET verdict.
`wqME` carries the AORC hindsight bound of `wqa` (at most about 4 points) unless measured again.
Expected before running: 0 to 4 points over `wqa`, probably failing the strict seed rule, as E4b did.

## 5. The satellite case, stated plainly

Satellite data enters this problem in three places, and the honest expectation differs in each:

1. **Rain of the last hours before the forecast.** Satellite rain (IMERG Early) reaches a forecaster about 4 hours after it falls, so at local
   midnight it covers the day to about 19:00-20:00 - the rung that the NLDAS experiment approximates (0.59 -> 0.67 NSE, 19% -> 30% of onsets).
   Radar (Stage IV, MRMS; AORC as the retrospective stand-in) covers the day to about 23:00 - the rung E1 measures. In the United States a
   service would use radar; in most of the world satellite rain is the only real-time rain there is, and that is the case Google's global
   system makes (`results/published_context.json`, Nearing et al. 2024: IMERG Early, ERA5-Land and ECMWF forecasts as inputs).
   Step 1c measures it directly instead of through NLDAS.
2. **Rivers without a gauge.** Where there is no gauge, satellite rain plus a model trained on many rivers is all there is. Our unseen-river
   test with 11 training basins lost 0.13 NSE; published models trained on 531 basins reach 0.69 for unseen basins (same-day simulation).
   Step 1d is the first measurement of what more rivers buy here; the satellite-rain input of `08_satellite_inputs.py` is its partner.
3. **How wet the ground is.** Satellite soil moisture (ESA CCI from 1978; SMAP only from 2015) is the one satellite quantity the gauge
   does not already tell the model. Expect a small gain with a gauge, possibly more without one; nobody has published the number.

The case *against* over-promising: satellite rain misses or misplaces single events (on one checked day IMERG Late read 45 mm where radar
read 24 mm in one basin), and no satellite product ends the 47% of the rain that has not yet fallen when the forecast is issued. The
satellite story is strongest as "what a global, gauge-free service could do", measured against the gauge-and-radar service in the same
basins.

## 6. Costs, and the "cluster" question

No cluster is needed at any step. The network has about 19,000 parameters; a training takes about three minutes on this laptop and the
63-basin and 516-basin runs under an hour each. Prices seen on 2026-10-07 (`results/data_access.json`, costs; AWS from its price files,
the rest from secondary pages and marked so):

| Item | Laptop | Paid option | Who can pay from China |
|---|---|---|---|
| GEFS reforecast rain, control member, 2000-2014 (61 GB) | free, 5-30 h over one or two nights, through the proxy | AWS c6i.xlarge in us-east-1, 6-12 h: USD 1.55-2.57 (reads from the bucket are free there; USD 100 sign-up credit); Alibaba Cloud Virginia: USD 2.3-3.2 | AWS needs an international card or a UnionPay *credit* card with a Chinese ID; Alibaba Cloud International takes Alipay |
| 5 ensemble members instead of 1 | 5x the above | the same VM for a day or two | as above |
| 516-basin training with equal training per basin | 7-20 h | AutoDL RTX 4090 about CNY 9-14 for 4 h (Alipay/WeChat; bring the data, its proxy does not reach NOAA); Kaggle about 30 GPU-hours a week free (unverified); AWS g4dn.xlarge USD 2.10 | AutoDL from China directly |
| AORC, IMERG Early, ESA CCI, NWM, Stage IV | free | - | - |

Recommendation: spend nothing until E2 is decided; then try the laptop download for one night with six parallel streams (free). If the
link gives less than about 0.5 MB/s, rent the VM for an afternoon (a few dollars).

## 7. Decisions for the team lead and the team

1. **E2 download:** DECIDED 2026-10-07 evening (the team lead): laptop, free. DONE the same evening: `code/fetch_gefs.py` fetched the control
   member's rain to forecast hour 78 for all 5,392 days 2000-01-01..2014-10-05 by byte ranges (48 GB read at 13-17 MB/s in about one hour,
   not overnight) and kept only the south-eastern box as monthly files (`data/gefs/c00/`, 141 MB; `--check`: consistent, complete;
   `data/gefs/README.md`). E2 itself (basin means, the two experiments) started at 22:10; see step 2 result when written.
2. **Nothing to buy for training now.** A GPU only if the 516-basin run with equal training per basin is wanted (about CNY 10-15).
3. **A free NASA Earthdata account** is no longer required for anything in steps 0-2; it would add NLDAS-2 hourly rain and the
   gauge-calibrated IMERG as cross-checks. Optional.
4. **When to rebuild the documents** (report, slides, guide): after E1's check, or after E2? This depends on the hand-in date, which is
   still not stated anywhere in the project.
5. **Which members own which step** - 1c and the soil-moisture input are natural for the RS/GIS member; the scoring of new sets for GNSS
   member 1; the training records for GNSS member 2 (the plan's roles).

## 8. What was done today, and what was not

Done by the night of 2026-10-07/08, each with an independent review record in `results/review_*.json` and the reviewers' scripts under
`code/review/`: E1 (ERA5 rain to midnight, 21 trainings), E1b (AORC rain to midnight, 12 trainings, plus the hindsight check and the
Stage IV real-time test on all 611 onsets), E3 (63 training rivers and 50-river never-seen test, 15 trainings, with the engine's optional
basin-set argument), E2 (the GEFS reforecast: 48 GB read in an hour, basin means, 30 trainings on two designs). Every results file carries a
computed `outcome_full` that documents must quote whole. Open for all of them: the `13_uncertainty.py` ranges, five seeds instead of three
where NSE decides, and the documents (report subsections, key-figure rungs, slides, guide), which wait for the hand-in date.

Done (2026-10-07, daytime): the diagnostic of section 2 (`check_onset_rain_clock.py`); E1 built with saved networks and run (result below);
access to every data source of section 3 measured; cost research; literature record of 19 sources (`results/published_context.json`);
scoping of E3 and P2 with the 51 extra basins verified. Not done: the independent check of E1 (rule of the work order; next), the report
subsection and key-figure rung for E1, E1b-1e, E2, any download larger than a sample, any purchase, any account.

**E1 result** (`results/rain_to_midnight.json`, run of 2026-10-07 11:21-11:41: 12 trainings in 19 minutes; test years, 1 day ahead,
pooled over the 12 basins; in brackets the lowest and highest of the three single trainings):

| Inputs, with the river gauge | Median NSE | Onsets warned | Warnings false |
|---|---|---|---|
| Rain table alone (`lstm_wq`) | 0.59 (0.54-0.60) | 18.7% (18.0-18.5%) | 59% |
| + NLDAS rain, day ends about 17:30 (`lstm_wqn`) | 0.67 (0.62-0.67) | 30.0% (29.0-31.6%) | 63% |
| + ERA5 rain to midnight, **new** (`lstm_wqh`) | 0.66 (0.57-0.68) | 30.0% (27.5-31.6%) | 56% |
| control: the same ERA5 rain one day older (`lstm_wqh_older`) | 0.62 (0.57-0.61) | 21.4% (19.5-22.4%) | 62% |
| + perfect weather forecast (`lstm_wqf`), the upper bound | 0.86 | 64.3% | 26% |

Without the gauge the pattern is the same: 17.2% -> 31.6% (NLDAS) -> 31.9% (to midnight; 31.6-32.6%), control 19.5%, upper bound 69.9%.

Outcome sentence of the results file: *Given the rain of the whole issue day up to local midnight (hourly ERA5 rain from 00:00 to 24:00 by
the local wall clock) as one extra input to the LSTM with the river gauge, onsets warned 1 day ahead go from 19% to 30% (control 21%); the
verdict is NOT YET.* The rain to midnight closes 25% of the onset gap to the upper bound (19% beyond its control), the same share as NLDAS
rain; every single training with it warns of more onsets than every training of the control; against NLDAS there is no clear difference
(the seed ranges overlap), so the E1 success rule of the work order ("clearly above 30% in every seed") is **not met**. On NSE the new input
is higher than its control in 12 of 12 basins (p = 0.0005) but one weak training (0.57) keeps the seed ranges overlapping: by the strict rule
of report section 4.2, no clear difference. Two and three days ahead nothing changes, as expected: that is forecast territory.

**Extension of the same day (17:27; `--extend` mode, 9 more trainings, the simple-model yardstick; nothing retrained):** three
controls with the gauge answer what the first run left open (seed averages; in brackets the single trainings; 611 onsets):

| Input set with the river gauge | Onsets warned | Hits | Median NSE |
|---|---|---|---|
| Rain table alone (`lstm_wq`) | 18.7% (18.0-18.5) | 114 | 0.59 |
| ERA5 over the table's own window, 09:00 to 09:00 (`lstm_wqh_to09`): a second product, no new hours | 20.5% (18.2-22.4) | 125 | 0.58 |
| ERA5 over the UTC day, ends 19:00 local (`lstm_wqh_utc`): what a satellite product by UTC day would hold | 21.8% (21.8-23.9) | 133 | 0.59 |
| ERA5 to midnight (`lstm_wqh`) | 30.0% (27.5-31.6) | 183 | 0.66 |
| NLDAS rain, ends about 17:30 (`lstm_wqn`) | 30.0% (29.0-31.6) | 183 | 0.67 |
| NLDAS and ERA5 to midnight together (`lstm_wqnh`) | 31.8% (30.1-32.4) | 194 | 0.69 |
| Ridge regression / random forest given the same new input (`base_linreg_wqh`, `base_rf_wqh`) | 26.8% / 27.0% | 164 / 165 | 0.62 / 0.66 |

What it means (fields `value_of_hours_09_to_24`, `value_of_evening_hours_beyond_utc_day`, `second_rain_product_alone`, `both_products`,
`yardstick` of the results file):
- **The hours are what count, and the last ones most.** The ERA5 rain of the hours 09:00 to 24:00 of the issue day is worth +9.5 points of
  onsets warned (every training above every control training: a clear gain), and +8.2 of those points come from the evening hours after
  19:00, the end of the UTC day; the daytime hours from 09:00 to 19:00 add only +1.3 (no clear difference). A second rain product with no
  new hours adds +1.8 (flat).
- **The product counts as well.** Over about the same window NLDAS (a gauge-based analysis) warns of 30.0% and ERA5 (model rain) of 21.8%.
  So the first run's tie with NLDAS was ERA5's quality cancelling the gain of its hours.
- **The two together warn of more, a little:** 31.8% (194 hits) against 183 for each alone; the two products warn of partly different onsets
  (133 shared, 233 by at least one); no clear difference by the seed rule.
- **Simple models given the same input gain about as much** (ridge +16 points, forest +9, LSTM +11): no clear difference between the LSTM
  and the forest at any lead - the project's main finding holds for the new input too.
- Two and three days ahead nothing changes; that is forecast territory. The verdict stays NOT YET.

Consequence for the plan: step 1b (AORC, gauge- and radar-anchored hourly rain with the right clock) is now the obvious next input - it
should combine NLDAS's quality (+11 points) with the evening hours (+8 points). For the satellite real-time case the lesson is sharper than
expected: a product that reaches the forecaster about 4 hours late loses exactly the hours that carry the gain, so a service built on satellite
rain should test issuing its forecast a few hours after midnight rather than at midnight (a design question for step 1c). What is still not
done for E1: the resampling ranges of `13_uncertainty.py` for the new sets, the report subsection and key-figure rung, The independent review ran three rounds (`results/review_rain_to_midnight.json`; its scripts are in `code/review/rain_to_midnight/`):
every number of every version was reproduced with the reviewer's own code, all 21 networks rebuilt bit for bit from independently built
inputs, no leakage; no must-fix remains. Open wording points the reviewer left: the two controls rise 2.8 and 1.8 points above the rain
table and the caveat calls one "not flat" and the other "flat" only because of the seed rule, and the caveat does not say that the LSTM
ends about 3 points above the two simple models on onsets (sign test p about 0.04; equal on NSE and CSI).
