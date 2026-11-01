# Next experiments: how to make the LSTM warn more onsets (work order of 28 October 2026)

Written by Claude Code (Fable 5.1) on the Mac on the evening of 2026-10-28, after reviewing the separate `Flood_Cloud_Model` package
(`handoff/extras/Flood_Cloud_Model_REVIEW_2026-10-28.md`) and after two diagnostics on this project's stored forecasts. It extends
`HANDOFF.md` section 4 and `ROADMAP.md`; it does not replace them. Every number below is read from a file named beside it; the numbers of
`results/diagnostics_2026-10-28/` are **preliminary** (one pass, no seed ranges, no intervals, not reviewed) and must not be quoted in the
documents before an experiment with its own review record reproduces them. Deadline for the hand-in: **5 November 2026**.

## 0. What was learned, in one paragraph

The warning is limited by the information the model has at the issue time, not by the network (`README.md`, "What we found"); this day's work
says *which* information. (1) With hourly flow at the issue time, the Flood_Cloud_Model package finds that observed rain adds almost nothing at
24 h while the forecast adds a lot: the state of the river at local midnight is the one input this project's LSTM has never seen (it reads the
daily mean of day t). (2) On the issue days of the test-year onsets, a median 45 % of the day's rain fell after 18:00 local time, against 7 % on
other wet days, and the wettest hour was 6.6 mm against 2.8 mm (`results/diagnostics_2026-10-28/subdaily_inputs_check.json`): the daily total hides
exactly what distinguishes an onset day. (3) At the GO rule's false share (30 %), the realistic models warn of 14-33 % of the 1-day onsets and
the perfect-rain models 57-61 % (`results/diagnostics_2026-10-28/warning_policy_rescore.json`): with daily inputs, GO at one day is out of reach
even with perfect rain, so a better decision layer alone cannot get there; new inputs can. (4) The USGS 15-minute flow of all 12 gauges is
available back to 1985 through the new USGS API (checked 2026-10-28, section 6), and the four perturbed members of the GEFSv12 reforecast are on
AWS beside the control member the project used.

## 1. Rules (unchanged; `HANDOFF.md` section 7)

Test years are scored once per experiment; every choice (trigger, probability cut, settings, stopping point) on the validation years; new
scripts start at `20_`; each experiment gets an independent review that recomputes the headline numbers from `results/preds`; five seeds where a
verdict rests on a point or two; numbers reach the documents only through `results/*.json` and the builders; preliminary numbers are labelled;
no bulk download without the exists / available / worth-it check and its cost stated first (the checks of section 6 are that check for E4d and E4e).

## 2. The experiments, in the order to do them

| # | Name | What it needs | Cost | What it can change | Script |
|---|---|---|---|---|---|
| E4a | Warning level and stacker on stored forecasts | nothing new (`code/warning_policy.py`) | a day | a second, honest operating point (30 % false) in the ladder; the window-scoring diagnostic | `20_warning_policy.py` |
| E4b | Within-day rain and Q95-relative flow as inputs | nothing new (`code/subdaily_inputs.py`, `data/aorc`) | 10-20 trainings, a day | lead-1 onsets, especially the 220 with little rain on the target day | `21_within_day_inputs.py` |
| E4e | The river at the issue time (hourly USGS flow) | download, about 2 GB, about 4 h (`code/fetch_usgs_iv.py`) | a night + 10-20 trainings | the lead-1 input the LSTM has never had; the only lever left toward GO at one day | `22_flow_state.py` |
| E4c | A warning head: exceedance probability beside the flow | code (a copy of `flood.fit`) | 2-3 days | sets the false share explicitly; cumulative windows; calibrated probabilities | `23_warning_head.py` |
| E4d | All five reforecast members, sub-daily bins, no bias factor | download, 4 x about 48 GB streamed, free | a night + 15-30 trainings | 2-3 day onsets, over-trust of exact slots | `24_forecast_members.py` |
| E4f | The LSTM on the 2021-2025 operational inputs of the package | nothing new (`handoff/extras/Flood_Cloud_Model/data`) | 2 days | shows whether the finding transfers to operational inputs; optional | `25_modern_split.py` |

For the hand-in of 5 November: E4a and E4b, each with its review; E4e if the download is started on the first evening; E4c and E4d after.

### E4a. Warning level and stacker on stored forecasts (no training)

`code/warning_policy.py` already does the work as a diagnostic; `20_warning_policy.py` turns it into an experiment: run it on the seed-mean sets
AND on each seed's set (seed ranges), add the block bootstrap of `13_uncertainty.py`, write `results/warning_policy.json` with `outcome_full`,
and get it reviewed. What it measures (test years, onsets, lead 1; `warning_policy_rescore.json`, preliminary):

| Set (main split, 611 onsets) | project rule (best validation CSI): trigger, warned, false | at most 30 % false on validation: trigger, warned, false | logistic stacker, cut at 30 % false: warned, false |
|---|---|---|---|
| lstm_wq (rain table + gauge) | 1.00, 18.7 %, 59.0 % | 1.55, 2.3 %, 30.0 % | 1.1 %, 30.0 % |
| lstm_wqa (+ AORC rain to midnight) | 1.00, 51.2 %, 50.4 % | 1.60, 24.7 %, 29.8 % | 23.7 %, 28.9 % |
| lstm_wqaF_sub (+ GEFS control, substituted) | 0.90, 53.8 %, 43.9 % | 1.65, 14.2 %, 25.0 % | 27.2 %, 24.2 % |
| lstm_wqaP (perfect rain in the slots) | 1.00, 66.1 %, 34.9 % | 1.10, 58.9 %, 28.7 % | 60.4 %, 30.6 % |
| lstm_wqf (perfect weather) | 0.95, 67.8 %, 30.8 % | 1.05, 60.6 %, 22.1 % | 63.8 %, 25.1 % |
| Satellite-era split, 335 onsets: lstm_wqa_s | 1.05, 37.6 %, 50.8 % | 1.50, 14.9 %, 24.2 % | 14.6 %, 22.2 % |
| lstm_wqaF_s (retrained with the forecast) | 1.00, 61.2 %, 46.8 % | 1.40, 33.1 %, 25.5 % | 34.6 %, 28.4 % |
| lstm_wqaP_s | 1.15, 61.2 %, 29.8 % | 1.20, 57.3 %, 27.8 % | 57.9 %, 28.4 % |

Reading: the project's CSI rule puts every realistic model at 44-59 % false warnings among the onset warnings; the GO rule allows 30 %. At 30 %
false the realistic models keep 14-33 % of the onsets; the stacker (logistic regression on the forecast over Q95, the seed spread and today's flow
over Q95, fitted on 2000-10..2003-03, cut chosen on 2003-04..2005-09) moves wqaF_sub from 14 % to 27 % at that false share and the others by a point
or two: a decision layer re-arranges the trade-off and does not create skill. At leads 2 and 3 the same holds (wqaF_sub 10.5 % and 13.5 % warned at 26.5 % and 30.7 % false, with the caveat that no trigger
of the grid kept its validation false share under 30 % there, so the lowest-false trigger is reported; perfect rain 69 % and 74 %). Window scoring (a target day counts as warned if any earlier issue warned of it): 45 of the 298 lead-1
misses of wqa and 58 of the 282 of wqaF_sub were warned two or three days before, so about a sixth of the lead-1 misses are timing, not blindness;
the union's false share is high because the lead-2/3 triggers of the CSI rule (0.75) fire often. Two things to carry into the ladder: report every set
at the 30 %-false operating point beside the CSI one, and report "warned by any lead" with its false share. The Flood_Cloud_Model
"persistence budget" rule (all-day false alarms at most persistence's) reproduces the CSI trigger for the realistic sets and is not needed.

Also in E4a: the gradient-boosting stacker of the package (`handoff/extras/Flood_Cloud_Model/run.py`: `train()` = HistGradientBoostingClassifier
with `max_leaf_nodes 15, min_samples_leaf 50, learning_rate 0.04, l2 2.0, 150 iterations`, Platt calibration, `choose()`), on the same features plus
AORC rain to midnight and the forecast slots. Keep it that small: the fitting half holds about 11,000 days and a few hundred onsets.

### E4b. Within-day rain and Q95-relative flow as inputs (data already here)

`code/subdaily_inputs.py`: `add_inputs(D, thr)` joins to every basin's daily table the columns `aorc_last3h`, `aorc_last6h`, `aorc_last12h`
(rain of the last hours of the issue day, wall clock of the gauge, summer time included, hours ENDING at or before 24:00), `aorc_max1h`,
`aorc_maxcell1h`, `aorc_wet_hours`, `aorc_filled_hours`, `aorc_to_midnight` (identical to script 16's column: two routes agree on every day), and
`q_rel = log1p(q_mm / Q95)`. Its self-checks pass (routes, order of the windows, 23/25-hour clock-change days, a made-up-hours test that proves
no value reads an hour after the issue time). Why it should matter (test years, issue days with at least 1 mm; `subdaily_inputs_check.json`):

| | onset issue days (541 of the 611 lead-1 onsets) | other wet issue days |
|---|---|---|
| median share of the day's rain after 18:00 | 0.45 | 0.07 |
| days with at least half the rain after 18:00 | 47 % | 29 % |
| median wettest hour (mm) | 6.6 | 2.8 |

70 onsets had under 1 mm on the issue day: those need the forecast or the river's state (E4e), not this.

Sets to train (settings of `results/lstm_all.json`, three seeds, then five): `wqaW` = wqa + the six within-day columns; `wqaWF_sub` = wqaF_sub +
the same; `wqaQ` = wqa + `q_rel` alone (to isolate the scaling change; the LSTM scales flow with one global mean and spread, `flood._design`, so
"near the flood level" is a different number in every basin and the network must decode it from the basin properties). Score as script 16 does
(seed rule, paired onsets, per-basin, the closure of the gap to wqaP) and additionally at the 30 %-false operating point of E4a. Expect the gain
in the 220 onsets with under 5 mm on the target day (`results/review_forecast_input.json`, must-fix 1), where the forecast model lost ground.
Where to plug in: build `D2 = subdaily_inputs.add_inputs(D)` and pass the new names in `cfg.dyn` (the engine reads `dyn` columns up to the issue
day; `_assert_causal` and `features_are_causal` of script 16 remain the proofs). The columns hold mm and counts; the engine standardizes them.

### E4e. The river at the issue time: hourly USGS flow

The input the Flood_Cloud_Model evidence points at. On its 2024-2025 cases, once the model sees the flow at T-2h with its 72 h trend, observed
rain lifts the 24 h onset hits only from 102 to 114 of 886 while the forecast lifts them to 278 (review, section 3.5). This project's AORC rain was
worth 32 points (19 % to 51 %) partly because it stands in for the rising limb the daily mean hides.

Data: `code/fetch_usgs_iv.py` (section 6 has the check). Run `python code/fetch_usgs_iv.py --start 1985-10 --end 2014-09` on the first evening
(about 4,200 requests, about 2 GB, about 4 h in one thread; resumable; the free API key if throttled). It writes hourly files with hour-ENDING
UTC stamps (the convention of `data/aorc`): `q_mean_cfs`, `q_last_cfs`, `n_values`, `approved`.

Inputs of day t (all at or before the issue time, local midnight; use the zone of `subdaily_inputs.zones()`): `q_end = log1p(q_last of the hour
ending 24:00 / Q95)`, `dq6 = log1p(q_end_cfs / Q95) - log1p(q at 18:00 / Q95)`, `dq12` likewise from 12:00, `q_dmax = log1p(max hourly flow of day t
/ Q95)`, `hours_above = hours of day t above Q95 (0-24)`. Be conservative about delivery: the package assumed the last available hour is T-2h; take
the hour ending 22:00 as "last available" in a second variant and report both. Convert cfs to mm/day as `camels.py` does (area_gages2).

Coverage: five gauges start between 1986 and 1992 (section 6). `flood._first_valid` would then drop every training day before 1992-10 for all basins.
Two ways: (a) a flag column `hourly_known` (1/0) and, where the hourly flow is missing, `q_end = log1p(daily mean / Q95)`, `dq6 = dq12 = 0`,
`q_dmax = q_end`, `hours_above = 24 x (daily mean > Q95)`, so the main split keeps 1980-2000; (b) the satellite-era split of script 18 (2000-2014),
where every gauge is complete. Do (b) first (cheap, clean), then (a). Sets: `wqaH` (wqa + the five columns), `wqaWH` (with E4b's columns),
`wqaWHF` (with the forecast). This is the experiment that can move the lead-1 ceiling itself, because the perfect-rain sets (wqaP: 66 %
warned at 35 % false; 59 % at 29 %) also read only the daily mean flow.

### E4c. A warning head: exceedance probability beside the flow

Now: squared error on linearly scaled flow (`flood.fit`, weights per basin), early stopping on mean validation NSE, warning = regressed flow
> trigger x Q95. A regressed conditional mean sits below Q95 whenever the chance of crossing is under about one half, so such onsets are missed by
construction; the trigger of the CSI rule (0.75-1.0) compensates by issuing many false warnings. The package predicts the event itself with a
calibrated probability and picks the cut on a held-out half.

Change (copy `fit` into `code/flood_prob.py` as `fit_prob`; leave `flood.py` as it is so every old result stays reproducible):
`LSTMNet.head = nn.Linear(hidden, 2 * len(LEADS))`; outputs `[:, :, :3]` stay the scaled flow, `[:, :, 3:]` are logits of "flow on day t+k above
Q95 of the basin" (thresholds of the split's training years, `flood.thresholds`); loss = the present weighted squared error on the flow outputs +
lambda x `BCEWithLogitsLoss(pos_weight = (1 - r) / r)` on the logits, r = the share of high-flow training days (about 0.05, so about 19); lambda chosen
on validation among {0.3, 1, 3}; `_assert_causal` unchanged (more outputs, same test). Early stopping: first keep mean NSE (comparable with the
ladder), then try the mean of NSE and the onset average precision on validation (report both). Store the probabilities with the forecasts
(`save_preds` gains an optional `prob` array; `load_preds` adds `p1..p3` columns when present), calibrate on the first half of the validation
years, choose the probability cut on the second half with the 30 %-false rule (`warning_policy.choose_trigger` works on any monotone score: pass
the probability as `ratio`), score the test years once. Second step: three more logits for the cumulative windows "above Q95 on any of days
t+1..t+k" (max of the next k days), the label the package uses and the operational question; score them with `warning_policy.any_lead` logic.

### E4d. All five reforecast members, sub-daily bins, no bias factor

The project fed the LSTM one bias-corrected daily total per lead from the control member; the reviewer found the single bias factor noisy and a
conditional bias left (the forecast carries 0.63 of the AORC rain on wet days), and raw and corrected forecasts give the same onsets
(`results/review_forecast_input.json`, should-fix 5). The package feeds raw per-interval ensemble statistics and lets the model learn.

Data: the GEFSv12 reforecast has five members every day (c00, p01-p04; 11 on Wednesdays, ignore those). `code/fetch_gefs.py` fetched c00 with one
Range request per day (48.5 GB streamed, 53 min at 11-15 MB/s on 2026-10-07); add a `--member` option and fetch p01-p04 the same way (about 4 x 48 GB
streamed, about 0.6 GB kept, free; check the speed on the new network first: 0.5-4 MB/s was also seen through the proxy). `code/gefs_to_basins.py`
then builds per-member basin tables (its `day_totals` and `rain_by_wall_clock_day` already split 3-hourly windows at local midnight).

Sub-daily bins for lead 1 without any download: the control member's 3-hourly boxes are cached in `data/gefs/c00` (147 MB). Replace the single
column for day t+1 by the rain of its first and second half (or four 6-hour bins): rain at 01:00 and rain at 22:00 of day t+1 do different things to
that day's mean flow. Keep daily totals for t+2 and t+3.

Members as scenarios rather than as spread features: in a copied trainer, keep one input tensor per member and draw a random member for every
training sequence (the network learns the forecast's error instead of trusting exact slots, which is the reviewer's open "over-trust"
hypothesis), then predict with all five members and three seeds: fifteen flow trajectories, the exceedance fraction is the probability, the
member mean the flow forecast. Drop the bias factor. If the member-as-scenario route is too much before the deadline, the package's
`preprocessing.stats()` (mean, variance, P10/P50/P90, fraction of members above 25 and 50 mm, per interval and cumulative) is the fallback
feature set. Constraint: real forecasts exist only from 2000, so this is the satellite-era split (E2b) or the main split with the perfect rain of
1980-2000 degraded by the measured error (the mixture idea of `what_was_not_done` in `results/forecast_input.json`).

### E4f. The LSTM on the 2021-2025 operational inputs (optional)

`handoff/extras/Flood_Cloud_Model/data/{training,calibration,selection,testing}.npz` hold, for every 6-hourly issue of the 12 basins, 268 prepared
features (57 flow/static, 21 MRMS rain, 190 GEFS ensemble; names in `feature_schema.json`), the labels `event` (above Q95 within 6/24/72 h), `onset`
(flow at T-2h at or below Q95), `issue`, `basin`. They load with plain numpy (also numpy 1.26). A sequence model over the 6-hourly rows per basin,
trained on 2021-2022, chosen on 2023 and scored once on 2024-2025 against the package's own boosting results on the identical cases (its
`reference/results.json`, onset-only recall at 24 h: 12.9 % flow + rain, 31.4 % with GEFS) would show whether "information, not the network"
holds with operational inputs. Score on distinct starts (its `run.py: event_score`) and on onset cases, never on its headline table (review,
section 3). Leaf River is nearly absent there (84 test cases).

## 3. Ceilings to keep in view

Perfect rain in the slots gives 66 % of the 1-day onsets at 35 % false and 59 % at 29 % (wqaP); perfect weather 68 % at 31 %. So with daily inputs,
70 % warned at 30 % false is not reachable at one day by any rule or decision layer; E4e is the only experiment above that can move this ceiling.
At two and three days the perfect-rain ceiling is 69-74 % at about 30 % false against 10-14 % now (wqaF_sub) and 33-35 % at 34 % false in the
retrained E2b model: that is where E4c and E4d pay. Report the gap to the ceiling with every result, as the ladder does.

## 4. What NOT to do

Do not import the package's trained models or its 2021-2025 arrays as inputs to the 1980-2014 models. Do not quote its headline table
(97.5 % accuracy, 68.7 % detection): they count cases where the river was already above Q95. Do not drop AORC rain to midnight because the package's
observed rain "adds nothing": in a daily model it is worth 32 points; the lesson is about hourly flow, not about rain. Do not tune anything on
2005-2014 (or 2010-2014 for the satellite-era split).

## 5. Files added on 2026-10-28

| File | What it is |
|---|---|
| `NEXT_EXPERIMENTS_2026-10-28.md` | this work order |
| `handoff/extras/Flood_Cloud_Model/` | the package, unchanged (its own `MANIFEST.json` verifies it; Python 3.12 environment, see its README) |
| `handoff/extras/Flood_Cloud_Model_REVIEW_2026-10-28.md` | the critical review and the comparison with this project |
| `code/warning_policy.py` | warning rules (CSI, 30 %-false cap, persistence budget), the logistic stacker, any-lead window scoring; `python code/warning_policy.py` rewrites the diagnostic |
| `code/subdaily_inputs.py` | within-day AORC inputs and `q_rel`; self-checks and the onset timing statistic |
| `code/fetch_usgs_iv.py` | the hourly USGS flow fetcher (tested on two months; the test files were removed) |
| `results/diagnostics_2026-10-28/warning_policy_rescore.json`, `subdaily_inputs_check.json` | the two preliminary diagnostics quoted above |
| `handoff/memory/flood-improvement-plan-2026-10-28.md` | Claude's memory note of this day |

`HANDOFF.md` (section 9) and `README.md` (one row, one line) point here; `handoff/MANIFEST.sha256` lists the new files. Nothing else in the project
was changed by this work order. On 2026-10-28 at about 16:10 these additions were merged into the Mac folder `~/AI_Class/Project-Flood` and into
the rebuilt transfer zip, together with the portability fixes of the hand-over session (`HANDOFF.md` sections 3 and 9).

## 6. Data checks made on 2026-10-28 (exists / available / worth it)

**USGS 15-minute discharge (parameter 00060), all 12 gauges.** Period of record from the USGS site catalogue: 02077200, 02108000, 02137727,
03439000, 03463300 from 1985-10-01; 02472000 from 1983-10-01; 02128000, 02298608 from 1986-10-01; 02221525, 02349900 from 1989-10-01; 03473000 from
1990-10-01; 02296500 from 1992-10-01; all to date. Available: `https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items`
(`monitoring_location_id=USGS-<id>&parameter_code=00060&datetime=<start>/<end>&f=csv&limit=10000`) returned 2,976 approved 15-minute values for
03439000 in May 2003 in 3.5 s (0.5 MB), and 96 values for single days in 1995, 2003 and 2012; the legacy `waterservices.usgs.gov/nwis/iv`
returned an empty body for the same requests. No API key was needed for a dozen requests; the limit without a key was not measured
(free key: `https://api.waterdata.usgs.gov/signup/`). Worth it: section 0 (1) and E4e. Cross-check: the hourly-derived local-day means of
03439000 for May-June 2003 correlate 1.0000 with the CAMELS daily flow and sit 1.6 % lower (the CAMELS daily value is USGS's own daily mean).

**GEFSv12 reforecast perturbed members.** `https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/2005/2005010100/p01/Days:1-10/apcp_sfc_2005010100_p01.grib2`
exists (22.3 MB; p04 22.4 MB; c00 25.7 MB), so the four perturbed members are there for the same days as the control member the project fetched.
Worth it: E4d; the project's own `results/gefs_basins.json` (skill of one member) and the package's 31-member design say why.

**Within-day AORC rain.** Already in `data/aorc` (298,296 hourly basin means per basin, 1980-09-25 to 2014-10-05, with the wettest cell per hour
and the ERA5-filled flag); no download.

## 7. Environment notes for the new machine

Project: Python 3.11, `requirements.txt` (torch 2.2.2 with numpy 1.26.4), `handoff/environment/venv_freeze.txt`; `code/fetch_usgs_iv.py` needs
only `requests`, `pandas`, `numpy` (all in the project environment); `code/warning_policy.py` uses scikit-learn (pinned there). The
Flood_Cloud_Model package needs its own Python 3.12 environment with its seven exact pins (`python run.py verify` refuses anything else); never
install it into the project environment. The checks of `HANDOFF.md` section 5 come first: `python code/test_flood.py`, `FLOOD_QUICK=1 ./run_all.sh`,
one saved network reloaded against `results/preds`. Then `python code/warning_policy.py` and `python code/subdaily_inputs.py` should print
"all checks passed" and rewrite the two diagnostics with the same numbers as the files shipped here (the first takes about 40 s, the second 5 s).
