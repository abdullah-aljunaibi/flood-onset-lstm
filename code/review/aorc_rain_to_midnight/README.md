# The independent reviewer's check scripts for experiment E1b (AORC rain up to local midnight)

These scripts belong to the **independent AI review** of experiment E1b (`code/16_aorc_rain_to_midnight.py`, its fetcher `code/fetch_aorc.py`, the cache `data/aorc/`, the builder's hindsight check
`code/check_aorc_hindsight.py`, and the results file `results/aorc_rain_to_midnight.json`). They were written by an AI reviewer agent (Claude), separately from the agent that wrote and ran E1b, and they are
**not part of the project's own code**: `run_all.sh`, the report builders and the numbered scripts neither run nor read them. The review itself is `results/review_aorc_rain_to_midnight.json` (checks, numbers,
discrepancies, must-fix and should-fix items, verdict, and a plain statement of how far the 51% can be trusted). That record states that the team's own check of E1b is still open; this AI review does not replace it.
The scripts of the review of E1 (`code/review/rain_to_midnight/`) are a separate folder and were not changed.

The scripts recompute E1b's numbers with the reviewer's own code, from `results/preds`, `results/models`, `data/aorc`, `data/camels_us` and, for the hindsight question, from hourly NCEP Stage IV rain that the
reviewer downloaded itself. They import only the engine (`code/flood.py`: `load_data`, `basins`, `thresholds`, `evaluate`, `load_preds`, `LSTMNet`, `Run.predict`, `preds_meta`), `code/features.py` and `code/camels.py`,
adding `code/` to `sys.path` as the numbered scripts do. **Nothing is trained and nothing is written into the project** (the output of every script goes to `$REVIEW_OUT`, default `<temp folder>/aorc_review`;
`PYTHONDONTWRITEBYTECODE` is set, so no `__pycache__` appears here), except that `a15_assemble_review.py --write-into-project` writes the review record.

## Run

From the project root, with the project environment, one thread:

```bash
~/AI_Class/.venv/bin/python code/review/aorc_rain_to_midnight/<script>.py          # one check (a10_simple.py takes ridge | forest | perturb; a6_stage4.py takes days | after | labels | check)
code/review/aorc_rain_to_midnight/run_fast_checks.sh      # 10 checks, no network, about 2 minutes (133 s in the final run)
code/review/aorc_rain_to_midnight/fetch_stage4.sh         # NETWORK: the Stage IV hours of the review's 28 days from the Iowa Mesonet archive, about 1.3 GB, 8-9 minutes, 6 connections, resume-safe (run once)
code/review/aorc_rain_to_midnight/run_slow_checks.sh      # 7 checks, about 8 minutes (490 s), needs run_fast_checks.sh and fetch_stage4.sh to have run with the same REVIEW_OUT; a5_cache.py makes about 4 MB of requests to the NOAA bucket
~/AI_Class/.venv/bin/python code/review/aorc_rain_to_midnight/a15_assemble_review.py   # builds the review record into $REVIEW_OUT; --write-into-project writes results/review_aorc_rain_to_midnight.json
```

Order matters in three places: `a3_clocks.py` writes `my_series_aorc.npz` (the reviewer's daily series), which `a4`, `a8`, `a10` and `a7a` read; `a7_hindsight.py` writes `a7_basin_days.csv`, which `a5b_stamp.py` and `a8` read
(the runners keep this order); `a1_integrity.py` should run before anything else so that its inventory is the baseline of `a15_assemble_review.py` (hashes of every reviewed file).
Optional variables: `PY` (the python), `REVIEW_OUT`, `AORC_CACHE` (a folder that already holds `stage4_means.csv`, so that no Stage IV file is fetched), `E1_ZIP_DIR` (the code folder of the hand-in zip of 4 October,
for the comparison of `flood.py` and `test_flood.py` with it), `AORC_RUN_LOGS` (the folder with the runner logs, for the times in the record), `R2G_SELFTEST=6` (moves the expected seed-bar ends of `a12_figure.py` by 6 px:
the check must then fail), and for `a7a_select_days.py` `N_ONSET_DAYS`, `N_NIGHT_DAYS`, `MIN_ONSETS_PER_DAY`, `MIN_DAYS_APART`.

## What each script checks (times of the final complete run on 2026-10-07: laptop, one thread, the 63-basin training and a download running)

| Script | Checks (brief item; check ids of the record) | Time |
|---|---|---|
| `a1_integrity.py` | sha256 of every reviewed file, the results file's own integrity record against the disk (30 saved files), script hash and `passes`, file times, `run_all.sh` lines, engine files against the zip (a, h; h2, h4, h5) | 2 s |
| `a2_headline.py` | headline numbers of every new set and the two simple models with the engine and with own pooled onset counts; seed averages; observed flow; per-basin picture (a; a1-a3) | 22 s |
| `a3_clocks.py` | own series of the three windows (wall-clock day with summer time by a hand-written rule and zoneinfo, 09:00-09:00, UTC day), membership, 90-case perturbation proof, scalers; writes `my_series_aorc.npz` (b; b1-b3) | 11 s |
| `a4_networks.py` | the 12 saved networks reproduced bit for bit through `flood.LSTMNet`/`Run.predict` from the reviewer's own tables (b; b4) | 42 s |
| `a14_readings.py` | the nine reading blocks, overlaps and sign tests, `gain_rule`, `e1b_success`, `closure`, verdict records, all recomputed (g; g4) | 18 s |
| `a9_byperiod.py` | the split at 2009-07-17, per-basin swing, AORC over Daymet by season (e; e1) | 5 s |
| `a10_simple.py ridge` | features, causality, settings, ridge refit (f; f1) | 9 s |
| `a11_outcome.py` | `outcome`, `outcome_caveat`, `outcome_full` rebuilt clause by clause; the NSE reading against the simple models; what each comparison changes for the last clause (g; g1-g3) | 5 s |
| `a12_figure.py` | `figures/aorc_rain_to_midnight.png` read on its pixels against the ladder of the JSON (h; h1) | 2 s |
| `a13_test_flood.py` | `code/test_flood.py` run unchanged with its output redirected; `flood.py`/`test_flood.py` against the zip (h; h3) | 17 s |
| `a7a_select_days.py` | the 28 days of the Stage IV sample (17 by onsets, 11 by night storms, no forecast used) | 5 s |
| `a5_cache.py` | Zarr metadata of three years, masks of all 12 basins, one basin-month and the filled and partial runs re-read from raw chunks (c; c1, c3, c4) | 44 s |
| `a7_hindsight.py` | the hindsight regressions (whole day, last six hours, 00:00-12:00, placebo, and a positive control: leaks of 3, 5 and 12% put into the data must be recovered), the factor model, dry-before-wet-after cases, the calibration for the network tests (d; d3, d4) | 60 s |
| `a5b_stamp.py` | stamp = end of the hour against Stage IV by lag, year and basin; structure of AORC against Stage IV; boundaries; ratio by era (c, d; c2, d2) | 13 s |
| `a8_hindsight_networks.py` | T1: saved networks given Stage IV for the issue day; T2, T3: dose-response of a leak of after-midnight rain (d; d5, d6) | 111 s |
| `a10_simple.py perturb` | no data after 2000-09-30 enters a simple-model fit (f; f1) | 81 s |
| `a10_simple.py forest` | the forest refit against `base_rf_wqa` (f; f1) | 186 s |
| `a6_stage4.py`, `fetch_stage4.sh` | own Stage IV reducer (rasterio, cell centres from the file's transform and projection) and its fetch; `check` compares with the builder's csv on all common hours (c; c5) | first fetch: 406 s + 67 s for 1,070 files |
| `a15_assemble_review.py` | builds the review record from the outputs above; refuses `--write-into-project` if a reviewed file changed | 2 s |
| `a16_compare_outputs.py OLD_DIR` | which outputs of two complete runs differ, leaf by leaf (the first piecemeal run against the final run: all identical, except the seconds of the forest refit and the fields that the Stage IV check and the positive control added) | seconds |

Helpers: `a_common.py` (paths, output folder), `a_scoring.py` (own NSE and onset counts), `a_clock.py` (own rebuild of the three windows from the cache), `a_stage4_data.py` (Stage IV and AORC series with the same hour labels), `a_simple_lib.py` (the simple-model refit).

## The decisive test that is not yet run

The Stage IV test of `a8_hindsight_networks.py` covers 171 of the 611 onsets (56 of 368 issue days). To cover every onset, fetch Stage IV for the other 312 issue days (about 10-13 GB, not the 88-97 GB of every test-year hour):

```bash
export REVIEW_OUT=...                                       # the folder of the earlier runs (it holds cache/stage4_means.csv)
N_ONSET_DAYS=400 N_NIGHT_DAYS=0 MIN_ONSETS_PER_DAY=1 MIN_DAYS_APART=0 code/review/aorc_rain_to_midnight/fetch_stage4.sh   # resume-safe: only the missing hours are fetched
~/AI_Class/.venv/bin/python code/review/aorc_rain_to_midnight/a7_hindsight.py ; ~/AI_Class/.venv/bin/python code/review/aorc_rain_to_midnight/a8_hindsight_networks.py
```

The Iowa Mesonet archive is run by volunteers: keep to 6 connections. The files measure 1.2 MB each (HEAD requests on 8 files: 1.12-1.37 MB).

## What the reviewer did and did not do

The reviewer downloaded 1,070 hourly Stage IV files (about 1.3 GB: more than the 'about 1 GB' it was asked to stay under, because it planned with 1.1 MB per file), about 4 MB from the NOAA AORC bucket, the NWS
methods document (1.3 MB) and the AWS registry page of the dataset. It did not run the project's numbered scripts, trained nothing, installed nothing, and wrote into the project only this folder and the review record.
