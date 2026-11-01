# The independent reviewer's check scripts for experiment E2 (the archived GEFSv12 reforecast rain as an LSTM input)

These scripts belong to the **independent AI review** of experiment E2 (`code/fetch_gefs.py`, `code/gefs_to_basins.py`, `code/18_forecast_input.py`, the data in `data/gefs/`, `results/gefs_basins.json`,
`results/forecast_input.json`, `figures/forecast_input.png`, the 30 networks in `results/models/` and the prediction sets in `results/preds/`). They were written by an AI reviewer agent (Claude), separately from
the agent that wrote and ran E2, and they are **not part of the project's own code**: `run_all.sh`, the report builders and the numbered scripts neither run nor read them. The review itself is
`results/review_forecast_input.json` (checks, numbers, discrepancies, must-fix and should-fix items, verdict, and a plain statement of how far the 2-3 day gains and the E2b 1-day result can be trusted). That record
states that the team's own check of E2 is still open; this AI review does not replace it. (The builder's `18_forecast_input.py` reads the `counts` of that file when it is there.)
The scripts of the reviews of E1 (`code/review/rain_to_midnight/`) and E1b (`code/review/aorc_rain_to_midnight/`) are separate folders and were not changed.

The scripts recompute E2's numbers with the reviewer's own code, from `results/preds`, `results/models`, `data/gefs`, `data/aorc`, `data/camels_us`, `data/hourly/usgs_sites.rdb` and, for the raw-data check, from three
raw archive days that `f2_raw_days.py` downloads itself. They import only the engine (`code/flood.py`: `load_data`, `basins`, `thresholds`, `evaluate`, `load_preds`, `LSTMNet`, `Run.predict`, `preds_meta`),
`code/features.py` and `code/camels.py`, adding `code/` to `sys.path` as the numbered scripts do. **Nothing is trained and nothing is written into the project**: the output of every script goes to `$REVIEW_OUT` (default
`<temp folder>/e2_review`), `PYTHONDONTWRITEBYTECODE` is set (no `__pycache__` appears here), and the two scripts that run project code that writes files (`f15_sandbox_rescore.sh`: `18_forecast_input.py --rescore`;
`f16_fetch_check.sh`: `fetch_gefs.py --check`) run it in a **sandbox copy** that is deleted at the end. Only `f19_assemble_review.py --write-into-project` writes into the project (the review record, and only when no reviewed file changed).

## Run

From the project root, with the project environment, one thread:

```bash
~/AI_Class/.venv/bin/python code/review/forecast_input/<script>.py          # one check (f11_simple.py takes ridge | forest | perturb)
code/review/forecast_input/run_fast_checks.sh      # 19 checks, no network, about 5 minutes (see the table); log in $REVIEW_OUT/fast_run.log
code/review/forecast_input/run_slow_checks.sh      # 6 checks, about 12 minutes (697 s); f2 reads 3 raw days (31.5 MB) from the open AWS bucket and, like f16, needs ~/AI_Class/.venv-grib; log in $REVIEW_OUT/slow_run.log
~/AI_Class/.venv/bin/python code/review/forecast_input/f19_assemble_review.py   # builds the review record into $REVIEW_OUT; --write-into-project writes results/review_forecast_input.json
~/AI_Class/.venv/bin/python code/review/forecast_input/f20_compare_outputs.py OLD_DIR   # which outputs of two complete runs differ, leaf by leaf
bash code/review/forecast_input/f15_sandbox_rescore.sh with-review   # after the record is in results/: does the script read its counts? (about 2 minutes)
```

Order matters in three places: `f4_basins.py` writes `f4_series.npz` (read by `f5`, `f6` through `f_tables.py`, and `f18`); `f5_skill.py` writes the bias factors that `f_tables.py` reads; `f6_tables.py` writes the table
pickles that `f7`, `f11`, `f12` and `f12c` read (the runners keep this order); `f1_integrity.py` should run before anything else so that its inventory is the baseline of `f19_assemble_review.py` (hashes of every reviewed file).
Optional variables: `PY` (the python), `PYG` (the GRIB python), `REVIEW_OUT`, `E1_ZIP_DIR` (the code folder of the hand-in zip of 4 October, for the comparison of `flood.py` and `test_flood.py` with it), `E2_RUN_LOGS`
(the folder with `fast_run.log` and `slow_run.log`, for the times in the record; default `$REVIEW_OUT`), `R2G_SELFTEST=3` (moves the expected values of `f10_figure.py` by 3 points / 0.03: the check must then fail).

## What each script checks (times of the final complete run on 2026-10-08: laptop, one thread, other jobs running)

| Script | Checks (brief item; check ids of the record) | Time |
|---|---|---|
| `f1_integrity.py` | sha256 of every reviewed file (505), the results file's own integrity record against the disk (72 saved files, tables), script hash and `passes`, file times, E1b base files and engine files against my E1b review and the zip, `run_all.sh` lines (e, h; h9) | 3 s |
| `f3_store.py` | the whole stored box (178 monthly files, 5,392 days) scanned with numpy only (a; a2) | 9 s |
| `f4_basins.py` | area weights of all 12 basins (shapely, equal-area projection), local-day sums for all issue days by a cumulative-rain route, the issue-time rule and clock proof, made-up-values test, AORC columns; writes `f4_series.npz` (b; b1-b3) | 43 s |
| `f5_skill.py` | correlation, mean ratio, lag test, bias factors, Brier skill (isotonic, logistic, monthly climatology) (b, c; b5-b7) | 11 s |
| `f5b_subdaily.py` | the 3-hour windows against AORC by shift (b; b4) | 4 s |
| `f6_tables.py` | the reviewer's tables, dyn lists and scalers of the 30 networks, wqaF_sub design, the controls, history/perturbation test; writes the table pickles (d, j; d1-d5) | 16 s |
| `f7_networks.py` | the 30 saved networks, 9 derived sets and 3 E1b networks reproduced bit for bit, the shared trajectory, made-up-values test on 5 sets (f, d; d6, f1, f2) | 65 s |
| `f8_headline.py` | headline numbers of 23 sets with the engine and with own pooled onset counts and NSE, on each set's own split; seed averages; observed flow (e; e1, e2) | 24 s |
| `f9_readings.py` | the reading blocks, overlaps, sign tests, `gain_rule`, `e2_success`, `closure`, verdict records (h; h1) | 34 s |
| `f10_figure.py` | `figures/forecast_input.png` read on its pixels against the ladder of the JSON, with a self-test (h; h4) | 15 s |
| `f11_simple.py ridge` | features, causality, settings, ridge refit (g; g1) | 8 s |
| `f12_interpretation.py` | the lead-1 loss: groups of onsets, the same networks with other slots, season, basin, slot ladder (i; i1) | 7 s |
| `f12b_trigger.py` | the loss and the gains at the plain trigger, at the validation-chosen trigger and at equal false-warning budgets (i; i2) | 5 s |
| `f12c_drizzle.py` | the lead-1 onsets by the size of the target day's rain; the nature of the slot values (i; i4) | 3 s |
| `f13_outcome.py` | `outcome`, `outcome_caveat` (seven clauses) and `outcome_full` rebuilt as sentences; the yardstick reading at 1 day; the real-time margin from the posting-latency table; the numbers quoted from the E1b review (h; h2, h3) | 11 s |
| `f14_test_flood.py` | `code/test_flood.py` run unchanged with its output redirected; `flood.py`/`test_flood.py` against the zip (h; h7) | 13 s |
| `f17_robustness.py` | block bootstrap over test days, gains by water year and without each basin (k; k1) | 9 s |
| `f18_roadmap.py` | the 30 statements about E2 in `ROADMAP.md` against the numbers (h; h5) | 3 s |
| `f21_readme_snippets.py` | the statements and snippets of `data/gefs/README.md` that run in seconds (the loader, `local_day_weights`, the basin sizes) (h; h8) | 1 s |
| `f2_raw_days.py` | three raw archive days decoded with eccodes and compared with the stored box value by value (a; a1) - needs `.venv-grib` and the network | 14 s |
| `f11_simple.py perturb` | no data after the training years enters a simple-model fit (g; g1) | 74 s |
| `f11_simple.py forest` | the forest refit (n_jobs=1) against `base_rf_wqaF` (g; g1) | 201 s |
| `f12d_trigger_free.py` | average precision and ROC area of the onset subset with a block bootstrap (i; i3) | 268 s |
| `f15_sandbox_rescore.sh [plain\|with-review]` | `18_forecast_input.py --rescore` in a sandbox: JSON, figure and 189 prediction sets against the project's (h; h6) | 135 s |
| `f16_fetch_check.sh` | `fetch_gefs.py --check` on a sandbox copy of the store (a; a3) - needs `.venv-grib` | 5 s |
| `f19_assemble_review.py` | builds the review record from the outputs above; refuses `--write-into-project` if a reviewed file changed | 2 s |
| `f20_compare_outputs.py OLD_DIR` | which outputs of two complete runs differ, leaf by leaf (the first piecemeal run against the final run) | seconds |

Helpers: `f_common.py` (paths, output folder, the two splits), `f_scoring.py` (own NSE and onset counts on each set's own split), `f_aorc.py` (own AORC local-day sums, `math.fsum`), `f_tables.py` (own rebuild of the daily
tables and of the shuffled controls), `f_simple_lib.py` (the simple-model refit).

## What is suggested but not run

The answer to item (i) of the brief in the record lists tests of the 'over-trust' reading: forward passes of the saved `wqaF_sub` networks with slots = (1-w) x perfect + w x forecast, with small forecast values set to
zero, and with the forecast's bias or its random error alone; a retraining with a mixture or noise in the training slots; a constant-slot control. None was run (the brief said suggest, do not run).

## What the reviewer did and did not do

It downloaded the three raw GEFS days of `f2_raw_days.py` (31.5 MB per run, with their `.idx` files; the script was run a few times while it was written and once in the final run), trained nothing, installed nothing,
did not run the project's numbered scripts in the project, and wrote into the project only this folder and the review record.
