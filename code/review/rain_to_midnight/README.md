# The independent reviewer's check scripts for experiment E1 (rain up to local midnight)

These scripts belong to the **independent AI review** of experiment E1 (`code/15_rain_to_midnight.py`, `results/rain_to_midnight.json`). They were written by an AI reviewer agent (Claude), separately from the
agent that wrote and ran E1, and they are **not part of the project's own code**: `run_all.sh`, the report builders and the numbered scripts neither run nor read them. The review itself is
`results/review_rain_to_midnight.json` (rounds 1, 2 and 3: checks, numbers, discrepancies, must-fix and should-fix items, verdicts). That record states that the team's own check of E1 is still open; this AI review does not replace it.

The scripts recompute E1's numbers with the reviewer's own code, from `results/preds`, `results/models`, `data/hourly` and `data/camels_us`. They import only the engine (`code/flood.py`: `load_data`, `basins`,
`thresholds`, `evaluate`, `load_preds`, `LSTMNet`, `Run.predict`, as the review allowed), `code/features.py` and `code/camels.py`, adding `code/` to `sys.path` as the numbered scripts do.

## Run

From the project root, with the project environment, one thread; nothing is trained and **nothing is written into the project** (the output of every check goes to `$REVIEW_OUT`, default
`<temp folder>/rain_to_midnight_review`; `PYTHONDONTWRITEBYTECODE` is set, so no `__pycache__` appears here either):

```bash
~/AI_Class/.venv/bin/python code/review/rain_to_midnight/<script>.py       # one check (r2d_simple.py takes ridge | forest | perturb)
code/review/rain_to_midnight/run_fast_checks.sh                             # the 20 fast checks in order: about 4 minutes (170 s for the first 17 when the laptop is idle, 252 s with two other jobs running)
code/review/rain_to_midnight/run_round3_extras.sh                           # round 3: the comparison of two runs, the first-run regression and the sandbox --rescore (needs copies of earlier states, see its header): about 3 minutes
code/review/rain_to_midnight/run_slow_checks.sh                             # the slower ones: about 15 minutes (run_fast_checks.sh must have run once with the same REVIEW_OUT)
```

Order matters in one place: `r2c_clocks.py` writes `my_series_r2.npz` (the reviewer's daily series), which `r2c2_networks.py` and the `r2d_*` scripts read; `r2i` before `r2j` is optional.
Optional variables: `PY` (the python), `REVIEW_OUT`, `E1_MANIFEST_28` (a `sha256  ./path` list of the 28 original model and prediction files made before the extension, used by `r2a_integrity.py`), `E1_R2_HASHES` (the list `sha256  results/...` of the 119 prediction and network files hashed in round 2, used by `r3a_integrity.py`),
`E1_PREVIOUS_RUN_OUT` (the output folder of an earlier run, for `r3b_compare_outputs.py`), `E1_FIRST_RUN_JSON` (a copy of the first-run `results/rain_to_midnight.json`, accepted only if its sha256 equals the one round 1 recorded, for `r3f_regression.py`), `R2G_SELFTEST=6` (moves the expected seed-bar ends of `r2g_figure.py` by 6 px: the check must then fail).

## What each script checks (times measured in this folder on 2026-10-07: laptop, one thread, other jobs running)

| Script | Checks | Time |
|---|---|---|
| `r2a_integrity.py` | the hashes recorded in round 1, the 28 original files, the script's own 51-file hash record, the engine files against the hand-in zip, which files changed; regression of 72 numbers of round 1; seed averages | 6 s |
| `r2b_headline.py` | headline numbers of the new sets and the simple models (engine) and by the reviewer's own pooled onset counts | 13 s |
| `r2c_clocks.py` | the clocks of `rain_to_09` and `rain_utc_day` (every hour inside its window and before the issue time, perturbation proof), the scalers of the new networks; writes `my_series_r2.npz` | 8 s |
| `r2c2_networks.py` | the 9 new and 12 old saved networks reproduced bit for bit from the reviewer's own tables; pipeline leakage test; the NLDAS column and the end of the NLDAS day | 54 s |
| `r2d_simple.py ridge` | how the new series enters the simple models (features), their causality, the ridge refit against `base_linreg_wqh` | 8 s |
| `r2e_readings.py` | the new reading fields, `gain_rule`, closure, `e1_success`, `outcome_caveat` | 9 s |
| `r2e2_yardstick.py` | the yardstick fields (LSTM against the simple models; what the new input adds) | 10 s |
| `r2f_byperiod.py` | the `by_period` table (before and from 2009-07-17) | 4 s |
| `r2g_figure.py` | `figures/rain_to_midnight.png` read on its pixels against the ladder of the JSON | 2 s |
| `r2h_records.py` | training records and per-seed scores of the 9 new networks, log lines, `passes`; paired onsets of the LSTM against the two simple models; the Daymet window per basin; `value_of_hours_09_to_24` without Myakka | 4 s |
| `r2i_daymet_window.py`, `r2j_late_window_basins.py` | how well the 09:00 window matches the Daymet rain table; the same reading without the basins whose window ends latest | 7 s, 2 s |
| `run_test_flood_redirected.py` | `code/test_flood.py` run unchanged with its output redirected; the file it writes against `results/fair_test.json` | 14 s |
| `r1_clock_midnight.py` | round-1 checks 3a-3e: the series `rain_to_midnight` rebuilt, summer time by two routes, last hour of a July and a January day, perturbation proof, scalers | 7 s |
| `r1_stamp_convention.py` | round-1 check 3f: hourly ERA5 against hourly MERRA-2 (NASA POWER) at lags -4..+4 h, the hour-stamp convention | 5 s |
| `r1_hour_sensitivity.py` | round-1 check 3g: the saved networks fed the series of the opposite stamp convention | 12 s |
| `r1_gain_rule.py` | round-1 checks 6a-6c: seed ranges, E1 success-rule flags, Wilcoxon p, gap shares, paired onsets | 5 s |
| `r2d_simple.py perturb` | no data after 2000-09-30 enters a fit (ridge for 12 basins, forest for 2) | 85 s |
| `r2e3_verdicts.py` | the verdict records of the new sets (warning level chosen on the validation years, own counting) | 30 s |
| `r2d_simple.py forest` | the forest refits (one job) against `base_rf_wqh` and `base_rf_wq`: max difference 7.1e-15 | 203 s + 146 s |
| `r2d2_ridge_exact.py`, `r2d3_forest_exact.py` | the same refits with the rain re-summed the way the script sums it (ridge: exact, 0.0; forest: the 7e-15 stays, it is the order of the parallel sums) | 9 s, about 3 min |
| `r2k_sandbox_modes.sh` | `15_rain_to_midnight.py --rescore` and `--extend` in sandbox copies (about 0.3 GB in the temp folder, removed afterwards): same JSON and figure, hashes of all prediction and network files unchanged, nothing trained | about 4 min |
| `r3a_integrity.py` | round 3 (a), (d): the 51 saved files against the results file's own record, my round-1 and round-2 records and my list of 119 hashes; `integrity.script_sha256` and the `passes` entries against the script on disk and the logs; writes the 51 hashes as they are now | 4 s |
| `r3c_clauses.py` | round 3 (c): the two new clauses of `outcome_caveat` rebuilt from my own counts; the onset-level yardstick fields (5 pairs x 3 leads, 435 values and 15 sentences); the `by_period` standard errors; the verdict records of the three added baselines; `what_was_not_done[3]` and the sentences appended to two readings; IMERG wording | 31 s |
| `r3d_texts.py` | round 3: strict JSON and the null count, the yardstick note, `what_was_not_done[0]` against the review record, what PLAN.md and README.md say about E1 | 2 s |
| `r3b_compare_outputs.py OLD_DIR` | round 3 (b): which outputs of two runs of the fast checks differ, leaf by leaf (added fields, changed numbers, changed text) | seconds |
| `r3f_regression.py` | round 3 (f): every leaf of the first-run results file against the current file (own comparison; lists aligned by their `set` key) | seconds |
| `r3_sandbox_rescore.sh` | round 3 (b), (e): `15_rain_to_midnight.py --rescore` in a sandbox copy (about 0.3 GB of links and copies, removed afterwards): the results file and the figure against the project's; the figure must be byte-identical | 150 s |

Helpers: `r2_common.py` (paths, output folder), `r2_clock.py` (own rebuild of any window of the hourly ERA5), `r2_scoring.py` (own NSE and onset counts), `r2_simple_lib.py` (the simple-model refit).

## Round-1 checks: where each one is

The round-1 scratch copy of the scripts was lost (the review session's temporary folder was recreated), so round 1 is covered by rewrites and by the round-2 scripts. Ids are those of `checks` in the record.

| Round-1 check | In this folder |
|---|---|
| 3a-3e, 3f, 3g, 6a-6c | rewritten: `r1_clock_midnight.py`, `r1_stamp_convention.py`, `r1_hour_sensitivity.py`, `r1_gain_rule.py` (6b: Wilcoxon p at leads 1-3 and the closure at lead 1); each prints whether it agrees with the numbers recorded in round 1 (all do) |
| 1a, 1c, 2 | in part: `r2a_integrity.py` recomputes the 72 per-seed onset shares and median NSEs of the first twelve trainings and the seed averages and compares them with round 1; `r2b_headline.py` does the engine-against-own-counts comparison for the new sets and the simple models |
| 5 | `r2c2_networks.py` (the 12 first and the 9 new networks reproduced bit for bit from the reviewer's own tables) |
| 7 | `r2g_figure.py` reads the present figure (round 1's figure no longer exists in that form) |
| 8a-8f | in part: `r2a_integrity.py` and `run_test_flood_redirected.py` (hashes, files changed, `test_flood.py`), `r2k_sandbox_modes.sh` (`--rescore` and `--extend`), `r2h_records.py` (records of the nine new networks) |
| 4a-4d | in part: `r2c2_networks.py` (design, early-stopping and split flags and the pipeline test of the 9 new networks), `r1_clock_midnight.py` (scalers of the first four), `r2_scoring.py` (thresholds from the training years) |
| not rewritten | 1b (round 1's outcome sentence, replaced by `outcome_caveat`, which `r2e_readings.py` checks); 6d (seed-level reading at leads 2 and 3); the closure shares at leads 2 and 3 and the cross-checks of the ROADMAP correlations and of hits by basin and water year; for the first twelve networks the design flags and pipeline test (4b), the shift of the "older" series (4a), all stored scalers (4c), the early-stopping reading (4d) and the records against the `.pt` files (8f) |

## Round 3 (small and final for E1): which script answers which question

(a) the 51 saved files byte-identical: `r3a_integrity.py`; (b) the fast checks against the current files and which outputs differ: `run_fast_checks.sh` + `r3b_compare_outputs.py`, and the sandbox rescore `r3_sandbox_rescore.sh`;
(c) the new clauses, the onset-level yardstick fields, the standard errors: `r3c_clauses.py`; (d) the `passes` entries and `integrity.script_sha256`: `r3a_integrity.py`; (e) the figure against the JSON: `r2g_figure.py` (markers and bars on the pixels) and the byte comparison of
`r3_sandbox_rescore.sh`; the caption and labels were read on the picture; (f) the first-run regression: `r3f_regression.py`. Three older scripts were patched in round 3: `r2e_readings.py` (the five clauses it rebuilds must be the beginning of `outcome_caveat`),
`r2g_figure.py` (the connecting line of a family, which touched one marker's column and made one bar look 4 px too long in rounds 2 and 3, is masked; `R2G_SELFTEST=6` shows that the check can fail) and `r2h_records.py` (the rescore pass is left out of the sum of the training runtimes).

## Read this before trusting a printed line

* **Verified from this folder, 2026-10-07 18:28-18:40.** All 17 fast checks ran in 170 s; every output equals, to 1e-12, the output the same script gave in the review session's scratch folder (only
  `r2a_integrity.py`'s lists of files changed and of engine files differ: other agents edited the project in between). The four `r1_*` scripts report agreement with the round-1 record. Of the slow checks,
  `r2d_simple.py perturb` and `forest`, `r2e3_verdicts.py` and `r2d2_ridge_exact.py` also ran here (outputs equal to the scratch ones, apart from the seconds); `r2d3_forest_exact.py` was run only up to its forest
  fit (every name resolves; it is the same fit as the first one of `r2d_simple.py forest`), and `r2k_sandbox_modes.sh` was only syntax-checked (`bash -n`) here. It last ran in round 2 from the scratch folder (same JSON and
  figure in both modes, hashes unchanged). It was not run again because the script under test was being edited and re-run by other agents at the time and the laptop had no free core.
  The runs above compared with `results/rain_to_midnight.json` and `figures/rain_to_midnight.png` as of 17:37; both were rewritten at 18:40 by a later `--rescore` run in the project (its log is
  `results/logs/15_rain_to_midnight_rescore.log`) and the script was edited at 18:31, so a fresh run now can differ by design (next point).
* **Round 3, 2026-10-07 18:51-19:15, from this folder against the files of 18:40:** the 17 scripts of rounds 1 and 2 in 252 s (two other jobs loading the laptop), the three new fast ones in 37 s; 13 of the 16 outputs identical to the earlier run, the 3 that differ by design (see the record, round_3 check b1);
  the sandbox rescore ran in 150 s and gave the same results file (three timing leaves differ) and the same figure bytes. `r3c_clauses.py` and `r3a_integrity.py` compare with the files as they are: after the next edit of the script, the results file or the figure they will say what changed.
* The scripts compare with the files **as they are when you run them**. The review record gives what they printed on 2026-10-07, for the versions named by sha256 in its `files_reviewed`. If
  `15_rain_to_midnight.py`, `results/rain_to_midnight.json` or the engine have changed since, some comparisons differ by design (the hashes and the lists of changed files in `r2a_integrity.py`, the
  `outcome_caveat` sentence in `r2e_readings.py`, the scores of any set that was trained again): read the lines, do not expect every one to say True.
* `r2k_sandbox_modes.sh` runs whatever version of `code/15_rain_to_midnight.py` is in the project against `results/rain_to_midnight.json` as it is: after the script has been edited and before the results file
  has been written again by it, `--rescore` or `--extend` can stop with an assertion (the script checks its own inputs); that says the pair is out of step, not that the review script is wrong.
* The scripts that assembled the review record are not included: they write into `results/review_rain_to_midnight.json`.
