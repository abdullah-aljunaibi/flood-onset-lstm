# The independent reviewer's check scripts for experiment E4e (the hourly USGS gauge flow at the issue time as LSTM inputs)

These scripts belong to the **independent AI review** of experiment E4e (`code/22_flow_state.py`, its results file `results/flow_state.json`, the figure
`figures/flow_state.png`, the 30 saved networks `results/models/lstm_{wqaH_s,wqaH22_s,wqaWH_s,wqaWH22_s,wqaWHF_s,wqaWHF22_s}_s{0..4}.pt` and the prediction sets
`results/preds/lstm_<set>_s<seed>.npz` and `lstm_<set>.npz`). They were written by an AI reviewer agent (Claude) on 2026-10-29, separately from the agent that wrote and
ran E4e, and they are **not part of the project's own code**: `run_all.sh`, the report builders and the numbered scripts neither run nor read them. The review itself is
`results/review_flow_state.json` (what was reproduced, the columns and networks rebuilt, the causality and leakage checks, the findings with must-fix and should-fix items,
the verdict). It states that the team's own check of E4e is still open; this AI review does not replace it.

The scripts recompute E4e's numbers with the reviewer's own code from `results/preds`, `results/models`, `data/hourly_flow`, `data/aorc`, `data/gefs/basins` and `data/camels_us`.
They import from the project only the engine's data and model classes (`flood.load_data`, `basins`, `thresholds` as a cross-check, `Cfg`, `LSTMNet`, `Run.predict`), the
within-day rain builder `subdaily_inputs.within_day_table` and `gefs_to_basins.load_columns` (both reviewed earlier), and, where the brief asked for a comparison with the
SCRIPT's columns, the function `flow_state_table` executed from the text of `22_flow_state.py` (the script cannot be imported: it runs on import). **Nothing is trained and
nothing is written into the project** (every output goes to `$REVIEW_OUT`, default `<temp folder>/fs_review`; `PYTHONDONTWRITEBYTECODE` is set), except that
`h6_assemble.py --write-into-project` writes the review record.

## Run

From the project root, with the project environment (`<venv>`, torch 2.2.2 CPU), one torch thread, `PYTHONUTF8=1`:

```bash
export REVIEW_OUT=/path/for/the/outputs            # optional
PY=<venv>/Scripts/python.exe
PYTHONUTF8=1 $PY code/review/flow_state/h1_integrity.py     # 1 s
PYTHONUTF8=1 $PY code/review/flow_state/h3_columns.py       # 15 s; writes $REVIEW_OUT/my_columns.npz, which h2 and h4 read
PYTHONUTF8=1 $PY code/review/flow_state/h2_headline.py      # 3 s
PYTHONUTF8=1 $PY code/review/flow_state/h4_networks.py      # 25 s
PYTHONUTF8=1 $PY code/review/flow_state/h5_outcome.py       # 1 s
PYTHONUTF8=1 $PY code/review/flow_state/h6_assemble.py [--write-into-project]
```

`run_checks.sh` runs them in this order.

## What each script checks

| Script | Checks |
|---|---|
| `h1_integrity.py` | sha256 of the 66 saved files, the 12 stored-set files and the 12 hourly files against the results file's own integrity record; the script's hash now against the one recorded at the start of the run and its modification time against the first network written (the success rule was in the hashed file before training); the meta of all 36 prediction sets; inside every .pt: settings = `results/lstm_all.json`, seed, columns, split, last available hour, carry, stopping point = argmax of the validation NSE, patience stop, parameter count, the training record against the results file; the log of the run |
| `h3_columns.py` | the five hourly-flow columns of both variants rebuilt from `data/hourly_flow/usgs_iv_<gauge>.npz` by another route (zoneinfo instants of 00:00 and H:00 local on the UTC axis, reduceat per local day, own carry of at most 6 h, own mismatch rule) for all 12 basins and compared with the script's `flow_state_table` on every day (bit for bit) with the fallback, carry and mismatch counts of `data_coverage`; the area (the CAMELS forcing-file header, as `camels.py`) and Q95 (training years of the split); the carry rule on the data (no known day is more than 6 h from a value; every 7 h gap falls back); South Toe 2004-09-17; on how many days the mismatch rule alone decides; which clock the USGS daily values follow; the reviewer's made-up-hours test on both builders (2 basins x 2 variants x 5-6 cut days, clock-change days included, plus a one-hour perturbation) |
| `h2_headline.py` | own scoring (no `flood.evaluate`, no `warning_policy`) of the 9 sets and their 39 single seeds: pooled onset and all-day tables at trigger 1.0, median NSE, leads 1-3, test and validation years, per basin; seed ranges, the seed rule, the E4e success record (also with the first three seeds); the 14 pair readings (NSE rule with Wilcoxon p, onset rule, paired onsets with the sign test); the closure; the CSI and 30%-false triggers chosen on the validation years and the test scores, single seeds and verdicts at them; the delivery table; the stored sets against `results/forecast_input.json`; Turkey Creek with and without its fallback days; the validation years; a GO scan; seed means = mean of the seed files |
| `h4_networks.py` | all 30 networks rebuilt from their .pt files and run on tables built as the script builds them and on tables with the reviewer's own hourly-flow columns, compared with the stored forecasts bit for bit; the scalers inside every file recomputed from the training rows; the settings; the reviewer's own made-up-values test on `lstm_wqaH_s_s2` and `lstm_wqaWHF_s_s1` (every input column and the flow after a cut day replaced by random values; 5 cuts each, among them the first known day after a Turkey Creek outage and the spring clock change) |
| `h5_outcome.py` | every clause of `outcome_full` rebuilt from the reviewer's numbers with the script's rounding and found in the text; the fixed wordings, the status, the disclosure against the work order, `what_was_not_done`; the delivery assumptions and the 'gauge's own record' claim as worded; the figure's labels (read on the image) against the JSON; the 70% of wqaWHF22_s against the GO rule |
| `h6_assemble.py` | builds the review record from the outputs above; `--write-into-project` writes `results/review_flow_state.json` after checking that no reviewed file changed |

Helper: `h_common.py` (paths, the output folder, the npz reader, the comparison collector, the script-function extractor).
