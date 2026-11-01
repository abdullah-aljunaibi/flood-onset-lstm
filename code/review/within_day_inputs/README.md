# The independent reviewer's check scripts for experiment E4b (within-day AORC rain and Q95-relative flow as LSTM inputs)

These scripts belong to the **independent AI review** of experiment E4b (`code/21_within_day_inputs.py` with `code/subdaily_inputs.py`, its results file
`results/within_day_inputs.json`, the figure `figures/within_day_inputs.png`, the 20 saved networks `results/models/lstm_{wqaQ,wqaW,wqaW_older,wqaWF_sub}_s{0..4}.pt`
and the prediction sets `results/preds/lstm_<set>_s<seed>.npz` and `lstm_<set>.npz`). They were written by an AI reviewer agent (Claude) on 2026-10-29, separately from
the agent that wrote and ran E4b, and they are **not part of the project's own code**: `run_all.sh`, the report builders and the numbered scripts neither run nor read them.
The review itself is `results/review_within_day_inputs.json` (what was reproduced, the networks rebuilt, the causality and leakage checks, the findings with must-fix and
should-fix items, the verdict). It states that the team's own check of E4b is still open; this AI review does not replace it.

The scripts recompute E4b's numbers with the reviewer's own code from `results/preds`, `results/models`, `data/aorc`, `data/gefs/basins` and `data/camels_us`. They import
from the project only the engine's data and model classes (`flood.load_data`, `basins`, `thresholds`, `load_preds`, `evaluate` as a cross-check, `Cfg`, `LSTMNet`, `Run.predict`)
and, where the brief asked for it, `subdaily_inputs.add_inputs` / `_table` (the reviewer's own column builder is compared with them). **Nothing is trained and nothing is written
into the project** (every output goes to `$REVIEW_OUT`, default `<temp folder>/wdi_review`; `PYTHONDONTWRITEBYTECODE` is set), except that `r6_assemble.py --write-into-project`
writes the review record.

## Run

From the project root, with the project environment (`<venv>`, torch 2.2.2 CPU), one torch thread, `PYTHONUTF8=1`:

```bash
export REVIEW_OUT=/path/for/the/outputs            # optional
PY=<venv>/Scripts/python.exe
PYTHONUTF8=1 $PY code/review/within_day_inputs/r1_integrity.py     # 2 s
PYTHONUTF8=1 $PY code/review/within_day_inputs/r3_columns.py       # 35 s; writes $REVIEW_OUT/my_columns.npz, which r2 and r4 read
PYTHONUTF8=1 $PY code/review/within_day_inputs/r2_headline.py      # 10 s
PYTHONUTF8=1 $PY code/review/within_day_inputs/r4_networks.py      # 70 s
PYTHONUTF8=1 $PY code/review/within_day_inputs/r5_outcome.py       # 3 s
PYTHONUTF8=1 $PY code/review/within_day_inputs/r6_assemble.py [--write-into-project]
```

`run_checks.sh` runs them in this order.

## What each script checks

| Script | Checks |
|---|---|
| `r1_integrity.py` | sha256 of the 44 saved files and the 12 stored-set files against the results file's own integrity record; the script's hash now against the one recorded at the start of the run and its modification time against the first network written (the success rule was in the hashed file before training); the meta of all 36 prediction sets; inside every .pt: settings = `results/lstm_all.json`, seed, columns, split, stopping point = argmax of the validation NSE, patience stop, parameter count, training record; the log of the run |
| `r3_columns.py` | the six within-day columns, `aorc_to_midnight` and `q_rel` rebuilt from `data/aorc/*.npz` by another route (UTC bounds of every local day from zoneinfo, hours in (start, end], last k hours in (end - k h, end]) and compared with `subdaily_inputs.add_inputs` (6e-11 mm); every hour of day t on local day t, the last hour ending at 24:00 local, the last-3/6/12-hour windows ending at 22-24 / 19-24 / 13-24 on the wall clock on every day including the 34 + 34 clock-change days of 23 and 25 hours; the made-up-hours test on both builders (36 cases); the hour ending exactly 24:00 belongs to day t; Q95 from the training years; the maxcell fill days; the zones |
| `r2_headline.py` | own scoring (no `flood.evaluate`) of the 7 sets and their 28 single seeds: pooled onset and all-day tables at trigger 1.0, median NSE, leads 1-3, test and validation years, per basin; seed ranges, the seed rule, the E4b success record (also with three seeds); Wilcoxon p and basin counts of the gain rule; paired onsets and sign tests; closure; the CSI and 30%-false triggers chosen on the validation years (both grids) and the test scores at them; verdict words; the little-rain slices; seed means = mean of the seed files; the stored sets' own results files; a cross-check against `flood.evaluate` |
| `r4_networks.py` | all 20 networks rebuilt from their .pt files and run on tables built as the script builds them (bit for bit) and on tables built from the reviewer's own columns (within 1e-5 mm/day, same warnings, same NSE); scalers inside every file recomputed from the training years; settings; the control's network against the test's; the reviewer's own made-up-values test on two saved wqaW networks (every input column and the flow after a cut day replaced by random values; 10 cases) |
| `r5_outcome.py` | every clause of `outcome_full` rebuilt from the reviewer's numbers with the script's rounding and found in the text; the fixed wordings, the status, the disclosure; the figure read on its pixels (bar tops and dot counts against the JSON) |
| `r6_assemble.py` | builds the review record from the outputs above; `--write-into-project` writes `results/review_within_day_inputs.json` after checking that no reviewed file changed |

Helper: `r_common.py` (paths, the output folder, the npz reader, the comparison collector).
