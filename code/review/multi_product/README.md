# The independent reviewer's check scripts for experiment E7 (several rain products side by side as LSTM inputs)

These scripts belong to the **independent AI review** of experiment E7 (`code/28_multi_product.py`, its results file `results/multi_product.json`, the figure
`figures/multi_product.png`, the 15 saved networks `results/models/lstm_{wqME,wqME_shuf,wqME_older}_s{0..4}.pt`, the prediction sets
`results/preds/lstm_<set>_s<seed>.npz`, `lstm_<set>.npz`, `base_linreg_wqME.npz` and `base_rf_wqME.npz`). They were written by an AI reviewer agent (Claude) on
2026-10-30, separately from the agent that wrote and ran E7, and they are **not part of the project's own code**: `run_all.sh`, the report builders and the numbered
scripts neither run nor read them. The review itself is `results/review_multi_product.json` (what was reproduced, the columns and networks rebuilt, the causality and
leak probes, the rule applied block by block, the findings with must-fix and should-fix items, the verdict). It states that the team's own check of E7 is still open;
this AI review does not replace it.

The scripts recompute E7's numbers with the reviewer's own code from `results/preds`, `results/models`, `data/aorc`, `data/hourly` and the raw CAMELS NLDAS files.
They import from the project only the engine's data and model classes (`flood.load_data`, `study_ids`, `basins`, `evaluate` as a second route, `Cfg`, `LSTMNet`,
`Run.predict`), `camels.load_basin` (as a cross-check of the reviewer's own NLDAS parser) and `features.feature_table` (the shared recipe of the simple models, for the
ridge refit). Nothing is imported from `code/28_multi_product.py` (it runs on import). **Nothing is trained and nothing is written into the project** (every output goes
to `$REVIEW_OUT`, default `<temp folder>/mp_review`; `PYTHONDONTWRITEBYTECODE` is set), except that `m6_assemble.py --write-into-project` writes the review record.

## Run

From the project root, with the project environment (`<venv>`, torch 2.2.2 CPU), one torch thread, `PYTHONUTF8=1`:

```bash
export REVIEW_OUT=/path/for/the/outputs            # optional
bash code/review/multi_product/run_checks.sh        # m1, m3, m2, m4, m7, m5, m6 in this order (about 2 minutes)
bash code/review/multi_product/run_checks.sh --write-into-project
```

## What each script checks

| Script | Checks |
|---|---|
| `m1_integrity.py` | sha256 of the 35 saved files, the 28 stored-set files and the 4 identity-check networks against the results file's own integrity record; the script's hash now against the training pass, its and ROADMAP.md's modification times against the first network written, the ROADMAP E7 section's hash; the meta of the 22 new prediction sets; inside every .pt: settings = `results/lstm_all.json`, seed, columns, split, stopping point = first argmax of the validation NSE, the patience stop, the parameter count, the input-column hashes, the training record; the log of the run |
| `m3_columns.py` | the seven input columns rebuilt by another route (zoneinfo instants of local midnight, hour-ending stamps in (start, end], cumulative sums; the same day assignment summed by pandas groupby for a bit-for-bit sha256 comparison with `inputs.sha256_of_input_columns`); NLDAS from the raw CAMELS file; the shuffle rebuilt from its documented algorithm (source-index sha256, counts, group constraints); every hour of day t on local day t; made-up hours on the reviewer's builder; the NLDAS day measured against the AORC hours; the AORC/ERA5 hourly lag; writes `$REVIEW_OUT/my_columns.npz` |
| `m2_headline.py` | own numpy scoring of the 3 new sets, the 5 stored LSTM sets (and all their single seeds) and the 4 simple models: pooled onset and all-day tables at trigger 1.0, median NSE, leads 1-3, test and validation years, per basin; seed ranges; seed means = mean of the seed files; flood.evaluate as a second route; the ROADMAP E7 rule block by block with an exact Wilcoxon p by enumeration; the seven pair readings and paired onsets; the CSI and 30%-false operating points (the documented rule of `warning_policy.choose_trigger`), their single seeds and verdicts; the verdict readings at 0.30-1.50; all 168 resampling statistics by 13_uncertainty.py's recipe (own code, plus a direct route on the first resamples); the stored sets against their own results files; descriptive extras for the findings |
| `m4_networks.py` | the 15 new networks and the 10 stored networks of lstm_wqa and lstm_wqnh rebuilt from their .pt files and run on tables whose columns are the reviewer's, compared with the stored forecasts bit for bit; the scalers recomputed from the training rows; the reviewer's own made-up-values test on 3 networks x 4 cut days |
| `m7_ridge.py` | the ridge yardstick `base_linreg_wqME` refitted from the reviewer's columns and compared with the stored forecasts |
| `m5_outcome.py` | every clause of `outcome_full` rebuilt from the reviewer's numbers with the script's rounding and found in the text; the fixed wordings, the status, the disclosure, `what_was_not_done`; missing caveats; the figure's bar labels (read on the image) against the reviewer's numbers |
| `m6_assemble.py` | builds the review record; `--write-into-project` writes `results/review_multi_product.json` after checking that no reviewed file changed |

Helper: `m_common.py` (paths, the output folder, the npz reader, the comparison collector, the conventions restated).
