# The independent reviewer's check scripts for experiment E4a (the warning operating point on stored forecasts)

These scripts belong to the **independent AI review** of experiment E4a (`code/20_warning_policy.py`, the module `code/warning_policy.py` it imports,
`results/warning_policy.json`, `figures/warning_policy.png`). They were written by an AI reviewer agent (Claude), separately from the agent session that
wrote and ran E4a, and they are **not part of the project's own code**: `run_all.sh`, the report builders and the numbered scripts neither run nor read them.
The review itself is `results/review_warning_policy.json` (what was reproduced, the leakage check, the findings with their severity, the verdict). That record
states that the team's own check of E4a is still open; this AI review does not replace it.

The scripts recompute E4a's numbers with the reviewer's own code, straight from the arrays of `results/preds/lstm_*.npz` (numpy and pandas: the npz is read
directly, Q95 is taken from its own `obs` over the training years of the set's split, the rows are pooled with the conventions of `flood.flood_scores`
restated in `w_common.py`, the triggers are chosen by the reviewer's own `choose()`). The project's `flood.py` is imported only where a check compares against
the project's reference (`flood.thresholds`, `flood.evaluate`), and `warning_policy.py` only in the made-up-values leakage test, where the code under review is
run on perturbed inputs. **Nothing is trained and nothing is written into the project**: every output goes to `$REVIEW_OUT` (default `<temp folder>/e4a_review`),
`PYTHONDONTWRITEBYTECODE` is set, and `w8_sandbox_rerun.py` runs `20_warning_policy.py` in a **sandbox copy** that is deleted at the end. Only
`w9_assemble_review.py --write-into-project` writes into the project (the review record, and only when no reviewed file changed since `w0` hashed it).

## Run

From the project root, with the project's environment (`<venv>`), `PYTHONUTF8=1`:

```bash
export REVIEW_OUT=/path/to/an/output/folder            # optional
export PYTHONDONTWRITEBYTECODE=1                       # w_common.py sets it too, but its own bytecode is written before that line runs
PY=<venv>/Scripts/python.exe
for s in w0_integrity w1_headline w2_seeds w3_any_lead w4_bootstrap w5_stacker w6_text_go; do PYTHONUTF8=1 $PY code/review/warning_policy/$s.py; done
PYTHONUTF8=1 $PY code/review/warning_policy/w8_sandbox_rerun.py              # about 2-3 minutes: copies 36 prediction files and runs the script in a sandbox
PYTHONUTF8=1 $PY code/review/warning_policy/w9_assemble_review.py            # builds the record into $REVIEW_OUT; --write-into-project writes results/review_warning_policy.json
```

`w1` to `w6` are independent of each other (each reads the record and the npz files); `w9` reads the outputs of all of them.

## What each script checks (times of the final run on the evening of 2026-10-28: laptop, one thread, other jobs running)

| Script | Checks | Time |
|---|---|---|
| `w0_integrity.py` | sha256 of the reviewed files against the record's `script_sha256` and `inputs_sha256` (36 prediction files), file times, the design constants (grid, cap, halves, splits, bootstrap settings) against the brief, the GO criteria, the shape and NaN pattern of every stored set | 1 s |
| `w1_headline.py` | for 9 sets x 3 leads x 2 rules: my trigger choice on the validation years, the validation and test tables (onsets and all days), the verdicts, the trigger-free average precision; my Q95 against `flood.thresholds`; my test tables against `flood.evaluate`; the CSI triggers against those stored by script 18 (`results/forecast_input.json`); ties in the choice | 118 s |
| `w2_seeds.py` | every seed file's own trigger, its test scores at its own and at the set's trigger, the seed ranges; the seed-mean set against the mean of its seed files; which single networks meet GO | 27 s |
| `w3_any_lead.py` | the any-lead window row (own implementation) at the triggers of both rules for every set; the 45 of 298 | 6 s |
| `w4_bootstrap.py` | my own block bootstrap (500 draws, seed 20261029) for all 54 set/lead/rule points against the record's 2000-draw intervals (overlap, endpoint gaps); the record's stated procedure repeated with its seed (must give the stored intervals exactly); a moving-block variant for `lstm_wqa` lead 1 | 5 s |
| `w5_stacker.py` | my refit of the logistic stacker (own features, own standardisation, scikit-learn's regression) against the record's cut, coefficients and tables for 6 sets x 3 leads; the periods (halves inside validation, validation before test); the made-up-values test: test-years forecasts and flows replaced by random numbers, the project's `choose_trigger` and `stack` run again, no choice may change | 44 s |
| `w6_text_go.py` | the fixed wordings; every number of `outcome_full` traced to a table of the same file; the GO scan over every stored table (onsets, all days, any lead, single seeds, stacker); `summary` against `sets`; the preliminary diagnostic against the experiment; the static texts of the script | 5 s |
| `w8_sandbox_rerun.py` | `20_warning_policy.py` run in a sandbox copy: its JSON against the project's leaf by leaf, the figure's bytes | 178 s |
| `w9_assemble_review.py` | builds the review record from the outputs above; refuses `--write-into-project` when a reviewed file changed | 1 s |

Helper: `w_common.py` (paths, the conventions restated, the npz reader, the own pooled rows, tables, trigger choice, verdict, average precision).

## What the reviewer did and did not do

It trained nothing, installed nothing, downloaded nothing, did not run the project's numbered scripts in the project (only in a sandbox copy), and wrote
into the project only this folder and the review record. The figure was read by eye against the `summary` of the record, not on its pixels.
