# The independent reviewer's check scripts for experiment E3 (63 rivers in training, and rivers never seen)

These scripts belong to the **independent AI review** of experiment E3: `code/17_more_basins.py`, `results/more_basins.json`, `figures/more_basins.png`, the engine change in `code/flood.py` and `code/test_flood.py` (the optional `ids` argument of
`basins()` and `load_data()`, `study_ids()`, `candidate_ids()`, the separate 63-basin cache), the prediction sets and saved networks of `lstm63_wq`, `lstm63_w`, `lstm63eq_wq`, `unseen50_w_static` and `unseen50_w_nostatic`
(`results/preds/`, `results/models/`), and the logs `results/logs/17_more_basins*.log`. They were written by an AI reviewer agent (Claude), separately from the agent that wrote and ran E3, and they are **not part of the project's own code**:
`run_all.sh`, the report builders and the numbered scripts neither run nor read them. The review itself is `results/review_more_basins.json` (checks with numbers, discrepancies, must-fix and should-fix items, verdict). That record states that the
team's own check of E3 is still open; this AI review does not replace it.

The scripts recompute E3's numbers with the reviewer's own code from `results/preds`, `results/models`, the raw CAMELS files (`data/camels_us`) and the caches (`data/derived`). They import the engine (`code/flood.py`: `load_preds`, `thresholds`, `evaluate`, `load_data`, `basins`, `candidate_ids`, `study_ids`, `LSTMNet`, `Run.predict`, and `fit` for the two one-update probes) where the work order allowed, adding `code/` to
`sys.path` as the numbered scripts do; the raw CAMELS files are read with the reviewer's own parsers (`m_common.py`).

## Run

From anywhere, with the project environment, one thread; **nothing is trained and nothing is written into the project** (the output of every check goes to `$REVIEW_OUT`, default `<temp folder>/more_basins_review`; `PYTHONDONTWRITEBYTECODE` is set, so no
`__pycache__` appears here either; `m5` and `m14` work in sandbox copies in the temp folder and remove them):

```bash
code/review/more_basins/run_fast_checks.sh     # 12 checks and the figure self-test: about 3 minutes
code/review/more_basins/run_slow_checks.sh     # test_flood.py with its eight broken variants, the 15 networks, --rescore in a sandbox: about 7 minutes
~/AI_Class/.venv/bin/python code/review/more_basins/m17_assemble_record.py [--write-project]    # rebuilds the record from the outputs above ($REVIEW_OUT/review_more_basins.json, or results/review_more_basins.json)
~/AI_Class/.venv/bin/python code/review/more_basins/<script>.py        # any one check
```

Optional variables: `PY` (the python), `REVIEW_OUT`, `E3_PROJECT` (another copy of the project), `REVIEW_SELFTEST=6` (moves every expected position of `m11_figure.py` by 6 px: the check must then fail), `KEEP=1` and `NO_OUTLINES=1` for `m14_sandbox_rescore.sh`
(keep the sandbox; build it without the outline folder, as in the `full` hand-in archive: the script then stops, which is finding g4). The hand-in files (`Project-Flood_handin.zip`) are unzipped into `$REVIEW_OUT/handin` when a script needs them.
`m17_assemble_record.py` needs the outputs of all the others (it reads the `.json` files they leave in `$REVIEW_OUT`); the others are independent of each other, except that `m12` reads the two files in `evidence_from_builder_scratch/`.

## What each script checks (times measured on 2026-10-07: laptop, one thread, other jobs running)

| Script | Checks | Time |
|---|---|---|
| `m_common.py` | helpers: the reviewer's own parsers of the raw CAMELS forcing and flow files, own NSE, own pooled onset counting, own haversine, own join of the attribute tables | - |
| `m1_engine_diff.py` | a1, a6: `flood.py` and `test_flood.py` against the hand-in copy statement by statement (ast) and by diff; every call of `basins(` / `load_data(` in `code/` and its arguments; star imports; name clashes | 2 s |
| `m2_defaults.py` | a2: the hand-in engine and the present engine each rebuild the 12-basin table from raw into an empty cache folder (sandbox); caches byte for byte, tables bitwise, `basins()`, thresholds, `evaluate` output, the headline `lstm_wq` score | 8 s |
| `m3_candidates_raw.py` | a3, a4: `candidate_ids()` against the audit rule; the audit's candidate set and all its numbers recomputed from the raw files (78 candidates); completeness of the 63; the 63-basin cache against tables rebuilt from raw, bit for bit; the 12 inside it; properties; river sets of the results file | 16 s |
| `m3b_area_errors.py` | a7: which of the 63 are on CAMELS's list of basins with an area error above 10%, and how the three areas compare | 4 s |
| `m5_testflood_mutants.py` | a5: `test_flood.py` run unchanged in sandboxes (baseline; hand-in test with the present engine; new test with the hand-in engine; eight deliberately broken `load_data()` variants): every broken variant must fail | 3 min |
| `m6_headline.py` | b1-b4: headline numbers of 5 new sets and 7 controls x 3 leads by the engine and by the reviewer's own counts against the results file; the 20 seed and average files; the 12 basins only; obs equal the project's flow; seed averages are bitwise means; metadata; the own-level onset warnings; the numbers of the review request | 32 s |
| `m7_networks.py` | c1-c3, f1: the 15 saved networks: training ids, settings, scalers recomputed from the training years, early-stopping replay, saved weights re-scored on the validation years of the training rivers, stored forecasts reproduced bit for bit (engine, 63-basin table, own input matrix), negative control | 2 min |
| `m8_unseen.py` | d1-d5: outline overlap of all 1953 pairs of the 63; the 12 never read in training (fit with the 12 absent; fit with a dictionary that raises on any access); proximity numbers; the budget of the 05 trainings and the 05 stopping rule replayed on the 50-river curves; the three seeds one by one against 05; seed averaging; the per-river gain against distance (Spearman, Kendall) | 32 s |
| `m9_gain_rule.py` | e1-e3: the gain rule for 10 pairs x 3 leads (Wilcoxon by scipy and by enumeration), the words, the own-level onset comparisons, the verdicts (144 readings; the stored tables of `lstm_wq` / `lstm_w`), paired onset counts (exact McNemar) | 38 s |
| `m10_extra_rivers.py` | d6 (information): the saved 50-river networks run on the 15 audit candidates with gaps in their flow record (never in any training); skill against the distance to the nearest training gauge over 12, 15 and 27 rivers | 14 s |
| `m11_figure.py` | g1: `figures/more_basins.png` read on its pixels: panels calibrated on the grid lines, 22 bars, 16 seed ranges against the results file (`REVIEW_SELFTEST=6` must fail) | 1 s |
| `m12_records_logs.py` | c5, g3, g7: log lines against the records and the network files; timeline; the interim pass and the script versions (two files of the builder's scratch folder, copied); ROADMAP.md numbers | 2 s |
| `m13_handin.py` | g4: dry run of `make_handin_zip.sh` (`light` and `full`): sizes, what E3 adds, whether `--rescore` could run from the unpacked `full` archive | 2 s |
| `m14_sandbox_rescore.sh`, `m14_compare.py` | g5: `17_more_basins.py --rescore` in a sandbox copy (clones of sets, networks and caches, links to the CAMELS inputs): the results file leaf by leaf and the figure byte by byte against the project's; the project's files hashed before and after | 1.7 min |
| `m15_integrity.py` | g6: sha256, size and time of the 63 files of the review; the hashes the results file recorded for its own 35 files against the files now; quick-mode flags | 1 s |
| `m16_probe_properties.py` | d7: the builder's probe 'the 12 rivers' properties replaced by the median of the 50', recomputed from the saved networks | 10 s |
| `m17_assemble_record.py` | the record `review_more_basins.json`: every number read from the outputs above, the text is the reviewer's | 2 s |

`evidence_from_builder_scratch/` holds copies of two files the builder kept in its session scratch folder (the interim results of 20:13, which scored the four non-equal sets before the equal-training arm was trained, and the script as it ran at 19:35) and the two hash lists it
made around the engine change; the scratch folder is not durable. Every conclusion drawn from them is also checked against the log and the final files (check c5).

## Where each check of the record is

Ids are those of `checks` in `results/review_more_basins.json`. (a) the engine change: a1, a2, a3, a4, a5, a6 (+ a7, data quality). (b) headline numbers: b1-b4. (c) protocol of the 63-river training: c1-c6. (d) rivers never seen: d1-d7. (e) gain rule, verdicts, outcome sentence: e1-e4.
(f) saved networks: f1. (g) figure, `run_all.sh`, logs, hand-in script, rescore, integrity: g1-g7.

## Notes

- A check that cannot fail checks nothing: `m5` breaks `load_data()` eight ways and requires every run to fail; `m11` moves every expected position by 6 px (`REVIEW_SELFTEST=6`) and requires the check to fail; `m6`, `m9`, `m3` compare the engine route with the reviewer's own counts, and both with the results file, to 0.0.
- `m5`: the first version of variant M2 changed only the cache read (a set with ids rebuilt on every call); that is functionally correct, an equivalent variant, and the test rightly does not catch it. M2 now also shares the file name between sets, so the second set overwrites the first set's cache.
- `m8`/`m10`/`m16` are information, not verdicts: the rank correlations have 12 to 27 points; they were asked for in order to judge the proximity caveat and are described as such in the record.
- `m14` writes about 0.2 GB into the temp folder (clones of the prediction sets, networks and caches; links to the CAMELS inputs) and removes it at the end (`KEEP=1` keeps it). The sandbox run of the script is the only way this review ran any part of `17_more_basins.py`.
- Not scripts: the reading of `code/flood.py`, `code/17_more_basins.py`, `code/04_lstm_all.py`, `code/05_lobo.py` and `code/01_data_audit.py`, the reading of the logs and of the figure's labels and legend, and the reading of `ROADMAP.md`, `README.md` and `NEXT_STEPS_FOR_AGENTS.md`; their findings are in the record (checks c4, c5, g1, g7 and the discrepancy list).
