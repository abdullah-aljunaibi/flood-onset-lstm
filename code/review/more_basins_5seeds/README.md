# Round 2 of the independent review of E3 (script 17), five seeds

AI review (Claude), 30 October 2026, after seeds 3 and 4 of every set of `code/17_more_basins.py` were trained on the Windows PC
(one run stopped by the low-memory guard at 12:47 after lstm63_wq seed 3, then the rerun 13:02-13:24).
It recomputes the five-seed numbers of `results/more_basins.json` from `results/preds` with its own numpy scoring and appends
the block `round_2` to `results/review_more_basins.json` (everything already in that record is kept byte for byte).

Run from the project root (about 1 minute; one torch thread; nothing trained; outputs to `REVIEW_OUT`, default `<temp>/e3_5seeds_review`):

    PY=<venv python> bash code/review/more_basins_5seeds/run_checks.sh                         # checks only
    PY=<venv python> bash code/review/more_basins_5seeds/run_checks.sh --write-into-project    # also appends round_2 (refuses if it exists)

- `r_common.py`: paths, own reader of the prediction files, own NSE / Q95 / onset / warning tables / level sweep, exact Wilcoxon and McNemar, leaf-by-leaf compare.
- `r1_integrity.py`: the 55 files of the five sets and their sha256; seeds 0-2 unchanged; meta and network records; seed averages bit for bit; passes and integrity bookkeeping.
- `r2_score.py`: scores the 46 prediction files one at a time.
- `r3_compare.py`: rebuilds summary_test, validation, seed_ranges, per_river, gain_rule, gap, stopping, verdicts and unseen and compares about 10,000 values at 1e-9; the outcome sentence word for word.
- `r4_three_vs_five.py`: readings, flags and levels changed against the three-seed backup; rank correlations; seed words in the texts; the SEED_NOTE.
- `r5_networks.py`: reloads three seed-3/4 networks with one thread (engine forward pass) and compares their forecasts with the stored files.
- `r6_assemble.py`: the record block.
