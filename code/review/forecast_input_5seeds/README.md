# Round 2 of the independent review of E2 (script 18), five seeds

AI review (Claude), 30 October 2026, after seeds 3 and 4 of every set of `code/18_forecast_input.py` were trained on the Windows PC.
It recomputes the five-seed numbers of `results/forecast_input.json` from `results/preds` with its own numpy scoring and appends
the block `round_2` to `results/review_forecast_input.json` (everything already in that record is kept byte for byte).

Run from the project root (one torch thread, nothing trained, outputs to `REVIEW_OUT`, default `<temp>/e2_5seeds_review`):

    bash code/review/forecast_input_5seeds/run_checks.sh                         # checks only
    bash code/review/forecast_input_5seeds/run_checks.sh --write-into-project    # also appends round_2 (refuses if it exists)

- `v1_score.py`: scores the 92 prediction files one at a time; seed averages against their seed files.
- `v2_compare.py`: every comparable number of the results file against v1 (about 10,800 values at 1e-9).
- `v3_networks.py`: reloads five networks (two new seed-3/4 ones, three derived sets) on tables from the round-1 builder (`../forecast_input/f_tables.py`).
- `v4_readings_text.py`: readings changed from three to five seeds (against the backup `../Project-Flood_backup_2026-10-30/results/forecast_input.json`), outcome clauses, seed wording, passes, SEED_NOTE.
- `v5_assemble.py`: the record block.
