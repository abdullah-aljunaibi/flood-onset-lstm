# E7: several rain products at once (design, 30 October 2026; not run)

Question: does an LSTM fed several rain products **side by side** warn more onsets than one fed the best single
product (AORC to midnight)?

Status: design only. Not part of the 2 November hand-in. Every number below is a stored result quoted to set the
prior; none is an E7 result. **MRMS is ruled out (the team lead, 30 October 2026): E7 = E7a only.** Sections 1 and 3
are kept as the record of why. Renamed from E5 on 30 October 2026: `ROADMAP.md` uses E5 (a loss aimed at peaks)
and E6 (network changes). Script number 23 is reserved for E4c, so the script is `28_multi_product.py`. The success
rule is fixed in `ROADMAP.md`, section "E7".

## 0. What is already known (sets the prior)

| Stored set (main split, 5 seeds, 1 day ahead) | Rain products the network sees | Onsets warned | False | Median NSE |
|---|---|---|---|---|
| lstm_wq | Daymet | 18.7% | 59.0% | 0.590 |
| lstm_wqn | Daymet + NLDAS | 30.0% | 62.7% | 0.674 |
| lstm_wqh | Daymet + ERA5 to midnight | 29.5% | 56.0% | 0.663 |
| lstm_wqnh | Daymet + NLDAS + ERA5 to midnight | 32.4% | 58.9% | 0.694 |
| lstm_wqa | Daymet + AORC to midnight | 51.7% | 50.9% | 0.764 |

Sources: `results/rain_to_midnight.json`, `results/aorc_rain_to_midnight.json` (`summary_test/sets/*/l1`).

- The idea has been partly tested already. `lstm_wqnh` combined two products and gained about 2 to 3 points over
  either one alone. Expect a small gain, not a jump.
- Every set already keeps Daymet beside the new product, so `lstm_wqa` is a two-product model.
- Most of the AORC gain comes from the clock: rain up to midnight. The same AORC over the Daymet window gives 18.0%
  (`wqa_to09`). A second product over the same hours adds little. The ERA5/NLDAS precedent points the same way.

## 1. Why MRMS cannot join the main split

- CAMELS-US flow and forcing end on 2014-09-30.
- MRMS is the operational multi-radar product from late 2014 onward. The only MRMS copy in the project is the
  teammates' `handoff/extras/Flood_Cloud_Model/data` (`mrms_point_basin_2020_2026.npz`, 2020 to 2026).
- So E7 has two arms. **E7a** runs on the main split with data already on disk and compares directly with the
  stored numbers. **E7b** runs on a 2021-2025 split that includes MRMS. E7b levels are comparable only inside that
  split.

## 2. E7a: all rain products together, main split (no download)

Split, seeds and settings: `flood.SPLIT` (train 1980-2000, validation 2000-2005, test 2005-2014). Seeds 0 to 4.
Settings from `results/lstm_all.json`, unchanged.

| Set | Inputs (all with the river gauge and the 10 static properties) | Role |
|---|---|---|
| `lstm_wqME` | Daymet + `prcp_nldas` + `rain_to_midnight` (ERA5) + `aorc_to_midnight` | treatment |
| `lstm_wqME_shuf` | as `wqME`, but the NLDAS and ERA5 columns shuffled within basin and calendar month across years, then retrained | capacity control: same column count and distributions, no timing information |
| `lstm_wqME_older` | as `wqME`, but NLDAS and ERA5 one day older | clock control: is any gain from the issue day's hours? |
| `lstm_wqa` (stored) | Daymet + AORC | the number to beat (51.7%) |
| `base_linreg_wqME`, `base_rf_wqME` | same columns, recipe of `02_baselines.py` / `features.py` | yardstick: information or network? |

Training: 3 new sets × 5 seeds = 15 trainings, one at a time under `F.cpu_lock`. The 50 E4 trainings took about
50 minutes on this PC, so expect 15 to 25 minutes.

**Success rule (fix in `ROADMAP.md` before the first training):**
1. Gain: every seed of `wqME` above every seed of `wqa` on onsets warned at 1 day (fixed level), with the false share
   not higher. Wilcoxon signed-rank over the 12 basins p < 0.05.
2. Capacity control flat: `wqME_shuf` within the seed range of `wqa`. If it rises as much as `wqME`, the gain is
   capacity, not information.
3. Clock: `wqME_older` within the seed range of `wqa`.
4. Network: if ridge and forest on `wqME` gain as much as the LSTM, report "the information, not the network" again.

Report also: median NSE at 1, 2 and 3 days. The 30%-false operating point (`warning_policy.py`, as E4a). The
resampling interval (`13_uncertainty.py` recipe). The GO / CAUTION / NOT YET verdict.

Caveat carried over: AORC is an analysis made after the fact. Any `wqME` number carries the same hindsight bound
as `wqa` (at most about 4 points) unless measured again. **Corrected by the review of 2026-10-30 (must-fix 1):** that
bound covers AORC only. NLDAS is published days late and ERA5 is a reanalysis, so `wqME` is a test of information, not
a forecast that could be issued. (The same sentence in `ROADMAP.md` section E7 is left as fixed before training.)

Script: `code/28_multi_product.py`, modelled on `16_aorc_rain_to_midnight.py` (`--rescore`, `--extend`,
`FLOOD_QUICK=1`, saved networks plus a reload check). Columns come from the builders scripts 15 and 16 already use.
No new data.

## 3. E7b: MRMS beside AORC, 2021-2025 split (RULED OUT, 30 October 2026)

**Ruled out by the team lead on 30 October 2026. Do not start it or re-propose it.** Step 0 below was checked that day
(metadata only, nothing downloaded):
- AORC on AWS: the bucket lists `2020.zarr` to `2025.zarr`, so it reaches 2021-2025 (whole-year coverage not checked).
- MRMS: the file named in section 1 does not exist anywhere under `15_AI_Class`. The package's `data/training.npz`
  holds only `observed_rain` (21 window summaries per 6-hourly issue), not an hourly basin series an LSTM can read.
  Running E7b would have needed a new MRMS bulk download or the teammates' raw file.
- Hourly USGS flow on disk (`data/hourly_flow/`) ends on 2014-10-01; 2021-2025 would need a new fetch.
- The `zstd` command-line tool that `code/fetch_aorc.py` needs is not installed on this PC.

The original design follows, unchanged.

Builds on E4f in `NEXT_EXPERIMENTS_2026-10-28.md`, which already plans a sequence model on the package's 2021-2025
inputs.

**Step 0 (gate; do this before anything else):**
- Does the AORC archive on AWS reach 2021-2025? Check the bucket's last year.
- Are the package's MRMS values area-weighted basin means? The file name says `point_basin`; `config.json` says
  basin means. Confirm which.
- Coverage per basin. Leaf River is nearly absent there (84 test cases).
- If AORC does not reach 2021, E7b becomes MRMS against MRMS + GEFS only. Say so.

| Set | Rain inputs (plus hourly gauge state and statics) |
|---|---|
| `s21_mrms` | MRMS (real time) |
| `s21_aorc` | AORC (after the fact) |
| `s21_mrms_aorc` | MRMS + AORC |
| `s21_mrms_aorc_shuf` | MRMS + AORC shuffled as in E7a |

- Split: train 2021-2022, choose on 2023, score once on 2024-2025, as E4f.
- Scoring: onset cases and distinct starts, against the package's own boosting results on identical cases.
- Limits: two training years is thin for an LSTM; few onsets; 6-hourly issues, not the midnight daily issue.
- Rerun the old recipe inside this split as its baseline. Never compare E7b levels with 51.7%; only gains inside
  E7b mean anything.

## 4. Answering "were the old numbers better?"

- **E7a: yes, directly.** Same split, test years, basins, seeds and warning level. `wqME` against `wqa` (51.7%),
  `wqnh` (32.4%) and `wq` (18.7%), with the rule of section 2.
- **E7b: ruled out (section 3).**

## 5. Review and reporting (same as every experiment)

- An independent AI reviewer recomputes the headline numbers from `results/preds` and the saved networks. It probes
  for leaks (no column after the issue time) and writes `results/review_multi_product.json`.
- Numbers enter documents only through `code/final_report/numbers.py`. Nothing goes into the handed-in report.

## 6. Expected outcome (stated before running)

E7a: 0 to 4 points over `wqa`, probably failing the strict seed rule, as E4b did. That would mean the products are
mostly redundant, and the clock of the rain matters more than the number of products. A clear gain would be news.
E7b: ruled out (section 3).
