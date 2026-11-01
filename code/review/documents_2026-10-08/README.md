# The final critic's check scripts for the documents of 8 October 2026

These scripts belong to the **independent AI review of the document set** (`Flood_Report`, `short_report.md`, the FINDINGS block of `README.md`, `Flood_Slides` and `slides_notes.md`, `Professor_QA`, `Project_Guide`,
`Project_Overview`) and of four files behind it: `results/stage4_realtime.json` (the real-time check on all 611 onsets, `code/check_stage4_realtime.py`), `results/onset_rain_clock.json` (the rain-clock diagnostic,
`code/check_onset_rain_clock.py`), the resampling ranges of `results/uncertainty.json` (`code/13_uncertainty.py`) and the key figure (`results/key_figure.json`, `figures/key_figure*.png`, `code/11_key_figure.py`).
They were written by an AI agent (Claude), the final critic, separately from the agents that wrote and ran the experiments and built the documents, and they are **not part of the project's own code**: `run_all.sh`
of the project, the report builders and the numbered scripts neither run nor read them. The review itself is `results/review_documents_2026-10-08.json` (the checks, the numbers, the discrepancies, the must-fix and
should-fix items with their status, the verdict and what was not checked). The documents cite it through `REVIEW_WORDING` in `code/report_sections/s0_summary.py`: *the real-time check on all onsets, the rain-clock
diagnostic, the resampling ranges and the key figure of 8 October 2026 were verified by the final critic agent only*. That record states that the team's own checks are still open; this AI review does not replace them.
The reviews of the four training experiments (`code/review/rain_to_midnight/`, `aorc_rain_to_midnight/`, `more_basins/`, `forecast_input/`) are separate folders and were not changed.

The scripts recompute with the critic's own code, from `results/preds`, `results/models`, `data/aorc`, `data/hourly` and the CAMELS tables that `code/flood.py` loads. They import only the engine's loaders
(`flood.load_data`: the daily tables; `flood.basins`: the ten basin properties) and, to compare, `flood.thresholds`, `flood.evaluate` and `flood.load_preds`. **Nothing is trained and nothing is written into the project**: every
output goes to `$REVIEW_OUT` (default `<temp folder>/documents_review`), one thread (`FLOOD_THREADS=1`, `torch.set_num_threads(1)`), `PYTHONDONTWRITEBYTECODE=1` (no `__pycache__` appears here: set it in the shell when you run
a script by hand, because Python writes the bytecode of the first imported module, `dcommon.py`, before it can switch writing off). Only `d9_assemble_review.py --write-into-project` writes into the project (the record,
and only when no reviewed results, data or code file is newer than the oldest check output).

## Run

From the project root, with the project environment:

```bash
export REVIEW_OUT=/some/folder PYTHONDONTWRITEBYTECODE=1
DOC_TEXT_DIR=/folder/with/pdf/text DOC_MD_DIR=/folder/with/md/files \
  FIGURE_OCR_DIR=/folder/with/ocr/json code/review/documents_2026-10-08/run_all.sh      # d5, d4, d1, d2, d2a, d3 in that order
DOC_SNAP_ROOT=/folder/that/holds/snap/ ~/AI_Class/.venv/bin/python code/review/documents_2026-10-08/d9_assemble_review.py                       # builds $REVIEW_OUT/review_documents_2026-10-08.json
~/AI_Class/.venv/bin/python code/review/documents_2026-10-08/d9_assemble_review.py --write-into-project                                        # also writes results/review_documents_2026-10-08.json
```

* `DOC_TEXT_DIR` holds `Flood_Report.txt`, `Flood_Slides.txt`, `Professor_QA.txt`, `Project_Guide.txt`, `Project_Overview.txt`: the text of each PDF with the pages separated by `=====PAGE n=====`. Make them with
  `swiftc pdf2txt.swift -o pdf2txt` and `./pdf2txt Flood_Report.pdf Flood_Report.txt` (macOS PDFKit). `DOC_MD_DIR` holds `short_report.md`, `slides_notes.md`, `README.md` (default: the project folder).
  **d1 checks the wording of the build of 8 October 10:21-10:24** (the version the critic judged); after a rebuild the claims whose sentence was changed fail by design until their wording is updated (round 2).
* `FIGURE_OCR_DIR` (optional, for d5) holds `key_figure.json`, `key_figure_plain.json`, `key_figure_talk.json` made with `swiftc ocr.swift -o ocr` and `./ocr figures/key_figure.png key_figure.json` (Apple Vision text recognition).
* `DOC_SNAP_ROOT` (for d9; optional) is a folder that holds the subfolders `snap/` (the documents of the build of 10:21-10:24), `snap2/` (slides of 11:30, README of 11:31) and `snap3/` (plan and plain-words page): the versions that were
  judged, whose hashes go into the record (`files_reviewed.documents_judged`, with a flag whether the project's file is still the same). `findings.json` in this folder holds the must-fix and should-fix items of the critique (and the two found while recomputing) with their status; d9 copies them into the record.

## What each script checks (final runs of 2026-10-08, laptop, one thread, other agents' jobs running: load average 20 to 58 on two cores, so the times are long)

| Script | Checks (check ids of the record) | Time |
|---|---|---|
| `d1_numbers.py` | 69 numbers of the documents (page 1, sections 4.6-4.8, section 8, slides 7 and 10-13, Q&A H1-H13, guide p.32, overview) against the results files they come from, page by page (c1) | seconds |
| `d2_headline_and_ranges.py` | part A: the critic's own scoring of 14 prediction sets at 3 leads (median NSE, all-day and onset warning tables) against `flood.evaluate`, 378 numbers (c2); part B: 4,000 draws of the 9 test water years and 12 basins with the critic's own random numbers: 24 statistics (onset shares with the ranges of section 8, the gap closed, the gain over the control, the NSE differences, the verdict shares) against `results/uncertainty.json`, point estimates at 1e-9 (c3) | 98 s |
| `d2a_first_pass_bootstrap.py` | the critic's first, shorter bootstrap (engine loaders, seed 1), kept because the critique quotes its numbers (c3) | 20 s |
| `d3_stage4.py` | the real-time check: own AORC day table and own forward pass of the saved `lstm_wqa` and `lstm_wqa_rep` networks reproduce the stored forecasts (largest difference 0.0); Stage IV hours of `data/aorc/stage4_issue_days.csv` substituted on the issue day of all 611 onsets (294, 275, 255, 319, 296, 274, 289 with the three networks of `wqa`), the 12-basin false-warning tally, intervals, the onsets without a replaced hour, the fits on the onset days, the agreement of the two products, and the three controls on the issue-day rain with own random numbers (c4-c7c) | 379 s |
| `d4_rain_clock.py` | the rain clock from the ERA5 hourly cache: four windows in local standard time, per-onset sums, 121 summary fields, the rain classes and the missed / warned flags (c8, c9) | 24 s |
| `d5_key_figure.py` | the key figure: its 607 sources re-read (491), cross-checked (777), recomputed (116), 360 of them re-derived from the stored forecasts with own scoring; the 34 printed labels of the three pictures and the lettering numbers against the sources; the counts of its own checks block (c10-c15) | 72 s |
| `d9_assemble_review.py` | builds the review record from the outputs and `findings.json` (the items with their round 2 status); refuses `--write-into-project` if a reviewed file is newer than the oldest output | seconds |
| `r2_check.py FOLDER` | round 2: the text of every PDF per page (PDFKit via `pdf2txt`), the Markdown by paragraph, the HTML and the PPTX with its notes searched for the patterns of the must-fix items M1-M8 and quick patterns of S5, S8, S10-S12, S14, S15; it lists the hits and decides nothing (every hit is read) | 1 minute |
| `r2_ctx.py TEXT_FOLDER FILE PATTERN [WIDTH]` | one pattern with its context, from the text files that `r2_check.py` wrote | seconds |
| `r2_new_numbers.py NAME=OLD.txt:NEW.txt ...` | round 2: the numbers that are more frequent in a rebuilt document than in the version judged; each is traced by hand to a results file (check c16) | seconds |
| `r3_tables.py` | round 3: Tables 1 and 2 of `Flood_Report_3pages.pdf` (15 rows) recomputed with the critic's own scoring from `results/preds` (per-seed and average forecasts), the 95% ranges from `results/uncertainty.json`, E2b from `results/forecast_input.json`; every row as printed is looked for on page 2 (r3a1) | 1 minute |
| `r3_check.py` | round 3: 130 checks on the text of the report, the slides (PDF and PPTX), the notes and the plan: dataset and flow numbers against `results/data_flow.json`, headline numbers, the wording rules (FINDING, SUPPORT, REVIEW_WORDING, NSE_FORECAST_WORDING, MOVED_POINT, the 51% and 48% wording), the dataset facts (no WorldCover, tiles, imagery note), pages, slides, notes, the plan's seconds and hand-overs (r3a2, r3a3, r3b-r3e) | seconds |
| `r3_new_numbers.py` | round 3: the numeric tokens of the four new documents that occur in none of the texts judged in rounds 1 and 2, and which of them `results/data_flow.json` holds; the rest is traced in the record (r3a3) | seconds |
| `r3_pptx_fit.py` | round 3: an estimate (Arial metrics) of the height of the text in every text frame of `Flood_Slides_5.pptx` against the frame | seconds |
| `r3_append.py [--write-into-project]` | round 3: assembles the `round_3` block (checks, must-fix M9, should-fix S18-S22, observations, counts, verdict) from the outputs above and adds it to the record in the project (any earlier `round_3` is replaced); run it after `d9_assemble_review.py`, which does not know round 3 | seconds |

Helpers: `dcommon.py` (paths, output folder, the 12 basins and their clocks, the high-flow levels and the onsets from the flow table), `pdf2txt.swift`, `ocr.swift`, `fontsizes.swift` (the printed font sizes of a PDF per page, PDFKit: `swift fontsizes in.pdf out.json`), `findings.json`.

## What the critic did and did not do

It trained nothing, installed nothing, downloaded nothing, did not run the project's numbered scripts, and wrote into the project only this folder and the review record. The document pass (wording, honesty, consistency, layout,
logic: the six checks) was done by reading and by grep over the text of every PDF, HTML, Markdown and PPTX file and about 60 renders; it has no script here except `d1_numbers.py` (the numbers) and `findings.json` (the
result). Not done: the raw Stage IV archive was not re-read (the check starts from the stored hourly series), the ERA5 and AORC caches were used as stored, the key figure was not redrawn and the positions of its marks were
not measured, the PPTX was not opened in PowerPoint; see `not_checked` in the record. **Round 2** (the final files of 13:48-13:50 against the must-fix and should-fix lists, and against the two items found while recomputing and the two found in the first look at the rebuilt files) was a reading and grep pass
with `r2_check.py`, `r2_ctx.py` and `r2_new_numbers.py`, contact sheets of every page, and a rerun of `d1`-`d5` on the regenerated results files; its result is the `round_2` block of the record and the statuses in `findings.json`
(`closed`, `partly closed`, `open`, with the evidence for each).

**Round 3** (the two hand-in documents of the afternoon, numbers and rules only: `Flood_Report_3pages.pdf/.html`, `Flood_Slides_5.pdf/.pptx` with `slides_notes_5.md` and `results/slide_plan_5.json`, `Presentation_Plan_5slides.pdf/.md`; final builds of 17:36-17:39) was run with
`R3_TEXT_DIR=<folder>` (the text of the three PDFs made with `pdf2txt`, the text of the PPTX, `fs_<name>.json` from `fontsizes.swift`, and `slide<n>_pic.json` from `ocr.swift` for the pictures of slides 1-3) and `R2_TEXT_DIR=<folder with the PDF texts of the long documents judged in round 2>`:
`r3_tables.py`, `r3_check.py`, `r3_new_numbers.py`, `r3_pptx_fit.py`, then `r3_append.py --write-into-project`. Its result is the `round_3` block of the record (checks r3a1-r3e, must-fix M9, should-fix S18-S22, `counts`, verdict). The builders of the report and the slides read every
`counts` block of every `results/review_*.json` for the line 'N review rounds, N listed checks' and for the sentence about the critic's comparisons, so the documents must be rebuilt after this block is written: they then read 9 rounds and 191 checks (S18). The pages were read at 100 dpi; PowerPoint was not available.
