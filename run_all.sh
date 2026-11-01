#!/bin/bash
# Reproduce the whole project from the raw CAMELS-US download, one step at a time (each step reads only what the steps above it wrote).
# The laptop has two cores: the training scripts take a lock so that only one runs at once, so just let this script run.
# A fast check that everything still works (tiny budgets, outputs in results/quick and figures/quick):  FLOOD_QUICK=1 ./run_all.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/AI_Class/.venv/bin/python}       # the environment the results were made with
[ -x "$PY" ] || PY=python                        # elsewhere: pip install -r requirements.txt first

$PY code/01_data_audit.py          # which basins have a clean record; picks the 12 study basins
$PY code/test_flood.py             # the fairness checks of the shared engine (code/flood.py)
$PY code/02_baselines.py           # 'tomorrow = today', ridge regression, random forest: the numbers to beat
$PY code/03_lstm.py                # LSTM on one river (French Broad): tuning, memory test
$PY code/04_lstm_all.py            # one LSTM for all 12 rivers, four sets of inputs, three seeds
$PY code/05_lobo.py                # leave-one-basin-out: a river the model has never seen
$PY code/07_basin_maps.py          # elevation and land cover per basin (downloads cached under data/dem, data/landcover)
$PY code/fetch_satellite.py        # IMERG rain and MERRA-2 variables from NASA POWER (cached under data/satellite)
$PY code/10_rain_timing.py         # the day stamp of the rain table, and what it means for the forecast gap (reads the satellite files)
$PY code/06_flood_analysis.py      # could the high-flow warnings be trusted; where and why they fail (reads the basin properties of 07)
$PY code/17_more_basins.py         # one LSTM for 63 rivers instead of 12: do more rivers help the same 12, and rivers never seen? (reads what 04, 05 and 06 wrote; trains 15 networks)
$PY code/check_2004_floods.py      # which hurricanes passed the French Broad gauge in September 2004 (uses the track file 06 downloaded)
$PY code/08_satellite_inputs.py    # satellite rain and soil wetness as model inputs
$PY code/09_model_comparison.py    # Transformer against LSTM, and the cost of training each
$PY code/check_day_windows.py      # which 24 hours each rain product's 'day' covers, measured against hourly rain (needs the satellite files)
$PY code/15_rain_to_midnight.py    # the rain of the whole issue day, up to midnight (hourly ERA5), as an input, and what it is worth (reads what check_day_windows.py and 06_flood_analysis.py wrote)
$PY code/16_aorc_rain_to_midnight.py # the same with NOAA AORC hourly rain (1 km, gauge- and radar-anchored) in place of ERA5, and the Daymet rain replaced by it (calls code/fetch_aorc.py, 4.5 GB and the zstd command-line tool, when data/aorc is missing; reads what 15 wrote)
$PY code/18_forecast_input.py     # the GEFSv12 reforecast rain (control member, bias-corrected basin means by code/gefs_to_basins.py) as an input beside the rain to midnight: substitution on the main split and retraining on the satellite-era split (reads what 16 wrote and data/gefs/basins, made by gefs_to_basins.py when missing)
$PY code/12_training_curves.py     # the learning curves that every training stored (no training)
$PY code/13_uncertainty.py         # how far the headline numbers move when test years and basins are drawn again (no training)
$PY code/19_data_flow.py           # the data text of the report and slides (what each product is, how it was obtained, the flow of the data, how it is computed) and the two flow diagrams; reads results/ and the caches (no training)
$PY code/check_same_misses.py      # do the kinds of model miss the same onsets? (stored forecasts, no training)
$PY code/11_key_figure.py          # it draws on the results above: the key figure of the main conclusion
$PY code/14_guide_figures.py       # four concept pictures for the step-by-step guide (stored forecasts, no training)

if [ "$FLOOD_QUICK" != "1" ]; then
  $PY code/build_report.py         # Flood_Report.html / .pdf and short_report.md, from results/ and figures/ only
  $PY code/build_report_short.py   # Flood_Report_3pages_dense.html / .pdf: the dense version of the report in exactly three A4 pages, from results/ and figures/ only
  $PY code/build_report_paper.py   # Flood_Report_3pages.docx / .pdf: the report as a formal scientific paper in exactly three A4 pages (the hand-in form; Word file and its PDF twin), from results/ and figures/ only
  $PY code/build_slides.py         # Flood_Slides.html / .pdf / .pptx and slides_notes.md
  $PY code/build_presentation_plan.py # Presentation_Plan_full_deck.html / .pdf / .md: who presents which slide of the 13-slide deck, for how long, what each member answers (reads results/slide_plan.json, written by build_slides.py)
  $PY code/build_slides_short.py   # Flood_Slides_5.html / .pdf / .pptx, slides_notes_5.md and results/slide_plan_5.json: the talk in five slides (the hand-in form, with a slide for the dataset from results/data_flow.json; the full deck above stays the detailed record)
  $PY code/build_presentation_plan_short.py # Presentation_Plan.html / .pdf / .md: the plan handed in, for the five-slide talk (reads results/slide_plan_5.json, written by build_slides_short.py)
  $PY code/build_plain_page.py     # Project_in_Plain_Words.html / .pdf: the project on one page in plain words, with the bar chart figures/plain_ladder.png
  $PY code/build_qa.py             # Professor_QA.md / .html / .pdf
  $PY code/build_overview.py       # the one-page overview
  $PY code/build_guide.py          # Project_Guide.html / .pdf: the project step by step, for readers new to machine learning
fi
