#!/usr/bin/env bash
# The critic's check scripts for the documents of 8 October 2026, in the order they were run. From the project root:
#   REVIEW_OUT=/some/folder DOC_TEXT_DIR=/folder/with/the/pdf/text DOC_MD_DIR=/folder/with/the/md/files code/review/documents_2026-10-08/run_all.sh
# One thread (FLOOD_THREADS=1, torch.set_num_threads(1)); nothing is trained; nothing is written into the project (outputs go to $REVIEW_OUT).
# d3 is the long one (about 6 minutes on an idle laptop; 10 on a loaded one); d2 about 3-10 minutes; the others seconds to a minute.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$HERE/../../.." && pwd)"
PY="${PY:-$HOME/AI_Class/.venv/bin/python}"
export REVIEW_OUT="${REVIEW_OUT:-${TMPDIR:-/tmp}/documents_review}"
export PYTHONDONTWRITEBYTECODE=1 FLOOD_THREADS=1
mkdir -p "$REVIEW_OUT"
cd "$PROJ"
# the text of the PDFs (only if DOC_TEXT_DIR is not given): swiftc pdf2txt.swift -o pdf2txt; ./pdf2txt Flood_Report.pdf Flood_Report.txt ... (pages are separated by '=====PAGE n=====')
# the OCR of the three key-figure renderings (optional, for d5): swiftc ocr.swift -o ocr; ./ocr figures/key_figure.png key_figure.json (and _plain, _talk); then FIGURE_OCR_DIR=<folder with the three .json>
for s in d5_key_figure d4_rain_clock d1_numbers d2_headline_and_ranges d2a_first_pass_bootstrap d3_stage4; do
  if [ "$s" = d1_numbers ] && [ -z "${DOC_TEXT_DIR:-}" ]; then echo "skipping d1_numbers (DOC_TEXT_DIR not set)"; continue; fi
  echo "=== $s $(date +%H:%M:%S)"
  "$PY" "code/review/documents_2026-10-08/$s.py" 2>&1 | tee "$REVIEW_OUT/$s.log" | tail -25
done
echo "=== done $(date +%H:%M:%S); outputs in $REVIEW_OUT; the record: $PY code/review/documents_2026-10-08/d9_assemble_review.py [--write-into-project], then (round 3, see README.md) r3_append.py [--write-into-project]"
