"""r2_ctx: print a regular expression with its context from one text file made by r2_check (pages split by '=====PAGE n====='):
    python code/review/documents_2026-10-08/r2_ctx.py TEXT_FOLDER Flood_Report.txt "tests one real forecast" 200
"""
import re
import sys
from pathlib import Path

base, doc, pat = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
w = int(sys.argv[4]) if len(sys.argv) > 4 else 300
t = (base / doc).read_text()
parts = re.split(r"\n=====PAGE (\d+)=====\n", "\n" + t) if doc.endswith(".txt") else [None, "1", t]
for i in range(1, len(parts), 2):
    pg, tx = parts[i], re.sub(r"\s+", " ", parts[i + 1])
    for m in re.finditer(pat, tx, re.I):
        print(f"[{doc} p{pg}] ...{tx[max(0, m.start() - w):m.end() + w]}...\n")
