"""r2_check: round 2 of the final critic - the must-fix list M1-M8 (and quick greps for should-fix items) against a folder of documents.

    python code/review/documents_2026-10-08/r2_check.py FOLDER [TEXT_FOLDER]

FOLDER holds the PDFs and the md / html / pptx files (the project folder, or a snapshot). The text of every PDF is extracted per page with the Swift PDFKit tool pdf2txt (compile it once:
swiftc pdf2txt.swift -o pdf2txt; or set PDF2TXT to a compiled binary) into TEXT_FOLDER (default $REVIEW_OUT/txt_<name of FOLDER>); the Markdown files are read as paragraphs, the HTML files with their tags
removed, the PPTX with python-pptx (text of every shape, then the notes). The output lists, for each item, the places where its pattern occurs (document, page or paragraph, context). It decides nothing:
every hit is read by a person (r2_ctx.py prints a pattern with more context). Written to stdout.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUTDIR = Path(os.environ.get("REVIEW_OUT") or (Path(os.environ.get("TMPDIR", "/tmp")) / "documents_review"))
TXT = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTDIR / ("txt_" + ROOT.name)
TXT.mkdir(parents=True, exist_ok=True)
PDFS = ["Flood_Report", "Flood_Slides", "Professor_QA", "Project_Guide", "Project_Overview", "Presentation_Plan", "Project_in_Plain_Words"]
MDS = ["short_report.md", "slides_notes.md", "README.md", "Professor_QA.md", "Presentation_Plan.md"]
SWIFT = Path(os.environ.get("PDF2TXT") or Path(__file__).parent / "pdf2txt")
DOC = {}                                    # name -> {page: text}
for n in PDFS:
    p = ROOT / f"{n}.pdf"
    if not p.exists():
        continue
    out = TXT / f"{n}.txt"
    subprocess.run([str(SWIFT), str(p), str(out)], capture_output=True)
    parts = re.split(r"\n=====PAGE (\d+)=====\n", "\n" + out.read_text())
    DOC[n + ".pdf"] = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
for n in MDS:
    p = ROOT / n
    if p.exists():
        DOC[n] = {i + 1: para for i, para in enumerate(re.split(r"\n\s*\n", p.read_text()))}        # 'page' = paragraph number for Markdown
for n in ("Flood_Report.html", "Professor_QA.html", "Project_Guide.html", "Project_Overview.html", "Flood_Slides.html"):
    p = ROOT / n
    if p.exists():
        t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", p.read_text(errors="ignore"), flags=re.S)
        t = re.sub(r"<img[^>]*>", " ", t)
        t = re.sub(r"<[^>]+>", " ", t)
        DOC[n] = {1: re.sub(r"&[a-z]+;|&#\d+;", " ", t)}
pptx = ROOT / "Flood_Slides.pptx"
if pptx.exists():
    from pptx import Presentation
    pr = Presentation(str(pptx))
    DOC["Flood_Slides.pptx"] = {i + 1: " ".join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame) + " NOTES: " + (s.notes_slide.notes_text_frame.text if s.has_notes_slide else "") for i, s in enumerate(pr.slides)}
norm = lambda s: re.sub(r"\s+", " ", s.replace("’", "'").replace("‘", "'").replace("−", "-").replace("–", "-"))


def hits(pat, flags=re.I, width=110, docs=None):
    out = []
    for d, pages in DOC.items():
        if docs and d not in docs:
            continue
        for pg, t in pages.items():
            t = norm(t)
            for m in re.finditer(pat, t, flags):
                out.append((d, pg, t[max(0, m.start() - width):m.end() + width]))
    return out


def show(title, hs, limit=14):
    print(f"\n--- {title}: {len(hs)} hit(s)")
    for d, pg, c in hs[:limit]:
        print(f"   {d} p/para {pg}: ...{c}...")
    if len(hs) > limit:
        print(f"   ... and {len(hs) - limit} more")


print("documents read:", {d: len(p) for d, p in DOC.items()})
show("M1 template placeholders { } with names (None / nan in running text are ordinary words and are read, not counted)", hits(r"\{[A-Za-z_]\w*(\[['\"]?\w+['\"]?\])?(\.\w+)?\}") + hits(r"\bnan\b|\bNaN\b|undefined", flags=0, docs=[d for d in DOC if not d.endswith('.html')]))
show("M2 'not tested' / 'results do not say where' about a weather forecast", hits(r"(was|were|is|are) not tested") + hits(r"results do not say where"), 6)
show("M3 'all three lead times' with mixed / NSE", hits(r"mixed[^.]{0,60}all three lead times|all three lead times[^.]{0,80}mixed|a mixed result at all three"))
show("M3 'not lost skill'", hits(r"not lost skill"), 6)
show("M3 fixed clause present (the forecast lowers the 1-day score)", hits(r"forecast lowers the 1-day score|lowers the 1-day (NSE|score)"), 4)
show("M4 'service could expect'", hits(r"service could expect|a service could"))
show("M4 'in real time' not followed by 'radar'", hits(r"in real time(?! radar)"))
show("M4 the retrospective-check sentence present", hits(r"not a (working )?service|check on the test years|a real-time radar check"), 6)
show("M5 AORC 'observed / measured / really fell'", hits(r"AORC[^.]{0,60}(observed|measured)|(observed|well-measured|measured)[^.]{0,30}AORC|rain that (really )?fell|really fell|the observed AORC"), 8)
no_h = []
for d, pages in DOC.items():
    if d.endswith(".html"):
        continue
    for pg, t in pages.items():
        t = norm(t)
        if re.search(r"\b51%", t) and not re.search(r"hindsight", t, re.I):
            m = re.search(r"\b51%", t)
            no_h.append((d, pg, t[max(0, m.start() - 90):m.end() + 90]))
show("M6 pages / paragraphs with 51% and no 'hindsight' on the same page (Markdown: same paragraph): read each: only a headline 51% needs it", no_h, 40)
show("M7 'every script' / 'each experiment script'", hits(r"every (experiment )?script|each experiment script|checked by its own independent|independent AI reviewers? checked every"), 6)
show("M7 final critic named", hits(r"final critic"), 4)
show("M8 'more rivers moved the onsets'", hits(r"more rivers moved|rivers moved the onsets|moved the onsets warned"), 6)
show("S8 'about 4 hours'", hits(r"about 4 hours"), 8)
show("S10 'No range reaches'", hits(r"No range reaches"))
show("S11 'tie' / 'tied'", hits(r"\b(tie|tied|ties)\b"), 12)
show("S12 '58 / 130 listed checks'", hits(r"\b(58|130) (listed )?checks|in (58|130) listed"), 12)
show("S14 '48 hours around'", hits(r"48 hours around|48 hours"), 8)
show("S15 'holds in every (draw|resample)'", hits(r"holds in every (draw|resample)"))
show("S5 'chose nothing'", hits(r"chose nothing|test years chose|years that chose"), 8)
