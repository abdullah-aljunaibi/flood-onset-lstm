"""Assemble Flood_Report.html, Flood_Report.pdf and short_report.md from results/*.json, results/*.csv and figures/*.png,
and write the block 'What we found' of README.md.

Never recomputes anything: each file in code/report_sections/ turns the stored results into one part of the report, and
every number in the text is read from the results files. Rebuild after any rerun with: python code/build_report.py
(python code/build_report.py s3_single s4_all  previews only those sections in results/quick/).

Three checks belong to the full build and speak on stderr: the page limit (PAGE_LIMIT; before that line is printed, section 8
gives up what the project lead's cut order names, see below), the tables and figures the text points to (each must have a
caption with that number), and the two marker lines of README.md.
"""
import importlib
import re
import sys

from common import RES, ROOT
from report_common import load_results, page, pdf_pages, print_pdf
from report_sections import s0_summary

SECTIONS = ["s0_summary", "s1_data", "s2_baselines", "s3_single", "s4_all", "s4b_lobo", "s4c_timing", "s4d_midnight", "s4e_rivers", "s4f_forecast", "s5_flood", "s6a_maps",
            "s6b_satellite", "s7_cost", "s8_limits"]
# The page limit of the report: 25 pages, self-imposed on 4 October 2026; raised to 30 on 8 October 2026 for the four experiments added on 7 and 8 October (sections 4.6 to 4.8:
# rain known up to the issue time with ERA5 and with AORC, more rivers in training, a real weather forecast as an input), which the earlier limit left no room for.
# A longer build says so on stderr.
PAGE_LIMIT = 31                             # raised from 30 to 31 on 8 October 2026 (afternoon) so that Table 8.1 stays after the critic's wording additions
# What section 8 gives up, in this order, when the full build is over the limit (s8_limits.CUTS): 1 = Table 8.1 (one sentence that lists
# the steps stays), 2 = also the reading guide that page 1 has no room for. Nothing is given up while the report fits.
CUT_WORDS = {1: "Table 8.1 left out", 2: "Table 8.1 and the reading guide left out"}
# The print scale of the report. Until 4 October 2026 nobody had set one, and the report was never printed at full size: Table 3.1 (section 3)
# is set not to wrap and is 44 px wider than the page, so the browser shrank the WHOLE report to make it fit, to 94% (measured on the printed
# title, in the version of 3 October and in the builds of 4 October alike). A shorter cell in that one table would have enlarged every page:
# at full size the same report has 28 pages instead of 25. The scale is therefore set here, at the value the report has had all along, so
# that it no longer depends on one table, and so that the test print of page 1 (s0_summary.page1_plan) is made at the scale of the report.
# 0.939 is just below the 0.9405 at which Table 3.1 fits, so the browser shrinks nothing further. None = leave the scale to the browser.
PRINT_SCALE = 0.939
# Page rules of the report only (the slides and the overview use the shared CSS with other pages, so these are not in report_common.CSS):
# a running title and 'page / pages' in the bottom margin of every page, and a blue bar that makes the start of a section easy to find.
# (Page 1 has its own, smaller margins in s0_summary.STYLE: 10 mm at the bottom, which still holds this footer clear of the paper edge.)
FOOT = "font:7.5pt 'Helvetica Neue',sans-serif;color:#898781"
REPORT_EXTRA = ("@page{@bottom-left{content:'Predicting river floods with AI \\00B7  AI course group project';" + FOOT + "}"
                "@bottom-right{content:counter(page) ' / ' counter(pages);" + FOOT + "}} "
                "h2{border-left:4px solid var(--blue);padding-left:7pt}" + (f" body{{zoom:{PRINT_SCALE}}}" if PRINT_SCALE else ""))
s0_summary.PRINT_CSS = REPORT_EXTRA         # the test print that fits page 1 on one page uses the same page rules
names = sys.argv[1:] or SECTIONS            # with section names as arguments: a preview of just those, written to results/quick/
full = not sys.argv[1:]

R, T = load_results()
parts, built = [], []
for name in names:
    try:
        parts.append(importlib.import_module(f"report_sections.{name}").html(R, T))
        built.append(name)
    except ModuleNotFoundError as e:
        if e.name != f"report_sections.{name}":
            raise
        print("section not written yet:", name, file=sys.stderr)
out = RES / "quick" / f"preview_{'-'.join(names)}.html" if not full else ROOT / "Flood_Report.html"
pdf = out.with_suffix(".pdf")
out.parent.mkdir(exist_ok=True)


def write():
    """Write the HTML and print it; the number of pages of the PDF, or None when no browser printed one."""
    out.write_text(page("Predicting river floods with AI", "\n".join(parts), css_extra=REPORT_EXTRA), encoding="utf-8")
    return pdf_pages(pdf) if print_pdf(out, pdf) else None


n_pages = write()
if full and "s8_limits" in built:           # over the page limit: section 8 gives up Table 8.1, then the reading guide, and the report is printed again
    s8 = importlib.import_module("report_sections.s8_limits")

    def again(cuts):
        s8.CUTS = cuts
        parts[built.index("s8_limits")] = s8.html(R, T)
        return write()

    kept, first = 0, n_pages
    for cuts in sorted(CUT_WORDS):
        if not (n_pages and n_pages > PAGE_LIMIT and hasattr(s8, "CUTS")):
            break
        n_new = again(cuts)
        if n_new and n_new < n_pages:        # a cut is kept only if it saves a page
            kept, n_pages = cuts, n_new
    if hasattr(s8, "CUTS") and s8.CUTS != kept:
        n_pages = again(kept)                # what did not save a page is put back
    if first and first > PAGE_LIMIT:
        print(f"build_report: {first} pages are over the limit of {PAGE_LIMIT}: "
              + (f"section 8 built with {CUT_WORDS[kept]} -> {n_pages} pages" if kept else
                 "leaving out Table 8.1 and the reading guide of section 8 saves no page, so they stay"))
print("wrote", out, f"({len(parts)} of {len(names)} sections)")
if n_pages is not None:
    print("wrote", pdf, f"({n_pages} pages)")
    if full and n_pages > PAGE_LIMIT:
        print(f"build_report: the report has {n_pages} pages, the limit is {PAGE_LIMIT}", file=sys.stderr)
if full:                                    # every table and figure the text points to must have its caption
    text = re.sub(r'src="data:[^"]*"', "", "\n".join(parts))
    defined = set(re.findall(r"<b>((?:Table|Figure) \d+\.\d+)\b", text))
    cited = {f"{kind} {n}" for kind, nums in re.findall(r"\b(Table|Figure)s? ((?:\d+\.\d+)(?:(?:,? and |, | to )\d+\.\d+)*)", re.sub(r"<[^>]+>", " ", text))
             for n in re.findall(r"\d+\.\d+", nums)}
    if cited - defined:
        print("build_report: the text points to", ", ".join(sorted(cited - defined)), "- no caption with that number exists", file=sys.stderr)
    # the full build also writes the readable one-page version, from the same results
    (ROOT / "short_report.md").write_text(s0_summary.md(R, T), encoding="utf-8")
    print("wrote", ROOT / "short_report.md")
    # ... and the block 'What we found' of README.md, between its two marker lines (the rest of that file is written by hand)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    a, b = text.find(s0_summary.README_START), text.find(s0_summary.README_END)
    if 0 <= a < b:
        new = text[:a + len(s0_summary.README_START)] + "\n" + s0_summary.readme_block(R, T) + "\n" + text[b:]
        if new != text:
            readme.write_text(new, encoding="utf-8")
        print("wrote the block 'What we found' of", readme)
    else:
        print("build_report: README.md has no FINDINGS markers (or they are out of order): its block 'What we found' was not written", file=sys.stderr)
