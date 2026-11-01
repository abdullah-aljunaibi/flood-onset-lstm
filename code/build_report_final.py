#!/usr/bin/env python
"""Builds the final report (hand-in item 3) on the professor's NSFC-style template, from the section files and the numbers registry.

    python -X utf8 code/build_report_final.py            (from the project root, system Python 3.13; torch-free)
    python -X utf8 code/build_report_final.py --no-pdf   (skip the PDF twin)

What it does (see BUILD_REPORT_FINAL.md in the hand-in folder for the counts of the last build):
  1. copies the professor's template once to code/final_report/report_template.docx (read-only source, never changed) and opens that copy;
  2. sets A4 with 2.5 cm margins, Normal = Times New Roman 11 pt at 1.15 line spacing; the template's heading styles, tables and footer stay;
  3. fills the cover (title lines from code/final_report/config.json, the team table T1, group leader, dates) and T3 Project Information;
  4. deletes the professor's "Document Requirements and Reporting Principles" block, every "Instructions: ..." paragraph and every
     bracketed placeholder paragraph;
  5. inserts, after each template heading, the prose of code/final_report/sections/<id>.md (abstract, 1-1 ... 10-3, appendix-a ... e):
     paragraphs, **bold**, {{num:key}} -> the registry text (results/final_report_numbers.json, written by code/final_report/numbers.py),
     [@key] -> IEEE numbers in order of first citation (code/final_report/references.py), "<!-- table: Tn | caption -->" + a Markdown table
     -> the template's table Tn moved to that point and filled (or a new table in the same style), "<!-- figure: path | caption [| width=13cm] -->" ->
     the PNG at up to 16 cm (or the width given) with its caption, "<!-- optional: prefix -->...<!-- /optional -->" dropped when a key of that prefix is missing;
  6. numbers tables and figures in order of appearance and rewrites the writers' "Table Tn" / "Figure n" references to match;
  7. writes section 11 References and regenerates the Table of Contents (plain paragraphs wrapped in a TOC field; page numbers from the PDF twin);
  8. saves Handin_2026-11-02/3_Final_Project_Report_Flood.docx and prints the PDF twin from an HTML twin with headless Edge
     (Times New Roman 11 pt, A4, figures embedded), stamps the template's footer and page numbers on every PDF page with PyMuPDF,
     counts the pages and writes Handin_2026-11-02/BUILD_REPORT_FINAL.md.

Rules kept: no number is typed here; every number is a registry text; a missing key prints "[MISSING: key]" and is counted; nothing from
results/diagnostics_2026-10-28/ is read (the registry never reads it); the fixed wordings are the registry's fw_* texts.
"""
import base64
import copy
import html as _html
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
FR = ROOT / "code" / "final_report"
SECTIONS_DIR = FR / "sections"
SRC_TEMPLATE = FR / "Final_Project_Report.docx"   # the course's report template as received (not part of the public repository)
TEMPLATE = FR / "report_template.docx"
REGISTRY = ROOT / "results" / "final_report_numbers.json"
CONFIG = FR / "config.json"
HANDIN = ROOT.parent / "Handin_2026-11-02"   # the hand-in folder, next to the project folder
OUT_DOCX = HANDIN / "3_Final_Project_Report_Flood.docx"
OUT_PDF = HANDIN / "3_Final_Project_Report_Flood.pdf"
BUILD_MD = HANDIN / "BUILD_REPORT_FINAL.md"

SUBMISSION_DATE = "1 November 2026"          # the planned day of sending; change here on the day of the e-mail
TEXT_WIDTH_CM = 16.0                         # A4 (21.0 cm) minus 2 x 2.5 cm
TEXT_WIDTH_TWIPS = int(TEXT_WIDTH_CM / 2.54 * 1440)
TABLE_PT = 9                                 # font size inside the tables (the template's Normal is used for the prose)
MAX_FIG_HEIGHT_CM = 20.0
FIG_WIDTH_CM = 14.0                          # default width of a figure; the drawings whose lettering needs the full column are listed here
FIG_WIDTH_BY_PATH = {"figures/data_flow.png": 16.0, "figures/key_figure.png": 15.0, "figures/day_windows.png": 15.0}
# the content plan's word budget per section (scratchpad audits/agent6.md section 2), printed beside the counts of the build so an overrun is visible
PLAN_WORDS = {"Abstract": 300, "1.1": 250, "2.1": 280, "2.2": 550, "2.3": 200, "3.1": 180, "3.2": 120, "3.3": 220, "4.1": 180, "4.2": 380, "4.3": 150,
              "5.1": 150, "5.2": 420, "5.3": 300, "5.4": 300, "6.1": 250, "6.2": 120, "6.3": 330, "7.1": 250, "7.2": 300, "7.3": 550, "7.4": 80,
              "8.1": 350, "8.2": 380, "8.3": 220, "9.1": 350, "9.2": 60, "9.3": 280, "10.1": 60, "10.2": 100, "10.3": 60}
SECTION_IDS = (["abstract", "1-1", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3", "4-1", "4-2", "4-3", "5-1", "5-2", "5-3", "5-4", "6-1", "6-2", "6-3",
                "7-1", "7-2", "7-3", "7-4", "8-1", "8-2", "8-3", "9-1", "9-2", "9-3", "10-1", "10-2", "10-3"]
               + [f"appendix-{c}" for c in "abcde"])
# the template's tables, by the writers' ids (T2 is the professor's requirements table and is deleted)
TEMPLATE_TABLE_INDEX = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "T6": 5, "T7": 6, "T8": 7, "T9": 8, "T10": 9, "T11": 10, "T12": 11}

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


# ---------------------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------------------
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REFS = load_module("final_report_references", FR / "references.py")
REG = json.loads(REGISTRY.read_text(encoding="utf-8"))
CFG = json.loads(CONFIG.read_text(encoding="utf-8"))


def num_text(key):
    e = REG.get(key)
    return None if e is None or not isinstance(e, dict) or "text" not in e else str(e["text"])


# ---------------------------------------------------------------------------------------------------------------------------
# Markdown of the sections -> blocks
# ---------------------------------------------------------------------------------------------------------------------------
class Block:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)


DIRECTIVE = re.compile(r"<!--\s*(table:|figure:|optional:|/optional)")
COMMENT = re.compile(r"<!--.*?-->", re.S)
OPTIONAL = re.compile(r"<!--\s*optional:\s*(\w+)\s*-->\n?(.*?)<!--\s*/optional\s*-->\n?", re.S)
TOKEN = re.compile(r"\{\{num:([^}]*)\}\}")
CITE_GROUP = re.compile(r"\[@\w+\](?:\s*,\s*\[@\w+\])*")
CITE = re.compile(r"\[@(\w+)\]")


class Build:
    """Everything the build counts, so the report can say what happened."""

    def __init__(self):
        self.missing = set()         # registry keys that no entry answered
        self.unknown_refs = []
        self.dropped_optional = []   # (section, prefix, missing key)
        self.cite_order = []         # reference keys in order of first citation
        self.numbering = None
        self.table_ids = []          # writers' ids in document order (T3 first)
        self.figure_ids = []         # writers' figure numbers in document order
        self.n_figures = 0
        self.n_tables = 0
        self.notes = []


B = Build()


def strip_comments(text, section):
    """Resolve optional blocks, drop the writers' notes (every comment that is not a directive)."""
    def opt(m):
        prefix, inner = m.group(1), m.group(2)
        for key in TOKEN.findall(inner):
            if key.startswith(prefix + "_") and num_text(key) is None:
                B.dropped_optional.append((section, prefix, key))
                return ""
        return inner
    text = OPTIONAL.sub(opt, text)
    return COMMENT.sub(lambda m: m.group(0) if DIRECTIVE.match(m.group(0)) else "", text)


def parse_section(section):
    text = (SECTIONS_DIR / f"{section}.md").read_text(encoding="utf-8")
    text = strip_comments(text, section)
    blocks, para, table, pending = [], [], None, None

    def flush_para():
        nonlocal para
        if para:
            blocks.append(Block("p", text=" ".join(l.strip() for l in para)))
            para = []

    def flush_table():
        nonlocal table, pending
        if table is not None:
            rows = [r for r in table if not re.fullmatch(r"\|[\s:|-]+\|", r.strip())]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            tid, cap = pending if pending else (None, "")
            blocks.append(Block("table", tid=tid, caption=cap, header=cells[0], rows=cells[1:]))
            table, pending = None, None

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            flush_para()
            table = (table or []) + [s]
            continue
        flush_table()
        if not s:
            flush_para()
            continue
        m = re.match(r"<!--\s*table:\s*(T\w+)\s*\|\s*(.*?)\s*-->$", s)
        if m:
            flush_para()
            pending = (m.group(1), m.group(2))
            continue
        m = re.match(r"<!--\s*figure:\s*(\S+)\s*\|\s*(.*?)\s*-->$", s)
        if m:
            flush_para()
            caption, width = m.group(2), None
            mw = re.search(r"\s*\|\s*width\s*=\s*(\d+(?:\.\d+)?)\s*cm\s*$", caption)      # optional "| width=13cm" at the end of the line
            if mw:
                caption, width = caption[:mw.start()].rstrip(), float(mw.group(1))
            blocks.append(Block("figure", path=m.group(1), caption=caption, width=width))
            continue
        if s.startswith("<!--"):
            continue
        para.append(s)
    flush_para()
    flush_table()
    return blocks


# ---------------------------------------------------------------------------------------------------------------------------
# Inline text: tokens, citations, bold, cross-references
# ---------------------------------------------------------------------------------------------------------------------------
def collect_cites(text):
    for key in CITE.findall(text):
        if key in REFS.REFERENCES:
            if key not in B.cite_order:
                B.cite_order.append(key)
        elif key not in B.unknown_refs:
            B.unknown_refs.append(key)


def substitute_tokens(text):
    def tok(m):
        key = m.group(1)
        t = num_text(key)
        if t is None:
            B.missing.add(key)
            return f"[MISSING: {key}]"
        return t
    return TOKEN.sub(tok, text)


def substitute_cites(text):
    def grp(m):
        keys = [k for k in CITE.findall(m.group(0)) if k in REFS.REFERENCES]
        bad = [k for k in CITE.findall(m.group(0)) if k not in REFS.REFERENCES]
        out = REFS.marker(B.numbering, *keys) if keys else ""
        if bad:
            out += "".join(f"[?{k}]" for k in bad)
        return out
    return CITE_GROUP.sub(grp, text)


TABLE_NO, FIGURE_NO = {}, {}


def renumber_refs(text):
    """The writers' 'Table T7' / 'Table 5' / 'Figure 3' -> the numbers of this document (order of appearance)."""
    def tab(m):
        tid = "T" + m.group(2)
        return f"{m.group(1)} {TABLE_NO.get(tid, m.group(2))}"
    text = re.sub(r"\b(Tables?) T?(\d+b?)\b", tab, text)

    def figs(m):
        return f"Figures {FIGURE_NO.get(int(m.group(1)), m.group(1))} and {FIGURE_NO.get(int(m.group(2)), m.group(2))}"
    text = re.sub(r"\bFigures (\d+) and (\d+)\b", figs, text)
    text = re.sub(r"\bFigure (\d+)\b", lambda m: f"Figure {FIGURE_NO.get(int(m.group(1)), m.group(1))}", text)
    return text


def inline_runs(text):
    """-> [(text, bold)]; **bold** toggles, `code` marks are dropped, tokens, citations and cross-references resolved."""
    text = renumber_refs(substitute_cites(substitute_tokens(text)))
    text = text.replace("`", "")
    runs, bold = [], False
    for i, part in enumerate(text.split("**")):
        if part:
            runs.append((part, bold))
        bold = not bold
    return runs


def plain(text):
    return "".join(t for t, _ in inline_runs(text))


# ---------------------------------------------------------------------------------------------------------------------------
# python-docx helpers (everything works on the template's own elements)
# ---------------------------------------------------------------------------------------------------------------------------
def strip_ids(el):
    for e in el.iter():
        for a in list(e.attrib):
            if a.endswith("}paraId") or a.endswith("}textId"):
                del e.attrib[a]


def para_of(el, doc):
    return Paragraph(el, doc._body)


def new_paragraph(doc, after_el, style=None):
    p = OxmlElement("w:p")
    after_el.addnext(p)
    par = para_of(p, doc)
    if style:
        par.style = doc.styles[style]
    return par


def add_runs(par, runs, size=None, font=None, color=None, italic=None):
    for text, bold in runs:
        r = par.add_run(text)
        if bold:
            r.bold = True
        if italic:
            r.italic = True
        if size:
            r.font.size = Pt(size)
        if font:
            r.font.name = font
            rpr = r._r.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            rf.set(qn("w:hAnsi"), font)
            rf.set(qn("w:cs"), font)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return par


def delete_el(el):
    el.getparent().remove(el)


def set_par_text(par, runs, plain_look=True):
    """Replace the runs of a template paragraph, keeping its first run's formatting (minus the placeholder look)."""
    for r in par.runs[1:]:
        delete_el(r._r)
    if par.runs:
        r0 = par.runs[0]
        if plain_look and r0._r.rPr is not None:
            for tag in ("w:i", "w:iCs", "w:color"):
                for el in r0._r.rPr.findall(qn(tag)):
                    r0._r.rPr.remove(el)
        text, bold = runs[0]
        r0.text = text
        if bold:
            r0.bold = True
        for text, bold in runs[1:]:
            r = par.add_run(text)
            if bold:
                r.bold = True
    else:
        add_runs(par, runs)
    return par


def cell_set(tc, runs, size=TABLE_PT, bold_all=False):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    p = ps[0] if ps else OxmlElement("w:p")
    if not ps:
        tc.append(p)
    for child in list(p):
        if child.tag in (qn("w:r"), qn("w:hyperlink"), qn("w:fldSimple")):
            p.remove(child)
    par = Paragraph(p, None)
    for text, bold in runs or [("", False)]:
        r = par.add_run(text)
        r.font.size = Pt(size)
        if bold or bold_all:
            r.bold = True
    return p


def set_tc_width(tc, twips):
    tcpr = tc.find(qn("w:tcPr"))
    if tcpr is None:
        tcpr = OxmlElement("w:tcPr")
        tc.insert(0, tcpr)
    w = tcpr.find(qn("w:tcW"))
    if w is None:
        w = OxmlElement("w:tcW")
        tcpr.insert(0, w)
    w.set(qn("w:w"), str(int(twips)))
    w.set(qn("w:type"), "dxa")


CELL_PAD_TWIPS = 216                                                 # the template's cell margins: 108 twips left and right
_TABLE_FONTS = {}


def text_width_twips(text, bold=False):
    """Width of `text` at TABLE_PT in twips, measured in Times New Roman (the tables' font) when Pillow and the font are at hand;
    otherwise about half an em per character."""
    key = "bold" if bold else "regular"
    if key not in _TABLE_FONTS:
        try:
            from PIL import ImageFont
            path = {"bold": "C:/Windows/Fonts/timesbd.ttf", "regular": "C:/Windows/Fonts/times.ttf"}[key]
            _TABLE_FONTS[key] = ImageFont.truetype(path, TABLE_PT * 10)
        except Exception:
            _TABLE_FONTS[key] = None
    font = _TABLE_FONTS[key]
    if font is None:
        return int(len(text) * TABLE_PT * 20 * (0.55 if bold else 0.5))
    return int(font.getlength(text) / 10 * 20)


def longest_word_twips(texts, bold):
    """The widest piece a word processor cannot break: the text split at spaces and after a hyphen. A slash is not a break
    (Word never breaks there; LibreOffice did so only sometimes in a test of 29 Oct 2026), so a path such as
    results/warning_policy.json counts as one piece."""
    w = 0
    for t in texts:
        for piece in re.split(r"\s+|(?<=-)", t):
            if piece:
                w = max(w, text_width_twips(piece, bold))
    return w


def widen_for_longest_word(widths, header, rows, n, min_w):
    """No column narrower than its longest unbreakable word plus the cell margins (LibreOffice and Word otherwise split the word at the
    cell edge: 'Implementatio|n', 'stage4_realtim|e.json', 'Environme|ntal' in the render of 29 Oct 2026), up to 30% of the text width;
    the extra width comes from the other columns in proportion to the room each has, so the table keeps the text width. The header
    row and the first column are bold."""
    need = []
    for c in range(n):
        head = [plain(header[c])] if c < len(header) else []
        body = [plain(r[c]) for r in rows if c < len(r)]
        w = max(longest_word_twips(head, True), longest_word_twips(body, c == 0))
        need.append(min(w + CELL_PAD_TWIPS + 40, int(TEXT_WIDTH_TWIPS * 0.3)))
    for c in sorted(range(n), key=lambda k: widths[k] - need[k]):
        short = need[c] - widths[c]
        if short <= 0:
            continue
        spare = {k: widths[k] - max(need[k], min_w) for k in range(n) if k != c}      # what each other column can give
        spare = {k: v for k, v in spare.items() if v > 0}
        total = sum(spare.values())
        if total <= 0:
            continue
        give = min(short, total)
        for k, v in spare.items():                                                    # in proportion to the room each has
            part = int(give * v / total)
            widths[k] -= part
            widths[c] += part
        rest = need[c] - widths[c]
        if 0 < rest <= give:                                                          # the rounding remainder, from the roomiest column
            k = max(spare, key=spare.get)
            widths[k] -= rest
            widths[c] += rest
    return widths


def column_widths(header, rows, n):
    """Widths in twips from the text each column carries (clamped), summing to the text width."""
    lens = []
    for c in range(n):
        texts = [plain(header[c]) if c < len(header) else ""] + [plain(r[c]) if c < len(r) else "" for r in rows]
        avg = sum(min(len(t), 160) for t in texts) / max(1, len(texts))
        lens.append(max(12.0, avg) ** 0.85)
    total = sum(lens)
    min_w = min(1700, int(TEXT_WIDTH_TWIPS / n * 0.75))               # a label column never narrower than 3 cm (less in wide tables)
    widths = [max(int(TEXT_WIDTH_TWIPS * l / total), min_w) for l in lens]
    scale = TEXT_WIDTH_TWIPS / sum(widths)
    widths = [int(w * scale) for w in widths]
    widths[-1] += TEXT_WIDTH_TWIPS - sum(widths)
    return widen_for_longest_word(widths, header, rows, n, min_w)


def fit_table(tbl, n_cols, n_rows):
    """Give a template table n_cols columns and n_rows rows (header + data), cloning its own header and first data row."""
    trs = tbl.findall(qn("w:tr"))
    head, proto = trs[0], trs[1] if len(trs) > 1 else copy.deepcopy(trs[0])
    for tr in trs[2:]:
        tbl.remove(tr)
    if proto.getparent() is None:
        head.addnext(proto)
    for tr in (head, proto):
        tcs = tr.findall(qn("w:tc"))
        while len(tcs) < n_cols:
            new = copy.deepcopy(tcs[-1])
            tcs[-1].addnext(new)
            tcs = tr.findall(qn("w:tc"))
        for tc in tcs[n_cols:]:
            tr.remove(tc)
    for _ in range(max(0, n_rows - 2)):
        new = copy.deepcopy(proto)
        strip_ids(new)
        tbl.findall(qn("w:tr"))[-1].addnext(new)
    if n_rows == 1:
        tbl.remove(proto)
    grid = tbl.find(qn("w:tblGrid"))
    for g in list(grid):
        grid.remove(g)
    for _ in range(n_cols):
        grid.append(OxmlElement("w:gridCol"))
    tblpr = tbl.find(qn("w:tblPr"))
    tw = tblpr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW")
        tblpr.append(tw)
    tw.set(qn("w:w"), str(TEXT_WIDTH_TWIPS))
    tw.set(qn("w:type"), "dxa")
    if tblpr.find(qn("w:tblLayout")) is None:
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblpr.append(lay)
    return tbl


def fill_table(tbl, header, rows):
    n = max(len(header), max((len(r) for r in rows), default=0))
    header = header + [""] * (n - len(header))
    rows = [r + [""] * (n - len(r)) for r in rows]
    fit_table(tbl, n, 1 + len(rows))
    widths = column_widths(header, rows, n)
    for g, w in zip(tbl.find(qn("w:tblGrid")).findall(qn("w:gridCol")), widths):
        g.set(qn("w:w"), str(w))
    trs = tbl.findall(qn("w:tr"))
    for tc, text, w in zip(trs[0].findall(qn("w:tc")), header, widths):
        cell_set(tc, inline_runs(text), bold_all=True)
        set_tc_width(tc, w)
    for tr, vals in zip(trs[1:], rows):
        for tc, text, w in zip(tr.findall(qn("w:tc")), vals, widths):
            cell_set(tc, inline_runs(text))
            set_tc_width(tc, w)
    for i, tr in enumerate(trs):                                     # the header row repeats on every page; a short row never splits over a page break
        trpr = tr.find(qn("w:trPr"))
        if trpr is None:
            trpr = OxmlElement("w:trPr")
            ex = tr.find(qn("w:tblPrEx"))
            (ex.addnext(trpr) if ex is not None else tr.insert(0, trpr))
        short = i == 0 or row_lines(rows[i - 1], widths) <= CANT_SPLIT_MAX_LINES
        cs = trpr.find(qn("w:cantSplit"))                            # a prototype row cloned from a filled table may carry the flag already
        if short and cs is None:
            trpr.append(OxmlElement("w:cantSplit"))
        elif not short and cs is not None:
            trpr.remove(cs)
        if i == 0 and trpr.find(qn("w:tblHeader")) is None:
            trpr.append(OxmlElement("w:tblHeader"))
    return tbl


CANT_SPLIT_MAX_LINES = 6                                             # rows up to this many estimated lines never split over a page break; taller text rows may


def row_lines(vals, widths):
    """Estimated wrapped lines of a row's tallest cell: TABLE_PT-point text at about half an em per character, 0.2 cm of cell padding."""
    n = 1
    for text, w in zip(vals, widths):
        cpl = max(4, int((w - 230) / (TABLE_PT * 20 * 0.5)))
        n = max(n, sum(max(1, -(-len(seg) // cpl)) for seg in plain(text).split("\n")))
    return n


def caption_paragraph(doc, after_el, label, body):
    """'Table n. ' / 'Figure n. ' as a literal label (never re-numbered) + the caption body with its tokens and citations resolved."""
    par = new_paragraph(doc, after_el, "Caption")
    add_runs(par, [(label, False)] + inline_runs(body))
    par.paragraph_format.keep_with_next = True
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(3)
    return par


def figure_paragraphs(doc, after_el, path, label, caption, width=None):
    from PIL import Image
    p = ROOT / path
    with Image.open(p) as im:
        w_px, h_px = im.size
    width_cm = min(TEXT_WIDTH_CM, width if width else FIG_WIDTH_BY_PATH.get(path.replace("\\", "/"), FIG_WIDTH_CM))
    if width_cm * h_px / w_px > MAX_FIG_HEIGHT_CM:
        width_cm = MAX_FIG_HEIGHT_CM * w_px / h_px
    par = new_paragraph(doc, after_el)
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(2)
    par.add_run().add_picture(str(p), width=Cm(width_cm))
    cap = new_paragraph(doc, par._p, "Caption")
    add_runs(cap, [(label, False)] + inline_runs(caption))
    cap.paragraph_format.space_after = Pt(8)
    return cap._p


def code_paragraphs(doc, after_el, text):
    cur = after_el
    for i, line in enumerate(text.split("\n")):
        par = new_paragraph(doc, cur)
        add_runs(par, [(line, False)], size=9, font="Consolas")
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.space_before = Pt(0)
        par.paragraph_format.left_indent = Cm(0.75)
        par.paragraph_format.line_spacing = 1.0
        cur = par._p
    last = para_of(cur, doc)
    last.paragraph_format.space_after = Pt(6)
    return cur


# ---------------------------------------------------------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------------------------------------------------------
def body_children(doc):
    return list(doc.element.body.iterchildren())


def paragraphs_by_text(doc, pred):
    return [p for p in doc.paragraphs if pred(p.text)]


def find_one(doc, pred, what):
    hits = paragraphs_by_text(doc, pred)
    if len(hits) != 1:
        raise SystemExit(f"template paragraph {what!r} found {len(hits)} times (the template changed?)")
    return hits[0]


def heading_for(doc, section):
    if section == "abstract":
        return find_one(doc, lambda t: t.startswith("Abstract"), "Abstract heading")
    m = re.fullmatch(r"(\d+)-(\d+)", section)
    if m:
        prefix = f"{m.group(1)}.{m.group(2)} "
        return find_one(doc, lambda t: t.startswith(prefix), prefix)
    m = re.fullmatch(r"appendix-([a-e])", section)
    prefix = f"Appendix {m.group(1).upper()}."
    return find_one(doc, lambda t: t.startswith(prefix), prefix)


def apply_page_setup(doc):
    for sec in doc.sections:
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    rpr = normal.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(5)
    # the cover banner is wider than the A4 text column: scale it to the text width
    for ext in doc.element.body.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent"):
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        if cx > Cm(TEXT_WIDTH_CM):
            ncx = int(Cm(TEXT_WIDTH_CM))
            ncy = int(cy * ncx / cx)
            ext.set("cx", str(ncx))
            ext.set("cy", str(ncy))
            for e in doc.element.body.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}ext"):
                if e.get("cx") == str(cx):
                    e.set("cx", str(ncx))
                    e.set("cy", str(ncy))
    # a page number beside the template's footer text
    for sec in doc.sections:
        fp = sec.footer.paragraphs[0]
        if "PAGE" not in fp._p.xml:
            fp.add_run(" | Page ")
            for instr in (" PAGE ", None, " NUMPAGES "):
                if instr is None:
                    fp.add_run(" of ")
                    continue
                fld = OxmlElement("w:fldSimple")
                fld.set(qn("w:instr"), instr)
                r = OxmlElement("w:r")
                t = OxmlElement("w:t")
                t.text = "1"
                r.append(t)
                fld.append(r)
                fp._p.append(fld)


def fill_cover(doc):
    title = CFG["title"]
    cut = title.find(" against ")      # the first line must stay inside the 16 cm column: a split at " on " left the word "models"
                                        # alone on a line of the cover (render check of 29 October 2026, LibreOffice and Word alike)
    line1, line2 = (title[:cut], title[cut + 1:]) if cut > 0 else (title, "")
    lines = paragraphs_by_text(doc, lambda t: t.startswith("PROJECT / RESEARCH TITLE:") or re.fullmatch(r"_{10,}", t.strip()) is not None)
    assert len(lines) == 2, "the template's two title lines were not found"
    set_par_text(lines[0], [("PROJECT / RESEARCH TITLE: ", False), (line1, True)], plain_look=False)
    if line2:
        set_par_text(lines[1], [(line2, True)], plain_look=False)
        lines[1].paragraph_format.left_indent = Cm(5.2)
    else:
        delete_el(lines[1]._p)
    team_h = find_one(doc, lambda t: t.strip() == "Project Team", "Project Team")
    set_par_text(team_h, [(f"Project Team ({num_text('group_number')})", False)], plain_look=False)
    t1 = doc.tables[TEMPLATE_TABLE_INDEX["T1"]]
    members = CFG["members"]
    for i, row in enumerate(t1.rows[1:]):
        if i < len(members):
            vals = [f"{i + 1}.", members[i]["name"], members[i]["student_id"], members[i]["role"]]
        else:
            vals = [f"{i + 1}.", "–", "–", "–"]
        for tc, v in zip(row._tr.findall(qn("w:tc")), vals):
            cell_set(tc, [(v, False)], size=11)
    for tc in t1.rows[0]._tr.findall(qn("w:tc")):
        p = tc.findall(qn("w:p"))[0]
        cell_set(tc, [(Paragraph(p, None).text, False)], size=11, bold_all=True)
    set_par_text(find_one(doc, lambda t: t.startswith("Group Leader (GL):"), "GL"), [("Group Leader (GL): " + CFG["group_leader"], False)], plain_look=False)
    set_par_text(find_one(doc, lambda t: t.startswith("Submission Date:"), "Submission"), [("Submission Date: " + SUBMISSION_DATE, False)], plain_look=False)
    set_par_text(find_one(doc, lambda t: t.startswith("Final Presentation:"), "Presentation"), [("Final Presentation: " + CFG["final_presentation"], False)], plain_look=False)


def delete_requirements_block(doc):
    h = find_one(doc, lambda t: t.startswith("Document Requirements and Reporting Principles"), "requirements heading")
    nxt = h._p.getnext()
    delete_el(h._p)
    while nxt is not None:
        after = nxt.getnext()
        tag = nxt.tag.split("}")[1]
        if tag == "p" and para_of(nxt, doc).text.startswith("Instructions:"):
            delete_el(nxt)
        elif tag == "tbl":
            delete_el(nxt)
            break
        else:
            break
        nxt = after


def t3_rows():
    """T3 Project Information: every number through the registry."""
    members = "; ".join(f"{m['name']} ({'GL; ' if i == 0 else ''}{m['role'].replace(', group leader', '')})" for i, m in enumerate(CFG["members"]))
    # the short names (results/data_flow.json datasets[*].short; full names, access and licences in Table 5 and Appendix B) grouped by role
    # (quality audit of 30 Oct 2026, should-fix 25: the eleven '(login: ...)' texts left this cell)
    sh = lambda keys: ", ".join("{{num:ds_%s_short}}" % k for k in keys)
    return [
        ["Project title", "{{num:project_title}}"],
        ["Research area", "AI for forecasting in hydrology: river-flow and flood-onset warning on small basins, with remote-sensing and GIS products as inputs and checks"],
        ["Duration", f"{{{{num:date_data_audit}}}} (data audit and choice of the basins) to {SUBMISSION_DATE} (hand-in); final presentation {{{{num:final_presentation}}}}"],
        ["Project leader and members", members],
        ["Primary datasets", "{{num:n_products}} public products ({{num:gb_read}} read, {{num:gb_kept}} kept under data/; Table T5): " + sh(["camels_us"])
         + " (all models); " + sh(["aorc", "gefs", "era5", "merra2", "imerg"]) + " and the hourly reduction of " + sh(["usgs_iv"])
         + " (inputs of single experiments); " + sh(["stage4", "elevation_tiles", "nlcd", "hurdat2"]) + " (checks and maps). Access and licences: Table T5 and Appendix B"],
        ["Implementation environment", "{{num:mac_machine}}; Python {{num:mac_python}}, torch {{num:torch_version}} on {{num:torch_threads}} CPU threads, numpy {{num:numpy_version}} "
         "(Appendix A); documents, stored-forecast analyses and the E4 experiments on a second machine, {{num:windows_pc}}; coding assistant: Claude Code (Section 8.3)"],
        ["Completed outputs", "The shared engine ({{num:count_code_py_files}} top-level scripts, {{num:fair_test_n_checks}} fairness checks); {{num:stored_prediction_sets}} stored forecast sets and "
         "{{num:stored_saved_networks}} saved networks as of {{num:date_data_record_written}}, {{num:count_preds_npz}} files and {{num:count_models_pt}} networks on {{num:seed_extension_date}} "
         "after E4b, E4e and the five-seed extension of scripts {{num:seed_extension_scripts}}; {{num:count_figures_png}} figures; {{num:review_records}} independent AI review records of the experiments and {{num:document_review_passes}} critic passes "
         "over the documents; this report, the final deck and Flood_Report.pdf; GIS layer export {{num:gis_layers_exported}}"],
    ]


def toc_entries(doc, toc_heading_el):
    """(level, text, paragraph) of every Heading 1 / Heading 2 after the Table of Contents heading."""
    out, seen = [], False
    for ch in doc.element.body.iterchildren():
        if ch is toc_heading_el:
            seen = True
            continue
        if seen and ch.tag.endswith("}p"):
            p = para_of(ch, doc)
            if p.style.name in ("Heading 1", "Heading 2") and p.text.strip():
                out.append((1 if p.style.name == "Heading 1" else 2, p.text.strip(), p))
    return out


def fld_run(kind):
    r = OxmlElement("w:r")
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), kind)
    r.append(fc)
    return r


def instr_run(text):
    r = OxmlElement("w:r")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = text
    r.append(it)
    return r


def clear_toc(doc):
    """Drop the template's TOC lines (every Normal paragraph after the TOC heading up to the page-break paragraph before the Abstract)."""
    h = find_one(doc, lambda t: t.strip() == "Table of Contents", "TOC heading")
    nxt = h._p.getnext()
    while nxt is not None and nxt.tag.endswith("}p"):
        p = para_of(nxt, doc)
        after = nxt.getnext()
        if p.style.name == "Normal" and p.text.strip() and not nxt.xpath('.//w:br[@w:type="page"]'):
            delete_el(nxt)
        elif p.style.name != "Normal":
            break
        nxt = after
    return h


def build_toc(doc, pages=None):
    """Plain paragraphs (indent per level, dotted tab to the page number) wrapped in a TOC field Word can update; returns the paragraphs."""
    h = find_one(doc, lambda t: t.strip() == "Table of Contents", "TOC heading")
    entries = toc_entries(doc, h._p)
    cur, paras = h._p, []
    for i, (level, text, _) in enumerate(entries):
        par = new_paragraph(doc, cur)
        par.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_WIDTH_CM), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        par.paragraph_format.space_after = Pt(0)                      # single-spaced, no gap: the contents fit on one page (critic 2 of 29 Oct, item 1)
        par.paragraph_format.space_before = Pt(0)
        par.paragraph_format.line_spacing = 1.0
        if level == 2:
            par.paragraph_format.left_indent = Cm(0.75)
        r = par.add_run(text)
        if level == 1:
            r.bold = True
        par.add_run("\t")
        par.add_run(str(pages[i]) if pages and pages[i] else "")
        cur = par._p
        paras.append(par)
    if paras:
        first = paras[0]._p
        at = 1 if first.find(qn("w:pPr")) is not None else 0
        for el in reversed([fld_run("begin"), instr_run(' TOC \\o "1-2" \\h \\z \\u '), fld_run("separate")]):
            first.insert(at, el)
        paras[-1]._p.append(fld_run("end"))
    return entries, paras


def set_toc_pages(paras, pages):
    for par, pg in zip(paras, pages):
        runs = [r for r in par.runs if r.text not in ("\t", "")] or par.runs
        last = par.runs[-1]
        if last._r.find(qn("w:fldChar")) is not None:          # the field end sits after the number run
            last = par.runs[-2]
        last.text = str(pg) if pg else ""


def build(doc):
    for s in SECTION_IDS:
        if not (SECTIONS_DIR / f"{s}.md").exists():
            raise SystemExit(f"missing section file {s}.md")
    blocks = {s: parse_section(s) for s in SECTION_IDS}

    # ---- pass A: citation order, table and figure numbers in document order
    B.table_ids = ["T3"]
    for s in SECTION_IDS:
        for b in blocks[s]:
            if b.kind == "p":
                collect_cites(b.text)
            elif b.kind == "table":
                if b.tid:
                    B.table_ids.append(b.tid)
                for cell in b.header + [c for r in b.rows for c in r]:
                    collect_cites(cell)
                collect_cites(b.caption)
            elif b.kind == "figure":
                m = re.match(r"Figure (\d+)\.\s*", b.caption)
                B.figure_ids.append(int(m.group(1)) if m else None)
                collect_cites(b.caption)
    for row in t3_rows():
        collect_cites(row[1])
    B.numbering = REFS.numbering(B.cite_order)
    TABLE_NO.update({tid: i + 1 for i, tid in enumerate(B.table_ids)})
    FIGURE_NO.update({fid: i + 1 for i, fid in enumerate(B.figure_ids) if fid is not None})

    # ---- template housekeeping
    apply_page_setup(doc)
    template_tables = {tid: doc.tables[i]._tbl for tid, i in TEMPLATE_TABLE_INDEX.items()}
    fill_cover(doc)
    delete_requirements_block(doc)
    clear_toc(doc)                                                 # the template's contents lines go first, so heading lookups are unique
    abstract_h = find_one(doc, lambda t: t.startswith("Abstract"), "Abstract heading")
    set_par_text(abstract_h, [("Abstract and Keywords", False)], plain_look=False)
    res_h = find_one(doc, lambda t: t.strip() == "10.3 Resource utilization", "10.3")
    set_par_text(res_h, [("10.3 Resource utilization", False)], plain_look=False)
    for letter in "ABCDE":                                         # the appendix title lines become sub-headings (TOC, structure)
        p = find_one(doc, lambda t, L=letter: t.startswith(f"Appendix {L}."), f"Appendix {letter}")
        p.style = doc.styles["Heading 2"]

    # ---- T3 Project Information (caption + rows)
    t3 = template_tables["T3"]
    fill_table(t3, ["Field", "Details"], t3_rows())
    caption_paragraph(doc, t3.getprevious(), f"Table {TABLE_NO['T3']}. ", "Project information (every number is read from results/*.json through the registry)")
    B.n_tables += 1

    # ---- the sections
    decl = find_one(doc, lambda t: t.startswith("Declaration:"), "Declaration")
    for s in SECTION_IDS:
        heading = heading_for(doc, s)
        cur = heading._p
        tail_after_declaration = []
        bl = blocks[s]
        if s == "appendix-e":
            idx = next((i for i, b in enumerate(bl) if b.kind == "p" and b.text.startswith("Declaration:")), None)
            if idx is not None:
                tail_after_declaration = bl[idx + 1:]
                bl = bl[:idx]
        # layout rule: a figure that opens a section follows the section's first paragraph, so the heading, the sentence that introduces
        # the figure and the figure stay together and a full-page drawing does not drag the headings onto a page of their own
        if len(bl) >= 2 and bl[0].kind == "figure" and bl[1].kind == "p":
            bl = [bl[1], bl[0]] + bl[2:]
        for b in bl:
            if b.kind == "p":
                if b.text.startswith("**Keywords:**"):
                    kw = find_one(doc, lambda t: t.startswith("Keywords:"), "Keywords")
                    set_par_text(kw, [("Keywords: ", True), (plain(b.text[len("**Keywords:**"):]).strip(), False)])
                    continue
                m = TOKEN.fullmatch(b.text.strip())
                if m and num_text(m.group(1)) and "\n" in num_text(m.group(1)):
                    cur = code_paragraphs(doc, cur, num_text(m.group(1)))
                    continue
                par = new_paragraph(doc, cur)
                add_runs(par, inline_runs(b.text))
                cur = par._p
            elif b.kind == "table":
                no = TABLE_NO.get(b.tid)
                cap = caption_paragraph(doc, cur, f"Table {no}. " if no else "", b.caption)
                cur = cap._p
                if b.tid in template_tables and b.tid != "T2":
                    tbl = template_tables[b.tid]
                    delete_el(tbl)
                else:
                    tbl = copy.deepcopy(template_tables["T7"])
                    strip_ids(tbl)
                cur.addnext(tbl)
                fill_table(tbl, b.header, b.rows)
                cur = tbl
                B.n_tables += 1
                # a small gap after the table
                gap = new_paragraph(doc, cur)
                gap.paragraph_format.space_after = Pt(2)
                gap.paragraph_format.line_spacing = 1.0
                for r in gap.runs:
                    delete_el(r._r)
                rpr = gap._p.get_or_add_pPr()
                mark = OxmlElement("w:rPr")
                sz = OxmlElement("w:sz")
                sz.set(qn("w:val"), "8")
                mark.append(sz)
                rpr.append(mark)
                cur = gap._p
            elif b.kind == "figure":
                m = re.match(r"Figure (\d+)\.\s*(.*)", b.caption, re.S)
                no = FIGURE_NO.get(int(m.group(1))) if m else None
                cur = figure_paragraphs(doc, cur, b.path, f"Figure {no}. " if no else "", m.group(2) if m else b.caption, width=getattr(b, "width", None))
                B.n_figures += 1
        if tail_after_declaration:
            cur = decl._p
            for b in tail_after_declaration:
                par = new_paragraph(doc, cur)
                add_runs(par, inline_runs(b.text))
                par.paragraph_format.space_before = Pt(10)
                cur = par._p

    # ---- section 11 References
    ref_h = find_one(doc, lambda t: t.strip() == "11. References", "References heading")
    ref_h.paragraph_format.page_break_before = True                   # the References never share a page with Table 12 (critic 2 of 29 Oct, item 1)
    cur = ref_h._p
    for key, n in B.numbering.items():
        par = new_paragraph(doc, cur)
        for text, italic in REFS.runs(key, n):
            r = par.add_run(text)
            if italic:
                r.italic = True
        par.paragraph_format.left_indent = Cm(0.9)
        par.paragraph_format.first_line_indent = Cm(-0.9)
        par.paragraph_format.space_after = Pt(3)
        cur = par._p
    ph = paragraphs_by_text(doc, lambda t: t.startswith("[1] Author(s)"))
    for p in ph:
        delete_el(p._p)

    # ---- the professor's instructions and placeholders
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if t.startswith("Instructions:") or (t.startswith("[") and t.endswith("]")):
            delete_el(p._p)
    if B.unknown_refs:
        B.notes.append("unknown reference keys: " + ", ".join(B.unknown_refs))
    entries, toc_paras = build_toc(doc)
    doc.core_properties.title = CFG["title"]
    doc.core_properties.author = "; ".join(m["name"] for m in CFG["members"])
    doc.core_properties.subject = CFG["course"]
    return entries, toc_paras


# ---------------------------------------------------------------------------------------------------------------------------
# The HTML twin (from the saved .docx, so the twins cannot differ) and the PDF
# ---------------------------------------------------------------------------------------------------------------------------
CSS = """
@page{size:A4;margin:25mm 25mm 25mm 25mm}
html,body{margin:0;padding:0}
body{background:#fff;color:#000;font:11pt/1.15 'Times New Roman',Times,serif}
.doc p{margin:0 0 5pt;orphans:2;widows:2}
.doc h1,.doc h2,.doc .title{font-family:Calibri,'Segoe UI',Arial,sans-serif;font-weight:bold;line-height:1.15;margin:0 0 5pt;break-after:avoid}
.doc h1{font-size:14pt;color:#173B67;margin-top:9pt}
.doc h2{font-size:11.5pt;color:#206A88;margin-top:9pt}
.doc .title{font-size:21pt;color:#173B67;text-align:center;border-bottom:1pt solid #4F81BD;padding-bottom:4pt;margin-top:9pt;line-height:1}
.doc .caption{font-size:9pt;font-weight:bold;color:#4F81BD;line-height:1.15;break-inside:avoid}
.doc img{max-width:100%;height:auto}
.doc table{border-collapse:collapse;margin:0 auto;table-layout:fixed;width:453.6pt;font-size:9pt;break-inside:auto}
.doc table.ls{border-top:1pt solid #4F81BD;border-bottom:1pt solid #4F81BD;color:#365F91;margin:0 auto 4pt}
.doc table.ls td{border:0.5pt solid #A9BFDC;padding:2pt 4pt;vertical-align:top;text-align:left;line-height:1.15;overflow-wrap:anywhere}
.doc table.ls td.c1{font-weight:bold}
.doc table.ls tr.h td{font-weight:bold;border-bottom:1pt solid #4F81BD}
.doc table.ls tr:nth-child(even) td{background:#D3DFEE}
.doc tr{break-inside:avoid}
.doc .pb{break-before:page}
.doc .toc{display:flex;align-items:baseline;margin:0 0 2pt}
.doc .toc .t{white-space:nowrap}
.doc .toc .dots{flex:1;border-bottom:1px dotted #444;margin:0 4pt 3pt}
.doc .toc .n{white-space:nowrap}
.doc .code{font:9pt/1.2 Consolas,'Courier New',monospace;margin:0 0 0 0.75cm}
.doc .ref{overflow-wrap:anywhere}
.doc .kwn{break-after:avoid}
.doc .fig{text-align:center;break-inside:avoid;break-after:avoid;margin:6pt 0 2pt}
"""
BASE = {"Normal": dict(size=11, bold=False, italic=False, color=None, font=None),
        "Heading 1": dict(size=14, bold=True, italic=False, color="173B67", font="Calibri"),
        "Heading 2": dict(size=11.5, bold=True, italic=False, color="206A88", font="Calibri"),
        "Title": dict(size=21, bold=True, italic=False, color="173B67", font="Calibri"),
        "Caption": dict(size=9, bold=True, italic=False, color="4F81BD", font=None)}


def to_html(path):
    d = Document(str(path))

    def val(el, tag, name="val"):
        e = None if el is None else el.find(W + tag)
        return None if e is None else e.get(W + name)

    def style_of(rpr, base):
        s = dict(base)
        if rpr is not None:
            for key, tag in (("bold", "b"), ("italic", "i")):
                e = rpr.find(W + tag)
                if e is not None:
                    s[key] = e.get(W + "val") not in ("0", "false")
            c = val(rpr, "color")
            if c and c != "auto":
                s["color"] = c
            sz = val(rpr, "sz")
            if sz:
                s["size"] = int(sz) / 2
            rf = rpr.find(W + "rFonts")
            if rf is not None and rf.get(W + "ascii"):
                s["font"] = rf.get(W + "ascii")
        return s

    def css(s):
        out = f"font-size:{s['size']:g}pt;"
        if s["bold"]:
            out += "font-weight:bold;"
        if s["italic"]:
            out += "font-style:italic;"
        if s["color"]:
            out += f"color:#{s['color']};"
        if s["font"] and s["font"] != "Times New Roman":
            out += f"font-family:'{s['font']}',monospace;" if s["font"] == "Consolas" else f"font-family:'{s['font']}',sans-serif;"
        return out

    def run_parts(p_el, base):
        """[(kind, html_or_text, style)] kind in text/tab."""
        parts = []
        for r in p_el.iter(W + "r"):
            if r.getparent().tag not in (W + "p", W + "hyperlink", W + "fldSimple"):
                continue
            s = style_of(r.find(W + "rPr"), base)
            for ch in r:
                if ch.tag == W + "t":
                    if ch.text:
                        parts.append(("text", f'<span style="{css(s)}">{_html.escape(ch.text)}</span>', s))
                elif ch.tag == W + "tab":
                    parts.append(("tab", "", s))
        return parts

    def pstyle(p_el, sty, first):
        ppr = p_el.find(W + "pPr")
        sp = None if ppr is None else ppr.find(W + "spacing")
        before, after = {"Heading 1": (9, 5), "Heading 2": (9, 5), "Title": (9, 5)}.get(sty, (0, 5))
        line = 1.15 if sty not in ("Caption", "Title") else 1.0
        if sp is not None:
            if sp.get(W + "before") is not None:
                before = int(sp.get(W + "before")) / 20
            if sp.get(W + "after") is not None:
                after = int(sp.get(W + "after")) / 20
            if sp.get(W + "line") is not None and sp.get(W + "lineRule", "auto") == "auto":
                line = int(sp.get(W + "line")) / 240
        jc = val(ppr, "jc")
        align = {"center": "center", "both": "justify", "right": "right"}.get(jc, "left")
        ind = None if ppr is None else ppr.find(W + "ind")
        extra = ""
        if ind is not None:
            if ind.get(W + "left"):
                extra += f"margin-left:{int(ind.get(W + 'left')) / 20:g}pt;"
            if ind.get(W + "hanging"):
                extra += f"text-indent:-{int(ind.get(W + 'hanging')) / 20:g}pt;"
            elif ind.get(W + "firstLine"):
                extra += f"text-indent:{int(ind.get(W + 'firstLine')) / 20:g}pt;"
        if ppr is not None and ppr.find(W + "keepNext") is not None:
            extra += "break-after:avoid;"
        mark = val(None if ppr is None else ppr.find(W + "rPr"), "sz")           # the paragraph mark's size sets the height of an empty line
        size = first["size"] if first else (int(mark) / 2 if mark else 11)
        return f"text-align:{align};margin:{before:g}pt 0 {after:g}pt;line-height:{line:.3f};font-size:{size:g}pt;{extra}"

    def table_html(t):
        tbl = t._tbl
        grid = [int(g.get(W + "w")) / 20 for g in tbl.tblGrid.findall(W + "gridCol")]
        out = [f'<table class="ls" style="width:{sum(grid):.1f}pt"><colgroup>' + "".join(f'<col style="width:{g:.1f}pt">' for g in grid) + "</colgroup>"]
        for ri, tr in enumerate(tbl.findall(W + "tr")):
            out.append('<tr class="h">' if ri == 0 else "<tr>")
            for ci, tc in enumerate(tr.findall(W + "tc")):
                body = []
                for p_el in tc.findall(W + "p"):
                    parts = run_parts(p_el, BASE["Normal"])
                    h = "".join(x[1] for x in parts if x[0] == "text")
                    body.append(f"<div>{h or '&nbsp;'}</div>")
                out.append(f'<td class="{"c1" if ci == 0 else ""}">{"".join(body)}</td>')
            out.append("</tr>")
        out.append("</table>")
        return "".join(out)

    out = ['<div class="doc">']
    for ch in d.element.body.iterchildren():
        tag = ch.tag.split("}")[1]
        if tag == "tbl":
            out.append(table_html(Table(ch, d)))
            continue
        if tag != "p":
            continue
        p = Paragraph(ch, d)
        sty = p.style.name
        base = BASE.get(sty, BASE["Normal"])
        if ch.xpath('.//w:br[@w:type="page"]'):
            out.append('<div class="pb"></div>')
            if not p.text.strip():
                continue
        blip = ch.xpath(".//a:blip")
        if blip:
            blob = d.part.related_parts[blip[0].get(R_NS + "embed")].blob
            cx = int(ch.xpath(".//wp:extent")[0].get("cx")) / 12700
            out.append(f'<p class="fig" style="{pstyle(ch, sty, None)}"><img alt="figure" style="width:{cx:.1f}pt" '
                       f'src="data:image/png;base64,{base64.b64encode(blob).decode()}"></p>')
            continue
        parts = run_parts(ch, base)
        first = next((s for k, _, s in parts if k == "text"), None)
        texts = [x for x in parts if x[0] == "text"]
        if sty == "Title":
            out.append(f'<p class="title">{"".join(h for _, h, _ in texts)}</p>')
        elif sty in ("Heading 1", "Heading 2"):
            n = sty[-1]
            ppr = ch.find(W + "pPr")
            jc = val(ppr, "jc")
            al = ' style="text-align:center"' if jc == "center" else ""
            out.append(f"<h{n}{al}>{_html.escape(p.text)}</h{n}>")
        elif sty == "Caption":
            out.append(f'<p class="caption" style="{pstyle(ch, sty, first)}">{"".join(h for _, h, _ in texts)}</p>')
        elif any(k == "tab" for k, _, _ in parts) and ch.find(W + "pPr") is not None and ch.find(W + "pPr").find(W + "tabs") is not None:
            left = "".join(h for k, h, _ in parts[: next(i for i, x in enumerate(parts) if x[0] == "tab")] if k == "text")
            right = "".join(h for k, h, _ in parts[next(i for i, x in enumerate(parts) if x[0] == "tab") + 1:] if k == "text")
            ind = ch.find(W + "pPr").find(W + "ind")
            ml = f"margin-left:{int(ind.get(W + 'left')) / 20:g}pt;" if ind is not None and ind.get(W + "left") else ""
            out.append(f'<p class="toc" style="{ml}"><span class="t">{left}</span><span class="dots"></span><span class="n">{right}</span></p>')
        elif first and first.get("font") == "Consolas":
            out.append(f'<p class="code">{"".join(h for _, h, _ in texts) or "&nbsp;"}</p>')
        else:
            cls = ' class="ref"' if re.match(r"\[\d+\] ", p.text) else ""
            out.append(f'<p{cls} style="{pstyle(ch, sty, first)}">{"".join(h for _, h, _ in texts) or "&nbsp;"}</p>')
    out.append("</div>")
    title = _html.escape(CFG["title"])
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{"".join(out)}</body></html>'


def find_browser():
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if Path(p).exists():
            return p
    return next((w for w in map(shutil.which, ("msedge", "google-chrome", "chromium", "chrome")) if w), None)


def print_pdf(html_path, pdf_path):
    browser = find_browser()
    if not browser:
        print("no Chromium browser found: HTML only", html_path)
        return False
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", Path(html_path).as_uri()],
                   check=True, timeout=300, capture_output=True)
    return True


def stamp_footer(pdf_path, footer_text):
    """The template's footer and 'Page n of N' on every page (Chromium prints no running footer), with PyMuPDF."""
    import fitz
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    for i, page in enumerate(doc, 1):
        w, h = page.rect.width, page.rect.height
        y = h - 36
        left = footer_text
        right = f"Page {i} of {n}"
        page.insert_text((72, y), left, fontsize=8, fontname="tiro", color=(0.3, 0.3, 0.3))
        tw = fitz.get_text_length(right, fontname="tiro", fontsize=8)
        page.insert_text((w - 72 - tw, y), right, fontsize=8, fontname="tiro", color=(0.3, 0.3, 0.3))
    tmp = str(pdf_path) + ".tmp"
    doc.save(tmp, garbage=3, deflate=True)
    doc.close()
    Path(tmp).replace(pdf_path)
    return n


def heading_pages(pdf_path, entries):
    """The 1-based page of each TOC entry's heading in the printed PDF. Headings are the only lines set in the heading font (Calibri),
    so the contents lines (Times New Roman) cannot be mistaken for them; each heading is searched after the previous one."""
    import fitz
    doc = fitz.open(str(pdf_path))
    lines = []                                                       # (page, text) of every line set in the heading font
    for pno, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if spans and all("calibri" in s["font"].lower() for s in spans):
                    lines.append((pno, re.sub(r"\s+", " ", "".join(s["text"] for s in spans)).strip()))
    doc.close()
    pages, start = [], 0
    for level, text, _ in entries:
        want = re.sub(r"\s+", " ", text).strip()
        found = None
        for i in range(start, len(lines)):
            pno, lt = lines[i]
            if lt == want or lt.startswith(want[:40]) or (len(lt) >= 20 and want.startswith(lt)):
                found, start = pno, i + 1
                break
        pages.append(found)
    return pages


def soffice():
    """LibreOffice, for a PDF rendered from the .docx itself (installed on the Windows PC on 2026-10-29); None when absent."""
    for c in (r"C:\Program Files\LibreOffice\program\soffice.exe", "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if Path(c).exists():
            return c
    return shutil.which("soffice")


def write_pdf(docx_path, pdf_path, footer_text):
    office = soffice()
    if office:                                                       # the .docx rendered by a word processor: the page count the reader will see
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run([office, "--headless", "--convert-to", "pdf", "--outdir", folder, str(docx_path)], check=True, timeout=600, capture_output=True)
            out = Path(folder) / (Path(docx_path).stem + ".pdf")
            if out.exists():
                shutil.copyfile(out, pdf_path)
                import fitz
                with fitz.open(str(pdf_path)) as pdf:                   # the page count, as stamp_footer returns it on the Chromium path
                    return pdf.page_count
    html = to_html(docx_path)
    with tempfile.TemporaryDirectory() as folder:
        src = Path(folder) / "report.html"
        src.write_text(html, encoding="utf-8")
        if not print_pdf(src, pdf_path):
            return None
    return stamp_footer(pdf_path, footer_text)


# ---------------------------------------------------------------------------------------------------------------------------
def count_words(docx_path):
    """Words of the body prose (Abstract to the References heading, paragraphs outside tables, captions left out), of the tables, of the appendices."""
    d = Document(str(docx_path))
    zone, body, tables, appendix, caps = "front", 0, 0, 0, 0
    for ch in d.element.body.iterchildren():
        tag = ch.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(ch, d)
            t = p.text.strip()
            if p.style.name == "Heading 1":
                if t.startswith("Abstract"):
                    zone = "body"
                elif t.startswith("11. References"):
                    zone = "refs"
                elif t.startswith("Appendices"):
                    zone = "appendix"
            n = len(re.findall(r"\S+", t))
            if p.style.name == "Caption":
                caps += n
            elif zone == "body":
                body += n
            elif zone == "appendix":
                appendix += n
        elif tag == "tbl":
            n = sum(len(re.findall(r"\S+", c.text)) for r in Table(ch, d).rows for c in r.cells)
            if zone == "body":
                tables += n
            elif zone == "appendix":
                appendix += n
    return body, tables, caps, appendix


def wording_check(docx_path):
    """HANDOFF.md section 7: the hindsight wording stands beside every AORC onset share (the registry's e1b_onset_warned_with_aorc_rain_to_midnight:
    51% with three seeds, 52% with five since 30 October 2026). Returns the texts that break the rule."""
    d = Document(str(docx_path))
    hind = num_text("fw_hindsight") or "hindsight"
    share = num_text("e1b_onset_warned_with_aorc_rain_to_midnight") or "51%"
    bad = []
    texts = [p.text for p in d.paragraphs] + [c.text for t in d.tables for r in t.rows for c in r.cells]
    verdict_cell = re.compile(r"^(GO|CAUTION|NOT YET) \(\d+%, \d+%\)$")
    for t in texts:
        if verdict_cell.match(t.strip()):
            continue
        if re.search(r"\b" + re.escape(share), t) and hind not in t:
            bad.append(t.strip()[:110])
    return bad


def section_words(docx_path):
    """{heading: (prose words, table words)} for the body sections (Abstract to 10.3), captions left out."""
    d = Document(str(docx_path))
    cur, out, zone = None, {}, "front"
    for ch in d.element.body.iterchildren():
        tag = ch.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(ch, d)
            t = p.text.strip()
            if p.style.name == "Heading 1" and t.startswith("Abstract"):
                zone = "body"
            if p.style.name == "Heading 1" and t.startswith("11. References"):
                zone = "refs"
            if zone != "body":
                continue
            if p.style.name in ("Heading 1", "Heading 2"):
                cur = t
                out.setdefault(cur, [0, 0])
                continue
            if cur and p.style.name != "Caption":
                out[cur][0] += len(re.findall(r"\S+", t))
        elif tag == "tbl" and zone == "body" and cur:
            out[cur][1] += sum(len(re.findall(r"\S+", c.text)) for r in Table(ch, d).rows for c in r.cells)
    return out


def main():
    no_pdf = "--no-pdf" in sys.argv
    HANDIN.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE.exists():
        shutil.copyfile(SRC_TEMPLATE, TEMPLATE)
        print("copied the template to", TEMPLATE)
    doc = Document(str(TEMPLATE))
    footer_text = doc.sections[0].footer.paragraphs[0].text.strip()
    entries, toc_paras = build(doc)
    doc.save(str(OUT_DOCX))
    print("wrote", OUT_DOCX)
    report = {"pages": None, "refs_page": None, "appendix_page": None, "abstract_page": None, "toc_pages": None}
    if not no_pdf:
        n = write_pdf(OUT_DOCX, OUT_PDF, footer_text)
        if n:
            # the TOC page numbers come from the printed twin; then print again and check they still hold
            for attempt in range(3):
                pages = heading_pages(OUT_PDF, entries)
                set_toc_pages(toc_paras, pages)
                doc.save(str(OUT_DOCX))
                n = write_pdf(OUT_DOCX, OUT_PDF, footer_text)
                check = heading_pages(OUT_PDF, entries)
                if check == pages:
                    break
                B.notes.append(f"TOC pass {attempt + 1}: page numbers moved after insertion ({check} vs {pages})")
            report["toc_pages"] = list(zip([e[1] for e in entries], pages))
            report["pages"] = n
            for text, pg in report["toc_pages"]:
                if text.startswith("Abstract"):
                    report["abstract_page"] = pg
                if text.startswith("11. References"):
                    report["refs_page"] = pg
                if text == "Appendices":
                    report["appendix_page"] = pg
            print("wrote", OUT_PDF, f"({n} pages)")
    body, tables, caps, appendix = count_words(OUT_DOCX)
    wc = wording_check(OUT_DOCX)
    lines = [
        "# Build report: 3_Final_Project_Report_Flood.docx / .pdf",
        "",
        f"Built by `python -X utf8 code/build_report_final.py` (system Python 3.13, python-docx; no torch) in {ROOT}.",
        f"Template: {SRC_TEMPLATE} (read-only source) copied once to {TEMPLATE}. Submission date printed: {SUBMISSION_DATE} (constant SUBMISSION_DATE at the top of the builder).",
        "",
        "## Counts",
        "",
        f"- PDF pages: {report['pages']} in all; References heading on page {report['refs_page']}, Appendices on page {report['appendix_page']}, Abstract on page {report['abstract_page']}"
        + (f"; pages before the References ({report['refs_page'] - 1}) = cover, contents and sections 1-10; body pages Abstract to end of section 10 = {report['refs_page'] - report['abstract_page']}"
           if report["refs_page"] and report["abstract_page"] else ""),
        f"- Words: body prose (Abstract to section 10, outside tables) {body:,}; table text in the body {tables:,}; captions {caps:,}; appendices (prose and tables) {appendix:,}",
        f"- Figures: {B.n_figures}; tables: {B.n_tables} (T1 on the cover not counted); references: {len(B.numbering)}; citations in order of first appearance",
        f"- MISSING tokens: {len(B.missing)}" + (" -> " + ", ".join(sorted(B.missing)) if B.missing else ""),
        f"- Optional blocks dropped: {len(B.dropped_optional)}" + (" -> " + "; ".join(f"{s}: {p} ({k} missing)" for s, p, k in B.dropped_optional) if B.dropped_optional else ""),
        f"- Unknown reference keys: {len(B.unknown_refs)}" + (" -> " + ", ".join(B.unknown_refs) if B.unknown_refs else ""),
        f"- Fixed-wording check (hindsight wording beside every AORC onset share, {num_text('e1b_onset_warned_with_aorc_rain_to_midnight')}): {len(wc)} paragraph(s) or cell(s) without it"
        + ("".join(f"\n    - {t}" for t in wc) if wc else ""),
        "",
        "## Numbering",
        "",
        "Tables are numbered in order of appearance (the template's own numbering had a gap where the professor's requirements table was removed): "
        + ", ".join(f"{tid} -> Table {n}" for tid, n in TABLE_NO.items()) + ".",
        "Figures are numbered in order of appearance: " + ", ".join(f"writers' Figure {k} -> Figure {v}" for k, v in sorted(FIGURE_NO.items())) + ".",
        "The writers' cross-references ('Table T7', 'Figure 3') were rewritten to these numbers by the builder.",
        "",
        "## Table of contents (page numbers from the PDF twin)",
        "",
    ] + [f"- {text}: page {pg}" for text, pg in (report["toc_pages"] or [])] + [
        "",
        "## Words per section against the content plan (audits/agent6.md section 2; the builder cuts nothing)",
        "",
        "| Section | Prose words | Plan | Table words |",
        "|---|---|---|---|",
    ]
    sw = section_words(OUT_DOCX)
    tot_p, tot_plan, tot_t = 0, 0, 0
    for heading, (pw, tw) in sw.items():
        if pw == 0 and tw == 0:
            continue
        key = "Abstract" if heading.startswith("Abstract") else heading.split(" ")[0]
        plan = PLAN_WORDS.get(key)
        tot_p, tot_t = tot_p + pw, tot_t + tw
        tot_plan += plan or 0
        flag = "" if plan is None or pw <= plan * 1.15 else " **over**"
        lines.append(f"| {heading[:60]} | {pw:,}{flag} | {plan if plan is not None else '-'} | {tw:,} |")
    lines += [f"| **Total (Abstract to 10.3)** | **{tot_p:,}** | **{tot_plan:,}** | **{tot_t:,}** (plan: about 1,200) |", "",
              "The template allows 15-25 pages excluding references and appendices. If the body runs over, the content plan's trim order is 7.3, 2.2, 8.2, 5.2, 7.2 captions,",
              "and the dense tables (T4 literature, T5 data inventory, T6 scenarios, T8 objectives) are the largest single items.",
              "",
              "## Rebuild",
        "",
        "```",
        "cd <project root>",
        "python -X utf8 code/final_report/numbers.py          # the registry results/final_report_numbers.json from results/*.json",
        "python -X utf8 code/final_report/references.py       # self-check of the reference list",
        "python -X utf8 code/build_report_final.py            # the .docx, the PDF twin, this file",
        "```",
        "",
        "On the day of sending: set SUBMISSION_DATE in code/build_report_final.py, fill group_number in code/final_report/config.json (prints 'Group #__' while null),",
        "replace the [REPO URL] and [RELEASE/DRIVE URL] placeholders in the registry sources once they exist, rerun numbers.py, then the builder.",
    ]
    if B.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in B.notes]
    BUILD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:16]))
    print("wrote", BUILD_MD)


if __name__ == "__main__":
    main()
