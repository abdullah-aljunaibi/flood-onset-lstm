"""Readers for the professor's-questions sheet (code/build_qa.py) in the final state of 30 October 2026. Nothing here is a result and nothing
here is typed: every function reads a file that already exists, and says on stderr when it cannot.

- FinalReport: the text of the final report, hand-in item 3 (Handin_2026-11-02/3_Final_Project_Report_Flood.docx; its PDF twin is the file
  a reader opens), cut into its sections, tables, figures and appendices. The sheet names a section, table, figure or appendix of that report
  only when the docx has it, and, where a pointer is given for a particular statement, only when that part of the report holds the words the
  pointer is given for (says). A pointer that fails is left out and one line on stderr says so. This replaces the pointers to the long report
  of 8 October 2026 (Flood_Report.pdf), whose section numbers the final report does not use.
- Registry: results/final_report_numbers.json, the registry of the final report (code/final_report/numbers.py): every number the report and the
  final deck print, with its source. The sheet prints its 'text' field, so that sheet, report and deck print the same number in the same form.
- deck_timing(): who presents which slide of the six-slide final deck and for how many seconds (the timing table of the speaker notes).
- team_check_members(): the member table of team_checks/team_check.py (the part, the slides and the figure of each member's own check).
- latest_round() / round_counts(): the latest round of an independent review record and its counts, read tolerantly (the rounds name their
  counts differently: numerical_discrepancies in the first rounds, numerical_mismatches in the five-seed rounds; some rounds list no checks).
- team_members(): the four members with their roles and student ids, and the group number (code/final_report/config.json).
"""
import html
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent                      # the project folder
HANDIN = ROOT.parent / "Handin_2026-11-02"                          # the hand-in folder of 2 November 2026, next to the project folder
FINAL_DOCX = HANDIN / "3_Final_Project_Report_Flood.docx"
FINAL_PDF = HANDIN / "3_Final_Project_Report_Flood.pdf"
DECK = HANDIN / "4_Final_Project_Presentation_Flood.pptx"
DECK_NOTES = HANDIN / "4_Final_Project_Presentation_notes.md"
HANDIN_README = HANDIN / "README_HANDIN.md"
CHECK_SCRIPT = ROOT / "team_checks" / "team_check.py"
CHECK_SHEET = ROOT / "team_checks" / "TEAM_CHECK_SHEET.pdf"
REGISTRY = ROOT / "results" / "final_report_numbers.json"
CONFIG = ROOT / "code" / "final_report" / "config.json"


def warn(text):
    print(f"build_qa: {text}", file=sys.stderr)


def esc(s):
    """A text read from a file, safe inside the sheet's HTML (apostrophes as the sheet writes them)."""
    return html.escape(str(s), quote=False).replace("'", "&rsquo;")


# ---- the final report ---------------------------------------------------------------------------------------------------------
class FinalReport:
    """The final report's text by part. Keys: '4.2' (a section of the second level), '7' (a section of the first level, with its
    subsections), 'T6' (Table 6: its caption and its cells), 'F8' (Figure 8: its caption), 'AppC' (Appendix C), 'Abstract'."""
    LABEL = {"T": "Table", "F": "Figure"}

    def __init__(self, path=FINAL_DOCX):
        self.path, self.parts, self.whole, self.n_references = path, {}, "", 0
        try:
            import docx                                                # python-docx (in the project's requirements)
            from docx.table import Table
            from docx.text.paragraph import Paragraph
            d = docx.Document(str(path))
        except Exception as err:                                       # no file, no library or a damaged file: every pointer degrades
            warn(f"the final report {path.name} could not be read ({err!r}): the sheet names the report without a section")
            return
        blocks = []                                                    # (kind, text) in the order of the document
        for child in d.element.body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                p = Paragraph(child, d)
                text = " ".join(p.text.split())
                if text:
                    blocks.append((p.style.name if p.style is not None else "", text))
            elif tag == "tbl":
                cells = []
                for row in Table(child, d).rows:
                    for c in row.cells:
                        t = " ".join(c.text.split())
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                blocks.append(("table", " | ".join(cells)))
        self.whole = "\n".join(t for _, t in blocks)
        open_parts = {}                                                # key -> list of texts while the part is open

        def close(level):
            for k in [k for k, lv in levels.items() if lv >= level]:
                self.parts[k] = "\n".join(open_parts.pop(k))
                levels.pop(k)
        levels = {}
        last_caption = None
        in_references = False
        for style, text in blocks:
            if style.startswith("Heading"):
                level = 1 if style.endswith("1") else 2
                m1 = re.match(r"(\d+)\. ", text) if level == 1 else None
                m2 = re.match(r"(\d+\.\d+) ", text) if level == 2 else None
                ma = re.match(r"Appendix ([A-Z])\.", text)
                key = (m1.group(1) if m1 else m2.group(1) if m2 else f"App{ma.group(1)}" if ma else
                       "Abstract" if text.startswith("Abstract") else None)
                close(1 if level == 1 else 2)
                in_references = bool(m1) and "References" in text
                if key:
                    open_parts[key], levels[key] = [text], level
                continue
            if style == "Caption":
                m = re.match(r"(Table|Figure) (\d+)\.", text)
                if m:
                    last_caption = f"{m.group(1)[0]}{m.group(2)}"
                    self.parts[last_caption] = text
            elif style == "table" and last_caption and last_caption.startswith("T") and last_caption in self.parts and "|" not in self.parts[last_caption]:
                self.parts[last_caption] += " | " + text               # a table's cells belong to the caption above it
            if in_references and re.match(r"\[\d+\] ", text):
                self.n_references += 1
            for k in open_parts:
                open_parts[k].append(text)
        close(1)

    def ok(self, key):
        return key in self.parts

    def says(self, key, phrase):
        """True if the part holds the phrase (case as written; spaces normalised)."""
        return self.ok(key) and " ".join(phrase.split()) in self.parts[key]

    def has(self, phrase):
        return " ".join(phrase.split()) in self.whole

    @classmethod
    def label(cls, key):
        if key == "Abstract":
            return "the abstract"
        if key.startswith("App"):
            return f"Appendix {key[3:]}"
        if key[0] in cls.LABEL:
            return f"{cls.LABEL[key[0]]} {key[1:]}"
        return f"Section {key}"

    def _checked(self, items, context):
        """The keys of items that hold up: an item is a key, or (key, phrase) when the pointer is given for a statement that part must hold."""
        out = []
        for it in items:
            key, phrase = (it, None) if isinstance(it, str) else it
            if not self.ok(key):
                if self.parts:                                         # (when the report could not be read at all, one line was printed already)
                    warn(f"the final report has no {self.label(key)}: the pointer{context} was left out")
            elif phrase and not self.says(key, phrase):
                warn(f"{self.label(key)} of the final report does not say '{phrase}': the pointer{context} was left out")
            elif key not in out:
                out.append(key)
        return out

    @classmethod
    def _join(cls, keys):
        """['4.2', '5.3', 'T6', 'AppC'] -> 'Sections 4.2 and 5.3, Table 6 and Appendix C' (sections, tables, figures, appendices, in that order)."""
        def order(k):                                                  # 4.2 before 10.1, Table 8 before Table 19
            return [int(x) for x in re.findall(r"\d+", k)] or [0]

        def group(sel, one, many, name):
            ks = sorted((k for k in keys if sel(k)), key=order)
            if not ks:
                return []
            names = [name(k) for k in ks]
            return [(one if len(ks) == 1 else many) + " " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1])]
        parts = (["the abstract"] if "Abstract" in keys else [])
        parts += group(lambda k: k[0].isdigit(), "Section", "Sections", str)
        parts += group(lambda k: k[0] == "T", "Table", "Tables", lambda k: k[1:])
        parts += group(lambda k: k[0] == "F", "Figure", "Figures", lambda k: k[1:])
        parts += group(lambda k: k.startswith("App"), "Appendix", "Appendices", lambda k: k[3:])
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]

    def where(self, *items, context=""):
        """The 'Where' pointer: 'Final report (3_Final_Project_Report_Flood.pdf): Sections 5.3 and 6.3, Table 19'."""
        keys = self._checked(items, context)
        head = f"Final report (<i>{FINAL_PDF.name}</i>)"
        return f"{head}: {self._join(keys)}" if keys else head

    def inline(self, *items, context=""):
        """A pointer inside a sentence: 'final report Section 6.3' ('the final report' when no part holds up)."""
        keys = self._checked(items, context)
        return f"final report {self._join(keys)}" if keys else "the final report"


_FR = None


def final_report():
    global _FR
    if _FR is None:
        _FR = FinalReport()
    return _FR


def pdf_pages(path=FINAL_PDF):
    """The number of pages of a PDF (PyMuPDF), or None."""
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            return doc.page_count
    except Exception as err:
        warn(f"{Path(path).name} could not be opened ({err!r}): its page count is not printed")
        return None


# ---- the registry of the final report ----------------------------------------------------------------------------------------
class Registry:
    """results/final_report_numbers.json: t(key) is the number as the report prints it (KeyError if the key is missing)."""

    def __init__(self, path=REGISTRY):
        try:
            self.d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            warn(f"the registry {Path(path).name} could not be read ({err!r}): the answers that quote it are left out")
            self.d = {}

    def t(self, key):
        return esc(self.d[key]["text"])

    def v(self, key):
        return self.d[key]["value"]

    def has(self, key):
        return key in self.d


# ---- the final deck, the team checks, the team --------------------------------------------------------------------------------
def deck_timing(path=DECK_NOTES):
    """[(slide, title, speaker, seconds)] from the timing table of the speaker notes of the final deck ([] if it cannot be read)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as err:
        warn(f"the speaker notes {Path(path).name} could not be read ({err!r}): the member page names no slides")
        return []
    rows = re.findall(r"^\| (\d+) \| ([^|]+?) \| ([^|]+?) \| (\d+) \| \d+ \| [\d.]+ \|$", text, flags=re.M)
    if not rows:
        warn(f"the timing table of {Path(path).name} was not found: the member page names no slides")
    return [(int(s), title.strip(), who.strip(), int(sec)) for s, title, who, sec in rows]


def team_check_members(path=CHECK_SCRIPT):
    """The MEMBERS table of team_checks/team_check.py ({key: dict(full, role, part, slides, figure)}; {} if it cannot be read). The script
    is only imported: it writes nothing on import (its main() runs only as a script)."""
    try:
        spec = importlib.util.spec_from_file_location("team_check_for_qa", str(path))
        mod = importlib.util.module_from_spec(spec)
        dont = sys.dont_write_bytecode
        spec.loader.exec_module(mod)
        sys.dont_write_bytecode = dont                                  # the script sets it for itself; the builder keeps its own setting
        return {k: {f: v for f, v in m.items() if f in ("full", "role", "part", "slides", "figure")} for k, m in mod.MEMBERS.items()}
    except Exception as err:
        warn(f"{Path(path).name} could not be read ({err!r}): the member page names no command")
        return {}


def team_members(path=CONFIG):
    """(group number, [dict(name, role, student_id)]) from code/final_report/config.json ((None, []) if it cannot be read)."""
    try:
        c = json.loads(Path(path).read_text(encoding="utf-8"))
        return c.get("group_number"), [dict(name=m["name"], role=m["role"], student_id=m.get("student_id", "")) for m in c["members"]]
    except (OSError, ValueError, KeyError, TypeError) as err:
        warn(f"{Path(path).name} could not be read ({err!r}): the members are named as team.json gives them")
        return None, []


def handin_audits(path=HANDIN_README):
    """The audits of the final report as the hand-in README names them ('three audits (compliance, quality, references and originality)'), or ''."""
    try:
        m = re.search(r"(\w+) audits \(([^)]*)\)", Path(path).read_text(encoding="utf-8"))
        return f"{m.group(1)} audits ({m.group(2)})" if m else ""
    except OSError:
        return ""


# ---- the independent review records, read tolerantly --------------------------------------------------------------------------
def rounds_of(rec):
    """The names of the later rounds of a review record, in order: ['round_2', 'round_3', ...]."""
    return sorted((k for k in rec if re.fullmatch(r"round_\d+", k)), key=lambda k: int(k[6:]))


def latest_round(rec):
    """(name, block) of the latest round: ('round_4', {...}) or ('round_1', the record itself)."""
    later = rounds_of(rec)
    return (later[-1], rec[later[-1]]) if later else ("round_1", rec)


def round_counts(block):
    """The counts of one round under one set of names: checks (listed), failed, mismatches (numerical_mismatches or numerical_discrepancies),
    values (compared), must, should. A count the round does not record is None."""
    c = block.get("counts") or {}
    mism = c.get("numerical_mismatches", c.get("numerical_discrepancies"))
    return dict(checks=c.get("checks_listed"), failed=c.get("failed"), mismatches=mism, values=c.get("values_compared"),
                must=c.get("must_fix"), should=c.get("should_fix"))


def failed_checks(block):
    """The checks of one round that did not pass, as (id, what) - a round lists them as dicts with 'result' or 'passed'."""
    out = []
    for c in block.get("checks") or []:
        if not isinstance(c, dict):
            continue
        res = c.get("result", c.get("status", c.get("passed")))
        if res is False or (isinstance(res, str) and res.lower().startswith("fail")):
            out.append((c.get("id"), str(c.get("what", c.get("check", "")))))
    return out
