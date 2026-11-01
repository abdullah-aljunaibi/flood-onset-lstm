"""Flood_Report_3pages.docx and Flood_Report_3pages.pdf: the report as a formal scientific paper, exactly three A4 pages (the hand-in form).

One list of blocks (title, authors, abstract, sections, tables, figures, references) feeds both files. The Word file is written with python-docx (styles for the body, the
headings and the captions; three-line tables made of cell borders; the two pictures inline with their alt text; a page-number field in the footer). The PDF is its twin: the same
strings are put into HTML with CSS that mirrors the Word layout and printed through the project's route (report_common.page() + print_pdf, headless Chrome), and fit_plan() widens the
two figures as far as three pages allow. The dense hand-in version of 8 October (small type, the whole dataset block) is Flood_Report_3pages_dense.pdf / .html, written by
code/build_report_short.py; the two builders never overwrite each other.
Rebuild: python code/build_report_paper.py      (--text prints the text, the word counts and the references without building anything).

Nothing is computed here beyond differences, counts and formatting. Every number is read from results/*.json through the readers that build_report_short.gather() collects (facts,
aorc_facts, stage4_facts, hindsight_facts, forecast_facts, rivers_facts, clock_facts, lobo_facts, day_windows ... of report_sections/s0_summary.py) or by key from the results files
(lstm_all, data_audit, fair_test, flood_analysis, key_figure, data_flow, data_access and the three ladders). The fixed wordings of s0_summary are used word for word: FINDING (title,
abstract, conclusion), SUPPORT (introduction), AORC_WORDING (beside every AORC), HINDSIGHT_WORDING (beside every 51%), REALTIME_WORDING (beside every 48% of Stage IV), INFO_NOT_NET,
MOVED_POINT, NSE_FORECAST_WORDING and REVIEW_WORDING (the contributions); the perfect forecast is always 'an upper bound that cannot be had in practice' and the numbers of E2b are
always marked as another split. A sentence whose wording depends on an outcome is checked against the stored readings; where they no longer support it, the sentence of the dense
report is used and one line says so on stderr. Honesty lines kept: Claude Code wrote and ran everything under the team lead's direction; the reviewers are AI agents, not team members; the
team's own checks are still open (team.json is read through team.py and never written).

FORMAT, as checked on 8 October 2026 (the text of each standard was read, not a summary of it; the copies are public university or publisher pages):
  GB/T 7713.2-2022 'Presentation of academic papers' (in force 2023-07-01; it replaces the paper part of GB/T 7713-1987). Text read from
  https://rdswxb.hainanu.edu.cn/fileRDSWXB/cms/news/info/2026/01/41e67574695042e1aa9c000514287220/ (file 'GBT 7713.2-2022 ...pdf').
    Structure 4.1-4.4 and Annex A: front matter (title, author information under the title, abstract after the authors, keywords after the abstract), body (introduction, main text,
    conclusion, acknowledgements, references), appendices. 4.2.3: the abstract carries no numbers of figures, tables or references (none here); an English abstract may carry more than
    the 400 Chinese characters advised. 4.2.4: 3 to 8 keywords (6 used). 4.3.4: the conclusion is numbered like a chapter. 5.1: A4. 5.2.2: numbering '1', '2.1', '3.2.1' flush left,
    one space, no full stop after the title. 5.2.4: figures and tables numbered in order of appearance. 5.4.3: figure number and caption centred BELOW the figure, a one-character
    gap. 5.5.3: table number and caption centred ABOVE the top rule, notes below the table. Annex B (informative): body 5 hao (10.5 pt), chapter headings xiao-4 hao (12 pt), section
    headings 5 hao (10.5 pt), abstract, keywords, table text, notes and references xiao-5 hao (9 pt), captions xiao-5 hao bold.
    5.6.4 advises thin-space grouping of numbers above 999; the commas of English-language writing and of the project's other documents are kept.
  CY/T 170-2019 'Academic publishing standards - Tables' (the standard GB/T 7713.2 points to), text read from
  https://computing.hit.edu.cn/_upload/article/files/66/98/590984de45dbb48bd930c2482dbc/dd0c91e9-2fec-45c8-b72c-ff29f6cd4138.pdf : 5.2 a table with only the top rule, the header rule and
  the bottom rule is a three-line table; 7.2.1 number and caption above the top rule, number before caption with a one-character gap; 7.2.2.1 a second header level is separated by a thin
  rule; 7.2.3 notes below the bottom rule, first line indented one or two characters, each ending with a full stop; 7.3.1 frame rules thick, the others thin (no weights are given);
  7.3.2-7.3.3 caption text smaller than the body and bolder than the table, table and note text smaller than the body, the body of a table not heavier than its header.
  CY/T 171-2019 'Academic publishing standards - Figures' (https://qk.njfu.edu.cn/DFS//file/2024/07/02/20240702170304532zcd7lw.pdf): 7.2.1 number and caption directly below the figure,
  smaller than the body.
  GB/T 7713.1-2006 (dissertations) was checked through a university specification written after it (https://graduate.bjut.edu.cn/beijinggongyedaxueyanjiushengxueweilunwenzhuanxieguifan.pdf):
  abstract and keywords before the body, body = introduction, main text, conclusion, then references; three-line table with thick rules of 0.75 pt and a thin rule of 0.5 pt, caption
  above the table and below the figure; page number at the foot, centred; references cited as [n] in order of appearance. (A 2025 edition of GB/T 7713.1 is listed by a standards shop;
  its text could not be fetched and is not used.)
  GB/T 7714-2015 'Rules for bibliographic references' (the standard's own text, https://lib.tsinghua.edu.cn/wj/GBT7714-2015.pdf): 9.1 and 10.1: sequential coding, numbered in the order of
  first citation, the number in square brackets, several as [1,3], a run of consecutive numbers with a short dash as [255-256]; 6.3 surname in capitals, given names as initials without
  dots; 8.1.2 up to three authors all listed, more than three: the first three and ', et al'; 4.4.2 'authors. Title[J]. Journal, year, volume(issue): pages.' (Annex A.9 has 'Nature,
  1992, 359: 605-609', so the issue is optional); A.10 'Author. Title[EB/OL]. (publication date)[access date]. URL.', without the publication date '[EB/OL]. [2012-03-27]. URL.';
  Table B.1 type marks M, C, G, N, J, D, R, S, P, DB, CP, EB, A, CM, DS, Z (DS, data set, is new in 2015: the brief asks for [EB/OL], which is used; [DS/OL] would be the more exact mark).
  Layout of a Chinese course paper in English: two web searches, no standard. They gave A4 with margins of 2.5 cm, Times New Roman, a bold centred title in 3 hao (16 pt), bold labels
  for the abstract and keywords, the figure caption below and bold, the page number at the foot, centred: as in the brief. University specifications differ (the BJUT one above uses
  25.4 mm and 31.7 mm margins and a line spacing of 1.3), so the common choice of the brief is made.
Where a convention varies or the standards give no number, the choice and its reason: rule weights 1.5 pt (top, bottom) and 0.75 pt (header, group heading), as common in Word templates
(the BJUT specification has 0.75 and 0.5 pt); both levels of numbered headings 12 pt bold as the brief says (Annex B has 10.5 pt for the second level: H2_PT is the one constant);
abstract and keywords 9 pt (Annex B); table text 8 pt and notes 7.5 pt (Annex B has 9 pt, CY/T 170 only 'smaller than the body'); references 8.5 pt and the contributions 8 pt
(Annex B 9 pt): all for the three pages; paragraphs indented 0.74 cm (two characters) with no space between them, as Chinese typesetting does; a caption is bold from its number on.
Not added, on purpose: a Chinese title, abstract and keywords. GB/T 7713.2-2022 4.2.1-4.2.4 advise a foreign-language (mostly English) title, authors, abstract and keywords for a Chinese
paper; for a paper written in English the counterpart would be Chinese versions. Whether the hand-in needs them (the course language and the professor's wish are not stated in the
project) is the team's decision.

REFERENCES: report_common.REFERENCES (checked on 3 to 8 October 2026) and results/published_context.json are the only sources of detail: nothing is typed from memory. Authors are
converted to the GB/T 7714 form (three, then 'et al'). Left out because the project's records do not hold them: the issue numbers of the journal articles (the standard allows
'year, volume: pages') and the DOI of Newman et al. (2015) and Addor et al. (2017); a DOI is required for online resources but not for [J], and none is printed for any article, to keep
the list uniform. The two data sources AORC and GEFSv12 are [EB/OL] with the base address of results/data_access.json and the access date of results/data_flow.json (7 October 2026);
Stage IV (also from results/data_access.json) is described in Table 1 and not cited, as the brief lists only the two. The project's results files are not references: the paragraph on
contributions points to them.

LAYOUT AND FIT: A4, margins 2.5 cm, Times New Roman 10.5 pt, single line spacing, justified; headings 12 pt bold; captions 9 pt bold; the footer carries the page number. Two figures
and two tables only. Figure 2 is figures/key_figure.png whole. Figure 1 is the ten stages of figures/data_flow_talk.png: its two rows of boxes (FLOW_BANDS, stacked into one picture in a
temporary folder), without the band of products above (Table 1 has it) and the band of extra inputs between the rows (the text has it); the dense report cuts the same bands. The imagery
note of results/data_flow.json (first two sentences, word for word) stands in the notes of Table 1 at 7.5 pt; REALTIME_WORDING and HINDSIGHT_WORDING stand in the notes of Table 2.
fit_plan() prints the paper at figure widths from 16 cm down to 12 cm (FIG_CM) and keeps the widest flow figure and then the widest key figure at which the print is exactly three pages
with SAFETY_MM of the last page to spare; the Word file gets the same widths. What three pages hold with these figures, tables and sizes is about 1,300 words from the title to the end of
the conclusion (the brief asked for 1,300 to 1,500): the same text at 1,500 words prints as 3.4 pages. The floats stand at the places PLACEMENT names (search_layout() finds them again
by printing every placement when the text has changed and PLACEMENT no longer fits).
Word could not be run here (no Word, no LibreOffice). The .docx was checked by re-opening it with python-docx (verify_docx), by the XSD validator of the docx skill (run with the system
python3, which has defusedxml and lxml: python3 validate.py Flood_Report_3pages.docx) and by Apple's Quick Look; its text, sizes, widths and line breaking are those of the PDF twin
(same font, same text widths, kerning and ligatures off in the print), and SAFETY_MM is the margin for what Word may still do differently. python-docx is not in requirements.txt:
without it the build writes the PDF and says so.
"""
import html as _html
import re
import struct
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import team
from common import FIG, ROOT
from flood import LEADS
from report_common import LEAD_WORD, REFERENCES, count, load_results, n2, page, pct, pdf_pages, print_pdf, signed
from report_sections import s0_summary as S0
from report_sections import s8_limits
import build_report_short as short

NAME = "Flood_Report_3pages"                                                 # .docx and .pdf (the dense version writes Flood_Report_3pages_dense.*)
UPPER = short.UPPER                                                          # 'an upper bound that cannot be had in practice'
first, k2, k3 = LEADS
word = S0._word

# ---- the layout, one place for Word and for the print -------------------------------------------------------------------------
TEXT_W_CM = 16.0                                                             # A4 21 cm less two margins of 2.5 cm
MARGIN_MM = 25
BODY_PT, TITLE_PT, H1_PT, H2_PT = 10.5, 16.0, 12.0, 12.0                      # H2_PT: Annex B of GB/T 7713.2-2022 has 10.5 for the section level
TAB_PT, NOTE_PT, CAP_PT, REF_PT, BACK_PT, ABS_PT = 8.0, 7.5, 9.0, 8.5, 8.0, 9.0
H1_BEFORE_PT, H2_BEFORE_PT = 3.5, 2.0                                       # space before the headings (after: 1 pt); Word adds before and after, so the print uses padding
INDENT_CM = 0.74                                                             # two characters of 10.5 pt
THICK_PT, THIN_PT = 1.5, 0.75                                                # the rules of the three-line tables
SAFETY_MM = 17                                                               # the last page ends at least this far above its foot margin
FIG_CM = tuple(x / 2 for x in range(32, 23, -1))                             # 16.0, 15.5 ... 12.0 cm
FIG_FLOOR_CM = 12.0
FLOW_BANDS = ((414, 654), (950, 1212))                                        # the two rows of boxes (the ten stages) of figures/data_flow_talk.png, 2640 x 1231 px: the band of products above and the band of extra inputs between the rows are left out (Table 1 and the text have them); None = the whole figure
NB = "\u00a0"
NBH = "\u2011"                                                                # a hyphen that does not break the line (dates)
GAP = NB + NB                                                                # 'Table 1  Caption': a gap of two spaces

CITED = []                                                                   # keys of REFERENCES in the order of first citation


def say(msg):
    """One line on stderr (once per build): the stored numbers no longer support a wording."""
    S0.note(f"build_report_paper: {msg}")


def cite(*keys):
    """'[1]', '[2-3]', '[1,4]': the numbers of the sequential coding system (GB/T 7714-2015 10.1), assigned in the order of first citation."""
    nums = []
    for k in keys:
        if k not in CITED:
            CITED.append(k)
        nums.append(CITED.index(k) + 1)
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}-{nums[j]}" if j > i else str(nums[i]))
        i = j + 1
    return "[" + ",".join(parts) + "]"


# ---- HTML fragments (the project's strings) -> runs, for the Word file and for the word count ----------------------------------
class _Runs(HTMLParser):
    """A fragment with <b>, <i>, <sub>, <sup>, <br> and entities -> [(text, bold, italic, 'sup'|'sub'|'')]; the text of '<wbr>' is a zero-width space."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.runs, self.b, self.i, self.sup, self.sub = [], 0, 0, 0, 0

    def _add(self, text):
        if text:
            self.runs.append((text, bool(self.b), bool(self.i), "sup" if self.sup else "sub" if self.sub else ""))

    def handle_starttag(self, tag, attrs):
        if tag in ("b", "strong"):
            self.b += 1
        elif tag in ("i", "em"):
            self.i += 1
        elif tag == "sup":
            self.sup += 1
        elif tag == "sub":
            self.sub += 1
        elif tag == "br":
            self._add("\n")
        elif tag == "wbr":
            self._add("\u200b")

    def handle_endtag(self, tag):
        if tag in ("b", "strong"):
            self.b -= 1
        elif tag in ("i", "em"):
            self.i -= 1
        elif tag == "sup":
            self.sup -= 1
        elif tag == "sub":
            self.sub -= 1

    def handle_data(self, data):
        self._add(re.sub(r"[ \t\r\n\f\v]+", " ", data))


def runs_of(fragment):
    p = _Runs()
    p.feed(fragment)
    p.close()
    return p.runs


def plain(fragment):
    """The text of a fragment: tags dropped, entities turned into characters, zero-width spaces dropped, U+2011 (a hyphen that does not break) read as '-'."""
    text = "".join(r[0] for r in runs_of(fragment))
    return re.sub(r"\s+", " ", text.replace("\u200b", "").replace(NBH, "-").replace(NB, " ")).strip()


def n_words(fragment):
    return len(plain(fragment).split())


def cap1(s):
    return s[0].upper() + s[1:]


def rng(r, share=True):
    return short.rng(r, share)


def rng_u(r, share=True):
    """The range of short.rng with a single '%' at the end (the table cells of this paper)."""
    return short.rng(r, share) if r else ""


# ---- the references, in the form of GB/T 7714-2015 --------------------------------------------------------------------------
JOURNALS = ("newman2015", "addor2017", "kratzert2019", "nearing2024", "guan2022")
DATA_SOURCES = {"aorc": ("aorc_v1_1_aws", "aorc"), "gefsv12": ("gefsv12_reforecast_aws", "gefs")}


def gbt_authors(text):
    """'Newman, A. J., Clark, M. P., ... ' -> 'NEWMAN A J, CLARK M P, SAMPSON K, et al': surname in capitals, initials without dots, the first three authors
    then 'et al' (GB/T 7714-2015 6.3 and 8.1.2). Only what the record holds: the initials as report_common.REFERENCES gives them."""
    found = re.findall(r"([A-Z][A-Za-z'\-]+),((?: [A-Z]\.)+)", text)
    names = [f"{surname.upper()} {' '.join(re.findall('[A-Z]', initials))}" for surname, initials in found]
    return ", ".join(names[:3]) + (", et al" if len(names) > 3 else "")


def journal_reference(key):
    """'AUTHORS. Title[J]. Journal, year, volume: pages.' from report_common.REFERENCES (no issue, no DOI: not in the record)."""
    full = REFERENCES[key][1]
    m = re.match(r"(?P<au>.+?) \((?P<year>\d{4})\)\. (?P<title>.+?)\. (?P<journal>[A-Z][^.]*?), (?P<vol>\d+)(?:\((?P<iss>\d+)\))?, (?P<pages>[\d&ndash;\-]+|e[\w]+)\.", full)
    if not m:
        raise ValueError(f"reference {key} is not in the expected form")
    au = gbt_authors(m["au"])
    pages = m["pages"].replace("&ndash;", "-")
    return f"{au}. {m['title']}[J]. {m['journal']}, {m['year']}, {m['vol']}: {pages}."


def data_reference(key, R, G):
    """'NOAA. Title[EB/OL]. [access date]. URL.' for a data source: the title of report_common.REFERENCES (the part before '; AWS Open Data'), the base address of
    results/data_access.json and the access date of results/data_flow.json."""
    acc, flow = DATA_SOURCES[key]
    groups = R["data_access"]["summary"]
    rec = next(v for g in groups.values() if isinstance(g, dict) for k, v in g.items() if k == acc)
    base = re.match(r"(https://[^/\s]+/)", rec["url"]).group(1)
    item = short.df_item(G["DF"], flow)
    day = re.match(r"(\d{4}-\d{2}-\d{2})", item["obtained"]["date_downloaded"]).group(1).replace("-", NBH)
    full = REFERENCES[key][1].split("; AWS Open Data")[0]
    author, title = full.split(" ", 1)                                          # 'NOAA' + the title
    return f"{author}. {title}[EB/OL]. [{day}]. {base}."


def references_html(R, G):
    out = []
    for key in CITED:
        out.append(journal_reference(key) if key in JOURNALS else data_reference(key, R, G))
    return out


# ---- the blocks of the paper: one list, written to Word and printed to PDF ---------------------------------------------------------
def C(h, span=1, rspan=1, align="c", under=False, **kw):
    """A cell of a three-line table: h = HTML fragment; span, rspan = columns, rows covered; align 'l' | 'c'; under = a thin rule under the cell (a group heading)."""
    return dict(h=h, span=span, rspan=rspan, align=align, under=under, **kw)


def date_of(text):
    """'2026-10-07 18:38 to 2026-10-07 19:44 laptop time' -> '2026-10-07'; two different days -> '2026-10-07/08'."""
    days = list(dict.fromkeys(re.findall(r"\d{4}-\d{2}-\d{2}", str(text))))
    if len(days) > 1 and days[0][:7] == days[1][:7]:
        return f"{days[0].replace('-', NBH)}/{days[1][8:]}"
    return days[0].replace("-", NBH) if days else ""


def table1(G):
    """Table 1: the data products, one line each where the text allows. Every fact is read from results/data_flow.json (figure_line, obtained)."""
    DF = G["DF"]
    it = lambda key: short.df_item(DF, key)
    fl = lambda key: it(key)["figure_line"]
    ob = lambda key: it(key)["obtained"]
    star = lambda key: "" if "recorded in a results file" in str(ob(key).get("bytes_read_kind", "")) else "*"
    got = lambda key, how=None: (f"{short.size_of(ob(key)['bytes_read'])}{star(key)} &middot; {how or short.login_of(ob(key)['login_needed'])} &middot; {date_of(ob(key)['date_downloaded'])}")
    po = lambda key: it(key)["figure_line"][1]
    head = [[C("Product", align="l"), C("Grid &middot; step &middot; period", align="l"), C("Obtained: size &middot; login &middot; date", align="l")]]
    rows = [
        [C(f"{fl('camels_us')[0]} (flow, weather, properties)", align="l"), C(fl("camels_us")[1], align="l"), C(got("camels_us", "by hand"), align="l")],
        [C(f"{fl('aorc')[0]}, {S0.AORC_WORDING}", align="l"), C(po("aorc"), align="l"), C(got("aorc"), align="l")],
        [C(f"{fl('gefs')[0]}, control member", align="l"), C(po("gefs"), align="l"), C(got("gefs"), align="l")],
        [C("ERA5 (Open-Meteo)", align="l"), C(po("era5"), align="l"), C(got("era5"), align="l")],
        [C("IMERG and MERRA-2 (NASA POWER)", align="l"), C(f"{po('imerg')}; {po('merra2')}", align="l"),
         C(f"{short.size_of(ob('imerg')['bytes_read'])}* + {short.size_of(ob('merra2')['bytes_read'])}* &middot; {short.login_of(ob('imerg')['login_needed'])} &middot; {date_of(ob('imerg')['date_downloaded'])}", align="l")],
        [C(f"{fl('stage4')[0]} radar and gauge rain", align="l"), C(po("stage4"), align="l"), C(got("stage4"), align="l")],
    ]
    return dict(widths=[5.4, 5.2, 5.4], head=head, rows=rows, cls="t1")


MIX = "<sup>&para;</sup>"                                                       # 'mixed' beside an NSE (the sign of Table 2, as the key figure prints 'mixed')


def table2(G):
    """Table 2: the ladder of inputs at 1, 2 and 3 days ahead, from the ladders of the results files (as short.tables reads them), and its notes."""
    R, AO, S4, FC, W, TF = G["R"], G["AO"], G["S4"], G["FC"], G["W"], G["TF"]
    X_A, X_E, X_F = R["aorc_rain_to_midnight"], R["rain_to_midnight"], G["FIN"]
    a = {r["set"]: r["by_lead"] for r in X_A["ladder"]["wq"]["rungs"]}
    e = {r["set"]: r["by_lead"] for r in X_E["ladder"]["wq"]["rungs"]}
    fm = {r["set"]: r["by_lead"] for r in X_F["ladder"]["main"]["rungs"]}
    fs = {r["set"]: r["by_lead"] for r in X_F["ladder"]["sat"]["rungs"]}
    mixed = lambda name, k: bool(name and G["mixed_rows"].get(f"{'upper' if k == first else 'lower'}.{name.removeprefix('lstm_')}.l{k}"))
    dash = C("&ndash;")

    def row(label, by_lead, name=None, prefix=""):
        g = lambda k, key: by_lead[f"l{k}"].get(prefix + key)
        on = [C(f"{100 * g(k, 'onset_warned'):.0f}") for k in LEADS]
        nse = [C(n2(g(k, "nse_median")) + (MIX if mixed(name, k) else "")) for k in LEADS]
        seeds = C(f"{rng_u(g(first, 'onset_seed_range')).removesuffix('%')} &middot; {rng_u(g(first, 'nse_seed_range'), share=False)}")
        return [C(label, align="l")] + on + nse + [seeds]

    head = [[C("Inputs, with the river gauge", rspan=2, align="l"), C("Onsets warned (%)", span=3, under=True), C("Median NSE", span=3, under=True),
             C("Seed range, 1 day (%, NSE)", rspan=2)],
            [C(LEAD_WORD[k].replace(" ahead", "")) for k in LEADS] * 2]
    rows = [row("Rain table alone", a["lstm_wq"]),
            row(f"+ NLDAS-2 rain, day ends about {W['nldas']}", e["lstm_wqn"], "lstm_wqn"),
            row("+ ERA5 rain to midnight (model rain)", e["lstm_wqh"]),
            row(f"+ AORC rain to midnight, {S0.AORC_WORDING} <sup>&Dagger;</sup>", a["lstm_wqa"]),
            row(f"control: AORC over the table&rsquo;s window ({W['daymet']}&ndash;{W['daymet']})", a["lstm_wqa"], prefix="control_"),
            row("+ GEFS forecast of the next three days (E2a)", fm["lstm_wqaF_sub"], "lstm_wqaF_sub"),
            row("perfect rain of the next three days <sup>&dagger;</sup>", fm["lstm_wqaP"]),
            row("perfect weather of the next three days <sup>&dagger;</sup>", fm["lstm_wqf"]),
            row("E2b, forecast retrained: another split <sup>&sect;</sup>", fs["lstm_wqaF_s"])]
    ns = lambda d: ", ".join(count(d[k]) for k in LEADS[:-1]) + " and " + count(d[LEADS[-1]])
    years = lambda s: f"{X_F[s]['test'][0][:4]}&ndash;{X_F[s]['test'][1][:4]}"
    r1 = TF["gauge"][first]
    notes = (f"Seed range: lowest&ndash;highest single training; onsets: {ns(FC['n'])} at {', '.join(str(k) for k in LEADS[:-1])} and {LEADS[-1]} days. "
             f"<sup>&dagger;</sup> {cap1(UPPER)}. <sup>&Dagger;</sup> AORC: {G['hind']}. Stage IV in place of AORC, {LEAD_WORD[first]}: {pct(G['share48'])} of the onsets warned ({pct(G['share45'])} with the last hour of the issue day left out); {S0.REALTIME_WORDING}. "
             f"E2a puts the forecast into the network trained on the AORC rain of the days being forecast; E2b retrains the network with it <sup>&sect;</sup> on another split (test years {years('split_satellite_era')}, "
             f"{ns(FC['n_sat'])} onsets; a weaker base: rain table alone {'/'.join(pct(FC['s_wq'][k]) for k in LEADS)}), not comparable with the other rows. "
             f"{MIX} Mixed: the median rises, but not clearly (higher in {r1['basins_better']} of {TF['n']} basins with NLDAS-2 rain; {FC['nse_basins'][k2]} and {FC['nse_basins'][k3]} of {FC['nse_n']} with the forecast).")
    return dict(widths=[7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0], head=head, rows=rows, notes=notes, cls="t2")


AFFILIATION = "The university"
COURSE = "AI course group project"
KEYWORDS = ("flood forecasting", "LSTM", "CAMELS-US", "rainfall timing", "weather forecast input", "NSE")


def gather_paper(R, T):
    G = short.gather(R, T)
    if G is None:
        return None
    K = G["K"]
    spread = (K.get("finding") or {}).get("spread_between_model_kinds") or {}
    G["spread_kinds"] = (spread.get("nse"), spread.get("onset_warned_points"), bool((K.get("finding") or {}).get("holds")))
    return G


def blocks(R, T, G):
    """The paper as a list of blocks, in document order (the order of first citation follows it): dicts with a 'kind' and an HTML fragment 'h'."""
    CITED.clear()
    f, A, AO, E1, S4, HS, FC, RV, TF, C_, LF, W, UNC, D, DF, FT = (G[k] for k in ("f", "A", "AO", "E1", "S4", "HS", "FC", "RV", "TF", "C", "LF", "W", "UNC", "D", "DF", "FT"))
    n, n_on, n_seeds = f["n"], f["n_on"], f["n_seeds"]
    cmp1, on0, up, hind, rt, aorc = G["cmp1"], G["on0"], G["up"], G["hind"], S0.REALTIME_WORDING, S0.AORC_WORDING
    verdict = G["verdict_words"]
    st = {x["id"]: x["counts"] for x in DF["flow"]["stages"]}
    hc = DF["how_computed"]
    B = []
    add = lambda kind, h="", **kw: B.append(dict(kind=kind, h=h, **kw))
    t = team.load()
    num = lambda x: f"{x:,}"
    s1 = lambda v: signed(v, 1)
    lead1 = LEAD_WORD[first]
    leads23 = S0._leads([k2, k3])

    # ---- title block -----------------------------------------------------------------------------------------------------------
    add("title", f"Predicting river floods with AI: {S0.FINDING[0].lower() + S0.FINDING[1:]}")
    add("authors", S0._esc(team.names_line()))
    add("affil", f"{S0._esc(t['university'] or AFFILIATION)}; {S0._esc(t['course'] or COURSE)}; {f['built']}")

    # ---- abstract: the question, data, method, the finding, the numbers, what a service would need ------------------------------
    add("abstract", f"Can AI predict a river&rsquo;s flow one to three days ahead and warn of high-flow days beforehand? One long short-term memory network (LSTM) "
                    f"for all {word(n)} CAMELS-US basins ({_years('train', 'test')}) was tested against simple models on years never used for any choice. "
                    f"{S0.FINDING}. With the river gauge the LSTM&rsquo;s median Nash&ndash;Sutcliffe efficiency (NSE) is {n2(cmp1['lstm_nse_median'])} against {n2(cmp1['baseline_nse_median'])} "
                    f"for the best simple model, and {pct(on0)} of {num(n_on)} flood onsets are warned a day ahead. Rain to midnight from AORC, {aorc}, lifts this to {pct(AO['with_'][first])} "
                    f"({hind}), and to {pct(G['share48'])} with Stage IV radar: {rt}. A public forecast gives {pct(FC['fc'][k2])} and {pct(FC['fc'][k3])} two and three days ahead. "
                    f"The onset verdict stays {verdict}. A service would need radar rain up to the forecast and a forecast of rain to come.")
    add("keywords", "; ".join(KEYWORDS))

    # ---- 1 Introduction ----------------------------------------------------------------------------------------------------------
    add("h1", "1 Introduction")
    add("p", id="intro", h=f"An LSTM, a network that reads the days in order and keeps a memory, can be trained on hundreds of rivers at once {cite('kratzert2019')} with the CAMELS data "
             f"{cite('newman2015', 'addor2017')}. Can it forecast a river&rsquo;s flow one to three days ahead, and warn of high-flow days before they happen, better than simple models? "
             f"We ask it in a fair test and, if the answer is no, what limits the forecast. We show that, {S0.SUPPORT[0].lower() + S0.SUPPORT[1:]}.")

    # ---- 2 Data and methods ------------------------------------------------------------------------------------------------------
    add("h1", "2 Data and methods")
    add("h2", "2.1 Data")
    cam = short.df_item(DF, "camels_us")
    area = re.search(r"are ([\d,]+) to ([\d,]+) km2", cam["spatial_grid"])
    dl = st["download"]
    au = st["audit"]
    per = word(int(re.search(r"(\d+) per landscape", next(x["words"] for x in DF["flow"]["stages"] if x["id"] == "audit")).group(1)))
    add("p", id="data1", h=f"From the {au['all_camels']} river basins of CAMELS-US {cite('newman2015', 'addor2017')} an audit ({au['all_camels']} &rarr; {au['candidates']} candidates &rarr; "
             f"{au['complete']} with a complete record &rarr; {au['chosen']} chosen by rule, {per} per landscape, using no flow statistic and no model score) leaves {word(au['chosen'])} basins of "
             f"{area.group(1)}&ndash;{area.group(2)} km&sup2;, {num(st['inputs']['days_per_basin'])} days each ({_years('train', 'test')}). The experiments add the public products of Table 1, none needing "
             f"an account or payment ({dl['gb_read']:.0f} GB read): AORC {cite('aorc')}, {aorc}; the GEFSv12 reforecast {cite('guan2022', 'gefsv12')}; ERA5; IMERG; MERRA-2; and Stage IV radar, for a check only.")
    imagery = re.split(r"(?<=\.) (?=[A-Z])", DF["imagery_note"])
    add("table", fid="T1", table=table1(G), caption="Data products: grid, step and period, and how each was obtained",
        notes=(f"* Size of the stored files (transfer size not recorded). Four more products are maps or checks, not model inputs; "
               f"{len(DF['unverified'])} facts are unverified (<i>results/data_flow.json</i>). <i>Imagery.</i> {_html.escape(' '.join(imagery[:2]), quote=False)}"))
    flow_q = hc["flow_mm_per_day"]["formula"].replace(" x ", " &times; ").replace("q_mm", "<i>q</i><sub>mm</sub>").replace("area_m2", "area<sub>m&sup2;</sub>")
    dm = hc["where_each_daily_value_ends"]["values"]["daymet"]
    add("p", id="data2", h=f"Flow is converted to mm/day: {flow_q}. Grid products are averaged over each basin (IMERG and GEFS cells weighted by overlap with the outline), and hourly rain is "
             f"summed over the local wall-clock day of the gauge, so that no hour after the issue time, local midnight, is read (Figure 1). Against hourly rain, "
             f"the Daymet day ends at about {dm['rounded']}, not at midnight.")
    add("figure", fid="F1", name="data_flow_talk.png", role="flow",
        alt=(f"The flow of the data through ten stages, in two rows of boxes: 1 raw downloads ({dl['products']} products, {dl['gb_read']:.0f} GB), 2 audit of the basins ({au['all_camels']}, "
             f"{au['candidates']}, {au['complete']}, {au['chosen']}), 3 caches, 4 daily inputs per basin, 5 the split into training, validation and test days, then 6 scaling and the high-flow level, "
             f"7 training, 8 stored forecasts, 9 scores, 10 documents."),
        caption="The ten stages of the flow of the data, from the downloads to the scores (the products are in Table 1)")

    # ---- 2.2 Model and baselines -------------------------------------------------------------------------------------------------
    add("h2", "2.2 Model and baselines")
    cfg = A["config"]
    add("p", id="model", h=f"One LSTM {cite('kratzert2019')} ({cfg['hidden']} memory cells, {num(st['training']['weights'])} weights; {word(st['inputs']['weather_columns'])} weather columns, "
             f"the river gauge, {word(st['inputs']['properties'])} basin properties) serves {word(n)} rivers and forecasts {word(len(LEADS))} days ahead; {word(n_seeds)} "
             f"trainings are averaged. Simple models given the same inputs: &lsquo;tomorrow = today&rsquo;, ridge regression, a random forest.")

    # ---- 2.3 The fair test and the scores ----------------------------------------------------------------------------------------
    add("h2", "2.3 The fair test and the scores")
    sp = st["split"]
    z = lambda v: n2(0.0 if abs(v) < 0.0005 else v, 3)
    leak = (f"On made-up rivers with known answers the scores ({', '.join(z(v) for v in FT['measured']['past_weather_only'][:-1])}, {z(FT['measured']['past_weather_only'][-1])}) "
            f"equal the most an honest model can reach ({', '.join(n2(v) for v in FT['expected']['past_weather_only'][:-1])}, {n2(FT['expected']['past_weather_only'][-1])}): "
            f"no leak." if FT["passed"] else "On made-up rivers with a known answer NOT all checks passed.")
    crit = G["go"]
    add("p", id="test", h=f"Training covers {S0._years('train')} ({num(sp['train_days'])} days), validation {S0._years('val')} ({num(sp['validation_days'])}) and test {S0._years('test')} ({num(sp['test_days'])}). "
             f"Settings and stopping points come from the validation years; the test years are only scored. {leak} NSE: 1 is "
             f"perfect, 0 the average flow. A high-flow day exceeds the level of {pct(1 - S0.HIGH_Q)} of training days; an <i>onset</i> is a high-flow day "
             f"forecast while the river was still below it. Verdicts, by criteria fixed beforehand: GO, at least {pct(crit['go_pod'])} of onsets warned, at most {pct(crit['go_far'])} "
             f"of warnings false; CAUTION, not GO, but hits at least {pct(crit['ca_csi'])} of hits, misses and false warnings together; NOT YET, neither.")

    # ---- 3 Results -----------------------------------------------------------------------------------------------------------------
    add("h1", "3 Results")
    add("h2", "3.1 The kind of model")
    cls = f["cls"]
    tied = all(v == "tie" for v in cls["wq"].values()) and f["acc"] == "no clear difference in accuracy"
    nk = G["spread_kinds"]
    kinds = (f"Four kinds of model lie within {n2(nk[0])} NSE and {nk[1]:.0f} points of onsets warned; a Transformer gives {f['acc'].replace(' in accuracy', '')} at {f['t_lo']:.0f} to {f['t_hi']:.0f} times the training time. "
             if nk[2] and nk[0] is not None else f"A Transformer gives {f['acc']} at {f['t_lo']:.0f} to {f['t_hi']:.0f} times the training time. ")
    if tied:
        t31 = (f"With the river gauge the LSTM&rsquo;s median NSE is {n2(cmp1['lstm_nse_median'])} against {n2(cmp1['baseline_nse_median'])} for the best simple model: "
               f"{S0.reading_word('no clear difference')} at any lead time ({word(n_seeds)} trainings of one LSTM differ by {n2(G['spread'])}). {kinds}"
               f"The milestone &lsquo;the LSTM beats the baselines&rsquo; is {f['m2'].split(' (')[0]}.")
    else:
        say("the readings of the model comparison changed: 3.1 is the sentence of the dense report")
        t31 = short.findings(G)[0][1]
    add("p", id="kind", h=t31)

    add("h2", "3.2 The information")
    r = TF["gauge"][first]
    nse_bit = {"clear": f"its NSE gain is clear as well (higher in {r['basins_better']} of {TF['n']} basins)", "mixed": f"its NSE gain is mixed",
               "none": "its NSE shows no clear difference"}[S0.nse_reading(r, TF["n"])]
    gap = f"{S0.share_word(r['onset_share_of_gap_closed'])} of the gap to the perfect weather forecast, {UPPER} ({pct(up)})"
    evening = (f"Rain up to the evening (NLDAS-2) lifts the onsets warned from {pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} "
               f"(control {pct(r['onset_warned_control'])}), {gap}; {nse_bit}." if S0.onset_clear(r) else
               f"Rain up to the evening of the issue day takes the onsets warned from {pct(r['onset_warned_base'])} to {pct(r['onset_warned_with_nldas'])} "
               f"(control {pct(r['onset_warned_control'])}), but the single seeds overlap: no clear gain; {nse_bit}.")
    flat = "" if AO["control_flat"] else "; the control is not flat"
    as_much = min(AO["gain"]["ridge"], AO["gain"]["forest"]) >= S0.AS_MUCH * AO["gain"]["lstm"]
    yard = (f"Ridge regression and a random forest given the AORC rain warn of {pct(AO['warned']['ridge'])} and {pct(AO['warned']['forest'])}: {S0.INFO_NOT_NET}."
            if as_much else f"Ridge regression and a random forest given the AORC rain warn of {pct(AO['warned']['ridge'])} and {pct(AO['warned']['forest'])}.")
    add("p", id="info", h=f"Table 2 and Figure 2 give the ladder of inputs. {evening} AORC, {aorc}, lifts them to {pct(AO['with_'][first])} ({hind}{flat}). "
             f"With Stage IV, {S4['stage4']} ({pct(G['share48'])}) of the {num(S4['n'])} onsets are warned, against {S4['aorc']} with AORC and {S4['table']} with the rain table "
             f"alone (Table 2, note <sup>&Dagger;</sup>). {yard} At {leads23} the GEFS forecast lifts the onsets warned from {pct(FC['rain'][k2])} and {pct(FC['rain'][k3])} with AORC rain alone to "
             f"{pct(FC['fc'][k2])} and {pct(FC['fc'][k3])}, {S0.reading_word(FC['gain_reading'][k2]) if FC['gain_reading'][k2] == FC['gain_reading'][k3] else 'a mixed result'}.")
    add("figure", fid="F2", name="key_figure.png", role="key",
        alt=(f"The main finding at a glance: onsets warned and median NSE for each set of inputs, {lead1} above and {leads23} below, test years {plain(f['test_years'])}, {n} basins. "
             f"Four kinds of model on the same inputs lie close together; rain up to the evening, rain up to midnight and a weather forecast lift the LSTM far more."),
        caption=f"The main finding at a glance (test years {f['test_years']}, {n} basins)")
    t2 = table2(G)
    add("table", fid="T2", table=t2, caption=f"Inputs and skill with the river gauge: onsets warned and median NSE, {lead1.replace(' ahead', '')}, two and three days ahead", notes=t2["notes"])

    add("h2", "3.3 Rivers never seen")
    add("p", id="rivers", h=f"A river never seen loses {n2(LF['loss'])} NSE {lead1} (median {n2(LF['seen'])} to {n2(LF['unseen'])}). With "
             f"{RV['n_train']['unseen50_w_static']} training rivers instead of {RV['n_train']['lobo_w_static']} the median rises from {n2(RV['unseen']['lobo_w_static'][first])} to "
             f"{n2(RV['unseen']['unseen50_w_static'][first])} ({n2(RV['unseen']['lobo_w_nostatic'][first])} to {n2(RV['unseen']['unseen50_w_nostatic'][first])} without basin properties), but the number, "
             f"variety and nearness of the training rivers changed together: scale and nearness are not separated.")

    add("h2", "3.4 The warnings")
    false_f = [pct(FC["false_fc"][k]) for k in (k2, k3)]
    add("p", id="warn", h=f"Onset warnings are {verdict} at all lead times and with every input tried: {pct(on0)} of {num(n_on)} onsets warned with today&rsquo;s inputs, and "
             f"{pct(cmp1['lstm_onset_far'])} of warnings are false ({pct(AO['false'][first])} with AORC; {false_f[0]} and {false_f[1]} with the forecast {leads23}). "
             f"All high-flow days: {G['v_all']} {lead1}" + (", as for &lsquo;tomorrow = today&rsquo;" if G["pers"] == G["v_all"] else "") + ".")

    # ---- 4 Discussion --------------------------------------------------------------------------------------------------------------
    add("h1", "4 Discussion")
    both = AO["met"] and FC["success"]
    head = ("The finding holds in both directions: taking information away lowered the skill" if both else "Taking information away and giving it: what the inputs changed")
    add("p", f"{head} (no river gauge {signed(-G['g1'])} NSE, a river never seen {signed(-LF['loss'])}), giving it raised it. The hours matter: the rain "
             f"table&rsquo;s day ends at about {W['daymet']}, so {pct(C_['hidden'])} of the rain around the onsets, falling from then to midnight, is hidden from the model (worth {short.n1(AO['hours'])} "
             f"points of onsets warned, {short.n1(AO['evening'])} after {W['imerg']}), and {pct(C_['future'])} falls after the forecast is issued.")
    ar = " (whether its archive holds first-posted or later-corrected hours is not established)" if S4["archive_unestablished"] else ""
    if FC["moved"]:
        moved = (f"At {lead1} the forecast warns of fewer onsets than AORC rain ({s1(FC['gain'][first])} points, to {pct(FC['fc'][first])}) with far fewer false warnings "
                 f"({pct(FC['false_fc'][first])} against {pct(FC['false_rain'][first])}): {S0.MOVED_POINT}, said of the warnings only; {S0.nse_forecast_words(FC)}.")
    else:
        moved = f"At {lead1} the forecast warns of {s1(FC['gain'][first])} points of onsets ({S0.reading_word(FC['gain_reading'][first])}); {S0.nse_forecast_words(FC)}."
    add("p", f"AORC is {aorc}, so its {pct(AO['with_'][first])} could hold hindsight: an independent review bounds it ({hind}), and the Stage IV test keeps "
             f"{pct(S4['share'])} of the effect; it is a check, not a service{ar}. {moved}")
    yr = re.search(r"hours before (\d{4}), which are all the training years", str(R["aorc_rain_to_midnight"].get("outcome_caveat", "")))
    before = (f"hours before {yr.group(1)}, all the training years, are disaggregated daily totals" if yr else "hours in the training years are disaggregated daily totals")
    pk = R["flood_analysis"]["failures"]["peak_size"]["lstm_wq"]["all_events"]["median"]
    add("p", f"The limits are {word(n)} basins, {word(UNC['resampling']['n_years'])} test years, {word(n_seeds)} trainings, one ensemble member of {word(FC['members'])}, and AORC, {aorc}, "
             f"whose {before}; the forecast run barely reaches the issue time (its last files appear about {abs(round(FC['late']))} minutes late in Eastern summer). "
             f"The team members&rsquo; own checks are still to be done.")

    # ---- 5 Conclusion ----------------------------------------------------------------------------------------------------------------
    add("h1", "5 Conclusion")
    took = f["new"]["took"]
    missing = took.split("Missing: ", 1)[1] if "Missing: " in took else took
    missing = re.sub(r"^about [^,]*? \(still [^)]*\), ", "", missing)                      # the false warnings are said in 3.4
    if FC["members"] and "the ensemble," in missing:
        missing = missing.replace("the ensemble,", f"the other {word(FC['members'] - 1)} ensemble members,", 1)
    add("p", f"{S0.FINDING}. A service would need radar rain up to the moment of the forecast and a weather forecast, as operational flood models use {cite('nearing2024')}, "
             f"yet the verdict stays {verdict}. Still missing: {missing[0].lower() + missing[1:]}")
    add("end_of_body")

    # ---- who did what, and the references -----------------------------------------------------------------------------------------------
    nw = S0.new_work(R)
    days = S0.run_day(R) + (f" and {nw['days']}" if nw else "")
    roles = s8_limits._plan()[0]
    clean = lambda o: re.sub(r" [(]week \d+[)]", "", o)
    owns = "; ".join(f"{team.label(m)}: {clean(o)[0].lower() + clean(o)[1:]}" for m, o in roles)
    own = ("The team members&rsquo; own checks that the plan assigns are still to be done." if team.checks_state()["n_done"] == 0 else S0._esc(cap1(team.checks_sentence())) + ".")
    add("contrib", f"<b>Author contributions and use of AI.</b> All code was written and all experiments were run by Claude Code, the team&rsquo;s coding assistant, under the team lead&rsquo;s direction ({days}). {S0.review_sentence()}. "
                   f"{own} &lsquo;We&rsquo; means the project, never a check made by a team member. Roles in the plan (<i>PLAN.md</i>): {owns}. Every number is read from "
                   f"<i>results/*.json</i>; the code and the full report (<i>Flood_Report.pdf</i>) are in the project folder.")
    add("back_h", "References")
    for i, ref in enumerate(references_html(R, G), 1):
        add("ref", ref, n=i)
    return B


def _years(a, b):
    """'1980&ndash;2014': the first year of the training part to the last of the test part (the split of results/lstm_all.json through flood.SPLIT)."""
    from flood import SPLIT
    return f"{SPLIT[a][0][:4]}&ndash;{SPLIT[b][1][:4]}"


def body_words(B):
    """Words from the title to the end of the conclusion, without tables, captions, notes and references; and the abstract and the sections alone."""
    total = abstract = sections = 0
    for b in B:
        if b["kind"] == "end_of_body":
            break
        if b["kind"] in ("table", "figure"):
            continue
        w = n_words(b["h"])
        total += w
        if b["kind"] == "abstract":
            abstract = w
        if b["kind"] in ("p", "h1", "h2"):
            sections += w
    return total, abstract, sections


def text_dump(B):
    for b in B:
        k = b["kind"]
        if k == "end_of_body":
            print("---- end of the conclusion ----\n")
        elif k == "table":
            print(f"[TABLE] {b['caption']}\n   notes: {plain(b.get('notes', ''))}\n")
        elif k == "figure":
            print(f"[FIGURE] {b['caption']}\n")
        else:
            print(f"[{k} {n_words(b['h'])}w] {plain(b['h'])}\n")


# ---- numbering of the figures and tables, in the order they stand --------------------------------------------------------------------
def numbered(B):
    """Give every figure and table its number in order of appearance (GB/T 7713.2-2022 5.2.4) and check that the text mentions each in that order."""
    nf = nt = 0
    for b in B:
        if b["kind"] == "figure":
            nf += 1
            b["n"] = nf
        elif b["kind"] == "table":
            nt += 1
            b["n"] = nt
    body = " ".join(plain(b["h"]) for b in B if b["kind"] in ("p", "abstract"))
    for word_, last in (("Figure", nf), ("Table", nt)):
        seen = [int(x) for x in re.findall(rf"{word_} (\d)", body)]
        order = list(dict.fromkeys(seen))
        if order != sorted(order) or (order and order[-1] != last):
            say(f"the text mentions the {word_.lower()}s in the order {order}, the {word_.lower()}s stand in the order 1 to {last}")
    return B


# ---- the HTML twin: CSS that mirrors the Word layout ---------------------------------------------------------------------------------
CSS_PAPER = """
@page{size:A4;margin:@MARGIN@mm;@bottom-center{content:counter(page);font:@BODY@pt 'Times New Roman',Times,'Liberation Serif',serif;color:#000}}
html,body{background:#fff;color:#000}
body{font:@BODY@pt/normal 'Times New Roman',Times,'Liberation Serif',serif;text-align:justify;margin:0;font-kerning:none;font-variant-ligatures:none;text-rendering:optimizeSpeed;hyphens:manual;orphans:2;widows:2}
p,h2,h3,figure,figcaption,div{margin:0}
p.title{font-size:@TITLE@pt;font-weight:bold;text-align:center;padding-bottom:6pt}
p.authors,p.affil{text-align:center}
p.affil{padding-bottom:8pt}
p.abs,p.kw{text-align:justify;font-size:@ABS@pt}
p.kw{padding-top:2pt}
h2.s1,h3.s2{font-weight:bold;text-align:left;break-after:avoid;line-height:normal}
h2.s1{font-size:@H1@pt;padding:@H1B@pt 0 .5pt}
h3.s2{font-size:@H2@pt;padding:@H2B@pt 0 .5pt}
p.b{text-indent:@INDENT@cm}
figure.fig{padding-top:3pt;break-inside:avoid;text-align:center}
figure.fig img{display:block;margin:0 auto}
.cap{font-size:@CAP@pt;font-weight:bold;text-align:center;padding:2pt 0 3pt;line-height:normal}
div.tblock{break-inside:avoid;padding-top:3pt}
p.tcap{font-size:@CAP@pt;font-weight:bold;text-align:center;padding-bottom:2pt;break-after:avoid;line-height:normal}
p.tnote{font-size:@NOTE@pt;line-height:normal;padding:2pt 0 3pt;text-align:justify;text-indent:.28cm}
sup{line-height:0;font-size:70%;vertical-align:baseline;position:relative;top:-.4em}
table.tt{border-collapse:collapse;table-layout:fixed;width:@TW@cm;font-size:@TAB@pt;margin:0}
table.tt th,table.tt td{border:0;padding:.6pt 2.5pt;line-height:normal;vertical-align:middle;text-align:center;font-weight:normal;color:#000;background:none;overflow-wrap:anywhere}
table.tt .l{text-align:left}
table.tt .top{border-top:@THICK@pt solid #000}
table.tt .hl,table.tt .grp{border-bottom:@THIN@pt solid #000}
table.tt .bot{border-bottom:@THICK@pt solid #000}
h2.back{font-size:@H1@pt;padding:@H1B@pt 0 .5pt}
p.contrib{font-size:@BACK@pt;text-indent:0;text-align:justify}
p.ref{font-size:@REF@pt;text-align:left;padding-left:.55cm;text-indent:-.55cm;overflow-wrap:anywhere}
p.ref .rn{display:inline-block;width:.55cm;text-indent:0}
figcaption,.cap,p.tcap,p.tnote,p.ref,p.contrib,p.abs,p.kw{color:#000}
.nb{white-space:nowrap}
.safety{height:@SAFETY@mm}
"""


def css_paper():
    sub = dict(MARGIN=MARGIN_MM, BODY=BODY_PT, TITLE=TITLE_PT, H1=H1_PT, H2=H2_PT, H1B=H1_BEFORE_PT, H2B=H2_BEFORE_PT, INDENT=INDENT_CM, CAP=CAP_PT, NOTE=NOTE_PT, TAB=TAB_PT, THICK=THICK_PT, THIN=THIN_PT,
               TW=TEXT_W_CM, BACK=BACK_PT, REF=REF_PT, ABS=ABS_PT, SAFETY=SAFETY_MM)
    out = CSS_PAPER
    for k, v in sub.items():
        out = out.replace(f"@{k}@", str(v))
    return out


_TMP = []


def fig_source(b):
    """The picture file of a figure block: figures/<name>, or for the flow figure the two rows of stages of it stacked into one picture (made once, in a temporary folder)."""
    if b["name"] == "data_flow_talk.png" and FLOW_BANDS:
        if not _TMP:
            import atexit
            import shutil
            from PIL import Image
            folder = tempfile.mkdtemp(prefix="paper_fig_")
            atexit.register(shutil.rmtree, folder, True)
            im = Image.open(FIG / b["name"]).convert("RGB")
            parts = [im.crop((0, top, im.width, bottom)) for top, bottom in FLOW_BANDS]
            gap = 10
            out = Image.new("RGB", (im.width, sum(x.height for x in parts) + gap * (len(parts) - 1)), im.getpixel((3, 3)))
            y = 0
            for x in parts:
                out.paste(x, (0, y))
                y += x.height + gap
            path = Path(folder) / "data_flow_stages.png"
            out.save(path, optimize=True)
            _TMP.append(path)
        return _TMP[0]
    return FIG / b["name"]


def png_size(path):
    with open(path, "rb") as fh:
        return struct.unpack(">II", fh.read(26)[16:24])


def fig_cm(b, plan):
    """(width, height) in cm of a figure block at the width of the plan."""
    w = plan[b["role"]]
    pw, ph = png_size(fig_source(b))
    return w, w * ph / pw


def img_of(path, alt):
    import base64
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}" alt="{_html.escape(alt)}">'


def table_html(tb):
    """A three-line table as HTML: thick rule above the header, thin rule under the header (and under a group heading), thick rule under the last row; no other rule."""
    rows = []
    for ri, row in enumerate(tb["head"]):
        last = ri == len(tb["head"]) - 1
        cells = ""
        for c in row:
            cls = ["l" if c["align"] == "l" else ""]
            cls += ["top"] if ri == 0 else []
            cls += ["hl"] if last or c["rspan"] > 1 else []
            cls += ["grp"] if c["under"] else []
            cells += (f'<th class="{" ".join(x for x in cls if x)}"' + (f' colspan="{c["span"]}"' if c["span"] > 1 else "") + (f' rowspan="{c["rspan"]}"' if c["rspan"] > 1 else "")
                      + f'>{c["h"]}</th>')
        rows.append(f"<tr>{cells}</tr>")
    for ri, row in enumerate(tb["rows"]):
        last = ri == len(tb["rows"]) - 1
        cells = ""
        for c in row:
            cls = (("l " if c["align"] == "l" else "") + ("bot" if last else "")).strip()
            cells += f'<td class="{cls}">{c["h"]}</td>'
        rows.append(f"<tr>{cells}</tr>")
    cols = "".join(f'<col style="width:{w}cm">' for w in tb["widths"])
    return f'<table class="tt {tb.get("cls", "")}"><colgroup>{cols}</colgroup>{"".join(rows)}</table>'


def _nowrap_dates(html):
    """Dates written with U+2011 become one unbreakable word in the print (Word gets w:noBreakHyphen)."""
    return re.sub(r"\d{4}\u2011\d{2}\u2011\d{2}(?:/\d{2})?", lambda m: f'<span class="nb">{m.group(0).replace(NBH, "-")}</span>', html)


def block_html(b, plan):
    """One block of the paper as HTML."""
    return _nowrap_dates(_block_html(b, plan))


def _block_html(b, plan):
    k, h = b["kind"], b["h"]
    if k == "title":
        return f'<p class="title">{h}</p>'
    if k in ("authors", "affil"):
        return f'<p class="{k}">{h}</p>'
    if k == "abstract":
        return f'<p class="abs"><b>Abstract:</b> {h}</p>'
    if k == "keywords":
        return f'<p class="kw"><b>Keywords:</b> {h}</p>'
    if k == "h1":
        return f'<h2 class="s1">{h}</h2>'
    if k == "h2":
        return f'<h3 class="s2">{h}</h3>'
    if k == "p":
        return f'<p class="b">{h}</p>'
    if k == "figure":
        w, hgt = fig_cm(b, plan)
        pic = img_of(fig_source(b), b["alt"]).replace("<img ", f'<img style="width:{w}cm" ', 1)
        return f'<figure class="fig">{pic}<figcaption class="cap">Figure {b["n"]}{GAP}{b["caption"]}</figcaption></figure>'
    if k == "table":
        return f'<div class="tblock"><p class="tcap">Table {b["n"]}{GAP}{b["caption"]}</p>{table_html(b["table"])}<p class="tnote">{b["notes"]}</p></div>'
    if k == "back_h":
        return f'<h2 class="s1 back">{h}</h2>'
    if k == "contrib":
        return f'<p class="contrib">{h}</p>'
    if k == "ref":
        return f'<p class="ref"><span class="rn">[{b["n"]}]</span>{h}</p>'
    return ""


def to_html(B, plan):
    """The whole paper as the body of one HTML page, for a plan {'key': cm, 'flow': cm}."""
    return "".join(block_html(b, plan) for b in B) + '<div class="safety"></div>'


def render(B, plan):
    return page("Predicting river floods with AI", to_html(B, plan), css_extra=css_paper())


# ---- fitting: exactly three pages ------------------------------------------------------------------------------------------------------
def pages_of(B, plan, keep=None):
    """The number of pages of the print of the paper at a plan, or None without a browser. keep: a path for the PDF."""
    with tempfile.TemporaryDirectory() as folder:
        src = Path(folder) / "paper.html"
        src.write_text(render(B, plan), encoding="utf-8")
        pdf = Path(keep) if keep else src.with_suffix(".pdf")
        if not print_pdf(src, pdf, quiet=True):
            return None
        return pdf_pages(pdf)


def fit_plan(B, floor=FIG_FLOOR_CM, verbose=True):
    """The plan {'key': cm, 'flow': cm} with the widest figures at which the paper prints as exactly three pages (SAFETY_MM to spare on the last one), or None. The flow figure
    is the one whose type is smallest, so it is kept as wide as possible first (16 cm); then the key figure is made as wide as the pages allow (not narrower than `floor`).
    Widths run from 16 cm down in steps of 0.5 cm; a print is a few seconds, so the key figure is found by bisection."""
    cache = {}

    def fits(kw, fw):
        key = (kw, fw)
        if key not in cache:
            cache[key] = pages_of(B, {"key": kw, "flow": fw})
            if verbose:
                print(f"  key figure {kw:g} cm, flow figure {fw:g} cm: {cache[key]} pages", file=sys.stderr)
        return cache[key] is not None and cache[key] <= 3

    widths = [w for w in FIG_CM if w >= floor]                                   # widest first
    if pages_of(B, {"key": widths[-1], "flow": widths[-1]}) is None:
        return None                                                              # no browser
    for fw in widths:                                                            # the widest flow figure for which some key figure fits
        if not fits(widths[-1], fw):
            continue
        lo, hi = 0, len(widths) - 1                                              # widths[lo] is the widest key figure that fits
        while lo < hi:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if fits(widths[mid], fw) else (mid + 1, hi)
        return {"key": widths[lo], "flow": fw}
    return None


def search_layout(B0, verbose=True):
    """Try every placement of the four floats (ANCHORS) and return (placement, plan) with the widest figures that print as exactly three pages, or None. Slow (one print per
    placement and a width search for those that fit): used only when the stored PLACEMENT no longer fits."""
    best = None
    floor = (FIG_FLOOR_CM, FIG_FLOOR_CM)
    for pl in placements():
        B = numbered(arrange(B0, pl))
        n = pages_of(B, {"key": floor[0], "flow": floor[1]})
        if verbose:
            print(f"  placement {pl}: {n} pages at the narrowest figures", file=sys.stderr)
        if n is None:
            return None
        if n > 3:
            continue
        plan = fit_plan(B, verbose=False)
        if plan and (best is None or (plan["key"], plan["flow"]) > (best[1]["key"], best[1]["flow"])):
            best = (pl, plan)
    return best


DEFAULT_PLAN = {"key": 12.5, "flow": 16.0}                                      # figure widths in cm for the Word file when there is no browser to fit them with


def write_outputs(B, plan, pdf=True):
    """The PDF twin (printed from the HTML of the same blocks) and the Word file, at the root of the project. Returns (pages of the PDF or None, problems found in the Word file)."""
    n = None
    if pdf:
        out = ROOT / f"{NAME}.pdf"
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / f"{NAME}.html"
            src.write_text(render(B, plan), encoding="utf-8")
            if print_pdf(src, out):
                n = pdf_pages(out)
    try:
        import docx                                                             # noqa: F401  (python-docx is in requirements.txt since 2026-10-28)
    except ImportError:
        print("build_report_paper: python-docx is not installed: the Word file is not written (pip install python-docx)", file=sys.stderr)
        return n, []
    path = ROOT / f"{NAME}.docx"
    to_docx(B, plan, path)
    return n, verify_docx(path, B)


def main():
    text_only = "--text" in sys.argv
    R, T = load_results()
    G = gather_paper(R, T)
    if G is None:
        sys.exit(1)
    B0 = blocks(R, T, G)
    place = dict(PLACEMENT)
    B = numbered(arrange(B0, place))
    total, abstract, sections = body_words(B)
    if text_only:
        text_dump(B)
        print(f"words, title to the end of the conclusion: {total} (abstract {abstract}; sections 1 to 5 {sections})")
        return
    print(f"body words (title to the end of the conclusion, without tables, captions, notes and references): {total}; abstract {abstract}; sections 1-5 {sections}")
    if not 150 <= abstract <= 200:
        print(f"build_report_paper: the abstract has {abstract} words, not 150 to 200", file=sys.stderr)
        sys.exit(1)
    browser = pages_of(B, {"key": FIG_FLOOR_CM, "flow": FIG_FLOOR_CM}) is not None
    if not browser:
        print("build_report_paper: no Chromium browser: no PDF; the Word file gets the default figure widths", file=sys.stderr)
        plan = dict(DEFAULT_PLAN)
    else:
        plan = fit_plan(B)
        if plan is None:
            print("build_report_paper: the stored placement of the floats no longer fits; trying every placement ...", file=sys.stderr)
            found = search_layout(B0)
            if found:
                place, plan = found
                B = numbered(arrange(B0, place))
                print(f"build_report_paper: placement found {place}: set PLACEMENT to it to skip this search", file=sys.stderr)
        if plan is None:
            print("build_report_paper: no plan prints as three pages: shorten the text", file=sys.stderr)
            sys.exit(1)
    n, problems = write_outputs(B, plan, pdf=browser)
    print(f"plan (figure widths in cm): {plan}; pages of the PDF: {n}")
    for line in problems:
        print("build_report_paper: Word file check:", line, file=sys.stderr)
    if (browser and n != 3) or problems:
        print(f"build_report_paper: the PDF has {n} pages, not 3, or the Word file differs from the text" if n != 3 else "build_report_paper: the Word file differs from the text", file=sys.stderr)
        sys.exit(1)
    print("wrote", ", ".join(str(ROOT / f"{NAME}.{ext}") for ext in (("pdf",) if browser else ()) + ("docx",) if (ROOT / f"{NAME}.{ext}").exists()) or "nothing")


# ---- where the floats stand ----------------------------------------------------------------------------------------------------------------
FLOATS = ("T1", "F1", "T2", "F2")                                              # Table 1, Figure 1, Table 2, Figure 2, in this order where they share a place
ANCHORS = {"T1": ("data1", "data2"), "F1": ("data2", "model", "test"), "T2": ("info", "rivers", "warn"), "F2": ("info", "rivers", "warn")}   # after which paragraph each may stand
PLACEMENT = {"T1": "data1", "F1": "test", "T2": "warn", "F2": "info"}          # found by search_layout(): Table 1 after the paragraph on CAMELS-US, Figure 1 after the fair test,
                                                                               # Figure 2 after the paragraph on the information, Table 2 after the warnings


def arrange(B, place):
    """The blocks with every float (a block with a 'fid') moved to stand right after the paragraph named in `place`."""
    floats = {b["fid"]: b for b in B if b.get("fid")}
    out = []
    for b in B:
        if b.get("fid"):
            continue
        out.append(b)
        for fid in FLOATS:
            if place[fid] == b.get("id"):
                out.append(floats[fid])
    return out


def placements():
    import itertools
    for combo in itertools.product(*(ANCHORS[f] for f in FLOATS)):
        yield dict(zip(FLOATS, combo))


# ---- the Word file -------------------------------------------------------------------------------------------------------------------------
TNR, EAST = "Times New Roman", "SimSun"
TWIP_CM = 567.0                                                                    # twips per cm


def _fonts_of(rpr):
    """Times New Roman for Latin text (SimSun for East Asian characters) in a run-properties element; the theme fonts of the template are removed."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in list(rf.attrib):
        del rf.attrib[a]
    for a, v in (("w:ascii", TNR), ("w:hAnsi", TNR), ("w:cs", TNR), ("w:eastAsia", EAST)):
        rf.set(qn(a), v)


def _style(doc, name, base="Normal", size=BODY_PT, bold=False, italic=False, align=None, before=0.0, after=0.0, first=None, left=None, hanging=None, keep_next=False, keep_lines=False):
    """Create or reset a paragraph style: Times New Roman, black, single line spacing, the given sizes in pt and indents in cm."""
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_LINE_SPACING
    styles = doc.styles
    try:
        st = styles[name]
    except KeyError:
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    if base and name != base:
        st.base_style = styles[base]
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.italic = italic
    st.font.color.rgb = RGBColor(0, 0, 0)
    rpr = st.element.get_or_add_rPr()
    _fonts_of(rpr)
    for tag in ("w:color",):
        for el in rpr.findall(f"{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}{tag[2:]}"):
            for a in ("themeColor", "themeShade", "themeTint"):
                el.attrib.pop(f"{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}{a}", None)
    pf = st.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.alignment = align
    pf.first_line_indent = None if first is None else Cm(first)
    pf.left_indent = None if left is None else Cm(left)
    if hanging is not None:
        pf.left_indent, pf.first_line_indent = Cm(hanging), Cm(-hanging)
    pf.keep_with_next, pf.keep_together = keep_next, keep_lines
    pf.widow_control = True
    ppr = st.element.get_or_add_pPr()
    for el in ppr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pBdr"):
        ppr.remove(el)                                                           # the template's Title has a blue rule
    for el in ppr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}contextualSpacing"):
        ppr.remove(el)
    for el in rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}spacing") + rpr.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}kern"):
        rpr.remove(el)
    return st


def _prepare(doc):
    """A4, margins 2.5 cm, the styles of the paper, the footer with the page number; Word 2013 mode."""
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(MARGIN_MM / 10)
    sec.header_distance, sec.footer_distance = Cm(1.25), Cm(1.25)
    # defaults of the document
    sty = doc.styles.element
    dd = sty.find(qn("w:docDefaults"))
    rpr = dd.find(qn("w:rPrDefault")).find(qn("w:rPr"))
    _fonts_of(rpr)
    for tag, val in (("w:sz", str(int(BODY_PT * 2))), ("w:szCs", str(int(BODY_PT * 2)))):
        el = rpr.find(qn(tag))
        el.set(qn("w:val"), val)
    lang = rpr.find(qn("w:lang"))
    lang.set(qn("w:val"), "en-US")
    lang.set(qn("w:eastAsia"), "zh-CN")
    sp = dd.find(qn("w:pPrDefault")).find(qn("w:pPr")).find(qn("w:spacing"))
    sp.set(qn("w:after"), "0")
    sp.set(qn("w:line"), "240")
    sp.set(qn("w:lineRule"), "auto")
    # every style of the template: Times New Roman, no theme fonts
    for st in sty.findall(qn("w:style")):
        r = st.find(qn("w:rPr"))
        if r is not None:
            rf = r.find(qn("w:rFonts"))
            if rf is not None:
                _fonts_of(r)
    _style(doc, "Normal", base=None, size=BODY_PT)
    _style(doc, "Title", size=TITLE_PT, bold=True, align=AL.CENTER, after=6, keep_next=True)
    _style(doc, "Paper Authors", size=BODY_PT, align=AL.CENTER, keep_next=True)
    _style(doc, "Paper Affiliation", size=BODY_PT, align=AL.CENTER, after=8, keep_next=True)
    _style(doc, "Abstract", size=ABS_PT, align=AL.JUSTIFY)
    _style(doc, "Keywords", size=ABS_PT, align=AL.JUSTIFY, before=2)
    _style(doc, "Heading 1", size=H1_PT, bold=True, align=AL.LEFT, before=H1_BEFORE_PT, after=0.5, keep_next=True, keep_lines=True)
    _style(doc, "Heading 2", size=H2_PT, bold=True, align=AL.LEFT, before=H2_BEFORE_PT, after=0.5, keep_next=True, keep_lines=True)
    _style(doc, "Body Text", size=BODY_PT, align=AL.JUSTIFY, first=INDENT_CM)
    _style(doc, "Figure", size=1, align=AL.CENTER, before=3, keep_next=True)                   # 1 pt: no font descent below the picture
    _style(doc, "Caption", size=CAP_PT, bold=True, align=AL.CENTER, before=2, after=3, keep_lines=True)
    _style(doc, "Table Caption", base="Caption", size=CAP_PT, bold=True, align=AL.CENTER, before=3, after=2, keep_next=True, keep_lines=True)
    _style(doc, "Table Text", size=TAB_PT, align=AL.CENTER, before=0.6, after=0.6)
    _style(doc, "Table Note", size=NOTE_PT, align=AL.JUSTIFY, before=2, after=3, first=0.28, keep_lines=True)
    _style(doc, "Reference", size=REF_PT, align=AL.LEFT, hanging=0.55)
    _style(doc, "Contributions", size=BACK_PT, align=AL.JUSTIFY, before=H1_BEFORE_PT)
    # Word 2013 mode (the template says Word 2010) and the East Asian language of the theme
    settings = doc.settings.element
    for cs in settings.iter(qn("w:compatSetting")):
        if cs.get(qn("w:name")) == "compatibilityMode":
            cs.set(qn("w:val"), "15")
    for tf in settings.iter(qn("w:themeFontLang")):
        tf.set(qn("w:eastAsia"), "zh-CN")
    for z in settings.iter(qn("w:zoom")):
        z.attrib.pop(qn("w:val"), None)
        z.set(qn("w:percent"), "100")
    # the footer: the page number, centred
    ft = sec.footer
    ft.is_linked_to_previous = False
    p = ft.paragraphs[0]
    p.style = doc.styles["Footer"]
    for el in p._p.pPr.findall(qn("w:tabs")) if p._p.pPr is not None else []:
        p._p.pPr.remove(el)
    p.alignment = AL.CENTER
    for kind, text in (("begin", None), (None, " PAGE "), ("separate", None), (None, "1"), ("end", None)):
        run = p.add_run()
        run.font.size = Pt(BODY_PT)
        if kind:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            run._r.append(fc)
        elif text.strip() == "PAGE":
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            run._r.append(it)
        else:
            run.text = text
    _style(doc, "Footer", size=BODY_PT, align=AL.CENTER)


def _runs(par, fragment, size=None, bold=False):
    """Write an HTML fragment into a paragraph as runs (bold, italic, superscript, subscript). U+2011 (the unbreakable hyphen of the dates in the print) is written as a plain hyphen:
    Word's own w:noBreakHyphen is drawn by some viewers (Apple's Quick Look) as nothing, and a date that breaks at the end of a line does no harm."""
    from docx.shared import Pt
    for text, b, i, sv in runs_of(fragment):
        for k, part in enumerate(text.replace(NBH, "-").split("\n")):
            if k:
                par.add_run().add_break()
            if not part:
                continue
            r = par.add_run(part)
            r.bold = True if (b or bold) else None
            r.italic = True if i else None
            if sv == "sup":
                r.font.superscript = True
            elif sv == "sub":
                r.font.subscript = True
            if size:
                r.font.size = Pt(size)


def _border(tc, top=None, bottom=None):
    """Cell borders in eighths of a point, black, no space: a three-line table has only these."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    pr = tc.get_or_add_tcPr()
    for old in pr.findall(qn("w:tcBorders")):
        pr.remove(old)
    box = OxmlElement("w:tcBorders")
    for side, pt in (("top", top), ("bottom", bottom)):
        if pt:
            el = OxmlElement(f"w:{side}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(int(round(pt * 8))))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            box.append(el)
    if len(box):
        # tcBorders goes after tcW, gridSpan, vMerge and before shd, noWrap, tcMar, vAlign
        after = [pr.find(qn(t)) for t in ("w:tcW", "w:gridSpan", "w:hMerge", "w:vMerge")]
        anchor = next((el for el in reversed(after) if el is not None), None)
        if anchor is not None:
            anchor.addnext(box)
        else:
            pr.insert(0, box)


def _table(doc, tb):
    """A three-line table: fixed layout, the column widths of the spec, merged header cells, rules only at the top, under the header (and the group headings) and at the bottom."""
    from docx.shared import Cm, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT as VA, WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    widths = tb["widths"]
    nh, nb = len(tb["head"]), len(tb["rows"])
    table = doc.add_table(rows=nh + nb, cols=len(widths))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl = table._tbl
    pr = tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout", "w:tblCellMar", "w:tblBorders"):
        for el in pr.findall(qn(tag)):
            pr.remove(el)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(round(sum(widths) * TWIP_CM)))
    tw.set(qn("w:type"), "dxa")
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 0), ("left", 50), ("bottom", 0), ("right", 50)):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(w))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    pr.append(tw)
    pr.append(lay)
    pr.append(mar)
    order = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd",
             "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]
    kids = sorted(list(pr), key=lambda el: order.index(el.tag.split("}")[1]) if el.tag.split("}")[1] in order else len(order))
    for el in kids:
        pr.remove(el)
    for el in kids:
        pr.append(el)
    for i, w in enumerate(widths):
        table.columns[i].width = Cm(w)
    # place the cells (spans and row spans), remember which spec covers which grid position
    owner = {}
    rows_spec = tb["head"] + tb["rows"]
    for r, row in enumerate(rows_spec):
        c = 0
        for spec in row:
            while (r, c) in owner:
                c += 1
            r2, c2 = r + spec["rspan"] - 1, c + spec["span"] - 1
            for rr in range(r, r2 + 1):
                for cc in range(c, c2 + 1):
                    owner[(rr, cc)] = (spec, r, c, r2, c2)
            first = table.cell(r, c)
            if (r2, c2) != (r, c):
                first = first.merge(table.cell(r2, c2))
            par = first.paragraphs[0]
            par.style = doc.styles["Table Text"]
            par.alignment = AL.LEFT if spec["align"] == "l" else AL.CENTER
            _runs(par, spec["h"])
            first.vertical_alignment = VA.CENTER
            c = c2 + 1
    last_head, last_row = nh - 1, nh + nb - 1
    # widths of every w:tc, borders, rows that do not split, keep the table in one piece
    for r, tr in enumerate(tbl.tr_lst):
        trpr = tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trpr.append(cs)
        c = 0
        for tc in tr.tc_lst:
            span = tc.grid_span
            spec, r0, c0, r2, c2 = owner[(r, c)]
            tc.get_or_add_tcPr().get_or_add_tcW().w = round(sum(widths[c:c + span]) * TWIP_CM)
            tc.get_or_add_tcPr().get_or_add_tcW().type = "dxa"
            top = THICK_PT if r == 0 else None
            bottom = THICK_PT if r == last_row else THIN_PT if (r == last_head or (spec["under"] and r == r2)) else None
            _border(tc, top=top, bottom=bottom)
            c += span
        for tc in tr.tc_lst:                                                    # every row keeps with the next, the last with the notes: the caption, the table and its notes stay together
            for p in tc.iterchildren(qn("w:p")):
                ppr = p.get_or_add_pPr()
                kn = OxmlElement("w:keepNext")
                ppr.insert(1 if ppr.find(qn("w:pStyle")) is not None else 0, kn)
    return table


def _picture(doc, b, plan):
    """The figure as one inline picture in a centred paragraph, with its alt text, and the caption below."""
    from docx.shared import Cm
    w, h = fig_cm(b, plan)
    p = doc.add_paragraph(style="Figure")
    shape = p.add_run().add_picture(str(fig_source(b)), width=Cm(w), height=Cm(h))
    inline = shape._inline
    inline.docPr.set("descr", b["alt"])
    inline.docPr.set("title", f"Figure {b['n']}")
    cap = doc.add_paragraph(style="Caption")
    _runs(cap, f"Figure {b['n']}{GAP}{b['caption']}")
    return p


def to_docx(B, plan, path):
    """The paper as a Word file at `path`, from the same blocks as the print."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
    import datetime
    doc = Document()
    _prepare(doc)
    body = doc.element.body
    for p in list(doc.paragraphs):
        body.remove(p._p)
    for b in B:
        k, h = b["kind"], b["h"]
        if k == "title":
            _runs(doc.add_paragraph(style="Title"), h)
        elif k == "authors":
            _runs(doc.add_paragraph(style="Paper Authors"), h)
        elif k == "affil":
            _runs(doc.add_paragraph(style="Paper Affiliation"), h)
        elif k == "abstract":
            _runs(doc.add_paragraph(style="Abstract"), f"<b>Abstract:</b> {h}")
        elif k == "keywords":
            _runs(doc.add_paragraph(style="Keywords"), f"<b>Keywords:</b> {h}")
        elif k == "h1" or k == "back_h":
            _runs(doc.add_paragraph(style="Heading 1"), h)
        elif k == "h2":
            _runs(doc.add_paragraph(style="Heading 2"), h)
        elif k == "p":
            _runs(doc.add_paragraph(style="Body Text"), h)
        elif k == "figure":
            _picture(doc, b, plan)
        elif k == "table":
            cap = doc.add_paragraph(style="Table Caption")
            _runs(cap, f"Table {b['n']}{GAP}{b['caption']}")
            _table(doc, b["table"])
            _runs(doc.add_paragraph(style="Table Note"), b["notes"])
        elif k == "contrib":
            _runs(doc.add_paragraph(style="Contributions"), h)
        elif k == "ref":
            par = doc.add_paragraph(style="Reference")
            par.paragraph_format.tab_stops.add_tab_stop(__import__("docx.shared", fromlist=["Cm"]).Cm(0.55))
            _runs(par, f"[{b['n']}]\t{h}")
    cp = doc.core_properties
    cp.title = plain(B[0]["h"])
    cp.subject = "AI course group project: predicting river floods with an LSTM, 12 CAMELS-US basins"
    cp.author = "Project team (placeholders from team.json); built by code/build_report_paper.py"
    cp.last_modified_by = "code/build_report_paper.py"
    cp.keywords = "; ".join(KEYWORDS)
    cp.comments = "Written and run by Claude Code under the team lead's direction; the team members' own checks are still to be done."
    cp.language = "en-US"
    cp.created = cp.modified = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, microsecond=0)
    # the template's thumbnail shows another page: drop it
    pkg = doc.part.package
    for rid, rel in list(pkg.rels.items()):
        if rel.reltype.endswith("/thumbnail"):
            del pkg.rels[rid]
    doc.save(path)


def verify_docx(path, B):
    """Re-open the Word file with python-docx and check it against the blocks it was written from: the text of every paragraph, two pictures, two tables, the headings, the page number
    field in the footer, A4 with margins of 2.5 cm, Times New Roman at 10.5 pt. Returns a list of problems (empty when everything agrees) and prints the facts."""
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(path)
    problems = []
    paras = [p for p in doc.paragraphs]
    text = "\n".join(p.text for p in paras)
    expected = []
    for b in B:
        if b["kind"] in ("figure", "table", "end_of_body"):
            continue
        t = plain(b["h"])
        if b["kind"] == "ref":
            t = f"[{b['n']}]\t{t}"
        elif b["kind"] == "abstract":
            t = "Abstract: " + t
        elif b["kind"] == "keywords":
            t = "Keywords: " + t
        expected.append(t)
    got = [re.sub(r"\s+", " ", p.text.replace("\u200b", "").replace(NB, " ")).strip() for p in paras]
    for t in expected:
        t2 = re.sub(r"\s+", " ", t.replace("\t", " ")).strip()
        if not any(g.replace("\t", " ") == t2 for g in got):
            problems.append(f"paragraph missing or changed: {t[:70]!r}")
    heads = [p.text for p in paras if p.style.name.startswith("Heading")]
    want_heads = [plain(b["h"]) for b in B if b["kind"] in ("h1", "h2", "back_h")]
    if heads != want_heads:
        problems.append(f"headings differ: {heads} != {want_heads}")
    n_img, n_tab = len(doc.inline_shapes), len(doc.tables)
    if n_img != 2:
        problems.append(f"{n_img} pictures, not 2")
    if n_tab != 2:
        problems.append(f"{n_tab} tables, not 2")
    caps = [p.text for p in paras if p.style.name in ("Caption", "Table Caption")]
    want_caps = [f"{'Table' if b['kind'] == 'table' else 'Figure'} {b['n']}{GAP}{b['caption']}" for b in B if b["kind"] in ("table", "figure")]
    norm = lambda x: re.sub(r"\s+", " ", x.replace(NB, " ")).strip()
    if sorted(norm(c) for c in caps) != sorted(norm(plain(w)) for w in want_caps):
        problems.append(f"captions differ: {caps}")
    sec = doc.sections[0]
    cm = lambda x: round(x.cm, 2)
    page = (cm(sec.page_width), cm(sec.page_height), cm(sec.left_margin), cm(sec.right_margin), cm(sec.top_margin), cm(sec.bottom_margin))
    if page != (21.0, 29.7, 2.5, 2.5, 2.5, 2.5):
        problems.append(f"page set-up {page}")
    foot = sec.footer._element.xml
    if "PAGE" not in foot or "fldChar" not in foot:
        problems.append("no page number field in the footer")
    normal = doc.styles["Normal"]
    if (normal.font.size.pt, normal.font.name) != (BODY_PT, TNR):
        problems.append(f"Normal is {normal.font.name} {normal.font.size.pt} pt")
    for ts in doc.tables:                                                         # three-line tables: no vertical rules, no other horizontal rule than top, header, bottom
        xml = ts._tbl.xml
        if "w:left" in xml.split("<w:tcBorders>")[-1][:0] or any(f"<w:{side} " in xml for side in ("left", "right", "insideV", "insideH")) and "<w:tcBorders>" not in xml:
            problems.append("a table has rules other than the three")
    rules = [(len(t._tbl.xpath(".//w:tcBorders")), len(t._tbl.xpath(".//w:tcBorders/w:top")), len(t._tbl.xpath(".//w:tcBorders/w:bottom"))) for t in doc.tables]
    print(f"docx: {len(paras)} paragraphs, {len(heads)} headings, {n_img} pictures, {n_tab} tables, footer page field {'yes' if 'PAGE' in foot else 'NO'}, A4 {page[:2]} cm, margins {page[2:]} cm; "
          f"tcBorders per table (cells, top rules, bottom rules): {rules}")
    return problems


if __name__ == "__main__":
    main()
