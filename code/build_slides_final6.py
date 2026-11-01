"""Build the CONCISE final deck of Project-Flood: 6 slides, 10 minutes, four speakers (hand-in item 4, talk on 2 November 2026).

    python -X utf8 code/build_slides_final6.py [--out <pptx>] [--notes <md>] [--cover sample|photo] [--pdf-dir <dir>] [--png-dir <dir>]

    from the project root; system Python 3.13 with python-pptx 1.0.2, Pillow and lxml; pyproj places the state names of the map;
    --pdf-dir renders the deck with LibreOffice (a private profile in the temp folder replaces the theme font Aptos, which this PC
    lacks, by Arial, the closest installed face; without it LibreOffice falls back to Arial Black and DejaVu Sans); --png-dir writes
    one 90-dpi PNG per page of that PDF (PyMuPDF). The builder refuses to overwrite a PPTX that WPS or PowerPoint holds open (an
    owner file "~$<name>" beside it): pass another --out.

Format: the professor's "Sample ppt.pptx" (copied by build_slides_final.prepare_template to code/final_report/deck_template.pptx)
  - its slide master, both layouts, the theme (Aptos / Aptos Display) and the logo are kept untouched;
  - slide 1 IS the sample's cover slide (layout 'Blank'), edited in place: full-bleed campus photo with its saturation effect, white
    veil (45 %), white band (55 %), logo top-left, the title box (Microsoft YaHei bold, spacing 3 pt, 035CBA, shadow) and the members
    box (a blue YaHei line for the group, Calibri bold italic D97C27 for the names); the sizes come down from 24/32 pt to fit the full
    title and four names with student IDs; the question and the date line are added under the band; the sample's slides 2-4 go;
    --cover photo instead rebuilds the cover of build_slides_final.py (USGS flood photo under the same veil and band);
  - slides 2-6 sit on layout 2 (blue top band, chevron and logo, bottom band with the 'Group #' text, which reads registry
    group_number): the sample's heading box (1244600, 1) 9753600 x 889847 in white bold (28 pt, one line), the slide number at the
    top right (26.67 pt white), the 16 pt bold 1B2B4B sub-heading under the band, the 13.33 pt 64748B footer line in the sample's
    slot at the bottom right (here: the presenter); no font is set on slides 2-6, so the theme font applies.
Mid-term features (proposal/web/index.html and its PPTX), drawn as native, editable shapes in the content area under the sub-heading:
  the orange upper-case kicker line, the basin map redrawn from code/proposal_web/assets/map/basin_map.json in the three landscape
  colours, the funnel 671 -> 78 -> 63 -> 12 as chevrons, the fair-split bar, the issue-time clock bar, the big-number card, the
  onsets-warned ladder with the dashed GO line, the TARGET · GO tiles (mid-term -> now -> target) and the closing line with the
  orange rule; palette of the mid-term (navy 0B2545, blue 2A78D6, orange EB6834, green 1BAF7A; landscapes 4A3AA7 / 008300 / E87BA4).

Honesty rules (HANDOFF.md section 7): every number on a slide or in the notes is read from results/final_report_numbers.json (built
by code/final_report/numbers.py; keys in code/final_report/NUMBERS_KEYS.md) through N()/V(); the fixed wordings are the fw_ keys and
stand beside their numbers (AORC = fw_aorc, beside the AORC share fw_hindsight; Stage IV = fw_realtime_short on the slides and
fw_realtime in the "If asked" line; perfect weather / rain = fw_upper; numbers of the satellite-era split "comparable only inside
it"); "we" means the project; fw_who_did_what, the independent AI review and fw_team_checks are said on slide 6. The only numerals
typed here are structural: slide numbers, the seconds of the talk plan, the pace rule, the hours of the clock axis (00:00, 24:00:
the day itself; 09:00 and 17:30 are read from the registry), the lead "a day ahead" / "2 and 3 days" (flood.py LEADS = 1, 2, 3),
the 600 m of the mountain landscape name, and geometry. Nothing is trained or computed here.

Checks (validate): 6 slides; slide 1 on 'Blank', slides 2-6 on layout 2; 'Group #__' in layout 2 and on the cover; every shape
(group children too) inside the slide; notes on every slide; no forbidden token; no run under 10 pt except the source lines (shape
names starting "Source"); text that may overflow its box (estimated with Arial metrics plus 6 %, the PowerPoint/WPS/LibreOffice
faces of Aptos); the fixed wordings beside the flagged shares, unit by unit (a paragraph, a whole group, a notes paragraph); the pace
of the notes (at most PACE words a second, the 'If asked' lines not spoken).
"""
import argparse
import copy
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lxml import etree
from pptx import Presentation          # pptx first: code/final_report/numbers.py would shadow the stdlib module 'numbers'
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_PATTERN
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_slides_final as BSF       # noqa: E402  template handling, registry, report reference numbers (not modified)

ROOT = BSF.ROOT
REGISTRY = BSF.REGISTRY
HANDIN = BSF.HANDIN
OUT_PPTX = HANDIN / "4_Final_Project_Presentation_Flood.pptx"
OUT_NOTES = HANDIN / "4_Final_Project_Presentation_notes.md"
MAP_JSON = ROOT / "code" / "proposal_web" / "assets" / "map" / "basin_map.json"
SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
PACE = 2.3                                     # spoken words a second at most (the talk plan of results/slide_plan_5.json)

# ---- geometry (inches; the slide is 13.333 x 7.5 in, 16:9) -------------------------------------------------------------------
IN = 914400
SLIDE_W, SLIDE_H = 12192000, 6858000
HEAD = (1244600, 1, 9753600, 889847)          # the sample's heading box (layout 2 band: 0 to 791845)
HEAD_PT = 28
X0, X1 = 0.60, 12.73                          # content area left / right (the sample's sub-heading starts at 548640 = 0.60 in)
KICK_Y, SUB_Y = 0.93, 1.20
TOP, BOTTOM = 1.74, 6.36                       # content area under the sub-heading, above the footer lines
HON_BOX = (0.60, 6.41, 12.13, 0.40)           # honesty line(s), 10.5 pt, bottom-anchored above the presenter line
PRES_BOX = (6705600, 6217920, 4998720, 243840)  # the sample's footer slot (13.33 pt 64748B): the presenter
MEAS_SAFETY = 1.06                            # Arial metrics x 1.06 stand in for Aptos (PowerPoint), its WPS face and Arial (LibreOffice)

# ---- palette: the sample's inks and the mid-term colours ----------------------------------------------------------------------
WHITE, SUBNAVY, GREY = "FFFFFF", "1B2B4B", "64748B"            # the sample's sub-heading navy and footer grey
DEEP, DEEP2, BLUE, BLUE_INK, BLUE_LT, BLUE_BAR = "0B2545", "13315C", "2A78D6", "1C5CAB", "9EC5F4", "86B6EF"
PANEL, PANEL2 = "E8F0FA", "D6E4F6"
ORANGE, ORANGE_INK, ORANGE_T, ORANGE_LINE = "EB6834", "B54A1E", "FDF0EA", "F3C7B3"
AQUA, AQUA_T, AQUA_INK = "1BAF7A", "E3F5ED", "0E7A53"
GOOD, CRIT, CRIT_T = "0CA30C", "D03B3B", "FBE9E9"
HATCH_FG, HATCH_BG, INK2 = "C9C9C2", "F4F4F1", "52514E"
LAND_C = {"m": "4A3AA7", "n": "008300", "s": "E87BA4"}           # mid-term landscapes: mountains, northern lowlands, southern lowlands
LAND_FILL, COAST = "E8EFF8", "9FB3C8"
TITLE_BLUE, BAND_BLUE, NAME_ORANGE = "035CBA", "005BAC", "D97C27"  # the sample cover's inks

FORBIDDEN = ("UNVERIFIED", "MISSING", "TODO", "{{", "}}", "None")
PROBLEMS = []                                  # failed checks (fatal)
WARN = []                                      # warnings (listed in the notes file)
CUR = {"n": 0}


def E(v):
    return Emu(int(round(v * IN)))


def warn(msg):
    WARN.append(msg)
    print("WARNING: " + msg, file=sys.stderr)


def problem(msg):
    PROBLEMS.append(msg)
    print("PROBLEM: " + msg, file=sys.stderr)


# ---- text measurement (Arial x MEAS_SAFETY; Calibri and Microsoft YaHei for the cover) ----------------------------------------
_FONT_FILES = {("arial", False, False): "arial.ttf", ("arial", True, False): "arialbd.ttf", ("arial", False, True): "ariali.ttf",
               ("arial", True, True): "arialbi.ttf", ("calibri", True, True): "calibriz.ttf", ("calibri", False, False): "calibri.ttf",
               ("yahei", True, False): "msyhbd.ttc", ("yahei", False, False): "msyh.ttc"}
_FONTS = {}


def _font(family, bold, italic):
    key = (family, bool(bold), bool(italic))
    if key not in _FONTS:
        fname = _FONT_FILES.get(key) or _FONT_FILES[(family, bool(bold), False)]
        _FONTS[key] = ImageFont.truetype(str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / fname), 1000)
    return _FONTS[key]


def width_pt(text, size, bold=False, italic=False, family="arial", spc=0.0):
    """Width of a single line in points: font metrics x MEAS_SAFETY plus the letter spacing (spc in points)."""
    return _font(family, bold, italic).getlength(text) / 1000 * size * MEAS_SAFETY + spc * len(text)


def wrap_count(text, avail_pt, size, bold=False, italic=False, family="arial", spc=0.0):
    """Lines a paragraph needs at this width (greedy word wrap; a word longer than the line counts its own lines)."""
    lines = 0
    for para in str(text).split("\n"):
        words, cur = para.split(" "), ""
        n = 1
        for w in words:
            trial = (cur + " " + w) if cur else w
            if width_pt(trial, size, bold, italic, family, spc) <= avail_pt or not cur:
                cur = trial
            else:
                n += 1
                cur = w
        lines += n
    return lines


# ---- shape helpers ---------------------------------------------------------------------------------------------------------
def runs_text(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return runs_text(item.get("runs", ""))
    return "".join(r if isinstance(r, str) else r[0] for r in item)


def _rgb(hexs):
    return RGBColor.from_string(hexs)


def tb(shapes, box, paras, *, size=14, color=SUBNAVY, bold=False, italic=False, align="l", anchor="t", ls=1.0, name=None,
       inset=(0.02, 0.01), wrap=True, check=True, family="arial", spc=0.0, fill=None):
    """A text box at box = (x, y, w, h) in inches. paras: str | list of runs | dict(runs=, size=, bold=, italic=, color=, align=,
    space=, ls=, spc=); a run is str or (str, fmt) with fmt keys size, bold, italic, color, spc, font. No font name is set unless
    fmt['font'] says so: the theme font applies. The height the text needs is estimated (Arial x MEAS_SAFETY) and checked."""
    x, y, w, h = box
    shp = shapes.add_textbox(E(x), E(y), E(w), E(h))
    if name:
        shp.name = name
    tf = shp.text_frame
    tf.word_wrap = wrap
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = E(inset[0])
    tf.margin_top = tf.margin_bottom = E(inset[1])
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    need = 0.0
    avail = (w - 2 * inset[0]) * 72
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        spec = para if isinstance(para, dict) else {"runs": para}
        runs = spec.get("runs", "")
        runs = [runs] if isinstance(runs, str) else runs
        psize = spec.get("size", size)
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[spec.get("align", align)]
        p.line_spacing = spec.get("ls", ls)
        if spec.get("space"):
            p.space_before = Pt(spec["space"])
        maxsize, anybold, anyital = psize, spec.get("bold", bold), spec.get("italic", italic)
        pspc = spec.get("spc", spc)
        for run in runs:
            txt, fmt = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = txt
            f = r.font
            rs = fmt.get("size", psize)
            f.size = Pt(rs)
            f.bold = fmt.get("bold", spec.get("bold", bold))
            f.italic = fmt.get("italic", spec.get("italic", italic))
            f.color.rgb = _rgb(fmt.get("color", spec.get("color", color)))
            if fmt.get("font"):
                f.name = fmt["font"]
            rspc = fmt.get("spc", pspc)
            if rspc:
                r._r.get_or_add_rPr().set("spc", str(int(round(rspc * 100))))
            maxsize = max(maxsize, rs)
            anybold = anybold or bool(fmt.get("bold"))
            anyital = anyital or bool(fmt.get("italic"))
        text = runs_text(runs)
        if wrap:
            n = wrap_count(text, avail, maxsize, anybold, anyital, family, pspc)
        else:
            n = text.count("\n") + 1
            wneed = max(width_pt(t, maxsize, anybold, anyital, family, pspc) for t in text.split("\n"))
            if check and wneed > avail * 1.001:
                problem(f"slide {CUR['n']}: one-line text wider than its box ({wneed / 72:.2f} in > {avail / 72:.2f} in): {text[:60]!r}")
        need += n * maxsize * 1.2 * p.line_spacing + (spec.get("space") or 0)
    if check and need > (h - 2 * inset[1]) * 72 * 1.001:
        problem(f"slide {CUR['n']}: text may overflow its box ({need / 72:.2f} in needed, {h - 2 * inset[1]:.2f} in): "
                f"{runs_text(paras[0])[:60]!r}")
    return shp


def shape(shapes, kind, box, fill=None, line=None, lw=0.75, name=None, adj=None, dash=None, pattern=None, shadow=False):
    x, y, w, h = box
    shp = shapes.add_shape(kind, E(x), E(y), E(w), E(h))
    if name:
        shp.name = name
    if adj is not None:
        for i, a in enumerate(adj if isinstance(adj, (list, tuple)) else [adj]):
            shp.adjustments[i] = a
    if pattern:
        fg, bg = pattern
        shp.fill.patterned()
        shp.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
        shp.fill.fore_color.rgb = _rgb(fg)
        shp.fill.back_color.rgb = _rgb(bg)
    elif fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = _rgb(line)
        shp.line.width = Pt(lw)
        if dash:
            shp.line.dash_style = dash
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if shadow:
        add_shadow(shp)
    return shp


def add_shadow(shp, alpha=18000, blur=127000, dist=25400):
    spPr = shp._element.spPr
    for old in spPr.findall(qn("a:effectLst")):
        spPr.remove(old)
    eff = etree.SubElement(spPr, qn("a:effectLst"))
    sh = etree.SubElement(eff, qn("a:outerShdw"), blurRad=str(blur), dist=str(dist), dir="5400000", algn="t", rotWithShape="0")
    clr = etree.SubElement(sh, qn("a:srgbClr"), val=DEEP)
    etree.SubElement(clr, qn("a:alpha"), val=str(alpha))


def line(shapes, x1, y1, x2, y2, color, lw=1.0, dash=None, name=None, head=None):
    c = shapes.add_connector(MSO_CONNECTOR.STRAIGHT, E(x1), E(y1), E(x2), E(y2))
    c.line.color.rgb = _rgb(color)
    c.line.width = Pt(lw)
    if dash:
        c.line.dash_style = dash
    if head:
        ln = c.line._get_or_add_ln()
        etree.SubElement(ln, qn("a:tailEnd"), type="triangle", w="med", len="med")
    if name:
        c.name = name
    return c


def group(slide, name):
    g = slide.shapes.add_group_shape()
    g.name = name
    return g


def close_group(g):
    g.shapes._recalculate_extents()
    return g


def pill(shapes, box, text, fill, color=WHITE, size=11, bold=True, name=None, spc=0.6):
    shape(shapes, MSO_SHAPE.ROUNDED_RECTANGLE, box, fill=fill, adj=0.5, name=(name or "Pill") + " shape")
    tb(shapes, box, [text], size=size, color=color, bold=bold, align="c", anchor="m", inset=(0.03, 0.0), wrap=False,
       spc=spc, name=(name or "Pill") + " text")


def kicker(shapes, x, y, w, text, color="C2501F", size=12, name="Kicker"):
    tb(shapes, (x, y, w, 0.27), [text.upper()], size=size, color=color, bold=True, spc=2.2, wrap=False, anchor="m", name=name)


# ---- the template ----------------------------------------------------------------------------------------------------------
def keep_cover_only(prs):
    """Remove the sample's slides 2..n (sldId entries and relationships); slide 1, the cover, stays as the professor drew it."""
    sldIdLst = prs.slides._sldIdLst
    removed = 0
    for sldId in list(sldIdLst)[1:]:
        rId = sldId.rId
        sldIdLst.remove(sldId)
        prs.part.drop_rel(rId)
        removed += 1
    return removed


def set_notes(slide, notes):
    tf = slide.notes_slide.notes_text_frame
    tf.clear()
    for i, para in enumerate([p for p in notes.split("\n") if p.strip()]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = para


def instructor_name(text):
    """'The course instructor' (the registry's form, surname first) -> 'The course instructor'."""
    m = re.match(r"^(.*?)([A-Z][A-Za-z\-]+),\s*([A-Za-z][A-Za-z\-]+)\s*$", text)
    if not m:
        return text
    return f"{m.group(1)}{m.group(3).title()} {m.group(2).title()}"


def _set_run(r_el, text, size=None):
    t = r_el.find(qn("a:t"))
    t.text = text
    if size is not None:
        r_el.find(qn("a:rPr")).set("sz", str(int(round(size * 100))))


def edit_sample_cover(prs, R, lines):
    """Slide 1 = the sample's cover. Title box ('Title') and members box ('Groups member Name+ ID') keep their run properties:
    only the text and the size change. lines: dict(title, subtitle, group, members=[str, str], question, date)."""
    slide = prs.slides[0]
    CUR["n"] = 1
    title_box = members_box = None
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == "Title":
            title_box = sh
        elif sh.has_text_frame and sh.text_frame.text.strip().startswith("Groups member"):
            members_box = sh
    if title_box is None or members_box is None:
        sys.exit("the sample's cover has changed: title or members box not found")
    # title: run 1 = the main title (24 pt as in the sample), a second paragraph = the subtitle in the same face at 16 pt
    p0 = title_box.text_frame.paragraphs[0]._p
    r0 = p0.findall(qn("a:r"))[0]
    _set_run(r0, lines["title"], 24)
    p1 = copy.deepcopy(p0)
    _set_run(p1.findall(qn("a:r"))[0], lines["subtitle"], 16)
    p1.findall(qn("a:r"))[0].find(qn("a:rPr")).set("spc", "100")
    p0.addnext(p1)
    title_box.name = "Title"
    tw = title_box.width / IN
    need = (24 * 1.2 * 1.2 * wrap_count(lines["title"], (tw - 0.2) * 72, 24, True, False, "yahei", 3.0)
            + 16 * 1.2 * 1.2 * wrap_count(lines["subtitle"], (tw - 0.2) * 72, 16, True, False, "yahei", 1.0))
    if need > (title_box.height / IN - 0.1) * 72:
        problem(f"slide 1: the title may overflow its box ({need / 72:.2f} in needed)")
    # members: paragraph 1 = the group in the blue YaHei style the sample keeps in the box's end-of-paragraph properties,
    # paragraphs 2.. = the names in the sample's orange Calibri bold italic run
    txBody = members_box.text_frame._txBody
    paras = txBody.findall(qn("a:p"))
    name_p = paras[0]
    name_r = name_p.findall(qn("a:r"))[0]
    blue_rpr = name_p.find(qn("a:endParaRPr"))
    gp = copy.deepcopy(name_p)
    for r in gp.findall(qn("a:r")):
        gp.remove(r)
    gr = etree.SubElement(gp, qn("a:r"))
    rpr = copy.deepcopy(blue_rpr)
    rpr.tag = qn("a:rPr")
    gr.append(rpr)
    etree.SubElement(gr, qn("a:t")).text = lines["group"]
    end = gp.find(qn("a:endParaRPr"))
    if end is not None:
        gp.remove(end)
        gp.append(end)
    new_ps = [gp]
    for m in lines["members"]:
        mp = copy.deepcopy(name_p)
        for r in mp.findall(qn("a:r"))[1:]:
            mp.remove(r)
        _set_run(mp.findall(qn("a:r"))[0], m, 22)
        new_ps.append(mp)
    for p in paras:
        txBody.remove(p)
    for p in new_ps:
        txBody.append(p)
    members_box.name = "Members"
    mw = members_box.width / IN
    need = 24 * 1.2 * 1.2 + sum(22 * 1.2 * 1.2 * wrap_count(m, (mw - 0.2) * 72, 22, True, True, "calibri") for m in lines["members"])
    band_h = 1833071 / IN
    if need > band_h * 72:
        problem(f"slide 1: the members may not fit in the band ({need / 72:.2f} in needed, band {band_h:.2f} in)")
    # centre the members box on the sample's band (it was 0.5 in lower for one 32 pt line)
    members_box.top = Emu(2214371)
    members_box.height = Emu(1833071)
    # question and date under the band, on the veiled photo
    tb(slide.shapes, (0.80, 4.60, 11.73, 1.62), [
        {"runs": [(lines["question"], {"bold": True, "color": SUBNAVY, "size": 22})], "align": "c", "ls": 1.05},
        {"runs": [(lines["date"][0], {"color": SUBNAVY, "size": 16})], "align": "c", "space": 10},
        {"runs": [(lines["date"][1], {"color": SUBNAVY, "size": 16, "bold": True})], "align": "c", "space": 2},
    ], anchor="m", name="Question and date")
    return slide


def cover_from_builder(prs, R, tmpdir):
    """--cover photo: the cover of build_slides_final.py (USGS flood photo under the sample's veil and band) on layout 'Blank';
    the sample's own slides are all removed first."""
    BSF.delete_all_slides(prs)
    BSF.CURRENT["n"] = 1
    CUR["n"] = 1
    logo = BSF.template_logo(tmpdir)
    slide, which, _ = BSF.cover_slide(prs, R, tmpdir, "photo", logo)
    return slide, which


def legend(shapes, x, y, w, items, size=12, sq=0.15, name="Legend"):
    """items: (colour | (fg, bg) hatch, text), stacked; returns the y under the last item."""
    cy = y
    for col, text in items:
        n = wrap_count(text, (w - sq - 0.12) * 72, size)
        hh = n * size * 1.2 / 72 + 0.05
        if isinstance(col, tuple):
            shape(shapes, MSO_SHAPE.RECTANGLE, (x, cy + 0.04, sq, sq), pattern=col, line=col[0], lw=0.5, name=f"{name} key")
        else:
            shape(shapes, MSO_SHAPE.RECTANGLE, (x, cy + 0.04, sq, sq), fill=col, name=f"{name} key")
        tb(shapes, (x + sq + 0.08, cy, w - sq - 0.08, hh), [text], size=size, color=SUBNAVY, ls=1.0, name=f"{name} text")
        cy += hh + 0.03
    return cy


def about_share(v, text):
    """'about half' when a share lies between 45 % and 55 %, else the registry's text (spoken summaries only)."""
    return "about half" if 0.45 <= float(v) <= 0.55 else text


# ---- common furniture of slides 2-6 ----------------------------------------------------------------------------------------
def content_slide(prs, number, heading, kick, sub, presenter, honesty):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    CUR["n"] = number
    hx, hy, hw, hh = HEAD
    tb(slide.shapes, (hx / IN, hy / IN, hw / IN, hh / IN), [heading], size=HEAD_PT, color=WHITE, bold=True, anchor="m",
       wrap=True, inset=(0.1, 0.05), name="Heading", check=False)
    if width_pt(heading, HEAD_PT, bold=True) > (hw / IN - 0.2) * 72 * 0.94:
        problem(f"slide {number}: heading too long for one line at {HEAD_PT} pt with a 6 % margin: {heading!r}")
    BSF.add_slidenum(slide, number)
    kicker(slide.shapes, X0, KICK_Y, X1 - X0, kick)
    tb(slide.shapes, (X0, SUB_Y, X1 - X0, 0.42), [sub], size=16, color=SUBNAVY, bold=True, anchor="m", wrap=False, name="Sub-heading")
    if honesty:
        tb(slide.shapes, HON_BOX, [{"runs": honesty, "size": 10.5}], color=GREY, anchor="b", ls=1.0, name="Honesty line")
    px, py, pw, ph = PRES_BOX
    tb(slide.shapes, (px / IN, py / IN, pw / IN, ph / IN), [presenter], size=13.33, color=GREY, align="r", anchor="m", wrap=False,
       inset=(0.02, 0.0), name="Presenter")
    return slide


# ---- slide 2 features ----------------------------------------------------------------------------------------------------------
def _parse_path(d):
    """SVG path with M / l / z only (basin_map.json) -> list of polygons [(x, y), ...] in px."""
    polys, cur, x, y = [], [], 0.0, 0.0
    for cmd, args in re.findall(r"([MmLlZz])([^MmLlZz]*)", d):
        nums = [float(v) for v in re.findall(r"-?\d*\.?\d+(?:e-?\d+)?", args)]
        if cmd == "M":
            if cur:
                polys.append(cur)
            x, y = nums[0], nums[1]
            cur = [(x, y)]
            for i in range(2, len(nums), 2):
                x, y = nums[i], nums[i + 1]
                cur.append((x, y))
        elif cmd == "l":
            for i in range(0, len(nums), 2):
                x, y = x + nums[i], y + nums[i + 1]
                cur.append((x, y))
        elif cmd in "Zz":
            if cur:
                polys.append(cur)
                x, y = cur[0]
            cur = []
        else:
            raise ValueError(f"path command {cmd} not handled")
    if cur:
        polys.append(cur)
    return polys


def _clip(poly, xmin, ymin, xmax, ymax):
    """Sutherland-Hodgman clipping of a polygon to a rectangle."""
    def clip_edge(pts, inside, inter):
        out = []
        for i, p in enumerate(pts):
            q = pts[i - 1]
            if inside(p):
                if not inside(q):
                    out.append(inter(q, p))
                out.append(p)
            elif inside(q):
                out.append(inter(q, p))
        return out

    def ix(xc):
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / (b[0] - a[0]))

    def iy(yc):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1]), yc)

    pts = poly
    for inside, inter in ((lambda p: p[0] >= xmin, ix(xmin)), (lambda p: p[0] <= xmax, ix(xmax)),
                          (lambda p: p[1] >= ymin, iy(ymin)), (lambda p: p[1] <= ymax, iy(ymax))):
        if not pts:
            break
        pts = clip_edge(pts, inside, inter)
    return pts


def _area(poly):
    return 0.5 * sum(poly[i - 1][0] * p[1] - p[0] * poly[i - 1][1] for i, p in enumerate(poly))


def _inside(pt, poly):
    x, y = pt
    c = False
    for i, (x2, y2) in enumerate(poly):
        x1, y1 = poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def _freeform(shapes, polys, ox, oy, s, fill, line_c, lw, name):
    """One freeform shape (several contours) from polygons in px; s = inches per px; (ox, oy) = slide inches of px (0, 0)."""
    k = 100                                                   # local units: 1/100 px (python-pptx rounds vertices to integers)
    first = polys[0]
    b = shapes.build_freeform(first[0][0] * k, first[0][1] * k, scale=s * IN / k)
    b.add_line_segments([(px * k, py * k) for px, py in first[1:]], close=True)
    for poly in polys[1:]:
        b.move_to(poly[0][0] * k, poly[0][1] * k)
        b.add_line_segments([(px * k, py * k) for px, py in poly[1:]], close=True)
    shp = b.convert_to_shape(E(ox), E(oy))
    shp.name = name
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = _rgb(fill)
    else:
        shp.fill.background()
    if line_c:
        shp.line.color.rgb = _rgb(line_c)
        shp.line.width = Pt(lw)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def draw_map(slide, box, R):
    """The basin map of the mid-term web deck (basin_map.json: EPSG:5070, 1500 m a px), redrawn as native shapes: land, the 12
    basin outlines in their landscape colours with locator rings and gauges, a 200 km bar, grid north, and a few place names
    placed with pyproj from their longitude and latitude."""
    d = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    vb_w, vb_h = d["viewBox"][2], d["viewBox"][3]
    x, y, w, h = box
    s = min(w / vb_w, h / vb_h)                                # inches per px
    ox, oy = x + (w - vb_w * s) / 2, y + (h - vb_h * s) / 2
    g = group(slide, "Map of the 12 basins")
    shape(g.shapes, MSO_SHAPE.RECTANGLE, (ox, oy, vb_w * s, vb_h * s), fill="EAF2FB", line=PANEL2, lw=0.75, name="Map frame")
    land = [p for p in (_clip(p, 0, 0, vb_w, vb_h) for p in _parse_path(d["land"])) if len(p) >= 3 and abs(_area(p)) > 4]
    big = sorted(land, key=lambda p: -abs(_area(p)))
    holes = [p for p in land if any(q is not p and abs(_area(q)) > abs(_area(p)) and _inside(p[0], q) for q in big[:3])]
    solid = [p for p in land if not any(p is hh for hh in holes)]
    _freeform(g.shapes, solid, ox, oy, s, "F3F2EE", "A9B8C8", 0.5, "Land")
    if holes:
        _freeform(g.shapes, holes, ox, oy, s, "EAF2FB", "A9B8C8", 0.5, "Lakes")
    for b in d["basins"]:
        col = LAND_C[b["landscape_key"]]
        cx, cy, r = b["ring"]
        shape(g.shapes, MSO_SHAPE.OVAL, (ox + (cx - r) * s, oy + (cy - r) * s, 2 * r * s, 2 * r * s), line=col, lw=1.0,
              name=f"Ring {b['name']}")
        polys = [p for p in _parse_path(b["path"]) if len(p) >= 3]
        _freeform(g.shapes, polys, ox, oy, s, col, col, 0.5, f"Basin {b['name']}")
        gx, gy = b["gauge"]
        gr = 4.0
        shape(g.shapes, MSO_SHAPE.OVAL, (ox + (gx - gr) * s, oy + (gy - gr) * s, 2 * gr * s, 2 * gr * s), fill=WHITE, line=DEEP,
              lw=0.75, name=f"Gauge {b['name']}")
    # place names (grey italic) and the two seas, positioned in EPSG:5070 exactly as the map's own transform
    try:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True)
        fx0, fy0, fx1, fy1 = d["transform"]["frame_epsg5070_m"]
        mpp = d["transform"]["metres_per_px"]

        def px(lon, lat):
            X, Y = t.transform(lon, lat)
            return (X - fx0) / mpp, (fy1 - Y) / mpp
        places = [("Virginia", -78.3, 37.15, GREY), ("North Carolina", -79.75, 34.55, GREY), ("Georgia", -82.55, 31.55, GREY),
                  ("Florida", -81.75, 29.6, GREY), ("Mississippi", -89.75, 33.25, GREY), ("Alabama", -86.75, 32.3, GREY),
                  ("South Carolina", -80.85, 33.85, GREY), ("Tennessee", -86.3, 35.95, GREY),
                  ("Atlantic Ocean", -78.6, 30.6, BLUE_INK), ("Gulf of Mexico", -87.3, 28.0, BLUE_INK)]
        for nm, lon, lat, col in places:
            cxp, cyp = px(lon, lat)
            if not (60 < cxp < vb_w - 60 and 20 < cyp < vb_h - 20):
                continue
            wbox = width_pt(nm, 10, italic=True) / 72 + 0.08
            tb(g.shapes, (ox + cxp * s - wbox / 2, oy + cyp * s - 0.11, wbox, 0.22), [nm], size=10, italic=True, color=col,
               align="c", anchor="m", wrap=False, inset=(0.02, 0.0), name=f"Place {nm}")
    except ImportError:
        warn("pyproj not installed: the map has no place names")
    # 200 km bar and grid north
    sb = d["scale_bar"]
    bx, by, bl = ox + sb["x"] * s, oy + sb["y"] * s, sb["px"] * s
    line(g.shapes, bx, by, bx + bl, by, DEEP, 1.25, name="Scale bar")
    for xx in (bx, bx + bl):
        line(g.shapes, xx, by - 0.04, xx, by + 0.04, DEEP, 1.0, name="Scale tick")
    tb(g.shapes, (bx, by - 0.30, bl, 0.24), [f"{sb['km']} km"], size=10, color=DEEP, align="c", anchor="b", wrap=False,
       name="Scale label")
    na = d["north_arrow"]
    line(g.shapes, ox + na["x"] * s, oy + na["y"] * s, ox + na["x"] * s, oy + (na["y"] - na["len"]) * s, DEEP, 1.25, head=True,
         name="North arrow")
    tb(g.shapes, (ox + na["x"] * s - 0.15, oy + na["y"] * s + 0.01, 0.30, 0.22), ["N"], size=10, bold=True, color=DEEP, align="c",
       wrap=False, name="North label")
    tb(g.shapes, (ox + vb_w * s - 3.6, oy + vb_h * s - 0.25, 3.55, 0.22),
       ["CAMELS-US outlines · Natural Earth coast · EPSG:5070"], size=8.5, color=GREY, align="r", anchor="b", wrap=False,
       name="Source map")
    close_group(g)
    return g


def draw_funnel(slide, x, y, w, h, steps):
    """steps: list of (number, label); the last one highlighted. Chevrons like the template's logo plate."""
    g = group(slide, "Funnel of the basins")
    n = len(steps)
    d = 0.26                                                       # depth of the point
    gap = 0.05
    cw = (w + (n - 1) * (d - gap)) / n
    for i, (num, lab) in enumerate(steps):
        cx = x + i * (cw - d + gap)
        last = i == n - 1
        kind = MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON
        shp = shape(g.shapes, kind, (cx, y, cw, h), fill=DEEP if last else PANEL, name=f"Funnel step {i + 1}")
        shp.adjustments[0] = d / h
        tx = cx + (0.06 if i == 0 else d) + 0.02
        tw = cw - (0.06 if i == 0 else d) - d - 0.02
        tb(g.shapes, (tx, y + 0.05, tw, h - 0.10), [
            {"runs": [(num, {"size": 22, "bold": True})], "ls": 0.95},
            {"runs": [(lab, {"size": 12})], "ls": 0.95},
        ], color=WHITE if last else DEEP, anchor="m", align="l", name=f"Funnel text {i + 1}")
    close_group(g)


def chips_row(shapes, x, y, items, size=12, sq=0.17, gap=0.28, name="Chips"):
    """items: list of (colour, text). Square chips with a label, left to right; returns the x where the row ends."""
    cx = x
    for col, text in items:
        shape(shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (cx, y + 0.05, sq, sq), fill=col, adj=0.2, name=f"{name} square")
        wt = width_pt(text, size) / 72 + 0.06
        tb(shapes, (cx + sq + 0.06, y, wt, 0.27), [text], size=size, color=SUBNAVY, anchor="m", wrap=False, name=f"{name} label")
        cx += sq + 0.06 + wt + gap
    return cx - gap


def draw_split(slide, x, y, w, R, N, V):
    g = group(slide, "Fair split")
    tb(g.shapes, (x, y, w, 0.26), ["FAIR SPLIT BY WATER YEAR · TEST SCORED ONCE PER EXPERIMENT"], size=11, bold=True, color=DEEP2,
       spc=1.2, wrap=False, anchor="m", name="Split label")
    yrs = [int(V("split_train_water_years")), int(V("split_val_water_years")), int(V("split_test_water_years"))]
    tot = sum(yrs)
    parts = [("Training", N("split_train_years_text"), "weights fitted", DEEP),
             ("Validation", N("split_val_years_text"), "every choice", BLUE),
             ("Test", N("split_test_years_text"), "scored once", ORANGE)]
    bx = x
    by = y + 0.30
    for (lab, years, role, col), n in zip(parts, yrs):
        bw = w * n / tot
        shape(g.shapes, MSO_SHAPE.RECTANGLE, (bx, by, bw - 0.03, 0.36), fill=col, name=f"Split {lab}")
        tb(g.shapes, (bx, by, bw - 0.03, 0.36), [lab], size=12, bold=True, color=WHITE, align="c", anchor="m", wrap=False,
           inset=(0.0, 0.0), name=f"Split {lab} label")
        lw_ = max(bw, 1.25)
        tb(g.shapes, (bx + bw / 2 - lw_ / 2, by + 0.39, lw_, 0.44), [
            {"runs": [(years.replace("-", "–"), {"bold": True})]}, {"runs": role}], size=12, color=SUBNAVY, align="c", ls=0.95,
           name=f"Split {lab} years")
        bx += bw
    close_group(g)
    return by + 0.39 + 0.44


# ---- slide 3 features ----------------------------------------------------------------------------------------------------------
def draw_clock(slide, x, y, w, N):
    """The issue-time clock of the mid-term (screen 2): day t takes three quarters of the bar, day t+1 one quarter."""
    g = group(slide, "Issue-time clock")
    day_end = N("day_end_daymet_rounded")                       # 'about 09:00'
    nldas_end = N("day_end_nldas_rounded")                      # 'about 17:30'
    hh = lambda s: re.search(r"(\d{1,2}):(\d{2})", s)
    h_table = int(hh(day_end).group(1)) + int(hh(day_end).group(2)) / 60
    h_nldas = int(hh(nldas_end).group(1)) + int(hh(nldas_end).group(2)) / 60
    f_day = 0.75
    xs = lambda hour: x + w * f_day * hour / 24
    bar_y, bar_h = y + 0.30, 0.62
    tb(g.shapes, (x, y, 3.0, 0.26), ["the issue day t"], size=12, bold=True, color=SUBNAVY, wrap=False, anchor="m",
       name="Clock day t")
    tb(g.shapes, (xs(24), y, w * (1 - f_day), 0.26), ["day t+1, the target"], size=12, bold=True, color=SUBNAVY, align="c",
       wrap=False, anchor="m", name="Clock day t+1")
    seen = [{"runs": [("seen by the model", {"bold": True})]},
            {"runs": f"since {hh(day_end).group(0).zfill(5)} the day before", "size": 10.5, "bold": False}]
    segs = [(x, xs(h_table), BLUE, None, seen, WHITE),
            (xs(h_table), xs(24), ORANGE, None, "hidden: fallen, but filed under tomorrow", WHITE),
            (xs(24), x + w, None, (HATCH_FG, HATCH_BG), "still to come", INK2)]
    for i, (a, b, col, pat, lab, ink) in enumerate(segs):
        shape(g.shapes, MSO_SHAPE.RECTANGLE, (a, bar_y, b - a, bar_h), fill=col, pattern=pat, name=f"Clock segment {i + 1}")
        if pat:
            lw_ = width_pt(lab, 12, True) / 72 + 0.16
            tb(g.shapes, (a + (b - a - lw_) / 2, bar_y + bar_h / 2 - 0.15, lw_, 0.30), [lab], size=12, bold=True, color=ink,
               align="c", anchor="m", inset=(0.04, 0.0), wrap=False, fill=HATCH_BG, name=f"Clock segment {i + 1} label")
        else:
            tb(g.shapes, (a, bar_y, b - a, bar_h), lab if isinstance(lab, list) else [lab], size=12, bold=True, color=ink,
               align="c", anchor="m", ls=0.95, inset=(0.06, 0.02), name=f"Clock segment {i + 1} label")
    # the marker where the rain table stops, and the tick labels
    line(g.shapes, xs(h_table), bar_y - 0.08, xs(h_table), bar_y + bar_h + 0.08, DEEP, 2.25, name="Clock marker")
    ticks = [("00:00", 0), (hh(day_end).group(0).zfill(5), h_table), (hh(nldas_end).group(0).zfill(5), h_nldas), ("24:00", 24)]
    for lab, hour in ticks:
        tx = xs(hour)
        line(g.shapes, tx, bar_y + bar_h, tx, bar_y + bar_h + 0.07, GREY, 0.75, name="Clock tick")
        tb(g.shapes, (tx - 0.4, bar_y + bar_h + 0.07, 0.8, 0.24), [lab], size=11, color=GREY, align="c", anchor="t", wrap=False,
           inset=(0.0, 0.0), name=f"Clock tick {lab}")
    tb(g.shapes, (xs(h_table) - 1.9, bar_y + bar_h + 0.31, 3.8, 0.26), [f"the rain table's day ends here ({day_end})"], size=12,
       bold=True, color=DEEP, align="c", wrap=False, name="Clock marker label")
    tb(g.shapes, (xs(h_nldas) - 0.05, bar_y + bar_h + 0.31, 1.9, 0.26), ["rain to the evening"], size=11, italic=True,
       color=GREY, align="l", wrap=False, name="Clock evening label")
    close_group(g)
    return bar_y + bar_h + 0.62, [(x, xs(h_table)), (xs(h_table), xs(24)), (xs(24), x + w)]


# ---- the deck --------------------------------------------------------------------------------------------------------------------
def build(args):
    R = BSF.Registry(REGISTRY)
    N, V = R.text, R.val
    BSF.prepare_template(False)
    prs = Presentation(str(BSF.TEMPLATE))
    if (prs.slide_width, prs.slide_height) != (SLIDE_W, SLIDE_H):
        sys.exit(f"unexpected slide size {prs.slide_width} x {prs.slide_height}")
    BSF.set_group_number(prs, N("group_number"))
    tmpdir = tempfile.mkdtemp(prefix="flood_deck6_")

    # ---- facts shared by the slides ----
    nm = [N(f"member{i}_name") for i in range(1, int(V("n_members")) + 1)]
    roles = [N(f"member{i}_role") for i in range(1, int(V("n_members")) + 1)]
    first = lambda s: s.split()[0]
    A, MU, MO, HI = nm
    pres = {k: f"presenter: {nm[i]} ({roles[i]})" for i, k in enumerate("A MU MO HI".split())}
    group_no = re.sub(r"[^0-9]", "", N("group_number"))
    fw_aorc, fw_hind, fw_rt, fw_up = N("fw_aorc"), N("fw_hindsight"), N("fw_realtime_short"), N("fw_upper")
    inside = "comparable only inside it"
    sat_years = "–".join(str(s)[:4] for s in V("split_sat_test_dates"))
    sat_onsets = f"{int(V('e4e_wqa_s_l1_onset_hits')) + int(V('e4e_wqa_s_l1_onset_misses')):,}"
    cap30 = N("go_max_false")
    go_w, go_f = float(V("target_go_warned")), float(V("go_max_false"))
    n5 = str(len(V("e4e_seeds")))
    n3 = N("seeds_per_set")
    km2 = lambda s: s.replace("km2", "km²")
    spoken_km2 = lambda s: s.replace("km2", "square kilometres")
    # product roles as in build_slides_final.py (the hourly USGS record became an input of E4e)
    products = ["camels_us", "aorc", "gefs", "era5", "merra2", "imerg", "stage4", "elevation_tiles", "nlcd", "hurdat2", "usgs_iv"]
    core, extra, checks = [], [], []
    for pid in products:
        role = N(f"ds_{pid}_role").lower()
        if role.startswith("the data every model"):
            core.append(pid)
        elif "extra input" in role or "tested" in role:
            extra.append(pid)
        else:                                                     # the registry's roles (report Table 3): 1 + 5 + 5
            checks.append(pid)
    if len(core) + len(extra) + len(checks) != int(V("n_products")):
        warn("product roles do not add up to n_products")
    words_n = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
               11: "eleven", 12: "twelve"}
    numw = lambda k: words_n.get(int(k), str(k))
    slides = []

    # ================= slide 1: cover (the team lead, 30 s) =================
    R.slide = 1
    title_main, title_sub = [s.strip() for s in N("project_title").split(":", 1)]
    members = [f"{nm[0]} ({N('member1_id')})  ·  {nm[1]} ({N('member2_id')})",
               f"{nm[2]} ({N('member3_id')})  ·  {nm[3]} ({N('member4_id')})"]
    question = "What limits a one-to-three-day flood warning on a small river: the kind of model, or the information it is given?"
    date_line = [N("course"), f"Final presentation · {N('final_presentation')} · {instructor_name(N('instructor'))}"]
    if args.cover == "sample":
        keep_cover_only(prs)
        cover = edit_sample_cover(prs, R, {"title": title_main, "subtitle": title_sub[:1].upper() + title_sub[1:],
                                           "group": N("group_number"), "members": members, "question": question, "date": date_line})
        cover_which = "the sample's own cover slide (campus photo), edited in place"
    else:
        cover, cover_which = cover_from_builder(prs, R, tmpdir)
    notes1 = (f"Hello. We are Group {group_no}: {first(MU)}, {first(MO)}, {first(HI)} and I, {first(A)}, the group leader. "
              f"Our question: {question[:1].lower() + question[1:]} {first(MU)} starts with the data, {first(MO)} shows why the "
              f"warning is limited, {first(HI)} what more information buys, and I close with what is new and our conclusions.\n"
              "If asked why rivers in the United States: CAMELS-US is the open benchmark data set of this field, with decades of "
              "daily flow and weather for every basin and no account needed.")
    set_notes(cover, notes1)
    slides.append((1, "Cover", A, 30, notes1))

    # ================= slide 2: data and method (member 2, 150 s) =================
    R.slide = 2
    land_names = [s.strip() for s in N("landscape_names").split(";")]
    short_land = {}
    for s_ in land_names:
        key = "m" if s_.lower().startswith("mountain") else ("n" if s_.lower().startswith("northern") else "s")
        short_land[key] = s_.split(" and ")[0].split(" (")[0] if key != "m" else s_
    honesty2 = [("Map: ", {"bold": True}), "drawn in EPSG:5070 (NAD83 / Conus Albers) from the CAMELS-US outlines and Natural Earth "
                "1:50 m land, as in the mid-term web deck; every product, its licence and how it was read: the report and the "
                "datasets page"]
    s2 = content_slide(prs, 2, "Twelve basins by rule, public data, a fair test", "Where and with what data",
                       f"{numw(V('n_products')).capitalize()} public products, {N('gb_read')} read, no account; GIS layers of all "
                       f"{N('basins_chosen')} basins exported",
                       pres["MU"], honesty2)
    draw_map(s2, (X0, TOP, 4.95, BOTTOM - TOP), R)
    rx, rw = 5.85, X1 - 5.85
    draw_funnel(s2, rx, TOP, rw, 0.92, [
        (N("basins_all_camels"), "CAMELS-US basins"), (N("basins_candidates"), "south-east candidates"),
        (N("basins_complete"), "complete record"), (N("basins_chosen"), f"chosen, {N('basins_rule_per_landscape')} per landscape")])
    gch = group(s2, "Landscape chips")
    chips_row(gch.shapes, rx, TOP + 1.02, [(LAND_C["m"], short_land["m"]), (LAND_C["n"], short_land["n"]),
                                           (LAND_C["s"], short_land["s"])], size=12, name="Landscape")
    close_group(gch)
    gro = group(s2, "Product roles")
    chips_row(gro.shapes, rx, TOP + 1.36, [(DEEP, "CAMELS-US: every model"), (BLUE, f"{numw(len(extra))} extra inputs"),
                                           (HATCH_FG, f"{numw(len(checks))} checks and maps")], size=12, gap=0.22, name="Role")
    close_group(gro)
    yb = draw_split(s2, rx, TOP + 1.70, rw, R, N, V)
    gim = group(s2, "Implementation")
    tb(gim.shapes, (rx, yb + 0.03, rw, 0.62), [{"runs": [
        ("Implementation: ", {"bold": True}),
        f"one LSTM for all {N('n_basins')} rivers, {N('config_hidden')} hidden units, dropout {N('config_dropout')}, learning rate "
        f"{N('config_lr')}, {N('config_warmup')}-day warm-up, early stopping on validation (patience {N('config_patience')}); "
        f"PyTorch {N('torch_version')}, CPU only"]}], size=12, color=SUBNAVY, ls=0.95, name="Implementation text")
    close_group(gim)
    gt = group(s2, "GO rule")
    ty = yb + 0.68
    th = BOTTOM - ty
    shape(gt.shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (rx, ty, rw, th), fill=AQUA_T, adj=0.08, name="GO tile")
    pill(gt.shapes, (rx + 0.16, ty + 0.14, 0.62, 0.32), "GO", GOOD, size=13, name="GO")
    tb(gt.shapes, (rx + 0.92, ty + 0.06, rw - 1.02, th - 0.12), [
        {"runs": [("Fixed in advance: ", {"bold": True}),
                  f"{N('target_go_warned')} of flood onsets warned a day ahead and {N('target_go_false')} of warnings false"]},
        {"runs": f"Flood level: {N('high_flow_percentile')} percentile of the training years; onset: first day above it",
         "size": 12, "color": INK2, "space": 3},
    ], size=14, anchor="m", name="GO text")
    close_group(gt)
    notes2 = (f"Thank you, {first(A)}. Where, and with what data. We study {numw(V('n_basins'))} rivers in the south-east of the "
              f"United States, chosen by rule before any model ran. We started from all {N('basins_all_camels')} basins of "
              f"CAMELS-US. {N('basins_candidates')} are candidates in the south-east: "
              f"{N('basins_rule_huc')}, snow under {N('basins_rule_max_snow_share')} of the precipitation, humid, and "
              f"{spoken_km2(N('basins_rule_area_km2'))} in area. {N('basins_complete')} of them have a complete record, and we "
              f"took {N('basins_rule_per_landscape')} in each of {numw(V('n_landscapes'))} landscapes, evenly spaced in area: the "
              "mountains, in purple on the map; the northern lowlands and Piedmont, in green; and the southern lowlands, in pink.\n"
              f"All the data are public: {numw(V('n_products'))} products, read with no account and no payment, {N('gb_read')} in "
              f"all. Every model reads CAMELS-US: the daily river flow measured by the US Geological Survey, the Daymet rain table "
              f"of each basin, and {N('n_properties')} basin properties. {numw(len(extra)).capitalize()} more products "
              "are extra inputs of single experiments, and "
              f"{numw(len(checks))} are checks and maps; one of them, the gauge's own 15-minute record, becomes an input in "
              f"{first(A)}'s new test. The model is one LSTM for all rivers, trained on a CPU. The outlines, gauges, elevation and "
              f"land cover of all {N('basins_chosen')} basins are exported as GIS layers, {N('gis_layers_files')} files.\n"
              f"The test is fair. The years are split in three: the training years, {N('split_train_years_text')}, fit the weights; "
              f"the validation years, {N('split_val_years_text')}, make every choice; and the test years, "
              f"{N('split_test_years_text')}, are scored once per experiment.\n"
              f"The verdict rule was fixed in advance. The flood level of each river is the {N('high_flow_percentile')} percentile "
              "of its training years, and a flood onset is the first day above it, the day a warning matters most. GO means "
              f"{N('target_go_warned')} of the onsets warned a day ahead, with {N('target_go_false')} of the warnings false. "
              f"{first(MO)} now shows how far the usual inputs get.\n"
              f"If asked how the GIS layers were checked: area within {N('gis_area_vs_gages2_median_pct')} of the CAMELS table "
              f"(median), elevation rank correlation {N('gis_elev_rank_correlation')}; independent review "
              f"{N('gis_layers_review_date')}: {N('gis_layers_review_verdict')}.")
    set_notes(s2, notes2)
    slides.append((2, "Data and method: twelve basins by rule, public data, a fair test", MU, 150, notes2))

    # ================= slide 3: why the warning is limited (member 3, 150 s) =================
    R.slide = 3
    honesty3 = (f"Test years {N('split_test_years_text').replace('-', '–')}, {N('onsets_l1')} onsets, a day ahead · LSTM with the rain "
                f"table and gauge: average of {N('lstm_wq_l1_n_seeds')} trainings · Transformer: its own experiment "
                f"({N('cmp_seeds')} seeds, {N('cmp_sequence_days')}-day sequences) · rain clock: hourly rain around each onset, "
                "no model involved")
    s3 = content_slide(prs, 3, "Move the issue time: how much rain is seen?", "Why · the gap",
                       f"A forecast issued at midnight sees the rain table only up to {N('day_end_daymet_rounded')}",
                       pres["MO"], honesty3)
    cw_ = 7.85
    yclock, segx = draw_clock(s3, X0, TOP, cw_, N)
    gsh = group(s3, "Rain shares")
    shares = [(N("clock_share_table"), "in the table at the issue time", BLUE),
              (N("clock_share_hidden"), "fallen, but filed under the next day", ORANGE),
              (N("clock_share_future"), "still to come", INK2)]
    for (val, lab, col), (a, b) in zip(shares, segx):
        tb(gsh.shapes, (a, yclock, b - a, 0.50), [val], size=28, bold=True, color=col, align="c", anchor="b", wrap=False,
           inset=(0.02, 0.0), name="Share value")
        tb(gsh.shapes, (a, yclock + 0.50, b - a, 0.50), [lab], size=12, color=SUBNAVY, align="c", anchor="t", ls=0.95,
           name="Share label")
    tb(gsh.shapes, (X0, yclock + 1.02, cw_, 0.27),
       [f"of the rain in the {N('clock_window_hours')} around each of the {N('clock_n_onsets')} flood onsets of the test years"],
       size=12, italic=True, color=GREY, align="c", wrap=False, name="Share caption")
    close_group(gsh)
    # big-number card
    gb = group(s3, "Big number")
    bx, bw_ = 8.80, X1 - 8.80
    bh = yclock + 1.29 - TOP
    shape(gb.shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (bx, TOP, bw_, bh), fill=WHITE, line=PANEL2, lw=0.75, adj=0.06, shadow=True,
          name="Big number card")
    tb(gb.shapes, (bx + 0.1, TOP + 0.02, bw_ - 0.2, 1.40), [N("kf_level_wq_onset_warned")], size=80, bold=True, color=DEEP,
       align="c", anchor="m", wrap=False, inset=(0.0, 0.0), name="Big number")
    tb(gb.shapes, (bx + 0.2, TOP + 1.38, bw_ - 0.4, bh - 1.48), [
        {"runs": [("of flood onsets warned a day ahead", {"bold": True})], "size": 15},
        {"runs": ["with the ", ("rain table and the river gauge", {"bold": True, "color": ORANGE_INK}), ", the usual inputs"],
         "space": 4},
    ], size=14, align="c", anchor="t", name="Big number text")
    close_group(gb)
    # three tiles
    ty3 = yclock + 1.47
    th3 = BOTTOM - ty3 - 0.52
    tw3 = (X1 - X0 - 2 * 0.16) / 3
    best_model = {"base_linreg_wq": "ridge regression", "base_rf_wq": "random forest", "base_persistence": "tomorrow = today"}.get(
        N("best_simple_without_weather_forecast_l1_model"), N("best_simple_without_weather_forecast_l1_model"))
    tiles = [(f"NSE {N('kf_level_wq_nse')} vs {N('best_simple_without_weather_forecast_l1_nse')}",
              f"LSTM vs the best simple model, a day ahead: {N('lstm_vs_best_simple_l1_reading')}"),
             ("Transformer: no better", f"{N('cmp_ratio_seconds_per_model')} the LSTM's training time, in its own experiment"),
             ("The same onsets missed", f"{N('same_misses_count')} of the {N('onsets_l1')} by all four kinds of model "
                                        f"({N('same_misses_share')})")]
    for i, (head, body) in enumerate(tiles):
        gx = X0 + i * (tw3 + 0.16)
        gg = group(s3, f"Tile {i + 1}")
        shape(gg.shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (gx, ty3, tw3, th3), fill=PANEL, adj=0.10, name="Tile")
        tb(gg.shapes, (gx + 0.15, ty3 + 0.08, tw3 - 0.3, th3 - 0.16), [
            {"runs": [(head, {"bold": True})], "size": 18, "color": DEEP},
            {"runs": body, "size": 14, "space": 3},
        ], anchor="m", name="Tile text")
        close_group(gg)
    gk = group(s3, "Takeaway")
    shape(gk.shapes, MSO_SHAPE.RECTANGLE, (X0, BOTTOM - 0.40, 0.07, 0.38), fill=ORANGE, name="Takeaway bar")
    tb(gk.shapes, (X0 + 0.16, BOTTOM - 0.42, X1 - X0 - 0.2, 0.42),
       [f"{N('fw_finding_short')[:1].upper() + N('fw_finding_short')[1:]}: the rain table stops at "
        f"{N('day_end_daymet_rounded')}."], size=17, bold=True, color=SUBNAVY, anchor="m", wrap=False, name="Takeaway text")
    close_group(gk)
    notes3 = (f"Thank you, {first(MU)}. Why is the warning limited? Look at the clock of one day. A forecast for tomorrow is issued "
              f"at the end of today, at midnight. But the CAMELS rain table, Daymet, closes its day at {N('day_end_daymet_rounded')} "
              "in the morning. So the rain that falls later is filed under tomorrow, and the model cannot see it when it "
              f"forecasts. The pipeline measured this on hourly rain in the {N('clock_window_hours')} around each of the "
              f"{N('clock_n_onsets')} flood onsets of the test years. Only {N('clock_share_table')} of that rain is in the table at "
              f"the issue time, in blue. {N('clock_share_hidden')} has already fallen but is filed under the next day, in orange. "
              f"And {N('clock_share_future')} is still to come, so only a weather forecast can supply it.\n"
              f"With the usual inputs, the rain table and the river gauge, the LSTM warns of {N('kf_level_wq_onset_warned')} of "
              "the flood onsets a day ahead. That is the big number on the right.\n"
              "Is the model to blame? We tested that in three ways. First, the LSTM against the best simple model, a "
              f"{best_model}: a median NSE of {N('kf_level_wq_nse')} against {N('best_simple_without_weather_forecast_l1_nse')} a "
              f"day ahead, {N('lstm_vs_best_simple_l1_reading')}. NSE is one for a perfect forecast of the flow. Second, a "
              "Transformer, the architecture behind large language models, in its own experiment: no better, and it took "
              f"{N('cmp_ratio_seconds_per_model').replace('x', ' times')} the training time. Third, all four kinds of model miss "
              f"the same onsets: {N('same_misses_count')} of the {N('onsets_l1')}, that is {N('same_misses_share')}, are missed by "
              "every one of them.\n"
              "So, among the models tried, the model is not the limit: the limit is the rain it is given at the issue time. "
              f"{first(HI)} now shows what happens when the same network is given more information.\n"
              f"If asked about the Transformer: its own experiment, {N('cmp_seeds')} seeds and {N('cmp_sequence_days')}-day "
              f"sequences; Transformer minus its LSTM twin {N('cmp_transformer_minus_lstm_l1')} NSE a day ahead.")
    set_notes(s3, notes3)
    slides.append((3, "Why the warning is limited: move the issue time", MO, 150, notes3))

    # ================= slide 4: what more information buys (member 4, 150 s) =================
    R.slide = 4
    seeds_aorc = N("lstm_wqa_l1_n_seeds")
    honesty4 = [f"AORC rain: {fw_aorc}; AORC {N('kf_level_wqa_onset_warned')}: {fw_hind} · Stage IV {N('s4_stage4_share')}: "
                f"{fw_rt} · perfect weather {N('kf_level_wqf_onset_warned')}: {fw_up} · averages of {seeds_aorc} trainings "
                f"(AORC, Stage IV, within-day rain) or {N('lstm_wq_l1_n_seeds')} (rain table, evening rain, perfect weather)"]
    s4 = content_slide(prs, 4, "Same network, more information, more warnings", "Results · the ladder",
                       f"Onsets warned a day ahead, same LSTM, more rain information · test years "
                       f"{N('split_test_years_text').replace('-', '–')}, {N('onsets_l1')} onsets",
                       pres["HI"], honesty4)
    ladder = [("kf_level_wq_onset_warned", "rain table + gauge", DEEP, WHITE, None),
              ("kf_level_wqn_onset_warned", f"+ rain to the evening ({re.sub('about ', '', N('day_end_nldas_rounded'))})", BLUE,
               WHITE, None),
              ("s4_stage4_share", "+ real-time radar rain to midnight", AQUA, WHITE, fw_rt),
              ("kf_level_wqa_onset_warned", "+ AORC rain to midnight", BLUE, WHITE, f"{fw_aorc}; {fw_hind}"),
              ("e4b_wqaw_l1_onset_pod", "+ within-day AORC rain", BLUE, WHITE, f"AORC: {fw_aorc}"),
              ("kf_level_wqf_onset_warned", "perfect weather of the next day", BLUE_LT, DEEP, fw_up)]
    cx0, cx1 = X0 + 0.05, 8.45
    base_y, top_y = 5.08, TOP + 0.48
    vmax = 0.80
    ppu = (base_y - top_y) / vmax                                   # inches per unit share
    colw, gapw = 0.98, ((cx1 - cx0) - 6 * 0.98) / 5
    gl = group(s4, "Ladder")
    line(gl.shapes, cx0 - 0.05, base_y, cx1 + 0.05, base_y, GREY, 1.0, name="Ladder baseline")
    gy = base_y - go_w * ppu
    line(gl.shapes, cx0 - 0.05, gy, cx1 + 0.05, gy, GOOD, 1.75, dash=MSO_LINE_DASH_STYLE.DASH, name="GO line")
    close_group(gl)
    for i, (key, lab, col, ink, wording) in enumerate(ladder):
        v = float(V(key))
        x_ = cx0 + i * (colw + gapw)
        h_ = v * ppu
        gcol = group(s4, f"Ladder column {i + 1}")
        shape(gcol.shapes, MSO_SHAPE.RECTANGLE, (x_, base_y - h_, colw, h_), fill=col, name="Column")
        tb(gcol.shapes, (x_ - 0.12, base_y - h_ - 0.44, colw + 0.24, 0.40), [N(key)], size=20, bold=True, color=DEEP, align="c",
           anchor="b", wrap=False, inset=(0.02, 0.0), fill=WHITE, name="Column value")
        if wording:
            tb(gcol.shapes, (x_ + 0.02, base_y - h_ + 0.08, colw - 0.04, h_ - 0.12), [wording], size=11, italic=True, color=ink,
               align="c", anchor="t", ls=0.95, inset=(0.03, 0.0), name="Column wording")
        tb(gcol.shapes, (x_ - gapw / 2 + 0.02, base_y + 0.05, colw + gapw - 0.04, 0.62), [lab], size=12, color=SUBNAVY,
           align="c", anchor="t", ls=0.95, inset=(0.01, 0.0), name="Column label")
        close_group(gcol)
    tb(s4.shapes, (cx0, gy - 0.31, 3.4, 0.28), [f"GO: {N('target_go_warned')} of onsets warned"], size=12, bold=True,
       color=GOOD, align="l", anchor="b", wrap=False, name="GO line label")
    # inset: 2 and 3 days ahead
    ix, iw = 8.95, X1 - 8.95
    gi = group(s4, "Inset 2 and 3 days")
    iy, ih = TOP, 2.95
    shape(gi.shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (ix, iy, iw, ih), fill=PANEL, adj=0.05, name="Inset panel")
    tb(gi.shapes, (ix + 0.15, iy + 0.08, iw - 0.3, 0.30), ["2 and 3 days ahead"], size=15, bold=True, color=DEEP, wrap=False,
       name="Inset title")
    ib = iy + ih - 0.62
    ippu = 1.60 / 0.5                                              # inches per unit share inside the inset (0 to 50 %)
    pairs = [("2 days", "kf_l2_wq_onset_warned", "kf_l2_wqaf_sub_onset_warned"), ("3 days", "kf_l3_wq_onset_warned",
                                                                                     "kf_l3_wqaf_sub_onset_warned")]
    bwi = 0.55
    for j, (lab, k0, k1) in enumerate(pairs):
        gx0 = ix + 0.35 + j * 1.75
        for q, (k, col) in enumerate(((k0, DEEP), (k1, BLUE))):
            hv = float(V(k)) * ippu
            xx = gx0 + q * (bwi + 0.08)
            shape(gi.shapes, MSO_SHAPE.RECTANGLE, (xx, ib - hv, bwi, hv), fill=col, name=f"Inset bar {lab}")
            tb(gi.shapes, (xx - 0.15, ib - hv - 0.34, bwi + 0.3, 0.32), [N(k)], size=15, bold=True, color=DEEP, align="c",
               anchor="b", wrap=False, inset=(0.0, 0.0), name="Inset value")
        tb(gi.shapes, (gx0 - 0.1, ib + 0.03, 2 * bwi + 0.28, 0.26), [lab], size=12, color=SUBNAVY, align="c", wrap=False,
           name="Inset label")
    line(gi.shapes, ix + 0.2, ib, ix + iw - 0.2, ib, GREY, 0.75, name="Inset baseline")
    close_group(gi)
    glg = group(s4, "Inset legend")
    legend(glg.shapes, ix + 0.05, iy + ih + 0.08, iw - 0.05, [
        (DEEP, "rain table alone"), (BLUE, "+ AORC rain to midnight + GEFS forecast")],
        size=12, name="Inset legend")
    close_group(glg)
    gk4 = group(s4, "Takeaway")
    shape(gk4.shapes, MSO_SHAPE.RECTANGLE, (X0, BOTTOM - 0.50, 0.07, 0.48), fill=ORANGE, name="Takeaway bar")
    tb(gk4.shapes, (X0 + 0.16, BOTTOM - 0.52, X1 - X0 - 0.2, 0.52), [
        {"runs": [f"Simple models given the same AORC rain warn of {N('kf_level_wqa_base_linreg_wqa_onset_warned')} (ridge "
                  f"regression) and {N('kf_level_wqa_base_rf_wqa_onset_warned')} (random forest), with "
                  f"{N('kf_level_wqa_base_linreg_wqa_onset_false')} and {N('kf_level_wqa_base_rf_wqa_onset_false')} of their "
                  f"warnings false against the LSTM's {N('kf_level_wqa_onset_false')}: ",
                  (N("fw_info_not_net"), {"bold": True, "color": ORANGE_INK})]}], size=14, color=SUBNAVY, anchor="m", ls=0.95,
       name="Takeaway text")
    close_group(gk4)
    notes4 = (f"Thank you, {first(MO)}. Here is the ladder. Every column is the same network, a day ahead, on the same "
              f"{N('onsets_l1')} onsets of the test years; only the information changes. The rain table and the gauge give "
              f"{N('kf_level_wq_onset_warned')}, in navy. Add the rain up to the evening, from NLDAS, which closes at "
              f"{N('day_end_nldas_rounded')}: {N('kf_level_wqn_onset_warned')}. The green column is the fair test: the real-time "
              f"radar product, Stage IV, gives the rain up to midnight: {N('s4_stage4_share')}, {fw_rt}. The same rain from AORC, "
              f"the analysis of record, gives {N('kf_level_wqa_onset_warned')}; AORC is {fw_aorc}, {fw_hind}. Add the rain "
              f"within the issue day, hour by hour, and {N('e4b_wqaw_l1_onset_pod')}. A perfect forecast of the next day's "
              f"weather gives {N('kf_level_wqf_onset_warned')}, {fw_up}. The dashed line is GO, {N('target_go_warned')}: "
              "nothing reaches it, not even the perfect weather.\n"
              f"On the right, two and three days ahead. The rain table alone warns of only {N('kf_l2_wq_onset_warned')} and "
              f"{N('kf_l3_wq_onset_warned')}; the rain to midnight alone, {N('kf_l2_wqa_onset_warned')} and "
              f"{N('kf_l3_wqa_onset_warned')}; with the GEFS weather forecast as well, {N('kf_l2_wqaf_sub_onset_warned')} and "
              f"{N('kf_l3_wqaf_sub_onset_warned')}.\n"
              f"Is it the network that does this? No. Give the best simple model, the {best_model}, the same rain to midnight, "
              f"and it warns of {N('kf_level_wqa_base_linreg_wqa_onset_warned')} of the onsets, though with more false warnings. "
              f"So it is {N('fw_info_not_net')}.\n"
              f"One caution: about half of these warnings are still false, {N('kf_level_wqa_onset_false')} with the rain to "
              f"midnight. {first(A)} now shows what is new since the mid-term: the river itself.\n"
              f"If asked what the Stage IV check is: {N('fw_realtime')}; {N('s4_stage4_no_last_hour_share')} without its last "
              "hour.")
    set_notes(s4, notes4)
    slides.append((4, "What more information buys: the ladder", HI, 150, notes4))

    # ================= slide 5: new since the mid-term (the team lead, 90 s) =================
    R.slide = 5
    rows5 = [("wqa_s", "AORC rain to midnight (the base)"), ("wqah_s", "+ the river's state at the issue time"),
             ("wqawh_s", "+ within-day AORC rain"), ("wqawhf_s", "+ GEFS weather forecast"),
             ("wqap_s", f"perfect rain: {fw_up}")]
    honesty5 = [f"{sat_years} satellite-era split, {sat_onsets} onsets: {inside} · the midnight variant assumes no delivery delay "
                f"and approved gauge values · {n5} trainings per set · * {N('e4e_wqawh_s_l1_far_cap_far')} false on the test "
                f"years, above the cap chosen on the validation years · AORC: {fw_aorc}"]
    s5 = content_slide(prs, 5, "The river's record nears the perfect-rain ceiling", "New · the river at the issue time",
                       f"Onsets warned a day ahead on the satellite-era split ({sat_years}; {sat_onsets} onsets; {inside})",
                       pres["A"], honesty5)
    lx, lw5 = X0, 2.95
    bx0, bx1 = X0 + 3.05, 8.25
    ppu5 = (bx1 - bx0) / vmax
    ge = group(s5, "River-state chart")                         # one unit: legend, rows, GO line and the split caption
    gx_ = chips_row(ge.shapes, X0, TOP - 0.02, [(BLUE_LT, "at the level chosen on the validation years"),
                                                (BLUE, f"at the {cap30}-false level (GO budget)")], size=12, gap=0.3,
                    name="Chart legend")
    ry0, rh = TOP + 0.42, 0.72
    line(ge.shapes, bx0, ry0 - 0.05, bx0, ry0 + len(rows5) * rh, GREY, 1.0, name="Chart axis")
    gox = bx0 + go_w * ppu5
    line(ge.shapes, gox, ry0 - 0.08, gox, ry0 + len(rows5) * rh, GOOD, 1.75, dash=MSO_LINE_DASH_STYLE.DASH, name="GO line")
    tb(ge.shapes, (gox + 0.04, ry0 - 0.34, 0.9, 0.26), [f"GO {N('target_go_warned').replace('at least ', '')}"], size=11,
       bold=True, color=GOOD, align="l", wrap=False, name="GO label")
    for i, (s_, lab) in enumerate(rows5):
        y_ = ry0 + i * rh
        tb(ge.shapes, (lx, y_ + 0.02, lw5, rh - 0.06), [lab], size=13, color=ORANGE_INK if s_ == "wqah_s" else SUBNAVY,
           bold=(s_ == "wqah_s"), anchor="m", ls=0.95, name="Row label")
        v1, v2 = float(V(f"e4e_{s_}_l1_onset_pod")), float(V(f"e4e_{s_}_l1_far_cap_pod"))
        far2 = N(f"e4e_{s_}_l1_far_cap_far")
        star = "*" if float(V(f"e4e_{s_}_l1_far_cap_far")) > go_f + 1e-9 else ""
        perfect = s_ == "wqap_s"
        shape(ge.shapes, MSO_SHAPE.RECTANGLE, (bx0, y_ + 0.07, v1 * ppu5, 0.27), fill=None if perfect else BLUE_LT,
              pattern=(BLUE_LT, WHITE) if perfect else None, line=BLUE_LT if perfect else None, name="Bar fixed level")
        shape(ge.shapes, MSO_SHAPE.RECTANGLE, (bx0, y_ + 0.37, v2 * ppu5, 0.27), fill=None if perfect else BLUE,
              pattern=(BLUE, WHITE) if perfect else None, line=BLUE if perfect else None, name="Bar 30%-false level")
        t1 = N(f"e4e_{s_}_l1_onset_pod")
        tb(ge.shapes, (bx0 + v1 * ppu5 + 0.05, y_ + 0.06, width_pt(t1, 13, True) / 72 + 0.08, 0.29), [t1], size=13, bold=True,
           color=DEEP, anchor="m", wrap=False, fill=WHITE, name="Value fixed level")
        t2 = f"{N(f'e4e_{s_}_l1_far_cap_pod')} · {far2} false{star}"
        tb(ge.shapes, (bx0 + v2 * ppu5 + 0.05, y_ + 0.36, width_pt(t2, 13, True) / 72 + 0.08, 0.29), [{"runs": [
            (N(f"e4e_{s_}_l1_far_cap_pod"), {"bold": True}), (f" · {far2} false{star}", {"size": 11, "color": GREY})]}],
           size=13, color=DEEP, anchor="m", wrap=False, fill=WHITE, name="Value 30%-false level")
    ny = ry0 + len(rows5) * rh + 0.10
    pill(ge.shapes, (X0, ny, 6.2, 0.34), f"median NSE {N('e4e_wqa_s_l1_nse')} → {N('e4e_wqawh_s_l1_nse')} "
                                         f"({sat_years} split, {inside})", PANEL, color=DEEP, size=11,
         spc=0.0, name="NSE")
    close_group(ge)
    # TARGET · GO tiles (one unit with their note: the "now" numbers are of the satellite-era split)
    tx5, tw5 = 8.80, X1 - 8.80
    best_pod, best_far = N("e4e_wqawhf_s_l1_far_cap_pod"), N("e4e_wqawhf_s_l1_far_cap_far")
    warned_met = float(V("e4e_wqawhf_s_l1_far_cap_pod")) >= go_w
    false_met_sat = float(V("e4e_wqawhf_s_l1_far_cap_far")) <= go_f + 1e-9
    false_met_main = float(V("e4a_lstm_wqa_ll1_far_cap_test_onset_far")) <= go_f + 1e-9
    # like with like (deck critic, finding 1): both tiles stay on the satellite-era split; tile 1 at the 30%-false level,
    # tile 2 at the fixed level (the level of the GO rule's verdicts), with the capped-level false share in its note
    fixed_far_met = float(V("e4e_wqawhf_s_l1_onset_far")) <= go_f + 1e-9
    tiles5 = [("Onsets warned a day ahead", N("e4e_wqa_s_l1_far_cap_pod"), best_pod,
               f"≥ {N('target_go_warned').replace('at least ', '')}", "not met" if not warned_met else "met",
               f"{cap30}-false level"),
              ("Warnings that are false", N("e4e_wqa_s_l1_onset_far"), N("e4e_wqawhf_s_l1_onset_far"),
               f"≤ {N('target_go_false').replace('at most ', '')}", "not met" if not fixed_far_met else "met",
               "level set on validation")]
    gtg = group(s5, "Targets")
    th5 = 1.45
    for i, (head, mid, now, target, status, level) in enumerate(tiles5):
        y_ = TOP + 0.12 + i * (th5 + 0.24)
        shape(gtg.shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (tx5, y_, tw5, th5), fill=PANEL, adj=0.07, name="Target tile")
        pill(gtg.shapes, (tx5 + tw5 - 1.42, y_ - 0.15, 1.30, 0.30), "TARGET · GO", "C2501F", size=10.5, name="Badge")
        tb(gtg.shapes, (tx5 + 0.15, y_ + 0.15, tw5 - 0.3, 0.28), [head], size=13, bold=True, color=SUBNAVY, wrap=False,
           anchor="m", name="Target head")
        tb(gtg.shapes, (tx5 + 0.15, y_ + 0.40, tw5 - 0.3, 0.50), [{"runs": [
            (f"{mid} → ", {"color": GREY, "size": 17}), (f"{now} → ", {"color": BLUE_INK, "size": 20, "bold": True}),
            (target, {"color": DEEP, "size": 23, "bold": True})]}], size=17, anchor="m", wrap=False, name="Target numbers")
        tb(gtg.shapes, (tx5 + 0.15, y_ + 0.88, tw5 - 0.3, 0.24), [f"AORC alone → best model, {level}"], size=10.5, color=GREY,
           wrap=False, anchor="m", name="Target axis")
        sw = width_pt(status.upper(), 10.5, True, spc=0.6) / 72 + 0.3
        pill(gtg.shapes, (tx5 + 0.15, y_ + th5 - 0.36, sw, 0.27), status.upper(),
             "B42318" if status.startswith("not") else AQUA_INK, size=10.5, name="Status")
    ny5 = TOP + 0.12 + 2 * (th5 + 0.24) - 0.10
    tb(gtg.shapes, (tx5, ny5, tw5, 0.62), [
        {"runs": f"{sat_years} split, {inside}. At the {cap30}-false level the best model has {best_far} false: met there, "
                 f"that split only" + (f"; main split {N('e4a_lstm_wqa_ll1_far_cap_test_onset_far')}" if not false_met_main else "")}],
       size=11, color=GREY, ls=0.95, name="Target note")
    v_go = "not met" if not (warned_met and false_met_sat) else "met"
    tb(gtg.shapes, (tx5, BOTTOM - 0.40, tw5, 0.40), [{"runs": [("GO needs both at once: ", {"bold": True}),
                                                                (v_go.upper(), {"bold": True, "color": CRIT})]}],
       size=16, color=SUBNAVY, anchor="m", wrap=False, name="GO verdict")
    close_group(gtg)
    notes5 = (f"What is new since the mid-term is the river itself. The gauge's own 15-minute record, reduced to hours, tells "
              "the model the river's state at the issue time. That record is complete only in recent years, so this test runs "
              f"on the satellite-era split, {sat_onsets} onsets, {inside}. At the level chosen on validation, the rain to midnight "
              f"alone warns of {N('e4e_wqa_s_l1_onset_pod')}; with the river's state, {N('e4e_wqah_s_l1_onset_pod')}; with "
              f"within-day rain, {N('e4e_wqawh_s_l1_onset_pod')}; with the GEFS forecast, {N('e4e_wqawhf_s_l1_onset_pod')}. At "
              f"the {cap30}-false level the gain is larger: {N('e4e_wqa_s_l1_far_cap_pod')} to {best_pod}, against "
              f"{N('e4e_wqap_s_l1_far_cap_pod')} for perfect rain, {fw_up}. The midnight cut assumes no delivery delay and "
              f"approved values. On the targets: {best_pod} warned, {N('target_go_warned')} needed; false warnings "
              f"{N('e4e_wqawhf_s_l1_onset_far')} at the chosen level. GO is not met.\n"
              f"If asked about delivery: with the gauge record cut at 22:00 instead of midnight the full model warns of "
              f"{N('e4e_wqawhf22_s_l1_onset_pod')} at {N('e4e_wqawhf22_s_l1_onset_far')} false at the fixed level, "
              f"{N('e4e_wqawhf22_s_l1_far_cap_pod')} at {N('e4e_wqawhf22_s_l1_far_cap_far')} false at the {cap30}-false level; "
              f"a service gets provisional values after a delay. Perfect rain at the fixed level: {N('e4e_wqap_s_l1_onset_pod')}, "
              f"{fw_up}.")
    set_notes(s5, notes5)
    slides.append((5, "New since the mid-term: the river at the issue time", A, 90, notes5))

    # ================= slide 6: conclusions and what was delivered (the team lead, 30 s) =================
    R.slide = 6
    three_scripts = re.search(r"baseline sets of scripts (\d+-\d+)", N("target_seeds_status"))
    three_scripts = three_scripts.group(1).replace("-", "–") if three_scripts else "02–10"
    new_sets = re.search(r"new sets of (\d+-\d+)", N("target_seeds_status"))
    new_sets = new_sets.group(1).replace("-", "–") if new_sets else "20–22"
    ext_ids = [str(x) for x in V("e1b_seeds") if str(x) not in {s.strip() for s in N("seeds_list").split(",")}]
    ext_txt = f"{ext_ids[0]}–{ext_ids[-1]}" if len(ext_ids) > 1 else "".join(ext_ids)
    seed_ext = (f"Seeds {ext_txt} of scripts {N('seed_extension_scripts')} were added on {N('seed_extension_date')}, after the "
                f"{n3}-seed test scores were known (disclosed); scripts {new_sets} had five from the start; scripts "
                f"{three_scripts} keep {n3}")
    s6 = content_slide(prs, 6, "Conclusions and what was delivered", "Conclusions", N("fw_finding"), pres["A"], [seed_ext])
    concl = [
        [("The information, not the network, limits the warning: ", {"bold": True}),
         f"the same LSTM warns of {N('kf_level_wq_onset_warned')} → {N('kf_level_wqa_onset_warned')} of the onsets a day ahead "
         f"with AORC rain to midnight ({fw_aorc}; {fw_hind})."],
        [("Half the onsets at the GO false budget: ", {"bold": True}),
         f"with the river's state, within-day rain and the GEFS forecast, {best_pod} warned at {best_far} false ({sat_years} "
         f"split, {inside}); the perfect-rain ceiling at the {cap30}-false level is {N('e4e_wqap_s_l1_far_cap_pod')} ({fw_up})."],
        [("GO is not met ", {"bold": True}), "by any model tried, not even with perfect rain."],
    ]
    if V("kf_practice_all_not_yet") is not True or warned_met:
        warn("the third conclusion says GO is not met, but the registry disagrees: check kf_practice_all_not_yet / E4e")
    cy = TOP + 0.02
    ch = [0.80, 1.00, 0.34]
    for i, runs in enumerate(concl):
        gc = group(s6, f"Conclusion {i + 1}")
        shape(gc.shapes, MSO_SHAPE.OVAL, (X0, cy + 0.02, 0.36, 0.36), fill=ORANGE, name="Number disc")
        tb(gc.shapes, (X0, cy + 0.02, 0.36, 0.36), [str(i + 1)], size=15, bold=True, color=WHITE, align="c", anchor="m",
           wrap=False, inset=(0.0, 0.0), name="Number")
        tb(gc.shapes, (X0 + 0.50, cy, 7.50, ch[i]), [{"runs": runs}], size=14, color=SUBNAVY, ls=1.0, name="Conclusion text")
        close_group(gc)
        cy += ch[i] + 0.08
    # references, numbered as in the report (short forms of its IEEE entries)
    ref_keys = ["kratzert2019", "nearing2024", "newman2015", "fall2023"]
    report_num = BSF.report_reference_numbers()
    if not all(k in report_num for k in ref_keys):
        problem("slide 6: a key reference is not numbered in the report DOCX")
    grf = group(s6, "Key references")
    rx6, rw6 = 8.70, X1 - 8.70
    kicker(grf.shapes, rx6, TOP - 0.02, rw6, "Key references · short forms", size=11, color="C2501F", name="References kicker")
    refs = []
    for k in sorted(ref_keys, key=lambda k: report_num.get(k, 999)):
        e = BSF.REFS.REFERENCES[k]["ieee"]
        m = re.match(r'^(.*?), "(.*?),?" (.*)$', e)
        authors, title, rest = (m.group(1), m.group(2), m.group(3)) if m else (e, "", "")
        au = authors.split(",")[0].replace(" et al.", "").strip()
        au = (au + " et al.") if ("et al." in authors or "," in authors) else au
        yrs = re.findall(r"\b((?:19|20)\d{2})\b", rest.split(", doi")[0])
        year = yrs[-1] if yrs else ""
        venue = re.split(r", vol\.|, \d{4}", rest)[0]
        refs.append({"runs": [(f"[{report_num.get(k, '?')}] ", {"bold": True}), f"{au}, “{title.rstrip(',')}”, {venue}, {year}."],
                     "space": 3})
    tb(grf.shapes, (rx6, TOP + 0.24, rw6, 2.40), refs, size=9, color=SUBNAVY, ls=0.95, name="Source references")
    close_group(grf)
    # the mid-term promises
    py6 = cy + 0.02
    gp6 = group(s6, "Mid-term promises")
    kicker(gp6.shapes, X0, py6, 6.0, "Mid-term promises", size=11, color="C2501F", name="Promises kicker")
    st_seeds = "done" if N("target_seeds_status").startswith("done") else "not done"
    st_gis = "done" if str(N("gis_layers_exported")).startswith("exported") else "not done"
    st_rev = "done" if int(V("review_five_seed_mismatches")) == 0 else "mismatches"
    team_n = int(V("team_checks_done") or 0) if not isinstance(V("team_checks_done"), str) else \
        int(re.match(r"\d+", N("team_checks_done")).group(0))
    st_team = "done" if team_n >= int(V("team_n_members")) else "open"
    st_radar = R.status_word("target_stretch_stage4_network") or "not attempted"
    ext_scripts_short = re.sub(r"(\d+), (\d+), (\d+) and (\d+)", r"\1–\4", N("seed_extension_scripts"))
    chips6 = [(st_seeds, f"Five trainings: {st_seeds}", f"{ext_scripts_short} (seeds {ext_txt} added later), {new_sets}; "
                                                        f"{three_scripts} keep {n3}"),
              (st_gis, f"GIS layers: {st_gis}", f"{N('basins_chosen')} of {N('basins_chosen')} basins, reviewed"),
              (st_rev, f"Independent reviews: {st_rev}", f"{N('review_five_seed_values_compared')} values, "
                                                         f"{N('review_five_seed_mismatches')} mismatches"),
              (st_team, "Team's own checks", f"{N('team_checks_done')}: not yet done"),
              (st_radar, "Radar-trained network", st_radar)]
    cwid = (X1 - X0 - 4 * 0.12) / 5
    for i, (st, head, body) in enumerate(chips6):
        cx = X0 + i * (cwid + 0.12)
        col = AQUA_T if st == "done" else (ORANGE_T if st == "open" else "EEF0F3")
        dot = AQUA_INK if st == "done" else (ORANGE if st == "open" else GREY)
        shape(gp6.shapes, MSO_SHAPE.ROUNDED_RECTANGLE, (cx, py6 + 0.29, cwid, 0.64), fill=col, adj=0.15, name="Promise chip")
        shape(gp6.shapes, MSO_SHAPE.OVAL, (cx + 0.10, py6 + 0.39, 0.14, 0.14), fill=dot, name="Promise dot")
        tb(gp6.shapes, (cx + 0.28, py6 + 0.31, cwid - 0.32, 0.60), [
            {"runs": [(head, {"bold": True})], "size": 11},
            {"runs": body, "size": 10, "color": INK2},
        ], color=SUBNAVY, ls=0.95, anchor="t", name="Promise text")
    close_group(gp6)
    hy6 = py6 + 0.96
    gh6 = group(s6, "Limitations and honesty")
    tb(gh6.shapes, (X0, hy6, X1 - X0, 0.86), [
        {"runs": [("Limitations: ", {"bold": True}),
                  f"AORC is {fw_aorc}; the river-state test assumes no delivery delay; {N('n_basins')} basins in one "
                  f"region; the stored sets keep {n3} trainings."]},
        {"runs": [(N("fw_who_did_what")[:-1] + "; independent AI agents reviewed it. ", {"italic": True}),
                  (N("fw_team_checks"), {"bold": True, "italic": True})], "space": 2},
    ], size=12, color=SUBNAVY, ls=1.0, name="Limitations text")
    close_group(gh6)
    gty = group(s6, "Thank you")
    ty6 = BOTTOM - 0.36
    line(gty.shapes, X0, ty6, X0 + 7.6, ty6, ORANGE, 2.5, name="Thank-you rule")
    tb(gty.shapes, (X0, ty6 + 0.02, X1 - X0, 0.37), [{"runs": [
        ("Thank you. ", {"color": DEEP}), ("We welcome your questions and your advice.", {"color": "C2501F"})]}],
       size=20, bold=True, anchor="m", wrap=False, inset=(0.02, 0.0), name="Thank-you text")
    close_group(gty)
    half = about_share(V("e4e_wqawhf_s_l1_far_cap_pod"), best_pod)
    notes6 = (f"Three conclusions. The information, not the network, limits the warning. The best model tried warns of {half} "
              "the onsets at the GO false budget, near the perfect-rain ceiling. GO is not met. Claude Code wrote and ran the code "
              "under my direction, independent AI agents reviewed it, and the team's own checks are not yet done. Thank you; we "
              "welcome your questions.\n"
              f"If asked about the seeds: {seed_ext[:1].lower() + seed_ext[1:]}; the five-seed numbers were recomputed by "
              f"independent reviewers, {N('review_five_seed_values_compared')} values, {N('review_five_seed_mismatches')} "
              "mismatches.")
    set_notes(s6, notes6)
    slides.append((6, "Conclusions and what was delivered", A, 30, notes6))

    # ---- save ----
    out = Path(args.out)
    owner = out.parent / ("~$" + out.name)
    if owner.exists() and not args.force:
        sys.exit(f"{out} is open in WPS or PowerPoint (owner file {owner.name}); pass another --out (or --force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"saved {out} ({len(prs.slides)} slides)")
    write_notes(Path(args.notes), slides, R, cover_which, out)
    return out, slides


def write_notes(path, slides, R, cover_which, pptx_path):
    today = _dt.date.today().isoformat()
    team = R.text("team_checks_done")
    lines = [f"# Final presentation (6 slides) — speaker notes · {R.text('final_presentation')} · {R.text('group_number')}", "",
             f"Deck: `{pptx_path.name}` (built {today} by `code/build_slides_final6.py` on the professor's template; cover: "
             f"{cover_which}). Every number comes from `results/final_report_numbers.json`.",
             f"Pace rule: at most {PACE} spoken words a second ('If asked' lines are not spoken). Four speakers, "
             f"{sum(s[3] for s in slides)} s in all.",
             f"'We' means the project. Claude Code wrote and ran the code under the team lead's direction and independent AI agents "
             f"reviewed it; the team's own checks are not yet done (team.json: {team}).", "",
             "## Timing", "", "| Slide | Title | Speaker | Seconds | Words | Words/s |", "|---|---|---|---|---|---|"]
    per = {}
    for n, title, sp, secs, notes in slides:
        spoken = " ".join(p for p in notes.split("\n") if not p.startswith("If asked"))
        words = len(spoken.split())
        per[sp] = per.get(sp, 0) + secs
        wps = words / secs
        flag = " ⚠" if wps > PACE + 1e-9 else ""
        lines.append(f"| {n} | {title} | {sp} | {secs} | {words} | {wps:.2f}{flag} |")
        if flag:
            problem(f"slide {n}: {words} words in {secs} s = {wps:.2f} words/s, over the pace rule of {PACE}")
    lines += ["", "| Speaker | Seconds |", "|---|---|"] + [f"| {s} | {t} |" for s, t in per.items()]
    clock, t0 = [], 0
    for n, title, sp, secs, notes in slides:
        clock.append(f"slide {n} at {t0 // 60}:{t0 % 60:02d}")
        t0 += secs
    lines += ["", "Clock checkpoints: " + ", ".join(clock) + f", end at {t0 // 60}:{t0 % 60:02d}.", ""]
    for n, title, sp, secs, notes in slides:
        lines += [f"## Slide {n} — {title} · {sp} · {secs} s", ""] + notes.split("\n") + [""]
    lines += ["## Registry keys used, by slide (results/final_report_numbers.json; sources in code/final_report/NUMBERS_KEYS.md)", ""]
    for n in sorted(R.by_slide):
        lines.append(f"- slide {n}: " + ", ".join(f"`{k}`" for k in sorted(R.by_slide[n])))
    if WARN:
        lines += ["", "## Builder warnings", ""] + [f"- {w}" for w in WARN]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"notes written: {path}")


# ---- checks on the saved file ------------------------------------------------------------------------------------------------
def iter_shapes(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type is not None and sh.shape_type == 6:          # MSO_SHAPE_TYPE.GROUP
            yield from iter_shapes(sh.shapes)


def group_text(sh):
    parts = []
    for s in iter_shapes(sh.shapes):
        if s.has_text_frame and s.text_frame.text.strip():
            parts.append(s.text_frame.text.strip())
    return " | ".join(parts)


def wording_rules(R):
    t = R.text
    up = t("fw_upper")
    rules = [(t("s4_stage4_share"), [t("fw_realtime_short")], r"Stage IV|radar", "Stage IV"),
             (t("kf_level_wqa_onset_warned"), [t("fw_aorc"), t("fw_hindsight")], r"AORC", "AORC"),
             (t("kf_level_wqf_onset_warned"), [up], r"perfect", "perfect weather"),
             (t("e4e_wqap_s_l1_onset_pod"), [up], r"perfect", "perfect rain"),
             (t("e4e_wqap_s_l1_far_cap_pod"), [up], r"perfect|ceiling", "perfect rain, capped level")]
    for k in ("e4e_wqawhf_s_l1_far_cap_pod", "e4e_wqah_s_l1_onset_pod", "e4e_wqawhf_s_l1_onset_pod"):
        rules.append((t(k), ["comparable only inside it"], r"split|river's state|GEFS|onsets warned", "satellite-era split"))
    return rules


def check_units(units, rules, i, report):
    for u in units:
        for share, needed, ctx, label in rules:
            if re.search(r"(?<![\d.])" + re.escape(share), u) and re.search(ctx, u):
                lacking = [w for w in needed if w not in u]
                if lacking:
                    report.append((i, label, u, lacking))


def validate(pptx_path, R):
    prs = Presentation(str(pptx_path))
    report = []
    rules = wording_rules(R)
    gno = R.text("group_number")
    if len(prs.slides) != 6:
        problem(f"{len(prs.slides)} slides, expected 6")
    lay2 = " ".join(sh.text_frame.text for sh in prs.slide_layouts[1].shapes if sh.has_text_frame).strip()
    if lay2 != gno:
        problem(f"layout 2 fixed text is {lay2!r}, expected {gno!r}")
    for i, slide in enumerate(prs.slides, 1):
        want = "Blank" if i == 1 else prs.slide_layouts[1].name
        if slide.slide_layout.name != want:
            problem(f"slide {i}: layout {slide.slide_layout.name!r}, expected {want!r}")
        units, texts = [], []
        for sh in slide.shapes:
            if sh.shape_type == 6:
                units.append(group_text(sh))
            elif sh.has_text_frame:
                units += [p.text for p in sh.text_frame.paragraphs if p.text.strip()]
        for sh in iter_shapes(slide.shapes):
            if sh.left < -50000 or sh.top < -50000 or sh.left + sh.width > SLIDE_W + 50000 or sh.top + sh.height > SLIDE_H + 50000:
                problem(f"slide {i}: shape {sh.name!r} outside the slide ({sh.left}, {sh.top}, {sh.width}, {sh.height})")
            if sh.has_text_frame and sh.text_frame.text.strip():
                texts.append(sh.text_frame.text.strip())
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.size is not None and r.font.size.pt < 10 and not sh.name.startswith("Source") and r.text.strip():
                            problem(f"slide {i}: {r.font.size.pt:g} pt run in {sh.name!r}: {r.text[:40]!r}")
        notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
        if not notes:
            problem(f"slide {i}: no speaker notes")
        units += ["notes: " + p for p in notes.split("\n") if p.strip()]
        check_units(units, rules, i, report)
        blob = "\n".join(texts) + "\n" + notes
        for tok in FORBIDDEN:
            if re.search(r"(?<![A-Za-z])" + re.escape(tok) + r"(?![a-z])", blob):
                problem(f"slide {i}: forbidden token {tok!r}")
        if i == 1 and gno not in blob:
            problem("slide 1: the group number is not on the cover")
        print(f"\n===== slide {i} ({slide.slide_layout.name}) =====")
        for t in texts:
            print("  " + t.replace("\n", " / ")[:300])
        print("  -- notes: " + notes.replace("\n", " ")[:2500])
    if report:
        for i, label, u, lacking in report:
            problem(f"slide {i}: {label} share without its wording ({'; '.join(lacking)}): {u[:160]!r}")
    else:
        print("\nwording check: every unit that prints a flagged share carries its fixed wording")
    return not PROBLEMS


# ---- rendering (optional) ----------------------------------------------------------------------------------------------------
def lo_profile():
    """A private LibreOffice profile whose font replacement table maps Aptos / Aptos Display to Arial (this PC has no Aptos)."""
    prof = Path(tempfile.gettempdir()) / "flood_deck_lo_profile"
    xcu = prof / "user" / "registrymodifications.xcu"
    if not xcu.exists() or "Font/Substitution" not in xcu.read_text(encoding="utf-8"):
        xcu.parent.mkdir(parents=True, exist_ok=True)
        items = ['<item oor:path="/org.openoffice.Office.Common/Font/Substitution"><prop oor:name="Replacement" oor:op="fuse">'
                 '<value>true</value></prop></item>']
        for i, src in enumerate(("Aptos", "Aptos Display")):
            items.append('<item oor:path="/org.openoffice.Office.Common/Font/Substitution/FontPairs">'
                         f'<node oor:name="_{i}" oor:op="replace"><prop oor:name="Always" oor:op="fuse"><value>true</value></prop>'
                         '<prop oor:name="OnScreenOnly" oor:op="fuse"><value>false</value></prop>'
                         f'<prop oor:name="ReplaceFont" oor:op="fuse"><value>{src}</value></prop>'
                         '<prop oor:name="SubstituteFont" oor:op="fuse"><value>Arial</value></prop></node></item>')
        if xcu.exists():
            s = xcu.read_text(encoding="utf-8").replace("</oor:items>", "\n".join(items) + "\n</oor:items>")
        else:
            s = ('<?xml version="1.0" encoding="UTF-8"?>\n<oor:items xmlns:oor="http://openoffice.org/2001/registry" '
                 'xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
                 + "\n".join(items) + "\n</oor:items>\n")
        xcu.write_text(s, encoding="utf-8")
    return prof


def render_pdf(pptx, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    prof = lo_profile().as_posix()
    cmd = [str(SOFFICE), f"-env:UserInstallation=file:///{prof}", "--headless", "--convert-to", "pdf", "--outdir", str(outdir),
           str(pptx)]
    subprocess.run(cmd, check=True, timeout=600)
    pdf = outdir / (Path(pptx).stem + ".pdf")
    print(f"PDF: {pdf}")
    return pdf


def render_png(pdf, outdir, dpi=90):
    import fitz
    outdir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf))
    for i, page in enumerate(doc, 1):
        page.get_pixmap(dpi=dpi).save(str(outdir / f"slide_{i}.png"))
    print(f"PNG: {doc.page_count} pages in {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT_PPTX))
    ap.add_argument("--notes", default=str(OUT_NOTES))
    ap.add_argument("--cover", choices=("sample", "photo"), default="sample",
                    help="sample: the professor's own cover slide edited in place (default); photo: the flood-photo cover of "
                         "build_slides_final.py under the same veil and band")
    ap.add_argument("--pdf-dir", default=None, help="render the PDF with LibreOffice into this folder")
    ap.add_argument("--png-dir", default=None, help="write 90-dpi PNGs of the PDF pages into this folder (needs --pdf-dir)")
    ap.add_argument("--force", action="store_true", help="write even when an owner file says the PPTX is open")
    args = ap.parse_args()
    out, slides = build(args)
    R = BSF.Registry(REGISTRY)
    ok = validate(out, R)
    if args.pdf_dir:
        pdf = render_pdf(out, Path(args.pdf_dir))
        if args.png_dir:
            render_png(pdf, Path(args.png_dir))
    if WARN:
        print(f"{len(WARN)} warning(s)", file=sys.stderr)
    if not ok:
        print(f"{len(PROBLEMS)} check(s) failed", file=sys.stderr)
    else:
        print("checks passed: 6 slides, layouts, Group number, shapes inside, notes, no forbidden token, sizes, wordings, pace")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
