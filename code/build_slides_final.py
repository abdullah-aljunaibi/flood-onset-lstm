"""Build the FINAL 14-slide deck of Project-Flood on the professor's template (hand-in item 4, talk on 2 November 2026).

    python -X utf8 code/build_slides_final.py            (from the project root; system Python 3.13 with python-pptx 1.0.2 and Pillow)
    options: --cover auto|photo|map   --refresh-template   --out <pptx>   --notes <md>

What it does
  1. copies "Sample ppt.pptx" (the professor's template, read-only in the download folder) to code/final_report/deck_template.pptx,
  2. opens that copy, deletes its four sample slides (sldId entries + rels) and keeps master, layouts, theme and logo,
  3. writes the group number ("Group #__" while config.json holds null) into the fixed text box of slide layout 2,
  4. adds the 14 slides of the final-presentation plan (scratchpad audits/agent3.md section 3): cover on layout "Blank",
     content slides on layout 2 with the sample's geometry (heading box 32 pt white at (1244600, 1) 9753600 x 889847,
     sub-heading 16 pt bold 1B2B4B at y 889848, footer line 13.33 pt 64748B, slide number 26.67 pt white top right),
  5. writes the speaker notes into the PPTX and into 4_Final_Project_Presentation_notes.md (speaker and seconds per slide),
  6. reopens the PPTX and checks: 14 slides, every shape inside the slide, every slide has notes, no forbidden token
     (UNVERIFIED, MISSING, TODO, {{...}}) in any text; then prints the text of every slide.

Honesty rules (HANDOFF.md section 7): EVERY number on a slide or in the notes is read from results/final_report_numbers.json
(the registry built by code/final_report/numbers.py; keys in code/final_report/NUMBERS_KEYS.md) through R.text()/R.val();
nothing is typed. The fixed wordings come from the fw_ keys of the registry. Nothing is read from results/diagnostics_2026-10-28/.
The E4 slot is an optional block: E4a, E4b and E4e numbers appear only because their results files AND review records exist
(e4x_file_present, e4x_review_file_present; on 29 October 2026 all three were reviewed, "numbers hold with wording fixes");
without a review record the slides print status sentences without numbers. The 30%-false operating points of the E4b/E4e sets
have no registry key yet (numbers.py flattens summary_test only): op_point() looks for the documented key names first
(e4e_<set>_l1_far_cap_pod / _far, the E4a shape, the E4a keys of a stored set) and otherwise reads
results/flow_state.json -> operating_points itself, warns, and lists every such number with its source in the notes file.
The only numerals typed in this file are structural: slide numbers, the seconds of the talk plan, the crop fraction of the
data-flow figure (DATA_CROP of build_slides_short.py), the lead horizon "1 to 3 days ahead" (flood.py LEADS = 1, 2, 3), the
hour names of the E4e delivery variants (22:00, midnight) and the dates of the E4 training pass (28–29 October 2026).

Nothing is trained or computed here. Registry keys used by each slide are listed at the end of the notes file.

Fixes of 29 October 2026 after the first critic pass (scratchpad audits/critic1.md, items 6, 9, 11, 15, 21, 22):
  - counts: slides 6, 11 and 14 print review_records / review_rounds / count_code_py_files from the registry; the 8 October
    data-flow counts (stored_prediction_sets, stored_saved_networks) are dated "as of" date_data_record_written, and the live
    file counts (count_preds_npz, count_models_pt) carry the date the registry was last written (its file time);
  - pace: every slide's notes stay at or below PACE spoken words a second (the builder warns above it, no 10 % allowance);
  - seeds: numbers quoted from before 28 October are seeds_per_set-seed numbers; the E4b/E4e sets of 28–29 October have
    len(e4b_seeds) / len(e4e_seeds) seeds (slides 11, 12, 13);
  - the share of a rain class warned with the perfect forecast (rescue_rain_75_100_perfect_share) says which onsets it counts;
  - the reference list of slide 14 is numbered as in the report: the numbers are read from the built DOCX (section 11,
    report_reference_numbers()) and fall back to the deck's own order of first citation with a warning;
  - the fixed wordings stand in the same cell or bullet as every Stage IV 48 %, AORC 51 % and perfect-forecast 64 %, and the
    E4e share at the capped level says "at the 30%-false level, 2010–2014 split"; validate() lists any unit that lacks them.

Fixes of 29 October 2026 after the second critic pass (scratchpad audits/round3_critic2_documents.md, items 4, 6, 10, 11 and the
note on the slide-12 footer):
  - notes: the fixed wordings stand beside every perfect-forecast / perfect-rain number (slides 9, 10, 11, 12) and beside the AORC
    share (slides 2 and 13, the slide-10 "If asked" line) in the spoken notes too, and beside the Stage IV share of slide 11;
    check_wordings() reads the notes paragraphs as units as well; the pace rule holds with words cut elsewhere (no seconds moved);
  - notes: "we computed / we measured / we read" became "the code computed / the pipeline measured / the project read", and the
    notes file header says that "we" means the project and that the team's own checks are not yet done (team.json);
  - review counts: when the registry carries document_review_passes, its review_records / review_rounds / review_checks_listed
    (or review_checks, if numbers.py ever renames it) count the experiment reviews only: the slides then print "experiment review
    records" and name the critic passes over the documents separately; without that key the earlier, all-inclusive keys are used;
  - slide 14: while repo_url / data_bundle_url are bracketed placeholders the links line reads "Repository and data links: see the
    report, section 9.2 and the datasets page" (the section number 9.2 is a structural constant of the report, typed here);
  - the footer's hindsight clause says "AORC 51%" so that it is not read as the E4e share at the capped level (slide 12's table).

Fixes of 30 October 2026 (five seeds: seeds 3 and 4 of scripts 15-18 trained on the Windows PC after the three-seed test scores
were known; registry keys seed_extension_*, stored_sets_three_seeds_list, target_seeds_status, review_five_seed_*, rev_*_latest_*):
  - seeds: every seed label names both counts: five (len(e1b_seeds)) for the sets of scripts 15-18 and the E4 sets of 20-22,
    three (seeds_per_set) for the stored sets of scripts 02-10 (stored_sets_three_seeds_list), read per set where a slide shows
    a single set (the <set>_l1_n_seeds keys); slides 5-14 and the notes;
  - hindsight: fw_hindsight now carries the five-seed bound ("about 4 points"); slide 13's "If asked" line gives both bounds once
    (hindsight_bound_five_seeds_*, hindsight_three_seed_wording, hindsight_bound_three_seeds_hits);
  - slide 11: the five-trainings row reads target_seeds_status (status "partly"), the review row adds the five-seed review rounds
    (review_five_seed_values_compared / _mismatches, rev_*_latest_verdict), the GIS row the layer review (gis_layers_review_*);
  - slide 13: E3 with its onset result (e3_onsets_*, e3_onset_seed_reading, e3_mcnemar_p), no "moved operating point";
  - slides 6 and 14: work_trainings / work_hours include the seed-extension trainings on the Windows PC; slide 14 counts the
    Windows-PC trainings (seed_extension_trainings + E4) and discloses when seeds 3-4 were added.

Fixes of 30 October 2026 after the third critic pass (results/review_documents_2026-10-30.json, items M2, S1-S3, S6-S9, N1):
  - slide 11: the mid-term "below 50% false" row is built from its two sources (lstm_wqaf_sub_l1_onset_far, lstm_wqa_l1_onset_far)
    instead of the typed text of midterm_ppt_target_50_status (M2); the false-share row prints the E4a and E4e false shares at the
    capped level and reads "not met on the main split", met alone only on the satellite-era split (S3);
  - slide 3: the hourly USGS gauge record of E4e counts as an extra input (six inputs, four checks and maps, as in the report), and
    the bullet dates the data-flow drawing and says it still shows the USGS record as a check (S2, S1);
  - slide 10: the 1-day NSE reading by the seed rule (S8) and the seed rule's onset reading beside the moved operating point (N1);
  - slide 12: review dates from each record's "when" (date_review_e4a/b/e) plus the five-seed round (rev_<record>_latest_*) (S6);
  - slide 13 notes: the three-seed hindsight bound is dated by round 1 of the E1b review (date_review_e1b, 7 October) (S7);
  - the critic-pass count is document_review_passes of the registry rebuilt after the third critic record (S9).
"""
import argparse
import datetime as _dt
import json
import re
import shutil
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "final_report_numbers.json"
CONFIG = ROOT / "code" / "final_report" / "config.json"
SAMPLE = ROOT / "code" / "final_report" / "Sample ppt.pptx"   # the course's slide template as received (not part of the public repository)
TEMPLATE = ROOT / "code" / "final_report" / "deck_template.pptx"
FIG = ROOT / "figures"
COVER_PHOTO = ROOT / "code" / "proposal_web" / "assets" / "cover_photo.jpg"
COVER_PHOTO_JSON = COVER_PHOTO.with_suffix(".json")
HANDIN = ROOT.parent / "Handin_2026-11-02"   # the hand-in folder, next to the project folder
OUT_PPTX = HANDIN / "4_Final_Project_Presentation_Flood.pptx"
OUT_NOTES = HANDIN / "4_Final_Project_Presentation_notes.md"
REPORT_DOCX = HANDIN / "3_Final_Project_Report_Flood.docx"   # the built report: its section 11 numbers the references the deck reuses
sys.path.insert(0, str(ROOT / "code" / "final_report"))
import references as REFS  # noqa: E402  (torch-free: only collections and difflib)

# ---- geometry of the sample (EMU; 914400 = 1 in) -------------------------------------------------------------------------
SLIDE_W, SLIDE_H = 12192000, 6858000
HEAD = (1244600, 1, 9753600, 889847)          # heading box in the top band: 32 pt bold white, anchor centre
SLDNUM = (11634069, 116543, 367408, 502766)   # slide number field: 26.67 pt white, in the band at the right
SUB_X, SUB_Y, SUB_W, SUB_H = 548640, 889848, 11094720, 365760   # sub-heading: 16 pt bold 1B2B4B (the sample's box, widened)
BODY_X, BODY_W = 548640, 11094720             # content area x: 0.60 in to 12.73 in
BODY_RIGHT = BODY_X + BODY_W
FOOT = (548640, 5870000, 11094720, 600000)    # footer: 13.33 pt grey 64748B (+ 11 pt honesty lines), anchored at the bottom
BODY_BOTTOM = FOOT[1] - 50000
COVER_VEIL_ALPHA, COVER_BAND_ALPHA = 45000, 55000   # the sample's white veil (45 %) and band (55 %)
COVER_BAND = (33440, 2214371, 12109877, 1833071)
DATA_CROP = 0.528                             # top share of data_flow_talk.png shown (build_slides_short.py DATA_CROP)
LEADS = "1 to 3 days ahead"                   # the lead horizon of every model (flood.py LEADS = 1, 2, 3): a design constant
PACE = 2.3                                    # words per second the talk plan allows (results/slide_plan_5.json words_per_second)

WHITE, NAVY, GREY, BLUE, TITLE_BLUE, ORANGE, BAND_BLUE = "FFFFFF", "1B2B4B", "64748B", "005CA2", "035CBA", "D97C27", "005BAC"
ZEBRA = "EEF3F8"
MARG_H, MARG_V = 54000, 30000                 # text-box and cell margins (EMU)
SHADOW_XML = ('<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:outerShdw blurRad="38100" '
              'dist="38100" dir="2700000" algn="tl"><a:srgbClr val="000000"><a:alpha val="43137"/></a:srgbClr></a:outerShdw>'
              '</a:effectLst>')

FORBIDDEN = ("UNVERIFIED", "MISSING", "TODO", "{{", "}}", "None")
WARNINGS = []
CURRENT = {"n": 0}                            # the slide being built (for the warnings)
DIRECT = {}                                   # label -> (source, text, slide): numbers read from a results file because the registry has no key yet
OPTIONAL_FILES = {"e4e": ROOT / "results" / "flow_state.json", "e4b": ROOT / "results" / "within_day_inputs.json"}
_OP_CACHE = {}


def warn(msg):
    WARNINGS.append(msg)
    print("WARNING: " + msg, file=sys.stderr)


def review_date(text):
    """'independent AI reviewer agent (Claude), 2026-10-29' (a review record's who field) -> '29 Oct 2026'; the text itself without a date."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%d %b %Y").lstrip("0") if m else text


def file_date(path):
    """The day a file was last written ('29 October 2026'): the date of the registry's live file counts."""
    return _dt.date.fromtimestamp(Path(path).stat().st_mtime).strftime("%d %B %Y").lstrip("0")


def report_reference_numbers():
    """{reference key: n} as the built report numbers its references (DOCX section 11, one '[n] <IEEE entry>' paragraph each),
    matched to code/final_report/references.py by the entry text. Empty (with a warning) when the DOCX is missing or unreadable;
    the deck then falls back to its own order of first citation (critic 1, item 21)."""
    if not REPORT_DOCX.exists():
        warn(f"report DOCX not found ({REPORT_DOCX}): the reference list keeps the deck's own numbering")
        return {}
    try:
        from docx import Document
    except ImportError:
        warn("python-docx not installed: the reference list keeps the deck's own numbering")
        return {}
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    by_text = {norm(v["ieee"]): k for k, v in REFS.REFERENCES.items()}
    out = {}
    for p in Document(str(REPORT_DOCX)).paragraphs:
        m = re.match(r"^\[(\d+)\]\s+(.*)$", norm(p.text))
        if m and m.group(2) in by_text:
            out.setdefault(by_text[m.group(2)], int(m.group(1)))
    if not out:
        warn(f"no numbered reference of references.py found in {REPORT_DOCX.name}: the reference list keeps the deck's own numbering")
    return out


def op_point(R, prefix, set_name, field, lead=1):
    """The 30%-false operating point of an E4b/E4e set, onsets on the test years: `field` = pod (onsets warned) or far (warnings false).
    The registry key when one exists (the names numbers.py documents, the E4a shape, then the E4a keys of a stored set, which hold
    the same forecasts at the same trigger), else the results file itself (operating_points.sets.<set>.l<lead>.rules.far_cap.test.onset),
    recorded in DIRECT and warned once per number. Returns (text in the registry's percent format, value)."""
    s = set_name.lower()
    cands = [f"{prefix}_{s}_l{lead}_far_cap_{field}", f"{prefix}_{s}_l{lead}_far_cap_test_onset_{field}",
             f"e4a_lstm_{s}_ll{lead}_far_cap_test_onset_{field}"]
    for k in cands:
        if R.has(k):
            return R.text(k), R.val(k)
    path = OPTIONAL_FILES[prefix]
    if path not in _OP_CACHE:
        _OP_CACHE[path] = json.loads(path.read_text(encoding="utf-8"))
    node = _OP_CACHE[path]["operating_points"]["sets"][set_name][f"l{lead}"]["rules"]["far_cap"]
    v = node["test"]["onset"][field]
    text = f"{v * 100:.0f}%"
    source = f"{path.relative_to(ROOT).as_posix()} -> operating_points.sets.{set_name}.l{lead}.rules.far_cap.test.onset.{field} (trigger x{node['trigger']:.2f})"
    if cands[0] not in DIRECT:
        DIRECT[cands[0]] = (source, text, R.slide)
        warn(f"registry key {cands[0]} absent (also tried {cands[1]}, {cands[2]}): {text} read directly from {source}")
    return text, v


# ---- the registry ----------------------------------------------------------------------------------------------------------
class Registry:
    """results/final_report_numbers.json: key -> {text, value, source, note}. text() is what a document prints."""

    def __init__(self, path):
        self.d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.used = {}          # key -> text, everything printed, for the notes file
        self.by_slide = {}      # slide number -> sorted keys
        self.slide = 0

    def has(self, key):
        return key in self.d and isinstance(self.d[key], dict)

    def _entry(self, key):
        if not self.has(key):
            raise KeyError(f"registry key missing: {key}")
        self.used[key] = self.d[key].get("text")
        self.by_slide.setdefault(self.slide, set()).add(key)
        return self.d[key]

    def text(self, key):
        t = self._entry(key).get("text")
        if t is None:
            raise KeyError(f"registry key without text: {key}")
        return str(t)

    def val(self, key):
        return self._entry(key).get("value")

    def note(self, key):
        return self._entry(key).get("note") or ""

    def status_word(self, key):
        """The status word the registry's note carries for a target (TARGET, NOT MET / NOT DONE / PARTLY / NOT ATTEMPTED)."""
        note = (self.note(key) or "").upper()
        for token in ("NOT ATTEMPTED", "NOT DONE", "NOT MET", "PARTLY", "MET"):
            if token in note:
                return token.lower()
        return ""


# ---- small helpers ---------------------------------------------------------------------------------------------------------
def est_lines(text, width_emu, size_pt, bold=False, spc_pt=0.0):
    """Rough count of wrapped lines: average character width 0.52 em (0.56 em bold) plus the letter spacing."""
    cw = (0.56 if bold else 0.52) * size_pt + spc_pt
    width_pt = max(10.0, width_emu / 12700 - 2 * MARG_H / 12700)
    cpl = max(8, int(width_pt / cw))
    lines = 0
    for para in str(text).split("\n"):
        lines += max(1, len(textwrap.wrap(para, cpl)) if para.strip() else 1)
    return lines


def runs_text(item):
    """The plain text of a paragraph spec: str, or a list of runs (str | (str, fmt))."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return runs_text(item.get("runs", ""))
    return "".join(r if isinstance(r, str) else r[0] for r in item)


def _insert_bullet(p, char="•", marL=228600, indent=-228600):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(marL))
    pPr.set("indent", str(indent))
    bu = etree.Element(qn("a:buChar"))
    bu.set("char", char)
    for i, child in enumerate(list(pPr)):
        if child.tag in (qn("a:tabLst"), qn("a:defRPr"), qn("a:extLst")):
            pPr.insert(i, bu)
            return
    pPr.append(bu)


def _format_run(r, fmt, size, color, bold, italic, font=None):
    f = r.font
    f.size = Pt(fmt.get("size", size))
    f.bold = fmt.get("bold", bold)
    f.italic = fmt.get("italic", italic)
    f.color.rgb = RGBColor.from_string(fmt.get("color", color))
    name = fmt.get("font", font)
    if name:
        f.name = name
    if fmt.get("spc"):
        r._r.get_or_add_rPr().set("spc", str(fmt["spc"]))
    if fmt.get("shadow"):
        r._r.get_or_add_rPr().append(etree.fromstring(SHADOW_XML))


def add_text(slide, box, paras, *, size=14, color=NAVY, bold=False, italic=False, align="l", anchor="t", font=None,
             line_spacing=1.08, space_before=0, name=None, check=True):
    """A text box. paras: list of paragraph specs (str | list of runs | dict(runs=..., size=, bold=, bullet=, color=, space=))."""
    x, y, w, h = box
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Emu(MARG_H)
    tf.margin_top = tf.margin_bottom = Emu(MARG_V)
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    est_pt = 0.0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        spec = para if isinstance(para, dict) else {"runs": para}
        runs = spec.get("runs", "")
        if isinstance(runs, str):
            runs = [runs]
        psize = spec.get("size", size)
        p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[spec.get("align", align)]
        p.line_spacing = spec.get("line_spacing", line_spacing)
        sb = spec.get("space", space_before)
        if sb:
            p.space_before = Pt(sb)
        if spec.get("bullet"):
            _insert_bullet(p, marL=spec.get("marL", 228600), indent=-spec.get("marL", 228600))
        for run in runs:
            txt, fmt = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = txt
            _format_run(r, fmt, psize, spec.get("color", color), spec.get("bold", bold), spec.get("italic", italic), font)
        est_pt += est_lines(runs_text(runs), w - spec.get("marL", 0), psize, spec.get("bold", bold),
                            spc_pt=(spec.get("spc", 0) or 0) / 100) * psize * p.line_spacing * 1.15 + sb
    if check and est_pt * 12700 + 2 * MARG_V > h * 1.02:
        warn(f"text may overflow its box on slide {CURRENT['n']}: ~{est_pt:.0f} pt of text in a {h / 12700:.0f} pt box: "
             f"{runs_text(paras[0] if not isinstance(paras[0], dict) else paras[0].get('runs', ''))[:60]!r}")
    return tb


def add_rect(slide, box, rgb, alpha=None, name=None):
    x, y, w, h = box
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h))
    if name:
        shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = RGBColor.from_string(rgb)
    shp.line.fill.background()
    shp.shadow.inherit = False
    if alpha is not None:
        srgb = shp._element.spPr.find(qn("a:solidFill")).find(qn("a:srgbClr"))
        etree.SubElement(srgb, qn("a:alpha")).set("val", str(alpha))
    shp.text_frame.text = ""
    return shp


def add_slidenum(slide, number):
    """The sample's slide number: a text box holding a slidenum field, 26.67 pt white, top right in the band."""
    x, y, w, h = SLDNUM
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tb.name = "Slide number"
    tf = tb.text_frame
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = Emu(0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    fld = etree.Element(qn("a:fld"), attrib={"id": "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", "type": "slidenum"})
    rPr = etree.SubElement(fld, qn("a:rPr"), attrib={"lang": "en-US", "sz": "2667"})
    sf = etree.SubElement(rPr, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:schemeClr")).set("val", "bg1")
    etree.SubElement(fld, qn("a:t")).text = str(number)
    pPr = p._p.find(qn("a:pPr"))
    p._p.insert(1 if pPr is not None else 0, fld)
    return tb


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def add_pic(slide, path, box, align="l", valign="t", name=None):
    """A picture scaled to fit the box (x, y, max_w, max_h), keeping its aspect ratio."""
    x, y, max_w, max_h = box
    pw, ph = image_size(path)
    scale = min(max_w / pw, max_h / ph)
    w, h = int(pw * scale), int(ph * scale)
    if align == "r":
        x = x + max_w - w
    elif align == "c":
        x = x + (max_w - w) // 2
    if valign == "m":
        y = y + (max_h - h) // 2
    pic = slide.shapes.add_picture(str(path), Emu(x), Emu(y), Emu(w), Emu(h))
    if name:
        pic.name = name
    return pic


def crop_image(path, tmpdir, box_frac, out_name, target_ratio=None):
    """Crop with PIL: box_frac = (left, top, right, bottom) as fractions; or crop to target_ratio (w/h) around the centre."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        if target_ratio:
            if w / h > target_ratio:
                nw = int(h * target_ratio)
                box = ((w - nw) // 2, 0, (w - nw) // 2 + nw, h)
            else:
                nh = int(w / target_ratio)
                box = (0, (h - nh) // 2, w, (h - nh) // 2 + nh)
        else:
            l, t, r, b = box_frac
            box = (int(w * l), int(h * t), int(w * r), int(h * b))
        out = Path(tmpdir) / out_name
        im.crop(box).save(out)
    return out


def add_table(slide, box, header, rows, col_w_in, *, size=13, first_col_bold=False, name=None):
    """A table that fits (x, y, w, max_h): row heights are estimated from the wrapped text; the font shrinks until it fits."""
    x, y, w, max_h = box
    ncols = len(header)
    col_w = [int(c * 914400) for c in col_w_in]
    scale = w / sum(col_w)
    col_w = [int(c * scale) for c in col_w]
    col_w[-1] += w - sum(col_w)
    heights, s = [], size
    sizes = []
    cand = float(size)
    while cand >= 8:
        sizes.append(cand)
        cand -= 0.5
    for s in sizes:
        heights = []
        for r_i, row in enumerate([header] + rows):
            lines = max(est_lines(runs_text(c), col_w[j] - 2 * MARG_H, s, bold=(r_i == 0 or (first_col_bold and j == 0)))
                        for j, c in enumerate(row))
            heights.append(int(lines * s * 1.22 * 12700 + 2 * MARG_V + 36000))
        if sum(heights) <= max_h:
            break
    if sum(heights) > max_h:
        warn(f"table on slide {CURRENT['n']} may overflow: {sum(heights) / 914400:.2f} in needed, {max_h / 914400:.2f} in available")
    print(f"slide {CURRENT['n']} table {name or ''}: {len(rows) + 1} rows at {s:g} pt, {sum(heights) / 914400:.2f} in of "
          f"{max_h / 914400:.2f} in available")
    gf = slide.shapes.add_table(len(rows) + 1, ncols, Emu(x), Emu(y), Emu(w), Emu(sum(heights)))
    if name:
        gf.name = name
    tbl = gf.table
    tbl.first_row = True
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(cw)
    for i, hgt in enumerate(heights):
        tbl.rows[i].height = Emu(hgt)
    for r_i, row in enumerate([header] + rows):
        for j, item in enumerate(row):
            cell = tbl.cell(r_i, j)
            cell.margin_left = cell.margin_right = Emu(MARG_H)
            cell.margin_top = cell.margin_bottom = Emu(MARG_V)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(BLUE if r_i == 0 else (ZEBRA if r_i % 2 == 0 else WHITE))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            runs = [item] if isinstance(item, str) else item
            for run in runs:
                txt, fmt = (run, {}) if isinstance(run, str) else run
                r = p.add_run()
                r.text = txt
                _format_run(r, fmt, s, WHITE if r_i == 0 else NAVY, r_i == 0 or (first_col_bold and j == 0), False)
    return gf, sum(heights), s


# ---- template handling -----------------------------------------------------------------------------------------------------
def prepare_template(refresh=False):
    if refresh or not TEMPLATE.exists():
        if not SAMPLE.exists():
            sys.exit(f"the professor's template is not where expected: {SAMPLE}")
        TEMPLATE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLE, TEMPLATE)      # copyfile: the copy is writable although the downloaded file is read-only
        print(f"template copied: {TEMPLATE}")
    return TEMPLATE


def delete_all_slides(prs):
    """Remove the sample's slides: the sldId entries of ppt/presentation.xml and the relationships behind them."""
    sldIdLst = prs.slides._sldIdLst
    n = 0
    for sldId in list(sldIdLst):
        rId = sldId.rId
        sldIdLst.remove(sldId)
        prs.part.drop_rel(rId)
        n += 1
    return n


def set_group_number(prs, text):
    """The fixed 'Group #' text box lives in slide layout 2 (userDrawn), not in the master (critic C6)."""
    done = 0
    for layout in prs.slide_layouts:
        for sh in layout.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip().startswith("Group #"):
                p = sh.text_frame.paragraphs[0]
                p.runs[0].text = text
                for r in p.runs[1:]:
                    r.text = ""
                done += 1
    if done != 1:
        warn(f"'Group #' text box found {done} times in the layouts (expected once)")


def template_logo(tmpdir):
    with zipfile.ZipFile(TEMPLATE) as z:
        out = Path(tmpdir) / "template_logo.png"
        out.write_bytes(z.read("ppt/media/image1.png"))
    return out


# ---- the slides ------------------------------------------------------------------------------------------------------------
def first_name(full):
    return full.split()[0]


def cap(s):
    """First letter upper-cased, the rest untouched (str.capitalize would lower-case 'NSE')."""
    return s[:1].upper() + s[1:] if s else s


def content_slide(prs, R, number, title, sub, elements, footer_items, presenter, notes):
    """One content slide on layout 2. elements: a function body_top -> list of element dicts."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    CURRENT["n"] = number
    # heading: 32 pt bold white; shrinks to 28/26/24 pt when the title needs two lines (the finding, word for word, on slide 8)
    hx, hy, hw, hh = HEAD
    hsize = 20                                                   # 20 pt on two lines (48 pt) is the fallback: the band is 70 pt high
    for cand in (32, 30, 28, 26, 24, 22, 20):                    # one line whenever possible; width margin 10%: PowerPoint/LibreOffice wrap earlier than the estimate (seen 2026-10-29)
        if est_lines(title, int(hw * 0.9), cand, bold=True) == 1:
            hsize = cand
            break
    add_text(slide, HEAD, [title], size=hsize, color=WHITE, bold=True, anchor="m", name="Heading", check=False)
    add_slidenum(slide, number)
    sub_lines = est_lines(sub, SUB_W, 16, bold=True)
    sub_h = max(SUB_H, int(sub_lines * 16 * 1.25 * 12700 + 2 * MARG_V + 20000))
    add_text(slide, (SUB_X, SUB_Y, SUB_W, sub_h), [sub], size=16, color=NAVY, bold=True, anchor="m", name="Sub-heading")
    body_top = SUB_Y + sub_h + 40000
    for el in elements(body_top):
        kind = el["kind"]
        if kind == "text":
            add_text(slide, el["box"], el["paras"], size=el.get("size", 14), color=el.get("color", NAVY), bold=el.get("bold", False),
                     italic=el.get("italic", False), align=el.get("align", "l"), anchor=el.get("anchor", "t"), name=el.get("name"),
                     line_spacing=el.get("line_spacing", 1.08))
        elif kind == "pic":
            add_pic(slide, el["path"], el["box"], align=el.get("align", "l"), valign=el.get("valign", "t"), name=el.get("name"))
        elif kind == "table":
            gf, h, s = add_table(slide, el["box"], el["header"], el["rows"], el["col_w"], size=el.get("size", 13),
                                 first_col_bold=el.get("first_col_bold", False), name=el.get("name"))
            if el.get("after"):
                for el2 in el["after"](el["box"][1] + h + el.get("after_gap", 60000)):
                    if el2["kind"] == "text":
                        add_text(slide, el2["box"], el2["paras"], size=el2.get("size", 12), color=el2.get("color", NAVY),
                                 name=el2.get("name"), line_spacing=el2.get("line_spacing", 1.08))
    # footer: the deck line (13.33 pt) and the honesty wordings of the slide (11 pt), bottom right above the band
    deck_line = f"{R.text('project_title').split(':')[0].strip()} · {R.text('group_number')} · presenter: {presenter}"
    paras = [{"runs": deck_line, "size": 13.33, "align": "r"}]
    if footer_items:
        paras.append({"runs": " · ".join(footer_items), "size": 11, "align": "r"})
    add_text(slide, FOOT, paras, color=GREY, anchor="b", name="Footer", line_spacing=1.0)
    set_notes(slide, notes)
    return slide


def set_notes(slide, notes):
    tf = slide.notes_slide.notes_text_frame
    tf.clear()
    for i, para in enumerate([p for p in notes.split("\n") if p.strip()]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = para


def cover_slide(prs, R, tmpdir, cover_mode, logo_path):
    """Slide 1 on layout 'Blank': full-bleed picture under the sample's veil (45 %) and band (55 %), title, members, question."""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    CURRENT["n"] = 1
    photo_ok = COVER_PHOTO.exists() and COVER_PHOTO_JSON.exists()
    use_photo = (cover_mode == "photo") or (cover_mode == "auto" and photo_ok)
    if use_photo and not photo_ok:
        warn(f"cover photo not found ({COVER_PHOTO}); the overview map is used instead")
        use_photo = False
    if use_photo:
        src, credit = COVER_PHOTO, json.loads(COVER_PHOTO_JSON.read_text(encoding="utf-8")).get("credit_line", "")
        which = f"cover_photo.jpg ({credit})"
    else:
        src = FIG / "overview_map.png"
        credit = "Figure: the 12 study basins (figures/overview_map.png, this project)".replace("12", R.text("n_basins"))
        which = "figures/overview_map.png"
    cropped = crop_image(src, tmpdir, None, "cover_16x9" + src.suffix, target_ratio=SLIDE_W / SLIDE_H)
    slide.shapes.add_picture(str(cropped), Emu(0), Emu(0), Emu(SLIDE_W), Emu(SLIDE_H)).name = "Cover picture"
    add_rect(slide, (0, 0, SLIDE_W, SLIDE_H), "F2F2F2", alpha=COVER_VEIL_ALPHA, name="Veil")
    add_rect(slide, COVER_BAND, WHITE, alpha=COVER_BAND_ALPHA, name="Band")
    slide.shapes.add_picture(str(logo_path), Emu(239351), Emu(176772), Emu(1200000), Emu(1200000)).name = "Logo"
    title_fmt = {"size": 24, "bold": True, "color": TITLE_BLUE, "font": "Microsoft YaHei", "spc": 300, "shadow": True}
    add_text(slide, (1700000, 330000, 10200000, 1780000), [{"runs": [(R.text("project_title"), title_fmt)], "align": "c",
                                                           "line_spacing": 1.2, "spc": 300}],
             size=24, anchor="m", name="Title")
    members = [{"runs": [(R.text("group_number"), {"size": 22, "bold": True, "color": BAND_BLUE, "spc": 300, "shadow": True,
                                                   "font": "Microsoft YaHei"})], "align": "c", "line_spacing": 1.2}]
    for i in range(1, int(R.val("n_members")) + 1):
        members.append({"runs": [(f"{R.text(f'member{i}_name')} · {R.text(f'member{i}_id')} · {R.text(f'member{i}_role')}",
                                  {"size": 20, "bold": True, "italic": True, "color": ORANGE, "font": "Calibri"})],
                        "align": "c", "line_spacing": 1.15})
    add_text(slide, (105321, 2250000, 11809307, 1760000), members, anchor="m", name="Members")
    question = ("What limits a one-to-three-day flood warning on a small river: the kind of model, or the information it is given?")
    add_text(slide, (548640, 4250000, 11094720, 1500000), [
        {"runs": [(question, {"size": 18, "bold": True, "color": NAVY})], "align": "c", "line_spacing": 1.15},
        {"runs": [(f"Final project · {R.text('course')} · {R.text('instructor')} · final presentation {R.text('final_presentation')}",
                   {"size": 14, "color": NAVY})], "align": "c", "space": 8},
        {"runs": [(f"Submission {R.text('submission_deadline')} · group leader: {R.text('group_leader')}",
                   {"size": 12, "color": GREY})], "align": "c", "space": 4},
    ], anchor="m", name="Question")
    add_text(slide, (548640, 6420000, 11094720, 300000), [{"runs": credit, "size": 10, "align": "r"}], color=GREY, anchor="b",
             name="Credit")
    notes = (f"We are {R.text('group_number')}. Our question: what limits a one-to-three-day flood warning on a small river, the kind "
             f"of model or the information it is given? {R.text('n_members')} of us speak; I take the questions.")
    set_notes(slide, notes)
    return slide, which, notes


def build(args):
    R = Registry(REGISTRY)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("group_number") is not None and str(cfg["group_number"]) not in R.text("group_number"):
        warn("config.json group_number differs from the registry: rebuild the registry (python -X utf8 code/final_report/numbers.py)")
    N, V = R.text, R.val
    prepare_template(args.refresh_template)
    prs = Presentation(str(TEMPLATE))
    if (prs.slide_width, prs.slide_height) != (SLIDE_W, SLIDE_H):
        sys.exit(f"unexpected slide size {prs.slide_width} x {prs.slide_height}")
    removed = delete_all_slides(prs)
    set_group_number(prs, N("group_number"))
    tmpdir = tempfile.mkdtemp(prefix="flood_deck_")
    logo = template_logo(tmpdir)

    # ---- facts every slide shares (registry) ----
    names = [N(f"member{i}_name") for i in range(1, int(V("n_members")) + 1)]
    A, MU, MO, HI = names[0], names[1], names[2], names[3]
    if "leader" not in N("member1_role").lower():
        warn("member 1 of config.json is not marked group leader")
    fw_aorc, fw_hind, fw_rt_short, fw_upper = N("fw_aorc"), N("fw_hindsight"), N("fw_realtime_short"), N("fw_upper")
    foot_aorc = f"AORC rain: {fw_aorc} · AORC {N('kf_level_wqa_onset_warned')}: {fw_hind}"   # "AORC 52%" (five seeds): named so that it is not read as another share
    foot_s4 = f"Stage IV {N('s4_stage4_share')}: {fw_rt_short}"
    foot_upper = f"perfect weather / perfect rain: {fw_upper}"
    team_sentence = (N("fw_team_checks") if int(V("team_checks_done") or 0) == 0
                     else f"The team's own checks recorded in team.json: {N('team_checks_done')}.")
    # review counts (critic 2, item 10): since the registry of 29 October 2026 20:27, review_records / review_rounds /
    # review_checks_listed exclude results/review_documents_*.json and document_review_passes counts those critic passes; a
    # registry without document_review_passes counts every results/review_*.json in the same keys, and the slides then say
    # "review records" without naming the critic passes in numbers. review_checks is read first in case numbers.py renames the key.
    rc_key = "review_checks" if R.has("review_checks") else "review_checks_listed"
    doc_passes = N("document_review_passes") if R.has("document_review_passes") else ""
    review_kind = "experiment review records" if doc_passes else "review records"
    critic_clause = f"; {doc_passes} critic passes over the report and the deck" if doc_passes else ""
    critic_spoken = f"a critic agent checked the documents in {doc_passes} passes" if doc_passes else "a critic agent verified the rest"
    # slide 14 links (critic 2, item 11): a bracketed registry text is a placeholder, not a link
    peek = lambda k: str(R.d[k].get("text", "")).strip() if R.has(k) else ""
    is_link = lambda k: bool(peek(k)) and not re.fullmatch(r"\[.*\]", peek(k))
    see_report = "see the report, section 9.2 and the datasets page"
    if is_link("repo_url"):
        links_runs = [("Repository: ", {"bold": True}),
                      f"{N('repo_url')} · data bundle: {N('data_bundle_url') if is_link('data_bundle_url') else see_report}"]
    else:
        links_runs = [("Repository and data links: ", {"bold": True}), see_report]
    # E4 optional block (present only when the results files exist; numbers only with a review record)
    e4a_present = R.has("e4a_file_present") and V("e4a_file_present") is True
    e4a_reviewed = e4a_present and R.has("e4a_review_file_present") and V("e4a_review_file_present") is True
    e4b_present = R.has("e4b_file_present") and V("e4b_file_present") is True
    e4b_reviewed = e4b_present and R.has("e4b_review_file_present") and V("e4b_review_file_present") is True
    e4e_present = R.has("e4e_file_present") and V("e4e_file_present") is True
    e4e_reviewed = e4e_present and R.has("e4e_review_file_present") and V("e4e_review_file_present") is True
    e4a_c1 = e4a_c2 = e4a_x = ""
    if e4a_reviewed:
        clauses = N("e4a_outcome").split("; ")
        e4a_c1 = next((c for c in clauses if "at or below" in c and "warns of" in c), "")
        e4a_c2 = next((c for c in clauses if "perfect-rain model" in c), "")
        m = re.search(r"warns of (\d+%)", e4a_c1)
        e4a_x = m.group(1) if m else ""
        if not (e4a_c1 and e4a_c2 and e4a_x):
            warn("E4a outcome text has a different shape than expected; the slides quote it verbatim where possible")
    e4a_line_slide9 = ""
    if e4a_reviewed:
        e4a_line_slide9 = (f"Second operating point on the stored forecasts (E4a, reviewed: {N('e4a_review_verdict')}) — the LSTM with "
                           f"AORC rain to midnight ({fw_aorc}): {e4a_c1}; {e4a_c2}.")
    elif e4a_present:
        e4a_line_slide9 = (f"{N('fw_preliminary_label')}: a second operating point, at the {N('go_max_false')}-false share the GO rule "
                           "allows, is being measured on the stored forecasts; it needs no new training and will be in the report "
                           "when its review record exists.")
    else:
        e4a_line_slide9 = (f"A second operating point, at the {N('go_max_false')}-false share the GO rule allows, is being measured "
                           "on the stored forecasts; it needs no new training and will be in the report when its review record exists.")
    # E4e (the river's state at the issue time, satellite-era split) and E4b (within-day rain, main split): numbers only with their
    # review records. The E4e ladder at the fixed level reads the e4e_* keys; its 30%-false operating points come from op_point().
    def review_dates(exp, record):
        """critic 3, item S6: '28 Oct 2026; five-seed round 30 Oct, 0 mismatches' — the first review's date from the record's
        "when" (date_review_<exp>; its "who" field carries the day the record was finished), then the five-seed round of
        30 October (rev_<record>_latest_*) when it is a later round than the first."""
        first = review_date(str(V(f"date_review_{exp}"))) if R.has(f"date_review_{exp}") else review_date(N(f"{exp}_review_who"))
        if R.has(f"rev_{record}_latest_round") and int(V(f"rev_{record}_latest_round")) > 1 and R.has(f"rev_{record}_latest_date"):
            later = review_date(str(V(f"rev_{record}_latest_date")))
            later = later.rsplit(" ", 1)[0] if later.endswith(first[-4:]) else later      # "30 Oct" when the year is the same
            first += f"; five-seed round {later}, {N(f'rev_{record}_latest_mismatches')} mismatches"
        return first

    sat_years = "–".join(str(s)[:4] for s in V("split_sat_test_dates"))                 # the years of the split's test dates
    e4e = {}                                                                             # (set, pod | far) -> (text, value) at the 30%-false level
    e4e_star = {}                                                                        # set -> "*" when its test false share is above the cap
    e4e_seeds = e4e_onsets = e4e_date = e4b_date = e4e_rule = e4b_rule = ""
    if e4e_reviewed:
        for s in ("wqa_s", "wqaH_s", "wqaWH_s", "wqaWHF_s", "wqaP_s"):
            for f_ in ("pod", "far"):
                e4e[(s, f_)] = op_point(R, "e4e", s, f_)
            e4e_star[s] = "*" if float(e4e[(s, "far")][1]) > float(V("go_max_false")) + 1e-9 else ""
        e4e_seeds = str(len(V("e4e_seeds")))                                             # trainings per new set
        e4e_onsets = f"{int(V('e4e_wqa_s_l1_onset_hits')) + int(V('e4e_wqa_s_l1_onset_misses')):,}"   # onsets of the split's test years
        e4e_date = review_dates("e4e", "flow_state")
        e4e_rule = "not met" if "is not met" in N("e4e_outcome_caveat") else "met"       # the success rule fixed before training
    if e4b_reviewed:
        e4b_date = review_dates("e4b", "within_day_inputs")
        e4b_rule = "not met" if "is not met" in N("e4b_outcome_caveat") else "met"
    e4_work = ""                                                                         # slide 14: the E4 training pass of 28–29 October
    n_e4 = None
    if R.has("e4b_n_trainings") and R.has("e4e_n_trainings"):
        n_e4 = int(V("e4b_n_trainings")) + int(V("e4e_n_trainings"))
        e4_minutes = (float(V("e4b_training_seconds")) + float(V("e4e_training_seconds"))) / 60
        e4_work = (f", plus {n_e4} of 28–29 October for E4 ({N('e4b_n_trainings')} E4b, {N('e4e_n_trainings')} E4e, about "
                   f"{e4_minutes:.0f} min; E4a trained nothing): {int(V('seed_extension_trainings')) + n_e4} on the Windows PC "
                   f"({', '.join(N('windows_pc').split(', ')[:2])}) in all")
    # seeds (critic 1, item 11): the stored sets quoted from before 28 October have seeds_per_set seeds; the E4b/E4e sets of
    # 28–29 October carry their own seed lists in the registry. new_seeds = "5" when both new experiments agree, else "5 / 5"-style.
    seed_counts = sorted({len(V(k)) for k in ("e4b_seeds", "e4e_seeds") if R.has(k) and isinstance(V(k), list)})
    new_seeds = " / ".join(str(c) for c in seed_counts)
    # five seeds (30 October 2026): the sets of scripts 15-18 (seeds 3-4 added on the Windows PC after the three-seed test scores
    # were known) and the E4 sets of 20-22 have five trainings; the stored sets of scripts 02-10 keep three (they saved no networks)
    n3 = N("seeds_per_set")                                                              # the stored sets of scripts 02-10
    five_list = V("e1b_seeds")                                                           # seeds of script 16 since 30 October
    n5 = str(len(five_list))
    if {len(V(k)) for k in ("e1_seeds", "e1b_seeds", "e4b_seeds", "e4e_seeds")} != {len(five_list)}:
        warn("the seed lists of scripts 15, 16, 20-22 differ in length: the five-seed label may be wrong")
    old_ids = {x.strip() for x in N("seeds_list").split(",")}
    ext_ids = [str(x) for x in five_list if str(x) not in old_ids]                       # the seeds added on 30 October
    ext_ids_text = "–".join([ext_ids[0], ext_ids[-1]]) if len(ext_ids) > 1 else "".join(ext_ids)
    ext_scripts = N("seed_extension_scripts")
    ext_date = N("seed_extension_date")
    three_list = N("stored_sets_three_seeds_list")
    three_lstm = ", ".join(x for x in re.split(r", | and ", three_list) if x.startswith("lstm_"))
    if not three_lstm:
        warn("stored_sets_three_seeds_list names no lstm_ set")
    seed_label = f"{n5} seeds for the sets of scripts {ext_scripts} and E4, {n3} for the stored sets {three_lstm}"
    seed_disclosure = (f"seeds {ext_ids_text} of scripts {ext_scripts} added on {ext_date}, after the {n3}-seed test scores "
                       "were known")
    seed_disclosure_short = f"seeds {ext_ids_text} added on {ext_date}, after the {n3}-seed test scores were known"
    seed_sentence = (f"seed averages of {n5} trainings for the sets of scripts {ext_scripts} and E4, {n3} for the stored sets "
                     f"{three_lstm} (the simple models: one fit each); {seed_disclosure}")
    # the five-seed review rounds of 30 October (one per experiment record: rev_<record>_latest_*)
    rev5 = sorted(k[len("rev_"):-len("_latest_mismatches")] for k in R.d
                  if k.startswith("rev_") and k.endswith("_latest_mismatches") and R.has(k))
    rev5_verdicts = [N(f"rev_{r}_latest_verdict") for r in rev5]
    if sum(int(V(f"rev_{r}_latest_mismatches")) for r in rev5) != int(V("review_five_seed_mismatches")):
        warn("rev_*_latest_mismatches do not add up to review_five_seed_mismatches")
    rev5_hold = "every verdict “numbers hold”" if all(v.startswith("numbers hold") for v in rev5_verdicts) else \
        "verdicts: " + "; ".join(sorted(set(rev5_verdicts)))
    rev5_line = (f"five-seed numbers recomputed by independent AI reviewer agents on {ext_date}: {len(rev5)} rounds, "
                 f"{N('review_five_seed_values_compared')} values, {N('review_five_seed_mismatches')} mismatches, {rev5_hold}")
    rev5_row = (f"five-seed numbers: {N('review_five_seed_values_compared')} values recomputed in {len(rev5)} rounds, "
                f"{N('review_five_seed_mismatches')} mismatches")
    rev5_short = (f"five-seed numbers recomputed by independent reviewers: {N('review_five_seed_values_compared')} values, "
                  f"{N('review_five_seed_mismatches')} mismatches")
    # stored forecast sets and networks (critic 1, items 6 and 7): the 8 October data-flow counts, dated, and the live file counts
    # the registry made on its last run, dated by the registry's file time
    registry_date = file_date(REGISTRY)
    stored_line = (f"{N('stored_prediction_sets')} stored forecast sets and {N('stored_saved_networks')} saved networks as of "
                   f"{N('date_data_record_written')} (data_flow.json)")
    if R.has("count_preds_npz") and R.has("count_models_pt"):
        new_nets = int(V("count_models_pt")) - int(V("stored_saved_networks"))
        n_ext = int(V("seed_extension_trainings"))
        why = (f", the {new_nets} new ones from the Windows PC (E4b, E4e, seeds {ext_ids_text})"
               if n_e4 is not None and new_nets == n_e4 + n_ext else
               (f", the {new_nets} new ones trained for E4b and E4e" if n_e4 is not None and new_nets == n_e4 else ""))
        stored_line += (f"; on {registry_date}, when the registry last counted them, {N('count_preds_npz')} forecast files and "
                        f"{N('count_models_pt')} saved networks{why}")
    best_model_name = {"base_linreg_wq": "ridge regression", "base_rf_wq": "random forest", "base_persistence": "tomorrow = today"}
    products = ["camels_us", "aorc", "gefs", "era5", "merra2", "imerg", "stage4", "elevation_tiles", "nlcd", "hurdat2", "usgs_iv"]
    short = {"camels_us": "CAMELS-US", "aorc": "AORC", "gefs": "GEFS", "era5": "ERA5", "merra2": "MERRA-2", "imerg": "IMERG",
             "stage4": "Stage IV", "elevation_tiles": "elevation tiles", "nlcd": "NLCD land cover", "hurdat2": "HURDAT2",
             "usgs_iv": "USGS 15-minute flow"}
    core, extra, checks = [], [], []
    for pid in products:
        role = N(f"ds_{pid}_role").lower()
        if role.startswith("the data every model"):
            core.append(short[pid])
        elif "extra input" in role or "tested" in role:
            extra.append(short[pid])
        elif pid == "usgs_iv" and e4e_reviewed and R.has("hf_file_present") and V("hf_file_present") is True:
            # critic 3, item S2: results/data_flow.json (8 October) files the USGS 15-minute record as a check; its hourly
            # reduction (data/hourly_flow, fetched hf_fetch_dates) is the input of E4e, so it counts as an extra input, as in the
            # report (4.1: six inputs of single experiments, four checks and maps)
            extra.append("the hourly USGS gauge record of E4e")
        else:
            checks.append(short[pid])
    if len(core) + len(extra) + len(checks) != int(V("n_products")):
        warn("product roles do not add up to n_products")
    usgs_moved = "the hourly USGS gauge record of E4e" in extra
    # the data-flow picture of slide 3 is the drawing of date_data_record_written: it still shows the USGS record as a check
    flow_fig_note = (f" (the colours above, drawn {N('date_data_record_written')}: the USGS record, a check then, became the hourly "
                     f"input of E4e on {N('hf_fetch_dates')})" if usgs_moved else " (the colours above)")

    slides = []      # (number, title, speaker, seconds, notes)

    # ---- slide 1: cover ----
    R.slide = 1
    _, cover_which, notes1 = cover_slide(prs, R, tmpdir, args.cover, logo)
    slides.append((1, "Cover", A, 15, notes1))

    # ---- slide 2: question, method, answer ----
    R.slide = 2
    # the AORC share carries both its wordings in the spoken notes too (critic 2, item 4); words were cut elsewhere to keep the pace
    notes2 = (f"{N('n_basins')} rivers. One LSTM forecasts the flow {LEADS} and warns when it crosses the flood level, the training "
              f"years' {N('high_flow_percentile')} percentile. GO, fixed beforehand: {N('target_go_warned')} of onsets warned, "
              f"{N('target_go_false')} false. {cap(N('fw_finding_short'))}: {N('kf_level_wq_onset_warned')} to "
              f"{N('kf_level_wqa_onset_warned')} of onsets warned a day ahead with AORC rain to midnight, {fw_aorc}; {fw_hind}. "
              f"{first_name(MU)} takes the data.")
    content_slide(prs, R, 2, "The question, the method in one line, the answer",
                  f"{N('n_basins')} rivers, one small LSTM, one verdict rule fixed in advance",
                  lambda top: [
                      {"kind": "text", "box": (BODY_X, top, 6900000, BODY_BOTTOM - top), "size": 14, "paras": [
                          {"runs": [("Data: ", {"bold": True}), f"CAMELS-US — {N('ds_camels_us_one_line')}; the {N('n_basins')} rivers lie "
                                    f"in the south-eastern United States ({N('water_years')} water years)"], "bullet": True},
                          {"runs": [("Model: ", {"bold": True}), f"one LSTM for all {N('n_basins')} rivers forecasts the flow {LEADS} and "
                                    f"warns when the forecast crosses the flood level = the {N('high_flow_percentile')} percentile of the "
                                    f"training years ({N('high_flow_level_range_mm_day')})"], "bullet": True, "space": 6},
                          {"runs": [("Fair test: ", {"bold": True}), f"every choice on the validation years {N('split_val_years_text')}; "
                                    f"the test years {N('split_test_years_text')} scored once per experiment"], "bullet": True, "space": 6},
                          {"runs": [("Verdict rule fixed in advance: ", {"bold": True}), f"GO = {N('target_go_warned')} of the flood onsets "
                                    f"warned and {N('target_go_false')} of the warnings false ({N('go_rule')})"], "bullet": True, "space": 6},
                          {"runs": [("Answer: ", {"bold": True}), (f"“{N('fw_finding')}”", {"italic": True}),
                                    f" — {N('kf_level_wq_onset_warned')} → {N('kf_level_wqa_onset_warned')} of the {N('onsets_l1')} onsets "
                                    f"warned a day ahead with the same network and more information (AORC rain to midnight, {fw_aorc}; "
                                    f"{fw_hind})"], "bullet": True, "space": 6},
                      ]},
                      {"kind": "pic", "path": FIG / "overview_map.png", "box": (7600000, top, BODY_RIGHT - 7600000, BODY_BOTTOM - top),
                       "align": "r", "name": "Figure overview map"},
                  ], [foot_aorc], A, notes2)
    slides.append((2, "The question, the method in one line, the answer", A, 35, notes2))

    # ---- slide 3: methodology I, the data ----
    R.slide = 3
    data_flow_crop = crop_image(FIG / "data_flow_talk.png", tmpdir, (0, 0, 1, DATA_CROP), "data_flow_top.png")
    imagery_first = N("imagery_note").split(". ")[0] + "."
    notes3 = (f"{N('n_products')} public products, one pipeline. Every model reads CAMELS-US: the river flow measured daily by the "
              f"US Geological Survey, the Daymet weather averaged over each basin, and {N('n_properties')} basin properties. "
              f"{len(extra)} more products are extra inputs of some experiments: {', '.join(extra)}. {len(checks)} are only checks "
              f"and maps. All are public, with no account and no payment. The project read {N('gb_read')} and keeps {N('gb_kept')}. "
              f"Each of the {N('n_basins')} basins has {N('days_per_basin')} days with {N('missing_values')} values missing. "
              f"The split is by target day: {N('split_train_days')} days to train, {N('split_val_days')} to validate, "
              f"{N('split_test_days')} to test, and the inputs are scaled with the training years only. The flow becomes "
              f"millimetres a day, the unit of rain. Gridded rain is averaged over each basin, every cell weighted by its overlap, "
              f"and hourly rain is summed over the local wall-clock day. The pipeline measured where each product's day ends: the Daymet "
              f"rain table at {N('day_end_daymet_rounded')}, NLDAS at {N('day_end_nldas_rounded')}, so the afternoon rain of a "
              f"Daymet day is filed under the next day. No optical satellite scene was classified, so capture date and sun angle "
              f"do not apply; the satellite products are used ready-made. Now, where.")
    fig_w = 9400000
    fig_h = int(fig_w * (image_size(data_flow_crop)[1] / image_size(data_flow_crop)[0]))
    content_slide(prs, R, 3, "Methodology I: the data — what, how obtained, the flow, how computed",
                  f"{N('n_products')} public products, one pipeline: {N('gb_read')} read, {N('gb_kept')} kept, no account, no payment, "
                  "no classified imagery",
                  lambda top: [
                      {"kind": "pic", "path": data_flow_crop, "box": (BODY_X, top, BODY_W, fig_h), "align": "c", "name": "Figure data flow"},
                      {"kind": "text", "box": (BODY_X, top + fig_h + 40000, BODY_W, BODY_BOTTOM - top - fig_h - 40000), "size": 11, "paras": [
                          {"runs": f"Every model reads {core[0]} ({N('ds_camels_us_one_line')}); {len(extra)} products are extra inputs of "
                                   f"some experiments, {len(checks)} only checks and maps{flow_fig_note}", "bullet": True},
                          {"runs": f"{N('n_basins')} basins × {N('days_per_basin')} days, {N('missing_values')} missing values; "
                                   f"{N('weather_columns')} weather columns + {N('flow_columns')} flow + {N('n_properties')} basin "
                                   f"properties; split by target day: train {N('split_train_days')} days, validation {N('split_val_days')}, "
                                   f"test {N('split_test_days')} ({N('split_test_years_text')}); inputs scaled with the training years only",
                           "bullet": True, "space": 3},
                          {"runs": f"How computed: flow in mm/day ({N('formula_flow_mm_per_day')}); gridded rain averaged over each "
                                   "outline, every cell weighted by its overlap; hourly rain summed over the local wall-clock day",
                           "bullet": True, "space": 3},
                          {"runs": f"Day ends measured against hourly rain: Daymet rain table {N('day_end_daymet_rounded')}, NLDAS "
                                   f"{N('day_end_nldas_rounded')} (measured median {N('day_end_nldas')}), river flow {N('day_end_river_flow')}: "
                                   "the afternoon rain of a Daymet day is filed under the next day", "bullet": True, "space": 3},
                          {"runs": f"Imagery: {imagery_first} Satellite-derived products were used ready-made.", "bullet": True, "space": 3},
                      ]},
                  ], [], MU, notes3)
    slides.append((3, "Methodology I: the data", MU, 95, notes3))

    # ---- slide 4: methodology II, where ----
    R.slide = 4
    gis_done = V("gis_layers_exported") is True
    gis_sentence = (f"The outlines, elevation and land cover of all {N('basins_chosen')} basins are delivered as GIS layers."
                    if gis_done else "The GIS-layer export is still open.")
    notes4 = (f"The {N('n_basins')} basins were chosen by rule before any model ran. From the {N('basins_all_camels')} CAMELS basins, "
              f"{N('basins_candidates')} are south-eastern candidates: little snow, humid, {N('basins_rule_area_km2')}. "
              f"{N('basins_complete')} have a complete record, and {N('basins_chosen')} were taken, {N('basins_rule_per_landscape')} "
              f"in each of {N('n_landscapes')} landscapes, evenly spaced in area. The code computed each basin's area, elevation, slope "
              f"and land cover from public maps and compared them with the CAMELS table; "
              f"they agree except the mean elevation of the {N('gis_elev_basins_over_25m')} mountain basins ({N('gis_elev_basins_over_25m_names')}: {N('gis_elev_over_25m_range')} against CAMELS), cause not found. {gis_sentence} {first_name(MO)} now explains the model and the test.")
    content_slide(prs, R, 4, f"Methodology II: where — {N('n_basins')} basins chosen by rule, and the GIS layers",
                  "Chosen before any model ran; every GIS statistic checked against the CAMELS table",
                  lambda top: [
                      {"kind": "text", "box": (BODY_X, top, 7050000, BODY_BOTTOM - top), "size": 13, "paras": [
                          {"runs": [(N("basins_funnel").replace("->", "→"), {"bold": True}), f": all CAMELS basins → candidates "
                                    f"({N('basins_rule_huc')}, snow under {N('basins_rule_max_snow_share')} of precipitation, aridity under "
                                    f"{N('basins_rule_max_aridity')}, area {N('basins_rule_area_km2')}) → complete record → "
                                    f"{N('basins_rule_per_landscape')} per landscape, evenly spaced in area, no model score"], "bullet": True},
                          {"runs": f"{N('n_landscapes')} landscapes: {N('landscape_names')}; areas {N('basin_area_range_km2')}",
                           "bullet": True, "space": 4},
                          {"runs": f"Layers: CAMELS outlines ({N('gis_outline_crs').split(' (')[0]}); elevation tiles at {N('gis_elevation_resolution').split(';')[0]} "
                                   f"in SRTM format, a composite tile set ({N('gis_elevation_tiles')} tiles); Annual NLCD {N('gis_land_cover_year')} "
                                   f"land cover at {N('gis_land_cover_resolution')}", "bullet": True, "space": 4},
                          {"runs": f"Our statistics against the CAMELS table: area within {N('gis_area_vs_gages2_median_pct')} (median), "
                                   f"elevation rank correlation {N('gis_elev_rank_correlation')}, slope rank correlation "
                                   f"{N('gis_slope_rank_correlation')}; largest elevation difference {N('gis_elev_largest_diff_m')} at "
                                   f"{N('gis_elev_largest_diff_basin')}, cause not found", "bullet": True, "space": 4},
                          {"runs": [("Mid-term target ", {}), (f"“{N('target_gis_layers')}”", {"italic": True}),
                                    f": maps and statistics exist for {N('basins_chosen')} of {N('basins_chosen')}; the export as GIS "
                                    f"layers: {N('gis_layers_exported')}" + ("" if gis_done else " (not done)")], "bullet": True, "space": 4},
                      ]},
                      {"kind": "pic", "path": FIG / "basin_maps.png", "box": (7300000, top, BODY_RIGHT - 7300000, BODY_BOTTOM - top),
                       "align": "r", "name": "Figure basin maps"},
                  ], [], MU, notes4)
    slides.append((4, "Methodology II: where", MU, 55, notes4))

    # ---- slide 5: methodology III, model, fair test, scoring ----
    R.slide = 5
    notes5 = (f"Each period has one job. The training years fit the models and set the flood level; the validation years choose "
              f"every setting and where training stops; the test years are only scored, once per experiment. The model is one LSTM "
              f"for all {N('n_basins')} rivers: {N('memory_cells')} memory cells, {N('weights_lstm_wq')} weights, trained "
              f"{n3} or {n5} times from different random starts and averaged. Beside it run three simple models: tomorrow "
              f"equals today, ridge regression and a random forest. The score for the flow is the NSE. For warnings the honest test "
              f"is the onset: a high-flow day forecast while the river was still below the level. GO needs {N('target_go_warned')} of "
              f"the onsets warned with {N('target_go_false')} false; these rules were fixed before any experiment.")
    content_slide(prs, R, 5, "Methodology III: model, fair test, scoring rules",
                  "Every choice on the validation years; the test years scored once; verdicts fixed in advance",
                  lambda top: [
                      {"kind": "table", "box": (BODY_X, top, BODY_W, 1500000), "col_w": [2.0, 3.6, 6.5], "size": 13, "first_col_bold": True,
                       "name": "Table periods",
                       "header": ["Period", "Years (water years)", "Its one job"],
                       "rows": [["Training", f"{N('split_train_years_text')} ({N('split_train_water_years')})",
                                 "fit the models and set the high-flow levels"],
                                ["Validation", f"{N('split_val_years_text')} ({N('split_val_water_years')})",
                                 "choose every setting, the warning levels and where training stops"],
                                ["Test", f"{N('split_test_years_text')} ({N('split_test_water_years')})",
                                 "only scored, once per experiment"]],
                       "after": lambda y: [
                           {"kind": "text", "box": (BODY_X, y, BODY_W, BODY_BOTTOM - y), "size": 12.5, "paras": [
                               {"runs": [("One LSTM for all rivers: ", {"bold": True}), f"{N('memory_cells')} memory cells, "
                                         f"{N('weights_lstm_wq')} weights, {N('inputs_per_day')} inputs a day, learning rate {N('config_lr')}, "
                                         f"dropout {N('config_dropout')}, batches of {N('sequences_per_update')} sequences of {N('sequence_days')} "
                                         f"days, at most {N('config_updates')} updates with a validation check every {N('config_eval_every')} "
                                         f"(patience {N('config_patience')}); {n3} trainings from different random starts ({n5} for the sets of "
                                         f"scripts {ext_scripts} and E4), averaged"], "bullet": True},
                               {"runs": [("Simple models beside it: ", {"bold": True}), "‘tomorrow = today’, ridge regression (alpha "
                                         f"{N('ridge_alpha')}), random forest ({N('rf_n_trees')} trees), each fitted per basin and lead on the "
                                         "training years"], "bullet": True, "space": 3},
                               {"runs": [("Scores: ", {"bold": True}), f"NSE = {N('words_nse').split('. Scored')[0]}; "
                                         "onset = a high-flow day forecast while the river was still at or below the level; share of the "
                                         "warnings false; CSI"], "bullet": True, "space": 3},
                               {"runs": [("Verdicts: ", {"bold": True}), f"GO when {N('go_rule')}; CAUTION when {N('caution_rule')}; "
                                         f"{N('not_yet_rule')} NOT YET — on the test years at the warning level chosen on the validation years"],
                                "bullet": True, "space": 3},
                               {"runs": [("Leak test on made-up rivers with a known answer: ", {"bold": True}),
                                         f"{N('fair_test_n_checks')} checks {N('fair_test_passed')} (rule: {N('fair_test_rule')})"],
                                "bullet": True, "space": 3},
                           ]}]},
                  ], [], MO, notes5)
    slides.append((5, "Methodology III: model, fair test, scoring rules", MO, 55, notes5))

    # ---- slide 6: implementation ----
    R.slide = 6
    notes6 = (f"The whole project is code. One shared engine does the split, the scoring and the training, and a test on made-up "
              f"rivers with a known answer proves no leak from later days. Numbered scripts each answer one question and "
              f"write a results file; every document, these slides included, reads its numbers from those files. "
              f"{N('work_trainings')} trainings took {N('work_hours')} on a laptop and a Windows PC, no GPU. To be plain: {N('fw_who_did_what')} "
              f"Independent AI reviewer agents checked the shared engine and every experiment script. {team_sentence}")
    content_slide(prs, R, 6, "Implementation: pipeline, training, checks",
                  "Everything is code; every number on these slides is read from a results file; the code was reviewed",
                  lambda top: [
                      {"kind": "text", "box": (BODY_X, top, 5800000, BODY_BOTTOM - top), "size": 12, "paras": [
                          {"runs": f"Shared engine code/flood.py (split, scoring, trainer) with its test test_flood.py; "
                                   f"{N('count_code_py_files')} top-level Python scripts in code/ (counted on {registry_date}); numbered "
                                   "scripts answer one question each and write results/*.json and figures/*.png", "bullet": True},
                          {"runs": "Every document, this deck included, is built from results/*.json by a builder script: never "
                                   "hand-type a number", "bullet": True, "space": 3},
                          {"runs": f"{N('work_trainings')} trainings in {N('work_hours')} ({N('run_lstm_single_trainings')} one-river tuning runs; "
                                   f"{N('seed_extension_trainings')} on the Windows PC), CPU only: {N('mac_machine')}",
                           "bullet": True, "space": 3},
                          {"runs": stored_line, "bullet": True, "space": 3},
                          {"runs": f"Engine review: {N('engine_review_n_reviewers')} independent AI reviewers, "
                                   f"{N('engine_review_checks_passed')} checks passed, {N('engine_review_major')} major findings, "
                                   f"{N('engine_review_all_fixed')} before the experiments; {N('review_records')} {review_kind}, "
                                   f"{N('review_rounds')} review rounds with {N(rc_key)} checks listed: "
                                   f"{N('review_none_failed')}{critic_clause}", "bullet": True, "space": 3},
                          {"runs": [(f"{N('fw_who_did_what')} ", {"bold": True}), "Independent AI reviewer agents checked the shared "
                                    f"engine and every experiment script. {team_sentence}"], "bullet": True, "space": 3},
                      ]},
                      {"kind": "pic", "path": FIG / "training_curves.png", "box": (6400000, top, BODY_RIGHT - 6400000, BODY_BOTTOM - top),
                       "align": "r", "name": "Figure learning curves"},
                  ], [], MO, notes6)
    slides.append((6, "Implementation: pipeline, training, checks", MO, 45, notes6))

    # ---- slide 7: results I, the number to beat ----
    R.slide = 7
    bm = {k: best_model_name.get(N(f"best_simple_without_weather_forecast_l{k}_model"), N(f"best_simple_without_weather_forecast_l{k}_model"))
          for k in (1, 2, 3)}
    notes7 = (f"The best simple model, chosen on the validation years, sets the number to beat: a median NSE of "
              f"{N('best_simple_without_weather_forecast_l1_nse')} one day ahead, {N('best_simple_without_weather_forecast_l2_nse')} and "
              f"{N('best_simple_without_weather_forecast_l3_nse')} at two and three days. One LSTM for all {N('n_basins')} rivers scores "
              f"{N('kf_level_wq_nse')}, {N('kf_l2_wq_nse')} and {N('kf_l3_wq_nse')}: {N('lstm_vs_best_simple_l1_reading')}, so "
              f"milestone two of our plan is {N('milestone_m2_status')}. A Transformer is no better and takes about "
              f"{N('cmp_ratio_seconds_per_model')} as long to train; that comparison used its own {N('cmp_seeds')}-seed LSTM with "
              f"shorter sequences. And all four kinds of model miss the same {N('same_misses_count')} of the {N('onsets_l1')} onsets. "
              f"So the model is not the limit. {first_name(HI)} shows what is.")
    content_slide(prs, R, 7, "Results I: the number to beat",
                  f"With the river gauge, the LSTM against the best simple model: {N('lstm_vs_best_simple_l1_reading')} "
                  f"(milestone 2: {N('milestone_m2_status')})",
                  lambda top: [
                      {"kind": "table", "box": (BODY_X, top, BODY_W, 2600000), "col_w": [2.6, 3.2, 2.5, 3.8], "size": 13, "first_col_bold": True,
                       "name": "Table number to beat",
                       "header": ["Test years, 12 basins".replace("12", N("n_basins")), "Best simple model (chosen on validation)",
                                  f"LSTM ({N('lstm_wq_l1_n_seeds')} seeds)",
                                  f"Transformer* (own experiment, {N('cmp_seeds')} seeds; its LSTM twin in brackets)"],
                       "rows": [["Median NSE, 1 day", f"{N('best_simple_without_weather_forecast_l1_nse')} ({bm[1]})", N("kf_level_wq_nse"),
                                 f"{N('cmp_transformer_l1_nse')} ({N('cmp_lstm_l1_nse')})"],
                                ["Median NSE, 2 days", f"{N('best_simple_without_weather_forecast_l2_nse')} ({bm[2]})", N("kf_l2_wq_nse"),
                                 f"{N('cmp_transformer_l2_nse')} ({N('cmp_lstm_l2_nse')})"],
                                ["Median NSE, 3 days", f"{N('best_simple_without_weather_forecast_l3_nse')} ({bm[3]})", N("kf_l3_wq_nse"),
                                 f"{N('cmp_transformer_l3_nse')} ({N('cmp_lstm_l3_nse')})"],
                                ["Onsets warned, 1 day", f"{N('best_simple_without_weather_forecast_l1_onset_pod')} ({bm[1]}); "
                                 f"{N('kf_level_wq_base_rf_wq_onset_warned')} (random forest)", N("kf_level_wq_onset_warned"),
                                 f"{N('cmp_transformer_l1_onset_pod')} ({N('cmp_lstm_l1_onset_pod')})"],
                                ["Warnings false, 1 day", f"{N('best_simple_without_weather_forecast_l1_onset_far')} ({bm[1]})",
                                 N("kf_level_wq_onset_false"), f"{N('cmp_transformer_l1_onset_far')} ({N('cmp_lstm_l1_onset_far')})"]],
                       "after": lambda y: [
                           {"kind": "text", "box": (BODY_X, y, BODY_W, BODY_BOTTOM - y), "size": 12, "paras": [
                               {"runs": f"LSTM minus best simple model, median NSE 1 day ahead: {N('unc_nse_diff_lstm_wq_minus_best_simple_l1')} "
                                        f"(resampling range {N('unc_nse_diff_lstm_wq_minus_best_simple_l1_interval')}); the LSTM is higher in "
                                        f"{N('lstm_wq_vs_base_linreg_wq_l1_n_basins_lstm_higher')}, {N('lstm_wq_vs_base_rf_wq_l2_n_basins_lstm_higher')} "
                                        f"and {N('lstm_wq_vs_base_rf_wq_l3_n_basins_lstm_higher')} basins against the best simple model at 1, 2 and "
                                        f"3 days: {N('lstm_vs_best_simple_l1_reading')}", "bullet": True},
                               {"runs": f"* The Transformer comparison is its own experiment with its own {N('cmp_seeds')}-seed LSTM twin and "
                                        f"sequences of {N('cmp_sequence_days')} days (the headline LSTM: {N('sequence_days')}); Transformer minus "
                                        f"twin {N('cmp_transformer_minus_lstm_l1')} / {N('cmp_transformer_minus_lstm_l2')} / "
                                        f"{N('cmp_transformer_minus_lstm_l3')} NSE at 1 / 2 / 3 days at {N('cmp_ratio_seconds_per_model')} "
                                        "the training time per model", "bullet": True, "space": 3},
                               {"runs": f"All four kinds of model miss the same {N('same_misses_count')} of the {N('onsets_l1')} onsets "
                                        f"({N('same_misses_share')}); resampling range of the LSTM's {N('kf_level_wq_onset_warned')}: "
                                        f"{N('unc_onsets_warned_lstm_wq_l1_interval')} (test years and basins drawn again)", "bullet": True,
                                "space": 3},
                           ]}]},
                  ], [], MO, notes7)
    slides.append((7, "Results I: the number to beat", MO, 50, notes7))

    # ---- slide 8: the key figure ----
    R.slide = 8
    notes8 = (f"Our finding in one picture. Each row gives the LSTM more information; left, the share of onsets warned; "
              f"right, the median NSE. On the same inputs, the four kinds of model lie within "
              f"{N('kf_spread_model_kinds_nse')} NSE and {N('kf_spread_model_kinds_onset_points')} of onsets. "
              f"Every step of information is larger: the smallest is {N('kf_smallest_step_onset_points')}. The hatched row "
              f"is a perfect weather forecast, {fw_upper}. The AORC row is {fw_aorc}, and the diamonds put the real-time radar "
              f"product in its place: {fw_rt_short}. The green band is what GO asks for; nothing realistic reaches it.")
    cap8 = (f"Same inputs: four kinds of model within {N('kf_spread_model_kinds_nse')} NSE and {N('kf_spread_model_kinds_onset_points')} "
            f"of onsets warned; the smallest gain from more information is {N('kf_smallest_step_onset_points')} / "
            f"{N('kf_smallest_step_nse')} NSE. Diamonds: Stage IV in AORC's place ({fw_rt_short}). Test years "
            f"{N('split_test_years_text')}, {N('onsets_l1')} onsets 1 day ahead; every mark a seed average ({seed_label}); "
            "Transformer: its own experiment (slide 7).")
    content_slide(prs, R, 8, N("fw_finding"), N("fw_support"),
                  lambda top: [
                      {"kind": "pic", "path": FIG / "key_figure_talk.png", "box": (BODY_X, top, BODY_W, BODY_BOTTOM - top - 580000),
                       "align": "c", "name": "Key figure"},
                      {"kind": "text", "box": (BODY_X, BODY_BOTTOM - 560000, BODY_W, 560000), "size": 10.5, "color": GREY, "anchor": "b",
                       "paras": [cap8], "name": "Caption"},
                  ], [foot_aorc, foot_s4, foot_upper], HI, notes8)
    slides.append((8, "Results II: the key figure", HI, 50, notes8))

    # ---- slide 9: the ladder, 1 day ahead ----
    R.slide = 9
    e4a_note9 = ""
    if e4a_reviewed and e4a_x:
        e4a_note9 = f" At the GO rule's {N('go_max_false')} false, rain to midnight warns of {e4a_x}."
    # the perfect-forecast share carries UPPER in the spoken notes too (critic 2, item 4); words were cut elsewhere to keep the pace
    notes9 = (f"Same network, more information, one day ahead. Rain table alone: {N('kf_level_wq_onset_warned')}. Rain to the "
              f"evening: {N('kf_level_wqn_onset_warned')}. Rain to midnight from AORC, {fw_aorc}: {N('kf_level_wqa_onset_warned')}, "
              f"{fw_hind}. Real-time radar in its place: {N('s4_stage4_share')}, {N('s4_stage4_no_last_hour_share')} without the last "
              f"hour: {fw_rt_short}. A perfect weather forecast gives {N('kf_level_wqf_onset_warned')}, {fw_upper}. Simple models gain "
              f"as much: {N('fw_info_not_net')}.{e4a_note9}\n"
              f"If asked what the Stage IV check is: {N('fw_realtime')}.")
    # sub-heading: the AORC and perfect-forecast shares carry their fixed wordings in the same unit (critic 1, item 22); the
    # Stage IV share stands in the table row and the footer that carry REALTIME_WORDING
    content_slide(prs, R, 9, "Results III: the ladder of information, 1 day ahead",
                  f"Test years {N('split_test_years_text')}, {N('onsets_l1')} onsets, {N('lstm_wqa_l1_n_seeds')} seeds (AORC, Stage IV) or "
                  f"{N('lstm_wq_l1_n_seeds')}: "
                  f"{N('kf_level_wq_onset_warned')} → {N('kf_level_wqn_onset_warned')} → {N('kf_level_wqa_onset_warned')} with AORC rain "
                  f"to midnight ({fw_aorc}; {fw_hind}) → {N('kf_level_wqf_onset_warned')} with a perfect weather forecast ({fw_upper}); "
                  "false share near half",
                  lambda top: [
                      {"kind": "table", "box": (BODY_X, top, BODY_W, 2900000), "col_w": [5.3, 2.4, 1.9, 2.5], "size": 12.5,
                       "first_col_bold": False, "name": "Table ladder",
                       "header": ["Inputs of the LSTM with the river gauge", "Onsets warned", "Warnings false", "Median NSE"],
                       "rows": [["rain table alone (Daymet)", N("kf_level_wq_onset_warned"), N("kf_level_wq_onset_false"), N("kf_level_wq_nse")],
                                [f"+ rain up to the evening of the issue day (NLDAS, {N('day_end_nldas_rounded')})",
                                 N("kf_level_wqn_onset_warned"), N("kf_level_wqn_onset_false"),
                                 f"{N('kf_level_wqn_nse')} ({N('kf_step_rain_up_to_evening_nse_reading')}: higher in "
                                 f"{N('kf_step_rain_up_to_evening_n_basins_better')} basins)"],
                                [[f"+ rain up to midnight (AORC, ", (fw_aorc, {"italic": True}), ")"],
                                 [N("kf_level_wqa_onset_warned"), (f" ({fw_hind})", {"italic": True})],
                                 N("kf_level_wqa_onset_false"), N("kf_level_wqa_nse")],
                                [["Stage IV radar in AORC's place on the issue day, whole day / without its last hour — ",
                                  (fw_rt_short, {"italic": True})],
                                 f"{N('s4_stage4_share')} / {N('s4_stage4_no_last_hour_share')} ({N('s4_stage4_onsets_warned')} / "
                                 f"{N('s4_stage4_no_last_hour_onsets_warned')} of {N('s4_onsets_tested')})",
                                 f"{N('s4_issue_days_stage4_far')} on the issue days", "—"],
                                [["+ perfect weather forecast — ", (fw_upper, {"italic": True})], N("kf_level_wqf_onset_warned"),
                                 N("kf_level_wqf_onset_false"), N("kf_level_wqf_nse")]],
                       "after": lambda y: [
                           {"kind": "text", "box": (BODY_X, y, BODY_W, BODY_BOTTOM - y), "size": 11, "paras": [
                               {"runs": f"The hours after {N('day_end_daymet_rounded')} are worth {N('e1b_hours_after_09_points')}; the same AORC "
                                        f"product over the rain table's own window adds nothing ({N('e1b_onset_warned_control')}, the control)",
                                "bullet": True},
                               {"runs": f"Ridge regression and the random forest given the same AORC input ({fw_aorc}) reach "
                                        f"{N('kf_level_wqa_base_linreg_wqa_onset_warned')} and {N('kf_level_wqa_base_rf_wqa_onset_warned')} "
                                        f"of the onsets: {N('fw_info_not_net')}", "bullet": True, "space": 2},
                               {"runs": e4a_line_slide9, "bullet": True, "space": 2},
                           ]}]},
                  ], [foot_aorc, foot_s4, foot_upper] + ([] if e4a_reviewed or not e4a_present else [N("fw_preliminary_label") + ": E4a"]),
                  HI, notes9)
    slides.append((9, "Results III: the ladder of information, 1 day ahead", HI, 40, notes9))

    # ---- slide 10: 2 and 3 days ahead ----
    R.slide = 10
    # the perfect-rain shares carry UPPER, the AORC share of the "If asked" line its two wordings (critic 2, item 4)
    notes10 = (f"Two and three days ahead the rain table alone warns of {N('kf_l2_wq_onset_warned')} and {N('kf_l3_wq_onset_warned')} "
               f"of onsets, rain to midnight of {N('kf_l2_wqa_onset_warned')} and {N('kf_l3_wqa_onset_warned')}. One GEFS member "
               f"lifts that to {N('kf_l2_wqaf_sub_onset_warned')} and {N('kf_l3_wqaf_sub_onset_warned')}. {cap(N('fw_nse_forecast'))}. "
               f"Perfect rain, {fw_upper}, gives {N('kf_l2_wqap_onset_warned')} and {N('kf_l3_wqap_onset_warned')}: still NOT YET.\n"
               f"If asked about the 1-day drop: {N('kf_l1_wqaf_sub_onset_warned')} with the forecast against "
               f"{N('kf_level_wqa_onset_warned')} with AORC rain to midnight ({fw_aorc}; {fw_hind}), but "
               f"{N('kf_l1_wqaf_sub_onset_false')} against {N('kf_level_wqa_onset_false')} false: {N('fw_moved_point')}.")
    content_slide(prs, R, 10, "Results IV: 2 and 3 days ahead with a real weather forecast",
                  "A public weather forecast (GEFS, one member) lifts the 2- and 3-day warnings; for onsets the verdict is still NOT YET",
                  lambda top: [
                      {"kind": "table", "box": (BODY_X, top, BODY_W, 2900000), "col_w": [6.1, 2.0, 2.0, 2.0], "size": 12.5, "name": "Table leads",
                       "header": [f"LSTM with the river gauge: share of onsets warned ({N('lstm_wqa_l1_n_seeds')} seeds; "
                                  f"{N('lstm_wq_l1_n_seeds')} for the rain table alone and perfect weather)",
                                  f"1 day ({N('onsets_l1')} onsets)",
                                  f"2 days ({N('onsets_l2')})", f"3 days ({N('onsets_l3')})"],
                       "rows": [["rain table alone", N("kf_level_wq_onset_warned"), N("kf_l2_wq_onset_warned"), N("kf_l3_wq_onset_warned")],
                                [["+ AORC rain to midnight (", (fw_aorc, {"italic": True}), ")"],
                                 [N("kf_level_wqa_onset_warned"), (f" ({fw_hind})", {"italic": True})], N("kf_l2_wqa_onset_warned"),
                                 N("kf_l3_wqa_onset_warned")],
                                ["+ GEFS forecast of the next three days, one member, substituted",
                                 f"{N('kf_l1_wqaf_sub_onset_warned')} ({N('kf_l1_wqaf_sub_onset_false')} false)",
                                 N("kf_l2_wqaf_sub_onset_warned"), N("kf_l3_wqaf_sub_onset_warned")],
                                [["perfect rain of the three days — ", (fw_upper, {"italic": True})], N("kf_l1_wqap_onset_warned"),
                                 N("kf_l2_wqap_onset_warned"), N("kf_l3_wqap_onset_warned")],
                                [["perfect weather — ", (fw_upper, {"italic": True})], N("kf_level_wqf_onset_warned"),
                                 N("kf_l2_wqf_onset_warned"), N("kf_l3_wqf_onset_warned")],
                                [f"own split (test {N('split_sat_test_dates')}), retrained with the forecast: comparable only inside that split",
                                 N("lstm_wqaf_s_l1_onset_pod"), N("lstm_wqaf_s_l2_onset_pod"), N("lstm_wqaf_s_l3_onset_pod")]],
                       "after": lambda y: [
                           {"kind": "text", "box": (BODY_X, y, BODY_W, BODY_BOTTOM - y), "size": 11.5, "paras": [
                               {"runs": [(cap(N("fw_nse_forecast")), {"italic": True}), f": median NSE {N('kf_level_wqa_nse')} → "
                                         # critic 3, item S8: the five-seed reading of the 1-day NSE step
                                         f"{N('kf_l1_wqaf_sub_nse')} at 1 day ({N('e2_forecast_nse_reading_l1')} by the seed rule); "
                                         f"{N('kf_l2_wqa_nse')} / {N('kf_l3_wqa_nse')} → "
                                         f"{N('kf_l2_wqaf_sub_nse')} / {N('kf_l3_wqaf_sub_nse')} at 2 / 3 days "
                                         f"({N('kf_l2_step_forecast_over_rain_to_midnight_nse_reading')})"], "bullet": True},
                               {"runs": [f"1 day ahead: fewer onsets warned ({N('kf_l1_wqaf_sub_onset_warned')} against the "
                                         f"{N('kf_level_wqa_onset_warned')} of AORC rain to midnight, {fw_aorc}; {fw_hind}) but far fewer "
                                         f"false warnings ({N('kf_l1_wqaf_sub_onset_false')} against {N('kf_level_wqa_onset_false')}): ",
                                         (N("fw_moved_point"), {"italic": True}),
                                         # critic 3, note N1: the seed rule on onsets reads the 1-day step as a loss
                                         f" (seed rule on onsets: {N('e2_forecast_reading_l1')} at the fixed level)"],
                                "bullet": True, "space": 2},
                               {"runs": f"Forecast rain correlates {N('gefs_correlation_l1')} / {N('gefs_correlation_l2')} / "
                                        f"{N('gefs_correlation_l3')} with the AORC rain of the target day at 1 / 2 / 3 days; the E2 rule "
                                        f"(a clear gain at 2 and 3 days) is {N('e2_met_at_2_and_3_days')}; verdict for onsets: "
                                        f"{N('kf_practice_all_not_yet')}", "bullet": True, "space": 2},
                           ]}]},
                  ], [foot_aorc, foot_upper], HI, notes10)
    slides.append((10, "Results IV: 2 and 3 days ahead with a real weather forecast", HI, 30, notes10))

    # ---- slide 11: the mid-term targets ----
    R.slide = 11
    go_warned_met = float(V("s4_stage4_share")) >= float(V("target_go_warned"))
    all_not_yet = V("kf_practice_all_not_yet") is True
    st1 = "met" if go_warned_met else "not met"
    st2 = "not met" if all_not_yet else "met"
    seeds_status = N("target_seeds_status")                                              # done for 15-18 and 20-22, not for 02-10
    st3 = ("partly" if "not for" in seeds_status else "done") if seeds_status.startswith("done") else "not done"
    st4 = R.status_word("target_scripts_reviewed") or "partly"
    st5 = "done" if gis_done else "not done"
    checks_n = int(V("team_checks_done") or 0)
    st6 = "done" if checks_n >= int(V("team_n_members")) else ("not done" if checks_n == 0 else "partly")
    st7 = R.status_word("target_stretch_stage4_network") or "not attempted"
    e4a_row2 = ""
    # critic 3, item S3: the false-share target is not met on the main split (the E4a capped level comes out above the cap on the
    # test years) and is met alone only by the E4e model on the satellite-era split; both false shares are printed
    cap_v = float(V("go_max_false")) + 1e-9
    sat_far_ok = e4e_reviewed and float(e4e[("wqaWHF_s", "far")][1]) <= cap_v
    main_far_ok = R.has("e4a_lstm_wqa_ll1_far_cap_test_onset_far") and float(V("e4a_lstm_wqa_ll1_far_cap_test_onset_far")) <= cap_v
    st2_cell = st2 + (" on the main split" if st2 == "not met" and sat_far_ok and not main_far_ok else "")
    if e4a_reviewed and e4a_x:
        e4a_row2 = (f"; at the {N('go_max_false')}-false level: {e4a_x} of the onsets warned on the main split, "
                    f"{N('e4a_lstm_wqa_ll1_far_cap_test_onset_far')} false (E4a, reviewed)")
        if e4e_reviewed:
            e4a_row2 += (f"; {e4e[('wqaWHF_s', 'pod')][0]} at the {N('go_max_false')}-false level, {sat_years} split, "
                         f"{e4e[('wqaWHF_s', 'far')][0]} false, with the river's state, within-day rain and the forecast (E4e, "
                         "reviewed; comparable only inside it)")
        if st2_cell != st2:
            e4a_row2 += (f"; {st2_cell} ({N('e4a_lstm_wqa_ll1_far_cap_test_onset_far')} false at the capped level), met alone only "
                         f"on the satellite-era split ({e4e[('wqaWHF_s', 'far')][0]}, comparable only inside it)")
    # critic 3, item M2: the mid-term "below 50% false" status is built from its two sources (the forecast model and the rain to
    # midnight alone, one day ahead), not from the typed text of midterm_ppt_target_50_status
    m50 = re.search(r"below (\d+)%", str(V("midterm_ppt_target_2").get("target", "")))
    lim50 = int(m50.group(1)) / 100 if m50 else None
    if lim50 is None:
        warn("midterm_ppt_target_2 has no 'below N%' target: the interim false-share row keeps the registry text")
        t50_text, t50_status = N("midterm_ppt_target_50_status"), str(V("midterm_ppt_target_50_status"))
    else:
        f_met = float(V("lstm_wqaf_sub_l1_onset_far")) < lim50
        a_met = float(V("lstm_wqa_l1_onset_far")) < lim50
        word = lambda ok: "met" if ok else "not met"
        t50_text = (f"{word(f_met)} at 1 day by the forecast model ({N('lstm_wqaf_sub_l1_onset_far')} false), "
                    f"{'and' if a_met else 'not'} by the rain to midnight alone ({N('lstm_wqa_l1_onset_far')})")
        t50_status = ("met by the forecast model only" if f_met and not a_met else
                      "met" if f_met and a_met else "met by the rain to midnight only" if a_met else "not met")
        if t50_text != N("midterm_ppt_target_50_status"):
            warn(f"midterm_ppt_target_50_status text differs from its sources; the deck prints: {t50_text}")
    e4_status = []
    for label, present, reviewed, vkey in (("E4a", e4a_present, e4a_reviewed, "e4a_review_verdict"),
                                           ("E4b", e4b_present, e4b_reviewed, "e4b_review_verdict"),
                                           ("E4e", e4e_present, e4e_reviewed, "e4e_review_verdict")):
        e4_status.append(f"{label}: done, reviewed ({N(vkey)})" if reviewed else
                         (f"{label}: {N('fw_preliminary_label')}, review pending" if present else f"{label}: not run"))
    if e4a_reviewed and e4b_reviewed and e4e_reviewed and len({N("e4a_review_verdict"), N("e4b_review_verdict"), N("e4e_review_verdict")}) == 1:
        e4_status = [f"E4a, E4b and E4e (28–29 Oct): done, reviewed ({N('e4e_review_verdict')})"]
    e4_status_text = " " + "; ".join(e4_status) + "."
    # the Stage IV share carries the short REALTIME wording and the perfect-weather share UPPER in the spoken notes too (critic 2,
    # item 4); the opener and the hand-over were shortened to keep the pace
    # critic 3, item S3: "not met on the main split" in the notes; "a day ahead" dropped to keep the pace (the table says 1 day)
    notes11 = (f"{cap(N('target_go_warned'))} of onsets warned with real-time inputs: {st1}, {N('s4_stage4_share')} with "
               f"Stage IV, {fw_rt_short}; perfect weather: {N('kf_level_wqf_onset_warned')}, {fw_upper}. {cap(N('target_go_false'))} "
               f"false: {st2_cell}. Five trainings: {st3}; baselines keep {n3}. GIS layers: {st5}. Own checks: {N('team_checks_done')}, {st6}. "
               f"Radar-trained network: {st7}. {first_name(A)} next.")
    seeds_header = f"Where it stands (test years; {n5} seeds for scripts {ext_scripts} and E4, {n3} for the stored sets)"
    seeds_cell = f"{seeds_status}; {seed_disclosure_short}"
    content_slide(prs, R, 11, "Results V: the mid-term targets, met and not met",
                  "Targets stay targets; every status is read from the results files on the build day",
                  lambda top: [
                      {"kind": "table", "box": (BODY_X, top, BODY_W, BODY_BOTTOM - top), "col_w": [3.8, 7.0, 1.3], "size": 11,
                       "first_col_bold": False, "name": "Table targets",
                       "header": ["Target set at mid-term", seeds_header, "Status"],
                       "rows": [
                           [f"Onsets warned 1 day ahead with inputs a service could read at the issue time: {N('target_go_warned')} (GO); "
                            f"mid-term {N('midterm_headline_warned_wq')} → {N('midterm_headline_warned_s4')}",
                            f"{N('s4_stage4_share')} with Stage IV in AORC's place ({fw_rt_short}); {N('kf_level_wqa_onset_warned')} with "
                            f"AORC ({fw_aorc}; {fw_hind}); even a perfect weather forecast {N('kf_level_wqf_onset_warned')} ({fw_upper})",
                            [(st1, {"bold": True})]],
                           [f"False onset warnings {N('target_go_false')} of the warnings (GO); mid-term {N('midterm_headline_far_wq')} → "
                            f"{N('kf_l1_wqaf_sub_onset_false')}",
                            f"{N('kf_level_wqa_onset_false')} with AORC rain to midnight; {N('kf_l1_wqaf_sub_onset_false')} with the GEFS "
                            f"forecast beside it{e4a_row2}; {N('kf_practice_all_not_yet')}", [(st2_cell, {"bold": True})]],
                           [f"Trainings per model where the NSE decides: {N('target_seeds')}", seeds_cell, [(st3, {"bold": True})]],
                           [f"Review records: {N('target_scripts_reviewed')}",
                            f"{N('review_records')} {review_kind}, {N('review_rounds')} rounds, {N(rc_key)} checks listed, "
                            f"{N('review_none_failed')}{critic_clause}; engine: {N('engine_review_n_reviewers')} reviewers, "
                            f"{N('engine_review_checks_passed')} checks; {N('engine_review_note_scripts')}" + e4_status_text + f" {cap(rev5_row)}.",
                            [(st4, {"bold": True})]],
                           [f"GIS layers: {N('target_gis_layers')}",
                            f"statistics, maps and the export as GIS layers: "
                            f"{N('gis_layers_exported')}; independent review ({N('gis_layers_review_date')}): "
                            f"{N('gis_layers_review_verdict')}", [(st5, {"bold": True})]],
                           [f"Own checks: {N('target_own_checks')} members", f"{N('team_checks_done')} (team.json). {team_sentence}",
                            [(st6, {"bold": True})]],
                           [f"Stretch: {N('target_stretch_stage4_network')}",
                            f"the real-time check (Stage IV in AORC's place at forecast time, {fw_rt_short}) stands in its place",
                            [(st7, {"bold": True})]],
                           [f"Mid-term PPT interim target: {N('midterm_ppt_target_1')}",
                            f"{N('midterm_ppt_target_45_status')} — {fw_rt_short}",
                            [(str(V("midterm_ppt_target_45_status")), {"bold": True})]],
                           [f"Mid-term PPT interim target: {N('midterm_ppt_target_2')}", t50_text,
                            [(t50_status, {"bold": True})]],
                       ]},
                  ], [foot_aorc, foot_s4, foot_upper], HI, notes11)
    slides.append((11, "Results V: the mid-term targets, met and not met", HI, 30, notes11))

    # ---- slide 12: discussion ----
    R.slide = 12
    if e4a_reviewed:
        e4a_text = [("E4a, warning level and stacker on the stored forecasts (no training; reviewed: ", {}),
                    (N("e4a_review_verdict"), {"italic": True}),
                    (f") — the LSTM with AORC rain to midnight ({fw_aorc}): {e4a_c1}; {e4a_c2}.", {})]
    elif e4a_present:
        e4a_text = [(f"{N('fw_preliminary_label')} — ", {"bold": True}), (e4a_line_slide9, {})]
    else:
        e4a_text = [(e4a_line_slide9, {})]
    if e4b_reviewed:
        e4b_text = f"E4b, within-day AORC rain as an input ({fw_aorc}): {N('e4b_outcome_full')}"
    elif e4b_present:
        e4b_text = (f"E4b, within-day AORC rain as an input: trained on 28–29 October; its results file is {N('e4b_status').split(':')[0].lower()} "
                    "until an independent review record exists, so no number of it is shown here")
    else:
        e4b_text = "Planned, not run: the daily total hides when the rain fell; a within-day rain input is to be trained (E4b)"
    if e4e_reviewed:
        e4e_text = f"E4e, the river at the issue time from the hourly USGS flow: {N('e4e_outcome_full')}"
    else:
        e4e_text = ("Next (E4e): the river's state at the issue time from the 15-minute USGS flow, the one lead-1 input the daily model "
                    "has never seen; its download and training did not fit before the deadline")
    clock_bullets = [
        {"runs": f"Rain clock ({N('clock_window_hours')} around each onset, mean {N('clock_total_mm')}): in the table at the "
                 f"issue time {N('clock_share_table')} ({N('clock_mean_mm_table')}); fallen but stamped on the next day "
                 f"{N('clock_share_hidden')} ({N('clock_mean_mm_hidden')}); still to come {N('clock_share_future')}", "bullet": True},
        {"runs": f"Of the {N('clock_missed_n')} onsets the current model misses, the median share of their rain known at "
                 f"the issue time is {N('clock_missed_median_share_known_by_table')}; {N('clock_missed_median_share_known_by_midnight')} "
                 "with the rain to midnight", "bullet": True, "space": 2},
    ]
    sub12 = (f"The missing information has a clock: of the rain around the {N('clock_n_onsets')} onsets, {N('clock_share_table')} is in "
             f"the table at the issue time, {N('clock_share_hidden')} has fallen but is filed under the next day, "
             f"{N('clock_share_future')} is still to come")
    if e4e_reviewed:
        # The reviewed E4 results: the E4e ladder on the satellite-era split as a table, E4b and E4a in one line each. No figure:
        # figures/flow_state.png cropped to its top panel would render its labels under 5 pt at the width left beside the table
        # (the report carries the figure); the table holds the same numbers.
        title12 = "Discussion: the missing information has a clock, and the river's state at the issue time supplies part of it"
        cap30 = N("go_max_false")
        split_line = (f"{sat_years} test years of the satellite-era split, {e4e_onsets} onsets, comparable only inside it; {e4e_seeds} seeds "
                      f"per set, the stored ones included; with the gauge record ending at 22:00 instead of midnight the full "
                      f"model warns of {N('e4e_wqawhf22_s_l1_onset_pod')} at {N('e4e_wqawhf22_s_l1_onset_far')} false; E4e reviewed "
                      f"{e4e_date}: {N('e4e_review_verdict')}")
        e4b_core = ((f"within-day AORC rain on the main split ({fw_aorc}): onsets warned 1 day ahead {N('e4b_wqa_l1_onset_pod')} "
                     f"({fw_hind}) → {N('e4b_wqaw_l1_onset_pod')} ({N('e4b_wqa_l1_onset_hits')} → {N('e4b_wqaw_l1_onset_hits')} of "
                     f"{N('onsets_l1')}), false {N('e4b_wqa_l1_onset_far')} → {N('e4b_wqaw_l1_onset_far')}; strict seed rule {e4b_rule}; "
                     f"at the {cap30}-false level nothing over the one-day-older control (reviewed {e4b_date})")
                    if e4b_reviewed else runs_text(e4b_text))
        e4b_line = ("E4b, " + e4b_core) if e4b_reviewed else e4b_core
        strict = N("e4e_outcome_caveat").split(":")[0]
        strict = strict[:1].lower() + strict[1:]
        e4a_line = (f"E4a, operating point on the stored forecasts (no training): the rain-to-midnight model ({fw_aorc}) warns of "
                    f"{N('e4a_lstm_wqa_ll1_far_cap_test_onset_pod')} of the onsets at the {cap30}-false level "
                    f"({N('e4a_lstm_wqa_ll1_far_cap_test_onset_far')} false) against {N('e4a_lstm_wqa_ll1_csi_test_onset_pod')} at the CSI "
                    f"level ({fw_hind}; {N('e4a_lstm_wqa_ll1_csi_test_onset_far')} false): a decision layer moves the trade-off and creates "
                    f"no skill (reviewed {review_dates('e4a', 'warning_policy')})")
        after_lines = [{"runs": split_line, "bullet": True}, {"runs": e4b_line, "bullet": True, "space": 2},
                       {"runs": e4a_line, "bullet": True, "space": 2}]
        if any(e4e_star.values()):
            after_lines.append({"runs": f"* above the {cap30} cap on the test years (every trigger is chosen on the validation years)",
                                "size": 10, "color": GREY, "space": 2})
        # a share at the capped level whose digits coincide with a flagged share of the main split (the AORC 51 %, the Stage IV
        # 48 %, the perfect-weather 64 %) names its level and split in its own cell (critic 1, item 22), in a smaller run
        at_cap = (f" at the {cap30}-false level, {sat_years} split", {"size": 8.5, "color": GREY})
        flagged_shares = {N("kf_level_wqa_onset_warned"), N("s4_stage4_share"), N("kf_level_wqf_onset_warned")}
        capped = lambda s: [(e4e[(s, "pod")][0], {}), at_cap] if e4e[(s, "pod")][0] in flagged_shares else e4e[(s, "pod")][0]
        ladder = [
            [["rain to midnight (wqa_s, stored; ", (fw_aorc, {"italic": True}), ")"], N("e4e_wqa_s_l1_onset_pod"),
             N("e4e_wqa_s_l1_onset_far"), capped("wqa_s"), e4e[("wqa_s", "far")][0] + e4e_star["wqa_s"]],
            ["+ river's state at the issue time (hourly USGS gauge; wqaH_s)", N("e4e_wqah_s_l1_onset_pod"),
             N("e4e_wqah_s_l1_onset_far"), capped("wqaH_s"), e4e[("wqaH_s", "far")][0] + e4e_star["wqaH_s"]],
            ["+ within-day AORC rain, six columns (wqaWH_s)", N("e4e_wqawh_s_l1_onset_pod"), N("e4e_wqawh_s_l1_onset_far"),
             capped("wqaWH_s"), e4e[("wqaWH_s", "far")][0] + e4e_star["wqaWH_s"]],
            ["+ GEFS weather forecast, one member (wqaWHF_s)", N("e4e_wqawhf_s_l1_onset_pod"), N("e4e_wqawhf_s_l1_onset_far"),
             capped("wqaWHF_s"), e4e[("wqaWHF_s", "far")][0] + e4e_star["wqaWHF_s"]],
            [["perfect rain (wqaP_s): ", (fw_upper, {"italic": True})], N("e4e_wqap_s_l1_onset_pod"),
             N("e4e_wqap_s_l1_onset_far"), capped("wqaP_s"), e4e[("wqaP_s", "far")][0] + e4e_star["wqaP_s"]],
        ]
        # the E4e share at the capped level names its level and split (the exempting phrase of wording_rules()) and the perfect-rain
        # share carries UPPER in the spoken notes (critic 2, item 4); words were cut elsewhere to keep the pace
        notes12 = (f"{N('clock_share_table')} of an onset's rain is in the table, {N('clock_share_hidden')} has fallen but is filed "
                   f"under tomorrow, {N('clock_share_future')} to come. The river's hourly record at the issue time is the new input. "
                   f"With it, within-day rain and a forecast, {e4e[('wqaWHF_s', 'pod')][0]} of onsets are warned at the {cap30}-false "
                   f"level, {sat_years} split, against {e4e[('wqa_s', 'pod')][0]} without; perfect rain: {e4e[('wqaP_s', 'pod')][0]}, "
                   f"{fw_upper}. GO not met; the gauge record comes late and provisional; comparable only inside that split.\n"
                   f"If asked about delivery: with the record ending at 22:00 instead of midnight the full model warns of "
                   f"{N('e4e_wqawhf22_s_l1_onset_pod')} at {N('e4e_wqawhf22_s_l1_onset_far')} false; the midnight variant assumes no delay "
                   "and approved values, and a service gets provisional values after a delay.\n"
                   f"If asked about the strict rule: {strict}.\n"
                   f"If asked about E4b: {e4b_core}.")
        # geometry seen in the LibreOffice render of 29 Oct: the two clock bullets take three lines (0.6 in), so the table starts
        # at 0.64 in and the E4 lines keep about 1.15 in above the footer; 30 Oct (critic 3, S6): the review clauses grew by the
        # five-seed rounds, so the table box is 2.24 in at 11 pt (was 2.52 in at 12 pt) and the E4 lines 10 pt (was 10.5)
        elements12 = lambda top: [
            {"kind": "text", "box": (BODY_X, top, BODY_W, 620000), "size": 11, "paras": clock_bullets, "name": "Clock"},
            {"kind": "table", "box": (BODY_X, top + 640000, BODY_W, 2050000), "col_w": [5.5, 1.3, 1.1, 2.9, 1.35], "size": 11,
             "first_col_bold": False, "name": "Table flow state",
             "header": [f"Onsets warned 1 day ahead ({sat_years} satellite-era split, LSTM + gauge)", "Fixed level", "False",
                        f"{cap30}-false level", "False there"],
             "rows": ladder, "after_gap": 70000,
             "after": lambda y: [{"kind": "text", "box": (BODY_X, y, BODY_W, BODY_BOTTOM - y), "size": 10, "paras": after_lines,
                                  "name": "E4 lines"}]},
        ]
        footer12 = [foot_aorc, foot_upper, "hourly flow: the gauge's own record, provisional and after a delay in a service"]
    else:
        title12 = "Discussion: why the warning is limited, and what is being tried now"
        e4_notes = []
        if e4a_reviewed and e4a_x:
            e4_notes.append(f"At the level that keeps the false share at the GO rule's {N('go_max_false')}, the rain-to-midnight model "
                            f"warns of {e4a_x}: a decision layer re-arranges the trade-off and creates no skill.")
        e4_notes.append("The next inputs, within-day rain and the river's state at the issue time, "
                        + ("wait for review." if e4b_present and not e4b_reviewed else "are planned, not run."))
        notes12 = (f"The missing information has a clock. Of the rain around an onset, {N('clock_share_table')} is in the table when the "
                   f"forecast is issued, {N('clock_share_hidden')} has fallen but is filed under tomorrow, and {N('clock_share_future')} "
                   f"has not fallen yet: real-time rain and a weather forecast. " + " ".join(e4_notes))
        elements12 = lambda top: [
            {"kind": "text", "box": (BODY_X, top, 6200000, BODY_BOTTOM - top), "size": 11, "paras": clock_bullets + [
                {"runs": f"{N('clock_class_75_100_n')} onsets whose rain is filed almost entirely under the target day are warned "
                         f"{N('fail_rain_75_100_onset_pod')} of the time today, {N('rescue_rain_75_100_perfect_share')} of these "
                         f"{N('clock_class_75_100_n')} with the perfect forecast ({fw_upper}; a share of this rain class, not of all "
                         f"{N('onsets_l1')} onsets)", "bullet": True, "space": 2},
                {"runs": "What this asks for: rain to the issue time from a real-time product for the second part, a weather "
                         "forecast for the third, and the river's state at the issue time, which the daily model has never seen",
                 "bullet": True, "space": 2},
                {"runs": e4a_text, "bullet": True, "space": 2},
                {"runs": e4b_text, "bullet": True, "space": 2},
                {"runs": e4e_text, "bullet": True, "space": 2},
            ]},
            {"kind": "pic", "path": FIG / "day_windows.png", "box": (6800000, top, BODY_RIGHT - 6800000, BODY_BOTTOM - top),
             "align": "r", "name": "Figure day windows"},
        ]
        footer12 = [foot_aorc, foot_upper] + ([N("fw_preliminary_label") + ": E4a"] if (e4a_present and not e4a_reviewed) else [])
    content_slide(prs, R, 12, title12, sub12, elements12, footer12, A, notes12)
    slides.append((12, title12, A, 35, notes12))

    # ---- slide 13: conclusions and limits ----
    R.slide = 13
    # the AORC share carries both its wordings in the spoken notes too (critic 2, item 4); words were cut elsewhere to keep the pace
    notes13 = (f"The finding holds both ways. Taking information away lowered skill: the river gauge is worth "
               f"{N('kf_step_river_gauge_nse')} NSE; a held-out basin loses {N('lobo_loss_median_nse')}. Giving information raised it: "
               f"{N('kf_level_wq_onset_warned')} to {N('kf_level_wqa_onset_warned')} of onsets a day ahead with AORC, {fw_aorc}, "
               f"{fw_hind}; {N('kf_l2_wq_onset_warned')} and {N('kf_l3_wq_onset_warned')} to {N('kf_l2_wqaf_sub_onset_warned')} and "
               f"{N('kf_l3_wqaf_sub_onset_warned')} at two and three days. About half the warnings are false: not usable yet. "
               f"Limits: {N('n_basins')} basins, {N('split_test_water_years')} test years, {n3} or {n5} trainings.\n"
               f"If asked about the hindsight bound: with {n5} seeds {N('fw_hindsight')} ({N('hindsight_bound_five_seeds_points')} "
               f"points, {N('hindsight_bound_five_seeds_hits')} hits; {N('hindsight_level_restored_points')} points, "
               f"{N('hindsight_level_restored_hits')} hits, with the AORC rain level restored); with the {n3} networks of round 1 of "
               # critic 3, item S7: the 16-of-313 bound is check d7 of the E1b review record, written on date_review_e1b
               f"the E1b review ({N('date_review_e1b')}), AORC {N('aorc_share_three_seeds')}, {N('hindsight_three_seed_wording')} "
               f"({N('hindsight_bound_three_seeds_hits')} hits).")
    content_slide(prs, R, 13, "Conclusions and limits",
                  "The finding holds in both directions; the warning is not usable yet; the limits are stated",
                  lambda top: [
                      {"kind": "text", "box": (BODY_X, top, BODY_W, BODY_BOTTOM - top), "size": 14, "paras": [
                          {"runs": [("Taking information away lowers skill: ", {"bold": True}), f"the river gauge is worth "
                                    f"{N('kf_step_river_gauge_nse')} NSE 1 day ahead (higher in {N('kf_step_river_gauge_n_basins_better')} basins); "
                                    f"a basin held out of training loses {N('lobo_loss_median_nse')} NSE (median; {N('lobo_loss_n_basins_lower')} "
                                    "basins lower)"], "bullet": True},
                          {"runs": [("Giving information raises it: ", {"bold": True}), f"onsets warned 1 day ahead {N('kf_level_wq_onset_warned')} → "
                                    f"{N('kf_level_wqa_onset_warned')} with AORC rain to midnight ({fw_aorc}; {fw_hind}), {N('s4_stage4_share')} "
                                    f"with Stage IV in its place ({fw_rt_short}); 2 and 3 days ahead {N('kf_l2_wq_onset_warned')} / "
                                    f"{N('kf_l3_wq_onset_warned')} → {N('kf_l2_wqaf_sub_onset_warned')} / {N('kf_l3_wqaf_sub_onset_warned')} with a "
                                    "public forecast"], "bullet": True, "space": 5},
                          *([{"runs": [("The river's state at the issue time ", {"bold": True}), "(E4e, the hourly USGS gauge record), with "
                                       f"within-day rain and a forecast, lifts the warning from {e4e[('wqa_s', 'pod')][0]} to "
                                       f"{e4e[('wqaWHF_s', 'pod')][0]} of the onsets a day ahead at the {N('go_max_false')}-false level, "
                                       f"{sat_years} split (comparable only inside it; ceiling with perfect rain {e4e[('wqaP_s', 'pod')][0]}, "
                                       f"{fw_upper}); GO is still not met"], "bullet": True, "space": 5}] if e4e_reviewed else []),
                          {"runs": [("More rivers in training ", {"bold": True}), f"({N('e3_n_rivers')} instead of {N('n_basins')}): "
                                    f"{N('e3_nse_reading')} in NSE for the {N('n_basins')} ({N('e3_nse_12_rivers')} → {N('e3_nse_63_rivers')}); "
                                    f"onsets warned {N('e3_onsets_12_rivers')} → {N('e3_onsets_63_rivers')} (seed rule: "
                                    f"{N('e3_onset_seed_reading')}; McNemar p = {N('e3_mcnemar_p')}); rivers never seen "
                                    f"{N('unseen_nse_11_rivers_with_properties')} → {N('unseen_nse_50_rivers_with_properties')} NSE"],
                           "bullet": True, "space": 5},
                          {"runs": [("Still missing: ", {"bold": True}), f"about half of the onset warnings are false "
                                    f"({N('kf_level_wqa_onset_false')} with the rain to midnight); {N('kf_practice_all_not_yet')}"],
                           "bullet": True, "space": 5},
                          {"runs": [("Limits: ", {"bold": True}), f"{N('n_basins')} basins, {N('split_test_water_years')} test years, "
                                    f"CPU only (a laptop and a Windows PC); {seed_sentence}; AORC is {fw_aorc}"],
                           "bullet": True, "space": 5},
                          {"runs": [("Milestone 2 of the plan ", {"bold": True}), f"(the LSTM beats the best simple model): {N('milestone_m2_status')}"],
                           "bullet": True, "space": 5},
                      ]},
                  ], [foot_aorc, foot_s4], A, notes13)
    slides.append((13, "Conclusions and limits", A, 35, notes13))

    # ---- slide 14: who did what, references, thank you ----
    R.slide = 14
    ref_keys = ["newman2015", "addor2017", "hochreiter1997", "kratzert2019", "kratzert2019pub", "kratzert2024", "feng2020",
                "nearing2024", "nash1970", "thornton2021"]
    # the numbers of the report's reference list (critic 1, item 21): [n] as DOCX section 11 prints them, sorted by that number;
    # the deck's own order of first citation only when the report cannot be read or lacks one of the keys
    report_num = report_reference_numbers()
    if all(k in report_num for k in ref_keys):
        refs = [REFS.format_ieee(k, report_num[k]) for k in sorted(ref_keys, key=lambda k: report_num[k])]
        refs_title = "References (IEEE), numbered as in the report"
    else:
        missing = [k for k in ref_keys if k not in report_num]
        if report_num:
            warn(f"reference keys not numbered in the report DOCX: {', '.join(missing)}; the deck keeps its own numbering")
        refs = REFS.cite_order(ref_keys)
        refs_title = "References (IEEE)"
    authors = N("flood_cloud_model_authors")
    authors = authors[:authors.find(")") + 1] if ")" in authors else authors
    authors_short = authors.split(" (")[0]
    notes14 = (f"Claude Code wrote all code and ran all experiments under my direction. "
               f"Independent AI reviewers checked the engine and every experiment script; {critic_spoken}. {team_sentence} "
               f"Flood_Cloud_Model, a separate gradient-boosting experiment, was built by {authors_short}. "
               f"Five-seed numbers recomputed independently: {N('review_five_seed_mismatches')} mismatches. "
               f"All data open; nothing bought. Thank you; we welcome your questions.")
    content_slide(prs, R, 14, "Who did what, references, thank you",
                  "The honesty statement in full, the work in numbers, the sources",
                  lambda top: [
                      {"kind": "text", "box": (BODY_X, top, 5500000, BODY_BOTTOM - top), "size": 11, "paras": [
                          {"runs": [("Who did what: ", {"bold": True}), f"{N('fw_who_did_what')[:-1]}; {N('fw_review')}."], "bullet": True},
                          {"runs": [(team_sentence, {"bold": True})], "bullet": True, "space": 3},
                          {"runs": [("Flood_Cloud_Model package ", {"bold": True}), "— a separate gradient-boosting experiment, not a version of "
                                    f"our LSTM and not part of our pipeline: built by {authors}; discussed in the report"], "bullet": True, "space": 3},
                          {"runs": [("The work in numbers: ", {"bold": True}), f"{N('work_trainings')} trainings in {N('work_hours')} "
                                    f"({N('seed_extension_trainings')} of them on the Windows PC){e4_work}; {rev5_short}; "
                                    f"{N('gb_read')} of open data read; {N('review_records')} {review_kind}, {N('review_rounds')} "
                                    f"review rounds with {N(rc_key)} checks listed, {N('review_none_failed')}{critic_clause}; money spent: "
                                    f"{N('money_spent_none')}"], "bullet": True, "space": 3},
                          {"runs": [("Disclosure: ", {"bold": True}), f"{cap(seed_disclosure)}."], "bullet": True, "space": 3},
                          {"runs": links_runs, "bullet": True, "space": 3},
                          {"runs": [("Thank you — we welcome your questions.", {"bold": True, "size": 14, "color": BLUE})], "space": 10},
                      ]},
                      {"kind": "text", "box": (6150000, top, BODY_RIGHT - 6150000, BODY_BOTTOM - top), "size": 8.5, "color": NAVY, "paras":
                          [{"runs": [(refs_title, {"bold": True, "size": 11})]}] + [{"runs": r, "space": 2} for r in refs],
                       "name": "References"},
                  ], [], A, notes14)
    slides.append((14, "Who did what, references, thank you", A, 30, notes14))

    # ---- save ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"saved {out} ({removed} sample slides removed, {len(prs.slides)} slides written)")
    write_notes(Path(args.notes), slides, R, cover_which, out)
    return out, slides, cover_which


def write_notes(path, slides, R, cover_which, pptx_path):
    today = _dt.date.today().isoformat()
    lines = [f"# Final presentation — speaker notes · {R.text('final_presentation')}", "",
             f"Deck: `{pptx_path}` (built {today} by `code/build_slides_final.py` on the professor's template; "
             f"cover picture: {cover_which}). Every number comes from `results/final_report_numbers.json`.",
             f"Pace rule: at most {PACE} words a second. Four speakers, {sum(s[3] for s in slides)} s in all.",
             ("'we' means the project; the team's own checks are not yet done (team.json)" if int(R.val("team_checks_done") or 0) == 0
              else f"'we' means the project; the team's own checks recorded in team.json: {R.text('team_checks_done')}"), "",
             "## Timing", "", "| Slide | Title | Speaker | Seconds | Words | Words/s |", "|---|---|---|---|---|---|"]
    per_speaker = {}
    for n, title, speaker, secs, notes in slides:
        spoken = " ".join(p for p in notes.split("\n") if not p.startswith("If asked"))   # 'If asked' facts are not spoken
        words = len(spoken.split())
        per_speaker[speaker] = per_speaker.get(speaker, 0) + secs
        wps = words / secs
        flag = " ⚠" if wps > PACE + 1e-9 else ""                  # the rule is the rule: no allowance above PACE (critic 1, item 9)
        lines.append(f"| {n} | {title} | {speaker} | {secs} | {words} | {wps:.2f}{flag} |")
        if flag:
            warn(f"slide {n}: {words} words in {secs} s = {wps:.2f} words/s, over the pace rule of {PACE}")
    lines += ["", "| Speaker | Seconds |", "|---|---|"] + [f"| {s} | {t} |" for s, t in per_speaker.items()]
    lines += ["", "Clock checkpoints (talk plan): slide 3 at 0:50, slide 5 at 3:20, slide 8 at 5:50, slide 12 at 8:20. "
              "Late cut if slide 11 starts after 8:30: member 4 reads only the first two rows of the targets table; "
              "The team lead drops slide 12's clock bullets and reads only the first and the fourth row of the ladder table.", ""]
    for n, title, speaker, secs, notes in slides:
        lines += [f"## Slide {n} — {title} · {speaker} · {secs} s", ""] + [p for p in notes.split("\n")] + [""]
    lines += ["## Registry keys used, by slide (results/final_report_numbers.json; sources in code/final_report/NUMBERS_KEYS.md)", ""]
    for n in sorted(R.by_slide):
        label = f"slide {n}" if n else "shared by several slides"
        lines.append(f"- {label}: " + ", ".join(f"`{k}`" for k in sorted(R.by_slide[n])))
    if DIRECT:
        lines += ["", "## Numbers read directly from a results file (the registry has no key for them yet: add the key to "
                  "code/final_report/numbers.py, rebuild the registry and this deck, and this list empties itself)", ""]
        for label, (source, text, n) in DIRECT.items():
            lines.append(f"- {'slide ' + str(n) if n else 'shared by several slides'}: `{label}` = {text} — {source}")
    if WARNINGS:
        lines += ["", "## Builder warnings", ""] + [f"- {w}" for w in WARNINGS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"notes written: {path}")


def wording_rules():
    """The fixed wordings every slide unit (a paragraph or a table row) must carry beside a flagged share (critic 1, item 22):
    the Stage IV share needs REALTIME_WORDING (short form), the AORC share needs the AORC and the hindsight wording, the
    perfect-weather share needs UPPER. Read from the registry so the shares and wordings are never typed here."""
    d = json.loads(REGISTRY.read_text(encoding="utf-8"))
    t = lambda k: str(d[k]["text"])
    sat_years = "–".join(str(s)[:4] for s in d["split_sat_test_dates"]["value"])
    exempt = f"at the {t('go_max_false')}-false level, {sat_years} split"    # the E4e share at the capped level, disambiguated
    return [(t("s4_stage4_share"), "Stage IV", [t("fw_realtime_short")], None),
            (t("kf_level_wqa_onset_warned"), "AORC", [t("fw_aorc"), t("fw_hindsight")], exempt),
            (t("kf_level_wqf_onset_warned"), "perfect weather", [t("fw_upper")], None)]


def check_wordings(slide, i, rules, report):
    """Every paragraph and every table row of the slide that prints a flagged share must carry its wording in that same unit,
    unless the unit carries the exempting phrase (the E4e share at the capped level names its level and split instead).
    A unit without it is appended to report as (slide, unit text, what is missing); nothing is changed."""
    units = []
    for sh in slide.shapes:
        if sh.has_text_frame:
            units += ["".join(r.text for r in p.runs) for p in sh.text_frame.paragraphs]
        if sh.has_table:
            units += [" | ".join(c.text for c in row.cells) for row in sh.table.rows]
    if slide.has_notes_slide:      # the speaker notes, paragraph by paragraph (critic 2, item 4)
        units += ["notes: " + p.text for p in slide.notes_slide.notes_text_frame.paragraphs if p.text.strip()]
    for u in units:
        for share, label, needed, exempt in rules:
            if re.search(r"(?<![\d.])" + re.escape(share), u) and not (exempt and exempt in u):
                lacking = [w for w in needed if w not in u]
                if lacking:
                    report.append((i, u, f"{share} ({label}) without: " + "; ".join(lacking)))


def validate(pptx_path, expected=14):
    """Reopen the deck: slide count, every shape inside the slide, notes on every slide, no forbidden token, the fixed wordings
    beside every flagged share (listed, not fatal: a different number with the same digits may be flagged); print the text."""
    prs = Presentation(str(pptx_path))
    problems = []
    wording_report = []
    rules = wording_rules()
    if len(prs.slides) != expected:
        problems.append(f"{len(prs.slides)} slides, expected {expected}")
    layout_text = " ".join(sh.text_frame.text for sh in prs.slide_layouts[1].shapes if sh.has_text_frame)
    print(f"\nlayout 2 fixed text: {layout_text.strip()!r}")
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        check_wordings(slide, i, rules, wording_report)
        for sh in slide.shapes:
            if sh.left < 0 or sh.top < 0 or sh.left + sh.width > prs.slide_width or sh.top + sh.height > prs.slide_height:
                problems.append(f"slide {i}: shape {sh.name!r} outside the slide ({sh.left}, {sh.top}, {sh.width}, {sh.height})")
            if sh.has_text_frame and sh.text_frame.text.strip():
                texts.append(sh.text_frame.text.strip())
            if sh.has_table:
                for row in sh.table.rows:
                    texts.append(" | ".join(c.text for c in row.cells))
        notes = slide.notes_slide.notes_text_frame.text.strip() if slide.has_notes_slide else ""
        if not notes:
            problems.append(f"slide {i}: no speaker notes")
        blob = "\n".join(texts) + "\n" + notes
        for tok in FORBIDDEN:
            if tok in blob:
                problems.append(f"slide {i}: forbidden token {tok!r} in the text")
        print(f"\n===== slide {i} ({slide.slide_layout.name}) =====")
        for t in texts:
            print(textwrap.indent(t, "  "))
        print("  -- notes: " + notes.replace("\n", " ")[:3000])
    print()
    if wording_report:
        print(f"wording check: {len(wording_report)} unit(s) print a flagged share without its fixed wording "
              "(judge each: a different number with the same digits is flagged too)", file=sys.stderr)
        for i, u, what in wording_report:
            print(f"  slide {i}: {what}: {u[:160]!r}", file=sys.stderr)
    else:
        print("wording check: every paragraph and table row that prints the Stage IV, AORC or perfect-weather share carries its wording")
    if problems:
        for p in problems:
            print("CHECK FAILED: " + p, file=sys.stderr)
        return False
    print(f"checks passed: {expected} slides, every shape inside the slide, notes on every slide, no forbidden token")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cover", choices=("auto", "photo", "map"), default="auto",
                    help="cover picture: the USGS public-domain photo of code/proposal_web/assets (auto, when it exists) or figures/overview_map.png")
    ap.add_argument("--refresh-template", action="store_true", help="copy the downloaded sample again over deck_template.pptx")
    ap.add_argument("--out", default=str(OUT_PPTX))
    ap.add_argument("--notes", default=str(OUT_NOTES))
    args = ap.parse_args()
    out, slides, cover_which = build(args)
    ok = validate(out, expected=14)
    print(f"cover picture: {cover_which}")
    if WARNINGS:
        print(f"{len(WARNINGS)} builder warning(s) above", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
