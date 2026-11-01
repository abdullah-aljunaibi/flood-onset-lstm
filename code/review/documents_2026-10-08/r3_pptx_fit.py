"""r3_pptx_fit: do the texts of the PPTX fit their frames? (the PDF of the slides is made from HTML, the PPTX is a separate build: python-pptx text frames, Arial)

Estimates, for every text frame of every slide, the height of its text (Arial metrics from the system font, wrapping at the frame width minus the margins, line spacing 1.2 x the size, the paragraph spacing of the frame)
and compares it with the frame height. A frame whose estimate exceeds its height by more than 3% is listed. An estimate, not a rendering: PowerPoint was not available. Output: OUT/r3_pptx_fit.json
"""
from dcommon import *
from PIL import ImageFont
from pptx import Presentation

FONTS = {}


def font(size, bold):
    key = (round(size * 4), bold)
    if key not in FONTS:
        path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
        FONTS[key] = ImageFont.truetype(path, int(round(size * 4)))             # 4 px per pt: widths are divided by 4 again
    return FONTS[key]


def width_pt(text, size, bold):
    return font(size, bold).getlength(text) / 4


def frame_height(tf, width_pt_avail):
    total = 0.0
    for p in tf.paragraphs:
        runs = [(r.text, (r.font.size.pt if r.font.size else 12), bool(r.font.bold)) for r in p.runs if r.text]
        if not runs:
            total += 12 * 1.2
            continue
        size = max(s for _, s, _ in runs)
        words = []
        for text, s, b in runs:
            for w in text.replace("\n", " ").split(" "):
                words.append((w, s, b))
        lines, cur = 1, 0.0
        for w, s, b in words:
            wl = width_pt(w + " ", s, b)
            if cur + wl > width_pt_avail + 0.5 and cur > 0:
                lines += 1
                cur = wl
            else:
                cur += wl
        sp = (p.space_after.pt if p.space_after is not None else 0) + (p.space_before.pt if p.space_before is not None else 0)
        total += lines * size * 1.2 + sp
    return total


pr = Presentation(str(PROJ / "Flood_Slides_5.pptx"))
rows, bad = [], []
for i, s in enumerate(pr.slides, 1):
    for sh in s.shapes:
        frames = []
        if sh.has_text_frame and sh.text_frame.text.strip():
            frames.append((sh.name, sh.text_frame, sh.width / 12700, sh.height / 12700))
        if getattr(sh, "has_table", False) and sh.has_table:
            tb = sh.table
            for ri, r in enumerate(tb.rows):
                for ci, c in enumerate(r.cells):
                    frames.append((f"{sh.name} r{ri}c{ci}", c.text_frame, tb.columns[ci].width / 12700, r.height / 12700))
        for name, tf, w, h in frames:
            dm = lambda v, d: (d if v is None else v) / 12700                                  # an explicit margin of 0 is a margin
            ml, mr, mt, mb = dm(tf.margin_left, 91440), dm(tf.margin_right, 91440), dm(tf.margin_top, 45720), dm(tf.margin_bottom, 45720)
            need = frame_height(tf, w - ml - mr) + mt + mb
            rows.append(dict(slide=i, frame=name, height_pt=round(h, 1), needed_pt=round(need, 1), ratio=round(need / h, 3) if h else None, chars=len(tf.text)))
            if h and need > 1.03 * h and "r" not in name.split()[-1][:1]:
                bad.append(rows[-1])
jsave("r3_pptx_fit.json", dict(frames=len(rows), over=bad, worst=sorted(rows, key=lambda r: -(r["ratio"] or 0))[:6]))
print(f"{len(rows)} text frames; {len(bad)} estimated taller than their frame by more than 3%")
for r in sorted(rows, key=lambda r: -(r["ratio"] or 0))[:6]:
    print("  ", r)
