"""Check (g), part 1: figures/more_basins.png against results/more_basins.json, read on the PIXELS.

The four panels are found from the drawn grid lines (known tick values), every bar is found by its fill colour (grey, blue, dark blue, orange) or by its outline, the bar tops are read in the
outer columns of the bar (the seed marks sit in the middle), and the seed range (black bar with caps and dots) is read from the dark pixels of the bar's centre column. Every value is compared with the
number the JSON holds for that bar; the printed value labels, legend and notes are compared with the JSON's numbers in text (as read on the picture, listed in the record).
REVIEW_SELFTEST=6 moves every expected position by 6 px: the check must then fail.
Run: python m11_figure.py
"""
from PIL import Image
from scipy import ndimage as ndi

from m_common import *

J = json.loads((PROJ / "results" / "more_basins.json").read_text())
im = np.asarray(Image.open(PROJ / "figures" / "more_basins.png").convert("RGB")).astype(int)
H, W, _ = im.shape
hx = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])
GRID, AXISC, GREY, BLUE, BLUE2, ORANGE, INK, INK2 = (hx(c) for c in ("#e1e0d9", "#c3c2b7", "#898781", "#2a78d6", "#184f95", "#eb6834", "#0b0b0b", "#52514e"))
SHIFT = float(os.environ.get("REVIEW_SELFTEST", "0"))
S, RG, UN = J["summary_test"]["sets"], J["seed_ranges"]["sets"], J["unseen"]["sets"]
PAN = {"TL": (150, 815, 100, 600), "TR": (985, 1655, 100, 600), "BL": (150, 815, 830, 1320), "BR": (985, 1655, 830, 1320)}
TICK = {"TL": (0.1, 7), "TR": (0.05, 4), "BL": (0.1, 5), "BR": (0.05, 5)}          # tick step and number of grid lines above the zero line (read on the figure's axes labels)


def calib(k):
    x0, x1, y0, y1 = PAN[k]
    sub = im[y0:y1, x0:x1]
    frac = (np.abs(sub - GRID).sum(2) < 14).mean(1)
    rows = np.flatnonzero(frac > 0.35) + y0
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 2:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    g = [float(np.mean(x)) for x in groups]
    ax = np.flatnonzero((np.abs(sub - AXISC).sum(2) < 14).mean(1) > 0.6) + y0
    step, n = TICK[k]
    assert len(g) == n, f"{k}: {len(g)} grid lines found, {n} expected"
    vals = [step * (n - i) for i in range(n)]                                         # top line first
    fit = np.polyfit(vals + [0.0], g + [float(ax.mean())], 1)                       # row = a * value + b, the zero line from the axis spine
    resid = float(np.max(np.abs(np.polyval(fit, vals + [0.0]) - (g + [float(ax.mean())]))))
    return fit, resid


def bars_of(k, color, min_area=700):
    x0, x1, y0, y1 = PAN[k]
    m = np.zeros(im.shape[:2], bool)
    m[y0:y1, x0:x1] = np.abs(im[y0:y1, x0:x1] - color).sum(2) < 20
    lab, n = ndi.label(m)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        mm = lab[sl] == i
        if mm.sum() < min_area:
            continue
        xa, xb = sl[1].start, sl[1].stop - 1
        w = xb - xa + 1
        cols = list(range(0, max(3, w // 5))) + list(range(w - max(3, w // 5), w))        # outer fifths of the bar: the seed marks are in the middle
        tops = [np.flatnonzero(mm[:, c])[0] + sl[0].start for c in cols if mm[:, c].any()]
        out.append(dict(x=(xa + xb) / 2, xa=xa, xb=xb, top=float(np.median(tops)), bottom=sl[0].stop - 1))
    return sorted(out, key=lambda b: b["x"])


def outline_bars(k, min_run=30):
    """Bars drawn as an outline only (facecolor none): the long horizontal runs of the INK2 colour; the top edge of each."""
    x0, x1, y0, y1 = PAN[k]
    m = np.zeros(im.shape[:2], bool)
    m[y0:y1, x0:x1] = np.abs(im[y0:y1, x0:x1] - INK2).sum(2) < 24
    lab, n = ndi.label(m)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if w >= 30 and h >= 60:                                                       # an outline rectangle: wide and tall, hollow
            out.append(dict(x=(sl[1].start + sl[1].stop - 1) / 2, xa=sl[1].start, xb=sl[1].stop - 1, top=float(sl[0].start + 1.9), bottom=sl[0].stop - 1))   # +1.9 px: the centre of the 3.97 px line
    return sorted(out, key=lambda b: b["x"])


DARK = np.abs(im - INK).sum(2) < 90
_op = ndi.binary_opening(DARK, structure=np.ones((5, 5)))                                # the seed dots (black cores 7 px across) survive; lines, caps and text strokes do not
_lab, _n = ndi.label(_op)
DOTS = []
for _i, _sl in enumerate(ndi.find_objects(_lab), 1):
    _area = int((_lab[_sl] == _i).sum())
    if _area >= 20 and (_sl[1].stop - _sl[1].start) <= 9:
        DOTS.append(dict(x=(_sl[1].start + _sl[1].stop - 1) / 2, top=_sl[0].start, bottom=_sl[0].stop - 1, area=_area))


def seed_extent(k, xc, ytop, ybot):
    """The seed range of a bar, read two ways: (caps) the horizontal caps of the black range bar, found as the rows where dark pixels stick out on BOTH sides beyond the dot and its white edge
    (columns 7-12 px left and right of the bar's centre; a dot is 9 px across, so only a cap reaches there); the highest group that lies within 3.5 px of the highest seed dot is the upper cap (a glyph of
    the value label above it is not), the lowest group is the lower cap; (dots) the highest seed dot, from the 5x5-opened black blobs. Returns (upper cap row, lower cap row, highest dot row, groups)."""
    x0, x1, y0, y1 = PAN[k]
    xi = int(round(xc))
    left, right = DARK[:, xi - 12:xi - 6], DARK[:, xi + 7:xi + 13]
    ls, rs = left.sum(1), right.sum(1)
    rows = np.flatnonzero((ls >= 1) & (rs >= 1) & (ls + rs >= 4))
    rows = rows[(rows >= max(y0, ytop - 120)) & (rows <= min(y1, ybot + 3))]
    mine = [d for d in DOTS if abs(d["x"] - xc) <= 6 and y0 <= d["top"] and d["bottom"] <= min(y1, ybot + 6)]
    top_dot = float(min(d["top"] for d in mine) + 3.0) if mine else None
    if len(rows) == 0:
        return None
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 1:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    groups = [g for g in groups if len(g) >= 2]
    if top_dot is not None:
        groups = [g for g in groups if g[-1] + 0.0 >= top_dot - 5.5]                 # drop glyph strokes of the label that lie above the highest dot
    if not groups:
        return None
    return float(groups[0][0] + 1.85), float(groups[-1][-1] - 1.85), top_dot, len(groups)


res = dict(panels={}, bars=[])
fits = {}
for k in PAN:
    fits[k], r = calib(k)
    res["panels"][k] = dict(grid_fit_max_residual_px=r, px_per_unit=float(-fits[k][0]))
    print(f"panel {k}: {-fits[k][0]:.1f} px per unit, fit residual {r:.2f} px")
val = lambda k, row: (row - fits[k][1]) / fits[k][0]
row_of = lambda k, v: float(np.polyval(fits[k], v)) + SHIFT


def exp_nse(n):
    return S[n]["l1"]["nse_median"], RG[n]["l1"]["by_seed"]["nse_median"] if RG[n]["l1"]["n_seeds"] > 1 else None


def exp_on(n):
    return S[n]["l1"]["onset"]["pod"], RG[n]["l1"]["by_seed"]["onset_pod"] if RG[n]["l1"]["n_seeds"] > 1 else None


def exp_own(n):
    o = UN[n]["own_level"]["l1"]
    return o["onset"]["pod"], o["by_seed"]["onset_pod"] if RG[n]["l1"]["n_seeds"] > 1 else None


PLAN = {"TL": (exp_nse, [("lstm_wq", GREY), ("lstm63_wq", BLUE), ("lstm63eq_wq", BLUE2), ("lstm_w", GREY), ("lstm63_w", BLUE)]),
        "TR": (exp_on, [("lstm_wq", GREY), ("lstm63_wq", BLUE), ("lstm63eq_wq", BLUE2), ("lstm_w", GREY), ("lstm63_w", BLUE)]),
        "BL": (exp_nse, [("lobo_w_static", GREY), ("unseen50_w_static", ORANGE), ("lstm_w", None), ("lobo_w_nostatic", GREY), ("unseen50_w_nostatic", ORANGE), ("lobo_ref_w_nostatic", None)]),
        "BR": (exp_own, [("lobo_w_static", GREY), ("unseen50_w_static", ORANGE), ("lstm_w", None), ("lobo_w_nostatic", GREY), ("unseen50_w_nostatic", ORANGE), ("lobo_ref_w_nostatic", None)])}
worst_bar, worst_seed, n_bars, n_seed = 0.0, 0.0, 0, 0
all_ok = True
for k, (fexp, plan) in PLAN.items():
    found = []
    for col in (GREY, BLUE, BLUE2, ORANGE):
        found += [dict(b, color=tuple(col)) for b in bars_of(k, col)]
    found += [dict(b, color=None) for b in outline_bars(k)]
    found = sorted(found, key=lambda b: b["x"])
    if len(found) != len(plan):
        print(f"{k}: found {len(found)} bars, expected {len(plan)}")
        all_ok = False
        continue
    print(f"\npanel {k}")
    for b, (name, col) in zip(found, plan):
        v, seeds = fexp(name)
        want_col = None if col is None else tuple(col)
        EDGE = 3.6 if col is not None else 0.0        # a filled bar has a light edge line (1.4 pt = 4.3 px, centred on the border: 2.1 px inside) plus the blended pixel that the colour test leaves out
        top_px = b["top"] - EDGE
        plotted = val(k, top_px)
        tol_px = 1.6
        dev_px = abs(top_px - row_of(k, v))
        ok = (b["color"] == want_col) and dev_px <= tol_px
        n_bars += 1
        worst_bar = max(worst_bar, dev_px)
        line = f"  {name:22s} bar top: plotted {plotted:8.4f}  JSON {v:8.4f}  ({dev_px:4.1f} px)"
        rec = dict(panel=k, set=name, json_value=v, plotted_value=plotted, deviation_px=dev_px, colour_as_expected=b["color"] == want_col)
        if seeds:
            ext = seed_extent(k, b["x"], b["top"], b["bottom"])
            lo, hi = min(seeds), max(seeds)
            e_top, e_bot = row_of(k, hi), row_of(k, lo)
            if ext is None:
                ok, line = False, line + "  seed marks NOT FOUND"
            else:
                d1, d2 = abs(ext[0] - e_top), abs(ext[1] - e_bot)
                d3 = abs(ext[2] - e_top) if ext[2] is not None else float("nan")
                n_seed += 1
                worst_seed = max(worst_seed, d1, d2)
                ok &= d1 <= 2.5 and d2 <= 2.5 and (d3 != d3 or d3 <= 3.5)
                line += f" | seeds {min(seeds):.4f}-{max(seeds):.4f}: caps {val(k, ext[0]):.4f} / {val(k, ext[1]):.4f} ({d1:.1f} / {d2:.1f} px), highest dot {('%.4f' % val(k, ext[2])) if ext[2] is not None else 'n/a'} ({d3:.1f} px)"
                rec.update(seed_range_json=[lo, hi], seed_caps_px=[ext[0], ext[1]], seed_highest_dot_px=ext[2], seed_deviation_px=[d1, d2, d3])
        else:
            line += " | no seed marks (one training)"
        rec["ok"] = bool(ok)
        all_ok &= ok
        res["bars"].append(rec)
        print(line + ("" if ok else "   <-- MISMATCH"))
res["n_bars_checked"], res["n_seed_ranges_checked"] = n_bars, n_seed
res["max_bar_top_deviation_px"], res["max_seed_extent_deviation_px"] = worst_bar, worst_seed
res["selftest_shift_px"] = SHIFT
res["all_ok"] = bool(all_ok)
print(f"\n{n_bars} bars and {n_seed} seed ranges read from the pixels; largest deviation {worst_bar:.1f} px (bar tops) and {worst_seed:.1f} px (seed extents); one pixel is "
      f"{1 / (-fits['TL'][0]) * 1000:.2f}e-3 NSE (top left), {100 / (-fits['TR'][0]):.3f} points (top right)")
print("ALL BARS AND SEED RANGES AGREE WITH THE JSON:", all_ok, "(self-test shift", SHIFT, "px)")
save("m11_figure" + ("_selftest" if SHIFT else ""), res)
sys.exit(0 if all_ok else 1)
