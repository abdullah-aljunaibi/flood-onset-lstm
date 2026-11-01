"""AORC review, check (h) part 2: figures/aorc_rain_to_midnight.png (two stacked panels, one family: the models with the river gauge). Every marker is located on the PNG pixels (colour and shape), the axes are
calibrated on the drawn grid lines (rows lost under a line are placed by the spacing of the others), and the plotted values are compared with the ladder of the results file at lead 1: 7 filled rungs, 2 hollow
circles (the one-day-older controls of the NLDAS and ERA5 rungs), 2 diamonds and 2 triangles (the table-window and UTC-day controls of the ERA5 and AORC rungs); the seed bars are checked by their outer extent
(the connecting line of the family, which can touch a marker's column, is masked); the legend markers are read too. The text of the picture (title, rung labels, caption lines) was read by eye and is listed in the
output of the check as the strings I compare it with. R2G_SELFTEST=6 shifts the expected bar ends by 6 px (the check must then fail). About 5 seconds."""
import sys; sys.dont_write_bytecode = True
from PIL import Image
from scipy import ndimage as ndi
from a_common import *

R = jload("results/aorc_rain_to_midnight.json")
im = np.asarray(Image.open(f"{PROJ}/figures/aorc_rain_to_midnight.png").convert("RGB")).astype(int)
H, W, _ = im.shape
print(f"figure {W} x {H} px")
GRID = np.array([int("e1e0d9"[i:i + 2], 16) for i in (0, 2, 4)])


def grid_rows(x0, x1, y0, y1):
    d = np.abs(im[y0:y1, x0:x1] - GRID).sum(2)
    rows = np.flatnonzero((d < 14).mean(1) > 0.6) + y0
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 2: cur.append(r)
        else: groups.append(cur); cur = [r]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def calibrate(rows, top_value, step):
    """value of each found grid row from the spacing (a row hidden under a line leaves a gap that is a whole number of steps)."""
    pitch = float(np.median(np.diff(rows)))
    k = np.round((np.array(rows) - rows[0]) / pitch)
    vals = top_value - step * k
    f = np.polyfit(vals, rows, 1)
    return f, float(np.abs(np.polyval(f, vals) - rows).max()), pitch


gt, gb = grid_rows(200, 900, 100, 700), grid_rows(200, 900, 780, 1360)
ft, res_t, pitch_t = calibrate(gt, 0.70, 0.10); fb, res_b, pitch_b = calibrate(gb, 0.90, 0.10)
v_top = lambda y: (y - ft[1]) / ft[0]; v_bot = lambda y: (y - fb[1]) / fb[0]
print("grid rows top:", gt, "bottom:", gb)
print("scale: %.2f px per percentage point (top), %.2f px per 0.01 NSE (bottom); fit residual max %.2f / %.2f px" % (abs(ft[0]) / 100, abs(fb[0]) / 100, res_t, res_b))


def comps(mask):
    lab, n = ndi.label(mask); out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i; h, w = m.shape; area = int(m.sum())
        if area < 40: continue
        out.append(dict(x=sl[1].start + w / 2, y=sl[0].start + h / 2, w=w, h=h, area=area, fill=area / (w * h)))
    return out


blue = comps((im[..., 2] - im[..., 0]) > 100)


def classify(c):
    if 17 <= c["w"] <= 21 and 17 <= c["h"] <= 21 and c["fill"] > 0.75: return "filled"
    if 24 <= c["w"] <= 28 and 24 <= c["h"] <= 28 and c["fill"] < 0.6: return "hollow_circle"
    if 35 <= c["w"] <= 41 and 35 <= c["h"] <= 41 and c["fill"] < 0.4: return "diamond"
    if 29 <= c["w"] <= 32 and 28 <= c["h"] <= 32 and c["fill"] < 0.5: return "triangle"
    return "other"
for c in blue: c["kind"] = classify(c)
PANEL = lambda c: "top" if 100 < c["y"] < 720 else "bottom" if 780 < c["y"] < 1360 else "legend"
for c in blue: c["panel"] = PANEL(c)
cnt = lambda kind, p: sorted([c for c in blue if c["kind"] == kind and c["panel"] == p], key=lambda c: c["x"])
found = {p: {k: len(cnt(k, p)) for k in ("filled", "hollow_circle", "diamond", "triangle")} for p in ("top", "bottom", "legend")}
print("found per panel (filled / hollow circle / diamond / triangle):", found)
assert found["top"] == found["bottom"] == dict(filled=7, hollow_circle=2, diamond=2, triangle=2), found

L = R["ladder"]["wq"]["rungs"]
rungs = {r["rung"]: r for r in L}
order = [r["rung"] for r in L]                                          # the rungs in the order of the ladder = the order of the x axis
expected = []                                                          # (kind, x-order index, set, onset, nse, onset_range, nse_range, treat)
for i, r in enumerate(L):
    e = r["by_lead"]["l1"]
    expected.append(("filled", i, r["set"], e["onset_warned"], e["nse_median"], e["onset_seed_range"], e["nse_seed_range"], True))
for rr, ctl, kind in ((L[1], "lstm_wqn_older", "hollow_circle"), (L[2], "lstm_wqh_older", "hollow_circle"), (L[2], "lstm_wqh_to09", "diamond"), (L[3], "lstm_wqa_to09", "diamond"), (L[2], "lstm_wqh_utc", "triangle"), (L[3], "lstm_wqa_utc", "triangle")):
    e = rr["controls"][ctl]["by_lead"]["l1"]
    expected.append((kind, None, ctl, e["onset_warned"], e["nse_median"], e["onset_seed_range"], e["nse_seed_range"], False))
rows_out, maxdev = [], {"top": 0.0, "bottom": 0.0}
for kind in ("filled", "hollow_circle", "diamond", "triangle"):
    ex = [e for e in expected if e[0] == kind]
    # the order along x: filled by rung; hollow circles NLDAS then ERA5; diamonds ERA5 then AORC; triangles ERA5 then AORC (the order of the expected list)
    for p, conv, idx in (("top", v_top, 3), ("bottom", v_bot, 4)):
        got = cnt(kind, p)
        assert len(got) == len(ex)
        for c, e in zip(got, ex):
            val = conv(c["y"]); d = val - e[idx]
            rows_out.append(dict(panel=p, kind=kind, set=e[2], plotted=float(val), json=float(e[idx]), diff=float(d), x=float(c["x"])))
            maxdev[p] = max(maxdev[p], abs(d))
print("\nplotted value (read from the pixels) against the ladder of the JSON, lead 1:")
for r in rows_out:
    sc = 100 if r["panel"] == "top" else 1
    print(f"  {r['panel']:6s} {r['kind']:13s} {r['set']:16s} plotted {r['plotted'] * sc:8.3f}  JSON {r['json'] * sc:8.3f}  diff {r['diff'] * sc:+.3f}")
print(f"\n{len(rows_out)} markers read; max deviation {100 * maxdev['top']:.3f} percentage points (onsets panel) and {maxdev['bottom']:.4f} NSE (accuracy panel)")
out = {"markers": rows_out, "max_dev_top_points": 100 * maxdev["top"], "max_dev_bottom_nse": maxdev["bottom"], "n_markers_checked": len(rows_out), "grid": dict(top=gt, bottom=gb, px_per_point=abs(ft[0]) / 100, px_per_0_01_nse=abs(fb[0]) / 100)}

# ---- seed bars: outer extent in the marker's pixel column --------------------------------------------------------------------------------------------------------------------
def hue_rows(x, y0, y1):
    band = im[y0:y1, int(x) - 1:int(x) + 2]
    return np.flatnonzero(((band[..., 2] - band[..., 0]) > 60).any(1)) + y0
def intervals(rowsv, gap=4):
    out_ = []
    for r in rowsv:
        if out_ and r - out_[-1][1] <= gap + 1: out_[-1][1] = r
        else: out_.append([r, r])
    return out_
SHIFT = float(os.environ.get("R2G_SELFTEST", "0"))
bar_rows, all_ok = [], True
for p, conv_f, (y0, y1), idx_rng in (("top", ft, (100, 700), 5), ("bottom", fb, (780, 1360), 6)):
    row_of = lambda v: float(np.polyval(conv_f, v))
    fills = [(c["x"], c["y"]) for c in cnt("filled", p)]
    line_y = lambda x: float(np.interp(x, [a for a, _ in fills], [b for _, b in fills])) if len(fills) > 1 and fills[0][0] <= x <= fills[-1][0] else None
    marks = []
    for kind in ("filled", "hollow_circle", "diamond", "triangle"):
        ex = [e for e in expected if e[0] == kind]
        for c, e in zip(cnt(kind, p), ex): marks.append((c, e))
    for c, e in marks:
        lo, hi = e[idx_rng]; treat = e[7]
        half = c["h"] / 2; top_marker, bot_marker = c["y"] - half, c["y"] + half
        cap = 2.2 if treat else 0.0
        exp_top = min(row_of(hi) - cap, top_marker) + SHIFT; exp_bot = max(row_of(lo) + cap, bot_marker) + SHIFT
        found_rows = hue_rows(c["x"], y0, y1); ly = line_y(c["x"])
        if ly is not None:
            found_rows = np.array([r_ for r_ in found_rows if not (abs(r_ - ly) <= 3.6 and (r_ < top_marker - 1 or r_ > bot_marker + 1))], dtype=int)
        obs = intervals(found_rows)
        near = [o for o in obs if o[1] >= exp_top - 6 and o[0] <= exp_bot + 6]
        near = [o for o in near if (o[1] - o[0] + 1) > 6 or (o[0] >= top_marker - 6 and o[1] <= bot_marker + 6)]
        if not near: print("  no pixels", p, e[2]); all_ok = False; continue
        o_top, o_bot = min(o[0] for o in near), max(o[1] for o in near)
        tol = 3.5 if treat or c["kind"] in ("filled", "hollow_circle") else 5.0
        ok = abs(o_top - exp_top) <= tol and abs(o_bot - exp_bot) <= tol
        all_ok &= ok
        bar_rows.append(dict(panel=p, set=e[2], kind=c["kind"], observed_rows=[o_top, o_bot], expected_rows=[exp_top, exp_bot], ok=bool(ok)))
        if not ok: print(f"  CHECK {p} {e[2]} {c['kind']}: observed {o_top}..{o_bot}, expected {exp_top:.1f}..{exp_bot:.1f}")
out["bars"] = bar_rows; out["bars_all_ok"] = bool(all_ok)
dmax = max(max(abs(b["observed_rows"][0] - b["expected_rows"][0]), abs(b["observed_rows"][1] - b["expected_rows"][1])) for b in bar_rows)
print(f"seed bars (outer extent, {len(bar_rows)} bars): all consistent with the JSON seed ranges: {all_ok}; largest deviation {dmax:.1f} px" + (f"   [SELF-TEST: expected ends moved by {SHIFT:g} px; {sum(1 for b_ in bar_rows if not b_['ok'])} of {len(bar_rows)} bars flagged]" if SHIFT else ""))
leg = [c for c in blue if c["panel"] == "legend" and c["kind"] != "other"]
print("legend markers found:", sorted([(c["kind"], round(c["x"]), round(c["y"])) for c in leg], key=lambda t: (t[2], t[1])))
out["legend_markers"] = sorted(c["kind"] for c in leg)
out["labels_to_compare"] = dict(figure_labels=R["figure_labels"], rung_labels=order, note="title, panel titles, the seven rung labels with their end hours, the legend text and the five caption lines were read on the picture")
jsave("a12_figure.json", out)
