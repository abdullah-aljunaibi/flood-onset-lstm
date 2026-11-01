"""ROUND 2, check (g): figures/rain_to_midnight.png (two stacked panels). Every marker is located on the PNG pixels (colour and shape), the axes are calibrated on the drawn grid lines,
and the plotted values are compared with the ladder of the JSON; seed bars are checked by their outer extent (the connecting line of a family, which can touch a marker's column, is masked: round 3);
the legend entries are read from the pixels too. R2G_SELFTEST=6 shifts the expected bar ends by 6 px to show that the check can fail."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from PIL import Image
from scipy import ndimage as ndi
from r2_common import *

R = jload("results/rain_to_midnight.json")
im = np.asarray(Image.open(f"{PROJ}/figures/rain_to_midnight.png").convert("RGB")).astype(int)
H, W, _ = im.shape
hx = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])
GRID = hx("#e1e0d9")

def grid_rows(x0, x1, y0, y1):
    d = np.abs(im[y0:y1, x0:x1] - GRID).sum(2)
    rows = np.flatnonzero((d < 14).mean(1) > 0.6) + y0
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 2: cur.append(r)
        else: groups.append(cur); cur = [r]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]

gt, gb = grid_rows(170, 740, 100, 720), grid_rows(170, 740, 760, 1370)
print("grid rows top:", gt, "bottom:", gb)
ft = np.polyfit([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][:len(gt)], gt, 1); fb = np.polyfit([0.9, 0.8, 0.7, 0.6, 0.5, 0.4][:len(gb)], gb, 1)
v_top = lambda y: (y - ft[1]) / ft[0]; v_bot = lambda y: (y - fb[1]) / fb[0]
print("scale: %.2f px per percentage point (top), %.2f px per 0.01 NSE (bottom); fit residual max %.2f / %.2f px" % (abs(ft[0]) / 100, abs(fb[0]) / 100, np.abs(np.polyval(ft, [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1][:len(gt)]) - gt).max(), np.abs(np.polyval(fb, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4][:len(gb)]) - gb).max()))

def comps(mask):
    lab, n = ndi.label(mask); out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i; h, w = m.shape; area = int(m.sum())
        if area < 40: continue
        out.append(dict(x=sl[1].start + w / 2, y=sl[0].start + h / 2, w=w, h=h, area=area, fill=area / (w * h)))        # bbox centre
    return out
blue, orange = comps((im[..., 2] - im[..., 0]) > 100), comps((im[..., 0] - im[..., 2]) > 100)

def classify(c):
    if 17 <= c["w"] <= 21 and 17 <= c["h"] <= 21 and c["fill"] > 0.75: return "filled"
    if 24 <= c["w"] <= 28 and 24 <= c["h"] <= 28 and c["fill"] < 0.7: return "hollow_circle_or_square"
    if 34 <= c["w"] <= 40 and 34 <= c["h"] <= 40 and c["fill"] < 0.4: return "diamond"
    if 27 <= c["w"] <= 31 and 27 <= c["h"] <= 32 and c["fill"] < 0.5: return "triangle"
    return "other"
for c in blue + orange: c["kind"] = classify(c)
PANEL = lambda c: "top" if 100 < c["y"] < 720 else "bottom" if 760 < c["y"] < 1370 else "legend"
for c in blue + orange: c["panel"] = PANEL(c)
cnt = lambda L, kind, p: sorted([c for c in L if c["kind"] == kind and c["panel"] == p], key=lambda c: c["x"])
print("found per panel (blue filled / hollow circle / diamond / triangle; orange filled / hollow square):",
      {p: (len(cnt(blue, "filled", p)), len(cnt(blue, "hollow_circle_or_square", p)), len(cnt(blue, "diamond", p)), len(cnt(blue, "triangle", p)), len(cnt(orange, "filled", p)), len(cnt(orange, "hollow_circle_or_square", p))) for p in ("top", "bottom")})

# the ladder of the JSON, in the order of the rungs on the x axis
RUNG = ["rain table (Daymet)", "NLDAS rain", "rain to midnight (ERA5)", "NLDAS rain + rain to midnight", "perfect weather forecast"]
lad = R["ladder"]
def fam_vals(fam, kind):
    key = "onset_warned" if kind == "top" else "nse_median"
    rungs = sorted(lad[fam]["rungs"], key=lambda r: RUNG.index(r["rung"]))
    filled = [r["by_lead"]["l1"][key] for r in rungs]
    ctrl = [r["by_lead"]["l1"]["control_" + key] for r in rungs if r["control_set"]]
    return rungs, filled, ctrl
out = {"markers": {}}
maxdev = {"top": 0, "bottom": 0}
rows = []
for p, conv, scale, unit in (("top", v_top, 100, "pts"), ("bottom", v_bot, 1, "NSE")):
    for fam, color_list, hollow_kind in (("wq", blue, "hollow_circle_or_square"), ("w", orange, "hollow_circle_or_square")):
        rungs, filled, ctrl = fam_vals(fam, p)
        got_f = [conv(c["y"]) for c in cnt(color_list, "filled", p)]
        got_h = [conv(c["y"]) for c in cnt(color_list, hollow_kind, p)]
        assert len(got_f) == len(filled), (p, fam, len(got_f), len(filled))
        assert len(got_h) == len(ctrl), (p, fam, len(got_h), len(ctrl))
        for r, g_, e_ in zip(rungs, got_f, filled):
            rows.append((p, fam, "filled", r["set"], g_, e_)); maxdev[p] = max(maxdev[p], abs(g_ - e_))
        ctrl_sets = [r["control_set"] for r in rungs if r["control_set"]]
        for s_, g_, e_ in zip(ctrl_sets, got_h, ctrl):
            rows.append((p, fam, "control", s_, g_, e_)); maxdev[p] = max(maxdev[p], abs(g_ - e_))
    # the two matched-window controls (with the gauge)
    mc = [r for r in lad["wq"]["rungs"] if r.get("matched_controls")][0]["matched_controls"]
    key = "onset_warned" if p == "top" else "nse_median"
    for shape, name in (("diamond", "lstm_wqh_to09"), ("triangle", "lstm_wqh_utc")):
        c = cnt(blue, shape, p); assert len(c) == 1
        g_, e_ = conv(c[0]["y"]), mc[name]["l1"][key]
        rows.append((p, "wq", shape, name, g_, e_)); maxdev[p] = max(maxdev[p], abs(g_ - e_))
print("\nplotted value (read from the pixels) against the ladder of the JSON, lead 1:")
for p, fam, kind, name, g_, e_ in rows:
    sc = 100 if p == "top" else 1
    print(f"  {p:6s} {fam:3s} {kind:8s} {name:16s} plotted {g_*sc:8.3f}  JSON {e_*sc:8.3f}  diff {(g_-e_)*sc:+.3f}")
out["markers"] = [dict(panel=p, family=fam, kind=kind, set=name, plotted=g_, json=e_) for p, fam, kind, name, g_, e_ in rows]
out["n_markers_checked"] = len(rows)
out["max_dev_top_points"] = 100 * maxdev["top"]; out["max_dev_bottom_nse"] = maxdev["bottom"]
print(f"\n{len(rows)} markers read; max deviation {100*maxdev['top']:.3f} percentage points (onsets panel) and {maxdev['bottom']:.4f} NSE (accuracy panel)")

# ---- seed bars: outer extent in the marker's pixel column ----
def hue_rows(is_blue, x, y0, y1):
    band = im[y0:y1, int(x) - 1:int(x) + 2]
    r, b = band[..., 0], band[..., 2]
    return np.flatnonzero((((b - r) > 60) if is_blue else ((r - b) > 60)).any(1)) + y0
def intervals(rowsv, gap=4):
    out_ = []
    for r in rowsv:
        if out_ and r - out_[-1][1] <= gap + 1: out_[-1][1] = r
        else: out_.append([r, r])
    return out_
bar_rows = []
all_ok = True
SHIFT = float(os.environ.get("R2G_SELFTEST", "0"))        # self-test of the check: with R2G_SELFTEST=6 every expected bar end is moved by 6 px and the check must fail
for p, conv_f, (y0, y1) in (("top", ft, (130, 700)), ("bottom", fb, (790, 1360))):
    row_of = lambda v: float(np.polyval(conv_f, v))
    for fam, is_blue, L in (("wq", True, blue), ("w", False, orange)):
        key = "onset" if p == "top" else "nse"
        rungs = sorted(lad[fam]["rungs"], key=lambda r: RUNG.index(r["rung"]))
        marks = []
        fills = [(c["x"], c["y"]) for c in cnt(L, "filled", p)]                                    # the family's connecting line runs through the centres of its filled markers
        line_y = lambda x: float(np.interp(x, [a for a, _ in fills], [b for _, b in fills])) if len(fills) > 1 and fills[0][0] <= x <= fills[-1][0] else None
        for c, r in zip(cnt(L, "filled", p), rungs):
            e = r["by_lead"]["l1"]; marks.append((c, e[("onset_warned" if p == "top" else "nse_median")], e[key + "_seed_range"], True, r["set"]))
        ctrl_rungs = [r for r in rungs if r["control_set"]]
        for c, r in zip(cnt(L, "hollow_circle_or_square", p), ctrl_rungs):
            e = r["by_lead"]["l1"]; marks.append((c, e["control_" + ("onset_warned" if p == "top" else "nse_median")], e["control_" + key + "_seed_range"], False, r["control_set"]))
        if fam == "wq":
            mc = [r for r in lad["wq"]["rungs"] if r.get("matched_controls")][0]["matched_controls"]
            for shape, name in (("diamond", "lstm_wqh_to09"), ("triangle", "lstm_wqh_utc")):
                e = mc[name]["l1"]; marks.append((cnt(L, shape, p)[0], e["onset_warned" if p == "top" else "nse_median"], e[key + "_seed_range"], False, name))
        for c, mark, rg, treat, name in marks:
            lo, hi = rg
            half = c["h"] / 2
            top_marker, bot_marker = c["y"] - half, c["y"] + half
            cap = 2.2 if treat else 0.0
            exp_top = min(row_of(hi) - cap, top_marker) + SHIFT; exp_bot = max(row_of(lo) + cap, bot_marker) + SHIFT
            found = hue_rows(is_blue, c["x"], y0, y1); ly = line_y(c["x"])
            if ly is not None:                                                                       # the family line crossing the column just outside the marker is not part of the bar
                found = np.array([r_ for r_ in found if not (abs(r_ - ly) <= 3.6 and (r_ < top_marker - 1 or r_ > bot_marker + 1))], dtype=int)
            obs = intervals(found)
            near = [o for o in obs if o[1] >= exp_top - 6 and o[0] <= exp_bot + 6]               # intervals that belong to this marker (the family line crossing the column is thin and elsewhere)
            near = [o for o in near if (o[1] - o[0] + 1) > 6 or (o[0] >= top_marker - 6 and o[1] <= bot_marker + 6)]
            if not near: print("  no pixels", p, fam, name); all_ok = False; continue
            o_top, o_bot = min(o[0] for o in near), max(o[1] for o in near)
            tol = 3.5 if treat or c["kind"] in ("filled", "hollow_circle_or_square") else 5.0     # the miter of the triangle and the diamond tips need a little more room
            ok = abs(o_top - exp_top) <= tol and abs(o_bot - exp_bot) <= tol
            all_ok &= ok
            bar_rows.append(dict(panel=p, family=fam, set=name, kind=c["kind"], observed_rows=[o_top, o_bot], expected_rows=[exp_top, exp_bot], ok=bool(ok)))
            if not ok: print(f"  CHECK {p} {fam} {name} {c['kind']}: observed {o_top}..{o_bot}, expected {exp_top:.1f}..{exp_bot:.1f}")
out["bars"] = bar_rows; out["bars_all_ok"] = bool(all_ok)
print(f"seed bars (outer extent, {len(bar_rows)} bars): all consistent with the JSON seed ranges: {all_ok}" + (f"   [SELF-TEST: expected ends moved by {SHIFT:g} px; {sum(1 for b_ in bar_rows if not b_['ok'])} of {len(bar_rows)} bars flagged]" if SHIFT else ""))

# ---- the legend, read on the pixels: marker kinds and positions ----
leg = [c for c in blue + orange if c["panel"] == "legend"]
print("legend markers found:", sorted([(c["kind"], "blue" if c in blue else "orange", round(c["x"]), round(c["y"])) for c in leg], key=lambda t: (t[2], t[3])))
out["legend_markers"] = [(c["kind"], "blue" if c in blue else "orange") for c in leg]
jsave("r2g_figure.json", out)
