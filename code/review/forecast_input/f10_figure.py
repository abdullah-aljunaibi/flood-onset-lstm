"""E2 review, check (h): figures/forecast_input.png read on its pixels against the ladder of results/forecast_input.json (2 rows x 3 columns: onsets warned above, median NSE below; 1, 2, 3 days ahead; five blue rungs of the
main split, four orange rungs of the satellite-era split, hollow controls, a diamond for the forecast on the plain gauge model). The axes are calibrated on the drawn grid lines (rows) and on the ends of the bottom spine
(columns: x = -0.6 at the left spine, 9.52 at the right end); every expected marker (72: 5 blue and 4 orange filled circles, 1 blue and 1 orange hollow circle, 1 blue diamond in each of the 6 panels) is looked for
in the pixels (disk or ring coverage in a window around the position the JSON predicts, the centre of the best coverage) and its centre is turned back into a value; the vertical extent of each seed bar is read in the pixel
column of its marker and compared with the single-seed range. R2G_SELFTEST=3 moves every expected value by 3 units of the panel (points / 0.03 NSE): the check must then fail. The text of the picture (title, legend, panel titles,
tick labels, the caption) was read by eye and is listed in the output. About 5 seconds."""
import sys; sys.dont_write_bytecode = True
from PIL import Image
from f_common import *

R = jload("results/forecast_input.json")
im = np.asarray(Image.open(f"{PROJ}/figures/forecast_input.png").convert("RGB")).astype(int)
H, W, _ = im.shape
print(f"figure {W} x {H} px")
hexc = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)])
GRID, AXIS, BLUE, ORANGE, SURF = hexc("#e1e0d9"), hexc("#c3c2b7"), hexc("#2a78d6"), hexc("#eb6834"), hexc("#fcfcfb")
near = lambda c, tol: (np.abs(im - c).sum(2) < tol)
SELF = float(os.environ.get("R2G_SELFTEST", "0"))

# ---- calibration -------------------------------------------------------------------------------------------------------------------------------------------------------------
axis_cols = [171.5, 1005.5, 1838.5]                                      # left spines of the three columns (x = -0.6)
a_mask = near(AXIS, 8)
bottom_rows = [r for r in range(H) if (a_mask[r, 300:700].mean() > 0.9)]                       # long horizontal lines of the AXIS colour inside the first panel columns
rows_axis = sorted(set(int(np.round(np.mean(g))) for g in np.split(bottom_rows, np.flatnonzero(np.diff(bottom_rows) > 3) + 1))) if bottom_rows else []
print("horizontal axis lines at rows", rows_axis)
y0_top, y0_bot = 788.5, 1423.5                                           # y of the 0% and the NSE 0 lines (the two bottom spines found above, two pixels thick)
def find_spines():
    rows = []
    for r in range(H):
        if a_mask[r, 180:940].mean() > 0.95: rows.append(r)
    grp = np.split(rows, np.flatnonzero(np.diff(rows) > 3) + 1)
    return [float(np.mean(g)) for g in grp]
sp = find_spines()
print("rows of the AXIS colour spanning the first panel:", sp)
y0_top, y0_bot = sp[0], sp[-1]
# grid rows in a clean strip right of the left spine (no markers there)
def grid_rows(c):
    x0 = int(axis_cols[c]) + 10
    strip = near(GRID, 8)[:, x0:x0 + 25].mean(1) > 0.8
    r = np.flatnonzero(strip)
    return [float(np.mean(g)) for g in np.split(r, np.flatnonzero(np.diff(r) > 2) + 1)] if len(r) else []
cal = {}
for c in range(3):
    gr = grid_rows(c)
    top = [y for y in gr if y < y0_top - 20 and y > 200 and y < 790]
    bot = [y for y in gr if y > 800 and y < y0_bot - 20]
    cal[c] = dict(top_rows=top, bottom_rows=bot)
    print(f"column {c}: grid rows in the onset panel {[round(y, 1) for y in top]}, in the NSE panel {[round(y, 1) for y in bot]}")
# onset panel: rows are the 80, 60, 40, 20 % lines (and 0 at the spine); NSE panel: 1.0 (top), 0.8, 0.6, 0.4, 0.2 lines and 0 at the spine
def fit(values, rows):
    return np.polyfit(values, rows, 1)
FIT = {}
for c in range(3):
    t = cal[c]["top_rows"]; b = cal[c]["bottom_rows"]
    ft = fit([0.8, 0.6, 0.4, 0.2][:len(t)] if len(t) == 4 else [0.8, 0.6, 0.4, 0.2], (t + [])[:4]) if len(t) == 4 else None
    fb = fit([1.0, 0.8, 0.6, 0.4, 0.2][:len(b)], b) if len(b) == 5 else (fit([0.8, 0.6, 0.4, 0.2], b[-4:]) if len(b) >= 4 else None)
    FIT[c] = (ft, fb)
    assert ft is not None and fb is not None, f"column {c}: grid rows not found {cal[c]}"
    print(f"   column {c}: onset panel {abs(ft[0]) / 100:.3f} px per point, 0% at row {ft[1]:.1f} (spine {y0_top:.1f}); NSE panel {abs(fb[0]) / 100:.3f} px per 0.01 NSE, 0 at row {fb[1]:.1f} (spine {y0_bot:.1f})")
# x: the left spine is x = -0.6 and the bottom spine ends at 9.52 (the right limit of the axes)
right_end = []
for c in range(3):
    r = int(round(y0_top)); x0 = int(axis_cols[c]) - 3
    xs = np.flatnonzero(a_mask[r, x0:x0 + 900]); run = np.split(xs, np.flatnonzero(np.diff(xs) > 3) + 1)[0]          # the first contiguous run: this panel's bottom spine
    right_end.append(float(run[-1] + x0 + 1))
u0 = (right_end[0] - axis_cols[0]) / (9.52 + 0.6)
xs_unit = [u0, u0, u0]                                                    # the three columns of one figure have the same width (a regular grid of axes); the spine of the other two is interrupted by a label, so the width is read on the first
print(f"x scale: the bottom spine of column 0 runs from {axis_cols[0]} to {right_end[0]} -> {u0:.3f} px per unit; the left spines of the columns are {axis_cols[1] - axis_cols[0]:.1f} and {axis_cols[2] - axis_cols[1]:.1f} px apart (equal spacing of a regular grid)")
xpix = lambda c, x: axis_cols[c] + (x + 0.6) * xs_unit[c]
ypix_top = lambda c, v: float(np.polyval(FIT[c][0], v)); ypix_bot = lambda c, v: float(np.polyval(FIT[c][1], v))
val_top = lambda c, y: float((y - FIT[c][0][1]) / FIT[c][0][0]); val_bot = lambda c, y: float((y - FIT[c][1][1]) / FIT[c][1][0])

# ---- masks and templates ---------------------------------------------------------------------------------------------------------------------------------------------------
M = {"blue": near(BLUE, 45), "orange": near(ORANGE, 45)}
yy, xx = np.mgrid[-14:15, -14:15]; rr = np.hypot(yy, xx)
DISK = (rr <= 8).astype(float); RING = ((rr >= 7.5) & (rr <= 10)).astype(float); INNER = (rr <= 4).astype(float); SMALL = (rr <= 5).astype(float)
def locate(mask, px, py, kind):
    """best centre near (px, py): filled circle = coverage of a disk of radius 8; hollow circle = coverage of the ring minus the centre; diamond = coverage of a disk of radius 5"""
    best, pos = -1.0, None
    cand = []
    for dy in range(-16, 17):
        for dx in range(-16, 17):
            cy, cx = int(round(py)) + dy, int(round(px)) + dx
            if cy < 15 or cx < 15 or cy > H - 16 or cx > W - 16: continue
            win = mask[cy - 14:cy + 15, cx - 14:cx + 15]
            if kind == "filled": sc = (win * DISK).sum() / DISK.sum()
            elif kind == "hollow": sc = (win * RING).sum() / RING.sum() - 0.7 * (win * INNER).sum() / INNER.sum()
            else: sc = (win * SMALL).sum() / SMALL.sum()
            cand.append((sc, cy, cx))
    mx = max(c[0] for c in cand)
    top = [c for c in cand if c[0] >= mx - 1e-9]
    return mx, float(np.mean([c[1] for c in top])), float(np.mean([c[2] for c in top]))

# ---- the expected markers -------------------------------------------------------------------------------------------------------------------------------------------------------
main, sat = R["ladder"]["main"]["rungs"], R["ladder"]["sat"]["rungs"]; extra = R["ladder"]["main"]["extras"][0]
X_MAIN = list(range(len(main))); X_SAT = [len(main) + 0.9 + i for i in range(len(sat))]
OFF = dict(model=-0.13, control=0.13, extra=0.36)
expected = []        # (column, row 0 onsets / 1 nse, kind, color name, x data, value, (lo, hi) or None, lead key, label)
for c, k in enumerate((1, 2, 3)):
    key = f"l{k}"
    for row, (vk, rk) in enumerate((("onset_warned", "onset_seed_range"), ("nse_median", "nse_seed_range"))):
        for xs_, rungs, color in ((X_MAIN, main, "blue"), (X_SAT, sat, "orange")):
            for x0, r in zip(xs_, rungs):
                e = r["by_lead"][key]; x = x0 + (OFF["model"] if r["control_set"] else 0.0)
                expected.append((c, row, "filled", color, x, e[vk], e[rk], key, r["set"]))
                if r["control_set"]:
                    expected.append((c, row, "hollow", color, x0 + OFF["control"], e["control_" + vk], e["control_" + rk], key, r["control_set"]))
        e = extra["by_lead"][key]
        expected.append((c, row, "diamond", "blue", X_MAIN[2] + OFF["extra"], e[vk], e[rk], key, extra["set"]))
n_exp = len(expected)
print(f"{n_exp} markers expected (6 panels x 12)")
res, maxdev = [], {0: 0.0, 1: 0.0}; bad = []
for (c, row, kind, color, x, v, rg, key, label) in expected:
    px = xpix(c, x); py = ypix_top(c, v) if row == 0 else ypix_bot(c, v)
    sc, cy, cx = locate(M[color], px, py, kind)
    got = val_top(c, cy) if row == 0 else val_bot(c, cy)
    d = got - (v + SELF / 100)                                             # in the unit of the panel (share / NSE), against the JSON's value (the self-test moves it by SELF hundredths: every marker must then be flagged)
    maxdev[row] = max(maxdev[row], abs(d))
    ok = sc > (0.9 if kind == "filled" else 0.45 if kind == "hollow" else 0.9) and abs(cx - px) < 5 and abs(d) < (0.01 if row == 0 else 0.008)
    res.append(dict(column=c, panel="onsets" if row == 0 else "nse", kind=kind, set=label, lead=key, plotted=got, json=v, diff=d, coverage=sc, dx_px=cx - px, ok=bool(ok)))
    if not ok: bad.append(res[-1])
print(f"markers found with the expected shape within the window: {sum(r['ok'] for r in res)} of {n_exp}; largest deviation of a plotted value from the JSON: {100 * maxdev[0]:.2f} points of onsets warned, {maxdev[1]:.4f} NSE" + (f"   [SELF-TEST: every expected position moved by {SELF:g} units; {len(bad)} markers flagged]" if SELF else ""))
if bad and not SELF:
    for b in bad[:6]: print("   CHECK", b)
# ---- seed bars ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
BL = {k: (0.55 * BLUE + 0.45 * SURF) for k in ("blue",)}; BL["orange"] = 0.55 * ORANGE + 0.45 * SURF
def bar_rows(c, row, x, color, alpha_full, py_lo, py_hi):
    px = int(round(xpix(c, x))); m = (near(BLUE if color == "blue" else ORANGE, 45) if alpha_full else near(BL[color], 40))
    col = m[:, px - 1:px + 2].any(1)
    ys = np.flatnonzero(col)
    return ys
bars, bar_bad = [], []
for (c, row, kind, color, x, v, rg, key, label) in expected:
    if not rg: continue
    lo, hi = rg
    f_ = ypix_top if row == 0 else ypix_bot
    ylo, yhi = f_(c, lo), f_(c, hi)                                      # pixel rows: yhi < ylo (higher value = smaller row)
    marker_r = 11.0
    alpha_full = (kind == "filled")
    px = int(round(xpix(c, x)))
    full = near(BLUE if color == "blue" else ORANGE, 45)
    m = full if alpha_full else (full | near(BL[color], 45))                  # the light bars (controls and the diamond: alpha 0.55) beside the full-colour marker
    col = m[:, px - 1:px + 2].any(1)
    # the bar is a run of rows in this pixel column that contains the marker's row range; take the connected run around the marker centre
    cy = int(round(f_(c, v)))
    if abs(yhi - ylo) < 2 * marker_r + 6:
        continue                                                          # the bar is not longer than the marker (and its white ring): nothing to read
    ys = np.flatnonzero(col); runs_ = np.split(ys, np.flatnonzero(np.diff(ys) > 6) + 1)           # the white ring around a marker (1.2 pt = 3 px) separates the marker from its bar
    run = [r_ for r_ in runs_ if len(r_) and r_[0] - 3 <= cy <= r_[-1] + 3]
    if not run: bar_bad.append((c, row, label, key, "no bar pixels")); continue
    top_obs, bot_obs = float(run[0][0]), float(run[0][-1])
    exp_top, exp_bot = min(yhi, cy - marker_r) + SELF * 6.0, max(ylo, cy + marker_r) + SELF * 6.0           # the self-test moves the expected rows by SELF x 6 px
    ok = abs(top_obs - exp_top) <= 5 and abs(bot_obs - exp_bot) <= 5
    bars.append(dict(column=c, panel="onsets" if row == 0 else "nse", set=label, lead=key, observed_rows=[top_obs, bot_obs], expected_rows=[float(exp_top), float(exp_bot)], ok=bool(ok)))
    if not ok: bar_bad.append((c, row, label, key, top_obs, bot_obs, exp_top, exp_bot))
dmax = max([max(abs(b["observed_rows"][0] - b["expected_rows"][0]), abs(b["observed_rows"][1] - b["expected_rows"][1])) for b in bars] or [0])
print(f"seed bars readable (longer than the marker): {len(bars)}; consistent with the JSON seed ranges: {all(b['ok'] for b in bars)}; largest deviation {dmax:.1f} px; not consistent: {len(bar_bad)}")
for b in bar_bad[:5]: print("   CHECK bar", b)
TEXT = dict(title="A real weather forecast (GEFS control member, rain) as an input: onsets warned and accuracy, 1, 2 and 3 days ahead",
            legend=["model, main split: trained on observed rain, run on the forecast", "model, satellite-era split: retrained with the forecast (own scores)", "control: the same forecasts, shuffled", "forecast on the plain gauge model (no rain to midnight)", "lowest to highest of 3 trainings"],
            panels=["1 day ahead", "2 days ahead", "3 days ahead"], group_labels=["main split, test years 2005-2014", "own split, test 2010-2014 (not comparable)"], go_line="GO asks for 70% warned (and at most 30% false)",
            ticks_main=["rain table", "+ rain to midnight", "+ forecast (substituted)", "perfect rain, 3 days", "perfect weather"], ticks_sat=["rain table", "+ rain to midnight", "+ forecast (retrained)", "perfect rain, 3 days"],
            caption=["Filled: forecast averaged over the trainings; the number beside it is that value. Hollow: the control (same forecasts, shuffled: the right climate, no information about the day). Every model has the river gauge and the 12 basins' properties.",
                     "Left (blue): the usual test years 2005-2014. 'GEFS forecast (substituted)': the network is trained on observed rain of the days t+1..t+3 and run on the bias-corrected forecast; 'perfect rain': the AORC rain of those days; 'perfect weather': an upper bound that cannot be had in practice.",
                     "Right (orange, shaded): the shorter satellite-era split (train 2000-2008, test 2010-2014), retrained with the forecast as an input: its scores are its own and are not comparable with the left. The line at 70% is what the GO verdict asks of onsets warned (with at most 30% false).",
                     "One member only (no ensemble); 0.25 degree cells over 113 to 1,928 km2 basins; in Eastern summer the last files of the run are up to 15 minutes late at the issue time. AORC rain to midnight is an analysis made after the fact (hours before 2002 are disaggregated daily totals)."])
out = dict(markers_expected=n_exp, markers_ok=int(sum(r["ok"] for r in res)), max_dev_onsets_points=100 * maxdev[0], max_dev_nse=maxdev[1], bars_readable=len(bars), bars_all_ok=bool(all(b["ok"] for b in bars)), bars_largest_deviation_px=dmax, selftest=SELF,
           calibration=dict(left_spines=axis_cols, px_per_unit_x=xs_unit, rows_top={c: cal[c]["top_rows"] for c in cal}, rows_bottom={c: cal[c]["bottom_rows"] for c in cal}), text_read_by_eye=TEXT, markers=res, bars=bars)
if not SELF: jsave("f10_figure.json", out)
