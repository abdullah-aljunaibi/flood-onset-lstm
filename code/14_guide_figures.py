"""Four concept pictures for the step-by-step guide (Project_Guide.pdf), for readers who have never seen a neural network.

Each picture is meant to be understood in ten seconds: few words, lettering of at least 9 points at the 6.7 inches of the text
width, at most 3.4 inches high. Nothing is trained and no score of the report is computed again.

  guide_pipeline.png   'From the weather to a warning'. Boxes from left to right: what goes in, the model, what comes out, the last
                       step. Under them ONE real forecast of the test years, picked by rule: among the onsets that the LSTM with the
                       river gauge warned of 1 day ahead (the river at or below its high-flow level on the day the forecast was made,
                       the forecast for the next day above it, and the river did rise above it), the one with the most rain on the
                       day the forecast was made. Drawn: rain and river flow of the 28 days up to that day, and the three forecasts
                       made on it. The case shows the path, not how often a warning comes in time (the JSON holds that share).
  guide_nse.png        'What the score NSE means'. One real stretch of the example river (French Broad) three times: with a perfect
                       forecast (NSE 1), with the stored forecast 1 day ahead of the LSTM with the river gauge, and with 'always the
                       average of these days' (NSE 0). The three scores are computed here with flood.nse and are the scores of the
                       stretch, not of the report. The stretch is picked by rule: 5 days before to 15 days after an onset of the
                       test years, no second onset inside, and of those the one whose NSE is closest to the median NSE of the 12
                       basins over all test years (results/lstm_all.json) - a forecast about as good as the model's usual score.
  guide_lstm_cell.png  'The memory of an LSTM, day by day'. One memory cell as a store of water with three taps. The logic is that of
                       torch.nn.LSTM as flood.LSTMNet uses it: memory today = keep x memory yesterday + add x today's news; what is
                       passed on = pass x the memory (squashed), a copy that leaves the memory as it is; the three taps and the news
                       are computed from today's inputs and from what all cells passed on the day before; the forecast for the next
                       three days is a weighted sum of what all cells pass on.
  guide_gradient.png   'How training lowers the error'. An illustration, not data of the project: gradient descent on a made-up
                       valley with three learning rates, the same number of steps from the same start.

Reads results/lstm_all.json (memory cells, basin properties, median NSE, high-flow levels, onset counts), results/preds/lstm_wq.npz
(the stored forecasts) and the cached daily data (data/derived). Writes the four PNGs to figures/ and results/guide_figures.json:
every number shown in a picture with its source, the two rules and what they chose, a caption for each picture, and notes.
The lettering of every picture is checked (smallest size, overlaps, lettering that leaves the picture). Run time: a few seconds.
Smoke test: FLOOD_QUICK=1 python code/14_guide_figures.py  reads and writes only results/quick and figures/quick; if an input is
missing there (or does not fit) it says so in one line and ends without an error.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator

import flood as F
from common import *

use_style()
t_start = time.time()
SCRIPT = "14_guide_figures.py"
W, DPI, MAX_H = 6.7, 220, 3.4                              # as wide as the other figures of the project; a guide picture is at most this high (inches)
FS, FOOT, TITLE, HEAD, MIN_PT = 9.2, 9.0, 10.6, 9.8, 9.0   # lettering in points at that width: labels, footnote, title, panel titles; the smallest allowed
X0, LS, CAP = 0.08, 1.25, 0.72                             # left edge of title and footnote (inches); line spacing of the footnote; height of a capital letter / font size
SET, LEAD = "lstm_wq", 1                                   # the stored forecasts shown: the all-river LSTM with the river gauge; the NSE picture shows 1 day ahead
RIVER = "03439000"                                         # the example river of the project (French Broad: 03_lstm.py, build_overview.py)
PAST_DAYS = 28                                             # days of rain and river flow shown under the boxes, up to the day the forecast is made
BEFORE, AFTER = 5, 15                                      # a stretch for the NSE picture: days before and after an onset (three weeks in all)
START, STEPS = -1.55, 6                                    # the made-up valley of the last picture: where the ball starts, and how many steps it takes
RATES = (("too small", 0.07), ("about right", 0.5), ("too large", 2.06))      # its three learning rates (the valley is error = weight squared / 2)
STATE = {"NC": "North Carolina", "GA": "Georgia", "FL": "Florida", "MS": "Mississippi", "VA": "Virginia"}
NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
CARD, CARD_BLUE, DARK, PIPE, WATER = "#f5f4f0", "#eaf2fc", SEQ[5], SEQ[2], SEQ[0]       # box fills; taps; pipes; the water in the store
DAY = pd.Timedelta(days=1)
RES_DIR = "results/quick" if F.QUICK else "results"
FIG_DIR = "figures/quick" if F.QUICK else "figures"
NUMBERS = []                                               # every number shown in a picture: where, how it is shown, its value, its source

word = lambda n: NUMBER.get(n, str(n))
date = lambda d: f"{d.day} {d:%B %Y}"
span = lambda a, b: (f"{a.day} to {date(b)}" if (a.year, a.month) == (b.year, b.month) else f"{a.day} {a:%B} to {date(b)}" if a.year == b.year else f"{date(a)} to {date(b)}")
n2 = lambda x: f"{round(x, 2) + 0:.2f}"                    # a score as the documents print it (never '-0.00')
river = lambda g: F.SHORT[g].rsplit(", ", 1)[0] + ", " + STATE.get(F.SHORT[g].rsplit(", ", 1)[1], F.SHORT[g].rsplit(", ", 1)[1])


def shown(picture, as_, value, source):
    """Record one number that a picture shows (as lettering or as a mark) with the place it comes from."""
    NUMBERS.append(dict(picture=picture, shown_as=as_, value=value, source=source))


def stop(msg):
    """An input is missing or does not fit: in a smoke test say so in one line and end without an error; otherwise end with the error."""
    if F.QUICK:
        print(f"{SCRIPT}: {msg} - smoke test skipped, nothing written")
        sys.exit(0)
    sys.exit(f"{SCRIPT}: {msg}")


# ---- the inputs, and the two real cases picked by rule -------------------------------------------------------------------------
def read():
    """The results file, the stored forecasts and the daily data, checked against each other."""
    need = [F.RES_OUT / "lstm_all.json", F.PRED / f"{SET}.npz"] + [F.DERIVED / "camels12_daymet.pkl"] * F.QUICK      # a smoke test does not rebuild the data cache
    gone = [str(f.relative_to(ROOT)) for f in need if not f.exists()]
    if gone:
        stop(f"not found: {', '.join(gone)} (run 04_lstm_all.py first)")
    A = json.loads((F.RES_OUT / "lstm_all.json").read_text())
    if bool(A.get("quick_mode")) != F.QUICK:
        stop(f"{RES_DIR}/lstm_all.json is {'not ' * F.QUICK}a smoke-test file")
    D, P = F.load_data(), F.load_preds(SET)
    thr = F.thresholds(D)
    try:
        facts = dict(cells=int(A["config"]["hidden"]), properties=len(A["static_inputs"]), history_days=int(A["config"]["warmup"]), block_days=int(A["config"]["block"]),
                     nse_median=float(A["sets"]["wq"]["test"]["summary"][f"l{LEAD}"]["nse_median"]), onset=A["sets"]["wq"]["test"]["summary"][f"l{LEAD}"]["onset"],
                     nse_river=float(A["sets"]["wq"]["test"]["basins"][RIVER][f"l{LEAD}"]["nse"]), weights=sorted({int(c["n_params"]) for c in A["sets"]["wq"]["cost"].values()}),
                     seeds=len(A["sets"]["wq"]["seeds"]), n_basins=len(A["basins"]), test=A["split"]["test"], stored_levels=A["thresholds"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        stop(f"{RES_DIR}/lstm_all.json does not hold what the pictures need ({type(e).__name__}: {e})")
    ok = (set(P) == set(D) and all(np.allclose(P[g].obs.values, D[g].q_mm.values) for g in P)                       # the forecasts belong to these data
          and all(abs(thr[g] - facts["stored_levels"][g]) < 1e-9 for g in P))                                         # and the high-flow levels are the stored ones
    if not ok:
        stop(f"{RES_DIR}/preds/{SET}.npz, the daily data and the high-flow levels of {RES_DIR}/lstm_all.json do not fit together")
    return facts, D, P, thr


def onsets(P, D, thr, t0, t1):
    """Every onset of the test years (as flood.flood_scores counts them, 1 day ahead): the day the forecast was made, whether it warned, the rain of that day."""
    rows = []
    for g in sorted(P):
        p = P[g]
        q, low = p.loc[t0:t1], (p.obs.shift(1) <= thr[g]).loc[t0:t1]                                # low: the river was at or below the level when the forecast was made
        for t in q.index[((q.obs > thr[g]) & low & q.l1.notna()).values]:
            rows.append(dict(gauge_id=g, issue=t - DAY, warned=bool(q.l1[t] > thr[g]), rain_mm=float(D[g].prcp[t - DAY]),
                             complete=bool(t + (len(F.LEADS) - 1) * DAY <= t1 and all(np.isfinite(p[f"l{k}"][t + (k - 1) * DAY]) for k in F.LEADS))))
    return pd.DataFrame(rows)


def pick_example(P, D, thr, facts):
    """The forecast under the boxes of the first picture: the warned onset of the test years with the most rain on the day the forecast was made."""
    t0, t1 = (pd.Timestamp(x) for x in facts["test"])
    O = onsets(P, D, thr, t0, t1)
    pool = O[O.warned & O.complete]
    fallback = pool.empty                                   # no onset was warned (a smoke test): then the onset with the most rain, warned or not
    pick = (O[O.complete] if fallback else pool).sort_values(["rain_mm", "gauge_id", "issue"], ascending=[False, True, True]).iloc[0]
    g, t = pick.gauge_id, pick.issue
    days = pd.date_range(t - (PAST_DAYS - 1) * DAY, t)
    fc = [float(P[g][f"l{k}"][t + k * DAY]) for k in F.LEADS]                                       # what the model said on day t about days t+1, t+2, t+3
    return dict(gauge_id=g, river=river(g), short=F.SHORT[g], issue_day=str(t.date()), days=[str(d.date()) for d in days], rain_mm=[float(v) for v in D[g].prcp[days]],
                flow_mm_day=[float(v) for v in D[g].q_mm[days]], forecast_days=[str((t + k * DAY).date()) for k in F.LEADS], forecast_mm_day=fc,
                high_flow_level_mm_day=float(thr[g]), warning=[bool(v > thr[g]) for v in fc], observed_later_mm_day=[float(P[g].obs[t + k * DAY]) for k in F.LEADS],
                rule=("no onset was warned: the test-year onset with the most rain on the day the forecast was made" if fallback else
                      "of the test-year onsets warned 1 day ahead (river at or below its high-flow level when the forecast was made, forecast for the next day above it, "
                      "and the river did rise above it), the one with the most rain on the day the forecast was made"),
                rule_fallback=bool(fallback), n_onsets=int(len(O)), n_onsets_warned=int(O.warned.sum()), n_candidates=int(len(O[O.complete] if fallback else pool)))


def pick_stretch(P, thr, facts):
    """The stretch of the NSE picture: around one onset of the example river, with an NSE close to the model's usual score."""
    t0, t1 = (pd.Timestamp(x) for x in facts["test"])
    p, col = P[RIVER], f"l{LEAD}"
    hi = p.obs > thr[RIVER]
    on = hi.index[(hi & ~hi.shift(1, fill_value=False)).values]                                     # first days above the level
    rows = []
    for t in on[(on >= t0) & (on <= t1)]:
        a, b = t - BEFORE * DAY, t + AFTER * DAY
        if a >= t0 and b <= t1 and ((on >= a) & (on <= b)).sum() == 1 and p.loc[a:b, col].notna().all():
            rows.append(dict(onset=t, start=a, end=b, nse=F.nse(p.obs[a:b], p.loc[a:b, col])))
    if not rows:
        stop(f"no stretch of {BEFORE + AFTER + 1} days with a single onset in the test years of basin {RIVER}")
    C = pd.DataFrame(rows)
    C["distance"] = (C.nse - facts["nse_median"]).abs()
    pick = C.sort_values(["distance", "onset"]).iloc[0]
    w = p.loc[pick.start:pick.end]
    obs, fc = w.obs.values.astype(float), w[col].values.astype(float)
    mean = float(obs.mean())
    score = dict(perfect=F.nse(obs, obs), forecast=F.nse(obs, fc), average=F.nse(obs, np.full(len(obs), mean)))
    assert abs(score["perfect"] - 1) < 1e-12 and abs(score["average"]) < 1e-12, "a perfect forecast must score 1 and 'always the average of the scored days' 0"
    return dict(gauge_id=RIVER, river=river(RIVER), short=F.SHORT[RIVER], prediction_set=SET, column=col, start=str(pick.start.date()), end=str(pick.end.date()),
                onset=str(pick.onset.date()), n_days=int(len(w)), days=[str(d.date()) for d in w.index], observed_mm_day=[float(v) for v in obs], forecast_mm_day=[float(v) for v in fc],
                average_mm_day=mean, nse=score, nse_all_test_years_this_river=facts["nse_river"], nse_median_all_test_years=facts["nse_median"],
                share_of_absolute_error=float(np.abs(fc - obs).sum() / np.abs(obs - mean).sum()),
                rule=(f"river: the example river of the project; stretch: {BEFORE} days before to {AFTER} days after an onset of the test years, no second onset inside; of these the one "
                      f"whose NSE {LEAD} day ahead is closest to the median NSE of the {facts['n_basins']} basins over all test years ({RES_DIR}/lstm_all.json: sets.wq.test.summary.l{LEAD}.nse_median)"),
                n_candidates=int(len(C)), candidates_nse=dict(lowest=float(C.nse.min()), median=float(C.nse.median()), highest=float(C.nse.max())))


def descend():
    """Gradient descent on the made-up valley error = weight squared / 2 (its slope is the weight itself): the path of the ball for each learning rate."""
    out = []
    for name, lr in RATES:
        w = [START]
        for _ in range(STEPS):
            w.append(w[-1] - lr * w[-1])                    # one step: against the slope, as far as the learning rate says
        out.append(dict(name=name, learning_rate=lr, weight=[float(v) for v in w], error=[float(0.5 * v * v) for v in w]))
    return out


# ---- drawing: the pieces every picture uses ------------------------------------------------------------------------------------
def figure():
    """An empty figure at the resolution of the PNG, so that lettering measured on it is as wide as it will be drawn."""
    return plt.figure(figsize=(W, 3.3), dpi=DPI)


def extent(fig, text, size, **kw):
    """(width, height) in inches of a text drawn with the figure's own fonts."""
    t = fig.text(0, 0, text, fontsize=size, **kw)
    b = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return b.width / fig.dpi, b.height / fig.dpi


def wrap(fig, text, width, size):
    """Break a text into lines no wider than `width` inches at font size `size`."""
    lines, cur = [], ""
    for w_ in text.split():
        if cur and extent(fig, f"{cur} {w_}", size)[0] > width:
            lines.append(cur)
            cur = w_
        else:
            cur = f"{cur} {w_}".strip()
    return lines + [cur]


def footnote(fig, text):
    """The lines of the footnote and their height in inches."""
    lines = wrap(fig, text, W - 2 * X0, FOOT)
    return lines, extent(fig, "\n".join(lines), FOOT, linespacing=LS)[1]


def frame(fig, H, title, note, sub=None):
    """Bold title top left, an optional line under it, the grey footnote at the bottom - as in the other figures of the project."""
    fig.set_size_inches(W, H)
    fig.text(X0 / W, 1 - 0.08 / H, title, fontsize=TITLE, fontweight="bold", color=INK, ha="left", va="top")
    if sub:
        fig.text(X0 / W, 1 - 0.305 / H, sub, fontsize=FS, color=INK2, ha="left", va="top")
    fig.text(X0 / W, 0.07 / H, "\n".join(note), fontsize=FOOT, color=INK2, ha="left", va="bottom", linespacing=LS)


def canvas(fig, H):
    """Axes over the whole figure with inches as units, for boxes, arrows and pipes."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
    return ax


def panel(fig, H, x, y, w, h):
    """Axes at a place given in inches, without ticks and grid; only the line at the bottom is kept."""
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.patch.set_alpha(0)
    ax.spines["left"].set_visible(False)
    return ax


def row(ax, fig, x, y, parts, size=FS, color=INK2):
    """One line of lettering from x to the right, its baseline at y; parts = [(text, weight), ...] with one space between them."""
    space = extent(fig, "a a", size)[0] - extent(fig, "aa", size)[0]
    for text, weight in parts:
        ax.text(x, y, text, fontsize=size, fontweight=weight, color=INK if weight == "bold" else color, ha="left", va="baseline", zorder=6)
        x += extent(fig, text, size, fontweight=weight)[0] + space
    return x - space


def row_width(fig, parts, size=FS):
    space = extent(fig, "a a", size)[0] - extent(fig, "aa", size)[0]
    return sum(extent(fig, text, size, fontweight=weight)[0] for text, weight in parts) + space * (len(parts) - 1)


def card(ax, x, y, w, h, core=False):
    """A box with round corners; core: the model, outlined in the blue of the LSTM."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", facecolor=CARD_BLUE if core else CARD, edgecolor=BLUE if core else AXIS,
                                linewidth=1.8 if core else 1.0, zorder=2))


def arrow(ax, a, b, color=INK2, lw=1.4, head=9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=head, color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=4))


def lettering(fig):
    """The smallest letter size, and the lettering that overlaps other lettering or leaves the figure (there should be none)."""
    fig.canvas.draw()
    r, (w, h) = fig.canvas.get_renderer(), fig.canvas.get_width_height()
    unseen = {id(t.label1) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for t in axis.get_major_ticks()
              if not min(axis.get_view_interval()) <= t.get_loc() <= max(axis.get_view_interval())}        # tick labels outside the axis range are not drawn
    T = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in unseen]
    box = [t.get_window_extent(r) for t in T]
    name = lambda t: " ".join(t.get_text().split())[:60]
    bad = [f"'{name(a)}' overlaps '{name(c)}'" for i, (a, p) in enumerate(zip(T, box)) for c, q in list(zip(T, box))[i + 1:] if p.overlaps(q)]
    bad += [f"'{name(t)}' leaves the figure" for t, b in zip(T, box) if b.x0 < 0 or b.y0 < 0 or b.x1 > w or b.y1 > h]
    return float(min(t.get_fontsize() for t in T)), bad, sorted({name(t) for t in T})


def save(fig, name, H, txt):
    """Check the lettering, write the PNG (beside its place first, then moved: a document built meanwhile never reads half a picture) and return its facts."""
    smallest, bad, words = lettering(fig)
    out = F.FIG_OUT / name
    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.png")
    fig.savefig(tmp, dpi=DPI)
    tmp.replace(out)
    plt.close(fig)
    return dict(file=f"{FIG_DIR}/{name}", width_in=W, height_in=round(H, 2), dpi=DPI, smallest_lettering_pt=smallest, lettering_problems=bad, title=txt["title"],
                footnote=txt["footnote"], lettering=words)


# ---- picture 1: from the weather to a warning -----------------------------------------------------------------------------------
def words_pipeline(E, facts):
    leads = ["tomorrow"] + [f"in {k} days" for k in F.LEADS[1:]]
    year = "one year" if facts["history_days"] == 365 else f"{facts['history_days']} days"
    low = "the river is still low" if E["flow_mm_day"][-1] <= E["high_flow_level_mm_day"] else "the river is already high"
    return dict(
        title="From the weather to a warning",
        boxes=[dict(head="What goes in", rows=[[("Past weather:", "bold"), (f"{year}, day by day", "normal")], [("rain, temperature, sunlight, humidity", "normal")],
                                               [("River gauge:", "bold"), ("the flow up to today", "normal")],
                                               [("The basin:", "bold"), (f"{word(facts['properties'])} fixed properties", "normal")]]),
               dict(head="The model", rows=[[("a rule found", "normal")], [("from examples", "normal")]], core=True),
               dict(head="What comes out", rows=[[("a", "normal"), ("forecast", "bold"), ("of the", "normal")], [("river flow", "normal")],
                                                 [(", ".join(leads[:-1]) + ",", "normal")], [(leads[-1], "normal")]]),
               dict(head="The last step", rows=[[("a", "normal"), ("warning", "bold"), ("if", "normal")], [("the forecast is", "normal")], [("above the", "normal")],
                                                [("high-flow level", "normal")]])],
        example=[("A real example:", "bold"), (f"{E['river']}. Today is {date(pd.Timestamp(E['issue_day']))}: {E['rain_mm'][-1]:.0f} mm of rain, and {low}.", "normal")],
        labels=dict(rain="rain", gauge="river gauge", level="high-flow level", past=f"{PAST_DAYS // 7} weeks ago", today="today", forecast="forecast", warning="warning",
                    no_warning="no warning", leads=leads),
        footnote=(("SMOKE TEST, not a result. " if F.QUICK else "")
                  + "The example is real: a forecast of the project's model with the river gauge, from the test years. It shows the path from the inputs to a warning, "
                    "not how often a warning comes in time."))


def draw_pipeline(E, txt):
    """Boxes and arrows from left to right; under them the real case E: the past four weeks on the left, the three forecast days on the right."""
    fig = figure()
    note, foot_h = footnote(fig, txt["footnote"])
    B, pad, pitch = txt["boxes"], 0.10, 0.168
    inner = [max([extent(fig, b["head"], HEAD, fontweight="bold")[0]] + [row_width(fig, r) for r in b["rows"]]) for b in B]
    bw = [v + 2 * pad for v in inner]
    gap = (W - 2 * X0 - sum(bw)) / (len(B) - 1)
    assert gap >= 0.17, f"the boxes of the first picture leave {gap:.2f} inches for an arrow: shorten their lettering"
    cap_h = CAP * HEAD / 72
    h_box = 0.10 + cap_h + 0.205 + (max(len(b["rows"]) for b in B) - 1) * pitch + 0.125
    h_flow, h_sep, h_rain = 0.78, 0.05, 0.21
    y_ax = 0.07 + foot_h + 0.09 + 0.17                      # lower edge of the small charts (their day labels are under it)
    y_cap = y_ax + h_flow + h_sep + h_rain + 0.10           # baseline of the line 'A real example'
    y_box = y_cap + 0.19                                    # lower edge of the boxes
    H = y_box + h_box + 0.36
    frame(fig, H, txt["title"], note)
    ax = canvas(fig, H)
    bx = [X0 + sum(bw[:i]) + i * gap for i in range(len(B))]
    for i, b in enumerate(B):
        card(ax, bx[i], y_box, bw[i], h_box, core=b.get("core", False))
        block = cap_h + 0.205 + (len(b["rows"]) - 1) * pitch                                         # from the top of the heading to the last baseline
        top = y_box + h_box - (0.10 if not b.get("core") else (h_box - block - 0.03) / 2)           # the model: its few words in the middle of the box
        x = bx[i] + pad if not b.get("core") else bx[i] + bw[i] / 2
        ax.text(x, top - cap_h, b["head"], fontsize=HEAD, fontweight="bold", color=INK, ha="center" if b.get("core") else "left", va="baseline", zorder=6)
        for j, r in enumerate(b["rows"]):
            y = top - cap_h - 0.205 - j * pitch
            row(ax, fig, x - (row_width(fig, r) / 2 if b.get("core") else 0), y, r)
        if i:
            arrow(ax, (bx[i - 1] + bw[i - 1] + 0.035, y_box + h_box / 2), (bx[i] - 0.035, y_box + h_box / 2))
    row(ax, fig, X0, y_cap, txt["example"])

    # the real case: the last four weeks (rain above, river flow below) and the three forecast days, on one scale of river flow
    L, thr, fc = txt["labels"], E["high_flow_level_mm_day"], E["forecast_mm_day"]
    x, rain, flow = np.arange(-len(E["days"]) + 1, 1), np.array(E["rain_mm"]), np.array(E["flow_mm_day"])
    top = max(1.25 * max(fc + [flow.max()]), 1.5 * thr)
    xl, wl = bx[0] + 0.02, bw[0] - 0.02                     # under 'what goes in'
    xr, wr = bx[2], bx[3] + bw[3] - bx[2]                   # under 'what comes out' and 'the last step'
    a_rain, a_flow = panel(fig, H, xl, y_ax + h_flow + h_sep, wl, h_rain), panel(fig, H, xl, y_ax, wl, h_flow)
    a_fc = panel(fig, H, xr, y_ax, wr, h_flow + h_sep + h_rain)
    a_rain.bar(x, rain, width=0.8, color=SEQ[2], zorder=3)
    a_rain.set_xlim(x[0] - 0.7, 0.7); a_rain.set_ylim(0, max(rain.max() * 1.05, 1))
    a_rain.annotate(L["rain"], (0, 1), xycoords="axes fraction", xytext=(2, -1), textcoords="offset points", ha="left", va="top", fontsize=FS, color=INK2)
    for a, t in ((a_flow, top), (a_fc, top * (h_flow + h_sep + h_rain) / h_flow)):                  # the same inches per unit of river flow in both
        a.axhspan(thr, t, color=RED, alpha=0.07, lw=0, zorder=1)
        a.axhline(thr, color=RED, lw=1.2, ls=(0, (5, 2)), zorder=2)
        a.set_ylim(0, t)
    a_flow.plot(x, flow, color=INK, lw=2, zorder=5)
    a_flow.plot(x[-1], flow[-1], ls="none", marker="o", ms=5, color=INK, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=6, clip_on=False)
    a_flow.set_xlim(x[0] - 0.7, 0.7)
    off = dict(textcoords="offset points", fontsize=FS, color=INK2)
    a_flow.annotate(L["level"], (x[0], thr), xytext=(1, 3), ha="left", va="bottom", **off)
    a_flow.annotate(L["gauge"], (x[0], flow[:10].max()), xytext=(1, 4), ha="left", va="bottom", **off)
    a_flow.annotate(L["past"], (x[0] - 0.7, 0), xycoords=("data", "axes fraction"), xytext=(0, -4), ha="left", va="top", **off)
    a_flow.annotate(L["today"], (0, 0), xycoords=("data", "axes fraction"), xytext=(0, -4), ha="center", va="top", **off)
    k = np.arange(1, len(fc) + 1)
    a_fc.plot(k, fc, color=BLUE, lw=1.5, zorder=4)
    a_fc.plot(k, fc, ls="none", marker="o", ms=7.2, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=5)
    a_fc.set_xlim(0.15, len(fc) + 0.45)
    a_fc.annotate(L["forecast"], (k[0], fc[0]), xytext=(-8, 0), ha="right", va="center", **off)
    for i, v in enumerate(fc):
        if E["warning"][i]:
            a_fc.annotate(L["warning"], (k[i], v), xytext=(0, 7), ha="center", va="bottom", textcoords="offset points", fontsize=FS, color=INK, fontweight="bold")
        else:
            a_fc.annotate(L["no_warning"], (k[i], v), xytext=(0, -7), ha="center", va="top", **off)
        a_fc.annotate(L["leads"][i], (k[i], 0), xycoords=("data", "axes fraction"), xytext=(0, -4), ha="center", va="top", **off)
    arrow(ax, (xl + wl + 0.26, y_ax + 0.36), (xr - 0.12, y_ax + 0.36), color=MUTED, lw=1.2, head=8)
    return save(fig, "guide_pipeline.png", H, txt)


# ---- picture 2: what the score NSE means ----------------------------------------------------------------------------------------
def words_nse(S):
    a, b = pd.Timestamp(S["start"]), pd.Timestamp(S["end"])
    return dict(
        title="What the score NSE means",
        subtitle=f"The same {S['n_days']} days of one river, scored with three forecasts",
        panels=[("A perfect forecast", "no error on any day", "NSE 1"), ("The project's forecast", f"made {LEAD} day ahead", f"NSE {n2(S['nse']['forecast'])}"),
                ("Always the average", "the same value every day", "NSE 0")],
        labels=dict(observed="observed", forecast="forecast", error="error", average="the average", same="forecast = observed,\non every day", y="river flow (mm per day)"),
        footnote=(("SMOKE TEST, not a result. " if F.QUICK else "")
                  + "The NSE compares the errors of a forecast with the errors of 'always the average' (errors are squared, so large ones count most): no error gives 1, "
                    f"errors as large as those of the average give 0. {S['river']}, {date(a)} to {date(b)}. These are the scores of this stretch, not of the report: "
                    "the scores of the report cover all test years."))


def draw_nse(S, txt):
    """Three panels over the same stretch: a perfect forecast, the stored forecast 1 day ahead, always the average; the error is shaded."""
    fig = figure()
    note, foot_h = footnote(fig, txt["footnote"])
    left, right, gap, h_ax = 0.56, 0.08, 0.16, 1.36
    pw = (W - left - right - 2 * gap) / 3
    y_ax = 0.07 + foot_h + 0.12 + 0.19
    H = y_ax + h_ax + 0.44 + 0.55
    frame(fig, H, txt["title"], note, sub=txt["subtitle"])
    obs, fc, mean, L = np.array(S["observed_mm_day"]), np.array(S["forecast_mm_day"]), S["average_mm_day"], txt["labels"]
    days, x = pd.DatetimeIndex(S["days"]), np.arange(S["n_days"])
    top = 1.24 * max(obs.max(), fc.max())
    ip = int(obs.argmax())                                  # the day of the peak
    jf = int(np.clip(ip - 2, 0, None)) + int(fc[max(ip - 2, 0):ip + 4].argmax())                    # the peak of the forecast near it
    ticks = [i for i, d in enumerate(days) if d.day in (1, 15)] or list(x[::14])
    lab = dict(fontsize=FS, color=INK2, zorder=7)
    for j, (name, sub, score) in enumerate(txt["panels"]):
        ax = fig.add_axes([(left + j * (pw + gap)) / W, y_ax / H, pw / W, h_ax / H])
        y_err = (obs[ip] + (fc[ip] if j == 1 else mean)) / 2                                         # a height inside the shaded gap on the day of the peak
        if j == 0:                                          # the forecast lies under the observed flow: a blue rim on both sides of the black line
            ax.plot(x, obs, color=BLUE, lw=4.8, solid_joinstyle="round", zorder=3)
            ax.text(ip + 2.6, 0.60 * top, L["same"], ha="left", va="center", linespacing=1.25, **lab)
        elif j == 1:
            ax.fill_between(x, obs, fc, color=BLUE, alpha=0.22, lw=0, zorder=2)
            ax.plot(x, fc, color=BLUE, lw=1.8, zorder=3)
            ax.annotate(L["observed"], (ip, obs[ip]), xytext=(8, -10), textcoords="offset points", ha="left", va="center", **lab)   # below the score, above the label of the forecast
            ax.annotate(L["forecast"], (jf, fc[jf]), xytext=(9, 0), textcoords="offset points", ha="left", va="center", **lab)
        else:
            ax.fill_between(x, obs, mean, color=MUTED, alpha=0.28, lw=0, zorder=2)
            ax.plot(x, np.full(len(x), mean), color=MUTED, lw=1.8, zorder=3)
            ax.annotate(L["average"], (x[-1], mean), xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", **lab)
        if j:                                               # the word 'error' with a thin line into the shaded gap
            ax.plot([ip - 3.0, ip - 0.05], [y_err, y_err], color=INK2, lw=0.8, solid_capstyle="butt", zorder=6)
            ax.plot(ip - 0.05, y_err, ls="none", marker="o", ms=2.6, color=INK2, zorder=6)
            ax.text(ip - 3.5, y_err, L["error"], ha="right", va="center", **lab)
        ax.plot(x, obs, color=INK, lw=1.9, zorder=5)
        ax.text(0.975, 0.955, score, transform=ax.transAxes, ha="right", va="top", fontsize=13, fontweight="bold", color=INK, zorder=7)
        ax.set_xlim(-0.8, len(x) - 0.2); ax.set_ylim(0, top)
        ax.set_xticks(ticks, [f"{days[i].day} {days[i]:%b}" for i in ticks])
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))
        ax.tick_params(labelsize=MIN_PT, length=0, labelleft=j == 0); ax.tick_params(axis="x", labelcolor=INK2); ax.grid(axis="x", visible=False)
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="left", va="bottom", fontsize=FS, color=INK2)
        ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(0, 18.5), textcoords="offset points", ha="left", va="bottom", fontsize=HEAD, fontweight="bold", color=INK)
        if j == 0:
            ax.set_ylabel(L["y"], fontsize=FS)
            ticks_y = [float(v) for v in ax.get_yticks() if 0 <= v <= top]
    return dict(save(fig, "guide_nse.png", H, txt), y_axis_ticks=ticks_y, x_axis_ticks=[S["days"][i] for i in ticks])


# ---- picture 3: one memory cell of the LSTM ------------------------------------------------------------------------------------
def words_cell(facts):
    return dict(
        title="The memory of an LSTM, day by day",
        subtitle="One memory cell, drawn as a store of water with three taps",
        labels=dict(yesterday="yesterday's\nmemory", today="today's memory", tomorrow="on to tomorrow", news="today's news", set="set the taps",
                    forecast=["forecast for the", f"next {word(len(F.LEADS))} days"],
                    inputs=[[("Today's inputs:", "bold"), ("the day's weather and the river gauge,", "normal")], [("and what the cells passed on yesterday", "normal")]],
                    taps=[("keep how much?", "(forget gate)"), ("add how much?", "(input gate)"), ("pass on how much?", "(output gate)")]),
        footnote=("Each tap is a gate: a number between 0 (shut) and 1 (open) that the network learns to set from the day's inputs. What is passed on is a copy: the memory "
                  f"itself goes on to tomorrow. The network of the project has {facts['cells']} such cells, and the forecast is made from what all of them pass on."))


def pipe(ax, xs, ys, lw=9.0):
    ax.plot(xs, ys, color=PIPE, lw=lw, solid_capstyle="butt", solid_joinstyle="round", zorder=3)


def tip(ax, x, y, dx, dy, size=0.125):
    """The head of a pipe: a triangle with its point at (x, y), pointing along (dx, dy)."""
    ax.add_patch(Polygon([(x, y), (x - dx * 1.5 * size - dy * size, y - dy * 1.5 * size - dx * size), (x - dx * 1.5 * size + dy * size, y - dy * 1.5 * size + dx * size)],
                         closed=True, facecolor=PIPE, edgecolor="none", zorder=3))


def tap(ax, x, y, upright, side):
    """A tap on a pipe: two triangles tip to tip, and a handle on one side (upright pipe: side -1 = left; level pipe: side -1 = below)."""
    s, h, stem, bar = 0.125, 0.10, 0.20, 0.075
    if upright:
        tri = [[(x - h, y - s), (x + h, y - s), (x, y)], [(x - h, y + s), (x + h, y + s), (x, y)]]
        ax.plot([x, x + side * stem], [y, y], color=DARK, lw=1.6, solid_capstyle="butt", zorder=5)
        ax.plot([x + side * stem] * 2, [y - bar, y + bar], color=DARK, lw=2.8, solid_capstyle="round", zorder=5)
    else:
        tri = [[(x - s, y - h), (x - s, y + h), (x, y)], [(x + s, y - h), (x + s, y + h), (x, y)]]
        ax.plot([x, x], [y, y + side * stem], color=DARK, lw=1.6, solid_capstyle="butt", zorder=5)
        ax.plot([x - bar, x + bar], [y + side * stem] * 2, color=DARK, lw=2.8, solid_capstyle="round", zorder=5)
    for t in tri:
        ax.add_patch(Polygon(t, closed=True, facecolor=DARK, edgecolor=SURFACE, linewidth=0.8, zorder=6))


def draw_cell(txt):
    """One memory cell: the memory runs from left to right through the store; three taps; today's inputs enter from below and set the taps."""
    fig = figure()
    note, foot_h = footnote(fig, txt["footnote"])
    L = txt["labels"]
    y_in, h_in = 0.07 + foot_h + 0.13, 0.44                 # the box of today's inputs
    y_top = y_in + h_in                                     # its upper edge: the grey arrows and the pipe of the news start here
    y_p = y_top + 1.20                                      # the pipe of the memory
    H = y_p + 0.53 + 0.14 + 0.50
    frame(fig, H, txt["title"], note, sub=txt["subtitle"])
    ax = canvas(fig, H)
    x_a, x_v1, x_j, x_t0, x_t1, x_out, x_fc, x_end, r_j = 0.12, 1.66, 2.74, 3.54, 4.94, 4.50, 5.02, W - 0.12, 0.105
    y_v2, y_v3, y_el = y_top + 0.62, y_top + 0.50, y_in + 0.25
    bold = dict(fontsize=9.6, fontweight="bold", color=INK, zorder=7)
    plain = dict(fontsize=FS, color=INK2, zorder=7)

    # the memory: from yesterday through the first tap, the joint where the news is added, the store, and on to tomorrow
    pipe(ax, [x_a, x_j - r_j - 0.19], [y_p, y_p]); tip(ax, x_j - r_j - 0.02, y_p, 1, 0)
    pipe(ax, [x_j + r_j, x_t0 - 0.17], [y_p, y_p]); tip(ax, x_t0, y_p, 1, 0)
    pipe(ax, [x_t1, x_end - 0.17], [y_p, y_p]); tip(ax, x_end, y_p, 1, 0)
    tap(ax, x_v1, y_p, upright=False, side=-1)
    ax.add_patch(Circle((x_j, y_p), r_j, facecolor=SURFACE, edgecolor=DARK, linewidth=1.4, zorder=6))
    ax.plot([x_j - 0.052, x_j + 0.052], [y_p, y_p], color=DARK, lw=1.6, zorder=7); ax.plot([x_j, x_j], [y_p - 0.052, y_p + 0.052], color=DARK, lw=1.6, zorder=7)
    y_t0, y_t1, y_w = y_p - 0.30, y_p + 0.36, y_p + 0.18                                            # the store: bottom, top, water level
    ax.add_patch(Rectangle((x_t0, y_t0), x_t1 - x_t0, y_w - y_t0, facecolor=WATER, edgecolor="none", zorder=2))
    ax.plot([x_t0, x_t1], [y_w, y_w], color=PIPE, lw=1.4, zorder=3)
    ax.plot([x_t0, x_t0, x_t1, x_t1], [y_t1, y_t0, y_t0, y_t1], color=DARK, lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=4)
    ax.text((x_t0 + x_t1) / 2, y_p - 0.02, L["today"], ha="center", va="center", **bold)
    ax.text(x_a, y_p + 0.15, L["yesterday"], ha="left", va="bottom", linespacing=1.2, **bold)
    ax.text(x_end, y_p + 0.15, L["tomorrow"], ha="right", va="bottom", **plain)

    # today's news: up from the inputs through the second tap into the joint
    pipe(ax, [x_j, x_j], [y_top, y_p - r_j - 0.19]); tip(ax, x_j, y_p - r_j - 0.02, 0, 1)
    tap(ax, x_j, y_v2, upright=True, side=-1)
    ax.text(x_j + 0.20, y_top + 0.17, L["news"], ha="left", va="center", **plain)

    # what is passed on: a thinner line (a copy) down from the store through the third tap to the forecast
    pipe(ax, [x_out, x_out, x_fc - 0.13], [y_t0, y_el, y_el], lw=4.5); tip(ax, x_fc, y_el, 1, 0, size=0.085)
    tap(ax, x_out, y_v3, upright=True, side=-1)
    card(ax, x_fc, y_el - 0.25, x_end - x_fc, 0.50)
    for i, line in enumerate(L["forecast"]):
        ax.text((x_fc + x_end) / 2, y_el + 0.085 - i * 0.17, line, ha="center", va="center", fontsize=FS, color=INK, zorder=7)

    # today's inputs: a box at the bottom; thin grey arrows from it to the handles of the three taps
    x_b0, x_b1 = 0.10, 4.20
    card(ax, x_b0, y_in, x_b1 - x_b0, h_in)
    for i, r in enumerate(L["inputs"]):
        row(ax, fig, x_b0 + 0.10, y_top - 0.165 - i * 0.168, r)
    grey = dict(color=MUTED, lw=1.1, head=7)
    arrow(ax, (x_v1, y_top), (x_v1, y_p - 0.20 - 0.035), **grey)
    for x_from, x_v, y_v in ((x_j - 0.50, x_j, y_v2), (x_out - 0.48, x_out, y_v3)):
        ax.plot([x_from, x_from], [y_top, y_v], color=MUTED, lw=1.1, solid_capstyle="butt", zorder=4)
        arrow(ax, (x_from, y_v), (x_v - 0.20 - 0.035, y_v), **grey)
    ax.text(x_v1 - 0.10, y_top + 0.52, L["set"], ha="right", va="center", **plain)

    # the three taps: the plain question, and under it the name the textbooks use
    (a1, b1), (a2, b2), (a3, b3) = L["taps"]
    ax.text(x_v1, y_p + 0.33, a1, ha="center", va="bottom", **bold); ax.text(x_v1, y_p + 0.15, b1, ha="center", va="bottom", **plain)
    ax.text(x_j + 0.20, y_v2 + 0.09, a2, ha="left", va="center", **bold); ax.text(x_j + 0.20, y_v2 - 0.085, b2, ha="left", va="center", **plain)
    ax.text(x_out + 0.20, y_v3 + 0.09, a3, ha="left", va="center", **bold); ax.text(x_out + 0.20, y_v3 - 0.085, b3, ha="left", va="center", **plain)
    return save(fig, "guide_lstm_cell.png", H, txt)


# ---- picture 4: how training lowers the error -----------------------------------------------------------------------------------
def words_gradient():
    what = {"too small": "still far from the bottom", "about right": "reaches the bottom", "too large": "jumps from side to side"}
    return dict(
        title="How training lowers the error",
        panels=[(f"Learning rate {name}", what[name]) for name, _ in RATES],
        labels=dict(x="one weight", y="error (loss)", start="start"),
        footnote=(f"An illustration with a made-up curve, not data of the project: {word(STEPS)} steps in each panel from the same start (hollow dot: the start; dark dot: "
                  f"after {word(STEPS)} steps). A real network has thousands of weights and does this for all of them at once."))


def draw_gradient(G, txt):
    """Three panels with the same valley; a ball takes the same number of steps downhill with three learning rates."""
    fig = figure()
    note, foot_h = footnote(fig, txt["footnote"])
    left, right, gap, h_ax = 0.40, 0.08, 0.20, 1.42
    pw = (W - left - right - 2 * gap) / 3
    y_ax = 0.07 + foot_h + 0.10 + 0.21
    H = y_ax + h_ax + 0.44 + 0.34
    frame(fig, H, txt["title"], note)
    L, w = txt["labels"], np.linspace(-2.5, 2.5, 201)
    xlim, ylim = (-2.62, 2.62), (-0.22, 3.3)
    inch = lambda a, b: float(np.hypot((b[0] - a[0]) * pw / (xlim[1] - xlim[0]), (b[1] - a[1]) * h_ax / (ylim[1] - ylim[0])))        # length of a step on paper
    for j, (g, (name, sub)) in enumerate(zip(G, txt["panels"])):
        ax = fig.add_axes([(left + j * (pw + gap)) / W, y_ax / H, pw / W, h_ax / H])
        ax.plot(w, 0.5 * w * w, color=INK2, lw=2, zorder=2)
        pts = list(zip(g["weight"], g["error"]))
        for a, b in zip(pts, pts[1:]):                      # a step: an arrow where it is long enough to carry a head, a plain line otherwise
            if inch(a, b) > 0.17:
                ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1, mutation_scale=8, shrinkA=3.6, shrinkB=4.2), zorder=4)
            else:
                ax.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=1.1, zorder=4)
        ax.plot(*zip(*pts[1:-1]), ls="none", marker="o", ms=5.6, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=5)
        ax.plot(*pts[0], ls="none", marker="o", ms=6.2, markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.6, zorder=6)
        ax.plot(*pts[-1], ls="none", marker="o", ms=7.4, color=DARK, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=7)
        ax.annotate(L["start"], pts[0], xytext=(-5, -5), textcoords="offset points", ha="right", va="top", fontsize=FS, color=INK2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel(L["x"], fontsize=FS, labelpad=4)
        ax.annotate(sub, (0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="left", va="bottom", fontsize=FS, color=INK2)
        ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(0, 18.5), textcoords="offset points", ha="left", va="bottom", fontsize=HEAD, fontweight="bold", color=INK)
        if j == 0:
            ax.set_ylabel(L["y"], fontsize=FS, labelpad=5)
    return save(fig, "guide_gradient.png", H, txt)


# ---- main ----------------------------------------------------------------------------------------------------------------------
def run():
    facts, D, P, thr = read()
    E, S, G = pick_example(P, D, thr, facts), pick_stretch(P, thr, facts), descend()
    hit, n_on = facts["onset"]["hits"], facts["onset"]["hits"] + facts["onset"]["misses"]
    if not F.QUICK:                                         # the onsets counted here are those of the results file
        assert (E["n_onsets"], E["n_onsets_warned"]) == (n_on, hit), f"{E['n_onsets_warned']} of {E['n_onsets']} onsets warned here, {hit} of {n_on} in {RES_DIR}/lstm_all.json"
    T = dict(pipeline=words_pipeline(E, facts), nse=words_nse(S), cell=words_cell(facts), gradient=words_gradient())
    pictures = dict(guide_pipeline=draw_pipeline(E, T["pipeline"]), guide_nse=draw_nse(S, T["nse"]), guide_lstm_cell=draw_cell(T["cell"]), guide_gradient=draw_gradient(G, T["gradient"]))
    return facts, E, S, G, pictures


if __name__ == "__main__":
    facts, E, S, G, pictures = run()
    for name, p in pictures.items():
        print(f"{p['file']}: {p['width_in']} x {p['height_in']} in, smallest lettering {p['smallest_lettering_pt']:g} pt" + ("" if not p["lettering_problems"] else "  LETTERING TO CHECK: " + "; ".join(p["lettering_problems"])))
    notes = [
        "guide_pipeline.png shows ONE real forecast of the LSTM with the river gauge (prediction set lstm_wq), picked by the rule under "
        "example_of_the_pipeline.rule: it shows the path from the inputs to a warning, not how often a warning comes in time "
        f"({E['n_onsets_warned']} of the {E['n_onsets']} onsets of the test years were warned 1 day ahead).",
        "guide_nse.png: the three scores are those of the days drawn, computed here with flood.nse; they are not scores of the report. "
        f"Over all test years this river scores {S['nse_all_test_years_this_river']:.2f} and the median of the {facts['n_basins']} basins is "
        f"{S['nse_median_all_test_years']:.2f}. 'Always the average' is the average of the days drawn, so its score is exactly 0.",
        f"The stretch of guide_nse.png was picked by the rule under stretch_of_the_score_picture.rule from {S['n_candidates']} stretches around an onset; their "
        f"scores run from {S['candidates_nse']['lowest']:.2f} to {S['candidates_nse']['highest']:.2f} (median {S['candidates_nse']['median']:.2f}), so a "
        "forecast around an onset usually scores lower than the one drawn.",
        "guide_lstm_cell.png is a drawing of the logic of one memory cell (torch.nn.LSTM as flood.LSTMNet uses it), not of data; "
        f"the network of the project has {facts['cells']} such cells.",
        "guide_gradient.png is an illustration with a made-up curve and made-up learning rates: no number in it is a result of the project.",
    ]
    out = dict(what="Four concept pictures of the step-by-step guide (Project_Guide.pdf): every number shown in a picture, the two rules that picked the "
                    "real examples and what they chose, and the lettering check of each picture.",
               quick_mode=F.QUICK, pictures=pictures, facts=facts, example_of_the_pipeline=E, stretch_of_the_score_picture=S,
               steps_of_the_training_picture=G, notes=(["SMOKE TEST: tiny budgets upstream, the numbers are not results."] if F.QUICK else []) + notes,
               sources=dict(facts=f"{RES_DIR}/lstm_all.json", forecasts=f"{RES_DIR}/preds/{SET}.npz", daily_data="data/derived/camels12_daymet.pkl"),
               runtime_s=round(time.time() - t_start, 1))
    target = F.RES_OUT / "guide_figures.json"
    tmp = target.with_name(f".{target.stem}.{os.getpid()}.tmp.json")
    tmp.write_text(json.dumps(out, indent=1, default=float))
    tmp.replace(target)
    print(f"wrote {RES_DIR}/guide_figures.json and four pictures in {FIG_DIR}/ in {out['runtime_s']:.0f} s")
