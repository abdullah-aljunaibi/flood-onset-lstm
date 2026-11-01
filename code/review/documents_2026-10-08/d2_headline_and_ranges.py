"""d2: the headline scores and the resampling ranges of the documents, recomputed with the critic's own code from the stored forecasts (results/preds/*.npz).

Part A, the scores. From the stored forecasts (basin x day x lead arrays by target day) the critic's own functions give, for 14 prediction sets, the median NSE of the 12 basins and the onset and all-day warning tables of the test
years (2005-10-01 .. 2014-09-30) at lead 1, 2 and 3; the high-flow level is the 95th percentile of each basin's flow in the training years (computed here from the flow table); an onset is a high-flow day forecast while the day
lead days before it was at or below the level. The same numbers are then taken from the engine (flood.evaluate: the one-line check of README.md) and compared: they must be equal.
Part B, the ranges of the report (section 8 'How sure are these numbers?', uncertainty.json). The scheme the documents state: 4,000 times, 9 water years are drawn from the 9 test water years and 12 basins from the 12 basins, with
replacement; the interval is the 2.5th to the 97.5th percentile. Here: the critic's own draws (own random numbers) for the share of onsets warned of lstm_wq, lstm_wqn, lstm_wqf, lstm_wqa (1 day), lstm_wqaF_sub (2 and 3 days), the share
of the gap closed by the rain up to the evening, how often its gain over the control is above zero, the NSE of the LSTM minus the best simple model (median over the drawn basins), and the shares of the draws in which the verdict
for all high-flow days of lstm_wq (1 day) and for the onsets of the new sets is GO / CAUTION / NOT YET. Compared with the stored ranges; the draws differ, so the criterion is: within one percentage point (0.01 NSE).
Output: OUT/d2_headline_and_ranges.json
"""
import time

from dcommon import *
import flood as F

t0 = time.time()
D = F.load_data()
ids = list(D)
THR = high_flow_levels(D)
TEST = (pd.Timestamp("2005-10-01"), pd.Timestamp("2014-09-30"))
SETS = ["lstm_wq", "lstm_wqn", "lstm_wqn_older", "lstm_wqh", "lstm_wqa", "lstm_wqf", "lstm_w", "lstm_wqaF_sub", "lstm_wqaF_clim", "lstm_wqaP", "lstm63_wq", "base_linreg_wq", "base_rf_wq", "base_persistence"]
P = {n: np.load(RES / "preds" / f"{n}.npz", allow_pickle=False) for n in SETS}
dates = pd.DatetimeIndex(P["lstm_wq"]["dates"])
assert list(P["lstm_wq"]["basins"]) == ids
in_test = (dates >= TEST[0]) & (dates <= TEST[1])
WY = np.where(dates.month >= 10, dates.year + 1, dates.year)                      # water year of each target day
YEARS = list(range(2006, 2015))


def nse(o, s):
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def basin_arrays(name, lead):
    """For each basin: observed flow, forecast, 'river was at or below the level lead days before' (all by target day), restricted to the test years, with both present."""
    z = P[name]
    out = []
    for i, g in enumerate(ids):
        o, s = z["obs"][i], z["pred"][i, :, lead - 1]
        low = np.concatenate([np.full(lead, False), o[:-lead] <= THR[g]])
        ok = in_test & np.isfinite(o) & np.isfinite(s)
        out.append((o[ok], s[ok], low[ok], WY[ok]))
    return out


def tables(name, lead):
    """The critic's scores: median NSE, and the pooled warning tables of all high-flow days and of the onsets (trigger 1.0)."""
    arr = basin_arrays(name, lead)
    nses = [nse(o, s) for o, s, _, _ in arr]
    tab = {"all": [0, 0, 0, 0], "onset": [0, 0, 0, 0]}
    for g, (o, s, low, _) in zip(ids, arr):
        hi, w = o > THR[g], s > THR[g]
        for kind, m in (("all", np.ones_like(hi)), ("onset", low)):
            h, ms, fa, cn = (hi & w & m).sum(), (hi & ~w & m).sum(), (~hi & w & m).sum(), (~hi & ~w & m).sum()
            for j, v in enumerate((h, ms, fa, cn)):
                tab[kind][j] += int(v)
    r = lambda t: dict(hits=t[0], misses=t[1], false_alarms=t[2], correct_neg=t[3], pod=t[0] / (t[0] + t[1]), far=(t[2] / (t[0] + t[2])) if (t[0] + t[2]) else float("nan"), csi=t[0] / (t[0] + t[1] + t[2]))
    return dict(nse_median=float(np.median(nses)), all=r(tab["all"]), onset=r(tab["onset"]))


mine = {n: {f"l{k}": tables(n, k) for k in (1, 2, 3)} for n in SETS}
eng = {}
cmp = []
for n in SETS:
    ev = F.evaluate(F.load_preds(n), F.thresholds(D))["summary"]
    for k in (1, 2, 3):
        e, m = ev[f"l{k}"], mine[n][f"l{k}"]
        pairs = [("nse_median", m["nse_median"], e["nse_median"])] + [(f"{kind}.{f}", m[kind][f], e[kind][f]) for kind in ("all", "onset") for f in ("hits", "misses", "false_alarms", "correct_neg")]
        for what, a, b in pairs:
            cmp.append(dict(set=n, lead=k, what=what, mine=a, engine=b, equal=bool(abs(a - b) < 1e-9)))
partA = dict(mine=mine, compared_with_flood_evaluate=len(cmp), all_equal=all(c["equal"] for c in cmp), not_equal=[c for c in cmp if not c["equal"]])
print(f"A: {len(cmp)} numbers of {len(SETS)} sets compared with flood.evaluate: all equal {partA['all_equal']}")

# ---- B: the draws -----------------------------------------------------------------------------------------------------------------------------------------------------
def cells(name, lead):
    """Per (water year, basin): onsets, onset hits; all-day hits, misses, false alarms; and the sums that give an NSE (n, sum obs, sum obs^2, sum squared error)."""
    arr = basin_arrays(name, lead)
    C = {k: np.zeros((len(YEARS), len(ids))) for k in ("on", "on_hit", "h", "m", "f", "n", "so", "so2", "sse")}
    for b, (g, (o, s, low, wy)) in enumerate(zip(ids, arr)):
        hi, w = o > THR[g], s > THR[g]
        for y, yr in enumerate(YEARS):
            q = wy == yr
            C["on"][y, b], C["on_hit"][y, b] = (hi & low & q).sum(), (hi & low & w & q).sum()
            C["h"][y, b], C["m"][y, b], C["f"][y, b] = (hi & w & q).sum(), (hi & ~w & q).sum(), (~hi & w & q).sum()
            C["n"][y, b], C["so"][y, b], C["so2"][y, b], C["sse"][y, b] = q.sum(), o[q].sum(), (o[q] ** 2).sum(), ((s[q] - o[q]) ** 2).sum()
    return C


CELLS = {(n, k): cells(n, k) for n, k in [("lstm_wq", 1), ("lstm_wqn", 1), ("lstm_wqn_older", 1), ("lstm_wqf", 1), ("lstm_wqa", 1), ("lstm_wqaF_sub", 2), ("lstm_wqaF_sub", 3), ("lstm_wq", 2), ("lstm_wq", 3),
                                    ("base_linreg_wq", 1), ("base_rf_wq", 2), ("base_rf_wq", 3), ("lstm_wqa", 2), ("lstm_wqa", 3), ("lstm_wqaF_sub", 1), ("lstm_wqn", 2), ("lstm_wqn", 3)]}
rng = np.random.default_rng(20261008)
N = 4000
draws = [(rng.integers(0, len(YEARS), len(YEARS)), rng.integers(0, len(ids), len(ids))) for _ in range(N)]


def share(name, lead):
    C = CELLS[(name, lead)]
    return np.array([C["on_hit"][np.ix_(y, b)].sum() / C["on"][np.ix_(y, b)].sum() for y, b in draws])


def nse_draw(C, y, b):
    n, so, so2, sse = (C[k][np.ix_(y, b)].sum(axis=0) for k in ("n", "so", "so2", "sse"))      # per drawn basin (columns), over the drawn years
    return 1 - sse / (so2 - so ** 2 / n)


def verdict(h, m, f, onset=False):
    pod, far, csi = h / (h + m), f / (h + f) if (h + f) else 1.0, h / (h + m + f)
    return "GO" if (pod >= 0.70 and far <= 0.30) else "CAUTION" if csi >= 0.40 else "NOT YET"


U = jload("uncertainty")
ST = U["stats"]
rows = []


def add(what, vals, key, point=None, scale=1.0):
    """One resampled statistic: my interval against the stored one (within 0.0105), and my point estimate (the test years as they are, no resampling) against the stored value (within 1e-9)."""
    lo, hi = np.percentile(vals, [2.5, 97.5])
    t = ST[key]["both"]
    r = dict(what=what, mine=[float(lo), float(hi)], builder=[t["lo"], t["hi"]], within_tolerance=bool(abs(lo - t["lo"]) <= 0.0105 * scale and abs(hi - t["hi"]) <= 0.0105 * scale))
    if point is not None:
        r.update(point_mine=float(point), point_builder=ST[key]["value"], point_equal=bool(abs(point - ST[key]["value"]) < 1e-9))
    rows.append(r)


def point_share(name, lead):
    C = CELLS[(name, lead)]
    return C["on_hit"].sum() / C["on"].sum()


for name, lead, key in [("lstm_wq", 1, "onsets_warned.lstm_wq.l1"), ("lstm_wqn", 1, "onsets_warned.lstm_wqn.l1"), ("lstm_wqf", 1, "onsets_warned.lstm_wqf.l1"), ("lstm_wqa", 1, "onsets_warned.lstm_wqa.l1"),
                        ("lstm_wqaF_sub", 2, "onsets_warned.lstm_wqaF_sub.l2"), ("lstm_wqaF_sub", 3, "onsets_warned.lstm_wqaF_sub.l3")]:
    add(f"share of onsets warned, {name}, {lead} day(s)", share(name, lead), key, point_share(name, lead))
s_wq, s_wqn, s_wqf, s_ctl = share("lstm_wq", 1), share("lstm_wqn", 1), share("lstm_wqf", 1), share("lstm_wqn_older", 1)
p_wq, p_wqn, p_wqf = point_share("lstm_wq", 1), point_share("lstm_wqn", 1), point_share("lstm_wqf", 1)
add("share of the gap to the perfect forecast closed by the rain up to the evening", (s_wqn - s_wq) / (s_wqf - s_wq), "onsets_gap_share_closed.wq.l1", (p_wqn - p_wq) / (p_wqf - p_wq))
gain_over_control = float(np.mean((s_wqn - s_ctl) > 0))
rows.append(dict(what="draws with the gain of the rain up to the evening over its control above zero", mine=gain_over_control, builder=ST["onsets_gain.evening_rain_minus_control.wq.l1"]["both"]["share_above_zero"],
                 mine_count=int(round(gain_over_control * N)), within_tolerance=bool(abs(gain_over_control - ST["onsets_gain.evening_rain_minus_control.wq.l1"]["both"]["share_above_zero"]) <= 0.002)))
for lead, best in ((1, "base_linreg_wq"), (2, "base_rf_wq"), (3, "base_rf_wq")):
    d = []
    for y, b in draws:
        d.append(np.median(nse_draw(CELLS[("lstm_wq", lead)], y, b) - nse_draw(CELLS[(best, lead)], y, b)))
    full = (np.arange(len(YEARS)), np.arange(len(ids)))                          # all years, all basins, as they are
    pt = np.median(nse_draw(CELLS[("lstm_wq", lead)], *full) - nse_draw(CELLS[(best, lead)], *full))
    add(f"NSE of the LSTM minus the best simple model (median over the drawn basins), {lead} day(s)", np.array(d), f"nse_median_of_diffs.lstm_wq_minus_best_simple.l{lead}", pt)
# the verdict for all high-flow days of lstm_wq at 1 day (the CAUTION that holds in 83% of the draws)
C = CELLS[("lstm_wq", 1)]
v = [verdict(*(C[k][np.ix_(y, b)].sum() for k in ("h", "m", "f"))) for y, b in draws]
vs = {k: float(np.mean([x == k for x in v])) for k in ("GO", "CAUTION", "NOT YET")}
bv = U["verdict_stability"]["lstm_wq.all.l1"]["share_of_resamples_with_each_verdict"]
rows.append(dict(what="verdict for all high-flow days, lstm_wq, 1 day: shares of the draws", mine=vs, builder=bv, within_tolerance=bool(all(abs(vs[k] - bv[k]) <= 0.012 for k in vs))))
# the onset verdicts of the new sets in the draws (the report says the verdict stays NOT YET for every new input; this is how often it is NOT YET in the draws)
onset_verdicts = {}
# the false warnings among the onset warnings need the 'was low' false alarms: recompute cells for them
def onset_fa_cells(name, lead):
    arr = basin_arrays(name, lead)
    fa = np.zeros((len(YEARS), len(ids)))
    for b, (g, (o, s, low, wy)) in enumerate(zip(ids, arr)):
        hi, w = o > THR[g], s > THR[g]
        for y, yr in enumerate(YEARS):
            fa[y, b] = (~hi & w & low & (wy == yr)).sum()
    return fa


for name, lead in [("lstm_wq", 1), ("lstm_wq", 2), ("lstm_wq", 3), ("lstm_wqn", 1), ("lstm_wqn", 2), ("lstm_wqn", 3), ("lstm_wqa", 1), ("lstm_wqa", 2), ("lstm_wqa", 3), ("lstm_wqaF_sub", 1), ("lstm_wqaF_sub", 2), ("lstm_wqaF_sub", 3)]:
    C, fa = CELLS[(name, lead)], onset_fa_cells(name, lead)
    vv = [verdict(C["on_hit"][np.ix_(y, b)].sum(), (C["on"] - C["on_hit"])[np.ix_(y, b)].sum(), fa[np.ix_(y, b)].sum()) for y, b in draws]
    onset_verdicts[f"{name}.l{lead}"] = {k: float(np.mean([x == k for x in vv])) for k in ("GO", "CAUTION", "NOT YET")}
    key = f"{name}.onset.l{lead}"
    if key in U["verdict_stability"]:
        bv_ = U["verdict_stability"][key]["share_of_resamples_with_each_verdict"]
        rows.append(dict(what=f"onset verdict shares of the draws, {name}, {lead} day(s)", mine=onset_verdicts[f'{name}.l{lead}'], builder=bv_,
                         within_tolerance=bool(all(abs(onset_verdicts[f'{name}.l{lead}'][k] - bv_[k]) <= 0.02 for k in bv_))))
partB = dict(draws=N, rows=rows, all_within_tolerance=all(r["within_tolerance"] for r in rows), all_point_estimates_equal=all(r.get("point_equal", True) for r in rows), onset_verdict_shares_in_the_draws=onset_verdicts)
for r in rows:
    print(("ok   " if r["within_tolerance"] else "DIFF ") + r["what"], "| mine", r["mine"], "| builder", r["builder"])
res = dict(part_A_scores=partA, part_B_ranges=partB, seconds=round(time.time() - t0, 1))
jsave("d2_headline_and_ranges.json", res)
print("B: all within tolerance:", partB["all_within_tolerance"], "| all point estimates equal:", partB["all_point_estimates_equal"])
