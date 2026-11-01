"""E2 review, check (i) continued: is the lead-1 loss linked to the NATURE of the slot values? In training the slots hold the AORC rain of the target days: many of them are exactly 0.0 (dry days). A bias-corrected forecast
is rarely exactly 0: it carries small amounts of drizzle on days the analysis calls dry. Table-level look (no network is run): (1) the distribution of the slot of lead 1 in the training years (AORC), and in the validation and test
years for the AORC and for the bias-corrected and the raw forecast: share exactly 0, below 0.1 / 0.5 / 1 / 5 mm; (2) the same on the lead-1 onsets that wqa warned and wqaF_sub missed ('lost'), that only wqaF_sub warned
('gained'), and on all the others; (3) the share of the onsets warned by wqa and by wqaF_sub by the size of the AORC slot of the onset day and of the forecast slot (classes), to see where the net loss sits.
Reads the saved prediction sets and the reviewer's rebuilt tables (f6_tables.py writes them). About 10 seconds."""
import sys; sys.dont_write_bytecode = True
import pickle
from f_common import *
from f_scoring import *

TAB, ids = pickle.load(open(f"{OUT}/f6_tables_csv.pkl", "rb"))
T = TAB["main"]
tr_end, val_start = pd.Timestamp("2000-09-30"), pd.Timestamp("2000-10-01")
def dist(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return dict(n=int(len(x)), exactly_0=float((x == 0).mean()), below_0p1=float((x < 0.1).mean()), below_0p5=float((x < 0.5).mean()), below_1=float((x < 1).mean()), below_5=float((x < 5).mean()), median_of_wet=float(np.median(x[x > 0])) if (x > 0).any() else float("nan"))
out = {}
tr = np.concatenate([T[g].loc[:tr_end, "p1"].values for g in ids]); vt = {c: np.concatenate([T[g].loc[val_start:, c].values for g in ids]) for c in ("p1", "fs1", "fr1")}
out["training_years_AORC_slot1"] = dist(tr); out["validation_and_test_years"] = {c: dist(v) for c, v in vt.items()}
print("slot of lead 1: share exactly 0 / below 0.1 / 0.5 / 1 / 5 mm, median of the wet values")
for k, d in [("training years, AORC (p1)", out["training_years_AORC_slot1"])] + [(f"validation+test years, {c}", d) for c, d in out["validation_and_test_years"].items()]:
    print(f"   {k:34s} n {d['n']:6d}: 0: {100 * d['exactly_0']:5.1f}%  <0.1: {100 * d['below_0p1']:5.1f}%  <0.5: {100 * d['below_0p5']:5.1f}%  <1: {100 * d['below_1']:5.1f}%  <5: {100 * d['below_5']:5.1f}%  median wet {d['median_of_wet']:.2f} mm")
# onset groups at lead 1
tA, tB = onset_table("lstm_wqa", 1), onset_table("lstm_wqaF_sub", 1)
rows = []
for g in ids:
    d, h, wA = tA[g]; d2, h2, wB = tB[g]
    assert (d == d2).all() and (h == h2).all()
    issue_day = d - pd.Timedelta(days=1)
    sub = T[g].reindex(issue_day)
    for i in np.flatnonzero(h):
        rows.append((g, d[i], bool(wA[i]), bool(wB[i]), sub["p1"].values[i], sub["fs1"].values[i], sub["fr1"].values[i], sub["aorc_to_midnight"].values[i]))
E = pd.DataFrame(rows, columns=["g", "date", "wqa", "wqaF", "aorc", "fcst", "raw", "to_midnight"])
grp = {"lost (wqa warned, wqaF_sub not)": E[E.wqa & ~E.wqaF], "gained (only wqaF_sub warned)": E[~E.wqa & E.wqaF], "both warned": E[E.wqa & E.wqaF], "neither": E[~E.wqa & ~E.wqaF]}
out["onset_groups_lead1"] = {}
print("\nlead-1 onsets by group: the AORC rain and the forecast rain of the onset day (mm)")
for k, e in grp.items():
    r = dict(n=int(len(e)), aorc_exactly_0=float((e.aorc == 0).mean()), forecast_exactly_0=float((e.fcst == 0).mean()), aorc_below_1=float((e.aorc < 1).mean()), forecast_below_1=float((e.fcst < 1).mean()), aorc_median=float(e.aorc.median()), forecast_median=float(e.fcst.median()), to_midnight_median=float(e.to_midnight.median()))
    out["onset_groups_lead1"][k] = r
    print(f"   {k:34s} n {r['n']:3d}: AORC 0: {100 * r['aorc_exactly_0']:4.0f}% (<1 mm {100 * r['aorc_below_1']:3.0f}%) median {r['aorc_median']:5.1f} | forecast 0: {100 * r['forecast_exactly_0']:4.0f}% (<1 mm {100 * r['forecast_below_1']:3.0f}%) median {r['forecast_median']:5.1f} | rain to midnight median {r['to_midnight_median']:5.1f}")
# classes of the AORC slot of the onset day
bins = [(-1, 0, "AORC 0 mm"), (0, 1, "AORC >0 to 1 mm"), (1, 5, "AORC 1 to 5 mm"), (5, 20, "AORC 5 to 20 mm"), (20, 1e9, "AORC over 20 mm")]
print("\nlead-1 onsets by the size of the AORC rain of the onset day: the share warned by wqa and by wqaF_sub, and the share the forecast puts in the same class")
out["by_aorc_class"] = {}
for lo, hi, name in bins:
    e = E[(E.aorc > lo) & (E.aorc <= hi)] if lo >= 0 else E[E.aorc <= hi]
    if len(e) == 0:
        continue
    same = float(((e.fcst > lo) & (e.fcst <= hi)).mean()) if lo >= 0 else float((e.fcst <= hi).mean())
    r = dict(n=int(len(e)), wqa=float(e.wqa.mean()), wqaF_sub=float(e.wqaF.mean()), forecast_in_same_class=same, forecast_median=float(e.fcst.median()))
    out["by_aorc_class"][name] = r
    print(f"   {name:18s} n {r['n']:3d}: wqa warned {100 * r['wqa']:5.1f}% | wqaF_sub warned {100 * r['wqaF_sub']:5.1f}% | forecast in the same class {100 * same:4.0f}% (forecast median {r['forecast_median']:.1f} mm)")
print("\nthe same by the size of the FORECAST slot:")
out["by_forecast_class"] = {}
for lo, hi, name in [(-1, 0, "forecast 0 mm"), (0, 1, "forecast >0 to 1 mm"), (1, 5, "forecast 1 to 5 mm"), (5, 20, "forecast 5 to 20 mm"), (20, 1e9, "forecast over 20 mm")]:
    e = E[(E.fcst > lo) & (E.fcst <= hi)] if lo >= 0 else E[E.fcst <= hi]
    if len(e) == 0:
        continue
    r = dict(n=int(len(e)), wqa=float(e.wqa.mean()), wqaF_sub=float(e.wqaF.mean()), aorc_median=float(e.aorc.median()))
    out["by_forecast_class"][name] = r
    print(f"   {name:20s} n {r['n']:3d}: wqa warned {100 * r['wqa']:5.1f}% | wqaF_sub warned {100 * r['wqaF_sub']:5.1f}% | AORC median {r['aorc_median']:.1f} mm")
jsave("f12c_drizzle.json", out)

# ---- part 2: who warns of what, by the size of tomorrow's rain, for the sets that see the slots differently. All issue days at or below the threshold (the onset subset: high = onset, not high = a day that could be a false warning)
SETS_BY_CLASS = ["lstm_wqa", "lstm_wqaF_sub", "lstm_wqaF_sub_perfectpred", "lstm_wqaP", "lstm_wqaF_sub_rawpred", "lstm_wqaF_clim", "base_linreg_wqaF", "base_rf_wqaF", "base_rf_wqa"]
frames = []
for name in SETS_BY_CLASS:
    t = onset_table(name, 1)
    for g in ids:
        d, h, w = t[g]
        frames.append(pd.DataFrame(dict(set=name, g=g, date=d, high=h, warn=w)))
Z = pd.concat(frames)
aorc = pd.concat([pd.DataFrame(dict(g=g, date=T[g].index + pd.Timedelta(days=1), aorc=T[g]["p1"].values, fcst=T[g]["fs1"].values)) for g in ids])
Z = Z.merge(aorc, on=["g", "date"], how="left")
Z["cls"] = pd.cut(Z.aorc, [-1, 0, 1, 5, 20, 1e9], labels=["0 mm", ">0 to 1 mm", "1 to 5 mm", "5 to 20 mm", "over 20 mm"])
tab = {}
print("\nlead 1, onset subset (issue-day flow at or below the threshold), by the AORC rain of the target day: share of the ONSETS warned (hits) / share of the NON-ONSET days warned (false-warning rate)")
head = "   " + f"{'class':12s} {'n onsets':>8s} {'n other':>8s} | " + " | ".join(f"{n.replace('lstm_', '').replace('wqaF_sub', 'F_sub'):>12s}" for n in SETS_BY_CLASS)
print(head)
for c in ["0 mm", ">0 to 1 mm", "1 to 5 mm", "5 to 20 mm", "over 20 mm"]:
    z = Z[Z.cls == c]; row = {}
    n_on = int(z[(z["set"] == SETS_BY_CLASS[0]) & z.high].shape[0]); n_off = int(z[(z["set"] == SETS_BY_CLASS[0]) & ~z.high].shape[0])
    cells = []
    for n in SETS_BY_CLASS:
        zz = z[z["set"] == n]; hit = float(zz[zz.high].warn.mean()); fa = float(zz[~zz.high].warn.mean())
        row[n] = dict(pod=hit, false_warning_rate=fa); cells.append(f"{100 * hit:5.1f} /{100 * fa:5.1f}")
    tab[c] = dict(n_onsets=n_on, n_other=n_off, sets=row)
    print("   " + f"{c:12s} {n_on:8d} {n_off:8d} | " + " | ".join(f"{x:>12s}" for x in cells))
tot = {}
for n in SETS_BY_CLASS:
    zz = Z[Z["set"] == n]; tot[n] = dict(pod=float(zz[zz.high].warn.mean()), false_warning_rate=float(zz[~zz.high].warn.mean()))
print("   " + f"{'all':12s} {'':8s} {'':8s} | " + " | ".join(f"{100 * tot[n]['pod']:5.1f} /{100 * tot[n]['false_warning_rate']:5.1f}".rjust(12) for n in SETS_BY_CLASS))
out["by_aorc_class_all_sets_lead1"] = dict(classes=tab, total=tot, columns=SETS_BY_CLASS)
jsave("f12c_drizzle.json", out)
