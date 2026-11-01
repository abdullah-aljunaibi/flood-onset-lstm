"""AORC review, check (c) part 3 and the structure behind (d): the hour-stamp evidence against hourly Stage IV and how AORC's hours relate to Stage IV's within a gauge day.
 1. lag correlations: AORC at hour-ending stamp T against Stage IV at label T-k, k = -3..+3, pooled over the 12 basins, for the builder's 50 days (data/aorc/stage_iv_hindsight.csv), for my 28 days, and pooled (r = 0.977 at
    lag 0 is claimed for four storm days); by test year and by basin: is the best lag 0 everywhere? (a stamp that were the START of the hour would peak at k = -1)
 2. structure: the AORC documentation says the daily AORC rain is disaggregated with the Stage IV hourly values. If so, within one gauge day (12:00 UTC to 12:00 UTC) the hours of AORC are Stage IV's hours times ONE
    factor. Tested on every basin-day of the samples: the 24 hours (labels 13:00 UTC of day t to 12:00 UTC of day t+1) have a ratio sum(A)/sum(S) and a spread of the hourly ratios A_h/S_h; and the ratio of the hours before
    and after a boundary is compared at three boundaries: 12:00 UTC (the documented gauge-day boundary), local midnight, and a placebo (18:00 UTC). About 20 seconds."""
import sys; sys.dont_write_bytecode = True
from a_stage4_data import *

H = jload("results/aorc_hindsight.json")
mine_days = jload("a7a_days.json") if False else json.load(open(f"{OUT}/a7a_days.json"))["days"]
builder_days = H["numbers"]["days"]
S_all, S_b, S_m = stage4("both"), stage4("builder"), stage4("mine")
out = {}


def lag_corr(S, k, gl=None):
    a_l, s_l = [], []
    for g in (gl or ids):
        a = aorc(g).reindex(S.index)
        s = S[g].reindex(S.index - pd.Timedelta(hours=k)).values
        ok = np.isfinite(a.values) & np.isfinite(s)
        a_l.append(a.values[ok]); s_l.append(s[ok])
    a, s = np.concatenate(a_l), np.concatenate(s_l)
    return float(np.corrcoef(a, s)[0, 1]), int(len(a))


print("1. lag correlations, AORC at stamp T against Stage IV at label T-k (pooled over basins and hours)")
res = {}
for name, S in (("builder 50 days", S_b), ("mine 28 days", S_m), ("pooled", S_all)):
    r = {k: lag_corr(S, k) for k in range(-3, 4)}
    res[name] = {str(k): v[0] for k, v in r.items()}
    print(f"  {name:16s} n={r[0][1]:6d}: " + "  ".join(f"k={k:+d}: {r[k][0]:.4f}" for k in range(-3, 4)) + f"   best lag {max(r, key=lambda k: r[k][0]):+d}")
out["lag_correlations"] = res
# by test year and by basin (pooled sample)
yr = {}
for y in range(2005, 2015):
    sel = S_all[S_all.index.year == y]
    if len(sel) < 50: continue
    r = {k: lag_corr(sel, k)[0] for k in range(-3, 4)}
    yr[y] = dict(best=max(r, key=r.get), r0=r[0], r_minus1=r[-1], r_plus1=r[1], n=int(len(sel)))
print("  by year (pooled sample):", {y: (v["best"], round(v["r0"], 3)) for y, v in yr.items()})
bb = {}
for g in ids:
    r = {k: lag_corr(S_all, k, [g])[0] for k in range(-3, 4)}
    bb[g] = max(r, key=r.get)
print("  best lag by basin (pooled sample):", bb)
out["by_year"] = yr; out["best_lag_by_basin"] = bb
out["stamp_is_end_of_hour_in_the_data"] = bool(all(v["best"] == 0 for v in yr.values()) and all(v == 0 for v in bb.values()))

# ---- 2. structure within a gauge day ------------------------------------------------------------------------------------------------------------------------------------
print("\n2. within a gauge day (12:00 UTC to 12:00 UTC): is AORC = Stage IV hours x one factor?")
rows = []
for d in sorted(set(builder_days) | set(mine_days)):
    d = pd.Timestamp(d)
    lab = pd.date_range(d + pd.Timedelta(hours=13), d + pd.Timedelta(hours=36), freq="h")             # labels 13:00 UTC of day t to 12:00 UTC of day t+1: the gauge day that holds the end of local day t
    if not all(t in S_all.index for t in lab): continue
    for g in ids:
        s = S_all.loc[lab, g]; a = aorc(g).reindex(lab)
        if s.isna().any(): continue
        mid = inst(d + pd.Timedelta(days=1), 0, g)                                                      # local midnight at the end of day t, as a UTC instant
        pre, post = lab <= mid, lab > mid
        rows.append(dict(day=str(d.date()), gauge=g, S=float(s.sum()), A=float(a.sum()), S_pre=float(s[pre].sum()), A_pre=float(a[pre].sum()), S_post=float(s[post].sum()), A_post=float(a[post].sum()),
                         r=float(np.corrcoef(a.values, s.values)[0, 1]) if s.std() > 0 and a.std() > 0 else np.nan, n_wet=int((s >= 0.5).sum()),
                         cv=float(np.std((a / s)[s >= 0.5]) / np.mean((a / s)[s >= 0.5])) if (s >= 0.5).sum() >= 5 else np.nan))
df = pd.DataFrame(rows)
rain = df[df.S >= 5]
print(f"  {len(df)} basin-gauge-days; with at least 5 mm of Stage IV rain: {len(rain)}")
print(f"  hourly correlation of AORC with Stage IV within the gauge day: median {rain.r.median():.4f}, p10 {rain.r.quantile(.1):.4f} (pooled over days and hours, lag 0: {res['pooled']['0']:.4f})")
print(f"  ratio of the daily sums A/S: median {(rain.A / rain.S).median():.3f}, p10 {(rain.A / rain.S).quantile(.1):.3f}, p90 {(rain.A / rain.S).quantile(.9):.3f}; spread of the hourly ratios A_h/S_h within a day (hours with S_h >= 0.5 mm, at least 5): median coefficient of variation {rain.cv.median():.3f}")
lowr = rain[rain.r < 0.98]
print(f"  gauge-day basin-days with at least 5 mm of Stage IV rain whose hourly AORC profile departs from Stage IV's (r < 0.98): {len(lowr)} of {len(rain)}; lowest r {rain.r.min():.3f}; with the hourly ratio spread CV > 0.15: {(rain.cv > 0.15).sum()} (a daily total spread by a model pattern, GDAS, instead of the radar hours would show here)")
out["structure_low_r"] = dict(n_r_below_0_98=int(len(lowr)), lowest_r=float(rain.r.min()), n_cv_above_0_15=int((rain.cv > 0.15).sum()), cases=lowr.sort_values("r").head(8)[["day", "gauge", "S", "A", "r", "cv"]].round(3).to_dict("records"))
out["structure"] = dict(n_basin_gauge_days=int(len(df)), n_rain_days=int(len(rain)), hourly_corr_within_day_median=float(rain.r.median()), hourly_corr_within_day_p10=float(rain.r.quantile(.1)), ratio_daily_median=float((rain.A / rain.S).median()),
                        ratio_daily_p10=float((rain.A / rain.S).quantile(.1)), ratio_daily_p90=float((rain.A / rain.S).quantile(.9)), cv_hourly_ratio_within_day_median=float(rain.cv.median()))


def boundary_stats(boundary):
    """for each basin-gauge-day with at least 3 mm of Stage IV rain in both parts around the boundary (hours of the 24 before / after it), the log ratio sum(A)/sum(S) before and after, their correlation and the median jump."""
    lr = []
    for d in sorted(set(builder_days) | set(mine_days)):
        d = pd.Timestamp(d)
        for g in ids:
            lab = pd.date_range(d + pd.Timedelta(hours=13), d + pd.Timedelta(hours=36), freq="h")
            if not all(t in S_all.index for t in lab): continue
            s = S_all.loc[lab, g]; a = aorc(g).reindex(lab)
            if s.isna().any(): continue
            b = boundary(d, g)
            before, after = lab <= b, lab > b
            if before.sum() < 3 or after.sum() < 3: continue
            sb, sa, ab, aa = s[before].sum(), s[after].sum(), a[before].sum(), a[after].sum()
            if sb >= 3 and sa >= 3 and ab > 0 and aa > 0: lr.append((np.log(ab / sb), np.log(aa / sa)))
    lr = np.array(lr)
    return dict(n=int(len(lr)), corr_log_ratios=float(np.corrcoef(lr[:, 0], lr[:, 1])[0, 1]), median_abs_jump=float(np.median(np.abs(lr[:, 1] - lr[:, 0]))), mean_log_ratio_before=float(lr[:, 0].mean()), mean_log_ratio_after=float(lr[:, 1].mean())) if len(lr) > 3 else dict(n=int(len(lr)))
bounds = {"12:00 UTC (the documented boundary of the gauge day)": lambda d, g: d + pd.Timedelta(hours=36),
          "local midnight (end of local day t)": lambda d, g: inst(d + pd.Timedelta(days=1), 0, g),
          "placebo, 18:00 UTC of day t+1": lambda d, g: d + pd.Timedelta(hours=42)}
# the 12:00 UTC boundary lies inside the 24 labels only for the day BEFORE: use the boundary 12:00 UTC of day t (between hour 13:00 of the previous label and ...): handled by shifting the window for that case
bs = {}
for name, b in (("local midnight (end of local day t)", bounds["local midnight (end of local day t)"]),):
    bs[name] = boundary_stats(b)
# for 12:00 UTC of day t the hours of the gauge days on either side are labels 05:00..12:00 UTC (7 hours) and 13:00 UTC .. the end of the window; use labels 05..23 UTC of day t
def boundary_12z(split_label_hour):
    lr = []
    for d in sorted(set(builder_days) | set(mine_days)):
        d = pd.Timestamp(d)
        lab = pd.date_range(d + pd.Timedelta(hours=6), d + pd.Timedelta(hours=23), freq="h")             # labels 06:00 .. 23:00 UTC of day t (the local day t in the east, no post-midnight hours)
        if not all(t in S_all.index for t in lab): continue
        for g in ids:
            s = S_all.loc[lab, g]; a = aorc(g).reindex(lab)
            if s.isna().any(): continue
            before = lab <= d + pd.Timedelta(hours=split_label_hour); after = ~before
            if before.sum() < 3 or after.sum() < 3: continue
            sb, sa, ab, aa = s[before].sum(), s[after].sum(), a[before].sum(), a[after].sum()
            if sb >= 3 and sa >= 3 and ab > 0 and aa > 0: lr.append((np.log(ab / sb), np.log(aa / sa)))
    lr = np.array(lr)
    return dict(n=int(len(lr)), corr_log_ratios=float(np.corrcoef(lr[:, 0], lr[:, 1])[0, 1]), median_abs_jump=float(np.median(np.abs(lr[:, 1] - lr[:, 0]))), mean_log_ratio_before=float(lr[:, 0].mean()), mean_log_ratio_after=float(lr[:, 1].mean())) if len(lr) > 3 else dict(n=int(len(lr)))
bs["12:00 UTC of day t (documented gauge-day boundary), hours 06-12 UTC against 13-23 UTC"] = boundary_12z(12)
bs["placebo: 09:00 UTC of day t, hours 06-09 against 10-23 UTC"] = boundary_12z(9)
bs["placebo: 16:00 UTC of day t, hours 06-16 against 17-23 UTC"] = boundary_12z(16)
print("\n  ratio sum(A)/sum(S) before and after a boundary (basin-days with at least 3 mm of Stage IV rain on both sides):")
for k, v in bs.items():
    print(f"   {k:92s} n={v['n']:4d}  corr of the log ratios {v.get('corr_log_ratios', float('nan')):+.3f}  median |log ratio after - before| {v.get('median_abs_jump', float('nan')):.3f}")
out["boundaries"] = bs

# the documented anchoring (methods document, section 4.1: from 2010 the daily total is Stage IV with a calendar-month ratio; 1979-2015 bias-corrected with monthly Livneh totals): is AORC / Stage IV a constant per calendar month?
bd = pd.read_csv(f"{OUT}/a7_basin_days.csv", dtype={"gauge": str}, parse_dates=["day"]) if os.path.exists(f"{OUT}/a7_basin_days.csv") else None
if bd is not None:
    dd = bd[(bd.S_exp >= 5) & (bd.A_exp > 0)].copy(); dd["lr"] = np.log(dd.A_exp / dd.S_exp); dd["era"] = np.where(dd.day.dt.year >= 2010, "2010-2014", "2005-2009")
    era = {}
    for e, f in dd.groupby("era"):
        mm = f.groupby(f.day.dt.month).lr.transform("mean")
        era[e] = dict(n=int(len(f)), median_ratio=float(np.exp(f.lr.median())), p10=float(np.exp(f.lr.quantile(.1))), p90=float(np.exp(f.lr.quantile(.9))), sd_log_ratio=float(f.lr.std()), share_explained_by_calendar_month=float(1 - ((f.lr - mm) ** 2).sum() / ((f.lr - f.lr.mean()) ** 2).sum()))
    out["ratio_by_era"] = era
    print("\n  AORC / Stage IV of the hours 12 UTC .. local midnight (basin-days with at least 5 mm): " + "; ".join(f"{e}: n {v['n']}, median {v['median_ratio']:.3f}, p10-p90 {v['p10']:.2f}-{v['p90']:.2f}, sd of the log ratio {v['sd_log_ratio']:.3f}, share explained by calendar-month means {v['share_explained_by_calendar_month']:.2f}" for e, v in era.items()))
jsave("a5b_stamp.json", out)
