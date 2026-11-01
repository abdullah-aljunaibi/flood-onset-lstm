"""AORC review, check (d): THE HINDSIGHT RISK, tested in the data harder than the builder's 50 stormiest days.
What the AORC v1.1 methods document says (https://www.weather.gov/media/owp/operations/aorc_v1_1_methods.pdf, section 4.1, read here): the daily reference of 1979-2001 is NLDAS2's 24-hour totals ending at 1200 UTC,
that of 2002 on Stage IV's daily precipitation (manual quality control), bias-corrected with monthly Livneh totals until 2015; 'the daily time series were then disaggregated using available hourly data': Stage IV hourly
data in the recent years, GDAS where radar and satellite are missing. So an AORC hour before local midnight is the Stage IV hour times a factor that is set by a gauge day (12:00 UTC to 12:00 UTC) which runs on
past midnight to 12:00 UTC (07:00 EST). This script measures, on the Stage IV hours I hold (builder's 50 days + my 28 + their extension), per basin and day t (wall clock of each gauge):
  A_*, S_* : AORC and Stage IV rain over  early (00:00 local .. 12 UTC of day t), exposed (12 UTC of day t .. 24:00 local: the hours that share a gauge day with the night after the issue time),
             late (18:00 .. 24:00 local), pre (00:00 .. 24:00 local: the input of the model), post (24:00 local .. 12 UTC of day t+1: the gauge day's rain after the issue time), after (12 UTC .. 12:00 local of t+1).
 1. A_pre = b0 + b1 S_pre + b2 S_post  and  A_late = b0 + b1 S_late + b2 S_post  (the form asked for), 95% interval of b2 by drawing DAYS again (2000 draws), for the builder's 50 days, my 28 days and both;
    the placebo: the rain after 12 UTC (which belongs to the next gauge day) added as a regressor must get a coefficient near zero;
 2. the factor model: log(A_exp / S_exp) = c0 + c1 * share_post, share_post = S_post / (S_exp + S_post): one factor per gauge day means pre-midnight hours move with the whole day;
 3. the ratio A/S before midnight against the after-midnight share, and the days with little rain before midnight and much after (S_pre < 0.3 mm, S_post > 3 mm).
Writes a7_hindsight.json (including the calibrations lambda and c used by a8_hindsight_networks.py). About 30 seconds."""
import sys; sys.dont_write_bytecode = True
from a_stage4_data import *

H = jload("results/aorc_hindsight.json")
mine_days = json.load(open(f"{OUT}/a7a_days.json"))["days"]; builder_days = H["numbers"]["days"]
S_all = stage4("both"); out = {}
rng = np.random.default_rng(20261007)
HR = pd.Timedelta(hours=1)
rows = []
for d in sorted(set(builder_days) | set(mine_days)):
    d = pd.Timestamp(d)
    for g in ids:
        m0, m24, m18 = inst(d, 0, g), inst(d + pd.Timedelta(days=1), 0, g), inst(d, 18, g)
        z12, z12n, noon_n = d + 12 * HR, d + pd.Timedelta(days=1) + 12 * HR, inst(d + pd.Timedelta(days=1), 12, g)
        W = dict(early=hours_between(m0, z12), exp=hours_between(z12, m24), late=hours_between(m18, m24), pre=hours_between(m0, m24), post=hours_between(m24, z12n), after=hours_between(z12n, noon_n))
        need = [t for k in ("pre", "post") for t in W[k]]
        if not all(t in S_all.index for t in need) or S_all.loc[need, g].isna().any(): continue
        row = dict(day=str(d.date()), gauge=g, sample="builder" if str(d.date()) in builder_days else "mine")
        a = aorc(g)
        for k, w in W.items():
            if k == "after" and not all(t in S_all.index for t in w): row[f"A_{k}"] = row[f"S_{k}"] = np.nan; continue
            row[f"A_{k}"] = float(a.reindex(w).sum()); row[f"S_{k}"] = float(S_all.loc[w, g].sum())
        rows.append(row)
df = pd.DataFrame(rows)
print(f"{len(df)} basin-days ({df.day.nunique()} days: builder's {df[df['sample'] == 'builder'].day.nunique()}, mine {df[df['sample'] == 'mine'].day.nunique()}); mean Stage IV rain: pre {df.S_pre.mean():.1f} mm, post (to 12 UTC) {df.S_post.mean():.2f} mm; "
      f"basin-days with S_post > 3 mm: {(df.S_post > 3).sum()} (builder {((df.S_post > 3) & (df['sample'] == 'builder')).sum()}, mine {((df.S_post > 3) & (df['sample'] == 'mine')).sum()})")
out["n_basin_days"] = int(len(df)); out["n_days"] = int(df.day.nunique())
out["mean_mm"] = {k: float(df[k].mean()) for k in ("S_pre", "A_pre", "S_post", "A_post", "S_exp", "A_exp", "S_late", "A_late")}


def ols(frame, y, xs):
    X = np.column_stack([np.ones(len(frame))] + [frame[x].values for x in xs]); b = np.linalg.lstsq(X, frame[y].values, rcond=None)[0]
    yv = frame[y].values; r2 = 1 - ((yv - X @ b) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
    return b, float(r2)


def boot(frame, y, xs, B=2000):
    by_day = {d: f for d, f in frame.groupby("day")}; keys = list(by_day); res = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        res.append(ols(pd.concat([by_day[keys[i]] for i in pick]), y, xs)[0])
    return np.array(res)


samples = {"builder 50 days": df[df["sample"] == "builder"], "my 28 days": df[df["sample"] == "mine"], "both (78 days)": df}
print("\n1. regressions (mm; one row per basin and day; 95% interval of each coefficient from drawing the days again)")
reg = {}
for sname, fr in samples.items():
    for y, x1 in (("A_pre", "S_pre"), ("A_late", "S_late")):
        b, r2 = ols(fr, y, [x1, "S_post"]); bt = boot(fr, y, [x1, "S_post"]); lo, hi = np.percentile(bt[:, 2], [2.5, 97.5]); lo1, hi1 = np.percentile(bt[:, 1], [2.5, 97.5])
        r2_0 = ols(fr, y, [x1])[1]
        reg[f"{sname}|{y}~{x1}+S_post"] = dict(n=int(len(fr)), b0=float(b[0]), b1=float(b[1]), b1_ci=[float(lo1), float(hi1)], b2=float(b[2]), b2_ci=[float(lo), float(hi)], r2=r2, r2_without_s_post=r2_0, mean_S_post=float(fr.S_post.mean()), mean_Y=float(fr[y].mean()),
                                               b2_times_mean_S_post_over_mean_Y_percent=float(100 * b[2] * fr.S_post.mean() / fr[y].mean()))
        print(f"  {sname:16s} {y} = b0 + b1 {x1} + b2 S_post: n={len(fr):4d} b1 {b[1]:.3f} [{lo1:.3f}, {hi1:.3f}]  b2 {b[2]:+.4f} [{lo:+.4f}, {hi:+.4f}]  R2 {r2_0:.4f} -> {r2:.4f}; b2 x mean S_post = {100 * b[2] * fr.S_post.mean() / fr[y].mean():+.2f}% of the mean {y} ({fr.S_post.mean():.2f} mm)")
out["regressions"] = reg
# placebo: the rain after 12 UTC of day t+1 (next gauge day) must not enter
fr = df.dropna(subset=["S_after"])
if len(fr) > 100:
    b, r2 = ols(fr, "A_pre", ["S_pre", "S_post", "S_after"]); bt = boot(fr, "A_pre", ["S_pre", "S_post", "S_after"]); lo, hi = np.percentile(bt[:, 3], [2.5, 97.5]); lo2, hi2 = np.percentile(bt[:, 2], [2.5, 97.5])
    out["placebo_after_12_utc"] = dict(n=int(len(fr)), days=int(fr.day.nunique()), b_post=float(b[2]), b_post_ci=[float(lo2), float(hi2)], b_after=float(b[3]), b_after_ci=[float(lo), float(hi)], mean_S_after=float(fr.S_after.mean()))
    print(f"   with the rain from 12 UTC to 12:00 local of day t+1 as a third regressor (my {fr.day.nunique()} days): b2 (to 12 UTC) {b[2]:+.4f} [{lo2:+.4f}, {hi2:+.4f}], b3 (after 12 UTC, the placebo) {b[3]:+.4f} [{lo:+.4f}, {hi:+.4f}] (mean S_after {fr.S_after.mean():.2f} mm)")

# the window the brief names, 00:00-12:00 local of day t+1 in one piece (S_0012 = S_post + S_after: from local midnight to 12:00 local, two gauge days), my 28 days, the only ones with the hours after 12 UTC
if len(fr) > 100:
    fr = fr.assign(S_0012=fr.S_post + fr.S_after)
    rng_main, rng = rng, np.random.default_rng(20261007)          # its own random draws, so that the draws of the other regressions (and the calibration the network tests use) stay as they were
    out["regressions_to_noon"] = {}
    for y, x1 in (("A_pre", "S_pre"), ("A_late", "S_late")):
        b, r2 = ols(fr, y, [x1, "S_0012"]); bt = boot(fr, y, [x1, "S_0012"]); lo, hi = np.percentile(bt[:, 2], [2.5, 97.5])
        out["regressions_to_noon"][f"my 28 days|{y}~{x1}+S_0012"] = dict(n=int(len(fr)), b2=float(b[2]), b2_ci=[float(lo), float(hi)], mean_S_0012=float(fr.S_0012.mean()), mean_Y=float(fr[y].mean()))
        print(f"   {y} = b0 + b1 {x1} + b2 S_0012 (00:00-12:00 local of t+1, mean {fr.S_0012.mean():.2f} mm), my {fr.day.nunique()} days: b2 {b[2]:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    rng = rng_main

# a positive control: a leak of known size put into the data must be found by the same regression. Y = A_pre + lambda * A_post (AORC's own rain from midnight to 12 UTC added to the hours before midnight),
# regressed on S_pre and S_post for the 78 days with the same day-bootstrap; the coefficient should rise by lambda x (A_post / S_post, about 1.09) and its interval should leave zero as lambda grows
rng_main, rng = rng, np.random.default_rng(20261008)
out["positive_control"] = {}
for lam in (0.03, 0.05, 0.12):
    fl = df.assign(Y=df.A_pre + lam * df.A_post)
    b, r2 = ols(fl, "Y", ["S_pre", "S_post"]); bt = boot(fl, "Y", ["S_pre", "S_post"]); lo, hi = np.percentile(bt[:, 2], [2.5, 97.5])
    out["positive_control"][str(lam)] = dict(b2=float(b[2]), b2_ci=[float(lo), float(hi)], expected_shift=float(lam * df.A_post.sum() / df.S_post.sum()), interval_above_zero=bool(lo > 0))
    print(f"   positive control, a leak of {100 * lam:.0f}% of AORC's own after-midnight rain added to the hours before midnight (78 days): b2 {b[2]:+.4f} [{lo:+.4f}, {hi:+.4f}] (the shift expected from the leak: {lam * df.A_post.sum() / df.S_post.sum():+.4f}); interval above zero: {bool(lo > 0)}")
rng = rng_main

# 2. the factor model
print("\n2. one factor per gauge day: log(A_exp / S_exp) = c0 + c1 share_post, share_post = S_post / (S_exp + S_post); basin-days with S_exp >= 2 mm")
fac = {}
for sname, fr0 in samples.items():
    fr = fr0[(fr0.S_exp >= 2) & (fr0.A_exp > 0)].copy(); fr["lr"] = np.log(fr.A_exp / fr.S_exp); fr["share"] = fr.S_post / (fr.S_exp + fr.S_post)
    b, r2 = ols(fr, "lr", ["share"]); bt = boot(fr, "lr", ["share"]); lo, hi = np.percentile(bt[:, 1], [2.5, 97.5])
    hi_share = fr[fr.share >= 0.3]; lo_share = fr[fr.share < 0.05]
    fac[sname] = dict(n=int(len(fr)), c0=float(b[0]), c1=float(b[1]), c1_ci=[float(lo), float(hi)], r2=r2, mean_share=float(fr.share.mean()), n_share_ge_0_3=int(len(hi_share)), mean_ratio_when_share_ge_0_3=float((hi_share.A_exp.sum() / hi_share.S_exp.sum())) if len(hi_share) else None,
                      mean_ratio_when_share_lt_0_05=float(lo_share.A_exp.sum() / lo_share.S_exp.sum()), median_ratio=float((fr.A_exp / fr.S_exp).median()))
    print(f"  {sname:16s} n={len(fr):4d}: c1 {b[1]:+.3f} [{lo:+.3f}, {hi:+.3f}] (a share of 1 would multiply A_exp by exp(c1)); c0 {b[0]:+.3f} (the ratio {np.exp(b[0]):.3f} at share 0); R2 {r2:.3f}; ratio A/S when the after-midnight share >= 0.3: "
          + (f"{fac[sname]['mean_ratio_when_share_ge_0_3']:.3f} (n={len(hi_share)})" if len(hi_share) else "n/a") + f", when < 0.05: {fac[sname]['mean_ratio_when_share_lt_0_05']:.3f} (n={len(lo_share)})")
out["factor_model"] = fac

# 3. little rain before midnight, much after
print("\n3. basin-days with little Stage IV rain before midnight (S_pre < 0.3 mm) and much after (S_post > 3 mm, to 12 UTC):")
dry = df[(df.S_pre < 0.3) & (df.S_post > 3)]
dry2 = df[(df.S_late < 0.3) & (df.S_post > 3)]
out["dry_before_wet_after"] = dict(n_pre=int(len(dry)), days_pre=int(dry.day.nunique()), mean_S_post=float(dry.S_post.mean()) if len(dry) else None, mean_A_pre=float(dry.A_pre.mean()) if len(dry) else None, max_A_pre=float(dry.A_pre.max()) if len(dry) else None,
                                   share_A_pre_gt_0_5=float((dry.A_pre > 0.5).mean()) if len(dry) else None, n_late=int(len(dry2)), mean_S_post_late=float(dry2.S_post.mean()) if len(dry2) else None, mean_A_late=float(dry2.A_late.mean()) if len(dry2) else None,
                                   max_A_late=float(dry2.A_late.max()) if len(dry2) else None, share_A_late_gt_0_5=float((dry2.A_late > 0.5).mean()) if len(dry2) else None,
                                   cases=dry.sort_values("S_post", ascending=False).head(12)[["day", "gauge", "S_pre", "A_pre", "S_post", "A_post"]].round(2).to_dict("records"))
print(f"   whole local day: {len(dry)} basin-days ({dry.day.nunique()} days): mean S_post {dry.S_post.mean():.1f} mm, mean A_pre {dry.A_pre.mean():.3f} mm, max A_pre {dry.A_pre.max():.2f} mm, share of them with A_pre > 0.5 mm {float((dry.A_pre > 0.5).mean()):.2f}")
print(f"   last six hours (18:00-24:00, S_late < 0.3 mm): {len(dry2)} basin-days: mean S_post {dry2.S_post.mean():.1f} mm, mean A_late {dry2.A_late.mean():.3f} mm, max {dry2.A_late.max():.2f} mm, share with A_late > 0.5 mm {float((dry2.A_late > 0.5).mean()):.2f}")
for r in out["dry_before_wet_after"]["cases"][:8]: print("     ", r)
# the ratio A/S before midnight in the days with the largest after-midnight share
big = df[(df.S_pre >= 2) & (df.S_post >= 5)]
out["big_night_rain"] = dict(n=int(len(big)), median_ratio_pre=float((big.A_pre / big.S_pre).median()), median_ratio_post=float((big.A_post / big.S_post).median()), corr_ratio_pre_with_S_post=float(np.corrcoef(big.A_pre / big.S_pre, big.S_post)[0, 1]) if len(big) > 3 else None)
print(f"   days with at least 2 mm before and 5 mm after midnight: n={len(big)}, median A/S before {out['big_night_rain']['median_ratio_pre']:.3f}, after {out['big_night_rain']['median_ratio_post']:.3f}")

# 4. the calibrations used by the network test (doses): additive lambda on AORC's own post-midnight rain; multiplicative c1 on the share
b = out["regressions"]["both (78 days)|A_pre~S_pre+S_post"]
ratio_post = float(df.A_post.sum() / df.S_post.sum())
cal = dict(ratio_aorc_over_stage_iv_post=ratio_post, b2=b["b2"], b2_ci=b["b2_ci"], lambda_point=b["b2"] / ratio_post, lambda_upper=b["b2_ci"][1] / ratio_post, lambda_lower=b["b2_ci"][0] / ratio_post,
           c1=fac["both (78 days)"]["c1"], c1_ci=fac["both (78 days)"]["c1_ci"])
out["calibration"] = cal
print(f"\n4. calibration for the network test: AORC/Stage IV after midnight {ratio_post:.3f}; additive: lambda = b2 / ratio = {cal['lambda_point']:+.4f} (95% interval {cal['lambda_lower']:+.4f} to {cal['lambda_upper']:+.4f}) of AORC's own rain between local midnight and 12 UTC; "
      f"multiplicative: c1 {cal['c1']:+.3f} ({cal['c1_ci'][0]:+.3f} to {cal['c1_ci'][1]:+.3f})")
df.to_csv(f"{OUT}/a7_basin_days.csv", index=False)
jsave("a7_hindsight.json", out)
