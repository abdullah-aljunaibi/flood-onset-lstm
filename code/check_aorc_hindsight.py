"""Does the AORC rain of the hours before local midnight carry hindsight (information about the rain that falls after midnight)? A check beside 16_aorc_rain_to_midnight.py.

AORC is an analysis made after the fact: according to the summary of its methods in results/data_access.json its hours are tied to daily gauge totals, and a gauge day does not end at local midnight (for the CPC daily gauge analysis behind NLDAS-2 the day
runs from 12:00 UTC to 12:00 UTC, which is 07:00 or 08:00 local time in the eastern United States; the day of the gauges behind AORC was not looked up). If an hour before midnight is scaled with the total of such a day, its value holds a
little of the rain that falls between midnight and the morning of the next day - rain that nobody knows at the issue time of the forecast (local midnight). The clock proof of 16_aorc_rain_to_midnight.py shows which hours are
summed; it cannot show this.
This script puts the AORC hours against the hourly Stage IV files (the radar analysis that AORC follows from 2002 on, available about an hour after the hour: what a real-time service would read) on the 50 stormiest days of
the test years (the days with the most ERA5 rain over the 12 basins, at least 4 days apart; the choice uses no flood or forecast score). For every basin and every such day (local day t, by the wall clock of the gauge) it takes
  A_pre, S_pre : the AORC and the Stage IV rain from 00:00 to 24:00 of day t (the input of the model),
  A_post, S_post: the same over the 7 hours after midnight (00:00 to 07:00 of day t+1: what is not known at the issue time),
and asks
 1. how large the AORC rain is against the Stage IV rain before and after midnight (the bias correction of AORC);
 2. whether the ratio A/S before midnight goes with the ratio after midnight (one factor for the whole gauge day), and with S_post;
 3. whether A_pre depends on S_post once S_pre is given: A_pre = b0 + b1 S_pre + b2 S_post over all basin-days of the sample, with the 95% interval of b2 from drawing the days again (a bootstrap over days, so
    that the basins of one storm count once). Without hindsight b2 is near zero (the Stage IV rain after midnight cannot be in an AORC hour before it, except through the persistence of storms that S_pre already holds).
Writes results/aorc_hindsight.json and data/aorc/stage_iv_hindsight.csv (the hourly Stage IV basin means, 50 days x 36 hours; the raw files are not kept). Needs the AORC series (code/fetch_aorc.py) and the network.
A second part looks at the hour stamps across 2002 (the year from which AORC's hours follow the radar analysis): the hourly AORC basin means against the hourly MERRA-2 rain (NASA POWER, cached by
check_day_windows.py in data/hourly; its key is the START of the hour, so the hour ends one hour after it) at lags of -3 to +3 hours, year by year from 2001 (the first year of that file) to 2013, pooled over the basins:
if the stamp convention of AORC changed at 2002 (the disaggregated hours before it and the radar hours after it), the best lag of 2001 would differ from that of the later years.
It does not retrain anything and says nothing about the forecast scores: it measures the input.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import fetch_aorc as AO
from common import DATA, RES

N_DAYS, MIN_APART, SEED = int(os.environ.get("HINDSIGHT_DAYS", 50)), 4, 0                                   # another number of days is a smoke test: its files go to results/quick
TEST = ("2005-10-02", "2014-09-28")                                              # the test years, without their first and last day so that every window lies inside the record
CSV = DATA / "aorc" / ("stage_iv_hindsight.csv" if N_DAYS == 50 else f"stage_iv_hindsight_smoke_{N_DAYS}.csv")
OUT = (RES if N_DAYS == 50 else RES / "quick") / "aorc_hindsight.json"
OUT.parent.mkdir(exist_ok=True)
t_start = time.time()
ids = AO.ids_of_the_study()
DW = json.loads((RES / "day_windows.json").read_text())
ZONE = {-5: "America/New_York", -6: "America/Chicago"}
zone_of = {g: ZONE[int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"])] for g in ids}

# ---- the stormiest days of the test years, by ERA5 (no flood or forecast score is used) -------------------------------------------------------------------
tot = None
for g in ids:
    e = json.loads((DATA / "hourly" / f"era5_{g}.json").read_text())
    s = pd.Series(np.array(e["hourly"]["precipitation"], float), index=pd.DatetimeIndex(pd.to_datetime(e["hourly"]["time"])))
    d = s.groupby((s.index - pd.Timedelta(minutes=1)).normalize()).sum()
    tot = d if tot is None else tot + d
tot = tot.loc[TEST[0]:TEST[1]] / len(ids)
days = []
for d, v in tot.sort_values(ascending=False).items():
    if all(abs((d - p).days) >= MIN_APART for p in days):
        days.append(d)
    if len(days) == N_DAYS:
        break
days = sorted(days)

# ---- the hourly Stage IV basin means for 03:00 UTC of day t to 14:00 UTC of day t+1 (covers 00:00 of day t to 07:00 of day t+1 in both zones, summer time included) ------------------
labels = sorted({t for d in days for t in pd.date_range(d + pd.Timedelta(hours=3), d + pd.Timedelta(hours=38), freq="h")})
if CSV.exists():
    st4 = pd.read_csv(CSV, index_col=0, parse_dates=True)
else:
    print(f"{len(labels)} hourly Stage IV files for {len(days)} days", flush=True)
    st4 = AO.stage_iv_basin_hourly(ids, labels)
    st4.to_csv(CSV)

# ---- the table: one row per basin and day ----------------------------------------------------------------------------------------------------------------------------
rows = []
for g in ids:
    stamp, x = AO.load_series(g)
    aorc = pd.Series(x, index=stamp)
    for d in days:
        a0 = pd.Timestamp(d).tz_localize(zone_of[g]).tz_convert("UTC").tz_localize(None)                              # 00:00 of day t on the wall clock, as a UTC time
        a1 = pd.Timestamp(d + pd.Timedelta(days=1)).tz_localize(zone_of[g]).tz_convert("UTC").tz_localize(None)       # 24:00 of day t: the issue time
        h = pd.date_range(a0 + pd.Timedelta(hours=1), a1 + pd.Timedelta(hours=7), freq="h")                          # the hours that END after 00:00 of day t and by 07:00 of day t+1
        if not all(t in st4.index for t in h) or st4.loc[h, g].isna().any():
            continue
        pre = h <= a1
        sx, ax = st4.loc[h, g], aorc.reindex(h)
        rows.append(dict(day=str(d.date()), gauge=g, A_pre=float(ax[pre].sum()), S_pre=float(sx[pre].sum()), A_post=float(ax[~pre].sum()), S_post=float(sx[~pre].sum())))
df = pd.DataFrame(rows)
assert len(df) > 0.8 * len(ids) * len(days), f"only {len(df)} of {len(ids) * len(days)} basin-days have a complete Stage IV window"

# ---- the numbers -------------------------------------------------------------------------------------------------------------------------------------------------------
rain = df[(df.S_pre > 2) & (df.S_post > 1)].copy()                                                                        # days with rain on both sides of midnight, for the ratios
rain["r_pre"], rain["r_post"] = rain.A_pre / rain.S_pre, rain.A_post / rain.S_post
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def coefficients(frame):
    """b0, b1, b2 of A_pre = b0 + b1 S_pre + b2 S_post, and R^2 of the fit with and without S_post."""
    y = frame.A_pre.values
    X1, X0 = np.column_stack([np.ones(len(frame)), frame.S_pre, frame.S_post]), np.column_stack([np.ones(len(frame)), frame.S_pre])
    b1, b0 = np.linalg.lstsq(X1, y, rcond=None)[0], np.linalg.lstsq(X0, y, rcond=None)[0]
    r2 = lambda X, b: float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return b1, r2(X1, b1), r2(X0, b0)


b, r2_with, r2_without = coefficients(df)
rng = np.random.default_rng(SEED)
by_day = {d: f for d, f in df.groupby("day")}
boot = []
for _ in range(2000):
    pick = rng.choice(list(by_day), size=len(by_day), replace=True)
    boot.append(coefficients(pd.concat([by_day[d] for d in pick]))[0])
boot = np.array(boot)
lo, hi = np.percentile(boot[:, 2], [2.5, 97.5])
dry = df[(df.S_pre < 0.3) & (df.S_post > 3)]                                                                                  # dry before midnight in Stage IV, wet after it: rain that AORC could only show before midnight by hindsight
numbers = dict(
    n_days=len(days), days=[str(d.date()) for d in days], n_basin_days=int(len(df)), n_basin_days_with_rain_on_both_sides=int(len(rain)),
    aorc_over_stage_iv=dict(before_midnight=float(df.A_pre.sum() / df.S_pre.sum()), seven_hours_after_midnight=float(df.A_post.sum() / df.S_post.sum())),
    dry_before_midnight_wet_after=dict(definition="Stage IV rain before midnight below 0.3 mm and in the 7 hours after midnight above 3 mm", n_basin_days=int(len(dry)),
                                       mean_stage_iv_rain_after_mm=float(dry.S_post.mean()) if len(dry) else None, mean_aorc_rain_before_mm=float(dry.A_pre.mean()) if len(dry) else None,
                                       share_with_more_than_half_a_mm_of_aorc_rain_before=float((dry.A_pre > 0.5).mean()) if len(dry) else None),
    ratio_before_midnight=dict(median=float(rain.r_pre.median()), p10=float(rain.r_pre.quantile(.1)), p90=float(rain.r_pre.quantile(.9))),
    corr_of_the_logs_of_the_ratios_before_and_after_midnight=corr(np.log(rain.r_pre), np.log(rain.r_post)),
    corr_of_the_ratio_before_midnight_with_stage_iv_rain_after_midnight=corr(rain.r_pre, rain.S_post),
    corr_of_the_ratio_before_midnight_with_stage_iv_rain_before_midnight=corr(rain.r_pre, rain.S_pre),
    regression=dict(form="A_pre = b0 + b1 S_pre + b2 S_post (mm, one row per basin and day)", b0=float(b[0]), b1=float(b[1]), b2=float(b[2]), b2_95_interval_bootstrap_over_days=[float(lo), float(hi)], r2_with_s_post=r2_with,
                    r2_without_s_post=r2_without, mean_s_post_mm=float(df.S_post.mean()), mean_a_pre_mm=float(df.A_pre.mean()), mean_s_pre_mm=float(df.S_pre.mean())))
b2 = numbers["regression"]["b2"]
reading = (f"On the {len(days)} stormiest days of the test years, AORC holds {numbers['aorc_over_stage_iv']['before_midnight']:.2f} times the Stage IV rain before local midnight and "
           f"{numbers['aorc_over_stage_iv']['seven_hours_after_midnight']:.2f} times in the 7 hours after it. Given the Stage IV rain before midnight, each mm of Stage IV rain in the 7 hours after midnight goes with "
           f"{b2:+.3f} mm more AORC rain before midnight (95% interval {lo:+.3f} to {hi:+.3f}, drawing the days again); the share of the variance of A_pre explained rises from {r2_without:.3f} to {r2_with:.3f} when S_post is added. "
           + ("The interval excludes zero: the AORC hours before midnight hold some hindsight of the rain after midnight; its size relative to the AORC rain of those hours is "
              f"{100 * b2 * numbers['regression']['mean_s_post_mm'] / numbers['regression']['mean_a_pre_mm']:.1f}% on average over this sample of storm days (b2 x mean S_post / mean A_pre)."
              if lo > 0 or hi < 0 else "The interval includes zero: this sample shows no dependence of the AORC rain before midnight on the Stage IV rain after midnight beyond what the Stage IV rain before midnight holds."))
# ---- the stamps across 2002: AORC against the hourly MERRA-2 rain, year by year ----------------------------------------------------------------------------------------------
mer = {}
for g in ids:
    j = json.loads((DATA / "hourly" / f"merra_{g}.json").read_text())
    p = j["properties"]["parameter"]["PRECTOTCORR"]
    v = np.array(list(p.values()), float)
    v[v < -900] = np.nan
    mer[g] = pd.Series(v, index=pd.to_datetime(list(p.keys()), format="%Y%m%d%H") + pd.Timedelta(hours=1))             # the key is the START of the hour: the hour ends one hour later
aorc_all = {g: pd.Series(AO.load_series(g)[1], index=AO.load_series(g)[0]) for g in ids}
by_year = {}
for y in range(2001, 2014):
    prof = {}
    for k in range(-3, 4):
        a_all, m_all = [], []
        for g in ids:
            a = aorc_all[g].loc[f"{y}-01-01":f"{y}-12-31"]
            m = mer[g].reindex(a.index).shift(k)
            ok = m.notna().values
            a_all.append(a.values[ok])
            m_all.append(m.values[ok])
        prof[str(k)] = float(np.corrcoef(np.concatenate(a_all), np.concatenate(m_all))[0, 1])
    by_year[str(y)] = dict(correlation_by_lag=prof, best_lag=int(max(prof, key=prof.get)))
best = {y: v["best_lag"] for y, v in by_year.items()}
gaps = [v["correlation_by_lag"][str(v["best_lag"])] - max(v["correlation_by_lag"][str(v["best_lag"] - 1)], v["correlation_by_lag"][str(v["best_lag"] + 1)]) for v in by_year.values() if abs(v["best_lag"]) < 3]   # the peak above its neighbours
stamps_across_2002 = dict(
    what="AORC basin-mean hourly rain at stamp T against the MERRA-2 hourly rain (PRECTOTCORR, hour ending at T-k hours), pooled over the 12 basins, one value per lag and year; k = 0: the two stamps are the end of the same hour",
    by_year=by_year, best_lag_by_year=best,
    reading=(f"The best lag is 0 in {sum(v == 0 for v in best.values())} of the {len(best)} years from 2001 to 2013 (by year: " + ", ".join(f"{y}: {v:+d}" for y, v in best.items()) + "); the curves are flat (the peak stands "
             f"{min(gaps):.3f} to {max(gaps):.3f} above its neighbours), because MERRA-2 is a 50 km model rain whose timing is itself uncertain, so this separates a whole-hour shift only when it is systematic. "
             + ("The best lag of 2001 (before the radar hours begin) equals that of the years after 2002, so these data show no change of the stamp convention at 2002." if best["2001"] == best["2002"] == best["2004"] == best["2008"] else
                "The best lag of 2001 differs from that of later years: the stamp convention may differ before 2002.")))
print(stamps_across_2002["reading"])
# ---- the limits of this sample, with the numbers (added after the independent review) -----------------------------------------------------------------------------------------
EVENTS = pd.read_csv(RES / "flood_events.csv", dtype={"gauge_id": str})
n_onsets_in_sample = int((pd.to_datetime(EVENTS.onset_date) - pd.Timedelta(days=1)).isin(days).sum())          # onsets whose issue day (onset date - 1) is one of the days
LIMIT_SAMPLE = (f"the sample of {len(days)} stormiest days holds little rain after midnight (mean Stage IV rain {df.S_post.mean():.1f} mm in the 7 hours after midnight against {df.S_pre.mean():.1f} mm before; "
                f"{int((df.S_post > 3).sum())} basin-days above 3 mm; {len(dry)} dry-before/wet-after cases; the issue days of {n_onsets_in_sample} of the {len(EVENTS)} onsets are among its days), so its interval "
                f"({lo:+.2f} to {hi:+.2f}) mostly reflects how little night rain it contains and does not exclude a leak of up to {100 * hi:.0f}% of the rain after midnight")
try:
    REV = json.loads((RES / "review_aorc_rain_to_midnight.json").read_text())
    Hr = REV["numbers"]["hindsight"]
    reg_r = Hr["regressions"]["both (78 days)|A_pre~S_pre+S_post"]
    d3_r = next(c for c in REV["checks"] if c.get("id") == "d3")["numbers"]["sample"]
    n_add, n_all = len(Hr["selected_days"]["days"]), len(Hr["selected_days"]["days"]) + len(Hr["selected_days"]["builder_days"])
    LIMIT_REVIEW = (f"the independent review (results/review_aorc_rain_to_midnight.json) added {n_add} days chosen by onsets and by night storms, which hold {d3_r['mean_stage_iv_mm_after_midnight_mine']:.1f} mm of Stage IV rain in the 7 hours "
                    f"after midnight on average ({Hr['dry_before_wet_after']['n_pre']} dry-before/wet-after basin-days in the {n_all} days together), and found no leak: coefficient {reg_r['b2']:+.3f}, 95% interval "
                    f"{reg_r['b2_ci'][0]:+.3f} to {reg_r['b2_ci'][1]:+.3f} on {reg_r['n']} basin-days")
except (OSError, KeyError, StopIteration, TypeError, ValueError):
    LIMIT_REVIEW = "no independent review of this check was found (results/review_aorc_rain_to_midnight.json)"
R = dict(question="Does the AORC rain before local midnight carry information about the rain that falls after midnight (hindsight of a retrospective analysis)?", method="see the header of code/check_aorc_hindsight.py",
         sample=dict(n_days=N_DAYS, min_days_apart=MIN_APART, period=TEST, choice="the days with the most pooled ERA5 rain over the 12 basins (no flood or forecast score)", windows="local day t from 00:00 to 24:00 on the wall clock of each gauge, and the 7 hours after it"),
         numbers=numbers, reading=reading, table_head=df.head(10).round(2).to_dict(orient="records"),
         limits=["the days of the test years are a sample of storms, not of all days; the days are the stormiest by ERA5 rain, so the relation is measured where it matters for floods and says little about dry days",
                 LIMIT_SAMPLE, LIMIT_REVIEW,
                 "Stage IV is itself a product made with gauge quality control; it is the nearest real-time analogue available, not the truth; the Iowa Mesonet archive may hold first postings or later versions of each hour (not established)",
                 "a dependence of A_pre on S_post can also come from storms that grow after midnight in the same way the radar saw them before it; the regression controls for S_pre only, so b2 is an upper estimate of hindsight",
                 "this script measures the input (the rain of the hours before midnight), not the forecast scores: the effect on the forecasts is measured elsewhere, by the independent review on the issue days of 171 of the 611 onsets (Stage IV in place of AORC, "
                 "the saved networks) and by code/check_stage4_realtime.py on the issue day of every onset (results/stage4_realtime.json, when it has been run)"],
         stamps_across_2002=stamps_across_2002, runtime_s=round(time.time() - t_start, 1))
OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
print(reading)
print(json.dumps(numbers, indent=1)[:2500])
