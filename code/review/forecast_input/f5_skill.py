"""E2 review, check (b) continued and (c): the forecast's own skill and the bias factors, recomputed from the reviewer's own day totals (f4_series.npz: my own basin means of the stored box data and my own AORC local days),
against results/gefs_basins.json:  correlation (all days 2000-2014, validation years, test years, by season), mean ratio, the LAG TEST (the raw forecast of local day t+k against the AORC rain of day t+k+s, s = -2..+2:
best at 0?), the BIAS FACTORS (per basin and lead = sum of the AORC rain / sum of the raw forecast over the issue days whose target day lies in 2000-10-01 .. 2005-09-30, recomputed; fitted on those years only; applied to every day),
the BRIER SKILL of days with at least 5 and 10 mm in the test years: yes/no forecast of the member, and a calibrated probability (isotonic map fitted on the VALIDATION years only, 12 basins pooled, per lead and threshold),
with the climatology of the builder (the event frequency of each basin's test years) and, as a stricter reference, a monthly climatology fitted on the validation years; a logistic map as another calibration.
Also the claims 0.72/0.66/0.59 and 0.44/0.39/0.33. About 30 seconds."""
import sys; sys.dont_write_bytecode = True
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from f_common import *

G = jload("results/gefs_basins.json")
Z = np.load(f"{OUT}/f4_series.npz")
ids = [str(g) for g in Z["ids"]]
T = pd.DatetimeIndex(Z["days"]); RAW = {g: Z["raw"][i] for i, g in enumerate(ids)}; AOR = {g: Z["aorc"][i] for i, g in enumerate(ids)}
VAL = ("2000-10-01", "2005-09-30"); TEST = ("2005-10-01", "2014-09-30"); ALL = ("2000-01-01", "2014-09-30")
TG = {k: T + pd.Timedelta(days=k) for k in KS}
inr = lambda k, r: np.asarray((TG[k] >= r[0]) & (TG[k] <= r[1]))
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
S = G["skill"]["summary"]; SK = G["skill"]["per_basin"]
out = {}
# ---- correlation and ratio -------------------------------------------------------------------------------------------------------------------------------------------------
mine = {}
for k in KS:
    c_all = [corr(RAW[g][inr(k, ALL), k - 1], AOR[g][inr(k, ALL), k - 1]) for g in ids]
    c_val = [corr(RAW[g][inr(k, VAL), k - 1], AOR[g][inr(k, VAL), k - 1]) for g in ids]
    c_tst = [corr(RAW[g][inr(k, TEST), k - 1], AOR[g][inr(k, TEST), k - 1]) for g in ids]
    r_all = [RAW[g][inr(k, ALL), k - 1].mean() / AOR[g][inr(k, ALL), k - 1].mean() for g in ids]
    mine[f"l{k}"] = dict(corr_all_median=float(np.median(c_all)), corr_val_median=float(np.median(c_val)), corr_test_median=float(np.median(c_tst)), ratio_all_median=float(np.median(r_all)), corr_all_min_max=[float(min(c_all)), float(max(c_all))],
                         dev_corr_all=max(abs(c - SK[g][f"l{k}"]["correlation_all"]) for c, g in zip(c_all, ids)), dev_ratio_all=max(abs(r - SK[g][f"l{k}"]["mean_ratio_all"]) for r, g in zip(r_all, ids)),
                         dev_corr_test=max(abs(c - SK[g][f"l{k}"]["test_years"]["correlation"]) for c, g in zip(c_tst, ids)), dev_corr_val=max(abs(c - SK[g][f"l{k}"]["validation_years"]["correlation"]) for c, g in zip(c_val, ids)))
    seasons = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    mine[f"l{k}"]["corr_by_season_median"] = {s: float(np.median([corr(RAW[g][inr(k, ALL) & np.isin(TG[k].month, m), k - 1], AOR[g][inr(k, ALL) & np.isin(TG[k].month, m), k - 1]) for g in ids])) for s, m in seasons.items()}
out["skill"] = mine
for k in KS:
    m = mine[f"l{k}"]; j = S["correlation_all_median_over_basins"][f"l{k}"]
    print(f"lead {k}: correlation (all days, median of 12 basins) {m['corr_all_median']:.4f} (JSON {j:.4f}); validation years {m['corr_val_median']:.3f}, test years {m['corr_test_median']:.3f}; mean ratio {m['ratio_all_median']:.3f} (JSON {S['mean_ratio_all_median_over_basins'][f'l{k}']:.3f}); by season {({s: round(v, 2) for s, v in m['corr_by_season_median'].items()})}; largest difference to the per-basin values of the JSON: corr {m['dev_corr_all']:.1e}, ratio {m['dev_ratio_all']:.1e}")
# ---- bias factors --------------------------------------------------------------------------------------------------------------------------------------------------------------
BIAS = {g: {k: float(AOR[g][inr(k, VAL), k - 1].sum() / RAW[g][inr(k, VAL), k - 1].sum()) for k in KS} for g in ids}
bdev = max(abs(BIAS[g][k] - G["bias"]["factors"][g][f"l{k}"]) for g in ids for k in KS)
out["bias"] = dict(factors_mine={g: {f"l{k}": BIAS[g][k] for k in KS} for g in ids}, max_abs_difference_to_the_json=bdev, fitted_on=list(VAL), median_by_lead={f"l{k}": float(np.median([BIAS[g][k] for g in ids])) for k in KS}, range_by_lead={f"l{k}": [float(min(BIAS[g][k] for g in ids)), float(max(BIAS[g][k] for g in ids))] for k in KS},
                  json_rule=G["bias"]["fitted_on_target_days"], largest_factors={f"l{k}": sorted(((round(BIAS[g][k], 3), g) for g in ids), reverse=True)[:3] for k in KS}, smallest_factors={f"l{k}": sorted(((round(BIAS[g][k], 3), g) for g in ids))[:3] for k in KS})
# the factor against the test years: out-of-sample ratio of the bias-corrected forecast
BC = {g: RAW[g] * np.array([BIAS[g][k] for k in KS])[None, :] for g in ids}
out["bias"]["corrected_mean_ratio_val_and_test_median"] = {f"l{k}": [float(np.median([BC[g][inr(k, VAL), k - 1].mean() / AOR[g][inr(k, VAL), k - 1].mean() for g in ids])), float(np.median([BC[g][inr(k, TEST), k - 1].mean() / AOR[g][inr(k, TEST), k - 1].mean() for g in ids]))] for k in KS}
# is the factor stable? fit on the first half of the validation years and on the second, apply to the other half and to the test years (a robustness look, not part of the design)
half = {k: (inr(k, ("2000-10-01", "2003-03-31")), inr(k, ("2003-04-01", "2005-09-30"))) for k in KS}
spread = {f"l{k}": float(np.median([abs(AOR[g][half[k][0], k - 1].sum() / RAW[g][half[k][0], k - 1].sum() - AOR[g][half[k][1], k - 1].sum() / RAW[g][half[k][1], k - 1].sum()) for g in ids])) for k in KS}
out["bias"]["median_difference_between_the_factors_of_the_two_halves_of_the_validation_years"] = spread
print(f"bias factors: recomputed from my own totals, largest difference to the JSON {bdev:.1e}; median {out['bias']['median_by_lead']}; range {({l: [round(x, 3) for x in v] for l, v in out['bias']['range_by_lead'].items()})}; corrected mean ratio (validation, test) {({l: [round(x, 3) for x in v] for l, v in out['bias']['corrected_mean_ratio_val_and_test_median'].items()})}; the factors of two halves of the validation years differ by (median) {({l: round(v, 3) for l, v in spread.items()})}")

# ---- the lag test ---------------------------------------------------------------------------------------------------------------------------------------------------------------
from f_aorc import *
ZONES, _ = gauge_zones(ids)
days_ext = np.arange(np.datetime64("1980-10-01"), np.datetime64("2014-10-04")); d_idx = {d: n for n, d in enumerate(days_ext)}
lag = {}
for g in ids:
    ser, _ = aorc_local_days(g, ZONES[g], days_ext)
    lag[g] = {}
    for k in KS:
        ok_days = inr(k, ALL)
        row = {}
        for s in (-2, -1, 0, 1, 2):
            tgt = (TG[k].values.astype("datetime64[D]") + np.timedelta64(s, "D"))
            sel = ok_days & np.array([(d in d_idx) for d in tgt])
            row[s] = corr(RAW[g][sel, k - 1], ser[[d_idx[d] for d in tgt[sel]]])
        lag[g][k] = row
med_lag = {f"l{k}": {str(s): float(np.median([lag[g][k][s] for g in ids])) for s in (-2, -1, 0, 1, 2)} for k in KS}
best = {g: {f"l{k}": max(lag[g][k], key=lag[g][k].get) for k in KS} for g in ids}
out["lag"] = dict(median_correlation_by_shift=med_lag, best_shift_by_basin_and_lead=best, every_basin_and_lead_best_at_zero=bool(all(v == 0 for b in best.values() for v in b.values())), json_median=S["lag_check"]["median_correlation_by_shift"],
                  max_abs_difference_median_to_json=max(abs(med_lag[l][s] - S["lag_check"]["median_correlation_by_shift"][l]["%+d" % int(s)]) for l in med_lag for s in med_lag[l]))
print(f"lag test: best shift 0 in every basin and lead: {out['lag']['every_basin_and_lead_best_at_zero']}; median correlation at lead 1 by shift {({s: round(v, 3) for s, v in med_lag['l1'].items()})}; lead 2 {({s: round(v, 3) for s, v in med_lag['l2'].items()})}; lead 3 {({s: round(v, 3) for s, v in med_lag['l3'].items()})}; largest difference of the medians to the JSON {out['lag']['max_abs_difference_median_to_json']:.1e}")

# ---- Brier skill ------------------------------------------------------------------------------------------------------------------------------------------------------------------
bs = lambda p, o: float(np.mean((np.asarray(p, float) - o) ** 2))
res = {}
for k in KS:
    for thr in (5.0, 10.0):
        xv = np.concatenate([RAW[g][inr(k, VAL), k - 1] for g in ids]); yv = np.concatenate([(AOR[g][inr(k, VAL), k - 1] >= thr).astype(float) for g in ids])
        iso = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(xv, yv)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log1p(xv)[:, None], yv)
        # a monthly climatology from the validation years
        mon_v = np.concatenate([TG[k][inr(k, VAL)].month.values for g in ids]); clim_m = {m: float(yv[mon_v == m].mean()) for m in range(1, 13)}
        per = []
        n_tot = 0; sums = dict(clim_basin=0.0, clim_pooled=0.0, yn=0.0, cal=0.0, log=0.0, clim_month=0.0)
        obs_all = []
        for g in ids:
            sel = inr(k, TEST); f = RAW[g][sel, k - 1]; o = (AOR[g][sel, k - 1] >= thr).astype(float); n = len(o); obar = o.mean()
            mm = TG[k][sel].month.values; pm = np.array([clim_m[m] for m in mm])
            row = dict(n=n, obar=obar, clim=obar * (1 - obar), yn=bs((f >= thr).astype(float), o), cal=bs(iso.predict(f), o), log=bs(lr.predict_proba(np.log1p(f)[:, None])[:, 1], o), clim_month=bs(pm, o))
            per.append(row); n_tot += n
            for key, v in (("clim_basin", row["clim"]), ("yn", row["yn"]), ("cal", row["cal"]), ("log", row["log"]), ("clim_month", row["clim_month"])): sums[key] += v * n
            obs_all.append(o)
        obar_pooled = float(np.concatenate(obs_all).mean())
        clim_b = sums["clim_basin"] / n_tot; clim_p = obar_pooled * (1 - obar_pooled)
        rec = dict(n_basin_days=n_tot, event_frequency=obar_pooled, bs_clim_builder_way=clim_b, bs_clim_pooled_frequency=clim_p,
                   bss_yes_no_pooled=1 - sums["yn"] / n_tot / clim_b, bss_calibrated_isotonic_pooled=1 - sums["cal"] / n_tot / clim_b, bss_calibrated_isotonic_pooled_freq_ref=1 - sums["cal"] / n_tot / clim_p, bss_calibrated_logistic_pooled=1 - sums["log"] / n_tot / clim_b,
                   bss_calibrated_isotonic_against_monthly_climatology=1 - sums["cal"] / sums["clim_month"], bss_median_over_basins_isotonic=float(np.median([1 - r["cal"] / r["clim"] for r in per])), bss_median_over_basins_yes_no=float(np.median([1 - r["yn"] / r["clim"] for r in per])),
                   isotonic_steps=int(len(np.unique(iso.predict(np.linspace(0, 100, 2001))))), validation_event_frequency=float(yv.mean()))
        res[f"l{k}_{thr:g}mm"] = rec
        j = S["brier_test_years_pooled"][f"l{k}"][f"{thr:g}mm"]
        print(f"lead {k}, >= {thr:g} mm: event frequency {rec['event_frequency']:.3f} (JSON {j['event_frequency']:.3f}); Brier skill pooled, yes/no {rec['bss_yes_no_pooled']:.3f} (JSON {j['yes_no_raw']['brier_skill_pooled']:.3f}); calibrated isotonic {rec['bss_calibrated_isotonic_pooled']:.3f} (JSON {j['calibrated_probability']['brier_skill_pooled']:.3f}); "
              f"calibrated logistic {rec['bss_calibrated_logistic_pooled']:.3f}; against a monthly climatology {rec['bss_calibrated_isotonic_against_monthly_climatology']:.3f}; pooled-frequency reference {rec['bss_calibrated_isotonic_pooled_freq_ref']:.3f}; median over basins {rec['bss_median_over_basins_isotonic']:.3f} (JSON {j['calibrated_probability']['brier_skill_median_over_basins']:.3f})")
out["brier"] = res
out["claims_of_the_brief"] = dict(correlation=[round(mine[f"l{k}"]["corr_all_median"], 2) for k in KS], calibrated_brier_5mm=[round(res[f"l{k}_5mm"]["bss_calibrated_isotonic_pooled"], 2) for k in KS], claimed_correlation=[0.72, 0.66, 0.59], claimed_brier=[0.44, 0.39, 0.33])
print("claims: correlation", out["claims_of_the_brief"]["correlation"], "(claimed 0.72/0.66/0.59); calibrated Brier skill 5 mm", out["claims_of_the_brief"]["calibrated_brier_5mm"], "(claimed 0.44/0.39/0.33)")
jsave("f5_skill.json", out)
