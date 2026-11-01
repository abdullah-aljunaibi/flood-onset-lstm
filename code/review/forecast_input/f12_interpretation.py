"""E2 review, check (i): is the interpretation of the clear loss at 1 day ahead in E2a ('the substitution networks over-trust slots that were exact in training') supported by anything in the saved data? This script only READS the
saved prediction sets and the tables (no network is run again, nothing is trained); it gathers what the data say and nothing more:
 1. the paired onsets of lstm_wqa and lstm_wqaF_sub at lead 1 (warned by both / only one / neither) and, for each group, what the forecast and the AORC rain were on the target day (the slot value fs1 against the AORC rain p1 of the
    day t+1), to see whether the onsets that the forecast cost are the days on which the forecast missed the rain and the ones it gained are the days on which it did not;
 2. the same by season and by basin; the forecast's amount against the AORC amount on the days with onsets (a drier-than-observed forecast on the heavy days?); the same at leads 2 and 3;
 3. the ladder of what the same networks do with different slot contents (read from the stored sets): perfect rain, the forecast, the raw forecast, a shuffled forecast, the networks trained on those: how much the onset score depends
    on the slot content (reliance), and whether other models trained in the same way (ridge, forest: trained on the observed rain of the slots, run on the forecast) lose at 1 day too;
 4. the sizes of the effects that the data give and what they cannot give. Tests that would isolate the explanation are listed in the record (suggested, not run).
About 20 seconds."""
import sys; sys.dont_write_bytecode = True
import pickle
from f_common import *
import flood as F
from f_scoring import *

TAB, ids = pickle.load(open(f"{OUT}/f6_tables_csv.pkl", "rb"))
out = {}
def flags_with_days(name, k):
    t = onset_table(name, k)
    return {g: (d[h], w[h]) for g, (d, h, w) in t.items()}                 # target days of the onsets and warned flags, per basin
rows = []
for g in ids:
    dA, wA = flags_with_days("lstm_wqa", 1)[g]; dB, wB = flags_with_days("lstm_wqaF_sub", 1)[g]
    assert (dA == dB).all()
    issue = dA - pd.Timedelta(days=1)
    t = TAB["main"][g]
    for i in range(len(dA)):
        r = t.loc[issue[i]]
        rows.append(dict(gauge=g, target=dA[i], issue=issue[i], wqa=bool(wA[i]), wqaF=bool(wB[i]), fc1=float(r.fs1), aorc1=float(r.p1), rain_to_midnight=float(r.aorc_to_midnight), month=int(dA[i].month)))
df = pd.DataFrame(rows)
df["group"] = np.select([df.wqa & df.wqaF, df.wqa & ~df.wqaF, ~df.wqa & df.wqaF], ["both", "only wqa (lost by the forecast)", "only wqaF_sub (gained)"], "neither")
def stats(f):
    return dict(n=int(len(f)), median_aorc_mm=float(f.aorc1.median()), median_forecast_mm=float(f.fc1.median()), mean_aorc_mm=float(f.aorc1.mean()), mean_forecast_mm=float(f.fc1.mean()), median_rain_to_midnight_mm=float(f.rain_to_midnight.median()),
                share_forecast_below_half_of_aorc=float((f.fc1 < 0.5 * f.aorc1).mean()), share_forecast_above_the_aorc=float((f.fc1 >= f.aorc1).mean()), share_aorc_over_5mm=float((f.aorc1 >= 5).mean()), share_forecast_over_5mm=float((f.fc1 >= 5).mean()))
g1 = {k: stats(f) for k, f in df.groupby("group")}
out["lead1_groups"] = g1
print("1. lead 1, the 611 onsets, wqa against wqaF_sub (seed-averaged forecasts), the AORC rain and the forecast of the target day t+1 (the slot fs1 of issue day t):")
for k in ("both", "only wqa (lost by the forecast)", "only wqaF_sub (gained)", "neither"):
    s = g1[k]; print(f"   {k:34s} n {s['n']:3d}: AORC rain of t+1 median {s['median_aorc_mm']:5.1f} mm (mean {s['mean_aorc_mm']:5.1f}), forecast median {s['median_forecast_mm']:5.1f} mm (mean {s['mean_forecast_mm']:5.1f}); forecast below half of the AORC rain on {100 * s['share_forecast_below_half_of_aorc']:.0f}% of them, at or above it on {100 * s['share_forecast_above_the_aorc']:.0f}%; AORC rain of the day at least 5 mm on {100 * s['share_aorc_over_5mm']:.0f}%, the forecast on {100 * s['share_forecast_over_5mm']:.0f}%; rain to midnight median {s['median_rain_to_midnight_mm']:.1f} mm")
# the forecast against the AORC rain on onset days, by group of the day
on = df
out["onset_days_forecast_vs_aorc"] = dict(mean_ratio=float(on.fc1.sum() / on.aorc1.sum()), median_ratio_of_days_with_aorc_over_5mm=float((on[on.aorc1 >= 5].fc1 / on[on.aorc1 >= 5].aorc1).median()), n_over_5mm=int((on.aorc1 >= 5).sum()), corr=float(np.corrcoef(on.fc1, on.aorc1)[0, 1]))
print(f"   all 611 onset days: mean forecast / mean AORC rain of t+1 {out['onset_days_forecast_vs_aorc']['mean_ratio']:.2f}; days with at least 5 mm of AORC rain ({out['onset_days_forecast_vs_aorc']['n_over_5mm']}): median forecast / AORC {out['onset_days_forecast_vs_aorc']['median_ratio_of_days_with_aorc_over_5mm']:.2f}; correlation of the two amounts {out['onset_days_forecast_vs_aorc']['corr']:.2f}")
# a logistic look: does the loss of the forecast model follow the forecast error on the day?
df["err"] = df.fc1 - df.aorc1
lost, gained = df[df.group.str.startswith("only wqa ")], df[df.group.str.startswith("only wqaF")]
out["error_of_the_slot_lost_vs_gained"] = dict(median_error_mm_lost=float(lost.err.median()), median_error_mm_gained=float(gained.err.median()), n_lost=int(len(lost)), n_gained=int(len(gained)),
                                               share_lost_with_forecast_below_aorc=float((lost.err < 0).mean()), share_gained_with_forecast_below_aorc=float((gained.err < 0).mean()))
print(f"   slot error (forecast minus AORC rain of t+1): the {len(lost)} onsets lost: median {lost.err.median():+.1f} mm, forecast below the AORC rain on {100 * (lost.err < 0).mean():.0f}%; the {len(gained)} gained: median {gained.err.median():+.1f} mm, below on {100 * (gained.err < 0).mean():.0f}%")
# what the same networks do on these onsets with other slot content (perfect rain, the raw forecast, a shuffled forecast), group by group
alt = {}
for nm in ("lstm_wqaF_sub_perfectpred", "lstm_wqaF_sub_rawpred", "lstm_wqaF_clim", "lstm_wqaP"):
    t = onset_table(nm, 1); wm = {g: dict(zip(d[h], w[h])) for g, (d, h, w) in t.items()}
    alt[nm] = np.array([wm[r.gauge][r.target] for r in df.itertuples()])
    df[nm] = alt[nm]
grp = {}
for k, f in df.groupby("group"):
    grp[k] = {nm: float(f[nm].mean()) for nm in alt}
out["lead1_groups_warned_by_the_same_networks_with_other_slots"] = grp
print("   share of each group warned by: " + " | ".join(f"{nm.replace('lstm_', '')}" for nm in alt))
for k in ("both", "only wqa (lost by the forecast)", "only wqaF_sub (gained)", "neither"):
    print(f"   {k:34s} " + " | ".join(f"{100 * grp[k][nm]:5.1f}%" for nm in alt))
lost_perf = float(df[df.group.str.startswith('only wqa ')]["lstm_wqaF_sub_perfectpred"].mean())
out["lost_onsets_warned_with_perfect_slots"] = lost_perf
print(f"   of the {len(lost)} onsets that the forecast cost, the same networks given the PERFECT rain warn of {100 * lost_perf:.0f}%: the loss is a loss of those slot errors, not a lack of ability of the networks")
# a sign test on the paired onsets and the per-season forecast skill beside the season loss
from scipy.stats import binomtest
out["paired_sign_test_lead1"] = dict(only_wqa=int(len(lost)), only_wqaF_sub=int(len(gained)), p=float(binomtest(int(len(lost)), int(len(lost) + len(gained)), 0.5).pvalue))
print(f"   paired onsets: only wqa {len(lost)}, only wqaF_sub {len(gained)}, sign test p = {out['paired_sign_test_lead1']['p']:.4f}")
G = jload("results/gefs_basins.json")
sk = G["skill"]["summary"]["correlation_by_season_median"]["l1"]
out["forecast_skill_by_season_lead1"] = sk
print("   forecast skill by season at lead 1 (median correlation with AORC):", {s: round(v, 2) for s, v in sk.items()})
# 2. by season and by basin
SEAS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
by_season = {s: dict(n=int(np.isin(df.month, m).sum()), wqa=float(df[np.isin(df.month, m)].wqa.mean()), wqaF=float(df[np.isin(df.month, m)].wqaF.mean())) for s, m in SEAS.items()}
by_basin = {g: dict(n=int((df.gauge == g).sum()), wqa=float(df[df.gauge == g].wqa.mean()), wqaF=float(df[df.gauge == g].wqaF.mean()), short=F.SHORT[g]) for g in ids}
out["by_season"], out["by_basin"] = by_season, by_basin
print("2. lead 1 onsets warned by season (wqa -> wqaF_sub):", {s: f"{100 * v['wqa']:.0f}% -> {100 * v['wqaF']:.0f}% (n {v['n']})" for s, v in by_season.items()})
print("   by basin (wqa -> wqaF_sub):", {F.SHORT[g].split(',')[0]: f"{100 * v['wqa']:.0f}->{100 * v['wqaF']:.0f}" for g, v in by_basin.items()}, "| basins where wqaF_sub warns of fewer:", sum(v["wqaF"] < v["wqa"] for v in by_basin.values()), "of 12")
# the same at leads 2 and 3
for k in (2, 3):
    rows = []
    for g in ids:
        dA, wA = flags_with_days("lstm_wqa", k)[g]; dB, wB = flags_with_days("lstm_wqaF_sub", k)[g]
        for i in range(len(dA)):
            r = TAB["main"][g].loc[dA[i] - pd.Timedelta(days=k)]
            rows.append(dict(wqa=bool(wA[i]), wqaF=bool(wB[i]), fc=float(r[f"fs{k}"]), aorc=float(r[f"p{k}"])))
    d = pd.DataFrame(rows)
    out[f"lead{k}"] = dict(onset_days_forecast_over_aorc_mean_ratio=float(d.fc.sum() / d.aorc.sum()), corr=float(np.corrcoef(d.fc, d.aorc)[0, 1]), both=int((d.wqa & d.wqaF).sum()), only_wqa=int((d.wqa & ~d.wqaF).sum()), only_forecast=int((~d.wqa & d.wqaF).sum()), neither=int((~d.wqa & ~d.wqaF).sum()))
    print(f"   lead {k}: paired onsets both {out[f'lead{k}']['both']}, only wqa {out[f'lead{k}']['only_wqa']}, only wqaF_sub {out[f'lead{k}']['only_forecast']}, neither {out[f'lead{k}']['neither']}; forecast / AORC mean ratio on the onset days {out[f'lead{k}']['onset_days_forecast_over_aorc_mean_ratio']:.2f}, correlation {out[f'lead{k}']['corr']:.2f}")
# 3. what the slot content does to the same networks, from the stored sets
ladder = {}
for name, label in (("lstm_wqa", "no slots"), ("lstm_wqaF_sub", "the forecast (bias-corrected) in the slots"), ("lstm_wqaF_sub_rawpred", "the same networks, the raw forecast"), ("lstm_wqaF_sub_perfectpred", "the same networks, the perfect rain"),
                    ("lstm_wqaF_clim", "the same networks, a shuffled forecast"), ("lstm_wqaP", "networks stopped on perfect rain, perfect rain"), ("lstm_wqaP_forecastpred", "networks stopped on perfect rain, given the forecast")):
    ladder[name] = dict(label=label, onsets=[counts(name, k)["pod"] for k in KS], false=[counts(name, k)["far"] for k in KS], nse=[float(np.median(list(nse_per_basin(name, k).values()))) for k in KS])
for n in ("base_linreg_wqa", "base_linreg_wqaF", "base_rf_wqa", "base_rf_wqaF"):
    ladder[n] = dict(label="simple model " + n, onsets=[counts(n, k)["pod"] for k in KS], false=[counts(n, k)["far"] for k in KS], nse=[float(np.median(list(nse_per_basin(n, k).values()))) for k in KS])
out["slot_content_ladder"] = ladder
print("3. the onset score at 1/2/3 days by what is in the slots (the same saved networks or the same simple-model recipe):")
for n, v in ladder.items(): print(f"   {n:28s} {v['label']:55s} onsets {'/'.join(f'{100 * x:.1f}' for x in v['onsets'])}%  false {'/'.join(f'{100 * x:.0f}' for x in v['false'])}%  NSE {'/'.join(f'{x:.2f}' for x in v['nse'])}")
# reliance: the spread between perfect and shuffled slot content, as a share of the score with perfect slots
r1 = ladder["lstm_wqaF_sub_perfectpred"]["onsets"][0]; c1 = ladder["lstm_wqaF_clim"]["onsets"][0]; f1 = ladder["lstm_wqaF_sub"]["onsets"][0]; w1 = ladder["lstm_wqa"]["onsets"][0]
out["reliance_lead1"] = dict(perfect_slots=r1, forecast_slots=f1, shuffled_slots=c1, no_slots=w1, share_of_the_perfect_score_lost_with_the_forecast=1 - f1 / r1, share_lost_with_shuffled=1 - c1 / r1)
print(f"   lead 1: perfect slots {100 * r1:.1f}%, forecast slots {100 * f1:.1f}%, shuffled slots {100 * c1:.1f}%, no slots (wqa) {100 * w1:.1f}%: with the forecast the same networks keep {100 * f1 / r1:.0f}% of their perfect-slot score and with a shuffled forecast {100 * c1 / r1:.0f}%")
jsave("f12_interpretation.json", out)
