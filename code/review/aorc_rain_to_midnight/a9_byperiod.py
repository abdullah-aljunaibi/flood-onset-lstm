"""AORC review, check (e): the before/after 2009-07-17 split flagged by the builder (gain over ERA5: median NSE -0.094 before, +0.136 after; onsets stable), recomputed with the reviewer's own loops from results/preds
(onsets, hits, shares and standard errors, median NSE of the seed averages, for the AORC, ERA5, rain-table and control sets), and a look at WHY with data of my own: the AORC daily rain over the window of the rain table
(my aorc_to_09 series) against the Daymet rain, by year and season, 2003-2014 (the AORC methods document says Stage IV was 'consistently biased low during the winter months from 2002 to approximately 2009' before
dual-polarisation radar and MPE improved it from 2010). About 20 seconds."""
import sys; sys.dont_write_bytecode = True
from a_common import *
import flood as F
from a_scoring import *
import camels

R = jload("results/aorc_rain_to_midnight.json"); BP = R["by_period"]
D = F.load_data(); ids = list(D)
PER = {"before_2009_07_17": ("2005-10-01", "2009-07-16"), "from_2009_07_17": ("2009-07-17", "2014-09-30")}
SETS = ["wq", "wqn", "wqh", "wqa", "wqa_to09", "wqa_utc", "wqa_rep", "wqf", "wqh_to09", "wqh_utc", "wqnh"]
out = {"sets": {}, "differences": {}}
dev = 0
for label, (a, b) in PER.items():
    for n in SETS:
        c = counts(f"lstm_{n}", 1, a=a, b=b)
        nse = float(np.median(list(nse_per_basin(f"lstm_{n}", 1, a, b).values())))
        p, nn = c["pod"], c["n_onsets"]
        mine = dict(onsets_warned=p, hits=c["hits"], onsets=nn, nse_median=nse, se=float(np.sqrt(p * (1 - p) / nn)))
        out["sets"].setdefault(n, {})[label] = mine
        if n in BP["sets"]:
            j = BP["sets"][n][label]
            dev = max(dev, abs(mine["onsets_warned"] - j["onsets_warned"]), abs(mine["hits"] - j["hits"]), abs(mine["onsets"] - j["onsets"]), abs(mine["nse_median"] - j["nse_median"]), abs(mine["se"] - j["onsets_warned_standard_error"]))
print("by_period: my own loops against the JSON (%d sets x 2 parts x [share, hits, onsets, NSE, standard error]): max abs deviation %.1e" % (sum(1 for n in SETS if n in BP["sets"]), dev))
out["max_dev"] = dev
print("\n%-10s %-34s %-34s" % ("set", "before 2009-07-17: share (hits/n) NSE", "from 2009-07-17: share (hits/n) NSE"))
for n in SETS:
    v = out["sets"][n]; b0, b1 = v["before_2009_07_17"], v["from_2009_07_17"]
    print("%-10s %5.1f%% (%3d/%3d)  %.3f   (se %.1f)     %5.1f%% (%3d/%3d)  %.3f   (se %.1f)" % (n, 100 * b0["onsets_warned"], b0["hits"], b0["onsets"], b0["nse_median"], 100 * b0["se"], 100 * b1["onsets_warned"], b1["hits"], b1["onsets"], b1["nse_median"], 100 * b1["se"]))
print("\ndifferences (first minus second): points of onsets (hits) / median NSE, before and from")
for T, C in (("wqa", "wq"), ("wqa", "wqh"), ("wqa", "wqa_to09"), ("wqa", "wqa_utc"), ("wqa", "wqn"), ("wqa_rep", "wq"), ("wqh", "wq")):
    rec = {p: dict(onset_points=100 * (out["sets"][T][p]["onsets_warned"] - out["sets"][C][p]["onsets_warned"]), hits=out["sets"][T][p]["hits"] - out["sets"][C][p]["hits"], nse=out["sets"][T][p]["nse_median"] - out["sets"][C][p]["nse_median"]) for p in PER}
    out["differences"][f"{T}_minus_{C}"] = rec
    key = f"{T}_minus_{C}"
    jd = BP["differences"].get(key)
    note = ""
    if jd: note = "  (JSON max dev %.1e)" % max(abs(rec[p]["onset_points"] - jd[p]["onset_points"]) + abs(rec[p]["nse"] - jd[p]["nse"]) + abs(rec[p]["hits"] - jd[p]["hits"]) for p in PER)
    print(f"  {key:22s} before {rec['before_2009_07_17']['onset_points']:+6.1f} pts ({rec['before_2009_07_17']['hits']:+4d}, NSE {rec['before_2009_07_17']['nse']:+.3f})   from {rec['from_2009_07_17']['onset_points']:+6.1f} pts ({rec['from_2009_07_17']['hits']:+4d}, NSE {rec['from_2009_07_17']['nse']:+.3f}){note}")
# the per-basin picture of the NSE swing wqa - wqh
print("\nper basin, median NSE of wqa minus wqh (ERA5), before and from 2009-07-17:")
sw = {}
for label, (a, b) in PER.items():
    na, nh = nse_per_basin("lstm_wqa", 1, a, b), nse_per_basin("lstm_wqh", 1, a, b)
    sw[label] = {g: float(na[g] - nh[g]) for g in ids}
print("  before:", {F.SHORT[g].split(",")[0]: round(v, 3) for g, v in sw["before_2009_07_17"].items()})
print("  from  :", {F.SHORT[g].split(",")[0]: round(v, 3) for g, v in sw["from_2009_07_17"].items()})
print(f"  basins where AORC has the higher NSE: before {sum(v > 0 for v in sw['before_2009_07_17'].values())} of 12, from {sum(v > 0 for v in sw['from_2009_07_17'].values())} of 12")
out["nse_swing_by_basin"] = sw
out["nse_swing_summary"] = {lab: dict(n_basins_aorc_higher=int(sum(v > 0 for v in d.values())), median_of_the_basin_differences=float(np.median(list(d.values()))), difference_of_the_medians=out["differences"]["wqa_minus_wqh"][lab]["nse"]) for lab, d in sw.items()}
print("  median of the 12 basin differences: " + ", ".join(f"{lab} {v['median_of_the_basin_differences']:+.3f} (difference of the two medians {v['difference_of_the_medians']:+.3f})" for lab, v in out["nse_swing_summary"].items()))
print("  median NSE of each set, before -> from 2009-07-17: " + "; ".join(f"{n} {out['sets'][n]['before_2009_07_17']['nse_median']:.3f} -> {out['sets'][n]['from_2009_07_17']['nse_median']:.3f}" for n in ("wq", "wqn", "wqh", "wqnh", "wqa", "wqa_to09", "wqa_utc", "wqa_rep", "wqf")))

# ---- why: AORC over the window of the rain table against the Daymet rain, by year and season ---------------------------------------------------------------------------------------------
Z = np.load(f"{OUT}/my_series_aorc.npz"); assert list(Z["basins"]) == ids
idx = D[ids[0]].index
rows = []
for gi, g in enumerate(ids):
    a = pd.Series(Z["to09"][gi], index=idx); d = D[g].prcp
    for y in range(2003, 2015):
        for sname, months in (("DJF", (12, 1, 2)), ("MAM", (3, 4, 5)), ("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
            m = (idx.year == y) & np.isin(idx.month, months) if sname != "DJF" else ((idx.year == y) & np.isin(idx.month, (1, 2))) | ((idx.year == y - 1) & (idx.month == 12))
            if m.sum() < 60: continue
            rows.append(dict(gauge=g, year=y, season=sname, aorc=float(a[m].sum()), daymet=float(d[m].sum()), corr=float(np.corrcoef(a[m], d[m])[0, 1])))
T = pd.DataFrame(rows)
by = T.groupby(["year", "season"]).apply(lambda f: pd.Series(dict(ratio=f.aorc.sum() / f.daymet.sum(), corr=f["corr"].median())), include_groups=False).unstack("season")
print("\nAORC (09:00-09:00 window) over Daymet rain, ratio of the season totals pooled over the 12 basins:")
print(by["ratio"].round(2).to_string())
winter_early = T[(T.season == "DJF") & (T.year.between(2005, 2009))]; winter_late = T[(T.season == "DJF") & (T.year.between(2011, 2014))]
r_e, r_l = winter_early.aorc.sum() / winter_early.daymet.sum(), winter_late.aorc.sum() / winter_late.daymet.sum()
other_e, other_l = T[(T.season != "DJF") & (T.year.between(2005, 2009))], T[(T.season != "DJF") & (T.year.between(2011, 2014))]
print(f"  winters (DJF) 2005-2009: AORC/Daymet {r_e:.3f}; winters 2011-2014: {r_l:.3f}; the other seasons: {other_e.aorc.sum() / other_e.daymet.sum():.3f} (2005-2009) against {other_l.aorc.sum() / other_l.daymet.sum():.3f} (2011-2014)")
cw_e, cw_l = winter_early["corr"].median(), winter_late["corr"].median()
print(f"  median daily correlation with Daymet in winters: {cw_e:.3f} (2005-2009) against {cw_l:.3f} (2011-2014); other seasons: {other_e['corr'].median():.3f} against {other_l['corr'].median():.3f}")
out["aorc_over_daymet"] = dict(ratio_by_year_season={f"{y}|{s}": float(v) for (y, s), v in by["ratio"].stack().items()}, winter_ratio_2005_2009=float(r_e), winter_ratio_2011_2014=float(r_l), other_ratio_2005_2009=float(other_e.aorc.sum() / other_e.daymet.sum()),
                               other_ratio_2011_2014=float(other_l.aorc.sum() / other_l.daymet.sum()), winter_corr_2005_2009=float(cw_e), winter_corr_2011_2014=float(cw_l))
jsave("a9_byperiod.json", out)
