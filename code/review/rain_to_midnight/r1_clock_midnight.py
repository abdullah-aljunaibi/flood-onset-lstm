"""ROUND 1, checks 3a to 3e (re-written in round 2; the round-1 scratch copy was lost): the daily series rain_to_midnight rebuilt for all 12 basins with the reviewer's own code (r2_clock.py: the value at
stamp T is the hour that ENDS at T; each hour goes to the wall-clock calendar day in which it ends; summer time by a hand-written US rule and by zoneinfo), compared with the statistics in
results/rain_to_midnight.json (mean mm/day, correlations with Daymet, day lengths, days on summer time at the issue time); the last hour of a July and a January day; the perturbation proof (+100 mm on every
hour that starts at or after the local midnight that ends day t changes no value up to day t); the scalers stored in the saved networks against the series. Every result is compared with the numbers
recorded in results/review_rain_to_midnight.json (round 1, checks 3a to 3e). About a minute."""
import sys; sys.dont_write_bytecode = True
import time
from r2_common import *
import flood as F
import r2_clock as C

t0 = time.time()
D = F.load_data(); ids = list(D)
R = jload("results/rain_to_midnight.json"); REC = {c["id"]: c["numbers"] for c in jload("results/review_rain_to_midnight.json")["checks"]}
st = pd.read_csv(f"{PROJ}/data/hourly/usgs_sites.rdb", sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
ZONE = {g: {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago"}[st.tz_cd[g]] for g in ids}   # zones from the USGS site file
DAYS = np.arange(C.FIRST_DAY, C.LAST_DAY + 1)
corr = lambda a, b: dict(day_before=float(np.corrcoef(a[1:], b[:-1])[0, 1]), same_day=float(np.corrcoef(a, b)[0, 1]), day_after=float(np.corrcoef(a[:-1], b[1:])[0, 1]))
ISSUE = {z: np.array([C.local_instant_utc(C.FIRST_DAY + i, 24, z, "rules") for i in range(C.N_DAYS)]) for z in set(ZONE.values())}      # the issue time of every day (local midnight at the end of day t), hand-written rules
mine, hours, offs, grid, out, NH = {}, {}, {}, {}, {}, {}
worst = dict(mean=0.0, corr=0.0, daylen=0, summer=0)
for g in ids:
    t, x, gp = C.read_era5(g); off = C.offset_rules(t, ZONE[g]); s, n = C.window_sum(t, x, off, "wall", 24)
    mine[g], hours[g], offs[g], grid[g], NH[g] = s, (t, x), off, tuple(gp), n
    a = D[g].prcp.values; cc = corr(a, s); jj = R["daily_rain"]["check"]["per_basin"][g]; dl = R["daily_rain"]["day_lengths"]["per_basin"][g]
    summer = int((C.offset_rules(ISSUE[ZONE[g]], ZONE[g]) > C.STD[ZONE[g]]).sum())
    worst["mean"] = max(worst["mean"], abs(float(s.mean()) - jj["mean_mm_day"]["era5"])); worst["corr"] = max(worst["corr"], *[abs(cc[k] - jj["daymet_against_era5_local_day"][k]) for k in cc])
    worst["daylen"] = max(worst["daylen"], abs(int((n == 23).sum()) - dl["days_with_23_hours"]), abs(int((n == 24).sum()) - dl["days_with_24_hours"]), abs(int((n == 25).sum()) - dl["days_with_25_hours"]))
    worst["summer"] = max(worst["summer"], abs(summer - dl["n_days_on_summer_time_at_the_issue_time"]))
    if g in ("02137727", "02472000"):
        out[g] = dict(mean_mm_day_era5=float(s.mean()), corr=cc)
print("3a: series rebuilt for 12 basins; worst deviation from the JSON statistics:", worst, "| Catawba: days of 23 / 24 / 25 hours:", int((NH["02137727"] == 23).sum()), int((NH["02137727"] == 24).sum()), int((NH["02137727"] == 25).sum()), "(the JSON: 34 / 12350 / 34); days on summer time at the issue time:", int((C.offset_rules(ISSUE[ZONE["02137727"]], ZONE["02137727"]) > C.STD[ZONE["02137727"]]).sum()), "(the JSON: 7147)")
rec3a = REC["3a"]
dev3a = max(abs(out["02137727"]["mean_mm_day_era5"] - rec3a["catawba"]["mean_mm_day_era5"]), abs(out["02472000"]["mean_mm_day_era5"] - rec3a["leaf_river"]["mean_mm_day_era5"]),
            *[abs(out["02137727"]["corr"][k] - rec3a["catawba"]["corr_d_minus_1_d_d_plus_1"][k]) for k in ("day_before", "same_day", "day_after")])
print(f"    against the round-1 record: |Catawba and Leaf River mean and correlations - recorded| = {dev3a:.1e}")
# 3b: the rules against zoneinfo at every stamp, and the daily sums by the two routes, for the two named basins
same = {}
for g in ("02137727", "02472000"):
    t, x = hours[g]; oz = C.offset_zoneinfo(t, ZONE[g]); sz, _ = C.window_sum(t, x, oz, "wall", 24)
    same[g] = (bool(np.array_equal(offs[g], oz)), bool(np.array_equal(sz, mine[g])))
print("3b: offsets by the hand-written rules equal zoneinfo at all", len(hours["02137727"][0]), "stamps; daily sums by the two routes equal:", same)
cells = {}
for g in ids: cells.setdefault(grid[g], []).append(F.SHORT[g])
print("    distinct ERA5 grid cells:", len(cells), "| shared:", {str(k): v for k, v in cells.items() if len(v) > 1})
# 3c: the last hour summed into a day
def stamps_of(g, day):
    t, x = hours[g]; d = C.window_day(t, offs[g], "wall", 24) - C.D0; i = int((np.datetime64(day, "D") - C.FIRST_DAY).astype(int)); sel = np.flatnonzero(d == i)
    return str(t[sel[-1]]), str(t[sel[0]]), len(sel)
ex = {("02137727", "2010-07-14"): stamps_of("02137727", "2010-07-14"), ("02472000", "2010-07-14"): stamps_of("02472000", "2010-07-14"), ("02137727", "2010-01-13"): stamps_of("02137727", "2010-01-13"), ("02472000", "2010-01-13"): stamps_of("02472000", "2010-01-13")}
n23, n25 = stamps_of("02137727", "2010-03-14")[2], stamps_of("02137727", "2010-11-07")[2]
rec3c = REC["3c"]
ok3c = (ex[("02137727", "2010-07-14")][0] == rec3c["catawba_2010_07_14_last_hour_ends_utc"] and ex[("02472000", "2010-07-14")][0] == rec3c["leaf_river_2010_07_14"] and ex[("02137727", "2010-01-13")][0] == rec3c["catawba_2010_01_13"]
        and ex[("02472000", "2010-01-13")][0] == rec3c["leaf_river_2010_01_13"] and (n23, n25) == (rec3c["hours_in_day"]["spring_2010_03_14"], rec3c["hours_in_day"]["autumn_2010_11_07"]))
print("3c: last hour of a day ends (UTC): Catawba 2010-07-14", ex[("02137727", "2010-07-14")][0], "(04:00 required), Leaf River", ex[("02472000", "2010-07-14")][0], "(05:00), Catawba 2010-01-13", ex[("02137727", "2010-01-13")][0], "(05:00), Leaf River", ex[("02472000", "2010-01-13")][0], "(06:00); hours of the clock-change days:", n23, n25, "| equals the round-1 record:", ok3c)
# 3d: the perturbation proof
rng = np.random.default_rng(987654); special = ["1981-04-26", "1981-10-25", "2007-03-11", "2007-11-04", "2010-07-14", "2010-01-13"]
tot = 0; ok_all = True
for g in ("02137727", "02472000"):
    t, x = hours[g]; off = offs[g]; s0 = mine[g]; start = t - np.timedelta64(60, "m")
    for tday in sorted(set(int(v) for v in rng.integers(2, C.N_DAYS - 3, 6)) | {int((np.datetime64(sd, "D") - C.FIRST_DAY).astype(int)) for sd in special}):
        day64 = C.FIRST_DAY + tday
        Mz, Mr = C.local_instant_utc(day64, 24, ZONE[g], "zoneinfo"), C.local_instant_utc(day64, 24, ZONE[g], "rules"); assert Mz == Mr; M = Mz
        s2, _ = C.window_sum(t, np.where(start >= M, x + 100.0, x), off, "wall", 24)
        unchanged = bool(np.array_equal(s2[:tday + 1], s0[:tday + 1])); raised = bool(s2[tday + 1] > s0[tday + 1])
        xb = x.copy(); xb[np.flatnonzero(t == M + np.timedelta64(60, "m"))[0]] += 100.0; sb, _ = C.window_sum(t, xb, off, "wall", 24)
        first_after = bool(np.array_equal(sb[:tday + 1], s0[:tday + 1])) and abs(sb[tday + 1] - s0[tday + 1] - 100.0) < 1e-9
        xl = x.copy(); xl[np.flatnonzero(t == M)[0]] += 100.0; sl, _ = C.window_sum(t, xl, off, "wall", 24)
        last_in = abs(sl[tday] - s0[tday] - 100.0) < 1e-9 and bool(np.array_equal(np.delete(sl, tday), np.delete(s0, tday)))
        ok_all &= unchanged and raised and first_after and last_in; tot += 1
print(f"3d: perturbation proof: {tot} cases (2 basins x 12 days incl. both kinds of clock-change day, a July and a January day): all passed: {ok_all} | round-1 record: {REC['3d']['days_tested']} days tested")
# 3e: the scalers stored in the old networks
tr = np.arange(np.datetime64("1980-10-01", "D"), np.datetime64("2000-09-30", "D") + 1); tri = (tr - C.FIRST_DAY).astype(int)
sc_dev = 0.0
for name in ("wqh", "wh", "wqh_older", "wh_older"):
    z = torch.load(f"{PROJ}/results/models/lstm_{name}_s0.pt", map_location="cpu", weights_only=True)
    ser = {g: (mine[g] if not name.endswith("older") else np.concatenate([[mine[g][0]], mine[g][:-1]])) for g in ids}
    pool = np.concatenate([ser[g][tri] for g in ids])
    sc_dev = max(sc_dev, abs(z["sc"]["dyn_mu"][-1] - pool.mean()), abs(z["sc"]["dyn_sd"][-1] - (pool.std() + 1e-9)), abs(z["sc"]["dyn_mu"][-1] - REC["3e"][name]["stored_mean"]))
print(f"3e: scalers stored in the 4 old networks (seed 0) against the training-year mean and std of my series: max deviation {sc_dev:.1e}")
agree = dict(a=(worst["mean"] < 1e-12 and worst["corr"] < 1e-12 and worst["daylen"] == 0 and worst["summer"] == 0 and dev3a < 1e-12), b=(all(all(v) for v in same.values())), c=ok3c, d=ok_all and tot == REC["3d"]["days_tested"], e=sc_dev < 1e-12)
print("summary: agrees with the round-1 record in 3a / 3b / 3c / 3d / 3e:", agree, f"({time.time()-t0:.0f} s)")
jsave("r1_clock_midnight.json", dict(worst=worst, agree=agree, perturbation_cases=tot, scaler_dev=sc_dev))
