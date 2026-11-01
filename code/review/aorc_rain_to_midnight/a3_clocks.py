"""AORC review, check (b) part 1: the clocks of the three AORC windows (aorc_to_midnight, aorc_to_09, aorc_utc_day), rebuilt independently for all 12 basins from the cache files data/aorc/aorc_<gauge>.npz with the
reviewer's own code (a_clock.py: the hour-ending stamp, summer time by a hand-written US rule and by zoneinfo, window sums), and compared with the results file; the proofs with own code (every hour in its window
and ending at or before the issue time; the perturbation proof: +100 mm on every hour that starts at or after the end of the window of day t changes nothing up to day t); the scalers of the 12 saved networks
against my series. Writes my_series_aorc.npz (used by a4_networks.py, a8, a9, a10). About half a minute."""
import sys; sys.dont_write_bytecode = True
import datetime as dt
from a_common import *
import flood as F
import camels
import a_clock as C

R = jload("results/aorc_rain_to_midnight.json")
D = F.load_data(); ids = list(D)
DAYS64 = np.arange(C.FIRST_DAY, C.LAST_DAY + 1)
DW = jload("results/day_windows.json")
UTC_OFF = {g: int(DW["day_ends"]["daymet"]["per_basin"][g]["utc_offset_h"]) for g in ids}
st = pd.read_csv(f"{PROJ}/data/hourly/usgs_sites.rdb", sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
ZONE = {g: {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago"}[st.tz_cd[g]] for g in ids}     # zones from the USGS site file, not from the JSON
assert all((C.STD[ZONE[g]] == UTC_OFF[g]) for g in ids)
WIN = {"midnight": ("wall", 24), "to09": ("wall", 9), "utc": ("utc", 24)}
names = {"midnight": "aorc_to_midnight", "to09": "aorc_to_09", "utc": "aorc_utc_day"}
SER, NH, OFF, HR, FILLED, NCELLS = {w: {} for w in WIN}, {w: {} for w in WIN}, {}, {}, {}, {}
out = {"per_basin": {}}
for g in ids:
    t, x, fl, nv, nc = C.read_aorc(g)
    off = C.offset_rules(t, ZONE[g]); OFF[g] = off; HR[g] = (t, x); FILLED[g] = fl; NCELLS[g] = nc
    for w, (kind, h) in WIN.items():
        s, n = C.window_sum(t, x, off, kind, h)
        SER[w][g], NH[w][g] = s, n
print(f"AORC cache read for 12 basins ({C.N_HOURS} hourly values each); daily series for 3 windows x 12 basins built with my own code")

# ---- 1. the offsets by the two routes agree at every stamp, for the two zones ------------------------------------------------------------------------------------------
t_any = HR[ids[0]][0]
agree = {z: bool(np.array_equal(C.offset_rules(t_any, z), C.offset_zoneinfo(t_any, z))) for z in set(ZONE.values())}
print("summer-time offsets: hand-written rules equal zoneinfo at all", len(t_any), "stamps:", agree)
out["offsets_rules_equal_zoneinfo"] = agree

# ---- 2. statistics against the results file ---------------------------------------------------------------------------------------------------------------------------
NL = {g: camels.load_basin(g, "nldas").loc[F.PERIOD[0]:F.PERIOD[1], "prcp"].astype(float).values for g in ids}
corr3 = lambda a, b: dict(day_before=float(np.corrcoef(a[1:], b[:-1])[0, 1]), same_day=float(np.corrcoef(a, b)[0, 1]), day_after=float(np.corrcoef(a[:-1], b[1:])[0, 1]))
dev_corr = dev_len = dev_cells = 0
for w in WIN:
    for g in ids:
        j = R["daily_rain"]["check_windows"]["per_basin"][names[w]][g]
        for src, a in (("daymet", D[g].prcp.values), ("nldas", NL[g])):
            mine = corr3(a, SER[w][g])
            dev_corr = max(dev_corr, *[abs(mine[k] - j[src][k]) for k in mine])
for g in ids:
    j = R["daily_rain"]["day_lengths"]["per_basin"][g]; n = NH["midnight"][g]
    dev_len = max(dev_len, abs(int((n == 23).sum()) - j["days_with_23_hours"]), abs(int((n == 24).sum()) - j["days_with_24_hours"]), abs(int((n == 25).sum()) - j["days_with_25_hours"]))
    dev_cells = max(dev_cells, abs(NCELLS[g] - R["daily_rain"]["mask"]["n_cells_by_basin"][g]))
for w in ("to09", "utc"):
    for g in ids:
        j = R["daily_rain"]["windows"][names[w]]["day_lengths"][g]; n = NH[w][g]
        dev_len = max(dev_len, abs(int((n == 23).sum()) - j["days_with_23_hours"]), abs(int((n == 24).sum()) - j["days_with_24_hours"]), abs(int((n == 25).sum()) - j["days_with_25_hours"]))
out["stats_vs_json"] = dict(max_dev_correlations_3_series_x_2_products_x_3_lags_x_12_basins=dev_corr, max_dev_day_length_counts=dev_len, max_dev_n_cells=dev_cells)
print("correlations of Daymet and NLDAS with the three AORC series (216 numbers) against check_windows in the JSON: max abs deviation", dev_corr, "| day-length counts (3 windows):", dev_len, "| cells per basin in the cache against the JSON:", dev_cells)
med = {names[w]: {src: {k: float(np.median([corr3(D[g].prcp.values if src == 'daymet' else NL[g], SER[w][g])[k] for g in ids])) for k in ("day_before", "same_day", "day_after")} for src in ("daymet", "nldas")} for w in WIN}
out["median_correlations"] = med
print("median same-day correlation: Daymet with aorc_to_09 %.3f, aorc_utc_day %.3f, aorc_to_midnight %.3f | NLDAS with aorc_to_09 %.3f, aorc_utc_day %.3f, aorc_to_midnight %.3f" % (
    med["aorc_to_09"]["daymet"]["same_day"], med["aorc_utc_day"]["daymet"]["same_day"], med["aorc_to_midnight"]["daymet"]["same_day"], med["aorc_to_09"]["nldas"]["same_day"], med["aorc_utc_day"]["nldas"]["same_day"], med["aorc_to_midnight"]["nldas"]["same_day"]))

# ---- 3. membership of every hour, by the wall clock and in UTC, for every basin and day ----------------------------------------------------------------------------------
print("\nmembership proofs, own code, all 12 basins x 12418 windows:")
ISSUE = {z: np.array([C.local_instant_utc(C.FIRST_DAY + i, 24, z, "zoneinfo") for i in range(C.N_DAYS)]) for z in set(ZONE.values())}      # the issue time of every day: local midnight at the end of day t, as a UTC instant
issue_margin = {}
for w in ("to09", "utc", "midnight"):
    kind, h = WIN[w]
    ok_all = True
    lens = set(); marg = []
    for g in ids:
        t, x = HR[g]; off = OFF[g]; zone = ZONE[g]
        d = C.window_day(t, off, kind, h)
        end_local = t + off.astype("timedelta64[h]").astype("timedelta64[m]")                 # wall-clock end of each hour
        day_date = d.astype("datetime64[D]")                                                  # the calendar day t each hour is assigned to
        hi = day_date.astype("datetime64[m]") + np.timedelta64(h * 60, "m")
        lo = hi - np.timedelta64(24 * 60, "m")
        member = (end_local > lo) & (end_local <= hi) if kind == "wall" else (t > lo) & (t <= hi)
        inside = (d - C.D0 >= 0) & (d - C.D0 < C.N_DAYS)
        ok_all &= bool(member.all())
        dd = d[inside] - C.D0
        m = (ISSUE[zone][dd] - t[inside]) / np.timedelta64(60, "m")
        marg.append((float(m.min()), float(m.max())))
        ok_all &= bool((m >= 0).all())
        lens |= set(np.unique(NH[w][g]).tolist())
    issue_margin[w] = (min(a for a, _ in marg), max(b for _, b in marg))
    print(f"  {w:9s} every hour lies in its window (clock of the window) and ends at or before the issue time: {ok_all}; hours per window {sorted(lens)}; the hours summed into day t end between {issue_margin[w][0]:g} and {issue_margin[w][1]:g} h BEFORE the issue time, worst hour over all basins and days")
    out[f"membership_{w}"] = dict(ok=ok_all, hours_per_window=sorted(lens), hours_before_issue_time_min_max_over_all_hours=issue_margin[w])
for w in ("to09", "utc"):
    j = R["daily_rain"]["windows"]["last_hour_ends_before_the_issue_time_h"][names[w]]
    margins = []
    for g in ids:
        zone = ZONE[g]
        issue = np.array([C.local_instant_utc(C.FIRST_DAY + i, 24, zone, "zoneinfo") for i in range(0, C.N_DAYS, 7)])
        wend = np.array([C.local_instant_utc(C.FIRST_DAY + i, 9, zone, "zoneinfo") for i in range(0, C.N_DAYS, 7)]) if w == "to09" else np.array([np.datetime64(str(C.FIRST_DAY + i + 1), "m") for i in range(0, C.N_DAYS, 7)])
        margins += list(((issue - wend) / np.timedelta64(60, "m")).astype(float))
    out[f"last_hour_margin_{w}"] = dict(mine=[min(margins), max(margins)], json=j, equal=(abs(min(margins) - j[0]) < 1e-9 and abs(max(margins) - j[1]) < 1e-9))
    print(f"  {w}: JSON says the LAST hour ends {j[0]:g} to {j[1]:g} h before the issue time; mine (every 7th day, 12 basins): {min(margins):g} to {max(margins):g} h")

# ---- 4. the perturbation proof -------------------------------------------------------------------------------------------------------------------------------------------------------
print("\nperturbation proof: +100 mm on every hour that starts at or after the END of the window of day t, recompute, compare up to day t")
rng = np.random.default_rng(20261007)
special = [str(np.datetime64(x, "D")) for x in ("1981-04-26", "1981-10-25", "1990-04-01", "1999-10-31", "2007-03-11", "2007-11-04", "2010-01-13", "2010-07-14", "2012-03-10")]
tot = 0; ok_pert = True; rows = []
for w in ("midnight", "to09", "utc"):
    kind, h = WIN[w]
    for g in ("02137727", "02472000"):
        t, x = HR[g]; off = OFF[g]; zone = ZONE[g]
        s0, n0 = SER[w][g], NH[w][g]
        days = sorted(set(int(v) for v in rng.integers(2, C.N_DAYS - 3, 6)) | {int((np.datetime64(sd, "D") - C.FIRST_DAY).astype(int)) for sd in special})
        for tday in days:
            day64 = C.FIRST_DAY + tday
            if kind == "wall":
                Mz, Mr = C.local_instant_utc(day64, h, zone, "zoneinfo"), C.local_instant_utc(day64, h, zone, "rules")
                assert Mz == Mr
                M = Mz
            else:
                M = np.datetime64(str(day64 + 1), "m")                                                  # 00:00 UTC of the next date
            start = t - np.timedelta64(60, "m")
            late = start >= M                                                                           # hours that start at or after the window end
            s2, _ = C.window_sum(t, np.where(late, x + 100.0, x), off, kind, h)
            unchanged = bool(np.array_equal(s2[:tday + 1], s0[:tday + 1])); raised = bool(s2[tday + 1] > s0[tday + 1])
            xb = x.copy(); xb[np.flatnonzero(t == M + np.timedelta64(60, "m"))[0]] += 100.0               # the hour that STARTS exactly at the end of the window (stamp M+1h): belongs to day t+1 only
            sb, _ = C.window_sum(t, xb, off, kind, h)
            first_after = bool(np.array_equal(sb[:tday + 1], s0[:tday + 1])) and abs(sb[tday + 1] - s0[tday + 1] - 100.0) < 1e-9
            xl = x.copy(); xl[np.flatnonzero(t == M)[0]] += 100.0                                         # the hour that ENDS exactly at the window end (stamp M): belongs to day t only
            sl, _ = C.window_sum(t, xl, off, kind, h)
            last_in = abs(sl[tday] - s0[tday] - 100.0) < 1e-9 and bool(np.array_equal(np.delete(sl, tday), np.delete(s0, tday)))
            good = unchanged and raised and first_after and last_in
            ok_pert &= good; tot += 1
            rows.append(dict(window=w, gauge=g, day=str(day64), window_end_utc=str(M), n_hours_of_day=int(n0[tday]), values_up_to_t_unchanged=unchanged, day_t_plus_1_raised=raised, first_hour_after_goes_to_next_day_only=first_after, last_hour_goes_to_day_t_only=bool(last_in)))
print(f"  {tot} cases (3 windows x 2 basins x {len(days)} days incl. clock-change days): all passed: {ok_pert}")
out["perturbation"] = dict(n_cases=tot, all_ok=ok_pert, rows=rows)

# ---- 5. the scalers of the 12 saved networks against my series -------------------------------------------------------------------------------------------------------
tr = np.arange(np.datetime64("1980-10-01", "D"), np.datetime64("2000-09-30", "D") + 1); tri = (tr - C.FIRST_DAY).astype(int)
cols_mine = {"aorc_to_midnight": SER["midnight"], "aorc_to_09": SER["to09"], "aorc_utc_day": SER["utc"]}
sc_out = {}
for name in ("wqa", "wqa_to09", "wqa_utc", "wqa_rep"):
    for seed in (0, 1, 2):
        z = torch.load(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
        dyn = z["dyn"]; dev = 0
        for ci, c in enumerate(dyn):
            if c in cols_mine:
                pool = np.concatenate([cols_mine[c][g][tri] for g in ids])
                dev = max(dev, abs(z["sc"]["dyn_mu"][ci] - pool.mean()), abs(z["sc"]["dyn_sd"][ci] - (pool.std() + 1e-9)))
        sc_out[f"{name}_s{seed}"] = dict(dyn=dyn, max_dev_stored_vs_my_training_year_mean_std=float(dev))
print("\nscalers stored in the 12 networks (the AORC columns: mean and std of the training years, pooled over the 12 basins) against my series: max deviation", max(v["max_dev_stored_vs_my_training_year_mean_std"] for v in sc_out.values()))
print("  dyn columns:", {k: v["dyn"] for k, v in sc_out.items() if k.endswith("s0")})
out["scalers"] = sc_out
np.savez(f"{OUT}/my_series_aorc.npz", basins=np.array(ids), midnight=np.stack([SER["midnight"][g] for g in ids]), to09=np.stack([SER["to09"][g] for g in ids]), utc=np.stack([SER["utc"][g] for g in ids]),
         nldas=np.stack([NL[g] for g in ids]))
jsave("a3_clocks.json", out)
