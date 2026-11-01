"""E4b review, check 3: the within-day columns rebuilt from the hourly AORC cache (data/aorc/aorc_<gauge>.npz: hour-ENDING UTC stamps) with the reviewer's own code and
compared with code/subdaily_inputs.add_inputs; the clock (every hour of day t ends at or before 24:00 local of day t, summer time included, the clock-change days, the hour
ending exactly 24:00 belongs to day t); the made-up-hours test on the reviewer's builder and on the project's; q_rel; the maxcell fill days; the _older shift. Writes
OUT/my_columns.npz (the reviewer's own columns) for r2_headline.py and r4_networks.py. About half a minute.

The reviewer's route: for every local calendar day d the UTC instants of its start (00:00 of d local) and end (00:00 of d+1 local) come from zoneinfo; the day's hours are the
stamps in (start, end] (an hour-ending stamp equal to the end of the day belongs to that day); the last k hours are the stamps in (end - k h, end]. No 'minus one minute'
trick, no groupby: the two routes share nothing but the cache and zoneinfo."""
from r_common import *
from zoneinfo import ZoneInfo
import flood as F
import subdaily_inputs as SI

t0 = time.time()
R = jload(RESULTS)
D = F.load_data()
ids = list(D)
days = pd.date_range(*PERIOD)
tr = within(days, *SPLIT["train"])
thr_mine = {g: float(np.quantile(D[g].q_mm.values[tr], 0.95)) for g in ids}
thr_proj = F.thresholds(D)
dw = jload("results/day_windows.json")["day_ends"]["daymet"]["per_basin"]
ZONE = {-5: "America/New_York", -6: "America/Chicago"}
zones = {g: ZONE[int(dw[g]["utc_offset_h"])] for g in ids}
A = F.basins()
HOUR = pd.Timedelta(hours=1)
out = dict(thresholds=dict(mine_equal_results_file=all(abs(thr_mine[g] - R["thresholds"][g]) < 1e-9 for g in ids), mine_equal_flood_thresholds=all(abs(thr_mine[g] - thr_proj[g]) < 1e-12 for g in ids),
                           source="the 95th percentile of the daily flow of the TRAINING years 1980-10-01..2000-09-30 only (numpy.quantile, linear), per basin", values=thr_mine),
           zones={g: dict(zone=zones[g], utc_offset_winter=int(dw[g]["utc_offset_h"]), gauge_lon=float(A.gauge_lon[g]), gauge_lat=float(A.gauge_lat[g]), state=str(A.gauge_name[g])[-2:] if "gauge_name" in A else None) for g in ids})
# plausibility of the zones from the longitude: the Central zone's eastern edge in the south-east runs near 85-88 W; Leaf River, MS (about 89.4 W) is the only Central basin
out["zones_plausible_by_longitude"] = all((zones[g] == "America/Chicago") == (float(A.gauge_lon[g]) < -87.5) for g in ids)


def my_table(g):
    z = np.load(PROJ / "data" / "aorc" / f"aorc_{g}.npz", allow_pickle=False)
    stamp = pd.DatetimeIndex(pd.to_datetime(z["stamp_utc_s"], unit="s"))                      # the END of each hour, UTC
    rain, maxcell, filled = z["rain_mm"].astype(float), z["max_cell_mm"].astype(float), z["filled"].astype(bool)
    assert (np.diff(stamp.values).astype("timedelta64[h]") == np.timedelta64(1, "h")).all(), "the cache is not hourly without gaps"
    tz = ZoneInfo(zones[g])
    start = days.tz_localize(tz).tz_convert("UTC").tz_localize(None)                          # 00:00 local of day d, as UTC
    end = (days + pd.Timedelta(days=1)).tz_localize(tz).tz_convert("UTC").tz_localize(None)   # 00:00 local of day d+1 = 24:00 of day d (the issue time), as UTC
    i_start, i_end = stamp.searchsorted(start), stamp.searchsorted(end)                        # positions of the hours ending exactly at the start and at the end
    assert (stamp[i_start] == start).all() and (stamp[i_end] == end).all(), "a local midnight is not on the hour (impossible for these zones)"
    lo, hi = i_start + 1, i_end + 1                                                            # the day's hours: positions lo .. hi-1 (the hour ending at the start belongs to the day before)
    n_hours = hi - lo
    c = np.concatenate([[0.0], np.cumsum(rain)])
    tab = dict(hours=n_hours, aorc_to_midnight=c[hi] - c[lo])
    for k in (3, 6, 12):
        tab[f"aorc_last{k}h"] = c[hi] - c[hi - k]                                              # the k hours ending at the end of the day
    wet = np.concatenate([[0], np.cumsum(rain >= 0.5)])
    fil = np.concatenate([[0], np.cumsum(filled)])
    tab["aorc_wet_hours"] = wet[hi] - wet[lo]
    tab["aorc_filled_hours"] = fil[hi] - fil[lo]
    last = hi[-1]                                                                              # the cache runs past 2014-09-30: cut it at the end of the last day, so reduceat's last segment is that day only
    mx = np.maximum.reduceat(rain[:last], lo)                                                  # reduceat over [lo_i, lo_{i+1}) = the day's hours (days are contiguous)
    assert (np.diff(lo) == n_hours[:-1]).all() and len(mx) == len(days)
    tab["aorc_max1h"] = mx
    mc = np.where(filled, -np.inf, np.where(np.isfinite(maxcell), maxcell, -np.inf))
    m2 = np.maximum.reduceat(mc[:last], lo)
    tab["aorc_maxcell1h"] = np.where(np.isfinite(m2), m2, np.nan)                              # NaN where every hour of the day was filled
    return tab, stamp, rain, maxcell, filled, start, end, lo, hi


W_proj = SI.add_inputs(D, thr_proj)                                                           # the project's columns (what the script trained on, before its fill)
MY = {c: np.zeros((len(ids), len(days))) for c in ("aorc_to_midnight", "aorc_last3h", "aorc_last6h", "aorc_last12h", "aorc_max1h", "aorc_maxcell1h", "aorc_wet_hours", "aorc_filled_hours", "hours")}
cmp, clock, fill = {}, {}, {}
for i, g in enumerate(ids):
    tab, stamp, rain, maxcell, filled, start, end, lo, hi = my_table(g)
    for c in MY:
        MY[c][i] = tab[c]
    P = W_proj[g]
    diffs = {c: float(np.nanmax(np.abs(tab[c] - P[c].values))) for c in MY if c != "hours"}
    nan_same = bool(np.array_equal(np.isnan(tab["aorc_maxcell1h"]), P.aorc_maxcell1h.isna().values))
    q_rel_ok = bool(np.allclose(P.q_rel.values, np.log1p(D[g].q_mm.values / thr_mine[g]), rtol=0, atol=1e-12))
    cmp[g] = dict(max_abs_diff_by_column=diffs, all_within_1e_9=max(diffs.values()) < 1e-9, maxcell_nan_pattern_equal=nan_same, q_rel_equals_log1p_q_over_my_Q95=q_rel_ok)
    # the clock: where do the hours of each window end, on the local wall clock?
    tz = ZoneInfo(zones[g])
    local = stamp.tz_localize("UTC").tz_convert(tz).tz_localize(None)
    h_end = (local - pd.Timedelta(minutes=1)).hour.values + 1                                  # 1..24: the wall-clock hour at which the hour ends
    day_of = (local - pd.Timedelta(minutes=1)).normalize().values
    pos = np.arange(len(stamp))
    # every hour of day d (positions lo..hi-1) lies on local day d; the last-3h hours end at 22, 23, 24
    in_day = np.concatenate([day_of[lo[j]:hi[j]] == days[j].to_datetime64() for j in range(len(days))])
    last3 = np.concatenate([h_end[hi[j] - 3:hi[j]] for j in range(len(days))]).reshape(len(days), 3)
    last6 = np.concatenate([h_end[hi[j] - 6:hi[j]] for j in range(len(days))]).reshape(len(days), 6)
    last12 = np.concatenate([h_end[hi[j] - 12:hi[j]] for j in range(len(days))]).reshape(len(days), 12)
    ends_at_24 = h_end[hi - 1] == 24
    margin_h = ((end.values - stamp.values[lo]).astype("timedelta64[h]").astype(int))         # hours between the first hour's end and the issue time
    n23, n25 = int((tab["hours"] == 23).sum()), int((tab["hours"] == 25).sum())
    d23 = days[tab["hours"] == 23]
    d25 = days[tab["hours"] == 25]
    clock[g] = dict(every_hour_of_day_t_lies_on_local_day_t=bool(in_day.all()), last_hour_of_every_day_ends_at_24_00_local=bool(ends_at_24.all()),
                    last3h_hours_end_at={",".join(map(str, sorted(set(last3.ravel()))))}, last6h_hours_end_at=sorted(set(last6.ravel().tolist())), last12h_hours_end_at=sorted(set(last12.ravel().tolist())),
                    last3h_always_22_23_24=bool((last3 == np.array([22, 23, 24])).all()), last6h_always_19_to_24=bool((last6 == np.arange(19, 25)).all()), last12h_always_13_to_24=bool((last12 == np.arange(13, 25)).all()),
                    hours_per_day=sorted(set(tab["hours"].tolist())), n_days_23h=n23, n_days_25h=n25, n_days_24h=int((tab["hours"] == 24).sum()),
                    first_23h_day=str(d23[0].date()), first_25h_day=str(d25[0].date()), last_23h_day=str(d23[-1].date()), last_25h_day=str(d25[-1].date()),
                    hours_before_the_issue_time_min_max=[int(min(0, (margin_h - tab["hours"]).min())), int(margin_h.max())],
                    issue_time_utc_examples={str(days[j].date()): str(end[j]) for j in (0, int(np.flatnonzero(tab["hours"] == 23)[0]), int(np.flatnonzero(tab["hours"] == 25)[0]))})
    # the fill days: maxcell NaN (every hour filled) and maxcell below max1h on partly filled days
    nan = np.isnan(tab["aorc_maxcell1h"])
    low = (~nan) & (tab["aorc_maxcell1h"] < tab["aorc_max1h"] - 1e-6)
    fill[g] = dict(filled_where_nan=[str(d.date()) for d in days[nan]], raised_to_max1h_on_partly_filled_days=[str(d.date()) for d in days[low]],
                   all_nan_days_wholly_filled=bool((tab["aorc_filled_hours"][nan] >= 23).all()) if nan.any() else True, all_low_days_partly_filled=bool((tab["aorc_filled_hours"][low] > 0).all()) if low.any() else True)
    print(f"{g} {zones[g]:17s} max diff {max(diffs.values()):.1e} | hours {clock[g]['hours_per_day']} 23h {n23} 25h {n25} | last3h ends at {clock[g]['last3h_hours_end_at']} | in-day {in_day.all()} | nan days {len(fill[g]['filled_where_nan'])} raised {len(fill[g]['raised_to_max1h_on_partly_filled_days'])}")
out["my_columns_vs_subdaily_inputs"] = dict(per_basin=cmp, all_basins_all_columns_within_1e_9=all(v["all_within_1e_9"] and v["maxcell_nan_pattern_equal"] and v["q_rel_equals_log1p_q_over_my_Q95"] for v in cmp.values()),
                                           worst=max(max(v["max_abs_diff_by_column"].values()) for v in cmp.values()))
out["clock"] = dict(per_basin=clock, all_basins_pass=all(v["every_hour_of_day_t_lies_on_local_day_t"] and v["last_hour_of_every_day_ends_at_24_00_local"] and v["last3h_always_22_23_24"] and v["last6h_always_19_to_24"] and v["last12h_always_13_to_24"] and v["hours_per_day"] == [23, 24, 25] and v["n_days_23h"] == 34 and v["n_days_25h"] == 34 for v in clock.values()))
rf = R["maxcell_fill_days"]["per_basin"]
out["maxcell_fill_days"] = dict(mine=fill, equal_to_results_file=all(fill[g]["filled_where_nan"] == rf[g]["filled_where_nan"] and fill[g]["raised_to_max1h_on_partly_filled_days"] == rf[g]["raised_to_max1h_on_partly_filled_days"] for g in ids),
                                n_filled=sum(len(v["filled_where_nan"]) for v in fill.values()), n_raised=sum(len(v["raised_to_max1h_on_partly_filled_days"]) for v in fill.values()),
                                results_file=(R["maxcell_fill_days"]["n_basin_days_filled"], R["maxcell_fill_days"]["n_basin_days_raised"]))

# the made-up-hours test on both builders: every hour ending AFTER the issue time of a cut day gets x7+100; rows up to the cut must not move, the next day must
rng = np.random.default_rng(20261029)
mu = {}
for g in ids:
    tab, stamp, rain, maxcell, filled, start, end, lo, hi = my_table(g)
    cuts = [int(rng.integers(9500, 12000)), int(np.flatnonzero(tab["hours"] == 23)[-1]), int(np.flatnonzero(tab["hours"] == 25)[-1])]       # a random test-year day, the last 23 h and 25 h days
    res = []
    for c in cuts:
        later = stamp > end[c]                                                                  # hours that end after 24:00 local of the cut day
        x2 = np.where(later, rain * 7 + 100, rain)
        m2 = np.where(later, maxcell * 7 + 100, maxcell)
        # my builder on the perturbed series
        c2 = np.concatenate([[0.0], np.cumsum(x2)])
        mine_same = all(np.allclose(c2[hi[:c + 1]] - c2[lo[:c + 1]], tab["aorc_to_midnight"][:c + 1]) and np.allclose(c2[hi[:c + 1]] - c2[hi[:c + 1] - k], tab[f"aorc_last{k}h"][:c + 1]) for k in (3, 6, 12))
        mine_moved = not np.isclose(c2[hi[c + 1]] - c2[lo[c + 1]], tab["aorc_to_midnight"][c + 1])
        # the project's builder (subdaily_inputs._table) on the same perturbed series
        T1 = SI._table(stamp, rain, maxcell, filled, zones[g], days)
        T2 = SI._table(stamp, x2, m2, filled, zones[g], days)
        proj_same = T1.iloc[: c + 1].fillna(-1).equals(T2.iloc[: c + 1].fillna(-1))
        proj_moved = not T1.iloc[c + 1].fillna(-1).equals(T2.iloc[c + 1].fillna(-1))
        # the hour that ends exactly at 24:00 of the cut day belongs to the cut day (and to its last-3h window), not to the next day
        x3 = rain.copy()
        x3[hi[c] - 1] += 50.0
        c3 = np.concatenate([[0.0], np.cumsum(x3)])
        T3 = SI._table(stamp, x3, maxcell, filled, zones[g], days)
        last_hour_ok = (np.isclose(c3[hi[c]] - c3[lo[c]], tab["aorc_to_midnight"][c] + 50) and np.isclose(c3[hi[c]] - c3[hi[c] - 3], tab["aorc_last3h"][c] + 50) and np.isclose(c3[hi[c + 1]] - c3[lo[c + 1]], tab["aorc_to_midnight"][c + 1])
                        and np.isclose(T3.aorc_to_midnight.iloc[c], T1.aorc_to_midnight.iloc[c] + 50) and np.isclose(T3.aorc_last3h.iloc[c], T1.aorc_last3h.iloc[c] + 50) and np.isclose(T3.aorc_to_midnight.iloc[c + 1], T1.aorc_to_midnight.iloc[c + 1]))
        res.append(dict(cut=str(days[c].date()), hours_of_the_cut_day=int(tab["hours"][c]), mine_rows_up_to_cut_unchanged=bool(mine_same), mine_next_day_moved=bool(mine_moved),
                        project_rows_up_to_cut_unchanged=bool(proj_same), project_next_day_moved=bool(proj_moved), hour_ending_24_00_belongs_to_the_cut_day_both_builders=bool(last_hour_ok)))
    mu[g] = res
out["made_up_hours"] = dict(per_basin=mu, n_cases=sum(len(v) for v in mu.values()), all_pass=all(all(r.values()) for v in mu.values() for r in v if True for r in [dict((k, x) for k, x in r.items() if isinstance(x, bool))]))
print("made-up hours:", out["made_up_hours"]["n_cases"], "cases, all pass:", out["made_up_hours"]["all_pass"])

# the _older shift, applied to my columns the way the script does it (shift one day later, first day keeps its own value): checked here on the definition, used in r4 on the networks
sh = {c: np.concatenate([MY[c][:, :1], MY[c][:, :-1]], axis=1) for c in WD_COLS}
out["older_definition"] = dict(what="x_older[t] = x[t-1] for t >= 1, x_older[0] = x[0] (the script: shift(1).bfill()); the network of wqaW_older on issue day t reads the within-day values of day t-1",
                               checked=bool(all(np.array_equal(np.nan_to_num(sh[c][:, 1:], nan=-1), np.nan_to_num(MY[c][:, :-1], nan=-1)) for c in WD_COLS)))
np.savez_compressed(OUT / "my_columns.npz", basins=np.array(ids), days=days.values.astype("datetime64[D]"), zones=np.array([zones[g] for g in ids]), **MY)
out["written"] = str(OUT / "my_columns.npz")
out["seconds"] = round(time.time() - t0, 1)
jsave("r3_columns.json", out)
print("my columns against subdaily_inputs.add_inputs: all within 1e-9:", out["my_columns_vs_subdaily_inputs"]["all_basins_all_columns_within_1e_9"], "| clock all pass:", out["clock"]["all_basins_pass"],
      "| fill days equal:", out["maxcell_fill_days"]["equal_to_results_file"], "| thresholds equal:", out["thresholds"]["mine_equal_results_file"], "| zones plausible:", out["zones_plausible_by_longitude"])
print("seconds:", out["seconds"])
