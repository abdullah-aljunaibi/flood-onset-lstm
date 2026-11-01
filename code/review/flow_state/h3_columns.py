"""E4e review, check 3: the hourly-flow columns rebuilt INDEPENDENTLY from data/hourly_flow/usgs_iv_<gauge>.npz for all 12 basins and both variants (24:00 and 22:00) with the
reviewer's own code, and compared with the SCRIPT's columns (its flow_state_table function, executed from the script text; the script cannot be imported) on the days it marks
hourly_known = 1; the fallback rules (carry of at most CARRY_H = 6 h, the daily-mean fallback, the 50% mismatch rule, South Toe 2004-09-17); the counts per basin against
data_coverage; the area and Q95 used; the local-day mean against the CAMELS daily mean, and which clock the USGS daily values follow; the made-up-hours test (own code,
on both builders, 2 basins x 2 variants + clock-change days). Writes OUT/my_columns.npz (the reviewer's own columns) for h2 and h4. About two minutes.

The reviewer's route: for every local calendar day d the UTC instants of 00:00 local and of the hours ending H:00 local (H = 24, 18, 12 or 22, 16, 10) come from zoneinfo
(ambiguous or nonexistent instants would raise); the day's hours are the stamps in (00:00 of d, H:00 of d] on the UTC axis, found with searchsorted; per-day aggregates
by numpy.reduceat over contiguous slices; the carry by index arithmetic (a missing hour takes the last finite value if it lies at most 6 positions back). No pandas groupby,
no 'minus one minute', no shared code with the script."""
from h_common import *
import glob
from zoneinfo import ZoneInfo
import flood as F
import subdaily_inputs as SI

t0 = time.time()
R = jload(RESULTS)
COV = R["data_coverage"]["per_basin"]
ZONES = zones()
ids = R["basins"]
dates, basins, OBS, _, _ = load_npz("lstm_wqa_s")
assert basins == ids
TR = within(dates, *SAT_SPLIT["train"])
Q95 = {g: float(np.quantile(OBS[i][TR], 0.95)) for i, g in enumerate(ids)}             # own Q95: the 95th percentile of the daily flow of the TRAINING years of the split
D = F.load_data()
assert all(d.index.equals(dates) for d in D.values()) and all(np.allclose(D[g].q_mm.values, OBS[i]) for i, g in enumerate(ids)), "the stored obs are not the CAMELS daily flow"
out = dict(thresholds=dict(mine=Q95, file_equal=all(abs(Q95[g] - R["thresholds"][g]) < 1e-9 for g in ids), engine_equal=all(abs(Q95[g] - F.thresholds(D, split=SAT_SPLIT)[g]) < 1e-12 for g in ids),
                           source="numpy.quantile(obs of 2000-10-01..2008-09-30, 0.95) per basin, from the stored obs array of lstm_wqa_s"))


def area_from_forcing_header(g):
    f = glob.glob(str(PROJ / "data" / "camels_us" / "basin_dataset_public_v1p2" / "basin_mean_forcing" / "daymet" / "*" / f"{g}_lump_cida_forcing_leap.txt"))[0]
    with open(f) as fh:
        fh.readline(); fh.readline()
        return float(fh.readline()) / 1e6


AREA = {g: area_from_forcing_header(g) for g in ids}
B = F.basins()
out["areas"] = {g: dict(forcing_header_km2=AREA[g], attrs_area_km2=float(D[g].attrs.get("area_km2", np.nan)), file_area_km2_used=COV[g]["area_km2_used"], area_gages2_km2=float(B.area_gages2[g]), file_area_gages2=COV[g]["area_gages2_km2"],
                        file_used_the_forcing_header_area=abs(COV[g]["area_km2_used"] - AREA[g]) < 1e-6, gages2_over_header=float(B.area_gages2[g]) / AREA[g]) for g in ids}
out["areas_all_from_forcing_header"] = all(v["file_used_the_forcing_header_area"] for v in out["areas"].values())
print("Q95 equals the file:", out["thresholds"]["file_equal"], "| area used = CAMELS forcing header in all basins:", out["areas_all_from_forcing_header"],
      "| gages2 / header ratio range:", round(min(v["gages2_over_header"] for v in out["areas"].values()), 4), round(max(v["gages2_over_header"] for v in out["areas"].values()), 4))
MM = {g: 0.0283168 * 86400 / (AREA[g] * 1e6) * 1000 for g in ids}


def hourly(g):
    z = np.load(PROJ / "data" / "hourly_flow" / f"usgs_iv_{g}.npz", allow_pickle=False)
    stamp = pd.DatetimeIndex(pd.to_datetime(z["stamp_utc_s"], unit="s"))
    assert (np.diff(z["stamp_utc_s"]) == 3600).all(), f"{g}: the hourly axis is not complete"
    return stamp, z["q_mean_cfs"].astype(float), z["q_last_cfs"].astype(float), z["approved"].astype(float), z["n_values"].astype(int)


def utc_of(days, hour, zone):
    """The UTC instant of the wall-clock time `hour`:00 of each local day (hour 24 = 00:00 of the next day); raises on an ambiguous or nonexistent instant."""
    t = pd.DatetimeIndex(days + pd.Timedelta(hours=hour))
    return t.tz_localize(ZoneInfo(zone), ambiguous="raise", nonexistent="raise").tz_convert("UTC").tz_localize(None)


def my_columns(stamp, qm_cfs, ql_cfs, zone, days, q_day, area_km2, q95, L, carry_h=CARRY_H):
    """The reviewer's five columns of the issue day (last available hour ending L:00 local) plus hourly_known, hourly_missing_hours, hourly_mismatch, and the day's hour count."""
    mm = 0.0283168 * 86400 / (area_km2 * 1e6) * 1000
    qm, ql = qm_cfs * mm, ql_cfs * mm
    n = len(stamp)
    fin = np.isfinite(qm)
    assert np.array_equal(fin, np.isfinite(ql)), "q_mean and q_last have different missing hours"
    idx = np.arange(n)
    last_fin = np.maximum.accumulate(np.where(fin, idx, -1))
    age = idx - last_fin
    filled = (last_fin >= 0) & (age <= carry_h)                                           # known or carried from a value at most carry_h hours back
    src = np.clip(last_fin, 0, None)
    qm_f, ql_f = np.where(filled, qm[src], np.nan), np.where(filled, ql[src], np.nan)
    carried, gap = filled & ~fin, ~filled
    start, end = utc_of(days, 0, zone), utc_of(days, L, zone)                          # 00:00 local of d; L:00 local of d (the last available hour's end)
    i0, i1 = stamp.searchsorted(start), stamp.searchsorted(end)                          # positions of those instants on the axis (hour-ending stamps)
    in_axis = (i0 < n) & (i1 < n)
    assert in_axis[:-1].all(), "a day before the last one is not on the axis"
    assert (stamp[np.clip(i0, 0, n - 1)] == start)[in_axis].all() and (stamp[np.clip(i1, 0, n - 1)] == end)[in_axis].all(), "a local instant is not on the hour"
    lo, hi = i0 + 1, np.minimum(i1 + 1, n)                                               # the day's hours: positions lo .. hi-1 (the hour ending at 00:00 belongs to the day before)
    hi = np.where(in_axis, hi, n)
    n_hours = hi - lo
    # per-day aggregates over contiguous slices [lo_i, hi_i) with reduceat (for L = 22 the slices are not contiguous: hours 23, 24 of each day are left out, so build explicit segments)
    def seg_sum(x):
        c = np.concatenate([[0.0], np.cumsum(x.astype(float))])
        return c[hi] - c[lo]
    def seg_max(x):
        x = np.where(np.isfinite(x), x, -np.inf)
        mx = np.array([x[a:b].max() if b > a else -np.inf for a, b in zip(lo, hi)])
        return np.where(np.isfinite(mx), mx, np.nan)
    n_gap, n_carried = seg_sum(gap).astype(int), seg_sum(carried).astype(int)
    n_avail = seg_sum(np.isfinite(qm_f))
    dmean = np.where(n_avail > 0, seg_sum(np.where(np.isfinite(qm_f), qm_f, 0.0)) / np.where(n_avail > 0, n_avail, 1), np.nan)   # the mean over the available (known or carried) hours
    q_dmax = seg_max(qm_f)
    hours_above = seg_sum(np.where(np.isfinite(qm_f), qm_f > q95, False)).astype(int)
    def at(hour):
        p = stamp.searchsorted(utc_of(days, hour, zone))
        okp = p < n
        v = np.full(len(days), np.nan)
        v[okp] = ql_f[p[okp]]
        return v
    q_end, q6, q12 = at(L), at(L - 6), at(L - 12)
    qd = np.asarray(q_day, float)
    mismatch = (qd > Q_MIN_FOR_MISMATCH) & (np.abs(dmean - qd) > MISMATCH_REL * qd)
    mismatch = np.where(np.isfinite(dmean), mismatch, False)
    known = (n_gap == 0) & np.isfinite(q_end) & np.isfinite(q6) & np.isfinite(q12) & ~mismatch
    lg = lambda q: np.log1p(q / q95)
    cols = dict(q_end=lg(q_end), dq6=lg(q_end) - lg(q6), dq12=lg(q_end) - lg(q12), q_dmax=lg(q_dmax), hours_above=hours_above.astype(float))
    fb = dict(q_end=lg(qd), dq6=np.zeros(len(days)), dq12=np.zeros(len(days)), q_dmax=lg(qd), hours_above=float(L) * (qd > q95))
    for c in cols:
        cols[c] = np.where(known, cols[c], fb[c])
    cols["hourly_known"] = known.astype(int)
    cols["hourly_missing_hours"] = n_carried + n_gap
    cols["hourly_mismatch"] = mismatch.astype(int)
    cols["n_hours"] = n_hours
    cols["_dmean"] = dmean
    cols["_n_gap"] = n_gap
    cols["_n_carried"] = n_carried
    cols["_mismatch_alone"] = (mismatch & (n_gap == 0) & np.isfinite(q_end) & np.isfinite(q6) & np.isfinite(q12)).astype(int)     # days the mismatch rule alone sends to the fallback
    cols["_known_hour_mean"] = np.where(n_avail - n_carried > 0, seg_sum(np.where(fin[:n] & np.isfinite(qm_f), qm_f, 0.0)) / np.maximum(n_avail - n_carried, 1), np.nan)   # the mean of the hours with values (no carry)
    return cols


# ---- the script's function, executed from its text with the names it needs (the brief: compare with the SCRIPT's columns) -------------------------------------------------
ns = dict(pd=pd, np=np, SI=SI, ZONES=ZONES, CARRY_H=CARRY_H, MISMATCH_REL=MISMATCH_REL, thr=R["thresholds"], MM=lambda g: 0.0283168 * 86400 / (COV[g]["area_km2_used"] * 1e6) * 1000)
flow_state_table, fst_src = script_function("flow_state_table", ns)
out["script_function"] = dict(name="flow_state_table", source_sha256=hashlib.sha256(fst_src.encode()).hexdigest(), lines=fst_src.count("\n") + 1)

MY, SCR, cmp, hourly_data = {}, {}, {}, {}
ALL_COLS = list(H_COLS) + ["hourly_known", "hourly_missing_hours", "hourly_mismatch"]
for g in ids:
    stamp, qm, ql, appr, nval = hourly(g)
    hourly_data[g] = (stamp, qm, ql)
    q_day = D[g].q_mm.reindex(T_ISS)
    mine = {L: my_columns(stamp, qm, ql, ZONES[g], T_ISS, q_day.values, AREA[g], Q95[g], L) for L in (24, 22)}
    scr = {L: flow_state_table(stamp, qm, ql, g, q_day, L) for L in (24, 22)}
    MY[g], SCR[g] = mine, scr
    rec = {}
    for L in (24, 22):
        m, s = mine[L], scr[L]
        known_s = s.hourly_known.values == 1
        d = {c: float(np.nanmax(np.abs(m[c][known_s] - s[c].values[known_s]))) for c in H_COLS}
        d_all = {c: float(np.nanmax(np.abs(m[c] - s[c].values))) for c in H_COLS}
        flags_equal = dict(hourly_known=bool(np.array_equal(m["hourly_known"], s.hourly_known.values)), hourly_missing_hours=bool(np.array_equal(m["hourly_missing_hours"], s.hourly_missing_hours.values)),
                           hourly_mismatch=bool(np.array_equal(m["hourly_mismatch"], s.hourly_mismatch.values)))
        rec[str(L)] = dict(max_abs_diff_on_known_days=d, max_abs_diff_all_days=d_all, all_within_1e_9_on_known_days=max(d.values()) < 1e-9, all_within_1e_9_all_days=max(d_all.values()) < 1e-9, quality_flags_equal=flags_equal,
                           n_known_mine=int(m["hourly_known"].sum()), n_fallback_mine=int((m["hourly_known"] == 0).sum()), n_fallback_script=int((s.hourly_known == 0).sum()),
                           n_fallback_test_mine=int((m["hourly_known"][within(T_ISS, *SAT_SPLIT["test"])] == 0).sum()), n_mismatch_mine=int(m["hourly_mismatch"].sum()),
                           hours_carried_on_known_days_mine=int(m["hourly_missing_hours"][m["hourly_known"] == 1].sum()), hours_in_day=dict(zip(*[x.tolist() for x in np.unique(m["n_hours"], return_counts=True)])),
                           hours_above_max=float(np.max(m["hours_above"])), fallback_days_in_file=COV[g][f"fallback_days_{L}"], fallback_test_in_file=COV[g][f"fallback_days_{L}_test_years"], mismatch_in_file=COV[g][f"mismatch_days_{L}"])
        rec[str(L)]["counts_equal_file"] = (rec[str(L)]["n_fallback_mine"] == COV[g][f"fallback_days_{L}"] and rec[str(L)]["n_fallback_test_mine"] == COV[g][f"fallback_days_{L}_test_years"] and rec[str(L)]["n_mismatch_mine"] == COV[g][f"mismatch_days_{L}"])
    rec["24"]["hours_carried_equal_file"] = rec["24"]["hours_carried_on_known_days_mine"] == COV[g]["hours_carried_24"]
    rec["24"]["mismatch_dates_mine"] = [str(x.date()) for x in T_ISS[mine[24]["hourly_mismatch"] == 1]]
    rec["24"]["mismatch_dates_equal_file"] = rec["24"]["mismatch_dates_mine"] == COV[g]["mismatch_dates_24"]
    # coverage: the gaps after 2000 (hours without values before any carry), the approved share, the local-day mean against the CAMELS daily mean
    after = stamp >= pd.Timestamp("2000-01-01")
    miss = ~np.isfinite(qm[after])
    rec["coverage"] = dict(missing_hours_after_2000=int(miss.sum()), file=COV[g]["missing_hours_after_2000"], equal=int(miss.sum()) == COV[g]["missing_hours_after_2000"], approved_share=float(np.nanmean(appr[after])),
                           approved_share_file=COV[g]["approved_share_after_2000"], values_all_approved=bool(np.nanmin(appr[after]) == 1.0), n_values_per_hour_median=float(np.median(nval[after][nval[after] > 0])),
                           first_hour_end_utc=str(stamp[0]), last_hour_end_utc=str(stamp[-1]), last_hour_end_local=str(stamp[-1].tz_localize("UTC").tz_convert(ZoneInfo(ZONES[g])).tz_localize(None)))
    cmp[g] = rec
    k24 = mine[24]["hourly_known"] == 1
    print(f"{g}: 24h max diff on known days {max(rec['24']['max_abs_diff_on_known_days'].values()):.1e} (all days {max(rec['24']['max_abs_diff_all_days'].values()):.1e}), flags equal {all(rec['24']['quality_flags_equal'].values())}, "
          f"fallback {rec['24']['n_fallback_mine']} (file {COV[g]['fallback_days_24']}), mismatch {rec['24']['n_mismatch_mine']} (file {COV[g]['mismatch_days_24']}); 22h max diff {max(rec['22']['max_abs_diff_on_known_days'].values()):.1e}, "
          f"flags equal {all(rec['22']['quality_flags_equal'].values())}, fallback {rec['22']['n_fallback_mine']} (file {COV[g]['fallback_days_22']}); hours in a day {rec['24']['hours_in_day']}  [{time.time() - t0:.0f} s]", flush=True)
out["columns_vs_script"] = cmp
out["all_basins_both_variants_equal_on_known_days"] = all(cmp[g][L]["all_within_1e_9_on_known_days"] and all(cmp[g][L]["quality_flags_equal"].values()) for g in ids for L in ("24", "22"))
out["all_basins_both_variants_equal_all_days"] = all(cmp[g][L]["all_within_1e_9_all_days"] for g in ids for L in ("24", "22"))
out["all_counts_equal_file"] = all(cmp[g][L]["counts_equal_file"] for g in ids for L in ("24", "22")) and all(cmp[g]["24"]["hours_carried_equal_file"] and cmp[g]["24"]["mismatch_dates_equal_file"] for g in ids)
out["fallback_total_mine"] = {L: sum(cmp[g][L]["n_fallback_mine"] for g in ids) for L in ("24", "22")}
out["fallback_total_file"] = R["data_coverage"]["fallback_days_total"]
out["fallback_test_total_mine"] = sum(cmp[g]["24"]["n_fallback_test_mine"] for g in ids)
out["mismatch_total_mine_24"] = sum(cmp[g]["24"]["n_mismatch_mine"] for g in ids)
out["basin_days_total"] = len(ids) * len(T_ISS)
print("columns equal to the script's on known days (both variants, all basins):", out["all_basins_both_variants_equal_on_known_days"], "| on all days:", out["all_basins_both_variants_equal_all_days"],
      "| counts equal the file:", out["all_counts_equal_file"], "| fallback totals mine", out["fallback_total_mine"], "file", out["fallback_total_file"])

# ---- the fallback rules on the data: the carry (a gap of 1-6 h is carried, 7+ falls back), the daily-mean fallback values, South Toe 2004-09-17 --------------------------------
g = "03463300"
stamp, qm, ql = hourly_data[g]
m24, s24 = MY[g][24], SCR[g][24]
i = int(np.flatnonzero(T_ISS == pd.Timestamp("2004-09-17"))[0])
south_toe = dict(date="2004-09-17", camels_daily_mean_mm=float(D[g].q_mm.loc["2004-09-17"]), hourly_mean_over_available_hours_mm=float(m24["_dmean"][i]), mean_of_the_hours_with_values_mm=float(m24["_known_hour_mean"][i]),
                 hours_still_missing_after_the_carry=int(m24["_n_gap"][i]), hours_carried=int(m24["_n_carried"][i]), relative_difference=float(abs(m24["_dmean"][i] - D[g].q_mm.loc["2004-09-17"]) / D[g].q_mm.loc["2004-09-17"]),
                 mismatch_mine=int(m24["hourly_mismatch"][i]), mismatch_script=int(s24.hourly_mismatch.iloc[i]), hourly_known_mine=int(m24["hourly_known"][i]), fallback_values_mine={c: float(m24[c][i]) for c in H_COLS},
                 falls_back_by_the_gap_rule_already=bool(m24["_n_gap"][i] > 0), decided_by_the_mismatch_rule_alone=bool(m24["_mismatch_alone"][i]),
                 expected_fallback={"q_end": float(np.log1p(D[g].q_mm.loc["2004-09-17"] / Q95[g])), "dq6": 0.0, "dq12": 0.0, "q_dmax": float(np.log1p(D[g].q_mm.loc["2004-09-17"] / Q95[g])), "hours_above": 24.0 * float(D[g].q_mm.loc["2004-09-17"] > Q95[g])},
                 in_file_mismatch_dates="2004-09-17" in COV[g]["mismatch_dates_24"], in_test_years=False)
south_toe["fallback_values_as_documented"] = all(abs(south_toe["fallback_values_mine"][c] - south_toe["expected_fallback"][c]) < 1e-12 for c in H_COLS)
# the hourly record of that day on the wall clock, for the reader
loc = stamp.tz_localize("UTC").tz_convert(ZoneInfo(ZONES[g])).tz_localize(None)
day_rows = (loc > pd.Timestamp("2004-09-17")) & (loc <= pd.Timestamp("2004-09-18"))
south_toe["hourly_means_cfs_that_day"] = [None if not np.isfinite(x) else round(float(x), 1) for x in qm[day_rows]]
south_toe["hours_with_values_that_day"] = int(np.isfinite(qm[day_rows]).sum())
south_toe["peak_cfs_of_the_day_in_the_15min_record"] = float(np.nanmax(qm[day_rows])) if np.isfinite(qm[day_rows]).any() else None
out["south_toe_2004_09_17"] = south_toe
print(f"South Toe 2004-09-17: CAMELS {south_toe['camels_daily_mean_mm']:.1f} mm/day, hourly mean over the available hours {south_toe['hourly_mean_over_available_hours_mm']:.1f} ({south_toe['hours_with_values_that_day']} hours with values, "
      f"their mean {south_toe['mean_of_the_hours_with_values_mm']:.1f}; {south_toe['hours_still_missing_after_the_carry']} hours still missing after the carry), mismatch mine {south_toe['mismatch_mine']} script {south_toe['mismatch_script']}, "
      f"fallback values as documented {south_toe['fallback_values_as_documented']}, falls back by the gap rule already {south_toe['falls_back_by_the_gap_rule_already']}")
# the mismatch rule: on how many days does it alone decide the fallback (every hour known or carried, q_end / q6 / q12 present, yet the hourly mean is off by more than 50%)?
alone = {g: dict(mismatch_days_24=int(MY[g][24]["hourly_mismatch"].sum()), decided_by_the_mismatch_rule_alone_24=int(MY[g][24]["_mismatch_alone"].sum()), dates_alone_24=[str(x.date()) for x in T_ISS[MY[g][24]["_mismatch_alone"] == 1]],
                 decided_by_the_mismatch_rule_alone_22=int(MY[g][22]["_mismatch_alone"].sum()),
                 details_24=[dict(date=str(T_ISS[j].date()), camels_mm=float(D[g].q_mm.reindex(T_ISS).values[j]), hourly_mean_mm=float(MY[g][24]["_dmean"][j]), carried_hours=int(MY[g][24]["_n_carried"][j]), in_test_years=bool(within(T_ISS, *SAT_SPLIT["test"])[j]))
                             for j in np.flatnonzero(MY[g][24]["_mismatch_alone"] == 1)]) for g in ids}
out["mismatch_rule_alone"] = dict(per_basin=alone, total_mismatch_days_24=sum(v["mismatch_days_24"] for v in alone.values()), total_decided_alone_24=sum(v["decided_by_the_mismatch_rule_alone_24"] for v in alone.values()),
                                  total_decided_alone_22=sum(v["decided_by_the_mismatch_rule_alone_22"] for v in alone.values()),
                                  in_test_years_24=sum(1 for v in alone.values() for d in v["details_24"] if d["in_test_years"]))
print(f"mismatch rule: {out['mismatch_rule_alone']['total_mismatch_days_24']} mismatch days (24:00), of which {out['mismatch_rule_alone']['total_decided_alone_24']} are decided by the mismatch rule alone "
      f"({out['mismatch_rule_alone']['in_test_years_24']} in the test years); 22:00: {out['mismatch_rule_alone']['total_decided_alone_22']} alone")
# the carry rule, counted over all basins: on known days, how many hours were carried and the longest carried run inside a known day; a gap of 7+ hours always falls back
carry = {}
for g in ids:
    stamp, qm, ql = hourly_data[g]
    m = MY[g][24]
    known = m["hourly_known"] == 1
    fin = np.isfinite(qm)
    idx = np.arange(len(qm))
    last_fin = np.maximum.accumulate(np.where(fin, idx, -1))
    age = idx - last_fin
    start, end = utc_of(T_ISS, 0, ZONES[g]), utc_of(T_ISS, 24, ZONES[g])
    i0, i1 = stamp.searchsorted(start), np.minimum(stamp.searchsorted(end), len(stamp) - 1)
    max_age_known = max((int(age[a + 1:b + 1].max()) for a, b, k in zip(i0, i1, known) if k and b > a), default=0)
    days_with_gap_over_6 = int(sum(1 for a, b in zip(i0, i1) if b > a and (age[a + 1:b + 1] > 6).any()))
    carry[g] = dict(max_hours_since_a_value_on_known_days=max_age_known, days_with_an_hour_more_than_6h_from_a_value=days_with_gap_over_6, those_days_all_fallback=all(not known[j] for j, (a, b) in enumerate(zip(i0, i1)) if b > a and (age[a + 1:b + 1] > 6).any()),
                    hours_carried_on_known_days=int(m["hourly_missing_hours"][known].sum()))
out["carry_rule"] = dict(per_basin=carry, carry_never_exceeds_6h_on_known_days=all(v["max_hours_since_a_value_on_known_days"] <= 6 for v in carry.values()), every_day_with_a_7h_gap_falls_back=all(v["those_days_all_fallback"] for v in carry.values()))
print("carry rule: at most 6 h on known days in every basin:", out["carry_rule"]["carry_never_exceeds_6h_on_known_days"], "| every day with a 7+ h gap falls back:", out["carry_rule"]["every_day_with_a_7h_gap_falls_back"])

# ---- the clock: the local-day mean of the hourly flow against the CAMELS daily mean, with the wall clock (summer time) and with standard time all year ----------------------------
clock = {}
for g in ids:
    stamp, qm, ql = hourly_data[g]
    q_day = D[g].q_mm.reindex(T_ISS).values
    res = {}
    for label, zone in (("wall_clock_with_summer_time", ZONES[g]), ("standard_time_all_year", "Etc/GMT+5" if ZONES[g] == "America/New_York" else "Etc/GMT+6")):
        start, end = utc_of(T_ISS, 0, zone), utc_of(T_ISS, 24, zone)
        i0, i1 = stamp.searchsorted(start), stamp.searchsorted(end)
        okd = (i1 < len(stamp))
        means = np.full(len(T_ISS), np.nan)
        full = np.zeros(len(T_ISS), bool)
        for j in np.flatnonzero(okd):
            x = qm[i0[j] + 1:i1[j] + 1]
            if np.isfinite(x).all() and len(x) >= 23:
                means[j], full[j] = x.mean() * MM[g], True
        summer = full & np.isin(T_ISS.month, [6, 7, 8]) & (q_day > 0)
        rel = np.abs(means[summer] - q_day[summer]) / q_day[summer]
        res[label] = dict(days_compared_summer=int(summer.sum()), median_relative_difference_summer=float(np.median(rel)), mean_relative_difference_summer=float(rel.mean()), share_within_1pct_summer=float((rel < 0.01).mean()),
                          corr_all_full_days=float(np.corrcoef(means[full & (q_day > 0)], q_day[full & (q_day > 0)])[0, 1]), ratio_of_sums_all_full_days=float(means[full & (q_day > 0)].sum() / q_day[full & (q_day > 0)].sum()))
    res["summer_days_fit_better_with"] = min(("wall_clock_with_summer_time", "standard_time_all_year"), key=lambda k: res[k]["median_relative_difference_summer"])
    clock[g] = res
out["usgs_daily_value_clock"] = dict(what="the local-day mean of the hourly flow (days with every hour known) against the CAMELS daily mean, in June-August, with the day bounded by the wall clock including summer time (the script's convention) "
                                          "and by standard time all year: which bound reproduces the USGS daily values?", per_basin=clock, basins_fitting_standard_time_better=[g for g in ids if clock[g]["summer_days_fit_better_with"] == "standard_time_all_year"],
                                     file_corr_days_used={g: COV[g]["local_day_mean_vs_camels"]["corr_days_used"] for g in ids})
print("USGS daily values follow standard time in", len(out["usgs_daily_value_clock"]["basins_fitting_standard_time_better"]), "of 12 basins (summer median relative difference, wall clock vs standard):",
      {g[-4:]: (round(clock[g]["wall_clock_with_summer_time"]["median_relative_difference_summer"], 4), round(clock[g]["standard_time_all_year"]["median_relative_difference_summer"], 4)) for g in ids})

# ---- the made-up-hours test, the reviewer's own, on both builders: 2 basins x 2 variants x 3 cut days (+ the clock-change days of the test years) --------------------------------
rng = np.random.default_rng(20261029)
mu = {}
for g in ("02077200", "02472000"):                                                        # New York and Chicago zones
    stamp, qm, ql = hourly_data[g]
    q_day = D[g].q_mm.reindex(T_ISS)
    ok_both = (MY[g][24]["hourly_known"] == 1) & (MY[g][22]["hourly_known"] == 1)
    ok_pair = ok_both & np.roll(ok_both, -1)
    cand = T_ISS[ok_pair & within(T_ISS, *SAT_SPLIT["test"])]
    cuts = [cand[len(cand) // 5], cand[len(cand) // 2], cand[4 * len(cand) // 5]] + [d for d in (pd.Timestamp("2011-11-06"), pd.Timestamp("2013-03-10"), pd.Timestamp("2012-11-04")) if d in cand]
    rows = []
    for L in (24, 22):
        base_m, base_s = MY[g][L], SCR[g][L]
        base_m24, base_s24 = MY[g][24], SCR[g][24]
        for cut in cuts:
            instant = utc_of(pd.DatetimeIndex([cut]), L, ZONES[g])[0]
            later = np.asarray(stamp > instant)
            qm2 = np.where(later, rng.uniform(0, 5000, len(qm)), qm)
            ql2 = np.where(later, rng.uniform(0, 5000, len(ql)), ql)
            alt_m = my_columns(stamp, qm2, ql2, ZONES[g], T_ISS, q_day.values, AREA[g], Q95[g], L)
            alt_s = flow_state_table(stamp, qm2, ql2, g, q_day, L)
            c_i, n_i = int(np.flatnonzero(T_ISS == cut)[0]), int(np.flatnonzero(T_ISS == cut)[0]) + 1
            same_m = all(np.array_equal(np.nan_to_num(base_m[c][:c_i + 1], nan=-1), np.nan_to_num(alt_m[c][:c_i + 1], nan=-1)) for c in ALL_COLS)
            same_s = base_s.iloc[:c_i + 1].fillna(-1).equals(alt_s.iloc[:c_i + 1].fillna(-1))
            moved_m = any(not np.array_equal(np.nan_to_num(base_m[c][n_i:n_i + 1], nan=-1), np.nan_to_num(alt_m[c][n_i:n_i + 1], nan=-1)) for c in ALL_COLS)
            moved_s = not base_s.iloc[n_i].fillna(-1).equals(alt_s.iloc[n_i].fillna(-1))
            # the 22:00 variant: the 24:00 columns of the cut day read the hours ending 23:00 and 24:00, so they must move
            if L == 22:
                alt_m24 = my_columns(stamp, qm2, ql2, ZONES[g], T_ISS, q_day.values, AREA[g], Q95[g], 24)
                alt_s24 = flow_state_table(stamp, qm2, ql2, g, q_day, 24)
                m24_moved = any(not np.array_equal(np.nan_to_num(base_m24[c][c_i:c_i + 1], nan=-1), np.nan_to_num(alt_m24[c][c_i:c_i + 1], nan=-1)) for c in H_COLS)
                s24_moved = not base_s24.iloc[c_i].fillna(-1).equals(alt_s24.iloc[c_i].fillna(-1))
            else:
                m24_moved = s24_moved = None
            # the sharpest form: only the FIRST hour after the issue instant is changed (one value): nothing up to the cut day may move
            one = np.asarray(stamp == instant + pd.Timedelta(hours=1))
            qm3, ql3 = np.where(one, 4999.0, qm), np.where(one, 4999.0, ql)
            alt1 = my_columns(stamp, qm3, ql3, ZONES[g], T_ISS, q_day.values, AREA[g], Q95[g], L)
            same_one = all(np.array_equal(np.nan_to_num(base_m[c][:c_i + 1], nan=-1), np.nan_to_num(alt1[c][:c_i + 1], nan=-1)) for c in ALL_COLS)
            rows.append(dict(last_hour=L, cut_day=str(cut.date()), issue_instant_utc=str(instant), hours_replaced=int(later.sum()), mine_rows_up_to_cut_unchanged=same_m, mine_next_day_moved=moved_m, script_rows_up_to_cut_unchanged=same_s,
                             script_next_day_moved=moved_s, columns_24_of_cut_day_moved_mine=m24_moved, columns_24_of_cut_day_moved_script=s24_moved, one_hour_after_the_instant_changed_rows_up_to_cut_unchanged=same_one,
                             clock_change_day=cut in (pd.Timestamp("2011-11-06"), pd.Timestamp("2013-03-10"), pd.Timestamp("2012-11-04"))))
    mu[g] = rows
flat = [r for v in mu.values() for r in v]
out["made_up_hours"] = dict(per_basin=mu, n_cases=len(flat), all_pass=all(r["mine_rows_up_to_cut_unchanged"] and r["mine_next_day_moved"] and r["script_rows_up_to_cut_unchanged"] and r["script_next_day_moved"] and r["one_hour_after_the_instant_changed_rows_up_to_cut_unchanged"]
                                                              and (r["last_hour"] == 24 or (r["columns_24_of_cut_day_moved_mine"] and r["columns_24_of_cut_day_moved_script"])) for r in flat),
                            what="every hourly value (mean and last) after the issue instant of the cut day (24:00 or 22:00 local) replaced by random numbers 0-5000 cfs; the reviewer's columns and the script's rebuilt: no row up to the cut day changed "
                                 "(all columns, quality flags included), the next day changed; for 22:00 the 24:00 columns of the cut day changed; and with only the first hour after the instant changed, no row up to the cut day changed")
print(f"made-up hours: {len(flat)} cases, all pass: {out['made_up_hours']['all_pass']}")

# ---- the reviewer's columns for h2 and h4 ---------------------------------------------------------------------------------------------------------------------------------------
arrs = {c: np.stack([MY[g][24][c] for g in ids]) for c in ALL_COLS}
arrs.update({c + "_22": np.stack([MY[g][22][c] for g in ids]) for c in ALL_COLS})
np.savez_compressed(OUT / "my_columns.npz", basins=np.array(ids), days=T_ISS.values.astype("datetime64[D]"), **arrs)
out["my_columns_file"] = str(OUT / "my_columns.npz")
out["seconds"] = round(time.time() - t0, 1)
jsave("h3_columns.json", out)
print("h3 done in", out["seconds"], "s")
