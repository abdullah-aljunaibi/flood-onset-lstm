"""Check (a), parts 3 and 4: the 63 basins and their tables.

 1. flood.candidate_ids() against the completeness rule of 01_data_audit.py applied to results/basin_audit.csv by the reviewer (63 basins, the file's order);
 2. the audit's candidate set recomputed from the CAMELS attribute tables (HUC 03 and 06, frac_snow < 0.1, aridity < 1, 100-2500 km2) and every audit number
    (n_days, missing_share, forcing_missing) recomputed from the RAW CAMELS files with the reviewer's own parser (m_common.raw_basin);
 3. completeness of flow and forcing of the 63 stated afresh: every day of 1980-10-01 .. 2014-09-30 present once in both files, flow >= 0, all seven forcing columns finite;
 4. the 63-basin cache (data/derived/camels63_daymet_<digest>.pkl) against tables rebuilt from raw by the reviewer, bit for bit; the digest in the file name;
 5. the 12 inside the 63-table against the stored 12-basin cache, bit for bit; the properties of the 63 against the reviewer's own join of the attribute tables;
 6. the river sets of the results file (12, 63, 51, 50) against the rule of the script.
Run: python m3_candidates_raw.py     (about 1-2 minutes: it parses 2 x 100 raw files)
"""
import time

from m_common import *

F = engine()
t0 = time.time()
res = {}

# ---- 1. candidate_ids() against the rule, from the csv --------------------------------------------------------------------------------------------
aud = pd.read_csv(PROJ / "results" / "basin_audit.csv", dtype={"gauge_id": str}).set_index("gauge_id")
mine = [g for g in aud.index if aud.loc[g, "missing_share"] == 0 and aud.loc[g, "forcing_missing"] == 0]
engine_ids = F.candidate_ids()
res["audit_csv"] = dict(rows=len(aud), complete_column_sum=int(aud.complete.sum()), rule_count=len(mine), complete_column_equals_rule=bool((aud.complete == ((aud.missing_share == 0) & (aud.forcing_missing == 0))).all()),
                        candidate_ids_equals_own_list=engine_ids == mine, n=len(engine_ids), unique=len(set(engine_ids)) == len(engine_ids), selected_in_csv=int(aud.selected.sum()))
print("1. audit csv rows", len(aud), "| complete by the rule", len(mine), "| flood.candidate_ids() equals the reviewer's list (same order):", engine_ids == mine)
sel12 = [r["gauge_id"] for r in json.loads((PROJ / "results" / "data_audit.json").read_text())["selected"]]
res["the_12"] = dict(ids=sel12, equal_to_study_ids=F.study_ids() == sel12, flagged_selected_in_csv=sorted(aud.index[aud.selected]) == sorted(sel12), all_12_complete=all(g in mine for g in sel12),
                     landscape_csv_equals_audit_json=all(aud.loc[g, "landscape"] == r["landscape"] for g, r in zip(sel12, json.loads((PROJ / "results" / "data_audit.json").read_text())["selected"])))
print("   the 12:", res["the_12"])

# ---- 2. the candidate set from the attribute tables, and every audit number from raw files --------------------------------------------------------
A = attribute_tables()
cand = A[A.huc_02.isin([3, 6]) & (A.frac_snow < 0.1) & (A.aridity < 1) & A.area_gages2.between(100, 2500)]
res["candidates"] = dict(n_from_attributes=len(cand), same_set_as_csv=set(cand.index) == set(aud.index), same_order_as_csv=list(cand.index) == list(aud.index), n_csv=len(aud))
print("2. candidates from the CAMELS attribute tables:", len(cand), "same set as the csv:", set(cand.index) == set(aud.index), "same order:", list(cand.index) == list(aud.index))
full = pd.date_range(*PERIOD)
rows, raw63 = {}, {}
for g in aud.index:
    forc, flow, area_m2, _ = raw_basin(g)
    fp = forc.loc[PERIOD[0]:PERIOD[1]]
    cfs = flow.cfs.reindex(full)                                  # a day with no row in the flow file is missing, like a -999
    miss = ~(cfs >= 0)
    q_present_rows = int(flow.loc[PERIOD[0]:PERIOD[1]].shape[0])
    rows[g] = dict(n_days=len(fp), missing_share=float(miss.mean()), forcing_missing=int(fp[["prcp", "tmax", "tmin", "srad", "vp"]].isna().sum().sum()),
                   forcing_all7_missing=int(fp.isna().sum().sum()), forcing_rows_unique=bool(forc.index.is_unique), flow_rows_in_period=q_present_rows, flow_rows_unique=bool(flow.index.is_unique),
                   forcing_dates_complete=bool(fp.index.equals(full)), flow_missing_days=int(miss.sum()), zero_flow_share=float((cfs == 0).mean()),
                   estimated_share=float(flow.flag.reindex(full).astype(str).str.contains("e").mean()))   # denominator: every day of the period, as the audit has it
R = pd.DataFrame(rows).T
cmp = dict(n_days_equal=bool((R.n_days.astype(int) == aud.n_days).all()), missing_share_max_abs_diff=float((R.missing_share.astype(float) - aud.missing_share).abs().max()),
           forcing_missing_equal=bool((R.forcing_missing.astype(int) == aud.forcing_missing).all()),
           zero_flow_share_max_abs_diff=float((R.zero_flow_share.astype(float) - aud.zero_flow_share).abs().max()), estimated_share_max_abs_diff=float((R.estimated_share.astype(float) - aud.estimated_share).abs().max()))
own_complete = [g for g in aud.index if R.loc[g, "missing_share"] == 0 and R.loc[g, "forcing_missing"] == 0]
cmp["complete_by_own_recount_equals_csv_complete"] = own_complete == mine
cmp["n_complete_by_own_recount"] = len(own_complete)
res["audit_recomputed_from_raw"] = cmp
print("   audit numbers recomputed from raw files for all", len(aud), "candidates:", cmp)

# ---- 3. the 63, stated afresh from the raw files -----------------------------------------------------------------------------------------------
c63 = {}
for g in mine:
    forc, flow, area_m2, _ = raw_basin(g)
    fp = forc.loc[PERIOD[0]:PERIOD[1]]
    cfs = flow.cfs.reindex(full)
    c63[g] = dict(forcing_dates_complete=bool(fp.index.equals(full)), forcing_rows_unique=bool(forc.index.is_unique), flow_dates_present=bool(flow.loc[PERIOD[0]:PERIOD[1]].index.equals(full)),
                  flow_rows_unique=bool(flow.index.is_unique), flow_all_nonnegative=bool((cfs >= 0).all()), flow_none_missing=bool(cfs.notna().all()), forcing7_all_finite=bool(np.isfinite(fp.values).all()),
                  prcp_nonneg=bool((fp.prcp >= 0).all()), area_km2=area_m2 / 1e6)
C = pd.DataFrame(c63).T
checks = {c: bool(C[c].all()) for c in C.columns if c != "area_km2"}
res["the_63_complete_from_raw"] = dict(checks=checks, n=len(C), all_pass=all(checks.values()), area_km2_range=[float(C.area_km2.min()), float(C.area_km2.max())])
print("3. the 63 afresh from raw:", checks, "| n =", len(C))

# ---- 4. the 63 cache against the reviewer's raw tables --------------------------------------------------------------------------------------------
import hashlib
ids12 = F.study_ids()
ids63 = ids12 + [g for g in engine_ids if g not in set(ids12)]
digest = hashlib.sha256(",".join(sorted(ids63)).encode()).hexdigest()[:10]
cache = PROJ / "data" / "derived" / f"camels63_daymet_{digest}.pkl"
D63 = pd.read_pickle(cache)
mine_tables = {g: raw_table(g) for g in ids63}
diff = {}
for g in ids63:
    a, b = D63[g], mine_tables[g]
    diff[g] = dict(index_equal=bool(a.index.equals(b.index)), columns_equal=list(a.columns) == list(b.columns), dtype_float64=bool((a.dtypes == "float64").all()),
                   bitwise=bool(all(a[c].values.tobytes() == b[c].values.tobytes() for c in a.columns)), max_abs_diff=float(np.max(np.abs(a.values - b.values))),
                   finite=bool(np.isfinite(a.values).all()))
Dd = pd.DataFrame(diff).T
res["cache63"] = dict(file=str(cache.relative_to(PROJ)), digest_in_name=digest, name_matches_rule=cache.name == f"camels63_daymet_{digest}.pkl", exists=cache.exists(), bytes=cache.stat().st_size, sha256=sha256(cache),
                      n_tables=len(D63), keys_equal_ids63=list(D63) == ids63, rows=sorted({len(d) for d in D63.values()}), all_index_equal=bool(Dd.index_equal.all()), all_columns_equal=bool(Dd.columns_equal.all()),
                      all_bitwise_equal_to_reviewer_raw=bool(Dd.bitwise.all()), max_abs_diff_any=float(Dd.max_abs_diff.astype(float).max()), all_finite=bool(Dd.finite.all()), n_bitwise=int(Dd.bitwise.sum()))
print("4. cache63", cache.name, "| keys = ids63 in the script's order:", list(D63) == ids63, "| bitwise equal to the reviewer's raw tables:", int(Dd.bitwise.sum()), "of", len(Dd), "| max abs diff", res["cache63"]["max_abs_diff_any"])
# the engine reading the cache for ids in another order
D63_rev = F.load_data(ids=list(reversed(ids63)))
res["cache63"]["load_data_reversed_ids_gives_that_order_and_same_tables"] = bool(list(D63_rev) == list(reversed(ids63)) and all(D63_rev[g].equals(D63[g]) for g in ids63))

# ---- 5. the 12 inside the 63 against the stored 12; properties -----------------------------------------------------------------------------------------
D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
res["twelve_in_63"] = dict(keys_equal=list(D12) == ids12, bitwise_equal=all(D63[g][c].values.tobytes() == D12[g][c].values.tobytes() and D63[g].index.equals(D12[g].index) for g in ids12 for c in D12[g].columns),
                           equals=all(D63[g].equals(D12[g]) for g in ids12))
print("5. the 12 inside the 63-table equal the stored 12 tables:", res["twelve_in_63"])
Sown = static_table(ids63, A)
Seng = F.basins(ids63)[F.STATIC].astype(float)
Seng12 = F.basins()[F.STATIC].astype(float)
res["properties"] = dict(own_finite=bool(np.isfinite(Sown.values).all()), max_abs_diff_engine_vs_own_63=float(np.max(np.abs(Seng.loc[ids63].values - Sown.loc[ids63].values))),
                         the_12_rows_equal_flood_basins=bool(Seng.loc[ids12].equals(Seng12)), static_list_equals_flood_STATIC=list(F.STATIC) == STATIC,
                         n_properties_from_flow_record=0, nan_in_engine_table=int(np.isnan(Seng.values).sum()),
                         landscape_csv_for_the_63_counts=aud.loc[ids63, "landscape"].value_counts().to_dict())
print("   properties: ", {k: v for k, v in res["properties"].items() if k != "landscape_csv_for_the_63_counts"})
# the landscape rule of 01_data_audit.py recomputed
land = np.where(aud.elev_mean > 600, "Mountains (above 600 m)", np.where(aud.gauge_lat < 32.5, "Southern lowlands (south of 32.5N)", "Northern lowlands and Piedmont"))
res["landscape_rule_equals_csv"] = bool((land == aud.landscape.values).all())

# ---- 6. the river sets in the results file ------------------------------------------------------------------------------------------------------
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
riv = J["design"]["rivers"]
ids51 = ids63[12:]
ids50 = [g for g in ids51 if g != "03471500"]
res["river_sets_in_results_file"] = dict(twelve_equal=riv["twelve"] == ids12, sixty_three_equal=riv["sixty_three"] == ids63, other_51_equal=riv["other_51"] == ids51, other_50_equal=riv["other_50"] == ids50,
                                         n=(len(riv["twelve"]), len(riv["sixty_three"]), len(riv["other_51"]), len(riv["other_50"])), disjoint_12_50=not (set(ids12) & set(ids50)), dropped=riv["dropped_for_the_never_seen_test"]["gauge_id"],
                                         cache_file_in_json=J["data_checks"]["cache_file"], cache_file_matches=J["data_checks"]["cache_file"].endswith(cache.name))
print("6. river sets of the results file:", res["river_sets_in_results_file"])
res["seconds"] = round(time.time() - t0, 1)
ok = (res["audit_csv"]["candidate_ids_equals_own_list"] and res["audit_csv"]["n"] == 63 and res["the_63_complete_from_raw"]["all_pass"] and res["cache63"]["all_bitwise_equal_to_reviewer_raw"]
      and res["twelve_in_63"]["bitwise_equal"] and cmp["complete_by_own_recount_equals_csv_complete"] and res["candidates"]["same_set_as_csv"])
res["all_pass"] = bool(ok)
print("\nALL PASS:", ok, f"({res['seconds']} s)")
save("m3_candidates_raw", res)
sys.exit(0 if ok else 1)
