"""E2 review, check (d) and (j): the design tables. The reviewer's own rebuild (f_tables.py) of the tables the networks read, and what is checked on them and on the saved networks:
 (1) the columns of every set as the .pt files name them (dyn), the scalers stored with each network against the mean and standard deviation of MY table over the TRAINING years (the trainer computed them from the table it
     trained on: equality proves that the training-year values of the slot columns are the ones of the design: perfect rain in the training years of wqaF_sub / wqaF_clim, the forecast or the shuffled forecast in every year of
     E2b);
 (2) wqaF_sub: in the training years (issue days up to 2000-09-30) the slots hold the AORC rain of t+1..t+3, from 2000-10-01 the bias-corrected forecast; no AORC rain of a target day in the slots of the validation and test
     years, and the p columns are not inputs of wqaF_sub, wqaF_clim, wqF_sub, wqaF_s (their dyn lists); the columns are dated by the issue day (the AORC value of the slot of day t is the AORC rain of local day t+k);
 (3) the control wqaF_clim / wqaF_clim_s: same values as the forecast within every (basin, calendar month), no day keeps its own calendar year, the three leads of a day move together (each row of the control is a row
     of the forecast of another day of the same calendar month and another year), validation/test years only in E2a (the training years hold p), every year in E2b, correlation with the AORC rain of the day;
 (4) the history window: which rows a forecast can read (warmup 365 + block 365: at most 729 days back) and what that means for the first validation days (the slots of the last training rows hold the rain of 2000-10-01..03);
     a perturbation test on the saved wqaF_sub network: the slots of the training rows replaced by random values change the forecasts of the first validation days and no forecast later than 2003-10-01;
 (5) the forecast columns of the tables (variant 'csv') equal my own totals times my own factors (variant 'own') to the rounding of the tables.
About 1 minute (one forward pass of a network per perturbation)."""
import sys; sys.dont_write_bytecode = True
import pickle, glob, time
from f_common import *
from f_tables import *
import flood as F

t0 = time.time()
TAB, ids = build_tables("csv")
TAB_OWN, _ = build_tables("own")
pickle.dump((TAB, ids), open(f"{OUT}/f6_tables_csv.pkl", "wb")); pickle.dump((TAB_OWN, ids), open(f"{OUT}/f6_tables_own.pkl", "wb"))
B = F.basins()
out = {}

# (5) the two variants
dev = {a: max(float(np.abs(TAB[a][g][c].values - TAB_OWN[a][g][c].values).max()) for g in ids for c in TAB[a][g].columns if c[:2] in ("fs", "fr", "fk", "fc", "fx") and c[1] in "sRkcx") for a in TAB}
cols_dev = {a: {c: max(float(np.abs(TAB[a][g][c].values - TAB_OWN[a][g][c].values).max()) for g in ids) for c in TAB[a][ids[0]].columns if c[:2] in ("fs", "fr", "fk", "fc", "fx")} for a in TAB}
out["own_totals_vs_the_table_columns"] = dict(max_abs_difference_mm=dev, by_column={a: {c: v for c, v in d.items()} for a, d in cols_dev.items()})
print(f"(5) the forecast columns built from my own totals and my own factors against the columns of the tables: largest difference main {dev['main']:.2e} mm, satellite arm {dev['sat']:.2e} mm")

# (1) dyn lists and scalers
rows = {}
for name in SETS_E2:
    arm = ARM[name]; split = SPLITS[arm]
    for seed in SEEDS:
        z = torch.load(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
        dyn = list(z["dyn"]); sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
        tr = np.asarray((TAB[arm][ids[0]].index >= split["train"][0]) & (TAB[arm][ids[0]].index <= split["train"][1]))
        X = np.concatenate([TAB[arm][g].loc[tr, dyn].values for g in ids])
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        qtr = np.concatenate([TAB[arm][g].q_mm.values[tr] for g in ids])
        S = np.concatenate([B.loc[ids, list(F.STATIC)].values.astype(float)]) if z["cfg"]["static"] else None
        rows[f"{name}_s{seed}"] = dict(arm=arm, dyn=dyn, split_in_file=z["split"], split_expected_equal=(z["split"] == {k: list(v) for k, v in split.items()} or z["split"] == split), max_dev_dyn_mu=float(np.abs(mu - sc["dyn_mu"]).max()), max_dev_dyn_sd=float(np.abs(sd - sc["dyn_sd"]).max()),
                                         max_dev_q=float(max(abs(qtr.mean() - sc["q_mu"]), abs(qtr.std() - sc["q_sd"]))), best_update=int(z["best_update"]), updates_run=int(z["updates_run"]), n_checks=len(z["history"]), cfg_equal_lstm_all=(z["cfg"]["hidden"], z["cfg"]["lr"], z["cfg"]["dropout"], z["cfg"]["warmup"], z["cfg"]["block"], z["cfg"]["batch"], z["cfg"]["updates"], z["cfg"]["eval_every"], z["cfg"]["patience"]) == (64, 0.002, 0.2, 365, 365, 32, 2000, 100, 6),
                                         use_flow=bool(z["cfg"]["use_flow"]), future=bool(z["cfg"]["future"]), static=list(z["cfg"]["static"]) == list(F.STATIC), script=z["script"], quick_mode=bool(z["quick_mode"]))
out["networks"] = rows
mx_mu = max(r["max_dev_dyn_mu"] for r in rows.values()); mx_sd = max(r["max_dev_dyn_sd"] for r in rows.values()); mx_q = max(r["max_dev_q"] for r in rows.values())
print(f"(1) 30 networks: dyn lists read from the files; the stored scalers against the mean/sd of my table over the training years of the set's own split: largest difference of the means {mx_mu:.2e}, of the sds {mx_sd:.2e}, of the flow scale {mx_q:.2e}; "
      f"all settings equal results/lstm_all.json: {all(r['cfg_equal_lstm_all'] for r in rows.values())}; use_flow {all(r['use_flow'] for r in rows.values())}; future flag false {not any(r['future'] for r in rows.values())}; quick mode {any(r['quick_mode'] for r in rows.values())}")
dyn_of = {n: rows[f"{n}_s0"]["dyn"] for n in SETS_E2}
for n, d in dyn_of.items(): print(f"    {n:14s} dyn = {d}")
out["dyn_by_set"] = dyn_of
ok_dyn = {n: [c for c in d if c[0] == "p" and c[1:].isdigit()] for n, d in dyn_of.items()}
out["p_columns_in_dyn"] = ok_dyn
print("    sets whose input columns include the perfect-rain columns p1..p3:", [n for n, v in ok_dyn.items() if v])

# (2) wqaF_sub design on the table
T_ = TAB["main"]; d0 = pd.Timestamp(SUB_FROM)
chk2 = dict(training_rows_equal_p=all(np.array_equal(T_[g][f"fs{k}"].loc[:"2000-09-30"].values, T_[g][f"p{k}"].loc[:"2000-09-30"].values) for g in ids for k in KS),
            validation_test_rows_equal_the_bias_corrected_forecast=all(np.array_equal(T_[g][f"fs{k}"].loc[SUB_FROM:].values, pd.read_csv(f"{PROJ}/data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)[f"fc_bc_{k}"].loc[SUB_FROM:].values) for g in ids for k in KS),
            raw_columns_equal_the_raw_forecast=all(np.array_equal(T_[g][f"fr{k}"].loc[SUB_FROM:].values, pd.read_csv(f"{PROJ}/data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)[f"fc_raw_{k}"].loc[SUB_FROM:].values) for g in ids for k in KS),
            share_of_validation_test_slots_equal_to_p=float(np.mean([np.mean(T_[g][f"fs{k}"].loc[SUB_FROM:].values == T_[g][f"p{k}"].loc[SUB_FROM:].values) for g in ids for k in KS])),
            slot_is_the_aorc_rain_of_day_t_plus_k=all(np.allclose(T_[g][f"p{k}"].values[:-k], T_[g]["aorc_to_midnight"].values[k:], atol=1e-9) for g in ids for k in KS),
            n_training_issue_days=int((T_[ids[0]].index < d0).sum()), n_validation_test_issue_days=int((T_[ids[0]].index >= d0).sum()))
out["wqaF_sub_design"] = chk2
print(f"(2) wqaF_sub table: training rows (issue days up to 2000-09-30: {chk2['n_training_issue_days']}) hold p: {chk2['training_rows_equal_p']}; validation and test rows ({chk2['n_validation_test_issue_days']}) hold the bias-corrected forecast: {chk2['validation_test_rows_equal_the_bias_corrected_forecast']}; "
      f"share of those slots that equal the AORC rain: {chk2['share_of_validation_test_slots_equal_to_p']:.3f} (zeros on dry days); p_k(t) = aorc_to_midnight(t+k): {chk2['slot_is_the_aorc_rain_of_day_t_plus_k']}")

# (3) the control
def control_checks(arm, fcols, ccols, first):
    info = dict(same_values_per_calendar_month=True, no_day_keeps_its_year=True, rows_move_together=True, corr_forecast=[], corr_control=[], same_year_share=[])
    for g in ids:
        t = TAB[arm][g].loc[first:]
        F_ = t[list(fcols)].values; C_ = t[list(ccols)].values
        for m in range(1, 13):
            sel = t.index.month == m
            info["same_values_per_calendar_month"] &= all(np.array_equal(np.sort(F_[sel][:, j]), np.sort(C_[sel][:, j])) for j in range(3))
        # each control row is a forecast row of another day of the same calendar month and another year: find it by matching the three values
        yrs, mon = t.index.year.values, t.index.month.values
        key = {tuple(r): i for i, r in enumerate(map(tuple, F_))}                       # rows are not unique when they are all zero: keep only unique non-zero rows for the test
        nz = np.flatnonzero((F_ > 0).any(1))
        counts_ = pd.Series(list(map(tuple, F_[nz]))).value_counts()
        for i in nz[:2000]:
            r = tuple(C_[i])
            if (F_ == np.array(r)).all(1).sum() == 1 and (C_[i] > 0).any():
                j = int(np.flatnonzero((F_ == np.array(r)).all(1))[0])
                info["no_day_keeps_its_year"] &= (yrs[j] != yrs[i]) and (mon[j] == mon[i])
        for k in KS:
            info["corr_forecast"].append(np.corrcoef(t[fcols[k - 1]].values, t[f"p{k}"].values)[0, 1]); info["corr_control"].append(np.corrcoef(t[ccols[k - 1]].values, t[f"p{k}"].values)[0, 1])
        info["same_year_share"].append(float(np.mean([(F_[i] == C_[i]).all() for i in range(len(t))])))
    info = dict(same_values_per_calendar_month=bool(info["same_values_per_calendar_month"]), no_day_keeps_its_year_and_month_kept=bool(info["no_day_keeps_its_year"]), median_corr_with_the_aorc_rain_forecast=float(np.median(info["corr_forecast"])),
                median_corr_with_the_aorc_rain_control=float(np.median(info["corr_control"])), control_corr_range=[float(min(info["corr_control"])), float(max(info["corr_control"]))], share_of_days_whose_control_row_equals_their_own_forecast_row=float(np.mean(info["same_year_share"])))
    return info
c_main = control_checks("main", ("fs1", "fs2", "fs3"), ("fk1", "fk2", "fk3"), SUB_FROM)
c_sat = control_checks("sat", ("fc1", "fc2", "fc3"), ("fx1", "fx2", "fx3"), "2000-01-01")
tr_ok = all(np.array_equal(T_[g][f"fk{k}"].loc[:"2000-09-30"].values, T_[g][f"p{k}"].loc[:"2000-09-30"].values) for g in ids for k in KS)
out["control"] = dict(main=c_main, sat=c_sat, main_training_rows_hold_p=tr_ok, seeds="rng seed list (18, 1, basin number, month) in the main arm and (18, 2, basin number, month) in the satellite arm: fixed")
print(f"(3) control, E2a (validation and test rows only; training rows hold p: {tr_ok}): {c_main}")
print(f"    control, E2b (every year of the table): {c_sat}")

# (4) the history window and the leak scan on the saved wqaF_sub network
def run_from_file(f):
    z = torch.load(f, map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"]); n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z
run, z = run_from_file(f"{PROJ}/results/models/lstm_wqaF_sub_s0.pt")
base = run.predict(TAB["main"], ids, B)
alt = {g: TAB["main"][g].copy() for g in ids}
rng = np.random.default_rng(3)
for g in ids:
    for k in KS:
        v = alt[g][f"fs{k}"].values.copy(); n_tr = int((alt[g].index < d0).sum()); v[:n_tr] = rng.uniform(0, 80, n_tr); alt[g][f"fs{k}"] = v
new = run.predict(alt, ids, B)
changed_days = {}
for k in KS:
    ch = np.zeros(len(base[ids[0]]), bool)
    for g in ids:
        a, b = base[g][f"l{k}"].values, new[g][f"l{k}"].values
        ch |= ~np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
    idx = base[ids[0]].index
    changed_days[f"l{k}"] = [str(d.date()) for d in idx[ch]]
last_changed = max(pd.Timestamp(v[-1]) for v in changed_days.values() if v) if any(changed_days.values()) else None
first_changed = min(pd.Timestamp(v[0]) for v in changed_days.values() if v) if any(changed_days.values()) else None
print(f"(4) saved wqaF_sub seed 0 with the slots of ALL training rows (issue days up to 2000-09-30) replaced by random values: forecasts for target days from {first_changed.date()} to {last_changed.date()} change (first change at 2000-10-01 on the validation years; the last one at {last_changed.date()}: "
      f"{(last_changed - d0).days} days after the training years end); the test years (2005-10-01 on) change: {last_changed >= pd.Timestamp('2005-10-01')}")
out["history_leak_scan"] = dict(replaced="the three slot columns of every row with issue day up to 2000-09-30 (random 0..80 mm)", n_days_changed={l: len(v) for l, v in changed_days.items()}, first_changed_target_day=str(first_changed.date()), last_changed_target_day=str(last_changed.date()),
                                days_between_the_end_of_the_training_years_and_the_last_change=int((last_changed - d0).days), test_years_affected=bool(last_changed >= pd.Timestamp("2005-10-01")), changed_days_lead1=changed_days["l1"][:12],
                                warmup_plus_block_days=int(run.cfg.warmup + run.cfg.block))
out["seconds"] = round(time.time() - t0, 1)
jsave("f6_tables.json", out)
print(f"done in {time.time() - t0:.0f} s")
