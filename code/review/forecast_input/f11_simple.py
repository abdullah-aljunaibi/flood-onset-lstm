"""E2 review, check (g): the yardstick. Ridge regression and the random forest given wqa + the forecast (code/18_forecast_input.py: features of wq + the rain features of the AORC rain to midnight + the forecast rain of the three
target days and their sum, substituted table of the main split, one model per basin and lead, fitted on the training years only), rebuilt with the reviewer's own wrapper around code/features.py (f_simple_lib.py) and refitted.
Usage: f11_simple.py ridge | forest | perturb     (ridge: seconds; perturb: about 2 minutes; forest: about 5 minutes on one thread)
 ridge:   the features (names and number against the stored list), the settings against results/baselines.json, causality of the features on the real tables (every column changed after a cut day: the rows up to the cut do
          not change, the next one does), the substitution applied the same way (training rows: observed rain in the slots; validation and test rows: the bias-corrected forecast), the rows used for fitting (target days
          up to 2000-09-30), the ridge refit against the stored base_linreg_wqaF (and the recipe: the wq features alone against the stored base_linreg_wq);
 forest:  the refit against the stored base_rf_wqaF (settings 150 trees, leaf 10, one third of the features, random_state 0 from baselines.json);
 perturb: nothing after the training years enters a fit: every data column and the flow after 2000-09-30 replaced (the slots of the last three training days keep their values: they hold the observed rain of
          2000-10-01 .. 03 by design), the fit rows, the coefficients and the forecasts for target days up to 2000-09-27 do not change, later forecasts do (ridge for all basins, the forest for two)."""
import sys; sys.dont_write_bytecode = True
import time
from f_simple_lib import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "ridge"
R = jload("results/forecast_input.json")
out = {}
if MODE == "ridge":
    meta = F.preds_meta("base_linreg_wqaF")
    X = feats(MAIN[ids[0]], "aorc_to_midnight", ("fs1", "fs2", "fs3"))
    out["features"] = dict(n=int(X.shape[1]), names_equal_stored=list(X.columns) == meta["features"], n_wq=int(feature_table(MAIN[ids[0]], True, 0).shape[1]), extra=[c for c in X.columns if c.startswith("aorc_to_midnight__")], forecast=[c for c in X.columns if c.startswith("fc_")])
    print(f"features: {out['features']['n']} (wq {out['features']['n_wq']}, AORC rain features {len(out['features']['extra'])}, forecast features {out['features']['forecast']}); names and order equal the stored list: {out['features']['names_equal_stored']}")
    for n in ("base_linreg_wqaF", "base_rf_wqaF"):
        m = F.preds_meta(n); ref = BL["base_linreg_wq" if "linreg" in n else "base_rf_wq"]["settings"]
        out[f"settings_{n}"] = dict(stored=m["settings"], baselines_json=ref, equal=all(m["settings"][k] == ref[k] for k in m["settings"]), script=m["script"], quick_mode=m["quick_mode"])
        print(f"settings of {n}: {m['settings']} equal to results/baselines.json (wq): {out[f'settings_{n}']['equal']}; script {m['script']}; quick_mode {m['quick_mode']}")
    # causality on the real tables
    cut, ok_c = 5000, True
    for g in ids:
        d = MAIN[g]; bad = d.copy(); bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
        A, A2 = feats(d, "aorc_to_midnight", ("fs1", "fs2", "fs3")), feats(bad, "aorc_to_midnight", ("fs1", "fs2", "fs3"))
        ok_c &= bool(A.columns.equals(A2.columns) and A.iloc[:cut + 1].equals(A2.iloc[:cut + 1]) and not np.allclose(A.iloc[cut + 1].values, A2.iloc[cut + 1].values))
    out["features_causal_on_real_tables"] = ok_c
    print(f"causality: every column of the table (the forecast slots too) changed after day {cut}: rows up to the cut unchanged, the next row changes, all 12 basins: {ok_c}")
    # the substitution as the simple models see it
    g = ids[3]; d = MAIN[g]; X = feats(d, "aorc_to_midnight", ("fs1", "fs2", "fs3")); ok = X.notna().all(axis=1).values
    tr = d.index < pd.Timestamp("2000-10-01")
    sub = dict(training_rows_forecast_features_equal_observed_rain=bool(np.array_equal(X.loc[tr & ok, "fc_day1"].values, d.loc[tr & ok, "p1"].values)), validation_test_rows_equal_the_forecast=bool(np.array_equal(X.loc[~tr & ok, "fc_day1"].values, pd.read_csv(f"{PROJ}/data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True).loc[X.index[~tr & ok], "fc_bc_1"].values)),
               n_fit_rows_lead1=int(fit_rows(d.index, ok, 1).sum()), last_target_day_of_the_fit_rows_lead1=str((d.index[fit_rows(d.index, ok, 1)][-1] + pd.Timedelta(days=1)).date()), first_target_day_of_the_fit_rows_lead1=str((d.index[fit_rows(d.index, ok, 1)][0] + pd.Timedelta(days=1)).date()),
               slots_of_the_last_training_issue_day_reach_into=str((pd.Timestamp("2000-09-30") + pd.Timedelta(days=3)).date()))
    out["substitution"] = sub
    print("substitution as the simple models see it (Catawba):", sub)
    t0 = time.time()
    P = forecast("ridge"); out["ridge_vs_stored_base_linreg_wqaF"] = compare(P, "base_linreg_wqaF")
    P0 = forecast("ridge", col=None, fc=None); out["recipe_ridge_wq_vs_stored_base_linreg_wq"] = compare(P0, "base_linreg_wq")
    print(f"ridge refit ({time.time() - t0:.0f} s): against the stored base_linreg_wqaF: {out['ridge_vs_stored_base_linreg_wqaF']}; the recipe on the input set wq alone against the stored base_linreg_wq: {out['recipe_ridge_wq_vs_stored_base_linreg_wq']}")
    # which rows are used by the fit: they are the rows with all features and the target day in 1980-10-01 .. 2000-09-30
    out["fit_rows_lead"] = {f"lead{k}": dict(n=int(fit_rows(d.index, ok, k).sum()), first_target=str((d.index[fit_rows(d.index, ok, k)][0] + pd.Timedelta(days=k)).date()), last_target=str((d.index[fit_rows(d.index, ok, k)][-1] + pd.Timedelta(days=k)).date())) for k in F.LEADS}
elif MODE == "forest":
    t0 = time.time()
    P = forecast("rf"); out["forest_vs_stored_base_rf_wqaF"] = compare(P, "base_rf_wqaF"); out["seconds"] = round(time.time() - t0, 1)
    print(f"forest refit (n_jobs=1, {out['seconds']} s): against the stored base_rf_wqaF: {out['forest_vs_stored_base_rf_wqaF']}", flush=True)
else:
    cutdate = pd.Timestamp("2000-09-30"); res = {}
    for kind, only in (("ridge", ids), ("rf", [ids[3], ids[8]])):
        for g in only:
            d = MAIN[g]; bad = d.copy(); m_ = np.asarray(d.index > cutdate)
            data_cols = [c for c in d.columns if c not in ("p1", "p2", "p3", "fs1", "fs2", "fs3", "fk1", "fk2", "fk3", "fr1", "fr2", "fr3")]
            bad.loc[m_, data_cols] = bad.loc[m_, data_cols] * 7 + 100                     # every data column after the cut (the slot columns keep their values for the issue days up to the cut: the slots of the last training days hold the observed rain of the three days after the cut by design)
            bad.loc[m_, ["fs1", "fs2", "fs3"]] = bad.loc[m_, ["fs1", "fs2", "fs3"]] * 7 + 100        # the forecast slots of the validation and test rows
            Pa, Ma = forecast(kind, {g: d}, only=[g], return_models=True); Pb, Mb = forecast(kind, {g: bad}, only=[g], return_models=True)
            tr_rows = np.asarray(d.index <= cutdate - pd.Timedelta(days=3))
            same_rows = all(Ma[g, k][1] == Mb[g, k][1] for k in F.LEADS)
            same_fc = all(np.array_equal(Pa[g][f"l{k}"].values[tr_rows], Pb[g][f"l{k}"].values[tr_rows], equal_nan=True) for k in F.LEADS)
            same_coef = (all(np.array_equal(Ma[g, k][0][-1].coef_, Mb[g, k][0][-1].coef_) and Ma[g, k][0][-1].intercept_ == Mb[g, k][0][-1].intercept_ and np.array_equal(Ma[g, k][0][0].mean_, Mb[g, k][0][0].mean_) for k in F.LEADS) if kind == "ridge" else None)
            moved_after = not np.array_equal(Pa[g]["l1"].values[~tr_rows], Pb[g]["l1"].values[~tr_rows], equal_nan=True)
            res[f"{kind}:{g}"] = dict(n_fit_rows_equal=same_rows, forecasts_up_to_the_cut_equal=same_fc, coefficients_equal=same_coef, forecasts_after_the_cut_do_change=moved_after)
            print(f"  {kind:5s} {F.SHORT[g]:18s} fit rows equal {same_rows}; coefficients/scalers equal {same_coef}; forecasts for target days up to 2000-09-27 equal {same_fc}; later forecasts do change (the test has power): {moved_after}", flush=True)
    out["perturbation"] = res
    out["all_ok"] = all(v["n_fit_rows_equal"] and v["forecasts_up_to_the_cut_equal"] and v["forecasts_after_the_cut_do_change"] and v["coefficients_equal"] in (True, None) for v in res.values())
    print("no data after the training years enters a fit:", out["all_ok"])
jsave(f"f11_{MODE}.json", out)
