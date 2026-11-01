"""ROUND 2, check (d): the simple models on the new input. Features rebuilt with code/features.py through the reviewer's own wrapper (r2_simple_lib.py; the new series enters as ten extra rain features),
the causality of the features on the real tables, ridge and forest refitted by the stored settings and compared with base_linreg_wqh / base_rf_wqh (and, to validate this wrapper, with the
stored base_linreg_wq / base_rf_wq of 02_baselines.py), and a perturbation test that no data after the training years enters a fit.
Usage: r2d_simple.py ridge | forest | perturb     (ridge: seconds, perturb: about a minute, forest: about five minutes on one thread)"""
import sys; sys.dont_write_bytecode = True
import sys, time
from r2_simple_lib import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "ridge"        # "ridge", "forest" or "perturb"
out_file = f"{OUT}/r2d_{MODE}.json"
out = {}

if MODE == "ridge":
    # ---- how the new column enters ----
    X0 = feats(D[ids[0]]); X1 = feats(D[ids[0]], mine[ids[0]])
    out["features"] = dict(n_wq=X0.shape[1], n_wqh=X1.shape[1], extra=[c for c in X1.columns if c.startswith("rain_to_midnight__")], columns_of_wq_unchanged=bool(list(X1.columns[:X0.shape[1]]) == list(X0.columns)),
                           rain_features_of_prcp=[c for c in X0.columns if c.startswith(("rain_", "wetness_"))])
    print("features of wq:", X0.shape[1], "| of wqh:", X1.shape[1], "| the", len(out["features"]["extra"]), "extra:", out["features"]["extra"])
    print("the rain features that prcp has:", out["features"]["rain_features_of_prcp"], "-> same names (with the prefix) for the new series:", [c[len("rain_to_midnight__"):] for c in out["features"]["extra"]] == out["features"]["rain_features_of_prcp"])
    # the extra features are exactly the same functions of the new series: check two by hand
    g = ids[3]; s = pd.Series(mine[g], index=D[g].index); X = feats(D[g], s)
    hand = dict(rain_d1=s.shift(1), rain_sum7=s.rolling(7).sum(), rain_sum90=s.rolling(90).sum())
    ok_hand = all(np.allclose(X[f"rain_to_midnight__{k}"].dropna().values, v.loc[X[f"rain_to_midnight__{k}"].dropna().index].values) for k, v in hand.items())
    a = 0.85; w = np.zeros(len(s)); acc = 0.0
    for i, v in enumerate(s.values):
        acc = acc * a + v; w[i] = acc
    ok_hand &= np.allclose(X["rain_to_midnight__wetness_085"].dropna().values, pd.Series(w, index=s.index).loc[X["rain_to_midnight__wetness_085"].dropna().index].values)
    out["extra_features_equal_hand_computation"] = bool(ok_hand)
    print("extra features equal a hand computation (rain_d1, rain_sum7, rain_sum90, wetness_085 on the new series):", ok_hand)
    # ---- causality on the real tables: change EVERYTHING after a cut day (every column and the new series) ----
    cut = 5000; ok_c = True
    for g in ids:
        d = D[g]; s = pd.Series(mine[g], index=d.index)
        bad = d.copy(); bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
        sb = s.copy(); sb.iloc[cut + 1:] = sb.iloc[cut + 1:] * 7 + 100
        A, A2 = feats(d, s), feats(bad, sb)
        ok_c &= bool(A.columns.equals(A2.columns) and A.iloc[:cut + 1].equals(A2.iloc[:cut + 1]) and not np.allclose(A.iloc[cut + 1].values, A2.iloc[cut + 1].values))
    out["features_causal_on_real_tables"] = ok_c
    print(f"causality: every column and the new series changed after day {cut} of the record (all 12 basins): rows up to the cut unchanged, next row changes: {ok_c}")
    # ---- settings ----
    for n, ref in (("base_linreg_wqh", "base_linreg_wq"), ("base_rf_wqh", "base_rf_wq")):
        meta = F.preds_meta(n); refset = B[ref]["settings"]
        same = all(meta["settings"][k] == refset[k] for k in meta["settings"]) 
        out[f"settings_{n}"] = dict(stored=meta["settings"], baselines_json_for_wq={k: refset[k] for k in meta["settings"]}, equal=same, meta_script=meta["script"], quick_mode=meta["quick_mode"], n_features=len(meta["features"]))
        print(f"settings stored with {n}: {meta['settings']}  equal to baselines.json for wq: {same}; quick_mode {meta['quick_mode']}; features {len(meta['features'])}")
    out["feature_names_equal_stored"] = list(X1.columns) == F.preds_meta("base_linreg_wqh")["features"]
    print("my feature names equal the list stored with base_linreg_wqh:", out["feature_names_equal_stored"])
    # ---- ridge refits: wq (validates the wrapper against 02_baselines.py's stored forecasts) and wqh ----
    t0 = time.time()
    P = forecast("ridge", None); out["ridge_wq_vs_stored_base_linreg_wq"] = compare(P, "base_linreg_wq")
    P = forecast("ridge", lambda g: mine[g]); out["ridge_wqh_vs_stored_base_linreg_wqh"] = compare(P, "base_linreg_wqh")
    print(f"ridge refit (12 basins x 3 leads, {time.time()-t0:.0f} s): wq against the stored base_linreg_wq: {out['ridge_wq_vs_stored_base_linreg_wq']}; wqh against the stored base_linreg_wqh: {out['ridge_wqh_vs_stored_base_linreg_wqh']}")
    # fit rows
    d = D[ids[0]]; X, ok = design(ids[0], mine[ids[0]])
    out["fit_rows_basin0"] = {f"lead{k}": dict(n=int(fit_rows(d.index, ok, k).sum()), first_target=str((d.index[fit_rows(d.index, ok, k)][0] + pd.Timedelta(days=k)).date()), last_target=str((d.index[fit_rows(d.index, ok, k)][-1] + pd.Timedelta(days=k)).date())) for k in F.LEADS}
    print("rows used for fitting (basin 0):", out["fit_rows_basin0"])
elif MODE == "forest":
    t0 = time.time()
    P = forecast("rf", lambda g: mine[g]); out["forest_wqh_vs_stored_base_rf_wqh"] = compare(P, "base_rf_wqh"); out["seconds_wqh"] = round(time.time() - t0, 1)
    print(f"forest refit wqh (n_jobs=1, {out['seconds_wqh']} s): against the stored base_rf_wqh: {out['forest_wqh_vs_stored_base_rf_wqh']}", flush=True)
    jsave(f"r2d_{MODE}.json", out)
    t1 = time.time()
    P = forecast("rf", None); out["forest_wq_vs_stored_base_rf_wq"] = compare(P, "base_rf_wq"); out["seconds_wq"] = round(time.time() - t1, 1)
    print(f"forest refit wq (n_jobs=1, {out['seconds_wq']} s): against the stored base_rf_wq (02_baselines.py, n_jobs=4): {out['forest_wq_vs_stored_base_rf_wq']}", flush=True)
else:
    # no data after the training years may enter a fit: replace EVERYTHING after 2000-09-30 (all columns, flow and the new series) by other numbers and refit: models and in-sample forecasts must not change
    cutdate = pd.Timestamp(TRAIN[1]); res = {}
    for kind, only in (("ridge", ids), ("rf", [ids[3], ids[8]])):
        for g in only:
            d = D[g]; s = pd.Series(mine[g], index=d.index)
            bad = d.copy(); m_ = np.asarray(d.index > cutdate); bad.loc[m_] = bad.loc[m_] * 7 + 100
            sb = s.copy(); sb.loc[m_] = sb.loc[m_] * 7 + 100
            (Pa, Ma) = forecast(kind, lambda gg: s, None, [g], True)
            Db = {g: bad}
            Pb, Mb = forecast(kind, lambda gg: sb, Db, [g], True)
            tr_rows = np.asarray(d.index <= cutdate - pd.Timedelta(days=3))
            same_rows = all(Ma[g, k][1] == Mb[g, k][1] for k in F.LEADS)
            same_fc = all(np.array_equal(Pa[g][f"l{k}"].values[tr_rows], Pb[g][f"l{k}"].values[tr_rows], equal_nan=True) for k in F.LEADS)
            if kind == "ridge":
                same_coef = all(np.array_equal(Ma[g, k][0][-1].coef_, Mb[g, k][0][-1].coef_) and Ma[g, k][0][-1].intercept_ == Mb[g, k][0][-1].intercept_ and np.array_equal(Ma[g, k][0][0].mean_, Mb[g, k][0][0].mean_) for k in F.LEADS)
            else:
                same_coef = None
            moved_after = not np.array_equal(Pa[g]["l1"].values[~tr_rows], Pb[g]["l1"].values[~tr_rows], equal_nan=True)
            res[f"{kind}:{g}"] = dict(n_fit_rows_equal=same_rows, in_sample_forecasts_equal=same_fc, coefficients_equal=same_coef, forecasts_after_the_training_years_do_change=moved_after)
            print(f"  {kind:5s} {F.SHORT[g]:18s} fit rows equal {same_rows}; coefficients/scalers equal {same_coef}; forecasts for target days up to 2000-09-27 equal {same_fc}; forecasts later do change (the test has power): {moved_after}", flush=True)
    out["perturbation"] = res
    out["all_ok"] = all(v["n_fit_rows_equal"] and v["in_sample_forecasts_equal"] and v["forecasts_after_the_training_years_do_change"] and v["coefficients_equal"] in (True, None) for v in res.values())
    print("no data after the training years enters a fit:", out["all_ok"])
jsave(f"r2d_{MODE}.json", out)
