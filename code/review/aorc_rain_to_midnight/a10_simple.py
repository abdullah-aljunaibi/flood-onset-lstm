"""AORC review, check (f): the simple models given the AORC input (base_linreg_wqa, base_rf_wqa), as in the E1 review (round 2, check d). Features rebuilt with code/features.py through the reviewer's own wrapper
(a_simple_lib.py: the AORC series enters as ten extra rain features), their causality on the real tables, the stored settings against results/baselines.json, ridge and forest refitted by the stored settings and
compared with the stored forecasts, and a perturbation test that no data after the training years enters a fit.
Usage: a10_simple.py ridge | forest | perturb     (ridge: seconds, perturb: about a minute, forest: about five minutes on one thread)"""
import sys; sys.dont_write_bytecode = True
import time
from a_simple_lib import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "ridge"
R = jload("results/aorc_rain_to_midnight.json")
out = {}
if MODE == "ridge":
    X0 = feats(D[ids[0]]); X1 = feats(D[ids[0]], mine[ids[0]])
    out["features"] = dict(n_wq=X0.shape[1], n_wqa=X1.shape[1], extra=[c for c in X1.columns if c.startswith("aorc_to_midnight__")], columns_of_wq_unchanged=bool(list(X1.columns[:X0.shape[1]]) == list(X0.columns)),
                           rain_features_of_prcp=[c for c in X0.columns if c.startswith(("rain_", "wetness_"))])
    print("features of wq:", X0.shape[1], "| of wqa:", X1.shape[1], "| the", len(out["features"]["extra"]), "extra:", out["features"]["extra"])
    same = [c[len("aorc_to_midnight__"):] for c in out["features"]["extra"]] == out["features"]["rain_features_of_prcp"]
    print("the rain features that prcp has:", out["features"]["rain_features_of_prcp"], "-> same names (with the prefix) for the new series:", same)
    g = ids[3]; s = pd.Series(mine[g], index=D[g].index); X = feats(D[g], s)
    hand = dict(rain_d1=s.shift(1), rain_sum7=s.rolling(7).sum(), rain_sum90=s.rolling(90).sum())
    ok_hand = all(np.allclose(X[f"aorc_to_midnight__{k}"].dropna().values, v.loc[X[f"aorc_to_midnight__{k}"].dropna().index].values) for k, v in hand.items())
    a = 0.85; w = np.zeros(len(s)); acc = 0.0
    for i, v in enumerate(s.values):
        acc = acc * a + v; w[i] = acc
    ok_hand &= np.allclose(X["aorc_to_midnight__wetness_085"].dropna().values, pd.Series(w, index=s.index).loc[X["aorc_to_midnight__wetness_085"].dropna().index].values)
    out["extra_features_equal_hand_computation"] = bool(ok_hand)
    print("extra features equal a hand computation (rain_d1, rain_sum7, rain_sum90, wetness_085 on the AORC series):", ok_hand)
    cut = 5000; ok_c = True
    for g in ids:
        d = D[g]; s = pd.Series(mine[g], index=d.index)
        bad = d.copy(); bad.iloc[cut + 1:] = bad.iloc[cut + 1:] * 7 + 100
        sb = s.copy(); sb.iloc[cut + 1:] = sb.iloc[cut + 1:] * 7 + 100
        A, A2 = feats(d, s), feats(bad, sb)
        ok_c &= bool(A.columns.equals(A2.columns) and A.iloc[:cut + 1].equals(A2.iloc[:cut + 1]) and not np.allclose(A.iloc[cut + 1].values, A2.iloc[cut + 1].values))
    out["features_causal_on_real_tables"] = ok_c
    print(f"causality: every column and the AORC series changed after day {cut} of the record (all 12 basins): rows up to the cut unchanged, next row changes: {ok_c}")
    for n, ref in (("base_linreg_wqa", "base_linreg_wq"), ("base_rf_wqa", "base_rf_wq")):
        meta = F.preds_meta(n); refset = B[ref]["settings"]
        same = all(meta["settings"][k] == refset[k] for k in meta["settings"])
        out[f"settings_{n}"] = dict(stored=meta["settings"], baselines_json_for_wq={k: refset[k] for k in meta["settings"]}, equal=same, meta_script=meta["script"], quick_mode=meta["quick_mode"], n_features=len(meta["features"]))
        print(f"settings stored with {n}: {meta['settings']}  equal to baselines.json for wq: {same}; script {meta['script']}; quick_mode {meta['quick_mode']}; features {len(meta['features'])}")
    out["feature_names_equal_stored"] = list(X1.columns) == F.preds_meta("base_linreg_wqa")["features"]
    print("my feature names equal the list stored with base_linreg_wqa:", out["feature_names_equal_stored"])
    t0 = time.time()
    P = forecast("ridge", None); out["ridge_wq_vs_stored_base_linreg_wq"] = compare(P, "base_linreg_wq")
    P = forecast("ridge", lambda g: mine[g]); out["ridge_wqa_vs_stored_base_linreg_wqa"] = compare(P, "base_linreg_wqa")
    print(f"ridge refit (12 basins x 3 leads, {time.time()-t0:.0f} s): wq against the stored base_linreg_wq: {out['ridge_wq_vs_stored_base_linreg_wq']}; wqa against the stored base_linreg_wqa: {out['ridge_wqa_vs_stored_base_linreg_wqa']}")
    d = D[ids[0]]; X, ok = design(ids[0], mine[ids[0]])
    out["fit_rows_basin0"] = {f"lead{k}": dict(n=int(fit_rows(d.index, ok, k).sum()), first_target=str((d.index[fit_rows(d.index, ok, k)][0] + pd.Timedelta(days=k)).date()), last_target=str((d.index[fit_rows(d.index, ok, k)][-1] + pd.Timedelta(days=k)).date())) for k in F.LEADS}
    print("rows used for fitting (basin 0):", out["fit_rows_basin0"])
elif MODE == "forest":
    t0 = time.time()
    P = forecast("rf", lambda g: mine[g]); out["forest_wqa_vs_stored_base_rf_wqa"] = compare(P, "base_rf_wqa"); out["seconds_wqa"] = round(time.time() - t0, 1)
    print(f"forest refit wqa (n_jobs=1, {out['seconds_wqa']} s): against the stored base_rf_wqa: {out['forest_wqa_vs_stored_base_rf_wqa']}", flush=True)
    jsave(f"a10_{MODE}.json", out)
else:
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
            same_coef = all(np.array_equal(Ma[g, k][0][-1].coef_, Mb[g, k][0][-1].coef_) and Ma[g, k][0][-1].intercept_ == Mb[g, k][0][-1].intercept_ and np.array_equal(Ma[g, k][0][0].mean_, Mb[g, k][0][0].mean_) for k in F.LEADS) if kind == "ridge" else None
            moved_after = not np.array_equal(Pa[g]["l1"].values[~tr_rows], Pb[g]["l1"].values[~tr_rows], equal_nan=True)
            res[f"{kind}:{g}"] = dict(n_fit_rows_equal=same_rows, in_sample_forecasts_equal=same_fc, coefficients_equal=same_coef, forecasts_after_the_training_years_do_change=moved_after)
            print(f"  {kind:5s} {F.SHORT[g]:18s} fit rows equal {same_rows}; coefficients/scalers equal {same_coef}; forecasts for target days up to 2000-09-27 equal {same_fc}; forecasts later do change (the test has power): {moved_after}", flush=True)
    out["perturbation"] = res
    out["all_ok"] = all(v["n_fit_rows_equal"] and v["in_sample_forecasts_equal"] and v["forecasts_after_the_training_years_do_change"] and v["coefficients_equal"] in (True, None) for v in res.values())
    print("no data after the training years enters a fit:", out["all_ok"])
jsave(f"a10_{MODE}.json", out)
