"""Functions shared by r2d_simple.py, r2d2_ridge_exact.py and r2d3_forest_exact.py: the reviewer's own wrapper around code/features.py (the new series enters as ten extra rain features), the fitting rows,
the refit of the simple models by the stored settings, and the comparison with the stored forecasts. Needs OUT/my_series_r2.npz (run r2c_clocks.py first)."""
import sys; sys.dont_write_bytecode = True
import sys, time
from r2_common import *
import flood as F
from features import feature_table
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = F.load_data(); ids = list(D)
if not os.path.exists(f"{OUT}/my_series_r2.npz"):
    raise SystemExit(f"{OUT}/my_series_r2.npz is missing: run r2c_clocks.py first (same REVIEW_OUT)")
Z = np.load(f"{OUT}/my_series_r2.npz"); assert list(Z["basins"]) == ids
mine = {g: Z["midnight"][i] for i, g in enumerate(ids)}
B = jload("results/baselines.json")["models"]
ALPHA = B["base_linreg_wq"]["settings"]["ridge_alpha"]; RFS = B["base_rf_wq"]["settings"]
TRAIN = ("1980-10-01", "2000-09-30")

def feats(d, series=None):
    """The features of the input set wq (flow=True) and, when series is given, the same rain features computed on that series (feature_table called with the series in the place of prcp)."""
    X = feature_table(d, True, 0)
    if series is None:
        return X
    Y = feature_table(d.assign(prcp=series), False, 0)
    extra = Y[[c for c in Y.columns if c.startswith(("rain_", "wetness_"))]].add_prefix("rain_to_midnight__")
    X2 = pd.concat([X, extra], axis=1)
    return X2.where(X2.notna().all(axis=1))

def design(g, series, dd=None):
    d = (D[g] if dd is None else dd)
    X = feats(d, series)
    ok = X.notna().all(axis=1).values
    return X, ok

def fit_rows(idx, ok, k):
    tgt = idx + pd.Timedelta(days=k)                                                # the target day of each issue day
    return ok & np.asarray((tgt >= pd.Timestamp(TRAIN[0])) & (tgt <= pd.Timestamp(TRAIN[1])))

def make(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    return RandomForestRegressor(RFS["n_trees"], min_samples_leaf=RFS["min_samples_leaf"], max_features=1 / 3, n_jobs=1, random_state=RFS["random_state"])

def forecast(kind, series_of, dd=None, only=None, return_models=False):
    P, M = {}, {}
    for g in (only or ids):
        d = D[g] if dd is None else dd[g]
        X, ok = design(g, series_of(g) if series_of else None, d)
        out_k = {}
        for k in F.LEADS:
            fit = fit_rows(d.index, ok, k)
            y = d.q_mm.shift(-k).values
            m = make(kind).fit(X.values[fit], y[fit])
            p = np.full(len(d), np.nan); p[ok] = np.clip(m.predict(X.values[ok]), 0, None)
            out_k[k] = p; M[g, k] = (m, int(fit.sum()))
        frame = pd.DataFrame({"obs": d.q_mm.values, **{f"l{k}": pd.Series(out_k[k], index=d.index).shift(k).values for k in F.LEADS}}, index=d.index)       # forecast for target day T = the one issued on T-k
        P[g] = frame
    return (P, M) if return_models else P

def compare(P, name):
    S = F.load_preds(name)
    dif = []; nanok = True
    for g in P:
        for k in F.LEADS:
            a, b = P[g][f"l{k}"].values, S[g][f"l{k}"].values
            nanok &= bool(np.array_equal(np.isnan(a), np.isnan(b)))
            dif.append(np.nanmax(np.abs(a - b)))
        assert np.array_equal(P[g].obs.values, S[g].obs.values)
    return dict(max_abs_diff=float(max(dif)), same_nan_pattern=nanok, n_basins=len(P))

