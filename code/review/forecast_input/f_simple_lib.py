"""Functions shared by f11_simple.py (adapted from the E1b review, code/review/aorc_rain_to_midnight/a_simple_lib.py): the reviewer's own wrapper around code/features.py for the E2 yardstick (ridge regression and random forest given
wqa + the forecast): the features of the input set wq (flow=True), the same rain features computed on the AORC rain to midnight (feature_table called with the series in the place of prcp), and four forecast features (the forecast
rain of the three target days and their sum, read from the slot columns fs1..fs3 of the SUBSTITUTED table of the main split: the observed rain in the training years, the bias-corrected forecast from 2000-10-01), the fitting rows
(target day in the training years 1980-10-01 .. 2000-09-30 of the main split), the refit by the stored settings, and the comparison with the stored forecasts. Needs OUT/f6_tables_csv.pkl (run f6_tables.py first)."""
import sys; sys.dont_write_bytecode = True
import pickle
from f_common import *
import flood as F
from features import feature_table
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = F.load_data(); ids = list(D)
if not os.path.exists(f"{OUT}/f6_tables_csv.pkl"):
    raise SystemExit(f"{OUT}/f6_tables_csv.pkl is missing: run f6_tables.py first (same REVIEW_OUT)")
TAB, _ids = pickle.load(open(f"{OUT}/f6_tables_csv.pkl", "rb")); assert _ids == ids
MAIN = TAB["main"]
BL = jload("results/baselines.json")["models"]
ALPHA = BL["base_linreg_wq"]["settings"]["ridge_alpha"]; RFS = BL["base_rf_wq"]["settings"]
TRAIN = SPLIT_MAIN["train"]
FC_NAMES = ("fc_day1", "fc_day2", "fc_day3", "fc_3day_total")


def feats(d, series_col=None, fc=None):
    """features.feature_table(d, flow=True) for wq; plus, with series_col, the rain features of that column (prefix 'aorc_to_midnight__'); plus, with fc (three column names), the four forecast features."""
    X = feature_table(d, True, 0)
    if series_col is not None:
        Y = feature_table(d.assign(prcp=d[series_col]), False, 0)
        X = pd.concat([X, Y[[c for c in Y.columns if c.startswith(("rain_", "wetness_"))]].add_prefix(f"{series_col}__")], axis=1)
    if fc is not None:
        for name, c in zip(FC_NAMES[:3], fc):
            X[name] = d[c].values
        X[FC_NAMES[3]] = d[list(fc)].sum(axis=1).values
    return X.where(X.notna().all(axis=1))


def fit_rows(idx, ok, k):
    tgt = idx + pd.Timedelta(days=k)
    return ok & np.asarray((tgt >= pd.Timestamp(TRAIN[0])) & (tgt <= pd.Timestamp(TRAIN[1])))


def make(kind, n_jobs=1):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    return RandomForestRegressor(RFS["n_trees"], min_samples_leaf=RFS["min_samples_leaf"], max_features=1 / 3, n_jobs=n_jobs, random_state=RFS["random_state"])


def forecast(kind, tables=None, col="aorc_to_midnight", fc=("fs1", "fs2", "fs3"), only=None, n_jobs=1, return_models=False):
    tables = tables or MAIN
    P, M = {}, {}
    for g in (only or ids):
        d = tables[g]
        X = feats(d, col, fc); ok = X.notna().all(axis=1).values
        out_k = {}
        for k in F.LEADS:
            fit = fit_rows(d.index, ok, k)
            y = d.q_mm.shift(-k).values
            m = make(kind, n_jobs).fit(X.values[fit], y[fit])
            p = np.full(len(d), np.nan); p[ok] = np.clip(m.predict(X.values[ok]), 0, None)
            out_k[k] = p; M[g, k] = (m, int(fit.sum()))
        P[g] = pd.DataFrame({"obs": d.q_mm.values, **{f"l{k}": pd.Series(out_k[k], index=d.index).shift(k).values for k in F.LEADS}}, index=d.index)
    return (P, M) if return_models else P


def compare(P, name):
    S = F.load_preds(name)
    dif, nanok = [], True
    for g in P:
        for k in F.LEADS:
            a, b = P[g][f"l{k}"].values, S[g][f"l{k}"].values
            nanok &= bool(np.array_equal(np.isnan(a), np.isnan(b)))
            dif.append(np.nanmax(np.abs(a - b)))
        assert np.array_equal(P[g].obs.values, S[g].obs.values)
    return dict(max_abs_diff=float(max(dif)), same_nan_pattern=nanok, n_basins=len(P))
