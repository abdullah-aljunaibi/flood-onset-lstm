"""E7 review, check 7: the yardstick's ridge regression refitted by the reviewer (rule block 4). The features of the input set wq (features.feature_table with the flow,
the project's shared recipe) and, for each of the three rain products in turn, the rain features of that series (rain_*, wetness_*), prefixed '<column>__'; one
StandardScaler + Ridge(alpha of results/baselines.json) per basin and lead, fitted on the issue days whose target day lies in the training years; forecasts clipped at 0
and stored at the target day. Compared with results/preds/base_linreg_wqME.npz and its meta feature list. The forest is not refitted (about 80 s with 4 jobs; its
recipe is the same code path and the script's recipe check reproduced script 16's forest to 1.4e-14). Needs OUT/my_columns.npz (m3). About 10 s."""
from m_common import *
import flood as F
from features import feature_table
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

t0 = time.time()
C = Compare()
MYC = np.load(OUT / "my_columns.npz", allow_pickle=False)
ids = [str(b) for b in MYC["basins"]]
D0 = F.load_data()
alpha = jload("results/baselines.json")["models"]["base_linreg_wq"]["settings"]["ridge_alpha"]
COLS = ("prcp_nldas", "rain_to_midnight", "aorc_to_midnight")
pred = np.full((len(ids), len(DAYS), 3), np.nan)
feat_names = None
for i, g in enumerate(ids):
    d = D0[g].copy()
    for c in COLS:
        d[c] = MYC[c][i]
    parts = [feature_table(d, True, 0)]
    for c in COLS:
        Y = feature_table(d.assign(prcp=d[c]), False, 0)
        parts.append(Y[[x for x in Y.columns if x.startswith(("rain_", "wetness_"))]].add_prefix(f"{c}__"))
    X = pd.concat(parts, axis=1)
    X = X.where(X.notna().all(axis=1))
    feat_names = list(X.columns)
    ok = X.notna().all(axis=1).values
    for j, k in enumerate(LEADS):
        target = DAYS + pd.Timedelta(days=k)
        fit = ok & np.asarray((target >= pd.Timestamp(SPLIT["train"][0])) & (target <= pd.Timestamp(SPLIT["train"][1])))
        y = d.q_mm.shift(-k).values
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X.values[fit], y[fit])
        f_issue = np.full(len(DAYS), np.nan)
        f_issue[ok] = np.clip(m.predict(X.values[ok]), 0, None)
        pred[i, k:, j] = f_issue[:-k]                         # stored at the target day
_, b, _, stored, meta = load_npz("base_linreg_wqME")
assert b == ids
C.add("features = meta features of base_linreg_wqME", meta["features"], feat_names)
same_nan = bool(np.array_equal(np.isnan(pred), np.isnan(stored)))
diff = float(np.nanmax(np.abs(pred - stored)))
C.add("refitted ridge = stored base_linreg_wqME (same days, below 1e-9 mm/day)", True, same_nan and diff < 1e-9)
out = dict(compare=C.summary(), rows=C.rows, ridge_alpha=alpha, n_features=len(feat_names), same_missing_days=same_nan, max_abs_diff=diff, seconds=round(time.time() - t0, 1))
jsave("m7_ridge.json", out)
print(json.dumps(out))
