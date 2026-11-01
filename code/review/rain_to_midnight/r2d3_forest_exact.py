"""ROUND 2, check (d) addendum: the forest refit (ONE job) with the daily rain re-summed by pandas groupby, as the script sums it, against the stored base_rf_wqh. If the residual of 7e-15 stays, it is not the data (it is the order in which parallel threads add up the trees' predictions in the four-job fit that made the stored file). About three minutes on one thread."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from r2_simple_lib import *
import r2_clock as C
st = pd.read_csv(f"{PROJ}/data/hourly/usgs_sites.rdb", sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
ZONE = {g: {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago"}[st.tz_cd[g]] for g in ids}
pd_series, dmax = {}, 0.0
for g in ids:
    t, x, _ = C.read_era5(g); off = C.offset_rules(t, ZONE[g])
    day = C.window_day(t, off, "wall", 24) - C.D0
    keep = (day >= 0) & (day < C.N_DAYS)
    s = pd.Series(x[keep]).groupby(day[keep]).sum().reindex(range(C.N_DAYS)).values
    dmax = max(dmax, float(np.abs(s - mine[g]).max())); pd_series[g] = s
print("max |pandas groupby sum - numpy bincount sum| of the daily rain over all basins and days:", dmax)
P = forecast("rf", lambda g: pd_series[g])
res = compare(P, "base_rf_wqh")
print("forest refit (n_jobs=1) with the pandas-summed series against the stored base_rf_wqh:", res)
jsave("r2d3_forest_exact.json", dict(max_series_difference=dmax, forest_vs_stored=res))
