"""ROUND 1, check 3f (re-written in round 2; the round-1 scratch copy was lost): is 'the Open-Meteo value at stamp T is the rain of the hour that ENDS at T' supported by the data?
Correlation of the hourly ERA5 (Open-Meteo stamp T) with the hourly MERRA-2 of NASA POWER at key T+lag, lag -4..+4 h; MERRA-2's key is the hour that STARTS at HH (validated by results/day_windows.json,
calibration merra2_daily_against_merra2_hourly, best shift 0). A peak at lag -1 is what 'hour ending at T' predicts; lag 0 would mean 'hour starting at T'.
The result is compared with the numbers recorded in results/review_rain_to_midnight.json (round 1, check 3f). A few seconds."""
import sys; sys.dont_write_bytecode = True
from r2_common import *
import flood as F

ids = list(F.load_data())
lags = list(range(-4, 5))
res, tot = {}, {L: [] for L in lags}
for g in ids:
    j = json.load(open(f"{PROJ}/data/hourly/era5_{g}.json"))
    te = np.array(j["hourly"]["time"], dtype="datetime64[h]"); xe = np.array(j["hourly"]["precipitation"], float)
    m = json.load(open(f"{PROJ}/data/hourly/merra_{g}.json"))["properties"]["parameter"]["PRECTOTCORR"]
    keys = np.array([f"{k[:4]}-{k[4:6]}-{k[6:8]}T{k[8:10]}" for k in m], dtype="datetime64[h]")
    xm = np.array(list(m.values()), float); xm[xm < -900] = np.nan
    t0, t1 = max(te[0], keys[0]), min(te[-1], keys[-1])
    n = int((t1 - t0) / np.timedelta64(1, "h")) + 1
    E = xe[int((t0 - te[0]) / np.timedelta64(1, "h")):][:n]; M = xm[int((t0 - keys[0]) / np.timedelta64(1, "h")):][:n]
    r = {}
    for L in lags:
        a, b = (E[:n - L] if L else E, M[L:]) if L >= 0 else (E[-L:], M[:n + L])      # corr(E[T], M[T+L])
        ok = np.isfinite(a) & np.isfinite(b)
        r[L] = float(np.corrcoef(a[ok], b[ok])[0, 1]); tot[L].append(r[L])
    res[g] = dict(short=F.SHORT[g], corr_by_lag=r, best_lag=max(r, key=r.get))
    print(f"{F.SHORT[g]:18s} best lag {res[g]['best_lag']:+d}   " + " ".join(f"{L:+d}:{r[L]:.3f}" for L in lags))
med = {L: float(np.median(tot[L])) for L in lags}
best = max(med, key=med.get)
print("median over basins:     " + " ".join(f"{L:+d}:{med[L]:.3f}" for L in lags), "-> best lag", best)
rec = next(c for c in jload("results/review_rain_to_midnight.json")["checks"] if c["id"] == "3f")["numbers"]
dev = max(abs(med[L] - rec["median_correlation_by_lag"][str(L)]) for L in lags)
same_best = {g: res[g]["best_lag"] for g in ids} == rec["best_lag_by_basin"]
print(f"against the round-1 record (check 3f): max |median correlation - recorded| = {dev:.1e}; best lag per basin equal to the record: {same_best}; median best lag equal: {best == rec['best_lag_median_h']}")
jsave("r1_stamp_convention.json", dict(median_over_basins=med, best_lag_median=best, per_basin=res, max_dev_vs_round1_record=dev, best_lags_equal_record=same_best))
