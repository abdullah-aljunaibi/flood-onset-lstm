"""E2 review, check (b) continued: the hours of the two products against each other, below the day. The forecast's 3-hour windows of every run (basin means from the reviewer's own weights, f4_basins.py logic) against
the AORC rain of the same absolute 3-hour window moved by s hours (s = -6 .. +6; an AORC value stamped T is the rain of the hour ENDING at T: the E1b review confirmed that against Stage IV), correlation pooled over all runs
of 2000-2014 and the first 26 windows, per basin and median over the basins: if the run times or the window ends were off by an hour or more, the best shift would not be 0. (The model's own timing error widens the peak and
may move it by a fraction of an hour; the curve is read between whole hours.) Also by lead (windows of the forecast hours 0-24, 24-48, 48-72). About 40 seconds."""
import sys; sys.dont_write_bytecode = True
import glob, time
import shapefile, shapely
from shapely.geometry import shape
from f_common import *
from f_aorc import *

t0 = time.time()
G = jload("results/gefs_basins.json"); ids = list(G["basins"])
f4 = jread("f4_basins.json")
LAT = 25.0 + 0.25 * np.arange(53); LON = -91.0 + 0.25 * np.arange(65)
W = {g: [(int(k.split(",")[0]), int(k.split(",")[1]), w) for k, w in f4["weights"][g]["weights_mine"].items()] for g in ids}
cells_all = sorted({(i, j) for g in ids for i, j, _ in W[g]}); col = {c: n for n, c in enumerate(cells_all)}
il, jl = np.array([c[0] for c in cells_all]), np.array([c[1] for c in cells_all])
inits, rain = [], []
for f in sorted(glob.glob(f"{PROJ}/data/gefs/c00/apcp_20*.npz")):
    z = np.load(f); inits.append(pd.to_datetime(z["init_date"].astype(str), format="%Y%m%d")); rain.append(z["apcp"][:, :, il, jl].astype(np.float64) / 10.0)
inits = pd.DatetimeIndex(np.concatenate([i.values for i in inits])); rain = np.concatenate(rain)
run_t = inits.values.astype("datetime64[m]")                                  # 00 UTC of each run
res = {}
for g in ids:
    w = np.einsum("rwc,c->rw", rain[:, :, [col[(i, j)] for i, j, _ in W[g]]], np.array([x for _, _, x in W[g]]))          # [run, 26] basin-mean mm per window
    t, x = read_aorc(g)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = {}
    for s in range(-6, 7):
        a = run_t[:, None] + (np.arange(26)[None, :] * 3 + s) * np.timedelta64(60, "m")                # window start (UTC) moved by s hours
        ia = ((a - t[0]) / np.timedelta64(60, "m")).astype(int)                                          # index of the stamp = window start; the hours of the window have stamps start+1 .. start+3
        ok = (ia >= 0) & (ia + 4 <= len(x))
        obs = cs[np.where(ok, ia + 4, 0)] - cs[np.where(ok, ia + 1, 0)]
        out[s] = float(np.corrcoef(w[ok], obs[ok])[0, 1])
        if s in (0,):
            by_lead = {}
            for lo_, hi_, name in ((0, 8, "forecast hours 0-24"), (8, 16, "24-48"), (16, 24, "48-72")):
                by_lead[name] = None
    res[g] = out
med = {s: float(np.median([res[g][s] for g in ids])) for s in range(-6, 7)}
best = {g: max(res[g], key=res[g].get) for g in ids}
print("pooled correlation of the forecast's 3-hour windows with the AORC rain of the same window moved by s hours (median over the 12 basins):")
print("   " + "  ".join(f"{s:+d}h {v:.3f}" for s, v in med.items()))
print(f"best shift of the median: {max(med, key=med.get):+d} h; best shift by basin: {dict(sorted(best.items()))}")
jsave("f5b_subdaily.json", dict(median_correlation_by_shift={str(s): v for s, v in med.items()}, best_shift_median=int(max(med, key=med.get)), best_shift_by_basin=best, per_basin={g: {str(s): v for s, v in r.items()} for g, r in res.items()}, json_median=G["clock"]["subdaily_alignment"]["median_over_basins"], seconds=round(time.time() - t0, 1)))
print("builder's median curve:", {k: round(v, 3) for k, v in G["clock"]["subdaily_alignment"]["median_over_basins"].items()})
