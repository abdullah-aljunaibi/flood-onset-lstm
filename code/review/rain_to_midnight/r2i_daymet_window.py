"""ROUND 2, extra: how well does the rain_to_09 window match the Daymet rain table? Own estimate of the end of the Daymet day from the hourly ERA5 (wall-clock windows ending at hour h of day d, h = 3..24),
and the prcp_nldas meta / prediction-file metadata of the new sets."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from r2_common import *
import flood as F
import r2_clock as C
D = F.load_data(); ids = list(D)
st = pd.read_csv(f"{PROJ}/data/hourly/usgs_sites.rdb", sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
ZONE = {g: {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago"}[st.tz_cd[g]] for g in ids}
hours = list(range(3, 25))
best, med_curve = {}, []
for g in ids:
    t, x, _ = C.read_era5(g); off = C.offset_rules(t, ZONE[g])
    cur = [float(np.corrcoef(D[g].prcp.values, C.window_sum(t, x, off, "wall", h)[0])[0, 1]) for h in hours]
    best[g] = hours[int(np.argmax(cur))]; med_curve.append(cur)
mc = np.median(np.array(med_curve), axis=0)
print("best wall-clock end hour of a 24-h ERA5 window for the Daymet rain, per basin:", {F.SHORT[g]: best[g] for g in ids})
print("median over basins of the correlation by end hour:", {h: round(float(v), 3) for h, v in zip(hours, mc) if h in (6, 8, 9, 10, 12, 14, 18, 24)}, "-> peak at", hours[int(np.argmax(mc))])
c09 = float(np.median(np.array(med_curve)[:, hours.index(9)])); cmax = float(mc.max())
print(f"median correlation at 09:00 {c09:.3f} vs at the peak {cmax:.3f}: the 09:00 window loses {cmax-c09:.3f} of correlation against the best window")
# metadata of the new prediction files
conf = jload("results/lstm_all.json")["config"]
for n in ("wqh_to09", "wqh_utc", "wqnh"):
    for s in ("", "_s0", "_s1", "_s2"):
        m = F.preds_meta(f"lstm_{n}{s}")
        assert m["script"] == "15_rain_to_midnight.py" and m["config"] == conf and m["inputs"] == n and m["model"] == "lstm", (n, s, m)
print("metadata of the 12 prediction files of the 3 new sets: script, config (= results/lstm_all.json) and input-set name as expected")
for n in ("base_linreg_wqh", "base_rf_wqh"):
    m = F.preds_meta(n); print(n, "meta:", {k: (v if k not in ("features", "hours_covered", "what") else "...") for k, v in m.items()})
jsave("r2i_daymet_window.json", dict(best_end_hour=best, median_corr_by_end_hour={str(h): float(v) for h, v in zip(hours, mc)}, corr_at_09=c09, corr_peak=cmax, peak_hour=hours[int(np.argmax(mc))]))
