"""E2 review, check (b): the basin means of the forecast rain, rebuilt with the reviewer's own code from the stored box data and compared with data/gefs/basins/gefs_c00_<gauge>.csv and results/gefs_basins.json.
 1. AREA WEIGHTS of the 0.25 degree boxes (centre +- 0.125 degree) over each basin outline (CAMELS HCDN_nhru_final_671.shp, read with pyshp), by exact intersection in a basin-centred Lambert azimuthal EQUAL-AREA
    projection (the builder uses EPSG:5070 Albers): compared with the weights of results/gefs_basins.json for ALL 12 basins.
 2. DAY TOTALS of the three local days t+1..t+3 for every issue day t and basin: the run is the 00 UTC run of the UTC date t+1; the local day t+k is the wall-clock day of the gauge (zone from usgs_sites.rdb, summer time
    by zoneinfo); the forecast rain of a day = S(end) - S(start), S(h) the cumulative rain of the run (rain uniform inside a 3-hour window: piecewise linear), a route different from the builder's window weights. Compared with the
    columns fc_raw_k of the tables for all 12 basins and all 5,387 issue days; the issue-time rule and the clock proof re-derived from absolute times: age of the run at the issue time, no fresher 00 UTC run, every hour used lies
    after the issue time and inside the day's UTC span, days of 23/24/25 hours, no hour beyond 78; the first day of each clock change and a Central-time basin shown; the made-up-values test (later runs replaced by random numbers).
 3. The AORC rain of the same local days (my own sums of the cache) against the columns aorc_k of the tables.
Writes f4_basins.json and f4_series.npz (my own raw day totals and AORC days, used by the later checks). About 1-2 minutes."""
import sys; sys.dont_write_bytecode = True
import datetime as dt, glob, time
from zoneinfo import ZoneInfo
import shapefile, shapely
from shapely.geometry import shape, box
from shapely.ops import transform
from pyproj import Transformer
from f_common import *
from f_aorc import *

t0 = time.time()
G = jload("results/gefs_basins.json")
ids = list(G["basins"])
ZONES, TZ = gauge_zones(ids)
SHP = f"{PROJ}/data/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp"
polys = {}
for sr in shapefile.Reader(SHP).iterShapeRecords():
    g = f"{int(sr.record['hru_id']):08d}"
    if g in ids:
        p = shape(sr.shape.__geo_interface__); polys[g] = p if p.is_valid else p.buffer(0)
assert set(polys) == set(ids)
LAT = 25.0 + 0.25 * np.arange(53); LON = -91.0 + 0.25 * np.arange(65)                 # the box (south to north, west to east), checked against the stored arrays in f3_store.py

# ---- 1. the weights -------------------------------------------------------------------------------------------------------------------------------------------------
W, wdev, wrec = {}, {}, {}
for g in ids:
    P = polys[g]; c = P.centroid
    tr = Transformer.from_crs("EPSG:4326", f"+proj=laea +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m", always_xy=True).transform
    PA = transform(tr, P)
    x0, y0, x1, y1 = P.bounds
    cells = []
    for i in np.flatnonzero((LAT + 0.125 > y0) & (LAT - 0.125 < y1)):
        for j in np.flatnonzero((LON + 0.125 > x0) & (LON - 0.125 < x1)):
            b = shapely.segmentize(box(LON[j] - 0.125, LAT[i] - 0.125, LON[j] + 0.125, LAT[i] + 0.125), 0.005)
            a = PA.intersection(transform(tr, b)).area
            if a > 0: cells.append((int(i), int(j), a))
    tot = sum(a for _, _, a in cells)
    W[g] = [(i, j, a / tot) for i, j, a in cells]
    theirs = {(c_["ilat"], c_["ilon"]): c_["weight"] for c_ in G["basins"][g]["weights"]}
    mine = {(i, j): w for i, j, w in W[g]}
    wdev[g] = max(abs(mine.get(k, 0.0) - theirs.get(k, 0.0)) for k in set(mine) | set(theirs))
    wrec[g] = dict(n_boxes_mine=len(mine), n_boxes_builder=len(theirs), same_boxes=set(mine) == set(theirs), max_abs_weight_difference=wdev[g], area_km2_laea=float(PA.area / 1e6), area_km2_builder=G["basins"][g]["outline_area_km2_equal_area"],
                   weights_mine={f"{i},{j}": w for (i, j), w in sorted(mine.items())}, zone=ZONES[g], tz_cd=TZ[g], zone_equals_builder=(ZONES[g] == G["basins"][g]["time_zone"]))
print(f"1. weights: boxes per basin {[len(W[g]) for g in ids]}; same boxes as the builder in every basin: {all(wrec[g]['same_boxes'] for g in ids)}; largest difference of a weight {max(wdev.values()):.2e} (LAEA against EPSG:5070); outline areas {min(r['area_km2_laea'] for r in wrec.values()):.0f}-{max(r['area_km2_laea'] for r in wrec.values()):.0f} km2; zone equals the builder's: {all(r['zone_equals_builder'] for r in wrec.values())}")

# ---- the store: the needed boxes of every run ------------------------------------------------------------------------------------------------------------------------
cells_all = sorted({(i, j) for g in ids for i, j, _ in W[g]}); col = {c: n for n, c in enumerate(cells_all)}
il, jl = np.array([c[0] for c in cells_all]), np.array([c[1] for c in cells_all])
inits, rain = [], []
for f in sorted(glob.glob(f"{PROJ}/data/gefs/c00/apcp_20*.npz")):
    z = np.load(f)
    assert (z["flag"] == 0).all()
    inits.append(pd.to_datetime(z["init_date"].astype(str), format="%Y%m%d")); rain.append(z["apcp"][:, :, il, jl].astype(np.float64) / 10.0)
inits = pd.DatetimeIndex(np.concatenate([i.values for i in inits])); rain = np.concatenate(rain)             # [run, 26 windows, cell]
assert inits.is_unique and inits.is_monotonic_increasing and len(inits) == 5392
pos_of = {d: n for n, d in enumerate(inits)}
T = pd.date_range("2000-01-01", "2014-09-30")                                                                  # the issue days (local calendar days)
pos = np.array([pos_of[d + pd.Timedelta(days=1)] for d in T])                                                  # the run of the UTC date t+1

# ---- 2. day totals and the clock, from absolute times -----------------------------------------------------------------------------------------------------------------
H = np.arange(0, 81, 3)                                                                                        # window edges in hours after the run
def day_total(w, lo, hi):
    """rain of the forecast hours (lo, hi] from the 26 window amounts w [n, 26] (uniform inside a window): S(hi) - S(lo), S piecewise linear."""
    cs = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(w, axis=1)], axis=1)                              # S at the edges
    def S(h):
        k = np.minimum((h // 3).astype(int), 25); fr = (h - 3 * k) / 3.0
        return cs[np.arange(len(h)), k] + fr * w[np.arange(len(h)), k]
    return S(hi) - S(lo)

RAW, AOR, clock, csv_dev, aorc_dev, hours_rec = {}, {}, {}, {}, {}, {}
days64 = T.values.astype("datetime64[D]")
days_ext = np.arange(np.datetime64("1980-10-01"), np.datetime64("2014-10-04"))
for g in ids:
    zone = ZONES[g]
    w = np.einsum("rwc,c->rw", rain[:, :, [col[(i, j)] for i, j, _ in W[g]]], np.array([x for _, _, x in W[g]]))[pos]          # basin-mean window rain of the run used on each issue day [n_issue, 26]
    Rt = np.array([np.datetime64(f"{d + np.timedelta64(1, 'D')}T00:00", "m") for d in days64])                   # 00 UTC of the UTC date t+1 (instant of the run)
    issue = np.array([local_midnight_utc(d + 1, zone) for d in days64])                                          # local midnight at the end of day t
    age_h = (issue - Rt) / np.timedelta64(60, "m")
    assert set(np.unique(age_h)) <= {4.0, 5.0, 6.0}, set(np.unique(age_h))
    assert (Rt <= issue).all() and ((Rt + np.timedelta64(24 * 60, "m")) > issue).all()                            # the run exists at the issue time; the next 00 UTC run (date t+2) does not
    tot = np.zeros((len(T), 3)); hrs = np.zeros((len(T), 3), int); first_h = np.full((len(T), 3), np.inf); last_h = np.full((len(T), 3), -np.inf)
    prev_hi = None
    for k in KS:
        a = np.array([local_midnight_utc(d + k, zone) for d in days64]); b = np.array([local_midnight_utc(d + k + 1, zone) for d in days64])
        lo = (a - Rt) / np.timedelta64(60, "m"); hi = (b - Rt) / np.timedelta64(60, "m")
        assert (lo >= age_h).all() and (lo > 0).all() and (hi <= 78).all() and (hi > lo).all(), f"{g}: a forecast hour used lies before the issue time or beyond hour 78"
        if k == 1: assert (lo == age_h).all()
        else: assert (lo == prev_hi).all()
        assert set(np.unique(hi - lo)) <= {23.0, 24.0, 25.0}
        assert np.allclose(hi % 1, 0) and np.allclose(lo % 1, 0)
        tot[:, k - 1] = day_total(w, lo, hi); hrs[:, k - 1] = (hi - lo).astype(int); first_h[:, k - 1] = lo; last_h[:, k - 1] = hi; prev_hi = hi
    RAW[g] = tot
    d = pd.read_csv(f"{PROJ}/data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)
    assert d.index.equals(T)
    assert (d.init_utc_date.values == (T + pd.Timedelta(days=1)).strftime("%Y%m%d").astype(int)).all()            # the run named in the table is the one of the UTC date t+1
    diff = np.abs(np.stack([d[f"fc_raw_{k}"].values for k in KS], 1) - tot)
    csv_dev[g] = dict(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()), n_above_1e_5=int((diff > 1e-5).sum()), max_abs_diff_by_lead=[float(diff[:, k].max()) for k in range(3)])
    # the AORC rain of the local days t+k
    A = np.zeros((len(T), 3))
    day_idx = {d_: n for n, d_ in enumerate(days_ext)}
    ser, nh = aorc_local_days(g, zone, days_ext)
    for k in KS:
        A[:, k - 1] = ser[[day_idx[d_ + np.timedelta64(k, "D")] for d_ in days64]]
    AOR[g] = A
    ad = np.abs(np.stack([d[f"aorc_{k}"].values for k in KS], 1) - A)
    aorc_dev[g] = dict(max_abs_diff=float(ad.max()), n_above_1e_5=int((ad > 1e-5).sum()))
    clock[g] = dict(zone=zone, age_of_the_run_h={str(int(v)): int((age_h == v).sum()) for v in np.unique(age_h)}, hours_per_day={f"k{k}": {str(v): int((hrs[:, k - 1] == v).sum()) for v in (23, 24, 25)} for k in KS},
                    first_forecast_hour_used_min_max=[float(first_h.min()), float(first_h.max())], last_forecast_hour_used_max=float(last_h.max()))
    # the made-up-values test: every run after the one used on issue day c is replaced by random rain; no total issued on or before c may change
    rng = np.random.default_rng(1); cut_ok = []
    h1 = hrs[:, 0]
    firsts = [int(np.flatnonzero(h1 == v)[0]) for v in (23, 25) if (h1 == v).any()]
    for c in sorted({*map(int, rng.integers(10, len(T) - 10, 3)), *[x for x in firsts if 10 < x < len(T) - 10]}):
        alt = rain.copy()
        alt[pos[c] + 1:] = rng.uniform(0, 200, alt[pos[c] + 1:].shape)
        w2 = np.einsum("rwc,c->rw", alt[:, :, [col[(i, j)] for i, j, _ in W[g]]], np.array([x for _, _, x in W[g]]))[pos]
        tot2 = np.zeros((len(T), 3)); prev = None
        for k in KS:
            a = np.array([local_midnight_utc(d + k, zone) for d in days64]); b = np.array([local_midnight_utc(d + k + 1, zone) for d in days64])
            tot2[:, k - 1] = day_total(w2, (a - Rt) / np.timedelta64(60, "m"), (b - Rt) / np.timedelta64(60, "m"))
        cut_ok.append(bool(np.array_equal(tot2[:c + 1], tot[:c + 1]) and not np.allclose(tot2[c + 1], tot[c + 1])))
        del alt
    clock[g]["made_up_values_test"] = dict(n_cuts=len(cut_ok), all_ok=bool(all(cut_ok)), cut_includes_first_23_and_25_hour_days=firsts)
    hours_rec[g] = hrs
print("2. clock (own derivation from absolute times): age of the run at the issue time (hours: days) per zone:", {z: clock[next(g for g in ids if ZONES[g] == z)]["age_of_the_run_h"] for z in sorted(set(ZONES.values()))})
print(f"   made-up-values test (later runs replaced by random rain; totals issued on or before the cut must not change): {all(clock[g]['made_up_values_test']['all_ok'] for g in ids)} in every basin")
print(f"   day totals against the tables (12 basins x 5,387 issue days x 3 leads): largest difference {max(v['max_abs_diff'] for v in csv_dev.values()):.2e} mm (the tables are rounded to 1e-6), values above 1e-5: {sum(v['n_above_1e_5'] for v in csv_dev.values())}")
print(f"   AORC local days (my own sums of the cache) against the table columns aorc_k: largest difference {max(v['max_abs_diff'] for v in aorc_dev.values()):.2e} mm, values above 1e-5: {sum(v['n_above_1e_5'] for v in aorc_dev.values())}")

# the clock-change days and a Central-time basin shown
def zone_changes(zone):
    out = []
    for d in days64[:-4]:
        h = [(local_midnight_utc(d + k + 1, zone) - local_midnight_utc(d + k, zone)) / np.timedelta64(60, "m") for k in KS]
        if any(x != 24 for x in h): out.append((str(d), h))
    return out
chg_east = zone_changes("America/New_York"); chg_cent = zone_changes("America/Chicago")
g_east, g_cent = "02137727", "02472000"
show = []
for name, g in (("Eastern (Catawba)", g_east), ("Central (Leaf River)", g_cent)):
    d = pd.read_csv(f"{PROJ}/data/gefs/basins/gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)
    for day in ("2005-04-02", "2005-10-29", "2007-03-10", "2010-11-06", "2013-03-09", "2014-03-08", "2008-11-01"):
        n = int(np.flatnonzero(T == pd.Timestamp(day))[0])
        show.append(dict(basin=name, issue_day=day, hours_of_the_three_days=[int(x) for x in hours_rec[g][n]], mine=[round(float(x), 6) for x in RAW[g][n]], table=[round(float(d[f"fc_raw_{k}"].iloc[n]), 6) for k in KS],
                         age_of_the_run_h=float((local_midnight_utc(days64[n] + 1, ZONES[g]) - np.datetime64(f"{days64[n] + np.timedelta64(1, 'D')}T00:00", "m")) / np.timedelta64(60, "m"))))
print("   clock-change and Central-time examples (hours of t+1, t+2, t+3; my totals; the table's):")
for s in show[:4] + show[7:11]: print("     ", s)
n_cc = lambda ch: sum(1 for _, h in ch if any(x != 24 for x in h))
print(f"   issue days on which one of the three days has 23 or 25 hours: Eastern {n_cc(chg_east)}, Central {n_cc(chg_cent)} of {len(days64) - 4}")
out = dict(weights=wrec, weights_max_difference=max(wdev.values()), csv_comparison=csv_dev, aorc_comparison=aorc_dev, clock=clock, examples=show, issue_days=len(T), n_clock_change_issue_days=dict(eastern=n_cc(chg_east), central=n_cc(chg_cent)),
           rule_proven=dict(run="00 UTC of the UTC date t+1 (init_utc_date of every table row equal, all basins)", ages_h=sorted({int(v) for g in ids for v in map(float, clock[g]["age_of_the_run_h"])}), no_fresher_run=True,
                            every_hour_after_the_issue_time_and_by_78=True, seconds=round(time.time() - t0, 1)))
jsave("f4_basins.json", out)
np.savez_compressed(f"{OUT}/f4_series.npz", ids=np.array(ids), days=days64.astype("datetime64[D]").astype(str), raw=np.stack([RAW[g] for g in ids]), aorc=np.stack([AOR[g] for g in ids]))
print(f"done in {time.time() - t0:.0f} s")
