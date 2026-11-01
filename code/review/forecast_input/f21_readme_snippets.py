"""E2 review, check h8: the statements and the code snippets of data/gefs/README.md that can be run in a few seconds, run as written (numpy only, the project's .venv): the loader `from fetch_gefs import load_gefs`
on one month (shapes, dtypes, orientation of lat and lon, step_end_hours, flags), the weights of `local_day_weights` (the weights of a day sum to 8 for a = 4, 5, 6 and k = 1, 2, 3; which steps they pick), the units
(mm = value / 10), and the statement about the size of the 12 study basins against the areas of f4_basins.json (run f4_basins.py first). About 2 seconds. Writes f21_readme_snippets.json."""
import sys; sys.dont_write_bytecode = True
from f_common import *
from fetch_gefs import load_gefs                       # the README's own import: numpy only, no GRIB library needed

init, flag, apcp, lat, lon, steps = load_gefs("2010-01-01", "2010-01-31")
mm = apcp[flag == 0] / 10.0
out = dict(shapes=dict(init=list(init.shape), flag=list(flag.shape), apcp=list(apcp.shape)), apcp_dtype=str(apcp.dtype), flags=sorted(set(flag.tolist())), lat_first_last=[float(lat[0]), float(lat[-1])], lat_south_to_north=bool(np.all(np.diff(lat) > 0)), lon_first_last=[float(lon[0]), float(lon[-1])],
           lon_west_to_east=bool(np.all(np.diff(lon) > 0)), steps_first_last=[int(steps[0]), int(steps[-1])], n_steps=int(len(steps)), mm_mean_per_3h=float(mm.mean()), mm_max=float(mm.max()), no_data_value_present=bool((apcp == 65535).any()))
def local_day_weights(a, k, n=26):               # the README's function, as written
    lo, hi = a + 24 * (k - 1), a + 24 * k
    j = np.arange(n)
    return np.clip(np.minimum(3 * j + 3, hi) - np.maximum(3 * j, lo), 0, 3) / 3
w = {f"a{a}_k{k}": dict(sum=float(local_day_weights(a, k).sum()), first_step=int(np.flatnonzero(local_day_weights(a, k))[0]), last_step=int(np.flatnonzero(local_day_weights(a, k))[-1]), partial_steps=[int(i) for i in np.flatnonzero((local_day_weights(a, k) > 0) & (local_day_weights(a, k) < 1))]) for a in (4, 5, 6) for k in KS}
out["local_day_weights"] = w
F4 = jread("f4_basins.json")["weights"]
area = {g: v["area_km2_laea"] for g, v in F4.items()}
out["basin_areas_km2"] = dict(min=min(area.values()), max=max(area.values()), larger_than_one_cell_630=int(sum(a > 630 for a in area.values())), within_110_330=int(sum(110 <= a <= 330 for a in area.values())), n=len(area))
cell = 0.25 * 111.19 * 0.25 * 111.19 * float(np.cos(np.radians(35.7)))
out["one_cell_km2_at_35_7N"] = cell
out["all_ok"] = bool(out["apcp_dtype"] == "uint16" and out["lat_south_to_north"] and out["lon_west_to_east"] and out["steps_first_last"] == [3, 78] and out["n_steps"] == 26 and all(abs(v["sum"] - 8) < 1e-9 for v in w.values()) and not out["no_data_value_present"])
print(f"load_gefs 2010-01: apcp {tuple(apcp.shape)} {apcp.dtype}, flags {out['flags']}, lat {out['lat_first_last']} south to north: {out['lat_south_to_north']}, lon {out['lon_first_last']}, steps {out['steps_first_last']}; mean {out['mm_mean_per_3h']:.3f} mm per 3 h")
print("weights of a local day sum to 8 for a = 4, 5, 6 and k = 1, 2, 3:", all(abs(v['sum'] - 8) < 1e-9 for v in w.values()), "| a=4, k=1 picks steps", w["a4_k1"]["first_step"], "to", w["a4_k1"]["last_step"], "with partial steps", w["a4_k1"]["partial_steps"])
print(f"basin areas {out['basin_areas_km2']['min']:.0f} to {out['basin_areas_km2']['max']:,.0f} km2; {out['basin_areas_km2']['within_110_330']} of {len(area)} inside 110-330 km2; {out['basin_areas_km2']['larger_than_one_cell_630']} larger than one cell ({cell:.0f} km2 at 35.7 N)")
jsave("f21_readme_snippets.json", out)
