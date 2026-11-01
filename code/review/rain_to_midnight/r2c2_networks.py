"""ROUND 2, check (c), part 2: reproduce the saved networks from tables built with the reviewer's own series (flood.LSTMNet + flood.Run.predict), pipeline leakage tests on the new sets,
the NLDAS column rebuilt as 10_rain_timing.py builds it, and an independent estimate of the end of the NLDAS day."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
import time
from r2_common import *
import flood as F
import camels
import r2_clock as C

t0 = time.time()
D, B = F.load_data(), F.basins(); ids = list(D)
Z = np.load(f"{OUT}/my_series_r2.npz")
assert list(Z["basins"]) == ids
mine = {c: {g: Z[c][i] for i, g in enumerate(ids)} for c in ("midnight", "to09", "utc", "nldas")}
older = {g: np.concatenate([[mine["midnight"][g][0]], mine["midnight"][g][:-1]]) for g in ids}
# the NLDAS column the way 10_rain_timing.py builds it: camels.load_basin(g, "nldas"), column prcp, over the same days as the Daymet table, as float
for g in ids:
    nl = camels.load_basin(g, "nldas").loc[F.PERIOD[0]:F.PERIOD[1], "prcp"].astype(float)
    assert nl.index.equals(D[g].index) and not nl.isna().any() and np.array_equal(nl.values, mine["nldas"][g])
D2 = {g: D[g].assign(rain_to_midnight=mine["midnight"][g], rain_to_midnight_older=older[g], rain_to_09=mine["to09"][g], rain_utc_day=mine["utc"][g], prcp_nldas=mine["nldas"][g]) for g in ids}

def load_run(name, seed):
    z = torch.load(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    return z, cfg, n_in, run

out = {"reproduction": {}, "design": {}, "perturbation": {}}
conf = jload("results/lstm_all.json")["config"]
print("reproduction of the stored forecasts from the reviewer's own tables (1 torch thread)")
print("%-14s %-6s %-8s %-10s %-12s %-12s" % ("set", "n_in", "dyn(+)", "n_nan", "max|diff|", "share equal"))
for name in ["wqh_to09", "wqh_utc", "wqnh", "wqh", "wqh_older", "wh", "wh_older"]:
    for seed in (0, 1, 2):
        z, cfg, n_in, run = load_run(name, seed)
        pred = run.predict(D2, z["ids"], B)
        old = F.load_preds(f"lstm_{name}_s{seed}")
        cols = ["obs", "l1", "l2", "l3"]
        a = np.stack([old[g][cols].values for g in ids]); b = np.stack([pred[g][cols].values for g in ids])
        fin = ~np.isnan(a)
        rec = dict(n_in=int(n_in), dyn_extra=list(cfg.dyn[6:]), n_values=int(a.size), n_nan=int(np.isnan(a).sum()), same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(b))),
                   max_abs_diff=float(np.abs(a[fin] - b[fin]).max()), share_exactly_equal=float((a[fin] == b[fin]).mean()),
                   future_false=cfg.future is False, use_flow=cfg.use_flow, static_is_STATIC=list(cfg.static) == F.STATIC, dyn_starts_with_FORCING=list(cfg.dyn[:6]) == F.FORCING,
                   settings_equal_lstm_all=all(z["cfg"][k] == conf[k] for k in conf) and z["cfg"]["seed"] == seed, best_update_is_argmax_val=(z["history"][int(np.argmax([h["val_nse"] for h in z["history"]]))]["update"] == z["best_update"]),
                   stopped_patience_after_best=(z["updates_run"] - z["best_update"] == conf["patience"] * conf["eval_every"]), split_ok=all(tuple(z["split"][k]) == tuple(F.SPLIT[k]) for k in F.SPLIT), quick_mode=z["quick_mode"])
        out["reproduction"][f"{name}_s{seed}"] = rec
        if seed == 0 or name in ("wqh_to09", "wqh_utc", "wqnh"):
            print("%-14s %-6d %-8s %-10d %-12.3e %-12.6f %s" % (f"{name}_s{seed}", n_in, "+".join(rec["dyn_extra"])[:30], rec["n_nan"], rec["max_abs_diff"], rec["share_exactly_equal"], "" if rec["same_nan_pattern"] else "NAN PATTERN DIFFERS"))
new = [k for k in out["reproduction"] if k.startswith(("wqh_to09", "wqh_utc", "wqnh"))]
out["new_9_bit_for_bit"] = all(out["reproduction"][k]["share_exactly_equal"] == 1.0 and out["reproduction"][k]["same_nan_pattern"] for k in new)
out["original_12_bit_for_bit"] = all(out["reproduction"][k]["share_exactly_equal"] == 1.0 and out["reproduction"][k]["same_nan_pattern"] for k in out["reproduction"] if k not in new)
out["worst_max_abs_diff"] = max(v["max_abs_diff"] for v in out["reproduction"].values())
chk = ["future_false", "use_flow", "static_is_STATIC", "dyn_starts_with_FORCING", "settings_equal_lstm_all", "best_update_is_argmax_val", "stopped_patience_after_best", "split_ok"]
out["design_flags_true_in"] = {k: f"{sum(bool(out['reproduction'][n][k]) for n in new)} of {len(new)}" for k in chk}
out["quick_mode_any"] = any(out["reproduction"][n]["quick_mode"] for n in new)
print(f"\nnew 9 networks bit for bit: {out['new_9_bit_for_bit']}; original 12: {out['original_12_bit_for_bit']}; worst max abs difference over all 21: {out['worst_max_abs_diff']}")
print("design flags in the 9 new networks:", out["design_flags_true_in"], "| quick_mode in any:", out["quick_mode_any"])
print("dyn columns of the new sets:", {n: out["reproduction"][f"{n}_s0"]["dyn_extra"] for n in ("wqh_to09", "wqh_utc", "wqnh")}, "n_in:", {n: out["reproduction"][f"{n}_s0"]["n_in"] for n in ("wqh_to09", "wqh_utc", "wqnh")})

# pipeline test with the saved networks: change the new columns AFTER day c; forecasts issued up to day c must not change
rng = np.random.default_rng(777)
idx = D2[ids[0]].index
cs = sorted(set(int(v) for v in rng.integers(400, 12000, 3)) | {int((pd.Timestamp("2006-07-14") - idx[0]).days), int((pd.Timestamp("2010-01-13") - idx[0]).days)})
print("\npipeline test: +100 mm on the new column(s) after day c, forecasts issued up to day c at leads 1-3 must be unchanged, the first after c must change")
ok_all = True
for name, which in (("wqh_to09", ["rain_to_09"]), ("wqh_utc", ["rain_utc_day"]), ("wqnh", ["prcp_nldas", "rain_to_midnight"]), ("wqnh", ["prcp_nldas"]), ("wqnh", ["rain_to_midnight"])):
    z, cfg, n_in, run = load_run(name, 0)
    base = run.predict(D2, z["ids"], B)
    res_all = []
    for c in cs:
        Dp = {g: D2[g].copy() for g in ids}
        for g in ids:
            for col in which:
                Dp[g].iloc[c + 1:, Dp[g].columns.get_loc(col)] += 100.0
        pp = run.predict(Dp, z["ids"], B)
        ok = True
        for k in (1, 2, 3):
            A = np.stack([base[g][f"l{k}"].values for g in ids]); Bp = np.stack([pp[g][f"l{k}"].values for g in ids])
            before = np.arange(len(idx)) <= c + k
            ok &= bool(np.array_equal(A[:, before], Bp[:, before], equal_nan=True))
            after = np.arange(len(idx)) == c + k + 1
            ok &= bool(np.nanmax(np.abs(A[:, after] - Bp[:, after])) > 1e-6)
        res_all.append(ok)
    ok_all &= all(res_all)
    out["perturbation"][f"{name}:{'+'.join(which)}"] = dict(cutoffs=[str(idx[c].date()) for c in cs], all_ok=all(res_all))
    print(f"  lstm_{name}_s0, columns {which}: {len(cs)} cut-off days {[str(idx[c].date()) for c in cs]} -> {'PASS' if all(res_all) else 'FAIL'}")
out["perturbation_all_ok"] = ok_all

# ---- the end of the NLDAS day, measured independently with the hourly ERA5 (own code) -----------------------------------------------------------------------------------------
print("\nthe end of the NLDAS day: correlation of NLDAS rain of day d with the ERA5 rain of the 24 hours ending at 00:00 UTC of date d+1 plus s hours; best s per basin")
st = pd.read_csv(f"{PROJ}/data/hourly/usgs_sites.rdb", sep="\t", comment="#", dtype=str).iloc[1:].set_index("site_no")
ZONE = {g: {"EST": "America/New_York", "EDT": "America/New_York", "CST": "America/Chicago", "CDT": "America/Chicago"}[st.tz_cd[g]] for g in ids}
shifts = list(range(-12, 13))
best, curves = {}, {}
for g in ids:
    t, x, _ = C.read_era5(g)
    cur = []
    for s in shifts:
        ser, _ = C.window_sum(t - np.timedelta64(s * 60, "m"), x, None, "utc", 24)           # a UTC-day window moved s hours later
        cur.append(float(np.corrcoef(mine["nldas"][g], ser)[0, 1]))
    curves[g] = cur
    best[g] = shifts[int(np.argmax(cur))]
med_curve = np.median(np.array([curves[g] for g in ids]), axis=0)
s_med = shifts[int(np.argmax(med_curve))]
end_local = {g: 24 + best[g] + C.STD[ZONE[g]] for g in ids}
print("  best shift s (hours after 00:00 UTC of d+1) by basin:", best, "| median curve peaks at s =", s_med)
print("  the NLDAS day then ends at 00:00 UTC + s = local STANDARD time:", {F.SHORT[g]: f"{end_local[g]:d}:00" for g in ids})
out["nldas_day_end"] = dict(best_shift_h_by_basin=best, median_curve_peak_shift_h=s_med, median_corr_curve={str(s): float(v) for s, v in zip(shifts, med_curve)}, end_local_standard_time_h_by_basin=end_local,
                            latest_end_local_standard_h=max(end_local.values()), issue_time_local_h=24, margin_h=24 - max(end_local.values()))
print(f"  latest end among the 12 basins: {max(end_local.values())}:00 local standard time; the issue time is 24:00 -> margin {24 - max(end_local.values())} h at least; the ERA5 reference is itself 0 to 2 h later than other products (round 1, 3f), so the true end is, if anything, earlier")
jsave("r2c2_networks.json", out)
print("seconds:", round(time.time() - t0))
