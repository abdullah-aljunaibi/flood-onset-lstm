"""E4e review, check 4: the saved networks. Every one of the 30 .pt files of script 22 rebuilt from its file alone (flood.Cfg, flood.LSTMNet, flood.Run.predict, as how_to_reload
says) and run (a) on tables built as the script builds them (the daily table from 2000-01-01, subdaily_inputs.within_day_table with the maxcell fill, fc_bc_1..3 of data/gefs/basins,
the script's flow_state_table executed from its text) and (b) on tables whose hourly-flow columns are the REVIEWER's own (OUT/my_columns.npz, h3), compared with the stored
forecasts bit for bit; the scalers inside every file recomputed from the training rows; the settings; the reviewer's own made-up-values test on a saved wqaH_s network and a
saved wqaWHF_s network (every input column and the flow after a cut day replaced by random values; no forecast issued on or before the cut may change). About 4 minutes with one thread."""
from h_common import *
import torch
torch.set_num_threads(int(os.environ.get("FLOOD_THREADS", "1")))
import flood as F
import gefs_to_basins as GB
import subdaily_inputs as SI

t0 = time.time()
R = jload(RESULTS)
conf = jload("results/lstm_all.json")["config"]
COV = R["data_coverage"]["per_basin"]
D, B, ZONES = F.load_data(), F.basins(), zones()
ids = R["basins"]
assert list(D) == ids
thr = R["thresholds"]
ns = dict(pd=pd, np=np, SI=SI, ZONES=ZONES, CARRY_H=CARRY_H, MISMATCH_REL=MISMATCH_REL, thr=thr, MM=lambda g: 0.0283168 * 86400 / (COV[g]["area_km2_used"] * 1e6) * 1000)
flow_state_table, _ = script_function("flow_state_table", ns)
MY = np.load(OUT / "my_columns.npz", allow_pickle=False)
assert [str(b) for b in MY["basins"]] == ids and pd.DatetimeIndex(MY["days"]).equals(T_ISS)
TAB, TAB_MINE = {}, {}
for i, g in enumerate(ids):
    z = np.load(PROJ / "data" / "hourly_flow" / f"usgs_iv_{g}.npz", allow_pickle=False)
    stamp = pd.DatetimeIndex(z["stamp_utc_s"].astype("datetime64[s]"))
    d = D[g].loc[T_ISS[0]:].copy()
    assert d.index.equals(T_ISS)
    W = SI.within_day_table(g, ZONES[g], T_ISS).drop(columns=["hours"])
    nan = W.aorc_maxcell1h.isna()
    W["aorc_maxcell1h"] = W.aorc_maxcell1h.where(~nan, W.aorc_max1h)
    c = GB.load_columns(g)
    fc = pd.DataFrame({f"fc{k}": c[f"fc_bc_{k}"].values for k in LEADS}, index=T_ISS)
    H24, H22 = flow_state_table(stamp, z["q_mean_cfs"].astype(float), z["q_last_cfs"].astype(float), g, d.q_mm, 24), flow_state_table(stamp, z["q_mean_cfs"].astype(float), z["q_last_cfs"].astype(float), g, d.q_mm, 22).add_suffix("_22")
    TAB[g] = pd.concat([d, W, fc, H24, H22], axis=1)
    mine = pd.DataFrame({c_: MY[c_][i] for c_ in list(H_COLS) + list(H22_COLS)}, index=T_ISS)
    TAB_MINE[g] = pd.concat([d, W, fc, mine], axis=1)
    assert np.isfinite(TAB[g][list(BASE_DYN + W_COLS + FC_COLS + H_COLS + H22_COLS)].values).all() and np.isfinite(TAB_MINE[g][list(BASE_DYN + W_COLS + FC_COLS + H_COLS + H22_COLS)].values).all()
table_diff = max(float(np.max(np.abs(TAB[g][list(H_COLS + H22_COLS)].values - TAB_MINE[g][list(H_COLS + H22_COLS)].values))) for g in ids)
print(f"tables built ({len(ids)} basins x {len(T_ISS)} days); the script's hourly-flow columns against mine: max abs difference {table_diff:.1e}  [{time.time() - t0:.0f} s]", flush=True)


def load_run(name, seed):
    z = torch.load(PROJ / "results" / "models" / f"lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    assert z["state_dict"]["lstm.weight_ih_l0"].shape[1] == n_in
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z, n_in


def stored(name):
    """The stored forecasts (whole record 1980-2014) restricted to the table's days, as arrays [basin, day, (obs, l1, l2, l3)]; and whether everything before 2000-01-01 is empty."""
    dates, basins, obs, pred, _ = load_npz(name)
    assert basins == ids
    sel = np.asarray((dates >= T_ISS[0]) & (dates <= T_ISS[-1]))
    before_empty = bool(np.isnan(pred[:, ~sel, :]).all())
    return np.concatenate([obs[:, sel, None], pred[:, sel, :]], axis=2), before_empty


def compare(a, new):
    b = np.stack([new[g][["obs", "l1", "l2", "l3"]].values for g in ids])
    fin = ~np.isnan(a)
    big = fin & (np.abs(a) > 0.1)
    return dict(same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs_diff=float(np.abs(a[fin] - b[fin]).max()), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)),
                allclose_rtol_1e_6=bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)), max_rel_diff_above_0_1_mm=float((np.abs(a[big] - b[big]) / np.abs(a[big])).max()) if big.any() else 0.0, n_values=int(a.size), n_nan=int((~fin).sum()))


TE = within(T_ISS, *SAT_SPLIT["test"])
tr = within(T_ISS, *SAT_SPLIT["train"])
out = dict(torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(), tables_script_route_vs_mine_max_abs_diff=table_diff, reproduction={})
for n in NEW:
    for s in SEEDS[n]:
        run, z, n_in = load_run(n, s)
        a, before_empty = stored(f"lstm_{n}_s{s}")
        p2, p3 = run.predict(TAB, ids, B), run.predict(TAB_MINE, ids, B)
        rec = dict(n_in=n_in, script_tables=compare(a, p2), my_tables=compare(a, p3), stored_empty_before_2000=before_empty,
                   first_forecast_day=str(T_ISS[int(np.argmax(np.isfinite(p2[ids[0]]["l1"].values)))].date()))
        dyn = np.concatenate([TAB[g].loc[tr, list(run.cfg.dyn)].values for g in ids])
        q_tr = np.concatenate([TAB[g].q_mm.values[tr] for g in ids])
        Sx = B.loc[ids, STATIC].values.astype(float)
        mine = dict(dyn_mu=np.nanmean(dyn, 0), dyn_sd=np.nanstd(dyn, 0) + 1e-9, q_mu=float(q_tr.mean()), q_sd=float(q_tr.std()), lq_mu=float(np.log1p(q_tr).mean()), lq_sd=float(np.log1p(q_tr).std()), st_mu=Sx.mean(0), st_sd=Sx.std(0) + 1e-9)
        sc = run.sc
        rec["scalers"] = dict(keys=sorted(sc), max_rel_diff=max(float(np.max(np.abs((np.asarray(sc[k]) - np.asarray(mine[k])) / (np.abs(np.asarray(mine[k])) + 1e-12)))) for k in mine),
                              from_training_rows_only=all(np.allclose(np.asarray(sc[k]), np.asarray(mine[k]), rtol=1e-9, atol=0) for k in mine))
        rec["settings"] = dict(settings_equal_lstm_all=all(z["cfg"][k] == conf[k] for k in conf), seed=z["cfg"]["seed"] == s, split_ok=all(tuple(z["split"][k]) == SAT_SPLIT[k] for k in SAT_SPLIT), future=z["cfg"]["future"], use_flow=z["cfg"]["use_flow"],
                               static=list(z["cfg"]["static"]) == STATIC, dyn_equals_design=list(z["cfg"]["dyn"]) == R["design"]["sets"][n]["dyn"], ids=list(z["ids"]) == ids)
        out["reproduction"][f"lstm_{n}_s{s}"] = rec
        print(f"lstm_{n}_s{s}: n_in {n_in:2d} | script tables: max diff {rec['script_tables']['max_abs_diff']:.1e} bit for bit {rec['script_tables']['bit_for_bit']} | my tables: max diff {rec['my_tables']['max_abs_diff']:.1e} bit for bit {rec['my_tables']['bit_for_bit']} "
              f"| scalers from training rows {rec['scalers']['from_training_rows_only']} | first forecast {rec['first_forecast_day']}  [{time.time() - t0:.0f} s]", flush=True)
rep = out["reproduction"]
out["all_30_bit_for_bit_on_script_tables"] = all(v["script_tables"]["bit_for_bit"] and v["script_tables"]["same_nan_pattern"] for v in rep.values())
out["all_30_bit_for_bit_on_my_tables"] = all(v["my_tables"]["bit_for_bit"] and v["my_tables"]["same_nan_pattern"] for v in rep.values())
out["all_30_within_rtol_1e_6"] = all(v["script_tables"]["allclose_rtol_1e_6"] and v["my_tables"]["allclose_rtol_1e_6"] for v in rep.values())
out["worst_max_abs_diff"] = max(max(v["script_tables"]["max_abs_diff"], v["my_tables"]["max_abs_diff"]) for v in rep.values())
out["scalers_from_training_rows_in"] = f"{sum(v['scalers']['from_training_rows_only'] for v in rep.values())} of {len(rep)}"
out["settings_true_in"] = {k: f"{sum(bool(v['settings'][k]) for v in rep.values())} of {len(rep)}" for k in ("settings_equal_lstm_all", "seed", "split_ok", "static", "dyn_equals_design", "ids")}
out["future_false_in_all"] = all(v["settings"]["future"] is False for v in rep.values())
out["use_flow_true_in_all"] = all(v["settings"]["use_flow"] is True for v in rep.values())
out["stored_empty_before_2000_in_all"] = all(v["stored_empty_before_2000"] for v in rep.values())
out["first_forecast_day"] = sorted({v["first_forecast_day"] for v in rep.values()})
out["file_reload_check_first_seed_bit_for_bit"] = all(v["bit_for_bit"] and v["passed"] for v in R["reload_check"]["per_set"].values())

# the reviewer's own made-up-values test on a saved wqaH_s network and a saved wqaWHF_s network (other seeds than the script's seed-0 test)
rng = np.random.default_rng(20261029)
test_idx, val_idx = np.flatnonzero(TE), np.flatnonzero(within(T_ISS, *SAT_SPLIT["val"]))
tc = ids.index("02349900")
fb_end = [j for j in test_idx[1:] if MY["hourly_known"][tc, j - 1] == 0 and MY["hourly_known"][tc, j] == 1]     # the first known day after a fallback run of Turkey Creek: a cut right where the columns switch route
cuts = sorted({int(rng.choice(test_idx[100:-100])), int(rng.choice(test_idx[100:-100])), int(rng.choice(val_idx[50:-50])), int(fb_end[len(fb_end) // 2]) if fb_end else int(test_idx[300]), int(np.flatnonzero(T_ISS == pd.Timestamp("2013-03-10"))[0])})
mu = {}
for n, s in (("wqaH_s", 2), ("wqaWHF_s", 1)):
    run, z, n_in = load_run(n, s)
    base = run.predict(TAB, ids, B)
    cols = sorted(set(run.cfg.dyn) | {"q_mm"})
    res = []
    for c in cuts:
        alt = {g: TAB[g].copy() for g in ids}
        for g in ids:
            for col in cols:
                x = alt[g][col].values.copy()
                x[c + 1:] = rng.integers(0, 25, len(x) - c - 1) if col.startswith("hours_above") or col == "aorc_wet_hours" else rng.uniform(-2.0, 8.0, len(x) - c - 1) if col.startswith(("dq", "q_end", "q_dmax")) else rng.uniform(0.0, 150.0, len(x) - c - 1)
                alt[g][col] = x
        new = run.predict(alt, ids, B)
        same, moved = True, 0
        for g in ids:
            for k in LEADS:
                a_, b_ = base[g][f"l{k}"].values, new[g][f"l{k}"].values
                same &= bool(np.array_equal(a_[: c + k + 1], b_[: c + k + 1], equal_nan=True))               # every forecast issued on or before the cut day (stored at the target day c+k at the latest)
            moved += int(not np.isclose(base[g]["l1"].values[c + 2], new[g]["l1"].values[c + 2], rtol=0, atol=1e-9))   # the first forecast issued after the cut (issue day c+1, target c+2)
        res.append(dict(cut=str(T_ISS[c].date()), part="test" if c in test_idx else "val", forecasts_up_to_the_cut_unchanged=same, basins_whose_first_forecast_after_the_cut_moved=moved, columns_perturbed=len(cols)))
    mu[f"lstm_{n}_s{s}"] = res
    print(f"made-up values lstm_{n}_s{s}: " + "; ".join(f"{r['cut']} ({r['part']}): unchanged {r['forecasts_up_to_the_cut_unchanged']}, moved after {r['basins_whose_first_forecast_after_the_cut_moved']}/12" for r in res), flush=True)
out["made_up_values"] = dict(per_network=mu, n_cases=sum(len(v) for v in mu.values()), all_pass=all(r["forecasts_up_to_the_cut_unchanged"] and r["basins_whose_first_forecast_after_the_cut_moved"] == len(ids) for v in mu.values() for r in v),
                             what="every dyn column and q_mm of all days after the cut replaced by random values (log-ratio columns -2..8, hour counts 0..24, the rest 0-150); forecasts stored at target days up to c+k (issued on or before c) must be exactly equal; "
                                  "the forecast issued on c+1 must move in every basin; cuts: two test-year days, one validation day, the first known day after a Turkey Creek fallback run, the spring clock change 2013-03-10")
out["seconds"] = round(time.time() - t0, 1)
jsave("h4_networks.json", out)
print(f"all 30 bit for bit on the script's tables: {out['all_30_bit_for_bit_on_script_tables']}; on my tables: {out['all_30_bit_for_bit_on_my_tables']}; worst max abs diff {out['worst_max_abs_diff']:.1e}; scalers from training rows in {out['scalers_from_training_rows_in']}; "
      f"made-up values all pass: {out['made_up_values']['all_pass']}; seconds {out['seconds']}")
