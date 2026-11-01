"""E4b review, check 4: the saved networks. Every one of the 20 .pt files of script 21 rebuilt from its file alone (flood.Cfg, flood.LSTMNet, flood.Run.predict, as
how_to_reload says) and run (a) on tables built as the script builds them (subdaily_inputs.add_inputs + the maxcell fill + the _older shift + fs1..fs3 from data/gefs/basins)
and (b) on tables built from the REVIEWER's own columns (OUT/my_columns.npz, r3), compared with the stored forecasts bit for bit; the scalers inside every file recomputed from the
training years; the settings against results/lstm_all.json; the control's network against the test's (same settings, same shapes, only the column names differ); the reviewer's own
made-up-values test on saved wqaW networks (every input column and the flow after a cut day replaced by random values; no forecast issued on or before the cut may change).
Needs OUT/my_columns.npz. About 4 minutes with one thread."""
from r_common import *
import torch
torch.set_num_threads(int(os.environ.get("FLOOD_THREADS", "1")))
import flood as F
import subdaily_inputs as SI

t0 = time.time()
R = jload(RESULTS)
conf = jload("results/lstm_all.json")["config"]
D, B = F.load_data(), F.basins()
ids = list(D)
thr = F.thresholds(D)
days = D[ids[0]].index
OLDER = tuple(c + "_older" for c in WD_COLS)


def fs_columns(g, d):
    """fs1..fs3 as script 21 builds them: the AORC rain of the local day t+k (aorc_to_midnight shifted) up to the issue day 2000-09-30, the bias-corrected GEFS control forecast from 2000-10-01."""
    c = pd.read_csv(PROJ / "data" / "gefs" / "basins" / f"gefs_c00_{g}.csv", index_col="issue_day", parse_dates=True)
    out = {}
    for k in LEADS:
        p = d.aorc_to_midnight.shift(-k)
        fs = p.copy()
        fs.loc["2000-10-01":] = c[f"fc_bc_{k}"].reindex(d.index).loc["2000-10-01":].values
        assert fs.notna().all()
        out[f"fs{k}"] = fs.values
    return out


def finish(d):
    """The script's steps after add_inputs: fill aorc_maxcell1h where NaN, raise it to aorc_max1h where below on partly filled days, add the _older columns and fs1..fs3."""
    d = d.copy()
    nan = d.aorc_maxcell1h.isna()
    d["aorc_maxcell1h"] = d.aorc_maxcell1h.where(~nan, d.aorc_max1h)
    low = d.aorc_maxcell1h < d.aorc_max1h - 1e-6
    d["aorc_maxcell1h"] = d.aorc_maxcell1h.where(~low, d.aorc_max1h)
    for c in WD_COLS:
        d[c + "_older"] = d[c].shift(1).bfill()
    return d


D2 = {g: finish(d) for g, d in SI.add_inputs(D, thr).items()}                               # (a) the script's tables
MY = np.load(OUT / "my_columns.npz", allow_pickle=False)
assert [str(b) for b in MY["basins"]] == ids
D3 = {}
for i, g in enumerate(ids):
    d = D[g].assign(**{c: MY[c][i] for c in ("aorc_to_midnight",) + WD_COLS}, q_rel=np.log1p(D[g].q_mm.values / thr[g]))
    D3[g] = finish(d)                                                                          # (b) the reviewer's tables
for g in ids:
    fs2, fs3 = fs_columns(g, D2[g]), fs_columns(g, D3[g])
    for k in LEADS:
        D2[g][f"fs{k}"], D3[g][f"fs{k}"] = fs2[f"fs{k}"], fs3[f"fs{k}"]
cols_all = list(BASE_DYN) + list(WD_COLS) + list(OLDER) + ["q_rel", "fs1", "fs2", "fs3"]
table_diff = max(float(np.nanmax(np.abs(D2[g][cols_all].values - D3[g][cols_all].values))) for g in ids)
print(f"tables built ({len(ids)} basins x {len(days)} days); the script's route against mine: max abs difference over all input columns {table_diff:.1e}")


def load_run(name, seed):
    z = torch.load(PROJ / "results" / "models" / f"lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg)
    net.load_state_dict(z["state_dict"])
    net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z, n_in


def compare(old, new):
    cols = ["obs", "l1", "l2", "l3"]
    a, b = np.stack([old[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    fin = ~np.isnan(a)
    big = fin & (np.abs(a) > 0.1)                                                               # forecasts above 0.1 mm/day: the relative difference where it means something
    return dict(same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs_diff=float(np.abs(a[fin] - b[fin]).max()), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)),
                allclose_rtol_1e_6=bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)), allclose_atol_1e_5=bool(np.allclose(a, b, rtol=0, atol=1e-5, equal_nan=True)),
                max_rel_diff_above_0_1_mm=float((np.abs(a[big] - b[big]) / np.abs(a[big])).max()), n_values=int(a.size), n_nan=int((~fin).sum()))


TE = within(days, *SPLIT["test"])


def decisions(P):
    """What the scores read: the pooled onset hits and false alarms at trigger 1.0 and the per-basin NSE on the test years, per lead (own code)."""
    out = {}
    for k in LEADS:
        h = fa = 0
        nse = []
        for g in ids:
            o, f = P[g].obs.values, P[g][f"l{k}"].values
            issue = np.concatenate([np.full(k, np.nan), o[:-k]])
            ok = TE & np.isfinite(o) & np.isfinite(f)
            hi, warn, low = (o > thr[g])[ok], (f > thr[g])[ok], (issue <= thr[g])[ok]
            h += int((hi & warn & low).sum()); fa += int((~hi & warn & low).sum())
            nse.append(1 - np.sum((f[ok] - o[ok]) ** 2) / np.sum((o[ok] - o[ok].mean()) ** 2))
        out[k] = (h, fa, np.array(nse))
    return out


def same_decisions(P1, P2):
    d1, d2 = decisions(P1), decisions(P2)
    return dict(same_onset_hits_and_false_alarms_all_leads=all(d1[k][:2] == d2[k][:2] for k in LEADS), max_abs_nse_diff=float(max(np.abs(d1[k][2] - d2[k][2]).max() for k in LEADS)))


out = dict(torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(), tables_script_route_vs_mine_max_abs_diff=table_diff, reproduction={}, scalers={}, settings={})
tr = within(days, *SPLIT["train"])
for n in NEW:
    for s in SEEDS[n]:
        run, z, n_in = load_run(n, s)
        old = F.load_preds(f"lstm_{n}_s{s}")
        p2, p3 = run.predict(D2, z["ids"], B), run.predict(D3, z["ids"], B)
        rec = dict(n_in=n_in, dyn=list(run.cfg.dyn), script_tables=compare(old, p2), my_tables=compare(old, p3), my_tables_decisions=same_decisions(old, p3))
        # the scalers: recomputed from the training years of the script's tables (flood.fit's formula), compared with the file's
        dyn = np.concatenate([D2[g].loc[tr, list(run.cfg.dyn)].values for g in ids])
        q_tr = np.concatenate([D2[g].q_mm.values[tr] for g in ids])
        Sx = B.loc[ids, STATIC].values.astype(float)
        mine = dict(dyn_mu=np.nanmean(dyn, 0), dyn_sd=np.nanstd(dyn, 0) + 1e-9, q_mu=float(q_tr.mean()), q_sd=float(q_tr.std()), lq_mu=float(np.log1p(q_tr).mean()), lq_sd=float(np.log1p(q_tr).std()), st_mu=Sx.mean(0), st_sd=Sx.std(0) + 1e-9)
        sc = run.sc
        rec["scalers"] = dict(keys=sorted(sc), max_rel_diff=max(float(np.max(np.abs((np.asarray(sc[k]) - np.asarray(mine[k])) / (np.abs(np.asarray(mine[k])) + 1e-12)))) for k in mine),
                              from_training_years_only=all(np.allclose(np.asarray(sc[k]), np.asarray(mine[k]), rtol=1e-9, atol=0) for k in mine))
        rec["settings"] = dict(settings_equal_lstm_all=all(z["cfg"][k] == conf[k] for k in conf), seed=z["cfg"]["seed"] == s, split_ok=all(tuple(z["split"][k]) == SPLIT[k] for k in SPLIT), future=z["cfg"]["future"], use_flow=z["cfg"]["use_flow"],
                               static=list(z["cfg"]["static"]) == STATIC, dyn_equals_design=list(z["cfg"]["dyn"]) == R["design"]["sets"][n]["dyn"])
        out["reproduction"][f"lstm_{n}_s{s}"] = rec
        print(f"lstm_{n}_s{s}: n_in {n_in:2d} | script tables: max diff {rec['script_tables']['max_abs_diff']:.1e} bit for bit {rec['script_tables']['bit_for_bit']} | my tables: max diff {rec['my_tables']['max_abs_diff']:.1e} bit for bit {rec['my_tables']['bit_for_bit']} | scalers from training years {rec['scalers']['from_training_years_only']}")
rep = out["reproduction"]
out["all_20_bit_for_bit_on_script_tables"] = all(v["script_tables"]["bit_for_bit"] and v["script_tables"]["same_nan_pattern"] for v in rep.values())
out["all_20_bit_for_bit_on_my_tables"] = all(v["my_tables"]["bit_for_bit"] and v["my_tables"]["same_nan_pattern"] for v in rep.values())
out["all_20_within_rtol_1e_6"] = all(v["script_tables"]["allclose_rtol_1e_6"] and v["my_tables"]["allclose_rtol_1e_6"] for v in rep.values())
out["all_20_within_atol_1e_5_on_my_tables"] = all(v["my_tables"]["allclose_atol_1e_5"] for v in rep.values())
out["worst_rel_diff_above_0_1_mm_on_my_tables"] = max(v["my_tables"]["max_rel_diff_above_0_1_mm"] for v in rep.values())
out["all_20_same_warnings_on_my_tables"] = all(v["my_tables_decisions"]["same_onset_hits_and_false_alarms_all_leads"] for v in rep.values())
out["worst_nse_diff_on_my_tables"] = max(v["my_tables_decisions"]["max_abs_nse_diff"] for v in rep.values())
out["worst_max_abs_diff"] = max(max(v["script_tables"]["max_abs_diff"], v["my_tables"]["max_abs_diff"]) for v in rep.values())
out["scalers_from_training_years_in"] = f"{sum(v['scalers']['from_training_years_only'] for v in rep.values())} of {len(rep)}"
out["settings_true_in"] = {k: f"{sum(bool(v['settings'][k]) for v in rep.values())} of {len(rep)}" for k in ("settings_equal_lstm_all", "seed", "split_ok", "static", "dyn_equals_design")}
out["future_false_in_all"] = all(v["settings"]["future"] is False for v in rep.values())
out["use_flow_true_in_all"] = all(v["settings"]["use_flow"] is True for v in rep.values())

# the control against the test: same settings, same shapes, only the names of the six columns differ (the _older columns are the within-day columns shifted one day later: r3 and the tables above)
zW, zO = (torch.load(PROJ / "results" / "models" / f"lstm_{n}_s0.pt", map_location="cpu", weights_only=True) for n in ("wqaW", "wqaW_older"))
cW, cO = dict(zW["cfg"]), dict(zO["cfg"])
out["control"] = dict(cfg_keys_that_differ=[k for k in cW if cW[k] != cO[k]], dyn_test=list(cW["dyn"]), dyn_control=list(cO["dyn"]),
                      control_columns_are_the_test_columns_plus_older=[c.replace("_older", "") for c in cO["dyn"]] == list(cW["dyn"]),
                      state_dict_shapes_equal=all(tuple(zW["state_dict"][k].shape) == tuple(zO["state_dict"][k].shape) for k in zW["state_dict"]) and set(zW["state_dict"]) == set(zO["state_dict"]),
                      n_params_equal=zW["n_params"] == zO["n_params"], split_equal=zW["split"] == zO["split"], extra_columns_control=list(zO["extra_columns"]),
                      older_shift_verified="the control's stored forecasts are reproduced bit for bit from the script's tables and within 1e-5 mm/day (same warnings, same NSE) from the reviewer's own tables, whose _older columns are the within-day columns shifted one day later (first day kept): " + str(all(rep[f"lstm_wqaW_older_s{s}"]["script_tables"]["bit_for_bit"] and rep[f"lstm_wqaW_older_s{s}"]["my_tables"]["allclose_atol_1e_5"] and rep[f"lstm_wqaW_older_s{s}"]["my_tables_decisions"]["same_onset_hits_and_false_alarms_all_leads"] for s in SEEDS["wqaW_older"])))
print("control vs test:", {k: v for k, v in out["control"].items() if k not in ("dyn_test", "dyn_control")})

# the reviewer's own made-up-values test on saved wqaW networks (seeds 0 and 3): every input column and the observed flow after a cut day replaced by random values
rng = np.random.default_rng(20261029)
test_idx = np.flatnonzero(within(days, *SPLIT["test"]))
val_idx = np.flatnonzero(within(days, *SPLIT["val"]))
cuts = sorted({int(rng.choice(test_idx[100:-100])), int(rng.choice(test_idx[100:-100])), int(rng.choice(val_idx[100:-100])), int(np.flatnonzero(days == pd.Timestamp("2009-11-11"))[0]), int(np.flatnonzero(days == pd.Timestamp("2013-07-04"))[0])})
mu = {}
for s in (0, 3):
    run, z, n_in = load_run("wqaW", s)
    base = run.predict(D2, ids, B)
    cols = sorted(set(run.cfg.dyn) | {"q_mm"})
    res = []
    for c in cuts:
        alt = {g: D2[g].copy() for g in ids}
        for g in ids:
            for col in cols:
                x = alt[g][col].values.copy()
                x[c + 1:] = rng.uniform(0.0, 150.0, len(x) - c - 1) if col != "aorc_wet_hours" else rng.integers(0, 25, len(x) - c - 1)
                alt[g][col] = x
        new = run.predict(alt, ids, B)
        same, moved = True, 0
        for g in ids:
            for k in LEADS:
                a, b = base[g][f"l{k}"].values, new[g][f"l{k}"].values
                same &= bool(np.array_equal(a[: c + k + 1], b[: c + k + 1], equal_nan=True))               # every forecast issued on or before the cut day (stored at target day c+k at the latest)
            moved += int(not np.isclose(base[g]["l1"].values[c + 2], new[g]["l1"].values[c + 2], rtol=0, atol=1e-9))   # the first forecast issued after the cut (issue day c+1, target c+2)
        res.append(dict(cut=str(days[c].date()), part="test" if c in test_idx else "val", forecasts_up_to_the_cut_unchanged=same, basins_whose_first_forecast_after_the_cut_moved=moved, columns_perturbed=len(cols)))
    mu[f"lstm_wqaW_s{s}"] = res
    print(f"made-up values lstm_wqaW_s{s}: " + "; ".join(f"{r['cut']} ({r['part']}): unchanged {r['forecasts_up_to_the_cut_unchanged']}, moved after {r['basins_whose_first_forecast_after_the_cut_moved']}/12" for r in res))
out["made_up_values"] = dict(per_network=mu, n_cases=sum(len(v) for v in mu.values()), all_pass=all(r["forecasts_up_to_the_cut_unchanged"] and r["basins_whose_first_forecast_after_the_cut_moved"] == len(ids) for v in mu.values() for r in v),
                             what="every dyn column and q_mm of all days after the cut replaced by random values (0-150 mm/day; wet hours 0-24); forecasts stored at target days up to c+k (issued on or before c) must be exactly equal; the forecast issued on c+1 must move")
out["seconds"] = round(time.time() - t0, 1)
jsave("r4_networks.json", out)
print(f"all 20 bit for bit on the script's tables: {out['all_20_bit_for_bit_on_script_tables']}; on my tables: {out['all_20_bit_for_bit_on_my_tables']} (within 1e-5 mm/day: {out['all_20_within_atol_1e_5_on_my_tables']}, same warnings: {out['all_20_same_warnings_on_my_tables']}, worst NSE diff {out['worst_nse_diff_on_my_tables']:.1e}); worst max abs diff {out['worst_max_abs_diff']:.1e}; scalers from training years in {out['scalers_from_training_years_in']}; made-up values all pass: {out['made_up_values']['all_pass']}; seconds {out['seconds']}")
