"""AORC review, check (b) part 2: reproduce the 12 saved AORC networks (4 sets x 3 seeds) from tables built with the reviewer's own series (flood.LSTMNet + flood.Run.predict; my_series_aorc.npz from a3_clocks.py),
bit for bit; design flags read from the networks (no input after the issue day, settings equal results/lstm_all.json, early stopping on the validation years, split); the pipeline test with the saved
networks: +100 mm on the AORC column after day c must leave every forecast issued up to day c unchanged and change the first one after it. About one minute."""
import sys; sys.dont_write_bytecode = True
import time
from a_common import *
import flood as F
import camels

t0 = time.time()
D, B = F.load_data(), F.basins(); ids = list(D)
Z = np.load(f"{OUT}/my_series_aorc.npz")
assert list(Z["basins"]) == ids
mine = {c: {g: Z[c][i] for i, g in enumerate(ids)} for c in ("midnight", "to09", "utc")}
D2 = {g: D[g].assign(aorc_to_midnight=mine["midnight"][g], aorc_to_09=mine["to09"][g], aorc_utc_day=mine["utc"][g]) for g in ids}
EXPECT = {"wqa": F.FORCING + ["aorc_to_midnight"], "wqa_to09": F.FORCING + ["aorc_to_09"], "wqa_utc": F.FORCING + ["aorc_utc_day"], "wqa_rep": ["aorc_to_midnight"] + F.FORCING[1:]}


def load_run(name, seed):
    z = torch.load(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt", map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"])
    n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    run = F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
    return z, cfg, n_in, run


out = {"reproduction": {}, "perturbation": {}}
conf = jload("results/lstm_all.json")["config"]
print("reproduction of the stored forecasts from the reviewer's own tables (1 torch thread)")
print("%-14s %-6s %-26s %-10s %-12s %-12s" % ("set", "n_in", "dyn", "n_nan", "max|diff|", "share equal"))
for name in ["wqa", "wqa_to09", "wqa_utc", "wqa_rep"]:
    for seed in (0, 1, 2):
        z, cfg, n_in, run = load_run(name, seed)
        pred = run.predict(D2, z["ids"], B)
        old = F.load_preds(f"lstm_{name}_s{seed}")
        cols = ["obs", "l1", "l2", "l3"]
        a = np.stack([old[g][cols].values for g in ids]); b = np.stack([pred[g][cols].values for g in ids])
        fin = ~np.isnan(a)
        rec = dict(n_in=int(n_in), dyn=list(cfg.dyn), n_values=int(a.size), n_nan=int(np.isnan(a).sum()), same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(b))),
                   max_abs_diff=float(np.abs(a[fin] - b[fin]).max()), share_exactly_equal=float((a[fin] == b[fin]).mean()),
                   future_false=cfg.future is False, use_flow=cfg.use_flow, static_is_STATIC=list(cfg.static) == F.STATIC, dyn_as_expected=list(cfg.dyn) == EXPECT[name],
                   settings_equal_lstm_all=all(z["cfg"][k] == conf[k] for k in conf) and z["cfg"]["seed"] == seed, best_update_is_argmax_val=(z["history"][int(np.argmax([h["val_nse"] for h in z["history"]]))]["update"] == z["best_update"]),
                   stopped_patience_after_best=(z["updates_run"] - z["best_update"] == conf["patience"] * conf["eval_every"]), split_ok=all(tuple(z["split"][k]) == tuple(F.SPLIT[k]) for k in F.SPLIT), quick_mode=z["quick_mode"], script=z["script"], input_set=z["input_set"])
        out["reproduction"][f"{name}_s{seed}"] = rec
        print("%-14s %-6d %-26s %-10d %-12.3e %-12.6f %s" % (f"{name}_s{seed}", n_in, "+".join(cfg.dyn[:1] + cfg.dyn[6:])[:26], rec["n_nan"], rec["max_abs_diff"], rec["share_exactly_equal"], "" if rec["same_nan_pattern"] else "NAN PATTERN DIFFERS"))
out["all_12_bit_for_bit"] = all(v["share_exactly_equal"] == 1.0 and v["same_nan_pattern"] for v in out["reproduction"].values())
out["worst_max_abs_diff"] = max(v["max_abs_diff"] for v in out["reproduction"].values())
chk = ["future_false", "use_flow", "static_is_STATIC", "dyn_as_expected", "settings_equal_lstm_all", "best_update_is_argmax_val", "stopped_patience_after_best", "split_ok"]
out["design_flags_true_in"] = {k: f"{sum(bool(v[k]) for v in out['reproduction'].values())} of 12" for k in chk}
out["quick_mode_any"] = any(v["quick_mode"] for v in out["reproduction"].values())
print(f"\nall 12 networks bit for bit: {out['all_12_bit_for_bit']}; worst max abs difference: {out['worst_max_abs_diff']}")
print("design flags:", out["design_flags_true_in"], "| quick_mode in any:", out["quick_mode_any"], "| script field:", sorted({v['script'] for v in out['reproduction'].values()}))

# pipeline test with the saved networks: change the AORC column AFTER day c; forecasts issued up to day c must not change
rng = np.random.default_rng(777)
idx = D2[ids[0]].index
cs = sorted(set(int(v) for v in rng.integers(400, 12000, 3)) | {int((pd.Timestamp("2006-07-14") - idx[0]).days), int((pd.Timestamp("2010-01-13") - idx[0]).days)})
print("\npipeline test: +100 mm on the AORC column after day c, forecasts issued up to day c at leads 1-3 must be unchanged, the first after c must change")
ok_all = True
for name, col in (("wqa", "aorc_to_midnight"), ("wqa_to09", "aorc_to_09"), ("wqa_utc", "aorc_utc_day"), ("wqa_rep", "aorc_to_midnight")):
    z, cfg, n_in, run = load_run(name, 0)
    base = run.predict(D2, z["ids"], B)
    res_all = []
    for c in cs:
        Dp = {g: D2[g].copy() for g in ids}
        for g in ids:
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
    out["perturbation"][f"{name}:{col}"] = dict(cutoffs=[str(idx[c].date()) for c in cs], all_ok=all(res_all))
    print(f"  lstm_{name}_s0, column {col}: {len(cs)} cut-off days {[str(idx[c].date()) for c in cs]} -> {'PASS' if all(res_all) else 'FAIL'}")
out["perturbation_all_ok"] = ok_all
jsave("a4_networks.json", out)
print("seconds:", round(time.time() - t0))
