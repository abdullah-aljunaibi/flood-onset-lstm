"""E2 review, check (f) and (d): the saved networks' forecasts reproduced from the reviewer's own tables (f_tables.py), through the engine's flood.LSTMNet / Run.predict (allowed). Not trained, not changed.
 1. ALL 30 saved networks (10 sets x 3 seeds; the brief asks for one seed of five sets) rebuilt from their .pt files and given my table of their arm (variant 'csv': the forecast columns read from the tables of
    data/gefs/basins, my own AORC days, my own shuffle copy); the forecasts over the whole record (1980-2014, or 2000-2014 for the satellite arm, put back on the whole record as the builder does) compared value by value
    with results/preds/lstm_<set>_s<seed>.npz: bit for bit expected. Variant 'own' (my own day totals x my own factors, rounded to six decimals): the same for one seed of five sets.
 2. The derived sets (the saved wqaF_sub networks given the RAW forecast, given the PERFECT rain; the saved wqaP networks given the forecast): rebuilt with my tables, compared with the stored lstm_wqaF_sub_rawpred,
    lstm_wqaF_sub_perfectpred, lstm_wqaP_forecastpred.
 3. The networks of E1b (lstm_wqa, seeds 0-2) given my main table (the base column aorc_to_midnight): the stored forecasts of script 16.
 4. The 'shared trajectory' of wqaF_sub, wqaF_clim and wqaP: the training loss at every common check, the best updates, whether the weights of the three are identical when the best updates are, what the stopping update changes.
 5. The made-up-values test of my own: for each of the five sets of the brief (first seed) and three cut days in the test years, every input column and the flow after the cut replaced by random numbers: no forecast issued
    on or before the cut changes, the later ones do.
About 3-4 minutes (one forward pass of a network per item)."""
import sys; sys.dont_write_bytecode = True
import pickle, time
from f_common import *
import flood as F

t0 = time.time()
TAB, ids = pickle.load(open(f"{OUT}/f6_tables_csv.pkl", "rb")); TABO, _ = pickle.load(open(f"{OUT}/f6_tables_own.pkl", "rb"))
D = F.load_data(); B = F.basins()
assert list(D) == ids


def run_from_file(f):
    z = torch.load(f, map_location="cpu", weights_only=True)
    cfg = F.Cfg(**z["cfg"]); n_in = len(cfg.dyn) * (1 + 3 * int(cfg.future)) + int(cfg.use_flow) + len(cfg.static)
    net = F.LSTMNet(n_in, cfg); net.load_state_dict(z["state_dict"]); net.eval()
    sc = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
    return F.Run(cfg, sc, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"]), z


def frames(run, tab, arm):
    p = run.predict(tab, ids, B)
    if arm == "sat":
        p = {g: p[g].reindex(D[g].index).assign(obs=D[g].q_mm) for g in ids}
    return p


def stored(name):
    return F.load_preds(name)


def compare(old, new):
    cols = ["obs", "l1", "l2", "l3"]
    a, b = np.stack([old[g][cols].values for g in ids]), np.stack([new[g][cols].values for g in ids])
    return dict(n_values=int(a.size), n_missing=int(np.isnan(a).sum()), same_missing=bool(np.array_equal(np.isnan(a), np.isnan(b))), max_abs_difference=float(np.nanmax(np.abs(a - b))), bit_for_bit=bool(np.array_equal(a, b, equal_nan=True)), allclose=bool(np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)))


out = {"networks": {}, "own_variant": {}, "derived": {}, "e1b_wqa": {}, "trajectory": {}, "made_up_values": {}}
runs = {}
for name in SETS_E2:
    arm = ARM[name]
    for seed in SEEDS:
        run, z = run_from_file(f"{PROJ}/results/models/lstm_{name}_s{seed}.pt"); runs[name, seed] = (run, z)
        new = frames(run, TAB[arm], arm)
        c = compare(stored(f"lstm_{name}_s{seed}"), new); c.update(best_update=int(z["best_update"]), updates_run=int(z["updates_run"]), best_val_nse=float(z["best_val_nse"]), input_set=z["input_set"], arm=z["arm"], script=z["script"])
        out["networks"][f"{name}_s{seed}"] = c
    print(f"{name:13s} ({ARM[name]}): " + "; ".join(f"seed {s}: max diff {out['networks'][f'{name}_s{s}']['max_abs_difference']:.1e}, bit for bit {out['networks'][f'{name}_s{s}']['bit_for_bit']}" for s in SEEDS), flush=True)
n_bit = sum(c["bit_for_bit"] for c in out["networks"].values())
print(f"1. {n_bit} of {len(out['networks'])} saved networks reproduce their stored forecasts bit for bit from my tables; largest difference {max(c['max_abs_difference'] for c in out['networks'].values()):.1e}; all allclose: {all(c['allclose'] for c in out['networks'].values())}")
# variant 'own'
for name in ("wqaF_sub", "wqaF_clim", "wqaP", "wqaF_s", "wqa_s"):
    arm = ARM[name]; run, z = runs[name, 0]
    c = compare(stored(f"lstm_{name}_s0"), frames(run, TABO[arm], arm)); out["own_variant"][f"{name}_s0"] = c
    print(f"   variant 'own' (my own totals x my own factors, six decimals): {name} seed 0: max difference {c['max_abs_difference']:.2e}, bit for bit {c['bit_for_bit']}, allclose {c['allclose']}", flush=True)
# derived sets
def variant_table(kind):
    t = {}
    for g in ids:
        m = TAB["main"][g].copy()
        if kind == "rawpred":
            for k in KS: m[f"fs{k}"] = m[f"fr{k}"].values
        elif kind == "perfectpred":
            for k in KS: m[f"fs{k}"] = m[f"p{k}"].values
        elif kind == "forecastpred":
            for k in KS: m[f"p{k}"] = m[f"fs{k}"].values
        t[g] = m
    return t
for dname, src, kind in (("wqaF_sub_rawpred", "wqaF_sub", "rawpred"), ("wqaF_sub_perfectpred", "wqaF_sub", "perfectpred"), ("wqaP_forecastpred", "wqaP", "forecastpred")):
    tab = variant_table(kind)
    for seed in SEEDS:
        run, z = runs[src, seed]
        c = compare(stored(f"lstm_{dname}_s{seed}"), frames(run, tab, "main")); out["derived"][f"{dname}_s{seed}"] = c
    print(f"2. {dname}: " + "; ".join(f"seed {s}: max diff {out['derived'][f'{dname}_s{s}']['max_abs_difference']:.1e}, bit for bit {out['derived'][f'{dname}_s{s}']['bit_for_bit']}" for s in SEEDS), flush=True)
# E1b's wqa networks
for seed in SEEDS:
    run, z = run_from_file(f"{PROJ}/results/models/lstm_wqa_s{seed}.pt")
    new = run.predict(TAB["main"], z["ids"], B)
    out["e1b_wqa"][f"s{seed}"] = dict(compare(stored(f"lstm_wqa_s{seed}"), new), script=z["script"], dyn=list(z["dyn"]))
print("3. E1b's saved wqa networks given my main table: " + "; ".join(f"seed {s}: max diff {out['e1b_wqa'][f's{s}']['max_abs_difference']:.1e}, bit for bit {out['e1b_wqa'][f's{s}']['bit_for_bit']}" for s in SEEDS), flush=True)
# the shared trajectory
for seed in SEEDS:
    zs = {n: runs[n, seed][1] for n in ("wqaF_sub", "wqaF_clim", "wqaP")}
    m = min(len(z["history"]) for z in zs.values())
    tl = {n: [h["train_loss"] for h in z["history"][:m]] for n, z in zs.items()}
    same_loss = all(abs(tl["wqaF_sub"][i] - tl[n][i]) < 1e-9 for n in ("wqaF_clim", "wqaP") for i in range(m))
    same_updates = {n: int(z["best_update"]) for n, z in zs.items()}
    # weights: equal when the best update is equal
    sd = {n: z["state_dict"] for n, z in zs.items()}
    def same_w(a, b): return all(torch.equal(sd[a][k], sd[b][k]) for k in sd[a])
    # compare the weights tensors that do not depend on the number of columns: all have the same shapes
    val = {n: [h["val_nse"] for h in z["history"][:m]] for n, z in zs.items()}
    out["trajectory"][f"seed{seed}"] = dict(n_common_checks=m, training_loss_identical_at_all_common_checks=bool(same_loss), best_update=same_updates, updates_run={n: int(z["updates_run"]) for n, z in zs.items()},
                                           weights_identical=dict(sub_vs_clim=same_w("wqaF_sub", "wqaF_clim"), sub_vs_perfect=same_w("wqaF_sub", "wqaP"), clim_vs_perfect=same_w("wqaF_clim", "wqaP")),
                                           validation_nse_at_the_checks={n: [round(x, 4) for x in v[:8]] for n, v in val.items()}, first_checks_equal_validation=False)
    print(f"4. seed {seed}: training loss identical at {m} common checks: {same_loss}; best updates {same_updates}; weights identical sub/clim {out['trajectory'][f'seed{seed}']['weights_identical']['sub_vs_clim']}, sub/perfect {out['trajectory'][f'seed{seed}']['weights_identical']['sub_vs_perfect']}")
# the made-up-values test
rng = np.random.default_rng(11)
for name in ("wqaF_sub", "wqaF_clim", "wqaP", "wqaF_s", "wqa_s"):
    arm = ARM[name]; run, z = runs[name, 0]
    idx = TAB[arm][ids[0]].index; a, b = SPLITS[arm]["test"]
    cand = np.flatnonzero((idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)))[200:-200]
    cuts = sorted(int(c) for c in rng.choice(cand, 3, replace=False))
    base = frames(run, TAB[arm], arm); cols = sorted(set(run.cfg.dyn) | {"q_mm"}); ok_same, ok_moved = True, True
    for c in cuts:
        alt = {g: t.copy() for g, t in TAB[arm].items()}
        for g in ids:
            for col in cols:
                x = alt[g][col].values.copy(); x[c + 1:] = rng.uniform(0.0, 120.0, len(x) - c - 1); alt[g][col] = x
        new = frames(run, alt, arm)
        for g in ids:
            for k in KS:
                ok_same &= bool(np.array_equal(base[g][f"l{k}"].values[:c + k + 1], new[g][f"l{k}"].values[:c + k + 1], equal_nan=True))
        ok_moved &= bool(any(not np.allclose(base[g]["l1"].values[c + 2:c + 12], new[g]["l1"].values[c + 2:c + 12], rtol=0, atol=1e-9) for g in ids))
    out["made_up_values"][name] = dict(cut_days=[str(idx[c].date()) for c in cuts], columns_replaced=cols, forecasts_issued_on_or_before_the_cut_unchanged=ok_same, later_forecasts_changed=ok_moved)
    print(f"5. made-up values (every input column and the flow after a cut day, 3 cuts in the test years): {name}: forecasts issued on or before the cut unchanged: {ok_same}; later ones changed: {ok_moved}", flush=True)
out["seconds"] = round(time.time() - t0, 1)
jsave("f7_networks.json", out)
print(f"done in {time.time() - t0:.0f} s")
