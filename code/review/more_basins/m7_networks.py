"""Checks (c), (d) and (f): the 15 saved networks of the script (results/models/{lstm63_wq,lstm63_w,lstm63eq_wq,unseen50_w_static,unseen50_w_nostatic}_s{0,1,2}.pt).

For every network, loaded with torch.load(weights_only=True):
 (c1) the training rivers it records: 63 (the 12 first, then the audit's 51) or 50 (no one of the 12), the split, the settings against results/lstm_all.json (x5 for lstm63eq_wq);
 (c2) its scalers (weather, flow, the ten properties) recomputed by the reviewer from the TRAINING YEARS of exactly those rivers (own code on the project's tables): must equal the stored ones;
      the same scalers computed from other choices (all days; the 50 plus the 12; the 12 alone) must differ - so the stored ones are from the training years of the training rivers;
 (c3) its early-stopping record replayed on the stored validation curve (best update, updates run, best score), and the saved weights scored on the validation years of the TRAINING rivers
      (the trainer's own score: mean NSE over rivers and the three leads): must equal the stored best_val_nse; for the never-seen networks the score of the 12 is shown to be another number;
 (c4) the stored property values of the training rivers and of the 12 against the reviewer's own join of the CAMELS attribute tables; the loss weights the trainer uses (formula from flood.fit, computed here);
 (f)  the stored forecasts reproduced bit for bit from the file alone by (A) flood.LSTMNet + flood.Run.predict with the stored cfg, scalers, ids and the properties of the 12 on the 12-basin table,
      and (B) the reviewer's own input matrix (own scaling) and own block-rolling over the same network; the 63-basin table gives the same forecasts; a negative control (properties scaled with
      the 12's own mean and spread instead of the training rivers') changes them.
Run: python m7_networks.py        (about 3-6 minutes: 15 networks, one thread; nothing is trained)
"""
import time

import torch

from m_common import *

F = engine()
t_start = time.time()
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
cfg04 = json.loads((PROJ / "results" / "lstm_all.json").read_text())["config"]
ids12 = F.study_ids()
ids63 = J["design"]["rivers"]["sixty_three"]
ids50 = J["design"]["rivers"]["other_50"]
D63 = F.load_data(ids=ids63)
D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
A = attribute_tables()
Sown = static_table(ids63, A)
idx = D63[ids63[0]].index
tr_mask = np.asarray((idx >= pd.Timestamp(SPLIT["train"][0])) & (idx <= pd.Timestamp(SPLIT["train"][1])))
val_mask = np.asarray((idx >= pd.Timestamp(SPLIT["val"][0])) & (idx <= pd.Timestamp(SPLIT["val"][1])))
SETS = {"lstm63_wq": dict(train=ids63, flow=True, static=True, conf=cfg04), "lstm63_w": dict(train=ids63, flow=False, static=True, conf=cfg04),
        "lstm63eq_wq": dict(train=ids63, flow=True, static=True, conf=dict(cfg04, updates=5 * cfg04["updates"], eval_every=5 * cfg04["eval_every"])),
        "unseen50_w_static": dict(train=ids50, flow=False, static=True, conf=cfg04), "unseen50_w_nostatic": dict(train=ids50, flow=False, static=False, conf=cfg04)}
SEEDS = [0, 1, 2]


def own_scalers(ids, D, S, part_mask=tr_mask):
    dyn = np.concatenate([D[g].loc[part_mask, FORCING].values for g in ids])
    q = np.concatenate([D[g].q_mm.values[part_mask] for g in ids])
    sc = dict(dyn_mu=np.nanmean(dyn, 0), dyn_sd=np.nanstd(dyn, 0) + 1e-9, q_mu=float(q.mean()), q_sd=float(q.std()), lq_mu=float(np.log1p(q).mean()), lq_sd=float(np.log1p(q).std()))
    St = S.loc[ids].values
    sc["st_mu"], sc["st_sd"] = St.mean(0), St.std(0) + 1e-9
    return sc


def max_rel(a, b):
    a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def own_design(Dd, ids, sc, use_flow, static_vals):
    """The reviewer's input matrix [basin, day, feature]: six scaled weather columns, then (gauge sets) the scaled log1p flow, then (property sets) the ten scaled properties repeated every day."""
    X = []
    for g in ids:
        d = Dd[g]
        parts = [(d[FORCING].values - np.asarray(sc["dyn_mu"])) / np.asarray(sc["dyn_sd"])]
        if use_flow:
            parts.append(((np.log1p(d.q_mm.values) - sc["lq_mu"]) / sc["lq_sd"])[:, None])
        if static_vals is not None:
            s = (np.asarray(static_vals[g], float) - np.asarray(sc["st_mu"])) / np.asarray(sc["st_sd"])
            parts.append(np.repeat(s[None], len(d), 0))
        X.append(np.hstack(parts))
    return torch.tensor(np.stack(X), dtype=torch.float32)


def own_roll(net, X, warmup, block):
    """Forecast for every issue day from `warmup` on, in blocks of `block` days after `warmup` days of reading (the engine's convention, written again)."""
    T = X.shape[1]
    out = torch.full((X.shape[0], T, 3), float("nan"))
    net.eval()
    with torch.no_grad():
        for c in range(warmup, T, block):
            out[:, c:c + block] = net(X[:, c - warmup:c + block])[:, warmup:]
    return out


def own_target_day(out, sc):
    o = np.clip(out.numpy().astype("float64") * sc["q_sd"] + sc["q_mu"], 0, None)
    p = np.full_like(o, np.nan)
    for j, k in enumerate(LEADS):
        p[:, k:, j] = o[:, :-k, j]
    return p


def val_score(net, D, ids, sc, cfg, static_df):
    """The trainer's validation score of a network: mean over rivers and the three leads of NSE on the validation target days (engine design/roll, own NSE)."""
    X, _ = F._design(D, ids, cfg, sc, static_df)
    p = F._to_target_day(F._roll(net, X, cfg.warmup, cfg.block, F._first_valid(X)), sc)
    v = np.array([[own_nse(D[g].q_mm.values[val_mask], p[i, val_mask, j]) for j in range(3)] for i, g in enumerate(ids)])
    return float(v.mean()), v


res = dict(networks={})
all_ok = True
for name, spec in SETS.items():
    for s in SEEDS:
        t0 = time.time()
        f = PROJ / "results" / "models" / f"{name}_s{s}.pt"
        z = torch.load(f, map_location="cpu", weights_only=True)
        cfg = F.Cfg(**z["cfg"])
        r = dict(file=str(f.relative_to(PROJ)), sha256=sha256(f), bytes=f.stat().st_size, keys=sorted(z.keys()))
        # (c1) the training rivers and the settings
        train = list(z["ids"])
        r["c1"] = dict(n_training_rivers=len(train), training_rivers_equal_expected=train == spec["train"], contains_none_of_the_12=not (set(train) & set(ids12)) if name.startswith("unseen") else None,
                       contains_all_12=set(ids12) <= set(train), twelve_first=train[:12] == ids12 if name.startswith("lstm") else None, scored_ids_are_the_12=list(z["scored_ids"]) == ids12,
                       split_equals_project_split=z["split"] == {k: list(v) for k, v in SPLIT.items()} or z["split"] == SPLIT, set_name=z["set"], script=z["script"], seed=z["seed"], quick_mode=z["quick_mode"],
                       cfg_use_flow=cfg.use_flow == spec["flow"], cfg_static_is_STATIC=(list(cfg.static) == STATIC) if spec["static"] else (len(cfg.static) == 0), cfg_future=cfg.future,
                       cfg_seed=cfg.seed == s, settings_equal_expected={k: getattr(cfg, k) for k in spec["conf"]} == spec["conf"],
                       static_values_train_keys_equal_training=(list(z["static_values_train"]) == train) if spec["static"] else (z["static_values_train"] == {}),
                       static_values_scored_keys_are_the_12=(list(z["static_values_scored"]) == ids12) if spec["static"] else (z["static_values_scored"] == {}),
                       n_params=int(z["n_params"]) == sum(v.numel() for v in z["state_dict"].values()), torch_version=z["torch_version"], numpy_version=z["numpy_version"], torch_threads_recorded=z["torch_threads"])
        # (c2) scalers recomputed from the training years of the training rivers
        sc_own = own_scalers(train, D63, Sown)
        keys = ["dyn_mu", "dyn_sd", "q_mu", "q_sd", "lq_mu", "lq_sd"] + (["st_mu", "st_sd"] if spec["static"] else [])
        sc_diff = {k: max_rel(z["sc"][k], sc_own[k]) for k in keys}
        sc_bitwise = {k: bool(np.array_equal(np.asarray(z["sc"][k], float), np.asarray(sc_own[k], float))) for k in keys}
        alt = {"all_days_instead_of_training_years": own_scalers(train, D63, Sown, np.ones(len(idx), bool)),
               "training_rivers_plus_the_12": own_scalers(sorted(set(train) | set(ids12), key=ids63.index), D63, Sown), "the_12_alone": own_scalers(ids12, D63, Sown)}
        r["c2"] = dict(keys=keys, max_relative_difference=sc_diff, all_bitwise_equal=all(sc_bitwise.values()), max_rel_all=max(sc_diff.values()),
                       alternatives_differ_by=dict({a: {k: max_rel(z["sc"][k], v[k]) for k in keys} for a, v in alt.items()}))
        r["c2"]["alt_max"] = {a: max(v.values()) for a, v in r["c2"]["alternatives_differ_by"].items()}
        if name.startswith("lstm63"):
            r["c2"]["alt_max"].pop("training_rivers_plus_the_12")        # the 12 are inside the 63: that alternative is the stored one
        # (c4) property values
        if spec["static"]:
            r["c4"] = dict(train_values_max_abs_diff=max(float(np.max(np.abs(np.asarray(z["static_values_train"][g]) - Sown.loc[g].values))) for g in train),
                           scored_values_max_abs_diff=max(float(np.max(np.abs(np.asarray(z["static_values_scored"][g]) - Sown.loc[g].values))) for g in ids12))
        # loss weights the trainer used: 1 / (basin spread of the training years / pooled spread + 0.1)^2, from the training rivers' own training years
        w = np.array([1 / (D63[g].q_mm.values[tr_mask].std() / sc_own["q_sd"] + 0.1) ** 2 for g in train])
        r["c4w"] = dict(weights_min=float(w.min()), weights_max=float(w.max()), formula="1 / (std of the basin's training-year flow / q_sd of the pooled training years + 0.1)^2 (flood.fit)")
        # network
        n_in = len(cfg.dyn) + int(cfg.use_flow) + len(cfg.static)
        net = F.LSTMNet(n_in, cfg)
        net.load_state_dict(z["state_dict"])
        net.eval()
        sc_arr = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in z["sc"].items()}
        run = F.Run(cfg, sc_arr, net, z["ids"], z["history"], z["seconds"], z["cpu_seconds"], z["updates_run"], z["best_update"], z["best_val_nse"], z["n_params"], z["split"])
        # (c3) replay of early stopping on the stored curve
        best, best_u, bad, stop_at = -np.inf, 0, 0, None
        for h in z["history"]:
            if h["val_nse"] > best:
                best, best_u, bad = h["val_nse"], h["update"], 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    stop_at = h["update"]
                    break
        last = z["history"][-1]["update"]
        expected_run = stop_at if stop_at is not None else min(cfg.updates, last)
        r["c3_replay"] = dict(replayed_best_update=best_u, stored_best_update=z["best_update"], replayed_best_val=best, stored_best_val=z["best_val_nse"], replayed_updates_run=expected_run, stored_updates_run=z["updates_run"],
                              stopped_by_patience=stop_at is not None, reached_update_limit=bool(z["updates_run"] >= cfg.updates), checks_made=len(z["history"]), eval_every=cfg.eval_every, patience=cfg.patience,
                              consistent=bool(best_u == z["best_update"] and best == z["best_val_nse"] and expected_run == z["updates_run"]), updates_after_best=z["updates_run"] - z["best_update"])
        # (c3) the saved weights scored on the validation years of the training rivers
        sdf_tr = Sown.loc[train] if spec["static"] else None
        v_train, vv = val_score(net, D63, train, sc_arr, cfg, sdf_tr)
        v12, _ = val_score(net, D63, ids12, sc_arr, cfg, Sown.loc[ids12] if spec["static"] else None)
        r["c3_val"] = dict(val_nse_of_the_saved_weights_on_the_training_rivers=v_train, stored_best_val_nse=z["best_val_nse"], abs_diff=abs(v_train - z["best_val_nse"]), n_rivers_scored=len(train),
                           val_nse_of_the_saved_weights_on_the_12=v12, json_validation_nse_of_the_12=[t for t in J["training"] if t["set"] == name and t["seed"] == s][0]["validation_nse_of_the_12_at_that_update"],
                           the_12_differ_from_the_training_score=abs(v12 - z["best_val_nse"]) > 1e-3)
        # (f) reproduction of the stored forecasts
        zp = load_npz(f"{name}_s{s}")
        order = [zp["basins"].index(g) for g in ids12]
        stored = zp["pred"][order]                                  # [12, day, 3]
        sdf12 = pd.DataFrame(z["static_values_scored"], index=list(cfg.static)).T if spec["static"] else None
        newA = run.predict(D12, ids12, sdf12)                       # route A: engine
        A_arr = np.stack([newA[g][["l1", "l2", "l3"]].values for g in ids12])
        newA63 = run.predict(D63, ids12, sdf12)                     # route A on the 63-basin table
        A63_arr = np.stack([newA63[g][["l1", "l2", "l3"]].values for g in ids12])
        stat_vals = {g: z["static_values_scored"][g] for g in ids12} if spec["static"] else None
        Xb = own_design(D12, ids12, sc_arr, cfg.use_flow, stat_vals)
        B_arr = own_target_day(own_roll(net, Xb, cfg.warmup, cfg.block), sc_arr)
        # negative control: properties scaled with the 12's own mean and spread
        neg = None
        if spec["static"]:
            S12 = Sown.loc[ids12].values
            sc_bad = dict(sc_arr, st_mu=S12.mean(0), st_sd=S12.std(0) + 1e-9)
            C_arr = own_target_day(own_roll(net, own_design(D12, ids12, sc_bad, cfg.use_flow, stat_vals), cfg.warmup, cfg.block), sc_bad)
            neg = float(np.nanmax(np.abs(C_arr - stored)))
        eq = lambda x: bool(np.array_equal(x, stored, equal_nan=True))
        r["f"] = dict(engine_bitwise=eq(A_arr), engine_max_abs_diff=float(np.nanmax(np.abs(A_arr - stored))), engine_on_63_table_bitwise=eq(A63_arr), own_input_bitwise=eq(B_arr),
                      own_input_max_abs_diff=float(np.nanmax(np.abs(B_arr - stored))), same_nan_pattern=bool(np.array_equal(np.isnan(B_arr), np.isnan(stored))), n_values=int(stored.size),
                      negative_control_12_scaled_properties_max_abs_diff=neg)
        ok = (r["c1"]["training_rivers_equal_expected"] and r["c1"]["settings_equal_expected"] and r["c2"]["max_rel_all"] < 1e-12 and r["c3_replay"]["consistent"]
              and r["c3_val"]["abs_diff"] < 1e-5 and r["f"]["engine_bitwise"] and r["f"]["own_input_bitwise"] and r["f"]["engine_on_63_table_bitwise"])
        r["ok"] = bool(ok)
        all_ok &= ok
        res["networks"][f"{name}_s{s}"] = r
        print(f"{name}_s{s}: train {len(train)} rivers (as expected {r['c1']['training_rivers_equal_expected']}), settings {r['c1']['settings_equal_expected']} | scalers: max rel diff {r['c2']['max_rel_all']:.1e}, bitwise {r['c2']['all_bitwise_equal']}, "
              f"alternatives differ by {', '.join(f'{k} {v:.2g}' for k, v in r['c2']['alt_max'].items())} | early stop replay consistent {r['c3_replay']['consistent']} (best {z['best_update']} of {z['updates_run']}, "
              f"patience stop {r['c3_replay']['stopped_by_patience']}, limit {r['c3_replay']['reached_update_limit']}) | val NSE of saved weights on the {len(train)} training rivers {v_train:.6f} vs stored {z['best_val_nse']:.6f}; on the 12: {v12:.4f} | "
              f"forecasts: engine bitwise {r['f']['engine_bitwise']}, 63-table bitwise {r['f']['engine_on_63_table_bitwise']}, own input bitwise {r['f']['own_input_bitwise']} (neg. control {neg if neg is None else round(neg, 3)}) [{time.time() - t0:.0f} s]", flush=True)
res["all_ok"] = bool(all_ok)
res["seconds"] = round(time.time() - t_start, 1)
print("\nALL 15 NETWORKS OK:", all_ok, f"({res['seconds']} s)")
save("m7_networks", res)
sys.exit(0 if all_ok else 1)
