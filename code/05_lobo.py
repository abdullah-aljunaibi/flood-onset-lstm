"""Week 3, second half - leave one basin out: can the model forecast a river it has never seen (an 'ungauged' basin)?

Each of the 12 basins is held out in turn: an LSTM is trained on the other 11 (scalers, loss weights and the stopping point
come from those 11 only) and then forecasts the held-out basin from its past weather alone - an ungauged basin has no river
gauge. Two variants, one seed each: with the ten basin properties as inputs, and with weather only (the model cannot tell
basins apart). A random forest fitted on the pooled days of the other 11 basins is the like-for-like simple method.
The held-out scores are set against the same basin's score when it WAS in training (lstm_w_s0 of 04_lstm_all.py).
Warnings follow the rule of ungauged systems: warn when the forecast exceeds the 95th percentile of the model's OWN forecasts
for that basin in the training years (no flow observation is needed for that level).
Writes results/lstm_lobo.json, figures/lobo.png and the prediction sets lobo_w_static, lobo_w_nostatic, lobo_rf_w
(plus lobo_ref_w_nostatic: one model for all 12 basins WITHOUT basin properties, the in-training partner of the second variant).
Without training again: python code/05_lobo.py --rescore  (all scores from the saved prediction sets);  --figure  (redraw only).
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestRegressor

import flood as F
from common import *
from features import feature_table

use_style()
SCRIPT, SEED, REF, REF_NOSTATIC = "05_lobo.py", 0, "lstm_w_s0", "lobo_ref_w_nostatic"
CAP = dict(updates=1000, patience=4)                                             # budget cap for the 25 trainings (04 used 2000 / 6)
FOREST = dict(n_estimators=100, max_samples=20000, min_samples_leaf=10, max_features=1 / 3, n_jobs=4, random_state=0)
if F.QUICK:
    CAP, FOREST = dict(updates=4, eval_every=4, patience=4), dict(FOREST, n_estimators=8, max_samples=4000, n_jobs=1)
VARIANTS = {"lobo_w_static": dict(static=tuple(F.STATIC), label="held out, weather + basin properties"),
            "lobo_w_nostatic": dict(static=(), label="held out, weather only (no basin properties)")}
HELD = {"lobo_w_static": "held_out_with_properties", "lobo_w_nostatic": "held_out_without_properties", "lobo_rf_w": "forest_held_out"}
FLOW_SIGNS = ["q_mean", "runoff_ratio", "baseflow_index", "high_q_freq"]         # from the flow record: never a model input, only used to describe a basin
lk = lambda k: f"l{k}"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"


def settings():
    """The setting chosen by 04_lstm_all.py (always its real run), with the training budget capped for 25 trainings."""
    src = json.loads((RES / "lstm_all.json").read_text())
    config = dict(src["config"], **{k: min(src["config"].get(k, v), v) for k, v in CAP.items()})
    return src, config


def replay(curve, config):
    """Which check of a recorded validation curve the early-stopping rule of `config` would have kept."""
    best, best_u, bad = -np.inf, 0, 0
    for h in curve:
        if h["update"] > config["updates"]:
            break
        if h["val_nse"] > best:
            best, best_u, bad = h["val_nse"], h["update"], 0
        else:
            bad += 1
            if bad >= config["patience"]:
                break
    return best_u


def paired(a, b):
    """Two per-basin score arrays (first minus second): in how many basins the first is higher, the median difference, Wilcoxon p."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:                                                           # all differences zero
        p = float("nan")
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=int(len(d)), median_nse_difference=float(np.median(d)), wilcoxon_p=p)


def unlike_the_rest(B, ids):
    """How far each basin's properties lie from the other 11, in the units the held-out model saw (mean and spread of those 11)."""
    S, out = B.loc[ids, F.STATIC].astype(float), {}
    for g in ids:
        o = S.drop(g)
        mu, sd = o.mean(), o.std(ddof=0)
        z = (S.loc[g] - mu) / sd
        dist = np.sqrt(((((o - mu) / sd) - z) ** 2).sum(axis=1))
        out[g] = dict(z_against_other_11={c: round(float(z[c]), 2) for c in F.STATIC}, largest_abs_z=float(z.abs().max()), largest_abs_z_property=str(z.abs().idxmax()),
                      outside_range_of_other_11=[c for c in F.STATIC if S.loc[g, c] > o[c].max() or S.loc[g, c] < o[c].min()],
                      nearest_basin=F.SHORT[dist.idxmin()], nearest_distance=float(dist.min()))
    return out


def own_levels(P):
    """Warning level without a river gauge: per basin and lead, the 95th percentile of the model's own training-years forecasts."""
    a, b = F.SPLIT["train"]
    return {g: {k: float(p[lk(k)].loc[a:b].quantile(F.HIGH_Q)) for k in F.LEADS} for g, p in P.items()}


def warnings_own_level(P, thr):
    """Warning tables on the test years: warn when the forecast is above the model's own level; a day is high when the observed flow is above thr."""
    lev, out = own_levels(P), {"basins": {}, "summary": {}}
    for g, p in P.items():
        assert thr[g] > 0 and all(lev[g][k] > 0 for k in F.LEADS), f"{g}: a level of zero cannot define a warning"
        out["basins"][g] = {lk(k): dict(model_level_mm=lev[g][k], observed_level_mm=thr[g], model_level_over_observed=lev[g][k] / thr[g],
                                        **F.flood_scores(p, thr[g], k, "test", trigger=lev[g][k] / thr[g])) for k in F.LEADS}
    for k in F.LEADS:
        pool = lambda key: F.rates({c: int(sum(out["basins"][g][lk(k)][key][c] for g in P)) for c in ("hits", "misses", "false_alarms", "correct_neg")})
        out["summary"][lk(k)] = dict(all=pool("all"), onset=pool("onset"))
    return out


def train_all(D, B, ids, config):
    """Under the CPU lock: the reference without basin properties, 12 x 2 held-out LSTMs, 12 x 3 held-out forests. Saves the prediction sets."""
    n_total, done, t_ask = 1 + len(ids) * len(VARIANTS), [0], time.perf_counter()
    PS, cost = {n: {} for n in list(VARIANTS) + ["lobo_rf_w"]}, {n: {} for n in list(VARIANTS) + ["lobo_rf_w"]}

    def train(static, train_ids, target, what):
        run = F.fit(F.Cfg(use_flow=False, future=False, static=static, seed=SEED, **config), D, train_ids, static_df=B)
        assert run.split == F.SPLIT and set(run.ids) == set(train_ids)
        P = run.predict(D, ids=target, static_df=B)
        ev = F.evaluate(P, F.thresholds({g: D[g] for g in target}), "test")["summary"]
        done[0] += 1
        print(f"training {done[0]:2d} of {n_total} | {what} | best at update {run.best_update} of {run.updates_run}, {run.seconds:.0f} s | validation NSE of the "
              f"{len(train_ids)} training basins {run.best_val_nse:.4f} | test NSE by lead " + " ".join(f"{ev[lk(k)]['nse_median']:.3f}" for k in F.LEADS)
              + f" | {time.perf_counter() - t1:.0f} s so far", flush=True)
        return P, dict(run.summary(), curve=[[h["update"], round(h["val_nse"], 4)] for h in run.history])

    with F.cpu_lock(SCRIPT):
        t1 = time.perf_counter()
        PS[REF_NOSTATIC], cost[REF_NOSTATIC] = train((), ids, ids, "reference: all 12 basins in training, weather only (no basin properties)")
        for g in ids:
            others = [b for b in ids if b != g]
            for name, v in VARIANTS.items():
                P, cost[name][g] = train(v["static"], others, [g], f"{F.SHORT[g]} {v['label']}")
                PS[name][g] = P[g]
        t2 = time.perf_counter()
        # the simple method under the same protocol: one forest per held-out basin and lead, fitted on the pooled issue days of the other 11
        X = {g: np.hstack([feature_table(D[g], flow=False).values, np.tile(B.loc[g, F.STATIC].values.astype(float), (len(D[g]), 1))]) for g in ids}
        ok = {g: np.isfinite(X[g]).all(axis=1) for g in ids}
        for g in ids:
            others, out, t3, rows = [b for b in ids if b != g], {}, time.perf_counter(), 0
            for k in F.LEADS:
                fit = {b: ok[b] & F.part_mask(D[b].index, k, "train") for b in others}             # target day in the training years
                Xt, yt = np.vstack([X[b][fit[b]] for b in others]), np.concatenate([D[b].q_mm.shift(-k).values[fit[b]] for b in others])
                m = RandomForestRegressor(**FOREST).fit(Xt, yt)
                out[k], rows = np.full(len(D[g]), np.nan), len(yt)
                out[k][ok[g]] = np.clip(m.predict(X[g][ok[g]]), 0, None)
            PS["lobo_rf_w"][g], cost["lobo_rf_w"][g] = F.frame_from_issue(D[g], out), dict(seconds=round(time.perf_counter() - t3, 1), pooled_training_days=rows)
            print(f"forest {ids.index(g) + 1:2d} of {len(ids)} | {F.SHORT[g]} held out | {rows} pooled training days | {cost['lobo_rf_w'][g]['seconds']:.0f} s", flush=True)
        timing = dict(n_trainings=done[0], training_seconds=round(t2 - t1, 1), forest_seconds=round(time.perf_counter() - t2, 1), waited_for_cpu_s=round(t1 - t_ask, 1))
    what = {REF_NOSTATIC: "one LSTM for all 12 basins, past weather only, NO basin properties (in-training reference)", "lobo_rf_w": "random forest, each basin forecast by a forest fitted on the other 11",
            **{n: f"LSTM, each basin forecast by a model trained on the other 11; {v['label']}" for n, v in VARIANTS.items()}}
    for n, P in PS.items():
        F.save_preds(n, P, dict(script=SCRIPT, model="rf" if n == "lobo_rf_w" else "lstm", inputs="w", seeds=[SEED], what=what[n], cost=cost[n], timing=timing,
                                **(dict(settings={k: v for k, v in FOREST.items() if k != "n_jobs"}) if n == "lobo_rf_w" else dict(config=config))))
    return PS, cost, timing


def run(rescore=False):
    t0 = time.perf_counter()
    D, B = F.load_data(), F.basins()
    ids, thr = list(D), F.thresholds(D)
    src, config = settings()
    names = list(VARIANTS) + ["lobo_rf_w"]
    if rescore:                                                                  # forecasts and training records come from the saved prediction sets
        PS = {n: F.load_preds(n) for n in names + [REF_NOSTATIC]}
        meta = {n: F.preds_meta(n) for n in PS}
        assert all(meta[n]["script"] == SCRIPT and meta[n].get("config", config) == config for n in PS), "saved lobo sets were made with other settings"
        cost, timing = {n: meta[n]["cost"] for n in PS}, meta[names[0]]["timing"]
    else:
        PS, cost, timing = train_all(D, B, ids, config)
    assert all(set(PS[n]) == set(ids) for n in PS)
    notes = []

    # ---- the in-training partners: lstm_w_s0 (same seed and inputs, with basin properties) and its other seeds as a yardstick ----
    PS[REF] = F.load_preds(REF)
    ref_meta = F.preds_meta(REF)
    assert ref_meta["inputs"] == "w" and ref_meta["seeds"] == [SEED], f"{REF} is not the past-weather model with seed {SEED}"
    ref_cost, same = src["sets"]["w"]["cost"][str(SEED)], None
    if not F.QUICK:
        assert ref_meta["config"] == src["config"]
        same = replay(src["sets"]["w"]["per_seed"][str(SEED)]["curve"], config) == ref_cost["best_update"]
    seed_sets = [n for n in ("lstm_w_s0", "lstm_w_s1", "lstm_w_s2") if F.have_preds(n)]
    seed_ev = [F.evaluate(F.load_preds(n), thr, "test")["basins"] for n in seed_sets]
    if F.have_preds("base_rf_w"):
        PS["base_rf_w"] = F.load_preds("base_rf_w")
    for n in (REF, "base_rf_w"):
        assert n not in PS or all(np.allclose(PS[n][g].obs.values, D[g].q_mm.values) for g in ids), f"{n}: observed flow differs"

    # ---- scores on the test years of each basin (held out, or in training for the references) ----------------------------
    EV = {n: dict(test=F.evaluate(P, thr, "test"), val=F.evaluate(P, thr, "val")["summary"], warn=warnings_own_level(P, thr)) for n, P in PS.items()}
    for n, e in EV.items():
        assert all(s[lk(k)]["coverage_min"] == 1 for s in (e["test"]["summary"], e["val"]) for k in F.LEADS), f"{n}: days without a forecast"
    nse = lambda n, g, k: EV[n]["test"]["basins"][g][lk(k)]["nse"]
    rows = []
    for g in ids:
        for k in F.LEADS:
            sv = [e[g][lk(k)]["nse"] for e in seed_ev]
            r = dict(gauge_id=g, basin=F.SHORT[g], landscape=B.landscape[g], lead=k, in_training=nse(REF, g, k), in_training_seed_min=min(sv), in_training_seed_max=max(sv),
                     in_training_no_properties=nse(REF_NOSTATIC, g, k), **{c: nse(n, g, k) for n, c in HELD.items()},
                     forest_in_training=nse("base_rf_w", g, k) if "base_rf_w" in EV else float("nan"))
            r.update(loss_with_properties=r["in_training"] - r["held_out_with_properties"], loss_without_properties=r["in_training_no_properties"] - r["held_out_without_properties"],
                     loss_forest=r["forest_in_training"] - r["forest_held_out"], properties_gain_held_out=r["held_out_with_properties"] - r["held_out_without_properties"],
                     properties_gain_in_training=r["in_training"] - r["in_training_no_properties"])
            rows.append(r)
    T = pd.DataFrame(rows)
    cols = ["in_training", "in_training_no_properties", "held_out_with_properties", "held_out_without_properties", "forest_held_out", "forest_in_training"]

    def verdict(d, yard):
        m = float(np.median(d))
        return "help" if m > yard else "hurt" if m < -yard else "no clear difference"

    summary = {}
    for k in F.LEADS:
        t = T[T.lead == k]
        yard = float((t.in_training_seed_max - t.in_training_seed_min).median())      # how far three seeds of the SAME model lie apart in a typical basin
        ph, pi = [dict(paired(t[a], t[b]), n_basins_lower_with_properties=int((t[a] < t[b]).sum()), mean_nse_difference=float((t[a] - t[b]).mean()),
                       difference_of_medians=float(t[a].median() - t[b].median()), verdict=verdict(t[a] - t[b], yard))
                  for a, b in (("held_out_with_properties", "held_out_without_properties"), ("in_training", "in_training_no_properties"))]
        summary[lk(k)] = dict(
            nse_median={c: float(t[c].median()) for c in cols}, nse_min={c: float(t[c].min()) for c in cols}, n_basins=len(t),
            n_basins_positive_nse={c: int((t[c] > 0).sum()) for c in cols},
            kge_median={n: EV[n]["test"]["summary"][lk(k)]["kge_median"] for n in EV},
            in_training_seed_range_median=yard,
            loss_from_not_seeing_the_basin={v: dict(median_nse_loss=float(t[c].median()), mean_nse_loss=float(t[c].mean()), largest_nse_loss=float(t[c].max()), n_basins_lower_when_held_out=int((t[c] > 0).sum()),
                                                    n_basins_loss_larger_than_seed_range=int((t[c] > yard).sum()), wilcoxon_p=paired(t[a], t[b])["wilcoxon_p"])
                                            for v, c, a, b in (("with_properties", "loss_with_properties", "in_training", "held_out_with_properties"),
                                                               ("without_properties", "loss_without_properties", "in_training_no_properties", "held_out_without_properties"),
                                                               ("forest", "loss_forest", "forest_in_training", "forest_held_out"))},
            held_out_below_every_in_training_seed={c: int((t[c] < t.in_training_seed_min).sum()) for c in ("held_out_with_properties", "held_out_without_properties")},
            basin_properties=dict(held_out=ph, in_training=pi),
            lstm_against_forest_held_out={v: paired(t[c], t.forest_held_out) for v, c in (("with_properties", "held_out_with_properties"), ("without_properties", "held_out_without_properties"))},
            warnings_model_level={n: {s: {c: EV[n]["warn"]["summary"][lk(k)][s][c] for c in ("pod", "far", "csi", "hits", "misses", "false_alarms")} for s in ("all", "onset")} for n in EV})

    # ---- which basins lose most when held out (three largest NSE losses of either variant), and what sets them apart --------
    U, t1 = unlike_the_rest(B, ids), T[T.lead == 1].set_index("gauge_id")
    worst = list(dict.fromkeys(list(t1.loss_with_properties.nlargest(3).index) + list(t1.loss_without_properties.nlargest(3).index)))
    on = lambda n, g: {c: EV[n]["warn"]["basins"][g]["l1"]["onset"][c] for c in ("pod", "far", "hits", "misses", "false_alarms")}
    pb = lambda n, g: EV[n]["test"]["basins"][g]["l1"]["pbias"]
    rank = lambda c, g: int(B.loc[ids, c].rank(ascending=False)[g])
    worst_basins = [dict(gauge_id=g, basin=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g]),
                         nse_in_training=[nse(REF, g, k) for k in F.LEADS], **{f"nse_{c}": [nse(n, g, k) for k in F.LEADS] for n, c in HELD.items()},
                         nse_in_training_no_properties=[nse(REF_NOSTATIC, g, k) for k in F.LEADS],
                         nse_loss_1_day_ahead=dict(with_properties=float(t1.loss_with_properties[g]), without_properties=float(t1.loss_without_properties[g]), forest=float(t1.loss_forest[g])),
                         onset_warnings_model_level_1_day_ahead=dict(in_training=on(REF, g), **{c: on(n, g) for n, c in HELD.items()}),
                         forecast_volume_error_pct_1_day_ahead=dict(in_training=pb(REF, g), **{c: pb(n, g) for n, c in HELD.items()}),
                         model_level_over_observed_level_1_day_ahead={c: EV[n]["warn"]["basins"][g]["l1"]["model_level_over_observed"] for n, c in HELD.items()},
                         properties_outside_range_of_other_11=U[g]["outside_range_of_other_11"], largest_abs_z=round(U[g]["largest_abs_z"], 2), largest_abs_z_property=U[g]["largest_abs_z_property"],
                         nearest_basin=U[g]["nearest_basin"], nearest_distance=round(U[g]["nearest_distance"], 2),
                         flow_record={c: float(B.loc[g, c]) for c in FLOW_SIGNS}, rank_among_12_highest_first={c: rank(c, g) for c in ("runoff_ratio", "q_mean", "slope_mean", "p_mean", "area_gages2")})
                    for g in sorted(worst, key=lambda g: -max(t1.loss_with_properties[g], t1.loss_without_properties[g]))]
    lowest = {c: [F.SHORT[g] for g in t1[c].nsmallest(3).index] for c in HELD.values()}                 # the three lowest held-out scores, 1 day ahead
    sp = lambda x: dict(zip(("rho", "p"), map(float, spearmanr(x, t1.loss_with_properties.loc[ids].values))))
    what_goes_with_the_loss = dict(what="Spearman rank correlation over the 12 basins between the NSE loss 1 day ahead (with basin properties) and a basin feature; 12 points only",
                                   nearest_distance=sp([U[g]["nearest_distance"] for g in ids]), largest_abs_z=sp([U[g]["largest_abs_z"] for g in ids]),
                                   **{c: sp(B.loc[ids, c].values.astype(float)) for c in ("runoff_ratio", "q_mean", "log_area", "slope_mean", "elev_mean")})
    by_land = {F.LAND_SHORT[land]: {c: float(grp[c].median()) for c in ("in_training", "held_out_with_properties", "held_out_without_properties", "forest_held_out", "loss_with_properties")}
               for land, grp in T[T.lead == 1].groupby("landscape")}

    # ---- the reverse case: basins that score HIGHER when held out, and how each river changed between the training and the test years ----
    years = {"training_years": "train", "validation_years": "val", "test_years": "test"}
    rain_share = {g: {p: float(D[g].q_mm.loc[slice(*F.SPLIT[q])].sum() / D[g].prcp.loc[slice(*F.SPLIT[q])].sum()) for p, q in years.items()} for g in ids}
    change = pd.Series({g: v["test_years"] / v["training_years"] for g, v in rain_share.items()})        # below 1: less of the rain reached the river in the test years
    by_years = {p: {n: EV[n]["test"]["basins"] if q == "test" else F.evaluate(PS[n], thr, q)["basins"] for n in (REF, *HELD)} for p, q in years.items()}
    per_model = lambda p, g, m: dict(in_training=by_years[p][REF][g]["l1"][m], **{c: by_years[p][n][g]["l1"][m] for n, c in HELD.items()})
    higher = [dict(gauge_id=g, basin=F.SHORT[g], nse_1_day_ahead={c: float(t1[c][g]) for c in ("in_training", "in_training_seed_min", "in_training_seed_max", "forest_in_training", *HELD.values())},
                   share_of_rain_reaching_the_river=rain_share[g], test_over_training=float(change[g]), rank_of_the_drop=int(change.rank()[g]),
                   forecast_volume_error_pct_1_day_ahead={p: per_model(p, g, "pbias") for p in years}, nse_validation_years_1_day_ahead=per_model("validation_years", g, "nse"))
              for g in ids if t1.loss_with_properties[g] < 0]
    flow_per_rain = dict(what="share of the rain that reached the river (total flow / total rain) in the training, validation and test years; from the gauge, used only to describe a basin",
                         basins={g: dict(rain_share[g], test_over_training=float(change[g])) for g in ids})

    # ---- plain-language findings, every number from the tables above ------------------------------------------------------
    findings = []
    for k in F.LEADS:
        s = summary[lk(k)]
        L, P_ = s["loss_from_not_seeing_the_basin"], s["basin_properties"]["held_out"]
        findings.append(f"{days(k).capitalize()}, never-seen basin: median test NSE {s['nse_median']['held_out_with_properties']:.3f} with basin properties and {s['nse_median']['held_out_without_properties']:.3f} with weather only, "
                        f"against {s['nse_median']['in_training']:.3f} when the basin was in training; {s['n_basins_positive_nse']['held_out_with_properties']} of {s['n_basins']} basins keep a positive NSE with properties, "
                        f"{s['n_basins_positive_nse']['held_out_without_properties']} of {s['n_basins']} with weather only, {s['n_basins_positive_nse']['forest_held_out']} of {s['n_basins']} for the random forest (median {s['nse_median']['forest_held_out']:.3f}).")
        findings.append(f"{days(k).capitalize()}, loss from not having seen the basin (median over basins): {L['with_properties']['median_nse_loss']:+.3f} NSE with basin properties ({L['with_properties']['n_basins_lower_when_held_out']} of {s['n_basins']} basins lower, "
                        f"{L['with_properties']['n_basins_loss_larger_than_seed_range']} by more than the typical seed range of {s['in_training_seed_range_median']:.3f}, {s['held_out_below_every_in_training_seed']['held_out_with_properties']} below every single in-training seed); "
                        f"{L['without_properties']['median_nse_loss']:+.3f} with weather only ({L['without_properties']['n_basins_lower_when_held_out']} lower, {L['without_properties']['n_basins_loss_larger_than_seed_range']} beyond the seed range); "
                        f"{L['forest']['median_nse_loss']:+.3f} for the random forest; largest loss {L['with_properties']['largest_nse_loss']:.2f} / {L['without_properties']['largest_nse_loss']:.2f} / {L['forest']['largest_nse_loss']:.2f}.")
        findings.append(f"{days(k).capitalize()}, basin properties for a never-seen basin: {P_['verdict']} by the median rule - higher with properties in {P_['n_basins_higher']} of {P_['n_basins']} basins, lower in {P_['n_basins_lower_with_properties']}; "
                        f"median difference {P_['median_nse_difference']:+.3f} NSE, mean {P_['mean_nse_difference']:+.3f}, difference of the medians {P_['difference_of_medians']:+.3f} (typical seed range {s['in_training_seed_range_median']:.3f}, Wilcoxon p = {P_['wilcoxon_p']:.3f}); "
                        f"for basins in training: {s['basin_properties']['in_training']['verdict']}, higher with properties in {s['basin_properties']['in_training']['n_basins_higher']} of {s['n_basins']}, median difference {s['basin_properties']['in_training']['median_nse_difference']:+.3f}.")
        A = s["lstm_against_forest_held_out"]
        findings.append(f"{days(k).capitalize()}, LSTM against the random forest, both never having seen the basin: the LSTM is higher in {A['with_properties']['n_basins_higher']} of {s['n_basins']} basins with basin properties "
                        f"(median difference {A['with_properties']['median_nse_difference']:+.3f}) and in {A['without_properties']['n_basins_higher']} of {s['n_basins']} with weather only ({A['without_properties']['median_nse_difference']:+.3f}).")
    pc = lambda x: f"{100 * x:.0f}%" if x == x else "no warning given"
    who = ((REF, "in training"), ("lobo_w_static", "held out with properties"), ("lobo_w_nostatic", "held out weather only"), ("lobo_rf_w", "held-out forest"))
    for k in F.LEADS:
        w = summary[lk(k)]["warnings_model_level"]
        findings.append(f"Warnings {days(k)} with the model's own level (pooled over {len(ids)} basins): " + "; ".join(
            f"{lab} {pc(w[n]['onset']['pod'])} of onsets warned with {pc(w[n]['onset']['far'])} of those warnings false (all high-flow days: {pc(w[n]['all']['pod'])} warned, {pc(w[n]['all']['far'])} false)" for n, lab in who) + ".")
    for b in worst_basins:
        v = b["forecast_volume_error_pct_1_day_ahead"]
        o = b["onset_warnings_model_level_1_day_ahead"]
        findings.append(f"Loses most when held out - {b['basin']} ({F.LAND_SHORT[b['landscape']]}, {b['area_km2']:.0f} km2): NSE 1 day ahead {b['nse_held_out_with_properties'][0]:.2f} with properties, {b['nse_held_out_without_properties'][0]:.2f} with weather only, "
                        f"{b['nse_forest_held_out'][0]:.2f} for the forest, against {b['nse_in_training'][0]:.2f} in training; forecast volume {v['held_out_with_properties']:+.0f}% / {v['held_out_without_properties']:+.0f}% / {v['forest_held_out']:+.0f}% (in training {v['in_training']:+.0f}%); "
                        f"onsets warned {pc(o['held_out_with_properties']['pod'])} / {pc(o['held_out_without_properties']['pod'])} / {pc(o['forest_held_out']['pod'])} (in training {pc(o['in_training']['pod'])}); properties outside the range of the other 11: "
                        f"{', '.join(b['properties_outside_range_of_other_11']) or 'none'}; nearest other basin: {b['nearest_basin']} (distance {b['nearest_distance']:.1f}); runoff ratio {b['flow_record']['runoff_ratio']:.2f} (rank {b['rank_among_12_highest_first']['runoff_ratio']} of 12, highest first).")
    for b in higher:
        n_, s_, v_, nv = b["nse_1_day_ahead"], b["share_of_rain_reaching_the_river"], b["forecast_volume_error_pct_1_day_ahead"], b["nse_validation_years_1_day_ahead"]
        vols = lambda p: f"{v_[p]['in_training']:+.0f}% in training, " + " / ".join(f"{v_[p][c]:+.0f}%" for c in HELD.values()) + " held out"
        fits = b["test_over_training"] < 1 and abs(v_["test_years"]["held_out_with_properties"]) < abs(v_["test_years"]["in_training"])
        usual = all(nv["in_training"] > nv[c] for c in HELD.values())                # in the validation years the basin still scores best when it was in training
        findings.append(f"Scores higher when held out - {b['basin']}: NSE 1 day ahead {n_['held_out_with_properties']:.2f} with properties, {n_['held_out_without_properties']:.2f} with weather only, {n_['forest_held_out']:.2f} for the forest, "
                        f"against {n_['in_training']:.2f} in training (single seeds {n_['in_training_seed_min']:.2f} to {n_['in_training_seed_max']:.2f}; forest in training {n_['forest_in_training']:.2f}). "
                        f"{100 * s_['training_years']:.0f}% of the rain reached this river in the training years and {100 * s_['test_years']:.0f}% in the test years (drop ranked {b['rank_of_the_drop']} of {len(ids)}, largest first). "
                        f"Forecast volume in the test years: {vols('test_years')}; in the training years: {vols('training_years')}. "
                        + (f"In the validation years the order is the usual one: NSE {nv['in_training']:.2f} in training, " + " / ".join(f"{nv[c]:.2f}" for c in HELD.values()) + " held out. " if usual else "")
                        + ("The higher score goes with a change in the river between the two periods; it is not evidence of skill for unseen rivers." if fits else "The volume errors do not explain it."))
    c_ = what_goes_with_the_loss
    findings.append(f"What goes with the loss (Spearman over {len(ids)} basins, 1 day ahead, with properties): distance to the nearest other basin in property space rho = {c_['nearest_distance']['rho']:+.2f} (p = {c_['nearest_distance']['p']:.2f}); "
                    f"runoff ratio {c_['runoff_ratio']['rho']:+.2f} (p = {c_['runoff_ratio']['p']:.2f}); basin size {c_['log_area']['rho']:+.2f} (p = {c_['log_area']['p']:.2f}). Lowest held-out scores with properties: {', '.join(lowest['held_out_with_properties'])}; "
                    f"with weather only: {', '.join(lowest['held_out_without_properties'])}.")

    capped = sum(c["updates_run"] == config["updates"] and c["updates_run"] - c["best_update"] < config["patience"] * config["eval_every"] for n in VARIANTS for c in cost[n].values())
    alt = [float((t.in_training - t.held_out_without_properties).median()) for t in (T[T.lead == k] for k in F.LEADS)]       # the weather-only variant against the reference WITH properties
    pb_ref, busy = [EV[REF]["test"]["basins"][g]["l1"]["pbias"] for g in ids], timing["training_seconds"] + timing["forest_seconds"]
    notes += [
        f"Protocol: for each basin, one LSTM is trained on the other 11 basins (flood.fit with ids = the 11): input scalers, flow scaler, loss weights and the early-stopping score all come from those 11 basins' training and validation years. "
        "The held-out basin's flow is never read before scoring; only its weather (and, in the first variant, its ten properties) enters the forecast.",
        f"Settings are those chosen by 04_lstm_all.py (results/lstm_all.json, key config) with the budget capped at {config['updates']} updates and patience {config['patience']} (04: {src['config']['updates']} / {src['config']['patience']}). "
        + ("Replaying the capped stopping rule on the recorded validation curve of the in-training reference lstm_w_s0 keeps the same check "
           f"(update {ref_cost['best_update']}), so the reference is exactly the model this budget would have produced." if same else
           "Smoke test: the reference check was skipped." if same is None else "The capped rule would have stopped the in-training reference lstm_w_s0 at another check: the reference had a larger budget than the held-out models."),
        f"{capped} of the {len(ids) * len(VARIANTS)} held-out trainings reached the limit of {config['updates']} updates before the stopping rule was met (best_update and updates_run are in cost_per_fold); a longer budget might have changed them. "
        "The stopping point is chosen on the validation years of the 11 training basins: it measures skill in new years for basins the model knows, not skill for a new basin, so it cannot notice a model that tells its training basins apart by their properties.",
        "That setting (hidden size, learning rate) was chosen in 04 on the validation years of all 12 basins, the held-out one included. 04 found the three candidates equal within seed noise, so this is a weak leak; a strict ungauged test would repeat the choice inside every fold.",
        "One seed per held-out model (seed 0, as the reference lstm_w_s0). A second seed was not affordable, so every difference is set against in_training_seed_range_median: the typical spread, per basin, of the "
        f"{len(seed_sets)} seeds of the in-training model ({', '.join(seed_sets)}). The 24 held-out models also differ in their training basins, so their own seed noise may be larger than that yardstick.",
        "Verdict rule for the basin properties: 'help' / 'hurt' when the median over basins of (with minus without) is beyond the typical seed range, otherwise 'no clear difference'. It describes, it does not prove; nothing was chosen with it. "
        "The count of basins, the mean and the difference of the medians are given beside it because the median hides a single large failure.",
        f"in_training = {REF} (one model for all 12 basins, past weather + basin properties). in_training_no_properties = {REF_NOSTATIC}, one extra training of this script (all 12 basins, weather only, same seed and budget), "
        "added so that the loss of the weather-only variant is measured against a model with the same inputs. loss_* = in training minus held out (positive = worse when never seen). "
        f"Measured against {REF} instead, the median loss of the weather-only variant would be {' / '.join(f'{x:.3f}' for x in alt)} NSE at {' / '.join(map(str, F.LEADS))} days ahead: a number that mixes 'never seen' with 'no basin properties'.",
        f"Not having seen the basin is not the only difference between a basin's two scores. In {int((change < 1).sum())} of the {len(ids)} basins a smaller share of the rain reached the river in the test years than in the training years "
        f"(test / training {change.min():.2f} to {change.max():.2f}; key flow_per_rain), and a model trained on a river has learned the older relation: the forecast volume of {REF} in the test years is {min(pb_ref):+.0f}% to {max(pb_ref):+.0f}% off, 1 day ahead. "
        "A held-out model never learned that relation, so where the change is large the loss measured in the test years can shrink or turn negative for a reason that has nothing to do with unseen rivers (key higher_when_held_out). "
        "flow_per_rain comes from the gauge and only describes; it is never an input.",
        "Random forest (lobo_rf_w): one forest per held-out basin and lead on the pooled issue days of the other 11 basins; features of features.py without river flow plus the ten basin properties; fixed settings "
        f"({', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in FOREST.items() if k != 'n_jobs')}), not tuned. The target is the flow in mm/day without per-basin weights. forest_in_training = base_rf_w of 02_baselines.py, "
        "one forest per basin fitted on that basin's own flow (leaf size tuned on validation years): a different set-up, shown for scale only.",
        "Warnings without a gauge: a warning is given when the forecast exceeds the 95th percentile of the model's own forecasts for that basin and lead on the target days of the training years (LSTM forecasts start after the 365-day warm-up, "
        "so 19 of the 20 training years). A day counts as high flow when the observed flow exceeds the basin's observed 95th percentile of the training years (flood.thresholds) - observations are used for scoring only. "
        "Counts are pooled over the 12 basins with flood.rates. The same rule is applied to the in-training models so that the comparison is like for like.",
        "The 'all' and 'onset' tables inside variants.<name>.test (flood.evaluate) use the observed level as the warning level - a rule an ungauged basin could not apply; they are kept for reference. Use warnings_model_level for the ungauged case.",
        "Three of the ten basin properties (p_mean, aridity, p_seasonality) are CAMELS climate averages of 1989-2009, which overlap the validation and test years (same caveat as in lstm_all.json). No property is computed from the flow record.",
        "worst_basins = the basins with the three largest NSE losses 1 day ahead of either LSTM variant (their union), largest loss first; lowest_held_out_nse_1_day_ahead lists the three lowest held-out scores, "
        "which also contain basins that are hard to forecast even when they are in training.",
        "compared_with_other_11 uses the mean and spread of the other 11 basins' properties - exactly the scaling the held-out model applied. flow_record values (runoff ratio, mean flow ...) come from the gauge and are used only to describe a basin, never as an input.",
        "All models are scored on identical days: coverage is 1.0 for every model, basin and lead in the validation and test years (asserted). The test years chose nothing.",
        "With 12 basins (four per landscape) a median can move with a single basin; the Wilcoxon and Spearman p-values describe 12 dependent basins and are not corrected for the many comparisons."]
    if not F.QUICK and busy > 25 * 60:
        notes.append(f"Time under the CPU lock: {busy / 60:.0f} minutes ({timing['n_trainings']} trainings {timing['training_seconds']:.0f} s, forests {timing['forest_seconds']:.0f} s), more than the 25 minutes planned. "
                     "Smoke tests of other scripts do not wait for the lock, so the seconds per fold depend on what else was running and are not a clean measure of cost.")
    if rescore:
        notes.append("This file was written by --rescore: every score, table and note was recomputed from the saved prediction sets without training again; the per-fold training records and the timings "
                     "(n_trainings, training_seconds, forest_seconds, waited_for_cpu_s, runtime_s) are those of the training run. rescore_runtime_s is the time of this pass.")
    if F.QUICK:
        notes.append("SMOKE TEST (FLOOD_QUICK=1): tiny training budget, numbers are meaningless.")

    R = dict(script=SCRIPT, question="Can the model forecast a river it has never seen?", quick_mode=F.QUICK, split=F.SPLIT, thresholds=thr,
             settings=dict(config=config, config_of_04=src["config"], seed=SEED, inputs="w", input_label=F.INPUT_LABEL["w"], basin_properties=F.STATIC, forest={k: v for k, v in FOREST.items() if k != "n_jobs"},
                           in_training_reference=REF, in_training_reference_cost=ref_cost, reference_identical_under_this_budget=same, in_training_seeds_for_spread=seed_sets,
                           warning_rule="forecast > 95th percentile of the model's own forecasts for that basin and lead in the training years"),
             basins={g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g]), properties={c: float(B.loc[g, c]) for c in F.STATIC},
                             flow_record={c: float(B.loc[g, c]) for c in FLOW_SIGNS}, compared_with_other_11=U[g]) for g in ids},
             variants={n: dict(label=VARIANTS[n]["label"] if n in VARIANTS else "held out, random forest (weather features + basin properties)", saved_as=n, model="rf" if n == "lobo_rf_w" else "lstm",
                               uses_basin_properties=n != "lobo_w_nostatic", test=EV[n]["test"], val_summary=EV[n]["val"], warnings_model_level=EV[n]["warn"], cost_per_fold=cost[n],
                               seconds_total=round(sum(c["seconds"] for c in cost[n].values()), 1)) for n in names},
             references={n: dict(label=lab, saved_as=n, test=EV[n]["test"], val_summary=EV[n]["val"], warnings_model_level=EV[n]["warn"], **extra) for n, lab, extra in
                         ((REF, "in training: one LSTM for all 12 basins, past weather + basin properties (04_lstm_all.py, seed 0)", dict(cost=ref_cost)),
                          (REF_NOSTATIC, "in training: one LSTM for all 12 basins, past weather only (this script, seed 0)", dict(cost=cost[REF_NOSTATIC])),
                          ("base_rf_w", "in training: one random forest per basin, past weather (02_baselines.py)", {})) if n in EV},
             comparison=rows, summary=summary, median_by_landscape_1_day_ahead=by_land, worst_basins=worst_basins, lowest_held_out_nse_1_day_ahead=lowest, higher_when_held_out=higher, flow_per_rain=flow_per_rain,
             what_goes_with_the_loss=what_goes_with_the_loss,
             findings=findings, notes=notes, **timing, runtime_s=round(time.perf_counter() - t0, 1))
    if rescore:                                                                  # keep the cost of the training run; the time of this pass goes beside it
        R["rescore_runtime_s"], R["runtime_s"] = R["runtime_s"], json.loads((F.RES_OUT / "lstm_lobo.json").read_text())["runtime_s"]
    (F.RES_OUT / "lstm_lobo.json").write_text(json.dumps(R, indent=1))
    return R


def figure(R):
    """Per basin, 1 day ahead: test NSE when the basin was in training against when it was held out (with / without basin properties)."""
    v = {r["gauge_id"]: r for r in R["comparison"] if r["lead"] == 1}
    held = ("held_out_with_properties", "held_out_without_properties")
    OFF = -0.5                                                                   # held-out scores below this are not drawn to scale: they would squeeze all the others
    low = min(min([x["in_training_seed_min"]] + [x[c] for c in held if x[c] >= OFF]) for x in v.values())
    lo = max(OFF, min(-0.1, np.floor(low * 10) / 10 - 0.05))
    wide = 0.25 * (1 - lo)                                                       # width of the strip left of the scale where those scores are parked,
    park = dict(zip(held, (lo - 0.84 * wide, lo - 0.39 * wide)))                 # one column per variant, with the value written beside the marker
    num = lambda x: f"{x:.2f}".replace("-", "−")
    fig, ax = plt.subplots(figsize=(6.7, 5.8))
    y, ticks, n_off = 0, {}, 0
    for land, col in F.LAND_COL.items():
        grp = sorted([g for g in v if v[g]["landscape"] == land], key=lambda g: -v[g]["in_training"])
        if not grp:
            continue
        ax.plot(0.012, y, marker="s", ms=6.5, color=col, transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
        ax.text(0.03, y, F.LAND_SHORT[land], transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.3, fontweight="bold", color=INK)
        y += 1
        for g in grp:
            x = v[g]
            px = {c: x[c] if x[c] >= OFF else park[c] for c in held}
            ends = [max(p, lo) for p in px.values()] + [x["in_training"]]
            ax.plot([min(ends), max(ends)], [y, y], color=AXIS, lw=1.2, zorder=2)
            ax.plot([x["in_training_seed_min"], x["in_training_seed_max"]], [y, y], color=col, lw=6.5, alpha=.3, solid_capstyle="butt", zorder=3)
            ax.plot(px[held[1]], y, marker="o", ms=8.6, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.1, zorder=6, clip_on=False)
            ax.plot(px[held[0]], y, marker="D", ms=5.6, markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.7, zorder=5, clip_on=False)
            ax.plot(x["in_training"], y, marker="o", ms=6.8, color=col, markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=4)
            for c in held:
                if x[c] < OFF:
                    n_off += 1
                    ax.text(park[c] + 0.087 * wide, y - 0.02, num(x[c]) if x[c] > -10 else num(x[c])[:-1], va="center", ha="left", fontsize=7.5, color=INK, zorder=7)
            ticks[y] = x["basin"]
            y += 1
        y += 0.5
    ax.set_ylim(y - 0.9, -0.7); ax.set_yticks(list(ticks), list(ticks.values())); ax.tick_params(labelsize=8.3, length=0); ax.tick_params(axis="y", labelcolor=INK2)
    ax.grid(axis="y", visible=False); ax.axvline(0, color=MUTED, lw=1, zorder=1)
    xt = [t for t in np.arange(-0.4, 1.01, 0.2) if t >= lo - 1e-9]
    ax.set_xlim(lo - wide if n_off else lo, 1.0); ax.set_xticks(xt, [num(t)[:-1] for t in xt])
    if n_off:                                                                    # the parking strip for scores below the scale
        ax.axvspan(lo - wide, lo, color=GRID, alpha=.55, lw=0, zorder=0)
        ax.text(lo - wide / 2, -0.017, f"below {num(OFF).rstrip('0')}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.3, color=MUTED)
    ax.set_xlabel("NSE in the test years, 1 day ahead (1 = perfect, 0 = no better than the mean flow)", fontsize=8.5)
    fig.suptitle("A river the model has never seen: flow 1 day ahead, in training against held out", fontsize=10.2, fontweight="bold", color=INK, x=0.012, y=0.985, ha="left")
    h = [Line2D([], [], ls="none", marker="o", ms=6.8, color=INK2, markeredgecolor=SURFACE, label="Basin in training\n(one LSTM for all 12 basins)"),
         Line2D([], [], ls="none", marker="D", ms=5.6, markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.7, label="Held out: trained on the other 11,\nweather + basin properties"),
         Line2D([], [], ls="none", marker="o", ms=8.6, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.1, label="Held out: trained on the other 11,\nweather only")]
    s, n = R["summary"]["l1"], len(v)
    L_ = s["loss_from_not_seeing_the_basin"]
    note = [f"All models read past weather only (no river gauge). Colour = landscape. Shaded bar: range of the {len(R['settings']['in_training_seeds_for_spread'])} single seeds of the in-training model.",
            f"Held out, NSE stays above 0 in {s['n_basins_positive_nse']['held_out_with_properties']} of {n} basins with basin properties and in {s['n_basins_positive_nse']['held_out_without_properties']} of {n} with weather only (one seed each).",
            f"Median change of NSE when held out: {num(-L_['with_properties']['median_nse_loss'])} with properties; {num(-L_['without_properties']['median_nse_loss'])} with weather only (against its own in-training model, not drawn)."]
    if n_off:
        note.insert(1, f"Grey strip: {n_off} held-out score{'s' * (n_off > 1)} below {num(OFF).rstrip('0')}, not drawn to scale; the value is written beside the marker.")
    fig.text(0.5, 0.012, "\n".join(note), ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.03 + 0.027 * len(note)), columnspacing=1.6, handletextpad=.5)
    fig.subplots_adjust(left=0.19, right=0.965, top=0.93, bottom=0.19 + 0.027 * len(note))
    fig.savefig(F.FIG_OUT / "lobo.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    R = json.loads((F.RES_OUT / "lstm_lobo.json").read_text()) if "--figure" in sys.argv else run("--rescore" in sys.argv)
    R = json.loads(json.dumps(R))                                               # the figure reads exactly what the JSON holds
    figure(R)
    pd.set_option("display.width", 250)
    T = pd.DataFrame(R["comparison"])
    print(T[T.lead == 1].drop(columns=["gauge_id", "lead"]).round(3).to_string(index=False))
    print(f"\n{R['n_trainings']} trainings in {R['training_seconds']:.0f} s, forests {R['forest_seconds']:.0f} s, waited {R['waited_for_cpu_s']:.0f} s for the CPU")
    print("\n".join(R["findings"]))
    print("notes:", *R["notes"], sep="\n  ")
