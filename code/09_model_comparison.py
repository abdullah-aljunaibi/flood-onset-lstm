"""Week 6 (optional) - Transformer against LSTM: which one forecasts river flow better here, and what does each cost to train?

Both networks go through the same trainer (flood.fit): the same 12 basins, the same inputs (past weather + river gauge +
basin properties), the same split, loss, stopping rule and scoring. Only the network differs.
1. Each network gets the same small learning-rate check on the validation years (one seed, few updates).
2. Final models: the same random seeds, update limit and patience for both.
3. Cost of each: weights, multiply-add operations per training update (from the formulas written out below), measured
   seconds per update, training time, and an energy estimate (seconds x an ASSUMED 15 W - nothing was measured).
4. A proof inside the script that both networks are causal: changing the inputs after day t leaves every forecast issued
   up to day t unchanged.
Writes results/model_comparison.json, the prediction sets cmp_lstm and cmp_transformer (seed averages) and
figures/model_comparison.png. Redraw the figure and rewrite the side-by-side comparison, the plain-language answer and the
notes on timing and on the limit of the learning-rate check from the saved numbers, without training:
python code/09_model_comparison.py --figure
"""
import json
import os
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D

import flood as F
from common import *

use_style()
SCRIPT, INPUT = "09_model_comparison.py", "wq"
MODELS = ("lstm", "transformer")
NAME = {"lstm": "LSTM", "transformer": "Transformer"}
SHARED = dict(hidden=64, warmup=180, block=185, batch=32, updates=1000, eval_every=100, patience=6)     # identical for both networks
DROPOUT = {"lstm": 0.2, "transformer": 0.1}
TF = dict(layers=2, heads=4, feed_forward=128)
CHECK_LR, CHECK = {"lstm": [1e-3, 2e-3], "transformer": [3e-4, 1e-3]}, dict(updates=300)                # learning-rate check: one seed, capped
SEEDS, WATT = [0, 1], 15.0
if F.QUICK:
    SHARED.update(updates=4, eval_every=4)
    CHECK_LR, CHECK, SEEDS = {m: v[1:] for m, v in CHECK_LR.items()}, dict(updates=2, eval_every=2), [0]
FORMULAS = {
    "symbols": "T = days per training sequence (warmup + block), B = sequences per update, n = inputs per day, h = hidden units, d = d_model, "
               "f = feed-forward width, L = layers, 3 = leads. One multiply-add = one weight times one number, added to a sum.",
    "lstm_forward": "B x T x [4 x h x (h + n) + 3 x h]: four gates, each a weight matrix of h x (h + n), on every day; plus the output layer.",
    "transformer_forward": "B x [T x n x d + L x (4 x T x d^2 + 2 x T^2 x d + 2 x T x d x f) + 3 x T x d]: input projection; per layer the four attention "
                           "projections (query, key, value, output), the day-by-day attention (T^2 x d for the scores and T^2 x d for mixing the values, "
                           "summed over the heads) and the feed-forward block; plus the output layer. The causal mask hides later days but does not save "
                           "operations: the full T x T table is still computed.",
    "training_update": "Forward pass only. A training update also runs the backward pass, which costs about twice the forward pass, so one update is "
                       "roughly 3 x the forward count (a rule of thumb, not measured). Biases, layer normalisation and the activation functions are not counted.",
    "energy": f"training seconds x {WATT:g} W / 3600 = watt-hours. {WATT:g} W is the nominal design power of this laptop's processor - an ASSUMPTION, "
              "not a measurement; the screen and the rest of the laptop are not included.",
    "skill": "skill = mean over the three leads of the median NSE over the 12 basins; one point of skill = 0.01 NSE.",
}
lk = lambda k: f"l{k}"
jf = lambda x: float(np.asarray(x, float))


class TransformerNet(nn.Module):
    """Transformer encoder read with a causal mask: the forecast issued on day t attends only to days up to t."""

    def __init__(self, n_in, cfg):
        super().__init__()
        d, T = cfg.hidden, cfg.warmup + cfg.block
        self.proj, self.head, self.causal = nn.Linear(n_in, d), nn.Linear(d, len(F.LEADS)), True
        layer = nn.TransformerEncoderLayer(d, TF["heads"], TF["feed_forward"], cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, TF["layers"], enable_nested_tensor=False)
        pos, div = torch.arange(T)[:, None], torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)                          # sinusoidal position code: day number within the sequence
        self.register_buffer("pe", pe)

    def forward(self, x):                                                                              # [sequence, day, feature] -> [sequence, day, lead]
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T) if self.causal else None
        return self.head(self.enc(self.proj(x) + self.pe[:T], mask=mask))


NETS = {"lstm": F.LSTMNet, "transformer": TransformerNet}


def operations(model, n_in, c):
    """Multiply-add operations of one forward pass over one training batch (FORMULAS says what is counted)."""
    T, d, n_out = c["warmup"] + c["block"], c["hidden"], len(F.LEADS)
    if model == "lstm":
        parts = dict(gates=4 * d * (d + n_in) * T, output_layer=n_out * d * T)
    else:
        L, f = TF["layers"], TF["feed_forward"]
        parts = dict(input_projection=n_in * d * T, attention_projections=L * 4 * d * d * T, attention_day_by_day=L * 2 * T * T * d,
                     feed_forward=L * 2 * d * f * T, output_layer=n_out * d * T)
    per_seq = sum(parts.values())
    return dict(per_sequence_forward=per_seq, per_update_forward=per_seq * c["batch"], per_update_training_approx=3 * per_seq * c["batch"], parts_per_sequence=parts)


def causality(net, n_in, T, days):
    """Change the inputs after day t and compare the outputs (evaluation mode, so dropout is off).

    Returns the largest change of any output up to day t (must be zero up to rounding) and the smallest 'largest change after
    day t' (must be clearly above zero, otherwise the test could not see anything)."""
    g = torch.Generator().manual_seed(123)
    x = torch.randn(3, T, n_in, generator=g)
    before, after = 0.0, np.inf
    net.eval()
    with torch.no_grad():
        y = net(x)
        for t in days:
            x2 = x.clone()
            x2[:, t + 1:] = torch.randn(3, T - t - 1, n_in, generator=g)
            y2 = net(x2)
            before, after = max(before, float((y2[:, :t + 1] - y[:, :t + 1]).abs().max())), min(after, float((y2[:, t + 1:] - y[:, t + 1:]).abs().max()))
    net.train()
    return before, after


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        cpu = ""
    return dict(cpu=cpu or platform.processor(), logical_cpus=os.cpu_count(), torch_threads=torch.get_num_threads(), torch=torch.__version__)


def run():
    t0 = time.perf_counter()
    D, B = F.load_data(), F.basins()
    ids, thr = list(D), F.thresholds(D)
    T = SHARED["warmup"] + SHARED["block"]
    n_total, done = sum(len(v) for v in CHECK_LR.values()) + len(MODELS) * len(SEEDS), [0]

    def train(model, lr, seed, purpose, **budget):
        """One model on all 12 basins. A hook on the network stamps the clock at every forward pass, which gives the time of every update."""
        stamps, info = [], {}

        def make(n_in, cfg):
            net = NETS[model](n_in, cfg)
            net.register_forward_pre_hook(lambda m, _: stamps.append((time.perf_counter(), time.process_time(), m.training)))
            info["n_in"] = n_in
            return net

        cfg = F.Cfg(use_flow=True, static=tuple(F.STATIC), seed=seed, lr=lr, dropout=DROPOUT[model], **dict(SHARED, **budget))
        r = F.fit(cfg, D, ids, static_df=B, make_net=make)
        s = np.array(stamps, float)
        upd = (s[1:, 2] > 0) & (s[:-1, 2] > 0)                                 # from one training pass to the next = one full update
        wall, cpu = np.diff(s[:, 0])[upd], np.diff(s[:, 1])[upd]
        tp = time.perf_counter()
        P = r.predict(D, static_df=B)
        forecast_s = time.perf_counter() - tp
        val = F.evaluate(P, thr, "val")
        cz = dict(zip(("largest_change_up_to_day_t", "smallest_change_after_day_t"), causality(r.net, info["n_in"], T, [0, 1, cfg.warmup - 1, T // 2, T - 2])))
        if model == "transformer":                                             # control: without the mask the same test must fail
            r.net.causal = False
            cz["largest_change_up_to_day_t_without_mask"] = causality(r.net, info["n_in"], T, [T // 2])[0]
            r.net.causal = True
            assert cz["largest_change_up_to_day_t_without_mask"] > 1e-3, "the causality test cannot tell a masked from an unmasked network"
        assert cz["largest_change_up_to_day_t"] < 1e-5 < cz["smallest_change_after_day_t"], f"{model} is not causal: {cz}"
        cost = dict(r.summary(), n_inputs=info["n_in"], seconds=r.seconds, cpu_seconds=r.cpu_seconds, s_per_update_median=jf(np.median(wall)),
                    cpu_s_per_update_median=jf(np.median(cpu)), s_per_update_mean=jf(wall.mean()), updates_timed=int(upd.sum()),
                    seconds_to_best=next(h["seconds"] for h in r.history if h["update"] == r.best_update),
                    energy_wh_assumed=r.seconds * WATT / 3600, forecast_seconds=forecast_s)
        done[0] += 1
        print(f"training {done[0]} of {n_total} ({purpose}): {NAME[model]}, lr {lr:g}, seed {seed} | best at update {r.best_update} of {r.updates_run}, {r.seconds:.0f} s "
              f"({cost['s_per_update_median']:.3f} s per update) | validation NSE {r.best_val_nse:.4f} (median by lead "
              + " ".join(f"{val['summary'][lk(k)]['nse_median']:.3f}" for k in F.LEADS) + ")", flush=True)
        return dict(P=P, val=val, cost=cost, score=r.best_val_nse, curve=r.history, causality=cz, lr=lr)

    t_ask = time.perf_counter()
    with F.cpu_lock(SCRIPT):
        t1 = time.perf_counter()
        check, diverged, pick, lr = {m: [] for m in MODELS}, [], {}, {}
        for m in MODELS:
            for c in CHECK_LR[m]:
                try:
                    check[m].append(train(m, c, SEEDS[0], "learning-rate check", **CHECK))
                except AssertionError as e:                                    # a diverged run gives not-a-number forecasts, which the trainer refuses
                    if "no forecast for some validation days" not in str(e):
                        raise
                    diverged.append(f"{NAME[m]} with learning rate {c:g}")
                    print(f"learning-rate check: {diverged[-1]} diverged and is dropped", flush=True)
            pick[m] = int(np.argmax([r["score"] for r in check[m]]))
            lr[m] = check[m][pick[m]]["lr"]
            print(f"{NAME[m]} learning rate chosen on the validation years: {lr[m]:g}", flush=True)
        runs = {m: {s: train(m, lr[m], s, "final model") for s in SEEDS} for m in MODELS}
        train_s = time.perf_counter() - t1

    # ---- scores: validation and test years, every seed and the seed average (the test years choose nothing) ---------------
    models = {}
    for m in MODELS:
        P = F.ensemble([r["P"] for r in runs[m].values()])
        settings = dict(SHARED, lr=lr[m], dropout=DROPOUT[m], **(TF if m == "transformer" else {}))
        F.save_preds(f"cmp_{m}", P, dict(script=SCRIPT, model=m, inputs=INPUT, seeds=SEEDS, config=settings,
                                         what=f"one {NAME[m]} for all 12 basins, 365-day sequences, day-by-day average of the seeds"))
        ev = {p: F.evaluate(P, thr, p) for p in ("val", "test")}
        for r in runs[m].values():
            r["test"] = F.evaluate(r["P"], thr, "test")
        for e in list(ev.values()) + [r[p] for r in runs[m].values() for p in ("val", "test")]:
            assert all(e["summary"][lk(k)]["coverage_min"] == 1 for k in F.LEADS), f"{m}: days without a forecast - the models would not be compared on identical days"
        c = [r["cost"] for r in runs[m].values()]
        mean, tot = lambda key: jf(np.mean([x[key] for x in c])), lambda key: jf(np.sum([x[key] for x in c]))
        models[m] = dict(
            label=NAME[m], saved_as=f"cmp_{m}", settings=settings, n_inputs=c[0]["n_inputs"], n_params=c[0]["n_params"], operations=operations(m, c[0]["n_inputs"], SHARED),
            cost=dict(s_per_update_median=jf(np.median([x["s_per_update_median"] for x in c])), cpu_s_per_update_median=jf(np.median([x["cpu_s_per_update_median"] for x in c])),
                      best_update_mean=mean("best_update"), updates_run_mean=mean("updates_run"), seconds_to_best_mean=mean("seconds_to_best"), seconds_per_model=mean("seconds"), cpu_seconds_per_model=mean("cpu_seconds"),
                      seconds_all_seeds=tot("seconds"), cpu_seconds_all_seeds=tot("cpu_seconds"), energy_wh_per_model_assumed=mean("energy_wh_assumed"),
                      energy_wh_all_seeds_assumed=tot("energy_wh_assumed"), forecast_seconds=mean("forecast_seconds")),
            per_seed={str(s): dict(cost=r["cost"], val=r["val"]["summary"], test=r["test"]["summary"], causality=r["causality"], curve=r["curve"],
                                   test_nse_by_basin={g: [r["test"]["basins"][g][lk(k)]["nse"] for k in F.LEADS] for g in ids}) for s, r in runs[m].items()},
            seed_average=ev)
    for m in MODELS:
        ck = [r["cost"] for r in check[m]]
        models[m]["cost"].update(check_seconds=jf(np.sum([x["seconds"] for x in ck])), check_cpu_seconds=jf(np.sum([x["cpu_seconds"] for x in ck])),
                                 check_energy_wh_assumed=jf(np.sum([x["energy_wh_assumed"] for x in ck])))
    all_runs = {m: list(runs[m].values()) + check[m] for m in MODELS}
    causal = {m: dict(largest_change_up_to_day_t=max(r["causality"]["largest_change_up_to_day_t"] for r in rs),
                      smallest_change_after_day_t=min(r["causality"]["smallest_change_after_day_t"] for r in rs), networks_tested=len(rs)) for m, rs in all_runs.items()}
    causal["transformer"]["largest_change_up_to_day_t_without_mask"] = min(r["causality"]["largest_change_up_to_day_t_without_mask"] for r in all_runs["transformer"])
    causal["test"] = (f"Every trained network, evaluation mode: 3 random sequences of {T} days; the inputs after day t are replaced by new random numbers for t = 0, 1, "
                      f"{SHARED['warmup'] - 1}, {T // 2}, {T - 2} (counted from 0). largest_change_up_to_day_t is the largest absolute change of any output up to day t (scaled flow "
                      "units; asserted below 1e-5), smallest_change_after_day_t shows that the later outputs do react, and the Transformer without its mask fails the same test.")

    gap = {m: max(r["score"] for r in check[m]) - min(r["score"] for r in check[m]) for m in MODELS}
    spread = {m: max(r["score"] for r in runs[m].values()) - min(r["score"] for r in runs[m].values()) for m in MODELS}
    ref = F.RES / "lstm_all.json"
    ref = json.loads(ref.read_text()).get("config") if ref.exists() else None
    at_limit = [f"{NAME[m]} seed {s}" for m in MODELS for s, r in runs[m].items() if r["cost"]["updates_run"] >= SHARED["updates"]]
    at_best = [f"{NAME[m]} seed {s}" for m in MODELS for s, r in runs[m].items() if r["cost"]["best_update"] >= SHARED["updates"]]
    notes = [
        f"Both networks are trained by flood.fit on the 12 basins together with past weather + river gauge + the ten basin properties ({models['lstm']['n_inputs']} inputs per day), "
        f"sequences of {T} days ({SHARED['warmup']} read before the first scored forecast, {SHARED['block']} scored), {SHARED['batch']} sequences per update, at most {SHARED['updates']} "
        f"updates, a validation check every {SHARED['eval_every']} updates and stop after {SHARED['patience']} checks without improvement. Only the network differs.",
        (f"Update limit: {', '.join(at_limit)} ran to the limit of {SHARED['updates']} updates without the early stop" + (f"; {', '.join(at_best)} had the best validation score at the very last "
         "check and might still have improved with more updates" if at_best else "; all of them had their best validation score before the last check") + ". The limit is the same for both "
         "networks and was set by the time available on this laptop." if at_limit else f"Every final model stopped early, before the limit of {SHARED['updates']} updates."),
        f"The sequences are {T} days here, not the 730 days of 04_lstm_all.py, because the attention cost grows with the square of the sequence length. cmp_lstm is therefore not the "
        "same model as lstm_wq and its scores may differ from results/lstm_all.json.",
        f"Settings: each network had only its learning rate checked, in the same way - LSTM {CHECK_LR['lstm']}, Transformer {CHECK_LR['transformer']}, seed {SEEDS[0]}, at most "
        f"{dict(SHARED, **CHECK)['updates']} updates, chosen by the trainer's own validation-years score (mean NSE over 12 basins and 3 leads). The candidates differ by {gap['lstm']:.4f} (LSTM) and "
        f"{gap['transformer']:.4f} (Transformer); the final seeds of the chosen setting differ by {spread['lstm']:.4f} and {spread['transformer']:.4f} in the same score. Fixed without a search: "
        f"LSTM hidden 64 and dropout {DROPOUT['lstm']:g} (the engine defaults); Transformer d_model 64, {TF['layers']} layers, {TF['heads']} heads, feed-forward {TF['feed_forward']}, dropout "
        f"{DROPOUT['transformer']:g}." + (f" For reference, the main all-basin LSTM of 04_lstm_all.py (730-day sequences) uses hidden {ref['hidden']}, learning rate {ref['lr']:g}, dropout "
                                          f"{ref['dropout']:g}." if ref else ""),
        "Neither network had its size, depth or dropout searched: a longer search could change the picture for either model, more likely for the Transformer, which has more settings.",
        "Fair test: scalers, thresholds and weights use the training years 1980-2000; both learning rates and every stopping point were chosen on the validation years 2000-2005; "
        "the test years 2005-2014 were scored once, after all choices. Which model is 'better' is read from the validation years; the test scores are reported next to it.",
        "Both models are scored on identical days: coverage is 1.0 for every model, seed, basin and lead in the validation and test years (asserted in the code).",
        f"Seed spread: only {len(SEEDS)} random seed(s) per model were affordable on this laptop, so 'min' and 'max' over seeds are a rough guide to the noise, not a confidence interval. "
        "cmp_<model> is the day-by-day average of the seeds' forecasts; an average usually scores a little higher than each single seed.",
        "Operations are counted from the formulas in 'formulas' (forward pass, multiply-adds only), not measured. Measured seconds need not scale with that count: it leaves out everything "
        "that is not a weight multiplication (softmax, normalisation, moving the day-by-day attention tables through memory) and how well each network uses the processor's threads.",
        f"Energy = seconds x {WATT:g} W, the nominal design power of the processor: an assumption, no power was measured. No cloud or GPU prices are quoted because none were given.",
        "The learning-rate checks are reported separately (check_seconds) and are not added to seconds_per_model; the tuning of weeks 2-3 is not counted at all.",
        "Warning scores are pooled over the 12 basins; high flow = above the basin's 95th percentile of the training years; onset = only forecasts issued while the river was still at or "
        "below that level.",
    ] + ([f"Learning-rate check: {', '.join(diverged)} diverged (forecasts became not-a-number) and was dropped."] if diverged else []) + (["QUICK smoke test: a handful of updates, one seed - the numbers are not results."] if F.QUICK else [])
    R = dict(split=F.SPLIT, quick_mode=F.QUICK, input_set=INPUT, input_label=F.INPUT_LABEL[INPUT], static_inputs=list(F.STATIC), seeds=SEEDS, thresholds=thr,
             settings=dict(shared=SHARED, sequence_days=T, lstm=models["lstm"]["settings"], transformer=models["transformer"]["settings"], assumed_power_watt=WATT, machine=machine(), lstm_all_config_for_reference=ref),
             **{f"{m}_check": dict(seed=SEEDS[0], candidates=CHECK_LR[m], budget=dict(SHARED, **CHECK), criterion="mean validation-years NSE over 12 basins and 3 leads (higher is better)",
                                   table=[dict(lr=r["lr"], val_nse_mean=r["score"], val_nse_median_by_lead=[r["val"]["summary"][lk(k)]["nse_median"] for k in F.LEADS],
                                               best_update=r["cost"]["best_update"], updates_run=r["cost"]["updates_run"], seconds=r["cost"]["seconds"], cpu_seconds=r["cost"]["cpu_seconds"],
                                               s_per_update_median=r["cost"]["s_per_update_median"], chosen=i == pick[m], curve=r["curve"]) for i, r in enumerate(check[m])]) for m in MODELS},
             causality=causal, formulas=FORMULAS, models=models, notes=notes, n_trainings=done[0], training_seconds=round(train_s, 1), waited_for_cpu_s=round(t1 - t_ask, 1))
    R["runtime_s"] = round(time.perf_counter() - t0 - (t1 - t_ask), 1)
    return R


def compare(R):
    """Side-by-side scores of the two models (validation and test years) and the cost of one point of skill; from the saved numbers only."""
    M, seeds, out = R["models"], [str(s) for s in R["seeds"]], {}
    for part in ("val", "test"):
        out[part] = {}
        per_seed_skill = {m: [jf(np.mean([M[m]["per_seed"][s][part][lk(k)]["nse_median"] for k in F.LEADS])) for s in seeds] for m in MODELS}
        for k in F.LEADS:
            row = {}
            for m in MODELS:
                a, ps = M[m]["seed_average"][part]["summary"][lk(k)], [M[m]["per_seed"][s][part][lk(k)] for s in seeds]
                row[m] = dict(nse_median=a["nse_median"], nse_median_by_seed=[p["nse_median"] for p in ps], nse_mean=a["nse_mean"], nse_min=a["nse_min"],
                              **{f"{w}_{c}": a[w][c] for w in ("all", "onset") for c in ("pod", "far", "csi")},
                              **{f"{w}_{c}_by_seed": [p[w][c] for p in ps] for w, c in (("all", "csi"), ("onset", "pod"), ("onset", "far"))})
            lo, hi = ({m: f(row[m]["nse_median_by_seed"]) for m in MODELS} for f in (min, max))
            nb = [M[m]["seed_average"][part]["basins"] for m in MODELS]
            row.update(nse_median_transformer_minus_lstm=row["transformer"]["nse_median"] - row["lstm"]["nse_median"],
                       seed_ranges_overlap=bool(lo["lstm"] <= hi["transformer"] and lo["transformer"] <= hi["lstm"]),
                       n_basins_lstm_higher=int(sum(nb[0][g][lk(k)]["nse"] > nb[1][g][lk(k)]["nse"] for g in nb[0])), n_basins=len(nb[0]))
            nse_b = [{g: b[g][lk(k)]["nse"] for g in b} for b in nb]            # a median over 12 basins hangs on its two middle basins: how much does ONE basin move it?
            gx = max(nse_b[0], key=lambda g: abs(nse_b[1][g] - nse_b[0][g]))    # the basin in which the two models differ most
            rest = [np.median([v[g] for g in v if g != gx]) for v in nse_b]
            row.update(nse_mean_transformer_minus_lstm=row["transformer"]["nse_mean"] - row["lstm"]["nse_mean"],
                       largest_basin_difference=dict(gauge_id=gx, basin=F.SHORT[gx], lstm=nse_b[0][gx], transformer=nse_b[1][gx], nse_median_transformer_minus_lstm_without_it=jf(rest[1] - rest[0])))
            out[part][lk(k)] = row
        sk = {m: jf(np.mean([out[part][lk(k)][m]["nse_median"] for k in F.LEADS])) for m in MODELS}
        mn = {m: jf(np.mean([out[part][lk(k)][m]["nse_mean"] for k in F.LEADS])) for m in MODELS}
        spread = max(max(v) - min(v) for v in per_seed_skill.values())
        out[part]["skill"] = dict(lstm=sk["lstm"], transformer=sk["transformer"], by_seed=per_seed_skill, transformer_minus_lstm=sk["transformer"] - sk["lstm"],
                                  mean_of_seeds_transformer_minus_lstm=jf(np.mean(per_seed_skill["transformer"]) - np.mean(per_seed_skill["lstm"])), largest_seed_spread=spread,
                                  seed_ranges_overlap=bool(min(per_seed_skill["lstm"]) <= max(per_seed_skill["transformer"]) and min(per_seed_skill["transformer"]) <= max(per_seed_skill["lstm"])),
                                  mean_nse_over_basins=dict(lstm=mn["lstm"], transformer=mn["transformer"], transformer_minus_lstm=mn["transformer"] - mn["lstm"]))
    out["val"]["trainer_score_by_seed"] = {m: [M[m]["per_seed"][s]["cost"]["best_val_nse"] for s in seeds] for m in MODELS}        # mean NSE over basins and leads
    cost = {}
    for m in MODELS:
        c, s = M[m]["cost"], jf(np.mean(out["test"]["skill"]["by_seed"][m]))
        pts = 100 * s if s > 0 else float("nan")
        cost[m] = dict(test_skill_mean_of_seeds=s, seconds_per_model=c["seconds_per_model"], cpu_seconds_per_model=c["cpu_seconds_per_model"], energy_wh_per_model_assumed=c["energy_wh_per_model_assumed"],
                       seconds_per_point=c["seconds_per_model"] / pts, cpu_seconds_per_point=c["cpu_seconds_per_model"] / pts, energy_wh_per_point_assumed=c["energy_wh_per_model_assumed"] / pts)
    ratio = {key: M["transformer"][a][b] / M["lstm"][a][b] if b else M["transformer"][a] / M["lstm"][a] for key, (a, b) in
             dict(parameters=("n_params", None), operations_per_update=("operations", "per_update_forward"), seconds_per_update=("cost", "s_per_update_median"),
                  cpu_seconds_per_update=("cost", "cpu_s_per_update_median"), updates_to_best=("cost", "best_update_mean"), seconds_per_model=("cost", "seconds_per_model"),
                  cpu_seconds_per_model=("cost", "cpu_seconds_per_model")).items()}
    R["comparison"], R["cost_of_skill"] = out, dict(definition=FORMULAS["skill"], **cost, transformer_over_lstm=ratio)
    R["cost_of_skill"]["transformer_over_lstm"]["seconds_per_point"] = cost["transformer"]["seconds_per_point"] / cost["lstm"]["seconds_per_point"]
    took = lambda m: " and ".join(f"{M[m]['per_seed'][x]['cost']['seconds']:.0f} s for {M[m]['per_seed'][x]['cost']['updates_run']} updates" for x in seeds)
    ck = [(NAME[m], r) for m in MODELS for r in R[f"{m}_check"]["table"]]
    rising = [f"{name} {r['lr']:g}" for name, r in ck if r["best_update"] >= r["updates_run"]]
    load, th = [h["load"] for m in MODELS for x in seeds for h in M[m]["per_seed"][x]["curve"]], R["settings"]["machine"]["torch_threads"]
    busy = (f"and other processes did run alongside: the 1-minute load average at the validation checks of the final trainings was {min(load):.1f} to {max(load):.1f}, where this job alone gives "
            f"at most about {th}, so all times are longer than on an idle laptop" if max(load) > th + 1 else "and may have run alongside")
    V, n_b = out["val"], out["val"]["l1"]["n_basins"]
    by3 = lambda part, key, f="{:+.3f}": " / ".join(f.format(out[part][lk(k)][key]) for k in F.LEADS)
    kx = max(F.LEADS, key=lambda k: abs(V[lk(k)]["nse_median_transformer_minus_lstm"]))
    bx = V[lk(kx)]["largest_basin_difference"]
    R["notes"] = [x for x in R["notes"] if not x.startswith(("Timing", "Learning-rate check limit", "Basins:"))] + [
        f"Basins: a median over {n_b} basins is set by its two middle basins, so a single basin can move it, and the seed spread does not show that. Transformer minus LSTM at 1 / 2 / 3 days "
        f"ahead, validation years: median NSE {by3('val', 'nse_median_transformer_minus_lstm')}, mean NSE {by3('val', 'nse_mean_transformer_minus_lstm')}, LSTM higher in "
        f"{by3('val', 'n_basins_lstm_higher', '{}')} of {n_b} basins; test years: median NSE {by3('test', 'nse_median_transformer_minus_lstm')}, mean NSE "
        f"{by3('test', 'nse_mean_transformer_minus_lstm')}, LSTM higher in {by3('test', 'n_basins_lstm_higher', '{}')} of {n_b} basins. The largest median difference on the validation years is at "
        f"{kx} day{'s' * (kx > 1)} ahead; there the two models differ most in {bx['basin']} (NSE {bx['lstm']:.3f} LSTM, {bx['transformer']:.3f} Transformer), and without that one basin the "
        f"median difference is {bx['nse_median_transformer_minus_lstm_without_it']:+.3f} instead of {V[lk(kx)]['nse_median_transformer_minus_lstm']:+.3f}.",
        f"Learning-rate check limit: {len(rising)} of the {len(ck)} candidates" + (f" ({', '.join(rising)})" if rising else "") + f" had their best validation score at the last check of the "
        f"{R['lstm_check']['budget']['updates']} updates of the check" + (", i.e. they were still improving when it ended. The check therefore ranks the learning rates by how fast they learn at the "
        f"start, not by where they end after {R['settings']['shared']['updates']} updates: a lower rate might still have overtaken." if rising else ".") + " The trainer keeps the learning rate "
        "constant for both networks (no warm-up, no decay).",
        f"Timing: taken under flood.cpu_lock on a 2-core laptop CPU with {th} torch threads, so no other full training job held the CPU, but smoke tests and "
        f"other light jobs of the project do not wait for that lock {busy}. Wall-clock seconds and CPU seconds (summed over threads) are both given; seconds per update is the median time "
        "between two consecutive training passes, measured inside the real training. Training time includes the validation checks.",
        f"Timing noise: the LSTM seeds took {took('lstm')}; the Transformer seeds {took('transformer')}. Where the same number of updates took a different time, the cause is the state "
        f"of the laptop (other processes, heat), not the model - so read the time ratios as rough: about {ratio['seconds_per_update']:.0f}x per update and {ratio['seconds_per_model']:.0f}x per model."]


def answer(R):
    """The plain-language answer, written from the numbers in R (every figure in these sentences is read from the results)."""
    M, C, K, n = R["models"], R["comparison"], R["cost_of_skill"], len(R["seeds"])
    X = K["transformer_over_lstm"]
    three = lambda part, m: " / ".join(f"{C[part][lk(k)][m]['nse_median']:.3f}" for k in F.LEADS)
    rng = lambda part, m: " / ".join(f"{min(C[part][lk(k)][m]['nse_median_by_seed']):.3f} to {max(C[part][lk(k)][m]['nse_median_by_seed']):.3f}" for k in F.LEADS)
    pc = lambda x: "none" if x != x else f"{100 * x:.0f}%"
    warn = lambda m, w, c: " / ".join(pc(C["test"][lk(k)][m][f"{w}_{c}"]) for k in F.LEADS)
    A = [f"Accuracy in the test years (median NSE over 12 basins at 1 / 2 / 3 days ahead, average forecast of {n} seed{'s' * (n > 1)}): LSTM {three('test', 'lstm')}, "
         f"Transformer {three('test', 'transformer')}. Single seeds: LSTM {rng('test', 'lstm')}, Transformer {rng('test', 'transformer')}."]
    v, t = C["val"]["skill"], C["test"]["skill"]
    noise = lambda s: "the single seeds of the two models overlap: within seed noise" if s["seed_ranges_overlap"] else "the single seeds of the two models do not overlap"
    ts = lambda m: ", ".join(f"{x:.3f}" for x in C["val"]["trainer_score_by_seed"][m])
    A.append(f"Which is more accurate (differences are Transformer minus LSTM; skill = mean over the three leads of the median NSE): on the validation years, where such a choice has to be made, "
             f"{v['transformer_minus_lstm']:+.3f} (LSTM {v['lstm']:.3f}, Transformer {v['transformer']:.3f}; {noise(v)}); the trainer's own validation score (mean NSE over basins and leads) is "
             f"{ts('lstm')} for the LSTM seeds and {ts('transformer')} for the Transformer seeds. In the test years {t['transformer_minus_lstm']:+.3f} (LSTM {t['lstm']:.3f}, Transformer "
             f"{t['transformer']:.3f}; {noise(t)}). Seeds of the same model differ by up to {v['largest_seed_spread']:.3f} (validation) and {t['largest_seed_spread']:.3f} (test) in this skill. "
             f"In the mean NSE over the basins, which one basin moves less than the median, the difference is {v['mean_nse_over_basins']['transformer_minus_lstm']:+.3f} on the validation years and "
             f"{t['mean_nse_over_basins']['transformer_minus_lstm']:+.3f} in the test years (see the note on basins).")
    A.append("By lead in the test years (Transformer minus LSTM, median NSE): " + ", ".join(
        f"{k} day{'s' * (k > 1)} ahead {C['test'][lk(k)]['nse_median_transformer_minus_lstm']:+.3f} (single seeds {'overlap' if C['test'][lk(k)]['seed_ranges_overlap'] else 'do not overlap'}; "
        f"LSTM higher in {C['test'][lk(k)]['n_basins_lstm_higher']} of {C['test'][lk(k)]['n_basins']} basins)" for k in F.LEADS) + ".")
    A.append(f"High-flow warnings in the test years, pooled over the basins, at 1 / 2 / 3 days ahead - share of high-flow days warned: LSTM {warn('lstm', 'all', 'pod')}, Transformer "
             f"{warn('transformer', 'all', 'pod')}; share of warnings that were false: LSTM {warn('lstm', 'all', 'far')}, Transformer {warn('transformer', 'all', 'far')}. Onsets only (river still "
             f"low when the forecast was issued) - warned: LSTM {warn('lstm', 'onset', 'pod')}, Transformer {warn('transformer', 'onset', 'pod')}; false among those warnings: LSTM "
             f"{warn('lstm', 'onset', 'far')}, Transformer {warn('transformer', 'onset', 'far')}.")
    L, T = M["lstm"], M["transformer"]
    big = lambda x: f"{x / 1e9:.2f} billion" if x >= 1e9 else f"{x / 1e6:.0f} million"
    A.append(f"Cost to train one model: the Transformer has {T['n_params']:,} weights against {L['n_params']:,} ({X['parameters']:.1f}x), needs about {big(T['operations']['per_update_forward'])} "
             f"multiply-adds per update (forward pass) against {big(L['operations']['per_update_forward'])} ({X['operations_per_update']:.1f}x), took {T['cost']['s_per_update_median']:.3f} s per "
             f"update against {L['cost']['s_per_update_median']:.3f} s ({X['seconds_per_update']:.1f}x), reached its best validation score after {T['cost']['best_update_mean']:.0f} updates "
             f"({T['cost']['seconds_to_best_mean']:.0f} s) against {L['cost']['best_update_mean']:.0f} ({L['cost']['seconds_to_best_mean']:.0f} s), and trained for {T['cost']['seconds_per_model']:.0f} s against {L['cost']['seconds_per_model']:.0f} s ({X['seconds_per_model']:.1f}x; "
             f"CPU seconds {T['cost']['cpu_seconds_per_model']:.0f} against {L['cost']['cpu_seconds_per_model']:.0f}). At an assumed {R['settings']['assumed_power_watt']:g} W that is about "
             f"{T['cost']['energy_wh_per_model_assumed']:.2f} Wh against {L['cost']['energy_wh_per_model_assumed']:.2f} Wh per model. The learning-rate checks cost another {T['cost']['check_seconds']:.0f} s (Transformer) and "
             f"{L['cost']['check_seconds']:.0f} s (LSTM).")
    kl, kt = K["lstm"], K["transformer"]
    costly, cheap = ("transformer", "lstm") if kt["seconds_per_model"] > kl["seconds_per_model"] else ("lstm", "transformer")
    d_skill, d_sec = K[costly]["test_skill_mean_of_seeds"] - K[cheap]["test_skill_mean_of_seeds"], K[costly]["seconds_per_model"] - K[cheap]["seconds_per_model"]
    per = lambda c: (f"{c['seconds_per_point']:.1f} s of training ({1000 * c['energy_wh_per_point_assumed']:.1f} mWh assumed)" if c["seconds_per_point"] == c["seconds_per_point"]
                     else "not defined (its skill is not above zero)")
    if d_skill <= 0:
        extra = f"The extra {d_sec:.0f} s of the {NAME[costly]} buy no extra skill: it scores {100 * -d_skill:.1f} points lower."
    elif t["seed_ranges_overlap"]:
        extra = (f"The extra {d_sec:.0f} s of the {NAME[costly]} buy no skill that stands clear of the seed noise: {100 * d_skill:+.1f} points, while seeds of the same model differ by up to "
                 f"{100 * t['largest_seed_spread']:.1f} points.")
    else:
        extra = f"The extra {d_sec:.0f} s of the {NAME[costly]} buy {100 * d_skill:.1f} extra points, i.e. {d_sec / (100 * d_skill):.0f} s per extra point."
    A.append(f"Cost of one point of skill (0.01 NSE, test years, mean of single seeds): LSTM {per(kl)}, Transformer {per(kt)}"
             + (f" - {X['seconds_per_point']:.1f} times as much. " if X["seconds_per_point"] == X["seconds_per_point"] else ". ") + extra)
    slower = f"{X['seconds_per_model']:.1f} times the training time" if X["seconds_per_model"] > 1 else f"{100 * X['seconds_per_model']:.0f}% of the training time"
    in_test = (f"in the test years the difference is {t['transformer_minus_lstm']:+.3f} NSE, " + ("within seed noise" if t["seed_ranges_overlap"] else "with single seeds that do not overlap"))
    if v["transformer_minus_lstm"] > 0 and not v["seed_ranges_overlap"] and t["transformer_minus_lstm"] > 0:
        verdict = (f"Is the Transformer worth it here? On the validation years it is more accurate by {v['transformer_minus_lstm']:.3f} NSE (single seeds do not overlap); {in_test}. It needs "
                   f"{slower}; whether that is worth it depends on how often the model must be retrained.")
    elif v["transformer_minus_lstm"] < 0 and not v["seed_ranges_overlap"]:
        verdict = (f"Is the Transformer worth it here? No. On the validation years it is less accurate than the LSTM by {-v['transformer_minus_lstm']:.3f} NSE in the median over the basins (single "
                   f"seeds do not overlap; in the mean over the basins the difference is {v['mean_nse_over_basins']['transformer_minus_lstm']:+.3f}); {in_test}. So it is at best as accurate as "
                   f"the LSTM, and it needs {slower}.")
    else:
        verdict = (f"Is the Transformer worth it here? No. On the validation years the two models cannot be told apart beyond seed noise (difference {v['transformer_minus_lstm']:+.3f} NSE, "
                   f"seed spread up to {v['largest_seed_spread']:.3f}); {in_test}. The Transformer needs {slower}.")
    A.append(verdict + f" This holds for this problem size - 12 basins, 20 training years, sequences of {R['settings']['sequence_days']} days, small networks on a laptop CPU, with only the learning rate "
             f"checked for each model; with {n} seed{'s' * (n > 1)} per model the noise estimate is rough. It is not a general verdict on Transformers.")
    R["answer"] = A


def figure(R):
    """Left: skill by lead for both models with the single seeds. Right: what one model costs to train (four small bar charts)."""
    M, C, n = R["models"], R["comparison"]["test"], len(R["seeds"])
    fig = plt.figure(figsize=(6.7, 4.25))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.12, 1], left=0.085, right=0.985, top=0.80, bottom=0.235, wspace=0.42, hspace=1.05)
    ax = fig.add_subplot(gs[:, 0])
    vals = []
    for m, off in zip(MODELS, (-0.07, 0.07)):
        x = np.array(F.LEADS) + off
        med, seeds = [C[lk(k)][m]["nse_median"] for k in F.LEADS], np.array([C[lk(k)][m]["nse_median_by_seed"] for k in F.LEADS])
        ax.vlines(x, seeds.min(1), seeds.max(1), color=F.MODEL_COL[m], lw=5, alpha=.28, zorder=2)
        ax.plot(np.repeat(x, seeds.shape[1]), seeds.ravel(), ls="none", marker="_", ms=13, markeredgewidth=1.5, color=F.MODEL_COL[m], zorder=3)
        ax.plot(x, med, "-o", color=F.MODEL_COL[m], lw=1.8, ms=5.5, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=4)
        for xx, v in zip(x, med):                                             # LSTM values below-left of the point, Transformer values above-right; the colour says whose value it is
            ax.annotate(f"{v:.3f}", (xx, v), xytext=(-7, -6) if off < 0 else (7, 6), textcoords="offset points", ha="right" if off < 0 else "left", va="top" if off < 0 else "bottom",
                        fontsize=7.8, fontweight="bold", color=F.MODEL_COL[m], zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
        vals += med + seeds.ravel().tolist()
    lo, hi = np.floor(min(vals) * 10) / 10, np.ceil(max(vals) * 10) / 10 + 0.2
    ax.set_ylim(lo, hi); ax.set_xlim(0.45, 3.55); ax.set_xticks(F.LEADS, ["1 day\nahead", "2 days\nahead", "3 days\nahead"]); ax.set_yticks(np.arange(lo, hi - 0.05, 0.1))
    ax.axhline(0, color=MUTED, lw=1, zorder=1); ax.grid(axis="x", visible=False); ax.tick_params(labelsize=8, length=0)
    ax.set_ylabel("median NSE of the 12 basins, test years (1 = perfect)", fontsize=8.2)
    h = [Line2D([], [], color=F.MODEL_COL[m], lw=1.8, marker="o", ms=5.5, markeredgecolor=SURFACE, label=f"{NAME[m]} (average of {n} seed{'s' * (n > 1)})") for m in MODELS]
    h.append(Line2D([], [], ls="none", marker="_", ms=13, markeredgewidth=1.5, color=INK2, label="single seed"))
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=8, handlelength=2.2, borderaxespad=0.2)

    big = lambda x: f"{x / 1e9:.2f} billion" if x >= 1e9 else f"{x / 1e6:.0f} million"
    rows = [("Weights to learn", lambda m: M[m]["n_params"], lambda x: f"{x:,.0f}"),
            ("Multiply-adds per update (formula)", lambda m: M[m]["operations"]["per_update_forward"], big),
            ("Seconds per update (measured, median)", lambda m: M[m]["cost"]["s_per_update_median"], lambda x: f"{x:.3f} s"),
            ("Training time per model (measured)", lambda m: M[m]["cost"]["seconds_per_model"], lambda x: f"{x:.0f} s")]
    for i, (title, get, fmt) in enumerate(rows):
        a = fig.add_subplot(gs[i, 1])
        v = [get(m) for m in MODELS]
        a.barh([0, 1], v, height=0.62, color=[F.MODEL_COL[m] for m in MODELS], zorder=3)
        for y, x in enumerate(v):
            a.text(x + 0.025 * max(v), y, fmt(x) + (f"  ({v[1] / v[0]:.1f}x)" if y == 1 else ""), va="center", ha="left", fontsize=8, color=INK2)
        a.set_xlim(0, max(v) * 1.75); a.set_ylim(1.6, -0.6); a.set_yticks([0, 1], [NAME[m] for m in MODELS]); a.tick_params(labelsize=8, length=0)
        a.set_xticks([]); a.grid(visible=False); a.spines["bottom"].set_visible(False)
        a.set_title(title, fontsize=8.2, fontweight="normal", color=INK2, pad=3)
    for x, head in ((0.012, "Accuracy: river flow 1 to 3 days ahead"), (gs.get_grid_positions(fig)[2][1] - 0.115, "Cost of training")):
        fig.text(x, 0.855, head, fontsize=9.3, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.suptitle("Transformer against LSTM: same data, same trainer, same test", fontsize=10.4, fontweight="bold", color=INK, x=0.012, y=0.985, ha="left")
    v, s = R["comparison"]["val"]["skill"], R["settings"]
    note = [f"One model for 12 basins, input: {R['input_label']}, sequences of {s['sequence_days']} days, {s['shared']['batch']} sequences per update.",
            f"Validation years (mean of 3 leads), LSTM / Transformer: median NSE {v['lstm']:.3f} / {v['transformer']:.3f} (seeds differ by up to {v['largest_seed_spread']:.3f}), "
            f"mean NSE {v['mean_nse_over_basins']['lstm']:.3f} / {v['mean_nse_over_basins']['transformer']:.3f}.",
            f"Times: 2-core laptop, no graphics card, mean of the seeds. Energy per model at an assumed {s['assumed_power_watt']:g} W: "
            f"{M['lstm']['cost']['energy_wh_per_model_assumed']:.2f} Wh (LSTM), {M['transformer']['cost']['energy_wh_per_model_assumed']:.2f} Wh (Transformer)."]
    fig.text(0.012, 0.012, "\n".join(note), ha="left", va="bottom", fontsize=7.6, color=INK2, linespacing=1.45)
    fig.savefig(F.FIG_OUT / "model_comparison.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    out = F.RES_OUT / "model_comparison.json"
    R = json.loads(out.read_text()) if "--figure" in sys.argv else run()
    R = json.loads(json.dumps(R))                                               # everything below reads exactly what the JSON holds
    compare(R)
    answer(R)
    R = {k: R[k] for k in [k for k in R if k not in ("notes", "n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s")] + ["notes", "n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s"]}
    out.write_text(json.dumps(R, indent=1))
    figure(R)
    print(f"\n{R['n_trainings']} trainings in {R['training_seconds']:.0f} s (waited {R['waited_for_cpu_s']:.0f} s for the CPU)")
    print("\n".join(R["answer"]))
    print("notes:", *R["notes"], sep="\n  ")
