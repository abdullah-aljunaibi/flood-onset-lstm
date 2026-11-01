"""Week 2 - an LSTM for ONE basin (French Broad River at Rosman, NC): which settings matter, does it beat the baselines,
and how far back in time does it look?

1. Tuning on the validation years only, with past weather + river gauge as input: one knob at a time around a base setting -
   model size (hidden units), learning rate, dropout, and the days of history read before the first scored forecast.
2. Final models with the chosen setting for the four input sets (three random seeds for the two that exist in practice).
3. Like-for-like comparison with the baselines of 02_baselines.py on the test years (NSE, KGE, high-flow warnings). With the
   perfect weather forecast only 3 days ahead is like-for-like (the LSTM reads all three coming days, the baselines stop at
   the target day), so no winner is called there for 1 and 2 days ahead.
4. Memory test: the test score when the model may read only the last c days.
5. A diagnostic, not a choice: the chosen setting trained for a fixed number of updates, to show what longer training does.
Writes results/lstm_single.json, results/hparam_log.csv (one row per training run), the prediction sets lstm1_w, lstm1_wq,
lstm1_wf, lstm1_wqf, and figures/lstm_single_tuning.png, lstm_single_hydrograph.png, lstm_memory.png.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import flood as F
from common import *

use_style()
T0 = time.time()
G = "03439000"
D = F.load_data()
d, thr = D[G], {G: F.thresholds(D)[G]}
SETS = {"w": dict(use_flow=False, future=False), "wq": dict(use_flow=True, future=False),
        "wf": dict(use_flow=False, future=True), "wqf": dict(use_flow=True, future=True)}
BASE = dict(hidden=64, lr=2e-3, dropout=0.2, warmup=365, block=365)
KNOBS = dict(hidden=[16, 32, 64, 128], lr=[5e-4, 1e-3, 2e-3, 5e-3], dropout=[0.0, 0.2, 0.4], warmup=[30, 90, 180, 365])
BUDGET = dict(updates=800, eval_every=25, patience=6)      # one update already reads 32 years of this basin: check often
FINAL = dict(updates=2000)                                 # final models may train longer than tuning runs if validation keeps improving
SEEDS, CONTEXTS, FIXED = [0, 1, 2], [2, 7, 14, 30, 60, 90, 180, 365, 730], [200, 600]
if F.QUICK:
    FINAL = {}
    KNOBS = dict(hidden=[32, 64], lr=[1e-3, 2e-3], dropout=[0.2, 0.4], warmup=[90, 365])
    BUDGET, SEEDS, CONTEXTS, FIXED = dict(updates=30, eval_every=15, patience=6), [0], [7, 90, 365], [30]
NOTES, LOG = [], []

# the validation years of this basin hold two hurricane floods (September 2004); a second validation score that stops one
# week before the larger one is logged next to the official one, as a diagnostic only
val_q = d.q_mm.loc[F.SPLIT["val"][0]:F.SPLIT["val"][1]]
SPLIT_CALM = dict(F.SPLIT, val=(F.SPLIT["val"][0], str((val_q.idxmax() - pd.Timedelta(days=7)).date())))


def by_lead(ev, key="nse"):
    return [ev[f"l{k}"][key] for k in F.LEADS]


def train(purpose, inp, knobs, seed, **budget):
    """Train one model, score it on validation and test years, and add one row to the hyper-parameter log."""
    b = dict(BUDGET, **budget)
    if knobs["hidden"] > 64:
        b["updates"] = b["updates"] // 2                                        # about 3x the cost per update: cap it
    run = F.fit(F.Cfg(**SETS[inp], **knobs, seed=seed, **b), D, [G])
    P = run.predict(D)
    ev = {part: F.evaluate(P, thr, part)["basins"][G] for part in ("train", "val", "test")}
    calm = F.evaluate(P, thr, "val", split=SPLIT_CALM)["basins"][G]
    row = dict(run_id=len(LOG) + 1, purpose=purpose, input_set=inp, **{k: knobs[k] for k in ("hidden", "lr", "dropout", "warmup", "block")},
               batch=run.cfg.batch, seed=seed, max_updates=b["updates"], eval_every=b["eval_every"], patience=b["patience"], n_params=run.n_params,
               updates_run=run.updates_run, best_update=run.best_update, seconds=round(run.seconds, 1), train_nse_mean=float(np.mean(by_lead(ev["train"]))), val_nse_mean=float(np.mean(by_lead(ev["val"]))),
               **{f"val_nse_l{k}": ev["val"][f"l{k}"]["nse"] for k in F.LEADS}, val_nse_mean_before_2004_floods=float(np.mean(by_lead(calm))),
               **{f"test_nse_l{k}": ev["test"][f"l{k}"]["nse"] for k in F.LEADS})
    LOG.append(row)
    write_log()
    print(f"run {row['run_id']:2d} {purpose:22s} {inp:3s} hidden {knobs['hidden']:3d} lr {knobs['lr']:.0e} dropout {knobs['dropout']:.1f} history {knobs['warmup']:3d} "
          f"seed {seed} | best at update {run.best_update:3d} of {run.updates_run:3d}, {run.seconds:4.0f} s | train NSE {row['train_nse_mean']:.3f} | val NSE {row['val_nse_mean']:.3f} | test NSE "
          + " ".join(f"{x:.3f}" for x in by_lead(ev["test"])), flush=True)
    return dict(run=run, P=P, ev=ev, row=row)


def write_log():
    pd.DataFrame(LOG).round(4).to_csv(F.RES_OUT / "hparam_log.csv", index=False)


def val_mean(runs):
    return float(np.mean([r["row"]["val_nse_mean"] for r in runs.values()]))


t_ask = time.time()
with F.cpu_lock("03_lstm.py (LSTM on one basin)"):
    WAITED = time.time() - t_ask                                                # another job may hold the CPU first: not part of the run time
    # ---- 1. tuning on validation years: one knob at a time, input = past weather + river gauge -----------------------
    base_runs = {s: train("tuning", "wq", BASE, s) for s in SEEDS}
    T = {(knob, v): base_runs[0] if v == BASE[knob] else train("tuning", "wq", dict(BASE, **{knob: v}), 0) for knob, vals in KNOBS.items() for v in vals}
    base_vals = [r["row"]["val_nse_mean"] for r in base_runs.values()]
    tol = float(max(base_vals) - min(base_vals))                                # seed spread at the base setting = the noise level
    chosen, per_knob = dict(BASE), {}
    for knob, vals in KNOBS.items():
        score = {v: T[knob, v]["row"]["val_nse_mean"] for v in vals}
        best = max(score, key=score.get)
        near = [v for v in vals if score[v] >= score[best] - tol]
        if knob in ("hidden", "warmup"):                                        # smaller is cheaper
            pick = min(near)
            why = "best validation score" if pick == best else f"smallest value whose validation score is within the seed spread ({tol:.3f}) of the best ({best})"
        else:                                                                   # no cheaper direction: keep the base value unless clearly beaten
            pick = BASE[knob] if BASE[knob] in near else best
            why = "best validation score" if pick == best else f"base value kept: within the seed spread ({tol:.3f}) of the best ({best})"
        chosen[knob] = pick
        per_knob[knob] = dict(values=vals, val_nse_mean=[score[v] for v in vals], best_update=[T[knob, v]["row"]["best_update"] for v in vals],
                              test_nse_l1_for_the_record=[T[knob, v]["row"]["test_nse_l1"] for v in vals], best=best, within_noise_of_best=near, chosen=pick, reason=why)
    # the chosen values were never run together: check the combination against the base setting, again on validation years
    cand = base_runs if chosen == BASE else {s: train("final", "wq", chosen, s, **FINAL) for s in SEEDS}
    check = dict(base_val_nse_seed_mean=val_mean(base_runs), chosen_val_nse_seed_mean=val_mean(cand), tolerance=tol, fell_back_to_base=False)
    setting, wq_runs = chosen, cand
    if chosen != BASE and val_mean(cand) < val_mean(base_runs) - tol:
        setting, wq_runs, check["fell_back_to_base"] = dict(BASE), base_runs, True
        for r in cand.values():
            r["row"]["purpose"] = "tuning (combination rejected)"
    if setting == BASE:
        for r in base_runs.values():
            r["row"]["purpose"] = "tuning + final"
    write_log()
    print(f"chosen setting: {setting}  (seed spread at the base setting {tol:.3f}; fell back to base: {check['fell_back_to_base']})", flush=True)

    # ---- 2. final models -----------------------------------------------------------------------------------------------
    FIN = {"w": {s: train("final", "w", setting, s, **FINAL) for s in SEEDS}, "wq": wq_runs,
           "wf": {0: train("final", "wf", setting, 0, **FINAL)}, "wqf": {0: train("final", "wqf", setting, 0, **FINAL)}}
    PF = {inp: F.ensemble([r["P"] for r in runs.values()]) for inp, runs in FIN.items()}
    for inp, P in PF.items():
        F.save_preds(f"lstm1_{inp}", P, dict(script="03_lstm.py", basin=G, input_set=inp, seeds=list(FIN[inp]), setting=setting,
                                             what="LSTM trained on this one basin; average over the seeds listed"))

    # ---- 4. memory test: the final seed-0 models may read only the last c days ----------------------------------------
    MEM = {inp: {str(c): by_lead(F.evaluate(FIN[inp][0]["run"].predict_context(D, context=c, part="test"), thr, "test")["basins"][G]) for c in CONTEXTS}
           for inp in ("w", "wq")}
    print("memory test done", flush=True)

    # ---- diagnostic, not a model choice: the same setting trained for a fixed number of updates, no early stopping ----
    DIAG = {inp: {n: train("diagnostic (fixed length)", inp, setting, 0, updates=n, eval_every=n) for n in FIXED} for inp in ("w", "wq")}

# ---- 3. comparison with the baselines on the test years (outside the lock: the baseline script may need the CPU) ------
COMPARE = {"w": ["base_linreg_w", "base_rf_w"], "wq": ["base_persistence", "base_linreg_wq", "base_rf_wq"],
           "wf": ["base_linreg_wf", "base_rf_wf"], "wqf": ["base_linreg_wqf", "base_rf_wqf"]}
PLAIN = {"base_persistence": "tomorrow = today", "base_linreg": "linear regression", "base_rf": "random forest", "lstm1": "LSTM (this basin only)"}
plain = lambda n: PLAIN[n if n in PLAIN else n.rsplit("_", 1)[0]]
need = sorted({n for v in COMPARE.values() for n in v})
t_wait = time.time()
while not all(F.have_preds(n) for n in need) and time.time() - t_wait < (0 if F.QUICK else 1800):
    print(f"waiting for the baseline prediction sets of 02_baselines.py ({sum(F.have_preds(n) for n in need)} of {len(need)} there)", flush=True)
    time.sleep(30)
test = slice(*F.SPLIT["test"])
BASEP, base_meta = {}, {}
for n in need:
    for attempt in range(3):                                                    # a file may be half-written while we look
        try:
            p = F.load_preds(n)[G]
            assert np.allclose(p.obs.loc[test].values, d.q_mm.loc[test].values), "observed flow differs"
            BASEP[n], base_meta[n] = {G: p}, F.preds_meta(n)
            break
        except Exception as e:
            err = repr(e)
            time.sleep(0 if F.QUICK or not F.have_preds(n) else 10)
    if n not in BASEP:
        NOTES.append(f"Baseline {n} could not be used ({'not written yet' if not F.have_preds(n) else err}); it is missing from the comparison.")
if "base_persistence" not in BASEP:
    BASEP["base_persistence"] = {G: F.frame_from_issue(d, {k: d.q_mm.values for k in F.LEADS})}
    NOTES.append("base_persistence was not available from 02_baselines.py; 'tomorrow = today' was computed here with flood.frame_from_issue.")

SPREAD = {inp: {k: float(np.ptp([r["ev"]["test"][f"l{k}"]["nse"] for r in runs.values()])) for k in F.LEADS} for inp, runs in FIN.items() if len(runs) > 1}
CMP, WINS = {}, []
for inp, names in COMPARE.items():
    models = {f"lstm1_{inp}": F.evaluate(PF[inp], thr, "test")["basins"][G], **{n: F.evaluate(BASEP[n], thr, "test")["basins"][G] for n in names if n in BASEP}}
    for n, e in models.items():
        if min(by_lead(e, "coverage")) < 1:
            NOTES.append(f"{n}: forecasts cover only {min(by_lead(e, 'coverage')):.3f} of the test days, so its scores are not on identical days.")
    verdict = {}
    for k in F.LEADS:
        lk, mine = f"l{k}", models[f"lstm1_{inp}"][f"l{k}"]
        seeds = [r["ev"]["test"][lk]["nse"] for r in FIN[inp].values()]
        others = {n: e[lk] for n, e in models.items() if n.startswith("base_")}
        yard = SPREAD[inp][k] if inp in SPREAD else max([sp[k] for sp in SPREAD.values()], default=0.0)   # one seed: borrow the spread of the seeded sets
        fair = not (SETS[inp]["future"] and k < max(F.LEADS))   # perfect forecast: the LSTM reads all three coming days at every lead, the baselines only up to the target day
        v = dict(lstm_nse=mine["nse"], lstm_nse_by_seed=seeds, n_seeds=len(seeds), seed_spread_used=yard, seed_spread_is_own=inp in SPREAD, like_for_like=fair)
        if others:
            bn = max(others, key=lambda n: others[n]["nse"])
            bb = others[bn]["nse"]
            call = "too close to call (difference within the seed spread)" if abs(mine["nse"] - bb) <= yard else "LSTM ahead" if mine["nse"] > bb else "baseline ahead"
            call += "" if inp in SPREAD else " (one seed; judged against the seed spread of the models without a weather forecast)"
            if not fair:
                call = "no call: not like-for-like at this lead (the LSTM also reads the weather stamped after the target day, the baselines do not)"
            csi, kge = {n: e[lk]["all"]["csi"] for n, e in models.items()}, {n: e[lk]["kge"] for n, e in models.items()}
            pod = {n: e[lk]["onset"]["pod"] for n, e in models.items()}
            v.update(best_baseline=bn, best_baseline_nse=bb, nse_margin=mine["nse"] - bb, nse_call=call, best_kge=max(kge, key=kge.get),
                     best_warning_csi_all_days=max(csi, key=csi.get),
                     best_onset_pod=[n for n in pod if pod[n] == max(pod.values()) and pod[n] > 0])       # every model that shares the best score; empty when no model caught an onset
            WINS.append(f"{F.INPUT_LABEL[inp]}, {k} day{'s' * (k > 1)} ahead: LSTM NSE {mine['nse']:.3f} (seed spread {yard:.3f}) against {plain(bn)} {bb:.3f} -> {call}"
                        + (f"; KGE: best is {plain(v['best_kge'])} ({kge[v['best_kge']]:.2f}, LSTM {kge[f'lstm1_{inp}']:.2f}); warnings on all high-flow days (CSI): best is "
                           f"{plain(v['best_warning_csi_all_days'])} ({csi[v['best_warning_csi_all_days']]:.2f}, LSTM {csi[f'lstm1_{inp}']:.2f}); onsets caught: LSTM {mine['onset']['hits']} of "
                           f"{mine['onset']['hits'] + mine['onset']['misses']}, {plain(bn)} {others[bn]['onset']['hits']}" if fair else "") + ".")
        verdict[lk] = v
    CMP[inp] = dict(label=F.INPUT_LABEL[inp], models=models, verdict=verdict)

# ---- why validation on this one basin is fragile: how much of the flow variance sits in a few days --------------------
def top_days(part, n=3):
    q = d.q_mm.loc[F.SPLIT[part][0]:F.SPLIT[part][1]]
    dev = (q - q.mean()) ** 2
    t = dev.nlargest(n)
    return dict(share_of_variance_in_top_days=float(t.sum() / dev.sum()), days=[dict(day=str(i.date()), flow_mm=float(q[i]), rain_mm_same_day=float(d.prcp[i]),
                rain_mm_day_before=float(d.prcp[i - pd.Timedelta(days=1)])) for i in t.index], max_rain_mm=float(d.prcp.loc[q.index].max()))

CONC = {part: top_days(part) for part in F.SPLIT}
# why weather stamped after the target day still helps: the river's day-to-day rise against the rain stamped the same day and later
rise = d.q_mm.diff().loc[F.SPLIT["train"][0]:F.SPLIT["train"][1]].iloc[1:-2]
RISE = {j: float(np.corrcoef(rise, d.prcp.shift(-j).loc[rise.index])[0, 1]) for j in (0, 1, 2)}

# ---- figures -----------------------------------------------------------------------------------------------------------
LSTM_C, WEATHER_C = F.MODEL_COL["lstm"], SEQ[2]                                 # LSTM with river gauge / LSTM with weather only
TITLE = {"hidden": "Model size (hidden units)", "lr": "Learning rate", "dropout": "Dropout (share of units switched off)", "warmup": "History read before the first forecast (days)"}
fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.7), sharey=True)
for ax, (knob, vals) in zip(axes.ravel(), KNOBS.items()):
    y, x = per_knob[knob]["val_nse_mean"], np.arange(len(vals))
    ax.axhspan(min(base_vals), max(base_vals), color=GRID, alpha=.8, lw=0)
    ax.plot(x, y, "-o", color=LSTM_C, lw=1.6, ms=4.5, zorder=3)
    ib = vals.index(BASE[knob])
    ax.scatter([ib] * (len(base_vals) - 1), base_vals[1:], s=14, color=MUTED, zorder=2)
    ic = vals.index(setting[knob])
    ax.scatter([ic], [y[ic]], s=110, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
    ax.set_xticks(x, [f"{v:g}" for v in vals]); ax.set_xlim(-.45, len(vals) - .55); ax.tick_params(labelsize=8)
    ax.set_title(TITLE[knob], fontsize=8.8)
lo, hi = min(min(p["val_nse_mean"]) for p in per_knob.values()), max(max(p["val_nse_mean"]) for p in per_knob.values() )
lo, hi = min(lo, min(base_vals)), max(hi, max(base_vals))
axes[0, 0].set_ylim(lo - .12 * (hi - lo + 1e-6), hi + .30 * (hi - lo + 1e-6))
h = [plt.Line2D([], [], marker="o", color=LSTM_C, lw=1.6, ms=4.5, label="one run (seed 0)"), plt.Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="none", ms=4.5, label="base setting, seeds 1 and 2"),
     plt.Rectangle((0, 0), 1, 1, color=GRID, label="seed spread at the base setting"), plt.Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=INK, ms=9, label="chosen value")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7.8, handletextpad=.5, columnspacing=1.3)
fig.supylabel("validation NSE (mean of 1 to 3 days ahead)", fontsize=8.5, color=INK2)
fig.suptitle(f"Tuning on the validation years: {F.SHORT[G]}, LSTM with past weather + river gauge", fontsize=9.8, fontweight="bold", color=INK, x=.012, ha="left")
fig.tight_layout(rect=(0.01, .055, 1, 1.0)); fig.savefig(F.FIG_OUT / "lstm_single_tuning.png", dpi=220); plt.close(fig)

# largest high-flow event of the test years: who rises on time, who is a day late
peak = d.q_mm.loc[test].idxmax()
win = slice(peak - pd.Timedelta(days=14), peak + pd.Timedelta(days=27))
fig, (a0, a1) = plt.subplots(2, 1, figsize=(6.7, 4.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 3.2], hspace=.08))
a0.bar(d.loc[win].index, d.prcp.loc[win], width=.8, color=SEQ[3])
a0.set_ylabel("rain\n(mm/day)", fontsize=8.5); a0.tick_params(labelsize=8)
a0.set_title(f"Largest flood of the test years, {F.SHORT[G]}: forecasts made 1 day ahead", fontsize=9.8)
a1.axhline(thr[G], color=RED, lw=1, ls=(0, (4, 3)), label=f"high-flow level ({thr[G]:.1f} mm/day)")
a1.plot(d.q_mm.loc[win], "-o", color=INK, lw=2, ms=3, label="observed flow", zorder=5)
a1.plot(BASEP["base_persistence"][G].l1.loc[win], color=F.MODEL_COL["persistence"], lw=1.5, label="'tomorrow = today'")
a1.plot(PF["w"][G].l1.loc[win], color=WEATHER_C, lw=1.8, label="LSTM, past weather only")
a1.plot(PF["wq"][G].l1.loc[win], color=LSTM_C, lw=1.8, label="LSTM, past weather + river gauge")
a1.set_ylabel("river flow (mm/day)", fontsize=8.5); a1.tick_params(labelsize=8); a1.set_ylim(0, None)
a1.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=peak.weekday())); a1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %Y"))
a1.legend(frameon=False, fontsize=8, loc="upper right")
fig.align_ylabels(); fig.savefig(F.FIG_OUT / "lstm_single_hydrograph.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# memory: test score 1 day ahead against the days of history the model may read
fig, ax = plt.subplots(figsize=(6.7, 3.5))
ax.axvspan(min(CONTEXTS) / 1.3, setting["warmup"], color=GRID, alpha=.6, lw=0)
ys = {inp: [MEM[inp][str(c)][0] for c in CONTEXTS] for inp in MEM}
for inp, col in (("w", WEATHER_C), ("wq", LSTM_C)):
    ax.plot(CONTEXTS, ys[inp], "-o", color=col, lw=1.8, ms=4.5, label="LSTM, past weather only" if inp == "w" else f"LSTM, {F.INPUT_LABEL[inp]}")
ylo, yhi = max(min(min(v) for v in ys.values()), -0.5), max(max(v) for v in ys.values())
ax.set_ylim(ylo - .08 * (yhi - ylo), yhi + .22 * (yhi - ylo))
ax.text(min(CONTEXTS) / 1.2, ax.get_ylim()[1], f" shorter than the {setting['warmup']} days the model had in training", fontsize=7.8, color=INK2, va="top")
ax.set_xscale("log"); ax.set_xticks(CONTEXTS, [str(c) for c in CONTEXTS]); ax.minorticks_off(); ax.set_xlim(min(CONTEXTS) / 1.3, max(CONTEXTS) * 1.3)
ax.set_xlabel("days of history the model may read before the forecast", fontsize=8.5); ax.set_ylabel("test NSE, 1 day ahead", fontsize=8.5); ax.tick_params(labelsize=8)
ax.set_title(f"How much history the one-basin LSTM uses ({F.SHORT[G]}, test years)", fontsize=9.8)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(F.FIG_OUT / "lstm_memory.png", dpi=220); plt.close(fig)

# ---- results file ------------------------------------------------------------------------------------------------------
B = F.basins().loc[G]
finals = {}
for inp, runs in FIN.items():
    finals[inp] = dict(label=F.INPUT_LABEL[inp], saved_as=f"lstm1_{inp}", n_seeds=len(runs),
                       seeds={str(s): dict(val=r["ev"]["val"], test=r["ev"]["test"], training=r["run"].summary()) for s, r in runs.items()},
                       seed_average={part: F.evaluate(PF[inp], thr, part)["basins"][G] for part in ("val", "test")},
                       test_nse_by_seed={f"l{k}": [r["ev"]["test"][f"l{k}"]["nse"] for r in runs.values()] for k in F.LEADS},
                       training_seconds_total=round(sum(r["run"].seconds for r in runs.values()), 1))
m3 = lambda x: float(np.mean(x))
stop_row = lambda label, r: dict(stopping_point=label, updates=r["run"].best_update, train_nse=by_lead(r["ev"]["train"]), val_nse=by_lead(r["ev"]["val"]),
                                 test_nse=by_lead(r["ev"]["test"]), val_nse_mean_before_2004_floods=r["row"]["val_nse_mean_before_2004_floods"], test=r["ev"]["test"])
OVER = {inp: [stop_row("chosen on validation years (final model, seed 0)", FIN[inp][0])] + [stop_row(f"fixed {n} updates, no early stopping", r) for n, r in DIAG[inp].items()] for inp in DIAG}
for inp, rows in OVER.items():
    a, z = rows[0], rows[-1]
    NOTES.append(f"Would longer training help? LSTM with {F.INPUT_LABEL[inp]}, seed 0, NSE as the mean of 1 to 3 days ahead: stopped where the validation years say (update {a['updates']}) "
                 f"training {m3(a['train_nse']):.2f}, validation {m3(a['val_nse']):.2f}, test {m3(a['test_nse']):.2f}; trained for a fixed {z['updates']} updates training {m3(z['train_nse']):.2f}, "
                 f"validation {m3(z['val_nse']):.2f}, test {m3(z['test_nse']):.2f}."
                 + (" The training score rises while the validation score falls: with only 20 years of one basin the model starts to memorise its training years (overfitting) almost at once, "
                    "so early stopping halts it while it has learned little." if m3(z["train_nse"]) > m3(a["train_nse"]) and m3(z["val_nse"]) < m3(a["val_nse"]) else "")
                 + " These fixed-length runs are a diagnostic, not a model choice; the reported model is the validation-chosen one.")
for inp, runs in FIN.items():
    late = [str(sd) for sd, r in runs.items() if r["run"].best_update > r["row"]["max_updates"] - r["row"]["patience"] * r["row"]["eval_every"]]
    if late:
        NOTES.append(f"lstm1_{inp} (seed {', '.join(late)}) was still improving on the validation years when the limit of {runs[int(late[0])]['row']['max_updates']} updates was reached; its scores are a lower estimate of what this model can do.")
    if len(runs) > 1:
        va, te = [r["row"]["val_nse_mean"] for r in runs.values()], [m3(by_lead(r["ev"]["test"])) for r in runs.values()]
        NOTES.append(f"lstm1_{inp}, seeds {list(runs)}: validation NSE (mean of leads) {', '.join(f'{x:.3f}' for x in va)}; test NSE {', '.join(f'{x:.3f}' for x in te)}."
                     + (" The seed with the best validation score is not the best on the test years: on this basin the validation years rank models poorly." if int(np.argmax(va)) != int(np.argmax(te)) else ""))
edge = [f"{TITLE[k].split(' (')[0].lower()} {setting[k]:g}" for k, v in KNOBS.items() if setting[k] in (min(v), max(v))]
peak_at = [r["best_update"] for r in LOG if r["input_set"] in ("w", "wq") and not r["purpose"].startswith("diagnostic")]
margin = {inp: " / ".join(f"{CMP[inp]['verdict'][f'l{k}'].get('nse_margin', float('nan')):+.2f}" for k in F.LEADS) for inp in ("wf", "wqf")}
NOTES += [
    "With the perfect weather forecast (wf, wqf) only the 3-days-ahead comparison with the baselines is like-for-like: the LSTM reads the weather of all three coming days at every lead, "
    "the baselines of 02_baselines.py only up to the target day. Weather stamped after the target day still tells about that day in this record: in the training years the river's day-to-day rise "
    f"is correlated with the rain stamped on the same day ({RISE[0]:.2f}) and on the next day ({RISE[1]:.2f}), not two days later ({RISE[2]:.2f}). So 1 and 2 days ahead the LSTM knows more than the baselines. "
    f"Test NSE of the LSTM minus the best baseline, 1 / 2 / 3 days ahead: {margin['wf']} (wf) and {margin['wqf']} (wqf). No winner is called for 1 and 2 days ahead with these two input sets "
    "(like_for_like is false in comparison.<set>.verdict).",
    "Everything was chosen on the validation years 2000-2005 (stopping point, settings); the test years 2005-2014 were only scored. Test NSE in the tuning table and in hparam_log.csv is for the record and was not used.",
    f"Validation on one basin is fragile: {CONC['val']['share_of_variance_in_top_days']:.0%} of the validation-year flow variance sits in 3 days (against {CONC['train']['share_of_variance_in_top_days']:.0%} in the "
    f"training and {CONC['test']['share_of_variance_in_top_days']:.0%} in the test years). The two largest are the September 2004 hurricane floods: up to {CONC['val']['max_rain_mm']:.0f} mm of rain in one day (training-year maximum "
    f"{CONC['train']['max_rain_mm']:.0f} mm), with the river peaking on the day of the rain. No model without a weather forecast can foresee that, so the validation NSE of this basin cannot be high and depends on a handful of days. "
    "Column val_nse_mean_before_2004_floods in hparam_log.csv scores the validation years up to one week before the larger flood (a drier stretch with less variance, so its level is not comparable); "
    "it is a diagnostic and was not used to choose anything.",
    "train_nse_mean in hparam_log.csv is scored on the training years the model was fitted to (without the first days that have no forecast); it is there to show overfitting, not skill.",
    f"Tuning used one seed per value. The noise level is the spread of validation NSE over {len(SEEDS)} seeds at the base setting ({tol:.3f}); differences smaller than that are not real. "
    "Within the noise the smaller value was taken for model size and history, and the base value was kept for learning rate and dropout.",
    "The chosen values were tuned one at a time and combined afterwards; the combination was checked against the base setting on validation years (see tuning.combination_check).",
    "The setting was tuned with the river gauge as an input (wq) and used unchanged for the three other input sets."
    + (f" Chosen values at the edge of the range that was tried: {', '.join(edge)}; the best setting may lie outside the ranges tried." if edge else ""),
    f"Budget per run: at most {BUDGET['updates']} updates in tuning (half for hidden 128) and {dict(BUDGET, **FINAL)['updates']} for the final models, validation check every {BUDGET['eval_every']} updates, stop after {BUDGET['patience']} checks without improvement. "
    "One update reads 32 stretches of one year, i.e. more than the whole 20 training years of this basin, so checks are closer together than for the all-basin model. "
    f"This departs from the brief (a patience of about 4 checks, at the engine's default of 100 updates between checks): without a weather forecast the validation score of this basin peaks "
    f"between update {min(peak_at)} and {max(peak_at)} and falls afterwards (tuning.base_curve), so checks 100 updates apart would step over the peak.",
    "'History' (warmup) is the minimum number of days read before a scored forecast; within a block of 365 scored days later forecasts have read up to 365 days more.",
    f"w and wq: {len(SEEDS)} seeds each, the saved prediction set is their day-by-day average. wf and wqf (perfect weather forecast, an upper bound that cannot be had in practice): one seed, no spread known.",
    f"Memory test: the model was trained with at least {setting['warmup']} days of history, so shorter contexts are outside what it was trained on; the test shows how much of the skill needs old weather, not how a model trained on short windows would do."
    + (f" Contexts above {setting['warmup'] + setting['block']} days are longer than any stretch the model read in training." if max(CONTEXTS) > setting["warmup"] + setting["block"] else ""),
    "Only the four knobs named in the plan were tuned. Batch size (32 one-year stretches per update) and the length of the scored stretch were left at the engine defaults; smaller batches might suit a single basin better and were not tried.",
    "High flow = above the 95th percentile of the training years; onset = forecasts issued while the river was still at or below that level.",
    "In the verdicts the LSTM is set against the like-for-like baseline with the highest test NSE at each lead, so it has to beat all of them; nothing was tuned or selected with that.",
    "Training seconds (finals.*.seeds.*.training, hparam_log.csv) are wall-clock times on a laptop that smoke tests of other scripts shared at times: a rough cost measure only. "
    "updates_run, best_update and n_params are exact.",
]
R = dict(basin=dict(gauge_id=G, name=str(B.gauge_name), short=F.SHORT[G], area_km2=float(B.area_gages2), elev_mean_m=float(B.elev_mean), landscape=str(B.landscape), high_flow_level_mm_day=thr[G]),
         split=F.SPLIT, quick_mode=F.QUICK,
         tuning=dict(input_set="wq", base=BASE, budget=BUDGET, final_budget=dict(BUDGET, **FINAL), noise_seed_spread=tol, base_val_nse_by_seed=base_vals, per_knob=per_knob, combination_check=check, chosen=setting,
                     reason="; ".join(f"{k}: {v['chosen']} ({v['reason']})" for k, v in per_knob.items()) + ("; the combination scored worse than the base setting on validation years, so the base setting is used" if check["fell_back_to_base"] else ""),
                     runs=[r for r in LOG if r["purpose"].startswith("tuning")],
                     base_curve=base_runs[0]["run"].history),
         finals=finals, comparison=CMP, who_wins=WINS, baseline_meta=base_meta,
         memory=dict(contexts_days=CONTEXTS, test_nse_by_lead={inp: MEM[inp] for inp in MEM}, model="final seed-0 models",
                     normal_forecast_test_nse={inp: by_lead(FIN[inp][0]["ev"]["test"]) for inp in MEM}),
         diagnostic_fixed_length=OVER,
         variance_concentration=CONC, rise_vs_rain_stamped_days_later_training_years={str(j): c for j, c in RISE.items()}, hparam_log="results/hparam_log.csv", notes=NOTES,
         waited_for_cpu_s=round(WAITED, 1), runtime_s=round(time.time() - T0 - WAITED, 1))
(F.RES_OUT / "lstm_single.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

print(f"\nchosen setting {setting}; {len(LOG)} training runs; {R['runtime_s']:.0f} s")
for inp in FIN:
    print(f"  lstm1_{inp:3s} test NSE by lead " + " ".join(f"{x:.3f}" for x in by_lead(finals[inp]["seed_average"]["test"])))
print("\n".join(WINS))
print("notes:", *NOTES, sep="\n  ")
