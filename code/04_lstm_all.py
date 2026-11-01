"""Week 3 - ONE LSTM for all 12 basins: does it beat the simple methods, and what do a river gauge and a weather forecast add?

A single model is trained on the 12 basins together; ten basin properties (size, slope, climate, soil ...) are extra inputs.
1. Setting check on the validation years (three candidates, input = past weather + river gauge); the best one is used.
2. The four input sets (past weather; + river gauge; + perfect 3-day weather forecast; + both), three random seeds each.
3. Like-for-like comparison on the test years with the baselines of 02_baselines.py that are given the same information.
4. Skill decomposition: what the river gauge adds and what a perfect weather forecast would add, for accuracy and warnings.
Writes results/lstm_all.json, results/lstm_all_scores.csv (per-basin score table), the prediction sets lstm_<set> and
lstm_<set>_s<seed>, and figures/lstm_all.png, figures/lstm_all_basins.png.
Without training again: python code/04_lstm_all.py --rescore  recomputes every score, table and note from the saved prediction
sets (the training records are carried over from the JSON);  python code/04_lstm_all.py --figure  only redraws the figures.
"""
import json
import sys
import time
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

import flood as F
from common import *

use_style()
SCRIPT = "04_lstm_all.py"
SETS = {"w": dict(use_flow=False, future=False), "wq": dict(use_flow=True, future=False),
        "wf": dict(use_flow=False, future=True), "wqf": dict(use_flow=True, future=True)}
CANDIDATES = [dict(hidden=64, lr=2e-3), dict(hidden=64, lr=1e-3), dict(hidden=32, lr=2e-3)]      # hidden is capped at 64: 2-core laptop, no GPU
SEEDS, BUDGET = [0, 1, 2], {}                                                                     # {} = the engine's default training budget
if F.QUICK:
    CANDIDATES, SEEDS, BUDGET = CANDIDATES[::2], [0, 1], dict(updates=8, eval_every=4)      # two seeds: the smoke test walks the same code as the full run
KEYS = ("hidden", "lr", "dropout", "warmup", "block", "batch", "updates", "eval_every", "patience")
COMPARE = {"w": ["base_linreg_w", "base_rf_w"], "wq": ["base_persistence", "base_linreg_wq", "base_rf_wq"],
           "wf": ["base_linreg_wf", "base_rf_wf"], "wqf": ["base_linreg_wqf", "base_rf_wqf"]}       # the same information as the LSTM
ALL_BASE = ["base_persistence", "base_climatology"] + [f"base_{m}_{s}" for m in ("linreg", "rf") for s in SETS]
STEPS = {"river_gauge_without_forecast": ("wq", "w"), "river_gauge_with_perfect_forecast": ("wqf", "wf"),
         "perfect_forecast_without_gauge": ("wf", "w"), "perfect_forecast_with_gauge": ("wqf", "wq")}        # (with, without)
PLAIN = {"persistence": "'Tomorrow = today'", "climatology": "Calendar-day average", "linreg": "Ridge regression", "rf": "Random forest"}
MARK = {"persistence": "D", "climatology": "v", "linreg": "^", "rf": "s", "lstm": "o"}
SPREAD = {"below every seed": "lower than every single LSTM seed", "above every seed": "higher than every single LSTM seed", "inside the seed spread": "inside the range of the single LSTM seeds"}
lk = lambda k: f"l{k}"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"


def config_of(cand):
    """The nine settings later scripts reuse, as a plain dict (everything not named in the candidate is the engine default)."""
    c = asdict(F.Cfg(**cand, **BUDGET))
    return {k: c[k] for k in KEYS}


def paired(a, b):
    """Two per-basin score arrays (first minus second): in how many basins the first is higher, the median difference, Wilcoxon p."""
    d = np.asarray(a) - np.asarray(b)
    try:
        p = float(wilcoxon(a, b).pvalue)                        # two-sided signed-rank test over the basins
    except ValueError:                                          # all differences zero
        p = float("nan")
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=int(len(d)), median_nse_difference=float(np.median(d)), wilcoxon_p=p)


def run(rescore=False):
    t0 = time.perf_counter()
    D, B = F.load_data(), F.basins()
    ids, thr = list(D), F.thresholds(D)
    nse_of = lambda ev, k: np.array([ev["basins"][g][lk(k)]["nse"] for g in ids])
    n_total, done = len(CANDIDATES) + len(SETS) * len(SEEDS) - 1, [0]

    def train(inp, config, seed, purpose):
        """One model on all 12 basins; scored here on the validation years only (the test years are scored after all choices)."""
        run = F.fit(F.Cfg(**SETS[inp], static=tuple(F.STATIC), seed=seed, **config), D, ids, static_df=B)
        P = run.predict(D, static_df=B)
        val = F.evaluate(P, thr, "val")
        done[0] += 1
        print(f"training {done[0]:2d} of {n_total} ({purpose}): LSTM with {F.INPUT_LABEL[inp]}, hidden {config['hidden']}, lr {config['lr']:g}, seed {seed} | "
              f"best at update {run.best_update} of {run.updates_run}, {run.seconds:.0f} s | validation NSE {run.best_val_nse:.4f} (median by lead "
              + " ".join(f"{val['summary'][lk(k)]['nse_median']:.3f}" for k in F.LEADS) + ")", flush=True)
        return dict(P=P, val=val, cost=run.summary(), score=run.best_val_nse, curve=run.history)

    def train_all():
        """Steps 1 and 2 under the CPU lock: the setting check, then the four input sets with every seed."""
        t_ask = time.perf_counter()
        with F.cpu_lock(SCRIPT):
            t1 = time.perf_counter()
            # ---- 1. setting check on the validation years: past weather + river gauge, seed 0 -----------------------------
            check = [train("wq", config_of(c), 0, "setting check") for c in CANDIDATES]
            pick = int(np.argmax([r["score"] for r in check]))
            config = config_of(CANDIDATES[pick])
            print(f"chosen setting (highest validation NSE): {config}", flush=True)
            # ---- 2. the four input sets, three seeds each ---------------------------------------------------------------
            runs = {inp: {} for inp in SETS}
            for inp in ("wq", "w", "wf", "wqf"):
                for s in SEEDS:
                    runs[inp][s] = check[pick] if (inp, s) == ("wq", 0) else train(inp, config, s, "final model")
                    F.save_preds(f"lstm_{inp}_s{s}", runs[inp][s]["P"], dict(script=SCRIPT, model="lstm", inputs=inp, seeds=[s], config=config,
                                                                             what="one LSTM for all 12 basins, a single seed"))
            train_s = time.perf_counter() - t1
        setting = dict(input_set="wq", seed=0, criterion="mean validation-years NSE over 12 basins and 3 leads (higher is better)",
                       table=[dict(**config_of(c), n_params=r["cost"]["n_params"], best_update=r["cost"]["best_update"], updates_run=r["cost"]["updates_run"], seconds=r["cost"]["seconds"],
                                   val_nse_mean=r["score"], val_nse_median_by_lead=[r["val"]["summary"][lk(k)]["nse_median"] for k in F.LEADS], chosen=i == pick)
                              for i, (c, r) in enumerate(zip(CANDIDATES, check))],
                       chosen_val_nse_mean_by_seed=[r["score"] for r in runs["wq"].values()])
        return config, setting, list(SEEDS), runs, dict(n_trainings=done[0], training_seconds=round(train_s, 1), waited_for_cpu_s=round(t1 - t_ask, 1))

    def reload():
        """--rescore: no training. Forecasts come from the saved prediction sets; the training records come from the JSON of the training run."""
        old = json.loads((F.RES_OUT / "lstm_all.json").read_text())
        config, seeds, runs = old["config"], old["sets"]["wq"]["seeds"], {inp: {} for inp in SETS}
        for inp in SETS:
            for s in seeds:
                P, meta = F.load_preds(f"lstm_{inp}_s{s}"), F.preds_meta(f"lstm_{inp}_s{s}")
                assert meta["config"] == config and meta["script"] == SCRIPT, f"lstm_{inp}_s{s} is not from the training run described in lstm_all.json"
                runs[inp][s] = dict(P=P, val=F.evaluate(P, thr, "val"), cost=old["sets"][inp]["cost"][str(s)], curve=old["sets"][inp]["per_seed"][str(s)]["curve"])
        return config, old["setting_check"], seeds, runs, {k: old[k] for k in ("n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s")}

    config, setting, seeds, runs, timing = reload() if rescore else train_all()
    PF = {inp: F.ensemble([r["P"] for r in runs[inp].values()]) for inp in SETS}
    for inp, P in PF.items() if not rescore else ():                            # a rescore reads the prediction sets and leaves them untouched
        F.save_preds(f"lstm_{inp}", P, dict(script=SCRIPT, model="lstm", inputs=inp, seeds=seeds, config=config,
                                            what="one LSTM for all 12 basins, day-by-day average of the seeds"))

    # ---- 3. scores: every seed and the seed average, validation and test years -----------------------------------------
    notes = []
    EV = {inp: {p: F.evaluate(PF[inp], thr, p) for p in ("val", "test")} for inp in SETS}
    for inp in SETS:
        for s, r in runs[inp].items():
            r["test"] = F.evaluate(r["P"], thr, "test")
    seed_med = lambda inp, k, part="test": [r[part]["summary"][lk(k)]["nse_median"] for r in runs[inp].values()]
    seed_on = lambda inp, k, c: [r["test"]["summary"][lk(k)]["onset"][c] for r in runs[inp].values()]       # pooled onset score of every single seed

    def lo_hi(v, name):
        """Smallest and largest single-seed value (a seed that gave no warning has no false-warning share and is left out)."""
        v = [x for x in v if x == x] or [float("nan")]
        return {f"{name}_seed_min": float(min(v)), f"{name}_seed_max": float(max(v))}

    sets = {inp: dict(label=F.INPUT_LABEL[inp], saved_as=f"lstm_{inp}", seeds=seeds,
                      per_seed={str(s): dict(test=r["test"]["summary"], val=r["val"]["summary"], test_nse_by_basin={g: [r["test"]["basins"][g][lk(k)]["nse"] for k in F.LEADS] for g in ids},
                                             curve=r["curve"]) for s, r in runs[inp].items()},
                      test=EV[inp]["test"], val=EV[inp]["val"], cost={str(s): r["cost"] for s, r in runs[inp].items()}) for inp in SETS}

    # the baselines of 02_baselines.py (wait if another job is still writing them)
    t_wait = time.perf_counter()
    while not all(F.have_preds(n) for n in ALL_BASE) and time.perf_counter() - t_wait < (0 if F.QUICK else 1800):
        print(f"waiting for the baseline prediction sets of 02_baselines.py ({sum(F.have_preds(n) for n in ALL_BASE)} of {len(ALL_BASE)} there)", flush=True)
        time.sleep(30)
    timing["waited_for_baselines_s"] = round(time.perf_counter() - t_wait, 1)
    base = {}
    for n in ALL_BASE:
        if not F.have_preds(n):
            notes.append(f"Baseline {n} was not available; it is missing from the comparison and from the score table.")
            continue
        P, meta = F.load_preds(n), F.preds_meta(n)
        assert all(np.allclose(P[g].obs.loc[F.SPLIT["val"][0]:].values, D[g].q_mm.loc[F.SPLIT["val"][0]:].values) for g in ids), f"{n}: observed flow differs"
        base[n] = dict(model=meta.get("model", n.split("_")[1]), inputs=meta.get("inputs", ""), val=F.evaluate(P, thr, "val")["summary"], test=F.evaluate(P, thr, "test"))
    for name, summ in [(f"lstm_{inp}", EV[inp][p]["summary"]) for inp in SETS for p in ("val", "test")] + [(n, x) for n, b in base.items() for x in (b["val"], b["test"]["summary"])]:
        assert all(summ[lk(k)]["coverage_min"] == 1 for k in F.LEADS), f"{name}: days without a forecast - models would not be compared on identical days"

    # With the perfect weather forecast the LSTM reads the weather stamped on all three coming days at every lead (flood.Cfg.future);
    # the baselines of 02_baselines.py read it only up to the day being forecast. The two are like-for-like only at the longest lead.
    like_for_like = lambda inp, k: not (inp in ("wf", "wqf") and k < max(F.LEADS))
    comparison = {}
    for inp, names in COMPARE.items():
        names = [n for n in names if n in base]
        best = {lk(k): max(names, key=lambda n: base[n]["val"][lk(k)]["nse_median"]) for k in F.LEADS} if names else {}
        table = {}
        for n in names:
            table[n] = {}
            for k in F.LEADS:
                a, b, sm = nse_of(EV[inp]["test"], k), nse_of(base[n]["test"], k), seed_med(inp, k)
                bm, ls, bs = base[n]["test"]["summary"][lk(k)]["nse_median"], EV[inp]["test"]["summary"][lk(k)], base[n]["test"]["summary"][lk(k)]
                pr = paired(a, b)
                table[n][lk(k)] = dict(like_for_like=like_for_like(inp, k), lstm_nse_median=ls["nse_median"], baseline_nse_median=bm, difference_of_medians=ls["nse_median"] - bm,
                                       n_basins_lstm_higher=pr["n_basins_higher"], n_basins=pr["n_basins"], median_nse_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"],
                                       lstm_seed_median_min=min(sm), lstm_seed_median_max=max(sm),
                                       baseline_against_seed_spread="below every seed" if bm < min(sm) else "above every seed" if bm > max(sm) else "inside the seed spread",
                                       n_basins_lstm_higher_by_seed=[int((nse_of(r["test"], k) > b).sum()) for r in runs[inp].values()],
                                       **{f"{who}_{key}": s_["onset"][c] for who, s_ in (("lstm", ls), ("baseline", bs)) for key, c in (("onset_pod", "pod"), ("onset_far", "far"))},
                                       lstm_csi=ls["all"]["csi"], baseline_csi=bs["all"]["csi"])
        comparison[inp] = dict(label=F.INPUT_LABEL[inp], best_baseline_by_validation=best,
                               baseline_val_nse_median={n: [base[n]["val"][lk(k)]["nse_median"] for k in F.LEADS] for n in names}, baselines=table)

    # ---- 4. skill decomposition: what each extra source of information adds ----------------------------------------------
    levels = {inp: {lk(k): dict(nse_median=EV[inp]["test"]["summary"][lk(k)]["nse_median"], nse_seed_median_min=min(seed_med(inp, k)), nse_seed_median_max=max(seed_med(inp, k)),
                                **{f"onset_{c}": EV[inp]["test"]["summary"][lk(k)]["onset"][c] for c in ("pod", "far", "csi", "hits", "misses", "false_alarms")},
                                **lo_hi(seed_on(inp, k, "pod"), "onset_pod"), **lo_hi(seed_on(inp, k, "far"), "onset_far"),
                                **{f"all_{c}": EV[inp]["test"]["summary"][lk(k)]["all"][c] for c in ("pod", "far", "csi")}) for k in F.LEADS} for inp in SETS}
    steps = {}
    for name, (hi, lo) in STEPS.items():
        steps[name] = dict(what=f"'{F.INPUT_LABEL[hi]}' minus '{F.INPUT_LABEL[lo]}'", with_set=hi, without_set=lo)
        for k in F.LEADS:
            H, L = levels[hi][lk(k)], levels[lo][lk(k)]
            by_seed = [a - b for a, b in zip(seed_med(hi, k), seed_med(lo, k))]                    # seed s of one against seed s of the other
            pod_seed = [a - b for a, b in zip(seed_on(hi, k, "pod"), seed_on(lo, k, "pod"))]
            pr = paired(nse_of(EV[hi]["test"], k), nse_of(EV[lo]["test"], k))
            steps[name][lk(k)] = dict(nse_median_gain=H["nse_median"] - L["nse_median"], nse_median_gain_seed_min=min(by_seed), nse_median_gain_seed_max=max(by_seed),
                                      clear_of_seed_noise=bool(H["nse_seed_median_min"] > L["nse_seed_median_max"] or H["nse_seed_median_max"] < L["nse_seed_median_min"]),
                                      n_basins_better=pr["n_basins_higher"], median_of_basin_gains=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"],
                                      onset_pod_with=H["onset_pod"], onset_pod_without=L["onset_pod"], onset_pod_gain=H["onset_pod"] - L["onset_pod"],
                                      onset_pod_gain_seed_min=min(pod_seed), onset_pod_gain_seed_max=max(pod_seed),
                                      onset_far_with=H["onset_far"], onset_far_without=L["onset_far"], onset_far_change=H["onset_far"] - L["onset_far"])
    pc = lambda x: f"{100 * x:.0f}%" if x == x else "no warning given"
    findings = []
    for inp in comparison:
        for k in F.LEADS if comparison[inp]["baselines"] else ():               # nothing to compare with when the baseline sets are missing
            n = comparison[inp]["best_baseline_by_validation"][lk(k)]
            c = comparison[inp]["baselines"][n][lk(k)]
            findings.append(("" if c["like_for_like"] else "NOT like-for-like (here the LSTM reads the weather of all three coming days, the simple method only up to the day being forecast; no winner is called): ")
                            + f"LSTM against the best simple method, {F.INPUT_LABEL[inp]}, {days(k)}: median test NSE {c['lstm_nse_median']:.3f} (single seeds {c['lstm_seed_median_min']:.3f} to "
                            f"{c['lstm_seed_median_max']:.3f}) against {c['baseline_nse_median']:.3f} for {PLAIN[base[n]['model']].lower()} ({SPREAD[c['baseline_against_seed_spread']]}); the LSTM seed average is higher in {c['n_basins_lstm_higher']} of {c['n_basins']} basins, "
                            f"its single seeds in {min(c['n_basins_lstm_higher_by_seed'])} to {max(c['n_basins_lstm_higher_by_seed'])} "
                            f"(median difference {c['median_nse_difference']:+.3f}, Wilcoxon p = {c['wilcoxon_p']:.4f}).")
    for name, title in (("river_gauge_without_forecast", "The river gauge adds (no weather forecast)"), ("perfect_forecast_without_gauge", "A perfect weather forecast would add (no river gauge)"),
                        ("perfect_forecast_with_gauge", "A perfect weather forecast would add (with river gauge)")):
        for k in F.LEADS:
            s = steps[name][lk(k)]
            findings.append(f"{title}, {days(k)}: median test NSE {s['nse_median_gain']:+.3f} (seed by seed {s['nse_median_gain_seed_min']:+.3f} to {s['nse_median_gain_seed_max']:+.3f}; better in "
                            f"{s['n_basins_better']} of {len(ids)} basins); onsets warned {pc(s['onset_pod_without'])} -> {pc(s['onset_pod_with'])} "
                            f"(seed by seed {100 * s['onset_pod_gain_seed_min']:+.0f} to {100 * s['onset_pod_gain_seed_max']:+.0f} points), "
                            f"false warnings among onset warnings {pc(s['onset_far_without'])} -> {pc(s['onset_far_with'])}.")

    # ---- per-basin score table (test years) ---------------------------------------------------------------------------
    rows = []
    for name, inputs, ev in [(f"lstm_{inp}", inp, EV[inp]["test"]) for inp in SETS] + [(n, b["inputs"], b["test"]) for n, b in base.items()]:
        for g in ids:
            for k in F.LEADS:
                e = ev["basins"][g][lk(k)]
                rows.append(dict(gauge_id=g, short=F.SHORT[g], landscape=B.landscape[g], area_km2=round(float(B.area_gages2[g]), 1), model=name, inputs=inputs, lead=k,
                                 nse=e["nse"], kge=e["kge"], pod=e["all"]["pod"], far=e["all"]["far"], csi=e["all"]["csi"], onset_pod=e["onset"]["pod"], onset_far=e["onset"]["far"],
                                 hits=e["all"]["hits"], misses=e["all"]["misses"], false_alarms=e["all"]["false_alarms"]))
    T = pd.DataFrame(rows)
    T[["nse", "kge", "pod", "far", "csi", "onset_pod", "onset_far"]] = T[["nse", "kge", "pod", "far", "csi", "onset_pod", "onset_far"]].round(4)
    T.to_csv(F.RES_OUT / "lstm_all_scores.csv", index=False)

    # ---- notes and results file ---------------------------------------------------------------------------------------
    scores, wq_seed_scores = [r["val_nse_mean"] for r in setting["table"]], setting["chosen_val_nse_mean_by_seed"]
    gap, spread = max(scores) - sorted(scores)[-2], max(wq_seed_scores) - min(wq_seed_scores)
    costs = [(f"setting check with hidden {r['hidden']}, lr {r['lr']:g}", r) for r in setting["table"] if not r["chosen"]] \
        + [(f"{F.INPUT_LABEL[inp]}, seed {s_}", r["cost"]) for inp in SETS for s_, r in runs[inp].items()]
    place = lambda v, lo, hi: "above" if v > hi else "below" if v < lo else "inside"
    where = {c: [place(L[f"onset_{c}"], L[f"onset_{c}_seed_min"], L[f"onset_{c}_seed_max"]) for L in (levels[i][lk(k)] for i in SETS for k in F.LEADS)] for c in ("pod", "far")}
    at_limit = [f"{n}: best at update {c['best_update']}" for n, c in costs if c["best_update"] > config["updates"] - config["patience"] * config["eval_every"]]      # the stopping rule could not be met
    notes += [
        "One LSTM (one layer) is trained on all 12 basins together; the ten basin properties in flood.STATIC are repeated on every day as extra inputs. The baselines are one model per basin and lead.",
        "Three of the ten basin properties (p_mean, aridity, p_seasonality: mean rain, dryness and rain season) are the CAMELS climate averages of October 1989 to September 2009 (stated in "
        "camels_attributes_v2.0.xlsx), a period that overlaps the validation years and four test years. Each is one fixed number per basin, not daily data, and no property comes from the flow record, "
        "but strictly they are not training-years-only information.",
        "Hidden size is capped at 64 because of this laptop (2 CPU cores, no GPU); larger models were not tried. Only hidden size and learning rate were checked; dropout, history length, "
        "batch and training budget are the engine defaults.",
        f"Setting check: one seed per candidate, chosen by the mean validation-years NSE over the 12 basins and 3 leads (the score the trainer itself uses for early stopping). The chosen candidate leads the "
        f"runner-up by {gap:.4f}; the {len(seeds)} seeds of the chosen setting differ by {spread:.4f} in the same score, so the choice is "
        + ("larger than the seed noise seen here." if gap > spread else "within seed noise: the candidates are practically equivalent and the pick is not a finding.") if len(seeds) > 1 and len(scores) > 1 else
        "Setting check: one seed per candidate, chosen by the mean validation-years NSE over the 12 basins and 3 leads.",
        f"Training budget: at most {config['updates']} updates, a validation check every {config['eval_every']}, stop after {config['patience']} checks without a better score. "
        f"{len(costs) - len(at_limit)} of the {len(costs)} trainings were stopped by that rule; {len(at_limit)} reached the update limit before the rule was met"
        + (f" ({'; '.join(at_limit)}): a longer budget might have improved them." if at_limit else "."),
        "Perfect weather forecast, LSTM against simple methods: like-for-like only at 3 days ahead (key like_for_like in the comparison). The LSTM reads the weather stamped on all three coming days "
        "at every lead; the simple methods of 02_baselines.py read it only up to the day being forecast. Because the rain table stamps rain by observer day (see 10_rain_timing.py), rain stamped one day "
        "after the target day still tells about the target day, so at 1 and 2 days ahead the LSTM knows more than the simple methods and no winner should be called there. The comparison of the LSTMs "
        "with each other (what the river gauge adds, what the weather of the coming days adds) is not affected.",
        "Fair test: scalers, thresholds and weights use the training years 1980-2000 only; the setting, every stopping point and the 'best' baseline were chosen on the validation years 2000-2005; "
        "the test years 2005-2014 were scored only after all choices were made and chose nothing. The setting-check table holds no test score.",
        "lstm_<set> is the day-by-day average of the forecasts of the seeds listed. An average of several models usually scores a little higher than each single one, so the per-seed medians "
        "(lstm_seed_median_min / max) and the basins won by each single seed (n_basins_lstm_higher_by_seed) are given next to every comparison. A baseline 'inside the seed spread' means that one "
        "training run of the LSTM could land on either side of it: read that as no clear difference.",
        f"Warnings of the seed average against its single seeds, over the {len(where['pod'])} combinations of input set and lead (test years, onset warnings, pooled): its share of onsets warned is above "
        f"every single seed in {where['pod'].count('above')}, inside their range in {where['pod'].count('inside')} and below every seed in {where['pod'].count('below')}; its share of false warnings is below "
        f"every seed in {where['far'].count('below')}, inside in {where['far'].count('inside')} and above in {where['far'].count('above')}. onset_pod_seed_min / max and onset_far_seed_min / max in "
        "decomposition.levels give the single-seed range next to the value of the seed average.",
        "All models are compared on identical days: coverage is 1.0 for every model, lead and basin in the validation and test years (asserted in the code).",
        "Wilcoxon signed-rank test, two-sided, over the 12 basins (LSTM seed average against the baseline). With 12 basins the smallest possible p is 0.0005; the basins are not independent "
        "(neighbours share storms) and many comparisons are made without correction, so read p as a description, not a proof.",
        "median_nse_difference is the median over basins of (LSTM minus baseline); difference_of_medians is the median LSTM NSE minus the median baseline NSE. They need not agree.",
        "Perfect weather forecast (wf, wqf) = the actual weather of the target days, an upper bound that cannot be had in practice. The comparison there is close but not exactly like-for-like: the LSTM "
        "reads all six weather variables of days t+1..t+3 at every lead, the baselines read rain and mean temperature of days t+1..t+k for lead k. At 3 days ahead both see the same days.",
        "Skill decomposition: 'gain' = difference of the median test NSE of two LSTMs (seed averages) that differ in one source of information; nse_median_gain_seed_min / max repeat it seed by seed "
        "(seed s against seed s). clear_of_seed_noise is true when the per-seed medians of the two models do not overlap.",
        "Warning scores are pooled over the 12 basins. High flow = above the basin's 95th percentile of the training years; onset = only forecasts issued while the river was still at or below that level. "
        "onset_far is the share of false warnings among the warnings issued in that situation.",
        "lstm_all_scores.csv: test years, one row per basin, lead and model. gauge_id is text with a leading zero - read it with dtype=str. model is the name of the prediction set; inputs: w / wq / wf / wqf as in flood.INPUT_LABEL, q = river gauge only, calendar = no live data. "
        "pod, far, csi, hits, misses, false_alarms count all high-flow days; onset_pod and onset_far only the forecasts issued before the river was high. far is empty where a model gave no warning in that basin.",
    ] + (["QUICK smoke test: a few updates, two seeds, two candidates - numbers are not results."] if F.QUICK else []) \
      + (["This file was written by --rescore: every score, table and note was recomputed from the saved prediction sets lstm_<set>_s<seed> without training again; the setting-check table, "
          "cost, training curves and the timings (n_trainings, training_seconds, waited_for_cpu_s, runtime_s) are those of the training run. rescore_runtime_s is the time of this pass."] if rescore else [])
    if rescore:
        timing["rescore_runtime_s"] = round(time.perf_counter() - t0, 1)
    else:
        timing["runtime_s"] = round(time.perf_counter() - t0 - timing["waited_for_cpu_s"] - timing["waited_for_baselines_s"], 1)
    R = dict(split=F.SPLIT, quick_mode=F.QUICK, thresholds=thr, static_inputs=list(F.STATIC),
             basins={g: dict(short=F.SHORT[g], landscape=B.landscape[g], area_km2=float(B.area_gages2[g])) for g in ids},
             setting_check=setting, config=config, sets=sets, baselines=base, comparison=comparison, decomposition=dict(levels=levels, steps=steps), findings=findings,
             score_table="results/lstm_all_scores.csv", notes=notes, **timing)
    (F.RES_OUT / "lstm_all.json").write_text(json.dumps(R, indent=1))
    return R


def fig_leads(R):
    """2 x 2 panels (rows: without / with river gauge; columns: without / with a perfect forecast): LSTM against the best simple method."""
    ids = list(R["basins"])
    nse = lambda ev, k: np.array([ev["basins"][g][lk(k)]["nse"] for g in ids])
    lstm = {inp: R["sets"][inp]["test"] for inp in SETS}
    best = {inp: R["comparison"][inp]["best_baseline_by_validation"] for inp in SETS}
    low = min(min(nse(lstm[inp], k).min(), nse(R["baselines"][best[inp][lk(k)]]["test"], k).min()) for inp in SETS for k in F.LEADS)
    lo = max(-0.5, np.floor(low * 4) / 4)
    below = sum(int((nse(lstm[inp], k) < lo).sum() + (nse(R["baselines"][best[inp][lk(k)]]["test"], k) < lo).sum()) for inp in SETS for k in F.LEADS)
    ref = [lstm["w"]["summary"][lk(k)]["nse_median"] for k in F.LEADS]
    title = {"w": "Past weather only", "wf": "+ perfect 3-day weather forecast", "wq": "+ river gauge", "wqf": "+ river gauge + perfect forecast"}
    off, used = 0.15, []

    seeds = lambda inp, k: [p["test"][lk(k)]["nse_median"] for p in R["sets"][inp]["per_seed"].values()]
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.7), sharex=True, sharey=True)
    for inp, ax in zip(("w", "wf", "wq", "wqf"), axes.ravel()):
        if inp != "w":
            ax.plot(np.array(F.LEADS) + off, ref, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
        bm, lm = [], []
        for k in F.LEADS:
            n = best[inp][lk(k)]
            fam = R["baselines"][n]["model"]
            used += [fam] if fam not in used else []
            vb, vl = nse(R["baselines"][n]["test"], k), nse(lstm[inp], k)
            ax.plot(np.full(len(vb), k - off), np.clip(vb, lo, None), ls="none", marker="o", ms=3, markeredgewidth=0, color=F.MODEL_COL[fam], alpha=.5)
            ax.plot(np.full(len(vl), k + off), np.clip(vl, lo, None), ls="none", marker="o", ms=3, markeredgewidth=0, color=F.MODEL_COL["lstm"], alpha=.4)
            bm.append((fam, np.median(vb))); lm.append(np.median(vl))
            c = R["comparison"][inp]["baselines"][n][lk(k)]
            ax.text(k, 0.03, f"{c['n_basins_lstm_higher']} of {len(ids)}" + ("" if c.get("like_for_like", True) else " *"), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, color=INK if c.get("like_for_like", True) else MUTED)
        ax.plot(np.array(F.LEADS) - off, [m for _, m in bm], color=MUTED, lw=1.2, zorder=3)
        for k, (fam, m) in zip(F.LEADS, bm):
            ax.plot(k - off, m, ls="none", marker=MARK[fam], ms=7.5, color=F.MODEL_COL[fam], markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)
        ax.plot(np.array(F.LEADS) + off, lm, "-o", color=F.MODEL_COL["lstm"], lw=1.8, ms=7.5, markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=5)
        for k in F.LEADS:                                                       # how far the single seeds lie apart, drawn beside the marker
            ax.plot([k + 2.1 * off] * 2, [min(seeds(inp, k)), max(seeds(inp, k))], color=INK, lw=1.3, marker="_", ms=5, markeredgewidth=1.3, solid_capstyle="butt", zorder=6)
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        ax.set_title(title[inp], fontsize=9); ax.tick_params(labelsize=8, length=0); ax.grid(axis="x", visible=False)
        ax.set_xlim(0.5, 3.6); ax.set_xticks(F.LEADS, [days(k) for k in F.LEADS]); ax.tick_params(axis="x", labelcolor=INK2)
    span = 1.0 - lo
    axes[0, 0].set_ylim(lo - 0.17 * span, 1.03); axes[0, 0].set_yticks(np.arange(lo, 1.01, 0.25))
    for ax in axes[:, 0]:
        ax.set_ylabel("NSE in the test years (1 = perfect)", fontsize=8.5)
    h = [Line2D([], [], color=F.MODEL_COL["lstm"], lw=1.8, marker="o", ms=7, markeredgecolor=SURFACE, label="LSTM, one model for all 12 basins")]
    h += [Line2D([], [], color=MUTED, lw=1.2, marker=MARK[f], ms=7, markerfacecolor=F.MODEL_COL[f], markeredgecolor=SURFACE, label=f"{PLAIN[f]}, one per basin") for f in MARK if f in used]
    n_seeds = len(R["sets"]["w"]["seeds"])
    h += [Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 3)), label="LSTM with past weather only (top left), repeated"),
          Line2D([], [], color=INK, lw=1.3, marker="_", ms=5, markeredgewidth=1.3, ls="none", label=f"Range of the {n_seeds} single LSTM seeds (median)")]
    rows = (len(h) + 1) // 2
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.088), columnspacing=2.0, handlelength=2.6)
    note = [f"Small dots: one basin each. Large markers: median of the {len(ids)} basins. Numbers in each panel: basins in which the LSTM scores higher.",
            "The simple model shown is the one with the best validation-years score among those given the same inputs."
            + (f" {below} basin scores below {lo:g} are drawn at {lo:g}." if below else ""),
            "* not like-for-like: there the LSTM reads the weather of all three coming days, the simple model only up to the day being forecast."]
    fig.text(0.5, 0.008, "\n".join(note), ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.4)
    fig.suptitle(f"One LSTM for {len(ids)} basins against the best simple model given the same inputs", fontsize=10.2, fontweight="bold", color=INK, x=0.012, y=0.988, ha="left")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.918, bottom=0.135 + 0.029 * rows, hspace=0.2, wspace=0.05)
    fig.savefig(F.FIG_OUT / "lstm_all.png", dpi=220)
    plt.close(fig)


def fig_basins(R):
    """Per basin, 1 day ahead: LSTM with river gauge against 'tomorrow = today' and the random forest with the same inputs."""
    S, lstm = R["sets"]["wq"], R["sets"]["wq"]["test"]["basins"]
    others = {n: R["baselines"][n]["test"]["basins"] for n in ("base_persistence", "base_rf_wq") if n in R["baselines"]}
    v = {g: dict(lstm=lstm[g]["l1"]["nse"], seeds=[p["test_nse_by_basin"][g][0] for p in S["per_seed"].values()], **{n: e[g]["l1"]["nse"] for n, e in others.items()}) for g in R["basins"]}
    low = min(min(x["lstm"], min(x["seeds"]), *[x[n] for n in others]) for x in v.values())
    lo = max(-0.5, np.floor(low * 10) / 10 - 0.05)
    clip = lambda x: max(x, lo)
    fig, ax = plt.subplots(figsize=(6.7, 5.3))
    y, ticks = 0, {}
    for land, col in F.LAND_COL.items():
        grp = sorted([g for g, b in R["basins"].items() if b["landscape"] == land], key=lambda g: -v[g]["lstm"])
        if not grp:
            continue
        ax.plot(0.012, y, marker="s", ms=6.5, color=col, transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
        ax.text(0.03, y, F.LAND_SHORT[land], transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.3, fontweight="bold", color=INK)
        y += 1
        for g in grp:
            x = v[g]
            allv = [clip(x["lstm"])] + [clip(x[n]) for n in others]
            ax.plot([min(allv), max(allv)], [y, y], color=AXIS, lw=1.2, zorder=2)
            ax.plot([clip(min(x["seeds"])), clip(max(x["seeds"]))], [y, y], color=col, lw=6.5, alpha=.3, solid_capstyle="butt", zorder=3)
            if "base_rf_wq" in others:                                           # a frame wider than the LSTM dot, so equal scores show as a dot inside a square
                ax.plot(clip(x["base_rf_wq"]), y, marker=MARK["rf"], ms=9.5, markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.5, zorder=4)
            ax.plot(clip(x["lstm"]), y, marker="o", ms=6.8, color=col, markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=5)
            if "base_persistence" in others:                                     # see-through where it sits on another marker
                alone = all(abs(clip(x["base_persistence"]) - clip(x[n])) > 0.035 for n in ["lstm"] + [n for n in others if n != "base_persistence"])
                ax.plot(clip(x["base_persistence"]), y, marker=MARK["persistence"], ms=6.2, markerfacecolor=SURFACE if alone else "none", markeredgecolor=INK, markeredgewidth=1.2, zorder=6)
            ticks[y] = R["basins"][g]["short"]
            y += 1
        y += 0.5
    ax.set_ylim(y - 0.9, -0.7); ax.set_yticks(list(ticks), list(ticks.values())); ax.tick_params(labelsize=8.3, length=0); ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(lo, 1.0); ax.grid(axis="y", visible=False); ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_xlabel("NSE in the test years, 1 day ahead (1 = perfect)", fontsize=8.5)
    fig.suptitle("River flow 1 day ahead, basin by basin: one LSTM against two simple models", fontsize=10.2, fontweight="bold", color=INK, x=0.012, y=0.985, ha="left")
    h = [Line2D([], [], ls="none", marker="o", ms=6.8, color=INK2, markeredgecolor=SURFACE, label="LSTM, one model for all 12 basins\n(past weather + river gauge)"),
         Line2D([], [], ls="none", marker=MARK["rf"], ms=9.5, markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.5, label="Random forest, one per basin\n(same inputs)"),
         Line2D([], [], ls="none", marker=MARK["persistence"], ms=6.2, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=1.2, label="'Tomorrow = today'\n(river gauge only)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.082), columnspacing=2.0, handletextpad=.5)
    n = len(v)
    win = {k: sum(x["lstm"] > x[k] for x in v.values()) for k in others}
    top = sum(all(x["lstm"] > x[k] for k in others) for x in v.values())
    note = [f"Colour = landscape. Shaded bar behind the LSTM dot: range of its {len(S['seeds'])} single seeds (the dot is their average forecast)."
            + (f" Scores below {lo:g} are drawn at {lo:g}." if low < lo else ""),
            f"The LSTM scores higher than 'tomorrow = today' in {win.get('base_persistence', 0)} of {n} basins, than the random forest in {win.get('base_rf_wq', 0)} of {n}, than both in {top} of {n}."]
    fig.text(0.5, 0.012, "\n".join(note), ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.subplots_adjust(left=0.19, right=0.975, top=0.925, bottom=0.265)
    fig.savefig(F.FIG_OUT / "lstm_all_basins.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    R = json.loads((F.RES_OUT / "lstm_all.json").read_text()) if "--figure" in sys.argv else run("--rescore" in sys.argv)
    R = json.loads(json.dumps(R))                                               # the figures read exactly what the JSON holds
    if all(R["comparison"][inp]["baselines"] for inp in SETS):
        fig_leads(R)
    else:
        print("lstm_all.png was not drawn: it needs a baseline for every input set (run 02_baselines.py, then this script with --rescore)")
    fig_basins(R)
    print(f"\nchosen setting {R['config']}; {R['n_trainings']} trainings in {R['training_seconds']:.0f} s")
    for inp in SETS:
        print(f"  lstm_{inp:3s} median test NSE by lead " + " ".join(f"{R['decomposition']['levels'][inp][lk(k)]['nse_median']:.3f}" for k in F.LEADS)
              + "   onsets warned " + " ".join(f"{R['decomposition']['levels'][inp][lk(k)]['onset_pod']:.2f}" for k in F.LEADS))
    print("\n".join(R["findings"]))
    print("notes:", *R["notes"], sep="\n  ")
