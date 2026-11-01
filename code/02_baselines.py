"""Week 1 - baselines: how well do simple methods predict the flow 1-3 days ahead? This is the number every later model must beat.

For each of the 12 basins separately and each lead: 'tomorrow = today', the calendar-day average of the training years, and a
ridge regression and a random forest on hand-made features (features.py) for four sets of inputs - past weather, + river
gauge, + a perfect weather forecast, + both. Everything is fitted on the training years; the ridge strength and the forest's
leaf size are chosen on the validation years; the test years are only scored (shared scoring in flood.py).
Writes results/baselines.json, figures/baselines.png and ten prediction sets results/preds/base_*.npz.
Redraw the figure from the saved JSON without refitting: python code/02_baselines.py --figure
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import flood
from common import *
from features import feature_table

use_style()
SCRIPT = "02_baselines.py"
SETS = {"w": (False, False), "wq": (True, False), "wf": (False, True), "wqf": (True, True)}      # (river gauge, perfect weather forecast)
ALPHAS = (10, 100) if flood.QUICK else (1, 10, 100, 1000)           # ridge strength (features are standardised first)
LEAVES = (10,) if flood.QUICK else (3, 10, 30)                        # forest: smallest number of days in a leaf
TREES, JOBS = (8, 1) if flood.QUICK else (150, 4)
NO_FC = ["base_persistence", "base_climatology", "base_linreg_w", "base_linreg_wq", "base_rf_w", "base_rf_wq"]
WITH_FC = ["base_linreg_wf", "base_linreg_wqf", "base_rf_wf", "base_rf_wqf"]
LABEL = {"base_persistence": "'Tomorrow = today' (river gauge only)", "base_climatology": "Calendar-day average (no live data)",
         "linreg": "Ridge regression", "rf": "Random forest"}


def climatology(d):
    """Mean training-years flow of each calendar day, smoothed over +-15 days, looked up for the target day of each lead."""
    tr = d.q_mm.loc[flood.SPLIT["train"][0]:flood.SPLIT["train"][1]]
    total, count = np.bincount(tr.index.dayofyear, tr.values, 367)[1:], np.bincount(tr.index.dayofyear, minlength=367)[1:]
    around = lambda v: np.convolve(np.tile(v, 3), np.ones(31), "same")[366:732]                 # 31-day sum, wrapping around New Year
    c = around(total) / around(count)
    return {k: c[(d.index + pd.Timedelta(days=k)).dayofyear - 1] for k in flood.LEADS}


def design(s, D):
    """Per basin and lead: the feature rows, which of them are complete, which are used for fitting, and the flow to predict."""
    T = {}
    for g, d in D.items():
        for k in flood.LEADS:
            X = feature_table(d, SETS[s][0], k if SETS[s][1] else 0)
            ok = X.notna().all(axis=1).values
            T[g, k] = X.values, ok, ok & flood.part_mask(d.index, k, "train"), d.q_mm.shift(-k).values       # fit: target day in the training years
    return T


def forecast(make, T, D):
    """One model per basin and lead, fitted on the training years; forecasts every issue day that has features."""
    P = {}
    for g, d in D.items():
        out = {}
        for k in flood.LEADS:
            X, ok, fit, y = T[g, k]
            m = make().fit(X[fit], y[fit])
            out[k] = np.full(len(d), np.nan)
            out[k][ok] = np.clip(m.predict(X[ok]), 0, None)                                     # a river cannot have negative flow
        P[g] = flood.frame_from_issue(d, out)
    return P


def tuned(make, options, s, D, thr, what):
    """Try each setting; keep the one with the best validation-years NSE (mean over basins and leads, as the LSTM trainer does)."""
    T, tried = design(s, D), {}
    for o in options:
        t = time.perf_counter()
        P = forecast(lambda: make(o), T, D)
        E = flood.evaluate(P, thr, "val")["basins"]
        tried[o] = (float(np.mean([E[g][f"l{k}"]["nse"] for g in P for k in flood.LEADS])), P)
        print(f"  {what} [{flood.INPUT_LABEL[s]}] setting {o}: validation NSE {tried[o][0]:.4f}  ({time.perf_counter() - t:.0f} s)", flush=True)
    best = max(tried, key=lambda o: tried[o][0])
    return tried[best][1], best, {str(o): round(v[0], 4) for o, v in tried.items()}


def run():
    t0 = time.perf_counter()
    D, B = flood.load_data(), flood.basins()
    thr = flood.thresholds(D)
    M = {"base_persistence": dict(P={g: flood.frame_from_issue(d, {k: d.q_mm.values for k in flood.LEADS}) for g, d in D.items()}, model="persistence",
                                  inputs="q", settings={"rule": "forecast for day t+k = flow of day t"}),
         "base_climatology": dict(P={g: flood.frame_from_issue(d, climatology(d)) for g, d in D.items()}, model="climatology", inputs="calendar",
                                  settings={"rule": "mean training-years flow of the target's calendar day, 31-day window"})}
    t_ask = time.perf_counter()
    with flood.cpu_lock(SCRIPT):
        t1 = time.perf_counter()
        for s in SETS:
            P, a, tried = tuned(lambda a: make_pipeline(StandardScaler(), Ridge(alpha=a)), ALPHAS, s, D, thr, "ridge regression")
            M[f"base_linreg_{s}"] = dict(P=P, model="linreg", inputs=s, settings=dict(ridge_alpha=a, standardised=True, validation_nse_by_alpha=tried))
        for s in SETS:
            P, leaf, tried = tuned(lambda leaf: RandomForestRegressor(TREES, min_samples_leaf=leaf, max_features=1 / 3, n_jobs=JOBS, random_state=0),
                                   LEAVES, s, D, thr, "random forest")
            M[f"base_rf_{s}"] = dict(P=P, model="rf", inputs=s, settings=dict(n_trees=TREES, min_samples_leaf=leaf, max_features="one third", random_state=0,
                                                                              validation_nse_by_leaf=tried))
        fit_s = time.perf_counter() - t1
    for name, m in M.items():
        m.update(uses_river_gauge=m["inputs"] in ("q", "wq", "wqf"), uses_weather_forecast=m["inputs"] in ("wf", "wqf"),
                 val=flood.evaluate(m["P"], thr, "val"), test=flood.evaluate(m["P"], thr, "test"))
        assert all(m[p]["summary"][f"l{k}"]["coverage_min"] == 1 for p in ("val", "test") for k in flood.LEADS), f"{name}: days without a forecast"
        flood.save_preds(name, m.pop("P"), dict(script=SCRIPT, model=m["model"], inputs=m["inputs"], settings=m["settings"]))

    def row(name, k, part="test"):
        s = M[name][part]["summary"][f"l{k}"]
        return dict(model=name, inputs=M[name]["inputs"], lead=k, **{c: s[c] for c in ("nse_median", "nse_mean", "nse_min")},
                    **{c: s["all"][c] for c in ("pod", "far", "csi")}, **{f"onset_{c}": s["onset"][c] for c in ("pod", "far", "csi")})

    def best(names, k):
        n = max(names, key=lambda m: M[m]["val"]["summary"][f"l{k}"]["nse_median"])
        return dict(model=n, val_nse_median=M[n]["val"]["summary"][f"l{k}"]["nse_median"], test=row(n, k))

    # like-for-like: the baselines that use no more information than an input set offers
    allowed = {"w": ["base_climatology", "base_linreg_w", "base_rf_w"], "wq": NO_FC,
               "wf": ["base_climatology", "base_linreg_w", "base_rf_w", "base_linreg_wf", "base_rf_wf"], "wqf": NO_FC + WITH_FC}
    share = lambda g, part: float((D[g].q_mm.loc[flood.SPLIT[part][0]:flood.SPLIT[part][1]] > thr[g]).mean())
    never = [n for n in M if all(M[n]["test"]["summary"][f"l{k}"]["all"]["hits"] + M[n]["test"]["summary"][f"l{k}"]["all"]["false_alarms"] == 0 for k in flood.LEADS)]
    edge = [n for n, m in M.items() for v, options in ((m["settings"].get("ridge_alpha"), ALPHAS), (m["settings"].get("min_samples_leaf"), LEAVES))
            if len(options) > 1 and v in (min(options), max(options))]                          # the chosen setting sits at an end of its list
    names = feature_table(next(iter(D.values())), True, 3).columns
    R = dict(split=flood.SPLIT, thresholds=thr,
             features={s: [c for c in names if (q or not c.startswith("flow")) and (f or "ahead" not in c)] for s, (q, f) in SETS.items()},
             models=M, summary_table=[row(n, k) for n in M for k in flood.LEADS],
             best_by_validation={f"l{k}": dict(without_weather_forecast=best(NO_FC, k), with_perfect_forecast=best(WITH_FC, k)) for k in flood.LEADS},
             best_by_validation_same_information={s: {f"l{k}": best(c, k) for k in flood.LEADS} for s, c in allowed.items()},
             high_flow={g: dict(basin=flood.SHORT[g], landscape=B.landscape[g], threshold_mm_day=thr[g], **{f"share_{p}_days_above": share(g, p) for p in flood.SPLIT})
                        for g in D},
             notes=["Every model is fitted per basin and per lead on issue days whose target day is in the training years; nothing is refitted on training + validation.",
                    "Ridge strength and forest leaf size: one value per input set, the one with the highest mean validation-years NSE over the 12 basins and 3 leads. "
                    "Test years were never used to choose anything.",
                    "Input codes: q = river gauge only, calendar = no live data, w / wq / wf / wqf as in flood.INPUT_LABEL. For wf and wqf the forecast for day t+k "
                    "gets the actual rain and mean temperature of days t+1..t+k - a perfect weather forecast, an upper bound that cannot be had in practice.",
                    "best_by_validation picks by median validation NSE; its test scores are the number to beat. best_by_validation_same_information does the same "
                    "among the baselines that use no more information than each input set (like-for-like).",
                    "Warning scores (pod, far, csi) are pooled over the 12 basins; a warning is a forecast above the basin's training-years 95th percentile. "
                    "far is NaN where a model gives no warning at all." + (f" No warning in the test years at any lead: {', '.join(never)}." if never else ""),
                    "high_flow: 5% of training days are above the threshold by construction; the validation and test shares show how wetter or drier later years were.",
                    "The forest is random but repeatable (random_state=0); one forest per basin, lead and input set. Only this one seed was run, so the "
                    "forest's seed-to-seed spread is not known."]
                   + ([f"The chosen setting is the smallest or largest value of its list for: {', '.join(edge)} (settings holds every value's validation score). "
                       "A value outside the list might do a little better, so these models may be slightly under-tuned and their scores are, if anything, a little low."]
                      if edge else [])
                   + (["QUICK smoke test: tiny forests and short setting lists - numbers are not results."] if flood.QUICK else []),
             fit_seconds=round(fit_s, 1), waited_for_cpu_s=round(t1 - t_ask, 1), runtime_s=round(time.perf_counter() - t0 - (t1 - t_ask), 1))
    (flood.RES_OUT / "baselines.json").write_text(json.dumps(R, indent=1))
    return R


def figure(R):
    """Left: test-years NSE by lead (each basin a small dot, the median a large marker). Right: pooled high-flow warning scores by lead."""
    M = R["models"]
    col = lambda n: flood.MODEL_COL[M[n]["model"]]
    solid = lambda n: M[n]["uses_river_gauge"] or M[n]["model"] == "climatology"                # weather-only models: hollow marker, dashed line
    look = lambda n: dict(marker="s" if M[n]["model"] == "climatology" else "o", markerfacecolor=col(n) if solid(n) else SURFACE,
                          markeredgecolor=MUTED if M[n]["model"] == "climatology" else col(n), markeredgewidth=1.6)
    line = lambda n: dict(color=col(n), lw=1.8, ls="-" if solid(n) else (0, (3.5, 2)), ms=6, **look(n))
    slot = dict(zip(NO_FC, np.linspace(-0.33, 0.33, len(NO_FC))))                               # the same model family keeps its place in both panels
    slot.update({n: slot[n[:-1]] for n in WITH_FC})                                             # base_rf_wqf sits where base_rf_wq sits
    nse_of = lambda n, k: np.array([b[f"l{k}"]["nse"] for b in M[n]["test"]["basins"].values()])
    lo = max(-1.5, np.floor(min(nse_of(n, k).min() for n in M for k in flood.LEADS) * 4) / 4)   # axis floor; basins below it are counted in the note
    below = sum(int((nse_of(n, k) < lo).sum()) for n in M for k in flood.LEADS)

    fig, ax = plt.subplots(2, 2, figsize=(6.7, 6.5), sharex=True)
    for a, group, title in ((ax[0, 0], NO_FC, "Accuracy without a weather forecast"), (ax[1, 0], WITH_FC, "Accuracy with a perfect weather forecast")):
        for n in group:
            for k in flood.LEADS:
                v = nse_of(n, k)
                a.plot(np.full(len(v), k + slot[n]), np.clip(v, lo, None), ls="none", marker="o", ms=2.8, markeredgewidth=0,
                       **(dict(color=MUTED, alpha=.3) if M[n]["model"] == "climatology" else dict(color=col(n), alpha=.45)))
                a.plot(k + slot[n], np.median(v), ls="none", ms=7, zorder=4, **look(n))
        a.axhline(0, color=MUTED, lw=1, zorder=1)
        a.set_ylim(lo - 0.04, 1.04); a.set_yticks(np.arange(lo, 1.01, 0.25)); a.set_xlim(0.5, 3.5)
        a.set_ylabel("NSE in the test years (1 = perfect)", fontsize=8.5); a.set_title(title, fontsize=9)
    ax[1, 0].text(0.5, 0.07, "An upper bound: in practice nobody knows\nthe coming days' rain exactly.", transform=ax[1, 0].transAxes, ha="center", va="bottom",
                  fontsize=7.5, color=INK2, linespacing=1.4)
    for a, key, title, hint, y in ((ax[0, 1], "pod", "High-flow days that were warned", "higher is better", 0.97),
                                   (ax[1, 1], "far", "Warnings that were false", "lower is better", 0.03)):
        for n in NO_FC:                                                                         # drawn side by side, so that equal scores stay visible
            a.plot(np.array(flood.LEADS) + 0.5 * slot[n], [100 * M[n]["test"]["summary"][f"l{k}"]["all"][key] for k in flood.LEADS], **line(n))
        a.set_ylim(-4, 104); a.set_yticks(range(0, 101, 25)); a.set_ylabel("test years, 12 basins together (%)", fontsize=8.5); a.set_title(title, fontsize=9)
        a.text(0.985, y, hint, transform=a.transAxes, ha="right", va="top" if y > 0.5 else "bottom", fontsize=7.5, color=INK2)
    for a in ax.ravel():
        a.tick_params(labelsize=8, length=0); a.grid(axis="x", visible=False)
        a.set_xticks(flood.LEADS, ["1 day\nahead", "2 days\nahead", "3 days\nahead"])
    text = {n: LABEL.get(n) or f"{LABEL[M[n]['model']]}, weather" + (" + river gauge" if M[n]["uses_river_gauge"] else " only") for n in NO_FC}
    order = [n for n in NO_FC if not M[n]["uses_river_gauge"]] + [n for n in NO_FC if M[n]["uses_river_gauge"]]      # left column: no gauge; right: gauge
    fig.legend([Line2D([], [], **line(n)) for n in order], [text[n] for n in order], loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.075), columnspacing=2.2, handlelength=2.8)
    silent = [n for n in NO_FC if all(M[n]["test"]["summary"][f"l{k}"]["all"]["hits"] + M[n]["test"]["summary"][f"l{k}"]["all"]["false_alarms"] == 0 for k in flood.LEADS)]
    note = ["Small dots: one basin each. Large markers: median of the 12 basins. High flow: above the level exceeded on 5% of training days."]
    note += ["The calendar-day average never gives a warning, so it has no share of false warnings."] if "base_climatology" in silent else []
    note += [f"{below} basin scores below {lo:g} are drawn at the bottom edge."] if below else []
    fig.text(0.5, 0.012, "\n".join(note), ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.tight_layout(rect=(0, 0.185 + 0.02 * (len(note) - 2), 1, 1), h_pad=1.4, w_pad=1.0)
    fig.savefig(flood.FIG_OUT / "baselines.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    R = json.loads((flood.RES_OUT / "baselines.json").read_text()) if "--figure" in sys.argv else run()
    figure(json.loads(json.dumps(R)))                                                           # the figure reads exactly what the JSON holds
    T = pd.DataFrame(R["summary_table"]).round(3)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    print(json.dumps(R["best_by_validation"], indent=1))
