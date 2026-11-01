"""When does the rain in the data fall? A check of the day stamps, and what it means for the forecasts.

The weather table used everywhere (Daymet) appears to stamp rain by the day of the rain-gauge observers: the rain it stamps "day d"
matches what two other rain products hold for the days d-1 and d, as if it fell roughly between the morning of day d-1 and the
morning of day d (an inference from the correlations of part 1; it holds least in Myakka, FL). If so, at the end of day t, when a
forecast is issued, the rain of the last half day and more has fallen but is not yet in the table: it appears under day t+1 - inside
what the "perfect weather forecast" variants read. This script
 1. measures the offset from the data: Daymet rain against NLDAS rain (shipped with CAMELS) and IMERG satellite rain (served by UTC
    day; used if code/fetch_satellite.py has run), NLDAS against IMERG (the CAMELS files do not say which day NLDAS uses), and each
    product against the day the rivers rise;
 2. trains the all-basin LSTM with one extra input, NLDAS rain up to the issue day (its day ends in the evening of that day, hours
    before the forecast is issued), to see how much of the "forecast gap" 1 day ahead is in fact a time-stamp gap;
 3. repeats that with the same NLDAS rain made one day older (a control): it is still a second rain product, but it no longer
    holds anything the Daymet table does not already cover. If the gain disappears, the gain came from the rain of the issue day.
Reads the settings chosen by 04_lstm_all.py and its prediction sets (lstm_w, lstm_wq, lstm_wf, lstm_wqf) for comparison.
Writes results/rain_timing.json, figures/rain_timing.png and the prediction sets lstm_wn, lstm_wqn and the controls
lstm_wn_older, lstm_wqn_older (+ one per seed each).
Without training again: python code/10_rain_timing.py --rescore  recomputes every score, table, note and the figure from the saved
prediction sets (the training records are carried over from the JSON).
"""
import contextlib
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import flood as F
from camels import load_basin
from common import *

use_style()
t_start = time.time()
RESCORE = "--rescore" in sys.argv                          # no training: scores, notes and figure again from the saved prediction sets
D, B = F.load_data(), F.basins()
ids, thr = list(D), F.thresholds(D)
SEEDS = [0] if F.QUICK else [0, 1, 2]
COST = ("seconds", "cpu_seconds", "updates_run", "best_update", "best_val_nse", "n_params")        # the training record of one model (Run.summary)
notes = []

# ---- 1. whose day is it? -----------------------------------------------------------------------------------------------
corr = lambda a, b, k: float(pd.concat([a, b.shift(k)], axis=1).dropna().corr().iloc[0, 1])        # corr(a on day d, b on day d-k)
three = lambda a, b: dict(day_before=corr(a, b, 1), same_day=corr(a, b, 0), day_after=corr(a, b, -1))
NL, IM, stamps = {}, {}, {}
for g in ids:
    d = D[g]
    NL[g] = load_basin(g, "nldas").loc[F.PERIOD[0]:F.PERIOD[1], "prcp"].astype(float)
    assert NL[g].index.equals(d.index), "the NLDAS and the Daymet table must cover the same days"
    f = DATA / "satellite" / f"{g}.csv"
    if f.exists():
        IM[g] = pd.read_csv(f, parse_dates=["date"]).set_index("date")["imerg"].reindex(d.index)
    rise = d.q_mm.diff().clip(lower=0)                                                             # how much the river rose from day d-1 to day d
    X = np.column_stack([NL[g].shift(1), NL[g], NL[g].shift(-1)])[1:-1]
    w = np.linalg.lstsq(X, d.prcp.values[1:-1], rcond=None)[0]                                    # Daymet day d as a mix of NLDAS days d-1, d, d+1
    s = dict(daymet_vs_nldas=three(d.prcp, NL[g]), river_rise_vs_daymet=three(rise, d.prcp), river_rise_vs_nldas=three(rise, NL[g]),
             daymet_as_mix_of_nldas_days=dict(zip(("day_before", "same_day", "day_after"), (w / w.sum()).round(3).tolist())))
    if g in IM:
        s.update(daymet_vs_imerg=three(d.prcp, IM[g]), nldas_vs_imerg=three(NL[g], IM[g]), river_rise_vs_imerg=three(rise, IM[g]))
    stamps[g] = s
keys = list(stamps[ids[0]])
stamp_median = {k: {c: float(np.median([stamps[g][k][c] for g in ids if k in stamps[g]])) for c in ("day_before", "same_day", "day_after")} for k in keys}
assert not any(NL[g].isna().any() for g in ids), "NLDAS rain has gaps"
if not IM:
    notes.append("IMERG files (data/satellite) were not found, so the day-stamp check uses NLDAS only.")

# ---- 2. the same LSTM with rain that is stamped up to the evening of the issue day ---------------------------------------
D2 = {g: D[g].assign(prcp_nldas=NL[g].values, prcp_nldas_older=NL[g].shift(1).bfill().values) for g in ids}      # older: day t holds the NLDAS rain of day t-1
VAR = {"wn": dict(use_flow=False, base="w", ceiling="wf", col="prcp_nldas"), "wqn": dict(use_flow=True, base="wq", ceiling="wqf", col="prcp_nldas"),
       "wn_older": dict(use_flow=False, base="w", ceiling="wf", col="prcp_nldas_older"), "wqn_older": dict(use_flow=True, base="wq", ceiling="wqf", col="prcp_nldas_older")}
MAIN = {"wn": "wn_older", "wqn": "wqn_older"}                                                       # the test and its control
out, P = {}, {}
t_ask = time.time()
with contextlib.nullcontext() if RESCORE else F.cpu_lock("10_rain_timing.py"):
    waited = time.time() - t_ask
    if RESCORE:                                            # the settings and training records of the training run
        old = json.loads((F.RES_OUT / "rain_timing.json").read_text())
        conf = old["config"]
        notes += [x for x in old["notes"] if x.startswith("results/lstm_all.json was not found")]
    else:
        conf_file = F.RES / "lstm_all.json"                # read once the CPU is ours: 04_lstm_all.py writes it when it finishes
        conf = json.loads(conf_file.read_text())["config"] if conf_file.exists() else {}
        if not conf:
            notes.append("results/lstm_all.json was not found: the engine's default settings were used.")
        if F.QUICK:
            conf = dict(conf, updates=40, eval_every=20)
    for name, v in VAR.items():
        runs, per_seed = [], []
        for i, seed in enumerate(SEEDS):
            if RESCORE:
                p, meta, rec = F.load_preds(f"lstm_{name}_s{seed}"), F.preds_meta(f"lstm_{name}_s{seed}"), old["models"][name]["per_seed"][i]
                assert meta["config"] == conf and meta["seed"] == rec["seed"] == seed, f"lstm_{name}_s{seed} is not from the training run described in rain_timing.json"
                cost = {k: rec[k] for k in COST}
            else:
                cfg = F.Cfg(dyn=tuple(F.FORCING) + (v["col"],), use_flow=v["use_flow"], static=tuple(F.STATIC), seed=seed, **conf)
                run = F.fit(cfg, D2, ids, static_df=B)
                p, cost = run.predict(D2, static_df=B), run.summary()
                F.save_preds(f"lstm_{name}_s{seed}", p, dict(script="10_rain_timing.py", inputs=name, seed=seed, config=conf))
            e = F.evaluate(p, thr)
            runs.append(p)
            per_seed.append(dict(seed=seed, **cost, test_nse_median=[e["summary"][f"l{k}"]["nse_median"] for k in F.LEADS],
                                 test_onset_warned=[e["summary"][f"l{k}"]["onset"]["pod"] for k in F.LEADS]))
            print(f"lstm_{name} seed {seed} | best at update {cost['best_update']} of {cost['updates_run']}, {cost['seconds']:.0f} s | validation NSE {cost['best_val_nse']:.3f} | "
                  f"test NSE median {' '.join(f'{x:.3f}' for x in per_seed[-1]['test_nse_median'])}", flush=True)
        P[name] = F.ensemble(runs)
        if not RESCORE:                                    # a rescore reads the prediction sets and leaves them untouched
            F.save_preds(f"lstm_{name}", P[name], dict(script="10_rain_timing.py", inputs=name, seeds=SEEDS, config=conf))
        out[name] = dict(per_seed=per_seed, test=F.evaluate(P[name], thr), val=F.evaluate(P[name], thr, "val"))
timing = {k: old[k] for k in ("n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s") if k in old} if RESCORE else \
    dict(n_trainings=len(VAR) * len(SEEDS), training_seconds=round(sum(r["seconds"] for n in VAR for r in out[n]["per_seed"]), 1), waited_for_cpu_s=round(waited, 1))

# ---- how much of the gap to the perfect forecast does it close? ------------------------------------------------------------
med = lambda E, k: E["summary"][f"l{k}"]["nse_median"]
nse_of = lambda E, k: np.array([E["basins"][g][f"l{k}"]["nse"] for g in ids])
ons = lambda E, k, c="pod": E["summary"][f"l{k}"]["onset"][c]                                       # onset warnings pooled over the basins
lo_hi = lambda v: [float(min(v)), float(max(v))]
div = lambda a, b: a / b if b else float("nan")                                                      # a share of a gap that may be zero in a smoke test
closure = {}
for name, ctrl in MAIN.items():
    v = VAR[name]
    if not (F.have_preds(f"lstm_{v['base']}") and F.have_preds(f"lstm_{v['ceiling']}")):
        notes.append(f"lstm_{v['base']} / lstm_{v['ceiling']} not found: no comparison for {name}.")
        continue
    Pb, Pc = F.load_preds(f"lstm_{v['base']}"), F.load_preds(f"lstm_{v['ceiling']}")
    Eb, Ec, En, Eo = F.evaluate(Pb, thr), F.evaluate(Pc, thr), out[name]["test"], out[ctrl]["test"]
    assert all(E["summary"][f"l{k}"]["coverage_min"] == 1 for E in (Eb, Ec, En, Eo) for k in F.LEADS), "days without a forecast: the models would not be compared on identical days"
    out[v["base"]], out[v["ceiling"]] = dict(test=Eb), dict(test=Ec)
    seeds_b = [F.evaluate(F.load_preds(f"lstm_{v['base']}_s{s}"), thr) for s in (0, 1, 2) if F.have_preds(f"lstm_{v['base']}_s{s}")]
    rows = []
    for k in F.LEADS:
        b, c, n = med(Eb, k), med(Ec, k), med(En, k)
        ob, oc, on, oo = (ons(E, k) for E in (Eb, Ec, En, Eo))
        better = int(sum(En["basins"][g][f"l{k}"]["nse"] > Eb["basins"][g][f"l{k}"]["nse"] for g in ids))
        o = med(Eo, k)
        rows.append(dict(lead=k, nse_base=b, nse_with_nldas=n, nse_perfect_forecast=c, gap=c - b, gain=n - b, share_of_gap_closed=(n - b) / (c - b),
                         nse_control_one_day_older=o, gain_control=o - b, basins_better_control=int(sum(Eo["basins"][g][f"l{k}"]["nse"] > Eb["basins"][g][f"l{k}"]["nse"] for g in ids)),
                         seed_range_control=[min(r["test_nse_median"][k - 1] for r in out[ctrl]["per_seed"]), max(r["test_nse_median"][k - 1] for r in out[ctrl]["per_seed"])],
                         onset_warned_control=Eo["summary"][f"l{k}"]["onset"]["pod"],
                         basins_better=better, onset_warned_base=ob, onset_warned_with_nldas=on, onset_warned_perfect_forecast=oc,
                         onset_false_base=Eb["summary"][f"l{k}"]["onset"]["far"], onset_false_with_nldas=En["summary"][f"l{k}"]["onset"]["far"],
                         seed_range_with_nldas=[min(r["test_nse_median"][k - 1] for r in out[name]["per_seed"]), max(r["test_nse_median"][k - 1] for r in out[name]["per_seed"])],
                         seed_range_base=[min(med(e, k) for e in seeds_b), max(med(e, k) for e in seeds_b)] if seeds_b else None,
                         # the test against its control, and the same for the onset warnings (with the range of the single seeds)
                         share_of_gap_closed_beyond_control=div(n - o, c - b), basins_better_than_control=int((nse_of(En, k) > nse_of(Eo, k)).sum()),
                         median_basin_gain=float(np.median(nse_of(En, k) - nse_of(Eb, k))), median_basin_gain_over_control=float(np.median(nse_of(En, k) - nse_of(Eo, k))),
                         onset_false_control=ons(Eo, k, "far"), onset_false_perfect_forecast=ons(Ec, k, "far"),
                         onset_share_of_gap_closed=div(on - ob, oc - ob), onset_share_of_gap_closed_beyond_control=div(on - oo, oc - ob),
                         onset_seed_range_with_nldas=lo_hi([r["test_onset_warned"][k - 1] for r in out[name]["per_seed"]]),
                         onset_seed_range_control=lo_hi([r["test_onset_warned"][k - 1] for r in out[ctrl]["per_seed"]]),
                         onset_seed_range_base=lo_hi([ons(e, k) for e in seeds_b]) if seeds_b else None,
                         basins_perfect_forecast_below_base=int((nse_of(Ec, k) < nse_of(Eb, k)).sum())))
    offset = [stamps[g]["daymet_as_mix_of_nldas_days"]["day_before"] for g in ids]                 # how much of a Daymet day is rain of the NLDAS day before
    rho, pval = spearmanr(offset, nse_of(En, 1) - nse_of(Eo, 1))
    closure[name] = dict(base=f"lstm_{v['base']}", ceiling=f"lstm_{v['ceiling']}", control=f"lstm_{ctrl}", by_lead=rows,
                         gain_over_control_against_stamp_offset=dict(lead=1, spearman=float(rho), p=float(pval), n_basins=len(ids),
                                                                     how_to_read="rank correlation over the basins between the stamp offset (stamps.per_basin.<gauge>.daymet_as_mix_of_nldas_days.day_before) "
                                                                                 "and the NSE gain of the test over its control (models.<set>.test.basins.<gauge>.l1.nse)"))

# ---- notes: every number in them is read from the tables above ------------------------------------------------------------
pc = lambda x: f"{round(100 * x) + 0}%" if x == x else "no value"                                    # a share in per cent (never "-0%")
span = lambda r, f=pc: f"{f(r[0])} to {f(r[1])}" if r else "no single seeds"
n3 = lambda x: f"{x:.3f}"
p1 = lambda x: f"{100 * x:.1f}%"
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]                                            # every single seed of a above every single seed of b
dn, rd, rn = stamp_median["daymet_vs_nldas"], stamp_median["river_rise_vs_daymet"], stamp_median["river_rise_vs_nldas"]
least = min(ids, key=lambda g: stamps[g]["daymet_as_mix_of_nldas_days"]["day_before"])             # the basin whose Daymet day is least offset
ld = stamps[least]["daymet_vs_nldas"]
notes += ["Day stamp of Daymet - an inference from correlations of daily rain (stamps; medians of the basins), not a documented fact checked here: the Daymet rain stamped day d correlates "
          f"{dn['same_day']:.2f} with NLDAS rain of the same day, {dn['day_before']:.2f} with NLDAS of the day before and {dn['day_after']:.2f} with the day after"
          + (f" (IMERG: {stamp_median['daymet_vs_imerg']['same_day']:.2f} / {stamp_median['daymet_vs_imerg']['day_before']:.2f} / {stamp_median['daymet_vs_imerg']['day_after']:.2f})" if IM else "")
          + ". A Daymet day shares its rain with two days of the other products, d-1 and d: it ends earlier than theirs, as a table built from rain gauges that are read in the morning would. "
          f"The offset is not the same in every basin (stamps.per_basin): it is smallest in {F.SHORT[least]} (same day {ld['same_day']:.2f}, day before {ld['day_before']:.2f}, day after {ld['day_after']:.2f}).",
          f"The rivers fit that reading but cannot prove it: the rise of the flow from day d-1 to day d correlates {rd['same_day']:.2f} with the Daymet rain stamped d, {rd['day_before']:.2f} with the "
          f"rain stamped d-1 and {rd['day_after']:.2f} with the rain stamped d+1 (NLDAS rain: {rn['same_day']:.2f} / {rn['day_before']:.2f} / {rn['day_after']:.2f}). River correlations also depend on the "
          "time the water needs to reach the gauge, so alone they could not fix a day stamp; the rain-to-rain correlations do not involve the river."]
if IM:
    ni, first_imerg = stamp_median["nldas_vs_imerg"], min(s.first_valid_index() for s in IM.values()).year
    earlier = sum(stamps[g]["nldas_vs_imerg"]["day_before"] > stamps[g]["nldas_vs_imerg"]["day_after"] for g in IM)
    notes.append("Is the NLDAS rain of the issue day known when the forecast is issued? IMERG is served by UTC day (19:00 to 19:00 local standard time in these basins; 18:00 to 18:00 in Leaf River, MS), "
                 "as NASA POWER states (results/satellite_data.json). The CAMELS files do not say how hourly NLDAS rain was summed to days, so it is measured here: NLDAS rain of day d correlates "
                 f"{ni['same_day']:.2f} with IMERG of the same UTC day, {ni['day_before']:.2f} with the UTC day before and {ni['day_after']:.2f} with the UTC day after (day before above day after in "
                 f"{earlier} of {len(IM)} basins). The NLDAS day is therefore the UTC day or ends a little earlier - not later. The forecast is issued at the end of the local day (the day of the daily "
                 "river flow), so the NLDAS value of the issue day is complete five hours or more before: its day stamp leaks nothing. "
                 f"This check covers {first_imerg}-{F.PERIOD[1][:4]} (IMERG has no earlier days), which holds the validation and the test years but few training years.")
else:
    notes.append("NLDAS is taken to be stamped by UTC day (19:00 to 19:00 local standard time in these basins; 18:00 to 18:00 in Leaf River, MS). The CAMELS files do not state it, and without the "
                 "IMERG files it could not be checked here.")
notes += ["NLDAS itself is published a few days late, and it is made after the fact: its hourly rain is a daily rain-gauge analysis spread over the hours, with radar from 1996 and with coarser "
          "hourly gauge data before (NASA, NLDAS-2 forcing documentation). Before 1996 its day boundary may therefore be less sharp; that can change what a model learns in the training years, not how "
          "it is scored in the test years. Real-time equivalents exist (radar and gauge analyses, satellite near-real-time runs). The point here is the time stamp of the rain, not the product: "
          "the test shows the best case for rain observed up to the evening.",
          "What the gap can hold: the 'perfect weather forecast' variants read the Daymet weather stamped t+1, t+2 and t+3 at every lead. If a Daymet day ends in the morning, most of the rain "
          "stamped t+1 had already fallen when the forecast was issued; the rest of it, and the rain stamped t+2 and t+3, fell afterwards. So the gap between the past-weather models and the "
          "perfect-forecast models 1 day ahead holds both rain that had fallen but was not yet in the table (a time-stamp gap) and rain still to come (a forecast gap). The test measures the first part.",
          f"Control: the same NLDAS rain made one day older (on issue day t the model reads the NLDAS rain of day t-1). The Daymet rain of day t correlates {dn['day_before']:.2f} with that NLDAS day and "
          f"{dn['same_day']:.2f} with the next one, so the Daymet table reaches later than the control's rain: the control adds a second rain product but no later rain. Test minus control is the worth "
          "of the NLDAS rain of the issue day. That is mostly rain the Daymet table does not hold yet, but also a second estimate of the rain of the night before, which the Daymet value of day t "
          "already covers; the experiment cannot separate the two. The control alone measures what a second rain product for the days up to t-1 is worth."]
for name, where in (("wqn", "with the river gauge"), ("wn", "without the river gauge")):
    if name not in closure:
        continue
    r, later, n_b = closure[name]["by_lead"][0], closure[name]["by_lead"][1:], len(ids)
    clear_on = above(r["onset_seed_range_with_nldas"], r["onset_seed_range_base"]) and above(r["onset_seed_range_with_nldas"], r["onset_seed_range_control"])
    clear_nse = above(r["seed_range_with_nldas"], r["seed_range_base"]) and above(r["seed_range_with_nldas"], r["seed_range_control"])
    seeds_say = lambda ok: "every single seed with it is above every seed without it and every seed of the control" if ok else "the single seeds overlap those of the model without it or of the control"
    notes.append(f"Result 1 day ahead {where} (test years; forecasts averaged over the seeds, lowest to highest single seed in brackets - the averaged forecast can score above every single "
                 f"seed). Onsets warned: {p1(r['onset_warned_base'])} ({span(r['onset_seed_range_base'], p1)}) without the extra rain, {p1(r['onset_warned_control'])} "
                 f"({span(r['onset_seed_range_control'], p1)}) with the control, {p1(r['onset_warned_with_nldas'])} ({span(r['onset_seed_range_with_nldas'], p1)}) with the NLDAS rain of the issue day, "
                 f"{p1(r['onset_warned_perfect_forecast'])} with the perfect forecast: "
                 f"the rain of the issue day closes {pc(r['onset_share_of_gap_closed'])} of that gap, {pc(r['onset_share_of_gap_closed_beyond_control'])} beyond the control; {seeds_say(clear_on)}. "
                 f"False warnings among the onset warnings, in the same order: {pc(r['onset_false_base'])}, {pc(r['onset_false_control'])}, {pc(r['onset_false_with_nldas'])}, "
                 f"{pc(r['onset_false_perfect_forecast'])}. Median NSE: {n3(r['nse_base'])} ({span(r['seed_range_base'], n3)}), {n3(r['nse_control_one_day_older'])} "
                 f"({span(r['seed_range_control'], n3)}), {n3(r['nse_with_nldas'])} ({span(r['seed_range_with_nldas'], n3)}), {n3(r['nse_perfect_forecast'])}: {pc(r['share_of_gap_closed'])} of the gap, "
                 f"{pc(r['share_of_gap_closed_beyond_control'])} beyond the control; {seeds_say(clear_nse)}. Basin by basin the NSE is higher than without the extra rain in {r['basins_better']} of "
                 f"{n_b} basins and higher than with the control in {r['basins_better_than_control']} (median basin difference {r['median_basin_gain']:+.3f} and "
                 f"{r['median_basin_gain_over_control']:+.3f}): the difference of two medians and the median of the basin differences need not agree. The perfect-forecast model is an upper bound "
                 f"in what it knows, not in every basin's score: it scores below the model without the extra rain in {r['basins_perfect_forecast_below_base']} of {n_b} basins (not examined here). "
                 f"2 and 3 days ahead: {' and '.join(pc(x['share_of_gap_closed']) for x in later)} of the NSE gap and {' and '.join(pc(x['onset_share_of_gap_closed']) for x in later)} of the "
                 "onset-warning gap.")
if closure:
    rank = {n: c["gain_over_control_against_stamp_offset"]["spearman"] for n, c in closure.items()}
    notes += ["How far the test reaches: the share closed 1 day ahead is not the whole time-stamp part of the gap. The NLDAS day ends at 19:00 local time or a little earlier, so rain of the last "
              "five hours or more of the issue day is in neither model, and what is left of the gap is therefore not all rain still to come. NLDAS and Daymet are also different products, so the "
              "split is approximate.",
              "Does the gain follow the day stamp? 1 day ahead, the NSE gain of the test over its control is compared, basin by basin, with the offset of the Daymet day (the share of the NLDAS day "
              "before in daymet_as_mix_of_nldas_days): rank correlation " + ", ".join(f"{rank[n]:+.2f} {w}" for n, w in (("wqn", "with the river gauge"), ("wn", "without it")) if n in rank)
              + f". {len(ids)} basins that share storms: a description, not a proof."]
notes += ["Same settings and seeds as 04_lstm_all.py (results/lstm_all.json, key config): the trainings differ from its models only in the extra input column. The comparison sets lstm_w, lstm_wq, "
          "lstm_wf, lstm_wqf are read from its prediction files, not retrained. All sets are compared on identical days (coverage 1.0, asserted).",
          "Fair test: nothing is chosen here. The settings were chosen by 04_lstm_all.py on the validation years, the trainer picks each stopping point on the validation years, and the test years "
          "are only scored. The closure table is test years only; models.<set>.val holds the validation-year scores of the four new sets.",
          "runtime_s leaves out the time the script waited for the CPU (waited_for_cpu_s); training_seconds is the time of the trainings alone." if "waited_for_cpu_s" in timing else
          "runtime_s is the time of the training run from start to end. That run did not record how long it waited for the CPU behind another job, so runtime_s includes the wait; "
          "training_seconds is the time of the trainings alone."]
if F.QUICK:
    notes.append("QUICK smoke test: a few updates and one seed - numbers are not results.")
if RESCORE:
    timing["rescore_runtime_s"] = round(time.time() - t_start, 1)
    notes.append("This file was written by --rescore: every score, table and note and the figure were recomputed from the saved prediction sets lstm_<set>_s<seed> without training again; the "
                 "training records in per_seed, n_trainings, training_seconds and runtime_s are those of the training run. rescore_runtime_s is the time of this pass.")
else:
    timing["runtime_s"] = round(time.time() - t_start - waited, 1)
R = dict(question="How much of the gap between the past-weather models and the perfect-forecast models is rain that had already fallen but was stamped with the next day?",
         quick_mode=F.QUICK, stamps=dict(per_basin=stamps, median=stamp_median, how_to_read="correlation of daily values, 1980-2014 (IMERG: from its first day); day_before = the second series one day earlier"),
         config=conf, seeds=SEEDS, models=out, closure=closure, notes=notes, **timing)
(F.RES_OUT / "rain_timing.json").write_text(json.dumps(R, indent=1))

# ---- figure ------------------------------------------------------------------------------------------------------------
FS = 8.5                                                   # drawn 193 mm wide: 7.5 pt when printed 170 mm wide
fig, ax = plt.subplots(1, 3, figsize=(7.6, 3.5), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
x = np.arange(3)
series = [("daymet_vs_nldas", "NLDAS rain", BLUE, "o"), ("daymet_vs_imerg", "satellite rain (IMERG)", AQUA, "s")]
for key, lab, col, mk in series:
    if key in stamp_median:
        y = [stamp_median[key][c] for c in ("day_before", "same_day", "day_after")]
        for g in ids:
            ax[0].plot(x, [stamps[g][key][c] for c in ("day_before", "same_day", "day_after")], color=col, lw=.6, alpha=.22, zorder=1)
        ax[0].plot(x, y, color=col, marker=mk, ms=5, zorder=3, label=lab)
ax[0].set_xticks(x, ["the day\nbefore", "the same\nday", "the day\nafter"], fontsize=FS); ax[0].set_ylim(0, 1); ax[0].tick_params(labelsize=FS)
ax[0].set_ylabel("correlation with the rain\nstamped day d (Daymet)", fontsize=FS)
ax[0].set_title("Daymet's day starts early", fontsize=FS + 1)
ax[0].set_xlabel("rain of the other product on ...", fontsize=FS)
h0, l0 = ax[0].get_legend_handles_labels()
fam = [("wn", "No river gauge"), ("wqn", "With river gauge")]
for a, (name, title) in zip(ax[1:], fam):
    v = VAR[name]
    lines = [(v["base"], "past weather (rain table: Daymet)", MUTED, "-"), (MAIN[name], "control: + NLDAS rain, one day older", YELLOW, ":"),
             (name, "+ NLDAS rain up to the evening of the issue day", ORANGE, "-"), (v["ceiling"], "+ perfect weather forecast (upper bound)", INK2, "--")]
    for key, lab, col, ls in lines:
        if key not in out:
            continue
        E = out[key]["test"]
        y = [med(E, k) for k in F.LEADS]
        for j, k in enumerate(F.LEADS):
            if key != MAIN[name]:
                a.scatter(np.full(len(ids), j) + (.0 if key == name else -.12 if key == v["base"] else .12), [E["basins"][g][f"l{k}"]["nse"] for g in ids], s=5, color=col, alpha=.3, edgecolors="none", zorder=1)
        a.plot(range(3), y, color=col, ls=ls, marker="o", ms=4.5, zorder=3, label=lab)
    a.set_xticks(range(3), ["1 day\nahead", "2 days\nahead", "3 days\nahead"], fontsize=FS); a.set_ylim(-0.25, 1); a.tick_params(labelsize=FS)
    a.axhline(0, color=AXIS, lw=.8, zorder=0); a.set_title(title, fontsize=FS + 1)
ax[1].set_ylabel("NSE in the test years\n(1 = perfect)", fontsize=FS)
h1, l1 = ax[1].get_legend_handles_labels()
fig.tight_layout(w_pad=1.0, rect=(0, .245, 1, 1))
fig.legend(h0, l0, frameon=False, fontsize=FS, loc="lower left", bbox_to_anchor=(.05, .10), ncol=1, handlelength=1.8)
fig.legend(h1, l1, frameon=False, fontsize=FS, loc="lower left", bbox_to_anchor=(.43, .005), ncol=1, handlelength=1.8)
fig.text(.062, .02, f"thin lines, small dots: one basin each;\nthick lines: median of the {len(ids)} basins", fontsize=FS, color=MUTED, ha="left", va="bottom")
fig.savefig(F.FIG_OUT / "rain_timing.png", dpi=220); plt.close(fig)

print(json.dumps(dict(stamp_median=stamp_median, closure={n: [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in c["by_lead"]] for n, c in closure.items()}, runtime_s=R["runtime_s"]), indent=1))
