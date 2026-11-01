"""What does the training of the LSTM look like? The learning curves that every training stored, as one figure.

The report says in words that the LSTM trained on one river memorises its training years almost at once (overfitting) and that
early stopping keeps the state with the best validation score. This script draws what the trainer recorded at every validation
check - the training loss and the validation score - for three training set-ups:
 1. one river (French Broad, NC) with past weather + river gauge: ONE training with the BASE setting of the tuning, the only
    one-river curve that 03_lstm.py stored. It is not the one-river model the report uses (that has the setting the tuning chose
    in the end); of that model only the update each training was kept at is stored, and it is copied to the JSON (key
    chosen_one_river) so that the report can relate its sentence on early stopping to this figure;
 2. all 12 rivers with past weather + river gauge (the trainings behind the report's twelve-river results with these inputs);
 3. all 12 rivers with the perfect weather forecast as well (an upper bound that cannot be had in practice).
Top row: training loss; bottom row: validation score, with a dot and a band at the state that was kept. Two rows, never two
scales on one axis; every column has its own scales.

What a stored curve holds (flood.fit): at every check, train_loss = the loss the training reduces (squared error of the scaled
flow 1 to 3 days ahead, weighted per basin, with dropout on), averaged over the updates since the check before; val_nse = the NSE
in the validation years, mean over the basins trained on (one, or twelve) and over the three lead times. The state with the
highest val_nse is kept; training stops after `patience` checks without a better one.

Two things the picture alone cannot say are put into numbers (and into the footnote and the notes), so that no document reads
more into it than the stored trainings hold:
 - an update is not a pass through the data: an update scores the same number of days in every column, so the updates of the
   one-river column and of the twelve-river columns are set side by side in passes through the training years (key passes);
 - the one-river curve is the only one that was stored: where every other stored training was kept - the other tuning trainings
   on that river, the chosen one-river setting with each set of inputs, the all-river trainings that are not drawn - is copied
   to the JSON (key not_drawn), with the range they span.

No model is trained and no forecast is loaded. Reads results/lstm_single.json (03_lstm.py) and results/lstm_all.json
(04_lstm_all.py) only; checks that the three set-ups share their network settings, that every curve belongs to the training
record stored next to it, and that the lettering of the figure is large enough and overlaps nothing.
Writes figures/training_curves.png and results/training_curves.json (every number the documents print, the drawn curves, a
caption, and notes in sentences the documents can use).
Smoke test: FLOOD_QUICK=1 python code/12_training_curves.py  reads and writes only results/quick and figures/quick; if a
smoke-test file of 03 or 04 is missing it says so in one line and ends without an error.
"""
import datetime
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, MaxNLocator

import flood as F
from common import *

use_style()
t_start = time.time()
SCRIPT = "12_training_curves.py"
W_IN, H_IN, DPI, FS = 6.7, 4.25, 220, 8.0                  # as wide as the other figures of the report; FS = the smallest lettering in points at that width
EDGE = 0.045                                               # distance of a label inside a panel from the edge of the panel, as a share of the panel height
SHARED = ("hidden", "lr", "dropout", "warmup", "block", "batch", "patience")       # the settings the three set-ups are checked to share
USES = {"w": (False, False), "wq": (True, False), "wf": (False, True), "wqf": (True, True)}       # input set -> (river gauge, perfect weather forecast)
WORD = {"hidden": "{} memory cells", "lr": "learning rate {:g}", "dropout": "dropout {:g}", "warmup": "{} days of history before the first scored forecast",
        "block": "{} scored days per stretch", "batch": "{} stretches per update", "patience": "stop after {} checks without a better score"}
SHORT_WORD = dict(WORD, warmup="{} days of history")
nth = lambda i: ("first", "second", "third", "fourth", "fifth")[i] if i < 5 else f"{i + 1}th"

num = lambda v: f"{v:,}"
n2 = lambda v: f"{v:.2f}"
upd = lambda lo, hi: f"update {num(lo)}" if lo == hi else f"updates {num(lo)} to {num(hi)}"
plural = lambda n, w: f"{n} {w}{'s' * (n != 1)}"
listed = lambda v: ", ".join(v[:-1]) + (" and " if len(v) > 1 else "") + v[-1]
mean = lambda v: sum(v) / len(v)


def stop(best, last, every, patience, limit):
    """Why a training ended: `patience` checks passed without a better validation score, or the limit of updates was reached first."""
    return "no better score" if last - best == patience * every else "limit of updates" if last >= limit else "other"


def inputs_of(inp, n_static=0):
    """The inputs of an input set in the words of the figure."""
    gauge, forecast = USES[inp]
    return F.INPUT_LABEL["wq" if gauge else "w"] + (" + perfect weather forecast" if forecast else "") + (f" + {n_static} basin properties" if n_static else "")


def training(seed, curve, rec, s):
    """The facts of one training, read off its stored curve. rec is the training record stored next to the curve; s the set-up."""
    kept, last = max(curve, key=lambda p: p["val_nse"]), curve[-1]              # max() takes the first of equal scores, as the trainer does
    assert (kept["update"], last["update"]) == (rec["best_update"], rec["updates_run"]) and abs(kept["val_nse"] - rec["best_val_nse"]) < 1e-4, \
        f"{s['key']}, seed {seed}: the stored curve is not that of the stored training record"
    assert rec["n_params"] == s["n_weights"], f"{s['key']}, seed {seed}: the stored number of weights is not that of one LSTM layer with {s['n_inputs']} inputs and a linear output layer"
    assert all(abs(p["val_nse"] - mean(p["val_nse_by_lead"])) < 1e-9 and len(p["val_nse_by_lead"]) == len(F.LEADS) for p in curve), "val_nse is not the mean of the lead times"
    assert all(p["update"] == s["check_every"] * (i + 1) for i, p in enumerate(curve[:-1])), f"{s['key']}, seed {seed}: the checks are not {s['check_every']} updates apart"
    after = [p for p in curve if p["update"] >= kept["update"]]
    low = min(after, key=lambda p: p["val_nse"])
    return dict(seed=seed, best_update=kept["update"], last_update=last["update"], val_nse_at_best=kept["val_nse"], val_nse_at_last=last["val_nse"],
                train_loss_first=curve[0]["train_loss"], train_loss_at_best=kept["train_loss"], train_loss_last=last["train_loss"], first_update=curve[0]["update"],
                n_checks=len(curve), val_nse_at_best_by_lead=kept["val_nse_by_lead"], val_nse_lowest_after_best=low["val_nse"], update_of_lowest_after_best=low["update"],
                stopped_by=stop(kept["update"], last["update"], s["check_every"], s["patience"], s["updates_allowed"]), passes_at_best=round(kept["update"] / s["updates_per_pass"], 1),
                updates_after_best=last["update"] - kept["update"], passes_after_best=round((last["update"] - kept["update"]) / s["updates_per_pass"], 1),
                val_nse_fall_after_best=kept["val_nse"] - last["val_nse"],
                train_loss_rises_after_best_at=[b["update"] for a, b in zip(after, after[1:]) if b["train_loss"] > a["train_loss"]],     # checks with a higher training loss than the check before
                val_nse_rises_after_best_at=[b["update"] for a, b in zip(after, after[1:]) if b["val_nse"] > a["val_nse"]],
                curve=dict(update=[p["update"] for p in curve], train_loss=[p["train_loss"] for p in curve], val_nse=[p["val_nse"] for p in curve]))


def setup(key, title, lines, label, inp, n_rivers, n_static, cf, runs, train_days, **more):
    """One column of the figure: a training set-up with its settings (cf), its inputs and the facts of each of its trainings."""
    gauge, forecast = USES[inp]
    n_in = len(F.FORCING) + gauge + forecast * len(F.LEADS) * len(F.FORCING) + n_static        # as flood._design puts the inputs together
    L = f"mean of {F.LEADS[0]} to {F.LEADS[-1]} days ahead"
    s = dict(key=key, title=title, lines=lines, validation_nse_label=label, input_set=inp,
             validation_nse_is=f"the NSE of this river in the validation years, {L}" if n_rivers == 1 else f"the mean over the {n_rivers} rivers of their NSE in the validation years, {L}",
             inputs=inputs_of(inp, n_static), n_rivers=n_rivers, n_trainings=len(runs), n_inputs=n_in, n_weights=4 * cf["hidden"] * (n_in + cf["hidden"] + 2) + len(F.LEADS) * (cf["hidden"] + 1),
             check_every=cf["eval_every"], updates_allowed=cf["updates"], patience=cf["patience"], training_days=n_rivers * train_days, scored_days_per_update=cf["batch"] * cf["block"],
             updates_per_pass=n_rivers * train_days / (cf["batch"] * cf["block"]), **more)             # updates that score as many days as the training years hold
    s["per_training"] = [training(seed, curve, rec, s) for seed, curve, rec in runs]
    b, z, v = ([t[k] for t in s["per_training"]] for k in ("best_update", "last_update", "passes_at_best"))
    s.update(best_update_min=min(b), best_update_max=max(b), last_update_min=min(z), last_update_max=max(z), passes_at_best_min=min(v), passes_at_best_max=max(v))
    return s


def read():
    """The three set-ups, the settings check and what is stored of the chosen one-river model, from the two results files."""
    S1, A = (json.loads((F.RES_OUT / f"{n}.json").read_text()) for n in ("lstm_single", "lstm_all"))
    assert F.QUICK or not (S1["quick_mode"] or A["quick_mode"]), "results/ holds a smoke-test file: run 03_lstm.py and 04_lstm_all.py again"
    assert S1["split"] == A["split"], "the two results files use different training, validation and test years"
    TU, cfg, split = S1["tuning"], A["config"], S1["split"]
    base, chosen, inp1 = TU["base"], TU["chosen"], TU["input_set"]
    day = datetime.date.fromisoformat
    train_days = (day(split["train"][1]) - day(split["train"][0])).days + 1                  # training days of one river
    rivers, statics = len(A["basins"]), len(A["static_inputs"])

    # ---- one river: the base setting of the tuning; the curve stored is that of its first seed (03_lstm.py: base_runs[0]) ----
    base_runs = sorted((r for r in TU["runs"] if r["input_set"] == inp1 and all(r[k] == base[k] for k in base)), key=lambda r: r["seed"])
    r0 = base_runs[0]
    rec = lambda r: dict(best_update=r["best_update"], updates_run=r["updates_run"], best_val_nse=r["val_nse_mean"], n_params=r["n_params"])
    one = dict(base, batch=r0["batch"], patience=r0["patience"], eval_every=r0["eval_every"], updates=r0["max_updates"])
    assert (one["eval_every"], one["patience"]) == (TU["budget"]["eval_every"], TU["budget"]["patience"]), "the first tuning run does not follow the tuning budget"
    river = S1["basin"]["short"]
    P = [setup("one_river", f"One river ({river})", [F.INPUT_LABEL[inp1], "base setting of the tuning · 1 training"], "NSE of this river", inp1, 1, 0, one,
               [(r0["seed"], TU["base_curve"], rec(r0))], train_days, river=river, gauge_id=S1["basin"]["gauge_id"], what="the base setting of the tuning, one training (its first seed)")]
    P[0]["other_trainings_without_curve"] = [dict(seed=r["seed"], best_update=r["best_update"], last_update=r["updates_run"], val_nse_at_best=r["val_nse_mean"],
                                                  passes_at_best=round(r["best_update"] / P[0]["updates_per_pass"], 1)) for r in base_runs[1:]]
    # ---- all rivers: the final models of 04_lstm_all.py, every seed, with the river gauge and with the perfect weather forecast as well ----
    for inp, title, first in (("wq", f"All {rivers} rivers", F.INPUT_LABEL["wq"]), ("wqf", f"All {rivers} rivers, upper bound", "+ perfect weather forecast")):
        X = A["sets"][inp]
        runs = [(s, X["per_seed"][str(s)]["curve"], X["cost"][str(s)]) for s in X["seeds"]]
        P.append(setup(f"all_{inp}", title, [first, plural(len(runs), "training")], f"mean NSE of the {rivers} rivers", inp, rivers, statics, cfg, runs, train_days,
                       what=f"the final all-river models of 04_lstm_all.py (prediction sets lstm_{inp}_s<seed>), every seed"))

    check = {k: dict(one_river=one[k], all_rivers=cfg[k], same=one[k] == cfg[k]) for k in SHARED}
    # ---- the one-river model the report uses: the setting the tuning chose; its curves were not stored, its training records were ----
    fin = S1["finals"][inp1]["seeds"]
    C = dict(settings=chosen, same_as_base=chosen == base, differs_from_base={k: dict(base=base[k], chosen=chosen[k]) for k in base if base[k] != chosen[k]},
             input_set=inp1, inputs=F.INPUT_LABEL[inp1], updates_allowed=TU["final_budget"]["updates"], check_every=TU["final_budget"]["eval_every"], patience=TU["final_budget"]["patience"],
             first_seed_is_left_column=chosen == base, source="results/lstm_single.json finals." + inp1 + ".seeds.<seed>.training; known_points: diagnostic_fixed_length." + inp1,
             per_training=[dict(seed=int(s), best_update=fin[s]["training"]["best_update"], last_update=fin[s]["training"]["updates_run"], val_nse_at_best=fin[s]["training"]["best_val_nse"],
                                n_weights=fin[s]["training"]["n_params"]) for s in sorted(fin, key=int)],
             known_points=[dict(update=r["updates"], what=r["stopping_point"], train_nse=mean(r["train_nse"]), val_nse=mean(r["val_nse"]), test_nse=mean(r["test_nse"]))
                           for r in S1.get("diagnostic_fixed_length", {}).get(inp1, [])])
    b = [t["best_update"] for t in C["per_training"]]
    C.update(best_update_first_seed=b[0], best_update_min=min(b), best_update_max=max(b))

    # ---- the stored trainings that are not drawn: only where each was kept. Is the one curve of the left column a typical one? ----
    def group(rows, **more):
        u = [r["best_update"] for r in rows]
        return dict(more, n_trainings=len(rows), best_update_min=min(u), best_update_max=max(u), best_update_median=float(np.median(u)), n_kept_at_min=u.count(min(u)),
                    n_stopped_by_no_better_score=sum(r["stopped_by"] == "no better score" for r in rows), per_training=rows)
    tuned = group([dict(run_id=r["run_id"], seed=r["seed"], **{k: r[k] for k in base}, differs_from_base=[k for k in base if r[k] != base[k]], best_update=r["best_update"],
                        last_update=r["updates_run"], val_nse_at_best=r["val_nse_mean"], passes_at_best=round(r["best_update"] * r["batch"] * r["block"] / train_days, 1),
                        stopped_by=stop(r["best_update"], r["updates_run"], r["eval_every"], r["patience"], r["max_updates"]), is_the_curve_drawn=r is r0)
                   for r in TU["runs"] if r["input_set"] == inp1],
                  what=f"every training of the tuning (03_lstm.py) on this river with {F.INPUT_LABEL[inp1]}; the curve of one of them was stored and is the left column",
                  inputs=F.INPUT_LABEL[inp1], source="results/lstm_single.json tuning.runs")
    fb = TU["final_budget"]
    limit = lambda inp: (TU["budget"] if inp == inp1 and chosen == base else fb)["updates"] // (2 if chosen["hidden"] > 64 else 1)        # as 03_lstm.py sets the limit of a final training
    finals = {inp: group([dict(seed=int(s), best_update=x["training"]["best_update"], last_update=x["training"]["updates_run"], val_nse_at_best=x["training"]["best_val_nse"],
                               stopped_by=stop(x["training"]["best_update"], x["training"]["updates_run"], fb["eval_every"], fb["patience"], limit(inp)))
                          for s, x in sorted(f["seeds"].items(), key=lambda kv: int(kv[0]))],
                         inputs=inputs_of(inp), perfect_weather_forecast=USES[inp][1], river_gauge=USES[inp][0], source=f"results/lstm_single.json finals.{inp}.seeds.<seed>.training")
              for inp, f in S1["finals"].items()}
    others = {inp: group([dict(seed=s, best_update=c["best_update"], last_update=c["updates_run"], val_nse_at_best=c["best_val_nse"],
                               stopped_by=stop(c["best_update"], c["updates_run"], cfg["eval_every"], cfg["patience"], cfg["updates"])) for s in X["seeds"] for c in [X["cost"][str(s)]]],
                         inputs=inputs_of(inp, statics), perfect_weather_forecast=USES[inp][1], river_gauge=USES[inp][0], source=f"results/lstm_all.json sets.{inp}.cost.<seed> (curves: per_seed.<seed>.curve)")
              for inp, X in A["sets"].items() if f"all_{inp}" not in [p["key"] for p in P]}
    N = dict(what="trainings whose curves are not in the figure: where each was kept (best_update) and how it ended", one_river_tuning=tuned,
             one_river_chosen_setting=dict(settings=chosen, by_input_set=finals), all_rivers=others)

    # ---- updates against passes through the training years: an update scores cfg.batch stretches of cfg.block days whatever the number of rivers ----
    one_p, mid, t0 = P[0], P[1], P[0]["per_training"][0]
    ran = {p["key"]: p["patience"] * p["check_every"] for p in P}                           # updates a training runs on after its kept state, when the stopping rule ends it
    u_same = t0["best_update"] + ran[mid["key"]] / mid["updates_per_pass"] * one_p["updates_per_pass"]     # the one-river update as many passes after its kept state
    at = next((i for i, u in enumerate(t0["curve"]["update"]) if abs(u - u_same) < 1e-6), None)
    PS = dict(what="an update scores the same number of days in every column; a pass = as many scored days as the training years of the set-up hold (the stretches are drawn at random, "
                   "so a pass is an average)",
              scored_days_per_update={p["key"]: p["scored_days_per_update"] for p in P}, updates_per_pass={p["key"]: p["updates_per_pass"] for p in P},
              one_river_kept_update=t0["best_update"], one_river_passes_at_kept_update=t0["passes_at_best"],
              all_rivers_updates_with_as_many_passes=round(t0["best_update"] / one_p["updates_per_pass"] * mid["updates_per_pass"]),
              run_on_after_kept_state={k: dict(updates=v, passes=round(v / p["updates_per_pass"], 1)) for (k, v), p in zip(ran.items(), P)},
              one_river_as_many_passes_after_kept_state_as_all_rivers=None if at is None else dict(
                  update=t0["curve"]["update"][at], updates_after_kept_state=t0["curve"]["update"][at] - t0["best_update"], val_nse=t0["curve"]["val_nse"][at],
                  val_nse_fall=t0["val_nse_at_best"] - t0["curve"]["val_nse"][at]),
              val_nse_fall_kept_state_to_last_check={p["key"]: dict(min=min(t["val_nse_fall_after_best"] for t in p["per_training"]), max=max(t["val_nse_fall_after_best"] for t in p["per_training"]))
                                                     for p in P})
    return dict(P=P, check=check, same=all(c["same"] for c in check.values()), cfg=cfg, base=base, chosen=C, split=split, river=river, rivers=rivers, statics=statics,
                not_drawn=N, passes=PS)


def words(D):
    """Title, footnote, caption and notes: every sentence is put together from the numbers read, so that it is true of the curves drawn."""
    P, C, cfg, check, n_riv, N, PS = D["P"], D["chosen"], D["cfg"], D["check"], D["rivers"], D["not_drawn"], D["passes"]
    one, mid, right = P
    t0, n_all = one["per_training"][0], sum(p["n_trainings"] for p in P)
    L = f"{F.LEADS[0]} to {F.LEADS[-1]} days ahead"
    diff = [SHORT_WORD[k].format(v["chosen"]) for k, v in C["differs_from_base"].items()]
    unlike = [f"{SHORT_WORD[k].format(c['one_river'])} against {SHORT_WORD[k].format(c['all_rivers'])}" for k, c in check.items() if not c["same"]]
    shared = listed([WORD[k].format(cfg[k]) for k in SHARED if check[k]["same"]])
    over = t0["last_update"] > t0["best_update"] and t0["val_nse_at_last"] < t0["val_nse_at_best"] and t0["train_loss_last"] < t0["train_loss_at_best"]
    kept_list = lambda v: listed([num(x) for x in v])
    ups = lambda n: f"{num(n)} update{'s' * (n != 1)}"
    rest1 = [t["best_update"] for t in one["other_trainings_without_curve"]]    # where the other seeds of the left column's setting were kept
    up_loss = t0["train_loss_rises_after_best_at"]                              # checks after the kept state at which the training loss is higher than at the check before
    grey = "still falls" if not up_loss else "falls further"                    # 'still falls' only when it falls at every check
    # an update scores the same number of days in every column: the one-river kept state, and the all-river update that comes after as many passes
    passes = (f"Updates are not passes through the data: {num(t0['best_update'])} on one river go through its training years about {t0['passes_at_best']:.0f} times, as "
              f"{num(PS['all_rivers_updates_with_as_many_passes'])} do on {n_riv} rivers.")

    # the left column is one training: say on the figure where the other seeds of its setting were kept (the details of the chosen setting are in the caption and the notes)
    left = "Left: the base setting of the tuning, " + (f"one of its {1 + len(rest1)} trainings (the other{'s were' if len(rest1) > 1 else ' was'} kept at "
                                                       f"update{'s' * (len(rest1) > 1)} {kept_list(rest1)})" if rest1 else "one training")
    left += (" - not the chosen setting" + ("." if rest1 else f" ({', '.join(diff)}).")) if diff else "; it is also the chosen setting."

    # ---- on the figure --------------------------------------------------------------------------------------------------
    title = "Learning curves of the LSTM: training loss, validation score and the state that was kept"
    foot = ["SMOKE TEST, not results."] * F.QUICK + [
            "Grey: training loss (weighted squared error on the training years); each point is the mean of the updates since the last check.",
            f"Blue: NSE in the validation years, mean of {L}. Dot and band: the state that was kept (highest validation score).",
            left,
            (f"Same network settings in all three columns ({SHORT_WORD['hidden'].format(cfg['hidden'])}, {SHORT_WORD['lr'].format(cfg['lr'])});" if D["same"] else
             f"The network settings of the left column differ from the others ({'; '.join(unlike)});") + " each column has its own scales.",
            passes,
            "Right: with the perfect weather forecast, an upper bound that cannot be had in practice."]

    # ---- for the documents: a caption, and notes ---------------------------------------------------------------------------
    see = ((f"on the left the blue line falls after update {num(t0['best_update'])} while the grey line {grey} (overfitting); " if over else
            f"on the left the state was kept at update {num(t0['best_update'])} of {num(t0['last_update'])}; ")
           + f"in the middle and on the right the blue lines peak at {upd(mid['best_update_min'], mid['best_update_max'])} and at {upd(right['best_update_min'], right['best_update_max'])}.")
    caption = (f"Learning curves of three training set-ups of the LSTM. Top: training loss. Bottom: NSE in the validation years, mean of {L} (left: of the one river; middle and right: "
               f"mean of the {n_riv} rivers); dot and band: the state that was kept. Left column: the base setting of the tuning, one training"
               + (f" (its other {plural(len(rest1), 'seed')} {'were' if len(rest1) > 1 else 'was'} kept at {'updates' if len(rest1) > 1 else 'update'} {kept_list(rest1)}; curves not stored)" if rest1 else "")
               + (f", not the chosen setting ({', '.join(diff)}). " if diff else ", which is also the chosen setting. ")
               + "Right column: with the perfect weather forecast, an upper bound that cannot be had in practice. Each column has its own scales. " + passes + " What to see: " + see)
    by_rule, at_limit = (sum(t["stopped_by"] == why for p in P for t in p["per_training"]) for why in ("no better score", "limit of updates"))
    notes = [
        f"The figure draws the curves that the trainer (flood.fit) stored at every validation check, for three training set-ups of the LSTM: one river ({D['river']}) with "
        f"{one['inputs']}, {plural(one['n_trainings'], 'training')}; all {n_riv} rivers with past weather + river gauge, {plural(mid['n_trainings'], 'training')}; and all {n_riv} rivers "
        f"with the perfect weather forecast as well, an upper bound that cannot be had in practice, {plural(right['n_trainings'], 'training')}. The all-river models also read "
        f"{D['statics']} basin properties. The middle and the right column are the trainings behind the report's {n_riv}-river results for these two sets of inputs.",
        f"Validation score (blue): the NSE in the validation years ({D['split']['val'][0]} to {D['split']['val'][1]}), mean of {L}. In the left column it is the NSE of the one river; in "
        f"the other two columns it is the mean over the {n_riv} rivers. It is the score that picks the state to keep, and the score by which the settings were chosen (the tuning on one "
        f"river, the setting check on all rivers); the report's tables of results give another number, the median of the {n_riv} rivers in the test years.",
        "Training loss (grey): the quantity the training reduces - the squared error of the forecast flow " + L + " on the training years, on a scaled flow and weighted per basin so that wet "
        f"and dry rivers count nearly alike (flood.fit). Each point is the mean over the updates since the last check ({one['check_every']} updates in the left column, {mid['check_every']} in the "
        "others), measured while training" + (", with dropout on; the validation score is measured with dropout off." if cfg["dropout"] else "."),
        f"The state that was kept (dot; the band spans the kept states of a column) is the check with the highest validation score; training stops after {cfg['patience']} checks without a better "
        "one. " + (f"All {n_all} trainings drawn stopped that way; none reached its limit of updates" if by_rule == n_all else
                   f"{by_rule} of the {n_all} trainings drawn stopped that way; {at_limit} reached the limit of updates first")
        + f" (the limit: {num(one['updates_allowed'])} updates in the left column, {num(mid['updates_allowed'])} in the others).",
        f"Left column: one training (seed {t0['seed']}) with the base setting of the tuning ({listed([SHORT_WORD[k].format(D['base'][k]) for k in ('hidden', 'lr', 'warmup')])}). It was kept at "
        f"update {num(t0['best_update'])} and stopped at update {num(t0['last_update'])}."
        + (f" The base setting was trained with {1 + len(one['other_trainings_without_curve'])} seeds, kept at updates "
           f"{kept_list([t0['best_update']] + [t['best_update'] for t in one['other_trainings_without_curve']])}; only the curve of the first was stored (tuning.base_curve)."
           if one["other_trainings_without_curve"] else ""),
    ]
    TN = N["one_river_tuning"]                                                  # is the one curve that was stored a typical one? where the other tuning trainings on this river were kept
    if TN["n_trainings"] > 1:
        n_t, lo_t, hi_t, rule_t = TN["n_trainings"], TN["best_update_min"], TN["best_update_max"], TN["n_stopped_by_no_better_score"]
        n_base = sum(not t["differs_from_base"] for t in TN["per_training"])
        how = (f" (the base setting with {plural(n_base, 'seed')}, and one setting changed at a time)" if all(len(t["differs_from_base"]) <= 1 for t in TN["per_training"]) else "")
        notes.append(
            f"Is the training of the left column a typical one? It is the only one-river curve that was stored (03_lstm.py keeps the curve of its first training). Of the {n_t} trainings of the "
            f"tuning on this river with {TN['inputs']}{how}, lstm_single.json (tuning.runs) holds where each was kept: "
            + (f"all at update {num(lo_t)}" if lo_t == hi_t else
               f"between update {num(lo_t)} and update {num(hi_t)} (median {TN['best_update_median']:g}; {TN['n_kept_at_min']} of the {n_t} at update {num(lo_t)})")
            + (f", and all {n_t} ended because {one['patience']} checks had passed without a better validation score." if rule_t == n_t else
               f"; {rule_t} of the {n_t} ended because {one['patience']} checks had passed without a better validation score.")
            + (f" So a peak within the first {num(hi_t)} updates is common to all of them; update {num(lo_t)}, where the curve drawn peaks, is the earliest of them."
               if t0["best_update"] == lo_t < hi_t and rule_t == n_t else ""))
    if C["same_as_base"]:
        notes.append("The setting the tuning chose in the end is the base setting, so the left column is the first training of the one-river model that the report uses.")
    else:
        b = [t["best_update"] for t in C["per_training"]]
        notes.append(
            f"The left column is not the one-river model that the report uses. That model has the setting the tuning chose in the end ({listed(diff)}); its curves were not stored, only "
            f"where each of its {plural(len(b), 'training')} was kept: " + (f"update {num(b[0])}" if len(b) == 1 else f"updates {kept_list(b)} ({listed([nth(i) for i in range(len(b))])} seed)")
            + f". The report's sentence 'early stopping kept the first seed as it was at update {num(b[0])}' is about that model"
            + (f"; in the left column of the figure update {num(t0['last_update'])} is where the base-setting training stopped, after being kept at update {num(t0['best_update'])}."
               if b[0] == t0["last_update"] else "."))
    FN = N["one_river_chosen_setting"]["by_input_set"]                          # one river, the chosen setting, each set of inputs: does the perfect weather forecast move the kept state there too?
    kept_of = lambda g: upd(g["best_update_min"], g["best_update_max"]) + f" ({plural(g['n_trainings'], 'seed')})"
    with_f, without_f = ([g for g in FN.values() if g["perfect_weather_forecast"] == f] for f in (True, False))
    if with_f and without_f:
        notes.append(
            "The one-river trainings of the " + ("chosen setting" if not C["same_as_base"] else "base setting, which is also the chosen one,") + " (lstm_single.json finals; curves not stored) were kept at: "
            + "; ".join(f"{g['inputs']} - {kept_of(g)}" for g in FN.values()) + "."
            + (" So on one river, too, the kept state comes later with the perfect weather forecast, an upper bound that cannot be had in practice."
               if min(g["best_update_min"] for g in with_f) > max(g["best_update_max"] for g in without_f) else ""))
    sizes = "; ".join(f"{p['n_inputs']} inputs and {num(p['n_weights'])} weights" for p in P)
    notes.append(
        (f"The three columns use the same network settings (the script compares tuning.base and the first tuning run of lstm_single.json with config of lstm_all.json): {shared}. " if D["same"] else
         f"The network settings of the left column differ from the other two: {'; '.join(unlike)}. Shared: {shared}. ")
        + f"The columns differ in the inputs and so in the number of weights ({sizes}, from left to right), in the check interval "
        f"({listed([str(p['check_every']) for p in P])} updates) and in the limit of updates ({listed([num(p['updates_allowed']) for p in P])}).")
    per = lambda p: "about " + (f"{p['passes_at_best_min']:.0f}" if round(p["passes_at_best_min"]) == round(p["passes_at_best_max"]) else f"{p['passes_at_best_min']:.0f} to {p['passes_at_best_max']:.0f}")
    spu, seeds1 = one["scored_days_per_update"], [t["passes_at_best"] for t in one["per_training"] + one["other_trainings_without_curve"]]
    notes.append(
        "Updates are not passes through the data. "
        + (f"In every column an update scores {cfg['batch']} stretches of {cfg['block']} days ({num(spu)} days): {spu / one['training_days']:.1f} times the {num(one['training_days'])} "
           f"training days of one river, but {spu / mid['training_days']:.0%} of the {num(mid['training_days'])} training days of {n_riv} rivers. " if check["batch"]["same"] and check["block"]["same"] else "")
        + "Counted in passes through the training years (epochs: the days scored up to the kept state divided by the training days; key passes_at_best; the stretches are drawn at random, so a "
        f"pass is an average), the kept states are at {per(one)} passes on one river"
        + (f" (the {len(seeds1)} seeds of the base setting: about {listed([f'{v:.0f}' for v in seeds1])})" if len(seeds1) > 1 else "")
        + f", {per(mid)} on {n_riv} rivers and {per(right)} with the perfect weather forecast."
        + (f" So the later peak of the middle column, counted in updates, goes with its training set, {n_riv} times as large: counted in passes it comes at about the same point as on one river."
           if mid["passes_at_best_min"] <= max(seeds1) and min(seeds1) <= mid["passes_at_best_max"] else "")
        + f" The footnote of the figure says it with one pair of numbers: {ups(t0['best_update'])} on one river and {ups(PS['all_rivers_updates_with_as_many_passes'])} on {n_riv} rivers are "
        f"both about {t0['passes_at_best']:.0f} passes (key passes).")
    # how far a training runs on after its kept state is set by the stopping rule, in checks: the falls after the dots are not like-for-like between the columns
    ran, same_p, fall = PS["run_on_after_kept_state"], PS["one_river_as_many_passes_after_kept_state_as_all_rivers"], PS["val_nse_fall_kept_state_to_last_check"]
    rng = lambda d: n2(d["min"]) if n2(d["min"]) == n2(d["max"]) else f"{n2(d['min'])} to {n2(d['max'])}"
    if by_rule == n_all:
        notes.append(
            "After its kept state a training runs on until " + (f"{cfg['patience']} checks have passed without a better score" if check["patience"]["same"] else "the stopping rule ends it")
            + f": {ups(ran[one['key']]['updates'])} (about {ran[one['key']]['passes']:.0f} "
            f"passes) in the left column, {ups(ran[mid['key']]['updates'])} (about {ran[mid['key']]['passes']:.0f} passes) in the others. So the falls of the blue lines after the dots are not "
            "like-for-like between the columns. "
            + (f"On one river the validation NSE is {n2(same_p['val_nse_fall'])} below the kept state {ups(same_p['updates_after_kept_state'])} later (update {num(same_p['update'])}, about "
               f"{ran[mid['key']]['passes']:.0f} passes later) and {n2(t0['val_nse_fall_after_best'])} below it at the last check" if same_p else
               f"On one river the validation NSE is {n2(t0['val_nse_fall_after_best'])} below the kept state at the last check")
            + f"; on all {n_riv} rivers it is {rng(fall[mid['key']])} below it at the last check, and {rng(fall[right['key']])} with the perfect weather forecast. That the validation score is lower "
            "at the last check than at the kept state follows from the stopping rule itself; what the curves add is the size of that fall"
            + (" and the training loss, which falls further." if all(t["train_loss_last"] < t["train_loss_at_best"] for p in P for t in p["per_training"]) else "."))
    loss_note = "" if not up_loss else ("; it rises once on the way, at update " + num(up_loss[0]) if len(up_loss) == 1 else f"; not at every check: it is higher than at the check before at updates {kept_list(up_loss)}")
    notes.append(
        "What to see: "
        + (f"in the left column the blue line peaks at update {num(t0['best_update'])} (NSE {n2(t0['val_nse_at_best'])}) and has fallen to {n2(t0['val_nse_at_last'])} by update "
           f"{num(t0['last_update'])}, while the grey line {grey} ({n2(t0['train_loss_at_best'])} at update {num(t0['best_update'])}, {n2(t0['train_loss_last'])} at update {num(t0['last_update'])}; "
           f"{n2(t0['train_loss_first'])} at the first check{loss_note}): the network fits its training years better and the validation years worse (overfitting), and early stopping keeps the "
           "state at the peak. "
           if over else f"in the left column the state was kept at update {num(t0['best_update'])} of {num(t0['last_update'])}. ")
        + f"With all {n_riv} rivers the kept states are at {upd(mid['best_update_min'], mid['best_update_max'])}, and with the perfect weather forecast at "
        f"{upd(right['best_update_min'], right['best_update_max'])}"
        + ("; there too, in every training, the training loss and the validation score are both lower at the last check than at the kept state."
           if all(t["train_loss_last"] < t["train_loss_at_best"] and t["val_nse_at_last"] < t["val_nse_at_best"] for p in (mid, right) for t in p["per_training"]) else "."))
    AN = N["all_rivers"]                                                        # the all-river trainings with the sets of inputs that are not drawn
    if AN:
        late = lambda g: g["n_trainings"] - g["n_stopped_by_no_better_score"]
        notes.append(
            f"Not drawn: the all-river trainings with the other {plural(len(AN), 'set')} of inputs (their curves are in lstm_all.json). "
            + "; ".join(f"{'with' if i else 'With'} {g['inputs']} they were kept at {upd(g['best_update_min'], g['best_update_max'])}"
                        + (f", and {late(g)} of the {g['n_trainings']} reached the limit of {num(mid['updates_allowed'])} updates before the stopping rule was met"
                           if late(g) and all(t["stopped_by"] in ("no better score", "limit of updates") for t in g["per_training"]) else "") for i, g in enumerate(AN.values())) + ".")
    for p, column in zip(P, ("left", "middle", "right")):  # a dip after the kept state that the curve recovers from: say that it is in the data
        for i, t in enumerate(p["per_training"]):
            if t["update_of_lowest_after_best"] < t["last_update"] and t["val_nse_at_last"] - t["val_nse_lowest_after_best"] > 0.03:
                notes.append(f"The dip in the {column} column is in the stored curve and is drawn as it is: the {nth(i)} training scores NSE {n2(t['val_nse_lowest_after_best'])} at update "
                             f"{num(t['update_of_lowest_after_best'])} and {n2(t['val_nse_at_last'])} at its last check, update {num(t['last_update'])}.")
    if F.QUICK:
        notes.append("QUICK smoke test: the curves are those of the smoke-test trainings (a few updates) - numbers are not results.")
    return title, foot, caption, notes


def wrap(fig, text, width_px, **font):
    """One statement broken into lines no wider than width_px, measured with the font it is drawn in."""
    probe, r, lines = fig.text(0, 0, "", **font), fig.canvas.get_renderer(), [""]
    for w in text.split():
        probe.set_text((lines[-1] + " " + w).strip())
        if lines[-1] and probe.get_window_extent(r).width > width_px:
            lines.append(w)
        else:
            lines[-1] = probe.get_text()
    probe.remove()
    return lines


def ticks(lo, hi, least):
    """Round values inside lo..hi for a y-axis, at the finest step of 1, 2 or 5 x 10^k that is not smaller than `least`."""
    step = next(m * 10.0 ** k for k in range(-6, 6) for m in (1, 2, 5) if m * 10.0 ** k >= least)
    return np.arange(np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9) + 1) * step


def lettering(fig):
    """The smallest letter size, and the lettering that overlaps other lettering, leaves the figure or lies on a curve (there should be none).
    'On a curve' counts the drawn width: a label must stay 1 point clear of the edge of every line and of every marker."""
    fig.canvas.draw()
    r, (w, h) = fig.canvas.get_renderer(), fig.canvas.get_width_height()
    unseen = {id(t.label1) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for t in axis.get_major_ticks()
              if not min(axis.get_view_interval()) <= t.get_loc() <= max(axis.get_view_interval())}        # tick labels outside the axis range are not drawn
    T = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in unseen]
    box = [t.get_window_extent(r) for t in T]
    name = lambda t: " ".join(t.get_text().split())[:60]
    bad = [f"'{name(a)}' overlaps '{name(c)}'" for i, (a, p) in enumerate(zip(T, box)) for c, q in list(zip(T, box))[i + 1:] if p.overlaps(q)]
    bad += [f"'{name(t)}' leaves the figure" for t, b in zip(T, box) if b.x0 < 0 or b.y0 < 0 or b.x1 > w or b.y1 > h]
    for ax in fig.axes:                                     # the labels inside a panel against the curves of that panel
        for t in ax.texts:
            b = t.get_window_extent(r)
            for ln in ax.lines:
                xy = ax.transData.transform(ln.get_xydata())
                xy = np.concatenate([np.linspace(a, c, 50) for a, c in zip(xy[:-1], xy[1:])]) if len(xy) > 1 else xy
                drawn = max(ln.get_linewidth() * (ln.get_linestyle() not in ("None", "none", "", " ")),
                            (ln.get_markersize() + ln.get_markeredgewidth()) * (ln.get_marker() not in ("None", "none", "", " ", None)))
                pad = (drawn / 2 + 1) * fig.dpi / 72              # half the drawn width plus 1 point, in pixels
                if ((xy[:, 0] > b.x0 - pad) & (xy[:, 0] < b.x1 + pad) & (xy[:, 1] > b.y0 - pad) & (xy[:, 1] < b.y1 + pad)).any():
                    bad.append(f"'{name(t)}' lies on a curve")
                    break
    return min(t.get_fontsize() for t in T), bad


def draw(D, title, foot):
    """3 columns (the set-ups) by 2 rows (training loss; validation score). Returns the facts of the figure, with what the lettering check found."""
    P, blue = D["P"], F.MODEL_COL["lstm"]
    fig, axes = plt.subplots(2, len(P), figsize=(W_IN, H_IN), sharex="col", squeeze=False)
    font = dict(fontsize=FS, color=INK2, linespacing=1.18)
    lines = [ln for s in foot for ln in wrap(fig, s, (W_IN - 0.16) * fig.dpi, **font)]
    span = lambda v: (max(v) - min(v)) or 1.0
    scales = []
    for j, p in enumerate(P):
        top, bot = axes[0, j], axes[1, j]
        for t in p["per_training"]:
            c = t["curve"]                                  # a small dot at every check: between two checks nothing was measured
            top.plot(c["update"], c["train_loss"], color=INK2, lw=1.4, alpha=0.85, marker="o", ms=2.6, markeredgewidth=0)
            bot.plot(c["update"], c["val_nse"], color=blue, lw=1.6, alpha=0.9, marker="o", ms=2.6, markeredgewidth=0)
            bot.plot([t["best_update"]], [t["val_nse_at_best"]], ls="none", marker="o", ms=6.6, markerfacecolor=blue, markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=5)
        lo, hi = p["best_update_min"], p["best_update_max"]
        half = 0.012 * p["last_update_max"] if lo == hi else 0                    # one kept state: a band wide enough to see
        loss, val = [x for t in p["per_training"] for x in t["curve"]["train_loss"]], [x for t in p["per_training"] for x in t["curve"]["val_nse"]]
        for ax in (top, bot):
            ax.axvspan(lo - half, hi + half, color=blue, alpha=0.09, lw=0, zorder=0)
            ax.tick_params(labelsize=FS, length=0)
            ax.grid(axis="x", visible=False)               # upright grid lines would run through the labels; the band marks the updates that matter
            ax.set_xlim(0, None)
            ax.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10], integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        kept_at = top.text(0.98, 1 - EDGE, "kept at " + upd(lo, hi), transform=top.transAxes, ha="right", va="top", fontsize=FS, color=INK2)
        score_is = bot.text(0.5, EDGE, p["validation_nse_label"], transform=bot.transAxes, ha="center", va="bottom", fontsize=FS, color=INK2)
        scales += [(top, loss, kept_at, True), (bot, val, score_is, False)]     # panel, its values, the label inside it, and whether that label stands above the curves
        for i, line in enumerate(reversed(p["lines"])):                           # the lines under the column title, the lowest first
            top.annotate(line, (0, 1), xycoords="axes fraction", xytext=(0, 5 + 10.8 * i), textcoords="offset points", ha="left", va="bottom", fontsize=FS, color=INK2)
        top.annotate(p["title"], (0, 1), xycoords="axes fraction", xytext=(0, 6.5 + 10.8 * len(p["lines"])), textcoords="offset points", ha="left", va="bottom",
                     fontsize=9.2, fontweight="bold", color=INK)
        bot.tick_params(axis="x", length=3, color=AXIS)
        bot.set_xlabel("updates of the weights", fontsize=FS + 0.2)
    axes[0, 0].set_ylabel("training loss\n(0 = perfect fit)", fontsize=FS + 0.2)
    axes[1, 0].set_ylabel("validation NSE\n(1 = perfect)", fontsize=FS + 0.2)
    fig.align_ylabels(axes[:, 0])
    fig.suptitle(title, x=0.012, y=1 - 0.07 / H_IN, ha="left", va="top", fontsize=10.4, fontweight="bold", color=INK)
    note = fig.text(0.012, 0.05 / H_IN, "\n".join(lines), ha="left", va="bottom", **font)
    foot_in = 0.05 + note.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi          # the top of the footnote above the lower edge, in inches
    fig.subplots_adjust(left=0.112, right=0.985, top=1 - 0.83 / H_IN, bottom=(foot_in + 0.51) / H_IN, hspace=0.13, wspace=0.30)
    r = fig.canvas.get_renderer()
    for ax, v, label, above in scales:
        # The scale of a panel keeps a strip free of curves for the label inside it: above the loss curves ('kept at ...'), below the score curves (what the
        # score is). The strip holds the label, its distance from the edge and 0.06 in of air, whatever the height of the panel, so no curve can reach a label.
        h = ax.get_position().height * H_IN
        strip, rest = EDGE + (label.get_window_extent(r).height / fig.dpi + 0.06) / h, (0.052 if above else 0.086)       # shares of the panel height: label side, other side
        full = span(v) / (1 - strip - rest)
        ax.set_ylim((min(v) - rest * full, max(v) + strip * full) if above else (min(v) - strip * full, max(v) + rest * full))
        ax.set_yticks(ticks(min(v), max(v), 0.13 / h * full))  # grid lines only where there are curves (none through the labels), no closer than their labels need (0.13 in)
    smallest, bad = lettering(fig)
    fig.savefig(F.FIG_OUT / "training_curves.png", dpi=DPI)
    plt.close(fig)
    return dict(file=f"figures/{'quick/' * F.QUICK}training_curves.png", width_in=W_IN, height_in=H_IN, dpi=DPI, smallest_lettering_pt=smallest, lettering_problems=bad,
                title=title, footnote=lines)


if __name__ == "__main__":
    gone = [n for n in ("lstm_single", "lstm_all") if not (F.RES_OUT / f"{n}.json").exists()]
    if gone:
        where = "results/quick" if F.QUICK else "results"
        msg = f"{SCRIPT}: {' and '.join(f'{where}/{n}.json' for n in gone)} missing (run 03_lstm.py and 04_lstm_all.py first) - no learning curves drawn"
        if F.QUICK:                                         # a smoke test of this script alone: nothing to draw from, and that is not an error
            print(msg)
            sys.exit(0)
        sys.exit(msg)
    try:
        D = read()
    except (KeyError, IndexError, AssertionError, TypeError, ValueError) as e:
        if F.QUICK:                                         # smoke-test files of different runs of 03 and 04 may lack a key or not fit together
            print(f"{SCRIPT}: the smoke-test results files cannot be used ({e!r}) - no learning curves drawn")
            sys.exit(0)
        raise
    title, foot, caption, notes = words(D)
    figure = draw(D, title, foot)
    assert figure["smallest_lettering_pt"] >= FS, "lettering smaller than the smallest size allowed"
    P, cfg = D["P"], D["cfg"]
    R = dict(question="What does the training of the LSTM look like, and where did early stopping keep it?", quick_mode=F.QUICK,
             source="results/lstm_single.json (tuning.base, tuning.budget, tuning.final_budget, tuning.runs, tuning.base_curve, tuning.chosen, finals, diagnostic_fixed_length) and "
                    "results/lstm_all.json (config, sets.wq and sets.wqf: per_seed.<seed>.curve, cost.<seed>; the other sets: cost.<seed>)",
             model="LSTM: one LSTM layer and a linear output layer that gives the flow 1, 2 and 3 days ahead (flood.LSTMNet)",
             same_settings_one_river_and_all=D["same"], settings={k: cfg[k] for k in SHARED}, settings_check=D["check"],
             check_every={"one_river": P[0]["check_every"], "all": cfg["eval_every"]}, updates_allowed={"one_river": P[0]["updates_allowed"], "all": cfg["updates"]},
             split=D["split"],
             panels={p["key"]: {k: v for k, v in p.items() if k not in ("key", "lines")} for p in P},
             chosen_one_river=D["chosen"], passes=D["passes"], not_drawn=D["not_drawn"], figure=figure, caption=caption, notes=notes)
    R["runtime_s"] = round(time.time() - t_start, 1)
    (F.RES_OUT / "training_curves.json").write_text(json.dumps(R, indent=1))

    for p in P:
        print(f"{p['title']:30s} {p['inputs']}: " + "; ".join(f"kept at update {t['best_update']} of {t['last_update']} (validation NSE {t['val_nse_at_best']:.3f} -> {t['val_nse_at_last']:.3f}, "
                                                              f"training loss {t['train_loss_first']:.2f} -> {t['train_loss_last']:.2f})" for t in p["per_training"]))
    print(f"same network settings in the three columns: {D['same']}  {R['settings']}")
    print("chosen one-river model kept at updates", [t["best_update"] for t in D["chosen"]["per_training"]], "- it is the left column:", D["chosen"]["first_seed_is_left_column"])
    print("caption:", caption)
    print("notes:", *notes, sep="\n  ")
    if figure["lettering_problems"]:
        print("LETTERING TO CHECK:", *figure["lettering_problems"], sep="\n  ")
    print(f"wrote {F.RES_OUT / 'training_curves.json'} and {F.FIG_OUT / 'training_curves.png'} ({W_IN} x {H_IN} in, smallest lettering {figure['smallest_lettering_pt']:g} pt) in {R['runtime_s']} s")
