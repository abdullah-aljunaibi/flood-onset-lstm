"""Week 5 - remote-sensing experiment: do satellite rain, soil wetness and precipitable water change the forecast skill?

All models are the all-basin LSTM of week 3 (one model for the 12 basins, basin properties as extra inputs). Two experiments:
(a) Rain source. Rain gauges are rare in most of the world; satellites see rain everywhere. The LSTM is given gauge-based rain
    (Daymet), satellite rain (GPM IMERG) instead, or both - on a SATELLITE-ERA split (training 2000-2008, validation 2008-2010,
    test 2010-2014), because the satellite record is short. These numbers are not comparable with the main split.
(b) Extra inputs on the main split: + soil wetness (top layer and root zone), + precipitable water, + both. Both come from the
    MERRA-2 reanalysis (a weather model that assimilates satellite observations), not from a direct satellite retrieval.
Within an experiment every variant - the base too - gets exactly the same days of data, so differences come from the inputs only.
Each variant is trained with several random seeds; a difference counts only when it is larger than the seed-to-seed noise.
Needs the tables of fetch_satellite.py. Writes results/satellite_inputs.json, figures/satellite_inputs.png and the prediction
sets sat_<variant>_<input set>. Without training: --rescore redoes the comparison, findings and notes from the scores saved in the JSON
and redraws the figure; --figure only redraws.
"""
import json
import sys
import time
from dataclasses import asdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import flood as F
from common import *
from fetch_satellite import satellite_table

use_style()
SCRIPT = "08_satellite_inputs.py"
SAT_SPLIT = {"train": ("2000-10-01", "2008-09-30"), "val": ("2008-10-01", "2010-09-30"), "test": ("2010-10-01", "2014-09-30")}
SETS = {"w": dict(use_flow=False), "wq": dict(use_flow=True)}
NO_RAIN = [c for c in F.FORCING if c != "prcp"]
KEYS = ("hidden", "lr", "dropout", "warmup", "block", "batch", "updates", "eval_every", "patience")
SEEDS, CAP = [0, 1], dict(updates=1000, patience=5)                      # the budget of this laptop: about 30 minutes for the whole script
# experiment -> variant -> (plain label, daily input columns, input sets it is trained for); the first variant is the base
EXPERIMENTS = {
    "a_rain_source": dict(
        title="Rain source", split=SAT_SPLIT, needs=["imerg"],
        question="How much skill is lost when rain comes from a satellite instead of rain gauges, and does satellite rain add anything on top of gauge rain?",
        variants={"gauge": ("gauge-based rain (Daymet)", F.FORCING, ("w", "wq")),
                  "imerg": ("satellite rain (IMERG) instead of gauge-based rain", ["imerg"] + NO_RAIN, ("w", "wq")),
                  "imerg_nosun": ("satellite rain, without Daymet's sunlight estimate", ["imerg"] + [c for c in NO_RAIN if c != "srad"], ("w",)),
                  "gauge_imerg": ("gauge-based rain + satellite rain", F.FORCING + ["imerg"], ("w", "wq"))}),
    "b_extra_inputs": dict(
        title="Extra inputs", split=F.SPLIT, needs=["gwettop", "gwetroot", "pw"],
        question="Does knowing how wet the soil is help - and more so without a river gauge? Does precipitable water help to warn of the onset of high flow?",
        variants={"base": ("no extra input", F.FORCING, ("w", "wq")),
                  "soil": ("+ soil wetness (top layer and root zone)", F.FORCING + ["gwettop", "gwetroot"], ("w", "wq")),
                  "pw": ("+ precipitable water", F.FORCING + ["pw"], ("w", "wq")),
                  "soil_pw": ("+ soil wetness + precipitable water", F.FORCING + ["gwettop", "gwetroot", "pw"], ("w", "wq"))}),
}
STYLE = {"imerg": (ORANGE, "o"), "imerg_nosun": (YELLOW, "D"), "gauge_imerg": (INK2, "s"), "soil": (AQUA, "o"), "pw": (RED, "D"), "soil_pw": (INK2, "s")}
METRICS = {"nse_median": lambda s: s["nse_median"], "onset_pod": lambda s: s["onset"]["pod"], "onset_far": lambda s: s["onset"]["far"],
           "onset_csi": lambda s: s["onset"]["csi"], "all_csi": lambda s: s["all"]["csi"]}
if F.QUICK:
    SEEDS, CAP = [0], dict(updates=8, eval_every=4, patience=5)
TAIL = ("n_trainings", "training_seconds", "waited_for_cpu_s", "runtime_s")
lk = lambda k: f"l{k}"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"
saved_as = lambda v, inp: f"sat_{v}_{inp}"


def settings():
    """The LSTM settings chosen by 04_lstm_all.py on the validation years if that result exists, otherwise the engine defaults; capped to the budget."""
    f, first = RES / "lstm_all.json", RES / "preds" / "lstm_wq_s0.npz"         # 04 saves its first final model, with the chosen settings, before it writes the JSON
    src = json.loads(f.read_text()) if f.exists() else {}
    if src.get("config") and not src.get("quick_mode"):
        c, where = dict(src["config"]), "results/lstm_all.json, key 'config' (chosen there on the validation years of the main split)"
    elif first.exists() and json.loads(str(np.load(first, allow_pickle=False)["meta"])).get("config"):
        c = dict(json.loads(str(np.load(first, allow_pickle=False)["meta"]))["config"])
        where = "the settings stored with the prediction set lstm_wq_s0 of 04_lstm_all.py (chosen there on the validation years of the main split; results/lstm_all.json was not written yet)"
    else:
        c, where = {k: asdict(F.Cfg())[k] for k in KEYS}, "the defaults of flood.Cfg (no result of 04_lstm_all.py existed when the training started)"
    c["updates"], c["patience"] = min(c["updates"], CAP["updates"]), min(c["patience"], CAP["patience"])
    c["eval_every"] = CAP.get("eval_every", c["eval_every"])
    return c, where


def spread(v):
    """Typical difference between two trainings of the same model: the mean absolute difference over all pairs of seeds."""
    return float(np.mean([abs(a - b) for a, b in combinations(v, 2)])) if len(v) > 1 else float("nan")


def chance_rate(n_variants, n_seeds, n=200000):
    """How often the 'larger than seed noise' rule fires when there is NO real difference (simulated: scores differ by random seed noise only)."""
    if n_seeds < 2:
        return float("nan")
    x = np.random.default_rng(0).normal(size=(n, n_variants, n_seeds))
    noise = np.mean([np.abs(x[:, :, i] - x[:, :, j]) for i, j in combinations(range(n_seeds), 2)], axis=0).mean(1)
    a, b = x[:, 1], x[:, 0]
    return float(((np.abs(a.mean(1) - b.mean(1)) > 2 * noise) & ((a.min(1) > b.max(1)) | (a.max(1) < b.min(1)))).mean())


def run():
    t0 = time.perf_counter()
    D0, B = F.load_data(), F.basins()
    ids = list(D0)
    sat = {g: satellite_table(g) for g in ids}
    cols = sorted({c for e in EXPERIMENTS.values() for c in e["needs"]})
    D = {g: D0[g].join(sat[g][cols]) for g in ids}
    assert all(D[g].index.equals(D0[g].index) for g in ids)
    n_total, done = sum(len(sets) for e in EXPERIMENTS.values() for _, _, sets in e["variants"].values()) * len(SEEDS), [0]

    # Daymet does not measure sunlight: is its estimate lower on days when its rain gauges report rain? (it could then tell a model about gauge rain)
    sun = []
    for g in ids:
        d = D0[g].loc[SAT_SPLIT["train"][0]:SAT_SPLIT["train"][1]]
        by = d.assign(wet=d.prcp >= 1.0, month=d.index.month).groupby(["month", "wet"]).srad.mean().unstack()
        sun.append(float((by[True] / by[False]).mean()))

    E, P_all = {}, {}
    t_ask = time.perf_counter()
    with F.cpu_lock(SCRIPT):
        t1 = time.perf_counter()
        config, config_source = settings()                                                      # read when the training starts, i.e. after the wait for the CPU
        print(f"settings from {config_source}: {config}; seeds {SEEDS}; {n_total} trainings", flush=True)
        for name, exp in EXPERIMENTS.items():
            split = exp["split"]
            start = max(D[g][c].first_valid_index() for g in ids for c in exp["needs"])        # first day on which the latest-starting input is valid
            Dx = {g: d.loc[start:] for g, d in D.items()}                                       # the SAME days for every variant of the experiment
            assert not any(Dx[g][exp["needs"]].isna().any().any() for g in ids), "gaps in the satellite inputs"
            thr = F.thresholds(D0, split=split)                                                 # 95th percentile of the training years of THIS split
            E[name] = dict(title=exp["title"], question=exp["question"], split=split, data_start=str(start.date()), thresholds=thr, base=next(iter(exp["variants"])), variants={})
            for v, (label, dyn, sets) in exp["variants"].items():
                E[name]["variants"][v] = dict(label=label, daily_inputs=list(dyn), sets={})
                for inp in sets:
                    Ps, per_seed = [], {}
                    for s in SEEDS:
                        r = F.fit(F.Cfg(dyn=tuple(dyn), static=tuple(F.STATIC), seed=s, **SETS[inp], **config), Dx, ids, split=split, static_df=B)
                        P = {g: p.reindex(D0[g].index).assign(obs=D0[g].q_mm) for g, p in r.predict(Dx, static_df=B).items()}      # back on the whole record
                        ev = {part: F.evaluate(P, thr, part, split=split) for part in ("val", "test")}
                        assert all(ev[part]["summary"][lk(k)]["coverage_min"] == 1 for part in ev for k in F.LEADS), f"{v}/{inp}: days without a forecast in the validation or test years"
                        Ps.append(P)
                        per_seed[str(s)] = dict(test=ev["test"]["summary"], val=ev["val"]["summary"], cost=r.summary(), best_update_near_the_limit=r.best_update > config["updates"] - config["eval_every"] * 2,
                                                first_forecast_day=str(P[ids[0]].l1.first_valid_index().date()),
                                                test_nse_by_basin={g: [ev["test"]["basins"][g][lk(k)]["nse"] for k in F.LEADS] for g in ids})
                        done[0] += 1
                        print(f"training {done[0]:2d} of {n_total} | {exp['title']} ({split['test'][0][:4]}-{split['test'][1][:4]} test) | {label} | {F.INPUT_LABEL[inp]} | seed {s} | best at update "
                              f"{r.best_update} of {r.updates_run}, {r.seconds:.0f} s | validation NSE {r.best_val_nse:.4f} | test NSE median by lead "
                              + " ".join(f"{ev['test']['summary'][lk(k)]['nse_median']:.3f}" for k in F.LEADS) + f" | {time.perf_counter() - t1:.0f} s so far", flush=True)
                    PE = F.ensemble(Ps)
                    P_all[saved_as(v, inp)] = PE
                    F.save_preds(saved_as(v, inp), PE, dict(script=SCRIPT, model="lstm", inputs=inp, variant=v, label=label, daily_inputs=list(dyn), experiment=name, split=split,
                                                            data_start=str(start.date()), seeds=SEEDS, config=config, what="one LSTM for all 12 basins, day-by-day average of the seeds"))
                    E[name]["variants"][v]["sets"][inp] = dict(saved_as=saved_as(v, inp), input_set=F.INPUT_LABEL[inp], seeds=SEEDS, per_seed=per_seed,
                                                               test=F.evaluate(PE, thr, "test", split=split), val=F.evaluate(PE, thr, "val", split=split)["summary"])
        train_s = time.perf_counter() - t1
    return analyse(dict(quick_mode=F.QUICK, config=config, config_source=config_source, seeds=SEEDS, static_inputs=list(F.STATIC), split_main=F.SPLIT, split_satellite_era=SAT_SPLIT,
                        satellite_data="results/satellite_data.json (source, grid, coverage, rain check)",
                        daymet_sunlight_on_rain_days=dict(ratio_wet_to_dry_median=float(np.median(sun)), by_basin=dict(zip(ids, sun))),
                        basins={g: dict(short=F.SHORT[g], landscape=B.landscape[g]) for g in ids}, experiments=E, n_trainings=done[0], training_seconds=round(train_s, 1),
                        waited_for_cpu_s=round(t1 - t_ask, 1), runtime_s=round(time.perf_counter() - t0 - (t1 - t_ask), 1)))


def next_day_rain(ids):
    """How much each input of issue day d already tells about the rain Daymet books on day d+1 (correlations, training years of experiment (a))."""
    D0 = F.load_data()
    a, b = SAT_SPLIT["train"]
    rows = {}
    for g in ids:
        d = D0[g].join(satellite_table(g)[["imerg", "gwettop", "pw"]]).loc[a:b]
        x = {"gauge-based rain (Daymet) itself": d.prcp, "satellite rain (IMERG)": d.imerg, "day-to-day change of top-layer soil wetness (MERRA-2)": d.gwettop.diff(), "precipitable water (MERRA-2)": d.pw}
        for name, v in x.items():
            rows.setdefault(name, []).append((v.corr(d.prcp), v.corr(d.prcp.shift(-1))))
    return {name: dict(correlation_with_daymet_rain_of_the_same_day=float(np.median([r[0] for r in v])), correlation_with_daymet_rain_of_the_next_day=float(np.median([r[1] for r in v])))
            for name, v in rows.items()}


def analyse(R):
    """Every variant against the base of its experiment (difference, seed noise, basins improved), the findings and the notes; writes the JSON."""
    E, ids, seeds, config_source, n_total = R["experiments"], list(R["basins"]), R["seeds"], R["config_source"], R["n_trainings"]
    sat_json = F.RES_OUT / "satellite_data.json"                                                    # written by fetch_satellite.py; a smoke test writes it under results/quick
    lagc = json.loads((sat_json if sat_json.exists() else RES / "satellite_data.json").read_text())["rain_check_summary"]["imerg"]
    ahead = next_day_rain(ids)
    # how much ONE basin's score differs between two trainings of the same model (1 day ahead): the yardstick for n_basins_improved
    one = sorted((abs(a - b), R["basins"][g]["short"], x["label"], F.INPUT_LABEL[inp]) for exp in E.values() for x in exp["variants"].values() for inp, s in x["sets"].items()
                 for g in ids for a, b in combinations([r["test_nse_by_basin"][g][0] for r in s["per_seed"].values()], 2))
    basin_noise = dict(what="absolute difference of one basin's test NSE, 1 day ahead, between two trainings of the same model; all experiments, variants and input sets",
                       n_pairs=len(one), median=float(np.median([d[0] for d in one])), largest=one[-1][0], largest_where=", ".join(one[-1][1:])) if one else None
    for name, exp in E.items():
        V, base = exp["variants"], exp["base"]
        first = {(v, inp, s): r["first_forecast_day"] for v in V for inp, x in V[v]["sets"].items() for s, r in x["per_seed"].items()}
        assert len(set(first.values())) == 1, f"{name}: variants start on different days {first}"
        exp["first_forecast_day"], exp["equal_days_check"] = next(iter(first.values())), "every variant and seed: same first forecast day, coverage 1.0 in the validation and test years (asserted)"
        exp["seed_noise"], exp["against_base"] = {}, {}
        for inp in SETS:
            seeds_of = lambda v, m, k: [METRICS[m](r["test"][lk(k)]) for r in V[v]["sets"][inp]["per_seed"].values()]
            noise = {m: {lk(k): float(np.nanmean([spread(seeds_of(v, m, k)) for v in V if inp in V[v]["sets"]])) for k in F.LEADS} for m in METRICS}
            exp["seed_noise"][inp] = noise
            for v in V:
                if v == base or inp not in V[v]["sets"]:
                    continue
                out = dict(variant=V[v]["label"], base=V[base]["label"], input_set=F.INPUT_LABEL[inp])
                for k in F.LEADS:
                    row = {}
                    for m in METRICS:
                        a, b = seeds_of(v, m, k), seeds_of(base, m, k)
                        pairs = [x - y for x in a for y in b]                                   # every seed of the variant against every seed of the base
                        d, n = float(np.mean(a) - np.mean(b)), noise[m][lk(k)]
                        row[m] = dict(base=float(np.mean(b)), variant=float(np.mean(a)), difference=d, difference_min=float(min(pairs)), difference_max=float(max(pairs)),
                                      base_by_seed=b, variant_by_seed=a, seed_noise=n,
                                      larger_than_seed_noise=bool(abs(d) > 2 * n and (min(pairs) > 0 or max(pairs) < 0)) if len(seeds) > 1 else None,
                                      seed_average_forecast=dict(base=METRICS[m](V[base]["sets"][inp]["test"]["summary"][lk(k)]), variant=METRICS[m](V[v]["sets"][inp]["test"]["summary"][lk(k)])))
                    nb = np.array([[V[x]["sets"][inp]["test"]["basins"][g][lk(k)]["nse"] for g in ids] for x in (v, base)])
                    row["n_basins_improved"], row["n_basins"], row["median_of_basin_nse_differences"] = int((nb[0] > nb[1]).sum()), len(ids), float(np.median(nb[0] - nb[1]))
                    out[lk(k)] = row
                exp["against_base"][f"{v}_{inp}"] = out

    # ---- plain-language findings, one block per question ---------------------------------------------------------------------
    pc = lambda x: f"{100 * x:.0f}%"
    verdict = lambda r: "no second seed to judge it" if r["larger_than_seed_noise"] is None else "larger than the seed noise" if r["larger_than_seed_noise"] else "NOT larger than the seed noise"

    def say(name, v, inp, intro):
        c = E[name]["against_base"][f"{v}_{inp}"]
        return [f"{intro}, {F.INPUT_LABEL[inp]}, {days(k)}: median test NSE {c[lk(k)]['nse_median']['base']:.3f} -> {c[lk(k)]['nse_median']['variant']:.3f} "
                f"({c[lk(k)]['nse_median']['difference']:+.3f}; seed pairs {c[lk(k)]['nse_median']['difference_min']:+.3f} to {c[lk(k)]['nse_median']['difference_max']:+.3f}; "
                f"two trainings of the same model differ by {c[lk(k)]['nse_median']['seed_noise']:.3f} on average) - {verdict(c[lk(k)]['nse_median'])}; "
                f"better in {c[lk(k)]['n_basins_improved']} of {c[lk(k)]['n_basins']} basins." for k in F.LEADS]

    def say_onset(name, v, inp, intro):
        c = E[name]["against_base"][f"{v}_{inp}"]
        return [f"{intro}, {F.INPUT_LABEL[inp]}, {days(k)}: onsets warned {pc(c[lk(k)]['onset_pod']['base'])} -> {pc(c[lk(k)]['onset_pod']['variant'])} "
                f"(seed pairs {100 * c[lk(k)]['onset_pod']['difference_min']:+.0f} to {100 * c[lk(k)]['onset_pod']['difference_max']:+.0f} points; {verdict(c[lk(k)]['onset_pod'])}), "
                f"false warnings among onset warnings {pc(c[lk(k)]['onset_far']['base'])} -> {pc(c[lk(k)]['onset_far']['variant'])}, onset CSI {c[lk(k)]['onset_csi']['base']:.3f} -> "
                f"{c[lk(k)]['onset_csi']['variant']:.3f} ({verdict(c[lk(k)]['onset_csi'])})." for k in F.LEADS]

    A, Bx = "a_rain_source", "b_extra_inputs"
    findings = {
        "read_first_gains_1_day_ahead": [
            "Satellite rain, soil wetness and precipitable water of the issue day are UTC-day values: their day ends at 19:00 local standard time. The gauge-based rain of the issue day ends earlier: it matches "
            f"satellite rain of the UTC day before ({lagc['median_correlation_satellite_one_day_earlier']:.2f}) as well as of the same UTC day ({lagc['median_correlation']:.2f}; median correlations). The UTC-day "
            "inputs therefore already hold rain that the gauge table books on the next day (correlation with next-day gauge rain: "
            + "; ".join(f"{k_} {v_['correlation_with_daymet_rain_of_the_next_day']:.2f}" for k_, v_ in ahead.items()) + "). A gain 1 day ahead can come from this later time stamp alone: it does not show that "
            "satellite rain is more accurate than rain gauges, or that soil storage or moisture in the air adds skill (see notes)."],
        "rain_from_satellite_instead_of_gauges": sum([say(A, "imerg", inp, "Satellite rain instead of gauge-based rain") for inp in SETS], [])
                                                 + say(A, "imerg_nosun", "w", "Satellite rain and no Daymet sunlight, against gauge-based rain"),
        "satellite_rain_on_top_of_gauge_rain": sum([say(A, "gauge_imerg", inp, "Satellite rain added to gauge-based rain") for inp in SETS], []),
        "soil_wetness": sum([say(Bx, "soil", inp, "Soil wetness added") for inp in SETS], []),
        "precipitable_water": sum([say(Bx, "pw", inp, "Precipitable water added") for inp in SETS], []) + sum([say_onset(Bx, "pw", inp, "Precipitable water added") for inp in SETS], []),
        "soil_wetness_and_precipitable_water": sum([say(Bx, "soil_pw", inp, "Soil wetness and precipitable water added") for inp in SETS], []),
        "onset_warnings_other_variants": sum([say_onset(n_, v, inp, intro) for n_, v, intro in ((A, "imerg", "Satellite rain instead of gauge-based rain"), (A, "gauge_imerg", "Satellite rain added to gauge-based rain"),
                                                                                            (Bx, "soil", "Soil wetness added")) for inp in SETS], []),
    }
    n_clear = {name: [int(sum(bool(c[lk(k)]["nse_median"]["larger_than_seed_noise"]) for c in exp["against_base"].values() for k in F.LEADS)), 3 * len(exp["against_base"])] for name, exp in E.items()}
    n_onset = {n_: sum(E[n_]["variants"][E[n_]["base"]]["sets"]["w"]["test"]["summary"]["l1"]["onset"][c] for c in ("hits", "misses")) for n_ in E}
    n_cap = sum(r["best_update_near_the_limit"] for exp in E.values() for v in exp["variants"].values() for x in v["sets"].values() for r in x["per_seed"].values())

    false_alarm = {2: "1 in 3", 3: "1 in 10"}.get(len(seeds), "unknown")
    by_chance = {f"{n_} variants in the noise estimate": chance_rate(n_, len(seeds)) for n_ in sorted({sum(inp in x["sets"] for x in e["variants"].values()) for e in E.values() for inp in SETS})}
    n_comp = 3 * sum(len(e["against_base"]) for e in E.values())
    findings["soil_wetness_without_and_with_river_gauge"] = [
        f"Soil wetness added, {days(k)}: change of the median test NSE without river gauge {E[Bx]['against_base']['soil_w'][lk(k)]['nse_median']['difference']:+.3f} "
        f"({verdict(E[Bx]['against_base']['soil_w'][lk(k)]['nse_median'])}), with river gauge {E[Bx]['against_base']['soil_wq'][lk(k)]['nse_median']['difference']:+.3f} "
        f"({verdict(E[Bx]['against_base']['soil_wq'][lk(k)]['nse_median'])})." for k in F.LEADS]
    notes = [
        f"LSTM settings: {config_source}; the number of updates is capped at {R['config']['updates']} and the patience at {R['config']['patience']} checks to keep the script near 30 minutes on this laptop. "
        f"{n_cap} of {n_total} trainings had their best validation score within the last two checks before the cap, i.e. might have improved a little with more updates.",
        "Experiment (a) uses its own satellite-era split (training 2000-10-01..2008-09-30, validation 2008-10-01..2010-09-30, test 2010-10-01..2014-09-30): 8 training years instead of 20, 4 test years instead of 9, "
        "and high-flow thresholds from ITS training years. Its numbers are NOT comparable with the main split (experiment (b), 04_lstm_all.py); only the differences inside the experiment mean something.",
        "Equal data: inside an experiment all variants, the base included, read tables cropped to the same first day (data_start), so they are trained and scored on identical days "
        "(same first forecast day; coverage 1.0 in the validation and test years - asserted). The base of (b) therefore starts three months later than lstm_w / lstm_wq of 04_lstm_all.py and is not identical to them.",
        f"Seeds: {len(seeds)} per variant. 'difference' = mean over the seeds of the variant minus mean over the seeds of the base, for the median test NSE of the 12 basins (single trainings, not the seed average). "
        "difference_min / max = every seed of the variant against every seed of the base. seed_noise = how much two trainings of the SAME model differ on average (mean absolute difference between seeds, "
        "averaged over the variants of the experiment and input set). larger_than_seed_noise is true only when the difference is more than twice the seed noise AND all seed pairs agree in sign. "
        f"With {len(seeds)} seeds the sign rule alone would be met by chance in {false_alarm} comparisons, which is why the second condition is added. A simulation with random seed noise and no real "
        f"difference (seed_noise_rule) makes the full rule fire in {' / '.join(f'{100 * v:.0f}%' for v in by_chance.values())} of comparisons ({' / '.join(by_chance)}); {n_comp} comparisons of the median NSE "
        f"are made here without correction, so about {n_comp * float(np.nanmean(list(by_chance.values()))):.0f} 'true' are expected by chance alone: a single 'true' is weak evidence, a pattern over leads and input sets counts for more.",
        "n_basins_improved, the per-basin scores under 'test' and the saved prediction sets use the seed-average forecast (day-by-day mean of the seeds) of each variant."
        + (f" Single basins are less stable than the median of the {len(ids)}: between two trainings of the same model the test NSE of one basin 1 day ahead differs by {basin_noise['median']:.2f} in "
           f"the median and by up to {basin_noise['largest']:.2f} ({basin_noise['largest_where']}; seed_noise_single_basin). n_basins_improved counts every basin with a higher NSE, however small the "
           f"gain, so only counts near 0 or near {len(ids)} carry weight, and a gain can sit in a few basins (per-basin scores: experiments, variants, sets, test, basins)." if basin_noise else ""),
        "The test years chose nothing: settings come from the validation years of the main split, every stopping point from the validation years of the experiment's own split, and no variant was selected here.",
        "Satellite rain = GPM IMERG as served by NASA POWER, averaged over each basin. From 2001 it is the Final Run, which is corrected with monthly rain-gauge analyses and published about 3.5 months late; "
        "the run before 2001 is not stated by the source. The experiment therefore shows the best case for satellite rain, not what a real-time warning system would have (Early/Late runs, not tested).",
        "The IMERG day is the UTC day, which ends at 19:00 local standard time (18:00 for Leaf River): the value of day t is complete at the issue time (end of local day t), so nothing from after the issue "
        "time is used (flood.fit also checks that the network cannot read later days). The Daymet day is a different one: its rain of day d matches the UTC days d-1 and d about equally well and the UTC day d+1 "
        f"hardly at all (median correlations {lagc['median_correlation_satellite_one_day_earlier']:.2f}, {lagc['median_correlation']:.2f}, {lagc['median_correlation_satellite_one_day_later']:.2f}; "
        "results/satellite_data.json), as if it held the 24 hours up to the morning of day d. Satellite rain of issue day t therefore includes rain that fell later on day t (up to 19:00) and that Daymet books on "
        "day t+1. A gain from satellite rain can come from this later day boundary as well as from the rain amounts; this experiment cannot separate the two, and a loss with 'satellite instead of gauge' is the net "
        "of worse amounts and a later boundary.",
        "The same holds for the MERRA-2 inputs, which are UTC-day values too. What each input of issue day d already tells about the rain that Daymet books on day d+1 (median correlation over the basins, "
        "training years of (a); what_inputs_know_about_next_daymet_day): " + "; ".join(f"{k_} {v_['correlation_with_daymet_rain_of_the_next_day']:.2f}" for k_, v_ in ahead.items())
        + ". An input that scores higher here than Daymet rain itself carries rain information the base model does not have at issue time. A gain 1 day ahead from satellite rain, soil wetness or precipitable "
        "water is therefore at least partly a gain from a later observation time, not proof that the quantity itself (soil storage, moisture in the air) adds skill. If this reading of the Daymet day is right, "
        "part of what 04_lstm_all.py reports as the value of a 'perfect weather forecast' 1 day ahead is also rain that had already fallen at issue time.",
        f"Daymet does not measure sunlight; it estimates it from the daily temperature range and from whether rain was reported. In these basins its sunlight is on average {100 * (1 - R['daymet_sunlight_on_rain_days']['ratio_wet_to_dry_median']):.0f}% lower on days "
        "with at least 1 mm of gauge-based rain than on drier days of the same calendar month (median of 12 basins, training years of (a)). A model without gauge rain can read rain days from it, so the variant "
        "'imerg_nosun' removes it (past weather only). Temperature and humidity are still Daymet's, i.e. station-based: a fully gauge-free input set was not tested.",
        "Soil wetness (GWETTOP, GWETROOT) and precipitable water (TQV) are MERRA-2 reanalysis values of the one 0.5 x 0.625 degree cell that holds the basin centroid: a weather model that assimilates satellite and "
        "other observations, not a direct satellite retrieval, and published weeks after the fact. Its rain over land is corrected with rain-gauge analyses and drives the land model that computes the soil "
        "wetness (results/satellite_data.json, notes and documentation), so soil wetness is a modelled quantity and not independent of rain gauges; like the IMERG Final Run, a product computed after the "
        "fact can carry corrections that rest on observations made after the day it describes. Precipitable water is the quantity GNSS receivers measure; no GNSS data were used.",
        "Warning scores are pooled over the 12 basins; high flow = above the basin's 95th percentile of the training years of the split; onset = forecasts issued while the river was still at or below that level. "
        f"The test years of experiment (a) hold {n_onset[A]} onset days (1 day ahead, all basins) against {n_onset[Bx]} in experiment (b), so the warning scores of (a) are the noisier ones.",
    ] + (["QUICK smoke test: a few updates, one seed - numbers are not results."] if R["quick_mode"] else [])
    R.update(comparisons_larger_than_seed_noise_nse=n_clear, what_inputs_know_about_next_daymet_day=ahead, seed_noise_single_basin=basin_noise,
             seed_noise_rule=dict(rule="|difference of seed means| > 2 x seed_noise AND every seed of the variant on the same side of every seed of the base",
                                  share_of_comparisons_flagged_when_there_is_no_real_difference=by_chance,
                                  how="simulation: 200000 draws of normally distributed seed noise, the same number of seeds and variants as here"),
             findings=findings, notes=notes)
    R = {k: R[k] for k in [k for k in R if k not in TAIL] + [k for k in TAIL if k in R]}        # costs last, as in the other result files
    (F.RES_OUT / "satellite_inputs.json").write_text(json.dumps(R, indent=1))
    return R


def figure(R):
    """Rows: the two experiments; columns: without / with river gauge. Change of the median test NSE against the base, with seed spread and noise band."""
    W, H = 6.7, 7.9                                                           # inches; everything below is placed in inches
    fig = plt.figure(figsize=(W, H))
    left, gap, pw, ph = 0.78, 0.16, 2.83, 2.05
    for (name, exp), top, leg_cols in zip(R["experiments"].items(), (H - 0.62, 3.72), (1, 3)):
        V, yrs = exp["variants"], f"{exp['split']['test'][0][:4]}-{exp['split']['test'][1][:4]}"
        shown = [v for v in V if v != exp["base"]]
        row = [fig.add_axes([(left + i * (pw + gap)) / W, (top - ph) / H, pw / W, ph / H]) for i in range(len(SETS))]
        lim = 0.0
        for ax, inp in zip(row, SETS):
            here = [v for v in shown if f"{v}_{inp}" in exp["against_base"]]
            off = np.linspace(-0.24, 0.24, len(here)) if len(here) > 1 else [0.0]
            for k in F.LEADS:
                n = exp["seed_noise"][inp]["nse_median"][lk(k)]
                if np.isfinite(n):
                    ax.fill_between([k - 0.42, k + 0.42], -2 * n, 2 * n, color=GRID, lw=0, zorder=1)
                    lim = max(lim, 2 * n)
                for v, o in zip(here, off):
                    c = exp["against_base"][f"{v}_{inp}"][lk(k)]["nse_median"]
                    ax.plot([k + o, k + o], [c["difference_min"], c["difference_max"]], color=STYLE[v][0], lw=2.2, solid_capstyle="butt", zorder=3)
                    ax.plot(k + o, c["difference"], ls="none", marker=STYLE[v][1], ms=7.5 if STYLE[v][1] == "o" else 6.5, color=STYLE[v][0],
                            markerfacecolor=STYLE[v][0] if c["larger_than_seed_noise"] else SURFACE, markeredgecolor=STYLE[v][0], markeredgewidth=1.6, zorder=4)
                    lim = max(lim, abs(c["difference_min"]), abs(c["difference_max"]))
            ax.axhline(0, color=MUTED, lw=1, zorder=2)
            base = [exp["against_base"][f"{here[0]}_{inp}"][lk(k)]["nse_median"]["base"] for k in F.LEADS]
            ax.set_xlim(0.5, 3.5); ax.set_xticks(F.LEADS, [f"{days(k)}\nbase NSE {b:.2f}" for k, b in zip(F.LEADS, base)])
            ax.tick_params(labelsize=8, length=0); ax.grid(axis="x", visible=False)
            ax.set_title(f"{exp['title']}: {'past weather + river gauge' if inp == 'wq' else 'past weather only'}", fontsize=8.8)
        for ax in row:
            ax.set_ylim(-1.15 * lim, 1.15 * lim)
        row[1].tick_params(labelleft=False)
        row[0].set_ylabel(f"change of the median NSE\n(test years {yrs})", fontsize=8.5)
        h = [Line2D([], [], ls="none", marker=STYLE[v][1], ms=6.5, color=STYLE[v][0], label=V[v]["label"]) for v in shown]
        fig.legend(handles=h, loc="upper left", bbox_to_anchor=(left / W, (top - ph - 0.43) / H), ncol=leg_cols, frameon=False, fontsize=8, columnspacing=1.1, handletextpad=.2,
                   borderaxespad=0, borderpad=0, title=f"Base = {V[exp['base']]['label']}. Change of skill with:", title_fontsize=8, alignment="left")
    n = len(R["seeds"])
    note = [f"Marker: mean of {n} training{'s' * (n > 1)} (random seeds) minus the mean of the base. Bar: every seed against every seed of the base.",
            "Grey band: twice the typical difference between two trainings of the same model. Filled marker: the change is outside the band",
            "and all seed pairs agree in sign; hollow: it cannot be told from chance. The rows use different years: do not compare their levels."]
    fig.text(0.012, 0.06 / H, "\n".join(note), ha="left", va="bottom", fontsize=7.5, color=INK2, linespacing=1.45)
    fig.suptitle("Do satellite rain, soil wetness and precipitable water change the forecast skill?", fontsize=10.2, fontweight="bold", color=INK, x=0.012, y=1 - 0.1 / H, ha="left", va="top")
    fig.savefig(F.FIG_OUT / "satellite_inputs.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    saved = lambda: json.loads((F.RES_OUT / "satellite_inputs.json").read_text())
    R = saved() if "--figure" in sys.argv else analyse(saved()) if "--rescore" in sys.argv else run()
    R = json.loads(json.dumps(R))                                               # the figure reads exactly what the JSON holds
    figure(R)
    print(f"\n{R['n_trainings']} trainings in {R['training_seconds']:.0f} s; comparisons larger than the seed noise (median NSE): {R['comparisons_larger_than_seed_noise_nse']}")
    for q, lines in R["findings"].items():
        print(f"\n{q}:", *lines, sep="\n  ")
    print("\nnotes:", *R["notes"], sep="\n  ")
