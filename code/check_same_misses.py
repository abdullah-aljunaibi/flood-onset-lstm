"""Do the models miss the same onsets? A check on the stored forecasts, day by day.

The report's main finding is that, among the models tried, the limit is the information in the inputs, not the network. Median scores
and pooled warning rates cannot show whether the models also fail on the SAME events. This script counts it, 1, 2 and 3 days ahead:
 1. Four kinds of model read past weather + river gauge: ridge regression (base_linreg_wq), random forest (base_rf_wq), the LSTM
    (lstm_wq) and the Transformer (cmp_transformer). How many onsets are missed by all four, and how many would a warning from any
    one of them cover?
 2. How many of the onsets that all four miss are warned by the LSTM given more information: rain up to the evening of the issue
    day (lstm_wqn), or the perfect weather forecast (lstm_wqf), an upper bound that cannot be had in practice?
 3. A yardstick: how far do two seeds of the same LSTM (lstm_wq_s<seed>) miss the same onsets?
 4. Does the answer hang on the warning level? The same counts at the level chosen for each model on the validation years (the rule of
    06_flood_analysis.py), and the share expected if the models' misses were unrelated (over all basins, and inside each basin).
Definitions exactly as in flood.flood_scores: test years; the high-flow level of flood.thresholds(); a warning is a forecast above the
level itself (warning level 1.0); an onset is a high-flow day whose forecast was issued while the river was still at or below the level.
Reads the prediction sets named above from results/preds, the flow record (flood.load_data) for the high-flow levels, and from
results/lstm_all.json, rain_timing.json, flood_analysis.json and model_comparison.json the onset counts and warning levels those scripts
stored (for its checks), the grid of warning levels (flood_analysis.json) and the names of the basin properties (for one note). Trains
nothing and runs in seconds.
Writes results/same_misses.json (no figure). With FLOOD_QUICK=1 it reads and writes under results/quick only; if a smoke-test input
is missing there it prints one line and ends without an error.
"""
import json
import os
import sys
from pathlib import Path
import time

import numpy as np

import flood as F
from common import ROOT

t_start = time.time()
SCRIPT = "check_same_misses.py"
KINDS = {"ridge regression": "base_linreg_wq", "random forest": "base_rf_wq", "LSTM": "lstm_wq", "Transformer": "cmp_transformer"}   # all read past weather + river gauge
MORE = {"warned_with_rain_to_evening": "lstm_wqn", "warned_with_perfect_forecast": "lstm_wqf"}     # the LSTM given more information
LSTM = KINDS["LSTM"]                                        # the other kinds are compared with its misses; its single seeds are the yardstick
INPUTS = F.INPUT_LABEL["wq"]                                # 'past weather + river gauge': what all four kinds are given
TRIGGER = 1.0                                               # a warning = a forecast above the high-flow level itself
COUNTS = ("hits", "misses", "false_alarms")
# where the scripts that made the forecasts stored the same onset counts: {prediction set: [(results file, key of one count), ...]}
STORED = {"lstm_wq": [("lstm_all", "decomposition.levels.wq.{lead}.onset_{count}"), ("flood_analysis", "skill.lstm_wq.plain.test.summary.{lead}.onset.{count}")],
          "lstm_wqn": [("rain_timing", "models.wqn.test.summary.{lead}.onset.{count}")],
          "lstm_wqf": [("lstm_all", "decomposition.levels.wqf.{lead}.onset_{count}"), ("flood_analysis", "skill.lstm_wqf.plain.test.summary.{lead}.onset.{count}")],
          "base_rf_wq": [("flood_analysis", "skill.base_rf_wq.plain.test.summary.{lead}.onset.{count}")],
          "base_linreg_wq": [("flood_analysis", "skill.base_linreg_wq.plain.test.summary.{lead}.onset.{count}")],
          "cmp_transformer": [("model_comparison", "models.transformer.seed_average.test.summary.{lead}.onset.{count}")]}
lk = lambda k: f"l{k}"
rel = lambda p: Path(os.path.relpath(p, ROOT)).as_posix()          # forward slashes on every system (a Windows rerun of 30 October 2026 wrote backslashes)
div = lambda x, y: x / y if y else float("nan")
share = lambda x, of: dict(count=int(x.sum()), share=div(int(x.sum()), int(of.sum())))            # how many of `of` are in x
p1 = lambda x: f"{100 * x:.1f}%" if x == x else "no value"
days = lambda k: f"{k} day{'s' * (k > 1)} ahead"
word = lambda n: {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))
listed = lambda v: (lambda x: ", ".join(x[:-1]) + " and " + x[-1])([str(i) for i in v])              # '611, 951 and 1116'


def need(paths):
    """Stop if an input file is missing: a smoke test says so in one line and ends without an error, a real run ends with one."""
    lack = [rel(p) for p in paths if not p.exists()]
    if lack and F.QUICK:
        print(f"{SCRIPT}: smoke-test input not found ({', '.join(lack)}) - nothing written.")
        sys.exit(0)
    if lack:
        sys.exit(f"{SCRIPT}: not found: {', '.join(lack)} - run the scripts that write them first (run_all.sh gives the order).")


def must(ok, message):
    """A built-in check: the script stops here if it fails (not an assert statement, which python -O would skip)."""
    if not ok:
        raise AssertionError(message)


def lookup(J, key):
    for part in key.split("."):
        J = J[part]
    return J


# ---- inputs: the stored forecasts ------------------------------------------------------------------------------------------
sets = list(KINDS.values()) + list(MORE.values())
stems = sorted({stem for v in STORED.values() for stem, _ in v})
need([F.PRED / f"{n}.npz" for n in sets] + [F.RES_OUT / f"{s}.json" for s in stems])
META = {n: F.preds_meta(n) for n in sets}
FAMILY = {n: META[n].get("model", "lstm") for n in sets}    # the engine's name of each kind (10_rain_timing.py stores none: its sets are LSTMs)
must(set(FAMILY.values()) <= set(F.MODEL_COL) and len({FAMILY[n] for n in KINDS.values()}) == len(KINDS), "the kinds must be different model families of flood.MODEL_COL")
SEEDS = list(META[LSTM].get("seeds", []))
seed_sets = [f"{LSTM}_s{s}" for s in SEEDS]
need([F.PRED / f"{n}.npz" for n in seed_sets])
D = F.load_data()
thr, ids, (a, b) = F.thresholds(D), sorted(D), F.SPLIT["test"]
P = {n: F.load_preds(n) for n in sets + seed_sets}
for n, p in P.items():                                      # the same rivers, days and observed flows in every set, or a day-by-day comparison means nothing
    must(sorted(p) == ids and all(p[g].index.equals(D[g].index) and np.array_equal(p[g].obs.values, D[g].q_mm.values, equal_nan=True) for g in ids),
         f"{n}: its basins, days or observed flows are not those of the data")

# ---- the same basin-days for every set ---------------------------------------------------------------------------------------
test_days = len(D[ids[0]].loc[a:b])
coverage = {n: min(float(np.mean([p[g].loc[a:b, lk(k)].notna().mean() for g in ids])) for k in F.LEADS) for n, p in P.items()}
record = {k: sum(int(((D[g].q_mm.shift(k) <= thr[g]) & (D[g].q_mm > thr[g])).loc[a:b].sum()) for g in ids) for k in F.LEADS}       # onsets in the flow record itself
days_dropped = {}
for k in F.LEADS:
    days_dropped[k] = 0
    for g in ids:
        have = np.logical_and.reduce([p[g].loc[a:b, lk(k)].notna().values for p in P.values()])
        days_dropped[k] += int((~have).sum())
        if not have.all():                                  # a day without a forecast in one set is dropped from every set
            for p in P.values():
                col = p[g][lk(k)].copy()
                col.loc[a:b] = col.loc[a:b].where(have)
                p[g][lk(k)] = col


# ---- counting ------------------------------------------------------------------------------------------------------------
def engine(n, lead, m=TRIGGER):
    """The onset table of one prediction set from the engine's own function, pooled over the basins as flood.evaluate pools it."""
    rows = [F.flood_scores(P[n][g], thr[g], lead, "test", m)["onset"] for g in ids]
    return F.rates({c: int(sum(r[c] for r in rows)) for c in COUNTS + ("correct_neg",)})


def tables(lead, level):
    """Day by day, the basins pooled: was the day high, which sets warned, and which basin it is - on the days flood.flood_scores scores for onsets.

    Same expressions as flood.flood_scores: the test days whose forecast was issued while the river was at or below the level (`lead`
    days before) and that have a forecast; a day is high when the flow is above the level, warned when the forecast is above the warning
    level. level: {prediction set: its warning level as a multiple of the high-flow level}; a set that is not named warns at TRIGGER."""
    c, high, basin, warned = lk(lead), [], [], {n: [] for n in P}
    for i, g in enumerate(ids):
        t, p0 = thr[g], P[LSTM][g]                          # the observed flow is the same in every set (checked above)
        keep = (p0.obs.shift(lead) <= t).loc[a:b].values & (p0.obs.loc[a:b].notna() & p0.loc[a:b, c].notna()).values
        high.append((p0.obs.loc[a:b] > t).values[keep])
        basin.append(np.full(int(keep.sum()), i))
        for n, p in P.items():
            warned[n].append((p[g].loc[a:b, c] > level.get(n, TRIGGER) * t).values[keep])
    return np.concatenate(high), {n: np.concatenate(v) for n, v in warned.items()}, np.concatenate(basin)


def unrelated_in_basins(o, basin, first, others):
    """Of the onsets in `first`: the share expected in every set of `others` too if, inside each basin, the models' misses were unrelated.

    The basins differ in how many of their onsets every model misses; this takes each model's share of missed onsets basin by basin."""
    per = lambda x: np.bincount(basin[x], minlength=len(ids)).astype(float)                          # how many of x in each basin
    n, e = per(o), per(first)
    for m in others:
        e = e * np.divide(per(m), n, out=np.zeros_like(n), where=n > 0)
    return div(float(e.sum()), int(first.sum()))


def seed_overlap(o, w):
    """Of the onsets one seed of the LSTM misses, the share another seed also misses: every ordered pair of seeds."""
    pairs = []
    for s in seed_sets:
        for s2 in seed_sets:
            if s2 != s:
                m, m2 = o & ~w[s], o & ~w[s2]
                pairs.append(dict(prediction_set=s, other=s2, misses=int(m.sum()), **share(m & m2, m), share_if_independent=div(int(m2.sum()), int(o.sum()))))
    v, e = [r["share"] for r in pairs if r["share"] == r["share"]], [r["share_if_independent"] for r in pairs if r["share_if_independent"] == r["share_if_independent"]]
    return dict(min=min(v), max=max(v), share_if_independent=dict(min=min(e), max=max(e)), pairs=pairs) if v and e else None


kinds = list(KINDS.values())


def count(k, level=None, seeds=True):
    """Every count of one lead. level: {prediction set: its warning level as a multiple of the high-flow level}; None = TRIGGER for every set."""
    level = level or {}
    o, w, basin = tables(k, level)
    names = list(P) if seeds else sets
    tab = {n: engine(n, k, level.get(n, TRIGGER)) for n in names}
    for n in names:                                         # every count is made twice; all sets hold the same days, so the same onsets
        mine = dict(hits=int((o & w[n]).sum()), misses=int((o & ~w[n]).sum()), false_alarms=int((~o & w[n]).sum()), correct_neg=int((~o & ~w[n]).sum()))
        must(mine == {c: tab[n][c] for c in mine}, f"{n}, {days(k)}: the day-by-day table and flood.flood_scores do not give the same counts")
    any_kind, all_kinds = np.logical_or.reduce([w[n] for n in kinds]), np.logical_and.reduce([w[n] for n in kinds])
    missed_all, lstm_miss, false_any = o & ~any_kind, o & ~w[LSTM], int((~o & any_kind).sum())
    miss = {n: o & ~w[n] for n in kinds}                    # the onsets each kind misses
    miss_share = {n: div(tab[n]["misses"], int(o.sum())) for n in kinds}                           # each kind's own share of missed onsets
    return dict(
        n_onsets=int(o.sum()), onsets_dropped=record[k] - int(o.sum()), basin_days_scored=len(o),
        models={n: {c: tab[n][c] for c in COUNTS + ("pod", "far")} for n in sets},
        missed_by_all_kinds=dict(share(missed_all, o), share_if_independent=float(np.prod([miss_share[n] for n in kinds])),
                                 share_if_independent_by_basin=unrelated_in_basins(o, basin, o, [miss[n] for n in kinds])),
        warned_by_any_kind=dict(share(o & any_kind, o), false_alarms=false_any, far=div(false_any, int((o & any_kind).sum()) + false_any)),
        warned_by_all_kinds=share(o & all_kinds, o),
        of_missed_by_all={key: share(missed_all & w[n], missed_all) for key, n in MORE.items()},
        lstm_misses=dict(count=int(lstm_miss.sum()), warned_by_another_kind=share(lstm_miss & any_kind, lstm_miss)),
        also_missed_of_lstm_misses={n: dict(share(lstm_miss & miss[n], lstm_miss), share_if_independent=miss_share[n],
                                            share_if_independent_by_basin=unrelated_in_basins(o, basin, lstm_miss, [miss[n]])) for n in kinds if n != LSTM},
        seed_to_seed=seed_overlap(o, w) if seeds else None)


out = {lk(k): count(k) for k in F.LEADS}
onsets_dropped = {k: out[lk(k)]["onsets_dropped"] for k in F.LEADS}

# ---- checks against the counts the other scripts stored -------------------------------------------------------------------------
J = {s: json.loads((F.RES_OUT / f"{s}.json").read_text()) for s in stems}
stored = []
for n, sources in STORED.items():
    for stem, key in sources:
        for k in F.LEADS:
            there = {c: int(lookup(J[stem], key.format(lead=lk(k), count=c))) for c in COUNTS}
            here = {c: out[lk(k)]["models"][n][c] for c in COUNTS}
            exact = not F.QUICK and days_dropped[k] == 0    # a smoke test compares but does not stop: its files may come from different smoke runs
            stored.append(dict(prediction_set=n, lead=k, file=rel(F.RES_OUT / f"{stem}.json"), key=key.format(lead=lk(k), count="<count>"), stored=there, this_script=here,
                               equal=there == here, asserted=exact))
            if exact:
                must(there == here, f"{n}, {days(k)}: {here} here, but {there} in {stored[-1]['file']} ({stored[-1]['key']})")
            elif not F.QUICK and onsets_dropped[k] == 0:    # days were dropped but no onset: the number of onsets must still be the stored one
                must(there["hits"] + there["misses"] == out[lk(k)]["n_onsets"], f"{n}, {days(k)}: {out[lk(k)]['n_onsets']} onsets here, not the number in {stored[-1]['file']}")
n_equal = sum(r["equal"] for r in stored)

# ---- the same counts at the warning level chosen for each model on the validation years ---------------------------------------
va, vb = F.SPLIT["val"]
GRID = [float(m) for m in J["flood_analysis"].get("warning_levels", [])]                           # the levels 06_flood_analysis.py tries: multiples of the high-flow level
SHOWN = ("n_onsets", "models", "missed_by_all_kinds", "warned_by_any_kind", "warned_by_all_kinds", "of_missed_by_all", "lstm_misses")


def chosen_level(n, lead):
    """The rule of 06_flood_analysis.py ('tuned'): the multiple of the high-flow level with the best CSI (hits / (hits + misses + false warnings))
    of all high-flow days in the validation years, basins pooled; a tie goes to the multiple closest to 1. None if a validation day has no forecast."""
    high = np.concatenate([(P[n][g].obs.loc[va:vb] > thr[g]).values for g in ids])
    f = np.concatenate([P[n][g].loc[va:vb, lk(lead)].values for g in ids])
    t = np.concatenate([np.full(len(P[n][g].loc[va:vb]), thr[g]) for g in ids])
    if not GRID or np.isnan(f).any():
        return None
    csi = []
    for m in GRID:
        warn = f > m * t
        h, mi, fa = int((high & warn).sum()), int((high & ~warn).sum()), int((~high & warn).sum())
        csi.append(h / (h + mi + fa) if h + mi + fa else -1.0)
    best = np.flatnonzero(np.array(csi) >= max(csi) - 1e-12)
    return GRID[int(best[np.argmin(np.abs(np.array(GRID)[best] - 1.0))])]


levels, level_rows = {n: {k: chosen_level(n, k) for k in F.LEADS} for n in sets}, []
for n in sets:                                              # the rule must give the levels 06_flood_analysis.py stored for the sets it scored
    tuned = J["flood_analysis"].get("skill", {}).get(n, {}).get("tuned")
    for k in F.LEADS if tuned else ():
        there, here = float(tuned[lk(k)]["trigger"]), levels[n][k]
        level_rows.append(dict(prediction_set=n, lead=k, stored=there, this_script=here, equal=there == here, asserted=not F.QUICK))
        if not F.QUICK:
            must(there == here, f"{n}, {days(k)}: " + ("no warning level could be chosen here (a validation day has no forecast)" if here is None else f"warning level x{here} chosen here")
                 + f", but x{there} is stored in {rel(F.RES_OUT / 'flood_analysis.json')} (skill.{n}.tuned.{lk(k)}.trigger)")
n_levels = sum(r["equal"] for r in level_rows)
have_levels = all(v is not None for d in levels.values() for v in d.values())
at_val = {lk(k): {key: v for key, v in count(k, {n: levels[n][k] for n in sets}, seeds=False).items() if key in SHOWN} for k in F.LEADS} if have_levels else None

checks = dict(same_basin_days=dict(basins=len(ids), test_days_per_basin=test_days, share_of_test_basin_days_with_a_forecast=coverage, every_set_has_every_day=not any(days_dropped.values()),
                                   by_lead={lk(k): dict(basin_days_dropped=days_dropped[k], onsets_in_the_flow_record=record[k], onsets_scored=out[lk(k)]["n_onsets"],
                                                        onsets_dropped=onsets_dropped[k]) for k in F.LEADS}),
              counted_twice="every count of every prediction set and lead" + (", at the high-flow level itself and at the levels chosen on the validation years," if at_val else "")
                            + " was made by flood.flood_scores and again from the day-by-day tables; the two agree (the script stops if they do not)",
              stored_counts=stored, stored_counts_equal=f"{n_equal} of {len(stored)}", all_stored_counts_equal=n_equal == len(stored),
              levels_chosen_on_validation=dict(stored_in=f"{rel(F.RES_OUT / 'flood_analysis.json')}: skill.<prediction set>.tuned.l<lead>.trigger", rows=level_rows,
                                               stored_levels_equal=f"{n_levels} of {len(level_rows)}", all_stored_levels_equal=n_levels == len(level_rows),
                                               no_stored_level=[n for n in sets if n not in {r["prediction_set"] for r in level_rows}]))

# ---- notes and headline: every number in them is read from the tables above -------------------------------------------------
L, NK = out["l1"], word(len(KINDS))
name_of = {v: ("the " if n[0].isupper() else "") + n for n, v in KINDS.items()}                    # 'the LSTM', 'ridge regression'
seeds_of = lambda n: len(META[n].get("seeds", []))
length = lambda n: f"sequences of {META[n]['config']['warmup'] + META[n]['config']['block']} days and at most {META[n]['config']['updates']:,} updates"
static = J["lstm_all"]["static_inputs"]                    # the basin properties a network for all basins reads beside the daily inputs
like_lstm = J["model_comparison"]["static_inputs"] == static and META[KINDS["Transformer"]].get("inputs") == META[LSTM].get("inputs")
others = [n for n in kinds if n != LSTM]
also = lambda r, n: f"{name_of[n]} also misses {p1(r['also_missed_of_lstm_misses'][n]['share'])} ({p1(r['also_missed_of_lstm_misses'][n]['share_if_independent'])} if independent)"
more = lambda r, key: r["of_missed_by_all"][key]
warnings = lambda r, n: r["models"][n]["hits"] + r["models"][n]["false_alarms"]
headline = (f"{days(1).capitalize()}, all {NK} kinds of model given {INPUTS} ({', '.join(KINDS)}) miss the same {L['missed_by_all_kinds']['count']} of the "
            f"{L['n_onsets']} onsets ({p1(L['missed_by_all_kinds']['share'])}); of those, the LSTM given rain up to the evening of the issue day warns of "
            f"{p1(more(L, 'warned_with_rain_to_evening')['share'])}, and the LSTM given the perfect weather forecast, an upper bound that cannot be had in practice, of "
            f"{p1(more(L, 'warned_with_perfect_forecast')['share'])}. (A warning is a forecast above the high-flow level itself; the Transformer is the "
            f"{word(seeds_of(KINDS['Transformer']))}-seed model of the cost comparison.)")
notes = [f"Definitions, exactly as in flood.flood_scores: test years ({a} to {b}), the {len(ids)} basins pooled. High flow: above the basin's high-flow level (flood.thresholds: the level "
         f"exceeded on {round(100 * (1 - F.HIGH_Q))}% of the days of the training years). Warning: a forecast above the level itself (warning level {TRIGGER}) in every count except those under "
         "at_levels_chosen_on_validation. Onset: a "
         "high-flow day whose forecast was issued while the river was still at or below the level, 1, 2 or 3 days before. An onset is missed when the forecast for that day stayed at or below the "
         "level; a false warning is a warning, issued while the river was at or below the level, for a day that stayed at or below it.",
         f"The {NK} kinds of model are given the same information, {INPUTS}. Ridge regression ({KINDS['ridge regression']}) and random forest ({KINDS['random forest']}) read it as "
         f"hand-made features (features.py) and are fitted once per basin and lead by 02_baselines.py. The LSTM ({LSTM}) reads it day by day; it is one model for all basins, so it also "
         f"reads {len(static)} basin properties, and its forecast is the day-by-day average of {word(seeds_of(LSTM))} seeds (04_lstm_all.py). The Transformer ({KINDS['Transformer']}) is the "
         f"{word(seeds_of(KINDS['Transformer']))}-seed model of the cost comparison, trained in its own experiment{' with the same inputs as the LSTM' if like_lstm else ''} "
         f"(09_model_comparison.py: {length(KINDS['Transformer'])}; the LSTM here: {length(LSTM)}).",
         f"The LSTM given more information is a training of its own with one more kind of input: {MORE['warned_with_rain_to_evening']} (10_rain_timing.py) also reads rain up to the evening of the "
         f"issue day; {MORE['warned_with_perfect_forecast']} (04_lstm_all.py) also reads the weather stamped on the three days being forecast, the perfect weather forecast: an upper bound that "
         "cannot be had in practice.",
         f"2 and 3 days ahead: {' and '.join(p1(out[lk(k)]['missed_by_all_kinds']['share']) for k in (2, 3))} of the onsets are missed by all {NK} kinds "
         f"({' and '.join(str(out[lk(k)]['missed_by_all_kinds']['count']) + ' of ' + str(out[lk(k)]['n_onsets']) for k in (2, 3))}); of those, the LSTM given rain up to the evening of the issue "
         f"day warns of {' and '.join(p1(more(out[lk(k)], 'warned_with_rain_to_evening')['share']) for k in (2, 3))}, and the LSTM given the perfect weather forecast, an upper bound that cannot "
         f"be had in practice, of {' and '.join(p1(more(out[lk(k)], 'warned_with_perfect_forecast')['share']) for k in (2, 3))}.",
         f"Would another kind of model catch what the LSTM misses? {days(1).capitalize()}, a warning from any one of the {NK} kinds (at least one forecast above the level) covers "
         f"{p1(L['warned_by_any_kind']['share'])} of the onsets, against {p1(L['models'][LSTM]['pod'])} for the LSTM alone, and {p1(L['warned_by_any_kind']['far'])} of those warnings are false "
         f"(LSTM alone: {p1(L['models'][LSTM]['far'])}). Of the {L['lstm_misses']['count']} onsets the LSTM misses, the other {word(len(others))} kinds together warn of "
         f"{L['lstm_misses']['warned_by_another_kind']['count']} ({p1(L['lstm_misses']['warned_by_another_kind']['share'])}).",
         f"The LSTM given more information also gives more warnings. {days(1).capitalize()}, counting only forecasts issued while the river was at or below the level: {warnings(L, LSTM)} "
         f"warnings from the LSTM, {p1(L['models'][LSTM]['far'])} of them false; {warnings(L, MORE['warned_with_rain_to_evening'])} with rain up to the evening of the issue day, "
         f"{p1(L['models'][MORE['warned_with_rain_to_evening']]['far'])} false; {warnings(L, MORE['warned_with_perfect_forecast'])} with the perfect weather forecast, an upper bound that "
         f"cannot be had in practice, {p1(L['models'][MORE['warned_with_perfect_forecast']]['far'])} false.",
         f"Each of the {NK} kinds misses most of the onsets ({p1(min(div(L['models'][n]['misses'], L['n_onsets']) for n in kinds))} to "
         f"{p1(max(div(L['models'][n]['misses'], L['n_onsets']) for n in kinds))} {days(1)}), so pairwise agreement is partly the base rate: a model that seldom warns misses most of what any "
         "other model misses. That is why "
         "share_if_independent is stored beside every such share: the share that would be seen if the second model's misses had nothing to do with the first one's, which is the second model's own "
         f"share of missed onsets. {days(1).capitalize()}, of the {L['lstm_misses']['count']} onsets the LSTM misses, {', '.join(also(L, n) for n in others)}. For 'missed by all {NK} kinds', "
         f"share_if_independent is the product of the {NK} single shares: {p1(L['missed_by_all_kinds']['share_if_independent'])} would be missed by all if their misses were unrelated, "
         f"{p1(L['missed_by_all_kinds']['share'])} are. Part of that difference is that some basins are hard for all {NK} kinds: with the misses unrelated inside each basin only, "
         f"{p1(L['missed_by_all_kinds']['share_if_independent_by_basin'])} would be missed by all (share_if_independent_by_basin; 2 and 3 days ahead "
         f"{' and '.join(p1(out[lk(k)]['missed_by_all_kinds']['share_if_independent_by_basin']) for k in (2, 3))}, against "
         f"{' and '.join(p1(out[lk(k)]['missed_by_all_kinds']['share']) for k in (2, 3))} missed). The share missed by all also depends on how many kinds are counted: it can only fall "
         f"when a kind is added. The LSTM alone misses {p1(div(L['models'][LSTM]['misses'], L['n_onsets']))}, so no more than that can be missed by all {NK}. "
         "Do not quote a pairwise share without its share_if_independent."]
sd = L["seed_to_seed"]
if sd:
    v = [L["also_missed_of_lstm_misses"][n] for n in others]
    (k_lo, k_hi), (e_lo, e_hi) = ((min(x[c] for x in v), max(x[c] for x in v)) for c in ("share", "share_if_independent"))
    place = ("inside" if sd["min"] <= k_lo and k_hi <= sd["max"] else "above" if k_lo > sd["max"] else "inside or above" if k_lo >= sd["min"] else "below" if k_hi < sd["min"]
             else "partly below")
    notes.append(f"Yardstick: two seeds of the same LSTM (the single trainings {', '.join(seed_sets)}). {days(1).capitalize()}, of the onsets one seed misses, another seed also misses "
                 f"{p1(sd['min'])} to {p1(sd['max'])} ({p1(sd['share_if_independent']['min'])} to {p1(sd['share_if_independent']['max'])} if independent; every ordered pair of seeds). The shares of "
                 f"the other kinds of model, {p1(k_lo)} to {p1(k_hi)} ({p1(e_lo)} to {p1(e_hi)} if independent), lie {place} that range. A rough yardstick only: the models differ in how often "
                 f"they warn at all (share_if_independent), and the kinds are compared with the misses of the {word(seeds_of(LSTM))}-seed average, the seeds with each other.")
else:
    notes.append(f"Yardstick: fewer than two single seeds of {LSTM} were found, so seed_to_seed is empty.")
if not any(days_dropped.values()):
    notes.append(f"Same basin-days: every prediction set holds the same basins, days and observed flows as the data and has a forecast on every day of the test years ({test_days:,} days x "
                 f"{len(ids)} basins at every lead; the script checks both). All sets are therefore scored on exactly the same basin-days and no onset is dropped: "
                 f"{listed(out[lk(k)]['n_onsets'] for k in F.LEADS)} onsets 1, 2 and 3 days ahead, the same in every set.")
else:
    notes.append("Same basin-days: not every prediction set has a forecast on every day of the test years, so all sets were restricted to the basin-days on which every set has one "
                 f"({listed(days_dropped[k] for k in F.LEADS)} basin-days dropped 1, 2 and 3 days ahead). That drops {listed(onsets_dropped[k] for k in F.LEADS)} of the "
                 f"{listed(record[k] for k in F.LEADS)} onsets in the flow record. At a lead where days were dropped the counts here need not equal the stored ones, and the script does not "
                 "stop if they differ (checks.stored_counts: asserted false); it still stops if the number of onsets differs although no onset was dropped.")
where = {n: " and ".join(f"{rel(F.RES_OUT / (stem + '.json'))} ({key.split('.{lead}')[0]})" for stem, key in v) for n, v in STORED.items()}
notes.append("Checks built into the script; it stops if one fails. (a) Every count is made twice, by the engine's own function flood.flood_scores and from the day-by-day tables that give "
             "the overlaps, and the two must agree. (b) The hits, misses and false warnings of every prediction set at every lead must equal the counts stored by the scripts that made and "
             f"scored the forecasts: {'; '.join(f'{n} in {w}' for n, w in where.items())}. {n_equal} of the {len(stored)} stored tables are equal. (c) The warning levels chosen "
             f"here on the validation years must equal the levels 06_flood_analysis.py stored for the prediction sets it scored, {checks['levels_chosen_on_validation']['stored_in']}: "
             f"{n_levels} of {len(level_rows)} are equal."
             + (" In a smoke test (b) and (c) are compared and reported, and the script does not stop." if F.QUICK else ""))
if at_val:                                                  # the same counts at the level chosen for each model on the validation years
    gap = max(abs(at_val[lk(k)]["missed_by_all_kinds"]["share"] - out[lk(k)]["missed_by_all_kinds"]["share"]) for k in F.LEADS)
    both = lambda key: listed(p1(at_val[lk(k)]["of_missed_by_all"][key]["share"]) for k in F.LEADS)
    no_level = checks["levels_chosen_on_validation"]["no_stored_level"]
    notes.insert(4, "Does the count hang on the warning level? The same count with each model warning at the level chosen for it on the validation years (the rule of 06_flood_analysis.py: "
                    "for each prediction set and lead, the multiple of the high-flow level with the best CSI, hits / (hits + misses + false warnings), of all high-flow days in the validation "
                    f"years; {days(1)}: {', '.join(f'{name} x{levels[n][1]:.2f}' for name, n in KINDS.items())}"
                    + (f"; 06_flood_analysis.py stores no level for {' and '.join(no_level)}, so theirs is chosen here by the same rule" if no_level else "")
                    + f"). Then {listed(p1(at_val[lk(k)]['missed_by_all_kinds']['share']) for k in F.LEADS)} of the onsets are missed "
                    f"by all {NK} kinds 1, 2 and 3 days ahead ({listed(str(at_val[lk(k)]['missed_by_all_kinds']['count']) + ' of ' + str(at_val[lk(k)]['n_onsets']) for k in F.LEADS)}), "
                    f"against {listed(p1(out[lk(k)]['missed_by_all_kinds']['share']) for k in F.LEADS)} at the high-flow level itself: at most {100 * gap:.1f} percentage points apart. Of "
                    f"those, the LSTM given rain up to the evening of the issue day warns of {both('warned_with_rain_to_evening')}, and the LSTM given the perfect weather forecast, an upper "
                    f"bound that cannot be had in practice, of {both('warned_with_perfect_forecast')} (at_levels_chosen_on_validation).")
    notes.append("What this check does not show. It looks at two warning levels for each model, the high-flow level itself and the level chosen on the validation years. A clearly lower "
                 "level gives every model more warnings, right and false, and fewer onsets are then missed by all; that trade-off is not looked at here (06_flood_analysis.py shows it model "
                 "by model). And it holds among the models tried, each with the settings of its own script: another network, or a longer search of settings, could miss other onsets.")
else:
    notes.append("What this check does not show. It uses one warning level, the high-flow level itself (a prediction set has no forecast on some validation day, so no level could be chosen "
                 "there): a lower level gives every model more warnings, right and false, and that trade-off is not looked at here (06_flood_analysis.py tunes the level model by model). And "
                 "it holds among the models tried, each with the settings of its own script: another network, or a longer search of settings, could miss other onsets.")
if F.QUICK:
    notes.append("QUICK smoke test: the forecasts are those of the tiny smoke-test models in results/quick/preds - numbers are not results.")

R = dict(question=f"Do the {NK} kinds of model that read {INPUTS} miss the same onsets, and how many of the onsets that all of them miss does the LSTM warn of when it "
                  "is given more information?",
         quick_mode=F.QUICK, headline=headline, kinds=KINDS, more_information=MORE, lstm_seeds=seed_sets, test_years=[a, b], n_basins=len(ids), warning_level=TRIGGER,
         prediction_sets={n: dict(family=FAMILY[n], written_by=META[n].get("script"), seeds=META[n].get("seeds", [])) for n in sets},
         how_to_read=dict(
             shares="every share is a fraction between 0 and 1; all counts are for the test years with the basins pooled; models and also_missed_of_lstm_misses are keyed by prediction set, and "
                    "kinds gives the plain name of each (prediction_sets: its family as in flood.MODEL_COL, the script that wrote it, its seeds)",
             n_onsets="high-flow days whose forecast was issued while the river was still at or below the high-flow level",
             onsets_dropped="onsets of the flow record left out because a prediction set has no forecast for the day (0 = every set is scored on every onset)",
             basin_days_scored="all days whose forecast was issued while the river was at or below the level: the onsets and the days that stayed at or below it",
             models="per prediction set: onsets warned (hits) and missed (misses); false_alarms = warnings, issued while the river was at or below the level, for days that stayed at or "
                    "below it; pod = hits / n_onsets; far = false_alarms / (hits + false_alarms)",
             missed_by_all_kinds="onsets that none of the kinds of model warned of; share = of n_onsets; share_if_independent = the product of the kinds' own shares of missed onsets; "
                                 "share_if_independent_by_basin = the same product taken basin by basin, weighted by the basin's onsets (the basins differ in how many of their onsets "
                                 "the kinds miss)",
             warned_by_any_kind="onsets with a warning from at least one kind (share = of n_onsets); false_alarms = days that stayed at or below the level with a warning from at least one "
                                "kind; far = false_alarms / (count + false_alarms)",
             warned_by_all_kinds="onsets that every kind warned of; share = of n_onsets",
             of_missed_by_all="of the onsets missed by all kinds: those the LSTM given more information warned of; share = of missed_by_all_kinds.count",
             lstm_misses="count = onsets the LSTM (lstm_wq) misses; warned_by_another_kind = those of them that at least one other kind warned of, share = of count",
             also_missed_of_lstm_misses="of the onsets the LSTM (lstm_wq) misses: those the other kind also misses; share = of lstm_misses.count; share_if_independent = the other kind's own "
                                        "share of missed onsets (misses / n_onsets), which is what share would be if the two models' misses were unrelated; "
                                        "share_if_independent_by_basin = the same with the other kind's share of missed onsets taken basin by basin",
             seed_to_seed="the same between two single seeds of the LSTM: of the onsets one seed misses, the share the other also misses; min and max over every ordered pair of seeds; "
                          "pairs lists them (misses = onsets the first seed misses, count = those the other seed also misses); share_if_independent as above",
             at_levels_chosen_on_validation="the same counts as under l1, l2 and l3 (same keys) when every prediction set warns at the level chosen for it on the validation years, not "
                                            "at the high-flow level itself; levels = that level per prediction set and lead, as a multiple of the high-flow level; rule = how it is chosen; "
                                            "null if no level could be chosen",
             checks="stored_counts: one row per prediction set, lead and results file; key holds <count> for hits, misses and false_alarms; asserted = the script stops if the row is not "
                    "equal. levels_chosen_on_validation: one row per prediction set and lead for which 06_flood_analysis.py stored a level"),
         **out,
         at_levels_chosen_on_validation=dict(
             rule="for each prediction set and lead, the multiple of the high-flow level with the best CSI (hits / (hits + misses + false warnings)) of all high-flow days in the validation "
                  f"years ({va} to {vb}), basins pooled; a tie goes to the multiple closest to 1; multiples tried: {GRID[0]:.2f} to {GRID[-1]:.2f} (the rule and the grid of "
                  "06_flood_analysis.py); the test years choose nothing",
             levels={n: {lk(k): levels[n][k] for k in F.LEADS} for n in sets}, **at_val) if at_val else None,
         checks=checks, notes=notes, runtime_s=round(time.time() - t_start, 1))
(F.RES_OUT / "same_misses.json").write_text(json.dumps(R, indent=1))

for k in F.LEADS:
    r = out[lk(k)]
    print(f"\n{days(k)}: {r['n_onsets']} onsets on {r['basin_days_scored']:,} basin-days ({onsets_dropped[k]} onsets dropped)")
    for n in sets:
        m = r["models"][n]
        print(f"  {n:16s} warned {m['hits']:4d} = {p1(m['pod']):>6s}   missed {m['misses']:4d}   false warnings {m['false_alarms']:4d} = {p1(m['far']):>6s}")
    print(f"  missed by all {NK} kinds: {r['missed_by_all_kinds']['count']} = {p1(r['missed_by_all_kinds']['share'])} ({p1(r['missed_by_all_kinds']['share_if_independent'])} if independent, "
          f"{p1(r['missed_by_all_kinds']['share_if_independent_by_basin'])} if independent inside each basin)"
          f" | warned by any: {r['warned_by_any_kind']['count']} = {p1(r['warned_by_any_kind']['share'])}, false warnings {r['warned_by_any_kind']['false_alarms']} = "
          f"{p1(r['warned_by_any_kind']['far'])} | warned by all: {r['warned_by_all_kinds']['count']}")
    print(f"  of the {r['missed_by_all_kinds']['count']} missed by all: " + "; ".join(f"{n} warns of {more(r, key)['count']} = {p1(more(r, key)['share'])}" for key, n in MORE.items()))
    print(f"  of the {r['lstm_misses']['count']} onsets the LSTM misses: " + "; ".join(also(r, n) for n in others)
          + f"; another kind warns of {r['lstm_misses']['warned_by_another_kind']['count']}")
    if r["seed_to_seed"]:
        s = r["seed_to_seed"]
        print(f"  two seeds of the LSTM: {p1(s['min'])} to {p1(s['max'])} ({p1(s['share_if_independent']['min'])} to {p1(s['share_if_independent']['max'])} if independent)")
    if at_val:
        v = at_val[lk(k)]
        print(f"  at the levels chosen on the validation years ({', '.join(f'{n} x{levels[n][k]:.2f}' for n in sets)}): missed by all {NK} kinds {v['missed_by_all_kinds']['count']} = "
              f"{p1(v['missed_by_all_kinds']['share'])} | warned by any: {v['warned_by_any_kind']['count']}, false warnings {p1(v['warned_by_any_kind']['far'])} | of those missed by all: "
              + "; ".join(f"{n} warns of {more(v, key)['count']} = {p1(more(v, key)['share'])}" for key, n in MORE.items()))
print(f"\n{headline}\nstored counts equal: {checks['stored_counts_equal']}; stored warning levels equal: {checks['levels_chosen_on_validation']['stored_levels_equal']}"
      + ("" if (checks["all_stored_counts_equal"] and checks["levels_chosen_on_validation"]["all_stored_levels_equal"]) or not F.QUICK else
         " (smoke test, the script does not stop: the quick results files and the quick forecasts may come from different smoke runs)"))
print(f"wrote {rel(F.RES_OUT / 'same_misses.json')} ({R['runtime_s']} s)")
