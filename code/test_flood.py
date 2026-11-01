"""Checks that the shared engine (flood.py) is fair: scores match hand calculations, forecasts are aligned to the right
day, nothing from validation/test years reaches the scalers or thresholds, and the model cannot see the future.

The last check uses made-up data where the answer is known: flow(d) = 0.6 * rain(d-1) + 0.3 * rain(d), with random rain.
A model that reads weather only up to the issue day can then explain 80% of tomorrow's flow and nothing of days 2 and 3;
a model given the target days' weather can explain everything. Any leak of future data would break those numbers.
Run: python code/test_flood.py   (writes results/fair_test.json)
"""
import hashlib
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd

import flood as F

# ---- scores against hand calculations ----------------------------------------------------------------------------------
o = np.array([1.0, 2.0, 3.0, 4.0])
assert F.nse(o, o) == 1 and abs(F.nse(o, np.full(4, 2.5))) < 1e-12 and abs(F.nse(o, o + 1) - (1 - 4 / 5)) < 1e-12
assert abs(F.kge(o, o) - 1) < 1e-12 and abs(F.kge(o, 2 * o) - (1 - np.sqrt(2))) < 1e-12
t = F.contingency([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
assert (t["hits"], t["misses"], t["false_alarms"], t["correct_neg"]) == (2, 1, 1, 4)
assert abs(t["pod"] - 2 / 3) < 1e-12 and abs(t["far"] - 1 / 3) < 1e-12 and abs(t["csi"] - 0.5) < 1e-12 and t["bias"] == 1

# ---- alignment: a forecast made on day t for lead k must land on day t+k -------------------------------------------------
idx = pd.date_range(*F.PERIOD)
rng = np.random.default_rng(1)
d = pd.DataFrame({"q_mm": rng.gamma(2.0, 1.0, len(idx))}, index=idx)
p = F.frame_from_issue(d, {k: d.q_mm.values for k in F.LEADS})                    # "tomorrow = today"
for k in F.LEADS:
    assert np.allclose(p[f"l{k}"].values[k:], d.q_mm.values[:-k]) and p[f"l{k}"].isna().sum() == k
m = F.part_mask(idx, 3, "test")
assert idx[m][0] == pd.Timestamp("2005-09-28") and idx[m][-1] == pd.Timestamp("2014-09-27")      # issue days whose day+3 is a test day

# ---- warning scores on a hand-made case ----------------------------------------------------------------------------------
q = pd.Series(1.0, index=idx)
q["2006-01-10":"2006-01-12"] = 9.0                                              # one 3-day high-flow event in the test years
pers = F.frame_from_issue(pd.DataFrame({"q_mm": q}), {k: q.values for k in F.LEADS})
s = F.flood_scores(pers, 5.0, 1)
assert (s["all"]["hits"], s["all"]["misses"], s["all"]["false_alarms"]) == (2, 1, 1)             # day 1 missed, the day after the end is a false alarm
assert (s["onset"]["hits"], s["onset"]["misses"], s["onset"]["false_alarms"]) == (0, 1, 0)       # persistence never warns of an onset
s3 = F.flood_scores(pers, 5.0, 3)
assert (s3["all"]["hits"], s3["all"]["misses"], s3["all"]["false_alarms"]) == (0, 3, 3) and s3["onset"]["misses"] == 3
E = F.evaluate({"x": pers}, {"x": 5.0})
assert E["basins"]["x"]["l1"]["coverage"] == 1 and E["summary"]["l1"]["all"]["hits"] == 2

# ---- made-up basins with a known rain -> flow rule -------------------------------------------------------------------------
def fake(seed):
    r = np.random.default_rng(seed)
    f = pd.DataFrame({c: r.normal(size=len(idx)) for c in F.FORCING}, index=idx)
    f["prcp"] = r.exponential(4.0, len(idx))
    f["q_mm"] = 0.6 * np.concatenate([[0.0], f.prcp.values[:-1]]) + 0.3 * f.prcp.values
    return f

D = {f"b{i}": fake(i) for i in range(3)}
thr = F.thresholds(D)
D2 = {g: v.copy() for g, v in D.items()}
for v in D2.values():
    v.loc[F.SPLIT["val"][0]:, ["prcp", "q_mm"]] *= 50                           # corrupt everything after the training years
assert F.thresholds(D2) == thr, "the flood threshold must depend on training years only"
tiny = dict(hidden=16, warmup=30, block=60, updates=1, eval_every=1, dropout=0.0)
a, b = F.fit(F.Cfg(use_flow=True, **tiny), D, list(D)), F.fit(F.Cfg(use_flow=True, **tiny), D2, list(D2))
assert all(np.allclose(a.sc[k], b.sc[k]) for k in a.sc), "scalers must depend on training years only"

small = dict(hidden=16, warmup=30, block=60, updates=400, eval_every=100, patience=4, dropout=0.0, lr=5e-3)
past = F.evaluate(F.fit(F.Cfg(**small), D, list(D)).predict(D), thr)["summary"]
perfect = F.evaluate(F.fit(F.Cfg(future=True, **small), D, list(D)).predict(D), thr)["summary"]
n_past, n_perf = [past[f"l{k}"]["nse_median"] for k in F.LEADS], [perfect[f"l{k}"]["nse_median"] for k in F.LEADS]
print("past weather only   test NSE by lead:", np.round(n_past, 3), " (expected about 0.80, 0, 0)")
print("with target-day rain test NSE by lead:", np.round(n_perf, 3), " (expected about 1, 1, 1)")
assert 0.70 < n_past[0] < 0.83, "lead 1 must explain the 80% that yesterday's rain decides - not more"
assert max(n_past[1:]) < 0.05, "leads 2-3 depend on rain that has not fallen: any skill here would be a leak"
assert min(n_perf) > 0.95, "with the target days' weather the rule is fully learnable"

# the memory-limited forecast must agree with the normal one when it is given the same amount of history
run = F.fit(F.Cfg(**small), D, ["b0"])
full, ctx = run.predict(D, warmup=30, block=1)["b0"], run.predict_context(D, context=31)["b0"]
both = (full.l1.notna() & ctx.l1.notna()).values
assert both.sum() > 3000 and np.allclose(full.l1.values[both], ctx.l1.values[both], atol=1e-4)
# a second basin set (load_data(ids=...)) gets a cache of its own and leaves the first set's tables alone (made-up records: no CAMELS file is read)
made = {f"m{i}": fake(40 + i) for i in range(5)}                                # the first set is m0-m2, the second m4, m2, m3 (one basin in common)
made["gap"] = made["m0"].copy()
made["gap"].iloc[100, made["gap"].columns.get_loc("q_mm")] = np.nan             # one flow value missing
made["short"] = made["m0"].iloc[:-1]                                            # one day missing
real = F.DERIVED, F.load_basin, F.basins
try:
    with tempfile.TemporaryDirectory() as tmp:
        F.DERIVED, F.load_basin = pathlib.Path(tmp), lambda g, source="daymet": made[g]
        F.basins = lambda ids=None: pd.DataFrame(index=["m0", "m1", "m2"] if ids is None else list(ids))
        first, f1 = F.load_data(), F.DERIVED / "camels12_daymet.pkl"
        h1 = hashlib.sha256(f1.read_bytes()).hexdigest()
        second = F.load_data(ids=["m4", "m2", "m3"])
        cols = F.FORCING + ["q_mm"]
        assert list(first) == ["m0", "m1", "m2"] and list(second) == ["m4", "m2", "m3"], "load_data(ids=...) must return the basins asked for, in the order asked"
        assert all(first[g].equals(made[g][cols]) for g in first) and all(second[g].equals(made[g][cols]) for g in second), "the tables must be the made-up records, unchanged"
        assert hashlib.sha256(f1.read_bytes()).hexdigest() == h1 and len(list(F.DERIVED.glob("camels*.pkl"))) == 2, "the second set needs a cache of its own and must leave the first one alone"
        again = F.load_data()
        assert list(again) == list(first) and all(again[g].equals(first[g]) for g in first), "loading a second set must not change the first set's tables"
        swapped = F.load_data(ids=["m3", "m4", "m2"])                           # the same set in another order: the same cache, the order asked for
        assert list(swapped) == ["m3", "m4", "m2"] and all(swapped[g].equals(second[g]) for g in swapped) and len(list(F.DERIVED.glob("camels*.pkl"))) == 2, \
            "the same ids in another order must come from the same cache file, in the order asked"
        for bad in ("gap", "short"):
            try:
                F.load_data(ids=[bad])
            except AssertionError:
                continue
            raise SystemExit(f"a basin with a gap ({bad}) was accepted")
finally:
    F.DERIVED, F.load_basin, F.basins = real
# candidate_ids() and basins(ids=...) on a small made-up audit file in a temporary folder (no project file is read or written)
with tempfile.TemporaryDirectory() as tmp:
    res, keep = pathlib.Path(tmp), (F.RES, F.attributes)
    aud = pd.DataFrame({"gauge_id": ["x1", "x2", "x3", "x4", "x5"], "missing_share": [0.0, 0.0001, 0.0, 0.0, 0.0], "forcing_missing": [0, 0, 0, 3, 0], "landscape": ["L1", "L2", "L1", "L3", "L2"]})
    aud["complete"] = (aud.missing_share == 0) & (aud.forcing_missing == 0)                    # x2 has a missing flow day, x4 missing forcing: x1, x3, x5 are complete
    aud.to_csv(res / "basin_audit.csv", index=False)
    attrs = pd.DataFrame({"area_gages2": [100.0, 200.0, 400.0, 800.0, 1600.0], **{c: np.arange(5, dtype=float) for c in F.STATIC if c != "log_area"}}, index=["x1", "x2", "x3", "x4", "x5"])
    F.RES, F.attributes = res, lambda: attrs
    try:
        assert F.candidate_ids() == ["x1", "x3", "x5"], "candidate_ids() must give the complete basins of the audit file, in the file's order"
        b = F.basins(ids=["x5", "x1"])
        assert list(b.index) == ["x5", "x1"] and list(b.landscape) == ["L2", "L1"], "basins(ids) must give the rows in the order asked, with the landscape of the audit file"
        assert np.allclose(b.log_area, np.log10([1600.0, 100.0])) and b.short.isna().all(), "log_area is log10 of the area, and short is empty for a basin that is not among the 12"
        for bad, why in ((["x1", "x1"], "a repeated id"), (["x1", "nope"], "an id that is not in the audit file")):
            try:
                F.basins(ids=bad)
            except AssertionError:
                continue
            raise SystemExit(f"basins(ids) accepted {why}")
        attrs.loc["x3", "elev_mean"] = np.nan                                                   # one missing property
        try:
            F.basins(ids=["x3"])
        except AssertionError:
            pass
        else:
            raise SystemExit("basins(ids) accepted a basin with a missing property")
        aud.loc[0, "complete"] = False                                                          # a complete column that does not follow the rule of 01_data_audit.py
        aud.to_csv(res / "basin_audit.csv", index=False)
        try:
            F.candidate_ids()
        except AssertionError:
            pass
        else:
            raise SystemExit("candidate_ids() accepted an audit file whose complete column does not follow the rule")
    finally:
        F.RES, F.attributes = keep
(F.RES_OUT / "fair_test.json").write_text(json.dumps(dict(
    rule="flow(d) = 0.6 * rain(d-1) + 0.3 * rain(d), random rain, 3 made-up basins",
    expected=dict(past_weather_only=[0.8, 0.0, 0.0], with_target_day_weather=[1.0, 1.0, 1.0]),
    measured=dict(past_weather_only=[round(float(v), 4) for v in n_past], with_target_day_weather=[round(float(v), 4) for v in n_perf]),
    checks=["scores equal hand calculations", "a forecast made on day t for lead k lands on day t+k", "warning tables equal a hand-made case",
            "flood threshold unchanged when all data after the training years is corrupted", "scalers unchanged when all data after the training years is corrupted",
            "no skill at leads 2-3 where the answer depends on rain that has not fallen", "full skill when the target days' weather is given",
            "memory-limited forecast equals the normal forecast given the same history",
            "a second basin set loaded with ids= has a cache of its own and leaves the first set's tables unchanged; a basin with a gap or a missing day is refused",
            "candidate_ids() gives the complete basins of a made-up audit file in file order and basins(ids=...) the rows in the order asked; a repeated or unknown id, a missing property and a complete column that breaks the rule are refused"],
    passed=True), indent=1))
print("all checks passed")
