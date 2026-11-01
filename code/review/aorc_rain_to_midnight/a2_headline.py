"""AORC review, check (a): headline numbers of the four new LSTM sets (wqa, wqa_to09, wqa_utc, wqa_rep) and the two simple models (base_linreg_wqa, base_rf_wqa) with their references: engine recomputation
(flood.evaluate) against the results file, and the reviewer's own pooled onset counts from the npz arrays (own thresholds, no flood.evaluate); seed averages equal the day-by-day mean of the three seed
files; the observed flow of every file equals the project's flow table; the statements of the headline (313 of 611 = 51.2%, single trainings 48.6-55.3%, 18.7%, 17.7%, 37.5%, 30.0%, NSE 0.767 against 0.590,
ridge 55.2%, forest 48.0%, 10 of 12 basins, leads 2-3). About 20 seconds."""
import sys; sys.dont_write_bytecode = True
from a_common import *
import flood as F
from a_scoring import *

R = jload("results/aorc_rain_to_midnight.json")
D = F.load_data(); thr = F.thresholds(D); ids = list(D)
NEW_L = ["wqa", "wqa_to09", "wqa_utc", "wqa_rep"]
NEW_S = ["base_linreg_wqa", "base_rf_wqa"]
REF_L = ["wq", "wqh", "wqn", "wqnh", "wqf", "wqh_to09", "wqh_utc"]
REF_S = ["base_linreg_wq", "base_rf_wq", "base_persistence", "base_linreg_wqh", "base_rf_wqh"]
out = {"engine": {}, "own": {}, "per_seed": {}}
dev_json = 0
EV = {}
names = [f"lstm_{x}" for x in NEW_L + REF_L] + NEW_S + REF_S
for n in names:
    E = F.evaluate(F.load_preds(n), thr); EV[n] = E
    out["engine"][n] = {}
    for k in F.LEADS:
        s = E["summary"][f"l{k}"]
        mine = dict(nse_median=s["nse_median"], hits=s["onset"]["hits"], misses=s["onset"]["misses"], false_alarms=s["onset"]["false_alarms"], pod=s["onset"]["pod"], far=s["onset"]["far"], csi=s["onset"]["csi"])
        out["engine"][n][f"l{k}"] = mine
        if n.startswith("lstm_"):
            j = R["summary_test"]["sets"][n[5:]][f"l{k}"]
        else:
            j = R["yardstick"]["table_test"][f"l{k}"][n]
        st = dict(nse_median=j["nse_median"], hits=j["onset"]["hits"], misses=j["onset"]["misses"], false_alarms=j["onset"]["false_alarms"], pod=j["onset"]["pod"], far=j["onset"]["far"], csi=j["onset"]["csi"])
        for c in mine:
            a, b = mine[c], st[c]
            if a != a and b is None: continue                       # a NaN share is written as null
            dev_json = max(dev_json, abs(a - b))
out["max_abs_dev_engine_vs_json"] = dev_json
print(f"engine recomputation against the JSON (summary_test / yardstick.table_test), {len(names)} sets x 3 leads x 7 numbers: max abs deviation", dev_json)
print()
print("%-18s %-9s %-24s %-26s %-26s" % ("set", "lead1 NSE", "lead1 onsets pod/far/csi", "lead2 NSE / pod", "lead3 NSE / pod"))
for n, e in out["engine"].items():
    print("%-18s %.4f   %.4f %.4f %.4f (%3d/%3d/%3d)  %.4f %.4f   %.4f %.4f" % (n, e["l1"]["nse_median"], e["l1"]["pod"], e["l1"]["far"] if e["l1"]["far"] == e["l1"]["far"] else float("nan"), e["l1"]["csi"], e["l1"]["hits"], e["l1"]["misses"], e["l1"]["false_alarms"], e["l2"]["nse_median"], e["l2"]["pod"], e["l3"]["nse_median"], e["l3"]["pod"]))

# own pooled counts without flood.evaluate
print("\nown pooled onset counts (lead 1, 2, 3), own loop over the npz arrays, own thresholds:")
want = {"lstm_wqa": 313, "lstm_wqa_to09": None, "lstm_wqa_utc": None, "lstm_wqh": 183, "lstm_wq": 114, "base_linreg_wqa": None, "base_rf_wqa": None}
dev_own = 0
for n in EV:
    out["own"][n] = {}
    for k in F.LEADS:
        c = counts(n, k)
        e = out["engine"][n][f"l{k}"]
        eq = (c["hits"], c["misses"], c["false_alarms"]) == (e["hits"], e["misses"], e["false_alarms"])
        out["own"][n][f"l{k}"] = dict(c, equal_engine=eq)
        dev_own = max(dev_own, abs(c["pod"] - e["pod"]), abs(c["csi"] - e["csi"]), (abs(c["far"] - e["far"]) if c["far"] == c["far"] and e["far"] == e["far"] else 0))
    c1 = out["own"][n]["l1"]
    print(f"  {n:18s} lead 1: hits {c1['hits']:3d} of {c1['n_onsets']}  misses {c1['misses']:3d}  false {c1['false_alarms']:3d}  pod {c1['pod']:.6f}  far {c1['far']:.4f}")
out["max_abs_dev_own_vs_engine"] = dev_own
out["all_counts_equal_engine"] = all(v["equal_engine"] for n in out["own"] for v in out["own"][n].values())
print(f"all {len(names) * 3} set-lead count triples equal the engine's:", out["all_counts_equal_engine"], "| max |pod,far,csi own - engine|:", dev_own)

# the statements of the headline
o = out["own"]; pods = {n: o[n]["l1"]["pod"] for n in o}; hits = {n: o[n]["l1"]["hits"] for n in o}
sd = {x: [counts(f"lstm_{x}_s{s}", 1)["pod"] for s in (0, 1, 2)] for x in NEW_L + ["wq", "wqh"]}
nse = {n: out["engine"][n]["l1"]["nse_median"] for n in out["engine"]}
claims = {
    "313 of 611 = 51.2% with the AORC rain to midnight": (hits["lstm_wqa"], round(100 * pods["lstm_wqa"], 1)),
    "single trainings 48.6-55.3%": (round(100 * min(sd["wqa"]), 1), round(100 * max(sd["wqa"]), 1)),
    "18.7% with the rain table alone": round(100 * pods["lstm_wq"], 1),
    "17.7% with AORC over the table's own window": round(100 * pods["lstm_wqa_to09"], 1),
    "37.5% with AORC over the UTC day": round(100 * pods["lstm_wqa_utc"], 1),
    "30.0% with ERA5 to midnight (script 15)": round(100 * pods["lstm_wqh"], 1),
    "median NSE 0.767 against 0.590": (round(nse["lstm_wqa"], 3), round(nse["lstm_wq"], 3)),
    "ridge 55.2% and forest 48.0% given the same input": (round(100 * pods["base_linreg_wqa"], 1), round(100 * pods["base_rf_wqa"], 1)),
    "the replaced table (wqa_rep): 52%, single trainings 44.8-54.8%": (round(100 * pods["lstm_wqa_rep"], 1), round(100 * min(sd["wqa_rep"]), 1), round(100 * max(sd["wqa_rep"]), 1)),
}
out["claims"] = claims
print("\nthe statements of the headline against my own numbers:")
for k, v in claims.items(): print(f"  {k}: mine {v}")
# in how many basins does the result hold (NSE and onsets), own per-basin numbers
nb_w, nb_q = nse_per_basin("lstm_wqa", 1), nse_per_basin("lstm_wq", 1)
on_w, on_q = onset_table("lstm_wqa", 1), onset_table("lstm_wq", 1)
def basin_pod(t, g): d, h, w = t[g]; return (h & w).sum() / max(h.sum(), 1)
n_nse = sum(nb_w[g] > nb_q[g] for g in ids); n_on = sum(basin_pod(on_w, g) > basin_pod(on_q, g) for g in ids); n_on_ge = sum(basin_pod(on_w, g) >= basin_pod(on_q, g) for g in ids)
out["basins"] = dict(nse_wqa_above_wq=int(n_nse), onsets_wqa_above_wq=int(n_on), onsets_wqa_at_least_wq=int(n_on_ge), per_basin_pod={g: (float(basin_pod(on_w, g)), float(basin_pod(on_q, g))) for g in ids})
print(f"basins where wqa has the higher median NSE than wq: {n_nse} of 12; the higher share of onsets: {n_on} of 12 (at least as high: {n_on_ge})")
for k in (2, 3):
    print(f"lead {k}: wqa onsets {100 * pods_k:.1f}% vs wq {100 * pq_k:.1f}%, NSE {nk:.3f} vs {nq:.3f}" if False else f"lead {k}: wqa onsets {100*o['lstm_wqa'][f'l{k}']['pod']:.1f}% vs wq {100*o['lstm_wq'][f'l{k}']['pod']:.1f}%, NSE {out['engine']['lstm_wqa'][f'l{k}']['nse_median']:.3f} vs {out['engine']['lstm_wq'][f'l{k}']['nse_median']:.3f}  (ERA5 wqh: {100*o['lstm_wqh'][f'l{k}']['pod']:.1f}%, {out['engine']['lstm_wqh'][f'l{k}']['nse_median']:.3f})")

# seed averages, observed flow, per-seed values against seed_ranges
print("\nseed averages equal the mean of their three seed files; the observed flow of every file equals the project's flow table:")
dev_avg, obs_ok = {}, True
for x in NEW_L:
    A = load_npz(f"lstm_{x}"); S = [load_npz(f"lstm_{x}_s{s}") for s in (0, 1, 2)]
    dates, basins, obs, pred = A
    mean = np.mean([s[3] for s in S], axis=0)
    both_nan = np.array_equal(np.isnan(mean), np.isnan(pred))
    dev_avg[x] = dict(max_abs_diff=float(np.nanmax(np.abs(mean - pred))), nan_pattern_equal=bool(both_nan), obs_equal_across_seeds=bool(all(np.array_equal(S[0][2], s[2], equal_nan=True) for s in S)), obs_equal_to_average=bool(np.array_equal(S[0][2], obs, equal_nan=True)))
    for i, g in enumerate(basins):
        q = D[g].q_mm.reindex(dates).values
        obs_ok &= bool(np.allclose(obs[i], q, equal_nan=True, rtol=0, atol=1e-6))
    print(f"  {x:10s} mean of seeds vs saved average: max |diff| {dev_avg[x]['max_abs_diff']:.1e}; nan pattern equal {both_nan}; obs equal across seeds {dev_avg[x]['obs_equal_across_seeds']}")
out["seed_average"] = dev_avg; out["obs_equal_project_flow"] = obs_ok
print("observed flow of the four averaged sets equals data/derived flow table (flood.load_data) to 1e-6 mm/day on every day:", obs_ok)
for n in NEW_S:
    z = load_npz(n); q_ok = all(np.allclose(z[2][i], D[g].q_mm.reindex(z[0]).values, equal_nan=True, rtol=0, atol=1e-6) for i, g in enumerate(z[1])); out.setdefault("obs_equal_project_flow_simple", {})[n] = bool(q_ok)
print("simple models' observed flow equals the project's flow table:", out["obs_equal_project_flow_simple"])
print("\nper-seed onset shares and median NSE (lead 1) of the new sets against seed_ranges in the JSON:")
for x in NEW_L:
    pods_s = sd[x]
    nses = [float(np.median(list(nse_per_basin(f"lstm_{x}_s{s}", 1).values()))) for s in (0, 1, 2)]
    j = R["seed_ranges"]["sets"][x]["l1"]
    d = max(abs(min(pods_s) - j["onset_pod"][0]), abs(max(pods_s) - j["onset_pod"][1]), abs(min(nses) - j["nse_median"][0]), abs(max(nses) - j["nse_median"][1]),
            *[abs(a - b) for a, b in zip(pods_s, j["by_seed"]["onset_pod"])], *[abs(a - b) for a, b in zip(nses, j["by_seed"]["nse_median"])])
    out["per_seed"][x] = dict(onset_pod=pods_s, nse_median=nses, max_dev_vs_json=d)
    print(f"  {x:10s} onsets {[round(100*p,2) for p in pods_s]} %  NSE {[round(v,3) for v in nses]}   max |own - JSON seed_ranges| {d:.1e}")
jsave("a2_headline.json", out)
