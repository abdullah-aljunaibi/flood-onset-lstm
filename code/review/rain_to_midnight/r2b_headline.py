"""ROUND 2, check (b): headline numbers of the three new LSTM sets and the two simple models: engine recomputation against the JSON, and the reviewer's own pooled onset counts."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from r2_common import *
import flood as F
from r2_scoring import *

R = jload("results/rain_to_midnight.json")
D = F.load_data(); thr = F.thresholds(D)
NEW_L = ["wqh_to09", "wqh_utc", "wqnh"]
NEW_S = ["base_linreg_wqh", "base_rf_wqh"]
REF_L = ["wq", "wqh", "wqh_older", "wqn", "wqf"]
REF_S = ["base_linreg_wq", "base_rf_wq", "base_persistence"]
out = {"engine": {}, "own": {}, "per_seed": {}}
dev_json = 0
EV = {}
for n in [f"lstm_{x}" for x in NEW_L + REF_L] + NEW_S + REF_S:
    E = F.evaluate(F.load_preds(n), thr); EV[n] = E
    out["engine"][n] = {}
    for k in F.LEADS:
        s = E["summary"][f"l{k}"]
        mine = dict(nse_median=s["nse_median"], hits=s["onset"]["hits"], misses=s["onset"]["misses"], false_alarms=s["onset"]["false_alarms"], pod=s["onset"]["pod"], far=s["onset"]["far"], csi=s["onset"]["csi"])
        out["engine"][n][f"l{k}"] = mine
        # the JSON: LSTM sets in summary_test; simple models in yardstick.sets[n].test.summary
        if n.startswith("lstm_"):
            j = R["summary_test"]["sets"][n[5:]][f"l{k}"]
            st = dict(nse_median=j["nse_median"], hits=j["onset"]["hits"], misses=j["onset"]["misses"], false_alarms=j["onset"]["false_alarms"], pod=j["onset"]["pod"], far=j["onset"]["far"], csi=j["onset"]["csi"])
        elif n in NEW_S:
            j = R["yardstick"]["sets"][n]["test"]["summary"][f"l{k}"]
            st = dict(nse_median=j["nse_median"], hits=j["onset"]["hits"], misses=j["onset"]["misses"], false_alarms=j["onset"]["false_alarm" + "s"], pod=j["onset"]["pod"], far=j["onset"]["far"], csi=j["onset"]["csi"])
        else:
            j = R["yardstick"]["table_test"][f"l{k}"][n]
            st = dict(nse_median=j["nse_median"], hits=j["onset"]["hits"], misses=j["onset"]["misses"], false_alarms=j["onset"]["false_alarms"], pod=j["onset"]["pod"], far=j["onset"]["far"], csi=j["onset"]["csi"])
        for c in mine:
            a, b = mine[c], st[c]
            if a != a and b is None: continue                       # a NaN share is written as null
            dev_json = max(dev_json, abs(a - b))
out["max_abs_dev_engine_vs_json"] = dev_json
print("engine recomputation against the JSON (summary_test / yardstick), 13 sets x 3 leads x 7 numbers: max abs deviation", dev_json)
print()
print("%-18s %-9s %-24s %-26s %-26s" % ("set", "lead1 NSE", "lead1 onsets pod/far/csi", "lead2 NSE / pod", "lead3 NSE / pod"))
for n, e in out["engine"].items():
    print("%-18s %.4f   %.4f %.4f %.4f (%3d/%3d/%3d)  %.4f %.4f   %.4f %.4f" % (n, e["l1"]["nse_median"], e["l1"]["pod"], e["l1"]["far"] if e["l1"]["far"] == e["l1"]["far"] else float("nan"), e["l1"]["csi"], e["l1"]["hits"], e["l1"]["misses"], e["l1"]["false_alarms"], e["l2"]["nse_median"], e["l2"]["pod"], e["l3"]["nse_median"], e["l3"]["pod"]))

# own pooled counts without flood.evaluate
print("\nown pooled onset counts (lead 1, 2, 3), own loop over the npz arrays, own thresholds:")
want = {"lstm_wqh_to09": 125, "lstm_wqh_utc": 133, "lstm_wqnh": 194, "base_linreg_wqh": 164, "base_rf_wqh": 165}
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
    flag = f"  report says {want[n]}: {'AGREES' if want[n] == c1['hits'] else 'DIFFERS'}" if n in want else ""
    print(f"  {n:18s} lead 1: hits {c1['hits']:3d} of {c1['n_onsets']}  misses {c1['misses']:3d}  false {c1['false_alarms']:3d}  pod {c1['pod']:.6f}  far {c1['far']:.4f}{flag}")
out["max_abs_dev_own_vs_engine"] = dev_own
out["hits_stated_by_the_coordinator"] = {n: dict(stated=w, mine=out["own"][n]["l1"]["hits"], agrees=(w == out["own"][n]["l1"]["hits"])) for n, w in want.items()}
out["all_counts_equal_engine"] = all(v["equal_engine"] for n in out["own"] for v in out["own"][n].values())
print("all 39 set-lead count triples equal the engine's:", out["all_counts_equal_engine"], "| max |pod,far,csi own - engine|:", dev_own)

# per-seed values of the new LSTM sets against seed_ranges in the JSON
print("\nper-seed onset shares and median NSE (lead 1) of the new LSTM sets:")
for x in NEW_L:
    pods = [counts(f"lstm_{x}_s{s}", 1)["pod"] for s in (0, 1, 2)]
    nses = [float(np.median(list(nse_per_basin(f"lstm_{x}_s{s}", 1).values()))) for s in (0, 1, 2)]
    j = R["seed_ranges"]["sets"][x]["l1"]
    d = max(abs(min(pods) - j["onset_pod"][0]), abs(max(pods) - j["onset_pod"][1]), abs(min(nses) - j["nse_median"][0]), abs(max(nses) - j["nse_median"][1]),
            *[abs(a - b) for a, b in zip(pods, j["by_seed"]["onset_pod"])], *[abs(a - b) for a, b in zip(nses, j["by_seed"]["nse_median"])])
    out["per_seed"][x] = dict(onset_pod=pods, nse_median=nses, max_dev_vs_json=d)
    print(f"  {x:10s} onsets {[round(100*p,2) for p in pods]} %  NSE {[round(v,3) for v in nses]}   max |own - JSON seed_ranges| {d:.1e}")
jsave("r2b_headline.json", out)
