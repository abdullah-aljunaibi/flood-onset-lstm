"""ROUND 2, check (e), second part: the yardstick fields of the JSON (LSTM against the simple models on the same inputs; what the new input adds to each model) recomputed with own code."""
import sys; sys.dont_write_bytecode = True       # keep the project folder free of __pycache__ even when this file is started without the runner
from scipy.stats import wilcoxon
from r2_common import *
import flood as F
from r2_scoring import *

R = jload("results/rain_to_midnight.json"); Y = R["yardstick"]
ids = list(data())
def nse_b(name, lead, a=TEST[0], b=TEST[1]):
    d = nse_per_basin(name, lead, a, b); return np.array([d[g] for g in ids])
def seeds_nse(name, lead):
    return [float(np.median(list(nse_per_basin(f"{name}_s{s}", lead).values()))) for s in (0, 1, 2)]
out = {}; bad = []; n_cmp = 0
def chk(label, a, b, tol=1e-9):
    global n_cmp
    n_cmp += 1
    if isinstance(a, (bool, str)): ok = (a == b)
    else: ok = (b is None and a != a) or abs(a - b) <= tol
    if not ok: bad.append((label, a, b))

for T, n in (("wqh", "base_linreg_wqh"), ("wqh", "base_rf_wqh"), ("wqh", "base_persistence"), ("wq", "base_linreg_wq"), ("wq", "base_rf_wq")):
    for k in (1, 2, 3):
        c = Y["lstm_against_simple"][f"lstm_{T}__{n}"][f"l{k}"]
        a, b = nse_b(f"lstm_{T}", k), nse_b(n, k)
        try: p = float(wilcoxon(a, b).pvalue)
        except ValueError: p = float("nan")
        sm = seeds_nse(f"lstm_{T}", k); bm = float(np.median(b))
        spread = "below every seed" if bm < min(sm) else "above every seed" if bm > max(sm) else "inside the seed spread"
        n_hi = int((a > b).sum())
        most, few = n_hi > 6, n_hi < 6
        small = p < 0.05
        below = spread == "below every seed"
        r = ("lstm" if below else "lstm_avg") if (most and small) else "simple" if (few and small) else ("lstm_med" if (most and below) else "tie")
        pods = [counts(f"lstm_{T}_s{s}", k)["pod"] for s in (0, 1, 2)]
        cT, cB = counts(f"lstm_{T}", k), counts(n, k)
        key = f"{T}__{n}__l{k}"
        for lab, mine, theirs in (("lstm_nse_median", float(np.median(a)), c["lstm_nse_median"]), ("baseline_nse_median", bm, c["baseline_nse_median"]), ("n_basins_lstm_higher", n_hi, c["n_basins_lstm_higher"]), ("wilcoxon_p", p, c["wilcoxon_p"]),
                                  ("median_nse_difference", float(np.median(a - b)), c["median_nse_difference"]), ("lstm_seed_median_min", min(sm), c["lstm_seed_median_min"]), ("lstm_seed_median_max", max(sm), c["lstm_seed_median_max"]),
                                  ("baseline_against_seed_spread", spread, c["baseline_against_seed_spread"]), ("lstm_onset_pod", cT["pod"], c["lstm_onset_pod"]), ("baseline_onset_pod", cB["pod"], c["baseline_onset_pod"]),
                                  ("lstm_onset_pod_seed_min", min(pods), c["lstm_onset_pod_seed_range"][0]), ("lstm_onset_pod_seed_max", max(pods), c["lstm_onset_pod_seed_range"][1]), ("reading", r, c["reading"])):
            chk(key + ":" + lab, mine, theirs)
        for si, s in enumerate((0, 1, 2)):
            chk(key + f":basins_by_seed{s}", int((nse_b(f"lstm_{T}_s{s}", k) > b).sum()), c["n_basins_lstm_higher_by_seed"][si])
        out[key] = dict(reading=r, n_higher=n_hi, p=p, baseline_vs_seeds=spread, lstm_pod=cT["pod"], baseline_pod=cB["pod"], lstm_seed_pod_range=[min(pods), max(pods)], hits=(cT["hits"], cB["hits"]))
print(f"lstm_against_simple (5 pairs x 3 leads): {n_cmp} values recomputed with own code, {len(bad)} differ", bad[:3])
for key in ("wqh__base_linreg_wqh__l1", "wqh__base_rf_wqh__l1", "wqh__base_persistence__l1"):
    v = out[key]
    print(f"  {key}: reading '{v['reading']}', LSTM higher in {v['n_higher']} of 12 basins, p {v['p']:.3f}, simple model {v['baseline_vs_seeds']}; onsets LSTM {100*v['lstm_pod']:.1f}% (seeds {100*v['lstm_seed_pod_range'][0]:.1f}-{100*v['lstm_seed_pod_range'][1]:.1f}%, {v['hits'][0]} hits) vs simple {100*v['baseline_pod']:.1f}% ({v['hits'][1]} hits)")

# what the new input adds to each model
n0 = n_cmp
for kind, (new, old, lstm) in {"lstm": ("lstm_wqh", "lstm_wq", True), "ridge regression": ("base_linreg_wqh", "base_linreg_wq", False), "random forest": ("base_rf_wqh", "base_rf_wq", False)}.items():
    for k in (1, 2, 3):
        j = Y["new_input_given_to_each_model"][kind]["by_lead"][f"l{k}"]
        a, b = nse_b(new, k), nse_b(old, k)
        try: p = float(wilcoxon(a, b).pvalue)
        except ValueError: p = float("nan")
        cn, co = counts(new, k), counts(old, k)
        for lab, mine, theirs in (("nse_median_with", float(np.median(a)), j["nse_median_with"]), ("nse_median_without", float(np.median(b)), j["nse_median_without"]), ("nse_gain", float(np.median(a) - np.median(b)), j["nse_gain"]), ("n_basins_higher", int((a > b).sum()), j["n_basins_higher"]),
                                  ("median_basin_difference", float(np.median(a - b)), j["median_basin_difference"]), ("wilcoxon_p", p, j["wilcoxon_p"]), ("onset_warned_with", cn["pod"], j["onset_warned_with"]), ("onset_warned_without", co["pod"], j["onset_warned_without"]),
                                  ("onset_gain_points", 100 * (cn["pod"] - co["pod"]), j["onset_gain_points"]), ("onset_hits_with", cn["hits"], j["onset_hits_with"]), ("onset_hits_without", co["hits"], j["onset_hits_without"]), ("onsets_in_the_test_years", cn["n_onsets"], j["onsets_in_the_test_years"]),
                                  ("onset_false_with", cn["far"], j["onset_false_with"]), ("onset_false_without", co["far"], j["onset_false_without"])):
            chk(f"{kind}:l{k}:{lab}", mine, theirs)
        if k == 1:
            out.setdefault("new_input_adds", {})[kind] = dict(nse_gain=float(np.median(a) - np.median(b)), onset_gain_points=100 * (cn["pod"] - co["pod"]), n_higher=int((a > b).sum()), p=p, hits=(cn["hits"], co["hits"]))
            print(f"  new input added to the {kind:17s}: NSE {np.median(a)-np.median(b):+.3f}, onsets {100*(cn['pod']-co['pod']):+.1f} points ({cn['hits']} vs {co['hits']} hits), NSE higher in {(a>b).sum()} of 12 basins, p {p:.4f}")
print(f"new_input_given_to_each_model (3 models x 3 leads): {n_cmp - n0} values recomputed, differences so far {len(bad)}")
# validation-year choice of the best simple model, and the verdicts
print()
best_json = Y["best_simple_on_the_new_input_by_validation"]
for k in (1, 2, 3):
    cands = ("base_persistence", "base_linreg_wqh", "base_rf_wqh")
    v = {n: float(np.median(list(nse_per_basin(n, k, "2000-10-01", "2005-09-30").values()))) for n in cands}
    best = max(cands, key=lambda n: v[n]); chk(f"best_simple_l{k}", best, best_json[f"l{k}"])
    print(f"  validation-year median NSE lead {k}: " + ", ".join(f"{n} {x:.3f}" for n, x in v.items()) + f" -> best {best} (JSON {best_json[f'l{k}']})")
out["n_compared"] = n_cmp; out["differences"] = bad
print("\nTOTAL values compared:", n_cmp, "| differences:", len(bad), bad[:3])
# the verdicts of the simple models (stored criteria): plain level
crit = R["verdicts"]["criteria"]
import re
pod_go, far_go, csi_c = [float(re.search(rf"{k} {op} ([0-9.]+)", t).group(1)) for k, op, t in (("pod", ">=", crit["GO"]), ("far", "<=", crit["GO"]), ("csi", ">=", crit["CAUTION"]))]
def verdict(t):
    if t["hits"] + t["misses"] == 0: return "NO EVENTS"
    if t["pod"] >= pod_go and t["far"] <= far_go: return "GO"
    return "CAUTION" if t["csi"] >= csi_c else "NOT YET"
vn = 0; vbad = []
for n in ("base_linreg_wqh", "base_rf_wqh", "lstm_wqh_to09", "lstm_wqh_utc", "lstm_wqnh"):
    for k in (1, 2, 3):
        c = counts(n, k)
        # the "all days" table is not recomputed here; the onset verdict at the plain level is
        mine = verdict(c)
        jv = (R["verdicts"]["sets"][n[5:]][f"l{k}"]["plain"]["onset"]["verdict"] if n.startswith("lstm_") else R["yardstick"]["table_test"][f"l{k}"][n]["onset_verdict_plain"])
        vn += 1; vbad += [(n, k, mine, jv)] if mine != jv else []
print(f"onset verdicts at the plain level of the 5 new sets x 3 leads: {vn} compared with my application of the stored criteria, {len(vbad)} differ", vbad)
out["verdicts_checked"] = vn; out["verdict_differences"] = vbad
jsave("r2e2_yardstick.json", out)
