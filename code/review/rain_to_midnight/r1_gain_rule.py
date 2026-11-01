"""ROUND 1, checks 6a to 6c (re-written in round 2): seed ranges and per-seed values of the first twelve trainings (own counts), the E1 success rule flags (with and without the river gauge), the Wilcoxon
signed-rank p of the per-basin NSE for six comparisons at three leads, the share of the gap to lstm_wqf closed, and the paired onset counts on the 611 onsets, each compared with the numbers recorded in
results/review_rain_to_midnight.json (round 1). About half a minute."""
import sys; sys.dont_write_bytecode = True
from scipy.stats import wilcoxon, binomtest
from r2_common import *
import flood as F
from r2_scoring import *

REC1 = jload("results/review_rain_to_midnight.json"); NUM = REC1["numbers"]
ids = list(data()); SEEDS = (0, 1, 2)
sets = ["wq", "wqn", "wqh", "wqh_older", "wqf", "w", "wn", "wh", "wh_older"]
pod = {n: [counts(f"lstm_{n}_s{s}", 1)["pod"] for s in SEEDS] for n in sets}
nse = {n: [float(np.median(list(nse_per_basin(f"lstm_{n}_s{s}", 1).values()))) for s in SEEDS] for n in sets}
dev = max(max(abs(a - b) for a, b in zip(pod[n], NUM["onsets_by_seed_lead1"][n])) for n in sets)
dev = max(dev, max(max(abs(a - b) for a, b in zip(nse[n], NUM["median_nse_by_seed_lead1"][n])) for n in sets))
print(f"6a: per-seed onset shares and median NSEs (lead 1) of 9 sets recomputed: max |own - round-1 record| = {dev:.1e}")
above = lambda a, b: min(a) > max(b)
avg_pod = {n: counts(f"lstm_{n}", 1)["pod"] for n in sets}
E1 = {}
for fam, (t, c, base, nl) in {"with_river_gauge": ("wqh", "wqh_older", "wq", "wqn"), "without_river_gauge": ("wh", "wh_older", "w", "wn")}.items():
    E1[fam] = dict(every_seed_above_every_nldas_seed=above(pod[t], pod[nl]), every_seed_above_30=min(pod[t]) > 0.30, control_flat=not above(pod[c], pod[base]), every_seed_above_every_control_seed=above(pod[t], pod[c]))
    E1[fam]["met"] = all([above(pod[t], pod[nl]), above(pod[t], pod[c]), above(pod[t], pod[base]), E1[fam]["control_flat"]])      # the four parts of the rule of 15_rain_to_midnight.py
    print(f"6c: E1 success rule {fam}: seeds {[round(100*x, 2) for x in pod[t]]}%, NLDAS seeds {[round(100*x, 2) for x in pod[nl]]}%, control seeds {[round(100*x, 2) for x in pod[c]]}%, rain table seeds {[round(100*x, 2) for x in pod[base]]}% -> {E1[fam]}")
rc = {c["id"]: c["numbers"] for c in REC1["checks"]}["6c"]
ok6c = all(E1[f][k] == rc[f][k] for f in E1 for k in ("met", "control_flat", "every_seed_above_every_nldas_seed", "every_seed_above_30"))
print("    equal to the round-1 record (met, control_flat, every seed above NLDAS / above 30%):", ok6c)
# Wilcoxon on per-basin NSE
ids_ = ids
def nse_b(name, lead): d = nse_per_basin(name, lead); return np.array([d[g] for g in ids_])
PAIRS = [("wqh", "wqh_older"), ("wqh", "wq"), ("wqh", "wqn"), ("wh", "wh_older"), ("wh", "w"), ("wh", "wn")]
wdev, wn = 0.0, 0
for k in (1, 2, 3):
    for a, b in PAIRS:
        va, vb = nse_b(f"lstm_{a}", k), nse_b(f"lstm_{b}", k); p = float(wilcoxon(va, vb).pvalue); r = NUM["wilcoxon_nse"][f"l{k}_{a}_vs_{b}"]
        wdev = max(wdev, abs(p - r["p"]), abs(float(np.median(va - vb)) - r["median_diff"]), abs(int((va > vb).sum()) - r["n_higher"])); wn += 1
        if k == 1: print(f"6b: lead 1 {a:3s} vs {b:9s}: higher in {int((va > vb).sum()):2d} of 12 basins, median basin difference {np.median(va - vb):+.4f}, Wilcoxon p = {p:.5f}")
print(f"    {wn} Wilcoxon results: max |own - round-1 record| = {wdev:.1e}")
# closure shares at lead 1
cl = {}
for fam, (t, c, base, ceil_, nl) in {"wqh": ("wqh", "wqh_older", "wq", "wqf", "wqn"), "wh": ("wh", "wh_older", "w", "wf", "wn")}.items():
    ob, oc, ot, oo = (counts(f"lstm_{x}", 1)["pod"] for x in (base, ceil_, t, c)); nb, nc, nt = (float(np.median(nse_b(f"lstm_{x}", 1))) for x in (base, ceil_, t))
    cl[fam] = dict(onsets=(ot - ob) / (oc - ob), onsets_beyond_control=(ot - oo) / (oc - ob), nse=(nt - nb) / (nc - nb))
rec6b = {c["id"]: c["numbers"] for c in REC1["checks"]}["6b"]
cdev = max(abs(cl["wqh"]["onsets"] - rec6b["share_of_gap_to_wqf_closed_with_gauge"]["onsets"]), abs(cl["wqh"]["onsets_beyond_control"] - rec6b["share_of_gap_to_wqf_closed_with_gauge"]["onsets_beyond_control"]), abs(cl["wqh"]["nse"] - rec6b["share_of_gap_to_wqf_closed_with_gauge"]["nse"]),
           abs(cl["wh"]["onsets"] - rec6b["share_of_gap_closed_without_gauge"]["onsets"]), abs(cl["wh"]["nse"] - rec6b["share_of_gap_closed_without_gauge"]["nse"]))
print(f"    share of the gap to lstm_wqf closed (lead 1): with gauge onsets {100*cl['wqh']['onsets']:.1f}% ({100*cl['wqh']['onsets_beyond_control']:.1f}% beyond the control), NSE {100*cl['wqh']['nse']:.1f}%; without gauge onsets {100*cl['wh']['onsets']:.1f}%, NSE {100*cl['wh']['nse']:.1f}%; max |own - record| {cdev:.1e} (record rounded to 4 decimals)")
# paired onset counts
def flags(name): t = onset_table(name, 1); return np.concatenate([w[h] for g in ids for (_, h, w) in [t[g]]])
FL = {n: flags(f"lstm_{n}") for n in ("wqh", "wqh_older", "wq", "wqn")}
mc = {}
for key, (a, b) in {"wqh_vs_wqh_older": ("wqh", "wqh_older"), "wqh_vs_wq": ("wqh", "wq"), "wqh_vs_wqn": ("wqh", "wqn"), "wqh_older_vs_wq": ("wqh_older", "wq")}.items():
    oa, ob = int((FL[a] & ~FL[b]).sum()), int((~FL[a] & FL[b]).sum()); mc[key] = (oa, ob, float(binomtest(oa, oa + ob, .5).pvalue))
    r = NUM["paired_onsets_mcnemar"][key]; assert (oa, ob) == (r["only_first"], r["only_second"]) and abs(mc[key][2] - r["p"]) < 1e-12, key
    print(f"    onsets warned only by the first / only by the second: {key}: {oa} / {ob}, exact sign-test p {mc[key][2]:.3g}")
print("summary: agrees with the round-1 record:", dict(a=dev < 1e-12, c=ok6c, b_wilcoxon=wdev < 1e-12, b_closure=cdev < 5e-5, paired=True))
jsave("r1_gain_rule.json", dict(per_seed_dev=dev, e1=E1, wilcoxon_dev=wdev, closure=cl, closure_dev=cdev, paired=mc))
