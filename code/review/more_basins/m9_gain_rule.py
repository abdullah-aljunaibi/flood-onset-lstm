"""Check (e): the gain-rule arithmetic, the words, the verdicts, and the fairness of the outcome sentence.

The rule of report section 4.2 (as results/more_basins.json applies it): for NSE, a set T is clearly better than a reference C when every single seed of T is above every single seed of C (median NSE
over the 12 rivers), T is higher in most rivers (7 of 12 or more), and the Wilcoxon signed-rank p over the 12 rivers is below 0.05; clearly worse in the mirror case; "mixed" when only the seed
condition holds; otherwise "no clear difference". For the onsets warned (pooled over the rivers) only the seeds can be compared: clear when every seed of T is above (below) every seed of C.
Recomputed here with the reviewer's own per-river NSE and onset counts (m_common), a brute-force exact Wilcoxon (all 2^12 sign patterns) beside scipy's, and compared field by field with
`gain_rule` and `unseen.against_the_leave_one_out_models` of the results file; the verdicts (GO / CAUTION / NOT YET by the criteria stored in results/flood_analysis.json) recomputed with the warning
level chosen on the validation years (best pooled CSI of the 12 over levels 0.30 .. 1.50 times the high-flow level, ties to the level nearest 1.0); the paired onset counts of the two main pairs
(exact McNemar), and the binomial standard error of a pooled onset share.
Run: python m9_gain_rule.py
"""
import itertools
import re

from scipy.stats import binomtest, wilcoxon

from m_common import *

F = engine()
J = json.loads((PROJ / "results" / "more_basins.json").read_text())
FA = json.loads((PROJ / "results" / "flood_analysis.json").read_text())
ids12 = F.study_ids()
D12 = pd.read_pickle(PROJ / "data" / "derived" / "camels12_daymet.pkl")
thr = own_thresholds(ids12, {g: D12[g].q_mm for g in ids12})
SEEDS = (0, 1, 2)
PAIRS_MAIN = [("lstm63_wq", "lstm_wq"), ("lstm63_w", "lstm_w"), ("lstm63eq_wq", "lstm_wq"), ("lstm63eq_wq", "lstm63_wq")]
PAIRS_UNSEEN = [("unseen50_w_static", "lobo_w_static"), ("unseen50_w_nostatic", "lobo_w_nostatic"), ("unseen50_w_static", "unseen50_w_nostatic"), ("unseen50_w_static", "lobo_rf_w"),
                ("unseen50_w_nostatic", "lobo_rf_w"), ("unseen50_w_static", "lstm63_w")]
ONE_TRAINING = ("lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w")


def seeds_of(n):
    return [n] if n in ONE_TRAINING else [f"{n}_s{s}" for s in SEEDS]


CACHE = {}


def sc(name, k, part="test", trigger=1.0):
    key = (name, k, part, trigger)
    if key not in CACHE:
        CACHE[key] = own_pooled(load_npz(name), thr, k, part, trigger)
    return CACHE[key]


def exact_wilcoxon_p(d):
    """Two-sided exact signed-rank p by enumerating every sign pattern of the ranks of |d| (average ranks for ties); zero differences are dropped (scipy's default)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan")
    ranks = pd.Series(np.abs(d)).rank().values
    w_obs = ranks[d > 0].sum()
    total = ranks.sum()
    cnt_le = cnt_ge = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = float((ranks * np.array(signs)).sum())
        cnt_le += w <= w_obs + 1e-9
        cnt_ge += w >= w_obs - 1e-9
    return float(min(1.0, 2 * min(cnt_le, cnt_ge) / 2 ** n))


def rule(T, C, k):
    a, c = np.array([sc(T, k)["nse"][g] for g in ids12]), np.array([sc(C, k)["nse"][g] for g in ids12])
    sT, sC = [sc(m, k)["nse_median"] for m in seeds_of(T)], [sc(m, k)["nse_median"] for m in seeds_of(C)]
    oT, oC = [sc(m, k)["onset"]["pod"] for m in seeds_of(T)], [sc(m, k)["onset"]["pod"] for m in seeds_of(C)]
    d = a - c
    p_scipy, p_exact = float(wilcoxon(a, c).pvalue), exact_wilcoxon_p(d)
    up, down = min(sT) > max(sC), min(sC) > max(sT)
    n_hi, n_lo = int((d > 0).sum()), int((d < 0).sum())
    small_p = p_scipy < 0.05
    gain, loss = up and n_hi > 6 and small_p, down and n_lo > 6 and small_p
    o_up, o_down = min(oT) > max(oC), min(oC) > max(oT)
    return dict(median_with=float(np.median(a)), median_without=float(np.median(c)), seed_range_with=[min(sT), max(sT)], seed_range_without=[min(sC), max(sC)], n_basins_higher=n_hi, n_basins_lower=n_lo,
                wilcoxon_p=p_scipy, wilcoxon_p_exact_enumeration=p_exact, single_seeds_all_above=up, single_seeds_all_below=down, clear_gain=bool(gain), clear_loss=bool(loss),
                reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference",
                onset_warned_with=sc(T, k)["onset"]["pod"], onset_warned_without=sc(C, k)["onset"]["pod"], onset_seed_range_with=[min(oT), max(oT)], onset_seed_range_without=[min(oC), max(oC)],
                onset_clear_gain=bool(o_up), onset_clear_loss=bool(o_down), onset_reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference",
                onset_false_with=sc(T, k)["onset"]["far"], onset_false_without=sc(C, k)["onset"]["far"])


def compare(mine, js, prefix=""):
    """Largest numeric difference and the list of text/bool fields that differ."""
    diffs, worst = [], 0.0
    num = lambda x, y: abs(x - y)
    pairs = [("median_with", js["nse"]["median_with"]), ("median_without", js["nse"]["median_without"]), ("wilcoxon_p", js["nse"]["wilcoxon_p"]), ("n_basins_higher", js["nse"]["n_basins_higher"]), ("n_basins_lower", js["nse"]["n_basins_lower"]),
             ("onset_warned_with", js["onset"]["warned_with"]), ("onset_warned_without", js["onset"]["warned_without"]), ("onset_false_with", js["onset"]["false_with"]), ("onset_false_without", js["onset"]["false_without"])]
    for key, v in pairs:
        if v is None and mine[key] != mine[key]:
            continue
        worst = max(worst, num(mine[key], v))
    for key, v in (("seed_range_with", js["nse"]["seed_range_with"]), ("seed_range_without", js["nse"]["seed_range_without"]), ("onset_seed_range_with", js["onset"]["seed_range_with"]), ("onset_seed_range_without", js["onset"]["seed_range_without"])):
        worst = max(worst, max(abs(x - y) for x, y in zip(mine[key], v)))
    for key, v in (("single_seeds_all_above", js["nse"]["single_seeds_all_above"]), ("single_seeds_all_below", js["nse"]["single_seeds_all_below"]), ("clear_gain", js["nse"]["clear_gain"]), ("clear_loss", js["nse"]["clear_loss"]),
                   ("reading", js["nse"]["reading"]), ("onset_clear_gain", js["onset"]["clear_gain"]), ("onset_clear_loss", js["onset"]["clear_loss"]), ("onset_reading", js["onset"]["reading"])):
        if mine[key] != v:
            diffs.append((key, mine[key], v))
    return worst, diffs


res = dict(main={}, unseen={})
n_cmp = 0
mx = 0.0
text_diffs = []
print("MAIN PAIRS (same 12 rivers; the rule of report section 4.2)")
for T, C in PAIRS_MAIN:
    key = f"{T}_against_{C}"
    res["main"][key] = {}
    for k in LEADS:
        m = rule(T, C, k)
        w, dif = compare(m, J["gain_rule"][key][f"l{k}"])
        mx, n_cmp = max(mx, w), n_cmp + 1
        text_diffs += [(key, k) + d for d in dif]
        res["main"][key][f"l{k}"] = dict(m, max_abs_numeric_diff_vs_json=w, text_diffs=dif)
        print(f"{key:34s} l{k}: NSE {m['median_without']:.4f} -> {m['median_with']:.4f} seeds {[round(x, 3) for x in m['seed_range_without']]} -> {[round(x, 3) for x in m['seed_range_with']]} | higher in {m['n_basins_higher']}/12, p {m['wilcoxon_p']:.4f} "
              f"(exact enumeration {m['wilcoxon_p_exact_enumeration']:.4f}) -> {m['reading']} | onsets {100 * m['onset_warned_without']:.2f}% {[round(100 * x, 2) for x in m['onset_seed_range_without']]} -> {100 * m['onset_warned_with']:.2f}% "
              f"{[round(100 * x, 2) for x in m['onset_seed_range_with']]} -> {m['onset_reading']}  [json diff {w:.1e}{' ' + str(dif) if dif else ''}]")
print("\nNEVER-SEEN PAIRS (the 12 as rivers never seen)")
for T, C in PAIRS_UNSEEN:
    key = f"{T}_against_{C}"
    res["unseen"][key] = {}
    for k in LEADS:
        m = rule(T, C, k)
        js = J["unseen"]["against_the_leave_one_out_models"][key][f"l{k}"]
        w, dif = compare(m, js)
        mx, n_cmp = max(mx, w), n_cmp + 1
        text_diffs += [(key, k) + d for d in dif]
        res["unseen"][key][f"l{k}"] = dict(m, max_abs_numeric_diff_vs_json=w, text_diffs=dif)
        print(f"{key:44s} l{k}: NSE {m['median_without']:.4f} -> {m['median_with']:.4f} seeds {[round(x, 3) for x in m['seed_range_without']]} -> {[round(x, 3) for x in m['seed_range_with']]} | higher in {m['n_basins_higher']}/12, p {m['wilcoxon_p']:.4f} "
              f"(exact {m['wilcoxon_p_exact_enumeration']:.4f}) -> {m['reading']}  [json diff {w:.1e}{' ' + str(dif) if dif else ''}]")
res["max_abs_numeric_difference_vs_json"] = mx
res["n_pair_lead_comparisons"] = n_cmp
res["text_or_flag_differences"] = text_diffs
res["wilcoxon_scipy_vs_exact_enumeration_max_abs_diff"] = max(abs(v["wilcoxon_p"] - v["wilcoxon_p_exact_enumeration"]) for part in ("main", "unseen") for pair in res[part].values() for v in pair.values())
print(f"\n{n_cmp} pair x lead comparisons: max numeric difference to the results file {mx:.1e}; text/flag differences: {text_diffs}; scipy vs exact enumeration of the Wilcoxon p: max abs diff {res['wilcoxon_scipy_vs_exact_enumeration_max_abs_diff']:.1e}")

# ---- own-level onset rule for the never-seen pairs (the `own_level` block of each pair in the results file) -------------------------------------------------------
def own_level_pod(name, k):
    z = load_npz(name)
    dates, obs, pred = z["dates"], z["obs"], z["pred"][:, :, k - 1]
    tr, sel = window(dates, "train"), window(dates, "test")
    h = m = 0
    for i, g in enumerate(z["basins"]):
        x = pred[i][tr]
        lev = float(np.nanquantile(x[np.isfinite(x)], 0.95))
        o, s = obs[i], pred[i]
        ok = sel & np.isfinite(o) & np.isfinite(s)
        issue = np.full(len(o), np.nan)
        issue[k:] = o[:-k]
        mm = ok & (issue <= thr[g])
        hi, warn = o > thr[g], s > (lev / thr[g]) * thr[g]
        h += int((hi & warn & mm).sum())
        m += int((hi & ~warn & mm).sum())
    return h / (h + m)


own_cmp, own_diffs = [], []
for T, C in PAIRS_UNSEEN:
    for k in LEADS:
        js = J["unseen"]["against_the_leave_one_out_models"][f"{T}_against_{C}"][f"l{k}"]["own_level"]
        a_seeds, c_seeds = [own_level_pod(m, k) for m in seeds_of(T)], [own_level_pod(m, k) for m in seeds_of(C)]
        mine = dict(warned_with=own_level_pod(T, k), warned_without=own_level_pod(C, k), seed_range_with=[min(a_seeds), max(a_seeds)], seed_range_without=[min(c_seeds), max(c_seeds)],
                    single_seeds_all_above=min(a_seeds) > max(c_seeds), single_seeds_all_below=min(c_seeds) > max(a_seeds))
        d = max(abs(mine["warned_with"] - js["warned_with"]), abs(mine["warned_without"] - js["warned_without"]), *[abs(x - y) for x, y in zip(mine["seed_range_with"], js["seed_range_with"])],
                *[abs(x - y) for x, y in zip(mine["seed_range_without"], js["seed_range_without"])])
        own_cmp.append(d)
        if mine["single_seeds_all_above"] != js["single_seeds_all_above"] or mine["single_seeds_all_below"] != js["single_seeds_all_below"]:
            own_diffs.append((T, C, k))
res["own_level_rule"] = dict(n_compared=len(own_cmp), max_abs_numeric_diff=float(max(own_cmp)), flag_differences=own_diffs)
print(f"own-level onset comparisons of the never-seen pairs: {len(own_cmp)} compared, max abs diff {max(own_cmp):.1e}, flag differences {own_diffs}")

# ---- verdicts by the stored criteria ------------------------------------------------------------------------------------------------------------------------
CRIT = FA["verdict"]["criteria"]
POD_GO, FAR_GO = [float(x) for x in re.search(r"pod >= ([0-9.]+) and far <= ([0-9.]+)", CRIT["GO"]).groups()]
CSI_C = float(re.search(r"csi >= ([0-9.]+)", CRIT["CAUTION"]).group(1))
LEVELS = [round(float(x), 2) for x in np.arange(0.30, 1.501, 0.05)]


def verdict(t):
    if t["hits"] + t["misses"] == 0:
        return "NO EVENTS"
    if t["pod"] >= POD_GO and (t["far"] == t["far"]) and t["far"] <= FAR_GO:
        return "GO"
    return "CAUTION" if t["csi"] >= CSI_C else "NOT YET"


def choose(name, k, kind):
    csi = np.array([(lambda t: t["csi"] if t["csi"] == t["csi"] else -1.0)(sc(name, k, "val", m)[kind]) for m in LEVELS])
    best = np.flatnonzero(csi >= csi.max() - 1e-12)
    return LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]


vres, vdiff = {}, []
NEWV = ["lstm63_wq", "lstm63_w", "lstm63eq_wq", "unseen50_w_static", "unseen50_w_nostatic", "lstm_wq", "lstm_w", "lstm_wqf", "lobo_w_static", "lobo_w_nostatic", "lobo_ref_w_nostatic", "lobo_rf_w"]
for n in NEWV:
    vres[n] = {}
    for k in LEADS:
        js = J["verdicts"]["sets"][n][f"l{k}"]
        rec = {}
        for label, kind, trig in (("plain_all", "all", 1.0), ("plain_onset", "onset", 1.0), ("chosen_all_days", "all", None), ("chosen_onsets", "onset", None)):
            if trig is None:
                crit = kind
                tr_ = choose(n, k, crit)
                t = sc(n, k, "test", tr_)[kind]
                js_rec = js["chosen_on_validation_all_days" if label == "chosen_all_days" else "chosen_on_validation_onsets"]
                jt = js_rec[kind]
                ok = js_rec["trigger"] == tr_
            else:
                tr_ = trig
                t = sc(n, k, "test", 1.0)[kind]
                jt = js["plain"][kind]
                ok = True
            v = verdict(t)
            rec[label] = dict(trigger=tr_, verdict=v, pod=t["pod"], far=t["far"], csi=t["csi"], hits=t["hits"], json_verdict=jt["verdict"], json_hits=jt["hits"], same=bool(v == jt["verdict"] and t["hits"] == jt["hits"] and ok))
            if not rec[label]["same"]:
                vdiff.append((n, k, label, v, jt["verdict"]))
        vres[n][f"l{k}"] = rec
res["verdicts"] = dict(criteria=CRIT, n_compared=len(NEWV) * 3 * 4, differences=vdiff, per_set=vres)
print(f"\nverdicts recomputed for {len(NEWV)} sets x 3 leads x 4 readings = {len(NEWV) * 12}: differences from the results file: {vdiff}")
# the controls against results/flood_analysis.json (the stored verdicts of the report): its `services` hold ONE warning level per lead, the level chosen on the validation years for the all-days table,
# and give the onset table at that same level (the level chosen for onsets is a separate entry, at_level_chosen_for_onsets)
svc, mism, n_ctrl = FA["verdict"]["services"], [], 0
for n in ("lstm_wq", "lstm_w"):
    for k in LEADS:
        s_ = svc[n][f"l{k}"]
        trig = choose(n, k, "all")
        if abs(s_["trigger"] - trig) > 1e-12:
            mism.append((n, k, "trigger", trig, s_["trigger"]))
        for kind in ("all", "onset"):
            mine = sc(n, k, "test", trig)[kind]
            n_ctrl += 1
            if verdict(mine) != s_[kind]["pooled"] or abs(mine["pod"] - s_[kind]["pod"]) > 1e-12 or abs(mine["far"] - s_[kind]["far"]) > 1e-12 or abs(mine["csi"] - s_[kind]["csi"]) > 1e-12:
                mism.append((n, k, kind, verdict(mine), s_[kind]["pooled"]))
        mine_on = sc(n, k, "test", choose(n, k, "onset"))["onset"]
        if verdict(mine_on) != s_["onset"]["at_level_chosen_for_onsets"]:
            mism.append((n, k, "onset_at_level_chosen_for_onsets", verdict(mine_on), s_["onset"]["at_level_chosen_for_onsets"]))
res["controls_against_flood_analysis_json"] = dict(n_tables_compared=n_ctrl, mismatches=mism, note="level of the stored table = the level chosen on the validation years for all days, used for both tables of a lead")
print(f"controls against results/flood_analysis.json (services: {n_ctrl} pooled tables of lstm_wq and lstm_w, at the stored warning level): mismatches:", mism)
print("\nverdict of the five new sets, lead 1 (plain level = the observed high-flow level):")
for n in NEWV[:5] + ["lstm_wq", "lstm_w"]:
    r = vres[n]["l1"]
    print(f"  {n:22s} all days: {r['plain_all']['verdict']:8s} (pod {100 * r['plain_all']['pod']:.1f}%, far {100 * r['plain_all']['far']:.1f}%) | onsets: {r['plain_onset']['verdict']:8s} (pod {100 * r['plain_onset']['pod']:.1f}%, far {100 * r['plain_onset']['far']:.1f}%, csi {r['plain_onset']['csi']:.3f}) | "
          f"chosen on validation: all {r['chosen_all_days']['verdict']} @ {r['chosen_all_days']['trigger']}, onsets {r['chosen_onsets']['verdict']} @ {r['chosen_onsets']['trigger']}")

# ---- the paired onset counts of the two main pairs, and the sampling error of a pooled share ------------------------------------------------------------------------
def onset_events(name, k):
    """Boolean hit flags of every onset event (basin, target day) of a prediction set: obs > thr on the target day, observed flow on the issue day at or below thr."""
    z = load_npz(name)
    sel = window(z["dates"], "test")
    ev, hit = [], []
    for i, g in enumerate(z["basins"]):
        o, s = z["obs"][i], z["pred"][i, :, k - 1]
        issue = np.full(len(o), np.nan)
        issue[k:] = o[:-k]
        m = sel & np.isfinite(o) & np.isfinite(s) & (issue <= thr[g]) & (o > thr[g])
        for j in np.flatnonzero(m):
            ev.append((g, j))
            hit.append(bool(s[j] > thr[g]))
    return ev, np.array(hit)


paired = {}
for T, C in (("lstm63_wq", "lstm_wq"), ("lstm63eq_wq", "lstm_wq"), ("lstm63eq_wq", "lstm63_wq"), ("lstm63_w", "lstm_w"), ("unseen50_w_static", "lobo_w_static")):
    e1, h1 = onset_events(T, 1)
    e2, h2 = onset_events(C, 1)
    assert e1 == e2, "the two sets have different onset events"
    b, c = int((h1 & ~h2).sum()), int((~h1 & h2).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else float("nan")
    paired[f"{T}_against_{C}"] = dict(n_onsets=len(e1), hits_T=int(h1.sum()), hits_C=int(h2.sum()), only_T=b, only_C=c, both=int((h1 & h2).sum()), neither=int((~h1 & ~h2).sum()), exact_mcnemar_p=p)
    print(f"paired onsets lead 1, {T} against {C}: T {int(h1.sum())} hits, C {int(h2.sum())}; only T {b}, only C {c}, both {int((h1 & h2).sum())}; exact McNemar p = {p:.3f}")
res["paired_onsets_lead1"] = paired
n_on = paired["lstm63_wq_against_lstm_wq"]["n_onsets"]
p0 = sc("lstm_wq", 1)["onset"]["pod"]
res["binomial_standard_error_of_a_pooled_onset_share_points"] = 100 * float(np.sqrt(p0 * (1 - p0) / n_on))
print(f"binomial standard error of a pooled onset share at {100 * p0:.1f}% of {n_on}: {res['binomial_standard_error_of_a_pooled_onset_share_points']:.2f} points; the onset change 12 -> 63 rivers is {100 * (sc('lstm63_wq', 1)['onset']['pod'] - p0):+.2f} points")
res["all_pass"] = bool(mx < 1e-9 and not text_diffs and not own_diffs and not vdiff and not mism and res["wilcoxon_scipy_vs_exact_enumeration_max_abs_diff"] < 1e-9)
print("\nALL PASS:", res["all_pass"])
save("m9_gain_rule", res)
sys.exit(0 if res["all_pass"] else 1)
