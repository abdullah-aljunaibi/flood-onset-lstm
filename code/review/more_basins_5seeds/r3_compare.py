"""Check s2-s9: the numbers of results/more_basins.json rebuilt from the reviewer's own scores (r2_scores.json) and compared leaf by leaf (relative 1e-9).

s2 summary_test, validation_used_for_no_choice.summary (12 sets x 3 leads)       s6 verdicts (plain, two validation-chosen levels, single seeds' verdicts)
s3 seed_ranges (lo, hi, by_seed), per_river (nse, nse_by_seed, shares warned)    s7 unseen: sets, own level, the 6 pairs against the 05 models, single trainings, losses,
s4 gain_rule (4 pairs x 3 leads; own exact Wilcoxon and McNemar)                    seed averaging, partners, by_river, by_distance, rank correlations
s5 gap_to_the_upper_bound, stopping (from the network records of r1)             s8 the outcome sentence rebuilt word for word; s9 the numbers asked for in the work order
Inputs taken from the results file (not recomputed here): the distances of unseen.by_river (proximity; unchanged from three seeds and reproduced by round 1).
Run: python code/review/more_basins_5seeds/r3_compare.py   (after r1_integrity.py and r2_score.py)
"""
from r_common import *

R = results()
S = load("r2_scores")["files"]
I1 = load("r1_integrity")
ids12 = sorted(load("r2_scores")["thresholds"])
lk = lambda k: f"l{k}"
CUT = lambda t: {c: t[c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}
seedfiles = lambda n: [f"{n}_s{s}" for s in seeds_of(n)] or [n]           # a 05 set is its own single training
med = lambda f, k, part="test": S[f][part][lk(k)]["nse_median"]
nse_arr = lambda f, k: np.array([S[f]["test"][lk(k)]["nse"][g] for g in ids12])
pod = lambda f, k, c="pod": S[f]["test"][lk(k)]["onset"][c]
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x is not None and x == x]) else None
above = lambda a, b: bool(a) and bool(b) and a[0] > b[1]
wil_fallback = []


def wil(a, b, where):
    p = wilcoxon_exact(a, b)
    if p is None:
        from scipy.stats import wilcoxon
        wil_fallback.append(where)
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p = float("nan")
    return p


def paired(a, b, where):
    d = np.asarray(a) - np.asarray(b)
    return dict(n_basins_higher=int((d > 0).sum()), n_basins=len(d), median_nse_difference=float(np.median(d)), wilcoxon_p=wil(a, b, where))


def seed_range(n, k):
    ss = [S[f]["test"][lk(k)] for f in seedfiles(n)]
    by = dict(nse_median=[s["nse_median"] for s in ss], **{f"{kind}_{c}": [s[kind][c] for s in ss] for kind in ("onset", "all") for c in ("pod", "far", "csi")})
    return dict(n_seeds=len(ss), **{key: lo_hi(v) for key, v in by.items()}, by_seed=by)


def paired_onsets(T, C, k):
    vT, vC = S[T]["onset_vector"][lk(k)], S[C]["onset_vector"][lk(k)]
    assert vT["keys"] == vC["keys"]
    a, b = np.array(vT["warned"], bool), np.array(vC["warned"], bool)
    oT, oC = int((a & ~b).sum()), int((~a & b).sum())
    return dict(n_onsets=len(a), both=int((a & b).sum()), only_with=oT, only_without=oC, neither=int((~a & ~b).sum()), mcnemar_p=mcnemar_exact(oT, oC))


def rule(T, C, k):
    a, b = nse_arr(T, k), nse_arr(C, k)
    pr, n_low = paired(a, b, f"{T}/{C}/l{k}"), int((a < b).sum())
    sT, sC = seed_range(T, k), seed_range(C, k)
    up, down = above(sT["nse_median"], sC["nse_median"]), above(sC["nse_median"], sT["nse_median"])
    small = bool(pr["wilcoxon_p"] < P_CUT)
    gain, loss = up and pr["n_basins_higher"] > 6 and small, down and n_low > 6 and small
    o_up, o_down = above(sT["onset_pod"], sC["onset_pod"]), above(sC["onset_pod"], sT["onset_pod"])
    tT, tC = S[T]["test"][lk(k)]["onset"], S[C]["test"][lk(k)]["onset"]
    return dict(against=C, lead=k, seeds_of_the_set=sT["n_seeds"], seeds_of_the_reference=sC["n_seeds"],
                nse=dict(median_with=med(T, k), median_without=med(C, k), gain=med(T, k) - med(C, k), seed_range_with=sT["nse_median"], seed_range_without=sC["nse_median"],
                         single_seeds_all_above=up, single_seeds_all_below=down, n_basins_higher=pr["n_basins_higher"], n_basins_lower=n_low, n_basins=12,
                         higher_in_most_basins=pr["n_basins_higher"] > 6, median_basin_difference=pr["median_nse_difference"], wilcoxon_p=pr["wilcoxon_p"], p_below_cut=small,
                         clear_gain=bool(gain), clear_loss=bool(loss), reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"),
                onset=dict(warned_with=pod(T, k), warned_without=pod(C, k), gain=pod(T, k) - pod(C, k), seed_range_with=sT["onset_pod"], seed_range_without=sC["onset_pod"],
                           false_with=pod(T, k, "far"), false_without=pod(C, k, "far"), single_seeds_all_above=o_up, single_seeds_all_below=o_down, clear_gain=o_up, clear_loss=o_down,
                           reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference", hits_with=tT["hits"], hits_without=tC["hits"], misses_with=tT["misses"],
                           misses_without=tC["misses"], false_alarms_with=tT["false_alarms"], false_alarms_without=tC["false_alarms"], csi_with=tT["csi"], csi_without=tC["csi"],
                           paired=paired_onsets(T, C, k)))


mine = {}
# ---- s2 ---------------------------------------------------------------------------------------------------------------------------------------------------
mine["summary_test"] = {"sets": {n: {lk(k): dict(nse_median=med(n, k), nse_mean=S[n]["test"][lk(k)]["nse_mean"], onset=CUT(S[n]["test"][lk(k)]["onset"]), all=CUT(S[n]["test"][lk(k)]["all"]))
                                     for k in LEADS} for n in ALL_SETS}}
mine["validation_used_for_no_choice"] = {"summary": {n: {lk(k): dict(nse_median=med(n, k, "val"), nse_mean=S[n]["val"][lk(k)]["nse_mean"], onset=CUT(S[n]["val"][lk(k)]["onset"]),
                                                                  all=CUT(S[n]["val"][lk(k)]["all"])) for k in LEADS} for n in ALL_SETS}}
# ---- s3 ---------------------------------------------------------------------------------------------------------------------------------------------------
mine["seed_ranges"] = {"sets": {n: {lk(k): seed_range(n, k) for k in LEADS} for n in ALL_SETS}}
mine["per_river"] = dict(nse={n: {lk(k): S[n]["test"][lk(k)]["nse"] for k in LEADS} for n in ALL_SETS},
                         nse_by_seed={n: {lk(k): {str(i): S[f]["test"][lk(k)]["nse"] for i, f in enumerate(seedfiles(n))} for k in LEADS} for n in NEW},
                         onset_warned={n: {lk(k): S[n]["test"][lk(k)]["onset_pod_by_basin"] for k in LEADS} for n in ALL_SETS},
                         all_days_warned={n: {lk(k): S[n]["test"][lk(k)]["all_pod_by_basin"] for k in LEADS} for n in ALL_SETS})
# ---- s4 ---------------------------------------------------------------------------------------------------------------------------------------------------
PAIRS = [("lstm63_wq", "lstm_wq"), ("lstm63_w", "lstm_w"), ("lstm63eq_wq", "lstm_wq"), ("lstm63eq_wq", "lstm63_wq")]
mine["gain_rule"] = {f"{T}_against_{C}": {lk(k): rule(T, C, k) for k in LEADS} for T, C in PAIRS}
# ---- s5 ---------------------------------------------------------------------------------------------------------------------------------------------------
div = lambda a, b: a / b if b else float("nan")
mine["gap_to_the_upper_bound"] = {"sets": {f"{T}_against_lstm_wq": {lk(k): dict(nse_gain=med(T, k) - med("lstm_wq", k), nse_gap=med("lstm_wqf", k) - med("lstm_wq", k),
                                                                                  nse_share_of_gap_closed=div(med(T, k) - med("lstm_wq", k), med("lstm_wqf", k) - med("lstm_wq", k)),
                                                                                  onset_gain=pod(T, k) - pod("lstm_wq", k), onset_gap=pod("lstm_wqf", k) - pod("lstm_wq", k),
                                                                                  onset_share_of_gap_closed=div(pod(T, k) - pod("lstm_wq", k), pod("lstm_wqf", k) - pod("lstm_wq", k)))
                                                    for k in LEADS} for T in ("lstm63_wq", "lstm63eq_wq")}}
NT = {"lstm63_wq": 63, "lstm63_w": 63, "lstm63eq_wq": 63, "unseen50_w_static": 50, "unseen50_w_nostatic": 50}
stop = {}
for n in NEW:
    rows = I1["i3"]["networks"][n]
    ur, bu = [rows[str(s)]["updates_run"] for s in SEEDS], [rows[str(s)]["best_update"] for s in SEEDS]
    stop[n] = dict(n_train_rivers=NT[n], updates_run=ur, best_update=bu, mean_updates_run=float(np.mean(ur)), mean_best_update=float(np.mean(bu)),
                   sequences_per_river_at_the_best_update=float(np.mean(bu)) * 32 / NT[n], sequences_per_river_in_all_updates_run=float(np.mean(ur)) * 32 / NT[n])
mine["stopping"] = {"sets": stop}
S12 = R["stopping"]["sets"]["lstm_wq"]
mine["stopping"]["ratios"] = {f"{a}_over_{b}": dict(mean_updates_run=stop[a]["mean_updates_run"] / R["stopping"]["sets"][b]["mean_updates_run"],
                                                   mean_best_update=stop[a]["mean_best_update"] / R["stopping"]["sets"][b]["mean_best_update"],
                                                   sequences_per_river_at_the_best_update=stop[a]["sequences_per_river_at_the_best_update"] / R["stopping"]["sets"][b]["sequences_per_river_at_the_best_update"])
                              for a, b in (("lstm63_wq", "lstm_wq"), ("lstm63_w", "lstm_w"), ("lstm63eq_wq", "lstm_wq"))}
# ---- s6 ---------------------------------------------------------------------------------------------------------------------------------------------------
crit = R["verdicts"]["criteria"]
assert crit["GO"] == "pod >= 0.70 and far <= 0.30" and crit["CAUTION"].startswith("csi >= 0.40")
rec = lambda t: dict(verdict=verdict(t), **CUT(t))
mine["verdicts"] = {"sets": {n: {lk(k): dict(plain=dict(trigger=1.0, all=rec(S[n]["test"][lk(k)]["all"]), onset=rec(S[n]["test"][lk(k)]["onset"])),
                                             **{lab: dict(trigger=S[n]["chosen"][lab][lk(k)]["trigger"], all=rec(S[n]["chosen"][lab][lk(k)]["all"]), onset=rec(S[n]["chosen"][lab][lk(k)]["onset"]))
                                                for lab in ("chosen_on_validation_all_days", "chosen_on_validation_onsets")},
                                             single_seeds_plain={kind: {str(i): verdict(S[f]["test"][lk(k)][kind]) for i, f in enumerate(seedfiles(n))} for kind in ("all", "onset")})
                                 for k in LEADS} for n in ALL_SETS}}
# ---- s7 ---------------------------------------------------------------------------------------------------------------------------------------------------
UN_SHOW = ["lstm_w", "lstm63_w", "lobo_ref_w_nostatic", "lobo_w_static", "lobo_w_nostatic", "lobo_rf_w", "unseen50_w_static", "unseen50_w_nostatic"]
own = lambda f, k, kind="onset", c="pod": S[f]["own_level"][lk(k)][kind][c]


def own_seed_range(n, k):
    return {f"{kind}_{c}": lo_hi([own(f, k, kind, c) for f in seedfiles(n)]) for kind in ("onset", "all") for c in ("pod", "far", "csi")}


def own_rule(T, C, k):
    a, b = own_seed_range(T, k), own_seed_range(C, k)
    return dict(warned_with=own(T, k), warned_without=own(C, k), gain=own(T, k) - own(C, k), seed_range_with=a["onset_pod"], seed_range_without=b["onset_pod"],
                false_with=own(T, k, "onset", "far"), false_without=own(C, k, "onset", "far"), single_seeds_all_above=above(a["onset_pod"], b["onset_pod"]),
                single_seeds_all_below=above(b["onset_pod"], a["onset_pod"]))


def single_trainings(T, C, k):
    out = []
    for s, f in zip(SEEDS, seedfiles(T)):
        pr = paired(nse_arr(f, k), nse_arr(C, k), f"{f}/{C}/l{k}")
        out.append(dict(seed=s, median_nse=med(f, k), difference_of_medians=med(f, k) - med(C, k), n_rivers_higher=pr["n_basins_higher"], median_river_difference=pr["median_nse_difference"],
                        wilcoxon_p=pr["wilcoxon_p"]))
    return out


def loss_row(fs, fu, k):
    a, b = nse_arr(fs, k), nse_arr(fu, k)
    d = a - b
    return dict(median_loss=float(np.median(d)), mean_loss=float(d.mean()), largest_loss=float(d.max()), n_rivers_lower_when_never_seen=int((d > 0).sum()), n_rivers=12,
                wilcoxon_p=wil(a, b, f"loss {fs}/{fu}/l{k}"))


U = {}
U["sets"] = {n: dict(trainings=len(seedfiles(n)), nse_median={lk(k): med(n, k) for k in LEADS}, nse_seed_range={lk(k): seed_range(n, k)["nse_median"] for k in LEADS},
                     n_rivers_with_positive_nse={lk(k): int((nse_arr(n, k) > 0).sum()) for k in LEADS},
                     own_level={lk(k): dict(onset=CUT(S[n]["own_level"][lk(k)]["onset"]), all=CUT(S[n]["own_level"][lk(k)]["all"]), seed_ranges=own_seed_range(n, k),
                                            by_seed=dict(onset_pod=[own(f, k) for f in seedfiles(n)], onset_far=[own(f, k, "onset", "far") for f in seedfiles(n)])) for k in LEADS})
             for n in UN_SHOW}
PU = [("unseen50_w_static", "lobo_w_static"), ("unseen50_w_nostatic", "lobo_w_nostatic"), ("unseen50_w_static", "unseen50_w_nostatic"), ("unseen50_w_static", "lobo_rf_w"),
      ("unseen50_w_nostatic", "lobo_rf_w"), ("unseen50_w_static", "lstm63_w")]
U["against_the_leave_one_out_models"] = {f"{T}_against_{C}": {lk(k): dict(rule(T, C, k), own_level=own_rule(T, C, k), single_trainings=single_trainings(T, C, k)) for k in LEADS} for T, C in PU}
LOSS = {"leave_one_out_with_properties__05_recomputed": ("lstm_w_s0", "lobo_w_static"), "leave_one_out_without_properties__05_recomputed": ("lobo_ref_w_nostatic", "lobo_w_nostatic"),
        "50_rivers_with_properties__against_12_river_seed_average": ("lstm_w", "unseen50_w_static"), "50_rivers_with_properties__single_seed_0_against_seed_0": ("lstm_w_s0", "unseen50_w_static_s0"),
        "50_rivers_with_properties__against_63_river_seed_average": ("lstm63_w", "unseen50_w_static"), "50_rivers_without_properties__against_12_river_model": ("lobo_ref_w_nostatic", "unseen50_w_nostatic")}
U["loss_from_not_having_seen_the_river"] = {name: {lk(k): loss_row(a, b, k) for k in LEADS} for name, (a, b) in LOSS.items()}
SA = ["lstm_wq", "lstm_w", "lstm63_wq", "lstm63_w", "unseen50_w_static", "unseen50_w_nostatic", "lstm63eq_wq"]
U["seed_averaging"] = {"sets": {n: {lk(k): dict(seed_average_median=med(n, k), single_trainings=[med(f, k) for f in seedfiles(n)],
                                                mean_of_the_single_trainings=float(np.mean([med(f, k) for f in seedfiles(n)])),
                                                average_minus_mean_of_the_single_trainings=med(n, k) - float(np.mean([med(f, k) for f in seedfiles(n)]))) for k in LEADS} for n in SA}}
lw1 = [med(f, 1) for f in seedfiles("lstm_w")]
U["in_training_partners"] = dict(lstm_w_single_trainings_median_nse_1_day=lw1, lstm_w_seed_0_is_the_best_of_its_seeds=lw1[0] == max(lw1), lstm_w_seed_average_median_nse_1_day=med("lstm_w", 1),
                                 lobo_ref_w_nostatic_median_nse_1_day=med("lobo_ref_w_nostatic", 1), lobo_ref_w_nostatic_trainings=1)
BR = R["unseen"]["by_river"]
U["by_river"] = {g: dict(nse_1_day_ahead={n: S[n]["test"]["l1"]["nse"][g] for n in UN_SHOW}) for g in ids12}
km50, km11 = {g: BR[g]["nearest_training_gauge_km_50"] for g in ids12}, {g: BR[g]["nearest_training_gauge_km_11"] for g in ids12}
bd = {}
for design, km, sets in (("50_training_rivers", km50, ("unseen50_w_static", "unseen50_w_nostatic")), ("11_training_rivers", km11, ("lobo_w_static", "lobo_w_nostatic"))):
    near, far = [g for g in ids12 if km[g] <= 50], [g for g in ids12 if km[g] > 50]
    e = {}
    for label, (a, b) in {"with_properties": ("lstm_w", sets[0]), "without_properties": ("lobo_ref_w_nostatic", sets[1])}.items():
        x = lambda n, g: S[n]["test"]["l1"]["nse"][g]
        e[label] = {grp: dict(n_rivers=len(ids), median_nse_in_training=float(np.median([x(a, g) for g in ids])) if ids else None,
                              median_nse_never_seen=float(np.median([x(b, g) for g in ids])) if ids else None,
                              median_loss=float(np.median([x(a, g) - x(b, g) for g in ids])) if ids else None) for grp, ids in (("within_cutoff", near), ("beyond", far))}
    bd[design] = dict(rivers_with_a_training_gauge_within_the_cutoff=near, rivers_beyond=far, **e)
U["by_distance"] = bd
x1 = lambda n: np.array([S[n]["test"]["l1"]["nse"][g] for g in ids12])
loss63, loss63n = x1("lstm63_w") - x1("unseen50_w_static"), x1("lobo_ref_w_nostatic") - x1("unseen50_w_nostatic")
pd50 = [BR[g]["property_distance_50"] for g in ids12]
U["what_goes_with_the_loss"] = dict(with_properties_against_63_river_model=dict(distance_km=spearman([km50[g] for g in ids12], loss63), property_distance=spearman(pd50, loss63)),
                                    without_properties_against_12_river_model=dict(distance_km=spearman([km50[g] for g in ids12], loss63n), property_distance=spearman(pd50, loss63n)))
gvd = {}
for v_, (T, C) in {"with_properties": ("unseen50_w_static", "lobo_w_static"), "without_properties": ("unseen50_w_nostatic", "lobo_w_nostatic")}.items():
    gain = x1(T) - x1(C)
    k50, k11 = np.array([km50[g] for g in ids12]), np.array([km11[g] for g in ids12])
    nc = [g for g in ids12 if km50[g] >= km11[g] - 1e-9]
    gvd[v_] = dict(spearman_gain_against_km_to_the_nearest_of_the_50=spearman(gain, k50), spearman_gain_against_km_the_50_save_over_the_11=spearman(gain, k11 - k50),
                   rivers_whose_nearest_training_gauge_is_no_closer_with_the_50=nc, median_gain_of_those_rivers=float(np.median(gain[[ids12.index(g) for g in nc]])),
                   median_gain_of_all_12=float(np.median(gain)), gain_by_river={g: float(v) for g, v in zip(ids12, gain)})
U["gain_over_the_05_models_against_distance"] = {"variants": gvd}
mine["unseen"] = U

# ---- compare --------------------------------------------------------------------------------------------------------------------------------------------
by_block, bad = {}, []
for blk, v in mine.items():
    rows = compare(v, R[blk], blk)
    by_block[blk] = dict(entries=len(rows), matching=sum(r[3] for r in rows))
    bad += [r for r in rows if not r[3]]
print("leaf comparisons:", {b: f"{v['matching']}/{v['entries']}" for b, v in by_block.items()})
print("not matching:", len(bad))
for r in bad[:40]:
    print("  ", r)
print("Wilcoxon by scipy (ties or zeros):", wil_fallback)

# ---- s8: the outcome sentence -----------------------------------------------------------------------------------------------------------------------------
n2 = lambda x: f"{x:.2f}"
pc = lambda x: f"{round(100 * x) + 0}%"
A_, B_, C_, D_ = med("lstm_wq", 1), med("lstm63_wq", 1), pod("lstm_wq", 1), pod("lstm63_wq", 1)
E_, F_, E2_, F2_ = med("lobo_w_static", 1), med("unseen50_w_static", 1), med("lobo_w_nostatic", 1), med("unseen50_w_nostatic", 1)
V_ = verdict(S["lstm63_wq"]["test"]["l1"]["onset"])
outcome = (f"Given training on 63 rivers instead of 12, median NSE 1 day ahead for the same 12 rivers goes from {n2(A_)} to {n2(B_)} (onsets warned {pc(C_)} to {pc(D_)}); "
           f"for rivers never seen, from {n2(E_)} to {n2(F_)} with basin properties ({n2(E2_)} to {n2(F2_)} without them; 50 training rivers instead of 11); the verdict is {V_}.")
s8 = dict(outcome_rebuilt=outcome, equal_to_file=outcome == R["outcome"], outcome_full_starts_with_outcome=R["outcome_full"].startswith(R["outcome"] + " "),
          outcome_full_equals_outcome_plus_caveat=R["outcome_full"] == R["outcome"] + " " + R["outcome_caveat"])
print("outcome rebuilt equal:", s8["equal_to_file"], "| outcome_full = outcome + caveat:", s8["outcome_full_equals_outcome_plus_caveat"])

# ---- s9: the numbers of the work order ---------------------------------------------------------------------------------------------------------------------
s9 = dict(same_12={lk(k): dict(lstm_wq=dict(nse=med("lstm_wq", k), onsets=pod("lstm_wq", k), hits=S["lstm_wq"]["test"][lk(k)]["onset"]["hits"]),
                                lstm63_wq=dict(nse=med("lstm63_wq", k), onsets=pod("lstm63_wq", k), hits=S["lstm63_wq"]["test"][lk(k)]["onset"]["hits"]),
                                lstm63eq_wq=dict(nse=med("lstm63eq_wq", k), onsets=pod("lstm63eq_wq", k), hits=S["lstm63eq_wq"]["test"][lk(k)]["onset"]["hits"]),
                                lstm_w=dict(nse=med("lstm_w", k), onsets=pod("lstm_w", k)), lstm63_w=dict(nse=med("lstm63_w", k), onsets=pod("lstm63_w", k))) for k in LEADS},
          never_seen={lk(k): {n: med(n, k) for n in ("lobo_w_static", "unseen50_w_static", "lobo_w_nostatic", "unseen50_w_nostatic")} for k in LEADS},
          verdict_lstm63_wq_l1_onset_plain=V_)
save("r3_compare", dict(by_block=by_block, not_matching=bad, wilcoxon_scipy_fallback=wil_fallback, s8=s8, s9=s9, mine=mine))
