"""Round 2, check 2: the five-seed numbers of results/forecast_input.json against the reviewer's own scores (OUT/v1_scores.pkl), at 1e-9:
summary_test (16 sets, leads 1-3, both splits: fixed level and the levels chosen on the validation years by the CSI rule, replicated), validation_used_for_no_choice.summary,
per_basin, seed_ranges (lo/hi/by_seed/n_seeds), models.*.per_seed and n_seeds_scored, verdicts (plain, two chosen levels, single seeds), gain_rule (12 pairs: NSE rule with
Wilcoxon, onset seed rule), every pair block with with_set/without_set (value_of_the_forecast, forecast_vs_climatology_control, retrained_vs_substituted, ...: onsets, nse,
overlap of warned onsets with the sign test), e2_success, value_of_the_forecast.operating_points and operating_points_by_set (fixed level, validation-chosen level, equal
false-warning budgets, average precision and ROC area; the bootstrap intervals are not recomputed). Output OUT/v2_compare.json."""
from v_common import *
import pickle
from scipy.stats import wilcoxon, binomtest

t0 = time.time()
V = pickle.load(open(OUT / "v1_scores.pkl", "rb"))
SC, ids = V["SC"], V["ids"]
R = jload(RESULTS)
C = Compare(1e-9)
lk = lambda k: f"l{k}"
CUTK = ("hits", "misses", "false_alarms", "pod", "far", "csi")
T = lambda n, k: SC[f"lstm_{n}"]["test"][lk(k)]
Vv = lambda n, k: SC[f"lstm_{n}"]["val"][lk(k)]
seedT = lambda n, s, k: SC[f"lstm_{n}_s{s}"]["test"][lk(k)]


def chosen(n, k, kind):
    i = choose(Vv(n, k)[f"sweep_{kind}"])
    return LEVELS[i], dict(onset=T(n, k)["sweep_onset"][i], all=T(n, k)["sweep_all"][i])


# 1. summary_test / validation / per_basin / verdicts
for n in SHOWN:
    for k in LEADS:
        f, m = R["summary_test"]["sets"][n][lk(k)], T(n, k)
        C.add(f"summary_test {n} {lk(k)} nse_median", f["nse_median"], float(np.median(m["nse"]))); C.add(f"summary_test {n} {lk(k)} nse_mean", f["nse_mean"], float(np.mean(m["nse"])))
        for kind in ("onset", "all"):
            for c in CUTK:
                C.add(f"summary_test {n} {lk(k)} {kind} {c}", f[kind][c], m[kind][c])
        assert LEVELS[14] == 1.0 and m["sweep_onset"][14] == m["onset"] and m["sweep_all"][14] == m["all"]
        vr = R["verdicts"]["sets"][n][lk(k)]
        for kind in ("onset", "all"):
            C.add(f"verdicts {n} {lk(k)} plain {kind} verdict", vr["plain"][kind]["verdict"], verdict(m[kind]))
        for crit, label in (("onset", "chosen_on_validation_onsets"), ("all", "chosen_on_validation_all_days")):
            trig, tabs = chosen(n, k, crit)
            fs = f[f"{crit}_at_validation_chosen_level"]
            C.add(f"summary_test {n} {lk(k)} {crit}_at_validation_chosen_level trigger", fs["trigger"], trig)
            C.add(f"summary_test {n} {lk(k)} {crit}_at_validation_chosen_level verdict", fs["verdict"], verdict(tabs[crit]))
            for c in CUTK:
                C.add(f"summary_test {n} {lk(k)} {crit}_at_validation_chosen_level {c}", fs[c], tabs[crit][c])
            C.add(f"verdicts {n} {lk(k)} {label} trigger", vr[label]["trigger"], trig)
            for kind in ("onset", "all"):
                C.add(f"verdicts {n} {lk(k)} {label} {kind} verdict", vr[label][kind]["verdict"], verdict(tabs[kind]))
                for c in CUTK:
                    C.add(f"verdicts {n} {lk(k)} {label} {kind} {c}", vr[label][kind][c], tabs[kind][c])
        for kind in ("onset", "all"):
            for j, s in enumerate(SEEDS[n]):
                C.add(f"verdicts {n} {lk(k)} single_seeds_plain {kind} {j}", vr["single_seeds_plain"][kind][str(j)], verdict(seedT(n, s, k)[kind]))
            C.add(f"verdicts {n} {lk(k)} single_seeds_plain {kind} count", len(vr["single_seeds_plain"][kind]), len(SEEDS[n]))
        fv, mv = R["validation_used_for_no_choice"]["summary"][n][lk(k)], Vv(n, k)
        C.add(f"validation {n} {lk(k)} nse_median", fv["nse_median"], float(np.median(mv["nse"]))); C.add(f"validation {n} {lk(k)} nse_mean", fv["nse_mean"], float(np.mean(mv["nse"])))
        for kind in ("onset", "all"):
            for c in CUTK:
                C.add(f"validation {n} {lk(k)} {kind} {c}", fv[kind][c], mv[kind][c])
        for i, g in enumerate(ids):
            C.add(f"per_basin nse {n} {lk(k)} {g}", R["per_basin"]["nse"][n][lk(k)][g], m["nse"][i])
            C.add(f"per_basin onset_warned {n} {lk(k)} {g}", R["per_basin"]["onset_warned"][n][lk(k)][g], m["onset_b"][i]["pod"])
            C.add(f"per_basin all_days_warned {n} {lk(k)} {g}", R["per_basin"]["all_days_warned"][n][lk(k)][g], m["all_b"][i]["pod"])


# 2. seed ranges, per_seed, n_seeds_scored
def seedvals(n, k):
    v = {c: [] for c in ("nse_median", "onset_pod", "onset_far", "onset_csi", "all_pod", "all_far", "all_csi")}
    for s in SEEDS[n]:
        e = seedT(n, s, k)
        v["nse_median"].append(float(np.median(e["nse"])))
        for kind in ("onset", "all"):
            for c in ("pod", "far", "csi"):
                v[f"{kind}_{c}"].append(e[kind][c])
    return v


SV = {n: {k: seedvals(n, k) for k in LEADS} for n in SHOWN}
for n in SHOWN:
    C.add(f"models {n} n_seeds_scored", R["models"][n]["n_seeds_scored"], len(SEEDS[n]))
    for k in LEADS:
        rg = R["seed_ranges"]["sets"][n][lk(k)]
        C.add(f"seed_ranges {n} {lk(k)} n_seeds", rg["n_seeds"], len(SEEDS[n]))
        for c, v in SV[n][k].items():
            C.add(f"seed_ranges {n} {lk(k)} {c} lo", rg[c][0], min(v)); C.add(f"seed_ranges {n} {lk(k)} {c} hi", rg[c][1], max(v))
            C.add(f"seed_ranges {n} {lk(k)} by_seed {c}", rg["by_seed"][c], v)
    if n in E2:
        ps = R["models"][n]["per_seed"]
        C.add(f"models {n} per_seed seeds", [p["seed"] for p in ps], SEEDS[n])
        for p in ps:
            s = p["seed"]
            C.add(f"models {n} per_seed s{s} test_nse_median", p["test_nse_median"], [float(np.median(seedT(n, s, k)["nse"])) for k in LEADS])
            C.add(f"models {n} per_seed s{s} test_onset_warned", p["test_onset_warned"], [seedT(n, s, k)["onset"]["pod"] for k in LEADS])
            C.add(f"models {n} per_seed s{s} test_onset_false", p["test_onset_false"], [seedT(n, s, k)["onset"]["far"] for k in LEADS])
            C.add(f"models {n} per_seed s{s} test_all_days_warned", p["test_all_days_warned"], [seedT(n, s, k)["all"]["pod"] for k in LEADS])

# 3. the seed rule and the NSE rule; gain_rule and every pair block
above = lambda a, b: bool(a) and bool(b) and min(a) > max(b)


def my_rule(Tn, Cn, k):
    a, b = T(Tn, k)["nse"], T(Cn, k)["nse"]
    d = a - b
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = float("nan")
    up, down = above(SV[Tn][k]["nse_median"], SV[Cn][k]["nse_median"]), above(SV[Cn][k]["nse_median"], SV[Tn][k]["nse_median"])
    nh, nl = int((d > 0).sum()), int((d < 0).sum())
    small = p < P_CUT
    gain, loss = up and nh > len(ids) / 2 and small, down and nl > len(ids) / 2 and small
    o_up, o_down = above(SV[Tn][k]["onset_pod"], SV[Cn][k]["onset_pod"]), above(SV[Cn][k]["onset_pod"], SV[Tn][k]["onset_pod"])
    return dict(p=p, up=up, down=down, nh=nh, nl=nl, gain=gain, loss=loss, o_up=o_up, o_down=o_down, med_diff=float(np.median(d)),
                nse_reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference",
                onset_reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference")


for key, blk in R["gain_rule"].items():
    Tn, Cn = key.split("_against_")
    for k in LEADS:
        r, f = my_rule(Tn, Cn, k), blk[lk(k)]
        C.add(f"gain_rule {key} {lk(k)} against", f["against"], f"lstm_{Cn}")
        fn, fo = f["nse"], f["onset"]
        for a_, b_ in (("median_with", float(np.median(T(Tn, k)["nse"]))), ("median_without", float(np.median(T(Cn, k)["nse"]))), ("seed_range_with", [min(SV[Tn][k]["nse_median"]), max(SV[Tn][k]["nse_median"])]),
                       ("seed_range_without", [min(SV[Cn][k]["nse_median"]), max(SV[Cn][k]["nse_median"])]), ("single_seeds_all_above", r["up"]), ("single_seeds_all_below", r["down"]), ("n_basins_higher", r["nh"]),
                       ("n_basins_lower", r["nl"]), ("median_basin_difference", r["med_diff"]), ("wilcoxon_p", r["p"]), ("clear_gain", r["gain"]), ("clear_loss", r["loss"]), ("reading", r["nse_reading"])):
            C.add(f"gain_rule {key} {lk(k)} nse {a_}", fn[a_], b_)
        for a_, b_ in (("warned_with", T(Tn, k)["onset"]["pod"]), ("warned_without", T(Cn, k)["onset"]["pod"]), ("false_with", T(Tn, k)["onset"]["far"]), ("false_without", T(Cn, k)["onset"]["far"]),
                       ("seed_range_with", [min(SV[Tn][k]["onset_pod"]), max(SV[Tn][k]["onset_pod"])]), ("seed_range_without", [min(SV[Cn][k]["onset_pod"]), max(SV[Cn][k]["onset_pod"])]),
                       ("single_seeds_all_above", r["o_up"]), ("single_seeds_all_below", r["o_down"]), ("reading", r["onset_reading"])):
            C.add(f"gain_rule {key} {lk(k)} onset {a_}", fo[a_], b_)


def walk(x, path, out):
    if isinstance(x, dict):
        if "with_set" in x and "without_set" in x:
            out.append((path, x))
        for kk, v in x.items():
            walk(v, path + [kk], out)


BLOCKS = []
walk(R, [], BLOCKS)
for path, b in BLOCKS:
    Tn, Cn, k = b["with_set"][5:], b["without_set"][5:], int(path[-1][1])
    key = "/".join(path)
    r = my_rule(Tn, Cn, k)
    tT, tC = T(Tn, k)["onset"], T(Cn, k)["onset"]
    o = b["onsets"]
    for a_, b_ in (("warned_with", tT["pod"]), ("warned_without", tC["pod"]), ("difference_points", 100 * (tT["pod"] - tC["pod"])), ("hits_with", tT["hits"]), ("hits_without", tC["hits"]),
                   ("onsets_in_the_test_years", tT["hits"] + tT["misses"]), ("hits_difference", tT["hits"] - tC["hits"]), ("seed_range_with", [min(SV[Tn][k]["onset_pod"]), max(SV[Tn][k]["onset_pod"])]),
                   ("seed_range_without", [min(SV[Cn][k]["onset_pod"]), max(SV[Cn][k]["onset_pod"])]), ("every_seed_above_every_seed", r["o_up"]), ("every_seed_below_every_seed", r["o_down"]),
                   ("difference_by_seed_points", [100 * (x - y) for x, y in zip(SV[Tn][k]["onset_pod"], SV[Cn][k]["onset_pod"])]), ("false_warnings_with", tT["far"]), ("false_warnings_without", tC["far"]), ("reading", r["onset_reading"])):
        C.add(f"{key} onsets {a_}", o[a_], b_)
    nb = b["nse"]
    for a_, b_ in (("median_with", float(np.median(T(Tn, k)["nse"]))), ("median_without", float(np.median(T(Cn, k)["nse"]))), ("gain", float(np.median(T(Tn, k)["nse"])) - float(np.median(T(Cn, k)["nse"]))),
                   ("every_seed_above_every_seed", r["up"]), ("difference_by_seed", [x - y for x, y in zip(SV[Tn][k]["nse_median"], SV[Cn][k]["nse_median"])]), ("n_basins_higher", r["nh"]),
                   ("median_basin_difference", r["med_diff"]), ("wilcoxon_p", r["p"]), ("reading", r["nse_reading"])):
        C.add(f"{key} nse {a_}", nb[a_], b_)
    ya, ra = T(Tn, k)["rows_y"], T(Tn, k)["rows_r"]; yc, rc = T(Cn, k)["rows_y"], T(Cn, k)["rows_r"]
    assert np.array_equal(ya, yc)
    fa, fc = (ra > 1.0)[ya], (rc > 1.0)[ya]
    ov = b["overlap_of_warned_onsets"]
    oa, oc = int((fa & ~fc).sum()), int((~fa & fc).sum())
    for a_, b_ in (("n_onsets", len(fa)), ("hits_first", int(fa.sum())), ("hits_second", int(fc.sum())), ("warned_by_both", int((fa & fc).sum())), ("warned_only_by_first", oa), ("warned_only_by_second", oc),
                   ("warned_by_neither", int((~fa & ~fc).sum())), ("warned_by_at_least_one", int((fa | fc).sum())), ("sign_test_p", float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan"))):
        C.add(f"{key} overlap {a_}", ov[a_], b_)

# 4. e2_success
E = R["e2_success"]
mine_e2 = {}
for k in LEADS:
    w, a, c = SV["wqaF_sub"][k]["onset_pod"], SV["wqa"][k]["onset_pod"], SV["wqaF_clim"][k]["onset_pod"]
    mine_e2[k] = dict(onset_warned=T("wqaF_sub", k)["onset"]["pod"], onset_warned_wqa=T("wqa", k)["onset"]["pod"], onset_warned_control=T("wqaF_clim", k)["onset"]["pod"],
                      seed_range=[min(w), max(w)], every_seed_above_wqa=above(w, a), every_seed_above_the_control=above(w, c))
    mine_e2[k]["met"] = mine_e2[k]["every_seed_above_wqa"] and mine_e2[k]["every_seed_above_the_control"]
    for a_, b_ in mine_e2[k].items():
        C.add(f"e2_success {lk(k)} {a_}", E["by_lead"][lk(k)][a_], b_)
met23 = mine_e2[2]["met"] and mine_e2[3]["met"]
C.add("e2_success met_at_2_and_3_days", E["met_at_2_and_3_days"], met23)


# 5. operating points
def pod_cap(n, k, cap):
    v = [t["pod"] for t in T(n, k)["sweep_onset"] if t["far"] == t["far"] and t["far"] <= cap]
    return max(v) if v else float("nan")


AP = {}
for n in SHOWN:
    for k in LEADS:
        m = T(n, k)
        AP[(n, k)] = (own_ap(m["rows_y"], m["rows_r"]), own_auc(m["rows_y"], m["rows_r"]))
        f = R["operating_points_by_set"]["sets"][n][lk(k)]
        C.add(f"operating_points_by_set {n} {lk(k)} arm", f["arm"], ARM[n])
        for c in CUTK:
            C.add(f"operating_points_by_set {n} {lk(k)} fixed_level {c}", f["fixed_level"][c], m["onset"][c])
        trig, tabs = chosen(n, k, "onset")
        C.add(f"operating_points_by_set {n} {lk(k)} chosen trigger", f["level_chosen_on_the_validation_years"]["trigger"], trig)
        C.add(f"operating_points_by_set {n} {lk(k)} chosen verdict", f["level_chosen_on_the_validation_years"]["verdict"], verdict(tabs["onset"]))
        for c in CUTK:
            C.add(f"operating_points_by_set {n} {lk(k)} chosen {c}", f["level_chosen_on_the_validation_years"][c], tabs["onset"][c])
        C.add(f"operating_points_by_set {n} {lk(k)} at_most_40", f["onsets_warned_with_at_most_40_percent_false"], pod_cap(n, k, 0.40))
        C.add(f"operating_points_by_set {n} {lk(k)} at_most_50", f["onsets_warned_with_at_most_50_percent_false"], pod_cap(n, k, 0.50))
        C.add(f"operating_points_by_set {n} {lk(k)} average_precision", f["average_precision"], AP[(n, k)][0], tol=1e-12)
        C.add(f"operating_points_by_set {n} {lk(k)} roc_area", f["roc_area"], AP[(n, k)][1], tol=1e-12)
        C.add(f"operating_points_by_set {n} {lk(k)} n_onsets", f["n_onsets"], int(m["rows_y"].sum())); C.add(f"operating_points_by_set {n} {lk(k)} n_rows", f["n_rows"], len(m["rows_y"]))
VOP = R["value_of_the_forecast"]["operating_points"]
for k in LEADS:
    f, Tn, Cn = VOP[lk(k)], "wqaF_sub", "wqa"
    pT, pC = T(Tn, k)["onset"], T(Cn, k)["onset"]
    fx = f["fixed_level"]
    for a_, b_ in (("points", 100 * (pT["pod"] - pC["pod"])), ("onsets_warned_with", pT["pod"]), ("onsets_warned_without", pC["pod"]), ("false_with", pT["far"]), ("false_without", pC["far"]), ("csi_with", pT["csi"]), ("csi_without", pC["csi"])):
        C.add(f"value_of_the_forecast operating_points {lk(k)} fixed_level {a_}", fx[a_], b_)
    (tw, sw), (to, so) = chosen(Tn, k, "onset"), chosen(Cn, k, "onset")
    lc = f["level_chosen_on_the_validation_years"]
    for a_, b_ in (("trigger_with", tw), ("trigger_without", to), ("points", 100 * (sw["onset"]["pod"] - so["onset"]["pod"])), ("onsets_warned_with", sw["onset"]["pod"]), ("onsets_warned_without", so["onset"]["pod"]),
                   ("false_with", sw["onset"]["far"]), ("false_without", so["onset"]["far"]), ("csi_with", sw["onset"]["csi"]), ("csi_without", so["onset"]["csi"])):
        C.add(f"value_of_the_forecast operating_points {lk(k)} level_chosen {a_}", lc[a_], b_)
    eb, ref = f["equal_false_warning_budgets"], pC["far"]
    for a_, b_ in (("points_at_most_40", 100 * (pod_cap(Tn, k, .4) - pod_cap(Cn, k, .4))), ("points_at_most_50", 100 * (pod_cap(Tn, k, .5) - pod_cap(Cn, k, .5))), ("reference_false_share", ref),
                   ("points_at_the_reference_false_share", 100 * (pod_cap(Tn, k, ref) - pod_cap(Cn, k, ref))), ("onsets_warned_with_at_most_40", pod_cap(Tn, k, .4)), ("onsets_warned_without_at_most_40", pod_cap(Cn, k, .4)),
                   ("onsets_warned_with_at_most_50", pod_cap(Tn, k, .5)), ("onsets_warned_without_at_most_50", pod_cap(Cn, k, .5))):
        C.add(f"value_of_the_forecast operating_points {lk(k)} equal_budgets {a_}", eb[a_], b_)
    tf = f["trigger_free"]
    for a_, b_ in (("average_precision_with", AP[(Tn, k)][0]), ("average_precision_without", AP[(Cn, k)][0]), ("roc_area_with", AP[(Tn, k)][1]), ("roc_area_without", AP[(Cn, k)][1]),
                   ("average_precision_difference", AP[(Tn, k)][0] - AP[(Cn, k)][0]), ("n_onsets", int(T(Tn, k)["rows_y"].sum()))):
        C.add(f"value_of_the_forecast operating_points {lk(k)} trigger_free {a_}", tf[a_], b_, tol=1e-12)
    bb = f["block_bootstrap_fixed_level"]
    C.add(f"value_of_the_forecast operating_points {lk(k)} bootstrap pod_points (point estimate)", bb["pod_points"], 100 * (pT["pod"] - pC["pod"]))
    C.add(f"value_of_the_forecast operating_points {lk(k)} bootstrap nse (point estimate)", bb["nse"], float(np.median(T(Tn, k)["nse"])) - float(np.median(T(Cn, k)["nse"])), tol=1e-12)

S = C.summary()
by_block = {}
for kk, v in C.rows.items():
    b = kk.split(" ")[0].split("/")[0]
    by_block.setdefault(b, [0, 0]); by_block[b][0] += 1; by_block[b][1] += v["ok"]
headline = {n: {lk(k): dict(nse_median=float(np.median(T(n, k)["nse"])), onset_hits=T(n, k)["onset"]["hits"], onsets=T(n, k)["onset"]["hits"] + T(n, k)["onset"]["misses"], pod=T(n, k)["onset"]["pod"], far=T(n, k)["onset"]["far"],
                            seed_range_pod=[min(SV[n][k]["onset_pod"]), max(SV[n][k]["onset_pod"])], seed_range_nse=[min(SV[n][k]["nse_median"]), max(SV[n][k]["nse_median"])],
                            chosen_trigger=chosen(n, k, "onset")[0], chosen_pod=chosen(n, k, "onset")[1]["onset"]["pod"], chosen_far=chosen(n, k, "onset")[1]["onset"]["far"]) for k in LEADS} for n in SHOWN}
jsave("v2_compare.json", dict(summary=S, by_block={b: dict(entries=v[0], matching=v[1]) for b, v in by_block.items()}, not_matching={k: C.rows[k] for k in S["not_matching"]}, n_pair_blocks=len(BLOCKS),
                              e2_success_mine={lk(k): v for k, v in mine_e2.items()}, met_at_2_and_3_days=met23, headline=headline, seconds=round(time.time() - t0, 1)))
print(S["entries"], "compared,", S["matching"], "matching;", len(BLOCKS), "pair blocks")
for b, v in by_block.items():
    print(f"  {b}: {v[1]}/{v[0]}")
for kk in S["not_matching"][:40]:
    print("MISMATCH", kk, C.rows[kk]["file"], C.rows[kk]["mine"])
print("e2_success mine:", {k: v["met"] for k, v in mine_e2.items()}, "met_at_2_and_3_days", met23)
