"""E7 review, check 2: the headline numbers recomputed from results/preds with the reviewer's own scoring (numpy: no flood.evaluate, no warning_policy, nothing of
script 28), for the three new sets, the five stored LSTM sets and the four simple models, seed means and every single seed: pooled onset and all-day warning tables at
trigger 1.0, median NSE, leads 1-3, test and validation years, per basin; seed ranges; the ROADMAP E7 rule block by block (own exact Wilcoxon p by enumeration beside
scipy's); the seven pair readings (NSE rule with Wilcoxon p, onset seed rule) and the paired onsets with the sign test; the CSI and 30%-false triggers chosen on the
validation years (code/warning_policy.choose_trigger's documented rule replicated) with test scores, single seeds and verdicts; the verdict readings at the level grid
0.30-1.50; the resampling intervals (13_uncertainty.py's documented recipe, own code, plus a direct route on the first resamples); the seed means = mean of the seed
files; the stored sets against their own results files; flood.evaluate as a second route. Also descriptive extras the findings use (not in the results file): the
controls at the 30%-false point, a level-matched comparison, the forecast level. About 1-2 minutes."""
import itertools

from scipy.stats import binomtest, wilcoxon

from m_common import *
import flood as F

t0 = time.time()
R = jload(RESULTS)
C = Compare(tol=1e-9)
ids = list(R["thresholds"])
CRIT = jload("results/flood_analysis.json")["verdict"]["criteria"]
C.add("criteria GO text", CRIT["GO"], "pod >= 0.70 and far <= 0.30")
POD_GO, FAR_GO, CSI_CAUTION = 0.70, 0.30, 0.40
C.add("n_seeds_per_set", R["n_seeds_per_set"], {n: len(SEEDS[n]) for n in LSTM})

# ---- data ------------------------------------------------------------------------------------------------------------------------------------------------
dates, basins, OBS, _, _ = load_npz("lstm_wqME")
assert basins == ids and dates.equals(DAYS)
TR, VA, TE = (within(dates, *SPLIT[p]) for p in ("train", "val", "test"))
THR = np.array([float(np.quantile(OBS[i][TR], 0.95)) for i in range(len(ids))])
for i, g in enumerate(ids):
    C.add(f"threshold {g}", R["thresholds"][g], THR[i])
PRED, META = {}, {}
for n in LSTM:
    for name in [f"lstm_{n}"] + [f"lstm_{n}_s{s}" for s in SEEDS[n]]:
        d, b, o, p, m = load_npz(name)
        assert b == ids and d.equals(dates) and np.array_equal(o, OBS, equal_nan=True), name
        PRED[name], META[name] = p, m
    # no further seed files on disk than the ones read
    extra = [s for s in range(5, 10) if (PROJ / f"results/preds/lstm_{n}_s{s}.npz").exists()] + [s for s in range(5) if s not in SEEDS[n] and (PROJ / f"results/preds/lstm_{n}_s{s}.npz").exists()]
    C.add(f"{n}: no other seed files on disk", [], extra)
for n in SIMPLE:
    d, b, o, p, m = load_npz(n)
    assert b == ids and d.equals(dates) and np.array_equal(o, OBS, equal_nan=True)
    PRED[n], META[n] = p, m
name_of = lambda n: n if n.startswith("base_") else f"lstm_{n}"

# seed means = the mean of the seed files
seed_mean = {}
for n in LSTM:
    S = np.stack([PRED[f"lstm_{n}_s{s}"] for s in SEEDS[n]])
    M = S.mean(0)
    P = PRED[f"lstm_{n}"]
    same_nan = bool(np.array_equal(np.isnan(M), np.isnan(P)))
    seed_mean[n] = dict(same_nan_pattern=same_nan, max_abs_diff=float(np.nanmax(np.abs(M - P))), n_values=int(np.isfinite(P).sum()), n_seeds=len(SEEDS[n]))
    C.add(f"{n}: seed-mean file = mean of the {len(SEEDS[n])} seed files (1e-12 mm/day)", True, same_nan and seed_mean[n]["max_abs_diff"] < 1e-12)

PART = {"train": TR, "val": VA, "test": TE}
n_part_days = {"val": int(VA.sum()), "test": int(TE.sum())}


def rates(T):
    h, m, f = T["hits"], T["misses"], T["false_alarms"]
    q = lambda a, b: a / b if b else float("nan")
    return dict(T, pod=q(h, h + m), far=q(f, h + f), csi=q(h, h + m + f))


def score(p, part="test", trigger=1.0, form="product"):
    """Per lead: per-basin NSE, pooled onset / all-day tables, per-basin onset POD and FAR, onset flags; warning = forecast > trigger x Q95 (form 'product') or
    forecast / Q95 > trigger (form 'ratio', the form of code/warning_policy.py)."""
    mask = PART[part]
    out = {}
    for j, k in enumerate(LEADS):
        A = dict(hits=0, misses=0, false_alarms=0, correct_neg=0)
        O = dict(A)
        nse, pod_b, far_b, cov, flags = [], [], [], [], []
        for i in range(len(ids)):
            o, f = OBS[i], p[i, :, j]
            prev = np.r_[np.full(k, np.nan), o[:-k]]
            ok = mask & np.isfinite(o) & np.isfinite(f)
            cov.append(ok.sum() / n_part_days.get(part, ok.sum()))
            oo, ff, pp = o[ok], f[ok], prev[ok]
            nse.append(1.0 - ((ff - oo) ** 2).sum() / ((oo - oo.mean()) ** 2).sum())
            hi = oo > THR[i]
            wa = ff > trigger * THR[i] if form == "product" else ff / THR[i] > trigger
            low = pp <= THR[i]
            for T, sel in ((A, np.ones(len(oo), bool)), (O, low)):
                T["hits"] += int((hi & wa & sel).sum()); T["misses"] += int((hi & ~wa & sel).sum())
                T["false_alarms"] += int((~hi & wa & sel).sum()); T["correct_neg"] += int((~hi & ~wa & sel).sum())
            h_, m_, f_ = int((hi & wa & low).sum()), int((hi & ~wa & low).sum()), int((~hi & wa & low).sum())
            pod_b.append(h_ / (h_ + m_) if h_ + m_ else float("nan")); far_b.append(f_ / (h_ + f_) if h_ + f_ else float("nan"))
            flags.append(wa[hi & low])
        out[lk(k)] = dict(nse=np.array(nse), nse_median=float(np.median(nse)), nse_mean=float(np.mean(nse)), onset=rates(O), all=rates(A), pod_b=np.array(pod_b), far_b=np.array(far_b),
                          coverage_min=float(min(cov)), flags=np.concatenate(flags))
    return out


EV = {name: score(PRED[name]) for name in PRED}
EVV = {name: score(PRED[name], "val") for name in PRED}
C.add("coverage 1 on every test and validation day for every set", True, all(EV[x][lk(k)]["coverage_min"] == 1 and EVV[x][lk(k)]["coverage_min"] == 1 for x in PRED for k in LEADS))

# ---- 1. summary_test, summary_val, per_basin ----------------------------------------------------------------------------------------------------------------------
for n in SHOWN:
    for k in LEADS:
        for part, E, Fs in (("test", EV, R["summary_test"]["sets"]), ("val", EVV, R["validation_used_for_no_choice"]["summary"])):
            m, f = E[name_of(n)][lk(k)], Fs[n][lk(k)]
            C.add(f"{n} {lk(k)} {part} nse_median", f["nse_median"], m["nse_median"]); C.add(f"{n} {lk(k)} {part} nse_mean", f["nse_mean"], m["nse_mean"])
            for kind in ("onset", "all"):
                for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    C.add(f"{n} {lk(k)} {part} {kind} {c}", f[kind][c], m[kind][c])
        m = EV[name_of(n)][lk(k)]
        for i, g in enumerate(ids):
            C.add(f"{n} {lk(k)} per_basin nse {g}", R["per_basin"]["nse"][n][lk(k)][g], m["nse"][i])
            C.add(f"{n} {lk(k)} per_basin onset_warned {g}", R["per_basin"]["onset_warned"][n][lk(k)][g], m["pod_b"][i])
            C.add(f"{n} {lk(k)} per_basin onset_false {g}", R["per_basin"]["onset_false"][n][lk(k)][g], m["far_b"][i])

# ---- 2. seed ranges and every single seed -------------------------------------------------------------------------------------------------------------------------
lo_hi = lambda v: [float(min(w)), float(max(w))] if (w := [x for x in v if x == x]) else None
RANGE = {}
for n in LSTM:
    RANGE[n] = {}
    for k in LEADS:
        S = [EV[f"lstm_{n}_s{s}"][lk(k)] for s in SEEDS[n]]
        by = dict(seeds=SEEDS[n], nse_median=[s["nse_median"] for s in S], **{f"{kind}_{c}": [s[kind][c] for s in S] for kind in ("onset", "all") for c in ("pod", "far", "csi")})
        RANGE[n][lk(k)] = dict(by_seed=by, **{key: lo_hi(v) for key, v in by.items() if key != "seeds"})
        f = R["seed_ranges"]["sets"][n][lk(k)]
        C.add(f"{n} {lk(k)} seed_ranges n_seeds", f["n_seeds"], len(SEEDS[n]))
        C.add(f"{n} {lk(k)} seed_ranges seeds", f["by_seed"]["seeds"], SEEDS[n])
        for key in ("nse_median", "onset_pod", "onset_far", "onset_csi", "all_pod", "all_far", "all_csi"):
            for i_, (x, y) in enumerate(zip(f["by_seed"][key], by[key])):
                C.add(f"{n} {lk(k)} seed {SEEDS[n][i_]} {key}", x, y)
            C.add(f"{n} {lk(k)} seed_range {key} lo", f[key][0], RANGE[n][lk(k)][key][0]); C.add(f"{n} {lk(k)} seed_range {key} hi", f[key][1], RANGE[n][lk(k)][key][1])
for n in NEW:
    for r in R["models"][n]["per_seed"]:
        s = r["seed"]
        for j, k in enumerate(LEADS):
            e = EV[f"lstm_{n}_s{s}"][lk(k)]
            for key, val in (("test_nse_median", e["nse_median"]), ("test_onset_warned", e["onset"]["pod"]), ("test_onset_false", e["onset"]["far"]), ("test_all_days_warned", e["all"]["pod"])):
                C.add(f"{n} models per_seed s{s} {lk(k)} {key}", r[key][j], val)

# ---- 3. second route: flood.evaluate on the seed means and the simple models ------------------------------------------------------------------------------------
thr_d = {g: float(THR[i]) for i, g in enumerate(ids)}
frames = lambda name: {g: pd.DataFrame({"obs": OBS[i], **{lk(k): PRED[name][i, :, j] for j, k in enumerate(LEADS)}}, index=dates) for i, g in enumerate(ids)}
second = {}
for n in ["wqME", "wqME_shuf", "wqME_older", "wqa", "wqnh"] + SIMPLE:
    E2 = F.evaluate(frames(name_of(n)), thr_d)
    for k in LEADS:
        s2, m = E2["summary"][lk(k)], EV[name_of(n)][lk(k)]
        ok = close(s2["nse_median"], m["nse_median"], 1e-12) and all(s2["onset"][c] == m["onset"][c] for c in ("hits", "misses", "false_alarms")) and all(s2["all"][c] == m["all"][c] for c in ("hits", "misses", "false_alarms"))
        second[f"{n} {lk(k)}"] = ok
C.add("flood.evaluate (second route) = my scoring for 9 sets x 3 leads", True, all(second.values()))

# ---- 4. the E7 rule, block by block ------------------------------------------------------------------------------------------------------------------------------
E7 = R["e7_rule"]
r1 = {n: RANGE[n]["l1"]["by_seed"]["onset_pod"] for n in ("wqME", "wqa", "wqME_shuf", "wqME_older")}
above = min(r1["wqME"]) > max(r1["wqa"])
fME, fA = EV["lstm_wqME"]["l1"]["onset"]["far"], EV["lstm_wqa"]["l1"]["onset"]["far"]
a_b, b_b = EV["lstm_wqME"]["l1"]["pod_b"], EV["lstm_wqa"]["l1"]["pod_b"]
d = a_b - b_b
nz = d[d != 0]


def exact_two_sided(nz):
    """Exact two-sided p of the Wilcoxon signed-rank statistic by enumerating every sign assignment (ranks of |d|, average ranks for ties): P(T+ as extreme)."""
    ranks = pd.Series(np.abs(nz)).rank().values
    tplus = ranks[nz > 0].sum()
    tot = ranks.sum()
    stats = np.array([sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product((0, 1), repeat=len(ranks))])
    lo = (stats <= min(tplus, tot - tplus) + 1e-9).mean()
    return float(min(1.0, 2 * lo)), float(tplus)


p_enum, tplus = exact_two_sided(nz)
with np.errstate(all="ignore"):
    p_auto = float(wilcoxon(a_b, b_b).pvalue)
    p_appr = float(wilcoxon(a_b, b_b, method="approx").pvalue)
W = E7["block_1_gain"]["wilcoxon_over_the_basins"]
C.add("block 1 every seed of wqME above every seed of wqa", E7["block_1_gain"]["every_seed_above"]["holds"], above)
C.add("block 1 wqME single seeds", E7["block_1_gain"]["every_seed_above"]["wqME_single_seeds"]["onset_warned"], r1["wqME"])
C.add("block 1 wqa single seeds", E7["block_1_gain"]["every_seed_above"]["wqa_single_seeds"]["onset_warned"], r1["wqa"])
C.add("block 1 false share not higher", E7["block_1_gain"]["false_share_not_higher"]["holds"], fME <= fA)
C.add("block 1 false share wqME", E7["block_1_gain"]["false_share_not_higher"]["wqME_seed_averaged"], fME)
C.add("block 1 false share wqa", E7["block_1_gain"]["false_share_not_higher"]["wqa_seed_averaged"], fA)
C.add("block 1 Wilcoxon n_used", W["n_used"], int(len(nz))); C.add("block 1 Wilcoxon zero differences", W["n_zero_differences"], int((d == 0).sum()))
C.add("block 1 Wilcoxon basins higher / lower", [W["n_basins_higher"], W["n_basins_lower"]], [int((d > 0).sum()), int((d < 0).sum())])
C.add("block 1 Wilcoxon median difference points", W["median_difference_points"], float(100 * np.median(d)))
for i, g in enumerate(ids):
    C.add(f"block 1 per-basin difference {g}", W["per_basin_difference_points"][g], float(100 * d[i]))
C.add("block 1 Wilcoxon p (scipy default)", W["p_value"], p_auto); C.add("block 1 Wilcoxon p (my exact enumeration) = file exact p", W["p_value_exact"], p_enum)
C.add("block 1 Wilcoxon p normal approximation", W["p_value_normal_approximation"], p_appr)
C.add("block 1 Wilcoxon holds (p < 0.05)", W["holds"], p_auto < 0.05)
b1 = bool(above and fME <= fA and p_auto < 0.05)
C.add("block 1 holds", E7["block_1_gain"]["holds"], b1)
blocks = {}
for key, n in (("block_2_capacity_control_flat", "wqME_shuf"), ("block_3_clock_control_flat", "wqME_older")):
    med_ = float(np.median(r1[n]))
    inside = min(r1["wqa"]) <= med_ <= max(r1["wqa"])
    blocks[key] = dict(median=med_, inside=inside)
    C.add(f"{key} median of the single seeds", E7[key]["median_of_the_single_seeds"], med_)
    C.add(f"{key} wqa range", E7[key]["wqa_lowest_highest_single_seed"], [min(r1["wqa"]), max(r1["wqa"])])
    C.add(f"{key} holds", E7[key]["holds"], inside)
    C.add(f"{key} seed-averaged onsets warned", E7[key]["seed_averaged_onsets_warned"], EV[f"lstm_{n}"]["l1"]["onset"]["pod"])
    C.add(f"{key} seed-averaged onset false", E7[key]["seed_averaged_onset_false"], EV[f"lstm_{n}"]["l1"]["onset"]["far"])
met = bool(b1 and blocks["block_2_capacity_control_flat"]["inside"] and blocks["block_3_clock_control_flat"]["inside"])
C.add("E7 rule met", E7["met"], met)
C.add("E7 reading", E7["reading"], "a clear gain" if met else "not a clear gain")
gains = {}
for kind, (T, Cc) in {"lstm": ("lstm_wqME", "lstm_wqa"), "ridge regression": ("base_linreg_wqME", "base_linreg_wqa"), "random forest": ("base_rf_wqME", "base_rf_wqa")}.items():
    for k in LEADS:
        ET, EC = EV[T][lk(k)], EV[Cc][lk(k)]
        f = E7["block_4_network_yardstick"]["by_model"][kind][lk(k)]
        C.add(f"block 4 {kind} {lk(k)} onset gain points", f["onset_gain_points"], 100 * (ET["onset"]["pod"] - EC["onset"]["pod"]))
        C.add(f"block 4 {kind} {lk(k)} hits with/without", [f["hits_with"], f["hits_without"]], [ET["onset"]["hits"], EC["onset"]["hits"]])
        C.add(f"block 4 {kind} {lk(k)} nse gain", f["nse_gain"], ET["nse_median"] - EC["nse_median"])
    gains[kind] = 100 * (EV[T]["l1"]["onset"]["pod"] - EV[Cc]["l1"]["onset"]["pod"])
y_read = ("the LSTM does not gain onsets warned from the wqME columns over lstm_wqa, so there is no LSTM gain for the simple models to match" if not gains["lstm"] > 0 else
          "the information, not the network: ridge and forest each gain at least as much from the wqME columns as the LSTM" if gains["ridge regression"] >= gains["lstm"] and gains["random forest"] >= gains["lstm"] else
          "the LSTM gains more from the wqME columns than ridge and forest" if gains["ridge regression"] < gains["lstm"] and gains["random forest"] < gains["lstm"] else "mixed")
C.add("block 4 reading", E7["block_4_network_yardstick"]["reading"], y_read)

# ---- 5. the seven pair readings and the paired onsets ------------------------------------------------------------------------------------------------------------
PAIRS = {"several_products_against_aorc_alone": ("wqME", "wqa"), "several_products_against_the_capacity_control": ("wqME", "wqME_shuf"),
         "several_products_against_the_clock_control": ("wqME", "wqME_older"), "capacity_control_against_aorc_alone": ("wqME_shuf", "wqa"),
         "clock_control_against_aorc_alone": ("wqME_older", "wqa"), "several_products_against_nldas_and_era5": ("wqME", "wqnh"), "several_products_against_the_rain_table": ("wqME", "wq")}
C.add("pair names", sorted(R["readings"]), sorted(PAIRS))
MYPAIRS = {}
for key, (T, Cc) in PAIRS.items():
    MYPAIRS[key] = {}
    for k in LEADS:
        ET, EC, sT, sC = EV[f"lstm_{T}"][lk(k)], EV[f"lstm_{Cc}"][lk(k)], RANGE[T][lk(k)], RANGE[Cc][lk(k)]
        f = R["readings"][key]["by_lead"][lk(k)]
        fo, fn = f["onsets"], f["nse"]
        up_o, down_o = sT["onset_pod"][0] > sC["onset_pod"][1], sC["onset_pod"][0] > sT["onset_pod"][1]
        C.add(f"{key} {lk(k)} onsets warned with/without, hits", [fo["warned_with"], fo["warned_without"], fo["hits_with"], fo["hits_without"], fo["onsets_in_the_test_years"]],
              [ET["onset"]["pod"], EC["onset"]["pod"], ET["onset"]["hits"], EC["onset"]["hits"], ET["onset"]["hits"] + ET["onset"]["misses"]])
        C.add(f"{key} {lk(k)} difference points", fo["difference_points"], 100 * (ET["onset"]["pod"] - EC["onset"]["pod"]))
        C.add(f"{key} {lk(k)} false with/without", [fo["false_warnings_with"], fo["false_warnings_without"]], [ET["onset"]["far"], EC["onset"]["far"]])
        C.add(f"{key} {lk(k)} onset every seed above/below", [fo["every_seed_above_every_seed"], fo["every_seed_below_every_seed"]], [bool(up_o), bool(down_o)])
        C.add(f"{key} {lk(k)} onset reading", fo["reading"], "clear gain" if up_o else "clear loss" if down_o else "no clear difference")
        pairs_ = [s for s in sT["by_seed"]["seeds"] if s in sC["by_seed"]["seeds"]]
        C.add(f"{key} {lk(k)} seeds paired", fo["seeds_paired"], pairs_)
        dsp = [100 * (sT["by_seed"]["onset_pod"][sT["by_seed"]["seeds"].index(s)] - sC["by_seed"]["onset_pod"][sC["by_seed"]["seeds"].index(s)]) for s in pairs_]
        for s, x, y in zip(pairs_, fo["difference_by_seed_points"], dsp):
            C.add(f"{key} {lk(k)} difference seed {s}", x, y)
        nT, nC = ET["nse"], EC["nse"]
        try:
            pw = float(wilcoxon(nT, nC).pvalue)
        except ValueError:
            pw = float("nan")
        higher, lower = int((nT > nC).sum()), int((nT < nC).sum())
        up, down = sT["nse_median"][0] > sC["nse_median"][1], sC["nse_median"][0] > sT["nse_median"][1]
        gain, loss = up and higher > 6 and pw < 0.05, down and lower > 6 and pw < 0.05
        rd = "clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference"
        C.add(f"{key} {lk(k)} nse median with/without", [fn["median_with"], fn["median_without"]], [ET["nse_median"], EC["nse_median"]])
        C.add(f"{key} {lk(k)} nse basins higher", fn["n_basins_higher"], higher)
        C.add(f"{key} {lk(k)} nse median basin difference", fn["median_basin_difference"], float(np.median(nT - nC)))
        C.add(f"{key} {lk(k)} nse wilcoxon p", fn["wilcoxon_p"], pw)
        C.add(f"{key} {lk(k)} nse reading", fn["reading"], rd)
        # paired onsets
        fa, fc = ET["flags"], EC["flags"]
        both, oa, oc = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum())
        sp = float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan")
        ov = R["readings"][key]["overlap_of_warned_onsets"][lk(k)]
        C.add(f"{key} {lk(k)} paired onsets both/only first/only second/neither", [ov["warned_by_both"], ov["warned_only_by_first"], ov["warned_only_by_second"], ov["warned_by_neither"]],
              [both, oa, oc, int((~fa & ~fc).sum())])
        C.add(f"{key} {lk(k)} paired onsets sign p", ov["sign_test_p"], sp)
        C.add(f"{key} {lk(k)} paired onsets = R.paired_onsets", R["paired_onsets"]["pairs"][key][lk(k)], ov)
        # gain_rule copy
        gr = R["gain_rule"]["sets"][T][f"against_{Cc}"][lk(k)]
        C.add(f"gain_rule {T} against {Cc} {lk(k)} readings", [gr["nse"]["reading"], gr["onset"]["reading"], gr["nse"]["n_basins_lower"]], [rd, fo["reading"], lower])
        MYPAIRS[key][lk(k)] = dict(diff_points=100 * (ET["onset"]["pod"] - EC["onset"]["pod"]), only_first=oa, only_second=oc, both=both, sign_p=sp, nse_higher=higher, nse_lower=lower, nse_p=pw,
                                   nse_reading=rd, onset_reading=fo["reading"], nse_with=ET["nse_median"], nse_without=EC["nse_median"])

# ---- 6. operating points (code/warning_policy.choose_trigger as documented: grid 0.50-2.00, csi / far_cap) ---------------------------------------------------------


def arrays(p, k, part):
    """Pooled over the basins: obs_hi, ratio = forecast / Q95, was_low on the target days of a part (lead k)."""
    j = LEADS.index(k)
    O, Rr, L = [], [], []
    for i in range(len(ids)):
        o, f = OBS[i], p[i, :, j]
        prev = np.r_[np.full(k, np.nan), o[:-k]]
        ok = PART[part] & np.isfinite(o) & np.isfinite(f) & np.isfinite(prev)
        O.append(o[ok] > THR[i]); Rr.append(f[ok] / THR[i]); L.append(prev[ok] <= THR[i])
    return np.concatenate(O), np.concatenate(Rr), np.concatenate(L)


def tab(o, w):
    return rates(dict(hits=int((o & w).sum()), misses=int((o & ~w).sum()), false_alarms=int((~o & w).sum()), correct_neg=int((~o & ~w).sum())))


def at(A, t):
    o, r, l = A
    w = r > t
    return dict(onset=tab(o[l], w[l]), all=tab(o, w))


nanv = lambda x, bad: bad if x != x else x


def choose(A, rule):
    S = {t: at(A, t) for t in GRID}
    if rule == "csi":
        best = GRID[0]
        for t in GRID:                                           # max of (csi, -|t-1|), first maximum in grid order
            if (nanv(S[t]["onset"]["csi"], -1.0), -abs(t - 1.0)) > (nanv(S[best]["onset"]["csi"], -1.0), -abs(best - 1.0)):
                best = t
        return best, True
    okt = [t for t in GRID if nanv(S[t]["onset"]["far"], 9) <= FAR_CAP]
    if okt:
        best = okt[0]
        for t in okt:
            if (nanv(S[t]["onset"]["pod"], -1.0), nanv(S[t]["onset"]["csi"], -1.0)) > (nanv(S[best]["onset"]["pod"], -1.0), nanv(S[best]["onset"]["csi"], -1.0)):
                best = t
        return best, True
    best = GRID[0]
    for t in GRID:
        if (nanv(S[t]["onset"]["far"], 9), -t) < (nanv(S[best]["onset"]["far"], 9), -best):
            best = t
    return best, False


verdict = lambda T: "NO EVENTS" if T["hits"] + T["misses"] == 0 else "GO" if T["pod"] >= POD_GO and T["far"] <= FAR_GO else "CAUTION" if T["csi"] >= CSI_CAUTION else "NOT YET"
OPS = {}
for n in SHOWN:
    OPS[n] = {}
    for k in LEADS:
        Av, At = arrays(PRED[name_of(n)], k, "val"), arrays(PRED[name_of(n)], k, "test")
        f = R["operating_points"]["sets"][n][lk(k)]
        C.add(f"op {n} {lk(k)} n_test_onsets", f["n_test_onsets"], int(At[0][At[2]].sum()))
        OPS[n][lk(k)] = {}
        for rule in ("csi", "far_cap"):
            t, reached = choose(Av, rule)
            v, s = at(Av, t), at(At, t)
            fr = f["rules"][rule]
            C.add(f"op {n} {lk(k)} {rule} trigger", fr["trigger"], t); C.add(f"op {n} {lk(k)} {rule} reached", fr["reached"], reached)
            for kind in ("onset", "all"):
                for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    C.add(f"op {n} {lk(k)} {rule} val {kind} {c}", fr["validation"][kind][c], v[kind][c])
                    C.add(f"op {n} {lk(k)} {rule} test {kind} {c}", fr["test"][kind][c], s[kind][c])
                C.add(f"op {n} {lk(k)} {rule} test verdict {kind}", fr["test_verdict"][kind], verdict(s[kind]))
            seeds_ = [at(arrays(PRED[f"lstm_{n}_s{x}"], k, "test"), t)["onset"] for x in SEEDS[n]] if n in LSTM else []
            for x, sv in zip(SEEDS.get(n, []), seeds_):
                i_ = fr["single_seeds_at_this_trigger"]["by_seed"]["seeds"].index(x)
                C.add(f"op {n} {lk(k)} {rule} seed {x} pod", fr["single_seeds_at_this_trigger"]["by_seed"]["onset_pod"][i_], sv["pod"])
                C.add(f"op {n} {lk(k)} {rule} seed {x} far", fr["single_seeds_at_this_trigger"]["by_seed"]["onset_far"][i_], sv["far"])
            C.add(f"op {n} {lk(k)} {rule} n single seeds", fr["single_seeds_at_this_trigger"]["n_seeds"], len(seeds_))
            OPS[n][lk(k)][rule] = dict(trigger=t, reached=reached, test_pod=s["onset"]["pod"], test_far=s["onset"]["far"], test_hits=s["onset"]["hits"], val_pod=v["onset"]["pod"], val_far=v["onset"]["far"],
                                       seeds_pod=[x["pod"] for x in seeds_], seeds_far=[x["far"] for x in seeds_])
# the product form of the warning (forecast > t x Q95) against the ratio form (forecast / Q95 > t) at the chosen triggers: do any warnings differ?
form_diff = 0
for n in SHOWN:
    for k in LEADS:
        for rule in ("csi", "far_cap"):
            t = OPS[n][lk(k)][rule]["trigger"]
            a1, a2 = score(PRED[name_of(n)], "test", t, "product")[lk(k)]["onset"], score(PRED[name_of(n)], "test", t, "ratio")[lk(k)]["onset"]
            form_diff += int(a1["hits"] != a2["hits"] or a1["false_alarms"] != a2["false_alarms"])

# ---- 7. verdict readings (level grid 0.30-1.50; tie to the level nearest 1.0) --------------------------------------------------------------------------------------
for n in SHOWN:
    VAL_BY_LEVEL = [score(PRED[name_of(n)], "val", m) for m in LEVELS]
    for k in LEADS:
        f = R["verdicts"]["sets"][n][lk(k)]
        E = EV[name_of(n)][lk(k)]
        for kind in ("all", "onset"):
            C.add(f"verdict {n} {lk(k)} plain {kind}", f["plain"][kind]["verdict"], verdict(E[kind]))
            C.add(f"verdict {n} {lk(k)} plain {kind} hits", f["plain"][kind]["hits"], E[kind]["hits"])
        for label, crit in (("chosen_on_validation_all_days", "all"), ("chosen_on_validation_onsets", "onset")):
            csis = np.array([nanv(VAL_BY_LEVEL[i_][lk(k)][crit]["csi"], -1.0) for i_ in range(len(LEVELS))])
            best = np.flatnonzero(csis >= csis.max() - 1e-12)
            trig = LEVELS[int(best[np.argmin(np.abs(np.array(LEVELS)[best] - 1.0))])]
            st = score(PRED[name_of(n)], "test", trig)[lk(k)]
            C.add(f"verdict {n} {lk(k)} {label} trigger", f[label]["trigger"], trig)
            for kind in ("all", "onset"):
                C.add(f"verdict {n} {lk(k)} {label} {kind}", [f[label][kind]["verdict"], f[label][kind]["hits"], f[label][kind]["false_alarms"]], [verdict(st[kind]), st[kind]["hits"], st[kind]["false_alarms"]])
        if n in LSTM:
            for kind in ("all", "onset"):
                C.add(f"verdict {n} {lk(k)} single seeds plain {kind}", f["single_seeds_plain"][kind], {str(s): verdict(EV[f"lstm_{n}_s{s}"][lk(k)][kind]) for s in SEEDS[n]})

# ---- 8. resampling (13_uncertainty.py's documented recipe, own code) ----------------------------------------------------------------------------------------------
RS = R["resampling"]
RS_SETS = ("wqME", "wqME_shuf", "wqME_older", "wqa", "wqnh", "wq", "base_linreg_wqME", "base_rf_wqME", "base_linreg_wqa", "base_rf_wqa")
RS_PAIRS = (("wqME", "wqa"), ("wqME", "wqME_shuf"), ("wqME", "wqME_older"), ("wqME_shuf", "wqa"), ("wqME_older", "wqa"), ("wqME", "wqnh"),
            ("base_linreg_wqME", "base_linreg_wqa"), ("base_rf_wqME", "base_rf_wqa"))
Tix = np.flatnonzero(TE)
wy = np.array([d_.year + (1 if d_.month >= 10 else 0) for d_ in dates[Tix]])       # water year (Oct-Sep) of each test day
years = sorted(set(wy))
C.add("test water years", RS["test_water_years"], years)
ny, nb = len(years), len(ids)
yidx = np.searchsorted(years, wy)
o_t = OBS[:, Tix]
hi_t = o_t > THR[:, None]
low_t = np.stack([OBS[:, Tix - k] <= THR[:, None] for k in LEADS], 2)
# per basin x water year sums with pandas groupby (another route than an einsum)
grp = lambda X: np.stack([pd.DataFrame(X[b]).groupby(yidx).sum().values for b in range(nb)])          # [basin, year, ...]
N_ = grp(np.ones((nb, len(Tix))))[:, :, 0]
SO, SOO = grp(o_t)[:, :, 0], grp(o_t ** 2)[:, :, 0]
ON = grp(np.stack([hi_t & low_t[:, :, j] for j in range(3)], 2).astype(float))
SSE, HIT, FAL = {}, {}, {}
for n in RS_SETS:
    P_ = PRED[name_of(n)][:, Tix]
    w_ = P_ > THR[:, None, None]
    SSE[n] = grp((P_ - o_t[:, :, None]) ** 2)
    HIT[n] = grp((hi_t[:, :, None] & w_ & low_t).astype(float))
    FAL[n] = grp((~hi_t[:, :, None] & w_ & low_t).astype(float))
rng = np.random.default_rng(SEED_RS)
BD, YD = np.empty((B_RS, nb), int), np.empty((B_RS, ny), int)
for r_ in range(B_RS):
    BD[r_], YD[r_] = rng.integers(0, nb, nb), rng.integers(0, ny, ny)


def stats_of(bd, yd):
    """All statistics of one resample: basins bd (with repeats) and years yd (with repeats)."""
    my = np.bincount(yd, minlength=ny).astype(float)
    wb = np.bincount(bd, minlength=nb).astype(float)
    nd, so, soo = N_ @ my, SO @ my, SOO @ my
    sst = soo - so ** 2 / nd
    out = {}
    for j, k in enumerate(LEADS):
        nse = {n: 1 - (SSE[n][:, :, j] @ my) / sst for n in RS_SETS}
        on = wb @ (ON[:, :, j] @ my)
        for n in RS_SETS:
            h, f = wb @ (HIT[n][:, :, j] @ my), wb @ (FAL[n][:, :, j] @ my)
            out[f"nse_median.{n}.{lk(k)}"] = float(np.median(nse[n][bd]))
            out[f"onsets_warned.{n}.{lk(k)}"] = h / on if on else float("nan")
            out[f"onset_false.{n}.{lk(k)}"] = f / (h + f) if h + f else float("nan")
        for A_, C_ in RS_PAIRS:
            out[f"onsets_diff.{A_}_minus_{C_}.{lk(k)}"] = out[f"onsets_warned.{A_}.{lk(k)}"] - out[f"onsets_warned.{C_}.{lk(k)}"]
            out[f"nse_diff_of_medians.{A_}_minus_{C_}.{lk(k)}"] = out[f"nse_median.{A_}.{lk(k)}"] - out[f"nse_median.{C_}.{lk(k)}"]
            out[f"nse_median_of_diffs.{A_}_minus_{C_}.{lk(k)}"] = float(np.median((nse[A_] - nse[C_])[bd]))
        for m in ("base_linreg", "base_rf"):
            out[f"onsets_gain_difference.lstm_minus_{m}.{lk(k)}"] = out[f"onsets_diff.wqME_minus_wqa.{lk(k)}"] - out[f"onsets_diff.{m}_wqME_minus_{m}_wqa.{lk(k)}"]
    return out


point = stats_of(np.arange(nb), np.arange(ny))
C.add("resampling: statistic names", sorted(RS["statistics"]), sorted(point))
designs = {"years": [], "basins": [], "both": []}
with np.errstate(all="ignore"):
    for r_ in range(B_RS):
        designs["years"].append(stats_of(np.arange(nb), YD[r_]))
        designs["basins"].append(stats_of(BD[r_], np.arange(ny)))
        designs["both"].append(stats_of(BD[r_], YD[r_]))


def interval(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), share_above_zero=float((v > 0).mean())) if v.size else dict(lo=None, hi=None, share_above_zero=None)


MYRS = {}
for key in point:
    MYRS[key] = dict(value=point[key], **{d_: interval([x[key] for x in designs[d_]]) for d_ in designs})
    f = RS["statistics"][key]
    C.add(f"resampling {key} value", f["value"], point[key])
    for d_ in designs:
        for c in ("lo", "hi", "share_above_zero"):
            C.add(f"resampling {key} {d_} {c}", f[d_][c], MYRS[key][d_][c])
# direct route on the first resamples: concatenate the drawn days and basins and score them from scratch
direct_worst = 0.0
for r_ in range(5):
    bd, yd = BD[r_], YD[r_]
    days_sel = np.concatenate([np.flatnonzero(yidx == y) for y in yd])
    for j, k in enumerate(LEADS):
        for n in ("wqME", "wqa"):
            P_ = PRED[name_of(n)][:, Tix][:, days_sel, j]
            o_ = o_t[:, days_sel]
            nse_b = [1 - ((P_[b] - o_[b]) ** 2).sum() / ((o_[b] - o_[b].mean()) ** 2).sum() for b in bd]
            hh = sum(int((hi_t[b, days_sel] & low_t[b, days_sel, j] & (P_[b] > THR[b])).sum()) for b in bd)
            oo = sum(int((hi_t[b, days_sel] & low_t[b, days_sel, j]).sum()) for b in bd)
            direct_worst = max(direct_worst, abs(float(np.median(nse_b)) - designs["both"][r_][f"nse_median.{n}.{lk(k)}"]), abs(hh / oo - designs["both"][r_][f"onsets_warned.{n}.{lk(k)}"]))
C.add("resampling: direct route on the first 5 resamples agrees (1e-9)", True, direct_worst < 1e-9)
# Monte-Carlo noise of the interval ends of the headline difference (16 parts, 13_uncertainty.py mc_noise as documented there), not done by the script
v = np.array([x["onsets_diff.wqME_minus_wqa.l1"] for x in designs["both"]])
ends = np.array([np.percentile(p_, [2.5, 97.5]) for p_ in np.array_split(v, 16)])
mc_noise_points = (100 * ends.std(0, ddof=1) / np.sqrt(16)).tolist()

# ---- 9. the stored sets against their own results files -----------------------------------------------------------------------------------------------------------
AR, RM = jload("results/aorc_rain_to_midnight.json"), jload("results/rain_to_midnight.json")
for n, J in (("wqa", AR), ("wq", AR), ("wqn", AR), ("wqh", RM), ("wqnh", RM)):
    s = J["summary_test"]["sets"][n]
    for k in LEADS:
        m = EV[f"lstm_{n}"][lk(k)]
        C.add(f"stored {n} {lk(k)} nse median vs own file", s[lk(k)]["nse_median"], m["nse_median"])
        C.add(f"stored {n} {lk(k)} onset hits vs own file", s[lk(k)]["onset"]["hits"], m["onset"]["hits"])
        C.add(f"stored {n} {lk(k)} onset pod vs own file", s[lk(k)]["onset"]["pod"], m["onset"]["pod"])
        C.add(f"stored {n} {lk(k)} onset far vs own file", s[lk(k)]["onset"]["far"], m["onset"]["far"])
for n in ("base_linreg_wqa", "base_rf_wqa"):
    s = AR["yardstick"]["sets"][n]["test"]["summary"]
    for k in LEADS:
        C.add(f"stored {n} {lk(k)} onset hits / nse vs own file", [s[lk(k)]["onset"]["hits"], round(s[lk(k)]["nse_median"], 12)], [EV[n][lk(k)]["onset"]["hits"], round(EV[n][lk(k)]["nse_median"], 12)])

# ---- 10. descriptive extras for the findings (test years seen; not used for any choice) -------------------------------------------------------------------------------
X = {}
# (a) the controls at the 30%-false operating point and at the CSI point
X["at_30pct_false_l1"] = {n: OPS[n]["l1"]["far_cap"] for n in ("wqME", "wqa", "wqME_shuf", "wqME_older", "wqnh")}
X["at_csi_point_l1"] = {n: OPS[n]["l1"]["csi"] for n in ("wqME", "wqa", "wqME_shuf", "wqME_older", "wqnh")}
# (b) wqME's own median single seed against the wqa range (does the 'flat' test separate the treatment from the controls?)
X["median_single_seed_l1"] = {n: float(np.median(r1[n])) for n in r1}
X["wqa_range_l1"] = [min(r1["wqa"]), max(r1["wqa"])]
X["wqME_median_single_seed_inside_wqa_range"] = bool(min(r1["wqa"]) <= X["median_single_seed_l1"]["wqME"] <= max(r1["wqa"]))
# (c) level-matched: the trigger at which wqME gives as many onset warnings as wqa gives at 1.0 (test years, descriptive)
Aw, Am = arrays(PRED["lstm_wqa"], 1, "test"), arrays(PRED["lstm_wqME"], 1, "test")
n_warn_wqa = int((Aw[1][Aw[2]] > 1.0).sum())
rat_on = np.sort(Am[1][Am[2]])[::-1]
t_match = float(rat_on[n_warn_wqa - 1])                       # the n-th largest ratio: warning above anything just below it gives n warnings
sM = at(Am, t_match - 1e-12)
X["level_matched_l1"] = dict(wqa_onset_warnings_at_1=n_warn_wqa, wqa_hits=EV["lstm_wqa"]["l1"]["onset"]["hits"], wqME_trigger_for_the_same_number=t_match,
                             wqME_onset_warnings=sM["onset"]["hits"] + sM["onset"]["false_alarms"], wqME_hits=sM["onset"]["hits"], wqME_pod=sM["onset"]["pod"], wqME_far=sM["onset"]["far"],
                             wqa_pod=EV["lstm_wqa"]["l1"]["onset"]["pod"], wqa_far=EV["lstm_wqa"]["l1"]["onset"]["far"])
# (d) the forecast level: total bias and the median forecast/Q95 on onset days and on not-high days with a low issue day (test, lead 1)
lev = {}
for n in ("wqME", "wqa", "wqME_shuf", "wqME_older", "wqnh"):
    o, r, l = arrays(PRED[f"lstm_{n}"], 1, "test")
    P_ = PRED[f"lstm_{n}"][:, TE, 0]
    lev[n] = dict(pbias_pooled_pct=float(100 * (np.nansum(P_) - np.nansum(OBS[:, TE])) / np.nansum(OBS[:, TE])),
                  median_ratio_on_onset_days=float(np.median(r[o & l])), median_ratio_on_low_issue_not_high_days=float(np.median(r[~o & l])),
                  onset_warnings_at_1=int((r[l] > 1).sum()))
X["forecast_level_l1"] = lev
X["nse_l1_basins"] = {"wqME_lower_than_wqa": int((EV["lstm_wqME"]["l1"]["nse"] < EV["lstm_wqa"]["l1"]["nse"]).sum()), "p": MYPAIRS["several_products_against_aorc_alone"]["l1"]["nse_p"]}
X["nse_by_lead"] = {n: [EV[f"lstm_{n}"][lk(k)]["nse_median"] for k in LEADS] for n in ("wqME", "wqa", "wqME_shuf", "wqME_older", "wqnh")}
X["seed_by_seed_wqME_minus_wqa_points"] = [100 * (a - b) for a, b in zip(r1["wqME"], r1["wqa"])]
X["all_pairs_wqME_minus_wqa_points"] = [100 * (a - b) for a in r1["wqME"] for b in r1["wqa"]]
X["mc_noise_of_the_interval_ends_points"] = mc_noise_points
X["warning_form_differences_at_chosen_triggers"] = form_diff
X["wilcoxon"] = dict(p_scipy_auto=p_auto, p_exact_enumeration=p_enum, p_approx=p_appr, t_plus=tplus, n_used=int(len(nz)), n_zero=int((d == 0).sum()))
# per-basin hit-count differences (the other reading of 'per-basin onsets warned')
hits_b = lambda name: np.array([int((PRED[name][i, TE, 0] > THR[i])[(OBS[i, TE] > THR[i]) & (np.r_[np.nan, OBS[i, :-1]][TE] <= THR[i])].sum()) for i in range(len(ids))])
dh = hits_b("lstm_wqME") - hits_b("lstm_wqa")
X["wilcoxon"]["on_hit_counts"] = dict(differences=dh.tolist(), n_used=int((dh != 0).sum()), p=float(wilcoxon(dh).pvalue) if (dh != 0).any() else None)
X["wqME_minus_controls_l1"] = {k_: MYPAIRS[k_]["l1"] for k_ in ("several_products_against_the_capacity_control", "several_products_against_the_clock_control", "several_products_against_nldas_and_era5")}
X["intervals_l1"] = {k_: MYRS[k_] for k_ in ("onsets_diff.wqME_minus_wqa.l1", "onsets_diff.wqME_minus_wqME_shuf.l1", "onsets_diff.wqME_minus_wqME_older.l1", "nse_diff_of_medians.wqME_minus_wqa.l1",
                                              "nse_median_of_diffs.wqME_minus_wqa.l1", "onsets_diff.wqME_shuf_minus_wqa.l1", "onsets_diff.wqME_older_minus_wqa.l1")}

MY = dict(headline={n: {lk(k): dict(nse_median=EV[name_of(n)][lk(k)]["nse_median"], onset={c: EV[name_of(n)][lk(k)]["onset"][c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")},
                                     all={c: EV[name_of(n)][lk(k)]["all"][c] for c in ("hits", "misses", "false_alarms", "pod", "far", "csi")}) for k in LEADS} for n in SHOWN},
          ranges={n: RANGE[n] for n in LSTM}, rule=dict(block_1=dict(every_seed_above=above, false_wqME=fME, false_wqa=fA, wilcoxon_p=p_auto, wilcoxon_p_exact=p_enum, n_used=int(len(nz)),
                                                                  higher=int((d > 0).sum()), lower=int((d < 0).sum()), median_diff_points=float(100 * np.median(d)), holds=b1),
                                                        blocks_2_3=blocks, met=met, yardstick_gains=gains, yardstick_reading=y_read),
          pairs=MYPAIRS, ops=OPS, resampling=MYRS, extras=X, seed_mean=seed_mean, thresholds=THR.tolist())
jsave("m2_headline.json", dict(compare=C.summary(), rows=C.rows, my=MY, seconds=round(time.time() - t0, 1)))
print(json.dumps(C.summary())[:2000])
print("block1", MY["rule"]["block_1"], "met", met)
print(json.dumps(X, default=str)[:4000])
print("seconds", round(time.time() - t0, 1))
