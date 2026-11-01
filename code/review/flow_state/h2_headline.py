"""E4e review, check 2: the headline numbers recomputed from results/preds with the reviewer's own scoring (numpy only: no flood.evaluate, no warning_policy), for the six new
sets and the three stored sets, seed means and every single seed: pooled onset and all-day warning tables at trigger 1.0, median NSE, leads 1-3, test and validation years, per basin;
the seed ranges and the seed rule; the readings of every pair (NSE rule with Wilcoxon p, onset rule, paired onsets with the sign test); the closure of the gap to wqaP_s; the
CSI and 30%-false triggers chosen on the validation years (code/warning_policy.choose_trigger replicated here) and the test scores at them, the single seeds at those triggers,
the GO / CAUTION / NOT YET verdicts; the 24:00 against 22:00 delivery table; the E4e success record (also with the first three seeds); the stored sets against their own results
file (results/forecast_input.json); Turkey Creek with and without its fallback days; the validation years; a GO scan. Every number compared with results/flow_state.json.
Needs OUT/my_columns.npz (h3). About a minute."""
from h_common import *
from scipy.stats import binomtest, wilcoxon

t0 = time.time()
R = jload(RESULTS)
S = R["summary_test"]["sets"]
C = Compare(tol=1e-9)
MY = np.load(OUT / "my_columns.npz", allow_pickle=False)
ids = [str(b) for b in MY["basins"]]
assert ids == R["basins"]
CRIT = jload("results/flood_analysis.json")["verdict"]["criteria"]
C.add("criteria GO text", CRIT["GO"], "pod >= 0.70 and far <= 0.30"); C.add("criteria CAUTION text", CRIT["CAUTION"], "csi >= 0.40 but not GO")
C.add("criteria copied into operating_points", R["operating_points"]["criteria"], CRIT)

# ---- the data: dates, observed flow (identical across sets), thresholds from the training years of the split --------------------------------------------------
dates, basins, OBS, _, _ = load_npz("lstm_wqa_s")
assert basins == ids
TR, VA, TE = (within(dates, *SAT_SPLIT[p]) for p in ("train", "val", "test"))
THR = np.array([float(np.quantile(OBS[i][TR], 0.95)) for i in range(len(ids))])
for i, g in enumerate(ids):
    C.add(f"thresholds {g}", R["thresholds"][g], THR[i])
PRED, OBS_OF, META = {}, {}, {}
for n in SHOWN:
    for name in [f"lstm_{n}"] + [f"lstm_{n}_s{s}" for s in SEEDS[n]]:
        d, b, o, p, m = load_npz(name)
        assert b == ids and d.equals(dates), name
        PRED[name], OBS_OF[name], META[name] = p, o, m
C.add("observed flow identical in every set and seed file", True, all(np.array_equal(OBS_OF[k], OBS) for k in OBS_OF))
C.add("stored sets are script 18's satellite-era sets", True, all(META[f"lstm_{n}"]["script"] == "18_forecast_input.py" and META[f"lstm_{n}"]["arm"] == "sat" for n in STORED))
n_days = {"val": int(VA.sum()), "test": int(TE.sum())}


def tables(p, part, trigger=1.0):
    """Per lead: per-basin NSE and the pooled onset / all-day tables at one trigger on the target days of one split part (own code)."""
    part_mask = {"train": TR, "val": VA, "test": TE}[part]
    res = {}
    for j, k in enumerate(LEADS):
        H, Ho = dict(hits=0, misses=0, false_alarms=0, correct_neg=0), dict(hits=0, misses=0, false_alarms=0, correct_neg=0)
        nse, pod_b, far_b, cover = [], [], [], []
        for i in range(len(ids)):
            o, f = OBS[i], p[i, :, j]
            issue = np.concatenate([np.full(k, np.nan), o[:-k]])                                   # the observed flow k days before the target day
            ok = part_mask & np.isfinite(o) & np.isfinite(f)
            cover.append(int(ok.sum()) / n_days[part] if part in n_days else np.nan)
            oo, ff, ii = o[ok], f[ok], issue[ok]
            nse.append(1.0 - np.sum((ff - oo) ** 2) / np.sum((oo - oo.mean()) ** 2))
            hi, warn, low = oo > THR[i], ff > trigger * THR[i], ii <= THR[i]                      # NaN issue flow: not an onset (as flood.flood_scores)
            for T, m in ((H, np.ones(len(oo), bool)), (Ho, low)):
                T["hits"] += int((hi & warn & m).sum()); T["misses"] += int((hi & ~warn & m).sum()); T["false_alarms"] += int((~hi & warn & m).sum()); T["correct_neg"] += int((~hi & ~warn & m).sum())
            h_, m_, f_ = int((hi & warn & low).sum()), int((hi & ~warn & low).sum()), int((~hi & warn & low).sum())
            pod_b.append(h_ / (h_ + m_) if h_ + m_ else np.nan); far_b.append(f_ / (h_ + f_) if h_ + f_ else np.nan)
        rates = lambda T: dict(T, pod=T["hits"] / (T["hits"] + T["misses"]) if T["hits"] + T["misses"] else np.nan, far=T["false_alarms"] / (T["hits"] + T["false_alarms"]) if T["hits"] + T["false_alarms"] else np.nan,
                               csi=T["hits"] / (T["hits"] + T["misses"] + T["false_alarms"]) if T["hits"] + T["misses"] + T["false_alarms"] else np.nan)
        res[f"l{k}"] = dict(nse=np.array(nse), nse_median=float(np.median(nse)), nse_mean=float(np.mean(nse)), onset=rates(Ho), all=rates(H), onset_pod_by_basin=pod_b, onset_far_by_basin=far_b, coverage_min=float(min(cover)))
    return res


EV = {name: tables(PRED[name], "test") for name in PRED}
EV_VAL = {name: tables(PRED[name], "val") for name in PRED}
C.add("every set has a forecast on every test and validation day (coverage 1)", True, all(EV[f"lstm_{n}"][lk(k)]["coverage_min"] == 1.0 and EV_VAL[f"lstm_{n}"][lk(k)]["coverage_min"] == 1.0 for n in SHOWN for k in LEADS))
# 1. summary_test and summary_val (seed means), per basin
for n in SHOWN:
    for k in LEADS:
        for part, E, F_ in (("test", EV, S), ("val", EV_VAL, R["summary_val"]["sets"])):
            m, f = E[f"lstm_{n}"][lk(k)], F_[n][lk(k)]
            C.add(f"{n} l{k} {part} nse_median", f["nse_median"], m["nse_median"]); C.add(f"{n} l{k} {part} nse_mean", f["nse_mean"], m["nse_mean"])
            for kind in ("onset", "all"):
                for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    C.add(f"{n} l{k} {part} {kind} {c}", f[kind][c], m[kind][c])
        m = EV[f"lstm_{n}"][lk(k)]
        for i, g in enumerate(ids):
            C.add(f"{n} l{k} per_basin nse {g}", R["per_basin"]["nse"][n][lk(k)][g], m["nse"][i])
            C.add(f"{n} l{k} per_basin onset_warned {g}", R["per_basin"]["onset_warned"][n][lk(k)][g], m["onset_pod_by_basin"][i])
            C.add(f"{n} l{k} per_basin onset_false {g}", R["per_basin"]["onset_false"][n][lk(k)][g], m["onset_far_by_basin"][i])
# 2. seed ranges and the seed means against the single seeds
seed_mean_check = {}
for n in SHOWN:
    rg = R["seed_ranges"]["sets"][n]
    for k in LEADS:
        vals = {c: [] for c in ("nse_median", "onset_pod", "onset_far", "onset_csi", "all_pod", "all_far", "all_csi")}
        for s in SEEDS[n]:
            e = EV[f"lstm_{n}_s{s}"][lk(k)]
            vals["nse_median"].append(e["nse_median"])
            for kind in ("onset", "all"):
                for c in ("pod", "far", "csi"):
                    vals[f"{kind}_{c}"].append(e[kind][c])
        C.add(f"{n} l{k} n_seeds", rg[lk(k)]["n_seeds"], len(SEEDS[n])); C.add(f"{n} l{k} by_seed seeds", rg[lk(k)]["by_seed"]["seeds"], SEEDS[n])
        for c, v in vals.items():
            C.add(f"{n} l{k} seed_range {c} lo", rg[lk(k)][c][0], min(v)); C.add(f"{n} l{k} seed_range {c} hi", rg[lk(k)][c][1], max(v))
            for j, s in enumerate(SEEDS[n]):
                C.add(f"{n} l{k} by_seed {c} s{s}", rg[lk(k)]["by_seed"][c][j], v[j])
    mean = np.mean([PRED[f"lstm_{n}_s{s}"] for s in SEEDS[n]], axis=0)
    a = PRED[f"lstm_{n}"]
    fin = np.isfinite(a)
    seed_mean_check[n] = dict(same_nan_pattern=bool(np.array_equal(np.isnan(a), np.isnan(mean))), max_abs_diff=float(np.abs(a[fin] - mean[fin]).max()), n_values=int(fin.sum()))
C.add("seed means equal the day-by-day mean of the seed files (max abs diff over 9 sets)", 0.0, max(v["max_abs_diff"] for v in seed_mean_check.values()), tol=1e-12)
# the per-seed records of models.*.per_seed (test scores per seed)
for n in NEW:
    for r in R["models"][n]["per_seed"]:
        e = EV[f"lstm_{n}_s{r['seed']}"]
        for j, k in enumerate(LEADS):
            C.add(f"models {n} s{r['seed']} l{k} test_nse_median", r["test_nse_median"][j], e[lk(k)]["nse_median"]); C.add(f"models {n} s{r['seed']} l{k} test_onset_warned", r["test_onset_warned"][j], e[lk(k)]["onset"]["pod"])
            C.add(f"models {n} s{r['seed']} l{k} test_onset_false", r["test_onset_false"][j], e[lk(k)]["onset"]["far"])
# 3. the readings: the seed rule and the gain rule of every pair, the paired onsets
above = lambda a, b: min(a) > max(b)
pods = lambda n, k: [EV[f"lstm_{n}_s{s}"][lk(k)]["onset"]["pod"] for s in SEEDS[n]]
nses = lambda n, k: [EV[f"lstm_{n}_s{s}"][lk(k)]["nse_median"] for s in SEEDS[n]]
PAIRS = [(n, BASE_OF[n]) for n in NEW] + [("wqaH_s", "wqaH22_s"), ("wqaWH_s", "wqaWH22_s"), ("wqaWHF_s", "wqaWHF22_s"), ("wqaWH_s", "wqaH_s"), ("wqaWHF_s", "wqaWH_s"), ("wqaH_s", "wqaP_s"), ("wqaWHF_s", "wqaP_s"), ("wqaF_s", "wqa_s")]
C.add("readings pairs", list(R["readings"]["pairs"].keys()), [f"{T}_vs_{Cc}" for T, Cc in PAIRS])


def flags(p, k, trigger=1.0, mask=None):
    """One True / False per onset of the test years (warned or not), basin after basin."""
    out, j = [], k - 1
    for i in range(len(ids)):
        o, f = OBS[i], p[i, :, j]
        issue = np.concatenate([np.full(k, np.nan), o[:-k]])
        ok = TE & np.isfinite(o) & np.isfinite(f)
        m = ok & (o > THR[i]) & (issue <= THR[i])
        if mask is not None:
            m &= mask[i]
        out.append((f > trigger * THR[i])[m])
    return np.concatenate(out)


rule_mine, zip_lengths = {}, {}
for T, Cc in PAIRS:
    key = f"{T}_vs_{Cc}"
    for k in LEADS:
        rd = R["readings"]["pairs"][key]["by_lead"][lk(k)]
        a, b = EV[f"lstm_{T}"][lk(k)]["nse"], EV[f"lstm_{Cc}"][lk(k)]["nse"]
        d = a - b
        try:
            p = float(wilcoxon(a, b).pvalue)
        except ValueError:
            p = float("nan")
        up, down = above(nses(T, k), nses(Cc, k)), above(nses(Cc, k), nses(T, k))
        n_hi, n_lo = int((d > 0).sum()), int((d < 0).sum())
        gain, loss = up and n_hi > 6 and p < 0.05, down and n_lo > 6 and p < 0.05
        o_up, o_down = above(pods(T, k), pods(Cc, k)), above(pods(Cc, k), pods(T, k))
        mine = dict(nse_reading="clear gain" if gain else "clear loss" if loss else "mixed" if (up or down) else "no clear difference", onset_reading="clear gain" if o_up else "clear loss" if o_down else "no clear difference")
        rule_mine[f"{key} l{k}"] = dict(mine, n_basins_higher=n_hi, median_basin_difference=float(np.median(d)), wilcoxon_p=p, every_seed_above_nse=up, every_seed_above_onsets=o_up, every_seed_below_onsets=o_down,
                                         onset_gain_points=100 * (EV[f"lstm_{T}"][lk(k)]["onset"]["pod"] - EV[f"lstm_{Cc}"][lk(k)]["onset"]["pod"]))
        C.add(f"readings {key} l{k} against", rd["against"], f"lstm_{Cc}")
        for fld, val in (("median_with", EV[f"lstm_{T}"][lk(k)]["nse_median"]), ("median_without", EV[f"lstm_{Cc}"][lk(k)]["nse_median"]), ("gain", EV[f"lstm_{T}"][lk(k)]["nse_median"] - EV[f"lstm_{Cc}"][lk(k)]["nse_median"]),
                         ("single_seeds_all_above", up), ("single_seeds_all_below", down), ("n_basins_higher", n_hi), ("n_basins_lower", n_lo), ("n_basins", 12), ("median_basin_difference", float(np.median(d))), ("wilcoxon_p", p),
                         ("clear_gain", gain), ("clear_loss", loss), ("reading", mine["nse_reading"])):
            C.add(f"readings {key} l{k} nse {fld}", rd["nse"][fld], val)
        C.add(f"readings {key} l{k} nse seed_range_with", rd["nse"]["seed_range_with"], [min(nses(T, k)), max(nses(T, k))]); C.add(f"readings {key} l{k} nse seed_range_without", rd["nse"]["seed_range_without"], [min(nses(Cc, k)), max(nses(Cc, k))])
        oT, oC = EV[f"lstm_{T}"][lk(k)]["onset"], EV[f"lstm_{Cc}"][lk(k)]["onset"]
        for fld, val in (("warned_with", oT["pod"]), ("warned_without", oC["pod"]), ("gain_points", 100 * (oT["pod"] - oC["pod"])), ("hits_with", oT["hits"]), ("hits_without", oC["hits"]), ("false_with", oT["far"]), ("false_without", oC["far"]),
                         ("single_seeds_all_above", o_up), ("single_seeds_all_below", o_down), ("reading", mine["onset_reading"])):
            C.add(f"readings {key} l{k} onset {fld}", rd["onset"][fld], val)
        C.add(f"readings {key} l{k} onset seed_range_with", rd["onset"]["seed_range_with"], [min(pods(T, k)), max(pods(T, k))]); C.add(f"readings {key} l{k} onset seed_range_without", rd["onset"]["seed_range_without"], [min(pods(Cc, k)), max(pods(Cc, k))])
        zipped = [100 * (x - y) for x, y in zip(pods(T, k), pods(Cc, k))]
        C.add(f"readings {key} l{k} onset difference_by_seed_points (zip over the two seed lists)", rd["onset"]["difference_by_seed_points"], zipped, tol=1e-9) if len(rd["onset"]["difference_by_seed_points"]) == len(zipped) and all(abs(x - y) < 1e-9 for x, y in zip(rd["onset"]["difference_by_seed_points"], zipped)) else C.add(f"readings {key} l{k} onset difference_by_seed_points", rd["onset"]["difference_by_seed_points"], zipped)
        zip_lengths[f"{key} l{k}"] = dict(entries=len(rd["onset"]["difference_by_seed_points"]), seeds_first=len(SEEDS[T]), seeds_second=len(SEEDS[Cc]))
        # paired onsets
        fa, fc = flags(PRED[f"lstm_{T}"], k), flags(PRED[f"lstm_{Cc}"], k)
        both, oa, oc, nei = int((fa & fc).sum()), int((fa & ~fc).sum()), int((~fa & fc).sum()), int((~fa & ~fc).sum())
        pp = R["readings"]["pairs"][key]["paired_onsets"][lk(k)]
        for name, v in (("n_onsets", len(fa)), ("hits_first", int(fa.sum())), ("hits_second", int(fc.sum())), ("warned_by_both", both), ("warned_only_by_first", oa), ("warned_only_by_second", oc), ("warned_by_neither", nei),
                        ("sign_test_p", float(binomtest(oa, oa + oc, 0.5).pvalue) if oa + oc else float("nan"))):
            C.add(f"paired {key} l{k} {name}", pp[name], v)
truncated = {k: v for k, v in zip_lengths.items() if v["entries"] < max(v["seeds_first"], v["seeds_second"])}
# 4. closure of the gap to wqaP_s
def closure(name, base, ceil):
    rows = []
    for k in LEADS:
        Eb, Ec, En = EV[f"lstm_{base}"][lk(k)], EV[f"lstm_{ceil}"][lk(k)], EV[f"lstm_{name}"][lk(k)]
        b, c, n = Eb["nse_median"], Ec["nse_median"], En["nse_median"]
        ob, oc, on = Eb["onset"]["pod"], Ec["onset"]["pod"], En["onset"]["pod"]
        rows.append(dict(lead=k, nse_base=b, nse_with=n, nse_ceiling=c, share_of_nse_gap_closed=(n - b) / (c - b) if c - b else np.nan, basins_better_than_base=int((En["nse"] > Eb["nse"]).sum()), onset_warned_base=ob, onset_warned_with=on,
                         onset_warned_ceiling=oc, onset_gain_points=100 * (on - ob), onset_share_of_gap_closed=(on - ob) / (oc - ob) if oc - ob else np.nan, onset_false_base=Eb["onset"]["far"], onset_false_with=En["onset"]["far"], onset_false_ceiling=Ec["onset"]["far"]))
    return rows
for key, (name, base) in {**{n: (n, BASE_OF[n]) for n in NEW}, "wqaF_s": ("wqaF_s", "wqa_s")}.items():
    rows = closure(name, base, "wqaP_s")
    C.add(f"closure {key} base", R["closure"]["sets"][key]["base"], f"lstm_{base}"); C.add(f"closure {key} ceiling", R["closure"]["sets"][key]["ceiling"], "lstm_wqaP_s")
    for j, k in enumerate(LEADS):
        fr = R["closure"]["sets"][key]["by_lead"][j]
        for c in rows[j]:
            C.add(f"closure {key} l{k} {c}", fr[c], rows[j][c])
# 5. the validation years choose the triggers (code/warning_policy.choose_trigger replicated); the test years are scored once at them
verdict = lambda t: "NO EVENTS" if t["hits"] + t["misses"] == 0 else "GO" if (t["pod"] >= POD_GO and t["far"] <= FAR_GO) else "CAUTION" if t["csi"] >= CSI_CAUTION else "NOT YET"
_f = lambda x, bad=-1.0: bad if x is None or (isinstance(x, float) and np.isnan(x)) else x


def arrays(p, part, k):
    """Pooled over the basins on the target days of a part: obs_hi, ratio = forecast / Q95, was_low (as code/warning_policy.arrays, own code)."""
    part_mask = {"val": VA, "test": TE}[part]
    O, Rt, L = [], [], []
    for i in range(len(ids)):
        o, f = OBS[i], p[i, :, k - 1]
        issue = np.concatenate([np.full(k, np.nan), o[:-k]])
        ok = part_mask & np.isfinite(o) & np.isfinite(f) & np.isfinite(issue)
        O.append((o > THR[i])[ok]); Rt.append((f / THR[i])[ok]); L.append((issue <= THR[i])[ok])
    return np.concatenate(O), np.concatenate(Rt), np.concatenate(L)


def cont(hi, warn):
    h, m, f = int((hi & warn).sum()), int((hi & ~warn).sum()), int((~hi & warn).sum())
    return dict(hits=h, misses=m, false_alarms=f, pod=h / (h + m) if h + m else float("nan"), far=f / (h + f) if h + f else float("nan"), csi=h / (h + m + f) if h + m + f else float("nan"))


def score_at(A, t):
    hi, r, low = A
    return dict(all=cont(hi, r > t), onset=cont(hi[low], (r > t)[low]))


def choose(A, rule):
    Sg = {t: score_at(A, t) for t in GRID}
    if rule == "plain":
        return 1.0, True, Sg
    if rule == "csi":
        return max(GRID, key=lambda t: (_f(Sg[t]["onset"]["csi"]), -abs(t - 1.0))), True, Sg
    ok = [t for t in GRID if _f(Sg[t]["onset"]["far"], 9) <= FAR_CAP]
    if ok:
        return max(ok, key=lambda t: (_f(Sg[t]["onset"]["pod"]), _f(Sg[t]["onset"]["csi"]))), True, Sg
    return min(GRID, key=lambda t: (_f(Sg[t]["onset"]["far"], 9), -t)), False, Sg


ops_mine, go_scan = {}, []
for n in SHOWN:
    OP = R["operating_points"]["sets"][n]
    for k in LEADS:
        A_val, A_test = arrays(PRED[f"lstm_{n}"], "val", k), arrays(PRED[f"lstm_{n}"], "test", k)
        C.add(f"operating_points {n} l{k} n_test_onsets", OP[lk(k)]["n_test_onsets"], int(A_test[0][A_test[2]].sum()))
        for rule in ("plain", "csi", "far_cap"):
            t, rch, Sg = choose(A_val, rule)
            fr = OP[lk(k)]["rules"][rule]
            st, sv = score_at(A_test, t), Sg[t]
            C.add(f"operating_points {n} l{k} {rule} trigger", fr["trigger"], t); C.add(f"operating_points {n} l{k} {rule} reached", fr["reached"], rch)
            for kind in ("onset", "all"):
                for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                    C.add(f"operating_points {n} l{k} {rule} validation {kind} {c}", fr["validation"][kind][c], sv[kind][c]); C.add(f"operating_points {n} l{k} {rule} test {kind} {c}", fr["test"][kind][c], st[kind][c])
                C.add(f"operating_points {n} l{k} {rule} test_verdict {kind}", fr["test_verdict"][kind], verdict(st[kind]))
            seeds = [score_at(arrays(PRED[f"lstm_{n}_s{s}"], "test", k), t)["onset"] for s in SEEDS[n]]
            ss = fr["single_seeds_at_this_trigger"]
            C.add(f"operating_points {n} l{k} {rule} single_seeds n", ss["n_seeds"], len(seeds))
            fars = [s["far"] for s in seeds if s["far"] == s["far"]]                                    # a seed that issues no onset warning has no false share (the file's lo_hi drops it too)
            C.add(f"operating_points {n} l{k} {rule} single_seeds pod lo", ss["onset_pod"][0], min(s["pod"] for s in seeds)); C.add(f"operating_points {n} l{k} {rule} single_seeds pod hi", ss["onset_pod"][1], max(s["pod"] for s in seeds))
            C.add(f"operating_points {n} l{k} {rule} single_seeds far lo", (ss["onset_far"] or [None])[0], min(fars) if fars else None); C.add(f"operating_points {n} l{k} {rule} single_seeds far hi", (ss["onset_far"] or [None, None])[1], max(fars) if fars else None)
            if len(fars) < len(seeds):
                C.rows[f"operating_points {n} l{k} {rule} seeds without an onset warning"] = dict(file=None, mine=len(seeds) - len(fars), ok=True)
            for j, s in enumerate(SEEDS[n]):
                C.add(f"operating_points {n} l{k} {rule} by_seed pod s{s}", ss["by_seed"]["onset_pod"][j], seeds[j]["pod"]); C.add(f"operating_points {n} l{k} {rule} by_seed far s{s}", ss["by_seed"]["onset_far"][j], seeds[j]["far"])
            ops_mine[f"{n} l{k} {rule}"] = dict(trigger=t, reached=rch, test_onset=st["onset"], test_all=st["all"], validation_onset=sv["onset"], test_verdict=dict(onset=verdict(st["onset"]), all=verdict(st["all"])),
                                                single_seeds_onset_pod=[s["pod"] for s in seeds], single_seeds_onset_far=[s["far"] for s in seeds])
            if verdict(st["onset"]) == "GO":
                go_scan.append(dict(set=n, lead=k, rule=rule, trigger=t, reading="onsets", pod=st["onset"]["pod"], far=st["onset"]["far"]))
            if verdict(st["all"]) == "GO":
                go_scan.append(dict(set=n, lead=k, rule=rule, trigger=t, reading="all high-flow days", pod=st["all"]["pod"], far=st["all"]["far"]))
            for j, s in enumerate(SEEDS[n]):
                if seeds[j]["pod"] >= POD_GO and seeds[j]["far"] <= FAR_GO:
                    go_scan.append(dict(set=n, lead=k, rule=rule, trigger=t, reading="onsets, single seed", seed=s, pod=seeds[j]["pod"], far=seeds[j]["far"]))
# 6. the delivery table (24:00 against 22:00)
for fam in ("H", "WH", "WHF"):
    a, b = f"wqa{fam}_s", f"wqa{fam}22_s"
    for k in LEADS:
        fr = R["delivery"]["families"][fam][lk(k)]
        rm = rule_mine[f"{a}_vs_{b} l{k}"]
        C.add(f"delivery {fam} l{k} last_hour_24", fr["last_hour_24"], a); C.add(f"delivery {fam} l{k} last_hour_22", fr["last_hour_22"], b)
        C.add(f"delivery {fam} l{k} onset_warned_24", fr["onset_warned_24"], EV[f"lstm_{a}"][lk(k)]["onset"]["pod"]); C.add(f"delivery {fam} l{k} onset_warned_22", fr["onset_warned_22"], EV[f"lstm_{b}"][lk(k)]["onset"]["pod"])
        C.add(f"delivery {fam} l{k} difference_points_24_minus_22", fr["difference_points_24_minus_22"], rm["onset_gain_points"])
        C.add(f"delivery {fam} l{k} seed_range_24", fr["seed_range_24"], [min(pods(a, k)), max(pods(a, k))]); C.add(f"delivery {fam} l{k} seed_range_22", fr["seed_range_22"], [min(pods(b, k)), max(pods(b, k))])
        C.add(f"delivery {fam} l{k} every_seed_24_above_22", fr["every_seed_24_above_22"], rm["every_seed_above_onsets"]); C.add(f"delivery {fam} l{k} every_seed_22_above_24", fr["every_seed_22_above_24"], rm["every_seed_below_onsets"])
        C.add(f"delivery {fam} l{k} onset_false_24", fr["onset_false_24"], EV[f"lstm_{a}"][lk(k)]["onset"]["far"]); C.add(f"delivery {fam} l{k} onset_false_22", fr["onset_false_22"], EV[f"lstm_{b}"][lk(k)]["onset"]["far"])
        C.add(f"delivery {fam} l{k} nse_median_24", fr["nse_median_24"], EV[f"lstm_{a}"][lk(k)]["nse_median"]); C.add(f"delivery {fam} l{k} nse_median_22", fr["nse_median_22"], EV[f"lstm_{b}"][lk(k)]["nse_median"])
        C.add(f"delivery {fam} l{k} nse_reading", fr["nse_reading"], rm["nse_reading"])
        for side, n_ in (("24", a), ("22", b)):
            pt = ops_mine[f"{n_} l{k} far_cap"]
            C.add(f"delivery {fam} l{k} far_cap_point_{side} trigger", fr[f"far_cap_point_{side}"]["trigger"], pt["trigger"])
            for c in ("hits", "misses", "false_alarms", "pod", "far", "csi"):
                C.add(f"delivery {fam} l{k} far_cap_point_{side} {c}", fr[f"far_cap_point_{side}"][c], pt["test_onset"][c])
# 7. the E4e success record, also with the first three seeds
E = R["e4e_success"]
r1 = {x: pods(x, 1) for x in SHOWN}
mine_success = dict(met=above(r1["wqaH_s"], r1["wqa_s"]), every_seed_of_wqaH22_s_above_every_seed_of_wqa_s=above(r1["wqaH22_s"], r1["wqa_s"]), every_seed_of_wqaWH_s_above_every_seed_of_wqa_s=above(r1["wqaWH_s"], r1["wqa_s"]),
                    every_seed_of_wqaWHF_s_above_every_seed_of_wqaF_s=above(r1["wqaWHF_s"], r1["wqaF_s"]))
for k_, v in mine_success.items():
    C.add(f"e4e_success {k_}", E[k_], v)
C.add("e4e_success lead", E["lead"], 1); C.add("e4e_success rule names the pooled lead-1 onsets at trigger 1.0, every seed", True, "every single seed of wqaH_s above every single seed of the stored lstm_wqa_s" in E["rule"] and "trigger 1.0" in E["rule"])
for x in SHOWN:
    C.add(f"e4e_success seed_ranges {x}", E["seed_ranges"][x], [min(r1[x]), max(r1[x])]); C.add(f"e4e_success n_seeds {x}", E["n_seeds"][x], len(SEEDS[x]))
    C.add(f"e4e_success onset_warned {x}", E["onset_warned"][x], EV[f"lstm_{x}"]["l1"]["onset"]["pod"]); C.add(f"e4e_success onset_false {x}", E["onset_false"][x], EV[f"lstm_{x}"]["l1"]["onset"]["far"])
three = dict(wqaH_s_seeds_0_2=[round(v, 4) for v in r1["wqaH_s"][:3]], wqa_s_seeds=[round(v, 4) for v in r1["wqa_s"]], met_with_three_seeds=above(r1["wqaH_s"][:3], r1["wqa_s"]),
             seeds_of_wqaH_s_at_or_below_the_best_seed_of_wqa_s=[s for s, v in zip(SEEDS["wqaH_s"], r1["wqaH_s"]) if v <= max(r1["wqa_s"])], best_seed_of_wqa_s=float(max(r1["wqa_s"])), worst_seed_of_wqaH_s=float(min(r1["wqaH_s"])),
             wqaH_s_seed_mean_above_every_seed_of_wqa_s=EV["lstm_wqaH_s"]["l1"]["onset"]["pod"] > max(r1["wqa_s"]), wqa_s_seed_spread_points=100 * (max(r1["wqa_s"]) - min(r1["wqa_s"])), wqaH_s_seed_spread_points=100 * (max(r1["wqaH_s"]) - min(r1["wqaH_s"])))
# 8. the stored sets' own results file (script 18, E2b) against my numbers
S18 = jload("results/forecast_input.json")["summary_test"]["sets"]
for n in STORED:
    for k in LEADS:
        m = EV[f"lstm_{n}"][lk(k)]
        C.add(f"stored {n} forecast_input.json l{k} nse_median", S18[n][lk(k)]["nse_median"], m["nse_median"]); C.add(f"stored {n} forecast_input.json l{k} onset pod", S18[n][lk(k)]["onset"]["pod"], m["onset"]["pod"])
        C.add(f"stored {n} forecast_input.json l{k} onset far", S18[n][lk(k)]["onset"]["far"], m["onset"]["far"]); C.add(f"stored {n} forecast_input.json l{k} onset hits", S18[n][lk(k)]["onset"]["hits"], m["onset"]["hits"])
        C.add(f"stored {n} forecast_input.json l{k} all csi", S18[n][lk(k)]["all"]["csi"], m["all"]["csi"])
    C.add(f"checks_against_stored_results {n} all_equal", R["checks_against_stored_results"][n]["all_equal"], True)
# 9. Turkey Creek (02349900) with and without its fallback days; all basins: onsets whose issue day is a fallback day
known24 = MY["hourly_known"].astype(bool)                                                    # [basin, day of T_ISS] the issue day's hourly record is used (24:00 variant)
off = int(np.flatnonzero(dates == T_ISS[0])[0])
full_known = np.zeros((len(ids), len(dates)), bool)
full_known[:, off:] = known24
issue_known = {k: np.concatenate([np.zeros((len(ids), k), bool), full_known[:, :-k]], axis=1) for k in LEADS}        # the issue day (target - k) was a known day
tc = ids.index("02349900")
tcq = {}
for n in ("wqa_s", "wqaH_s", "wqaH22_s", "wqaWH_s", "wqaWHF_s", "wqaF_s"):
    p = PRED[f"lstm_{n}"]
    rec = {}
    for k in LEADS:
        o, f = OBS[tc], p[tc, :, k - 1]
        issue = np.concatenate([np.full(k, np.nan), o[:-k]])
        ok = TE & np.isfinite(o) & np.isfinite(f)
        ons = ok & (o > THR[tc]) & (issue <= THR[tc])
        kn = issue_known[k][tc]
        res = {}
        for label, m in (("all", ok), ("issue_day_known", ok & kn), ("issue_day_fallback", ok & ~kn)):
            oo, ff = o[m], f[m]
            on_m = ons & m
            h, mi, fa = int(((f > THR[tc]) & on_m).sum()), int((~(f > THR[tc]) & on_m).sum()), int(((f > THR[tc]) & ok & m & (o <= THR[tc]) & (issue <= THR[tc])).sum())
            res[label] = dict(days=int(m.sum()), nse=float(1 - np.sum((ff - oo) ** 2) / np.sum((oo - oo.mean()) ** 2)) if m.sum() > 1 else None, onsets=int(on_m.sum()), onset_hits=h, onset_pod=h / (h + mi) if h + mi else None, onset_false_alarms=fa,
                              onset_far=fa / (h + fa) if h + fa else None)
        rec[lk(k)] = res
    tcq[n] = rec
pooled_fb = {}
for n in ("wqa_s", "wqaH_s", "wqaH22_s", "wqaWH_s"):
    for k in (1,):
        fa_all = flags(PRED[f"lstm_{n}"], k)
        kn = flags(np.where(issue_known[k][:, :, None], 1e9, -1e9) * np.ones((1, 1, 3)), k)      # True where the issue day is known (trick: a 'forecast' above the threshold exactly on known issue days)
        pooled_fb[n] = dict(onsets=int(len(fa_all)), onsets_with_known_issue_day=int(kn.sum()), onsets_with_fallback_issue_day=int((~kn).sum()), pod_known=float(fa_all[kn].mean()), pod_fallback=float(fa_all[~kn].mean()) if (~kn).any() else None,
                            hits_known=int(fa_all[kn].sum()), hits_fallback=int(fa_all[~kn].sum()))
turkey = dict(basin="02349900 Turkey Creek, GA", fallback_days_test_years=int((~full_known[tc] & TE).sum()), test_days=int(TE.sum()), fallback_share_test=float((~full_known[tc] & TE).mean() / TE.mean()),
              fallback_runs_test_years=[(str(dates[a].date()), str(dates[b - 1].date()), int(b - a)) for a, b in zip(*[np.flatnonzero(np.diff(np.concatenate([[0], (~full_known[tc] & TE).astype(int), [0]])) == d_) for d_ in (1, -1)]) if b - a >= 7],
              per_set=tcq, pooled_lead1_onsets_by_issue_day_status=pooled_fb)
print("Turkey Creek test years:", turkey["fallback_days_test_years"], "fallback days of", turkey["test_days"], "| runs >= 7 days:", turkey["fallback_runs_test_years"])
print("   lead 1 onsets: " + "; ".join(f"{n}: known {tcq[n]['l1']['issue_day_known']['onsets']} onsets pod {tcq[n]['l1']['issue_day_known']['onset_pod']}, fallback {tcq[n]['l1']['issue_day_fallback']['onsets']} onsets pod {tcq[n]['l1']['issue_day_fallback']['onset_pod']}" for n in ("wqa_s", "wqaH_s")))
print("   pooled lead-1 onsets by issue-day status:", pooled_fb)
# 10. the validation years as a reading (seed means at trigger 1.0, best validation NSE of the stopping points)
val_reading = {n: dict(onset_pod_l1=EV_VAL[f"lstm_{n}"]["l1"]["onset"]["pod"], onset_far_l1=EV_VAL[f"lstm_{n}"]["l1"]["onset"]["far"], nse_median_l1=EV_VAL[f"lstm_{n}"]["l1"]["nse_median"],
                       single_seeds_onset_pod_l1=[EV_VAL[f"lstm_{n}_s{s}"]["l1"]["onset"]["pod"] for s in SEEDS[n]]) for n in SHOWN}
for n in NEW:
    val_reading[n]["best_val_nse_of_the_stopping_points"] = [r["best_val_nse"] for r in R["models"][n]["per_seed"]]
summ = C.summary()
headline = {n: {lk(k): dict(nse_median=EV[f"lstm_{n}"][lk(k)]["nse_median"], onset=EV[f"lstm_{n}"][lk(k)]["onset"], all=EV[f"lstm_{n}"][lk(k)]["all"], seed_onset_pod=pods(n, k), seed_onset_far=[EV[f"lstm_{n}_s{s}"][lk(k)]["onset"]["far"] for s in SEEDS[n]],
                            seed_nse_median=nses(n, k)) for k in LEADS} for n in SHOWN}
paired_mine = {f"{T}_vs_{Cc}": {lk(k): dict(zip(("warned_by_both", "warned_only_by_first", "warned_only_by_second", "warned_by_neither"), [int(x) for x in ((flags(PRED[f'lstm_{T}'], k) & flags(PRED[f'lstm_{Cc}'], k)).sum(), (flags(PRED[f'lstm_{T}'], k) & ~flags(PRED[f'lstm_{Cc}'], k)).sum(),
                                                                                                                                              (~flags(PRED[f'lstm_{T}'], k) & flags(PRED[f'lstm_{Cc}'], k)).sum(), (~flags(PRED[f'lstm_{T}'], k) & ~flags(PRED[f'lstm_{Cc}'], k)).sum())])) for k in LEADS} for T, Cc in PAIRS[:6]}
jsave("h2_headline.json", dict(summary=summ, rows=C.rows, headline_mine=headline, seed_mean_check=seed_mean_check, success_mine=mine_success, success_with_three_seeds=three, rule_mine=rule_mine, operating_points_mine=ops_mine,
                               paired_mine=paired_mine, go_scan=go_scan, difference_by_seed_points_truncated=truncated, turkey_creek=turkey, validation_reading=val_reading, thresholds=dict(zip(ids, THR.tolist())),
                               n_test_onsets={lk(k): int(len(flags(PRED["lstm_wqa_s"], k))) for k in LEADS}, seconds=round(time.time() - t0, 1)))
print(f"compared {summ['entries']} values: {summ['matching']} match, {len(summ['not_matching'])} differ" + (f": {summ['not_matching'][:20]}" if summ["not_matching"] else ""))
print("headline (test, trigger 1.0, lead 1): " + "; ".join(f"{n} {100 * headline[n]['l1']['onset']['pod']:.1f}% warned, {100 * headline[n]['l1']['onset']['far']:.1f}% false, NSE {headline[n]['l1']['nse_median']:.3f}" for n in SHOWN))
print("success rule (mine):", mine_success, "| with three seeds:", three)
print("GO scan (test years):", go_scan)
print("difference_by_seed_points lists shorter than the longer seed list:", len(truncated), "of", len(zip_lengths))
print("seconds:", round(time.time() - t0, 1))
